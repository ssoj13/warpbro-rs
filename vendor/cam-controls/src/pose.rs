//! Roll-capable camera pose ([`CameraPose`]) shared by the fly / spaceflight
//! backends in [`crate::controller`].
//!
//! # Why a second pose type next to [`crate::rig::OrbitPose`]
//!
//! [`OrbitPose`](crate::rig::OrbitPose) is a *constrained* orbit pose: it stores
//! `(orientation, distance, target)` and always frames the world with
//! [`WORLD_UP`](crate::convention::WORLD_UP) as the screen-up reference — it has
//! no roll degree of freedom. The FPS / spaceflight rigs need a free 6-DoF pose
//! where roll is a first-class axis, so they store an explicit `eye` position
//! plus a full camera `orientation` quaternion.
//!
//! # Convention (kept identical to [`OrbitPose`] for equivalent state)
//!
//! - Right-handed, Y-up world (same as the rest of `cam-controls`).
//! - The camera's **local** frame follows the glam camera convention: the lens
//!   looks down local **-Z**, local **+X** is screen-right, local **+Y** is
//!   screen-up. `forward = orientation * -Z`, `right = orientation * +X`,
//!   `up = orientation * +Y`.
//! - `view_matrix()` uses [`glam::Mat4::look_to_rh`], which is exactly what
//!   [`OrbitPose::view_matrix`](crate::rig::OrbitPose::view_matrix)'s
//!   `look_at_rh(eye, target, WORLD_UP)` reduces to when
//!   `forward = (target - eye).normalize()` and the up vector is the orbit
//!   screen-up. [`CameraPose::from_orbit`] builds the orientation from exactly
//!   that basis, so the resulting `view_proj` is bit-for-bit identical (within
//!   normalization epsilon) to the orbit rig's for the same on-screen state.
//! - Projection reuses [`PerspectiveSettings`] via the same
//!   `perspective_rh(fov_y, aspect, near, far)` call OrbitPose uses; `CameraPose`
//!   carries its own `fov_y / znear / zfar` so a backend can vary them, but
//!   [`CameraPose::projection_matrix`] takes a [`PerspectiveSettings`] to mirror
//!   the orbit signature and uses *its own* fov/near/far overrides applied onto
//!   that settings path (see the method docs).

use cam_viewport::{ViewportPoint, ViewportRay, ViewportSize};
use glam::{Mat4, Quat, Vec3};

use crate::convention::WORLD_UP;
use crate::projection::PerspectiveSettings;
use crate::rig::OrbitPose;

/// Roll-capable, free 6-DoF camera pose (eye + full orientation quaternion).
///
/// **What:** stores an explicit world-space `eye`, a camera `orientation`
/// (local -Z = forward, +Y = up, +X = right), and per-pose projection
/// (`fov_y`, `znear`, `zfar`). `Copy` so it can travel the GPU upload path the
/// same way [`OrbitPose`] does.
///
/// **Why:** the FPS and spaceflight backends are not orbit-constrained — they
/// fly free and can roll, which [`OrbitPose`] cannot represent. This is the
/// common pose all four [`CameraController`](crate::controller::CameraController)
/// backends emit from `pose()` so render code has one uniform type.
///
/// **Where:** produced by `CameraController::pose`, converted from/to
/// [`OrbitPose`] via [`CameraPose::from_orbit`] / [`OrbitPose`] continuity, and
/// consumed by render passes through [`CameraPose::view_proj`] /
/// [`CameraPose::inv_view_proj_cols`] / [`CameraPose::viewport_ray`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraPose {
    /// World-space camera position.
    pub eye: Vec3,
    /// Camera orientation: local -Z looks forward, +Y is up, +X is right.
    pub orientation: Quat,
    /// Vertical field of view in radians.
    pub fov_y: f32,
    /// Near clip plane (world units).
    pub znear: f32,
    /// Far clip plane (world units).
    pub zfar: f32,
}

impl Default for CameraPose {
    /// Default pose: derived from [`OrbitPose::default`] so the very first frame
    /// of a fresh fly/space rig frames the world like the default orbit camera.
    fn default() -> Self {
        Self::from_orbit(&OrbitPose::default(), PerspectiveSettings::default())
    }
}

impl CameraPose {
    /// Build a pose from an explicit eye, forward direction and up reference.
    ///
    /// **Why:** central constructor used by [`CameraPose::from_orbit`] and the
    /// fly backends. It orthonormalizes `forward`/`up` into a camera basis and
    /// stores the equivalent quaternion (local -Z → `forward`, +Y → `up`).
    /// Falls back to `WORLD_UP` / `+X` when the inputs are degenerate so it
    /// never produces a NaN quaternion at the poles.
    pub fn from_eye_forward_up(
        eye: Vec3,
        forward: Vec3,
        up: Vec3,
        fov_y: f32,
        znear: f32,
        zfar: f32,
    ) -> Self {
        let f = forward.normalize_or_zero();
        let f = if f.length_squared() < 1e-12 { -Vec3::Z } else { f };
        // Right = forward × up_ref; guard against forward ∥ up.
        let mut right = f.cross(up);
        if right.length_squared() < 1e-10 {
            right = f.cross(WORLD_UP);
        }
        if right.length_squared() < 1e-10 {
            right = f.cross(Vec3::X);
        }
        let right = right.normalize();
        let true_up = right.cross(f).normalize();
        // Columns of the camera→world rotation: X=right, Y=up, Z=-forward
        // (camera looks down local -Z, so world `forward` is local `-Z`).
        let rot = Mat4::from_cols(
            right.extend(0.0),
            true_up.extend(0.0),
            (-f).extend(0.0),
            Vec3::ZERO.extend(1.0),
        );
        let orientation = Quat::from_mat4(&rot).normalize();
        Self {
            eye,
            orientation,
            fov_y,
            znear,
            zfar,
        }
    }

    /// Convert an [`OrbitPose`] into the equivalent free pose.
    ///
    /// **What:** reproduces the orbit camera's exact on-screen framing —
    /// `eye = orbit.eye()`, forward toward `orbit.target`, screen-up taken from
    /// [`OrbitPose::view_plane_axes`] so the resulting `view_matrix` equals the
    /// orbit's `look_at_rh(eye, target, WORLD_UP)`.
    ///
    /// **Why:** lets the Houdini backend (which wraps `OrbitRig`) and any
    /// backend switch keep view continuity, and is the correctness anchor for
    /// the unit tests (`HoudiniOrbit.pose().view_proj == OrbitRig.view_proj`).
    pub fn from_orbit(orbit: &OrbitPose, settings: PerspectiveSettings) -> Self {
        let eye = orbit.eye();
        let forward = (orbit.target - eye).normalize_or_zero();
        // `view_plane_axes` returns the same up the orbit `look_at_rh` implies.
        let (_right, up) = orbit.view_plane_axes();
        Self::from_eye_forward_up(
            eye,
            forward,
            up,
            settings.fov_y,
            settings.near,
            settings.far,
        )
    }

    /// Camera forward (look) direction in world space (`orientation * -Z`).
    pub fn forward(&self) -> Vec3 {
        (self.orientation * -Vec3::Z).normalize_or_zero()
    }

    /// Camera right direction in world space (`orientation * +X`).
    pub fn right(&self) -> Vec3 {
        (self.orientation * Vec3::X).normalize_or_zero()
    }

    /// Camera up direction in world space (`orientation * +Y`).
    pub fn up(&self) -> Vec3 {
        (self.orientation * Vec3::Y).normalize_or_zero()
    }

    /// World position of the eye as a `[f32; 3]` (GPU upload convenience,
    /// mirrors [`OrbitPose::eye_pos`](crate::rig::OrbitPose::eye_pos)).
    pub fn eye_pos(&self) -> [f32; 3] {
        self.eye.to_array()
    }

    /// Screen-plane basis `(right, up)` in world space.
    ///
    /// **Why:** matches [`OrbitPose::view_plane_axes`] so the screen-corner XYZ
    /// gizmo and any pan math that consumes the orbit basis work unchanged when
    /// fed a free pose. For a zero-roll pose this is `(right(), up())`.
    pub fn view_plane_axes(&self) -> (Vec3, Vec3) {
        (self.right(), self.up())
    }

    /// Right-handed view matrix.
    ///
    /// Uses glam's `rh::view::look_to_mat4`, which is identical to the orbit
    /// rig's `look_at_mat4(eye, target, up)` for the equivalent basis (verified
    /// by the `houdini_pose_matches_orbit` test).
    pub fn view_matrix(&self) -> Mat4 {
        glam::camera::rh::view::look_to_mat4(self.eye, self.forward(), self.up())
    }

    /// Right-handed perspective projection.
    ///
    /// **Why the `settings` argument when the pose carries its own fov/near/far:**
    /// this mirrors [`OrbitPose::projection_matrix`]'s signature so call-sites
    /// are interchangeable, but the *pose* is authoritative for `fov_y / znear /
    /// zfar` (a fly rig may zoom its lens independently of the shared orbit
    /// settings). `settings` is accepted for API parity and currently only its
    /// presence is required; the pose fields win. The projection call itself is
    /// the same projection path OrbitPose uses.
    pub fn projection_matrix(&self, aspect: f32, _settings: PerspectiveSettings) -> Mat4 {
        // Reverse-Z: near/far swapped so near→NDC depth 1, far→0 (see OrbitPose).
        // `directx` is glam 0.33's name for the [0,1] depth-clip convention —
        // wgpu's, and what the old `Mat4::perspective_rh` produced.
        glam::camera::rh::proj::directx::perspective(self.fov_y, aspect, self.zfar, self.znear)
    }

    /// `projection * view` for a viewport of `width × height`.
    pub fn view_proj(&self, width: u32, height: u32, settings: PerspectiveSettings) -> Mat4 {
        let aspect = width as f32 / height.max(1) as f32;
        self.projection_matrix(aspect, settings) * self.view_matrix()
    }

    /// Column-major inverse of [`CameraPose::view_proj`] for GPU unproject.
    pub fn inv_view_proj_cols(
        &self,
        width: u32,
        height: u32,
        settings: PerspectiveSettings,
    ) -> [[f32; 4]; 4] {
        self.view_proj(width, height, settings)
            .inverse()
            .to_cols_array_2d()
    }

    /// World-space pick ray through a viewport pixel.
    ///
    /// **Why `from_inv_view_proj` (which uses `project_point3`):** `inv_view_proj`
    /// is non-affine; `transform_point3` would collapse every pixel onto the
    /// camera forward axis. This mirrors
    /// [`OrbitPose::viewport_ray`](crate::rig::OrbitPose::viewport_ray) exactly —
    /// the documented bug to avoid.
    pub fn viewport_ray(
        &self,
        point: ViewportPoint,
        size: ViewportSize,
        settings: PerspectiveSettings,
    ) -> ViewportRay {
        ViewportRay::from_inv_view_proj(
            point,
            size,
            self.inv_view_proj_cols(size.width, size.height, settings),
            self.eye_pos(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_orbit_matches_orbit_view_proj() {
        let settings = PerspectiveSettings::default();
        let orbit = OrbitPose::from_yaw_pitch(0.7, 0.3, 22.0, Vec3::new(1.0, 2.0, -1.0));
        let pose = CameraPose::from_orbit(&orbit, settings);
        let a = orbit.view_proj(800, 600, settings);
        let b = pose.view_proj(800, 600, settings);
        let mut max = 0.0_f32;
        for (x, y) in a.to_cols_array().iter().zip(b.to_cols_array().iter()) {
            max = max.max((x - y).abs());
        }
        assert!(max < 1e-3, "view_proj mismatch, max delta {max}");
    }

    #[test]
    fn basis_is_orthonormal() {
        let pose = CameraPose::default();
        let f = pose.forward();
        let r = pose.right();
        let u = pose.up();
        assert!((f.length() - 1.0).abs() < 1e-4);
        assert!(r.dot(u).abs() < 1e-4);
        assert!(f.dot(r).abs() < 1e-4);
    }
}
