//! Orbit rig: [`OrbitPose`] (Copy, GPU path) + [`OrbitRig`] (inertia / transitions).

use cam_viewport::{NdcPoint, ViewportPoint, ViewportRay, ViewportSize};
use glam::{Mat4, Quat, Vec3};

use crate::convention::{DEFAULT_OFFSET_AXIS, WORLD_UP};
use crate::dynamics::{InertiaState, TransitionState};
use crate::fit::{aabb_center_radius, distance_to_frame_sphere};
use crate::orientation::{
    apply_orbit_delta, orientation_from_yaw_pitch, OrientationMode, DEFAULT_PITCH_LIMIT,
};
use crate::projection::PerspectiveSettings;
use crate::settings::CameraNavigationSettings;

/// Minimal orbit pose (Copy) — used by render passes and legacy `OrbitCamera`.
#[derive(Debug, Clone, Copy)]
pub struct OrbitPose {
    pub orientation: Quat,
    pub distance: f32,
    pub target: Vec3,
}

impl Default for OrbitPose {
    fn default() -> Self {
        Self::from_yaw_pitch(0.6, 0.35, 28.0, Vec3::ZERO)
    }
}

impl OrbitPose {
    pub fn from_yaw_pitch(yaw: f32, pitch: f32, distance: f32, target: Vec3) -> Self {
        Self {
            orientation: orientation_from_yaw_pitch(yaw, pitch),
            distance,
            target,
        }
    }

    pub fn offset(&self) -> Vec3 {
        self.orientation * (DEFAULT_OFFSET_AXIS * self.distance)
    }

    pub fn eye(&self) -> Vec3 {
        self.target + self.offset()
    }

    pub fn view_matrix(&self) -> Mat4 {
        glam::camera::rh::view::look_at_mat4(self.eye(), self.target, WORLD_UP)
    }

    /// Reverse-Z perspective: `near`/`far` are passed SWAPPED so the near plane
    /// maps to NDC depth 1 and far to 0. Paired with a `Greater` depth test +
    /// clear-to-0 in the render passes, this concentrates float-depth precision
    /// far from the camera — essential for the big-bang galaxy's huge depth span
    /// (kills distant z-fighting on ribbons/nodes).
    ///
    /// `directx` is glam 0.33's name for the [0,1] depth-clip convention (wgpu's),
    /// i.e. exactly what the deprecated `Mat4::perspective_rh` produced.
    pub fn projection_matrix(&self, aspect: f32, settings: PerspectiveSettings) -> Mat4 {
        glam::camera::rh::proj::directx::perspective(
            settings.fov_y,
            aspect,
            settings.far,
            settings.near,
        )
    }

    pub fn view_proj(&self, width: u32, height: u32, settings: PerspectiveSettings) -> Mat4 {
        let aspect = width as f32 / height.max(1) as f32;
        self.projection_matrix(aspect, settings) * self.view_matrix()
    }

    pub fn orbit(
        &mut self,
        yaw_delta: f32,
        pitch_delta: f32,
        mode: OrientationMode,
        pitch_limit: f32,
    ) {
        apply_orbit_delta(
            mode,
            &mut self.orientation,
            yaw_delta,
            pitch_delta,
            pitch_limit,
        );
    }

    /// World-space basis vectors for the camera's screen plane.
    ///
    /// **What:** returns `(right, up)` — the world directions of the
    /// screen-right and screen-up axes derived from `eye → target` and
    /// `WORLD_UP`. `right` and `up` are unit vectors; `right × up` aligns with
    /// the forward vector (away from the eye).
    ///
    /// **Why:** used internally for pan/zoom and externally by
    /// `gitnexus-tree-runtime::worker::publish_gpu_surface` to ship the basis
    /// to the GUI as part of `ViewportGpuFrameReady`, where the screen-corner
    /// XYZ gizmo projects each world axis to 2D via `dot(axis, right)` /
    /// `dot(axis, up)`.
    ///
    /// **Where:** internal `pan`, `zoom_at_cursor`, public consumers via
    /// [`OrbitRig::pose`].
    pub fn view_plane_axes(&self) -> (Vec3, Vec3) {
        let eye = self.eye();
        let forward = (self.target - eye).normalize();
        let mut right = forward.cross(WORLD_UP);
        if right.length_squared() < 1e-8 {
            right = forward.cross(Vec3::X);
        }
        let right = right.normalize();
        let up = right.cross(forward).normalize();
        (right, up)
    }

    pub fn pan(&mut self, dx_px: f32, dy_px: f32, pan_sensitivity: f32) {
        let (right, up) = self.view_plane_axes();
        let scale = self.distance * pan_sensitivity;
        self.target += right * (-dx_px * scale) + up * (dy_px * scale);
    }

    pub fn zoom(&mut self, factor: f32, settings: PerspectiveSettings) {
        self.distance = settings.clamp_distance(self.distance * factor);
    }

    pub fn zoom_at_cursor(
        &mut self,
        point: ViewportPoint,
        size: ViewportSize,
        factor: f32,
        settings: PerspectiveSettings,
    ) {
        let ndc = point.to_ndc(size);
        let (right, up) = self.view_plane_axes();
        let half_tan = (settings.fov_y * 0.5).tan();
        let half_w = self.distance * half_tan * size.aspect();
        let half_h = self.distance * half_tan;
        let focus = self.target + right * (ndc.x * half_w) + up * (ndc.y * half_h);
        self.zoom(factor, settings);
        let (right2, up2) = self.view_plane_axes();
        let half_w2 = self.distance * half_tan * size.aspect();
        let half_h2 = self.distance * half_tan;
        self.target = focus - (right2 * (ndc.x * half_w2) + up2 * (ndc.y * half_h2));
    }

    pub fn frame_bounds(
        &mut self,
        min: Vec3,
        max: Vec3,
        viewport_width: u32,
        viewport_height: u32,
        margin: f32,
        settings: PerspectiveSettings,
    ) {
        let (center, radius) = aabb_center_radius(min, max);
        let aspect = viewport_width as f32 / viewport_height.max(1) as f32;
        // Robust against a runaway / non-finite scene AABB (e.g. the big-bang sim
        // — or the “Big Bang” material preset — flinging a node to a huge or NaN/Inf
        // position): a non-finite centre would aim the camera at infinity and a
        // huge radius would dolly it out past the usable range (“flies into
        // space”). Sanitise the centre, and CLAMP the framed distance to the same
        // `[distance_min, distance_max]` band manual zoom uses — so Fit-All can
        // never send the camera further than you could zoom by hand.
        self.target = if center.is_finite() { center } else { Vec3::ZERO };
        let d = distance_to_frame_sphere(radius, aspect, settings.fov_y, margin, settings);
        self.distance = if d.is_finite() {
            settings.clamp_distance(d)
        } else {
            settings.distance_max
        };
    }

    pub fn viewport_ndc(&self, point: ViewportPoint, size: ViewportSize) -> [f32; 2] {
        NdcPoint::from_viewport(point, size).as_array()
    }

    pub fn screen_ndc(&self, px: f32, py: f32, width: u32, height: u32) -> [f32; 2] {
        let size = ViewportSize::new(width, height);
        let point = ViewportPoint::new_clamped(px, py, size);
        NdcPoint::from_viewport(point, size).as_array()
    }

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

    pub fn eye_pos(&self) -> [f32; 3] {
        self.eye().to_array()
    }

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

    pub fn screen_ray(
        &self,
        px: f32,
        py: f32,
        width: u32,
        height: u32,
        settings: PerspectiveSettings,
    ) -> ([f32; 3], [f32; 3]) {
        let size = ViewportSize::new(width, height);
        let ray = self.viewport_ray(ViewportPoint::new_clamped(px, py, size), size, settings);
        (ray.origin, ray.dir)
    }
}

/// Full orbit rig: pose + projection + orientation policy + inertia + transitions.
#[derive(Debug, Clone)]
pub struct OrbitRig {
    pub pose: OrbitPose,
    pub projection: PerspectiveSettings,
    pub navigation: CameraNavigationSettings,
    pub orientation_mode: OrientationMode,
    pub pitch_limit: f32,
    pub inertia: InertiaState,
    pub transition: TransitionState,
    pub inertia_friction: f32,
    pub inertia_cutoff: f32,
    pub transition_speed: f32,
}

impl Default for OrbitRig {
    fn default() -> Self {
        Self::from_yaw_pitch(0.6, 0.35, 28.0, Vec3::ZERO)
    }
}

impl OrbitRig {
    pub fn from_yaw_pitch(yaw: f32, pitch: f32, distance: f32, target: Vec3) -> Self {
        let projection = PerspectiveSettings::default();
        let navigation = CameraNavigationSettings::default();
        Self {
            pose: OrbitPose {
                orientation: orientation_from_yaw_pitch(yaw, pitch),
                distance: projection.clamp_distance(distance),
                target,
            },
            projection,
            navigation,
            orientation_mode: OrientationMode::HorizontalPlaneEuler,
            pitch_limit: DEFAULT_PITCH_LIMIT,
            inertia: InertiaState::default(),
            transition: TransitionState::default(),
            inertia_friction: navigation.inertia_friction,
            inertia_cutoff: 0.0001,
            transition_speed: 8.0,
        }
    }

    pub fn from_pose(pose: OrbitPose) -> Self {
        let mut rig = Self::default();
        rig.pose = pose;
        rig
    }

    // ─── Pose accessors (render passes use [`OrbitPose`] via `render_scene::OrbitCamera`) ──

    pub fn orientation(&self) -> Quat {
        self.pose.orientation
    }

    pub fn orientation_mut(&mut self) -> &mut Quat {
        &mut self.pose.orientation
    }

    pub fn distance(&self) -> f32 {
        self.pose.distance
    }

    pub fn distance_mut(&mut self) -> &mut f32 {
        &mut self.pose.distance
    }

    pub fn target(&self) -> Vec3 {
        self.pose.target
    }

    pub fn target_mut(&mut self) -> &mut Vec3 {
        &mut self.pose.target
    }

    pub fn offset(&self) -> Vec3 {
        self.pose.offset()
    }

    pub fn eye(&self) -> Vec3 {
        self.pose.eye()
    }

    pub fn view_proj(&self, width: u32, height: u32) -> Mat4 {
        self.pose.view_proj(width, height, self.projection)
    }

    pub fn orbit(&mut self, yaw_delta: f32, pitch_delta: f32) {
        self.inertia.stop();
        self.pose
            .orbit(yaw_delta, pitch_delta, self.orientation_mode, self.pitch_limit);
    }

    pub fn pan(&mut self, dx_px: f32, dy_px: f32) {
        self.inertia.stop();
        self.pose
            .pan(dx_px, dy_px, self.navigation.pan_sensitivity);
    }

    pub fn zoom(&mut self, factor: f32) {
        self.inertia.stop();
        self.pose.zoom(factor, self.projection);
    }

    pub fn zoom_at_cursor(&mut self, point: ViewportPoint, size: ViewportSize, factor: f32) {
        self.inertia.stop();
        self.pose
            .zoom_at_cursor(point, size, factor, self.projection);
    }

    pub fn apply_navigation(&mut self, navigation: CameraNavigationSettings) {
        self.navigation = navigation;
        self.inertia_friction = navigation.inertia_friction;
    }

    pub fn frame_bounds(
        &mut self,
        min: Vec3,
        max: Vec3,
        viewport_width: u32,
        viewport_height: u32,
        margin: f32,
    ) {
        self.pose.frame_bounds(
            min,
            max,
            viewport_width,
            viewport_height,
            margin,
            self.projection,
        );
        self.inertia.stop();
        self.transition.cancel();
    }

    // ─── Inertia ─────────────────────────────────────────────────────────────

    /// `yaw_delta` / `pitch_delta` are radians applied on the last orbit frame (egui drag delta × sensitivity).
    /// Convert to rad/s so [`update_inertia`] coast matches release speed.
    pub fn orbit_inertia(&mut self, yaw_delta: f32, pitch_delta: f32) {
        const REF_FPS: f32 = 60.0;
        self.inertia.yaw_velocity += yaw_delta * REF_FPS;
        self.inertia.pitch_velocity += pitch_delta * REF_FPS;
    }

    pub fn pan_inertia(&mut self, delta_x: f32, delta_y: f32) {
        const REF_FPS: f32 = 60.0;
        let s = self.pose.distance * self.navigation.pan_sensitivity * 0.4 * REF_FPS;
        let (right, up) = self.pose.view_plane_axes();
        self.inertia.target_velocity += right * (-delta_x * s) + up * (delta_y * s);
    }

    pub fn zoom_inertia(&mut self, delta: f32) {
        self.inertia.distance_velocity += delta * 0.0005;
    }

    pub fn stop_inertia(&mut self) {
        self.inertia.stop();
    }

    pub fn has_inertia(&self) -> bool {
        self.inertia.has_motion(self.inertia_cutoff)
    }

    pub fn update_inertia(&mut self, dt: f32) -> bool {
        if !self.inertia.has_motion(self.inertia_cutoff) {
            return false;
        }
        // Do not call `orbit()` — it clears inertia every frame.
        self.pose.orbit(
            self.inertia.yaw_velocity * dt,
            self.inertia.pitch_velocity * dt,
            self.orientation_mode,
            self.pitch_limit,
        );
        self.pose.distance = self.projection.clamp_distance(
            self.pose.distance * (self.inertia.distance_velocity * dt).exp(),
        );
        self.pose.target += self.inertia.target_velocity * dt;

        let decay = (-self.inertia_friction * dt).exp();
        let threshold = self.inertia_cutoff.max(1e-6);
        self.inertia.yaw_velocity *= decay;
        self.inertia.pitch_velocity *= decay;
        self.inertia.distance_velocity *= decay;
        self.inertia.target_velocity *= decay;

        if self.inertia.yaw_velocity.abs() < threshold {
            self.inertia.yaw_velocity = 0.0;
        }
        if self.inertia.pitch_velocity.abs() < threshold {
            self.inertia.pitch_velocity = 0.0;
        }
        if self.inertia.distance_velocity.abs() < threshold {
            self.inertia.distance_velocity = 0.0;
        }
        if self.inertia.target_velocity.length() < threshold {
            self.inertia.target_velocity = Vec3::ZERO;
        }

        self.inertia.has_motion(threshold)
    }

    // ─── Transitions ─────────────────────────────────────────────────────────

    pub fn animate_to(&mut self, yaw: f32, pitch: f32, distance: f32, target: Vec3) {
        let orientation = orientation_from_yaw_pitch(yaw, pitch);
        let distance = self.projection.clamp_distance(distance);
        self.transition
            .animate_to(orientation, distance, target, &mut self.inertia);
    }

    pub fn cancel_animation(&mut self) {
        self.transition.cancel();
    }

    pub fn is_animating(&self) -> bool {
        self.transition.animating
    }

    pub fn update_animation(&mut self, dt: f32) -> bool {
        self.transition.step(
            &mut self.pose.orientation,
            &mut self.pose.distance,
            &mut self.pose.target,
            dt,
            self.transition_speed,
        )
    }

    pub fn update_dynamics(&mut self, dt: f32) -> bool {
        let a = self.update_inertia(dt);
        let b = self.update_animation(dt);
        a || b
    }

    // ─── Viewport helpers ────────────────────────────────────────────────────

    pub fn viewport_size(&self, width: u32, height: u32) -> ViewportSize {
        ViewportSize::new(width, height)
    }

    pub fn viewport_ndc(&self, point: ViewportPoint, size: ViewportSize) -> [f32; 2] {
        self.pose.viewport_ndc(point, size)
    }

    pub fn screen_ndc(&self, px: f32, py: f32, width: u32, height: u32) -> [f32; 2] {
        self.pose.screen_ndc(px, py, width, height)
    }

    pub fn inv_view_proj_cols(&self, width: u32, height: u32) -> [[f32; 4]; 4] {
        self.pose
            .inv_view_proj_cols(width, height, self.projection)
    }

    pub fn eye_pos(&self) -> [f32; 3] {
        self.pose.eye_pos()
    }

    pub fn viewport_ray(&self, point: ViewportPoint, size: ViewportSize) -> ViewportRay {
        self.pose
            .viewport_ray(point, size, self.projection)
    }

    pub fn screen_ray(&self, px: f32, py: f32, width: u32, height: u32) -> ([f32; 3], [f32; 3]) {
        self.pose
            .screen_ray(px, py, width, height, self.projection)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_bounds_sets_target_and_distance() {
        let mut rig = OrbitRig::default();
        rig.frame_bounds(
            Vec3::new(-10.0, -5.0, -10.0),
            Vec3::new(10.0, 5.0, 10.0),
            800,
            600,
            1.2,
        );
        assert!(rig.pose.target.length() < 1e-3);
        assert!(rig.pose.distance > rig.projection.distance_min);
    }

    #[test]
    fn inertia_coasts_multiple_frames() {
        let mut rig = OrbitRig::default();
        rig.inertia_friction = 2.0;
        let q0 = rig.pose.orientation;
        rig.orbit_inertia(0.08, 0.0);
        assert!(rig.update_inertia(1.0 / 60.0));
        assert_ne!(q0, rig.pose.orientation);
        let mut frames = 0_u32;
        while rig.update_inertia(1.0 / 60.0) {
            frames += 1;
            if frames > 300 {
                break;
            }
        }
        assert!(frames > 8, "expected multi-frame coast, got {frames}");
    }

    #[test]
    fn inertia_decays() {
        let mut rig = OrbitRig::default();
        rig.orbit_inertia(0.05, 0.0);
        // Inertia must eventually decay to a full stop. With the default friction
        // (0.8) this takes ~774 frames at 60 Hz, so iterate with generous headroom;
        // the assertion is that it DID come to rest within the cap — not that it
        // stops by some arbitrary early frame (the original 500-frame bound was too
        // tight and made this test fail spuriously).
        let mut moving = true;
        let mut frames = 0_u32;
        for _ in 0..2000 {
            moving = rig.update_inertia(1.0 / 60.0);
            frames += 1;
            if !moving {
                break;
            }
        }
        assert!(!moving, "inertia failed to decay to rest within {frames} frames");
    }

    #[test]
    fn orbit_pose_is_copy() {
        let _ = OrbitPose::default();
    }
}
