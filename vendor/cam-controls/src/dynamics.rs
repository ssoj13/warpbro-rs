//! Inertia and animated transitions (ported from vfx-rs `render-camera`).
//!
//! Two families live here:
//! - [`InertiaState`] / [`TransitionState`] — orbit-rig coast and fly-to (the
//!   original, untouched API used by [`crate::rig::OrbitRig`]).
//! - [`FlyInertia`] + [`damp_scalar`] / [`damp_vec3`] — shared 6-DoF momentum
//!   helpers used by the FPS / spaceflight backends. Added additively; the
//!   orbit path does not depend on them.

use glam::{Quat, Vec3};

use crate::orientation::yaw_pitch_from_orientation;

/// Exponential decay factor for a per-frame velocity given a damping rate.
///
/// **What:** returns `exp(-damping * dt)`, the multiplier to apply to a velocity
/// so it decays toward zero with a time-constant of `1 / damping` seconds.
///
/// **Why:** the orbit inertia uses `(-friction * dt).exp()` inline; the fly /
/// space backends share the same frame-rate-independent decay through this
/// single helper so all rigs feel consistent and tuning is centralized.
///
/// **Where:** [`FlyInertia::integrate`], spaceflight angular damping.
pub fn damp_factor(damping: f32, dt: f32) -> f32 {
    (-damping.max(0.0) * dt).exp()
}

/// Damp a scalar velocity toward zero, snapping to `0.0` below `cutoff`.
///
/// **Why:** avoids endless micro-jitter; mirrors the threshold logic in
/// [`InertiaState::integrate_euler_orbit`] but reusable by the fly rigs.
pub fn damp_scalar(value: &mut f32, damping: f32, cutoff: f32, dt: f32) {
    *value *= damp_factor(damping, dt);
    if value.abs() < cutoff.max(1e-6) {
        *value = 0.0;
    }
}

/// Damp a `Vec3` velocity toward zero, snapping to `ZERO` below `cutoff`.
///
/// **Why:** shared linear-momentum decay for [`FlyInertia`] (FPS / spaceflight
/// translation) so damping behaviour matches [`damp_scalar`].
pub fn damp_vec3(value: &mut Vec3, damping: f32, cutoff: f32, dt: f32) {
    *value *= damp_factor(damping, dt);
    if value.length() < cutoff.max(1e-6) {
        *value = Vec3::ZERO;
    }
}

/// 6-DoF momentum for the fly / spaceflight backends.
///
/// **What:** holds a world-space linear velocity and a body-space angular
/// velocity `(yaw, pitch, roll)` in rad/s. Thrust / look / roll inputs add to
/// these velocities; [`FlyInertia::integrate`] coasts and damps them so the
/// camera keeps drifting after the input stops (the "spaceship" feel).
///
/// **Why a separate type from [`InertiaState`]:** `InertiaState` is expressed in
/// orbit terms (yaw/pitch/distance/target velocities around a pivot).
/// [`FlyInertia`] is pivot-free: linear momentum on the eye + angular momentum
/// on the orientation. Keeping them separate avoids changing the orbit API.
///
/// **Where:** [`crate::controller::FpsFly`], [`crate::controller::SpaceFlight`].
#[derive(Debug, Clone, Copy)]
pub struct FlyInertia {
    /// World-space linear velocity (units/s).
    pub linear: Vec3,
    /// Body-space angular velocity (yaw, pitch, roll) in rad/s.
    pub angular: Vec3,
}

impl Default for FlyInertia {
    fn default() -> Self {
        Self {
            linear: Vec3::ZERO,
            angular: Vec3::ZERO,
        }
    }
}

impl FlyInertia {
    /// Reset all momentum to zero.
    pub fn stop(&mut self) {
        *self = Self::default();
    }

    /// True while either linear or angular velocity exceeds `cutoff`.
    pub fn has_motion(&self, cutoff: f32) -> bool {
        let c = cutoff.max(1e-6);
        self.linear.length() > c || self.angular.length() > c
    }

    /// Integrate momentum into `eye` / `orientation`, then damp.
    ///
    /// **What:** advances the eye by `linear * dt`, rotates the orientation by
    /// the body-space angular velocity (`yaw` about world-up so heading stays
    /// level for fly rigs that pass `roll_locked = true`, otherwise about the
    /// camera-local up so roll carries through), pitch about camera-right, roll
    /// about camera-forward; then decays both velocities. Returns `true` while
    /// still moving.
    ///
    /// **Why `roll_locked`:** the FPS rig must never roll and yaws about world
    /// up; the spaceflight rig is fully free and yaws about its own up.
    pub fn integrate(
        &mut self,
        eye: &mut Vec3,
        orientation: &mut Quat,
        dt: f32,
        linear_damping: f32,
        angular_damping: f32,
        cutoff: f32,
        roll_locked: bool,
    ) -> bool {
        if !self.has_motion(cutoff) {
            return false;
        }
        *eye += self.linear * dt;

        let (yaw, pitch, roll) = (self.angular.x, self.angular.y, self.angular.z);
        if yaw != 0.0 {
            let yaw_axis = if roll_locked {
                WORLD_UP_VEC
            } else {
                (*orientation * Vec3::Y).normalize_or_zero()
            };
            if yaw_axis.length_squared() > 1e-10 {
                *orientation =
                    (Quat::from_axis_angle(yaw_axis, yaw * dt) * *orientation).normalize();
            }
        }
        if pitch != 0.0 {
            let right = (*orientation * Vec3::X).normalize_or_zero();
            if right.length_squared() > 1e-10 {
                *orientation =
                    (Quat::from_axis_angle(right, pitch * dt) * *orientation).normalize();
            }
        }
        if !roll_locked && roll != 0.0 {
            let fwd = (*orientation * -Vec3::Z).normalize_or_zero();
            if fwd.length_squared() > 1e-10 {
                *orientation =
                    (Quat::from_axis_angle(fwd, roll * dt) * *orientation).normalize();
            }
        }

        damp_vec3(&mut self.linear, linear_damping, cutoff, dt);
        damp_scalar(&mut self.angular.x, angular_damping, cutoff, dt);
        damp_scalar(&mut self.angular.y, angular_damping, cutoff, dt);
        damp_scalar(&mut self.angular.z, angular_damping, cutoff, dt);

        self.has_motion(cutoff)
    }
}

/// Local copy of the world up axis (the [`crate::convention::WORLD_UP`] const)
/// kept here so `dynamics` has no upward module dependency cycle worry.
const WORLD_UP_VEC: Vec3 = Vec3::Y;

/// Exponential friction decay for orbit inertia (rad/s and world units/s).
#[derive(Debug, Clone, Copy)]
pub struct InertiaState {
    pub yaw_velocity: f32,
    pub pitch_velocity: f32,
    pub distance_velocity: f32,
    pub target_velocity: Vec3,
}

impl Default for InertiaState {
    fn default() -> Self {
        Self {
            yaw_velocity: 0.0,
            pitch_velocity: 0.0,
            distance_velocity: 0.0,
            target_velocity: Vec3::ZERO,
        }
    }
}

impl InertiaState {
    pub fn stop(&mut self) {
        *self = Self::default();
    }

    pub fn has_motion(&self, threshold: f32) -> bool {
        let t = threshold.max(1e-6);
        self.yaw_velocity.abs() > t
            || self.pitch_velocity.abs() > t
            || self.distance_velocity.abs() > t
            || self.target_velocity.length() > t
    }

    /// Integrate velocities into yaw/pitch/distance/target. Returns true while moving.
    pub fn integrate_euler_orbit(
        &mut self,
        yaw: &mut f32,
        pitch: &mut f32,
        distance: &mut f32,
        target: &mut Vec3,
        dt: f32,
        friction: f32,
        cutoff: f32,
        pitch_limit: f32,
        distance_min: f32,
        distance_max: f32,
    ) -> bool {
        let decay = (-friction * dt).exp();
        let threshold = cutoff.max(1e-6);

        *yaw += self.yaw_velocity * dt;
        *pitch = (*pitch + self.pitch_velocity * dt).clamp(-pitch_limit, pitch_limit);
        *distance = (*distance * (self.distance_velocity * dt).exp()).clamp(distance_min, distance_max);
        *target += self.target_velocity * dt;

        self.yaw_velocity *= decay;
        self.pitch_velocity *= decay;
        self.distance_velocity *= decay;
        self.target_velocity *= decay;

        if self.yaw_velocity.abs() < threshold {
            self.yaw_velocity = 0.0;
        }
        if self.pitch_velocity.abs() < threshold {
            self.pitch_velocity = 0.0;
        }
        if self.distance_velocity.abs() < threshold {
            self.distance_velocity = 0.0;
        }
        if self.target_velocity.length() < threshold {
            self.target_velocity = Vec3::ZERO;
        }

        self.has_motion(threshold)
    }
}

/// Smooth fly-to targets (orientation via yaw/pitch decomposition + slerp optional at rig layer).
#[derive(Debug, Clone)]
pub struct TransitionState {
    pub orientation_target: Quat,
    pub distance_target: f32,
    pub target_target: Vec3,
    pub animating: bool,
}

impl Default for TransitionState {
    fn default() -> Self {
        Self {
            orientation_target: Quat::IDENTITY,
            distance_target: 28.0,
            target_target: Vec3::ZERO,
            animating: false,
        }
    }
}

impl TransitionState {
    pub fn animate_to(
        &mut self,
        orientation: Quat,
        distance: f32,
        target: Vec3,
        inertia: &mut InertiaState,
    ) {
        self.orientation_target = orientation;
        self.distance_target = distance;
        self.target_target = target;
        self.animating = true;
        inertia.stop();
    }

    pub fn cancel(&mut self) {
        self.animating = false;
    }

    /// Lerp/slerp toward targets. Returns true while animating.
    pub fn step(
        &mut self,
        orientation: &mut Quat,
        distance: &mut f32,
        target: &mut Vec3,
        dt: f32,
        speed: f32,
    ) -> bool {
        if !self.animating {
            return false;
        }
        let t = (speed * dt).min(1.0);
        *orientation = orientation.slerp(self.orientation_target, t);
        *distance += (self.distance_target - *distance) * t;
        *target += (self.target_target - *target) * t;

        let (y0, p0) = yaw_pitch_from_orientation(*orientation);
        let (y1, p1) = yaw_pitch_from_orientation(self.orientation_target);
        let eps = 0.001_f32;
        if (y0 - y1).abs() < eps
            && (p0 - p1).abs() < eps
            && (*distance - self.distance_target).abs() < eps
            && (*target - self.target_target).length() < eps
        {
            *orientation = self.orientation_target;
            *distance = self.distance_target;
            *target = self.target_target;
            self.animating = false;
        }
        self.animating
    }
}
