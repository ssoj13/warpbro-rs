//! Pluggable camera backends behind one [`CameraController`] enum.
//!
//! Four navigation styles share a single [`CameraPose`] output so render code is
//! backend-agnostic:
//!
//! | Variant                 | Backend         | Style |
//! |-------------------------|-----------------|-------|
//! | [`CameraController::Houdini`] | [`HoudiniOrbit`] | classic euler tumble/dolly/pan, wraps the existing [`OrbitRig`] (visually identical to today's orbit) |
//! | [`CameraController::Quat`]    | [`QuatOrbit`]    | gimbal-free quaternion orbit around a target |
//! | [`CameraController::Fps`]     | [`FpsFly`]       | yaw/pitch mouse-look + WASD thrust, gravity-free, no roll |
//! | [`CameraController::Space`]   | [`SpaceFlight`]  | full 6-DoF spaceship: free eye + quaternion orientation, thrust, roll, boost, inertial drift |
//!
//! All backends expose `apply_intent` / `update_dynamics` / `pose` /
//! `set_pose` / `from_pose` so the orchestrator can swap styles while keeping
//! view continuity (see [`CameraController::set_pose`]).

use cam_viewport::ViewportSize;
use glam::{Quat, Vec2, Vec3};

use crate::controls::CameraIntent;
use crate::convention::WORLD_UP;
use crate::dynamics::FlyInertia;
use crate::orientation::DEFAULT_PITCH_LIMIT;
use crate::pose::CameraPose;
use crate::projection::PerspectiveSettings;
use crate::rig::{OrbitPose, OrbitRig};
use crate::settings::{CameraNavigationSettings, InertiaSettings};

/// Reference frame rate used to turn a per-frame mouse-look delta into a coast
/// velocity (rad/s), mirroring [`OrbitRig::orbit_inertia`]'s `REF_FPS`.
const REF_FPS: f32 = 60.0;

// ───────────────────────── Houdini (orbit wrapper) ─────────────────────────

/// Classic orbit backend — a thin wrapper over the existing [`OrbitRig`].
///
/// **What:** delegates every operation to [`OrbitRig`] so its tumble / dolly /
/// pan feel and its inertia are byte-for-byte the same as the current viewport.
/// `pose()` converts the rig's [`OrbitPose`] into the shared [`CameraPose`].
///
/// **Why a wrapper instead of reimplementing:** the orbit rig is the
/// correctness anchor downstream; reusing it guarantees the Houdini style stays
/// visually identical (the `houdini_pose_matches_orbit` test enforces this).
#[derive(Debug, Clone, Default)]
pub struct HoudiniOrbit {
    /// The wrapped orbit rig (public so callers needing the full orbit API can
    /// reach it; `pose()` is the backend-agnostic accessor).
    pub rig: OrbitRig,
}

impl HoudiniOrbit {
    /// Wrap an existing [`OrbitRig`].
    pub fn new(rig: OrbitRig) -> Self {
        Self { rig }
    }

    /// Apply one intent. Orbit-native intents drive the rig; fly/space intents
    /// are no-ops (this backend is not free-flying).
    pub fn apply_intent(&mut self, intent: CameraIntent, viewport: ViewportSize) {
        match intent {
            CameraIntent::Orbit { yaw_delta, pitch_delta } => {
                self.rig.orbit(yaw_delta, pitch_delta)
            }
            CameraIntent::OrbitInertia { yaw_delta, pitch_delta } => {
                self.rig.orbit_inertia(yaw_delta, pitch_delta)
            }
            CameraIntent::Pan { dx_px, dy_px } => self.rig.pan(dx_px, dy_px),
            CameraIntent::PanInertia { dx_px, dy_px } => self.rig.pan_inertia(dx_px, dy_px),
            CameraIntent::Zoom { factor } => self.rig.zoom(factor),
            CameraIntent::ZoomInertia { delta } => self.rig.zoom_inertia(delta),
            CameraIntent::ZoomAtCursor { cursor_x, cursor_y, factor } => {
                self.rig.zoom_at_cursor(
                    cam_viewport::ViewportPoint { x: cursor_x, y: cursor_y },
                    viewport,
                    factor,
                );
            }
            CameraIntent::FrameBounds { min, max, margin } => {
                self.rig
                    .frame_bounds(min, max, viewport.width, viewport.height, margin);
            }
            CameraIntent::Look { .. }
            | CameraIntent::LookRate { .. }
            | CameraIntent::Roll { .. }
            | CameraIntent::Thrust { .. }
            | CameraIntent::Boost(_) => {}
        }
    }

    /// Convert the orbit pose into the shared [`CameraPose`].
    pub fn pose(&self) -> CameraPose {
        CameraPose::from_orbit(&self.rig.pose, self.rig.projection)
    }
}

// ───────────────────────────── Quaternion orbit ────────────────────────────

/// Gimbal-free orbit around a target using quaternion accumulation.
///
/// **What:** like the orbit rig it keeps a `target` + `distance`, but the
/// orientation is a free quaternion accumulated incrementally (yaw about the
/// camera's current up, pitch about its current right) instead of a
/// yaw/pitch euler pair. This removes the pole singularity the euler turntable
/// suffers from.
///
/// **Why separate from [`HoudiniOrbit`]:** the Houdini style intentionally
/// matches the legacy clamped turntable; this one trades that for pole-safe
/// free tumbling.
#[derive(Debug, Clone)]
pub struct QuatOrbit {
    /// Pivot the camera orbits.
    pub target: Vec3,
    /// Distance from `target` to `eye`.
    pub distance: f32,
    /// Free camera orientation (local -Z = forward toward `target`).
    pub orientation: Quat,
    pub projection: PerspectiveSettings,
    pub navigation: CameraNavigationSettings,
    pub inertia: InertiaSettings,
    /// Body-space angular momentum for orbit coast (linear unused here).
    pub momentum: FlyInertia,
}

impl Default for QuatOrbit {
    fn default() -> Self {
        Self::from_pose(CameraPose::default())
    }
}

impl QuatOrbit {
    /// Build from a free pose: the orbit target is placed `eye + forward * d`
    /// where `d` is a sensible default distance derived from the projection.
    pub fn from_pose(pose: CameraPose) -> Self {
        let projection = PerspectiveSettings::default();
        let distance = projection.clamp_distance(28.0);
        let target = pose.eye + pose.forward() * distance;
        Self {
            target,
            distance,
            orientation: pose.orientation,
            projection,
            navigation: CameraNavigationSettings::default(),
            inertia: InertiaSettings::default(),
            momentum: FlyInertia::default(),
        }
    }

    /// Current eye position (`target - forward * distance`).
    fn eye(&self) -> Vec3 {
        self.target - (self.orientation * -Vec3::Z).normalize_or_zero() * self.distance
    }

    /// Tumble by `(dyaw, dpitch)` radians about the camera's current up/right.
    fn tumble(&mut self, dyaw: f32, dpitch: f32) {
        if dyaw != 0.0 {
            let up = (self.orientation * Vec3::Y).normalize_or_zero();
            if up.length_squared() > 1e-10 {
                self.orientation =
                    (Quat::from_axis_angle(up, dyaw) * self.orientation).normalize();
            }
        }
        if dpitch != 0.0 {
            let right = (self.orientation * Vec3::X).normalize_or_zero();
            if right.length_squared() > 1e-10 {
                self.orientation =
                    (Quat::from_axis_angle(right, dpitch) * self.orientation).normalize();
            }
        }
    }

    /// Apply one intent. Orbit / look intents tumble; zoom changes distance;
    /// pan slides the target on the view plane; thrust/roll/boost are no-ops.
    pub fn apply_intent(&mut self, intent: CameraIntent, _viewport: ViewportSize) {
        match intent {
            CameraIntent::Orbit { yaw_delta, pitch_delta } => {
                self.momentum.stop();
                self.tumble(-yaw_delta, -pitch_delta);
            }
            CameraIntent::Look { dyaw, dpitch } => {
                self.momentum.stop();
                self.tumble(-dyaw, -dpitch);
            }
            CameraIntent::OrbitInertia { yaw_delta, pitch_delta } => {
                self.momentum.angular.x += -yaw_delta * REF_FPS;
                self.momentum.angular.y += -pitch_delta * REF_FPS;
            }
            CameraIntent::Zoom { factor } => {
                self.distance = self.projection.clamp_distance(self.distance * factor);
            }
            CameraIntent::ZoomInertia { delta } => {
                self.distance = self
                    .projection
                    .clamp_distance(self.distance * (1.0 + delta * 0.001));
            }
            CameraIntent::ZoomAtCursor { factor, .. } => {
                self.distance = self.projection.clamp_distance(self.distance * factor);
            }
            CameraIntent::Pan { dx_px, dy_px } => {
                let pose = self.pose();
                let (right, up) = pose.view_plane_axes();
                let scale = self.distance * self.navigation.pan_sensitivity;
                self.target += right * (-dx_px * scale) + up * (dy_px * scale);
            }
            CameraIntent::FrameBounds { min, max, margin } => {
                self.frame_aabb(min, max, 1, 1, margin);
            }
            CameraIntent::PanInertia { .. }
            | CameraIntent::LookRate { .. }
            | CameraIntent::Roll { .. }
            | CameraIntent::Thrust { .. }
            | CameraIntent::Boost(_) => {}
        }
    }

    /// Frame an AABB (shared with [`CameraController::frame_bounds`]).
    fn frame_aabb(&mut self, min: Vec3, max: Vec3, width: u32, height: u32, margin: f32) {
        let (center, radius) = crate::fit::aabb_center_radius(min, max);
        let aspect = width as f32 / height.max(1) as f32;
        self.target = center;
        self.distance = crate::fit::distance_to_frame_sphere(
            radius,
            aspect,
            self.projection.fov_y,
            margin,
            self.projection,
        );
        self.momentum.stop();
    }

    /// Coast angular momentum (tumble drift). Returns `true` while moving.
    pub fn update_dynamics(&mut self, dt: f32) -> bool {
        if !self.momentum.has_motion(self.inertia.cutoff) {
            return false;
        }
        let dyaw = self.momentum.angular.x * dt;
        let dpitch = self.momentum.angular.y * dt;
        self.tumble(dyaw, dpitch);
        crate::dynamics::damp_scalar(
            &mut self.momentum.angular.x,
            self.inertia.angular_damping,
            self.inertia.cutoff,
            dt,
        );
        crate::dynamics::damp_scalar(
            &mut self.momentum.angular.y,
            self.inertia.angular_damping,
            self.inertia.cutoff,
            dt,
        );
        self.momentum.has_motion(self.inertia.cutoff)
    }

    /// Build the shared pose from the current orbit state.
    pub fn pose(&self) -> CameraPose {
        CameraPose {
            eye: self.eye(),
            orientation: self.orientation,
            fov_y: self.projection.fov_y,
            znear: self.projection.near,
            zfar: self.projection.far,
        }
    }
}

// ───────────────────────────────── FPS fly ─────────────────────────────────

/// First-person fly backend: yaw/pitch mouse-look + WASD-style thrust.
///
/// **What:** the eye flies freely; orientation is constrained to yaw (about
/// world up) + pitch (clamped) with **no roll**, like an FPS camera. `Thrust`
/// moves the eye relative to the camera basis. Gravity-free.
///
/// **Why:** game-style navigation distinct from the spaceship — heading stays
/// level so the horizon never tilts.
#[derive(Debug, Clone)]
pub struct FpsFly {
    pub eye: Vec3,
    /// Yaw about world up (radians).
    pub yaw: f32,
    /// Pitch about camera right (radians, clamped to ±[`DEFAULT_PITCH_LIMIT`]).
    pub pitch: f32,
    pub projection: PerspectiveSettings,
    pub inertia: InertiaSettings,
    /// Linear momentum (angular unused — look is applied directly).
    pub momentum: FlyInertia,
    boost: bool,
    /// Held thrust axes (x=forward, y=right, z=up), refreshed every frame by the
    /// `Thrust` intent. Integrated into `momentum.linear` with REAL dt in
    /// `update_dynamics`, so flight speed is frame-rate-independent.
    thrust_input: Vec3,
    /// Held mouse-steer axes (yaw/pitch ∈ [−1,1], deflection from screen centre),
    /// refreshed every frame by the `LookRate` intent in `MouseSteer::Deflection`.
    /// Integrated into yaw/pitch with REAL dt in `update_dynamics`; zero in
    /// `MouseSteer::Relative` (the classic per-frame `Look` delta is used instead).
    look_input: Vec2,
}

impl Default for FpsFly {
    fn default() -> Self {
        Self::from_pose(CameraPose::default())
    }
}

impl FpsFly {
    /// Build from a free pose, decomposing its forward into yaw/pitch (roll is
    /// discarded — this rig cannot roll).
    pub fn from_pose(pose: CameraPose) -> Self {
        let fwd = pose.forward();
        // Invert `forward = Ry(yaw) * Rx(pitch) * -Z`:
        //   forward = (-cosθ·sin(yaw), sinθ, -cosθ·cos(yaw))
        // so yaw = atan2(-fwd.x, -fwd.z) and pitch = asin(fwd.y).
        let yaw = (-fwd.x).atan2(-fwd.z);
        let horiz = Vec3::new(fwd.x, 0.0, fwd.z).length();
        let pitch = fwd.y.atan2(horiz.max(1e-6));
        Self {
            eye: pose.eye,
            yaw,
            pitch: pitch.clamp(-DEFAULT_PITCH_LIMIT, DEFAULT_PITCH_LIMIT),
            projection: PerspectiveSettings::default(),
            inertia: InertiaSettings::fps(),
            momentum: FlyInertia::default(),
            boost: false,
            thrust_input: Vec3::ZERO,
            look_input: Vec2::ZERO,
        }
    }

    /// Orientation from yaw/pitch: yaw about world up, then pitch about right.
    fn orientation(&self) -> Quat {
        Quat::from_axis_angle(WORLD_UP, self.yaw)
            * Quat::from_axis_angle(Vec3::X, self.pitch)
    }

    /// Forward direction implied by yaw/pitch.
    fn forward(&self) -> Vec3 {
        (self.orientation() * -Vec3::Z).normalize_or_zero()
    }

    /// Apply one intent: look turns the head, thrust adds body-relative
    /// velocity, boost toggles the speed multiplier. Roll is ignored.
    pub fn apply_intent(&mut self, intent: CameraIntent, _viewport: ViewportSize) {
        match intent {
            CameraIntent::Look { dyaw, dpitch } | CameraIntent::Orbit { yaw_delta: dyaw, pitch_delta: dpitch } => {
                self.yaw += dyaw;
                self.pitch =
                    (self.pitch + dpitch).clamp(-DEFAULT_PITCH_LIMIT, DEFAULT_PITCH_LIMIT);
            }
            CameraIntent::Thrust { forward, right, up } => {
                // Record the held thrust axes (x=forward, y=right, z=up). The
                // velocity change is integrated with REAL dt in `update_dynamics`
                // (frame-rate-independent); `gather_fly_intents` emits this every
                // frame, zero on release, so the state always tracks the keys.
                self.thrust_input = Vec3::new(forward, right, up);
            }
            CameraIntent::LookRate { yaw, pitch } => {
                // Held mouse-steer axes (deflection from screen centre → turn
                // rate), refreshed every frame. Integrated with real dt in
                // `integrate_look`; FPS keeps yaw level and clamps pitch.
                self.look_input = Vec2::new(yaw, pitch);
            }
            CameraIntent::Boost(on) => self.boost = on,
            // FPS cannot roll; orbit-specific intents do not apply.
            CameraIntent::Roll { .. }
            | CameraIntent::Pan { .. }
            | CameraIntent::PanInertia { .. }
            | CameraIntent::OrbitInertia { .. }
            | CameraIntent::Zoom { .. }
            | CameraIntent::ZoomInertia { .. }
            | CameraIntent::ZoomAtCursor { .. }
            | CameraIntent::FrameBounds { .. } => {}
        }
    }

    /// Integrate the held thrust input into linear momentum over REAL `dt`
    /// (frame-rate-independent acceleration). No-op when no key is held.
    fn integrate_thrust(&mut self, dt: f32) {
        if self.thrust_input == Vec3::ZERO {
            return;
        }
        let o = self.orientation();
        let f = (o * -Vec3::Z).normalize_or_zero();
        let r = (o * Vec3::X).normalize_or_zero();
        let accel = self.inertia.thrust_sensitivity
            * if self.boost { self.inertia.boost_multiplier } else { 1.0 };
        self.momentum.linear += (f * self.thrust_input.x
            + r * self.thrust_input.y
            + WORLD_UP * self.thrust_input.z)
            * accel
            * dt;
    }

    /// Integrate the held mouse-steer axes (deflection → rate) into yaw/pitch over
    /// REAL `dt` (frame-rate-independent). `steer_rate` is the turn rate (rad/s) at
    /// full deflection; pitch is clamped so the FPS horizon stays sane. Only active
    /// in `MouseSteer::Deflection` (the axes stay zero otherwise).
    fn integrate_look(&mut self, dt: f32) {
        if self.look_input == Vec2::ZERO {
            return;
        }
        let rate = self.inertia.steer_rate * dt;
        self.yaw += self.look_input.x * rate;
        self.pitch = (self.pitch + self.look_input.y * rate)
            .clamp(-DEFAULT_PITCH_LIMIT, DEFAULT_PITCH_LIMIT);
    }

    /// Coast linear momentum (orientation is direct, not inertial). Held thrust +
    /// mouse-steer are integrated first (real dt). Returns `true` while still
    /// drifting or actively steering.
    pub fn update_dynamics(&mut self, dt: f32) -> bool {
        self.integrate_thrust(dt);
        self.integrate_look(dt);
        let steering = self.look_input != Vec2::ZERO;
        if self.momentum.linear.length() <= self.inertia.cutoff.max(1e-6) {
            self.momentum.linear = Vec3::ZERO;
            return steering;
        }
        self.eye += self.momentum.linear * dt;
        crate::dynamics::damp_vec3(
            &mut self.momentum.linear,
            self.inertia.linear_damping,
            self.inertia.cutoff,
            dt,
        );
        self.momentum.linear.length() > self.inertia.cutoff.max(1e-6) || steering
    }

    /// Build the shared pose.
    pub fn pose(&self) -> CameraPose {
        CameraPose {
            eye: self.eye,
            orientation: self.orientation(),
            fov_y: self.projection.fov_y,
            znear: self.projection.near,
            zfar: self.projection.far,
        }
    }

    /// Frame an AABB: place the eye back from the box center along the current
    /// forward so the sphere fits.
    fn frame_aabb(&mut self, min: Vec3, max: Vec3, width: u32, height: u32, margin: f32) {
        let (center, radius) = crate::fit::aabb_center_radius(min, max);
        let aspect = width as f32 / height.max(1) as f32;
        let dist = crate::fit::distance_to_frame_sphere(
            radius,
            aspect,
            self.projection.fov_y,
            margin,
            self.projection,
        );
        self.eye = center - self.forward() * dist;
        self.momentum.linear = Vec3::ZERO;
        self.thrust_input = Vec3::ZERO;
        self.look_input = Vec2::ZERO;
    }
}

// ──────────────────────────────── Spaceflight ──────────────────────────────

/// Full 6-DoF "spaceship" backend.
///
/// **What:** free `eye` + free `orientation` quaternion, body-relative `Thrust`
/// (forward/right/up), `Roll`, a `Boost` speed multiplier, and **inertial
/// drift** — momentum keeps the ship moving and rotating after input stops,
/// damping slowly. There is no up constraint, so the horizon can roll freely.
///
/// **Why:** the most permissive style, for free exploration / flythroughs.
#[derive(Debug, Clone)]
pub struct SpaceFlight {
    pub eye: Vec3,
    pub orientation: Quat,
    pub projection: PerspectiveSettings,
    pub inertia: InertiaSettings,
    pub momentum: FlyInertia,
    boost: bool,
    /// Held thrust axes (x=forward, y=right, z=up), refreshed every frame by the
    /// `Thrust` intent. Integrated with REAL dt in `update_dynamics`.
    thrust_input: Vec3,
    /// Held roll axis (−1 = roll left / Q, +1 = roll right / E), refreshed every
    /// frame by the `Roll` intent. Integrated into angular momentum with REAL dt
    /// in `update_dynamics` (frame-rate-independent), mirroring `thrust_input`.
    roll_input: f32,
    /// Held mouse-steer axes (yaw/pitch ∈ [−1,1], deflection from screen centre),
    /// refreshed every frame by the `LookRate` intent in `MouseSteer::Deflection`.
    /// Integrated DIRECTLY into the orientation with REAL dt in `update_dynamics`.
    look_input: Vec2,
}

impl Default for SpaceFlight {
    fn default() -> Self {
        Self::from_pose(CameraPose::default())
    }
}

impl SpaceFlight {
    /// Build from a free pose (eye + orientation preserved exactly, including
    /// roll).
    pub fn from_pose(pose: CameraPose) -> Self {
        Self {
            eye: pose.eye,
            orientation: pose.orientation,
            projection: PerspectiveSettings::default(),
            inertia: InertiaSettings::space(),
            momentum: FlyInertia::default(),
            boost: false,
            thrust_input: Vec3::ZERO,
            roll_input: 0.0,
            look_input: Vec2::ZERO,
        }
    }

    /// Apply one intent: look turns the ship DIRECTLY (1:1, no momentum), roll
    /// adds angular momentum, thrust adds linear momentum (body-relative), boost
    /// scales thrust.
    pub fn apply_intent(&mut self, intent: CameraIntent, _viewport: ViewportSize) {
        match intent {
            CameraIntent::Look { dyaw, dpitch } | CameraIntent::Orbit { yaw_delta: dyaw, pitch_delta: dpitch } => {
                // Mouse-look is applied DIRECTLY to the orientation, NOT through
                // angular momentum: routing it through `momentum.angular` (× REF_FPS,
                // then slow damping) made the view coast/overshoot after the cursor
                // stopped — the "drunken" drift the look-sensitivity slider couldn't
                // tame. Yaw about the ship's own up, pitch about its right; roll +
                // thrust below keep their inertial spaceship feel.
                if dyaw != 0.0 {
                    let up = (self.orientation * Vec3::Y).normalize_or_zero();
                    if up.length_squared() > 1e-10 {
                        self.orientation =
                            (Quat::from_axis_angle(up, dyaw) * self.orientation).normalize();
                    }
                }
                if dpitch != 0.0 {
                    let right = (self.orientation * Vec3::X).normalize_or_zero();
                    if right.length_squared() > 1e-10 {
                        self.orientation =
                            (Quat::from_axis_angle(right, dpitch) * self.orientation).normalize();
                    }
                }
            }
            CameraIntent::Roll { d } => {
                // Held roll axis (−1/0/+1 from Q/E), refreshed every frame like
                // thrust. Integrated with real dt in `integrate_roll` so a held
                // key ramps angular velocity frame-rate-independently — the raw
                // per-frame `+= d * REF_FPS` add was the same accumulation bug the
                // thrust refactor fixed.
                self.roll_input = d;
            }
            CameraIntent::Thrust { forward, right, up } => {
                // Record held thrust axes; integrated with REAL dt in
                // `update_dynamics` (frame-rate-independent). See FpsFly.
                self.thrust_input = Vec3::new(forward, right, up);
            }
            CameraIntent::LookRate { yaw, pitch } => {
                // Held mouse-steer axes (deflection → turn rate), refreshed every
                // frame; integrated DIRECTLY into orientation with real dt in
                // `integrate_look` (no momentum → stops the instant the mouse
                // recentres, no coast).
                self.look_input = Vec2::new(yaw, pitch);
            }
            CameraIntent::Boost(on) => self.boost = on,
            CameraIntent::OrbitInertia { .. }
            | CameraIntent::Pan { .. }
            | CameraIntent::PanInertia { .. }
            | CameraIntent::Zoom { .. }
            | CameraIntent::ZoomInertia { .. }
            | CameraIntent::ZoomAtCursor { .. }
            | CameraIntent::FrameBounds { .. } => {}
        }
    }

    /// Integrate the held thrust input into linear momentum over REAL `dt`.
    fn integrate_thrust(&mut self, dt: f32) {
        if self.thrust_input == Vec3::ZERO {
            return;
        }
        let f = (self.orientation * -Vec3::Z).normalize_or_zero();
        let r = (self.orientation * Vec3::X).normalize_or_zero();
        let u = (self.orientation * Vec3::Y).normalize_or_zero();
        let accel = self.inertia.thrust_sensitivity
            * if self.boost { self.inertia.boost_multiplier } else { 1.0 };
        self.momentum.linear += (f * self.thrust_input.x
            + r * self.thrust_input.y
            + u * self.thrust_input.z)
            * accel
            * dt;
    }

    /// Integrate the held roll axis into angular momentum over REAL `dt`
    /// (frame-rate-independent). `roll_sensitivity` is the angular acceleration
    /// (rad/s²) per unit axis; after the key is released the roll coasts and
    /// damps via `momentum.integrate` (angular_damping), keeping the inertial
    /// spaceship feel. Mirrors `integrate_thrust`.
    fn integrate_roll(&mut self, dt: f32) {
        if self.roll_input == 0.0 {
            return;
        }
        self.momentum.angular.z += self.roll_input * self.inertia.roll_sensitivity * dt;
    }

    /// Integrate the held mouse-steer axes (deflection → turn rate) DIRECTLY into
    /// the orientation over REAL `dt`: yaw about the ship's own up, pitch about its
    /// right. Free roll, no pitch clamp. `steer_rate` is the turn rate (rad/s) at
    /// full deflection. Direct (not momentum) so the ship stops turning the instant
    /// the mouse recentres — no coast. Zero axes (Relative mode / centred) = no-op.
    fn integrate_look(&mut self, dt: f32) {
        if self.look_input == Vec2::ZERO {
            return;
        }
        let rate = self.inertia.steer_rate * dt;
        let dyaw = self.look_input.x * rate;
        let dpitch = self.look_input.y * rate;
        if dyaw != 0.0 {
            let up = (self.orientation * Vec3::Y).normalize_or_zero();
            if up.length_squared() > 1e-10 {
                self.orientation =
                    (Quat::from_axis_angle(up, dyaw) * self.orientation).normalize();
            }
        }
        if dpitch != 0.0 {
            let right = (self.orientation * Vec3::X).normalize_or_zero();
            if right.length_squared() > 1e-10 {
                self.orientation =
                    (Quat::from_axis_angle(right, dpitch) * self.orientation).normalize();
            }
        }
    }

    /// Integrate + damp 6-DoF momentum. `roll_locked = false` so the ship rolls
    /// freely. Held thrust + roll + mouse-steer are integrated first (real dt).
    /// Returns `true` while still moving (incl. active steering).
    pub fn update_dynamics(&mut self, dt: f32) -> bool {
        self.integrate_thrust(dt);
        self.integrate_roll(dt);
        self.integrate_look(dt);
        let moving = self.momentum.integrate(
            &mut self.eye,
            &mut self.orientation,
            dt,
            self.inertia.linear_damping,
            self.inertia.angular_damping,
            self.inertia.cutoff,
            false,
        );
        moving || self.look_input != Vec2::ZERO
    }

    /// Build the shared pose.
    pub fn pose(&self) -> CameraPose {
        CameraPose {
            eye: self.eye,
            orientation: self.orientation,
            fov_y: self.projection.fov_y,
            znear: self.projection.near,
            zfar: self.projection.far,
        }
    }

    /// Frame an AABB: pull the eye back along forward so the sphere fits,
    /// keeping the current orientation (roll preserved).
    fn frame_aabb(&mut self, min: Vec3, max: Vec3, width: u32, height: u32, margin: f32) {
        let (center, radius) = crate::fit::aabb_center_radius(min, max);
        let aspect = width as f32 / height.max(1) as f32;
        let dist = crate::fit::distance_to_frame_sphere(
            radius,
            aspect,
            self.projection.fov_y,
            margin,
            self.projection,
        );
        let fwd = (self.orientation * -Vec3::Z).normalize_or_zero();
        self.eye = center - fwd * dist;
        self.momentum.stop();
        self.thrust_input = Vec3::ZERO;
        self.roll_input = 0.0;
        self.look_input = Vec2::ZERO;
    }
}

// ───────────────────────────── The enum facade ─────────────────────────────

/// One of four pluggable camera navigation backends.
///
/// See the [module docs](self) for the per-variant behaviour table. The inherent
/// methods dispatch to the active backend, so callers never match on the
/// variant directly — they drive whatever style is selected and read a uniform
/// [`CameraPose`] back.
#[derive(Debug, Clone)]
pub enum CameraController {
    /// Classic euler orbit (wraps [`OrbitRig`]) — see [`HoudiniOrbit`].
    Houdini(HoudiniOrbit),
    /// Gimbal-free quaternion orbit — see [`QuatOrbit`].
    Quat(QuatOrbit),
    /// FPS fly (mouse-look + thrust, no roll) — see [`FpsFly`].
    Fps(FpsFly),
    /// 6-DoF spaceship (thrust + roll + boost + drift) — see [`SpaceFlight`].
    Space(SpaceFlight),
}

impl Default for CameraController {
    /// Defaults to the Houdini orbit so behaviour matches the current viewport.
    fn default() -> Self {
        Self::Houdini(HoudiniOrbit::default())
    }
}

impl CameraController {
    /// Apply one [`CameraIntent`] to the active backend.
    ///
    /// `dt` is the frame delta; orbit-style backends ignore it (their inputs
    /// are per-frame deltas already), but it is part of the uniform signature
    /// the orchestrator calls. `viewport` carries the real pixel dimensions —
    /// required so cursor-anchored zoom (`ZoomAtCursor`) maps the cursor to the
    /// correct NDC and keeps the point under the cursor fixed.
    pub fn apply_intent(&mut self, intent: CameraIntent, _dt: f32, viewport: ViewportSize) {
        match self {
            CameraController::Houdini(b) => b.apply_intent(intent, viewport),
            CameraController::Quat(b) => b.apply_intent(intent, viewport),
            CameraController::Fps(b) => b.apply_intent(intent, viewport),
            CameraController::Space(b) => b.apply_intent(intent, viewport),
        }
    }

    /// Advance inertia / coast by `dt`. Returns `true` if still moving (the
    /// host should keep requesting frames).
    pub fn update_dynamics(&mut self, dt: f32) -> bool {
        match self {
            CameraController::Houdini(b) => b.rig.update_dynamics(dt),
            CameraController::Quat(b) => b.update_dynamics(dt),
            CameraController::Fps(b) => b.update_dynamics(dt),
            CameraController::Space(b) => b.update_dynamics(dt),
        }
    }

    /// True if the backend currently has residual motion (inertia/coast).
    pub fn has_motion(&self) -> bool {
        match self {
            CameraController::Houdini(b) => b.rig.has_inertia() || b.rig.is_animating(),
            CameraController::Quat(b) => b.momentum.has_motion(b.inertia.cutoff),
            CameraController::Fps(b) => b.momentum.linear.length() > b.inertia.cutoff.max(1e-6),
            CameraController::Space(b) => b.momentum.has_motion(b.inertia.cutoff),
        }
    }

    /// Current pose (uniform output for render code).
    pub fn pose(&self) -> CameraPose {
        match self {
            CameraController::Houdini(b) => b.pose(),
            CameraController::Quat(b) => b.pose(),
            CameraController::Fps(b) => b.pose(),
            CameraController::Space(b) => b.pose(),
        }
    }

    /// Frame an AABB into view for the active backend.
    pub fn frame_bounds(&mut self, min: Vec3, max: Vec3, width: u32, height: u32, margin: f32) {
        match self {
            CameraController::Houdini(b) => b.rig.frame_bounds(min, max, width, height, margin),
            CameraController::Quat(b) => b.frame_aabb(min, max, width, height, margin),
            CameraController::Fps(b) => b.frame_aabb(min, max, width, height, margin),
            CameraController::Space(b) => b.frame_aabb(min, max, width, height, margin),
        }
    }

    /// Push navigation tuning into the active backend.
    ///
    /// Only the orbit-style backends consume [`CameraNavigationSettings`]; the
    /// fly / space backends store it but drive motion from [`InertiaSettings`]
    /// (the dedicated fly tuning).
    /// Apply live fly-tuning [`InertiaSettings`] to the active backend.
    ///
    /// Houdini (classic orbit) tunes via [`CameraNavigationSettings`] /
    /// [`OrbitRig`], not `InertiaSettings`, so it is a no-op here; the
    /// quaternion-orbit and fly backends store the new inertia for their coast
    /// integration.
    pub fn apply_inertia(&mut self, inertia: InertiaSettings) {
        match self {
            CameraController::Houdini(_) => {}
            CameraController::Quat(b) => b.inertia = inertia,
            CameraController::Fps(b) => b.inertia = inertia,
            CameraController::Space(b) => b.inertia = inertia,
        }
    }

    pub fn apply_navigation(&mut self, nav: CameraNavigationSettings) {
        match self {
            CameraController::Houdini(b) => b.rig.apply_navigation(nav),
            CameraController::Quat(b) => b.navigation = nav,
            CameraController::Fps(_) | CameraController::Space(_) => {}
        }
    }

    /// Replace the active backend's perspective projection (`fov / near / far /
    /// distance clamps`).
    ///
    /// **What:** fans the supplied [`PerspectiveSettings`] into whichever rig is
    /// active — the Houdini wrapper writes the inner [`OrbitRig::projection`], the
    /// other three backends write their own `projection` field.
    ///
    /// **Why:** the worker recomputes an *adaptive* perspective every frame from
    /// the live galaxy AABB (so an expanding "big-bang" cloud never clips the far
    /// plane nor escapes the zoom-out clamp). Pushing it here makes the pose the
    /// backend emits carry the live `far / near`, and — because
    /// [`CameraPose::projection_matrix`] is authoritative for `fov / near / far`
    /// — every downstream `view_proj` / `viewport_ray` (beauty, pick, cull) uses
    /// the SAME projection. It also re-targets the orbit zoom-out clamp and the
    /// Fit-All `frame_bounds`, both of which read `projection.distance_max` /
    /// `distance_min`.
    pub fn set_projection(&mut self, projection: PerspectiveSettings) {
        match self {
            CameraController::Houdini(b) => b.rig.projection = projection,
            CameraController::Quat(b) => b.projection = projection,
            CameraController::Fps(b) => b.projection = projection,
            CameraController::Space(b) => b.projection = projection,
        }
    }

    /// Reset the active backend to `pose`, preserving the backend kind.
    ///
    /// **Why:** when the orchestrator switches navigation style it builds the
    /// new backend [`from_pose`](CameraController::from_pose); but to *re-seed*
    /// the current backend (e.g. a saved bookmark) without changing its kind,
    /// `set_pose` keeps continuity.
    pub fn set_pose(&mut self, pose: CameraPose) {
        match self {
            CameraController::Houdini(b) => {
                let proj = b.rig.projection;
                let nav = b.rig.navigation;
                // Reproject the free pose onto the orbit rig: target = eye +
                // forward * current distance, orientation rebuilt by from_orbit
                // round-trip via OrbitPose look basis.
                let distance = b.rig.pose.distance;
                let target = pose.eye + pose.forward() * distance;
                let new_orbit = OrbitPose {
                    orientation: orbit_orientation_from_pose(&pose),
                    distance,
                    target,
                };
                let mut rig = OrbitRig::from_pose(new_orbit);
                rig.projection = proj;
                rig.navigation = nav;
                b.rig = rig;
            }
            CameraController::Quat(b) => *b = QuatOrbit::from_pose(pose),
            CameraController::Fps(b) => *b = FpsFly::from_pose(pose),
            CameraController::Space(b) => *b = SpaceFlight::from_pose(pose),
        }
    }

    /// Build a controller of the given `kind` seeded from `pose` (view
    /// continuity across a backend switch).
    pub fn from_pose(kind: CameraKind, pose: CameraPose) -> Self {
        match kind {
            CameraKind::Houdini => {
                let distance = 28.0;
                let target = pose.eye + pose.forward() * distance;
                let projection = PerspectiveSettings::default();
                let orbit = OrbitPose {
                    orientation: orbit_orientation_from_pose(&pose),
                    distance: projection.clamp_distance(distance),
                    target,
                };
                CameraController::Houdini(HoudiniOrbit::new(OrbitRig::from_pose(orbit)))
            }
            CameraKind::Quat => CameraController::Quat(QuatOrbit::from_pose(pose)),
            CameraKind::Fps => CameraController::Fps(FpsFly::from_pose(pose)),
            CameraKind::Space => CameraController::Space(SpaceFlight::from_pose(pose)),
        }
    }

    /// Stable string name of the active backend (UI / logging).
    pub fn kind(&self) -> &'static str {
        match self {
            CameraController::Houdini(_) => "houdini",
            CameraController::Quat(_) => "quat",
            CameraController::Fps(_) => "fps",
            CameraController::Space(_) => "space",
        }
    }

    /// True for the free-flight backends (FPS / Spaceship), false for the orbit
    /// backends (Houdini / Quaternion). Used to gate fly-only UI such as the HUD
    /// overlay's `enabled_in_flight` / `enabled_in_orbit` modes.
    pub fn is_fly(&self) -> bool {
        matches!(self, CameraController::Fps(_) | CameraController::Space(_))
    }
}

/// Backend selector for [`CameraController::from_pose`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CameraKind {
    /// [`HoudiniOrbit`].
    #[default]
    Houdini,
    /// [`QuatOrbit`].
    Quat,
    /// [`FpsFly`].
    Fps,
    /// [`SpaceFlight`].
    Space,
}

/// Derive the orbit `orientation` quaternion (used by [`OrbitPose`]) from a free
/// [`CameraPose`].
///
/// **Why:** [`OrbitPose::eye`] is `target + orientation * (+X * distance)`, i.e.
/// the orbit orientation maps **+X** to the eye→target *offset* direction
/// (`-forward`). This finds the rotation that takes `+X` onto `-forward`, which
/// is the inverse of the orbit forward mapping, so a Houdini backend seeded from
/// a free pose looks the same direction.
fn orbit_orientation_from_pose(pose: &CameraPose) -> Quat {
    let offset_dir = (-pose.forward()).normalize_or_zero();
    let offset_dir = if offset_dir.length_squared() < 1e-10 {
        Vec3::X
    } else {
        offset_dir
    };
    Quat::from_rotation_arc(Vec3::X, offset_dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rig::OrbitRig;

    #[test]
    fn houdini_pose_matches_orbit() {
        // The Houdini backend's pose().view_proj must equal the wrapped
        // OrbitRig.view_proj for the same state (view continuity anchor).
        let rig = OrbitRig::from_yaw_pitch(0.5, 0.25, 20.0, Vec3::new(2.0, 1.0, 0.0));
        let backend = HoudiniOrbit::new(rig.clone());
        let a = rig.view_proj(1280, 720);
        let b = backend.pose().view_proj(1280, 720, rig.projection);
        let mut max = 0.0_f32;
        for (x, y) in a.to_cols_array().iter().zip(b.to_cols_array().iter()) {
            max = max.max((x - y).abs());
        }
        assert!(max < 1e-3, "houdini pose mismatch, max delta {max}");
    }

    #[test]
    fn quat_orbit_no_nan_near_poles() {
        let mut q = QuatOrbit::default();
        // Pitch far past where an euler turntable would gimbal-lock.
        for _ in 0..200 {
            q.apply_intent(CameraIntent::Orbit { yaw_delta: 0.0, pitch_delta: 0.1 }, ViewportSize::new(1, 1));
            let p = q.pose();
            assert!(p.eye.is_finite(), "eye went non-finite: {:?}", p.eye);
            assert!(p.orientation.is_finite(), "orientation NaN");
            let f = p.forward();
            assert!(f.is_finite() && (f.length() - 1.0).abs() < 1e-3, "forward bad: {f:?}");
        }
    }

    #[test]
    fn space_thrust_and_roll_move_then_damp() {
        let mut s = SpaceFlight::default();
        let eye0 = s.eye;
        let orient0 = s.orientation;
        s.apply_intent(CameraIntent::Thrust { forward: 1.0, right: 0.0, up: 0.0 }, ViewportSize::new(1, 1));
        // Roll is a HELD axis (Q/E), integrated with dt like thrust — it stays
        // applied every frame until released, so set-once + coast turns the ship.
        s.apply_intent(CameraIntent::Roll { d: 1.0 }, ViewportSize::new(1, 1));
        // Coast a few frames (thrust + roll integrate with dt inside update_dynamics).
        let mut moved = false;
        for _ in 0..5 {
            moved |= s.update_dynamics(1.0 / 60.0);
        }
        assert!(moved, "spaceflight should report motion while coasting");
        assert!((s.eye - eye0).length() > 1e-3, "thrust did not move the eye");
        assert!(orient0.angle_between(s.orientation) > 1e-4, "roll did not turn the ship");
        // Release thrust AND roll (gather emits zero on key-up) so momentum damps.
        s.apply_intent(CameraIntent::Thrust { forward: 0.0, right: 0.0, up: 0.0 }, ViewportSize::new(1, 1));
        s.apply_intent(CameraIntent::Roll { d: 0.0 }, ViewportSize::new(1, 1));
        // Damp to rest.
        let mut frames = 0;
        while s.update_dynamics(1.0 / 60.0) {
            frames += 1;
            if frames > 100_000 {
                panic!("spaceflight never damped to rest");
            }
        }
        assert!(!s.momentum.has_motion(s.inertia.cutoff), "momentum should be at rest");
    }

    #[test]
    fn space_mouse_steer_turns_then_stops() {
        let mut s = SpaceFlight::default();
        let o0 = s.orientation;
        // Held deflection (yaw right) turns the ship continuously over dt.
        s.apply_intent(
            CameraIntent::LookRate { yaw: 1.0, pitch: 0.0 },
            ViewportSize::new(1, 1),
        );
        for _ in 0..5 {
            s.update_dynamics(1.0 / 60.0);
        }
        assert!(
            o0.angle_between(s.orientation) > 1e-3,
            "deflection steer did not turn the ship"
        );
        // Recentre (zero axis) → direct steering stops immediately (no coast): the
        // orientation must not drift further once the held axis is zero.
        let o1 = s.orientation;
        s.apply_intent(
            CameraIntent::LookRate { yaw: 0.0, pitch: 0.0 },
            ViewportSize::new(1, 1),
        );
        s.update_dynamics(1.0 / 60.0);
        assert!(
            o1.angle_between(s.orientation) < 1e-6,
            "steering kept turning after the cursor recentred"
        );
    }

    #[test]
    fn set_pose_round_trips() {
        let pose = CameraPose::from_eye_forward_up(
            Vec3::new(5.0, 3.0, -2.0),
            Vec3::new(0.0, -0.2, -1.0),
            Vec3::Y,
            45.0_f32.to_radians(),
            0.1,
            500.0,
        );
        let mut ctrl = CameraController::Space(SpaceFlight::default());
        ctrl.set_pose(pose);
        let got = ctrl.pose();
        assert!((got.eye - pose.eye).length() < 1e-4, "eye not preserved");
        assert!(got.orientation.angle_between(pose.orientation) < 1e-3, "orientation not preserved");

        // Fps round-trip preserves eye and (roll-free) heading.
        let mut fps = CameraController::Fps(FpsFly::default());
        fps.set_pose(pose);
        let g2 = fps.pose();
        assert!((g2.eye - pose.eye).length() < 1e-4, "fps eye not preserved");
        assert!(g2.forward().dot(pose.forward()) > 0.999, "fps forward not preserved");
    }
}
