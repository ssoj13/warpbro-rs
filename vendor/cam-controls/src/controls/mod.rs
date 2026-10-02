//! Input → camera intent layer (0.0.1).
//!
//! Platform adapters (`cam-controls-winit`, `cam-controls-egui`) translate raw events
//! into [`CameraIntent`] and call [`Controls::apply_intent`].

use cam_viewport::{ViewportPoint, ViewportSize};
use glam::Vec3;

use crate::rig::OrbitRig;
use crate::settings::CameraNavigationSettings;

/// High-level camera manipulation request (host-agnostic).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CameraIntent {
    Orbit { yaw_delta: f32, pitch_delta: f32 },
    /// Coast after pointer release (integrates via [`OrbitRig::update_inertia`]).
    OrbitInertia { yaw_delta: f32, pitch_delta: f32 },
    Pan { dx_px: f32, dy_px: f32 },
    PanInertia { dx_px: f32, dy_px: f32 },
    Zoom { factor: f32 },
    ZoomInertia { delta: f32 },
    ZoomAtCursor {
        cursor_x: f32,
        cursor_y: f32,
        factor: f32,
    },
    FrameBounds {
        min: Vec3,
        max: Vec3,
        margin: f32,
    },
    /// Mouse-look delta for the fly / spaceflight backends (radians applied as
    /// yaw/pitch). Ignored by the orbit rig (use [`CameraIntent::Orbit`] there).
    Look { dyaw: f32, dpitch: f32 },
    /// Held mouse-steer RATE axes (yaw/pitch ∈ [−1,1], deflection of the cursor
    /// from the viewport centre) for the “космосим” steering mode. Integrated
    /// with real dt in the fly backends' `update_dynamics`, so the ship turns
    /// continuously while the cursor is held off-centre. No-op for the orbit rig.
    LookRate { yaw: f32, pitch: f32 },
    /// Roll delta (radians) for the spaceflight backend. No-op for orbit / FPS.
    Roll { d: f32 },
    /// Body-relative thrust for the fly / spaceflight backends: `forward` /
    /// `right` / `up` are signed thrust magnitudes (e.g. WASD axes). No-op for
    /// the orbit rig.
    Thrust { forward: f32, right: f32, up: f32 },
    /// Toggle the speed-boost modifier for the spaceflight backend. No-op for
    /// orbit / FPS rigs.
    Boost(bool),
}

/// Who owns pointer/keyboard focus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ControlLock {
    #[default]
    Unlocked,
    CameraOnly,
    HostUi,
    Fps,
}

/// Stateful controller: navigation tuning + input lock.
#[derive(Debug, Clone)]
pub struct Controls {
    pub lock: ControlLock,
    pub navigation: CameraNavigationSettings,
}

impl Default for Controls {
    fn default() -> Self {
        Self {
            lock: ControlLock::default(),
            navigation: CameraNavigationSettings::default(),
        }
    }
}

impl Controls {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn enter_fps(&mut self) {
        self.lock = ControlLock::Fps;
    }

    pub fn exit_fps(&mut self) {
        self.lock = ControlLock::Unlocked;
    }

    /// Apply one intent to an orbit rig (host-agnostic).
    pub fn apply_intent(
        &self,
        intent: CameraIntent,
        rig: &mut OrbitRig,
        viewport: ViewportSize,
    ) {
        match intent {
            CameraIntent::Orbit {
                yaw_delta,
                pitch_delta,
            } => rig.orbit(yaw_delta, pitch_delta),
            CameraIntent::OrbitInertia {
                yaw_delta,
                pitch_delta,
            } => rig.orbit_inertia(yaw_delta, pitch_delta),
            CameraIntent::Pan { dx_px, dy_px } => rig.pan(dx_px, dy_px),
            CameraIntent::PanInertia { dx_px, dy_px } => rig.pan_inertia(dx_px, dy_px),
            CameraIntent::Zoom { factor } => rig.zoom(factor),
            CameraIntent::ZoomInertia { delta } => rig.zoom_inertia(delta),
            CameraIntent::ZoomAtCursor {
                cursor_x,
                cursor_y,
                factor,
            } => {
                let point = ViewportPoint {
                    x: cursor_x,
                    y: cursor_y,
                };
                rig.zoom_at_cursor(point, viewport, factor);
            }
            CameraIntent::FrameBounds { min, max, margin } => {
                rig.frame_bounds(min, max, viewport.width, viewport.height, margin);
            }
            // The following intents target the fly / spaceflight backends in
            // `CameraController`; the orbit rig has no analog, so they are
            // deliberate no-ops here (kept exhaustive & compiling).
            CameraIntent::Look { .. }
            | CameraIntent::LookRate { .. }
            | CameraIntent::Roll { .. }
            | CameraIntent::Thrust { .. }
            | CameraIntent::Boost(_) => {}
        }
    }

    pub fn apply_intents(
        &self,
        intents: impl IntoIterator<Item = CameraIntent>,
        rig: &mut OrbitRig,
        viewport: ViewportSize,
    ) {
        for intent in intents {
            self.apply_intent(intent, rig, viewport);
        }
    }
}
