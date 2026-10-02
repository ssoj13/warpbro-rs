//! Orbit navigation tuning: zoom mode, sensitivities, inertia.

use serde::{Deserialize, Serialize};

/// How wheel / dolly input maps to a multiplicative distance factor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ZoomMode {
    /// `factor ≈ 1 - scroll * linear_rate` (clamped).
    Linear,
    /// `factor ≈ exp(-scroll * exponential_rate)` (clamped). Default — stable relative zoom.
    Exponential,
}

impl Default for ZoomMode {
    fn default() -> Self {
        Self::Exponential
    }
}

/// Single source of truth for tree-style orbit navigation (UI gather + GPU worker pan scale).
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default)] // missing fields fall back to `Default`
pub struct CameraNavigationSettings {
    pub zoom_mode: ZoomMode,
    /// When `true`, the mouse wheel zooms toward the cursor (cursor-anchored).
    /// When `false` (default), it zooms straight toward the camera target /
    /// centre of aim — the steadier, less disorienting behaviour.
    #[serde(default)]
    pub zoom_to_cursor: bool,
    pub orbit_sensitivity: f32,
    /// Per pixel of egui `smooth_scroll_delta.y` when [`ZoomMode::Linear`].
    pub zoom_scroll_linear: f32,
    /// Per pixel of scroll when [`ZoomMode::Exponential`]: `exp(-scroll * rate)`.
    pub zoom_scroll_exponential: f32,
    /// Pan scale: `target += axis * distance * pan_sensitivity`.
    pub pan_sensitivity: f32,
    /// RMB dolly; uses the same [`ZoomMode`] as the wheel.
    pub dolly_sensitivity: f32,
    /// Multiplier on the last orbit frame when LMB is released (inertia seed).
    pub orbit_release_inertia_scale: f32,
    /// Exponential friction for [`crate::rig::OrbitRig::update_inertia`].
    pub inertia_friction: f32,
    /// Invert the horizontal (yaw) orbit-drag axis. `#[serde(default)]` so older
    /// presets load with inversion off (the historical behaviour).
    #[serde(default)]
    pub invert_orbit_x: bool,
    /// Invert the vertical (pitch) orbit-drag axis.
    #[serde(default)]
    pub invert_orbit_y: bool,
}

impl Default for CameraNavigationSettings {
    fn default() -> Self {
        Self {
            zoom_mode: ZoomMode::Linear,
            zoom_to_cursor: false,
            orbit_sensitivity: 0.008,
            zoom_scroll_linear: 0.0025,
            zoom_scroll_exponential: 0.0035,
            pan_sensitivity: 0.0025,
            dolly_sensitivity: 0.012,
            orbit_release_inertia_scale: 0.20,
            inertia_friction: 0.8,
            invert_orbit_x: false,
            invert_orbit_y: false,
        }
    }
}

impl CameraNavigationSettings {
    // Per-FRAME zoom cap. egui smears one wheel notch across several frames, so
    // a loose cap (e.g. 0.85) compounds (0.85^6 ≈ 0.38 → ~62% of the distance
    // gone in one notch — the "whole cloud in one click" jump). A tight per-frame
    // cap keeps each notch smooth regardless of how many frames egui spreads it
    // over.
    const ZOOM_CLAMP_MIN: f32 = 0.97;
    const ZOOM_CLAMP_MAX: f32 = 1.031;

    /// Multiplicative distance factor from one frame of vertical scroll (egui units).
    pub fn scroll_zoom_factor(&self, scroll_y: f32) -> f32 {
        if scroll_y.abs() < f32::EPSILON {
            return 1.0;
        }
        let factor = match self.zoom_mode {
            ZoomMode::Linear => 1.0 - scroll_y * self.zoom_scroll_linear,
            ZoomMode::Exponential => (-scroll_y * self.zoom_scroll_exponential).exp(),
        };
        factor.clamp(Self::ZOOM_CLAMP_MIN, Self::ZOOM_CLAMP_MAX)
    }

    /// Multiplicative distance factor from RMB vertical drag (pixels this frame).
    pub fn dolly_zoom_factor(&self, drag_dy_px: f32) -> f32 {
        if drag_dy_px.abs() < f32::EPSILON {
            return 1.0;
        }
        let factor = match self.zoom_mode {
            ZoomMode::Linear => 1.0 - drag_dy_px * self.dolly_sensitivity,
            // Slightly gentler exponent than wheel so drag-dolly feels similar.
            ZoomMode::Exponential => (-drag_dy_px * self.dolly_sensitivity * 0.15).exp(),
        };
        factor.clamp(Self::ZOOM_CLAMP_MIN, Self::ZOOM_CLAMP_MAX)
    }
}

/// Per-backend tuning for the fly / spaceflight rigs (sensitivity + damping).
///
/// **What:** sensitivities convert raw input (pixels of mouse drag, normalized
/// thrust axis values) into angular / linear velocity, and the damping rates
/// feed [`crate::dynamics::FlyInertia::integrate`] so momentum coasts and decays
/// frame-rate-independently. `boost_multiplier` is the speed factor applied
/// while a `Boost` modifier is held (spaceflight).
///
/// **Why:** the UI will later expose inertia / damping / sensitivity sliders;
/// keeping them in one serde struct gives a single persisted source of truth.
/// Added additively — the orbit rig keeps using [`CameraNavigationSettings`].
///
/// **Where:** [`crate::controller::FpsFly`], [`crate::controller::SpaceFlight`],
/// and (for the angular-only fields) [`crate::controller::QuatOrbit`].
/// Mouse steering model for the fly backends (`FpsFly` / `SpaceFlight`).
///
/// `Relative` is the classic FPS look (per-frame pointer motion turns the view
/// 1:1). `Deflection` is "космосим" rate steering: the cursor's offset from the
/// viewport centre sets a continuous yaw/pitch turn rate, so holding the cursor
/// off-centre keeps the ship turning. Only the SpaceFlight backend honours
/// `Deflection`; FpsFly always uses `Relative`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum MouseSteer {
    /// Per-frame pointer motion turns the view 1:1 (classic FPS look).
    #[default]
    Relative,
    /// Cursor offset from the viewport centre → continuous yaw/pitch turn rate.
    Deflection,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)] // missing fields fall back to `Default`
pub struct InertiaSettings {
    /// Radians of yaw/pitch per pixel of mouse-look drag.
    pub look_sensitivity: f32,
    /// Angular acceleration (rad/s²) per unit of held roll axis (Q/E); the
    /// spaceflight backend integrates it as `accel * dt` into angular momentum,
    /// so roll is frame-rate-independent. Terminal roll rate ≈ `roll_sensitivity
    /// / angular_damping`.
    pub roll_sensitivity: f32,
    /// Thrust acceleration (world units/s²) per unit of held thrust axis; the
    /// backends integrate it as `accel * dt` in `update_dynamics`, so travel is
    /// frame-rate-independent. Terminal cruise speed ≈ `thrust_sensitivity /
    /// linear_damping`.
    pub thrust_sensitivity: f32,
    /// Exponential damping rate (1/s) for linear momentum.
    pub linear_damping: f32,
    /// Exponential damping rate (1/s) for angular momentum.
    pub angular_damping: f32,
    /// Velocity magnitude below which momentum snaps to rest.
    pub cutoff: f32,
    /// Speed multiplier applied while `Boost` is active.
    pub boost_multiplier: f32,
    /// Invert the horizontal (yaw) mouse-look axis for the fly backends. Defaults
    /// ON — most users expect cursor-right to turn the ship's nose right in
    /// deflection steering. Filled from the container `#[serde(default)]`.
    pub invert_look_x: bool,
    /// Invert the vertical (pitch) mouse-look axis for the fly backends.
    #[serde(default)]
    pub invert_look_y: bool,
    /// Mouse steering model (see [`MouseSteer`]). `Deflection` engages only on the
    /// SpaceFlight backend; FpsFly always steers `Relative`. Filled from the
    /// container `#[serde(default)]`, so old presets load with the struct default.
    pub mouse_steer: MouseSteer,
    /// Turn rate (rad/s) at FULL cursor deflection in `MouseSteer::Deflection`.
    pub steer_rate: f32,
    /// Dead-zone as a fraction (0..1) of the half-viewport: deflection magnitudes
    /// below this map to zero turn, so a near-centred cursor holds heading steady.
    pub steer_deadzone: f32,
    /// Draw the steering reticle overlay (neutral cross + deflection dot) in the
    /// viewport while steering in `MouseSteer::Deflection` — on-screen feedback of
    /// the current deflection (essential when the cursor is hidden via capture).
    pub steer_reticle: bool,
    /// Steering neutral-point ORIGIN: `true` = fixed at the viewport CENTRE
    /// (absolute — the reticle sits in the middle of the screen); `false` =
    /// captured at the cursor position when fly engages (relative — no snap on
    /// entry, but the neutral point can sit anywhere). `MouseSteer::Deflection` only.
    pub steer_center_screen: bool,
    /// Steer decay (relative origin only): when set, the steering anchor drifts
    /// toward the cursor, so holding the mouse still eases the turn back to neutral
    /// (the ship turns only while the mouse is *moving*). When clear, deflection
    /// persists — hold off the neutral point to keep turning.
    pub steer_decay: bool,
    /// Time constant (seconds) of the steer-decay drift — larger = slower return to
    /// neutral. Only used when `steer_decay` is set.
    pub steer_decay_secs: f32,
    /// Hide the OS cursor while flying (Unity/Unreal-style capture). Default OFF.
    /// NOTE: hide ONLY — no `CursorGrab`, because `CursorGrab::Confined` froze
    /// pointer input entirely in this winit build. Deflection reads the cursor's
    /// live absolute position regardless.
    pub steer_capture_cursor: bool,
}

impl Default for InertiaSettings {
    fn default() -> Self {
        Self {
            look_sensitivity: 0.005,
            roll_sensitivity: 6.0,
            thrust_sensitivity: 40.0,
            linear_damping: 4.0,
            angular_damping: 8.0,
            cutoff: 0.0001,
            boost_multiplier: 4.0,
            invert_look_x: true,
            invert_look_y: false,
            mouse_steer: MouseSteer::Deflection,
            steer_rate: 2.0,
            steer_deadzone: 0.06,
            steer_reticle: true,
            steer_center_screen: true,
            steer_decay: false,
            steer_decay_secs: 0.4,
            steer_capture_cursor: false,
        }
    }
}

impl InertiaSettings {
    /// Tuning preset for the FPS fly rig: snappier angular damping, no roll,
    /// modest boost.
    pub fn fps() -> Self {
        Self {
            angular_damping: 14.0,
            boost_multiplier: 2.5,
            ..Self::default()
        }
    }

    /// Tuning preset for the spaceflight rig: low damping so momentum drifts
    /// (the "spaceship" coast) and a strong boost.
    pub fn space() -> Self {
        Self {
            linear_damping: 1.2,
            angular_damping: 2.0,
            boost_multiplier: 6.0,
            ..Self::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_zoom_mode_is_linear() {
        assert_eq!(CameraNavigationSettings::default().zoom_mode, ZoomMode::Linear);
    }

    #[test]
    fn exponential_scroll_zoom_moves_distance() {
        let s = CameraNavigationSettings::default();
        let f = s.scroll_zoom_factor(10.0);
        assert!(f < 1.0 && f >= CameraNavigationSettings::ZOOM_CLAMP_MIN);
    }
}
