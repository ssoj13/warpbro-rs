//! Canvas transfer shared with exr-view and Playa.
#[cfg(test)]
pub use egui_display::transfer::eotf;
pub use egui_display::transfer::oetf;
#[allow(dead_code)]
pub fn dial(v: f32, gamma: f32) -> f32 {
    v.abs().powf(1.0 / gamma.max(1e-3)).copysign(v)
}
#[cfg(test)]
pub fn code8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}
