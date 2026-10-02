//! Perspective projection settings shared by rigs.

/// Perspective settings (WebGPU RH). The matrix builders render with **reverse-Z**
/// (near→NDC depth 1, far→0) by passing `near`/`far` swapped to `perspective_rh`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PerspectiveSettings {
    /// Vertical field of view in radians.
    pub fov_y: f32,
    pub near: f32,
    pub far: f32,
    pub distance_min: f32,
    pub distance_max: f32,
}

impl Default for PerspectiveSettings {
    fn default() -> Self {
        Self {
            fov_y: 45.0_f32.to_radians(),
            near: 0.1,
            // Wide range so a large spread-out galaxy can be framed (zoom out
            // to 600) and inspected up close (zoom in to 1). `far` covers the
            // farthest node at max distance.
            far: 1500.0,
            distance_min: 1.0,
            distance_max: 600.0,
        }
    }
}

impl PerspectiveSettings {
    pub fn clamp_distance(self, distance: f32) -> f32 {
        distance.clamp(self.distance_min, self.distance_max)
    }
}
