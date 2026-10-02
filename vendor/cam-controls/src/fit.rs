//! Frame / zoom-to-fit math (AABB in world space).

use glam::Vec3;

use crate::projection::PerspectiveSettings;

/// Union AABB → center and bounding sphere radius.
pub fn aabb_center_radius(min: Vec3, max: Vec3) -> (Vec3, f32) {
    let min_v = min.min(max);
    let max_v = min.max(max);
    let center = (min_v + max_v) * 0.5;
    let extent = max_v - min_v;
    let mut radius = extent.length() * 0.5;
    if radius < 1e-4 {
        radius = 1.0;
    }
    (center, radius)
}

/// Distance from pivot so a sphere of `radius` fits in a perspective view.
pub fn distance_to_frame_sphere(
    radius: f32,
    aspect: f32,
    fov_y: f32,
    margin: f32,
    settings: PerspectiveSettings,
) -> f32 {
    let half_tan_y = (fov_y * 0.5).tan();
    let half_tan_x = half_tan_y * aspect;
    let dist_y = radius / half_tan_y;
    let dist_x = radius / half_tan_x;
    settings.clamp_distance(dist_x.max(dist_y) * margin.max(1.0))
}
