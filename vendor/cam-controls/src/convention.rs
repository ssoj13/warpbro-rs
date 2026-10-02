//! Coordinate conventions (documented SSOT for all rigs).

use glam::Vec3;

/// World up axis (right-handed, Y-up).
pub const WORLD_UP: Vec3 = Vec3::Y;

/// Default orbit offset axis before orientation is applied (+X in pivot space).
pub const DEFAULT_OFFSET_AXIS: Vec3 = Vec3::X;
