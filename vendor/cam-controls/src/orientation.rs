//! Orientation policies for orbit rigs (storage is always [`glam::Quat`]).

use glam::{Quat, Vec3};

use crate::convention::{DEFAULT_OFFSET_AXIS, WORLD_UP};

/// Max |pitch| in radians for turntable / quaternion policies.
pub const DEFAULT_PITCH_LIMIT: f32 = 1.2;

/// How mouse orbit deltas are applied to a unit quaternion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OrientationMode {
    /// Yaw around world +Y, then pitch around camera-local Z (legacy; can gimbal near poles).
    Quaternion,
    /// Turntable: yaw world Y, pitch around camera right (no gimbal at horizon).
    #[default]
    HorizontalPlaneEuler,
    /// Explicit yaw-then-pitch euler rebuild (documented order: Y then Z in offset space).
    Euler,
}

/// Apply orbit mouse deltas to `orientation`.
pub fn apply_orbit_delta(
    mode: OrientationMode,
    orientation: &mut Quat,
    yaw_delta: f32,
    pitch_delta: f32,
    pitch_limit: f32,
) {
    match mode {
        OrientationMode::Quaternion | OrientationMode::HorizontalPlaneEuler => {
            if yaw_delta != 0.0 {
                *orientation = (Quat::from_rotation_y(yaw_delta) * *orientation).normalize();
            }
            if pitch_delta != 0.0 {
                let pitch_axis = match mode {
                    OrientationMode::HorizontalPlaneEuler => {
                        let forward = (*orientation * DEFAULT_OFFSET_AXIS).normalize_or_zero();
                        forward.cross(WORLD_UP).normalize_or_zero()
                    }
                    OrientationMode::Quaternion => (*orientation * Vec3::Z).normalize_or_zero(),
                    OrientationMode::Euler => unreachable!(),
                };
                if pitch_axis.length_squared() > 1e-8 {
                    *orientation = (Quat::from_axis_angle(pitch_axis, pitch_delta) * *orientation)
                        .normalize();
                }
            }
            clamp_pitch_offset(orientation, pitch_limit);
        }
        OrientationMode::Euler => {
            let (yaw, pitch) = yaw_pitch_from_orientation(*orientation);
            let yaw = yaw + yaw_delta;
            let pitch = (pitch + pitch_delta).clamp(-pitch_limit, pitch_limit);
            *orientation = orientation_from_yaw_pitch(yaw, pitch);
        }
    }
}

/// Turntable orientation: yaw about world +Y, pitch about camera right after yaw.
pub fn orientation_from_yaw_pitch(yaw: f32, pitch: f32) -> Quat {
    let yaw_q = Quat::from_rotation_y(yaw);
    let offset = yaw_q * DEFAULT_OFFSET_AXIS;
    let mut right = offset.cross(WORLD_UP);
    if right.length_squared() < 1e-8 {
        right = Vec3::Z;
    } else {
        right = right.normalize();
    }
    (Quat::from_axis_angle(right, pitch) * yaw_q).normalize()
}

pub fn yaw_pitch_from_orientation(orientation: Quat) -> (f32, f32) {
    let dir = orientation * DEFAULT_OFFSET_AXIS;
    let horiz = Vec3::new(dir.x, 0.0, dir.z).length();
    let pitch = dir.y.atan2(horiz.max(1e-6));
    let yaw = (-dir.z).atan2(dir.x);
    (yaw, pitch)
}

fn clamp_pitch_offset(orientation: &mut Quat, pitch_limit: f32) {
    let (yaw, pitch) = yaw_pitch_from_orientation(*orientation);
    let clamped = pitch.clamp(-pitch_limit, pitch_limit);
    if (pitch - clamped).abs() < 1e-5 {
        return;
    }
    *orientation = orientation_from_yaw_pitch(yaw, clamped);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn euler_round_trip() {
        let q = orientation_from_yaw_pitch(0.5, 0.2);
        let (y, p) = yaw_pitch_from_orientation(q);
        let q2 = orientation_from_yaw_pitch(y, p);
        let d = (q - q2).length();
        assert!(d < 1e-4, "recomposed quat delta {d}");
    }
}
