//! Frame-domain camera orbit integration. Seeking and export evaluate the same trajectory.
//! Speed is degrees/second; integration is split at keys, including hold discontinuities.
use playa_engine::entities::anim::Channel;

/// Integral in degree-frames; the document divides by its FPS exactly once. Exact for every
/// tangent kind (`curves::Track::integrate`): holds, curved segments and the constant ends.
pub(crate) fn integrate(channel: &Channel, from: f64, to: f64) -> f64 {
    if channel.is_empty() {
        return 0.0;
    }
    channel.integrate(from, to)
}

#[cfg(test)]
mod tests {
    use super::*;
    use curves::Tan;
    use playa_engine::entities::anim::Keyframe;

    #[test]
    fn hold_boundaries_and_reversed_subframes_integrate_without_jumps() {
        let mut channel = Channel::new();
        channel.upsert_key(Keyframe::with_tan(0.0, 12.0, Tan::Constant));
        channel.upsert_key(Keyframe::with_tan(24.0, -6.0, Tan::Constant));
        channel.upsert_key(Keyframe::new(48.0, 3.0));
        assert_eq!(integrate(&channel, 0.0, 24.0), 288.0);
        assert_eq!(integrate(&channel, 0.0, 48.0), 144.0);
        assert_eq!(integrate(&channel, 24.0, 24.5), -3.0);
        assert_eq!(integrate(&channel, 24.5, 24.0), 3.0);
        assert_eq!(integrate(&channel, -2.0, 50.0), 174.0);
    }

    #[test]
    fn speed_ramps_and_symmetric_eases_have_the_expected_area() {
        for kind in [
            Tan::Linear,
            Tan::Smooth,
            Tan::Flat,
            Tan::CatmullRom,
        ] {
            let mut channel = Channel::new();
            channel.upsert_key(Keyframe::with_tan(0.0, 0.0, kind));
            // Both ends take the kind: a ramp is symmetric only when both sides of its segment match.
            channel.upsert_key(Keyframe::with_tan(24.0, 90.0, kind));
            assert!((integrate(&channel, 0.0, 24.0) / 24.0 - 45.0).abs() < 0.0001);
            assert!(
                (integrate(&channel, 0.0, 12.0) + integrate(&channel, 12.0, 24.0)
                    - integrate(&channel, 0.0, 24.0))
                .abs()
                    < 0.001
            );
        }
    }
}
