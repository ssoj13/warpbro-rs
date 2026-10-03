//! Frame-domain camera orbit integration. Seeking and export evaluate the same trajectory.
//! Speed is degrees/second; integration is split at keys, including hold discontinuities.
use curves::CurveKind;
use playa_engine::entities::anim::Channel;

/// Integral in degree-frames; the document divides by its FPS exactly once.
pub(crate) fn integrate(channel: &Channel, from: f64, to: f64) -> f64 {
    if to < from {
        return -integrate(channel, to, from);
    }
    if from == to || channel.is_empty() {
        return 0.0;
    }
    let keys = channel.keys();
    let mut total = 0.0;
    let first = &keys[0];
    if from < first.frame {
        total += (to.min(first.frame) - from).max(0.0) * f64::from(first.value);
    }
    for pair in keys.windows(2) {
        let start = from.max(pair[0].frame);
        let end = to.min(pair[1].frame);
        if end <= start {
            continue;
        }
        let sample = |t| f64::from(channel.sample_at(t));
        total += match pair[0].interp {
            CurveKind::Step => (end - start) * f64::from(pair[0].value),
            CurveKind::Linear => (end - start) * (sample(start) + sample(end)) * 0.5,
            CurveKind::Bezier => {
                let whole = simpson(&sample, start, end);
                adaptive(&sample, start, end, whole, 1.0e-4, 10)
            }
            // Smooth, Catmull-Rom, Hermite and monotone segments are cubic polynomials.
            _ => simpson(&sample, start, end),
        };
    }
    if let Some(last) = keys.last() {
        if to > last.frame {
            total += (to - from.max(last.frame)).max(0.0) * f64::from(last.value);
        }
    }
    total
}

fn simpson(f: &impl Fn(f64) -> f64, a: f64, b: f64) -> f64 {
    (b - a) * (f(a) + 4.0 * f((a + b) * 0.5) + f(b)) / 6.0
}

fn adaptive(f: &impl Fn(f64) -> f64, a: f64, b: f64, whole: f64, tolerance: f64, depth: u8) -> f64 {
    let mid = (a + b) * 0.5;
    let left = simpson(f, a, mid);
    let right = simpson(f, mid, b);
    let error = left + right - whole;
    if depth == 0 || error.abs() <= 15.0 * tolerance {
        left + right + error / 15.0
    } else {
        adaptive(f, a, mid, left, tolerance * 0.5, depth - 1)
            + adaptive(f, mid, b, right, tolerance * 0.5, depth - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use playa_engine::entities::anim::Keyframe;

    #[test]
    fn hold_boundaries_and_reversed_subframes_integrate_without_jumps() {
        let mut channel = Channel::new();
        channel.upsert_key(Keyframe::with_interp(0.0, 12.0, CurveKind::Step));
        channel.upsert_key(Keyframe::with_interp(24.0, -6.0, CurveKind::Step));
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
            CurveKind::Linear,
            CurveKind::Smooth,
            CurveKind::Hermite,
            CurveKind::Bezier,
        ] {
            let mut channel = Channel::new();
            channel.upsert_key(Keyframe::with_interp(0.0, 0.0, kind));
            channel.upsert_key(Keyframe::new(24.0, 90.0));
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
