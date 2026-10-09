//! Device-safe convergence policies; kernel integration follows paired benchmark capture.
//!
//! PBRT 3ed uses luminance and delayed roulette; 4ed removes refractive radiance scaling
//! through etaScale. Our Standard Surface BTDF uses adjoint flux weights and omits that
//! scaling (standard-surface-bsdf/src/transmission.rs), so its eta compensation is one.

pub const ROULETTE_START_DEPTH: u32 = 3;
pub const MIN_SURVIVAL: f32 = 0.05;

#[inline(always)]
pub fn luminance(rgb: [f32; 3]) -> f32 {
    let y = standard_surface_bsdf::consts::SS_LUMA;
    rgb[0] * y[0] + rgb[1] * y[1] + rgb[2] * y[2]
}

/// Probability for a finite path throughput. Survivors must divide by this probability.
/// `eta_compensation` removes radiance-mode interface scaling, if the BSDF includes it.
#[inline(always)]
pub fn roulette_probability(throughput: [f32; 3], eta_compensation: f32, depth: u32) -> f32 {
    if depth < ROULETTE_START_DEPTH {
        return 1.0;
    }
    if throughput == [0.0; 3] {
        return 0.0;
    }
    // Signed working-space RGB can have nonpositive Y without being black.
    // A zero survival probability there would bias its nonzero RGB components.
    let energy = (luminance(throughput) * eta_compensation).max(0.0);
    energy.clamp(MIN_SURVIVAL, 1.0)
}

/// Biased cap of a single indirect contribution, preserving its RGB ratios.
/// Zero disables the cap. Camera-visible emission/background/direct lighting stay exact.
/// Returns discarded scene-linear AP1 luminance for energy-loss reporting.
#[inline(always)]
pub fn clamp_indirect(contribution: [f32; 3], maximum: f32, depth: u32) -> ([f32; 3], f32) {
    if maximum <= 0.0 || depth == 0 {
        return (contribution, 0.0);
    }
    let energy = luminance(contribution);
    if energy <= maximum {
        return (contribution, 0.0);
    }
    let scale = maximum / energy;
    (
        [
            contribution[0] * scale,
            contribution[1] * scale,
            contribution[2] * scale,
        ],
        energy - maximum,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roulette_expectation_preserves_each_saturated_rgb_channel() {
        // Enumerate a uniform stratified roulette variable. This tests the estimator's
        // expectation, including a low-luminance saturated channel, rather than just p.
        for rgb in [
            [0.001, 0.0, 0.0],
            [0.0, 0.002, 0.0],
            [0.0, 0.0, 0.5],
            [0.3; 3],
            [2.0; 3],
            [1.0, -1.0, 0.0],
        ] {
            let p = roulette_probability(rgb, 1.0, 3);
            let trials = 100_000;
            let survivors = (0..trials)
                .filter(|i| (*i as f64 + 0.5) / f64::from(trials) < f64::from(p))
                .count();
            for channel in rgb {
                let mean = f64::from(channel / p) * survivors as f64 / f64::from(trials);
                assert!(
                    (mean - f64::from(channel)).abs() <= f64::from(channel).abs() * 0.0002 + 1e-10
                );
            }
        }
        assert_eq!(roulette_probability([0.0; 3], 1.0, 3), 0.0);
        assert_eq!(roulette_probability([0.001; 3], 1.0, 2), 1.0);
    }

    #[test]
    fn eta_compensation_removes_only_radiance_transport_scaling() {
        let original = [0.4, 0.2, 0.1];
        let eta_squared = 1.5_f32 * 1.5;
        let radiance_weights = original.map(|v| v / eta_squared);
        let flux = roulette_probability(original, 1.0, 3);
        let compensated = roulette_probability(radiance_weights, eta_squared, 3);
        assert!((flux - compensated).abs() < 1e-7);
        assert!(roulette_probability(radiance_weights, 1.0, 3) < flux);
    }

    #[test]
    fn clamp_reports_exact_luminance_loss_and_preserves_direct_light_and_hue() {
        let original = [100.0, 20.0, 5.0];
        assert_eq!(clamp_indirect(original, 0.0, 3), (original, 0.0));
        assert_eq!(clamp_indirect(original, 1.0, 0), (original, 0.0));
        let (capped, loss) = clamp_indirect(original, 2.0, 1);
        assert!((luminance(capped) - 2.0).abs() < 1e-6);
        assert!((luminance(capped) + loss - luminance(original)).abs() < 1e-5);
        assert!((capped[0] / capped[1] - original[0] / original[1]).abs() < 1e-6);
        assert_eq!(clamp_indirect([0.1; 3], 2.0, 1), ([0.1; 3], 0.0));
    }
}
