//! Palettes: the built-in colour schemes of ofx-rs `ofx-gen/src/palette.rs` (copied; the
//! validated host-curve API dropped), sampled into the LUT the kernels read.


use crate::params::PALETTE_SAMPLES;

/// Built-in color schemes; custom host gradients use `PaletteLut::from_samples`.
/// The first four entries retain their existing public order.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum PaletteScheme {
    Classic,
    Fire,
    Ice,
    Mono,
    Sunset,
    Aurora,
    Ocean,
    Ember,
    Amethyst,
    Verdant,
    Copper,
    Neon,
    RoseGold,
    Twilight,
}


#[derive(Clone, Copy)]
struct Gradient {
    stops: [[f32; 3]; 5],
    interior: [f32; 3],
}

const GRADIENTS: [Gradient; 10] = [
    Gradient {
        stops: [
            [0.012, 0.009, 0.08],
            [0.30, 0.04, 0.35],
            [0.85, 0.15, 0.25],
            [1.0, 0.55, 0.12],
            [1.0, 0.90, 0.55],
        ],
        interior: [0.012, 0.009, 0.08],
    },
    Gradient {
        stops: [
            [0.005, 0.025, 0.07],
            [0.02, 0.30, 0.32],
            [0.08, 0.72, 0.45],
            [0.55, 0.45, 0.90],
            [0.86, 0.95, 1.0],
        ],
        interior: [0.005, 0.025, 0.07],
    },
    Gradient {
        stops: [
            [0.005, 0.02, 0.08],
            [0.02, 0.08, 0.30],
            [0.02, 0.55, 0.70],
            [0.25, 0.80, 0.70],
            [0.85, 0.95, 0.90],
        ],
        interior: [0.005, 0.02, 0.08],
    },
    Gradient {
        stops: [
            [0.03, 0.005, 0.005],
            [0.30, 0.01, 0.02],
            [0.80, 0.06, 0.025],
            [1.0, 0.38, 0.04],
            [1.0, 0.90, 0.35],
        ],
        interior: [0.03, 0.005, 0.005],
    },
    Gradient {
        stops: [
            [0.025, 0.005, 0.05],
            [0.20, 0.06, 0.34],
            [0.55, 0.20, 0.70],
            [0.85, 0.45, 0.75],
            [0.96, 0.82, 0.98],
        ],
        interior: [0.025, 0.005, 0.05],
    },
    Gradient {
        stops: [
            [0.005, 0.03, 0.01],
            [0.025, 0.20, 0.04],
            [0.30, 0.55, 0.04],
            [0.70, 0.80, 0.10],
            [0.95, 0.95, 0.65],
        ],
        interior: [0.005, 0.03, 0.01],
    },
    Gradient {
        stops: [
            [0.02, 0.012, 0.01],
            [0.18, 0.07, 0.03],
            [0.56, 0.23, 0.08],
            [0.84, 0.52, 0.20],
            [0.98, 0.86, 0.58],
        ],
        interior: [0.02, 0.012, 0.01],
    },
    Gradient {
        stops: [
            [0.005, 0.005, 0.025],
            [0.23, 0.02, 0.55],
            [0.90, 0.03, 0.60],
            [0.02, 0.90, 0.95],
            [0.95, 1.0, 0.25],
        ],
        interior: [0.005, 0.005, 0.025],
    },
    Gradient {
        stops: [
            [0.03, 0.01, 0.025],
            [0.22, 0.04, 0.14],
            [0.70, 0.25, 0.35],
            [0.96, 0.55, 0.43],
            [0.99, 0.88, 0.70],
        ],
        interior: [0.03, 0.01, 0.025],
    },
    Gradient {
        stops: [
            [0.008, 0.012, 0.06],
            [0.12, 0.08, 0.32],
            [0.34, 0.27, 0.62],
            [0.75, 0.38, 0.55],
            [0.95, 0.70, 0.58],
        ],
        interior: [0.008, 0.012, 0.06],
    },
];

fn gradient_color(gradient: Gradient, t: f32) -> [f32; 3] {
    let scaled = t * 4.0;
    let segment = (scaled as usize).min(3);
    let blend = scaled - segment as f32;
    let left = gradient.stops[segment];
    let right = gradient.stops[segment + 1];
    std::array::from_fn(|channel| left[channel] + (right[channel] - left[channel]) * blend)
}

/// The 1024-entry ramp plus the interior colour, as `[r, g, b, 0]` rows: the layout the
/// kernels read (`gpu.rs palette_color`), same as ofx-fractal's WGSL storage buffer.
pub fn build_lut(scheme: PaletteScheme) -> Vec<[f32; 4]> {
    let mut packed = Vec::with_capacity(PALETTE_SAMPLES + 1);
    for i in 0..PALETTE_SAMPLES {
        let t = i as f32 / (PALETTE_SAMPLES - 1) as f32;
        let rgb = match scheme {
            PaletteScheme::Classic => classic_color(t),
            PaletteScheme::Fire => [
                (2.0 * t).min(1.0),
                (2.0 * t - 0.55).clamp(0.0, 1.0),
                (2.0 * t - 1.3).clamp(0.0, 1.0),
            ],
            PaletteScheme::Ice => [
                (1.4 * t - 0.4).clamp(0.0, 1.0),
                (1.25 * t).min(1.0),
                (0.16 + 0.84 * t).min(1.0),
            ],
            PaletteScheme::Mono => [t, t, t],
            _ => gradient_color(GRADIENTS[scheme as usize - PaletteScheme::Sunset as usize], t),
        };
        packed.push([rgb[0], rgb[1], rgb[2], 0.0]);
    }
    let interior = match scheme {
        PaletteScheme::Classic => [0.015, 0.02, 0.06],
        PaletteScheme::Fire => [0.02, 0.005, 0.0],
        PaletteScheme::Ice => [0.0, 0.015, 0.04],
        PaletteScheme::Mono => [0.0; 3],
        _ => GRADIENTS[scheme as usize - PaletteScheme::Sunset as usize].interior,
    };
    packed.push([interior[0], interior[1], interior[2], 0.0]);
    packed
}

fn classic_color(t: f32) -> [f32; 3] {
    let inv = 1.0 - t;
    [
        9.0 * inv * t * t * t,
        15.0 * inv * inv * t * t,
        8.5 * inv * inv * inv * t,
    ]
}

