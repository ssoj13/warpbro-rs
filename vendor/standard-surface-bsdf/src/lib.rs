//! Autodesk Standard Surface BSDF, ported from MaterialX `v1.39.5-22-g47cecce6`
//! (`vfx.ref/MaterialX/libraries`), as a Rust `f32` CPU twin and a WGSL library with the same
//! functions in the same statement order.
//!
//! Scope: `standard_surface` with `opacity = 1` and `thin_walled = false`, in which the graph
//! reduces exactly to coat over (metal | specular over mix(transmission, sheen over
//! mix(Oren-Nayar diffuse, subsurface))), with thin-film iridescence, anisotropy and the
//! coat-tinted emission. The transmission lobe is a path-traced rough dielectric BTDF
//! ([`transmission`]; MaterialX's own `T` mode is an environment lookup); `transmission_depth`,
//! `_scatter`, `_scatter_anisotropy` and `_dispersion` (the interior medium) are not inputs.
//! The subsurface closure is MaterialX's Burley-diffusion approximation
//! ([`diffuse::mx_subsurface_bsdf_reflection`]) at the caller's [`ShadingFrame::curvature`].
//! With `transmission = 0` and `subsurface = 0` every entry point returns the
//! reflection-only model's bits. It is shared by the render-rs raster pipeline
//! (`standard-surface`), the PT megakernel (`pt-megakernel`) and the ofx-rs fractal renderer.
//!
//! - [`surface`]: [`SurfaceInputs`] (MaterialX input names and defaults, fallible
//!   [`SurfaceInputs::validate`]), [`ShadingFrame`], and the infallible hot-path entry points
//!   [`eval_light`], [`eval_light_disc`], [`eval_environment`], [`eval_emission`],
//!   [`lobe_weights`] and [`dominant_dir`], all returning the [`Lobes`] partition.
//! - [`fresnel`], [`microfacet`], [`diffuse`], [`sheen`]: the MaterialX closures and their
//!   helpers, one Rust function per GLSL function (`mx_` names kept).
//! - [`sampling`], [`sample`]: counter RNG, samplers and `sample`/`pdf` (wave R2).
//! - [`transmission`]: the rough dielectric BTDF, its density and Snell refraction (Q6).
//! - [`wgsl`]: [`wgsl::wgsl_source`], the WGSL twin with the constants of [`consts`]
//!   prepended, and the [`wgsl::TWIN_FUNCTIONS`] registry.
//!
//! Transcendentals use the pure-Rust `libm` crate, so CPU results are identical across
//! platforms; WGSL results agree within the GPU's documented accuracy, not bit for bit.
//!
//! Licence: Apache-2.0, a derivative of MaterialX (Copyright Contributors to the MaterialX
//! Project); see `NOTICE`.

#![deny(missing_docs)]
#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

pub mod consts;
mod fm;
pub mod diffuse;
pub mod fresnel;
pub mod math;
pub mod microfacet;
pub mod sample;
pub mod sampling;
pub mod sheen;
pub mod surface;
pub mod transmission;
pub mod wgsl;

pub use microfacet::ThinFilmEnergy;
pub use surface::{
    Environment, InputRange, Lobe, LobeWeights, Lobes, ShadingFrame, SurfaceInputs, dominant_dir,
    eval_emission, eval_environment, eval_light, eval_light_disc, eval_light_materialx,
    lobe_weights,
};
pub use wgsl::wgsl_source;

/// Why [`SurfaceInputs::validate`] rejected an input. Evaluation never clamps or repairs
/// inputs; callers validate once, outside the hot path.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InputsError {
    /// The input is NaN or infinite.
    NonFinite {
        /// MaterialX input name, with `.r`/`.g`/`.b` for a colour channel.
        input: &'static str,
        /// The offending value.
        value: f32,
    },
    /// The input lies outside its domain (see [`SurfaceInputs::domain`]).
    OutOfRange {
        /// MaterialX input name, with `.r`/`.g`/`.b` for a colour channel.
        input: &'static str,
        /// The offending value.
        value: f32,
        /// Inclusive lower bound.
        min: f32,
        /// Inclusive upper bound.
        max: f32,
    },
}

impl core::fmt::Display for InputsError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            InputsError::NonFinite { input, value } => {
                write!(
                    f,
                    "standard surface input `{input}` is not finite ({value})"
                )
            }
            InputsError::OutOfRange {
                input,
                value,
                min,
                max,
            } => write!(
                f,
                "standard surface input `{input}` = {value} is outside [{min}, {max}]"
            ),
        }
    }
}

impl std::error::Error for InputsError {}
