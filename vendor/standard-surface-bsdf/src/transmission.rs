//! The transmission lobe: a rough dielectric BTDF (Walter, Marschner, Li, Torrance 2007,
//! "Microfacet Models for Refraction through Rough Surfaces", EGSR, eqs. 16-17 and 21), with
//! the anisotropic GGX NDF of [`mx_ggx_ndf`], the anisotropic height-correlated Smith `G2`
//! (Heitz 2014, JCGT 3(2), eq. 99) and visible-normal sampling
//! ([`crate::sampling::mx_ggx_importance_sample_vndf`] followed by [`refract`]).
//!
//! MaterialX has no path-traced BTDF to port: its `dielectric_bsdf` in `T` mode returns the
//! prefiltered environment behind a refracted solid sphere (`mx_transmission_refract.glsl:3-14`,
//! `mx_environment_prefilter.glsl:10-23`). This module is the path-tracing counterpart of
//! `transmission_bsdf` (`bxdf/standard_surface.mtlx:159-168`); [`crate::surface`] layers it as
//! MaterialX does (`transmission_mix` under `specular_layer`).
//!
//! Conventions:
//! - `eta = eta_wi / eta_wo` is the IOR of the medium `wi` lies in over the IOR of the medium of
//!   `wo`; `n` faces `wo`, so `wo` is above and a transmitted `wi` below the surface.
//! - Outside, the lobe is **Fresnel-free**: under `specular_layer` the dielectric specular's
//!   throughput `1 - E comp` already removes the reflected energy, so the transmitted part is
//!   what the layer passes down (`mx_layer_bsdf.glsl:3-7`); Walter's `(1 - F(wo.m))` would
//!   remove it a second time. Inside a transmissive object there is no layering
//!   ([`crate::surface::layers`]) and the BTDF carries Walter's Fresnel transmittance
//!   (`fresnel = true`), which is where total internal reflection happens.
//! - The value is the adjoint (flux) form: `f_t |cos_i| = D G2 |wo.m| |wi.m| eta^2 /
//!   (|cos_o| (eta wi.m + wo.m)^2)`. Its albedo is at most 1 (a white furnace passes), and it
//!   obeys `f(wo, wi) / eta_wi^2 = f(wi, wo) / eta_wo^2`. The radiance factor
//!   `(eta_wo / eta_wi)^2` of pbrt-v4's `TransportMode::Radiance` is omitted: it cancels on every
//!   path that enters and leaves a closed object, and Cycles and Arnold omit it too.
//!
//! Twin of `wgsl/transmission.wgsl`.

use crate::fresnel::mx_fresnel_dielectric;
use crate::math::{add3, cross3, dot3, normalize3, scale3, splat3, sub3};
use crate::microfacet::{MxDielectric, is_delta, mx_ggx_ndf, ndf_alpha};
use crate::sampling::{ggx_lambda_aniso, ggx_smith_g1_aniso, to_local};

/// WGSL functions of `wgsl/transmission.wgsl`, sorted: the twins of this module's functions
/// (`ss_` names are the Rust names with the prefix). Checked by the
/// `twin_registry_matches_wgsl_and_rust` test; collected by [`crate::wgsl::TWIN_FUNCTIONS`].
pub const WGSL_TWINS: &[&str] = &[
    "ss_ggx_transmission",
    "ss_ggx_transmission_pdf",
    "ss_refract",
    "ss_transmission_half",
    "ss_transmission_is_delta",
];

/// Refracts the unit `wo` (pointing away from the surface, `wo.m > 0`) through the microfacet
/// normal `m` into the medium of relative IOR `eta = eta_wi / eta_wo` (Snell:
/// `sin_i = sin_o / eta`). Returns the unit transmitted direction, or the zero vector under
/// total internal reflection (`eta < 1` only).
pub fn refract(wo: [f32; 3], m: [f32; 3], eta: f32) -> [f32; 3] {
    let cos_o = dot3(wo, m);
    let inv_eta = 1.0 / eta;
    let sin2_i = inv_eta * inv_eta * (1.0 - cos_o * cos_o).max(0.0);
    if sin2_i >= 1.0 {
        return splat3(0.0);
    }
    let cos_i = crate::fm::sqrtf(1.0 - sin2_i);
    add3(scale3(wo, -inv_eta), scale3(m, inv_eta * cos_o - cos_i))
}

/// The generalized half vector of a refraction, `normalize(wo + eta wi)` turned to the side of
/// `n` (Walter et al. 2007, eq. 16): the microfacet normal that refracts `wo` into `wi`.
pub fn transmission_half(wo: [f32; 3], wi: [f32; 3], n: [f32; 3], eta: f32) -> [f32; 3] {
    let h = normalize3(add3(wo, scale3(wi, eta)));
    if dot3(h, n) < 0.0 { scale3(h, -1.0) } else { h }
}

/// `true` when the transmission lobe is a delta (smooth refraction): its alpha is a delta alpha
/// ([`is_delta`]) or `eta = 1`, where every microfacet passes the ray straight through.
pub fn transmission_is_delta(alpha: [f32; 2], eta: f32) -> bool {
    is_delta(alpha) || eta == 1.0
}

/// `f_t(wo, wi) |n.wi|` of the rough dielectric BTDF (module documentation) of the closure
/// `p` (its `roughness` alpha pair and relative `ior = eta`; weight and tint are the caller's),
/// with the NDF widened by `ndf_widen` ([`ndf_alpha`]; 0 = none), `G2` on the material's own
/// alpha, and the Fresnel transmittance `1 - F(wo.m)` when `fresnel` (the bare interface). `t`
/// is the anisotropy axis (re-orthogonalised against `n`). Zero unless `wo` is above and `wi` below
/// the surface and the half vector faces both correctly (`wo.m > 0 > wi.m`). The caller
/// excludes delta lobes ([`transmission_is_delta`]).
pub fn ggx_transmission(
    wo: [f32; 3],
    wi: [f32; 3],
    n: [f32; 3],
    t: [f32; 3],
    p: &MxDielectric,
    ndf_widen: f32,
    fresnel: bool,
) -> f32 {
    let alpha = p.roughness;
    let eta = p.ior;
    let x = normalize3(sub3(t, scale3(n, dot3(t, n))));
    let y = cross3(n, x);
    let v = to_local(wo, x, y, n);
    let l = to_local(wi, x, y, n);
    let [_, _, vz] = v;
    let [_, _, lz] = l;
    let sides = vz > 0.0 && lz < 0.0;
    if !sides {
        return 0.0;
    }
    let h = transmission_half(v, l, [0.0, 0.0, 1.0], eta);
    let vdoth = dot3(v, h);
    let ldoth = dot3(l, h);
    let facing = vdoth > 0.0 && ldoth < 0.0;
    if !facing {
        return 0.0;
    }
    let d = mx_ggx_ndf(h, ndf_alpha(alpha, ndf_widen));
    let g = 1.0 / (1.0 + ggx_lambda_aniso(v, alpha) + ggx_lambda_aniso(l, alpha));
    let denom = eta * ldoth + vdoth;
    let transmittance = if fresnel {
        1.0 - mx_fresnel_dielectric(vdoth, eta)
    } else {
        1.0
    };
    transmittance * d * g * vdoth * -ldoth * eta * eta / (vz * denom * denom)
}

/// Solid-angle density of a transmitted direction sampled by [`refract`] of a VNDF half vector
/// ([`crate::sampling::mx_ggx_importance_sample_vndf`]): `D_V(m) |dm/dwi|` with
/// `D_V = G1_aniso(V) (V.m) D(m) / V.z` and `|dm/dwi| = eta^2 |wi.m| / (eta wi.m + wo.m)^2`
/// (Walter et al. 2007, eq. 17), for the closure `p` (alpha and `ior = eta`). Zero wherever
/// [`ggx_transmission`] is zero; over the lower hemisphere it integrates to 1 minus the share
/// of total internal reflection.
pub fn ggx_transmission_pdf(
    wo: [f32; 3],
    wi: [f32; 3],
    n: [f32; 3],
    t: [f32; 3],
    p: &MxDielectric,
) -> f32 {
    let alpha = p.roughness;
    let eta = p.ior;
    let x = normalize3(sub3(t, scale3(n, dot3(t, n))));
    let y = cross3(n, x);
    let v = to_local(wo, x, y, n);
    let l = to_local(wi, x, y, n);
    let [_, _, vz] = v;
    let [_, _, lz] = l;
    let sides = vz > 0.0 && lz < 0.0;
    if !sides {
        return 0.0;
    }
    let h = transmission_half(v, l, [0.0, 0.0, 1.0], eta);
    let vdoth = dot3(v, h);
    let ldoth = dot3(l, h);
    let facing = vdoth > 0.0 && ldoth < 0.0;
    if !facing {
        return 0.0;
    }
    let visible = ggx_smith_g1_aniso(v, alpha) * vdoth * mx_ggx_ndf(h, alpha) / vz;
    let denom = eta * ldoth + vdoth;
    visible * eta * eta * -ldoth / (denom * denom)
}
