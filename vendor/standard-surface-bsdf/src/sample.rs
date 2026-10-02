//! One-sample lobe selection over [`crate::lobe_weights`]: [`sample`] draws a direction and
//! [`pdf`] evaluates the density of that very sampler (plan §2, "sample/pdf").
//!
//! - **Lobe choice.** `u.x` picks lobe `k` with probability `p_k` from [`lobe_weights`], the one
//!   function computing the weights (I12); `u.yz` sample inside the lobe.
//! - **GGX lobes** (coat, specular, metal) use visible-normal sampling with spherical caps
//!   ([`mx_ggx_importance_sample_vndf`], MaterialX's own sampler,
//!   `lib/mx_microfacet_specular.glsl:41-62`). Its density is `D G1(V) / (4 NdotV)` with the
//!   **anisotropic** Smith `G1` ([`ggx_smith_g1_aniso`], C1), while [`crate::eval_light`] keeps
//!   MaterialX's shading `G2(avgAlpha)`: `eval / pdf` stays unbiased because the pdf is the
//!   sampler's true density.
//! - **Diffuse and sheen** use cosine-hemisphere sampling
//!   ([`mx_cosine_sample_hemisphere`], `mx_microfacet.glsl:85-94`): MaterialX has no closed-form
//!   importance sampler for the Imageworks sheen.
//! - **Transmission** ([`crate::transmission`], [`sample_transmission`]) samples a VNDF half
//!   vector and refracts `wo` through it ([`refract`]); the direction is below the surface,
//!   total internal reflection or a refraction back above the horizon gives `valid = false`.
//!   Inside a transmissive surface the specular lobe is the interface's reflection and the
//!   selection splits by [`crate::surface::interface_reflect_share`].
//! - **Mixture pdf.** [`pdf`] is `sum_k p_k pdf_k(wi)` over the non-delta lobes, exactly zero
//!   when `n.wo <= 0` (I1). Above the horizon it sums the reflection lobes, below it is the
//!   transmission lobe's density alone (zero when `transmission = 0`, so opaque materials keep
//!   `pdf = 0` below the horizon). VNDF reflections that fall below the horizon are returned
//!   with `valid = false` (zero contribution), so the upper-hemisphere integral of the
//!   reflection part is `1` minus that below-horizon share; [`ggx_lobe_pdf`] alone is the
//!   full-sphere density and integrates to 1.
//! - **Delta lobes** (C2, R3): a GGX lobe with `max(alpha) < SS_ALPHA_MIN` contributes nothing
//!   to [`pdf`] and [`crate::eval_light`]. When selected, [`sample`] returns the mirror direction
//!   with `delta = true`, `pdf = p_k` (a discrete probability, not a density) and
//!   `weight = F(NdotV) comp (layer factors) / p_k` ([`delta_response`]). A delta transmission
//!   lobe (smooth glass, or `specular_IOR = 1`) returns the refracted direction with the weight
//!   [`transmission_albedo`] `/ p_k`. Callers apply no MIS to a delta sample. Throughputs and
//!   `p_k` are unchanged, so a delta coat still darkens the diffuse beneath it and is still
//!   selected.
//! - **Estimator.** For a continuous sample `weight = eval_light(wo, wi).sum() / pdf(wo, wi)`
//!   (cosine included). Continuous and delta branches together are unbiased for the layered
//!   MaterialX model: each branch's expectation is its own part of the BSDF.
//!
//! The `_with` variants take the weights (a path tracer computing them once per vertex), the
//! `_prepared` variants additionally take [`crate::surface::layers`]; all paths go through
//! the `_prepared` functions. Twin of `wgsl/sample.wgsl` (struct `SsSample`, lobe as `u32`).

use crate::consts::MX_FLOAT_EPS;
use crate::fresnel::{
    mx_compute_fresnel, mx_ggx_energy_compensation, mx_init_fresnel_conductor,
    mx_init_fresnel_dielectric,
};
use crate::math::{
    add3, clampf, cross3, div3s, dot3, max3s, mul3, normalize3, reflect3, scale3, splat3, sub3,
};
use crate::microfacet::{
    MxConductor, MxDielectric, is_delta, mx_average_alpha, mx_dielectric_bsdf_indirect,
    mx_forward_facing_normal, mx_orthonormal_basis,
};
use crate::sampling::{
    ggx_smith_g1_aniso, mx_cosine_hemisphere_pdf, mx_cosine_sample_hemisphere,
    mx_ggx_importance_sample_vndf, mx_ggx_vndf_reflection_pdf, to_local, to_world,
};
use crate::surface::{
    Layers, Lobe, LobeWeights, ShadingFrame, SurfaceInputs, eval_light, layers, lobe_weights,
    transmission_albedo,
};
use crate::transmission::{ggx_transmission_pdf, refract, transmission_is_delta};

/// WGSL functions of `wgsl/sample.wgsl`, sorted: the twins of this module's functions
/// (`mx_` names kept, `ss_` names are the Rust names with the prefix). Checked by the
/// `twin_registry_matches_wgsl_and_rust` test; collected by [`crate::wgsl::TWIN_FUNCTIONS`].
pub const WGSL_TWINS: &[&str] = &[
    "ss_conductor_mirror_response",
    "ss_delta_response",
    "ss_dielectric_mirror_response",
    "ss_ggx_lobe_pdf",
    "ss_pdf",
    "ss_pdf_prepared",
    "ss_pdf_transmission",
    "ss_pdf_with",
    "ss_sample",
    "ss_sample_prepared",
    "ss_sample_transmission",
    "ss_sample_with",
    "ss_select_lobe",
];

/// A sampled direction. WGSL twin struct: `SsSample` (with `lobe: u32`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sample {
    /// Incident direction (unit, world space).
    pub wi: [f32; 3],
    /// `f cos / pdf` for a continuous sample; `delta_response / p_k` for a delta sample.
    pub weight: [f32; 3],
    /// Mixture solid-angle density [`pdf`] for a continuous sample; the selection probability
    /// `p_k` for a delta sample.
    pub pdf: f32,
    /// The lobe that generated `wi`.
    pub lobe: Lobe,
    /// `false` means zero contribution (no lobe reflects, `n.wo <= 0`, or a VNDF reflection
    /// below the horizon). Defined behaviour, not an error.
    pub valid: bool,
    /// The sample came from a delta (mirror) lobe: no MIS, never seen by next-event estimation.
    pub delta: bool,
}

impl Sample {
    /// The zero-contribution sample.
    pub const INVALID: Self = Self {
        wi: [0.0; 3],
        weight: [0.0; 3],
        pdf: 0.0,
        lobe: Lobe::Diffuse,
        valid: false,
        delta: false,
    };
}

/// Picks the lobe for `u` in `[0, 1)` from the normalised `weights`: the first lobe whose
/// cumulative interval contains `u`, lobes with `p_k = 0` never chosen; `None` when every
/// weight is 0. A `u` at or beyond the rounded total selects the last positive lobe. The WGSL
/// twin returns the lobe index, or -1 for none.
pub fn select_lobe(weights: &LobeWeights, u: f32) -> Option<Lobe> {
    let mut chosen = None;
    let mut acc = 0.0_f32;
    for (lobe, p) in Lobe::ALL.iter().zip(weights) {
        if *p > 0.0 {
            chosen = Some(*lobe);
            if u < acc + p {
                return chosen;
            }
        }
        acc += p;
    }
    chosen
}

/// Full-sphere density of a reflection sampled from one GGX lobe's VNDF: `D(H) G1_aniso(V) /
/// (4 V.z)` in the lobe's tangent frame (`t` re-orthogonalised against `n` exactly as the
/// closures do, `mx_dielectric_bsdf.glsl:34-35`), zero when `V.z <= 0` or `V.H <= 0`. It is
/// defined below the horizon too (where [`pdf`] is zero), so it integrates to 1 over the sphere.
pub fn ggx_lobe_pdf(wo: [f32; 3], wi: [f32; 3], n: [f32; 3], t: [f32; 3], alpha: [f32; 2]) -> f32 {
    let x = normalize3(sub3(t, scale3(n, dot3(t, n))));
    let y = cross3(n, x);
    let v = to_local(wo, x, y, n);
    let l = to_local(wi, x, y, n);
    let h = normalize3(add3(v, l));
    let [_, _, vz] = v;
    let visible = vz > 0.0 && dot3(v, h) > 0.0;
    if !visible {
        return 0.0;
    }
    let g1 = ggx_smith_g1_aniso(v, alpha);
    mx_ggx_vndf_reflection_pdf(h, alpha, g1, vz)
}

/// The mixture density of [`sample_prepared`] at `wi` (the single implementation behind
/// [`pdf`] and [`pdf_with`]).
pub fn pdf_prepared(s: &Layers, wo: [f32; 3], wi: [f32; 3], weights: &LobeWeights) -> f32 {
    let n = s.n;
    let above = dot3(n, wo) > 0.0 && dot3(n, wi) > 0.0;
    if !above {
        return pdf_transmission(s, wo, wi, weights);
    }
    let [p_coat, p_specular, p_metal, p_sheen, p_diffuse, _] = *weights;
    let cos_l = dot3(n, wi);
    let mut pdf = 0.0_f32;
    if p_coat > 0.0 && !is_delta(s.coat.roughness) {
        pdf += p_coat * ggx_lobe_pdf(wo, wi, n, s.coat_tangent, s.coat.roughness);
    }
    if p_specular > 0.0 && !is_delta(s.specular.roughness) {
        pdf += p_specular * ggx_lobe_pdf(wo, wi, n, s.main_tangent, s.specular.roughness);
    }
    if p_metal > 0.0 && !is_delta(s.metal.roughness) {
        pdf += p_metal * ggx_lobe_pdf(wo, wi, n, s.main_tangent, s.metal.roughness);
    }
    if p_sheen > 0.0 {
        pdf += p_sheen * mx_cosine_hemisphere_pdf(cos_l);
    }
    if p_diffuse > 0.0 {
        pdf += p_diffuse * mx_cosine_hemisphere_pdf(cos_l);
    }
    pdf
}

/// The transmission lobe's share of [`pdf_prepared`]: `p_transmission`
/// [`ggx_transmission_pdf`] when `wo` is above and `wi` below the horizon and the lobe is not a
/// delta; zero otherwise.
pub fn pdf_transmission(s: &Layers, wo: [f32; 3], wi: [f32; 3], weights: &LobeWeights) -> f32 {
    let n = s.n;
    let p_transmission = weights
        .get(Lobe::Transmission.index())
        .copied()
        .unwrap_or(0.0);
    let eta = s.transmission.ior;
    let continuous = dot3(n, wo) > 0.0
        && dot3(n, wi) < 0.0
        && p_transmission > 0.0
        && !transmission_is_delta(s.transmission.roughness, eta);
    if !continuous {
        return 0.0;
    }
    p_transmission * ggx_transmission_pdf(wo, wi, n, s.main_tangent, &s.transmission)
}

/// [`pdf`] with caller-supplied weights (from [`lobe_weights`] for the same `wo`).
pub fn pdf_with(
    i: &SurfaceInputs,
    f: &ShadingFrame,
    wo: [f32; 3],
    wi: [f32; 3],
    weights: &LobeWeights,
) -> f32 {
    pdf_prepared(&layers(i, f, wo), wo, wi, weights)
}

/// Solid-angle density with which [`sample`] generates the continuous direction `wi`; zero
/// outside the hemisphere and for delta lobes.
pub fn pdf(i: &SurfaceInputs, f: &ShadingFrame, wo: [f32; 3], wi: [f32; 3]) -> f32 {
    pdf_with(i, f, wo, wi, &lobe_weights(i, f, wo))
}

/// The `alpha -> 0` limit of the dielectric reflection closure integrated over `wi`:
/// `F(NdotV) comp tint w` (the NDF integrates to 1 and `G2 -> 1`; `mx_dielectric_bsdf.glsl:43-51`).
pub fn dielectric_mirror_response(v: [f32; 3], n: [f32; 3], p: &MxDielectric) -> [f32; 3] {
    if p.weight < MX_FLOAT_EPS {
        return splat3(0.0);
    }
    let n = mx_forward_facing_normal(n, v);
    let ndotv = clampf(dot3(n, v), MX_FLOAT_EPS, 1.0);
    let fd = mx_init_fresnel_dielectric(p.ior, p.thinfilm_thickness, p.thinfilm_ior);
    let [rx, ry] = p.roughness;
    let safe_alpha = [clampf(rx, MX_FLOAT_EPS, 1.0), clampf(ry, MX_FLOAT_EPS, 1.0)];
    let avg_alpha = mx_average_alpha(safe_alpha);
    let f = mx_compute_fresnel(ndotv, &fd);
    let comp = mx_ggx_energy_compensation(ndotv, avg_alpha, &fd);
    scale3(mul3(mul3(f, comp), max3s(p.tint, 0.0)), p.weight)
}

/// The `alpha -> 0` limit of the conductor reflection closure: `F(NdotV) comp w`
/// (`mx_conductor_bsdf.glsl:36-43`).
pub fn conductor_mirror_response(v: [f32; 3], n: [f32; 3], p: &MxConductor) -> [f32; 3] {
    if p.weight < MX_FLOAT_EPS {
        return splat3(0.0);
    }
    let n = mx_forward_facing_normal(n, v);
    let ndotv = clampf(dot3(n, v), MX_FLOAT_EPS, 1.0);
    let fd = mx_init_fresnel_conductor(p.ior_n, p.ior_k, p.thinfilm_thickness, p.thinfilm_ior);
    let [rx, ry] = p.roughness;
    let safe_alpha = [clampf(rx, MX_FLOAT_EPS, 1.0), clampf(ry, MX_FLOAT_EPS, 1.0)];
    let avg_alpha = mx_average_alpha(safe_alpha);
    let f = mx_compute_fresnel(ndotv, &fd);
    let comp = mx_ggx_energy_compensation(ndotv, avg_alpha, &fd);
    scale3(mul3(f, comp), p.weight)
}

/// The reflected energy of a delta GGX lobe toward the mirror direction, with every layer
/// factor above it: coat `F comp w`; specular `T_coat att (1 - m) F comp tint w`; metal
/// `T_coat att m F comp` (the delta limit of the [`crate::surface::combine`] partition). For a
/// delta transmission lobe the refracted energy [`transmission_albedo`]. Zero for the diffuse
/// and sheen lobes, which are never delta.
pub fn delta_response(s: &Layers, wo: [f32; 3], lobe: Lobe) -> [f32; 3] {
    let coat_throughput = mx_dielectric_bsdf_indirect(wo, s.n, &s.coat, splat3(1.0)).throughput;
    let under_coat = mul3(coat_throughput, s.attenuation);
    let m = s.metalness;
    match lobe {
        Lobe::Coat => dielectric_mirror_response(wo, s.n, &s.coat),
        Lobe::Specular => mul3(
            scale3(under_coat, 1.0 - m),
            dielectric_mirror_response(wo, s.n, &s.specular),
        ),
        Lobe::Metal => mul3(
            scale3(under_coat, m),
            conductor_mirror_response(wo, s.n, &s.metal),
        ),
        Lobe::Transmission => transmission_albedo(s, wo),
        Lobe::Sheen | Lobe::Diffuse => splat3(0.0),
    }
}

/// The transmission branch of [`sample_prepared`], `p_k` the lobe's selection probability: a
/// delta lobe returns the Snell refraction of `wo` about `n` (invalid under total internal
/// reflection), otherwise a VNDF half vector ([`mx_ggx_importance_sample_vndf`]) refracts `wo`
/// ([`refract`]) and the direction must end below the horizon.
pub fn sample_transmission(
    i: &SurfaceInputs,
    f: &ShadingFrame,
    s: &Layers,
    wo: [f32; 3],
    xi: [f32; 2],
    weights: &LobeWeights,
    p_k: f32,
) -> Sample {
    let n = s.n;
    let lobe = Lobe::Transmission;
    let alpha = s.transmission.roughness;
    let eta = s.transmission.ior;
    if transmission_is_delta(alpha, eta) {
        let wi = refract(wo, n, eta);
        return Sample {
            wi,
            weight: div3s(delta_response(s, wo, lobe), p_k),
            pdf: p_k,
            lobe,
            valid: dot3(n, wi) < 0.0,
            delta: true,
        };
    }
    let x = normalize3(sub3(s.main_tangent, scale3(n, dot3(s.main_tangent, n))));
    let y = cross3(n, x);
    let v = to_local(wo, x, y, n);
    let h = mx_ggx_importance_sample_vndf(xi, v, alpha);
    let wi = to_world(refract(v, h, eta), x, y, n);
    // Total internal reflection gives the zero vector, which is not below the horizon.
    let below = dot3(n, wi) < 0.0;
    if !below {
        return Sample {
            wi,
            lobe,
            ..Sample::INVALID
        };
    }
    let density = pdf_prepared(s, wo, wi, weights);
    let positive = density > 0.0;
    if !positive {
        return Sample {
            wi,
            lobe,
            ..Sample::INVALID
        };
    }
    Sample {
        wi,
        weight: div3s(eval_light(i, f, wo, wi).transmission, density),
        pdf: density,
        lobe,
        valid: true,
        delta: false,
    }
}

/// One BSDF sample (the single implementation behind [`sample`] and [`sample_with`]):
/// `u.x` selects the lobe, `u.yz` sample it (see the module documentation).
pub fn sample_prepared(
    i: &SurfaceInputs,
    f: &ShadingFrame,
    s: &Layers,
    wo: [f32; 3],
    u: [f32; 3],
    weights: &LobeWeights,
) -> Sample {
    let n = s.n;
    let facing = dot3(n, wo) > 0.0;
    if !facing {
        return Sample::INVALID;
    }
    let [ux, uy, uz] = u;
    let xi = [uy, uz];
    let Some(lobe) = select_lobe(weights, ux) else {
        return Sample::INVALID;
    };
    let p_k = weights.get(lobe.index()).copied().unwrap_or(0.0);
    if lobe == Lobe::Transmission {
        return sample_transmission(i, f, s, wo, xi, weights, p_k);
    }
    let (alpha, tangent, ggx) = match lobe {
        Lobe::Coat => (s.coat.roughness, s.coat_tangent, true),
        Lobe::Specular => (s.specular.roughness, s.main_tangent, true),
        Lobe::Metal => (s.metal.roughness, s.main_tangent, true),
        Lobe::Sheen | Lobe::Diffuse | Lobe::Transmission => ([1.0, 1.0], n, false),
    };

    if ggx && is_delta(alpha) {
        let wi = reflect3(scale3(wo, -1.0), n);
        return Sample {
            wi,
            weight: div3s(delta_response(s, wo, lobe), p_k),
            pdf: p_k,
            lobe,
            valid: dot3(n, wi) > 0.0,
            delta: true,
        };
    }

    let wi = if ggx {
        let x = normalize3(sub3(tangent, scale3(n, dot3(tangent, n))));
        let y = cross3(n, x);
        let v = to_local(wo, x, y, n);
        let h = mx_ggx_importance_sample_vndf(xi, v, alpha);
        let l = reflect3(scale3(v, -1.0), h);
        to_world(l, x, y, n)
    } else {
        let [bx, by, _] = mx_orthonormal_basis(n);
        to_world(mx_cosine_sample_hemisphere(xi), bx, by, n)
    };
    let above = dot3(n, wi) > 0.0;
    if !above {
        return Sample {
            wi,
            lobe,
            ..Sample::INVALID
        };
    }
    let density = pdf_prepared(s, wo, wi, weights);
    let positive = density > 0.0;
    if !positive {
        return Sample {
            wi,
            lobe,
            ..Sample::INVALID
        };
    }
    Sample {
        wi,
        weight: div3s(eval_light(i, f, wo, wi).sum(), density),
        pdf: density,
        lobe,
        valid: true,
        delta: false,
    }
}

/// [`sample`] with caller-supplied weights (from [`lobe_weights`] for the same `wo`).
pub fn sample_with(
    i: &SurfaceInputs,
    f: &ShadingFrame,
    wo: [f32; 3],
    u: [f32; 3],
    weights: &LobeWeights,
) -> Sample {
    sample_prepared(i, f, &layers(i, f, wo), wo, u, weights)
}

/// One BSDF sample for the view direction `wo` and three uniforms `u` in `[0, 1)`.
pub fn sample(i: &SurfaceInputs, f: &ShadingFrame, wo: [f32; 3], u: [f32; 3]) -> Sample {
    sample_with(i, f, wo, u, &lobe_weights(i, f, wo))
}
