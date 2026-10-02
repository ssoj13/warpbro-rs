//! Binding-free sampling helpers: the counter RNG (`pcg4d`, bit-identical on CPU and GPU),
//! cosine-hemisphere, uniform-cone and GGX visible-normal (VNDF) samplers, the anisotropic
//! Smith `G1` that is the VNDF sampler's true density term, tangent-frame transforms and the
//! MIS heuristics.
//!
//! Twin of `wgsl/sampling.wgsl`. The MaterialX samplers are ported from
//! `pbrlib/genglsl/lib/mx_microfacet.glsl:85-100` and
//! `pbrlib/genglsl/lib/mx_microfacet_specular.glsl:41-68` (MaterialX `v1.39.5-22-g47cecce6`),
//! with the GLSL names lower-cased (`VNDF`/`PDF` -> `vndf`/`pdf`) because Rust function names
//! are snake case; the WGSL twin uses the same lower-case names.
//!
//! The RNG is a pure function of `(pixel_x, pixel_y, sample, dimension ^ seed)`, so a render
//! is tile-, pass- and thread-invariant by construction, and its `u32 -> f32` map
//! ([`u32_to_unit`]) is exact, so CPU and WGSL uniforms are bit-identical.

use crate::consts::{MX_PI_INV, MX_TWO_PI, SS_PCG_INC, SS_PCG_MUL, SS_PCG_SHIFT, SS_U32_TO_UNIT};
use crate::math::{add3, clampf, dot3, normalize3, scale3};

/// WGSL functions of `wgsl/sampling.wgsl`, sorted: the twins of this module's functions
/// (`mx_` names kept, `ss_` names are the Rust names with the prefix). Checked by the
/// `twin_registry_matches_wgsl_and_rust` test; collected by [`crate::wgsl::TWIN_FUNCTIONS`].
pub const WGSL_TWINS: &[&str] = &[
    "mx_cosine_hemisphere_pdf",
    "mx_cosine_sample_hemisphere",
    "mx_ggx_importance_sample_vndf",
    "mx_ggx_vndf_reflection_pdf",
    "ss_balance_heuristic",
    "ss_cone_pdf",
    "ss_ggx_lambda_aniso",
    "ss_ggx_smith_g1_aniso",
    "ss_pcg4d",
    "ss_power_heuristic",
    "ss_rng",
    "ss_sample_cone",
    "ss_to_local",
    "ss_to_world",
    "ss_u32_to_unit",
];

/// `pcg4d` (Jarzynski & Olano 2020, "Hash Functions for GPU Rendering", JCGT 9(3)): a 4D
/// counter-based hash with wrapping `u32` arithmetic, identical on CPU and GPU.
pub fn pcg4d(v: [u32; 4]) -> [u32; 4] {
    let [x, y, z, w] = v;
    let mut x = x.wrapping_mul(SS_PCG_MUL).wrapping_add(SS_PCG_INC);
    let mut y = y.wrapping_mul(SS_PCG_MUL).wrapping_add(SS_PCG_INC);
    let mut z = z.wrapping_mul(SS_PCG_MUL).wrapping_add(SS_PCG_INC);
    let mut w = w.wrapping_mul(SS_PCG_MUL).wrapping_add(SS_PCG_INC);
    x = x.wrapping_add(y.wrapping_mul(w));
    y = y.wrapping_add(z.wrapping_mul(x));
    z = z.wrapping_add(x.wrapping_mul(y));
    w = w.wrapping_add(y.wrapping_mul(z));
    x ^= x >> SS_PCG_SHIFT;
    y ^= y >> SS_PCG_SHIFT;
    z ^= z >> SS_PCG_SHIFT;
    w ^= w >> SS_PCG_SHIFT;
    x = x.wrapping_add(y.wrapping_mul(w));
    y = y.wrapping_add(z.wrapping_mul(x));
    z = z.wrapping_add(x.wrapping_mul(y));
    w = w.wrapping_add(y.wrapping_mul(z));
    [x, y, z, w]
}

/// The top 24 bits of `x` as a float in `[0, 1)`: `(x >> 8) * 2^-24`, exact in f32.
pub fn u32_to_unit(x: u32) -> f32 {
    (x >> 8) as f32 * SS_U32_TO_UNIT
}

/// One uniform in `[0, 1)` for dimension `dim` of sample `sample` of pixel `(px, py)`:
/// `u32_to_unit(pcg4d(px, py, sample, dim ^ seed).x)` (plan §2 "Shared sampling module").
pub fn rng(px: u32, py: u32, sample: u32, dim: u32, seed: u32) -> f32 {
    let [x, _, _, _] = pcg4d([px, py, sample, dim ^ seed]);
    u32_to_unit(x)
}

/// Cosine-weighted direction on the `+z` hemisphere: `pbrlib/genglsl/lib/mx_microfacet.glsl:85-94`.
pub fn mx_cosine_sample_hemisphere(xi: [f32; 2]) -> [f32; 3] {
    let [xi_x, xi_y] = xi;
    let phi = MX_TWO_PI * xi_x;
    let cos_theta = crate::fm::sqrtf(xi_y);
    let sin_theta = crate::fm::sqrtf(1.0 - xi_y);
    [
        crate::fm::cosf(phi) * sin_theta,
        crate::fm::sinf(phi) * sin_theta,
        cos_theta,
    ]
}

/// Density of [`mx_cosine_sample_hemisphere`], `max(cos, 0) / pi`:
/// `pbrlib/genglsl/lib/mx_microfacet.glsl:96-100`.
pub fn mx_cosine_hemisphere_pdf(cos_theta: f32) -> f32 {
    cos_theta.max(0.0) * MX_PI_INV
}

/// Uniform direction in the cone about `+z` whose half-angle has `1 - cos = one_minus_cos_max`
/// (pass `2 sin^2(half_angle / 2)`, which stays accurate for the 0.27 degree sun, where
/// `1 - cos` in f32 would lose most of its bits). Density: [`cone_pdf`].
pub fn sample_cone(xi: [f32; 2], one_minus_cos_max: f32) -> [f32; 3] {
    let [xi_x, xi_y] = xi;
    let phi = MX_TWO_PI * xi_x;
    let one_minus_cos = xi_y * one_minus_cos_max;
    let cos_theta = 1.0 - one_minus_cos;
    let sin_theta = crate::fm::sqrtf((one_minus_cos * (2.0 - one_minus_cos)).max(0.0));
    [
        crate::fm::cosf(phi) * sin_theta,
        crate::fm::sinf(phi) * sin_theta,
        cos_theta,
    ]
}

/// Solid-angle density of [`sample_cone`], `1 / (2 pi (1 - cos_max))`.
pub fn cone_pdf(one_minus_cos_max: f32) -> f32 {
    1.0 / (MX_TWO_PI * one_minus_cos_max)
}

/// GGX visible-normal sampling with spherical caps (Dupuy & Benyoub 2023):
/// `lib/mx_microfacet_specular.glsl:41-62`. `v` and the returned half vector are in the
/// tangent frame `(X, Y, N)`; the reflection direction is `reflect(-v, h)`.
pub fn mx_ggx_importance_sample_vndf(xi: [f32; 2], v: [f32; 3], alpha: [f32; 2]) -> [f32; 3] {
    let [xi_x, xi_y] = xi;
    let [vx, vy, vz] = v;
    let [ax, ay] = alpha;
    // Transform the view direction to the hemisphere configuration.
    let [hvx, hvy, hvz] = normalize3([vx * ax, vy * ay, vz]);

    // Sample a spherical cap in (-V.z, 1].
    let phi = MX_TWO_PI * xi_x;
    let z = (1.0 - xi_y) * (1.0 + hvz) - hvz;
    let sin_theta = crate::fm::sqrtf(clampf(1.0 - z * z, 0.0, 1.0));
    let x = sin_theta * crate::fm::cosf(phi);
    let y = sin_theta * crate::fm::sinf(phi);

    // Compute the microfacet normal.
    let hx = x + hvx;
    let hy = y + hvy;
    let hz = z + hvz;

    // Transform the microfacet normal back to the ellipsoid configuration.
    normalize3([hx * ax, hy * ay, hz.max(0.0)])
}

/// Density of a reflection direction sampled from the GGX VNDF, `D(H) G1(V) / (4 NdotV)`:
/// `lib/mx_microfacet_specular.glsl:64-68`. Pass the **anisotropic** `G1`
/// ([`ggx_smith_g1_aniso`]): MaterialX's only caller passes the isotropic
/// `G1(NdotV, avgAlpha)` (`lib/mx_environment_fis.glsl:14-16`), which is not the sampler's
/// density when `alpha_x != alpha_y` (plan §2, C1).
pub fn mx_ggx_vndf_reflection_pdf(h: [f32; 3], alpha: [f32; 2], g1v: f32, ndotv: f32) -> f32 {
    crate::microfacet::mx_ggx_ndf(h, alpha) * g1v / (4.0 * ndotv)
}

/// Anisotropic GGX Smith `Lambda(V) = (-1 + sqrt(1 + (ax^2 Vx^2 + ay^2 Vy^2) / Vz^2)) / 2`
/// (Heitz 2014, JCGT 3(2), §5), `v` in the tangent frame on either side of the surface (`Vz`
/// enters squared). Shared by [`ggx_smith_g1_aniso`] and the height-correlated `G2` of the
/// transmission lobe ([`crate::transmission::ggx_transmission`]).
pub fn ggx_lambda_aniso(v: [f32; 3], alpha: [f32; 2]) -> f32 {
    let [vx, vy, vz] = v;
    let [ax, ay] = alpha;
    let a2 = (ax * ax * vx * vx + ay * ay * vy * vy) / (vz * vz);
    (-1.0 + crate::fm::sqrtf(1.0 + a2)) * 0.5
}

/// Anisotropic Smith masking `G1(V) = 1 / (1 + Lambda(V))` ([`ggx_lambda_aniso`], Heitz 2014,
/// JCGT 3(2), §5), `v` in the tangent frame. Equals `mx_ggx_smith_g1(Vz, alpha)` when `ax = ay`.
pub fn ggx_smith_g1_aniso(v: [f32; 3], alpha: [f32; 2]) -> f32 {
    1.0 / (1.0 + ggx_lambda_aniso(v, alpha))
}

/// World to tangent frame: `(v.X, v.Y, v.N)`.
pub fn to_local(v: [f32; 3], x: [f32; 3], y: [f32; 3], n: [f32; 3]) -> [f32; 3] {
    [dot3(v, x), dot3(v, y), dot3(v, n)]
}

/// Tangent frame to world: `l.x X + l.y Y + l.z N`.
pub fn to_world(l: [f32; 3], x: [f32; 3], y: [f32; 3], n: [f32; 3]) -> [f32; 3] {
    let [lx, ly, lz] = l;
    add3(add3(scale3(x, lx), scale3(y, ly)), scale3(n, lz))
}

/// Power heuristic (beta = 2) weight of strategy `a` against `b` (Veach 1997), 0 when both
/// densities are 0.
pub fn power_heuristic(pdf_a: f32, pdf_b: f32) -> f32 {
    let a2 = pdf_a * pdf_a;
    let b2 = pdf_b * pdf_b;
    let sum = a2 + b2;
    if sum > 0.0 { a2 / sum } else { 0.0 }
}

/// Balance heuristic weight of strategy `a` against `b` (Veach 1997), 0 when both densities
/// are 0.
pub fn balance_heuristic(pdf_a: f32, pdf_b: f32) -> f32 {
    let sum = pdf_a + pdf_b;
    if sum > 0.0 { pdf_a / sum } else { 0.0 }
}
