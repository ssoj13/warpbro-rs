//! Imageworks (Conty & Kulla 2017) sheen, the `conty_kulla` mode that standard_surface uses
//! (`pbrlib/pbrlib_defs.mtlx:133-138`).
//!
//! Port of `pbrlib/genglsl/lib/mx_microfacet_sheen.glsl:1-91` and
//! `pbrlib/genglsl/mx_sheen_bsdf.glsl` (MaterialX `v1.39.5-22-g47cecce6`). The Zeltner LTC mode
//! is not ported: standard_surface never selects it. The WGSL twin is `wgsl/sheen.wgsl`.

use crate::consts::{
    MX_FLOAT_EPS, MX_SHEEN_ALBEDO_C0, MX_SHEEN_ALBEDO_C1, MX_SHEEN_ALBEDO_C2, MX_SHEEN_ALBEDO_C3,
    MX_SHEEN_ALBEDO_C4, MX_SHEEN_ALBEDO_C5, MX_SHEEN_ROUGHNESS_MIN, MX_TWO_PI,
};
use crate::math::{
    MxBsdf, add2, add3, clampf, dot3, mul3, mx_square, normalize3, scale2, scale3, splat3,
};
use crate::microfacet::mx_forward_facing_normal;

/// WGSL functions of `wgsl/sheen.wgsl`, sorted: the twins of this module's functions
/// (`mx_` names kept, `ss_` names are the Rust names with the prefix). Checked by the
/// `twin_registry_matches_wgsl_and_rust` test; collected by [`crate::wgsl::TWIN_FUNCTIONS`].
pub const WGSL_TWINS: &[&str] = &[
    "mx_imageworks_sheen_brdf",
    "mx_imageworks_sheen_dir_albedo",
    "mx_imageworks_sheen_dir_albedo_analytic",
    "mx_imageworks_sheen_ndf",
    "mx_sheen_bsdf_indirect",
    "mx_sheen_bsdf_reflection",
];

/// Imageworks sheen NDF, `(2 + 1/r) sin^(1/r) / (2 pi)` with `r >= 0.005`:
/// `lib/mx_microfacet_sheen.glsl:3-11` (Conty & Kulla 2017, eq. 2).
pub fn mx_imageworks_sheen_ndf(ndoth: f32, roughness: f32) -> f32 {
    let inv_roughness = 1.0 / roughness.max(MX_SHEEN_ROUGHNESS_MIN);
    let cos2 = ndoth * ndoth;
    let sin2 = 1.0 - cos2;
    (2.0 + inv_roughness) * crate::fm::powf(sin2, inv_roughness * 0.5) / MX_TWO_PI
}

/// Imageworks sheen BRDF with the smoother Neubelt-Pettineo denominator
/// `4 (NdotL + NdotV - NdotL NdotV)` (F = G = 1): `lib/mx_microfacet_sheen.glsl:13-25`.
pub fn mx_imageworks_sheen_brdf(ndotl: f32, ndotv: f32, ndoth: f32, roughness: f32) -> f32 {
    let d = mx_imageworks_sheen_ndf(ndoth, roughness);
    d / (4.0 * (ndotl + ndotv - ndotl * ndotv))
}

/// Rational fit of the Imageworks sheen directional albedo:
/// `lib/mx_microfacet_sheen.glsl:27-37`.
pub fn mx_imageworks_sheen_dir_albedo_analytic(ndotv: f32, roughness: f32) -> f32 {
    let r = add2(
        add2(
            add2(
                add2(
                    add2(MX_SHEEN_ALBEDO_C0, scale2(MX_SHEEN_ALBEDO_C1, ndotv)),
                    scale2(MX_SHEEN_ALBEDO_C2, roughness),
                ),
                scale2(scale2(MX_SHEEN_ALBEDO_C3, ndotv), roughness),
            ),
            scale2(MX_SHEEN_ALBEDO_C4, mx_square(ndotv)),
        ),
        scale2(MX_SHEEN_ALBEDO_C5, mx_square(roughness)),
    );
    let [rx, ry] = r;
    rx / ry
}

/// Imageworks sheen directional albedo (analytic method), clamped to `[0, 1]`:
/// `lib/mx_microfacet_sheen.glsl:81-91`.
pub fn mx_imageworks_sheen_dir_albedo(ndotv: f32, roughness: f32) -> f32 {
    let dir_albedo = mx_imageworks_sheen_dir_albedo_analytic(ndotv, roughness);
    clampf(dir_albedo, 0.0, 1.0)
}

/// Sheen reflection closure for direct light, `response = color f NdotL w`,
/// `throughput = 1 - E(NdotV) w`: `pbrlib/genglsl/mx_sheen_bsdf.glsl:4-33,42`.
pub fn mx_sheen_bsdf_reflection(
    v: [f32; 3],
    l: [f32; 3],
    n: [f32; 3],
    weight: f32,
    color: [f32; 3],
    roughness: f32,
) -> MxBsdf {
    if weight < MX_FLOAT_EPS {
        return MxBsdf::EMPTY;
    }
    let n = mx_forward_facing_normal(n, v);
    let ndotv = clampf(dot3(n, v), MX_FLOAT_EPS, 1.0);

    let h = normalize3(add3(l, v));
    let ndotl = clampf(dot3(n, l), MX_FLOAT_EPS, 1.0);
    let ndoth = clampf(dot3(n, h), MX_FLOAT_EPS, 1.0);

    let fr = scale3(
        color,
        mx_imageworks_sheen_brdf(ndotl, ndotv, ndoth, roughness),
    );
    let dir_albedo = mx_imageworks_sheen_dir_albedo(ndotv, roughness);
    MxBsdf {
        response: scale3(scale3(fr, ndotl), weight),
        throughput: splat3(1.0 - dir_albedo * weight),
    }
}

/// Sheen closure for environment light, `response = irradiance color E(NdotV) w`:
/// `pbrlib/genglsl/mx_sheen_bsdf.glsl:44-60`. `irradiance` follows the convention of
/// [`crate::diffuse::mx_oren_nayar_diffuse_bsdf_indirect`].
pub fn mx_sheen_bsdf_indirect(
    v: [f32; 3],
    n: [f32; 3],
    weight: f32,
    color: [f32; 3],
    roughness: f32,
    irradiance: [f32; 3],
) -> MxBsdf {
    if weight < MX_FLOAT_EPS {
        return MxBsdf::EMPTY;
    }
    let n = mx_forward_facing_normal(n, v);
    let ndotv = clampf(dot3(n, v), MX_FLOAT_EPS, 1.0);

    let dir_albedo = mx_imageworks_sheen_dir_albedo(ndotv, roughness);
    MxBsdf {
        response: scale3(scale3(mul3(irradiance, color), dir_albedo), weight),
        throughput: splat3(1.0 - dir_albedo * weight),
    }
}
