//! Oren-Nayar diffuse: the qualitative model standard_surface uses (energy compensation off,
//! `pbrlib/pbrlib_defs.mtlx:26-31`), its analytic directional albedo, and the Fujii /
//! OpenPBR energy-compensated variant the MaterialX node also offers; and the subsurface
//! closure `subsurface_bsdf` (MaterialX's Burley-diffusion approximation).
//!
//! Port of `pbrlib/genglsl/lib/mx_microfacet_diffuse.glsl:1-136,158-199`,
//! `pbrlib/genglsl/mx_oren_nayar_diffuse_bsdf.glsl` and `pbrlib/genglsl/mx_subsurface_bsdf.glsl`
//! (MaterialX `v1.39.5-22-g47cecce6`), including the `s <= 0` branch of the qualitative model,
//! which returns `A` exactly (`mx_microfacet_diffuse.glsl:11`). The WGSL twin is
//! `wgsl/diffuse.wgsl`.

use crate::consts::{
    MX_BURLEY_MFP_MIN, MX_BURLEY_SAMPLE_COUNT, MX_BURLEY_SAMPLE_WIDTH, MX_FLOAT_EPS,
    MX_FUJII_CONSTANT_1, MX_FUJII_CONSTANT_2, MX_ON_A_OFFSET, MX_ON_ALBEDO_C0, MX_ON_ALBEDO_C1,
    MX_ON_ALBEDO_C2, MX_ON_ALBEDO_C3, MX_ON_B_OFFSET, MX_ON_B_SCALE, MX_PI, MX_PI_INV,
    MX_SUBSURFACE_CURVATURE_MIN,
};
use crate::math::{
    MxBsdf, add2, add3, clampf, div3, div3s, dot3, exp3, max3s, mix3v, mul3, mx_square, scale2,
    scale3, splat3, sub3,
};
use crate::microfacet::mx_forward_facing_normal;

/// WGSL functions of `wgsl/diffuse.wgsl`, sorted: the twins of this module's functions
/// (`mx_` names kept, `ss_` names are the Rust names with the prefix). Checked by the
/// `twin_registry_matches_wgsl_and_rust` test; collected by [`crate::wgsl::TWIN_FUNCTIONS`].
pub const WGSL_TWINS: &[&str] = &[
    "mx_burley_diffusion_profile",
    "mx_integrate_burley_diffusion",
    "mx_oren_nayar_compensated_diffuse",
    "mx_oren_nayar_compensated_diffuse_dir_albedo",
    "mx_oren_nayar_diffuse",
    "mx_oren_nayar_diffuse_bsdf_indirect",
    "mx_oren_nayar_diffuse_bsdf_reflection",
    "mx_oren_nayar_diffuse_dir_albedo",
    "mx_oren_nayar_diffuse_dir_albedo_analytic",
    "mx_oren_nayar_fujii_diffuse_avg_albedo",
    "mx_oren_nayar_fujii_diffuse_dir_albedo",
    "mx_subsurface_bsdf_indirect",
    "mx_subsurface_bsdf_reflection",
    "ss_subsurface_scattering_approx",
];

/// Qualitative Oren-Nayar reflectance factor `A + B s/t` (Oren & Nayar 1994):
/// `lib/mx_microfacet_diffuse.glsl:6-18`. Multiply by `color / pi` for the BRDF.
pub fn mx_oren_nayar_diffuse(ndotv: f32, ndotl: f32, ldotv: f32, roughness: f32) -> f32 {
    let s = ldotv - ndotl * ndotv;
    let stinv = if s > 0.0 { s / ndotl.max(ndotv) } else { 0.0 };

    let sigma2 = mx_square(roughness);
    let a = 1.0 - 0.5 * (sigma2 / (sigma2 + MX_ON_A_OFFSET));
    let b = MX_ON_B_SCALE * sigma2 / (sigma2 + MX_ON_B_OFFSET);

    a + b * stinv
}

/// Rational fit of the qualitative Oren-Nayar directional albedo:
/// `lib/mx_microfacet_diffuse.glsl:20-28`.
pub fn mx_oren_nayar_diffuse_dir_albedo_analytic(ndotv: f32, roughness: f32) -> f32 {
    let r = add2(
        add2(
            add2(MX_ON_ALBEDO_C0, scale2(MX_ON_ALBEDO_C1, roughness)),
            scale2(scale2(MX_ON_ALBEDO_C2, ndotv), roughness),
        ),
        scale2(MX_ON_ALBEDO_C3, mx_square(roughness)),
    );
    let [rx, ry] = r;
    rx / ry
}

/// Qualitative Oren-Nayar directional albedo (analytic method), clamped to `[0, 1]`:
/// `lib/mx_microfacet_diffuse.glsl:75-83`.
pub fn mx_oren_nayar_diffuse_dir_albedo(ndotv: f32, roughness: f32) -> f32 {
    let dir_albedo = mx_oren_nayar_diffuse_dir_albedo_analytic(ndotv, roughness);
    clampf(dir_albedo, 0.0, 1.0)
}

/// Fujii's improved Oren-Nayar directional albedo: `lib/mx_microfacet_diffuse.glsl:85-95`.
pub fn mx_oren_nayar_fujii_diffuse_dir_albedo(cos_theta: f32, roughness: f32) -> f32 {
    let a = 1.0 / (1.0 + MX_FUJII_CONSTANT_1 * roughness);
    let b = roughness * a;
    let si = crate::fm::sqrtf((1.0 - mx_square(cos_theta)).max(0.0));
    let g = si * (crate::fm::acosf(clampf(cos_theta, -1.0, 1.0)) - si * cos_theta)
        + 2.0 * ((si / cos_theta) * (1.0 - si * si * si) - si) / 3.0;
    a + (b * g * MX_PI_INV)
}

/// Fujii's improved Oren-Nayar average albedo: `lib/mx_microfacet_diffuse.glsl:97-101`.
pub fn mx_oren_nayar_fujii_diffuse_avg_albedo(roughness: f32) -> f32 {
    let a = 1.0 / (1.0 + MX_FUJII_CONSTANT_1 * roughness);
    a * (1.0 + MX_FUJII_CONSTANT_2 * roughness)
}

/// Energy-compensated Oren-Nayar (OpenPBR): single- plus multi-scatter lobes,
/// `lib/mx_microfacet_diffuse.glsl:103-127`. The `s <= 0` branch keeps `s` here, unlike the
/// qualitative model.
pub fn mx_oren_nayar_compensated_diffuse(
    ndotv: f32,
    ndotl: f32,
    ldotv: f32,
    roughness: f32,
    color: [f32; 3],
) -> [f32; 3] {
    let s = ldotv - ndotl * ndotv;
    let stinv = if s > 0.0 { s / ndotl.max(ndotv) } else { s };

    // Single-scatter lobe.
    let a = 1.0 / (1.0 + MX_FUJII_CONSTANT_1 * roughness);
    let lobe_single_scatter = scale3(scale3(color, a), 1.0 + roughness * stinv);

    // Multi-scatter lobe.
    let dir_albedo_v = mx_oren_nayar_fujii_diffuse_dir_albedo(ndotv, roughness);
    let dir_albedo_l = mx_oren_nayar_fujii_diffuse_dir_albedo(ndotl, roughness);
    let avg_albedo = mx_oren_nayar_fujii_diffuse_avg_albedo(roughness);
    let color_multi_scatter = div3(
        scale3(mul3(color, color), avg_albedo),
        sub3(splat3(1.0), scale3(color, (1.0 - avg_albedo).max(0.0))),
    );
    let lobe_multi_scatter = div3s(
        scale3(
            scale3(color_multi_scatter, MX_FLOAT_EPS.max(1.0 - dir_albedo_v)),
            MX_FLOAT_EPS.max(1.0 - dir_albedo_l),
        ),
        MX_FLOAT_EPS.max(1.0 - avg_albedo),
    );

    add3(lobe_single_scatter, lobe_multi_scatter)
}

/// Directional albedo of the energy-compensated Oren-Nayar:
/// `lib/mx_microfacet_diffuse.glsl:129-136`.
pub fn mx_oren_nayar_compensated_diffuse_dir_albedo(
    cos_theta: f32,
    roughness: f32,
    color: [f32; 3],
) -> [f32; 3] {
    let dir_albedo = mx_oren_nayar_fujii_diffuse_dir_albedo(cos_theta, roughness);
    let avg_albedo = mx_oren_nayar_fujii_diffuse_avg_albedo(roughness);
    let color_multi_scatter = div3(
        scale3(mul3(color, color), avg_albedo),
        sub3(splat3(1.0), scale3(color, (1.0 - avg_albedo).max(0.0))),
    );
    mix3v(color_multi_scatter, color, splat3(dir_albedo))
}

/// Oren-Nayar reflection closure for direct light, `response = f * NdotL` with
/// `f = ON * color * w / pi`; `throughput = 0`: `pbrlib/genglsl/mx_oren_nayar_diffuse_bsdf.glsl:4-28`.
pub fn mx_oren_nayar_diffuse_bsdf_reflection(
    v: [f32; 3],
    l: [f32; 3],
    n: [f32; 3],
    weight: f32,
    color: [f32; 3],
    roughness: f32,
    energy_compensation: bool,
) -> MxBsdf {
    if weight < MX_FLOAT_EPS {
        return MxBsdf::default();
    }
    let n = mx_forward_facing_normal(n, v);
    let ndotv = clampf(dot3(n, v), MX_FLOAT_EPS, 1.0);
    let ndotl = clampf(dot3(n, l), MX_FLOAT_EPS, 1.0);
    let ldotv = clampf(dot3(l, v), MX_FLOAT_EPS, 1.0);

    let diffuse = if energy_compensation {
        mx_oren_nayar_compensated_diffuse(ndotv, ndotl, ldotv, roughness, color)
    } else {
        scale3(color, mx_oren_nayar_diffuse(ndotv, ndotl, ldotv, roughness))
    };
    MxBsdf {
        response: scale3(scale3(scale3(diffuse, weight), ndotl), MX_PI_INV),
        throughput: splat3(0.0),
    }
}

/// Oren-Nayar closure for environment light, `response = irradiance * albedo * color * w`:
/// `pbrlib/genglsl/mx_oren_nayar_diffuse_bsdf.glsl:29-36`. `irradiance` is the caller's
/// cosine-convolved environment radiance about `n` (irradiance / pi, MaterialX's
/// `$envIrradiance` convention: a uniform environment of radiance 1 passes 1).
pub fn mx_oren_nayar_diffuse_bsdf_indirect(
    v: [f32; 3],
    n: [f32; 3],
    weight: f32,
    color: [f32; 3],
    roughness: f32,
    energy_compensation: bool,
    irradiance: [f32; 3],
) -> MxBsdf {
    if weight < MX_FLOAT_EPS {
        return MxBsdf::default();
    }
    let n = mx_forward_facing_normal(n, v);
    let ndotv = clampf(dot3(n, v), MX_FLOAT_EPS, 1.0);

    let diffuse = if energy_compensation {
        mx_oren_nayar_compensated_diffuse_dir_albedo(ndotv, roughness, color)
    } else {
        scale3(color, mx_oren_nayar_diffuse_dir_albedo(ndotv, roughness))
    };
    MxBsdf {
        response: scale3(mul3(irradiance, diffuse), weight),
        throughput: splat3(0.0),
    }
}

/// Burley's normalised diffusion profile `(exp(-s d) + exp(-s d / 3)) / max(d, eps)` per channel
/// (Christensen & Burley 2015): `lib/mx_microfacet_diffuse.glsl:158-166`.
pub fn mx_burley_diffusion_profile(dist: f32, shape: [f32; 3]) -> [f32; 3] {
    let num1 = exp3(scale3(shape, -dist));
    let num2 = exp3(div3s(scale3(shape, -dist), 3.0));
    let denom = dist.max(MX_FLOAT_EPS);
    div3s(add3(num1, num2), denom)
}

/// The Burley profile integrated over a circle of the given curvature `radius` (Penner 2011,
/// pre-integrated skin): the profile-weighted mean of `max(cos(theta + x), 0)` over
/// `MX_BURLEY_SAMPLE_COUNT` arc positions `x`, `theta` the angle between `n` and `l`, shape
/// `1 / max(mfp, 0.1)`: `lib/mx_microfacet_diffuse.glsl:168-192`. `dot(n, l)` is clamped to
/// `[-1, 1]` before `acos` (MaterialX passes it unclamped; an f32 dot of unit vectors can leave
/// the domain by an ulp).
pub fn mx_integrate_burley_diffusion(
    n: [f32; 3],
    l: [f32; 3],
    radius: f32,
    mfp: [f32; 3],
) -> [f32; 3] {
    let theta = crate::fm::acosf(clampf(dot3(n, l), -1.0, 1.0));

    // Estimate the Burley diffusion shape from mean free path.
    let shape = div3(splat3(1.0), max3s(mfp, MX_BURLEY_MFP_MIN));

    // Integrate the profile over the sphere.
    let mut sum_d = splat3(0.0);
    let mut sum_r = splat3(0.0);
    for i in 0..MX_BURLEY_SAMPLE_COUNT {
        let x = -MX_PI + (i as f32 + 0.5) * MX_BURLEY_SAMPLE_WIDTH;
        let dist = radius * (2.0 * crate::fm::sinf(x * 0.5)).abs();
        let r = mx_burley_diffusion_profile(dist, shape);
        sum_d = add3(sum_d, scale3(r, crate::fm::cosf(theta + x).max(0.0)));
        sum_r = add3(sum_r, r);
    }

    div3(sum_d, sum_r)
}

/// `mx_subsurface_scattering_approx` (`lib/mx_microfacet_diffuse.glsl:194-199`),
/// `albedo * D / pi` with `D` the integrated profile at `radius = 1 / max(curvature, 0.01)`,
/// taking the surface curvature from the caller ([`crate::ShadingFrame::curvature`]) instead of
/// MaterialX's screen-space estimate `length(fwidth(N)) / length(fwidth(P))` (`:196`), which
/// only a rasteriser can form.
pub fn subsurface_scattering_approx(
    n: [f32; 3],
    l: [f32; 3],
    curvature: f32,
    albedo: [f32; 3],
    mfp: [f32; 3],
) -> [f32; 3] {
    let radius = 1.0 / curvature.max(MX_SUBSURFACE_CURVATURE_MIN);
    div3s(
        mul3(albedo, mx_integrate_burley_diffusion(n, l, radius, mfp)),
        MX_PI,
    )
}

/// Subsurface closure for direct light, `response = sss * w` with `sss` from
/// [`subsurface_scattering_approx`] and `throughput = 0` (a base closure):
/// `pbrlib/genglsl/mx_subsurface_bsdf.glsl:4-26`. Occlusion is 1 as for every closure here, so
/// MaterialX's `visibleOcclusion = 1 - NdotL (1 - occlusion)` (`:23-24`) is 1. MaterialX's
/// `anisotropy` input is not read by the closure body, so it is not a parameter. `radius` is the
/// mean free path (`subsurface_radius * subsurface_scale`).
pub fn mx_subsurface_bsdf_reflection(
    v: [f32; 3],
    l: [f32; 3],
    n: [f32; 3],
    curvature: f32,
    weight: f32,
    color: [f32; 3],
    radius: [f32; 3],
) -> MxBsdf {
    if weight < MX_FLOAT_EPS {
        return MxBsdf::default();
    }
    let n = mx_forward_facing_normal(n, v);
    let sss = subsurface_scattering_approx(n, l, curvature, color, radius);
    MxBsdf {
        response: scale3(sss, weight),
        throughput: splat3(0.0),
    }
}

/// Subsurface closure for environment light, rendered by MaterialX "as simple indirect
/// diffuse": `response = irradiance * color * w`, `throughput = 0`
/// (`pbrlib/genglsl/mx_subsurface_bsdf.glsl:27-31`); `irradiance` as for
/// [`mx_oren_nayar_diffuse_bsdf_indirect`].
pub fn mx_subsurface_bsdf_indirect(weight: f32, color: [f32; 3], irradiance: [f32; 3]) -> MxBsdf {
    if weight < MX_FLOAT_EPS {
        return MxBsdf::default();
    }
    MxBsdf {
        response: scale3(mul3(irradiance, color), weight),
        throughput: splat3(0.0),
    }
}
