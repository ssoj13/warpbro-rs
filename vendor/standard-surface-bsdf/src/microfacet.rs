//! GGX microfacet model and the GGX closures (dielectric and conductor reflection).
//!
//! Port of `pbrlib/genglsl/lib/mx_microfacet_specular.glsl` (NDF, Smith G, albedo fit, average
//! alpha), `pbrlib/genglsl/lib/mx_microfacet.glsl` (forward-facing normal, orthonormal basis),
//! `pbrlib/genglsl/mx_roughness_anisotropy.glsl`, `pbrlib/genglsl/mx_dielectric_bsdf.glsl` and
//! `pbrlib/genglsl/mx_conductor_bsdf.glsl` (MaterialX `v1.39.5-22-g47cecce6`). The WGSL twin is
//! `wgsl/microfacet.wgsl`.
//!
//! Two additions to MaterialX, both defined here and used by [`crate::surface`]:
//! - [`ndf_alpha`]: the sun-disc widening of the NDF alpha (`widen = 0` is MaterialX exactly);
//! - [`is_delta`]: the delta-lobe classification against [`SS_ALPHA_MIN`].
//!
//! Closure differences from MaterialX, all exact for the shared BSDF's scope: `occlusion` is 1
//! (a multiply by 1.0 is dropped), the `retroreflective` and `T`-mode branches are absent (plan
//! §2; the path-traced transmission lobe is [`crate::transmission`]), and the indirect closures
//! take the caller's environment radiance instead of a latlong texture lookup
//! (`lib/mx_environment_prefilter.glsl:10-23` returns `Li * FG`).

use crate::consts::{MX_ANISOTROPY_MAX, MX_FLOAT_EPS, MX_GGX_ALBEDO_C0, MX_GGX_ALBEDO_C1};
use crate::consts::{MX_GGX_ALBEDO_C2, MX_GGX_ALBEDO_C3, MX_GGX_ALBEDO_C4, MX_GGX_ALBEDO_C5};
use crate::consts::{MX_GGX_ALBEDO_C6, MX_GGX_ALBEDO_C7, MX_GGX_ALBEDO_C8, MX_PI, SS_ALPHA_MIN};
use crate::fresnel::{
    MxFresnelData, film_dir_albedo, mx_compute_fresnel, mx_ggx_energy_compensation,
    mx_init_fresnel_conductor, mx_init_fresnel_dielectric, mx_ior_to_f0,
};
use crate::math::{
    MxBsdf, add3, add4, clampf, cross3, div3s, dot3, max3s, mul3, mx_square, normalize3, scale3,
    scale4, splat3, sub3,
};

/// WGSL functions of `wgsl/microfacet.wgsl`, sorted: the twins of this module's functions
/// (`mx_` names kept, `ss_` names are the Rust names with the prefix). Checked by the
/// `twin_registry_matches_wgsl_and_rust` test; collected by [`crate::wgsl::TWIN_FUNCTIONS`].
pub const WGSL_TWINS: &[&str] = &[
    "mx_average_alpha",
    "mx_conductor_bsdf_indirect",
    "mx_conductor_bsdf_reflection",
    "mx_dielectric_bsdf_indirect",
    "mx_dielectric_bsdf_reflection",
    "mx_forward_facing_normal",
    "mx_ggx_dir_albedo_analytic",
    "mx_ggx_dir_albedo_scalar",
    "mx_ggx_ndf",
    "mx_ggx_smith_g1",
    "mx_ggx_smith_g2",
    "mx_orthonormal_basis",
    "mx_roughness_anisotropy",
    "ss_dielectric_dir_albedo",
    "ss_is_delta",
    "ss_ndf_alpha",
];

/// How a dielectric GGX closure with a thin film computes its throughput (plan TODO "found
/// during R1", task R2b; an explicit input, never an environment variable).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ThinFilmEnergy {
    /// MaterialX 1.39.5 exactly: the throughput uses the film-free scalar albedo
    /// `E(NdotV, F0 = ior_to_f0(ior), F90 = 1)` (`mx_dielectric_bsdf.glsl:26,47-49`) while the
    /// response uses the Airy Fresnel, so a film adds energy (white furnace 1.334 at 400 nm).
    #[default]
    MaterialX,
    /// The dielectric throughput uses the same film-aware albedo as the environment response,
    /// and that albedo ([`crate::fresnel::film_dir_albedo`]) weights MaterialX's Airy mirror
    /// term by the single-scatter energy, so dielectric and conductor lobes with a film stay at
    /// or below 1 for a white base, up to the albedo-fit error (tasks R2b/R2c). Without a film
    /// both modes agree bit for bit.
    Conserving,
}

impl ThinFilmEnergy {
    /// The `SS_THIN_FILM_ENERGY_*` code of this mode in the WGSL twin (`SsInputs`,
    /// `MxDielectric`).
    pub const fn code(self) -> u32 {
        match self {
            ThinFilmEnergy::MaterialX => crate::consts::SS_THIN_FILM_ENERGY_MATERIALX,
            ThinFilmEnergy::Conserving => crate::consts::SS_THIN_FILM_ENERGY_CONSERVING,
        }
    }
}

/// Inputs of MaterialX `dielectric_bsdf` in reflection mode (`pbrlib/pbrlib_defs.mtlx:62-73`,
/// `mx_dielectric_bsdf.glsl:4`), minus the constant `normal`/`tangent`/`distribution`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MxDielectric {
    /// Closure weight; below `M_FLOAT_EPS` the closure is skipped.
    pub weight: f32,
    /// Response tint (clamped to `>= 0`).
    pub tint: [f32; 3],
    /// Real IOR.
    pub ior: f32,
    /// GGX alpha pair `(alpha_x, alpha_y)` from [`mx_roughness_anisotropy`].
    pub roughness: [f32; 2],
    /// Thin-film thickness in nanometres (0 disables Airy).
    pub thinfilm_thickness: f32,
    /// Thin-film IOR.
    pub thinfilm_ior: f32,
    /// Throughput model with a thin film (not a MaterialX input; [`ThinFilmEnergy`]).
    pub thin_film_energy: ThinFilmEnergy,
}

/// Inputs of MaterialX `conductor_bsdf` (`mx_conductor_bsdf.glsl:4`), minus the constant
/// `normal`/`tangent`/`distribution`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MxConductor {
    /// Closure weight; below `M_FLOAT_EPS` the closure is skipped.
    pub weight: f32,
    /// Real IOR `n` per channel.
    pub ior_n: [f32; 3],
    /// Extinction `k` per channel.
    pub ior_k: [f32; 3],
    /// GGX alpha pair `(alpha_x, alpha_y)`.
    pub roughness: [f32; 2],
    /// Thin-film thickness in nanometres (0 disables Airy).
    pub thinfilm_thickness: f32,
    /// Thin-film IOR.
    pub thinfilm_ior: f32,
    /// Film energy model of the environment response (not a MaterialX input;
    /// [`ThinFilmEnergy`]).
    pub thin_film_energy: ThinFilmEnergy,
}

/// Anisotropic GGX NDF `D(H)` with `H` in the tangent frame `(X, Y, N)`:
/// `lib/mx_microfacet_specular.glsl:32-39` (Burley 2012, B.2 eq. 13).
pub fn mx_ggx_ndf(h: [f32; 3], alpha: [f32; 2]) -> f32 {
    let [hx, hy, hz] = h;
    let [ax, ay] = alpha;
    let hex = hx / ax;
    let hey = hy / ay;
    let denom = hex * hex + hey * hey + mx_square(hz);
    1.0 / (MX_PI * ax * ay * mx_square(denom))
}

/// Isotropic Smith `G1` for GGX (Walter et al. 2007, eq. 34):
/// `lib/mx_microfacet_specular.glsl:70-77`.
pub fn mx_ggx_smith_g1(cos_theta: f32, alpha: f32) -> f32 {
    let cos_theta2 = mx_square(cos_theta);
    let tan_theta2 = (1.0 - cos_theta2) / cos_theta2;
    2.0 / (1.0 + crate::fm::sqrtf(1.0 + mx_square(alpha) * tan_theta2))
}

/// Height-correlated Smith masking-shadowing `G2` (Heitz 2014, eqs. 72 and 99):
/// `lib/mx_microfacet_specular.glsl:79-88`.
pub fn mx_ggx_smith_g2(ndotl: f32, ndotv: f32, alpha: f32) -> f32 {
    let alpha2 = mx_square(alpha);
    let lambda_l = crate::fm::sqrtf(alpha2 + (1.0 - alpha2) * mx_square(ndotl));
    let lambda_v = crate::fm::sqrtf(alpha2 + (1.0 - alpha2) * mx_square(ndotv));
    2.0 * ndotl * ndotv / (lambda_l * ndotv + lambda_v * ndotl)
}

/// Rational fit of the GGX directional albedo, `F0 * A + F90 * B`:
/// `lib/mx_microfacet_specular.glsl:90-108`. MaterialX's codegen default albedo method is
/// this analytic fit (`GenOptions.h:89`), the only method ported.
pub fn mx_ggx_dir_albedo_analytic(ndotv: f32, alpha: f32, f0: [f32; 3], f90: [f32; 3]) -> [f32; 3] {
    let x = ndotv;
    let y = alpha;
    let x2 = mx_square(x);
    let y2 = mx_square(y);
    let r = add4(
        add4(
            add4(
                add4(
                    add4(
                        add4(
                            add4(
                                add4(MX_GGX_ALBEDO_C0, scale4(MX_GGX_ALBEDO_C1, x)),
                                scale4(MX_GGX_ALBEDO_C2, y),
                            ),
                            scale4(scale4(MX_GGX_ALBEDO_C3, x), y),
                        ),
                        scale4(MX_GGX_ALBEDO_C4, x2),
                    ),
                    scale4(MX_GGX_ALBEDO_C5, y2),
                ),
                scale4(scale4(MX_GGX_ALBEDO_C6, x2), y),
            ),
            scale4(scale4(MX_GGX_ALBEDO_C7, x), y2),
        ),
        scale4(scale4(MX_GGX_ALBEDO_C8, x2), y2),
    );
    let [rx, ry, rz, rw] = r;
    let ab_x = clampf(rx / rz, 0.0, 1.0);
    let ab_y = clampf(ry / rw, 0.0, 1.0);
    add3(scale3(f0, ab_x), scale3(f90, ab_y))
}

/// Scalar overload of the GGX directional albedo, `mx_ggx_dir_albedo(float, float, float,
/// float)`: `lib/mx_microfacet_specular.glsl:171-174`.
pub fn mx_ggx_dir_albedo_scalar(ndotv: f32, alpha: f32, f0: f32, f90: f32) -> f32 {
    let [e, _, _] = mx_ggx_dir_albedo_analytic(ndotv, alpha, splat3(f0), splat3(f90));
    e
}

/// Average of an anisotropic alpha pair, `sqrt(alpha_x * alpha_y)`:
/// `lib/mx_microfacet_specular.glsl:176-180`.
pub fn mx_average_alpha(alpha: [f32; 2]) -> f32 {
    let [ax, ay] = alpha;
    crate::fm::sqrtf(ax * ay)
}

/// Roughness and anisotropy to a GGX alpha pair, `r^2` clamped to `[M_FLOAT_EPS, 1]`:
/// `pbrlib/genglsl/mx_roughness_anisotropy.glsl:1-15`.
pub fn mx_roughness_anisotropy(roughness: f32, anisotropy: f32) -> [f32; 2] {
    let roughness_sqr = clampf(roughness * roughness, MX_FLOAT_EPS, 1.0);
    if anisotropy > 0.0 {
        let aspect = crate::fm::sqrtf(1.0 - clampf(anisotropy, 0.0, MX_ANISOTROPY_MAX));
        [(roughness_sqr / aspect).min(1.0), roughness_sqr * aspect]
    } else {
        [roughness_sqr, roughness_sqr]
    }
}

/// Flips `n` toward `v`: `pbrlib/genglsl/lib/mx_microfacet.glsl:55-59`.
pub fn mx_forward_facing_normal(n: [f32; 3], v: [f32; 3]) -> [f32; 3] {
    if dot3(n, v) < 0.0 { scale3(n, -1.0) } else { n }
}

/// Orthonormal basis `[X, Y, N]` (columns) from a unit `N` (Duff et al. 2017):
/// `pbrlib/genglsl/lib/mx_microfacet.glsl:108-118`. The WGSL twin returns `mat3x3<f32>`.
pub fn mx_orthonormal_basis(n: [f32; 3]) -> [[f32; 3]; 3] {
    let [nx, ny, nz] = n;
    let sign = if nz < 0.0 { -1.0 } else { 1.0 };
    let a = -1.0 / (sign + nz);
    let b = nx * ny * a;
    let x = [1.0 + sign * nx * nx * a, sign * b, -sign * nx];
    let y = [b, sign + ny * ny * a, -ny];
    [x, y, n]
}

/// The NDF alpha of a lobe lit by a disc light: per axis `min(1, sqrt(alpha^2 + widen^2))`
/// with `widen = tan(half_angle) / 2` (plan §4, "Disc widening"); `widen <= 0` returns `alpha`
/// unchanged, so point lights evaluate MaterialX exactly. Only the NDF is widened: `G`, the
/// Fresnel, the energy compensation and the throughput keep the material's own alpha.
pub fn ndf_alpha(alpha: [f32; 2], widen: f32) -> [f32; 2] {
    if widen > 0.0 {
        let [ax, ay] = alpha;
        let w2 = widen * widen;
        [
            crate::fm::sqrtf(ax * ax + w2).min(1.0),
            crate::fm::sqrtf(ay * ay + w2).min(1.0),
        ]
    } else {
        alpha
    }
}

/// `true` when a GGX lobe of NDF alpha `alpha` is a delta (mirror) lobe,
/// `max(alpha_x, alpha_y) < SS_ALPHA_MIN` (plan §2, C2).
pub fn is_delta(alpha: [f32; 2]) -> bool {
    let [ax, ay] = alpha;
    ax.max(ay) < SS_ALPHA_MIN
}

/// Directional albedo behind a dielectric closure's throughput, `E comp`: MaterialX's scalar
/// `E(NdotV, F0, 1) comp` (`mx_dielectric_bsdf.glsl:47-48`) or, in
/// [`ThinFilmEnergy::Conserving`], the film-aware `E(fd) comp`.
pub fn dielectric_dir_albedo(
    ndotv: f32,
    avg_alpha: f32,
    f0: f32,
    fd: &MxFresnelData,
    comp: [f32; 3],
    energy: ThinFilmEnergy,
) -> [f32; 3] {
    match energy {
        ThinFilmEnergy::MaterialX => {
            scale3(comp, mx_ggx_dir_albedo_scalar(ndotv, avg_alpha, f0, 1.0))
        }
        ThinFilmEnergy::Conserving => mul3(film_dir_albedo(ndotv, avg_alpha, fd, energy), comp),
    }
}

/// Dielectric GGX reflection closure for direct light (`CLOSURE_TYPE_REFLECTION`):
/// `pbrlib/genglsl/mx_dielectric_bsdf.glsl:4-52`. `response = D F G comp tint w / (4 NdotV)`
/// (the cosine is included); `throughput = 1 - E(NdotV, F0) comp w` with the scalar `F0` of
/// the IOR even when a thin film is present (`:26,47-49`). `ndf_widen` widens only `D`
/// ([`ndf_alpha`]; 0 = MaterialX).
pub fn mx_dielectric_bsdf_reflection(
    v: [f32; 3],
    l: [f32; 3],
    n: [f32; 3],
    x: [f32; 3],
    p: &MxDielectric,
    ndf_widen: f32,
) -> MxBsdf {
    if p.weight < MX_FLOAT_EPS {
        return MxBsdf::EMPTY;
    }
    let n = mx_forward_facing_normal(n, v);
    let ndotv = clampf(dot3(n, v), MX_FLOAT_EPS, 1.0);

    let fd = mx_init_fresnel_dielectric(p.ior, p.thinfilm_thickness, p.thinfilm_ior);
    let f0 = mx_ior_to_f0(p.ior);

    let [rx, ry] = p.roughness;
    let safe_alpha = [clampf(rx, MX_FLOAT_EPS, 1.0), clampf(ry, MX_FLOAT_EPS, 1.0)];
    let avg_alpha = mx_average_alpha(safe_alpha);
    let safe_tint = max3s(p.tint, 0.0);

    let x = normalize3(sub3(x, scale3(n, dot3(x, n))));
    let y = cross3(n, x);
    let h = normalize3(add3(l, v));

    let ndotl = clampf(dot3(n, l), MX_FLOAT_EPS, 1.0);
    let vdoth = clampf(dot3(v, h), MX_FLOAT_EPS, 1.0);

    let ht = [dot3(h, x), dot3(h, y), dot3(h, n)];

    let f = mx_compute_fresnel(vdoth, &fd);
    let d = mx_ggx_ndf(ht, ndf_alpha(safe_alpha, ndf_widen));
    let g = mx_ggx_smith_g2(ndotl, ndotv, avg_alpha);

    let comp = mx_ggx_energy_compensation(ndotv, avg_alpha, &fd);
    let dir_albedo = dielectric_dir_albedo(ndotv, avg_alpha, f0, &fd, comp, p.thin_film_energy);
    MxBsdf {
        response: div3s(
            scale3(
                mul3(mul3(scale3(scale3(f, d), g), comp), safe_tint),
                p.weight,
            ),
            4.0 * ndotv,
        ),
        throughput: sub3(splat3(1.0), scale3(dir_albedo, p.weight)),
    }
}

/// Dielectric GGX closure for environment light (`CLOSURE_TYPE_INDIRECT`):
/// `pbrlib/genglsl/mx_dielectric_bsdf.glsl:64-72` with `mx_environment_radiance`
/// (`lib/mx_environment_prefilter.glsl:10-23`) as `radiance * FG(fd)`. `radiance` is the
/// caller's prefiltered environment radiance along the lobe's reflection direction.
pub fn mx_dielectric_bsdf_indirect(
    v: [f32; 3],
    n: [f32; 3],
    p: &MxDielectric,
    radiance: [f32; 3],
) -> MxBsdf {
    if p.weight < MX_FLOAT_EPS {
        return MxBsdf::EMPTY;
    }
    let n = mx_forward_facing_normal(n, v);
    let ndotv = clampf(dot3(n, v), MX_FLOAT_EPS, 1.0);

    let fd = mx_init_fresnel_dielectric(p.ior, p.thinfilm_thickness, p.thinfilm_ior);
    let f0 = mx_ior_to_f0(p.ior);

    let [rx, ry] = p.roughness;
    let safe_alpha = [clampf(rx, MX_FLOAT_EPS, 1.0), clampf(ry, MX_FLOAT_EPS, 1.0)];
    let avg_alpha = mx_average_alpha(safe_alpha);
    let safe_tint = max3s(p.tint, 0.0);

    let comp = mx_ggx_energy_compensation(ndotv, avg_alpha, &fd);
    let dir_albedo = dielectric_dir_albedo(ndotv, avg_alpha, f0, &fd, comp, p.thin_film_energy);

    let li = mul3(
        radiance,
        film_dir_albedo(ndotv, avg_alpha, &fd, p.thin_film_energy),
    );
    MxBsdf {
        response: scale3(mul3(mul3(li, safe_tint), comp), p.weight),
        throughput: sub3(splat3(1.0), scale3(dir_albedo, p.weight)),
    }
}

/// Conductor GGX reflection closure for direct light: `pbrlib/genglsl/mx_conductor_bsdf.glsl:4-44`.
/// `response = D F G comp w / (4 NdotV)`; `throughput = 0` (nothing lies beneath a metal).
pub fn mx_conductor_bsdf_reflection(
    v: [f32; 3],
    l: [f32; 3],
    n: [f32; 3],
    x: [f32; 3],
    p: &MxConductor,
    ndf_widen: f32,
) -> MxBsdf {
    if p.weight < MX_FLOAT_EPS {
        return MxBsdf::default();
    }
    let n = mx_forward_facing_normal(n, v);
    let ndotv = clampf(dot3(n, v), MX_FLOAT_EPS, 1.0);

    let fd = mx_init_fresnel_conductor(p.ior_n, p.ior_k, p.thinfilm_thickness, p.thinfilm_ior);

    let [rx, ry] = p.roughness;
    let safe_alpha = [clampf(rx, MX_FLOAT_EPS, 1.0), clampf(ry, MX_FLOAT_EPS, 1.0)];
    let avg_alpha = mx_average_alpha(safe_alpha);

    let x = normalize3(sub3(x, scale3(n, dot3(x, n))));
    let y = cross3(n, x);
    let h = normalize3(add3(l, v));

    let ndotl = clampf(dot3(n, l), MX_FLOAT_EPS, 1.0);
    let vdoth = clampf(dot3(v, h), MX_FLOAT_EPS, 1.0);

    let ht = [dot3(h, x), dot3(h, y), dot3(h, n)];

    let f = mx_compute_fresnel(vdoth, &fd);
    let d = mx_ggx_ndf(ht, ndf_alpha(safe_alpha, ndf_widen));
    let g = mx_ggx_smith_g2(ndotl, ndotv, avg_alpha);

    let comp = mx_ggx_energy_compensation(ndotv, avg_alpha, &fd);
    MxBsdf {
        response: div3s(
            scale3(mul3(scale3(scale3(f, d), g), comp), p.weight),
            4.0 * ndotv,
        ),
        throughput: splat3(0.0),
    }
}

/// Conductor GGX closure for environment light: `pbrlib/genglsl/mx_conductor_bsdf.glsl:45-50`,
/// `response = radiance FG(fd) comp w`, `throughput = 0`.
pub fn mx_conductor_bsdf_indirect(
    v: [f32; 3],
    n: [f32; 3],
    p: &MxConductor,
    radiance: [f32; 3],
) -> MxBsdf {
    if p.weight < MX_FLOAT_EPS {
        return MxBsdf::default();
    }
    let n = mx_forward_facing_normal(n, v);
    let ndotv = clampf(dot3(n, v), MX_FLOAT_EPS, 1.0);

    let fd = mx_init_fresnel_conductor(p.ior_n, p.ior_k, p.thinfilm_thickness, p.thinfilm_ior);

    let [rx, ry] = p.roughness;
    let safe_alpha = [clampf(rx, MX_FLOAT_EPS, 1.0), clampf(ry, MX_FLOAT_EPS, 1.0)];
    let avg_alpha = mx_average_alpha(safe_alpha);

    let comp = mx_ggx_energy_compensation(ndotv, avg_alpha, &fd);
    let li = mul3(
        radiance,
        film_dir_albedo(ndotv, avg_alpha, &fd, p.thin_film_energy),
    );
    MxBsdf {
        response: scale3(mul3(li, comp), p.weight),
        throughput: splat3(0.0),
    }
}
