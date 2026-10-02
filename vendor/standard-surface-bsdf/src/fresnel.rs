//! Fresnel terms: dielectric, conductor, Hoffman F82 Schlick, Airy thin-film iridescence,
//! the artistic metal IOR, and the directional-albedo / energy-compensation helpers that take
//! a [`MxFresnelData`].
//!
//! Port of `pbrlib/genglsl/lib/mx_microfacet_specular.glsl` (MaterialX
//! `v1.39.5-22-g47cecce6`) and `pbrlib/genglsl/mx_artistic_ior.glsl`. The WGSL twin is
//! `wgsl/fresnel.wgsl`, statement for statement. One deliberate change: the Airy spectral
//! sensitivity reduces its cosine argument into `[-pi, pi]` ([`reduce_phase`]) because WGSL
//! `cos` has no useful accuracy bound at the ~500 rad arguments a 2000 nm film reaches.

use crate::consts::{
    MX_AIRY_FRESNEL_ITERATIONS, MX_AIRY_NM_TO_M, MX_AIRY_NORM, MX_AIRY_POS, MX_AIRY_VAL,
    MX_AIRY_VAR, MX_AIRY_X2_POS, MX_AIRY_X2_VAL, MX_AIRY_X2_VAR, MX_COS_THETA_FACTOR,
    MX_COS_THETA_MAX, MX_F0_TO_IOR_MAX, MX_F0_TO_IOR_MIN, MX_FRESNEL_AVERAGE_FACTOR,
    MX_FRESNEL_MODEL_CONDUCTOR, MX_FRESNEL_MODEL_DIELECTRIC, MX_FRESNEL_MODEL_SCHLICK, MX_PI,
    MX_TWO_PI, MX_XYZ_TO_RGB_R0, MX_XYZ_TO_RGB_R1, MX_XYZ_TO_RGB_R2, SS_INV_TWO_PI,
};
use crate::math::{
    add3, clamp3, clampf, div3, div3s, dot3, max3s, mix3, mix3v, mul3, mx_pow6, mx_square, scale3,
    splat3, sqrt3, sub3,
};
use crate::microfacet::{ThinFilmEnergy, mx_ggx_dir_albedo_analytic, mx_ggx_dir_albedo_scalar};

/// WGSL functions of `wgsl/fresnel.wgsl`, sorted: the twins of this module's functions
/// (`mx_` names kept, `ss_` names are the Rust names with the prefix). Checked by the
/// `twin_registry_matches_wgsl_and_rust` test; collected by [`crate::wgsl::TWIN_FUNCTIONS`].
pub const WGSL_TWINS: &[&str] = &[
    "mx_artistic_ior",
    "mx_compute_fresnel",
    "mx_eval_sensitivity",
    "mx_f0_to_ior_vec3",
    "mx_fresnel_airy",
    "mx_fresnel_average",
    "mx_fresnel_conductor",
    "mx_fresnel_conductor_phase_polarized",
    "mx_fresnel_conductor_polarized",
    "mx_fresnel_dielectric",
    "mx_fresnel_dielectric_polarized",
    "mx_fresnel_hoffman_schlick",
    "mx_fresnel_schlick_exp",
    "mx_ggx_dir_albedo_fd",
    "mx_ggx_energy_compensation",
    "mx_init_fresnel_conductor",
    "mx_init_fresnel_dielectric",
    "mx_init_fresnel_schlick",
    "mx_ior_to_f0",
    "ss_airy_gaussian",
    "ss_film_dir_albedo",
    "ss_reduce_phase",
];

/// Parameters of a Fresnel evaluation: MaterialX `struct FresnelData`
/// (`lib/mx_microfacet_specular.glsl:8-30`). The `refraction` member is omitted because the
/// shared BSDF has no transmission lobe (plan §2, Q7).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MxFresnelData {
    /// One of `MX_FRESNEL_MODEL_{DIELECTRIC, CONDUCTOR, SCHLICK}`.
    pub model: u32,
    /// Thin-film (Airy) evaluation, `tf_thickness > 0`.
    pub airy: bool,
    /// Real IOR (dielectric uses `.x`; conductor per channel).
    pub ior: [f32; 3],
    /// Conductor extinction coefficient `k`.
    pub extinction: [f32; 3],
    /// Generalized Schlick `F0`.
    pub f0: [f32; 3],
    /// Generalized Schlick `F82` tint.
    pub f82: [f32; 3],
    /// Generalized Schlick `F90`.
    pub f90: [f32; 3],
    /// Generalized Schlick exponent.
    pub exponent: f32,
    /// Thin-film thickness in nanometres.
    pub tf_thickness: f32,
    /// Thin-film IOR.
    pub tf_ior: f32,
}

/// A polarised pair of RGB quantities (parallel `p`, perpendicular `s`). Stands in for the two
/// `out vec3` parameters of `mx_fresnel_conductor_polarized` and
/// `mx_fresnel_conductor_phase_polarized` (`lib/mx_microfacet_specular.glsl:246,273`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MxPolarized {
    /// Parallel polarisation.
    pub p: [f32; 3],
    /// Perpendicular polarisation.
    pub s: [f32; 3],
}

/// Output of [`mx_artistic_ior`]: the complex IOR `n + ik` of an artist-friendly metal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MxArtisticIor {
    /// Real part `n`.
    pub ior: [f32; 3],
    /// Extinction `k`.
    pub extinction: [f32; 3],
}

/// Normal-incidence reflectance of a real IOR, `((ior - 1) / (ior + 1))^2`:
/// `lib/mx_microfacet_specular.glsl:183-186`.
pub fn mx_ior_to_f0(ior: f32) -> f32 {
    mx_square((ior - 1.0) / (ior + 1.0))
}

/// Inverse of [`mx_ior_to_f0`] with `F0` clamped to `[0.01, 0.99]`, per channel:
/// `lib/mx_microfacet_specular.glsl:194-198`. Used by the Airy path for a Schlick substrate.
pub fn mx_f0_to_ior_vec3(f0: [f32; 3]) -> [f32; 3] {
    let sqrt_f0 = sqrt3(clamp3(f0, MX_F0_TO_IOR_MIN, MX_F0_TO_IOR_MAX));
    div3(add3(splat3(1.0), sqrt_f0), sub3(splat3(1.0), sqrt_f0))
}

/// Generalized Schlick Fresnel with a variable exponent, `mix(F0, F90, pow(1 - cos, e))`:
/// `pbrlib/genglsl/lib/mx_microfacet.glsl:49-53`. Used by the coat emission EDF.
pub fn mx_fresnel_schlick_exp(
    cos_theta: f32,
    f0: [f32; 3],
    f90: [f32; 3],
    exponent: f32,
) -> [f32; 3] {
    let x = clampf(1.0 - cos_theta, 0.0, 1.0);
    mix3(f0, f90, crate::fm::powf(x, exponent))
}

/// Hoffman's F82-tinted Schlick (the generalized Schlick model):
/// `lib/mx_microfacet_specular.glsl:200-209`.
pub fn mx_fresnel_hoffman_schlick(cos_theta: f32, fd: &MxFresnelData) -> [f32; 3] {
    let x = clampf(cos_theta, 0.0, 1.0);
    let a = scale3(
        mul3(
            mix3(
                fd.f0,
                fd.f90,
                crate::fm::powf(1.0 - MX_COS_THETA_MAX, fd.exponent),
            ),
            sub3(splat3(1.0), fd.f82),
        ),
        MX_COS_THETA_FACTOR,
    );
    sub3(
        mix3(fd.f0, fd.f90, crate::fm::powf(1.0 - x, fd.exponent)),
        scale3(scale3(a, x), mx_pow6(1.0 - x)),
    )
}

/// Unpolarised dielectric Fresnel reflectance (1 on total internal reflection):
/// `lib/mx_microfacet_specular.glsl:211-225`.
pub fn mx_fresnel_dielectric(cos_theta: f32, ior: f32) -> f32 {
    let c = cos_theta;
    let g2 = ior * ior + c * c - 1.0;
    if g2 < 0.0 {
        // Total internal reflection.
        return 1.0;
    }
    let g = crate::fm::sqrtf(g2);
    0.5 * mx_square((g - c) / (g + c))
        * (1.0 + mx_square(((g + c) * c - 1.0) / ((g - c) * c + 1.0)))
}

/// Polarised dielectric Fresnel reflectance `[Rp, Rs]`:
/// `lib/mx_microfacet_specular.glsl:227-243`.
pub fn mx_fresnel_dielectric_polarized(cos_theta: f32, ior: f32) -> [f32; 2] {
    let cos_theta2 = mx_square(clampf(cos_theta, 0.0, 1.0));
    let sin_theta2 = 1.0 - cos_theta2;

    let t0 = (ior * ior - sin_theta2).max(0.0);
    let t1 = t0 + cos_theta2;
    let t2 = 2.0 * crate::fm::sqrtf(t0) * cos_theta;
    let rs = (t1 - t2) / (t1 + t2);

    let t3 = cos_theta2 * t0 + sin_theta2 * sin_theta2;
    let t4 = t2 * sin_theta2;
    let rp = rs * (t3 - t4) / (t3 + t4);

    [rp, rs]
}

/// Polarised conductor Fresnel reflectance for the complex IOR `n + ik`:
/// `lib/mx_microfacet_specular.glsl:245-263`.
pub fn mx_fresnel_conductor_polarized(cos_theta: f32, n: [f32; 3], k: [f32; 3]) -> MxPolarized {
    let cos_theta2 = mx_square(clampf(cos_theta, 0.0, 1.0));
    let sin_theta2 = 1.0 - cos_theta2;
    let n2 = mul3(n, n);
    let k2 = mul3(k, k);

    let t0 = sub3(sub3(n2, k2), splat3(sin_theta2));
    let a2plusb2 = sqrt3(add3(mul3(t0, t0), mul3(scale3(n2, 4.0), k2)));
    let t1 = add3(a2plusb2, splat3(cos_theta2));
    let a = sqrt3(max3s(scale3(add3(a2plusb2, t0), 0.5), 0.0));
    let t2 = scale3(scale3(a, 2.0), cos_theta);
    let rs = div3(sub3(t1, t2), add3(t1, t2));

    let t3 = add3(
        scale3(a2plusb2, cos_theta2),
        splat3(sin_theta2 * sin_theta2),
    );
    let t4 = scale3(t2, sin_theta2);
    let rp = div3(mul3(rs, sub3(t3, t4)), add3(t3, t4));

    MxPolarized { p: rp, s: rs }
}

/// Unpolarised conductor Fresnel reflectance `0.5 (Rp + Rs)`:
/// `lib/mx_microfacet_specular.glsl:265-270`.
pub fn mx_fresnel_conductor(cos_theta: f32, n: [f32; 3], k: [f32; 3]) -> [f32; 3] {
    let r = mx_fresnel_conductor_polarized(cos_theta, n, k);
    scale3(add3(r.p, r.s), 0.5)
}

/// Polarised phase shift at a conductor interface (Belcour & Barla 2017):
/// `lib/mx_microfacet_specular.glsl:272-285`.
pub fn mx_fresnel_conductor_phase_polarized(
    cos_theta: f32,
    eta1: f32,
    eta2: [f32; 3],
    kappa2: [f32; 3],
) -> MxPolarized {
    let k2 = div3(kappa2, eta2);
    let sin_theta_sqr = sub3(splat3(1.0), splat3(cos_theta * cos_theta));
    let one_minus_k2k2 = sub3(splat3(1.0), mul3(k2, k2));
    let a = sub3(
        mul3(mul3(eta2, eta2), one_minus_k2k2),
        scale3(sin_theta_sqr, eta1 * eta1),
    );
    let two_eta2_eta2_k2 = mul3(mul3(scale3(eta2, 2.0), eta2), k2);
    let b = sqrt3(add3(mul3(a, a), mul3(two_eta2_eta2_k2, two_eta2_eta2_k2)));
    let u = sqrt3(div3s(add3(a, b), 2.0));
    let v = max3s(sqrt3(div3s(sub3(b, a), 2.0)), 0.0);

    let uu_vv = add3(mul3(u, u), mul3(v, v));
    let phi_s = crate::math::atan2_3(
        scale3(scale3(v, 2.0 * eta1), cos_theta),
        sub3(uu_vv, splat3(mx_square(eta1 * cos_theta))),
    );
    let one_plus_k2k2 = add3(splat3(1.0), mul3(k2, k2));
    let phi_p_y = mul3(
        scale3(mul3(scale3(eta2, 2.0 * eta1), eta2), cos_theta),
        sub3(mul3(scale3(k2, 2.0), u), mul3(one_minus_k2k2, v)),
    );
    let phi_p_x_base = scale3(mul3(mul3(eta2, eta2), one_plus_k2k2), cos_theta);
    let phi_p = crate::math::atan2_3(
        phi_p_y,
        sub3(mul3(phi_p_x_base, phi_p_x_base), scale3(uu_vv, eta1 * eta1)),
    );
    MxPolarized { p: phi_p, s: phi_s }
}

/// Reduces a phase to `[-pi, pi]`: `x - 2 pi * floor(x / (2 pi) + 0.5)` (plan §2, "Airy
/// phase"). `floor(x + 0.5)` is used on both sides because WGSL `round` is half-to-even and Rust
/// `round` is half-away-from-zero; `1 / (2 pi)` is a multiply by [`SS_INV_TWO_PI`].
pub fn reduce_phase(x: f32) -> f32 {
    x - MX_TWO_PI * crate::fm::floorf(x * SS_INV_TWO_PI + 0.5)
}

/// One Gaussian of the Airy spectral sensitivity fit,
/// `val sqrt(2 pi var) cos(pos phase + shift) exp(-var phase^2)`
/// (`lib/mx_microfacet_specular.glsl:295-296`), cosine argument reduced by [`reduce_phase`].
pub fn airy_gaussian(val: f32, pos: f32, var: f32, phase: f32, shift: f32) -> f32 {
    val * crate::fm::sqrtf(MX_TWO_PI * var)
        * crate::fm::cosf(reduce_phase(pos * phase + shift))
        * crate::fm::expf(-var * phase * phase)
}

/// Airy XYZ spectral sensitivity for an optical path difference `opd` (metres) and phase
/// `shift`: `lib/mx_microfacet_specular.glsl:287-298`, with every cosine argument reduced by
/// [`reduce_phase`].
pub fn mx_eval_sensitivity(opd: f32, shift: [f32; 3]) -> [f32; 3] {
    // Gaussian fits, given by 3 parameters: val, pos and var.
    let phase = MX_TWO_PI * opd;
    let [val_x, val_y, val_z] = MX_AIRY_VAL;
    let [pos_x, pos_y, pos_z] = MX_AIRY_POS;
    let [var_x, var_y, var_z] = MX_AIRY_VAR;
    let [shift_x, shift_y, shift_z] = shift;
    let xyz = [
        airy_gaussian(val_x, pos_x, var_x, phase, shift_x),
        airy_gaussian(val_y, pos_y, var_y, phase, shift_y),
        airy_gaussian(val_z, pos_z, var_z, phase, shift_z),
    ];
    let [x, y, z] = xyz;
    let xyz = [
        x + airy_gaussian(
            MX_AIRY_X2_VAL,
            MX_AIRY_X2_POS,
            MX_AIRY_X2_VAR,
            phase,
            shift_x,
        ),
        y,
        z,
    ];
    div3s(xyz, MX_AIRY_NORM)
}

/// Thin-film iridescence (Belcour & Barla 2017), `AIRY_FRESNEL_ITERATIONS` = 2 orders per
/// polarisation, returned in CIE 1931 RGB (illuminant E) clamped to `[0, 1]`:
/// `lib/mx_microfacet_specular.glsl:300-399`.
pub fn mx_fresnel_airy(cos_theta: f32, fd: &MxFresnelData) -> [f32; 3] {
    // Assume vacuum on the outside.
    let eta1 = 1.0_f32;
    let eta2 = fd.tf_ior.max(eta1);
    let schlick = fd.model == MX_FRESNEL_MODEL_SCHLICK;
    let eta3 = if schlick {
        mx_f0_to_ior_vec3(fd.f0)
    } else {
        fd.ior
    };
    let kappa3 = if schlick { splat3(0.0) } else { fd.extinction };
    let cos_theta_t = crate::fm::sqrtf(1.0 - (1.0 - mx_square(cos_theta)) * mx_square(eta1 / eta2));

    // First interface.
    let mut r12 = mx_fresnel_dielectric_polarized(cos_theta, eta2 / eta1);
    if cos_theta_t <= 0.0 {
        // Total internal reflection.
        r12 = [1.0, 1.0];
    }
    let [r12_p, r12_s] = r12;
    let t121_p = 1.0 - r12_p;
    let t121_s = 1.0 - r12_s;

    // Second interface.
    let r23 = if schlick {
        let f = mx_fresnel_hoffman_schlick(cos_theta_t, fd);
        MxPolarized {
            p: scale3(f, 0.5),
            s: scale3(f, 0.5),
        }
    } else {
        mx_fresnel_conductor_polarized(cos_theta_t, div3s(eta3, eta2), div3s(kappa3, eta2))
    };

    // Phase shift.
    let cos_b = crate::fm::cosf(crate::fm::atanf(eta2 / eta1));
    let phi21_p = if cos_theta < cos_b { 0.0 } else { MX_PI };
    let phi21_s = MX_PI;
    let phi23 = if schlick {
        let [e3x, e3y, e3z] = eta3;
        let p = [
            if e3x < eta2 { MX_PI } else { 0.0 },
            if e3y < eta2 { MX_PI } else { 0.0 },
            if e3z < eta2 { MX_PI } else { 0.0 },
        ];
        MxPolarized { p, s: p }
    } else {
        mx_fresnel_conductor_phase_polarized(cos_theta_t, eta2, eta3, kappa3)
    };
    let r123p = max3s(sqrt3(scale3(r23.p, r12_p)), 0.0);
    let r123s = max3s(sqrt3(scale3(r23.s, r12_s)), 0.0);

    // Iridescence term.
    let mut i = splat3(0.0);

    // Optical path difference.
    let dist_meters = fd.tf_thickness * MX_AIRY_NM_TO_M;
    let opd = 2.0 * eta2 * cos_theta_t * dist_meters;

    // Parallel polarisation: reflectance term for m = 0 (DC term amplitude).
    let rs = div3(
        scale3(r23.p, mx_square(t121_p)),
        sub3(splat3(1.0), scale3(r23.p, r12_p)),
    );
    i = add3(i, add3(splat3(r12_p), rs));

    // Reflectance terms for m > 0 (pairs of diracs).
    let mut cm = sub3(rs, splat3(t121_p));
    for m in 1..=MX_AIRY_FRESNEL_ITERATIONS {
        let mf = m as f32;
        cm = mul3(cm, r123p);
        let sm = scale3(
            mx_eval_sensitivity(mf * opd, scale3(add3(phi23.p, splat3(phi21_p)), mf)),
            2.0,
        );
        i = add3(i, mul3(cm, sm));
    }

    // Perpendicular polarisation: reflectance term for m = 0 (DC term amplitude).
    let rp = div3(
        scale3(r23.s, mx_square(t121_s)),
        sub3(splat3(1.0), scale3(r23.s, r12_s)),
    );
    i = add3(i, add3(splat3(r12_s), rp));

    // Reflectance terms for m > 0 (pairs of diracs).
    cm = sub3(rp, splat3(t121_s));
    for m in 1..=MX_AIRY_FRESNEL_ITERATIONS {
        let mf = m as f32;
        cm = mul3(cm, r123s);
        let sm = scale3(
            mx_eval_sensitivity(mf * opd, scale3(add3(phi23.s, splat3(phi21_s)), mf)),
            2.0,
        );
        i = add3(i, mul3(cm, sm));
    }

    // Average parallel and perpendicular polarisation.
    i = scale3(i, 0.5);

    // Convert back to RGB reflectance.
    clamp3(
        [
            dot3(MX_XYZ_TO_RGB_R0, i),
            dot3(MX_XYZ_TO_RGB_R1, i),
            dot3(MX_XYZ_TO_RGB_R2, i),
        ],
        0.0,
        1.0,
    )
}

/// Fresnel data for a dielectric of real IOR `ior`: `lib/mx_microfacet_specular.glsl:401-416`.
pub fn mx_init_fresnel_dielectric(ior: f32, tf_thickness: f32, tf_ior: f32) -> MxFresnelData {
    MxFresnelData {
        model: MX_FRESNEL_MODEL_DIELECTRIC,
        airy: tf_thickness > 0.0,
        ior: splat3(ior),
        extinction: splat3(0.0),
        f0: splat3(0.0),
        f82: splat3(0.0),
        f90: splat3(0.0),
        exponent: 0.0,
        tf_thickness,
        tf_ior,
    }
}

/// Fresnel data for a conductor of complex IOR `ior + i extinction`:
/// `lib/mx_microfacet_specular.glsl:418-433`.
pub fn mx_init_fresnel_conductor(
    ior: [f32; 3],
    extinction: [f32; 3],
    tf_thickness: f32,
    tf_ior: f32,
) -> MxFresnelData {
    MxFresnelData {
        model: MX_FRESNEL_MODEL_CONDUCTOR,
        airy: tf_thickness > 0.0,
        ior,
        extinction,
        f0: splat3(0.0),
        f82: splat3(0.0),
        f90: splat3(0.0),
        exponent: 0.0,
        tf_thickness,
        tf_ior,
    }
}

/// Fresnel data for the generalized (F82) Schlick model:
/// `lib/mx_microfacet_specular.glsl:435-450`.
pub fn mx_init_fresnel_schlick(
    f0: [f32; 3],
    f82: [f32; 3],
    f90: [f32; 3],
    exponent: f32,
    tf_thickness: f32,
    tf_ior: f32,
) -> MxFresnelData {
    MxFresnelData {
        model: MX_FRESNEL_MODEL_SCHLICK,
        airy: tf_thickness > 0.0,
        ior: splat3(0.0),
        extinction: splat3(0.0),
        f0,
        f82,
        f90,
        exponent,
        tf_thickness,
        tf_ior,
    }
}

/// Fresnel reflectance for any model, Airy first: `lib/mx_microfacet_specular.glsl:452-470`.
pub fn mx_compute_fresnel(cos_theta: f32, fd: &MxFresnelData) -> [f32; 3] {
    if fd.airy {
        mx_fresnel_airy(cos_theta, fd)
    } else if fd.model == MX_FRESNEL_MODEL_DIELECTRIC {
        let [ior, _, _] = fd.ior;
        splat3(mx_fresnel_dielectric(cos_theta, ior))
    } else if fd.model == MX_FRESNEL_MODEL_CONDUCTOR {
        mx_fresnel_conductor(cos_theta, fd.ior, fd.extinction)
    } else {
        mx_fresnel_hoffman_schlick(cos_theta, fd)
    }
}

/// GGX directional albedo for any Fresnel model (the prefiltered-environment `FG` term):
/// `lib/mx_microfacet_specular.glsl:472-499`. With a thin film it blends the mirror Fresnel
/// and the rough fit by `sqrt(alpha)`.
pub fn mx_ggx_dir_albedo_fd(ndotv: f32, alpha: f32, fd: &MxFresnelData) -> [f32; 3] {
    if fd.airy {
        // Approximation using a blend between mirror (alpha = 0) and rougher cases.
        let mirror_dir_albedo = mx_compute_fresnel(ndotv, fd);
        let f0 = mx_fresnel_airy(1.0, fd);
        let rough_dir_albedo = mx_ggx_dir_albedo_analytic(ndotv, alpha, f0, splat3(1.0));
        mix3(mirror_dir_albedo, rough_dir_albedo, crate::fm::sqrtf(alpha))
    } else if fd.model == MX_FRESNEL_MODEL_DIELECTRIC {
        let [ior, _, _] = fd.ior;
        let f0 = mx_ior_to_f0(ior);
        mx_ggx_dir_albedo_analytic(ndotv, alpha, splat3(f0), splat3(1.0))
    } else if fd.model == MX_FRESNEL_MODEL_CONDUCTOR {
        let f0 = mx_fresnel_conductor(1.0, fd.ior, fd.extinction);
        mx_ggx_dir_albedo_analytic(ndotv, alpha, f0, splat3(1.0))
    } else {
        mx_ggx_dir_albedo_analytic(ndotv, alpha, fd.f0, fd.f90)
    }
}

/// The directional albedo `FG` of a GGX lobe with Fresnel data `fd`, per thin-film energy
/// model; the one albedo behind the environment response of both GGX closures and the
/// Conserving dielectric throughput (tasks R2b/R2c).
///
/// - [`ThinFilmEnergy::MaterialX`] is [`mx_ggx_dir_albedo_fd`] exactly.
/// - [`ThinFilmEnergy::Conserving`] with a film fixes the cause of `FG comp > 1`: MaterialX's
///   Airy branch (`lib/mx_microfacet_specular.glsl:475-483`) blends the **unshadowed** mirror
///   reflectance `F_airy(NdotV)` with the rough fit by `sqrt(alpha)`. That mirror term is the
///   albedo of an `alpha = 0` lobe; for a rough lobe it exceeds the GGX single-scatter energy
///   `Ess(NdotV, alpha)` (the albedo with `F = 1`), and the Turquin compensation
///   (`:516-521`, `comp = 1 + F_avg (1 - Ess) / Ess`) then divides by `Ess` a second time, so a
///   white conductor reaches `FG comp = 1.13`. Here the mirror term is weighted by `Ess`
///   (`F_airy(NdotV) Ess`, the single-scatter albedo of a lobe whose Fresnel is the film's
///   reflectance at the view angle). The rough term `F0 A + B` already satisfies the same
///   bound, so `FG <= Ess max(F)` and `FG comp <= Ess + F_avg (1 - Ess) <= 1` (up to the
///   fit's own overshoot of `Ess`); no clamp is involved. As `alpha -> 0` (`Ess -> 1`) it
///   tends to MaterialX's value. Without a film it is MaterialX exactly.
pub fn film_dir_albedo(
    ndotv: f32,
    alpha: f32,
    fd: &MxFresnelData,
    energy: ThinFilmEnergy,
) -> [f32; 3] {
    match energy {
        ThinFilmEnergy::Conserving if fd.airy => {
            let ess = mx_ggx_dir_albedo_scalar(ndotv, alpha, 1.0, 1.0);
            let mirror_dir_albedo = scale3(mx_compute_fresnel(ndotv, fd), ess);
            let f0 = mx_fresnel_airy(1.0, fd);
            let rough_dir_albedo = mx_ggx_dir_albedo_analytic(ndotv, alpha, f0, splat3(1.0));
            mix3(mirror_dir_albedo, rough_dir_albedo, crate::fm::sqrtf(alpha))
        }
        ThinFilmEnergy::MaterialX | ThinFilmEnergy::Conserving => {
            mx_ggx_dir_albedo_fd(ndotv, alpha, fd)
        }
    }
}

/// Cosine-weighted hemispherical average of the Fresnel reflectance, `F0 + (F90 - F0) / 21`:
/// `lib/mx_microfacet_specular.glsl:501-511`.
pub fn mx_fresnel_average(fd: &MxFresnelData) -> [f32; 3] {
    let f0 = mx_compute_fresnel(1.0, fd);
    let f90 = if fd.model == MX_FRESNEL_MODEL_SCHLICK && !fd.airy {
        fd.f90
    } else {
        splat3(1.0)
    };
    add3(f0, scale3(sub3(f90, f0), MX_FRESNEL_AVERAGE_FACTOR))
}

/// Turquin multiple-scattering energy compensation, `1 + F_avg (1 - E) / E`:
/// `lib/mx_microfacet_specular.glsl:513-521`.
pub fn mx_ggx_energy_compensation(ndotv: f32, alpha: f32, fd: &MxFresnelData) -> [f32; 3] {
    let fss = mx_fresnel_average(fd);
    let ess = mx_ggx_dir_albedo_scalar(ndotv, alpha, 1.0, 1.0);
    add3(splat3(1.0), div3s(scale3(fss, 1.0 - ess), ess))
}

/// Gulbrandsen's artist-friendly metallic Fresnel: complex IOR from a reflectivity and an edge
/// colour, reflectivity clamped to `[0, 0.99]`: `pbrlib/genglsl/mx_artistic_ior.glsl:1-17`.
pub fn mx_artistic_ior(reflectivity: [f32; 3], edge_color: [f32; 3]) -> MxArtisticIor {
    let r = clamp3(reflectivity, 0.0, MX_F0_TO_IOR_MAX);
    let r_sqrt = sqrt3(r);
    let n_min = div3(sub3(splat3(1.0), r), add3(splat3(1.0), r));
    let n_max = div3(add3(splat3(1.0), r_sqrt), sub3(splat3(1.0), r_sqrt));
    let ior = mix3v(n_max, n_min, edge_color);

    let np1 = add3(ior, splat3(1.0));
    let nm1 = sub3(ior, splat3(1.0));
    let k2 = div3(
        sub3(mul3(mul3(np1, np1), r), mul3(nm1, nm1)),
        sub3(splat3(1.0), r),
    );
    let k2 = max3s(k2, 0.0);
    MxArtisticIor {
        ior,
        extinction: sqrt3(k2),
    }
}
