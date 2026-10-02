//! Every numeric constant of the BSDF, defined exactly once.
//!
//! The Rust twin reads these items directly. The WGSL twin receives them through
//! [`wgsl_header`], which emits one `const` per entry of [`WGSL_CONSTANTS`] with a
//! shortest-round-trip `f`-suffixed literal, so both sides use identical bits (the
//! `constants_have_identical_bits_in_wgsl` test parses the header with naga and compares
//! `to_bits()`). Fit tables are copied verbatim from MaterialX `v1.39.5-22-g47cecce6`
//! (`vfx.ref/MaterialX/libraries`); every item cites its source line.

/// `M_PI`: `stdlib/genglsl/lib/mx_math.glsl:2`, `pbrlib/genglsl/lib/mx_microfacet.glsl:1`.
pub const MX_PI: f32 = core::f32::consts::PI;
/// `M_PI_INV = 1 / M_PI`: `pbrlib/genglsl/lib/mx_microfacet.glsl:2`.
pub const MX_PI_INV: f32 = 1.0 / MX_PI;
/// `2 * M_PI`, the Airy sensitivity phase factor (`lib/mx_microfacet_specular.glsl:291`) and
/// the period of [`crate::fresnel::reduce_phase`].
pub const MX_TWO_PI: f32 = 2.0 * MX_PI;
/// `1 / (2 * M_PI)`, used by the Airy phase range reduction ([`crate::fresnel::reduce_phase`]).
pub const SS_INV_TWO_PI: f32 = 1.0 / MX_TWO_PI;
/// Degrees to radians, `mx_radians` of `stdlib/genglsl/mx_rotate_vector3.glsl:8`, written as an
/// explicit multiply so both twins use the same arithmetic (WGSL `radians()` is
/// implementation-defined).
pub const MX_DEG_TO_RAD: f32 = MX_PI / 180.0;
/// `M_FLOAT_EPS`: `stdlib/genglsl/lib/mx_math.glsl:1`. Lower clamp of cosines, squared roughness
/// (`mx_roughness_anisotropy.glsl:3`) and closure weights.
pub const MX_FLOAT_EPS: f32 = 1e-8;

/// A GGX lobe whose `max(alpha_x, alpha_y)` is below this is a **delta** lobe
/// ([`crate::microfacet::is_delta`]): its response is zero in `eval_light`, while its
/// throughput and selection weight are unchanged. `1e-3` corresponds to roughness ~0.032.
/// Plan `2026-09-23-standard-surface-bsdf.md` §2 (C2); value **\[R\]**.
pub const SS_ALPHA_MIN: f32 = 1e-3;
/// Squared length below which a projected shading tangent is degenerate (`t` parallel to `n`)
/// and [`crate::microfacet::mx_orthonormal_basis`] supplies the tangent instead.
pub const SS_TANGENT_MIN_LEN2: f32 = 1e-12;
/// Upper clamp of the anisotropy in `mx_roughness_anisotropy`:
/// `pbrlib/genglsl/mx_roughness_anisotropy.glsl:6`.
pub const MX_ANISOTROPY_MAX: f32 = 0.98;
/// Default `thinfilm_ior` of `dielectric_bsdf`, used by the (film-free) coat closure:
/// `pbrlib/pbrlib_defs.mtlx:69`.
pub const MX_THINFILM_IOR_DEFAULT: f32 = 1.5;
/// Exponent of the coat-tinted `generalized_schlick_edf`: `bxdf/standard_surface.mtlx:402`.
pub const MX_COAT_EMISSION_EXPONENT: f32 = 5.0;
/// Luminance coefficients used for the lobe-selection weights: the MaterialX `luminance`
/// default (ACEScg), `stdlib/stdlib_defs.mtlx:3246`.
pub const SS_LUMA: [f32; 3] = [0.272_228_7, 0.674_081_8, 0.053_689_5];

/// GGX directional-albedo rational fit, term `1`: `lib/mx_microfacet_specular.glsl:97`.
pub const MX_GGX_ALBEDO_C0: [f32; 4] = [0.1003, 0.9345, 1.0, 1.0];
/// GGX albedo fit, term `x`: `lib/mx_microfacet_specular.glsl:98`.
pub const MX_GGX_ALBEDO_C1: [f32; 4] = [-0.6303, -2.323, -1.765, 0.2281];
/// GGX albedo fit, term `y`: `lib/mx_microfacet_specular.glsl:99`.
pub const MX_GGX_ALBEDO_C2: [f32; 4] = [9.748, 2.229, 8.263, 15.94];
/// GGX albedo fit, term `x*y`: `lib/mx_microfacet_specular.glsl:100`.
pub const MX_GGX_ALBEDO_C3: [f32; 4] = [-2.038, -3.748, 11.53, -55.83];
/// GGX albedo fit, term `x^2`: `lib/mx_microfacet_specular.glsl:101`.
pub const MX_GGX_ALBEDO_C4: [f32; 4] = [29.34, 1.424, 28.96, 13.08];
/// GGX albedo fit, term `y^2`: `lib/mx_microfacet_specular.glsl:102`.
pub const MX_GGX_ALBEDO_C5: [f32; 4] = [-8.245, -0.7684, -7.507, 41.26];
/// GGX albedo fit, term `x^2*y`: `lib/mx_microfacet_specular.glsl:103`.
pub const MX_GGX_ALBEDO_C6: [f32; 4] = [-26.44, 1.436, -36.11, 54.9];
/// GGX albedo fit, term `x*y^2`: `lib/mx_microfacet_specular.glsl:104`.
pub const MX_GGX_ALBEDO_C7: [f32; 4] = [19.99, 0.2913, 15.86, 300.2];
/// GGX albedo fit, term `x^2*y^2`: `lib/mx_microfacet_specular.glsl:105`.
pub const MX_GGX_ALBEDO_C8: [f32; 4] = [-5.448, 0.6286, 33.37, -285.1];

/// Oren-Nayar `A` denominator offset `0.33`: `lib/mx_microfacet_diffuse.glsl:14`.
pub const MX_ON_A_OFFSET: f32 = 0.33;
/// Oren-Nayar `B` scale `0.45`: `lib/mx_microfacet_diffuse.glsl:15`.
pub const MX_ON_B_SCALE: f32 = 0.45;
/// Oren-Nayar `B` denominator offset `0.09`: `lib/mx_microfacet_diffuse.glsl:15`.
pub const MX_ON_B_OFFSET: f32 = 0.09;
/// Oren-Nayar directional-albedo fit, term `1`: `lib/mx_microfacet_diffuse.glsl:23`.
pub const MX_ON_ALBEDO_C0: [f32; 2] = [1.0, 1.0];
/// Oren-Nayar albedo fit, term `r`: `lib/mx_microfacet_diffuse.glsl:24`.
pub const MX_ON_ALBEDO_C1: [f32; 2] = [-0.4297, -0.6076];
/// Oren-Nayar albedo fit, term `NdotV*r`: `lib/mx_microfacet_diffuse.glsl:25`.
pub const MX_ON_ALBEDO_C2: [f32; 2] = [-0.7632, -0.4993];
/// Oren-Nayar albedo fit, term `r^2`: `lib/mx_microfacet_diffuse.glsl:26`.
pub const MX_ON_ALBEDO_C3: [f32; 2] = [1.4385, 2.0315];
/// Fujii constant `0.5 - 2/(3 pi)`: `lib/mx_microfacet_diffuse.glsl:3`.
pub const MX_FUJII_CONSTANT_1: f32 = 0.5 - 2.0 / (3.0 * MX_PI);
/// Fujii constant `2/3 - 28/(15 pi)`: `lib/mx_microfacet_diffuse.glsl:4`.
pub const MX_FUJII_CONSTANT_2: f32 = 2.0 / 3.0 - 28.0 / (15.0 * MX_PI);

/// Samples of the Burley diffusion integration over the curvature circle, `SAMPLE_COUNT`:
/// `lib/mx_microfacet_diffuse.glsl:180`.
pub const MX_BURLEY_SAMPLE_COUNT: i32 = 32;
/// Angular width of one Burley integration sample, `SAMPLE_WIDTH = (2 pi) / SAMPLE_COUNT`:
/// `lib/mx_microfacet_diffuse.glsl:181`.
pub const MX_BURLEY_SAMPLE_WIDTH: f32 = (2.0 * MX_PI) / 32.0;
/// Lower clamp of the mean free path in the Burley shape `1 / max(mfp, 0.1)`:
/// `lib/mx_microfacet_diffuse.glsl:175`.
pub const MX_BURLEY_MFP_MIN: f32 = 0.1;
/// Lower clamp of the surface curvature, `radius = 1 / max(curvature, 0.01)`:
/// `lib/mx_microfacet_diffuse.glsl:197`.
pub const MX_SUBSURFACE_CURVATURE_MIN: f32 = 0.01;

/// Lower clamp of the Imageworks sheen roughness: `lib/mx_microfacet_sheen.glsl:7`.
pub const MX_SHEEN_ROUGHNESS_MIN: f32 = 0.005;
/// Imageworks sheen albedo fit, term `1`: `lib/mx_microfacet_sheen.glsl:30`.
pub const MX_SHEEN_ALBEDO_C0: [f32; 2] = [13.673, 1.0];
/// Sheen albedo fit, term `NdotV`: `lib/mx_microfacet_sheen.glsl:31`.
pub const MX_SHEEN_ALBEDO_C1: [f32; 2] = [-68.78018, 61.57746];
/// Sheen albedo fit, term `r`: `lib/mx_microfacet_sheen.glsl:32`.
#[allow(
    clippy::excessive_precision,
    reason = "verbatim MaterialX digits; f32 rounds them exactly as GLSL float does"
)]
pub const MX_SHEEN_ALBEDO_C2: [f32; 2] = [799.08825, 442.78211];
/// Sheen albedo fit, term `NdotV*r`: `lib/mx_microfacet_sheen.glsl:33`.
#[allow(
    clippy::excessive_precision,
    reason = "verbatim MaterialX digits; f32 rounds them exactly as GLSL float does"
)]
pub const MX_SHEEN_ALBEDO_C3: [f32; 2] = [-905.00061, 2597.49308];
/// Sheen albedo fit, term `NdotV^2`: `lib/mx_microfacet_sheen.glsl:34`.
pub const MX_SHEEN_ALBEDO_C4: [f32; 2] = [60.28956, 121.81241];
/// Sheen albedo fit, term `r^2`: `lib/mx_microfacet_sheen.glsl:35`.
#[allow(
    clippy::excessive_precision,
    reason = "verbatim MaterialX digits; f32 rounds them exactly as GLSL float does"
)]
pub const MX_SHEEN_ALBEDO_C5: [f32; 2] = [1086.96473, 3045.55075];

/// `FRESNEL_MODEL_DIELECTRIC`: `lib/mx_microfacet_specular.glsl:3`.
pub const MX_FRESNEL_MODEL_DIELECTRIC: u32 = 0;
/// `FRESNEL_MODEL_CONDUCTOR`: `lib/mx_microfacet_specular.glsl:4`.
pub const MX_FRESNEL_MODEL_CONDUCTOR: u32 = 1;
/// `FRESNEL_MODEL_SCHLICK`: `lib/mx_microfacet_specular.glsl:5`.
pub const MX_FRESNEL_MODEL_SCHLICK: u32 = 2;
/// `COS_THETA_MAX = 1/7` of the Hoffman F82 Schlick: `lib/mx_microfacet_specular.glsl:203`.
pub const MX_COS_THETA_MAX: f32 = 1.0 / 7.0;
/// `COS_THETA_FACTOR = 1 / (COS_THETA_MAX * pow(1 - COS_THETA_MAX, 6))`:
/// `lib/mx_microfacet_specular.glsl:204`. Evaluated once here by repeated multiplication
/// (a constant, so both twins read the same bits).
pub const MX_COS_THETA_FACTOR: f32 = {
    let o = 1.0 - MX_COS_THETA_MAX;
    let o2 = o * o;
    1.0 / (MX_COS_THETA_MAX * (o2 * o2 * o2))
};
/// Cosine-weighted Fresnel average constant `1/21` (Schlick exponent 5):
/// `lib/mx_microfacet_specular.glsl:510`.
pub const MX_FRESNEL_AVERAGE_FACTOR: f32 = 1.0 / 21.0;
/// Lower clamp of `F0` in `mx_f0_to_ior`: `lib/mx_microfacet_specular.glsl:196`.
pub const MX_F0_TO_IOR_MIN: f32 = 0.01;
/// Upper clamp of `F0` in `mx_f0_to_ior`, and of the artistic reflectivity
/// (`mx_artistic_ior.glsl:6`): `lib/mx_microfacet_specular.glsl:196`.
pub const MX_F0_TO_IOR_MAX: f32 = 0.99;

/// Airy thin-film iterations, the MaterialX codegen default `hwAiryFresnelIterations(2)`:
/// `source/MaterialXGenShader/GenOptions.h:91`.
pub const MX_AIRY_FRESNEL_ITERATIONS: i32 = 2;
/// Thin-film thickness unit, nanometres to metres: `lib/mx_microfacet_specular.glsl:359`.
pub const MX_AIRY_NM_TO_M: f32 = 1.0e-9;
/// Airy XYZ sensitivity Gaussian amplitudes `val`: `lib/mx_microfacet_specular.glsl:292`.
pub const MX_AIRY_VAL: [f32; 3] = [5.4856e-13, 4.4201e-13, 5.2481e-13];
/// Airy XYZ sensitivity Gaussian positions `pos`: `lib/mx_microfacet_specular.glsl:293`.
pub const MX_AIRY_POS: [f32; 3] = [1.6810e+06, 1.7953e+06, 2.2084e+06];
/// Airy XYZ sensitivity Gaussian variances `var`: `lib/mx_microfacet_specular.glsl:294`.
pub const MX_AIRY_VAR: [f32; 3] = [4.3278e+09, 9.3046e+09, 6.6121e+09];
/// Second Gaussian of the X sensitivity, amplitude: `lib/mx_microfacet_specular.glsl:296`.
pub const MX_AIRY_X2_VAL: f32 = 9.7470e-14;
/// Second Gaussian of the X sensitivity, position: `lib/mx_microfacet_specular.glsl:296`.
pub const MX_AIRY_X2_POS: f32 = 2.2399e+06;
/// Second Gaussian of the X sensitivity, variance: `lib/mx_microfacet_specular.glsl:296`.
pub const MX_AIRY_X2_VAR: f32 = 4.5282e+09;
/// Sensitivity normalisation divisor: `lib/mx_microfacet_specular.glsl:297`.
pub const MX_AIRY_NORM: f32 = 1.0685e-7;
/// Row 0 of `XYZ_TO_RGB` (CIE 1931 RGB, illuminant E). The GLSL `mat3` of
/// `lib/mx_microfacet_specular.glsl:305` is column-major, so its row 0 is elements 0, 3, 6.
pub const MX_XYZ_TO_RGB_R0: [f32; 3] = [2.370_674_3, -0.900_040_5, -0.470_633_8];
/// Row 1 of `XYZ_TO_RGB` (elements 1, 4, 7 of `lib/mx_microfacet_specular.glsl:305`).
pub const MX_XYZ_TO_RGB_R1: [f32; 3] = [-0.513_885, 1.425_303_6, 0.088_581_4];
/// Row 2 of `XYZ_TO_RGB` (elements 2, 5, 8 of `lib/mx_microfacet_specular.glsl:305`).
pub const MX_XYZ_TO_RGB_R2: [f32; 3] = [0.005_298_2, -0.014_694_9, 1.009_396_8];

/// Selection-weight index of the coat lobe in [`crate::surface::lobe_weights`].
pub const SS_LOBE_COAT: u32 = 0;
/// Selection-weight index of the dielectric specular lobe.
pub const SS_LOBE_SPECULAR: u32 = 1;
/// Selection-weight index of the metal (conductor) lobe.
pub const SS_LOBE_METAL: u32 = 2;
/// Selection-weight index of the sheen lobe.
pub const SS_LOBE_SHEEN: u32 = 3;
/// Selection-weight index of the diffuse lobe.
pub const SS_LOBE_DIFFUSE: u32 = 4;
/// Selection-weight index of the transmission (rough dielectric BTDF) lobe.
pub const SS_LOBE_TRANSMISSION: u32 = 5;
/// Number of lobes, the length of the [`crate::surface::lobe_weights`] array
/// ([`crate::surface::LobeWeights`]; WGSL `array<f32, SS_LOBE_COUNT>`). Callers size their lobe
/// arrays with this constant, never with a literal.
pub const SS_LOBE_COUNT: u32 = 6;

/// `pcg4d` LCG multiplier (Jarzynski & Olano 2020, "Hash Functions for GPU Rendering", JCGT
/// 9(3), `pcg4d`), used by [`crate::sampling::pcg4d`].
pub const SS_PCG_MUL: u32 = 1_664_525;
/// `pcg4d` LCG increment (Jarzynski & Olano 2020).
pub const SS_PCG_INC: u32 = 1_013_904_223;
/// `pcg4d` xorshift amount (Jarzynski & Olano 2020).
pub const SS_PCG_SHIFT: u32 = 16;
/// `2^-24`: maps the top 24 bits of a `u32` to `[0, 1)` exactly in f32
/// ([`crate::sampling::u32_to_unit`]), so CPU and GPU uniforms are bit-identical.
pub const SS_U32_TO_UNIT: f32 = 1.0 / 16_777_216.0;

/// WGSL code of [`crate::microfacet::ThinFilmEnergy::MaterialX`].
pub const SS_THIN_FILM_ENERGY_MATERIALX: u32 = 0;
/// WGSL code of [`crate::microfacet::ThinFilmEnergy::Conserving`].
pub const SS_THIN_FILM_ENERGY_CONSERVING: u32 = 1;

/// A constant value as emitted into WGSL by [`wgsl_header`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WgslConst {
    /// `f32`.
    F32(f32),
    /// `i32`.
    I32(i32),
    /// `u32`.
    U32(u32),
    /// `vec2<f32>`.
    Vec2([f32; 2]),
    /// `vec3<f32>`.
    Vec3([f32; 3]),
    /// `vec4<f32>`.
    Vec4([f32; 4]),
}

/// Every constant the WGSL twin reads, by its WGSL name (identical to the Rust item name).
pub const WGSL_CONSTANTS: &[(&str, WgslConst)] = &[
    ("MX_PI", WgslConst::F32(MX_PI)),
    ("MX_PI_INV", WgslConst::F32(MX_PI_INV)),
    ("MX_TWO_PI", WgslConst::F32(MX_TWO_PI)),
    ("SS_INV_TWO_PI", WgslConst::F32(SS_INV_TWO_PI)),
    ("MX_DEG_TO_RAD", WgslConst::F32(MX_DEG_TO_RAD)),
    ("MX_FLOAT_EPS", WgslConst::F32(MX_FLOAT_EPS)),
    ("SS_ALPHA_MIN", WgslConst::F32(SS_ALPHA_MIN)),
    ("SS_TANGENT_MIN_LEN2", WgslConst::F32(SS_TANGENT_MIN_LEN2)),
    ("MX_ANISOTROPY_MAX", WgslConst::F32(MX_ANISOTROPY_MAX)),
    (
        "MX_THINFILM_IOR_DEFAULT",
        WgslConst::F32(MX_THINFILM_IOR_DEFAULT),
    ),
    (
        "MX_COAT_EMISSION_EXPONENT",
        WgslConst::F32(MX_COAT_EMISSION_EXPONENT),
    ),
    ("SS_LUMA", WgslConst::Vec3(SS_LUMA)),
    ("MX_GGX_ALBEDO_C0", WgslConst::Vec4(MX_GGX_ALBEDO_C0)),
    ("MX_GGX_ALBEDO_C1", WgslConst::Vec4(MX_GGX_ALBEDO_C1)),
    ("MX_GGX_ALBEDO_C2", WgslConst::Vec4(MX_GGX_ALBEDO_C2)),
    ("MX_GGX_ALBEDO_C3", WgslConst::Vec4(MX_GGX_ALBEDO_C3)),
    ("MX_GGX_ALBEDO_C4", WgslConst::Vec4(MX_GGX_ALBEDO_C4)),
    ("MX_GGX_ALBEDO_C5", WgslConst::Vec4(MX_GGX_ALBEDO_C5)),
    ("MX_GGX_ALBEDO_C6", WgslConst::Vec4(MX_GGX_ALBEDO_C6)),
    ("MX_GGX_ALBEDO_C7", WgslConst::Vec4(MX_GGX_ALBEDO_C7)),
    ("MX_GGX_ALBEDO_C8", WgslConst::Vec4(MX_GGX_ALBEDO_C8)),
    ("MX_ON_A_OFFSET", WgslConst::F32(MX_ON_A_OFFSET)),
    ("MX_ON_B_SCALE", WgslConst::F32(MX_ON_B_SCALE)),
    ("MX_ON_B_OFFSET", WgslConst::F32(MX_ON_B_OFFSET)),
    ("MX_ON_ALBEDO_C0", WgslConst::Vec2(MX_ON_ALBEDO_C0)),
    ("MX_ON_ALBEDO_C1", WgslConst::Vec2(MX_ON_ALBEDO_C1)),
    ("MX_ON_ALBEDO_C2", WgslConst::Vec2(MX_ON_ALBEDO_C2)),
    ("MX_ON_ALBEDO_C3", WgslConst::Vec2(MX_ON_ALBEDO_C3)),
    ("MX_FUJII_CONSTANT_1", WgslConst::F32(MX_FUJII_CONSTANT_1)),
    ("MX_FUJII_CONSTANT_2", WgslConst::F32(MX_FUJII_CONSTANT_2)),
    (
        "MX_BURLEY_SAMPLE_COUNT",
        WgslConst::I32(MX_BURLEY_SAMPLE_COUNT),
    ),
    (
        "MX_BURLEY_SAMPLE_WIDTH",
        WgslConst::F32(MX_BURLEY_SAMPLE_WIDTH),
    ),
    ("MX_BURLEY_MFP_MIN", WgslConst::F32(MX_BURLEY_MFP_MIN)),
    (
        "MX_SUBSURFACE_CURVATURE_MIN",
        WgslConst::F32(MX_SUBSURFACE_CURVATURE_MIN),
    ),
    (
        "MX_SHEEN_ROUGHNESS_MIN",
        WgslConst::F32(MX_SHEEN_ROUGHNESS_MIN),
    ),
    ("MX_SHEEN_ALBEDO_C0", WgslConst::Vec2(MX_SHEEN_ALBEDO_C0)),
    ("MX_SHEEN_ALBEDO_C1", WgslConst::Vec2(MX_SHEEN_ALBEDO_C1)),
    ("MX_SHEEN_ALBEDO_C2", WgslConst::Vec2(MX_SHEEN_ALBEDO_C2)),
    ("MX_SHEEN_ALBEDO_C3", WgslConst::Vec2(MX_SHEEN_ALBEDO_C3)),
    ("MX_SHEEN_ALBEDO_C4", WgslConst::Vec2(MX_SHEEN_ALBEDO_C4)),
    ("MX_SHEEN_ALBEDO_C5", WgslConst::Vec2(MX_SHEEN_ALBEDO_C5)),
    (
        "MX_FRESNEL_MODEL_DIELECTRIC",
        WgslConst::U32(MX_FRESNEL_MODEL_DIELECTRIC),
    ),
    (
        "MX_FRESNEL_MODEL_CONDUCTOR",
        WgslConst::U32(MX_FRESNEL_MODEL_CONDUCTOR),
    ),
    (
        "MX_FRESNEL_MODEL_SCHLICK",
        WgslConst::U32(MX_FRESNEL_MODEL_SCHLICK),
    ),
    ("MX_COS_THETA_MAX", WgslConst::F32(MX_COS_THETA_MAX)),
    ("MX_COS_THETA_FACTOR", WgslConst::F32(MX_COS_THETA_FACTOR)),
    (
        "MX_FRESNEL_AVERAGE_FACTOR",
        WgslConst::F32(MX_FRESNEL_AVERAGE_FACTOR),
    ),
    ("MX_F0_TO_IOR_MIN", WgslConst::F32(MX_F0_TO_IOR_MIN)),
    ("MX_F0_TO_IOR_MAX", WgslConst::F32(MX_F0_TO_IOR_MAX)),
    (
        "MX_AIRY_FRESNEL_ITERATIONS",
        WgslConst::I32(MX_AIRY_FRESNEL_ITERATIONS),
    ),
    ("MX_AIRY_NM_TO_M", WgslConst::F32(MX_AIRY_NM_TO_M)),
    ("MX_AIRY_VAL", WgslConst::Vec3(MX_AIRY_VAL)),
    ("MX_AIRY_POS", WgslConst::Vec3(MX_AIRY_POS)),
    ("MX_AIRY_VAR", WgslConst::Vec3(MX_AIRY_VAR)),
    ("MX_AIRY_X2_VAL", WgslConst::F32(MX_AIRY_X2_VAL)),
    ("MX_AIRY_X2_POS", WgslConst::F32(MX_AIRY_X2_POS)),
    ("MX_AIRY_X2_VAR", WgslConst::F32(MX_AIRY_X2_VAR)),
    ("MX_AIRY_NORM", WgslConst::F32(MX_AIRY_NORM)),
    ("MX_XYZ_TO_RGB_R0", WgslConst::Vec3(MX_XYZ_TO_RGB_R0)),
    ("MX_XYZ_TO_RGB_R1", WgslConst::Vec3(MX_XYZ_TO_RGB_R1)),
    ("MX_XYZ_TO_RGB_R2", WgslConst::Vec3(MX_XYZ_TO_RGB_R2)),
    ("SS_LOBE_COAT", WgslConst::U32(SS_LOBE_COAT)),
    ("SS_LOBE_SPECULAR", WgslConst::U32(SS_LOBE_SPECULAR)),
    ("SS_LOBE_METAL", WgslConst::U32(SS_LOBE_METAL)),
    ("SS_LOBE_SHEEN", WgslConst::U32(SS_LOBE_SHEEN)),
    ("SS_LOBE_DIFFUSE", WgslConst::U32(SS_LOBE_DIFFUSE)),
    ("SS_LOBE_TRANSMISSION", WgslConst::U32(SS_LOBE_TRANSMISSION)),
    ("SS_LOBE_COUNT", WgslConst::U32(SS_LOBE_COUNT)),
    ("SS_PCG_MUL", WgslConst::U32(SS_PCG_MUL)),
    ("SS_PCG_INC", WgslConst::U32(SS_PCG_INC)),
    ("SS_PCG_SHIFT", WgslConst::U32(SS_PCG_SHIFT)),
    ("SS_U32_TO_UNIT", WgslConst::F32(SS_U32_TO_UNIT)),
    (
        "SS_THIN_FILM_ENERGY_MATERIALX",
        WgslConst::U32(SS_THIN_FILM_ENERGY_MATERIALX),
    ),
    (
        "SS_THIN_FILM_ENERGY_CONSERVING",
        WgslConst::U32(SS_THIN_FILM_ENERGY_CONSERVING),
    ),
];

/// Formats one `f32` as a WGSL `f`-suffixed literal. Rust's `{:?}` is the shortest decimal
/// that round-trips to the same bits, and the `f` suffix makes WGSL round the decimal
/// directly to `f32` (no intermediate abstract-float rounding).
fn wgsl_f32(out: &mut String, v: f32) {
    out.push_str(&format!("{v:?}f"));
}

/// Formats a vector constant as `vecN<f32>(a, b, ...)`.
fn wgsl_vec(out: &mut String, n: usize, values: &[f32]) {
    out.push_str(&format!("vec{n}<f32>("));
    for (k, v) in values.iter().enumerate() {
        if k > 0 {
            out.push_str(", ");
        }
        wgsl_f32(out, *v);
    }
    out.push(')');
}

/// Emits [`WGSL_CONSTANTS`] as WGSL `const` declarations. This is the first part of
/// [`crate::wgsl::wgsl_source`]; it is also valid WGSL on its own (the bit test parses it alone).
pub fn wgsl_header() -> String {
    let mut out =
        String::from("// Generated by standard_surface_bsdf::consts::wgsl_header; do not edit.\n");
    for (name, value) in WGSL_CONSTANTS {
        out.push_str(&format!("const {name}"));
        match *value {
            WgslConst::F32(v) => {
                out.push_str(": f32 = ");
                wgsl_f32(&mut out, v);
            }
            WgslConst::I32(v) => {
                out.push_str(&format!(": i32 = {v}i"));
            }
            WgslConst::U32(v) => {
                out.push_str(&format!(": u32 = {v}u"));
            }
            WgslConst::Vec2(v) => {
                out.push_str(": vec2<f32> = ");
                wgsl_vec(&mut out, 2, &v);
            }
            WgslConst::Vec3(v) => {
                out.push_str(": vec3<f32> = ");
                wgsl_vec(&mut out, 3, &v);
            }
            WgslConst::Vec4(v) => {
                out.push_str(": vec4<f32> = ");
                wgsl_vec(&mut out, 4, &v);
            }
        }
        out.push_str(";\n");
    }
    out
}
