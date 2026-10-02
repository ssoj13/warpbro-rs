//! The layered Autodesk Standard Surface: inputs, shading frame, and the public evaluation
//! entry points (direct light, point or disc; pre-integrated environment; emission; lobe
//! selection weights).
//!
//! The layer graph is MaterialX's `NG_standard_surface_surfaceshader_100`
//! (`bxdf/standard_surface.mtlx:113-429`) with `opacity = 1` and `thin_walled = false`, so the
//! `selected_subsurface_bsdf` is the `subsurface_bsdf` (`:244-251`). With `T` a closure's
//! throughput, `att = clamp(mix(1, coat_color, coat), 0, 1)`, `m = metalness`,
//! `t = transmission` and `ss = subsurface`, every entry point returns the partition
//!
//! ```text
//! specular     = coat.r + T_coat att ((1 - m) spec.r + m metal.r)
//! base         = T_coat att (1 - m) T_spec (1 - t) (sheen.r + T_sheen mix(diffuse.r, subsurface.r, ss))
//! transmission = T_coat att (1 - m) T_spec t transmission.r      (wi below the surface)
//! ```
//!
//! which sums to the nested `coat_layer` value (`transmission_mix`,
//! `standard_surface.mtlx:169-173`; the transmission closure is [`crate::transmission`];
//! `subsurface_mix`, `:252-256`, [`subsurface_mix_reflection`]). With `ss = 0` the subsurface
//! closure is not evaluated and every output keeps the bits of the model without it. On the
//! inside of a transmissive surface the graph is the bare dielectric interface ([`layers`]).
//! With `t = 0` every entry point returns the bits of the reflection-only model. Contract
//! additions to MaterialX (plan §2): [`eval_light`] is exactly zero when `n.wo <= 0`, and when
//! `n.wi <= 0` except for the transmission lobe (MaterialX clamps instead, which leaks light
//! below the horizon into next-event estimation), and a GGX lobe with
//! `max(alpha) < SS_ALPHA_MIN` is a delta lobe whose response is zero in [`eval_light`] while
//! its throughput still attenuates the lobes beneath it and its selection weight is kept. The
//! layered model is not reciprocal (throughputs and energy compensation depend on `NdotV`
//! only, `lib/mx_microfacet_specular.glsl:516-521`), exactly as in MaterialX. Also kept from
//! MaterialX: the dielectric throughput uses the film-free scalar `F0`
//! (`mx_dielectric_bsdf.glsl:26,47-49`), so with a thin film the layered white furnace can
//! exceed 1 (measured 1.33 at 400 nm by `white_furnace_of_the_environment_path_is_bounded`);
//! without a film it stays at or below 1. [`SurfaceInputs::thin_film_energy`] =
//! [`ThinFilmEnergy::Conserving`] (task R2b) takes that throughput from the film-aware albedo
//! instead, and bounds the film albedo `FG(fd)` of every GGX lobe by the single-scatter energy
//! ([`crate::fresnel::film_dir_albedo`], task R2c), which keeps dielectric and conductor stacks
//! with a film at 1 (MaterialX: 1.34 dielectric, 1.13 conductor).
//!
//! Every function here has a WGSL twin in `wgsl/surface.wgsl` named `ss_<name>` (MaterialX
//! helpers keep their `mx_` name); the WGSL structs are `SsInputs`, `SsFrame`, `SsLobes`,
//! `SsEnvironment` and `SsLayers`.

use crate::InputsError;
use crate::consts::{
    MX_COAT_EMISSION_EXPONENT, MX_DEG_TO_RAD, MX_THINFILM_IOR_DEFAULT, SS_LOBE_COAT, SS_LOBE_COUNT,
    SS_LOBE_DIFFUSE, SS_LOBE_METAL, SS_LOBE_SHEEN, SS_LOBE_SPECULAR, SS_LOBE_TRANSMISSION, SS_LUMA,
    SS_TANGENT_MIN_LEN2,
};
use crate::diffuse::{
    mx_oren_nayar_diffuse_bsdf_indirect, mx_oren_nayar_diffuse_bsdf_reflection,
    mx_subsurface_bsdf_indirect, mx_subsurface_bsdf_reflection,
};
use crate::fresnel::{
    mx_artistic_ior, mx_fresnel_dielectric, mx_fresnel_schlick_exp, mx_ior_to_f0,
};
use crate::math::{
    MxBsdf, add3, clamp3, clampf, cross3, dot3, max3s, mix3, mixf, mul3, normalize3, pow3s,
    reflect3, scale3, splat3, sub3,
};
use crate::microfacet::{
    MxConductor, MxDielectric, ThinFilmEnergy, is_delta, mx_average_alpha,
    mx_conductor_bsdf_indirect, mx_conductor_bsdf_reflection, mx_dielectric_bsdf_indirect,
    mx_dielectric_bsdf_reflection, mx_forward_facing_normal, mx_orthonormal_basis,
    mx_roughness_anisotropy, ndf_alpha,
};
use crate::sheen::{mx_sheen_bsdf_indirect, mx_sheen_bsdf_reflection};
use crate::transmission::{ggx_transmission, transmission_is_delta};

/// WGSL functions of `wgsl/surface.wgsl`, sorted: the twins of this module's functions
/// (`mx_` names kept, `ss_` names are the Rust names with the prefix). Checked by the
/// `twin_registry_matches_wgsl_and_rust` test; collected by [`crate::wgsl::TWIN_FUNCTIONS`].
pub const WGSL_TWINS: &[&str] = &[
    "mx_generalized_schlick_edf",
    "mx_layer_bsdf",
    "mx_mix_bsdf",
    "mx_multiply_bsdf_color3",
    "mx_rotate_vector3",
    "ss_coat_alpha",
    "ss_combine",
    "ss_dominant_dir",
    "ss_eval_emission",
    "ss_eval_environment",
    "ss_eval_light",
    "ss_eval_light_disc",
    "ss_eval_light_materialx",
    "ss_eval_light_widened",
    "ss_eval_transmission",
    "ss_interface_fresnel",
    "ss_interface_reflect_share",
    "ss_layers",
    "ss_lobe_weights",
    "ss_luminance",
    "ss_main_alpha",
    "ss_shading_tangent",
    "ss_subsurface_mix_indirect",
    "ss_subsurface_mix_reflection",
    "ss_transmission_albedo",
    "ss_transmission_alpha",
    "ss_transmission_tint",
];

/// The Standard Surface inputs of the reflection-only subset, named after the MaterialX
/// inputs (`bxdf/standard_surface.mtlx:14-100`; Rust snake case, so `specular_IOR` is
/// `specular_ior`). Hot-path evaluation is infallible and assumes [`SurfaceInputs::validate`]
/// passed; `validate` is the one fallible entry point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceInputs {
    /// Diffuse weight (`base`).
    pub base: f32,
    /// Diffuse colour; also the metal reflectivity (`base_color * base`).
    pub base_color: [f32; 3],
    /// Oren-Nayar roughness (`diffuse_roughness`).
    pub diffuse_roughness: f32,
    /// Metal blend (`metalness`).
    pub metalness: f32,
    /// Dielectric specular weight; also the metal edge tint (`specular_color * specular`).
    pub specular: f32,
    /// Dielectric specular tint and metal edge colour.
    pub specular_color: [f32; 3],
    /// Specular (and metal) roughness.
    pub specular_roughness: f32,
    /// Dielectric specular IOR (`specular_IOR`).
    pub specular_ior: f32,
    /// Specular anisotropy (`specular_anisotropy`).
    pub specular_anisotropy: f32,
    /// Rotation of the anisotropy axis in turns (`specular_rotation`; `* 360` degrees).
    pub specular_rotation: f32,
    /// Transmission weight (`transmission`): the share of the base under the dielectric
    /// specular that refracts ([`crate::transmission`]) instead of reaching sheen and diffuse.
    pub transmission: f32,
    /// Transmission tint (`transmission_color`), applied once per interface crossing.
    pub transmission_color: [f32; 3],
    /// Added to `specular_roughness` for the transmission lobe, the sum clamped to `[0, 1]`
    /// (`transmission_extra_roughness`).
    pub transmission_extra_roughness: f32,
    /// Blend between the diffuse and the subsurface closure (`subsurface`; `subsurface_mix`,
    /// `standard_surface.mtlx:252-256`).
    pub subsurface: f32,
    /// Subsurface albedo (`subsurface_color`), coat-affected like the diffuse colour.
    pub subsurface_color: [f32; 3],
    /// Per-channel mean free path of the Burley profile (`subsurface_radius`), times
    /// `subsurface_scale`.
    pub subsurface_radius: [f32; 3],
    /// Scale of `subsurface_radius` (`subsurface_scale`).
    pub subsurface_scale: f32,
    /// Scattering direction (`subsurface_anisotropy`). Wired by the graph
    /// (`standard_surface.mtlx:241`) but not read by MaterialX's `subsurface_bsdf` body
    /// (`mx_subsurface_bsdf.glsl:4-32`), so it has no effect here either.
    pub subsurface_anisotropy: f32,
    /// Sheen weight.
    pub sheen: f32,
    /// Sheen colour.
    pub sheen_color: [f32; 3],
    /// Sheen roughness.
    pub sheen_roughness: f32,
    /// Clear-coat weight.
    pub coat: f32,
    /// Coat transmission colour (attenuates everything beneath; tints emission).
    pub coat_color: [f32; 3],
    /// Coat roughness.
    pub coat_roughness: f32,
    /// Coat anisotropy.
    pub coat_anisotropy: f32,
    /// Coat anisotropy rotation in turns.
    pub coat_rotation: f32,
    /// Coat IOR (`coat_IOR`).
    pub coat_ior: f32,
    /// Coat gamma on the diffuse colour (`coat_affect_color`).
    pub coat_affect_color: f32,
    /// Coat influence on the specular roughness (`coat_affect_roughness`).
    pub coat_affect_roughness: f32,
    /// Thin-film thickness in nanometres (`thin_film_thickness`; 0 disables Airy).
    pub thin_film_thickness: f32,
    /// Thin-film IOR (`thin_film_IOR`).
    pub thin_film_ior: f32,
    /// Emission weight.
    pub emission: f32,
    /// Emission colour.
    pub emission_color: [f32; 3],
    /// Throughput model of the dielectric specular with a thin film (not a MaterialX input;
    /// [`ThinFilmEnergy`], default MaterialX). WGSL: `thin_film_energy: u32`
    /// (`ThinFilmEnergy::code`).
    pub thin_film_energy: ThinFilmEnergy,
}

impl SurfaceInputs {
    /// The MaterialX v1.0.1 nodedef defaults (`bxdf/standard_surface.mtlx:6-12` over the
    /// v1.0.0 values of `:14-96`): base 1, base colour 0.8, specular roughness 0.2, IORs 1.5,
    /// coat roughness 0.1, sheen roughness 0.3, everything else 0 or white.
    pub const MATERIALX_DEFAULT: Self = Self {
        base: 1.0,
        base_color: [0.8, 0.8, 0.8],
        diffuse_roughness: 0.0,
        metalness: 0.0,
        specular: 1.0,
        specular_color: [1.0, 1.0, 1.0],
        specular_roughness: 0.2,
        specular_ior: 1.5,
        specular_anisotropy: 0.0,
        specular_rotation: 0.0,
        transmission: 0.0,
        transmission_color: [1.0, 1.0, 1.0],
        transmission_extra_roughness: 0.0,
        subsurface: 0.0,
        subsurface_color: [1.0, 1.0, 1.0],
        subsurface_radius: [1.0, 1.0, 1.0],
        subsurface_scale: 1.0,
        subsurface_anisotropy: 0.0,
        sheen: 0.0,
        sheen_color: [1.0, 1.0, 1.0],
        sheen_roughness: 0.3,
        coat: 0.0,
        coat_color: [1.0, 1.0, 1.0],
        coat_roughness: 0.1,
        coat_anisotropy: 0.0,
        coat_rotation: 0.0,
        coat_ior: 1.5,
        coat_affect_color: 0.0,
        coat_affect_roughness: 0.0,
        thin_film_thickness: 0.0,
        thin_film_ior: 1.5,
        emission: 0.0,
        emission_color: [1.0, 1.0, 1.0],
        thin_film_energy: ThinFilmEnergy::MaterialX,
    };

    /// Every scalar input and colour channel with its MaterialX name and the domain the
    /// evaluation is defined on, in declaration order. [`SurfaceInputs::validate`] checks this
    /// list; consumers (e.g. a UI table) can prove their ranges lie inside it.
    pub fn domain(&self) -> [(&'static str, f32, InputRange); 49] {
        let [bc_r, bc_g, bc_b] = self.base_color;
        let [sc_r, sc_g, sc_b] = self.specular_color;
        let [tc_r, tc_g, tc_b] = self.transmission_color;
        let [ssc_r, ssc_g, ssc_b] = self.subsurface_color;
        let [ssr_r, ssr_g, ssr_b] = self.subsurface_radius;
        let [sh_r, sh_g, sh_b] = self.sheen_color;
        let [cc_r, cc_g, cc_b] = self.coat_color;
        let [ec_r, ec_g, ec_b] = self.emission_color;
        [
            ("base", self.base, InputRange::UNIT),
            ("base_color.r", bc_r, InputRange::NON_NEGATIVE),
            ("base_color.g", bc_g, InputRange::NON_NEGATIVE),
            ("base_color.b", bc_b, InputRange::NON_NEGATIVE),
            (
                "diffuse_roughness",
                self.diffuse_roughness,
                InputRange::UNIT,
            ),
            ("metalness", self.metalness, InputRange::UNIT),
            ("specular", self.specular, InputRange::UNIT),
            ("specular_color.r", sc_r, InputRange::NON_NEGATIVE),
            ("specular_color.g", sc_g, InputRange::NON_NEGATIVE),
            ("specular_color.b", sc_b, InputRange::NON_NEGATIVE),
            (
                "specular_roughness",
                self.specular_roughness,
                InputRange::UNIT,
            ),
            ("specular_IOR", self.specular_ior, InputRange::IOR),
            (
                "specular_anisotropy",
                self.specular_anisotropy,
                InputRange::UNIT,
            ),
            (
                "specular_rotation",
                self.specular_rotation,
                InputRange::UNIT,
            ),
            ("transmission", self.transmission, InputRange::UNIT),
            ("transmission_color.r", tc_r, InputRange::NON_NEGATIVE),
            ("transmission_color.g", tc_g, InputRange::NON_NEGATIVE),
            ("transmission_color.b", tc_b, InputRange::NON_NEGATIVE),
            (
                "transmission_extra_roughness",
                self.transmission_extra_roughness,
                InputRange::SIGNED_UNIT,
            ),
            ("subsurface", self.subsurface, InputRange::UNIT),
            ("subsurface_color.r", ssc_r, InputRange::NON_NEGATIVE),
            ("subsurface_color.g", ssc_g, InputRange::NON_NEGATIVE),
            ("subsurface_color.b", ssc_b, InputRange::NON_NEGATIVE),
            ("subsurface_radius.r", ssr_r, InputRange::NON_NEGATIVE),
            ("subsurface_radius.g", ssr_g, InputRange::NON_NEGATIVE),
            ("subsurface_radius.b", ssr_b, InputRange::NON_NEGATIVE),
            (
                "subsurface_scale",
                self.subsurface_scale,
                InputRange::NON_NEGATIVE,
            ),
            (
                "subsurface_anisotropy",
                self.subsurface_anisotropy,
                InputRange::SIGNED_UNIT,
            ),
            ("sheen", self.sheen, InputRange::UNIT),
            ("sheen_color.r", sh_r, InputRange::NON_NEGATIVE),
            ("sheen_color.g", sh_g, InputRange::NON_NEGATIVE),
            ("sheen_color.b", sh_b, InputRange::NON_NEGATIVE),
            ("sheen_roughness", self.sheen_roughness, InputRange::UNIT),
            ("coat", self.coat, InputRange::UNIT),
            ("coat_color.r", cc_r, InputRange::NON_NEGATIVE),
            ("coat_color.g", cc_g, InputRange::NON_NEGATIVE),
            ("coat_color.b", cc_b, InputRange::NON_NEGATIVE),
            ("coat_roughness", self.coat_roughness, InputRange::UNIT),
            ("coat_anisotropy", self.coat_anisotropy, InputRange::UNIT),
            ("coat_rotation", self.coat_rotation, InputRange::UNIT),
            ("coat_IOR", self.coat_ior, InputRange::IOR),
            (
                "coat_affect_color",
                self.coat_affect_color,
                InputRange::UNIT,
            ),
            (
                "coat_affect_roughness",
                self.coat_affect_roughness,
                InputRange::UNIT,
            ),
            (
                "thin_film_thickness",
                self.thin_film_thickness,
                InputRange::NON_NEGATIVE,
            ),
            ("thin_film_IOR", self.thin_film_ior, InputRange::IOR),
            ("emission", self.emission, InputRange::NON_NEGATIVE),
            ("emission_color.r", ec_r, InputRange::NON_NEGATIVE),
            ("emission_color.g", ec_g, InputRange::NON_NEGATIVE),
            ("emission_color.b", ec_b, InputRange::NON_NEGATIVE),
        ]
    }

    /// Checks every input against [`SurfaceInputs::domain`]: finite and inside its range.
    /// Returns the first offending input; never clamps.
    pub fn validate(&self) -> Result<(), InputsError> {
        for (input, value, range) in self.domain() {
            if !value.is_finite() {
                return Err(InputsError::NonFinite { input, value });
            }
            if value < range.min || value > range.max {
                return Err(InputsError::OutOfRange {
                    input,
                    value,
                    min: range.min,
                    max: range.max,
                });
            }
        }
        Ok(())
    }
}

/// A closed domain `[min, max]` of one input (see [`SurfaceInputs::domain`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InputRange {
    /// Inclusive lower bound.
    pub min: f32,
    /// Inclusive upper bound (`f32::MAX` = unbounded above, finiteness still required).
    pub max: f32,
}

impl InputRange {
    /// Weights, roughnesses, anisotropies, rotations: `[0, 1]` (MaterialX `uimin`/`uimax`).
    pub const UNIT: Self = Self { min: 0.0, max: 1.0 };
    /// `transmission_extra_roughness` and `subsurface_anisotropy`: `[-1, 1]` (MaterialX
    /// `uimin = -1`, `uimax = 1`).
    pub const SIGNED_UNIT: Self = Self {
        min: -1.0,
        max: 1.0,
    };
    /// Colours, emission, film thickness, subsurface radius and scale: `[0, inf)`.
    pub const NON_NEGATIVE: Self = Self {
        min: 0.0,
        max: f32::MAX,
    };
    /// Real IORs of the reflection-only subset (outside medium vacuum): `[1, inf)`. MaterialX
    /// allows `uimin = 0`, but an IOR below 1 models the interior side, which the subset
    /// excludes, and the Airy path clamps a film IOR below 1 silently
    /// (`lib/mx_microfacet_specular.glsl:309`).
    pub const IOR: Self = Self {
        min: 1.0,
        max: f32::MAX,
    };
}

/// The shading frame at a hit: unit normal `n` (forward-facing toward the viewer, the caller's
/// responsibility), a tangent hint and the side of the surface the viewer is on. The
/// anisotropy axis is `tangent` projected onto the tangent plane and rotated about `n` by
/// `rotation * 360` degrees ([`shading_tangent`]); a tangent parallel to `n` falls back to
/// [`mx_orthonormal_basis`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShadingFrame {
    /// Unit shading normal.
    pub n: [f32; 3],
    /// Tangent hint (need not be unit or orthogonal to `n`).
    pub tangent: [f32; 3],
    /// `wo` lies inside the object (the ray hit a back face; `n` has been flipped toward `wo`).
    /// With `transmission > 0` the surface is then the bare dielectric interface seen from the
    /// dense side (see [`layers`]); with `transmission = 0` it is ignored (an opaque surface has
    /// no interior, so a back face shades like a front face).
    pub inside: bool,
    /// Surface curvature at the hit (1 / radius, `>= 0`), read only by the subsurface closure:
    /// MaterialX estimates it in screen space as `length(fwidth(N)) / length(fwidth(P))`
    /// (`mx_microfacet_diffuse.glsl:196`); a rasteriser passes that, a path tracer its
    /// geometric counterpart. 0 is a flat surface (MaterialX then uses radius 100).
    pub curvature: f32,
}

/// The layered value split into the part under the dielectric specular (diffuse and sheen,
/// `base`), the refracted part (`transmission`) and the rest (coat, dielectric specular and
/// metal, `specular`). `base + specular + transmission` is the full `coat_layer` value.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Lobes {
    /// Diffuse + sheen contribution after every throughput above them.
    pub base: [f32; 3],
    /// Coat + dielectric specular + metal contribution.
    pub specular: [f32; 3],
    /// Transmission contribution ([`crate::transmission`]): for direct light non-zero only for
    /// `wi` below the surface, for the environment from [`Environment::transmission_radiance`].
    pub transmission: [f32; 3],
}

impl Lobes {
    /// All-zero value (outside the hemisphere, or nothing reflects).
    pub const ZERO: Self = Self {
        base: [0.0; 3],
        specular: [0.0; 3],
        transmission: [0.0; 3],
    };

    /// `base + specular + transmission`, the full layered value.
    pub fn sum(&self) -> [f32; 3] {
        add3(add3(self.base, self.specular), self.transmission)
    }
}

/// Caller-supplied environment lighting for [`eval_environment`], MaterialX's prefiltered
/// form (`lib/mx_environment_prefilter.glsl:10-29`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Environment {
    /// Cosine-convolved radiance about `n`, i.e. irradiance / pi (MaterialX `$envIrradiance`
    /// convention: a uniform environment of radiance 1 passes 1). Lights diffuse and sheen.
    pub irradiance: [f32; 3],
    /// Radiance prefiltered for the specular/metal lobe, e.g. sampled along
    /// [`dominant_dir`] with [`main_alpha`]'s average. Multiplied by `FG * comp` here.
    pub specular_radiance: [f32; 3],
    /// Radiance prefiltered for the coat lobe (along [`dominant_dir`] with [`coat_alpha`]).
    pub coat_radiance: [f32; 3],
    /// Radiance prefiltered for the transmission lobe, seen along the refracted direction
    /// (MaterialX looks it up behind a refracting solid sphere, `mx_refraction_solid_sphere`,
    /// with [`transmission_alpha`]). Multiplied by [`transmission_tint`] only: unlike
    /// `mx_environment_radiance` in refraction mode (`mx_environment_prefilter.glsl:15-16`),
    /// no `1 - FG`, since the specular layer's throughput has already removed the reflection.
    pub transmission_radiance: [f32; 3],
}

/// A lobe of the layered BSDF; [`Lobe::index`] is its slot in [`lobe_weights`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Lobe {
    /// Clear coat (dielectric GGX, `coat_IOR`).
    Coat,
    /// Dielectric specular (GGX, `specular_IOR`, thin film).
    Specular,
    /// Metal (conductor GGX, artistic IOR, thin film).
    Metal,
    /// Imageworks sheen.
    Sheen,
    /// Oren-Nayar diffuse mixed with the subsurface closure (`subsurface_mix`); both are
    /// cosine-sampled, so one lobe carries both.
    Diffuse,
    /// Rough dielectric transmission ([`crate::transmission`], `specular_IOR`).
    Transmission,
}

/// Normalised lobe-selection probabilities, indexed by [`Lobe::index`]; the length is
/// [`SS_LOBE_COUNT`] (WGSL `array<f32, SS_LOBE_COUNT>`).
pub type LobeWeights = [f32; SS_LOBE_COUNT as usize];

impl Lobe {
    /// Every lobe in [`lobe_weights`] order.
    pub const ALL: [Lobe; SS_LOBE_COUNT as usize] = [
        Lobe::Coat,
        Lobe::Specular,
        Lobe::Metal,
        Lobe::Sheen,
        Lobe::Diffuse,
        Lobe::Transmission,
    ];

    /// Index of this lobe in [`lobe_weights`] (the `SS_LOBE_*` constants).
    pub const fn index(self) -> usize {
        match self {
            Lobe::Coat => SS_LOBE_COAT as usize,
            Lobe::Specular => SS_LOBE_SPECULAR as usize,
            Lobe::Metal => SS_LOBE_METAL as usize,
            Lobe::Sheen => SS_LOBE_SHEEN as usize,
            Lobe::Diffuse => SS_LOBE_DIFFUSE as usize,
            Lobe::Transmission => SS_LOBE_TRANSMISSION as usize,
        }
    }
}

/// Everything the entry points derive from the inputs and the view before evaluating closures:
/// the graph's intermediate nodes (`main_roughness`, `artistic_ior`, `coat_attenuation`,
/// `coat_affected_diffuse_color`, the tangents). WGSL twin struct: `SsLayers`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Layers {
    /// Shading normal (the frame's `n`).
    pub n: [f32; 3],
    /// View direction `wo`.
    pub v: [f32; 3],
    /// Specular/metal anisotropy axis (`main_tangent`, `standard_surface.mtlx:153-169`).
    pub main_tangent: [f32; 3],
    /// Coat anisotropy axis (`coat_tangent`, `standard_surface.mtlx:173-189`).
    pub coat_tangent: [f32; 3],
    /// `coat_bsdf` (`standard_surface.mtlx:349-358`): tint 1, no film (`pbrlib_defs.mtlx:68-69`).
    pub coat: MxDielectric,
    /// `specular_bsdf` (`standard_surface.mtlx:288-299`).
    pub specular: MxDielectric,
    /// `metal_bsdf` (`standard_surface.mtlx:306-327`, artistic IOR included).
    pub metal: MxConductor,
    /// `coat_attenuation` clamped by `multiply_bsdf` (`standard_surface.mtlx:336-344`,
    /// `mx_multiply_bsdf_color3.glsl:5`).
    pub attenuation: [f32; 3],
    /// `metalness`.
    pub metalness: f32,
    /// `coat_affected_diffuse_color` (`standard_surface.mtlx:193-211`).
    pub diffuse_color: [f32; 3],
    /// `transmission_bsdf` (`standard_surface.mtlx:159-168`): weight 1, tint
    /// `transmission_color`, `ior` the relative IOR `eta_wi / eta_wo` of a refracted `wi`
    /// (`specular_IOR` outside, its inverse inside), alpha [`transmission_alpha`], no film.
    pub transmission: MxDielectric,
    /// `transmission_mix` weight (`standard_surface.mtlx:169-173`): `transmission`, or 1 on the
    /// inside of a transmissive surface.
    pub transmission_mix: f32,
    /// The view is on the inside of a transmissive surface: the specular and transmission are
    /// the two halves of one dielectric interface (see [`layers`]).
    pub inside: bool,
    /// `coat_affected_subsurface_color` (`standard_surface.mtlx:212-219`).
    pub subsurface_color: [f32; 3],
    /// `subsurface_radius_scaled`, the Burley mean free path (`standard_surface.mtlx:233-236`).
    pub subsurface_radius: [f32; 3],
    /// `subsurface_mix` weight (`standard_surface.mtlx:252-256`): `subsurface`.
    pub subsurface_mix: f32,
}

/// `mx_layer_bsdf`: `response = top.r + top.T base.r`, `throughput = top.T base.T`
/// (`pbrlib/genglsl/mx_layer_bsdf.glsl:3-7`).
pub fn mx_layer_bsdf(top: MxBsdf, base: MxBsdf) -> MxBsdf {
    MxBsdf {
        response: add3(top.response, mul3(base.response, top.throughput)),
        throughput: mul3(top.throughput, base.throughput),
    }
}

/// `mx_mix_bsdf`: `mix(bg, fg, t)` of response and throughput
/// (`pbrlib/genglsl/mx_mix_bsdf.glsl:3-7`).
pub fn mx_mix_bsdf(fg: MxBsdf, bg: MxBsdf, mix_value: f32) -> MxBsdf {
    MxBsdf {
        response: mix3(bg.response, fg.response, mix_value),
        throughput: mix3(bg.throughput, fg.throughput, mix_value),
    }
}

/// `mx_multiply_bsdf_color3`: response times `clamp(tint, 0, 1)`, throughput unchanged
/// (`pbrlib/genglsl/mx_multiply_bsdf_color3.glsl:3-8`).
pub fn mx_multiply_bsdf_color3(in1: MxBsdf, in2: [f32; 3]) -> MxBsdf {
    MxBsdf {
        response: mul3(in1.response, clamp3(in2, 0.0, 1.0)),
        throughput: in1.throughput,
    }
}

/// `mx_generalized_schlick_edf`: `base * mix(color0, color90, (1 - NdotV)^exponent)`
/// (`pbrlib/genglsl/mx_generalized_schlick_edf.glsl:4-13`).
pub fn mx_generalized_schlick_edf(
    v: [f32; 3],
    n: [f32; 3],
    color0: [f32; 3],
    color90: [f32; 3],
    exponent: f32,
    base: [f32; 3],
) -> [f32; 3] {
    let n = mx_forward_facing_normal(n, v);
    let ndotv = clampf(dot3(n, v), crate::consts::MX_FLOAT_EPS, 1.0);
    let f = mx_fresnel_schlick_exp(ndotv, color0, color90, exponent);
    mul3(base, f)
}

/// Rodrigues rotation of `v` by `amount` degrees about `axis`
/// (`stdlib/genglsl/mx_rotate_vector3.glsl:1-13`), degrees converted by [`MX_DEG_TO_RAD`].
pub fn mx_rotate_vector3(v: [f32; 3], amount: f32, axis: [f32; 3]) -> [f32; 3] {
    let axis = normalize3(axis);
    let rotation_radians = amount * MX_DEG_TO_RAD;
    let s = crate::fm::sinf(rotation_radians);
    let c = crate::fm::cosf(rotation_radians);
    let oc = 1.0 - c;
    add3(
        add3(scale3(v, c), scale3(cross3(v, axis), s)),
        scale3(scale3(axis, dot3(axis, v)), oc),
    )
}

/// The anisotropy axis of a lobe: `tangent` projected onto the plane of `n` (the Duff basis
/// when `tangent` is parallel to `n`), normalised, then rotated about `n` by `rotation * 360`
/// degrees when `anisotropy > 0` (`standard_surface.mtlx:152-189`: MaterialX rotates, then
/// the closure projects; the order is swapped so the fallback axis is rotated too, which is
/// identical for every non-degenerate tangent).
pub fn shading_tangent(n: [f32; 3], t: [f32; 3], rotation: f32, anisotropy: f32) -> [f32; 3] {
    let p = sub3(t, scale3(n, dot3(t, n)));
    let len2 = dot3(p, p);
    let x = if len2 < SS_TANGENT_MIN_LEN2 {
        let [basis_x, _, _] = mx_orthonormal_basis(n);
        basis_x
    } else {
        normalize3(p)
    };
    if anisotropy > 0.0 {
        normalize3(mx_rotate_vector3(x, rotation * 360.0, n))
    } else {
        x
    }
}

/// The specular/metal GGX alpha pair, graph node `main_roughness`
/// (`standard_surface.mtlx:117-133`): `mix(specular_roughness, 1, coat_affect_roughness *
/// coat * coat_roughness)` through [`mx_roughness_anisotropy`].
pub fn main_alpha(i: &SurfaceInputs) -> [f32; 2] {
    let coat_affect = i.coat_affect_roughness * i.coat * i.coat_roughness;
    let coat_affected_roughness = mixf(i.specular_roughness, 1.0, coat_affect);
    mx_roughness_anisotropy(coat_affected_roughness, i.specular_anisotropy)
}

/// The coat GGX alpha pair, graph node `coat_roughness_vector`
/// (`standard_surface.mtlx:345-348`).
pub fn coat_alpha(i: &SurfaceInputs) -> [f32; 2] {
    mx_roughness_anisotropy(i.coat_roughness, i.coat_anisotropy)
}

/// The transmission GGX alpha pair, graph node `transmission_roughness`
/// (`standard_surface.mtlx:135-150`): `clamp(specular_roughness +
/// transmission_extra_roughness, 0, 1)`, pulled toward 1 by the coat as in [`main_alpha`].
pub fn transmission_alpha(i: &SurfaceInputs) -> [f32; 2] {
    let coat_affect = i.coat_affect_roughness * i.coat * i.coat_roughness;
    let roughness = clampf(
        i.specular_roughness + i.transmission_extra_roughness,
        0.0,
        1.0,
    );
    let coat_affected_roughness = mixf(roughness, 1.0, coat_affect);
    mx_roughness_anisotropy(coat_affected_roughness, i.specular_anisotropy)
}

/// Derives the closure parameters and intermediate graph values for one view direction.
///
/// On the inside of a transmissive surface (`f.inside` and `transmission > 0`) the graph is the
/// bare dielectric interface seen from the dense side, with no layering: no coat (weight 0,
/// attenuation 1), no metal, no sheen or diffuse (`transmission_mix = 1`). The specular becomes
/// the interface's reflection (weight 1, white, no film, the transmission alpha) and the
/// transmission its refraction with the Fresnel transmittance `1 - F(wo.m)` (Walter et al. 2007,
/// eq. 21), both with the relative IOR `1 / specular_IOR`, so total internal reflection moves
/// the energy into the reflection. The layered model of the outside cannot hold here: its
/// specular throughput comes from albedo fits of `F0`, which know nothing of total internal
/// reflection (the white furnace near the critical angle would exceed 1).
pub fn layers(i: &SurfaceInputs, f: &ShadingFrame, wo: [f32; 3]) -> Layers {
    let main = main_alpha(i);
    let artistic = mx_artistic_ior(
        scale3(i.base_color, i.base),
        scale3(i.specular_color, i.specular),
    );
    let coat_gamma = clampf(i.coat, 0.0, 1.0) * i.coat_affect_color + 1.0;
    let mut s = Layers {
        n: f.n,
        v: wo,
        main_tangent: shading_tangent(f.n, f.tangent, i.specular_rotation, i.specular_anisotropy),
        coat_tangent: shading_tangent(f.n, f.tangent, i.coat_rotation, i.coat_anisotropy),
        coat: MxDielectric {
            weight: i.coat,
            tint: splat3(1.0),
            ior: i.coat_ior,
            roughness: coat_alpha(i),
            thinfilm_thickness: 0.0,
            thinfilm_ior: MX_THINFILM_IOR_DEFAULT,
            thin_film_energy: i.thin_film_energy,
        },
        specular: MxDielectric {
            weight: i.specular,
            tint: i.specular_color,
            ior: i.specular_ior,
            roughness: main,
            thinfilm_thickness: i.thin_film_thickness,
            thinfilm_ior: i.thin_film_ior,
            thin_film_energy: i.thin_film_energy,
        },
        metal: MxConductor {
            weight: 1.0,
            ior_n: artistic.ior,
            ior_k: artistic.extinction,
            roughness: main,
            thinfilm_thickness: i.thin_film_thickness,
            thinfilm_ior: i.thin_film_ior,
            thin_film_energy: i.thin_film_energy,
        },
        attenuation: clamp3(mix3(splat3(1.0), i.coat_color, i.coat), 0.0, 1.0),
        metalness: i.metalness,
        diffuse_color: pow3s(max3s(i.base_color, 0.0), coat_gamma),
        transmission: MxDielectric {
            weight: 1.0,
            tint: i.transmission_color,
            ior: i.specular_ior,
            roughness: transmission_alpha(i),
            thinfilm_thickness: 0.0,
            thinfilm_ior: MX_THINFILM_IOR_DEFAULT,
            thin_film_energy: i.thin_film_energy,
        },
        transmission_mix: i.transmission,
        inside: false,
        subsurface_color: pow3s(max3s(i.subsurface_color, 0.0), coat_gamma),
        subsurface_radius: scale3(i.subsurface_radius, i.subsurface_scale),
        subsurface_mix: i.subsurface,
    };
    let inside = f.inside && i.transmission > 0.0;
    if inside {
        let inverse_ior = 1.0 / i.specular_ior;
        s.coat.weight = 0.0;
        s.attenuation = splat3(1.0);
        s.metalness = 0.0;
        s.specular.weight = 1.0;
        s.specular.tint = splat3(1.0);
        s.specular.ior = inverse_ior;
        s.specular.roughness = s.transmission.roughness;
        s.specular.thinfilm_thickness = 0.0;
        s.transmission.ior = inverse_ior;
        s.transmission_mix = 1.0;
        s.inside = true;
    }
    s
}

/// Graph node `subsurface_mix` for direct light (`standard_surface.mtlx:252-256`):
/// `mix(diffuse_bsdf, subsurface_bsdf, subsurface)` with the subsurface closure of weight 1
/// (`:237-243`, [`mx_subsurface_bsdf_reflection`] at the frame's curvature). Exactly the
/// Oren-Nayar closure when `subsurface = 0`: the subsurface closure is then not evaluated.
pub fn subsurface_mix_reflection(
    i: &SurfaceInputs,
    f: &ShadingFrame,
    s: &Layers,
    wo: [f32; 3],
    wi: [f32; 3],
) -> MxBsdf {
    let diffuse = mx_oren_nayar_diffuse_bsdf_reflection(
        wo,
        wi,
        s.n,
        i.base,
        s.diffuse_color,
        i.diffuse_roughness,
        false,
    );
    if s.subsurface_mix <= 0.0 {
        return diffuse;
    }
    let subsurface = mx_subsurface_bsdf_reflection(
        wo,
        wi,
        s.n,
        f.curvature,
        1.0,
        s.subsurface_color,
        s.subsurface_radius,
    );
    mx_mix_bsdf(subsurface, diffuse, s.subsurface_mix)
}

/// Graph node `subsurface_mix` for environment light: [`subsurface_mix_reflection`] with the
/// `CLOSURE_TYPE_INDIRECT` closures and `irradiance`.
pub fn subsurface_mix_indirect(
    i: &SurfaceInputs,
    s: &Layers,
    wo: [f32; 3],
    irradiance: [f32; 3],
) -> MxBsdf {
    let diffuse = mx_oren_nayar_diffuse_bsdf_indirect(
        wo,
        s.n,
        i.base,
        s.diffuse_color,
        i.diffuse_roughness,
        false,
        irradiance,
    );
    if s.subsurface_mix <= 0.0 {
        return diffuse;
    }
    let subsurface = mx_subsurface_bsdf_indirect(1.0, s.subsurface_color, irradiance);
    mx_mix_bsdf(subsurface, diffuse, s.subsurface_mix)
}

/// Assembles the [`Lobes`] partition from the five reflection closure values (plan §2 "eval");
/// sheen and diffuse are scaled by `1 - transmission_mix` (MaterialX `transmission_mix`), the
/// transmission part is zero (it lies below the surface or comes from the environment).
pub fn combine(
    s: &Layers,
    coat: MxBsdf,
    specular: MxBsdf,
    metal: MxBsdf,
    sheen: MxBsdf,
    diffuse: MxBsdf,
) -> Lobes {
    let m = s.metalness;
    let under_coat = mul3(coat.throughput, s.attenuation);
    let dielectric_under = scale3(under_coat, 1.0 - m);
    Lobes {
        base: mul3(
            mul3(dielectric_under, specular.throughput),
            scale3(
                add3(sheen.response, mul3(sheen.throughput, diffuse.response)),
                1.0 - s.transmission_mix,
            ),
        ),
        specular: add3(
            coat.response,
            mul3(
                under_coat,
                add3(
                    scale3(specular.response, 1.0 - m),
                    scale3(metal.response, m),
                ),
            ),
        ),
        transmission: splat3(0.0),
    }
}

/// Everything that multiplies the transmission BTDF at `wo`: outside `T_coat att (1 - m) T_spec
/// transmission_mix max(transmission_color, 0)` (the layer factors above `transmission_bsdf` and
/// its tint), inside the bare tint (the interface's Fresnel is in the BTDF). Zero, and not
/// evaluated, when `transmission_mix = 0`.
pub fn transmission_tint(s: &Layers, wo: [f32; 3]) -> [f32; 3] {
    if s.transmission_mix <= 0.0 {
        return splat3(0.0);
    }
    if s.inside {
        return scale3(max3s(s.transmission.tint, 0.0), s.transmission_mix);
    }
    let unit = splat3(1.0);
    let coat = mx_dielectric_bsdf_indirect(wo, s.n, &s.coat, unit);
    let specular = mx_dielectric_bsdf_indirect(wo, s.n, &s.specular, unit);
    let under_coat = mul3(coat.throughput, s.attenuation);
    let dielectric_under = scale3(under_coat, 1.0 - s.metalness);
    scale3(
        mul3(
            mul3(dielectric_under, specular.throughput),
            max3s(s.transmission.tint, 0.0),
        ),
        s.transmission_mix,
    )
}

/// Dielectric Fresnel reflectance `F(n.wo)` of the specular closure's IOR (TIR-aware): the
/// reflected share of the bare interface on the inside of a transmissive surface.
pub fn interface_fresnel(s: &Layers, wo: [f32; 3]) -> f32 {
    let ndotv = clampf(dot3(s.n, wo), crate::consts::MX_FLOAT_EPS, 1.0);
    mx_fresnel_dielectric(ndotv, s.specular.ior)
}

/// The share of lobe selections given to the reflection of the bare interface inside a
/// transmissive surface: `mix(F(n.wo), 1/2, sqrt(avg alpha))`. Only a selection weight: a rough
/// interface reflects below and refracts beyond the macroscopic critical angle through its
/// microfacets, so neither lobe may get a zero share unless the interface is smooth (where the
/// macroscopic [`interface_fresnel`] is exact).
pub fn interface_reflect_share(s: &Layers, wo: [f32; 3]) -> f32 {
    let roughness = crate::fm::sqrtf(mx_average_alpha(s.transmission.roughness));
    mixf(interface_fresnel(s, wo), 0.5, roughness)
}

/// The energy the transmission lobe passes at `wo` with the BTDF's own albedo taken as 1: the
/// weight of a delta (smooth) refraction, the transmission of [`eval_environment`] and the
/// lobe's selection weight. Outside [`transmission_tint`]; inside `(1 - F(n.wo))` times it
/// ([`interface_fresnel`]).
pub fn transmission_albedo(s: &Layers, wo: [f32; 3]) -> [f32; 3] {
    let tint = transmission_tint(s, wo);
    if s.inside {
        return scale3(tint, 1.0 - interface_fresnel(s, wo));
    }
    tint
}

/// Direct light with the GGX NDFs widened by `ndf_widen` ([`ndf_alpha`]); the single
/// implementation behind [`eval_light`] (`ndf_widen = 0`, `split_delta`), [`eval_light_disc`]
/// and [`eval_light_materialx`] (no delta split). With `split_delta`, a delta GGX lobe has zero
/// response (plan §2, C2); without it, the lobe keeps MaterialX's clamped-alpha response.
pub fn eval_light_widened(
    i: &SurfaceInputs,
    f: &ShadingFrame,
    wo: [f32; 3],
    wi: [f32; 3],
    ndf_widen: f32,
    split_delta: bool,
) -> Lobes {
    // Hemisphere rule (plan §2, I1): the reflection lobes need both directions above the
    // horizon (NaN counts as outside); otherwise only the transmission lobe can respond.
    let above = dot3(f.n, wo) > 0.0 && dot3(f.n, wi) > 0.0;
    if !above {
        return eval_transmission(i, f, wo, wi, ndf_widen, split_delta);
    }
    let s = layers(i, f, wo);
    let mut coat = mx_dielectric_bsdf_reflection(wo, wi, s.n, s.coat_tangent, &s.coat, ndf_widen);
    let mut specular =
        mx_dielectric_bsdf_reflection(wo, wi, s.n, s.main_tangent, &s.specular, ndf_widen);
    let mut metal = mx_conductor_bsdf_reflection(wo, wi, s.n, s.main_tangent, &s.metal, ndf_widen);
    let sheen = mx_sheen_bsdf_reflection(wo, wi, s.n, i.sheen, i.sheen_color, i.sheen_roughness);
    let diffuse = subsurface_mix_reflection(i, f, &s, wo, wi);
    // Delta lobes (plan §2, C2): zero response, throughput kept.
    if split_delta && is_delta(ndf_alpha(s.coat.roughness, ndf_widen)) {
        coat.response = splat3(0.0);
    }
    if split_delta && is_delta(ndf_alpha(s.specular.roughness, ndf_widen)) {
        specular.response = splat3(0.0);
    }
    if split_delta && is_delta(ndf_alpha(s.metal.roughness, ndf_widen)) {
        metal.response = splat3(0.0);
    }
    combine(&s, coat, specular, metal, sheen, diffuse)
}

/// The transmission part of direct light ([`eval_light_widened`] for `wi` below the horizon):
/// `transmission_tint * f_t |n.wi|` ([`ggx_transmission`]) when `wo` is above and `wi` below
/// the horizon and `transmission > 0`; zero otherwise, for a delta lobe with `split_delta`, and
/// for `eta = 1` (the straight-through delta). Only [`Lobes::transmission`] is non-zero.
pub fn eval_transmission(
    i: &SurfaceInputs,
    f: &ShadingFrame,
    wo: [f32; 3],
    wi: [f32; 3],
    ndf_widen: f32,
    split_delta: bool,
) -> Lobes {
    let transmits = dot3(f.n, wo) > 0.0 && dot3(f.n, wi) < 0.0 && i.transmission > 0.0;
    if !transmits {
        return Lobes::ZERO;
    }
    let s = layers(i, f, wo);
    let eta = s.transmission.ior;
    let delta = transmission_is_delta(ndf_alpha(s.transmission.roughness, ndf_widen), eta);
    if (split_delta && delta) || eta == 1.0 {
        return Lobes::ZERO;
    }
    let t = ggx_transmission(
        wo,
        wi,
        s.n,
        s.main_tangent,
        &s.transmission,
        ndf_widen,
        s.inside,
    );
    Lobes {
        transmission: scale3(transmission_tint(&s, wo), t),
        ..Lobes::ZERO
    }
}

/// Direct light from a point or directional light: `f(wo, wi) * |cos(n, wi)|` as [`Lobes`],
/// multiplied by the light's irradiance by the caller. Zero for a view below the horizon and
/// for delta lobes (plan §2); for `wi` below the horizon only the transmission lobe responds
/// ([`eval_transmission`]).
pub fn eval_light(i: &SurfaceInputs, f: &ShadingFrame, wo: [f32; 3], wi: [f32; 3]) -> Lobes {
    eval_light_widened(i, f, wo, wi, 0.0, true)
}

/// Direct light exactly as MaterialX's raster closures compute it: the roughness clamp of
/// `mx_roughness_anisotropy.glsl:3` (`alpha >= M_FLOAT_EPS`) and no delta split, so a roughness-0
/// lobe keeps its (extremely peaked) response. For rasterisers without next-event estimation or
/// MIS (the `standard-surface` raster pipeline, plan R4); path tracers use [`eval_light`]. The
/// hemisphere rule still applies.
pub fn eval_light_materialx(
    i: &SurfaceInputs,
    f: &ShadingFrame,
    wo: [f32; 3],
    wi: [f32; 3],
) -> Lobes {
    eval_light_widened(i, f, wo, wi, 0.0, false)
}

/// Direct light from a disc light of angular radius `half_angle` (radians, `0 <= half_angle <
/// pi/2`) evaluated at the disc centre `wi`: every GGX NDF is widened to
/// `alpha' = min(1, sqrt(alpha^2 + (tan(half_angle) / 2)^2))` (plan §4, "Disc widening"), with
/// no Karis energy factor; multiply by the disc irradiance `L * Omega`. A mirror then sees
/// `F * L`. `half_angle = 0` is bit-identical to [`eval_light`].
pub fn eval_light_disc(
    i: &SurfaceInputs,
    f: &ShadingFrame,
    wo: [f32; 3],
    wi: [f32; 3],
    half_angle: f32,
) -> Lobes {
    eval_light_widened(i, f, wo, wi, crate::fm::tanf(half_angle) * 0.5, true)
}

/// Pre-integrated environment light (MaterialX's `CLOSURE_TYPE_INDIRECT` path): GGX lobes
/// return `radiance * FG * comp`, diffuse and sheen `irradiance * albedo`, layered by the same
/// throughputs as [`eval_light`]. Delta lobes are finite here and are included.
pub fn eval_environment(
    i: &SurfaceInputs,
    f: &ShadingFrame,
    wo: [f32; 3],
    env: &Environment,
) -> Lobes {
    let s = layers(i, f, wo);
    let coat = mx_dielectric_bsdf_indirect(wo, s.n, &s.coat, env.coat_radiance);
    let specular = mx_dielectric_bsdf_indirect(wo, s.n, &s.specular, env.specular_radiance);
    let metal = mx_conductor_bsdf_indirect(wo, s.n, &s.metal, env.specular_radiance);
    let sheen = mx_sheen_bsdf_indirect(
        wo,
        s.n,
        i.sheen,
        i.sheen_color,
        i.sheen_roughness,
        env.irradiance,
    );
    let diffuse = subsurface_mix_indirect(i, &s, wo, env.irradiance);
    let mut lobes = combine(&s, coat, specular, metal, sheen, diffuse);
    lobes.transmission = mul3(transmission_albedo(&s, wo), env.transmission_radiance);
    lobes
}

/// Emitted radiance toward `wo`: `mix(E, E coat_color (1 - F0(coat_IOR)) (1 - (1 - NdotV)^5),
/// coat)` with `E = emission * emission_color` (`standard_surface.mtlx:365-409`,
/// `mx_generalized_schlick_edf.glsl:9-11`).
pub fn eval_emission(i: &SurfaceInputs, f: &ShadingFrame, wo: [f32; 3]) -> [f32; 3] {
    let emission = scale3(i.emission_color, i.emission);
    let coat_f0 = mx_ior_to_f0(i.coat_ior);
    let coat_emission = mx_generalized_schlick_edf(
        wo,
        f.n,
        splat3(1.0 - coat_f0),
        splat3(0.0),
        MX_COAT_EMISSION_EXPONENT,
        mul3(emission, i.coat_color),
    );
    mix3(emission, coat_emission, i.coat)
}

/// Luminance with the MaterialX default (ACEScg) coefficients [`SS_LUMA`].
pub fn luminance(c: [f32; 3]) -> f32 {
    dot3(c, SS_LUMA)
}

/// Normalised lobe-selection probabilities at `wo`, indexed by [`Lobe::index`] (plan §2,
/// "sample/pdf"): each lobe's directional albedo times every throughput above it, by
/// luminance. All zero when nothing reflects. Delta lobes keep their weight. This is the one
/// function computing the weights; sampling and pdf evaluation both read it (I12). The
/// transmission weight is the luminance of [`transmission_albedo`]; sheen and diffuse are
/// scaled by `1 - transmission_mix`. Inside a transmissive surface the specular and
/// transmission weights split by [`interface_reflect_share`] (the transmission's times the
/// tint's luminance) instead of the outside's albedo fits, which know no total internal
/// reflection.
pub fn lobe_weights(i: &SurfaceInputs, f: &ShadingFrame, wo: [f32; 3]) -> LobeWeights {
    let s = layers(i, f, wo);
    let unit = splat3(1.0);
    let coat = mx_dielectric_bsdf_indirect(wo, s.n, &s.coat, unit);
    let specular = mx_dielectric_bsdf_indirect(wo, s.n, &s.specular, unit);
    let metal = mx_conductor_bsdf_indirect(wo, s.n, &s.metal, unit);
    let sheen = mx_sheen_bsdf_indirect(wo, s.n, i.sheen, i.sheen_color, i.sheen_roughness, unit);
    let diffuse = subsurface_mix_indirect(i, &s, wo, unit);
    let m = s.metalness;
    let under_coat = mul3(coat.throughput, s.attenuation);
    let dielectric_under = scale3(under_coat, 1.0 - m);
    let under_specular = scale3(
        mul3(dielectric_under, specular.throughput),
        1.0 - s.transmission_mix,
    );

    let p_coat = luminance(sub3(unit, coat.throughput)).max(0.0);
    let p_specular = if s.inside {
        interface_reflect_share(&s, wo)
    } else {
        luminance(mul3(dielectric_under, sub3(unit, specular.throughput))).max(0.0)
    };
    let p_metal = luminance(mul3(scale3(under_coat, m), metal.response)).max(0.0);
    let p_sheen = luminance(mul3(under_specular, sub3(unit, sheen.throughput))).max(0.0);
    let p_diffuse = luminance(mul3(
        mul3(under_specular, sheen.throughput),
        diffuse.response,
    ))
    .max(0.0);
    let p_transmission = if s.inside {
        luminance(max3s(s.transmission.tint, 0.0)).max(0.0)
            * (1.0 - interface_reflect_share(&s, wo))
    } else {
        luminance(transmission_albedo(&s, wo)).max(0.0)
    };

    let total = p_coat + p_specular + p_metal + p_sheen + p_diffuse + p_transmission;
    if total > 0.0 {
        [
            p_coat / total,
            p_specular / total,
            p_metal / total,
            p_sheen / total,
            p_diffuse / total,
            p_transmission / total,
        ]
    } else {
        [0.0; SS_LOBE_COUNT as usize]
    }
}

/// Dominant direction of a GGX lobe for environment lookups (Lagarde & de Rousiers 2014,
/// Frostbite §4.9.3): `normalize(mix(n, R, (1 - alpha)(sqrt(1 - alpha) + alpha)))` with
/// `R = reflect(-wo, n)` and `alpha` in `[0, 1]` (pass [`crate::microfacet::mx_average_alpha`]
/// of [`main_alpha`] or [`coat_alpha`]).
pub fn dominant_dir(f: &ShadingFrame, wo: [f32; 3], alpha: f32) -> [f32; 3] {
    let r = reflect3(scale3(wo, -1.0), f.n);
    let smoothness = 1.0 - alpha;
    let lerp_factor = smoothness * (crate::fm::sqrtf(smoothness) + alpha);
    normalize3(mix3(f.n, r, lerp_factor))
}
