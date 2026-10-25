//! The formula family registry: [`FORMULAS`], one [`FormulaMeta`] row per family code (the 8
//! classic families and the 20 extended families, all shared by CUDA and WGPU), and the parameter
//! structs and packing of the extended families.
//!
//! [`FORMULAS`] is the single source of truth for what the rest of the crate knows about a
//! family: its display name (`scene::Formula::name`), whether it reads the shared orbit constant
//! (`scene::Formula::has_orbit_constant`), whether its estimate is signed and whether it iterates
//! (the WGSL `SIGNED_MASK` / `ITERATING_MASK` of [`SIGNED_MASK`] / [`ITERATING_MASK`]), and per
//! parameter the UI label, hint, hard range, [`Slider`] span, enum choices and row order that
//! `WorldDocument` attributes read. Each extended formula additionally defines:
//! - strongly-typed parameters with `Serialize`/`Deserialize` so `WorldDocument` discovers
//!   `/formula/<Variant>/<field>` attributes automatically,
//! - parameter packing into the per-object formula slot block, offsets `0..EXT_SLOTS` from
//!   `P_BULB_POWER` ([`pack_extended_formula`]),
//! - framing/bounding radii (`scene::Formula::framing_radius` / `bound_radius`).

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::params::*;

// Family IDs belong to the shared host/device ABI.
pub use fractal_core::params::{
    FAMILY_AMAZING_SURFACE, FAMILY_ASTROLABE, FAMILY_ATTRACTOR_TUBE, FAMILY_BRISTORBROT,
    FAMILY_BURNING_SHIP, FAMILY_CRYSTAL_GEODE, FAMILY_DUPIN_CYCLIDE, FAMILY_GYROID_TPMS,
    FAMILY_HARMONIC_BULB, FAMILY_HOPF_FIBRATION, FAMILY_KALEIDO_TWIST, FAMILY_LIDINOID_TPMS,
    FAMILY_MANDELBAR, FAMILY_MOBIUS_IFS, FAMILY_NAUTILUS, FAMILY_PHYLLOTAXIS,
    FAMILY_POLYHEDRAL_KIFS, FAMILY_SUPERFORMULA, FAMILY_TORUS_KNOT, FAMILY_VORONOI_FOAM,
    TOTAL_FORMULA_COUNT,
};

// =============================================================================
// Declarative Metadata Types
// =============================================================================

/// The interactive slider span of a numeric attribute: soft limits for dragging (the hard
/// limits are the attribute's range). The formula registry fills it per parameter; `world.rs`
/// attaches it to every numeric `WorldAttribute`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Slider {
    pub min: f64,
    pub max: f64,
    /// Logarithmic for values spanning orders of magnitude.
    pub log: bool,
}

/// A linear slider span, for the [`FORMULAS`] table.
const fn lin(min: f64, max: f64) -> Option<Slider> {
    Some(Slider {
        min,
        max,
        log: false,
    })
}

/// A logarithmic slider span, for the [`FORMULAS`] table.
const fn log(min: f64, max: f64) -> Option<Slider> {
    Some(Slider {
        min,
        max,
        log: true,
    })
}

#[derive(Clone, Copy, Debug)]
pub struct ParamMeta {
    /// Attribute field name under `/formula/<Variant>/<name>`.
    pub name: &'static str,
    /// Human-readable label shown in the Attribute Editor and Timeline.
    pub label: &'static str,
    /// Tooltip description shown when hovering the parameter row.
    pub hint: &'static str,
    /// Hard validation range `(min, max)`.
    pub range: Option<(f64, f64)>,
    /// Soft slider span for interactive dragging.
    pub slider: Option<Slider>,
    /// Enum choices (if this parameter is a discrete enum).
    pub choices: &'static [&'static str],
    /// Relative sort order of the row within the Fractal section: Hybrid's step list first
    /// (12), then the family parameters from 20 up.
    pub order: u16,
}

#[derive(Clone, Copy, Debug)]
pub struct FormulaMeta {
    /// The family code (`P_FAMILY`), equal to the row's index in [`FORMULAS`].
    pub code: u32,
    /// The family's WGSL distance estimator (`render_wgpu.wgsl`), called by the generated
    /// `family_distance` (`render_wgpu::family_distance_wgsl`).
    pub wgsl: &'static str,
    /// Enum variant key in `Formula` (e.g. `"AmazingSurface"`).
    pub variant: &'static str,
    /// The family's name in the Formula dropdown, the inspector and new scene names
    /// (`scene::Formula::name`).
    pub display_name: &'static str,
    /// Tooltip description of the formula family.
    pub hint: &'static str,
    /// Whether the formula uses the shared Julia/orbit constant (`P_JULIA`, `P_JULIA_C`, `P_ORBIT_OFFSET`).
    pub has_orbit_constant: bool,
    /// Whether the distance estimate is signed (negative inside): the normal stencils then sample
    /// around the hit instead of one footprint back along the ray (fractal-kernels
    /// `family_signed`; WGSL `SIGNED_MASK`).
    pub signed: bool,
    /// Whether the formula runs an iteration loop controlled by `/render/iterations`
    /// (`P_ITERATIONS`, fractional counts blended); analytical and procedural shapes use their own
    /// octave / fiber / ring counts (WGSL `ITERATING_MASK`).
    pub iterates: bool,
    /// Parameter metadata table for all fields of this formula.
    pub params: &'static [ParamMeta],
}

/// Bit `code` set for every family whose `FormulaMeta` field picked by `signed` (else
/// `iterates`) is true.
const fn family_mask(signed: bool) -> u32 {
    let mut mask = 0u32;
    let mut i = 0;
    while i < FORMULAS.len() {
        let meta = &FORMULAS[i];
        if (signed && meta.signed) || (!signed && meta.iterates) {
            mask |= 1 << meta.code;
        }
        i += 1;
    }
    mask
}

/// Families with a signed estimate, bit per code: emitted as the WGSL `SIGNED_MASK`.
pub const SIGNED_MASK: u32 = family_mask(true);
/// Families iterating `P_ITERATIONS` times, bit per code: emitted as the WGSL `ITERATING_MASK`.
pub const ITERATING_MASK: u32 = family_mask(false);

// The family code is the row index (lookups index `FORMULAS[code]`), and a mask bit exists for
// every code.
const _: () = {
    assert!(TOTAL_FORMULA_COUNT <= u32::BITS);
    assert!(SIGNED_MASK == FAMILY_SIGNED_MASK);
    assert!(ITERATING_MASK == FAMILY_ITERATING_MASK);
    let mut i = 0;
    while i < FORMULAS.len() {
        assert!(FORMULAS[i].code as usize == i);
        i += 1;
    }
};

// =============================================================================
// 20 Extended Formula Structs & Enums
// =============================================================================

/// 1. Amazing Surface (KaliBox): 2-axis box fold + offset + sphere fold + scale + rotation.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AmazingSurface {
    pub scale: f32,
    pub min_radius: f32,
    pub fixed_radius: f32,
    pub fold_limit: [f32; 3],
    pub shift: [f32; 3],
    pub rotation_degrees: [f32; 3],
    pub bound_radius: f32,
}

impl AmazingSurface {
    pub const PRESET: Self = Self {
        scale: -1.65,
        min_radius: 0.35,
        fixed_radius: 1.0,
        fold_limit: [1.0, 1.0, 0.85],
        shift: [0.0, -0.35, 0.0],
        rotation_degrees: [5.0, 0.0, 0.0],
        bound_radius: 2.4,
    };
}

/// 2. Bristorbrot: Quadratic 3D hypercomplex fractal with fine dendritic filaments.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Bristorbrot {
    pub scale: f32,
    pub bailout: f32,
    pub y_sign: f32,
    pub z_sign: f32,
    pub twist: f32,
    pub rotation_degrees: [f32; 3],
}

impl Bristorbrot {
    pub const PRESET: Self = Self {
        scale: 1.0,
        bailout: 4.0,
        y_sign: -1.0,
        z_sign: -1.0,
        twist: 0.0,
        rotation_degrees: [0.0; 3],
    };
}

/// 3. Burning Ship 3D / Mandelbulb-Abs: Absolute-value folded triplex fractal.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BurningShip {
    pub power: f32,
    pub bailout: f32,
    pub abs_axes: [f32; 3],
    pub shift: [f32; 3],
    pub rotation_degrees: [f32; 3],
}

impl BurningShip {
    pub const PRESET: Self = Self {
        power: 5.0,
        bailout: 3.0,
        abs_axes: [1.0, 1.0, 1.0],
        shift: [0.0, 0.0, 0.0],
        rotation_degrees: [0.0; 3],
    };
}

/// 4. Kaleidoscopic Twist IFS (Marble Marcher): Menger fold + dual inter-step rotations + shift.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct KaleidoTwist {
    pub scale: f32,
    pub shift: [f32; 3],
    pub rotation_degrees: [f32; 3],
    pub post_rotation_degrees: [f32; 3],
    pub clamp_z: f32,
    pub bound_radius: f32,
}

impl KaleidoTwist {
    pub const PRESET: Self = Self {
        scale: 1.92,
        shift: [2.1, 2.4, 1.8],
        rotation_degrees: [18.0, -12.0, 9.0],
        post_rotation_degrees: [0.0, 14.0, -7.0],
        clamp_z: 1.0,
        bound_radius: 2.5,
    };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PolyhedralSymmetry {
    Icosahedron,
    Dodecahedron,
}

impl PolyhedralSymmetry {
    pub const fn code(self) -> u32 {
        match self {
            Self::Icosahedron => 0,
            Self::Dodecahedron => 1,
        }
    }
}

/// 5. Polyhedral KIFS: Golden-ratio icosahedral/dodecahedral reflection planes (Knighty).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PolyhedralKifs {
    pub symmetry: PolyhedralSymmetry,
    pub scale: f32,
    pub offset: [f32; 3],
    pub shell_thickness: f32,
    pub rotation_degrees: [f32; 3],
    pub bound_radius: f32,
}

impl PolyhedralKifs {
    pub const PRESET: Self = Self {
        symmetry: PolyhedralSymmetry::Icosahedron,
        scale: 2.05,
        offset: [0.850_650_8, 0.525_731_1, 0.0],
        shell_thickness: 0.12,
        rotation_degrees: [8.0, 12.0, 0.0],
        bound_radius: 1.85,
    };
}

/// 6. Harmonic Bulb: Spherical harmonics modulated triplex fractal.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct HarmonicBulb {
    pub power: f32,
    pub bailout: f32,
    pub harmonic_l: f32,
    pub harmonic_m: f32,
    pub amplitude: f32,
    pub rotation_degrees: [f32; 3],
}

impl HarmonicBulb {
    pub const PRESET: Self = Self {
        power: 7.0,
        bailout: 2.5,
        harmonic_l: 4.0,
        harmonic_m: 3.0,
        amplitude: 0.28,
        rotation_degrees: [0.0; 3],
    };
}

/// 7. Mandelbar 3D (Tricorn): Conjugated hypercomplex fractal with gothic arches.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Mandelbar3D {
    pub power: f32,
    pub bailout: f32,
    pub conjugate_xy: f32,
    pub conjugate_z: f32,
    pub twist: f32,
    pub rotation_degrees: [f32; 3],
}

impl Mandelbar3D {
    pub const PRESET: Self = Self {
        power: 4.0,
        bailout: 3.0,
        conjugate_xy: -1.0,
        conjugate_z: -1.0,
        twist: 0.0,
        rotation_degrees: [0.0; 3],
    };
}

/// 8. Möbius Sphere-Inversion IFS: Multi-sphere inversion with spiral twist.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MobiusIfs {
    pub scale: f32,
    pub min_radius: f32,
    pub fixed_radius: f32,
    pub lattice_pitch: [f32; 3],
    pub twist: f32,
    pub rotation_degrees: [f32; 3],
    pub bound_radius: f32,
}

impl MobiusIfs {
    pub const PRESET: Self = Self {
        scale: 1.55,
        min_radius: 0.28,
        fixed_radius: 1.05,
        lattice_pitch: [1.2, 1.2, 1.2],
        twist: 0.18,
        rotation_degrees: [6.0, 10.0, 0.0],
        bound_radius: 1.75,
    };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TpmsKind {
    Gyroid,
    SchwarzP,
    SchwarzD,
    Neovius,
}

impl TpmsKind {
    pub const fn code(self) -> u32 {
        match self {
            Self::Gyroid => 0,
            Self::SchwarzP => 1,
            Self::SchwarzD => 2,
            Self::Neovius => 3,
        }
    }
}

/// 9. Multi-Scale Gyroid / Schwarz / Neovius TPMS with FBM octave detailing.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct GyroidTpms {
    pub surface: TpmsKind,
    pub frequency: f32,
    pub thickness: f32,
    pub bias: f32,
    pub octaves: u32,
    pub lacunarity: f32,
    pub gain: f32,
    pub twist: f32,
    pub bound_radius: f32,
}

impl GyroidTpms {
    pub const PRESET: Self = Self {
        surface: TpmsKind::Gyroid,
        frequency: 3.4,
        thickness: 0.08,
        bias: 0.0,
        octaves: 3,
        lacunarity: 2.1,
        gain: 0.35,
        twist: 0.25,
        bound_radius: 1.45,
    };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComplexTpmsKind {
    Lidinoid,
    FischerKochS,
    SplitP,
}

impl ComplexTpmsKind {
    pub const fn code(self) -> u32 {
        match self {
            Self::Lidinoid => 0,
            Self::FischerKochS => 1,
            Self::SplitP => 2,
        }
    }
}

/// 10. Lidinoid / Fischer-Koch S / Split-P higher-order minimal surfaces.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LidinoidTpms {
    pub surface: ComplexTpmsKind,
    pub frequency: f32,
    pub thickness: f32,
    pub bias: f32,
    pub harmonic_mix: f32,
    pub radial_warp: f32,
    pub bound_radius: f32,
}

impl LidinoidTpms {
    pub const PRESET: Self = Self {
        surface: ComplexTpmsKind::Lidinoid,
        frequency: 2.6,
        thickness: 0.09,
        bias: 0.15,
        harmonic_mix: 0.5,
        radial_warp: 0.15,
        bound_radius: 1.45,
    };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttractorKind {
    Aizawa,
    Halvorsen,
    Thomas,
}

impl AttractorKind {
    pub const fn code(self) -> u32 {
        match self {
            Self::Aizawa => 0,
            Self::Halvorsen => 1,
            Self::Thomas => 2,
        }
    }
}

/// 11. Strange Attractor Flow Shell (Aizawa / Halvorsen / Thomas stream-surface).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AttractorTube {
    pub system: AttractorKind,
    pub step_dt: f32,
    pub tube_radius: f32,
    pub swirl: f32,
    pub param_a: f32,
    pub param_b: f32,
    pub bound_radius: f32,
}

impl AttractorTube {
    pub const PRESET: Self = Self {
        system: AttractorKind::Aizawa,
        step_dt: 0.045,
        tube_radius: 0.065,
        swirl: 1.35,
        param_a: 0.95,
        param_b: 0.7,
        bound_radius: 1.55,
    };
}

/// 12. Iterated Torus Knot Braid: Recursive `(p, q)` torus knot with braided sub-strands.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TorusKnotBraid {
    pub winds_p: f32,
    pub winds_q: f32,
    pub major_radius: f32,
    pub minor_radius: f32,
    pub tube_radius: f32,
    pub sub_strands: u32,
    pub braid_depth: f32,
    pub bound_radius: f32,
}

impl TorusKnotBraid {
    pub const PRESET: Self = Self {
        winds_p: 3.0,
        winds_q: 5.0,
        major_radius: 0.85,
        minor_radius: 0.32,
        tube_radius: 0.085,
        sub_strands: 4,
        braid_depth: 0.045,
        bound_radius: 1.45,
    };
}

/// 13. Hopf Fibration Linked Tori: Stereographic projection of S^3 fibers into nested linked rings.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct HopfFibration {
    pub fibers: u32,
    pub shells: u32,
    pub latitude: f32,
    pub tube_radius: f32,
    pub twist: f32,
    pub scale: f32,
    pub bound_radius: f32,
}

impl HopfFibration {
    pub const PRESET: Self = Self {
        fibers: 8,
        shells: 3,
        latitude: 0.65,
        tube_radius: 0.065,
        twist: 1.0,
        scale: 0.85,
        bound_radius: 1.65,
    };
}

/// 14. Dupin Cyclide / Horned Apollonian Tori: Exact quartic/inversion cyclide shells with ribs.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DupinCyclide {
    pub major_a: f32,
    pub minor_b: f32,
    pub eccentricity_c: f32,
    pub radius_d: f32,
    pub shell_thickness: f32,
    pub rib_frequency: f32,
    pub rib_depth: f32,
    pub bound_radius: f32,
}

impl DupinCyclide {
    pub const PRESET: Self = Self {
        major_a: 1.0,
        minor_b: 0.92,
        eccentricity_c: 0.38,
        radius_d: 0.42,
        shell_thickness: 0.06,
        rib_frequency: 12.0,
        rib_depth: 0.03,
        bound_radius: 1.6,
    };
}

/// 15. 3D Gielis Superformula / Radiolaria Shell: Spherical product of two superformulas with pores.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Superformula3D {
    pub m1: f32,
    pub n1_1: f32,
    pub n2_1: f32,
    pub n3_1: f32,
    pub m2: f32,
    pub n1_2: f32,
    pub n2_2: f32,
    pub n3_2: f32,
    pub shell_thickness: f32,
    pub pore_frequency: f32,
    pub bound_radius: f32,
}

impl Superformula3D {
    pub const PRESET: Self = Self {
        m1: 6.0,
        n1_1: 1.2,
        n2_1: 1.7,
        n3_1: 1.7,
        m2: 4.0,
        n1_2: 1.0,
        n2_2: 1.5,
        n3_2: 1.5,
        shell_thickness: 0.08,
        pore_frequency: 10.0,
        bound_radius: 1.65,
    };
}

/// 16. Spherical Fibonacci Phyllotaxis (Romanesco): O(1) inverse spherical Fibonacci lattice.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PhyllotaxisShell {
    pub count: f32,
    pub cone_height: f32,
    pub cone_sharpness: f32,
    pub spiral_twist: f32,
    pub sub_florets: f32,
    pub base_radius: f32,
    pub bound_radius: f32,
}

impl PhyllotaxisShell {
    pub const PRESET: Self = Self {
        count: 144.0,
        cone_height: 0.24,
        cone_sharpness: 1.6,
        spiral_twist: 1.0,
        sub_florets: 8.0,
        base_radius: 0.95,
        bound_radius: 1.45,
    };
}

/// 17. Logarithmic Helicospiral Shell (Nautilus / Ammonite): Raup morphospace + septa chambers.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct NautilusShell {
    pub expansion_rate: f32,
    pub coil_tightness: f32,
    pub spire_translation: f32,
    pub wall_thickness: f32,
    pub rib_frequency: f32,
    pub rib_amplitude: f32,
    pub turns: f32,
    pub bound_radius: f32,
}

impl NautilusShell {
    pub const PRESET: Self = Self {
        expansion_rate: 0.18,
        coil_tightness: 1.15,
        spire_translation: 0.0,
        wall_thickness: 0.045,
        rib_frequency: 18.0,
        rib_amplitude: 0.025,
        turns: 2.75,
        bound_radius: 1.65,
    };
}

/// 18. Multi-Scale 3D Voronoi / Trabecular Bone Foam: Analytical F2 - F1 cellular struts.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct VoronoiFoam {
    pub cell_density: f32,
    pub strut_radius: f32,
    pub jitter: f32,
    pub octaves: u32,
    pub lacunarity: f32,
    pub shell_hollow: f32,
    pub bound_radius: f32,
}

impl VoronoiFoam {
    pub const PRESET: Self = Self {
        cell_density: 2.4,
        strut_radius: 0.075,
        jitter: 0.82,
        octaves: 2,
        lacunarity: 2.0,
        shell_hollow: 0.35,
        bound_radius: 1.35,
    };
}

/// 19. Procedural Quartz Crystal Cluster / Geode: Hexagonal prisms + rhombohedral terminations.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CrystalGeode {
    pub crystal_count: u32,
    pub prism_length: f32,
    pub prism_radius: f32,
    pub tip_slope: f32,
    pub cluster_spread: f32,
    pub geode_cavity: f32,
    pub bound_radius: f32,
}

impl CrystalGeode {
    pub const PRESET: Self = Self {
        crystal_count: 14,
        prism_length: 0.95,
        prism_radius: 0.22,
        tip_slope: 1.25,
        cluster_spread: 0.65,
        geode_cavity: 0.0,
        bound_radius: 1.55,
    };
}

/// 20. Mechanical Astrolabe / Nested Gyroscopic Gimbal Rings with gear teeth.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AstrolabeGimbal {
    pub ring_count: u32,
    pub outer_radius: f32,
    pub ring_spacing: f32,
    pub ring_width: f32,
    pub ring_thickness: f32,
    pub gear_teeth: f32,
    pub tooth_depth: f32,
    pub tilt_step_degrees: f32,
    pub spin_degrees: f32,
    pub bound_radius: f32,
}

impl AstrolabeGimbal {
    pub const PRESET: Self = Self {
        ring_count: 5,
        outer_radius: 1.15,
        ring_spacing: 0.16,
        ring_width: 0.11,
        ring_thickness: 0.055,
        gear_teeth: 24.0,
        tooth_depth: 0.022,
        tilt_step_degrees: 28.0,
        spin_degrees: 15.0,
        bound_radius: 1.45,
    };
}

// =============================================================================
// Declarative Metadata Catalog for All 28 Formulas
// =============================================================================

const KIFS_KIND_CHOICES: &[&str] = &["Tetrahedron", "Octahedron", "Menger"];
const HYBRID_STEP_CHOICES: &[&str] = &["Off", "Mandelbulb", "Mandelbox", "KifsFold", "Inversion"];
const POLY_SYM_CHOICES: &[&str] = &["Icosahedron", "Dodecahedron"];
const TPMS_KIND_CHOICES: &[&str] = &["Gyroid", "SchwarzP", "SchwarzD", "Neovius"];
const COMPLEX_TPMS_CHOICES: &[&str] = &["Lidinoid", "FischerKochS", "SplitP"];
const ATTRACTOR_CHOICES: &[&str] = &["Aizawa", "Halvorsen", "Thomas"];

pub const FORMULAS: [FormulaMeta; TOTAL_FORMULA_COUNT as usize] = [
    // 0..8: Classic 8 families
    FormulaMeta {
        code: FAMILY_BULB,
        wgsl: "bulb_distance",
        variant: "Mandelbulb",
        display_name: "Mandelbulb",
        hint: "Spherical triplex power fractal (White & Nylander)",
        has_orbit_constant: true,
        signed: false,
        iterates: true,
        params: &[
            ParamMeta {
                name: "power",
                label: "Power",
                hint: "Triplex radial/angular power exponent",
                range: None,
                slider: lin(1.0, 64.0),
                choices: &[],
                order: 20,
            },
            ParamMeta {
                name: "bailout",
                label: "Bailout",
                hint: "Escape radius threshold",
                range: None,
                slider: log(1.0, 64.0),
                choices: &[],
                order: 21,
            },
            ParamMeta {
                name: "rotation_degrees",
                label: "Rotation (°)",
                hint: "Per-iteration Euler rotation in degrees",
                range: None,
                slider: lin(-720.0, 720.0),
                choices: &[],
                order: 22,
            },
            ParamMeta {
                name: "angle_scale",
                label: "Angle scale",
                hint: "Theta and phi angle multipliers",
                range: None,
                slider: lin(-8.0, 8.0),
                choices: &[],
                order: 23,
            },
            ParamMeta {
                name: "angle_phase_degrees",
                label: "Angle phase (°)",
                hint: "Theta and phi phase offsets in degrees",
                range: None,
                slider: lin(-720.0, 720.0),
                choices: &[],
                order: 24,
            },
        ],
    },
    FormulaMeta {
        code: FAMILY_BOX,
        wgsl: "box_distance",
        variant: "Mandelbox",
        display_name: "Mandelbox",
        hint: "Box-fold and sphere-fold conformal fractal (Tglad)",
        has_orbit_constant: true,
        signed: false,
        iterates: true,
        params: &[
            ParamMeta {
                name: "scale",
                label: "Scale",
                hint: "Mandelbox linear scale factor",
                range: None,
                slider: lin(-16.0, 16.0),
                choices: &[],
                order: 20,
            },
            ParamMeta {
                name: "min_radius_ratio",
                label: "Min radius ratio",
                hint: "Inner sphere fold radius as a fraction of fixed radius",
                range: None,
                slider: lin(0.0, 1.0),
                choices: &[],
                order: 21,
            },
            ParamMeta {
                name: "fixed_radius",
                label: "Fixed radius",
                hint: "Outer sphere fold radius",
                range: None,
                slider: log(0.001, 32.0),
                choices: &[],
                order: 22,
            },
            ParamMeta {
                name: "fold_limit",
                label: "Fold limit",
                hint: "Axis-aligned box fold half-extent",
                range: None,
                slider: log(0.001, 32.0),
                choices: &[],
                order: 23,
            },
            ParamMeta {
                name: "rotation_degrees",
                label: "Rotation (°)",
                hint: "Per-iteration Euler rotation in degrees",
                range: None,
                slider: lin(-720.0, 720.0),
                choices: &[],
                order: 24,
            },
        ],
    },
    FormulaMeta {
        code: FAMILY_QUAT,
        wgsl: "quat_distance",
        variant: "QuaternionJulia",
        display_name: "Quaternion Julia",
        hint: "3D slice of a 4D quaternion quadratic Julia set",
        has_orbit_constant: false,
        signed: false,
        iterates: true,
        params: &[
            ParamMeta {
                name: "constant",
                label: "Constant",
                hint: "4D quaternion Julia constant (x, y, z, w)",
                range: None,
                slider: lin(-8.0, 8.0),
                choices: &[],
                order: 20,
            },
            ParamMeta {
                name: "slice_w",
                label: "Slice W",
                hint: "4th-dimension W coordinate of the 3D slice hyperplane",
                range: None,
                slider: lin(-8.0, 8.0),
                choices: &[],
                order: 21,
            },
            ParamMeta {
                name: "rotation_degrees",
                label: "Rotation (°)",
                hint: "4D XW/YW/ZW slice rotation angles in degrees",
                range: None,
                slider: lin(-720.0, 720.0),
                choices: &[],
                order: 22,
            },
            ParamMeta {
                name: "bailout",
                label: "Bailout",
                hint: "Escape radius threshold",
                range: None,
                slider: log(1.0, 64.0),
                choices: &[],
                order: 23,
            },
        ],
    },
    FormulaMeta {
        code: FAMILY_KIFS,
        wgsl: "kifs_distance",
        variant: "Kifs",
        display_name: "KIFS",
        hint: "Kaleidoscopic Iterated Function System (Tetrahedron, Octahedron, Menger)",
        has_orbit_constant: false,
        signed: true,
        iterates: true,
        params: &[
            ParamMeta {
                name: "kind",
                label: "Kind",
                hint: "Polyhedral reflection symmetry group",
                range: None,
                slider: None,
                choices: KIFS_KIND_CHOICES,
                order: 20,
            },
            ParamMeta {
                name: "scale",
                label: "Scale",
                hint: "Homothety scale per fold step",
                range: None,
                slider: lin(1.0, 16.0),
                choices: &[],
                order: 21,
            },
            ParamMeta {
                name: "offset",
                label: "Offset",
                hint: "Fold center offset vector",
                range: None,
                slider: lin(-16.0, 16.0),
                choices: &[],
                order: 22,
            },
            ParamMeta {
                name: "rotation_degrees",
                label: "Rotation (°)",
                hint: "Per-iteration Euler rotation in degrees",
                range: None,
                slider: lin(-720.0, 720.0),
                choices: &[],
                order: 23,
            },
        ],
    },
    FormulaMeta {
        code: FAMILY_KLEINIAN,
        wgsl: "kleinian_distance",
        variant: "Kleinian",
        display_name: "Kleinian",
        hint: "Maskit slice Kleinian limit set via box wrap and sphere inversion",
        has_orbit_constant: false,
        signed: true,
        iterates: true,
        params: &[
            ParamMeta {
                name: "a",
                label: "A",
                hint: "Maskit parameter Re(mu) (vertical period)",
                range: None,
                slider: lin(1.0, 8.0),
                choices: &[],
                order: 20,
            },
            ParamMeta {
                name: "b",
                label: "B",
                hint: "Maskit parameter Im(mu) (horizontal skew)",
                range: None,
                slider: lin(-8.0, 8.0),
                choices: &[],
                order: 21,
            },
            ParamMeta {
                name: "bound_radius",
                label: "Bound radius",
                hint: "Spherical clip bound (0 = unbounded)",
                range: None,
                slider: lin(0.0, 32.0),
                choices: &[],
                order: 22,
            },
        ],
    },
    FormulaMeta {
        code: FAMILY_PSEUDO_KLEINIAN,
        wgsl: "pseudo_kleinian_distance",
        variant: "PseudoKleinian",
        display_name: "Pseudo-Kleinian",
        hint: "Knighty Mandelbox-Kleinian limit set with cylindrical Thingy estimator",
        has_orbit_constant: false,
        signed: true,
        iterates: true,
        params: &[
            ParamMeta {
                name: "box_size",
                label: "Box size",
                hint: "3D box fold half-size per axis",
                range: None,
                slider: lin(0.0, 16.0),
                choices: &[],
                order: 20,
            },
            ParamMeta {
                name: "size",
                label: "Size",
                hint: "Inversion sphere squared radius",
                range: None,
                slider: log(0.001, 32.0),
                choices: &[],
                order: 21,
            },
            ParamMeta {
                name: "c",
                label: "C",
                hint: "Translation shift vector per iteration",
                range: None,
                slider: lin(-8.0, 8.0),
                choices: &[],
                order: 22,
            },
            ParamMeta {
                name: "thickness",
                label: "Thickness",
                hint: "Surface shell thickness offset",
                range: None,
                slider: log(0.000001, 1.0),
                choices: &[],
                order: 23,
            },
            ParamMeta {
                name: "offset",
                label: "Offset",
                hint: "Thingy estimator origin offset",
                range: None,
                slider: lin(-16.0, 16.0),
                choices: &[],
                order: 24,
            },
            ParamMeta {
                name: "bound_radius",
                label: "Bound radius",
                hint: "Spherical clip bound (0 = unbounded)",
                range: None,
                slider: lin(0.0, 32.0),
                choices: &[],
                order: 25,
            },
        ],
    },
    FormulaMeta {
        code: FAMILY_APOLLONIAN,
        wgsl: "apollonian_distance",
        variant: "Apollonian",
        display_name: "Apollonian",
        hint: "3D Apollonian gasket foam via periodic lattice fold and sphere inversion",
        has_orbit_constant: false,
        signed: false,
        iterates: true,
        params: &[
            ParamMeta {
                name: "scale",
                label: "Scale",
                hint: "Apollonian sphere inversion numerator",
                range: None,
                slider: log(0.01, 32.0),
                choices: &[],
                order: 20,
            },
            ParamMeta {
                name: "bound_radius",
                label: "Bound radius",
                hint: "Spherical clip bound (0 = unbounded)",
                range: None,
                slider: lin(0.0, 32.0),
                choices: &[],
                order: 21,
            },
        ],
    },
    FormulaMeta {
        code: FAMILY_HYBRID,
        wgsl: "hybrid_distance",
        variant: "Hybrid",
        display_name: "Hybrid",
        hint: "4-slot alternating hybrid of Mandelbulb, Mandelbox, KIFS, and Apollonian steps",
        has_orbit_constant: false,
        signed: false,
        iterates: true,
        params: &[
            ParamMeta {
                name: "steps",
                label: "Steps",
                hint: "Sequence of 4 operator slots cycled per iteration",
                range: None,
                slider: None,
                choices: HYBRID_STEP_CHOICES,
                order: 12,
            },
            ParamMeta {
                name: "apollonian_scale",
                label: "Apollonian scale",
                hint: "Scale for Inversion slots",
                range: None,
                slider: log(0.01, 32.0),
                choices: &[],
                order: 20,
            },
            ParamMeta {
                name: "bailout",
                label: "Bailout",
                hint: "Escape radius threshold",
                range: None,
                slider: log(0.01, 64.0),
                choices: &[],
                order: 21,
            },
        ],
    },
    // 8..28: 20 Extended Fractals & Procedural Shapes
    FormulaMeta {
        code: FAMILY_AMAZING_SURFACE,
        wgsl: "amazing_surface_distance",
        variant: "AmazingSurface",
        display_name: "Amazing Surface (KaliBox)",
        hint: "Kali's Amazing Surface: independent 3-axis box fold, pre-sphere shift, and sphere inversion for architectural arches and gothic ruins",
        has_orbit_constant: true,
        signed: true,
        iterates: true,
        params: &[
            ParamMeta {
                name: "scale",
                label: "Scale",
                hint: "Conformal scale factor (negative values invert orientation each step)",
                range: Some((-4.0, 4.0)),
                slider: lin(-2.5, 2.5),
                choices: &[],
                order: 20,
            },
            ParamMeta {
                name: "min_radius",
                label: "Min radius",
                hint: "Inner sphere fold clamping radius",
                range: Some((0.01, 2.0)),
                slider: lin(0.05, 1.0),
                choices: &[],
                order: 21,
            },
            ParamMeta {
                name: "fixed_radius",
                label: "Fixed radius",
                hint: "Outer sphere fold inversion radius",
                range: Some((0.1, 4.0)),
                slider: lin(0.4, 2.0),
                choices: &[],
                order: 22,
            },
            ParamMeta {
                name: "fold_limit",
                label: "Fold limit",
                hint: "Per-axis box fold limit (set Z=0 to disable Z folding for planar KaliSurface)",
                range: Some((0.0, 4.0)),
                slider: lin(0.0, 2.0),
                choices: &[],
                order: 23,
            },
            ParamMeta {
                name: "shift",
                label: "Pre-sphere shift",
                hint: "Translation applied between box fold and sphere fold",
                range: Some((-3.0, 3.0)),
                slider: lin(-1.5, 1.5),
                choices: &[],
                order: 24,
            },
            ParamMeta {
                name: "rotation_degrees",
                label: "Rotation (°)",
                hint: "Per-iteration Euler rotation in degrees",
                range: Some((-360.0, 360.0)),
                slider: lin(-180.0, 180.0),
                choices: &[],
                order: 25,
            },
            ParamMeta {
                name: "bound_radius",
                label: "Bound radius",
                hint: "Outer bounding sphere radius",
                range: Some((0.2, 20.0)),
                slider: lin(0.8, 5.0),
                choices: &[],
                order: 26,
            },
        ],
    },
    FormulaMeta {
        code: FAMILY_BRISTORBROT,
        wgsl: "bristorbrot_distance",
        variant: "Bristorbrot",
        display_name: "Bristorbrot",
        hint: "Bristor's quadratic 3D hypercomplex fractal with coral-like dendritic filaments",
        has_orbit_constant: true,
        signed: false,
        iterates: true,
        params: &[
            ParamMeta {
                name: "scale",
                label: "Scale",
                hint: "Quadratic step multiplier",
                range: Some((0.2, 3.0)),
                slider: lin(0.5, 1.8),
                choices: &[],
                order: 20,
            },
            ParamMeta {
                name: "bailout",
                label: "Bailout",
                hint: "Escape radius threshold",
                range: Some((1.5, 64.0)),
                slider: lin(2.0, 12.0),
                choices: &[],
                order: 21,
            },
            ParamMeta {
                name: "y_sign",
                label: "Y cross sign",
                hint: "Cross-term sign on Y (2xy - yz*sign)",
                range: Some((-2.0, 2.0)),
                slider: lin(-1.5, 1.5),
                choices: &[],
                order: 22,
            },
            ParamMeta {
                name: "z_sign",
                label: "Z cross sign",
                hint: "Cross-term sign on Z (2xz + yz*sign)",
                range: Some((-2.0, 2.0)),
                slider: lin(-1.5, 1.5),
                choices: &[],
                order: 23,
            },
            ParamMeta {
                name: "twist",
                label: "Twist",
                hint: "Helical twist around the X axis per iteration",
                range: Some((-3.0, 3.0)),
                slider: lin(-1.0, 1.0),
                choices: &[],
                order: 24,
            },
            ParamMeta {
                name: "rotation_degrees",
                label: "Rotation (°)",
                hint: "Per-iteration Euler rotation in degrees",
                range: Some((-360.0, 360.0)),
                slider: lin(-180.0, 180.0),
                choices: &[],
                order: 25,
            },
        ],
    },
    FormulaMeta {
        code: FAMILY_BURNING_SHIP,
        wgsl: "burning_ship_distance",
        variant: "BurningShip",
        display_name: "Burning Ship 3D",
        hint: "3D Burning Ship / Mandelbulb-Abs with per-axis absolute-value folding before spherical exponentiation",
        has_orbit_constant: true,
        signed: false,
        iterates: true,
        params: &[
            ParamMeta {
                name: "power",
                label: "Power",
                hint: "Spherical exponentiation power",
                range: Some((1.5, 24.0)),
                slider: lin(2.0, 12.0),
                choices: &[],
                order: 20,
            },
            ParamMeta {
                name: "bailout",
                label: "Bailout",
                hint: "Escape radius threshold",
                range: Some((1.5, 64.0)),
                slider: lin(2.0, 12.0),
                choices: &[],
                order: 21,
            },
            ParamMeta {
                name: "abs_axes",
                label: "Abs fold axes",
                hint: "Blend between raw z (0) and |z| (1) per axis before raising to power",
                range: Some((0.0, 1.0)),
                slider: lin(0.0, 1.0),
                choices: &[],
                order: 22,
            },
            ParamMeta {
                name: "shift",
                label: "Fold shift",
                hint: "Offset applied after absolute-value fold",
                range: Some((-2.0, 2.0)),
                slider: lin(-0.8, 0.8),
                choices: &[],
                order: 23,
            },
            ParamMeta {
                name: "rotation_degrees",
                label: "Rotation (°)",
                hint: "Per-iteration Euler rotation in degrees",
                range: Some((-360.0, 360.0)),
                slider: lin(-180.0, 180.0),
                choices: &[],
                order: 24,
            },
        ],
    },
    FormulaMeta {
        code: FAMILY_KALEIDO_TWIST,
        wgsl: "kaleido_twist_distance",
        variant: "KaleidoTwist",
        display_name: "Kaleido Twist IFS",
        hint: "Marble Marcher / Menger-Twist IFS: pre-rotation, abs fold, Menger sorting fold, scale, shift, and post-rotation",
        has_orbit_constant: false,
        signed: true,
        iterates: true,
        params: &[
            ParamMeta {
                name: "scale",
                label: "Scale",
                hint: "Homothety scale per fold iteration",
                range: Some((1.1, 4.0)),
                slider: lin(1.35, 2.8),
                choices: &[],
                order: 20,
            },
            ParamMeta {
                name: "shift",
                label: "Shift",
                hint: "Translation vector subtracted after scaling",
                range: Some((-6.0, 6.0)),
                slider: lin(0.0, 4.0),
                choices: &[],
                order: 21,
            },
            ParamMeta {
                name: "rotation_degrees",
                label: "Pre-rotation (°)",
                hint: "Euler rotation applied before the Menger sorting fold",
                range: Some((-360.0, 360.0)),
                slider: lin(-90.0, 90.0),
                choices: &[],
                order: 22,
            },
            ParamMeta {
                name: "post_rotation_degrees",
                label: "Post-rotation (°)",
                hint: "Second Euler rotation applied after scaling and shifting",
                range: Some((-360.0, 360.0)),
                slider: lin(-90.0, 90.0),
                choices: &[],
                order: 23,
            },
            ParamMeta {
                name: "clamp_z",
                label: "Z fold limit",
                hint: "Half-space Z fold limit (Marble Marcher Z-clamp)",
                range: Some((0.0, 4.0)),
                slider: lin(0.0, 2.0),
                choices: &[],
                order: 24,
            },
            ParamMeta {
                name: "bound_radius",
                label: "Bound radius",
                hint: "Outer bounding sphere radius",
                range: Some((0.2, 20.0)),
                slider: lin(1.0, 5.0),
                choices: &[],
                order: 25,
            },
        ],
    },
    FormulaMeta {
        code: FAMILY_POLYHEDRAL_KIFS,
        wgsl: "polyhedral_kifs_distance",
        variant: "PolyhedralKifs",
        display_name: "Polyhedral KIFS (Icosa/Dodeca)",
        hint: "Knighty's golden-ratio icosahedral & dodecahedral reflection group fractal",
        has_orbit_constant: false,
        signed: true,
        iterates: true,
        params: &[
            ParamMeta {
                name: "symmetry",
                label: "Symmetry",
                hint: "Icosahedral or Dodecahedral reflection group",
                range: None,
                slider: None,
                choices: POLY_SYM_CHOICES,
                order: 20,
            },
            ParamMeta {
                name: "scale",
                label: "Scale",
                hint: "Scale factor per polyhedral fold step",
                range: Some((1.1, 4.0)),
                slider: lin(1.4, 2.8),
                choices: &[],
                order: 21,
            },
            ParamMeta {
                name: "offset",
                label: "Vertex offset",
                hint: "Polyhedral fold vertex direction/offset",
                range: Some((-3.0, 3.0)),
                slider: lin(-1.5, 1.5),
                choices: &[],
                order: 22,
            },
            ParamMeta {
                name: "shell_thickness",
                label: "Shell thickness",
                hint: "Polyhedral strut/shell thickness",
                range: Some((0.01, 1.0)),
                slider: lin(0.02, 0.4),
                choices: &[],
                order: 23,
            },
            ParamMeta {
                name: "rotation_degrees",
                label: "Rotation (°)",
                hint: "Per-iteration Euler rotation in degrees",
                range: Some((-360.0, 360.0)),
                slider: lin(-90.0, 90.0),
                choices: &[],
                order: 24,
            },
            ParamMeta {
                name: "bound_radius",
                label: "Bound radius",
                hint: "Outer bounding sphere radius",
                range: Some((0.2, 20.0)),
                slider: lin(0.8, 4.0),
                choices: &[],
                order: 25,
            },
        ],
    },
    FormulaMeta {
        code: FAMILY_HARMONIC_BULB,
        wgsl: "harmonic_bulb_distance",
        variant: "HarmonicBulb",
        display_name: "Harmonic Bulb",
        hint: "Spherical-harmonics modulated Mandelbulb with organic botanical and marine symmetry",
        has_orbit_constant: true,
        signed: false,
        iterates: true,
        params: &[
            ParamMeta {
                name: "power",
                label: "Power",
                hint: "Base radial/angular bulb power",
                range: Some((1.5, 24.0)),
                slider: lin(2.0, 12.0),
                choices: &[],
                order: 20,
            },
            ParamMeta {
                name: "bailout",
                label: "Bailout",
                hint: "Escape radius threshold",
                range: Some((1.5, 64.0)),
                slider: lin(2.0, 10.0),
                choices: &[],
                order: 21,
            },
            ParamMeta {
                name: "harmonic_l",
                label: "Harmonic L (lat)",
                hint: "Latitude spherical harmonic degree L",
                range: Some((0.0, 24.0)),
                slider: lin(1.0, 12.0),
                choices: &[],
                order: 22,
            },
            ParamMeta {
                name: "harmonic_m",
                label: "Harmonic M (lon)",
                hint: "Longitude spherical harmonic order M",
                range: Some((0.0, 24.0)),
                slider: lin(1.0, 12.0),
                choices: &[],
                order: 23,
            },
            ParamMeta {
                name: "amplitude",
                label: "Harmonic amp",
                hint: "Modulation depth of the spherical harmonic perturbation",
                range: Some((-1.0, 1.0)),
                slider: lin(-0.6, 0.6),
                choices: &[],
                order: 24,
            },
            ParamMeta {
                name: "rotation_degrees",
                label: "Rotation (°)",
                hint: "Per-iteration Euler rotation in degrees",
                range: Some((-360.0, 360.0)),
                slider: lin(-180.0, 180.0),
                choices: &[],
                order: 25,
            },
        ],
    },
    FormulaMeta {
        code: FAMILY_MANDELBAR,
        wgsl: "mandelbar_distance",
        variant: "Mandelbar3D",
        display_name: "Mandelbar 3D (Tricorn)",
        hint: "3D Tricorn / Mandelbar fractal using complex conjugation across azimuthal and polar angles",
        has_orbit_constant: true,
        signed: false,
        iterates: true,
        params: &[
            ParamMeta {
                name: "power",
                label: "Power",
                hint: "Triplex exponentiation power",
                range: Some((1.5, 24.0)),
                slider: lin(2.0, 12.0),
                choices: &[],
                order: 20,
            },
            ParamMeta {
                name: "bailout",
                label: "Bailout",
                hint: "Escape radius threshold",
                range: Some((1.5, 64.0)),
                slider: lin(2.0, 10.0),
                choices: &[],
                order: 21,
            },
            ParamMeta {
                name: "conjugate_xy",
                label: "Conjugate XY",
                hint: "Azimuthal conjugation multiplier (-1 = full Tricorn conjugation)",
                range: Some((-2.0, 2.0)),
                slider: lin(-1.5, 1.5),
                choices: &[],
                order: 22,
            },
            ParamMeta {
                name: "conjugate_z",
                label: "Conjugate Z",
                hint: "Polar elevation conjugation multiplier",
                range: Some((-2.0, 2.0)),
                slider: lin(-1.5, 1.5),
                choices: &[],
                order: 23,
            },
            ParamMeta {
                name: "twist",
                label: "Twist",
                hint: "Azimuthal twist per iteration (radians)",
                range: Some((-std::f64::consts::PI, std::f64::consts::PI)),
                slider: lin(-1.5, 1.5),
                choices: &[],
                order: 24,
            },
            ParamMeta {
                name: "rotation_degrees",
                label: "Rotation (°)",
                hint: "Per-iteration Euler rotation in degrees",
                range: Some((-360.0, 360.0)),
                slider: lin(-180.0, 180.0),
                choices: &[],
                order: 25,
            },
        ],
    },
    FormulaMeta {
        code: FAMILY_MOBIUS_IFS,
        wgsl: "mobius_ifs_distance",
        variant: "MobiusIfs",
        display_name: "Möbius Sphere IFS",
        hint: "Conformal Möbius group IFS combining periodic lattice wrap, spiral twist, and sphere inversion",
        has_orbit_constant: false,
        signed: true,
        iterates: true,
        params: &[
            ParamMeta {
                name: "scale",
                label: "Scale",
                hint: "Conformal scale multiplier per step",
                range: Some((0.5, 4.0)),
                slider: lin(1.0, 2.5),
                choices: &[],
                order: 20,
            },
            ParamMeta {
                name: "min_radius",
                label: "Min radius",
                hint: "Inner sphere clamping radius",
                range: Some((0.01, 1.5)),
                slider: lin(0.05, 0.8),
                choices: &[],
                order: 21,
            },
            ParamMeta {
                name: "fixed_radius",
                label: "Fixed radius",
                hint: "Inversion sphere radius",
                range: Some((0.1, 3.0)),
                slider: lin(0.5, 1.8),
                choices: &[],
                order: 22,
            },
            ParamMeta {
                name: "lattice_pitch",
                label: "Lattice pitch",
                hint: "3D periodic cell wrap dimensions",
                range: Some((0.2, 4.0)),
                slider: lin(0.5, 2.5),
                choices: &[],
                order: 23,
            },
            ParamMeta {
                name: "twist",
                label: "Spiral twist",
                hint: "Helical twist angle per unit Z",
                range: Some((-2.0, 2.0)),
                slider: lin(-0.8, 0.8),
                choices: &[],
                order: 24,
            },
            ParamMeta {
                name: "rotation_degrees",
                label: "Rotation (°)",
                hint: "Per-iteration Euler rotation in degrees",
                range: Some((-360.0, 360.0)),
                slider: lin(-90.0, 90.0),
                choices: &[],
                order: 25,
            },
            ParamMeta {
                name: "bound_radius",
                label: "Bound radius",
                hint: "Outer bounding sphere radius",
                range: Some((0.2, 20.0)),
                slider: lin(0.8, 4.0),
                choices: &[],
                order: 26,
            },
        ],
    },
    FormulaMeta {
        code: FAMILY_GYROID_TPMS,
        wgsl: "gyroid_tpms_distance",
        variant: "GyroidTpms",
        display_name: "Gyroid / Schwarz TPMS",
        hint: "Triply Periodic Minimal Surface (Gyroid, Schwarz P, Schwarz D, Neovius) with multi-octave FBM fractal detailing",
        has_orbit_constant: false,
        signed: true,
        iterates: false,
        params: &[
            ParamMeta {
                name: "surface",
                label: "Surface type",
                hint: "Minimal surface topology (Gyroid, Schwarz P, Schwarz D, Neovius)",
                range: None,
                slider: None,
                choices: TPMS_KIND_CHOICES,
                order: 20,
            },
            ParamMeta {
                name: "frequency",
                label: "Frequency",
                hint: "Base spatial wave frequency",
                range: Some((0.2, 24.0)),
                slider: lin(1.0, 10.0),
                choices: &[],
                order: 21,
            },
            ParamMeta {
                name: "thickness",
                label: "Wall thickness",
                hint: "Minimal surface shell half-thickness",
                range: Some((0.002, 0.8)),
                slider: lin(0.01, 0.3),
                choices: &[],
                order: 22,
            },
            ParamMeta {
                name: "bias",
                label: "Iso-surface bias",
                hint: "Iso-level shift (-1..1 pinches or swells channels)",
                range: Some((-1.5, 1.5)),
                slider: lin(-0.8, 0.8),
                choices: &[],
                order: 23,
            },
            ParamMeta {
                name: "octaves",
                label: "Fractal octaves",
                hint: "Number of recursive TPMS detail octaves (1..6)",
                range: Some((1.0, 6.0)),
                slider: lin(1.0, 5.0),
                choices: &[],
                order: 24,
            },
            ParamMeta {
                name: "lacunarity",
                label: "Lacunarity",
                hint: "Frequency multiplier per octave",
                range: Some((1.1, 4.0)),
                slider: lin(1.5, 3.0),
                choices: &[],
                order: 25,
            },
            ParamMeta {
                name: "gain",
                label: "Octave gain",
                hint: "Amplitude persistence per octave",
                range: Some((0.0, 0.8)),
                slider: lin(0.1, 0.6),
                choices: &[],
                order: 26,
            },
            ParamMeta {
                name: "twist",
                label: "Spatial twist",
                hint: "Helical coordinate twist along Y",
                range: Some((-3.0, 3.0)),
                slider: lin(-1.2, 1.2),
                choices: &[],
                order: 27,
            },
            ParamMeta {
                name: "bound_radius",
                label: "Bound radius",
                hint: "Outer bounding sphere radius",
                range: Some((0.2, 20.0)),
                slider: lin(0.5, 4.0),
                choices: &[],
                order: 28,
            },
        ],
    },
    FormulaMeta {
        code: FAMILY_LIDINOID_TPMS,
        wgsl: "lidinoid_tpms_distance",
        variant: "LidinoidTpms",
        display_name: "Lidinoid / Fischer-Koch TPMS",
        hint: "Higher-order chiral minimal surfaces (Lidinoid, Fischer-Koch S, Split-P) with double-gyroid channels",
        has_orbit_constant: false,
        signed: true,
        iterates: false,
        params: &[
            ParamMeta {
                name: "surface",
                label: "Surface type",
                hint: "Higher-order TPMS family (Lidinoid, FischerKochS, SplitP)",
                range: None,
                slider: None,
                choices: COMPLEX_TPMS_CHOICES,
                order: 20,
            },
            ParamMeta {
                name: "frequency",
                label: "Frequency",
                hint: "Spatial wave frequency",
                range: Some((0.2, 20.0)),
                slider: lin(1.0, 8.0),
                choices: &[],
                order: 21,
            },
            ParamMeta {
                name: "thickness",
                label: "Wall thickness",
                hint: "Minimal surface shell thickness",
                range: Some((0.002, 0.8)),
                slider: lin(0.01, 0.3),
                choices: &[],
                order: 22,
            },
            ParamMeta {
                name: "bias",
                label: "Iso-surface bias",
                hint: "Iso-value level shift",
                range: Some((-1.5, 1.5)),
                slider: lin(-0.8, 0.8),
                choices: &[],
                order: 23,
            },
            ParamMeta {
                name: "harmonic_mix",
                label: "Harmonic mix",
                hint: "Balance between primary and secondary harmonic terms",
                range: Some((0.0, 2.0)),
                slider: lin(0.1, 1.2),
                choices: &[],
                order: 24,
            },
            ParamMeta {
                name: "radial_warp",
                label: "Radial warp",
                hint: "Spherical radial frequency modulation",
                range: Some((-2.0, 2.0)),
                slider: lin(-0.8, 0.8),
                choices: &[],
                order: 25,
            },
            ParamMeta {
                name: "bound_radius",
                label: "Bound radius",
                hint: "Outer bounding sphere radius",
                range: Some((0.2, 20.0)),
                slider: lin(0.5, 4.0),
                choices: &[],
                order: 26,
            },
        ],
    },
    FormulaMeta {
        code: FAMILY_ATTRACTOR_TUBE,
        wgsl: "attractor_tube_distance",
        variant: "AttractorTube",
        display_name: "Strange Attractor Flow",
        hint: "3D chaotic strange-attractor stream-surface (Aizawa, Halvorsen, Thomas) advected along RK2 trajectories",
        has_orbit_constant: false,
        signed: true,
        iterates: true,
        params: &[
            ParamMeta {
                name: "system",
                label: "Dynamical system",
                hint: "Chaotic ODE system (Aizawa, Halvorsen, Thomas)",
                range: None,
                slider: None,
                choices: ATTRACTOR_CHOICES,
                order: 20,
            },
            ParamMeta {
                name: "step_dt",
                label: "Advection step",
                hint: "ODE integration time step per iteration",
                range: Some((0.005, 0.2)),
                slider: lin(0.01, 0.1),
                choices: &[],
                order: 21,
            },
            ParamMeta {
                name: "tube_radius",
                label: "Ribbon thickness",
                hint: "Stream-ribbon / filament thickness",
                range: Some((0.005, 0.4)),
                slider: lin(0.02, 0.18),
                choices: &[],
                order: 22,
            },
            ParamMeta {
                name: "swirl",
                label: "Vorticity swirl",
                hint: "Rotational velocity multiplier",
                range: Some((0.1, 4.0)),
                slider: lin(0.4, 2.5),
                choices: &[],
                order: 23,
            },
            ParamMeta {
                name: "param_a",
                label: "System param A",
                hint: "Primary attractor bifurcation constant",
                range: Some((0.05, 3.0)),
                slider: lin(0.2, 2.0),
                choices: &[],
                order: 24,
            },
            ParamMeta {
                name: "param_b",
                label: "System param B",
                hint: "Secondary attractor damping/folding constant",
                range: Some((0.05, 3.0)),
                slider: lin(0.2, 2.0),
                choices: &[],
                order: 25,
            },
            ParamMeta {
                name: "bound_radius",
                label: "Bound radius",
                hint: "Outer bounding sphere radius",
                range: Some((0.2, 20.0)),
                slider: lin(0.8, 4.0),
                choices: &[],
                order: 26,
            },
        ],
    },
    FormulaMeta {
        code: FAMILY_TORUS_KNOT,
        wgsl: "torus_knot_braid_distance",
        variant: "TorusKnotBraid",
        display_name: "Torus Knot Braid",
        hint: "Exact (p, q) torus knot with recursive N-strand twisted cable braiding",
        has_orbit_constant: false,
        signed: true,
        iterates: false,
        params: &[
            ParamMeta {
                name: "winds_p",
                label: "Winds P (toroidal)",
                hint: "Longitudinal windings around the torus",
                range: Some((1.0, 16.0)),
                slider: lin(1.0, 9.0),
                choices: &[],
                order: 20,
            },
            ParamMeta {
                name: "winds_q",
                label: "Winds Q (poloidal)",
                hint: "Meridional windings through the torus hole",
                range: Some((1.0, 24.0)),
                slider: lin(1.0, 13.0),
                choices: &[],
                order: 21,
            },
            ParamMeta {
                name: "major_radius",
                label: "Major radius",
                hint: "Main torus ring radius",
                range: Some((0.2, 3.0)),
                slider: lin(0.4, 1.5),
                choices: &[],
                order: 22,
            },
            ParamMeta {
                name: "minor_radius",
                label: "Minor radius",
                hint: "Poloidal knot excursion radius",
                range: Some((0.05, 1.5)),
                slider: lin(0.1, 0.7),
                choices: &[],
                order: 23,
            },
            ParamMeta {
                name: "tube_radius",
                label: "Strand radius",
                hint: "Cross-sectional radius of the braided strand",
                range: Some((0.01, 0.5)),
                slider: lin(0.02, 0.2),
                choices: &[],
                order: 24,
            },
            ParamMeta {
                name: "sub_strands",
                label: "Sub-strands",
                hint: "Number of braided sub-cables twisted around the core (1..12)",
                range: Some((1.0, 12.0)),
                slider: lin(1.0, 8.0),
                choices: &[],
                order: 25,
            },
            ParamMeta {
                name: "braid_depth",
                label: "Braid relief",
                hint: "Radial depth of the sub-strand cable twist",
                range: Some((0.0, 0.25)),
                slider: lin(0.0, 0.12),
                choices: &[],
                order: 26,
            },
            ParamMeta {
                name: "bound_radius",
                label: "Bound radius",
                hint: "Outer bounding sphere radius",
                range: Some((0.2, 20.0)),
                slider: lin(0.8, 4.0),
                choices: &[],
                order: 27,
            },
        ],
    },
    FormulaMeta {
        code: FAMILY_HOPF_FIBRATION,
        wgsl: "hopf_fibration_distance",
        variant: "HopfFibration",
        display_name: "Hopf Fibration Tori",
        hint: "Stereographic projection of S^3 -> S^2 Hopf fibers forming Villarceau circles and linked Clifford tori",
        has_orbit_constant: false,
        signed: true,
        iterates: false,
        params: &[
            ParamMeta {
                name: "fibers",
                label: "Fibers per shell",
                hint: "Number of linked Villarceau fiber rings per torus shell (2..24)",
                range: Some((2.0, 24.0)),
                slider: lin(3.0, 16.0),
                choices: &[],
                order: 20,
            },
            ParamMeta {
                name: "shells",
                label: "Concentric shells",
                hint: "Number of nested latitude torus shells (1..6)",
                range: Some((1.0, 6.0)),
                slider: lin(1.0, 5.0),
                choices: &[],
                order: 21,
            },
            ParamMeta {
                name: "latitude",
                label: "Base latitude",
                hint: "S^2 polar angle controlling torus aspect ratio",
                range: Some((0.15, 1.45)),
                slider: lin(0.3, 1.25),
                choices: &[],
                order: 22,
            },
            ParamMeta {
                name: "tube_radius",
                label: "Fiber radius",
                hint: "Cross-sectional radius of each Hopf circle",
                range: Some((0.01, 0.35)),
                slider: lin(0.02, 0.15),
                choices: &[],
                order: 23,
            },
            ParamMeta {
                name: "twist",
                label: "Hopf chirality",
                hint: "Fiber helicity / twist multiplier",
                range: Some((-3.0, 3.0)),
                slider: lin(-1.5, 1.5),
                choices: &[],
                order: 24,
            },
            ParamMeta {
                name: "scale",
                label: "Projection scale",
                hint: "Stereographic projection scale",
                range: Some((0.2, 2.5)),
                slider: lin(0.4, 1.5),
                choices: &[],
                order: 25,
            },
            ParamMeta {
                name: "bound_radius",
                label: "Bound radius",
                hint: "Outer bounding sphere radius",
                range: Some((0.2, 20.0)),
                slider: lin(0.8, 4.0),
                choices: &[],
                order: 26,
            },
        ],
    },
    FormulaMeta {
        code: FAMILY_DUPIN_CYCLIDE,
        wgsl: "dupin_cyclide_distance",
        variant: "DupinCyclide",
        display_name: "Dupin Cyclide",
        hint: "Quartic Dupin cyclide (inversion of a torus) with Crescent/Horned/Spindle topology and carved ribs",
        has_orbit_constant: false,
        signed: true,
        iterates: false,
        params: &[
            ParamMeta {
                name: "major_a",
                label: "Major axis A",
                hint: "Focal ellipse semi-major axis a",
                range: Some((0.2, 3.0)),
                slider: lin(0.5, 1.8),
                choices: &[],
                order: 20,
            },
            ParamMeta {
                name: "minor_b",
                label: "Minor axis B",
                hint: "Focal ellipse semi-minor axis b",
                range: Some((0.2, 3.0)),
                slider: lin(0.4, 1.8),
                choices: &[],
                order: 21,
            },
            ParamMeta {
                name: "eccentricity_c",
                label: "Asymmetry C",
                hint: "Focal linear eccentricity c (controls horn pinch)",
                range: Some((0.0, 2.0)),
                slider: lin(0.0, 1.0),
                choices: &[],
                order: 22,
            },
            ParamMeta {
                name: "radius_d",
                label: "Tube radius D",
                hint: "Cyclide mean tube radius parameter d",
                range: Some((0.05, 2.0)),
                slider: lin(0.15, 1.0),
                choices: &[],
                order: 23,
            },
            ParamMeta {
                name: "shell_thickness",
                label: "Shell thickness",
                hint: "Wall thickness (0 = solid cyclide interior)",
                range: Some((0.0, 0.5)),
                slider: lin(0.0, 0.2),
                choices: &[],
                order: 24,
            },
            ParamMeta {
                name: "rib_frequency",
                label: "Rib count",
                hint: "Number of conformal ribs along the cyclide lines of curvature",
                range: Some((0.0, 48.0)),
                slider: lin(0.0, 24.0),
                choices: &[],
                order: 25,
            },
            ParamMeta {
                name: "rib_depth",
                label: "Rib depth",
                hint: "Amplitude of curvature-aligned fluting",
                range: Some((-0.2, 0.2)),
                slider: lin(-0.08, 0.08),
                choices: &[],
                order: 26,
            },
            ParamMeta {
                name: "bound_radius",
                label: "Bound radius",
                hint: "Outer bounding sphere radius",
                range: Some((0.2, 20.0)),
                slider: lin(0.8, 4.0),
                choices: &[],
                order: 27,
            },
        ],
    },
    FormulaMeta {
        code: FAMILY_SUPERFORMULA,
        wgsl: "superformula_distance",
        variant: "Superformula3D",
        display_name: "3D Superformula (Radiolaria)",
        hint: "3D Gielis superformula spherical product with Haeckel radiolaria skeletal pores",
        has_orbit_constant: false,
        signed: true,
        iterates: false,
        params: &[
            ParamMeta {
                name: "m1",
                label: "Symmetry M1 (lon)",
                hint: "Longitudinal rotational symmetry lobes m1",
                range: Some((0.0, 32.0)),
                slider: lin(1.0, 16.0),
                choices: &[],
                order: 20,
            },
            ParamMeta {
                name: "n1_1",
                label: "Pinch N1.1",
                hint: "Longitudinal outer exponent n1",
                range: Some((0.15, 20.0)),
                slider: lin(0.3, 6.0),
                choices: &[],
                order: 21,
            },
            ParamMeta {
                name: "n2_1",
                label: "Cosine N2.1",
                hint: "Longitudinal cosine exponent n2",
                range: Some((0.15, 20.0)),
                slider: lin(0.3, 6.0),
                choices: &[],
                order: 22,
            },
            ParamMeta {
                name: "n3_1",
                label: "Sine N3.1",
                hint: "Longitudinal sine exponent n3",
                range: Some((0.15, 20.0)),
                slider: lin(0.3, 6.0),
                choices: &[],
                order: 23,
            },
            ParamMeta {
                name: "m2",
                label: "Symmetry M2 (lat)",
                hint: "Latitudinal rotational symmetry lobes m2",
                range: Some((0.0, 32.0)),
                slider: lin(1.0, 16.0),
                choices: &[],
                order: 24,
            },
            ParamMeta {
                name: "n1_2",
                label: "Pinch N1.2",
                hint: "Latitudinal outer exponent n1",
                range: Some((0.15, 20.0)),
                slider: lin(0.3, 6.0),
                choices: &[],
                order: 25,
            },
            ParamMeta {
                name: "n2_2",
                label: "Cosine N2.2",
                hint: "Latitudinal cosine exponent n2",
                range: Some((0.15, 20.0)),
                slider: lin(0.3, 6.0),
                choices: &[],
                order: 26,
            },
            ParamMeta {
                name: "n3_2",
                label: "Sine N3.2",
                hint: "Latitudinal sine exponent n3",
                range: Some((0.15, 20.0)),
                slider: lin(0.3, 6.0),
                choices: &[],
                order: 27,
            },
            ParamMeta {
                name: "shell_thickness",
                label: "Shell thickness",
                hint: "Hollow radiolaria shell thickness (0 = solid)",
                range: Some((0.0, 0.5)),
                slider: lin(0.0, 0.2),
                choices: &[],
                order: 28,
            },
            ParamMeta {
                name: "pore_frequency",
                label: "Skeletal pores",
                hint: "Radiolaria lattice pore frequency (0 = smooth shell)",
                range: Some((0.0, 32.0)),
                slider: lin(0.0, 20.0),
                choices: &[],
                order: 29,
            },
            ParamMeta {
                name: "bound_radius",
                label: "Bound radius",
                hint: "Outer bounding sphere radius",
                range: Some((0.2, 20.0)),
                slider: lin(0.8, 4.0),
                choices: &[],
                order: 30,
            },
        ],
    },
    FormulaMeta {
        code: FAMILY_PHYLLOTAXIS,
        wgsl: "phyllotaxis_distance",
        variant: "PhyllotaxisShell",
        display_name: "Fibonacci Phyllotaxis (Romanesco)",
        hint: "Closed-form O(1) inverse spherical Fibonacci spiral lattice with recursive Romanesco conical florets",
        has_orbit_constant: false,
        signed: true,
        iterates: false,
        params: &[
            ParamMeta {
                name: "count",
                label: "Floret count",
                hint: "Number of primary golden-angle spiral florets on the sphere",
                range: Some((12.0, 600.0)),
                slider: lin(34.0, 300.0),
                choices: &[],
                order: 20,
            },
            ParamMeta {
                name: "cone_height",
                label: "Cone height",
                hint: "Radial height of each Romanesco bud",
                range: Some((0.0, 0.8)),
                slider: lin(0.05, 0.45),
                choices: &[],
                order: 21,
            },
            ParamMeta {
                name: "cone_sharpness",
                label: "Cone sharpness",
                hint: "Profile exponent of the conical buds (1 = cone, 0.5 = dome, 2 = pagoda)",
                range: Some((0.3, 4.0)),
                slider: lin(0.5, 2.5),
                choices: &[],
                order: 22,
            },
            ParamMeta {
                name: "spiral_twist",
                label: "Spiral twist",
                hint: "Helical twist inside each bud",
                range: Some((-4.0, 4.0)),
                slider: lin(-2.0, 2.0),
                choices: &[],
                order: 23,
            },
            ParamMeta {
                name: "sub_florets",
                label: "Sub-floret spirals",
                hint: "Secondary Fibonacci spiral ribs on each bud",
                range: Some((0.0, 21.0)),
                slider: lin(0.0, 13.0),
                choices: &[],
                order: 24,
            },
            ParamMeta {
                name: "base_radius",
                label: "Core radius",
                hint: "Inner sphere radius supporting the florets",
                range: Some((0.2, 2.0)),
                slider: lin(0.5, 1.3),
                choices: &[],
                order: 25,
            },
            ParamMeta {
                name: "bound_radius",
                label: "Bound radius",
                hint: "Outer bounding sphere radius",
                range: Some((0.2, 20.0)),
                slider: lin(0.8, 4.0),
                choices: &[],
                order: 26,
            },
        ],
    },
    FormulaMeta {
        code: FAMILY_NAUTILUS,
        wgsl: "nautilus_distance",
        variant: "NautilusShell",
        display_name: "Nautilus / Ammonite Shell",
        hint: "Raup logarithmic helicospiral shell with internal chamber septa and ammonite growth ribs",
        has_orbit_constant: false,
        signed: true,
        iterates: false,
        params: &[
            ParamMeta {
                name: "expansion_rate",
                label: "Whorl expansion",
                hint: "Logarithmic growth rate b in r = exp(b * theta)",
                range: Some((0.05, 0.45)),
                slider: lin(0.1, 0.3),
                choices: &[],
                order: 20,
            },
            ParamMeta {
                name: "coil_tightness",
                label: "Coil overlap",
                hint: "Tube radius relative to whorl radius",
                range: Some((0.4, 2.0)),
                slider: lin(0.7, 1.5),
                choices: &[],
                order: 21,
            },
            ParamMeta {
                name: "spire_translation",
                label: "Spire height",
                hint: "Conispiral translation along Z (0 = planispiral Nautilus, >0 = Turritella)",
                range: Some((0.0, 2.0)),
                slider: lin(0.0, 1.0),
                choices: &[],
                order: 22,
            },
            ParamMeta {
                name: "wall_thickness",
                label: "Wall thickness",
                hint: "Shell and internal septum wall thickness",
                range: Some((0.01, 0.2)),
                slider: lin(0.02, 0.1),
                choices: &[],
                order: 23,
            },
            ParamMeta {
                name: "rib_frequency",
                label: "Septa / Rib count",
                hint: "Chambers and external ribs per full turn",
                range: Some((0.0, 48.0)),
                slider: lin(4.0, 32.0),
                choices: &[],
                order: 24,
            },
            ParamMeta {
                name: "rib_amplitude",
                label: "Rib relief",
                hint: "External ammonite rib corrugation amplitude",
                range: Some((0.0, 0.15)),
                slider: lin(0.0, 0.06),
                choices: &[],
                order: 25,
            },
            ParamMeta {
                name: "turns",
                label: "Shell turns",
                hint: "Total number of spiral revolutions",
                range: Some((0.5, 5.0)),
                slider: lin(1.5, 4.0),
                choices: &[],
                order: 26,
            },
            ParamMeta {
                name: "bound_radius",
                label: "Bound radius",
                hint: "Outer bounding sphere radius",
                range: Some((0.2, 20.0)),
                slider: lin(0.8, 4.0),
                choices: &[],
                order: 27,
            },
        ],
    },
    FormulaMeta {
        code: FAMILY_VORONOI_FOAM,
        wgsl: "voronoi_foam_distance",
        variant: "VoronoiFoam",
        display_name: "Voronoi Trabecular Foam",
        hint: "Multi-scale 3D Voronoi F2-F1 cellular skeleton and radiolaria/bone trabecular struts",
        has_orbit_constant: false,
        signed: true,
        iterates: false,
        params: &[
            ParamMeta {
                name: "cell_density",
                label: "Cell density",
                hint: "Voronoi cells per unit length",
                range: Some((0.5, 10.0)),
                slider: lin(1.0, 5.0),
                choices: &[],
                order: 20,
            },
            ParamMeta {
                name: "strut_radius",
                label: "Strut thickness",
                hint: "Cell edge/wall thickness",
                range: Some((0.005, 0.4)),
                slider: lin(0.02, 0.18),
                choices: &[],
                order: 21,
            },
            ParamMeta {
                name: "jitter",
                label: "Cell jitter",
                hint: "Randomness of Voronoi seed positions (0 = regular BCC/cubic grid, 1 = organic foam)",
                range: Some((0.0, 1.0)),
                slider: lin(0.0, 1.0),
                choices: &[],
                order: 22,
            },
            ParamMeta {
                name: "octaves",
                label: "Foam octaves",
                hint: "Number of nested hierarchical Voronoi scales (1..3)",
                range: Some((1.0, 3.0)),
                slider: lin(1.0, 3.0),
                choices: &[],
                order: 23,
            },
            ParamMeta {
                name: "lacunarity",
                label: "Lacunarity",
                hint: "Cell density multiplier for sub-foam octaves",
                range: Some((1.5, 4.0)),
                slider: lin(1.8, 3.0),
                choices: &[],
                order: 24,
            },
            ParamMeta {
                name: "shell_hollow",
                label: "Inner hollow",
                hint: "Hollow core radius fraction (0 = full ball of foam)",
                range: Some((0.0, 0.9)),
                slider: lin(0.0, 0.75),
                choices: &[],
                order: 25,
            },
            ParamMeta {
                name: "bound_radius",
                label: "Bound radius",
                hint: "Outer bounding sphere radius",
                range: Some((0.2, 20.0)),
                slider: lin(0.6, 3.5),
                choices: &[],
                order: 26,
            },
        ],
    },
    FormulaMeta {
        code: FAMILY_CRYSTAL_GEODE,
        wgsl: "crystal_geode_distance",
        variant: "CrystalGeode",
        display_name: "Quartz Crystal Cluster",
        hint: "Procedural mineral crystal cluster with hexagonal prisms and 6-sided rhombohedral terminations",
        has_orbit_constant: false,
        signed: true,
        iterates: false,
        params: &[
            ParamMeta {
                name: "crystal_count",
                label: "Crystal count",
                hint: "Number of radiating quartz crystals in the cluster (1..24)",
                range: Some((1.0, 24.0)),
                slider: lin(3.0, 20.0),
                choices: &[],
                order: 20,
            },
            ParamMeta {
                name: "prism_length",
                label: "Prism length",
                hint: "Length of the primary hexagonal crystal shafts",
                range: Some((0.2, 2.5)),
                slider: lin(0.4, 1.6),
                choices: &[],
                order: 21,
            },
            ParamMeta {
                name: "prism_radius",
                label: "Prism radius",
                hint: "Cross-sectional hexagonal radius of the crystals",
                range: Some((0.04, 0.8)),
                slider: lin(0.08, 0.4),
                choices: &[],
                order: 22,
            },
            ParamMeta {
                name: "tip_slope",
                label: "Pyramid slope",
                hint: "Termination pyramid facet slope (tan of termination angle)",
                range: Some((0.3, 3.0)),
                slider: lin(0.6, 2.0),
                choices: &[],
                order: 23,
            },
            ParamMeta {
                name: "cluster_spread",
                label: "Splay angle",
                hint: "Angular spread of the radiating crystal shafts",
                range: Some((0.0, 1.5)),
                slider: lin(0.1, 1.1),
                choices: &[],
                order: 24,
            },
            ParamMeta {
                name: "geode_cavity",
                label: "Hollow cavity",
                hint: "Carve an inner geode druzy cavity (>0)",
                range: Some((0.0, 1.0)),
                slider: lin(0.0, 0.7),
                choices: &[],
                order: 25,
            },
            ParamMeta {
                name: "bound_radius",
                label: "Bound radius",
                hint: "Outer bounding sphere radius",
                range: Some((0.2, 20.0)),
                slider: lin(0.8, 3.5),
                choices: &[],
                order: 26,
            },
        ],
    },
    FormulaMeta {
        code: FAMILY_ASTROLABE,
        wgsl: "astrolabe_distance",
        variant: "AstrolabeGimbal",
        display_name: "Mechanical Astrolabe",
        hint: "Concentric tilted gyroscopic gimbal rings with precision gear teeth and axial pivots",
        has_orbit_constant: false,
        signed: true,
        iterates: false,
        params: &[
            ParamMeta {
                name: "ring_count",
                label: "Gimbal rings",
                hint: "Number of nested concentric gimbal rings (1..8)",
                range: Some((1.0, 8.0)),
                slider: lin(2.0, 7.0),
                choices: &[],
                order: 20,
            },
            ParamMeta {
                name: "outer_radius",
                label: "Outer radius",
                hint: "Radius of the outermost gimbal ring",
                range: Some((0.3, 3.0)),
                slider: lin(0.6, 1.8),
                choices: &[],
                order: 21,
            },
            ParamMeta {
                name: "ring_spacing",
                label: "Ring spacing",
                hint: "Radial step between consecutive inner rings",
                range: Some((0.05, 0.5)),
                slider: lin(0.08, 0.28),
                choices: &[],
                order: 22,
            },
            ParamMeta {
                name: "ring_width",
                label: "Band height",
                hint: "Axial height of each gimbal ring band",
                range: Some((0.02, 0.5)),
                slider: lin(0.04, 0.25),
                choices: &[],
                order: 23,
            },
            ParamMeta {
                name: "ring_thickness",
                label: "Band thickness",
                hint: "Radial wall thickness of each gimbal ring",
                range: Some((0.01, 0.25)),
                slider: lin(0.02, 0.12),
                choices: &[],
                order: 24,
            },
            ParamMeta {
                name: "gear_teeth",
                label: "Gear teeth",
                hint: "Number of gear teeth around each ring (0 = smooth rings)",
                range: Some((0.0, 64.0)),
                slider: lin(0.0, 36.0),
                choices: &[],
                order: 25,
            },
            ParamMeta {
                name: "tooth_depth",
                label: "Tooth depth",
                hint: "Radial height of the gear teeth",
                range: Some((0.0, 0.1)),
                slider: lin(0.0, 0.05),
                choices: &[],
                order: 26,
            },
            ParamMeta {
                name: "tilt_step_degrees",
                label: "Tilt step (°)",
                hint: "Incremental inclination angle per ring in degrees",
                range: Some((-180.0, 180.0)),
                slider: lin(-60.0, 60.0),
                choices: &[],
                order: 27,
            },
            ParamMeta {
                name: "spin_degrees",
                label: "Precession (°)",
                hint: "Azimuthal spin offset per ring in degrees",
                range: Some((-360.0, 360.0)),
                slider: lin(-180.0, 180.0),
                choices: &[],
                order: 28,
            },
            ParamMeta {
                name: "bound_radius",
                label: "Bound radius",
                hint: "Outer bounding sphere radius",
                range: Some((0.2, 20.0)),
                slider: lin(0.8, 3.5),
                choices: &[],
                order: 29,
            },
        ],
    },
];

// =============================================================================
// Metadata Lookup Helpers (Used by `world.rs` and `world_ui.rs`)
// =============================================================================

/// Split `/formula/<Variant>/<field>[/<comp>]` into `(Variant, field)`.
pub fn parse_formula_attr_path(path: &str) -> Option<(&str, &str)> {
    let rest = path.strip_prefix("/formula/")?;
    let (variant, tail) = rest.split_once('/')?;
    let field = match tail.split_once('/') {
        Some((f, comp)) if comp.parse::<usize>().is_ok() => f,
        Some(_) => return None,
        None => tail,
    };
    Some((variant, field))
}

pub fn find_formula_meta_by_variant(variant: &str) -> Option<&'static FormulaMeta> {
    FORMULAS.iter().find(|m| m.variant == variant)
}

pub fn find_param_meta(path: &str) -> Option<&'static ParamMeta> {
    let (variant, field) = parse_formula_attr_path(path)?;
    let fm = find_formula_meta_by_variant(variant)?;
    fm.params.iter().find(|p| p.name == field)
}

pub fn formula_param_label(path: &str) -> Option<String> {
    find_param_meta(path).map(|p| p.label.to_owned())
}

pub fn formula_param_range(path: &str) -> Option<(f64, f64)> {
    find_param_meta(path).and_then(|p| p.range)
}

pub fn formula_param_slider(path: &str) -> Option<Slider> {
    find_param_meta(path).and_then(|p| p.slider)
}

pub fn formula_param_choices(path: &str) -> Option<Vec<Value>> {
    let pm = find_param_meta(path)?;
    if pm.choices.is_empty() {
        None
    } else {
        Some(pm.choices.iter().map(|&c| json!(c)).collect())
    }
}

pub fn formula_param_order(path: &str) -> Option<u16> {
    find_param_meta(path).map(|p| p.order)
}

/// Why a formula sub-parameter does nothing in the node's current state (shown greyed in the
/// Attribute Editor), or `None` when active.
pub fn formula_param_inactive_reason<'a>(
    path: &str,
    value: impl Fn(&str) -> Option<&'a Value>,
) -> Option<String> {
    let (variant, field) = parse_formula_attr_path(path)?;
    match (variant, field) {
        ("GyroidTpms", "lacunarity" | "gain") => {
            (value("/formula/GyroidTpms/octaves")?.as_f64()? <= 1.0)
                .then(|| "Fractal octaves is 1".into())
        }
        ("VoronoiFoam", "lacunarity") => (value("/formula/VoronoiFoam/octaves")?.as_f64()? <= 1.0)
            .then(|| "Foam octaves is 1".into()),
        ("TorusKnotBraid", "sub_strands") => {
            (value("/formula/TorusKnotBraid/braid_depth")?.as_f64()? == 0.0)
                .then(|| "Braid relief is 0".into())
        }
        ("DupinCyclide", "rib_depth") => {
            (value("/formula/DupinCyclide/rib_frequency")?.as_f64()? == 0.0)
                .then(|| "Rib count is 0".into())
        }
        ("DupinCyclide", "rib_frequency") => {
            (value("/formula/DupinCyclide/rib_depth")?.as_f64()? == 0.0)
                .then(|| "Rib depth is 0".into())
        }
        ("PhyllotaxisShell", "cone_sharpness" | "sub_florets") => {
            (value("/formula/PhyllotaxisShell/cone_height")?.as_f64()? == 0.0)
                .then(|| "Cone height is 0".into())
        }
        ("PhyllotaxisShell", "spiral_twist") => {
            if value("/formula/PhyllotaxisShell/cone_height")?.as_f64()? == 0.0 {
                Some("Cone height is 0".into())
            } else if value("/formula/PhyllotaxisShell/sub_florets")?.as_f64()? == 0.0 {
                Some("Sub-floret spirals is 0".into())
            } else {
                None
            }
        }
        ("NautilusShell", "rib_amplitude") => {
            (value("/formula/NautilusShell/rib_frequency")?.as_f64()? == 0.0)
                .then(|| "Septa / Rib count is 0".into())
        }
        ("AstrolabeGimbal", "tooth_depth") => {
            (value("/formula/AstrolabeGimbal/gear_teeth")?.as_f64()? < 1.0)
                .then(|| "Gear teeth is 0".into())
        }
        ("AstrolabeGimbal", "gear_teeth") => {
            (value("/formula/AstrolabeGimbal/tooth_depth")?.as_f64()? == 0.0)
                .then(|| "Tooth depth is 0".into())
        }
        _ => None,
    }
}

// =============================================================================
// Packing Helpers: the extended families reuse the classic families' per-object formula slots
// `P_BULB_POWER..=P_HYBRID_COUNT` (69..=125 at the pinned fractal-core), offsets 0..EXT_SLOTS.
// =============================================================================

/// Formula slots an extended family packs from `P_BULB_POWER` ([`pack_extended_formula`]): the
/// largest is Kaleido Twist's post-rotation matrix, offsets 6..=14.
pub const EXT_SLOTS: usize = 15;

// The extended block must stay inside the per-object formula slots, before the colour / trap
// slots that every object packs on its own.
const _: () = assert!(P_BULB_POWER + EXT_SLOTS <= P_COLOR_MODE);

#[inline]
fn put_vec3(p: &mut [f32], slot: usize, v: [f32; 3]) {
    p[slot..slot + 3].copy_from_slice(&v);
}

#[inline]
fn put_rot(p: &mut [f32], degrees: [f32; 3], euler_matrix: fn([f32; 3]) -> [[f32; 3]; 3]) {
    let reduced = degrees.map(|a| a % 360.0);
    if reduced.iter().all(|&a| a == 0.0) {
        p[P_ITER_ROTATE] = 0.0;
    } else {
        p[P_ITER_ROTATE] = 1.0;
        let m = euler_matrix(reduced);
        for r in 0..3 {
            p[P_ITER_ROT + 3 * r..P_ITER_ROT + 3 * r + 3].copy_from_slice(&m[r]);
        }
    }
}

pub fn pack_extended_formula(
    p: &mut [f32],
    formula: &crate::scene::Formula,
    bulb_escape: f32,
    euler_matrix: fn([f32; 3]) -> [[f32; 3]; 3],
) {
    use crate::scene::Formula;
    // Offsets b..b + EXT_SLOTS (see `EXT_SLOTS`).
    let b = P_BULB_POWER;
    match formula {
        Formula::AmazingSurface(f) => {
            let min_r2 = (f.min_radius * f.min_radius).max(1.0e-6);
            let fixed_r2 = (f.fixed_radius * f.fixed_radius).max(min_r2);
            p[b] = f.scale;
            p[b + 1] = min_r2;
            p[b + 2] = fixed_r2;
            put_vec3(p, b + 3, f.fold_limit);
            put_vec3(p, b + 6, f.shift);
            put_rot(p, f.rotation_degrees, euler_matrix);
        }
        Formula::Bristorbrot(f) => {
            p[b] = f.scale;
            p[b + 1] = f.y_sign;
            p[b + 2] = f.z_sign;
            p[b + 3] = f.twist;
            p[P_BAILOUT] = f.bailout.max(bulb_escape);
            put_rot(p, f.rotation_degrees, euler_matrix);
        }
        Formula::BurningShip(f) => {
            p[b] = f.power;
            put_vec3(p, b + 1, f.abs_axes);
            put_vec3(p, b + 4, f.shift);
            p[P_BAILOUT] = f.bailout.max(bulb_escape);
            put_rot(p, f.rotation_degrees, euler_matrix);
        }
        Formula::KaleidoTwist(f) => {
            p[b] = f.scale;
            put_vec3(p, b + 1, f.shift);
            p[b + 4] = f.clamp_z;
            let post_red = f.post_rotation_degrees.map(|a| a % 360.0);
            if post_red.iter().all(|&a| a == 0.0) {
                p[b + 5] = 0.0;
            } else {
                p[b + 5] = 1.0;
                let m = euler_matrix(post_red);
                for r in 0..3 {
                    p[b + 6 + 3 * r..b + 6 + 3 * r + 3].copy_from_slice(&m[r]);
                }
            }
            put_rot(p, f.rotation_degrees, euler_matrix);
        }
        Formula::PolyhedralKifs(f) => {
            p[b] = f.symmetry.code() as f32;
            p[b + 1] = f.scale;
            put_vec3(p, b + 2, f.offset);
            p[b + 5] = f.shell_thickness;
            put_rot(p, f.rotation_degrees, euler_matrix);
        }
        Formula::HarmonicBulb(f) => {
            p[b] = f.power;
            p[b + 1] = f.harmonic_l;
            p[b + 2] = f.harmonic_m;
            p[b + 3] = f.amplitude;
            p[P_BAILOUT] = f.bailout.max(bulb_escape);
            put_rot(p, f.rotation_degrees, euler_matrix);
        }
        Formula::Mandelbar3D(f) => {
            p[b] = f.power;
            p[b + 1] = f.conjugate_xy;
            p[b + 2] = f.conjugate_z;
            p[b + 3] = f.twist;
            p[P_BAILOUT] = f.bailout.max(bulb_escape);
            put_rot(p, f.rotation_degrees, euler_matrix);
        }
        Formula::MobiusIfs(f) => {
            let min_r2 = (f.min_radius * f.min_radius).max(1.0e-6);
            let fixed_r2 = (f.fixed_radius * f.fixed_radius).max(min_r2);
            p[b] = f.scale;
            p[b + 1] = min_r2;
            p[b + 2] = fixed_r2;
            put_vec3(p, b + 3, f.lattice_pitch);
            p[b + 6] = f.twist;
            put_rot(p, f.rotation_degrees, euler_matrix);
        }
        Formula::GyroidTpms(f) => {
            p[b] = f.surface.code() as f32;
            p[b + 1] = f.frequency;
            p[b + 2] = f.thickness;
            p[b + 3] = f.bias;
            p[b + 4] = f.octaves.clamp(1, 6) as f32;
            p[b + 5] = f.lacunarity;
            p[b + 6] = f.gain;
            p[b + 7] = f.twist;
        }
        Formula::LidinoidTpms(f) => {
            p[b] = f.surface.code() as f32;
            p[b + 1] = f.frequency;
            p[b + 2] = f.thickness;
            p[b + 3] = f.bias;
            p[b + 4] = f.harmonic_mix;
            p[b + 5] = f.radial_warp;
        }
        Formula::AttractorTube(f) => {
            p[b] = f.system.code() as f32;
            p[b + 1] = f.step_dt;
            p[b + 2] = f.tube_radius;
            p[b + 3] = f.swirl;
            p[b + 4] = f.param_a;
            p[b + 5] = f.param_b;
        }
        Formula::TorusKnotBraid(f) => {
            p[b] = f.winds_p.round().max(1.0);
            p[b + 1] = f.winds_q.round().max(1.0);
            p[b + 2] = f.major_radius;
            p[b + 3] = f.minor_radius;
            p[b + 4] = f.tube_radius;
            p[b + 5] = f.sub_strands.clamp(1, 12) as f32;
            p[b + 6] = f.braid_depth;
        }
        Formula::HopfFibration(f) => {
            p[b] = f.fibers.clamp(2, 24) as f32;
            p[b + 1] = f.shells.clamp(1, 6) as f32;
            p[b + 2] = f.latitude;
            p[b + 3] = f.tube_radius;
            p[b + 4] = f.twist;
            p[b + 5] = f.scale;
        }
        Formula::DupinCyclide(f) => {
            p[b] = f.major_a;
            p[b + 1] = f.minor_b;
            p[b + 2] = f.eccentricity_c;
            p[b + 3] = f.radius_d;
            p[b + 4] = f.shell_thickness;
            p[b + 5] = f.rib_frequency;
            p[b + 6] = f.rib_depth;
        }
        Formula::Superformula3D(f) => {
            p[b] = f.m1;
            p[b + 1] = f.n1_1;
            p[b + 2] = f.n2_1;
            p[b + 3] = f.n3_1;
            p[b + 4] = f.m2;
            p[b + 5] = f.n1_2;
            p[b + 6] = f.n2_2;
            p[b + 7] = f.n3_2;
            p[b + 8] = f.shell_thickness;
            p[b + 9] = f.pore_frequency;
        }
        Formula::PhyllotaxisShell(f) => {
            p[b] = f.count.max(8.0);
            p[b + 1] = f.cone_height;
            p[b + 2] = f.cone_sharpness;
            p[b + 3] = f.spiral_twist;
            p[b + 4] = f.sub_florets;
            p[b + 5] = f.base_radius;
        }
        Formula::NautilusShell(f) => {
            p[b] = f.expansion_rate;
            p[b + 1] = f.coil_tightness;
            p[b + 2] = f.spire_translation;
            p[b + 3] = f.wall_thickness;
            p[b + 4] = f.rib_frequency;
            p[b + 5] = f.rib_amplitude;
            p[b + 6] = f.turns;
        }
        Formula::VoronoiFoam(f) => {
            p[b] = f.cell_density;
            p[b + 1] = f.strut_radius;
            p[b + 2] = f.jitter;
            p[b + 3] = f.octaves.clamp(1, 3) as f32;
            p[b + 4] = f.lacunarity;
            p[b + 5] = f.shell_hollow;
        }
        Formula::CrystalGeode(f) => {
            p[b] = f.crystal_count.clamp(1, 24) as f32;
            p[b + 1] = f.prism_length;
            p[b + 2] = f.prism_radius;
            p[b + 3] = f.tip_slope;
            p[b + 4] = f.cluster_spread;
            p[b + 5] = f.geode_cavity;
        }
        Formula::AstrolabeGimbal(f) => {
            p[b] = f.ring_count.clamp(1, 8) as f32;
            p[b + 1] = f.outer_radius;
            p[b + 2] = f.ring_spacing;
            p[b + 3] = f.ring_width;
            p[b + 4] = f.ring_thickness;
            p[b + 5] = f.gear_teeth;
            p[b + 6] = f.tooth_depth;
            p[b + 7] = f.tilt_step_degrees.to_radians();
            p[b + 8] = f.spin_degrees.to_radians();
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::Formula;

    #[test]
    fn every_family_has_one_registry_row_at_its_code() {
        assert_eq!(FORMULAS.len(), Formula::COUNT as usize);
        for (code, meta) in FORMULAS.iter().enumerate() {
            assert_eq!(meta.code as usize, code);
            let preset = Formula::PRESETS[code];
            assert_eq!(preset.code() as usize, code);
            assert_eq!(preset.name(), meta.display_name);
            assert_eq!(preset.has_orbit_constant(), meta.has_orbit_constant);
            assert!(!meta.display_name.is_empty() && !meta.hint.is_empty());
            // The serde variant key is the registry's.
            let json = serde_json::to_value(preset).unwrap();
            assert!(
                json.get(meta.variant).is_some(),
                "{} vs {json}",
                meta.variant
            );
            assert_eq!(
                find_formula_meta_by_variant(meta.variant).map(|m| m.code),
                Some(meta.code)
            );
        }
        let mut names: Vec<_> = FORMULAS.iter().map(|m| m.display_name).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), FORMULAS.len(), "display names are unique");
    }

    #[test]
    fn the_family_masks_follow_the_registry_and_fractal_kernels() {
        // fractal-kernels `family_signed`: of the classic eight only KIFS, Kleinian and
        // Pseudo-Kleinian are signed.
        let classic_signed = (0..8).filter(|&c| SIGNED_MASK >> c & 1 == 1);
        assert_eq!(
            classic_signed.collect::<Vec<u32>>(),
            vec![FAMILY_KIFS, FAMILY_KLEINIAN, FAMILY_PSEUDO_KLEINIAN]
        );
        for meta in &FORMULAS {
            assert_eq!(
                SIGNED_MASK >> meta.code & 1 == 1,
                meta.signed,
                "{}",
                meta.variant
            );
            assert_eq!(
                ITERATING_MASK >> meta.code & 1 == 1,
                meta.iterates,
                "{}",
                meta.variant
            );
        }
        assert_eq!(SIGNED_MASK >> Formula::COUNT, 0);
        assert_eq!(ITERATING_MASK >> Formula::COUNT, 0);
        // Every classic family iterates; the analytical shapes do not.
        assert!((0..8).all(|c| ITERATING_MASK >> c & 1 == 1));
        assert!(!FORMULAS[FAMILY_GYROID_TPMS as usize].iterates);
        assert!(FORMULAS[FAMILY_ATTRACTOR_TUBE as usize].iterates);
    }

    #[test]
    fn classic_sliders_carry_their_log_flags() {
        let slider = |path: &str| formula_param_slider(path).unwrap();
        let bailout = slider("/formula/Mandelbulb/bailout");
        assert!(bailout.log && bailout.min == 1.0 && bailout.max == 64.0);
        assert!(!slider("/formula/Mandelbulb/power").log);
        assert!(slider("/formula/Mandelbox/fixed_radius").log);
        assert!(!slider("/formula/Mandelbox/scale").log);
        assert!(slider("/formula/PseudoKleinian/thickness").log);
        assert!(slider("/formula/Hybrid/bailout").log);
        assert!(!slider("/formula/AmazingSurface/scale").log);
        // Every slider span is ordered.
        for meta in &FORMULAS {
            for p in meta.params {
                if let Some(s) = p.slider {
                    assert!(s.min < s.max, "{}/{}", meta.variant, p.name);
                    assert!(!s.log || s.min > 0.0, "{}/{}", meta.variant, p.name);
                }
            }
        }
    }

    #[test]
    fn extended_packing_stays_in_its_slot_block() {
        fn turn(degrees: [f32; 3]) -> [[f32; 3]; 3] {
            [[degrees[0], 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        }
        let block = P_BULB_POWER..P_BULB_POWER + EXT_SLOTS;
        // Besides the block, a family may write only its bailout and per-iteration rotation.
        let shared = |i: usize| {
            i == P_BAILOUT || i == P_ITER_ROTATE || (P_ITER_ROT..P_ITER_ROT + 9).contains(&i)
        };
        for code in 8..Formula::COUNT {
            let mut formula = Formula::PRESETS[code as usize];
            // Nonzero rotations so every rotation slot is written.
            match &mut formula {
                Formula::KaleidoTwist(f) => {
                    f.rotation_degrees = [1.0; 3];
                    f.post_rotation_degrees = [1.0; 3];
                }
                Formula::AmazingSurface(f) => f.rotation_degrees = [1.0; 3],
                _ => {}
            }
            let mut p = vec![f32::NAN; P_COUNT];
            pack_extended_formula(&mut p, &formula, 1.0, turn);
            for (i, v) in p.iter().enumerate() {
                if !v.is_nan() {
                    assert!(
                        block.contains(&i) || shared(i),
                        "family {code} wrote slot {i} outside its block"
                    );
                }
            }
        }
        // Kaleido Twist's post-rotation reaches the last slot of the block.
        let mut p = vec![f32::NAN; P_COUNT];
        let kaleido = Formula::KaleidoTwist(KaleidoTwist {
            post_rotation_degrees: [1.0; 3],
            ..KaleidoTwist::PRESET
        });
        pack_extended_formula(&mut p, &kaleido, 1.0, turn);
        assert!(!p[P_BULB_POWER + EXT_SLOTS - 1].is_nan());
    }
}
