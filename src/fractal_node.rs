//! First-class `"Fractal"` scene node specification (`FractalNodeSpec`), 28-family
//! `Formula` definitions, the parsed fractal model (`FractalParams`, built once per scene by
//! `FractalNodeSpec::prepare` and read through `FractalSceneExt`), and kernel parameter
//! packing (`FractalParams::pack`).
//!
//! `sl-studio` has zero built-in knowledge of fractals or formulas. By registering
//! [`FractalNodeSpec`] via [`sl_studio::node_spec::register_node_type`], WarpBro turns
//! `"Fractal"` into a first-class renderable scene node in `sl-studio`. When a
//! `"Fractal"` node is present and visible in the graph, it is evaluated into `scene.objects`
//! and rendered; when absent or hidden, `sl-studio` renders the 3D studio environment.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, LazyLock};

use fractal_core::params::*;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use sl_studio::custom_branches::{BranchMap, CustomBranches, PreparedNode};
use sl_studio::node_spec::NodeTypeSpec;
use sl_studio::palette::PaletteScheme;
use sl_studio::render_profiles::Backend;
use sl_studio::scene::{
    Adaptive, Camera, Coloring, DEFAULT_GLASS_PROBES, DEFAULT_MAX_STEPS, Material, MaterialModel,
    ObjectTransform, Render, Scene, Sky, Sun, length,
};
use sl_studio::world::{AttrValue, Attrs, Slider, WorldAttribute};

use crate::formulas::*;

/// The `WorldKind` of a WarpBro `"Fractal"` node in `sl-studio`.
#[cfg(test)]
pub const FRACTAL_KIND: sl_studio::world::WorldKind =
    sl_studio::world::WorldKind::Custom("Fractal");

// =============================================================================
// Formulas
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Bulb {
    pub power: f32,
    pub bailout: f32,
    pub rotation_degrees: [f32; 3],
    pub angle_scale: [f32; 2],
    pub angle_phase_degrees: [f32; 2],
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MandelBox {
    pub scale: f32,
    pub min_radius_ratio: f32,
    pub fixed_radius: f32,
    pub fold_limit: f32,
    pub rotation_degrees: [f32; 3],
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Quat {
    pub constant: [f32; 4],
    pub slice_w: f32,
    pub rotation_degrees: [f32; 3],
    pub bailout: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum KifsKind {
    Tetrahedron,
    Octahedron,
    Menger,
}
impl KifsKind {
    pub const ALL: [Self; 3] = [Self::Tetrahedron, Self::Octahedron, Self::Menger];

    pub const fn centre(self) -> [f32; 3] {
        match self {
            Self::Tetrahedron | Self::Menger => [1.0, 1.0, 1.0],
            Self::Octahedron => [1.0, 0.0, 0.0],
        }
    }
    pub const fn preset_scale(self) -> f32 {
        match self {
            Self::Tetrahedron | Self::Octahedron => 2.0,
            Self::Menger => 3.0,
        }
    }
    pub const fn bounding_radius(self) -> f32 {
        match self {
            Self::Tetrahedron | Self::Menger => 1.732_050_8,
            Self::Octahedron => 1.0,
        }
    }
    pub const fn code(self) -> u32 {
        self as u32
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Kifs {
    pub kind: KifsKind,
    pub scale: f32,
    pub offset: [f32; 3],
    pub rotation_degrees: [f32; 3],
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Kleinian {
    pub a: f32,
    pub b: f32,
    pub bound_radius: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PseudoKleinian {
    pub box_size: [f32; 3],
    pub size: f32,
    pub c: [f32; 3],
    pub thickness: f32,
    pub offset: [f32; 3],
    pub bound_radius: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Apollonian {
    pub scale: f32,
    pub bound_radius: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum HybridStep {
    Off,
    Mandelbulb,
    Mandelbox,
    KifsFold,
    Inversion,
}
impl HybridStep {
    pub const ALL: [Self; 5] = [
        Self::Off,
        Self::Mandelbulb,
        Self::Mandelbox,
        Self::KifsFold,
        Self::Inversion,
    ];
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Hybrid {
    pub steps: [HybridStep; 4],
    pub bulb: Bulb,
    pub mandelbox: MandelBox,
    pub kifs: Kifs,
    pub apollonian_scale: f32,
    pub bailout: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Formula {
    Mandelbulb(Bulb),
    Mandelbox(MandelBox),
    QuaternionJulia(Quat),
    Kifs(Kifs),
    Kleinian(Kleinian),
    PseudoKleinian(PseudoKleinian),
    Apollonian(Apollonian),
    Hybrid(Hybrid),
    AmazingSurface(AmazingSurface),
    Bristorbrot(Bristorbrot),
    BurningShip(BurningShip),
    KaleidoTwist(KaleidoTwist),
    PolyhedralKifs(PolyhedralKifs),
    HarmonicBulb(HarmonicBulb),
    Mandelbar3D(Mandelbar3D),
    MobiusIfs(MobiusIfs),
    GyroidTpms(GyroidTpms),
    LidinoidTpms(LidinoidTpms),
    AttractorTube(AttractorTube),
    TorusKnotBraid(TorusKnotBraid),
    HopfFibration(HopfFibration),
    DupinCyclide(DupinCyclide),
    Superformula3D(Superformula3D),
    PhyllotaxisShell(PhyllotaxisShell),
    NautilusShell(NautilusShell),
    VoronoiFoam(VoronoiFoam),
    CrystalGeode(CrystalGeode),
    AstrolabeGimbal(AstrolabeGimbal),
}

impl Bulb {
    pub const PRESET: Self = Self {
        power: 8.0,
        bailout: 2.0,
        rotation_degrees: [0.0; 3],
        angle_scale: [1.0; 2],
        angle_phase_degrees: [0.0; 2],
    };
}
impl MandelBox {
    pub const PRESET: Self = Self {
        scale: 2.0,
        min_radius_ratio: 0.5,
        fixed_radius: 1.0,
        fold_limit: 1.0,
        rotation_degrees: [0.0; 3],
    };
}
impl Quat {
    pub const PRESET: Self = Self {
        constant: [-0.2, 0.6, 0.2, -0.4],
        slice_w: 0.0,
        rotation_degrees: [0.0; 3],
        bailout: 4.0,
    };
}
impl Kifs {
    pub const fn preset(kind: KifsKind) -> Self {
        Self {
            kind,
            scale: kind.preset_scale(),
            offset: [0.0; 3],
            rotation_degrees: [0.0; 3],
        }
    }
}
impl Kleinian {
    pub const PRESET: Self = Self {
        a: 1.846_275_6,
        b: 0.096_275_8,
        bound_radius: 2.0,
    };
}
impl PseudoKleinian {
    pub const PRESET: Self = Self {
        box_size: [0.974_78, 1.042_02, 0.974_78],
        size: 1.0,
        c: [-0.059_72, 0.149_2, -0.298_52],
        thickness: 0.01,
        offset: [0.436_36, 0.854_54, -0.145_46],
        bound_radius: 0.0,
    };
}
impl Apollonian {
    pub const PRESET: Self = Self {
        scale: 1.3,
        bound_radius: 1.5,
    };
}
impl Hybrid {
    pub const PRESET: Self = Self {
        steps: [
            HybridStep::Mandelbulb,
            HybridStep::Mandelbox,
            HybridStep::Off,
            HybridStep::Off,
        ],
        bulb: Bulb::PRESET,
        mandelbox: MandelBox::PRESET,
        kifs: Kifs::preset(KifsKind::Tetrahedron),
        apollonian_scale: 1.3,
        bailout: 8.0,
    };
}

const BOUND_FRAMING: f32 = 1.2;
const REACH_MARGIN_FRAMES: f32 = 2.5;
const INVERSION_FAR_REACH: f32 = 1.0e4;

const _: () = {
    let mut i = 0;
    while i < Formula::PRESETS.len() {
        assert!(Formula::PRESETS[i].code() as usize == i);
        i += 1;
    }
};

impl Formula {
    pub const COUNT: u32 = TOTAL_FORMULA_COUNT;

    pub const PRESETS: [Formula; TOTAL_FORMULA_COUNT as usize] = [
        Self::Mandelbulb(Bulb::PRESET),
        Self::Mandelbox(MandelBox::PRESET),
        Self::QuaternionJulia(Quat::PRESET),
        Self::Kifs(Kifs::preset(KifsKind::Tetrahedron)),
        Self::Kleinian(Kleinian::PRESET),
        Self::PseudoKleinian(PseudoKleinian::PRESET),
        Self::Apollonian(Apollonian::PRESET),
        Self::Hybrid(Hybrid::PRESET),
        Self::AmazingSurface(AmazingSurface::PRESET),
        Self::Bristorbrot(Bristorbrot::PRESET),
        Self::BurningShip(BurningShip::PRESET),
        Self::KaleidoTwist(KaleidoTwist::PRESET),
        Self::PolyhedralKifs(PolyhedralKifs::PRESET),
        Self::HarmonicBulb(HarmonicBulb::PRESET),
        Self::Mandelbar3D(Mandelbar3D::PRESET),
        Self::MobiusIfs(MobiusIfs::PRESET),
        Self::GyroidTpms(GyroidTpms::PRESET),
        Self::LidinoidTpms(LidinoidTpms::PRESET),
        Self::AttractorTube(AttractorTube::PRESET),
        Self::TorusKnotBraid(TorusKnotBraid::PRESET),
        Self::HopfFibration(HopfFibration::PRESET),
        Self::DupinCyclide(DupinCyclide::PRESET),
        Self::Superformula3D(Superformula3D::PRESET),
        Self::PhyllotaxisShell(PhyllotaxisShell::PRESET),
        Self::NautilusShell(NautilusShell::PRESET),
        Self::VoronoiFoam(VoronoiFoam::PRESET),
        Self::CrystalGeode(CrystalGeode::PRESET),
        Self::AstrolabeGimbal(AstrolabeGimbal::PRESET),
    ];

    pub const fn code(&self) -> u32 {
        match self {
            Self::Mandelbulb(_) => FAMILY_BULB,
            Self::Mandelbox(_) => FAMILY_BOX,
            Self::QuaternionJulia(_) => FAMILY_QUAT,
            Self::Kifs(_) => FAMILY_KIFS,
            Self::Kleinian(_) => FAMILY_KLEINIAN,
            Self::PseudoKleinian(_) => FAMILY_PSEUDO_KLEINIAN,
            Self::Apollonian(_) => FAMILY_APOLLONIAN,
            Self::Hybrid(_) => FAMILY_HYBRID,
            Self::AmazingSurface(_) => FAMILY_AMAZING_SURFACE,
            Self::Bristorbrot(_) => FAMILY_BRISTORBROT,
            Self::BurningShip(_) => FAMILY_BURNING_SHIP,
            Self::KaleidoTwist(_) => FAMILY_KALEIDO_TWIST,
            Self::PolyhedralKifs(_) => FAMILY_POLYHEDRAL_KIFS,
            Self::HarmonicBulb(_) => FAMILY_HARMONIC_BULB,
            Self::Mandelbar3D(_) => FAMILY_MANDELBAR,
            Self::MobiusIfs(_) => FAMILY_MOBIUS_IFS,
            Self::GyroidTpms(_) => FAMILY_GYROID_TPMS,
            Self::LidinoidTpms(_) => FAMILY_LIDINOID_TPMS,
            Self::AttractorTube(_) => FAMILY_ATTRACTOR_TUBE,
            Self::TorusKnotBraid(_) => FAMILY_TORUS_KNOT,
            Self::HopfFibration(_) => FAMILY_HOPF_FIBRATION,
            Self::DupinCyclide(_) => FAMILY_DUPIN_CYCLIDE,
            Self::Superformula3D(_) => FAMILY_SUPERFORMULA,
            Self::PhyllotaxisShell(_) => FAMILY_PHYLLOTAXIS,
            Self::NautilusShell(_) => FAMILY_NAUTILUS,
            Self::VoronoiFoam(_) => FAMILY_VORONOI_FOAM,
            Self::CrystalGeode(_) => FAMILY_CRYSTAL_GEODE,
            Self::AstrolabeGimbal(_) => FAMILY_ASTROLABE,
        }
    }

    pub fn meta(&self) -> &'static FormulaMeta {
        let table: &'static [FormulaMeta] = &FORMULAS;
        &table[self.code() as usize]
    }

    pub fn name(&self) -> &'static str {
        self.meta().display_name
    }

    pub fn hint(&self) -> &'static str {
        self.meta().hint
    }

    pub fn framing_radius(&self) -> f32 {
        match self {
            Self::Mandelbulb(_) => 1.0,
            Self::Mandelbox(_) => 4.0637,
            Self::QuaternionJulia(_) => std::f32::consts::GOLDEN_RATIO,
            Self::Kifs(k) => k.kind.bounding_radius(),
            Self::Kleinian(_) => BOUND_FRAMING * Kleinian::PRESET.bound_radius,
            Self::PseudoKleinian(_) => 1.0,
            Self::Apollonian(_) => BOUND_FRAMING * Apollonian::PRESET.bound_radius,
            Self::Hybrid(_) => 2.0,
            Self::AmazingSurface(f) => f.bound_radius.max(1.2),
            Self::Bristorbrot(_) => 1.25,
            Self::BurningShip(_) => 1.2,
            Self::KaleidoTwist(f) => f.bound_radius.max(1.2),
            Self::PolyhedralKifs(f) => f.bound_radius.max(1.0),
            Self::HarmonicBulb(_) => 1.15,
            Self::Mandelbar3D(_) => 1.15,
            Self::MobiusIfs(f) => f.bound_radius.max(1.0),
            Self::GyroidTpms(f) => f.bound_radius.max(1.0),
            Self::LidinoidTpms(f) => f.bound_radius.max(1.0),
            Self::AttractorTube(f) => f.bound_radius.max(1.0),
            Self::TorusKnotBraid(f) => f.bound_radius.max(1.0),
            Self::HopfFibration(f) => f.bound_radius.max(1.0),
            Self::DupinCyclide(f) => f.bound_radius.max(1.0),
            Self::Superformula3D(f) => f.bound_radius.max(1.0),
            Self::PhyllotaxisShell(f) => f.bound_radius.max(1.0),
            Self::NautilusShell(f) => f.bound_radius.max(1.0),
            Self::VoronoiFoam(f) => f.bound_radius.max(1.0),
            Self::CrystalGeode(f) => f.bound_radius.max(1.0),
            Self::AstrolabeGimbal(f) => f.bound_radius.max(1.0),
        }
    }

    pub fn bound_radius(&self) -> f32 {
        match self {
            Self::Kleinian(k) => k.bound_radius,
            Self::PseudoKleinian(k) => k.bound_radius,
            Self::Apollonian(k) => k.bound_radius,
            Self::AmazingSurface(f) => f.bound_radius,
            Self::KaleidoTwist(f) => f.bound_radius,
            Self::PolyhedralKifs(f) => f.bound_radius,
            Self::MobiusIfs(f) => f.bound_radius,
            Self::GyroidTpms(f) => f.bound_radius,
            Self::LidinoidTpms(f) => f.bound_radius,
            Self::AttractorTube(f) => f.bound_radius,
            Self::TorusKnotBraid(f) => f.bound_radius,
            Self::HopfFibration(f) => f.bound_radius,
            Self::DupinCyclide(f) => f.bound_radius,
            Self::Superformula3D(f) => f.bound_radius,
            Self::PhyllotaxisShell(f) => f.bound_radius,
            Self::NautilusShell(f) => f.bound_radius,
            Self::VoronoiFoam(f) => f.bound_radius,
            Self::CrystalGeode(f) => f.bound_radius,
            Self::AstrolabeGimbal(f) => f.bound_radius,
            _ => 0.0,
        }
    }

    pub fn has_orbit_constant(&self) -> bool {
        self.meta().has_orbit_constant
    }
}

// =============================================================================
// Parsed fractal (`FractalParams`) and the typed Scene extension (`FractalSceneExt`)
// =============================================================================

/// A fractal node's custom branches parsed into its typed model: what
/// [`FractalNodeSpec::prepare`] builds once whenever a `Scene` is built from the branches, and
/// what every render-side question (framing, bounds, family, packing) is answered from
/// (`sl_studio::custom_branches`). Before, each of those calls re-parsed the formula JSON, and
/// a malformed value could only be logged and replaced by a default; now a scene with one
/// cannot be built. Fields are the branches of the same names.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FractalParams {
    pub formula: Formula,
    pub julia: [f32; 3],
    pub julia_mix: f32,
    pub orbit_offset: [f32; 3],
    pub inversion_strength: f32,
    pub inversion_center: [f32; 3],
    pub inversion_radius: f32,
    pub trap_point: [f32; 3],
    pub trap_axis: u32,
    pub trap_scale: f32,
}

impl FractalParams {
    /// `formula` with every other branch at its default (no Julia mix, inversion or offset).
    pub fn new(formula: Formula) -> Self {
        Self {
            formula,
            julia: [0.0; 3],
            julia_mix: 0.0,
            orbit_offset: [0.0; 3],
            inversion_strength: 0.0,
            inversion_center: [0.0; 3],
            inversion_radius: 1.0,
            trap_point: [0.0; 3],
            trap_axis: 1,
            trap_scale: 1.5,
        }
    }

    /// Read every branch with the strict readers below, plus the two ranges packing relies on
    /// (`trap_axis` indexes an axis, `inversion_radius` divides).
    fn parse(branches: &BranchMap) -> Result<Self, String> {
        let get = |key: &str| branches.get(key);
        let params = Self {
            formula: custom_formula(get("formula"))?,
            julia: custom_vec3("julia", get("julia"))?,
            julia_mix: custom_f32("julia_mix", get("julia_mix"))?,
            orbit_offset: custom_vec3("orbit_offset", get("orbit_offset"))?,
            inversion_strength: custom_f32("inversion_strength", get("inversion_strength"))?,
            inversion_center: custom_vec3("inversion_center", get("inversion_center"))?,
            inversion_radius: custom_f32("inversion_radius", get("inversion_radius"))?,
            trap_point: custom_vec3("trap_point", get("trap_point"))?,
            trap_axis: custom_u32("trap_axis", get("trap_axis"))?,
            trap_scale: custom_f32("trap_scale", get("trap_scale"))?,
        };
        if params.inversion_radius <= 0.0 {
            return Err("/inversion_radius must be positive".into());
        }
        if params.trap_axis > 2 {
            return Err("/trap_axis must be 0, 1 or 2".into());
        }
        Ok(params)
    }

    /// The branches these parameters are stored as (`Scene::custom`, a node's template).
    pub fn branches(&self) -> BranchMap {
        BranchMap::from([
            // Plain data: serializing it cannot fail (a non-finite value becomes `null`, which
            // `parse` then rejects).
            (
                "formula".into(),
                serde_json::to_value(self.formula).expect("Formula serializes to JSON"),
            ),
            ("julia".into(), json!(self.julia)),
            ("julia_mix".into(), json!(self.julia_mix)),
            ("orbit_offset".into(), json!(self.orbit_offset)),
            ("inversion_strength".into(), json!(self.inversion_strength)),
            ("inversion_center".into(), json!(self.inversion_center)),
            ("inversion_radius".into(), json!(self.inversion_radius)),
            ("trap_point".into(), json!(self.trap_point)),
            ("trap_axis".into(), json!(self.trap_axis)),
            ("trap_scale".into(), json!(self.trap_scale)),
        ])
    }
}

pub fn default_fractal_custom_branches(formula: Formula) -> Map<String, Value> {
    FractalParams::new(formula).branches().into_iter().collect()
}

/// The one `FractalNodeSpec` instance branches are built for (`CustomBranches::for_spec`).
fn fractal_spec() -> Arc<dyn NodeTypeSpec> {
    static SPEC: LazyLock<Arc<dyn NodeTypeSpec>> = LazyLock::new(|| Arc::new(FractalNodeSpec));
    SPEC.clone()
}

/// `params` as a scene's custom branches, parsed back by the spec like any other build.
fn fractal_branches(params: &FractalParams) -> Result<CustomBranches, String> {
    CustomBranches::for_spec(fractal_spec(), params.branches())
}

fn base_fractal_scene(
    name: &str,
    formula: Formula,
    fov: f32,
    iterations: f32,
    hit_epsilon: f32,
    palette: PaletteScheme,
) -> Result<Scene, String> {
    Ok(Scene {
        world_render: false,
        camera_reference: None,
        objects: Vec::new(),
        lights: Vec::new(),
        object_world: None,
        masks: Vec::new(),
        bounds: Vec::new(),
        direct: None,
        node: None,
        mask: Default::default(),
        document: None,
        animation: Default::default(),
        environment: Default::default(),
        name: name.into(),
        colour: sl_studio::color::default_selection(),
        object: ObjectTransform {
            offset: [0.0; 3],
            rotation_degrees: [0.0; 3],
            scale: 1.0,
        },
        camera: Camera::default_fov(fov),
        sun: Sun::default(),
        sky: Sky::default(),
        material: Material::default(),
        palette,
        coloring: Coloring::Radius,
        custom: fractal_branches(&FractalParams::new(formula))?,
        render: Render {
            iterations,
            max_steps: DEFAULT_MAX_STEPS,
            glass_probes: DEFAULT_GLASS_PROBES,
            hit_epsilon,
            step_factor: 0.85,
            max_bounces: 6,
            exposure_stops: 0.0,
            saturation: 1.0,
            reinhard: false,
            denoise: Default::default(),
            adaptive: Adaptive::default(),
            backend: Backend::default(),
        },
    })
}

/// Typed access to a scene's fractal ([`FractalParams`], parsed once by
/// `FractalNodeSpec::prepare`). Building or editing one goes through the same parse as a
/// loaded document, so a preset or edit that breaks an invariant is an error, never a scene.
pub trait FractalSceneExt {
    fn fractal_preset(f: Formula) -> Result<Scene, String>;
    fn classic_fractal_gallery() -> Result<Vec<Scene>, String>;
    fn fractal_gallery() -> Result<Vec<Scene>, String>;

    /// This scene's fractal; `None` for a scene without one (a studio scene).
    fn fractal(&self) -> Option<&FractalParams>;

    /// Change this scene's fractal. The edited parameters are stored as branches and parsed
    /// again; on `Err` the scene is unchanged.
    fn edit_fractal(&mut self, edit: impl FnOnce(&mut FractalParams)) -> Result<(), String>;
}

/// Test shorthands over [`FractalSceneExt`]: a test may panic on a scene without a fractal.
#[cfg(test)]
pub trait FractalTestScene {
    fn preset_fractal(f: Formula) -> Scene;
    fn params(&self) -> FractalParams;
    fn edit(&mut self, edit: impl FnOnce(&mut FractalParams));

    fn formula(&self) -> Formula {
        self.params().formula
    }
    fn julia_mix(&self) -> f32 {
        self.params().julia_mix
    }
    fn orbit_offset(&self) -> [f32; 3] {
        self.params().orbit_offset
    }
    fn inversion_strength(&self) -> f32 {
        self.params().inversion_strength
    }
    fn inversion_center(&self) -> [f32; 3] {
        self.params().inversion_center
    }
    fn inversion_radius(&self) -> f32 {
        self.params().inversion_radius
    }
    fn trap_point(&self) -> [f32; 3] {
        self.params().trap_point
    }
    fn trap_axis(&self) -> u32 {
        self.params().trap_axis
    }
    fn trap_scale(&self) -> f32 {
        self.params().trap_scale
    }
    fn set_formula(&mut self, v: Formula) {
        self.edit(|f| f.formula = v);
    }
    fn modify_formula(&mut self, edit: impl FnOnce(&mut Formula)) {
        self.edit(|f| edit(&mut f.formula));
    }
    fn set_julia(&mut self, v: [f32; 3]) {
        self.edit(|f| f.julia = v);
    }
    fn set_julia_mix(&mut self, v: f32) {
        self.edit(|f| f.julia_mix = v);
    }
    fn set_orbit_offset(&mut self, v: [f32; 3]) {
        self.edit(|f| f.orbit_offset = v);
    }
    fn set_inversion_strength(&mut self, v: f32) {
        self.edit(|f| f.inversion_strength = v);
    }
    fn set_inversion_center(&mut self, v: [f32; 3]) {
        self.edit(|f| f.inversion_center = v);
    }
    fn set_inversion_radius(&mut self, v: f32) {
        self.edit(|f| f.inversion_radius = v);
    }
    fn set_trap_point(&mut self, v: [f32; 3]) {
        self.edit(|f| f.trap_point = v);
    }
    fn set_trap_axis(&mut self, v: u32) {
        self.edit(|f| f.trap_axis = v);
    }
    fn set_trap_scale(&mut self, v: f32) {
        self.edit(|f| f.trap_scale = v);
    }
}

#[cfg(test)]
impl FractalTestScene for Scene {
    fn preset_fractal(f: Formula) -> Scene {
        Scene::fractal_preset(f).expect("preset fractal")
    }
    fn params(&self) -> FractalParams {
        *self.fractal().expect("a fractal scene")
    }
    fn edit(&mut self, edit: impl FnOnce(&mut FractalParams)) {
        self.edit_fractal(edit).expect("valid fractal edit");
    }
}

// Strict readers of the fractal's custom branches: a missing key or a value of the wrong type
// is an error, never a default (no legacy-data compatibility). `FractalParams::parse` runs
// them whenever a scene is built from the branches (document load, edit, snapshot).

fn custom_f32(key: &str, value: Option<&Value>) -> Result<f32, String> {
    let v = value.ok_or_else(|| format!("/{key} is missing"))?;
    let f = v
        .as_f64()
        .ok_or_else(|| format!("/{key} must be a number, got {v}"))? as f32;
    if f.is_finite() {
        Ok(f)
    } else {
        Err(format!("/{key} must be a finite f32, got {v}"))
    }
}

fn custom_u32(key: &str, value: Option<&Value>) -> Result<u32, String> {
    let v = value.ok_or_else(|| format!("/{key} is missing"))?;
    v.as_u64()
        .and_then(|n| u32::try_from(n).ok())
        .ok_or_else(|| format!("/{key} must be a whole number, got {v}"))
}

fn custom_vec3(key: &str, value: Option<&Value>) -> Result<[f32; 3], String> {
    let v = value.ok_or_else(|| format!("/{key} is missing"))?;
    let a = v
        .as_array()
        .filter(|a| a.len() == 3)
        .ok_or_else(|| format!("/{key} must be 3 numbers, got {v}"))?;
    let mut out = [0.0; 3];
    for (slot, item) in out.iter_mut().zip(a) {
        *slot = custom_f32(key, Some(item))?;
    }
    Ok(out)
}

fn custom_formula(value: Option<&Value>) -> Result<Formula, String> {
    let v = value.ok_or("/formula is missing")?;
    Formula::deserialize(v).map_err(|e| format!("Invalid /formula: {e}"))
}

/// Formula leaves whose model type is an integer (`u32` counts such as
/// `/formula/GyroidTpms/octaves`). Read from the presets' JSON, where serde writes an integer
/// field as an integer number and an `f32` as a float, so the model types stay the one source.
/// Such a leaf must stay whole: it is validated as one and keyed discretely (interpolating
/// `3 -> 4` would produce `3.5`, which the `u32` field cannot deserialize).
fn integer_formula_leaf(path: &str) -> bool {
    fn collect(value: &Value, path: &str, out: &mut HashSet<String>) {
        match value {
            Value::Object(map) => {
                for (k, v) in map {
                    collect(v, &format!("{path}/{k}"), out);
                }
            }
            Value::Number(n) if n.is_u64() => {
                out.insert(path.to_owned());
            }
            _ => {}
        }
    }
    static LEAVES: LazyLock<HashSet<String>> = LazyLock::new(|| {
        let mut out = HashSet::new();
        for formula in Formula::PRESETS {
            // A preset is plain finite data: serializing it cannot fail.
            let value = serde_json::to_value(formula).expect("Formula serializes to JSON");
            collect(&value, "/formula", &mut out);
        }
        out
    });
    LEAVES.contains(path)
}

impl FractalSceneExt for Scene {
    fn fractal_preset(f: Formula) -> Result<Scene, String> {
        use PaletteScheme as P;
        let code = f.code();
        let name = f.name();
        let mut s = match code {
            FAMILY_BULB => base_fractal_scene(name, f, 38.0, 20.0, 0.001, P::Classic),
            FAMILY_BOX => base_fractal_scene(name, f, 35.0, 12.0, 0.003, P::Ice),
            FAMILY_QUAT => base_fractal_scene(name, f, 38.0, 16.0, 0.001, P::Amethyst),
            FAMILY_KIFS => base_fractal_scene(name, f, 38.0, 12.0, 0.0005, P::Copper),
            FAMILY_KLEINIAN => base_fractal_scene(name, f, 38.0, 16.0, 0.001, P::Ocean),
            FAMILY_PSEUDO_KLEINIAN => base_fractal_scene(name, f, 38.0, 6.0, 0.001, P::Ember),
            FAMILY_APOLLONIAN => base_fractal_scene(name, f, 38.0, 8.0, 0.001, P::Twilight),
            FAMILY_AMAZING_SURFACE => base_fractal_scene(name, f, 36.0, 14.0, 0.001, P::Copper),
            FAMILY_BRISTORBROT => base_fractal_scene(name, f, 38.0, 16.0, 0.001, P::Sunset),
            FAMILY_BURNING_SHIP => base_fractal_scene(name, f, 38.0, 14.0, 0.001, P::Fire),
            FAMILY_KALEIDO_TWIST => base_fractal_scene(name, f, 38.0, 12.0, 0.0008, P::Neon),
            FAMILY_POLYHEDRAL_KIFS => base_fractal_scene(name, f, 38.0, 12.0, 0.0008, P::RoseGold),
            FAMILY_HARMONIC_BULB => base_fractal_scene(name, f, 38.0, 14.0, 0.001, P::Aurora),
            FAMILY_MANDELBAR => base_fractal_scene(name, f, 38.0, 14.0, 0.001, P::Amethyst),
            FAMILY_MOBIUS_IFS => base_fractal_scene(name, f, 38.0, 12.0, 0.001, P::Ocean),
            FAMILY_GYROID_TPMS => base_fractal_scene(name, f, 38.0, 8.0, 0.001, P::Verdant),
            FAMILY_LIDINOID_TPMS => base_fractal_scene(name, f, 38.0, 8.0, 0.001, P::Ice),
            FAMILY_ATTRACTOR_TUBE => base_fractal_scene(name, f, 38.0, 10.0, 0.001, P::Ember),
            FAMILY_TORUS_KNOT => base_fractal_scene(name, f, 38.0, 8.0, 0.001, P::Copper),
            FAMILY_HOPF_FIBRATION => base_fractal_scene(name, f, 38.0, 8.0, 0.001, P::Twilight),
            FAMILY_DUPIN_CYCLIDE => base_fractal_scene(name, f, 38.0, 8.0, 0.001, P::RoseGold),
            FAMILY_SUPERFORMULA => base_fractal_scene(name, f, 38.0, 8.0, 0.001, P::Ocean),
            FAMILY_PHYLLOTAXIS => base_fractal_scene(name, f, 38.0, 8.0, 0.001, P::Verdant),
            FAMILY_NAUTILUS => base_fractal_scene(name, f, 38.0, 8.0, 0.001, P::Sunset),
            FAMILY_VORONOI_FOAM => base_fractal_scene(name, f, 38.0, 8.0, 0.001, P::Mono),
            FAMILY_CRYSTAL_GEODE => base_fractal_scene(name, f, 38.0, 8.0, 0.001, P::Amethyst),
            FAMILY_ASTROLABE => base_fractal_scene(name, f, 38.0, 8.0, 0.001, P::Copper),
            _ => base_fractal_scene(name, f, 38.0, 12.0, 0.001, P::Aurora),
        }?;
        if code >= FAMILY_AMAZING_SURFACE {
            s.coloring = Coloring::TrapOrigin;
        }
        match code {
            FAMILY_KALEIDO_TWIST => {
                s.material.metalness = 0.35;
                s.material.specular_roughness = 0.2;
            }
            FAMILY_POLYHEDRAL_KIFS => {
                s.material.metalness = 0.65;
                s.material.specular_roughness = 0.22;
            }
            FAMILY_TORUS_KNOT => {
                s.material.metalness = 0.85;
                s.material.specular_roughness = 0.2;
            }
            FAMILY_DUPIN_CYCLIDE => {
                s.material.metalness = 0.5;
                s.material.specular_roughness = 0.18;
            }
            FAMILY_CRYSTAL_GEODE => {
                s.material.model = MaterialModel::StandardSurface;
                s.material.coat = 1.0;
                s.material.specular_roughness = 0.12;
            }
            FAMILY_ASTROLABE => {
                s.material.metalness = 0.9;
                s.material.specular_roughness = 0.2;
            }
            _ => {}
        }
        s.camera.yaw_degrees = 35.0;
        s.camera.pitch_degrees = 20.0;
        s.camera.distance = 3.1;
        Ok(s)
    }

    fn classic_fractal_gallery() -> Result<Vec<Scene>, String> {
        use PaletteScheme as P;
        let mut v = Formula::PRESETS[..8]
            .iter()
            .map(|f| Self::fractal_preset(*f))
            .collect::<Result<Vec<_>, _>>()?;

        let mut s = Self::fractal_preset(Formula::Mandelbulb(Bulb::PRESET))?;
        s.name = "Bulb · power 3".into();
        s.edit_fractal(|f| {
            if let Formula::Mandelbulb(b) = &mut f.formula {
                b.power = 3.0;
            }
        })?;
        s.palette = P::Sunset;
        v.push(s);

        let mut s = Self::fractal_preset(Formula::Mandelbulb(Bulb::PRESET))?;
        s.name = "Bulb · power 12, gold".into();
        s.edit_fractal(|f| {
            if let Formula::Mandelbulb(b) = &mut f.formula {
                b.power = 12.0;
            }
        })?;
        s.palette = P::Copper;
        s.material.metalness = 1.0;
        s.material.specular_roughness = 0.25;
        v.push(s);

        let mut s = Self::fractal_preset(Formula::Mandelbulb(Bulb::PRESET))?;
        s.name = "Bulb · Julia".into();
        s.edit_fractal(|f| {
            f.julia = [0.35, 0.35, -0.4];
            f.julia_mix = 1.0;
        })?;
        s.camera.distance = 3.4;
        s.palette = P::Aurora;
        s.coloring = Coloring::TrapOrigin;
        v.push(s);

        let mut s = Self::fractal_preset(Formula::Mandelbulb(Bulb::PRESET))?;
        s.name = "Bulb · twisted".into();
        s.edit_fractal(|f| {
            if let Formula::Mandelbulb(b) = &mut f.formula {
                b.angle_phase_degrees = [40.0, 0.0];
                b.rotation_degrees = [0.0, 25.0, 0.0];
            }
        })?;
        s.palette = P::Neon;
        v.push(s);

        let mut s = Self::fractal_preset(Formula::Mandelbox(MandelBox::PRESET))?;
        s.name = "Mandelbox · scale −1.5".into();
        s.edit_fractal(|f| {
            if let Formula::Mandelbox(b) = &mut f.formula {
                b.scale = -1.5;
            }
        })?;
        s.camera.distance = 1.6;
        s.palette = P::RoseGold;
        v.push(s);

        let mut s = Self::fractal_preset(Formula::Kifs(Kifs::preset(KifsKind::Tetrahedron)))?;
        s.name = "Menger sponge".into();
        s.edit_fractal(|f| f.formula = Formula::Kifs(Kifs::preset(KifsKind::Menger)))?;
        s.palette = P::Mono;
        s.material.model = MaterialModel::StandardSurface;
        s.material.coat = 1.0;
        v.push(s);

        let mut s = Self::fractal_preset(Formula::Kifs(Kifs::preset(KifsKind::Tetrahedron)))?;
        s.name = "Octahedron KIFS".into();
        s.edit_fractal(|f| {
            f.formula = Formula::Kifs(Kifs {
                rotation_degrees: [12.0, 0.0, 8.0],
                ..Kifs::preset(KifsKind::Octahedron)
            });
        })?;
        s.palette = P::Verdant;
        v.push(s);

        let mut s = Self::fractal_preset(Formula::QuaternionJulia(Quat::PRESET))?;
        s.name = "Quaternion · glass-coat".into();
        s.edit_fractal(|f| {
            if let Formula::QuaternionJulia(q) = &mut f.formula {
                q.constant = [-0.291, -0.399, 0.339, 0.437];
            }
        })?;
        s.material.model = MaterialModel::StandardSurface;
        s.material.coat = 1.0;
        s.material.thin_film_thickness = 450.0;
        s.palette = P::Ice;
        v.push(s);

        let mut s = Self::fractal_preset(Formula::Hybrid(Hybrid::PRESET))?;
        s.name = "Hybrid · bulb + KIFS".into();
        s.edit_fractal(|f| {
            if let Formula::Hybrid(h) = &mut f.formula {
                h.steps = [
                    HybridStep::Mandelbulb,
                    HybridStep::KifsFold,
                    HybridStep::Off,
                    HybridStep::Off,
                ];
            }
        })?;
        s.palette = P::Fire;
        v.push(s);

        let mut s = Self::fractal_preset(Formula::PseudoKleinian(PseudoKleinian::PRESET))?;
        s.name = "Pseudo-Kleinian · cave".into();
        s.camera.distance = 0.6;
        s.camera.fov_y_degrees = 60.0;
        s.sky.intensity = 12.0;
        v.push(s);

        Ok(v)
    }

    fn fractal_gallery() -> Result<Vec<Scene>, String> {
        let mut v = Self::classic_fractal_gallery()?;
        for f in &Formula::PRESETS[8..] {
            v.push(Self::fractal_preset(*f)?);
        }
        Ok(v)
    }

    fn fractal(&self) -> Option<&FractalParams> {
        self.custom.node()?.downcast()
    }

    fn edit_fractal(&mut self, edit: impl FnOnce(&mut FractalParams)) -> Result<(), String> {
        let mut params = *self.fractal().ok_or("The scene has no fractal")?;
        edit(&mut params);
        self.custom = fractal_branches(&params)?;
        Ok(())
    }
}

// =============================================================================
// First-Class `"Fractal"` Node Specification (`FractalNodeSpec`)
// =============================================================================

pub struct FractalNodeSpec;

fn variants<T: Serialize>(all: impl IntoIterator<Item = T>) -> Result<Vec<Value>, String> {
    all.into_iter()
        .map(|v| serde_json::to_value(v).map_err(|e| e.to_string()))
        .collect()
}

/// A formula parameter's tooltip: the `FORMULAS` table is the one source for every family
/// (Hybrid's sub-sections are mapped to their own families by the caller).
fn formula_hint(family: &str, field: &str) -> Option<&'static str> {
    crate::formulas::find_formula_meta_by_variant(family)
        .and_then(|fm| fm.params.iter().find(|p| p.name == field))
        .map(|p| p.hint)
}

fn to_attr(v: &Value) -> AttrValue {
    match v {
        Value::Bool(b) => AttrValue::Bool(*b),
        Value::Number(n) if n.is_u64() || n.is_i64() => AttrValue::Int64(n.as_i64().unwrap_or(0)),
        Value::Number(n) => AttrValue::Float(n.as_f64().unwrap_or(0.0) as f32),
        Value::String(s) => AttrValue::Str(s.clone()),
        Value::Array(a) if a.iter().all(Value::is_number) => match a.len() {
            3 => AttrValue::Vec3(std::array::from_fn(|i| a[i].as_f64().unwrap_or(0.0) as f32)),
            4 => AttrValue::Vec4(std::array::from_fn(|i| a[i].as_f64().unwrap_or(0.0) as f32)),
            _ => AttrValue::List(a.iter().map(to_attr).collect()),
        },
        _ => AttrValue::Json(v.to_string()),
    }
}

fn attr_json(v: AttrValue) -> Value {
    match v {
        AttrValue::Bool(v) => json!(v),
        AttrValue::Str(v) => json!(v),
        AttrValue::Int(v) => json!(v),
        AttrValue::Int64(v) => json!(v),
        AttrValue::Int8(v) => json!(v),
        AttrValue::UInt(v) => json!(v),
        AttrValue::Float(v) => json!(v),
        AttrValue::Vec3(v) => json!(v),
        AttrValue::Vec4(v) => json!(v),
        AttrValue::List(v) => Value::Array(v.into_iter().map(attr_json).collect()),
        AttrValue::Json(v) => serde_json::from_str(&v).unwrap_or(Value::Null),
        AttrValue::Uuid(v) => json!(v),
        other => serde_json::to_value(other).unwrap_or(Value::Null),
    }
}

fn discover_formula(value: &Value, path: &str, attrs: &mut Attrs) {
    match value {
        Value::Object(map) => {
            if path == "/formula" {
                attrs.set(path, to_attr(value));
            }
            for (k, v) in map {
                discover_formula(
                    v,
                    &format!("{path}/{}", k.replace('~', "~0").replace('/', "~1")),
                    attrs,
                );
            }
        }
        _ => attrs.set(path, to_attr(value)),
    }
}

impl NodeTypeSpec for FractalNodeSpec {
    fn type_name(&self) -> &'static str {
        "Fractal"
    }

    fn label(&self) -> &'static str {
        "Fractal"
    }

    fn group(&self) -> &'static str {
        "Objects"
    }

    fn icon(&self) -> &'static str {
        sl_config::icons::FRACTAL
    }

    fn color(&self) -> egui::Color32 {
        egui::Color32::from_rgb(105, 153, 197)
    }

    fn is_renderable(&self) -> bool {
        true
    }

    fn supports_material(&self) -> bool {
        true
    }

    fn has_transform(&self) -> bool {
        true
    }

    fn branches(&self) -> &'static [&'static str] {
        &[
            "formula",
            "julia",
            "julia_mix",
            "orbit_offset",
            "inversion_strength",
            "inversion_center",
            "inversion_radius",
            "palette",
            "coloring",
            "trap_point",
            "trap_axis",
            "trap_scale",
            "render",
        ]
    }

    fn viewport_filter_key(&self) -> Option<&'static str> {
        Some("fractal")
    }

    fn viewport_filter_label(&self) -> &'static str {
        "Fractals"
    }

    fn default_custom_branches(&self) -> Map<String, Value> {
        default_fractal_custom_branches(Formula::Mandelbulb(Bulb::PRESET))
    }

    fn is_atomic_path(&self, path: &str) -> bool {
        path == "/formula"
    }

    fn is_discrete_path(&self, path: &str) -> bool {
        // Whole-number values hold between keys instead of interpolating to fractions.
        path == "/formula" || path == "/trap_axis" || integer_formula_leaf(path)
    }

    fn is_hidden_attribute(&self, path: &str) -> bool {
        (path.starts_with("/render/") && path != "/render/iterations")
            || matches!(
                path,
                "/formula/Hybrid/mandelbox/rotation_degrees"
                    | "/formula/Hybrid/kifs/rotation_degrees"
                    | "/formula/Hybrid/bulb/bailout"
            )
    }

    fn validate_attribute(&self, path: &str, value: &Value) -> Result<(), String> {
        match path {
            "/formula" => {
                Formula::deserialize(value).map_err(|e| format!("Invalid formula: {e}"))?;
                return Ok(());
            }
            "/formula/Hybrid/steps" => {
                <[HybridStep; 4]>::deserialize(value)
                    .map_err(|e| format!("Invalid hybrid steps: {e}"))?;
                return Ok(());
            }
            "/coloring" => {
                Coloring::deserialize(value).map_err(|e| format!("Invalid coloring: {e}"))?;
                return Ok(());
            }
            "/palette" => {
                PaletteScheme::deserialize(value).map_err(|e| format!("Invalid palette: {e}"))?;
                return Ok(());
            }
            "/render/backend" => {
                Backend::deserialize(value).map_err(|e| format!("Invalid render backend: {e}"))?;
                return Ok(());
            }
            "/render/denoise/mode" => {
                sl_studio::denoise::Mode::deserialize(value)
                    .map_err(|e| format!("Invalid denoise mode: {e}"))?;
                return Ok(());
            }
            "/render/denoise/quality" => {
                sl_studio::denoise::Quality::deserialize(value)
                    .map_err(|e| format!("Invalid denoise quality: {e}"))?;
                return Ok(());
            }
            "/trap_axis" => {
                return custom_u32("trap_axis", Some(value)).and_then(|axis| {
                    (axis <= 2)
                        .then_some(())
                        .ok_or_else(|| "/trap_axis must be 0, 1 or 2".into())
                });
            }
            _ => {}
        }
        if integer_formula_leaf(path) {
            value
                .as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| format!("{path} must be a whole number, got {value}"))?;
        }
        if let Some(choices) = self.attribute_choices(path) {
            let choices = choices?;
            if !choices.is_empty() && !choices.contains(value) {
                return Err(format!("Invalid enum value: {path}"));
            }
        }
        Ok(())
    }

    fn prepare(&self, branches: &BranchMap) -> Result<Arc<dyn PreparedNode>, String> {
        Ok(Arc::new(FractalParams::parse(branches)?))
    }

    fn attribute_choices(&self, path: &str) -> Option<Result<Vec<Value>, String>> {
        static FORMULA_CHOICES: LazyLock<Result<Vec<Value>, String>> =
            LazyLock::new(|| variants(Formula::PRESETS));
        if path == "/formula" {
            return Some(FORMULA_CHOICES.clone());
        }
        if path == "/formula/Hybrid/steps" {
            return None;
        }
        // Covers every family's enum leaves and `/formula/Hybrid/steps/<slot>`.
        if let Some(choices) = crate::formulas::formula_param_choices(path) {
            return Some(Ok(choices.to_vec()));
        }
        // Hybrid's nested KIFS sub-section is not a `/formula/<Variant>/<field>` path.
        (path == "/formula/Hybrid/kifs/kind")
            .then(|| Ok(crate::formulas::kifs_kind_choices().to_vec()))
    }

    fn choice_label(&self, path: &str, value: &Value) -> Option<String> {
        if path == "/formula" {
            return Formula::deserialize(value)
                .map(|f| f.name().to_owned())
                .ok()
                .or_else(|| value.as_object()?.keys().next().cloned());
        }
        None
    }

    fn choice_hover(&self, path: &str, value: &Value) -> Option<String> {
        if path == "/formula" {
            let f = Formula::deserialize(value).ok()?;
            return Some(format!("{}: {}", f.name(), f.hint()));
        }
        None
    }

    fn attribute_range(&self, path: &str) -> Option<(f64, f64)> {
        if let Some(range) = crate::formulas::formula_param_range(path) {
            return Some(range);
        }
        match path {
            "/trap_axis" => Some((0.0, 2.0)),
            "/trap_scale" => Some((0.0, f32::MAX as f64)),
            "/julia_mix" | "/inversion_strength" => Some((0.0, 1.0)),
            "/inversion_radius" => Some((1.0e-6, 1.0e6)),
            _ => None,
        }
    }

    fn attribute_slider(&self, path: &str) -> Option<Slider> {
        let lin = |min: f64, max: f64| {
            Some(Slider {
                min,
                max,
                log: false,
            })
        };
        let log = |min: f64, max: f64| {
            Some(Slider {
                min,
                max,
                log: true,
            })
        };
        if let Some(rest) = path.strip_prefix("/formula/") {
            let (family, param) = rest.split_once('/')?;
            return match (family, param.split_once('/')) {
                ("Hybrid", Some(("bulb", p))) => {
                    crate::formulas::formula_param_slider(&format!("/formula/Mandelbulb/{p}"))
                }
                ("Hybrid", Some(("mandelbox", p))) => {
                    crate::formulas::formula_param_slider(&format!("/formula/Mandelbox/{p}"))
                }
                ("Hybrid", Some(("kifs", p))) => {
                    crate::formulas::formula_param_slider(&format!("/formula/Kifs/{p}"))
                }
                _ => crate::formulas::formula_param_slider(path),
            };
        }
        match path {
            "/julia" => lin(-8.0, 8.0),
            "/julia_mix" | "/inversion_strength" => lin(0.0, 1.0),
            "/orbit_offset" | "/inversion_center" => lin(-4.0, 4.0),
            "/inversion_radius" => log(0.01, 10.0),
            "/trap_point" => lin(-16.0, 16.0),
            "/trap_scale" => log(0.001, 32.0),
            "/trap_axis" => lin(0.0, 2.0),
            _ => None,
        }
    }

    fn attribute_hint(&self, path: &str) -> Option<&'static str> {
        let path = match path.rsplit_once('/') {
            Some((parent, last)) if last.parse::<usize>().is_ok() => parent,
            _ => path,
        };
        if let Some(rest) = path.strip_prefix("/formula/") {
            let (family, field) = rest.split_once('/')?;
            let (family, field) = match (family, field.split_once('/')) {
                ("Hybrid", Some(("bulb", field))) => ("Mandelbulb", field),
                ("Hybrid", Some(("mandelbox", field))) => ("Mandelbox", field),
                ("Hybrid", Some(("kifs", field))) => ("Kifs", field),
                _ => (family, field),
            };
            return formula_hint(family, field);
        }
        match path {
            "/formula" => Some("The fractal family and its parameters."),
            "/julia" => Some(
                "Julia constant J (formulas with an orbit constant): the orbit constant the Julia mix blends towards.",
            ),
            "/julia_mix" => Some(
                "Mandelbrot to Julia (formulas with an orbit constant): every iteration adds mix(point, Julia constant, this). 0 is the Mandelbrot form, 1 the Julia form; keys morph between them.",
            ),
            "/orbit_offset" => Some(
                "Constant added every iteration on top of the orbit constant (formulas with an orbit constant): z -> f(z) + c + offset. Shifts and reshapes the set.",
            ),
            "/inversion_strength" => Some(
                "Spherical inversion of space before iterating (every formula): 0 off, 1 the full inversion in the sphere below, which turns the set inside out; between them a continuous blend.",
            ),
            "/inversion_center" => {
                Some("Centre of the inversion sphere, in the fractal's own space.")
            }
            "/inversion_radius" => Some(
                "Radius of the inversion sphere: points on it stay in place, nearer ones go far and far ones come near.",
            ),
            "/coloring" => Some(
                "Where the palette coordinate comes from: Radius (escape radius) or an orbit trap (origin, plane, point).",
            ),
            "/palette" => Some(
                "The colour ramp the fractal's colouring indexes (when the material takes its colour from the palette).",
            ),
            "/trap_point" => Some(
                "Orbit trap point: colour by how close the orbit comes to it (Trap point colouring).",
            ),
            "/trap_axis" => Some("Orbit trap plane normal axis (Trap plane colouring)."),
            "/trap_scale" => Some("How fast the trap distance runs through the palette."),
            _ => None,
        }
    }

    fn attribute_order(&self, base_path: &str) -> Option<u16> {
        Some(match base_path {
            "/formula" => 10,
            "/render/iterations" => 11,
            "/formula/Hybrid/steps" => 12,
            p if p.starts_with("/formula/") => {
                crate::formulas::formula_param_order(p).unwrap_or(20)
            }
            "/julia_mix" => 30,
            "/julia" => 31,
            "/orbit_offset" => 32,
            "/inversion_strength" => 40,
            "/inversion_radius" => 41,
            "/inversion_center" => 42,
            "/coloring" => 50,
            "/palette" => 51,
            "/trap_scale" => 52,
            "/trap_point" => 53,
            "/trap_axis" => 54,
            _ => return None,
        })
    }

    fn attribute_label(&self, path: &str) -> Option<String> {
        match path {
            "/formula/Hybrid/mandelbox/min_radius_ratio" => {
                return Some("Mandelbox · Min ratio".into());
            }
            "/formula/Hybrid/mandelbox/rotation_degrees" => {
                return Some("Mandelbox · Rotate".into());
            }
            "/formula/Hybrid/steps/0" => return Some("Step 1".into()),
            "/formula/Hybrid/steps/1" => return Some("Step 2".into()),
            "/formula/Hybrid/steps/2" => return Some("Step 3".into()),
            "/formula/Hybrid/steps/3" => return Some("Step 4".into()),
            _ => {}
        }
        if let Some((variant, _)) = crate::formulas::parse_formula_attr_path(path)
            && variant != "Hybrid"
            && let Some(label) = crate::formulas::formula_param_label(path)
        {
            if let Some((_, comp)) = path.rsplit_once('/')
                && comp.parse::<usize>().is_ok()
            {
                return Some(format!("{label} · {comp}"));
            }
            return Some(label);
        }
        if !path.starts_with("/formula/") {
            return None;
        }
        let parts: Vec<_> = path.trim_start_matches('/').split('/').collect();
        let tail = match parts.as_slice() {
            ["formula", "Hybrid", rest @ ..] => rest,
            ["formula", _, rest @ ..] if !rest.is_empty() => rest,
            all => all,
        };
        Some(
            tail.iter()
                .map(|part| {
                    let text = part.replace('_', " ");
                    let mut chars = text.chars();
                    chars
                        .next()
                        .map(|first| first.to_uppercase().collect::<String>() + chars.as_str())
                        .unwrap_or_default()
                })
                .collect::<Vec<_>>()
                .join(" · "),
        )
    }

    fn attribute_default(&self, path: &str) -> Option<Value> {
        static DEFAULTS: LazyLock<HashMap<String, Value>> = LazyLock::new(|| {
            let mut defaults = HashMap::new();
            for (k, v) in default_fractal_custom_branches(Formula::Mandelbulb(Bulb::PRESET)) {
                defaults.insert(format!("/{k}"), v);
            }
            defaults.insert("/render/iterations".into(), json!(20.0));
            defaults.insert("/coloring".into(), json!("Radius"));
            defaults.insert("/palette".into(), json!("Classic"));
            for formula in Formula::PRESETS {
                if let Ok(formula_json) = serde_json::to_value(formula) {
                    let mut f_attrs = Attrs::new();
                    discover_formula(&formula_json, "/formula", &mut f_attrs);
                    for (p, v) in f_attrs.iter() {
                        if p != "/formula" {
                            defaults
                                .entry(p.clone())
                                .or_insert_with(|| attr_json(v.clone()));
                        }
                    }
                }
            }
            for (slot, step) in Hybrid::PRESET.steps.into_iter().enumerate() {
                if let Ok(v) = serde_json::to_value(step) {
                    defaults.insert(format!("/formula/Hybrid/steps/{slot}"), v);
                }
            }
            defaults
        });
        DEFAULTS.get(path).cloned()
    }

    fn attribute_section(&self, path: &str) -> Option<&'static str> {
        if path == "/render/iterations"
            || path == "/coloring"
            || path == "/palette"
            || path.starts_with("/formula")
            || path.starts_with("/julia")
            || path.starts_with("/orbit_offset")
            || path.starts_with("/inversion_")
            || path.starts_with("/trap_")
        {
            Some("Fractal")
        } else {
            None
        }
    }

    fn attribute_sections(&self) -> &'static [&'static str] {
        &["Fractal"]
    }

    fn section_color(&self, group: &str) -> Option<egui::Color32> {
        (group == "Fractal").then(|| self.color())
    }

    fn inactive_reason(
        &self,
        path: &str,
        get: &dyn Fn(&str) -> Option<Value>,
    ) -> Option<Option<&'static str>> {
        static NO_ORBIT_REASON: LazyLock<[&'static str; TOTAL_FORMULA_COUNT as usize]> =
            LazyLock::new(|| {
                std::array::from_fn(|i| {
                    Box::leak(
                        format!("The {} formula has no orbit constant", FORMULAS[i].variant)
                            .into_boxed_str(),
                    ) as &'static str
                })
            });
        static ANALYTICAL_REASON: LazyLock<[&'static str; TOTAL_FORMULA_COUNT as usize]> =
            LazyLock::new(|| {
                std::array::from_fn(|i| {
                    Box::leak(
                        format!(
                            "{} is analytical (no iteration loop)",
                            FORMULAS[i].display_name
                        )
                        .into_boxed_str(),
                    ) as &'static str
                })
            });

        let under = |prefix: &str| {
            path.strip_prefix(prefix)
                .is_some_and(|rest| rest.is_empty() || rest.starts_with('/'))
        };
        if under("/julia") || under("/julia_mix") || under("/orbit_offset") {
            let formula = get("/formula")?;
            let family = formula.as_object()?.keys().next()?.as_str();
            // Every `Formula` variant has a `FORMULAS` row, and a validated node holds one.
            if let Some(meta) = crate::formulas::find_formula_meta_by_variant(family)
                && !meta.has_orbit_constant
            {
                return Some(Some(NO_ORBIT_REASON[meta.code as usize]));
            }
            if under("/julia") && get("/julia_mix")?.as_f64()? == 0.0 {
                return Some(Some("Julia mix is 0"));
            }
            return Some(None);
        }
        if under("/inversion_center") || under("/inversion_radius") {
            if get("/inversion_strength")?.as_f64()? == 0.0 {
                return Some(Some("Inversion strength is 0"));
            }
            return Some(None);
        }
        if under("/trap_point") || under("/trap_axis") || under("/trap_scale") {
            let coloring_val = get("/coloring")?;
            let coloring = coloring_val.as_str()?;
            let reason = match coloring {
                "Radius" => Some("Not used by the Radius colouring"),
                "TrapOrigin" if !under("/trap_scale") => {
                    Some("Not used by the TrapOrigin colouring")
                }
                "TrapPoint" if under("/trap_axis") => Some("Not used by the TrapPoint colouring"),
                _ => None,
            };
            return Some(reason);
        }
        if path == "/render/iterations" {
            let formula = get("/formula")?;
            let family = formula.as_object()?.keys().next()?;
            if let Some(meta) = crate::formulas::find_formula_meta_by_variant(family)
                && !meta.iterates
            {
                return Some(Some(ANALYTICAL_REASON[meta.code as usize]));
            }
            return Some(None);
        }
        if !path.starts_with("/formula/") {
            return None;
        }
        if let Some(rest) = path.strip_prefix("/formula/Hybrid/") {
            let hybrid_own = |p: &str| {
                rest.strip_prefix(p)
                    .is_some_and(|r| r.is_empty() || r.starts_with('/'))
            };
            if hybrid_own("mandelbox/rotation_degrees") || hybrid_own("kifs/rotation_degrees") {
                return Some(Some("A hybrid turns by the Mandelbulb rotation"));
            }
            if hybrid_own("bulb/bailout") {
                return Some(Some("A hybrid uses its own bailout"));
            }
            let (step, missing_reason) = match rest.split('/').next()? {
                "bulb" if hybrid_own("bulb/rotation_degrees") => return Some(None),
                "bulb" => ("Mandelbulb", "No hybrid step is Mandelbulb"),
                "mandelbox" => ("Mandelbox", "No hybrid step is Mandelbox"),
                "kifs" => ("KifsFold", "No hybrid step is KifsFold"),
                "apollonian_scale" => ("Inversion", "No hybrid step is Inversion"),
                _ => return Some(None),
            };
            let evaluated_steps = get("/formula/Hybrid/steps");
            let formula_fallback = if evaluated_steps.as_ref().is_some_and(Value::is_array) {
                None
            } else {
                Some(get("/formula")?)
            };
            let steps = match evaluated_steps.as_ref().and_then(Value::as_array) {
                Some(arr) => arr,
                None => formula_fallback
                    .as_ref()?
                    .get("Hybrid")?
                    .get("steps")?
                    .as_array()?,
            };
            let runs = steps.iter().any(|s| s.as_str() == Some(step))
                || (step == "Mandelbulb" && steps.iter().all(|s| s.as_str() == Some("Off")));
            if !runs {
                return Some(Some(missing_reason));
            }
            return Some(None);
        }
        let (variant, _) = crate::formulas::parse_formula_attr_path(path)?;
        let control_keys: [&str; 2] = match variant {
            "GyroidTpms" => ["/formula/GyroidTpms/octaves", ""],
            "VoronoiFoam" => ["/formula/VoronoiFoam/octaves", ""],
            "TorusKnotBraid" => ["/formula/TorusKnotBraid/braid_depth", ""],
            "DupinCyclide" => [
                "/formula/DupinCyclide/rib_frequency",
                "/formula/DupinCyclide/rib_depth",
            ],
            "PhyllotaxisShell" => [
                "/formula/PhyllotaxisShell/cone_height",
                "/formula/PhyllotaxisShell/sub_florets",
            ],
            "NautilusShell" => ["/formula/NautilusShell/rib_frequency", ""],
            "AstrolabeGimbal" => [
                "/formula/AstrolabeGimbal/gear_teeth",
                "/formula/AstrolabeGimbal/tooth_depth",
            ],
            _ => return None,
        };
        let slots = control_keys.map(|k| (!k.is_empty()).then(|| get(k)).flatten());
        let reason = crate::formulas::formula_param_inactive_reason(path, |k| {
            if k == control_keys[0] {
                slots[0].as_ref()
            } else if k == control_keys[1] {
                slots[1].as_ref()
            } else {
                None
            }
        })?;
        Some(Some(reason))
    }

    fn expand_attributes(&self, out: &mut Vec<WorldAttribute>) -> Result<(), String> {
        if let Some(steps) = out
            .iter()
            .find(|a| a.path == "/formula/Hybrid/steps")
            .map(|a| a.value.clone())
            .and_then(|v| v.as_array().cloned())
        {
            let choices = crate::formulas::hybrid_step_choices().to_vec();
            for (slot, value) in steps.iter().enumerate() {
                let path = format!("/formula/Hybrid/steps/{slot}");
                let label = self
                    .attribute_label(&path)
                    .unwrap_or_else(|| format!("Step {}", slot + 1));
                let default = self.attribute_default(&path);
                out.push(WorldAttribute {
                    path,
                    label,
                    value: value.clone(),
                    frames: vec![],
                    keyable: false,
                    component: Some(slot),
                    choices: choices.clone(),
                    choice_labels: vec![],
                    range: None,
                    slider: None,
                    color: false,
                    default,
                });
            }
        }
        Ok(())
    }

    fn sync_component_edit(&self, parent: &str, vector: &Value, attrs: &mut Attrs) {
        if parent != "/formula/Hybrid/steps" {
            return;
        }
        // The trait cannot fail: a mismatch is logged (an edit that leaves the formula invalid
        // is then rejected and rolled back by `prepare`).
        let Some(mut formula) = attrs.get("/formula").cloned().map(attr_json) else {
            log::warn!("Hybrid step edit without a /formula attribute; formula not synced");
            return;
        };
        let Some(slots) = formula
            .pointer_mut("/Hybrid/steps")
            .and_then(Value::as_array_mut)
        else {
            log::warn!("Hybrid step edit on a non-Hybrid formula; formula not synced");
            return;
        };
        let Some(fresh) = vector.as_array() else {
            log::warn!("Hybrid steps edit is not an array: {vector}");
            return;
        };
        if slots.len() != fresh.len() {
            log::warn!(
                "Hybrid steps edit has {} slots, the formula has {}",
                fresh.len(),
                slots.len()
            );
            return;
        }
        slots.clone_from_slice(fresh);
        attrs.set("/formula", to_attr(&formula));
    }

    fn family_count(&self) -> u32 {
        Formula::COUNT
    }

    fn payload_kind(&self) -> sl_studio::NodePayloadKind {
        sl_studio::NodePayloadKind::SdfField
    }
}

/// KIFS shift vector and Menger fold height, as packed into `P_KIFS_SHIFT` /
/// `P_KIFS_FOLD_HEIGHT`: each step is `z' = s * fold(rot(z)) - shift`.
fn kifs_layout(k: &Kifs) -> ([f32; 3], f32) {
    let s = k.scale;
    let base = k.kind.centre();
    let cc = [
        base[0] + k.offset[0],
        base[1] + k.offset[1],
        base[2] + k.offset[2],
    ];
    let (shift_z, fold_height) = match k.kind {
        KifsKind::Menger => (
            0.0,
            if s != 0.0 {
                0.5 * cc[2] * (s - 1.0) / s
            } else {
                0.0
            },
        ),
        _ => ((s - 1.0) * cc[2], 0.0),
    };
    ([(s - 1.0) * cc[0], (s - 1.0) * cc[1], shift_z], fold_height)
}

/// Radius of a ball holding the whole KIFS, from its parameters (plan1 D1: the fixed
/// per-kind radius cut sets moved by `offset`). Each step is `s * fold(rot(z)) - shift`; the
/// rotation and the symmetry folds are reflections through the origin (norm-preserving) and
/// the Menger plane fold at `h` moves a point by at most `2|h|`, so
/// `|z'| >= s(|z| - 2|h|) - |shift|`. Beyond `R* = (|shift| + 2|h|s) / (s - 1)` the orbit
/// only grows, so points farther than `max(base radius, R*)` never meet the final base test.
/// `None` for `s <= 1`, where the map does not expand and no such bound exists.
fn kifs_reach(k: &Kifs) -> Option<f32> {
    let s = k.scale;
    if s <= 1.0 {
        return None;
    }
    let (shift, h) = kifs_layout(k);
    let r = (length(shift) + 2.0 * h.abs() * s) / (s - 1.0);
    Some(k.kind.bounding_radius().max(r))
}

/// Radius of a ball holding the whole Mandelbox, from its scale and fold limit (plan1 D1: the
/// fixed 4.0637 framing cut sets with a scale near 1). Per axis, with the orbit constant the
/// point itself: for `s > 1` the extreme bounded orbit is the 2-cycle `B -> -2F -> B`, so
/// `|c_i| <= 2F(s + 1)/(s - 1)` (6 for the classic s = 2, F = 1); for `s < -1` it is the fixed
/// point `fold(B) = 0`, `|c_i| <= 2F`. The sphere fold only scales by >= 1, which moves a
/// point out faster, so the set lies in that cube: radius `sqrt(3) * B`. `None` for
/// `|s| <= 1`, where the map does not expand. Checked by
/// `tests::mandelbox_and_kifs_clips_hold_the_set` with an escape-time probe of `box_step`.
fn mandelbox_reach(b: &MandelBox) -> Option<f32> {
    let (s, f) = (b.scale, b.fold_limit.abs());
    let axis = if s > 1.0 {
        2.0 * f * (s + 1.0) / (s - 1.0)
    } else if s < -1.0 {
        2.0 * f
    } else {
        return None;
    };
    Some(axis * 3.0f32.sqrt())
}

/// What packing a fractal needs besides the fractal itself: its object's placement and the
/// few scene values the fractal's parameter slots carry.
struct Placement {
    scale: f32,
    offset: [f32; 3],
    camera_dist: f32,
    camera_target: [f32; 3],
    coloring: Coloring,
    iterations: f32,
}

impl Placement {
    /// A unit object at the origin: for the local clip radius (`local_pick_radius`), which
    /// reads only `P_CLIP_RADIUS` (the camera, colouring and iterations do not enter it).
    const LOCAL: Self = Self {
        scale: 1.0,
        offset: [0.0; 3],
        camera_dist: 0.0,
        camera_target: [0.0; 3],
        coloring: Coloring::Radius,
        iterations: 0.0,
    };

    fn of(scene: &Scene, camera_dist: f32) -> Self {
        Self {
            scale: scene.object.scale,
            offset: scene.object.offset,
            camera_dist,
            camera_target: scene.camera.target,
            coloring: scene.coloring,
            iterations: scene.render.iterations,
        }
    }
}

impl PreparedNode for FractalParams {
    fn framing_radius(&self) -> f32 {
        self.formula.framing_radius()
    }

    fn bound_radius(&self) -> f32 {
        self.formula.bound_radius()
    }

    fn local_pick_radius(&self) -> f32 {
        if self.inversion_strength != 0.0 {
            let mut p = [0.0f32; P_COUNT];
            p[P_BAILOUT] = 2.0;
            pack_fractal(self, &Placement::LOCAL, &mut p);
            p[P_CLIP_RADIUS]
        } else {
            self.formula.framing_radius()
        }
    }

    fn family_code(&self) -> u32 {
        self.formula.code()
    }

    fn summary_label(&self) -> &str {
        self.formula.name()
    }

    fn pack_object(&self, scene: &Scene, p: &mut [f32], camera_dist: f32) {
        pack_fractal(self, &Placement::of(scene, camera_dist), p);
    }
}

fn pack_fractal(f: &FractalParams, at: &Placement, p: &mut [f32]) {
    debug_assert!(p.len() >= P_COUNT);
    if p.len() < P_COUNT {
        return;
    }
    let (object_scale, object_offset, camera_dist) = (at.scale, at.offset, at.camera_dist);
    let formula = f.formula;
    let radius = formula.framing_radius();
    let FractalParams {
        julia,
        julia_mix,
        orbit_offset,
        inversion_strength,
        inversion_center,
        inversion_radius,
        trap_point,
        trap_axis,
        trap_scale,
        ..
    } = *f;

    let bulb_escape = |b: &Bulb| -> f32 {
        b.bailout
            .max(julia_mix * length(julia) + length(orbit_offset))
    };

    let abs_scale = object_scale.abs();
    let mut object_radius = (REACH_MARGIN_FRAMES * radius).max(formula.bound_radius()) * abs_scale;
    if let Formula::Mandelbulb(b) = &formula
        && (julia_mix != 0.0 || orbit_offset != [0.0; 3])
    {
        object_radius = object_radius.max(bulb_escape(b) * abs_scale);
    }

    // The parameter-derived reach only ever widens the framing-based clip: where no bound is
    // derived (non-contracting scales, see `mandelbox_reach` / `kifs_reach`) the old clip stays.
    let framed = REACH_MARGIN_FRAMES * radius;
    let clip_object = match &formula {
        Formula::Mandelbulb(_)
        | Formula::QuaternionJulia(_)
        | Formula::Hybrid(_)
        | Formula::Bristorbrot(_)
        | Formula::BurningShip(_)
        | Formula::HarmonicBulb(_)
        | Formula::Mandelbar3D(_) => None,
        Formula::Mandelbox(b) => Some(mandelbox_reach(b).map_or(framed, |r| r.max(framed))),
        Formula::Kifs(k) => Some(kifs_reach(k).map_or(framed, |r| r.max(framed))),
        _ if formula.bound_radius() > 0.0 => Some(formula.bound_radius()),
        _ => Some(framed),
    };

    p[P_FAMILY] = formula.code() as f32;
    p[P_ITERATIONS] = at.iterations;
    let orbit = formula.has_orbit_constant();
    p[P_JULIA] = if orbit { julia_mix } else { 0.0 };
    put3(p, P_JULIA_C, julia);
    put3(
        p,
        P_ORBIT_OFFSET,
        if orbit { orbit_offset } else { [0.0; 3] },
    );
    p[P_INVERT_STRENGTH] = inversion_strength;
    put3(p, P_INVERT_CENTER, inversion_center);
    p[P_INVERT_RADIUS] = inversion_radius;
    p[P_BOUND_RADIUS] = formula.bound_radius();

    let set_rotation = |p: &mut [f32], degrees: [f32; 3]| {
        let reduced = degrees.map(|a| a % 360.0);
        if reduced.iter().all(|&a| a == 0.0) {
            p[P_ITER_ROTATE] = 0.0;
        } else {
            p[P_ITER_ROTATE] = 1.0;
            put9(p, P_ITER_ROT, euler_matrix(reduced));
        }
    };
    let pack_bulb = |p: &mut [f32], b: &Bulb| {
        p[P_BULB_POWER] = b.power;
        p[P_BULB_THETA_POWER] = b.angle_scale[0] * b.power;
        p[P_BULB_PHI_POWER] = b.angle_scale[1] * b.power;
        p[P_BULB_THETA_PHASE] = b.angle_phase_degrees[0].to_radians();
        p[P_BULB_PHI_PHASE] = b.angle_phase_degrees[1].to_radians();
        p[P_BULB_GROWTH] = 1.0f32
            .max(b.angle_scale[0].abs())
            .max(b.angle_scale[1].abs());
    };
    let pack_box = |p: &mut [f32], b: &MandelBox| {
        let min_r = b.min_radius_ratio * b.fixed_radius;
        // A zero inner radius would divide by zero in the sphere fold (`fixed_r2 / r2` at the
        // origin); the same guard as AmazingSurface's and Mobius's packing.
        let min_r2 = (min_r * min_r).max(1.0e-6);
        p[P_BOX_SCALE] = b.scale;
        p[P_BOX_FOLD] = b.fold_limit;
        p[P_BOX_MIN_R2] = min_r2;
        p[P_BOX_FIXED_R2] = (b.fixed_radius * b.fixed_radius).max(min_r2);
    };
    let pack_kifs = |p: &mut [f32], k: &Kifs| {
        let (shift, fold_height) = kifs_layout(k);
        p[P_KIFS_KIND] = k.kind.code() as f32;
        p[P_KIFS_SCALE] = k.scale;
        put3(p, P_KIFS_SHIFT, shift);
        p[P_KIFS_FOLD_HEIGHT] = fold_height;
        p[P_KIFS_BOUND] = k.kind.bounding_radius();
    };

    match &formula {
        Formula::Mandelbulb(b) => {
            pack_bulb(p, b);
            set_rotation(p, b.rotation_degrees);
            p[P_BAILOUT] = bulb_escape(b);
            p[P_BULB_FAST8] = (b.power == 8.0
                && b.angle_scale == [1.0, 1.0]
                && b.angle_phase_degrees == [0.0, 0.0]
                && p[P_ITER_ROTATE] == 0.0) as u32 as f32;
        }
        Formula::Mandelbox(b) => {
            pack_box(p, b);
            set_rotation(p, b.rotation_degrees);
        }
        Formula::QuaternionJulia(q) => {
            let m = rotation_matrix4(q.rotation_degrees);
            for row in 0..4 {
                for col in 0..3 {
                    p[P_QUAT_ROWS + 3 * row + col] = m[row][col];
                }
                p[P_QUAT_OFFSET + row] = m[row][3] * q.slice_w;
            }
            p[P_QUAT_C..P_QUAT_C + 4].copy_from_slice(&q.constant);
            let cl = q.constant.iter().map(|v| v * v).sum::<f32>().sqrt();
            p[P_BAILOUT] = q.bailout.max(cl);
        }
        Formula::Kifs(k) => {
            pack_kifs(p, k);
            set_rotation(p, k.rotation_degrees);
        }
        Formula::Kleinian(k) => {
            let (a, b) = (k.a as f64, k.b as f64);
            p[P_KLEIN_A] = k.a;
            p[P_KLEIN_B] = k.b;
            p[P_KLEIN_SKEW] = if a != 0.0 { (b.abs() / a) as f32 } else { 0.0 };
            p[P_KLEIN_LINE_AMP] = ((2.0 * a - 1.95) / 4.0) as f32;
            p[P_KLEIN_LINE_RATE] = (7.2 - (1.95 - a) * 15.0) as f32;
        }
        Formula::PseudoKleinian(k) => {
            put3(p, P_PK_BOX, k.box_size);
            p[P_PK_SIZE] = k.size;
            put3(p, P_PK_C, k.c);
            put3(p, P_PK_OFFSET, k.offset);
            p[P_PK_THICKNESS] = k.thickness;
        }
        Formula::Apollonian(a) => {
            p[P_APOLLO_SCALE] = a.scale;
        }
        Formula::Hybrid(h) => {
            pack_bulb(p, &h.bulb);
            pack_box(p, &h.mandelbox);
            pack_kifs(p, &h.kifs);
            p[P_APOLLO_SCALE] = h.apollonian_scale;
            p[P_BAILOUT] = h.bailout.max(0.0);
            set_rotation(p, h.bulb.rotation_degrees);
            let mut packed_steps = 0u32;
            let mut active_count = 0u32;
            for step in h.steps {
                let code = match step {
                    HybridStep::Off => continue,
                    HybridStep::Mandelbulb => 1u32,
                    HybridStep::Mandelbox => 2u32,
                    HybridStep::KifsFold => 3u32,
                    HybridStep::Inversion => 4u32,
                };
                packed_steps |= code << (3 * active_count);
                active_count += 1;
            }
            if active_count == 0 {
                packed_steps = 1;
                active_count = 1;
            }
            p[P_HYBRID_STEPS] = packed_steps as f32;
            p[P_HYBRID_COUNT] = active_count as f32;
        }
        other => {
            let orbit_escape = julia_mix * length(julia) + length(orbit_offset);
            crate::formulas::pack_extended_formula(p, other, orbit_escape, euler_matrix);
        }
    }

    let clip = match clip_object {
        Some(r) => r,
        None => p[P_BAILOUT].max(0.0) * 1.001,
    };
    let clip = if inversion_strength == 0.0 {
        clip
    } else {
        fractal_core::inversion::reach(inversion_center, inversion_radius, inversion_strength, clip)
            .unwrap_or(INVERSION_FAR_REACH)
    };
    let clip_scaled = clip * abs_scale;
    object_radius = object_radius.max(clip_scaled);
    p[P_CLIP_RADIUS] = clip_scaled;
    p[P_MAX_DISTANCE] =
        camera_dist + length(at.camera_target) + length(object_offset) + object_radius;

    p[P_COLOR_MODE] = at.coloring as u32 as f32;
    put3(p, P_TRAP_POINT, trap_point);
    let mut axis = [0.0; 3];
    axis[trap_axis.min(2) as usize] = 1.0;
    put3(p, P_TRAP_NORMAL, axis);
    p[P_TRAP_SCALE] = trap_scale;
}

fn put3(p: &mut [f32], i: usize, v: [f32; 3]) {
    p[i..i + 3].copy_from_slice(&v);
}

fn put9(p: &mut [f32], i: usize, m: [[f32; 3]; 3]) {
    for r in 0..3 {
        p[i + 3 * r..i + 3 * r + 3].copy_from_slice(&m[r]);
    }
}

fn euler_matrix(degrees: [f32; 3]) -> [[f32; 3]; 3] {
    let [(sx, cx), (sy, cy), (sz, cz)] = degrees.map(|a| (a as f64).to_radians().sin_cos());
    [
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy, cy * sx, cy * cx],
    ]
    .map(|row| row.map(|v| v as f32))
}

fn rotation_matrix4(degrees: [f32; 3]) -> [[f32; 4]; 4] {
    let mut m = [[0.0f64; 4]; 4];
    for (i, row) in m.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for (axis, angle) in degrees.into_iter().enumerate() {
        let (s, c) = (angle as f64).to_radians().sin_cos();
        let mut turn = [[0.0f64; 4]; 4];
        for (i, row) in turn.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        turn[axis][axis] = c;
        turn[3][3] = c;
        turn[axis][3] = -s;
        turn[3][axis] = s;
        let mut out = [[0.0f64; 4]; 4];
        for r in 0..4 {
            for k in 0..4 {
                out[r][k] = (0..4).map(|j| turn[r][j] * m[j][k]).sum();
            }
        }
        m = out;
    }
    m.map(|row| row.map(|v| v as f32))
}

#[cfg(test)]
mod tests {
    use sl_studio::render_prep::WorldUpload;
    use sl_studio::world::{WorldCommand, WorldDocument, WorldEditor};

    use super::*;

    /// plan1 F1/F2: a broken formula or a mistyped branch is a typed error at every boundary
    /// (rejected by `prepare`, an edit rolled back), never a silent Mandelbulb or default.
    #[test]
    fn invalid_fractal_branches_are_rejected_not_defaulted() {
        crate::engine::init_warpbro();
        let defaults = FractalParams::new(Formula::Mandelbulb(Bulb::PRESET)).branches();
        let parsed = FractalParams::parse(&defaults).expect("default branches are valid");
        // The stored form round-trips: what `edit_fractal` writes is what `prepare` reads.
        assert_eq!(parsed, FractalParams::new(Formula::Mandelbulb(Bulb::PRESET)));
        for (key, bad) in [
            ("formula", json!({"NotAFormula": {}})),
            ("julia", json!([0.0, "x", 0.0])),
            ("julia_mix", json!("0.5")),
            ("inversion_radius", json!(0.0)),
            ("trap_axis", json!(1.0)),
            ("trap_axis", json!(3)),
        ] {
            let mut broken = defaults.clone();
            broken.insert(key.into(), bad.clone());
            assert!(
                FractalNodeSpec.prepare(&broken).is_err(),
                "{key} = {bad} must be rejected"
            );
        }
        let mut missing = defaults;
        missing.remove("trap_scale");
        assert!(FractalNodeSpec.prepare(&missing).is_err());
        // A typed edit that breaks an invariant is refused and leaves the scene as it was.
        let mut bulb = Scene::preset_fractal(Formula::Mandelbulb(Bulb::PRESET));
        assert!(bulb.edit_fractal(|f| f.trap_axis = 3).is_err());
        assert_eq!(bulb.trap_axis(), 1);

        let scene = Scene::preset_fractal(Formula::Mandelbulb(Bulb::PRESET));
        let mut editor = WorldEditor::new(WorldDocument::from_scene(&scene));
        let id = editor
            .document
            .nodes()
            .into_iter()
            .find(|n| n.kind == FRACTAL_KIND)
            .expect("Fractal node")
            .id;
        // A string passes the leaf check (no range, no choices) but breaks the node: only
        // `prepare` can reject it. (A numeric value out of range would just be clamped
        // to `attribute_range` by the editor.)
        let edit = editor.execute_edit(
            WorldCommand::SetAttribute {
                id,
                path: "/julia_mix".into(),
                value: json!("half"),
                frame: 0.0,
            },
            None,
        );
        assert!(edit.unwrap_err().contains("julia_mix"));
        let snap = editor
            .document
            .snapshot(0.0)
            .expect("snapshot after rollback");
        assert_eq!(snap.objects[0].julia_mix(), 0.0);
    }

    /// plan1 F3: integer formula leaves accept whole numbers only and hold between keys.
    #[test]
    fn integer_formula_leaves_stay_whole() {
        let path = "/formula/GyroidTpms/octaves";
        assert!(integer_formula_leaf(path));
        assert!(!integer_formula_leaf("/formula/GyroidTpms/frequency"));
        assert!(FractalNodeSpec.is_discrete_path(path));
        assert!(FractalNodeSpec.validate_attribute(path, &json!(4)).is_ok());
        assert!(
            FractalNodeSpec
                .validate_attribute(path, &json!(2.5))
                .is_err()
        );
        // A float, even a whole one, is what the `u32` field fails to deserialize.
        assert!(
            FractalNodeSpec
                .validate_attribute(path, &json!(3.0))
                .is_err()
        );
    }

    /// plan1 S5: every choice the Attribute Editor offers deserializes in its family.
    #[test]
    fn every_formula_choice_deserializes() {
        for meta in &FORMULAS {
            let preset = serde_json::to_value(Formula::PRESETS[meta.code as usize])
                .expect("preset serializes");
            for param in meta.params {
                let Some(choices) = param.choices else {
                    continue;
                };
                for choice in choices() {
                    let mut formula = preset.clone();
                    let slot = formula
                        .pointer_mut(&format!("/{}/{}", meta.variant, param.name))
                        .expect("choice parameter exists in the preset");
                    match slot {
                        Value::Array(slots) => slots.fill(choice.clone()),
                        other => *other = choice.clone(),
                    }
                    Formula::deserialize(&formula).unwrap_or_else(|e| {
                        panic!("{}/{} = {choice}: {e}", meta.variant, param.name)
                    });
                }
            }
        }
        let kifs = FractalNodeSpec
            .attribute_choices("/formula/Hybrid/kifs/kind")
            .expect("Hybrid KIFS kind has choices")
            .expect("choices build");
        assert_eq!(kifs.len(), KifsKind::ALL.len());
    }

    /// plan1 D1: the Mandelbox and KIFS clip spheres hold the whole set at the parameters the
    /// sliders allow, not only at the presets (the fixed radii cut sets with a scale near 1 or
    /// a large KIFS offset). The Mandelbox side is an escape-time probe mirroring the kernel's
    /// `box_step` (fractal-kernels): every point just outside the clip must escape.
    #[test]
    fn mandelbox_and_kifs_clips_hold_the_set() {
        fn clip_of(formula: Formula) -> f32 {
            let mut p = [0.0f32; P_COUNT];
            pack_fractal(&FractalParams::new(formula), &Placement::LOCAL, &mut p);
            p[P_CLIP_RADIUS]
        }
        fn box_fold(v: f32, limit: f32) -> f32 {
            if v > limit {
                2.0 * limit - v
            } else if v < -limit {
                -2.0 * limit - v
            } else {
                v
            }
        }
        fn escapes(b: &MandelBox, c: [f32; 3]) -> bool {
            let min_r = b.min_radius_ratio * b.fixed_radius;
            let min_r2 = (min_r * min_r).max(1.0e-6);
            let fixed_r2 = (b.fixed_radius * b.fixed_radius).max(min_r2);
            let mut z = c;
            for _ in 0..400 {
                let t = z.map(|v| box_fold(v, b.fold_limit));
                let r2 = t[0] * t[0] + t[1] * t[1] + t[2] * t[2];
                let k = if r2 < min_r2 {
                    fixed_r2 / min_r2
                } else if r2 < fixed_r2 {
                    fixed_r2 / r2
                } else {
                    1.0
                };
                z = std::array::from_fn(|i| t[i] * k * b.scale + c[i]);
                if length(z) > 1.0e6 {
                    return true;
                }
            }
            false
        }
        // A Fibonacci sphere plus the cube diagonal, where the bound is tight.
        let golden = std::f32::consts::PI * (3.0 - 5.0f32.sqrt());
        let mut dirs: Vec<[f32; 3]> = (0..256)
            .map(|i| {
                let y = 1.0 - 2.0 * (i as f32 + 0.5) / 256.0;
                let r = (1.0 - y * y).sqrt();
                let a = golden * i as f32;
                [r * a.cos(), y, r * a.sin()]
            })
            .collect();
        dirs.push([1.0 / 3.0f32.sqrt(); 3]);
        for scale in [1.25f32, 1.5, 2.0, 3.0, -1.5, -2.0, -3.0] {
            for fixed_radius in [1.0f32, 2.0] {
                for min_radius_ratio in [0.25f32, 0.5] {
                    for fold_limit in [0.5f32, 1.0] {
                        let b = MandelBox {
                            scale,
                            min_radius_ratio,
                            fixed_radius,
                            fold_limit,
                            rotation_degrees: [0.0; 3],
                        };
                        let clip = clip_of(Formula::Mandelbox(b));
                        for d in &dirs {
                            for beyond in [1.02f32, 1.2] {
                                let c = d.map(|v| v * clip * beyond);
                                assert!(
                                    escapes(&b, c),
                                    "{b:?}: {c:?} beyond clip {clip} stays bounded"
                                );
                            }
                        }
                    }
                }
            }
        }
        for kind in KifsKind::ALL {
            let mut k = Kifs::preset(kind);
            k.offset = [12.0, -9.0, 5.0];
            let centre = kind.centre();
            let cc: [f32; 3] = std::array::from_fn(|i| centre[i] + k.offset[i]);
            let clip = clip_of(Formula::Kifs(k));
            assert!(
                clip >= length(cc),
                "{kind:?}: clip {clip} < |cc| {}",
                length(cc)
            );
        }
    }

    /// plan1 S6: labels and hints of the classic families come from `FORMULAS` too.
    #[test]
    fn classic_family_labels_and_hints_come_from_the_table() {
        assert_eq!(
            FractalNodeSpec
                .attribute_label("/formula/Mandelbulb/rotation_degrees")
                .as_deref(),
            Some("Rotation (°)")
        );
        assert_eq!(
            FractalNodeSpec.attribute_hint("/formula/Hybrid/bulb/power"),
            find_param_meta("/formula/Mandelbulb/power").map(|p| p.hint)
        );
    }

    #[test]
    fn fractal_node_lifecycle_and_parameter_roundtrip() {
        use sl_studio::StudioModule;
        crate::engine::init_warpbro();
        assert_eq!(
            FractalNodeSpec.payload_kind(),
            sl_studio::NodePayloadKind::SdfField
        );
        assert_eq!(crate::engine::WarpBroFractalModule.id(), "warpbro.fractals");
        assert_eq!(
            crate::engine::WarpBroFractalModule.title(),
            "WarpBro 3D Fractals"
        );
        assert_eq!(crate::engine::WarpBroFractalModule.node_specs().len(), 1);
        let mut scene = Scene::preset_fractal(Formula::Mandelbulb(Bulb::PRESET));
        scene.set_orbit_offset([0.1, -0.2, 0.3]);
        scene.set_inversion_strength(0.5);
        scene.set_inversion_center([0.25, 0.5, -0.25]);
        scene.set_inversion_radius(1.75);
        scene.set_trap_point([1.0, 2.0, 3.0]);
        scene.set_trap_axis(2);
        scene.set_trap_scale(2.5);

        let mut editor = WorldEditor::new(WorldDocument::from_scene(&scene));
        let fractal_id = editor
            .document
            .nodes()
            .into_iter()
            .find(|n| n.kind == FRACTAL_KIND)
            .expect("WorldDocument::from_scene must create a Fractal node")
            .id;

        let snap = editor
            .document
            .snapshot(0.0)
            .expect("snapshot with Fractal node");
        assert_eq!(snap.objects.len(), 1);
        assert_eq!(snap.orbit_offset(), [0.1, -0.2, 0.3]);
        assert_eq!(snap.inversion_strength(), 0.5);
        assert_eq!(snap.inversion_center(), [0.25, 0.5, -0.25]);
        assert_eq!(snap.inversion_radius(), 1.75);
        assert_eq!(snap.trap_point(), [1.0, 2.0, 3.0]);
        assert_eq!(snap.trap_axis(), 2);
        assert_eq!(snap.trap_scale(), 2.5);
        let upload = WorldUpload::new(&snap, 64, 64).expect("world upload with Fractal node");
        assert_eq!(upload.params[P_OBJECT_COUNT], 1.0);
        assert!(!upload.objects.is_empty());

        editor
            .execute_edit(
                WorldCommand::SetAttribute {
                    id: fractal_id,
                    path: "/visible".into(),
                    value: json!(false),
                    frame: 0.0,
                },
                None,
            )
            .expect("hide Fractal node");
        let hidden_snap = editor
            .document
            .snapshot(0.0)
            .expect("snapshot with hidden Fractal");
        assert!(hidden_snap.objects.is_empty());
        let hidden_upload =
            WorldUpload::new(&hidden_snap, 64, 64).expect("world upload with hidden Fractal");
        assert_eq!(hidden_upload.params[P_OBJECT_COUNT], 0.0);
        assert!(hidden_upload.objects.is_empty());

        editor
            .execute_edit(WorldCommand::Delete(fractal_id), None)
            .expect("delete Fractal node");
        let empty_snap = editor
            .document
            .snapshot(0.0)
            .expect("snapshot without Fractal");
        assert!(empty_snap.objects.is_empty());
        let empty_upload =
            WorldUpload::new(&empty_snap, 64, 64).expect("world upload without Fractal");
        assert_eq!(empty_upload.params[P_OBJECT_COUNT], 0.0);
        assert!(empty_upload.objects.is_empty());
    }

    #[test]
    fn hybrid_steps_validation_defaults_and_inactive_reasons() {
        crate::engine::init_warpbro();
        assert!(
            FractalNodeSpec
                .attribute_choices("/formula/Hybrid/steps")
                .is_none()
        );
        assert_eq!(
            FractalNodeSpec
                .attribute_choices("/formula/Hybrid/steps/0")
                .expect("slot choices")
                .expect("valid choices")
                .len(),
            HybridStep::ALL.len()
        );
        assert!(
            FractalNodeSpec
                .validate_attribute(
                    "/formula/Hybrid/steps",
                    &json!(["Mandelbulb", "KifsFold", "Off", "Off"]),
                )
                .is_ok()
        );
        assert!(
            FractalNodeSpec
                .validate_attribute("/formula/Hybrid/steps", &json!("Mandelbulb"))
                .is_err()
        );
        assert!(
            FractalNodeSpec
                .validate_attribute("/coloring", &json!("TrapPoint"))
                .is_ok()
        );
        assert!(
            FractalNodeSpec
                .validate_attribute("/coloring", &json!("InvalidColoring"))
                .is_err()
        );
        assert!(
            FractalNodeSpec
                .validate_attribute("/palette", &json!(PaletteScheme::Fire))
                .is_ok()
        );
        assert!(
            FractalNodeSpec
                .validate_attribute("/palette", &json!("InvalidPalette"))
                .is_err()
        );
        assert!(
            FractalNodeSpec
                .validate_attribute("/render/backend", &json!("InvalidBackend"))
                .is_err()
        );
        let mut editor = WorldEditor::new(WorldDocument::from_scene(&Scene::preset_fractal(
            Formula::Mandelbulb(Bulb::PRESET),
        )));
        let fractal_id = editor
            .document
            .nodes()
            .into_iter()
            .find(|n| n.kind == FRACTAL_KIND)
            .expect("Fractal node")
            .id;
        assert!(
            editor
                .execute_edit(
                    WorldCommand::SetAttribute {
                        id: fractal_id,
                        path: "/coloring".into(),
                        value: json!("NotAColoring"),
                        frame: 0.0,
                    },
                    None,
                )
                .is_err()
        );
        assert!(
            editor
                .execute_edit(
                    WorldCommand::SetAttribute {
                        id: fractal_id,
                        path: "/palette".into(),
                        value: json!("NotAPalette"),
                        frame: 0.0,
                    },
                    None,
                )
                .is_err()
        );
        for (slot, expected) in Hybrid::PRESET.steps.into_iter().enumerate() {
            assert_eq!(
                FractalNodeSpec.attribute_default(&format!("/formula/Hybrid/steps/{slot}")),
                Some(json!(expected))
            );
        }

        let static_formula = serde_json::to_value(Formula::Hybrid(Hybrid::PRESET)).unwrap();
        let animated_steps = json!(["KifsFold", "Off", "Off", "Off"]);
        let kifs_reason =
            FractalNodeSpec.inactive_reason("/formula/Hybrid/kifs/scale", &|p| match p {
                "/formula/Hybrid/steps" => Some(animated_steps.clone()),
                "/formula" => Some(static_formula.clone()),
                _ => None,
            });
        assert_eq!(kifs_reason, Some(None));
        let box_reason =
            FractalNodeSpec.inactive_reason("/formula/Hybrid/mandelbox/scale", &|p| match p {
                "/formula/Hybrid/steps" => Some(animated_steps.clone()),
                "/formula" => Some(static_formula.clone()),
                _ => None,
            });
        assert_eq!(box_reason, Some(Some("No hybrid step is Mandelbox")));
        assert_eq!(
            FractalNodeSpec.inactive_reason("/julia", &|p| {
                (p == "/formula").then(|| json!({ "Kifs": {} }))
            }),
            Some(Some("The Kifs formula has no orbit constant"))
        );
        assert_eq!(
            FractalNodeSpec.inactive_reason("/render/iterations", &|p| {
                (p == "/formula").then(|| json!({ "GyroidTpms": {} }))
            }),
            Some(Some(
                "Gyroid / Schwarz TPMS is analytical (no iteration loop)"
            ))
        );
        // An unknown variant never reaches the inspector (`prepare` rejects it), so it
        // gets no guessed "no orbit constant" reason: the check falls through to the Julia mix,
        // which this probe does not provide.
        assert_eq!(
            FractalNodeSpec.inactive_reason("/julia", &|p| {
                (p == "/formula").then(|| json!({ "UnknownCustomVariant": {} }))
            }),
            None
        );
    }

    #[test]
    fn packed_max_distance_covers_clip_sphere_across_gallery_and_scales() {
        crate::engine::init_warpbro();
        for mut scene in Scene::fractal_gallery().unwrap() {
            let (_, _, dist) = scene.camera_eye();
            let p = scene.pack(64, 64);
            assert!(
                p.iter().all(|v| v.is_finite()),
                "non-finite pack in {}",
                scene.name
            );
            assert!(p[P_CLIP_RADIUS] >= 0.0);
            let min_max_dist =
                dist + length(scene.camera.target) + length(scene.object.offset) + p[P_CLIP_RADIUS];
            assert!(
                p[P_MAX_DISTANCE] + 1.0e-4 >= min_max_dist,
                "{}: P_MAX_DISTANCE {} < {}",
                scene.name,
                p[P_MAX_DISTANCE],
                min_max_dist
            );

            scene.object.scale = -1.5;
            let p_neg = scene.pack(64, 64);
            assert!(p_neg.iter().all(|v| v.is_finite()));
            assert!(p_neg[P_CLIP_RADIUS] >= 0.0);
            let min_neg = dist
                + length(scene.camera.target)
                + length(scene.object.offset)
                + p_neg[P_CLIP_RADIUS];
            assert!(p_neg[P_MAX_DISTANCE] + 1.0e-4 >= min_neg);
        }

        let mut klein = Scene::preset_fractal(Formula::Kleinian(Kleinian {
            a: 0.0,
            b: 0.0,
            bound_radius: 0.0,
        }));
        assert!(klein.pack(32, 32).iter().all(|v| v.is_finite()));
        klein.set_formula(Formula::Kleinian(Kleinian {
            a: 0.0,
            b: 0.0,
            ..Kleinian::PRESET
        }));
        assert!(klein.pack(32, 32).iter().all(|v| v.is_finite()));
        klein.set_formula(Formula::Kifs(Kifs {
            kind: KifsKind::Menger,
            scale: 0.0,
            offset: [0.0, 0.0, -1.0],
            rotation_degrees: [0.0; 3],
        }));
        assert!(klein.pack(32, 32).iter().all(|v| v.is_finite()));

        let neg_hybrid = Scene::preset_fractal(Formula::Hybrid(Hybrid {
            bailout: -5.0,
            ..Hybrid::PRESET
        }));
        let p_hybrid = neg_hybrid.pack(32, 32);
        assert!(p_hybrid.iter().all(|v| v.is_finite()));
        assert_eq!(p_hybrid[P_BAILOUT], 0.0);
        assert!(p_hybrid[P_CLIP_RADIUS] >= 0.0);
    }
}
