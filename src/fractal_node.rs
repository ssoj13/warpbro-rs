//! First-class `"Fractal"` scene node specification (`FractalNodeSpec`), 28-family
//! `Formula` definitions, typed `FractalSceneExt` accessors over `Scene::custom`,
//! and kernel parameter packing (`pack_object`).
//!
//! `soluwell-studio` has zero built-in knowledge of fractals or formulas. By registering
//! [`FractalNodeSpec`] via [`soluwell_studio::node_spec::register_node_type`], WarpBro turns
//! `"Fractal"` into a first-class renderable scene node in `soluwell-studio`. When a
//! `"Fractal"` node is present and visible in the graph, it is evaluated into `scene.objects`
//! and rendered; when absent or hidden, `soluwell-studio` renders the 3D studio environment.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{LazyLock, RwLock};

use fractal_core::params::*;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use soluwell_studio::node_spec::NodeTypeSpec;
use soluwell_studio::palette::PaletteScheme;
use soluwell_studio::render_profiles::Backend;
use soluwell_studio::scene::{
    Adaptive, Camera, Coloring, DEFAULT_GLASS_PROBES, DEFAULT_MAX_STEPS, Material, MaterialModel,
    ObjectTransform, Render, Scene, Sky, Sun, length,
};
use soluwell_studio::world::{AttrValue, Attrs, Slider, WorldAttribute};

use crate::formulas::*;

/// The `WorldKind` of a WarpBro `"Fractal"` node in `soluwell-studio`.
#[cfg(test)]
pub const FRACTAL_KIND: soluwell_studio::world::WorldKind =
    soluwell_studio::world::WorldKind::Custom("Fractal");

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
// Typed Scene Extension (`FractalSceneExt`)
// =============================================================================

pub fn default_fractal_custom_branches(formula: Formula) -> Map<String, Value> {
    let mut map = Map::new();
    map.insert(
        "formula".into(),
        serde_json::to_value(formula).expect("Formula serializes to JSON"),
    );
    map.insert("julia".into(), json!([0.0, 0.0, 0.0]));
    map.insert("julia_mix".into(), json!(0.0));
    map.insert("orbit_offset".into(), json!([0.0, 0.0, 0.0]));
    map.insert("inversion_strength".into(), json!(0.0));
    map.insert("inversion_center".into(), json!([0.0, 0.0, 0.0]));
    map.insert("inversion_radius".into(), json!(1.0));
    map.insert("trap_point".into(), json!([0.0, 0.0, 0.0]));
    map.insert("trap_axis".into(), json!(1u32));
    map.insert("trap_scale".into(), json!(1.5));
    map
}

fn base_fractal_scene(
    name: &str,
    formula: Formula,
    fov: f32,
    iterations: f32,
    hit_epsilon: f32,
    palette: PaletteScheme,
) -> Scene {
    Scene {
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
        colour: soluwell_studio::color::default_selection(),
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
        custom: default_fractal_custom_branches(formula)
            .into_iter()
            .collect(),
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
    }
}

pub trait FractalSceneExt {
    fn preset_fractal(f: Formula) -> Scene;
    fn classic_fractal_gallery() -> Vec<Scene>;
    fn fractal_gallery() -> Vec<Scene>;

    fn formula(&self) -> Formula;
    fn set_formula(&mut self, f: Formula);
    fn modify_formula(&mut self, edit: impl FnOnce(&mut Formula));

    fn julia(&self) -> [f32; 3];
    fn set_julia(&mut self, v: [f32; 3]);

    fn julia_mix(&self) -> f32;
    fn set_julia_mix(&mut self, v: f32);

    fn orbit_offset(&self) -> [f32; 3];
    #[cfg(test)]
    fn set_orbit_offset(&mut self, v: [f32; 3]);

    fn inversion_strength(&self) -> f32;
    #[cfg(test)]
    fn set_inversion_strength(&mut self, v: f32);

    fn inversion_center(&self) -> [f32; 3];
    #[cfg(test)]
    fn set_inversion_center(&mut self, v: [f32; 3]);

    fn inversion_radius(&self) -> f32;
    #[cfg(test)]
    fn set_inversion_radius(&mut self, v: f32);

    fn trap_point(&self) -> [f32; 3];
    #[cfg(test)]
    fn set_trap_point(&mut self, v: [f32; 3]);

    fn trap_axis(&self) -> u32;
    #[cfg(test)]
    fn set_trap_axis(&mut self, v: u32);

    fn trap_scale(&self) -> f32;
    fn set_trap_scale(&mut self, v: f32);
}

fn get_f32(custom: &BTreeMap<String, Value>, key: &str, default: f32) -> f32 {
    custom
        .get(key)
        .and_then(Value::as_f64)
        .map(|v| v as f32)
        .unwrap_or(default)
}

fn get_u32(custom: &BTreeMap<String, Value>, key: &str, default: u32) -> u32 {
    custom
        .get(key)
        .and_then(|v| v.as_u64().or_else(|| v.as_f64().map(|f| f as u64)))
        .map(|v| v as u32)
        .unwrap_or(default)
}

fn get_vec3(custom: &BTreeMap<String, Value>, key: &str, default: [f32; 3]) -> [f32; 3] {
    custom
        .get(key)
        .and_then(Value::as_array)
        .filter(|a| a.len() == 3)
        .map(|a| std::array::from_fn(|i| a[i].as_f64().unwrap_or(0.0) as f32))
        .unwrap_or(default)
}

impl FractalSceneExt for Scene {
    fn preset_fractal(f: Formula) -> Scene {
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
        };
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
        s
    }

    fn classic_fractal_gallery() -> Vec<Scene> {
        use PaletteScheme as P;
        let mut v: Vec<Scene> = Formula::PRESETS[..8]
            .iter()
            .map(|f| Self::preset_fractal(*f))
            .collect();

        let mut s = Self::preset_fractal(Formula::Mandelbulb(Bulb::PRESET));
        s.name = "Bulb · power 3".into();
        s.modify_formula(|f| {
            if let Formula::Mandelbulb(b) = f {
                b.power = 3.0;
            }
        });
        s.palette = P::Sunset;
        v.push(s);

        let mut s = Self::preset_fractal(Formula::Mandelbulb(Bulb::PRESET));
        s.name = "Bulb · power 12, gold".into();
        s.modify_formula(|f| {
            if let Formula::Mandelbulb(b) = f {
                b.power = 12.0;
            }
        });
        s.palette = P::Copper;
        s.material.metalness = 1.0;
        s.material.specular_roughness = 0.25;
        v.push(s);

        let mut s = Self::preset_fractal(Formula::Mandelbulb(Bulb::PRESET));
        s.name = "Bulb · Julia".into();
        s.set_julia([0.35, 0.35, -0.4]);
        s.set_julia_mix(1.0);
        s.camera.distance = 3.4;
        s.palette = P::Aurora;
        s.coloring = Coloring::TrapOrigin;
        v.push(s);

        let mut s = Self::preset_fractal(Formula::Mandelbulb(Bulb::PRESET));
        s.name = "Bulb · twisted".into();
        s.modify_formula(|f| {
            if let Formula::Mandelbulb(b) = f {
                b.angle_phase_degrees = [40.0, 0.0];
                b.rotation_degrees = [0.0, 25.0, 0.0];
            }
        });
        s.palette = P::Neon;
        v.push(s);

        let mut s = Self::preset_fractal(Formula::Mandelbox(MandelBox::PRESET));
        s.name = "Mandelbox · scale −1.5".into();
        s.modify_formula(|f| {
            if let Formula::Mandelbox(b) = f {
                b.scale = -1.5;
            }
        });
        s.camera.distance = 1.6;
        s.palette = P::RoseGold;
        v.push(s);

        let mut s = Self::preset_fractal(Formula::Kifs(Kifs::preset(KifsKind::Tetrahedron)));
        s.name = "Menger sponge".into();
        s.set_formula(Formula::Kifs(Kifs::preset(KifsKind::Menger)));
        s.palette = P::Mono;
        s.material.model = MaterialModel::StandardSurface;
        s.material.coat = 1.0;
        v.push(s);

        let mut s = Self::preset_fractal(Formula::Kifs(Kifs::preset(KifsKind::Tetrahedron)));
        s.name = "Octahedron KIFS".into();
        s.set_formula(Formula::Kifs(Kifs {
            rotation_degrees: [12.0, 0.0, 8.0],
            ..Kifs::preset(KifsKind::Octahedron)
        }));
        s.palette = P::Verdant;
        v.push(s);

        let mut s = Self::preset_fractal(Formula::QuaternionJulia(Quat::PRESET));
        s.name = "Quaternion · glass-coat".into();
        s.modify_formula(|f| {
            if let Formula::QuaternionJulia(q) = f {
                q.constant = [-0.291, -0.399, 0.339, 0.437];
            }
        });
        s.material.model = MaterialModel::StandardSurface;
        s.material.coat = 1.0;
        s.material.thin_film_thickness = 450.0;
        s.palette = P::Ice;
        v.push(s);

        let mut s = Self::preset_fractal(Formula::Hybrid(Hybrid::PRESET));
        s.name = "Hybrid · bulb + KIFS".into();
        s.modify_formula(|f| {
            if let Formula::Hybrid(h) = f {
                h.steps = [
                    HybridStep::Mandelbulb,
                    HybridStep::KifsFold,
                    HybridStep::Off,
                    HybridStep::Off,
                ];
            }
        });
        s.palette = P::Fire;
        v.push(s);

        let mut s = Self::preset_fractal(Formula::PseudoKleinian(PseudoKleinian::PRESET));
        s.name = "Pseudo-Kleinian · cave".into();
        s.camera.distance = 0.6;
        s.camera.fov_y_degrees = 60.0;
        s.sky.intensity = 12.0;
        v.push(s);

        v
    }

    fn fractal_gallery() -> Vec<Scene> {
        let mut v = Self::classic_fractal_gallery();
        v.extend(
            Formula::PRESETS[8..]
                .iter()
                .map(|f| Self::preset_fractal(*f)),
        );
        v
    }

    fn formula(&self) -> Formula {
        self.custom
            .get("formula")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or(Formula::Mandelbulb(Bulb::PRESET))
    }

    fn set_formula(&mut self, f: Formula) {
        self.custom.insert(
            "formula".into(),
            serde_json::to_value(f).expect("Formula serializes to JSON"),
        );
    }

    fn modify_formula(&mut self, edit: impl FnOnce(&mut Formula)) {
        let mut f = self.formula();
        edit(&mut f);
        self.set_formula(f);
    }

    fn julia(&self) -> [f32; 3] {
        get_vec3(&self.custom, "julia", [0.0; 3])
    }

    fn set_julia(&mut self, v: [f32; 3]) {
        self.custom.insert("julia".into(), json!(v));
    }

    fn julia_mix(&self) -> f32 {
        get_f32(&self.custom, "julia_mix", 0.0)
    }

    fn set_julia_mix(&mut self, v: f32) {
        self.custom.insert("julia_mix".into(), json!(v));
    }

    fn orbit_offset(&self) -> [f32; 3] {
        get_vec3(&self.custom, "orbit_offset", [0.0; 3])
    }

    #[cfg(test)]
    fn set_orbit_offset(&mut self, v: [f32; 3]) {
        self.custom.insert("orbit_offset".into(), json!(v));
    }

    fn inversion_strength(&self) -> f32 {
        get_f32(&self.custom, "inversion_strength", 0.0)
    }

    #[cfg(test)]
    fn set_inversion_strength(&mut self, v: f32) {
        self.custom.insert("inversion_strength".into(), json!(v));
    }

    fn inversion_center(&self) -> [f32; 3] {
        get_vec3(&self.custom, "inversion_center", [0.0; 3])
    }

    #[cfg(test)]
    fn set_inversion_center(&mut self, v: [f32; 3]) {
        self.custom.insert("inversion_center".into(), json!(v));
    }

    fn inversion_radius(&self) -> f32 {
        get_f32(&self.custom, "inversion_radius", 1.0)
    }

    #[cfg(test)]
    fn set_inversion_radius(&mut self, v: f32) {
        self.custom.insert("inversion_radius".into(), json!(v));
    }

    fn trap_point(&self) -> [f32; 3] {
        get_vec3(&self.custom, "trap_point", [0.0; 3])
    }

    #[cfg(test)]
    fn set_trap_point(&mut self, v: [f32; 3]) {
        self.custom.insert("trap_point".into(), json!(v));
    }

    fn trap_axis(&self) -> u32 {
        get_u32(&self.custom, "trap_axis", 1)
    }

    #[cfg(test)]
    fn set_trap_axis(&mut self, v: u32) {
        self.custom.insert("trap_axis".into(), json!(v));
    }

    fn trap_scale(&self) -> f32 {
        get_f32(&self.custom, "trap_scale", 1.5)
    }

    fn set_trap_scale(&mut self, v: f32) {
        self.custom.insert("trap_scale".into(), json!(v));
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

fn formula_hint(family: &str, field: &str) -> Option<&'static str> {
    Some(match (family, field) {
        ("Mandelbulb", "power") => {
            "Mandelbulb exponent: z -> z^power in spherical coordinates. 8 is the classic bulb; higher gives more, finer lobes."
        }
        ("Mandelbulb", "angle_scale") => {
            "Multiplies the power for the polar (theta) and azimuth (phi) angles separately; 1, 1 is the plain bulb. Stretches the lobes along one angle."
        }
        ("Mandelbulb", "angle_phase_degrees") => {
            "Offsets added to the polar and azimuth angles every iteration: twists the bulb."
        }
        ("Mandelbulb" | "QuaternionJulia" | "Hybrid", "bailout") => {
            "Escape radius: an orbit farther than this is outside. Larger is more exact near the surface and slower."
        }
        ("Mandelbulb" | "Mandelbox" | "Kifs", "rotation_degrees") => {
            "Rotation applied to the point every iteration: folds the structure into spirals. 0 keeps the formula symmetric."
        }
        ("Mandelbox", "scale") => {
            "Mandelbox scale per iteration, typically 2 to 3; negative values give the inverted box."
        }
        ("Mandelbox", "min_radius_ratio") => {
            "Inner radius of the sphere fold as a fraction of the fixed radius: points inside it are scaled up by the largest factor."
        }
        ("Mandelbox", "fixed_radius") => {
            "Radius of the sphere fold: points between the inner radius and this are inverted."
        }
        ("Mandelbox", "fold_limit") => "Box fold: coordinates beyond +/- this are reflected back.",
        ("Kifs", "kind") => {
            "The polyhedron whose symmetry planes fold space: tetrahedron, octahedron or Menger sponge."
        }
        ("Kifs", "scale") => "KIFS scale per iteration (above 1): how much smaller each copy is.",
        ("Kifs", "offset") => "Shifts the fold centre: the copies move apart or together.",
        ("QuaternionJulia", "constant") => {
            "The Julia constant c (x, y, z, w) of z -> z^2 + c: the shape of the set."
        }
        ("QuaternionJulia", "slice_w") => "The 4D set is cut by a 3D slice at this w.",
        ("QuaternionJulia", "rotation_degrees") => "Turns the 3D slice through the 4D set.",
        ("Kleinian", "a") => {
            "Kleinian group parameter a: the generator's translation, the size of the circle-packing cells."
        }
        ("Kleinian", "b") => {
            "Kleinian group parameter b: the generator's shear; 0 is the symmetric limit set."
        }
        ("PseudoKleinian", "box_size") => {
            "Half sizes of the box the point is folded into every iteration."
        }
        ("PseudoKleinian", "size") => "Sphere inversion radius squared: larger opens bigger holes.",
        ("PseudoKleinian", "c") => "Offset added after every inversion.",
        ("PseudoKleinian", "offset") => "Centre of the final shape test.",
        ("PseudoKleinian", "thickness") => "Thickness of the final shape: thicker fills the holes.",
        ("Apollonian", "scale") | ("Hybrid", "apollonian_scale") => {
            "Apollonian inversion strength: larger packs more, smaller spheres."
        }
        ("Kleinian" | "PseudoKleinian" | "Apollonian", "bound_radius") => {
            "Radius of the sphere that holds the whole set: rays only march inside it. Too small cuts the fractal off."
        }
        ("Hybrid", "steps") => {
            "The formulas applied in turn every iteration (Off skips a slot); their parameters are the Mandelbulb / Mandelbox / KIFS sub-sections."
        }
        _ => {
            return crate::formulas::find_formula_meta_by_variant(family)
                .and_then(|fm| fm.params.iter().find(|p| p.name == field))
                .map(|p| p.hint);
        }
    })
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

/// Interns the bounded set of dynamic inactive-reason messages (at most ~35 distinct strings over
/// the process lifetime: 28 formula names, 4 colouring modes, 4 hybrid step names).
fn intern_static_reason(s: String) -> &'static str {
    static INTERNED: LazyLock<RwLock<BTreeSet<&'static str>>> =
        LazyLock::new(|| RwLock::new(BTreeSet::new()));
    if let Ok(read) = INTERNED.read()
        && let Some(&existing) = read.get(s.as_str())
    {
        return existing;
    }
    let mut write = INTERNED.write().expect("Interned reason lock poisoned");
    if let Some(&existing) = write.get(s.as_str()) {
        return existing;
    }
    let leaked: &'static str = Box::leak(s.into_boxed_str());
    write.insert(leaked);
    leaked
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
        soluwell_config::icons::FRACTAL
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
        path == "/formula"
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
        if path == "/formula" {
            serde_json::from_value::<Formula>(value.clone())
                .map_err(|e| format!("Invalid formula: {e}"))?;
            return Ok(());
        }
        if let Some(choices) = self.attribute_choices(path) {
            let choices = choices?;
            if !choices.is_empty() && !choices.contains(value) {
                return Err(format!("Invalid enum value: {path}"));
            }
        }
        Ok(())
    }

    fn attribute_choices(&self, path: &str) -> Option<Result<Vec<Value>, String>> {
        if path == "/formula" {
            return Some(variants(Formula::PRESETS));
        }
        if let Some(choices) = crate::formulas::formula_param_choices(path) {
            return Some(Ok(choices));
        }
        match path {
            p if p.ends_with("/kind") && p.starts_with("/formula") => Some(variants(KifsKind::ALL)),
            p if p.starts_with("/formula/Hybrid/steps/") => Some(variants(HybridStep::ALL)),
            _ => None,
        }
    }

    fn choice_label(&self, path: &str, value: &Value) -> Option<String> {
        if path == "/formula" {
            return serde_json::from_value::<Formula>(value.clone())
                .map(|f| f.name().to_owned())
                .ok()
                .or_else(|| value.as_object()?.keys().next().cloned());
        }
        None
    }

    fn choice_hover(&self, path: &str, value: &Value) -> Option<String> {
        if path == "/formula" {
            let f: Formula = serde_json::from_value(value.clone()).ok()?;
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
            let (family, param) = match (family, param.split_once('/')) {
                ("Hybrid", Some(("bulb", p))) => ("Mandelbulb", p),
                ("Hybrid", Some(("mandelbox", p))) => ("Mandelbox", p),
                ("Hybrid", Some(("kifs", p))) => ("Kifs", p),
                _ => (family, param),
            };
            return crate::formulas::formula_param_slider(&format!("/formula/{family}/{param}"));
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
                "Julia constant J (Mandelbulb, Mandelbox): the orbit constant the Julia mix blends towards.",
            ),
            "/julia_mix" => Some(
                "Mandelbrot to Julia (Mandelbulb, Mandelbox): every iteration adds mix(point, Julia constant, this). 0 is the Mandelbrot form, 1 the Julia form; keys morph between them.",
            ),
            "/orbit_offset" => Some(
                "Constant added every iteration on top of the orbit constant (Mandelbulb, Mandelbox): z -> z^p + c + offset. Shifts and reshapes the set.",
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
            && crate::formulas::find_formula_meta_by_variant(variant).is_some_and(|m| m.code >= 8)
            && let Some(label) = crate::formulas::formula_param_label(path)
        {
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
        let under = |prefix: &str| {
            path.strip_prefix(prefix)
                .is_some_and(|rest| rest.is_empty() || rest.starts_with('/'))
        };
        if under("/julia") || under("/julia_mix") || under("/orbit_offset") {
            let formula = get("/formula")?;
            let family = formula.as_object()?.keys().next()?.clone();
            let has_orbit = crate::formulas::find_formula_meta_by_variant(&family)
                .map(|m| m.has_orbit_constant)
                .unwrap_or_else(|| matches!(family.as_str(), "Mandelbulb" | "Mandelbox"));
            if !has_orbit {
                return Some(Some(intern_static_reason(format!(
                    "The {family} formula has no orbit constant"
                ))));
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
            let used = match coloring {
                "Radius" => false,
                "TrapOrigin" => under("/trap_scale"),
                "TrapPoint" => !under("/trap_axis"),
                _ => true,
            };
            if !used {
                return Some(Some(intern_static_reason(format!(
                    "Not used by the {coloring} colouring"
                ))));
            }
            return Some(None);
        }
        if path == "/render/iterations" {
            let formula = get("/formula")?;
            let family = formula.as_object()?.keys().next()?;
            if let Some(meta) = crate::formulas::find_formula_meta_by_variant(family)
                && !meta.iterates
            {
                return Some(Some(intern_static_reason(format!(
                    "{} is analytical (no iteration loop)",
                    meta.display_name
                ))));
            }
            return Some(None);
        }
        let mut cache: HashMap<String, Value> = HashMap::new();
        for key in [
            "/formula/GyroidTpms/octaves",
            "/formula/VoronoiFoam/octaves",
            "/formula/TorusKnotBraid/braid_depth",
            "/formula/DupinCyclide/rib_frequency",
            "/formula/DupinCyclide/rib_depth",
            "/formula/PhyllotaxisShell/cone_height",
            "/formula/PhyllotaxisShell/sub_florets",
            "/formula/NautilusShell/rib_frequency",
            "/formula/AstrolabeGimbal/gear_teeth",
            "/formula/AstrolabeGimbal/tooth_depth",
        ] {
            if let Some(val) = get(key) {
                cache.insert(key.to_owned(), val);
            }
        }
        if let Some(reason) = crate::formulas::formula_param_inactive_reason(path, |k| cache.get(k))
        {
            return Some(Some(intern_static_reason(reason)));
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
            let step = match rest.split('/').next()? {
                "bulb" if hybrid_own("bulb/rotation_degrees") => return Some(None),
                "bulb" => "Mandelbulb",
                "mandelbox" => "Mandelbox",
                "kifs" => "KifsFold",
                "apollonian_scale" => "Inversion",
                _ => return Some(None),
            };
            let formula = get("/formula")?;
            let steps = formula.get("Hybrid")?.get("steps")?.as_array()?;
            let runs = steps.iter().any(|s| s.as_str() == Some(step))
                || (step == "Mandelbulb" && steps.iter().all(|s| s.as_str() == Some("Off")));
            if !runs {
                return Some(Some(intern_static_reason(format!(
                    "No hybrid step is {step}"
                ))));
            }
            return Some(None);
        }
        None
    }

    fn expand_attributes(&self, out: &mut Vec<WorldAttribute>) -> Result<(), String> {
        if let Some(steps) = out
            .iter()
            .find(|a| a.path == "/formula/Hybrid/steps")
            .map(|a| a.value.clone())
            .and_then(|v| v.as_array().cloned())
        {
            let choices = variants(HybridStep::ALL)?;
            for (slot, value) in steps.iter().enumerate() {
                let path = format!("/formula/Hybrid/steps/{slot}");
                let label = self
                    .attribute_label(&path)
                    .unwrap_or_else(|| format!("Step {}", slot + 1));
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
                    default: None,
                });
            }
        }
        Ok(())
    }

    fn sync_component_edit(&self, parent: &str, vector: &Value, attrs: &mut Attrs) {
        if parent != "/formula/Hybrid/steps" {
            return;
        }
        let Some(mut formula) = attrs.get("/formula").cloned().map(attr_json) else {
            return;
        };
        let Some(slots) = formula
            .pointer_mut("/Hybrid/steps")
            .and_then(Value::as_array_mut)
        else {
            return;
        };
        let Some(fresh) = vector.as_array() else {
            return;
        };
        if slots.len() != fresh.len() {
            return;
        }
        slots.clone_from_slice(fresh);
        attrs.set("/formula", to_attr(&formula));
    }

    fn framing_radius(&self, scene: &Scene) -> f32 {
        scene.formula().framing_radius()
    }

    fn bound_radius(&self, scene: &Scene) -> f32 {
        scene.formula().bound_radius()
    }

    fn local_pick_radius(&self, scene: &Scene) -> f32 {
        if scene.inversion_strength() != 0.0 {
            let mut local = scene.clone();
            local.object = ObjectTransform {
                offset: [0.0; 3],
                rotation_degrees: [0.0; 3],
                scale: 1.0,
            };
            local.pack(1, 1)[P_CLIP_RADIUS]
        } else {
            scene.formula().framing_radius()
        }
    }

    fn family_code(&self, scene: &Scene) -> u32 {
        scene.formula().code()
    }

    fn family_count(&self) -> u32 {
        Formula::COUNT
    }

    fn summary_label(&self, scene: &Scene) -> String {
        scene.formula().name().to_owned()
    }

    fn pack_object(&self, scene: &Scene, p: &mut [f32], camera_dist: f32) {
        let formula = scene.formula();
        let radius = formula.framing_radius();
        let julia = scene.julia();
        let julia_mix = scene.julia_mix();
        let orbit_offset = scene.orbit_offset();
        let inversion_strength = scene.inversion_strength();
        let inversion_center = scene.inversion_center();
        let inversion_radius = scene.inversion_radius();
        let trap_point = scene.trap_point();
        let trap_axis = scene.trap_axis();
        let trap_scale = scene.trap_scale();

        let bulb_escape = |b: &Bulb| -> f32 {
            b.bailout
                .max(julia_mix * length(julia) + length(orbit_offset))
        };

        let mut object_radius =
            (REACH_MARGIN_FRAMES * radius).max(formula.bound_radius()) * scene.object.scale;
        if let Formula::Mandelbulb(b) = &formula
            && (julia_mix != 0.0 || orbit_offset != [0.0; 3])
        {
            object_radius = object_radius.max(bulb_escape(b) * scene.object.scale);
        }

        let clip_object = match &formula {
            Formula::Mandelbulb(_)
            | Formula::QuaternionJulia(_)
            | Formula::Hybrid(_)
            | Formula::Bristorbrot(_)
            | Formula::BurningShip(_)
            | Formula::HarmonicBulb(_)
            | Formula::Mandelbar3D(_) => None,
            _ if formula.bound_radius() > 0.0 => Some(formula.bound_radius()),
            _ => Some(REACH_MARGIN_FRAMES * radius),
        };

        p[P_FAMILY] = formula.code() as f32;
        p[P_ITERATIONS] = scene.render.iterations;
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
            p[P_BOX_SCALE] = b.scale;
            p[P_BOX_FOLD] = b.fold_limit;
            p[P_BOX_MIN_R2] = min_r * min_r;
            p[P_BOX_FIXED_R2] = b.fixed_radius * b.fixed_radius;
        };
        let pack_kifs = |p: &mut [f32], k: &Kifs| {
            let s = k.scale;
            let base = k.kind.centre();
            let cc = [
                base[0] + k.offset[0],
                base[1] + k.offset[1],
                base[2] + k.offset[2],
            ];
            let (shift_z, fold_height) = match k.kind {
                KifsKind::Menger => (0.0, 0.5 * cc[2] * (s - 1.0) / s),
                _ => ((s - 1.0) * cc[2], 0.0),
            };
            p[P_KIFS_KIND] = k.kind.code() as f32;
            p[P_KIFS_SCALE] = s;
            put3(
                p,
                P_KIFS_SHIFT,
                [(s - 1.0) * cc[0], (s - 1.0) * cc[1], shift_z],
            );
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
                p[P_KLEIN_SKEW] = (b.abs() / a) as f32;
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
                p[P_BAILOUT] = h.bailout;
                set_rotation(p, h.bulb.rotation_degrees);
                let active: Vec<u32> = h
                    .steps
                    .iter()
                    .filter(|s| **s != HybridStep::Off)
                    .map(|s| match s {
                        HybridStep::Mandelbulb => 1,
                        HybridStep::Mandelbox => 2,
                        HybridStep::KifsFold => 3,
                        HybridStep::Inversion => 4,
                        HybridStep::Off => 0,
                    })
                    .collect();
                let active = if active.is_empty() { vec![1] } else { active };
                p[P_HYBRID_STEPS] = active
                    .iter()
                    .enumerate()
                    .fold(0u32, |acc, (i, c)| acc | (c << (3 * i)))
                    as f32;
                p[P_HYBRID_COUNT] = active.len() as f32;
            }
            other => {
                let orbit_escape = julia_mix * length(julia) + length(orbit_offset);
                crate::formulas::pack_extended_formula(p, other, orbit_escape, euler_matrix);
            }
        }

        let clip = match clip_object {
            Some(r) => r,
            None => p[P_BAILOUT] * 1.001,
        };
        let (clip, object_radius) = if inversion_strength == 0.0 {
            (clip, object_radius)
        } else {
            let clip = fractal_core::inversion::reach(
                inversion_center,
                inversion_radius,
                inversion_strength,
                clip,
            )
            .unwrap_or(INVERSION_FAR_REACH);
            (clip, object_radius.max(clip * scene.object.scale))
        };
        p[P_CLIP_RADIUS] = clip * scene.object.scale;
        p[P_MAX_DISTANCE] =
            camera_dist + length(scene.camera.target) + length(scene.object.offset) + object_radius;

        p[P_COLOR_MODE] = scene.coloring as u32 as f32;
        put3(p, P_TRAP_POINT, trap_point);
        let mut axis = [0.0; 3];
        axis[trap_axis.min(2) as usize] = 1.0;
        put3(p, P_TRAP_NORMAL, axis);
        p[P_TRAP_SCALE] = trap_scale;
    }

    fn payload_kind(&self) -> soluwell_studio::NodePayloadKind {
        soluwell_studio::NodePayloadKind::SdfField
    }
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
    use soluwell_studio::world::{WorldCommand, WorldDocument, WorldEditor};

    use super::*;

    #[test]
    fn fractal_node_lifecycle_and_parameter_roundtrip() {
        use soluwell_studio::StudioModule;
        crate::engine::init_warpbro();
        assert_eq!(
            FractalNodeSpec.payload_kind(),
            soluwell_studio::NodePayloadKind::SdfField
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
        let packed = snap.pack(64, 64);
        assert_eq!(packed[P_OBJECT_COUNT], 1.0);

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
        assert_eq!(hidden_snap.pack(64, 64)[P_OBJECT_COUNT], 0.0);

        editor
            .execute_edit(WorldCommand::Delete(fractal_id), None)
            .expect("delete Fractal node");
        let empty_snap = editor
            .document
            .snapshot(0.0)
            .expect("snapshot without Fractal");
        assert!(empty_snap.objects.is_empty());
        assert_eq!(empty_snap.pack(64, 64)[P_OBJECT_COUNT], 0.0);
    }
}
