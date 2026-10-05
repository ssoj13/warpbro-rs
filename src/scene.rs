//! Scene description (serialisable), presets and packing into the kernel parameter block.
//! Formula parameters, presets and the per-render derived constants follow ofx-rs
//! `ofx-fractal/src/fractal3d.rs` (BulbParams::frame, BoxParams::squared_radii, QuatParams::slice,
//! KifsParams::frame, KleinianParams frame, Scene3d presets / max_distance / footprint).

use serde::{Deserialize, Serialize};

use crate::palette::PaletteScheme;
use crate::params::*;

// =============================================================================
// formulas
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
impl KifsKind {
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
/// fractal3d.rs HIT_EPSILON_PER_PIXEL: a footprint is hit_epsilon / this of a pixel.
const HIT_EPSILON_PER_PIXEL: f32 = 0.008;

impl Formula {
    pub const NAMES: [&'static str; 8] = [
        "Mandelbulb",
        "Mandelbox",
        "Quaternion Julia",
        "KIFS",
        "Kleinian",
        "Pseudo-Kleinian",
        "Apollonian",
        "Hybrid",
    ];

    pub fn code(&self) -> u32 {
        match self {
            Self::Mandelbulb(_) => FAMILY_BULB,
            Self::Mandelbox(_) => FAMILY_BOX,
            Self::QuaternionJulia(_) => FAMILY_QUAT,
            Self::Kifs(_) => FAMILY_KIFS,
            Self::Kleinian(_) => FAMILY_KLEINIAN,
            Self::PseudoKleinian(_) => FAMILY_PSEUDO_KLEINIAN,
            Self::Apollonian(_) => FAMILY_APOLLONIAN,
            Self::Hybrid(_) => FAMILY_HYBRID,
        }
    }

    pub fn name(&self) -> &'static str {
        Self::NAMES[self.code() as usize]
    }

    /// Formula3d::framing_radius: the radius the camera distance is measured in.
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
        }
    }

    pub fn bound_radius(&self) -> f32 {
        match self {
            Self::Kleinian(k) => k.bound_radius,
            Self::PseudoKleinian(k) => k.bound_radius,
            Self::Apollonian(k) => k.bound_radius,
            _ => 0.0,
        }
    }

    pub fn supports_julia(&self) -> bool {
        matches!(self, Self::Mandelbulb(_) | Self::Mandelbox(_))
    }

    /// The default formula of each family (index = family code).
    pub fn preset(code: u32) -> Self {
        match code {
            FAMILY_BULB => Self::Mandelbulb(Bulb::PRESET),
            FAMILY_BOX => Self::Mandelbox(MandelBox::PRESET),
            FAMILY_QUAT => Self::QuaternionJulia(Quat::PRESET),
            FAMILY_KIFS => Self::Kifs(Kifs::preset(KifsKind::Tetrahedron)),
            FAMILY_KLEINIAN => Self::Kleinian(Kleinian::PRESET),
            FAMILY_PSEUDO_KLEINIAN => Self::PseudoKleinian(PseudoKleinian::PRESET),
            FAMILY_APOLLONIAN => Self::Apollonian(Apollonian::PRESET),
            _ => Self::Hybrid(Hybrid::PRESET),
        }
    }
}

// =============================================================================
// scene
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Camera {
    pub target: [f32; 3],
    pub yaw_degrees: f32,
    pub pitch_degrees: f32,
    #[serde(default)]
    pub roll_degrees: f32,
    #[serde(default)]
    pub free_flight: bool,
    /// In framing radii of the formula.
    pub distance: f32,
    pub fov_y_degrees: f32,
    pub aperture: f32,
    /// In framing radii; 0 = focus on the target.
    pub focus_distance: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObjectTransform {
    pub offset: [f32; 3],
    pub rotation_degrees: [f32; 3],
    pub scale: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Lighting {
    pub sun_azimuth: f32,
    pub sun_elevation: f32,
    pub sun_color: [f32; 3],
    pub sun_intensity: f32,
    pub sun_angle: f32,
    pub sky_intensity: f32,
    pub sky_horizon: [f32; 3],
    pub sky_zenith: [f32; 3],
    /// Show the sky behind the fractal (otherwise only its light).
    pub background: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaterialModel {
    Fast,
    StandardSurface,
}

/// Where the base colour comes from: the fractal's palette (escape / trap colouring) or a
/// solid material colour (the material library).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColorSource {
    #[default]
    Palette,
    Material,
}

/// usd-rs pt-material-ext facing mix: the look blends from the material (head-on) toward
/// material B by `pow(1 - |N.V|, exponent)` (pearlescent / falloff).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Facing {
    pub color: [f32; 3],
    pub roughness: f32,
    pub metallic: f32,
    pub exponent: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Material {
    pub model: MaterialModel,
    #[serde(default)]
    pub color_source: ColorSource,
    #[serde(default = "default_base_color")]
    pub base_color: [f32; 3],
    #[serde(default)]
    pub facing: Option<Facing>,
    /// The library preset this material came from (UI label only).
    #[serde(default)]
    pub preset: Option<String>,
    pub base: f32,
    pub base_tint: [f32; 3],
    pub diffuse_roughness: f32,
    pub metalness: f32,
    pub specular: f32,
    pub specular_color: [f32; 3],
    pub specular_roughness: f32,
    pub specular_ior: f32,
    pub specular_anisotropy: f32,
    pub specular_rotation: f32,
    /// Fraction of dielectric base energy that refracts rather than diffuses.
    #[serde(default)]
    pub transmission: f32,
    /// Interface tint when depth is zero; Beer-Lambert transmittance at depth otherwise.
    #[serde(default = "default_transmission_color")]
    pub transmission_color: [f32; 3],
    #[serde(default)]
    pub transmission_extra_roughness: f32,
    /// World-space reference distance for absorption. Zero disables volume absorption.
    #[serde(default)]
    pub transmission_depth: f32,
    pub sheen: f32,
    pub sheen_color: [f32; 3],
    pub sheen_roughness: f32,
    pub coat: f32,
    pub coat_color: [f32; 3],
    pub coat_roughness: f32,
    pub coat_ior: f32,
    pub coat_affect_color: f32,
    pub coat_affect_roughness: f32,
    pub thin_film_thickness: f32,
    pub thin_film_ior: f32,
    pub emission: f32,
    pub emission_color: [f32; 3],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Coloring {
    Radius,
    TrapOrigin,
    TrapPlane,
    TrapPoint,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Render {
    pub iterations: u32,
    pub max_steps: u32,
    pub hit_epsilon: f32,
    /// Sphere-tracing step as a fraction of the estimate (ofx-fractal: 0.5; 0.85 measured
    /// unbiased on the presets at 1.5x the speed).
    pub step_factor: f32,
    pub max_bounces: u32,
    pub exposure_stops: f32,
    pub saturation: f32,
    pub reinhard: bool,
    /// OIDN works on neutral scene-linear samples before display transforms.
    #[serde(default)]
    pub denoise: crate::denoise::Settings,
    #[serde(default)]
    pub adaptive: Adaptive,
}

/// Adaptive sampling (V-Ray noise threshold / Redshift adaptive error): 8x4 pixel tiles stop
/// receiving samples once their noise is below the threshold; the sample target stays the maximum.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Adaptive {
    pub enabled: bool,
    /// Relative standard error of a pixel's mean luminance, `se / sqrt(mean)`.
    pub noise_threshold: f32,
    /// Samples every pixel receives before its tile may stop.
    pub min_samples: u32,
}
impl Default for Adaptive {
    fn default() -> Self {
        Self { enabled: true, noise_threshold: 0.01, min_samples: 16 }
    }
}
impl Adaptive {
    /// Fewest samples a variance estimate is trusted from. One sample has zero variance, so
    /// a minimum of 1 stopped every tile after the first sample; Cycles floors the user value
    /// at 4 the same way (Cycles `scene/integrator.cpp`: `max(4, adaptive_min_samples)`).
    pub const MIN_SAMPLES_FLOOR: u32 = 4;
    /// The minimum the kernel and the scheduler use.
    pub fn min(&self) -> u32 {
        self.min_samples.max(Self::MIN_SAMPLES_FLOOR)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Scene {
    /// Evaluated world data; the editor owns the authoring document.
    #[serde(skip)]
    pub world_render: bool,
    #[serde(skip)]
    pub camera_reference: Option<f32>,
    #[serde(skip)]
    pub objects: Vec<Scene>,
    #[serde(skip)]
    pub lights: Vec<Lighting>,
    #[serde(skip)]
    pub object_world: Option<[[f32; 4]; 4]>,
    /// Frozen document carried only by exports/bookmark entries, never CUDA requests.
    #[serde(skip)]
    pub document: Option<Box<crate::world::WorldDocument>>,
    #[serde(default)]
    pub animation: crate::animation::Animation,
    #[serde(default)]
    pub environment: crate::environment::Environment,
    pub name: String,
    pub formula: Formula,
    pub julia: Option<[f32; 3]>,
    pub object: ObjectTransform,
    pub camera: Camera,
    pub lighting: Lighting,
    pub material: Material,
    pub palette: PaletteScheme,
    pub coloring: Coloring,
    pub trap_point: [f32; 3],
    pub trap_axis: u32,
    pub trap_scale: f32,
    pub render: Render,
    #[serde(default = "crate::color::default_selection")]
    pub colour: crate::ocio::Sel,
}

impl Camera {
    pub fn orientation(&self) -> glam::Quat {
        glam::Quat::from_euler(
            glam::EulerRot::YXZ,
            self.yaw_degrees.to_radians(),
            -self.pitch_degrees.to_radians(),
            self.roll_degrees.to_radians(),
        )
    }
    /// Aim along `forward` (yaw / pitch), keeping the roll.
    pub fn set_forward(&mut self, forward: glam::Vec3) {
        let f = forward.normalize_or_zero();
        // forward = Ry(yaw) Rx(-pitch) (-Z) = (-cos p sin y, -sin p, -cos p cos y).
        self.yaw_degrees = (-f.x).atan2(-f.z).to_degrees();
        self.pitch_degrees = -f.y.clamp(-1.0, 1.0).asin().to_degrees();
    }
    /// ofx-fractal default_camera: eye on +Z, level, CAMERA_DEFAULT_DISTANCE framing radii.
    pub const fn default_fov(fov: f32) -> Self {
        Self {
            target: [0.0; 3],
            yaw_degrees: 0.0,
            pitch_degrees: 0.0,
            roll_degrees: 0.0,
            free_flight: false,
            distance: 2.5318,
            fov_y_degrees: fov,
            aperture: 0.0,
            focus_distance: 0.0,
        }
    }
}

impl Default for Lighting {
    /// ofx-fractal Lighting3d::default.
    fn default() -> Self {
        let d = normalize([0.56, 0.76, 0.36]);
        Self {
            sun_azimuth: d[0].atan2(d[2]).to_degrees(),
            sun_elevation: d[1].asin().to_degrees(),
            sun_color: [1.0; 3],
            sun_intensity: 2.0,
            sun_angle: 0.53,
            sky_intensity: 8.0,
            sky_horizon: [0.012, 0.02, 0.045],
            sky_zenith: [0.047, 0.075, 0.14],
            background: true,
        }
    }
}

fn default_transmission_color() -> [f32; 3] {
    [1.0; 3]
}

fn default_base_color() -> [f32; 3] {
    [0.8, 0.8, 0.8]
}

impl Default for Material {
    /// ofx-fractal Surface3d::default (MaterialX defaults, base_tint 1), fast model.
    fn default() -> Self {
        Self {
            model: MaterialModel::Fast,
            color_source: ColorSource::Palette,
            base_color: default_base_color(),
            facing: None,
            preset: None,
            base: 1.0,
            base_tint: [1.0; 3],
            diffuse_roughness: 0.0,
            metalness: 0.0,
            specular: 1.0,
            specular_color: [1.0; 3],
            specular_roughness: 0.2,
            specular_ior: 1.5,
            specular_anisotropy: 0.0,
            specular_rotation: 0.0,
            transmission: 0.0,
            transmission_color: [1.0; 3],
            transmission_extra_roughness: 0.0,
            transmission_depth: 0.0,
            sheen: 0.0,
            sheen_color: [1.0; 3],
            sheen_roughness: 0.3,
            coat: 0.0,
            coat_color: [1.0; 3],
            coat_roughness: 0.1,
            coat_ior: 1.5,
            coat_affect_color: 0.0,
            coat_affect_roughness: 0.0,
            thin_film_thickness: 0.0,
            thin_film_ior: 1.5,
            emission: 0.0,
            emission_color: [1.0; 3],
        }
    }
}

impl Scene {
    /// World-space bounds: use the DE's explicit bounds, otherwise a local 10³ box.
    pub fn framing_bounds(&self) -> (glam::Vec3, glam::Vec3) {
        use glam::Vec3;
        if self.world_render {
            let mut lower = Vec3::splat(f32::INFINITY);
            let mut upper = Vec3::splat(f32::NEG_INFINITY);
            for object in &self.objects {
                let matrix = object
                    .object_world
                    .map(|m| glam::Mat4::from_cols_array_2d(&m))
                    .unwrap_or_else(|| {
                        let rotation = glam::Mat3::from_cols_array_2d(&transpose(euler_matrix(
                            object.object.rotation_degrees,
                        )));
                        glam::Mat4::from_scale_rotation_translation(
                            Vec3::splat(object.object.scale),
                            glam::Quat::from_mat3(&rotation),
                            Vec3::from_array(object.object.offset),
                        )
                    });
                let mut local = object.clone();
                local.object = ObjectTransform {
                    offset: [0.0; 3],
                    rotation_degrees: [0.0; 3],
                    scale: 1.0,
                };
                let radius = local.pack(1, 1)[P_CLIP_RADIUS];
                let columns = matrix.to_cols_array_2d();
                let half = Vec3::from_array(std::array::from_fn(|axis| {
                    radius
                        * (0..3)
                            .map(|col| columns[col][axis] * columns[col][axis])
                            .sum::<f32>()
                            .sqrt()
                }));
                let center = matrix.transform_point3(Vec3::ZERO);
                lower = lower.min(center - half);
                upper = upper.max(center + half);
            }
            return if self.objects.is_empty() {
                (Vec3::splat(-1.0), Vec3::splat(1.0))
            } else {
                (lower, upper)
            };
        }
        let half_extent =
            match self.formula {
                Formula::Kifs(k) if k.kind != KifsKind::Menger => {
                    Vec3::splat(k.kind.bounding_radius() * self.object.scale)
                }
                Formula::Mandelbulb(_) | Formula::QuaternionJulia(_) | Formula::Hybrid(_) => {
                    Vec3::splat(self.pack(1, 1)[P_CLIP_RADIUS])
                }
                _ if self.formula.bound_radius() > 0.0 => {
                    Vec3::splat(self.formula.bound_radius() * self.object.scale)
                }
                _ => {
                    let half = if matches!(self.formula, Formula::Kifs(_)) {
                        1.0
                    } else {
                        5.0
                    };
                    let rot = euler_matrix(self.object.rotation_degrees);
                    Vec3::from_array(rot.map(|row| {
                        row.iter().map(|v| v.abs()).sum::<f32>() * half * self.object.scale
                    }))
                }
            };
        let center = Vec3::from_array(self.object.offset);
        (center - half_extent, center + half_extent)
    }

    fn base(
        name: &str,
        formula: Formula,
        fov: f32,
        iterations: u32,
        max_steps: u32,
        hit_epsilon: f32,
        palette: PaletteScheme,
    ) -> Self {
        Self {
            world_render: false,
            camera_reference: None,
            objects: Vec::new(),
            lights: Vec::new(),
            object_world: None,
            document: None,
            animation: Default::default(),
            environment: Default::default(),
            name: name.into(),
            colour: crate::color::default_selection(),
            formula,
            julia: None,
            object: ObjectTransform {
                offset: [0.0; 3],
                rotation_degrees: [0.0; 3],
                scale: 1.0,
            },
            camera: Camera::default_fov(fov),
            lighting: Lighting::default(),
            material: Material::default(),
            palette,
            coloring: Coloring::Radius,
            trap_point: [0.0; 3],
            trap_axis: 1,
            trap_scale: 1.5,
            render: Render {
                iterations,
                max_steps,
                hit_epsilon,
                step_factor: 0.85,
                max_bounces: 6,
                exposure_stops: -1.0,
                saturation: 1.0,
                reinhard: false,
                denoise: crate::denoise::Settings::default(),
                adaptive: Adaptive::default(),
            },
        }
    }

    /// The family preset (Scene3d::mandelbulb / mandelbox / ... / hybrid).
    pub fn preset(code: u32) -> Self {
        use PaletteScheme as P;
        let f = Formula::preset(code);
        let name = f.name();
        let mut s = match code {
            FAMILY_BULB => Self::base(name, f, 38.0, 20, 256, 0.001, P::Classic),
            FAMILY_BOX => Self::base(name, f, 35.0, 12, 256, 0.003, P::Ice),
            FAMILY_QUAT => Self::base(name, f, 38.0, 16, 256, 0.001, P::Amethyst),
            FAMILY_KIFS => Self::base(name, f, 38.0, 12, 128, 0.0005, P::Copper),
            FAMILY_KLEINIAN => Self::base(name, f, 38.0, 16, 256, 0.001, P::Ocean),
            FAMILY_PSEUDO_KLEINIAN => Self::base(name, f, 38.0, 6, 256, 0.001, P::Ember),
            FAMILY_APOLLONIAN => Self::base(name, f, 38.0, 8, 256, 0.001, P::Twilight),
            _ => Self::base(name, f, 38.0, 12, 256, 0.001, P::Aurora),
        };
        // A three-quarter view a little further out than ofx-fractal's neutral default reads
        // better in the gallery.
        s.camera.yaw_degrees = 35.0;
        s.camera.pitch_degrees = 20.0;
        s.camera.distance = 3.1;
        s
    }

    /// The built-in gallery: the eight family presets plus variations.
    pub fn gallery() -> Vec<Scene> {
        use PaletteScheme as P;
        let mut v: Vec<Scene> = (0..8).map(Self::preset).collect();

        let mut s = Self::preset(FAMILY_BULB);
        s.name = "Bulb · power 3".into();
        if let Formula::Mandelbulb(b) = &mut s.formula {
            b.power = 3.0;
        }
        s.palette = P::Sunset;
        v.push(s);

        let mut s = Self::preset(FAMILY_BULB);
        s.name = "Bulb · power 12, gold".into();
        if let Formula::Mandelbulb(b) = &mut s.formula {
            b.power = 12.0;
        }
        s.palette = P::Copper;
        s.material.metalness = 1.0;
        s.material.specular_roughness = 0.25;
        s.material.model = MaterialModel::StandardSurface;
        v.push(s);

        let mut s = Self::preset(FAMILY_BULB);
        s.name = "Bulb · Julia".into();
        s.julia = Some([0.35, 0.35, -0.4]);
        s.camera.distance = 3.4;
        s.palette = P::Aurora;
        s.coloring = Coloring::TrapOrigin;
        v.push(s);

        let mut s = Self::preset(FAMILY_BULB);
        s.name = "Bulb · twisted".into();
        if let Formula::Mandelbulb(b) = &mut s.formula {
            b.angle_phase_degrees = [40.0, 0.0];
            b.rotation_degrees = [0.0, 25.0, 0.0];
        }
        s.palette = P::Neon;
        v.push(s);

        let mut s = Self::preset(FAMILY_BOX);
        s.name = "Mandelbox · scale −1.5".into();
        if let Formula::Mandelbox(b) = &mut s.formula {
            b.scale = -1.5;
        }
        s.camera.distance = 1.6;
        s.palette = P::RoseGold;
        v.push(s);

        let mut s = Self::preset(FAMILY_KIFS);
        s.name = "Menger sponge".into();
        s.formula = Formula::Kifs(Kifs::preset(KifsKind::Menger));
        s.palette = P::Mono;
        s.material.model = MaterialModel::StandardSurface;
        s.material.coat = 1.0;
        v.push(s);

        let mut s = Self::preset(FAMILY_KIFS);
        s.name = "Octahedron KIFS".into();
        s.formula = Formula::Kifs(Kifs {
            rotation_degrees: [12.0, 0.0, 8.0],
            ..Kifs::preset(KifsKind::Octahedron)
        });
        s.palette = P::Verdant;
        v.push(s);

        let mut s = Self::preset(FAMILY_QUAT);
        s.name = "Quaternion · glass-coat".into();
        if let Formula::QuaternionJulia(q) = &mut s.formula {
            q.constant = [-0.291, -0.399, 0.339, 0.437];
        }
        s.material.model = MaterialModel::StandardSurface;
        s.material.coat = 1.0;
        s.material.thin_film_thickness = 450.0;
        s.palette = P::Ice;
        v.push(s);

        let mut s = Self::preset(FAMILY_HYBRID);
        s.name = "Hybrid · bulb + KIFS".into();
        if let Formula::Hybrid(h) = &mut s.formula {
            h.steps = [
                HybridStep::Mandelbulb,
                HybridStep::KifsFold,
                HybridStep::Off,
                HybridStep::Off,
            ];
        }
        s.palette = P::Fire;
        v.push(s);

        let mut s = Self::preset(FAMILY_PSEUDO_KLEINIAN);
        s.name = "Pseudo-Kleinian · cave".into();
        s.camera.distance = 0.6;
        s.camera.fov_y_degrees = 60.0;
        s.lighting.sky_intensity = 12.0;
        v.push(s);

        v
    }

    /// Pack this scene into the kernel parameter block for a `w` x `h` frame.
    pub fn pack(&self, w: u32, h: u32) -> Vec<f32> {
        let mut p = vec![0.0f32; P_COUNT];
        let radius = self.formula.framing_radius();
        let c = &self.camera;
        let camera_radius = self.camera_reference.unwrap_or(radius);

        // --- camera (ofx-gen OrbitCamera: eye on +Z at yaw = pitch = 0, around the target)
        let dist = c.distance * camera_radius;
        let orientation = c.orientation();
        let fwd = (orientation * -glam::Vec3::Z).to_array();
        let right = (orientation * glam::Vec3::X).to_array();
        let up = (orientation * glam::Vec3::Y).to_array();
        let eye =
            (glam::Vec3::from_array(c.target) - glam::Vec3::from_array(fwd) * dist).to_array();
        let half_h = (c.fov_y_degrees.to_radians() * 0.5).tan();
        let half_w = half_h * w as f32 / h as f32;
        p[P_WIDTH] = w as f32;
        p[P_HEIGHT] = h as f32;
        put3(&mut p, P_CAM_ORIGIN, eye);
        put3(&mut p, P_CAM_FORWARD, fwd);
        put3(&mut p, P_CAM_RIGHT, right);
        put3(&mut p, P_CAM_UP, up);
        p[P_HALF_W] = half_w;
        p[P_HALF_H] = half_h;
        p[P_APERTURE] = c.aperture * camera_radius * 0.05;
        p[P_FOCUS_DISTANCE] = if c.focus_distance > 0.0 {
            c.focus_distance * camera_radius
        } else {
            dist
        };

        // --- march (Scene3d::max_distance, pixel_footprint, sample_cone)
        let r = &self.render;
        let object_radius =
            (REACH_MARGIN_FRAMES * radius).max(self.formula.bound_radius()) * self.object.scale;
        let object_radius = match (self.formula, self.julia) {
            (Formula::Mandelbulb(b), Some(j)) => {
                object_radius.max(b.bailout.max(length(j)) * self.object.scale)
            }
            _ => object_radius,
        };
        p[P_MAX_DISTANCE] = dist + length(c.target) + length(self.object.offset) + object_radius;
        // Bounding sphere (see gpu.rs march): exact where the estimate itself says "outside".
        let clip_object = match self.formula {
            Formula::Mandelbulb(_) | Formula::QuaternionJulia(_) | Formula::Hybrid(_) => None, // set below from P_BAILOUT
            _ if self.formula.bound_radius() > 0.0 => Some(self.formula.bound_radius()),
            _ => Some(REACH_MARGIN_FRAMES * radius),
        };
        put3(&mut p, P_CLIP_CENTER, self.object.offset);
        p[P_MAX_STEPS] = r.max_steps as f32;
        let pixel = 2.0 * half_h / h as f32;
        let hit_epsilon = r.hit_epsilon;
        p[P_FOOTPRINT] = pixel * hit_epsilon / HIT_EPSILON_PER_PIXEL;
        p[P_SAMPLE_CONE] = (HIT_EPSILON_PER_PIXEL / hit_epsilon).max(1.0);
        p[P_MAX_BOUNCES] = r.max_bounces as f32;
        p[P_SECONDARY_STEPS] = r.max_steps as f32;
        p[P_SECONDARY_EPS] = 1.0;
        p[P_STEP_FACTOR] = r.step_factor;
        p[P_MATERIAL_MODEL] = (self.material.model == MaterialModel::StandardSurface
            || self.material.transmission > 0.0) as u32 as f32;

        // --- formula: common
        p[P_FAMILY] = self.formula.code() as f32;
        p[P_ITERATIONS] = r.iterations as f32;
        let julia_ok = self.formula.supports_julia() && self.julia.is_some();
        p[P_JULIA] = julia_ok as u32 as f32;
        put3(&mut p, P_JULIA_C, self.julia.unwrap_or([0.0; 3]));
        // ObjectFrame: rows of R^T so mat3(P_OBJ_AXES, x) = R^T x.
        let rot = euler_matrix(self.object.rotation_degrees);
        let rt = transpose(rot);
        put9(&mut p, P_OBJ_AXES, rt);
        put3(&mut p, P_OBJ_OFFSET, self.object.offset);
        p[P_OBJ_SCALE] = self.object.scale;
        p[P_BOUND_RADIUS] = self.formula.bound_radius();

        let set_rotation = |p: &mut Vec<f32>, degrees: [f32; 3]| {
            let reduced = degrees.map(|a| a % 360.0);
            if reduced.iter().all(|&a| a == 0.0) {
                p[P_ITER_ROTATE] = 0.0;
            } else {
                p[P_ITER_ROTATE] = 1.0;
                put9(p, P_ITER_ROT, euler_matrix(reduced));
            }
        };
        let pack_bulb = |p: &mut Vec<f32>, b: &Bulb| {
            p[P_BULB_POWER] = b.power;
            p[P_BULB_THETA_POWER] = b.angle_scale[0] * b.power;
            p[P_BULB_PHI_POWER] = b.angle_scale[1] * b.power;
            p[P_BULB_THETA_PHASE] = b.angle_phase_degrees[0].to_radians();
            p[P_BULB_PHI_PHASE] = b.angle_phase_degrees[1].to_radians();
            p[P_BULB_GROWTH] = 1.0f32
                .max(b.angle_scale[0].abs())
                .max(b.angle_scale[1].abs());
        };
        let pack_box = |p: &mut Vec<f32>, b: &MandelBox| {
            let min_r = b.min_radius_ratio * b.fixed_radius;
            p[P_BOX_SCALE] = b.scale;
            p[P_BOX_FOLD] = b.fold_limit;
            p[P_BOX_MIN_R2] = min_r * min_r;
            p[P_BOX_FIXED_R2] = b.fixed_radius * b.fixed_radius;
        };
        let pack_kifs = |p: &mut Vec<f32>, k: &Kifs| {
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

        match &self.formula {
            Formula::Mandelbulb(b) => {
                pack_bulb(&mut p, b);
                set_rotation(&mut p, b.rotation_degrees);
                let escape = if julia_ok {
                    b.bailout.max(length(self.julia.unwrap_or([0.0; 3])))
                } else {
                    b.bailout
                };
                p[P_BAILOUT] = escape;
                p[P_BULB_FAST8] = (b.power == 8.0
                    && b.angle_scale == [1.0, 1.0]
                    && b.angle_phase_degrees == [0.0, 0.0]
                    && p[P_ITER_ROTATE] == 0.0) as u32 as f32;
            }
            Formula::Mandelbox(b) => {
                pack_box(&mut p, b);
                set_rotation(&mut p, b.rotation_degrees);
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
                pack_kifs(&mut p, k);
                set_rotation(&mut p, k.rotation_degrees);
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
                put3(&mut p, P_PK_BOX, k.box_size);
                p[P_PK_SIZE] = k.size;
                put3(&mut p, P_PK_C, k.c);
                put3(&mut p, P_PK_OFFSET, k.offset);
                p[P_PK_THICKNESS] = k.thickness;
            }
            Formula::Apollonian(a) => {
                p[P_APOLLO_SCALE] = a.scale;
            }
            Formula::Hybrid(h) => {
                pack_bulb(&mut p, &h.bulb);
                pack_box(&mut p, &h.mandelbox);
                pack_kifs(&mut p, &h.kifs);
                p[P_APOLLO_SCALE] = h.apollonian_scale;
                p[P_BAILOUT] = h.bailout;
                set_rotation(&mut p, h.bulb.rotation_degrees);
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
        }

        p[P_CLIP_RADIUS] = match clip_object {
            Some(r) => r,
            // the escape radius, plus a hair for the f32 transform
            None => p[P_BAILOUT] * 1.001,
        } * self.object.scale;

        // --- colour
        p[P_COLOR_MODE] = self.coloring as u32 as f32;
        put3(&mut p, P_TRAP_POINT, self.trap_point);
        let mut axis = [0.0; 3];
        axis[self.trap_axis.min(2) as usize] = 1.0;
        put3(&mut p, P_TRAP_NORMAL, axis);
        p[P_TRAP_SCALE] = self.trap_scale;

        // --- light
        let l = &self.lighting;
        let (az, el) = (l.sun_azimuth.to_radians(), l.sun_elevation.to_radians());
        put3(
            &mut p,
            P_LIGHT_DIR,
            [el.cos() * az.sin(), el.sin(), el.cos() * az.cos()],
        );
        put_rgb(&mut p, P_LIGHT_COLOR, l.sun_color);
        p[P_LIGHT_INTENSITY] = l.sun_intensity;
        let half_angle = (l.sun_angle * 0.5).to_radians().max(1.0e-4);
        let omc = 1.0 - half_angle.cos();
        p[P_SUN_ONE_MINUS_COS] = omc;
        p[P_SUN_CONE_PDF] = 1.0 / (2.0 * std::f32::consts::PI * omc);
        p[P_SKY_INTENSITY] = l.sky_intensity;
        put_rgb(&mut p, P_SKY_HORIZON, l.sky_horizon);
        put_rgb(&mut p, P_SKY_ZENITH, l.sky_zenith);
        p[P_BACKGROUND] = l.background as u32 as f32;
        p[P_ENV_INTENSITY] = self.environment.intensity.max(0.0);
        p[P_ENV_ROTATION] = self.environment.rotation_degrees.to_radians();

        // --- material
        let m = &self.material;
        p[P_BASE] = m.base;
        put_rgb(&mut p, P_BASE_TINT, m.base_tint);
        p[P_DIFFUSE_ROUGHNESS] = m.diffuse_roughness;
        p[P_METALNESS] = m.metalness;
        p[P_SPECULAR] = m.specular;
        put_rgb(&mut p, P_SPECULAR_COLOR, m.specular_color);
        p[P_SPECULAR_ROUGHNESS] = m.specular_roughness;
        p[P_SPECULAR_IOR] = m.specular_ior;
        p[P_SPECULAR_ANISOTROPY] = m.specular_anisotropy;
        p[P_SPECULAR_ROTATION] = m.specular_rotation;
        p[P_TRANSMISSION] = m.transmission.clamp(0.0, 1.0);
        put_rgb(&mut p, P_TRANSMISSION_COLOR, m.transmission_color);
        p[P_TRANSMISSION_EXTRA_ROUGHNESS] = m.transmission_extra_roughness;
        p[P_TRANSMISSION_DEPTH] = m.transmission_depth;
        p[P_SHEEN] = m.sheen;
        put_rgb(&mut p, P_SHEEN_COLOR, m.sheen_color);
        p[P_SHEEN_ROUGHNESS] = m.sheen_roughness;
        p[P_COAT] = m.coat;
        put_rgb(&mut p, P_COAT_COLOR, m.coat_color);
        p[P_COAT_ROUGHNESS] = m.coat_roughness;
        p[P_COAT_IOR] = m.coat_ior;
        p[P_COAT_AFFECT_COLOR] = m.coat_affect_color;
        p[P_COAT_AFFECT_ROUGHNESS] = m.coat_affect_roughness;
        p[P_THIN_FILM_THICKNESS] = m.thin_film_thickness;
        p[P_THIN_FILM_IOR] = m.thin_film_ior;
        p[P_EMISSION] = m.emission;
        put_rgb(&mut p, P_EMISSION_COLOR, m.emission_color);
        p[P_COLOR_SOURCE] = (m.color_source == ColorSource::Material) as u32 as f32;
        put_rgb(&mut p, P_BASE_COLOR, m.base_color);
        if let Some(f) = m.facing {
            p[P_FACING_EXPONENT] = f.exponent;
            put_rgb(&mut p, P_FACING_COLOR, f.color);
            p[P_FACING_ROUGHNESS] = f.roughness;
            p[P_FACING_METALLIC] = f.metallic;
        }

        // --- tonemap
        p[P_EXPOSURE] = 2f32.powf(r.exposure_stops);
        p[P_SATURATION] = r.saturation;
        p[P_TONEMAP] = r.reinhard as u32 as f32;
        p[P_ADAPT_THRESHOLD] = r.adaptive.noise_threshold;
        p[P_ADAPT_MIN] = r.adaptive.min() as f32;
        p
    }
}

// =============================================================================
// math helpers (host)
// =============================================================================

pub fn normalize(a: [f32; 3]) -> [f32; 3] {
    let l = length(a);
    [a[0] / l, a[1] / l, a[2] / l]
}
pub fn length(a: [f32; 3]) -> f32 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}
fn put3(p: &mut [f32], i: usize, v: [f32; 3]) {
    p[i..i + 3].copy_from_slice(&v);
}
/// Authored colours are Rec.709; the kernels work in [`crate::color::WORKING`].
fn put_rgb(p: &mut [f32], i: usize, v: [f32; 3]) {
    put3(p, i, crate::color::to_working(v));
}
fn put9(p: &mut [f32], i: usize, m: [[f32; 3]; 3]) {
    for r in 0..3 {
        p[i + 3 * r..i + 3 * r + 3].copy_from_slice(&m[r]);
    }
}
fn transpose(m: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

/// fractal3d.rs euler_matrix: about world X, then Y, then Z (extrinsic), evaluated in f64.
fn euler_matrix(degrees: [f32; 3]) -> [[f32; 3]; 3] {
    let [(sx, cx), (sy, cy), (sz, cz)] = degrees.map(|a| (a as f64).to_radians().sin_cos());
    [
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy, cy * sx, cy * cx],
    ]
    .map(|row| row.map(|v| v as f32))
}

/// fractal3d.rs QuatParams::rotation_matrix: plane rotations of each axis with w, X then Y then Z.
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
