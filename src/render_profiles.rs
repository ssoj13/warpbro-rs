//! Canonical settings-node contracts. These values are evaluated snapshots, never stored DTOs.
use crate::scene::Render;
use crate::world::NodeId;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CatalogRole {
    Profile,
    Template,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RenderMethod {
    Fast,
    Full,
}
/// Compute backend for fractal rendering: `Auto` uses CUDA when available and falls back to WGPU;
/// `Cuda` and `Wgpu` require that specific backend without silent fallback.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Backend {
    #[default]
    Auto,
    Cuda,
    Wgpu,
}
impl Backend {
    pub const ALL: [Self; 3] = [Self::Auto, Self::Cuda, Self::Wgpu];
    pub fn label(self) -> &'static str {
        match self {
            Self::Auto => "Auto",
            Self::Cuda => "CUDA",
            Self::Wgpu => "WGPU",
        }
    }
}
/// What a render profile shows of the fractals: their path-traced light, Direct shading (one
/// deterministic pass: key lights with soft shadows, occlusion, the sky's harmonics; no path
/// tracing), or their bounds alone (drawn by the viewport, nothing traced).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Display {
    #[default]
    PathTraced,
    Direct,
    Bounds,
}
/// The Direct display's shading (fractal-kernels `kernels/direct.rs`, OFX's Direct mode): a
/// soft shadow per light and five-probe ambient occlusion over the sky's harmonics. Defaults
/// are OFX's (`fractal3d` Lighting3d).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DirectShading {
    /// 0 (no shadow) .. 1 (black umbra).
    pub shadow_strength: f32,
    /// March steps of a shadow ray; 0 turns shadows off.
    pub shadow_steps: u32,
    /// 0 (none) .. 1 (full occlusion).
    pub ao_strength: f32,
    /// Probes per occlusion direction; 0 turns occlusion off.
    pub ao_steps: u32,
    /// Distance the occlusion probes reach, in scene units.
    pub ao_radius: f32,
}
impl Default for DirectShading {
    fn default() -> Self {
        Self {
            shadow_strength: 1.0,
            shadow_steps: 16,
            ao_strength: 1.0,
            ao_steps: 4,
            ao_radius: 0.2,
        }
    }
}
impl DirectShading {
    /// Largest side of the stratum grid: 16 x 16 = 256 samples per pixel. OFX stops at 3
    /// because its CPU and WGSL engines share the 1 / 4 / 9 table; the CUDA kernel takes any
    /// side (`trace_pixel`: `index % grid`, `index / grid`).
    pub const MAX_GRID: u32 = 16;
    /// Side of Direct's per-pixel stratum grid for a profile's sample target: the largest
    /// square not above it, one deterministic sample per stratum, so the image is final after
    /// `grid²` samples.
    pub fn grid(samples: u32) -> u32 {
        (f64::from(samples).sqrt() as u32).clamp(1, Self::MAX_GRID)
    }
}

/// What a Direct render carries to the parameter pack (`Scene::direct`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DirectRender {
    pub shading: DirectShading,
    /// `DirectShading::grid` of the profile's samples.
    pub grid: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ViewportMode {
    Auto,
    Locked,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProfileTarget {
    Moving,
    Still,
    Manual,
    Output,
}
impl ProfileTarget {
    pub fn viewport_path(self) -> Option<&'static str> {
        match self {
            Self::Moving => Some("/viewport/moving_id"),
            Self::Still => Some("/viewport/still_id"),
            Self::Manual => Some("/viewport/manual_id"),
            Self::Output => None,
        }
    }
}
#[derive(Clone, Debug, PartialEq)]
pub struct EffectiveRender {
    pub profile: NodeId,
    pub method: RenderMethod,
    pub render: Render,
    pub samples: u32,
    pub resolution_scale: f32,
    pub display: Display,
    /// The Direct display's shading; read only when `display` is Direct.
    pub direct: DirectShading,
    /// The share of each fractal left out, 0 (all of it) .. 1 (none): a Cells mask of that
    /// density over the object (`EffectiveRender::show`).
    pub thinning: f32,
}
impl EffectiveRender {
    /// Apply only to an evaluated scene. Authored material/geometry nodes remain untouched.
    pub fn apply_to(&self, scene: &mut crate::scene::Scene) {
        let iterations = scene.render.iterations;
        scene.render = self.render;
        scene.render.iterations = iterations;
        if self.method == RenderMethod::Fast && scene.material.transmission <= 0.0 {
            scene.material.model = crate::scene::MaterialModel::Fast;
        }
        // Direct shades each stratum of its grid once (no path, no noise): adaptive sampling
        // would leave strata unsampled.
        scene.direct = (self.display == Display::Direct).then(|| {
            scene.render.adaptive.enabled = false;
            DirectRender {
                shading: self.direct,
                grid: DirectShading::grid(self.samples),
            }
        });
        for object in &mut scene.objects {
            self.apply_to(object);
        }
    }

    /// Apply the profile's display to an evaluated world scene: thinning adds a Cells mask of
    /// `thinning` density to every fractal (cells a quarter of its framing radius, in its own
    /// space, so they move with it); Bounds keeps each fractal's bounds for the viewport to draw
    /// (`Scene::bounds`) and traces none of them.
    pub fn show(&self, scene: &mut crate::scene::Scene) -> Result<(), String> {
        use crate::scene::{Bounds, Mask, MaskInstance, MaskShape};
        let world = |object: &crate::scene::Scene| {
            object
                .object_world
                .unwrap_or(glam::Mat4::IDENTITY.to_cols_array_2d())
        };
        if self.thinning > 0.0 {
            for object in &mut scene.objects {
                if object.masks.len() >= fractal_core::params::MAX_MASKS {
                    return Err(format!(
                        "{} has {} masks: thinning needs one more",
                        object.name,
                        object.masks.len()
                    ));
                }
                let edge = object.formula.framing_radius() * 0.25;
                object.masks.push(MaskInstance {
                    mask: Mask {
                        shape: MaskShape::Cells,
                        size: [edge; 3],
                        density: self.thinning,
                        ..Mask::default()
                    },
                    world: world(object),
                });
            }
        }
        if self.display == Display::Bounds {
            scene.bounds = scene
                .objects
                .iter()
                .map(|object| Bounds {
                    world: world(object),
                    half: object.formula.framing_radius(),
                    node: object.node,
                })
                .collect();
            scene.objects.clear();
        }
        Ok(())
    }

    /// The `(samples, resolution_scale)` an export takes from the Output profile; refuses
    /// [`Display::Bounds`], which only the viewport draws.
    pub fn export_quality(&self) -> Result<(u32, f32), String> {
        if self.display == Display::Bounds {
            return Err(
                "The Output profile shows Bounds: nothing to render; pick Path traced or Direct"
                    .into(),
            );
        }
        Ok((self.samples, self.resolution_scale))
    }
}
/// The scene node kinds a viewport filters, each with the name of its attributes
/// (`/viewport/show/<name>`, `/viewport/select/<name>`) and its label.
pub const VIEW_KINDS: [(crate::world::WorldKind, &str, &str); 6] = {
    use crate::world::WorldKind::*;
    [
        (Fractal, "fractal", "Fractals"),
        (Mask, "mask", "Masks"),
        (Group, "group", "Groups"),
        (Camera, "camera", "Cameras"),
        (DirectionalLight, "light", "Lights"),
        (Environment, "environment", "Environment"),
    ]
};

/// A set of the [`VIEW_KINDS`]: what a viewport shows or lets pick (Blender's per-viewport
/// `object_type_exclude_viewport` / `_select`, held as the kinds included). Kinds a viewport does
/// not filter (settings, materials) are always in it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Kinds(u8);
impl Kinds {
    pub const ALL: Self = Self(u8::MAX);
    fn bit(kind: crate::world::WorldKind) -> Option<u8> {
        VIEW_KINDS
            .iter()
            .position(|(k, ..)| *k == kind)
            .map(|i| 1 << i)
    }
    pub fn contains(self, kind: crate::world::WorldKind) -> bool {
        Self::bit(kind).is_none_or(|bit| self.0 & bit != 0)
    }
    pub fn with(self, kind: crate::world::WorldKind, on: bool) -> Self {
        match Self::bit(kind) {
            Some(bit) if on => Self(self.0 | bit),
            Some(bit) => Self(self.0 & !bit),
            None => self,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewportPolicy {
    pub mode: ViewportMode,
    pub moving_id: NodeId,
    pub still_id: NodeId,
    pub manual_id: NodeId,
    pub target_fps: f32,
    pub settle_delay_ms: f32,
    pub batch_budget_ms: f32,
    pub paused: bool,
    pub frozen: bool,
    /// The kinds the viewport shows (render and extras) and lets pick (the manipulator).
    pub shown: Kinds,
    pub selectable: Kinds,
    /// Draw the extras: wires of masks, group locators, cameras and lights (`overlay`).
    pub extras: bool,
}
impl ViewportPolicy {
    /// The binding the viewport renders: Auto follows motion between Moving and Still,
    /// Locked always renders Manual.
    pub fn selected(self, moving: bool) -> (ProfileTarget, NodeId) {
        match self.mode {
            ViewportMode::Locked => (ProfileTarget::Manual, self.manual_id),
            ViewportMode::Auto if moving => (ProfileTarget::Moving, self.moving_id),
            ViewportMode::Auto => (ProfileTarget::Still, self.still_id),
        }
    }
}
/// What the viewport renders in one UI frame: the policy's binding and its evaluated profile.
/// The App keeps the one it sent to the worker so every indicator shows that exact choice.
#[derive(Clone, Debug, PartialEq)]
pub struct ViewportRender {
    pub target: ProfileTarget,
    pub effective: EffectiveRender,
}
pub(crate) const QUALITY_RENDER_FIELDS: &[&str] = &[
    "max_steps",
    "hit_epsilon",
    "step_factor",
    "max_bounces",
    "glass_probes",
    "adaptive",
];
pub(crate) const PROFILE_RENDER_FIELDS: &[&str] = &["backend", "denoise"];
pub(crate) const LOOK_RENDER_FIELDS: &[&str] = &["exposure_stops", "saturation", "reinhard"];
pub(crate) fn quality_values(render: &Value) -> Value {
    let mut quality = serde_json::Map::new();
    for key in QUALITY_RENDER_FIELDS {
        quality.insert((*key).into(), render[*key].clone());
    }
    quality.insert("samples".into(), json!(32));
    quality.insert("resolution_scale".into(), json!(1.0));
    Value::Object(quality)
}
pub(crate) fn render_values(render: &Value) -> Value {
    let mut values = serde_json::Map::new();
    for key in PROFILE_RENDER_FIELDS {
        values.insert((*key).into(), render[*key].clone());
    }
    values.insert("method".into(), json!(RenderMethod::Full));
    values.insert("display".into(), json!(Display::PathTraced));
    values.insert("direct".into(), json!(DirectShading::default()));
    values.insert("thinning".into(), json!(0.0));
    Value::Object(values)
}
pub(crate) fn look_values(render: &Value) -> Value {
    let mut values = serde_json::Map::new();
    for key in LOOK_RENDER_FIELDS {
        values.insert((*key).into(), render[*key].clone());
    }
    Value::Object(values)
}
pub(crate) fn viewport_values(moving: NodeId, still: NodeId) -> Value {
    let all: serde_json::Map<String, Value> = VIEW_KINDS
        .iter()
        .map(|(_, name, _)| ((*name).to_owned(), json!(true)))
        .collect();
    json!({"mode":ViewportMode::Auto,"moving_id":moving,"still_id":still,"manual_id":still,"target_fps":30.0,"settle_delay_ms":180.0,"batch_budget_ms":8.0,"paused":false,"frozen":false,"show":all,"select":all,"extras":true})
}
/// The `/output/*` attribute edits that turn recipe `before` into `after` on OutputSettings
/// node `id`: one command per changed field, so an edit touches only what the form changed.
pub(crate) fn output_edits(
    id: NodeId,
    before: &crate::export::OutputSettings,
    after: &crate::export::OutputSettings,
) -> Result<Vec<crate::world::WorldCommand>, String> {
    let fields = |output: &crate::export::OutputSettings| match serde_json::to_value(output) {
        Ok(Value::Object(fields)) => Ok(fields),
        Ok(_) => Err("OutputSettings must serialize to an object".to_string()),
        Err(error) => Err(error.to_string()),
    };
    let before = fields(before)?;
    Ok(fields(after)?
        .into_iter()
        .filter(|(key, value)| before.get(key) != Some(value))
        .map(|(key, value)| crate::world::WorldCommand::SetAttribute {
            id,
            path: format!("/output/{key}"),
            value,
            frame: 0.0,
        })
        .collect())
}
pub(crate) fn reference_kind(path: &str) -> Option<crate::world::WorldKind> {
    use crate::world::WorldKind;
    match path {
        "/viewport/moving_id" | "/viewport/still_id" | "/viewport/manual_id" => {
            Some(WorldKind::RenderSettings)
        }
        _ => None,
    }
}
/// Document-level policy branches (viewport routing, export recipe): one choice for the whole
/// document, never varying over time.
pub(crate) fn static_branch(path: &str) -> bool {
    path.starts_with("/viewport/") || path.starts_with("/output/")
}
/// Settings attributes that must stay static: typed references and the policy branches.
/// Commands, loading and the Attribute Editor all ask this one predicate.
pub(crate) fn static_path(path: &str) -> bool {
    reference_kind(path).is_some() || static_branch(path) || path == "/render/backend"
}
pub(crate) fn setting_kind(kind: crate::world::WorldKind) -> bool {
    matches!(
        kind,
        crate::world::WorldKind::RenderSettings
            | crate::world::WorldKind::ViewportSettings
            | crate::world::WorldKind::OutputSettings
    )
}
