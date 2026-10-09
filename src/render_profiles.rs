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
    pub quality: NodeId,
    pub method: RenderMethod,
    pub render: Render,
    pub samples: u32,
    pub resolution_scale: f32,
}
impl EffectiveRender {
    /// Apply only to an evaluated scene. Authored material/geometry nodes remain untouched.
    pub fn apply_to(&self, scene: &mut crate::scene::Scene) {
        let iterations = scene.render.iterations;
        scene.render = self.render.clone();
        scene.render.iterations = iterations;
        if self.method == RenderMethod::Fast && scene.material.transmission <= 0.0 {
            scene.material.model = crate::scene::MaterialModel::Fast;
        }
        for object in &mut scene.objects {
            self.apply_to(object);
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
pub(crate) const PROFILE_RENDER_FIELDS: &[&str] =
    &["exposure_stops", "saturation", "reinhard", "denoise"];
pub(crate) fn quality_values(render: &Value) -> Value {
    let mut quality = serde_json::Map::new();
    for key in QUALITY_RENDER_FIELDS {
        quality.insert((*key).into(), render[*key].clone());
    }
    quality.insert("samples".into(), json!(512));
    quality.insert("resolution_scale".into(), json!(1.0));
    Value::Object(quality)
}
pub(crate) fn render_values(render: &Value, quality: NodeId) -> Value {
    let mut values = serde_json::Map::new();
    for key in PROFILE_RENDER_FIELDS {
        values.insert((*key).into(), render[*key].clone());
    }
    values.insert("method".into(), json!(RenderMethod::Full));
    values.insert("quality_id".into(), json!(quality));
    Value::Object(values)
}
pub(crate) fn viewport_values(moving: NodeId, still: NodeId) -> Value {
    json!({"mode":ViewportMode::Auto,"moving_id":moving,"still_id":still,"manual_id":still,"target_fps":30.0,"settle_delay_ms":180.0,"batch_budget_ms":8.0,"paused":false,"frozen":false})
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
        "/render/quality_id" => Some(WorldKind::QualitySettings),
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
    reference_kind(path).is_some() || static_branch(path)
}
pub(crate) fn setting_kind(kind: crate::world::WorldKind) -> bool {
    matches!(
        kind,
        crate::world::WorldKind::RenderSettings
            | crate::world::WorldKind::QualitySettings
            | crate::world::WorldKind::ViewportSettings
            | crate::world::WorldKind::OutputSettings
    )
}
