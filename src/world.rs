//! Playa graph nodes own world attributes; typed scenes are render snapshots.
#[cfg(test)]
#[path = "world_tests.rs"]
mod tests;
use crate::scene::Scene;
use curves::Tan;
use playa_engine::entities::anim::{Animation, Channel, Keyframe};
use playa_engine::entities::{AttrValue, Attrs};
pub use playa_graph::NodeId;
use playa_graph::SubnetFile;
#[cfg(test)]
use playa_graph::{Graph, Node, RustBox};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::LazyLock;

/// Marker of WarpBro nodes on the system clipboard, so foreign text is never pasted as nodes.
const CLIPBOARD_KEY: &str = "warpbro_nodes";
const CLIPBOARD_VERSION: u32 = 1;

/// The node data of a `WorldDocument::copy_fragment` clipboard text; None for any other text.
pub fn parse_clipboard(text: &str) -> Option<HashMap<String, Value>> {
    let value: Value = serde_json::from_str(text.trim()).ok()?;
    if value[CLIPBOARD_KEY].as_u64() != Some(u64::from(CLIPBOARD_VERSION)) {
        return None;
    }
    serde_json::from_value(value["nodes"].clone()).ok()
}

pub(crate) const CAMERA_ORBIT_SPEED: &str = "/camera/orbit_speed_degrees";
pub(crate) const CAMERA_ORBIT_PHASE: &str = "/camera/orbit_phase_degrees";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WorldKind {
    Fractal,
    Camera,
    DirectionalLight,
    Environment,
    Group,
    Material,
    RenderSettings,
    QualitySettings,
    ViewportSettings,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorldDocument {
    pub graph: SubnetFile,
    pub active_camera: Option<NodeId>,
    pub active_environment: Option<NodeId>,
    pub first: u32,
    pub last: u32,
    pub fps: f64,
    /// Numbered time marks (After Effects composition markers): slot 0-9 -> frame.
    pub marks: BTreeMap<u8, u32>,
}
impl PartialEq for WorldDocument {
    fn eq(&self, other: &Self) -> bool {
        serde_json::to_value(self).ok() == serde_json::to_value(other).ok()
    }
}
#[derive(Clone, Debug)]
pub struct WorldNodeInfo {
    pub id: NodeId,
    pub kind: WorldKind,
    pub name: String,
    pub parent: Option<NodeId>,
    pub visible: bool,
    pub locked: bool,
    pub solo: bool,
    pub start: f64,
    pub end: f64,
}
#[derive(Clone, Debug)]
pub struct WorldAttribute {
    pub path: String,
    pub label: String,
    pub value: Value,
    pub frames: Vec<f64>,
    pub keyable: bool,
    pub component: Option<usize>,
    pub choices: Vec<Value>,
    /// Hard limits; the document clamps to them (`attribute_range`).
    pub range: Option<(f64, f64)>,
    /// The slider span (`attribute_slider`); components inherit their vector's.
    pub slider: Option<Slider>,
    /// An RGB colour (`is_color_attribute`).
    pub color: bool,
    /// What "Reset" restores: the attribute's value in a fresh world ([`attribute_default`]).
    /// None: the attribute has no counterpart there (another formula family's parameter).
    pub default: Option<Value>,
}

/// The span a parameter's slider covers: its useful range. Typing may go past it, up to the
/// hard limit (`WorldAttribute::range`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Slider {
    pub min: f64,
    pub max: f64,
    /// Logarithmic for values spanning orders of magnitude.
    pub log: bool,
}
#[derive(Clone, Debug)]
pub enum WorldCommand {
    Create {
        kind: WorldKind,
        name: String,
        parent: Option<NodeId>,
    },
    Delete(NodeId),
    /// Copy the nodes (with descendants) in place; the copies are selected (Ctrl+D).
    Duplicate(Vec<NodeId>),
    /// Insert a clipboard fragment from `WorldDocument::copy_fragment` (Ctrl+V).
    Paste(String),
    Rename {
        id: NodeId,
        name: String,
    },
    Reparent {
        id: NodeId,
        parent: Option<NodeId>,
    },
    SetAttribute {
        id: NodeId,
        path: String,
        value: Value,
        frame: f64,
    },
    Key {
        id: NodeId,
        path: String,
        frame: f64,
    },
    RemoveKey {
        id: NodeId,
        path: String,
        frame: f64,
    },
    MoveKeys {
        id: NodeId,
        path: String,
        frames: Vec<f64>,
        delta: f64,
    },
    Interpolation {
        id: NodeId,
        path: String,
        frames: Vec<f64>,
        kind: Tan,
    },
    SetMetadata {
        id: NodeId,
        path: String,
        value: Value,
    },
    Reorder {
        ids: Vec<NodeId>,
    },
    SetTimeRange {
        first: u32,
        last: u32,
        fps: f64,
    },
    /// Set (Some) or clear (None) time mark `slot` (0-9).
    SetMark {
        slot: u8,
        frame: Option<u32>,
    },
    AssignMaterial {
        id: NodeId,
        material: Option<NodeId>,
    },
    /// Create and select a standalone material without changing object assignments.
    CreateMaterial {
        material: crate::scene::Material,
        name: String,
    },
    /// Create a material and assign it to `targets`: one undo step for "assign a library preset".
    CreateMaterialFor {
        material: crate::scene::Material,
        name: String,
        targets: Vec<NodeId>,
    },
    /// Instantiate and assign to a fractal, or edit a selected material node.
    ApplyMaterial {
        id: NodeId,
        material: crate::scene::Material,
        name: String,
        frame: f64,
    },
    CreateRenderProfile {
        name: String,
        role: crate::render_profiles::CatalogRole,
        source: Option<NodeId>,
        target: Option<crate::render_profiles::ProfileTarget>,
    },
    CreateQualityProfile {
        name: String,
        role: crate::render_profiles::CatalogRole,
        source: Option<NodeId>,
        target: Option<NodeId>,
    },
    InstantiateQualityTemplate {
        id: NodeId,
        name: String,
        target: Option<NodeId>,
    },
    InstantiateRenderTemplate {
        id: NodeId,
        name: String,
        target: Option<crate::render_profiles::ProfileTarget>,
    },
    SetCatalogRole {
        id: NodeId,
        role: crate::render_profiles::CatalogRole,
    },
    SetOutputRender(NodeId),
    SetActiveCamera(NodeId),
    SetActiveEnvironment(NodeId),
    ReloadEnvironment(NodeId),
    Batch(Vec<WorldCommand>),
    #[cfg(test)]
    SetVisible {
        id: NodeId,
        visible: bool,
    },
    #[cfg(test)]
    SetLocked {
        id: NodeId,
        locked: bool,
    },
    #[cfg(test)]
    SetSolo {
        id: NodeId,
        solo: bool,
    },
    SetSpan {
        id: NodeId,
        start: f64,
        end: f64,
    },
    SetAnimation {
        id: NodeId,
        path: String,
        enabled: bool,
        frame: f64,
    },
}
type WorldEditSnapshot = (WorldDocument, Option<NodeId>, Vec<NodeId>);

#[derive(Clone, Debug)]
struct PendingWorldEdit {
    gesture: u64,
    original: WorldEditSnapshot,
}

#[derive(Clone, Debug)]
pub struct WorldEditor {
    pub document: WorldDocument,
    pub selection: Option<NodeId>,
    pub selected: Vec<NodeId>,
    undo: Vec<WorldEditSnapshot>,
    redo: Vec<WorldEditSnapshot>,
    pending_edit: Option<PendingWorldEdit>,
    revision: u64,
    /// Interpolation of every key this editor creates (Settings > Animation > New key type).
    pub new_key: Tan,
}
impl WorldEditor {
    pub fn new(document: WorldDocument) -> Self {
        let selection = document
            .nodes()
            .iter()
            .find(|n| n.kind == WorldKind::Fractal)
            .map(|n| n.id);
        Self {
            document,
            selection,
            selected: selection.into_iter().collect(),
            undo: vec![],
            redo: vec![],
            revision: 0,
            pending_edit: None,
            new_key: Tan::Smooth,
        }
    }
    /// Cache invalidation token, combined with the document UUID by consumers.
    pub fn revision(&self) -> u64 {
        self.revision
    }
    pub fn execute(&mut self, command: WorldCommand) -> Result<(), String> {
        self.execute_edit(command, None)
    }

    /// Apply live edits, deferring their single Undo step until the gesture finishes.
    pub fn execute_edit(
        &mut self,
        command: WorldCommand,
        gesture: Option<u64>,
    ) -> Result<(), String> {
        self.apply_edit(gesture, |editor| editor.apply(command))
    }

    pub fn active_edit(&self) -> Option<u64> {
        self.pending_edit.as_ref().map(|pending| pending.gesture)
    }

    /// Commit the gesture's original snapshot only when its final state differs.
    /// A gesture returned to its original state preserves the existing redo stack.
    pub fn finish_edit(&mut self) -> bool {
        let Some(pending) = self.pending_edit.take() else {
            return false;
        };
        if !self.differs_from(&pending.original) {
            return false;
        }
        self.undo.push(pending.original);
        self.redo.clear();
        true
    }

    pub fn finish_edit_unless(&mut self, active: Option<u64>) -> bool {
        if self.active_edit() != active {
            self.finish_edit()
        } else {
            false
        }
    }

    fn differs_from(&self, snapshot: &WorldEditSnapshot) -> bool {
        self.document != snapshot.0 || self.selection != snapshot.1 || self.selected != snapshot.2
    }

    fn record_edit(&mut self, before: WorldEditSnapshot, gesture: Option<u64>) {
        if !self.differs_from(&before) {
            return;
        }
        self.revision = self.revision.wrapping_add(1);
        if let Some(gesture) = gesture {
            if self.pending_edit.is_none() {
                self.pending_edit = Some(PendingWorldEdit {
                    gesture,
                    original: before,
                });
            }
        } else {
            self.undo.push(before);
            self.redo.clear();
        }
    }

    fn apply_edit(
        &mut self,
        gesture: Option<u64>,
        apply: impl FnOnce(&mut Self) -> Result<(), String>,
    ) -> Result<(), String> {
        self.finish_edit_unless(gesture);
        // This rollback snapshot is temporary; only the first changed edit's snapshot
        // is retained for the gesture's history entry.
        let before = (self.document.clone(), self.selection, self.selected.clone());
        if let Err(error) = apply(self).and_then(|()| {
            self.document
                .validate_render_profiles(f64::from(self.document.first))
        }) {
            self.document = before.0;
            self.selection = before.1;
            self.selected = before.2;
            return Err(error);
        }
        self.record_edit(before, gesture);
        Ok(())
    }
    /// Commit a capture as one undoable edit, preserving keys outside its recorded interval.
    pub fn record_camera(
        &mut self,
        recording: &crate::camera_recorder::Recording,
    ) -> Result<(usize, usize, f64, u32), String> {
        if self.document != recording.document || self.revision() != recording.revision {
            return Err("The document changed during camera recording".into());
        }
        let prepared = recording.prepare()?;
        self.apply_edit(None, |editor| {
            editor.document.assert_unlocked(prepared.camera)?;
            let mut attrs = editor.document.attrs(prepared.camera)?;
            for (path, mut animation) in prepared.channels.clone() {
                if !attrs.contains(path) {
                    return Err(format!("Missing camera channel {path}"));
                }
                if let Some(previous) = attrs.anim(path) {
                    if previous.channels.len() != animation.channels.len() {
                        return Err(format!("Camera channel {path} changed shape"));
                    }
                    for (channel, old) in animation.channels.iter_mut().zip(&previous.channels) {
                        let mut keys: Vec<_> = old
                            .keys()
                            .iter()
                            .filter(|key| key.t() < prepared.first || key.t() > prepared.last)
                            .cloned()
                            .collect();
                        keys.extend_from_slice(channel.keys());
                        keys.sort_by(|a, b| a.t().total_cmp(&b.t()));
                        let mut track =
                            curves::Track::from_keys(keys).map_err(|e| e.to_string())?;
                        track.set_extrap(old.track().pre(), old.track().post());
                        channel.replace_track(track);
                    }
                }
                // Captured values describe the visible pose; stale navigation offsets would apply twice.
                attrs.remove(&format!("/_navigation{path}"));
                attrs.set_conn(path, None);
                attrs.set_anim(path, Some(animation));
            }
            editor.document.store_attrs(prepared.camera, &attrs)?;
            if prepared.extend_last > editor.document.last {
                let old_end = f64::from(editor.document.last) + 1.0;
                for node in editor.document.nodes() {
                    let mut attrs = editor.document.attrs(node.id)?;
                    if !attrs.is_animated("/end")
                        && attrs.get("/end").cloned().map(attr_json) == Some(json!(old_end))
                    {
                        attrs.set(
                            "/end",
                            to_attr(&json!(f64::from(prepared.extend_last) + 1.0)),
                        );
                        editor.document.store_attrs(node.id, &attrs)?;
                    }
                }
                editor.document.last = prepared.extend_last;
            }
            Ok(())
        })?;
        Ok((
            prepared.sample_count,
            prepared.key_count,
            prepared.max_error,
            prepared.last.ceil() as u32,
        ))
    }

    pub fn undo(&mut self) -> bool {
        self.finish_edit();
        if let Some((d, s, selected)) = self.undo.pop() {
            self.redo.push((
                std::mem::replace(&mut self.document, d),
                self.selection,
                self.selected.clone(),
            ));
            self.selection = s;
            self.selected = selected;
            self.revision = self.revision.wrapping_add(1);
            true
        } else {
            false
        }
    }
    pub fn redo(&mut self) -> bool {
        self.finish_edit();
        if let Some((d, s, selected)) = self.redo.pop() {
            self.undo.push((
                std::mem::replace(&mut self.document, d),
                self.selection,
                self.selected.clone(),
            ));
            self.selection = s;
            self.selected = selected;
            self.revision = self.revision.wrapping_add(1);
            true
        } else {
            false
        }
    }
    #[cfg(test)]
    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }
    /// Insert `nodes` (UUID -> node data) with fresh UUIDs and select the pasted roots. References
    /// inside the fragment follow the remap; a parent or material that is neither in the fragment
    /// nor in this document is cleared. A fragment that still cannot be evaluated (e.g. an
    /// attribute connection to an absent node) is an error, so the edit rolls back.
    fn insert_fragment(&mut self, nodes: HashMap<String, Value>) -> Result<Vec<NodeId>, String> {
        if nodes.is_empty() {
            return Err("Nothing to paste".into());
        }
        // Copies hold references (UUIDs), never the referenced nodes; a reference that resolves
        // in neither the fragment nor this document is cleared (reported by `unresolved_references`).
        let mut nodes = nodes;
        for (key, field) in self.document.unresolved_references(&nodes) {
            if let Some(data) = nodes.get_mut(&key) {
                data[field] = Value::Null;
            }
        }
        let mut fragment = SubnetFile {
            nodes,
            ..self.document.graph.clone()
        };
        let map = playa_graph::remap_node_ids(&mut fragment);
        let fresh: HashSet<NodeId> = map.values().copied().collect();
        for data in fragment.nodes.values_mut() {
            let mut attrs: Attrs =
                serde_json::from_value(data["host"].clone()).map_err(|e| e.to_string())?;
            for path in [
                "/render/quality_id",
                "/viewport/moving_id",
                "/viewport/still_id",
                "/viewport/manual_id",
            ] {
                if let Some(new) = attrs
                    .get(path)
                    .cloned()
                    .map(attr_json)
                    .and_then(|v| v.as_str().and_then(NodeId::parse))
                    .and_then(|old| map.get(&old))
                {
                    attrs.set(path, to_attr(&json!(new)));
                    if let Some(reference) = data["gpu"].pointer_mut(path) {
                        *reference = json!(new);
                    }
                }
            }
            data["host"] = serde_json::to_value(attrs).map_err(|e| e.to_string())?;
            // Host UUID references are ordinary data, remapped alongside Playa graph references.
            if let Some(new) = data["material"]
                .as_str()
                .and_then(NodeId::parse)
                .and_then(|old| map.get(&old))
            {
                data["material"] = json!(new);
            }
        }
        let roots: Vec<NodeId> = fragment
            .nodes
            .iter()
            .filter_map(|(id, data)| {
                let id = NodeId::parse(id)?;
                let parent = data["parent"].as_str().and_then(NodeId::parse);
                (!parent.is_some_and(|p| fresh.contains(&p))).then_some(id)
            })
            .collect();
        self.document.graph.nodes.extend(fragment.nodes);
        self.document.rebuild_children();
        self.document
            .snapshot(f64::from(self.document.first))
            .map_err(|e| format!("Pasted nodes do not fit this scene: {e}"))?;
        let order = self.document.nodes();
        let mut roots = roots;
        roots.sort_by_key(|id| order.iter().position(|n| n.id == *id));
        self.selection = roots.last().copied();
        self.selected = roots.clone();
        Ok(roots)
    }
    fn apply(&mut self, command: WorldCommand) -> Result<(), String> {
        match command {
            #[cfg(test)]
            WorldCommand::SetVisible { id, visible } => self.apply(WorldCommand::SetAttribute {
                id,
                path: "/visible".into(),
                value: json!(visible),
                frame: 0.0,
            })?,
            #[cfg(test)]
            WorldCommand::SetLocked { id, locked } => self.apply(WorldCommand::SetAttribute {
                id,
                path: "/locked".into(),
                value: json!(locked),
                frame: 0.0,
            })?,
            #[cfg(test)]
            WorldCommand::SetSolo { id, solo } => self.apply(WorldCommand::SetAttribute {
                id,
                path: "/solo".into(),
                value: json!(solo),
                frame: 0.0,
            })?,
            WorldCommand::SetSpan { id, start, end } => {
                if !start.is_finite() || !end.is_finite() || start >= end {
                    return Err("Span must have a finite end after its start".into());
                }
                self.apply(WorldCommand::Batch(vec![
                    WorldCommand::SetAttribute {
                        id,
                        path: "/start".into(),
                        value: json!(start),
                        frame: 0.0,
                    },
                    WorldCommand::SetAttribute {
                        id,
                        path: "/end".into(),
                        value: json!(end),
                        frame: 0.0,
                    },
                ]))?;
            }
            WorldCommand::SetAnimation {
                id,
                path,
                enabled,
                frame,
            } => {
                self.document.assert_unlocked(id)?;
                if enabled {
                    self.apply(WorldCommand::Key { id, path, frame })?;
                } else {
                    let held = self.document.attribute_value(id, &path, frame)?;
                    let mut attrs = self.document.attrs(id)?;
                    if let Some((parent, index)) = self.document.component_path(id, &path)? {
                        let mut base = attrs
                            .get(&parent)
                            .cloned()
                            .map(attr_json)
                            .ok_or("Missing component")?;
                        base[index] =
                            self.document
                                .without_navigation_offset(id, &path, held.clone())?;
                        attrs.set(&parent, to_attr(&base));
                        if let Some(anim) = attrs.anim_mut(&parent) {
                            anim.channels[index] = Channel::new();
                        }
                        if attrs.anim(&parent).is_some_and(Animation::is_empty) {
                            attrs.set_anim(&parent, None);
                        }
                        self.document.store_attrs(id, &attrs)?;
                        return Ok(());
                    }
                    attrs.set_anim(&path, None);
                    self.document.store_attrs(id, &attrs)?;
                    self.document
                        .set_attribute(id, &path, held, frame, false, self.new_key)?;
                }
            }
            WorldCommand::Batch(commands) => {
                for c in commands {
                    self.apply(c)?;
                }
            }
            WorldCommand::Create { kind, name, parent } => {
                self.document.validate_parent(None, parent)?;
                let scene = self.document.snapshot(0.0)?;
                let id = self.document.insert(kind, &name, &scene, parent)?;
                self.selection = Some(id);
                self.selected = vec![id];
            }
            WorldCommand::Delete(id) => {
                self.document.assert_unlocked(id)?;
                let mut remove = HashSet::from([id]);
                loop {
                    let n = remove.len();
                    for node in self.document.nodes() {
                        if node.parent.is_some_and(|p| remove.contains(&p)) {
                            remove.insert(node.id);
                        }
                    }
                    if n == remove.len() {
                        break;
                    }
                }
                self.document.guard_settings_delete(&remove)?;
                for id in &remove {
                    self.document.graph.nodes.remove(&id.to_string());
                }
                for data in self.document.graph.nodes.values_mut() {
                    if data["material"]
                        .as_str()
                        .and_then(NodeId::parse)
                        .is_some_and(|id| remove.contains(&id))
                    {
                        data["material"] = Value::Null;
                    }
                }
                self.selected.retain(|id| !remove.contains(id));
                if self.selection.is_some_and(|id| remove.contains(&id)) {
                    self.selection = self.selected.first().copied();
                }
                if self
                    .document
                    .active_camera
                    .is_some_and(|id| remove.contains(&id))
                {
                    self.document.active_camera = None;
                }
                if self
                    .document
                    .active_environment
                    .is_some_and(|id| remove.contains(&id))
                {
                    self.document.active_environment = None;
                }
            }
            WorldCommand::Duplicate(ids) => {
                for &id in &ids {
                    self.document.assert_unlocked(id)?;
                }
                let nodes = self.document.fragment(&ids)?;
                let roots = self.insert_fragment(nodes)?;
                for &id in &roots {
                    let name = format!("{} copy", self.document.info(id)?.name);
                    self.document.node_mut(id)?["name"] = json!(name);
                }
            }
            WorldCommand::Paste(text) => {
                let nodes = parse_clipboard(&text).ok_or("The clipboard holds no WarpBro nodes")?;
                self.insert_fragment(nodes)?;
            }
            WorldCommand::Rename { id, name } => {
                self.document.assert_unlocked(id)?;
                if name.trim().is_empty() {
                    return Err("Name cannot be empty".into());
                }
                self.document.node_mut(id)?["name"] = json!(name);
            }
            WorldCommand::Reparent { id, parent } => {
                self.document.assert_unlocked(id)?;
                self.document.validate_parent(Some(id), parent)?;
                self.document.node_mut(id)?["parent"] = json!(parent);
            }
            WorldCommand::Reorder { ids } => {
                let known: HashSet<_> = self.document.nodes().iter().map(|n| n.id).collect();
                let unique: HashSet<_> = ids.iter().copied().collect();
                if unique.len() != ids.len() || !unique.is_subset(&known) {
                    return Err("Layer order contains unknown or duplicate IDs".into());
                }
                let mut order = ids;
                order.extend(
                    self.document
                        .nodes()
                        .into_iter()
                        .filter(|n| !unique.contains(&n.id))
                        .map(|n| n.id),
                );
                self.document
                    .graph
                    .bus_slots
                    .insert("layer_order".into(), json!(order));
            }
            WorldCommand::SetTimeRange { first, last, fps } => {
                if first > last || !fps.is_finite() || fps <= 0.0 {
                    return Err("Invalid animation time range".into());
                }
                self.document.first = first;
                self.document.last = last;
                self.document.fps = fps;
            }
            WorldCommand::SetMark { slot, frame } => {
                if slot > 9 {
                    return Err(format!("Time mark {slot} is not 0-9"));
                }
                match frame {
                    Some(frame) => self.document.marks.insert(slot, frame),
                    None => self.document.marks.remove(&slot),
                };
            }
            WorldCommand::CreateMaterial { material, name } => {
                let scene = Scene {
                    material,
                    ..Scene::preset(0)
                };
                let id = self
                    .document
                    .insert(WorldKind::Material, &name, &scene, None)?;
                self.selection = Some(id);
                self.selected = vec![id];
            }
            WorldCommand::CreateMaterialFor {
                material,
                name,
                targets,
            } => {
                if targets.is_empty() {
                    return Err("Select an object that supports materials".into());
                }
                self.apply(WorldCommand::CreateMaterial { material, name })?;
                let created = self.selection.ok_or("The new material was not selected")?;
                for id in targets {
                    self.apply(WorldCommand::AssignMaterial {
                        id,
                        material: Some(created),
                    })?;
                }
            }
            WorldCommand::ApplyMaterial {
                id,
                material,
                name,
                frame,
            } => {
                if !frame.is_finite() {
                    return Err("Invalid material frame".into());
                }
                self.document.assert_unlocked(id)?;
                match self.document.info(id)?.kind {
                    WorldKind::Fractal => {
                        if let Some(assigned) = self.document.assigned_material(id)? {
                            return self.apply(WorldCommand::ApplyMaterial {
                                id: assigned,
                                material,
                                name,
                                frame,
                            });
                        }
                        let mut scene = self.document.snapshot(frame)?;
                        scene.material = material;
                        let material =
                            self.document
                                .insert(WorldKind::Material, &name, &scene, None)?;
                        self.apply(WorldCommand::AssignMaterial {
                            id,
                            material: Some(material),
                        })?;
                    }
                    WorldKind::Material => {
                        let value = serde_json::to_value(material).map_err(|e| e.to_string())?;
                        let mut replacements = Attrs::new();
                        discover(&value, "/material", &mut replacements);
                        for (path, value) in replacements.iter() {
                            self.document.set_attribute(
                                id,
                                path,
                                attr_json(value.clone()),
                                frame,
                                false,
                                self.new_key,
                            )?;
                        }
                    }
                    _ => {
                        return Err("Select a fractal or material to apply a library preset".into());
                    }
                }
            }
            WorldCommand::AssignMaterial { id, material } => {
                self.document.assert_unlocked(id)?;
                if !self.document.supports_material(id) {
                    return Err("Selected node has no material reference field".into());
                }
                if let Some(material) = material {
                    if self.document.info(material)?.kind != WorldKind::Material {
                        return Err("Select a material layer".into());
                    }
                }
                self.document.node_mut(id)?["material"] = json!(material);
            }
            WorldCommand::CreateRenderProfile {
                name,
                role,
                source,
                target,
            } => {
                self.create_render_profile(
                    source.unwrap_or(self.document.output_render_profile()?),
                    name,
                    role,
                    target,
                )?;
            }
            WorldCommand::CreateQualityProfile {
                name,
                role,
                source,
                target,
            } => {
                let source = match source {
                    Some(source) => source,
                    None => self
                        .document
                        .render_quality(self.document.output_render_profile()?)?,
                };
                self.create_quality_profile(source, name, role, target)?;
            }
            WorldCommand::InstantiateQualityTemplate { id, name, target } => {
                if self.document.catalog_role(id)?
                    != Some(crate::render_profiles::CatalogRole::Template)
                {
                    return Err("The selected node is not a quality template".into());
                }
                self.create_quality_profile(
                    id,
                    name,
                    crate::render_profiles::CatalogRole::Profile,
                    target,
                )?;
            }
            WorldCommand::InstantiateRenderTemplate { id, name, target } => {
                if self.document.catalog_role(id)?
                    != Some(crate::render_profiles::CatalogRole::Template)
                {
                    return Err("The selected node is not a render template".into());
                }
                self.create_render_profile(
                    id,
                    name,
                    crate::render_profiles::CatalogRole::Profile,
                    target,
                )?;
            }
            WorldCommand::SetCatalogRole { id, role } => {
                self.document.assert_unlocked(id)?;
                if !crate::render_profiles::setting_kind(self.document.info(id)?.kind) {
                    return Err("Only settings nodes can be profiles or templates".into());
                }
                self.document.node_mut(id)?["metadata"]["catalog_role"] = json!(role);
            }
            WorldCommand::SetOutputRender(id) => {
                self.document
                    .require_live_settings(id, WorldKind::RenderSettings)?;
                self.document
                    .graph
                    .bus_slots
                    .insert("output_render".into(), json!(id));
            }
            WorldCommand::SetActiveCamera(id) => {
                if self.document.info(id)?.kind != WorldKind::Camera {
                    return Err("Select a camera".into());
                }
                self.document.active_camera = Some(id);
            }
            WorldCommand::ReloadEnvironment(id) => {
                self.document.assert_unlocked(id)?;
                if self.document.info(id)?.kind != WorldKind::Environment {
                    return Err("Select an environment layer".into());
                }
                let revision = self.document.node(id)?["environment_revision"]
                    .as_u64()
                    .unwrap_or(0)
                    .wrapping_add(1);
                self.document.node_mut(id)?["environment_revision"] = json!(revision);
            }
            WorldCommand::SetActiveEnvironment(id) => {
                if self.document.info(id)?.kind != WorldKind::Environment {
                    return Err("Select an environment".into());
                }
                self.document.active_environment = Some(id);
            }
            WorldCommand::SetMetadata { id, path, value } => {
                self.document.assert_unlocked(id)?;
                let data = self.document.node_mut(id)?;
                if path.is_empty() || path == "/" {
                    data["metadata"] = value;
                } else {
                    set_pointer(&mut data["metadata"], &path, value, true)?;
                }
            }
            WorldCommand::SetAttribute {
                id,
                path,
                value,
                frame,
            } => {
                if path == "/material_id" {
                    let material = if value.is_null() {
                        None
                    } else {
                        Some(
                            value
                                .as_str()
                                .and_then(NodeId::parse)
                                .ok_or("Invalid material UUID")?,
                        )
                    };
                    return self.apply(WorldCommand::AssignMaterial { id, material });
                }
                if path != "/locked" {
                    self.document.assert_unlocked(id)?;
                }
                self.document
                    .set_attribute(id, &path, value, frame, false, self.new_key)?;
            }
            WorldCommand::Key { id, path, frame } => {
                self.document.assert_unlocked(id)?;
                let value = self.document.attribute_value(id, &path, frame)?;
                self.document
                    .set_attribute(id, &path, value, frame, true, self.new_key)?;
            }
            WorldCommand::RemoveKey { id, path, frame } => {
                self.document.assert_unlocked(id)?;
                let held = self.document.attribute_value(id, &path, frame)?;
                let component = self.document.component_path(id, &path)?;
                let mut attrs = self.document.attrs(id)?;
                if let Some((parent, index)) = &component {
                    if let Some(channel) = attrs
                        .anim_mut(parent)
                        .and_then(|a| a.channels.get_mut(*index))
                    {
                        channel.remove_key(frame);
                    }
                    if attrs
                        .anim(parent)
                        .and_then(|a| a.channels.get(*index))
                        .is_some_and(Channel::is_empty)
                    {
                        let mut base = attrs
                            .get(parent)
                            .cloned()
                            .map(attr_json)
                            .ok_or("Missing component")?;
                        base[*index] =
                            self.document
                                .without_navigation_offset(id, &path, held.clone())?;
                        attrs.set(parent, to_attr(&base));
                    }
                    if attrs.anim(parent).is_some_and(Animation::is_empty) {
                        attrs.set_anim(parent, None);
                    }
                } else {
                    attrs.remove_key(&path, frame);
                }
                self.document.store_attrs(id, &attrs)?;
                if component.is_none() && !attrs.is_animated(&path) {
                    self.document
                        .set_attribute(id, &path, held, frame, false, self.new_key)?;
                }
            }
            WorldCommand::MoveKeys {
                id,
                path,
                frames,
                delta,
            } => {
                self.document.assert_unlocked(id)?;
                if !delta.is_finite() || frames.iter().any(|f| !(f + delta).is_finite()) {
                    return Err("Invalid key time".into());
                }
                let component = self.document.component_path(id, &path)?;
                let parent = component.as_ref().map(|(p, _)| p.as_str()).unwrap_or(&path);
                let mut attrs = self.document.attrs(id)?;
                if let Some(anim) = attrs.anim_mut(parent) {
                    for (index, ch) in anim.channels.iter_mut().enumerate() {
                        if component.as_ref().is_some_and(|(_, i)| *i != index) {
                            continue;
                        }
                        let moves: Vec<_> = ch
                            .keys()
                            .iter()
                            .filter(|k| frames.contains(&k.t()))
                            .map(|k| (k.t(), k.t() + delta))
                            .collect();
                        ch.move_keys(&moves);
                    }
                }
                self.document.store_attrs(id, &attrs)?;
            }
            WorldCommand::Interpolation {
                id,
                path,
                frames,
                kind,
            } => {
                self.document.assert_unlocked(id)?;
                let discrete = if self.document.component_path(id, &path)?.is_some() {
                    false
                } else {
                    self.document.is_discrete(id, &path)?
                };
                let component = self.document.component_path(id, &path)?;
                let parent = component.as_ref().map(|(p, _)| p.as_str()).unwrap_or(&path);
                let mut attrs = self.document.attrs(id)?;
                if let Some(anim) = attrs.anim_mut(parent) {
                    for (index, ch) in anim.channels.iter_mut().enumerate() {
                        if component.as_ref().is_some_and(|(_, i)| *i != index) {
                            continue;
                        }
                        let times: Vec<_> = ch
                            .keys()
                            .iter()
                            .map(|k| k.t())
                            .filter(|t| frames.contains(t))
                            .collect();
                        for t in times {
                            ch.set_tan(t, if discrete { Tan::Constant } else { kind });
                        }
                    }
                }
                self.document.store_attrs(id, &attrs)?;
            }
        }
        self.document.rebuild_children();
        Ok(())
    }
    /// Author viewport navigation on the active Camera layer, without implicit key creation.
    pub fn navigate_camera(
        &mut self,
        after: &Scene,
        frame: f64,
        auto_key: bool,
        gesture: Option<u64>,
    ) -> Result<(), String> {
        let id = self.document.active_camera.ok_or("No active camera")?;
        let writes = self.document.camera_navigation_pose(id, after, frame)?;
        self.apply_edit(gesture, |editor| {
            editor.document.assert_unlocked(id)?;
            let mut attrs = editor.document.attrs(id)?;
            let mut changed = false;
            for (path, value) in writes {
                changed |= editor.document.write_navigation_value(
                    id,
                    &mut attrs,
                    path,
                    value,
                    frame,
                    auto_key,
                    editor.new_key,
                )?;
            }
            // Navigation mode is a static preference on the camera, never a generated key.
            if attrs.is_animated("/camera/free_flight")
                || attrs.conn("/camera/free_flight").is_some()
            {
                if editor
                    .document
                    .attribute_value(id, "/camera/free_flight", frame)?
                    != json!(after.camera.free_flight)
                {
                    return Err("Camera navigation mode is animated or connected".into());
                }
            } else if attrs.get("/camera/free_flight").cloned().map(attr_json)
                != Some(json!(after.camera.free_flight))
            {
                attrs.set(
                    "/camera/free_flight",
                    to_attr(&json!(after.camera.free_flight)),
                );
                changed = true;
            }
            if changed {
                editor.document.store_attrs(id, &attrs)?;
            }
            Ok(())
        })
    }

    /// Bridge typed inspector edits once a control reports a change.
    pub fn edit_snapshot(
        &mut self,
        selected: Option<NodeId>,
        before: &Scene,
        after: &Scene,
        frame: f64,
    ) -> Result<(), String> {
        self.edit_snapshot_with_gesture(selected, before, after, frame, None)
    }

    /// Bridge evaluated/global controls into the same deferred gesture transaction.
    pub fn edit_snapshot_with_gesture(
        &mut self,
        selected: Option<NodeId>,
        before: &Scene,
        after: &Scene,
        frame: f64,
        gesture: Option<u64>,
    ) -> Result<(), String> {
        self.finish_edit_unless(gesture);
        let before = scene_json(before)?;
        let after = scene_json(after)?;
        let mut commands = vec![];
        for node in self.document.nodes() {
            let material = selected
                .and_then(|id| self.document.node(id).ok())
                .and_then(|n| n["material"].as_str())
                .and_then(NodeId::parse);
            if crate::render_profiles::setting_kind(node.kind) {
                continue;
            }
            if node.kind == WorldKind::Fractal && Some(node.id) != selected {
                continue;
            }
            if node.kind == WorldKind::Material
                && Some(node.id) != selected
                && Some(node.id) != material
            {
                continue;
            }
            if node.kind == WorldKind::Camera {
                // Viewport navigation has explicit Auto Key semantics in navigate_camera().
                continue;
            }
            if node.kind == WorldKind::Environment
                && Some(node.id) != self.document.active_environment
            {
                continue;
            }
            for attr in self.document.attributes(node.id, frame)? {
                if attr.component.is_some() || attr.path.starts_with("/transform/") {
                    continue;
                }
                if let (Some(a), Some(b)) = (before.pointer(&attr.path), after.pointer(&attr.path))
                {
                    if a != b {
                        commands.push(WorldCommand::SetAttribute {
                            id: node.id,
                            path: attr.path.clone(),
                            value: b.clone(),
                            frame,
                        });
                    }
                }
            }
            if node.kind == WorldKind::Fractal
                && Some(node.id) == selected
                && before.get("object") != after.get("object")
            {
                for (path, value) in [
                    ("/transform/position", json!(after["object"]["offset"])),
                    (
                        "/transform/rotation_degrees",
                        Value::Array(
                            after["object"]["rotation_degrees"]
                                .as_array()
                                .ok_or("Invalid evaluated rotation")?
                                .iter()
                                .map(|v| json!(-v.as_f64().unwrap_or(0.0)))
                                .collect(),
                        ),
                    ),
                    (
                        "/transform/scale",
                        json!(([after["object"]["scale"].as_f64().unwrap_or(1.0); 3])),
                    ),
                ] {
                    commands.push(WorldCommand::SetAttribute {
                        id: node.id,
                        path: path.into(),
                        value,
                        frame,
                    });
                }
            }
        }
        if before.get("render") != after.get("render") {
            let render = self.document.output_render_profile()?;
            let quality = self.document.render_quality(render)?;
            for (path, target) in self.document.render_edit_paths(render, quality, frame)? {
                let scene_path = path.replacen("/quality/", "/render/", 1);
                if let (Some(a), Some(b)) =
                    (before.pointer(&scene_path), after.pointer(&scene_path))
                {
                    if a != b {
                        commands.push(WorldCommand::SetAttribute {
                            id: target,
                            path,
                            value: b.clone(),
                            frame,
                        });
                    }
                }
            }
        }
        // Commands and global settings share one atomic rollback/history snapshot.
        self.apply_edit(gesture, |editor| {
            editor.apply(WorldCommand::Batch(commands))?;
            for key in ["colour", "name"] {
                if before.get(key) != after.get(key) {
                    editor
                        .document
                        .graph
                        .bus_slots
                        .get_mut("world")
                        .ok_or("Missing settings")?[key] = after[key].clone();
                }
            }
            Ok(())
        })
    }
}
fn attribute_choices(path: &str) -> Vec<Value> {
    if path == "/formula" {
        return (0..8)
            .map(|family| {
                serde_json::to_value(Scene::preset(family).formula)
                    .expect("Formula defaults serialize")
            })
            .collect();
    }
    if path == "/julia" {
        return vec![Value::Null, json!([0.0, 0.0, 0.0])];
    }
    let values: &[&str] = match path {
        "/render/method" => &["Fast", "Full"],
        "/viewport/mode" => &["Auto", "Locked"],
        "/palette" => &[
            "Classic", "Fire", "Ice", "Mono", "Sunset", "Aurora", "Ocean", "Ember", "Amethyst",
            "Verdant", "Copper", "Neon", "RoseGold", "Twilight",
        ],
        "/coloring" => &["Radius", "TrapOrigin", "TrapPlane", "TrapPoint"],
        "/render/denoise/mode" => &["Color", "ColorAlbedo", "ColorAlbedoNormal"],
        "/render/denoise/quality" => &["Fast", "Balanced", "High"],
        "/material/model" => &["Fast", "StandardSurface"],
        "/material/color_source" => &["Palette", "Material"],
        p if p.ends_with("/kind") && p.starts_with("/formula") => {
            &["Tetrahedron", "Octahedron", "Menger"]
        }
        _ => &[],
    };
    values.iter().map(|v| json!(v)).collect()
}
pub(crate) fn attribute_range(path: &str) -> Option<(f64, f64)> {
    if let Some(tail) = path.strip_prefix("/quality/") {
        return match tail {
            "samples" => Some((1.0, 65536.0)),
            "resolution_scale" => Some((0.0625, 1.0)),
            "max_bounces" => Some((0.0, 64.0)),
            "hit_epsilon" => Some((0.0000001, 1.0)),
            "step_factor" => Some((0.0001, 1.0)),
            _ => attribute_range(&format!("/render/{tail}")),
        };
    }
    match path {
        "/viewport/target_fps" => Some((1.0, 240.0)),
        "/viewport/settle_delay_ms" => Some((0.0, 10000.0)),
        "/viewport/batch_budget_ms" => Some((0.1, 1000.0)),
        "/camera/fov_y_degrees" => Some((1.0, 179.0)),
        "/camera/f_number" => Some((0.0, f32::MAX as f64)),
        "/camera/sensor_height" => Some((0.000001, f32::MAX as f64)),
        // Axis index of the trap plane (X, Y, Z); the kernel reads min(2).
        "/trap_axis" => Some((0.0, 2.0)),
        "/render/denoise/interval" => Some((0.0, u32::MAX as f64)),
        // Zero would trace nothing (no step, no exit probe): one is the least that means a march.
        "/render/max_steps" | "/render/glass_probes" => Some((1.0, u32::MAX as f64)),
        "/render/adaptive/noise_threshold" => Some((0.0005, 1.0)),
        "/render/adaptive/min_samples" => {
            Some((crate::scene::Adaptive::MIN_SAMPLES_FLOOR as f64, 65536.0))
        }
        "/material/transmission" => Some((0.0, 1.0)),
        "/material/transmission_depth" => Some((0.0, f32::MAX as f64)),
        p if p.ends_with("roughness") || p.ends_with("metallic") || p == "/material/opacity" => {
            Some((0.0, 1.0))
        }
        _ => None,
    }
}
/// The row label of attribute `path`: its tail, because the Attribute Editor's section already
/// names the category, and a formula section the family ("/formula/Mandelbulb/power" is "Power").
/// A Hybrid keeps its step as the one qualifier that tells two sub-formulas' parameters apart
/// ("/formula/Hybrid/bulb/power" is "Bulb · Power"). ONE function for every attribute view.
pub(crate) fn attribute_label(path: &str) -> String {
    match path {
        "/material_id" => return "Material".into(),
        "/camera/f_number" => return "f-number".into(),
        "/formula/Hybrid/mandelbox/min_radius_ratio" => return "Mandelbox · Min ratio".into(),
        "/formula/Hybrid/mandelbox/rotation_degrees" => return "Mandelbox · Rotate".into(),
        "/material/transmission_extra_roughness" => return "Refraction roughness".into(),
        CAMERA_ORBIT_SPEED => return "Orbit speed (°/s)".into(),
        CAMERA_ORBIT_PHASE => return "Orbit phase (°)".into(),
        "/transform/position" => return "Translate".into(),
        "/transform/rotation" | "/transform/rotation_degrees" => return "Rotate".into(),
        "/transform/scale" => return "Scale".into(),
        _ => {}
    }
    let parts: Vec<_> = path.trim_start_matches('/').split('/').collect();
    let tail = match parts.as_slice() {
        ["formula", "Hybrid", rest @ ..] => rest,
        ["formula", _, rest @ ..] if !rest.is_empty() => rest,
        [_, rest @ ..] if !rest.is_empty() => rest,
        all => all,
    };
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
        .join(" · ")
}

/// The reset value of attribute `path` of a node of `kind`: its value at frame 0 in a FRESH world,
/// the scene the app starts with. ONE source for every reset button; a path the fresh world's node
/// does not have (another formula family's parameter) has none.
pub(crate) fn attribute_default(kind: WorldKind, path: &str) -> Option<Value> {
    static FRESH: LazyLock<HashMap<(WorldKind, String), Value>> = LazyLock::new(|| {
        let world = WorldDocument::from_scene(&Scene::preset(crate::params::FAMILY_BULB));
        let mut defaults = HashMap::new();
        for node in world.nodes() {
            let Ok(attrs) = world.attrs(node.id) else {
                continue;
            };
            for (path, _) in attrs.iter() {
                if let Ok(value) = world.attribute_value(node.id, path, 0.0) {
                    // The first node of a kind defines it (one camera, one sun, ...).
                    defaults.entry((node.kind, path.clone())).or_insert(value);
                }
            }
        }
        defaults
    });
    FRESH.get(&(kind, path.to_owned())).cloned()
}

/// The attribute grid's options of a numeric attribute: its slider span, a step of 1 for an
/// integer, "log" for a logarithmic span and "soft" when the hard limits reach past it. One
/// rule for the Attribute Editor (from `WorldAttribute::slider` / `range`, which components
/// inherit from their vector) and Render settings (from the path's tables).
pub(crate) fn slider_options(
    slider: Option<Slider>,
    range: Option<(f64, f64)>,
    integer: bool,
) -> Vec<String> {
    let Some(slider) = slider else {
        return Vec::new();
    };
    let mut options = vec![slider.min.to_string(), slider.max.to_string()];
    if integer {
        options.push("1".into());
    }
    if slider.log {
        options.push("log".into());
    }
    if range != Some((slider.min, slider.max)) {
        options.push("soft".into());
    }
    options
}

/// Slider spans of every numeric parameter, one table (the operator chose explicit spans over
/// guesses from the current value). Fractal parameters match per family, a Hybrid's sub-formula
/// like its standalone family. A parameter missing here edits as a plain number.
fn attribute_slider(path: &str) -> Option<Slider> {
    if let Some(tail) = path.strip_prefix("/quality/") {
        return match tail {
            "samples" => Some(Slider {
                min: 1.0,
                max: 4096.0,
                log: true,
            }),
            "resolution_scale" => Some(Slider {
                min: 0.0625,
                max: 1.0,
                log: false,
            }),
            _ => attribute_slider(&format!("/render/{tail}")),
        };
    }
    if let Some((min, max)) = attribute_range(path).filter(|_| path.starts_with("/viewport/")) {
        return Some(Slider {
            min,
            max,
            log: false,
        });
    }
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
    // RGB channels (their component rows): 0..1 on the rail, HDR values typed past it.
    if is_color_attribute(path) {
        return lin(0.0, 1.0);
    }
    if let Some(rest) = path.strip_prefix("/formula/") {
        let (family, param) = rest.split_once('/')?;
        let (family, param) = match (family, param.split_once('/')) {
            ("Hybrid", Some(("bulb", p))) => ("Mandelbulb", p),
            ("Hybrid", Some(("mandelbox", p))) => ("Mandelbox", p),
            ("Hybrid", Some(("kifs", p))) => ("Kifs", p),
            _ => (family, param),
        };
        return match (family, param) {
            (_, "rotation_degrees" | "angle_phase_degrees") => lin(-720.0, 720.0),
            ("Mandelbulb", "power") => lin(1.0, 64.0),
            // The logarithmic DE requires an escaped radius >= 1.
            ("Mandelbulb" | "QuaternionJulia", "bailout") => log(1.0, 64.0),
            ("Mandelbulb", "angle_scale") => lin(-8.0, 8.0),
            ("Mandelbox", "scale") => lin(-16.0, 16.0),
            ("Mandelbox", "fixed_radius" | "fold_limit") => log(0.001, 32.0),
            ("Mandelbox", "min_radius_ratio") => lin(0.0, 1.0),
            ("Kifs", "scale") => lin(1.0, 16.0),
            ("Kifs", "offset") => lin(-16.0, 16.0),
            ("Apollonian", "scale") | ("Hybrid", "apollonian_scale") => log(0.01, 32.0),
            // Zero disables the explicit bounding sphere.
            ("Apollonian" | "Kleinian" | "PseudoKleinian", "bound_radius") => lin(0.0, 32.0),
            ("Kleinian", "a") => lin(1.0, 8.0),
            ("Kleinian", "b") => lin(-8.0, 8.0),
            ("PseudoKleinian", "box_size") => lin(0.0, 16.0),
            ("PseudoKleinian", "c") => lin(-8.0, 8.0),
            ("PseudoKleinian", "offset") => lin(-16.0, 16.0),
            ("PseudoKleinian", "size") => log(0.001, 32.0),
            ("PseudoKleinian", "thickness") => log(0.000001, 1.0),
            ("QuaternionJulia", "constant") => lin(-8.0, 8.0),
            ("QuaternionJulia", "slice_w") => lin(-8.0, 8.0),
            // Hybrid uses length(z) / dr, without the logarithmic DE restriction.
            ("Hybrid", "bailout") => log(0.01, 64.0),
            _ => None,
        };
    }
    match path {
        "/julia" => lin(-8.0, 8.0),
        "/trap_point" => lin(-16.0, 16.0),
        "/trap_scale" => log(0.001, 32.0),
        "/trap_axis" => lin(0.0, 2.0),
        "/transform/position" | "/transform/pivot" => lin(-10.0, 10.0),
        "/transform/rotation_degrees" => lin(-180.0, 180.0),
        "/transform/scale" => log(0.01, 10.0),
        "/camera/target" => lin(-5.0, 5.0),
        "/camera/f_number" => lin(0.0, 32.0),
        "/camera/sensor_height" => log(0.001, 0.1),
        "/camera/distance" => log(0.1, 20.0),
        "/camera/focus_distance" => lin(0.0, 10.0),
        "/camera/fov_y_degrees" => lin(5.0, 120.0),
        "/camera/pitch_degrees" => lin(-89.0, 89.0),
        "/camera/yaw_degrees" | "/camera/roll_degrees" | "/camera/orbit_phase_degrees" => {
            lin(-180.0, 180.0)
        }
        "/camera/orbit_speed_degrees" => lin(-90.0, 90.0),
        "/environment/intensity" => lin(0.0, 10.0),
        "/environment/rotation_degrees" => lin(-180.0, 180.0),
        "/lighting/sky_intensity" | "/lighting/sun_intensity" => lin(0.0, 20.0),
        // Angular diameter in degrees (the sun is ~0.53).
        "/lighting/sun_angle" => lin(0.0, 10.0),
        "/lighting/sun_azimuth" => lin(0.0, 360.0),
        "/lighting/sun_elevation" => lin(-90.0, 90.0),
        "/material/coat_ior" | "/material/specular_ior" | "/material/thin_film_ior" => {
            lin(1.0, 3.0)
        }
        "/material/emission" => lin(0.0, 10.0),
        // Nanometres (the iridescent preset uses 450).
        "/material/thin_film_thickness" => lin(0.0, 1500.0),
        "/material/transmission_depth" => lin(0.0, 10.0),
        // Fresnel-like falloff power of the facing blend (presets use ~2.3).
        "/material/facing/exponent" => lin(0.5, 8.0),
        p if p.starts_with("/material/") => lin(0.0, 1.0),
        "/render/adaptive/min_samples" => log(4.0, 256.0),
        "/render/adaptive/noise_threshold" => log(0.001, 0.1),
        "/render/denoise/interval" => lin(0.0, 1024.0),
        "/render/exposure_stops" => lin(-10.0, 10.0),
        "/render/hit_epsilon" => log(0.00001, 0.01),
        "/render/iterations" => lin(1.0, 256.0),
        "/render/max_bounces" => lin(0.0, 16.0),
        "/render/max_steps" => log(16.0, 16384.0),
        "/render/glass_probes" => log(16.0, 16384.0),
        "/render/saturation" => lin(0.0, 2.0),
        "/render/step_factor" => lin(0.1, 1.0),
        "/start" | "/end" => lin(0.0, 500.0),
        _ => None,
    }
}

/// `attribute_hint` for a formula parameter: `family` is the formula (or the Hybrid step it
/// belongs to), `field` its parameter.
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
        _ => return None,
    })
}

/// What an attribute does, for its label's hover text (`AttrField::hint`): the one table
/// every attribute view reads (Attribute Editor, Render settings). Formula parameters match by
/// their tail, so a Hybrid's Mandelbulb / Mandelbox / KIFS step shares the standalone text.
/// None: the label says it all.
pub(crate) fn attribute_hint(path: &str) -> Option<&'static str> {
    if let Some(tail) = path.strip_prefix("/quality/") {
        return match tail {
            "samples" => Some(
                "Maximum samples per pixel. Adaptive sampling may converge earlier; increasing this target preserves compatible accumulated samples.",
            ),
            "resolution_scale" => Some(
                "Render width and height as this fraction of the requested extent. Lower resolution reduces work and starts a separate accumulation.",
            ),
            _ => attribute_hint(&format!("/render/{tail}")),
        };
    }
    // A channel row ("/camera/target/1") explains its vector.
    let path = match path.rsplit_once('/') {
        Some((parent, last)) if last.parse::<usize>().is_ok() => parent,
        _ => path,
    };
    // "/formula/<Family>/<field>"; a Hybrid's steps sit one level deeper ("bulb/power").
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
    Some(match path {
        "/render/method" => {
            "Fast evaluates opaque materials with the existing simplified material model; transmission keeps Standard Surface. Full preserves each authored material model. Quality remains controlled by the linked QualitySettings."
        }
        "/render/quality_id" => {
            "The QualitySettings node shared by this render profile. Editing that node affects every profile that references its UUID; templates must be instantiated before use."
        }
        "/viewport/mode" => {
            "Auto uses Moving during navigation, playback, or scene edits and Still after the settle delay. Locked always uses Manual."
        }
        "/viewport/moving_id" => {
            "The render profile used while the viewport is moving in Auto mode. This reference does not change the output profile."
        }
        "/viewport/still_id" => {
            "The render profile used after activity stops and the settle delay expires in Auto mode."
        }
        "/viewport/manual_id" => {
            "The render profile used in Locked mode, both while moving and at rest."
        }
        "/viewport/target_fps" => {
            "Target viewport update rate for scheduling GPU work. This is a scheduling target, not a guaranteed frame rate or an output frame rate."
        }
        "/viewport/settle_delay_ms" => {
            "Milliseconds without navigation, playback, or scene edits before Auto switches from Moving to Still."
        }
        "/viewport/batch_budget_ms" => {
            "GPU time budget used to choose samples per viewport batch. It changes scheduling, not the saved quality profile."
        }
        "/viewport/paused" => {
            "Pause new viewport rendering batches without changing its profile assignments or export settings."
        }
        "/viewport/frozen" => {
            "Keep the currently displayed viewport image and stop new viewport batches; the image may no longer reflect scene edits."
        }
        "/formula" => "The fractal family and its parameters.",
        "/julia" => {
            "Julia mode (Mandelbulb, Mandelbox): every iteration adds this fixed constant instead of the starting point."
        }
        "/coloring" => {
            "Where the palette coordinate comes from: Radius (escape radius) or an orbit trap (origin, plane, point)."
        }
        "/palette" => {
            "The colour ramp the fractal's colouring indexes (when the material takes its colour from the palette)."
        }
        "/trap_point" => {
            "Orbit trap point: colour by how close the orbit comes to it (Trap point colouring)."
        }
        "/trap_axis" => "Orbit trap plane normal axis (Trap plane colouring).",
        "/trap_scale" => "How fast the trap distance runs through the palette.",
        "/material_id" => "The material this fractal is rendered with.",
        "/camera/target" => "The point the camera looks at and orbits.",
        "/camera/yaw_degrees" => "Camera heading around world up.",
        "/camera/pitch_degrees" => "Camera tilt up / down.",
        "/camera/roll_degrees" => "Camera roll around the view axis.",
        "/camera/distance" => {
            "Distance from the target in framing radii of the fractal (1 frames it)."
        }
        "/camera/fov_y_degrees" => "Vertical field of view.",
        "/camera/f_number" => {
            "Physical f-number: focal length / pupil diameter. 0 disables depth of field."
        }
        "/camera/sensor_height" => {
            "Full vertical sensor gate in scene units (0.024 = 24 mm when one unit is one metre). Focal length = gate / (2 tan(FOV / 2)); independent of camera transform scale."
        }
        "/camera/focus_distance" => "Focus distance in framing radii; 0 focuses on the target.",
        "/camera/free_flight" => "Free flight: roll is free. Off: the horizon stays level.",
        "/camera/orbit_speed_degrees" => {
            "Turntable: degrees per second the camera orbits the target (integrated over the animation)."
        }
        "/camera/orbit_phase_degrees" => "Turntable starting angle added to the orbit.",
        "/environment/enabled" => {
            "Light the scene with the environment image (else the sky gradient)."
        }
        "/environment/path" => "Lat-long HDR / EXR environment image.",
        "/environment/intensity" => "Environment brightness multiplier.",
        "/environment/rotation_degrees" => "Turns the environment around world up.",
        "/lighting/sun_azimuth" => "Sun direction around world up.",
        "/lighting/sun_elevation" => "Sun height above the horizon (negative: below).",
        "/lighting/sun_color" => "Sun colour.",
        "/lighting/sun_intensity" => "Sun brightness.",
        "/lighting/sun_angle" => {
            "Sun disc diameter in degrees: larger gives softer shadows (the real sun is ~0.53)."
        }
        "/lighting/sky_intensity" => "Sky light brightness.",
        "/lighting/sky_horizon" => "Sky colour at the horizon.",
        "/lighting/sky_zenith" => "Sky colour straight up.",
        "/lighting/background" => {
            "Show the sky behind the fractal; off keeps its light but renders the background black."
        }
        "/material/model" => {
            "Fast: a quick metal / plastic model. Standard Surface: the full Autodesk Standard Surface (glass, coat, sheen, film); transmission always uses it."
        }
        "/material/color_source" => {
            "Base colour from the fractal's palette (colouring) or the solid Base color."
        }
        "/material/base_color" => "Solid base colour (when the colour source is Material).",
        "/material/preset" => "The library preset this material came from (a label only).",
        "/material/facing" => {
            "Facing blend: toward a second look at grazing angles by (1 - |N.V|)^exponent (pearlescent, falloff)."
        }
        "/material/facing/color" => "The base colour the facing blend reaches at grazing angles.",
        "/material/facing/roughness" => {
            "The specular roughness the facing blend reaches at grazing angles."
        }
        "/material/facing/metallic" => "The metalness the facing blend reaches at grazing angles.",
        "/material/facing/exponent" => {
            "Falloff power of the facing blend: higher keeps it to the very edge."
        }
        "/material/base" => "Diffuse / base weight.",
        "/material/base_tint" => "Multiplies the base colour.",
        "/material/diffuse_roughness" => {
            "Oren-Nayar roughness of the diffuse base: 0 Lambert, 1 dusty."
        }
        "/material/metalness" => "0 dielectric, 1 metal (the base colour becomes the reflectance).",
        "/material/specular" => "Specular reflection weight.",
        "/material/specular_color" => "Specular tint.",
        "/material/specular_roughness" => {
            "Microfacet roughness of reflection and refraction: 0 mirror, 1 matte."
        }
        "/material/specular_ior" => {
            "Index of refraction: Fresnel reflectance of dielectrics and the bending of glass (1.5 glass, 1.33 water)."
        }
        "/material/specular_anisotropy" => {
            "Stretches the highlight along the tangent (brushed metal)."
        }
        "/material/specular_rotation" => "Turns the anisotropy direction.",
        "/material/transmission" => {
            "Fraction of the dielectric base that refracts through instead of diffusing: 1 is glass."
        }
        "/material/transmission_color" => {
            "Glass tint: at the interface when Transmission depth is 0, else the colour left after travelling that depth inside."
        }
        "/material/transmission_depth" => {
            "Distance over which light inside the glass is tinted to Transmission color (Beer-Lambert); 0 tints at the surface only."
        }
        "/material/transmission_extra_roughness" => {
            "Extra roughness of refraction only (frosted glass)."
        }
        "/material/sheen" => "Velvet-like sheen weight at grazing angles.",
        "/material/sheen_color" => "Sheen colour.",
        "/material/sheen_roughness" => "Sheen spread.",
        "/material/coat" => "Clear coat layer weight (lacquer over the base).",
        "/material/coat_color" => "Coat tint (absorbs what passes through it).",
        "/material/coat_roughness" => "Coat microfacet roughness.",
        "/material/coat_ior" => "Coat index of refraction.",
        "/material/coat_affect_color" => "How much the coat darkens and saturates the base.",
        "/material/coat_affect_roughness" => "How much the coat roughness roughens the base.",
        "/material/thin_film_thickness" => {
            "Thin-film interference thickness in nanometres (soap bubble, oil): 0 off."
        }
        "/material/thin_film_ior" => "Thin-film index of refraction.",
        "/material/emission" => "Emitted light strength.",
        "/material/emission_color" => "Emitted light colour.",
        "/render/iterations" => {
            "Fractal iterations per distance estimate: more resolves finer detail and costs proportionally."
        }
        "/render/max_steps" => {
            "Sphere-tracing step budget per ray. A ray that runs out of steps is unresolved: a camera ray shows the background (the OFX Direct preview keeps a sub-pixel near miss), bounce and shadow rays bring no light; the status bar reports the share. Steps a ray does not need cost nothing."
        }
        "/render/glass_probes" => {
            "Probes per exit through glass: the most steps an exit march takes and its shortest step (the object's size over this), so the thinnest interior wall or cavity a refracted ray can find. Fields that are zero inside (Mandelbulb and other escape-time fractals) only probe; signed fields (KIFS, Kleinians) march their distance but never step shorter. More finds finer walls and costs proportionally (4096 vs 256: 10-30x on a glass Mandelbulb). An exit it cannot find ends the path and is counted as unresolved."
        }
        "/render/hit_epsilon" => {
            "Surface detail: how close a ray must come to count as a hit, relative to the pixel footprint (0.008 is one pixel). Smaller resolves finer detail and needs more steps."
        }
        "/render/step_factor" => {
            "Sphere-tracing step as a fraction of the distance estimate: lower is safer on fractals that overestimate, slower."
        }
        "/render/max_bounces" => {
            "Path tracing bounces: indirect light, reflections and refractions (glass needs several)."
        }
        "/render/exposure_stops" => "Exposure in stops before the display transform.",
        "/render/saturation" => "Saturation of the displayed image (1 unchanged).",
        "/render/reinhard" => "Legacy Reinhard tone curve instead of the OCIO display transform.",
        "/render/denoise/enabled" => {
            "OIDN denoising of the displayed image; the raw samples are kept."
        }
        "/render/denoise/interval" => {
            "Denoise every N samples while rendering (0: only when the render completes)."
        }
        "/render/denoise/mode" => {
            "Guide images for OIDN: colour only, + albedo, + albedo and normal (sharpest)."
        }
        "/render/denoise/quality" => "OIDN quality: faster or sharper.",
        "/render/adaptive/enabled" => {
            "Adaptive sampling: stop sampling tiles whose noise is below the threshold."
        }
        "/render/adaptive/noise_threshold" => {
            "Relative noise a tile must fall under to stop; lower is cleaner and slower."
        }
        "/render/adaptive/min_samples" => "Samples every pixel takes before its tile may stop.",
        "/transform/position" => "Position in the parent's space.",
        "/transform/rotation_degrees" => "Rotation in degrees.",
        "/transform/scale" => "Scale per axis.",
        "/transform/pivot" => "Point the rotation and scale turn around.",
        "/visible" => "Show this node in renders.",
        "/locked" => "Lock against edits in the viewport and the editors.",
        "/solo" => "Render only soloed nodes.",
        "/start" => "First frame this node exists on.",
        "/end" => "Last frame this node exists on.",
        _ => return None,
    })
}

/// Why a node's parameter does nothing in its current mode (shown greyed in the Attribute
/// Editor), or None when it is in use. `value` reads the node's other attributes. The rules follow
/// what `Scene::pack` and the kernels read: Julia only for Mandelbulb / Mandelbox, the trap
/// parameters per colouring mode, a Hybrid's sub-formulas only when one of its steps runs them.
pub(crate) fn inactive_reason<'a>(
    path: &str,
    value: impl Fn(&str) -> Option<&'a Value>,
) -> Option<String> {
    // `path` is `prefix` or one of its components; no allocation (runs per row per frame).
    let under = |prefix: &str| {
        path.strip_prefix(prefix)
            .is_some_and(|rest| rest.is_empty() || rest.starts_with('/'))
    };
    if under("/julia") || under("/trap_point") || under("/trap_axis") || under("/trap_scale") {
        if under("/julia") {
            let family = value("/formula")?.as_object()?.keys().next()?.clone();
            return (!matches!(family.as_str(), "Mandelbulb" | "Mandelbox"))
                .then(|| format!("The {family} formula has no Julia mode"));
        }
        let coloring = value("/coloring")?.as_str()?;
        // gpu.rs trap3 / hit_palette: Radius reads none, TrapOrigin only the scale, TrapPlane
        // all three, TrapPoint the point and the scale.
        let used = match coloring {
            "Radius" => false,
            "TrapOrigin" => under("/trap_scale"),
            "TrapPoint" => !under("/trap_axis"),
            _ => true,
        };
        return (!used).then(|| format!("Not used by the {coloring} colouring"));
    }
    let rest = path.strip_prefix("/formula/Hybrid/")?;
    // A hybrid packs only the sub-formulas' shape parameters: one rotation for the whole hybrid
    // (the bulb's) and its own bailout (`Scene::pack`, Formula::Hybrid).
    let hybrid_own = |p: &str| {
        rest.strip_prefix(p)
            .is_some_and(|r| r.is_empty() || r.starts_with('/'))
    };
    if hybrid_own("mandelbox/rotation_degrees") || hybrid_own("kifs/rotation_degrees") {
        return Some("A hybrid turns by the Mandelbulb rotation".into());
    }
    if hybrid_own("bulb/bailout") {
        return Some("A hybrid uses its own bailout".into());
    }
    let step = match rest.split('/').next()? {
        // The bulb's rotation turns the whole hybrid (`Scene::pack`: set_rotation).
        "bulb" if hybrid_own("bulb/rotation_degrees") => return None,
        "bulb" => "Mandelbulb",
        "mandelbox" => "Mandelbox",
        "kifs" => "KifsFold",
        "apollonian_scale" => "Inversion",
        _ => return None,
    };
    let steps = value("/formula")?.get("Hybrid")?.get("steps")?.as_array()?;
    let runs = steps.iter().any(|s| s.as_str() == Some(step))
        // No step on: the kernel runs a Mandelbulb step.
        || (step == "Mandelbulb" && steps.iter().all(|s| s.as_str() == Some("Off")));
    (!runs).then(|| format!("No hybrid step is {step}"))
}

/// RGB colours: exactly the attributes `Scene::pack` writes with `put_rgb`.
fn is_color_attribute(path: &str) -> bool {
    matches!(
        path,
        "/lighting/sun_color"
            | "/lighting/sky_horizon"
            | "/lighting/sky_zenith"
            | "/material/base_color"
            | "/material/base_tint"
            | "/material/specular_color"
            | "/material/transmission_color"
            | "/material/sheen_color"
            | "/material/coat_color"
            | "/material/emission_color"
            | "/material/facing/color"
    )
}

/// A number clamped to the attribute's hard limits; other values pass through.
/// Tangent for a value written at `frame`: a key that already exists keeps its out kind (editing a
/// value must not reset its interpolation), a new key takes `new` (Settings > Animation).
fn key_tan(ch: &Channel, frame: f64, new: Tan) -> Tan {
    ch.keys()
        .iter()
        .find(|k| k.t() == frame)
        .map_or(new, |k| k.out.kind)
}

fn clamp_to_range(path: &str, value: Value) -> Value {
    match (attribute_range(path), value.as_f64()) {
        (Some((min, max)), Some(v)) if v < min || v > max => {
            numeric_like(&value, v.clamp(min, max))
        }
        _ => value,
    }
}

/// `v` as an integer when `template` is one and `v` is whole, else as a float: a limit never
/// rounds away (0 clamped to a 0.0005 floor stays 0.0005).
fn numeric_like(template: &Value, v: f64) -> Value {
    match (template.is_u64() || template.is_i64(), v.fract() == 0.0) {
        (true, true) if v >= 0.0 => json!(v as u64),
        (true, true) => json!(v as i64),
        _ => json!(v),
    }
}

fn scene_json(scene: &Scene) -> Result<Value, String> {
    let mut v = serde_json::to_value(scene).map_err(|e| e.to_string())?;
    if let Some(o) = v.as_object_mut() {
        for key in [
            "document",
            "animation",
            "objects",
            "lights",
            "world_render",
            "object_world",
        ] {
            o.remove(key);
        }
    }
    Ok(v)
}
fn branches(kind: WorldKind) -> &'static [&'static str] {
    match kind {
        WorldKind::Fractal => &[
            "formula",
            "julia",
            "palette",
            "coloring",
            "trap_point",
            "trap_axis",
            "trap_scale",
            "render",
        ],
        WorldKind::Camera => &["camera"],
        WorldKind::DirectionalLight => &["lighting"],
        WorldKind::Environment => &["environment", "lighting"],
        WorldKind::Material => &["material"],
        WorldKind::Group => &[],
        WorldKind::RenderSettings => &["render"],
        WorldKind::QualitySettings => &["quality"],
        WorldKind::ViewportSettings => &["viewport"],
    }
}
fn discover(value: &Value, path: &str, attrs: &mut Attrs) {
    match value {
        Value::Object(map) => {
            if path == "/formula" || path == "/material/facing" {
                attrs.set(path, to_attr(value));
            }
            for (k, v) in map {
                discover(
                    v,
                    &format!("{path}/{}", k.replace('~', "~0").replace('/', "~1")),
                    attrs,
                );
            }
        }
        _ => attrs.set(path, to_attr(value)),
    }
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
fn numeric(v: &Value) -> bool {
    v.is_number()
        || v.as_array()
            .is_some_and(|a| !a.is_empty() && a.iter().all(Value::is_number))
}
fn set_pointer(root: &mut Value, path: &str, value: Value, create: bool) -> Result<(), String> {
    if path.is_empty() {
        *root = value;
        return Ok(());
    }
    let parts: Vec<_> = path
        .trim_start_matches('/')
        .split('/')
        .map(|s| s.replace("~1", "/").replace("~0", "~"))
        .collect();
    fn put(root: &mut Value, parts: &[String], value: Value, create: bool) -> Result<(), String> {
        if parts.is_empty() {
            *root = value;
            return Ok(());
        }
        if let Value::Array(a) = root {
            let i: usize = parts[0].parse().map_err(|_| "Invalid array component")?;
            let slot = a.get_mut(i).ok_or("Unavailable array component")?;
            return put(slot, &parts[1..], value, create);
        }
        if root.is_null() && create {
            *root = json!({});
        }
        let map = root.as_object_mut().ok_or("Unavailable parameter")?;
        if !map.contains_key(&parts[0]) {
            if !create {
                return Err("Unavailable parameter".into());
            }
            map.insert(parts[0].clone(), Value::Null);
        }
        put(
            map.get_mut(&parts[0]).ok_or("Unavailable parameter")?,
            &parts[1..],
            value,
            create,
        )
    }
    put(root, &parts, value, create)
}

impl WorldEditor {
    fn create_quality_profile(
        &mut self,
        source: NodeId,
        name: String,
        role: crate::render_profiles::CatalogRole,
        target: Option<NodeId>,
    ) -> Result<(), String> {
        if name.trim().is_empty() {
            return Err("Profile name cannot be empty".into());
        }
        if role == crate::render_profiles::CatalogRole::Template && target.is_some() {
            return Err("Instantiate a template before assigning it".into());
        }
        self.document
            .require_settings(source, WorldKind::QualitySettings)?;
        let mut data = self.document.node(source)?.clone();
        data["parent"] = Value::Null;
        data["children"] = json!([]);
        let roots = self.insert_fragment(HashMap::from([(source.to_string(), data)]))?;
        let quality = *roots.first().ok_or("Cloned quality missing")?;
        self.document.node_mut(quality)?["name"] = json!(name.trim());
        self.document.node_mut(quality)?["metadata"]["catalog_role"] = json!(role);
        self.selection = Some(quality);
        self.selected = vec![quality];
        if let Some(render) = target {
            self.document
                .require_live_settings(render, WorldKind::RenderSettings)?;
            self.document.assert_unlocked(render)?;
            self.document.set_attribute(
                render,
                "/render/quality_id",
                json!(quality),
                f64::from(self.document.first),
                false,
                self.new_key,
            )?;
        }
        Ok(())
    }
    fn create_render_profile(
        &mut self,
        source: NodeId,
        name: String,
        role: crate::render_profiles::CatalogRole,
        target: Option<crate::render_profiles::ProfileTarget>,
    ) -> Result<(), String> {
        if name.trim().is_empty() {
            return Err("Profile name cannot be empty".into());
        }
        if role == crate::render_profiles::CatalogRole::Template && target.is_some() {
            return Err("Instantiate a template before assigning it".into());
        }
        self.document
            .require_settings(source, WorldKind::RenderSettings)?;
        let quality = self.document.render_quality(source)?;
        let mut nodes = HashMap::from([
            (source.to_string(), self.document.node(source)?.clone()),
            (quality.to_string(), self.document.node(quality)?.clone()),
        ]);
        // A settings subgraph never includes unrelated scene children or parents.
        for data in nodes.values_mut() {
            data["parent"] = Value::Null;
            data["children"] = json!([]);
        }
        let roots = self.insert_fragment(nodes)?;
        let render = roots
            .iter()
            .copied()
            .find(|id| {
                self.document
                    .info(*id)
                    .is_ok_and(|n| n.kind == WorldKind::RenderSettings)
            })
            .ok_or("Cloned render node missing")?;
        let quality = self.document.render_quality(render)?;
        self.document.node_mut(render)?["name"] = json!(name.trim());
        self.document.node_mut(quality)?["name"] = json!(format!("{} Quality", name.trim()));
        for id in [render, quality] {
            self.document.node_mut(id)?["metadata"]["catalog_role"] = json!(role);
        }
        self.selection = Some(render);
        self.selected = vec![render];
        if let Some(target) = target {
            if let Some(path) = target.viewport_path() {
                let viewport = self.document.viewport_settings_id()?;
                self.document.assert_unlocked(viewport)?;
                self.document.set_attribute(
                    viewport,
                    path,
                    json!(render),
                    f64::from(self.document.first),
                    false,
                    self.new_key,
                )?;
            } else {
                self.apply(WorldCommand::SetOutputRender(render))?;
            }
        }
        Ok(())
    }
}
impl WorldDocument {
    fn require_settings(&self, id: NodeId, kind: WorldKind) -> Result<(), String> {
        if self.info(id)?.kind != kind {
            return Err(format!("Node {id} must be {kind:?}"));
        }
        Ok(())
    }
    fn require_live_settings(&self, id: NodeId, kind: WorldKind) -> Result<(), String> {
        self.require_settings(id, kind)?;
        if self.catalog_role(id)? != Some(crate::render_profiles::CatalogRole::Profile) {
            return Err(format!("Instantiate template {id} before assigning it"));
        }
        Ok(())
    }
    pub fn catalog_role(
        &self,
        id: NodeId,
    ) -> Result<Option<crate::render_profiles::CatalogRole>, String> {
        if !crate::render_profiles::setting_kind(self.info(id)?.kind) {
            return Ok(None);
        }
        serde_json::from_value(self.node(id)?["metadata"]["catalog_role"].clone())
            .map(Some)
            .map_err(|e| format!("Invalid settings catalog role for {id}: {e}"))
    }
    pub fn render_profiles(
        &self,
        role: Option<crate::render_profiles::CatalogRole>,
    ) -> Vec<WorldNodeInfo> {
        self.nodes()
            .into_iter()
            .filter(|node| {
                node.kind == WorldKind::RenderSettings
                    && role
                        .is_none_or(|role| self.catalog_role(node.id).ok().flatten() == Some(role))
            })
            .collect()
    }
    fn settings_bus_id(&self, key: &str, kind: WorldKind) -> Result<NodeId, String> {
        let id = self
            .graph
            .bus_slots
            .get(key)
            .and_then(Value::as_str)
            .and_then(NodeId::parse)
            .ok_or_else(|| format!("Missing or invalid {key} UUID"))?;
        self.require_live_settings(id, kind)?;
        Ok(id)
    }
    pub fn viewport_settings_id(&self) -> Result<NodeId, String> {
        self.settings_bus_id("viewport_settings", WorldKind::ViewportSettings)
    }
    pub fn output_render_profile(&self) -> Result<NodeId, String> {
        self.settings_bus_id("output_render", WorldKind::RenderSettings)
    }
    fn static_settings_value(&self, id: NodeId, path: &str) -> Result<Value, String> {
        let attrs = self.attrs(id)?;
        if attrs.anim(path).is_some() || attrs.conn(path).is_some() {
            return Err(format!("Settings attribute {path} must be static"));
        }
        attrs
            .get(path)
            .cloned()
            .map(attr_json)
            .ok_or_else(|| format!("Missing settings attribute {path}"))
    }
    fn static_settings_reference(&self, id: NodeId, path: &str) -> Result<NodeId, String> {
        self.static_settings_value(id, path)?
            .as_str()
            .and_then(NodeId::parse)
            .ok_or_else(|| format!("Settings reference {path} must be a UUID"))
    }
    pub fn render_quality(&self, render: NodeId) -> Result<NodeId, String> {
        self.require_settings(render, WorldKind::RenderSettings)?;
        let id = self.static_settings_reference(render, "/render/quality_id")?;
        self.require_settings(id, WorldKind::QualitySettings)?;
        Ok(id)
    }
    fn settings_choices(&self, consumer: NodeId, path: &str) -> Option<Vec<Value>> {
        let kind = crate::render_profiles::reference_kind(path)?;
        let template = self.catalog_role(consumer).ok().flatten()
            == Some(crate::render_profiles::CatalogRole::Template);
        Some(
            self.nodes()
                .into_iter()
                .filter(|node| {
                    node.kind == kind
                        && (template
                            || self.catalog_role(node.id).ok().flatten()
                                == Some(crate::render_profiles::CatalogRole::Profile))
                })
                .map(|node| json!(node.id))
                .collect(),
        )
    }
    fn settings_template(&self, kind: WorldKind, scene: &Scene) -> Result<Value, String> {
        use crate::render_profiles::{quality_values, render_values, viewport_values};
        let render = serde_json::to_value(&scene.render).map_err(|e| e.to_string())?;
        match kind {
            WorldKind::QualitySettings => Ok(json!({"quality":quality_values(&render)})),
            WorldKind::RenderSettings => {
                let quality = self
                    .nodes()
                    .into_iter()
                    .find(|node| {
                        node.kind == WorldKind::QualitySettings
                            && self.catalog_role(node.id).ok().flatten()
                                == Some(crate::render_profiles::CatalogRole::Profile)
                    })
                    .map(|node| node.id)
                    .ok_or("Create QualitySettings before RenderSettings")?;
                Ok(json!({"render":render_values(&render, quality)}))
            }
            WorldKind::ViewportSettings => {
                let profiles =
                    self.render_profiles(Some(crate::render_profiles::CatalogRole::Profile));
                let still = profiles
                    .first()
                    .ok_or("Create a RenderSettings profile before ViewportSettings")?
                    .id;
                Ok(json!({"viewport":viewport_values(still, still)}))
            }
            _ => Err("Not a settings node".into()),
        }
    }
    fn seed_render_profiles(&mut self, scene: &Scene) -> Result<(), String> {
        use crate::render_profiles::RenderMethod;
        let output_quality =
            self.insert(WorldKind::QualitySettings, "Output Quality", scene, None)?;
        let output = self.insert(WorldKind::RenderSettings, "Output", scene, None)?;
        self.graph
            .bus_slots
            .insert("output_render".into(), json!(output));
        let still_quality =
            self.insert(WorldKind::QualitySettings, "Still Quality", scene, None)?;
        let still = self.insert(WorldKind::RenderSettings, "Still", scene, None)?;
        self.set_attribute(
            still,
            "/render/quality_id",
            json!(still_quality),
            0.0,
            false,
            Tan::Constant,
        )?;
        let moving_quality =
            self.insert(WorldKind::QualitySettings, "Moving Quality", scene, None)?;
        let moving = self.insert(WorldKind::RenderSettings, "Moving", scene, None)?;
        self.set_attribute(
            moving,
            "/render/quality_id",
            json!(moving_quality),
            0.0,
            false,
            Tan::Constant,
        )?;
        self.set_attribute(
            output,
            "/render/quality_id",
            json!(output_quality),
            0.0,
            false,
            Tan::Constant,
        )?;
        for (id, path, value) in [
            (moving, "/render/method", json!(RenderMethod::Fast)),
            (moving_quality, "/quality/samples", json!(64)),
            (moving_quality, "/quality/resolution_scale", json!(0.5)),
            (moving_quality, "/quality/max_bounces", json!(2)),
        ] {
            self.set_attribute(id, path, value, 0.0, false, Tan::Constant)?;
        }
        let viewport = self.insert(WorldKind::ViewportSettings, "Viewport", scene, None)?;
        self.set_attribute(
            viewport,
            "/viewport/moving_id",
            json!(moving),
            0.0,
            false,
            Tan::Constant,
        )?;
        self.set_attribute(
            viewport,
            "/viewport/still_id",
            json!(still),
            0.0,
            false,
            Tan::Constant,
        )?;
        self.set_attribute(
            viewport,
            "/viewport/manual_id",
            json!(still),
            0.0,
            false,
            Tan::Constant,
        )?;
        self.graph
            .bus_slots
            .insert("viewport_settings".into(), json!(viewport));
        Ok(())
    }
    pub fn effective_render(
        &self,
        profile: NodeId,
        frame: f64,
    ) -> Result<crate::render_profiles::EffectiveRender, String> {
        use crate::render_profiles::{
            EffectiveRender, PROFILE_RENDER_FIELDS, QUALITY_RENDER_FIELDS, RenderMethod,
        };
        if !frame.is_finite() {
            return Err("Invalid render profile time".into());
        }
        let quality = self.render_quality(profile)?;
        let method: RenderMethod =
            serde_json::from_value(self.attribute_value(profile, "/render/method", frame)?)
                .map_err(|e| format!("Invalid render method: {e}"))?;
        let mut values = serde_json::Map::new();
        // World kernels read geometry iterations from each object's block. The transient root
        // block carries a neutral value; it must never proxy a particular fractal's geometry.
        values.insert(
            "iterations".into(),
            json!(Scene::preset(crate::params::FAMILY_BULB).render.iterations),
        );
        let evaluated_quality = json!({"quality":self.settings_values(quality, "quality", frame)?});
        for key in QUALITY_RENDER_FIELDS {
            let mut value = evaluated_quality["quality"][*key].clone();
            if value.is_null() {
                return Err(format!("Missing quality field {key}"));
            }
            // Numeric integer channels evaluate as float; round to the integer renderer contract.
            if matches!(*key, "max_steps" | "max_bounces" | "glass_probes") {
                let number = value
                    .as_f64()
                    .filter(|v| v.is_finite() && *v >= 0.0 && *v <= u32::MAX as f64)
                    .ok_or("Invalid integer quality field")?;
                value = json!(number.round() as u32);
            } else if *key == "adaptive" {
                let minimum = value["min_samples"]
                    .as_f64()
                    .filter(|v| v.is_finite() && *v >= 0.0 && *v <= 65536.0)
                    .ok_or("Invalid adaptive sample minimum")?;
                value["min_samples"] = json!(minimum.round() as u32);
            }
            values.insert((*key).into(), value);
        }
        let evaluated = json!({"render":self.settings_values(profile, "render", frame)?});
        for key in PROFILE_RENDER_FIELDS {
            values.insert((*key).into(), evaluated["render"][*key].clone());
        }
        let render: crate::scene::Render = serde_json::from_value(Value::Object(values))
            .map_err(|e| format!("Invalid effective RenderSettings: {e}"))?;
        let samples_value = self.attribute_value(quality, "/quality/samples", frame)?;
        let samples = samples_value
            .as_f64()
            .filter(|v| v.is_finite() && (1.0..=65536.0).contains(v))
            .ok_or("Samples must be from 1 to 65536")?
            .round() as u32;
        let resolution_scale =
            self.attribute_value(quality, "/quality/resolution_scale", frame)?
                .as_f64()
                .filter(|v| v.is_finite() && (0.0625..=1.0).contains(v))
                .ok_or("Resolution scale must be between 1/16 and 1")? as f32;
        if render.max_steps == 0
            || render.glass_probes == 0
            || render.max_bounces > 64
            || !render.hit_epsilon.is_finite()
            || render.hit_epsilon <= 0.0
            || !render.step_factor.is_finite()
            || !(0.0..=1.0).contains(&render.step_factor)
            || render.step_factor == 0.0
            || !render.adaptive.noise_threshold.is_finite()
            || !(0.0005..=1.0).contains(&render.adaptive.noise_threshold)
            || !(crate::scene::Adaptive::MIN_SAMPLES_FLOOR..=65536)
                .contains(&render.adaptive.min_samples)
        {
            return Err("QualitySettings contain invalid tracing or adaptive limits".into());
        }
        Ok(EffectiveRender {
            profile,
            quality,
            method,
            render,
            samples,
            resolution_scale,
        })
    }
    pub fn viewport_policy(
        &self,
        frame: f64,
    ) -> Result<crate::render_profiles::ViewportPolicy, String> {
        use crate::render_profiles::{ViewportMode, ViewportPolicy};
        if !frame.is_finite() {
            return Err("Invalid viewport policy time".into());
        }
        let id = self.viewport_settings_id()?;
        let value = |path: &str| self.static_settings_value(id, path);
        let reference = |path| -> Result<NodeId, String> {
            let target = self.static_settings_reference(id, path)?;
            self.require_live_settings(target, WorldKind::RenderSettings)?;
            Ok(target)
        };
        let number = |path: &str, min: f64, max: f64| -> Result<f32, String> {
            value(path)?
                .as_f64()
                .filter(|v| v.is_finite() && (min..=max).contains(v))
                .map(|v| v as f32)
                .ok_or_else(|| format!("Invalid viewport setting {path}"))
        };
        let boolean = |path: &str| {
            value(path)?
                .as_bool()
                .ok_or_else(|| format!("Invalid viewport setting {path}"))
        };
        let mode: ViewportMode =
            serde_json::from_value(value("/viewport/mode")?).map_err(|e| e.to_string())?;
        Ok(ViewportPolicy {
            mode,
            moving_id: reference("/viewport/moving_id")?,
            still_id: reference("/viewport/still_id")?,
            manual_id: reference("/viewport/manual_id")?,
            target_fps: number("/viewport/target_fps", 1.0, 240.0)?,
            settle_delay_ms: number("/viewport/settle_delay_ms", 0.0, 10000.0)?,
            batch_budget_ms: number("/viewport/batch_budget_ms", 0.1, 1000.0)?,
            paused: boolean("/viewport/paused")?,
            frozen: boolean("/viewport/frozen")?,
        })
    }
    pub fn viewport_render(
        &self,
        frame: f64,
        moving: bool,
    ) -> Result<crate::render_profiles::ViewportRender, String> {
        let (target, id) = self.viewport_policy(frame)?.selected(moving);
        Ok(crate::render_profiles::ViewportRender {
            target,
            effective: self.effective_render(id, frame)?,
        })
    }
    fn settings_values(&self, id: NodeId, branch: &str, frame: f64) -> Result<Value, String> {
        let mut value = self.node(id)?["gpu"][branch].clone();
        if !value.is_object() {
            return Err(format!("Settings schema {branch} missing"));
        }
        let mut schema = Attrs::new();
        discover(&value, &format!("/{branch}"), &mut schema);
        for (path, base) in schema.iter() {
            let mut current = self.attribute_value(id, path, frame)?;
            if let Some((min, max)) = attribute_range(path) {
                let number = current
                    .as_f64()
                    .filter(|v| v.is_finite() && (min..=max).contains(v))
                    .ok_or_else(|| format!("Settings value {path} is outside its limits"))?;
                if attr_json(base.clone()).is_i64() {
                    current = json!(number.round() as u64);
                }
            } else if attr_json(base.clone()).is_i64() {
                let number = current
                    .as_f64()
                    .filter(|v| v.is_finite() && *v >= 0.0 && *v <= u32::MAX as f64)
                    .ok_or_else(|| format!("Invalid integer settings value {path}"))?;
                current = json!(number.round() as u32);
            }
            set_pointer(&mut value, &path[branch.len() + 1..], current, false)?;
        }
        Ok(value)
    }
    pub fn validate_render_profiles(&self, frame: f64) -> Result<(), String> {
        // Routing and policy are static document choices. Runtime branching never authors them.
        for node in self.nodes() {
            let attrs = self.attrs(node.id)?;
            for (path, _) in attrs.iter() {
                if (crate::render_profiles::reference_kind(path).is_some()
                    || path.starts_with("/viewport/"))
                    && (attrs.anim(path).is_some() || attrs.conn(path).is_some())
                {
                    return Err(format!("Settings attribute {path} must be static"));
                }
            }
        }
        self.output_render_profile()?;
        self.viewport_policy(frame)?;
        for node in self
            .nodes()
            .into_iter()
            .filter(|node| crate::render_profiles::setting_kind(node.kind))
        {
            let role = self.catalog_role(node.id)?.ok_or("Settings role missing")?;
            match node.kind {
                WorldKind::RenderSettings => {
                    let quality = self.render_quality(node.id)?;
                    if role == crate::render_profiles::CatalogRole::Profile {
                        self.require_live_settings(quality, WorldKind::QualitySettings)?;
                    }
                    self.effective_render(node.id, frame)?;
                }
                WorldKind::ViewportSettings => {
                    let values = self.settings_values(node.id, "viewport", frame)?;
                    serde_json::from_value::<crate::render_profiles::ViewportMode>(
                        values["mode"].clone(),
                    )
                    .map_err(|e| e.to_string())?;
                    for key in ["paused", "frozen"] {
                        if !values[key].is_boolean() {
                            return Err(format!("Invalid viewport {key}"));
                        }
                    }
                    for path in [
                        "/viewport/moving_id",
                        "/viewport/still_id",
                        "/viewport/manual_id",
                    ] {
                        let id = self.static_settings_reference(node.id, path)?;
                        if role == crate::render_profiles::CatalogRole::Profile {
                            self.require_live_settings(id, WorldKind::RenderSettings)?;
                        } else {
                            self.require_settings(id, WorldKind::RenderSettings)?;
                        }
                    }
                }
                WorldKind::QualitySettings => {
                    let values = self.settings_values(node.id, "quality", frame)?;
                    for key in crate::render_profiles::QUALITY_RENDER_FIELDS
                        .iter()
                        .chain(["samples", "resolution_scale"].iter())
                    {
                        if values[*key].is_null() {
                            return Err(format!("Missing quality field {key}"));
                        }
                    }
                    serde_json::from_value::<crate::scene::Adaptive>(values["adaptive"].clone())
                        .map_err(|e| e.to_string())?;
                }
                _ => {}
            }
        }
        Ok(())
    }
    fn guard_settings_delete(&self, remove: &HashSet<NodeId>) -> Result<(), String> {
        for key in ["output_render", "viewport_settings"] {
            if self
                .graph
                .bus_slots
                .get(key)
                .and_then(Value::as_str)
                .and_then(NodeId::parse)
                .is_some_and(|id| remove.contains(&id))
            {
                return Err(format!("Reassign {key} before deleting its settings node"));
            }
        }
        for node in self
            .nodes()
            .into_iter()
            .filter(|node| !remove.contains(&node.id))
        {
            for attr in self.attributes(node.id, f64::from(self.first))? {
                if crate::render_profiles::reference_kind(&attr.path).is_some()
                    && attr
                        .value
                        .as_str()
                        .and_then(NodeId::parse)
                        .is_some_and(|id| remove.contains(&id))
                {
                    return Err(format!(
                        "Node {} still references this settings node through {}",
                        node.id, attr.path
                    ));
                }
            }
        }
        Ok(())
    }
    fn render_edit_paths(
        &self,
        render: NodeId,
        quality: NodeId,
        frame: f64,
    ) -> Result<Vec<(String, NodeId)>, String> {
        let mut paths = Vec::new();
        for id in [render, quality] {
            paths.extend(
                self.attributes(id, frame)?
                    .into_iter()
                    .filter(|a| {
                        a.component.is_none()
                            && (a.path.starts_with("/render/") || a.path.starts_with("/quality/"))
                            && a.path != "/render/quality_id"
                            && a.path != "/render/method"
                            && a.path != "/quality/samples"
                            && a.path != "/quality/resolution_scale"
                    })
                    .map(|a| (a.path, id)),
            );
        }
        Ok(paths)
    }
}

impl WorldDocument {
    pub fn from_scene(scene: &Scene) -> Self {
        if let Some(world) = &scene.document {
            return (**world).clone();
        }
        let mut document = Self {
            graph: SubnetFile {
                format_version: playa_graph::SUBNET_FORMAT_VERSION,
                id: NodeId::new().uuid(),
                nodes: HashMap::new(),
                bus_slots: HashMap::from([(
                    "world".into(),
                    scene_json(scene).expect("Scene serialization"),
                )]),
            },
            active_camera: None,
            active_environment: None,
            first: scene.animation.first,
            last: scene.animation.last,
            fps: scene.animation.fps,
            marks: BTreeMap::new(),
        };
        document
            .seed_render_profiles(scene)
            .expect("Valid render settings");
        document
            .graph
            .bus_slots
            .get_mut("world")
            .expect("World settings")
            .as_object_mut()
            .expect("Scene object")
            .remove("render");
        let fractal = document
            .insert(WorldKind::Fractal, &scene.name, scene, None)
            .expect("Valid scene");
        let camera = document
            .insert(WorldKind::Camera, "Camera", scene, None)
            .expect("Valid scene");
        document
            .insert(WorldKind::DirectionalLight, "Sun", scene, None)
            .expect("Valid scene");
        let environment = document
            .insert(WorldKind::Environment, "Environment", scene, None)
            .expect("Valid scene");
        let material = document
            .insert(WorldKind::Material, "Material", scene, None)
            .expect("Valid scene");
        document.node_mut(fractal).expect("Fractal")["material"] = json!(material);
        document.active_camera = Some(camera);
        document.active_environment = Some(environment);
        document
    }
    fn insert(
        &mut self,
        kind: WorldKind,
        name: &str,
        scene: &Scene,
        parent: Option<NodeId>,
    ) -> Result<NodeId, String> {
        let mut template = scene_json(scene)?;
        if crate::render_profiles::setting_kind(kind) {
            template = self.settings_template(kind, scene)?;
        } else if kind == WorldKind::Fractal {
            template["render"] = json!({"iterations":scene.render.iterations});
        }
        let mut attrs = Attrs::new();
        for branch in branches(kind) {
            if let Some(v) = template.get(*branch) {
                discover(v, &format!("/{branch}"), &mut attrs);
            }
        }
        for (path, value) in [
            ("/visible", json!(true)),
            ("/locked", json!(false)),
            ("/solo", json!(false)),
            ("/start", json!(self.first as f64)),
            ("/end", json!(self.last as f64 + 1.0)),
            ("/transform/position", json!(([0.0; 3]))),
            ("/transform/rotation_degrees", json!(([0.0; 3]))),
            ("/transform/scale", json!(([1.0; 3]))),
            ("/transform/pivot", json!(([0.0; 3]))),
        ] {
            attrs.set(path, to_attr(&value));
        }
        if kind == WorldKind::Camera {
            for path in [CAMERA_ORBIT_SPEED, CAMERA_ORBIT_PHASE] {
                attrs.set(path, to_attr(&json!(0.0)));
            }
        }
        if kind == WorldKind::Fractal {
            attrs.set("/transform/position", to_attr(&json!(scene.object.offset)));
            attrs.set(
                "/transform/rotation_degrees",
                to_attr(&json!(scene.object.rotation_degrees.map(|v| -v))),
            );
            attrs.set(
                "/transform/scale",
                to_attr(&json!(([scene.object.scale; 3]))),
            );
        }
        let id = NodeId::new();
        let metadata = if crate::render_profiles::setting_kind(kind) {
            json!({"catalog_role":crate::render_profiles::CatalogRole::Profile})
        } else {
            json!({})
        };
        self.graph.nodes.insert(id.to_string(),json!({"type":kind,"name":name,"parent":parent,"children":[],"host":serde_json::to_value(attrs).map_err(|e|e.to_string())?,"gpu":template,"discrete":{},"metadata":metadata,"material":null,"environment_revision":scene.environment.revision}));
        let order = self
            .graph
            .bus_slots
            .entry("layer_order".into())
            .or_insert_with(|| json!([]));
        if let Some(order) = order.as_array_mut() {
            order.push(json!(id));
        }
        Ok(id)
    }
    #[cfg(test)]
    pub fn runtime_graph(&self) -> Result<Graph, String> {
        let mut graph = Graph::new();
        graph.id = self.graph.id;
        for (id, data) in &self.graph.nodes {
            let id = NodeId::parse(id).ok_or("Invalid node UUID")?;
            graph.nodes.insert(
                id,
                Node {
                    id,
                    data: RustBox::from_json_value(data.clone(), Default::default()),
                },
            );
        }
        graph.rebuild_index();
        Ok(graph)
    }
    fn node(&self, id: NodeId) -> Result<&Value, String> {
        self.graph
            .nodes
            .get(&id.to_string())
            .ok_or_else(|| format!("Node {id} no longer exists"))
    }
    fn node_mut(&mut self, id: NodeId) -> Result<&mut Value, String> {
        self.graph
            .nodes
            .get_mut(&id.to_string())
            .ok_or_else(|| format!("Node {id} no longer exists"))
    }
    fn attrs(&self, id: NodeId) -> Result<Attrs, String> {
        let node = self.node(id)?;
        serde_json::from_value(node["host"].clone()).map_err(|e| e.to_string())
    }
    fn store_attrs(&mut self, id: NodeId, attrs: &Attrs) -> Result<(), String> {
        self.node_mut(id)?["host"] = serde_json::to_value(attrs).map_err(|e| e.to_string())?;
        Ok(())
    }
    pub fn info(&self, id: NodeId) -> Result<WorldNodeInfo, String> {
        let data = self.node(id)?;
        let a = self.attrs(id)?;
        Ok(WorldNodeInfo {
            id,
            kind: serde_json::from_value(data["type"].clone()).map_err(|e| e.to_string())?,
            name: data["name"].as_str().unwrap_or("").into(),
            parent: data["parent"].as_str().and_then(NodeId::parse),
            visible: a.get_bool_or("/visible", true),
            locked: a.get_bool_or("/locked", false),
            solo: a.get_bool_or("/solo", false),
            start: f64::from(a.get_float_or("/start", 0.0)),
            end: f64::from(a.get_float_or("/end", self.last as f32 + 1.0)),
        })
    }
    /// Single capability check for the Material reference exposed by the node
    /// schema, shared by the Attribute Editor, gallery and assignment command.
    /// A storage placeholder exists on every node; it is not itself a capability.
    /// `ids` and all their descendants, as stored node data (UUID -> data).
    pub fn fragment(&self, ids: &[NodeId]) -> Result<HashMap<String, Value>, String> {
        let mut set: HashSet<NodeId> = ids.iter().copied().collect();
        let nodes = self.nodes();
        loop {
            let n = set.len();
            for node in &nodes {
                if node.parent.is_some_and(|p| set.contains(&p)) {
                    set.insert(node.id);
                }
            }
            if n == set.len() {
                break;
            }
        }
        let mut out = HashMap::new();
        for id in set {
            out.insert(id.to_string(), self.node(id)?.clone());
        }
        Ok(out)
    }
    /// References of a fragment that point outside it and are absent from this document
    /// (material, parent), by node key. A paste clears exactly these; the UI reports them.
    pub fn unresolved_references(
        &self,
        nodes: &HashMap<String, Value>,
    ) -> Vec<(String, &'static str)> {
        let resolves = |id: &str| nodes.contains_key(id) || self.graph.nodes.contains_key(id);
        let mut out = Vec::new();
        for (key, data) in nodes {
            for field in ["material", "parent"] {
                if data[field].as_str().is_some_and(|id| !resolves(id)) {
                    out.push((key.clone(), field));
                }
            }
        }
        out
    }
    /// The system-clipboard text of `ids` (with descendants): see `parse_clipboard`.
    pub fn copy_fragment(&self, ids: &[NodeId]) -> Result<String, String> {
        if ids.is_empty() {
            return Err("Nothing selected to copy".into());
        }
        serde_json::to_string(
            &json!({ CLIPBOARD_KEY: CLIPBOARD_VERSION, "nodes": self.fragment(ids)? }),
        )
        .map_err(|e| e.to_string())
    }
    /// Every node with a material reference (fractals), in document order.
    pub fn material_consumers(&self) -> Vec<NodeId> {
        self.nodes()
            .into_iter()
            .map(|n| n.id)
            .filter(|id| self.supports_material(*id))
            .collect()
    }
    pub fn supports_material(&self, id: NodeId) -> bool {
        self.node(id)
            .is_ok_and(|data| data["type"].as_str() == Some("Fractal"))
    }

    /// Resolve a fractal's material UUID without building a scene snapshot.
    pub fn assigned_material(&self, id: NodeId) -> Result<Option<NodeId>, String> {
        let data = self.node(id)?;
        let material = data["material"].as_str().and_then(NodeId::parse);
        if let Some(material) = material {
            let target = self.node(material)?;
            if serde_json::from_value::<WorldKind>(target["type"].clone())
                .map_err(|e| e.to_string())?
                != WorldKind::Material
            {
                return Err("Assigned UUID is not a material".into());
            }
        }
        Ok(material)
    }
    pub fn nodes(&self) -> Vec<WorldNodeInfo> {
        let mut nodes: Vec<_> = self
            .graph
            .nodes
            .keys()
            .filter_map(|id| NodeId::parse(id).and_then(|id| self.info(id).ok()))
            .collect();
        let order: Vec<NodeId> = self
            .graph
            .bus_slots
            .get("layer_order")
            .and_then(Value::as_array)
            .map(|v| {
                v.iter()
                    .filter_map(|v| v.as_str().and_then(NodeId::parse))
                    .collect()
            })
            .unwrap_or_default();
        nodes.sort_by(|a, b| {
            order
                .iter()
                .position(|id| *id == a.id)
                .unwrap_or(usize::MAX)
                .cmp(
                    &order
                        .iter()
                        .position(|id| *id == b.id)
                        .unwrap_or(usize::MAX),
                )
                .then(a.name.cmp(&b.name))
                .then(a.id.cmp(&b.id))
        });
        nodes
    }
    pub fn metadata(&self, id: NodeId) -> Result<Value, String> {
        Ok(self.node(id)?["metadata"].clone())
    }
    pub fn attributes(&self, id: NodeId, frame: f64) -> Result<Vec<WorldAttribute>, String> {
        let attrs = self.attrs(id)?;
        let kind = self.info(id)?.kind;
        let mut out = vec![];
        for (path, _) in attrs.iter() {
            if path.starts_with("/_navigation/") {
                continue;
            }
            if kind == WorldKind::Fractal
                && path.starts_with("/render/")
                && path != "/render/iterations"
            {
                continue;
            }
            let keyable = !matches!(path.as_str(), "/locked" | "/solo" | "/start" | "/end")
                && crate::render_profiles::reference_kind(path).is_none()
                && !path.starts_with("/viewport/");
            out.push(WorldAttribute {
                path: path.clone(),
                label: attribute_label(path),
                value: self.attribute_value(id, path, frame)?,
                frames: attrs.key_frames(path),
                keyable,
                component: None,
                choices: self
                    .settings_choices(id, path)
                    .unwrap_or_else(|| attribute_choices(path)),
                range: attribute_range(path),
                slider: attribute_slider(path),
                color: is_color_attribute(path),
                default: keyable.then(|| attribute_default(kind, path)).flatten(),
            });
        }
        if self.supports_material(id) {
            out.push(WorldAttribute {
                path: "/material_id".into(),
                label: "Material".into(),
                value: self.node(id)?["material"].clone(),
                frames: vec![],
                keyable: false,
                component: None,
                choices: self
                    .nodes()
                    .into_iter()
                    .filter(|n| n.kind == WorldKind::Material)
                    .map(|n| json!(n.id))
                    .collect(),
                range: None,
                slider: None,
                color: false,
                default: None,
            });
        }
        let parents = out.clone();
        for parent in parents {
            if let Some(values) = parent
                .value
                .as_array()
                .filter(|v| v.iter().all(Value::is_number))
            {
                for (component, value) in values.iter().enumerate() {
                    let mut attr = parent.clone();
                    attr.path = format!("{}/{}", parent.path, component);
                    attr.default = parent
                        .default
                        .as_ref()
                        .and_then(|d| d.get(component))
                        .cloned();
                    attr.label = format!(
                        "{} / {}",
                        parent.label,
                        ["X", "Y", "Z", "W"]
                            .get(component)
                            .copied()
                            .unwrap_or("Component")
                    );
                    attr.value = value.clone();
                    attr.component = Some(component);
                    attr.choices.clear();
                    attr.frames = attrs
                        .anim(&parent.path)
                        .and_then(|anim| anim.channels.get(component))
                        .map(|ch| ch.keys().iter().map(|k| k.t()).collect())
                        .unwrap_or_default();
                    out.push(attr);
                }
            }
        }
        out.sort_by(|a, b| a.path.cmp(&b.path));
        Ok(out)
    }
    fn is_discrete(&self, id: NodeId, path: &str) -> Result<bool, String> {
        let a = self.attrs(id)?;
        Ok(!numeric(&attr_json(
            a.get(path).cloned().ok_or("Unavailable parameter")?,
        )) || self.node(id)?["discrete"].get(path).is_some())
    }
    pub fn attribute_value(&self, id: NodeId, path: &str, frame: f64) -> Result<Value, String> {
        if let Some((parent, component)) = self.component_path(id, path)? {
            let value = self.attribute_value(id, &parent, frame)?;
            return value
                .get(component)
                .cloned()
                .ok_or("Invalid component".into());
        }
        let value = self.resolve_attribute(id, path, frame, &mut HashSet::new())?;
        self.with_navigation_offset(id, path, value)
    }
    fn resolve_attribute(
        &self,
        id: NodeId,
        path: &str,
        frame: f64,
        visiting: &mut HashSet<(NodeId, String)>,
    ) -> Result<Value, String> {
        if !frame.is_finite() {
            return Err("Invalid frame".into());
        }
        if path == "/material_id" {
            return Ok(self.node(id)?["material"].clone());
        }
        if !visiting.insert((id, path.into())) {
            return Err("Attribute connection cycle".into());
        }
        let a = self.attrs(id)?;
        if !a.is_animated(path) {
            for parent in ["/formula", "/julia", "/material/facing"] {
                if path.starts_with(&format!("{parent}/")) && a.is_animated(parent) {
                    let v = self.resolve_attribute(id, parent, frame, visiting)?;
                    if let Some(value) = v.pointer(&path[parent.len()..]) {
                        return Ok(value.clone());
                    }
                }
            }
        }
        if let Some(conn) = a.conn(path) {
            let source = NodeId(conn.source_layer);
            let value = self.resolve_attribute(source, &conn.source_key, frame, visiting)?;
            return self.with_navigation_offset(source, &conn.source_key, value);
        }
        let value = a
            .eval_at(path, frame)
            .ok_or_else(|| format!("Unavailable parameter {path}"))?;
        if let Some(dict) = self.node(id)?["discrete"]
            .get(path)
            .and_then(Value::as_array)
        {
            let index = attr_json(value).as_f64().unwrap_or(0.0).round().max(0.0) as usize;
            return dict
                .get(index)
                .cloned()
                .ok_or_else(|| format!("Invalid discrete key {path}"));
        }
        let mut value = attr_json(value);
        if let Some(anim) = a.anim(path) {
            if let Some(values) = value.as_array_mut() {
                if let Some(base) = a
                    .get(path)
                    .cloned()
                    .map(attr_json)
                    .and_then(|v| v.as_array().cloned())
                {
                    for (index, ch) in anim.channels.iter().enumerate() {
                        if ch.is_empty() {
                            if let (Some(slot), Some(base)) =
                                (values.get_mut(index), base.get(index))
                            {
                                *slot = base.clone();
                            }
                        }
                    }
                }
            }
        }
        Ok(value)
    }
    fn component_path(&self, id: NodeId, path: &str) -> Result<Option<(String, usize)>, String> {
        let attrs = self.attrs(id)?;
        if attrs.contains(path) {
            return Ok(None);
        }
        if let Some((parent, last)) = path.rsplit_once('/') {
            if let Ok(component) = last.parse::<usize>() {
                if let Some(value) = attrs.get(parent) {
                    if attr_json(value.clone())
                        .as_array()
                        .is_some_and(|v| component < v.len())
                    {
                        return Ok(Some((parent.into(), component)));
                    }
                }
            }
        }
        Ok(None)
    }
    fn set_attribute(
        &mut self,
        id: NodeId,
        path: &str,
        value: Value,
        frame: f64,
        key: bool,
        kind: Tan,
    ) -> Result<(), String> {
        if !frame.is_finite() {
            return Err("Invalid key time".into());
        }
        if key && path.starts_with("/viewport/") {
            return Err("Viewport policy attributes cannot be animated".into());
        }
        if let Some(expected) = crate::render_profiles::reference_kind(path) {
            if key {
                return Err("Settings references cannot be animated".into());
            }
            let target = value
                .as_str()
                .and_then(NodeId::parse)
                .ok_or("Settings reference must be a UUID")?;
            if self.catalog_role(id)? == Some(crate::render_profiles::CatalogRole::Template) {
                self.require_settings(target, expected)?;
            } else {
                self.require_live_settings(target, expected)?;
            }
        }
        // Hard limits hold for every attribute edit (AE, timeline, commands), not per widget.
        let value = clamp_to_range(path, value);
        let value = self.without_navigation_offset(id, path, value)?;
        if let Some((parent, component)) = self.component_path(id, path)? {
            let mut a = self.attrs(id)?;
            let mut vector = self.resolve_attribute(id, &parent, frame, &mut HashSet::new())?;
            vector[component] = value.clone();
            if key
                || a.anim(&parent)
                    .and_then(|anim| anim.channels.get(component))
                    .is_some_and(|ch| !ch.is_empty())
            {
                if !a.is_animated(&parent) {
                    a.set_anim(
                        &parent,
                        Some(Animation::with_arity(
                            vector.as_array().ok_or("Invalid vector")?.len(),
                        )),
                    );
                }
                let scalar = value.as_f64().ok_or("Component must be numeric")? as f32;
                if let Some(ch) = a
                    .anim_mut(&parent)
                    .and_then(|anim| anim.channels.get_mut(component))
                {
                    ch.upsert_key(Keyframe::with_tan(frame, scalar, key_tan(ch, frame, kind)));
                }
            } else {
                a.set(&parent, to_attr(&vector));
            }
            return self.store_attrs(id, &a);
        }
        let mut a = self.attrs(id)?;
        match path {
            "/formula" => {
                serde_json::from_value::<crate::scene::Formula>(value.clone())
                    .map_err(|e| format!("Invalid formula: {e}"))?;
            }
            "/julia" => {
                serde_json::from_value::<Option<[f32; 3]>>(value.clone())
                    .map_err(|e| format!("Invalid Julia constant: {e}"))?;
            }
            _ => {
                let choices = attribute_choices(path);
                if !choices.is_empty() && !choices.contains(&value) {
                    return Err(format!("Invalid enum value: {path}"));
                }
            }
        }
        let discrete = matches!(path, "/formula" | "/julia" | "/material/facing")
            || !numeric(&value)
            || self.node(id)?["discrete"].get(path).is_some();
        let animated = a.is_animated(path);
        if key || animated {
            let attr = if discrete {
                let data = self.node_mut(id)?;
                if data["discrete"].get(path).is_none() {
                    let base = a
                        .get(path)
                        .cloned()
                        .map(attr_json)
                        .unwrap_or_else(|| value.clone());
                    data["discrete"][path] = json!([base]);
                }
                let dict = data["discrete"][path]
                    .as_array_mut()
                    .ok_or("Invalid discrete dictionary")?;
                let index = if let Some(i) = dict.iter().position(|v| v == &value) {
                    i
                } else {
                    dict.push(value.clone());
                    dict.len() - 1
                };
                if !animated {
                    a.set(path, AttrValue::Float(0.0));
                }
                AttrValue::Float(index as f32)
            } else {
                to_attr(&value)
            };
            let tans: Vec<Tan> = a
                .anim(path)
                .map(|anim| {
                    anim.channels
                        .iter()
                        .map(|c| key_tan(c, frame, kind))
                        .collect()
                })
                .unwrap_or_default();
            a.add_key(path, frame, &attr);
            if let Some(anim) = a.anim_mut(path) {
                for (i, channel) in anim.channels.iter_mut().enumerate() {
                    let tan = if discrete {
                        Tan::Constant
                    } else {
                        tans.get(i).copied().unwrap_or(kind)
                    };
                    channel.set_tan(frame, tan);
                }
            }
        } else {
            if let Some(dict) = self.node_mut(id)?["discrete"].as_object_mut() {
                dict.remove(path);
            }
            a.set(path, to_attr(&value));
        }
        if matches!(path, "/formula" | "/julia" | "/material/facing") && !key && !animated {
            let mut children = Attrs::new();
            discover(&value, path, &mut children);
            for (child, v) in children.iter() {
                if child != path && !a.is_animated(child) {
                    a.set(child, v.clone());
                }
            }
        }
        self.store_attrs(id, &a)
    }
    pub(crate) fn assert_unlocked(&self, id: NodeId) -> Result<(), String> {
        let mut current = Some(id);
        let mut seen = HashSet::new();
        while let Some(id) = current {
            if !seen.insert(id) {
                return Err("Parent cycle".into());
            }
            let info = self.info(id)?;
            if info.locked {
                return Err("Layer or ancestor is locked".into());
            }
            current = info.parent;
        }
        Ok(())
    }
    fn validate_parent(&self, id: Option<NodeId>, parent: Option<NodeId>) -> Result<(), String> {
        let mut current = parent;
        let mut seen = HashSet::new();
        while let Some(p) = current {
            if Some(p) == id || !seen.insert(p) {
                return Err("Parenting would create a cycle".into());
            }
            current = self.info(p)?.parent;
        }
        Ok(())
    }
    fn rebuild_children(&mut self) {
        let nodes = self.nodes();
        for data in self.graph.nodes.values_mut() {
            data["children"] = json!([]);
        }
        for node in nodes {
            if let Some(p) = node.parent {
                if let Some(parent) = self.graph.nodes.get_mut(&p.to_string()) {
                    if let Some(children) = parent["children"].as_array_mut() {
                        children.push(json!(node.id));
                    }
                }
            }
        }
    }
    fn evaluated_node(&self, id: NodeId, frame: f64) -> Result<Value, String> {
        let data = self.node(id)?;
        let mut v = data["gpu"].clone();
        let attrs = self.attrs(id)?;
        let mut paths: Vec<_> = attrs.iter().map(|(path, _)| path).collect();
        paths.sort();
        for path in paths {
            if matches!(
                path.as_str(),
                "/visible" | "/locked" | "/solo" | "/start" | "/end"
            ) || path.starts_with("/transform/")
                || path.starts_with("/custom/")
            {
                continue;
            }
            // Evaluation includes persisted channels hidden by the inspector schema.
            // Parent structural switches sort before their variant/optional children.
            if v.pointer(path).is_some() {
                set_pointer(&mut v, path, self.attribute_value(id, path, frame)?, false)?;
            }
        }
        if self.info(id)?.kind == WorldKind::Fractal {
            let iterations = v["render"]["iterations"].clone();
            v["render"] = serde_json::to_value(
                self.effective_render(self.output_render_profile()?, frame)?
                    .render,
            )
            .map_err(|e| e.to_string())?;
            v["render"]["iterations"] = iterations;
        }
        Ok(v)
    }
    pub(crate) fn camera_navigation_pose(
        &self,
        id: NodeId,
        after: &Scene,
        frame: f64,
    ) -> Result<Vec<(&'static str, Value)>, String> {
        use glam::{Mat3, Mat4, Quat, Vec3};
        after.camera.validate_lens()?;
        if !frame.is_finite() || !after.camera.distance.is_finite() || after.camera.distance <= 0.0
        {
            return Err("Invalid camera navigation pose".into());
        }
        let parent = self
            .info(id)?
            .parent
            .map(|parent| self.world_matrix(parent, frame, &mut HashSet::new()))
            .transpose()?
            .unwrap_or(Mat4::IDENTITY);
        if !parent.is_finite() || parent.determinant().abs() < 1e-10 {
            return Err("Camera parent transform is singular".into());
        }
        let inverse = parent.inverse();
        let vector = |path| -> Result<Vec3, String> {
            let value = self.attribute_value(id, path, frame)?;
            let items = value
                .as_array()
                .filter(|v| v.len() == 3)
                .ok_or("Invalid camera transform")?;
            let v = Vec3::new(
                items[0].as_f64().ok_or("Invalid camera transform")? as f32,
                items[1].as_f64().ok_or("Invalid camera transform")? as f32,
                items[2].as_f64().ok_or("Invalid camera transform")? as f32,
            );
            if !v.is_finite() {
                return Err("Invalid camera transform".into());
            }
            Ok(v)
        };
        let scale = vector("/transform/scale")?;
        let pivot = vector("/transform/pivot")?;
        if scale.abs().min_element() < 1e-8 {
            return Err("Camera transform is singular".into());
        }
        let mut canonical: crate::scene::Camera =
            serde_json::from_value(self.evaluated_node(id, frame)?["camera"].clone())
                .map_err(|error| error.to_string())?;
        canonical.yaw_degrees += self.camera_orbit_angle(id, frame)?;
        let canonical_q = canonical.orientation();
        let desired_q = after.camera.orientation();
        let basis = |back: Vec3, up: Vec3| -> Result<Quat, String> {
            let back = back.normalize_or_zero();
            let right = up.cross(back).normalize_or_zero();
            let up = back.cross(right).normalize_or_zero();
            if !back.is_finite() || !up.is_finite() || right.length_squared() < 0.5 {
                return Err("Camera orientation is singular".into());
            }
            Ok(Quat::from_mat3(&Mat3::from_cols(right, up, back)).normalize())
        };
        // Recover the local orthogonal basis from the same Gram-Schmidt projection used
        // by snapshot(). Inverse-parent up can contain a back component, removed here.
        let desired_local = basis(
            inverse.transform_vector3(desired_q * Vec3::Z),
            inverse.transform_vector3(desired_q * Vec3::Y),
        )?;
        let scaled_canonical = basis(
            scale * (canonical_q * Vec3::Z),
            scale * (canonical_q * Vec3::Y),
        )?;
        let rotation = (desired_local * scaled_canonical.inverse()).normalize();
        let (z, y, x) = rotation.to_euler(glam::EulerRot::ZYX);
        // Playa uses clockwise-positive ZYX rotation, unlike Camera's YXZ orbit basis.
        let mut degrees = [-x.to_degrees(), -y.to_degrees(), -z.to_degrees()];
        let previous = vector("/transform/rotation_degrees")?;
        for (angle, previous) in degrees.iter_mut().zip(previous.to_array()) {
            *angle = previous + (*angle - previous + 180.0).rem_euclid(360.0) - 180.0;
        }
        let target = Vec3::from(after.camera.target);
        if !target.is_finite() {
            return Err("Invalid camera target".into());
        }
        // Playa: T(position) * R * S * T(-pivot); pivot is not translated back.
        let position = inverse.transform_point3(target)
            + rotation * (scale * (pivot - Vec3::from(canonical.target)));
        let back_scale = parent
            .transform_vector3(rotation * (scale * (canonical_q * Vec3::Z)))
            .length();
        if !back_scale.is_finite() || back_scale < 1e-8 {
            return Err("Camera transform is singular".into());
        }
        Ok(vec![
            ("/transform/position", json!(position.to_array())),
            ("/transform/rotation_degrees", json!(degrees)),
            (
                "/camera/distance",
                json!(after.camera.distance / back_scale),
            ),
            ("/camera/fov_y_degrees", json!(after.camera.fov_y_degrees)),
            ("/camera/f_number", json!(after.camera.f_number)),
            ("/camera/focus_distance", json!(after.camera.focus_distance)),
        ])
    }

    fn with_navigation_offset(
        &self,
        id: NodeId,
        path: &str,
        mut value: Value,
    ) -> Result<Value, String> {
        if let Some(offset) = self.navigation_offset(id, path)? {
            if let Some(items) = value.as_array_mut() {
                for (index, item) in items.iter_mut().enumerate() {
                    *item = json!(
                        item.as_f64().ok_or("Invalid navigation channel")?
                            + offset[index].as_f64().unwrap_or(0.0)
                    );
                }
            } else {
                value = json!(
                    value.as_f64().ok_or("Invalid navigation channel")?
                        + offset.as_f64().unwrap_or(0.0)
                );
            }
        }
        Ok(value)
    }
    fn navigation_offset(&self, id: NodeId, path: &str) -> Result<Option<Value>, String> {
        if self.node(id)?["type"].as_str() != Some("Camera")
            || !matches!(
                path,
                "/transform/position"
                    | "/transform/rotation_degrees"
                    | "/camera/distance"
                    | "/camera/fov_y_degrees"
                    | "/camera/f_number"
                    | "/camera/focus_distance"
            )
        {
            return Ok(None);
        }
        Ok(self
            .attrs(id)?
            .get(&format!("/_navigation{path}"))
            .cloned()
            .map(attr_json))
    }

    fn without_navigation_offset(
        &self,
        id: NodeId,
        path: &str,
        mut value: Value,
    ) -> Result<Value, String> {
        if let Some((parent, component)) = self.component_path(id, path)? {
            if let Some(offset) = self.navigation_offset(id, &parent)? {
                value = json!(
                    value.as_f64().ok_or("Invalid navigation channel")?
                        - offset[component].as_f64().unwrap_or(0.0)
                );
            }
        } else if let Some(offset) = self.navigation_offset(id, path)? {
            if let Some(items) = value.as_array_mut() {
                for (index, item) in items.iter_mut().enumerate() {
                    *item = json!(
                        item.as_f64().ok_or("Invalid navigation channel")?
                            - offset[index].as_f64().unwrap_or(0.0)
                    );
                }
            } else {
                value = json!(
                    value.as_f64().ok_or("Invalid navigation channel")?
                        - offset.as_f64().unwrap_or(0.0)
                );
            }
        }
        Ok(value)
    }

    fn write_navigation_value(
        &self,
        id: NodeId,
        attrs: &mut Attrs,
        path: &str,
        desired: Value,
        frame: f64,
        auto_key: bool,
        kind: Tan,
    ) -> Result<bool, String> {
        let sampled = self.resolve_attribute(id, path, frame, &mut HashSet::new())?;
        let mut base = attrs
            .get(path)
            .cloned()
            .map(attr_json)
            .ok_or("Missing camera channel")?;
        let offset_path = format!("/_navigation{path}");
        let mut offset = attrs
            .get(&offset_path)
            .cloned()
            .map(attr_json)
            .unwrap_or_else(|| {
                if desired.is_array() {
                    json!([0.0, 0.0, 0.0])
                } else {
                    json!(0.0)
                }
            });
        let arity = desired.as_array().map_or(1, Vec::len);
        let mut base_changed = false;
        let mut offset_changed = false;
        let mut keys_changed = false;
        for component in 0..arity {
            let lane = |v: &Value| -> Result<f64, String> {
                (if arity == 1 { v } else { &v[component] })
                    .as_f64()
                    .ok_or_else(|| "Invalid camera channel".to_string())
            };
            let wanted = lane(&desired)?;
            let raw = lane(&sampled)?;
            let added = lane(&offset)?;
            if !wanted.is_finite() || !raw.is_finite() || !added.is_finite() {
                return Err("Invalid camera navigation value".into());
            }
            if (wanted - raw - added).abs() < 1e-5 {
                continue;
            }
            if attrs.conn(path).is_some() {
                return Err("Camera navigation channel is connected".into());
            }
            let animated = attrs
                .anim(path)
                .and_then(|a| a.channels.get(component))
                .is_some_and(|channel| !channel.is_empty());
            if auto_key {
                keys_changed = true;
                if attrs.anim(path).is_none() {
                    attrs.set_anim(path, Some(Animation::with_arity(arity)));
                }
                let ch = attrs
                    .anim_mut(path)
                    .and_then(|a| a.channels.get_mut(component))
                    .ok_or("Invalid camera animation arity")?;
                let tan = key_tan(ch, frame, kind);
                ch.upsert_key(Keyframe::with_tan(frame, (wanted - added) as f32, tan));
            } else if animated {
                if arity == 1 {
                    offset = json!(wanted - raw);
                } else {
                    offset[component] = json!(wanted - raw);
                }
                offset_changed = true;
            } else {
                if arity == 1 {
                    base = json!(wanted - added);
                } else {
                    base[component] = json!(wanted - added);
                }
                base_changed = true;
            }
        }
        if base_changed {
            attrs.set(path, to_attr(&base));
        }
        if offset_changed {
            attrs.set(&offset_path, to_attr(&offset));
        }
        Ok(base_changed || offset_changed || keys_changed)
    }

    fn world_matrix(
        &self,
        id: NodeId,
        frame: f64,
        seen: &mut HashSet<NodeId>,
    ) -> Result<glam::Mat4, String> {
        if !seen.insert(id) {
            return Err("Parent cycle".into());
        }
        let vec = |path: &str, default: [f32; 3]| -> Result<[f32; 3], String> {
            let v = self.attribute_value(id, path, frame)?;
            Ok(v.as_array()
                .filter(|v| v.len() == 3)
                .map(|v| std::array::from_fn(|i| v[i].as_f64().unwrap_or(default[i] as f64) as f32))
                .unwrap_or(default))
        };
        let p = vec("/transform/position", [0.0; 3])?;
        let r = vec("/transform/rotation_degrees", [0.0; 3])?.map(|v| v.to_radians());
        let s = vec("/transform/scale", [1.0; 3])?;
        let pivot = vec("/transform/pivot", [0.0; 3])?;
        let local = glam::Mat4::from_cols_array_2d(
            &playa_engine::entities::transform::build_model_matrix(p, r, s, pivot)
                .to_cols_array_2d(),
        );
        if let Some(parent) = self.info(id)?.parent {
            Ok(self.world_matrix(parent, frame, seen)? * local)
        } else {
            Ok(local)
        }
    }
    fn visible_at(
        &self,
        id: NodeId,
        frame: f64,
        seen: &mut HashSet<NodeId>,
    ) -> Result<bool, String> {
        if !seen.insert(id) {
            return Err("Parent cycle".into());
        }
        let visible = self
            .attribute_value(id, "/visible", frame)?
            .as_bool()
            .unwrap_or(true);
        let start = self
            .attribute_value(id, "/start", frame)?
            .as_f64()
            .unwrap_or(0.0);
        let end = self
            .attribute_value(id, "/end", frame)?
            .as_f64()
            .unwrap_or(self.last as f64 + 1.0);
        if !visible || frame < start || frame >= end {
            return Ok(false);
        }
        if let Some(parent) = self.info(id)?.parent {
            self.visible_at(parent, frame, seen)
        } else {
            Ok(true)
        }
    }
    /// Evaluate one authoritative material payload without building a world snapshot.
    pub fn material(&self, id: NodeId, frame: f64) -> Result<crate::scene::Material, String> {
        if !frame.is_finite() {
            return Err("Invalid material frame".into());
        }
        if self.info(id)?.kind != WorldKind::Material {
            return Err("Select a material layer".into());
        }
        serde_json::from_value(self.evaluated_node(id, frame)?["material"].clone())
            .map_err(|error| error.to_string())
    }
    #[cfg(test)]
    pub fn node_scene(&self, id: NodeId, frame: f64) -> Result<Scene, String> {
        let mut scene: Scene =
            serde_json::from_value(self.evaluated_node(id, frame)?).map_err(|e| e.to_string())?;
        let global = self.snapshot(frame)?;
        scene.camera = global.camera;
        scene.camera_reference = global.camera_reference;
        scene.environment = global.environment;
        scene.lighting = global.lighting;
        if self.info(id)?.kind != WorldKind::Fractal {
            scene.render = global.render;
        }
        scene.colour = global.colour;
        scene.document = None;
        if self.info(id)?.kind == WorldKind::Fractal {
            scene.object_world = Some(
                self.world_matrix(id, frame, &mut HashSet::new())?
                    .to_cols_array_2d(),
            );
            scene.object.offset =
                serde_json::from_value(self.attribute_value(id, "/transform/position", frame)?)
                    .map_err(|e| e.to_string())?;
            scene.object.rotation_degrees = serde_json::from_value::<[f32; 3]>(
                self.attribute_value(id, "/transform/rotation_degrees", frame)?,
            )
            .map_err(|e| e.to_string())?
            .map(|v| -v);
            let scale = self.attribute_value(id, "/transform/scale", frame)?;
            scene.object.scale = scale[0].as_f64().unwrap_or(1.0) as f32;
            if let Some(mat) = self.node(id)?["material"].as_str().and_then(NodeId::parse) {
                scene.material =
                    serde_json::from_value(self.evaluated_node(mat, frame)?["material"].clone())
                        .map_err(|e| e.to_string())?;
            }
        }
        Ok(scene)
    }
    /// Integrate the effective scalar track, following the same attribute connections as evaluation.
    fn integrated_scalar(
        &self,
        id: NodeId,
        path: &str,
        from: f64,
        to: f64,
        visiting: &mut HashSet<(NodeId, String)>,
    ) -> Result<f64, String> {
        if !visiting.insert((id, path.into())) {
            return Err("Attribute connection cycle".into());
        }
        let (storage_path, component) = self
            .component_path(id, path)?
            .unwrap_or_else(|| (path.into(), 0));
        let attrs = self.attrs(id)?;
        if let Some(conn) = attrs.conn(&storage_path) {
            let source_path = if storage_path != path {
                format!("{}/{}", conn.source_key, component)
            } else {
                conn.source_key.clone()
            };
            return self.integrated_scalar(
                NodeId(conn.source_layer),
                &source_path,
                from,
                to,
                visiting,
            );
        }
        if let Some(channel) = attrs
            .anim(&storage_path)
            .and_then(|a| a.channels.get(component))
            .filter(|channel| !channel.is_empty())
        {
            return Ok(crate::camera_orbit::integrate(channel, from, to));
        }
        let value = self
            .attribute_value(id, path, from)?
            .as_f64()
            .filter(|value| value.is_finite())
            .ok_or("Orbit speed must be a finite number")?;
        Ok(value * (to - from))
    }

    /// Offset from the document's first frame, independent of playback direction or wall-clock time.
    fn camera_orbit_angle(&self, id: NodeId, frame: f64) -> Result<f32, String> {
        if !self.fps.is_finite() || self.fps <= 0.0 {
            return Err("FPS must be positive and finite".into());
        }
        let phase = self
            .attribute_value(id, CAMERA_ORBIT_PHASE, frame)?
            .as_f64()
            .filter(|v| v.is_finite())
            .ok_or("Orbit phase must be a finite number")?;
        let angle = phase
            + self.integrated_scalar(
                id,
                CAMERA_ORBIT_SPEED,
                f64::from(self.first),
                frame,
                &mut HashSet::new(),
            )? / self.fps;
        let angle = angle as f32;
        if angle.is_finite() {
            Ok(angle)
        } else {
            Err("Camera orbit angle overflow".into())
        }
    }

    pub fn snapshot(&self, frame: f64) -> Result<Scene, String> {
        self.snapshot_with_render_profile(self.output_render_profile()?, frame)
    }
    pub fn snapshot_with_render_profile(
        &self,
        profile: NodeId,
        frame: f64,
    ) -> Result<Scene, String> {
        self.require_live_settings(profile, WorldKind::RenderSettings)?;
        if self.graph.format_version != playa_graph::SUBNET_FORMAT_VERSION {
            return Err(format!(
                "Unsupported world graph version {} (expected {})",
                self.graph.format_version,
                playa_graph::SUBNET_FORMAT_VERSION
            ));
        }
        if !frame.is_finite() {
            return Err("Invalid frame".into());
        }
        self.validate_render_profiles(frame)?;
        let mut base = self
            .graph
            .bus_slots
            .get("world")
            .ok_or("World settings missing")?
            .clone();
        base["render"] = serde_json::to_value(self.effective_render(profile, frame)?.render)
            .map_err(|e| e.to_string())?;
        let mut scene: Scene =
            serde_json::from_value(base).map_err(|e| format!("Invalid world settings: {e}"))?;
        scene.document = None;
        scene.animation = Default::default();
        scene.objects.clear();
        scene.lights.clear();
        scene.world_render = true;
        scene.object_world = None;
        let nodes = self.nodes();
        if let Some(id) = self.active_camera {
            let v = self.evaluated_node(id, frame)?;
            scene.camera =
                serde_json::from_value(v["camera"].clone()).map_err(|e| e.to_string())?;
            let reference: crate::scene::Formula =
                serde_json::from_value(self.node(id)?["gpu"]["formula"].clone())
                    .map_err(|e| e.to_string())?;
            scene.camera_reference = Some(reference.framing_radius());
            scene.camera.yaw_degrees += self.camera_orbit_angle(id, frame)?;
            let m = self.world_matrix(id, frame, &mut HashSet::new())?;
            if m != glam::Mat4::IDENTITY {
                let orientation = scene.camera.orientation();
                let back = m.transform_vector3(orientation * glam::Vec3::Z);
                let scale = back.length();
                let back = back.normalize_or_zero();
                let up = m.transform_vector3(orientation * glam::Vec3::Y);
                let right = up.cross(back).normalize_or_zero();
                let up = back.cross(right).normalize_or_zero();
                if scale <= 0.0 || right.length_squared() == 0.0 {
                    return Err("Camera transform is singular".into());
                }
                let q = glam::Quat::from_mat3(&glam::Mat3::from_cols(right, up, back));
                let (yaw, pitch, roll) = q.to_euler(glam::EulerRot::YXZ);
                scene.camera.yaw_degrees = yaw.to_degrees();
                scene.camera.pitch_degrees = -pitch.to_degrees();
                scene.camera.roll_degrees = roll.to_degrees();
                scene.camera.distance *= scale;
                scene.camera.target = m
                    .transform_point3(glam::Vec3::from(scene.camera.target))
                    .to_array();
            }
        }
        if let Some(id) = self.active_environment.filter(|id| {
            self.visible_at(*id, frame, &mut HashSet::new())
                .unwrap_or(false)
        }) {
            let v = self.evaluated_node(id, frame)?;
            scene.environment =
                serde_json::from_value(v["environment"].clone()).map_err(|e| e.to_string())?;
            scene.environment.revision =
                self.node(id)?["environment_revision"].as_u64().unwrap_or(0);
        } else {
            scene.environment.enabled = false;
        }
        let environment_light = if let Some(id) = self.active_environment {
            if self.visible_at(id, frame, &mut HashSet::new())? {
                Some(
                    serde_json::from_value::<crate::scene::Lighting>(
                        self.evaluated_node(id, frame)?["lighting"].clone(),
                    )
                    .map_err(|e| e.to_string())?,
                )
            } else {
                None
            }
        } else {
            None
        };
        let solo = nodes.iter().any(|n| n.solo);
        for node in &nodes {
            if !self.visible_at(node.id, frame, &mut HashSet::new())? {
                continue;
            }
            if solo
                && !node.solo
                && matches!(node.kind, WorldKind::Fractal | WorldKind::DirectionalLight)
            {
                continue;
            }
            if node.kind == WorldKind::DirectionalLight {
                let v = self.evaluated_node(node.id, frame)?;
                let mut light: crate::scene::Lighting =
                    serde_json::from_value(v["lighting"].clone()).map_err(|e| e.to_string())?;
                let az = light.sun_azimuth.to_radians();
                let el = light.sun_elevation.to_radians();
                let direction = glam::Vec3::new(az.sin() * el.cos(), el.sin(), az.cos() * el.cos());
                let d = self
                    .world_matrix(node.id, frame, &mut HashSet::new())?
                    .transform_vector3(direction)
                    .normalize_or_zero();
                light.sun_elevation = d.y.clamp(-1.0, 1.0).asin().to_degrees();
                light.sun_azimuth = d.x.atan2(d.z).to_degrees();
                scene.lights.push(light);
            }
        }
        if let Some(light) = scene.lights.first() {
            scene.lighting = *light;
        } else {
            scene.lighting.sun_intensity = 0.0;
        }
        if let Some(env) = environment_light {
            scene.lighting.sky_intensity = env.sky_intensity;
            scene.lighting.sky_horizon = env.sky_horizon;
            scene.lighting.sky_zenith = env.sky_zenith;
            scene.lighting.background = env.background;
        } else {
            scene.lighting.sky_intensity = 0.0;
            scene.lighting.background = false;
        }
        for node in nodes {
            if node.kind != WorldKind::Fractal
                || !self.visible_at(node.id, frame, &mut HashSet::new())?
                || (solo && !node.solo)
            {
                continue;
            }
            let mut object: Scene = serde_json::from_value(self.evaluated_node(node.id, frame)?)
                .map_err(|e| e.to_string())?;
            if let Some(material) = self.node(node.id)?["material"]
                .as_str()
                .and_then(NodeId::parse)
            {
                object.material = serde_json::from_value(
                    self.evaluated_node(material, frame)?["material"].clone(),
                )
                .map_err(|e| e.to_string())?;
            }
            object.camera = scene.camera;
            object.camera_reference = scene.camera_reference;
            object.environment = scene.environment.clone();
            object.lighting = scene.lighting;
            // Per-object DE parameters stay on the fractal; the top-level scene owns ray settings.
            object.colour = scene.colour.clone();
            object.document = None;
            object.animation = Default::default();
            object.objects.clear();
            object.lights.clear();
            object.world_render = false;
            object.object_world = Some(
                self.world_matrix(node.id, frame, &mut HashSet::new())?
                    .to_cols_array_2d(),
            );
            object.name = node.name;
            scene.objects.push(object);
        }
        if let Some(first) = scene.objects.first() {
            scene.formula = first.formula;
            scene.julia = first.julia;
            scene.object = first.object;
            scene.object_world = first.object_world;
            scene.material = first.material.clone();
            scene.palette = first.palette;
            scene.coloring = first.coloring;
            scene.trap_point = first.trap_point;
            scene.trap_axis = first.trap_axis;
            scene.trap_scale = first.trap_scale;
        }
        self.effective_render(profile, frame)?.apply_to(&mut scene);
        scene.camera.validate_lens()?;
        Ok(scene)
    }
}
