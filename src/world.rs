//! Playa graph nodes own world attributes; typed scenes are render snapshots.
#[cfg(test)]
#[path = "world_tests.rs"]
mod tests;
use crate::scene::{Bulb, Formula, Scene};
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
use std::sync::{Arc, LazyLock};

/// The local `/transform` position, rotation (degrees, the attribute convention) and scale that
/// put a node's pivot at world `pose` under its parent's world matrix `parent`: the inverse of
/// [`WorldDocument::manip_frame`]. A parent with nonuniform scale and a relative rotation
/// shears the child, which `T R S` cannot hold; that shear is dropped.
pub fn local_pose(parent: glam::Mat4, pose: glam::Mat4) -> ([f32; 3], [f32; 3], [f32; 3]) {
    let local = parent.inverse() * pose;
    playa_engine::entities::transform::decompose_pose(glam::Mat4::from_cols_array_2d(
        &local.to_cols_array_2d(),
    ))
}

/// The viewport manipulator tools (move, rotate, scale) a node of `kind` takes. Placed objects
/// take all three. The sun only turns: its world rotation is its direction (`snapshot`), and it
/// has no place or size. A camera moves by navigation; settings and environments have no place.
pub fn manip_tools(kind: WorldKind) -> [bool; 3] {
    match kind {
        WorldKind::Fractal | WorldKind::Group | WorldKind::Mask => [true, true, true],
        WorldKind::DirectionalLight => [false, true, false],
        _ => [false, false, false],
    }
}

/// Marker of WarpBro nodes on the system clipboard, so foreign text is never pasted as nodes.
const CLIPBOARD_KEY: &str = "warpbro_nodes";
const CLIPBOARD_VERSION: u32 = 1;

/// The nodes of a `WorldDocument::copy_fragment` clipboard text and the document they were
/// copied from (its graph id; null in a text without one); None for any other text.
pub fn parse_clipboard(text: &str) -> Option<(Value, HashMap<String, Value>)> {
    let value: Value = serde_json::from_str(text.trim()).ok()?;
    if value[CLIPBOARD_KEY].as_u64() != Some(u64::from(CLIPBOARD_VERSION)) {
        return None;
    }
    let nodes = serde_json::from_value(value["nodes"].clone()).ok()?;
    Some((value["document"].clone(), nodes))
}

/// The take a node plays: node data (`node["take"]`, a take id or null), shown and set as this
/// attribute path like `/material_id`. Static: which take plays is not keyed.
pub(crate) const TAKE_CLIP: &str = "/take/clip";
/// Frames the take starts later than its own start frame (keyable).
pub(crate) const TAKE_OFFSET: &str = "/take/offset";
/// The take's playback rate (keyable; 0 holds, negative plays backwards): the time warp
/// integrates it from the take's start (`WorldDocument::take_frame`).
pub(crate) const TAKE_SPEED: &str = "/take/speed";

/// The kind of take a node of `kind` plays: the camera's visible pose, or the local transform of
/// a node the manipulator moves (`manip_tools`).
pub fn take_kind(kind: WorldKind) -> Option<crate::takes::TakeKind> {
    if kind == WorldKind::Camera {
        Some(crate::takes::TakeKind::Camera)
    } else if manip_tools(kind).contains(&true) {
        Some(crate::takes::TakeKind::Transform)
    } else {
        None
    }
}

pub(crate) const CAMERA_ORBIT_SPEED: &str = "/camera/orbit_speed_degrees";
pub(crate) const CAMERA_ORBIT_PHASE: &str = "/camera/orbit_phase_degrees";
/// The scene size a camera's distance and focus distance are measured in (scene units per
/// framing radius, `Scene::camera_reference`). A camera attribute of its own, set from the
/// fractal it was created to frame, so editing that fractal's formula never moves the camera.
pub(crate) const CAMERA_FRAMING: &str = "/camera/framing_radius";

/// Default byte budget of the undo / redo history (serialized document size, `history_bytes`).
/// A fresh scene serializes to tens of kilobytes, so this keeps over a thousand steps of an
/// ordinary scene; the resident size is a few times larger (`serde_json::Value` trees), which
/// bounds the history to a few hundred megabytes in the worst case.
const UNDO_BUDGET: usize = 64 << 20;

/// What a scene is evaluated for: an output (export, Output profile) shows every node kind, the
/// viewport only the kinds its ViewportSettings show (`WorldDocument::shown_kinds`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum View {
    Output,
    Viewport,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum WorldKind {
    Fractal,
    Camera,
    DirectionalLight,
    Environment,
    Group,
    Material,
    RenderSettings,
    ViewportSettings,
    /// Export encode recipe (`export::OutputSettings`): format, codec, colour, size.
    OutputSettings,
    /// A shape the fractals in its scope are intersected with (`scene::Mask`): inside a fractal
    /// (Outliner parent) it sections that fractal, at the root every fractal.
    Mask,
}
/// Deserialize a persisted `Option` field that must be present: serde fills a missing `Option`
/// with `None` on its own, but with `deserialize_with` a missing field is an error (strict
/// decode: nothing is filled in), while `null` still reads as `None`. Use as
/// `#[serde(deserialize_with = "present")]` (`crate::world::present` outside this module).
pub(crate) fn present<'de, D, T>(d: D) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::deserialize(d)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorldDocument {
    pub graph: SubnetFile,
    #[serde(deserialize_with = "present")]
    pub active_camera: Option<NodeId>,
    #[serde(deserialize_with = "present")]
    pub active_environment: Option<NodeId>,
    pub first: u32,
    pub last: u32,
    pub fps: f64,
    /// Numbered time marks (After Effects composition markers): slot 0-9 -> frame.
    pub marks: BTreeMap<u8, u32>,
    /// Recorded navigation takes. Saved with the scene, but outside undo/redo and outside
    /// equality: a take is a recording, not an edit (`crate::takes`).
    #[serde(default, skip_serializing_if = "crate::takes::Takes::is_empty")]
    pub takes: crate::takes::Takes,
    /// The viewport toolbar's CamClip as saved with this scene (the live one is application
    /// state, `App::camera_slots`, written here on save and read back on load). Like the takes,
    /// not an edit: outside undo / redo and equality.
    #[serde(
        default,
        skip_serializing_if = "crate::camera_slots::CameraSlots::is_empty"
    )]
    pub camera_slots: crate::camera_slots::CameraSlots,
}
impl PartialEq for WorldDocument {
    fn eq(&self, other: &Self) -> bool {
        self.first == other.first
            && self.last == other.last
            && self.fps == other.fps
            && self.active_camera == other.active_camera
            && self.active_environment == other.active_environment
            && self.marks == other.marks
            && self.graph.format_version == other.graph.format_version
            && self.graph.id == other.graph.id
            && self.graph.bus_slots == other.graph.bus_slots
            && self.graph.nodes == other.graph.nodes
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
    /// Hidden from the timeline while its Shy switch is on (After Effects shy).
    pub shy: bool,
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
    /// Display names of `choices` (same order) where the values are ids; empty: the values name
    /// themselves.
    pub choice_labels: Vec<String>,
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

pub use crate::formulas::Slider;
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum WorldCommand {
    Create {
        kind: WorldKind,
        name: String,
        parent: Option<NodeId>,
    },
    Delete(NodeId),
    /// Copy the nodes (with descendants) in place; the copies are selected
    /// (`WorldEditor::duplicate` picks them for the UI).
    Duplicate(Vec<NodeId>),
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
    /// A whole-curve effect (`curves::Effect`: Simplify, Smooth, Euler filter) on the animation
    /// of `targets` (node, attribute or component path), one undo step, its times in seconds
    /// (`curves::Effect::scale_time` to the keys' frames). Simplify and Smooth edit
    /// each continuous channel; discrete ones (choices, switches) are left alone. The Euler filter
    /// edits the rotation each target belongs to: `/transform/rotation_degrees` (ZYX, y the
    /// middle angle) or a camera's yaw / pitch / roll (whole turns only: the pitch is limited).
    CurveEffect {
        targets: Vec<(NodeId, String)>,
        effect: curves::Effect<f64>,
    },
    SetMetadata {
        id: NodeId,
        path: String,
        value: Value,
    },
    /// Put `ids` in this order among the layer-order slots they occupy; every other node keeps
    /// its slot. A filtered view (the timeline without its shy nodes) reorders what it shows
    /// without moving what it hides.
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
    /// Copy an OutputSettings node (default: the assigned one) into a new profile or template;
    /// `assign` also binds the new profile as the export recipe, in the same Undo step.
    CreateOutputSettings {
        name: String,
        role: crate::render_profiles::CatalogRole,
        source: Option<NodeId>,
        assign: bool,
    },
    /// Instantiate an output template as an independent profile, optionally binding it.
    InstantiateOutputTemplate {
        id: NodeId,
        name: String,
        assign: bool,
    },
    SetOutputSettings(NodeId),
    SetActiveCamera(NodeId),
    SetActiveEnvironment(NodeId),
    ReloadEnvironment(NodeId),
    Batch(Vec<WorldCommand>),
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
/// The document and the selection (its last node is the active one): what undo restores.
type WorldEditSnapshot = (WorldDocument, Vec<NodeId>);

/// One undo / redo state and what it costs of the history budget (`WorldEditor::push_undo`).
#[derive(Clone, Debug)]
struct HistoryStep {
    state: WorldEditSnapshot,
    /// Serialized size of the state's graph ([`graph_bytes`]), measured once when stored.
    bytes: usize,
}
impl HistoryStep {
    fn new(state: WorldEditSnapshot) -> Self {
        let bytes = graph_bytes(&state.0);
        Self { state, bytes }
    }
}

/// The serialized size of `document`'s graph, counted without building the text: the measure of
/// the undo budget (the graph is what every state holds its own copy of; the takes are shared
/// `Arc`s and the rest is a few numbers).
fn graph_bytes(document: &WorldDocument) -> usize {
    struct Count(usize);
    impl std::io::Write for Count {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0 += bytes.len();
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut count = Count(0);
    match serde_json::to_writer(&mut count, &document.graph) {
        Ok(()) => count.0,
        Err(error) => {
            // A graph of JSON values with string keys always serializes; logged, not hidden.
            log::error!("Undo history: cannot measure a document state: {error}");
            0
        }
    }
}

#[derive(Clone, Debug)]
struct PendingWorldEdit {
    gesture: u64,
    original: WorldEditSnapshot,
}

/// Paste report: the pasted roots and the references the paste cleared because they point to
/// nothing in this scene (`WorldEditor::paste`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Pasted {
    pub roots: usize,
    /// Material references to a material in neither the clipboard nor this scene.
    pub materials: usize,
    /// Parent references to a node in neither the clipboard nor this scene.
    pub parents: usize,
    /// Take references: takes are per document, so all of them from another scene, else the
    /// ones to a take this scene no longer has (or of another kind).
    pub takes: usize,
}

#[derive(Clone, Debug)]
pub struct WorldEditor {
    pub document: WorldDocument,
    /// The selection: ordered, unique, the LAST node is the active (lead) one, as in Maya. One
    /// field, so "the active node is selected" holds by construction; read and written only
    /// through the methods below.
    selected: Vec<NodeId>,
    undo: Vec<HistoryStep>,
    redo: Vec<HistoryStep>,
    /// Bytes the undo and redo states may hold together ([`UNDO_BUDGET`]); the oldest undo
    /// states go first. Dropping them also lets the takes only they played go
    /// (`takes_in_use` reads the history).
    undo_budget: usize,
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
            selected: selection.into_iter().collect(),
            undo: vec![],
            redo: vec![],
            undo_budget: UNDO_BUDGET,
            revision: 0,
            pending_edit: None,
            new_key: Tan::Smooth,
        }
    }
    /// Select `id` as a click does: alone, or with `add` toggled in or out of the selection
    /// (it becomes the current node when added). None (a click on nothing) clears the
    /// selection unless `add`. The one click rule of the outliner, timeline and viewport.
    pub fn click_select(&mut self, id: Option<NodeId>, add: bool) {
        match (id, add) {
            (Some(id), true) => self.toggle(id),
            (Some(id), false) => self.select_only(id),
            (None, true) => {}
            (None, false) => self.clear_selection(),
        }
    }
    /// The active (lead) node: the last selected. The Attribute Editor, the inspector and the
    /// range anchor follow it.
    pub fn active(&self) -> Option<NodeId> {
        self.selected.last().copied()
    }
    /// The whole selection in order; the active node is last.
    pub fn selected(&self) -> &[NodeId] {
        &self.selected
    }
    /// Make `id` the whole selection (a plain click, a create, a paste of one node).
    pub fn select_only(&mut self, id: NodeId) {
        self.selected.clear();
        self.selected.push(id);
    }
    /// Shift/Ctrl click: add `id` as the new active node, or take it out when selected.
    pub fn toggle(&mut self, id: NodeId) {
        match self.selected.iter().position(|n| *n == id) {
            Some(i) => {
                self.selected.remove(i);
            }
            None => self.selected.push(id),
        }
    }
    /// Select nothing.
    pub fn clear_selection(&mut self) {
        self.selected.clear();
    }
    /// Replace the selection by `ids` (a range, a box, a paste): duplicates dropped, the order
    /// kept, so the last one becomes active. A caller that wants another lead calls
    /// [`Self::set_lead`] after.
    pub fn set_selection(&mut self, ids: Vec<NodeId>) {
        self.selected.clear();
        for id in ids {
            if !self.selected.contains(&id) {
                self.selected.push(id);
            }
        }
    }
    /// Make `id` the active node when it is selected (it moves to the end); otherwise nothing
    /// changes. For gestures that name the row they landed on, which need not be the last one.
    pub fn set_lead(&mut self, id: NodeId) {
        if let Some(i) = self.selected.iter().position(|n| *n == id) {
            self.selected.remove(i);
            self.selected.push(id);
        }
    }
    /// The nodes a command acts on: the selection, or `row` alone when the command comes from a
    /// row that is not part of it (a right-clicked outliner or timeline row). `row` None: a
    /// toolbar button or hotkey, the selection. The ONE "selected, else this one" rule of
    /// Duplicate, Delete, the property filters and the take targets.
    pub fn targets(&self, row: Option<NodeId>) -> Vec<NodeId> {
        match row {
            Some(row) if !self.selected.contains(&row) => vec![row],
            _ => self.selected.clone(),
        }
    }
    /// Duplicate `targets(row)` with their descendants as one undo step; the copies become the
    /// selection. Returns how many nodes were copied. Every Duplicate of the UI ends here.
    pub fn duplicate(&mut self, row: Option<NodeId>) -> Result<usize, String> {
        let ids = self.targets(row);
        if ids.is_empty() {
            return Err("Select the nodes to duplicate".into());
        }
        let count = ids.len();
        self.execute(WorldCommand::Duplicate(ids))?;
        Ok(count)
    }
    /// Paste a clipboard fragment of `WorldDocument::copy_fragment` (Ctrl+V) as one undo step; the
    /// pasted roots become the selection. The report counts what the paste cleared.
    pub fn paste(&mut self, text: &str) -> Result<Pasted, String> {
        let (source, nodes) =
            parse_clipboard(text).ok_or("The clipboard holds no WarpBro nodes")?;
        let same_document = source == json!(self.document.graph.id);
        let mut pasted = Pasted::default();
        self.apply_edit(None, |editor| {
            pasted = editor.insert_fragment(nodes, same_document)?.1;
            Ok(())
        })?;
        Ok(pasted)
    }
    /// Replace the document by `document` as one undoable edit (Edit > Reset scene). Live state
    /// stays as undo keeps it (`keep_live`: takes, CamClip, identity); the selection becomes the
    /// first fractal, as `new` picks it.
    pub fn reset(&mut self, document: WorldDocument) -> Result<(), String> {
        let document = self.keep_live(document);
        let selection = document
            .nodes()
            .iter()
            .find(|n| n.kind == WorldKind::Fractal)
            .map(|n| n.id);
        self.apply_edit(None, |editor| {
            editor.document = document;
            editor.set_selection(selection.into_iter().collect());
            Ok(())
        })
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
        self.push_undo(pending.original);
        self.redo.clear();
        true
    }

    /// Store an undo state, then drop the oldest ones while undo and redo together hold more
    /// than the budget; the newest undo state always stays, however large.
    fn push_undo(&mut self, state: WorldEditSnapshot) {
        self.undo.push(HistoryStep::new(state));
        let total: usize = self.undo.iter().chain(&self.redo).map(|s| s.bytes).sum();
        let excess = total.saturating_sub(self.undo_budget);
        let mut freed = 0;
        let dropped = self
            .undo
            .iter()
            .take(self.undo.len().saturating_sub(1))
            .take_while(|step| {
                let more = freed < excess;
                freed += step.bytes;
                more
            })
            .count();
        self.undo.drain(..dropped);
    }

    pub fn finish_edit_unless(&mut self, active: Option<u64>) -> bool {
        if self.active_edit() != active {
            self.finish_edit()
        } else {
            false
        }
    }

    fn differs_from(&self, snapshot: &WorldEditSnapshot) -> bool {
        self.document != snapshot.0 || self.selected != snapshot.1
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
            self.push_undo(before);
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
        let before = (self.document.clone(), self.selected.clone());
        if let Err(error) = apply(self).and_then(|()| {
            self.document
                .validate_render_profiles(f64::from(self.document.first))
        }) {
            self.document = before.0;
            self.selected = before.1;
            return Err(error);
        }
        self.record_edit(before, gesture);
        Ok(())
    }
    /// Commit a camera capture as one undoable edit, preserving keys outside its recorded interval.
    pub fn record_camera(
        &mut self,
        recording: &crate::camera_recorder::Recording,
    ) -> Result<(usize, usize, f64, u32), String> {
        if self.document != recording.document || self.revision() != recording.revision {
            return Err("The document changed during camera recording".into());
        }
        self.record_prepared(recording.prepare()?)
    }

    /// Write baked keys as one undoable edit: inside `[first, last]` they replace the node's
    /// keys, outside it the old keys remain; the work area grows to hold them. A take the node
    /// played is detached in the same edit, or it would hide the keys. Returns the samples, keys,
    /// largest sample error and last frame.
    pub fn record_prepared(
        &mut self,
        prepared: crate::camera_recorder::Prepared,
    ) -> Result<(usize, usize, f64, u32), String> {
        self.apply_edit(None, |editor| {
            editor.document.assert_unlocked(prepared.node)?;
            let mut attrs = editor.document.attrs(prepared.node)?;
            for (path, mut animation) in prepared.channels.clone() {
                if !attrs.contains(path) {
                    return Err(format!("Missing channel {path}"));
                }
                if let Some(previous) = attrs.anim(path) {
                    if previous.channels.len() != animation.channels.len() {
                        return Err(format!("Channel {path} changed shape"));
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
            editor.document.store_attrs(prepared.node, &attrs)?;
            editor.document.attach_take(prepared.node, None)?;
            if prepared.extend_last > editor.document.last {
                editor.document.set_last(prepared.extend_last)?;
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
        if let Some(HistoryStep {
            state: (d, selected),
            ..
        }) = self.undo.pop()
        {
            let d = self.keep_live(d);
            self.redo.push(HistoryStep::new((
                std::mem::replace(&mut self.document, d),
                self.selected.clone(),
            )));
            self.selected = selected;
            self.revision = self.revision.wrapping_add(1);
            true
        } else {
            false
        }
    }
    pub fn redo(&mut self) -> bool {
        self.finish_edit();
        if let Some(HistoryStep {
            state: (d, selected),
            ..
        }) = self.redo.pop()
        {
            let d = self.keep_live(d);
            let before = (
                std::mem::replace(&mut self.document, d),
                self.selected.clone(),
            );
            self.push_undo(before);
            self.selected = selected;
            self.revision = self.revision.wrapping_add(1);
            true
        } else {
            false
        }
    }
    /// A history snapshot carrying the live takes, CamClip and identity: undo/redo restore
    /// edits, never recordings or the clipboard, and never turn the document back into the one
    /// it was copied from.
    fn keep_live(&self, mut document: WorldDocument) -> WorldDocument {
        document.takes = self.document.takes.clone();
        document.camera_slots = self.document.camera_slots;
        document.graph.id = self.document.graph.id;
        document
    }

    /// Make the document a new one (Save As to another file): a fresh UUID, so it and the file
    /// it was copied from are told apart (clipboard take references, caches keyed by it).
    pub fn renew_identity(&mut self) {
        self.document.renew_identity();
        self.revision = self.revision.wrapping_add(1);
    }

    /// The document states of the undo history: undo, redo and a pending gesture's original.
    fn history_mut(&mut self) -> impl Iterator<Item = &mut WorldDocument> {
        self.undo
            .iter_mut()
            .chain(self.redo.iter_mut())
            .map(|step| &mut step.state)
            .chain(self.pending_edit.iter_mut().map(|p| &mut p.original))
            .map(|(document, ..)| document)
    }

    /// The ids of the takes any document state plays, the current one and every undo / redo
    /// state: retention keeps them, so undo never brings back a reference to a dropped take.
    fn takes_in_use(&self) -> Vec<u64> {
        let mut used = self.document.takes_in_use();
        for (document, ..) in self
            .undo
            .iter()
            .chain(&self.redo)
            .map(|step| &step.state)
            .chain(self.pending_edit.iter().map(|p| &p.original))
        {
            used.extend(document.takes_in_use());
        }
        used.sort_unstable();
        used.dedup();
        used
    }

    /// Store a finished take (no undo step). Returns its id. Every change of the takes moves
    /// the revision: views of them (the Take pickers, the timeline's take bars) refresh by it.
    pub fn add_take(&mut self, take: crate::takes::Take) -> u64 {
        let used = self.takes_in_use();
        self.revision = self.revision.wrapping_add(1);
        self.document.takes.add(take, &used)
    }

    /// Play take `id` on `nodes` (None: play none) as one undoable edit, or as part of the
    /// `gesture` that releases them (a drag or navigation continuing from the take's pose):
    /// their `/take/clip` (`WorldDocument::attach_take`).
    pub fn bind_take(
        &mut self,
        nodes: &[NodeId],
        id: Option<u64>,
        gesture: Option<u64>,
    ) -> Result<(), String> {
        self.apply_edit(gesture, |editor| {
            for node in nodes {
                editor.document.assert_unlocked(*node)?;
                editor.document.attach_take(*node, id)?;
            }
            Ok(())
        })
    }

    /// Delete a take no node plays (no undo step: takes are recordings, not edits). A played take
    /// is refused: removing it would silently change the scene's motion. Undo / redo states that
    /// played it play none from now on: the take is gone from every state of the scene.
    pub fn delete_take(&mut self, id: u64) -> Result<bool, String> {
        let users = self.document.take_users(id);
        if !users.is_empty() {
            let names: Vec<String> = users
                .iter()
                .filter_map(|n| self.document.info(*n).ok().map(|i| i.name))
                .collect();
            return Err(format!(
                "Take {id} plays on {}: detach it first",
                names.join(", ")
            ));
        }
        for document in self.history_mut() {
            document.release_take(id);
        }
        self.revision = self.revision.wrapping_add(1);
        Ok(self.document.takes.remove(id))
    }

    /// Rename, pin, retime or filter a take (a change to a played take moves its nodes). An
    /// edit that leaves the take invalid (a filter that fails on its motion) is refused and the
    /// take stays as it was (`Takes::edit`). Ok(false): there is no take `id`.
    pub fn edit_take(
        &mut self,
        id: u64,
        apply: impl FnOnce(&mut crate::takes::Take),
    ) -> Result<bool, String> {
        let used = self.takes_in_use();
        let edited = self.document.takes.edit(id, &used, apply)?;
        if edited {
            self.revision = self.revision.wrapping_add(1);
        }
        Ok(edited)
    }

    /// Key take `id` onto `node` as one undoable edit (keys outside its span remain) that also
    /// detaches any take from the node (`record_prepared`): its keys now show the same motion. A camera take bakes onto the active
    /// camera through the navigation pose solver, a transform take onto `node`'s transform.
    pub fn bake_take(
        &mut self,
        node: NodeId,
        id: u64,
        options: crate::camera_recorder::Options,
    ) -> Result<(usize, usize, f64, u32), String> {
        self.finish_edit();
        let mut take = self
            .document
            .takes
            .get(id)
            .cloned()
            .ok_or(format!("No take {id}"))?;
        // A node that plays this take through a time warp shows the warped motion: bake that.
        if self.document.take_clip(node)? == Some(id) && !self.document.take_warp_identity(node)? {
            take = Arc::new(self.document.shown_take(node)?);
        }
        let summary = match take.kind() {
            crate::takes::TakeKind::Camera => {
                if self.document.active_camera != Some(node) {
                    return Err("A camera take bakes onto the active camera".into());
                }
                let recording = crate::camera_recorder::Recording::from_take(
                    &self.document,
                    self.revision(),
                    &take,
                    options,
                )?;
                self.record_camera(&recording)?
            }
            crate::takes::TakeKind::Transform => {
                let prepared =
                    crate::camera_recorder::bake_transform(&self.document, node, &take, options)?;
                self.record_prepared(prepared)?
            }
        };
        Ok(summary)
    }

    #[cfg(test)]
    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }
    /// Insert `nodes` (UUID -> node data) with fresh UUIDs and select the inserted roots.
    /// References inside the fragment follow the remap. This alone decides which outside
    /// references are cleared, and counts them: a parent or material in neither the fragment nor
    /// this document, and a take reference unless the fragment comes from this document
    /// (`same_document`: take ids are per document) and this scene still has that take, of the
    /// kind the node plays. A fragment that still cannot be evaluated (e.g. an attribute
    /// connection to an absent node) is an error, so the edit rolls back.
    fn insert_fragment(
        &mut self,
        nodes: HashMap<String, Value>,
        same_document: bool,
    ) -> Result<(Vec<NodeId>, Pasted), String> {
        if nodes.is_empty() {
            return Err("Nothing to paste".into());
        }
        // Clipboard text is foreign data: check the stored shape before indexing into it.
        for (key, data) in &nodes {
            node_shape(key, data)?;
        }
        // Copies hold references (UUIDs), never the referenced nodes.
        let mut nodes = nodes;
        let mut pasted = Pasted::default();
        for (key, field) in self.document.unresolved_references(&nodes, same_document) {
            if let Some(data) = nodes.get_mut(&key).and_then(Value::as_object_mut) {
                data.insert(field.into(), Value::Null);
                match field {
                    "material" => pasted.materials += 1,
                    "parent" => pasted.parents += 1,
                    _ => pasted.takes += 1,
                }
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
            check_arity(&attrs)?;
            for path in [
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
        self.set_selection(roots.clone());
        pasted.roots = roots.len();
        Ok((roots, pasted))
    }
    fn apply(&mut self, command: WorldCommand) -> Result<(), String> {
        match command {
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
                    self.document.unkey(id, &path, frame, true, self.new_key)?;
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
                self.select_only(id);
            }
            WorldCommand::Delete(id) => {
                let remove = self.document.subtree(&[id]);
                // One lock rule: no node that goes may be locked, a descendant included.
                for id in &remove {
                    self.document.assert_unlocked(*id)?;
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
                if let Some(order) = self
                    .document
                    .graph
                    .bus_slots
                    .get_mut("layer_order")
                    .and_then(Value::as_array_mut)
                {
                    order.retain(|v| {
                        v.as_str()
                            .and_then(NodeId::parse)
                            .is_none_or(|id| !remove.contains(&id))
                    });
                }
                self.selected.retain(|id| !remove.contains(id));
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
                let (roots, _) = self.insert_fragment(nodes, true)?;
                for &id in &roots {
                    let name = format!("{} copy", self.document.info(id)?.name);
                    self.document.node_mut(id)?["name"] = json!(name);
                }
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
                let mut order: Vec<NodeId> =
                    self.document.nodes().into_iter().map(|n| n.id).collect();
                let unique: HashSet<_> = ids.iter().copied().collect();
                if unique.len() != ids.len() || !unique.iter().all(|id| order.contains(id)) {
                    return Err("Layer order contains unknown or duplicate IDs".into());
                }
                // The listed ids fill the slots they occupy, in their new order; every other
                // node keeps its slot. The stored order is rewritten whole (no stale ids).
                let slots: Vec<usize> = order
                    .iter()
                    .enumerate()
                    .filter(|(_, id)| unique.contains(id))
                    .map(|(slot, _)| slot)
                    .collect();
                for (slot, id) in slots.into_iter().zip(ids) {
                    if let Some(at) = order.get_mut(slot) {
                        *at = id;
                    }
                }
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
                self.document.set_last(last)?;
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
                    ..Scene::preset(Formula::Mandelbulb(Bulb::PRESET))
                };
                let id = self
                    .document
                    .insert(WorldKind::Material, &name, &scene, None)?;
                self.select_only(id);
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
                let created = self.active().ok_or("The new material was not selected")?;
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
                if let Some(material) = material
                    && self.document.info(material)?.kind != WorldKind::Material
                {
                    return Err("Select a material layer".into());
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
            WorldCommand::CreateOutputSettings {
                name,
                role,
                source,
                assign,
            } => {
                let source = match source {
                    Some(source) => source,
                    None => self.document.output_settings_id()?,
                };
                self.create_output_settings(source, name, role, assign)?;
            }
            WorldCommand::InstantiateOutputTemplate { id, name, assign } => {
                if self.document.catalog_role(id)?
                    != Some(crate::render_profiles::CatalogRole::Template)
                {
                    return Err("The selected node is not an output template".into());
                }
                self.create_output_settings(
                    id,
                    name,
                    crate::render_profiles::CatalogRole::Profile,
                    assign,
                )?;
            }
            WorldCommand::SetOutputSettings(id) => {
                self.document
                    .require_live_settings(id, WorldKind::OutputSettings)?;
                self.document
                    .graph
                    .bus_slots
                    .insert("output_settings".into(), json!(id));
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
                if path == TAKE_CLIP {
                    self.document.assert_unlocked(id)?;
                    let clip = match &value {
                        Value::Null => None,
                        v => Some(v.as_u64().filter(|t| *t > 0).ok_or("Invalid take id")?),
                    };
                    return self.document.attach_take(id, clip);
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
                self.document.unkey(id, &path, frame, false, self.new_key)?;
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
                let mut attrs = self.document.attrs(id)?;
                let (attr, channel) = WorldDocument::lane(&attrs, &path);
                WorldDocument::for_channels(&mut attrs, &attr, channel, |_, ch| {
                    let moves: Vec<_> = ch
                        .keys()
                        .iter()
                        .filter(|k| frames.contains(&k.t()))
                        .map(|k| (k.t(), k.t() + delta))
                        .collect();
                    ch.move_keys(&moves);
                    Ok::<(), String>(())
                })?;
                self.document.store_attrs(id, &attrs)?;
            }
            WorldCommand::Interpolation {
                id,
                path,
                frames,
                kind,
            } => {
                self.document.assert_unlocked(id)?;
                let mut attrs = self.document.attrs(id)?;
                let (attr, channel) = WorldDocument::lane(&attrs, &path);
                // A discrete attribute (choices, switches) only steps.
                let kind = if channel.is_none() && self.document.is_discrete(id, &attr)? {
                    Tan::Constant
                } else {
                    kind
                };
                WorldDocument::for_channels(&mut attrs, &attr, channel, |_, ch| {
                    let times: Vec<_> = ch
                        .keys()
                        .iter()
                        .map(|k| k.t())
                        .filter(|t| frames.contains(t))
                        .collect();
                    for t in times {
                        ch.set_tan(t, kind);
                    }
                    Ok::<(), String>(())
                })?;
                self.document.store_attrs(id, &attrs)?;
            }
            WorldCommand::CurveEffect { targets, effect } => self.curve_effect(targets, effect)?,
        }
        self.document.rebuild_children();
        Ok(())
    }

    fn curve_effect(
        &mut self,
        targets: Vec<(NodeId, String)>,
        effect: curves::Effect<f64>,
    ) -> Result<(), String> {
        const CAMERA_ROTATION: [&str; 3] = [
            "/camera/yaw_degrees",
            "/camera/pitch_degrees",
            "/camera/roll_degrees",
        ];
        const ROTATION: &str = "/transform/rotation_degrees";
        let error = |e: curves::KeysError| e.to_string();
        // The effect is given in seconds; the keys are in frames.
        let effect = effect.scale_time(self.document.fps);
        if let curves::Effect::EulerFilter { .. } = effect {
            let mut rotations: Vec<(NodeId, bool)> = Vec::new();
            for (id, path) in &targets {
                let camera = CAMERA_ROTATION.contains(&path.as_str());
                if (camera || path.starts_with(ROTATION)) && !rotations.contains(&(*id, camera)) {
                    rotations.push((*id, camera));
                }
            }
            if rotations.is_empty() {
                return Err("Euler filter: select keys of a rotation".into());
            }
            for (id, camera) in rotations {
                self.document.assert_unlocked(id)?;
                let mut attrs = self.document.attrs(id)?;
                if camera {
                    let mut tracks = CAMERA_ROTATION
                        .iter()
                        .map(|path| {
                            attrs
                                .anim(path)
                                .and_then(|a| a.channels.first())
                                .map(|c| c.track().clone())
                        })
                        .collect::<Option<Vec<_>>>()
                        .ok_or("Euler filter: animate yaw, pitch and roll")?;
                    curves::Effect::EulerFilter { middle: None }
                        .apply(&mut tracks)
                        .map_err(error)?;
                    for (path, track) in CAMERA_ROTATION.iter().zip(tracks) {
                        if let Some(channel) =
                            attrs.anim_mut(path).and_then(|a| a.channels.first_mut())
                        {
                            channel.replace_track(track);
                        }
                    }
                } else {
                    let anim = attrs
                        .anim_mut(ROTATION)
                        .filter(|a| a.channels.len() == 3)
                        .ok_or("Euler filter: animate the rotation")?;
                    let mut tracks: Vec<_> =
                        anim.channels.iter().map(|c| c.track().clone()).collect();
                    curves::Effect::EulerFilter { middle: Some(1) }
                        .apply(&mut tracks)
                        .map_err(error)?;
                    for (channel, track) in anim.channels.iter_mut().zip(tracks) {
                        channel.replace_track(track);
                    }
                }
                self.document.store_attrs(id, &attrs)?;
            }
            return Ok(());
        }
        for (id, path) in targets {
            self.document.assert_unlocked(id)?;
            let mut attrs = self.document.attrs(id)?;
            let (attr, channel) = WorldDocument::lane(&attrs, &path);
            if !attrs.is_animated(&attr)
                || (channel.is_none() && self.document.is_discrete(id, &attr)?)
            {
                continue;
            }
            WorldDocument::for_channels(&mut attrs, &attr, channel, |_, ch| {
                let mut tracks = [ch.track().clone()];
                effect.apply(&mut tracks).map_err(error)?;
                let [track] = tracks;
                ch.replace_track(track);
                Ok::<(), String>(())
            })?;
            self.document.store_attrs(id, &attrs)?;
        }
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
                    auto_key.then_some(editor.new_key),
                )?;
            }
            // Navigation mode is a static preference on the camera, never a generated key.
            if WorldDocument::driven(&attrs, "/camera/free_flight") {
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
            if node.kind == WorldKind::DirectionalLight {
                // A scene holds the first sun in world space (`sun_light`): writing it back would
                // copy it onto every Sun and apply their parents twice. Suns edit as attributes.
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
                    && a != b
                {
                    commands.push(WorldCommand::SetAttribute {
                        id: node.id,
                        path: attr.path.clone(),
                        value: b.clone(),
                        frame,
                    });
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
            let viewport = self.document.viewport_settings_id()?;
            for (path, target) in self.document.render_edit_paths(render, viewport, frame)? {
                let scene_path = path.replacen("/quality/", "/render/", 1);
                if let (Some(a), Some(b)) =
                    (before.pointer(&scene_path), after.pointer(&scene_path))
                    && a != b
                {
                    commands.push(WorldCommand::SetAttribute {
                        id: target,
                        path,
                        value: b.clone(),
                        frame,
                    });
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
/// The values attribute `path` may take, in menu order; empty for a free value. Every enum lists
/// its own variants (`T::ALL`), so the choices follow the types: a new variant cannot be
/// missing from its menu.
fn attribute_choices(path: &str) -> Result<Vec<Value>, String> {
    // A variant that failed to serialize would silently vanish from its menu and become
    // unsettable (`set_attribute` rejects values outside the choices): an error instead.
    fn variants<T: Serialize>(all: impl IntoIterator<Item = T>) -> Result<Vec<Value>, String> {
        all.into_iter()
            .map(|v| serde_json::to_value(v).map_err(|e| e.to_string()))
            .collect()
    }
    use crate::scene::{
        ArrayAxis, ArrayKind, ColorSource, Coloring, HybridStep, KifsKind, MaskShape, MaterialModel,
    };
    if path == "/formula" {
        return variants(crate::scene::Formula::PRESETS);
    }
    if let Some(choices) = crate::formulas::formula_param_choices(path) {
        return Ok(choices);
    }
    match path {
        "/render/backend" => variants(crate::render_profiles::Backend::ALL),
        "/render/method" => variants(crate::render_profiles::RenderMethod::ALL),
        "/render/display" => variants(crate::render_profiles::Display::ALL),
        "/viewport/mode" => variants(crate::render_profiles::ViewportMode::ALL),
        "/palette" => variants(crate::palette::PaletteScheme::ALL),
        "/coloring" => variants(Coloring::ALL),
        "/mask/shape" => variants(MaskShape::ALL),
        "/mask/array/kind" => variants(ArrayKind::ALL),
        "/mask/array/axis" => variants(ArrayAxis::ALL),
        "/render/denoise/mode" => variants(crate::denoise::Mode::ALL),
        "/render/denoise/quality" => variants(crate::denoise::Quality::ALL),
        "/material/model" => variants(MaterialModel::ALL),
        "/material/color_source" => variants(ColorSource::ALL),
        p if p.ends_with("/kind") && p.starts_with("/formula") => variants(KifsKind::ALL),
        // One Hybrid step slot: Off skips it, the rest run their sub-formula. The whole
        // `/formula/Hybrid/steps` array keeps no choices: it edits slot by slot
        // (`set_attribute` rejects a whole value outside choices).
        p if p.starts_with("/formula/Hybrid/steps/") => variants(HybridStep::ALL),
        _ => Ok(Vec::new()),
    }
}
pub(crate) fn attribute_range(path: &str) -> Option<(f64, f64)> {
    if let Some(range) = crate::formulas::formula_param_range(path) {
        return Some(range);
    }
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
        "/mask/size" => Some((1.0e-6, 1.0e6)),
        // Bounds the time warp's integration span (`WorldDocument::take_frame`).
        TAKE_OFFSET => Some((-100_000.0, 100_000.0)),
        TAKE_SPEED => Some((-1000.0, 1000.0)),
        "/mask/density"
        | "/render/thinning"
        | "/render/direct/shadow_strength"
        | "/render/direct/ao_strength" => Some((0.0, 1.0)),
        "/render/direct/shadow_steps" | "/render/direct/ao_steps" => Some((0.0, 256.0)),
        "/render/direct/ao_radius" => Some((1.0e-4, 1.0e4)),
        "/mask/seed" => Some((0.0, 16_777_215.0)),
        "/mask/array/count" => Some((1.0, fractal_core::params::ARRAY_MAX as f64)),
        "/mask/array/sweep_degrees" => Some((0.0, 360.0)),
        "/mask/array/radius" => Some((1.0e-4, 1.0e4)),
        "/viewport/target_fps" => Some((1.0, 240.0)),
        "/viewport/settle_delay_ms" => Some((0.0, 10000.0)),
        "/viewport/batch_budget_ms" => Some((0.1, 1000.0)),
        "/camera/fov_y_degrees" => Some((1.0, 179.0)),
        "/camera/f_number" => Some((0.0, f32::MAX as f64)),
        "/camera/sensor_height" => Some((0.000001, f32::MAX as f64)),
        // A zero radius would put the camera on its target whatever its distance.
        CAMERA_FRAMING => Some((1.0e-6, f32::MAX as f64)),
        // Axis index of the trap plane (X, Y, Z); the kernel reads min(2).
        "/trap_axis" => Some((0.0, 2.0)),
        // The palette coordinate is 1 - exp(-trap * scale): a negative scale pins it at 0.
        "/trap_scale" => Some((0.0, f32::MAX as f64)),
        // Blend weights: outside 0..1 the mix and the inversion extrapolate past both forms.
        "/julia_mix" | "/inversion_strength" => Some((0.0, 1.0)),
        // A zero radius collapses the full inversion (no stretch to divide the estimate by).
        "/inversion_radius" => Some((1.0e-6, 1.0e6)),
        // The kernels split the count into whole + fraction; f32 counts are exact below 2^24.
        "/render/iterations" => Some((0.0, 16_777_216.0)),
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
        TAKE_CLIP => return "Take".into(),
        TAKE_OFFSET => return "Take offset".into(),
        TAKE_SPEED => return "Take speed".into(),
        "/camera/f_number" => return "f-number".into(),
        "/formula/Hybrid/mandelbox/min_radius_ratio" => return "Mandelbox · Min ratio".into(),
        "/formula/Hybrid/mandelbox/rotation_degrees" => return "Mandelbox · Rotate".into(),
        "/formula/Hybrid/steps/0" => return "Step 1".into(),
        "/formula/Hybrid/steps/1" => return "Step 2".into(),
        "/formula/Hybrid/steps/2" => return "Step 3".into(),
        "/formula/Hybrid/steps/3" => return "Step 4".into(),
        "/material/transmission_extra_roughness" => return "Refraction roughness".into(),
        CAMERA_ORBIT_SPEED => return "Orbit speed (°/s)".into(),
        CAMERA_ORBIT_PHASE => return "Orbit phase (°)".into(),
        "/transform/position" => return "Translate".into(),
        "/transform/rotation_degrees" => return "Rotate".into(),
        "/transform/scale" => return "Scale".into(),
        "/output/preset" => return "Encode recipe".into(),
        "/output/output_display" => return "Display override".into(),
        "/output/output_view" => return "View override".into(),
        "/output/denoise_at_completion" => return "Denoise at completion".into(),
        _ => {}
    }
    if let Some((variant, _)) = crate::formulas::parse_formula_attr_path(path)
        && variant != "Hybrid"
        && crate::formulas::find_formula_meta_by_variant(variant).is_some_and(|m| m.code >= 8)
        && let Some(label) = crate::formulas::formula_param_label(path)
    {
        return label;
    }
    let parts: Vec<_> = path.trim_start_matches('/').split('/').collect();
    let tail = match parts.as_slice() {
        ["formula", "Hybrid", rest @ ..] => rest,
        ["formula", _, rest @ ..] if !rest.is_empty() => rest,
        ["mask", "array", rest @ ..] if !rest.is_empty() => rest,
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
    // The time warp exists only on nodes a take was attached to: a fresh world has none.
    match path {
        TAKE_OFFSET => return Some(json!(0.0)),
        TAKE_SPEED => return Some(json!(1.0)),
        _ => {}
    }
    static FRESH: LazyLock<HashMap<(WorldKind, String), Value>> = LazyLock::new(|| {
        let world = WorldDocument::from_scene(&Scene::preset(Formula::Mandelbulb(Bulb::PRESET)));
        let mut defaults = HashMap::new();
        for node in world.nodes() {
            let Ok(attrs) = world.attrs(node.id) else {
                continue;
            };
            // The first node of a kind defines it (one camera, one sun, ...).
            for (path, _) in attrs.iter() {
                if let Ok(value) = world.attribute_value(node.id, path, 0.0) {
                    defaults.entry((node.kind, path.clone())).or_insert(value);
                }
            }
        }
        for formula in crate::scene::Formula::PRESETS {
            if let Ok(formula_json) = serde_json::to_value(formula) {
                let mut f_attrs = Attrs::new();
                discover(&formula_json, "/formula", &mut f_attrs);
                for (p, v) in f_attrs.iter() {
                    if p != "/formula" {
                        defaults
                            .entry((WorldKind::Fractal, p.clone()))
                            .or_insert_with(|| attr_json(v.clone()));
                    }
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
    if let Some((min, max)) =
        attribute_range(path).filter(|_| crate::render_profiles::static_branch(path))
    {
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
        // A Hybrid's sub-formula slides like its standalone family.
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
        "/mask/array/count" => lin(1.0, fractal_core::params::ARRAY_MAX as f64),
        "/mask/array/offset" => lin(-16.0, 16.0),
        "/mask/array/radius" => log(0.01, 32.0),
        "/mask/array/sweep_degrees" => lin(0.0, 360.0),
        "/transform/position" | "/transform/pivot" => lin(-10.0, 10.0),
        "/transform/rotation_degrees" => lin(-180.0, 180.0),
        "/transform/scale" => log(0.01, 10.0),
        "/render/thinning" | "/render/direct/shadow_strength" | "/render/direct/ao_strength" => {
            lin(0.0, 1.0)
        }
        "/render/direct/shadow_steps" => lin(0.0, 64.0),
        "/render/direct/ao_steps" => lin(0.0, 16.0),
        "/render/direct/ao_radius" => log(0.01, 4.0),
        TAKE_OFFSET => lin(-240.0, 240.0),
        TAKE_SPEED => lin(-4.0, 4.0),
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
        CAMERA_FRAMING => log(0.01, 100.0),
        "/environment/intensity" => lin(0.0, 10.0),
        "/environment/rotation_degrees" => lin(-180.0, 180.0),
        "/sky/intensity" | "/sun/intensity" => lin(0.0, 20.0),
        // Angular diameter in degrees (the sun is ~0.53).
        "/sun/angle" => lin(0.0, 10.0),
        "/sun/azimuth" => lin(0.0, 360.0),
        "/sun/elevation" => lin(-90.0, 90.0),
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
        _ => {
            return crate::formulas::find_formula_meta_by_variant(family)
                .and_then(|fm| fm.params.iter().find(|p| p.name == field))
                .map(|p| p.hint);
        }
    })
}

/// What an attribute does, for its label's hover text (`AttrField::hint`): the one table
/// every attribute view reads (Attribute Editor, Render settings). Formula parameters match by
/// their tail, so a Hybrid's Mandelbulb / Mandelbox / KIFS step shares the standalone text.
/// None: the label says it all.
pub(crate) fn attribute_hint(path: &str) -> Option<&'static str> {
    // The viewport's kind filter: one hint per switch family, whatever the kind.
    if path.starts_with("/viewport/show/") {
        return Some(
            "Show this kind of node in the viewport: off leaves it out of the viewport's picture and extras (a mask keeps cutting). Exports show everything.",
        );
    }
    if path.starts_with("/viewport/select/") {
        return Some("Let the viewport's manipulator take this kind of node.");
    }
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
        "/render/display" => {
            "What this profile shows of the fractals: path traced; Direct, one fast noise-free pass (key lights with soft shadows, ambient occlusion, the sky's soft light; no bounces, done after the largest square number of samples up to Samples, at most 16 x 16 = 256); or their bounds alone (nothing traced, drawn by the viewport only; refused on the Output profile)."
        }
        "/render/direct/shadow_strength" => {
            "Direct: how dark a light's shadow gets, 0 (no shadow) to 1 (black)."
        }
        "/render/direct/shadow_steps" => {
            "Direct: march steps of each shadow ray; more find thinner occluders. 0 turns shadows off."
        }
        "/render/direct/ao_strength" => {
            "Direct: ambient occlusion, how much nearby surfaces darken the sky light, 0 (none) to 1."
        }
        "/render/direct/ao_steps" => {
            "Direct: occlusion probes per direction (five directions). 0 turns occlusion off."
        }
        "/render/direct/ao_radius" => "Direct: how far the occlusion probes reach, in scene units.",
        "/render/thinning" => {
            "Leaves part of every fractal out, 0 (all of it) to 1 (none): a 3D grid of cells is removed at random, so the march has less to trace."
        }
        "/render/method" => {
            "Fast evaluates opaque materials with the existing simplified material model; transmission keeps Standard Surface. Full preserves each authored material model. Quality remains controlled by the profile's Quality section."
        }
        "/render/backend" => {
            "Compute backend: Auto uses CUDA when an NVIDIA driver is present and falls back to WGPU; CUDA and WGPU require that backend without fallback."
        }
        "/viewport/extras" => {
            "Draw the extras over the image: mask shapes, group locators, the other cameras and the lights."
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
        "/output/preset" => {
            "The encode recipe (format, size, codec, quality, still options): edited as a whole in Render / Encode."
        }
        "/output/denoise_at_completion" => {
            "Run OIDN once after all samples of each exported frame, overriding the Output profile's denoise cadence."
        }
        "/output/output_display" | "/output/output_view" => {
            "OCIO display / view overriding the automatic output transform; empty is automatic."
        }
        "/formula" => "The fractal family and its parameters.",
        "/julia" => {
            "Julia constant J (Mandelbulb, Mandelbox): the orbit constant the Julia mix blends towards."
        }
        "/julia_mix" => {
            "Mandelbrot to Julia (Mandelbulb, Mandelbox): every iteration adds mix(point, Julia constant, this). 0 is the Mandelbrot form, 1 the Julia form; keys morph between them."
        }
        "/orbit_offset" => {
            "Constant added every iteration on top of the orbit constant (Mandelbulb, Mandelbox): z -> z^p + c + offset. Shifts and reshapes the set."
        }
        "/inversion_strength" => {
            "Spherical inversion of space before iterating (every formula): 0 off, 1 the full inversion in the sphere below, which turns the set inside out; between them a continuous blend."
        }
        "/inversion_center" => "Centre of the inversion sphere, in the fractal's own space.",
        "/inversion_radius" => {
            "Radius of the inversion sphere: points on it stay in place, nearer ones go far and far ones come near."
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
        TAKE_CLIP => {
            "The take whose recorded motion this node plays instead of its keys (Takes panel)."
        }
        TAKE_OFFSET => "Frames the take starts after its recorded start frame.",
        TAKE_SPEED => {
            "Playback rate of the take: keyable, so it ramps (0 holds, negative plays back)."
        }
        "/camera/target" => "The point the camera looks at and orbits.",
        "/mask/shape" => {
            "Sphere (radius Size X), Box (half extents Size) or Cells (cubes of edge Size X, a Density share removed)."
        }
        "/mask/size" => {
            "Sphere radius (X), box half extents, or cell edge (X), in the mask's own space."
        }
        "/mask/density" => "Cells: the share of cells removed, 0 keeps all, 1 removes all.",
        "/mask/seed" => "Cells: which cells are removed.",
        "/mask/invert" => "Cut the shape away instead of keeping only what is inside it.",
        "/mask/array/kind" => {
            "Linear steps copies by Offset (Blender Array Line). Radial places them on a ring (Blender Array Circle)."
        }
        "/mask/array/count" => "How many copies. 1 is the single shape.",
        "/mask/array/offset" => {
            "Linear: the step to the next copy, in the mask's own space. The first copy stays at the origin."
        }
        "/mask/array/radius" => "Radial: distance from the mask origin to each copy.",
        "/mask/array/axis" => "Radial: the axis the ring stands on (Blender Central Axis).",
        "/mask/array/sweep_degrees" => {
            "Radial arc in degrees. 0 is a full circle, copies spaced by 360 / count."
        }
        "/mask/array/align" => {
            "Radial boxes turn so their X axis faces out from the ring. A sphere is unchanged."
        }
        "/camera/yaw_degrees" => "Camera heading around world up.",
        "/camera/pitch_degrees" => "Camera tilt up / down.",
        "/camera/roll_degrees" => "Camera roll around the view axis.",
        "/camera/distance" => {
            "Distance from the target in framing radii (Framing radius; 1 frames the fractal)."
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
        CAMERA_FRAMING => {
            "The scene size the distance and focus distance are measured in: 1 distance is this far. Set from the fractal the camera was made to frame."
        }
        "/environment/enabled" => {
            "Light the scene with the environment image (else the sky gradient)."
        }
        "/environment/path" => "Lat-long HDR / EXR environment image.",
        "/environment/intensity" => "Environment brightness multiplier.",
        "/environment/rotation_degrees" => "Turns the environment around world up.",
        "/sun/azimuth" => "Sun direction around world up.",
        "/sun/elevation" => "Sun height above the horizon (negative: below).",
        "/sun/color" => "Sun colour.",
        "/sun/intensity" => "Sun brightness.",
        "/sun/angle" => {
            "Sun disc diameter in degrees: larger gives softer shadows (the real sun is ~0.53)."
        }
        "/sky/intensity" => "Sky light brightness.",
        "/sky/horizon" => "Sky colour at the horizon.",
        "/sky/zenith" => "Sky colour straight up.",
        "/sky/background" => {
            "Show the sky behind the fractal; off keeps its light but renders the background black."
        }
        "/sky/sun_disc" => {
            "Show the suns' discs in the sky behind the fractal; off keeps their light and reflections."
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
            "Fractal iterations per distance estimate: more resolves finer detail and costs proportionally. Fractional counts blend into the next iteration, so keyed counts grow the detail smoothly."
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
        "/render/denoise/quality" => "OIDN network: Fast (small) or Balanced (standard).",
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
        "/shy" => {
            "Shy: hidden from the timeline while its Shy switch is on. Settings nodes start shy."
        }
        "/start" => "First frame this node exists on.",
        "/end" => "Frame this node ends before (exclusive): it exists up to the frame before.",
        _ => return None,
    })
}

/// Why a node's parameter does nothing in its current mode (shown greyed in the Attribute
/// Editor), or None when it is in use. `value` reads the node's other attributes. The rules follow
/// what `Scene::pack` and the kernels read: the orbit constant (Julia mix and constant, offset)
/// only for Mandelbulb / Mandelbox and the constant only at a nonzero mix, the inversion sphere
/// only at a nonzero strength, the trap parameters per colouring mode, a Hybrid's sub-formulas
/// only when one of its steps runs them.
pub(crate) fn inactive_reason<'a>(
    path: &str,
    value: impl Fn(&str) -> Option<&'a Value>,
) -> Option<String> {
    // `path` is `prefix` or one of its components; no allocation (runs per row per frame).
    let under = |prefix: &str| {
        path.strip_prefix(prefix)
            .is_some_and(|rest| rest.is_empty() || rest.starts_with('/'))
    };
    if under("/render/direct") {
        return (value("/render/display")?.as_str()? != "Direct")
            .then(|| "Only the Direct display shades with these".into());
    }
    if under("/mask/array") {
        if value("/mask/shape")?.as_str()? == "Cells" {
            return Some("A cell grid is not an array of copies".into());
        }
        let kind = value("/mask/array/kind")?.as_str()?;
        let radial = under("/mask/array/radius")
            || under("/mask/array/axis")
            || under("/mask/array/sweep_degrees")
            || under("/mask/array/align");
        if kind == "Linear" && radial {
            return Some("A linear array steps by its offset".into());
        }
        if kind == "Radial" && under("/mask/array/offset") {
            return Some("A radial array steps around its axis".into());
        }
        return None;
    }
    if under("/julia") || under("/julia_mix") || under("/orbit_offset") {
        let family = value("/formula")?.as_object()?.keys().next()?.clone();
        let has_orbit = crate::formulas::find_formula_meta_by_variant(&family)
            .map(|m| m.has_orbit_constant)
            .unwrap_or_else(|| matches!(family.as_str(), "Mandelbulb" | "Mandelbox"));
        if !has_orbit {
            return Some(format!("The {family} formula has no orbit constant"));
        }
        return (under("/julia") && value("/julia_mix")?.as_f64()? == 0.0)
            .then(|| "Julia mix is 0".into());
    }
    if under("/inversion_center") || under("/inversion_radius") {
        return (value("/inversion_strength")?.as_f64()? == 0.0)
            .then(|| "Inversion strength is 0".into());
    }
    if under("/trap_point") || under("/trap_axis") || under("/trap_scale") {
        let coloring = value("/coloring")?.as_str()?;
        // fractal-kernels trap3 / hit_palette: Radius reads none, TrapOrigin only the scale, TrapPlane
        // all three, TrapPoint the point and the scale.
        let used = match coloring {
            "Radius" => false,
            "TrapOrigin" => under("/trap_scale"),
            "TrapPoint" => !under("/trap_axis"),
            _ => true,
        };
        return (!used).then(|| format!("Not used by the {coloring} colouring"));
    }
    if path == "/render/iterations" {
        let family = value("/formula")?.as_object()?.keys().next()?;
        if let Some(meta) = crate::formulas::find_formula_meta_by_variant(family)
            && !meta.iterates
        {
            return Some(format!(
                "{} is analytical (no iteration loop)",
                meta.display_name
            ));
        }
        return None;
    }
    if let Some(reason) = crate::formulas::formula_param_inactive_reason(path, &value) {
        return Some(reason);
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
        "/sun/color"
            | "/sky/horizon"
            | "/sky/zenith"
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

/// Tangent for a value written at `frame`: a key that already exists keeps its out kind (editing a
/// value must not reset its interpolation), a new key takes `new` (Settings > Animation).
fn key_tan(ch: &Channel, frame: f64, new: Tan) -> Tan {
    ch.keys()
        .iter()
        .find(|k| k.t() == frame)
        .map_or(new, |k| k.out.kind)
}

/// A number clamped to the attribute's hard limits; other values pass through. The number keeps
/// its type (`set_attribute` has already given it the attribute's): the limits of an integer
/// attribute are whole.
fn clamp_to_range(path: &str, value: Value) -> Value {
    match (attribute_range(path), value.as_f64()) {
        (Some((min, max)), Some(v)) if v < min || v > max => {
            numeric_like(&value, v.clamp(min, max))
        }
        _ => value,
    }
}

/// `v` as a number of `template`'s type: rounded to an integer when `template` is one (unsigned
/// stays at or above 0), else a float. The ONE rule of "a number like this one": attribute edits
/// (`set_attribute`), settings evaluation, the stored-value writes of `set_pointer` and the
/// Attribute Editor's typed fields (`world_ui::grid_json`).
pub(crate) fn numeric_like(template: &Value, v: f64) -> Value {
    if template.is_u64() {
        json!(v.round().max(0.0) as u64)
    } else if template.is_i64() {
        json!(v.round() as i64)
    } else {
        json!(v)
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
/// The scene branches a node of `kind` authors: its attributes (`discover`) and its stored `gpu`
/// template (`node_template`) hold these and nothing else. A Sun has its light, the Environment
/// its image and the sky.
fn branches(kind: WorldKind) -> &'static [&'static str] {
    match kind {
        WorldKind::Fractal => &[
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
        ],
        WorldKind::Camera => &["camera"],
        WorldKind::DirectionalLight => &["sun"],
        WorldKind::Environment => &["environment", "sky"],
        WorldKind::Material => &["material"],
        WorldKind::Group => &[],
        WorldKind::RenderSettings => &["render", "quality"],
        WorldKind::ViewportSettings => &["viewport", "render"],
        WorldKind::OutputSettings => &["output"],
        WorldKind::Mask => &["mask"],
    }
}
fn discover(value: &Value, path: &str, attrs: &mut Attrs) {
    match value {
        // The export recipe (`video_io::job::Preset`) is one atomic attribute: its enum objects
        // (`rate: {"Crf": n}`) and optional fields change their leaf set with the value, and the
        // Render / Encode dialog edits it as a whole (`render_profiles::output_edits`).
        Value::Object(_) if path == "/output/preset" => attrs.set(path, to_attr(value)),
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
/// The fields of stored node data, as `WorldDocument::insert` writes them; a node that plays a
/// take also has `take` (`attach_take`).
const NODE_FIELDS: [&str; 10] = [
    "type",
    "name",
    "parent",
    "children",
    "host",
    "gpu",
    "discrete",
    "metadata",
    "material",
    "environment_revision",
];

/// Stored node data has this shape before anything indexes into it: an object with exactly the
/// `NODE_FIELDS` (and an optional `take`), a known `type`, `host`, `gpu`, `metadata` and
/// `discrete` objects, `children` an array, and a `gpu` holding exactly its kind's
/// `branches`. Decoding (`WorldDocument::decode_json`) and pasting
/// (`WorldEditor::insert_fragment`) check it, so a malformed, hand-edited or old-format node is
/// an error instead of a panic on assignment or a silently missing value.
fn node_shape(key: &str, data: &Value) -> Result<(), String> {
    let node = data
        .as_object()
        .ok_or_else(|| format!("Node {key} is not an object"))?;
    if let Some(field) = NODE_FIELDS.iter().find(|f| !node.contains_key(**f)) {
        return Err(format!("Node {key}: missing {field}"));
    }
    if let Some(field) = node
        .keys()
        .find(|k| k.as_str() != "take" && !NODE_FIELDS.contains(&k.as_str()))
    {
        return Err(format!("Node {key}: unknown field {field}"));
    }
    for field in ["host", "gpu", "metadata", "discrete"] {
        if !node.get(field).is_some_and(Value::is_object) {
            return Err(format!("Node {key}: {field} must be an object"));
        }
    }
    if !node.get("children").is_some_and(Value::is_array) {
        return Err(format!("Node {key}: children must be an array"));
    }
    let kind: WorldKind = serde_json::from_value(node.get("type").cloned().unwrap_or_default())
        .map_err(|e| format!("Node {key}: type: {e}"))?;
    let expected = branches(kind);
    let gpu = node.get("gpu").and_then(Value::as_object);
    if gpu.is_none_or(|gpu| {
        gpu.len() != expected.len() || expected.iter().any(|b| !gpu.contains_key(*b))
    }) {
        return Err(format!(
            "Node {key}: gpu must hold exactly the {kind:?} branches {expected:?}"
        ));
    }
    Ok(())
}

/// Every animation of `attrs` has one channel per component of its value, and every navigation
/// offset (`/_navigation/<path>`) the shape of its channel: the key commands and the navigation
/// address channels by component. Checked with `node_shape`.
fn check_arity(attrs: &Attrs) -> Result<(), String> {
    let width = |value: &AttrValue| attr_json(value.clone()).as_array().map_or(1, Vec::len);
    for (path, value) in attrs.iter() {
        if let Some(anim) = attrs.anim(path)
            && anim.channels.len() != width(value)
        {
            return Err(format!(
                "{path} has {} animation channels for a value of {}",
                anim.channels.len(),
                width(value)
            ));
        }
        if let Some(channel) = path.strip_prefix("/_navigation")
            && let Some(base) = attrs.get(channel)
            && width(base) != width(value)
        {
            return Err(format!("Navigation offset {path} does not match {channel}"));
        }
    }
    Ok(())
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
            // An existing number keeps its type (a keyed integer channel evaluates as float).
            *root = match value.as_f64() {
                Some(f) if !create && root.is_number() => numeric_like(root, f),
                _ => value,
            };
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
    /// Copy one standalone settings node (Render or Output) under a new UUID, name and role,
    /// and select it. `assigned`: the copy is about to be bound, so it cannot be a template.
    fn clone_settings(
        &mut self,
        kind: WorldKind,
        source: NodeId,
        name: String,
        role: crate::render_profiles::CatalogRole,
        assigned: bool,
    ) -> Result<NodeId, String> {
        if name.trim().is_empty() {
            return Err("Profile name cannot be empty".into());
        }
        if role == crate::render_profiles::CatalogRole::Template && assigned {
            return Err("Instantiate a template before assigning it".into());
        }
        self.document.require_settings(source, kind)?;
        let mut data = self.document.node(source)?.clone();
        data["parent"] = Value::Null;
        data["children"] = json!([]);
        let (roots, _) = self.insert_fragment(HashMap::from([(source.to_string(), data)]), true)?;
        let id = *roots.first().ok_or("Cloned settings node missing")?;
        self.document.node_mut(id)?["name"] = json!(name.trim());
        self.document.node_mut(id)?["metadata"]["catalog_role"] = json!(role);
        self.select_only(id);
        Ok(id)
    }
    fn create_output_settings(
        &mut self,
        source: NodeId,
        name: String,
        role: crate::render_profiles::CatalogRole,
        assign: bool,
    ) -> Result<(), String> {
        let output = self.clone_settings(WorldKind::OutputSettings, source, name, role, assign)?;
        if assign {
            self.apply(WorldCommand::SetOutputSettings(output))?;
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
        let render = self.clone_settings(
            WorldKind::RenderSettings,
            source,
            name,
            role,
            target.is_some(),
        )?;
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
    ) -> Result<Vec<WorldNodeInfo>, String> {
        self.settings_nodes(WorldKind::RenderSettings, role)
    }
    /// The catalog of one settings kind, optionally only Profiles or only Templates. A node with
    /// an invalid catalog role is an error, not silently left out of the catalog.
    pub fn settings_nodes(
        &self,
        kind: WorldKind,
        role: Option<crate::render_profiles::CatalogRole>,
    ) -> Result<Vec<WorldNodeInfo>, String> {
        let mut out = Vec::new();
        for node in self.nodes() {
            if node.kind != kind {
                continue;
            }
            if let Some(role) = role
                && self.catalog_role(node.id)? != Some(role)
            {
                continue;
            }
            out.push(node);
        }
        Ok(out)
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
    pub fn output_settings_id(&self) -> Result<NodeId, String> {
        self.settings_bus_id("output_settings", WorldKind::OutputSettings)
    }
    /// The assigned export recipe (`output_settings_of`).
    pub fn output_settings(&self) -> Result<crate::export::OutputSettings, String> {
        self.output_settings_of(self.output_settings_id()?)
    }
    /// The recipe of any OutputSettings node, strictly typed. The recipe's limits and
    /// cross-field rules (HEVC 4:2:0 needs even sizes) are export-start checks
    /// (`ExportSettings::validate`), so an edit can pass through such a state in the form.
    pub fn output_settings_of(&self, id: NodeId) -> Result<crate::export::OutputSettings, String> {
        self.require_settings(id, WorldKind::OutputSettings)?;
        // Static values: any time reads the same; the document start is a valid one.
        let values = self.settings_values(id, "output", f64::from(self.first))?;
        serde_json::from_value(values).map_err(|e| format!("Invalid OutputSettings {id}: {e}"))
    }
    fn static_settings_value_with(attrs: &Attrs, path: &str) -> Result<Value, String> {
        if Self::driven(attrs, path) {
            return Err(format!("Settings attribute {path} must be static"));
        }
        attrs
            .get(path)
            .cloned()
            .map(attr_json)
            .ok_or_else(|| format!("Missing settings attribute {path}"))
    }
    fn static_settings_reference_with(attrs: &Attrs, path: &str) -> Result<NodeId, String> {
        Self::static_settings_value_with(attrs, path)?
            .as_str()
            .and_then(NodeId::parse)
            .ok_or_else(|| format!("Settings reference {path} must be a UUID"))
    }
    /// The settings nodes a reference attribute `path` of `consumer` may point at (a template
    /// may reference templates too); None when `path` is not a settings reference.
    fn settings_choices(&self, consumer: NodeId, path: &str) -> Result<Option<Vec<Value>>, String> {
        use crate::render_profiles::CatalogRole;
        let Some(kind) = crate::render_profiles::reference_kind(path) else {
            return Ok(None);
        };
        let template = self.catalog_role(consumer)? == Some(CatalogRole::Template);
        let mut choices = Vec::new();
        for node in self.nodes() {
            if node.kind == kind
                && (template || self.catalog_role(node.id)? == Some(CatalogRole::Profile))
            {
                choices.push(json!(node.id));
            }
        }
        Ok(Some(choices))
    }
    fn settings_template(&self, kind: WorldKind, scene: &Scene) -> Result<Value, String> {
        use crate::render_profiles::{look_values, quality_values, render_values, viewport_values};
        let render = serde_json::to_value(scene.render).map_err(|e| e.to_string())?;
        match kind {
            WorldKind::RenderSettings => Ok(json!({
                "render": render_values(&render),
                "quality": quality_values(&render),
            })),
            WorldKind::ViewportSettings => {
                let profiles =
                    self.render_profiles(Some(crate::render_profiles::CatalogRole::Profile))?;
                let still = profiles
                    .first()
                    .ok_or("Create a RenderSettings profile before ViewportSettings")?
                    .id;
                Ok(json!({
                    "viewport": viewport_values(still, still),
                    "render": look_values(&render),
                }))
            }
            WorldKind::OutputSettings => Ok(json!({
                "output": serde_json::to_value(crate::export::OutputSettings::default())
                    .map_err(|e| e.to_string())?
            })),
            _ => Err("Not a settings node".into()),
        }
    }
    fn seed_settings(&mut self, scene: &Scene) -> Result<(), String> {
        use crate::render_profiles::RenderMethod;
        let still = self.insert(WorldKind::RenderSettings, "Still", scene, None)?;
        self.graph
            .bus_slots
            .insert("output_render".into(), json!(still));
        let moving = self.insert(WorldKind::RenderSettings, "Moving", scene, None)?;
        for (path, value) in [
            ("/render/method", json!(RenderMethod::Fast)),
            ("/quality/adaptive/enabled", json!(false)),
            ("/quality/samples", json!(8)),
            ("/quality/resolution_scale", json!(1.0)),
            ("/quality/max_bounces", json!(2)),
        ] {
            self.set_attribute(moving, path, value, 0.0, false, Tan::Constant)?;
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
        let output_file = self.insert(WorldKind::OutputSettings, "Output File", scene, None)?;
        self.graph
            .bus_slots
            .insert("output_settings".into(), json!(output_file));
        Ok(())
    }
    pub fn effective_render(
        &self,
        profile: NodeId,
        frame: f64,
    ) -> Result<crate::render_profiles::EffectiveRender, String> {
        use crate::render_profiles::{
            EffectiveRender, LOOK_RENDER_FIELDS, PROFILE_RENDER_FIELDS, QUALITY_RENDER_FIELDS,
            RenderMethod,
        };
        if !frame.is_finite() {
            return Err("Invalid render profile time".into());
        }
        self.require_settings(profile, WorldKind::RenderSettings)?;
        let profile_node = self.node(profile)?;
        let profile_attrs = self.attrs(profile)?;
        let viewport = self.viewport_settings_id()?;
        let viewport_node = self.node(viewport)?;
        let viewport_attrs = self.attrs(viewport)?;
        let method: RenderMethod = serde_json::from_value(self.attribute_value_with(
            profile,
            profile_node,
            &profile_attrs,
            "/render/method",
            frame,
        )?)
        .map_err(|e| format!("Invalid render method: {e}"))?;
        let mut values = serde_json::Map::new();
        // World kernels read geometry iterations from each object's block. The transient root
        // block carries a neutral value; it must never proxy a particular fractal's geometry.
        values.insert(
            "iterations".into(),
            json!(
                Scene::preset(Formula::Mandelbulb(Bulb::PRESET))
                    .render
                    .iterations
            ),
        );
        let evaluated_quality = json!({
            "quality": self.settings_values_with(profile, profile_node, &profile_attrs, "quality", frame)?
        });
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
        let evaluated = json!({
            "render": self.settings_values_with(profile, profile_node, &profile_attrs, "render", frame)?
        });
        for key in PROFILE_RENDER_FIELDS {
            values.insert((*key).into(), evaluated["render"][*key].clone());
        }
        let evaluated_look =
            self.settings_values_with(viewport, viewport_node, &viewport_attrs, "render", frame)?;
        for key in LOOK_RENDER_FIELDS {
            values.insert((*key).into(), evaluated_look[*key].clone());
        }
        let render: crate::scene::Render = serde_json::from_value(Value::Object(values))
            .map_err(|e| format!("Invalid effective RenderSettings: {e}"))?;
        let samples_value = self.attribute_value_with(
            profile,
            profile_node,
            &profile_attrs,
            "/quality/samples",
            frame,
        )?;
        let samples = samples_value
            .as_f64()
            .filter(|v| v.is_finite() && (1.0..=65536.0).contains(v))
            .ok_or("Samples must be from 1 to 65536")?
            .round() as u32;
        let resolution_scale =
            self.attribute_value_with(
                profile,
                profile_node,
                &profile_attrs,
                "/quality/resolution_scale",
                frame,
            )?
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
        let display: crate::render_profiles::Display =
            serde_json::from_value(self.attribute_value_with(
                profile,
                profile_node,
                &profile_attrs,
                "/render/display",
                frame,
            )?)
            .map_err(|e| format!("Invalid display: {e}"))?;
        let thinning = self
            .attribute_value_with(
                profile,
                profile_node,
                &profile_attrs,
                "/render/thinning",
                frame,
            )?
            .as_f64()
            .filter(|v| (0.0..=1.0).contains(v))
            .ok_or("Thinning must be between 0 and 1")? as f32;
        // Integer channels evaluate as float: round them to the shading's integer fields.
        let mut direct = evaluated["render"]["direct"].clone();
        for key in ["shadow_steps", "ao_steps"] {
            let steps = direct[key]
                .as_f64()
                .filter(|v| v.is_finite() && (0.0..=256.0).contains(v))
                .ok_or("Direct shadow and occlusion steps must be 0 to 256")?;
            direct[key] = json!(steps.round() as u32);
        }
        let direct: crate::render_profiles::DirectShading =
            serde_json::from_value(direct).map_err(|e| format!("Invalid Direct shading: {e}"))?;
        // Direct's image is final once every stratum of its grid has its one sample.
        let samples = if display == crate::render_profiles::Display::Direct {
            crate::render_profiles::DirectShading::grid(samples).pow(2)
        } else {
            samples
        };
        Ok(EffectiveRender {
            profile,
            method,
            render,
            samples,
            resolution_scale,
            display,
            direct,
            thinning,
        })
    }
    pub fn viewport_policy(
        &self,
        frame: f64,
    ) -> Result<crate::render_profiles::ViewportPolicy, String> {
        use crate::render_profiles::{Kinds, ViewportMode, ViewportPolicy};
        if !frame.is_finite() {
            return Err("Invalid viewport policy time".into());
        }
        let id = self.viewport_settings_id()?;
        let attrs = self.attrs(id)?;
        let value = |path: &str| Self::static_settings_value_with(&attrs, path);
        let reference = |path| -> Result<NodeId, String> {
            let target = Self::static_settings_reference_with(&attrs, path)?;
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
        let (mut shown, mut selectable) = (Kinds::ALL, Kinds::ALL);
        for (kind, name, _) in crate::render_profiles::VIEW_KINDS {
            shown = shown.with(kind, boolean(&format!("/viewport/show/{name}"))?);
            selectable = selectable.with(kind, boolean(&format!("/viewport/select/{name}"))?);
        }
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
            shown,
            selectable,
            extras: boolean("/viewport/extras")?,
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
        let node = self.node(id)?;
        let attrs = self.attrs(id)?;
        self.settings_values_with(id, node, &attrs, branch, frame)
    }
    fn settings_values_with(
        &self,
        id: NodeId,
        node: &Value,
        attrs: &Attrs,
        branch: &str,
        frame: f64,
    ) -> Result<Value, String> {
        let mut value = node["gpu"][branch].clone();
        if !value.is_object() {
            return Err(format!("Settings schema {branch} missing"));
        }
        let mut schema = Attrs::new();
        discover(&value, &format!("/{branch}"), &mut schema);
        for (path, base) in schema.iter() {
            let mut current = self.attribute_value_with(id, node, attrs, path, frame)?;
            let base = attr_json(base.clone());
            // Integer settings are unsigned (counts, steps); an unlimited one fits u32.
            let limits = attribute_range(path)
                .or_else(|| base.is_i64().then_some((0.0, f64::from(u32::MAX))));
            if let Some((min, max)) = limits {
                let number = current
                    .as_f64()
                    .filter(|v| v.is_finite() && (min..=max).contains(v))
                    .ok_or_else(|| format!("Settings value {path} is outside its limits"))?;
                if base.is_i64() {
                    current = numeric_like(&base, number);
                }
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
                if crate::render_profiles::static_path(path) && Self::driven(&attrs, path) {
                    return Err(format!("Settings attribute {path} must be static"));
                }
            }
        }
        self.output_render_profile()?;
        self.output_settings()?;
        self.viewport_policy(frame)?;
        for node in self
            .nodes()
            .into_iter()
            .filter(|node| crate::render_profiles::setting_kind(node.kind))
        {
            let role = self.catalog_role(node.id)?.ok_or("Settings role missing")?;
            match node.kind {
                WorldKind::RenderSettings => {
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
                    let look = self.settings_values(node.id, "render", frame)?;
                    for key in ["exposure_stops", "saturation"] {
                        if !look[key].as_f64().is_some_and(f64::is_finite) {
                            return Err(format!("Invalid viewport look {key}"));
                        }
                    }
                    if !look["reinhard"].is_boolean() {
                        return Err("Invalid viewport look reinhard".into());
                    }
                    let attrs = self.attrs(node.id)?;
                    for path in [
                        "/viewport/moving_id",
                        "/viewport/still_id",
                        "/viewport/manual_id",
                    ] {
                        let id = Self::static_settings_reference_with(&attrs, path)?;
                        if role == crate::render_profiles::CatalogRole::Profile {
                            self.require_live_settings(id, WorldKind::RenderSettings)?;
                        } else {
                            self.require_settings(id, WorldKind::RenderSettings)?;
                        }
                    }
                }
                WorldKind::OutputSettings => {
                    self.output_settings_of(node.id)?;
                }
                _ => {}
            }
        }
        Ok(())
    }
    fn guard_settings_delete(&self, remove: &HashSet<NodeId>) -> Result<(), String> {
        for key in ["output_render", "viewport_settings", "output_settings"] {
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
        viewport: NodeId,
        frame: f64,
    ) -> Result<Vec<(String, NodeId)>, String> {
        let mut paths = Vec::new();
        for id in [render, viewport] {
            paths.extend(
                self.attributes(id, frame)?
                    .into_iter()
                    .filter(|a| {
                        a.component.is_none()
                            && (a.path.starts_with("/render/") || a.path.starts_with("/quality/"))
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
    /// The take node `id` plays (`/take/clip`); None when it plays none.
    pub fn take_clip(&self, id: NodeId) -> Result<Option<u64>, String> {
        match &self.node(id)?["take"] {
            Value::Null => Ok(None),
            v => v
                .as_u64()
                .filter(|t| *t > 0)
                .map(Some)
                .ok_or_else(|| format!("Invalid take reference {v}")),
        }
    }

    /// The ids of the takes nodes play: retention and deletion keep them.
    pub fn takes_in_use(&self) -> Vec<u64> {
        self.graph
            .nodes
            .values()
            .filter_map(|data| data["take"].as_u64())
            .collect()
    }

    /// The nodes that play take `take`.
    pub fn take_users(&self, take: u64) -> Vec<NodeId> {
        self.nodes()
            .into_iter()
            .map(|n| n.id)
            .filter(|id| {
                self.node(*id)
                    .is_ok_and(|data| data["take"].as_u64() == Some(take))
            })
            .collect()
    }

    /// Every node that plays take `take` plays none.
    fn release_take(&mut self, take: u64) {
        for data in self.graph.nodes.values_mut() {
            if data["take"].as_u64() == Some(take) {
                data["take"] = Value::Null;
            }
        }
    }

    /// Whether node `id`'s time warp shows its take as recorded: offset 0 and speed 1, neither
    /// keyed nor connected.
    pub fn take_warp_identity(&self, id: NodeId) -> Result<bool, String> {
        let attrs = self.attrs(id)?;
        let constant = |path: &str| {
            (!Self::driven(&attrs, path))
                .then(|| attrs.get(path).cloned().map(attr_json))
                .flatten()
                .and_then(|v| v.as_f64())
        };
        Ok(constant(TAKE_OFFSET) == Some(0.0) && constant(TAKE_SPEED) == Some(1.0))
    }

    /// The motion node `id` shows from the take it plays, time warp included, as a take of its
    /// own: four samples per frame over the work area, recorded speed from `first`. What a bake
    /// of a warped take keys.
    pub fn shown_take(&self, id: NodeId) -> Result<crate::takes::Take, String> {
        use crate::takes::{Sample, Take};
        const PER_FRAME: u32 = 4;
        let source = self
            .take_frame(id, f64::from(self.first))?
            .ok_or("The node plays no take")?
            .0
            .source;
        let frames = (self.last - self.first) * PER_FRAME;
        let sample = |i: u32| -> Result<(f64, Arc<Take>, f64), String> {
            let frame = f64::from(self.first) + f64::from(i) / f64::from(PER_FRAME);
            let (take, shown) = self
                .take_frame(id, frame)?
                .ok_or("The node plays no take")?;
            Ok((
                f64::from(i) / (f64::from(PER_FRAME) * self.fps),
                take,
                shown,
            ))
        };
        let (first, last, fps) = (self.first, self.last, self.fps);
        let take = match self
            .take_frame(id, f64::from(first))?
            .ok_or("The node plays no take")?
            .0
            .kind()
        {
            crate::takes::TakeKind::Camera => Take::new(
                f64::from(first),
                source,
                (0..=frames)
                    .map(|i| {
                        let (seconds, take, shown) = sample(i)?;
                        Ok(Sample {
                            seconds,
                            pose: take
                                .camera_at(shown, first, last, fps)
                                .ok_or("Not a camera take")?,
                        })
                    })
                    .collect::<Result<Vec<_>, String>>()?,
            ),
            crate::takes::TakeKind::Transform => Take::new(
                f64::from(first),
                source,
                (0..=frames)
                    .map(|i| {
                        let (seconds, take, shown) = sample(i)?;
                        Ok(Sample {
                            seconds,
                            pose: take
                                .trs_at(shown, first, last, fps)
                                .ok_or("Not a transform take")?,
                        })
                    })
                    .collect::<Result<Vec<_>, String>>()?,
            ),
        }?;
        Ok(take)
    }

    /// Play take `clip` on node `id` (None: play none). The take must exist and be of the kind
    /// the node plays ([`take_kind`]); attaching gives the node its time warp (`/take/offset` 0,
    /// `/take/speed` 1) unless it has one already.
    pub fn attach_take(&mut self, id: NodeId, clip: Option<u64>) -> Result<(), String> {
        if let Some(clip) = clip {
            let take = self.takes.get(clip).ok_or(format!("No take {clip}"))?;
            let kind = self.info(id)?.kind;
            if take_kind(kind) != Some(take.kind()) {
                return Err(format!(
                    "Take \"{}\" cannot drive a {kind:?} node",
                    take.name
                ));
            }
            let mut attrs = self.attrs(id)?;
            let mut added = false;
            for (path, value) in [(TAKE_OFFSET, 0.0), (TAKE_SPEED, 1.0)] {
                if !attrs.contains(path) {
                    attrs.set(path, to_attr(&json!(value)));
                    added = true;
                }
            }
            if added {
                self.store_attrs(id, &attrs)?;
            }
        }
        self.node_mut(id)?["take"] = json!(clip);
        Ok(())
    }

    /// The take node `id` plays and the frame of the take's own timing it shows at document
    /// `frame`: the time warp. The take starts on its start frame plus `/take/offset` and
    /// advances by `/take/speed` per frame, integrated, so a keyed speed ramps the motion
    /// smoothly instead of jumping along it:
    /// `start + integral of speed(f) df from (start + offset) to frame` ([`Self::warp`]).
    /// A take id the scene does not have is an error.
    pub fn take_frame(
        &self,
        id: NodeId,
        frame: f64,
    ) -> Result<Option<(Arc<crate::takes::Take>, f64)>, String> {
        let Some(clip) = self.take_clip(id)? else {
            return Ok(None);
        };
        let take = self.takes.get(clip).cloned().ok_or_else(|| {
            format!(
                "{} plays take {clip}, which this scene does not have",
                self.info(id).map(|i| i.name).unwrap_or_default()
            )
        })?;
        let &[shown] = self.warp(id, &take, &[frame])?.as_slice() else {
            return Err("Time warp lost its frame".into());
        };
        Ok(Some((take, shown)))
    }

    /// The work-area frames node `id`'s take plays on, `[start, end)`: those whose warped take
    /// time lies within the take (held first / last poses are not playing). None when the node
    /// plays no take, or none of it inside the work area. What the timeline draws as its bar.
    pub fn take_span(&self, id: NodeId) -> Result<Option<(i64, i64)>, String> {
        let Some(clip) = self.take_clip(id)? else {
            return Ok(None);
        };
        let take = self
            .takes
            .get(clip)
            .cloned()
            .ok_or_else(|| format!("No take {clip}"))?;
        let frames: Vec<f64> = (self.first..=self.last).map(f64::from).collect();
        let shown = self.warp(id, &take, &frames)?;
        let mut span: Option<(i64, i64)> = None;
        for (frame, shown) in frames.iter().zip(shown) {
            let seconds = take.seconds_at(shown, self.first, self.last, self.fps);
            if (0.0..=take.duration()).contains(&seconds) {
                let frame = *frame as i64;
                span = Some(span.map_or((frame, frame), |(start, _)| (start, frame)));
            }
        }
        Ok(span.map(|(start, last)| (start, last + 1)))
    }

    /// The time warp of node `id` playing `take`: the frame of the take's own timing shown at
    /// each of `frames`, `anchor + integral of speed from (anchor + offset(f)) to f`, with
    /// `anchor` the take's play start. The ONE warp of `take_frame` and `take_span`.
    ///
    /// A static speed multiplies. A keyed (or connected) one is integrated by Simpson's rule on
    /// a half-frame grid swept once over every point asked for (error O(h^4) on a smooth speed;
    /// a key between samples rounds its corner), so the cost is linear in the span however many
    /// frames are asked; `attribute_range` bounds the offset and with it the span.
    fn warp(
        &self,
        id: NodeId,
        take: &crate::takes::Take,
        frames: &[f64],
    ) -> Result<Vec<f64>, String> {
        const H: f64 = 0.5;
        let node = self.node(id)?;
        let attrs = self.attrs(id)?;
        let number = |path: &str, at: f64| -> Result<f64, String> {
            self.attribute_value_with(id, node, &attrs, path, at)?
                .as_f64()
                .filter(|v| v.is_finite())
                .ok_or_else(|| format!("{path} must be a finite number"))
        };
        let anchor = take.play_start(self.first);
        let begins = frames
            .iter()
            .map(|f| Ok(anchor + number(TAKE_OFFSET, *f)?))
            .collect::<Result<Vec<f64>, String>>()?;
        if !Self::driven(&attrs, TAKE_SPEED) {
            return frames
                .iter()
                .zip(&begins)
                .map(|(f, begin)| Ok(anchor + number(TAKE_SPEED, *f)? * (f - begin)))
                .collect();
        }
        let speed = |at: f64| number(TAKE_SPEED, at);
        // Simpson on [a, b] given the speed at both ends: exact for cubics.
        let simpson = |a: f64, b: f64, fa: f64, fb: f64| -> Result<f64, String> {
            Ok((fa + 4.0 * speed((a + b) * 0.5)? + fb) * (b - a) / 6.0)
        };
        // Every point the integral is asked at (each frame and its warp start), swept in order:
        // the running integral from the grid's start reaches each one once.
        let mut points: Vec<(f64, usize)> = frames
            .iter()
            .chain(&begins)
            .copied()
            .enumerate()
            .map(|(slot, x)| (x, slot))
            .collect();
        points.sort_by(|a, b| a.0.total_cmp(&b.0));
        let lo = points.first().map_or(0.0, |p| p.0).floor();
        let mut integral = vec![0.0; points.len()];
        let (mut k, mut left, mut sum) = (0u32, speed(lo)?, 0.0);
        for (x, slot) in points {
            while lo + f64::from(k + 1) * H <= x {
                let a = lo + f64::from(k) * H;
                let right = speed(a + H)?;
                sum += simpson(a, a + H, left, right)?;
                (k, left) = (k + 1, right);
            }
            let a = lo + f64::from(k) * H;
            let partial = if x > a {
                simpson(a, x, left, speed(x)?)?
            } else {
                0.0
            };
            if let Some(at) = integral.get_mut(slot) {
                *at = sum + partial;
            }
        }
        let (ends, starts) = integral.split_at(frames.len());
        Ok(ends
            .iter()
            .zip(starts)
            .map(|(end, start)| anchor + end - start)
            .collect())
    }

    /// The local transform the transform take node `id` plays shows at `frame`; None when it
    /// plays none, or when it is a camera playing a camera take (that one replaces the visible
    /// camera pose in `snapshot`, not the node's transform). A camera take on another node is
    /// an error.
    pub fn take_trs(&self, id: NodeId, frame: f64) -> Result<Option<crate::takes::Trs>, String> {
        let Some((take, shown)) = self.take_frame(id, frame)? else {
            return Ok(None);
        };
        match take.trs_at(shown, self.first, self.last, self.fps) {
            Some(trs) => Ok(Some(trs)),
            None if self.info(id)?.kind == WorldKind::Camera => Ok(None),
            None => Err(format!(
                "{} plays camera take \"{}\"",
                self.info(id)?.name,
                take.name
            )),
        }
    }

    /// The active camera's visible pose on every frame of the work area from its own animation,
    /// a take it plays ignored: the timeline motion as take poses ("Take from animation").
    pub fn camera_frames(&self) -> Result<Vec<crate::scene::Camera>, String> {
        let mut unbound = self.clone();
        if let Some(camera) = self.active_camera {
            unbound.attach_take(camera, None)?;
        }
        (self.first..=self.last)
            .map(|frame| unbound.snapshot(f64::from(frame)).map(|scene| scene.camera))
            .collect()
    }

    /// `node`'s local transform on every frame of the work area from its own animation (a take
    /// it plays ignored): its timeline motion as transform take poses.
    pub fn trs_frames(&self, node: NodeId) -> Result<Vec<crate::takes::Trs>, String> {
        (self.first..=self.last)
            .map(|frame| self.authored_trs(node, f64::from(frame)))
            .collect()
    }

    /// `node`'s local transform at `frame` from its attributes and keys.
    pub fn authored_trs(&self, node: NodeId, frame: f64) -> Result<crate::takes::Trs, String> {
        let data = self.node(node)?;
        let attrs = self.attrs(node)?;
        let vec = |path: &str, default: [f32; 3]| -> Result<[f32; 3], String> {
            let v = self.attribute_value_with(node, data, &attrs, path, frame)?;
            Ok(v.as_array()
                .filter(|v| v.len() == 3)
                .map(|v| {
                    std::array::from_fn(|i| v[i].as_f64().unwrap_or(f64::from(default[i])) as f32)
                })
                .unwrap_or(default))
        };
        Ok(crate::takes::Trs {
            position: vec("/transform/position", [0.0; 3])?,
            rotation_degrees: vec("/transform/rotation_degrees", [0.0; 3])?,
            scale: vec("/transform/scale", [1.0; 3])?,
        })
    }

    /// Move the work area's last frame. Growing it also grows every node span that reached the
    /// old end or beyond it but stops short of the new one (a non-animated `/end` in
    /// `[old_end, new_end)`), so nothing that was visible at the old end vanishes in the added
    /// frames. Equality alone missed spans ending between the two (made while the work area was
    /// longer), and the scene then changed after a camera recording grew the work area.
    pub fn set_last(&mut self, last: u32) -> Result<(), String> {
        if last > self.last {
            let old_end = f64::from(self.last) + 1.0;
            let new_end = f64::from(last) + 1.0;
            for node in self.nodes() {
                let mut attrs = self.attrs(node.id)?;
                let end = attrs
                    .get("/end")
                    .cloned()
                    .map(attr_json)
                    .and_then(|v| v.as_f64());
                if !attrs.is_animated("/end")
                    && end.is_some_and(|end| (old_end..new_end).contains(&end))
                {
                    attrs.set("/end", to_attr(&json!(new_end)));
                    self.store_attrs(node.id, &attrs)?;
                }
            }
        }
        self.last = last;
        Ok(())
    }

    /// A fresh UUID: this document is a new one, not the template, preset, bookmark or file it
    /// was made from.
    pub fn renew_identity(&mut self) {
        self.graph.id = NodeId::new().uuid();
    }

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
            // A new scene starts with marks 1 and 2 on its range's ends: 1 / 2 jump there.
            marks: BTreeMap::from([(1, scene.animation.first), (2, scene.animation.last)]),
            takes: Default::default(),
            camera_slots: Default::default(),
        };
        document
            .seed_settings(scene)
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
    /// The stored `gpu` template of a new `kind` node made from `scene`: exactly its `branches`
    /// (a fractal's `render` holds only its iterations: the profile renders it). Every node
    /// evaluates onto this (`evaluated_node`); a snapshot lays a fractal's branches over the
    /// world's (`snapshot_with_render_profile`). Used by `insert`; `node_shape` requires a
    /// stored `gpu` to hold exactly these branches.
    fn node_template(&self, kind: WorldKind, scene: &Scene) -> Result<Value, String> {
        if crate::render_profiles::setting_kind(kind) {
            return self.settings_template(kind, scene);
        }
        let full = scene_json(scene)?;
        let mut template = serde_json::Map::new();
        for branch in branches(kind) {
            let value = match *branch {
                "render" => json!({"iterations": scene.render.iterations}),
                _ => full
                    .get(*branch)
                    .cloned()
                    .ok_or_else(|| format!("Scene has no {branch}"))?,
            };
            template.insert((*branch).into(), value);
        }
        Ok(Value::Object(template))
    }
    /// The attributes a new `kind` node made from `scene` starts with besides its branches: the
    /// layer flags, its span over the work area, the transform (a fractal's from the scene's
    /// object), and a camera's orbit and framing radius (the scene's fractal as it is now,
    /// `CAMERA_FRAMING`). Used by `insert`.
    fn base_attrs(&self, kind: WorldKind, scene: &Scene) -> Attrs {
        let mut attrs = Attrs::new();
        for (path, value) in [
            ("/visible", json!(true)),
            ("/locked", json!(false)),
            ("/solo", json!(false)),
            ("/shy", json!(crate::render_profiles::setting_kind(kind))),
            ("/start", json!(f64::from(self.first))),
            ("/end", json!(f64::from(self.last) + 1.0)),
            // `json!` reads a bare `[..]` as a JSON array: the parentheses keep the Rust array.
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
            attrs.set(
                CAMERA_FRAMING,
                to_attr(&json!(scene.formula.framing_radius())),
            );
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
        attrs
    }
    fn insert(
        &mut self,
        kind: WorldKind,
        name: &str,
        scene: &Scene,
        parent: Option<NodeId>,
    ) -> Result<NodeId, String> {
        let template = self.node_template(kind, scene)?;
        let mut attrs = self.base_attrs(kind, scene);
        for branch in branches(kind) {
            if let Some(v) = template.get(*branch) {
                discover(v, &format!("/{branch}"), &mut attrs);
            }
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
    /// Node `id`'s identity and layer flags, the flags as at the work area's first frame
    /// ([`Self::info_at`]).
    pub fn info(&self, id: NodeId) -> Result<WorldNodeInfo, String> {
        self.info_at(id, f64::from(self.first))
    }
    /// Node `id`'s identity and its layer flags at `frame`: a keyed `/visible` is its value there,
    /// so the outliner, the timeline and a render read one evaluated flag ([`Self::nodes_at`]).
    pub fn info_at(&self, id: NodeId, frame: f64) -> Result<WorldNodeInfo, String> {
        let data = self.node(id)?;
        let a = self.attrs(id)?;
        self.info_at_with(id, data, &a, frame)
    }
    fn info_with(&self, id: NodeId, data: &Value, a: &Attrs) -> Result<WorldNodeInfo, String> {
        self.info_at_with(id, data, a, f64::from(self.first))
    }
    fn info_at_with(
        &self,
        id: NodeId,
        data: &Value,
        a: &Attrs,
        frame: f64,
    ) -> Result<WorldNodeInfo, String> {
        let kind: WorldKind =
            serde_json::from_value(data["type"].clone()).map_err(|e| e.to_string())?;
        // A flag keyed at `frame` is its keyed value (a key turns its base into a discrete index).
        let flag = |path: &str, default: bool| {
            self.attribute_value_with(id, data, a, path, frame)
                .ok()
                .and_then(|v| v.as_bool())
                .unwrap_or(default)
        };
        Ok(WorldNodeInfo {
            id,
            kind,
            name: data["name"].as_str().unwrap_or("").into(),
            parent: data["parent"].as_str().and_then(NodeId::parse),
            visible: flag("/visible", true),
            locked: flag("/locked", false),
            solo: flag("/solo", false),
            shy: flag("/shy", crate::render_profiles::setting_kind(kind)),
            start: f64::from(a.get_float_or("/start", 0.0)),
            end: f64::from(a.get_float_or("/end", self.last as f32 + 1.0)),
        })
    }
    /// `ids` and all their descendants.
    fn subtree(&self, ids: &[NodeId]) -> HashSet<NodeId> {
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
        set
    }
    /// `ids` and all their descendants, as stored node data (UUID -> data).
    pub fn fragment(&self, ids: &[NodeId]) -> Result<HashMap<String, Value>, String> {
        let mut out = HashMap::new();
        for id in self.subtree(ids) {
            out.insert(id.to_string(), self.node(id)?.clone());
        }
        Ok(out)
    }
    /// References of a fragment that resolve in neither it nor this document, by node key and
    /// field: a material or parent, or a take. Take ids are per document: a fragment from another
    /// one (`same_document` false) resolves none; from this one a take resolves when the scene
    /// still has it, of the kind the node plays. What `WorldEditor::insert_fragment` clears.
    fn unresolved_references(
        &self,
        nodes: &HashMap<String, Value>,
        same_document: bool,
    ) -> Vec<(String, &'static str)> {
        let resolves = |id: &str| nodes.contains_key(id) || self.graph.nodes.contains_key(id);
        let mut out = Vec::new();
        for (key, data) in nodes {
            for field in ["material", "parent"] {
                if data[field].as_str().is_some_and(|id| !resolves(id)) {
                    out.push((key.clone(), field));
                }
            }
            if !data["take"].is_null() {
                let kind = serde_json::from_value::<WorldKind>(data["type"].clone())
                    .ok()
                    .and_then(take_kind);
                let fits = same_document
                    && data["take"]
                        .as_u64()
                        .and_then(|id| self.takes.get(id))
                        .is_some_and(|take| Some(take.kind()) == kind);
                if !fits {
                    out.push((key.clone(), "take"));
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
        serde_json::to_string(&json!({
            CLIPBOARD_KEY: CLIPBOARD_VERSION,
            "document": self.graph.id,
            "nodes": self.fragment(ids)?,
        }))
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
    /// Single capability check for the Material reference exposed by the node
    /// schema, shared by the Attribute Editor, gallery and assignment command.
    /// A storage placeholder exists on every node; it is not itself a capability.
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
    /// Every node in layer order, its flags as at the work area's first frame ([`Self::nodes_at`]).
    pub fn nodes(&self) -> Vec<WorldNodeInfo> {
        self.nodes_at(f64::from(self.first))
    }
    /// Every node in layer order, its layer flags at `frame`: what the outliner and the timeline
    /// list at the playhead.
    pub fn nodes_at(&self, frame: f64) -> Vec<WorldNodeInfo> {
        let mut nodes: Vec<_> = self
            .graph
            .nodes
            .keys()
            .filter_map(|id| NodeId::parse(id).and_then(|id| self.info_at(id, frame).ok()))
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
        let node = self.node(id)?;
        let attrs = self.attrs(id)?;
        let kind = self.info_with(id, node, &attrs)?.kind;
        let mut out = vec![];
        for (path, _) in attrs.iter() {
            if path.starts_with("/_navigation/") {
                continue;
            }
            if take_kind(kind).is_none() && path.starts_with("/transform/") {
                continue;
            }
            if kind == WorldKind::Fractal
                && path.starts_with("/render/")
                && path != "/render/iterations"
            {
                continue;
            }
            // A hybrid packs one rotation (the bulb's) and its own bailout
            // (`Scene::pack`): these sub-formula rows are never read — hide them
            // instead of a permanently grey row (`inactive_reason` keeps guarding
            // direct API writes).
            if matches!(
                path.as_str(),
                "/formula/Hybrid/mandelbox/rotation_degrees"
                    | "/formula/Hybrid/kifs/rotation_degrees"
                    | "/formula/Hybrid/bulb/bailout"
            ) {
                continue;
            }
            let keyable = !matches!(
                path.as_str(),
                "/locked" | "/solo" | "/shy" | "/start" | "/end"
            ) && !crate::render_profiles::static_path(path);
            out.push(WorldAttribute {
                path: path.clone(),
                label: attribute_label(path),
                value: self.attribute_value_with(id, node, &attrs, path, frame)?,
                frames: attrs.key_frames(path),
                keyable,
                component: None,
                choices: match self.settings_choices(id, path)? {
                    Some(choices) => choices,
                    None => attribute_choices(path)?,
                },
                choice_labels: vec![],
                range: attribute_range(path),
                slider: attribute_slider(path),
                color: is_color_attribute(path),
                default: keyable.then(|| attribute_default(kind, path)).flatten(),
            });
        }
        if let Some(plays) = take_kind(kind) {
            let takes: Vec<&crate::takes::Take> = self
                .takes
                .list()
                .iter()
                .map(AsRef::as_ref)
                .filter(|t| t.kind() == plays)
                .collect();
            out.push(WorldAttribute {
                path: TAKE_CLIP.into(),
                label: attribute_label(TAKE_CLIP),
                value: node["take"].clone(),
                frames: vec![],
                keyable: false,
                component: None,
                choices: std::iter::once(Value::Null)
                    .chain(takes.iter().map(|t| json!(t.id)))
                    .collect(),
                choice_labels: std::iter::once("None".to_owned())
                    .chain(takes.iter().map(|t| t.name.clone()))
                    .collect(),
                range: None,
                slider: None,
                color: false,
                default: Some(Value::Null),
            });
        }
        if self.supports_material(id) {
            out.push(WorldAttribute {
                path: "/material_id".into(),
                label: "Material".into(),
                value: node["material"].clone(),
                frames: vec![],
                keyable: false,
                component: None,
                choices: self
                    .nodes()
                    .into_iter()
                    .filter(|n| n.kind == WorldKind::Material)
                    .map(|n| json!(n.id))
                    .collect(),
                choice_labels: vec![],
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
        // Hybrid steps are an array of enums: one choice row per slot so each
        // sub-formula can be switched on (`inactive_reason` gates the rows by
        // step). The slots are not keyable: keying a component requires a numeric
        // channel (`set_attribute`), while steps key as a whole via `/formula`.
        if let Some(steps) = out
            .iter()
            .find(|a| a.path == "/formula/Hybrid/steps")
            .map(|a| a.value.clone())
            .and_then(|v| v.as_array().cloned())
        {
            for (slot, value) in steps.iter().enumerate() {
                let path = format!("/formula/Hybrid/steps/{slot}");
                out.push(WorldAttribute {
                    path: path.clone(),
                    label: attribute_label(&path),
                    value: value.clone(),
                    frames: vec![],
                    keyable: false,
                    component: Some(slot),
                    choices: attribute_choices(&path)?,
                    choice_labels: vec![],
                    range: None,
                    slider: None,
                    color: false,
                    default: None,
                });
            }
        }
        out.sort_by(|a, b| {
            let base_a = attribute_base_path(&a.path);
            let base_b = attribute_base_path(&b.path);
            attribute_order(base_a)
                .cmp(&attribute_order(base_b))
                .then_with(|| base_a.cmp(base_b))
                .then_with(|| a.component.cmp(&b.component))
                .then_with(|| a.path.cmp(&b.path))
        });
        Ok(out)
    }
    fn is_discrete(&self, id: NodeId, path: &str) -> Result<bool, String> {
        let a = self.attrs(id)?;
        Ok(!numeric(&attr_json(
            a.get(path).cloned().ok_or("Unavailable parameter")?,
        )) || self.node(id)?["discrete"].get(path).is_some())
    }
    pub fn attribute_value(&self, id: NodeId, path: &str, frame: f64) -> Result<Value, String> {
        let node = self.node(id)?;
        let attrs = self.attrs(id)?;
        self.attribute_value_with(id, node, &attrs, path, frame)
    }
    fn attribute_value_with(
        &self,
        id: NodeId,
        node: &Value,
        attrs: &Attrs,
        path: &str,
        frame: f64,
    ) -> Result<Value, String> {
        if let Some((parent, component)) = Self::component_path_with(attrs, path) {
            let value = self.attribute_value_with(id, node, attrs, &parent, frame)?;
            return value
                .get(component)
                .cloned()
                .ok_or("Invalid component".into());
        }
        let value =
            self.resolve_attribute_with(id, Some((node, attrs)), path, frame, &mut HashSet::new())?;
        Self::with_navigation_offset_with(node, attrs, path, value)
    }
    fn resolve_attribute(
        &self,
        id: NodeId,
        path: &str,
        frame: f64,
        visiting: &mut HashSet<(NodeId, String)>,
    ) -> Result<Value, String> {
        self.resolve_attribute_with(id, None, path, frame, visiting)
    }
    fn resolve_attribute_with(
        &self,
        id: NodeId,
        cached: Option<(&Value, &Attrs)>,
        path: &str,
        frame: f64,
        visiting: &mut HashSet<(NodeId, String)>,
    ) -> Result<Value, String> {
        if !frame.is_finite() {
            return Err("Invalid frame".into());
        }
        let node = match cached {
            Some((node, _)) => node,
            None => self.node(id)?,
        };
        if path == "/material_id" {
            return Ok(node["material"].clone());
        }
        if path == TAKE_CLIP {
            return Ok(node["take"].clone());
        }
        let owned_attrs;
        let a = match cached {
            Some((_, attrs)) => attrs,
            None => {
                owned_attrs = self.attrs(id)?;
                &owned_attrs
            }
        };
        if !a.is_animated(path) {
            for parent in ["/formula", "/material/facing"] {
                if path.starts_with(parent)
                    && path.as_bytes().get(parent.len()) == Some(&b'/')
                    && a.is_animated(parent)
                {
                    if !visiting.insert((id, path.into())) {
                        return Err("Attribute connection cycle".into());
                    }
                    let v =
                        self.resolve_attribute_with(id, Some((node, a)), parent, frame, visiting)?;
                    if let Some(value) = v.pointer(&path[parent.len()..]) {
                        return Ok(value.clone());
                    }
                }
            }
        }
        if let Some((source, source_path)) = Self::connection(a, path) {
            if !visiting.insert((id, path.into())) {
                return Err("Attribute connection cycle".into());
            }
            let value = self.resolve_attribute_with(source, None, &source_path, frame, visiting)?;
            return self.with_navigation_offset(source, &source_path, value);
        }
        let value = a
            .eval_at(path, frame)
            .ok_or_else(|| format!("Unavailable parameter {path}"))?;
        if let Some(dict) = node["discrete"].get(path).and_then(Value::as_array) {
            let index = attr_json(value).as_f64().unwrap_or(0.0).round().max(0.0) as usize;
            return dict
                .get(index)
                .cloned()
                .ok_or_else(|| format!("Invalid discrete key {path}"));
        }
        let mut value = attr_json(value);
        if let Some(anim) = a.anim(path)
            && let Some(values) = value.as_array_mut()
            && let Some(base) = a
                .get(path)
                .cloned()
                .map(attr_json)
                .and_then(|v| v.as_array().cloned())
        {
            for (index, ch) in anim.channels.iter().enumerate() {
                if ch.is_empty()
                    && let (Some(slot), Some(base)) = (values.get_mut(index), base.get(index))
                {
                    *slot = base.clone();
                }
            }
        }
        Ok(value)
    }
    fn component_path(&self, id: NodeId, path: &str) -> Result<Option<(String, usize)>, String> {
        let attrs = self.attrs(id)?;
        Ok(Self::component_path_with(&attrs, path))
    }
    fn component_path_with(attrs: &Attrs, path: &str) -> Option<(String, usize)> {
        if attrs.contains(path) {
            return None;
        }
        if let Some((parent, last)) = path.rsplit_once('/')
            && let Ok(component) = last.parse::<usize>()
            && let Some(value) = attrs.get(parent)
            && attr_json(value.clone())
                .as_array()
                .is_some_and(|v| component < v.len())
        {
            return Some((parent.into(), component));
        }
        None
    }
    /// The node and attribute a connection on `path` reads from; a component path keeps its
    /// component ("/a/1" connected through "/a" reads "<source>/1"). None: not connected.
    ///
    /// Connections (Playa `conn`) are an unfinished feature: nothing in the app makes one yet,
    /// only tests. This is the ONE place that reads them: evaluation (`resolve_attribute_with`),
    /// integration (`integrated_scalar`) and the "static" test ([`Self::driven`]) all follow it.
    fn connection(attrs: &Attrs, path: &str) -> Option<(NodeId, String)> {
        let (attr, channel) = Self::lane(attrs, path);
        let conn = attrs.conn(&attr)?;
        let source = match channel {
            Some(component) => format!("{}/{component}", conn.source_key),
            None => conn.source_key.clone(),
        };
        Some((NodeId(conn.source_layer), source))
    }
    /// Whether `path` varies over time or comes from elsewhere: keyed or connected. What a static
    /// setting, a constant time warp and the camera's navigation mode must not be.
    fn driven(attrs: &Attrs, path: &str) -> bool {
        attrs.is_animated(path) || Self::connection(attrs, path).is_some()
    }
    /// The attribute `path` names and the animation channel it selects: a component path
    /// ("/transform/position/1") names its vector and that channel, any other path the attribute
    /// and all of its channels (None).
    fn lane(attrs: &Attrs, path: &str) -> (String, Option<usize>) {
        match Self::component_path_with(attrs, path) {
            Some((parent, component)) => (parent, Some(component)),
            None => (path.to_owned(), None),
        }
    }
    /// Run `edit` on each animation channel of `attr` that `channel` selects (None: all of
    /// them), with its index; nothing when `attr` is not animated. The ONE channel walk of the
    /// key commands: keying a component, Remove key, Static, Move keys, Interpolation and the
    /// curve effects.
    fn for_channels<E>(
        attrs: &mut Attrs,
        attr: &str,
        channel: Option<usize>,
        mut edit: impl FnMut(usize, &mut Channel) -> Result<(), E>,
    ) -> Result<(), E> {
        let Some(anim) = attrs.anim_mut(attr) else {
            return Ok(());
        };
        for (index, ch) in anim.channels.iter_mut().enumerate() {
            if channel.is_none_or(|c| c == index) {
                edit(index, ch)?;
            }
        }
        Ok(())
    }
    /// Drop keys of the channels `path` selects (`lane`): every key (`all`), else the one at
    /// `frame`. A channel left without keys holds the value it showed at `frame`; once none has
    /// keys the attribute is static and takes that value as its base (a discrete one leaves its
    /// value dictionary, `set_attribute`). The one unkeying of Remove key and Static.
    fn unkey(
        &mut self,
        id: NodeId,
        path: &str,
        frame: f64,
        all: bool,
        new: Tan,
    ) -> Result<(), String> {
        let held = self.attribute_value(id, path, frame)?;
        let mut attrs = self.attrs(id)?;
        let (attr, channel) = Self::lane(&attrs, path);
        let mut emptied = Vec::new();
        Self::for_channels(&mut attrs, &attr, channel, |index, ch| {
            if all {
                *ch = Channel::new();
            } else {
                ch.remove_key(frame);
            }
            if ch.is_empty() {
                emptied.push(index);
            }
            Ok::<(), String>(())
        })?;
        if attrs.anim(&attr).is_some_and(Animation::is_empty) {
            attrs.set_anim(&attr, None);
            self.store_attrs(id, &attrs)?;
            return self.set_attribute(id, path, held, frame, false, new);
        }
        if !emptied.is_empty() {
            // Some channels of a vector lost their keys: their base components hold the value.
            let raw = self.without_navigation_offset(id, path, held)?;
            let mut base = attrs
                .get(&attr)
                .cloned()
                .map(attr_json)
                .ok_or_else(|| format!("Unavailable parameter {attr}"))?;
            for index in emptied {
                let value = match channel {
                    Some(_) => raw.clone(),
                    None => raw.get(index).cloned().ok_or("Invalid component")?,
                };
                *base.get_mut(index).ok_or("Invalid component")? = value;
            }
            attrs.set(&attr, to_attr(&base));
        }
        self.store_attrs(id, &attrs)
    }
    /// Refresh the `/formula` choice blob after a Hybrid steps write. Slot edits
    /// land on the `/formula/Hybrid/steps` array entry, while `inactive_reason`
    /// gates sub-formula rows by the steps the blob carries: without the sync
    /// the rows would stay grey until the family itself is re-picked.
    fn sync_formula_steps_blob(attrs: &mut Attrs, steps: &Value) {
        let mut formula = match attrs.get("/formula").cloned().map(attr_json) {
            Some(v) => v,
            None => return,
        };
        let Some(slots) = formula
            .pointer_mut("/Hybrid/steps")
            .and_then(Value::as_array_mut)
        else {
            return;
        };
        let Some(fresh) = steps.as_array() else {
            return;
        };
        if slots.len() != fresh.len() {
            return;
        }
        slots.clone_from_slice(fresh);
        attrs.set("/formula", to_attr(&formula));
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
        if key && crate::render_profiles::static_branch(path) {
            return Err("Viewport policy and output settings cannot be animated".into());
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
        // A number takes its attribute's stored type (an integer attribute rounds), so an edit
        // never changes the type a value is stored, evaluated and edited as (`numeric_like`).
        let value = match (value.as_f64(), self.node(id)?["discrete"].get(path)) {
            (Some(v), None) => match self.attrs(id)?.get(path).cloned().map(attr_json) {
                Some(stored) if stored.is_number() => numeric_like(&stored, v),
                _ => value,
            },
            _ => value,
        };
        // Hard limits hold for every attribute edit (AE, timeline, commands), not per widget.
        let value = clamp_to_range(path, value);
        let value = self.without_navigation_offset(id, path, value)?;
        if let Some((parent, component)) = self.component_path(id, path)? {
            // Enum slots (Hybrid steps) only take their listed values; the grid
            // offers exactly these, this guards direct API writes. Numeric
            // components have no choices and skip the check.
            let slot_choices = attribute_choices(path)?;
            if !slot_choices.is_empty() && !slot_choices.contains(&value) {
                return Err(format!("Invalid enum value: {path}"));
            }
            let mut a = self.attrs(id)?;
            let mut vector = self.resolve_attribute(id, &parent, frame, &mut HashSet::new())?;
            // The vector comes from evaluation (a connection's source, a discrete value): its
            // shape is checked here, not assumed.
            *vector.get_mut(component).ok_or("Invalid component")? = value.clone();
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
                Self::for_channels(&mut a, &parent, Some(component), |_, ch| {
                    ch.upsert_key(Keyframe::with_tan(frame, scalar, key_tan(ch, frame, kind)));
                    Ok::<(), String>(())
                })?;
            } else {
                a.set(&parent, to_attr(&vector));
                if parent == "/formula/Hybrid/steps" {
                    Self::sync_formula_steps_blob(&mut a, &vector);
                }
            }
            return self.store_attrs(id, &a);
        }
        let mut a = self.attrs(id)?;
        match path {
            "/formula" => {
                serde_json::from_value::<crate::scene::Formula>(value.clone())
                    .map_err(|e| format!("Invalid formula: {e}"))?;
            }
            _ => {
                let choices = attribute_choices(path)?;
                if !choices.is_empty() && !choices.contains(&value) {
                    return Err(format!("Invalid enum value: {path}"));
                }
            }
        }
        let discrete = matches!(path, "/formula" | "/material/facing")
            || !numeric(&value)
            || self.node(id)?["discrete"].get(path).is_some();
        let animated = a.is_animated(path);
        if key || animated {
            let attr = if discrete {
                let base = a
                    .get(path)
                    .cloned()
                    .map(attr_json)
                    .unwrap_or_else(|| value.clone());
                let dict = self
                    .node_mut(id)?
                    .get_mut("discrete")
                    .and_then(Value::as_object_mut)
                    .ok_or("Invalid discrete dictionaries")?
                    .entry(path)
                    .or_insert_with(|| json!([base]))
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
            if path == "/formula/Hybrid/steps" {
                Self::sync_formula_steps_blob(&mut a, &value);
            }
        }
        if matches!(path, "/formula" | "/material/facing") && !key && !animated {
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
            if let Some(p) = node.parent
                && let Some(parent) = self.graph.nodes.get_mut(&p.to_string())
                && let Some(children) = parent["children"].as_array_mut()
            {
                children.push(json!(node.id));
            }
        }
    }
    fn evaluated_node(&self, id: NodeId, frame: f64) -> Result<Value, String> {
        self.evaluated_node_with_render(id, frame, None)
    }
    fn evaluated_node_with_render(
        &self,
        id: NodeId,
        frame: f64,
        render_override: Option<&Value>,
    ) -> Result<Value, String> {
        let data = self.node(id)?;
        let mut v = data["gpu"].clone();
        let attrs = self.attrs(id)?;
        let mut paths: Vec<_> = attrs.iter().map(|(path, _)| path).collect();
        paths.sort();
        for path in paths {
            if matches!(
                path.as_str(),
                "/visible" | "/locked" | "/solo" | "/shy" | "/start" | "/end"
            ) || path.starts_with("/transform/")
                || path.starts_with("/custom/")
            {
                continue;
            }
            // Evaluation includes persisted channels hidden by the inspector schema.
            // Parent structural switches sort before their variant/optional children.
            if v.pointer(path).is_some() {
                set_pointer(
                    &mut v,
                    path,
                    self.attribute_value_with(id, data, &attrs, path, frame)?,
                    false,
                )?;
            }
        }
        if self.info_with(id, data, &attrs)?.kind == WorldKind::Fractal {
            let iterations = v["render"]["iterations"].clone();
            v["render"] = match render_override {
                Some(render) => render.clone(),
                None => serde_json::to_value(
                    self.effective_render(self.output_render_profile()?, frame)?
                        .render,
                )
                .map_err(|e| e.to_string())?,
            };
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
        let data = self.node(id)?;
        let attrs = self.attrs(id)?;
        let parent = self
            .info_with(id, data, &attrs)?
            .parent
            .map(|parent| self.world_matrix(parent, frame, &mut HashSet::new()))
            .transpose()?
            .unwrap_or(Mat4::IDENTITY);
        if !parent.is_finite() || parent.determinant().abs() < 1e-10 {
            return Err("Camera parent transform is singular".into());
        }
        let inverse = parent.inverse();
        let vector = |path| -> Result<Vec3, String> {
            let value = self.attribute_value_with(id, data, &attrs, path, frame)?;
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
        // The equivalent nearest the authored rotation: both Euler solutions and whole turns.
        let degrees = curves::nearest_euler(
            [-x.to_degrees(), -y.to_degrees(), -z.to_degrees()],
            vector("/transform/rotation_degrees")?.to_array(),
            Some(1),
        );
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
        value: Value,
    ) -> Result<Value, String> {
        let node = self.node(id)?;
        if node["type"].as_str() != Some("Camera") {
            return Ok(value);
        }
        let attrs = self.attrs(id)?;
        Self::with_navigation_offset_with(node, &attrs, path, value)
    }
    fn with_navigation_offset_with(
        node: &Value,
        attrs: &Attrs,
        path: &str,
        mut value: Value,
    ) -> Result<Value, String> {
        if let Some(offset) = Self::navigation_offset_with(node, attrs, path) {
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
        let node = self.node(id)?;
        if node["type"].as_str() != Some("Camera") {
            return Ok(None);
        }
        let attrs = self.attrs(id)?;
        Ok(Self::navigation_offset_with(node, &attrs, path))
    }
    fn navigation_offset_with(node: &Value, attrs: &Attrs, path: &str) -> Option<Value> {
        if node["type"].as_str() != Some("Camera")
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
            return None;
        }
        attrs
            .get(&format!("/_navigation{path}"))
            .cloned()
            .map(attr_json)
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

    /// `auto_key`: Some(tangent) keys the change at `frame` with that new-key tangent;
    /// None writes it as a navigation offset (or the static base) without a key.
    fn write_navigation_value(
        &self,
        id: NodeId,
        attrs: &mut Attrs,
        path: &str,
        desired: Value,
        frame: f64,
        auto_key: Option<Tan>,
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
            if Self::connection(attrs, path).is_some() {
                return Err("Camera navigation channel is connected".into());
            }
            let animated = attrs
                .anim(path)
                .and_then(|a| a.channels.get(component))
                .is_some_and(|channel| !channel.is_empty());
            if let Some(kind) = auto_key {
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
            } else {
                // A stored offset or base of another shape than the pose is an error, not a panic.
                let (slot, value) = if animated {
                    offset_changed = true;
                    (&mut offset, wanted - raw)
                } else {
                    base_changed = true;
                    (&mut base, wanted - added)
                };
                if arity == 1 {
                    *slot = json!(value);
                } else {
                    *slot
                        .get_mut(component)
                        .ok_or("Camera channel does not match its pose")? = json!(value);
                }
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

    /// The node's frame for a viewport manipulator: its world matrix at the pivot (parent and
    /// `T R S`, where the handles sit and rotation turns around) and the parent's world matrix
    /// (identity at the root) that an edit is expressed under ([`local_pose`]).
    pub fn manip_frame(&self, id: NodeId, frame: f64) -> Result<(glam::Mat4, glam::Mat4), String> {
        let data = self.node(id)?;
        let attrs = self.attrs(id)?;
        let pivot = self
            .attribute_value_with(id, data, &attrs, "/transform/pivot", frame)?
            .as_array()
            .filter(|v| v.len() == 3)
            .map(|v| glam::Vec3::from(std::array::from_fn(|i| v[i].as_f64().unwrap_or(0.0) as f32)))
            .unwrap_or(glam::Vec3::ZERO);
        let world = self.world_matrix(id, frame, &mut HashSet::new())?;
        let parent = match self.info_with(id, data, &attrs)?.parent {
            Some(parent) => self.world_matrix(parent, frame, &mut HashSet::new())?,
            None => glam::Mat4::IDENTITY,
        };
        Ok((world * glam::Mat4::from_translation(pivot), parent))
    }

    /// `node` is `ancestor` or lies in its subtree.
    pub(crate) fn descends(&self, node: NodeId, ancestor: NodeId) -> bool {
        let mut current = Some(node);
        for _ in 0..=self.graph.nodes.len() {
            match current {
                Some(id) if id == ancestor => return true,
                Some(id) => {
                    current = self
                        .node(id)
                        .ok()
                        .and_then(|d| d["parent"].as_str().and_then(NodeId::parse))
                }
                None => return false,
            }
        }
        false
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
        let data = self.node(id)?;
        let attrs = self.attrs(id)?;
        let vec = |path: &str, default: [f32; 3]| -> Result<[f32; 3], String> {
            let v = self.attribute_value_with(id, data, &attrs, path, frame)?;
            Ok(v.as_array()
                .filter(|v| v.len() == 3)
                .map(|v| std::array::from_fn(|i| v[i].as_f64().unwrap_or(default[i] as f64) as f32))
                .unwrap_or(default))
        };
        // A transform take the node plays replaces its own position, rotation and scale.
        let (p, r, s) = match self.take_trs(id, frame)? {
            Some(t) => (t.position, t.rotation_degrees, t.scale),
            None => (
                vec("/transform/position", [0.0; 3])?,
                vec("/transform/rotation_degrees", [0.0; 3])?,
                vec("/transform/scale", [1.0; 3])?,
            ),
        };
        let r = r.map(|v| v.to_radians());
        let pivot = vec("/transform/pivot", [0.0; 3])?;
        let local = glam::Mat4::from_cols_array_2d(
            &playa_engine::entities::transform::build_model_matrix(p, r, s, pivot)
                .to_cols_array_2d(),
        );
        if let Some(parent) = self.info_with(id, data, &attrs)?.parent {
            Ok(self.world_matrix(parent, frame, seen)? * local)
        } else {
            Ok(local)
        }
    }
    /// Camera node `id`'s visible pose at `frame` and the framing radius its distance is in:
    /// its attributes, orbit, transform (parents included) and the take it plays. What the
    /// active camera renders (`snapshot`) and what the viewport draws of the others (`overlay`).
    pub fn camera_pose(
        &self,
        id: NodeId,
        frame: f64,
    ) -> Result<(crate::scene::Camera, f32), String> {
        let v = self.evaluated_node(id, frame)?;
        let mut camera: crate::scene::Camera =
            serde_json::from_value(v["camera"].clone()).map_err(|e| e.to_string())?;
        let reference =
            self.attribute_value(id, CAMERA_FRAMING, frame)?
                .as_f64()
                .filter(|r| r.is_finite() && *r > 0.0)
                .ok_or("The camera's framing radius must be a positive number")? as f32;
        camera.yaw_degrees += self.camera_orbit_angle(id, frame)?;
        let m = self.world_matrix(id, frame, &mut HashSet::new())?;
        if m != glam::Mat4::IDENTITY {
            let orientation = camera.orientation();
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
            camera.yaw_degrees = yaw.to_degrees();
            camera.pitch_degrees = -pitch.to_degrees();
            camera.roll_degrees = roll.to_degrees();
            camera.distance *= scale;
            camera.target = m
                .transform_point3(glam::Vec3::from(camera.target))
                .to_array();
        }
        // A camera take the camera plays is the visible pose itself: it replaces the evaluated
        // camera.
        if let Some((take, shown)) = self.take_frame(id, frame)? {
            camera = take
                .camera_at(shown, self.first, self.last, self.fps)
                .ok_or_else(|| format!("Take \"{}\" is not a camera take", take.name))?;
        }
        Ok((camera, reference))
    }

    /// Sun node `id`'s light at `frame`, its direction turned by its transform (parents
    /// included) into world azimuth / elevation.
    pub fn sun_light(&self, id: NodeId, frame: f64) -> Result<crate::scene::Sun, String> {
        let v = self.evaluated_node(id, frame)?;
        let mut sun: crate::scene::Sun =
            serde_json::from_value(v["sun"].clone()).map_err(|e| e.to_string())?;
        let d = self
            .world_matrix(id, frame, &mut HashSet::new())?
            .transform_vector3(sun.direction())
            .normalize_or_zero();
        sun.elevation = d.y.clamp(-1.0, 1.0).asin().to_degrees();
        sun.azimuth = d.x.atan2(d.z).to_degrees();
        Ok(sun)
    }

    /// Node `id`'s object-to-world matrix at `frame` (parents included).
    pub fn node_world(&self, id: NodeId, frame: f64) -> Result<glam::Mat4, String> {
        self.world_matrix(id, frame, &mut HashSet::new())
    }

    /// A mask node's settings at `frame`.
    pub fn mask(&self, id: NodeId, frame: f64) -> Result<crate::scene::Mask, String> {
        serde_json::from_value(self.evaluated_node(id, frame)?["mask"].clone())
            .map_err(|e| e.to_string())
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
        let data = self.node(id)?;
        let attrs = self.attrs(id)?;
        let visible = self
            .attribute_value_with(id, data, &attrs, "/visible", frame)?
            .as_bool()
            .unwrap_or(true);
        let start = self
            .attribute_value_with(id, data, &attrs, "/start", frame)?
            .as_f64()
            .unwrap_or(0.0);
        let end = self
            .attribute_value_with(id, data, &attrs, "/end", frame)?
            .as_f64()
            .unwrap_or(self.last as f64 + 1.0);
        if !visible || frame < start || frame >= end {
            return Ok(false);
        }
        if let Some(parent) = self.info_with(id, data, &attrs)?.parent {
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
    /// A typed scene of the world settings `base` (a complete scene's JSON, the render
    /// included) with a node's evaluated `branches` laid over it: what a world object packs
    /// (`snapshot_with_render_profile`). Nodes store only their own branches (`node_template`).
    fn overlay_scene(base: &Value, branches: Value) -> Result<Scene, String> {
        let Value::Object(branches) = branches else {
            return Err("A node's evaluated branches must be an object".into());
        };
        let mut value = base.clone();
        value
            .as_object_mut()
            .ok_or("World settings must be an object")?
            .extend(branches);
        serde_json::from_value(value).map_err(|e| e.to_string())
    }
    #[cfg(test)]
    pub fn node_scene(&self, id: NodeId, frame: f64) -> Result<Scene, String> {
        let global = self.snapshot(frame)?;
        let mut base = self
            .graph
            .bus_slots
            .get("world")
            .cloned()
            .ok_or("World settings missing")?;
        base["render"] = serde_json::to_value(global.render).map_err(|e| e.to_string())?;
        let mut scene = Self::overlay_scene(&base, self.evaluated_node(id, frame)?)?;
        scene.camera = global.camera;
        scene.camera_reference = global.camera_reference;
        scene.environment = global.environment;
        scene.sun = global.sun;
        scene.sky = global.sky;
        let data = self.node(id)?;
        let attrs = self.attrs(id)?;
        let kind = self.info_with(id, data, &attrs)?.kind;
        if kind != WorldKind::Fractal {
            scene.render = global.render;
        }
        scene.colour = global.colour;
        scene.document = None;
        if kind == WorldKind::Fractal {
            scene.object_world = Some(
                self.world_matrix(id, frame, &mut HashSet::new())?
                    .to_cols_array_2d(),
            );
            scene.object.offset = serde_json::from_value(self.attribute_value_with(
                id,
                data,
                &attrs,
                "/transform/position",
                frame,
            )?)
            .map_err(|e| e.to_string())?;
            scene.object.rotation_degrees = serde_json::from_value::<[f32; 3]>(
                self.attribute_value_with(id, data, &attrs, "/transform/rotation_degrees", frame)?,
            )
            .map_err(|e| e.to_string())?
            .map(|v| -v);
            let scale = self.attribute_value_with(id, data, &attrs, "/transform/scale", frame)?;
            scene.object.scale = scale[0].as_f64().unwrap_or(1.0) as f32;
            if let Some(mat) = data["material"].as_str().and_then(NodeId::parse) {
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
        let data = self.node(id)?;
        let attrs = self.attrs(id)?;
        if let Some((source, source_path)) = Self::connection(&attrs, path) {
            return self.integrated_scalar(source, &source_path, from, to, visiting);
        }
        let (attr, channel) = Self::lane(&attrs, path);
        if let Some(channel) = attrs
            .anim(&attr)
            .and_then(|a| a.channels.get(channel.unwrap_or(0)))
            .filter(|channel| !channel.is_empty())
        {
            return Ok(crate::camera_orbit::integrate(channel, from, to));
        }
        let value = self
            .attribute_value_with(id, data, &attrs, path, from)?
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
        self.snapshot_with_render_profile(self.output_render_profile()?, frame, View::Output)
    }

    /// Whether node `id` (of `kind`) takes part in a picture at `frame`: visible over its span
    /// with its parents, and of a kind `shown` includes. The one test of renders, the viewport,
    /// its extras and its manipulator (Blender's `BKE_object_is_visible_in_viewport`).
    pub fn shows(
        &self,
        id: NodeId,
        kind: WorldKind,
        frame: f64,
        shown: crate::render_profiles::Kinds,
    ) -> Result<bool, String> {
        Ok(shown.contains(kind) && self.visible_at(id, frame, &mut HashSet::new())?)
    }

    /// The node kinds a picture of `view` at `frame` includes: all of them in an output, the
    /// ones the viewport shows in the viewport (`ViewportPolicy::shown`).
    pub fn shown_kinds(
        &self,
        view: View,
        frame: f64,
    ) -> Result<crate::render_profiles::Kinds, String> {
        Ok(match view {
            View::Output => crate::render_profiles::Kinds::ALL,
            View::Viewport => self.viewport_policy(frame)?.shown,
        })
    }

    /// The scene `view` renders at `frame` through render profile `profile`. A kind the viewport
    /// hides leaves a viewport picture (fractals, lights, the environment); a mask's cut is its
    /// fractal's shape and stays (as a Boolean modifier does when its cutter is hidden).
    pub fn snapshot_with_render_profile(
        &self,
        profile: NodeId,
        frame: f64,
        view: View,
    ) -> Result<Scene, String> {
        self.require_live_settings(profile, WorldKind::RenderSettings)?;
        self.check_format()?;
        if !frame.is_finite() {
            return Err("Invalid frame".into());
        }
        self.validate_render_profiles(frame)?;
        let effective = self.effective_render(profile, frame)?;
        let mut base = self
            .graph
            .bus_slots
            .get("world")
            .ok_or("World settings missing")?
            .clone();
        let render_value = serde_json::to_value(effective.render).map_err(|e| e.to_string())?;
        base["render"] = render_value.clone();
        let mut scene: Scene = serde_json::from_value(base.clone())
            .map_err(|e| format!("Invalid world settings: {e}"))?;
        scene.document = None;
        scene.animation = Default::default();
        scene.objects.clear();
        scene.lights.clear();
        scene.world_render = true;
        scene.object_world = None;
        let nodes = self.nodes_at(frame);
        let shown = self.shown_kinds(view, frame)?;
        if let Some(id) = self.active_camera {
            (scene.camera, scene.camera_reference) = self
                .camera_pose(id, frame)
                .map(|(camera, reference)| (camera, Some(reference)))?;
        }
        let environment = match self.active_environment {
            Some(id) if self.shows(id, WorldKind::Environment, frame, shown)? => Some(id),
            _ => None,
        };
        // The sky is the shown Environment's; without one the world is dark behind and around.
        if let Some(id) = environment {
            let v = self.evaluated_node_with_render(id, frame, Some(&render_value))?;
            scene.environment =
                serde_json::from_value(v["environment"].clone()).map_err(|e| e.to_string())?;
            scene.environment.revision =
                self.node(id)?["environment_revision"].as_u64().unwrap_or(0);
            scene.sky = serde_json::from_value(v["sky"].clone()).map_err(|e| e.to_string())?;
        } else {
            scene.environment.enabled = false;
            scene.sky.intensity = 0.0;
            scene.sky.background = false;
        }
        let solo = nodes.iter().any(|n| n.solo);
        for node in &nodes {
            if node.kind != WorldKind::DirectionalLight
                || (solo && !node.solo)
                || !self.shows(node.id, node.kind, frame, shown)?
            {
                continue;
            }
            scene.lights.push(self.sun_light(node.id, frame)?);
        }
        // The single-object key light is the first sun; without one it is off.
        match scene.lights.first() {
            Some(sun) => scene.sun = *sun,
            None => scene.sun.intensity = 0.0,
        }
        // Visible masks with their scope: the subtree of their parent, the whole scene at the root.
        let mut masks = Vec::new();
        for node in nodes.iter().filter(|n| n.kind == WorldKind::Mask) {
            if !self.visible_at(node.id, frame, &mut HashSet::new())? {
                continue;
            }
            let mask = self.mask(node.id, frame)?;
            let world = self.node_world(node.id, frame)?.to_cols_array_2d();
            masks.push((node.parent, crate::scene::MaskInstance { mask, world }));
        }
        let mut evaluated_materials: HashMap<NodeId, crate::scene::Material> = HashMap::new();
        for node in nodes {
            if node.kind != WorldKind::Fractal
                || (solo && !node.solo)
                || !self.shows(node.id, node.kind, frame, shown)?
            {
                continue;
            }
            let mut in_scope = Vec::new();
            for (scope, mask) in &masks {
                if scope.is_none_or(|s| self.descends(node.id, s)) {
                    in_scope.push(*mask);
                }
            }
            // The fractal's own branches over the world settings: a fractal without a material
            // renders the world's.
            let mut object = Self::overlay_scene(
                &base,
                self.evaluated_node_with_render(node.id, frame, Some(&render_value))?,
            )?;
            if let Some(material) = self.node(node.id)?["material"]
                .as_str()
                .and_then(NodeId::parse)
            {
                let mat = if let Some(cached) = evaluated_materials.get(&material) {
                    cached.clone()
                } else {
                    let parsed: crate::scene::Material = serde_json::from_value(
                        self.evaluated_node(material, frame)?["material"].clone(),
                    )
                    .map_err(|e| e.to_string())?;
                    evaluated_materials.insert(material, parsed.clone());
                    parsed
                };
                object.material = mat;
            }
            object.camera = scene.camera;
            object.camera_reference = scene.camera_reference;
            object.environment = scene.environment.clone();
            object.sun = scene.sun;
            object.sky = scene.sky;
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
            object.node = Some(node.id);
            object.masks = in_scope;
            scene.objects.push(object);
        }
        if let Some(first) = scene.objects.first() {
            scene.formula = first.formula;
            scene.julia = first.julia;
            scene.julia_mix = first.julia_mix;
            scene.orbit_offset = first.orbit_offset;
            scene.inversion_strength = first.inversion_strength;
            scene.inversion_center = first.inversion_center;
            scene.inversion_radius = first.inversion_radius;
            scene.object = first.object;
            scene.object_world = first.object_world;
            scene.material = first.material.clone();
            scene.palette = first.palette;
            scene.coloring = first.coloring;
            scene.trap_point = first.trap_point;
            scene.trap_axis = first.trap_axis;
            scene.trap_scale = first.trap_scale;
        }
        effective.apply_to(&mut scene);
        effective.show(&mut scene)?;
        scene.camera.validate_lens()?;
        Ok(scene)
    }

    /// This document's graph is of the format this build reads and writes.
    fn check_format(&self) -> Result<(), String> {
        if self.graph.format_version != playa_graph::SUBNET_FORMAT_VERSION {
            return Err(format!(
                "Unsupported world graph version {} (expected {})",
                self.graph.format_version,
                playa_graph::SUBNET_FORMAT_VERSION
            ));
        }
        Ok(())
    }

    /// The invariants every reader of a document relies on: a work area `first <= last` at a
    /// positive finite rate, world settings that are an object, and nodes of the stored shape
    /// whose animations have their values' arity (`node_shape`, `check_arity`). Checked when a
    /// document is decoded, so a malformed or hand-edited file is an error instead of a panic.
    fn validate(&self) -> Result<(), String> {
        if self.first > self.last {
            return Err(format!(
                "The work area starts at frame {} after it ends at {}",
                self.first, self.last
            ));
        }
        if !(self.fps.is_finite() && self.fps > 0.0) {
            return Err(format!("Invalid frame rate {}", self.fps));
        }
        if !self
            .graph
            .bus_slots
            .get("world")
            .is_some_and(Value::is_object)
        {
            return Err("World settings missing".into());
        }
        for (key, data) in &self.graph.nodes {
            node_shape(key, data)?;
            let attrs: Attrs = serde_json::from_value(data["host"].clone())
                .map_err(|e| format!("Node {key}: {e}"))?;
            check_arity(&attrs).map_err(|e| format!("Node {key}: {e}"))?;
        }
        Ok(())
    }

    /// Decode a persisted `WorldDocument` from JSON text, strictly: an unknown or missing field
    /// is an error (`deny_unknown_fields`, `node_shape`), the graph is of this build's format
    /// (`check_format`) and the document passes `validate`. Nothing is upgraded or filled in, so
    /// a decoded document re-encodes to the same JSON value (not necessarily the same bytes):
    /// older layouts were converted once (plan1 Q1) and are not read. The one reader of saved
    /// documents (`io_service::decode_scene`).
    pub fn decode_json(text: &str) -> Result<Self, String> {
        let document: Self = serde_json::from_str(text).map_err(|e| e.to_string())?;
        document.check_format()?;
        document.validate()?;
        Ok(document)
    }
}

fn attribute_base_path(path: &str) -> &str {
    match path.rsplit_once('/') {
        Some((parent, last)) if last.parse::<usize>().is_ok() => parent,
        _ => path,
    }
}

fn attribute_order(path: &str) -> u16 {
    match path {
        // Quality: sampling & adaptive first, then ray marching
        "/quality/samples" => 10,
        "/quality/resolution_scale" => 11,
        "/quality/max_bounces" => 12,
        "/quality/adaptive/enabled" => 20,
        "/quality/adaptive/min_samples" => 21,
        "/quality/adaptive/noise_threshold" => 22,
        "/quality/max_steps" => 30,
        "/quality/step_factor" => 31,
        "/quality/hit_epsilon" => 32,
        "/quality/glass_probes" => 33,

        // Render: display/method/backend & denoise first, then thinning, then direct lighting
        "/render/display" => 10,
        "/render/method" => 11,
        "/render/backend" => 12,
        "/render/denoise/enabled" => 20,
        "/render/denoise/quality" => 21,
        "/render/denoise/mode" => 22,
        "/render/denoise/interval" => 23,
        "/render/exposure_stops" => 30,
        "/render/saturation" => 31,
        "/render/reinhard" => 32,
        "/render/thinning" => 40,
        "/render/direct/shadow_strength" => 50,
        "/render/direct/shadow_steps" => 51,
        "/render/direct/ao_strength" => 52,
        "/render/direct/ao_steps" => 53,
        "/render/direct/ao_radius" => 54,

        // Viewport: mode & profile references first, then pause/freeze, pacing, overlays
        "/viewport/mode" => 10,
        "/viewport/moving_id" => 11,
        "/viewport/still_id" => 12,
        "/viewport/manual_id" => 13,
        "/viewport/paused" => 20,
        "/viewport/frozen" => 21,
        "/viewport/target_fps" => 30,
        "/viewport/settle_delay_ms" => 31,
        "/viewport/batch_budget_ms" => 32,
        "/viewport/extras" => 40,
        p if p.starts_with("/viewport/show/") => 50,
        p if p.starts_with("/viewport/select/") => 60,

        // Output: the recipe and final denoise first, then color management
        "/output/preset" => 10,
        "/output/denoise_at_completion" => 13,
        "/output/output_display" => 40,
        "/output/output_view" => 41,

        // Fractal: formula & iterations first, then formula parameters, Julia, inversion, coloring
        "/formula" => 10,
        "/render/iterations" => 11,
        "/formula/Hybrid/steps" => 12,
        p if p.starts_with("/formula/") => crate::formulas::formula_param_order(p).unwrap_or(20),
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

        // Material: assignment/preset/model & base/roughness/metalness first, then transmission, emission, coat, sheen, thin film, facing
        "/material_id" => 5,
        "/material/preset" => 6,
        "/material/model" => 10,
        "/material/color_source" => 11,
        "/material/base_color" => 12,
        "/material/base" => 13,
        "/material/base_tint" => 14,
        "/material/metalness" => 15,
        "/material/specular_roughness" => 16,
        "/material/specular_ior" => 17,
        "/material/specular" => 18,
        "/material/specular_color" => 19,
        "/material/diffuse_roughness" => 20,
        "/material/transmission" => 30,
        "/material/transmission_color" => 31,
        "/material/transmission_depth" => 32,
        "/material/transmission_extra_roughness" => 33,
        "/material/emission" => 40,
        "/material/emission_color" => 41,
        "/material/coat" => 50,
        "/material/coat_color" => 51,
        "/material/coat_roughness" => 52,
        "/material/coat_ior" => 53,
        "/material/coat_affect_color" => 54,
        "/material/coat_affect_roughness" => 55,
        "/material/specular_anisotropy" => 60,
        "/material/specular_rotation" => 61,
        "/material/sheen" => 70,
        "/material/sheen_color" => 71,
        "/material/sheen_roughness" => 72,
        "/material/thin_film_thickness" => 80,
        "/material/thin_film_ior" => 81,
        "/material/facing" => 90,
        "/material/facing/color" => 91,
        "/material/facing/roughness" => 92,
        "/material/facing/metallic" => 93,
        "/material/facing/exponent" => 94,

        // Camera: framing/pose first, then depth of field, then orbit animation
        "/camera/fov_y_degrees" => 10,
        "/camera/distance" => 11,
        "/camera/target" => 12,
        "/camera/yaw_degrees" => 13,
        "/camera/pitch_degrees" => 14,
        "/camera/roll_degrees" => 15,
        "/camera/free_flight" => 16,
        "/camera/f_number" => 20,
        "/camera/focus_distance" => 21,
        "/camera/sensor_height" => 22,
        "/camera/orbit_speed_degrees" => 30,
        "/camera/orbit_phase_degrees" => 31,
        CAMERA_FRAMING => 40,

        // Environment: HDRI first, then the sky; Light: the sun
        "/environment/enabled" => 10,
        "/environment/path" => 11,
        "/environment/intensity" => 12,
        "/environment/rotation_degrees" => 13,
        "/sun/intensity" => 20,
        "/sun/color" => 21,
        "/sun/azimuth" => 22,
        "/sun/elevation" => 23,
        "/sun/angle" => 24,
        "/sky/intensity" => 30,
        "/sky/horizon" => 31,
        "/sky/zenith" => 32,
        "/sky/background" => 33,
        "/sky/sun_disc" => 34,

        // Mask & Array
        "/mask/shape" => 10,
        "/mask/size" => 11,
        "/mask/invert" => 12,
        "/mask/density" => 13,
        "/mask/seed" => 14,
        "/mask/array/kind" => 20,
        "/mask/array/count" => 21,
        "/mask/array/offset" => 22,
        "/mask/array/radius" => 23,
        "/mask/array/axis" => 24,
        "/mask/array/sweep_degrees" => 25,
        "/mask/array/align" => 26,

        // Transform: TRS + Pivot
        "/transform/position" => 10,
        "/transform/rotation_degrees" => 11,
        "/transform/scale" => 13,
        "/transform/pivot" => 14,

        // Custom / Layer metadata: Take clip & warp, then layer flags, then span, then custom attributes
        TAKE_CLIP => 10,
        TAKE_OFFSET => 11,
        TAKE_SPEED => 12,
        "/visible" => 20,
        "/solo" => 21,
        "/locked" => 22,
        "/shy" => 23,
        "/start" => 30,
        "/end" => 31,
        p if p.starts_with("/custom/") => 40,

        _ => 500,
    }
}
