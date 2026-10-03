//! Object panels. Widgets report intents; WorldEditor owns all document changes.
use crate::ui_style::AttributeMetrics;
use crate::world::{NodeId, WorldAttribute, WorldCommand, WorldEditor, WorldKind, WorldNodeInfo};
use curves::CurveKind;
use egui::{Color32, Pos2, Rect, Sense, Vec2};
use egui_attr_grid::{AttrField, AttrGridHooks, AttrValue as GridValue, render_grid_with_config};
use egui_outliner::{ContextItem, OutlinerAction, OutlinerConfig, OutlinerModel, TreeNode};
use egui_phosphor::regular as ph;
use egui_track_timeline::{
    Clip, KeyPos, Keyframe, PropLane, TimelineAction, TimelineConfig, TimelineModel, TimelineView,
    Track, TrackTimeline, WorkArea,
};
use egui_widgets_config::attr_layout::{
    ValueEditorLayout, cell_ui, square_icon, square_icon_with_paint,
};
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

pub(crate) fn parameter_gesture(ctx: &egui::Context) -> Option<u64> {
    ctx.dragged_id()
        .or_else(|| {
            if ctx.text_edit_focused() {
                ctx.memory(|memory| memory.focused())
            } else {
                None
            }
        })
        .map(|id| id.value())
}
fn value_gesture_eligible(attr: &WorldAttribute, value: &Value) -> bool {
    value.is_number()
        || value
            .as_array()
            .is_some_and(|values| !values.is_empty() && values.iter().all(Value::is_number))
        || (value.is_string() && attr.choices.is_empty() && attr.path != "/material_id")
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct KeyIdentity {
    node: NodeId,
    path: String,
    frame: u64,
}
#[derive(Clone)]
struct Lane {
    label: String,
    path: Option<String>,
    value: Value,
    frames: Vec<f64>,
    depth: usize,
    group: bool,
    attr: Option<WorldAttribute>,
}
#[derive(Clone)]
struct LayerDrag {
    id: NodeId,
    order: Option<std::sync::Arc<Vec<NodeId>>>,
}
fn layer_drop_order(
    nodes: &[WorldNodeInfo],
    frozen: &[NodeId],
    moved: NodeId,
    target: NodeId,
    after: bool,
) -> Vec<NodeId> {
    let current = || nodes.iter().map(|n| n.id).collect::<Vec<_>>();
    if moved == target
        || !nodes.iter().any(|n| n.id == moved)
        || !nodes.iter().any(|n| n.id == target)
    {
        return current();
    }
    let mut ids: Vec<_> = frozen
        .iter()
        .copied()
        .filter(|id| nodes.iter().any(|n| n.id == *id))
        .collect();
    // An external edit may add a node during a gesture. Keep it in the document;
    // the gesture's frozen UUID order still determines its original targets.
    for node in nodes {
        if !ids.contains(&node.id) {
            ids.push(node.id);
        }
    }
    ids.retain(|id| *id != moved);
    let at = ids.iter().position(|id| *id == target).unwrap_or(ids.len()) + usize::from(after);
    ids.insert(at.min(ids.len()), moved);
    ids
}

#[derive(Default)]
struct AttributeCache {
    document: Option<NodeId>,
    revision: u64,
    frame: u64,
    nodes: Vec<WorldNodeInfo>,
    attributes: HashMap<NodeId, Vec<WorldAttribute>>,
    sections: HashMap<(NodeId, &'static str), GridSectionCache>,
}
#[derive(Default)]
struct OutlinerCache {
    document: Option<NodeId>,
    revision: u64,
    nodes: Vec<WorldNodeInfo>,
    map: HashMap<u64, NodeId>,
    model: OutlinerModel,
    config: OutlinerConfig,
}
#[derive(Default)]
struct TimelineCache {
    document: Option<NodeId>,
    revision: u64,
    projection: u64,
    frame: u64,
    lanes: Vec<Vec<Lane>>,
    model: TimelineModel,
}
#[derive(Default)]
struct GridSectionCache {
    state: egui_attr_grid::AttrGridState,
    fields: Vec<egui_attr_grid::AttrField>,
    values: Vec<Value>,
    labels: Vec<String>,
    indices: Vec<usize>,
    revision: u64,
    frame: u64,
    projection: u64,
}

pub struct WorldUi {
    pub playhead: u32,
    pub playing: bool,
    pub looping: bool,
    pub attribute_label_width: f32,
    pub attribute_metrics: AttributeMetrics,
    pub material_library_requested: bool,
    fraction: f64,
    view: TimelineView,
    expanded: HashSet<NodeId>,
    groups: HashSet<(NodeId, String)>,
    component_open: HashMap<NodeId, HashSet<String>>,
    component_closed: HashMap<NodeId, HashSet<String>>,
    section_closed: HashMap<NodeId, HashSet<&'static str>>,
    keys: HashSet<KeyIdentity>,
    snap: bool,
    error: Option<String>,
    metadata: String,
    metadata_node: Option<NodeId>,
    metadata_document: Option<NodeId>,
    metadata_revision: u64,
    metadata_dirty: bool,
    custom_name: String,
    custom_value: f64,
    picker: Option<(NodeId, egui_file_dialog::FileDialog)>,
    environment_file: egui_file_field::FileFieldState,
    cache: AttributeCache,
    drag_order: Option<std::sync::Arc<Vec<NodeId>>>,
    timeline_cache: TimelineCache,
    outliner_cache: OutlinerCache,
}
impl Default for WorldUi {
    fn default() -> Self {
        Self {
            playhead: 0,
            playing: false,
            looping: true,
            attribute_label_width: 180.0,
            attribute_metrics: AttributeMetrics::default(),
            material_library_requested: false,
            fraction: 0.0,
            view: TimelineView::default(),
            expanded: HashSet::new(),
            groups: HashSet::new(),
            component_open: HashMap::new(),
            component_closed: HashMap::new(),
            section_closed: HashMap::new(),
            keys: HashSet::new(),
            snap: true,
            error: None,
            metadata: String::new(),
            metadata_node: None,
            metadata_document: None,
            metadata_revision: 0,
            metadata_dirty: false,
            custom_name: String::new(),
            custom_value: 0.0,
            picker: None,
            environment_file: egui_file_field::FileFieldState::default(),
            cache: AttributeCache::default(),
            drag_order: None,
            timeline_cache: TimelineCache::default(),
            outliner_cache: OutlinerCache::default(),
        }
    }
}
fn wid(id: NodeId) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    id.hash(&mut h);
    h.finish()
}
fn kind_label(k: WorldKind) -> &'static str {
    match k {
        WorldKind::Fractal => "Fractal",
        WorldKind::Camera => "Camera",
        WorldKind::DirectionalLight => "Light",
        WorldKind::Environment => "Environment",
        WorldKind::Group => "Group",
        WorldKind::Material => "Material",
    }
}
fn kind_icon(k: WorldKind) -> &'static str {
    match k {
        WorldKind::Fractal => ph::CUBE,
        WorldKind::Camera => ph::CAMERA,
        WorldKind::DirectionalLight => ph::SUN,
        WorldKind::Environment => ph::GLOBE,
        WorldKind::Group => ph::FOLDER,
        WorldKind::Material => ph::PALETTE,
    }
}
fn kind_color(k: WorldKind) -> Color32 {
    match k {
        WorldKind::Fractal => Color32::from_rgb(105, 153, 197),
        WorldKind::Camera => Color32::from_rgb(164, 126, 192),
        WorldKind::DirectionalLight => Color32::from_rgb(204, 181, 99),
        WorldKind::Environment => Color32::from_rgb(95, 169, 150),
        WorldKind::Group => Color32::from_rgb(142, 149, 163),
        WorldKind::Material => Color32::from_rgb(190, 132, 155),
    }
}
fn category(path: &str) -> &'static str {
    match path.split('/').nth(1).unwrap_or("") {
        "transform" => "Transform",
        "camera" => "Camera",
        "material" | "material_id" => "Material",
        "fractal" | "formula" | "julia" | "palette" | "coloring" | "trap_point" | "trap_axis"
        | "trap_scale" => "Fractal",
        "light" | "lighting" => "Light",
        "environment" => "Environment",
        "render" => "Render",
        "colour" | "color" => "Color",
        _ => "Custom",
    }
}
fn reparent_order(
    nodes: &[WorldNodeInfo],
    id: NodeId,
    parent: Option<NodeId>,
    index: usize,
) -> Vec<NodeId> {
    let rest = nodes.iter().filter(|n| n.id != id).collect::<Vec<_>>();
    let siblings = rest
        .iter()
        .filter(|n| n.parent == parent)
        .collect::<Vec<_>>();
    let index = index.min(siblings.len());
    let at = if let Some(next) = siblings.get(index) {
        rest.iter().position(|n| n.id == next.id).unwrap()
    } else if let Some(last) = siblings.last() {
        rest.iter().position(|n| n.id == last.id).unwrap() + 1
    } else {
        parent
            .and_then(|parent| rest.iter().position(|n| n.id == parent).map(|i| i + 1))
            .unwrap_or(rest.len())
    };
    let mut ids = rest.into_iter().map(|n| n.id).collect::<Vec<_>>();
    ids.insert(at, id);
    ids
}
fn tree(nodes: &[WorldNodeInfo], parent: Option<NodeId>, depth: usize) -> Vec<TreeNode> {
    if depth > 64 {
        return vec![];
    }
    nodes
        .iter()
        .filter(|n| n.parent == parent)
        .map(|n| {
            TreeNode::new(wid(n.id), n.name.clone())
                .with_icon(kind_icon(n.kind))
                .with_type(kind_label(n.kind))
                .with_visible(n.visible)
                .with_children(tree(nodes, Some(n.id), depth + 1))
        })
        .collect()
}
impl WorldUi {
    pub fn seek(&mut self, frame: u32) {
        self.playhead = frame;
        self.fraction = 0.0;
    }
    pub fn reset(&mut self) {
        let attribute_metrics = self.attribute_metrics;
        let attribute_label_width = self.attribute_label_width;
        *self = Self::default();
        self.attribute_metrics = attribute_metrics;
        self.attribute_label_width = attribute_label_width;
    }
    pub fn advance(&mut self, dt: f32, editor: &WorldEditor) -> bool {
        if !self.playing {
            return false;
        }
        let first = editor.document.first;
        let last = editor.document.last.max(first);
        self.fraction += f64::from(dt.max(0.0)) * editor.document.fps.max(1.0);
        let step = self.fraction.floor() as u32;
        self.fraction -= f64::from(step);
        if step == 0 {
            return false;
        }
        let old = self.playhead;
        let next = self.playhead.saturating_add(step);
        if next > last {
            if self.looping {
                self.playhead = first + (next - first) % (last - first + 1);
            } else {
                self.playhead = last;
                self.playing = false;
            }
        } else {
            self.playhead = next.max(first);
        }
        old != self.playhead
    }
    fn command(&mut self, e: &mut WorldEditor, c: WorldCommand) {
        if let Err(err) = e.execute(c) {
            self.error = Some(err);
        }
    }
    fn value_command(&mut self, ui: &egui::Ui, e: &mut WorldEditor, c: WorldCommand) {
        let gesture = parameter_gesture(ui.ctx()).or_else(|| {
            if ui.input(|input| input.pointer.primary_released()) {
                e.active_edit()
            } else {
                None
            }
        });
        if let Err(err) = e.execute_edit(c, gesture) {
            self.error = Some(err);
        }
    }
    fn status(&mut self, ui: &mut egui::Ui) {
        if let Some(err) = self.error.clone() {
            ui.horizontal(|ui| {
                ui.colored_label(Color32::from_rgb(226, 135, 126), err);
                if ui.small_button("Dismiss").clicked() {
                    self.error = None;
                }
            });
        }
    }
    fn select(
        &mut self,
        e: &mut WorldEditor,
        id: NodeId,
        add: bool,
        range: bool,
        nodes: &[WorldNodeInfo],
    ) {
        if range {
            if let Some(anchor) = e.selection {
                if let (Some(a), Some(b)) = (
                    nodes.iter().position(|n| n.id == anchor),
                    nodes.iter().position(|n| n.id == id),
                ) {
                    e.selected = nodes[a.min(b)..=a.max(b)].iter().map(|n| n.id).collect();
                }
            }
        } else if add {
            if let Some(i) = e.selected.iter().position(|n| *n == id) {
                e.selected.remove(i);
            } else {
                e.selected.push(id);
            }
        } else {
            e.selected = vec![id];
        }
        e.selection = if e.selected.contains(&id) {
            Some(id)
        } else {
            e.selected.last().copied()
        };
    }
    pub fn outliner(&mut self, ui: &mut egui::Ui, e: &mut WorldEditor) {
        ui.horizontal(|ui| {
            ui.menu_button(format!("{} Add", ph::PLUS), |ui| {
                for kind in [
                    WorldKind::Fractal,
                    WorldKind::Camera,
                    WorldKind::DirectionalLight,
                    WorldKind::Environment,
                    WorldKind::Group,
                    WorldKind::Material,
                ] {
                    if ui
                        .button(format!("{} {}", kind_icon(kind), kind_label(kind)))
                        .clicked()
                    {
                        self.command(
                            e,
                            WorldCommand::Create {
                                kind,
                                name: kind_label(kind).into(),
                                parent: None,
                            },
                        );
                        ui.close();
                    }
                }
            });
            if ui
                .add_enabled(e.selection.is_some(), egui::Button::new(ph::COPY))
                .on_hover_text("Duplicate selected object")
                .clicked()
            {
                if let Some(id) = e.selection {
                    self.command(e, WorldCommand::Duplicate(id));
                }
            }
            if ui
                .add_enabled(e.selection.is_some(), egui::Button::new(ph::TRASH))
                .on_hover_text("Delete selected objects")
                .clicked()
            {
                let ids = e.selected.clone();
                self.command(
                    e,
                    WorldCommand::Batch(ids.into_iter().map(WorldCommand::Delete).collect()),
                );
            }
        });
        self.status(ui);
        let cache = self.take_outliner_cache(e);
        let nodes = &cache.nodes;
        let map = &cache.map;
        let model = &cache.model;
        let cfg = &cache.config;
        let actions = ui
            .push_id("world_outliner", |ui| egui_outliner::show(ui, &model, &cfg))
            .inner;
        for a in actions {
            match a {
                OutlinerAction::Select {
                    id,
                    additive,
                    range,
                } => {
                    if let Some(id) = map.get(&id) {
                        self.select(e, *id, additive, range, &nodes);
                    }
                }
                OutlinerAction::SelectMany { ids } => {
                    e.selected = ids.iter().filter_map(|id| map.get(id).copied()).collect();
                    e.selection = e.selected.last().copied();
                }
                OutlinerAction::ClearSelection => {
                    e.selection = None;
                    e.selected.clear();
                }
                OutlinerAction::ToggleVisible { id, visible } => {
                    if let Some(id) = map.get(&id) {
                        self.command(
                            e,
                            WorldCommand::SetAttribute {
                                id: *id,
                                path: "/visible".into(),
                                value: json!(visible),
                                frame: self.playhead as f64,
                            },
                        );
                    }
                }
                OutlinerAction::Rename { id, name } => {
                    if let Some(id) = map.get(&id) {
                        self.command(e, WorldCommand::Rename { id: *id, name });
                    }
                }
                OutlinerAction::Move {
                    id,
                    new_parent,
                    index,
                } => {
                    if let Some(id) = map.get(&id).copied() {
                        let parent = new_parent.and_then(|p| map.get(&p).copied());
                        let ids = reparent_order(&nodes, id, parent, index);
                        self.command(
                            e,
                            WorldCommand::Batch(vec![
                                WorldCommand::Reparent { id, parent },
                                WorldCommand::Reorder { ids },
                            ]),
                        );
                    }
                }
                OutlinerAction::Context { id, action } => {
                    if let Some(id) = map.get(&id).copied() {
                        match action.as_str() {
                            "duplicate" => self.command(e, WorldCommand::Duplicate(id)),
                            "delete" => self.command(e, WorldCommand::Delete(id)),
                            "unparent" => {
                                self.command(e, WorldCommand::Reparent { id, parent: None })
                            }
                            "active" => {
                                if let Some(n) = nodes.iter().find(|n| n.id == id) {
                                    match n.kind {
                                        WorldKind::Camera => {
                                            self.command(e, WorldCommand::SetActiveCamera(id))
                                        }
                                        WorldKind::Environment => {
                                            self.command(e, WorldCommand::SetActiveEnvironment(id))
                                        }
                                        _ => {}
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }
        self.outliner_cache = cache;
    }
    fn refresh_metadata(&mut self, e: &WorldEditor, id: NodeId) {
        let document = Some(NodeId(e.document.graph.id));
        let changed = self.metadata_document != document || self.metadata_node != Some(id);
        if changed || self.metadata_revision != e.revision() {
            if changed || !self.metadata_dirty {
                self.metadata = e
                    .document
                    .metadata(id)
                    .ok()
                    .and_then(|value| serde_json::to_string_pretty(&value).ok())
                    .unwrap_or_else(|| "{}".into());
                self.metadata_dirty = false;
            }
            self.metadata_node = Some(id);
            self.metadata_document = document;
            self.metadata_revision = e.revision();
        }
    }
    fn apply_metadata(&mut self, e: &mut WorldEditor, id: NodeId) -> bool {
        let value = match serde_json::from_str(&self.metadata) {
            Ok(value) => value,
            Err(err) => {
                self.error = Some(format!("Invalid metadata JSON: {err}"));
                return false;
            }
        };
        match e.execute(WorldCommand::SetMetadata {
            id,
            path: "".into(),
            value,
        }) {
            Ok(()) => {
                self.metadata_dirty = false;
                self.metadata_document = Some(NodeId(e.document.graph.id));
                self.metadata_node = Some(id);
                self.metadata_revision = e.revision();
                true
            }
            Err(err) => {
                self.error = Some(err);
                false
            }
        }
    }
    fn take_outliner_cache(&mut self, e: &WorldEditor) -> OutlinerCache {
        let mut cache = std::mem::take(&mut self.outliner_cache);
        let document = Some(NodeId(e.document.graph.id));
        if cache.document != document || cache.revision != e.revision() {
            cache.nodes = e.document.nodes();
            cache.map.clear();
            cache
                .map
                .extend(cache.nodes.iter().map(|n| (wid(n.id), n.id)));
            cache.model.roots = tree(&cache.nodes, None, 0);
            if cache.document.is_none() {
                cache.config = OutlinerConfig::default()
                    .with_filter(true)
                    .with_editable_labels(true)
                    .with_draggable(true)
                    .with_visibility_on_right(true)
                    .with_row_height(24.0);
                cache.config.context_items = vec![
                    ContextItem::new("duplicate", "Duplicate"),
                    ContextItem::new("unparent", "Move to root"),
                    ContextItem::new("active", "Make active camera / environment"),
                    ContextItem::new("delete", "Delete").with_separator(),
                ];
            }
            cache.document = document;
            cache.revision = e.revision();
        }
        cache.model.selection.clear();
        cache
            .model
            .selection
            .extend(e.selected.iter().map(|id| wid(*id)));
        cache
    }
    fn take_cache(&mut self, e: &WorldEditor) -> AttributeCache {
        let mut cache = std::mem::take(&mut self.cache);
        let document = Some(NodeId(e.document.graph.id));
        let new_document = cache.document != document;
        let schema_changed = new_document || cache.revision != e.revision();
        let frame = (self.playhead as f64).to_bits();
        if new_document {
            cache.document = document;
            cache.attributes.clear();
            cache.sections.clear();
        }
        if schema_changed {
            cache.nodes = e.document.nodes();
        }
        if schema_changed || cache.frame != frame {
            for node in &mut cache.nodes {
                if schema_changed {
                    cache.attributes.insert(
                        node.id,
                        e.document
                            .attributes(node.id, self.playhead as f64)
                            .unwrap_or_default(),
                    );
                } else if let Some(attrs) = cache.attributes.get_mut(&node.id) {
                    let mut changed_shape = false;
                    for attr in attrs.iter_mut() {
                        match e
                            .document
                            .attribute_value(node.id, &attr.path, self.playhead as f64)
                        {
                            Ok(value) if same_value_shape(&attr.value, &value) => {
                                attr.value = value
                            }
                            _ => {
                                changed_shape = true;
                                break;
                            }
                        }
                    }
                    if changed_shape {
                        *attrs = e
                            .document
                            .attributes(node.id, self.playhead as f64)
                            .unwrap_or_default();
                        for ((id, _), section) in &mut cache.sections {
                            if *id == node.id {
                                section.fields.clear();
                            }
                        }
                    }
                }
                if let Some(attrs) = cache.attributes.get(&node.id) {
                    for attr in attrs {
                        if let Some(value) = attr.value.as_bool() {
                            match attr.path.as_str() {
                                "/visible" => node.visible = value,
                                "/locked" => node.locked = value,
                                "/solo" => node.solo = value,
                                _ => {}
                            }
                        }
                    }
                }
            }
        }
        cache.revision = e.revision();
        cache.frame = frame;
        cache
    }
    fn prepare_section(
        &self,
        section: &mut GridSectionCache,
        attrs: &[WorldAttribute],
        id: NodeId,
        group: &'static str,
        revision: u64,
        frame: u64,
    ) {
        let mut hash = std::collections::hash_map::DefaultHasher::new();
        for attr in attrs
            .iter()
            .filter(|a| a.component.is_none() && a.value.is_array() && category(&a.path) == group)
        {
            self.components_open(id, &attr.path, attrs).hash(&mut hash);
        }
        let projection = hash.finish();
        if section.fields.is_empty()
            || section.revision != revision
            || section.projection != projection
        {
            section.indices.clear();
            section.fields.clear();
            section.labels.clear();
            section.values.clear();
            for (index, attr) in attrs
                .iter()
                .enumerate()
                .filter(|(_, a)| a.component.is_none() && category(&a.path) == group)
            {
                section.indices.push(index);
                if self.components_open(id, &attr.path, attrs) {
                    section.indices.extend(
                        attrs
                            .iter()
                            .enumerate()
                            .filter(|(_, a)| {
                                a.component.is_some()
                                    && a.path
                                        .rsplit_once('/')
                                        .is_some_and(|(parent, _)| parent == attr.path)
                            })
                            .map(|(i, _)| i),
                    );
                }
            }
            for (order, &index) in section.indices.iter().enumerate() {
                let attr = &attrs[index];
                section.fields.push(
                    AttrField::new(&attr.path, grid_value(&attr.value)).with_order(order as f32),
                );
                section.labels.push(
                    attr.component
                        .map(component_label)
                        .map(str::to_owned)
                        .unwrap_or_else(|| property_label(&attr.path)),
                );
                section.values.push(attr.value.clone());
            }
            section.revision = revision;
            section.projection = projection;
            section.frame = frame;
        } else if section.frame != frame {
            for ((field, value), &index) in section
                .fields
                .iter_mut()
                .zip(&mut section.values)
                .zip(&section.indices)
            {
                field.value = grid_value(&attrs[index].value);
                value.clone_from(&attrs[index].value);
            }
            section.frame = frame;
        }
    }

    fn components_open(&self, id: NodeId, path: &str, attrs: &[WorldAttribute]) -> bool {
        self.component_open
            .get(&id)
            .is_some_and(|paths| paths.contains(path))
            || (!self
                .component_closed
                .get(&id)
                .is_some_and(|paths| paths.contains(path))
                && attrs.iter().any(|a| {
                    a.component.is_some()
                        && a.path
                            .rsplit_once('/')
                            .is_some_and(|(parent, _)| parent == path)
                        && !a.frames.is_empty()
                }))
    }
    fn toggle_components(&mut self, id: NodeId, path: &str, open: bool) {
        if let Some(paths) = self.component_open.get_mut(&id) {
            paths.remove(path);
        }
        if let Some(paths) = self.component_closed.get_mut(&id) {
            paths.remove(path);
        }
        if open {
            self.component_closed
                .entry(id)
                .or_default()
                .insert(path.into());
        } else {
            self.component_open
                .entry(id)
                .or_default()
                .insert(path.into());
        }
    }
    fn lanes(&self, id: NodeId, attrs: &[WorldAttribute]) -> Vec<Lane> {
        let mut lanes = Vec::new();
        for group in ATTRIBUTE_GROUPS {
            let parents: Vec<_> = attrs
                .iter()
                .filter(|a| a.keyable && a.component.is_none() && category(&a.path) == group)
                .collect();
            if parents.is_empty() {
                continue;
            }
            lanes.push(Lane {
                label: group.into(),
                path: Some(format!("@{group}")),
                value: Value::Null,
                frames: vec![],
                depth: 1,
                group: true,
                attr: None,
            });
            if !self.groups.contains(&(id, format!("@{group}"))) {
                continue;
            }
            for a in parents {
                lanes.push(Lane {
                    label: property_label(&a.path),
                    path: Some(a.path.clone()),
                    value: a.value.clone(),
                    frames: a.frames.clone(),
                    depth: 2,
                    group: false,
                    attr: Some(a.clone()),
                });
                if self.components_open(id, &a.path, &attrs) {
                    for child in attrs.iter().filter(|child| {
                        child.component.is_some()
                            && child
                                .path
                                .rsplit_once('/')
                                .is_some_and(|(parent, _)| parent == a.path)
                    }) {
                        lanes.push(Lane {
                            label: component_label(child.component.unwrap()).into(),
                            path: Some(child.path.clone()),
                            value: child.value.clone(),
                            frames: child.frames.clone(),
                            depth: 3,
                            group: false,
                            attr: Some(child.clone()),
                        });
                    }
                }
            }
        }
        lanes
    }
    pub fn timeline(&mut self, ui: &mut egui::Ui, e: &mut WorldEditor) {
        self.transport(ui, e);
        self.status(ui);
        let cache = self.take_cache(e);
        let nodes = &cache.nodes;
        let mut timeline = std::mem::take(&mut self.timeline_cache);
        let mut hash = std::collections::hash_map::DefaultHasher::new();
        for node in nodes {
            node.id.hash(&mut hash);
            self.expanded.contains(&node.id).hash(&mut hash);
            for group in ATTRIBUTE_GROUPS {
                self.groups
                    .iter()
                    .any(|(id, path)| *id == node.id && path.strip_prefix('@') == Some(group))
                    .hash(&mut hash);
            }
            if let Some(attrs) = cache.attributes.get(&node.id) {
                for attr in attrs {
                    attr.path.hash(&mut hash);
                    if attr.component.is_none() && attr.value.is_array() {
                        self.components_open(node.id, &attr.path, attrs)
                            .hash(&mut hash);
                    }
                }
            }
        }
        let projection = hash.finish();
        if timeline.document != cache.document
            || timeline.revision != cache.revision
            || timeline.projection != projection
        {
            timeline.lanes = nodes
                .iter()
                .map(|n| {
                    self.lanes(
                        n.id,
                        cache
                            .attributes
                            .get(&n.id)
                            .map(Vec::as_slice)
                            .unwrap_or_default(),
                    )
                })
                .collect();
            timeline.model.tracks = nodes
                .iter()
                .zip(&timeline.lanes)
                .map(|(n, ls)| {
                    let clip = Clip::new(
                        wid(n.id),
                        n.start.round() as i64,
                        (n.end - n.start).round().max(1.0) as i64,
                        n.name.clone(),
                    )
                    .with_color(kind_color(n.kind));
                    let props = ls
                        .iter()
                        .map(|l| {
                            PropLane::new(
                                l.label.clone(),
                                l.frames
                                    .iter()
                                    .filter(|_| !l.value.is_array())
                                    .map(|f| Keyframe::new(*f, l.value.as_f64().unwrap_or(0.0)))
                                    .collect(),
                            )
                            .with_color(kind_color(n.kind))
                        })
                        .collect();
                    Track::new(n.name.clone(), vec![clip])
                        .with_lanes(props)
                        .expanded(self.expanded.contains(&n.id))
                })
                .collect();
            timeline.document = cache.document;
            timeline.revision = cache.revision;
            timeline.projection = projection;
        }
        if timeline.frame != cache.frame {
            for (node, ls) in nodes.iter().zip(&mut timeline.lanes) {
                if let Some(attrs) = cache.attributes.get(&node.id) {
                    for lane in ls {
                        if let Some(attr) = attrs
                            .iter()
                            .find(|attr| Some(&attr.path) == lane.path.as_ref())
                        {
                            lane.value.clone_from(&attr.value);
                            if let Some(descriptor) = &mut lane.attr {
                                descriptor.value.clone_from(&attr.value);
                            }
                        }
                    }
                }
            }
            timeline.frame = cache.frame;
        }
        for ((node, track), lanes) in nodes
            .iter()
            .zip(&mut timeline.model.tracks)
            .zip(&timeline.lanes)
        {
            for (lane, prop) in lanes.iter().zip(&mut track.lanes) {
                for key in &mut prop.keys {
                    key.selected = self.keys.iter().any(|selected| {
                        selected.node == node.id
                            && Some(&selected.path) == lane.path.as_ref()
                            && selected.frame == key.frame.to_bits()
                    });
                }
            }
        }
        let lanes = &mut timeline.lanes;
        let model = &mut timeline.model;
        model.fps = e.document.fps as f32;
        model.playhead = self.playhead as i64;
        model.selection.clear();
        model.selection.extend(e.selected.iter().map(|id| wid(*id)));
        model.work_area = Some(WorkArea {
            start: e.document.first as i64,
            end: e.document.last as i64 + 1,
        });
        let cfg = TimelineConfig {
            row_height: self.attribute_metrics.row_height(),
            lane_height: self.attribute_metrics.row_height(),
            snap_threshold: if self.snap { 7.0 } else { 0.0 },
            snap_keys_to_frames: self.snap,
            ..Default::default()
        };
        let left_w = (ui.available_width() * 0.36)
            .clamp(270.0, 430.0)
            .min((ui.available_width() - 140.0).max(80.0));
        let width = ui.available_width();
        let mut actions = vec![];
        egui::ScrollArea::vertical()
            .id_salt("world_timeline_scroll")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                let origin = ui.cursor().min;
                let content_height = 20.0
                    + model
                        .tracks
                        .iter()
                        .map(|t| t.height(cfg.row_height, cfg.lane_height))
                        .sum::<f32>()
                        .max(cfg.row_height);
                let right_rect = Rect::from_min_size(
                    origin + Vec2::new(left_w, 0.0),
                    Vec2::new((width - left_w).max(100.0), content_height),
                );
                let mut canvas = ui.new_child(
                    egui::UiBuilder::new()
                        .id_salt("world_timeline_canvas")
                        .max_rect(right_rect),
                );
                canvas.set_clip_rect(canvas.clip_rect().intersect(right_rect));
                let resp = TrackTimeline::new(cfg).show(&mut canvas, &mut self.view, &model);
                let header =
                    Rect::from_min_size(origin, Vec2::new(left_w, resp.ruler_rect.height()));
                ui.painter()
                    .rect_filled(header, 0.0, ui.visuals().faint_bg_color);
                ui.painter().text(
                    header.left_center() + Vec2::new(7.0, 0.0),
                    egui::Align2::LEFT_CENTER,
                    "Name / animated properties",
                    egui::FontId::proportional(11.0),
                    ui.visuals().weak_text_color(),
                );
                for (ti, n) in nodes.iter().enumerate() {
                    let y = resp.track_tops[ti];
                    let row = Rect::from_min_size(
                        Pos2::new(origin.x, y),
                        Vec2::new(left_w, cfg.row_height),
                    );
                    self.layer_row(ui, e, n, &nodes, row);
                    if model.tracks[ti].expanded {
                        for (li, l) in lanes[ti].iter_mut().enumerate() {
                            let rect = Rect::from_min_size(
                                Pos2::new(
                                    origin.x,
                                    y + cfg.row_height + li as f32 * resp.lane_height,
                                ),
                                Vec2::new(left_w, resp.lane_height),
                            );
                            self.lane_row(
                                ui,
                                e,
                                n.id,
                                l,
                                rect,
                                cache
                                    .attributes
                                    .get(&n.id)
                                    .map(Vec::as_slice)
                                    .unwrap_or_default(),
                                nodes,
                            );
                        }
                    }
                }
                actions = resp.actions;
                ui.allocate_rect(
                    Rect::from_min_size(
                        origin,
                        Vec2::new(width, resp.ruler_rect.height() + resp.content_height),
                    ),
                    Sense::hover(),
                );
            });
        for a in actions {
            self.timeline_action(e, &nodes, &lanes, a);
        }
        if ui.input(|i| i.pointer.any_released()) {
            self.drag_order = None;
        }
        self.timeline_cache = timeline;
        self.cache = cache;
    }
    fn layer_row(
        &mut self,
        ui: &mut egui::Ui,
        e: &mut WorldEditor,
        n: &WorldNodeInfo,
        nodes: &[WorldNodeInfo],
        r: Rect,
    ) {
        self.attribute_metrics.apply(ui);
        if !r.intersects(ui.clip_rect()) {
            return;
        }
        if e.selected.contains(&n.id) {
            ui.painter()
                .rect_filled(r, 0.0, ui.visuals().selection.bg_fill);
        }
        ui.painter().rect_filled(
            Rect::from_min_size(r.min, Vec2::new(3.0, r.height())),
            0.0,
            kind_color(n.kind),
        );
        ui.push_id(("layer", wid(n.id)), |ui| {
            let body = r.shrink2(Vec2::new(5.0, 1.0));
            let side = self
                .attribute_metrics
                .icon_side
                .min(body.height())
                .min((body.width() / 6.0).max(0.0));
            let mut x = body.left();
            let gap = self.attribute_metrics.component_gap;
            let mut next = || {
                let rect = Rect::from_min_size(
                    Pos2::new(x, body.center().y - side * 0.5),
                    Vec2::splat(side),
                );
                x += side + gap;
                rect
            };
            if square_icon(
                ui,
                next(),
                if self.expanded.contains(&n.id) {
                    ph::CARET_DOWN
                } else {
                    ph::CARET_RIGHT
                },
                true,
            )
            .on_hover_text("Expand layer properties")
            .clicked()
            {
                if !self.expanded.remove(&n.id) {
                    self.expanded.insert(n.id);
                    self.groups.insert((n.id, "@Transform".into()));
                }
            }
            for (path, state, on, off, tip) in [
                ("/visible", n.visible, ph::EYE, ph::EYE_SLASH, "Visibility"),
                ("/locked", n.locked, ph::LOCK, ph::LOCK_OPEN, "Lock"),
                ("/solo", n.solo, "S", "s", "Solo"),
            ] {
                if icon_cell(ui, next(), if state { on } else { off }, state, true)
                    .on_hover_text(tip)
                    .clicked()
                {
                    self.command(
                        e,
                        WorldCommand::SetAttribute {
                            id: n.id,
                            path: path.into(),
                            value: json!(!state),
                            frame: self.playhead as f64,
                        },
                    );
                }
            }
            let icon = next();
            ui.painter().text(
                icon.center(),
                egui::Align2::CENTER_CENTER,
                kind_icon(n.kind),
                egui::FontId::proportional(side),
                kind_color(n.kind),
            );
            let name_rect =
                Rect::from_min_max(Pos2::new(x.min(body.right()), body.top()), body.max);
            let payload = LayerDrag {
                id: n.id,
                order: self.drag_order.clone(),
            };
            let source = cell_ui(ui, name_rect, "layer_name", |ui| {
                ui.dnd_drag_source(ui.id().with(("layer_drag", wid(n.id))), payload, |ui| {
                    ui.add_sized(
                        name_rect.size(),
                        egui::Button::new(())
                            .left_text(&n.name)
                            .truncate()
                            .selected(e.selected.contains(&n.id))
                            .frame(false),
                    )
                })
            })
            .inner;
            if source.response.drag_started() && self.drag_order.is_none() {
                self.drag_order = Some(std::sync::Arc::new(nodes.iter().map(|n| n.id).collect()));
            }
            if source.inner.clicked() {
                let modifiers = ui.input(|i| i.modifiers);
                self.select(
                    e,
                    n.id,
                    modifiers.command || modifiers.ctrl,
                    modifiers.shift,
                    nodes,
                );
            }
            let target = ui.interact(r, ui.id().with("layer_drop"), Sense::hover());
            if let Some(payload) = target.dnd_hover_payload::<LayerDrag>() {
                if payload.id != n.id {
                    let after = ui
                        .ctx()
                        .pointer_interact_pos()
                        .is_some_and(|p| p.y > r.center().y);
                    let y = if after { r.bottom() } else { r.top() };
                    ui.painter().line_segment(
                        [Pos2::new(r.left() + 3.0, y), Pos2::new(r.right(), y)],
                        egui::Stroke::new(2.0, ui.visuals().selection.stroke.color),
                    );
                }
            }
            if let Some(payload) = target.dnd_release_payload::<LayerDrag>() {
                if let Some(order) = &payload.order {
                    let ids = layer_drop_order(
                        nodes,
                        order,
                        payload.id,
                        n.id,
                        ui.ctx()
                            .pointer_interact_pos()
                            .is_some_and(|p| p.y > r.center().y),
                    );
                    if ids.iter().copied().ne(nodes.iter().map(|n| n.id)) {
                        self.command(e, WorldCommand::Reorder { ids });
                    }
                }
            }
        });
    }

    fn lane_row(
        &mut self,
        ui: &mut egui::Ui,
        e: &mut WorldEditor,
        id: NodeId,
        lane: &mut Lane,
        r: Rect,
        attrs: &[WorldAttribute],
        nodes: &[WorldNodeInfo],
    ) {
        if !r.intersects(ui.clip_rect()) {
            return;
        }
        let Some(path) = &lane.path else {
            return;
        };
        self.attribute_metrics.apply(ui);
        let cells = property_rects(r, self.attribute_label_width, self.attribute_metrics);
        ui.push_id((wid(id), path), |ui| {
            if lane.group {
                let open = self.groups.contains(&(id, path.clone()));
                if square_icon(
                    ui,
                    cells.prefix,
                    if open {
                        ph::CARET_DOWN
                    } else {
                        ph::CARET_RIGHT
                    },
                    true,
                )
                .clicked()
                {
                    if !self.groups.remove(&(id, path.clone())) {
                        self.groups.insert((id, path.clone()));
                    }
                }
                cell_ui(ui, cells.label, "group", |ui| {
                    ui.strong(&lane.label);
                });
                return;
            }
            let Some(attr) = lane.attr.as_ref() else {
                return;
            };
            if let Some(command) = animation_controls(
                ui,
                id,
                attr,
                self.playhead as f64,
                cells.prefix,
                self.attribute_metrics,
            ) {
                self.command(e, command);
            }
            let label = Rect::from_min_max(
                cells.label.min
                    + Vec2::new(
                        (lane.depth.saturating_sub(2) as f32 * 10.0).min(cells.label.width()),
                        0.0,
                    ),
                cells.label.max,
            );
            cell_ui(ui, label, "label", |ui| {
                ui.add(egui::Label::new(&lane.label).truncate())
                    .on_hover_text(path);
            });
            let changed = cell_ui(ui, cells.value, "value", |ui| {
                self.edit_attribute(ui, id, &mut lane.value, attr, nodes)
            })
            .inner;
            if changed {
                let command = WorldCommand::SetAttribute {
                    id,
                    path: path.clone(),
                    value: lane.value.clone(),
                    frame: self.playhead as f64,
                };
                if value_gesture_eligible(attr, &lane.value) {
                    self.value_command(ui, e, command);
                } else {
                    self.command(e, command);
                }
            }
            let left = Rect::from_min_size(
                cells.actions.min,
                Vec2::new(
                    (cells.actions.width() * 0.5 - self.attribute_metrics.component_gap * 0.5)
                        .max(0.0),
                    cells.actions.height(),
                ),
            );
            let right = Rect::from_min_max(
                Pos2::new(
                    cells.actions.center().x + self.attribute_metrics.component_gap * 0.5,
                    cells.actions.top(),
                ),
                cells.actions.max,
            );
            if path == "/julia" {
                let enabled = !lane.value.is_null();
                if icon_cell(ui, left, ph::POWER, enabled, true)
                    .on_hover_text("Enable / disable Julia")
                    .clicked()
                {
                    self.command(
                        e,
                        WorldCommand::SetAttribute {
                            id,
                            path: path.clone(),
                            value: if enabled {
                                Value::Null
                            } else {
                                json!([0.0, 0.0, 0.0])
                            },
                            frame: self.playhead as f64,
                        },
                    );
                }
            } else if is_color(path, &lane.value) {
                let mut rgb = std::array::from_fn::<_, 3, _>(|i| {
                    lane.value[i].as_f64().unwrap_or_default() as f32
                });
                if cell_ui(ui, left, "color", |ui| {
                    ui.spacing_mut().interact_size = Vec2::splat(self.attribute_metrics.icon_side);
                    ui.color_edit_button_rgb(&mut rgb).changed()
                })
                .inner
                {
                    self.value_command(
                        ui,
                        e,
                        WorldCommand::SetAttribute {
                            id,
                            path: path.clone(),
                            value: json!(rgb),
                            frame: self.playhead as f64,
                        },
                    );
                }
            }
            if attr.component.is_none() && lane.value.is_array() {
                let open = self.components_open(id, path, attrs);
                if icon_cell(
                    ui,
                    right,
                    if open {
                        ph::CARET_DOWN
                    } else {
                        ph::CARET_RIGHT
                    },
                    false,
                    true,
                )
                .on_hover_text("Show component channels")
                .clicked()
                {
                    self.toggle_components(id, path, open);
                }
            }
        });
    }
    fn key_identity(
        nodes: &[WorldNodeInfo],
        lanes: &[Vec<Lane>],
        k: KeyPos,
    ) -> Option<KeyIdentity> {
        Some(KeyIdentity {
            node: nodes.get(k.track)?.id,
            path: lanes
                .get(k.track)?
                .get(k.lane)?
                .path
                .clone()
                .filter(|p| !p.starts_with('@'))?,
            frame: k.frame.to_bits(),
        })
    }
    fn timeline_action(
        &mut self,
        e: &mut WorldEditor,
        nodes: &[WorldNodeInfo],
        lanes: &[Vec<Lane>],
        a: TimelineAction,
    ) {
        match a {
            TimelineAction::Seek { frame } => {
                self.seek(frame.clamp(e.document.first as i64, e.document.last as i64) as u32)
            }
            TimelineAction::Select {
                id,
                additive,
                range,
            } => {
                if let Some(n) = nodes.iter().find(|n| wid(n.id) == id) {
                    self.select(e, n.id, additive, range, nodes);
                }
            }
            TimelineAction::ClearSelection => {
                e.selection = None;
                e.selected.clear();
            }
            TimelineAction::SetWorkArea { start, end } => {
                self.command(
                    e,
                    WorldCommand::SetTimeRange {
                        first: start.max(0) as u32,
                        last: (end - 1).max(start).max(0) as u32,
                        fps: e.document.fps,
                    },
                );
                self.playhead = self.playhead.clamp(e.document.first, e.document.last);
            }
            TimelineAction::SelectKeys { keys, add } => {
                if !add {
                    self.keys.clear();
                }
                for k in keys {
                    if let Some(key) = Self::key_identity(nodes, lanes, k) {
                        e.selection = Some(key.node);
                        self.keys.insert(key);
                    }
                }
                e.selected = nodes
                    .iter()
                    .filter(|n| self.keys.iter().any(|k| k.node == n.id))
                    .map(|n| n.id)
                    .collect();
            }
            TimelineAction::ClearKeySelection => self.keys.clear(),
            TimelineAction::AddKey { track, lane, frame } => {
                if let Some(k) = Self::key_identity(nodes, lanes, KeyPos { track, lane, frame }) {
                    self.command(
                        e,
                        WorldCommand::Key {
                            id: k.node,
                            path: k.path,
                            frame,
                        },
                    );
                }
            }
            TimelineAction::MoveKeys { keys, delta } => {
                let mut grouped: HashMap<(NodeId, String), Vec<f64>> = HashMap::new();
                let mut moved = HashSet::new();
                for k in keys {
                    if let Some(k) = Self::key_identity(nodes, lanes, k) {
                        let f = f64::from_bits(k.frame);
                        grouped.entry((k.node, k.path.clone())).or_default().push(f);
                        moved.insert(KeyIdentity {
                            frame: (f + delta).to_bits(),
                            ..k
                        });
                    }
                }
                let command = WorldCommand::Batch(
                    grouped
                        .into_iter()
                        .map(|((id, path), frames)| WorldCommand::MoveKeys {
                            id,
                            path,
                            frames,
                            delta,
                        })
                        .collect(),
                );
                match e.execute(command) {
                    Ok(()) => self.keys = moved,
                    Err(err) => self.error = Some(err),
                }
            }
            TimelineAction::DeleteKeys { keys } => {
                let cmds = keys
                    .into_iter()
                    .filter_map(|k| Self::key_identity(nodes, lanes, k))
                    .map(|k| WorldCommand::RemoveKey {
                        id: k.node,
                        path: k.path,
                        frame: f64::from_bits(k.frame),
                    })
                    .collect();
                match e.execute(WorldCommand::Batch(cmds)) {
                    Ok(()) => self.keys.clear(),
                    Err(err) => self.error = Some(err),
                }
            }
            TimelineAction::MoveClip {
                id,
                new_start,
                new_track,
            } => {
                if let Some(n) = nodes.iter().find(|n| wid(n.id) == id) {
                    let mut ids = nodes.iter().map(|n| n.id).collect::<Vec<_>>();
                    if let Some(old) = ids.iter().position(|id| *id == n.id) {
                        let moved = ids.remove(old);
                        let to = new_track.min(ids.len());
                        ids.insert(to, moved);
                    }
                    self.command(
                        e,
                        WorldCommand::Batch(vec![
                            WorldCommand::SetSpan {
                                id: n.id,
                                start: new_start as f64,
                                end: new_start as f64 + n.end - n.start,
                            },
                            WorldCommand::Reorder { ids },
                        ]),
                    );
                }
            }
            TimelineAction::TrimStart { id, delta } => {
                if let Some(n) = nodes.iter().find(|n| wid(n.id) == id) {
                    self.command(
                        e,
                        WorldCommand::SetSpan {
                            id: n.id,
                            start: (n.start + delta as f64).min(n.end - 1.0),
                            end: n.end,
                        },
                    );
                }
            }
            TimelineAction::TrimEnd { id, delta } => {
                if let Some(n) = nodes.iter().find(|n| wid(n.id) == id) {
                    self.command(
                        e,
                        WorldCommand::SetSpan {
                            id: n.id,
                            start: n.start,
                            end: (n.end - delta as f64).max(n.start + 1.0),
                        },
                    );
                }
            }
            _ => {}
        }
    }
    fn transport(&mut self, ui: &mut egui::Ui, e: &mut WorldEditor) {
        self.attribute_metrics.apply(ui);
        let (mut first, mut last, mut fps) = (e.document.first, e.document.last, e.document.fps);
        let before = (first, last, fps);
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = Vec2::new(2.0, 0.0);
            ui.spacing_mut().interact_size.y = self.attribute_metrics.field_height;
            if transport_icon(ui, ph::SKIP_BACK, false, self.attribute_metrics)
                .on_hover_text("First frame")
                .clicked()
            {
                self.seek(e.document.first);
            }
            if transport_icon(ui, ph::CARET_LEFT, false, self.attribute_metrics).clicked() {
                self.seek(self.playhead.saturating_sub(1).max(e.document.first));
            }
            if transport_icon(
                ui,
                if self.playing { ph::PAUSE } else { ph::PLAY },
                self.playing,
                self.attribute_metrics,
            )
            .on_hover_text("Play / pause")
            .clicked()
            {
                self.playing = !self.playing;
            }
            if transport_icon(ui, ph::CARET_RIGHT, false, self.attribute_metrics).clicked() {
                self.seek(self.playhead.saturating_add(1).min(e.document.last));
            }
            if transport_icon(ui, ph::SKIP_FORWARD, false, self.attribute_metrics).clicked() {
                self.seek(e.document.last);
            }
            if transport_icon(
                ui,
                ph::ARROWS_CLOCKWISE,
                self.looping,
                self.attribute_metrics,
            )
            .on_hover_text("Loop work area")
            .clicked()
            {
                self.looping = !self.looping;
            }
            ui.label("Frame");
            ui.add_sized(
                [
                    self.attribute_metrics.numeric_width,
                    self.attribute_metrics.field_height,
                ],
                egui::DragValue::new(&mut self.playhead).range(e.document.first..=e.document.last),
            );
            ui.separator();
            ui.label("FPS");
            ui.add_sized(
                [
                    self.attribute_metrics.numeric_width,
                    self.attribute_metrics.field_height,
                ],
                egui::DragValue::new(&mut fps).speed(1).range(1.0..=240.0),
            );
            ui.label("In");
            ui.add_sized(
                [
                    self.attribute_metrics.numeric_width,
                    self.attribute_metrics.field_height,
                ],
                egui::DragValue::new(&mut first).range(0..=last),
            );
            ui.label("Out");
            ui.add_sized(
                [
                    self.attribute_metrics.numeric_width,
                    self.attribute_metrics.field_height,
                ],
                egui::DragValue::new(&mut last).range(first..=100000),
            );
            ui.checkbox(&mut self.snap, "Snap");
            if ui.small_button("Fit").clicked() {
                self.view.pan_offset = e.document.first as f32;
                self.view.zoom = ((ui.available_width().max(500.0) * 0.55)
                    / (e.document.last - e.document.first + 1) as f32
                    / 2.0)
                    .clamp(0.02, 50.0);
            }
            ui.add(
                egui::Slider::new(&mut self.view.zoom, 0.05..=20.0)
                    .logarithmic(true)
                    .text("Zoom"),
            );
            ui.menu_button("Interpolation", |ui| {
                for kind in CurveKind::all() {
                    if ui
                        .add_enabled(!self.keys.is_empty(), egui::Button::new(kind.label()))
                        .clicked()
                    {
                        let mut grouped: HashMap<(NodeId, String), Vec<f64>> = HashMap::new();
                        for k in &self.keys {
                            grouped
                                .entry((k.node, k.path.clone()))
                                .or_default()
                                .push(f64::from_bits(k.frame));
                        }
                        self.command(
                            e,
                            WorldCommand::Batch(
                                grouped
                                    .into_iter()
                                    .map(|((id, path), frames)| WorldCommand::Interpolation {
                                        id,
                                        path,
                                        frames,
                                        kind,
                                    })
                                    .collect(),
                            ),
                        );
                        ui.close();
                    }
                }
            });
        });
        if (first, last, fps) != before {
            self.command(e, WorldCommand::SetTimeRange { first, last, fps });
            self.playhead = self.playhead.clamp(e.document.first, e.document.last);
        }
    }
    fn edit_attribute(
        &mut self,
        ui: &mut egui::Ui,
        id: NodeId,
        value: &mut Value,
        a: &WorldAttribute,
        materials: &[WorldNodeInfo],
    ) -> bool {
        if a.path == "/material_id" {
            let selected = value.as_str().and_then(NodeId::parse);
            let label = materials
                .iter()
                .find(|n| Some(n.id) == selected)
                .map(|n| n.name.as_str())
                .unwrap_or("Object material");
            let mut changed = false;
            let area = Rect::from_min_size(
                ui.cursor().min,
                Vec2::new(ui.available_width(), ui.spacing().interact_size.y),
            );
            let button = Rect::from_center_size(
                Pos2::new(
                    area.right() - self.attribute_metrics.icon_side * 0.5,
                    area.center().y,
                ),
                Vec2::splat(self.attribute_metrics.icon_side),
            );
            let combo = Rect::from_min_max(
                area.min,
                Pos2::new((button.left() - 4.0).max(area.left()), area.bottom()),
            );
            cell_ui(ui, combo, "material_reference", |ui| {
                egui::ComboBox::from_id_salt("material_reference")
                    .width(combo.width())
                    .selected_text(label)
                    .show_ui(ui, |ui| {
                        changed |= ui
                            .selectable_value(value, Value::Null, "Object material")
                            .changed();
                        for material in materials.iter().filter(|n| n.kind == WorldKind::Material) {
                            changed |= ui
                                .selectable_value(value, json!(material.id), &material.name)
                                .changed();
                        }
                    });
            });
            if icon_cell(ui, button, ph::PALETTE, false, true)
                .on_hover_text("Open material library")
                .clicked()
            {
                self.material_library_requested = true;
            }
            changed
        } else if a.path == "/environment/path" {
            if !value.is_string() {
                *value = json!("");
            }
            let Value::String(path) = value else {
                return false;
            };
            let response = egui_file_field::FileField::new(path, &mut self.environment_file)
                .hint("HDR / EXR environment")
                .tooltip("Browse HDR / EXR environment")
                .show(ui);
            if response.browse_clicked() {
                let start = self.environment_file.start_dir(path);
                let mut picker = egui_file_dialog::FileDialog::new().add_file_filter(
                    "HDR / EXR",
                    egui_file_dialog::Filter::new(|p: &std::path::Path| {
                        p.extension()
                            .and_then(|ext| ext.to_str())
                            .is_some_and(|ext| {
                                ext.eq_ignore_ascii_case("hdr") || ext.eq_ignore_ascii_case("exr")
                            })
                    }),
                );
                if !start.is_empty() {
                    picker = picker.initial_directory(start.into());
                }
                picker.pick_file();
                self.picker = Some((id, picker));
            }
            response.changed()
        } else {
            schema_editor(ui, value, a, self.attribute_metrics.value_layout())
        }
    }
    pub fn inspector(&mut self, ui: &mut egui::Ui, e: &mut WorldEditor) {
        self.attribute_metrics.apply(ui);
        if let Some((id, picker)) = &mut self.picker {
            picker.update(ui.ctx());
            if let Some(path) = picker.take_picked() {
                let id = *id;
                self.environment_file.remember(&path.to_string_lossy());
                self.command(
                    e,
                    WorldCommand::Batch(vec![
                        WorldCommand::SetAttribute {
                            id,
                            path: "/environment/path".into(),
                            value: json!(path.to_string_lossy()),
                            frame: self.playhead as f64,
                        },
                        WorldCommand::SetAttribute {
                            id,
                            path: "/environment/enabled".into(),
                            value: json!(true),
                            frame: self.playhead as f64,
                        },
                    ]),
                );
                self.picker = None;
            }
        }
        self.status(ui);
        let Some(id) = e.selection else {
            ui.label("Select an object to edit its properties.");
            return;
        };
        let mut cache = self.take_cache(e);
        let Some(node) = cache.nodes.iter().find(|n| n.id == id) else {
            self.cache = cache;
            return;
        };
        ui.horizontal(|ui| {
            ui.colored_label(kind_color(node.kind), kind_icon(node.kind));
            ui.heading(&node.name);
        });
        if e.selected.len() > 1 {
            ui.weak(format!(
                "{} objects selected; editing {}",
                e.selected.len(),
                node.name
            ));
        }
        let mut name = node.name.clone();
        if ui
            .add(egui::TextEdit::singleline(&mut name).desired_width(f32::INFINITY))
            .changed()
        {
            self.command(e, WorldCommand::Rename { id, name });
        }
        if node.kind == WorldKind::Camera {
            if ui.button("Use as active camera").clicked() {
                self.command(e, WorldCommand::SetActiveCamera(id));
            }
        }
        if node.kind == WorldKind::Environment {
            ui.horizontal(|ui| {
                if ui.button("Reload").clicked() {
                    self.command(e, WorldCommand::ReloadEnvironment(id));
                }
                if ui.button("Clear").clicked() {
                    self.command(
                        e,
                        WorldCommand::Batch(vec![
                            WorldCommand::SetAttribute {
                                id,
                                path: "/environment/path".into(),
                                value: json!(""),
                                frame: self.playhead as f64,
                            },
                            WorldCommand::SetAttribute {
                                id,
                                path: "/environment/enabled".into(),
                                value: json!(false),
                                frame: self.playhead as f64,
                            },
                        ]),
                    );
                }
                if ui.button("Use environment").clicked() {
                    self.command(e, WorldCommand::SetActiveEnvironment(id));
                }
            });
        }
        let attrs = cache
            .attributes
            .get(&id)
            .map(Vec::as_slice)
            .unwrap_or_default();
        let revision = cache.revision;
        let frame = cache.frame;
        egui::ScrollArea::vertical()
            .id_salt(("world_inspector", wid(id)))
            .show(ui, |ui| {
                for group in ATTRIBUTE_GROUPS {
                    if !attrs
                        .iter()
                        .any(|a| a.component.is_none() && category(&a.path) == group)
                    {
                        continue;
                    }
                    let open = !self
                        .section_closed
                        .get(&id)
                        .is_some_and(|sections| sections.contains(group));
                    let section = cache.sections.entry((id, group)).or_default();
                    self.prepare_section(section, attrs, id, group, revision, frame);
                    let response = egui_titlebar::CollapsingSection::new(group)
                        .id_salt((wid(id), group))
                        .open(open)
                        .tint(kind_color(node.kind), 0.16)
                        .show(ui, |ui| {
                            let config = self.attribute_metrics.grid_config();
                            section.state.table.widths.resize(1, 0.0);
                            section.state.table.widths[0] = self.attribute_label_width;
                            let mut commands = Vec::new();
                            let mut value_commands = Vec::new();
                            let mut hooks = WorldGridHooks {
                                ui_state: self,
                                id,
                                frame: f64::from_bits(frame),
                                attrs,
                                indices: &section.indices,
                                labels: &section.labels,
                                values: &mut section.values,
                                materials: &cache.nodes,
                                commands: &mut commands,
                                value_commands: &mut value_commands,
                            };
                            let changes = render_grid_with_config(
                                ui,
                                &mut section.fields,
                                &mut section.state,
                                &HashSet::new(),
                                &config,
                                &mut hooks,
                            );
                            for (path, value) in changes {
                                if let Some(attr) = attrs.iter().find(|a| a.path == path) {
                                    let value = grid_json(value, &attr.value);
                                    let eligible = value_gesture_eligible(attr, &value);
                                    let command = WorldCommand::SetAttribute {
                                        id,
                                        path,
                                        value,
                                        frame: f64::from_bits(frame),
                                    };
                                    if eligible {
                                        value_commands.push(command);
                                    } else {
                                        commands.push(command);
                                    }
                                }
                            }
                            if let Some(width) = section.state.table.widths.first() {
                                self.attribute_label_width = *width;
                            }
                            if !value_commands.is_empty() {
                                self.value_command(ui, e, WorldCommand::Batch(value_commands));
                            }
                            if !commands.is_empty() {
                                self.command(e, WorldCommand::Batch(commands));
                            }
                        });
                    if response.header.open != open {
                        let closed = self.section_closed.entry(id).or_default();
                        if response.header.open {
                            closed.remove(group);
                        } else {
                            closed.insert(group);
                        }
                    }
                }
                egui::CollapsingHeader::new("Custom numeric attribute").show(ui, |ui| {
                    ui.add(
                        egui::TextEdit::singleline(&mut self.custom_name)
                            .hint_text("Attribute name"),
                    );
                    ui.add(egui::DragValue::new(&mut self.custom_value).speed(0.1));
                    if ui
                        .add_enabled(
                            !self.custom_name.trim().is_empty(),
                            egui::Button::new("Add attribute"),
                        )
                        .clicked()
                    {
                        let name = self.custom_name.trim().replace('/', "_");
                        self.command(
                            e,
                            WorldCommand::SetAttribute {
                                id,
                                path: format!("/custom/{name}"),
                                value: json!(self.custom_value),
                                frame: self.playhead as f64,
                            },
                        );
                        self.custom_name.clear();
                    }
                });
                egui::CollapsingHeader::new("Metadata").show(ui, |ui| {
                    self.refresh_metadata(e, id);
                    if ui
                        .add(
                            egui::TextEdit::multiline(&mut self.metadata)
                                .code_editor()
                                .desired_rows(8)
                                .desired_width(f32::INFINITY),
                        )
                        .changed()
                    {
                        self.metadata_dirty = true;
                    }
                    if ui.button("Apply metadata").clicked() {
                        self.apply_metadata(e, id);
                    }
                });
            });
        self.cache = cache;
    }
}
struct WorldGridHooks<'a> {
    ui_state: &'a mut WorldUi,
    id: NodeId,
    frame: f64,
    attrs: &'a [WorldAttribute],
    indices: &'a [usize],
    labels: &'a [String],
    values: &'a mut [Value],
    materials: &'a [WorldNodeInfo],
    commands: &'a mut Vec<WorldCommand>,
    value_commands: &'a mut Vec<WorldCommand>,
}
impl WorldGridHooks<'_> {
    fn index(&self, field: &AttrField) -> Option<usize> {
        self.indices
            .iter()
            .position(|&index| self.attrs[index].path == field.key)
    }
}
impl AttrGridHooks for WorldGridHooks<'_> {
    fn display_label<'a>(&'a self, field: &'a AttrField) -> &'a str {
        self.index(field)
            .map(|index| self.labels[index].as_str())
            .unwrap_or(&field.key)
    }
    fn prefix(&mut self, ui: &mut egui::Ui, field: &AttrField) {
        if let Some(index) = self.index(field) {
            let rect = ui.max_rect();
            if let Some(command) = animation_controls(
                ui,
                self.id,
                &self.attrs[self.indices[index]],
                self.frame,
                rect,
                self.ui_state.attribute_metrics,
            ) {
                self.commands.push(command);
            }
        }
    }
    fn actions(&mut self, ui: &mut egui::Ui, field: &AttrField) {
        let Some(index) = self.index(field) else {
            return;
        };
        let attr = &self.attrs[self.indices[index]];
        let rect = ui.max_rect();
        let left = Rect::from_min_size(
            rect.min,
            Vec2::new((rect.width() * 0.5 - 2.0).max(0.0), rect.height()),
        );
        let right = Rect::from_min_max(Pos2::new(rect.center().x + 2.0, rect.top()), rect.max);
        if attr.path == "/julia" {
            let enabled = !self.values[index].is_null();
            if icon_cell(ui, left, ph::POWER, enabled, true)
                .on_hover_text("Enable / disable Julia")
                .clicked()
            {
                self.commands.push(WorldCommand::SetAttribute {
                    id: self.id,
                    path: attr.path.clone(),
                    value: if enabled {
                        Value::Null
                    } else {
                        json!([0.0, 0.0, 0.0])
                    },
                    frame: self.frame,
                });
            }
        } else if is_color(&attr.path, &self.values[index]) {
            let mut rgb = std::array::from_fn::<_, 3, _>(|i| {
                self.values[index][i].as_f64().unwrap_or_default() as f32
            });
            let changed = cell_ui(ui, left, (&field.key, "color"), |ui| {
                ui.spacing_mut().interact_size =
                    Vec2::splat(self.ui_state.attribute_metrics.icon_side);
                ui.color_edit_button_rgb(&mut rgb).changed()
            })
            .inner;
            if changed {
                self.value_commands.push(WorldCommand::SetAttribute {
                    id: self.id,
                    path: attr.path.clone(),
                    value: json!(rgb),
                    frame: self.frame,
                });
            }
        }
        if attr.component.is_none() && attr.value.is_array() {
            let open = self
                .ui_state
                .components_open(self.id, &attr.path, self.attrs);
            if icon_cell(
                ui,
                right,
                if open {
                    ph::CARET_DOWN
                } else {
                    ph::CARET_RIGHT
                },
                false,
                true,
            )
            .on_hover_text("Show component channels")
            .clicked()
            {
                self.ui_state.toggle_components(self.id, &attr.path, open);
            }
        }
    }
    fn editor(
        &mut self,
        ui: &mut egui::Ui,
        field: &mut AttrField,
        _mixed: bool,
        _layout: &ValueEditorLayout,
    ) -> Option<bool> {
        record_grid_rect(&field.key, ui.max_rect());
        let index = self.index(field)?;
        let attr = &self.attrs[self.indices[index]];
        if !attr.choices.is_empty()
            || attr.path == "/material_id"
            || attr.path == "/environment/path"
            || attr.path.starts_with("/custom/")
            || attr.range.is_some()
            || matches!(field.value, GridValue::Label(_))
        {
            let changed = self.ui_state.edit_attribute(
                ui,
                self.id,
                &mut self.values[index],
                attr,
                self.materials,
            );
            if changed {
                let command = WorldCommand::SetAttribute {
                    id: self.id,
                    path: attr.path.clone(),
                    value: self.values[index].clone(),
                    frame: self.frame,
                };
                if value_gesture_eligible(attr, &self.values[index]) {
                    self.value_commands.push(command);
                } else {
                    self.commands.push(command);
                }
            }
            // Host-special values already emitted a typed command; do not also emit the
            // grid's display-only Label/string representation as a document edit.
            return Some(false);
        }
        None
    }
}
fn same_value_shape(before: &Value, after: &Value) -> bool {
    match (before, after) {
        (Value::Null, Value::Null)
        | (Value::Bool(_), Value::Bool(_))
        | (Value::String(_), Value::String(_))
        | (Value::Number(_), Value::Number(_)) => true,
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| same_value_shape(a, b))
        }
        (Value::Object(a), Value::Object(b)) => {
            a.len() == b.len()
                && a.iter()
                    .all(|(key, a)| b.get(key).is_some_and(|b| same_value_shape(a, b)))
        }
        _ => false,
    }
}
fn is_color(path: &str, value: &Value) -> bool {
    value.as_array().is_some_and(|v| v.len() == 3)
        && (path.contains("color") || path.contains("colour"))
}
fn grid_value(value: &Value) -> GridValue {
    match value {
        Value::Bool(v) => GridValue::Bool(*v),
        Value::String(v) => GridValue::Str(v.clone()),
        Value::Number(v) if v.is_u64() => v
            .as_u64()
            .and_then(|v| u32::try_from(v).ok())
            .map(GridValue::UInt)
            .unwrap_or_else(|| GridValue::Label(v.to_string())),
        Value::Number(v) if v.is_i64() => GridValue::Int64(v.as_i64().unwrap_or_default()),
        Value::Number(v) => GridValue::Float(v.as_f64().unwrap_or_default() as f32),
        Value::Array(v) if v.len() == 3 && v.iter().all(Value::is_number) => {
            GridValue::Vec3(std::array::from_fn(|i| {
                v[i].as_f64().unwrap_or_default() as f32
            }))
        }
        Value::Array(v) if v.len() == 4 && v.iter().all(Value::is_number) => {
            GridValue::Vec4(std::array::from_fn(|i| {
                v[i].as_f64().unwrap_or_default() as f32
            }))
        }
        _ => GridValue::Label(value.to_string()),
    }
}
fn grid_json(value: GridValue, template: &Value) -> Value {
    match value {
        GridValue::Bool(v) => json!(v),
        GridValue::Str(v) => json!(v),
        GridValue::Int8(v) => numeric_value(template, f64::from(v)),
        GridValue::Int(v) => numeric_value(template, f64::from(v)),
        GridValue::Int64(v) => json!(v),
        GridValue::UInt(v) => numeric_value(template, f64::from(v)),
        GridValue::Float(v) => numeric_value(template, f64::from(v)),
        GridValue::Vec3(v) => json!(v),
        GridValue::Vec4(v) => json!(v),
        GridValue::List(v) => Value::Array(
            v.into_iter()
                .enumerate()
                .map(|(i, value)| grid_json(value, &template[i]))
                .collect(),
        ),
        _ => template.clone(),
    }
}

const ATTRIBUTE_GROUPS: [&str; 9] = [
    "Transform",
    "Fractal",
    "Camera",
    "Light",
    "Environment",
    "Material",
    "Render",
    "Color",
    "Custom",
];
fn property_rects(
    row: Rect,
    label_width: f32,
    metrics: AttributeMetrics,
) -> egui_widgets_config::attr_layout::AttrRowRects {
    metrics.grid_config().row_rects(row, label_width)
}
fn transport_icon(
    ui: &mut egui::Ui,
    glyph: &str,
    active: bool,
    metrics: AttributeMetrics,
) -> egui::Response {
    let (rect, _) = ui.allocate_exact_size(
        Vec2::new(metrics.icon_side, metrics.row_height()),
        Sense::hover(),
    );
    icon_cell(ui, rect, glyph, active, true)
}
fn component_label(index: usize) -> &'static str {
    ["X", "Y", "Z", "W"].get(index).copied().unwrap_or("")
}
fn property_label(path: &str) -> String {
    match path {
        "/material_id" => return "Material".into(),
        "/transform/position" => return "Translate".into(),
        "/transform/rotation" | "/transform/rotation_degrees" => return "Rotate".into(),
        "/transform/scale" => return "Scale".into(),
        _ => {}
    }
    let parts: Vec<_> = path.trim_start_matches('/').split('/').collect();
    let parts = if parts.len() > 1 {
        &parts[1..]
    } else {
        &parts[..]
    };
    parts
        .iter()
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
fn icon_cell(
    ui: &mut egui::Ui,
    rect: Rect,
    icon: &str,
    active: bool,
    enabled: bool,
) -> egui::Response {
    let side = ui.spacing().icon_width.min(rect.width()).min(rect.height());
    let rect = Rect::from_center_size(rect.center(), Vec2::splat(side));
    let mut font = egui::TextStyle::Button.resolve(ui.style());
    font.size = side * 0.8;
    let color = active.then(|| ui.visuals().selection.stroke.color);
    square_icon_with_paint(ui, rect, icon, enabled, Some(font), color)
}
fn animation_controls(
    ui: &mut egui::Ui,
    id: NodeId,
    attr: &WorldAttribute,
    frame: f64,
    rect: Rect,
    metrics: AttributeMetrics,
) -> Option<WorldCommand> {
    if !attr.keyable {
        return None;
    }
    let width = metrics
        .icon_side
        .min(((rect.width() - metrics.component_gap) / 2.0).max(0.0));
    let rect = Rect::from_center_size(
        rect.center(),
        Vec2::new(rect.width(), metrics.icon_side.min(rect.height())),
    );
    let timer = Rect::from_min_size(rect.min, Vec2::new(width, rect.height()));
    let diamond = Rect::from_min_size(
        Pos2::new(timer.right() + metrics.component_gap, rect.top()),
        Vec2::new(width, rect.height()),
    );
    if icon_cell(ui, timer, ph::TIMER, !attr.frames.is_empty(), true)
        .on_hover_text("Enable or disable animation")
        .clicked()
    {
        return Some(WorldCommand::SetAnimation {
            id,
            path: attr.path.clone(),
            enabled: attr.frames.is_empty(),
            frame,
        });
    }
    let keyed = attr.frames.contains(&frame);
    if icon_cell(ui, diamond, ph::DIAMOND, keyed, true)
        .on_hover_text("Add / remove a key at the current frame")
        .clicked()
    {
        return Some(if keyed {
            WorldCommand::RemoveKey {
                id,
                path: attr.path.clone(),
                frame,
            }
        } else {
            WorldCommand::Key {
                id,
                path: attr.path.clone(),
                frame,
            }
        });
    }
    None
}

fn numeric_value(template: &Value, n: f64) -> Value {
    if template.is_u64() {
        json!(n.round().max(0.0) as u64)
    } else if template.is_i64() {
        json!(n.round() as i64)
    } else {
        json!(n)
    }
}
fn choice_label(path: &str, value: &Value) -> String {
    if path == "/formula" {
        return serde_json::from_value::<crate::scene::Formula>(value.clone())
            .map(|formula| formula.name().to_owned())
            .unwrap_or_else(|_| {
                value
                    .as_object()
                    .and_then(|v| v.keys().next())
                    .cloned()
                    .unwrap_or_else(|| "Formula".into())
            });
    }
    if path == "/julia" {
        return if value.is_null() {
            "Disabled"
        } else {
            "Enabled"
        }
        .into();
    }
    value
        .as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| value.to_string())
}
fn choice_selected(path: &str, value: &Value, choice: &Value) -> bool {
    if path == "/formula" {
        return value.as_object().and_then(|v| v.keys().next())
            == choice.as_object().and_then(|v| v.keys().next());
    }
    if path == "/julia" {
        return value.is_null() == choice.is_null();
    }
    value == choice
}
fn apply_choice(path: &str, value: &mut Value, choice: &Value) {
    if !choice_selected(path, value, choice) {
        *value = choice.clone();
    }
}
fn schema_editor(
    ui: &mut egui::Ui,
    value: &mut Value,
    attr: &WorldAttribute,
    layout: ValueEditorLayout,
) -> bool {
    if attr.path == "/julia" {
        return value_editor(ui, value, &attr.path, layout);
    }
    if !attr.choices.is_empty() {
        let mut changed = false;
        egui::ComboBox::from_id_salt(&attr.path)
            .selected_text(choice_label(&attr.path, value))
            .show_ui(ui, |ui| {
                for choice in &attr.choices {
                    if ui
                        .selectable_label(
                            choice_selected(&attr.path, value, choice),
                            choice_label(&attr.path, choice),
                        )
                        .clicked()
                    {
                        if !choice_selected(&attr.path, value, choice) {
                            apply_choice(&attr.path, value, choice);
                            changed = true;
                        }
                    }
                }
            });
        if attr.path == "/julia" && value.is_array() {
            value_editor(ui, value, &attr.path, layout);
        }
        return changed;
    }
    if let Some(range) = attr.range.filter(|_| value.is_number()) {
        let area = Rect::from_min_size(
            ui.cursor().min,
            Vec2::new(ui.available_width(), ui.spacing().interact_size.y),
        );
        if let Some(rect) = layout.numeric_cells_iter(area, 1).next() {
            return numeric_editor(ui, value, &attr.path, rect, None, Some(range));
        }
        return false;
    }
    value_editor(ui, value, &attr.path, layout)
}
fn value_editor(
    ui: &mut egui::Ui,
    value: &mut Value,
    path: &str,
    layout: ValueEditorLayout,
) -> bool {
    let height = layout.row_height.unwrap_or(ui.spacing().interact_size.y);
    let area = Rect::from_center_size(
        Pos2::new(
            ui.cursor().left() + ui.available_width() * 0.5,
            ui.max_rect().center().y,
        ),
        Vec2::new(ui.available_width(), height.min(ui.max_rect().height())),
    );
    match value {
        Value::Bool(v) => {
            let Some(rect) = layout.numeric_cells_iter(area, 1).next() else {
                return false;
            };
            cell_ui(ui, rect, (path, "bool"), |ui| {
                let response = ui.checkbox(v, "");
                record_control(path, "bool", response.rect, area);
                response.changed()
            })
            .inner
        }
        Value::Number(_) => {
            let Some(rect) = layout.numeric_cells_iter(area, 1).next() else {
                return false;
            };
            numeric_editor(ui, value, path, rect, None, None)
        }
        Value::String(v) => {
            cell_ui(ui, area, (path, "text"), |ui| {
                ui.add_sized(
                    area.size(),
                    egui::TextEdit::singleline(v)
                        .desired_width(area.width())
                        .margin(ui.spacing().button_padding),
                )
                .changed()
            })
            .inner
        }
        Value::Array(a) if a.iter().all(Value::is_number) => {
            let mut changed = false;
            let count = a.len();
            for (i, (value, rect)) in a
                .iter_mut()
                .zip(layout.numeric_cells_iter(area, count))
                .enumerate()
            {
                changed |= numeric_editor(ui, value, path, rect, Some(i), None);
            }
            changed
        }
        _ => {
            ui.add(egui::Label::new(choice_label(path, value)).truncate());
            false
        }
    }
}
fn numeric_editor(
    ui: &mut egui::Ui,
    value: &mut Value,
    path: &str,
    rect: Rect,
    component: Option<usize>,
    range: Option<(f64, f64)>,
) -> bool {
    let response = cell_ui(ui, rect, (path, component), |ui| {
        ui.spacing_mut().interact_size.x = rect.width();
        let axis = component.map(component_label).unwrap_or("");
        if value.is_u64() {
            let mut number = value.as_u64().unwrap_or_default();
            let response = ui.add_sized(
                rect.size(),
                egui::DragValue::new(&mut number)
                    .speed(1.0)
                    .prefix(axis)
                    .range(
                        range
                            .map(|(a, b)| a..=b)
                            .unwrap_or(f64::NEG_INFINITY..=f64::INFINITY),
                    ),
            );
            if response.changed() {
                *value = json!(number);
            }
            response
        } else if value.is_i64() {
            let mut number = value.as_i64().unwrap_or_default();
            let response = ui.add_sized(
                rect.size(),
                egui::DragValue::new(&mut number)
                    .speed(1.0)
                    .prefix(axis)
                    .range(
                        range
                            .map(|(a, b)| a..=b)
                            .unwrap_or(f64::NEG_INFINITY..=f64::INFINITY),
                    ),
            );
            if response.changed() {
                *value = json!(number);
            }
            response
        } else {
            let mut number = value.as_f64().unwrap_or_default();
            let response = ui.add_sized(
                rect.size(),
                egui::DragValue::new(&mut number)
                    .speed(if path.contains("rotation") { 0.5 } else { 0.02 })
                    .range(
                        range
                            .map(|(a, b)| a..=b)
                            .unwrap_or(f64::NEG_INFINITY..=f64::INFINITY),
                    )
                    .prefix(axis),
            );
            if response.changed() {
                *value = json!(number);
            }
            response
        }
    })
    .inner;
    #[cfg(test)]
    NUMERIC_RECT_TRACES.with(|trace| {
        trace
            .borrow_mut()
            .push((response.rect, response.interact_rect, rect))
    });
    response.clone().on_hover_ui(|ui| {
        ui.label(value.to_string());
    });
    record_control(
        path,
        component.map(component_label).unwrap_or("scalar"),
        response.rect,
        rect,
    );
    response.changed()
}
#[cfg(test)]
thread_local! { static NUMERIC_RECT_TRACES: std::cell::RefCell<Vec<(Rect,Rect,Rect)>> = const { std::cell::RefCell::new(Vec::new()) }; }
#[cfg(test)]
thread_local! { static GRID_RECT_TRACES: std::cell::RefCell<HashMap<String,Rect>> = std::cell::RefCell::new(HashMap::new()); }
fn record_grid_rect(_path: &str, _rect: Rect) {
    #[cfg(test)]
    GRID_RECT_TRACES.with(|trace| {
        trace.borrow_mut().insert(_path.into(), _rect);
    });
}
#[cfg(test)]
#[derive(Clone, Debug)]
struct ControlTrace {
    path: String,
    kind: String,
    response: Rect,
    area: Rect,
}
#[cfg(test)]
thread_local! { static CONTROL_TRACES: std::cell::RefCell<Vec<ControlTrace>> = const { std::cell::RefCell::new(Vec::new()) }; }
fn record_control(_path: &str, _kind: &str, _response: Rect, _area: Rect) {
    #[cfg(test)]
    CONTROL_TRACES.with(|trace| {
        trace.borrow_mut().push(ControlTrace {
            path: _path.into(),
            kind: _kind.into(),
            response: _response,
            area: _area,
        })
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::Scene;
    use crate::world::WorldDocument;
    fn editor() -> WorldEditor {
        WorldEditor::new(WorldDocument::from_scene(&Scene::preset(
            crate::params::FAMILY_BULB,
        )))
    }
    #[test]
    fn resetting_document_ui_preserves_appearance_preferences() {
        let metrics = AttributeMetrics {
            field_height: 22.0,
            numeric_width: 74.0,
            ..Default::default()
        };
        let mut state = WorldUi {
            attribute_metrics: metrics,
            attribute_label_width: 237.0,
            playhead: 90,
            playing: true,
            ..Default::default()
        };
        state.reset();
        assert_eq!(state.attribute_metrics, metrics);
        assert_eq!(state.attribute_label_width, 237.0);
        assert_eq!(state.playhead, 0);
        assert!(!state.playing);
    }
    #[test]
    fn key_addresses_keep_uuid_and_fractional_time_after_reordering() {
        let e = editor();
        let mut nodes = e.document.nodes();
        assert!(nodes.len() > 1);
        nodes.swap(0, 1);
        let expected = nodes[0].id;
        let lanes = nodes
            .iter()
            .map(|_| {
                vec![Lane {
                    label: "X".into(),
                    path: Some("/transform/position/0".into()),
                    value: json!(0),
                    frames: vec![12.25],
                    depth: 3,
                    group: false,
                    attr: None,
                }]
            })
            .collect::<Vec<_>>();
        let key = WorldUi::key_identity(
            &nodes,
            &lanes,
            KeyPos {
                track: 0,
                lane: 0,
                frame: 12.25,
            },
        )
        .unwrap();
        assert_eq!(key.node, expected);
        assert_eq!(key.path, "/transform/position/0");
        assert_eq!(f64::from_bits(key.frame), 12.25);
    }
    #[test]
    fn outliner_reparent_respects_sibling_index_and_undo() {
        let mut e = editor();
        let nodes = e.document.nodes();
        let original = nodes.iter().map(|n| n.id).collect::<Vec<_>>();
        let moved = nodes
            .iter()
            .find(|n| n.kind == WorldKind::Fractal)
            .unwrap()
            .id;
        let parent = nodes
            .iter()
            .find(|n| n.kind == WorldKind::Group)
            .unwrap()
            .id;
        let ids = reparent_order(&nodes, moved, Some(parent), 0);
        e.execute(WorldCommand::Batch(vec![
            WorldCommand::Reparent {
                id: moved,
                parent: Some(parent),
            },
            WorldCommand::Reorder { ids },
        ]))
        .unwrap();
        assert_eq!(e.document.info(moved).unwrap().parent, Some(parent));
        let ordered = e
            .document
            .nodes()
            .into_iter()
            .map(|n| n.id)
            .collect::<Vec<_>>();
        assert_eq!(
            ordered.iter().position(|id| *id == moved).unwrap(),
            ordered.iter().position(|id| *id == parent).unwrap() + 1
        );
        assert!(e.undo());
        assert_eq!(e.document.info(moved).unwrap().parent, None);
        assert_eq!(
            e.document
                .nodes()
                .into_iter()
                .map(|n| n.id)
                .collect::<Vec<_>>(),
            original
        );
        let reordered = reparent_order(&nodes, nodes[0].id, None, 2);
        assert_eq!(reordered[2], nodes[0].id);
    }
    #[test]
    fn timeline_reorder_is_document_owned_and_undoable() {
        let mut e = editor();
        let nodes = e.document.nodes();
        let original = nodes.iter().map(|n| n.id).collect::<Vec<_>>();
        let mut state = WorldUi::default();
        state.timeline_action(
            &mut e,
            &nodes,
            &[],
            TimelineAction::MoveClip {
                id: wid(nodes[0].id),
                new_start: nodes[0].start as i64,
                new_track: 1,
            },
        );
        let ordered = e
            .document
            .nodes()
            .into_iter()
            .map(|n| n.id)
            .collect::<Vec<_>>();
        assert_eq!(ordered[0], original[1]);
        assert_eq!(ordered[1], original[0]);
        assert!(e.undo());
        assert_eq!(
            e.document
                .nodes()
                .into_iter()
                .map(|n| n.id)
                .collect::<Vec<_>>(),
            original
        );
    }
    #[test]
    fn structural_choices_preserve_custom_parameters_and_integer_fields() {
        let mut formula =
            serde_json::to_value(crate::scene::Scene::preset(crate::params::FAMILY_BULB).formula)
                .unwrap();
        let default = formula.clone();
        let key = formula.as_object().unwrap().keys().next().unwrap().clone();
        formula[&key]["power"] = json!(9.5);
        let customized = formula.clone();
        assert_eq!(choice_label("/formula", &formula), "Mandelbulb");
        assert!(choice_selected("/formula", &formula, &default));
        apply_choice("/formula", &mut formula, &default);
        assert_eq!(formula, customized);
        let next =
            serde_json::to_value(crate::scene::Scene::preset(crate::params::FAMILY_BOX).formula)
                .unwrap();
        apply_choice("/formula", &mut formula, &next);
        assert_eq!(choice_label("/formula", &formula), "Mandelbox");

        let mut julia = Value::Null;
        apply_choice("/julia", &mut julia, &json!([0.0, 0.0, 0.0]));
        julia[0] = json!(0.25);
        apply_choice("/julia", &mut julia, &json!([0.0, 0.0, 0.0]));
        assert_eq!(julia[0], json!(0.25));
        assert_eq!(choice_label("/julia", &julia), "Enabled");
        apply_choice("/julia", &mut julia, &Value::Null);
        assert_eq!(choice_label("/julia", &julia), "Disabled");
        assert_eq!(numeric_value(&json!(12u32), 13.75), json!(14u32));
    }
    #[test]
    fn numeric_edit_preserves_scene_integer_types() {
        assert_eq!(numeric_value(&json!(12u32), 13.75), json!(14u32));
        assert_eq!(numeric_value(&json!(-12i32), -13.75), json!(-14i32));
        assert_eq!(numeric_value(&json!(12.5), 13.75), json!(13.75));
    }
    #[test]
    fn transport_wraps_within_nonzero_work_area() {
        let mut e = editor();
        e.document.first = 10;
        e.document.last = 19;
        e.document.fps = 10.0;
        let mut state = WorldUi::default();
        state.seek(19);
        state.playing = true;
        assert!(state.advance(0.2, &e));
        assert_eq!(state.playhead, 11);
        state.looping = false;
        state.seek(19);
        state.advance(0.2, &e);
        assert_eq!(state.playhead, 19);
        assert!(!state.playing);
    }

    #[test]
    fn clean_metadata_tracks_undo_redo_while_dirty_drafts_survive_unrelated_edits() {
        let mut e = editor();
        let id = e.selection.unwrap();
        let mut state = WorldUi::default();
        e.execute(WorldCommand::SetMetadata {
            id,
            path: "".into(),
            value: json!({"label":"Before"}),
        })
        .unwrap();
        state.refresh_metadata(&e, id);
        assert_eq!(
            serde_json::from_str::<Value>(&state.metadata).unwrap(),
            json!({"label":"Before"})
        );
        state.metadata = json!({"label":"Applied"}).to_string();
        state.metadata_dirty = true;
        assert!(state.apply_metadata(&mut e, id));
        assert!(!state.metadata_dirty);
        assert!(e.undo());
        state.refresh_metadata(&e, id);
        assert_eq!(
            serde_json::from_str::<Value>(&state.metadata).unwrap(),
            json!({"label":"Before"})
        );
        assert!(e.redo());
        state.refresh_metadata(&e, id);
        assert_eq!(
            serde_json::from_str::<Value>(&state.metadata).unwrap(),
            json!({"label":"Applied"})
        );
        state.metadata = "{\"unsaved\":99}".into();
        state.metadata_dirty = true;
        e.execute(WorldCommand::Rename {
            id,
            name: "Unrelated edit".into(),
        })
        .unwrap();
        state.refresh_metadata(&e, id);
        assert_eq!(state.metadata, "{\"unsaved\":99}");
        assert!(state.metadata_dirty);
        state.metadata = "invalid JSON".into();
        assert!(!state.apply_metadata(&mut e, id));
        assert!(state.metadata_dirty);
        assert_eq!(e.document.metadata(id).unwrap(), json!({"label":"Applied"}));
        let other = e.document.nodes().iter().find(|n| n.id != id).unwrap().id;
        state.refresh_metadata(&e, other);
        assert!(!state.metadata_dirty);
        assert_ne!(state.metadata, "invalid JSON");
    }

    #[test]
    fn outliner_cache_keeps_tree_and_updates_selection_and_revision() {
        let mut e = editor();
        let id = e.selection.unwrap();
        let mut state = WorldUi::default();
        let cache = state.take_outliner_cache(&e);
        let roots = cache.model.roots.as_ptr();
        let nodes = cache.nodes.as_ptr();
        let map_capacity = cache.map.capacity();
        state.outliner_cache = cache;
        e.selected.clear();
        e.selection = None;
        let cache = state.take_outliner_cache(&e);
        assert_eq!(cache.model.roots.as_ptr(), roots);
        assert_eq!(cache.nodes.as_ptr(), nodes);
        assert_eq!(cache.map.capacity(), map_capacity);
        assert!(cache.model.selection.is_empty());
        state.outliner_cache = cache;
        e.selected.push(id);
        e.selection = Some(id);
        let cache = state.take_outliner_cache(&e);
        assert_eq!(cache.model.selection, vec![wid(id)]);
        assert_eq!(cache.model.roots.as_ptr(), roots);
        state.outliner_cache = cache;
        e.execute(WorldCommand::Rename {
            id,
            name: "Outliner changed".into(),
        })
        .unwrap();
        let cache = state.take_outliner_cache(&e);
        assert_eq!(
            cache.nodes.iter().find(|n| n.id == id).unwrap().name,
            "Outliner changed"
        );
        assert_eq!(cache.map.get(&wid(id)), Some(&id));
        state.outliner_cache = cache;
        assert!(e.undo());
        let cache = state.take_outliner_cache(&e);
        assert_ne!(
            cache.nodes.iter().find(|n| n.id == id).unwrap().name,
            "Outliner changed"
        );
    }

    #[test]
    fn idle_cache_keeps_descriptor_and_timeline_storage() {
        let mut e = editor();
        let mut state = WorldUi::default();
        let id = e.selection.unwrap();
        let cache = state.take_cache(&e);
        let node_storage = cache.nodes.as_ptr();
        let attr_storage = cache.attributes[&id].as_ptr();
        state.cache = cache;
        let cache = state.take_cache(&e);
        assert_eq!(cache.nodes.as_ptr(), node_storage);
        assert_eq!(cache.attributes[&id].as_ptr(), attr_storage);
        state.cache = cache;
        state.expanded.insert(id);
        state.groups.insert((id, "@Transform".into()));
        let ctx = egui::Context::default();
        let _ = ctx.run_ui(Default::default(), |ui| {
            egui::CentralPanel::default().show(ui, |ui| {
                state.timeline(ui, &mut e);
            });
        });
        let tracks = state.timeline_cache.model.tracks.as_ptr();
        let lanes = state.timeline_cache.lanes.as_ptr();
        let _ = ctx.run_ui(Default::default(), |ui| {
            egui::CentralPanel::default().show(ui, |ui| {
                state.timeline(ui, &mut e);
            });
        });
        assert_eq!(state.timeline_cache.model.tracks.as_ptr(), tracks);
        assert_eq!(state.timeline_cache.lanes.as_ptr(), lanes);
        e.execute(WorldCommand::Rename {
            id,
            name: "Changed".into(),
        })
        .unwrap();
        let cache = state.take_cache(&e);
        assert_eq!(
            cache.nodes.iter().find(|n| n.id == id).unwrap().name,
            "Changed"
        );
    }

    #[test]
    fn frame_cache_updates_values_and_flags_without_rebuilding_stable_schema() {
        let mut e = editor();
        let id = e.selection.unwrap();
        for (path, value) in [
            ("/transform/position/0", json!(10.0)),
            ("/visible", json!(false)),
            ("/julia", json!([0.2, 0.3, 0.4])),
        ] {
            e.execute(WorldCommand::SetAnimation {
                id,
                path: path.into(),
                enabled: true,
                frame: 0.0,
            })
            .unwrap();
            e.execute(WorldCommand::SetAttribute {
                id,
                path: path.into(),
                value,
                frame: 10.0,
            })
            .unwrap();
        }
        let mut state = WorldUi::default();
        let cache = state.take_cache(&e);
        let ptr = cache.attributes[&id].as_ptr();
        assert!(!cache.attributes[&id].iter().any(|a| a.path == "/julia/0"));
        state.cache = cache;
        state.playhead = 5;
        let cache = state.take_cache(&e);
        assert_eq!(cache.attributes[&id].as_ptr(), ptr);
        assert_eq!(
            cache.attributes[&id]
                .iter()
                .find(|a| a.path == "/transform/position/0")
                .unwrap()
                .value,
            e.document
                .attribute_value(id, "/transform/position/0", 5.0)
                .unwrap()
        );
        assert!(cache.nodes.iter().find(|n| n.id == id).unwrap().visible);
        state.cache = cache;
        state.playhead = 10;
        let cache = state.take_cache(&e);
        assert!(!cache.nodes.iter().find(|n| n.id == id).unwrap().visible);
        assert!(cache.attributes[&id].iter().any(|a| a.path == "/julia/0"));
        assert_eq!(
            cache.attributes[&id]
                .iter()
                .find(|a| a.path == "/julia/0")
                .unwrap()
                .value,
            json!(0.2)
        );
    }

    #[test]
    fn layer_drop_is_one_undo_and_preserves_spans_keys_and_selection() {
        let mut e = editor();
        let id = e.selection.unwrap();
        e.execute(WorldCommand::Key {
            id,
            path: "/transform/position/0".into(),
            frame: 12.25,
        })
        .unwrap();
        e.execute(WorldCommand::SetSpan {
            id,
            start: 7.0,
            end: 42.0,
        })
        .unwrap();
        let nodes = e.document.nodes();
        let order = nodes.iter().map(|n| n.id).collect::<Vec<_>>();
        let target = *order
            .iter()
            .rev()
            .find(|&&candidate| candidate != id)
            .unwrap();
        let before = e.document.attributes(id, 12.25).unwrap();
        let selection = e.selected.clone();
        let reordered = layer_drop_order(&nodes, &order, id, target, true);
        e.execute(WorldCommand::Reorder {
            ids: reordered.clone(),
        })
        .unwrap();
        assert_eq!(
            e.document.nodes().iter().map(|n| n.id).collect::<Vec<_>>(),
            reordered
        );
        let info = e.document.info(id).unwrap();
        assert_eq!((info.start, info.end), (7.0, 42.0));
        let after = e.document.attributes(id, 12.25).unwrap();
        for (a, b) in before.iter().zip(&after) {
            assert_eq!(
                (&a.path, &a.value, &a.frames),
                (&b.path, &b.value, &b.frames)
            );
        }
        assert_eq!(e.selected, selection);
        assert!(e.undo());
        assert_eq!(
            e.document.nodes().iter().map(|n| n.id).collect::<Vec<_>>(),
            order
        );
        assert_eq!(e.document.info(id).unwrap().start, 7.0);
    }

    #[test]
    fn numeric_cells_keep_compact_and_large_hit_rectangles_with_long_negative_values() {
        for metrics in [
            AttributeMetrics::default(),
            AttributeMetrics {
                field_height: 24.0,
                numeric_width: 84.0,
                icon_side: 18.0,
                row_gap: 6.0,
                component_gap: 5.0,
            },
        ] {
            let ctx = egui::Context::default();
            let mut values = json!([-123456789.1234567, -0.000000000123, 987654321.0]);
            let expected = values.clone();
            NUMERIC_RECT_TRACES.with(|trace| trace.borrow_mut().clear());
            let mut output = ctx.run_ui(Default::default(), |ui| {
                egui::CentralPanel::default().show(ui, |ui| {
                    metrics.apply(ui);
                    let row = Rect::from_min_size(
                        Pos2::new(30.0, 50.0),
                        Vec2::new(400.0, metrics.row_height()),
                    );
                    let layout = metrics.value_layout();
                    let cells = layout.numeric_cells_iter(row, 3);
                    for (index, rect) in cells.enumerate() {
                        numeric_editor(
                            ui,
                            &mut values[index],
                            "/long/vector",
                            rect,
                            Some(index),
                            None,
                        );
                    }
                });
            });
            output.textures_delta.clear();
            assert_eq!(
                values, expected,
                "Drawing must preserve full numeric precision"
            );
            NUMERIC_RECT_TRACES.with(|trace| {
                let trace = trace.borrow();
                assert_eq!(trace.len(), 3);
                for (response, interaction, cell) in trace.iter() {
                    assert!(
                        response.height() <= metrics.field_height + 0.6,
                        "{metrics:?}: {response:?}"
                    );
                    assert!(
                        interaction.width() <= metrics.numeric_width + 0.1,
                        "{metrics:?}: {interaction:?}"
                    );
                    assert!(
                        cell.contains_rect(*interaction),
                        "{interaction:?} escapes {cell:?}"
                    );
                    assert!((response.center().y - row_center(metrics)).abs() < 0.6);
                }
            });
        }
        fn row_center(metrics: AttributeMetrics) -> f32 {
            50.0 + metrics.row_height() * 0.5
        }
    }

    #[test]
    fn lane_controls_share_column_origins_and_row_centers_at_mixed_depths() {
        for width in [430.0, 300.0] {
            CONTROL_TRACES.with(|trace| trace.borrow_mut().clear());
            let mut e = editor();
            let id = e.selection.unwrap();
            let attrs = e.document.attributes(id, 0.0).unwrap();
            let nodes = e.document.nodes();
            let mut template = attrs
                .iter()
                .find(|a| a.path == "/transform/position")
                .unwrap()
                .clone();
            let cases = [
                ("/test/scalar", json!(0.5), 2, true),
                ("/test/vector", json!([0.0, 1.0, 2.0]), 2, true),
                ("/test/bool", json!(true), 3, true),
                ("/test/static", json!(1.0), 4, false),
            ];
            let mut state = WorldUi::default();
            let ctx = egui::Context::default();
            let _ = ctx.run_ui(Default::default(), |ui| {
                egui::CentralPanel::default().show(ui, |ui| {
                    for (i, (path, value, depth, keyable)) in cases.iter().enumerate() {
                        template.path = (*path).into();
                        template.value = value.clone();
                        template.component = None;
                        template.choices.clear();
                        template.range = None;
                        template.keyable = *keyable;
                        let mut lane = Lane {
                            label: "Long property label".into(),
                            path: Some((*path).into()),
                            value: value.clone(),
                            frames: vec![],
                            depth: *depth,
                            group: false,
                            attr: Some(template.clone()),
                        };
                        let row = Rect::from_min_size(
                            Pos2::new(20.0, 50.0 + i as f32 * state.attribute_metrics.row_height()),
                            Vec2::new(width, state.attribute_metrics.row_height()),
                        );
                        state.lane_row(ui, &mut e, id, &mut lane, row, &attrs, &nodes);
                    }
                });
            });
            CONTROL_TRACES.with(|trace| {
                let trace = trace.borrow();
                let first = |path: &str| trace.iter().find(|t| t.path == path).unwrap();
                let scalar = first("/test/scalar");
                for (i, (path, _, _, _)) in cases.iter().enumerate() {
                    let actual = first(path);
                    assert!(
                        (actual.response.left() - scalar.response.left()).abs() < 0.1,
                        "width {width}: {actual:?} vs {scalar:?}"
                    );
                    assert!(
                        (actual.response.center().y
                            - (50.0
                                + state.attribute_metrics.row_height() * 0.5
                                + i as f32 * state.attribute_metrics.row_height()))
                        .abs()
                            < 0.6,
                        "{actual:?}"
                    );
                    assert!(actual.area.width() > 0.0);
                }
                let vector = trace
                    .iter()
                    .filter(|t| t.path == "/test/vector")
                    .collect::<Vec<_>>();
                assert_eq!(vector.len(), 3);
                assert_eq!(
                    vector.iter().map(|t| t.kind.as_str()).collect::<Vec<_>>(),
                    ["X", "Y", "Z"]
                );
                assert!(
                    vector
                        .windows(2)
                        .all(|pair| pair[0].response.left() < pair[1].response.left())
                );
            });
        }
    }

    #[test]
    fn actual_inspector_numeric_drag_is_one_undo_after_release() {
        let mut e = editor();
        let id = e.selection.unwrap();
        let mut state = WorldUi::default();
        let ctx = egui::Context::default();
        let initial = e
            .document
            .attribute_value(id, "/transform/position", 0.0)
            .unwrap();
        let render = |state: &mut WorldUi, e: &mut WorldEditor, events, time| {
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 4000.0))),
                    events,
                    time: Some(time),
                    ..Default::default()
                },
                |ui| {
                    egui::CentralPanel::default().show(ui, |ui| {
                        state.inspector(ui, e);
                    });
                },
            );
            output.textures_delta.clear();
            e.finish_edit_unless(parameter_gesture(&ctx));
            GRID_RECT_TRACES.with(|trace| trace.borrow()["/transform/position"])
        };
        let cell = render(&mut state, &mut e, vec![], 0.0);
        let at = state
            .attribute_metrics
            .value_layout()
            .numeric_cells_iter(cell, 3)
            .next()
            .unwrap()
            .center();
        let button = |pos, pressed| egui::Event::PointerButton {
            pos,
            pressed,
            button: egui::PointerButton::Primary,
            modifiers: egui::Modifiers::NONE,
        };
        render(
            &mut state,
            &mut e,
            vec![egui::Event::PointerMoved(at), button(at, true)],
            0.1,
        );
        render(
            &mut state,
            &mut e,
            vec![egui::Event::PointerMoved(at + Vec2::new(20.0, 0.0))],
            0.2,
        );
        assert!(e.active_edit().is_some());
        let first = e
            .document
            .attribute_value(id, "/transform/position", 0.0)
            .unwrap();
        assert_ne!(first, initial);
        let final_pos = at + Vec2::new(40.0, 0.0);
        render(
            &mut state,
            &mut e,
            vec![egui::Event::PointerMoved(final_pos)],
            0.3,
        );
        let last = e
            .document
            .attribute_value(id, "/transform/position", 0.0)
            .unwrap();
        assert_ne!(last, first);
        render(&mut state, &mut e, vec![button(final_pos, false)], 0.4);
        assert!(e.active_edit().is_none());
        assert!(e.undo());
        assert_eq!(
            e.document
                .attribute_value(id, "/transform/position", 0.0)
                .unwrap(),
            initial
        );
        assert!(
            !e.undo(),
            "one pointer gesture must contribute one history item"
        );
        assert!(e.redo());
        assert_eq!(
            e.document
                .attribute_value(id, "/transform/position", 0.0)
                .unwrap(),
            last
        );
    }

    #[test]
    fn inspector_splitter_drag_updates_every_section_and_persists_into_timeline() {
        let mut e = editor();
        let id = e.selection.unwrap();
        let mut state = WorldUi::default();
        let ctx = egui::Context::default();
        let render = |state: &mut WorldUi, e: &mut WorldEditor, events, time| {
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 4000.0))),
                    events,
                    time: Some(time),
                    ..Default::default()
                },
                |ui| {
                    egui::CentralPanel::default().show(ui, |ui| {
                        state.inspector(ui, e);
                    });
                },
            );
            output.textures_delta.clear();
            GRID_RECT_TRACES.with(|trace| trace.borrow()["/transform/position"])
        };
        let initial = render(&mut state, &mut e, vec![], 0.0);
        let at = Pos2::new(initial.left() - 6.0, initial.center().y);
        let button = |pos, pressed| egui::Event::PointerButton {
            pos,
            pressed,
            button: egui::PointerButton::Primary,
            modifiers: egui::Modifiers::NONE,
        };
        render(
            &mut state,
            &mut e,
            vec![egui::Event::PointerMoved(at), button(at, true)],
            0.1,
        );
        let moved = at + Vec2::new(40.0, 0.0);
        render(
            &mut state,
            &mut e,
            vec![egui::Event::PointerMoved(moved)],
            0.2,
        );
        render(&mut state, &mut e, vec![button(moved, false)], 0.3);
        let restored = render(&mut state, &mut e, vec![], 0.4);
        assert!(
            (state.attribute_label_width - 220.0).abs() < 0.1,
            "shared width {}",
            state.attribute_label_width
        );
        assert!((restored.left() - initial.left() - 40.0).abs() < 0.1);
        let sections = state
            .cache
            .sections
            .iter()
            .filter(|((node, _), _)| *node == id)
            .collect::<Vec<_>>();
        assert!(sections.len() > 1);
        for (_, section) in sections {
            assert!((section.state.table.widths[0] - 220.0).abs() < 0.1);
        }
        let attrs = e.document.attributes(id, 0.0).unwrap();
        let attr = attrs
            .iter()
            .find(|a| a.path == "/transform/position")
            .unwrap();
        let mut lane = Lane {
            label: "Translate".into(),
            path: Some(attr.path.clone()),
            value: attr.value.clone(),
            frames: attr.frames.clone(),
            depth: 2,
            group: false,
            attr: Some(attr.clone()),
        };
        CONTROL_TRACES.with(|trace| trace.borrow_mut().clear());
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 4000.0))),
                ..Default::default()
            },
            |ui| {
                egui::CentralPanel::default().show(ui, |ui| {
                    let row = Rect::from_min_size(
                        Pos2::new(initial.left() - 180.0 - 8.0, 320.0),
                        Vec2::new(700.0, state.attribute_metrics.row_height()),
                    );
                    state.lane_row(ui, &mut e, id, &mut lane, row, &attrs, &[]);
                });
            },
        );
        output.textures_delta.clear();
        CONTROL_TRACES.with(|trace| {
            assert!((trace.borrow()[0].response.left() - restored.left()).abs() < 0.1)
        });
    }

    #[test]
    fn grid_and_timeline_use_the_same_actual_editor_origins() {
        for width in [430.0, 300.0] {
            let mut e = editor();
            let id = e.selection.unwrap();
            let original = e.document.attributes(id, 0.0).unwrap();
            let base = original
                .iter()
                .find(|a| a.path == "/transform/position")
                .unwrap();
            let mut attrs = Vec::new();
            for (path, value, keyable) in [
                ("/custom/scalar", json!(0.5), true),
                ("/custom/vector", json!([0.0, 1.0, 2.0]), true),
                ("/custom/bool", json!(true), true),
                ("/custom/static", json!(1.0), false),
            ] {
                let mut attr = base.clone();
                attr.path = path.into();
                attr.value = value;
                attr.component = None;
                attr.choices.clear();
                attr.range = None;
                attr.keyable = keyable;
                attrs.push(attr);
            }
            let nodes = e.document.nodes();
            let mut state = WorldUi::default();
            let mut section = GridSectionCache::default();
            state.prepare_section(&mut section, &attrs, id, "Custom", 0, 0);
            state.attribute_label_width = 210.0;
            section.state.table.widths.resize(1, 0.0);
            section.state.table.widths[0] = state.attribute_label_width;
            let ctx = egui::Context::default();
            CONTROL_TRACES.with(|trace| trace.borrow_mut().clear());
            let _ = ctx.run_ui(Default::default(), |ui| {
                egui::CentralPanel::default().show(ui, |ui| {
                    let mut grid_ui =
                        ui.new_child(egui::UiBuilder::new().id_salt("geometry_grid").max_rect(
                            Rect::from_min_size(Pos2::new(20.0, 50.0), Vec2::new(width, 200.0)),
                        ));
                    state.attribute_metrics.apply(&mut grid_ui);
                    let config = state.attribute_metrics.grid_config();
                    let mut commands = Vec::new();
                    let mut value_commands = Vec::new();
                    let mut hooks = WorldGridHooks {
                        ui_state: &mut state,
                        id,
                        frame: 0.0,
                        attrs: &attrs,
                        indices: &section.indices,
                        labels: &section.labels,
                        values: &mut section.values,
                        materials: &nodes,
                        commands: &mut commands,
                        value_commands: &mut value_commands,
                    };
                    let changed = render_grid_with_config(
                        &mut grid_ui,
                        &mut section.fields,
                        &mut section.state,
                        &HashSet::new(),
                        &config,
                        &mut hooks,
                    );
                    assert!(changed.is_empty());
                    assert!(commands.is_empty());
                    assert!(value_commands.is_empty());
                });
            });
            let grid = CONTROL_TRACES.with(|trace| std::mem::take(&mut *trace.borrow_mut()));
            let _ = ctx.run_ui(Default::default(), |ui| {
                egui::CentralPanel::default().show(ui, |ui| {
                    for (i, attr) in attrs.iter().enumerate() {
                        let mut lane = Lane {
                            label: property_label(&attr.path),
                            path: Some(attr.path.clone()),
                            value: attr.value.clone(),
                            frames: vec![],
                            depth: 2 + i,
                            group: false,
                            attr: Some(attr.clone()),
                        };
                        state.lane_row(
                            ui,
                            &mut e,
                            id,
                            &mut lane,
                            Rect::from_min_size(
                                Pos2::new(
                                    20.0,
                                    320.0 + i as f32 * state.attribute_metrics.row_height(),
                                ),
                                Vec2::new(width, state.attribute_metrics.row_height()),
                            ),
                            &attrs,
                            &nodes,
                        );
                    }
                });
            });
            CONTROL_TRACES.with(|trace| {
                let timeline = trace.borrow();
                assert_eq!(grid.len(), 6);
                assert_eq!(timeline.len(), 6);
                for (a, b) in grid.iter().zip(timeline.iter()) {
                    assert_eq!((&a.path, &a.kind), (&b.path, &b.kind));
                    assert!(
                        (a.response.left() - b.response.left()).abs() < 0.1,
                        "width {width}: grid {a:?}, timeline {b:?}"
                    );
                    assert!(
                        (a.response.height() - b.response.height()).abs() < 0.1,
                        "grid {a:?}, timeline {b:?}"
                    );
                }
                let y0 = grid[0].response.center().y;
                for (i, attr) in attrs.iter().enumerate() {
                    let response = grid.iter().find(|t| t.path == attr.path).unwrap().response;
                    assert!(
                        (response.center().y
                            - (y0 + i as f32 * state.attribute_metrics.row_height()))
                        .abs()
                            < 0.6
                    );
                }
            });
        }
    }

    #[test]
    fn vector_components_default_collapsed_and_open_when_keyed() {
        let mut e = editor();
        let id = e.selection.unwrap();
        let mut state = WorldUi::default();
        state.groups.insert((id, "@Transform".into()));
        let attrs = e.document.attributes(id, 0.0).unwrap();
        assert!(!state.components_open(id, "/transform/position", &attrs));
        assert!(
            state
                .lanes(id, &attrs)
                .iter()
                .any(|l| l.path.as_deref() == Some("/transform/position"))
        );
        assert!(
            !state
                .lanes(id, &attrs)
                .iter()
                .any(|l| l.path.as_deref() == Some("/transform/position/0"))
        );
        e.execute(WorldCommand::Key {
            id,
            path: "/transform/position/0".into(),
            frame: 12.25,
        })
        .unwrap();
        let attrs = e.document.attributes(id, 0.0).unwrap();
        assert!(state.components_open(id, "/transform/position", &attrs));
        assert!(
            state
                .lanes(id, &attrs)
                .iter()
                .any(|l| l.path.as_deref() == Some("/transform/position/0"))
        );
        state.toggle_components(id, "/transform/position", true);
        assert!(!state.components_open(id, "/transform/position", &attrs));
    }

    #[test]
    fn panels_render_without_a_native_window() {
        let ctx = egui::Context::default();
        let mut e = editor();
        let mut state = WorldUi::default();
        let id = e.selection.unwrap();
        state.expanded.insert(id);
        state.groups.insert((id, "@Transform".into()));
        let render_node = e
            .document
            .nodes()
            .into_iter()
            .find(|n| n.kind == WorldKind::Group)
            .expect("World settings node");
        state.groups.insert((render_node.id, "@Render".into()));
        let render_lanes = state.lanes(
            render_node.id,
            &e.document.attributes(render_node.id, 0.0).unwrap(),
        );
        assert!(
            render_lanes
                .iter()
                .any(|lane| lane.path.as_deref() == Some("/render/iterations")),
            "Render iterations must remain projected after grouping"
        );
        let _ = ctx.run_ui(Default::default(), |ui| {
            egui::CentralPanel::default().show(ui, |ui| {
                state.timeline(ui, &mut e);
            });
        });
        let _ = ctx.run_ui(Default::default(), |ui| {
            egui::CentralPanel::default().show(ui, |ui| {
                state.outliner(ui, &mut e);
            });
        });
        let _ = ctx.run_ui(Default::default(), |ui| {
            egui::CentralPanel::default().show(ui, |ui| {
                state.inspector(ui, &mut e);
            });
        });
    }
}
