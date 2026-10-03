//! Object panels. Widgets report intents; WorldEditor owns all document changes.
use crate::world::{NodeId, WorldAttribute, WorldCommand, WorldEditor, WorldKind, WorldNodeInfo};
use curves::CurveKind;
use egui::{Color32, Pos2, Rect, Sense, Vec2};
use egui_outliner::{ContextItem, OutlinerAction, OutlinerConfig, OutlinerModel, TreeNode};
use egui_phosphor::regular as ph;
use egui_track_timeline::{
    Clip, KeyPos, Keyframe, PropLane, TimelineAction, TimelineConfig, TimelineModel, TimelineView,
    Track, TrackTimeline, WorkArea,
};
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::hash::{Hash, Hasher};

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
}
pub struct WorldUi {
    pub playhead: u32,
    pub playing: bool,
    pub looping: bool,
    fraction: f64,
    view: TimelineView,
    expanded: HashSet<NodeId>,
    groups: HashSet<(NodeId, String)>,
    keys: HashSet<KeyIdentity>,
    snap: bool,
    error: Option<String>,
    metadata: String,
    metadata_node: Option<NodeId>,
    custom_name: String,
    custom_value: f64,
    picker: Option<(NodeId, egui_file_dialog::FileDialog)>,
    environment_file: egui_file_field::FileFieldState,
}
impl Default for WorldUi {
    fn default() -> Self {
        Self {
            playhead: 0,
            playing: false,
            looping: true,
            fraction: 0.0,
            view: TimelineView::default(),
            expanded: HashSet::new(),
            groups: HashSet::new(),
            keys: HashSet::new(),
            snap: true,
            error: None,
            metadata: String::new(),
            metadata_node: None,
            custom_name: String::new(),
            custom_value: 0.0,
            picker: None,
            environment_file: egui_file_field::FileFieldState::default(),
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
        *self = Self::default();
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
        let nodes = e.document.nodes();
        let map: HashMap<_, _> = nodes.iter().map(|n| (wid(n.id), n.id)).collect();
        let mut model = OutlinerModel::with_roots(tree(&nodes, None, 0));
        model.selection = e.selected.iter().map(|id| wid(*id)).collect();
        let mut cfg = OutlinerConfig::default()
            .with_filter(true)
            .with_editable_labels(true)
            .with_draggable(true)
            .with_visibility_on_right(true)
            .with_row_height(24.0);
        cfg.context_items = vec![
            ContextItem::new("duplicate", "Duplicate"),
            ContextItem::new("unparent", "Move to root"),
            ContextItem::new("active", "Make active camera / environment"),
            ContextItem::new("delete", "Delete").with_separator(),
        ];
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
    }
    fn lanes(&self, id: NodeId, attrs: Vec<WorldAttribute>) -> Vec<Lane> {
        let mut by_group: BTreeMap<&str, Vec<WorldAttribute>> = BTreeMap::new();
        for a in attrs.into_iter().filter(|a| a.keyable) {
            by_group.entry(category(&a.path)).or_default().push(a);
        }
        let mut lanes = Vec::new();
        for group in [
            "Transform",
            "Fractal",
            "Camera",
            "Light",
            "Environment",
            "Material",
            "Render",
            "Color",
            "Custom",
        ] {
            let Some(attrs) = by_group.remove(group) else {
                continue;
            };
            lanes.push(Lane {
                label: group.into(),
                path: Some(format!("@{group}")),
                value: Value::Null,
                frames: vec![],
                depth: 1,
                group: true,
            });
            if !self.groups.contains(&(id, format!("@{group}"))) {
                continue;
            }
            for a in attrs {
                lanes.push(Lane {
                    label: a.label,
                    path: Some(a.path),
                    value: a.value,
                    frames: a.frames,
                    depth: if a.component.is_some() { 3 } else { 2 },
                    group: false,
                });
            }
        }
        lanes
    }
    pub fn timeline(&mut self, ui: &mut egui::Ui, e: &mut WorldEditor) {
        self.transport(ui, e);
        self.status(ui);
        let nodes = e.document.nodes();
        let lanes: Vec<Vec<Lane>> = nodes
            .iter()
            .map(|n| {
                self.lanes(
                    n.id,
                    e.document
                        .attributes(n.id, self.playhead as f64)
                        .unwrap_or_default(),
                )
            })
            .collect();
        let tracks = nodes
            .iter()
            .zip(&lanes)
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
                                .map(|f| {
                                    Keyframe::new(*f, l.value.as_f64().unwrap_or(0.0)).selected(
                                        self.keys.contains(&KeyIdentity {
                                            node: n.id,
                                            path: l.path.clone().unwrap_or_default(),
                                            frame: f.to_bits(),
                                        }),
                                    )
                                })
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
        let mut model = TimelineModel::new(e.document.fps as f32, tracks);
        model.playhead = self.playhead as i64;
        model.selection = e.selected.iter().map(|id| wid(*id)).collect();
        model.work_area = Some(WorkArea {
            start: e.document.first as i64,
            end: e.document.last as i64 + 1,
        });
        let cfg = TimelineConfig {
            row_height: 26.0,
            lane_height: 24.0,
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
                        for (li, l) in lanes[ti].iter().enumerate() {
                            let rect = Rect::from_min_size(
                                Pos2::new(
                                    origin.x,
                                    y + cfg.row_height + li as f32 * resp.lane_height,
                                ),
                                Vec2::new(left_w, resp.lane_height),
                            );
                            self.lane_row(ui, e, n.id, l, rect);
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
    }
    fn layer_row(
        &mut self,
        ui: &mut egui::Ui,
        e: &mut WorldEditor,
        n: &WorldNodeInfo,
        nodes: &[WorldNodeInfo],
        r: Rect,
    ) {
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
        let mut row = ui.new_child(
            egui::UiBuilder::new()
                .id_salt(("layer", wid(n.id)))
                .max_rect(r.shrink2(Vec2::new(5.0, 1.0)))
                .layout(egui::Layout::left_to_right(egui::Align::Center)),
        );
        row.spacing_mut().item_spacing.x = 3.0;
        if row
            .small_button(if self.expanded.contains(&n.id) {
                ph::CARET_DOWN
            } else {
                ph::CARET_RIGHT
            })
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
            if row
                .small_button(if state { on } else { off })
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
        row.colored_label(kind_color(n.kind), kind_icon(n.kind));
        if row
            .selectable_label(e.selected.contains(&n.id), &n.name)
            .clicked()
        {
            let m = row.input(|i| i.modifiers);
            self.select(e, n.id, m.command || m.ctrl, m.shift, nodes);
        }
        row.menu_button(ph::DOTS_THREE, |ui| {
            for (label, delta) in [("Move up", -1isize), ("Move down", 1)] {
                if ui.button(label).clicked() {
                    let mut ids = nodes.iter().map(|n| n.id).collect::<Vec<_>>();
                    if let Some(i) = ids.iter().position(|id| *id == n.id) {
                        let to = i as isize + delta;
                        if to >= 0 && (to as usize) < ids.len() {
                            ids.swap(i, to as usize);
                            self.command(e, WorldCommand::Reorder { ids });
                        }
                    }
                    ui.close();
                }
            }
        });
    }

    fn lane_row(&mut self, ui: &mut egui::Ui, e: &mut WorldEditor, id: NodeId, l: &Lane, r: Rect) {
        if !r.intersects(ui.clip_rect()) {
            return;
        }
        let Some(path) = &l.path else { return };
        let mut row = ui.new_child(
            egui::UiBuilder::new()
                .id_salt((wid(id), path))
                .max_rect(r.shrink2(Vec2::new(5.0, 1.0)))
                .layout(egui::Layout::left_to_right(egui::Align::Center)),
        );
        row.spacing_mut().item_spacing.x = 4.0;
        row.add_space(l.depth as f32 * 13.0);
        if l.group {
            let open = self.groups.contains(&(id, path.clone()));
            if row
                .small_button(if open {
                    ph::CARET_DOWN
                } else {
                    ph::CARET_RIGHT
                })
                .clicked()
            {
                if !self.groups.remove(&(id, path.clone())) {
                    self.groups.insert((id, path.clone()));
                }
            }
            row.strong(&l.label);
        } else {
            if row
                .small_button(
                    egui::RichText::new(ph::TIMER).color(if l.frames.is_empty() {
                        row.visuals().weak_text_color()
                    } else {
                        row.visuals().selection.stroke.color
                    }),
                )
                .on_hover_text("Enable or disable animation")
                .clicked()
            {
                self.command(
                    e,
                    WorldCommand::SetAnimation {
                        id,
                        path: path.clone(),
                        enabled: l.frames.is_empty(),
                        frame: self.playhead as f64,
                    },
                );
            }
            if row
                .small_button(egui::RichText::new(ph::DIAMOND).color(
                    if l.frames.contains(&(self.playhead as f64)) {
                        row.visuals().selection.stroke.color
                    } else {
                        row.visuals().weak_text_color()
                    },
                ))
                .on_hover_text("Add / remove a key at the current frame")
                .clicked()
            {
                let frame = self.playhead as f64;
                self.command(
                    e,
                    if l.frames.contains(&frame) {
                        WorldCommand::RemoveKey {
                            id,
                            path: path.clone(),
                            frame,
                        }
                    } else {
                        WorldCommand::Key {
                            id,
                            path: path.clone(),
                            frame,
                        }
                    },
                );
            }
            row.label(&l.label);
            let mut v = l.value.clone();
            if value_editor(&mut row, &mut v, path) {
                self.command(
                    e,
                    WorldCommand::SetAttribute {
                        id,
                        path: path.clone(),
                        value: v,
                        frame: self.playhead as f64,
                    },
                );
            }
        }
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
        let (mut first, mut last, mut fps) = (e.document.first, e.document.last, e.document.fps);
        let before = (first, last, fps);
        ui.horizontal_wrapped(|ui| {
            if ui
                .small_button(ph::SKIP_BACK)
                .on_hover_text("First frame")
                .clicked()
            {
                self.seek(e.document.first);
            }
            if ui.small_button(ph::CARET_LEFT).clicked() {
                self.seek(self.playhead.saturating_sub(1).max(e.document.first));
            }
            if ui
                .small_button(if self.playing { ph::PAUSE } else { ph::PLAY })
                .on_hover_text("Play / pause")
                .clicked()
            {
                self.playing = !self.playing;
            }
            if ui.small_button(ph::CARET_RIGHT).clicked() {
                self.seek(self.playhead.saturating_add(1).min(e.document.last));
            }
            if ui.small_button(ph::SKIP_FORWARD).clicked() {
                self.seek(e.document.last);
            }
            ui.toggle_value(&mut self.looping, ph::ARROWS_CLOCKWISE)
                .on_hover_text("Loop work area");
            ui.label("Frame");
            ui.add(
                egui::DragValue::new(&mut self.playhead).range(e.document.first..=e.document.last),
            );
            ui.separator();
            ui.label("FPS");
            ui.add(egui::DragValue::new(&mut fps).speed(1).range(1.0..=240.0));
            ui.label("In");
            ui.add(egui::DragValue::new(&mut first).range(0..=last));
            ui.label("Out");
            ui.add(egui::DragValue::new(&mut last).range(first..=100000));
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
    pub fn inspector(&mut self, ui: &mut egui::Ui, e: &mut WorldEditor) {
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
        let Some(node) = e.document.nodes().into_iter().find(|n| n.id == id) else {
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
        let materials = e
            .document
            .nodes()
            .into_iter()
            .filter(|n| n.kind == WorldKind::Material)
            .collect::<Vec<_>>();
        let attrs = e
            .document
            .attributes(id, self.playhead as f64)
            .unwrap_or_default();
        egui::ScrollArea::vertical()
            .id_salt(("world_inspector", wid(id)))
            .show(ui, |ui| {
                for group in [
                    "Transform",
                    "Fractal",
                    "Camera",
                    "Light",
                    "Environment",
                    "Material",
                    "Render",
                    "Color",
                    "Custom",
                ] {
                    let group_attrs: Vec<_> = attrs
                        .iter()
                        .filter(|a| category(&a.path) == group && a.component.is_none())
                        .collect();
                    if group_attrs.is_empty() {
                        continue;
                    }
                    egui::CollapsingHeader::new(group)
                        .default_open(true)
                        .show(ui, |ui| {
                            for a in group_attrs {
                                ui.push_id(&a.path, |ui| {
                                    ui.horizontal(|ui| {
                                        if a.keyable
                                            && ui
                                                .small_button(egui::RichText::new(ph::TIMER).color(
                                                    if a.frames.is_empty() {
                                                        ui.visuals().weak_text_color()
                                                    } else {
                                                        ui.visuals().selection.stroke.color
                                                    },
                                                ))
                                                .on_hover_text("Enable or disable animation")
                                                .clicked()
                                        {
                                            self.command(
                                                e,
                                                WorldCommand::SetAnimation {
                                                    id,
                                                    path: a.path.clone(),
                                                    enabled: a.frames.is_empty(),
                                                    frame: self.playhead as f64,
                                                },
                                            );
                                        }
                                        if a.keyable
                                            && ui
                                                .small_button(ph::DIAMOND)
                                                .on_hover_text("Key this property at current frame")
                                                .clicked()
                                        {
                                            self.command(
                                                e,
                                                WorldCommand::Key {
                                                    id,
                                                    path: a.path.clone(),
                                                    frame: self.playhead as f64,
                                                },
                                            );
                                        }
                                        ui.label(&a.label);
                                        let mut value = a.value.clone();
                                        let changed = if a.path == "/material_id" {
                                            let before = value.clone();
                                            let selected = value.as_str().and_then(NodeId::parse);
                                            let label = materials
                                                .iter()
                                                .find(|n| Some(n.id) == selected)
                                                .map(|n| n.name.as_str())
                                                .unwrap_or("Object material");
                                            egui::ComboBox::from_id_salt("material_reference")
                                                .selected_text(label)
                                                .show_ui(ui, |ui| {
                                                    ui.selectable_value(
                                                        &mut value,
                                                        Value::Null,
                                                        "Object material",
                                                    );
                                                    for material in &materials {
                                                        ui.selectable_value(
                                                            &mut value,
                                                            json!(material.id),
                                                            &material.name,
                                                        );
                                                    }
                                                });
                                            value != before
                                        } else if a.path == "/environment/path" {
                                            let mut path =
                                                value.as_str().unwrap_or_default().to_owned();
                                            let response = egui_file_field::FileField::new(
                                                &mut path,
                                                &mut self.environment_file,
                                            )
                                            .hint("HDR / EXR environment")
                                            .tooltip("Browse HDR / EXR environment")
                                            .show(ui);
                                            if response.browse_clicked() {
                                                let start = self.environment_file.start_dir(&path);
                                                let mut picker =
                                                    egui_file_dialog::FileDialog::new()
                                                        .add_file_filter(
                                                            "HDR / EXR",
                                                            egui_file_dialog::Filter::new(
                                                                |p: &std::path::Path| {
                                                                    p.extension()
                                                                    .and_then(|ext| ext.to_str())
                                                                    .is_some_and(|ext| {
                                                                        ext.eq_ignore_ascii_case(
                                                                            "hdr",
                                                                        ) || ext
                                                                            .eq_ignore_ascii_case(
                                                                                "exr",
                                                                            )
                                                                    })
                                                                },
                                                            ),
                                                        );
                                                if !start.is_empty() {
                                                    picker = picker.initial_directory(start.into());
                                                }
                                                picker.pick_file();
                                                self.picker = Some((id, picker));
                                            }
                                            value = json!(path);
                                            response.changed()
                                        } else {
                                            schema_editor(ui, &mut value, a)
                                        };
                                        if changed {
                                            self.command(
                                                e,
                                                WorldCommand::SetAttribute {
                                                    id,
                                                    path: a.path.clone(),
                                                    value,
                                                    frame: self.playhead as f64,
                                                },
                                            );
                                        }
                                    });
                                });
                            }
                        });
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
                    if self.metadata_node != Some(id) {
                        self.metadata = e
                            .document
                            .metadata(id)
                            .ok()
                            .and_then(|v| serde_json::to_string_pretty(&v).ok())
                            .unwrap_or_else(|| "{}".into());
                        self.metadata_node = Some(id);
                    }
                    ui.add(
                        egui::TextEdit::multiline(&mut self.metadata)
                            .code_editor()
                            .desired_rows(8)
                            .desired_width(f32::INFINITY),
                    );
                    if ui.button("Apply metadata").clicked() {
                        match serde_json::from_str(&self.metadata) {
                            Ok(value) => self.command(
                                e,
                                WorldCommand::SetMetadata {
                                    id,
                                    path: "".into(),
                                    value,
                                },
                            ),
                            Err(err) => self.error = Some(format!("Invalid metadata JSON: {err}")),
                        }
                    }
                });
            });
    }
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
fn schema_editor(ui: &mut egui::Ui, value: &mut Value, attr: &WorldAttribute) -> bool {
    if !attr.choices.is_empty() {
        let before = value.clone();
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
                        apply_choice(&attr.path, value, choice);
                    }
                }
            });
        if attr.path == "/julia" && value.is_array() {
            value_editor(ui, value, &attr.path);
        }
        return *value != before;
    }
    if let (Some((lo, hi)), Some(n)) = (attr.range, value.as_f64()) {
        let mut n = n;
        if ui
            .add(egui::Slider::new(&mut n, lo..=hi).clamping(egui::SliderClamping::Never))
            .changed()
        {
            *value = numeric_value(value, n);
            return true;
        }
        return false;
    }
    value_editor(ui, value, &attr.path)
}
fn value_editor(ui: &mut egui::Ui, value: &mut Value, path: &str) -> bool {
    if path == "/formula" {
        ui.weak(choice_label(path, value));
        return false;
    }
    if path == "/julia" && value.is_null() {
        ui.weak("Disabled");
        return false;
    }
    match value {
        Value::Bool(v) => ui.checkbox(v, "").changed(),
        Value::Number(v) => {
            let integer = v.is_u64() || v.is_i64();
            let template = Value::Number(v.clone());
            let mut n = v.as_f64().unwrap_or(0.0);
            let changed = ui
                .add(egui::DragValue::new(&mut n).speed(if integer {
                    1.0
                } else if path.contains("rotation") {
                    0.5
                } else {
                    0.02
                }))
                .changed();
            if changed {
                *value = numeric_value(&template, n);
            }
            changed
        }
        Value::String(v) => ui
            .add(egui::TextEdit::singleline(v).desired_width(140.0))
            .changed(),
        Value::Array(a) => {
            let mut changed = false;
            if a.len() == 3 && path.contains("color") {
                let mut rgb = [
                    a[0].as_f64().unwrap_or(0.0) as f32,
                    a[1].as_f64().unwrap_or(0.0) as f32,
                    a[2].as_f64().unwrap_or(0.0) as f32,
                ];
                if ui.color_edit_button_rgb(&mut rgb).changed() {
                    *value = json!(rgb);
                    return true;
                }
            }
            for (i, v) in a.iter_mut().enumerate() {
                ui.weak(["X", "Y", "Z", "W"].get(i).copied().unwrap_or(""));
                changed |= value_editor(ui, v, path);
            }
            changed
        }
        _ => {
            ui.weak(value.to_string());
            false
        }
    }
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
            e.document.attributes(render_node.id, 0.0).unwrap(),
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
