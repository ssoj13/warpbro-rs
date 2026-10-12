//! The Takes panel: the document's recorded navigation takes (`crate::takes`) as a managed list.
//! Filter, multi-select (Ctrl / Shift click), pin, bind as the scene camera, play, rename,
//! copy / paste / duplicate (the system clipboard, so takes move between scenes), delete, export /
//! import clip files, takes made from the timeline animation or the current pose, and
//! non-destructive edits (trim, speed, smooth, split: each makes a new take, since takes have no
//! undo). Bake to keys keys the bound take onto the camera as one undo step.
use super::App;
use crate::camera_recorder::Timing;
use crate::takes::{Take, TakeKind};
use crate::world::NodeId;
use egui_phosphor::regular as ph;
use egui_widgets_config::icons;
use std::collections::BTreeSet;

/// Panel state, not saved with the scene.
pub(super) struct View {
    filter: String,
    pinned_only: bool,
    selection: BTreeSet<u64>,
    /// Shift-click range anchor.
    anchor: Option<u64>,
    /// The take whose name is being edited.
    rename: Option<(u64, String)>,
    /// Edit parameters of the selected take: trim in / out (seconds), speed, smooth window.
    trim: (f64, f64),
    speed: f64,
    smooth: f64,
    /// The take `trim` was initialised for: a new selection resets it to the whole take.
    trim_of: Option<u64>,
    /// Export (Some take) or import (None) file dialog.
    picker: Option<(Option<u64>, egui_file_dialog::FileDialog)>,
    /// Copied takes text, handed to the system clipboard by the next caller holding a context.
    clipboard: Option<String>,
}
impl Default for View {
    fn default() -> Self {
        Self {
            filter: String::new(),
            pinned_only: false,
            selection: BTreeSet::new(),
            anchor: None,
            rename: None,
            trim: (0.0, 0.0),
            speed: 1.0,
            smooth: 0.25,
            trim_of: None,
            picker: None,
            clipboard: None,
        }
    }
}

/// What a row or toolbar asked for; applied after drawing (drawing borrows the document).
pub(super) enum Action {
    Select {
        id: u64,
        ctrl: bool,
        shift: bool,
    },
    Pin(Vec<u64>, bool),
    /// Bind the take to its targets, or unbind it everywhere when it is bound.
    Bind(u64),
    Play(u64),
    Rename(u64, String),
    Delete(Vec<u64>),
    Copy(Vec<u64>),
    Duplicate(Vec<u64>),
    Export(u64),
    Import,
    FromAnimation,
    FromPose,
    /// A new take derived from `source` (trim, speed, smooth, split).
    Derive(u64, Derive),
    SetStart(u64, f64),
    SetTiming(u64, Timing),
    Bake(u64),
}
pub(super) enum Derive {
    Trim(f64, f64),
    Speed(f64),
    Smooth(f64),
    Split,
}

impl App {
    pub(super) fn takes_panel(&mut self, ui: &mut egui::Ui) {
        let mut actions = Vec::new();
        self.takes_toolbar(ui, &mut actions);
        ui.separator();
        self.takes_list(ui, &mut actions);
        ui.separator();
        self.takes_editor(ui, &mut actions);
        for action in actions {
            self.take_action(action);
        }
        if let Some(text) = self.takes_view.clipboard.take() {
            ui.ctx().copy_text(text);
        }
    }

    /// The takes passing the filter, newest first: the order rows are drawn and ranges select in.
    fn visible_takes(&self) -> Vec<u64> {
        let needle = self.takes_view.filter.to_lowercase();
        self.world
            .document
            .takes
            .list()
            .iter()
            .rev()
            .filter(|t| !self.takes_view.pinned_only || t.pinned)
            .filter(|t| needle.is_empty() || t.name.to_lowercase().contains(&needle))
            .map(|t| t.id)
            .collect()
    }

    fn selected_takes(&self) -> Vec<u64> {
        let takes = &self.world.document.takes;
        self.takes_view
            .selection
            .iter()
            .copied()
            .filter(|id| takes.get(*id).is_some())
            .collect()
    }

    fn takes_toolbar(&mut self, ui: &mut egui::Ui, actions: &mut Vec<Action>) {
        let selected = self.selected_takes();
        let any = !selected.is_empty();
        ui.horizontal_wrapped(|ui| {
            ui.add(
                egui::TextEdit::singleline(&mut self.takes_view.filter)
                    .hint_text(format!("{} Filter", icons::SEARCH))
                    .desired_width(110.0),
            );
            ui.toggle_value(&mut self.takes_view.pinned_only, ph::PUSH_PIN)
                .on_hover_text("Pinned takes only");
            ui.separator();
            if ui
                .button(ph::FILM_SLATE)
                .on_hover_text("New takes from the timeline animation (work area) of the selected objects, or of the active camera")
                .clicked()
            {
                actions.push(Action::FromAnimation);
            }
            if ui
                .button(icons::CAMERA)
                .on_hover_text("New still takes of the current pose of the selected objects, or of the camera")
                .clicked()
            {
                actions.push(Action::FromPose);
            }
            ui.separator();
            let all_pinned = any
                && selected
                    .iter()
                    .all(|id| self.world.document.takes.get(*id).is_some_and(|t| t.pinned));
            if ui
                .add_enabled(any, egui::Button::new(ph::PUSH_PIN))
                .on_hover_text("Pin / unpin the selection")
                .clicked()
            {
                actions.push(Action::Pin(selected.clone(), !all_pinned));
            }
            if ui
                .add_enabled(any, egui::Button::new(icons::COPY))
                .on_hover_text("Copy (Ctrl+C): pastes into any scene")
                .clicked()
            {
                actions.push(Action::Copy(selected.clone()));
            }
            if ui
                .add_enabled(any, egui::Button::new(ph::COPY_SIMPLE))
                .on_hover_text("Duplicate (Ctrl+D)")
                .clicked()
            {
                actions.push(Action::Duplicate(selected.clone()));
            }
            if ui
                .add_enabled(selected.len() == 1, egui::Button::new(ph::EXPORT))
                .on_hover_text("Export clip file")
                .clicked()
            {
                actions.push(Action::Export(selected[0]));
            }
            if ui
                .button(ph::DOWNLOAD_SIMPLE)
                .on_hover_text("Import clip file (Ctrl+V pastes copied takes)")
                .clicked()
            {
                actions.push(Action::Import);
            }
            if ui
                .add_enabled(any, egui::Button::new(icons::TRASH))
                .on_hover_text("Delete the selection (Delete)")
                .clicked()
            {
                actions.push(Action::Delete(selected));
            }
        });
    }

    fn takes_list(&mut self, ui: &mut egui::Ui, actions: &mut Vec<Action>) {
        let visible = self.visible_takes();
        let fps = self.world.document.fps;
        let document = &self.world.document;
        let name_of = |node: NodeId| document.info(node).map(|n| n.name).unwrap_or_default();
        if self.world.document.takes.list().is_empty() {
            ui.label("Every viewport navigation gesture is recorded here as a take.");
        }
        egui::ScrollArea::vertical()
            .auto_shrink([false, true])
            .max_height((ui.available_height() - 170.0).max(80.0))
            .show(ui, |ui| {
                for id in visible {
                    let Some(take) = self.world.document.takes.get(id) else {
                        continue;
                    };
                    let selected = self.takes_view.selection.contains(&id);
                    let bound: Vec<String> =
                        document.take_users(id).into_iter().map(name_of).collect();
                    let kind_icon = match take.kind() {
                        TakeKind::Camera => icons::CAMERA,
                        TakeKind::Transform => icons::MOVE,
                    };
                    ui.horizontal(|ui| {
                        ui.label(kind_icon).on_hover_text(match (take.kind(), take.source) {
                            (TakeKind::Camera, _) => "Camera take: a camera's pose and lens".to_owned(),
                            (TakeKind::Transform, Some(node)) => {
                                format!("Transform take, recorded on {}", name_of(node))
                            }
                            (TakeKind::Transform, None) => "Transform take".to_owned(),
                        });
                        if ui
                            .selectable_label(take.pinned, ph::PUSH_PIN)
                            .on_hover_text("Pinned takes are never dropped (20 newest unpinned are kept)")
                            .clicked()
                        {
                            actions.push(Action::Pin(vec![id], !take.pinned));
                        }
                        let linked = !bound.is_empty();
                        if ui
                            .selectable_label(linked, if linked { icons::LINK } else { ph::LINK_BREAK })
                            .on_hover_text(if linked {
                                format!("Drives {} (click: unbind)", bound.join(", "))
                            } else {
                                "Bind: a camera take drives the active camera, a transform take the selected objects (or the one it was recorded on)".to_owned()
                            })
                            .clicked()
                        {
                            actions.push(Action::Bind(id));
                        }
                        if ui.button(icons::PLAY).on_hover_text("Bind and play from its start").clicked() {
                            actions.push(Action::Play(id));
                        }
                        match &mut self.takes_view.rename {
                            Some((editing, name)) if *editing == id => {
                                let edit = ui.add(egui::TextEdit::singleline(name).desired_width(130.0));
                                if edit.lost_focus() {
                                    actions.push(Action::Rename(id, name.clone()));
                                } else {
                                    edit.request_focus();
                                }
                            }
                            _ => {
                                let label = ui
                                    .selectable_label(selected, &take.name)
                                    .on_hover_text("Click: select · Ctrl / Shift: add / range · Double-click: rename");
                                if label.double_clicked() {
                                    self.takes_view.rename = Some((id, take.name.clone()));
                                } else if label.clicked() {
                                    let m = ui.input(|i| i.modifiers);
                                    actions.push(Action::Select {
                                        id,
                                        ctrl: m.command,
                                        shift: m.shift,
                                    });
                                }
                            }
                        }
                        ui.weak(format!(
                            "{:.1} s · frame {:.0} · {}",
                            take.duration(),
                            take.start,
                            take.len()
                        ))
                        .on_hover_text(format!(
                            "{:.2} s from frame {:.1} ({:.1} frames at {fps} fps), {} samples",
                            take.duration(),
                            take.start,
                            take.duration() * fps,
                            take.len()
                        ));
                    });
                }
            });
    }

    /// Edits of the single selected take, and the bake of the bound one.
    fn takes_editor(&mut self, ui: &mut egui::Ui, actions: &mut Vec<Action>) {
        let selected = self.selected_takes();
        let take = (selected.len() == 1)
            .then(|| self.world.document.takes.get(selected[0]).cloned())
            .flatten();
        if let Some(take) = &take {
            if self.takes_view.trim_of != Some(take.id) {
                self.takes_view.trim = (0.0, take.duration());
                self.takes_view.trim_of = Some(take.id);
            }
            let id = take.id;
            let view = &mut self.takes_view;
            egui::Grid::new("take_edit").num_columns(3).show(ui, |ui| {
                ui.label("Start frame");
                let mut start = take.start;
                if ui
                    .add(egui::DragValue::new(&mut start).speed(1.0))
                    .changed()
                {
                    actions.push(Action::SetStart(id, start));
                }
                let mut timing = take.timing;
                ui.horizontal(|ui| {
                    ui.radio_value(&mut timing, Timing::KeepSpeed, "Keep speed")
                        .on_hover_text("Plays at recorded speed from its start frame");
                    ui.radio_value(&mut timing, Timing::FitWorkArea, "Fit work area");
                });
                if timing != take.timing {
                    actions.push(Action::SetTiming(id, timing));
                }
                ui.end_row();

                ui.label("Trim (s)");
                let duration = take.duration();
                ui.horizontal(|ui| {
                    ui.add(
                        egui::DragValue::new(&mut view.trim.0)
                            .range(0.0..=duration)
                            .speed(0.01),
                    );
                    ui.add(
                        egui::DragValue::new(&mut view.trim.1)
                            .range(0.0..=duration)
                            .speed(0.01),
                    );
                });
                if ui
                    .button("Trim")
                    .on_hover_text("New take: the part between In and Out")
                    .clicked()
                {
                    actions.push(Action::Derive(id, Derive::Trim(view.trim.0, view.trim.1)));
                }
                ui.end_row();

                ui.label("Speed");
                ui.add(
                    egui::DragValue::new(&mut view.speed)
                        .range(0.05..=20.0)
                        .speed(0.01),
                );
                if ui
                    .button("Retime")
                    .on_hover_text("New take played this many times as fast")
                    .clicked()
                {
                    actions.push(Action::Derive(id, Derive::Speed(view.speed)));
                }
                ui.end_row();

                ui.label("Smooth (s)");
                ui.add(
                    egui::DragValue::new(&mut view.smooth)
                        .range(0.0..=5.0)
                        .speed(0.01),
                );
                if ui
                    .button("Smooth")
                    .on_hover_text("New take with the hand shake averaged out over this window")
                    .clicked()
                {
                    actions.push(Action::Derive(id, Derive::Smooth(view.smooth)));
                }
                ui.end_row();

                ui.label("");
                ui.label("");
                if ui
                    .button("Split")
                    .on_hover_text("Two new takes, cut at the playhead")
                    .clicked()
                {
                    actions.push(Action::Derive(id, Derive::Split));
                }
                ui.end_row();
            });
        } else {
            ui.weak("Select one take to edit it.");
        }
        let options = &mut self.recorder_options;
        ui.horizontal_wrapped(|ui| {
            ui.label("Bake");
            ui.checkbox(&mut options.transform, "Transform");
            ui.checkbox(&mut options.focus, "Focus");
            ui.checkbox(&mut options.zoom, "Zoom");
            ui.checkbox(&mut options.f_number, "f-number");
            ui.add(
                egui::DragValue::new(&mut options.tolerance)
                    .range(0.0..=1.0)
                    .speed(0.0001),
            )
            .on_hover_text(
                "Reduction error: maximum absolute error at the samples, in each channel's units",
            );
        });
        if ui
            .add_enabled(
                take.is_some(),
                egui::Button::new(format!("{} Bake to keys", icons::RECORD)),
            )
            .on_hover_text(
                "Key the selected take onto the nodes it drives (else its targets, as Bind), one undo each; keys outside its span remain",
            )
            .clicked()
            && let Some(take) = &take
        {
            actions.push(Action::Bake(take.id));
        }
    }

    pub(super) fn take_action(&mut self, action: Action) {
        let fps = self.world.document.fps;
        let result: Result<(), String> = match action {
            Action::Select { id, ctrl, shift } => {
                self.select_take(id, ctrl, shift);
                Ok(())
            }
            Action::Pin(ids, pinned) => {
                for id in ids {
                    self.world.edit_take(id, |t| t.pinned = pinned);
                }
                Ok(())
            }
            Action::Bind(id) => self.toggle_binding(id),
            Action::Play(id) => self.bind_targets(id).map(|()| {
                if let Some(take) = self.world.document.takes.get(id) {
                    let (first, last) = (self.world.document.first, self.world.document.last);
                    self.world_ui
                        .seek((take.start.round().max(0.0) as u32).clamp(first, last));
                }
                self.world_ui.playing = true;
            }),
            Action::Rename(id, name) => {
                self.takes_view.rename = None;
                let name = name.trim().to_owned();
                if !name.is_empty() {
                    self.world.edit_take(id, |t| t.name = name);
                }
                Ok(())
            }
            Action::Delete(ids) => {
                let mut count = 0;
                let mut refused = Ok(());
                for id in ids {
                    match self.world.delete_take(id) {
                        Ok(removed) => {
                            count += usize::from(removed);
                            self.takes_view.selection.remove(&id);
                        }
                        Err(error) => refused = Err(error),
                    }
                }
                self.status = format!("Deleted {count} take(s)");
                refused
            }
            Action::Copy(ids) => self.copy_takes(&ids),
            Action::Duplicate(ids) => {
                let copies: Vec<Take> = ids
                    .iter()
                    .filter_map(|id| self.world.document.takes.get(*id))
                    .map(|t| Take {
                        name: format!("{} copy", t.name),
                        pinned: false,
                        ..(**t).clone()
                    })
                    .collect();
                self.add_takes(copies, "Duplicated");
                Ok(())
            }
            Action::Export(id) => {
                self.take_dialog(Some(id));
                Ok(())
            }
            Action::Import => {
                self.take_dialog(None);
                Ok(())
            }
            Action::FromAnimation => self.takes_from_selection(true),
            Action::FromPose => self.takes_from_selection(false),
            Action::Derive(id, derive) => {
                let Some(take) = self.world.document.takes.get(id).cloned() else {
                    return;
                };
                let named = |t: Take, suffix: &str| Take {
                    name: format!("{} {suffix}", take.name),
                    pinned: false,
                    ..t
                };
                let playhead = take.seconds_at(
                    f64::from(self.world_ui.playhead),
                    self.world.document.first,
                    self.world.document.last,
                    fps,
                );
                match derive {
                    Derive::Trim(from, to) => {
                        take.trimmed(from, to, fps).map(|t| vec![named(t, "trim")])
                    }
                    Derive::Speed(speed) => take
                        .retimed(speed)
                        .map(|t| vec![named(t, &format!("x{speed}"))]),
                    Derive::Smooth(window) => Ok(vec![named(take.smoothed(window), "smooth")]),
                    Derive::Split => take
                        .split(playhead, fps)
                        .map(|(a, b)| vec![named(a, "A"), named(b, "B")]),
                }
                .map(|takes| self.add_takes(takes, "Made"))
            }
            Action::SetStart(id, start) => {
                self.world.edit_take(id, |t| t.start = start);
                Ok(())
            }
            Action::SetTiming(id, timing) => {
                self.world.edit_take(id, |t| t.timing = timing);
                Ok(())
            }
            Action::Bake(id) => {
                self.bake_take(id);
                Ok(())
            }
        };
        self.evaluated_world = None;
        if let Err(error) = result {
            self.status = error;
        }
    }

    /// Plain click selects one take, Ctrl toggles it, Shift extends from the anchor over the
    /// visible (filtered) order.
    fn select_take(&mut self, id: u64, ctrl: bool, shift: bool) {
        if shift && let Some(anchor) = self.takes_view.anchor {
            let visible = self.visible_takes();
            let view = &mut self.takes_view;
            let (a, b) = (
                visible.iter().position(|t| *t == anchor),
                visible.iter().position(|t| *t == id),
            );
            if let (Some(a), Some(b)) = (a, b) {
                if !ctrl {
                    view.selection.clear();
                }
                view.selection.extend(&visible[a.min(b)..=a.max(b)]);
                return;
            }
        }
        let view = &mut self.takes_view;
        if ctrl {
            if !view.selection.remove(&id) {
                view.selection.insert(id);
            }
        } else {
            view.selection.clear();
            view.selection.insert(id);
        }
        view.anchor = Some(id);
    }

    /// Store new takes and select them.
    fn add_takes(&mut self, takes: Vec<Take>, verb: &str) {
        let count = takes.len();
        self.takes_view.selection.clear();
        for take in takes {
            let id = self.world.add_take(take);
            self.takes_view.selection.insert(id);
            self.takes_view.anchor = Some(id);
        }
        self.status = format!("{verb} {count} take(s)");
    }

    fn copy_takes(&mut self, ids: &[u64]) -> Result<(), String> {
        let takes: Vec<&Take> = ids
            .iter()
            .filter_map(|id| self.world.document.takes.get(*id).map(|t| &**t))
            .collect();
        if takes.is_empty() {
            return Ok(());
        }
        let text = crate::takes::to_clipboard(&takes, self.world.document.fps)?;
        let count = takes.len();
        self.takes_view.clipboard = Some(text);
        self.status = format!("Copied {count} take(s)");
        Ok(())
    }

    /// Ctrl+C / Ctrl+D / Delete while the pointer is over the Takes panel.
    pub(super) fn takes_hotkeys(&mut self, ctx: &egui::Context) {
        use crate::hotkeys::{self, Command as Hotkey, Scope};
        let selected = self.selected_takes();
        if hotkeys::consume(ctx, Scope::Global, Hotkey::Duplicate) && !selected.is_empty() {
            self.take_action(Action::Duplicate(selected.clone()));
        }
        if hotkeys::consume(ctx, Scope::Takes, Hotkey::Delete) && !selected.is_empty() {
            self.take_action(Action::Delete(selected.clone()));
        }
        if hotkeys::take_copy(ctx, !selected.is_empty()) {
            self.take_action(Action::Copy(selected));
        }
        if let Some(text) = self.takes_view.clipboard.take() {
            ctx.copy_text(text);
        }
    }

    /// Paste takes copied in any WarpBro scene (Ctrl+V anywhere: only takes text is claimed).
    pub(super) fn paste_takes(&mut self, ctx: &egui::Context) {
        let Some(text) =
            crate::hotkeys::take_paste(ctx, |text| crate::takes::from_clipboard(text).is_some())
        else {
            return;
        };
        match crate::takes::from_clipboard(&text) {
            Some(Ok(takes)) => self.add_takes(takes, "Pasted"),
            Some(Err(error)) => self.status = error,
            None => {}
        }
    }

    /// The placed objects among the selected nodes: what a transform take binds to and is made of.
    fn selected_objects(&self) -> Vec<NodeId> {
        let selected: Vec<NodeId> = if self.world.selected.is_empty() {
            self.world.selection.into_iter().collect()
        } else {
            self.world.selected.clone()
        };
        selected
            .into_iter()
            .filter(|id| {
                self.world.document.info(*id).is_ok_and(|n| {
                    n.kind != crate::world::WorldKind::Camera
                        && crate::world::manip_tools(n.kind).contains(&true)
                })
            })
            .collect()
    }

    /// The nodes a take drives when bound: the nodes already bound to it; else the active camera
    /// for a camera take, the selected objects (or the one it was recorded on) for a transform take.
    fn take_targets(&self, id: u64) -> Result<Vec<NodeId>, String> {
        let document = &self.world.document;
        let take = document.takes.get(id).ok_or(format!("No take {id}"))?;
        let bound = document.take_users(id);
        if !bound.is_empty() {
            return Ok(bound);
        }
        let targets = match take.kind() {
            TakeKind::Camera => document.active_camera.into_iter().collect(),
            TakeKind::Transform => {
                let selected = self.selected_objects();
                if selected.is_empty() {
                    take.source.into_iter().collect()
                } else {
                    selected
                }
            }
        };
        if targets.is_empty() {
            return Err(match take.kind() {
                TakeKind::Camera => "No active camera to drive".into(),
                TakeKind::Transform => "Select the objects the take should drive".into(),
            });
        }
        Ok(targets)
    }

    /// Bind take `id` to its targets ([`Self::take_targets`]).
    fn bind_targets(&mut self, id: u64) -> Result<(), String> {
        let targets = self.take_targets(id)?;
        self.world.bind_take(&targets, Some(id), None)
    }

    /// Unbind take `id` everywhere when it drives anything, else bind it to its targets.
    fn toggle_binding(&mut self, id: u64) -> Result<(), String> {
        let bound = self.world.document.take_users(id);
        if bound.is_empty() {
            return self.bind_targets(id);
        }
        self.world.bind_take(&bound, None, None)
    }

    /// New takes from the selected objects (transform takes) or, with none selected, from the
    /// active camera: their timeline animation over the work area (`animation`), or their
    /// current pose as a still take.
    fn takes_from_selection(&mut self, animation: bool) -> Result<(), String> {
        let document = &self.world.document;
        let fps = document.fps;
        let (first, playhead) = (f64::from(document.first), f64::from(self.world_ui.playhead));
        let name = if animation { "Animation" } else { "Pose" };
        let objects = self.selected_objects();
        let mut made = Vec::new();
        if objects.is_empty() {
            let take = if animation {
                Take::from_frames(
                    first,
                    fps,
                    document.active_camera,
                    document.camera_frames()?,
                )?
            } else {
                Take::still(self.scene.camera, playhead, fps, document.active_camera)?
            };
            made.push(take);
        }
        for node in objects {
            let take = if animation {
                Take::from_frames(first, fps, Some(node), document.trs_frames(node)?)?
            } else {
                Take::still(
                    document.authored_trs(node, playhead)?,
                    playhead,
                    fps,
                    Some(node),
                )?
            };
            made.push(take);
        }
        let made = made
            .into_iter()
            .map(|t| Take {
                name: name.into(),
                ..t
            })
            .collect();
        self.add_takes(made, "Recorded");
        Ok(())
    }

    /// Key take `id` onto the nodes it drives (else its targets) as one undoable edit each, then
    /// release the bindings: the keys now show the same motion. The playhead stays where it is.
    fn bake_take(&mut self, id: u64) {
        let Some(name) = self.world.document.takes.get(id).map(|t| t.name.clone()) else {
            return;
        };
        let result = self.take_targets(id).and_then(|targets| {
            let mut total = (0, 0, 0.0f64);
            for node in targets {
                let (samples, keys, error, _) =
                    self.world.bake_take(node, id, self.recorder_options)?;
                total = (total.0 + samples, total.1 + keys, total.2.max(error));
            }
            Ok(total)
        });
        self.evaluated_world = None;
        self.status = match result {
            Ok((samples, keys, error)) => {
                format!("Baked {name}: {samples} samples -> {keys} keys (sample error {error:.6})")
            }
            Err(error) => format!("Bake failed: {error}"),
        };
    }

    /// Export `Some(take)` as a clip file, or import one (None).
    fn take_dialog(&mut self, export: Option<u64>) {
        use crate::file_dialogs::{TAKE_EXPORT, TAKE_IMPORT};
        let key = if export.is_some() {
            TAKE_EXPORT
        } else {
            TAKE_IMPORT
        };
        let mut dialog = egui_file_dialog::FileDialog::new()
            .add_file_filter_extensions("Take clip", vec!["json"]);
        if let Some(take) = export.and_then(|id| self.world.document.takes.get(id)) {
            dialog = dialog
                .default_file_name(&format!("{}.take.json", crate::fs_name::stem(&take.name)));
        }
        dialog = self
            .world_ui
            .file_dialogs
            .prepare(dialog, key, None, "Take clip");
        if export.is_some() {
            dialog.save_file();
        } else {
            dialog.pick_file();
        }
        self.takes_view.picker = Some((export, dialog));
    }

    pub(super) fn update_take_picker(&mut self, ctx: &egui::Context) {
        let Some((export, picker)) = &mut self.takes_view.picker else {
            return;
        };
        picker.update(ctx);
        let key = if export.is_some() {
            crate::file_dialogs::TAKE_EXPORT
        } else {
            crate::file_dialogs::TAKE_IMPORT
        };
        self.world_ui.file_dialogs.observe(key, picker);
        let Some(path) = picker.take_picked() else {
            if matches!(
                picker.state(),
                egui_file_dialog::DialogState::Closed | egui_file_dialog::DialogState::Cancelled
            ) {
                self.takes_view.picker = None;
            }
            return;
        };
        let export = *export;
        self.takes_view.picker = None;
        let command = match export {
            Some(id) => {
                let Some(take) = self.world.document.takes.get(id) else {
                    return;
                };
                match take.to_clip(self.world.document.fps) {
                    Ok(text) => crate::io_service::Command::Write { path, text },
                    Err(error) => {
                        self.status = error;
                        return;
                    }
                }
            }
            None => crate::io_service::Command::OpenTake { path },
        };
        if let Err(error) = self.io.send(command) {
            self.status = error;
        }
    }
}
