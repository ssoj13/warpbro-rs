//! The Takes panel: the document's recorded navigation takes (`crate::takes`) as a managed list.
//! Filter, multi-select (Ctrl / Shift click), pin, bind as the scene camera, play, rename,
//! copy / paste / duplicate (the system clipboard, so takes move between scenes), delete, export /
//! import clip files, takes made from the timeline animation or the current pose, non-destructive
//! per-take curve filters (Smooth, Simplify, Euler filter), and derived takes (trim, speed, split).
//! Bake to keys keys the bound take onto the camera as one undo step.
use super::App;
use crate::camera_recorder::Timing;
use crate::takes::{Take, TakeFilters, TakeKind};
use crate::world::NodeId;
use soluwell_config::icons;
use soluwell_config::phosphor::regular as ph;
use std::{
    collections::{BTreeSet, VecDeque},
    path::PathBuf,
};

/// Panel state, not saved with the scene.
pub(super) struct View {
    filter: String,
    pinned_only: bool,
    selection: BTreeSet<u64>,
    /// Shift-click range anchor.
    anchor: Option<u64>,
    /// The take whose name is being edited.
    rename: Option<(u64, String)>,
    /// Edit parameters of the selected take: trim in / out (seconds), speed.
    trim: (f64, f64),
    speed: f64,
    /// The take `trim` was initialised for: a new selection resets it to the whole take.
    trim_of: Option<u64>,
    /// Export (Some take) or import (None) dialog, bound to the document that opened it.
    picker: Option<(Option<u64>, NodeId, egui_file_dialog::FileDialog)>,
    /// Origin documents for the I/O worker's FIFO take-import completions.
    pending_imports: VecDeque<NodeId>,
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
            trim_of: None,
            picker: None,
            pending_imports: VecDeque::new(),
        }
    }
}

/// Concrete take intents, published by the UI and applied only by the application event handler.
#[derive(Clone)]
pub(crate) enum Action {
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
    /// A new take derived from `source` (trim, speed, split).
    Derive(u64, Derive),
    SetStart(u64, f64),
    SetTiming(u64, Timing),
    SetFilters(u64, TakeFilters),
    Bake(u64),
    /// Resolved before publishing; the handler never reads a later UI selection.
    Selection {
        ids: BTreeSet<u64>,
        anchor: Option<u64>,
    },
    BindTargets {
        id: u64,
        targets: Result<Vec<NodeId>, String>,
        bind: bool,
    },
    PlayTargets(u64, Result<Vec<NodeId>, String>),
    BakeTargets(u64, Result<Vec<NodeId>, String>),
    Create {
        animation: bool,
        objects: Vec<NodeId>,
        camera: Option<NodeId>,
        camera_pose: crate::scene::Camera,
    },
    Imported(Result<Take, String>),
    Paste(String),
    Recording(bool),
    MinSeconds(f64),
    RecorderOptions(crate::camera_recorder::Options),
    Filter(String),
    PinnedOnly(bool),
    PickerComplete {
        export: Option<u64>,
        path: PathBuf,
    },
    PickerHistory {
        export: bool,
        directory: Option<PathBuf>,
        filter: Option<String>,
    },
}
#[derive(Clone)]
pub(crate) enum Derive {
    Trim(f64, f64),
    Speed(f64),
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
            self.emit_take(action);
        }
    }

    /// The takes passing the filter, newest first: the order rows are drawn and ranges select in.
    fn visible_takes(&self) -> Vec<u64> {
        let needle = self.takes_view.filter.to_lowercase();
        let min_secs = self.min_take_secs;
        self.world
            .document
            .takes
            .list()
            .iter()
            .rev()
            .filter(|t| !self.takes_view.pinned_only || t.pinned)
            .filter(|t| {
                min_secs <= crate::takes::MIN_SECS
                    || t.pinned
                    || t.duration() >= min_secs
                    || t.duration() < crate::takes::MIN_SECS
                    || self.takes_view.selection.contains(&t.id)
                    || !self.world.document.take_users(t.id).is_empty()
            })
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
        let mut filter = self.takes_view.filter.clone();
        let mut pinned_only = self.takes_view.pinned_only;
        ui.horizontal_wrapped(|ui| {
            if ui.add(
                egui::TextEdit::singleline(&mut filter)
                    .hint_text(format!("{} Filter", icons::SEARCH))
                    .desired_width(110.0),
            ).changed() {
                actions.push(Action::Filter(filter));
            }
            if ui.toggle_value(&mut pinned_only, ph::PUSH_PIN)
                .on_hover_text("Pinned takes only").changed() {
                actions.push(Action::PinnedOnly(pinned_only));
            }
            ui.separator();
            // Record / pause: whether navigation and manipulator drags make takes.
            let recording = self.record_takes;
            if ui
                .selectable_label(recording, if recording { ph::RECORD } else { ph::PAUSE })
                .on_hover_text(if recording {
                    "Recording: every navigation gesture and manipulator drag becomes a take (click: pause)"
                } else {
                    "Paused: no takes are recorded (click: record)"
                })
                .clicked()
            {
                actions.push(Action::Recording(!recording));
            }
            let mut min_take_secs = self.min_take_secs;
            if ui
                .add(
                    egui::DragValue::new(&mut min_take_secs)
                        .range(crate::takes::MIN_SECS..=600.0)
                        .speed(0.05)
                        .max_decimals(2)
                        .prefix("> ")
                        .suffix(" s"),
                )
                .on_hover_text("Keep only takes longer than this duration (seconds)")
                .changed()
            {
                actions.push(Action::MinSeconds(min_take_secs));
            }
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
                        if ui
                            .selectable_label(take.filters.smooth, ph::WAVE_SINE)
                            .on_hover_text("Smooth: zero-phase Butterworth low-pass filter (non-destructive)")
                            .clicked()
                        {
                            let mut f = take.filters;
                            f.smooth = !f.smooth;
                            actions.push(Action::SetFilters(id, f));
                        }
                        if ui
                            .selectable_label(take.filters.simplify, ph::LINE_SEGMENTS)
                            .on_hover_text("Simplify: reduce curve keys within error tolerance (non-destructive)")
                            .clicked()
                        {
                            let mut f = take.filters;
                            f.simplify = !f.simplify;
                            actions.push(Action::SetFilters(id, f));
                        }
                        if ui
                            .selectable_label(take.filters.euler, ph::ARROWS_CLOCKWISE)
                            .on_hover_text("Euler filter: keep rotation angles continuous across ±180° flips and turns (non-destructive)")
                            .clicked()
                        {
                            let mut f = take.filters;
                            f.euler = !f.euler;
                            actions.push(Action::SetFilters(id, f));
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

                let mut f = take.filters;
                if f.smooth {
                    ui.label("Smooth");
                    ui.horizontal(|ui| {
                        ui.add(
                            egui::DragValue::new(&mut f.cutoff)
                                .range(0.01..=100.0)
                                .speed(0.05)
                                .prefix("cutoff ")
                                .suffix(" Hz"),
                        )
                        .on_hover_text(
                            "Motion faster than this many cycles per second is removed, slower kept",
                        );
                        ui.add(
                            egui::DragValue::new(&mut f.order)
                                .range(2..=16)
                                .speed(0.1)
                                .prefix("order "),
                        )
                        .on_hover_text("Filter poles: higher cuts sharper at the cutoff");
                        ui.add(
                            egui::DragValue::new(&mut f.blend)
                                .range(0.0..=60.0)
                                .speed(0.01)
                                .prefix("blend ")
                                .suffix(" s"),
                        )
                        .on_hover_text("Eases from the original at each end into the filtered motion");
                    });
                    f.order += f.order % 2;
                    if ui
                        .button("Reset")
                        .on_hover_text("Reset Smooth parameters to defaults")
                        .clicked()
                    {
                        let d = TakeFilters::default();
                        f.cutoff = d.cutoff;
                        f.order = d.order;
                        f.blend = d.blend;
                    }
                    ui.end_row();
                }

                if f.simplify {
                    ui.label("Simplify");
                    ui.horizontal(|ui| {
                        ui.add(
                            egui::DragValue::new(&mut f.tolerance)
                                .range(0.0..=1.0e3)
                                .speed(0.001)
                                .prefix("error "),
                        )
                        .on_hover_text("Largest allowed change of the curve, in the channel's units");
                        ui.add(
                            egui::DragValue::new(&mut f.max_keys)
                                .range(0..=100_000)
                                .prefix("max keys "),
                        )
                        .on_hover_text(
                            "0: no limit. Otherwise the smallest error that fits in this many keys",
                        );
                    });
                    if ui
                        .button("Reset")
                        .on_hover_text("Reset Simplify parameters to defaults")
                        .clicked()
                    {
                        let d = TakeFilters::default();
                        f.tolerance = d.tolerance;
                        f.max_keys = d.max_keys;
                    }
                    ui.end_row();
                }

                if f != take.filters {
                    actions.push(Action::SetFilters(id, f));
                }

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
        let mut options = self.recorder_options;
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
        if options != self.recorder_options {
            actions.push(Action::RecorderOptions(options));
        }
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

    /// Resolve selection-sensitive intents at publication, before another event can change it.
    pub(super) fn capture_take_action(&self, action: Action) -> Action {
        match action {
            Action::Select { id, ctrl, shift } => self.take_selection_action(id, ctrl, shift),
            Action::Bind(id) => Action::BindTargets {
                id,
                targets: self.take_targets(id),
                bind: self.world.document.take_users(id).is_empty(),
            },
            Action::Play(id) => Action::PlayTargets(id, self.take_targets(id)),
            Action::Bake(id) => Action::BakeTargets(id, self.take_targets(id)),
            Action::FromAnimation | Action::FromPose => Action::Create {
                animation: matches!(action, Action::FromAnimation),
                objects: self.selected_objects(),
                camera: self.world.document.active_camera,
                camera_pose: self.scene.camera,
            },
            action => action,
        }
    }

    pub(super) fn handle_take_action(
        &mut self,
        action: Action,
        frame: f64,
        ctx: &egui::Context,
    ) -> Result<serde_json::Value, String> {
        let fps = self.world.document.fps;
        let result: Result<(), String> = match action {
            Action::Select { .. }
            | Action::Bind(_)
            | Action::Play(_)
            | Action::Bake(_)
            | Action::FromAnimation
            | Action::FromPose => Err("Take intent was not captured before publishing".into()),
            Action::Selection { ids, anchor } => {
                for id in &ids {
                    if self.world.document.takes.get(*id).is_none() {
                        return Err(format!("No take {id}"));
                    }
                }
                self.takes_view.selection = ids;
                self.takes_view.anchor = anchor;
                Ok(())
            }
            Action::Pin(ids, pinned) => ids
                .into_iter()
                .try_for_each(|id| self.world.edit_take(id, |t| t.pinned = pinned).map(drop)),
            Action::BindTargets { id, targets, bind } => {
                self.world.bind_take(&targets?, bind.then_some(id), None)
            }
            Action::PlayTargets(id, targets) => self.play_take(id, targets?),
            Action::Rename(id, name) => {
                self.takes_view.rename = None;
                let name = name.trim().to_owned();
                if name.is_empty() {
                    Ok(())
                } else {
                    self.world.edit_take(id, |t| t.name = name).map(drop)
                }
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
            Action::Copy(ids) => self.copy_takes(&ids, ctx),
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
                self.world
                    .document
                    .takes
                    .get(id)
                    .ok_or_else(|| format!("No take {id}"))?;
                self.take_dialog(Some(id));
                Ok(())
            }
            Action::Import => {
                self.take_dialog(None);
                Ok(())
            }
            Action::Create {
                animation,
                objects,
                camera,
                camera_pose,
            } => self.takes_from_targets(animation, objects, camera, camera_pose, frame),
            Action::Imported(result) => {
                let take = result.map_err(|error| format!("Take import failed: {error}"))?;
                let name = take.name.clone();
                self.world.add_take(take);
                self.status = format!("Imported take {name}");
                Ok(())
            }
            Action::Derive(id, derive) => {
                let take = self
                    .world
                    .document
                    .takes
                    .get(id)
                    .cloned()
                    .ok_or_else(|| format!("No take {id}"))?;
                let named = |t: Take, suffix: &str| Take {
                    name: format!("{} {suffix}", take.name),
                    pinned: false,
                    ..t
                };
                let playhead = take.seconds_at(
                    frame,
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
                    Derive::Split => take
                        .split(playhead, fps)
                        .map(|(a, b)| vec![named(a, "A"), named(b, "B")]),
                }
                .map(|takes| self.add_takes(takes, "Made"))
            }
            Action::SetStart(id, start) => self.world.edit_take(id, |t| t.start = start).map(drop),
            Action::SetTiming(id, timing) => {
                self.world.edit_take(id, |t| t.timing = timing).map(drop)
            }
            Action::SetFilters(id, filters) => {
                self.world.edit_take(id, |t| t.filters = filters).map(drop)
            }
            Action::BakeTargets(id, targets) => self.bake_take(id, targets?),
            Action::Paste(text) => match crate::takes::from_clipboard(&text) {
                Some(takes) => takes.map(|takes| self.add_takes(takes, "Pasted")),
                None => Err("Clipboard does not contain WarpBro takes".into()),
            },
            Action::Recording(recording) => {
                self.record_takes = recording;
                if !recording {
                    self.takes_recorder.cancel();
                    self.drag_recorder.cancel();
                }
                Ok(())
            }
            Action::MinSeconds(seconds) => {
                if !seconds.is_finite() || !(crate::takes::MIN_SECS..=600.0).contains(&seconds) {
                    return Err("Invalid minimum take duration".into());
                }
                self.min_take_secs = seconds;
                self.takes_recorder.min_secs = seconds;
                self.drag_recorder.min_secs = seconds;
                Ok(())
            }
            Action::RecorderOptions(options) => {
                if !options.tolerance.is_finite() || !(0.0..=1.0).contains(&options.tolerance) {
                    return Err("Invalid take reduction tolerance".into());
                }
                self.recorder_options = options;
                Ok(())
            }
            Action::Filter(filter) => {
                self.takes_view.filter = filter;
                Ok(())
            }
            Action::PinnedOnly(pinned_only) => {
                self.takes_view.pinned_only = pinned_only;
                Ok(())
            }
            Action::PickerComplete { export, path } => {
                let command = match export {
                    Some(id) => {
                        let take = self
                            .world
                            .document
                            .takes
                            .get(id)
                            .ok_or_else(|| format!("No take {id}"))?;
                        crate::io_service::Command::Write {
                            path,
                            text: take.to_clip(self.world.document.fps)?,
                        }
                    }
                    None => crate::io_service::Command::OpenTake { path },
                };
                self.io.send(command)?;
                if export.is_none() {
                    self.takes_view
                        .pending_imports
                        .push_back(NodeId(self.world.document.graph.id));
                }
                Ok(())
            }
            Action::PickerHistory {
                export,
                directory,
                filter,
            } => {
                let key = if export {
                    crate::file_dialogs::TAKE_EXPORT
                } else {
                    crate::file_dialogs::TAKE_IMPORT
                };
                if let Some(directory) = directory {
                    self.world_ui
                        .file_dialogs
                        .directories
                        .insert(key.into(), directory);
                }
                self.world_ui
                    .file_dialogs
                    .filters
                    .insert(key.into(), filter);
                Ok(())
            }
        };
        result.map(|()| serde_json::Value::Null)
    }

    /// Plain click selects one take, Ctrl toggles it, Shift extends from the anchor over the
    /// visible (filtered) order.
    fn take_selection_action(&self, id: u64, ctrl: bool, shift: bool) -> Action {
        let mut selection = self.takes_view.selection.clone();
        if shift && let Some(anchor) = self.takes_view.anchor {
            let visible = self.visible_takes();
            let (a, b) = (
                visible.iter().position(|t| *t == anchor),
                visible.iter().position(|t| *t == id),
            );
            if let (Some(a), Some(b)) = (a, b) {
                if !ctrl {
                    selection.clear();
                }
                selection.extend(&visible[a.min(b)..=a.max(b)]);
                return Action::Selection {
                    ids: selection,
                    anchor: Some(anchor),
                };
            }
        }
        if ctrl {
            if !selection.remove(&id) {
                selection.insert(id);
            }
        } else {
            selection.clear();
            selection.insert(id);
        }
        Action::Selection {
            ids: selection,
            anchor: Some(id),
        }
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

    fn copy_takes(&mut self, ids: &[u64], ctx: &egui::Context) -> Result<(), String> {
        let takes: Vec<&Take> = ids
            .iter()
            .filter_map(|id| self.world.document.takes.get(*id).map(|t| &**t))
            .collect();
        if takes.is_empty() {
            return Ok(());
        }
        let text = crate::takes::to_clipboard(&takes, self.world.document.fps)?;
        let count = takes.len();
        ctx.copy_text(text);
        self.status = format!("Copied {count} take(s)");
        Ok(())
    }

    /// Ctrl+C / Ctrl+D / Delete while the pointer is over the Takes panel.
    pub(super) fn takes_hotkeys(&mut self, ctx: &egui::Context) {
        use crate::hotkeys::{self, Command as Hotkey, Scope};
        let selected = self.selected_takes();
        if hotkeys::consume(ctx, Scope::Global, Hotkey::Duplicate) && !selected.is_empty() {
            self.emit_take(Action::Duplicate(selected.clone()));
        }
        if hotkeys::consume(ctx, Scope::Takes, Hotkey::Delete) && !selected.is_empty() {
            self.emit_take(Action::Delete(selected.clone()));
        }
        if hotkeys::take_copy(ctx, !selected.is_empty()) {
            self.emit_take(Action::Copy(selected));
        }
    }

    /// Paste takes copied in any WarpBro scene (Ctrl+V anywhere: only takes text is claimed).
    pub(super) fn paste_takes(&mut self, ctx: &egui::Context) {
        let Some(text) =
            crate::hotkeys::take_paste(ctx, |text| crate::takes::from_clipboard(text).is_some())
        else {
            return;
        };
        self.emit_take(Action::Paste(text));
    }

    /// The placed objects among the selected nodes: what a transform take binds to and is made of.
    fn selected_objects(&self) -> Vec<NodeId> {
        self.world
            .targets(None)
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

    /// Bind take `id` and play from where it begins playing: the first frame of its bar on the
    /// timeline (`WorldDocument::take_span`: its timing and the node's `/take/offset` and
    /// speed), else, when none of it falls in the work area, its play start.
    fn play_take(&mut self, id: u64, targets: Vec<NodeId>) -> Result<(), String> {
        self.world.bind_take(&targets, Some(id), None)?;
        let document = &self.world.document;
        let span = match targets.first() {
            Some(node) => document.take_span(*node)?,
            None => None,
        };
        let start = match (span, document.takes.get(id)) {
            (Some((start, _)), _) => start as f64,
            (None, Some(take)) => take.play_start(document.first),
            (None, None) => return Err(format!("No take {id}")),
        };
        let (first, last) = (document.first, document.last);
        self.world_ui
            .seek((start.round().max(0.0) as u32).clamp(first, last));
        self.world_ui.playing = true;
        Ok(())
    }

    /// New takes from the selected objects (transform takes) or, with none selected, from the
    /// active camera: their timeline animation over the work area (`animation`), or their
    /// current pose as a still take.
    fn takes_from_targets(
        &mut self,
        animation: bool,
        objects: Vec<NodeId>,
        camera: Option<NodeId>,
        camera_pose: crate::scene::Camera,
        frame: f64,
    ) -> Result<(), String> {
        let document = &self.world.document;
        let fps = document.fps;
        let first = f64::from(document.first);
        let name = if animation { "Animation" } else { "Pose" };
        let mut made = Vec::new();
        if objects.is_empty() {
            let take = if animation {
                let mut camera_document = document.clone();
                camera_document.active_camera = camera;
                Take::from_frames(first, fps, camera, camera_document.camera_frames()?)?
            } else {
                Take::still(camera_pose, frame, fps, camera)?
            };
            made.push(take);
        }
        for node in objects {
            let take = if animation {
                Take::from_frames(first, fps, Some(node), document.trs_frames(node)?)?
            } else {
                Take::still(document.authored_trs(node, frame)?, frame, fps, Some(node))?
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
    fn bake_take(&mut self, id: u64, targets: Vec<NodeId>) -> Result<(), String> {
        let name = self
            .world
            .document
            .takes
            .get(id)
            .map(|t| t.name.clone())
            .ok_or_else(|| format!("No take {id}"))?;
        let result: Result<_, String> = (|| {
            let mut total = (0, 0, 0.0f64);
            for node in targets {
                let (samples, keys, error, _) =
                    self.world.bake_take(node, id, self.recorder_options)?;
                total = (total.0 + samples, total.1 + keys, total.2.max(error));
            }
            Ok(total)
        })();
        self.evaluated_world = None;
        let (samples, keys, error) = result.map_err(|error| format!("Bake failed: {error}"))?;
        self.status =
            format!("Baked {name}: {samples} samples -> {keys} keys (sample error {error:.6})");
        Ok(())
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
        self.takes_view.picker = Some((export, NodeId(self.world.document.graph.id), dialog));
    }

    pub(super) fn update_take_picker(&mut self, ctx: &egui::Context) {
        let Some((export, document, picker)) = &mut self.takes_view.picker else {
            return;
        };
        let document = *document;
        picker.update(ctx);
        let key = if export.is_some() {
            crate::file_dialogs::TAKE_EXPORT
        } else {
            crate::file_dialogs::TAKE_IMPORT
        };
        let mut actions = Vec::new();
        if !matches!(picker.state(), egui_file_dialog::DialogState::Closed) {
            let directory = picker.directory().map(std::path::Path::to_path_buf);
            let filter = picker
                .selected_file_filter()
                .map(|filter| filter.name.clone());
            if directory.as_ref().is_some_and(|directory| {
                self.world_ui.file_dialogs.directories.get(key) != Some(directory)
            }) || self.world_ui.file_dialogs.filters.get(key) != Some(&filter)
            {
                actions.push(Action::PickerHistory {
                    export: export.is_some(),
                    directory,
                    filter,
                });
            }
        }
        if let Some(path) = picker.take_picked() {
            actions.push(Action::PickerComplete {
                export: *export,
                path,
            });
            self.takes_view.picker = None;
        } else if matches!(
            picker.state(),
            egui_file_dialog::DialogState::Closed | egui_file_dialog::DialogState::Cancelled
        ) {
            self.takes_view.picker = None;
        }
        for action in actions {
            self.emit_take_for_document(action, document);
        }
    }

    /// Complete imports through the same bus, retaining the request's original document.
    pub(super) fn take_import_result(&mut self, result: Result<Take, String>, ctx: &egui::Context) {
        let (document, result) = match self.takes_view.pending_imports.pop_front() {
            Some(document) => (document, result),
            None => (
                NodeId(self.world.document.graph.id),
                Err("Unexpected take import completion without a request".into()),
            ),
        };
        self.emit_retained(
            crate::events::UiCommand {
                document: Some(document),
                frame: None,
                camera: None,
                action: crate::events::Action::Take(Action::Imported(result)),
            },
            ctx,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn camera_take(app: &mut App) -> u64 {
        let take = Take::still(
            app.scene.camera,
            f64::from(app.world.document.first),
            app.world.document.fps,
            app.world.document.active_camera,
        )
        .unwrap();
        app.world.add_take(take)
    }

    #[test]
    fn take_intents_wait_for_event_dispatch() {
        let mut app = App::new();
        let ctx = egui::Context::default();
        let id = camera_take(&mut app);
        let before_name = app.world.document.takes.get(id).unwrap().name.clone();
        let before_secs = app.min_take_secs;
        let before_recording = app.record_takes;
        app.emit_take(Action::Rename(id, "Event take".into()));
        app.emit_take(Action::Recording(!before_recording));
        app.emit_take(Action::MinSeconds(before_secs + 1.0));
        app.emit_take(Action::Select {
            id,
            ctrl: false,
            shift: false,
        });
        assert_eq!(app.world.document.takes.get(id).unwrap().name, before_name);
        assert_eq!(app.min_take_secs, before_secs);
        assert_eq!(app.record_takes, before_recording);
        assert!(app.takes_view.selection.is_empty());
        app.handle_events(&ctx);
        assert_eq!(app.world.document.takes.get(id).unwrap().name, "Event take");
        assert_eq!(app.min_take_secs, before_secs + 1.0);
        assert_eq!(app.record_takes, !before_recording);
        assert_eq!(app.takes_view.selection, BTreeSet::from([id]));
    }

    #[test]
    fn import_completion_keeps_its_origin_document() {
        let mut app = App::new();
        let ctx = egui::Context::default();
        let take = Take::still(app.scene.camera, 0.0, app.world.document.fps, None).unwrap();
        let count = app.world.document.takes.list().len();
        let origin = NodeId(app.world.document.graph.id);
        app.takes_view.pending_imports.push_back(origin);
        app.world.document.graph.id = NodeId::new().uuid();
        app.take_import_result(Ok(take.clone()), &ctx);
        assert!(app.takes_view.pending_imports.is_empty());
        assert_eq!(app.world.document.takes.list().len(), count);
        app.handle_events(&ctx);
        assert_eq!(app.world.document.takes.list().len(), count);
        // An unrequested completion must report a failure instead of adopting the current document.
        app.take_import_result(Ok(take), &ctx);
        app.handle_events(&ctx);
        assert_eq!(app.world.document.takes.list().len(), count);
        assert!(app.status.contains("Unexpected take import completion"));
    }

    #[test]
    fn queued_take_play_uses_the_camera_captured_at_publication() {
        let mut app = App::new();
        let ctx = egui::Context::default();
        let camera = app.world.document.active_camera.unwrap();
        let id = camera_take(&mut app);
        app.emit_take(Action::Play(id));
        assert_eq!(app.world.document.take_clip(camera).unwrap(), None);
        app.world.document.active_camera = None;
        app.handle_events(&ctx);
        assert_eq!(app.world.document.take_clip(camera).unwrap(), Some(id));
        assert_eq!(app.world.document.active_camera, None);
        assert!(app.world_ui.playing);
    }
}
