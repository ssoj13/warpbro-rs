//! Playa's shared dope sheet widget, adapted to one fractal scene's parameter tracks.
use crate::{animation, scene::Scene};
use curves::CurveKind;
use egui_track_timeline::{
    KeyPos, Keyframe, PropLane, TimelineAction, TimelineConfig, TimelineModel, TimelineView, Track,
    TrackTimeline, WorkArea,
};
use serde_json::Value;

pub struct Timeline {
    pub frame: u32,
    pub playing: bool,
    pub looping: bool,
    view: TimelineView,
    selected: Vec<(String, f64)>,
    parameter: String,
    filter: String,
    elapsed: f64,
    fitted: bool,
    grid: egui_attr_grid::AttrGridState,
}
impl Default for Timeline {
    fn default() -> Self {
        Self {
            frame: 0,
            playing: false,
            looping: true,
            view: Default::default(),
            selected: Vec::new(),
            parameter: "/camera/yaw_degrees".into(),
            filter: String::new(),
            elapsed: 0.0,
            fitted: false,
            grid: Default::default(),
        }
    }
}
impl Timeline {
    pub fn reset(&mut self, scene: &Scene) {
        *self = Self::default();
        self.frame = scene.animation.first;
    }
    pub fn seek(&mut self, frame: u32, scene: &Scene) {
        self.frame = frame.clamp(
            scene.animation.first,
            scene.animation.last.max(scene.animation.first),
        );
        self.elapsed = 0.0;
    }
    pub fn advance(&mut self, dt: f64, scene: &Scene) {
        if !self.playing {
            self.elapsed = 0.0;
            return;
        }
        let a = &scene.animation;
        if !a.fps.is_finite() || a.fps <= 0.0 {
            self.playing = false;
            return;
        }
        let first = u64::from(a.first);
        let last = u64::from(a.last.max(a.first));
        self.elapsed += dt.clamp(0.0, 0.25) * a.fps;
        let steps = self.elapsed.floor() as u64;
        self.elapsed -= steps as f64;
        if steps == 0 {
            return;
        }
        let next = u64::from(self.frame) + steps;
        if next > last {
            if self.looping {
                self.frame = (first + (next - first) % (last - first + 1)) as u32;
            } else {
                self.frame = last as u32;
                self.playing = false;
            }
        } else {
            self.frame = next as u32;
        }
    }
    /// Stopwatch and diamond controls beside a discoverable parameter editor.
    pub fn parameter_ui(&mut self, ui: &mut egui::Ui, scene: &mut Scene) -> Result<(), String> {
        ui.horizontal(|ui| {
            ui.label("Find parameter");
            ui.text_edit_singleline(&mut self.filter);
        });
        let params = animation::parameters(scene);
        if !params.iter().any(|(p, _)| p == &self.parameter) {
            self.parameter = params.first().map(|(p, _)| p.clone()).unwrap_or_default();
        }
        egui::ComboBox::from_id_salt("animated_parameter")
            .selected_text(animation::label(&self.parameter))
            .width(ui.available_width().min(460.0))
            .show_ui(ui, |ui| {
                let filter = self.filter.to_lowercase();
                for (path, _) in &params {
                    let label = animation::label(path);
                    if label.to_lowercase().contains(&filter) {
                        ui.selectable_value(&mut self.parameter, path.clone(), label);
                    }
                }
            });
        let path = self.parameter.clone();
        let Some((_, value)) = params.into_iter().find(|(p, _)| p == &path) else {
            return Ok(());
        };
        let frame = f64::from(self.frame);
        let animated = scene.animation.tracks.contains_key(&path);
        let at_key = scene
            .animation
            .tracks
            .get(&path)
            .is_some_and(|t| t.keys().iter().any(|k| k.frame == frame));
        ui.horizontal(|ui| {
            if ui
                .selectable_label(animated, egui_phosphor::regular::TIMER)
                .on_hover_text("Enable animation; disable freezes the current value")
                .clicked()
            {
                if animated {
                    scene.animation.tracks.remove(&path);
                    self.selected.retain(|(p, _)| p != &path);
                } else {
                    scene.key_parameter(&path, frame);
                }
            }
            if ui.button("◀").on_hover_text("Previous key").clicked() {
                if let Some(t) = scene.animation.tracks.get(&path) {
                    if let Some(k) = t.keys().iter().rev().find(|k| k.frame < frame) {
                        self.seek(k.frame.round().max(0.0) as u32, scene);
                    }
                }
            }
            if ui
                .selectable_label(at_key, "◆")
                .on_hover_text("Add / remove key at the current frame")
                .clicked()
            {
                if at_key {
                    if let Some(t) = scene.animation.tracks.get_mut(&path) {
                        t.remove(frame);
                    }
                } else {
                    scene.key_parameter(&path, frame);
                }
            }
            if ui.button("▶").on_hover_text("Next key").clicked() {
                if let Some(t) = scene.animation.tracks.get(&path) {
                    if let Some(k) = t.keys().iter().find(|k| k.frame > frame) {
                        self.seek(k.frame.round().max(0.0) as u32, scene);
                    }
                }
            }
            ui.label(format!("Frame {}", self.frame));
            if animated {
                ui.weak("Edits create keys");
            }
        });
        if let Some(av) = to_widget(&value) {
            let mut field = [egui_attr_grid::AttrField::new(animation::label(&path), av)];
            for (_, edit) in
                egui_attr_grid::render_grid(ui, &mut field, &mut self.grid, &Default::default())
            {
                if let Some(value) = from_widget(edit) {
                    animation::set_parameter(scene, &path, value)?;
                }
            }
        }
        ui.small("Animate any scene parameter here. Its usual inspector control also updates the key at the playhead.");
        Ok(())
    }
    pub fn ui(&mut self, ui: &mut egui::Ui, scene: &mut Scene) {
        let first = scene.animation.first;
        let last = scene.animation.last.max(first);
        self.frame = self.frame.clamp(first, last);
        ui.horizontal_wrapped(|ui| {
            if ui.button("|◀").on_hover_text("First frame").clicked() {
                self.seek(first, scene);
            }
            if ui.button("◀").on_hover_text("Previous frame").clicked() {
                self.seek(self.frame.saturating_sub(1), scene);
            }
            if ui
                .button(if self.playing { "Pause" } else { "Play" })
                .clicked()
            {
                self.playing = !self.playing;
            }
            if ui.button("▶").on_hover_text("Next frame").clicked() {
                self.seek(self.frame.saturating_add(1), scene);
            }
            if ui.button("▶|").on_hover_text("Last frame").clicked() {
                self.seek(last, scene);
            }
            ui.checkbox(&mut self.looping, "Loop");
            if ui
                .add(
                    egui::DragValue::new(&mut self.frame)
                        .range(first..=last)
                        .prefix("Frame "),
                )
                .changed()
            {
                self.elapsed = 0.0;
            }
            ui.add(
                egui::DragValue::new(&mut scene.animation.fps)
                    .range(1.0..=240.0)
                    .prefix("FPS "),
            );
            ui.add(
                egui::DragValue::new(&mut scene.animation.first)
                    .range(0..=last)
                    .prefix("In "),
            );
            ui.add(
                egui::DragValue::new(&mut scene.animation.last)
                    .range(scene.animation.first..=1_000_000)
                    .prefix("Out "),
            );
            if ui.button("Fit").clicked() {
                self.fitted = false;
            }
        });
        let paths: Vec<_> = scene.animation.tracks.keys().cloned().collect();
        ui.horizontal(|ui| {
            let mut interpolation = CurveKind::Linear;
            egui::ComboBox::from_id_salt("key_interpolation")
                .selected_text("Interpolation")
                .show_ui(ui, |ui| {
                    for (name, kind) in [
                        ("Linear", CurveKind::Linear),
                        ("Smooth", CurveKind::Smooth),
                        ("Hold", CurveKind::Step),
                    ] {
                        if ui
                            .selectable_value(&mut interpolation, kind, name)
                            .clicked()
                        {
                            for (path, time) in &self.selected {
                                if let Some(track) = scene.animation.tracks.get_mut(path) {
                                    if let Some(mut key) =
                                        track.keys().iter().find(|k| k.frame == *time).cloned()
                                    {
                                        key.interpolation = kind;
                                        track.upsert(key);
                                    }
                                }
                            }
                        }
                    }
                });
            if ui
                .add_enabled(!self.selected.is_empty(), egui::Button::new("Delete keys"))
                .clicked()
            {
                self.delete_selected(scene);
            }
            ui.weak("Ctrl+click: add key · drag: move · Delete: remove · wheel: zoom · MMB: pan");
        });
        let config = TimelineConfig {
            row_height: 24.0,
            lane_height: 24.0,
            ..Default::default()
        };
        if !self.fitted {
            let width = (ui.available_width() - 260.0).max(50.0);
            self.view.zoom =
                (width / ((last - first + 1) as f32 * config.pixels_per_frame)).clamp(0.01, 100.0);
            self.view.pan_offset = -(first as f32) * self.view.ppf(&config) + 12.0;
            self.fitted = true;
        }
        let lanes = paths
            .iter()
            .map(|path| {
                let keys = scene.animation.tracks[path]
                    .keys()
                    .iter()
                    .map(|k| {
                        Keyframe::new(k.frame, k.value.as_f64().unwrap_or(0.0))
                            .selected(self.selected.contains(&(path.clone(), k.frame)))
                    })
                    .collect();
                PropLane::new(animation::label(path), keys)
            })
            .collect();
        let mut model = TimelineModel::new(
            scene.animation.fps as f32,
            vec![
                Track::new(&scene.name, vec![])
                    .with_lanes(lanes)
                    .expanded(true),
            ],
        );
        model.playhead = i64::from(self.frame);
        model.work_area = Some(WorkArea {
            start: i64::from(first),
            end: i64::from(last) + 1,
        });
        let mut actions = Vec::new();
        egui::ScrollArea::vertical()
            .id_salt("timeline_vertical")
            .show(ui, |ui| {
                ui.horizontal_top(|ui| {
                    ui.allocate_ui_with_layout(
                        egui::vec2(250.0, 44.0 + 24.0 * paths.len() as f32),
                        egui::Layout::top_down(egui::Align::Min),
                        |ui| {
                            ui.spacing_mut().item_spacing.y = 0.0;
                            ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Truncate);
                            ui.add_sized([250.0, 20.0], egui::Label::new("Parameters"));
                            ui.add_sized([250.0, 24.0], egui::Label::new(&scene.name));
                            for path in &paths {
                                if ui
                                    .add_sized(
                                        [250.0, 24.0],
                                        egui::Button::selectable(
                                            self.parameter == *path,
                                            animation::label(path),
                                        ),
                                    )
                                    .clicked()
                                {
                                    self.parameter = path.clone();
                                }
                            }
                        },
                    );
                    egui::ScrollArea::horizontal()
                        .id_salt("timeline_horizontal")
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            actions = TrackTimeline::new(config)
                                .show(ui, &mut self.view, &model)
                                .actions;
                        });
                });
            });
        for action in actions {
            self.action(action, &paths, scene);
        }
        if paths.is_empty() {
            ui.weak("Enable a parameter's stopwatch in Inspector → Animation to create a track.");
        }
    }
    fn delete_selected(&mut self, scene: &mut Scene) {
        for (path, frame) in self.selected.drain(..) {
            if let Some(t) = scene.animation.tracks.get_mut(&path) {
                t.remove(frame);
            }
        }
    }
    fn action(&mut self, action: TimelineAction, paths: &[String], scene: &mut Scene) {
        let identities = |keys: Vec<KeyPos>| {
            keys.into_iter()
                .filter(|k| k.track == 0)
                .filter_map(|k| paths.get(k.lane).map(|p| (p.clone(), k.frame)))
                .collect::<Vec<_>>()
        };
        match action {
            TimelineAction::Seek { frame } => {
                self.playing = false;
                self.seek(frame.max(0) as u32, scene);
            }
            TimelineAction::SetWorkArea { start, end } => {
                scene.animation.first = start.clamp(0, 1_000_000) as u32;
                scene.animation.last =
                    (end - 1).clamp(i64::from(scene.animation.first), 1_000_000) as u32;
            }
            TimelineAction::AddKey {
                track: 0,
                lane,
                frame,
            } => {
                if let Some(path) = paths.get(lane) {
                    if let Ok(sample) = scene.evaluated(frame) {
                        if let Some((_, v)) = animation::parameters(&sample)
                            .into_iter()
                            .find(|(p, _)| p == path)
                        {
                            scene
                                .animation
                                .tracks
                                .entry(path.clone())
                                .or_default()
                                .set(frame, v);
                        }
                    }
                }
            }
            TimelineAction::SelectKeys { keys, add } => {
                if !add {
                    self.selected.clear();
                }
                for k in identities(keys) {
                    if !self.selected.contains(&k) {
                        self.selected.push(k);
                    }
                }
            }
            TimelineAction::ClearKeySelection => self.selected.clear(),
            TimelineAction::MoveKeys { keys, delta } => {
                let keys = identities(keys);
                for path in paths {
                    let frames = keys
                        .iter()
                        .filter(|(p, _)| p == path)
                        .map(|(_, f)| *f)
                        .collect::<Vec<_>>();
                    if let Some(t) = scene.animation.tracks.get_mut(path) {
                        t.move_keys(&frames, delta);
                    }
                }
                self.selected = keys.into_iter().map(|(p, f)| (p, f + delta)).collect();
            }
            TimelineAction::DeleteKeys { keys } => {
                self.selected = identities(keys);
                self.delete_selected(scene);
            }
            _ => {}
        }
    }
}

fn to_widget(v: &Value) -> Option<egui_attr_grid::AttrValue> {
    use egui_attr_grid::AttrValue as A;
    match v {
        Value::Number(n) if n.is_u64() => Some(A::UInt(n.as_u64()? as u32)),
        Value::Number(n) => Some(A::Float(n.as_f64()? as f32)),
        Value::Bool(b) => Some(A::Bool(*b)),
        Value::String(s) => Some(A::Label(s.clone())), // enum choices stay in the typed inspector
        Value::Array(v) => Some(A::List(v.iter().map(to_widget).collect::<Option<_>>()?)),
        _ => None,
    }
}
fn from_widget(v: egui_attr_grid::AttrValue) -> Option<Value> {
    use egui_attr_grid::AttrValue as A;
    match v {
        A::UInt(v) => Some(Value::from(v)),
        A::Float(v) => Some(Value::from(v)),
        A::Bool(v) => Some(Value::from(v)),
        A::List(v) => Some(Value::Array(
            v.into_iter().map(from_widget).collect::<Option<_>>()?,
        )),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn playback_steps_and_wraps_inclusive_range() {
        let mut scene = Scene::preset(0);
        scene.animation.first = 10;
        scene.animation.last = 12;
        let mut timeline = Timeline::default();
        timeline.reset(&scene);
        timeline.playing = true;
        timeline.advance(2.0 / 24.0, &scene);
        assert_eq!(timeline.frame, 12);
        timeline.advance(1.0 / 24.0, &scene);
        assert_eq!(timeline.frame, 10);
        timeline.looping = false;
        timeline.advance(3.0 / 24.0, &scene);
        assert_eq!(timeline.frame, 12);
        assert!(!timeline.playing);
    }
}
