//! The Takes panel: every recorded navigation gesture of the document (`crate::takes`), the
//! binding of the current take as the scene camera, Bake to keys, and clip export / import.
use super::App;
use crate::camera_recorder::{Recording, Timing};
use egui_phosphor::regular as ph;
use egui_widgets_config::icons;

/// What a row asked for; applied after the list is drawn (the list borrows the document).
enum Action {
    Pin(u64, bool),
    Bind(Option<u64>),
    Play(u64),
    Rename(u64, String),
    Export(u64),
    Delete(u64),
}

impl App {
    pub(super) fn takes_panel(&mut self, ui: &mut egui::Ui) {
        let takes = &self.world.document.takes;
        let current = takes.current_id();
        let mut actions = Vec::new();
        if takes.list().is_empty() {
            ui.label("Every viewport navigation gesture is recorded here as a take.");
        }
        egui::ScrollArea::vertical()
            .auto_shrink([false, true])
            .max_height(ui.available_height() - 150.0)
            .show(ui, |ui| {
                for take in takes.list().iter().rev() {
                    ui.horizontal(|ui| {
                        let pin = if take.pinned {
                            ph::PUSH_PIN
                        } else {
                            ph::PUSH_PIN_SIMPLE
                        };
                        if ui
                            .selectable_label(take.pinned, pin)
                            .on_hover_text(
                                "Pinned takes are never dropped (the 20 newest unpinned are kept)",
                            )
                            .clicked()
                        {
                            actions.push(Action::Pin(take.id, !take.pinned));
                        }
                        let bound = current == Some(take.id);
                        if ui
                            .selectable_label(
                                bound,
                                if bound { icons::LINK } else { ph::LINK_BREAK },
                            )
                            .on_hover_text("Drive the scene camera with this take")
                            .clicked()
                        {
                            actions.push(Action::Bind((!bound).then_some(take.id)));
                        }
                        if ui
                            .button(icons::PLAY)
                            .on_hover_text("Bind and play from its start")
                            .clicked()
                        {
                            actions.push(Action::Play(take.id));
                        }
                        match &mut self.take_rename {
                            Some((id, name)) if *id == take.id => {
                                let edit =
                                    ui.add(egui::TextEdit::singleline(name).desired_width(120.0));
                                if edit.lost_focus() {
                                    actions.push(Action::Rename(*id, name.clone()));
                                } else {
                                    edit.request_focus();
                                }
                            }
                            _ => {
                                if ui
                                    .add(egui::Label::new(&take.name).sense(egui::Sense::click()))
                                    .on_hover_text("Double-click to rename")
                                    .double_clicked()
                                {
                                    self.take_rename = Some((take.id, take.name.clone()));
                                }
                            }
                        }
                        ui.weak(format!("{:.1} s", take.duration()));
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.button(icons::TRASH).on_hover_text("Delete").clicked() {
                                actions.push(Action::Delete(take.id));
                            }
                            if ui.button(ph::EXPORT).on_hover_text("Export clip").clicked() {
                                actions.push(Action::Export(take.id));
                            }
                        });
                    });
                }
            });
        for action in actions {
            self.take_action(action);
        }
        ui.separator();
        self.take_footer(ui);
    }

    fn take_action(&mut self, action: Action) {
        let result = match action {
            Action::Pin(id, pinned) => {
                self.world.edit_take(id, |t| t.pinned = pinned);
                Ok(())
            }
            Action::Bind(id) => self.world.set_current_take(id),
            Action::Play(id) => self.world.set_current_take(Some(id)).map(|()| {
                if let Some(take) = self.world.document.takes.get(id) {
                    let (first, last) = (self.world.document.first, self.world.document.last);
                    self.world_ui
                        .seek((take.start.round().max(0.0) as u32).clamp(first, last));
                }
                self.world_ui.playing = true;
            }),
            Action::Rename(id, name) => {
                self.take_rename = None;
                let name = name.trim().to_owned();
                if !name.is_empty() {
                    self.world.edit_take(id, |t| t.name = name);
                }
                Ok(())
            }
            Action::Export(id) => {
                self.take_dialog(Some(id));
                Ok(())
            }
            Action::Delete(id) => {
                self.world.delete_take(id);
                Ok(())
            }
        };
        self.evaluated_world = None;
        if let Err(error) = result {
            self.status = error;
        }
    }

    /// Timing of the current take, the bake channels and tolerance, Bake to keys and Import.
    fn take_footer(&mut self, ui: &mut egui::Ui) {
        let current = self
            .world
            .document
            .takes
            .current()
            .map(|t| (t.id, t.timing));
        ui.add_enabled_ui(current.is_some(), |ui| {
            let (id, mut timing) = current.unwrap_or((0, Timing::KeepSpeed));
            ui.horizontal(|ui| {
                ui.radio_value(&mut timing, Timing::KeepSpeed, "Keep speed")
                    .on_hover_text("Plays at recorded speed from its start frame; a bake extends the work area");
                ui.radio_value(&mut timing, Timing::FitWorkArea, "Fit work area");
            });
            if current.is_some_and(|(_, old)| old != timing) {
                self.world.edit_take(id, |t| t.timing = timing);
                self.evaluated_world = None;
            }
        });
        let options = &mut self.recorder_options;
        ui.horizontal_wrapped(|ui| {
            ui.checkbox(&mut options.transform, "Transform");
            ui.checkbox(&mut options.focus, "Focus");
            ui.checkbox(&mut options.zoom, "Zoom");
            ui.checkbox(&mut options.f_number, "f-number");
        });
        ui.horizontal(|ui| {
            ui.label("Reduction error");
            ui.add(
                egui::DragValue::new(&mut options.tolerance)
                    .range(0.0..=1.0)
                    .speed(0.0001),
            )
            .on_hover_text("Maximum absolute error at the samples, in each channel's units");
        });
        ui.horizontal(|ui| {
            if ui
                .add_enabled(
                    current.is_some(),
                    egui::Button::new(format!("{} Bake to keys", icons::RECORD)),
                )
                .on_hover_text(
                    "Key the current take onto the camera (one undo); keys outside its span remain",
                )
                .clicked()
                && let Some((id, _)) = current
            {
                self.bake_take(id);
            }
            if ui
                .button(format!("{} Import", ph::DOWNLOAD_SIMPLE))
                .clicked()
            {
                self.take_dialog(None);
            }
        });
    }

    /// Key the take onto the active camera as one undoable edit, then release the binding: the
    /// keys now show the same motion. The playhead stays where it is.
    fn bake_take(&mut self, id: u64) {
        self.world.finish_edit();
        let Some(take) = self.world.document.takes.get(id).cloned() else {
            return;
        };
        let result = Recording::from_take(
            &self.world.document,
            self.world.revision(),
            &take,
            self.recorder_options,
        )
        .and_then(|recording| self.world.record_camera(&recording))
        .and_then(|summary| self.world.set_current_take(None).map(|()| summary));
        self.evaluated_world = None;
        self.status = match result {
            Ok((samples, keys, error, _)) => format!(
                "Baked {}: {samples} samples -> {keys} keys (sample error {error:.6})",
                take.name
            ),
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
        self.take_picker = Some((export, dialog));
    }

    pub(super) fn update_take_picker(&mut self, ctx: &egui::Context) {
        let Some((export, picker)) = &mut self.take_picker else {
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
                self.take_picker = None;
            }
            return;
        };
        let export = *export;
        self.take_picker = None;
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
