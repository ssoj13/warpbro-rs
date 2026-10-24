//! Accepted application intents and their single mutation boundary.
use super::*;
use serde_json::{Value, json};

#[derive(Clone)]
pub(crate) enum AppAction {
    SettingsPatch {
        before: Value,
        after: Value,
    },
    PatchSettings(Value),
    SceneDocument(WorldDocument),
    GalleryLoad(usize),
    ApplyLibraryMaterial {
        index: usize,
    },
    CameraPatch(Value),
    CameraOrbit {
        yaw_degrees: f32,
        pitch_degrees: f32,
    },
    CameraPan {
        dx_px: f32,
        dy_px: f32,
    },
    CameraZoom {
        factor: f32,
    },
    CameraFly {
        translation: [f32; 3],
        rotation_degrees: [f32; 3],
    },
    ColourPreset(usize),
    SetPaused(bool),
    Exit,
    OpenSettings(SettingsPage),
    OpenPanel(dock::Panel),
    ShowUi(bool),
    History(History),
    NewScene,
    LoadScene {
        scene: Box<Scene>,
        new_document: bool,
        card: Option<Card>,
    },
    SceneDialog(crate::templates::FileAction),
    SceneFile {
        path: PathBuf,
        action: crate::templates::FileAction,
    },
    SaveScene,
    RefreshTemplates,
    BuiltinTemplate(usize),
    SaveBookmark,
    DeleteBookmark(PathBuf),
    SaveFrame {
        monitor: Monitor,
        file: Option<crate::render_service::FrameFile>,
    },
    WindowShot {
        files: crate::window_shot::WindowFiles,
        monitor: Monitor,
    },
    CopyScene,
    CopyNodes(Vec<crate::world::NodeId>),
    PasteNodes(String),
    SelectMaterial(crate::world::NodeId),
    InspectNode(crate::world::NodeId),
    AssignMaterial {
        material: crate::world::NodeId,
        targets: Vec<crate::world::NodeId>,
    },
    CreateMaterial {
        material: Material,
        name: String,
        targets: Option<Vec<crate::world::NodeId>>,
        open: bool,
    },
    Edit {
        command: crate::world::WorldCommand,
        gesture: Option<u64>,
    },
    RefreshMaterial(crate::world::NodeId),
    TransformDrag {
        commands: Vec<crate::world::WorldCommand>,
        detached: Vec<crate::world::NodeId>,
        gesture: u64,
        frame: f64,
        dt: f64,
    },
    EndDrag,
    DetachCameraTake {
        camera: crate::world::NodeId,
        gesture: Option<u64>,
    },
    RecordedTake(crate::takes::Take),
    NoiseSeed,
    SetRaw(bool),
    SetTool(ViewTool),
    GizmoSize(bool),
    ViewportFlag {
        path: String,
        value: bool,
    },
    Exposure {
        value: f32,
        frame: f64,
        gesture: Option<u64>,
    },
    ColourSelection(crate::ocio::Sel),
    ColourConfig(String),
    BrowseColour,
    ReloadColour,
    ColourPresets(crate::ocio::ColourPresets),
    StoreCameraSlot(usize),
    RecallCameraSlot(usize),
    CameraPose {
        camera: Camera,
        gesture: Option<u64>,
        navigated: bool,
    },
    FocusDistance(f32),
    FrameCamera {
        width: u32,
        height: u32,
    },
    FrameCurrentCamera,
    RestoreCamera,
    ToggleFlight,
    RecenterCamera,
    Fly(FlyInput),
    Orbit(OrbitInput),
    Select {
        id: Option<crate::world::NodeId>,
        add: bool,
    },
    Pick {
        ndc: [f32; 2],
        add: bool,
        focus: bool,
    },
    SetPlaying(bool),
    Preview {
        first: u32,
        last: u32,
        mode: crate::preview::PreviewMode,
    },
    PlayPreview {
        first: u32,
        last: u32,
        head: u32,
    },
    CancelPreview,
    ExportJob(crate::export::ExportJob),
    OutputRecipe {
        id: crate::world::NodeId,
        before: crate::export::OutputSettings,
        after: crate::export::OutputSettings,
        gesture: Option<u64>,
    },
    StartExport,
    CancelExport,
    ExportPreset {
        save: bool,
        path: PathBuf,
    },
    ExportPresetRequest(video_io_egui::PresetRequest),
    ExportPresetDialog(video_io_egui::PresetPickRequest),
    DockAction(soluwell_chrome::dock_layout::Action<dock::Panel>),
    RestoreLayout(soluwell_chrome::dock_layout::Layout<dock::Panel>),
    ResetLayout,
    SaveLayout(String),
    LayoutStore(soluwell_chrome::layout_manager::LayoutStore),
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn preference_delta_keeps_intervening_edits_and_inserts_file_history() {
        let before = json!({"controls":{"look":1,"speed":2},"file_dialogs":{"directories":{}}});
        let after = json!({"controls":{"look":3,"speed":2},"file_dialogs":{"directories":{"scene.open":"C:/Scenes"}}});
        let mut current = json!({"controls":{"look":1,"speed":7},"file_dialogs":{"directories":{"take.import":"C:/Takes"}},"gui_fps":120});
        merge_changes(&mut current, &before, &after);
        assert_eq!(
            current,
            json!({"controls":{"look":3,"speed":7},"file_dialogs":{"directories":{"scene.open":"C:/Scenes","take.import":"C:/Takes"}},"gui_fps":120})
        );
    }
}

#[derive(Clone)]
pub(crate) struct FlyInput {
    pub intents: Vec<cam_controls::CameraIntent>,
    pub wheel: f32,
    pub held: bool,
    pub align: bool,
    pub dt: f32,
    pub width: u32,
    pub height: u32,
    pub speed_scale: f32,
}
#[derive(Clone)]
pub(crate) struct OrbitInput {
    pub intents: Vec<cam_controls::CameraIntent>,
    pub dt: f32,
    pub width: u32,
    pub height: u32,
}

/// Apply only fields a producer changed, preserving intervening edits from other panels.
fn merge_changes(current: &mut Value, before: &Value, after: &Value) {
    if before == after {
        return;
    }
    if let (Some(current), Some(before), Some(after)) = (
        current.as_object_mut(),
        before.as_object(),
        after.as_object(),
    ) {
        for (key, value) in after {
            match (before.get(key), current.get_mut(key)) {
                (Some(previous), Some(target)) => merge_changes(target, previous, value),
                (None, _) => {
                    current.insert(key.clone(), value.clone());
                }
                _ => {}
            }
        }
        for key in before.keys().filter(|key| !after.contains_key(*key)) {
            current.remove(key);
        }
    } else {
        *current = after.clone();
    }
}

fn apply_patch(target: &mut Value, patch: &Value) -> Result<(), String> {
    let object = patch.as_object().ok_or("Patch must be an object")?;
    let target = target
        .as_object_mut()
        .ok_or("Patch target must be an object")?;
    for (key, value) in object {
        let field = target
            .get_mut(key)
            .ok_or_else(|| format!("Unknown field: {key}"))?;
        if field.is_object() && value.is_object() {
            apply_patch(field, value)?;
        } else {
            *field = value.clone();
        }
    }
    Ok(())
}

impl App {
    pub(crate) fn handle_app_action(
        &mut self,
        action: AppAction,
        frame: Option<f64>,
        ctx: &egui::Context,
    ) -> Result<Value, String> {
        let edit_frame = frame.unwrap_or_else(|| f64::from(self.world_ui.playhead));
        if matches!(
            &action,
            AppAction::CameraPose { .. }
                | AppAction::CameraPatch(_)
                | AppAction::CameraOrbit { .. }
                | AppAction::CameraPan { .. }
                | AppAction::CameraZoom { .. }
                | AppAction::CameraFly { .. }
                | AppAction::FrameCamera { .. }
                | AppAction::FrameCurrentCamera
                | AppAction::RestoreCamera
                | AppAction::ToggleFlight
                | AppAction::RecenterCamera
                | AppAction::Fly(_)
                | AppAction::Orbit(_)
                | AppAction::RecallCameraSlot(_)
                | AppAction::StoreCameraSlot(_)
                | AppAction::FocusDistance(_)
        ) {
            let key = (
                playa_graph::NodeId(self.world.document.graph.id),
                self.world.revision(),
                edit_frame.to_bits(),
            );
            if self.evaluated_world != Some(key) {
                self.scene = self.world.document.snapshot(edit_frame)?;
                self.evaluated_world = Some(key);
            }
        }
        match action {
            AppAction::SettingsPatch { before, after } => {
                let mut current =
                    serde_json::to_value(self.current_settings()).map_err(|e| e.to_string())?;
                merge_changes(&mut current, &before, &after);
                if before.get("layout") == after.get("layout") {
                    current["layout"] = Value::Null;
                }
                self.apply_settings(serde_json::from_value(current).map_err(|e| e.to_string())?);
            }
            AppAction::PatchSettings(patch) => {
                let mut settings =
                    serde_json::to_value(self.current_settings()).map_err(|e| e.to_string())?;
                apply_patch(&mut settings, &patch)?;
                if patch.get("layout").is_none() {
                    settings["layout"] = Value::Null;
                }
                self.apply_settings(serde_json::from_value(settings).map_err(|e| e.to_string())?);
            }
            AppAction::SceneDocument(document) => {
                let mut scene = document.snapshot(f64::from(document.first))?;
                scene.document = Some(Box::new(document));
                self.load(scene, Load::Same);
            }
            AppAction::GalleryLoad(index) => {
                let scene = self
                    .gallery
                    .get(index)
                    .ok_or("Gallery index out of range")?
                    .scene
                    .clone();
                self.load(scene, Load::New);
                self.card = Some(Card::Gallery(index));
            }
            AppAction::ApplyLibraryMaterial { index } => {
                crate::materials::PRESETS
                    .get(index)
                    .ok_or("Material preset index out of range")?;
                self.assign_library_material(index, AssignTo::Selected)?;
            }
            AppAction::CameraPatch(patch) => {
                self.cancel_navigation_preview();
                let mut camera =
                    serde_json::to_value(self.scene.camera).map_err(|e| e.to_string())?;
                apply_patch(&mut camera, &patch)?;
                let camera = serde_json::from_value(camera).map_err(|e| e.to_string())?;
                self.fly = None;
                self.orbit.rig.stop_inertia();
                self.author_camera(camera, edit_frame, self.world_ui.auto_key, None, false)?;
            }
            AppAction::CameraOrbit {
                yaw_degrees,
                pitch_degrees,
            } => {
                self.cancel_navigation_preview();
                self.orbit.rig.stop_inertia();
                let [width, height] = self.action_extent(ctx);
                self.apply_orbit_input(
                    OrbitInput {
                        intents: vec![cam_controls::CameraIntent::Orbit {
                            yaw_delta: yaw_degrees.to_radians(),
                            pitch_delta: pitch_degrees.to_radians(),
                        }],
                        dt: 0.0,
                        width,
                        height,
                    },
                    edit_frame,
                )?;
                self.world.finish_edit();
            }
            AppAction::CameraPan { dx_px, dy_px } => {
                self.cancel_navigation_preview();
                self.orbit.rig.stop_inertia();
                let [width, height] = self.action_extent(ctx);
                self.apply_orbit_input(
                    OrbitInput {
                        intents: vec![cam_controls::CameraIntent::Pan { dx_px, dy_px }],
                        dt: 0.0,
                        width,
                        height,
                    },
                    edit_frame,
                )?;
                self.world.finish_edit();
            }
            AppAction::CameraZoom { factor } => {
                if !factor.is_finite() || factor <= 0.0 {
                    return Err("Zoom factor must be finite and positive".into());
                }
                self.cancel_navigation_preview();
                self.orbit.rig.stop_inertia();
                let [width, height] = self.action_extent(ctx);
                self.apply_orbit_input(
                    OrbitInput {
                        intents: vec![cam_controls::CameraIntent::Zoom { factor }],
                        dt: 0.0,
                        width,
                        height,
                    },
                    edit_frame,
                )?;
                self.world.finish_edit();
            }
            AppAction::CameraFly {
                translation,
                rotation_degrees,
            } => {
                self.cancel_navigation_preview();
                let mut camera = self.scene.camera;
                let delta = camera.orientation()
                    * glam::Vec3::new(translation[0], translation[1], -translation[2]);
                camera.target = (glam::Vec3::from_array(camera.target) + delta).to_array();
                camera.yaw_degrees += rotation_degrees[0];
                camera.pitch_degrees += rotation_degrees[1];
                camera.roll_degrees += rotation_degrees[2];
                self.fly = None;
                self.orbit.rig.stop_inertia();
                self.author_camera(camera, edit_frame, self.world_ui.auto_key, None, false)?;
            }
            AppAction::ColourPreset(index) => {
                let mut selection = self.scene.colour.clone();
                if !self.colour.presets.restore(index, &mut selection) {
                    return Err("Colour preset index out of range".into());
                }
                self.commit_colour_selection(selection.clone(), edit_frame)?;
                self.colour.set_sel(&selection);
            }
            AppAction::SetPaused(value) => {
                self.author_viewport_flag("/viewport/paused", value, edit_frame)?
            }
            AppAction::Exit => ctx.send_viewport_cmd(egui::ViewportCommand::Close),
            AppAction::OpenSettings(page) => self.open_settings(page),
            AppAction::OpenPanel(panel) => self.panels_to_open.push(panel),
            AppAction::ShowUi(shown) => self.show_ui = shown,
            AppAction::History(step) => self.step_history(step),
            AppAction::NewScene => self.new_scene(),
            AppAction::LoadScene {
                scene,
                new_document,
                card,
            } => {
                self.load(*scene, if new_document { Load::New } else { Load::Same });
                self.card = card;
            }
            AppAction::SceneDialog(action) => self.scene_action_dialog(action),
            AppAction::SceneFile { path, action } => self.submit_scene_action(path, action),
            AppAction::SaveScene => self.save_scene(),
            AppAction::RefreshTemplates => self.refresh_templates(),
            AppAction::BuiltinTemplate(index) => {
                let scene = crate::templates::builtin_scene(index)?;
                self.load(scene, Load::New);
            }
            AppAction::SaveBookmark => self.save_bookmark(),
            AppAction::DeleteBookmark(path) => {
                let id = self.next_file_id();
                self.io
                    .send(crate::io_service::Command::Delete { id, path })?;
                self.bookmark_ops.push((id, BookmarkOp::Delete));
            }
            AppAction::SaveFrame { monitor, file } => self.save_frame(monitor, file),
            AppAction::WindowShot { files, monitor } => self.request_window_shot(files, monitor),
            AppAction::CopyScene => ctx.copy_text(
                serde_json::to_string_pretty(&self.world.document).map_err(|e| e.to_string())?,
            ),
            AppAction::CopyNodes(ids) => ctx.copy_text(self.world.document.copy_fragment(&ids)?),
            AppAction::PasteNodes(text) => {
                self.world.finish_edit();
                self.world.paste(&text)?;
            }
            AppAction::SelectMaterial(id) => {
                self.world.document.info(id)?;
                self.select_material_node(id);
            }
            AppAction::InspectNode(id) => {
                self.world.document.info(id)?;
                self.world.click_select(Some(id), false);
                self.panels_to_open.push(dock::Panel::Inspector);
            }
            AppAction::AssignMaterial { material, targets } => {
                if targets.is_empty() {
                    return Err("Select an object that supports materials".into());
                }
                self.world.finish_edit();
                self.world.execute(crate::world::WorldCommand::Batch(
                    targets
                        .into_iter()
                        .map(|id| crate::world::WorldCommand::AssignMaterial {
                            id,
                            material: Some(material),
                        })
                        .collect(),
                ))?;
            }
            AppAction::CreateMaterial {
                material,
                name,
                targets,
                open,
            } => {
                self.world.finish_edit();
                self.world.execute(match targets {
                    Some(targets) => crate::world::WorldCommand::CreateMaterialFor {
                        material,
                        name,
                        targets,
                    },
                    None => crate::world::WorldCommand::CreateMaterial { material, name },
                })?;
                self.material_gallery.invalidate();
                if open {
                    if let Some(id) = self.world.active() {
                        self.select_material_node(id);
                    }
                    self.panels_to_open.push(dock::Panel::Materials);
                }
            }
            AppAction::Edit { command, gesture } => {
                if let Some(gesture) = gesture {
                    self.world.execute_edit(command, Some(gesture))?;
                } else {
                    self.world.execute(command)?;
                }
            }
            AppAction::RefreshMaterial(id) => {
                self.material_gallery.invalidate_thumbnail(id);
            }
            AppAction::TransformDrag {
                commands,
                detached,
                gesture,
                frame,
                dt,
            } => {
                let ids: Vec<_> = commands
                    .iter()
                    .filter_map(|c| match c {
                        crate::world::WorldCommand::SetAttribute { id, .. } => Some(*id),
                        _ => None,
                    })
                    .collect();
                self.world
                    .execute_edit(crate::world::WorldCommand::Batch(commands), Some(gesture))?;
                if !detached.is_empty() {
                    self.world.bind_take(&detached, None, Some(gesture))?;
                }
                self.gizmo_gesture = Some(gesture);
                if self.record_takes {
                    let mut poses = Vec::new();
                    for id in ids {
                        if !poses.iter().any(|(existing, _)| *existing == id) {
                            poses.push((id, self.world.document.authored_trs(id, frame)?));
                        }
                    }
                    self.drag_recorder.sample(dt, frame, &poses);
                }
            }
            AppAction::DetachCameraTake { camera, gesture } => {
                self.world.bind_take(&[camera], None, gesture)?
            }
            AppAction::RecordedTake(take) => {
                self.world.add_take(take);
            }
            AppAction::EndDrag => {
                if self.gizmo_gesture.take().is_some() {
                    self.drag_recorder.min_secs = self.min_take_secs;
                    for take in self.drag_recorder.finish() {
                        match take {
                            Ok(take) => {
                                self.world.add_take(take);
                            }
                            Err(error) => self.status = format!("Recorded take dropped: {error}"),
                        }
                    }
                }
            }
            AppAction::NoiseSeed => self.seed = self.seed.wrapping_add(7920),
            AppAction::SetRaw(raw) => self.raw_view = raw,
            AppAction::SetTool(tool) => self.tool = tool,
            AppAction::GizmoSize(larger) => {
                self.controls.gizmo_size = step_gizmo_size(self.controls.gizmo_size, larger)
            }
            AppAction::ViewportFlag { path, value } => {
                self.author_viewport_flag(&path, value, edit_frame)?
            }
            AppAction::Exposure {
                value,
                frame,
                gesture,
            } => {
                let id = self.world.document.viewport_settings_id()?;
                self.world.execute_edit(
                    crate::world::WorldCommand::SetAttribute {
                        id,
                        path: "/render/exposure_stops".into(),
                        value: json!(value),
                        frame,
                    },
                    gesture,
                )?;
            }
            AppAction::ColourSelection(selection) => {
                self.commit_colour_selection(selection.clone(), edit_frame)?;
                self.colour.set_sel(&selection);
            }
            AppAction::BrowseColour => {
                self.config_picker = self.world_ui.file_dialogs.prepare(
                    egui_file_dialog::FileDialog::new()
                        .add_file_filter_extensions("OCIO config", vec!["ocio"]),
                    crate::file_dialogs::OCIO,
                    std::path::Path::new(&self.colour.sel.config).parent(),
                    "OCIO config",
                );
                self.config_picker.pick_file();
            }
            AppAction::ColourConfig(path) => {
                let selection = crate::ocio::Sel {
                    on: self.scene.colour.on,
                    config: path,
                    ..crate::ocio::Sel::default()
                };
                self.commit_colour_selection(selection.clone(), edit_frame)?;
                self.colour.set_sel(&selection);
            }
            AppAction::ReloadColour => {
                self.colour.reload();
                self.renderer
                    .try_command(Command::ReloadColour)
                    .map_err(|e| e.to_string())?;
            }
            AppAction::ColourPresets(presets) => self.colour.presets = presets,
            AppAction::StoreCameraSlot(index) => {
                if index >= crate::camera_slots::COUNT {
                    return Err("Camera slot index out of range".into());
                }
                self.camera_slots.store(index, &self.scene.camera);
            }
            AppAction::RecallCameraSlot(index) => {
                let mut camera = self.scene.camera;
                if !self.camera_slots.restore(index, &mut camera) {
                    return Err(format!("CamClip {} is empty", index + 1));
                }
                self.fly = None;
                self.orbit.rig.stop_inertia();
                self.author_camera(camera, edit_frame, self.world_ui.auto_key, None, false)?;
            }
            AppAction::FocusDistance(distance) => {
                let mut camera = self.scene.camera;
                camera.focus_distance = distance;
                self.author_camera(camera, edit_frame, self.world_ui.auto_key, None, false)?;
            }
            AppAction::CameraPose {
                camera,
                gesture,
                navigated,
            } => {
                self.cancel_navigation_preview();
                self.fly = None;
                self.orbit.rig.stop_inertia();
                self.author_camera(
                    camera,
                    edit_frame,
                    self.world_ui.auto_key,
                    gesture,
                    navigated,
                )?;
            }
            AppAction::FrameCurrentCamera => {
                let [width, height] = self.action_extent(ctx);
                return self.handle_app_action(
                    AppAction::FrameCamera { width, height },
                    Some(edit_frame),
                    ctx,
                );
            }
            AppAction::FrameCamera { width, height } => {
                self.cancel_navigation_preview();
                self.orbit.rig.stop_inertia();
                self.frame_camera(width, height);
                self.author_camera(
                    self.scene.camera,
                    edit_frame,
                    self.world_ui.auto_key,
                    None,
                    false,
                )?;
            }
            AppAction::RestoreCamera => {
                self.cancel_navigation_preview();
                self.orbit.rig.stop_inertia();
                self.restore_camera(ctx);
                self.author_camera(
                    self.scene.camera,
                    edit_frame,
                    self.world_ui.auto_key,
                    None,
                    false,
                )?;
            }
            AppAction::ToggleFlight => {
                self.cancel_navigation_preview();
                self.toggle_flight_mode();
                self.author_camera(
                    self.scene.camera,
                    edit_frame,
                    self.world_ui.auto_key,
                    Some(self.camera_gesture()),
                    false,
                )?;
            }
            AppAction::RecenterCamera => {
                let mut camera = self.scene.camera;
                camera.target = [0.0; 3];
                self.author_camera(
                    camera,
                    edit_frame,
                    self.world_ui.auto_key,
                    Some(self.camera_gesture()),
                    false,
                )?;
            }
            AppAction::Fly(input) => self.apply_fly_input(input, edit_frame, ctx)?,
            AppAction::Orbit(input) => self.apply_orbit_input(input, edit_frame)?,
            AppAction::Select { id, add } => {
                if let Some(id) = id {
                    self.world.document.info(id)?;
                }
                self.world.click_select(id, add);
            }
            AppAction::Pick { ndc, add, focus } => {
                self.queue_pick(ndc, add, focus, edit_frame)?;
                return Ok(json!({"queued": true, "pick_id": self.pick_sequence}));
            }
            AppAction::SetPlaying(playing) => {
                if playing {
                    self.world_ui.playing = true;
                } else {
                    self.cancel_navigation_preview();
                }
            }
            AppAction::Preview { first, last, mode } => self.begin_preview(first, last, mode, ctx),
            AppAction::PlayPreview { first, last, head } => {
                self.begin_preview(first, last, crate::preview::PreviewMode::Play, ctx);
                self.preview.seek(head.clamp(first, last));
                self.world_ui.seek(head.clamp(first, last));
            }
            AppAction::CancelPreview => self.cancel_navigation_preview(),
            AppAction::ExportJob(job) => self.export.restore(job),
            AppAction::OutputRecipe {
                id,
                before,
                after,
                gesture,
            } => {
                let commands = crate::render_profiles::output_edits(id, &before, &after)?;
                if !commands.is_empty() {
                    self.world
                        .execute_edit(crate::world::WorldCommand::Batch(commands), gesture)?;
                }
            }
            AppAction::StartExport => {
                self.world.finish_edit();
                self.refresh_scene()?;
                let scene = freeze_world_scene(&self.scene, &self.world.document);
                self.export
                    .start(&scene, &self.renderer, self.colour.config())?;
                return Ok(json!({"queued": true}));
            }
            AppAction::CancelExport => self.export.cancel(),
            AppAction::ExportPreset { save, path } => {
                let id = self.world.document.output_settings_id()?;
                let before = self.world.document.output_settings_of(id)?;
                self.export.output = before.clone();
                if let Some(preset) = self.export.preset_file(save, &path)? {
                    let mut after = before.clone();
                    after.preset = preset;
                    self.world.execute(crate::world::WorldCommand::Batch(
                        crate::render_profiles::output_edits(id, &before, &after)?,
                    ))?;
                }
            }
            AppAction::ExportPresetRequest(request) => {
                self.export.handle_preset_request(request)?
            }
            AppAction::ExportPresetDialog(request) => self.export.open_preset_dialog(request),
            AppAction::DockAction(action) => self
                .dock
                .apply_with_permissions(action, dock::Panel::permissions)
                .map_err(|e| e.to_string())?,
            AppAction::RestoreLayout(layout) => {
                if !dock::valid_layout(&layout) {
                    return Err("Layout must contain one viewport and no duplicate panels".into());
                }
                self.dock = layout;
                self.panels_to_open.clear();
            }
            AppAction::ResetLayout => self.reset_layout(),
            AppAction::SaveLayout(name) => {
                let blob =
                    soluwell_chrome::dock_layout::to_blob(&self.dock).map_err(|e| e.to_string())?;
                self.layouts
                    .store
                    .save(&name, blob)
                    .map_err(|e| e.to_string())?;
            }
            AppAction::LayoutStore(store) => self.layouts.store = store,
        }
        self.evaluated_world = None;
        Ok(
            json!({"success": true, "revision": self.world.revision(), "selection": self.world.selected(), "camera": self.scene.camera}),
        )
    }

    fn action_extent(&self, ctx: &egui::Context) -> [u32; 2] {
        let (width, height) = self
            .viewport_rect
            .map(|rect| viewport_pixels(rect.size(), ctx.pixels_per_point()))
            .or_else(|| {
                self.request
                    .as_ref()
                    .map(|request| (request.width, request.height))
            })
            .unwrap_or((1600, 940));
        [width as u32, height as u32]
    }

    fn author_viewport_flag(&mut self, path: &str, value: bool, frame: f64) -> Result<(), String> {
        let id = self.world.document.viewport_settings_id()?;
        self.world
            .execute(crate::world::WorldCommand::SetAttribute {
                id,
                path: path.into(),
                value: json!(value),
                frame,
            })
    }

    fn commit_colour_selection(
        &mut self,
        selection: crate::ocio::Sel,
        frame: f64,
    ) -> Result<(), String> {
        let before = self.world.document.snapshot(frame)?;
        let mut edited = before.clone();
        edited.colour = selection.clone();
        edited.render.reinhard = false;
        self.world.edit_snapshot_with_gesture(
            self.world.active(),
            &before,
            &edited,
            frame,
            None,
        )?;
        self.default_colour = selection;
        self.scene = edited;
        Ok(())
    }

    fn cancel_navigation_preview(&mut self) {
        self.preview.cancel();
        self.preview_key = None;
        self.showing_preview = false;
        self.world_ui.playing = false;
    }

    fn author_camera(
        &mut self,
        camera: Camera,
        frame: f64,
        auto_key: bool,
        gesture: Option<u64>,
        navigated: bool,
    ) -> Result<(), String> {
        camera.validate_lens()?;
        if camera.distance <= 0.0
            || !camera.distance.is_finite()
            || camera.target.iter().any(|v| !v.is_finite())
            || [
                camera.yaw_degrees,
                camera.pitch_degrees,
                camera.roll_degrees,
            ]
            .iter()
            .any(|v| !v.is_finite())
        {
            return Err("Camera pose must be finite, with positive distance".into());
        }
        let mut scene = self.world.document.snapshot(frame)?;
        if scene.camera == camera {
            self.scene.camera = camera;
            self.navigated |= navigated;
            return Ok(());
        }
        let sensor_changed = scene.camera.sensor_height != camera.sensor_height;
        scene.camera = camera;
        let bound = self
            .world
            .document
            .active_camera
            .filter(|id| matches!(self.world.document.take_clip(*id), Ok(Some(_))));
        let mut staged = (sensor_changed || bound.is_some()).then(|| self.world.clone());
        let world = staged.as_mut().unwrap_or(&mut self.world);
        if gesture.is_none() {
            world.finish_edit();
        }
        let edit_gesture = gesture.or(sensor_changed.then_some(u64::MAX));
        if let Some(id) = bound {
            world.bind_take(&[id], None, edit_gesture)?;
        }
        world.navigate_camera(&scene, frame, auto_key, edit_gesture)?;
        if sensor_changed {
            world.execute_edit(
                crate::world::WorldCommand::SetAttribute {
                    id: world.document.active_camera.ok_or("No active camera")?,
                    path: "/camera/sensor_height".into(),
                    value: json!(camera.sensor_height),
                    frame,
                },
                edit_gesture,
            )?;
        }
        if gesture.is_none() {
            world.finish_edit();
        }
        if let Some(staged) = staged {
            self.world = staged;
        }
        self.scene.camera = camera;
        self.navigated |= navigated;
        Ok(())
    }

    fn queue_pick(
        &mut self,
        ndc: [f32; 2],
        add: bool,
        focus: bool,
        frame: f64,
    ) -> Result<(), String> {
        if ndc.iter().any(|v| !v.is_finite() || v.abs() > 1.0) {
            return Err("Pick NDC must be in [-1,1]".into());
        }
        let request = self.request.as_ref().ok_or("No viewport request")?;
        self.pick_sequence = self.pick_sequence.wrapping_add(1);
        let pending = PendingPick {
            id: self.pick_sequence,
            document: playa_graph::NodeId(self.world.document.graph.id),
            camera: self.world.document.active_camera,
            frame,
            generation: request.generation,
            nodes: request.scene.objects.iter().map(|o| o.node).collect(),
            add,
            focus,
            camera_radius: request.scene.camera_radius(),
        };
        self.renderer
            .try_command(Command::Pick {
                id: pending.id,
                scene: Box::new(request.scene.clone()),
                width: request.width,
                height: request.height,
                ndc,
            })
            .map_err(|e| e.to_string())?;
        self.pick_pending = Some(pending);
        Ok(())
    }
}

impl App {
    fn apply_fly_input(
        &mut self,
        input: FlyInput,
        frame: f64,
        ctx: &egui::Context,
    ) -> Result<(), String> {
        use cam_controls::CameraIntent;
        use glam::Vec3;
        if input.held {
            self.cancel_navigation_preview();
        }
        let radius = self.scene.camera_radius();
        if input.held && self.fly.is_none() {
            self.start_flight();
        }
        let mut camera = self.scene.camera;
        let distance = camera.distance * radius;
        let lock = (!camera.free_flight).then_some(Vec3::Y);
        let Some(fly) = self.fly.as_mut() else {
            return Ok(());
        };
        fly.inertia = self.controls.inertia(radius);
        if fly.horizon_lock().is_some() != lock.is_some() && !fly.set_horizon_lock(lock) {
            return Err("Horizon lock needs a view that is not straight up or down".into());
        }
        let viewport = cam_viewport::ViewportSize::new(input.width, input.height);
        if input.held {
            ctx.set_cursor_icon(egui::CursorIcon::None);
            ctx.send_viewport_cmd(egui::ViewportCommand::CursorGrab(egui::CursorGrab::Locked));
            self.fly_speed_scale = input.speed_scale;
            for intent in input.intents {
                fly.apply_intent(intent, viewport);
            }
            if input.wheel != 0.0 {
                let forward = fly.orientation * -Vec3::Z;
                fly.eye += forward
                    * (input.wheel
                        * 0.002
                        * radius
                        * self.controls.fly_speed
                        * self.fly_speed_scale);
            }
            fly.set_align(input.align.then_some((
                cam_controls::OrbitSnapMode::Smart,
                self.controls.align_seconds,
            )));
        } else {
            fly.set_align(None);
            ctx.send_viewport_cmd(egui::ViewportCommand::CursorGrab(egui::CursorGrab::None));
            for intent in [
                CameraIntent::Thrust {
                    forward: 0.0,
                    right: 0.0,
                    up: 0.0,
                },
                CameraIntent::Roll { d: 0.0 },
                CameraIntent::SpeedScale(1.0),
            ] {
                fly.apply_intent(intent, viewport);
            }
            self.fly_speed_scale = 1.0;
        }
        let moving = fly.update_dynamics(input.dt) || fly.has_motion();
        let pose = fly.pose();
        let (yaw, pitch, roll) = pose.orientation.to_euler(glam::EulerRot::YXZ);
        camera.yaw_degrees = yaw.to_degrees();
        camera.pitch_degrees = -pitch.to_degrees();
        camera.roll_degrees = roll.to_degrees();
        camera.target = (pose.eye + pose.forward() * distance).to_array();
        if !input.held && !moving {
            self.fly = None;
        }
        self.author_camera(
            camera,
            frame,
            self.world_ui.auto_key,
            Some(self.camera_gesture()),
            true,
        )?;
        ctx.request_repaint();
        Ok(())
    }

    fn apply_orbit_input(&mut self, input: OrbitInput, frame: f64) -> Result<(), String> {
        use cam_controls::{CameraIntent, OrbitPose};
        use glam::Vec3;
        if !input.intents.is_empty() {
            self.cancel_navigation_preview();
            self.fly = None;
        }
        let radius = self.scene.camera_radius();
        let mut camera = self.scene.camera;
        let rig = &mut self.orbit.rig;
        if rig.has_inertia() && self.orbit_written != Some(camera) {
            rig.stop_inertia();
        }
        rig.apply_navigation(
            self.controls
                .navigation(camera.fov_y_degrees.to_radians(), input.height as f32),
        );
        rig.align_seconds = self.controls.align_seconds;
        rig.align_dt = input.dt;
        rig.pitch_limit = 89.0_f32.to_radians();
        rig.projection.distance_min = 0.05 * radius;
        rig.projection.distance_max = f32::MAX;
        let seed = OrbitPose::from_yaw_pitch(
            (camera.yaw_degrees - 90.0).to_radians(),
            camera.pitch_degrees.to_radians(),
            camera.distance * radius,
            Vec3::from_array(camera.target),
        );
        rig.pose = seed;
        let viewport = cam_viewport::ViewportSize::new(input.width, input.height);
        let (sin, cos) = camera.roll_degrees.to_radians().sin_cos();
        let unroll = |x: f32, y: f32| (x * cos + y * sin, y * cos - x * sin);
        let mut smart_align = false;
        for intent in input.intents {
            if matches!(
                intent,
                CameraIntent::OrbitSnap {
                    mode: cam_controls::OrbitSnapMode::Smart,
                    ..
                }
            ) {
                smart_align = true;
            }
            let intent = match intent {
                CameraIntent::Pan { dx_px, dy_px } => {
                    let (dx_px, dy_px) = unroll(dx_px, dy_px);
                    CameraIntent::Pan { dx_px, dy_px }
                }
                CameraIntent::PanInertia { dx_rate, dy_rate } => {
                    let (dx_rate, dy_rate) = unroll(dx_rate, dy_rate);
                    CameraIntent::PanInertia { dx_rate, dy_rate }
                }
                other => other,
            };
            self.orbit.apply_intent(intent, viewport);
        }
        let rig = &mut self.orbit.rig;
        if rig.has_inertia() {
            rig.update_dynamics(input.dt);
        }
        let pose = rig.pose;
        let mut changed = false;
        if pose != seed {
            let (yaw, pitch) = cam_controls::yaw_pitch_from_orientation(pose.orientation);
            camera.yaw_degrees = yaw.to_degrees() + 90.0;
            camera.pitch_degrees = pitch.to_degrees();
            camera.target = pose.target.to_array();
            camera.distance = pose.distance / radius;
            changed = true;
        }
        if smart_align && camera.roll_degrees != 0.0 {
            let tau = self.controls.align_seconds;
            if tau <= 1e-5 || camera.roll_degrees.abs() < 1e-3 {
                camera.roll_degrees = 0.0;
            } else {
                camera.roll_degrees *= (-input.dt / tau).exp();
            }
            changed = true;
        }
        self.orbit_written = rig.has_inertia().then_some(camera);
        if changed {
            self.author_camera(
                camera,
                frame,
                self.world_ui.auto_key,
                Some(self.camera_gesture()),
                true,
            )?;
        }
        Ok(())
    }
}
