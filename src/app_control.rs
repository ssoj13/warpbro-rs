//! WarpBro's adapter to the REST command bus. All document edits use WorldEditor.
//! Camera actions use the shared navigation rig and author through navigate_camera.
use super::*;
use crate::world::{NodeId, WorldCommand};
use app_control::Reply;
use serde::Deserialize;
use serde_json::{Value, json};

const COMMANDS: &[&str] = &[
    "scene.new",
    "scene.load",
    "scene.document",
    "gallery.load",
    "world.command",
    "selection.set",
    "history.undo",
    "history.redo",
    "timeline.seek",
    "timeline.play",
    "timeline.pause",
    "viewport.camera",
    "viewport.orbit",
    "viewport.pan",
    "viewport.zoom",
    "viewport.fly",
    "viewport.frame",
    "viewport.home",
    "viewport.pick",
    "viewport.raw",
    "viewport.tool",
    "settings.set",
    "material.create",
    "material.apply",
    "export.start",
    "export.cancel",
    "app.ui",
    "app.exit",
];

fn payload<T: serde::de::DeserializeOwned>(value: Value) -> Result<T, String> {
    serde_json::from_value(value).map_err(|e| e.to_string())
}
fn value(value: impl serde::Serialize) -> Result<Value, String> {
    serde_json::to_value(value).map_err(|e| e.to_string())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Index {
    index: usize,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Ids {
    ids: Vec<NodeId>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Seek {
    frame: u32,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Orbit {
    yaw_degrees: f32,
    pitch_degrees: f32,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Pan {
    dx_px: f32,
    dy_px: f32,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Zoom {
    factor: f32,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Fly {
    translation: [f32; 3],
    rotation_degrees: [f32; 3],
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Enabled {
    enabled: bool,
}

impl App {
    pub(super) fn poll_control(&mut self, ctx: &egui::Context) {
        if !self.control_started {
            self.control_started = true;
            crate::render_diagnostics::set_wake(self.wake.clone());
            // Unit tests construct many apps; none opens an external listener.
            #[cfg(not(test))]
            if std::env::var("WARPBRO_API").as_deref() != Ok("0") {
                let port = std::env::var("WARPBRO_API_PORT")
                    .map_or(Ok(9877), |v| v.parse::<u16>().map_err(|e| e.to_string()));
                let wake = self.wake.clone();
                match port.and_then(|port| {
                    app_control::Server::start(port, std::sync::Arc::new(move || wake.wake()))
                }) {
                    Ok(server) => {
                        log::info!(
                            "WarpBro REST API: http://127.0.0.1:{}/api/health",
                            server.port()
                        );
                        self.control = Some(server);
                    }
                    Err(error) => {
                        log::error!("{error}");
                        self.status = error;
                    }
                }
            }
        }
        // Borrow the transport only to receive; dispatch owns the normal mutable App.
        for _ in 0..32 {
            let Some(call) = self.control.as_ref().and_then(|s| s.calls.try_recv().ok()) else {
                break;
            };
            if call.expired() {
                call.respond(Err("Command expired before execution".into()));
                continue;
            }
            let result = self.control_command(&call.event_type, call.payload.clone(), ctx);
            call.respond(result);
        }
    }

    fn control_status(&self) -> Value {
        let frame = self.current();
        json!({
            "application":"WarpBro", "pid":std::process::id(),
            "scene":self.scene.name, "document":self.world.document.graph.id,
            "revision":self.world.revision(), "selection":self.world.selected(),
            "camera":self.scene.camera,
            "timeline":{"frame":self.world_ui.playhead,"playing":self.world_ui.playing,"first":self.world.document.first,"last":self.world.document.last,"fps":self.world.document.fps},
            "render":{
                "requested_generation":self.request.as_ref().map(|r|r.generation),
                "frame_generation":frame.map(|f|f.generation),
                "size":frame.map(|f|[f.width,f.height]),
                "samples":frame.map(|f|f.samples), "target_spp":self.target_spp,
                "last_ms":frame.map(|f|f.last_ms),"last_spp":frame.map(|f|f.last_spp),
                "denoise_ms":frame.map(|f|f.denoise_ms),
                "colour_error":frame.and_then(|f|f.colour_error.as_ref()),
                "denoise_error":frame.and_then(|f|f.denoise_error.as_ref()),
                "worker_stopped":self.renderer.preview_stopped()
            },
            "thumbnails":{"gallery_ready":self.gallery.iter().filter(|e|e.thumb.is_some()).count(),"gallery_total":self.gallery.len(),
                "materials_ready":self.material_gallery.entries.iter().filter(|e|e.thumb.is_some()).count(),
                "swatches_ready":self.swatches.iter().filter(|e|e.is_some()).count(),"pending":self.thumb_pending.as_ref().map(|p|p.0),"failed":self.thumb_failed.len()},
            "diagnostics":crate::render_diagnostics::snapshot(),
            "status":self.status
        })
    }

    fn control_command(
        &mut self,
        command: &str,
        data: Value,
        ctx: &egui::Context,
    ) -> Result<Reply, String> {
        if matches!(
            command,
            "viewport.camera"
                | "viewport.orbit"
                | "viewport.pan"
                | "viewport.zoom"
                | "viewport.fly"
                | "viewport.frame"
                | "viewport.home"
                | "viewport.pick"
        ) {
            // A cached preview deliberately freezes scene. Evaluate the displayed
            // playhead before deriving any relative camera movement.
            self.preview.cancel();
            self.preview_key = None;
            self.showing_preview = false;
            self.world_ui.playing = false;
            self.evaluated_world = None;
            self.refresh_scene()?;
        }
        if command == "get.screenshot.png" {
            let frame = self.current().ok_or("No rendered viewport frame")?;
            let mut bytes = Vec::new();
            {
                let mut encoder =
                    png::Encoder::new(&mut bytes, frame.width as u32, frame.height as u32);
                encoder.set_color(png::ColorType::Rgba);
                encoder.set_depth(png::BitDepth::Eight);
                let mut writer = encoder.write_header().map_err(|e| e.to_string())?;
                writer
                    .write_image_data(&frame.sdr_bytes)
                    .map_err(|e| e.to_string())?;
            }
            return Ok(Reply::Bytes {
                content_type: "image/png",
                data: bytes,
            });
        }
        let response = match command {
            "get.status" => self.control_status(),
            "get.commands" => json!({"commands":COMMANDS,"world_command_format":"externally tagged serde enum; world.command accepts every WorldCommand including Batch","queries":["status","commands","scene","nodes","gallery","materials","camera","render","diagnostics","settings","screenshot.png","attributes/<uuid>"]}),
            "get.scene" => value(&self.world.document)?,
            "get.camera" => value(self.world.document.snapshot(f64::from(self.world_ui.playhead))?.camera)?,
            "get.settings" => value(self.current_settings())?,
            "get.render" => self.control_status()["render"].clone(),
            "get.diagnostics" => crate::render_diagnostics::snapshot(),
            "get.nodes" => Value::Array(self.world.document.nodes().into_iter().map(|n|json!({
                "id":n.id,"kind":n.kind,"name":n.name,"parent":n.parent,"visible":n.visible,"locked":n.locked,"solo":n.solo,"start":n.start,"end":n.end
            })).collect()),
            "get.gallery" => Value::Array(self.gallery.iter().enumerate().map(|(i,e)|json!({
                "index":i,"name":e.scene.name,"formula":e.scene.formula,"thumbnail":e.thumb.is_some()
            })).collect()),
            "get.materials" => Value::Array(crate::materials::PRESETS.iter().enumerate().map(|(i,m)|json!({"index":i,"preset":m.name()})).collect()),
            "scene.new" => { self.new_scene(); json!({"success":true}) }
            "scene.load" => {
                // The public API accepts the strict document, just as File > Open does.
                let doc = crate::world::WorldDocument::decode_json(&data.to_string())?;
                let mut scene = doc.snapshot(f64::from(doc.first))?;
                scene.document = Some(Box::new(doc));
                self.load(scene, Load::Same); json!({"success":true})
            }
            "scene.document" => {
                let doc = crate::world::WorldDocument::decode_json(&data.to_string())?;
                self.world.reset(doc)?; json!({"success":true})
            }
            "gallery.load" => {
                let Index { index } = payload(data)?;
                let scene = self.gallery.get(index).ok_or("Gallery index out of range")?.scene.clone();
                self.load(scene, Load::New);
                json!({"success":true,"index":index})
            }
            "world.command" => {
                self.world.execute(payload::<WorldCommand>(data)?)?;
                json!({"success":true,"revision":self.world.revision(),"selection":self.world.selected()})
            }
            "selection.set" => {
                let Ids { ids } = payload(data)?;
                for id in &ids { self.world.document.info(*id)?; }
                self.world.set_selection(ids);
                json!({"success":true,"selection":self.world.selected()})
            }
            "history.undo" => { self.world.undo(); json!({"success":true}) }
            "history.redo" => { self.world.redo(); json!({"success":true}) }
            "timeline.seek" => {
                let Seek { frame } = payload(data)?;
                self.preview.cancel(); self.world_ui.seek(frame); json!({"success":true,"frame":frame})
            }
            "timeline.play" => { self.world_ui.playing = true; json!({"success":true}) }
            "timeline.pause" => { self.world_ui.playing = false; self.preview.cancel(); json!({"success":true}) }
            "viewport.camera" => {
                let mut camera = value(self.scene.camera)?;
                let patch = data.as_object().ok_or("camera payload must be an object")?;
                for (key,val) in patch { if camera.get(key).is_none() {return Err(format!("Unknown camera field: {key}"));} camera[key] = val.clone(); }
                self.control_camera(payload(camera)?)?
            }
            "viewport.orbit" | "viewport.pan" | "viewport.zoom" => self.control_orbit(command, data, ctx)?,
            "viewport.fly" => {
                let Fly { translation, rotation_degrees } = payload(data)?;
                let mut cam = self.scene.camera;
                let delta = cam.orientation() * glam::Vec3::new(translation[0],translation[1],-translation[2]);
                cam.target = (glam::Vec3::from_array(cam.target)+delta).to_array();
                cam.yaw_degrees += rotation_degrees[0]; cam.pitch_degrees += rotation_degrees[1]; cam.roll_degrees += rotation_degrees[2];
                self.control_camera(cam)?
            }
            "viewport.frame" => {
                let [w,h] = self.control_extent(ctx);
                self.frame_camera(w,h);
                self.control_camera(self.scene.camera)?
            }
            "viewport.home" => self.control_camera(self.origin.camera)?,
            "viewport.pick" => {
                #[derive(Deserialize)] #[serde(deny_unknown_fields)] struct Pick { ndc:[f32;2] }
                let Pick { ndc } = payload(data)?;
                if ndc.iter().any(|v|!v.is_finite() || v.abs()>1.0) { return Err("Pick NDC must be in [-1,1]".into()); }
                let [w,h] = self.control_extent(ctx);
                self.pick_sequence = self.pick_sequence.wrapping_add(1);
                self.renderer.try_command(Command::Pick { id:self.pick_sequence,scene:Box::new(self.scene.clone()),width:w as usize,height:h as usize,ndc }).map_err(|e|e.to_string())?;
                self.pick_pending = Some(PendingPick {
                    id:self.pick_sequence, nodes:self.scene.objects.iter().map(|o|o.node).collect(),
                    add:false, focus:false, camera_radius:self.scene.camera_radius(),
                });
                // This acknowledgement explicitly names enqueueing, not a completed GPU pick.
                json!({"queued":true,"pick_id":self.pick_sequence})
            }
            "viewport.raw" => { let Enabled { enabled } = payload(data)?; self.raw_view=enabled; json!({"success":true}) }
            "viewport.tool" => {
                #[derive(Deserialize)] #[serde(deny_unknown_fields)] struct Tool { tool:String }
                let Tool { tool }=payload(data)?;
                self.tool = match tool.as_str() {"select"=>ViewTool::Select,"move"=>ViewTool::Move,"rotate"=>ViewTool::Rotate,"scale"=>ViewTool::Scale,_=>return Err("Tool must be select, move, rotate or scale".into())};
                json!({"success":true})
            }
            "settings.set" => {
                let mut settings = value(self.current_settings())?;
                merge_settings(&mut settings, &data)?;
                self.apply_settings(payload(settings)?); json!({"success":true})
            }
            "material.create" => {
                let Index { index } = payload(data)?;
                let preset = crate::materials::PRESETS.get(index).ok_or("Material preset index out of range")?;
                self.world.execute(WorldCommand::CreateMaterial { material:swatch_scene(preset).material,name:preset.name().into() })?;
                json!({"success":true,"selection":self.world.selected()})
            }
            "material.apply" => {
                let Index { index } = payload(data)?;
                crate::materials::PRESETS.get(index).ok_or("Material preset index out of range")?;
                self.assign_library_material(index, AssignTo::Selected)?;
                json!({"success":true})
            }
            "export.start" => {
                let scene = freeze_world_scene(&self.scene, &self.world.document);
                self.export.start(&scene,&self.renderer,self.colour.config())?;
                json!({"queued":true})
            }
            "export.cancel" => { self.export.cancel(); json!({"success":true}) }
            "app.ui" => { let Enabled { enabled }=payload(data)?; self.show_ui=enabled; json!({"success":true}) }
            "app.exit" => { ctx.send_viewport_cmd(egui::ViewportCommand::Close); json!({"success":true}) }
            query if query.starts_with("get.attributes.") => {
                let id: NodeId = payload(json!(&query["get.attributes.".len()..]))?;
                Value::Array(self.world.document.attributes(id,f64::from(self.world_ui.playhead))?.into_iter().map(|a|json!({
                    "path":a.path,"label":a.label,"value":a.value,"keyable":a.keyable,"frames":a.frames,"choices":a.choices,"range":a.range
                })).collect())
            }
            _ => return Err(format!("Unknown command: {command}; GET /api/commands lists supported commands")),
        };
        self.refresh_scene()?;
        ctx.request_repaint();
        Ok(Reply::Json(response))
    }

    fn control_camera(&mut self, camera: crate::scene::Camera) -> Result<Value, String> {
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
        self.fly = None;
        self.orbit.rig.stop_inertia();
        let mut scene = self.scene.clone();
        scene.camera = camera;
        // Lens edits include sensor height, which the navigation bridge does not
        // write. Stage the combined edit so a rejected sensor channel cannot leave
        // an accepted camera movement behind, and share one undo gesture.
        let sensor_changed = camera.sensor_height != self.scene.camera.sensor_height;
        let mut staged = sensor_changed.then(|| self.world.clone());
        let world = staged.as_mut().unwrap_or(&mut self.world);
        world.finish_edit();
        let gesture = sensor_changed.then_some(u64::MAX);
        world.navigate_camera(
            &scene,
            f64::from(self.world_ui.playhead),
            self.world_ui.auto_key,
            gesture,
        )?;
        if sensor_changed {
            world.execute_edit(
                WorldCommand::SetAttribute {
                    id: world.document.active_camera.ok_or("No active camera")?,
                    path: "/camera/sensor_height".into(),
                    value: json!(camera.sensor_height),
                    frame: f64::from(self.world_ui.playhead),
                },
                gesture,
            )?;
        }
        world.finish_edit();
        if let Some(staged) = staged {
            self.world = staged;
        }
        self.evaluated_world = None;
        self.refresh_scene()?;
        Ok(json!({"success":true,"camera":self.scene.camera,"revision":self.world.revision()}))
    }
    fn control_extent(&self, ctx: &egui::Context) -> [u32; 2] {
        let (w, h) = self
            .viewport_rect
            .map(|r| viewport_pixels(r.size(), ctx.pixels_per_point()))
            .or_else(|| self.request.as_ref().map(|r| (r.width, r.height)))
            .unwrap_or((1600, 940));
        [w as u32, h as u32]
    }
    fn control_orbit(
        &mut self,
        command: &str,
        data: Value,
        ctx: &egui::Context,
    ) -> Result<Value, String> {
        use cam_controls::{CameraIntent, OrbitPose};
        let mut rig = cam_controls::HoudiniOrbit::default();
        let cam = self.scene.camera;
        let reference = self.scene.camera_radius();
        let [w, h] = self.control_extent(ctx);
        rig.rig.apply_navigation(
            self.controls
                .navigation(cam.fov_y_degrees.to_radians(), h as f32),
        );
        rig.rig.pitch_limit = 89.0_f32.to_radians();
        rig.rig.projection.distance_min = reference * 0.05;
        rig.rig.projection.distance_max = f32::MAX;
        rig.rig.pose = OrbitPose::from_yaw_pitch(
            (cam.yaw_degrees - 90.0).to_radians(),
            cam.pitch_degrees.to_radians(),
            cam.distance * reference,
            glam::Vec3::from_array(cam.target),
        );
        let intent = match command {
            "viewport.orbit" => {
                let Orbit {
                    yaw_degrees,
                    pitch_degrees,
                } = payload(data)?;
                CameraIntent::Orbit {
                    yaw_delta: yaw_degrees.to_radians(),
                    pitch_delta: pitch_degrees.to_radians(),
                }
            }
            "viewport.pan" => {
                let Pan { dx_px, dy_px } = payload(data)?;
                let (sin, cos) = cam.roll_degrees.to_radians().sin_cos();
                CameraIntent::Pan {
                    dx_px: dx_px * cos + dy_px * sin,
                    dy_px: dy_px * cos - dx_px * sin,
                }
            }
            _ => {
                let Zoom { factor } = payload(data)?;
                if factor <= 0.0 || !factor.is_finite() {
                    return Err("Zoom factor must be finite and positive".into());
                }
                CameraIntent::Zoom { factor }
            }
        };
        rig.apply_intent(intent, cam_viewport::ViewportSize::new(w, h));
        let pose = rig.rig.pose;
        let (yaw, pitch) = cam_controls::yaw_pitch_from_orientation(pose.orientation);
        self.control_camera(crate::scene::Camera {
            yaw_degrees: yaw.to_degrees() + 90.0,
            pitch_degrees: pitch.to_degrees(),
            target: pose.target.to_array(),
            distance: pose.distance / reference,
            ..cam
        })
    }
}
fn merge_settings(target: &mut Value, patch: &Value) -> Result<(), String> {
    let fields = patch
        .as_object()
        .ok_or("Settings patch must be an object")?;
    for (key, val) in fields {
        let dst = target
            .get_mut(key)
            .ok_or_else(|| format!("Unknown setting: {key}"))?;
        if val.is_object() && dst.is_object() {
            merge_settings(dst, val)?;
        } else {
            *dst = val.clone();
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rest_navigation_authors_camera_and_unknown_commands_fail() {
        let mut app = App::new();
        let ctx = egui::Context::default();
        let before = app.scene.camera;
        assert!(
            app.control_command(
                "viewport.orbit",
                json!({"yaw_degrees":20.0,"pitch_degrees":5.0}),
                &ctx
            )
            .is_ok()
        );
        assert_ne!(app.scene.camera, before);
        let camera = app.scene.camera;
        app.world.undo();
        app.evaluated_world = None;
        app.refresh_scene().unwrap();
        assert_eq!(app.scene.camera, before);
        app.world.redo();
        app.evaluated_world = None;
        app.refresh_scene().unwrap();
        assert_eq!(app.scene.camera, camera);
        assert!(
            app.control_command("invented.command", json!({}), &ctx)
                .is_err()
        );
        assert!(
            app.control_command("viewport.zoom", json!({"factor":-1.0}), &ctx)
                .is_err()
        );
    }
    #[test]
    fn rest_camera_sensor_height_and_pose_share_one_undo_step() {
        let mut app = App::new();
        let ctx = egui::Context::default();
        let before = app.scene.camera;
        app.control_command(
            "viewport.camera",
            json!({
                "sensor_height": before.sensor_height * 1.5,
                "yaw_degrees": before.yaw_degrees + 15.0
            }),
            &ctx,
        )
        .unwrap();
        assert_eq!(app.scene.camera.sensor_height, before.sensor_height * 1.5);
        assert!((app.scene.camera.yaw_degrees - before.yaw_degrees - 15.0).abs() < 0.001);
        app.world.undo();
        app.evaluated_world = None;
        app.refresh_scene().unwrap();
        assert_eq!(app.scene.camera, before);
    }

    #[test]
    fn rest_navigation_evaluates_the_document_camera_and_stops_playback() {
        let mut app = App::new();
        let ctx = egui::Context::default();
        let before = app.scene.camera;
        // Cached playback can keep a camera from an earlier frame in scene.
        app.scene.camera.yaw_degrees += 60.0;
        app.world_ui.playing = true;
        app.control_command(
            "viewport.orbit",
            json!({
                "yaw_degrees": 10.0, "pitch_degrees": 0.0
            }),
            &ctx,
        )
        .unwrap();
        assert!(!app.world_ui.playing);
        assert!(((app.scene.camera.yaw_degrees - before.yaw_degrees).abs() - 10.0).abs() < 0.001);
    }
}
