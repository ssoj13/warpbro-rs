//! REST decodes into the SAME typed events widgets publish. No state mutations live here.
use super::*;
use crate::events::{Action, UiCommand};
use crate::world::{NodeId, WorldCommand};
use crate::world_events::WorldAction;
use app_control::Reply;
use serde::Deserialize;
use serde_json::{Value, json};

const COMMANDS: &[&str] = &[
    "scene.new",
    "scene.load",
    "scene.document",
    "scene.open",
    "scene.save",
    "gallery.load",
    "world.command",
    "selection.set",
    "history.undo",
    "history.redo",
    "history.reset",
    "timeline.seek",
    "timeline.play",
    "timeline.pause",
    "timeline.loop",
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
    "viewport.pause",
    "viewport.flight.toggle",
    "viewport.slot.store",
    "viewport.slot.restore",
    "settings.set",
    "colour.set",
    "colour.preset",
    "colour.reload",
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
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Path {
    path: PathBuf,
}

/// Runs on the HTTP parser thread: parses values but cannot access or mutate App.
pub(super) fn decode_control(command: &str, data: Value) -> Result<UiCommand, String> {
    let app = |action| Action::App(Box::new(action));
    let action = match command {
        query if query.starts_with("get.") => Action::Query(query.to_owned()),
        "scene.new" => app(AppAction::NewScene),
        "scene.load" | "scene.document" => app(AppAction::SceneDocument(
            crate::world::WorldDocument::decode_json(&data.to_string())?,
        )),
        "scene.open" | "scene.save" => {
            let Path { path } = payload(data)?;
            app(AppAction::SceneFile {
                path,
                action: if command == "scene.open" {
                    crate::templates::FileAction::Open
                } else {
                    crate::templates::FileAction::Save
                },
            })
        }
        "gallery.load" => {
            let Index { index } = payload(data)?;
            app(AppAction::GalleryLoad(index))
        }
        "world.command" => Action::World(WorldAction::Edit {
            command: payload::<WorldCommand>(data)?,
            gesture: None,
        }),
        "selection.set" => {
            let Ids { ids } = payload(data)?;
            let lead = ids.last().copied();
            Action::World(WorldAction::Selection { ids, lead })
        }
        "history.undo" => app(AppAction::History(History::Undo)),
        "history.redo" => app(AppAction::History(History::Redo)),
        "history.reset" => app(AppAction::History(History::Reset)),
        "timeline.seek" => {
            let Seek { frame } = payload(data)?;
            Action::World(WorldAction::Seek(frame))
        }
        "timeline.play" => app(AppAction::SetPlaying(true)),
        "timeline.pause" => app(AppAction::SetPlaying(false)),
        "timeline.loop" => {
            let Enabled { enabled } = payload(data)?;
            Action::World(WorldAction::Looping(enabled))
        }
        "viewport.camera" => app(AppAction::CameraPatch(data)),
        "viewport.orbit" => {
            let Orbit {
                yaw_degrees,
                pitch_degrees,
            } = payload(data)?;
            app(AppAction::CameraOrbit {
                yaw_degrees,
                pitch_degrees,
            })
        }
        "viewport.pan" => {
            let Pan { dx_px, dy_px } = payload(data)?;
            app(AppAction::CameraPan { dx_px, dy_px })
        }
        "viewport.zoom" => {
            let Zoom { factor } = payload(data)?;
            if factor <= 0.0 || !factor.is_finite() {
                return Err("Zoom factor must be finite and positive".into());
            }
            app(AppAction::CameraZoom { factor })
        }
        "viewport.fly" => {
            let Fly {
                translation,
                rotation_degrees,
            } = payload(data)?;
            app(AppAction::CameraFly {
                translation,
                rotation_degrees,
            })
        }
        "viewport.frame" => app(AppAction::FrameCurrentCamera),
        "viewport.home" => app(AppAction::RestoreCamera),
        "viewport.pick" => {
            #[derive(Deserialize)]
            #[serde(deny_unknown_fields)]
            struct Pick {
                ndc: [f32; 2],
                #[serde(default)]
                add: bool,
                #[serde(default)]
                focus: bool,
            }
            let Pick { ndc, add, focus } = payload(data)?;
            if ndc.iter().any(|v| !v.is_finite() || v.abs() > 1.0) {
                return Err("Pick NDC must be in [-1,1]".into());
            }
            app(AppAction::Pick { ndc, add, focus })
        }
        "viewport.raw" => {
            let Enabled { enabled } = payload(data)?;
            app(AppAction::SetRaw(enabled))
        }
        "viewport.pause" => {
            let Enabled { enabled } = payload(data)?;
            app(AppAction::SetPaused(enabled))
        }
        "viewport.flight.toggle" => app(AppAction::ToggleFlight),
        "viewport.slot.store" => {
            let Index { index } = payload(data)?;
            app(AppAction::StoreCameraSlot(index))
        }
        "viewport.slot.restore" => {
            let Index { index } = payload(data)?;
            app(AppAction::RecallCameraSlot(index))
        }
        "viewport.tool" => {
            #[derive(Deserialize)]
            #[serde(deny_unknown_fields)]
            struct Tool {
                tool: String,
            }
            let Tool { tool } = payload(data)?;
            app(AppAction::SetTool(match tool.as_str() {
                "select" => ViewTool::Select,
                "move" => ViewTool::Move,
                "rotate" => ViewTool::Rotate,
                "scale" => ViewTool::Scale,
                _ => return Err("Tool must be select, move, rotate or scale".into()),
            }))
        }
        "settings.set" => app(AppAction::PatchSettings(data)),
        "colour.set" => app(AppAction::ColourSelection(payload(data)?)),
        "colour.preset" => {
            let Index { index } = payload(data)?;
            app(AppAction::ColourPreset(index))
        }
        "colour.reload" => app(AppAction::ReloadColour),
        "material.create" => {
            let Index { index } = payload(data)?;
            let preset = crate::materials::PRESETS
                .get(index)
                .ok_or("Material preset index out of range")?;
            app(AppAction::CreateMaterial {
                material: swatch_scene(preset).material,
                name: preset.name().into(),
                targets: None,
                open: false,
            })
        }
        "material.apply" => {
            let Index { index } = payload(data)?;
            app(AppAction::ApplyLibraryMaterial { index })
        }
        "export.start" => app(AppAction::StartExport),
        "export.cancel" => app(AppAction::CancelExport),
        "app.ui" => {
            let Enabled { enabled } = payload(data)?;
            app(AppAction::ShowUi(enabled))
        }
        "app.exit" => app(AppAction::Exit),
        _ => {
            return Err(format!(
                "Unknown command: {command}; GET /api/commands lists supported commands"
            ));
        }
    };
    Ok(UiCommand {
        document: None,
        frame: None,
        camera: None,
        action,
    })
}
impl App {
    pub(super) fn poll_control(&mut self, _ctx: &egui::Context) {
        if self.control_started {
            return;
        }
        self.control_started = true;
        crate::render_diagnostics::set_wake(self.wake.clone());
        #[cfg(not(test))]
        if std::env::var("WARPBRO_API").as_deref() != Ok("0") {
            let port = std::env::var("WARPBRO_API_PORT")
                .map_or(Ok(9877), |v| v.parse::<u16>().map_err(|e| e.to_string()));
            let wake = self.wake.clone();
            let emitter = self.event_emitter();
            match port.and_then(|port| {
                app_control::Server::start(
                    port,
                    emitter,
                    Arc::new(decode_control),
                    Arc::new(move || wake.wake()),
                )
            }) {
                Ok(server) => {
                    log::info!(
                        "WarpBro EventBus REST API: http://127.0.0.1:{}/api/health",
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

    pub(super) fn control_query(&self, query: &str, _ctx: &egui::Context) -> Result<Reply, String> {
        if query == "get.screenshot.png" {
            let frame = self.current().ok_or("No rendered viewport frame")?;
            if let Some(error) = &frame.colour_error {
                return Err(error.clone());
            }
            let mut bytes = Vec::new();
            {
                let mut encoder =
                    png::Encoder::new(&mut bytes, frame.width as u32, frame.height as u32);
                encoder.set_color(png::ColorType::Rgba);
                encoder.set_depth(png::BitDepth::Eight);
                let mut writer = encoder.write_header().map_err(|e| e.to_string())?;
                writer
                    .write_image_data(bytemuck::cast_slice(&frame.pixels))
                    .map_err(|e| e.to_string())?;
            }
            return Ok(Reply::Bytes {
                content_type: "image/png",
                data: bytes,
            });
        }
        let result=match query {
            "get.status"=>self.control_status(),
            "get.commands"=>json!({"commands":COMMANDS,"event_bus":"playa_events::EventBus","world_command_format":"externally tagged serde enum; world.command accepts every WorldCommand including Batch","queries":["status","commands","scene","nodes","gallery","materials","takes","camera","render","diagnostics","settings","screenshot.png","attributes/<uuid>"]}),
            "get.scene"=>value(&self.world.document)?,
            "get.camera"=>value(self.world.document.snapshot(f64::from(self.world_ui.playhead))?.camera)?,
            "get.settings"=>value(self.current_settings())?,
            "get.render"=>self.control_status()["render"].clone(),
            "get.diagnostics"=>crate::render_diagnostics::snapshot(),
            "get.nodes"=>Value::Array(self.world.document.nodes().into_iter().map(|n|json!({"id":n.id,"kind":n.kind,"name":n.name,"parent":n.parent,"visible":n.visible,"locked":n.locked,"solo":n.solo,"start":n.start,"end":n.end})).collect()),
            "get.gallery"=>Value::Array(self.gallery.iter().enumerate().map(|(i,e)|json!({"index":i,"name":e.scene.name,"formula":e.scene.formula,"thumbnail":e.thumb.is_some()})).collect()),
            "get.materials"=>Value::Array(crate::materials::PRESETS.iter().enumerate().map(|(i,m)|json!({"index":i,"preset":m.name()})).collect()),
            "get.takes"=>value(self.world.document.takes.list())?,
            q if q.starts_with("get.attributes.")=>{
                let id:NodeId=payload(json!(&q["get.attributes.".len()..]))?;
                Value::Array(self.world.document.attributes(id,f64::from(self.world_ui.playhead))?.into_iter().map(|a|json!({"path":a.path,"label":a.label,"value":a.value,"keyable":a.keyable,"frames":a.frames,"choices":a.choices,"range":a.range})).collect())
            },
            _=>return Err(format!("Unknown query: {query}")),
        };
        Ok(Reply::Json(result))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::UiEvent;
    fn publish(app: &App, command: &str, data: Value) {
        app.event_emitter()
            .try_emit(UiEvent::ui(decode_control(command, data).unwrap()))
            .unwrap_or_else(|_| panic!("EventBus full"));
    }
    #[test]
    fn stopped_file_worker_returns_http_error_without_changing_document_identity() {
        use std::io::{Read, Write};
        use std::time::{Duration, Instant};
        let _gpu_test = crate::test_gpu::lock();
        let mut app = App::new();
        app.io.stop_for_test();
        let before = serde_json::to_value(&app.world.document).unwrap();
        let server = app_control::Server::start(
            0,
            app.event_emitter(),
            Arc::new(decode_control),
            Arc::new(|| {}),
        )
        .unwrap();
        let port = server.port();
        let client = std::thread::spawn(move || {
            let body = json!({"path":"not-written.frac.json"}).to_string();
            let mut socket = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
            socket
                .set_read_timeout(Some(Duration::from_secs(3)))
                .unwrap();
            write!(socket, "POST /api/scene/save HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body).unwrap();
            let mut result = String::new();
            socket.read_to_string(&mut result).unwrap();
            result
        });
        let started = Instant::now();
        while app.event_bus.queue_len() == 0 {
            assert!(started.elapsed() < Duration::from_secs(2));
            std::thread::yield_now();
        }
        app.handle_events(&egui::Context::default());
        let response = client.join().unwrap();
        assert!(response.contains("400 Bad Request"), "{response}");
        assert!(response.contains("File writer stopped"), "{response}");
        assert_eq!(serde_json::to_value(&app.world.document).unwrap(), before);
        assert!(app.scene_file_pending.is_none());
        assert!(app.scene_file_path.is_none());
    }

    #[test]
    fn settings_and_camera_keys_use_the_accepted_interpolation_in_one_drain() {
        let _gpu_test = crate::test_gpu::lock();
        let mut app = App::new();
        let ctx = egui::Context::default();
        let camera = app.world.document.active_camera.unwrap();
        let before = app.scene.camera.fov_y_degrees;
        app.world_ui.seek(10);
        publish(
            &app,
            "settings.set",
            json!({"new_key": curves::Tan::Linear, "auto_key":true}),
        );
        publish(
            &app,
            "scene.document",
            serde_json::to_value(&app.world.document).unwrap(),
        );
        publish(&app, "timeline.seek", json!({"frame":10}));
        publish(
            &app,
            "world.command",
            json!(WorldCommand::Key {
                id: camera,
                path: "/camera/fov_y_degrees".into(),
                frame: 0.0,
            }),
        );
        app.emit_app(AppAction::CameraPatch(json!({"fov_y_degrees":before+40.0})));
        app.handle_events(&ctx);
        let quarter = app
            .world
            .document
            .attribute_value(camera, "/camera/fov_y_degrees", 2.5)
            .unwrap()
            .as_f64()
            .unwrap();
        assert!(
            (quarter - f64::from(before + 10.0)).abs() < 0.001,
            "{quarter}"
        );
        assert_eq!(app.world_ui.playhead, 10);
    }

    #[test]
    fn a_noop_colour_event_keeps_the_current_frame_after_an_earlier_seek() {
        let _gpu_test = crate::test_gpu::lock();
        let mut app = App::new();
        let ctx = egui::Context::default();
        app.world_ui.auto_key = true;
        app.emit_app(AppAction::CameraPatch(json!({"fov_y_degrees":40.0})));
        app.handle_events(&ctx);
        app.world_ui.seek(10);
        app.emit_app(AppAction::CameraPatch(json!({"fov_y_degrees":80.0})));
        app.handle_events(&ctx);
        app.world_ui.seek(0);
        app.refresh_scene().unwrap();
        assert_eq!(app.scene.camera.fov_y_degrees, 40.0);
        assert!(!app.scene.render.reinhard);
        let revision = app.world.revision();
        let colour = app.scene.colour.clone();
        publish(&app, "timeline.seek", json!({"frame":10}));
        app.emit_app(AppAction::ColourSelection(colour));
        app.handle_events(&ctx);
        assert_eq!(
            app.world.revision(),
            revision,
            "colour edit is a true no-op"
        );
        assert_eq!(app.world_ui.playhead, 10);
        assert_eq!(app.scene.camera.fov_y_degrees, 80.0);
        assert_eq!(
            app.scene.camera,
            app.world.document.snapshot(10.0).unwrap().camera
        );
    }

    #[test]
    fn rest_and_ui_navigation_publish_then_use_the_same_dispatcher() {
        let _gpu_test = crate::test_gpu::lock();
        let mut ui = App::new();
        let mut rest = App::new();
        let ctx = egui::Context::default();
        rest.world = ui.world.clone();
        rest.evaluated_world = None;
        rest.refresh_scene().unwrap();
        let before = ui.scene.camera;
        ui.emit_app(AppAction::CameraOrbit {
            yaw_degrees: 20.0,
            pitch_degrees: 5.0,
        });
        publish(
            &rest,
            "viewport.orbit",
            json!({"yaw_degrees":20.0,"pitch_degrees":5.0}),
        );
        assert_eq!(ui.scene.camera, before);
        assert_eq!(rest.scene.camera, before);
        ui.handle_events(&ctx);
        rest.handle_events(&ctx);
        assert_ne!(ui.scene.camera, before);
        assert_eq!(ui.scene.camera, rest.scene.camera);
        publish(&rest, "history.undo", json!({}));
        rest.handle_events(&ctx);
        assert_eq!(rest.scene.camera, before);
        assert!(decode_control("invented.command", json!({})).is_err());
        assert!(decode_control("viewport.zoom", json!({"factor":-1.0})).is_err());
    }
    #[test]
    fn camera_lens_and_colour_changes_are_undoable_bus_actions() {
        let _gpu_test = crate::test_gpu::lock();
        let mut app = App::new();
        let ctx = egui::Context::default();
        let before = app.scene.camera;
        publish(
            &app,
            "viewport.camera",
            json!({"sensor_height":before.sensor_height*1.5,"yaw_degrees":before.yaw_degrees+15.0}),
        );
        app.handle_events(&ctx);
        assert_eq!(app.scene.camera.sensor_height, before.sensor_height * 1.5);
        publish(&app, "history.undo", json!({}));
        app.handle_events(&ctx);
        assert_eq!(app.scene.camera, before);
        let old = app.scene.colour.clone();
        publish(
            &app,
            "colour.set",
            json!({"on":false,"config":"","display":"","view":"","look":"","working_input":""}),
        );
        assert_eq!(app.scene.colour, old);
        app.handle_events(&ctx);
        assert!(!app.scene.colour.on);
        publish(&app, "history.undo", json!({}));
        app.handle_events(&ctx);
        assert_eq!(app.scene.colour, old);
    }
    #[test]
    fn viewport_intent_cannot_edit_a_different_active_camera() {
        let _gpu_test = crate::test_gpu::lock();
        let mut app = App::new();
        let ctx = egui::Context::default();
        app.emit_app(AppAction::CameraOrbit {
            yaw_degrees: 20.0,
            pitch_degrees: 0.0,
        });
        app.world.document.active_camera = None;
        let before = serde_json::to_value(&app.world.document).unwrap();
        app.handle_events(&ctx);
        assert_eq!(serde_json::to_value(&app.world.document).unwrap(), before);
        assert!(app.status.contains("no longer active"));
    }
    #[test]
    fn stale_document_action_is_rejected_without_overwriting_current_scene() {
        let _gpu_test = crate::test_gpu::lock();
        let mut app = App::new();
        let ctx = egui::Context::default();
        let before = app.scene.camera;
        app.emit_app(AppAction::NewScene);
        app.emit_app(AppAction::CameraOrbit {
            yaw_degrees: 20.0,
            pitch_degrees: 0.0,
        });
        app.handle_events(&ctx);
        assert_eq!(app.scene.camera, before);
        assert!(app.status.contains("no longer open"));
    }
}
