//! Background filesystem work. Settings are coalesced; explicit saves are bounded.
use crate::render_service::Frame;
use std::{
    path::PathBuf,
    sync::{Arc, Mutex, mpsc},
    thread,
};

pub enum Command {
    SaveFrame {
        frame: Arc<Frame>,
        path: PathBuf,
        file: crate::render_service::FrameFile,
        /// Mastering peak of an HDR PNG.
        peak_nits: f32,
    },
    Write {
        path: PathBuf,
        text: String,
    },
    Delete(PathBuf),
    RefreshTemplates {
        directory: PathBuf,
    },
    OpenScene {
        id: u64,
        path: PathBuf,
    },
    SaveScene {
        id: u64,
        path: PathBuf,
        document: Box<crate::world::WorldDocument>,
    },
}
pub struct SceneEvent {
    pub id: u64,
    pub path: PathBuf,
    pub result: Result<Option<Box<crate::scene::Scene>>, String>,
}

pub(crate) fn decode_scene(text: &str) -> Result<crate::scene::Scene, String> {
    let document = match serde_json::from_str::<crate::world::WorldDocument>(text) {
        Ok(document) => document,
        Err(world_error) => {
            let legacy =
                serde_json::from_str::<crate::scene::Scene>(text).map_err(|legacy_error| {
                    format!("Invalid scene: {world_error}; legacy scene: {legacy_error}")
                })?;
            crate::world::WorldDocument::from_scene(&legacy)
        }
    };
    let mut scene = document.snapshot(f64::from(document.first))?;
    scene.animation.first = document.first;
    scene.animation.last = document.last;
    scene.animation.fps = document.fps;
    scene.document = Some(Box::new(document));
    Ok(scene)
}

pub struct IoService {
    tx: mpsc::SyncSender<Command>,
    settings: Arc<Mutex<Option<(PathBuf, String)>>>,
    events: mpsc::Receiver<Result<String, String>>,
    scene_events: mpsc::Receiver<SceneEvent>,
    template_events: mpsc::Receiver<Result<Vec<crate::templates::Entry>, String>>,
}
impl IoService {
    pub fn spawn() -> Self {
        let (tx, rx) = mpsc::sync_channel(8);
        let (done, events) = mpsc::channel();
        let (scene_done, scene_events) = mpsc::channel();
        let (template_done, template_events) = mpsc::channel();
        let settings = Arc::new(Mutex::new(None::<(PathBuf, String)>));
        let pending = settings.clone();
        thread::Builder::new()
            .name("frac-file-writer".into())
            .spawn(move || {
                let write = |path: &std::path::Path, text: &str| {
                    if let Some(dir) = path.parent() {
                        std::fs::create_dir_all(dir)?;
                    }
                    // Write a sibling then replace: a failed write leaves the old file intact.
                    let tmp = path.with_extension("json.tmp");
                    std::fs::write(&tmp, text)?;
                    std::fs::rename(tmp, path)
                };
                loop {
                    match rx.recv_timeout(std::time::Duration::from_millis(50)) {
                        Ok(Command::SaveFrame { frame, path, file, peak_nits }) => {
                            let result = frame.save(&path, file, peak_nits);
                            let _ = done.send(result.map(|_| format!("Saved {}", path.display())));
                        }
                        Ok(Command::Write { path, text }) => {
                            let _ = done.send(
                                write(&path, &text)
                                    .map(|_| format!("Saved {}", path.display()))
                                    .map_err(|e| e.to_string()),
                            );
                        }
                        Ok(Command::OpenScene { id, path }) => {
                            let result = std::fs::read_to_string(&path)
                                .map_err(|error| error.to_string())
                                .and_then(|text| decode_scene(&text))
                                .map(|scene| Some(Box::new(scene)));
                            let _ = scene_done.send(SceneEvent { id, path, result });
                        }
                        Ok(Command::SaveScene { id, path, document }) => {
                            let result = serde_json::to_string_pretty(&document)
                                .map_err(|error| error.to_string())
                                .and_then(|text| {
                                    write(&path, &text).map_err(|error| error.to_string())
                                })
                                .map(|_| None);
                            let _ = scene_done.send(SceneEvent { id, path, result });
                        }
                        Ok(Command::RefreshTemplates { directory }) => {
                            let _ = template_done.send(crate::templates::catalog(&directory));
                        }
                        Ok(Command::Delete(path)) => {
                            if let Err(e) = std::fs::remove_file(path) {
                                let _ = done.send(Err(e.to_string()));
                            }
                        }
                        Err(mpsc::RecvTimeoutError::Disconnected) => {
                            if let Some((path, text)) = pending.lock().unwrap().take() {
                                let _ = write(&path, &text);
                            }
                            break;
                        }
                        Err(mpsc::RecvTimeoutError::Timeout) => {}
                    }
                    let next = pending.lock().unwrap().take();
                    if let Some((path, text)) = next
                        && let Err(e) = write(&path, &text)
                    {
                        let _ = done.send(Err(format!("Settings save failed: {e}")));
                    }
                }
            })
            .expect("filesystem worker");
        Self {
            tx,
            settings,
            events,
            scene_events,
            template_events,
        }
    }
    pub fn send(&self, cmd: Command) -> Result<(), String> {
        self.tx
            .try_send(cmd)
            .map_err(|e| format!("File writer busy: {e}"))
    }
    pub fn settings(&self, path: PathBuf, json: String) {
        *self.settings.lock().unwrap() = Some((path, json));
    }
    pub fn poll_templates(&self) -> Option<Result<Vec<crate::templates::Entry>, String>> {
        self.template_events.try_recv().ok()
    }
    pub fn poll_scene(&self) -> Option<SceneEvent> {
        self.scene_events.try_recv().ok()
    }
    pub fn poll(&self) -> Option<Result<String, String>> {
        self.events.try_recv().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scene_file_worker_roundtrips_authoring_keys_materials_and_metadata() {
        use crate::world::{WorldCommand, WorldDocument, WorldEditor, WorldKind};
        let mut editor = WorldEditor::new(WorldDocument::from_scene(&crate::scene::Scene::preset(
            crate::params::FAMILY_BULB,
        )));
        editor
            .execute(WorldCommand::SetTimeRange {
                first: 7,
                last: 256,
                fps: 30.0,
            })
            .unwrap();
        let object = editor.selection.unwrap();
        editor
            .execute(WorldCommand::SetMetadata {
                id: object,
                path: "/asset".into(),
                value: serde_json::json!({"owner":"artist","nested":[1,true]}),
            })
            .unwrap();
        editor
            .execute(WorldCommand::Key {
                id: object,
                path: "/transform/position/0".into(),
                frame: 12.25,
            })
            .unwrap();
        editor
            .execute(WorldCommand::Create {
                kind: WorldKind::Material,
                name: "Saved material".into(),
                parent: None,
            })
            .unwrap();
        let material = editor.selection.unwrap();
        editor
            .execute(WorldCommand::AssignMaterial {
                id: object,
                material: Some(material),
            })
            .unwrap();
        let expected = serde_json::to_value(&editor.document).unwrap();
        let path =
            std::env::temp_dir().join(format!("frac-scene-{}.frac.json", std::process::id()));
        let io = IoService::spawn();
        io.send(Command::SaveScene {
            id: 1,
            path: path.clone(),
            document: Box::new(editor.document),
        })
        .unwrap();
        fn wait(io: &IoService) -> SceneEvent {
            let until = std::time::Instant::now() + std::time::Duration::from_secs(5);
            loop {
                if let Some(event) = io.poll_scene() {
                    return event;
                }
                assert!(std::time::Instant::now() < until);
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        }
        let saved = wait(&io);
        assert_eq!(saved.id, 1);
        assert!(saved.result.unwrap().is_none());
        io.send(Command::OpenScene {
            id: 2,
            path: path.clone(),
        })
        .unwrap();
        let loaded = wait(&io).result.unwrap().unwrap();
        assert_eq!(
            (
                loaded.animation.first,
                loaded.animation.last,
                loaded.animation.fps
            ),
            (7, 256, 30.0)
        );
        assert_eq!(
            serde_json::to_value(loaded.document.as_ref().unwrap()).unwrap(),
            expected
        );
        std::fs::remove_file(&path).unwrap();
        io.send(Command::OpenScene { id: 3, path }).unwrap();
        assert!(wait(&io).result.is_err());
    }

    #[test]
    fn scene_decoder_migrates_legacy_and_rejects_invalid_input() {
        let legacy = crate::scene::Scene::preset(crate::params::FAMILY_BOX);
        let decoded = decode_scene(&serde_json::to_string(&legacy).unwrap()).unwrap();
        let document = decoded.document.as_ref().unwrap();
        assert_eq!(document.snapshot(0.0).unwrap().formula, legacy.formula);
        assert!(decode_scene("{broken json").is_err());
    }

    #[test]
    fn coalesced_settings_flush_on_shutdown_and_explicit_save_reports_failure() {
        let dir = std::env::temp_dir().join(format!("frac-io-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("settings.json");
        let io = IoService::spawn();
        for n in 0..100 {
            io.settings(path.clone(), n.to_string());
        }
        drop(io);
        let until = std::time::Instant::now() + std::time::Duration::from_secs(3);
        while std::fs::read_to_string(&path).ok().as_deref() != Some("99") {
            assert!(
                std::time::Instant::now() < until,
                "last settings must flush after drop"
            );
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        let io = IoService::spawn();
        let blocker = dir.join("file");
        std::fs::write(&blocker, "original").unwrap();
        io.send(Command::Write {
            path: blocker.join("child.json"),
            text: "{}".into(),
        })
        .unwrap();
        loop {
            if let Some(result) = io.poll() {
                assert!(result.is_err());
                break;
            }
            assert!(std::time::Instant::now() < until);
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert_eq!(std::fs::read_to_string(&blocker).unwrap(), "original");
        drop(io);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
