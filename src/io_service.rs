//! Background filesystem work. Settings are coalesced; explicit saves are bounded. Each command
//! runs contained: a panic in one becomes that command's error, and the worker keeps serving.
//! Dropping the service closes its queue and waits for the worker, so the last coalesced settings
//! reach the disk before the process exits.
use crate::render_service::Frame;
use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex, PoisonError, mpsc},
    thread,
};

/// Deliver a finished result to the app. When the app has gone (receiver dropped) nobody can
/// show it, so it is logged, described by `describe`, rather than lost silently.
fn deliver<T>(channel: &mpsc::Sender<T>, value: T, describe: impl FnOnce(&T) -> String) {
    if let Err(mpsc::SendError(value)) = channel.send(value) {
        log::warn!("Filesystem result has no receiver: {}", describe(&value));
    }
}

fn report(done: &mpsc::Sender<Result<String, String>>, result: Result<String, String>) {
    deliver(done, result, |result| format!("{result:?}"));
}

/// A scene event without its document: id, path and the outcome.
fn describe_scene(event: &SceneEvent) -> String {
    let outcome = match &event.result {
        Ok(_) => "ok".to_owned(),
        Err(error) => error.clone(),
    };
    format!("scene {} {}: {outcome}", event.id, event.path.display())
}

/// Run one command's work; a panic in it becomes the error `panicked` makes of its message, so
/// the requester gets an answer and the worker (and every later save) survives.
fn guarded<T, E>(
    work: impl FnOnce() -> Result<T, E>,
    panicked: impl FnOnce(String) -> E,
) -> Result<T, E> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(work)).unwrap_or_else(|payload| {
        let message = payload
            .downcast_ref::<&str>()
            .map(|s| (*s).to_owned())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "unknown panic".to_owned());
        Err(panicked(format!("File worker panicked: {message}")))
    })
}

/// Write `text` to `path` through the sibling `<file>.tmp`, flushed to the disk, then renamed over
/// the file: a failed or interrupted write leaves the old file intact, and a temp the rename could
/// not move is removed.
fn write(path: &Path, text: &str) -> std::io::Result<()> {
    use std::io::Write as _;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut name = path
        .file_name()
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("{} names no file", path.display()),
            )
        })?
        .to_os_string();
    name.push(".tmp");
    let tmp = path.with_file_name(name);
    let result = std::fs::File::create(&tmp)
        .and_then(|mut file| {
            file.write_all(text.as_bytes())?;
            file.sync_all()
        })
        .and_then(|()| std::fs::rename(&tmp, path));
    if result.is_err()
        && let Err(error) = std::fs::remove_file(&tmp)
        && error.kind() != std::io::ErrorKind::NotFound
    {
        log::warn!("Temporary file {} not removed: {error}", tmp.display());
    }
    result
}

pub enum Command {
    SaveFrame {
        frame: Arc<Frame>,
        path: PathBuf,
        file: crate::render_service::FrameFile,
        /// Nits of relative light's 1.0 in an HDR PNG (`render_service::hdr_scale`).
        sdr_white_nits: f32,
    },
    /// Read back a window screenshot submitted by the window after a presented frame and write
    /// its files (`window_shot::save`); the result arrives as a [`WindowEvent`].
    SaveWindow {
        shot: crate::window_shot::WindowShot,
        capture: soluwell_viewport::display::PendingCapture,
        /// The window's device: the readback waits on it here, never on the window's thread.
        device: wgpu::Device,
    },
    Write {
        path: PathBuf,
        text: String,
    },
    /// Delete a file (a bookmark); the outcome arrives as a [`SceneEvent`] with this `id`
    /// (`Ok(None)` when deleted), so the requester changes its list only once the file is gone.
    Delete {
        id: u64,
        path: PathBuf,
    },
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
    /// Read a take clip file (`takes::Take::from_clip`); the result arrives from `poll_take`.
    OpenTake {
        path: PathBuf,
    },
}
/// A finished window screenshot ([`Command::SaveWindow`]): the files written, or why none were.
pub struct WindowEvent {
    /// The request asked to close the window afterwards (`FRAC_SNAP`).
    pub quit: bool,
    pub result: Result<Vec<PathBuf>, crate::window_shot::ShotError>,
}

/// The outcome of a file request answered by id: a scene opened (`Ok(Some)`), a scene or
/// bookmark saved or a file deleted (`Ok(None)`), or why not.
pub struct SceneEvent {
    pub id: u64,
    pub path: PathBuf,
    pub result: Result<Option<Box<crate::scene::Scene>>, String>,
}

pub(crate) fn decode_scene(text: &str) -> Result<crate::scene::Scene, String> {
    let document = crate::world::WorldDocument::decode_json(text)
        .map_err(|e| format!("Invalid scene: {e}"))?;
    let mut scene = document.snapshot(f64::from(document.first))?;
    scene.animation.first = document.first;
    scene.animation.last = document.last;
    scene.animation.fps = document.fps;
    scene.document = Some(Box::new(document));
    Ok(scene)
}

/// The senders of the worker's results, one per kind of answer.
struct Replies {
    done: mpsc::Sender<Result<String, String>>,
    scene: mpsc::Sender<SceneEvent>,
    templates: mpsc::Sender<Result<Vec<crate::templates::Entry>, String>>,
    window: mpsc::Sender<WindowEvent>,
    take: mpsc::Sender<Result<crate::takes::Take, String>>,
    /// Woken after every answer, so the UI takes it without polling (`IoService::spawn`).
    ui: crate::render_service::Wake,
}

/// Run one command and answer it on its channel; a panic in its work is its error.
fn run(command: Command, replies: &Replies) {
    match command {
        Command::SaveFrame {
            frame,
            path,
            file,
            sdr_white_nits,
        } => {
            let result = guarded(|| frame.save(&path, file, sdr_white_nits), |e| e);
            report(
                &replies.done,
                result.map(|_| format!("Saved {}", path.display())),
            );
        }
        Command::SaveWindow {
            shot,
            capture,
            device,
        } => {
            let quit = shot.quit;
            let result = guarded(
                || {
                    capture
                        .wait(&device)
                        .map_err(crate::window_shot::ShotError::from)
                        .and_then(|capture| crate::window_shot::save(&shot, &capture))
                },
                crate::window_shot::ShotError::Worker,
            );
            deliver(&replies.window, WindowEvent { quit, result }, |event| {
                format!("window screenshot: {:?}", event.result)
            });
        }
        Command::Write { path, text } => {
            let result = guarded(|| write(&path, &text).map_err(|e| e.to_string()), |e| e);
            report(
                &replies.done,
                result.map(|_| format!("Saved {}", path.display())),
            );
        }
        Command::OpenScene { id, path } => {
            let result = guarded(
                || {
                    std::fs::read_to_string(&path)
                        .map_err(|error| error.to_string())
                        .and_then(|text| decode_scene(&text))
                        .map(|scene| Some(Box::new(scene)))
                },
                |e| e,
            );
            deliver(
                &replies.scene,
                SceneEvent { id, path, result },
                describe_scene,
            );
        }
        Command::SaveScene { id, path, document } => {
            let result = guarded(
                || {
                    serde_json::to_string_pretty(&document)
                        .map_err(|error| error.to_string())
                        .and_then(|text| write(&path, &text).map_err(|error| error.to_string()))
                        .map(|_| None)
                },
                |e| e,
            );
            deliver(
                &replies.scene,
                SceneEvent { id, path, result },
                describe_scene,
            );
        }
        Command::OpenTake { path } => {
            let result = guarded(
                || {
                    std::fs::read_to_string(&path)
                        .map_err(|error| format!("{}: {error}", path.display()))
                        .and_then(|text| crate::takes::Take::from_clip(&text))
                },
                |e| e,
            );
            deliver(&replies.take, result, |result| match result {
                Ok(take) => format!("take {}", take.name),
                Err(error) => error.clone(),
            });
        }
        Command::RefreshTemplates { directory } => {
            let result = guarded(|| crate::templates::catalog(&directory), |e| e);
            deliver(&replies.templates, result, |result| match result {
                Ok(entries) => format!("{} templates", entries.len()),
                Err(error) => error.clone(),
            });
        }
        Command::Delete { id, path } => {
            let result = guarded(
                || {
                    std::fs::remove_file(&path)
                        .map(|()| None)
                        .map_err(|e| e.to_string())
                },
                |e| e,
            );
            deliver(
                &replies.scene,
                SceneEvent { id, path, result },
                describe_scene,
            );
        }
    }
}

/// The latest settings text not yet written: written between commands, and last when the queue
/// closes.
type Pending = Arc<Mutex<Option<(PathBuf, String)>>>;

/// Write the pending settings, if any; `failed` gets the error.
fn flush(pending: &Pending, failed: impl FnOnce(String)) {
    let next = pending
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .take();
    if let Some((path, text)) = next
        && let Err(error) = guarded(|| write(&path, &text).map_err(|e| e.to_string()), |e| e)
    {
        failed(format!("Settings save {} failed: {error}", path.display()));
    }
}

pub struct IoService {
    /// The command queue; taken by `Drop` to close it before joining the worker.
    tx: Option<mpsc::SyncSender<Command>>,
    /// The worker, joined by `Drop`; None when its thread could not start (every `send` says so).
    worker: Option<thread::JoinHandle<()>>,
    settings: Pending,
    events: mpsc::Receiver<Result<String, String>>,
    scene_events: mpsc::Receiver<SceneEvent>,
    template_events: mpsc::Receiver<Result<Vec<crate::templates::Entry>, String>>,
    window_events: mpsc::Receiver<WindowEvent>,
    take_events: mpsc::Receiver<Result<crate::takes::Take, String>>,
}
impl IoService {
    /// Start the file worker; `ui` is woken whenever an answer waits for the UI.
    pub fn spawn(ui: crate::render_service::Wake) -> Self {
        let (tx, rx) = mpsc::sync_channel(8);
        let (done, events) = mpsc::channel();
        let (scene, scene_events) = mpsc::channel();
        let (templates, template_events) = mpsc::channel();
        let (window, window_events) = mpsc::channel();
        let (take, take_events) = mpsc::channel();
        let settings: Pending = Arc::new(Mutex::new(None));
        let pending = settings.clone();
        let replies = Replies {
            done,
            scene,
            templates,
            window,
            take,
            ui,
        };
        let worker = thread::Builder::new()
            .name("frac-file-writer".into())
            .spawn(move || {
                loop {
                    match rx.recv_timeout(std::time::Duration::from_millis(50)) {
                        // No lock is held here: `run` answers on channels (`Wake::wake`).
                        Ok(command) => {
                            run(command, &replies);
                            replies.ui.wake();
                        }
                        Err(mpsc::RecvTimeoutError::Disconnected) => {
                            // The app is closing: nobody reads `done` any more, so a failed
                            // final save goes to the log.
                            flush(&pending, |error| log::error!("{error}"));
                            break;
                        }
                        Err(mpsc::RecvTimeoutError::Timeout) => {}
                    }
                    flush(&pending, |error| {
                        report(&replies.done, Err(error));
                        replies.ui.wake();
                    });
                }
            });
        let worker = match worker {
            Ok(worker) => Some(worker),
            Err(error) => {
                log::error!("File writer could not start: {error}");
                None
            }
        };
        Self {
            tx: Some(tx),
            worker,
            settings,
            events,
            scene_events,
            template_events,
            window_events,
            take_events,
        }
    }
    #[cfg(test)]
    pub(crate) fn stop_for_test(&mut self) {
        self.tx.take();
        if let Some(worker) = self.worker.take() {
            worker.join().expect("file worker must stop cleanly");
        }
    }

    pub fn send(&self, cmd: Command) -> Result<(), String> {
        let tx = self.tx.as_ref().ok_or("File writer stopped")?;
        tx.try_send(cmd).map_err(|e| match e {
            mpsc::TrySendError::Full(_) => "File writer busy: its queue is full".to_owned(),
            mpsc::TrySendError::Disconnected(_) => {
                "File writer stopped: its thread could not start".to_owned()
            }
        })
    }
    pub fn settings(&self, path: PathBuf, json: String) {
        *self.settings.lock().unwrap_or_else(PoisonError::into_inner) = Some((path, json));
    }
    pub fn poll_templates(&self) -> Option<Result<Vec<crate::templates::Entry>, String>> {
        self.template_events.try_recv().ok()
    }
    pub fn poll_scene(&self) -> Option<SceneEvent> {
        self.scene_events.try_recv().ok()
    }
    pub fn poll_window(&self) -> Option<WindowEvent> {
        self.window_events.try_recv().ok()
    }
    pub fn poll_take(&self) -> Option<Result<crate::takes::Take, String>> {
        self.take_events.try_recv().ok()
    }
    pub fn poll(&self) -> Option<Result<String, String>> {
        self.events.try_recv().ok()
    }
}
impl Drop for IoService {
    /// Close the queue, then wait for the worker: it finishes the queued commands and writes the
    /// last coalesced settings before it exits, so they reach the disk before the process does.
    fn drop(&mut self) {
        self.tx.take();
        if let Some(worker) = self.worker.take()
            && worker.join().is_err()
        {
            log::error!("File writer thread panicked outside a command");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{Bulb, Formula, MandelBox};
    #[test]
    fn scene_file_worker_roundtrips_authoring_keys_materials_and_metadata() {
        use crate::world::{WorldCommand, WorldDocument, WorldEditor, WorldKind};
        let mut editor = WorldEditor::new(WorldDocument::from_scene(&crate::scene::Scene::preset(
            Formula::Mandelbulb(Bulb::PRESET),
        )));
        editor
            .execute(WorldCommand::SetTimeRange {
                first: 7,
                last: 256,
                fps: 30.0,
            })
            .unwrap();
        let object = editor.active().unwrap();
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
        let material = editor.active().unwrap();
        editor
            .execute(WorldCommand::AssignMaterial {
                id: object,
                material: Some(material),
            })
            .unwrap();
        let expected = serde_json::to_value(&editor.document).unwrap();
        let path =
            std::env::temp_dir().join(format!("frac-scene-{}.frac.json", std::process::id()));
        let io = IoService::spawn(crate::render_service::Wake::default());
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
    fn scene_decoder_reads_a_world_document_and_rejects_anything_else() {
        let scene = crate::scene::Scene::preset(Formula::Mandelbox(MandelBox::PRESET));
        let world = crate::world::WorldDocument::from_scene(&scene);
        let decoded = decode_scene(&serde_json::to_string(&world).unwrap()).unwrap();
        let document = decoded.document.as_ref().unwrap();
        assert_eq!(document.snapshot(0.0).unwrap().formula, scene.formula);
        assert!(decode_scene("{broken json").is_err());
        // The pre-world single-scene format is not read.
        assert!(decode_scene(&serde_json::to_string(&scene).unwrap()).is_err());
    }

    /// Dropping the service waits for the worker: the last coalesced settings are on disk when
    /// `drop` returns. A failed explicit save is reported, leaves the old file, and leaves no temp.
    #[test]
    fn coalesced_settings_flush_on_shutdown_and_explicit_save_reports_failure() {
        let dir = std::env::temp_dir().join(format!("frac-io-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("settings.json");
        let io = IoService::spawn(crate::render_service::Wake::default());
        for n in 0..100 {
            io.settings(path.clone(), n.to_string());
        }
        drop(io);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "99");
        assert!(!dir.join("settings.json.tmp").exists());
        let until = std::time::Instant::now() + std::time::Duration::from_secs(3);
        let io = IoService::spawn(crate::render_service::Wake::default());
        let blocker = dir.join("file");
        std::fs::write(&blocker, "original").unwrap();
        // The rename target is a directory that holds a file: the temp is written, the rename
        // fails, and the temp is removed again.
        let occupied = dir.join("occupied.json");
        std::fs::create_dir_all(occupied.join("inside")).unwrap();
        for target in [blocker.join("child.json"), occupied.clone()] {
            io.send(Command::Write {
                path: target,
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
        }
        assert_eq!(std::fs::read_to_string(&blocker).unwrap(), "original");
        assert!(!dir.join("occupied.json.tmp").exists());
        drop(io);
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// A panic in a command's work is that command's error, not the worker's end.
    #[test]
    fn a_panicking_command_is_its_own_error() {
        let result: Result<(), String> = guarded(|| panic!("boom"), |e| e);
        assert!(result.unwrap_err().contains("boom"));
        assert_eq!(guarded(|| Ok::<_, String>(7), |e| e), Ok(7));
    }

    /// A delete answers by id on the scene channel: done, then refused once the file is gone.
    #[test]
    fn delete_answers_by_id() {
        let io = IoService::spawn(crate::render_service::Wake::default());
        let path = std::env::temp_dir().join(format!("warpbro-delete-{}.json", std::process::id()));
        std::fs::write(&path, "{}").unwrap();
        let wait = |io: &IoService| {
            let until = std::time::Instant::now() + std::time::Duration::from_secs(5);
            loop {
                if let Some(event) = io.poll_scene() {
                    return event;
                }
                assert!(std::time::Instant::now() < until);
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        };
        io.send(Command::Delete {
            id: 7,
            path: path.clone(),
        })
        .unwrap();
        let deleted = wait(&io);
        assert_eq!((deleted.id, deleted.path.as_path()), (7, path.as_path()));
        assert!(deleted.result.unwrap().is_none());
        assert!(!path.exists());
        io.send(Command::Delete {
            id: 8,
            path: path.clone(),
        })
        .unwrap();
        let missing = wait(&io);
        assert_eq!(missing.id, 8);
        assert!(missing.result.is_err());
    }
}
