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
        exr: bool,
    },
    Write {
        path: PathBuf,
        text: String,
    },
    Delete(PathBuf),
}
pub struct IoService {
    tx: mpsc::SyncSender<Command>,
    settings: Arc<Mutex<Option<(PathBuf, String)>>>,
    events: mpsc::Receiver<Result<String, String>>,
}
impl IoService {
    pub fn spawn() -> Self {
        let (tx, rx) = mpsc::sync_channel(8);
        let (done, events) = mpsc::channel();
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
                        Ok(Command::SaveFrame { frame, path, exr }) => {
                            let result = if exr {
                                frame.save_display_exr(&path)
                            } else {
                                frame.save_png(&path)
                            };
                            let _ = done.send(result.map(|_| format!("Saved {}", path.display())));
                        }
                        Ok(Command::Write { path, text }) => {
                            let _ = done.send(
                                write(&path, &text)
                                    .map(|_| format!("Saved {}", path.display()))
                                    .map_err(|e| e.to_string()),
                            );
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
    pub fn poll(&self) -> Option<Result<String, String>> {
        self.events.try_recv().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
