//! Rendering is owned by one worker thread, which drives CUDA or WGPU through `render::Gpu`.
//! UI requests and completed viewport frames occupy single replaceable slots; commands and
//! completion events have bounded queues.
use std::collections::VecDeque;
use std::path::Path;
use std::sync::mpsc::{SyncSender, TrySendError};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError, TryLockError};
use std::thread;
use std::time::{Duration, Instant};

use crate::render::{Batch, Gpu, RenderError, Target, validate_size};
use crate::scene::Scene;

pub use video_io::export::{HdrLevels, PngEncoding, hdr_scale, write_png};
#[cfg(test)]
use video_io::{color::HLG_REF_PEAK, export::HdrScale};

const QUEUE_LIMIT: usize = 16;

/// The batch budget callers give background jobs (exports, preview caching, thumbnails) in
/// their requests. A GPU launch cannot be preempted, so a background batch that may run while
/// the viewport moves stays this short ([`background_batch_allowed`]).
pub const BACKGROUND_BUDGET_MS: f32 = 8.0;

#[derive(Clone, Debug, PartialEq)]
pub struct ViewportRequest {
    pub generation: u64,
    pub scene: Scene,
    pub width: usize,
    pub height: usize,
    pub target_spp: u32,
    pub paused: bool,
    /// Show the raw samples instead of the denoised image (viewport A/B; the scene's denoise
    /// settings, and exports, are untouched).
    pub raw: bool,
    pub active: bool,
    pub interactive: bool,
    /// Selected moving profile in Auto mode; the worker never substitutes hidden quality.
    pub preview: bool,
    pub batch_budget_ms: f32,
    pub target_fps: f32,
    pub seed: u32,
    pub output_hdr: bool,
    pub white_nits: f32,
}

/// The file a viewport snapshot (or a PNG export) is saved as. An SDR PNG holds the SDR
/// rendering (an HDR view is rendered for SDR too, `Frame::sdr_bytes`); an HDR PNG and the
/// display EXR hold the display light an HDR monitor shows, unclipped above SDR white.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameFile {
    Png(PngEncoding),
    /// Linear display light, display primaries, `whiteLuminance` = the nits of its 1.0 in the
    /// same frame's HDR10 PNG ([`write_display_exr`]).
    DisplayExr,
}
impl FrameFile {
    /// The suffix after the file stem (`PngEncoding::suffix` for PNGs).
    pub fn suffix(self) -> &'static str {
        match self {
            Self::Png(encoding) => encoding.suffix(),
            Self::DisplayExr => "display.exr",
        }
    }
}

/// CPU data only. The GPU context, buffers and progressive targets never leave the worker.
pub struct Frame {
    pub generation: u64,
    pub preview: bool,
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<u32>,
    pub light: Vec<[f32; 4]>,
    /// Unexposed linear ACEScg radiance, populated for final EXR export frames only.
    pub radiance: Vec<[f32; 4]>,
    /// Relative (SDR view) or absolute (HDR view, with its peak) display light.
    pub light_kind: crate::color::DisplayLight,
    pub colour_error: Option<String>,
    pub denoised_samples: u32,
    pub denoise_ms: f32,
    pub denoise_error: Option<String>,
    pub samples: u32,
    /// Tiles still sampling / all tiles, after the last adaptive decision.
    pub active_tiles: (usize, usize),
    /// Adaptive sampling found every tile converged before `samples` reached the target.
    pub converged: bool,
    pub last_ms: f32,
    pub last_spp: u32,
    /// Share of the accumulated samples with an unresolved march ([`crate::render::Target::unresolved`]).
    pub unresolved: f32,
    pub sdr_bytes: Arc<Vec<u8>>,
    /// Extended-sRGB encoded RGBA32F, prepared with the requested output reference white.
    pub hdr_bytes: Arc<Vec<u8>>,
}
/// Write linear Rec.709 display light as the display EXR: float RGB, BT.709 `chromaticities`,
/// `whiteLuminance` = the nits of the light's 1.0 in the same frame's HDR10 PNG (`hdr_scale`):
/// 100 for an HDR view's absolute light, `sdr_white_nits` for relative light, so the EXR and
/// the PNG agree on luminance. The one display-EXR writer: viewport snapshots, the CLI's
/// `--display-exr` and window screenshots (`window_shot`).
pub fn write_display_exr(
    path: &Path,
    width: usize,
    height: usize,
    light: &[[f32; 4]],
    kind: crate::color::DisplayLight,
    sdr_white_nits: f32,
    overwrite: bool,
) -> Result<(), String> {
    let white = hdr_scale(kind, light, sdr_white_nits, PngEncoding::Hdr10).unit_nits;
    crate::exr_io::write_rgb(
        path,
        width,
        height,
        light,
        &crate::color::DISPLAY_PRIMS,
        Some(white),
        overwrite,
    )
}
pub(crate) fn hdr_canvas_bytes_into(light: &[[f32; 4]], gain: f32, out: &mut Vec<u8>) {
    out.resize(light.len() * 16, 0);
    let chunks = out.as_chunks_mut::<16>().0;
    let encode_pixel = |(dst, pixel): (&mut [u8; 16], &[f32; 4])| {
        let r = crate::color::oetf(pixel[0] * gain).to_ne_bytes();
        let g = crate::color::oetf(pixel[1] * gain).to_ne_bytes();
        let b = crate::color::oetf(pixel[2] * gain).to_ne_bytes();
        let a = 1.0f32.to_ne_bytes();
        dst[0..4].copy_from_slice(&r);
        dst[4..8].copy_from_slice(&g);
        dst[8..12].copy_from_slice(&b);
        dst[12..16].copy_from_slice(&a);
    };
    if light.len() >= 16_384 {
        use rayon::prelude::*;
        chunks
            .par_iter_mut()
            .zip(light.par_iter())
            .for_each(encode_pixel);
    } else {
        chunks.iter_mut().zip(light.iter()).for_each(encode_pixel);
    }
}
#[cfg(test)]
pub(crate) fn hdr_canvas_bytes(light: &[[f32; 4]], gain: f32) -> Vec<u8> {
    let mut out = Vec::with_capacity(light.len() * 16);
    hdr_canvas_bytes_into(light, gain, &mut out);
    out
}
/// Clear and refill a frame's shared byte buffer in place while the frame is its only owner,
/// or start a new one while the UI's upload still holds the old one (it is never written).
fn refill(bytes: &mut Arc<Vec<u8>>, fill: impl FnOnce(&mut Vec<u8>)) {
    match Arc::get_mut(bytes) {
        Some(owned) => {
            owned.clear();
            fill(owned);
        }
        None => {
            let mut fresh = Vec::new();
            fill(&mut fresh);
            *bytes = Arc::new(fresh);
        }
    }
}
impl Frame {
    /// A CPU copy of `target`'s display in fresh buffers ([`Self::snapshot_into`]).
    pub(crate) fn snapshot(
        target: &Target,
        generation: u64,
        preview: bool,
        output_hdr: bool,
        white_nits: f32,
    ) -> Self {
        let mut frame = empty_presentation();
        frame.snapshot_into(target, generation, preview, output_hdr, white_nits);
        frame
    }
    /// Copy `target`'s display into this frame, reusing its buffers: the viewport publishes into
    /// pooled frames ([`PresentationPool`]) instead of allocating about 0.3 GB per 4K publish.
    /// The scene-linear radiance is not part of a display copy (exports read it separately).
    pub(crate) fn snapshot_into(
        &mut self,
        target: &Target,
        generation: u64,
        preview: bool,
        output_hdr: bool,
        white_nits: f32,
    ) {
        self.pixels.clone_from(&target.pixels);
        self.light.clone_from(&target.light);
        refill(&mut self.sdr_bytes, |bytes| {
            bytes.extend_from_slice(bytemuck::cast_slice(&target.pixels))
        });
        let gain = if target.light_kind.hdr() && output_hdr {
            100.0 / white_nits.max(1.0)
        } else {
            1.0
        };
        refill(&mut self.hdr_bytes, |bytes| {
            if output_hdr {
                hdr_canvas_bytes_into(&target.light, gain, bytes);
            }
        });
        self.generation = generation;
        self.preview = preview;
        self.width = target.width;
        self.height = target.height;
        self.radiance.clear();
        self.light_kind = target.light_kind;
        self.colour_error.clone_from(&target.colour_error);
        // What the image shows: a raw viewport A/B reports no denoise.
        self.denoised_samples = if target.denoise_shown() {
            target.denoise.last_samples
        } else {
            0
        };
        self.denoise_ms = target.denoise.last_ms;
        self.denoise_error.clone_from(&target.denoise.error);
        self.samples = target.samples;
        self.active_tiles = target.active_tiles;
        self.converged = target.converged;
        self.last_ms = target.last_ms;
        self.last_spp = target.last_spp;
        self.unresolved = target.unresolved;
    }
    /// Finished for a target of `spp`: all samples taken, or adaptive sampling converged.
    pub fn complete(&self, spp: u32) -> bool {
        self.samples >= spp || self.converged
    }
    /// Samples per second of the last traced batch, in millions (status bars, the GPU bench).
    pub fn msamples_per_s(&self) -> f64 {
        if self.last_ms <= 0.0 {
            return 0.0;
        }
        (self.width * self.height) as f64 * self.last_spp as f64
            / (self.last_ms as f64 / 1000.0)
            / 1.0e6
    }
    /// `sdr_white_nits`: the nits of relative light's 1.0 in an HDR file (`hdr_scale`).
    /// Returns what an HDR file records (`write_png`).
    pub fn save_png(
        &self,
        path: &Path,
        encoding: PngEncoding,
        sdr_white_nits: f32,
        overwrite: bool,
    ) -> Result<Option<HdrLevels>, String> {
        if let Some(e) = &self.colour_error {
            return Err(format!("Colour transform failed: {e}"));
        }
        let scale = hdr_scale(self.light_kind, &self.light, sdr_white_nits, encoding);
        write_png(
            path,
            self.width,
            self.height,
            &self.light,
            || self.sdr_bytes.as_ref().clone(),
            encoding,
            scale,
            overwrite,
        )
        .map_err(|e| e.to_string())
    }
    /// The display EXR ([`write_display_exr`]); `sdr_white_nits` as in [`Self::save_png`].
    pub fn save_display_exr(&self, path: &Path, sdr_white_nits: f32) -> Result<(), String> {
        if let Some(e) = &self.colour_error {
            return Err(format!("Colour transform failed: {e}"));
        }
        write_display_exr(
            path,
            self.width,
            self.height,
            &self.light,
            self.light_kind,
            sdr_white_nits,
            true,
        )
    }
    /// Save this frame as `file` (viewport snapshots: File menu and the viewport toolbar);
    /// `sdr_white_nits` as in [`Self::save_png`].
    pub fn save(&self, path: &Path, file: FrameFile, sdr_white_nits: f32) -> Result<(), String> {
        match file {
            FrameFile::Png(encoding) => self
                .save_png(path, encoding, sdr_white_nits, true)
                .map(|_| ()),
            FrameFile::DisplayExr => self.save_display_exr(path, sdr_white_nits),
        }
    }
}

pub enum Command {
    /// The scene is boxed in both render commands so Command stays small in every queue
    /// and in Result<(), Command> hand-backs. A thumbnail's quality is its request's: `spp`,
    /// `seed`, the batch budget and `max_bounces`, the bounce cap applied to the evaluated
    /// scene (a bookmark's frozen world carries its own render settings).
    Thumbnail {
        id: u64,
        scene: Box<Scene>,
        width: usize,
        height: usize,
        spp: u32,
        seed: u32,
        budget_ms: f32,
        max_bounces: u32,
    },
    /// One export frame at the request's quality: `spp`, `seed` and the batch budget.
    RenderExport {
        id: u64,
        scene: Box<Scene>,
        width: usize,
        height: usize,
        spp: u32,
        seed: u32,
        budget_ms: f32,
        /// Read the scene-linear radiance back at completion (`Frame::radiance`): only the
        /// formats that store it (EXR) pay for the extra readback.
        radiance: bool,
        /// A sequence coordinator receives these directly, independently of GUI polling.
        reply: Option<SyncSender<RenderEvent>>,
    },
    CancelExport,
    /// The OCIO config was read again from disk (the Colour panel's Reload) with the selection
    /// unchanged: the viewport redisplays. The new load has a new `Ocio::serial`, so the display
    /// key and the transform cache see it themselves; a selection change needs no command, its
    /// request differs.
    ReloadColour,
    /// A viewport click: the fractal of `scene` (the viewport's) under `ndc` (`Gpu::pick`),
    /// answered with [`RenderEvent::Picked`].
    Pick {
        id: u64,
        scene: Box<Scene>,
        width: usize,
        height: usize,
        ndc: [f32; 2],
    },
    BeginPreview {
        request: crate::preview::PreviewRequest,
        cache_all: bool,
    },
    /// Replaceable completed viewport frame; pixel copying stays on the worker.
    CacheViewport {
        request: crate::preview::PreviewRequest,
        number: u32,
        frame: Arc<Frame>,
    },
    SeekPreview {
        generation: u64,
        number: u32,
    },
    CancelPreview {
        generation: u64,
    },
    RecyclePreviewFrame {
        frame: Arc<Frame>,
    },
}

/// Why a command was not queued ([`RenderPort::try_command`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CommandError {
    /// The queue (or the preview mailbox) is momentarily full: retry on a later tick.
    Full,
    /// The render worker is gone (it failed to start, panicked or shut down).
    Stopped,
    /// The request can never be accepted.
    Invalid(String),
}
impl std::fmt::Display for CommandError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Full => f.write_str("Renderer queue is full; retry next UI tick"),
            Self::Stopped => f.write_str("Render worker is unavailable"),
            Self::Invalid(error) => f.write_str(error),
        }
    }
}
impl std::error::Error for CommandError {}

#[derive(Clone)]
pub enum RenderEvent {
    Ready {
        name: String,
    },
    /// A viewport or worker error for the status line; never a job's terminal event.
    Error(String),
    Thumbnail {
        id: u64,
        frame: Arc<Frame>,
    },
    /// [`Command::Thumbnail`]'s failure: admission or a render step failed.
    ThumbnailFailed {
        id: u64,
        error: String,
    },
    /// [`Command::Pick`]'s answer: the hit object's index in the picked scene's objects and the
    /// hit's depth along the view axis, None on a miss.
    Picked {
        id: u64,
        hit: Result<Option<(usize, f32)>, String>,
    },
    ExportProgress {
        id: u64,
        samples: u32,
        total: u32,
    },
    ExportFrame {
        id: u64,
        frame: Arc<Frame>,
    },
    ExportFailed {
        id: u64,
        error: String,
    },
    PreviewFrame {
        generation: u64,
        number: u32,
        frame: Arc<Frame>,
    },
    PreviewProgress {
        generation: u64,
        completed: u32,
        total: u32,
        bytes: usize,
        resident: Arc<[u32]>,
    },
    PreviewReady {
        generation: u64,
    },
    PreviewFailed {
        generation: u64,
        error: String,
    },
}

#[derive(Default)]
struct Mailbox {
    /// Wakes the UI when a frame or an event is posted for it (`push_event`, `publish_frame`).
    ui: Wake,
    viewport: Option<ViewportRequest>,
    requested_generation: u64,
    requested_interactive: bool,
    newest_completed_generation: u64,
    commands: VecDeque<Command>,
    events: VecDeque<RenderEvent>,
    latest: Option<Arc<Frame>>,
    stopping: bool,
    cancel_export: bool,
    active_export_id: Option<u64>,
    active_export_reply: Option<SyncSender<RenderEvent>>,
    pending_export: Option<RenderEvent>,
    pending_failure: Option<RenderEvent>,
    /// The newest pick answer when the event queue is full (a click is never lost).
    pending_pick: Option<RenderEvent>,
    /// The newest thumbnail failure when the event queue is full (its card never waits forever).
    pending_thumbnail: Option<RenderEvent>,
    last_error: Option<String>,
    preview_generation: Option<u64>,
    preview_control: Option<Command>,
    preview_seek: Option<(u64, u32)>,
    preview_store: Option<Command>,
    pending_preview_frame: Option<RenderEvent>,
    pending_preview_status: Option<RenderEvent>,
    recycled_preview: [Option<Arc<Frame>>; 3],
}
/// Wakes the UI from a worker thread when the worker posts a result the UI takes by polling
/// (render frames and events, file results, colour config loads, export progress), so an idle
/// window sleeps instead of polling (`window.rs`). One instance is made by the app and cloned
/// into every worker; until the app attaches its egui context (headless tests) it wakes nothing.
#[derive(Clone, Default)]
pub struct Wake(Arc<std::sync::OnceLock<egui::Context>>);
impl Wake {
    /// Attach the UI's context; the first one stays (the app's single window).
    pub fn attach(&self, ctx: &egui::Context) {
        self.0.get_or_init(|| ctx.clone());
    }
    /// Ask the UI for a frame (from any thread). Never call it while holding a lock:
    /// `request_repaint` takes egui's context lock, and a UI thread waiting for that lock of
    /// ours while egui's is held would deadlock. Drop the guard first, then wake.
    pub fn wake(&self) {
        if let Some(ctx) = self.0.get() {
            ctx.request_repaint();
        }
    }
}

struct Shared {
    state: Mutex<Mailbox>,
    wake: Condvar,
}
/// The mailbox, also after a panic poisoned its lock: the worker's panic handler and every
/// producer must still reach it to report the failure and stop.
fn lock(shared: &Shared) -> MutexGuard<'_, Mailbox> {
    shared.state.lock().unwrap_or_else(PoisonError::into_inner)
}
pub struct RenderService {
    shared: Arc<Shared>,
    #[cfg(test)]
    worker_thread: Option<thread::JoinHandle<()>>,
}
impl RenderService {
    /// Start the render worker; `ui` is woken whenever a frame or an event waits for the UI.
    pub fn spawn(ui: Wake) -> Self {
        let shared = Arc::new(Shared {
            state: Mutex::new(Mailbox {
                ui,
                ..Mailbox::default()
            }),
            wake: Condvar::new(),
        });
        let worker = shared.clone();
        // The GPU objects are constructed here, on the worker, never moved between threads.
        let spawned = thread::Builder::new()
            .name("render-worker".into())
            .spawn(move || {
                let result =
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_worker(&worker)));
                if let Err(error) = result {
                    let message = error
                        .downcast_ref::<String>()
                        .cloned()
                        .or_else(|| error.downcast_ref::<&str>().map(|s| (*s).to_owned()))
                        .unwrap_or_else(|| "Render worker panicked".into());
                    let (export_id, reply) = {
                        let mut state = lock(&worker);
                        (
                            state.active_export_id.take(),
                            state.active_export_reply.take(),
                        )
                    };
                    if let Some(id) = export_id {
                        dispatch_export(
                            &worker,
                            reply.as_ref(),
                            RenderEvent::ExportFailed {
                                id,
                                error: message.clone(),
                            },
                        );
                    }
                    push_event(&worker, RenderEvent::Error(message));
                    lock(&worker).stopping = true;
                }
            });
        let worker_thread = match spawned {
            Ok(handle) => Some(handle),
            Err(error) => {
                // No worker: the service reports itself stopped and says why.
                push_event(
                    &shared,
                    RenderEvent::Error(format!("Render worker did not start: {error}")),
                );
                lock(&shared).stopping = true;
                None
            }
        };
        // Production never joins the worker (`Drop`); it runs detached until it sees `stopping`.
        #[cfg(not(test))]
        drop(worker_thread);
        Self {
            shared,
            #[cfg(test)]
            worker_thread,
        }
    }
    pub fn request_viewport(&self, request: ViewportRequest) {
        let mut state = lock(&self.shared);
        state.requested_generation = request.generation;
        state.requested_interactive =
            request.active && !request.paused && request.interactive && request.preview;
        state.viewport = Some(request);
        self.shared.wake.notify_one();
    }
    pub fn take_latest_frame(&self) -> Option<Arc<Frame>> {
        let mut state = lock(&self.shared);
        let frame = state.latest.take()?;
        frame_eligible(&frame, &state).then_some(frame)
    }
    pub fn drain_events(&self) -> Vec<RenderEvent> {
        let mut state = lock(&self.shared);
        let mut events: Vec<_> = state.events.drain(..).collect();
        events.extend(state.pending_export.take());
        events.extend(state.pending_failure.take());
        events.extend(state.pending_pick.take());
        events.extend(state.pending_thumbnail.take());
        events.extend(state.last_error.take().map(RenderEvent::Error));
        events.extend(state.pending_preview_frame.take());
        events.extend(state.pending_preview_status.take());
        self.shared.wake.notify_one();
        events
    }
    /// Replaceable preview intents never wait for a renderer/cache mutex on the UI; a refused
    /// command comes back for the next tick.
    pub fn try_preview_command(&self, command: Command) -> Result<(), Command> {
        queue_preview_command(&self.shared, command).map_err(|refused| refused.0)
    }

    pub(crate) fn preview_stopped(&self) -> bool {
        match self.shared.state.try_lock() {
            Ok(state) => state.stopping,
            Err(TryLockError::Poisoned(error)) => error.into_inner().stopping,
            Err(TryLockError::WouldBlock) => false,
        }
    }

    pub fn port(&self) -> RenderPort {
        RenderPort {
            shared: self.shared.clone(),
        }
    }
    pub fn try_command(&self, command: Command) -> Result<(), CommandError> {
        self.port().try_command(command)
    }
}

/// Keep `frame` in the first free slot unless it is already there: the one recycle of the fixed
/// presentation slots (UI hand-backs, replaced preview results, the presentation pools, the
/// preview controller). Every slot taken: the frame comes back.
pub(crate) fn put_slot(
    slots: &mut [Option<Arc<Frame>>],
    frame: Arc<Frame>,
) -> Result<(), Arc<Frame>> {
    if slots.iter().flatten().any(|old| Arc::ptr_eq(old, &frame)) {
        return Ok(());
    }
    match slots.iter_mut().find(|slot| slot.is_none()) {
        Some(slot) => {
            *slot = Some(frame);
            Ok(())
        }
        None => Err(frame),
    }
}

/// Queue a replaceable preview intent in its mailbox slot. A refused command comes back with
/// the reason; boxed because `Command` is large and refusals are rare.
fn queue_preview_command(
    shared: &Shared,
    command: Command,
) -> Result<(), Box<(Command, CommandError)>> {
    let mut state = match shared.state.try_lock() {
        Ok(state) => state,
        Err(TryLockError::Poisoned(error)) => error.into_inner(),
        Err(TryLockError::WouldBlock) => return Err(Box::new((command, CommandError::Full))),
    };
    if state.stopping {
        return Err(Box::new((command, CommandError::Stopped)));
    }
    match command {
        command @ Command::BeginPreview { .. } => {
            if let Command::BeginPreview { request, .. } = &command {
                state.preview_generation = Some(request.generation);
            }
            state.preview_control = Some(command);
            state.preview_seek = None;
            state.preview_store = None;
        }
        command @ Command::CacheViewport { .. } => {
            if let Command::CacheViewport { request, .. } = &command {
                if state
                    .preview_generation
                    .is_some_and(|id| id != request.generation)
                {
                    return Ok(());
                }
                state.preview_generation = Some(request.generation);
            }
            state.preview_store = Some(command);
        }
        Command::SeekPreview { generation, number } => {
            if state.preview_generation == Some(generation) {
                state.preview_seek = Some((generation, number));
            }
        }
        command @ Command::CancelPreview { generation } => {
            if state.preview_generation == Some(generation) {
                state.preview_generation = None;
                state.preview_control = Some(command);
                state.preview_seek = None;
                state.preview_store = None;
            }
        }
        Command::RecyclePreviewFrame { frame } => {
            if let Err(frame) = put_slot(&mut state.recycled_preview, frame) {
                return Err(Box::new((
                    Command::RecyclePreviewFrame { frame },
                    CommandError::Full,
                )));
            }
        }
        other => {
            return Err(Box::new((
                other,
                CommandError::Invalid("Not a preview command".into()),
            )));
        }
    }
    shared.wake.notify_one();
    Ok(())
}

/// A command producer for background coordinators. Dropping it never stops rendering.
#[derive(Clone)]
pub struct RenderPort {
    shared: Arc<Shared>,
}
impl RenderPort {
    pub fn is_available(&self) -> bool {
        !lock(&self.shared).stopping
    }
    /// Validate and queue `command`. A full queue ([`CommandError::Full`]) is worth a retry; a
    /// stopped worker or an invalid request is not.
    pub fn try_command(&self, command: Command) -> Result<(), CommandError> {
        match &command {
            Command::Thumbnail {
                width, height, spp, ..
            }
            | Command::RenderExport {
                width, height, spp, ..
            } => validate_size(*width, *height, *spp)
                .map_err(|error| CommandError::Invalid(error.to_string()))?,
            Command::BeginPreview { request, .. } | Command::CacheViewport { request, .. } => {
                request.validate().map_err(CommandError::Invalid)?
            }
            _ => {}
        }
        if matches!(
            &command,
            Command::BeginPreview { .. }
                | Command::CacheViewport { .. }
                | Command::SeekPreview { .. }
                | Command::CancelPreview { .. }
                | Command::RecyclePreviewFrame { .. }
        ) {
            return queue_preview_command(&self.shared, command).map_err(|refused| refused.1);
        }
        let mut state = lock(&self.shared);
        if state.stopping {
            return Err(CommandError::Stopped);
        }
        if matches!(command, Command::CancelExport) {
            state.cancel_export = true;
            state
                .commands
                .retain(|c| !matches!(c, Command::RenderExport { .. }));
        } else {
            if state.commands.len() >= QUEUE_LIMIT {
                return Err(CommandError::Full);
            }
            state.commands.push_back(command);
        }
        self.shared.wake.notify_one();
        Ok(())
    }
}
impl Drop for RenderService {
    fn drop(&mut self) {
        lock(&self.shared).stopping = true;
        self.shared.wake.notify_one();
        // Production never joins the UI thread. Tests must release the adapter only after the
        // worker has dropped its CUDA/OIDN resources, otherwise the next test races teardown.
        #[cfg(test)]
        if let Some(worker) = self.worker_thread.take()
            && worker.join().is_err()
        {
            log::error!("Render test worker panicked during shutdown");
        }
    }
}
fn recycle_mailbox_frame(state: &mut Mailbox, frame: Arc<Frame>) {
    if let Err(frame) = put_slot(&mut state.recycled_preview, frame) {
        // Every slot is taken: this presentation is freed instead of reused.
        drop(frame);
    }
}
/// Post `event` for the UI (`RenderService::drain_events`) and wake the UI to take it.
fn push_event(shared: &Shared, event: RenderEvent) {
    let ui = {
        let mut state = lock(shared);
        queue_event(&mut state, event);
        state.ui.clone()
    };
    ui.wake();
}
/// Put `event` into the mailbox by its kind: replaceable progress and preview results keep only
/// the newest, final results survive a full queue in their overflow slots.
fn queue_event(state: &mut Mailbox, event: RenderEvent) {
    match &event {
        RenderEvent::PreviewFrame { generation, .. } => {
            if state.preview_generation == Some(*generation) {
                if let Some(RenderEvent::PreviewFrame { frame, .. }) =
                    state.pending_preview_frame.replace(event)
                {
                    recycle_mailbox_frame(state, frame);
                }
            } else if let RenderEvent::PreviewFrame { frame, .. } = event {
                recycle_mailbox_frame(state, frame);
            }
            return;
        }
        RenderEvent::PreviewReady { generation }
        | RenderEvent::PreviewFailed { generation, .. } => {
            if state.preview_generation == Some(*generation) {
                state.pending_preview_status = Some(event);
            }
            return;
        }
        RenderEvent::PreviewProgress { generation, .. } => {
            if state.preview_generation != Some(*generation) {
                return;
            }
            if let Some(index) = state.events.iter().position(|e| matches!(e, RenderEvent::PreviewProgress { generation: other, .. } if other == generation)) {
                state.events[index] = event; return;
            }
            if state.events.len() >= QUEUE_LIMIT {
                return;
            }
        }
        _ => {}
    }
    if let RenderEvent::ExportProgress { id, .. } = &event {
        if let Some(index) = state
            .events
            .iter()
            .position(|e| matches!(e, RenderEvent::ExportProgress { id: other, .. } if other == id))
        {
            state.events[index] = event;
            return;
        }
        // Progress is replaceable; completion/error events must never be discarded.
        if state.events.len() >= QUEUE_LIMIT {
            return;
        }
    }
    if state.stopping {
        return;
    }
    if state.events.len() >= QUEUE_LIMIT {
        match event {
            event @ RenderEvent::ExportFrame { .. } => state.pending_export = Some(event),
            event @ RenderEvent::ExportFailed { .. } => state.pending_failure = Some(event),
            event @ RenderEvent::Picked { .. } => state.pending_pick = Some(event),
            event @ RenderEvent::ThumbnailFailed { .. } => state.pending_thumbnail = Some(event),
            RenderEvent::Error(error) => state.last_error = Some(error),
            _ => {}
        }
    } else {
        state.events.push_back(event);
    }
}
/// Dedicated export replies are not gated by the GUI event queue. Progress may be
/// discarded, but final results retry a bounded channel while checking cancellation.
fn dispatch_export(
    shared: &Shared,
    reply: Option<&SyncSender<RenderEvent>>,
    mut event: RenderEvent,
) {
    let Some(reply) = reply else {
        push_event(shared, event);
        return;
    };
    if matches!(event, RenderEvent::ExportProgress { .. }) {
        // Progress is replaceable: a full channel drops this one and the next replaces it.
        if let Err(TrySendError::Disconnected(_)) = reply.try_send(event) {
            log::debug!("export progress dropped: its coordinator has gone");
        }
        return;
    }
    loop {
        match reply.try_send(event) {
            Ok(()) | Err(TrySendError::Disconnected(_)) => return,
            Err(TrySendError::Full(returned)) => event = returned,
        }
        let state = lock(shared);
        if state.stopping || state.cancel_export {
            return;
        }
        let (_state, _timeout) = shared
            .wake
            .wait_timeout(state, Duration::from_millis(5))
            .unwrap_or_else(PoisonError::into_inner);
    }
}

/// A completed preview is useful during continuous dragging even if the next request
/// has already arrived. Full renders retain strict generation matching.
fn frame_eligible(frame: &Frame, state: &Mailbox) -> bool {
    frame.generation == state.requested_generation
        || (frame.preview
            && state.requested_interactive
            && frame.generation < state.requested_generation)
}
fn publish_frame(shared: &Shared, frame: Arc<Frame>) {
    // The UI is woken after the mailbox lock is released (`Wake::wake`).
    let stored = {
        let mut state = lock(shared);
        let eligible =
            frame_eligible(&frame, &state) && frame.generation >= state.newest_completed_generation;
        if eligible {
            state.newest_completed_generation = frame.generation;
            state.latest = Some(frame);
        }
        eligible.then(|| state.ui.clone())
    };
    if let Some(ui) = stored {
        ui.wake();
    }
}

struct Viewport {
    request: ViewportRequest,
    target: Option<Target>,
    dirty: bool,
    last_preview: bool,
    last_publish: Instant,
    /// The frames this viewport publishes, reused once the UI dropped them.
    pool: PresentationPool,
}
impl Viewport {
    fn new(request: ViewportRequest) -> Self {
        Self {
            request,
            target: None,
            dirty: true,
            last_preview: false,
            last_publish: Instant::now() - Duration::from_millis(20),
            pool: PresentationPool::default(),
        }
    }
    fn update(&mut self, request: ViewportRequest) {
        // The seed is deliberately absent from Gpu's accumulation key (every other traced input
        // resets the film there), so a new seed starts a new film here.
        if request.seed != self.request.seed {
            self.target = None;
        }
        self.dirty |= self.request != request;
        self.request = request;
    }
}
struct RenderJob {
    id: u64,
    scene: Scene,
    target: Target,
    spp: u32,
    /// The job's seed and batch budget: its request's, never the worker's.
    seed: u32,
    budget_ms: f32,
    /// Read the scene-linear radiance at completion (EXR exports).
    radiance: bool,
    reply: Option<SyncSender<RenderEvent>>,
}

struct PreviewSession {
    cache: crate::preview::PreviewCache,
    wanted: Option<u32>,
    next_fill: u32,
    fill: bool,
    job: Option<(u32, u64, RenderJob)>,
    target: Option<Target>,
}

/// Three presentations cover the UI frame, upload staging, and the worker. Reuse
/// waits for both frame and byte ownership; never clone an in-flight pixel buffer.
#[derive(Default)]
struct PresentationPool {
    slots: [Option<Arc<Frame>>; 3],
    created: usize,
}
impl PresentationPool {
    fn acquire(&mut self) -> Option<Arc<Frame>> {
        if let Some(index) = self.slots.iter().position(|slot| {
            slot.as_ref().is_some_and(|frame| {
                Arc::strong_count(frame) == 1
                    && Arc::strong_count(&frame.sdr_bytes) == 1
                    && Arc::strong_count(&frame.hdr_bytes) == 1
            })
        }) {
            return self.slots[index].take();
        }
        if self.created >= self.slots.len() {
            return None;
        }
        self.created += 1;
        Some(Arc::new(empty_presentation()))
    }
    fn recycle(&mut self, frame: Arc<Frame>) {
        if let Err(frame) = put_slot(&mut self.slots, frame) {
            // Every slot is taken: this presentation is freed instead of reused.
            drop(frame);
        }
    }
}
fn empty_presentation() -> Frame {
    Frame {
        generation: 0,
        preview: true,
        width: 0,
        height: 0,
        pixels: Vec::new(),
        light: Vec::new(),
        radiance: Vec::new(),
        light_kind: crate::color::DisplayLight::Relative,
        colour_error: None,
        denoised_samples: 0,
        denoise_ms: 0.0,
        denoise_error: None,
        samples: 0,
        active_tiles: (0, 0),
        converged: false,
        last_ms: 0.0,
        last_spp: 0,
        unresolved: 0.0,
        sdr_bytes: Arc::new(Vec::new()),
        hdr_bytes: Arc::new(Vec::new()),
    }
}
fn preview_current(shared: &Shared, generation: u64) -> bool {
    let state = lock(shared);
    !state.stopping && state.preview_generation == Some(generation)
}
fn fail_preview(session: &mut PreviewSession, shared: &Shared, error: String) {
    session.fill = false;
    session.wanted = None;
    session.job = None;
    push_event(
        shared,
        RenderEvent::PreviewFailed {
            generation: session.cache.request.generation,
            error,
        },
    );
}
/// The cache's progress for the UI: the one builder of [`RenderEvent::PreviewProgress`].
fn progress(cache: &crate::preview::PreviewCache) -> RenderEvent {
    RenderEvent::PreviewProgress {
        generation: cache.request.generation,
        completed: cache.count(),
        total: cache.frames(),
        bytes: cache.bytes(),
        resident: cache.resident(),
    }
}
fn present_preview(
    session: &mut PreviewSession,
    pool: &mut PresentationPool,
    shared: &Shared,
) -> bool {
    let Some(number) = session
        .wanted
        .filter(|number| session.cache.contains(*number))
    else {
        return false;
    };
    let Some(mut frame) = pool.acquire() else {
        return false;
    };
    // The pool hands out frames nothing else holds.
    let filled = match Arc::get_mut(&mut frame) {
        Some(owned) => session.cache.get_into(number, owned).is_some(),
        None => false,
    };
    if !filled {
        pool.recycle(frame);
        return false;
    }
    session.wanted = None;
    push_event(
        shared,
        RenderEvent::PreviewFrame {
            generation: session.cache.request.generation,
            number,
            frame,
        },
    );
    true
}
fn step_preview(
    gpu: &mut Gpu,
    session: &mut PreviewSession,
    pool: &mut PresentationPool,
    shared: &Shared,
    interactive: bool,
) -> bool {
    let generation = session.cache.request.generation;
    if !preview_current(shared, generation) {
        return false;
    }
    let mut worked = present_preview(session, pool, shared);
    if session.job.is_none() && !interactive {
        let count = session.cache.frames();
        let number = if session.fill {
            while session.next_fill < count
                && session
                    .cache
                    .contains(session.cache.request.first + session.next_fill)
            {
                session.next_fill += 1;
            }
            if session.next_fill == count {
                session.fill = false;
                push_event(shared, RenderEvent::PreviewReady { generation });
                return true;
            }
            Some(session.cache.request.first + session.next_fill)
        } else {
            session
                .wanted
                .filter(|number| !session.cache.contains(*number))
        };
        if let Some(number) = number {
            let request = &session.cache.request;
            let scene = if let Some(document) = &request.scene.document {
                match request.render_profile {
                    Some(profile) => document.snapshot_with_render_profile(
                        profile,
                        number as f64,
                        crate::world::View::Viewport,
                    ),
                    None => document.snapshot(number as f64),
                }
            } else {
                Ok(request.scene.as_ref().clone())
            };
            let scene = match scene.and_then(|scene| {
                gpu.prepare_scene(&scene, request.width, request.height)?;
                Ok(scene)
            }) {
                Ok(scene) => scene,
                Err(error) => {
                    fail_preview(session, shared, error);
                    return true;
                }
            };
            session.job = Some((
                number,
                session.cache.manager.current_epoch(),
                RenderJob {
                    id: generation,
                    scene,
                    target: session
                        .target
                        .take()
                        .unwrap_or_else(|| Target::new(request.width, request.height)),
                    spp: request.spp,
                    seed: request.seed,
                    budget_ms: request.budget_ms,
                    radiance: false,
                    reply: None,
                },
            ));
        }
    }
    let Some((_, _, job)) = &mut session.job else {
        return worked;
    };
    if !background_batch_allowed(
        job.target.last_ms,
        job.target.last_spp,
        job.budget_ms,
        interactive,
    ) {
        return worked;
    }
    let advanced = gpu.advance(
        &mut job.target,
        &job.scene,
        job.spp,
        job.seed,
        Batch::Budget(job.budget_ms),
    );
    worked = true;
    let complete = match advanced {
        Ok(complete) => complete,
        Err(error) => {
            fail_preview(session, shared, error.to_string());
            return worked;
        }
    };
    if !preview_current(shared, generation) || !complete {
        return worked;
    }
    let Some((number, epoch, job)) = session.job.take() else {
        return worked;
    };
    let request = &session.cache.request;
    let frame = Frame::snapshot(
        &job.target,
        generation,
        true,
        request.output_hdr,
        request.white_nits,
    );
    session.target = Some(job.target);
    if let Err(error) = session.cache.store(number, &frame, epoch) {
        fail_preview(session, shared, error);
        return worked;
    }
    if session.fill && number == session.cache.request.first + session.next_fill {
        session.next_fill += 1;
    }
    push_event(shared, progress(&session.cache));
    worked |= present_preview(session, pool, shared);
    worked
}

/// Move cache ownership between quality profiles without cloning pixels. Scene
/// edits fail compatibility and cannot reuse an old profile's rendered content.
fn switch_preview_cache(
    active: &mut Option<PreviewSession>,
    parked: &mut Option<crate::preview::PreviewCache>,
    request: crate::preview::PreviewRequest,
    cache_all: bool,
) -> Result<crate::preview::PreviewCache, String> {
    let reusable = parked
        .take()
        .filter(|cache| cache.request.compatible_content(&request));
    *parked = active.take().map(|session| session.cache);
    if let Some(mut cache) = reusable {
        cache.restart(request, cache_all)?;
        Ok(cache)
    } else {
        crate::preview::PreviewCache::new(request, cache_all)
    }
}

/// Admission of a viewport, export or thumbnail request on the worker: the one size rule, then
/// the backend, packing and environment of the exact dimensions ([`Gpu::prepare_scene`]).
fn admit(
    gpu: &mut Gpu,
    scene: &Scene,
    width: usize,
    height: usize,
    spp: u32,
) -> Result<(), RenderError> {
    validate_size(width, height, spp)?;
    gpu.prepare_scene(scene, width, height)
}

/// What one worker iteration takes from the mailbox, under one lock.
struct Mail {
    request: Option<ViewportRequest>,
    command: Option<Command>,
    cancel: bool,
    preview_control: Option<Command>,
    preview_seek: Option<(u64, u32)>,
    preview_store: Option<Command>,
    recycled: [Option<Arc<Frame>>; 3],
}

/// Everything the render worker owns: the GPU and its jobs. Lives on the worker thread only.
/// Every dequeued intent is applied ([`Self::receive`]): a failure becomes that intent's typed
/// event, never a skipped iteration that would drop the rest of the mail.
struct Worker {
    gpu: Gpu,
    viewport: Option<Viewport>,
    export: Option<RenderJob>,
    thumbnail: Option<RenderJob>,
    preview: Option<PreviewSession>,
    /// Keep the other quality profile resident when switching draft/final transport.
    /// Both caches retain Playa's byte budget and eviction policy.
    parked_preview: Option<crate::preview::PreviewCache>,
    presentations: PresentationPool,
}

impl Worker {
    /// The next mail, or None once the service is stopping.
    fn take(&self, shared: &Shared) -> Option<Mail> {
        let mut state = lock(shared);
        if state.stopping {
            return None;
        }
        // Allocate background render targets only once motion settles; allocation itself
        // can be expensive for final-resolution exports.
        let moving = self.viewport.as_ref().is_some_and(viewport_interactive)
            || state
                .viewport
                .as_ref()
                .is_some_and(|r| r.active && !r.paused && r.interactive);
        let index = state.commands.iter().position(|command| match command {
            Command::Thumbnail { .. } => {
                !moving && self.thumbnail.is_none() && state.events.len() < QUEUE_LIMIT
            }
            Command::RenderExport { reply, .. } => {
                !moving
                    && self.export.is_none()
                    && (reply.is_some() || state.pending_export.is_none())
            }
            _ => true,
        });
        let command = index.and_then(|index| state.commands.remove(index));
        Some(Mail {
            request: state.viewport.take(),
            command,
            cancel: std::mem::take(&mut state.cancel_export),
            preview_control: state.preview_control.take(),
            preview_seek: state.preview_seek.take(),
            preview_store: state.preview_store.take(),
            recycled: std::mem::take(&mut state.recycled_preview),
        })
    }

    fn receive(&mut self, shared: &Shared, mail: Mail) {
        for frame in mail.recycled.into_iter().flatten() {
            self.presentations.recycle(frame);
        }
        if mail.cancel {
            self.end_export(shared);
        }
        if let Some(request) = mail.request {
            self.admit_viewport(shared, request);
        }
        if let Some(control) = mail.preview_control {
            self.preview_control(shared, control);
        }
        if let Some(store) = mail.preview_store {
            self.preview_store(shared, store);
        }
        if let Some((generation, number)) = mail.preview_seek
            && let Some(session) = self
                .preview
                .as_mut()
                .filter(|p| p.cache.request.generation == generation)
            && session.cache.index(number).is_some()
        {
            session.cache.manager.increment_generation();
            session.wanted = Some(number);
            // Presentation is served below after a reusable pool slot becomes available.
        }
        if let Some(command) = mail.command {
            self.command(shared, command);
        }
    }

    /// A refused request leaves the previous viewport idle instead of rendering a stale scene;
    /// the next accepted request resumes it.
    fn admit_viewport(&mut self, shared: &Shared, request: ViewportRequest) {
        if let Err(error) = admit(
            &mut self.gpu,
            &request.scene,
            request.width,
            request.height,
            request.target_spp,
        ) {
            if let Some(viewport) = &mut self.viewport {
                viewport.request.active = false;
            }
            push_event(shared, RenderEvent::Error(error.to_string()));
        } else if let Some(viewport) = &mut self.viewport {
            viewport.update(request);
        } else {
            self.viewport = Some(Viewport::new(request));
        }
    }

    fn preview_control(&mut self, shared: &Shared, control: Command) {
        match control {
            Command::BeginPreview { request, cache_all } => {
                let generation = request.generation;
                if let Some(session) = self
                    .preview
                    .as_mut()
                    .filter(|session| session.cache.request.compatible_content(&request))
                {
                    let first = request.first;
                    if let Err(error) = session.cache.restart(request, cache_all) {
                        session.cache.request.generation = generation;
                        fail_preview(session, shared, error);
                        return;
                    }
                    session.wanted = Some(first);
                    session.next_fill = 0;
                    session.fill = cache_all;
                    if let Some((_, _, job)) = &mut session.job {
                        job.id = generation;
                    }
                    if let Some(v) = &mut self.viewport {
                        v.request.active = false;
                    }
                    push_event(shared, progress(&session.cache));
                } else {
                    match switch_preview_cache(
                        &mut self.preview,
                        &mut self.parked_preview,
                        request,
                        cache_all,
                    ) {
                        Ok(cache) => {
                            if let Some(v) = &mut self.viewport {
                                v.request.active = false;
                            }
                            push_event(shared, progress(&cache));
                            self.preview = Some(PreviewSession {
                                wanted: Some(cache.request.first),
                                next_fill: 0,
                                fill: cache_all,
                                cache,
                                job: None,
                                target: None,
                            });
                        }
                        Err(error) => {
                            push_event(shared, RenderEvent::PreviewFailed { generation, error })
                        }
                    }
                }
            }
            Command::CancelPreview { generation } => {
                if let Some(session) = self
                    .preview
                    .as_mut()
                    .filter(|p| p.cache.request.generation == generation)
                {
                    session.fill = false;
                    session.wanted = None;
                    session.job = None;
                    session.cache.manager.increment_generation();
                }
            }
            _ => {}
        }
    }

    fn preview_store(&mut self, shared: &Shared, store: Command) {
        let Command::CacheViewport {
            request,
            number,
            frame,
        } = store
        else {
            return;
        };
        let generation = request.generation;
        if !preview_current(shared, generation) {
            return;
        }
        if self
            .preview
            .as_ref()
            .is_none_or(|s| !s.cache.request.compatible_content(&request))
        {
            match switch_preview_cache(&mut self.preview, &mut self.parked_preview, request, false)
            {
                Ok(cache) => {
                    self.preview = Some(PreviewSession {
                        cache,
                        wanted: None,
                        next_fill: 0,
                        fill: false,
                        job: None,
                        target: None,
                    })
                }
                Err(error) => {
                    push_event(shared, RenderEvent::PreviewFailed { generation, error });
                    return;
                }
            }
        }
        if let Some(session) = self
            .preview
            .as_mut()
            .filter(|s| s.cache.request.generation == generation)
        {
            let epoch = session.cache.manager.current_epoch();
            match session.cache.store(number, &frame, epoch) {
                Ok(()) => push_event(shared, progress(&session.cache)),
                Err(error) => push_event(shared, RenderEvent::PreviewFailed { generation, error }),
            }
        }
    }

    fn command(&mut self, shared: &Shared, command: Command) {
        match command {
            // Preview intents travel through their own mailbox slots (`queue_preview_command`).
            Command::BeginPreview { .. }
            | Command::CacheViewport { .. }
            | Command::SeekPreview { .. }
            | Command::CancelPreview { .. }
            | Command::RecyclePreviewFrame { .. } => {}
            Command::CancelExport => self.end_export(shared),
            Command::Pick {
                id,
                scene,
                width,
                height,
                ndc,
            } => {
                let hit = validate_size(width, height, 1)
                    .and_then(|()| self.gpu.pick(&scene, width, height, ndc))
                    .map_err(|error| error.to_string());
                // Its parameters replaced the viewport's: the next viewport launch sets its own.
                push_event(shared, RenderEvent::Picked { id, hit });
            }
            Command::ReloadColour => {
                if let Some(v) = &mut self.viewport {
                    v.dirty = true;
                }
            }
            Command::RenderExport {
                id,
                scene,
                width,
                height,
                spp,
                seed,
                budget_ms,
                radiance,
                reply,
            } => {
                if let Err(error) = admit(&mut self.gpu, &scene, width, height, spp) {
                    dispatch_export(
                        shared,
                        reply.as_ref(),
                        RenderEvent::ExportFailed {
                            id,
                            error: error.to_string(),
                        },
                    );
                } else {
                    {
                        let mut state = lock(shared);
                        state.active_export_id = Some(id);
                        state.active_export_reply = reply.clone();
                    }
                    self.export = Some(RenderJob {
                        id,
                        scene: *scene,
                        target: Target::new(width, height),
                        spp,
                        seed,
                        budget_ms,
                        radiance,
                        reply,
                    });
                }
            }
            Command::Thumbnail {
                id,
                scene,
                width,
                height,
                spp,
                seed,
                budget_ms,
                max_bounces,
            } => {
                let admitted = thumbnail_snapshot(*scene, max_bounces).and_then(|scene| {
                    admit(&mut self.gpu, &scene, width, height, spp)?;
                    Ok(scene)
                });
                match admitted {
                    Ok(scene) => {
                        self.thumbnail = Some(RenderJob {
                            id,
                            scene,
                            target: Target::new(width, height),
                            spp,
                            seed,
                            budget_ms,
                            radiance: false,
                            reply: None,
                        })
                    }
                    Err(error) => push_event(shared, RenderEvent::ThumbnailFailed { id, error }),
                }
            }
        }
    }

    /// Drop the export job (cancelled, finished or failed) and its panic-path registration.
    fn end_export(&mut self, shared: &Shared) {
        self.export = None;
        let mut state = lock(shared);
        state.active_export_id = None;
        state.active_export_reply = None;
    }

    /// One step of every job that has work. Returns whether any GPU work ran.
    fn work(&mut self, shared: &Shared) -> bool {
        let interactive = self.viewport.as_ref().is_some_and(viewport_interactive);
        let mut worked = false;
        if let Some(viewport) = self.viewport.as_mut() {
            match step_viewport(&mut self.gpu, viewport, shared) {
                Ok(stepped) => worked |= stepped,
                Err(error) => {
                    // Rendering the same request again would fail the same way.
                    viewport.request.active = false;
                    push_event(shared, RenderEvent::Error(error.to_string()));
                }
            }
        }
        worked |= self.step_export(shared, interactive);
        if self.export.is_none()
            && let Some(session) = &mut self.preview
        {
            worked |= step_preview(
                &mut self.gpu,
                session,
                &mut self.presentations,
                shared,
                interactive,
            );
        }
        worked |= self.step_thumbnail(shared, interactive);
        worked
    }

    fn step_export(&mut self, shared: &Shared, interactive: bool) -> bool {
        let Some(job) = self.export.as_mut() else {
            return false;
        };
        if !background_batch_allowed(
            job.target.last_ms,
            job.target.last_spp,
            job.budget_ms,
            interactive,
        ) {
            return false;
        }
        let outcome = self
            .gpu
            .advance(
                &mut job.target,
                &job.scene,
                job.spp,
                job.seed,
                Batch::Budget(job.budget_ms),
            )
            .and_then(|complete| {
                if !complete {
                    return Ok(None);
                }
                let mut frame = Frame::snapshot(&job.target, job.id, false, false, 100.0);
                if job.radiance {
                    frame.radiance = self.gpu.scene_linear(&job.target)?;
                }
                Ok(Some(frame))
            });
        let cancelled = lock(shared).cancel_export;
        match outcome {
            Err(error) => {
                let (id, reply) = (job.id, job.reply.take());
                self.end_export(shared);
                dispatch_export(
                    shared,
                    reply.as_ref(),
                    RenderEvent::ExportFailed {
                        id,
                        error: error.to_string(),
                    },
                );
            }
            Ok(_) if cancelled => {}
            Ok(Some(frame)) => {
                dispatch_export(
                    shared,
                    job.reply.as_ref(),
                    RenderEvent::ExportFrame {
                        id: job.id,
                        frame: Arc::new(frame),
                    },
                );
                self.end_export(shared);
            }
            Ok(None) => dispatch_export(
                shared,
                job.reply.as_ref(),
                RenderEvent::ExportProgress {
                    id: job.id,
                    samples: job.target.samples,
                    total: job.spp,
                },
            ),
        }
        true
    }

    fn step_thumbnail(&mut self, shared: &Shared, interactive: bool) -> bool {
        let Some(job) = self.thumbnail.as_mut() else {
            return false;
        };
        let mut worked = false;
        if !interactive && !job.target.complete(job.spp) {
            worked = true;
            if let Err(error) = self.gpu.advance(
                &mut job.target,
                &job.scene,
                job.spp,
                job.seed,
                Batch::Budget(job.budget_ms),
            ) {
                let id = job.id;
                self.thumbnail = None;
                push_event(
                    shared,
                    RenderEvent::ThumbnailFailed {
                        id,
                        error: error.to_string(),
                    },
                );
                return worked;
            }
        }
        if job.target.complete(job.spp) && lock(shared).events.len() < QUEUE_LIMIT {
            let frame = Arc::new(Frame::snapshot(&job.target, job.id, false, false, 100.0));
            push_event(shared, RenderEvent::Thumbnail { id: job.id, frame });
            self.thumbnail = None;
        }
        worked
    }
}

fn run_worker(shared: &Shared) {
    if lock(shared).stopping {
        return;
    }
    let gpu = match Gpu::new() {
        Ok(gpu) => gpu,
        Err(error) => {
            push_event(shared, RenderEvent::Error(error.to_string()));
            lock(shared).stopping = true;
            return;
        }
    };
    push_event(
        shared,
        RenderEvent::Ready {
            name: gpu.name.clone(),
        },
    );
    let mut worker = Worker {
        gpu,
        viewport: None,
        export: None,
        thumbnail: None,
        preview: None,
        parked_preview: None,
        presentations: PresentationPool::default(),
    };
    while let Some(mail) = worker.take(shared) {
        worker.receive(shared, mail);
        if worker.work(shared) {
            continue;
        }
        let state = lock(shared);
        if state.stopping {
            return;
        }
        if state.viewport.is_none()
            && !state.cancel_export
            && state.preview_control.is_none()
            && state.preview_seek.is_none()
            && state.preview_store.is_none()
            && state.recycled_preview.iter().all(Option::is_none)
        {
            let (_state, _timeout) = shared
                .wake
                .wait_timeout(state, Duration::from_millis(20))
                .unwrap_or_else(PoisonError::into_inner);
        }
    }
}
/// The scene a thumbnail renders: its world evaluated at its first frame, then the request's
/// bounce cap on the evaluated root (a bookmark's frozen document carries its own render
/// settings, so the cap cannot be applied before evaluation).
fn thumbnail_snapshot(scene: Scene, max_bounces: u32) -> Result<Scene, String> {
    let document = crate::world::WorldDocument::from_scene(&scene);
    let mut scene = document.snapshot(f64::from(document.first))?;
    scene.render.max_bounces = scene.render.max_bounces.min(max_bounces);
    Ok(scene)
}

fn viewport_interactive(viewport: &Viewport) -> bool {
    viewport.request.active && !viewport.request.paused && viewport.request.interactive
}
/// A background batch may run while the viewport moves only when one previously measured
/// sample fits the job's own `budget_ms`: a GPU launch cannot be preempted, so an unknown or
/// larger cost waits until motion ends.
fn background_batch_allowed(
    last_ms: f32,
    last_spp: u32,
    budget_ms: f32,
    interactive: bool,
) -> bool {
    !interactive || (last_spp > 0 && last_ms > 0.0 && last_ms / last_spp as f32 <= budget_ms)
}
/// One viewport step: prepare (once), size the batch from the request's budget, trace, and
/// publish into a pooled frame when the display is due. Returns whether samples were traced.
fn step_viewport(
    gpu: &mut Gpu,
    active: &mut Viewport,
    shared: &Shared,
) -> Result<bool, RenderError> {
    let request = &active.request;
    if !request.active {
        return Ok(false);
    }
    let preview = request.preview;
    let (width, height) = (request.width, request.height);
    // Profiles share the film whenever their effective trace inputs and extent agree.
    if active
        .target
        .as_ref()
        .is_some_and(|t| t.width != width || t.height != height)
    {
        active.target = None;
    }
    let fresh = active.target.is_none();
    let target = active
        .target
        .get_or_insert_with(|| Target::new(width, height));
    // Detect accumulation resets without the extra tonemap/OCIO/readback that a zero-spp
    // step would perform. The selected traced or display-only batch runs exactly once.
    let changed = active.dirty || fresh || active.last_preview != preview;
    let prepared = if changed {
        Some(gpu.prepare(target, &request.scene)?)
    } else {
        None
    };
    target.raw_view = request.raw;
    let goal = request.target_spp;
    let spp = if request.paused {
        0
    } else {
        Batch::Budget(request.batch_budget_ms).size(target, goal.saturating_sub(target.samples))
    };
    if spp == 0 && !changed {
        return Ok(false);
    }
    let launch = match prepared {
        Some(launch) => launch,
        None => gpu.prepare(target, &request.scene)?,
    };
    let final_pass = target.samples.saturating_add(spp) >= goal;
    // A restarted film is always shown, so the viewport never keeps the old accumulation.
    let mut publish = changed || launch.reset;
    gpu.step_with_display(
        target,
        &request.scene,
        &launch,
        spp,
        request.seed,
        final_pass,
        |target| {
            publish |= target.complete(goal)
                || active.last_publish.elapsed()
                    >= Duration::from_secs_f32(1.0 / request.target_fps);
            publish
        },
    )?;
    if publish {
        let mut frame = active
            .pool
            .acquire()
            .unwrap_or_else(|| Arc::new(empty_presentation()));
        match Arc::get_mut(&mut frame) {
            Some(owned) => owned.snapshot_into(
                target,
                request.generation,
                preview,
                request.output_hdr,
                request.white_nits,
            ),
            None => {
                frame = Arc::new(Frame::snapshot(
                    target,
                    request.generation,
                    preview,
                    request.output_hdr,
                    request.white_nits,
                ))
            }
        }
        active.pool.recycle(frame.clone());
        publish_frame(shared, frame);
        active.last_publish = Instant::now();
    }
    active.dirty = false;
    active.last_preview = preview;
    Ok(spp > 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{Bulb, Formula, Kifs, KifsKind, MandelBox};

    #[test]
    #[ignore = "requires CUDA; verifies Playa cache then play through the real worker"]
    fn cuda_preview_caches_inclusive_range_then_replays_native_presentation() {
        let _gpu_test = crate::test_gpu::lock();
        let service = RenderService::spawn(Wake::default());
        let mut scene = Scene::preset(Formula::Mandelbulb(Bulb::PRESET));
        scene.colour.on = false;
        scene.render.denoise.enabled = false;
        let request = crate::preview::PreviewRequest {
            generation: 901,
            scene: Arc::new(scene),
            render_profile: None,
            first: 250,
            last: 252,
            fps: 24.0,
            width: 32,
            height: 18,
            spp: 2,
            seed: 7,
            output_hdr: true,
            white_nits: 203.0,
            cache_fraction: 0.01,
            reserve_gb: 0.0,
            budget_ms: BACKGROUND_BUDGET_MS,
        };
        service
            .try_command(Command::BeginPreview {
                request,
                cache_all: true,
            })
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(90);
        let mut ready = false;
        let mut replayed = false;
        while Instant::now() < deadline {
            for event in service.drain_events() {
                match event {
                    RenderEvent::PreviewProgress {
                        completed, total, ..
                    } => {
                        assert_eq!(total, 3);
                        assert!(completed <= 3);
                    }
                    RenderEvent::PreviewReady { generation: 901 } => {
                        ready = true;
                        service
                            .try_command(Command::SeekPreview {
                                generation: 901,
                                number: 252,
                            })
                            .unwrap();
                    }
                    RenderEvent::PreviewFrame { number, frame, .. } => {
                        assert_eq!(frame.samples, 2);
                        assert_eq!(frame.light.len(), 32 * 18);
                        assert_eq!(frame.sdr_bytes.len(), 32 * 18 * 4);
                        assert_eq!(frame.hdr_bytes.len(), 32 * 18 * 16);
                        assert!(frame.light.iter().flatten().all(|value| value.is_finite()));
                        assert!(frame.colour_error.is_none());
                        if ready && number == 252 {
                            replayed = true;
                        }
                        service
                            .try_command(Command::RecyclePreviewFrame { frame })
                            .unwrap();
                    }
                    RenderEvent::PreviewFailed { error, .. } | RenderEvent::Error(error) => {
                        panic!("{error}")
                    }
                    _ => {}
                }
            }
            if replayed {
                break;
            }
            thread::sleep(Duration::from_millis(5));
        }
        assert!(
            ready && replayed,
            "inclusive three-frame cache and last frame playback"
        );
        service
            .try_command(Command::CancelPreview { generation: 901 })
            .unwrap();
    }

    #[test]
    fn switching_draft_and_final_profiles_preserves_final_pixels() {
        let _gpu_test = crate::test_gpu::lock();
        let full = crate::preview::PreviewRequest {
            generation: 41,
            scene: Arc::new(request().scene),
            render_profile: None,
            first: 0,
            last: 0,
            fps: 24.0,
            width: 1,
            height: 1,
            spp: 8,
            seed: 0,
            output_hdr: false,
            white_nits: 100.0,
            cache_fraction: 0.01,
            reserve_gb: 0.0,
            budget_ms: BACKGROUND_BUDGET_MS,
        };
        let mut cache = crate::preview::PreviewCache::new(full.clone(), false).unwrap();
        let mut frame = completed_frame(41, false);
        Arc::get_mut(&mut frame).unwrap().samples = 8;
        cache
            .store(0, &frame, cache.manager.current_epoch())
            .unwrap();
        let session = |cache| PreviewSession {
            cache,
            wanted: None,
            next_fill: 0,
            fill: false,
            job: None,
            target: None,
        };
        let mut active = Some(session(cache));
        let mut parked = None;
        let mut draft = full.clone();
        draft.generation += 1;
        draft.spp = 1;
        let cache = switch_preview_cache(&mut active, &mut parked, draft, true).unwrap();
        assert!(!cache.contains(0));
        assert!(parked.as_ref().unwrap().contains(0));
        active = Some(session(cache));
        let mut restored = full;
        restored.generation += 2;
        let cache = switch_preview_cache(&mut active, &mut parked, restored, false).unwrap();
        assert!(
            cache.contains(0),
            "returning to final quality must reuse its completed pixels"
        );
        let mut output = empty_presentation();
        cache.get_into(0, &mut output).unwrap();
        assert_eq!(output.samples, 8);
        assert_eq!(output.generation, 43);
        assert_eq!(parked.as_ref().unwrap().request.spp, 1);
    }

    #[test]
    fn preview_pool_waits_for_staged_bytes_and_reuses_arc_identity() {
        let _gpu_test = crate::test_gpu::lock();
        let mut pool = PresentationPool::default();
        let frame = pool.acquire().unwrap();
        let identity = Arc::as_ptr(&frame);
        let staged = frame.hdr_bytes.clone();
        pool.recycle(frame);
        let second = pool.acquire().unwrap();
        let third = pool.acquire().unwrap();
        assert!(
            pool.acquire().is_none(),
            "fixed three-slot pool must not grow while staging holds bytes"
        );
        drop(staged);
        let reused = pool.acquire().unwrap();
        assert_eq!(Arc::as_ptr(&reused), identity);
        pool.recycle(second);
        pool.recycle(third);
        pool.recycle(reused);
        assert_eq!(pool.created, 3);
    }
    #[test]
    fn preview_mailbox_retains_begin_coalesces_seek_and_recycles_replaced_results() {
        let _gpu_test = crate::test_gpu::lock();
        let shared = Shared {
            state: Mutex::new(Mailbox::default()),
            wake: Condvar::new(),
        };
        let req = crate::preview::PreviewRequest {
            generation: 5,
            scene: Arc::new(request().scene),
            render_profile: None,
            first: 10,
            last: 12,
            fps: 24.0,
            width: 1,
            height: 1,
            spp: 8,
            seed: 0,
            output_hdr: false,
            white_nits: 100.0,
            cache_fraction: 0.01,
            reserve_gb: 0.0,
            budget_ms: BACKGROUND_BUDGET_MS,
        };
        assert!(
            queue_preview_command(
                &shared,
                Command::BeginPreview {
                    request: req,
                    cache_all: true
                }
            )
            .is_ok()
        );
        assert!(
            queue_preview_command(
                &shared,
                Command::SeekPreview {
                    generation: 5,
                    number: 12
                }
            )
            .is_ok()
        );
        assert!(matches!(
            shared.state.lock().unwrap().preview_control,
            Some(Command::BeginPreview { .. })
        ));
        let old = completed_frame(5, true);
        let ptr = Arc::as_ptr(&old);
        push_event(
            &shared,
            RenderEvent::PreviewFrame {
                generation: 5,
                number: 10,
                frame: old,
            },
        );
        push_event(
            &shared,
            RenderEvent::PreviewFrame {
                generation: 5,
                number: 12,
                frame: completed_frame(5, true),
            },
        );
        assert!(
            shared
                .state
                .lock()
                .unwrap()
                .recycled_preview
                .iter()
                .flatten()
                .any(|frame| Arc::as_ptr(frame) == ptr)
        );
        assert!(queue_preview_command(&shared, Command::CancelPreview { generation: 5 }).is_ok());
        assert!(
            queue_preview_command(
                &shared,
                Command::SeekPreview {
                    generation: 5,
                    number: 10
                }
            )
            .is_ok()
        );
        assert_eq!(
            shared.state.lock().unwrap().preview_generation,
            None,
            "late seek must not resurrect cancelled work"
        );
    }
    #[test]
    fn preview_controller_shutdown_drops_pending_transport_without_waiting() {
        let _gpu_test = crate::test_gpu::lock();
        let service = RenderService {
            worker_thread: None,
            shared: Arc::new(Shared {
                state: Mutex::new(Mailbox {
                    stopping: true,
                    ..Mailbox::default()
                }),
                wake: Condvar::new(),
            }),
        };
        let req = crate::preview::PreviewRequest {
            generation: 5,
            scene: Arc::new(request().scene),
            render_profile: None,
            first: 10,
            last: 12,
            fps: 24.0,
            width: 1,
            height: 1,
            spp: 8,
            seed: 0,
            output_hdr: false,
            white_nits: 100.0,
            cache_fraction: 0.01,
            reserve_gb: 0.0,
            budget_ms: BACKGROUND_BUDGET_MS,
        };
        let mut controller = crate::preview::PreviewController::default();
        controller.start(req, false).unwrap();
        assert_eq!(controller.update(0.01, &service), None);
        assert_eq!(controller.position(), None);
        assert!(!controller.running());
        assert_eq!(controller.error(), Some("Render worker is unavailable"));
    }

    #[test]
    fn hdr_canvas_direct_bytes_match_original_float_canvas() {
        let _gpu_test = crate::test_gpu::lock();
        let pixels = [[-2.0, 0.0031308, 7.0, 0.0], [0.0, 1.0, 0.25, 0.5]];
        for gain in [1.0, 0.5, 100.0 / 203.0] {
            let original: Vec<[f32; 4]> = pixels
                .iter()
                .map(|p| {
                    [
                        crate::color::oetf(p[0] * gain),
                        crate::color::oetf(p[1] * gain),
                        crate::color::oetf(p[2] * gain),
                        1.0,
                    ]
                })
                .collect();
            assert_eq!(
                hdr_canvas_bytes(&pixels, gain),
                bytemuck::cast_slice::<_, u8>(&original)
            );
        }
    }
    fn request() -> ViewportRequest {
        ViewportRequest {
            generation: 1,
            scene: Scene::preset(Formula::Kifs(Kifs::preset(KifsKind::Tetrahedron))),
            width: 32,
            height: 24,
            target_spp: 8,
            paused: false,
            raw: false,
            active: true,
            interactive: false,
            preview: false,
            batch_budget_ms: 8.0,
            target_fps: 60.0,
            seed: 0,
            output_hdr: false,
            white_nits: 100.0,
        }
    }
    fn completed_frame(generation: u64, preview: bool) -> Arc<Frame> {
        Arc::new(Frame {
            generation,
            preview,
            width: 1,
            height: 1,
            pixels: vec![0],
            light: vec![[0.0; 4]],
            radiance: Vec::new(),
            light_kind: crate::color::DisplayLight::Relative,
            colour_error: None,
            denoised_samples: 0,
            denoise_ms: 0.0,
            denoise_error: None,
            samples: 1,
            active_tiles: (0, 0),
            converged: false,
            last_ms: 1.0,
            last_spp: 1,
            unresolved: 0.0,
            sdr_bytes: Arc::new(vec![0; 4]),
            hdr_bytes: Arc::new(Vec::new()),
        })
    }

    /// Relative light's 1.0 lands on the SDR white it is given (the monitor's, or BT.2408's
    /// 203), absolute light keeps 1.0 = 100 nits and its view's peak: an HDR10 snapshot is as
    /// bright as the screen.
    #[test]
    fn hdr_scale_puts_sdr_white_where_the_monitor_shows_it() {
        let _gpu_test = crate::test_gpu::lock();
        use crate::color::DisplayLight;
        let light = [[1.0, 0.5, 0.25, 1.0], [2.0, 0.0, 0.0, 1.0]];
        let relative = hdr_scale(DisplayLight::Relative, &light, 203.0, PngEncoding::Hdr10);
        assert_eq!(
            relative,
            HdrScale {
                unit_nits: 203.0,
                peak_nits: 406.0
            }
        );
        let hlg = hdr_scale(DisplayLight::Relative, &light, 203.0, PngEncoding::Hlg);
        assert_eq!(
            hlg,
            HdrScale {
                unit_nits: 203.0,
                peak_nits: HLG_REF_PEAK
            }
        );
        let absolute = hdr_scale(
            DisplayLight::Absolute { peak_nits: 1000.0 },
            &light,
            203.0,
            PngEncoding::Hdr10,
        );
        assert_eq!(
            absolute,
            HdrScale {
                unit_nits: 100.0,
                peak_nits: 1000.0
            }
        );
        let unbounded = hdr_scale(
            DisplayLight::Relative,
            &[[1.0e3, 0.0, 0.0, 1.0]],
            203.0,
            PngEncoding::Hdr10,
        );
        assert_eq!(
            unbounded.peak_nits, 10_000.0,
            "mDCV never exceeds what PQ carries"
        );

        // The written PQ code of white is PQ(203 nits).
        let mut frame = Arc::try_unwrap(completed_frame(1, false)).ok().unwrap();
        frame.light = vec![[1.0, 1.0, 1.0, 1.0]];
        let dir = std::env::temp_dir().join(format!("frac-hdr-scale-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("white.pq.png");
        frame
            .save(&path, FrameFile::Png(PngEncoding::Hdr10), 203.0)
            .unwrap();
        let decoder =
            png::Decoder::new(std::io::BufReader::new(std::fs::File::open(&path).unwrap()));
        let mut reader = decoder.read_info().unwrap();
        let mut buf = vec![0; reader.output_buffer_size().unwrap()];
        reader.next_frame(&mut buf).unwrap();
        let green = u16::from_be_bytes([buf[2], buf[3]]);
        let expected = (soluwell_viewport::display::pq(203.0) * 65535.0 + 0.5) as u16;
        assert!(
            green.abs_diff(expected) <= 1,
            "{green} vs PQ(203) {expected}"
        );
        // The metadata says the same: SDR white at 203 nits, mastered for the brightest light.
        let info = reader.info();
        let white = info
            .uncompressed_latin1_text
            .iter()
            .find(|t| t.keyword == "SDRReferenceWhite")
            .map(|t| t.text.clone());
        assert_eq!(white.as_deref(), Some("203 nits"));
        let mdcv = info.mastering_display_color_volume.expect("mDCV");
        assert_eq!(mdcv.max_luminance, 2_030_000, "203 nits in 0.0001 cd/m2");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// An HDR10 PNG of a P3-limited HDR view (ACES 2.0, 500 nits) is what the view rendered:
    /// `mDCV` records the measured 500-nit peak, `cICP` says PQ / BT.2020, and colour outside
    /// BT.709 (negative linear Rec.709 light) reaches the BT.2020 codes unclipped.
    #[test]
    fn hdr10_png_of_a_p3_view_keeps_its_peak_and_gamut() {
        let _gpu_test = crate::test_gpu::lock();
        use crate::color::DisplayLight;
        let ocio = crate::ocio::Ocio::load("ocio://studio-config-latest").unwrap();
        let sel = crate::ocio::Sel {
            display: "Rec.2100-PQ - Display".into(),
            view: "ACES 2.0 - HDR 500 nits (P3 D65)".into(),
            ..crate::color::default_selection()
        };
        let transform = ocio
            .transform(&ocio.resolve(&sel, true).unwrap(), true)
            .unwrap();
        let kind = transform.light().unwrap();
        let DisplayLight::Absolute { peak_nits } = kind else {
            panic!("{kind:?}")
        };
        assert!((peak_nits - 500.0).abs() < 5.0, "{peak_nits}");

        // Saturated ACEScg green (a brighter one is tone-mapped toward white, inside BT.709),
        // and a highlight far above the view's range.
        let mut light = [[0.0, 1.0, 0.0, 1.0], [1.0e3, 1.0e3, 1.0e3, 1.0]];
        transform.processor().apply_rgba(&mut light);
        assert!(
            light[0][..3].iter().any(|&v| v < -1.0e-3),
            "P3 green lies outside BT.709: {:?}",
            light[0]
        );

        let mut frame = Arc::try_unwrap(completed_frame(1, false)).ok().unwrap();
        frame.width = 2;
        frame.light = light.to_vec();
        frame.light_kind = kind;
        let dir = std::env::temp_dir().join(format!("frac-hdr-p3-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("p3.pq.png");
        frame
            .save(&path, FrameFile::Png(PngEncoding::Hdr10), 203.0)
            .unwrap();

        let bytes = std::fs::read(&path).unwrap();
        let cicp = bytes
            .windows(8)
            .find(|w| &w[..4] == b"cICP")
            .map(|w| w[4..8].to_vec());
        assert_eq!(cicp, Some(vec![9, 16, 0, 1]));
        let decoder =
            png::Decoder::new(std::io::BufReader::new(std::fs::File::open(&path).unwrap()));
        let mut reader = decoder.read_info().unwrap();
        let mut buf = vec![0; reader.output_buffer_size().unwrap()];
        reader.next_frame(&mut buf).unwrap();
        let mdcv = reader.info().mastering_display_color_volume.expect("mDCV");
        assert_eq!(
            mdcv.max_luminance,
            (peak_nits * 10_000.0).round() as u32,
            "the view's measured peak"
        );

        let codes = |nits: [f32; 3]| {
            nits.map(|v| {
                (soluwell_viewport::display::pq(v.clamp(0.0, 10_000.0)) * 65535.0 + 0.5) as u16
            })
        };
        let written: Vec<u16> = buf[..6]
            .chunks(2)
            .map(|c| u16::from_be_bytes([c[0], c[1]]))
            .collect();
        let exact = codes(soluwell_viewport::display::rec2020_nits(
            [light[0][0], light[0][1], light[0][2]],
            100.0,
        ));
        let clipped = codes(soluwell_viewport::display::rec2020_nits(
            light[0][..3]
                .iter()
                .map(|v| v.max(0.0))
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
            100.0,
        ));
        for c in 0..3 {
            assert!(
                written[c].abs_diff(exact[c]) <= 1,
                "channel {c}: {written:?} vs {exact:?}"
            );
        }
        assert_ne!(
            exact, clipped,
            "the test colour must tell a gamut clip apart"
        );
        // The highlight lands at the view's peak, not at SDR white.
        let highlight = u16::from_be_bytes([buf[10], buf[11]]);
        let peak_code = (soluwell_viewport::display::pq(peak_nits) * 65535.0 + 0.5) as u16;
        assert!(
            highlight.abs_diff(peak_code) <= 70,
            "{highlight} vs PQ({peak_nits}) {peak_code}"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Every snapshot file goes through `Frame::save`: an SDR frame saved as HDR10 is a PQ /
    /// BT.2020 PNG (cICP 9/16/0/1), the as-displayed SDR PNG carries no cICP, and the file
    /// names tell the transfers apart.
    #[test]
    fn snapshot_files_encode_what_they_name() {
        let _gpu_test = crate::test_gpu::lock();
        let frame = completed_frame(1, false);
        let dir = std::env::temp_dir().join(format!("frac-snapshot-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let cicp = |path: &std::path::Path| {
            let bytes = std::fs::read(path).unwrap();
            bytes
                .windows(8)
                .find(|w| &w[..4] == b"cICP")
                .map(|w| w[4..8].to_vec())
        };
        let srgb = |path: &std::path::Path| {
            std::fs::read(path)
                .unwrap()
                .windows(4)
                .any(|w| w == b"sRGB")
        };
        for (file, expected) in [
            (FrameFile::Png(PngEncoding::Sdr8), None),
            (FrameFile::Png(PngEncoding::Hdr10), Some(vec![9, 16, 0, 1])),
        ] {
            let path = dir.join(format!("shot.{}", file.suffix()));
            frame
                .save(&path, file, crate::color::BT2408_SDR_WHITE_NITS)
                .unwrap();
            assert_eq!(cicp(&path), expected, "{file:?}");
            // SDR says so with the sRGB chunk (egui-display's contract), HDR with cICP.
            assert_eq!(srgb(&path), expected.is_none(), "{file:?}");
        }
        let exr = dir.join(format!("shot.{}", FrameFile::DisplayExr.suffix()));
        frame
            .save(
                &exr,
                FrameFile::DisplayExr,
                crate::color::BT2408_SDR_WHITE_NITS,
            )
            .unwrap();
        // Relative (SDR view) light: 1.0 is the HDR10 PNG's SDR white, so the EXR says so.
        assert_eq!(
            crate::exr_io::read_attr::<f32>(&exr, "whiteLuminance"),
            Some(crate::color::BT2408_SDR_WHITE_NITS)
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    #[ignore = "requires actual CUDA and shared wgpu OIDN inference"]
    fn cuda_export_forces_oidn_below_interval_and_publishes_linear_result() {
        let _gpu_test = crate::test_gpu::lock();
        let service = RenderService::spawn(Wake::default());
        let mut scene = Scene::preset(Formula::Mandelbulb(Bulb::PRESET));
        scene.world_render = true;
        scene.camera.target = [1000.0; 3];
        scene.sun.intensity = 0.0;
        scene.sky.intensity = 1.0;
        scene.sky.horizon = [4.0, 2.0, 1.0];
        scene.sky.zenith = [4.0, 2.0, 1.0];
        scene.sky.background = true;
        scene.colour.on = false;
        scene.render.reinhard = false;
        scene.render.exposure_stops = 2.0;
        scene.render.saturation = 1.0;
        scene.render.denoise.enabled = true;
        scene.render.denoise.interval = 128;
        let (reply, receiver) = std::sync::mpsc::sync_channel(4);
        service
            .try_command(Command::RenderExport {
                id: 401,
                scene: Box::new(scene),
                width: 33,
                height: 25,
                spp: 2,
                seed: 401,
                budget_ms: BACKGROUND_BUDGET_MS,
                radiance: true,
                reply: Some(reply),
            })
            .unwrap();
        let until = Instant::now() + Duration::from_secs(90);
        loop {
            let event = receiver
                .recv_timeout(until.saturating_duration_since(Instant::now()))
                .expect("export completion");
            match event {
                RenderEvent::ExportFrame { id, frame } => {
                    assert_eq!(id, 401);
                    assert_eq!(frame.samples, 2);
                    assert_eq!(frame.denoised_samples, 2);
                    assert!(frame.denoise_error.is_none(), "{:?}", frame.denoise_error);
                    assert!(frame.colour_error.is_none());
                    assert_eq!(frame.radiance.len(), 33 * 25);
                    assert!(frame.radiance.iter().flatten().all(|v| v.is_finite()));
                    assert!(frame.radiance.iter().any(|p| p[0] > 1.0));
                    // OCIO is off: display light is the exposed working-space radiance in Rec.709.
                    for (light, radiance) in frame.light.iter().zip(&frame.radiance) {
                        let want = crate::color::to_709(
                            [radiance[0], radiance[1], radiance[2]].map(|v| 4.0 * v),
                        );
                        for k in 0..3 {
                            assert!((light[k] - want[k]).abs() < 1e-4);
                        }
                    }
                    break;
                }
                RenderEvent::ExportFailed { error, .. } | RenderEvent::Error(error) => {
                    panic!("{error}")
                }
                _ => {}
            }
        }
    }

    #[test]
    fn continuous_requests_cannot_starve_completed_previews_or_move_backwards() {
        let _gpu_test = crate::test_gpu::lock();
        let service = RenderService {
            worker_thread: None,
            shared: Arc::new(Shared {
                state: Mutex::new(Mailbox::default()),
                wake: Condvar::new(),
            }),
        };
        for completed in 1..100 {
            let mut next = request();
            next.interactive = true;
            next.preview = true;
            next.generation = completed + 3;
            service.request_viewport(next.clone());
            publish_frame(&service.shared, completed_frame(completed, true));
            // Another GUI tick arrives before the completed frame is consumed.
            next.generation += 1;
            service.request_viewport(next);
            let frame = service
                .take_latest_frame()
                .expect("Moving preview must make progress despite newer requests");
            assert_eq!(frame.generation, completed);
            publish_frame(
                &service.shared,
                completed_frame(completed.saturating_sub(1), true),
            );
            assert!(
                service.take_latest_frame().is_none(),
                "Older completion must not replace a newer preview"
            );
        }
    }

    #[test]
    fn obsolete_full_frames_and_idle_previews_remain_strictly_filtered() {
        let _gpu_test = crate::test_gpu::lock();
        let service = RenderService {
            worker_thread: None,
            shared: Arc::new(Shared {
                state: Mutex::new(Mailbox::default()),
                wake: Condvar::new(),
            }),
        };
        let mut next = request();
        next.generation = 10;
        next.interactive = true;
        next.preview = true;
        service.request_viewport(next.clone());
        publish_frame(&service.shared, completed_frame(9, false));
        assert!(service.take_latest_frame().is_none());
        next.interactive = false;
        service.request_viewport(next);
        publish_frame(&service.shared, completed_frame(9, true));
        assert!(service.take_latest_frame().is_none());
        publish_frame(&service.shared, completed_frame(10, false));
        assert_eq!(service.take_latest_frame().unwrap().generation, 10);
    }

    #[test]
    fn locked_moving_request_rejects_late_auto_preview() {
        let service = RenderService {
            worker_thread: None,
            shared: Arc::new(Shared {
                state: Mutex::new(Mailbox::default()),
                wake: Condvar::new(),
            }),
        };
        let mut next = request();
        next.generation = 10;
        next.interactive = true;
        next.preview = true;
        service.request_viewport(next.clone());
        publish_frame(&service.shared, completed_frame(9, true));
        // Switch policy while the camera is still moving, before consuming the old frame.
        next.generation = 11;
        next.preview = false;
        service.request_viewport(next);
        assert!(service.take_latest_frame().is_none());
        publish_frame(&service.shared, completed_frame(10, true));
        assert!(service.take_latest_frame().is_none());
        publish_frame(&service.shared, completed_frame(11, false));
        assert_eq!(service.take_latest_frame().unwrap().generation, 11);
    }

    #[test]
    fn interactive_export_batches_require_a_known_small_device_cost() {
        let _gpu_test = crate::test_gpu::lock();
        assert!(!background_batch_allowed(0.0, 0, 8.0, true));
        assert!(!background_batch_allowed(32.0, 1, 8.0, true));
        assert!(background_batch_allowed(16.0, 4, 8.0, true));
        assert!(background_batch_allowed(32.0, 1, 8.0, false));
        assert!(background_batch_allowed(0.0, 0, 8.0, false));
        assert!(
            background_batch_allowed(32.0, 1, 40.0, true),
            "the job's own budget"
        );
    }

    /// The seed is absent from Gpu's accumulation key, so only a new seed drops the viewport's
    /// film; every other change (camera, display, pause) keeps it and lets Gpu decide.
    #[test]
    fn viewport_drops_its_film_only_for_a_new_seed() {
        let mut viewport = Viewport::new(request());
        viewport.target = Some(Target::new(32, 24));
        let mut changed = request();
        changed.scene.camera.yaw_degrees += 1.0;
        changed.scene.render.exposure_stops += 1.0;
        changed.paused = true;
        viewport.update(changed.clone());
        assert!(viewport.target.is_some() && viewport.dirty);
        changed.seed += 1;
        viewport.update(changed);
        assert!(viewport.target.is_none());
    }
    #[test]
    fn bookmark_thumbnail_evaluates_frozen_world_at_first_frame_before_quality_cap() {
        let _gpu_test = crate::test_gpu::lock();
        let mut scene = Scene::preset(Formula::Mandelbulb(Bulb::PRESET));
        scene.render.max_bounces = 8;
        let document = crate::world::WorldDocument::from_scene(&scene);
        let expected = document.snapshot(document.first as f64).unwrap();
        scene.document = Some(Box::new(document));
        let thumbnail = thumbnail_snapshot(scene, 3).unwrap();
        assert!(thumbnail.document.is_none() && thumbnail.world_render);
        assert_eq!(thumbnail.objects, expected.objects);
        assert_eq!(thumbnail.lights, expected.lights);
        assert_eq!(thumbnail.render.max_bounces, 3);
        let mut preset = Scene::preset(Formula::Mandelbox(MandelBox::PRESET));
        preset.render.max_bounces = 7;
        let preset = thumbnail_snapshot(preset, 3).unwrap();
        assert!(preset.world_render && !preset.objects.is_empty());
        assert_eq!(preset.render.max_bounces, 3);
    }
    #[test]
    fn latest_mailbox_coalesces_and_commands_are_bounded_with_priority_cancel() {
        let _gpu_test = crate::test_gpu::lock();
        let service = RenderService {
            worker_thread: None,
            shared: Arc::new(Shared {
                state: Mutex::new(Mailbox::default()),
                wake: Condvar::new(),
            }),
        };
        for generation in 1..100 {
            let mut r = request();
            r.generation = generation;
            service.request_viewport(r);
        }
        assert_eq!(
            service
                .shared
                .state
                .lock()
                .unwrap()
                .viewport
                .as_ref()
                .unwrap()
                .generation,
            99
        );
        for _ in 0..QUEUE_LIMIT {
            service.try_command(Command::ReloadColour).unwrap();
        }
        assert_eq!(
            service.try_command(Command::ReloadColour),
            Err(CommandError::Full)
        );
        service.try_command(Command::CancelExport).unwrap();
        assert!(service.shared.state.lock().unwrap().cancel_export);
    }
    #[test]
    fn identical_paused_requests_do_not_dirty_and_preview_expires_without_ui_updates() {
        let _gpu_test = crate::test_gpu::lock();
        let mut initial = request();
        initial.paused = true;
        let mut viewport = Viewport::new(initial.clone());
        viewport.dirty = false;
        viewport.update(initial.clone());
        assert!(!viewport.dirty);
        initial.interactive = true;
        initial.paused = false;
        viewport.update(initial.clone());
        assert!(viewport_interactive(&viewport));
        // The UI evaluates the authored settle delay and explicitly sends the still profile.
        initial.interactive = false;
        initial.preview = false;
        viewport.update(initial);
        assert!(!viewport_interactive(&viewport));
        assert!(!viewport.request.preview);
    }

    #[test]
    fn export_reply_and_port_lifetime_are_independent_of_gui_backpressure() {
        let _gpu_test = crate::test_gpu::lock();
        let service = RenderService {
            worker_thread: None,
            shared: Arc::new(Shared {
                state: Mutex::new(Mailbox::default()),
                wake: Condvar::new(),
            }),
        };
        for id in 0..QUEUE_LIMIT as u64 {
            push_event(
                &service.shared,
                RenderEvent::ExportProgress {
                    id,
                    samples: 1,
                    total: 2,
                },
            );
        }
        assert_eq!(
            service.shared.state.lock().unwrap().events.len(),
            QUEUE_LIMIT
        );
        let port = service.port();
        drop(port.clone());
        assert!(
            port.is_available(),
            "Dropping a producer must not stop the service"
        );
        let (reply, receiver) = std::sync::mpsc::sync_channel(1);
        dispatch_export(
            &service.shared,
            Some(&reply),
            RenderEvent::ExportFailed {
                id: 400,
                error: "test".into(),
            },
        );
        assert!(matches!(
            receiver.try_recv(),
            Ok(RenderEvent::ExportFailed { id: 400, .. })
        ));
        assert_eq!(
            service.shared.state.lock().unwrap().events.len(),
            QUEUE_LIMIT
        );
        drop(service);
        assert!(!port.is_available());
        assert_eq!(
            port.try_command(Command::ReloadColour),
            Err(CommandError::Stopped)
        );
    }

    #[test]
    fn cancelled_completion_does_not_block_on_an_undrained_reply() {
        let _gpu_test = crate::test_gpu::lock();
        let shared = Shared {
            state: Mutex::new(Mailbox::default()),
            wake: Condvar::new(),
        };
        let (reply, _receiver) = std::sync::mpsc::sync_channel(1);
        reply
            .try_send(RenderEvent::ExportProgress {
                id: 1,
                samples: 1,
                total: 3,
            })
            .unwrap_or_else(|_| panic!("empty channel"));
        shared.state.lock().unwrap().cancel_export = true;
        dispatch_export(
            &shared,
            Some(&reply),
            RenderEvent::ExportFailed {
                id: 1,
                error: "cancelled".into(),
            },
        );
    }

    #[test]
    fn progress_is_coalesced_and_zero_or_huge_allocations_are_rejected() {
        let _gpu_test = crate::test_gpu::lock();
        let shared = Shared {
            state: Mutex::new(Mailbox::default()),
            wake: Condvar::new(),
        };
        for samples in 1..100 {
            push_event(
                &shared,
                RenderEvent::ExportProgress {
                    id: 4,
                    samples,
                    total: 100,
                },
            );
        }
        let state = shared.state.lock().unwrap();
        assert_eq!(state.events.len(), 1);
        assert!(matches!(
            state.events.front(),
            Some(RenderEvent::ExportProgress { samples: 99, .. })
        ));
        assert!(validate_size(0, 1, 1).is_err());
        assert!(validate_size(16384, 16384, 1).is_err());
        assert!(validate_size(32, 24, 0).is_err());
    }

    /// A thumbnail failure is a typed terminal event that survives a full event queue (plan1
    /// 1.7), and an invalid request is refused as such, not as a full queue (1.9).
    #[test]
    fn thumbnail_failures_survive_backpressure_and_invalid_requests_are_typed() {
        let service = RenderService {
            worker_thread: None,
            shared: Arc::new(Shared {
                state: Mutex::new(Mailbox::default()),
                wake: Condvar::new(),
            }),
        };
        for _ in 0..QUEUE_LIMIT {
            push_event(&service.shared, RenderEvent::Error("busy".into()));
        }
        push_event(
            &service.shared,
            RenderEvent::ThumbnailFailed {
                id: 9,
                error: "no kernel".into(),
            },
        );
        assert!(
            service
                .drain_events()
                .iter()
                .any(|e| matches!(e, RenderEvent::ThumbnailFailed { id: 9, .. }))
        );
        let refused = service.try_command(Command::Thumbnail {
            id: 1,
            scene: Box::new(request().scene),
            width: 8,
            height: 8,
            spp: 0,
            seed: 7,
            budget_ms: BACKGROUND_BUDGET_MS,
            max_bounces: 3,
        });
        assert!(matches!(refused, Err(CommandError::Invalid(_))));
    }
}
