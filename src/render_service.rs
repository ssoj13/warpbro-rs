//! CUDA rendering is owned by one worker. UI requests and completed viewport frames
//! occupy single replaceable slots; commands and completion events have bounded queues.
use std::collections::VecDeque;
use std::path::Path;
use std::sync::mpsc::{SyncSender, TrySendError};
use std::sync::{Arc, Condvar, Mutex, TryLockError};
use std::thread;
use std::time::{Duration, Instant};

use crate::render::{Gpu, Target};
use crate::scene::Scene;

const QUEUE_LIMIT: usize = 16;
const PREVIEW_HOLD: Duration = Duration::from_millis(180);

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
    pub seed: u32,
    pub output_hdr: bool,
    pub white_nits: f32,
}

/// How a PNG encodes a frame's display light.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum PngEncoding {
    /// 8-bit sRGB / BT.709 SDR: the SDR view's codes (an HDR view is rendered for SDR too).
    #[default]
    Sdr8,
    /// 16-bit BT.2100 PQ, BT.2020 primaries; `cICP`, `mDCV`, `cLLI`.
    Hdr10,
    /// 16-bit BT.2100 HLG, BT.2020 primaries, for a display of the given peak; `cICP`, `mDCV`.
    Hlg,
}
impl PngEncoding {
    pub const ALL: [Self; 3] = [Self::Sdr8, Self::Hdr10, Self::Hlg];
    pub fn label(self) -> &'static str {
        match self {
            Self::Sdr8 => "SDR · 8-bit sRGB / BT.709",
            Self::Hdr10 => "HDR10 · 16-bit PQ / BT.2020",
            Self::Hlg => "HLG · 16-bit BT.2020",
        }
    }
    pub fn hdr(self) -> bool {
        self != Self::Sdr8
    }
    /// What a monitor shows: HDR10 for an HDR output, else SDR. PQ, not HLG: the monitor shows
    /// absolute display light, which PQ encodes as it is.
    pub fn displayed(hdr: bool) -> Self {
        if hdr { Self::Hdr10 } else { Self::Sdr8 }
    }
    /// The file suffix after the stem, for every PNG writer (snapshot, export, CLI): HDR PNGs
    /// name their transfer so they are not taken for SDR (an HDR10 PNG in a viewer that
    /// ignores `cICP` looks washed out).
    pub fn suffix(self) -> &'static str {
        match self {
            Self::Sdr8 => "png",
            Self::Hdr10 => "pq.png",
            Self::Hlg => "hlg.png",
        }
    }
}

/// How a frame's display light becomes nits in an HDR PNG.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HdrScale {
    /// Nits of the light's 1.0: 100 for an HDR view's absolute light, the SDR white for
    /// relative light (the monitor's when it shows HDR, else BT.2408's 203).
    pub unit_nits: f32,
    /// The mastering display (`mDCV`, HLG system gamma): the HDR view's measured peak, or the
    /// brightest relative light in nits.
    pub peak_nits: f32,
}

/// The BT.2100 HLG reference display: the peak an HLG file of relative light is graded for,
/// so its SDR white (BT.2408: 203 nits) lands near 75 % signal, not at full signal.
pub const HLG_REFERENCE_PEAK_NITS: f32 = 1000.0;

/// The most light PQ carries (BT.2100).
const PQ_PEAK_NITS: f32 = 10_000.0;

/// The one rule from a frame's light to HDR file nits (`HdrScale`) for `encoding`, for every
/// PNG writer. Absolute light: 1.0 = 100 nits, the view's measured peak. Relative light: 1.0
/// at `sdr_white_nits`; the peak is the brightest light for PQ, the HLG reference display for
/// HLG (HLG is relative to its display).
pub fn hdr_scale(
    kind: crate::color::DisplayLight,
    light: &[[f32; 4]],
    sdr_white_nits: f32,
    encoding: PngEncoding,
) -> HdrScale {
    let (unit_nits, peak_nits) = match (kind, encoding) {
        (crate::color::DisplayLight::Absolute { peak_nits }, _) => (100.0, peak_nits),
        (crate::color::DisplayLight::Relative, PngEncoding::Hlg) => (sdr_white_nits, HLG_REFERENCE_PEAK_NITS),
        (crate::color::DisplayLight::Relative, _) => {
            let max = light.iter().map(|p| p[0].max(p[1]).max(p[2])).fold(1.0f32, f32::max);
            (sdr_white_nits, sdr_white_nits * max)
        }
    };
    HdrScale { unit_nits, peak_nits: peak_nits.min(PQ_PEAK_NITS) }
}

/// The file a viewport snapshot (or a PNG export) is saved as. An SDR PNG holds the SDR
/// rendering (an HDR view is rendered for SDR too, `Frame::sdr_bytes`); an HDR PNG and the
/// display EXR hold the display light an HDR monitor shows, unclipped above SDR white.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameFile {
    Png(PngEncoding),
    /// Linear display light, display primaries, `whiteLuminance` 100 nits.
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

/// Write display light as a PNG: the one encoder behind the viewport, export and CLI writers.
/// `light` is linear Rec.709 display light, `scale` says how it becomes nits (`hdr_scale`);
/// `sdr` yields the 8-bit sRGB codes, read only for SDR.
#[allow(clippy::too_many_arguments)]
pub fn write_png(
    path: &Path,
    width: usize,
    height: usize,
    light: &[[f32; 4]],
    sdr: impl FnOnce() -> Vec<u8>,
    encoding: PngEncoding,
    scale: HdrScale,
    overwrite: bool,
) -> Result<(), String> {
    let peak_nits = scale.peak_nits;
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    av_util_core::outfile::check_overwrite(path, overwrite).map_err(|e| e.to_string())?;
    let u16s = |codes: [f32; 3]| codes.map(|c| (c.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16);
    let nits = |p: &[f32; 4]| egui_display::rec2020_nits([p[0], p[1], p[2]], scale.unit_nits);
    let hdr = |code: &dyn Fn([f32; 3]) -> [f32; 3]| {
        egui_display::screenshot::Pixels::Rgba16(
            light
                .iter()
                .flat_map(|p| {
                    let [r, g, b] = u16s(code(nits(p)));
                    [r, g, b, 65535]
                })
                .collect(),
        )
    };
    let (output, pixels) = match encoding {
        PngEncoding::Sdr8 => (egui_display::Output::Sdr8, egui_display::screenshot::Pixels::Rgba8(sdr())),
        PngEncoding::Hdr10 => (
            egui_display::Output::Hdr10,
            hdr(&|n| n.map(|v| egui_display::pq(v.clamp(0.0, 10000.0)))),
        ),
        PngEncoding::Hlg => (
            egui_display::Output::Hlg,
            hdr(&|n| egui_display::hlg(n.map(|v| v.max(0.0)), peak_nits)),
        ),
    };
    let capture = egui_display::screenshot::Capture {
        output,
        width: width as u32,
        height: height as u32,
        // The file says where its SDR white is and which display it was graded for.
        white_nits: scale.unit_nits,
        peak_nits: if encoding.hdr() { peak_nits } else { scale.unit_nits },
        pixels,
    };
    capture.save(path).map(|_| ()).map_err(|e| e.to_string())
}

/// CPU data only. The GPU context, buffers and progressive targets never leave the worker.
pub struct Frame {
    pub generation: u64,
    pub preview: bool,
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<u32>,
    pub light: Vec<[f32; 4]>,
    /// Unexposed linear ACEScg radiance, populated for final export frames only.
    pub radiance: Vec<[f32; 4]>,
    /// Relative (SDR view) or absolute (HDR view, with its peak) display light.
    pub light_kind: crate::color::DisplayLight,
    pub colour_error: Option<String>,
    pub denoised_samples: u32,
    pub denoise_ms: f32,
    pub denoise_error: Option<String>,
    pub samples: u32,
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
pub(crate) fn hdr_canvas_bytes(light: &[[f32; 4]], gain: f32) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(light.len() * std::mem::size_of::<[f32; 4]>());
    for pixel in light {
        for value in [
            crate::color::oetf(pixel[0] * gain),
            crate::color::oetf(pixel[1] * gain),
            crate::color::oetf(pixel[2] * gain),
            1.0,
        ] {
            bytes.extend_from_slice(&value.to_ne_bytes());
        }
    }
    bytes
}
impl Frame {
    pub(crate) fn snapshot(
        target: &Target,
        generation: u64,
        preview: bool,
        output_hdr: bool,
        white_nits: f32,
    ) -> Self {
        let pixels = target.pixels.clone();
        let light = target.light.clone();
        let sdr_bytes = Arc::new(pixels.iter().flat_map(|p| p.to_le_bytes()).collect());
        let gain = if target.light_kind.hdr() && output_hdr {
            100.0 / white_nits.max(1.0)
        } else {
            1.0
        };
        let hdr_bytes = if output_hdr {
            Arc::new(hdr_canvas_bytes(&light, gain))
        } else {
            Arc::new(Vec::new())
        };
        Self {
            generation,
            preview,
            width: target.width,
            height: target.height,
            pixels,
            light,
            radiance: Vec::new(),
            light_kind: target.light_kind,
            colour_error: target.colour_error.clone(),
            // What the image shows: a raw viewport A/B reports no denoise.
            denoised_samples: if target.denoise_shown() {
                target.denoise.last_samples
            } else {
                0
            },
            denoise_ms: target.denoise.last_ms,
            denoise_error: target.denoise.error.clone(),
            samples: target.samples,
            converged: target.converged,
            last_ms: target.last_ms,
            last_spp: target.last_spp,
            unresolved: target.unresolved,
            sdr_bytes,
            hdr_bytes,
        }
    }
    /// Finished for a target of `spp`: all samples taken, or adaptive sampling converged.
    pub fn complete(&self, spp: u32) -> bool {
        self.samples >= spp || self.converged
    }
    pub fn msamples_per_s(&self) -> f64 {
        if self.last_ms <= 0.0 {
            return 0.0;
        }
        (self.width * self.height) as f64 * self.last_spp as f64
            / (self.last_ms as f64 / 1000.0)
            / 1.0e6
    }
    /// `sdr_white_nits`: the nits of relative light's 1.0 in an HDR file (`hdr_scale`).
    pub fn save_png(&self, path: &Path, encoding: PngEncoding, sdr_white_nits: f32, overwrite: bool) -> Result<(), String> {
        if let Some(e) = &self.colour_error {
            return Err(format!("Colour transform failed: {e}"));
        }
        let scale = hdr_scale(self.light_kind, &self.light, sdr_white_nits, encoding);
        write_png(path, self.width, self.height, &self.light, || self.sdr_bytes.as_ref().clone(), encoding, scale, overwrite)
    }
    pub fn save_display_exr(&self, path: &Path) -> Result<(), String> {
        if let Some(e) = &self.colour_error {
            return Err(format!("Colour transform failed: {e}"));
        }
        crate::exr_io::write_rgb(path, self.width, self.height, &self.light, &crate::color::DISPLAY_PRIMS, Some(100.0), true)
    }
    /// Save this frame as `file` (viewport snapshots: File menu and the viewport toolbar);
    /// `sdr_white_nits` as in [`Self::save_png`].
    pub fn save(&self, path: &Path, file: FrameFile, sdr_white_nits: f32) -> Result<(), String> {
        match file {
            FrameFile::Png(encoding) => self.save_png(path, encoding, sdr_white_nits, true),
            FrameFile::DisplayExr => self.save_display_exr(path),
        }
    }
}

pub enum Command {
    Thumbnail {
        id: u64,
        scene: Scene,
        width: usize,
        height: usize,
        spp: u32,
    },
    RenderExport {
        id: u64,
        scene: Scene,
        width: usize,
        height: usize,
        spp: u32,
        /// A sequence coordinator receives these directly, independently of GUI polling.
        reply: Option<SyncSender<RenderEvent>>,
    },
    CancelExport,
    ReloadColour,
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
#[derive(Clone)]
pub enum RenderEvent {
    Ready {
        name: String,
    },
    Error(String),
    Thumbnail {
        id: u64,
        frame: Arc<Frame>,
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
    last_error: Option<String>,
    preview_generation: Option<u64>,
    preview_control: Option<Command>,
    preview_seek: Option<(u64, u32)>,
    preview_store: Option<Command>,
    pending_preview_frame: Option<RenderEvent>,
    pending_preview_status: Option<RenderEvent>,
    recycled_preview: [Option<Arc<Frame>>; 3],
}
struct Shared {
    state: Mutex<Mailbox>,
    wake: Condvar,
}
pub struct RenderService {
    shared: Arc<Shared>,
}
impl RenderService {
    pub fn spawn() -> Self {
        let shared = Arc::new(Shared {
            state: Mutex::new(Mailbox::default()),
            wake: Condvar::new(),
        });
        let worker = shared.clone();
        // CUDA objects are constructed here, not on the caller and moved between threads.
        thread::Builder::new()
            .name("frac-cuda".into())
            .spawn(move || {
                let result =
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_worker(&worker)));
                if let Err(error) = result {
                    let message = error
                        .downcast_ref::<String>()
                        .cloned()
                        .or_else(|| error.downcast_ref::<&str>().map(|s| (*s).to_owned()))
                        .unwrap_or_else(|| "CUDA worker panicked".into());
                    let (export_id, reply) = {
                        let mut state = worker.state.lock().unwrap_or_else(|e| e.into_inner());
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
                    worker
                        .state
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .stopping = true;
                }
            })
            .expect("start CUDA worker");
        Self { shared }
    }
    pub fn request_viewport(&self, request: ViewportRequest) {
        let mut state = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
        state.requested_generation = request.generation;
        state.requested_interactive = request.active && !request.paused && request.interactive;
        state.viewport = Some(request);
        self.shared.wake.notify_one();
    }
    pub fn take_latest_frame(&self) -> Option<Arc<Frame>> {
        let mut state = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
        let frame = state.latest.take()?;
        frame_eligible(&frame, &state).then_some(frame)
    }
    pub fn drain_events(&self) -> Vec<RenderEvent> {
        let mut state = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
        let mut events: Vec<_> = state.events.drain(..).collect();
        events.extend(state.pending_export.take());
        events.extend(state.pending_failure.take());
        events.extend(state.last_error.take().map(RenderEvent::Error));
        events.extend(state.pending_preview_frame.take());
        events.extend(state.pending_preview_status.take());
        self.shared.wake.notify_one();
        events
    }
    /// Replaceable preview intents never wait for a renderer/cache mutex on the UI.
    pub fn try_preview_command(&self, command: Command) -> Result<(), Command> {
        queue_preview_command(&self.shared, command)
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
    pub fn try_command(&self, command: Command) -> Result<(), String> {
        self.port().try_command(command)
    }
}

fn queue_preview_command(shared: &Shared, command: Command) -> Result<(), Command> {
    let mut state = match shared.state.try_lock() {
        Ok(state) => state,
        Err(TryLockError::Poisoned(error)) => error.into_inner(),
        Err(TryLockError::WouldBlock) => return Err(command),
    };
    if state.stopping {
        return Err(command);
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
            if !state
                .recycled_preview
                .iter()
                .flatten()
                .any(|old| Arc::ptr_eq(old, &frame))
            {
                let Some(slot) = state
                    .recycled_preview
                    .iter_mut()
                    .find(|slot| slot.is_none())
                else {
                    return Err(Command::RecyclePreviewFrame { frame });
                };
                *slot = Some(frame);
            }
        }
        other => return Err(other),
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
        !self
            .shared
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .stopping
    }
    #[allow(dead_code)] // Availability alias for shutdown-aware background consumers.
    pub fn is_stopped(&self) -> bool {
        !self.is_available()
    }
    pub fn try_command(&self, command: Command) -> Result<(), String> {
        match &command {
            Command::Thumbnail {
                width, height, spp, ..
            }
            | Command::RenderExport {
                width, height, spp, ..
            } => validate_size(*width, *height, *spp)?,
            Command::BeginPreview { request, .. } | Command::CacheViewport { request, .. } => {
                request.validate()?
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
            return queue_preview_command(&self.shared, command)
                .map_err(|_| "Renderer mailbox is busy; retry next UI tick".to_string());
        }
        let mut state = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.stopping {
            return Err("CUDA worker is unavailable".into());
        }
        if matches!(command, Command::CancelExport) {
            state.cancel_export = true;
            state
                .commands
                .retain(|c| !matches!(c, Command::RenderExport { .. }));
        } else {
            if state.commands.len() >= QUEUE_LIMIT {
                return Err("Renderer command queue is full; retry next UI tick".into());
            }
            state.commands.push_back(command);
        }
        self.shared.wake.notify_one();
        Ok(())
    }
}
impl Drop for RenderService {
    fn drop(&mut self) {
        self.shared
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .stopping = true;
        self.shared.wake.notify_one();
        // Do not join on the UI thread; the worker finishes its bounded CUDA batch and exits.
    }
}
fn recycle_mailbox_frame(state: &mut Mailbox, frame: Arc<Frame>) {
    if state
        .recycled_preview
        .iter()
        .flatten()
        .any(|old| Arc::ptr_eq(old, &frame))
    {
        return;
    }
    if let Some(slot) = state
        .recycled_preview
        .iter_mut()
        .find(|slot| slot.is_none())
    {
        *slot = Some(frame);
    }
}
fn push_event(shared: &Shared, event: RenderEvent) {
    let mut state = shared.state.lock().unwrap_or_else(|e| e.into_inner());
    match &event {
        RenderEvent::PreviewFrame { generation, .. } => {
            if state.preview_generation == Some(*generation) {
                if let Some(RenderEvent::PreviewFrame { frame, .. }) =
                    state.pending_preview_frame.replace(event)
                {
                    recycle_mailbox_frame(&mut state, frame);
                }
            } else if let RenderEvent::PreviewFrame { frame, .. } = event {
                recycle_mailbox_frame(&mut state, frame);
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
        let _ = reply.try_send(event);
        return;
    }
    loop {
        match reply.try_send(event) {
            Ok(()) | Err(TrySendError::Disconnected(_)) => return,
            Err(TrySendError::Full(returned)) => event = returned,
        }
        let state = shared.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.stopping || state.cancel_export {
            return;
        }
        let _ = shared.wake.wait_timeout(state, Duration::from_millis(5));
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
    let mut state = shared.state.lock().unwrap_or_else(|e| e.into_inner());
    if frame_eligible(&frame, &state) && frame.generation >= state.newest_completed_generation {
        state.newest_completed_generation = frame.generation;
        state.latest = Some(frame);
    }
}

fn trace_key(request: &ViewportRequest) -> Vec<f32> {
    use crate::params::*;
    let mut key = request
        .scene
        .pack(request.width as u32, request.height as u32);
    for index in [
        P_EXPOSURE,
        P_SATURATION,
        P_TONEMAP,
        P_SAMPLE_BEGIN,
        P_SPP,
        P_SEED,
    ] {
        key[index] = 0.0;
    }
    key.extend(crate::render::scene_trace_data(&request.scene));
    key
}
struct Viewport {
    request: ViewportRequest,
    key: Vec<f32>,
    changed: Instant,
    full: Option<Target>,
    preview: Option<Target>,
    dirty: bool,
    last_preview: bool,
    last_publish: Instant,
}
impl Viewport {
    fn new(request: ViewportRequest) -> Self {
        let key = trace_key(&request);
        let changed = if request.interactive {
            Instant::now()
        } else {
            Instant::now() - PREVIEW_HOLD
        };
        Self {
            request,
            key,
            changed,
            full: None,
            preview: None,
            dirty: true,
            last_preview: false,
            last_publish: Instant::now() - Duration::from_millis(20),
        }
    }
    fn update(&mut self, request: ViewportRequest) {
        let key = trace_key(&request);
        if key != self.key
            || request.scene.palette != self.request.scene.palette
            || request.scene.environment.key() != self.request.scene.environment.key()
            || request.seed != self.request.seed
        {
            self.changed = Instant::now();
            // Seed is deliberately absent from Gpu's accumulation key, so explicitly discard its targets.
            if request.seed != self.request.seed {
                self.full = None;
                self.preview = None;
            }
            self.key = key;
        }
        if request.interactive && !self.request.interactive {
            self.changed = Instant::now();
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
        if self
            .slots
            .iter()
            .flatten()
            .any(|old| Arc::ptr_eq(old, &frame))
        {
            return;
        }
        if let Some(slot) = self.slots.iter_mut().find(|slot| slot.is_none()) {
            *slot = Some(frame);
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
        converged: false,
        last_ms: 0.0,
        last_spp: 0,
        unresolved: 0.0,
        sdr_bytes: Arc::new(Vec::new()),
        hdr_bytes: Arc::new(Vec::new()),
    }
}
fn preview_current(shared: &Shared, generation: u64) -> bool {
    let state = shared.state.lock().unwrap_or_else(|e| e.into_inner());
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
    if session
        .cache
        .get_into(number, Arc::get_mut(&mut frame).expect("unique pool frame"))
        .is_none()
    {
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
        let count = session
            .cache
            .request
            .frame_count()
            .expect("validated range");
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
                document.snapshot(number as f64)
            } else {
                Ok(request.scene.as_ref().clone())
            };
            let scene = match scene.and_then(|scene| {
                gpu.prepare_scene(&scene, request.width, request.height)
                    .map(|()| scene)
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
                        .unwrap_or_else(|| gpu.target(request.width, request.height)),
                    spp: request.spp,
                    reply: None,
                },
            ));
        }
    }
    let Some((_, _, job)) = &mut session.job else {
        return worked;
    };
    if !background_batch_allowed(job.target.last_ms, job.target.last_spp, interactive) {
        return worked;
    }
    gpu.prepare_target(&mut job.target, &job.scene, None);
    let batch = batch_size(&job.target, job.spp.saturating_sub(job.target.samples));
    let final_pass = job.target.samples.saturating_add(batch) >= job.spp;
    gpu.step(
        &mut job.target,
        &job.scene,
        batch,
        session.cache.request.seed,
        None,
        final_pass,
    );
    worked = true;
    if !preview_current(shared, generation) {
        return worked;
    }
    if job.target.complete(job.spp) {
        let (number, epoch, job) = session.job.take().expect("completed preview job");
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
        push_event(
            shared,
            RenderEvent::PreviewProgress {
                generation,
                completed: session.cache.count(),
                total: session
                    .cache
                    .request
                    .frame_count()
                    .expect("validated range"),
                bytes: session.cache.bytes(),
                resident: session.cache.resident(),
            },
        );
        worked |= present_preview(session, pool, shared);
    }
    worked
}

pub(crate) fn validate_size(width: usize, height: usize, spp: u32) -> Result<(), String> {
    if width == 0
        || height == 0
        || width > 16384
        || height > 16384
        || width.checked_mul(height).is_none_or(|n| n > 67_108_864)
    {
        return Err(
            "Render dimensions must be positive and at most 64 megapixels / 16384 per axis".into(),
        );
    }
    if spp == 0 || spp > 1_000_000 {
        return Err("Sample count must be 1..=1000000".into());
    }
    Ok(())
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

fn run_worker(shared: &Shared) {
    if shared
        .state
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .stopping
    {
        return;
    }
    let mut gpu = match Gpu::new() {
        Ok(gpu) => gpu,
        Err(error) => {
            push_event(shared, RenderEvent::Error(error));
            shared.state.lock().unwrap().stopping = true;
            return;
        }
    };
    push_event(
        shared,
        RenderEvent::Ready {
            name: gpu.name.clone(),
        },
    );
    let mut viewport: Option<Viewport> = None;
    let mut export: Option<RenderJob> = None;
    let mut thumbnail: Option<RenderJob> = None;
    let mut preview: Option<PreviewSession> = None;
    // Keep the other quality profile resident when switching draft/final transport.
    // Both caches retain Playa's byte budget and eviction policy.
    let mut parked_preview: Option<crate::preview::PreviewCache> = None;
    let mut presentations = PresentationPool::default();
    loop {
        let (request, command, cancel, preview_control, preview_seek, preview_store, recycled) = {
            let mut state = shared.state.lock().unwrap_or_else(|e| e.into_inner());
            if state.stopping {
                return;
            }
            // Allocate background render targets only once motion settles; allocation itself
            // can be expensive for final-resolution exports.
            let moving = viewport.as_ref().is_some_and(viewport_interactive)
                || state
                    .viewport
                    .as_ref()
                    .is_some_and(|r| r.active && !r.paused && r.interactive);
            let index = state.commands.iter().position(|command| match command {
                Command::Thumbnail { .. } => {
                    !moving && thumbnail.is_none() && state.events.len() < QUEUE_LIMIT
                }
                Command::RenderExport { reply, .. } => {
                    !moving
                        && export.is_none()
                        && (reply.is_some() || state.pending_export.is_none())
                }
                _ => true,
            });
            let command = index.and_then(|index| state.commands.remove(index));
            (
                state.viewport.take(),
                command,
                std::mem::take(&mut state.cancel_export),
                state.preview_control.take(),
                state.preview_seek.take(),
                state.preview_store.take(),
                std::mem::take(&mut state.recycled_preview),
            )
        };
        for frame in recycled.into_iter().flatten() {
            presentations.recycle(frame);
        }
        if cancel {
            export = None;
            let mut state = shared.state.lock().unwrap();
            state.active_export_id = None;
            state.active_export_reply = None;
        }
        if let Some(request) = request {
            if let Err(error) = validate_size(request.width, request.height, request.target_spp)
                .and_then(|()| gpu.prepare_scene(&request.scene, request.width, request.height))
            {
                push_event(shared, RenderEvent::Error(error));
            } else if let Some(active) = &mut viewport {
                active.update(request);
            } else {
                viewport = Some(Viewport::new(request));
            }
        }
        if let Some(control) = preview_control {
            match control {
                Command::BeginPreview { request, cache_all } => {
                    let generation = request.generation;
                    if let Some(session) = preview
                        .as_mut()
                        .filter(|session| session.cache.request.compatible_content(&request))
                    {
                        let first = request.first;
                        if let Err(error) = session.cache.restart(request, cache_all) {
                            session.cache.request.generation = generation;
                            fail_preview(session, shared, error);
                            continue;
                        }
                        session.wanted = Some(first);
                        session.next_fill = 0;
                        session.fill = cache_all;
                        if let Some((_, _, job)) = &mut session.job {
                            job.id = generation;
                        }
                        if let Some(v) = &mut viewport {
                            v.request.active = false;
                        }
                        push_event(
                            shared,
                            RenderEvent::PreviewProgress {
                                generation,
                                completed: session.cache.count(),
                                total: session
                                    .cache
                                    .request
                                    .frame_count()
                                    .expect("validated range"),
                                bytes: session.cache.bytes(),
                                resident: session.cache.resident(),
                            },
                        );
                    } else {
                        match switch_preview_cache(
                            &mut preview,
                            &mut parked_preview,
                            request,
                            cache_all,
                        ) {
                            Ok(cache) => {
                                if let Some(v) = &mut viewport {
                                    v.request.active = false;
                                }
                                push_event(
                                    shared,
                                    RenderEvent::PreviewProgress {
                                        generation,
                                        completed: cache.count(),
                                        total: cache
                                            .request
                                            .frame_count()
                                            .expect("validated range"),
                                        bytes: cache.bytes(),
                                        resident: cache.resident(),
                                    },
                                );
                                preview = Some(PreviewSession {
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
                    if let Some(session) = preview
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
        if let Some(Command::CacheViewport {
            request,
            number,
            frame,
        }) = preview_store
        {
            let generation = request.generation;
            if preview_current(shared, generation) {
                if preview
                    .as_ref()
                    .is_none_or(|s| !s.cache.request.compatible_content(&request))
                {
                    match switch_preview_cache(&mut preview, &mut parked_preview, request, false) {
                        Ok(cache) => {
                            preview = Some(PreviewSession {
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
                            continue;
                        }
                    }
                }
                if let Some(session) = preview
                    .as_mut()
                    .filter(|s| s.cache.request.generation == generation)
                {
                    let epoch = session.cache.manager.current_epoch();
                    match session.cache.store(number, &frame, epoch) {
                        Ok(()) => push_event(
                            shared,
                            RenderEvent::PreviewProgress {
                                generation,
                                completed: session.cache.count(),
                                total: session
                                    .cache
                                    .request
                                    .frame_count()
                                    .expect("validated range"),
                                bytes: session.cache.bytes(),
                                resident: session.cache.resident(),
                            },
                        ),
                        Err(error) => {
                            push_event(shared, RenderEvent::PreviewFailed { generation, error })
                        }
                    }
                }
            }
        }
        if let Some((generation, number)) = preview_seek {
            if let Some(session) = preview
                .as_mut()
                .filter(|p| p.cache.request.generation == generation)
            {
                if session.cache.index(number).is_some() {
                    session.cache.manager.increment_generation();
                    session.wanted = Some(number);
                    // Presentation is served below after a reusable pool slot becomes available.
                }
            }
        }
        if let Some(command) = command {
            match command {
                Command::BeginPreview { .. }
                | Command::CacheViewport { .. }
                | Command::SeekPreview { .. }
                | Command::CancelPreview { .. }
                | Command::RecyclePreviewFrame { .. } => {}
                Command::CancelExport => export = None,
                Command::ReloadColour => {
                    gpu.invalidate_colour();
                    if let Some(v) = &mut viewport {
                        v.dirty = true;
                    }
                }
                Command::RenderExport {
                    id,
                    scene,
                    width,
                    height,
                    spp,
                    reply,
                } => {
                    if let Err(error) = validate_size(width, height, spp)
                        .and_then(|()| gpu.prepare_scene(&scene, width, height))
                    {
                        dispatch_export(
                            shared,
                            reply.as_ref(),
                            RenderEvent::ExportFailed { id, error },
                        );
                    } else {
                        {
                            let mut state = shared.state.lock().unwrap();
                            state.active_export_id = Some(id);
                            state.active_export_reply = reply.clone();
                        }
                        export = Some(RenderJob {
                            id,
                            scene,
                            target: gpu.target(width, height),
                            spp,
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
                } => {
                    let scene = match thumbnail_snapshot(scene) {
                        Ok(scene) => scene,
                        Err(error) => {
                            push_event(shared, RenderEvent::Error(error));
                            continue;
                        }
                    };
                    if let Err(error) = validate_size(width, height, spp)
                        .and_then(|()| gpu.prepare_scene(&scene, width, height))
                    {
                        push_event(shared, RenderEvent::Error(error));
                    } else {
                        thumbnail = Some(RenderJob {
                            id,
                            scene,
                            target: gpu.target(width, height),
                            spp,
                            reply: None,
                        });
                    }
                }
            }
        }
        let mut worked = false;
        let interactive = viewport.as_ref().is_some_and(viewport_interactive);
        if let Some(active) = viewport.as_mut() {
            worked |= step_viewport(&mut gpu, active, shared);
        }
        if let Some(job) = export.as_mut().filter(|job| {
            background_batch_allowed(job.target.last_ms, job.target.last_spp, interactive)
        }) {
            let batch = batch_size(&job.target, job.spp.saturating_sub(job.target.samples));
            let final_pass = job.target.samples.saturating_add(batch) >= job.spp;
            gpu.step(
                &mut job.target,
                &job.scene,
                batch,
                job.id as u32,
                None,
                final_pass,
            );
            worked = true;
            let cancelled = shared.state.lock().unwrap().cancel_export;
            if !cancelled {
                if job.target.complete(job.spp) {
                    let mut frame = Frame::snapshot(&job.target, job.id, false, false, 100.0);
                    frame.radiance = gpu.scene_linear(&job.target);
                    dispatch_export(
                        shared,
                        job.reply.as_ref(),
                        RenderEvent::ExportFrame {
                            id: job.id,
                            frame: Arc::new(frame),
                        },
                    );
                    export = None;
                    let mut state = shared.state.lock().unwrap();
                    state.active_export_id = None;
                    state.active_export_reply = None;
                } else {
                    dispatch_export(
                        shared,
                        job.reply.as_ref(),
                        RenderEvent::ExportProgress {
                            id: job.id,
                            samples: job.target.samples,
                            total: job.spp,
                        },
                    );
                }
            }
        }
        if export.is_none() {
            if let Some(session) = &mut preview {
                worked |= step_preview(&mut gpu, session, &mut presentations, shared, interactive);
            }
        }
        if let Some(job) = &mut thumbnail {
            if !interactive && !job.target.complete(job.spp) {
                let batch = batch_size(&job.target, job.spp.saturating_sub(job.target.samples));
                let final_pass = job.target.samples.saturating_add(batch) >= job.spp;
                gpu.step(&mut job.target, &job.scene, batch, 7, None, final_pass);
                worked = true;
            }
            if job.target.complete(job.spp)
                && shared.state.lock().unwrap().events.len() < QUEUE_LIMIT
            {
                let frame = Arc::new(Frame::snapshot(&job.target, job.id, false, false, 100.0));
                push_event(shared, RenderEvent::Thumbnail { id: job.id, frame });
                thumbnail = None;
            }
        }
        if !worked {
            let state = shared.state.lock().unwrap_or_else(|e| e.into_inner());
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
                let _ = shared.wake.wait_timeout(state, Duration::from_millis(20));
            }
        }
    }
}
fn thumbnail_snapshot(mut scene: Scene) -> Result<Scene, String> {
    if let Some(document) = scene.document.take() {
        scene = document.snapshot(document.first as f64)?;
    }
    scene.render.max_bounces = scene.render.max_bounces.min(3);
    Ok(scene)
}

fn viewport_interactive(viewport: &Viewport) -> bool {
    viewport.request.active
        && !viewport.request.paused
        && (viewport.request.interactive || viewport.changed.elapsed() < PREVIEW_HOLD)
}
fn background_batch_allowed(last_ms: f32, last_spp: u32, interactive: bool) -> bool {
    // A CUDA kernel cannot be preempted here. If even one previously measured sample
    // exceeds the interactive budget (or its cost is unknown), defer it until motion ends.
    !interactive || (last_spp > 0 && last_ms > 0.0 && last_ms / last_spp as f32 <= 8.0)
}
pub(crate) fn batch_size(target: &Target, remaining: u32) -> u32 {
    if remaining == 0 {
        return 0;
    }
    let per_sample = target.last_ms / target.last_spp.max(1) as f32;
    let batch = if per_sample > 0.0 {
        (8.0 / per_sample) as u32
    } else {
        1
    };
    batch.clamp(1, 4).min(remaining)
}
fn step_viewport(gpu: &mut Gpu, active: &mut Viewport, shared: &Shared) -> bool {
    let request = &active.request;
    if !request.active {
        return false;
    }
    let preview =
        !request.paused && (request.interactive || active.changed.elapsed() < PREVIEW_HOLD);
    let (width, height) = if preview {
        ((request.width / 2).max(1), (request.height / 2).max(1))
    } else {
        (request.width, request.height)
    };
    let slot = if preview {
        &mut active.preview
    } else {
        &mut active.full
    };
    let fresh = slot
        .as_ref()
        .is_none_or(|t| t.width != width || t.height != height);
    if fresh {
        *slot = Some(gpu.target(width, height));
    }
    let target = slot.as_mut().unwrap();
    // Detect accumulation resets without the extra tonemap/OCIO/readback that a zero-spp
    // step would perform. The selected traced or display-only batch runs exactly once.
    if active.dirty || fresh || active.last_preview != preview {
        gpu.prepare_target(target, &request.scene, preview.then_some(2));
    }
    target.raw_view = request.raw;
    let goal = if preview {
        request.target_spp.min(64)
    } else {
        request.target_spp
    };
    let spp = if request.paused {
        0
    } else {
        batch_size(target, goal.saturating_sub(target.samples))
    };
    if spp == 0 && !active.dirty && !fresh && active.last_preview == preview {
        return false;
    }
    let final_pass = !preview && target.samples.saturating_add(spp) >= goal;
    gpu.step(
        target,
        &request.scene,
        spp,
        request.seed,
        preview.then_some(2),
        final_pass,
    );
    if active.dirty
        || fresh
        || active.last_preview != preview
        || target.complete(goal)
        || active.last_publish.elapsed() >= Duration::from_millis(16)
    {
        let frame = Arc::new(Frame::snapshot(
            target,
            request.generation,
            preview,
            request.output_hdr,
            request.white_nits,
        ));
        publish_frame(shared, frame);
        active.last_publish = Instant::now();
    }
    active.dirty = false;
    active.last_preview = preview;
    spp > 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires CUDA; verifies Playa cache then play through the real worker"]
    fn cuda_preview_caches_inclusive_range_then_replays_native_presentation() {
        let service = RenderService::spawn();
        let mut scene = Scene::preset(crate::params::FAMILY_BULB);
        scene.colour.on = false;
        scene.render.denoise.enabled = false;
        let request = crate::preview::PreviewRequest {
            generation: 901,
            scene: Arc::new(scene),
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
        let full = crate::preview::PreviewRequest {
            generation: 41,
            scene: Arc::new(request().scene),
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
        let shared = Shared {
            state: Mutex::new(Mailbox::default()),
            wake: Condvar::new(),
        };
        let req = crate::preview::PreviewRequest {
            generation: 5,
            scene: Arc::new(request().scene),
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
        let service = RenderService {
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
        };
        let mut controller = crate::preview::PreviewController::default();
        controller.start(req, false).unwrap();
        assert_eq!(controller.update(0.01, &service), None);
        assert_eq!(controller.position(), None);
        assert!(!controller.running());
        assert_eq!(controller.error(), Some("CUDA worker is unavailable"));
    }

    #[test]
    fn hdr_canvas_direct_bytes_match_original_float_canvas() {
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
            scene: Scene::preset(crate::params::FAMILY_KIFS),
            width: 32,
            height: 24,
            target_spp: 8,
            paused: false,
            raw: false,
            active: true,
            interactive: false,
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
        use crate::color::DisplayLight;
        let light = [[1.0, 0.5, 0.25, 1.0], [2.0, 0.0, 0.0, 1.0]];
        let relative = hdr_scale(DisplayLight::Relative, &light, 203.0, PngEncoding::Hdr10);
        assert_eq!(relative, HdrScale { unit_nits: 203.0, peak_nits: 406.0 });
        let hlg = hdr_scale(DisplayLight::Relative, &light, 203.0, PngEncoding::Hlg);
        assert_eq!(hlg, HdrScale { unit_nits: 203.0, peak_nits: HLG_REFERENCE_PEAK_NITS });
        let absolute = hdr_scale(DisplayLight::Absolute { peak_nits: 1000.0 }, &light, 203.0, PngEncoding::Hdr10);
        assert_eq!(absolute, HdrScale { unit_nits: 100.0, peak_nits: 1000.0 });
        let unbounded = hdr_scale(DisplayLight::Relative, &[[1.0e3, 0.0, 0.0, 1.0]], 203.0, PngEncoding::Hdr10);
        assert_eq!(unbounded.peak_nits, 10_000.0, "mDCV never exceeds what PQ carries");

        // The written PQ code of white is PQ(203 nits).
        let mut frame = Arc::try_unwrap(completed_frame(1, false)).ok().unwrap();
        frame.light = vec![[1.0, 1.0, 1.0, 1.0]];
        let dir = std::env::temp_dir().join(format!("frac-hdr-scale-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("white.pq.png");
        frame.save(&path, FrameFile::Png(PngEncoding::Hdr10), 203.0).unwrap();
        let decoder = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(&path).unwrap()));
        let mut reader = decoder.read_info().unwrap();
        let mut buf = vec![0; reader.output_buffer_size().unwrap()];
        reader.next_frame(&mut buf).unwrap();
        let green = u16::from_be_bytes([buf[2], buf[3]]);
        let expected = (egui_display::pq(203.0) * 65535.0 + 0.5) as u16;
        assert!(green.abs_diff(expected) <= 1, "{green} vs PQ(203) {expected}");
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
        use crate::color::DisplayLight;
        let ocio = crate::ocio::Ocio::load("ocio://studio-config-latest").unwrap();
        let sel = crate::ocio::Sel {
            display: "Rec.2100-PQ - Display".into(),
            view: "ACES 2.0 - HDR 500 nits (P3 D65)".into(),
            ..crate::color::default_selection()
        };
        let transform = ocio.transform(&ocio.resolve(&sel, true).unwrap(), true).unwrap();
        let kind = transform.light().unwrap();
        let DisplayLight::Absolute { peak_nits } = kind else { panic!("{kind:?}") };
        assert!((peak_nits - 500.0).abs() < 5.0, "{peak_nits}");

        // Saturated ACEScg green, and a highlight far above the view's range.
        let mut light = [[0.0, 20.0, 0.0, 1.0], [1.0e3, 1.0e3, 1.0e3, 1.0]];
        transform.processor().apply_rgba(&mut light);
        assert!(light[0][..3].iter().any(|&v| v < -1.0e-3), "P3 green lies outside BT.709: {:?}", light[0]);

        let mut frame = Arc::try_unwrap(completed_frame(1, false)).ok().unwrap();
        frame.width = 2;
        frame.light = light.to_vec();
        frame.light_kind = kind;
        let dir = std::env::temp_dir().join(format!("frac-hdr-p3-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("p3.pq.png");
        frame.save(&path, FrameFile::Png(PngEncoding::Hdr10), 203.0).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        let cicp = bytes.windows(8).find(|w| &w[..4] == b"cICP").map(|w| w[4..8].to_vec());
        assert_eq!(cicp, Some(vec![9, 16, 0, 1]));
        let decoder = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(&path).unwrap()));
        let mut reader = decoder.read_info().unwrap();
        let mut buf = vec![0; reader.output_buffer_size().unwrap()];
        reader.next_frame(&mut buf).unwrap();
        let mdcv = reader.info().mastering_display_color_volume.expect("mDCV");
        assert_eq!(mdcv.max_luminance, (peak_nits * 10_000.0).round() as u32, "the view's measured peak");

        let codes = |nits: [f32; 3]| nits.map(|v| (egui_display::pq(v.clamp(0.0, 10_000.0)) * 65535.0 + 0.5) as u16);
        let written: Vec<u16> = buf[..6].chunks(2).map(|c| u16::from_be_bytes([c[0], c[1]])).collect();
        let exact = codes(egui_display::rec2020_nits([light[0][0], light[0][1], light[0][2]], 100.0));
        let clipped = codes(egui_display::rec2020_nits(light[0][..3].iter().map(|v| v.max(0.0)).collect::<Vec<_>>().try_into().unwrap(), 100.0));
        for c in 0..3 {
            assert!(written[c].abs_diff(exact[c]) <= 1, "channel {c}: {written:?} vs {exact:?}");
        }
        assert_ne!(exact, clipped, "the test colour must tell a gamut clip apart");
        // The highlight lands at the view's peak, not at SDR white.
        let highlight = u16::from_be_bytes([buf[10], buf[11]]);
        let peak_code = (egui_display::pq(peak_nits) * 65535.0 + 0.5) as u16;
        assert!(highlight.abs_diff(peak_code) <= 70, "{highlight} vs PQ({peak_nits}) {peak_code}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Every snapshot file goes through `Frame::save`: an SDR frame saved as HDR10 is a PQ /
    /// BT.2020 PNG (cICP 9/16/0/1), the as-displayed SDR PNG carries no cICP, and the file
    /// names tell the transfers apart.
    #[test]
    fn snapshot_files_encode_what_they_name() {
        let frame = completed_frame(1, false);
        let dir = std::env::temp_dir().join(format!("frac-snapshot-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let cicp = |path: &std::path::Path| {
            let bytes = std::fs::read(path).unwrap();
            bytes.windows(8).find(|w| &w[..4] == b"cICP").map(|w| w[4..8].to_vec())
        };
        let srgb = |path: &std::path::Path| std::fs::read(path).unwrap().windows(4).any(|w| w == b"sRGB");
        for (file, expected) in [
            (FrameFile::Png(PngEncoding::Sdr8), None),
            (FrameFile::Png(PngEncoding::Hdr10), Some(vec![9, 16, 0, 1])),
        ] {
            let path = dir.join(format!("shot.{}", file.suffix()));
            frame.save(&path, file, crate::color::BT2408_SDR_WHITE_NITS).unwrap();
            assert_eq!(cicp(&path), expected, "{file:?}");
            // SDR says so with the sRGB chunk (egui-display's contract), HDR with cICP.
            assert_eq!(srgb(&path), expected.is_none(), "{file:?}");
        }
        let exr = dir.join(format!("shot.{}", FrameFile::DisplayExr.suffix()));
        frame.save(&exr, FrameFile::DisplayExr, crate::color::BT2408_SDR_WHITE_NITS).unwrap();
        assert!(exr.is_file());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    #[ignore = "requires actual CUDA and shared wgpu OIDN inference"]
    fn cuda_export_forces_oidn_below_interval_and_publishes_linear_result() {
        let service = RenderService::spawn();
        let mut scene = Scene::preset(crate::params::FAMILY_BULB);
        scene.world_render = true;
        scene.camera.target = [1000.0; 3];
        scene.lighting.sun_intensity = 0.0;
        scene.lighting.sky_intensity = 1.0;
        scene.lighting.sky_horizon = [4.0, 2.0, 1.0];
        scene.lighting.sky_zenith = [4.0, 2.0, 1.0];
        scene.lighting.background = true;
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
                scene,
                width: 33,
                height: 25,
                spp: 2,
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
                        let want = crate::color::to_709([radiance[0], radiance[1], radiance[2]].map(|v| 4.0 * v));
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
        let service = RenderService {
            shared: Arc::new(Shared {
                state: Mutex::new(Mailbox::default()),
                wake: Condvar::new(),
            }),
        };
        for completed in 1..100 {
            let mut next = request();
            next.interactive = true;
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
        let service = RenderService {
            shared: Arc::new(Shared {
                state: Mutex::new(Mailbox::default()),
                wake: Condvar::new(),
            }),
        };
        let mut next = request();
        next.generation = 10;
        next.interactive = true;
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
    fn interactive_export_batches_require_a_known_small_gpu_cost() {
        assert!(!background_batch_allowed(0.0, 0, true));
        assert!(!background_batch_allowed(32.0, 1, true));
        assert!(background_batch_allowed(16.0, 4, true));
        assert!(background_batch_allowed(32.0, 1, false));
        assert!(background_batch_allowed(0.0, 0, false));
    }

    #[test]
    fn display_and_pause_changes_reuse_the_trace_key_but_camera_changes_do_not() {
        let original = request();
        let mut changed = original.clone();
        changed.scene.render.exposure_stops += 1.0;
        changed.scene.render.saturation = 0.5;
        changed.scene.colour.view = "another display".into();
        changed.scene.render.denoise.enabled = false;
        changed.scene.render.denoise.interval = 256;
        changed.scene.render.denoise.quality = crate::denoise::Quality::High;
        changed.scene.render.denoise.mode = crate::denoise::Mode::Color;
        changed.paused = true;
        changed.raw = true;
        changed.output_hdr = true;
        changed.white_nits = 200.0;
        assert_eq!(trace_key(&original), trace_key(&changed));
        changed.scene.camera.yaw_degrees += 1.0;
        assert_ne!(trace_key(&original), trace_key(&changed));
    }
    #[test]
    fn bookmark_thumbnail_evaluates_frozen_world_at_first_frame_before_quality_cap() {
        let mut scene = Scene::preset(crate::params::FAMILY_BULB);
        scene.render.max_bounces = 8;
        let document = crate::world::WorldDocument::from_scene(&scene);
        let expected = document.snapshot(document.first as f64).unwrap();
        scene.document = Some(Box::new(document));
        let thumbnail = thumbnail_snapshot(scene).unwrap();
        assert!(thumbnail.document.is_none() && thumbnail.world_render);
        assert_eq!(thumbnail.objects, expected.objects);
        assert_eq!(thumbnail.lights, expected.lights);
        assert_eq!(thumbnail.render.max_bounces, 3);
        let mut legacy = Scene::preset(crate::params::FAMILY_BOX);
        legacy.render.max_bounces = 7;
        let legacy = thumbnail_snapshot(legacy).unwrap();
        assert!(!legacy.world_render && legacy.objects.is_empty());
        assert_eq!(legacy.render.max_bounces, 3);
    }
    #[test]
    fn world_trace_key_tracks_evaluated_objects_and_lights_but_ignores_labels() {
        let mut original = request();
        original.scene.world_render = true;
        original
            .scene
            .objects
            .push(Scene::preset(crate::params::FAMILY_BULB));
        original
            .scene
            .lights
            .push(crate::scene::Lighting::default());
        let mut changed = original.clone();
        changed.scene.objects[0].name = "rename".into();
        changed.scene.objects[0].material.preset = Some("label".into());
        assert_eq!(trace_key(&original), trace_key(&changed));
        changed.scene.objects[0].material.base += 0.1;
        assert_ne!(trace_key(&original), trace_key(&changed));
        changed = original.clone();
        changed.scene.objects[0].object_world =
            Some(glam::Mat4::from_translation(glam::Vec3::X).to_cols_array_2d());
        assert_ne!(trace_key(&original), trace_key(&changed));
        changed = original.clone();
        changed.scene.lights[0].sun_azimuth += 1.0;
        assert_ne!(trace_key(&original), trace_key(&changed));
        changed = original.clone();
        changed.scene.objects.clear();
        assert_ne!(trace_key(&original), trace_key(&changed));
    }
    #[test]
    fn latest_mailbox_coalesces_and_commands_are_bounded_with_priority_cancel() {
        let service = RenderService {
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
        assert!(service.try_command(Command::ReloadColour).is_err());
        service.try_command(Command::CancelExport).unwrap();
        assert!(service.shared.state.lock().unwrap().cancel_export);
    }
    #[test]
    fn identical_paused_requests_do_not_dirty_and_preview_expires_without_ui_updates() {
        let mut initial = request();
        initial.paused = true;
        let mut viewport = Viewport::new(initial.clone());
        viewport.dirty = false;
        viewport.update(initial.clone());
        assert!(!viewport.dirty);
        initial.interactive = true;
        initial.paused = false;
        viewport.update(initial.clone());
        assert!(viewport.changed.elapsed() < PREVIEW_HOLD);
        viewport.changed = Instant::now() - PREVIEW_HOLD;
        viewport.update(initial);
        assert!(
            viewport.changed.elapsed() >= PREVIEW_HOLD,
            "A stale interactive flag cannot hold preview forever"
        );
    }

    #[test]
    fn export_reply_and_port_lifetime_are_independent_of_gui_backpressure() {
        let service = RenderService {
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
        assert!(port.is_stopped());
    }

    #[test]
    fn cancelled_completion_does_not_block_on_an_undrained_reply() {
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
}
