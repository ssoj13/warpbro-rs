//! CUDA rendering is owned by one worker. UI requests and completed viewport frames
//! occupy single replaceable slots; commands and completion events have bounded queues.
use std::collections::VecDeque;
use std::path::Path;
use std::sync::mpsc::{SyncSender, TrySendError};
use std::sync::{Arc, Condvar, Mutex};
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
    pub active: bool,
    pub interactive: bool,
    pub seed: u32,
    pub output_hdr: bool,
    pub white_nits: f32,
}

/// CPU data only. The GPU context, buffers and progressive targets never leave the worker.
pub struct Frame {
    pub generation: u64,
    pub preview: bool,
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<u32>,
    pub light: Vec<[f32; 4]>,
    /// Unexposed linear Rec.709 radiance, populated for final export frames only.
    pub radiance: Vec<[f32; 4]>,
    pub hdr: bool,
    pub colour_error: Option<String>,
    pub samples: u32,
    pub last_ms: f32,
    pub last_spp: u32,
    pub sdr_bytes: Arc<Vec<u8>>,
    /// Extended-sRGB encoded RGBA32F, prepared with the requested output reference white.
    pub hdr_bytes: Arc<Vec<u8>>,
}
impl Frame {
    fn snapshot(
        target: &Target,
        generation: u64,
        preview: bool,
        output_hdr: bool,
        white_nits: f32,
    ) -> Self {
        let pixels = target.pixels.clone();
        let light = target.light.clone();
        let sdr_bytes = Arc::new(pixels.iter().flat_map(|p| p.to_le_bytes()).collect());
        let gain = if target.hdr && output_hdr {
            100.0 / white_nits.max(1.0)
        } else {
            1.0
        };
        let hdr_bytes = if output_hdr {
            let canvas: Vec<[f32; 4]> = light
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
            Arc::new(bytemuck::cast_slice(&canvas).to_vec())
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
            hdr: target.hdr,
            colour_error: target.colour_error.clone(),
            samples: target.samples,
            last_ms: target.last_ms,
            last_spp: target.last_spp,
            sdr_bytes,
            hdr_bytes,
        }
    }
    pub fn msamples_per_s(&self) -> f64 {
        if self.last_ms <= 0.0 {
            return 0.0;
        }
        (self.width * self.height) as f64 * self.last_spp as f64
            / (self.last_ms as f64 / 1000.0)
            / 1.0e6
    }
    pub fn save_png(&self, path: &Path) -> Result<(), String> {
        if let Some(e) = &self.colour_error {
            return Err(format!("Colour transform failed: {e}"));
        }
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        }
        let pixels = if self.hdr {
            let codes = self
                .light
                .iter()
                .flat_map(|p| {
                    let nits = egui_display::rec2020_nits([p[0], p[1], p[2]], 100.0);
                    let pq =
                        |n: f32| (egui_display::pq(n.clamp(0.0, 10000.0)) * 65535.0 + 0.5) as u16;
                    [pq(nits[0]), pq(nits[1]), pq(nits[2]), 65535]
                })
                .collect();
            egui_display::screenshot::Pixels::Rgba16(codes)
        } else {
            egui_display::screenshot::Pixels::Rgba8(self.sdr_bytes.as_ref().clone())
        };
        let capture = egui_display::screenshot::Capture {
            output: if self.hdr {
                egui_display::Output::Hdr10
            } else {
                egui_display::Output::Sdr8
            },
            width: self.width as u32,
            height: self.height as u32,
            white_nits: 100.0,
            peak_nits: if self.hdr { 1000.0 } else { 100.0 },
            pixels,
        };
        capture.save(path).map(|_| ()).map_err(|e| e.to_string())
    }
    pub fn save_display_exr(&self, path: &Path) -> Result<(), String> {
        if let Some(e) = &self.colour_error {
            return Err(format!("Colour transform failed: {e}"));
        }
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        }
        use exr::prelude::*;
        let mut image = Image::from_channels(
            (self.width, self.height),
            SpecificChannels::rgb(|pos: Vec2<usize>| {
                let p = self.light[pos.y() * self.width + pos.x()];
                (p[0], p[1], p[2])
            }),
        );
        image.attributes.chromaticities = Some(attribute::Chromaticities {
            red: Vec2(0.64, 0.33),
            green: Vec2(0.30, 0.60),
            blue: Vec2(0.15, 0.06),
            white: Vec2(0.3127, 0.3290),
        });
        image.layer_data.attributes.white_luminance = Some(100.0);
        image.write().to_file(path).map_err(|e| e.to_string())
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
}
#[derive(Clone)]
pub enum RenderEvent {
    Ready { name: String },
    Error(String),
    Thumbnail { id: u64, frame: Arc<Frame> },
    ExportProgress { id: u64, samples: u32, total: u32 },
    ExportFrame { id: u64, frame: Arc<Frame> },
    ExportFailed { id: u64, error: String },
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
        self.shared.wake.notify_one();
        events
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
            _ => {}
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
fn push_event(shared: &Shared, event: RenderEvent) {
    let mut state = shared.state.lock().unwrap_or_else(|e| e.into_inner());
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
fn validate_size(width: usize, height: usize, spp: u32) -> Result<(), String> {
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
    loop {
        let (request, command, cancel) = {
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
            )
        };
        if cancel {
            export = None;
            let mut state = shared.state.lock().unwrap();
            state.active_export_id = None;
            state.active_export_reply = None;
        }
        if let Some(request) = request {
            if let Err(error) = validate_size(request.width, request.height, request.target_spp) {
                push_event(shared, RenderEvent::Error(error));
            } else if let Some(active) = &mut viewport {
                active.update(request);
            } else {
                viewport = Some(Viewport::new(request));
            }
        }
        if let Some(command) = command {
            match command {
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
                    if let Err(error) = validate_size(width, height, spp) {
                        push_event(shared, RenderEvent::Error(error));
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
                    if let Err(error) = validate_size(width, height, spp) {
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
            gpu.step(&mut job.target, &job.scene, batch, job.id as u32, None);
            worked = true;
            let cancelled = shared.state.lock().unwrap().cancel_export;
            if !cancelled {
                if job.target.samples >= job.spp {
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
        if let Some(job) = &mut thumbnail {
            if !interactive && job.target.samples < job.spp {
                let batch = batch_size(&job.target, job.spp.saturating_sub(job.target.samples));
                gpu.step(&mut job.target, &job.scene, batch, 7, None);
                worked = true;
            }
            if job.target.samples >= job.spp
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
            if state.viewport.is_none() && !state.cancel_export {
                let _ = shared.wake.wait_timeout(state, Duration::from_millis(20));
            }
        }
    }
}
fn viewport_interactive(viewport: &Viewport) -> bool {
    viewport.request.active && !viewport.request.paused && viewport.changed.elapsed() < PREVIEW_HOLD
}
fn background_batch_allowed(last_ms: f32, last_spp: u32, interactive: bool) -> bool {
    // A CUDA kernel cannot be preempted here. If even one previously measured sample
    // exceeds the interactive budget (or its cost is unknown), defer it until motion ends.
    !interactive || (last_spp > 0 && last_ms > 0.0 && last_ms / last_spp as f32 <= 8.0)
}
fn batch_size(target: &Target, remaining: u32) -> u32 {
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
    let preview = !request.paused && active.changed.elapsed() < PREVIEW_HOLD;
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
    gpu.step(
        target,
        &request.scene,
        spp,
        request.seed,
        preview.then_some(2),
    );
    if active.dirty
        || fresh
        || active.last_preview != preview
        || target.samples >= goal
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
    fn request() -> ViewportRequest {
        ViewportRequest {
            generation: 1,
            scene: Scene::preset(crate::params::FAMILY_KIFS),
            width: 32,
            height: 24,
            target_spp: 8,
            paused: false,
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
            hdr: false,
            colour_error: None,
            samples: 1,
            last_ms: 1.0,
            last_spp: 1,
            sdr_bytes: Arc::new(vec![0; 4]),
            hdr_bytes: Arc::new(Vec::new()),
        })
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
        changed.paused = true;
        changed.output_hdr = true;
        changed.white_nits = 200.0;
        assert_eq!(trace_key(&original), trace_key(&changed));
        changed.scene.camera.yaw_degrees += 1.0;
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
