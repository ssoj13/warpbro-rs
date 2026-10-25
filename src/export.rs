//! Event-driven export coordinator and bounded CPU writer; the render worker renders every frame
//! (CUDA or WGPU), never the UI thread.
//!
//! The recipe is video-io's own: the document's [`OutputSettings`] hold a
//! `video_io::job::Preset` exactly as the panel (video-io-egui's `ExportDialog`) edits it, and an
//! [`ExportSettings`] is a `video_io::job::ExportSettings` (job, preset, folder: video-io's file
//! names and checks) plus what WarpBro fixes when a job starts (samples, output transform).
use crate::{
    ocio::{Ocio, OutputKind, OutputTransform, Sel},
    render_service::{
        Command, Frame, HdrLevels, PngEncoding, RenderEvent, RenderPort, RenderService,
    },
    scene::Scene,
};
use serde::{Deserialize, Serialize};
use std::{
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, SyncSender, TrySendError},
    },
    time::Duration,
};
use video_io::catalog::{Codec, Depth};
pub use video_io::job::{ExportFormat, ExportJob, Preset};
use video_io::still::{Still, StillOptions, StillPixels};
use video_io::video::{Backend, Outcome, Ticket};

/// The encode recipe of an export: the values of the document's `OutputSettings` node (branch
/// `/output`), deserialized strictly. `preset` is video-io's recipe as the export panel
/// (video-io-egui) edits it, stored as it is: no WarpBro mirror of it and no conversion, so a
/// format or codec switch never rewrites another field. Its `png_peak_nits` picks the HDR view
/// of an HDR export (`Ocio::output_transform`); the files record that view's measured peak
/// (`ExportSettings::light`). The rest is WarpBro's own. `Default` seeds the node in a fresh
/// document, so the defaults exist once.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputSettings {
    /// Format, base size (the Output render profile's resolution scale applies once when a job
    /// starts), the Video format's encode, the signal, the HDR peak, the PNG sequence's video and
    /// the image options.
    pub preset: Preset,
    /// Override world cadence for offline renders: OIDN runs once at the final sample.
    pub denoise_at_completion: bool,
    /// OCIO display / view overriding the automatic output transform; empty = automatic.
    pub output_display: String,
    pub output_view: String,
}
impl Default for OutputSettings {
    /// video-io's defaults, with WarpBro's choices for a path tracer's output: float EXR (the
    /// renderer's precision), the GPU HEVC encoder, ProRes 422 HQ, Deflate TIFF.
    fn default() -> Self {
        let preset = Preset::default();
        Self {
            preset: Preset {
                video: video_io::catalog::VideoSettings {
                    encoder: Some(video_io::catalog::Encoder::HevcVulkan),
                    prores: video_io::catalog::ProResProfile::Hq,
                    ..preset.video.clone()
                },
                still: StillOptions {
                    exr_depth: video_io::still::ExrDepth::Float,
                    tiff_compression: video_io::still::TiffCompression::Deflate,
                    ..preset.still.clone()
                },
                ..preset
            },
            denoise_at_completion: false,
            output_display: String::new(),
            output_view: String::new(),
        }
    }
}
impl OutputSettings {
    /// What a file of this format holds, under its current options.
    pub fn hint(&self) -> &'static str {
        match self.preset.format {
            ExportFormat::Exr => {
                "Scene-linear ACEScg (AP1-tagged) OpenEXR sequence; no display transform or exposure baked in."
            }
            ExportFormat::ExrPassThrough => {
                "Copies source EXR files; WarpBro renders its frames, so it has none to copy."
            }
            ExportFormat::Png => {
                "The monitor rendering baked in: SDR 8/16-bit sRGB / BT.709, or HDR10 / HLG 16-bit BT.2020 with cICP. Video: the finished sequence also encoded by ffmpeg."
            }
            ExportFormat::Jpeg => {
                "8-bit sRGB / BT.709 JPEG still sequence (baseline DCT, RGB only) with the display transform baked in."
            }
            ExportFormat::Tiff => {
                "8-bit or 16-bit sRGB / BT.709 TIFF still sequence (None, LZW, Deflate, or PackBits compression) with the display transform baked in."
            }
            ExportFormat::Tga => {
                "8-bit RLE-compressed Truevision TGA still sequence (RGB or RGBA) with the display transform baked in."
            }
            ExportFormat::Video => match self.preset.video.codec {
                Codec::Hevc => {
                    "H.265 / HEVC video (GPU Vulkan Video or CPU Kvazaar, 8/10-bit SDR or 10-bit HDR10 / HLG). Cancel saves completed frames."
                }
                Codec::H264 => {
                    "H.264 / AVC 8-bit SDR video (GPU Vulkan Video or CPU OpenH264) for universal compatibility. Cancel saves completed frames."
                }
                Codec::Av1 => {
                    "AV1 video (GPU Vulkan Video or CPU rav1e, 8/10-bit SDR or 10-bit HDR10 / HLG). Cancel saves completed frames."
                }
                Codec::Vp9 => {
                    "VP9 video (CPU, 8/10-bit SDR or 10-bit HDR10 / HLG). Cancel saves completed frames."
                }
                Codec::ProRes => {
                    "Apple ProRes 10-bit intra-frame mastering video in QuickTime (.mov) across all 6 profiles (Proxy, LT, 422, 422 HQ, 4444, 4444 XQ)."
                }
                Codec::Exv => {
                    "EXV float16 HDR container (.exv): stores linear display light directly in all 8 profiles (BC6H lossless/fast, PIZ, ZIPS, ZIP, RLE, DWAA, DWAB)."
                }
            },
        }
    }
    /// The display a file of this format is encoded for: its signal for the formats that can
    /// hold HDR, SDR for the others; None for scene-linear EXR.
    pub fn output_kind(&self) -> Option<OutputKind> {
        match self.preset.format {
            ExportFormat::Exr | ExportFormat::ExrPassThrough => None,
            format if format.hdr() => Some(OutputKind::from(self.preset.png)),
            _ => Some(OutputKind::Sdr),
        }
    }
    /// The output kind this recipe renders through OCIO under the scene's selection `current`;
    /// None for scene-linear EXR and for an SDR file of the built-in display (OCIO off). HDR
    /// always needs an OCIO HDR view.
    fn ocio_kind(&self, current: &Sel) -> Option<OutputKind> {
        self.output_kind()
            .filter(|&kind| kind != OutputKind::Sdr || current.on)
    }
    /// The output transform from ACEScg for this format, independent of the viewport: the
    /// override when set, else `Ocio::output_transform`; None when no OCIO transform applies
    /// ([`Self::ocio_kind`]).
    pub fn resolve_transform(
        &self,
        ocio: &Ocio,
        current: &Sel,
    ) -> Result<Option<OutputTransform>, String> {
        let Some(kind) = self.ocio_kind(current) else {
            return Ok(None);
        };
        if !self.output_display.is_empty() {
            let picked = Sel {
                display: self.output_display.clone(),
                view: self.output_view.clone(),
                ..current.clone()
            };
            let names = ocio
                .resolve(&picked, kind != OutputKind::Sdr)
                .map_err(|e| e.to_string())?;
            // The format can change without this override (Attribute Editor, a recalled
            // profile): a display / view of another kind would encode the wrong transfer.
            if !ocio.display_is(&names.display, kind)
                || !ocio.view_is(&names.display, &names.view, kind)
            {
                return Err(format!(
                    "Output transform override {} · {} does not encode {kind:?}; clear it or pick a {kind:?} display",
                    names.display, names.view
                ));
            }
            // The user picked this exact transform: nothing was replaced.
            return Ok(Some(OutputTransform {
                display: names.display,
                view: names.view,
                note: None,
            }));
        }
        ocio.output_transform(current, kind, self.preset.png_peak_nits)
            .map(Some)
            .map_err(|e| e.to_string())
    }
    /// An override names a display of one kind (SDR / PQ / HLG): a new kind starts automatic.
    pub fn normalize_override(&mut self, kind_before: Option<OutputKind>) {
        if self.output_kind() != kind_before {
            self.output_display.clear();
            self.output_view.clear();
        }
    }
}

/// One export: video-io's export (the preferences job, the document's recipe scaled by the
/// Output quality, the folder; video-io's file names and checks) and the values WarpBro fixes
/// when it starts. Assembled at runtime ([`Self::new`]), never persisted as a whole.
#[derive(Clone, Debug)]
pub struct ExportSettings {
    /// The job (stem, range, cadence, overwrite), the recipe and the folder: `io.output()`,
    /// `io.frame_path(n)` and `io.video_output()` name the files.
    pub io: video_io::job::ExportSettings,
    /// OIDN once at the final sample of every frame (`OutputSettings::denoise_at_completion`).
    pub denoise_at_completion: bool,
    /// Resolved Output quality target for a frozen export job, never authored here.
    pub samples: u32,
    /// The display / view this export renders through (`OutputSettings::resolve_transform`),
    /// with the note when it replaced the scene's view; None = scene-linear (EXR) or the scene's
    /// own built-in SDR display.
    pub transform: Option<OutputTransform>,
    /// The light of that transform: an HDR view's measured peak is a video's mastering peak, by
    /// the same rule (`video_io::export::hdr_scale`) the PNG writer applies to each frame's light.
    pub light: crate::color::DisplayLight,
    /// Display light back to that display's code values (`Ocio::encoder`), for 16-bit SDR
    /// stills; None = the built-in sRGB display.
    pub encoder: Option<Arc<vfx_ocio::Processor>>,
}
impl Default for ExportSettings {
    fn default() -> Self {
        Self {
            io: video_io::job::ExportSettings::default(),
            denoise_at_completion: false,
            samples: 32,
            transform: None,
            light: crate::color::DisplayLight::Relative,
            encoder: None,
        }
    }
}
impl ExportSettings {
    /// One export of `output` for `job`, with the Output quality fixed: `samples` per frame and
    /// the base size scaled once by `resolution_scale`; no folder yet (`io.resolve`). `start` and
    /// the form's check both build it here, so the form refuses exactly what a start would (an
    /// odd scaled HEVC size, for one).
    pub fn new(
        job: ExportJob,
        output: &OutputSettings,
        samples: u32,
        resolution_scale: f32,
    ) -> Self {
        let scale = |size: usize| ((size as f32 * resolution_scale).round() as usize).max(1);
        let mut preset = output.preset.clone();
        preset.width = scale(preset.width);
        preset.height = scale(preset.height);
        Self {
            io: video_io::job::ExportSettings {
                job,
                preset,
                dir: PathBuf::new(),
            },
            denoise_at_completion: output.denoise_at_completion,
            samples,
            ..Self::default()
        }
    }
    /// Apply after animation evaluation so authored denoise keys cannot restore periodic passes.
    pub fn apply_denoise_policy(&self, scene: &mut Scene) {
        if self.denoise_at_completion {
            scene.render.denoise.enabled = true;
            scene.render.denoise.interval = 0;
        }
    }
    /// The frame scene rendered through this export's output transform. An OCIO output replaces
    /// the legacy Reinhard curve, which would otherwise bypass OCIO (`ColorPipeline::apply_into`)
    /// and write an HDR file of SDR-relative light.
    pub fn apply_transform(&self, scene: &mut Scene) {
        if let Some(transform) = &self.transform {
            scene.colour.on = true;
            scene.colour.display = transform.display.clone();
            scene.colour.view = transform.view.clone();
            scene.render.reinhard = false;
        }
    }
    /// The encode backend of this export's video: the Video format's frames exist only in
    /// memory (the in-process encoder), a PNG sequence's video is encoded by ffmpeg afterwards.
    fn backend(&self) -> Backend {
        if self.io.preset.format == ExportFormat::Video {
            Backend::Native
        } else {
            Backend::Cli
        }
    }
    /// The export can start: video-io's checks of the name, the recipe, the range, the folder,
    /// the cadence and the encode backend (`video_io::job::ExportSettings::validate`), then the
    /// renderer's frame limits (`render::validate_size`). A pass-through copies source
    /// files; WarpBro renders every frame, so it has none.
    pub fn validate(&self) -> Result<(), String> {
        let preset = &self.io.preset;
        if !preset.format.renders() {
            return Err(format!(
                "{} copies source files; WarpBro renders its frames: choose EXR",
                preset.format.label()
            ));
        }
        self.io
            .validate(self.backend())
            .map_err(|e| e.to_string())?;
        crate::render::validate_size(preset.width, preset.height, self.samples)
            .map_err(|e| e.to_string())
    }
    /// Whether the files are 16-bit SDR codes (a 16-bit SDR PNG or TIFF): they come from the
    /// frame's display light through [`Self::encoder`], the same display encoding as the 8-bit
    /// codes the frame carries.
    fn codes16(&self) -> bool {
        let preset = &self.io.preset;
        preset.still.sixteen_bit
            && !preset.png.hdr()
            && matches!(preset.format, ExportFormat::Png | ExportFormat::Tiff)
    }
    /// Fix the colour of this export: the display / view `output` renders through under the
    /// scene's selection `current` (`OutputSettings::resolve_transform`), that view's light, and
    /// for 16-bit SDR stills its encoder. `ocio` is the loaded config (`ocio::State::config`); it
    /// is needed only when an OCIO transform applies, and must be the scene's own config.
    fn resolve_colour(
        &mut self,
        output: &OutputSettings,
        current: &Sel,
        ocio: Result<&Ocio, String>,
    ) -> Result<(), String> {
        let Some(kind) = output.ocio_kind(current) else {
            return Ok(());
        };
        let ocio = ocio?;
        let src = crate::ocio::source(&current.config);
        if ocio.src != src {
            return Err(format!(
                "The colour configuration {src} is still loading; start again in a moment"
            ));
        }
        let Some(transform) = output.resolve_transform(ocio, current)? else {
            return Ok(());
        };
        let names = ocio
            .resolve(
                &Sel {
                    display: transform.display.clone(),
                    view: transform.view.clone(),
                    ..current.clone()
                },
                true,
            )
            .map_err(|e| format!("{e:#}"))?;
        self.light = ocio
            .transform(&names, true)
            .map_err(|e| format!("{e:#}"))?
            .light()?;
        if kind != OutputKind::Sdr && !self.light.hdr() {
            return Err(format!(
                "{} · {} is not an HDR view",
                transform.display, transform.view
            ));
        }
        if self.codes16() {
            self.encoder = Some(Arc::new(
                ocio.encoder(&names).map_err(|e| format!("{e:#}"))?,
            ));
        }
        self.transform = Some(transform);
        Ok(())
    }
    /// `status` with the note of a replaced output view appended: the user is told at start and
    /// again in the final status, so the file's transform is never a silent substitute.
    fn with_note(&self, status: String) -> String {
        match self.transform.as_ref().and_then(|t| t.note.as_deref()) {
            Some(note) => format!("{status} ({note})"),
            None => status,
        }
    }
}

#[derive(Default)]
pub(crate) struct UiActions {
    pub job: Option<ExportJob>,
    pub output: Option<OutputSettings>,
    pub start: bool,
    pub cancel: bool,
    pub preset_file: Option<(bool, PathBuf)>,
    pub preset_dialog: Option<video_io_egui::PresetPickRequest>,
    pub preset_requests: Vec<video_io_egui::PresetRequest>,
}

pub struct ExportController {
    /// The preferences part of the next export (stem, range, cadence, overwrite).
    pub job: ExportJob,
    /// A copy of the document's bound OutputSettings recipe: the host fills it before
    /// [`Self::ui`] and authors any change back to the node (`App::export_ui`).
    pub output: OutputSettings,
    pub status: String,
    pub completed: u32,
    pub sample_progress: (u32, u32),
    run: Option<Run>,
    next_id: u64,
    /// Root of the per-export folders (`~/.warpbro/out`).
    pub out_root: PathBuf,
    /// The resolved settings of the last started export: where its files went.
    pub last: Option<ExportSettings>,
    /// The Output quality (samples, resolution scale) the host evaluated from the document
    /// for the form's check, or why the Output render profile cannot be exported.
    pub quality: Result<(u32, f32), String>,
    /// Shared `video-io-egui` export widget (`Preset`, `Format`, `Resolution`, and format/codec
    /// rows). It edits [`Self::output`]'s preset; it remembers per-codec settings across codec
    /// switches itself.
    dialog: video_io_egui::ExportDialog,
    /// Active file dialog for saving (`true`) or loading (`false`) a `.json` preset file.
    preset_picker: Option<(bool, egui_file_dialog::FileDialog)>,
    /// Woken by the export coordinator at every progress change (`Reports`).
    ui: crate::render_service::Wake,
}
#[derive(Clone, Default)]
struct Progress {
    status: String,
    completed: u32,
    samples: (u32, u32),
    done: bool,
    /// The light levels of the written HDR PNGs, for the video
    /// (`video_io::job::ExportSettings::encode_video`).
    levels: Option<HdrLevels>,
}
/// The coordinator's progress, shared with the UI thread, which is woken at every change
/// (`ExportController::update` reads it without waiting).
struct Reports {
    state: Mutex<Progress>,
    ui: crate::render_service::Wake,
}
impl Reports {
    /// Change the progress, then wake the UI to show it, after the lock is released
    /// (`Wake::wake`).
    fn edit<R>(&self, change: impl FnOnce(&mut Progress) -> R) -> R {
        let result = {
            let mut state = self.read();
            change(&mut state)
        };
        self.ui.wake();
        result
    }
    /// The progress, also after a panic poisoned its lock.
    fn read(&self) -> std::sync::MutexGuard<'_, Progress> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}
struct Run {
    settings: ExportSettings,
    progress: Arc<Reports>,
    cancel: Arc<AtomicBool>,
    coordinator: Option<std::thread::JoinHandle<()>>,
    /// The video encode the coordinator started after the last PNG, handed to the UI thread.
    video: Arc<Mutex<Option<VideoRun>>>,
    /// That encode once the UI took it over (`ExportController::update` polls it).
    encoding: Option<VideoRun>,
}
impl Drop for Run {
    /// Cancel and wait: closing WarpBro mid-export stops the render and an ffmpeg child (a
    /// dropped video-io `Ticket` cancels and joins its encode) instead of orphaning them.
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Release);
        if let Some(thread) = self.coordinator.take()
            && thread.join().is_err()
        {
            log::error!("export coordinator panicked");
        }
    }
}
impl Default for ExportController {
    fn default() -> Self {
        let mut dialog =
            video_io_egui::ExportDialog::new("render_encode", video_io_egui::Export::default());
        dialog.set_deferred_presets(true);
        dialog.set_presets_dir(video_io_egui::default_presets_dir("video-io"));
        Self {
            job: ExportJob::default(),
            output: OutputSettings::default(),
            status: String::new(),
            completed: 0,
            sample_progress: (0, 0),
            run: None,
            next_id: 100,
            out_root: crate::out_root(),
            last: None,
            ui: crate::render_service::Wake::default(),
            quality: Err("The Output render profile cannot be evaluated".into()),
            dialog,
            preset_picker: None,
        }
    }
}
/// Remove `dir` when it holds nothing: an export that failed or was cancelled before it wrote a
/// file (an encoder that cannot open, a cancel before the first frame) leaves no empty dated
/// folder behind.
fn remove_if_empty(dir: &Path) {
    match std::fs::read_dir(dir).map(|mut entries| entries.next().is_none()) {
        Ok(true) => {
            if let Err(error) = std::fs::remove_dir(dir) {
                log::warn!("Empty export folder {} not removed: {error}", dir.display());
            }
        }
        Ok(false) => {}
        Err(error) => log::warn!("Export folder {}: {error}", dir.display()),
    }
}
impl ExportController {
    /// Restore the preferences part; the recipe always comes from the document.
    pub fn restore(&mut self, job: ExportJob) {
        self.job = job;
    }
    pub fn is_running(&self) -> bool {
        self.run.is_some()
    }
    /// An export whose coordinator wakes the UI at every change (`Reports`), the UI waking
    /// nothing more: false while the video encode runs, which the UI must poll (`update`).
    pub fn is_encoding(&self) -> bool {
        self.run.as_ref().is_some_and(|run| run.encoding.is_some())
    }
    /// The controller of an app whose UI `ui` wakes at every export progress change.
    pub fn new(ui: crate::render_service::Wake) -> Self {
        Self {
            ui,
            ..Self::default()
        }
    }
    /// Once per UI frame: the coordinator's progress, then the video encode it started, which
    /// is polled here (never blocks) until it ends the run.
    pub fn update(&mut self) {
        let Some(run) = self.run.as_mut() else {
            return;
        };
        if run.encoding.is_none() {
            let Some(progress) = run.progress.state.try_lock().ok().map(|p| p.clone()) else {
                return;
            };
            self.status = progress.status;
            self.completed = progress.completed;
            self.sample_progress = progress.samples;
            if !progress.done {
                return;
            }
            let Some(encoding) = run.video.lock().unwrap_or_else(|e| e.into_inner()).take() else {
                self.run = None;
                return;
            };
            // A Cancel that came while the coordinator was handing the encode over.
            if run.cancel.load(Ordering::Acquire) {
                encoding.ticket.cancel();
            }
            run.encoding = Some(encoding);
        }
        let Some(encoding) = run.encoding.as_mut() else {
            return;
        };
        match encoding.ticket.try_finish() {
            None => self.status = encoding.status(),
            Some(result) => {
                self.status = encoding.finished(result);
                self.run = None;
            }
        }
    }
    /// Start exporting the frozen `scene`: its document's recipe and Output render profile, the
    /// job, and `ocio` (the loaded colour config, `ocio::State::config`) when the format renders
    /// through OCIO.
    pub fn start(
        &mut self,
        scene: &Scene,
        service: &RenderService,
        ocio: Result<&Ocio, String>,
    ) -> Result<(), String> {
        if self.run.is_some() {
            return Err("An export is already running".into());
        }
        // The frozen document is the only source of the recipe and the Output render profile.
        let document = scene
            .document
            .as_deref()
            .ok_or("Export needs the frozen world document")?;
        let output = document.output_settings()?;
        let (samples, scale) = document
            .effective_render(document.output_render_profile()?, f64::from(self.job.first))?
            .export_quality()?;
        let mut settings = ExportSettings::new(self.job.clone(), &output, samples, scale);
        // Validate against the output root first so a rejected export leaves no empty folder.
        settings.io.resolve(&self.out_root);
        settings.validate()?;
        settings.resolve_colour(&output, &scene.colour, ocio)?;
        settings.io.resolve(&crate::new_out_dir(&self.out_root)?);
        let scene = scene.clone();
        let port = service.port();
        let id = self.next_id;
        self.next_id += u64::from(settings.io.job.frame_count()) + 1;
        let progress = Arc::new(Reports {
            state: Mutex::new(Progress {
                status: settings.with_note("Starting export".into()),
                samples: (0, settings.samples),
                ..Default::default()
            }),
            ui: self.ui.clone(),
        });
        let cancel = Arc::new(AtomicBool::new(false));
        let video_slot = Arc::new(Mutex::new(None));
        let shared = progress.clone();
        let stop = cancel.clone();
        let handover = video_slot.clone();
        let config = settings.clone();
        let coordinator = std::thread::Builder::new()
            .name("frac-export-coordinator".into())
            .spawn(move || {
                // Ok(Some(status)): complete; Ok(None): cancelled before every frame was written.
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                    || -> Result<Option<String>, String> {
                        if !coordinate_export(&config, &scene, &port, id, &shared, &stop)? {
                            return Ok(None);
                        }
                        let folder = config.io.dir.display();
                        let Some(video) = config.io.video_output() else {
                            return Ok(Some(format!("Export complete: {folder}")));
                        };
                        if stop.load(Ordering::Acquire) {
                            return Ok(Some(format!(
                                "Video cancelled; the PNG sequence is in {folder}"
                            )));
                        }
                        let levels = shared.read().levels;
                        // ffmpeg writes a temp sibling, published only when it succeeds and
                        // removed on failure, Cancel or a dropped ticket. The UI polls the
                        // started encode (`ExportController::update`).
                        Ok(Some(
                            match config.io.encode_video(config.backend(), levels) {
                                Ok(ticket) => {
                                    let run = VideoRun {
                                        ticket,
                                        video,
                                        folder: config.io.dir.clone(),
                                    };
                                    let status = run.status();
                                    *handover.lock().unwrap_or_else(|e| e.into_inner()) = Some(run);
                                    status
                                }
                                Err(error) => {
                                    format!(
                                        "Video failed: {error}; the PNG sequence is in {folder}"
                                    )
                                }
                            },
                        ))
                    },
                ));
                let video = config.io.preset.format == ExportFormat::Video;
                let complete = matches!(result, Ok(Ok(Some(_))));
                let status = match result {
                    Ok(Ok(Some(status))) => status,
                    Ok(Ok(None)) => {
                        let completed = shared.read().completed;
                        if video && completed > 0 {
                            format!(
                                "Export cancelled; saved {completed} frames to {}",
                                config.io.output().display()
                            )
                        } else if video {
                            "Export cancelled before any complete frames".into()
                        } else {
                            format!("Export cancelled; {completed} frames retained")
                        }
                    }
                    Ok(Err(error)) => format!("Export failed: {error}"),
                    Err(_) => "Export failed: coordinator panicked".to_string(),
                };
                if !complete {
                    remove_if_empty(&config.io.dir);
                }
                let status = config.with_note(status);
                shared.edit(|state| {
                    state.status = status;
                    state.done = true;
                });
            })
            .map_err(|e| e.to_string())?;
        self.completed = 0;
        self.sample_progress = (0, settings.samples);
        self.status = settings.with_note("Starting export".into());
        self.last = Some(settings.clone());
        self.run = Some(Run {
            settings,
            progress,
            cancel,
            coordinator: Some(coordinator),
            video: video_slot,
            encoding: None,
        });
        Ok(())
    }
    /// Ask the running export to stop: the coordinator sees the flag between frames and ends the
    /// worker's export itself (`Command::CancelExport`), an encode in progress is cancelled.
    pub fn cancel(&mut self) {
        if let Some(run) = &self.run {
            run.cancel.store(true, Ordering::Release);
            if let Some(encoding) = &run.encoding {
                encoding.ticket.cancel();
            }
            self.status = "Cancelling export…".into();
        }
    }
    pub(crate) fn open_preset_dialog(&mut self, req: video_io_egui::PresetPickRequest) {
        let save = req.kind == video_io_egui::PresetPickKind::Save;
        let mut dialog = egui_file_dialog::FileDialog::new()
            .id(egui::Id::new(if save {
                "export.preset.save"
            } else {
                "export.preset.load"
            }))
            .add_file_filter_extensions("Export preset (.json)", vec!["json"])
            .initial_directory(req.dir);
        dialog.config_mut().default_file_filter = Some("Export preset (.json)".into());
        if save {
            dialog = dialog.default_file_name(&req.default_name);
            dialog.save_file();
        } else {
            dialog.pick_file();
        }
        self.preset_picker = Some((save, dialog));
    }
    /// The preset file dialog: a picked file is saved from, or loaded into, the dialog's preset
    /// (video-io-egui's preset files); a load becomes the recipe the host authors.
    fn update_preset_picker(&mut self, ctx: &egui::Context) -> Option<(bool, PathBuf)> {
        let Some((save, picker)) = &mut self.preset_picker else {
            return None;
        };
        picker.update(ctx);
        let picked = picker.take_picked().map(|path| (*save, path));
        if picked.is_some()
            || matches!(
                picker.state(),
                egui_file_dialog::DialogState::Closed | egui_file_dialog::DialogState::Cancelled
            )
        {
            self.preset_picker = None;
        }
        picked
    }

    pub(crate) fn preset_file(
        &mut self,
        save: bool,
        path: &Path,
    ) -> Result<Option<Preset>, String> {
        self.dialog.export_mut().preset = self.output.preset.clone();
        if save {
            let stem = self.dialog.save_preset_file(path)?;
            self.status = format!("Saved preset {stem}");
            Ok(None)
        } else {
            let stem = self.dialog.load_preset_file(path, None)?;
            self.status = format!("Loaded preset {stem}");
            Ok(Some(self.dialog.export().preset.clone()))
        }
    }

    pub(crate) fn handle_preset_request(
        &mut self,
        request: video_io_egui::PresetRequest,
    ) -> Result<(), String> {
        let result = request.apply()?;
        self.status = match &result {
            video_io_egui::PresetResult::Saved(named) => format!("Saved preset {}", named.name),
            video_io_egui::PresetResult::Deleted(name) => format!("Deleted preset {name}"),
            video_io_egui::PresetResult::Reloaded(_) => self.status.clone(),
        };
        self.dialog.accept_preset_result(result);
        Ok(())
    }

    /// The video-io-egui panel over [`Self::output`]'s preset, then the host extras (the output
    /// transform). The dialog edits the document's preset as it is: it gets a copy, and only an
    /// edit (a preset that differs afterwards) is written back, so showing the panel never
    /// changes the recipe. `ocio` / `current`: the loaded config and the scene's selection.
    fn panel(
        &mut self,
        ui: &mut egui::Ui,
        ocio: Result<&Ocio, &str>,
        current: &Sel,
        output: &mut OutputSettings,
        actions: &mut UiActions,
    ) {
        let kind_before = output.output_kind();
        self.dialog.export_mut().preset = output.preset.clone();
        let config = video_io_egui::ShowConfig {
            show_range: false,
            show_size: true,
            allow_pass_through: false,
            allow_bitrate: false,
            allow_speed_preset: false,
            allowed_codecs: None,
            show_backend: false,
            ..video_io_egui::ShowConfig::default()
        };
        self.dialog.show_panel_with(ui, &config, &mut |ui| {
            if let (Some(kind), Ok(ocio)) = (kind_before, ocio) {
                output.output_transform_ui(ui, ocio, current, kind);
            }
        });
        if self.dialog.export().preset != output.preset {
            output.preset = self.dialog.export().preset.clone();
        }
        if let Some(req) = self.dialog.take_preset_pick_request() {
            actions.preset_dialog = Some(req);
        }
        actions.preset_requests = self.dialog.take_preset_requests();
        // An override names a display of one kind (SDR / PQ / HLG): a new kind starts automatic.
        (*output).normalize_override(kind_before);
    }
    /// `timeline` is (first, last, fps, playhead) of the world document. The host fills
    /// [`Self::output`] from the document's OutputSettings node before this call and authors
    /// any change made here back to that node (`App::export_ui`). `ocio` is the loaded colour
    /// config (`ocio::State::config`), `current` the scene's selection. `rows` draws the host's
    /// own rows (its bindings) at the top of the form's grid, `freeze` the scene a start
    /// renders; both get the host's `host` (its document editor).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn ui(
        &mut self,
        ui: &mut egui::Ui,
        timeline: (u32, u32, f64, u32),
        ocio: Result<&Ocio, String>,
        current: &Sel,
        rows: impl FnOnce(&mut egui::Ui),
    ) -> UiActions {
        let running = self.is_running();
        let mut actions = UiActions {
            preset_file: self.update_preset_picker(ui.ctx()),
            ..Default::default()
        };
        let mut job = self.job.clone();
        let mut output = self.output.clone();
        ui.add_enabled_ui(!running, |ui| {
            egui::Grid::new("render_encode").num_columns(2).show(ui, |ui| {
                rows(ui);
                ui.label("Name");
                ui.horizontal(|ui| {
                    ui.text_edit_singleline(&mut job.name);
                    ui.label(format!(".{}", output.preset.suffix()));
                });
                ui.end_row();
                ui.label("Samples / frame");
                ui.label(match &self.quality {
                    Ok((samples, scale)) => {
                        format!("{samples} · Output Quality · {:.0}% resolution", scale * 100.0)
                    }
                    Err(_) => "Output Quality unavailable".into(),
                });
                ui.end_row();
                ui.label("Denoise");
                ui.checkbox(&mut output.denoise_at_completion, "Once at completion")
                    .on_hover_text(
                        "Run OIDN once after all samples of each exported frame; override Output profile denoise cadence.",
                    );
                ui.end_row();
                ui.label("Frame range");
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut job.first).range(0..=u32::MAX));
                    ui.label("…");
                    ui.add(egui::DragValue::new(&mut job.last).range(0..=u32::MAX));
                    if ui
                        .button("Current frame")
                        .on_hover_text("Render only the frame under the playhead.")
                        .clicked()
                    {
                        job.first = timeline.3;
                        job.last = timeline.3;
                    }
                    if ui
                        .button("Timeline")
                        .on_hover_text("The timeline's range, and its FPS for video.")
                        .clicked()
                    {
                        job.first = timeline.0;
                        job.last = timeline.1;
                        job.set_fps(timeline.2);
                    }
                });
                ui.end_row();
                // Every video: the Video format, a PNG sequence's video.
                if output.preset.video().is_some() {
                    self.fps_row(ui, &mut job);
                }
            });
            self.panel(ui, ocio.as_deref().map_err(String::as_str), current, &mut output, &mut actions);
            ui.small(output.hint());
            ui.weak(format!("Written to {}", self.out_root.join("<date_time>").display()));
            ui.weak("Animation is sampled at each frame. The scene and keys are frozen when export starts.");
            // The same quality, scaling and checks a start applies (`start`).
            let validation = self.quality.clone().and_then(|(samples, scale)| {
                let mut preview = ExportSettings::new(job.clone(), &output, samples, scale);
                preview.io.resolve(&self.out_root);
                preview.validate()
            });
            if let Err(error) = &validation {
                ui.colored_label(egui::Color32::LIGHT_RED, error);
            }
            if ui
                .add_enabled(validation.is_ok(), egui::Button::new("Start render"))
                .clicked()
            {
                actions.start = true;
            }
        });
        if running {
            let total = self
                .run
                .as_ref()
                .map(|r| r.settings.io.job.frame_count())
                .unwrap_or(1);
            ui.add(
                egui::ProgressBar::new(self.completed as f32 / total as f32)
                    .text(format!("{} / {total} frames written", self.completed)),
            );
            if ui.button("Cancel").clicked() {
                actions.cancel = true;
            }
        }
        ui.label(&self.status);
        if job != self.job {
            actions.job = Some(job);
        }
        if output != self.output {
            actions.output = Some(output);
        }
        actions
    }
    /// The "FPS" grid row of every video (the Video format, a PNG export's video).
    fn fps_row(&mut self, ui: &mut egui::Ui, job: &mut ExportJob) {
        ui.label("FPS");
        ui.horizontal(|ui| {
            let mut fps = job.fps();
            if ui
                .add(
                    egui::DragValue::new(&mut fps)
                        .speed(0.1)
                        .range(1.0..=240.0)
                        .max_decimals(3),
                )
                .changed()
            {
                job.set_fps(fps);
            }
            egui::ComboBox::from_id_salt("export_fps_presets")
                .selected_text("Presets")
                .show_ui(ui, |ui| {
                    for rate in [23.976, 24.0, 25.0, 29.97, 30.0, 50.0, 59.94, 60.0, 120.0] {
                        if ui
                            .selectable_label((job.fps() - rate).abs() < 0.001, rate.to_string())
                            .clicked()
                        {
                            job.set_fps(rate);
                            ui.close();
                        }
                    }
                });
        });
        ui.end_row();
    }
}
impl OutputSettings {
    /// "Output transform": what this export renders through from ACEScg, automatic by default.
    fn output_transform_ui(
        &mut self,
        ui: &mut egui::Ui,
        ocio: &Ocio,
        current: &Sel,
        kind: OutputKind,
    ) {
        let hdr = kind != OutputKind::Sdr;
        let resolved = self.resolve_transform(ocio, current);
        ui.label("Output transform").on_hover_text(
            "The OCIO display / view this file is rendered through from ACEScg, independent of the viewport. Automatic: the scene's own view when it fits the format, else the config's first fitting display (PQ / HLG by name) and, for HDR, the view whose measured peak is nearest the HDR peak.",
        );
        ui.vertical(|ui| {
            match &resolved {
                Ok(Some(t)) => {
                    ui.label(format!("{} · {}", t.display, t.view));
                    if let Some(note) = &t.note {
                        ui.colored_label(ui.visuals().warn_fg_color, note);
                    }
                }
                Ok(None) => {
                    ui.label("Built-in display (OCIO off)");
                }
                Err(error) => {
                    ui.colored_label(egui::Color32::LIGHT_RED, error);
                }
            }
            ui.horizontal(|ui| {
                let owned = |v: Vec<&str>| v.into_iter().map(str::to_owned).collect::<Vec<_>>();
                let displays = owned(
                    ocio.displays(hdr)
                        .into_iter()
                        .filter(|d| ocio.display_is(d, kind))
                        .collect(),
                );
                let before = self.output_display.clone();
                crate::ocio::combo(
                    ui,
                    "export.output_display",
                    &mut self.output_display,
                    &displays,
                    "auto",
                );
                if self.output_display != before {
                    self.output_view.clear();
                }
                let display = if self.output_display.is_empty() {
                    resolved
                        .as_ref()
                        .ok()
                        .and_then(Option::as_ref)
                        .map(|t| t.display.clone())
                        .unwrap_or_default()
                } else {
                    self.output_display.clone()
                };
                let views = owned(
                    ocio.views(&display, hdr)
                        .into_iter()
                        .filter(|v| ocio.view_is(&display, v, kind))
                        .collect(),
                );
                ui.add_enabled_ui(!self.output_display.is_empty(), |ui| {
                    crate::ocio::combo(
                        ui,
                        "export.output_view",
                        &mut self.output_view,
                        &views,
                        "first",
                    );
                });
            });
        });
        ui.end_row();
    }
}
/// Autonomous coordinator: GUI ticks are never required to render or publish the next frame.
fn coordinate_export(
    settings: &ExportSettings,
    scene: &Scene,
    port: &RenderPort,
    first_id: u64,
    progress: &Reports,
    cancel: &AtomicBool,
) -> Result<bool, String> {
    let mut writer = ExportWriter::spawn(settings.clone())?;
    let (reply, events) = mpsc::sync_channel(16);
    let result = (|| -> Result<bool, String> {
        for number in settings.io.job.first..=settings.io.job.last {
            let mut frame_scene = scene.evaluated(f64::from(number))?;
            settings.apply_denoise_policy(&mut frame_scene);
            settings.apply_transform(&mut frame_scene);
            let id = first_id + u64::from(number - settings.io.job.first);
            // The frame's quality is this request's: the profile's samples, the frame number as
            // its seed (re-exporting a frame reproduces it), the background batch budget, and
            // the scene-linear readback only for EXR, the one format that stores it.
            let denoise = frame_scene.render.denoise.enabled;
            loop {
                if cancel.load(Ordering::Acquire) {
                    return Ok(false);
                }
                match port.try_command(Command::RenderExport {
                    id,
                    scene: Box::new(frame_scene.clone()),
                    width: settings.io.preset.width,
                    height: settings.io.preset.height,
                    spp: settings.samples,
                    seed: number,
                    budget_ms: crate::render_service::BACKGROUND_BUDGET_MS,
                    radiance: settings.io.preset.format == ExportFormat::Exr,
                    reply: Some(reply.clone()),
                }) {
                    Ok(()) => break,
                    Err(crate::render_service::CommandError::Full) => {
                        std::thread::sleep(Duration::from_millis(5))
                    }
                    Err(error) => return Err(error.to_string()),
                }
            }
            let frame = loop {
                if cancel.load(Ordering::Acquire) {
                    return Ok(false);
                }
                if !port.is_available() {
                    return Err("Render worker stopped".into());
                }
                match writer.events.try_recv() {
                    Ok(WriteEvent::Written(written)) => {
                        acknowledge(progress, settings.io.job.first, written)?
                    }
                    Ok(WriteEvent::Failed(error)) => return Err(error),
                    Ok(WriteEvent::Cancelled) => return Ok(false),
                    _ => {}
                }
                match events.recv_timeout(Duration::from_millis(20)) {
                    Ok(RenderEvent::ExportProgress {
                        id: job,
                        samples,
                        total,
                    }) if job == id => {
                        progress.edit(|state| {
                            state.status = format!("Frame {number}: {samples}/{total} samples");
                            state.samples = (samples, total);
                        });
                    }
                    Ok(RenderEvent::ExportFrame { id: job, frame }) if job == id => break frame,
                    Ok(RenderEvent::ExportFailed { id: job, error }) if job == id => {
                        return Err(error);
                    }
                    Err(mpsc::RecvTimeoutError::Disconnected) => {
                        return Err("Render event bus closed".into());
                    }
                    _ => {}
                }
            };
            // A requested denoise that failed would silently write raw samples.
            if denoise && let Some(error) = &frame.denoise_error {
                return Err(format!("Frame {number}: denoising failed: {error}"));
            }
            progress.edit(|state| {
                state.status = format!("Writing frame {number}");
                state.samples = (settings.samples, settings.samples);
            });
            let mut pending = (number, frame);
            loop {
                // A fully rendered frame must enter the writer even if Cancel
                // arrives here; only incomplete GPU work is discarded.
                match writer
                    .frames
                    .as_ref()
                    .ok_or("Export input closed")?
                    .try_send(pending)
                {
                    Ok(()) => break,
                    Err(TrySendError::Full(frame)) => {
                        pending = frame;
                        match writer.events.try_recv() {
                            Ok(WriteEvent::Written(written)) => {
                                acknowledge(progress, settings.io.job.first, written)?
                            }
                            Ok(WriteEvent::Failed(error)) => return Err(error),
                            Ok(WriteEvent::Cancelled) => return Ok(false),
                            _ => {}
                        }
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    Err(TrySendError::Disconnected(_)) => {
                        return Err("Export writer stopped".into());
                    }
                }
            }
            // The GPU renders the next frame while the CPU writer handles this one.
            // A single queued writer frame bounds memory and applies backpressure off the GUI.
            if number != settings.io.job.last {
                continue;
            }
            loop {
                if cancel.load(Ordering::Acquire) {
                    return Ok(false);
                }
                match writer.events.recv_timeout(Duration::from_millis(20)) {
                    Ok(WriteEvent::Written(written)) => {
                        acknowledge(progress, settings.io.job.first, written)?
                    }
                    Ok(WriteEvent::Finished(levels)) => {
                        progress.edit(|state| state.levels = levels);
                        return Ok(true);
                    }
                    Ok(WriteEvent::Failed(error)) => return Err(error),
                    Ok(WriteEvent::Cancelled) => return Ok(false),
                    Err(mpsc::RecvTimeoutError::Disconnected) => {
                        return Err("Writer event bus closed".into());
                    }
                    _ => {}
                }
            }
        }
        Err("Export pipeline ended without finalization".into())
    })();
    if matches!(result, Ok(false)) {
        // Cancel is a graceful end of input: drain complete queued frames and
        // flush delayed encoder packets before publishing the partial movie.
        // This wait runs on the coordinator, never on the GUI thread.
        if let Err(error) = port.try_command(Command::CancelExport) {
            log::warn!("export cancel not sent to renderer: {error}");
        }
        progress.edit(|state| state.status = "Finishing export; saving completed frames…".into());
        writer.frames.take();
        loop {
            match writer.events.recv_timeout(Duration::from_millis(20)) {
                Ok(WriteEvent::Written(written)) => {
                    acknowledge(progress, settings.io.job.first, written)?
                }
                Ok(WriteEvent::Finished(_)) => break,
                Ok(WriteEvent::Failed(error)) => return Err(error),
                Ok(WriteEvent::Cancelled) => {
                    return Err("Export writer aborted during finalization".into());
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    return Err("Writer closed before finalization".into());
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
            }
        }
    } else if result.is_err() {
        writer.cancel.store(true, Ordering::Release);
        if let Err(error) = port.try_command(Command::CancelExport) {
            log::warn!("export cancel not sent to renderer: {error}");
        }
        loop {
            match writer.events.recv_timeout(Duration::from_millis(20)) {
                Ok(WriteEvent::Cancelled)
                | Ok(WriteEvent::Failed(_))
                | Ok(WriteEvent::Finished(_))
                | Err(mpsc::RecvTimeoutError::Disconnected) => break,
                _ => {}
            }
        }
    }
    result
}

fn acknowledge(progress: &Reports, first: u32, written: u32) -> Result<(), String> {
    progress.edit(|state| {
        if written != first + state.completed {
            return Err("Writer acknowledgements arrived out of order".into());
        }
        state.completed += 1;
        Ok(())
    })
}

/// A PNG sequence's video encode: started by the export coordinator once every frame is
/// written, then polled by the UI thread (`ExportController::update`).
struct VideoRun {
    ticket: Ticket,
    video: PathBuf,
    /// The export folder, named in the final status.
    folder: PathBuf,
}
impl VideoRun {
    /// The status line while it encodes: ffmpeg's encoded frames.
    fn status(&self) -> String {
        let name = self
            .video
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let (done, total) = (self.ticket.progress(), self.ticket.frames());
        format!("Encoding {name}: frame {done} / {total}")
    }

    /// The status line of how it ended.
    fn finished(&self, result: Result<Outcome, video_io::Error>) -> String {
        let folder = self.folder.display();
        match result {
            Ok(Outcome::Published) => {
                format!("Export complete: {folder}; video {}", self.video.display())
            }
            Ok(Outcome::Cancelled) => format!("Video cancelled; the PNG sequence is in {folder}"),
            Err(error) => format!("Video failed: {error}; the PNG sequence is in {folder}"),
        }
    }
}

enum WriteEvent {
    Written(u32),
    /// All frames written; the HDR PNGs' merged light levels (`HdrLevels::merge`).
    Finished(Option<HdrLevels>),
    Cancelled,
    Failed(String),
}
struct ExportWriter {
    frames: Option<SyncSender<(u32, Arc<Frame>)>>,
    events: Receiver<WriteEvent>,
    cancel: Arc<AtomicBool>,
}
impl ExportWriter {
    fn spawn(settings: ExportSettings) -> Result<Self, String> {
        settings.validate()?;
        let (frames, rx) = mpsc::sync_channel::<(u32, Arc<Frame>)>(1);
        let (tx, events) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let stop = cancel.clone();
        std::thread::Builder::new()
            .name("frac-export-writer".into())
            .spawn(move || {
                let send_terminal = |event: WriteEvent| {
                    if tx.send(event).is_err() && !stop.load(Ordering::Acquire) {
                        log::warn!("export writer terminal event dropped before cancellation");
                    }
                };
                let result = (|| -> Result<(), String> {
                    let preset = &settings.io.preset;
                    let mut video = if preset.format == ExportFormat::Video {
                        Some(VideoSink::new(&settings)?)
                    } else {
                        None
                    };
                    let mut levels: Option<HdrLevels> = None;
                    loop {
                        if stop.load(Ordering::Acquire) {
                            send_terminal(WriteEvent::Cancelled);
                            return Ok(());
                        }
                        let (number, frame) = match rx.recv_timeout(Duration::from_millis(30)) {
                            Ok(value) => value,
                            Err(mpsc::RecvTimeoutError::Timeout) => continue,
                            Err(mpsc::RecvTimeoutError::Disconnected) => {
                                if let Some(sink) = video {
                                    sink.finish(&stop)?;
                                }
                                send_terminal(WriteEvent::Finished(levels));
                                return Ok(());
                            }
                        };
                        if !frame.complete(settings.samples) {
                            return Err(format!(
                                "Incomplete frame: {} < {} samples",
                                frame.samples, settings.samples
                            ));
                        }
                        if frame.width != preset.width || frame.height != preset.height {
                            return Err("Export frame resolution changed".into());
                        }
                        let path = settings.io.frame_path(number);
                        let overwrite = settings.io.job.overwrite;
                        match preset.format {
                            ExportFormat::Video => {
                                video.as_mut().ok_or("Video sink missing")?.write(&frame)?
                            }
                            ExportFormat::Exr => {
                                write_exr(&path, &frame, &preset.still, overwrite)?
                            }
                            // Refused by `ExportSettings::validate` before the writer starts.
                            ExportFormat::ExrPassThrough => {
                                return Err("An EXR pass-through has no rendered frames".into());
                            }
                            // The mastering peak of an HDR PNG is the rendered view's measured
                            // one (`hdr_scale`).
                            format => {
                                if let Some(frame_levels) =
                                    write_still(&path, format, &frame, &settings)?
                                {
                                    levels = Some(
                                        levels.map_or(frame_levels, |l| l.merge(frame_levels)),
                                    );
                                }
                            }
                        }
                        if tx.send(WriteEvent::Written(number)).is_err() {
                            if stop.load(Ordering::Acquire) {
                                return Ok(());
                            }
                            return Err(
                                "Export coordinator closed before frame acknowledgement".into()
                            );
                        }
                        if number == settings.io.job.last {
                            if let Some(sink) = video {
                                sink.finish(&stop)?;
                            }
                            if stop.load(Ordering::Acquire) {
                                send_terminal(WriteEvent::Cancelled);
                            } else {
                                send_terminal(WriteEvent::Finished(levels));
                            }
                            return Ok(());
                        }
                    }
                })();
                if let Err(error) = result {
                    send_terminal(WriteEvent::Failed(error));
                }
            })
            .map_err(|e| e.to_string())?;
        Ok(Self {
            frames: Some(frames),
            events,
            cancel,
        })
    }
}
impl Drop for ExportWriter {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Release);
    }
}

fn write_exr(
    path: &Path,
    frame: &Frame,
    still: &StillOptions,
    overwrite: bool,
) -> Result<(), String> {
    if frame.radiance.len() != frame.width * frame.height {
        return Err("Scene-linear radiance is missing".into());
    }
    crate::exr_io::write_exr(
        path,
        frame.width,
        frame.height,
        &frame.radiance,
        &crate::color::WORKING_PRIMS,
        None,
        still.exr_depth,
        still.exr_compression,
        still.exr_dwa_level,
        still.channels,
        overwrite,
    )
}

/// Write `frame` as a still of `format` (PNG, JPEG, TIFF, TGA) under `settings`' recipe. Every
/// SDR file holds the display codes of the frame's output transform: its 8-bit codes as the
/// renderer made them, or 16-bit ones from its display light through the same display encoding
/// (`ExportSettings::encoder`). A PNG of display light (8-bit SDR or HDR) is video-io's PNG
/// writer over the light. Missing codes or light are an error, never another encoding.
fn write_still(
    path: &Path,
    format: ExportFormat,
    frame: &Frame,
    settings: &ExportSettings,
) -> Result<Option<HdrLevels>, String> {
    if let Some(e) = &frame.colour_error {
        return Err(format!("Colour transform failed: {e}"));
    }
    let preset = &settings.io.preset;
    let signal = preset.png;
    let codes16: Vec<u16>;
    let pixels = if format == ExportFormat::Png && (signal.hdr() || !preset.still.sixteen_bit) {
        StillPixels::DisplayLight {
            light: &frame.light,
            sdr: &frame.sdr_bytes,
            scale: video_io::export::hdr_scale(
                frame.light_kind,
                &frame.light,
                crate::color::BT2408_SDR_WHITE_NITS,
                signal,
            ),
        }
    } else if settings.codes16() {
        codes16 = sdr_codes16(frame, settings.encoder.as_deref())?;
        StillPixels::Codes16(&codes16)
    } else {
        if frame.sdr_bytes.len() != frame.width * frame.height * 4 {
            return Err("SDR display codes are missing".into());
        }
        StillPixels::Codes8(&frame.sdr_bytes)
    };
    let still = Still {
        width: frame.width as u32,
        height: frame.height as u32,
        pixels,
        signal,
    };
    video_io::still::write_still(
        path,
        format,
        &preset.still,
        &still,
        settings.io.job.overwrite,
    )
    .map_err(|e| e.to_string())
}

/// 16-bit SDR codes of `frame`'s display light: through `encoder` (`Ocio::encoder`: the inverse
/// of the output transform's light chain, so the codes are the display's own), or the sRGB OETF
/// of the built-in display (whose 8-bit codes are `color::srgb_u8` of the same light).
fn sdr_codes16(frame: &Frame, encoder: Option<&vfx_ocio::Processor>) -> Result<Vec<u16>, String> {
    if frame.light.len() != frame.width * frame.height {
        return Err("Display light is missing".into());
    }
    let mut codes = frame.light.clone();
    match encoder {
        Some(encoder) => encoder.apply_rgba(&mut codes),
        None => {
            for p in &mut codes {
                for c in &mut p[..3] {
                    *c = crate::color::oetf(*c);
                }
            }
        }
    }
    Ok(codes
        .iter()
        .flat_map(|p| p.map(|v| (v.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16))
        .collect())
}

/// Native video sink over [`video_io::native::Encoder`]: every codec of the recipe's
/// `VideoSettings` (`Hevc`, `H264`, `Av1`, `Vp9`, `ProRes`, `Exv`), 8-bit (`Rgb24`), 10-bit
/// (`Rgba64` SDR/HDR10/HLG), and float16 EXV (`RgbaF32`).
struct VideoSink {
    encoder: video_io::native::Encoder,
    codec: Codec,
    depth: Depth,
    signal: PngEncoding,
    rgb: Vec<u8>,
    rgba16: Vec<u16>,
    width: usize,
    height: usize,
    frames: usize,
}
impl VideoSink {
    fn new(settings: &ExportSettings) -> Result<Self, String> {
        let preset = &settings.io.preset;
        let video = &preset.video;
        let signal = preset.png;
        // The levels are fixed when the encoder opens: the mastering peak is the measured peak of
        // the view the frames render through, by the same rule as an HDR PNG's.
        let levels = signal.hdr().then(|| HdrLevels {
            peak_nits: video_io::export::hdr_scale(
                settings.light,
                &[],
                crate::color::BT2408_SDR_WHITE_NITS,
                signal,
            )
            .peak_nits,
            content: None,
        });
        let (width, height) = (preset.width, preset.height);
        let params = video_io::native::Params {
            video,
            signal,
            width: width as u32,
            height: height as u32,
            fps_num: settings.io.job.fps_num,
            fps_den: settings.io.job.fps_den,
            levels,
            audio: None,
        };
        let encoder = video_io::native::Encoder::open(
            &params,
            &settings.io.output(),
            settings.io.job.overwrite,
        )
        .map_err(|error| match video.encoder.filter(|e| e.info().hardware) {
            Some(gpu) => format!(
                "{} cannot encode {width}×{height}: {error}. Pick a CPU implementation or Auto.",
                gpu.info().label
            ),
            None => error.to_string(),
        })?;
        let n = width * height;
        Ok(Self {
            encoder,
            codec: video.codec,
            depth: video.depth,
            signal,
            rgb: if video.codec != Codec::Exv && video.depth == Depth::Eight {
                vec![0u8; n * 3]
            } else {
                Vec::new()
            },
            rgba16: if video.codec != Codec::Exv && video.depth == Depth::Ten {
                vec![0u16; n * 4]
            } else {
                Vec::new()
            },
            width,
            height,
            frames: 0,
        })
    }
    fn write(&mut self, frame: &Frame) -> Result<(), String> {
        if let Some(error) = &frame.colour_error {
            return Err(format!("Colour transform failed: {error}"));
        }
        if frame.light.len() != self.width * self.height {
            return Err("Linear display pixel buffer is incomplete".into());
        }
        if self.codec == Codec::Exv {
            self.encoder
                .push(video_io::native::Pixels::RgbaF32(bytemuck::cast_slice(
                    &frame.light,
                )))
                .map_err(|e| e.to_string())?;
            self.frames += 1;
            return Ok(());
        }
        if !self.signal.hdr() && frame.light_kind.hdr() {
            return Err("SDR video sink cannot encode HDR display codes".into());
        }
        match self.depth {
            Depth::Eight => {
                for (dst, p) in self.rgb.as_chunks_mut::<3>().0.iter_mut().zip(&frame.light) {
                    for c in 0..3 {
                        dst[c] = (crate::color::bt1886_code(p[c]) * 255.0 + 0.5) as u8;
                    }
                }
                self.encoder
                    .push(video_io::native::Pixels::Rgb24(&self.rgb))
                    .map_err(|e| e.to_string())?;
            }
            Depth::Ten => {
                let scale = self.signal.hdr().then(|| {
                    video_io::export::hdr_scale(
                        frame.light_kind,
                        &frame.light,
                        crate::color::BT2408_SDR_WHITE_NITS,
                        self.signal,
                    )
                });
                for (dst, p) in self
                    .rgba16
                    .as_chunks_mut::<4>()
                    .0
                    .iter_mut()
                    .zip(&frame.light)
                {
                    let [r, g, b] = match (self.signal, scale) {
                        (PngEncoding::Sdr8, _) => [p[0], p[1], p[2]].map(crate::color::bt1886_code),
                        (PngEncoding::Hdr10, Some(s)) => {
                            video_io::color::rec2020_nits([p[0], p[1], p[2]], s.unit_nits).map(
                                |v| video_io::color::pq(v.clamp(0.0, video_io::color::PQ_PEAK)),
                            )
                        }
                        (PngEncoding::Hlg, Some(s)) => video_io::color::hlg(
                            video_io::color::rec2020_nits([p[0], p[1], p[2]], s.unit_nits),
                            s.peak_nits,
                        ),
                        _ => [0.0; 3],
                    };
                    dst[0] = (r.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
                    dst[1] = (g.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
                    dst[2] = (b.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
                    dst[3] = (p[3].clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
                }
                self.encoder
                    .push(video_io::native::Pixels::Rgba64(&self.rgba16))
                    .map_err(|e| e.to_string())?;
            }
        }
        self.frames += 1;
        Ok(())
    }
    /// Finish the file with the frames pushed. A cancel, or a sink that got no frame, drops the
    /// encoder unfinished instead: video-io then publishes nothing (`native::Encoder`: an
    /// unfinished encoder never replaces the output). A cancel after frames goes through here
    /// only when the coordinator aborts the writer; a graceful stop closes the frame channel and
    /// finishes the partial movie.
    fn finish(self, cancel: &AtomicBool) -> Result<(), String> {
        if cancel.load(Ordering::Acquire) || self.frames == 0 {
            return Ok(());
        }
        self.encoder
            .finish_or_cancel(&|| cancel.load(Ordering::Acquire))
            .map(|_| ())
            .map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{Bulb, Formula, Kifs, KifsKind};
    use video_io::catalog::{Encoder, ExvProfile, PngVideo, ProResProfile, VideoSettings};
    use video_io::still::{Channels, ExrCompression, ExrDepth, TiffCompression};

    /// An export of `preset` for `job` into `dir`, rendered at `samples` per frame.
    fn export(preset: Preset, job: ExportJob, dir: PathBuf, samples: u32) -> ExportSettings {
        ExportSettings {
            io: video_io::job::ExportSettings { job, preset, dir },
            samples,
            ..Default::default()
        }
    }
    /// The Video format with `video`, at `width` x `height`.
    fn video(video: VideoSettings, width: usize, height: usize) -> Preset {
        Preset {
            format: ExportFormat::Video,
            video,
            width,
            height,
            ..Preset::default()
        }
    }
    /// HEVC on the CPU (Kvazaar): no GPU encoder needed.
    fn kvazaar() -> VideoSettings {
        VideoSettings {
            encoder: Some(Encoder::Kvazaar),
            ..VideoSettings::of(Codec::Hevc)
        }
    }

    #[test]
    fn fps_ui_values_preserve_broadcast_timing_and_reduce_integer_rates() {
        let mut job = ExportJob::default();
        for (fps, exact) in [
            (24.0, (24, 1)),
            (25.0, (25, 1)),
            (23.976, (24000, 1001)),
            (29.97, (30000, 1001)),
            (59.94, (60000, 1001)),
            (27.5, (55, 2)),
        ] {
            job.set_fps(fps);
            assert_eq!((job.fps_num, job.fps_den), exact);
            assert!((job.fps() - fps).abs() < 0.0005);
        }
        job.set_fps(f64::NAN);
        assert_eq!((job.fps_num, job.fps_den), (24, 1));
    }
    #[test]
    fn final_denoise_override_runs_once_and_preserves_world_settings() {
        let _gpu_test = crate::test_gpu::lock();
        let mut world = crate::scene::Scene::preset(Formula::Mandelbulb(Bulb::PRESET));
        world.render.denoise.interval = 8;
        world.render.denoise.enabled = false;
        let mut frame = world.clone();
        let settings = ExportSettings::new(
            ExportJob::default(),
            &OutputSettings {
                denoise_at_completion: true,
                ..OutputSettings::default()
            },
            32,
            1.0,
        );
        settings.apply_denoise_policy(&mut frame);
        let mut cadence = crate::denoise::State::default();
        let mut attempts = Vec::new();
        for samples in (4..=64).step_by(4) {
            if cadence.due(&frame.render.denoise, samples, samples == 64) {
                attempts.push(samples);
            }
        }
        assert_eq!(attempts, vec![64]);
        assert!(!cadence.due(&frame.render.denoise, 64, true));
        assert!(!world.render.denoise.enabled);
        assert_eq!(world.render.denoise.interval, 8);
        ExportSettings::default().apply_denoise_policy(&mut world);
        assert_eq!(world.render.denoise.interval, 8);
        assert!(!world.render.denoise.enabled);
    }

    fn frame(width: usize, height: usize) -> Frame {
        Frame {
            generation: 1,
            preview: false,
            width,
            height,
            pixels: vec![u32::from_le_bytes([180, 80, 20, 255]); width * height],
            light: vec![[0.5, 0.2, 0.05, 1.]; width * height],
            radiance: vec![[2., 0.5, 0.125, 1.]; width * height],
            light_kind: crate::color::DisplayLight::Relative,
            colour_error: None,
            denoised_samples: 0,
            denoise_ms: 0.0,
            denoise_error: None,
            samples: 4,
            active_tiles: (0, 0),
            converged: false,
            last_ms: 1.,
            last_spp: 4,
            unresolved: 0.0,
            sdr_bytes: Arc::new(Vec::new()),
            hdr_bytes: Arc::new(Vec::new()),
        }
    }

    /// One frame of the export panel over `controller`, with `events`, in a window large enough
    /// for every row.
    fn panel_frame(
        ctx: &egui::Context,
        controller: &mut ExportController,
        current: &Sel,
        events: Vec<egui::Event>,
    ) -> egui::FullOutput {
        ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1400.0, 1600.0),
                )),
                events,
                ..Default::default()
            },
            |root| {
                egui::CentralPanel::default().show(root, |ui| {
                    let mut output = controller.output.clone();
                    let mut actions = UiActions::default();
                    controller.panel(
                        ui,
                        Err("no colour config in this test"),
                        current,
                        &mut output,
                        &mut actions,
                    );
                    controller.output = output;
                });
            },
        )
    }
    /// Click the panel's button labelled `label` (a press frame, then a release frame).
    fn click(ctx: &egui::Context, controller: &mut ExportController, current: &Sel, label: &str) {
        let output = panel_frame(ctx, controller, current, vec![]);
        let pos = output
            .shapes
            .iter()
            .find_map(|s| match &s.shape {
                egui::epaint::Shape::Text(t) if t.galley.text() == label => {
                    Some(t.pos + t.galley.rect.center().to_vec2())
                }
                _ => None,
            })
            .unwrap_or_else(|| panic!("the panel shows no {label:?}"));
        for pressed in [true, false] {
            panel_frame(
                ctx,
                controller,
                current,
                vec![
                    egui::Event::PointerMoved(pos),
                    egui::Event::PointerButton {
                        pos,
                        button: egui::PointerButton::Primary,
                        pressed,
                        modifiers: Default::default(),
                    },
                ],
            );
        }
    }

    /// The panel edits the document's preset as it is (3.1): drawing it changes nothing, so the
    /// app authors no edit and no undo step; switching format or codec never rewrites another
    /// field: JPEG then PNG keeps alpha, ProRes then HEVC keeps the chosen encoder.
    #[test]
    fn the_panel_edits_the_recipe_in_place() {
        let ctx = egui::Context::default();
        let current = crate::color::default_selection();
        let mut controller = ExportController::default();
        let original = OutputSettings {
            preset: video(kvazaar(), 1920, 1080),
            ..OutputSettings::default()
        };
        controller.output = original.clone();
        for _ in 0..3 {
            panel_frame(&ctx, &mut controller, &current, vec![]);
        }
        assert_eq!(controller.output, original, "showing the panel is no edit");

        click(&ctx, &mut controller, &current, "ProRes");
        assert_eq!(controller.output.preset.video.codec, Codec::ProRes);
        click(&ctx, &mut controller, &current, "HEVC (H.265)");
        assert_eq!(
            controller.output.preset.video,
            kvazaar(),
            "HEVC's own settings come back"
        );

        click(&ctx, &mut controller, &current, "JPEG");
        assert_eq!(controller.output.preset.format, ExportFormat::Jpeg);
        click(&ctx, &mut controller, &current, "PNG");
        assert_eq!(controller.output.preset.format, ExportFormat::Png);
        assert_eq!(
            controller.output.preset.still.channels,
            Channels::Rgba,
            "alpha is kept"
        );
        assert_eq!(
            controller.output.preset.video,
            kvazaar(),
            "the video is untouched"
        );
        let edited = controller.output.clone();
        panel_frame(&ctx, &mut controller, &current, vec![]);
        assert_eq!(controller.output, edited);
    }

    /// The ffmpeg command of the PNG sequence's video (`ExportSettings::encode_video`'s).
    fn ffmpeg_args(
        s: &ExportSettings,
        levels: Option<HdrLevels>,
        output: &Path,
    ) -> Result<Vec<String>, String> {
        let input = s.io.video_input();
        let video =
            s.io.preset
                .png_video
                .settings(u32::from(s.io.preset.qp))
                .ok_or("PNG video export is off")?;
        video_io::cli::ffmpeg_args(&s.io.sequence(&input), &video, levels, output)
            .map_err(|e| e.to_string())?
            .into_iter()
            .map(|arg| {
                arg.into_string()
                    .map_err(|_| "Non-Unicode argument in export fixture".into())
            })
            .collect()
    }

    /// A PNG export's video carries the PNGs' colour description, and an HDR10 HEVC their
    /// mastering peak and the clip's content light level: ffprobe reads the container and the
    /// bitstream apart. The first frame is the dimmer one, so a container CLL lifted from it
    /// (ffmpeg's default) would be caught.
    #[test]
    #[ignore = "runs ffmpeg and ffprobe from PATH"]
    fn png_video_is_tagged_like_its_pngs() {
        let _gpu_test = crate::test_gpu::lock();
        let dir = temp_dir("png-video");
        let probe = |path: &Path, what: &str| {
            let out = std::process::Command::new("ffprobe")
                .args([
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    what,
                    "-read_intervals",
                    "%+#1",
                    "-of",
                    "default=nw=1",
                ])
                .arg(path)
                .output()
                .unwrap();
            String::from_utf8_lossy(&out.stdout).into_owned()
        };
        for (png, png_video, tags) in [
            (
                PngEncoding::Hdr10,
                PngVideo::Hevc,
                ["bt2020", "smpte2084", "bt2020nc"],
            ),
            (
                PngEncoding::Hdr10,
                PngVideo::ProRes,
                ["bt2020", "smpte2084", "bt2020nc"],
            ),
            (
                PngEncoding::Hlg,
                PngVideo::Hevc,
                ["bt2020", "arib-std-b67", "bt2020nc"],
            ),
            (
                PngEncoding::Sdr8,
                PngVideo::ProRes,
                ["bt709", "bt709", "bt709"],
            ),
        ] {
            let settings = export(
                Preset {
                    format: ExportFormat::Png,
                    png,
                    png_video,
                    width: 16,
                    height: 16,
                    ..Preset::default()
                },
                ExportJob {
                    name: format!("clip-{png:?}-{png_video:?}"),
                    first: 1,
                    last: 2,
                    overwrite: true,
                    ..ExportJob::default()
                },
                dir.clone(),
                4,
            );
            let writer = ExportWriter::spawn(settings.clone()).unwrap();
            for number in 1..=2 {
                let mut f = frame(16, 16);
                f.light = vec![[2.0 * number as f32, 1.0, 0.5, 1.0]; 16 * 16];
                f.light_kind = crate::color::DisplayLight::Absolute { peak_nits: 1000.0 };
                f.sdr_bytes = Arc::new(vec![128; 16 * 16 * 4]);
                writer
                    .frames
                    .as_ref()
                    .unwrap()
                    .send((number, Arc::new(f)))
                    .unwrap();
            }
            let levels = loop {
                match writer.events.recv_timeout(Duration::from_secs(30)).unwrap() {
                    WriteEvent::Written(_) => {}
                    WriteEvent::Finished(levels) => break levels,
                    WriteEvent::Failed(error) => panic!("{error}"),
                    WriteEvent::Cancelled => panic!("cancelled"),
                }
            };
            let output = settings.io.video_output().unwrap();
            let ticket = settings.io.encode_video(Backend::Cli, levels).unwrap();
            assert_eq!(ticket.wait().unwrap(), Outcome::Published);

            let stream = probe(&output, "-show_streams");
            let [primaries, transfer, matrix] = tags;
            for tag in [
                format!("color_primaries={primaries}"),
                format!("color_transfer={transfer}"),
                format!("color_space={matrix}"),
                "color_range=tv".into(),
            ] {
                assert!(
                    stream.contains(&tag),
                    "{png:?} {png_video:?} lacks {tag}:\n{stream}"
                );
            }
            if png_video == PngVideo::ProRes {
                assert!(stream.contains("profile=XQ"), "{stream}");
            }
            // An SDR video holds BT.1886 codes of the PNGs' light, not their sRGB codes.
            if png == PngEncoding::Sdr8 {
                let decoded = std::process::Command::new("ffmpeg")
                    .args(["-v", "error", "-i"])
                    .arg(&output)
                    .args([
                        "-frames:v",
                        "1",
                        "-f",
                        "rawvideo",
                        "-pix_fmt",
                        "rgb48le",
                        "-",
                    ])
                    .output()
                    .unwrap()
                    .stdout;
                let code = f32::from(u16::from_le_bytes([decoded[0], decoded[1]])) / 65535.0;
                let srgb = 128.0f32 / 255.0;
                let light = ((srgb + 0.055) / 1.055).powf(2.4);
                let expected = crate::color::bt1886_code(light);
                assert!(
                    (code - expected).abs() < 0.004,
                    "{code} vs BT.1886 {expected} (sRGB {srgb})"
                );
            }
            // The container never carries a first-frame CLL; an HDR10 HEVC's bitstream carries
            // the clip's (the brighter second frame's) and the mastering peak.
            assert!(
                !stream.contains("max_content="),
                "{png:?} {png_video:?} container CLL:\n{stream}"
            );
            if (png, png_video) == (PngEncoding::Hdr10, PngVideo::Hevc) {
                let content = levels.and_then(|l| l.content).unwrap();
                let brightest = soluwell_viewport::display::rec2020_nits([4.0, 1.0, 0.5], 100.0)
                    .into_iter()
                    .fold(0.0f32, f32::max);
                assert!(
                    (content.max_cll as f32 - brightest).abs() < brightest * 0.01,
                    "{content:?} vs {brightest}"
                );
                let frames = probe(&output, "-show_frames");
                assert!(stream.contains("pix_fmt=yuv420p10le"), "{stream}");
                for tag in [
                    "max_luminance=10000000/10000".to_owned(),
                    format!("max_content={}", content.max_cll.ceil()),
                ] {
                    assert!(frames.contains(&tag), "SEI lacks {tag}:\n{frames}");
                }
            }
        }
        finish_fixture(dir);
    }

    /// The ffmpeg command of every PNG encoding tags the video like its PNGs, and only an HDR10
    /// HEVC asks for HDR10 metadata (which then needs the clip's levels).
    #[test]
    fn png_video_arguments_follow_the_png_encoding() {
        let _gpu_test = crate::test_gpu::lock();
        let mut s = export(
            Preset {
                format: ExportFormat::Png,
                png_video: PngVideo::Hevc,
                ..Preset::default()
            },
            ExportJob {
                name: "shot".into(),
                first: 3,
                last: 9,
                ..ExportJob::default()
            },
            PathBuf::from("out"),
            32,
        );
        let out = Path::new("out/shot.mp4.part");
        for (png, tags) in [
            (PngEncoding::Sdr8, ["bt709", "bt709", "bt709"]),
            (PngEncoding::Hlg, ["bt2020", "arib-std-b67", "bt2020nc"]),
        ] {
            s.io.preset.png = png;
            let args = ffmpeg_args(&s, None, out).unwrap().join(" ");
            assert!(
                args.contains(&format!(
                    "setparams=color_primaries={}:color_trc={}:colorspace={}:range=tv",
                    tags[0], tags[1], tags[2]
                )),
                "{args}"
            );
            assert!(
                args.contains("sidedata=mode=delete:type=CONTENT_LIGHT_LEVEL"),
                "{args}"
            );
            assert!(!args.contains("master-display"), "{args}");
        }
        s.io.preset.png = PngEncoding::Hdr10;
        assert!(
            ffmpeg_args(&s, None, out).is_err(),
            "HDR10 metadata needs the measured levels"
        );
        let levels = HdrLevels {
            peak_nits: 500.0,
            content: Some(soluwell_viewport::display::screenshot::ContentLight {
                max_cll: 480.2,
                max_fall: 61.5,
            }),
        };
        let args = ffmpeg_args(&s, Some(levels), out).unwrap().join(" ");
        assert!(args.contains("-start_number 3 -i out"), "{args}");
        assert!(args.contains("shot.%06d.pq.png"), "{args}");
        assert!(args.contains("master-display=G(8500,39850)B(6550,2300)R(35400,14600)WP(15635,16450)L(5000000,1):max-cll=481,62"), "{args}");
        assert_eq!(
            s.io.video_output(),
            Some(Path::new("out").join("shot.pq.mp4"))
        );

        // A single frame is its file as it is: no pattern, no frame number, no `%` escaping.
        s.io.job.first = 5;
        s.io.job.last = 5;
        s.io.job.name = "50% grey".into();
        let args = ffmpeg_args(&s, Some(levels), out).unwrap();
        let input = args
            .iter()
            .position(|a| a == "-i")
            .map(|i| args[i + 1].clone())
            .unwrap();
        assert_eq!(Path::new(&input), Path::new("out").join("50% grey.pq.png"));
        assert!(
            args.windows(2).any(|w| w == ["-pattern_type", "none"]),
            "{args:?}"
        );
        assert!(!args.iter().any(|a| a == "-start_number"), "{args:?}");
    }

    /// The writer hands the HDR PNGs' levels to the video: the brightest mastering peak and the
    /// clip's MaxCLL / MaxFALL (CTA-861.3: maxima over the frames); an SDR export has none.
    #[test]
    fn png_writer_reports_the_clips_light_levels() {
        let _gpu_test = crate::test_gpu::lock();
        let dir = std::env::temp_dir().join(format!("frac-png-levels-{}", std::process::id()));
        for png in [PngEncoding::Hdr10, PngEncoding::Sdr8] {
            let settings = export(
                Preset {
                    format: ExportFormat::Png,
                    png,
                    width: 4,
                    height: 4,
                    ..Preset::default()
                },
                ExportJob {
                    first: 1,
                    last: 2,
                    overwrite: true,
                    ..ExportJob::default()
                },
                dir.join(format!("{png:?}")),
                4,
            );
            let writer = ExportWriter::spawn(settings).unwrap();
            for (number, (light, peak)) in [
                (1, ([4.0, 1.0, 0.5], 1000.0)),
                (2, ([1.0, 1.0, 1.0], 500.0)),
            ] {
                let mut f = frame(4, 4);
                f.light = vec![[light[0], light[1], light[2], 1.0]; 16];
                f.light_kind = crate::color::DisplayLight::Absolute { peak_nits: peak };
                f.sdr_bytes = Arc::new(vec![128; 16 * 4]);
                writer
                    .frames
                    .as_ref()
                    .unwrap()
                    .send((number, Arc::new(f)))
                    .unwrap();
            }
            let levels = loop {
                match writer.events.recv_timeout(Duration::from_secs(30)).unwrap() {
                    WriteEvent::Written(_) => {}
                    WriteEvent::Finished(levels) => break levels,
                    WriteEvent::Failed(error) => panic!("{error}"),
                    WriteEvent::Cancelled => panic!("cancelled"),
                }
            };
            if png == PngEncoding::Sdr8 {
                assert_eq!(levels, None);
                continue;
            }
            let levels = levels.expect("HDR PNGs record their levels");
            assert_eq!(
                levels.peak_nits, 1000.0,
                "the brighter frame's mastering peak"
            );
            let content = levels.content.unwrap();
            let max = |rgb: [f32; 3]| {
                soluwell_viewport::display::rec2020_nits(rgb, 100.0)
                    .into_iter()
                    .fold(0.0f32, f32::max)
            };
            let (bright, white) = (max([4.0, 1.0, 0.5]), max([1.0, 1.0, 1.0]));
            assert!(
                (content.max_cll as f32 - bright).abs() < bright * 0.01,
                "{content:?} vs {bright}"
            );
            // Every pixel of a frame alike: its average is its brightest channel; the clip's MaxFALL
            // is the brighter frame's, not the mean of both.
            assert!(
                (content.max_fall as f32 - bright).abs() < bright * 0.01,
                "{content:?}"
            );
            assert!(bright > white * 2.0);
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A 16-bit SDR still holds the same display encoding as the 8-bit codes of its frame: the
    /// built-in display's sRGB OETF of the light, or the output transform's encoder (3.5). A frame
    /// without light is an error, never 8-bit codes widened.
    #[test]
    fn sixteen_bit_codes_follow_the_display_encoding() {
        let f = frame(2, 2);
        let builtin = sdr_codes16(&f, None).unwrap();
        let srgb = |v: f32| (crate::color::oetf(v).clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
        assert_eq!(builtin[..4], [srgb(0.5), srgb(0.2), srgb(0.05), 65535]);
        let ocio = Ocio::load("ocio://studio-config-latest").unwrap();
        let names = ocio
            .resolve(
                &Sel {
                    display: "Display P3 - Display".into(),
                    view: "ACES 2.0 - SDR 100 nits (P3 D65)".into(),
                    ..crate::color::default_selection()
                },
                false,
            )
            .unwrap();
        let mut want = [[0.5f32, 0.2, 0.05, 1.0]];
        let mut light = want;
        ocio.transform(&names, false)
            .unwrap()
            .processor()
            .apply_rgba(&mut want);
        ocio.transform(&names, true)
            .unwrap()
            .processor()
            .apply_rgba(&mut light);
        let mut lit = frame(1, 1);
        lit.light = light.to_vec();
        let codes = sdr_codes16(&lit, Some(&ocio.encoder(&names).unwrap())).unwrap();
        for c in 0..3 {
            let code = f32::from(codes[c]) / 65535.0;
            assert!((code - want[0][c]).abs() < 1e-4, "{codes:?} vs {want:?}");
        }
        lit.light.clear();
        assert!(sdr_codes16(&lit, None).is_err());
    }

    /// Bind recipe `output` in the document, through the same edits the Render / Encode panel
    /// authors (`render_profiles::output_edits`).
    fn set_recipe(editor: &mut crate::world::WorldEditor, output: &OutputSettings) {
        let id = editor.document.output_settings_id().unwrap();
        let before = editor.document.output_settings().unwrap();
        editor
            .execute(crate::world::WorldCommand::Batch(
                crate::render_profiles::output_edits(id, &before, output).unwrap(),
            ))
            .unwrap();
        assert_eq!(&editor.document.output_settings().unwrap(), output);
    }
    fn temp_dir(name: &str) -> PathBuf {
        let root = std::env::var_os("WARP_BRO_VIDEO_FIXTURE")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        let path = root.join(format!(
            "frac-export-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&path).unwrap();
        path
    }
    fn finish_fixture(path: PathBuf) {
        if std::env::var_os("WARP_BRO_VIDEO_FIXTURE").is_some() {
            eprintln!("Retained export fixture: {}", path.display());
        } else {
            std::fs::remove_dir_all(path).unwrap();
        }
    }
    #[test]
    fn exr_sequence_writer_preserves_float_radiance() {
        let _gpu_test = crate::test_gpu::lock();
        let dir = temp_dir("exr");
        let settings = export(
            Preset {
                width: 2,
                height: 2,
                ..OutputSettings::default().preset
            },
            ExportJob {
                name: "seq".into(),
                first: 7,
                last: 8,
                ..ExportJob::default()
            },
            dir.clone(),
            4,
        );
        let writer = ExportWriter::spawn(settings.clone()).unwrap();
        for number in 7..=8 {
            writer
                .frames
                .as_ref()
                .unwrap()
                .send((number, Arc::new(frame(2, 2))))
                .unwrap();
            match writer.events.recv_timeout(Duration::from_secs(30)).unwrap() {
                WriteEvent::Written(n) => assert_eq!(n, number),
                WriteEvent::Failed(e) => panic!("{e}"),
                _ => panic!("unexpected event"),
            }
        }
        assert!(matches!(
            writer.events.recv_timeout(Duration::from_secs(30)).unwrap(),
            WriteEvent::Finished(_)
        ));
        let pixels = crate::exr_io::read_rgb(&settings.io.frame_path(7))
            .unwrap()
            .pixels;
        assert_eq!(pixels[0], [2., 0.5, 0.125]);
        assert!(settings.io.frame_path(8).exists());
        finish_fixture(dir);
    }
    #[test]
    fn hevc_mux_finishes_and_abort_never_publishes() {
        let _gpu_test = crate::test_gpu::lock();
        let dir = temp_dir("hevc");
        let mut settings = export(
            video(kvazaar(), 64, 64),
            ExportJob {
                name: "clip".into(),
                ..ExportJob::default()
            },
            dir.clone(),
            32,
        );
        let mut sink = VideoSink::new(&settings).unwrap();
        sink.write(&frame(64, 64)).unwrap();
        sink.write(&frame(64, 64)).unwrap();
        sink.finish(&AtomicBool::new(false)).unwrap();
        let bytes = std::fs::read(settings.io.output()).unwrap();
        for name in [b"ftyp", b"moov", b"hvcC", b"hvc1"] {
            assert!(bytes.windows(4).any(|b| b == name), "{:?}", name);
        }
        settings.io.job.name = "cancelled".into();
        let mut sink = VideoSink::new(&settings).unwrap();
        sink.write(&frame(64, 64)).unwrap();
        sink.finish(&AtomicBool::new(true)).unwrap();
        assert!(!settings.io.output().exists());
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1);
        finish_fixture(dir);
    }

    #[test]
    fn graceful_video_stop_drains_queued_frames_and_flushes_delayed_packets() {
        let _gpu_test = crate::test_gpu::lock();
        let dir = temp_dir("partial-hevc");
        for count in [0, 1, 9] {
            let settings = export(
                video(kvazaar(), 66, 50),
                ExportJob {
                    name: format!("partial-{count}"),
                    first: 17,
                    last: 100,
                    fps_num: 24000,
                    fps_den: 1001,
                    ..ExportJob::default()
                },
                dir.clone(),
                4,
            );
            let mut writer = ExportWriter::spawn(settings.clone()).unwrap();
            for number in 17..17 + count {
                writer
                    .frames
                    .as_ref()
                    .unwrap()
                    .send((number, Arc::new(frame(66, 50))))
                    .unwrap();
            }
            writer.frames.take();
            let mut written = 0;
            loop {
                match writer.events.recv_timeout(Duration::from_secs(30)).unwrap() {
                    WriteEvent::Written(number) => {
                        assert_eq!(number, 17 + written);
                        written += 1;
                    }
                    WriteEvent::Finished(_) => break,
                    WriteEvent::Failed(error) => panic!("{error}"),
                    WriteEvent::Cancelled => panic!("graceful stop was aborted"),
                }
            }
            assert_eq!(written, count);
            if count == 0 {
                assert!(!settings.io.output().exists());
            } else {
                let demux = av_format_mov::Demuxer::open(settings.io.output()).unwrap();
                assert_eq!(demux.sample_count(), count as usize);
                assert_eq!(demux.start_time(), 0);
                assert_eq!(demux.presented_frame_count().unwrap(), count as usize);
                assert_eq!(demux.presented_duration().unwrap(), count as u64 * 1001);
            }
        }
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 2);
        finish_fixture(dir);
    }

    #[test]
    #[ignore = "requires Vulkan Video HEVC hardware encode"]
    fn vulkan_hevc_partial_export_flushes_and_has_exact_timing() {
        let _gpu_test = crate::test_gpu::lock();
        let dir = temp_dir("vulkan-partial");
        let settings = export(
            video(
                VideoSettings {
                    encoder: Some(Encoder::HevcVulkan),
                    ..VideoSettings::of(Codec::Hevc)
                },
                256,
                256,
            ),
            ExportJob {
                name: "partial".into(),
                first: 0,
                last: 100,
                fps_num: 24000,
                fps_den: 1001,
                ..ExportJob::default()
            },
            dir.clone(),
            4,
        );
        let mut writer = ExportWriter::spawn(settings.clone()).unwrap();
        for number in 0..9 {
            writer
                .frames
                .as_ref()
                .unwrap()
                .send((number, Arc::new(frame(256, 256))))
                .unwrap_or_else(|_| {
                    match writer.events.try_iter().find_map(|event| {
                        if let WriteEvent::Failed(error) = event {
                            Some(error)
                        } else {
                            None
                        }
                    }) {
                        Some(error) => panic!("{error}"),
                        None => panic!("writer stopped"),
                    }
                });
        }
        writer.frames.take();
        let mut written = 0;
        loop {
            match writer.events.recv_timeout(Duration::from_secs(30)).unwrap() {
                WriteEvent::Written(number) => {
                    assert_eq!(number, written);
                    written += 1;
                }
                WriteEvent::Finished(_) => break,
                WriteEvent::Failed(error) => panic!("{error}"),
                WriteEvent::Cancelled => panic!("partial GPU movie aborted"),
            }
        }
        assert_eq!(written, 9);
        let demux = av_format_mov::Demuxer::open(settings.io.output()).unwrap();
        assert_eq!(demux.sample_count(), 9);
        assert_eq!(demux.start_time(), 0);
        assert_eq!(demux.presented_frame_count().unwrap(), 9);
        assert_eq!(demux.presented_duration().unwrap(), 9 * 1001);
        finish_fixture(dir);
    }

    /// Moving, detailed frames exercise inter prediction; constant swatches cannot
    /// expose stale references. Set WARP_BRO_VIDEO_FIXTURE to retain RGB oracles
    /// and movies for an independent decoder comparison.
    #[test]
    fn gpu_hevc_motion_fixture_encodes_every_source_frame() {
        let _gpu_test = crate::test_gpu::lock();
        use std::io::Write;
        let retained = std::env::var_os("WARP_BRO_VIDEO_FIXTURE");
        let dir = retained
            .as_ref()
            .map(PathBuf::from)
            .unwrap_or_else(|| temp_dir("motion"));
        std::fs::create_dir_all(&dir).unwrap();
        let mut reference = std::fs::File::create(dir.join("motion.rgb")).unwrap();
        let mut source = frame(256, 256);
        let encoders = [Encoder::Kvazaar, Encoder::HevcVulkan];
        let mut sinks: Vec<_> = encoders
            .into_iter()
            .map(|encoder| {
                let settings = export(
                    video(
                        VideoSettings {
                            encoder: Some(encoder),
                            ..VideoSettings::of(Codec::Hevc)
                        },
                        256,
                        256,
                    ),
                    ExportJob {
                        name: format!("motion-{encoder:?}"),
                        ..ExportJob::default()
                    },
                    dir.clone(),
                    32,
                );
                VideoSink::new(&settings).unwrap()
            })
            .collect();
        for number in 0..51 {
            let mut rgb = Vec::with_capacity(256 * 256 * 3);
            for y in 0..256 {
                for x in 0..256 {
                    let u = (x as f32 - 128.0) / 128.0;
                    let v = (y as f32 - 128.0) / 128.0;
                    let radius = (u * u + v * v).sqrt();
                    let wave = (radius * 55.0 + v.atan2(u) * 7.0 + number as f32 * 0.23).sin();
                    let values = [0.45 + wave * 0.4, 0.3 + wave * 0.25, 0.2 + wave * 0.15];
                    source.light[y * 256 + x] = [values[0], values[1], values[2], 1.0];
                    for value in values {
                        rgb.push((crate::color::bt1886_code(value) * 255.0 + 0.5) as u8);
                    }
                }
            }
            reference.write_all(&rgb).unwrap();
            for sink in &mut sinks {
                sink.write(&source).unwrap();
            }
        }
        for sink in sinks {
            sink.finish(&AtomicBool::new(false)).unwrap();
        }
        for encoder in encoders {
            let demux =
                av_format_mov::Demuxer::open(dir.join(format!("motion-{encoder:?}.mp4"))).unwrap();
            assert_eq!(demux.presented_frame_count().unwrap(), 51);
        }
        if retained.is_none() {
            finish_fixture(dir);
        }
    }

    #[test]
    fn reordered_hevc_clips_start_at_zero_with_exact_rational_frame_timing() {
        let _gpu_test = crate::test_gpu::lock();
        let dir = temp_dir("hevc-timing");
        for rate in [(24000, 1001), (25, 1)] {
            let settings = export(
                video(kvazaar(), 66, 50),
                ExportJob {
                    name: format!("clip-{}-{}", rate.0, rate.1),
                    fps_num: rate.0,
                    fps_den: rate.1,
                    ..ExportJob::default()
                },
                dir.clone(),
                32,
            );
            let mut sink = VideoSink::new(&settings).unwrap();
            for _ in 0..32 {
                sink.write(&frame(66, 50)).unwrap();
            }
            sink.finish(&AtomicBool::new(false)).unwrap();
            let mut demux = av_format_mov::Demuxer::open(settings.io.output()).unwrap();
            // movenc doubles a video track's timescale until it reaches 10000 when no
            // `video_track_timescale` is set (FFmpeg `movenc.c`, ffmpeg-rs `TrackClock::video`):
            // 24000/1001 keeps 24000, 25/1 becomes 12800. Frame timing stays exact, in ticks of
            // `scale` per source tick.
            let mut scale = 1u32;
            while rate.0 * scale < 10_000 {
                scale *= 2;
            }
            let tick = u64::from(rate.1) * u64::from(scale);
            assert_eq!(demux.sample_count(), 32);
            assert_eq!(demux.timescale(), rate.0 * scale);
            assert_eq!(demux.start_time(), 0);
            assert_eq!(demux.presented_duration().unwrap(), 32 * tick);
            assert_eq!(demux.presented_frame_count().unwrap(), 32);
            assert!(demux.edit_list().iter().all(|e| e.media_time >= 0));
            let mut pts: Vec<_> = (0..32).map(|i| demux.display_pts(i).unwrap()).collect();
            pts.sort_unstable();
            assert_eq!(pts, (0..32).map(|n| n * tick as i64).collect::<Vec<_>>());
            let colr = demux.color_info().expect("encoded MOV colour description");
            assert_eq!(
                (colr.primaries, colr.transfer, colr.matrix, colr.full_range),
                (1, 1, 1, Some(false)),
                "BT.709 limited container signalling"
            );
            // Decode without seeding colour from the container: the elementary
            // stream must carry the same description, not merely an outer tag.
            let mut decoder = av_codec_core::avcodec_alloc_context3();
            decoder.width = settings.io.preset.width as i32;
            decoder.height = settings.io.preset.height as i32;
            av_codec::avcodec_open_decoder(
                &mut decoder,
                av_codec_core::AVCodecID::Hevc,
                Some("none"),
                &demux.parameter_sets_annexb(),
            )
            .unwrap();
            let colour = av_codec_core::AVCodecColorMetadata::bt709_limited();
            let mut decoded_pts = Vec::new();
            let mut output = av_util_frame::av_frame_alloc();
            let mut receive = |decoder: &mut av_codec_core::AVCodecContext| loop {
                match decoder.avcodec_receive_frame(&mut output) {
                    Ok(()) => {
                        assert_eq!((output.width, output.height), (66, 50));
                        assert_eq!(output.format, av_util_pixfmt::AVPixelFormat::YUV420P as i32);
                        assert_eq!(
                            (
                                output.color_primaries,
                                output.color_trc,
                                output.colorspace,
                                output.color_range
                            ),
                            (
                                colour.primaries,
                                colour.transfer,
                                colour.matrix,
                                colour.range
                            ),
                            "native decoded colour description"
                        );
                        let luma = output.data[0].as_ref().expect("decoded luma plane").data();
                        let extent = 49 * output.linesize[0] as usize + 66;
                        assert!(luma.len() >= extent);
                        decoded_pts.push(output.pts);
                    }
                    Err(av_util_core::AvError::Posix(11)) | Err(av_util_core::AvError::Eof) => {
                        break;
                    }
                    Err(error) => panic!("native HEVC decode failed: {error:?}"),
                }
            };
            let mut packet = av_codec_core::AVPacket::new();
            for index in 0..32 {
                demux
                    .read_sample_into(index, &mut packet, index == 0)
                    .unwrap();
                packet.pts = demux.display_pts(index).unwrap();
                packet.duration = i64::from(rate.1);
                decoder.avcodec_send_packet(&packet).unwrap();
                receive(&mut decoder);
            }
            decoder
                .avcodec_send_packet(&av_codec_core::AVPacket::new())
                .unwrap();
            receive(&mut decoder);
            assert_eq!(
                decoded_pts, pts,
                "every encoded frame decodes at its exact PTS"
            );
        }
        finish_fixture(dir);
    }

    /// The scene's loaded colour config, as the app hands it to `start` (`ocio::State::config`).
    fn config_of(scene: &Scene) -> Arc<Ocio> {
        crate::ocio::shared(&crate::ocio::source(&scene.colour.config)).unwrap()
    }

    #[test]
    fn cuda_export_coordinator_samples_animation_and_writes_each_frame_once() {
        let _gpu_test = crate::test_gpu::lock();
        let dir = temp_dir("cuda");
        let service = RenderService::spawn(crate::render_service::Wake::default());
        let mut controller = ExportController {
            out_root: dir.clone(),
            job: ExportJob {
                first: 3,
                last: 4,
                ..ExportJob::default()
            },
            ..Default::default()
        };
        let mut scene = Scene::preset(Formula::Kifs(Kifs::preset(KifsKind::Tetrahedron)));
        scene.render.denoise.enabled = false; // This test checks exact physical animation output.
        scene.camera.target = [1000.0; 3];
        scene.sun.intensity = 0.0;
        scene.sky.intensity = 0.0;
        // The keyed procedural sky is the only light: no environment map.
        scene.environment.enabled = false;
        // Sky intensity 0 at frame 3, 2 at frame 4, keyed on the environment node.
        let mut editor =
            crate::world::WorldEditor::new(crate::world::WorldDocument::from_scene(&scene));
        let environment = editor
            .document
            .nodes()
            .into_iter()
            .find(|n| n.kind == crate::world::WorldKind::Environment)
            .unwrap()
            .id;
        for (frame, value) in [(3.0, 0.0), (4.0, 2.0)] {
            let path = "/sky/intensity".to_string();
            editor
                .execute(crate::world::WorldCommand::Key {
                    id: environment,
                    path: path.clone(),
                    frame,
                })
                .unwrap();
            editor
                .execute(crate::world::WorldCommand::SetAttribute {
                    id: environment,
                    path,
                    value: serde_json::json!(value),
                    frame,
                })
                .unwrap();
        }
        let output = editor.document.output_render_profile().unwrap();
        editor
            .execute(crate::world::WorldCommand::SetAttribute {
                id: output,
                path: "/quality/samples".into(),
                value: serde_json::json!(2),
                frame: 0.0,
            })
            .unwrap();
        set_recipe(
            &mut editor,
            &OutputSettings {
                preset: Preset {
                    width: 16,
                    height: 16,
                    ..OutputSettings::default().preset
                },
                ..OutputSettings::default()
            },
        );
        scene.document = Some(Box::new(editor.document));
        // Scene-linear EXR renders through no output transform: no colour config is needed.
        controller
            .start(&scene, &service, Err("no colour config".into()))
            .unwrap();
        let until = std::time::Instant::now() + Duration::from_secs(90);
        // Deliberately never poll GUI events or call update while the pipeline runs.
        while !controller.run.as_ref().unwrap().progress.read().done {
            assert!(
                std::time::Instant::now() < until,
                "autonomous export timed out"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
        controller.update();
        assert!(!controller.is_running());
        assert_eq!(controller.completed, 2, "{}", controller.status);
        assert!(
            controller.status.starts_with("Export complete"),
            "{}",
            controller.status
        );
        let written = controller.last.clone().unwrap();
        assert!(
            written.io.frame_path(3).starts_with(&dir),
            "exports land in a folder under the output root"
        );
        assert!(written.io.frame_path(3).exists());
        assert!(written.io.frame_path(4).exists());
        let pixel = |number| {
            crate::exr_io::read_rgb(&written.io.frame_path(number))
                .unwrap()
                .pixels[0]
        };
        assert_eq!(pixel(3)[0], 0.0);
        assert!(
            pixel(4)[0] > 0.0,
            "Each exported frame must evaluate its animation"
        );
        assert_eq!(
            std::fs::read_dir(written.io.frame_path(3).parent().unwrap())
                .unwrap()
                .count(),
            2
        );
        finish_fixture(dir);
    }
    #[test]
    fn cuda_cancel_controller_publishes_completed_movie_without_gui_waiting() {
        let _gpu_test = crate::test_gpu::lock();
        let dir = temp_dir("controller-cancel");
        let service = RenderService::spawn(crate::render_service::Wake::default());
        let mut controller = ExportController {
            out_root: dir.clone(),
            job: ExportJob {
                name: "partial".into(),
                last: 10000,
                ..ExportJob::default()
            },
            ..Default::default()
        };
        let mut scene = Scene::preset(Formula::Kifs(Kifs::preset(KifsKind::Tetrahedron)));
        scene.render.denoise.enabled = false;
        let mut editor =
            crate::world::WorldEditor::new(crate::world::WorldDocument::from_scene(&scene));
        // The job's sample target comes from the Output render profile, as in the application.
        let output = editor.document.output_render_profile().unwrap();
        editor
            .execute(crate::world::WorldCommand::SetAttribute {
                id: output,
                path: "/quality/samples".into(),
                value: serde_json::json!(1),
                frame: 0.0,
            })
            .unwrap();
        set_recipe(
            &mut editor,
            &OutputSettings {
                preset: video(
                    VideoSettings {
                        encoder: Some(Encoder::HevcVulkan),
                        ..VideoSettings::of(Codec::Hevc)
                    },
                    256,
                    256,
                ),
                ..OutputSettings::default()
            },
        );
        scene.document = Some(Box::new(editor.document));
        let ocio = config_of(&scene);
        controller
            .start(&scene, &service, Ok(ocio.as_ref()))
            .unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(90);
        while controller.completed == 0 {
            controller.update();
            assert!(controller.is_running(), "{}", controller.status);
            assert!(std::time::Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(5));
        }
        let started = std::time::Instant::now();
        controller.cancel();
        assert!(started.elapsed() < Duration::from_millis(100));
        while controller.is_running() {
            controller.update();
            assert!(std::time::Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(
            controller.status.starts_with("Export cancelled; saved"),
            "{}",
            controller.status
        );
        let demux =
            av_format_mov::Demuxer::open(controller.last.clone().unwrap().io.output()).unwrap();
        assert_eq!(
            demux.presented_frame_count().unwrap(),
            controller.completed as usize
        );
        assert!(controller.completed > 0 && controller.completed < 10001);
        finish_fixture(dir);
    }

    /// An export that writes nothing leaves no folder behind (3.8).
    #[test]
    fn an_empty_export_folder_is_removed_and_a_written_one_kept() {
        let dir = temp_dir("empty-folder");
        let empty = dir.join("empty");
        let written = dir.join("written");
        std::fs::create_dir_all(&empty).unwrap();
        std::fs::create_dir_all(&written).unwrap();
        std::fs::write(written.join("frame.exr"), "x").unwrap();
        remove_if_empty(&empty);
        remove_if_empty(&written);
        assert!(!empty.exists() && written.exists());
        finish_fixture(dir);
    }

    #[test]
    fn each_format_renders_through_its_own_output_transform() {
        let _gpu_test = crate::test_gpu::lock();
        let ocio = Ocio::load("ocio://studio-config-latest").unwrap();
        let scene = crate::color::default_selection(); // SDR sRGB view in the viewport
        let mut output = OutputSettings {
            preset: Preset {
                format: ExportFormat::Png,
                ..Preset::default()
            },
            ..OutputSettings::default()
        };
        let sdr = output.resolve_transform(&ocio, &scene).unwrap().unwrap();
        assert_eq!(
            (sdr.display.as_str(), sdr.view.as_str(), sdr.note),
            (scene.display.as_str(), scene.view.as_str(), None)
        );
        output.preset.png = PngEncoding::Hdr10;
        let pq = output.resolve_transform(&ocio, &scene).unwrap().unwrap();
        assert!(
            ocio.display_is(&pq.display, OutputKind::Pq) && pq.view.contains("1000 nits"),
            "{pq:?}"
        );
        assert!(
            pq.note.is_some(),
            "the SDR scene view was replaced for PQ: {pq:?}"
        );
        output.preset.png_peak_nits = 500.0;
        assert!(
            output
                .resolve_transform(&ocio, &scene)
                .unwrap()
                .unwrap()
                .view
                .contains("500 nits")
        );
        output.preset.png = PngEncoding::Hlg;
        let display = output
            .resolve_transform(&ocio, &scene)
            .unwrap()
            .unwrap()
            .display;
        assert!(display.contains("HLG"), "{display}");
        // An SDR-only format renders for SDR whatever the signal field says.
        output.preset.format = ExportFormat::Jpeg;
        assert_eq!(output.output_kind(), Some(OutputKind::Sdr));
        // EXR is scene-linear; an SDR file with OCIO off keeps the built-in display.
        output.preset.format = ExportFormat::Exr;
        assert_eq!(output.resolve_transform(&ocio, &scene).unwrap(), None);
        output.preset.format = ExportFormat::Video;
        output.preset.png = PngEncoding::Sdr8;
        let off = Sel {
            on: false,
            ..scene.clone()
        };
        assert_eq!(output.resolve_transform(&ocio, &off).unwrap(), None);
        // The frame scene gets the resolved display / view.
        let s = ExportSettings {
            transform: Some(OutputTransform {
                display: display.clone(),
                view: "view".into(),
                note: None,
            }),
            ..ExportSettings::default()
        };
        let mut frame = Scene::preset(Formula::Mandelbulb(Bulb::PRESET));
        frame.colour.on = false;
        s.apply_transform(&mut frame);
        assert!(frame.colour.on && frame.colour.display == display);
    }

    /// The colour an export fixes at start: the HDR view's measured peak is the video's
    /// mastering peak (3.6), a 16-bit SDR still gets the display's encoder (3.5), and a config
    /// that is not the scene's own is refused instead of used.
    #[test]
    fn start_fixes_the_views_light_and_the_sdr_encoder() {
        let current = crate::color::default_selection();
        let ocio = crate::ocio::shared(&crate::ocio::source(&current.config)).unwrap();
        let hdr = OutputSettings {
            preset: Preset {
                format: ExportFormat::Video,
                png: PngEncoding::Hdr10,
                png_peak_nits: 4000.0,
                ..Preset::default()
            },
            ..OutputSettings::default()
        };
        let mut s = ExportSettings::new(ExportJob::default(), &hdr, 4, 1.0);
        s.resolve_colour(&hdr, &current, Ok(ocio.as_ref())).unwrap();
        let crate::color::DisplayLight::Absolute { peak_nits } = s.light else {
            panic!("an HDR view has absolute light: {:?}", s.light);
        };
        assert!((peak_nits - 4000.0).abs() < 40.0, "{peak_nits}");
        assert!(s.encoder.is_none(), "no 16-bit SDR codes to make");
        let still = OutputSettings {
            preset: Preset {
                format: ExportFormat::Tiff,
                still: StillOptions {
                    sixteen_bit: true,
                    ..StillOptions::default()
                },
                ..Preset::default()
            },
            ..OutputSettings::default()
        };
        let mut s = ExportSettings::new(ExportJob::default(), &still, 4, 1.0);
        s.resolve_colour(&still, &current, Ok(ocio.as_ref()))
            .unwrap();
        assert!(s.encoder.is_some() && s.transform.is_some() && !s.light.hdr());
        let other = Ocio::load("ocio://default").unwrap();
        let mut s = ExportSettings::new(ExportJob::default(), &still, 4, 1.0);
        assert!(s.resolve_colour(&still, &current, Ok(&other)).is_err());
        // OCIO off: the built-in display, no config needed.
        let off = Sel {
            on: false,
            ..current.clone()
        };
        let mut s = ExportSettings::new(ExportJob::default(), &still, 4, 1.0);
        s.resolve_colour(&still, &off, Err("not loaded".into()))
            .unwrap();
        assert!(s.transform.is_none() && s.encoder.is_none());
    }

    #[test]
    fn sequence_paths_and_validation() {
        let _gpu_test = crate::test_gpu::lock();
        let mut s = export(
            OutputSettings::default().preset,
            ExportJob {
                name: "image".into(),
                first: 7,
                last: 9,
                ..ExportJob::default()
            },
            PathBuf::from("some folder"),
            32,
        );
        assert_eq!(s.io.job.frame_count(), 3);
        assert_eq!(
            s.io.frame_path(8),
            Path::new("some folder").join("image.000008.exr")
        );
        assert!(s.validate().is_ok(), "{:?}", s.validate());
        // An HDR PNG sequence numbers before the whole suffix (`fs_name::frame_file`).
        s.io.preset.format = ExportFormat::Png;
        s.io.preset.png = PngEncoding::Hdr10;
        s.io.job.name = "shot".into();
        s.io.resolve(Path::new("out"));
        assert_eq!(
            s.io.frame_path(8),
            Path::new("out").join("shot.000008.pq.png")
        );
        s.io.preset.png = PngEncoding::Sdr8;
        // A one-frame range is a still: the resolved file itself, no frame number.
        s.io.job.name = "still".into();
        s.io.job.first = 5;
        s.io.job.last = 5;
        assert_eq!(s.io.frame_path(5), Path::new("out").join("still.png"));
        assert!(s.validate().is_ok());
        s.io.preset.png_peak_nits = 50.0;
        assert!(s.validate().unwrap_err().contains("peak"));
        s.io.preset.png_peak_nits = 1000.0;
        s.io.job.name = "a/b".into();
        assert!(s.validate().unwrap_err().contains("cannot hold"));
        s.io.job.name = "CON".into();
        assert!(s.validate().unwrap_err().contains("device"));
        s.io.job.name = "frame".into();
        s.io.job.first = 7;
        s.io.job.last = 9;
        s.io.preset.format = ExportFormat::Video;
        s.io.preset.width = 17;
        assert!(s.validate().unwrap_err().contains("even"));
        s.io.preset.width = 64;
        s.io.preset.height = 64;
        s.io.job.last = 6;
        assert!(s.validate().is_err());
        // WarpBro's own limits: the renderer's frame size and samples, no pass-through.
        s.io.job.last = 9;
        s.io.preset.format = ExportFormat::Tga;
        s.samples = 0;
        assert!(s.validate().is_err());
        s.samples = 4;
        s.io.preset.width = 16384;
        s.io.preset.height = 16384;
        assert!(s.validate().unwrap_err().contains("64 megapixels"));
        s.io.preset.format = ExportFormat::ExrPassThrough;
        assert!(s.validate().is_err());
    }

    #[test]
    fn still_formats_and_exr_profiles_write_valid_files() {
        let _gpu_test = crate::test_gpu::lock();
        let dir = temp_dir("stills");
        for (format, ext) in [
            (ExportFormat::Jpeg, "jpg"),
            (ExportFormat::Tiff, "tif"),
            (ExportFormat::Tga, "tga"),
            (ExportFormat::Png, "png"),
            (ExportFormat::Exr, "exr"),
        ] {
            let settings = export(
                Preset {
                    format,
                    width: 8,
                    height: 8,
                    still: StillOptions {
                        sixteen_bit: true,
                        channels: Channels::Rgba,
                        jpeg_quality: 90,
                        tiff_compression: TiffCompression::Lzw,
                        exr_depth: ExrDepth::Half,
                        exr_compression: ExrCompression::Dwaa,
                        exr_dwa_level: 60.0,
                        ..StillOptions::default()
                    },
                    ..Preset::default()
                },
                ExportJob {
                    name: format!("still-{ext}"),
                    first: 1,
                    last: 1,
                    overwrite: true,
                    ..ExportJob::default()
                },
                dir.clone(),
                4,
            );
            assert_eq!(settings.io.preset.suffix(), ext);
            let writer = ExportWriter::spawn(settings.clone()).unwrap();
            let mut f = frame(8, 8);
            f.sdr_bytes = Arc::new(vec![180; 8 * 8 * 4]);
            writer
                .frames
                .as_ref()
                .unwrap()
                .send((1, Arc::new(f)))
                .unwrap();
            assert!(matches!(
                writer.events.recv_timeout(Duration::from_secs(30)).unwrap(),
                WriteEvent::Written(1)
            ));
            assert!(matches!(
                writer.events.recv_timeout(Duration::from_secs(30)).unwrap(),
                WriteEvent::Finished(_)
            ));
            let out = settings.io.output();
            assert!(out.exists(), "missing {}", out.display());
            assert!(std::fs::metadata(&out).unwrap().len() > 16);
            if format == ExportFormat::Exr {
                assert_eq!(
                    crate::exr_io::read_attr::<f32>(&out, "dwaCompressionLevel"),
                    Some(60.0)
                );
            }
        }
        finish_fixture(dir);
    }

    #[test]
    fn all_exv_and_prores_profiles_and_software_codecs_encode_cleanly() {
        let _gpu_test = crate::test_gpu::lock();
        let dir = temp_dir("video-profiles");
        let encode = |video_settings: VideoSettings, size: usize, name: String, suffix: &str| {
            let settings = export(
                video(video_settings, size, size),
                ExportJob {
                    name,
                    overwrite: true,
                    ..ExportJob::default()
                },
                dir.clone(),
                32,
            );
            assert_eq!(settings.io.preset.suffix(), suffix);
            settings.io.preset.validate().unwrap();
            let mut sink = VideoSink::new(&settings).unwrap();
            sink.write(&frame(size, size)).unwrap();
            sink.finish(&AtomicBool::new(false)).unwrap();
            assert!(settings.io.output().exists());
        };
        for profile in ExvProfile::ALL {
            encode(
                VideoSettings {
                    exv: profile,
                    ..VideoSettings::of(Codec::Exv)
                },
                16,
                format!("exv-{profile:?}"),
                "exv",
            );
        }
        for profile in ProResProfile::ALL {
            encode(
                VideoSettings {
                    prores: profile,
                    ..VideoSettings::of(Codec::ProRes)
                },
                16,
                format!("prores-{profile:?}"),
                "mov",
            );
        }
        for (codec, encoder) in [
            (Codec::H264, Encoder::OpenH264),
            (Codec::Av1, Encoder::Rav1e),
        ] {
            encode(
                VideoSettings {
                    encoder: Some(encoder),
                    ..VideoSettings::of(codec)
                },
                64,
                format!("sw-{codec:?}"),
                "mp4",
            );
        }
        finish_fixture(dir);
    }

    /// A recipe saved as a preset file and loaded back is the same recipe, every field of it
    /// (video-io-egui's preset files, the panel's Save / Load).
    #[test]
    fn preset_files_round_trip_and_backend_hints_are_exposed() {
        assert!(!Backend::Native.hint().is_empty());
        assert!(Backend::Cli.hint().contains("PNG sequence"));
        let dir = temp_dir("preset-files");
        let original = Preset {
            width: 3840,
            height: 2160,
            qp: 16,
            still: StillOptions {
                channels: Channels::Rgb,
                sixteen_bit: true,
                jpeg_quality: 92,
                tiff_compression: TiffCompression::Lzw,
                exr_depth: ExrDepth::Half,
                exr_compression: ExrCompression::Dwab,
                exr_dwa_level: 35.0,
                ..StillOptions::default()
            },
            ..video(
                VideoSettings {
                    exv: ExvProfile::Htj2kLossy,
                    prores: ProResProfile::P4444,
                    ..VideoSettings::of(Codec::Exv)
                },
                0,
                0,
            )
        };
        let mut saver = video_io_egui::ExportDialog::new("save", video_io_egui::Export::default());
        saver.export_mut().preset = original.clone();
        let path = dir.join("my_master_preset");
        assert_eq!(saver.save_preset_file(&path).unwrap(), "my_master_preset");
        let mut controller = ExportController::default();
        let loaded = controller
            .dialog
            .load_preset_file(&dir.join("my_master_preset.json"), None)
            .unwrap();
        assert_eq!(loaded, "my_master_preset");
        controller.output.preset = controller.dialog.export().preset.clone();
        assert_eq!(controller.output.preset, original);
        finish_fixture(dir);
    }
}
