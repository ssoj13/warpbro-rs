//! Event-driven export coordinator and bounded CPU writer. CUDA never runs on the UI thread.
use crate::{
    render_service::{Command, Frame, PngEncoding, RenderEvent, RenderPort, RenderService},
    scene::Scene,
};
use serde::{Deserialize, Serialize};
use std::{
    fs::File,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, SyncSender, TrySendError},
    },
    time::Duration,
};

const HIGH_QUALITY_QP: u8 = 18;
/// The HDR PNG export's default target peak (`ExportSettings::png_peak_nits`).
const DEFAULT_PNG_PEAK_NITS: f32 = 1000.0;
const HIGH_QUALITY_PRESET: usize = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExportFormat {
    Exr,
    Hevc,
    Png,
}
impl ExportFormat {
    /// The format tabs of the panel, in order.
    pub const ALL: [Self; 3] = [Self::Exr, Self::Png, Self::Hevc];
    fn label(self) -> &'static str {
        match self {
            Self::Exr => "EXR sequence",
            Self::Png => "PNG",
            Self::Hevc => "Video",
        }
    }
    /// What a file of this format holds, under its options.
    fn hint(self) -> &'static str {
        match self {
            Self::Exr => "Scene-linear ACEScg (AP1-tagged); no display transform or exposure baked in.",
            Self::Png => "The monitor rendering baked in: SDR 8-bit sRGB / BT.709, or HDR10 / HLG 16-bit BT.2020 with cICP. HDR keeps SDR white at 100 nits; pick an HDR view for HDR highlights.",
            Self::Hevc => "HEVC 8-bit sRGB / Rec.709 SDR; display transform baked in. Cancel saves completed frames.",
        }
    }
    pub fn extension(self) -> &'static str {
        match self {
            Self::Exr => "exr",
            Self::Png => "png",
            Self::Hevc => "mp4",
        }
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum VideoEncoder {
    #[default]
    Vulkan,
    Kvazaar,
}
impl VideoEncoder {
    fn label(self) -> &'static str {
        match self {
            Self::Vulkan => "GPU · Vulkan Video",
            Self::Kvazaar => "CPU · Kvazaar (I-frames)",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ExportSettings {
    pub format: ExportFormat,
    pub encoder: VideoEncoder,
    /// File name stem; every export writes into a new `~/.warpbro/out/<timestamp>` folder.
    pub name: String,
    /// The folder this export writes into (`resolve`: a new dated one per export). The files are
    /// derived from it, `name` and the format's suffix only (`output`, `frame_path`).
    #[serde(skip)]
    pub dir: PathBuf,
    pub width: usize,
    pub height: usize,
    pub samples: u32,
    pub first: u32,
    pub last: u32,
    pub fps_num: u32,
    pub fps_den: u32,
    pub qp: u8,
    /// Retained for reading older settings. Unsafe inter presets are disabled.
    pub preset: usize,
    pub overwrite: bool,
    /// Override world cadence for offline renders: OIDN runs once at the final sample.
    pub denoise_at_completion: bool,
    pub png: PngEncoding,
    /// The HDR view an HDR PNG export renders through when the scene's own view is not an HDR
    /// view of that kind: the one whose measured peak is nearest this (`Ocio::output_transform`).
    /// The file records the rendered view's measured peak (`render_service::hdr_scale`).
    pub png_peak_nits: f32,
    /// OCIO display / view overriding the automatic output transform; empty = automatic.
    pub output_display: String,
    pub output_view: String,
    /// The display / view this export renders through (`resolve_transform`); None = scene-linear
    /// (EXR) or the scene's own built-in SDR display.
    #[serde(skip)]
    pub transform: Option<(String, String)>,
}
impl Default for ExportSettings {
    fn default() -> Self {
        Self {
            format: ExportFormat::Exr,
            encoder: VideoEncoder::Vulkan,
            name: "frame".into(),
            dir: PathBuf::new(),
            width: 1920,
            height: 1080,
            samples: 256,
            first: 1,
            last: 1,
            fps_num: 24,
            fps_den: 1,
            qp: HIGH_QUALITY_QP,
            preset: HIGH_QUALITY_PRESET,
            overwrite: false,
            denoise_at_completion: false,
            png: PngEncoding::Sdr8,
            png_peak_nits: DEFAULT_PNG_PEAK_NITS,
            output_display: String::new(),
            output_view: String::new(),
            transform: None,
        }
    }
}
impl ExportSettings {
    pub fn fps(&self) -> f64 {
        self.fps_num as f64 / self.fps_den.max(1) as f64
    }
    /// UI values use ordinary frames/second. Recognize broadcast rates before
    /// reducing the custom millisecond fraction, preserving exact encoder timing.
    pub fn set_fps(&mut self, fps: f64) {
        let fps = if fps.is_finite() {
            fps.clamp(1.0, 240.0)
        } else {
            24.0
        };
        for (num, den) in [(24000, 1001), (30000, 1001), (60000, 1001), (120000, 1001)] {
            if (fps - num as f64 / den as f64).abs() < 0.0005 {
                self.fps_num = num;
                self.fps_den = den;
                return;
            }
        }
        let num = (fps * 1000.0).round() as u32;
        let (mut a, mut b) = (num, 1000);
        while b != 0 {
            (a, b) = (b, a % b);
        }
        self.fps_num = num / a;
        self.fps_den = 1000 / a;
    }

    /// Apply after animation evaluation so authored denoise keys cannot restore periodic passes.
    pub fn apply_denoise_policy(&self, scene: &mut Scene) {
        if self.denoise_at_completion {
            scene.render.denoise.enabled = true;
            scene.render.denoise.interval = 0;
        }
    }
    /// The display a file of this format is encoded for; None for scene-linear EXR.
    pub fn output_kind(&self) -> Option<crate::ocio::OutputKind> {
        use crate::ocio::OutputKind;
        match self.format {
            ExportFormat::Exr => None,
            ExportFormat::Hevc => Some(OutputKind::Sdr),
            ExportFormat::Png => Some(match self.png {
                PngEncoding::Sdr8 => OutputKind::Sdr,
                PngEncoding::Hdr10 => OutputKind::Pq,
                PngEncoding::Hlg => OutputKind::Hlg,
            }),
        }
    }

    /// The output transform from ACEScg for this format, independent of the viewport: the
    /// override when set, else `Ocio::output_transform`. An SDR output with OCIO off keeps the
    /// built-in display (None); HDR always needs an OCIO HDR view.
    pub fn resolve_transform(&self, ocio: &crate::ocio::Ocio, current: &crate::ocio::Sel) -> Result<Option<(String, String)>, String> {
        let Some(kind) = self.output_kind() else {
            return Ok(None);
        };
        if kind == crate::ocio::OutputKind::Sdr && !current.on {
            return Ok(None);
        }
        if !self.output_display.is_empty() {
            let picked = crate::ocio::Sel {
                display: self.output_display.clone(),
                view: self.output_view.clone(),
                ..current.clone()
            };
            let names = ocio.resolve(&picked, kind != crate::ocio::OutputKind::Sdr).map_err(|e| e.to_string())?;
            return Ok(Some((names.display, names.view)));
        }
        ocio.output_transform(current, kind, self.png_peak_nits)
            .map(Some)
            .map_err(|e| e.to_string())
    }
    /// The frame scene rendered through this export's output transform. An OCIO output replaces
    /// the legacy Reinhard curve, which would otherwise bypass OCIO (`ColorPipeline::apply`) and
    /// write an HDR file of SDR-relative light.
    pub fn apply_transform(&self, scene: &mut Scene) {
        if let Some((display, view)) = &self.transform {
            scene.colour.on = true;
            scene.colour.display = display.clone();
            scene.colour.view = view.clone();
            scene.render.reinhard = false;
        }
    }

    /// The suffix after the file stem: a PNG names its transfer (`PngEncoding::suffix`), so an
    /// HDR PNG is not taken for an SDR one.
    fn suffix(&self) -> &'static str {
        match self.format {
            ExportFormat::Png => self.png.suffix(),
            format => format.extension(),
        }
    }
    /// Write into `dir`.
    pub fn resolve(&mut self, dir: &Path) {
        self.dir = dir.to_path_buf();
    }
    /// The file of a single frame or a video: `dir/<name>.<suffix>` (`fs_name::frame_file`).
    pub fn output(&self) -> PathBuf {
        self.dir.join(crate::fs_name::frame_file(self.name.trim(), None, self.suffix()))
    }
    pub fn validate(&self) -> Result<(), String> {
        crate::fs_name::check(self.name.trim())?;
        if !(100.0..=10000.0).contains(&self.png_peak_nits) {
            return Err("HDR peak must be 100–10000 nits".into());
        }
        if self.width == 0 || self.height == 0 || self.width > 16384 || self.height > 16384 {
            return Err("Resolution must be 1–16384 pixels per axis".into());
        }
        if self
            .width
            .checked_mul(self.height)
            .is_none_or(|n| n > 67_108_864)
        {
            return Err("Resolution exceeds the 64 megapixel renderer limit".into());
        }
        if self.samples == 0 || self.samples > 1_000_000 {
            return Err("Samples must be 1–1000000".into());
        }
        if self.first > self.last || self.last - self.first > 100_000 {
            return Err("Frame range must be ordered and contain at most 100001 frames".into());
        }
        if self.dir.as_os_str().is_empty() {
            return Err("The output folder is not resolved".into());
        }
        if self.fps_num == 0 || self.fps_den == 0 {
            return Err("FPS numerator and denominator must be positive".into());
        }
        if self.qp > 51 {
            return Err("Invalid encoder quality".into());
        }
        // The suffix follows the format (`suffix`), so only the encoder's own limits remain.
        if self.format == ExportFormat::Hevc
            && (!self.width.is_multiple_of(2) || !self.height.is_multiple_of(2))
        {
            return Err("HEVC 4:2:0 requires even width and height".into());
        }
        Ok(())
    }
    pub fn frame_count(&self) -> u32 {
        self.last.saturating_sub(self.first).saturating_add(1)
    }
    /// The file of frame `number`: `output` for a single frame, else `name.000042.<suffix>` (the
    /// number before the whole suffix, as in every writer).
    pub fn frame_path(&self, number: u32) -> PathBuf {
        if self.first == self.last {
            return self.output();
        }
        self.dir.join(crate::fs_name::frame_file(self.name.trim(), Some(number), self.suffix()))
    }
}

pub struct ExportController {
    pub settings: ExportSettings,
    pub status: String,
    pub completed: u32,
    pub sample_progress: (u32, u32),
    run: Option<Run>,
    next_id: u64,
    /// Root of the per-export folders (`~/.warpbro/out`).
    pub out_root: PathBuf,
    /// The resolved settings of the last started export: where its files went.
    pub last: Option<ExportSettings>,
}
#[derive(Clone, Default)]
struct Progress {
    status: String,
    completed: u32,
    samples: (u32, u32),
    done: bool,
}
struct Run {
    settings: ExportSettings,
    progress: Arc<Mutex<Progress>>,
    cancel: Arc<AtomicBool>,
}
impl Drop for Run {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Release);
    }
}
impl Default for ExportController {
    fn default() -> Self {
        Self {
            settings: ExportSettings::default(),
            status: String::new(),
            completed: 0,
            sample_progress: (0, 0),
            run: None,
            next_id: 100,
            out_root: crate::out_root(),
            last: None,
        }
    }
}
impl ExportController {
    pub fn settings(&self) -> &ExportSettings {
        &self.settings
    }
    pub fn restore(&mut self, settings: ExportSettings) {
        self.settings = settings;
    }
    pub fn is_running(&self) -> bool {
        self.run.is_some()
    }
    /// Sequence events use a dedicated bus. This hook keeps the host's event dispatch uniform.
    pub fn handle(&mut self, _event: &RenderEvent, _service: &RenderService) {}
    pub fn update(&mut self, _service: &RenderService) {
        let snapshot = self
            .run
            .as_ref()
            .and_then(|run| run.progress.try_lock().ok().map(|p| p.clone()));
        if let Some(progress) = snapshot {
            self.status = progress.status;
            self.completed = progress.completed;
            self.sample_progress = progress.samples;
            if progress.done {
                self.run = None;
            }
        }
    }
    pub fn start(&mut self, scene: &Scene, service: &RenderService) -> Result<(), String> {
        if self.run.is_some() {
            return Err("An export is already running".into());
        }
        let mut settings = self.settings.clone();
        // Validate against the output root first so a rejected export leaves no empty folder.
        settings.resolve(&self.out_root);
        settings.validate()?;
        settings.transform = match settings.output_kind() {
            Some(_) => {
                let ocio = crate::ocio::Ocio::load(&crate::ocio::source(&scene.colour.config)).map_err(|e| e.to_string())?;
                settings.resolve_transform(&ocio, &scene.colour)?
            }
            None => None,
        };
        settings.resolve(&crate::new_out_dir(&self.out_root)?);
        let scene = scene.clone();
        let port = service.port();
        let id = self.next_id;
        self.next_id += u64::from(settings.frame_count()) + 1;
        let progress = Arc::new(Mutex::new(Progress {
            status: "Starting export".into(),
            samples: (0, settings.samples),
            ..Default::default()
        }));
        let cancel = Arc::new(AtomicBool::new(false));
        let shared = progress.clone();
        let stop = cancel.clone();
        let config = settings.clone();
        std::thread::Builder::new()
            .name("frac-export-coordinator".into())
            .spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    coordinate_export(&config, &scene, &port, id, &shared, &stop)
                }));
                let status = match result {
                    Ok(Ok(true)) => format!("Export complete: {}", config.frame_path(config.first).parent().unwrap_or(Path::new("")).display()),
                    Ok(Ok(false)) => {
                        let completed = shared.lock().unwrap_or_else(|e| e.into_inner()).completed;
                        if config.format == ExportFormat::Hevc && completed > 0 {
                            format!(
                                "Export cancelled; saved {completed} frames to {}",
                                config.output().display()
                            )
                        } else if config.format == ExportFormat::Hevc {
                            "Export cancelled before any complete frames".into()
                        } else {
                            format!("Export cancelled; {completed} frames retained")
                        }
                    }
                    Ok(Err(error)) => format!("Export failed: {error}"),
                    Err(_) => "Export failed: coordinator panicked".to_string(),
                };
                let mut state = shared.lock().unwrap_or_else(|e| e.into_inner());
                state.status = status;
                state.done = true;
            })
            .map_err(|e| e.to_string())?;
        self.completed = 0;
        self.sample_progress = (0, settings.samples);
        self.status = "Starting export".into();
        self.last = Some(settings.clone());
        self.run = Some(Run {
            settings,
            progress,
            cancel,
        });
        Ok(())
    }
    pub fn cancel(&mut self, _service: &RenderService) {
        if let Some(run) = &self.run {
            run.cancel.store(true, Ordering::Release);
            self.status = "Cancelling export…".into();
        }
    }
    /// `timeline` is (first, last, fps, playhead) of the world document.
    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        timeline: (u32, u32, f64, u32),
        service: &RenderService,
        ocio: Option<&crate::ocio::Ocio>,
        current: &crate::ocio::Sel,
        freeze: impl FnOnce() -> Scene,
    ) {
        ui.heading("Render / Encode");
        let running = self.is_running();
        let kind_before = self.settings.output_kind();
        ui.add_enabled_ui(!running, |ui| {
            ui.weak(format!("Written to {}", self.out_root.join("<date_time>").display()));
            ui.weak("Animation is sampled at each frame. The scene and keys are frozen when export starts.");
            // One grid, so every value column lines up: the settings every format shares, drawn
            // once, then the format row (the tabs) and only that format's options under it.
            egui::Grid::new("render_encode").num_columns(2).show(ui, |ui| {
                ui.label("Name"); ui.horizontal(|ui| {
                    ui.text_edit_singleline(&mut self.settings.name);
                    ui.label(format!(".{}", self.settings.suffix()));
                }); ui.end_row();
                ui.label("Resolution"); ui.horizontal(|ui| { ui.add(egui::DragValue::new(&mut self.settings.width).range(1..=16384)); ui.label("×"); ui.add(egui::DragValue::new(&mut self.settings.height).range(1..=16384)); }); ui.end_row();
                ui.label("Samples / frame"); ui.add(egui::DragValue::new(&mut self.settings.samples).range(1..=1_000_000)); ui.end_row();
                ui.label("Denoise"); ui.checkbox(&mut self.settings.denoise_at_completion, "Once at completion").on_hover_text("Run OIDN once after all samples of each exported frame; override World Settings cadence."); ui.end_row();
                ui.label("Frame range"); ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut self.settings.first).range(0..=u32::MAX)); ui.label("…"); ui.add(egui::DragValue::new(&mut self.settings.last).range(0..=u32::MAX));
                    if ui.button("Current frame").on_hover_text("Render only the frame under the playhead.").clicked() {
                        self.settings.first = timeline.3;
                        self.settings.last = timeline.3;
                    }
                    if ui.button("Timeline").on_hover_text("The timeline's range, and its FPS for video.").clicked() {
                        self.settings.first = timeline.0;
                        self.settings.last = timeline.1;
                        self.settings.set_fps(timeline.2);
                    }
                }); ui.end_row();
                ui.label("Format"); ui.horizontal(|ui| {
                    for format in ExportFormat::ALL {
                        ui.selectable_value(&mut self.settings.format, format, format.label());
                    }
                }); ui.end_row();
                if let (Some(kind), Some(ocio)) = (self.settings.output_kind(), ocio) {
                    self.output_transform_ui(ui, ocio, current, kind);
                }
                if self.settings.format == ExportFormat::Png {
                    ui.label("Encoding");
                    egui::ComboBox::from_id_salt("png_encoding").selected_text(self.settings.png.label()).show_ui(ui, |ui| {
                        for encoding in PngEncoding::ALL {
                            ui.selectable_value(&mut self.settings.png, encoding, encoding.label());
                        }
                    }); ui.end_row();
                    if self.settings.png.hdr() {
                        ui.label("HDR peak");
                        ui.add(egui::DragValue::new(&mut self.settings.png_peak_nits).range(100.0..=10000.0).suffix(" nits"))
                            .on_hover_text("When the scene's view is not an HDR view of this kind, the HDR view whose measured peak is nearest this. The file records the rendered view's measured peak (mDCV, HLG system gamma).");
                        ui.end_row();
                    }
                }
                if self.settings.format == ExportFormat::Hevc {
                    ui.label("Encoder");
                    egui::ComboBox::from_id_salt("video_encoder").selected_text(self.settings.encoder.label()).show_ui(ui, |ui| {
                        for encoder in [VideoEncoder::Vulkan, VideoEncoder::Kvazaar] {
                            ui.selectable_value(&mut self.settings.encoder, encoder, encoder.label());
                        }
                    }); ui.end_row();
                    ui.label("FPS"); ui.horizontal(|ui| {
                        let mut fps = self.settings.fps();
                        if ui.add(egui::DragValue::new(&mut fps).speed(0.1).range(1.0..=240.0).max_decimals(3)).changed() {
                            self.settings.set_fps(fps);
                        }
                        egui::ComboBox::from_id_salt("export_fps_presets").selected_text("Presets").show_ui(ui, |ui| {
                            for rate in [23.976, 24.0, 25.0, 29.97, 30.0, 50.0, 59.94, 60.0, 120.0] {
                                if ui.selectable_label((self.settings.fps() - rate).abs() < 0.001, rate.to_string()).clicked() {
                                    self.settings.set_fps(rate); ui.close();
                                }
                            }
                        });
                    }); ui.end_row();
                    ui.label("Quality"); ui.horizontal(|ui| {
                        ui.add(egui::Slider::new(&mut self.settings.qp,0..=51).text("QP"))
                            .on_hover_text("Lower QP preserves more detail and produces larger files.");
                        if ui.button("High quality").clicked() {
                            self.settings.qp = HIGH_QUALITY_QP;
                            self.settings.preset = HIGH_QUALITY_PRESET;
                        }
                    }); ui.end_row();
                    if self.settings.encoder == VideoEncoder::Kvazaar {
                        ui.label("Prediction"); ui.label("Independent I-frames"); ui.end_row();
                    }
                }
            });
            // An override names a display of one kind (SDR / PQ / HLG): a new kind starts automatic.
            if self.settings.output_kind() != kind_before {
                self.settings.output_display.clear();
                self.settings.output_view.clear();
            }
            ui.small(self.settings.format.hint());
            let validation = {
                let mut preview = self.settings.clone();
                preview.resolve(&self.out_root);
                preview.validate()
            };
            if let Err(error) = &validation { ui.colored_label(egui::Color32::LIGHT_RED,error); }
            if ui.add_enabled(validation.is_ok(),egui::Button::new("Start render")).clicked() {
                let scene = freeze();
                if let Err(error) = self.start(&scene,service) { self.status = error; }
            }
        });
        if running {
            let total = self
                .run
                .as_ref()
                .map(|r| r.settings.frame_count())
                .unwrap_or(1);
            ui.add(
                egui::ProgressBar::new(self.completed as f32 / total as f32)
                    .text(format!("{} / {total} frames written", self.completed)),
            );
            if ui.button("Cancel").clicked() {
                self.cancel(service);
            }
        }
        ui.label(&self.status);
    }
    /// "Output transform": what this export renders through from ACEScg, automatic by default.
    fn output_transform_ui(
        &mut self,
        ui: &mut egui::Ui,
        ocio: &crate::ocio::Ocio,
        current: &crate::ocio::Sel,
        kind: crate::ocio::OutputKind,
    ) {
        let hdr = kind != crate::ocio::OutputKind::Sdr;
        let resolved = self.settings.resolve_transform(ocio, current);
        ui.label("Output transform").on_hover_text(
            "The OCIO display / view this file is rendered through from ACEScg, independent of the viewport. Automatic: the scene's own view when it fits the format, else the config's first fitting display (PQ / HLG by name) and, for HDR, the view whose measured peak is nearest the HDR peak.",
        );
        ui.vertical(|ui| {
            match &resolved {
                Ok(Some((display, view))) => ui.label(format!("{display} · {view}")),
                Ok(None) => ui.label("Built-in display (OCIO off)"),
                Err(error) => ui.colored_label(egui::Color32::LIGHT_RED, error),
            };
            ui.horizontal(|ui| {
                let owned = |v: Vec<&str>| v.into_iter().map(str::to_owned).collect::<Vec<_>>();
                let displays = owned(ocio.displays(hdr).into_iter().filter(|d| ocio.display_is(d, kind)).collect());
                let before = self.settings.output_display.clone();
                crate::ocio::combo(ui, "export.output_display", &mut self.settings.output_display, &displays, "auto");
                if self.settings.output_display != before {
                    self.settings.output_view.clear();
                }
                let display = if self.settings.output_display.is_empty() {
                    resolved.as_ref().ok().and_then(Option::as_ref).map(|(d, _)| d.clone()).unwrap_or_default()
                } else {
                    self.settings.output_display.clone()
                };
                let views = owned(
                    ocio.views(&display, hdr)
                        .into_iter()
                        .filter(|v| ocio.view_is(&display, v, kind))
                        .collect(),
                );
                ui.add_enabled_ui(!self.settings.output_display.is_empty(), |ui| {
                    crate::ocio::combo(ui, "export.output_view", &mut self.settings.output_view, &views, "first");
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
    progress: &Mutex<Progress>,
    cancel: &AtomicBool,
) -> Result<bool, String> {
    let mut writer = ExportWriter::spawn(settings.clone())?;
    let (reply, events) = mpsc::sync_channel(16);
    let result = (|| -> Result<bool, String> {
        for number in settings.first..=settings.last {
            let mut frame_scene = scene.evaluated(f64::from(number))?;
            settings.apply_denoise_policy(&mut frame_scene);
            settings.apply_transform(&mut frame_scene);
            let id = first_id + u64::from(number - settings.first);
            loop {
                if cancel.load(Ordering::Acquire) {
                    return Ok(false);
                }
                match port.try_command(Command::RenderExport {
                    id,
                    scene: frame_scene.clone(),
                    width: settings.width,
                    height: settings.height,
                    spp: settings.samples,
                    reply: Some(reply.clone()),
                }) {
                    Ok(()) => break,
                    Err(error) if error.contains("queue is full") => {
                        std::thread::sleep(Duration::from_millis(5))
                    }
                    Err(error) => return Err(error),
                }
            }
            let frame = loop {
                if cancel.load(Ordering::Acquire) {
                    return Ok(false);
                }
                if !port.is_available() {
                    return Err("CUDA worker stopped".into());
                }
                match writer.events.try_recv() {
                    Ok(WriteEvent::Written(written)) => {
                        acknowledge(progress, settings.first, written)?
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
                        let mut state = progress.lock().unwrap_or_else(|e| e.into_inner());
                        state.status = format!("Frame {number}: {samples}/{total} samples");
                        state.samples = (samples, total);
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
            {
                let mut state = progress.lock().unwrap_or_else(|e| e.into_inner());
                state.status = format!("Writing frame {number}");
                state.samples = (settings.samples, settings.samples);
            }
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
                                acknowledge(progress, settings.first, written)?
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
            // CUDA starts the next frame while the CPU writer handles this one.
            // A single queued writer frame bounds memory and applies backpressure off the GUI.
            if number != settings.last {
                continue;
            }
            loop {
                if cancel.load(Ordering::Acquire) {
                    return Ok(false);
                }
                match writer.events.recv_timeout(Duration::from_millis(20)) {
                    Ok(WriteEvent::Written(written)) => {
                        acknowledge(progress, settings.first, written)?
                    }
                    Ok(WriteEvent::Finished) => return Ok(true),
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
        let _ = port.try_command(Command::CancelExport);
        progress.lock().unwrap_or_else(|e| e.into_inner()).status =
            "Finishing export; saving completed frames…".into();
        writer.frames.take();
        loop {
            match writer.events.recv_timeout(Duration::from_millis(20)) {
                Ok(WriteEvent::Written(written)) => acknowledge(progress, settings.first, written)?,
                Ok(WriteEvent::Finished) => break,
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
        let _ = port.try_command(Command::CancelExport);
        loop {
            match writer.events.recv_timeout(Duration::from_millis(20)) {
                Ok(WriteEvent::Cancelled)
                | Ok(WriteEvent::Failed(_))
                | Ok(WriteEvent::Finished)
                | Err(mpsc::RecvTimeoutError::Disconnected) => break,
                _ => {}
            }
        }
    }
    result
}

fn acknowledge(progress: &Mutex<Progress>, first: u32, written: u32) -> Result<(), String> {
    let mut state = progress.lock().unwrap_or_else(|e| e.into_inner());
    if written != first + state.completed {
        return Err("Writer acknowledgements arrived out of order".into());
    }
    state.completed += 1;
    Ok(())
}

enum WriteEvent {
    Written(u32),
    Finished,
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
                let result = (|| -> Result<(), String> {
                    let mut hevc = if settings.format == ExportFormat::Hevc {
                        Some(HevcSink::new(&settings)?)
                    } else {
                        None
                    };
                    loop {
                        if stop.load(Ordering::Acquire) {
                            let _ = tx.send(WriteEvent::Cancelled);
                            return Ok(());
                        }
                        let (number, frame) = match rx.recv_timeout(Duration::from_millis(30)) {
                            Ok(value) => value,
                            Err(mpsc::RecvTimeoutError::Timeout) => continue,
                            Err(mpsc::RecvTimeoutError::Disconnected) => {
                                if let Some(sink) = hevc {
                                    sink.finish(&stop)?;
                                }
                                let _ = tx.send(WriteEvent::Finished);
                                return Ok(());
                            }
                        };
                        if !frame.complete(settings.samples) {
                            return Err(format!(
                                "Incomplete frame: {} < {} samples",
                                frame.samples, settings.samples
                            ));
                        }
                        if frame.width != settings.width || frame.height != settings.height {
                            return Err("Export frame resolution changed".into());
                        }
                        let path = settings.frame_path(number);
                        match settings.format {
                            ExportFormat::Hevc => hevc.as_mut().ok_or("HEVC sink missing")?.write(&frame)?,
                            ExportFormat::Exr => write_exr(&path, &frame, settings.overwrite)?,
                            // The mastering peak is the rendered view's measured one (`hdr_scale`);
                            // an OCIO export never has relative light (Reinhard is off), so the
                            // SDR white argument only matters for an SDR file.
                            ExportFormat::Png => frame.save_png(&path, settings.png, crate::color::BT2408_SDR_WHITE_NITS, settings.overwrite)?,
                        }
                        let _ = tx.send(WriteEvent::Written(number));
                        if number == settings.last {
                            if let Some(sink) = hevc {
                                sink.finish(&stop)?;
                            }
                            if stop.load(Ordering::Acquire) {
                                let _ = tx.send(WriteEvent::Cancelled);
                            } else {
                                let _ = tx.send(WriteEvent::Finished);
                            }
                            return Ok(());
                        }
                    }
                })();
                if let Err(error) = result {
                    let _ = tx.send(WriteEvent::Failed(error));
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

fn write_exr(path: &Path, frame: &Frame, overwrite: bool) -> Result<(), String> {
    if frame.radiance.len() != frame.width * frame.height {
        return Err("Scene-linear radiance is missing".into());
    }
    crate::exr_io::write_rgb(path, frame.width, frame.height, &frame.radiance, &crate::color::WORKING_PRIMS, None, overwrite)
}

struct HevcSink {
    ctx: av_codec_core::AVCodecContext,
    sws: Box<av_swscale::SwsContext>,
    upload_sws: Option<Box<av_swscale::SwsContext>>,
    rgb: Box<av_util_frame::AVFrame>,
    yuv: Box<av_util_frame::AVFrame>,
    nv12: Option<Box<av_util_frame::AVFrame>>,
    hardware: bool,
    writer: Option<av_format_movenc::MovWriter<File>>,
    output: Option<av_util_core::outfile::AtomicOut>,
    width: usize,
    height: usize,
    fps: (u32, u32),
    frames: i64,
    header: bool,
    timing: VideoTiming,
}

/// Presentation interval on MovWriter's implicit decode clock. B-frame reordering
/// must not introduce a leading empty edit into a clip that starts at frame zero.
#[derive(Default)]
struct VideoTiming {
    first_dts: Option<i64>,
    bounds: Option<(i64, i64)>,
}
impl VideoTiming {
    fn record(&mut self, pts: i64, dts: i64, duration: i64) -> Result<(), String> {
        if duration <= 0 {
            return Err("Invalid HEVC packet duration".into());
        }
        let end = pts.checked_add(duration).ok_or("HEVC timestamp overflow")?;
        self.first_dts.get_or_insert(dts);
        self.bounds = Some(match self.bounds {
            Some((first, last)) => (first.min(pts), last.max(end)),
            None => (pts, end),
        });
        Ok(())
    }
    fn window(&self) -> Result<(i64, i64), String> {
        let origin = self.first_dts.ok_or("HEVC output has no packets")?;
        let (first, last) = self
            .bounds
            .ok_or("HEVC output has no presentation interval")?;
        let start = first
            .checked_sub(origin)
            .ok_or("HEVC edit start overflow")?;
        let end = last.checked_sub(origin).ok_or("HEVC edit end overflow")?;
        if start < 0 || end <= start {
            return Err("Invalid HEVC presentation interval".into());
        }
        Ok((start, end))
    }
}

fn av_error(error: impl std::fmt::Debug) -> String {
    format!("{error:?}")
}
/// Fixed-size conversion buffers belong to one export writer and are reused
/// for its lifetime. CPU encoding copies input; Vulkan upload owns its surface.
fn video_buffer(
    width: usize,
    height: usize,
    format: av_util_pixfmt::AVPixelFormat,
) -> Result<Box<av_util_frame::AVFrame>, String> {
    let mut frame = av_util_frame::av_frame_alloc();
    frame.format = format as i32;
    frame.width = width as i32;
    frame.height = height as i32;
    av_util_frame::av_frame_get_buffer(&mut frame, 0).map_err(av_error)?;
    Ok(frame)
}
impl HevcSink {
    fn new(settings: &ExportSettings) -> Result<Self, String> {
        use av_util_pixfmt::{AVColorSpace as C, AVPixelFormat as P};
        let hardware = settings.encoder == VideoEncoder::Vulkan;
        let entry = av_codec::avcodec_find_encoder_by_name(if hardware {
            "hevc_vulkan"
        } else {
            "hevc_kvz"
        })
        .ok_or("Selected ffmpeg-rs HEVC encoder is not registered")?;
        let sw_format = if hardware { P::NV12 } else { P::YUV420P };
        let mut ctx = av_codec_core::avcodec_alloc_context3();
        ctx.width = settings.width as i32;
        ctx.height = settings.height as i32;
        ctx.pix_fmt = if hardware { P::VULKAN } else { sw_format };
        ctx.time_base.num =
            i32::try_from(settings.fps_den).map_err(|_| "FPS denominator overflow")?;
        ctx.time_base.den =
            i32::try_from(settings.fps_num).map_err(|_| "FPS numerator overflow")?;
        if hardware {
            ctx.hw_frames_ctx = Some(av_hwaccel_vulkan::hevc_vulkan_alloc_src_frames(
                ctx.width, ctx.height, 4, sw_format,
            ).map_err(|error| format!("Vulkan Video unavailable: {error:?}. Select CPU · Kvazaar to use software encoding."))?.into());
            ctx.max_b_frames = 0;
            ctx.gop_size = (settings.fps_num / settings.fps_den).max(1) as i32;
        }
        ctx.flags |=
            av_codec_core::AV_CODEC_FLAG_GLOBAL_HEADER | av_codec_core::AV_CODEC_FLAG_QSCALE;
        ctx.global_quality = i32::from(settings.qp) * av_codec_core::FF_QP2LAMBDA;
        if hardware {
            let qp = settings.qp.to_string();
            ctx.open_with_opts((entry.make)(), &[("qp", &qp), ("async_depth", "1")])
                .map_err(|error| format!("Vulkan Video cannot encode {}×{}: {error:?}. Select CPU · Kvazaar for software encoding.", settings.width, settings.height))?;
        } else {
            // The pinned Kvazaar inter path produces block displacement on
            // moving sources. Its no-preset route is bounded all-intra, with
            // QP still controlled by global_quality; no temporal references.
            ctx.gop_size = 1;
            ctx.open_with_opts((entry.make)(), &[]).map_err(av_error)?;
        }
        let mut sws = av_swscale::sws_alloc_context();
        sws.set_src(settings.width, settings.height, P::RGB24);
        sws.set_dst(settings.width, settings.height, P::YUV420P);
        sws.set_colorspace(C::AVCOL_SPC_BT709);
        sws.set_range(true, false);
        av_swscale::sws_init_context(&mut sws)
            .map_err(|e| format!("RGB to YUV conversion: {e:?}"))?;
        // RGB → NV12 is not yet supported by the shared scaler. Convert color
        // once to planar 4:2:0, then only repack chroma for Vulkan upload.
        let upload_sws = if hardware {
            let mut packing = av_swscale::sws_alloc_context();
            packing.set_src(settings.width, settings.height, P::YUV420P);
            packing.set_dst(settings.width, settings.height, P::NV12);
            packing.set_range(false, false);
            av_swscale::sws_init_context(&mut packing).map_err(av_error)?;
            Some(packing)
        } else {
            None
        };
        let mut output = av_util_core::outfile::AtomicOut::create(
            &settings.output(),
            settings.overwrite,
        )
        .map_err(|e| e.to_string())?;
        let writer =
            av_format_movenc::MovWriter::new(output.take_file().map_err(|e| e.to_string())?)
                .map_err(av_error)?;
        let mut sink = Self {
            ctx,
            sws,
            upload_sws,
            rgb: video_buffer(settings.width, settings.height, P::RGB24)?,
            yuv: video_buffer(settings.width, settings.height, P::YUV420P)?,
            nv12: if hardware {
                Some(video_buffer(settings.width, settings.height, P::NV12)?)
            } else {
                None
            },
            hardware,
            writer: Some(writer),
            output: Some(output),
            width: settings.width,
            height: settings.height,
            fps: (settings.fps_num, settings.fps_den),
            frames: 0,
            header: false,
            timing: VideoTiming::default(),
        };
        if !sink.ctx.extradata.is_empty() {
            sink.header(&sink.ctx.extradata.clone())?;
        }
        Ok(sink)
    }
    fn header(&mut self, au: &[u8]) -> Result<(), String> {
        let config =
            av_codec::hvcc_from_au(au).map_err(|e| format!("HEVC configuration: {e:?}"))?;
        let writer = self.writer.as_mut().ok_or("HEVC writer is closed")?;
        writer
            .add_video(
                av_format_core::Codec::Hevc,
                &config,
                self.width as u32,
                self.height as u32,
                self.fps.0,
                &av_format_core::UNITY_MATRIX,
            )
            .map_err(av_error)?;
        // Rec.709 primaries, sRGB transfer (the current SDR renderer codes), BT.709 matrix, limited YUV.
        writer
            .set_video_colr(&[b'n', b'c', b'l', b'x', 0, 1, 0, 13, 0, 1, 0])
            .map_err(av_error)?;
        self.header = true;
        Ok(())
    }
    fn write(&mut self, frame: &Frame) -> Result<(), String> {
        if frame.light_kind.hdr() {
            return Err("HEVC SDR sink cannot encode HDR display codes".into());
        }
        if let Some(error) = &frame.colour_error {
            return Err(format!("Colour transform failed: {error}"));
        }
        if frame.light.len() != self.width * self.height {
            return Err("Linear display pixel buffer is incomplete".into());
        }
        let src = &mut self.rgb;
        let stride = src.linesize[0] as usize;
        let plane = src.data[0]
            .as_mut()
            .and_then(|b| b.data_mut())
            .ok_or("RGB input plane is not writable")?;
        for y in 0..self.height {
            for x in 0..self.width {
                let p = frame.light[y * self.width + x];
                let i = y * stride + x * 3;
                for c in 0..3 {
                    plane[i + c] = (crate::color::oetf(p[c]).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                }
            }
        }
        let dst = &mut self.yuv;
        av_swscale::sws_scale_frame(&self.sws, dst, src).map_err(av_error)?;
        dst.pts = self.frames * i64::from(self.fps.1);
        dst.duration = i64::from(self.fps.1);
        if self.hardware {
            let pool = self
                .ctx
                .hw_frames_ctx
                .as_ref()
                .ok_or("Vulkan frame pool is missing")?;
            let nv = self.nv12.as_mut().ok_or("NV12 buffer missing")?;
            av_swscale::sws_scale_frame(
                self.upload_sws.as_ref().ok_or("NV12 packer missing")?,
                nv,
                dst,
            )
            .map_err(av_error)?;
            let mut hw = av_hwaccel_vulkan::hwupload_frame(pool, nv).map_err(av_error)?;
            hw.pts = dst.pts;
            hw.duration = dst.duration;
            self.ctx.avcodec_send_frame(Some(&hw)).map_err(av_error)?;
        } else {
            self.ctx.avcodec_send_frame(Some(dst)).map_err(av_error)?;
        }
        self.frames += 1;
        self.drain()
    }
    fn drain(&mut self) -> Result<(), String> {
        loop {
            let mut packet = av_codec_core::AVPacket::new();
            match self.ctx.avcodec_receive_packet(&mut packet) {
                Ok(()) => {
                    if !self.header {
                        self.header(packet.data())?;
                    }
                    self.timing
                        .record(packet.pts, packet.dts, packet.duration)?;
                    if packet.duration != i64::from(self.fps.1) {
                        return Err("Unexpected HEVC packet duration".into());
                    }
                    let delta = i32::try_from(
                        packet
                            .pts
                            .checked_sub(packet.dts)
                            .ok_or("HEVC composition timestamp overflow")?,
                    )
                    .map_err(|_| "HEVC composition timestamp exceeds i32")?;
                    if delta < 0 {
                        return Err("HEVC packet has PTS before DTS".into());
                    }
                    self.writer
                        .as_mut()
                        .ok_or("HEVC writer closed")?
                        .write_sample(
                            0,
                            &av_codec::annexb_to_length_prefixed(packet.data()),
                            self.fps.1,
                            delta,
                            packet.flags & av_codec_core::AV_PKT_FLAG_KEY != 0,
                        )
                        .map_err(av_error)?;
                }
                Err(av_util_core::AvError::Posix(11)) | Err(av_util_core::AvError::Eof) => {
                    return Ok(());
                }
                Err(error) => return Err(av_error(error)),
            }
        }
    }
    fn finish(mut self, cancel: &AtomicBool) -> Result<(), String> {
        if cancel.load(Ordering::Acquire) || self.frames == 0 {
            return Ok(());
        }
        self.ctx.avcodec_send_frame(None).map_err(av_error)?;
        self.drain()?;
        if cancel.load(Ordering::Acquire) {
            return Ok(());
        }
        self.writer
            .as_mut()
            .ok_or("HEVC writer closed")?
            .set_video_edits(&[self.timing.window()?])
            .map_err(av_error)?;
        self.writer
            .take()
            .ok_or("HEVC writer closed")?
            .finish()
            .map_err(av_error)?;
        if cancel.load(Ordering::Acquire) {
            return Ok(());
        }
        self.output
            .take()
            .ok_or("HEVC output closed")?
            .commit()
            .map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn fps_ui_values_preserve_broadcast_timing_and_reduce_integer_rates() {
        let mut settings = super::ExportSettings::default();
        for (fps, exact) in [
            (24.0, (24, 1)),
            (25.0, (25, 1)),
            (23.976, (24000, 1001)),
            (29.97, (30000, 1001)),
            (59.94, (60000, 1001)),
            (27.5, (55, 2)),
        ] {
            settings.set_fps(fps);
            assert_eq!((settings.fps_num, settings.fps_den), exact);
            assert!((settings.fps() - fps).abs() < 0.0005);
        }
        settings.set_fps(f64::NAN);
        assert_eq!((settings.fps_num, settings.fps_den), (24, 1));
    }
    #[test]
    fn final_denoise_override_runs_once_and_preserves_world_settings() {
        let mut world = crate::scene::Scene::preset(crate::params::FAMILY_BULB);
        world.render.denoise.interval = 8;
        world.render.denoise.enabled = false;
        let mut frame = world.clone();
        let settings = super::ExportSettings {
            denoise_at_completion: true,
            ..Default::default()
        };
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
        super::ExportSettings::default().apply_denoise_policy(&mut world);
        assert_eq!(world.render.denoise.interval, 8);
        assert!(!world.render.denoise.enabled);
    }
    use super::*;

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
            converged: false,
            last_ms: 1.,
            last_spp: 4,
            unresolved: 0.0,
            sdr_bytes: Arc::new(Vec::new()),
            hdr_bytes: Arc::new(Vec::new()),
        }
    }
    /// Vulkan Video sessions are a per-device hardware resource: two tests opening one at once
    /// fail with `Vulkan Video unavailable`. Tests that may use the GPU encoder hold this.
    fn vulkan_video() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: Mutex<()> = Mutex::new(());
        LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }
    fn temp_dir(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
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
    #[test]
    fn exr_sequence_writer_preserves_float_radiance() {
        let dir = temp_dir("exr");
        let mut settings = ExportSettings::default();
        settings.name = "seq".into();
        settings.dir = dir.clone();
        settings.width = 2;
        settings.height = 2;
        settings.samples = 4;
        settings.first = 7;
        settings.last = 8;
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
            WriteEvent::Finished
        ));
        let (_, _, pixels, _) = crate::exr_io::read_rgb(&settings.frame_path(7)).unwrap();
        assert_eq!(pixels[0], [2., 0.5, 0.125]);
        assert!(settings.frame_path(8).exists());
        std::fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn hevc_mux_finishes_and_abort_never_publishes() {
        let _video = vulkan_video();
        let dir = temp_dir("hevc");
        let mut settings = ExportSettings::default();
        settings.format = ExportFormat::Hevc;
        settings.encoder = VideoEncoder::Kvazaar;
        settings.name = "clip".into();
        settings.dir = dir.clone();
        settings.width = 64;
        settings.height = 64;
        settings.preset = 0;
        let mut sink = HevcSink::new(&settings).unwrap();
        sink.write(&frame(64, 64)).unwrap();
        sink.write(&frame(64, 64)).unwrap();
        sink.finish(&AtomicBool::new(false)).unwrap();
        let bytes = std::fs::read(settings.output()).unwrap();
        for name in [b"ftyp", b"moov", b"hvcC", b"hvc1"] {
            assert!(bytes.windows(4).any(|b| b == name), "{:?}", name);
        }
        settings.name = "cancelled".into();
        let mut sink = HevcSink::new(&settings).unwrap();
        sink.write(&frame(64, 64)).unwrap();
        sink.finish(&AtomicBool::new(true)).unwrap();
        assert!(!settings.output().exists());
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn graceful_video_stop_drains_queued_frames_and_flushes_delayed_packets() {
        let dir = temp_dir("partial-hevc");
        for count in [0, 1, 9] {
            let settings = ExportSettings {
                format: ExportFormat::Hevc,
                encoder: VideoEncoder::Kvazaar,
                name: format!("partial-{count}"),
                dir: dir.clone(),
                width: 66,
                height: 50,
                samples: 4,
                first: 17,
                last: 100,
                fps_num: 24000,
                fps_den: 1001,
                preset: HIGH_QUALITY_PRESET,
                ..Default::default()
            };
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
                    WriteEvent::Finished => break,
                    WriteEvent::Failed(error) => panic!("{error}"),
                    WriteEvent::Cancelled => panic!("graceful stop was aborted"),
                }
            }
            assert_eq!(written, count);
            if count == 0 {
                assert!(!settings.output().exists());
            } else {
                let demux = av_format_mov::Demuxer::open(settings.output()).unwrap();
                assert_eq!(demux.sample_count(), count as usize);
                assert_eq!(demux.start_time(), 0);
                assert_eq!(demux.presented_frame_count().unwrap(), count as usize);
                assert_eq!(demux.presented_duration().unwrap(), count as u64 * 1001);
            }
        }
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 2);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    #[ignore = "requires Vulkan Video HEVC hardware encode"]
    fn vulkan_hevc_partial_export_flushes_and_has_exact_timing() {
        let _video = vulkan_video();
        let dir = temp_dir("vulkan-partial");
        let settings = ExportSettings {
            format: ExportFormat::Hevc,
            encoder: VideoEncoder::Vulkan,
            name: "partial".into(),
            dir: dir.clone(),
            width: 256,
            height: 256,
            samples: 4,
            first: 0,
            last: 100,
            fps_num: 24000,
            fps_den: 1001,
            ..Default::default()
        };
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
                WriteEvent::Finished => break,
                WriteEvent::Failed(error) => panic!("{error}"),
                WriteEvent::Cancelled => panic!("partial GPU movie aborted"),
            }
        }
        assert_eq!(written, 9);
        let demux = av_format_mov::Demuxer::open(settings.output()).unwrap();
        assert_eq!(demux.sample_count(), 9);
        assert_eq!(demux.start_time(), 0);
        assert_eq!(demux.presented_frame_count().unwrap(), 9);
        assert_eq!(demux.presented_duration().unwrap(), 9 * 1001);
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// Moving, detailed frames exercise inter prediction; constant swatches cannot
    /// expose stale references. Set WARP_BRO_VIDEO_FIXTURE to retain RGB oracles
    /// and movies for an independent decoder comparison.
    #[test]
    fn hevc_motion_fixture_encodes_every_source_frame() {
        let _video = vulkan_video();
        use std::io::Write;
        let retained = std::env::var_os("WARP_BRO_VIDEO_FIXTURE");
        let dir = retained
            .as_ref()
            .map(PathBuf::from)
            .unwrap_or_else(|| temp_dir("motion"));
        std::fs::create_dir_all(&dir).unwrap();
        let mut reference = File::create(dir.join("motion.rgb")).unwrap();
        let mut source = frame(256, 256);
        let mut sinks: Vec<_> = [VideoEncoder::Kvazaar, VideoEncoder::Vulkan]
            .into_iter()
            .map(|encoder| {
                let settings = ExportSettings {
                    encoder,
                    format: ExportFormat::Hevc,
                    width: 256,
                    height: 256,
                    preset: HIGH_QUALITY_PRESET,
                    name: format!("motion-{encoder:?}"),
                    dir: dir.clone(),
                    ..Default::default()
                };
                HevcSink::new(&settings).unwrap()
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
                        rgb.push((crate::color::oetf(value).clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
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
        for encoder in [VideoEncoder::Kvazaar, VideoEncoder::Vulkan] {
            let demux =
                av_format_mov::Demuxer::open(&dir.join(format!("motion-{encoder:?}.mp4"))).unwrap();
            assert_eq!(demux.presented_frame_count().unwrap(), 51);
        }
        if retained.is_none() {
            std::fs::remove_dir_all(dir).unwrap();
        }
    }

    #[test]
    fn reordered_hevc_clips_start_at_zero_with_exact_rational_frame_timing() {
        let dir = temp_dir("hevc-timing");
        for preset in [1, HIGH_QUALITY_PRESET] {
            let settings = ExportSettings {
                format: ExportFormat::Hevc,
                encoder: VideoEncoder::Kvazaar,
                name: format!("clip-{preset}"),
                dir: dir.clone(),
                width: 66,
                height: 50,
                preset,
                fps_num: 24000,
                fps_den: 1001,
                ..ExportSettings::default()
            };
            let mut sink = HevcSink::new(&settings).unwrap();
            for _ in 0..32 {
                sink.write(&frame(66, 50)).unwrap();
            }
            sink.finish(&AtomicBool::new(false)).unwrap();
            let demux = av_format_mov::Demuxer::open(settings.output()).unwrap();
            assert_eq!(demux.sample_count(), 32);
            assert_eq!(demux.timescale(), 24000);
            assert_eq!(demux.start_time(), 0);
            assert_eq!(demux.presented_duration().unwrap(), 32 * 1001);
            assert_eq!(demux.presented_frame_count().unwrap(), 32);
            assert!(demux.edit_list().iter().all(|e| e.media_time >= 0));
            let mut pts: Vec<_> = (0..32).map(|i| demux.display_pts(i).unwrap()).collect();
            pts.sort_unstable();
            assert_eq!(pts, (0..32).map(|n| n * 1001).collect::<Vec<_>>());
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn cuda_export_coordinator_samples_animation_and_writes_each_frame_once() {
        let dir = temp_dir("cuda");
        let service = RenderService::spawn();
        let mut controller = ExportController::default();
        controller.out_root = dir.clone();
        controller.settings.width = 16;
        controller.settings.height = 16;
        controller.settings.samples = 2;
        controller.settings.first = 3;
        controller.settings.last = 4;
        let mut scene = Scene::preset(crate::params::FAMILY_KIFS);
        scene.render.denoise.enabled = false; // This test checks exact physical animation output.
        scene.camera.target = [1000.0; 3];
        scene.lighting.sun_intensity = 0.0;
        scene.lighting.sky_intensity = 0.0;
        scene.key_parameter("/lighting/sky_intensity", 3.0);
        scene.lighting.sky_intensity = 2.0;
        scene.key_parameter("/lighting/sky_intensity", 4.0);
        controller.start(&scene, &service).unwrap();
        let until = std::time::Instant::now() + Duration::from_secs(90);
        // Deliberately never poll GUI events or call update while the pipeline runs.
        while !controller
            .run
            .as_ref()
            .unwrap()
            .progress
            .lock()
            .unwrap()
            .done
        {
            assert!(
                std::time::Instant::now() < until,
                "autonomous export timed out"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
        controller.update(&service);
        assert!(!controller.is_running());
        assert_eq!(controller.completed, 2, "{}", controller.status);
        assert!(controller.status.starts_with("Export complete"), "{}", controller.status);
        let written = controller.last.clone().unwrap();
        assert!(written.frame_path(3).starts_with(&dir), "exports land in a folder under the output root");
        assert!(written.frame_path(3).exists());
        assert!(written.frame_path(4).exists());
        let pixel = |number| crate::exr_io::read_rgb(&written.frame_path(number)).unwrap().2[0];
        assert_eq!(pixel(3)[0], 0.0);
        assert!(
            pixel(4)[0] > 0.0,
            "Each exported frame must evaluate its animation"
        );
        assert_eq!(std::fs::read_dir(written.frame_path(3).parent().unwrap()).unwrap().count(), 2);
        std::fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn cancel_controller_publishes_completed_movie_without_gui_waiting() {
        let _video = vulkan_video();
        let dir = temp_dir("controller-cancel");
        let service = RenderService::spawn();
        let mut controller = ExportController::default();
        controller.out_root = dir.clone();
        controller.settings = ExportSettings {
            format: ExportFormat::Hevc,
            encoder: VideoEncoder::Vulkan,
            name: "partial".into(),
            width: 256,
            height: 256,
            samples: 1,
            last: 10000,
            ..Default::default()
        };
        let mut scene = Scene::preset(crate::params::FAMILY_KIFS);
        scene.render.denoise.enabled = false;
        controller.start(&scene, &service).unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(90);
        while controller.completed == 0 {
            controller.update(&service);
            assert!(controller.is_running(), "{}", controller.status);
            assert!(std::time::Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(5));
        }
        let started = std::time::Instant::now();
        controller.cancel(&service);
        assert!(started.elapsed() < Duration::from_millis(100));
        while controller.is_running() {
            controller.update(&service);
            assert!(std::time::Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(
            controller.status.starts_with("Export cancelled; saved"),
            "{}",
            controller.status
        );
        let demux = av_format_mov::Demuxer::open(&controller.last.clone().unwrap().output()).unwrap();
        assert_eq!(
            demux.presented_frame_count().unwrap(),
            controller.completed as usize
        );
        assert!(controller.completed > 0 && controller.completed < 10001);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn each_format_renders_through_its_own_output_transform() {
        use crate::ocio::{Ocio, OutputKind};
        let ocio = Ocio::load("ocio://studio-config-latest").unwrap();
        let scene = crate::color::default_selection(); // SDR sRGB view in the viewport
        let mut s = ExportSettings { format: ExportFormat::Png, ..Default::default() };
        let sdr = s.resolve_transform(&ocio, &scene).unwrap().unwrap();
        assert_eq!((sdr.0.as_str(), sdr.1.as_str()), (scene.display.as_str(), scene.view.as_str()));
        s.png = PngEncoding::Hdr10;
        let (display, view) = s.resolve_transform(&ocio, &scene).unwrap().unwrap();
        assert!(ocio.display_is(&display, OutputKind::Pq) && view.contains("1000 nits"), "{display} / {view}");
        s.png_peak_nits = 500.0;
        assert!(s.resolve_transform(&ocio, &scene).unwrap().unwrap().1.contains("500 nits"));
        s.png = PngEncoding::Hlg;
        let (display, _) = s.resolve_transform(&ocio, &scene).unwrap().unwrap();
        assert!(display.contains("HLG"), "{display}");
        // EXR is scene-linear; an SDR file with OCIO off keeps the built-in display.
        s.format = ExportFormat::Exr;
        assert_eq!(s.resolve_transform(&ocio, &scene).unwrap(), None);
        s.format = ExportFormat::Hevc;
        let off = crate::ocio::Sel { on: false, ..scene.clone() };
        assert_eq!(s.resolve_transform(&ocio, &off).unwrap(), None);
        // The frame scene gets the resolved display / view.
        s.transform = Some((display.clone(), "view".into()));
        let mut frame = Scene::preset(crate::params::FAMILY_BULB);
        frame.colour.on = false;
        s.apply_transform(&mut frame);
        assert!(frame.colour.on && frame.colour.display == display);
    }
    #[test]
    fn sequence_paths_and_validation() {
        let mut s = ExportSettings::default();
        s.name = "image".into();
        s.resolve(Path::new("some folder"));
        s.first = 7;
        s.last = 9;
        assert_eq!(s.frame_count(), 3);
        assert_eq!(s.frame_path(8), Path::new("some folder").join("image.000008.exr"));
        assert!(s.validate().is_ok());
        // An HDR PNG sequence numbers before the whole suffix (`fs_name::frame_file`).
        s.format = ExportFormat::Png;
        s.png = PngEncoding::Hdr10;
        s.name = "shot".into();
        s.resolve(Path::new("out"));
        assert_eq!(s.frame_path(8), Path::new("out").join("shot.000008.pq.png"));
        s.png = PngEncoding::Sdr8;
        // A one-frame range is a still: the resolved file itself, no frame number.
        s.format = ExportFormat::Png;
        s.name = "still".into();
        s.resolve(Path::new("out"));
        s.first = 5;
        s.last = 5;
        assert_eq!(s.frame_path(5), Path::new("out").join("still.png"));
        assert!(s.validate().is_ok());
        s.png_peak_nits = 50.0;
        assert!(s.validate().unwrap_err().contains("peak"));
        s.png_peak_nits = 1000.0;
        s.name = "a/b".into();
        assert!(s.validate().unwrap_err().contains("cannot hold"));
        s.name = "CON".into();
        assert!(s.validate().unwrap_err().contains("device"));
        s.name = "frame".into();
        s.first = 7;
        s.last = 9;
        s.format = ExportFormat::Hevc;
        s.dir = PathBuf::from("out");
        s.width = 17;
        assert!(s.validate().unwrap_err().contains("even"));
        s.width = 64;
        s.height = 64;
        s.last = 6;
        assert!(s.validate().is_err());
    }
}
