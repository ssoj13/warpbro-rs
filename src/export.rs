//! Event-driven export coordinator and bounded CPU writer. CUDA never runs on the UI thread.
use crate::{
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
pub use video_io::catalog::{
    Chroma, Codec, CodecInfo, Container, Depth, Encoder, ExvProfile, PngVideo, ProResProfile, Rate,
    VideoSettings,
};
use video_io::cli::ffmpeg;
pub use video_io::job::{ExportJob, MAX_AXIS, MAX_QP, PEAK_NITS};
pub use video_io::still::{
    Channels, DWA_LEVEL, ExrCompression, ExrDepth, Still, StillOptions, StillPixels,
    TiffCompression,
};
use video_io::video::{Outcome, Ticket};

const HIGH_QUALITY_QP: u8 = 18;
/// The HDR PNG export's default target peak (`ExportSettings::png_peak_nits`).
const DEFAULT_PNG_PEAK_NITS: f32 = 1000.0;
/// OpenEXR DWA compression level range (`0.0..=250.0`).
pub const DWA_LEVEL_RANGE: std::ops::RangeInclusive<f32> = 0.0..=250.0;
/// The renderer's 64 megapixel frame limit.
const MAX_PIXELS: usize = 67_108_864;

fn codec_info(codec: Codec) -> &'static CodecInfo {
    video_io::catalog::CODECS
        .iter()
        .find(|c| c.codec == codec)
        .expect("all Codec variants are in CODECS")
}
fn container_label(container: Container) -> &'static str {
    match container {
        Container::Mp4 => "MP4 (.mp4)",
        Container::Mov => "QuickTime (.mov)",
        Container::Exv => "EXV (.exv)",
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExportFormat {
    Exr,
    Png,
    Jpeg,
    Tiff,
    Tga,
    Hevc,
}
impl ExportFormat {
    /// The format tabs of the panel, in order.
    pub const ALL: [Self; 6] = [
        Self::Exr,
        Self::Png,
        Self::Jpeg,
        Self::Tiff,
        Self::Tga,
        Self::Hevc,
    ];
    fn label(self) -> &'static str {
        match self {
            Self::Exr => "EXR",
            Self::Png => "PNG",
            Self::Jpeg => "JPEG",
            Self::Tiff => "TIFF",
            Self::Tga => "TGA",
            Self::Hevc => "Video",
        }
    }
    pub fn extension(self) -> &'static str {
        match self {
            Self::Exr => "exr",
            Self::Png => "png",
            Self::Jpeg => "jpg",
            Self::Tiff => "tif",
            Self::Tga => "tga",
            Self::Hevc => "mp4",
        }
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum VideoEncoder {
    Auto,
    #[default]
    Vulkan,
    Kvazaar,
}
impl VideoEncoder {
    pub const ALL: [Self; 3] = [Self::Auto, Self::Vulkan, Self::Kvazaar];
    fn label(self) -> &'static str {
        match self {
            Self::Auto => "Auto (GPU, fallback CPU)",
            Self::Vulkan => "GPU · Vulkan Video",
            Self::Kvazaar => "CPU · Software",
        }
    }
}

/// The encode recipe of an export: the values of the document's `OutputSettings` node (branch
/// `/output`), deserialized strictly. Its `Default` seeds that node in a fresh document, so the
/// defaults exist once; `validate` is the one check for the node and for a starting export.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputSettings {
    pub format: ExportFormat,
    pub encoder: VideoEncoder,
    /// Base size; the Output render profile's resolution scale applies once when a job starts.
    pub width: usize,
    pub height: usize,
    /// HEVC / H.264 / AV1 QP/CRF of the built-in encoder, or libx265 CRF for a PNG export's video.
    pub qp: u8,
    /// Override world cadence for offline renders: OIDN runs once at the final sample.
    pub denoise_at_completion: bool,
    pub png: PngEncoding,
    /// The HDR view an HDR export renders through when the scene's own view is not an HDR
    /// view of that kind: the one whose measured peak is nearest this (`Ocio::output_transform`).
    /// The file records the rendered view's measured peak (`render_service::hdr_scale`).
    pub png_peak_nits: f32,
    /// The video a PNG export also encodes from its sequence (ffmpeg); `fps` and `qp` (CRF)
    /// are shared with the Video format.
    pub png_video: PngVideo,
    /// Video codec when `format == ExportFormat::Hevc` (`Hevc`, `H264`, `Av1`, `ProRes`, `Exv`).
    pub codec: Codec,
    /// Video container for codecs that support multiple containers (`Mp4` / `Mov`).
    pub container: Container,
    /// Video bit depth for SDR video (`Eight` / `Ten`).
    pub video_depth: Depth,
    /// Video signal transfer (`Sdr8`, `Hdr10`, `Hlg`) for codecs that carry display light.
    pub video_signal: PngEncoding,
    /// Apple ProRes profile (`Proxy`, `Lt`, `Standard`, `Hq`, `FourFourFourFour`, `Xq`).
    pub prores_profile: ProResProfile,
    /// EXV float16 video profile (all 8 `ExvProfile`s: `Bc6hLossless`, `Bc6h`, `Piz`, `Zips`, `Zip`, `Rle`, `Dwaa`, `Dwab`).
    pub exv_profile: ExvProfile,
    /// Channel layout (`Rgb` / `Rgba`) for still formats (`EXR`, `PNG`, `TIFF`, `TGA`).
    pub channels: Channels,
    /// 16-bit integer output for SDR `PNG` and `TIFF` still sequences.
    pub sixteen_bit: bool,
    /// JPEG quality `1..=100`.
    pub jpeg_quality: u8,
    /// TIFF compression (`None`, `Lzw`, `Deflate`, `PackBits`).
    pub tiff_compression: TiffCompression,
    /// OpenEXR channel depth (`Half` 16-bit float or `Float` 32-bit float).
    pub exr_depth: ExrDepth,
    /// OpenEXR compression mode (all 12 modes including `Zip`, `Piz`, `Dwaa`, `Dwab`, `Htj2k32`, `Htj2k256`).
    pub exr_compression: ExrCompression,
    /// OpenEXR DWA compression level (`0.0..=250.0`, used by `Dwaa` and `Dwab`).
    pub exr_dwa_level: f32,
    /// OCIO display / view overriding the automatic output transform; empty = automatic.
    pub output_display: String,
    pub output_view: String,
}
impl Default for OutputSettings {
    fn default() -> Self {
        Self {
            format: ExportFormat::Exr,
            encoder: VideoEncoder::Vulkan,
            width: 1920,
            height: 1080,
            qp: HIGH_QUALITY_QP,
            denoise_at_completion: false,
            png: PngEncoding::Sdr8,
            png_peak_nits: DEFAULT_PNG_PEAK_NITS,
            png_video: PngVideo::Off,
            codec: Codec::Hevc,
            container: Container::Mp4,
            video_depth: Depth::Eight,
            video_signal: PngEncoding::Sdr8,
            prores_profile: ProResProfile::Hq,
            exv_profile: ExvProfile::Bc6hLossless,
            channels: Channels::Rgba,
            sixteen_bit: false,
            jpeg_quality: 95,
            tiff_compression: TiffCompression::Deflate,
            exr_depth: ExrDepth::Float,
            exr_compression: ExrCompression::Zip,
            exr_dwa_level: DWA_LEVEL,
            output_display: String::new(),
            output_view: String::new(),
        }
    }
}
impl OutputSettings {
    /// Effective container for the active video codec.
    pub fn effective_container(&self) -> Container {
        match self.codec {
            Codec::Exv => Container::Exv,
            Codec::ProRes => Container::Mov,
            Codec::Av1 => Container::Mp4,
            Codec::Hevc | Codec::H264 => {
                if self.container == Container::Mov {
                    Container::Mov
                } else {
                    Container::Mp4
                }
            }
        }
    }
    /// Effective bit depth for the active video codec and signal.
    pub fn effective_video_depth(&self) -> Depth {
        match self.codec {
            Codec::ProRes | Codec::Exv => Depth::Ten,
            Codec::H264 => Depth::Eight,
            Codec::Hevc | Codec::Av1 => {
                if self.video_signal.hdr() {
                    Depth::Ten
                } else {
                    self.video_depth
                }
            }
        }
    }
    /// Effective signal for the active video codec (H.264 and EXV are always `Sdr8` in `video-io` validation).
    pub fn effective_video_signal(&self) -> PngEncoding {
        match self.codec {
            Codec::H264 | Codec::Exv => PngEncoding::Sdr8,
            Codec::Hevc | Codec::Av1 | Codec::ProRes => self.video_signal,
        }
    }
    /// Build the [`VideoSettings`] passed to `video-io::native::Encoder`.
    pub fn video_settings(&self) -> VideoSettings {
        let encoder = match self.codec {
            Codec::Hevc => match self.encoder {
                VideoEncoder::Auto => None,
                VideoEncoder::Vulkan => Some(Encoder::HevcVulkan),
                VideoEncoder::Kvazaar => Some(Encoder::Kvazaar),
            },
            Codec::H264 => match self.encoder {
                VideoEncoder::Auto => None,
                VideoEncoder::Vulkan => Some(Encoder::H264Vulkan),
                VideoEncoder::Kvazaar => Some(Encoder::OpenH264),
            },
            Codec::Av1 | Codec::ProRes | Codec::Exv => None,
        };
        VideoSettings {
            codec: self.codec,
            container: self.effective_container(),
            depth: self.effective_video_depth(),
            chroma: Chroma::Yuv420,
            rate: Rate::Crf(u32::from(self.qp)),
            preset: None,
            prores: self.prores_profile,
            exv: self.exv_profile,
            encoder,
        }
    }
    /// Build the [`StillOptions`] for `PNG` / `JPEG` / `TIFF` / `TGA` still exports.
    pub fn still_options(&self) -> StillOptions {
        StillOptions {
            channels: match self.format {
                ExportFormat::Jpeg => Channels::Rgb,
                _ => self.channels,
            },
            sixteen_bit: self.sixteen_bit,
            jpeg_quality: self.jpeg_quality,
            tiff_compression: self.tiff_compression,
            exr_depth: self.exr_depth,
            exr_compression: self.exr_compression,
            exr_dwa_level: self.exr_dwa_level,
            ..StillOptions::default()
        }
    }
    /// Default shared presets directory (`$OFX_RS_PRESETS_DIR/video-io` or `~/.ofx-rs/presets/video-io`).
    pub fn default_presets_dir() -> Option<PathBuf> {
        let root = std::env::var_os("OFX_RS_PRESETS_DIR")
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
            .filter(|p| p.is_absolute())
            .or_else(|| {
                dirs::home_dir()
                    .filter(|p| p.is_absolute())
                    .map(|h| h.join(".ofx-rs").join("presets"))
            })?;
        Some(root.join("video-io"))
    }
    /// Convert this recipe to a shared [`video_io::job::Preset`].
    pub fn to_preset(&self) -> video_io::job::Preset {
        video_io::job::Preset {
            format: match self.format {
                ExportFormat::Exr => video_io::job::ExportFormat::Exr,
                ExportFormat::Png => video_io::job::ExportFormat::Png,
                ExportFormat::Jpeg => video_io::job::ExportFormat::Jpeg,
                ExportFormat::Tiff => video_io::job::ExportFormat::Tiff,
                ExportFormat::Tga => video_io::job::ExportFormat::Tga,
                ExportFormat::Hevc => video_io::job::ExportFormat::Video,
            },
            width: self.width,
            height: self.height,
            video: self.video_settings(),
            qp: self.qp.min(MAX_QP),
            png: if self.format == ExportFormat::Hevc {
                self.video_signal
            } else {
                self.png
            },
            png_peak_nits: self.png_peak_nits,
            png_video: self.png_video,
            still: self.still_options(),
        }
    }
    /// Apply a shared [`video_io::job::Preset`] onto this recipe while preserving WarpBro-specific fields.
    pub fn apply_preset(&mut self, preset: &video_io::job::Preset) {
        self.format = match preset.format {
            video_io::job::ExportFormat::Exr | video_io::job::ExportFormat::ExrPassThrough => {
                ExportFormat::Exr
            }
            video_io::job::ExportFormat::Png => ExportFormat::Png,
            video_io::job::ExportFormat::Jpeg => ExportFormat::Jpeg,
            video_io::job::ExportFormat::Tiff => ExportFormat::Tiff,
            video_io::job::ExportFormat::Tga => ExportFormat::Tga,
            video_io::job::ExportFormat::Video => ExportFormat::Hevc,
        };
        self.width = preset.width;
        self.height = preset.height;
        self.codec = preset.video.codec;
        self.container = preset.video.container;
        self.video_depth = preset.video.depth;
        self.prores_profile = preset.video.prores;
        self.exv_profile = preset.video.exv;
        self.encoder = match preset.video.encoder {
            None => VideoEncoder::Auto,
            Some(Encoder::HevcVulkan | Encoder::H264Vulkan) => VideoEncoder::Vulkan,
            Some(_) => VideoEncoder::Kvazaar,
        };
        self.qp = if preset.format == video_io::job::ExportFormat::Video {
            match preset.video.rate {
                Rate::Crf(q) => q.min(255) as u8,
                Rate::BitrateKbps(_) => HIGH_QUALITY_QP,
            }
        } else {
            preset.qp
        };
        if self.format == ExportFormat::Hevc {
            self.video_signal = preset.png;
        } else {
            self.png = preset.png;
        }
        self.png_peak_nits = preset.png_peak_nits;
        self.png_video = preset.png_video;
        self.channels = preset.still.channels;
        self.sixteen_bit = preset.still.sixteen_bit;
        self.jpeg_quality = preset.still.jpeg_quality;
        self.tiff_compression = preset.still.tiff_compression;
        self.exr_depth = preset.still.exr_depth;
        self.exr_compression = preset.still.exr_compression;
        self.exr_dwa_level = preset.still.exr_dwa_level;
    }
    /// Save this recipe as a `.json` preset file on disk (compatible with `video-io-egui`).
    pub fn save_preset_file(&self, path: &Path) -> Result<String, String> {
        let mut target = path.to_path_buf();
        if target.extension().is_none() {
            target.set_extension("json");
        }
        if let Some(parent) = target.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let stem = target
            .file_stem()
            .and_then(|s| s.to_str())
            .filter(|s| !s.trim().is_empty())
            .unwrap_or("preset")
            .to_owned();
        let json = serde_json::to_string_pretty(&self.to_preset()).map_err(|e| e.to_string())?;
        std::fs::write(&target, json).map_err(|e| e.to_string())?;
        Ok(stem)
    }
    /// Load a `.json` preset file from disk (accepts [`video_io::job::Preset`], `{ "preset": ... }`, or [`OutputSettings`]).
    pub fn load_preset_file(&mut self, path: &Path) -> Result<String, String> {
        #[derive(Deserialize)]
        struct NamedPreset {
            preset: video_io::job::Preset,
        }
        let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        if let Ok(preset) = serde_json::from_str::<video_io::job::Preset>(&text)
            .or_else(|_| serde_json::from_str::<NamedPreset>(&text).map(|n| n.preset))
        {
            self.apply_preset(&preset);
        } else {
            let output = serde_json::from_str::<Self>(&text).map_err(|e| e.to_string())?;
            *self = output;
        }
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .filter(|s| !s.trim().is_empty())
            .unwrap_or("preset")
            .to_owned();
        Ok(stem)
    }
    /// List all `*.json` preset files in [`Self::default_presets_dir`], sorted by stem name.
    pub fn list_preset_files() -> Vec<(String, PathBuf)> {
        let Some(dir) = Self::default_presets_dir() else {
            return Vec::new();
        };
        let _ = std::fs::create_dir_all(&dir);
        let Ok(entries) = std::fs::read_dir(&dir) else {
            return Vec::new();
        };
        let mut list = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            if path
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("json"))
                && path.is_file()
                && let Some(stem) = path.file_stem().and_then(|s| s.to_str())
                && !stem.is_empty()
            {
                list.push((stem.to_owned(), path));
            }
        }
        list.sort_by(|a, b| a.0.cmp(&b.0));
        list
    }
    /// What a file of this format holds, under its current options.
    pub fn hint(&self) -> &'static str {
        match self.format {
            ExportFormat::Exr => {
                "Scene-linear ACEScg (AP1-tagged) OpenEXR sequence; no display transform or exposure baked in."
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
            ExportFormat::Hevc => match self.codec {
                Codec::Hevc => {
                    "H.265 / HEVC video (GPU Vulkan Video or CPU Kvazaar, 8/10-bit SDR or 10-bit HDR10 / HLG). Cancel saves completed frames."
                }
                Codec::H264 => {
                    "H.264 / AVC 8-bit SDR video (GPU Vulkan Video or CPU OpenH264) for universal compatibility. Cancel saves completed frames."
                }
                Codec::Av1 => {
                    "AV1 video (CPU rav1e, 8/10-bit SDR or 10-bit HDR10 / HLG). Cancel saves completed frames."
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
    /// The recipe's limits, checked when an export starts and shown by the form. The document
    /// enforces each field's range on edit; the cross-field rules live only here. The job adds
    /// its name, range, folder and cadence checks (`ExportSettings::validate`).
    pub fn validate(&self) -> Result<(), String> {
        if !PEAK_NITS.contains(&self.png_peak_nits) {
            return Err(format!(
                "HDR peak must be {}–{} nits",
                PEAK_NITS.start(),
                PEAK_NITS.end()
            ));
        }
        if self.width == 0 || self.height == 0 || self.width > MAX_AXIS || self.height > MAX_AXIS {
            return Err(format!("Resolution must be 1–{MAX_AXIS} pixels per axis"));
        }
        if self
            .width
            .checked_mul(self.height)
            .is_none_or(|n| n > MAX_PIXELS)
        {
            return Err("Resolution exceeds the 64 megapixel renderer limit".into());
        }
        if !(1..=100).contains(&self.jpeg_quality) {
            return Err("JPEG quality must be 1–100".into());
        }
        if !DWA_LEVEL_RANGE.contains(&self.exr_dwa_level) {
            return Err(format!(
                "EXR DWA compression level must be {}–{}",
                DWA_LEVEL_RANGE.start(),
                DWA_LEVEL_RANGE.end()
            ));
        }
        if self.format == ExportFormat::Hevc {
            let max_q = if self.codec == Codec::Av1 {
                255
            } else {
                MAX_QP
            };
            if self.qp > max_q {
                return Err("Invalid encoder quality".into());
            }
            let video = self.video_settings();
            video_io::video::Backend::Native
                .supports(&video, self.effective_video_signal())
                .map_err(|e| e.to_string())?;
            if video.subsampled()
                && (!self.width.is_multiple_of(2) || !self.height.is_multiple_of(2))
            {
                return Err(format!(
                    "{} 4:2:0 requires even width and height",
                    video.info().label
                ));
            }
            if self.codec == Codec::H264
                && self.encoder == VideoEncoder::Kvazaar
                && (!self.width.is_multiple_of(16) || !self.height.is_multiple_of(16))
            {
                return Err(
                    "H.264 CPU (OpenH264) requires width and height multiples of 16".into(),
                );
            }
        } else {
            if self.qp > MAX_QP {
                return Err("Invalid encoder quality".into());
            }
            if self.format == ExportFormat::Png
                && self.png_video == PngVideo::Hevc
                && (!self.width.is_multiple_of(2) || !self.height.is_multiple_of(2))
            {
                return Err("HEVC 4:2:0 requires even width and height".into());
            }
        }
        Ok(())
    }
    /// The display a file of this format is encoded for; None for scene-linear EXR.
    pub fn output_kind(&self) -> Option<crate::ocio::OutputKind> {
        use crate::ocio::OutputKind;
        match self.format {
            ExportFormat::Exr => None,
            ExportFormat::Jpeg | ExportFormat::Tiff | ExportFormat::Tga => Some(OutputKind::Sdr),
            ExportFormat::Png => Some(match self.png {
                PngEncoding::Sdr8 => OutputKind::Sdr,
                PngEncoding::Hdr10 => OutputKind::Pq,
                PngEncoding::Hlg => OutputKind::Hlg,
            }),
            ExportFormat::Hevc => Some(match self.effective_video_signal() {
                PngEncoding::Sdr8 => OutputKind::Sdr,
                PngEncoding::Hdr10 => OutputKind::Pq,
                PngEncoding::Hlg => OutputKind::Hlg,
            }),
        }
    }
    /// The output transform from ACEScg for this format, independent of the viewport: the
    /// override when set, else `Ocio::output_transform`. An SDR output with OCIO off keeps the
    /// built-in display (None); HDR always needs an OCIO HDR view.
    pub fn resolve_transform(
        &self,
        ocio: &crate::ocio::Ocio,
        current: &crate::ocio::Sel,
    ) -> Result<Option<(String, String)>, String> {
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
            let names = ocio
                .resolve(&picked, kind != crate::ocio::OutputKind::Sdr)
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
            return Ok(Some((names.display, names.view)));
        }
        ocio.output_transform(current, kind, self.png_peak_nits)
            .map(Some)
            .map_err(|e| e.to_string())
    }
    /// An override names a display of one kind (SDR / PQ / HLG): a new kind starts automatic.
    pub fn normalize_override(&mut self, kind_before: Option<crate::ocio::OutputKind>) {
        if self.output_kind() != kind_before {
            self.output_display.clear();
            self.output_view.clear();
        }
    }
}

/// One export: the preferences job, the document's recipe and the values fixed when it
/// starts. Assembled at runtime, never persisted as a whole.
#[derive(Clone, Debug, PartialEq)]
pub struct ExportSettings {
    pub job: ExportJob,
    /// The document's `OutputSettings` values (`App::export_ui`, `ExportController::start`).
    pub output: OutputSettings,
    /// The folder this export writes into (`resolve`: a new dated one per export). The files are
    /// derived from it, `name` and the format's suffix only (`output`, `frame_path`).
    pub dir: PathBuf,
    /// Resolved Output quality target for a frozen export job, never authored here.
    pub samples: u32,
    /// The display / view this export renders through (`resolve_transform`); None = scene-linear
    /// (EXR) or the scene's own built-in SDR display.
    pub transform: Option<(String, String)>,
}
impl Default for ExportSettings {
    fn default() -> Self {
        Self {
            job: ExportJob::default(),
            output: OutputSettings::default(),
            dir: PathBuf::new(),
            samples: 32,
            transform: None,
        }
    }
}
impl ExportSettings {
    /// Fix the Output quality for this job: its sample target, and the base size scaled once
    /// by its resolution scale. `start` and the form's check both go through here, so the
    /// form refuses exactly what a start would (an odd scaled HEVC size, for one).
    pub fn apply_quality(&mut self, samples: u32, resolution_scale: f32) {
        self.samples = samples;
        let scale = |size: usize| ((size as f32 * resolution_scale).round() as usize).max(1);
        self.output.width = scale(self.output.width);
        self.output.height = scale(self.output.height);
    }
    /// Apply after animation evaluation so authored denoise keys cannot restore periodic passes.
    pub fn apply_denoise_policy(&self, scene: &mut Scene) {
        if self.output.denoise_at_completion {
            scene.render.denoise.enabled = true;
            scene.render.denoise.interval = 0;
        }
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
    /// HDR PNG is not taken for an SDR one; a Video export uses the effective container's extension.
    fn suffix(&self) -> &'static str {
        match self.output.format {
            ExportFormat::Png => self.output.png.suffix(),
            ExportFormat::Hevc => self.output.effective_container().extension(),
            format => format.extension(),
        }
    }
    /// Write into `dir`.
    pub fn resolve(&mut self, dir: &Path) {
        self.dir = dir.to_path_buf();
    }
    /// The file of a single frame or a video: `dir/<name>.<suffix>` (`fs_name::frame_file`).
    pub fn output(&self) -> PathBuf {
        self.dir.join(crate::fs_name::frame_file(
            self.job.name.trim(),
            None,
            self.suffix(),
        ))
    }
    /// The PNG export's video: `dir/<name>.<transfer>.<ext>` (`name.pq.mov`), named like its
    /// frames so its transfer is visible. None when it encodes none.
    pub fn video_output(&self) -> Option<PathBuf> {
        let ext = (self.output.format == ExportFormat::Png)
            .then_some(self.output.png_video.extension())
            .flatten()?;
        let transfer = self.output.png.suffix().strip_suffix("png").unwrap_or("");
        Some(self.dir.join(crate::fs_name::frame_file(
            self.job.name.trim(),
            None,
            &format!("{transfer}{ext}"),
        )))
    }
    /// The literal single-frame file or the escaped sequence pattern.
    fn video_input(&self) -> PathBuf {
        if self.job.first == self.job.last {
            self.frame_path(self.job.first)
        } else {
            PathBuf::from(crate::fs_name::sequence_pattern(
                &self.dir,
                self.job.name.trim(),
                self.output.png.suffix(),
            ))
        }
    }
    /// Thin settings adapter to the shared PNG-video sequence exporter.
    fn sequence<'a>(&self, input: &'a Path) -> video_io::video::SequenceJob<'a> {
        video_io::video::SequenceJob {
            encoding: self.output.png,
            input,
            start_number: self.job.first,
            frame_count: self.frame_count(),
            fps_num: self.job.fps_num,
            fps_den: self.job.fps_den,
        }
    }
    /// Inspect the host settings adapter through the existing argument oracles.
    #[cfg(test)]
    fn ffmpeg_args(&self, levels: Option<HdrLevels>, output: &Path) -> Result<Vec<String>, String> {
        let input = self.video_input();
        let settings = self
            .output
            .png_video
            .settings(u32::from(self.output.qp))
            .ok_or_else(|| "PNG video export is off".to_string())?;
        video_io::cli::ffmpeg_args(&self.sequence(&input), &settings, levels, output)
            .map_err(|e| e.to_string())?
            .into_iter()
            .map(|arg| {
                arg.into_string()
                    .map_err(|_| "Non-Unicode argument in export fixture".into())
            })
            .collect()
    }
    pub fn validate(&self) -> Result<(), String> {
        crate::fs_name::check(self.job.name.trim())?;
        self.output.validate()?;
        if self.samples == 0 || self.samples > 1_000_000 {
            return Err("Samples must be 1–1000000".into());
        }
        if self.job.first > self.job.last || self.job.last - self.job.first > 100_000 {
            return Err("Frame range must be ordered and contain at most 100001 frames".into());
        }
        if self.dir.as_os_str().is_empty() {
            return Err("The output folder is not resolved".into());
        }
        if self.job.fps_num == 0 || self.job.fps_den == 0 {
            return Err("FPS numerator and denominator must be positive".into());
        }
        if self.video_output().is_some() {
            ffmpeg().map_err(|e| e.to_string())?;
        }
        Ok(())
    }
    pub fn frame_count(&self) -> u32 {
        self.job
            .last
            .saturating_sub(self.job.first)
            .saturating_add(1)
    }
    /// The file of frame `number`: `output` for a single frame, else `name.000042.<suffix>` (the
    /// number before the whole suffix, as in every writer).
    pub fn frame_path(&self, number: u32) -> PathBuf {
        if self.job.first == self.job.last {
            return self.output();
        }
        self.dir.join(crate::fs_name::frame_file(
            self.job.name.trim(),
            Some(number),
            self.suffix(),
        ))
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
    /// The Output quality (samples, resolution scale) the host evaluated from the document
    /// for the form's check, or why the Output render profile cannot be exported.
    pub quality: Result<(u32, f32), String>,
    /// Name of the last loaded/saved file preset in `~/.ofx-rs/presets/video-io`.
    pub selected_preset: Option<String>,
    /// Active file dialog for saving (`true`) or loading (`false`) a `.json` preset file.
    preset_picker: Option<(bool, egui_file_dialog::FileDialog)>,
}
#[derive(Clone, Default)]
struct Progress {
    status: String,
    completed: u32,
    samples: (u32, u32),
    done: bool,
    /// The light levels of the written HDR PNGs, for the video (`ExportSettings::ffmpeg_args`).
    levels: Option<HdrLevels>,
}
struct Run {
    settings: ExportSettings,
    progress: Arc<Mutex<Progress>>,
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
        Self {
            settings: ExportSettings::default(),
            status: String::new(),
            completed: 0,
            sample_progress: (0, 0),
            run: None,
            next_id: 100,
            out_root: crate::out_root(),
            last: None,
            quality: Err("The Output render profile cannot be evaluated".into()),
            selected_preset: None,
            preset_picker: None,
        }
    }
}
impl ExportController {
    pub fn settings(&self) -> &ExportSettings {
        &self.settings
    }
    /// Restore the preferences part; the recipe always comes from the document.
    pub fn restore(&mut self, job: ExportJob) {
        self.settings.job = job;
    }
    pub fn is_running(&self) -> bool {
        self.run.is_some()
    }
    /// Sequence events use a dedicated bus. This hook keeps the host's event dispatch uniform.
    pub fn handle(&mut self, _event: &RenderEvent, _service: &RenderService) {}
    /// Once per UI frame: the coordinator's progress, then the video encode it started, which
    /// is polled here (never blocks) until it ends the run.
    pub fn update(&mut self, _service: &RenderService) {
        let Some(run) = self.run.as_mut() else {
            return;
        };
        if run.encoding.is_none() {
            let Some(progress) = run.progress.try_lock().ok().map(|p| p.clone()) else {
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
    pub fn start(&mut self, scene: &Scene, service: &RenderService) -> Result<(), String> {
        if self.run.is_some() {
            return Err("An export is already running".into());
        }
        let mut settings = self.settings.clone();
        // The frozen document is the only source of the recipe and the Output render profile.
        let document = scene
            .document
            .as_deref()
            .ok_or("Export needs the frozen world document")?;
        settings.output = document.output_settings()?;
        let (samples, scale) = document
            .effective_render(
                document.output_render_profile()?,
                f64::from(settings.job.first),
            )?
            .export_quality()?;
        settings.apply_quality(samples, scale);
        // Validate against the output root first so a rejected export leaves no empty folder.
        settings.resolve(&self.out_root);
        settings.validate()?;
        settings.transform = match settings.output.output_kind() {
            Some(_) => {
                let ocio = crate::ocio::Ocio::load(&crate::ocio::source(&scene.colour.config))
                    .map_err(|e| e.to_string())?;
                settings.output.resolve_transform(&ocio, &scene.colour)?
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
                        let folder = config.dir.display();
                        let Some(video) = config.video_output() else {
                            return Ok(Some(format!("Export complete: {folder}")));
                        };
                        if stop.load(Ordering::Acquire) {
                            return Ok(Some(format!(
                                "Video cancelled; the PNG sequence is in {folder}"
                            )));
                        }
                        let levels = shared.lock().unwrap_or_else(|e| e.into_inner()).levels;
                        // The UI polls the started encode (`ExportController::update`).
                        Ok(Some(match config.encode_video(levels, &video) {
                            Ok(ticket) => {
                                let run = VideoRun {
                                    ticket,
                                    video,
                                    folder: config.dir.clone(),
                                };
                                let status = run.status();
                                *handover.lock().unwrap_or_else(|e| e.into_inner()) = Some(run);
                                status
                            }
                            Err(error) => {
                                format!("Video failed: {error}; the PNG sequence is in {folder}")
                            }
                        }))
                    },
                ));
                let status = match result {
                    Ok(Ok(Some(status))) => status,
                    Ok(Ok(None)) => {
                        let completed = shared.lock().unwrap_or_else(|e| e.into_inner()).completed;
                        if config.output.format == ExportFormat::Hevc && completed > 0 {
                            format!(
                                "Export cancelled; saved {completed} frames to {}",
                                config.output().display()
                            )
                        } else if config.output.format == ExportFormat::Hevc {
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
            coordinator: Some(coordinator),
            video: video_slot,
            encoding: None,
        });
        Ok(())
    }
    pub fn cancel(&mut self, _service: &RenderService) {
        if let Some(run) = &self.run {
            run.cancel.store(true, Ordering::Release);
            if let Some(encoding) = &run.encoding {
                encoding.ticket.cancel();
            }
            self.status = "Cancelling export…".into();
        }
    }
    fn open_preset_dialog(&mut self, save: bool) {
        let mut dialog = egui_file_dialog::FileDialog::new()
            .id(egui::Id::new(if save {
                "export.preset.save"
            } else {
                "export.preset.load"
            }))
            .add_file_filter_extensions("Export preset (.json)", vec!["json"]);
        dialog.config_mut().default_file_filter = Some("Export preset (.json)".into());
        if let Some(dir) = OutputSettings::default_presets_dir() {
            let _ = std::fs::create_dir_all(&dir);
            dialog = dialog.initial_directory(dir);
        }
        if save {
            let base = self
                .selected_preset
                .as_deref()
                .unwrap_or(self.settings.job.name.as_str());
            let stem = crate::fs_name::stem(base);
            let stem = if stem.is_empty() { "preset" } else { &stem };
            dialog = dialog.default_file_name(&format!("{stem}.json"));
            dialog.save_file();
        } else {
            dialog.pick_file();
        }
        self.preset_picker = Some((save, dialog));
    }
    fn update_preset_picker(&mut self, ctx: &egui::Context) {
        let Some((save, picker)) = &mut self.preset_picker else {
            return;
        };
        picker.update(ctx);
        let Some(path) = picker.take_picked() else {
            if matches!(
                picker.state(),
                egui_file_dialog::DialogState::Closed | egui_file_dialog::DialogState::Cancelled
            ) {
                self.preset_picker = None;
            }
            return;
        };
        let save = *save;
        self.preset_picker = None;
        if save {
            match self.settings.output.save_preset_file(&path) {
                Ok(stem) => {
                    self.selected_preset = Some(stem.clone());
                    self.status = format!("Saved preset {stem}");
                }
                Err(error) => self.status = format!("Preset save failed: {error}"),
            }
        } else {
            match self.settings.output.load_preset_file(&path) {
                Ok(stem) => {
                    self.selected_preset = Some(stem.clone());
                    self.status = format!("Loaded preset {stem}");
                }
                Err(error) => self.status = format!("Preset load failed: {error}"),
            }
        }
    }
    fn preset_row(&mut self, ui: &mut egui::Ui) {
        let presets = OutputSettings::list_preset_files();
        if let Some(sel) = &self.selected_preset
            && !presets.iter().any(|(name, _)| name == sel)
        {
            self.selected_preset = None;
        }
        ui.label("Preset").on_hover_text(
            "Shared file-based export presets in ~/.ofx-rs/presets/video-io (*.json), compatible across WarpBro, Playa, and exr-view.",
        );
        ui.horizontal(|ui| {
            let selected_text = self.selected_preset.as_deref().unwrap_or("—").to_owned();
            let mut chosen_path: Option<PathBuf> = None;
            egui::ComboBox::from_id_salt("export_file_presets")
                .selected_text(selected_text)
                .show_ui(ui, |ui| {
                    for (name, path) in &presets {
                        let is_selected = self.selected_preset.as_deref() == Some(name.as_str());
                        if ui.selectable_label(is_selected, name).clicked() {
                            chosen_path = Some(path.clone());
                        }
                    }
                });
            if let Some(path) = chosen_path {
                match self.settings.output.load_preset_file(&path) {
                    Ok(stem) => {
                        self.selected_preset = Some(stem.clone());
                        self.status = format!("Loaded preset {stem}");
                    }
                    Err(error) => self.status = format!("Preset load failed: {error}"),
                }
            }
            if ui
                .button("Save Preset")
                .on_hover_text("Save the current encode recipe to a .json preset file (starts in ~/.ofx-rs/presets/video-io).")
                .clicked()
            {
                self.open_preset_dialog(true);
            }
            if ui
                .button("Load Preset")
                .on_hover_text("Load an encode recipe from a .json preset file on disk.")
                .clicked()
            {
                self.open_preset_dialog(false);
            }
            if ui
                .add_enabled(self.selected_preset.is_some(), egui::Button::new("Delete"))
                .on_hover_text("Delete the selected preset file from ~/.ofx-rs/presets/video-io.")
                .clicked()
                && let Some(sel) = self.selected_preset.take()
                && let Some((_, path)) = presets.iter().find(|(n, _)| n == &sel)
            {
                let _ = std::fs::remove_file(path);
                self.status = format!("Deleted preset {sel}");
            }
        });
        ui.end_row();
    }
    /// `timeline` is (first, last, fps, playhead) of the world document. The host fills
    /// `settings.output` from the document's OutputSettings node before this call and authors
    /// any change made here back to that node (`App::export_ui`). `rows` draws the host's own
    /// rows (its bindings) at the top of the form's grid, `freeze` the scene a start renders;
    /// both get the host's `host` (its document editor).
    #[allow(clippy::too_many_arguments)]
    pub fn ui<H>(
        &mut self,
        ui: &mut egui::Ui,
        host: &mut H,
        timeline: (u32, u32, f64, u32),
        service: &RenderService,
        ocio: Option<&crate::ocio::Ocio>,
        current: &crate::ocio::Sel,
        rows: impl FnOnce(&mut egui::Ui, &mut H),
        freeze: impl FnOnce(&mut H) -> Scene,
    ) {
        let running = self.is_running();
        let kind_before = self.settings.output.output_kind();
        self.update_preset_picker(ui.ctx());
        ui.add_enabled_ui(!running, |ui| {
            // One grid, so every value column lines up: the host's rows, the settings every
            // format shares, then the format row (the tabs) and only that format's options.
            egui::Grid::new("render_encode").num_columns(2).show(ui, |ui| {
                rows(ui, host);
                self.preset_row(ui);
                ui.label("Name"); ui.horizontal(|ui| {
                    ui.text_edit_singleline(&mut self.settings.job.name);
                    ui.label(format!(".{}", self.settings.suffix()));
                }); ui.end_row();
                ui.label("Resolution"); ui.horizontal(|ui| { ui.add(egui::DragValue::new(&mut self.settings.output.width).range(1..=MAX_AXIS)); ui.label("×"); ui.add(egui::DragValue::new(&mut self.settings.output.height).range(1..=MAX_AXIS)); }); ui.end_row();
                ui.label("Samples / frame"); ui.label(match &self.quality {
                    Ok((samples, scale)) => format!("{samples} · Output Quality · {:.0}% resolution", scale * 100.0),
                    Err(_) => "Output Quality unavailable".into(),
                }); ui.end_row();
                ui.label("Denoise"); ui.checkbox(&mut self.settings.output.denoise_at_completion, "Once at completion").on_hover_text("Run OIDN once after all samples of each exported frame; override Output profile denoise cadence."); ui.end_row();
                ui.label("Frame range"); ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut self.settings.job.first).range(0..=u32::MAX)); ui.label("…"); ui.add(egui::DragValue::new(&mut self.settings.job.last).range(0..=u32::MAX));
                    if ui.button("Current frame").on_hover_text("Render only the frame under the playhead.").clicked() {
                        self.settings.job.first = timeline.3;
                        self.settings.job.last = timeline.3;
                    }
                    if ui.button("Timeline").on_hover_text("The timeline's range, and its FPS for video.").clicked() {
                        self.settings.job.first = timeline.0;
                        self.settings.job.last = timeline.1;
                        self.settings.job.set_fps(timeline.2);
                    }
                }); ui.end_row();
                ui.label("Format"); ui.horizontal(|ui| {
                    for format in ExportFormat::ALL {
                        let r = ui.selectable_value(&mut self.settings.output.format, format, format.label());
                        if format == ExportFormat::Hevc {
                            r.on_hover_text(video_io::video::Backend::Native.hint());
                        } else if format == ExportFormat::Png {
                            r.on_hover_text(video_io::video::Backend::Cli.hint());
                        }
                    }
                }); ui.end_row();
                if let (Some(kind), Some(ocio)) = (self.settings.output.output_kind(), ocio) {
                    self.output_transform_ui(ui, ocio, current, kind);
                }
                match self.settings.output.format {
                    ExportFormat::Exr => {
                        self.exr_ui(ui);
                    }
                    ExportFormat::Png => {
                        self.png_ui(ui);
                    }
                    ExportFormat::Jpeg => {
                        self.jpeg_ui(ui);
                    }
                    ExportFormat::Tiff => {
                        self.tiff_ui(ui);
                    }
                    ExportFormat::Tga => {
                        self.channels_row(ui);
                    }
                    ExportFormat::Hevc => {
                        self.video_ui(ui);
                    }
                }
            });
            // An override names a display of one kind (SDR / PQ / HLG): a new kind starts automatic.
            self.settings.output.normalize_override(kind_before);
            ui.small(self.settings.output.hint());
            ui.weak(format!("Written to {}", self.out_root.join("<date_time>").display()));
            ui.weak("Animation is sampled at each frame. The scene and keys are frozen when export starts.");
            // The same quality, scaling and checks a start applies (`start`).
            let validation = self.quality.clone().and_then(|(samples, scale)| {
                let mut preview = self.settings.clone();
                preview.apply_quality(samples, scale);
                preview.resolve(&self.out_root);
                preview.validate()
            });
            if let Err(error) = &validation { ui.colored_label(egui::Color32::LIGHT_RED,error); }
            if ui.add_enabled(validation.is_ok(),egui::Button::new("Start render")).clicked() {
                let scene = freeze(host);
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
    fn exr_ui(&mut self, ui: &mut egui::Ui) {
        ui.label("Depth");
        ui.horizontal(|ui| {
            for depth in ExrDepth::ALL {
                ui.selectable_value(&mut self.settings.output.exr_depth, depth, depth.label());
            }
        });
        ui.end_row();
        ui.label("Compression");
        egui::ComboBox::from_id_salt("exr_compression")
            .selected_text(self.settings.output.exr_compression.label())
            .show_ui(ui, |ui| {
                for comp in ExrCompression::ALL {
                    ui.selectable_value(
                        &mut self.settings.output.exr_compression,
                        comp,
                        comp.label(),
                    );
                }
            });
        ui.end_row();
        if self.settings.output.exr_compression.dwa() {
            ui.label("DWA level");
            ui.add(
                egui::Slider::new(&mut self.settings.output.exr_dwa_level, DWA_LEVEL_RANGE)
                    .text("compression"),
            )
            .on_hover_text(
                "DreamWorks lossy wavelet level (45 is OpenEXR's default; higher compresses more).",
            );
            ui.end_row();
        }
        self.channels_row(ui);
    }
    fn png_ui(&mut self, ui: &mut egui::Ui) {
        ui.label("Encoding");
        egui::ComboBox::from_id_salt("png_encoding")
            .selected_text(self.settings.output.png.label())
            .show_ui(ui, |ui| {
                for encoding in PngEncoding::ALL {
                    ui.selectable_value(&mut self.settings.output.png, encoding, encoding.label());
                }
            });
        ui.end_row();
        if self.settings.output.png.hdr() {
            self.hdr_peak_row(ui);
        } else {
            self.sixteen_bit_row(ui);
        }
        self.channels_row(ui);
        ui.label("Video")
            .on_hover_text(video_io::video::Backend::Cli.hint());
        let found = ffmpeg().map(|p| {
            format!(
                "{}\nAlso encode the finished sequence with {} into the PNGs' name with .mov / .mp4 (name.pq.mp4), tagged like the PNGs; an HDR10 HEVC carries the PNGs' mastering peak and the clip's MaxCLL / MaxFALL.",
                video_io::video::Backend::Cli.hint(),
                p.display()
            )
        });
        egui::ComboBox::from_id_salt("png_video")
            .selected_text(self.settings.output.png_video.label())
            .show_ui(ui, |ui| {
                for video in PngVideo::ALL {
                    ui.selectable_value(&mut self.settings.output.png_video, video, video.label());
                }
            })
            .response
            .on_hover_text(
                found
                    .unwrap_or_else(|e| format!("{}\n({e})", video_io::video::Backend::Cli.hint())),
            );
        ui.end_row();
        if self.settings.output.png_video != PngVideo::Off {
            self.fps_row(ui);
        }
        if self.settings.output.png_video == PngVideo::Hevc {
            self.quality_row(ui, "CRF", MAX_QP);
        }
    }
    fn jpeg_ui(&mut self, ui: &mut egui::Ui) {
        ui.label("Quality");
        ui.add(egui::Slider::new(&mut self.settings.output.jpeg_quality, 1..=100).text("%"))
            .on_hover_text("JPEG baseline DCT quality (1–100; 95 is near-lossless).");
        ui.end_row();
    }
    fn tiff_ui(&mut self, ui: &mut egui::Ui) {
        ui.label("Compression");
        egui::ComboBox::from_id_salt("tiff_compression")
            .selected_text(self.settings.output.tiff_compression.label())
            .show_ui(ui, |ui| {
                for comp in TiffCompression::ALL {
                    ui.selectable_value(
                        &mut self.settings.output.tiff_compression,
                        comp,
                        comp.label(),
                    );
                }
            });
        ui.end_row();
        self.sixteen_bit_row(ui);
        self.channels_row(ui);
    }
    fn channels_row(&mut self, ui: &mut egui::Ui) {
        ui.label("Channels");
        ui.horizontal(|ui| {
            for ch in Channels::ALL {
                ui.selectable_value(&mut self.settings.output.channels, ch, ch.label());
            }
        });
        ui.end_row();
    }
    fn sixteen_bit_row(&mut self, ui: &mut egui::Ui) {
        ui.label("Bit depth");
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.settings.output.sixteen_bit, false, "8-bit");
            ui.selectable_value(&mut self.settings.output.sixteen_bit, true, "16-bit");
        });
        ui.end_row();
    }
    fn hdr_peak_row(&mut self, ui: &mut egui::Ui) {
        ui.label("HDR peak");
        ui.add(
            egui::DragValue::new(&mut self.settings.output.png_peak_nits)
                .range(PEAK_NITS)
                .suffix(" nits"),
        )
        .on_hover_text("When the scene's view is not an HDR view of this kind, the HDR view whose measured peak is nearest this. The file records the rendered view's measured peak (mDCV, HLG system gamma).");
        ui.end_row();
    }
    fn video_ui(&mut self, ui: &mut egui::Ui) {
        let prev_codec = self.settings.output.codec;
        ui.label("Codec");
        egui::ComboBox::from_id_salt("video_codec")
            .selected_text(codec_info(self.settings.output.codec).label)
            .show_ui(ui, |ui| {
                for info in video_io::catalog::CODECS {
                    ui.selectable_value(&mut self.settings.output.codec, info.codec, info.label);
                }
            });
        ui.end_row();
        if self.settings.output.codec != prev_codec {
            self.settings.output.container = codec_info(self.settings.output.codec).containers[0];
            if self.settings.output.codec != Codec::Av1 && self.settings.output.qp > MAX_QP {
                self.settings.output.qp = HIGH_QUALITY_QP;
            }
        }
        let codec = self.settings.output.codec;
        match codec {
            Codec::Exv => {
                ui.label("Profile");
                egui::ComboBox::from_id_salt("exv_profile")
                    .selected_text(self.settings.output.exv_profile.label())
                    .show_ui(ui, |ui| {
                        for profile in ExvProfile::ALL {
                            ui.selectable_value(
                                &mut self.settings.output.exv_profile,
                                profile,
                                profile.label(),
                            );
                        }
                    });
                ui.end_row();
                self.fps_row(ui);
            }
            Codec::ProRes => {
                ui.label("Profile");
                egui::ComboBox::from_id_salt("prores_profile")
                    .selected_text(self.settings.output.prores_profile.label())
                    .show_ui(ui, |ui| {
                        for profile in ProResProfile::ALL {
                            ui.selectable_value(
                                &mut self.settings.output.prores_profile,
                                profile,
                                profile.label(),
                            );
                        }
                    });
                ui.end_row();
                self.video_signal_rows(ui);
                self.fps_row(ui);
            }
            Codec::Av1 => {
                self.video_signal_rows(ui);
                if !self.settings.output.video_signal.hdr() {
                    self.video_depth_row(ui);
                }
                self.fps_row(ui);
                self.quality_row(ui, "Q", 255);
            }
            Codec::H264 => {
                self.video_container_row(ui);
                self.video_encoder_row(ui);
                self.fps_row(ui);
                self.quality_row(ui, "QP", MAX_QP);
            }
            Codec::Hevc => {
                self.video_container_row(ui);
                self.video_signal_rows(ui);
                if !self.settings.output.video_signal.hdr() {
                    self.video_depth_row(ui);
                }
                self.video_encoder_row(ui);
                self.fps_row(ui);
                self.quality_row(ui, "QP", MAX_QP);
                if self.settings.output.encoder == VideoEncoder::Kvazaar {
                    ui.label("Prediction");
                    ui.label("Independent I-frames");
                    ui.end_row();
                }
            }
        }
    }
    fn video_container_row(&mut self, ui: &mut egui::Ui) {
        ui.label("Container");
        ui.horizontal(|ui| {
            for &c in codec_info(self.settings.output.codec).containers {
                ui.selectable_value(&mut self.settings.output.container, c, container_label(c));
            }
        });
        ui.end_row();
    }
    fn video_signal_rows(&mut self, ui: &mut egui::Ui) {
        ui.label("Signal");
        egui::ComboBox::from_id_salt("video_signal")
            .selected_text(self.settings.output.video_signal.label())
            .show_ui(ui, |ui| {
                for sig in PngEncoding::ALL {
                    ui.selectable_value(&mut self.settings.output.video_signal, sig, sig.label());
                }
            });
        ui.end_row();
        if self.settings.output.video_signal.hdr() {
            self.hdr_peak_row(ui);
        }
    }
    fn video_depth_row(&mut self, ui: &mut egui::Ui) {
        ui.label("Bit depth");
        ui.horizontal(|ui| {
            for &d in codec_info(self.settings.output.codec).depths {
                ui.selectable_value(&mut self.settings.output.video_depth, d, d.label());
            }
        });
        ui.end_row();
    }
    fn video_encoder_row(&mut self, ui: &mut egui::Ui) {
        ui.label("Encoder")
            .on_hover_text(video_io::video::Backend::Native.hint());
        egui::ComboBox::from_id_salt("video_encoder")
            .selected_text(self.settings.output.encoder.label())
            .show_ui(ui, |ui| {
                for encoder in VideoEncoder::ALL {
                    ui.selectable_value(
                        &mut self.settings.output.encoder,
                        encoder,
                        encoder.label(),
                    );
                }
            })
            .response
            .on_hover_text(video_io::video::Backend::Native.hint());
        ui.end_row();
    }
    /// The "FPS" grid row of every video (the Video format, a PNG export's video).
    fn fps_row(&mut self, ui: &mut egui::Ui) {
        ui.label("FPS");
        ui.horizontal(|ui| {
            let mut fps = self.settings.job.fps();
            if ui
                .add(
                    egui::DragValue::new(&mut fps)
                        .speed(0.1)
                        .range(1.0..=240.0)
                        .max_decimals(3),
                )
                .changed()
            {
                self.settings.job.set_fps(fps);
            }
            egui::ComboBox::from_id_salt("export_fps_presets")
                .selected_text("Presets")
                .show_ui(ui, |ui| {
                    for rate in [23.976, 24.0, 25.0, 29.97, 30.0, 50.0, 59.94, 60.0, 120.0] {
                        if ui
                            .selectable_label(
                                (self.settings.job.fps() - rate).abs() < 0.001,
                                rate.to_string(),
                            )
                            .clicked()
                        {
                            self.settings.job.set_fps(rate);
                            ui.close();
                        }
                    }
                });
        });
        ui.end_row();
    }
    /// The "Quality" grid row of a video encoder: `qp` as QP/CRF (`0..=max_q`).
    fn quality_row(&mut self, ui: &mut egui::Ui, unit: &str, max_q: u8) {
        ui.label("Quality");
        ui.horizontal(|ui| {
            ui.add(egui::Slider::new(&mut self.settings.output.qp, 0..=max_q).text(unit))
                .on_hover_text(format!(
                    "Lower {unit} preserves more detail and produces larger files."
                ));
            if ui.button("High quality").clicked() {
                self.settings.output.qp = if max_q > MAX_QP { 60 } else { HIGH_QUALITY_QP };
            }
        });
        ui.end_row();
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
        let resolved = self.settings.output.resolve_transform(ocio, current);
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
                let displays = owned(
                    ocio.displays(hdr)
                        .into_iter()
                        .filter(|d| ocio.display_is(d, kind))
                        .collect(),
                );
                let before = self.settings.output.output_display.clone();
                crate::ocio::combo(
                    ui,
                    "export.output_display",
                    &mut self.settings.output.output_display,
                    &displays,
                    "auto",
                );
                if self.settings.output.output_display != before {
                    self.settings.output.output_view.clear();
                }
                let display = if self.settings.output.output_display.is_empty() {
                    resolved
                        .as_ref()
                        .ok()
                        .and_then(Option::as_ref)
                        .map(|(d, _)| d.clone())
                        .unwrap_or_default()
                } else {
                    self.settings.output.output_display.clone()
                };
                let views = owned(
                    ocio.views(&display, hdr)
                        .into_iter()
                        .filter(|v| ocio.view_is(&display, v, kind))
                        .collect(),
                );
                ui.add_enabled_ui(!self.settings.output.output_display.is_empty(), |ui| {
                    crate::ocio::combo(
                        ui,
                        "export.output_view",
                        &mut self.settings.output.output_view,
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
    progress: &Mutex<Progress>,
    cancel: &AtomicBool,
) -> Result<bool, String> {
    let mut writer = ExportWriter::spawn(settings.clone())?;
    let (reply, events) = mpsc::sync_channel(16);
    let result = (|| -> Result<bool, String> {
        for number in settings.job.first..=settings.job.last {
            let mut frame_scene = scene.evaluated(f64::from(number))?;
            settings.apply_denoise_policy(&mut frame_scene);
            settings.apply_transform(&mut frame_scene);
            let id = first_id + u64::from(number - settings.job.first);
            loop {
                if cancel.load(Ordering::Acquire) {
                    return Ok(false);
                }
                match port.try_command(Command::RenderExport {
                    id,
                    scene: Box::new(frame_scene.clone()),
                    width: settings.output.width,
                    height: settings.output.height,
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
                        acknowledge(progress, settings.job.first, written)?
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
                                acknowledge(progress, settings.job.first, written)?
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
            if number != settings.job.last {
                continue;
            }
            loop {
                if cancel.load(Ordering::Acquire) {
                    return Ok(false);
                }
                match writer.events.recv_timeout(Duration::from_millis(20)) {
                    Ok(WriteEvent::Written(written)) => {
                        acknowledge(progress, settings.job.first, written)?
                    }
                    Ok(WriteEvent::Finished(levels)) => {
                        progress.lock().unwrap_or_else(|e| e.into_inner()).levels = levels;
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
        progress.lock().unwrap_or_else(|e| e.into_inner()).status =
            "Finishing export; saving completed frames…".into();
        writer.frames.take();
        loop {
            match writer.events.recv_timeout(Duration::from_millis(20)) {
                Ok(WriteEvent::Written(written)) => {
                    acknowledge(progress, settings.job.first, written)?
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

fn acknowledge(progress: &Mutex<Progress>, first: u32, written: u32) -> Result<(), String> {
    let mut state = progress.lock().unwrap_or_else(|e| e.into_inner());
    if written != first + state.completed {
        return Err("Writer acknowledgements arrived out of order".into());
    }
    state.completed += 1;
    Ok(())
}

impl ExportSettings {
    /// Start encoding the finished PNG sequence into `video` (video-io's ffmpeg backend): ffmpeg
    /// writes a temp sibling, published only when it succeeds and removed on failure, Cancel or
    /// a dropped ticket. `levels`: the sequence's merged HDR levels (`HdrLevels::merge`).
    fn encode_video(
        &self,
        levels: Option<HdrLevels>,
        video: &Path,
    ) -> Result<Ticket, video_io::Error> {
        let input = self.video_input();
        let settings = self
            .output
            .png_video
            .settings(u32::from(self.output.qp))
            .ok_or_else(|| video_io::Error::Invalid("PNG video export is off".into()))?;
        video_io::video::encode(
            &self.sequence(&input),
            &settings,
            video_io::video::Backend::Cli,
            levels,
            video,
            self.job.overwrite,
        )
    }
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
                    let mut hevc = if settings.output.format == ExportFormat::Hevc {
                        Some(HevcSink::new(&settings)?)
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
                                if let Some(sink) = hevc {
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
                        if frame.width != settings.output.width
                            || frame.height != settings.output.height
                        {
                            return Err("Export frame resolution changed".into());
                        }
                        let path = settings.frame_path(number);
                        match settings.output.format {
                            ExportFormat::Hevc => {
                                hevc.as_mut().ok_or("Video sink missing")?.write(&frame)?
                            }
                            ExportFormat::Exr => {
                                write_exr(&path, &frame, &settings.output, settings.job.overwrite)?
                            }
                            // The mastering peak is the rendered view's measured one (`hdr_scale`);
                            // an OCIO export never has relative light (Reinhard is off), so the
                            // SDR white argument only matters for an SDR file.
                            ExportFormat::Png => {
                                if let Some(frame_levels) = write_still(
                                    &path,
                                    video_io::job::ExportFormat::Png,
                                    &frame,
                                    settings.output.png,
                                    settings.output.still_options(),
                                    settings.job.overwrite,
                                )? {
                                    levels = Some(
                                        levels.map_or(frame_levels, |l| l.merge(frame_levels)),
                                    );
                                }
                            }
                            ExportFormat::Jpeg => {
                                write_still(
                                    &path,
                                    video_io::job::ExportFormat::Jpeg,
                                    &frame,
                                    PngEncoding::Sdr8,
                                    settings.output.still_options(),
                                    settings.job.overwrite,
                                )?;
                            }
                            ExportFormat::Tiff => {
                                write_still(
                                    &path,
                                    video_io::job::ExportFormat::Tiff,
                                    &frame,
                                    PngEncoding::Sdr8,
                                    settings.output.still_options(),
                                    settings.job.overwrite,
                                )?;
                            }
                            ExportFormat::Tga => {
                                write_still(
                                    &path,
                                    video_io::job::ExportFormat::Tga,
                                    &frame,
                                    PngEncoding::Sdr8,
                                    settings.output.still_options(),
                                    settings.job.overwrite,
                                )?;
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
                        if number == settings.job.last {
                            if let Some(sink) = hevc {
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
    output: &OutputSettings,
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
        output.exr_depth,
        output.exr_compression,
        output.exr_dwa_level,
        output.channels,
        overwrite,
    )
}

fn write_still(
    path: &Path,
    format: video_io::job::ExportFormat,
    frame: &Frame,
    encoding: PngEncoding,
    opts: StillOptions,
    overwrite: bool,
) -> Result<Option<HdrLevels>, String> {
    if let Some(e) = &frame.colour_error {
        return Err(format!("Colour transform failed: {e}"));
    }
    if format == video_io::job::ExportFormat::Png
        && (!opts.sixteen_bit || encoding != PngEncoding::Sdr8)
    {
        let scale = video_io::export::hdr_scale(
            frame.light_kind,
            &frame.light,
            crate::color::BT2408_SDR_WHITE_NITS,
            encoding,
        );
        let still = Still {
            width: frame.width as u32,
            height: frame.height as u32,
            pixels: StillPixels::DisplayLight {
                light: &frame.light,
                sdr: &frame.sdr_bytes,
                scale,
            },
            signal: encoding,
        };
        return video_io::still::write_still(path, format, &opts, &still, overwrite)
            .map_err(|e| e.to_string());
    }
    let codes16_buf: Vec<u16>;
    let codes8_buf: Vec<u8>;
    let pixels = if opts.sixteen_bit
        && matches!(
            format,
            video_io::job::ExportFormat::Png | video_io::job::ExportFormat::Tiff
        )
        && !frame.light.is_empty()
    {
        codes16_buf = frame
            .light
            .iter()
            .flat_map(|&[r, g, b, a]| {
                [
                    (crate::color::oetf(r) * 65535.0 + 0.5) as u16,
                    (crate::color::oetf(g) * 65535.0 + 0.5) as u16,
                    (crate::color::oetf(b) * 65535.0 + 0.5) as u16,
                    (a.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16,
                ]
            })
            .collect();
        StillPixels::Codes16(&codes16_buf)
    } else if !frame.sdr_bytes.is_empty() {
        StillPixels::Codes8(&frame.sdr_bytes)
    } else {
        codes8_buf = frame
            .light
            .iter()
            .flat_map(|&[r, g, b, a]| {
                [
                    (crate::color::oetf(r) * 255.0 + 0.5) as u8,
                    (crate::color::oetf(g) * 255.0 + 0.5) as u8,
                    (crate::color::oetf(b) * 255.0 + 0.5) as u8,
                    (a.clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
                ]
            })
            .collect();
        StillPixels::Codes8(&codes8_buf)
    };
    let still = Still {
        width: frame.width as u32,
        height: frame.height as u32,
        pixels,
        signal: encoding,
    };
    video_io::still::write_still(path, format, &opts, &still, overwrite).map_err(|e| e.to_string())
}

/// Native video sink over [`video_io::native::Encoder`] supporting all 5 codecs (`Hevc`, `H264`,
/// `Av1`, `ProRes`, `Exv`), 8-bit (`Rgb24`), 10-bit (`Rgba64` SDR/HDR10/HLG), and float16 EXV
/// (`RgbaF32`).
struct HevcSink {
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
impl HevcSink {
    fn new(settings: &ExportSettings) -> Result<Self, String> {
        let video = settings.output.video_settings();
        let signal = settings.output.effective_video_signal();
        let levels = signal.hdr().then_some(HdrLevels {
            peak_nits: settings.output.png_peak_nits,
            content: None,
        });
        let params = video_io::native::Params {
            video: &video,
            signal,
            width: settings.output.width as u32,
            height: settings.output.height as u32,
            fps_num: settings.job.fps_num,
            fps_den: settings.job.fps_den,
            levels,
            audio: None,
        };
        let hardware = settings.output.encoder == VideoEncoder::Vulkan
            && matches!(video.codec, Codec::Hevc | Codec::H264);
        let encoder =
            video_io::native::Encoder::open(&params, &settings.output(), settings.job.overwrite)
                .map_err(|error| {
                    if hardware {
                        format!(
                            "Vulkan Video cannot encode {}×{}: {error}. Select CPU · Software or Auto for software encoding.",
                            settings.output.width, settings.output.height
                        )
                    } else {
                        error.to_string()
                    }
                })?;
        let n = settings.output.width * settings.output.height;
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
            width: settings.output.width,
            height: settings.output.height,
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
    #[test]
    fn fps_ui_values_preserve_broadcast_timing_and_reduce_integer_rates() {
        let _gpu_test = crate::test_gpu::lock();
        let mut settings = super::ExportSettings::default();
        for (fps, exact) in [
            (24.0, (24, 1)),
            (25.0, (25, 1)),
            (23.976, (24000, 1001)),
            (29.97, (30000, 1001)),
            (59.94, (60000, 1001)),
            (27.5, (55, 2)),
        ] {
            settings.job.set_fps(fps);
            assert_eq!((settings.job.fps_num, settings.job.fps_den), exact);
            assert!((settings.job.fps() - fps).abs() < 0.0005);
        }
        settings.job.set_fps(f64::NAN);
        assert_eq!((settings.job.fps_num, settings.job.fps_den), (24, 1));
    }
    #[test]
    fn final_denoise_override_runs_once_and_preserves_world_settings() {
        let _gpu_test = crate::test_gpu::lock();
        let mut world = crate::scene::Scene::preset(crate::params::FAMILY_BULB);
        world.render.denoise.interval = 8;
        world.render.denoise.enabled = false;
        let mut frame = world.clone();
        let settings = super::ExportSettings {
            output: OutputSettings {
                denoise_at_completion: true,
                ..OutputSettings::default()
            },
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
            active_tiles: (0, 0),
            converged: false,
            last_ms: 1.,
            last_spp: 4,
            unresolved: 0.0,
            sdr_bytes: Arc::new(Vec::new()),
            hdr_bytes: Arc::new(Vec::new()),
        }
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
        for (png, video, tags) in [
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
            let settings = ExportSettings {
                samples: 4,
                dir: dir.clone(),
                output: OutputSettings {
                    format: ExportFormat::Png,
                    png,
                    png_video: video,
                    width: 16,
                    height: 16,
                    ..OutputSettings::default()
                },
                job: ExportJob {
                    name: format!("clip-{png:?}-{video:?}"),
                    first: 1,
                    last: 2,
                    overwrite: true,
                    ..ExportJob::default()
                },
                ..Default::default()
            };
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
            let output = settings.video_output().unwrap();
            let ticket = settings.encode_video(levels, &output).unwrap();
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
                    "{png:?} {video:?} lacks {tag}:\n{stream}"
                );
            }
            if video == PngVideo::ProRes {
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
                "{png:?} {video:?} container CLL:\n{stream}"
            );
            if (png, video) == (PngEncoding::Hdr10, PngVideo::Hevc) {
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
        let mut s = ExportSettings {
            dir: PathBuf::from("out"),
            output: OutputSettings {
                format: ExportFormat::Png,
                png_video: PngVideo::Hevc,
                ..OutputSettings::default()
            },
            job: ExportJob {
                name: "shot".into(),
                first: 3,
                last: 9,
                ..ExportJob::default()
            },
            ..Default::default()
        };
        let out = Path::new("out/shot.mp4.part");
        for (png, tags) in [
            (PngEncoding::Sdr8, ["bt709", "bt709", "bt709"]),
            (PngEncoding::Hlg, ["bt2020", "arib-std-b67", "bt2020nc"]),
        ] {
            s.output.png = png;
            let args = s.ffmpeg_args(None, out).unwrap().join(" ");
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
        s.output.png = PngEncoding::Hdr10;
        assert!(
            s.ffmpeg_args(None, out).is_err(),
            "HDR10 metadata needs the measured levels"
        );
        let levels = HdrLevels {
            peak_nits: 500.0,
            content: Some(soluwell_viewport::display::screenshot::ContentLight {
                max_cll: 480.2,
                max_fall: 61.5,
            }),
        };
        let args = s.ffmpeg_args(Some(levels), out).unwrap().join(" ");
        assert!(args.contains("-start_number 3 -i out"), "{args}");
        assert!(args.contains("shot.%06d.pq.png"), "{args}");
        assert!(args.contains("master-display=G(8500,39850)B(6550,2300)R(35400,14600)WP(15635,16450)L(5000000,1):max-cll=481,62"), "{args}");
        assert_eq!(s.video_output(), Some(Path::new("out").join("shot.pq.mp4")));

        // A single frame is its file as it is: no pattern, no frame number, no `%` escaping.
        s.job.first = 5;
        s.job.last = 5;
        s.job.name = "50% grey".into();
        let args = s.ffmpeg_args(Some(levels), out).unwrap();
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
            let settings = ExportSettings {
                samples: 4,
                dir: dir.join(format!("{png:?}")),
                output: OutputSettings {
                    format: ExportFormat::Png,
                    png,
                    width: 4,
                    height: 4,
                    ..OutputSettings::default()
                },
                job: ExportJob {
                    first: 1,
                    last: 2,
                    overwrite: true,
                    ..ExportJob::default()
                },
                ..Default::default()
            };
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
        let mut settings = ExportSettings::default();
        settings.job.name = "seq".into();
        settings.dir = dir.clone();
        settings.output.width = 2;
        settings.output.height = 2;
        settings.samples = 4;
        settings.job.first = 7;
        settings.job.last = 8;
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
        let pixels = crate::exr_io::read_rgb(&settings.frame_path(7))
            .unwrap()
            .pixels;
        assert_eq!(pixels[0], [2., 0.5, 0.125]);
        assert!(settings.frame_path(8).exists());
        finish_fixture(dir);
    }
    #[test]
    fn hevc_mux_finishes_and_abort_never_publishes() {
        let _gpu_test = crate::test_gpu::lock();
        let dir = temp_dir("hevc");
        let mut settings = ExportSettings::default();
        settings.output.format = ExportFormat::Hevc;
        settings.output.encoder = VideoEncoder::Kvazaar;
        settings.job.name = "clip".into();
        settings.dir = dir.clone();
        settings.output.width = 64;
        settings.output.height = 64;
        let mut sink = HevcSink::new(&settings).unwrap();
        sink.write(&frame(64, 64)).unwrap();
        sink.write(&frame(64, 64)).unwrap();
        sink.finish(&AtomicBool::new(false)).unwrap();
        let bytes = std::fs::read(settings.output()).unwrap();
        for name in [b"ftyp", b"moov", b"hvcC", b"hvc1"] {
            assert!(bytes.windows(4).any(|b| b == name), "{:?}", name);
        }
        settings.job.name = "cancelled".into();
        let mut sink = HevcSink::new(&settings).unwrap();
        sink.write(&frame(64, 64)).unwrap();
        sink.finish(&AtomicBool::new(true)).unwrap();
        assert!(!settings.output().exists());
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1);
        finish_fixture(dir);
    }

    #[test]
    fn graceful_video_stop_drains_queued_frames_and_flushes_delayed_packets() {
        let _gpu_test = crate::test_gpu::lock();
        let dir = temp_dir("partial-hevc");
        for count in [0, 1, 9] {
            let settings = ExportSettings {
                dir: dir.clone(),
                samples: 4,
                output: OutputSettings {
                    format: ExportFormat::Hevc,
                    encoder: VideoEncoder::Kvazaar,
                    width: 66,
                    height: 50,
                    ..OutputSettings::default()
                },
                job: ExportJob {
                    name: format!("partial-{count}"),
                    first: 17,
                    last: 100,
                    fps_num: 24000,
                    fps_den: 1001,
                    ..ExportJob::default()
                },
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
                    WriteEvent::Finished(_) => break,
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
        finish_fixture(dir);
    }

    #[test]
    #[ignore = "requires Vulkan Video HEVC hardware encode"]
    fn vulkan_hevc_partial_export_flushes_and_has_exact_timing() {
        let _gpu_test = crate::test_gpu::lock();
        let dir = temp_dir("vulkan-partial");
        let settings = ExportSettings {
            dir: dir.clone(),
            samples: 4,
            output: OutputSettings {
                format: ExportFormat::Hevc,
                encoder: VideoEncoder::Vulkan,
                width: 256,
                height: 256,
                ..OutputSettings::default()
            },
            job: ExportJob {
                name: "partial".into(),
                first: 0,
                last: 100,
                fps_num: 24000,
                fps_den: 1001,
                ..ExportJob::default()
            },
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
                WriteEvent::Finished(_) => break,
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
        let mut sinks: Vec<_> = [VideoEncoder::Kvazaar, VideoEncoder::Vulkan]
            .into_iter()
            .map(|encoder| {
                let settings = ExportSettings {
                    dir: dir.clone(),
                    output: OutputSettings {
                        encoder,
                        format: ExportFormat::Hevc,
                        width: 256,
                        height: 256,
                        ..OutputSettings::default()
                    },
                    job: ExportJob {
                        name: format!("motion-{encoder:?}"),
                        ..ExportJob::default()
                    },
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
        for encoder in [VideoEncoder::Kvazaar, VideoEncoder::Vulkan] {
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
            let settings = ExportSettings {
                dir: dir.clone(),
                output: OutputSettings {
                    format: ExportFormat::Hevc,
                    encoder: VideoEncoder::Kvazaar,
                    width: 66,
                    height: 50,
                    ..OutputSettings::default()
                },
                job: ExportJob {
                    name: format!("clip-{}-{}", rate.0, rate.1),
                    fps_num: rate.0,
                    fps_den: rate.1,
                    ..ExportJob::default()
                },
                ..ExportSettings::default()
            };
            let mut sink = HevcSink::new(&settings).unwrap();
            for _ in 0..32 {
                sink.write(&frame(66, 50)).unwrap();
            }
            sink.finish(&AtomicBool::new(false)).unwrap();
            let mut demux = av_format_mov::Demuxer::open(settings.output()).unwrap();
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
            decoder.width = settings.output.width as i32;
            decoder.height = settings.output.height as i32;
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

    #[test]
    fn cuda_export_coordinator_samples_animation_and_writes_each_frame_once() {
        let _gpu_test = crate::test_gpu::lock();
        let dir = temp_dir("cuda");
        let service = RenderService::spawn();
        let mut controller = ExportController {
            out_root: dir.clone(),
            settings: ExportSettings {
                samples: 2,
                job: ExportJob {
                    first: 3,
                    last: 4,
                    ..ExportJob::default()
                },
                ..Default::default()
            },
            ..Default::default()
        };
        let mut scene = Scene::preset(crate::params::FAMILY_KIFS);
        scene.render.denoise.enabled = false; // This test checks exact physical animation output.
        scene.camera.target = [1000.0; 3];
        scene.lighting.sun_intensity = 0.0;
        scene.lighting.sky_intensity = 0.0;
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
            let path = "/lighting/sky_intensity".to_string();
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
                width: 16,
                height: 16,
                ..OutputSettings::default()
            },
        );
        scene.document = Some(Box::new(editor.document));
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
        assert!(
            controller.status.starts_with("Export complete"),
            "{}",
            controller.status
        );
        let written = controller.last.clone().unwrap();
        assert!(
            written.frame_path(3).starts_with(&dir),
            "exports land in a folder under the output root"
        );
        assert!(written.frame_path(3).exists());
        assert!(written.frame_path(4).exists());
        let pixel = |number| {
            crate::exr_io::read_rgb(&written.frame_path(number))
                .unwrap()
                .pixels[0]
        };
        assert_eq!(pixel(3)[0], 0.0);
        assert!(
            pixel(4)[0] > 0.0,
            "Each exported frame must evaluate its animation"
        );
        assert_eq!(
            std::fs::read_dir(written.frame_path(3).parent().unwrap())
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
        let service = RenderService::spawn();
        let mut controller = ExportController {
            out_root: dir.clone(),
            settings: ExportSettings {
                samples: 1,
                job: ExportJob {
                    name: "partial".into(),
                    last: 10000,
                    ..ExportJob::default()
                },
                ..Default::default()
            },
            ..Default::default()
        };
        let mut scene = Scene::preset(crate::params::FAMILY_KIFS);
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
                format: ExportFormat::Hevc,
                encoder: VideoEncoder::Vulkan,
                width: 256,
                height: 256,
                ..OutputSettings::default()
            },
        );
        scene.document = Some(Box::new(editor.document));
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
        let demux =
            av_format_mov::Demuxer::open(controller.last.clone().unwrap().output()).unwrap();
        assert_eq!(
            demux.presented_frame_count().unwrap(),
            controller.completed as usize
        );
        assert!(controller.completed > 0 && controller.completed < 10001);
        finish_fixture(dir);
    }

    #[test]
    fn each_format_renders_through_its_own_output_transform() {
        let _gpu_test = crate::test_gpu::lock();
        use crate::ocio::{Ocio, OutputKind};
        let ocio = Ocio::load("ocio://studio-config-latest").unwrap();
        let scene = crate::color::default_selection(); // SDR sRGB view in the viewport
        let mut s = ExportSettings {
            output: OutputSettings {
                format: ExportFormat::Png,
                ..OutputSettings::default()
            },
            ..Default::default()
        };
        let sdr = s.output.resolve_transform(&ocio, &scene).unwrap().unwrap();
        assert_eq!(
            (sdr.0.as_str(), sdr.1.as_str()),
            (scene.display.as_str(), scene.view.as_str())
        );
        s.output.png = PngEncoding::Hdr10;
        let (display, view) = s.output.resolve_transform(&ocio, &scene).unwrap().unwrap();
        assert!(
            ocio.display_is(&display, OutputKind::Pq) && view.contains("1000 nits"),
            "{display} / {view}"
        );
        s.output.png_peak_nits = 500.0;
        assert!(
            s.output
                .resolve_transform(&ocio, &scene)
                .unwrap()
                .unwrap()
                .1
                .contains("500 nits")
        );
        s.output.png = PngEncoding::Hlg;
        let (display, _) = s.output.resolve_transform(&ocio, &scene).unwrap().unwrap();
        assert!(display.contains("HLG"), "{display}");
        // EXR is scene-linear; an SDR file with OCIO off keeps the built-in display.
        s.output.format = ExportFormat::Exr;
        assert_eq!(s.output.resolve_transform(&ocio, &scene).unwrap(), None);
        s.output.format = ExportFormat::Hevc;
        let off = crate::ocio::Sel {
            on: false,
            ..scene.clone()
        };
        assert_eq!(s.output.resolve_transform(&ocio, &off).unwrap(), None);
        // The frame scene gets the resolved display / view.
        s.transform = Some((display.clone(), "view".into()));
        let mut frame = Scene::preset(crate::params::FAMILY_BULB);
        frame.colour.on = false;
        s.apply_transform(&mut frame);
        assert!(frame.colour.on && frame.colour.display == display);
    }
    #[test]
    fn sequence_paths_and_validation() {
        let _gpu_test = crate::test_gpu::lock();
        let mut s = ExportSettings::default();
        s.job.name = "image".into();
        s.resolve(Path::new("some folder"));
        s.job.first = 7;
        s.job.last = 9;
        assert_eq!(s.frame_count(), 3);
        assert_eq!(
            s.frame_path(8),
            Path::new("some folder").join("image.000008.exr")
        );
        assert!(s.validate().is_ok());
        // An HDR PNG sequence numbers before the whole suffix (`fs_name::frame_file`).
        s.output.format = ExportFormat::Png;
        s.output.png = PngEncoding::Hdr10;
        s.job.name = "shot".into();
        s.resolve(Path::new("out"));
        assert_eq!(s.frame_path(8), Path::new("out").join("shot.000008.pq.png"));
        s.output.png = PngEncoding::Sdr8;
        // A one-frame range is a still: the resolved file itself, no frame number.
        s.output.format = ExportFormat::Png;
        s.job.name = "still".into();
        s.resolve(Path::new("out"));
        s.job.first = 5;
        s.job.last = 5;
        assert_eq!(s.frame_path(5), Path::new("out").join("still.png"));
        assert!(s.validate().is_ok());
        s.output.png_peak_nits = 50.0;
        assert!(s.validate().unwrap_err().contains("peak"));
        s.output.png_peak_nits = 1000.0;
        s.job.name = "a/b".into();
        assert!(s.validate().unwrap_err().contains("cannot hold"));
        s.job.name = "CON".into();
        assert!(s.validate().unwrap_err().contains("device"));
        s.job.name = "frame".into();
        s.job.first = 7;
        s.job.last = 9;
        s.output.format = ExportFormat::Hevc;
        s.dir = PathBuf::from("out");
        s.output.width = 17;
        assert!(s.validate().unwrap_err().contains("even"));
        s.output.width = 64;
        s.output.height = 64;
        s.job.last = 6;
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
            let settings = ExportSettings {
                dir: dir.clone(),
                samples: 4,
                output: OutputSettings {
                    format,
                    width: 8,
                    height: 8,
                    sixteen_bit: true,
                    channels: Channels::Rgba,
                    jpeg_quality: 90,
                    tiff_compression: TiffCompression::Lzw,
                    exr_depth: ExrDepth::Half,
                    exr_compression: ExrCompression::Dwaa,
                    exr_dwa_level: 60.0,
                    ..OutputSettings::default()
                },
                job: ExportJob {
                    name: format!("still-{ext}"),
                    first: 1,
                    last: 1,
                    overwrite: true,
                    ..ExportJob::default()
                },
                ..Default::default()
            };
            assert_eq!(settings.suffix(), ext);
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
            let out = settings.output();
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
        for profile in ExvProfile::ALL {
            let settings = ExportSettings {
                dir: dir.clone(),
                output: OutputSettings {
                    format: ExportFormat::Hevc,
                    codec: Codec::Exv,
                    exv_profile: profile,
                    width: 16,
                    height: 16,
                    ..OutputSettings::default()
                },
                job: ExportJob {
                    name: format!("exv-{profile:?}"),
                    overwrite: true,
                    ..ExportJob::default()
                },
                ..Default::default()
            };
            assert_eq!(settings.suffix(), "exv");
            assert!(settings.output.validate().is_ok());
            let mut sink = HevcSink::new(&settings).unwrap();
            sink.write(&frame(16, 16)).unwrap();
            sink.finish(&AtomicBool::new(false)).unwrap();
            assert!(settings.output().exists());
        }
        for profile in ProResProfile::ALL {
            let settings = ExportSettings {
                dir: dir.clone(),
                output: OutputSettings {
                    format: ExportFormat::Hevc,
                    codec: Codec::ProRes,
                    prores_profile: profile,
                    width: 16,
                    height: 16,
                    ..OutputSettings::default()
                },
                job: ExportJob {
                    name: format!("prores-{profile:?}"),
                    overwrite: true,
                    ..ExportJob::default()
                },
                ..Default::default()
            };
            assert_eq!(settings.suffix(), "mov");
            assert!(settings.output.validate().is_ok());
            let mut sink = HevcSink::new(&settings).unwrap();
            sink.write(&frame(16, 16)).unwrap();
            sink.finish(&AtomicBool::new(false)).unwrap();
            assert!(settings.output().exists());
        }
        for codec in [Codec::H264, Codec::Av1] {
            let settings = ExportSettings {
                dir: dir.clone(),
                output: OutputSettings {
                    format: ExportFormat::Hevc,
                    codec,
                    encoder: VideoEncoder::Kvazaar,
                    width: 64,
                    height: 64,
                    ..OutputSettings::default()
                },
                job: ExportJob {
                    name: format!("sw-{codec:?}"),
                    overwrite: true,
                    ..ExportJob::default()
                },
                ..Default::default()
            };
            assert_eq!(settings.suffix(), "mp4");
            assert!(settings.output.validate().is_ok());
            let mut sink = HevcSink::new(&settings).unwrap();
            sink.write(&frame(64, 64)).unwrap();
            sink.finish(&AtomicBool::new(false)).unwrap();
            assert!(settings.output().exists());
        }
        finish_fixture(dir);
    }

    #[test]
    fn preset_files_round_trip_and_backend_hints_are_exposed() {
        assert!(!video_io::video::Backend::Native.hint().is_empty());
        assert!(
            video_io::video::Backend::Cli
                .hint()
                .contains("PNG sequence")
        );
        let dir = temp_dir("preset-files");
        let original = OutputSettings {
            format: ExportFormat::Hevc,
            encoder: VideoEncoder::Kvazaar,
            width: 3840,
            height: 2160,
            qp: 16,
            codec: Codec::Exv,
            exv_profile: ExvProfile::Htj2kLossy,
            prores_profile: ProResProfile::P4444,
            channels: Channels::Rgb,
            sixteen_bit: true,
            jpeg_quality: 92,
            tiff_compression: TiffCompression::Lzw,
            exr_depth: ExrDepth::Half,
            exr_compression: ExrCompression::Dwab,
            exr_dwa_level: 35.0,
            ..OutputSettings::default()
        };
        let path = dir.join("my_master_preset");
        let stem = original.save_preset_file(&path).unwrap();
        assert_eq!(stem, "my_master_preset");
        assert!(dir.join("my_master_preset.json").exists());

        let mut loaded = OutputSettings::default();
        let loaded_stem = loaded
            .load_preset_file(&dir.join("my_master_preset.json"))
            .unwrap();
        assert_eq!(loaded_stem, "my_master_preset");
        assert_eq!(loaded.format, ExportFormat::Hevc);
        assert_eq!(loaded.width, 3840);
        assert_eq!(loaded.height, 2160);
        assert_eq!(loaded.codec, Codec::Exv);
        assert_eq!(loaded.exv_profile, ExvProfile::Htj2kLossy);
        assert_eq!(loaded.exr_compression, ExrCompression::Dwab);
        assert_eq!(loaded.exr_dwa_level, 35.0);
        finish_fixture(dir);
    }
}
