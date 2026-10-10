//! Whole-window screenshots: the viewport and every egui panel, menu and status line, at the
//! window's physical resolution, from the frame the window just presented.
//!
//! **Capture.** The window composites everything into egui-display's float canvas
//! (`egui_display::CANVAS_FORMAT`, extended sRGB, 1.0 = SDR reference white) and encodes that canvas
//! for its surface in the present pass. A shot re-runs the same pass over the same canvas right
//! after the frame was presented (`egui_display::PresentPass::capture`), into an offscreen texture
//! instead of the swapchain, and reads it back on the file worker (`io_service`), never on the
//! window's thread.
//!
//! **Colour.** For the files of [`WindowFiles`] the canvas is captured as scRGB with the scRGB
//! white (80 nits), which makes the encoded signal exactly the canvas decoded to linear light:
//! linear BT.709 / D65, relative, 1.0 = SDR reference white, unclipped above 1 and below 0
//! ([`linear_light`]). Reference white in nits ([`crate::app::Monitor::sdr_white_nits`]): the
//! monitor's SDR white when the window shows HDR (so the files are as bright as the screen), else
//! ITU-R BT.2408's 203 nits ([`crate::color::BT2408_SDR_WHITE_NITS`]). The files go through the
//! encoders the viewport snapshot uses:
//! - EXR (`exr_io::write_rgb`): float RGB of that linear light, `chromaticities` BT.709 / D65,
//!   `whiteLuminance` = the reference white, so value x whiteLuminance = nits on screen.
//! - PQ PNG (`egui_display::export::write_png`, HDR10): 16-bit SMPTE ST 2084 of BT.2020 nits
//!   (reference white at those nits), `cICP` 9/16/0/1, `mDCV` and a measured `cLLI`.
//!
//! [`ShotKind::Displayed`] is the window's own surface signal instead (`FRAC_SNAP`, the README
//! images): an 8-bit sRGB PNG on an SDR window.

use std::fmt;
use std::path::{Path, PathBuf};

use egui_display::screenshot::{Capture, Pixels};
use egui_display::{Output, Target};

use crate::color::DisplayLight;
use crate::render_service::{PngEncoding, hdr_scale, write_png};

/// Which files a user's window screenshot writes (File menu, `F12` writes both).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowFiles {
    ExrAndPq,
    Exr,
    PqPng,
}

impl WindowFiles {
    /// Every choice, in menu order.
    pub const ALL: [Self; 3] = [Self::ExrAndPq, Self::Exr, Self::PqPng];

    /// Menu text.
    pub fn label(self) -> &'static str {
        match self {
            Self::ExrAndPq => "Window screenshot: EXR + PQ PNG",
            Self::Exr => "Window screenshot: EXR",
            Self::PqPng => "Window screenshot: PQ PNG",
        }
    }

    pub fn exr(self) -> bool {
        matches!(self, Self::ExrAndPq | Self::Exr)
    }

    pub fn pq_png(self) -> bool {
        matches!(self, Self::ExrAndPq | Self::PqPng)
    }
}

/// The file suffix of the linear window EXR (after the scene stem).
pub const EXR_SUFFIX: &str = "window.exr";

/// The file suffix of the PQ window PNG: `PngEncoding::Hdr10`'s, so it is named like every PQ PNG.
pub fn pq_suffix() -> String {
    format!("window.{}", PngEncoding::Hdr10.suffix())
}

/// What a shot writes and where.
#[derive(Clone, Debug, PartialEq)]
pub enum ShotKind {
    /// [`WindowFiles`] of the linear canvas into a new timestamped folder under `root`
    /// (`crate::new_out_dir`), named `<stem>.window.exr` / `<stem>.window.pq.png`.
    Linear {
        root: PathBuf,
        stem: String,
        files: WindowFiles,
        /// Nits of the canvas's 1.0 (the reference white, see the module docs).
        sdr_white_nits: f32,
    },
    /// The window's own surface signal at `path` (extension replaced: png, or exr for scRGB).
    Displayed { path: PathBuf },
}

/// One window screenshot request: armed by the app, captured by the window after the next
/// presented frame (so a menu that requested it is already closed), written by the file worker.
#[derive(Clone, Debug, PartialEq)]
pub struct WindowShot {
    pub kind: ShotKind,
    /// Close the window once the files are written (`FRAC_SNAP`).
    pub quit: bool,
}

impl WindowShot {
    /// The output the capture encodes the canvas for: scRGB (linear float) for the linear files,
    /// the window's own output (`shown`) for [`ShotKind::Displayed`].
    pub fn output(&self, shown: Output) -> Output {
        match self.kind {
            ShotKind::Linear { .. } => Output::Scrgb,
            ShotKind::Displayed { .. } => shown,
        }
    }

    /// The levels the capture encodes with: the scRGB white (80 nits) makes scRGB's
    /// `eotf(canvas) * white / 80` the bare decoded canvas; `Displayed` keeps the window's.
    pub fn target(&self, shown: Target) -> Target {
        match self.kind {
            ShotKind::Linear { .. } => Target {
                hdr: true,
                white: egui_display::SCRGB_NITS,
                peak: shown.peak,
            },
            ShotKind::Displayed { .. } => shown,
        }
    }
}

/// Why a window screenshot failed; shown in the status line and logged.
#[derive(Debug)]
pub enum ShotError {
    /// The capture could not be submitted or read back.
    Capture(egui_display::CaptureError),
    /// The read-back pixels are not what the request captures for.
    Pixels(String),
    /// The output folder could not be created.
    Folder(String),
    /// The file worker did not take the request (its queue is full or it has stopped).
    Worker(String),
    /// A file encoder failed.
    Write { path: PathBuf, message: String },
}

impl fmt::Display for ShotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Capture(e) => write!(f, "window capture: {e}"),
            Self::Pixels(e) => write!(f, "window capture pixels: {e}"),
            Self::Folder(e) => write!(f, "screenshot folder: {e}"),
            Self::Worker(e) => write!(f, "{e}"),
            Self::Write { path, message } => write!(f, "{}: {message}", path.display()),
        }
    }
}

impl std::error::Error for ShotError {}

impl From<egui_display::CaptureError> for ShotError {
    fn from(e: egui_display::CaptureError) -> Self {
        Self::Capture(e)
    }
}

/// The linear light of a scRGB capture made for [`ShotKind::Linear`]: RGBA, linear BT.709, 1.0 =
/// SDR reference white (see [`WindowShot::target`]). Any other pixel type or a sample count
/// that does not match the size is an error, never a padded or guessed image.
pub fn linear_light(capture: &Capture) -> Result<Vec<[f32; 4]>, ShotError> {
    let Pixels::RgbaF16(px) = &capture.pixels else {
        return Err(ShotError::Pixels(format!(
            "{:?} capture is not linear float",
            capture.output
        )));
    };
    let count = capture.width as usize * capture.height as usize;
    if count == 0 || px.len() != count * 4 {
        return Err(ShotError::Pixels(format!(
            "{} half samples for a {}x{} RGBA capture",
            px.len(),
            capture.width,
            capture.height
        )));
    }
    Ok(px
        .as_chunks::<4>()
        .0
        .iter()
        .map(|p| p.map(|h| h.to_f32()))
        .collect())
}

/// Write `files` of the linear `light` (`width` x `height`, see [`linear_light`]) as
/// `dir/<stem>.window.exr` and `dir/<stem>.window.pq.png`, the canvas's 1.0 at `sdr_white_nits`.
/// Returns the paths written. Existing files are refused (each shot has its own folder).
pub fn write_linear(
    dir: &Path,
    stem: &str,
    light: &[[f32; 4]],
    (width, height): (usize, usize),
    files: WindowFiles,
    sdr_white_nits: f32,
) -> Result<Vec<PathBuf>, ShotError> {
    let mut written = Vec::new();
    if files.exr() {
        let path = dir.join(crate::fs_name::frame_file(stem, None, EXR_SUFFIX));
        crate::exr_io::write_rgb(
            &path,
            width,
            height,
            light,
            &crate::color::DISPLAY_PRIMS,
            Some(sdr_white_nits),
            false,
        )
        .map_err(|message| ShotError::Write {
            path: path.clone(),
            message,
        })?;
        written.push(path);
    }
    if files.pq_png() {
        let path = dir.join(crate::fs_name::frame_file(stem, None, &pq_suffix()));
        let encoding = PngEncoding::Hdr10;
        let scale = hdr_scale(DisplayLight::Relative, light, sdr_white_nits, encoding);
        // An HDR10 PNG never reads the SDR codes.
        write_png(
            &path,
            width,
            height,
            light,
            Vec::new,
            encoding,
            scale,
            false,
        )
        .map_err(|message| ShotError::Write {
            path: path.clone(),
            message,
        })?;
        written.push(path);
    }
    Ok(written)
}

/// Write the files of `shot` from its read-back `capture` (the file worker). Returns the paths.
pub fn save(shot: &WindowShot, capture: &Capture) -> Result<Vec<PathBuf>, ShotError> {
    match &shot.kind {
        ShotKind::Linear {
            root,
            stem,
            files,
            sdr_white_nits,
        } => {
            let light = linear_light(capture)?;
            let dir = crate::new_out_dir(root).map_err(ShotError::Folder)?;
            write_linear(
                &dir,
                stem,
                &light,
                (capture.width as usize, capture.height as usize),
                *files,
                *sdr_white_nits,
            )
        }
        ShotKind::Displayed { path } => {
            capture
                .save(path)
                .map(|path| vec![path])
                .map_err(|e| ShotError::Write {
                    path: path.clone(),
                    message: e.to_string(),
                })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh scratch folder of this test process.
    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("warpbro-window-shot-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// The raw chunks of a PNG byte stream.
    fn chunks(png: &[u8]) -> Vec<([u8; 4], Vec<u8>)> {
        let mut out = Vec::new();
        let mut at = 8;
        while at + 8 <= png.len() {
            let len = u32::from_be_bytes(png[at..at + 4].try_into().unwrap()) as usize;
            let kind: [u8; 4] = png[at + 4..at + 8].try_into().unwrap();
            out.push((kind, png[at + 8..at + 8 + len].to_vec()));
            at += 12 + len;
        }
        out
    }

    /// The 16-bit RGBA samples of a PNG file.
    fn png16(path: &Path) -> Vec<u16> {
        let decoder =
            png::Decoder::new(std::io::BufReader::new(std::fs::File::open(path).unwrap()));
        let mut reader = decoder.read_info().unwrap();
        assert_eq!(reader.info().bit_depth, png::BitDepth::Sixteen);
        let mut buf = vec![0; reader.output_buffer_size().unwrap()];
        reader.next_frame(&mut buf).unwrap();
        buf.as_chunks::<2>()
            .0
            .iter()
            .map(|b| u16::from_be_bytes(*b))
            .collect()
    }

    /// The ST 2084 curve against its published values (SMPTE ST 2084 / BT.2100 constants,
    /// computed independently in f64), and the EOTF inverting it across the range.
    #[test]
    fn pq_curve_round_trips_at_known_nits() {
        let known = [
            (0.1, 0.062_336_9),
            (1.0, 0.149_945_7),
            (100.0, 0.508_078_4),
            (203.0, 0.580_688_9),
            (1000.0, 0.751_827_1),
            (4000.0, 0.902_572_4),
            (10_000.0, 1.0),
        ];
        for (nits, code) in known {
            let signal = egui_display::pq(nits);
            // f32 evaluation; 1e-5 is below one 16-bit PNG code (1.5e-5).
            assert!(
                (signal - code).abs() < 1e-5,
                "pq({nits}) = {signal}, want {code}"
            );
            let back = egui_display::pq_nits(signal);
            assert!(
                (back - nits).abs() <= nits * 1e-3,
                "pq_nits(pq({nits})) = {back}"
            );
        }
    }

    /// The linear writer: reference white at 203 nits lands on PQ 0.5807, a 1000-nit pixel
    /// (4.926 x white) on 0.7518 unclipped; the PNG says BT.2020 / PQ / RGB / full range in
    /// cICP, records mDCV and cLLI; the EXR keeps the light bit-exactly (above 1 and negative)
    /// with BT.709 chromaticities and whiteLuminance = the reference white.
    #[test]
    fn linear_files_carry_pq_cicp_and_exr_tags() {
        let dir = scratch("encoders");
        let white = crate::color::BT2408_SDR_WHITE_NITS;
        let bright = 1000.0 / white;
        let light = vec![
            [1.0, 1.0, 1.0, 1.0],
            [bright, bright, bright, 1.0],
            [-0.25, 0.0, 0.5, 1.0],
        ];
        let written =
            write_linear(&dir, "shot", &light, (3, 1), WindowFiles::ExrAndPq, white).unwrap();
        let exr = dir.join("shot.window.exr");
        let png = dir.join("shot.window.pq.png");
        assert_eq!(written, vec![exr.clone(), png.clone()]);

        let bytes = std::fs::read(&png).unwrap();
        let chunks = chunks(&bytes);
        let find = |kind: &[u8; 4]| chunks.iter().position(|(k, _)| k == kind);
        let cicp = find(b"cICP").expect("cICP");
        assert!(
            cicp < find(b"IDAT").expect("IDAT"),
            "cICP precedes the image data"
        );
        assert_eq!(chunks[cicp].1, [9, 16, 0, 1]);
        assert!(find(b"mDCV").is_some() && find(b"cLLI").is_some());
        assert!(find(b"sRGB").is_none(), "a PQ PNG is not tagged sRGB");
        let px = png16(&png);
        let nits = |i: usize| egui_display::pq_nits(f32::from(px[i]) / 65535.0);
        assert!((nits(0) - white).abs() < 0.5, "white at {} nits", nits(0));
        assert!(
            (nits(4) - 1000.0).abs() < 2.0,
            "HDR pixel at {} nits",
            nits(4)
        );
        assert_eq!(px[8], 0, "negative light is black in PQ");

        let back = crate::exr_io::read_rgb(&exr).unwrap();
        assert_eq!(
            back.pixels,
            light.iter().map(|p| [p[0], p[1], p[2]]).collect::<Vec<_>>()
        );
        use exr_core::attr::Chromaticities;
        assert_eq!(
            crate::exr_io::read_attr::<Chromaticities>(&exr, "chromaticities"),
            Some(Chromaticities::default()),
            "BT.709 / D65"
        );
        assert_eq!(
            crate::exr_io::read_attr::<f32>(&exr, "whiteLuminance"),
            Some(white)
        );

        assert!(
            write_linear(&dir, "shot", &light, (3, 1), WindowFiles::Exr, white).is_err(),
            "an existing file is refused"
        );
        let only = write_linear(&dir, "pq", &light, (3, 1), WindowFiles::PqPng, white).unwrap();
        assert_eq!(only, vec![dir.join("pq.window.pq.png")]);
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// A linear capture decodes half floats as they are; any other pixel type or a short
    /// sample count is refused.
    #[test]
    fn linear_light_reads_only_matching_float_captures() {
        let h = |v: f32| half::f16::from_f32(v);
        let capture = |pixels| Capture {
            output: Output::Scrgb,
            width: 2,
            height: 1,
            white_nits: egui_display::SCRGB_NITS,
            peak_nits: 1000.0,
            pixels,
        };
        let light = linear_light(&capture(Pixels::RgbaF16(
            [0.5, 2.0, -0.25, 1.0, 0.0, 12.0, 1.0, 1.0].map(h).to_vec(),
        )))
        .unwrap();
        assert_eq!(light, vec![[0.5, 2.0, -0.25, 1.0], [0.0, 12.0, 1.0, 1.0]]);
        assert!(linear_light(&capture(Pixels::RgbaF16(vec![h(0.0); 4]))).is_err());
        assert!(linear_light(&capture(Pixels::Rgba8(vec![0; 8]))).is_err());
    }

    /// Linear shots capture scRGB at the scRGB white (the bare decoded canvas); a displayed shot
    /// keeps the window's output and levels.
    #[test]
    fn shots_choose_their_capture() {
        let shown = Target {
            hdr: true,
            white: 240.0,
            peak: 800.0,
        };
        let linear = WindowShot {
            kind: ShotKind::Linear {
                root: PathBuf::new(),
                stem: "s".into(),
                files: WindowFiles::ExrAndPq,
                sdr_white_nits: 240.0,
            },
            quit: false,
        };
        assert_eq!(linear.output(Output::Hdr10), Output::Scrgb);
        let target = linear.target(shown);
        assert_eq!(target.white, egui_display::SCRGB_NITS);
        let canvas = 0.73;
        let encoded = egui_display::encode(
            [canvas; 3],
            Output::Scrgb.encoding(wgpu::TextureFormat::Rgba16Float),
            target,
        );
        assert_eq!(encoded[0], egui_display::transfer::eotf(canvas));
        let displayed = WindowShot {
            kind: ShotKind::Displayed {
                path: PathBuf::from("x.png"),
            },
            quit: true,
        };
        assert_eq!(displayed.output(Output::Hdr10), Output::Hdr10);
        assert_eq!(displayed.target(shown), shown);
    }
}
