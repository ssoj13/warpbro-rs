//! Adapted from exr-view/src/ocio.rs.
//! The OCIO display transform of the colour image: the config, the user's choice
//! (display, view, look), the transform the tile shader and the CPU twin both run, and
//! the Colour panel. The input is always the tracer's working space
//! (`color::WORKING`): a renderer fact, not a choice, so it is never persisted.
//!
//! Everything comes from `vfx-ocio` (the OpenColorIO 2.5 port Playa and vfx-view use):
//! ONE `Processor` gives the WGSL spliced into the tile shader (`wgsl_prelude`, bound
//! through `GpuOcioResources::build_for(.., FRAGMENT)` in `tiles::render`) and the
//! exact CPU twin (`apply_rgba`) that Save view and the GPU test use.
//!
//! The chain in colour mode (`channel_mode == RGBA`): the exposure gain on the input
//! values, the OCIO display/view transform (with its look), then the gamma dial on the
//! display-encoded result (where OCIO's own `ociodisplay` applies its display gamma:
//! an `ExponentTransform` after the display/view transform). No sRGB OETF: the
//! display colour space already encodes. Isolated channels are data and keep the
//! built-in path. The exposure is a gain on the input (linear ACEScg) values, which is
//! scene-linear exposure.
//!
//! What the transform ends with follows the window's output (`present::Output`):
//!
//! - **SDR output: encoded.** The display's own code values go to the canvas as
//!   they are (the user picks the display that matches the monitor, as in any OCIO
//!   viewer). Only SDR displays are offered: a view is hidden when its display
//!   colour space says `encoding: hdr-video` (one without an encoding, OCIO v1, is
//!   offered), and a display needs a non-data SDR view (in OCIO v2 a shared view
//!   renders into the DISPLAY's space, so a PQ display's "SDR" views are PQ too).
//! - **HDR output: linear.** Every display is offered. One processor runs the
//!   display/view transform, then the display colour space back to OCIO's display
//!   reference (linear CIE XYZ D65, 1.0 = 100 nits, the config's own
//!   `to_display_reference` or inverted `from_display_reference`), then XYZ to
//!   linear Rec.709: linear light with 1.0 = SDR reference white, which the tile
//!   shader encodes like the built-in chain and the present pass maps to the
//!   surface. A data view (`Raw`) passes its values through as that light.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context as _, Result, anyhow, bail};
pub use egui_display::export::OutputKind;
use egui_display::export::{OutputView, OutputViewEncoding, resolve_output_view};
use serde::{Deserialize, Serialize};
use vfx_ocio::builtin::embedded;
use vfx_ocio::color_matrix::{Adaptation, REC709, conversion_matrix_from_xyz_d65};
use vfx_ocio::{
    DisplayViewTransform, Encoding, GpuLanguage, GpuProcessor, GpuShaderCode, GroupTransform,
    MatrixTransform, Processor, ReferenceSpaceType, TransformDirection,
};

/// What the user picked; persisted (`persist::Config::ocio`). Empty names mean the
/// config's defaults ([`Ocio::resolve`]).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sel {
    /// OCIO is the display transform of the colour image (else the built-in one).
    pub on: bool,
    /// A config file or an `ocio://` URI; empty = `$OCIO`, else `ocio://default`.
    pub config: String,
    /// Display; empty = the first one offered.
    pub display: String,
    /// View; empty = the display's first one offered.
    pub view: String,
    /// Look applied instead of the view's own looks; empty = the view's looks.
    pub look: String,
    /// This config's name for the tracer's working space (linear AP1 / ACEScg); empty = found
    /// automatically ([`Ocio::working_space`]). Not `input`: scenes saved while the input was a
    /// free choice stored "Linear Rec.709", which would now shift every colour.
    pub working_input: String,
}

/// The config `sel_config` stands for: itself, else `$OCIO` (the standard OCIO
/// variable), else OCIO's default built-in (the ACES 2.0 CG config).
pub fn source(sel_config: &str) -> String {
    if !sel_config.trim().is_empty() {
        return sel_config.trim().to_owned();
    }
    std::env::var("OCIO")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "ocio://default".to_owned())
}

/// Names a [`Sel`] resolves to in one config.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Names {
    pub input: String,
    pub display: String,
    pub view: String,
    pub look: String,
}

/// A loaded config.
pub struct Ocio {
    /// What it was loaded from ([`source`]).
    pub src: String,
    /// Which load this is (process-wide counter): part of every transform key, so a
    /// reload of the same source rebuilds the tile pipeline.
    serial: u64,
    cfg: vfx_ocio::Config,
    /// [`Ocio::working_space`], found once per load.
    working: std::sync::OnceLock<Option<String>>,
    /// Measured peaks keyed by every resolved processor input, within this config load:
    /// building a view's processor to probe it is too slow to repeat every UI frame.
    peaks: std::sync::Mutex<std::collections::HashMap<Names, Option<f32>>>,
}

/// The load counter behind [`Ocio::serial`].
static LOADS: AtomicU64 = AtomicU64::new(0);

impl Ocio {
    /// Load `src` (a file or an `ocio://` URI).
    pub fn load(src: &str) -> Result<Self> {
        let cfg = vfx_ocio::Config::from_file(src).with_context(|| format!("OCIO config {src}"))?;
        log::info!(
            "OCIO config {src}: {} colour spaces",
            cfg.colorspaces().len()
        );
        Ok(Self {
            src: src.to_owned(),
            serial: LOADS.fetch_add(1, Ordering::Relaxed),
            cfg,
            working: std::sync::OnceLock::new(),
            peaks: Default::default(),
        })
    }

    /// Every colour space, config order.
    pub fn inputs(&self) -> Vec<&str> {
        self.cfg.colorspace_names().collect()
    }

    /// Whether colour space `name` is linear AP1 (ACEScg): its transform to the config's
    /// `aces_interchange` role (ACES2065-1) must be the AP1 -> AP0 matrix. None when the config
    /// has no such role (OCIO v1 / non-ACES configs), so the transform cannot be checked.
    pub fn is_working(&self, name: &str) -> Option<bool> {
        if !self.cfg.has_role("aces_interchange") {
            return None;
        }
        let Ok(processor) = self.cfg.processor(name, "aces_interchange") else {
            return Some(false);
        };
        let mut probe = [
            [1.0f32, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.18, 0.5, 2.0],
        ];
        let want = probe.map(crate::color::to_ap0);
        processor.apply_rgb(&mut probe);
        Some(probe.iter().zip(want).all(|(got, want)| {
            got.iter()
                .zip(want)
                .all(|(g, w)| (g - w).abs() <= 1e-4 * w.abs().max(1.0))
        }))
    }

    /// This config's name for the tracer's working space: the space whose transform is linear
    /// AP1 ([`Self::is_working`]), else (configs without `aces_interchange`) the first of the
    /// ACES names ([`crate::color::WORKING_NAMES`]) the config knows. Found once per load.
    pub fn working_space(&self) -> Option<&str> {
        self.working
            .get_or_init(|| {
                let by_transform = self.cfg.has_role("aces_interchange").then(|| {
                    // ACES names first: the transform check then usually succeeds on the first try.
                    crate::color::WORKING_NAMES
                        .iter()
                        .copied()
                        .filter(|n| self.cfg.colorspace(n).is_some())
                        .chain(self.cfg.colorspace_names())
                        .find(|n| self.is_working(n) == Some(true))
                        .map(str::to_owned)
                });
                match by_transform {
                    Some(found) => found,
                    None => crate::color::WORKING_NAMES
                        .iter()
                        .find(|n| self.cfg.colorspace(n).is_some())
                        .map(|n| (*n).to_owned()),
                }
            })
            .as_deref()
    }

    /// The active displays offered: with `hdr` every one with a picture (non-data)
    /// view; without, those with a non-data SDR view. A data view (`Raw`) does not
    /// count: every display has one.
    pub fn displays(&self, hdr: bool) -> Vec<&str> {
        self.cfg
            .display_names()
            .into_iter()
            .filter(|d| {
                self.cfg.get_views(d).into_iter().any(|v| {
                    let enc = self.encoding(d, v);
                    enc != Some(Encoding::Data) && (hdr || enc != Some(Encoding::Hdr))
                })
            })
            .collect()
    }

    /// The views of `display` offered (config order): all with `hdr`, else the
    /// non-HDR ones (data views included).
    pub fn views(&self, display: &str, hdr: bool) -> Vec<&str> {
        self.cfg
            .get_views(display)
            .into_iter()
            .filter(|v| hdr || self.encoding(display, v) != Some(Encoding::Hdr))
            .map(|v| v.name())
            .collect()
    }

    /// The encoding of the colour space `view` of `display` renders into; `None`
    /// when it names none (OCIO v1 configs), which is offered as SDR.
    fn encoding(&self, display: &str, view: &vfx_ocio::View) -> Option<Encoding> {
        let enc = self
            .cfg
            .colorspace(view.effective_colorspace(display))?
            .encoding();
        (enc != Encoding::Unknown).then_some(enc)
    }

    /// The encoding of view `view` of `display`, by name.
    fn view_encoding(&self, display: &str, view: &str) -> Option<Encoding> {
        let v = self
            .cfg
            .get_views(display)
            .into_iter()
            .find(|v| v.name() == view)?;
        self.encoding(display, v)
    }

    /// Whether `display` is a `kind` display. OCIO records only "hdr-video", not PQ or HLG, so
    /// HDR displays are told apart by their names (the ACES configs' convention, as Nuke and
    /// Resolve do); the export panel shows the choice and lets it be overridden.
    pub fn display_is(&self, display: &str, kind: OutputKind) -> bool {
        kind.display_is(display)
    }

    /// Whether view `view` of `display` renders for `kind` (HDR views for PQ / HLG, picture SDR
    /// views for SDR).
    pub fn view_is(&self, display: &str, view: &str, kind: OutputKind) -> bool {
        self.output_view(display, view).encoding.fits(kind)
    }

    fn output_view<'a>(&self, display: &'a str, view: &'a str) -> OutputView<'a> {
        OutputView {
            display,
            view,
            encoding: match self.view_encoding(display, view) {
                Some(Encoding::Hdr) => OutputViewEncoding::Hdr,
                Some(Encoding::Data) => OutputViewEncoding::Data,
                _ => OutputViewEncoding::Picture,
            },
        }
    }

    /// Display and view an output of `kind` renders through. The scene's own choice (`current`)
    /// wins when it already fits; otherwise the config's first fitting display, with its view
    /// whose measured peak ([`Ocio::view_peak`]) is closest to `peak_nits` for HDR, its first
    /// view for SDR.
    pub fn output_transform(
        &self,
        current: &Sel,
        kind: OutputKind,
        peak_nits: f32,
    ) -> Result<(String, String)> {
        let hdr = kind != OutputKind::Sdr;
        let names = self.resolve(current, hdr).ok();
        let mut candidates = Vec::new();
        for display in self.displays(hdr) {
            for view in self.views(display, hdr) {
                candidates.push(self.output_view(display, view));
            }
        }
        let selected = resolve_output_view(
            names
                .as_ref()
                .map(|n| (n.display.as_str(), n.view.as_str())),
            kind,
            peak_nits,
            &candidates,
            |display, view| self.view_peak(current, display, view),
        )
        .map_err(anyhow::Error::msg)?;
        Ok((selected.display.to_owned(), selected.view.to_owned()))
    }

    /// The measured peak in nits of HDR `view` on `display` (with `current`'s input and look),
    /// cached per load; None when it is not an HDR view or cannot be built / measured.
    fn view_peak(&self, current: &Sel, display: &str, view: &str) -> Option<f32> {
        let sel = Sel {
            display: display.into(),
            view: view.into(),
            ..current.clone()
        };
        let names = self.resolve(&sel, true).ok()?;
        if let Ok(peaks) = self.peaks.lock()
            && let Some(peak) = peaks.get(&names)
        {
            return *peak;
        }
        let peak = self
            .transform(&names, true)
            .ok()
            .and_then(|t| t.light().ok())
            .and_then(|light| match light {
                crate::color::DisplayLight::Absolute { peak_nits } => Some(peak_nits),
                crate::color::DisplayLight::Relative => None,
            });
        if let Ok(mut peaks) = self.peaks.lock() {
            peaks.insert(names, peak);
        }
        peak
    }

    /// Every look.
    pub fn looks(&self) -> Vec<&str> {
        self.cfg.looks().names().collect()
    }

    /// `sel`'s names in this config, empty ones replaced by the defaults. A name the
    /// config lacks is an error, not a silent substitute: the panel says which.
    pub fn resolve(&self, sel: &Sel, hdr: bool) -> Result<Names> {
        let input = if sel.working_input.is_empty() {
            self.working_space()
                .ok_or_else(|| {
                    anyhow!("the config has no linear AP1 (ACEScg) colour space: pick it in Input")
                })?
                .to_owned()
        } else if self.cfg.colorspace(&sel.working_input).is_some() {
            sel.working_input.clone()
        } else {
            bail!("the config has no colour space \"{}\"", sel.working_input);
        };
        // What a name is missing as: an HDR display on an SDR output is "no SDR display".
        let kind = if hdr { "" } else { "SDR " };
        let displays = self.displays(hdr);
        let display = if sel.display.is_empty() {
            displays
                .first()
                .copied()
                .ok_or_else(|| anyhow!("the config has no {kind}display"))?
        } else if displays.contains(&sel.display.as_str()) {
            sel.display.as_str()
        } else {
            bail!("the config has no {kind}display \"{}\"", sel.display);
        };
        let views = self.views(display, hdr);
        let view = if sel.view.is_empty() {
            views
                .first()
                .copied()
                .ok_or_else(|| anyhow!("\"{display}\" has no {kind}view"))?
        } else if views.contains(&sel.view.as_str()) {
            sel.view.as_str()
        } else {
            bail!("\"{display}\" has no {kind}view \"{}\"", sel.view);
        };
        if !sel.look.is_empty() && self.cfg.looks().get(&sel.look).is_none() {
            bail!("the config has no look \"{}\"", sel.look);
        }
        Ok(Names {
            input,
            display: display.to_owned(),
            view: view.to_owned(),
            look: sel.look.clone(),
        })
    }

    /// The display/view transform of `names` (OCIO `DisplayViewTransform`; a look
    /// replaces the view's own looks, OCIO's looks override); `linear` continues it
    /// to linear Rec.709 light (see the module doc).
    pub fn transform(&self, names: &Names, linear: bool) -> Result<Transform> {
        let dvt = DisplayViewTransform {
            src: names.input.clone(),
            display: names.display.clone(),
            view: names.view.clone(),
            looks_override: names.look.clone(),
            looks_override_enabled: !names.look.is_empty(),
            ..DisplayViewTransform::default()
        };
        let proc = if linear {
            let mut chain = vec![vfx_ocio::Transform::DisplayView(dvt)];
            chain.extend(self.to_display_light(&names.display, &names.view)?);
            let group = vfx_ocio::Transform::Group(GroupTransform {
                name: String::new(),
                transforms: chain,
                direction: TransformDirection::Forward,
            });
            self.cfg
                .processor_from_transform(&group, TransformDirection::Forward)?
        } else {
            self.cfg.processor_for_display_view_transform(&dvt)?
        };
        let key = format!(
            "{}#{}|{}|{}|{}|{}|{}",
            self.src,
            self.serial,
            names.input,
            names.display,
            names.view,
            names.look,
            if linear { "linear" } else { "encoded" }
        );
        // Absolute light: the view renders into an HDR display colour space.
        let absolute = linear
            && self
                .cfg
                .get_views(&names.display)
                .into_iter()
                .find(|v| v.name() == names.view)
                .is_some_and(|v| self.encoding(&names.display, v) == Some(Encoding::Hdr));
        Transform::new(proc, key, linear, absolute)
    }

    /// The transforms that take `display`/`view`'s output to linear Rec.709 light:
    /// its colour space to the display reference (CIE XYZ D65, 1.0 = 100 nits), then
    /// XYZ to Rec.709. None for a data view (its values are the light). A
    /// scene-referred view colour space (OCIO v1 configs) has no display reference:
    /// an error, not a guess at its curve.
    fn to_display_light(&self, display: &str, view: &str) -> Result<Vec<vfx_ocio::Transform>> {
        let v = self
            .cfg
            .get_views(display)
            .into_iter()
            .find(|v| v.name() == view)
            .ok_or_else(|| anyhow!("\"{display}\" has no view \"{view}\""))?;
        let name = v.effective_colorspace(display);
        let cs = self
            .cfg
            .colorspace(name)
            .ok_or_else(|| anyhow!("the config has no colour space \"{name}\""))?;
        if cs.is_data() {
            return Ok(Vec::new());
        }
        if cs.reference_space_type() != ReferenceSpaceType::Display {
            bail!(
                "HDR output needs a display-referred view colour space; \"{name}\" is scene-referred (an OCIO v1 config?)"
            );
        }
        let mut out: Vec<vfx_ocio::Transform> = cs
            .to_display_reference()
            .cloned()
            .or_else(|| cs.from_display_reference().map(|t| t.clone().inverse()))
            .into_iter()
            .collect();
        // OCIO's own matrix, that of its builtin `DISPLAY - CIE-XYZ-D65_to_sRGB`.
        out.push(vfx_ocio::Transform::Matrix(MatrixTransform {
            name: String::new(),
            matrix: conversion_matrix_from_xyz_d65(&REC709, Adaptation::None)?,
            offset: [0.0; 4],
            direction: TransformDirection::Forward,
        }));
        Ok(out)
    }
}

/// One built display transform: the processor (CPU twin) and its WGSL.
pub struct Transform {
    /// What built it; the tile pipeline rebuilds when it changes.
    pub key: String,
    proc: Processor,
    /// `fn ocio_transform(vec4<f32>) -> vec4<f32>` with its bindings (groups 1 and 2).
    pub prelude: String,
    /// The uniforms and LUTs `GpuOcioResources` binds.
    pub code: GpuShaderCode,
    /// It ends in linear Rec.709 light (HDR output), not in display code values.
    linear: bool,
    /// Its light is ABSOLUTE (an HDR display colour space, e.g. PQ: display
    /// reference 1.0 = 100 nits), not relative to SDR white: see [`Self::gain`].
    absolute: bool,
}

/// Scene light (working space) far above every view's range, so a tone-mapped view returns its
/// peak for it (ACES 2.0's tone scale is asymptotic: at 1e5 it is at its peak to f32 precision).

impl Transform {
    pub(crate) fn processor(&self) -> &Processor {
        &self.proc
    }
    pub(crate) fn absolute(&self) -> bool {
        self.absolute
    }
    /// Compile `proc` for the GPU; `linear` says it ends in linear light,
    /// `absolute` that the light is in absolute nits (see [`Self::gain`]).
    pub fn new(proc: Processor, key: String, linear: bool, absolute: bool) -> Result<Self> {
        let gpu = GpuProcessor::from_processor_wgsl(&proc)?;
        let code = gpu.generate_shader(GpuLanguage::Wgsl)?;
        Ok(Self {
            key,
            prelude: gpu.wgsl_prelude(),
            code,
            proc,
            linear,
            absolute,
        })
    }

    /// What its light is: relative to SDR white, or the absolute light of an HDR view with the
    /// view's peak, measured on the transform itself: a probe far above any view's range goes
    /// through its processor, and the view's tone curve returns its peak. A view whose probe
    /// exceeds what PQ can carry (10 000 nits) has no tone curve to measure - an error, not a
    /// peak. Only for linear transforms (`Ocio::transform(.., true)`).
    pub fn light(&self) -> Result<crate::color::DisplayLight, String> {
        if !self.absolute {
            return Ok(crate::color::DisplayLight::Relative);
        }
        let peak_nits =
            egui_display::export::measure_display_peak(|probe| self.proc.apply_rgb(probe), 100.0)?;
        Ok(crate::color::DisplayLight::Absolute { peak_nits })
    }

    /// The gain from the transform's light to the canvas (1.0 = SDR white of
    /// `white` nits). An SDR view's light is relative (its 1.0 IS SDR white, as the
    /// OS shows SDR content on an HDR display): 1. An HDR view's light is absolute
    /// (display reference 1.0 = 100 nits; a 1000-nit view peaks at 10): 100/white,
    /// so the present pass, which scales canvas 1.0 to `white` nits, lands on the
    /// nits the view meant.
    pub fn gain(&self, white: f32) -> f32 {
        if self.absolute {
            100.0 / white.max(1.0)
        } else {
            1.0
        }
    }

    /// It ends in linear Rec.709 light, which the tile shader encodes.
    pub fn linear(&self) -> bool {
        self.linear
    }

    /// Whether it samples LUT textures (`Rgba32Float` with a filtering sampler, which
    /// needs `wgpu::Features::FLOAT32_FILTERABLE`).
    pub fn has_luts(&self) -> bool {
        !self.code.textures().is_empty()
    }

    /// CPU twin of the tile shader's OCIO colour branch over linear RGBA pixels:
    /// the exposure gain, the transform, for a linear transform `gain`
    /// ([`Self::gain`]), the gamma dial and, for a linear transform, the canvas
    /// encoding; alpha stays the input's.
    pub fn display(&self, px: &mut [[f32; 4]], exposure_ev: f32, gamma: f32, gain: f32) {
        let mut alpha = vec![0.0; px.len()];
        self.display_kernel(px, &mut alpha, exposure_ev, gamma, gain);
    }

    /// The exporter supplies admitted reusable alpha storage alongside its row.
    pub(crate) fn display_scratch(
        &self,
        px: &mut [[f32; 4]],
        alpha: &mut [f32],
        exposure_ev: f32,
        gamma: f32,
        gain: f32,
    ) -> Result<(), &'static str> {
        if alpha.len() != px.len() {
            return Err("OCIO alpha scratch must match the pixel row");
        }
        self.display_kernel(px, alpha, exposure_ev, gamma, gain);
        Ok(())
    }

    fn display_kernel(
        &self,
        px: &mut [[f32; 4]],
        alpha: &mut [f32],
        exposure_ev: f32,
        gamma: f32,
        gain: f32,
    ) {
        use rayon::prelude::*;
        let mult = 2.0_f32.powf(exposure_ev);
        px.par_iter_mut()
            .zip(alpha.par_iter_mut())
            .for_each(|(p, a)| {
                *a = p[3];
                // Alpha is restored verbatim below. A NaN alpha must not enter RGB
                // matrix operations, where even a zero alpha coefficient propagates it.
                p[3] = 1.0;
                for c in &mut p[..3] {
                    *c *= mult;
                }
            });
        self.proc.apply_rgba(px);
        let linear = self.linear;
        px.par_iter_mut().zip(alpha.par_iter()).for_each(|(p, &a)| {
            for c in &mut p[..3] {
                *c = if linear {
                    crate::transfer::oetf(crate::transfer::dial(*c * gain, gamma))
                } else {
                    crate::transfer::dial(*c, gamma)
                };
            }
            p[3] = a;
        });
    }
}

/// A replaceable colour request; stale selections never accumulate in a queue.
struct ColourRequest {
    generation: u64,
    reload: u64,
    src: String,
    sel: Sel,
    hdr: bool,
    filterable: bool,
}
type LoadedConfig = Result<Arc<Ocio>, String>;
struct ColourResult {
    generation: u64,
    src: String,
    config: LoadedConfig,
    active: Result<Option<Arc<Transform>>, String>,
}
#[derive(Default)]
struct ColourMailbox {
    request: Option<ColourRequest>,
    result: Option<ColourResult>,
    stop: bool,
}
struct ColourWorker {
    mailbox: Arc<(std::sync::Mutex<ColourMailbox>, std::sync::Condvar)>,
    latest: Arc<AtomicU64>,
}
impl ColourWorker {
    fn new() -> std::io::Result<Self> {
        let mailbox = Arc::new((
            std::sync::Mutex::new(ColourMailbox::default()),
            std::sync::Condvar::new(),
        ));
        let latest = Arc::new(AtomicU64::new(0));
        let worker_mailbox = mailbox.clone();
        let worker_latest = latest.clone();
        std::thread::Builder::new().name("frac-colour".into()).spawn(move || {
            let mut cached: Option<(String, u64, LoadedConfig)> = None;
            loop {
                let request = {
                    let (mutex, wake) = &*worker_mailbox;
                    let mut slot = mutex.lock().unwrap_or_else(|e| e.into_inner());
                    while slot.request.is_none() && !slot.stop {
                        slot = wake.wait(slot).unwrap_or_else(|e| e.into_inner());
                    }
                    if slot.stop { return; }
                    slot.request.take().unwrap()
                };
                if cached.as_ref().is_none_or(|(src, reload, _)| src != &request.src || *reload != request.reload) {
                    let loaded = Ocio::load(&request.src).map(Arc::new).map_err(|e| format!("{e:#}"));
                    cached = Some((request.src.clone(), request.reload, loaded));
                }
                if worker_latest.load(Ordering::Acquire) != request.generation { continue; }
                let config = cached.as_ref().unwrap().2.clone();
                let active = if !request.sel.on {
                    Ok(None)
                } else {
                    (|| {
                        let ocio = config.as_ref().map_err(Clone::clone)?;
                        let names = ocio.resolve(&request.sel, request.hdr).map_err(|e| format!("{e:#}"))?;
                        let transform = ocio.transform(&names, request.hdr).map_err(|e| format!("{e:#}"))?;
                        if transform.has_luts() && !request.filterable {
                            return Err("this transform samples LUT textures, and this GPU cannot filter 32-bit float textures".into());
                        }
                        Ok(Some(Arc::new(transform)))
                    })()
                };
                let mut slot = worker_mailbox.0.lock().unwrap_or_else(|e| e.into_inner());
                if slot.stop { return; }
                if worker_latest.load(Ordering::Acquire) == request.generation {
                    slot.result = Some(ColourResult {
                        generation: request.generation, src: request.src, config, active,
                    });
                }
            }
        })?;
        Ok(Self { mailbox, latest })
    }
    fn submit(&self, request: ColourRequest) {
        self.latest.store(request.generation, Ordering::Release);
        let mut slot = self.mailbox.0.lock().unwrap_or_else(|e| e.into_inner());
        slot.request = Some(request);
        slot.result = None;
        self.mailbox.1.notify_one();
    }
    fn take_ready(&self) -> Option<ColourResult> {
        self.mailbox.0.try_lock().ok()?.result.take()
    }
}
impl Drop for ColourWorker {
    fn drop(&mut self) {
        let mut slot = self.mailbox.0.lock().unwrap_or_else(|e| e.into_inner());
        slot.stop = true;
        slot.request = None;
        self.mailbox.1.notify_one();
        // No join on the UI: the worker exits after its current finite build.
    }
}

/// The Colour panel's state. File loading and processor/WGSL construction run
/// on its worker; the UI consumes only ready metadata and transforms.
pub struct State {
    pub sel: Sel,
    pub presets: ColourPresets,
    preset_edit: Option<PresetEdit>,
    filterable: bool,
    hdr: bool,
    cfg: Option<(String, LoadedConfig)>,
    active: Option<Arc<Transform>>,
    pub err: Option<String>,
    worker: Option<ColourWorker>,
    generation: u64,
    reload_serial: u64,
    pending: bool,
}

impl State {
    pub fn new(sel: Sel) -> Self {
        let (worker, err) = match ColourWorker::new() {
            Ok(worker) => (Some(worker), None),
            Err(error) => (
                None,
                Some(format!("Colour worker could not start: {error}")),
            ),
        };
        let mut state = Self {
            sel,
            presets: Default::default(),
            preset_edit: None,
            filterable: true,
            hdr: false,
            cfg: None,
            active: None,
            err,
            worker,
            generation: 0,
            reload_serial: 0,
            pending: false,
        };
        state.rebuild();
        state
    }

    pub fn set_filterable(&mut self, filterable: bool) {
        if self.filterable != filterable {
            self.filterable = filterable;
            self.rebuild();
        }
    }

    pub fn set_hdr(&mut self, hdr: bool) {
        if self.hdr != hdr {
            self.hdr = hdr;
            self.rebuild();
        }
    }

    pub fn active(&self) -> Option<&Arc<Transform>> {
        self.active.as_ref()
    }

    /// Schedule the latest selection without loading files or compiling colour
    /// processors on the caller. Until completion, no stale transform is exposed.
    pub fn rebuild(&mut self) {
        let Some(worker) = &self.worker else {
            return;
        };
        self.generation = self.generation.wrapping_add(1);
        self.err = None;
        self.active = None;
        self.pending = true;
        let src = source(&self.sel.config);
        if self
            .cfg
            .as_ref()
            .is_some_and(|(current, _)| current != &src)
        {
            self.cfg = None;
        }
        worker.submit(ColourRequest {
            generation: self.generation,
            reload: self.reload_serial,
            src,
            sel: self.sel.clone(),
            hdr: self.hdr,
            filterable: self.filterable,
        });
    }

    /// Consume an already-ready event. Never waits for a build or a worker lock.
    pub fn poll(&mut self) -> bool {
        let Some(result) = self.worker.as_ref().and_then(ColourWorker::take_ready) else {
            return false;
        };
        if result.generation != self.generation {
            return false;
        }
        self.pending = false;
        self.cfg = Some((result.src, result.config));
        match result.active {
            Ok(active) => {
                self.active = active;
                self.err = None;
            }
            Err(error) => {
                self.active = None;
                self.err = Some(error);
            }
        }
        true
    }

    /// The loaded config, or why there is none yet.
    pub fn config(&self) -> Result<&Ocio, String> {
        match &self.cfg {
            Some((_, Ok(config))) => Ok(config.as_ref()),
            Some((_, Err(error))) => Err(error.clone()),
            None => Err("Loading colour configuration…".into()),
        }
    }

    /// Use another config: the names of the old one mean nothing there, so they go
    /// back to the new config's defaults.
    pub fn set_config(&mut self, config: String) {
        self.sel = Sel {
            on: self.sel.on,
            config,
            ..Sel::default()
        };
        self.rebuild();
    }

    /// Switch OCIO on or off.
    pub fn set_on(&mut self, on: bool) {
        self.sel.on = on;
        self.rebuild();
    }

    /// Read the config again (it changed on disk, or a failed load was fixed) and
    /// rebuild; the new load serial makes the tile pipeline rebuild too.
    pub fn reload(&mut self) {
        self.cfg = None;
        self.reload_serial = self.reload_serial.wrapping_add(1);
        self.rebuild();
    }

    /// Compact direct View control; lists use the already-loaded metadata only.
    pub fn quick_view_ui(&mut self, ui: &mut egui::Ui) -> bool {
        self.poll();
        let (current, views) = self
            .config()
            .ok()
            .and_then(|ocio| {
                let names = ocio.resolve(&self.sel, self.hdr).ok()?;
                let views = ocio
                    .views(&names.display, self.hdr)
                    .into_iter()
                    .map(str::to_owned)
                    .collect::<Vec<_>>();
                Some((names.view, views))
            })
            .unwrap_or_else(|| (self.sel.view.clone(), Vec::new()));
        ui.label("View:");
        let mut selected = current.clone();
        egui::ComboBox::from_id_salt("viewport_ocio_view")
            .width(150.0)
            .selected_text(if current.is_empty() {
                "Loading…"
            } else {
                &current
            })
            .show_ui(ui, |ui| {
                for name in views {
                    ui.selectable_value(&mut selected, name.clone(), name);
                }
            });
        if selected != current {
            self.sel.on = true;
            self.sel.view = selected;
            self.rebuild();
            true
        } else {
            false
        }
    }

    /// The Colour panel. Returns whether the choice changed; `browse` is set when the
    /// user asks for a config file (the app runs the file dialog).
    /// `swap` exchanges the preset buttons' store / recall clicks (`hotkeys::slot_click`).
    pub fn ui(&mut self, ui: &mut egui::Ui, browse: &mut bool, swap: bool) -> bool {
        self.poll();
        let before = self.sel.clone();
        let mut reload = false;
        self.presets
            .ui(ui, &mut self.sel, &mut self.preset_edit, swap);
        ui.add_space(4.0);
        ui.checkbox(&mut self.sel.on, "OCIO display transform")
            .on_hover_text("Show the colour image through the OCIO display / view (isolated channels stay raw).\nOff: the built-in sRGB display with the tone operator.");
        ui.add_space(4.0);

        let mut config = self.sel.config.clone();
        egui::Grid::new("ocio.grid").num_columns(2).spacing([8.0, 4.0]).show(ui, |ui| {
            ui.label("Config");
            ui.horizontal(|ui| {
                config_combo(ui, &mut config);
                if ui.button("File…").on_hover_text("Load a config file (.ocio / .ocioz).").clicked() {
                    *browse = true;
                }
                reload = ui.button("Reload").on_hover_text("Read the config again (after it changed on disk).").clicked();
            });
            ui.end_row();
            if config != self.sel.config {
                // The names of the old config mean nothing in the new one.
                self.sel = Sel { on: self.sel.on, config, ..Sel::default() };
            }
            if self.sel.config != before.config || reload {
                self.cfg = None;
            }
            let Some(ocio) = self.cfg.as_ref().and_then(|(_, config)| config.as_ref().ok()).cloned() else {
                return;
            };
            let owned = |v: Vec<&str>| v.into_iter().map(str::to_owned).collect::<Vec<_>>();
            let hdr = self.hdr;
            let displays = owned(ocio.displays(hdr));
            let resolved = ocio.resolve(&self.sel, hdr).ok();
            let views = resolved.as_ref().map(|n| owned(ocio.views(&n.display, hdr))).unwrap_or_default();
            let (inputs, looks) = (owned(ocio.inputs()), owned(ocio.looks()));
            let auto_input = ocio.working_space().map_or_else(|| "none found".to_owned(), |n| format!("auto: {n}"));
            let input_warning = resolved
                .as_ref()
                .and_then(|n| (ocio.is_working(&n.input) == Some(false)).then(|| n.input.clone()));
            let dflt = |f: fn(&Names) -> &str| resolved.as_ref().map_or(String::new(), |n| f(n).to_owned());

            ui.label("Input").on_hover_text(
                "This config's linear AP1 (ACEScg) colour space: the tracer renders in ACEScg. Default: found by its transform to the aces_interchange role (else by the ACES names).",
            );
            ui.horizontal(|ui| {
                combo(ui, "ocio.working_input", &mut self.sel.working_input, &inputs, &auto_input);
                if let Some(name) = &input_warning {
                    ui.colored_label(egui_widgets_config::semantic(ui).warn, egui_widgets_config::icons::WARNING)
                        .on_hover_text(format!("\"{name}\" is not linear AP1: colours will be wrong."));
                }
            });
            ui.end_row();
            ui.label("Display").on_hover_text(
                "The target display for rendering and export. HDR targets remain selectable on SDR screens; WarpBro presents an SDR preview there. Settings > Display selects the actual window output.",
            );
            let d_before = self.sel.display.clone();
            combo(ui, "ocio.display", &mut self.sel.display, &displays, &dflt(|n| &n.display));
            if self.sel.display != d_before {
                // The view list belongs to the display.
                self.sel.view.clear();
            }
            ui.end_row();
            ui.label("View").on_hover_text("The rendering of the scene for the display.");
            combo(ui, "ocio.view", &mut self.sel.view, &views, &dflt(|n| &n.view));
            ui.end_row();
            ui.label("Look").on_hover_text("A look applied instead of the view's own looks.");
            combo(ui, "ocio.look", &mut self.sel.look, &looks, "(the view's looks)");
            ui.end_row();
        });

        // One rebuild for whatever changed this frame.
        let changed = self.sel != before || reload;
        if changed {
            if reload {
                self.reload();
            } else {
                self.rebuild();
            }
        }
        if self.pending {
            ui.label("Preparing colour configuration…");
            ui.ctx()
                .request_repaint_after(std::time::Duration::from_millis(33));
        } else if let Some(e) = &self.err {
            ui.add_space(4.0);
            ui.colored_label(ui.visuals().error_fg_color, format!("OCIO not shown: {e}"));
        } else if let Err(e) = self.config() {
            ui.add_space(4.0);
            ui.colored_label(ui.visuals().error_fg_color, e);
        }
        changed
    }
}

/// Named application colour slots, persisted with preferences rather than scenes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ColourPreset {
    pub name: String,
    pub selection: Sel,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]

pub struct ColourPresets {
    pub slots: [ColourPreset; 4],
}

impl Default for ColourPresets {
    fn default() -> Self {
        let choices = [
            (
                "SDR · sRGB",
                "sRGB - Display",
                "ACES 2.0 - SDR 100 nits (Rec.709)",
                "",
            ),
            (
                "SDR · P3",
                "Display P3 - Display",
                "ACES 2.0 - SDR 100 nits (P3 D65)",
                "",
            ),
            ("sRGB, no rendering", "sRGB - Display", "Un-tone-mapped", ""),
            (
                "HDR · 1000 nits",
                "Display P3 HDR - Display",
                "ACES 2.0 - HDR 1000 nits (P3 D65)",
                "ACES 1.3 Reference Gamut Compression",
            ),
        ];
        Self {
            slots: choices.map(|(name, display, view, look)| ColourPreset {
                name: name.into(),
                selection: Sel {
                    on: true,
                    config: "ocio://studio-config-latest".into(),
                    display: display.into(),
                    view: view.into(),
                    look: look.into(),
                    ..Sel::default()
                },
            }),
        }
    }
}

struct PresetEdit {
    index: usize,
    name: String,
    selection: Sel,
    focus: bool,
}

impl ColourPresets {
    pub fn store(&mut self, index: usize, name: &str, selection: &Sel) -> bool {
        let Some(slot) = self.slots.get_mut(index) else {
            return false;
        };
        let name = name.trim();
        if name.is_empty() {
            return false;
        }
        *slot = ColourPreset {
            name: name.into(),
            selection: selection.clone(),
        };
        true
    }

    pub fn restore(&self, index: usize, selection: &mut Sel) -> bool {
        let Some(slot) = self.slots.get(index) else {
            return false;
        };
        *selection = slot.selection.clone();
        true
    }

    fn ui(
        &mut self,
        ui: &mut egui::Ui,
        selection: &mut Sel,
        edit: &mut Option<PresetEdit>,
        swap: bool,
    ) {
        use crate::hotkeys::{SlotClick, slot_click, slot_hint};
        let mapping = slot_hint(
            swap,
            "name and save the current colour settings",
            "restore this preset",
        );
        ui.horizontal_wrapped(|ui| {
            ui.label("Color presets");
            for (index, slot) in self.slots.iter().enumerate() {
                let s = &slot.selection;
                let hint = format!(
                    "{mapping}\n\nConfig: {}\nInput: {}\nDisplay: {}\nView: {}\nLook: {}\nOCIO: {}",
                    source(&s.config),
                    if s.working_input.is_empty() {
                        "auto ACEScg"
                    } else {
                        &s.working_input
                    },
                    s.display,
                    s.view,
                    if s.look.is_empty() {
                        "the view's looks"
                    } else {
                        &s.look
                    },
                    if s.on { "on" } else { "off" },
                );
                let response = ui
                    .add(egui::Button::new(&slot.name).selected(s == selection))
                    .on_hover_text(hint);
                match slot_click(&response, swap) {
                    Some(SlotClick::Store) => {
                        *edit = Some(PresetEdit {
                            index,
                            name: slot.name.clone(),
                            selection: selection.clone(),
                            focus: true,
                        });
                    }
                    Some(SlotClick::Recall) => *selection = slot.selection.clone(),
                    None => {}
                }
            }
        });
        let mut save = false;
        let mut cancel = false;
        let mut open = true;
        if let Some(draft) = edit.as_mut() {
            egui::Window::new("Save color preset")
                .id(egui::Id::new("ocio.preset_name"))
                .collapsible(false)
                .resizable(false)
                .open(&mut open)
                .show(ui.ctx(), |ui| {
                    ui.label(format!("Preset {}", draft.index + 1));
                    let response = ui
                        .add(egui::TextEdit::singleline(&mut draft.name).hint_text("Preset name"));
                    if draft.focus {
                        response.request_focus();
                        draft.focus = false;
                    }
                    let valid = !draft.name.trim().is_empty();
                    save = valid
                        && response.lost_focus()
                        && ui.input(|i| i.key_pressed(egui::Key::Enter));
                    cancel = ui.input(|i| i.key_pressed(egui::Key::Escape));
                    ui.horizontal(|ui| {
                        save |= ui.add_enabled(valid, egui::Button::new("Save")).clicked();
                        cancel |= ui.button("Cancel").clicked();
                    });
                });
        }
        if save {
            if let Some(draft) = edit.take() {
                self.store(draft.index, &draft.name, &draft.selection);
            }
        } else if cancel || !open {
            *edit = None;
        }
    }
}

/// The config combo: automatic ($OCIO or the default built-in), OCIO's built-in
/// configs, or the file in use.
fn config_combo(ui: &mut egui::Ui, config: &mut String) {
    let name = |c: &str| {
        if c.is_empty() {
            return format!("Auto ({})", source(""));
        }
        embedded::find_ocio_builtin(c).map_or_else(|| c.to_owned(), |e| e.ui_name.to_owned())
    };
    egui::ComboBox::from_id_salt("ocio.config")
        .selected_text(name(config))
        .width(260.0)
        .show_ui(ui, |ui| {
            ui.selectable_value(config, String::new(), name(""))
                .on_hover_text(
                    "$OCIO when set, else OCIO's default built-in config (ACES 2.0 CG).",
                );
            for e in embedded::ocio_builtins() {
                ui.selectable_value(config, e.uri(), e.ui_name);
            }
        })
        .response
        .on_hover_text(source(config));
}

/// A combo over `items` with an empty choice labelled `empty` (the default).
pub(crate) fn combo(ui: &mut egui::Ui, id: &str, cur: &mut String, items: &[String], empty: &str) {
    let text = if cur.is_empty() {
        format!("{empty} (default)")
    } else {
        cur.clone()
    };
    egui::ComboBox::from_id_salt(id)
        .selected_text(text)
        .width(260.0)
        .show_ui(ui, |ui| {
            ui.selectable_value(cur, String::new(), format!("{empty} (default)"));
            for it in items {
                ui.selectable_value(cur, it.clone(), it);
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The store click opens the naming prompt (Enter saves, Escape cancels), the recall click
    /// restores; both mappings of `hotkeys::slot_click` (default and swapped).
    #[test]
    fn colour_slot_clicks_prompt_save_cancel_and_restore() {
        for swap in [false, true] {
            slot_clicks(swap);
        }
    }

    fn slot_clicks(swap: bool) {
        let (store, recall) = if swap {
            (egui::PointerButton::Primary, egui::PointerButton::Secondary)
        } else {
            (egui::PointerButton::Secondary, egui::PointerButton::Primary)
        };
        let frame = |ctx: &egui::Context,
                     presets: &mut ColourPresets,
                     sel: &mut Sel,
                     edit: &mut Option<PresetEdit>,
                     events: Vec<egui::Event>|
         -> egui::FullOutput {
            ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(900.0, 600.0),
                    )),
                    events,
                    ..Default::default()
                },
                |root| {
                    egui::CentralPanel::default().show(root, |ui| presets.ui(ui, sel, edit, swap));
                },
            )
        };
        fn mouse(pos: egui::Pos2, button: egui::PointerButton, pressed: bool) -> Vec<egui::Event> {
            vec![
                egui::Event::PointerMoved(pos),
                egui::Event::PointerButton {
                    pos,
                    button,
                    pressed,
                    modifiers: Default::default(),
                },
            ]
        }
        fn key(key: egui::Key) -> Vec<egui::Event> {
            vec![egui::Event::Key {
                key,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: Default::default(),
            }]
        }
        let ctx = egui::Context::default();
        let mut presets = ColourPresets::default();
        let original = presets.clone();
        let mut sel = presets.slots[3].selection.clone();
        let expected = sel.clone();
        let mut edit = None;
        frame(&ctx, &mut presets, &mut sel, &mut edit, vec![]);
        let output = frame(&ctx, &mut presets, &mut sel, &mut edit, vec![]);
        let pos = output
            .shapes
            .iter()
            .find_map(|s| match &s.shape {
                egui::epaint::Shape::Text(t) if t.galley.text() == "SDR · sRGB" => {
                    Some(t.pos + t.galley.rect.center().to_vec2())
                }
                _ => None,
            })
            .unwrap();
        frame(
            &ctx,
            &mut presets,
            &mut sel,
            &mut edit,
            mouse(pos, store, true),
        );
        frame(
            &ctx,
            &mut presets,
            &mut sel,
            &mut edit,
            mouse(pos, store, false),
        );
        assert!(edit.is_some());
        assert_eq!(presets, original, "opening the prompt does not overwrite");
        frame(&ctx, &mut presets, &mut sel, &mut edit, vec![]);
        edit.as_mut().unwrap().name = "My HDR".into();
        frame(
            &ctx,
            &mut presets,
            &mut sel,
            &mut edit,
            key(egui::Key::Enter),
        );
        assert!(edit.is_none());
        assert_eq!(presets.slots[0].name, "My HDR");
        assert_eq!(presets.slots[0].selection, expected);
        let saved = presets.clone();
        edit = Some(PresetEdit {
            index: 0,
            name: "Cancelled".into(),
            selection: Sel::default(),
            focus: true,
        });
        frame(&ctx, &mut presets, &mut sel, &mut edit, vec![]);
        frame(
            &ctx,
            &mut presets,
            &mut sel,
            &mut edit,
            key(egui::Key::Escape),
        );
        assert!(edit.is_none());
        assert_eq!(presets, saved);
        sel = Sel::default();
        frame(&ctx, &mut presets, &mut sel, &mut edit, vec![]);
        frame(&ctx, &mut presets, &mut sel, &mut edit, vec![]);
        frame(
            &ctx,
            &mut presets,
            &mut sel,
            &mut edit,
            mouse(pos, recall, true),
        );
        frame(
            &ctx,
            &mut presets,
            &mut sel,
            &mut edit,
            mouse(pos, recall, false),
        );
        assert_eq!(sel, expected);
        assert!(edit.is_none());
    }

    #[test]
    fn colour_slots_store_whole_selection_and_defaults_resolve() {
        let mut presets = ColourPresets::default();
        for slot in &presets.slots {
            let o = Ocio::load(&slot.selection.config).unwrap();
            let n = o.resolve(&slot.selection, true).unwrap();
            assert!(o.transform(&n, true).is_ok());
        }
        let custom = Sel {
            on: false,
            config: "custom.ocio".into(),
            display: "custom display".into(),
            view: "custom view".into(),
            look: "custom look".into(),
            working_input: "custom input".into(),
        };
        let before = presets.clone();
        assert!(!presets.store(0, "  ", &custom));
        assert!(!presets.store(4, "invalid slot", &custom));
        assert_eq!(presets, before);
        assert!(presets.store(2, "  My colour  ", &custom));
        let json = serde_json::to_string(&presets).unwrap();
        let restored: ColourPresets = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, presets);
        assert_eq!(restored.slots[2].name, "My colour");
        let mut selection = crate::color::default_selection();
        assert!(restored.restore(2, &mut selection));
        assert_eq!(selection, custom);
        assert!(!restored.restore(4, &mut selection));
        assert_eq!(selection, custom);
    }

    #[test]
    fn hdr_preset_reproduces_studio_p3_setup() {
        let presets = ColourPresets::default();
        let mut sel = Sel {
            config: "other.ocio".into(),
            working_input: "Linear Rec.709".into(),
            look: "other look".into(),
            ..Sel::default()
        };
        assert!(presets.restore(3, &mut sel));
        assert!(sel.on);
        assert_eq!(sel.config, "ocio://studio-config-latest");
        assert!(sel.working_input.is_empty());
        let o = Ocio::load(&sel.config).unwrap();
        let n = o.resolve(&sel, true).unwrap();
        assert_eq!(n.input, "ACEScg");
        assert_eq!(n.display, "Display P3 HDR - Display");
        assert_eq!(n.view, "ACES 2.0 - HDR 1000 nits (P3 D65)");
        assert_eq!(n.look, "ACES 1.3 Reference Gamut Compression");
        assert!(o.transform(&n, true).unwrap().linear());
    }

    fn cg() -> Ocio {
        Ocio::load("ocio://default").expect("the default built-in config")
    }

    /// An HDR output picks the view whose MEASURED peak is nearest the requested one (no name
    /// parsing): from an SDR scene, 4000 asks for the 4000-nit PQ view, 1000 for the 1000-nit.
    #[test]
    fn hdr_output_picks_the_view_by_its_measured_peak() {
        let o = Ocio::load("ocio://studio-config-latest").unwrap();
        let sdr = crate::color::default_selection();
        for (want, nits) in [(4000.0, "4000"), (1000.0, "1000")] {
            let (_, view) = o.output_transform(&sdr, OutputKind::Pq, want).unwrap();
            assert!(view.contains(nits), "{want} nits -> {view}");
        }
        let names = o
            .resolve(
                &Sel {
                    display: "Rec.2100-PQ - Display".into(),
                    view: "ACES 2.0 - HDR 1000 nits (P3 D65)".into(),
                    ..sdr.clone()
                },
                true,
            )
            .unwrap();
        match o.transform(&names, true).unwrap().light().unwrap() {
            crate::color::DisplayLight::Absolute { peak_nits } => {
                assert!((peak_nits - 1000.0).abs() < 10.0, "{peak_nits}")
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn hdr_view_peak_cache_validates_current_input_and_look() {
        let o = Ocio::load("ocio://studio-config-latest").unwrap();
        let current = crate::color::default_selection();
        let display = "Rec.2100-PQ - Display";
        let view = "ACES 2.0 - HDR 1000 nits (P3 D65)";
        let warmed = o.view_peak(&current, display, view).unwrap();
        assert!((warmed - 1000.0).abs() < 10.0);
        for invalid in [
            Sel {
                working_input: "nonexistent input".into(),
                ..current.clone()
            },
            Sel {
                look: "nonexistent look".into(),
                ..current.clone()
            },
        ] {
            let selected = Sel {
                display: display.into(),
                view: view.into(),
                ..invalid.clone()
            };
            // The real native processor rejects this request. A warmed view must
            // not bypass its changed input/look validation or reuse that peak.
            assert!(
                o.resolve(&selected, true)
                    .and_then(|names| o.transform(&names, true))
                    .is_err()
            );
            assert_eq!(o.view_peak(&invalid, display, view), None);
            assert_eq!(o.view_peak(&current, display, view), Some(warmed));
        }
    }

    /// The defaults of the ACES 2.0 CG config, and scene 18 % grey through them at
    /// the value vfx-ocio's reference points give (0.349 encoded for the SDR sRGB view).
    #[test]
    fn default_transform_renders_mid_grey() {
        let o = cg();
        let n = o.resolve(&Sel::default(), false).unwrap();
        assert_eq!(n.input, crate::color::WORKING);
        // The working space is identified by its transform, not only by its name.
        assert_eq!(o.is_working(crate::color::WORKING), Some(true));
        assert_eq!(o.is_working("Linear Rec.709 (sRGB)"), Some(false));
        let studio = Ocio::load("ocio://studio-config-latest").unwrap();
        assert_eq!(studio.working_space(), Some(crate::color::WORKING));
        let picked = Sel {
            working_input: "lin_ap1".into(),
            ..Sel::default()
        };
        assert_eq!(
            studio.is_working(&studio.resolve(&picked, false).unwrap().input),
            Some(true)
        );
        assert_eq!(
            (n.display.as_str(), n.view.as_str()),
            ("sRGB - Display", "ACES 2.0 - SDR 100 nits (Rec.709)")
        );
        let t = o.transform(&n, false).unwrap();
        let mut px = [[0.18, 0.18, 0.18, 0.4]];
        t.display(&mut px, 0.0, 1.0, 1.0);
        for c in &px[0][..3] {
            assert!((c - 0.349).abs() < 2e-3, "{px:?}");
        }
        assert_eq!(px[0][3], 0.4, "alpha is the input's");
        eprintln!("default transform: LUT textures = {}", t.has_luts());
    }

    /// Exposure is a gain on the input and the dial acts on the display value.
    #[test]
    fn exposure_and_dial_bracket_the_transform() {
        let o = cg();
        let t = o
            .transform(&o.resolve(&Sel::default(), false).unwrap(), false)
            .unwrap();
        let mut a = [[0.09, 0.09, 0.09, 1.0]];
        let mut b = [[0.18, 0.18, 0.18, 1.0]];
        t.display(&mut a, 1.0, 1.0, 1.0);
        t.display(&mut b, 0.0, 1.0, 1.0);
        assert!((a[0][0] - b[0][0]).abs() < 1e-6, "{a:?} vs {b:?}");
        let mut g = [[0.18, 0.18, 0.18, 1.0]];
        t.display(&mut g, 0.0, 2.0, 1.0);
        assert!((g[0][0] - b[0][0].sqrt()).abs() < 1e-6, "{g:?}");
    }

    /// HDR views are hidden (the Studio config has PQ displays), SDR ones stay, and a
    /// name the config lacks is an error, not a substitute.
    #[test]
    fn only_sdr_is_offered_and_unknown_names_fail() {
        let o = Ocio::load("ocio://studio-config-latest").unwrap();
        let all: Vec<&str> = o.cfg.display_names();
        assert!(all.contains(&"Rec.2100-PQ - Display"));
        assert!(!o.displays(false).contains(&"Rec.2100-PQ - Display"));
        assert!(o.displays(false).contains(&"sRGB - Display"));
        let hdr = Sel {
            display: "Rec.2100-PQ - Display".into(),
            ..Sel::default()
        };
        assert!(o.resolve(&hdr, false).is_err());
        let bad = Sel {
            display: "no such display".into(),
            ..Sel::default()
        };
        assert!(
            o.resolve(&bad, false)
                .unwrap_err()
                .to_string()
                .contains("no such display")
        );
        // An HDR output offers the PQ display and its HDR views.
        assert!(o.displays(true).contains(&"Rec.2100-PQ - Display"));
        let n = o.resolve(&hdr, true).unwrap();
        assert!(
            o.views(&n.display, true)
                .contains(&"ACES 2.0 - HDR 1000 nits (P3 D65)")
        );
    }

    /// A look replaces the view's looks and changes the picture.
    #[test]
    fn a_look_changes_the_picture() {
        let o = Ocio::load("ocio://studio-config-latest").unwrap();
        let looks = o.looks();
        let look = *looks.first().expect("the Studio config has looks");
        let plain = o
            .transform(&o.resolve(&Sel::default(), false).unwrap(), false)
            .unwrap();
        let looked = o
            .transform(
                &o.resolve(
                    &Sel {
                        look: look.into(),
                        ..Sel::default()
                    },
                    false,
                )
                .unwrap(),
                false,
            )
            .unwrap();
        let (mut a, mut b) = ([[0.9, 0.2, 0.05, 1.0]], [[0.9, 0.2, 0.05, 1.0]]);
        plain.display(&mut a, 0.0, 1.0, 1.0);
        looked.display(&mut b, 0.0, 1.0, 1.0);
        assert_ne!(a, b, "look {look}");
    }

    /// Linear (HDR output) = the encoded transform decoded to light: for the sRGB
    /// display the display's code values run back through its own curve and
    /// primaries, so the linear result is the sRGB EOTF of the encoded one (within
    /// the float noise of two matrices); an HDR view uses the headroom (a 1000-nit
    /// view reaches ~10x SDR white) where the SDR view stays at white.
    #[test]
    fn linear_is_the_light_of_the_display() {
        let o = Ocio::load("ocio://studio-config-latest").unwrap();
        let sdr = o.resolve(&Sel::default(), true).unwrap();
        let enc = o.transform(&sdr, false).unwrap();
        let lin = o.transform(&sdr, true).unwrap();
        assert!(lin.linear() && !enc.linear() && lin.key != enc.key);
        for v in [0.02f32, 0.18, 0.7, 3.0] {
            let (mut a, mut b) = ([[v, v * 0.8, v * 0.5, 1.0]], [[v, v * 0.8, v * 0.5, 1.0]]);
            enc.display(&mut a, 0.0, 1.0, 1.0);
            lin.display(&mut b, 0.0, 1.0, lin.gain(240.0));
            for c in 0..3 {
                // `display` returns the canvas encoding: sRGB-encoded both ways.
                assert!(
                    (a[0][c] - b[0][c]).abs() < 2e-3,
                    "{v}: encoded {:?} vs linear {:?}",
                    a[0],
                    b[0]
                );
            }
        }
        // Light in display-reference units (gain 1: 1.0 = 100 nits for an HDR view).
        let peak = |t: &Transform| {
            let mut px = [[64.0, 64.0, 64.0, 1.0]];
            t.display(&mut px, 0.0, 1.0, 1.0);
            crate::transfer::eotf(px[0][1])
        };
        let hdr_sel = Sel {
            display: "Rec.2100-PQ - Display".into(),
            view: "ACES 2.0 - HDR 1000 nits (P3 D65)".into(),
            ..Sel::default()
        };
        let hdr = o
            .transform(&o.resolve(&hdr_sel, true).unwrap(), true)
            .unwrap();
        assert!(peak(&hdr) > 5.0, "1000-nit view peak {}", peak(&hdr));
        assert!(peak(&lin) < 1.05, "SDR view peak {}", peak(&lin));
        // An HDR view is absolute: on a 250-nit SDR white its 100 nits are 0.4 of it;
        // an SDR view (and any encoded transform) is relative.
        assert_eq!(hdr.gain(250.0), 0.4);
        assert_eq!((lin.gain(250.0), enc.gain(250.0)), (1.0, 1.0));
    }

    /// A config change resets the names; OCIO off shows the built-in transform.
    #[test]
    fn state_follows_the_choice() {
        fn ready(state: &mut State) {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
            while state.pending {
                state.poll();
                assert!(
                    std::time::Instant::now() < deadline,
                    "colour worker timed out"
                );
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
        }
        let mut s = State::new(Sel {
            on: true,
            config: "ocio://default".into(),
            ..Sel::default()
        });
        ready(&mut s);
        assert!(s.active().is_some(), "{:?}", s.err);
        s.sel.view = "no such view".into();
        s.rebuild();
        ready(&mut s);
        assert!(
            s.active().is_none() && s.err.as_deref().is_some_and(|e| e.contains("no such view"))
        );
        s.set_config("ocio://studio-config-latest".into());
        ready(&mut s);
        assert!(s.sel.view.is_empty() && s.active().is_some());
        // A reload is a new transform (new key), so the tile pipeline rebuilds too.
        let key = s.active().unwrap().key.clone();
        s.reload();
        ready(&mut s);
        assert_ne!(s.active().unwrap().key, key);
        // An HDR output makes the transform linear (a new key: the pipeline rebuilds).
        let key = s.active().unwrap().key.clone();
        s.set_hdr(true);
        ready(&mut s);
        assert!(s.active().unwrap().linear() && s.active().unwrap().key != key);
        s.set_on(false);
        ready(&mut s);
        assert!(s.active().is_none() && s.err.is_none());
    }

    #[test]
    fn colour_panel_draws_while_the_worker_mailbox_is_busy() {
        let mut state = State::new(Sel::default());
        let mailbox = state.worker.as_ref().unwrap().mailbox.clone();
        let _busy = mailbox.0.lock().unwrap();
        assert!(!state.poll());
        let ctx = egui::Context::default();
        let mut browse = false;
        let _ = ctx.run_ui(Default::default(), |root| {
            egui::CentralPanel::default().show(root, |ui| {
                assert!(!state.ui(ui, &mut browse, false));
                ui.label("The rest of the UI still draws");
            });
        });
        assert!(!browse);
    }

    #[test]
    fn colour_worker_keeps_only_the_latest_selection() {
        let mut state = State::new(Sel {
            on: true,
            ..Default::default()
        });
        for _ in 0..20 {
            state.sel.view = "superseded invalid view".into();
            state.rebuild();
            state.sel.view.clear();
            state.rebuild();
        }
        let expected_generation = state.generation;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        while state.pending {
            state.poll();
            assert!(
                std::time::Instant::now() < deadline,
                "colour worker timed out"
            );
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        assert_eq!(state.generation, expected_generation);
        assert!(state.err.is_none(), "{:?}", state.err);
        assert!(state.active().is_some());
        state.sel.on = false;
        state.rebuild();
        // Disabling is visible immediately, regardless of worker completion.
        assert!(state.active().is_none());
    }

    #[test]
    fn admitted_alpha_scratch_is_reused_and_preserves_alpha_bits() {
        let ocio = Ocio::load("ocio://default").unwrap();
        let transform = ocio
            .transform(&ocio.resolve(&Sel::default(), false).unwrap(), false)
            .unwrap();
        let alpha_bits = [0x8000_0000, 0x7fc0_1234, 0x3f80_0000, 0x0000_0000];
        let mut scratch = vec![0.0; 4096];
        let allocation = scratch.as_ptr();
        let capacity = scratch.capacity();
        for _ in 0..3 {
            let mut pixels: Vec<[f32; 4]> = (0..4096)
                .map(|i| [0.18, 0.18, 0.18, f32::from_bits(alpha_bits[i % 4])])
                .collect();
            transform
                .display_scratch(&mut pixels, &mut scratch, 0.0, 1.0, 1.0)
                .unwrap();
            for (i, pixel) in pixels.iter().enumerate() {
                assert_eq!(pixel[3].to_bits(), alpha_bits[i % 4]);
                assert_eq!(crate::transfer::code8(pixel[0]), 89);
            }
            assert_eq!(scratch.as_ptr(), allocation);
            assert_eq!(scratch.capacity(), capacity);
        }
    }

    #[test]
    fn short_alpha_scratch_refuses_before_changing_pixels() {
        let ocio = Ocio::load("ocio://default").unwrap();
        let transform = ocio
            .transform(&ocio.resolve(&Sel::default(), false).unwrap(), false)
            .unwrap();
        let mut pixels = [[0.18, 0.5, 1.0, f32::from_bits(0x7fc0_4567)]];
        let before = pixels.map(|p| p.map(f32::to_bits));
        assert!(
            transform
                .display_scratch(&mut pixels, &mut [], 1.0, 1.2, 1.0)
                .is_err()
        );
        assert_eq!(pixels.map(|p| p.map(f32::to_bits)), before);
    }
}
