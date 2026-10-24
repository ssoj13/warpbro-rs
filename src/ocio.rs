//! WarpBro's OCIO adapter over `vfx-ocio` (the OpenColorIO 2.5 port Playa and vfx-view use):
//! the scene's colour choice, the loaded configs, the display transforms the renderer and the
//! export writers run, and the Colour panel.
//!
//! - [`Sel`]: what the scene shows through (config, display, view, look, working input). The
//!   world document owns it; the Colour panel's [`State`] follows it (`State::set_sel`).
//! - [`shared`]: one loaded [`Ocio`] per config source for the whole process: the panel's worker,
//!   the renderer's `color::ColorPipeline` and export start all read the same load. The panel's
//!   Reload drops it ([`forget`]); every transform cache is keyed by the load ([`Ocio::serial`]).
//! - [`Ocio::transform`]: the display/view transform (with its look) from the tracer's working
//!   space (`color::WORKING`, ACEScg: a renderer fact, never persisted). "Encoded" ends in the
//!   display's code values (the SDR preview). "Linear" continues to linear Rec.709 display light,
//!   what the viewport and every file writer take: the display colour space back to OCIO's display
//!   reference (CIE XYZ D65, 1.0 = 100 nits: the config's `to_display_reference` or the inverted
//!   `from_display_reference`), then XYZ to Rec.709. An SDR view's light is relative (1.0 = SDR
//!   white), an HDR view's absolute with its measured peak ([`Transform::light`]); a data view
//!   (`Raw`) passes its values through as that light. [`Ocio::encoder`] takes the light back to
//!   the display's codes (16-bit SDR stills).
//! - [`Ocio::output_transform`]: the display / view an output of a signal (SDR, PQ, HLG) renders
//!   through (video-io's output-view policy over this config's classification).
//! - [`State`]: the Colour panel. Config loads and the selection check run on its worker; the
//!   panel shows their errors.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex, OnceLock, PoisonError};

use anyhow::{Context as _, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use vfx_ocio::builtin::embedded;
use vfx_ocio::color_matrix::{Adaptation, REC709, conversion_matrix_from_xyz_d65};
use vfx_ocio::{
    DisplayViewTransform, Encoding, GroupTransform, MatrixTransform, Processor, ReferenceSpaceType,
    TransformDirection,
};
pub use video_io::output_view::OutputKind;
use video_io::output_view::{OutputView, ViewSignal, resolve_output_view};

/// The signal of an HDR (`encoding: hdr-video`) view of `display`. OCIO records only
/// "hdr-video", not PQ or HLG, so HDR displays are told apart by their names (the ACES configs'
/// convention, as Nuke and Resolve do); the export panel shows the choice and lets it be
/// overridden. An HDR display named neither way carries no signal an export writes (`Data`).
fn hdr_signal(display: &str) -> ViewSignal {
    let name = display.to_ascii_uppercase();
    if name.contains("PQ") || name.contains("ST2084") || name.contains("ST-2084") {
        ViewSignal::Pq
    } else if name.contains("HLG") {
        ViewSignal::Hlg
    } else {
        ViewSignal::Data
    }
}

/// The colour choice of a scene. The world document stores it (`bus_slots["world"]["colour"]`,
/// read into `Scene::colour`); the Colour panel edits a copy ([`State::sel`]) that the app
/// authors back; `Settings.colour` is the choice a new document starts with and the colour
/// presets ([`ColourPresets`]) store whole ones. Empty names mean the config's defaults
/// ([`Ocio::resolve`]).
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
    OCIO_ENV
        .clone()
        .unwrap_or_else(|| "ocio://default".to_owned())
}

/// `$OCIO`, read once: [`source`] runs every rendered frame (`color::ColorPipeline`), and reading
/// the environment takes the process-wide environment lock. A value that is not Unicode cannot
/// name a config; it is reported once and the default applies.
static OCIO_ENV: LazyLock<Option<String>> = LazyLock::new(|| match std::env::var("OCIO") {
    Ok(value) => Some(value).filter(|v| !v.trim().is_empty()),
    Err(std::env::VarError::NotPresent) => None,
    Err(std::env::VarError::NotUnicode(raw)) => {
        log::warn!("$OCIO is not Unicode ({raw:?}); using ocio://default");
        None
    }
});

/// A config load, or why it failed (shared as it is: the error is the same for every reader).
pub type Loaded = Result<Arc<Ocio>, String>;

/// Every config this process loaded, one per source ([`shared`]).
static CONFIGS: LazyLock<Mutex<HashMap<String, Arc<OnceLock<Loaded>>>>> =
    LazyLock::new(Default::default);

/// The config `src` ([`source`]) names, loaded once for the whole process: the Colour panel's
/// worker, the renderer's `color::ColorPipeline` and export start read the same [`Ocio`], so they
/// can never disagree about a file that changed on disk. A failed load is shared as well, until
/// [`forget`]. The first reader loads it; readers of the same source wait for that load, readers
/// of other sources do not (the map lock is held only to find the slot).
pub fn shared(src: &str) -> Loaded {
    let slot = {
        let mut configs = CONFIGS.lock().unwrap_or_else(PoisonError::into_inner);
        match configs.get(src) {
            Some(slot) => slot.clone(),
            None => configs.entry(src.to_owned()).or_default().clone(),
        }
    };
    slot.get_or_init(|| Ocio::load(src).map(Arc::new).map_err(|e| format!("{e:#}")))
        .clone()
}

/// Drop `src`'s config: the next [`shared`] reads the file again (the Colour panel's Reload). A
/// reader still holding the old load keeps it until it asks again; caches keyed by
/// [`Ocio::serial`] then rebuild.
pub fn forget(src: &str) {
    CONFIGS
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .remove(src);
}

/// Names a [`Sel`] resolves to in one config.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Names {
    pub input: String,
    pub display: String,
    pub view: String,
    pub look: String,
}

/// The display / view an output renders through ([`Ocio::output_transform`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputTransform {
    pub display: String,
    pub view: String,
    /// Set when the scene's own view exists but cannot be written to the file's signal and was
    /// replaced: the sentence the user MUST be shown (video-io `OutputChoice::note`), so a file
    /// is never written through another view without saying so.
    pub note: Option<String>,
}

/// A loaded config.
pub struct Ocio {
    /// What it was loaded from ([`source`]).
    pub src: String,
    /// Which load this is (process-wide counter, [`Self::serial`]).
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

    /// Which load this is: part of every cached transform's key (`color::ColorPipeline`), so a
    /// reloaded file ([`forget`]) is never served from a cache of the old one.
    pub fn serial(&self) -> u64 {
        self.serial
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

    /// Whether `display` is a `kind` display: every display is an SDR one, an HDR kind needs
    /// the display's name to say it ([`hdr_signal`]).
    pub fn display_is(&self, display: &str, kind: OutputKind) -> bool {
        kind == OutputKind::Sdr || hdr_signal(display).fits(kind)
    }

    /// Whether view `view` of `display` renders for `kind` (HDR views of a PQ / HLG display for
    /// PQ / HLG, picture SDR views for SDR).
    pub fn view_is(&self, display: &str, view: &str, kind: OutputKind) -> bool {
        self.output_view(display, view).fits(kind)
    }

    /// The view as video-io's output-view selection sees it: its signal is classified here
    /// (video-io takes it as host data and never guesses it from names). Only an SDR (or
    /// untagged, OCIO v1) view fits an SDR file: an EDR view's highlights above 1.0 would clip
    /// in it, and linear / log views are not display code values, so all of those are `Data`
    /// (never chosen, and a current choice of one is replaced and reported).
    fn output_view<'a>(&self, display: &'a str, view: &'a str) -> OutputView<'a> {
        OutputView {
            display,
            view,
            signal: match self.view_encoding(display, view) {
                None | Some(Encoding::Sdr | Encoding::Unknown) => ViewSignal::Sdr,
                Some(Encoding::Hdr) => hdr_signal(display),
                Some(
                    Encoding::Edr
                    | Encoding::SceneLinear
                    | Encoding::DisplayLinear
                    | Encoding::Log
                    | Encoding::Data,
                ) => ViewSignal::Data,
            },
        }
    }

    /// Display and view an output of `kind` renders through. The scene's own choice (`current`)
    /// wins when it already fits; otherwise the config's first fitting display, with its view
    /// whose measured peak ([`Ocio::view_peak`]) is closest to `peak_nits` for HDR, its first
    /// view for SDR. A `current` naming a colour space, display, view or look this config lacks
    /// is an error, never a silent substitute: a stale choice would export through another
    /// transform. `current` is resolved against every view and the candidates are every view of
    /// the config (video-io's contract): an existing view that does not fit `kind` (an HDR or
    /// EDR view for an SDR file) is replaced, and [`OutputTransform::note`] says so for the user.
    pub fn output_transform(
        &self,
        current: &Sel,
        kind: OutputKind,
        peak_nits: f32,
    ) -> Result<OutputTransform> {
        let names = self.resolve(current, true)?;
        let mut candidates = Vec::new();
        for display in self.displays(true) {
            for view in self.views(display, true) {
                candidates.push(self.output_view(display, view));
            }
        }
        let selected = resolve_output_view(
            Some((names.display.as_str(), names.view.as_str())),
            kind,
            peak_nits,
            &candidates,
            |display, view| self.view_peak(current, display, view),
        )
        .map_err(anyhow::Error::msg)?;
        Ok(OutputTransform {
            display: selected.view.display.to_owned(),
            view: selected.view.view.to_owned(),
            note: selected.note(),
        })
    }

    /// The measured peak in nits of HDR `view` on `display` (with `current`'s input and look),
    /// cached per load; None when it is not an HDR view or cannot be built / measured.
    fn view_peak(&self, current: &Sel, display: &str, view: &str) -> Option<f32> {
        let sel = Sel {
            display: display.into(),
            view: view.into(),
            ..current.clone()
        };
        // A selection that does not resolve (its input or look is not in this config) only ranks
        // the view last here; the Colour panel reports the selection's own error (`State::err`).
        let names = self.resolve(&sel, true).ok()?;
        if let Some(peak) = self
            .peaks
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&names)
        {
            return *peak;
        }
        let peak = match self
            .transform(&names, true)
            .map_err(|e| format!("{e:#}"))
            .and_then(|t| t.light())
        {
            Ok(crate::color::DisplayLight::Absolute { peak_nits }) => Some(peak_nits),
            Ok(crate::color::DisplayLight::Relative) => None,
            // Cached below, so each view is reported once per load.
            Err(error) => {
                log::warn!("OCIO view {display} / {view} has no measurable peak: {error}");
                None
            }
        };
        self.peaks
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(names, peak);
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
        // Absolute light: the view renders into an HDR display colour space.
        let absolute =
            linear && self.view_encoding(&names.display, &names.view) == Some(Encoding::Hdr);
        Ok(Transform { proc, absolute })
    }

    /// The cheap check of `sel` the Colour panel shows (`State::err`): its names resolve in this
    /// config and its view has a display reference to reach light from ([`Self::transform`]'s
    /// linear chain), without building a processor. The renderer builds and reports the rest.
    pub fn check(&self, sel: &Sel) -> Result<Names> {
        let names = self.resolve(sel, true)?;
        self.to_display_light(&names.display, &names.view)?;
        Ok(names)
    }

    /// Linear Rec.709 display light of `names`' view (what `transform(names, true)` ends in) back
    /// to that display's code values: the inverse of its light chain, so the codes are exactly the
    /// encoded transform's. A writer that holds only a frame's light gets the same codes the
    /// frame's 8-bit preview has, at any precision (16-bit SDR stills).
    pub fn encoder(&self, names: &Names) -> Result<Processor> {
        let light = vfx_ocio::Transform::Group(GroupTransform {
            name: String::new(),
            transforms: self.to_display_light(&names.display, &names.view)?,
            direction: TransformDirection::Forward,
        });
        Ok(self
            .cfg
            .processor_from_transform(&light, TransformDirection::Forward)?
            .inverse()?)
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

/// One built display transform ([`Ocio::transform`]): its processor, and whether its light is
/// absolute. `color::ColorPipeline` compiles the processor for the GPU; the CPU applies it for
/// probes (peaks, the preview check).
pub struct Transform {
    proc: Processor,
    /// Its light is ABSOLUTE (an HDR display colour space, e.g. PQ: display reference 1.0 =
    /// 100 nits), not relative to SDR white. Only a linear transform has light.
    absolute: bool,
}

impl Transform {
    pub(crate) fn processor(&self) -> &Processor {
        &self.proc
    }
    pub(crate) fn absolute(&self) -> bool {
        self.absolute
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
            video_io::export::measure_display_peak(|probe| self.proc.apply_rgb(probe), 100.0)
                .map_err(|e| e.to_string())?;
        Ok(crate::color::DisplayLight::Absolute { peak_nits })
    }
}

/// A replaceable colour request; stale selections never accumulate in a queue.
struct ColourRequest {
    generation: u64,
    src: String,
    sel: Sel,
}
struct ColourResult {
    generation: u64,
    src: String,
    config: Loaded,
    /// [`Ocio::check`] of the selection (Ok while OCIO is off).
    check: Result<(), String>,
}
#[derive(Default)]
struct ColourMailbox {
    request: Option<ColourRequest>,
    result: Option<ColourResult>,
    stop: bool,
}
struct ColourWorker {
    mailbox: Arc<(Mutex<ColourMailbox>, std::sync::Condvar)>,
    latest: Arc<AtomicU64>,
}
impl ColourWorker {
    /// Start the colour worker; `ui` is woken when a load or check result is ready.
    fn new(ui: crate::render_service::Wake) -> std::io::Result<Self> {
        let mailbox = Arc::new((
            Mutex::new(ColourMailbox::default()),
            std::sync::Condvar::new(),
        ));
        let latest = Arc::new(AtomicU64::new(0));
        let worker_mailbox = mailbox.clone();
        let worker_latest = latest.clone();
        std::thread::Builder::new()
            .name("frac-colour".into())
            .spawn(move || {
                loop {
                    let request = {
                        let (mutex, wake) = &*worker_mailbox;
                        let mut slot = mutex.lock().unwrap_or_else(PoisonError::into_inner);
                        loop {
                            if slot.stop {
                                return;
                            }
                            if let Some(request) = slot.request.take() {
                                break request;
                            }
                            slot = wake.wait(slot).unwrap_or_else(PoisonError::into_inner);
                        }
                    };
                    let config = shared(&request.src);
                    if worker_latest.load(Ordering::Acquire) != request.generation {
                        continue;
                    }
                    let check = if request.sel.on {
                        config.as_ref().map_err(Clone::clone).and_then(|ocio| {
                            ocio.check(&request.sel)
                                .map(drop)
                                .map_err(|e| format!("{e:#}"))
                        })
                    } else {
                        Ok(())
                    };
                    // The UI is woken after the mailbox lock is released (`Wake::wake`).
                    let stored = {
                        let mut slot = worker_mailbox
                            .0
                            .lock()
                            .unwrap_or_else(PoisonError::into_inner);
                        if slot.stop {
                            return;
                        }
                        let current = worker_latest.load(Ordering::Acquire) == request.generation;
                        if current {
                            slot.result = Some(ColourResult {
                                generation: request.generation,
                                src: request.src,
                                config,
                                check,
                            });
                        }
                        current
                    };
                    if stored {
                        ui.wake();
                    }
                }
            })?;
        Ok(Self { mailbox, latest })
    }
    fn submit(&self, request: ColourRequest) {
        self.latest.store(request.generation, Ordering::Release);
        let mut slot = self
            .mailbox
            .0
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
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
        let mut slot = self
            .mailbox
            .0
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        slot.stop = true;
        slot.request = None;
        self.mailbox.1.notify_one();
        // No join on the UI: the worker exits after its current finite load.
    }
}

/// The Colour panel's state: the working copy of the document's selection, the config it names
/// and that selection's check. Config loads and checks run on its worker; the UI consumes only
/// ready results. Every display (SDR and HDR) is offered: the selection is the rendering and
/// export target, and an SDR window shows an HDR view through its SDR preview.
pub struct State {
    pub sel: Sel,
    pub presets: ColourPresets,
    preset_edit: Option<PresetEdit>,
    cfg: Option<(String, Loaded)>,
    pub err: Option<String>,
    worker: Option<ColourWorker>,
    generation: u64,
    pending: bool,
}

pub(crate) struct ColourUi {
    pub selection: Option<Sel>,
    pub reload: bool,
    pub presets: Option<ColourPresets>,
}

impl State {
    /// The colour state of `sel`, its config loading on a worker that wakes `ui` when ready.
    pub fn new(sel: Sel, ui: crate::render_service::Wake) -> Self {
        let (worker, err) = match ColourWorker::new(ui) {
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
            cfg: None,
            err,
            worker,
            generation: 0,
            pending: false,
        };
        state.rebuild();
        state
    }

    /// Follow the document's selection (`App::refresh_scene`, after load, undo, redo and every
    /// edit): rebuilds only when it differs, so calling it every frame costs a comparison.
    pub fn set_sel(&mut self, sel: &Sel) {
        if self.sel != *sel {
            self.sel = sel.clone();
            self.rebuild();
        }
    }

    /// Schedule the latest selection without loading files on the caller. Until it completes,
    /// no stale error is shown.
    pub fn rebuild(&mut self) {
        let Some(worker) = &self.worker else {
            return;
        };
        self.generation = self.generation.wrapping_add(1);
        self.err = None;
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
            src,
            sel: self.sel.clone(),
        });
    }

    /// Consume an already-ready event. Never waits for a load or a worker lock.
    pub fn poll(&mut self) -> bool {
        let Some(result) = self.worker.as_ref().and_then(ColourWorker::take_ready) else {
            return false;
        };
        if result.generation != self.generation {
            return false;
        }
        self.pending = false;
        self.cfg = Some((result.src, result.config));
        self.err = result.check.err();
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

    /// Read the config again (it changed on disk, or a failed load was fixed): the shared load
    /// is dropped ([`forget`]), so the renderer and export read the new one too.
    pub fn reload(&mut self) {
        forget(&source(&self.sel.config));
        self.cfg = None;
        self.rebuild();
    }

    /// Compact direct View control; returns a proposed selection for the app event bus.
    pub fn quick_view_ui(&mut self, ui: &mut egui::Ui) -> Option<Sel> {
        self.poll();
        let mut selection = self.sel.clone();
        let (display, current, views) = self
            .config()
            .ok()
            .and_then(|ocio| {
                let names = ocio.resolve(&selection, true).ok()?;
                let views = ocio
                    .views(&names.display, true)
                    .into_iter()
                    .map(str::to_owned)
                    .collect::<Vec<_>>();
                Some((names.display, names.view, views))
            })
            .unwrap_or_else(|| {
                (
                    selection.display.clone(),
                    selection.view.clone(),
                    Vec::new(),
                )
            });
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
            selection.on = true;
            // The view belongs to the display it was listed from: store both, so the choice
            // never depends on which display an empty field defaults to elsewhere.
            selection.display = display;
            selection.view = selected;
            Some(selection)
        } else {
            None
        }
    }

    /// The Colour panel. Returns whether the choice changed; `browse` is set when the
    /// user asks for a config file (the app runs the file dialog).
    /// `swap` exchanges the preset buttons' store / recall clicks (`hotkeys::slot_click`).
    pub(crate) fn ui(&mut self, ui: &mut egui::Ui, browse: &mut bool, swap: bool) -> ColourUi {
        self.poll();
        let before = self.sel.clone();
        let mut selection = before.clone();
        let mut presets = self.presets.clone();
        let mut reload = false;
        presets.ui(ui, &mut selection, &mut self.preset_edit, swap);
        ui.add_space(4.0);
        ui.checkbox(&mut selection.on, "OCIO display transform")
            .on_hover_text("Show the colour image through the OCIO display / view (isolated channels stay raw).\nOff: the built-in sRGB display with the tone operator.");
        ui.add_space(4.0);

        let mut config = selection.config.clone();
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
            if config != selection.config {
                // The names of the old config mean nothing in the new one.
                selection = Sel { on: selection.on, config, ..Sel::default() };
            }
            if selection.config != before.config || reload {
                return;
            }
            let Some(ocio) = self.cfg.as_ref().and_then(|(_, config)| config.as_ref().ok()).cloned() else {
                return;
            };
            let owned = |v: Vec<&str>| v.into_iter().map(str::to_owned).collect::<Vec<_>>();
            let displays = owned(ocio.displays(true));
            let resolved = ocio.resolve(&selection, true).ok();
            let views = resolved.as_ref().map(|n| owned(ocio.views(&n.display, true))).unwrap_or_default();
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
                combo(ui, "ocio.working_input", &mut selection.working_input, &inputs, &auto_input);
                if let Some(name) = &input_warning {
                    ui.colored_label(soluwell_config::semantic(ui).warn, soluwell_config::icons::WARNING)
                        .on_hover_text(format!("\"{name}\" is not linear AP1: colours will be wrong."));
                }
            });
            ui.end_row();
            ui.label("Display").on_hover_text(
                "The target display for rendering and export. HDR targets remain selectable on SDR screens; WarpBro presents an SDR preview there. Settings > Display selects the actual window output.",
            );
            let d_before = selection.display.clone();
            combo(ui, "ocio.display", &mut selection.display, &displays, &dflt(|n| &n.display));
            if selection.display != d_before {
                // The view list belongs to the display.
                selection.view.clear();
            }
            ui.end_row();
            ui.label("View").on_hover_text("The rendering of the scene for the display.");
            combo(ui, "ocio.view", &mut selection.view, &views, &dflt(|n| &n.view));
            ui.end_row();
            ui.label("Look").on_hover_text("A look applied instead of the view's own looks.");
            combo(ui, "ocio.look", &mut selection.look, &looks, "(the view's looks)");
            ui.end_row();
        });

        // One rebuild for whatever changed this frame.
        let changed = selection != before || reload;

        // The worker wakes the UI when the load is done (`State::new`).
        if self.pending {
            ui.label("Preparing colour configuration…");
        } else if let Some(e) = &self.err {
            ui.add_space(4.0);
            ui.colored_label(ui.visuals().error_fg_color, format!("OCIO not shown: {e}"));
        } else if let Err(e) = self.config() {
            ui.add_space(4.0);
            ui.colored_label(ui.visuals().error_fg_color, e);
        }
        ColourUi {
            selection: changed.then_some(selection),
            reload,
            presets: (presets != self.presets).then_some(presets),
        }
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
                    Some(SlotClick::Recall) => {
                        self.restore(index, selection);
                    }
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
        // Display P3 HDR is an EDR display: its light is relative to SDR white (1.0 = SDR
        // white), not absolute nits, and HDR highlights extend above 1.0.
        let t = o.transform(&n, true).unwrap();
        assert_eq!(t.light().unwrap(), crate::color::DisplayLight::Relative);
        let mut px = [[64.0, 64.0, 64.0, 1.0]];
        t.processor().apply_rgba(&mut px);
        assert!(px[0][1] > 5.0, "EDR highlight {}", px[0][1]);
    }

    fn cg() -> Ocio {
        Ocio::load("ocio://default").expect("the default built-in config")
    }

    /// A stale choice (a view the loaded config lacks) fails the output transform instead of
    /// exporting through the config's first fitting view; an existing HDR view on an SDR output
    /// is replaced by an SDR one.
    #[test]
    fn output_transform_refuses_a_view_the_config_lacks() {
        let o = Ocio::load("ocio://studio-config-latest").unwrap();
        let sdr = crate::color::default_selection();
        let stale = Sel {
            view: "No such view".into(),
            ..sdr.clone()
        };
        let err = o
            .output_transform(&stale, OutputKind::Sdr, 100.0)
            .unwrap_err()
            .to_string();
        assert!(err.contains("No such view"), "{err}");
        let hdr = Sel {
            display: "Rec.2100-PQ - Display".into(),
            view: "ACES 2.0 - HDR 1000 nits (P3 D65)".into(),
            ..sdr.clone()
        };
        let out = o.output_transform(&hdr, OutputKind::Sdr, 100.0).unwrap();
        assert!(
            o.view_is(&out.display, &out.view, OutputKind::Sdr),
            "{out:?}"
        );
        let note = out.note.expect("a replaced choice is reported");
        assert!(
            note.contains(&hdr.view) && note.contains(&out.view),
            "{note}"
        );
        // A fitting choice is kept, silently.
        let kept = o.output_transform(&sdr, OutputKind::Sdr, 100.0).unwrap();
        assert_eq!((kept.view.as_str(), kept.note), (sdr.view.as_str(), None));
    }

    /// An EDR view (extended range relative to SDR white, e.g. Display P3 HDR) cannot be written
    /// to an SDR file without clipping its highlights: it is replaced by an SDR view, reported.
    #[test]
    fn edr_view_is_replaced_for_an_sdr_file() {
        let o = Ocio::load("ocio://studio-config-latest").unwrap();
        let edr = Sel {
            display: "Display P3 HDR - Display".into(),
            view: "ACES 2.0 - SDR 100 nits (Rec.709)".into(),
            ..crate::color::default_selection()
        };
        assert_eq!(
            o.view_encoding(&edr.display, &edr.view),
            Some(Encoding::Edr),
            "the test needs an EDR view"
        );
        let out = o.output_transform(&edr, OutputKind::Sdr, 100.0).unwrap();
        assert_eq!(
            o.view_encoding(&out.display, &out.view),
            Some(Encoding::Sdr),
            "{out:?}"
        );
        let note = out.note.expect("the EDR choice was replaced");
        assert!(
            note.contains(&edr.view) && note.contains(&out.view),
            "{note}"
        );
    }

    /// An HDR output picks the view whose MEASURED peak is nearest the requested one (no name
    /// parsing): from an SDR scene, 4000 asks for the 4000-nit PQ view, 1000 for the 1000-nit.
    #[test]
    fn hdr_output_picks_the_view_by_its_measured_peak() {
        let o = Ocio::load("ocio://studio-config-latest").unwrap();
        let sdr = crate::color::default_selection();
        for (want, nits) in [(4000.0, "4000"), (1000.0, "1000")] {
            let view = o.output_transform(&sdr, OutputKind::Pq, want).unwrap().view;
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
        let mut px = [[0.18, 0.18, 0.18, 1.0]];
        t.processor().apply_rgba(&mut px);
        for c in &px[0][..3] {
            assert!((c - 0.349).abs() < 2e-3, "{px:?}");
        }
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
        // The panel's check fails the same way, without building a processor.
        assert!(o.check(&bad).is_err());
        assert_eq!(o.check(&hdr).unwrap(), n);
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
        plain.processor().apply_rgba(&mut a);
        looked.processor().apply_rgba(&mut b);
        assert_ne!(a, b, "look {look}");
    }

    /// Linear = the encoded transform decoded to light: for the sRGB display the display's code
    /// values run back through its own curve and primaries, so the encoded result is the sRGB
    /// OETF of the linear one (within the float noise of two matrices); an HDR view uses the
    /// headroom (a 1000-nit view reaches ~10x SDR white, display reference 1.0 = 100 nits) where
    /// the SDR view stays at white. An SDR view's light is relative, an HDR view's absolute.
    #[test]
    fn linear_is_the_light_of_the_display() {
        let o = Ocio::load("ocio://studio-config-latest").unwrap();
        let sdr = o.resolve(&Sel::default(), true).unwrap();
        let enc = o.transform(&sdr, false).unwrap();
        let lin = o.transform(&sdr, true).unwrap();
        for v in [0.02f32, 0.18, 0.7, 3.0] {
            let (mut a, mut b) = ([[v, v * 0.8, v * 0.5, 1.0]], [[v, v * 0.8, v * 0.5, 1.0]]);
            enc.processor().apply_rgba(&mut a);
            lin.processor().apply_rgba(&mut b);
            for c in 0..3 {
                assert!(
                    (a[0][c] - crate::color::oetf(b[0][c])).abs() < 2e-3,
                    "{v}: encoded {:?} vs linear {:?}",
                    a[0],
                    b[0]
                );
            }
        }
        let peak = |t: &Transform| {
            let mut px = [[64.0, 64.0, 64.0, 1.0]];
            t.processor().apply_rgba(&mut px);
            px[0][1]
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
        assert!(hdr.light().unwrap().hdr());
        assert_eq!(lin.light().unwrap(), crate::color::DisplayLight::Relative);
        assert_eq!(enc.light().unwrap(), crate::color::DisplayLight::Relative);
    }

    /// The encoder takes a view's light back to exactly the codes the encoded transform gives:
    /// for the sRGB display and for a Display P3 one, whose codes an sRGB OETF of the Rec.709
    /// light would put in the wrong primaries.
    #[test]
    fn the_encoder_inverts_the_light_to_the_displays_codes() {
        let o = Ocio::load("ocio://studio-config-latest").unwrap();
        let p3 = o
            .displays(false)
            .into_iter()
            .find(|d| d.contains("P3") && !d.contains("HDR"))
            .expect("the Studio config has an SDR P3 display")
            .to_owned();
        for display in ["sRGB - Display".to_owned(), p3] {
            let names = o
                .resolve(
                    &Sel {
                        display: display.clone(),
                        ..Sel::default()
                    },
                    false,
                )
                .unwrap();
            let encoder = o.encoder(&names).unwrap();
            let probe = [
                [0.001f32, 0.02, 0.05, 1.0],
                [0.18, 0.5, 0.9, 1.0],
                [0.9, 0.2, 0.05, 1.0],
                [2.0, 2.0, 2.0, 1.0],
            ];
            let (mut codes, mut light) = (probe, probe);
            o.transform(&names, false)
                .unwrap()
                .processor()
                .apply_rgba(&mut codes);
            o.transform(&names, true)
                .unwrap()
                .processor()
                .apply_rgba(&mut light);
            encoder.apply_rgba(&mut light);
            for (got, want) in light.iter().zip(codes) {
                for c in 0..3 {
                    assert!(
                        (got[c] - want[c]).abs() < 1e-4,
                        "{display}: {got:?} vs {want:?}"
                    );
                }
            }
        }
    }

    /// One load per source for the whole process; a reload (forget) reads the file again.
    #[test]
    fn configs_are_shared_per_source_until_forgotten() {
        let src = "ocio://cg-config-latest";
        let a = shared(src).unwrap();
        let b = shared(src).unwrap();
        assert!(Arc::ptr_eq(&a, &b));
        forget(src);
        let c = shared(src).unwrap();
        assert_ne!(a.serial(), c.serial());
        assert!(shared("no such config.ocio").is_err());
    }

    /// The panel follows the document's selection (`set_sel` rebuilds only on change), a config
    /// change resets the names, a bad name is the panel's error, and a reload is a new load.
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
        let mut s = State::new(
            Sel {
                on: true,
                config: "ocio://default".into(),
                ..Sel::default()
            },
            crate::render_service::Wake::default(),
        );
        ready(&mut s);
        assert!(s.err.is_none() && s.config().is_ok(), "{:?}", s.err);
        let generation = s.generation;
        let same = s.sel.clone();
        s.set_sel(&same);
        assert_eq!(
            s.generation, generation,
            "an unchanged selection does not rebuild"
        );
        s.set_sel(&Sel {
            view: "no such view".into(),
            ..same.clone()
        });
        ready(&mut s);
        assert!(s.err.as_deref().is_some_and(|e| e.contains("no such view")));
        s.set_config("ocio://studio-config-latest".into());
        ready(&mut s);
        assert!(s.sel.view.is_empty() && s.err.is_none(), "{:?}", s.err);
        let serial = s.config().unwrap().serial();
        s.reload();
        ready(&mut s);
        assert_ne!(s.config().unwrap().serial(), serial);
        s.set_sel(&Sel {
            on: false,
            ..s.sel.clone()
        });
        ready(&mut s);
        assert!(s.err.is_none());
    }

    #[test]
    fn colour_panel_draws_while_the_worker_mailbox_is_busy() {
        let mut state = State::new(Sel::default(), crate::render_service::Wake::default());
        let mailbox = state.worker.as_ref().unwrap().mailbox.clone();
        let _busy = mailbox.0.lock().unwrap();
        assert!(!state.poll());
        let ctx = egui::Context::default();
        let mut browse = false;
        let _ = ctx.run_ui(Default::default(), |root| {
            egui::CentralPanel::default().show(root, |ui| {
                let proposed = state.ui(ui, &mut browse, false);
                assert!(
                    proposed.selection.is_none() && !proposed.reload && proposed.presets.is_none()
                );
                ui.label("The rest of the UI still draws");
            });
        });
        assert!(!browse);
    }

    #[test]
    fn colour_worker_keeps_only_the_latest_selection() {
        let mut state = State::new(
            Sel {
                on: true,
                ..Default::default()
            },
            crate::render_service::Wake::default(),
        );
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
        assert!(state.config().is_ok());
    }
}
