//! The application shell (egui): `App` owns the authored document (`world::WorldEditor`, the
//! one source of scene state), its evaluated snapshot (`App::scene`, rebuilt by
//! `refresh_scene`), the render and file workers and every panel of the dock (`dock.rs`):
//! viewport (progressive render, navigation, manipulator, picking), gallery, bookmarks and
//! material cards with GPU thumbnails, Attribute Editor, Settings (saved to
//! `~/.warpbro/settings.json`) and Render / Encode (`export.rs`; output lands in
//! `~/.warpbro/out/<timestamp>`). Widgets and REST publish the same typed Playa events;
//! `App::handle_events` accepts them at frame boundaries and authors undoable document edits.
//! `App::scene` is evaluated scratch state. The window (`window.rs`) redraws only while
//! `App::animating`.

#[path = "app_actions.rs"]
mod actions;
#[path = "app_control.rs"]
mod control;
#[path = "app_events.rs"]
mod event_dispatch;
pub(crate) use actions::{AppAction, FlyInput, OrbitInput};

use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use egui::{self, Color32, ColorImage, RichText, Sense, TextureHandle, TextureOptions, Vec2};

use crate::render_service::{Command, Frame, RenderEvent, RenderService, ViewportRequest};
use crate::scene::*;
use crate::world::WorldDocument;
use soluwell_config::AttrMetrics;
use soluwell_panels::attr_grid::{
    AttrField, AttrGridHooks, AttrGridState, AttrValue, render_grid_with_config,
};
use std::collections::HashSet;
use std::sync::Arc;

#[path = "dock.rs"]
mod dock;
#[path = "takes_ui.rs"]
pub(crate) mod takes_ui;

const THUMB_W: usize = 224;
const THUMB_H: usize = 126;
const THUMB_SPP: u32 = 24;

pub fn run() -> anyhow::Result<()> {
    crate::window::run()
}

struct Entry {
    scene: Scene,
    thumb: Option<TextureHandle>,
    path: Option<PathBuf>,
}

/// The gallery or bookmark card the current document was made from: the card the browser
/// highlights. Identity, not scene equality: a loaded card's document gets a new id.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Card {
    Gallery(usize),
    Bookmark(PathBuf),
}

/// A bookmark file operation waiting for the file worker (answered as an
/// `io_service::SceneEvent` with its id): the bookmark list changes only once the file did.
enum BookmarkOp {
    /// The card a successful save adds (its path is the event's); boxed: an entry holds a scene.
    Save(Box<Entry>),
    Delete,
}

/// A document history intent shared by the Edit menu, hotkeys and REST.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum History {
    Undo,
    Redo,
    /// Back to the document the scene was loaded as (`App::origin`), one undoable step.
    Reset,
}

/// The preview cache's RAM budget (Settings > Animation): a share of the installed memory the
/// OS reports, never more than leaves `reserve_gb` to the system (`CacheManager` reads the OS).
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
struct PreviewCache {
    /// Share of the installed RAM, (0, 1].
    fraction: f32,
    /// GiB always left to the system.
    reserve_gb: f32,
}
impl Default for PreviewCache {
    fn default() -> Self {
        Self {
            fraction: 0.05,
            reserve_gb: 2.0,
        }
    }
}

const SWATCH: usize = 96;
const SWATCH_SPP: u32 = 32;

/// The scene a material-library swatch is rendered with: the preset on a unit sphere (a
/// quaternion Julia set with c = 0 is exactly the unit ball), three-quarter light, sky behind.
fn swatch_scene(preset: &crate::materials::MaterialPreset) -> Scene {
    let mut material = Material::default();
    preset.apply(&mut material);
    material.model = MaterialModel::StandardSurface;
    material_preview_scene(material)
}

/// Render the evaluated Material-node value; the card never owns authoring state.
fn material_preview_scene(material: Material) -> Scene {
    let mut s = Scene::preset(Formula::QuaternionJulia(Quat::PRESET));
    if let Formula::QuaternionJulia(q) = &mut s.formula {
        q.constant = [0.0; 4];
    }
    s.camera.distance = 2.3;
    s.camera.fov_y_degrees = 30.0;
    s.camera.yaw_degrees = 20.0;
    s.camera.pitch_degrees = 15.0;
    s.render.max_bounces = 4;
    s.render.iterations = 8.0;
    s.render.exposure_stops = -0.5;
    // Neutral grey "studio" sky so metals show their own tint, not the blue sky's.
    s.sun.azimuth = 45.0;
    s.sun.elevation = 40.0;
    s.sun.intensity = 3.0;
    s.sky.horizon = [0.03, 0.03, 0.03];
    s.sky.zenith = [0.14, 0.14, 0.14];
    s.sky.intensity = 5.0;
    s.material = material;
    s
}

/// Which objects a material "Assign to …" action addresses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AssignTo {
    /// The remembered object selection (`remember_material_targets`).
    Selected,
    /// Every node with a material reference.
    AllFractals,
}

pub(crate) struct App {
    renderer: RenderService,
    io: crate::io_service::IoService,
    gpu_name: String,
    frame: Option<Arc<Frame>>,
    frame_from_cache: bool,
    request: Option<ViewportRequest>,
    /// Authoring identity captured when submitting a viewport render, not when it completes.
    viewport_stamp: Option<(u64, playa_graph::NodeId, u64, u32)>,
    generation: u64,
    staged_frame: Option<Arc<Frame>>,
    staged_output: Option<(bool, f32)>,
    layouts: dock::Layouts,
    export: crate::export::ExportController,
    thumb_pending: Option<(u64, Scene)>,
    /// Encoded `ThumbId`s of gallery, bookmark and swatch cards whose render failed: not asked
    /// again (material cards keep their own state in `material_gallery`). Bookmark ids are
    /// indices, so they are forgotten whenever the bookmark list changes.
    thumb_failed: HashSet<u64>,
    /// The viewport click waiting for its GPU pick (`App::viewport_click`), and the last id.
    pick_pending: Option<PendingPick>,
    pick_sequence: u64,
    gui_fps: u32,
    scene: Scene,
    world: crate::world::WorldEditor,
    world_ui: crate::world_ui::WorldUi,
    preview: crate::preview::PreviewController,
    preview_key: Option<PreviewKey>,
    preview_sequence: u64,
    material_targets: Vec<crate::world::NodeId>,
    material_selection_stamp: Option<(playa_graph::NodeId, u64, Option<crate::world::NodeId>, u64)>,
    preview_progress: Option<(u32, u32, usize)>,
    evaluated_world: Option<(playa_graph::NodeId, u64, u64)>,
    load_revision: u64,
    /// The scene as it was loaded, for Edit > Reset scene and View > Restore camera.
    origin: Scene,
    /// The card `origin` was loaded from, if any (`browser` highlights it).
    card: Option<Card>,
    /// The colour selection a new document starts with (File > New, startup): `Settings::colour`.
    /// The last selection the user chose in the colour controls; a loaded document keeps its own.
    default_colour: crate::ocio::Sel,

    showing_preview: bool,
    hdr_view: std::sync::Arc<std::sync::Mutex<soluwell_viewport::hdr_view::HdrView>>,
    pub display: soluwell_viewport::display::DisplayPrefs,
    colour: crate::ocio::State,
    prefs: soluwell_panels::prefs::PrefsPanelState,
    /// Channels, tolerance and timing of "Bake to keys" (Takes panel; saved with the settings).
    recorder_options: crate::camera_recorder::Options,
    /// Navigation gestures become takes (`Settings::record_takes`).
    record_takes: bool,
    /// Minimum duration (seconds) required to keep a recorded take (`Settings::min_take_secs`).
    min_take_secs: f64,
    /// The viewport toolbar's CamClip: application state (a new scene keeps it), saved with a
    /// scene (`WorldDocument::camera_slots`) and replaced by a scene that carries one.
    camera_slots: crate::camera_slots::CameraSlots,
    /// Cuts every viewport navigation gesture into a take (`crate::takes`).
    takes_recorder: crate::takes::Recorder,
    /// The flight or orbit rig moved the camera this frame: only navigation makes takes.
    navigated: bool,
    /// Takes panel state (not saved): see `takes_ui`.
    takes_view: takes_ui::View,
    /// The viewport tool (Q / W / E / R) and its manipulator, drawn on the selected objects.
    tool: ViewTool,
    gizmo: soluwell_viewport::gizmo::Gizmo,
    /// The manipulator drag in progress: its edits share one undo step.
    gizmo_gesture: Option<u64>,
    /// Records each manipulator drag as transform takes of the dragged nodes.
    drag_recorder: crate::takes::DragRecorder,
    /// Splitter state of the Settings grids, in page order: Controls' camera controls, camera
    /// alignment, manipulator, slot buttons and attribute controls, then Animation's preview cache.
    prefs_grids: [AttrGridState; 6],
    dock: soluwell_chrome::dock_layout::Layout<dock::Panel>,
    /// The panel the user last clicked into: new tabs join its group.
    dock_focus: Option<dock::Panel>,
    panels_to_open: Vec<dock::Panel>,
    toolbar: soluwell_viewport::toolbar::ToolbarState,
    fonts: dock::Fonts,
    applied_fonts: Option<dock::Fonts>,
    viewport_visible: bool,
    viewport_rect: Option<egui::Rect>,
    config_picker: egui_file_dialog::FileDialog,
    scene_picker: Option<(crate::templates::FileAction, egui_file_dialog::FileDialog)>,
    scene_file_path: Option<PathBuf>,
    /// The last id of a file-worker request answered by id (scene files and bookmarks).
    scene_file_sequence: u64,
    scene_file_pending: Option<(u64, u64, crate::templates::FileAction)>,
    /// Bookmark saves and deletes waiting for the file worker, by request id.
    bookmark_ops: Vec<(u64, BookmarkOp)>,
    /// Wakes this UI from the workers (render, file, colour, export) when they post a result;
    /// attached to the window's egui context on the first frame (`ui`).
    wake: crate::render_service::Wake,
    templates: Vec<crate::templates::Entry>,
    templates_requested: bool,
    templates_pending: bool,
    persisted_settings: Option<Settings>,
    snapshot_before: Option<Scene>,
    #[cfg(test)]
    settings_serializations: usize,
    #[cfg(test)]
    snapshot_clones: usize,
    status_layout: soluwell_chrome::statusbar::StatusBarLayout,
    status_resizable: bool,
    gallery: Vec<Entry>,
    bookmarks: Vec<Entry>,

    swatches: Vec<Option<TextureHandle>>,
    material_gallery: crate::material_gallery::MaterialGallery,
    /// The preview cache's memory budget (`Settings::preview_cache`).
    preview_cache: PreviewCache,
    target_spp: u32,
    spp_per_frame: u32,
    last_change: Instant,
    last_scene: Scene,
    /// Evaluated selected-profile scene; invalidated by graph revision, time or profile.
    viewport_scene: Option<(playa_graph::NodeId, u64, u64, Scene)>,
    /// The viewport extras and what they were built for (document, revision, playhead,
    /// selection, aspect): `App::draw_overlays`.
    #[allow(clippy::type_complexity)]
    overlays: Option<(
        (
            playa_graph::NodeId,
            u64,
            u64,
            Vec<crate::world::NodeId>,
            u32,
        ),
        Vec<crate::overlay::Wire>,
    )>,
    /// The profile binding sent with the last viewport request; the toolbar shows and edits it.
    viewport_render: Option<crate::render_profiles::ViewportRender>,
    /// Viewport A/B: raw samples instead of the denoised image (view state, not the scene).
    raw_view: bool,
    /// Exposure parked by the toolbar's EV bypass (`exposure_control`).
    exposure_hold: f32,
    show_ui: bool,
    status: String,
    event_bus: playa_events::EventBus,
    control: Option<app_control::Server>,
    control_started: bool,
    frame_ms: f32,
    seed: u32,
    /// `FRAC_SNAP=out.png [FRAC_SNAP_PRESET=i] [FRAC_SNAP_SPP=n]`: screenshot the window once the
    /// thumbnails and n viewport samples are done, then quit (for docs).
    snap: Option<Snap>,
    /// The armed window screenshot (`window_shot`) and whether a frame was drawn since it was
    /// requested: the window captures it only then, so a menu that asked for it is closed.
    window_shot: Option<(crate::window_shot::WindowShot, bool)>,
    /// An error the window exits with (non-zero): a failed `FRAC_SNAP` (`take_fatal`).
    fatal: Option<String>,
    /// Flight via cam-controls' inertial `SpaceFlight`, alive while RMB is held and
    /// while its momentum coasts after release.
    fly: Option<cam_controls::SpaceFlight>,
    /// Last speed scale sent to the flight rig (Shift / Alt latch, 1.0 = normal).
    fly_speed_scale: f32,
    /// Houdini orbit rig (LMB tumble, MMB pan, wheel zoom). The scene camera stays the
    /// source of truth: the rig is re-seeded from it every frame and only carries the coast.
    orbit: cam_controls::HoudiniOrbit,
    orbit_drag: cam_controls_egui::OrbitDragState,
    /// The camera as the orbit coast last wrote it; any other edit stops the coast.
    orbit_written: Option<crate::scene::Camera>,
    /// Persistent mouse sensitivity and flight speed (Settings > Controls).
    controls: Controls,
}

/// The `FRAC_SNAP` documentation screenshot (`App::handle_snap`), its environment read once.
#[derive(Clone, Debug)]
struct Snap {
    path: PathBuf,
    /// Viewport samples to wait for (`FRAC_SNAP_SPP`, default 256), capped by the profile's target.
    spp: u32,
    /// Wait for every gallery / bookmark / swatch thumbnail too (`FRAC_SNAP_WAIT_THUMBS` != 0).
    wait_thumbs: bool,
    /// The window shot is armed: nothing more to do but wait for it.
    requested: bool,
}

impl Snap {
    /// The `FRAC_SNAP*` environment, read once: None without `FRAC_SNAP`.
    fn from_env() -> Result<Option<Self>, String> {
        let Some(path) = std::env::var_os("FRAC_SNAP") else {
            return Ok(None);
        };
        Ok(Some(Self {
            path: PathBuf::from(path),
            spp: env_number("FRAC_SNAP_SPP")?.unwrap_or(256),
            wait_thumbs: !std::env::var("FRAC_SNAP_WAIT_THUMBS").is_ok_and(|v| v == "0"),
            requested: false,
        }))
    }
}

/// Environment number `name`: None when unset; set but malformed is an error for the status
/// line, never the default silently (a typo would otherwise shoot at the wrong sample count).
fn env_number<T: std::str::FromStr>(name: &str) -> Result<Option<T>, String>
where
    T::Err: std::fmt::Display,
{
    match std::env::var(name) {
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(format!("{name}: {error}")),
        Ok(value) => value
            .parse()
            .map(Some)
            .map_err(|error| format!("{name}={value:?}: {error}")),
    }
}

/// Shared fitted tile geometry for scene, workspace-material and library cards.
fn thumbnail_grid(width: f32, maximum: Vec2, gap: f32) -> (usize, Vec2) {
    let width = width.max(1.0);
    let columns = ((width + gap) / (maximum.x + gap)).floor().max(1.0) as usize;
    let cell = ((width - gap * (columns - 1) as f32) / columns as f32).clamp(1.0, maximum.x);
    (columns, Vec2::new(cell, cell * maximum.y / maximum.x))
}

/// The bookmark directory, `~/.warpbro/bookmarks`.
fn bookmarks_dir() -> PathBuf {
    crate::warpbro_dir().join("bookmarks")
}

/// The saved bookmarks in file-name order, and why the others could not be read (a missing
/// directory is no error: nothing saved yet).
fn load_bookmarks() -> (Vec<Entry>, Vec<String>) {
    let dir = bookmarks_dir();
    let mut errors = Vec::new();
    let listing = match std::fs::read_dir(&dir) {
        Ok(listing) => listing,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return (Vec::new(), errors);
        }
        Err(error) => {
            errors.push(format!("{}: {error}", dir.display()));
            return (Vec::new(), errors);
        }
    };
    let mut entries: Vec<Entry> = Vec::new();
    for item in listing {
        let path = match item {
            Ok(item) => item.path(),
            Err(error) => {
                errors.push(format!("{}: {error}", dir.display()));
                continue;
            }
        };
        if path.extension().is_none_or(|x| x != "json") {
            continue;
        }
        match std::fs::read_to_string(&path)
            .map_err(|error| error.to_string())
            .and_then(|text| crate::io_service::decode_scene(&text))
        {
            Ok(scene) => entries.push(Entry {
                scene,
                thumb: None,
                path: Some(path),
            }),
            Err(error) => errors.push(format!("{}: {error}", path.display())),
        }
    }
    entries.sort_by(|a, b| a.path.cmp(&b.path));
    (entries, errors)
}

/// The card a finished gallery / bookmark thumbnail of `scene` belongs to: the first one still
/// without a thumbnail showing that scene (equal scenes fill one card each, in order; the
/// list may have changed while it rendered, so not by index).
fn pending_thumbnail_entry(entries: &[Entry], scene: &Scene) -> Option<usize> {
    entries
        .iter()
        .position(|entry| entry.thumb.is_none() && entry.scene == *scene)
}

/// The window's display state (`soluwell_viewport::display::DisplayState`), which the window
/// stores into the egui context every frame; None before the first frame. The one reader.
fn display_state(ctx: &egui::Context) -> Option<soluwell_viewport::display::DisplayState> {
    ctx.data(|d| {
        d.get_temp::<soluwell_viewport::display::DisplayState>(
            soluwell_viewport::display::state_id(),
        )
    })
}

/// What the window shows the viewport on, read from `soluwell_viewport::display::DisplayState` by everything
/// that depends on it (the viewport canvas and its snapshots): HDR output or not, and the SDR
/// white in nits the present pass maps canvas 1.0 to. No window state yet: SDR, 100 nits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Monitor {
    pub hdr: bool,
    pub white_nits: f32,
}
impl Monitor {
    pub(crate) fn read(ctx: &egui::Context) -> Self {
        let state = display_state(ctx);
        Self {
            hdr: state.as_ref().is_some_and(|s| s.output.is_hdr()),
            white_nits: state.as_ref().map_or(100.0, |s| s.target.white),
        }
    }
    /// Nits of relative light's 1.0 (SDR reference white) in an HDR file: the monitor's SDR white
    /// when it shows HDR, so the file is as bright as the screen, else ITU-R BT.2408's 203 nits.
    /// Viewport snapshots and window screenshots share this rule.
    pub(crate) fn sdr_white_nits(self) -> f32 {
        if self.hdr {
            self.white_nits
        } else {
            crate::color::BT2408_SDR_WHITE_NITS
        }
    }
}

/// The colour of every status warning (preview, OIDN failure, out of steps).
const WARNING: Color32 = Color32::from_rgb(230, 180, 60);

/// The status bar's sections in their order (`App::status_item`), the same in the resizable and
/// the fixed layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StatusItem {
    Gpu,
    Flight,
    Size,
    Samples,
    Speed,
    Batch,
    Denoise,
    UiFps,
    March,
    Message,
}
impl StatusItem {
    const ALL: [Self; 10] = [
        Self::Gpu,
        Self::Flight,
        Self::Size,
        Self::Samples,
        Self::Speed,
        Self::Batch,
        Self::Denoise,
        Self::UiFps,
        Self::March,
        Self::Message,
    ];
    /// The resizable layout's default section widths, in `ALL` order (0: the rest).
    const WIDTHS: [f32; 10] = [
        200.0, 95.0, 95.0, 220.0, 130.0, 100.0, 170.0, 80.0, 150.0, 0.0,
    ];
}

/// The status bar's sample progress and adaptive tile convergence indicator, shared by the
/// resizable and fixed status bars.
fn sample_status_ui(
    ui: &mut egui::Ui,
    frame: &Frame,
    target_spp: u32,
    preview: bool,
    bar_width: f32,
) {
    if preview {
        ui.colored_label(WARNING, "preview");
        return;
    }
    let (active, total) = frame.active_tiles;
    let hover = if frame.converged && total > 0 {
        Some(format!(
            "Adaptive sampling: all {total} tiles converged at {} / {target_spp} spp.",
            frame.samples
        ))
    } else if total > 0 && active < total {
        let converged_pct = ((total - active) * 100) / total;
        Some(format!(
            "Adaptive sampling: {active} of {total} tiles still sampling ({converged_pct}% converged)."
        ))
    } else {
        None
    };
    let label = ui.label(format!("{} / {} spp", frame.samples, target_spp));
    let fraction = if frame.converged {
        1.0
    } else {
        frame.samples as f32 / target_spp.max(1) as f32
    };
    let bar_w = (ui.available_width() - 4.0).clamp(0.0, bar_width.max(0.0));
    let mut bar = egui::ProgressBar::new(fraction).desired_width(bar_w);
    if frame.converged {
        bar = bar.text("converged");
    } else if total > 0 && active < total {
        let converged_pct = ((total - active) * 100) / total;
        bar = bar.text(format!("{converged_pct}% converged"));
    }
    let bar_resp = ui.add(bar);
    if let Some(hover) = &hover {
        label.on_hover_text(hover);
        bar_resp.on_hover_text(hover);
    }
}

/// The status warning for samples with an unresolved march (`gpu::Outcome`): a nonzero share
/// means holes or missing light the step budget (Render > March steps) is too small to close.
fn march_limit_label(ui: &mut egui::Ui, unresolved: f32) {
    if unresolved > 0.0 {
        let percent = 100.0 * unresolved;
        let text = if percent < 0.01 {
            "<0.01% out of steps".to_owned()
        } else {
            format!("{percent:.2}% out of steps")
        };
        ui.label(
            RichText::new(text)
                .color(WARNING),
        )
        .on_hover_text(
            "Samples with a march out of steps: camera rays show the background, bounce and shadow rays bring no light. Raise Render > March steps.",
        );
    }
}

/// Settings panel pages in tab order: the one name of a page wherever it is opened (menu,
/// toolbar gear, cross-page links, `FRAC_SETTINGS`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SettingsPage {
    Display,
    Color,
    Controls,
    Fonts,
    Animation,
    Render,
}
impl SettingsPage {
    const ALL: [Self; 6] = [
        Self::Display,
        Self::Color,
        Self::Controls,
        Self::Fonts,
        Self::Animation,
        Self::Render,
    ];
    fn category(self) -> soluwell_panels::prefs::Category<'static> {
        use soluwell_config::icons as ph;
        match self {
            Self::Display => soluwell_panels::prefs::Category::new(ph::MONITOR, "Display"),
            Self::Color => soluwell_panels::prefs::Category::new(ph::MONITOR, "Color"),
            Self::Controls => soluwell_panels::prefs::Category::new(ph::MONITOR, "Controls"),
            Self::Fonts => soluwell_panels::prefs::Category::new(ph::TEXT, "Fonts"),
            Self::Animation => soluwell_panels::prefs::Category::new(ph::MEDIA, "Animation"),
            Self::Render => soluwell_panels::prefs::Category::new(ph::GEAR, "Render & Viewport"),
        }
    }
    /// The page named `name` (case-insensitive tab label).
    fn named(name: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|page| page.category().label.eq_ignore_ascii_case(name))
    }
}

/// Settings > Controls span of the manipulator size, and the + / − step.
const GIZMO_SIZE_MIN: f32 = 40.0;
const GIZMO_SIZE_MAX: f32 = 400.0;
const GIZMO_SIZE_STEP: f32 = 10.0;

fn step_gizmo_size(size: f32, larger: bool) -> f32 {
    let next = if larger {
        size + GIZMO_SIZE_STEP
    } else {
        size - GIZMO_SIZE_STEP
    };
    next.clamp(GIZMO_SIZE_MIN, GIZMO_SIZE_MAX)
}

/// The Settings > Controls page: viewport flight preferences (`inertia` turns them into the
/// shared `cam_controls` flight settings, the one place WarpBro configures the flight rig) and
/// the mouse mapping of the slot strips.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
struct Controls {
    look_sensitivity: f32,
    fly_speed: f32,
    /// Seconds for the translation speed to fall to 1/e once thrust stops.
    translate_decay: f32,
    /// Seconds for the rotation rate (roll, a released orbit drag) to fall to 1/e; also the
    /// horizon-lock levelling speed.
    rotate_decay: f32,
    /// Flight mouse look eases onto the pointer's turn (time constant `look_smoothing`, s)
    /// instead of turning at once; either way the turn is the pointer's, never more.
    inertial_look: bool,
    look_smoothing: f32,
    /// Horizon lock: roll angle (degrees) past which the lock flips to the next plane.
    flip_degrees: f32,
    /// Thrust multipliers while Shift (fast) / Alt (slow) are held.
    fast_multiplier: f32,
    slow_multiplier: f32,
    /// Slot strips (CamClip, colour presets): the left button stores and the right recalls,
    /// instead of the default left = recall, right = store (`hotkeys::slot_click`).
    swap_slot_buttons: bool,
    /// The viewport manipulator's size in points and its line width.
    gizmo_size: f32,
    gizmo_stroke: f32,
    /// Draw a wireframe bounding box around selected fractal object(s) in the viewport.
    selection_bounds: bool,
    /// Holding Ctrl during orbit or flight eases the view to the nearest world coordinate
    /// plane or axis (within 15° of two planes) with this time constant in seconds.
    align_seconds: f32,
}
impl Default for Controls {
    fn default() -> Self {
        Self {
            look_sensitivity: 0.5,
            fly_speed: 1.0,
            translate_decay: 0.96,
            rotate_decay: 1.0,
            inertial_look: true,
            look_smoothing: 0.08,
            flip_degrees: 48.6,
            fast_multiplier: 4.0,
            slow_multiplier: 0.1,
            swap_slot_buttons: false,
            gizmo_size: 120.0,
            gizmo_stroke: 3.0,
            selection_bounds: true,
            align_seconds: 0.5,
        }
    }
}
impl Controls {
    /// The flight rig settings for a scene of framing radius `radius` (thrust scales with it).
    fn inertia(&self, radius: f32) -> cam_controls::InertiaSettings {
        let defaults = cam_controls::InertiaSettings::default();
        cam_controls::InertiaSettings {
            // Relative mouse look (pointer motion), not cursor-deflection steering.
            mouse_steer: cam_controls::MouseSteer::Relative,
            // Base rates: look 5 mrad per pixel (smoothed or not: the smoothing only eases the
            // same turn in), roll 2 rad/s², thrust 2 radii/s² (cruise 2 radii x translate
            // decay per second).
            look_sensitivity: 0.005 * self.look_sensitivity,
            roll_sensitivity: 2.0,
            thrust_sensitivity: 2.0 * radius * self.fly_speed,
            linear_damping: 1.0 / self.translate_decay.max(0.02),
            angular_damping: 1.0 / self.rotate_decay.max(0.02),
            inertial_look: self.inertial_look,
            look_smoothing: self.look_smoothing,
            // Keep the hold angle 15 degrees past the flip angle: a held key always flips.
            lock_flip: self.flip_degrees.to_radians(),
            lock_hold: (self.flip_degrees + 15.0).to_radians(),
            fast_multiplier: self.fast_multiplier,
            slow_multiplier: self.slow_multiplier,
            ..defaults
        }
    }

    /// The orbit navigation settings for a view of vertical FOV `fov_y` (radians) that is
    /// `height` points tall: pan moves the target exactly with the cursor, the wheel zooms
    /// exponentially, and a released drag coasts at its own speed, decaying like the flight
    /// rotation (`rotate_decay`).
    fn navigation(&self, fov_y: f32, height: f32) -> cam_controls::CameraNavigationSettings {
        cam_controls::CameraNavigationSettings {
            zoom_mode: cam_controls::ZoomMode::Exponential,
            zoom_to_cursor: false,
            orbit_sensitivity: 0.1_f32.to_radians() * self.look_sensitivity,
            zoom_scroll_exponential: 0.0015,
            pan_sensitivity: 2.0 * (fov_y * 0.5).tan() / height.max(1.0),
            orbit_release_inertia_scale: 1.0,
            inertia_friction: 1.0 / self.rotate_decay.max(0.02),
            ..cam_controls::CameraNavigationSettings::default()
        }
    }
}

/// The preferences file (`settings_path`): exactly what `App::current_settings` reads off the
/// app and `App::apply_settings` puts back, every field required (no compat defaults: an
/// older file is reported and replaced by the defaults).
#[derive(PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Settings {
    display: soluwell_viewport::display::DisplayPrefs,
    /// The colour selection of new documents (`App::default_colour`).
    colour: crate::ocio::Sel,
    colour_presets: crate::ocio::ColourPresets,
    panel: soluwell_panels::prefs::PrefsPanelState,
    controls: Controls,
    recorder: crate::camera_recorder::Options,
    /// Navigation gestures are recorded as takes (the Takes panel's record switch).
    record_takes: bool,
    /// Minimum gesture duration (seconds) required to keep a recorded take.
    min_take_secs: f64,
    /// The CamClip's last state, kept between scenes and runs (`App::camera_slots`).
    camera_slots: crate::camera_slots::CameraSlots,
    fonts: dock::Fonts,
    layout: Option<String>,
    toolbar: soluwell_viewport::toolbar::ToolbarState,
    layouts: soluwell_chrome::layout_manager::LayoutStore,
    export: crate::export::ExportJob,
    gui_fps: u32,
    status_layout: soluwell_chrome::statusbar::StatusBarLayout,
    status_resizable: bool,
    attribute_metrics: soluwell_config::AttrMetrics,
    auto_key: bool,
    new_key: curves::Tan,
    timeline_outline_width: f32,
    file_dialogs: crate::file_dialogs::History,
    add_menu: crate::world_ui::AddMenuMemory,
    preview_cache: PreviewCache,
}
impl Default for Settings {
    fn default() -> Self {
        Self {
            display: Default::default(),
            colour: crate::color::default_selection(),
            colour_presets: Default::default(),
            panel: Default::default(),
            controls: Default::default(),
            recorder: Default::default(),
            record_takes: true,
            min_take_secs: crate::takes::DEFAULT_MIN_SECS,
            camera_slots: Default::default(),
            fonts: Default::default(),
            layout: None,
            toolbar: Default::default(),
            layouts: Default::default(),
            export: Default::default(),
            gui_fps: 60,
            status_layout: Default::default(),
            status_resizable: true,
            attribute_metrics: Default::default(),
            auto_key: false,
            new_key: curves::Tan::Smooth,
            timeline_outline_width: 340.0,
            file_dialogs: Default::default(),
            add_menu: Default::default(),
            preview_cache: Default::default(),
        }
    }
}

/// One Settings grid row bound to the value it edits, so its label is written once: the grid
/// shows it and an edit of that row writes the value back (`prefs_grid`).
enum PrefRow<'a> {
    /// A slider over `range`; Ctrl-click / the row menu resets it to `default`.
    Float {
        label: &'static str,
        value: &'a mut f32,
        range: std::ops::RangeInclusive<f32>,
        default: f32,
    },
    /// A checkbox.
    Bool {
        label: &'static str,
        value: &'a mut bool,
        default: bool,
    },
}
impl PrefRow<'_> {
    fn label(&self) -> &'static str {
        match self {
            Self::Float { label, .. } | Self::Bool { label, .. } => label,
        }
    }
    fn field(&self) -> AttrField {
        match self {
            Self::Float {
                label,
                value,
                range,
                default,
            } => AttrField::new(*label, AttrValue::Float(**value))
                .with_ui_options(vec![range.start().to_string(), range.end().to_string()])
                .with_default(AttrValue::Float(*default)),
            Self::Bool {
                label,
                value,
                default,
            } => AttrField::new(*label, AttrValue::Bool(**value))
                .with_default(AttrValue::Bool(*default)),
        }
    }
    /// Store an edited value of this row's kind.
    fn set(&mut self, edited: AttrValue) {
        match (self, edited) {
            (Self::Float { value, .. }, AttrValue::Float(v)) => **value = v,
            (Self::Bool { value, .. }, AttrValue::Bool(v)) => **value = v,
            _ => {}
        }
    }
}

/// Draws Settings `rows` as an attribute grid with the Attribute Editor's metrics; an edit of a
/// row is written to the value that row is bound to.
fn prefs_grid(
    ui: &mut egui::Ui,
    metrics: AttrMetrics,
    state: &mut AttrGridState,
    bound: &mut [PrefRow<'_>],
) {
    let mut rows: Vec<AttrField> = bound.iter().map(PrefRow::field).collect();
    struct Plain;
    impl AttrGridHooks for Plain {}
    let metrics = metrics.normalized();
    // Preferences are scalar rows with no animation controls: the editor needs one field, not the
    // attribute grid's three, and the label column keeps the room (a label never collapses to "...").
    let mut config = soluwell_panels::attr_grid::AttrGridConfig {
        prefix_width: 0.0,
        action_width: 0.0,
        min_editor_width: metrics.numeric_width + metrics.component_gap + metrics.field_height,
        ..crate::world_ui::grid_config(metrics)
    };
    ui.scope(|ui| {
        metrics.apply(ui);
        config.value_box_width = Some(soluwell_panels::attr_grid::value_box_width(
            ui,
            &rows,
            metrics.numeric_width,
        ));
        // The label column starts at the old table's width, scaled with the body font.
        if state.table.widths.is_empty() {
            let body = ui
                .style()
                .text_styles
                .get(&egui::TextStyle::Body)
                .map_or(13.0, |f| f.size);
            state.table.widths.push(130.0 * (body / 13.0).max(1.0));
        }
        // One object per row: store the value. A gesture's closing `Done` that wrote nothing
        // more (`parts` empty) carries no change.
        for edit in
            render_grid_with_config(ui, &mut rows, state, &HashSet::new(), &config, &mut Plain)
        {
            if !edit.parts.is_empty()
                && let Some(row) = bound.iter_mut().find(|row| row.label() == edit.key)
            {
                row.set(edit.value);
            }
        }
    });
}

fn settings_path() -> PathBuf {
    crate::warpbro_dir().join("settings.json")
}

fn freeze_world_scene(scene: &Scene, document: &WorldDocument) -> Scene {
    let mut scene = scene.clone();
    scene.document = Some(Box::new(document.clone()));
    scene.animation.first = document.first;
    scene.animation.last = document.last;
    scene.animation.fps = document.fps;
    scene
}

impl App {
    pub(crate) fn new() -> Self {
        let gallery: Vec<Entry> = Scene::gallery()
            .into_iter()
            .map(|scene| Entry {
                scene,
                thumb: None,
                path: None,
            })
            .collect();
        // `start_scene` before the settings: its colour is the default selection here.
        let scene = Scene::preset(Formula::PRESETS[0]);
        let (bookmarks, unreadable) = load_bookmarks();
        for error in &unreadable {
            log::error!("Bookmark not loaded: {error}");
        }
        let status = match unreadable.as_slice() {
            [] => String::new(),
            [error] => format!("Bookmark not loaded: {error}"),
            [error, ..] => format!(
                "{} bookmarks not loaded (see the log), e.g. {error}",
                unreadable.len()
            ),
        };
        let (snap, status) = match Snap::from_env() {
            Ok(snap) => (snap, status),
            Err(error) => {
                log::error!("Documentation screenshot not armed: {error}");
                (None, format!("Documentation screenshot not armed: {error}"))
            }
        };

        let wake = crate::render_service::Wake::default();
        let event_bus = playa_events::EventBus::new();
        let mut world_ui = crate::world_ui::WorldUi::default();
        world_ui.set_emitter(event_bus.emitter());

        Self {
            renderer: RenderService::spawn(wake.clone()),
            io: crate::io_service::IoService::spawn(wake.clone()),
            gpu_name: "Initializing CUDA…".into(),
            frame: None,
            frame_from_cache: false,
            request: None,
            viewport_stamp: None,
            generation: 0,
            staged_frame: None,
            staged_output: None,
            layouts: Default::default(),
            export: crate::export::ExportController::new(wake.clone()),
            thumb_pending: None,
            thumb_failed: HashSet::new(),
            pick_pending: None,
            pick_sequence: 0,
            gui_fps: 60,
            origin: scene.clone(),
            card: None,
            default_colour: scene.colour.clone(),
            last_scene: scene.clone(),
            viewport_scene: None,
            overlays: None,
            viewport_render: None,
            scene: scene.clone(),
            world: crate::world::WorldEditor::new(crate::world::WorldDocument::from_scene(&scene)),
            world_ui,
            preview: Default::default(),
            preview_key: None,
            preview_sequence: 0,
            material_targets: Vec::new(),
            material_selection_stamp: None,
            preview_progress: None,
            evaluated_world: None,
            load_revision: 0,

            showing_preview: false,
            hdr_view: std::sync::Arc::new(std::sync::Mutex::new(
                soluwell_viewport::hdr_view::HdrView::new(),
            )),
            display: Default::default(),
            colour: crate::ocio::State::new(scene.colour.clone(), wake.clone()),
            prefs: Default::default(),
            recorder_options: Default::default(),
            record_takes: true,
            min_take_secs: crate::takes::DEFAULT_MIN_SECS,
            camera_slots: Default::default(),
            takes_recorder: crate::takes::Recorder::with_min_secs(crate::takes::DEFAULT_MIN_SECS),
            navigated: false,
            takes_view: Default::default(),
            tool: ViewTool::Select,
            gizmo: soluwell_viewport::gizmo::Gizmo::new(
                soluwell_viewport::gizmo::GizmoConfig::default(),
            ),
            gizmo_gesture: None,
            drag_recorder: crate::takes::DragRecorder::with_min_secs(
                crate::takes::DEFAULT_MIN_SECS,
            ),
            prefs_grids: Default::default(),
            dock: dock::default_layout(),
            dock_focus: None,
            panels_to_open: Vec::new(),
            toolbar: Default::default(),
            fonts: Default::default(),
            applied_fonts: None,
            viewport_visible: false,
            viewport_rect: None,
            config_picker: egui_file_dialog::FileDialog::new(),
            scene_picker: None,
            scene_file_path: None,
            scene_file_sequence: 0,
            scene_file_pending: None,
            bookmark_ops: Vec::new(),
            wake,
            templates: Vec::new(),
            templates_requested: false,
            templates_pending: false,
            persisted_settings: None,
            snapshot_before: None,
            #[cfg(test)]
            settings_serializations: 0,
            #[cfg(test)]
            snapshot_clones: 0,
            status_layout: Default::default(),
            status_resizable: true,
            gallery,
            bookmarks,

            swatches: (0..crate::materials::PRESETS.len()).map(|_| None).collect(),
            material_gallery: Default::default(),
            preview_cache: Default::default(),
            target_spp: 1024,
            spp_per_frame: 1,
            last_change: Instant::now(),
            raw_view: false,
            exposure_hold: 0.0,
            show_ui: true,
            event_bus,
            control: None,
            control_started: false,
            status,
            frame_ms: 16.0,
            seed: 0,
            fly: None,
            fly_speed_scale: 1.0,
            orbit: cam_controls::HoudiniOrbit::default(),
            orbit_drag: cam_controls_egui::OrbitDragState::default(),
            orbit_written: None,
            controls: Default::default(),
            snap,
            window_shot: None,
            fatal: None,
        }
        .with_settings()
        .with_snap_preset()
    }

    /// The scene WarpBro opens with and File > New loads: the first gallery preset
    /// (`Scene::gallery` starts with the family presets) in the colour selection of new
    /// documents (`default_colour`).
    fn start_scene(&self) -> Scene {
        let mut scene = Scene::preset(Formula::PRESETS[0]);
        scene.colour = self.default_colour.clone();
        scene
    }

    fn with_settings(mut self) -> Self {
        // No file is a first start; an unreadable one is reported, since the defaults that
        // replace it are saved over it at the next change.
        let path = settings_path();
        let settings = match std::fs::read(&path) {
            Ok(data) => {
                Some(serde_json::from_slice::<Settings>(&data).map_err(|error| error.to_string()))
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => Some(Err(error.to_string())),
        };
        if let Some(Err(error)) = &settings {
            log::error!("Preferences {}: {error}", path.display());
            self.status = format!("Preferences unreadable, defaults in use: {error}");
        }
        if let Some(Ok(settings)) = settings {
            self.apply_settings(settings);
        }
        if let Ok(category) = std::env::var("FRAC_SETTINGS") {
            if category.eq_ignore_ascii_case("export") {
                self.panels_to_open.push(dock::Panel::Export);
            } else {
                match SettingsPage::named(&category) {
                    Some(page) => self.open_settings(page),
                    None => self.status = format!("FRAC_SETTINGS: no settings page {category}"),
                }
            }
        }
        // The first document is a new one: it starts in the saved colour selection.
        let scene = self.start_scene();
        self.load(scene, Load::New);
        self
    }

    /// Put saved preferences into effect (`with_settings`). Destructured field by field, so a
    /// new `Settings` field does not compile until it is applied here and read back in
    /// `current_settings`.
    fn apply_settings(&mut self, settings: Settings) {
        let Settings {
            display,
            colour,
            colour_presets,
            panel,
            controls,
            recorder,
            record_takes,
            min_take_secs,
            camera_slots,
            fonts,
            layout,
            toolbar,
            layouts,
            export,
            gui_fps,
            status_layout,
            status_resizable,
            attribute_metrics,
            auto_key,
            new_key,
            timeline_outline_width,
            file_dialogs,
            add_menu,
            preview_cache,
        } = settings;
        self.display = display;
        self.default_colour = colour;
        self.colour.presets = colour_presets;
        self.prefs = panel;
        self.controls = controls;
        self.recorder_options = recorder;
        self.record_takes = record_takes;
        if min_take_secs.is_finite() {
            self.min_take_secs = min_take_secs.clamp(crate::takes::MIN_SECS, 600.0);
            self.takes_recorder.min_secs = self.min_take_secs;
            self.drag_recorder.min_secs = self.min_take_secs;
        }
        self.camera_slots = camera_slots;
        self.fonts = fonts;
        if let Some(blob) = layout {
            // A layout saved before the dock engine change is refused here once (no compat).
            match dock::restore_layout(&blob) {
                Ok(layout) => self.dock = layout,
                Err(error) => {
                    self.status =
                        format!("Saved layout could not be restored ({error}); using default")
                }
            }
        }
        self.toolbar = toolbar;
        self.layouts.store = layouts;
        self.export.restore(export);
        self.gui_fps = gui_fps.clamp(15, 240);
        self.status_layout = status_layout;
        self.status_resizable = status_resizable;
        self.world_ui.attribute_metrics = attribute_metrics.normalized();
        self.world_ui.auto_key = auto_key;
        self.world_ui.new_key = new_key;
        self.world.new_key = new_key;
        if timeline_outline_width.is_finite() {
            self.world_ui.timeline_outline_width = timeline_outline_width.max(80.0);
        }
        self.world_ui.file_dialogs = file_dialogs;
        self.world_ui.set_add_menu_memory(add_menu);
        self.preview_cache = preview_cache;
    }

    /// The preferences as they are now: what the autosave writes (`changed_settings_json`).
    fn current_settings(&self) -> Settings {
        Settings {
            display: self.display,
            colour: self.default_colour.clone(),
            colour_presets: self.colour.presets.clone(),
            panel: self.prefs.clone(),
            controls: self.controls,
            recorder: self.recorder_options,
            record_takes: self.record_takes,
            min_take_secs: self.min_take_secs,
            camera_slots: self.camera_slots,
            fonts: self.fonts.clone(),
            layout: self.layouts.cache.blob.clone(),
            toolbar: self.toolbar,
            layouts: self.layouts.store.clone(),
            export: self.export.job.clone(),
            gui_fps: self.gui_fps,
            status_layout: self.status_layout.clone(),
            status_resizable: self.status_resizable,
            attribute_metrics: self.world_ui.attribute_metrics,
            auto_key: self.world_ui.auto_key,
            new_key: self.world_ui.new_key,
            timeline_outline_width: self.world_ui.timeline_outline_width,
            file_dialogs: self.world_ui.file_dialogs.clone(),
            add_menu: self.world_ui.add_menu_memory(),
            preview_cache: self.preview_cache,
        }
    }

    /// Open the Settings panel on `page`.
    pub(crate) fn open_settings(&mut self, page: SettingsPage) {
        self.prefs.selected = page as usize;
        self.panels_to_open.push(dock::Panel::Settings);
    }

    pub(super) fn profile_ui_actions(&mut self, actions: crate::render_profiles_ui::Actions) {
        if let Some(id) = actions.edit {
            self.emit_app(AppAction::InspectNode(id));
        }
        if let Some(error) = actions.error {
            self.status = error;
        }
    }

    /// Set a switch of the ViewportSettings node (`/viewport/...`) at the playhead, one undo
    /// step, as the Attribute Editor would: View > Pause rendering and the viewport toolbar's
    /// visibility menu.
    #[cfg(test)]
    fn set_viewport_flag(&mut self, path: &str, value: bool) {
        let frame = f64::from(self.world_ui.playhead);
        let result = self.world.document.viewport_settings_id().and_then(|id| {
            self.world
                .execute(crate::world::WorldCommand::SetAttribute {
                    id,
                    path: path.into(),
                    value: serde_json::json!(value),
                    frame,
                })
        });
        if let Err(error) = result {
            self.status = error;
        }
    }

    /// The undo gesture id of the active camera's viewport edits (navigation, the toolbar's
    /// exposure drag): one id, so they share one open undo step.
    fn camera_gesture(&self) -> u64 {
        egui::Id::new(("viewport_camera_edit", self.world.document.active_camera)).value()
    }

    /// Settings layout and category bodies adapted directly from exr-view::ui_settings.
    fn settings_ui(&mut self, ui: &mut egui::Ui) {
        let state = display_state(ui.ctx());
        let emitter = self.event_emitter();
        let mut draft = self.current_settings();
        let before = match serde_json::to_value(&draft) {
            Ok(value) => value,
            Err(error) => {
                self.status = error.to_string();
                return;
            }
        };
        let mut colour_actions = Vec::new();
        let mut browse = false;
        let mut category = draft.panel.selected;
        let mut destination = None;
        let mut prefs = std::mem::take(&mut draft.panel);
        let categories = SettingsPage::ALL.map(SettingsPage::category);
        let opts = soluwell_panels::prefs::PanelOpts {
            reset: true,
            ..Default::default()
        };
        let reset = soluwell_panels::prefs::draw(
            ui,
            "settings",
            &mut prefs,
            &categories,
            &opts,
            |ui, idx| {
                category = idx;
                match SettingsPage::ALL
                    .get(idx)
                    .copied()
                    .unwrap_or(SettingsPage::Display)
                {
                    SettingsPage::Display => {
                        soluwell_viewport::display::settings_ui(
                            ui,
                            "settings.display",
                            &mut draft.display,
                            state.as_ref(),
                        );
                        if state
                            .as_ref()
                            .is_some_and(|s| !s.available.iter().any(|o| o.is_hdr()))
                        {
                            ui.label("This window surface offers SDR only. PQ/HDR targets in Color still render and export HDR; the screen uses an SDR preview.");
                        }
                        ui.add(
                            egui::DragValue::new(&mut draft.gui_fps)
                                .range(15..=240)
                                .suffix(" GUI FPS"),
                        );
                        ui.add_space(8.0);
                        if ui.button("Colour management & monitor presets…").clicked() {
                            destination = Some(SettingsPage::Color);
                        }
                    }
                    SettingsPage::Color => {
                        soluwell_panels::prefs::section_header(ui, "Colour management");
                        if let Some(state) = &state {
                            ui.label(format!("Window output: {}.", state.output.label()));
                        }
                        ui.label("Monitor presets choose rendering and export. PQ/HDR remains available on SDR screens using an SDR preview. HDR 1000 nits is the rendering peak; SDR reference white controls UI brightness.");
                        if ui.button("Display output & reference white…").clicked() {
                            destination = Some(SettingsPage::Display);
                        }
                        ui.add_space(8.0);
                        let colour =
                            self.colour
                                .ui(ui, &mut browse, draft.controls.swap_slot_buttons);
                        if let Some(selection) = colour.selection {
                            colour_actions.push(AppAction::ColourSelection(selection));
                        }
                        if colour.reload {
                            colour_actions.push(AppAction::ReloadColour);
                        }
                        if let Some(presets) = colour.presets {
                            colour_actions.push(AppAction::ColourPresets(presets));
                        }
                    }
                    SettingsPage::Render => {
                        let actions = crate::render_profiles_ui::settings(
                            ui,
                            &self.world,
                            &emitter,
                            &mut self.world_ui,
                        );
                        self.profile_ui_actions(actions);
                    }
                    SettingsPage::Fonts => self.fonts_ui(ui, &mut draft.fonts),
                    SettingsPage::Animation => {
                        soluwell_panels::prefs::section_header(ui, "Keys");
                        ui.horizontal(|ui| {
                            ui.label("New key type")
                                .on_hover_text("Interpolation of every key created by Key, Auto Key and edits of animated values.");
                            for tan in [curves::Tan::Linear, curves::Tan::Smooth] {
                                ui.selectable_value(&mut draft.new_key, tan, tan.label());
                            }
                        });
                        soluwell_panels::prefs::section_header(ui, "Preview cache");
                        let defaults = PreviewCache::default();
                        let cache = &mut draft.preview_cache;
                        prefs_grid(
                            ui,
                            draft.attribute_metrics,
                            &mut self.prefs_grids[5],
                            &mut [
                                PrefRow::Float {
                                    label: "Share of installed RAM",
                                    value: &mut cache.fraction,
                                    range: 0.01..=0.9,
                                    default: defaults.fraction,
                                },
                                PrefRow::Float {
                                    label: "Kept for the system, GiB",
                                    value: &mut cache.reserve_gb,
                                    range: 0.0..=64.0,
                                    default: defaults.reserve_gb,
                                },
                            ],
                        );
                        ui.label("Played and cached previews keep their frames in RAM within this budget.");
                    }
                    SettingsPage::Controls => {
                        let am = draft.attribute_metrics;
                        let c = &mut draft.controls;
                        let d = Controls::default();
                        let grids = &mut self.prefs_grids;
                        soluwell_panels::prefs::section_header(ui, "Camera controls");
                        prefs_grid(
                            ui,
                            am,
                            &mut grids[0],
                            &mut [
                                PrefRow::Float {
                                    label: "Mouse sensitivity",
                                    value: &mut c.look_sensitivity,
                                    range: 0.1..=5.0,
                                    default: d.look_sensitivity,
                                },
                                PrefRow::Float {
                                    label: "Flight speed ×",
                                    value: &mut c.fly_speed,
                                    range: 0.02..=50.0,
                                    default: d.fly_speed,
                                },
                                PrefRow::Float {
                                    label: "Translate decay, s",
                                    value: &mut c.translate_decay,
                                    range: 0.02..=5.0,
                                    default: d.translate_decay,
                                },
                                PrefRow::Float {
                                    label: "Rotate decay, s",
                                    value: &mut c.rotate_decay,
                                    range: 0.02..=5.0,
                                    default: d.rotate_decay,
                                },
                                PrefRow::Bool {
                                    label: "Smooth look",
                                    value: &mut c.inertial_look,
                                    default: d.inertial_look,
                                },
                                PrefRow::Float {
                                    label: "Look smoothing, s",
                                    value: &mut c.look_smoothing,
                                    range: 0.0..=0.5,
                                    default: d.look_smoothing,
                                },
                                PrefRow::Float {
                                    label: "Horizon flip, °",
                                    value: &mut c.flip_degrees,
                                    range: 20.0..=85.0,
                                    default: d.flip_degrees,
                                },
                                PrefRow::Float {
                                    label: "Shift fast ×",
                                    value: &mut c.fast_multiplier,
                                    range: 1.0..=20.0,
                                    default: d.fast_multiplier,
                                },
                                PrefRow::Float {
                                    label: "Alt slow ×",
                                    value: &mut c.slow_multiplier,
                                    range: 0.01..=1.0,
                                    default: d.slow_multiplier,
                                },
                            ],
                        );
                        soluwell_panels::prefs::section_header(ui, "Camera alignment (hold Ctrl)");
                        prefs_grid(
                            ui,
                            am,
                            &mut grids[1],
                            &mut [PrefRow::Float {
                                label: "Ease time, s",
                                value: &mut c.align_seconds,
                                range: 0.0..=5.0,
                                default: d.align_seconds,
                            }],
                        );
                        soluwell_panels::prefs::section_header(ui, "Manipulator");
                        prefs_grid(
                            ui,
                            am,
                            &mut grids[2],
                            &mut [
                                PrefRow::Float {
                                    label: "Size, pt",
                                    value: &mut c.gizmo_size,
                                    range: GIZMO_SIZE_MIN..=GIZMO_SIZE_MAX,
                                    default: d.gizmo_size,
                                },
                                PrefRow::Float {
                                    label: "Line width, pt",
                                    value: &mut c.gizmo_stroke,
                                    range: 0.5..=10.0,
                                    default: d.gizmo_stroke,
                                },
                                PrefRow::Bool {
                                    label: "Selection bounding box",
                                    value: &mut c.selection_bounds,
                                    default: d.selection_bounds,
                                },
                            ],
                        );
                        soluwell_panels::prefs::section_header(ui, "Slot buttons");
                        prefs_grid(
                            ui,
                            am,
                            &mut grids[3],
                            &mut [PrefRow::Bool {
                                label: "Swap copy/paste mouse buttons",
                                value: &mut c.swap_slot_buttons,
                                default: d.swap_slot_buttons,
                            }],
                        );
                        ui.label(format!(
                            "CamClip and colour presets · {}",
                            crate::hotkeys::slot_hint(c.swap_slot_buttons, "copy", "paste")
                        ));
                        soluwell_panels::prefs::section_header(ui, "Attribute controls");
                        let defaults = AttrMetrics::default();
                        let metrics = &mut draft.attribute_metrics;
                        prefs_grid(
                            ui,
                            am,
                            &mut grids[4],
                            &mut [
                                PrefRow::Float {
                                    label: "Field height",
                                    value: &mut metrics.field_height,
                                    range: AttrMetrics::FIELD_HEIGHT_RANGE,
                                    default: defaults.field_height,
                                },
                                PrefRow::Float {
                                    label: "Numeric field width",
                                    value: &mut metrics.numeric_width,
                                    range: AttrMetrics::NUMERIC_WIDTH_RANGE,
                                    default: defaults.numeric_width,
                                },
                                PrefRow::Float {
                                    label: "Icon size",
                                    value: &mut metrics.icon_side,
                                    range: AttrMetrics::ICON_SIDE_RANGE,
                                    default: defaults.icon_side,
                                },
                                PrefRow::Float {
                                    label: "Row spacing",
                                    value: &mut metrics.row_gap,
                                    range: AttrMetrics::ROW_GAP_RANGE,
                                    default: defaults.row_gap,
                                },
                                PrefRow::Float {
                                    label: "Component spacing",
                                    value: &mut metrics.component_gap,
                                    range: AttrMetrics::COMPONENT_GAP_RANGE,
                                    default: defaults.component_gap,
                                },
                            ],
                        );
                        *metrics = metrics.normalized();
                        ui.label(
                            "Attribute Editor, Timeline and Render Settings share these sizes.",
                        );
                        ui.label("RMB: fly · wheel: forward / back · L: horizon/free flight");
                        ui.label("H: restore camera · F: frame bounds (without RMB)");
                    }
                }
            },
        );
        draft.panel = prefs;
        if reset {
            let defaults = Settings::default();
            match SettingsPage::ALL.get(category) {
                Some(SettingsPage::Display) => {
                    draft.display = defaults.display;
                    draft.gui_fps = defaults.gui_fps;
                }
                Some(SettingsPage::Color) => {
                    colour_actions.push(AppAction::ColourSelection(defaults.colour));
                    colour_actions.push(AppAction::ReloadColour);
                }
                Some(SettingsPage::Fonts) => draft.fonts = defaults.fonts,
                Some(SettingsPage::Animation) => {
                    draft.new_key = defaults.new_key;
                    draft.preview_cache = defaults.preview_cache;
                }
                Some(SettingsPage::Render) => {
                    self.status = "Reset profile values in Attribute Editor".into();
                }
                Some(SettingsPage::Controls) => {
                    draft.controls = defaults.controls;
                    draft.attribute_metrics = defaults.attribute_metrics;
                }
                None => {}
            }
        }
        if let Some(page) = destination {
            draft.panel.selected = page as usize;
        }
        if browse {
            self.emit_app(AppAction::BrowseColour);
        }
        for action in colour_actions {
            self.emit_app(action);
        }
        match serde_json::to_value(&draft) {
            Ok(after) if after != before => {
                self.emit_app(AppAction::SettingsPatch { before, after })
            }
            Err(error) => self.status = error.to_string(),
            _ => {}
        }
    }

    /// Draw the viewport's extras (`overlay::extras`, cached per document state, playhead,
    /// selection and aspect) and the cubes of a Bounds display over the image.
    fn draw_overlays(&mut self, ui: &egui::Ui, rect: egui::Rect) {
        let frame = f64::from(self.world_ui.playhead);
        let aspect = rect.width() / rect.height().max(1.0);
        let selected = self.world.targets(None);
        let key = (
            playa_graph::NodeId(self.world.document.graph.id),
            self.world.revision(),
            frame.to_bits(),
            selected,
            aspect.to_bits(),
        );
        if self
            .overlays
            .as_ref()
            .is_none_or(|(cached, _)| *cached != key)
        {
            let document = &self.world.document;
            let wires = document.viewport_policy(frame).and_then(|policy| {
                crate::overlay::extras(document, frame, &policy, &key.3, aspect)
            });
            match wires {
                Ok(wires) => self.overlays = Some((key, wires)),
                Err(error) => {
                    self.overlays = None;
                    self.status = error;
                    return;
                }
            }
        }
        let (view, projection) = self.scene.camera_matrices(aspect);
        let clip = projection * view;
        let painter = ui.painter_at(rect);
        let visuals = ui.visuals();
        let plain = egui::Stroke::new(1.0, visuals.widgets.noninteractive.fg_stroke.color);
        let selected = egui::Stroke::new(1.5, visuals.selection.stroke.color);
        let bounds = self
            .viewport_scene
            .as_ref()
            .map(|cached| cached.3.bounds.as_slice())
            .unwrap_or_default();
        let objects = if self.controls.selection_bounds {
            self.viewport_scene
                .as_ref()
                .map(|cached| cached.3.objects.as_slice())
                .unwrap_or(self.scene.objects.as_slice())
        } else {
            &[]
        };
        let wires = self
            .overlays
            .iter()
            .flat_map(|(_, wires)| wires.iter().cloned());
        let selection = self
            .overlays
            .as_ref()
            .map(|(key, _)| key.3.clone())
            .unwrap_or_default();
        for wire in wires
            .chain(crate::overlay::bounds(bounds, &selection))
            .chain(crate::overlay::selected_object_bounds(
                objects,
                &self.world.document,
                &selection,
            ))
        {
            for [a, b] in crate::overlay::project(&wire, clip, rect) {
                match wire.style {
                    crate::overlay::Style::Plain => {
                        painter.line_segment([a, b], plain);
                    }
                    crate::overlay::Style::Selected => {
                        painter.line_segment([a, b], selected);
                    }
                    crate::overlay::Style::Cut => {
                        painter.extend(egui::Shape::dashed_line(&[a, b], plain, 6.0, 4.0));
                    }
                }
            }
        }
    }

    /// The transform manipulator of the current tool on the selected unlocked nodes that take it
    /// (`world::manip_tools`: fractals and groups, the sun for Rotate), drawn over the viewport
    /// image. It shows each object's world pose at its pivot; a drag writes back only the edited
    /// channel (position, rotation or scale) as the local pose under the parent, one undo step
    /// per drag, keyed like any edit at the playhead. The edits change the scene, so the viewport
    /// switches to its Moving profile while dragging, as during navigation.
    fn manipulate(&mut self, ui: &egui::Ui, rect: egui::Rect) {
        use soluwell_viewport::gizmo::{
            GizmoConfig, GizmoOrientation, GizmoSpace, GizmoTool, Transform,
        };
        /// One manipulated node, resolved whole before it joins the list, so the gizmo's
        /// targets and the edits its answer is applied to are one list and cannot drift.
        struct Handle {
            id: crate::world::NodeId,
            /// The handles' pose (`Transform`) the gizmo draws and drags.
            target: Transform,
            /// The parent's world matrix: the drag is written as the local pose under it.
            parent: glam::Mat4,
            /// The node's world scale (kept by Move and Rotate).
            scale: glam::Vec3,
            /// The local pose the handles show: the take the node plays, else the authored one.
            shown: crate::takes::Trs,
            /// A take drives the node (the drag releases it).
            bound: bool,
        }
        let tool = match self.tool {
            ViewTool::Select => None,
            ViewTool::Move => Some(GizmoTool::Move),
            ViewTool::Rotate => Some(GizmoTool::Rotate),
            ViewTool::Scale => Some(GizmoTool::Scale),
        };
        let frame = f64::from(self.world_ui.playhead);
        let document = &self.world.document;
        let policy = match document.viewport_policy(frame) {
            Ok(policy) => policy,
            Err(error) => {
                self.status = error;
                return;
            }
        };
        let mut handles = Vec::new();
        let mut failed = None;
        for id in self.world.targets(None) {
            let Ok(info) = document.info(id) else {
                continue;
            };
            let [moves, turns, scales] = crate::world::manip_tools(info.kind);
            let takes_tool = match self.tool {
                ViewTool::Select => false,
                ViewTool::Move => moves,
                ViewTool::Rotate => turns,
                ViewTool::Scale => scales,
            };
            if info.locked || !takes_tool {
                continue;
            }
            let handle = viewport_selects(document, &policy, frame, id).and_then(|selects| {
                // The viewport manipulates only what it shows and lets pick (its kind filter).
                if !selects {
                    return Ok(None);
                }
                let (pose, parent) = document.manip_frame(id, frame)?;
                let (scale, rotation, translation) = pose.to_scale_rotation_translation();
                let take = document.take_trs(id, frame)?;
                let shown = match take {
                    Some(take) => take,
                    None => document.authored_trs(id, frame)?,
                };
                // Move and Rotate draw unit handles: a scaled object would stretch them.
                let handle_scale = if self.tool == ViewTool::Scale {
                    scale
                } else {
                    glam::Vec3::ONE
                };
                Ok(Some(Handle {
                    id,
                    target: Transform::from_scale_rotation_translation(
                        handle_scale,
                        rotation,
                        translation,
                    ),
                    parent,
                    scale,
                    shown,
                    bound: take.is_some(),
                }))
            });
            match handle {
                Ok(Some(handle)) => handles.push(handle),
                Ok(None) => {}
                Err(error) => failed = Some(error),
            }
        }
        if let Some(error) = failed {
            self.status = error;
        }
        let Some(tool) = tool.filter(|_| !handles.is_empty() && rect.height() > 0.0) else {
            self.gizmo.clear_hover();
            self.end_drag();
            return;
        };
        let (view, projection) = self.scene.camera_matrices(rect.width() / rect.height());
        self.gizmo.update_config(GizmoConfig {
            view,
            projection,
            viewport: rect,
            space: GizmoSpace::ThreeD,
            tool,
            orientation: GizmoOrientation::Local,
            // Shift: 5 degree / 0.1 increments for Rotate and Scale.
            snapping: tool != GizmoTool::Move && ui.input(|i| i.modifiers.shift),
            snap_angle: 5.0_f32.to_radians(),
            snap_scale: 0.1,
            visuals: soluwell_viewport::gizmo::GizmoVisuals {
                gizmo_size: self.controls.gizmo_size,
                stroke_width: self.controls.gizmo_stroke,
                ..Default::default()
            },
            ..Default::default()
        });
        let targets: Vec<Transform> = handles.iter().map(|handle| handle.target).collect();
        let Some((_, moved)) = self.gizmo.interact(ui, &targets) else {
            self.end_drag();
            return;
        };
        let gesture = egui::Id::new("viewport_gizmo").value();
        let mut commands = Vec::new();
        let mut detached = Vec::new();
        for (handle, t) in handles.into_iter().zip(moved) {
            let Handle {
                id,
                parent,
                scale,
                shown,
                bound,
                ..
            } = handle;
            let scale = if self.tool == ViewTool::Scale {
                t.scale
            } else {
                scale
            };
            let pose =
                glam::Mat4::from_scale_rotation_translation(scale, t.rotation, t.translation);
            let (position, rotation, scale) = crate::world::local_pose(parent, pose);
            // The equivalent angles nearest the shown ones (both Euler solutions, whole
            // turns): no flip at +-180, and the turns of the value and its keys are kept.
            let rotation = curves::nearest_euler(rotation, shown.rotation_degrees, Some(1));
            let edited_channel = match self.tool {
                ViewTool::Move => ("/transform/position", position),
                ViewTool::Rotate => ("/transform/rotation_degrees", rotation),
                ViewTool::Scale => ("/transform/scale", scale),
                ViewTool::Select => continue,
            };
            // A node a take drives is released by the drag, which continues from the take's
            // pose: the whole pose is authored, not just the edited channel, or the other two
            // would jump back to the node's own values.
            let all = [
                ("/transform/position", position),
                ("/transform/rotation_degrees", rotation),
                ("/transform/scale", scale),
            ];
            let writes: &[(&str, [f32; 3])] = if bound {
                &all
            } else {
                std::slice::from_ref(&edited_channel)
            };
            for &(path, value) in writes {
                commands.push(crate::world::WorldCommand::SetAttribute {
                    id,
                    path: path.into(),
                    value: serde_json::json!(value),
                    frame,
                });
            }
            if bound {
                detached.push(id);
            }
        }
        self.emit_app(AppAction::TransformDrag {
            commands,
            detached,
            gesture,
            frame,
            dt: f64::from(nav_dt(ui.ctx())),
        });
        ui.ctx().request_repaint();
    }

    /// A manipulator drag ended: its transform takes go to the document.
    fn end_drag(&mut self) {
        if self.gizmo_gesture.is_some() {
            self.emit_app(AppAction::EndDrag);
        }
    }

    /// Feed this frame's navigation to the take recorder. Any camera change while a take drives
    /// the camera (a navigation gesture or an Attribute Editor edit) releases the take first:
    /// the change continues from the take's pose and is authored on the camera node like any
    /// other, instead of being written to the node but hidden behind the take.
    fn record_take(
        &mut self,
        ctx: &egui::Context,
        before: crate::scene::Camera,
        gesture: Option<u64>,
    ) {
        let after = if self.navigated {
            self.scene.camera
        } else {
            before
        };
        // Recording off (the Takes panel's switch): gestures make no takes.
        self.takes_recorder.min_secs = self.min_take_secs;
        let update = if !self.record_takes {
            crate::takes::Update::default()
        } else {
            self.takes_recorder.tick(&crate::takes::Tick {
                dt: f64::from(nav_dt(ctx)),
                before,
                after,
                frame: f64::from(self.world_ui.playhead),
                held: self.fly.is_some()
                    || self.orbit.rig.has_inertia()
                    || ctx.input(|i| i.pointer.any_down()),
            })
        };
        let camera = self.world.document.active_camera;
        if self.scene.camera != before
            && let Some(camera) =
                camera.filter(|c| matches!(self.world.document.take_clip(*c), Ok(Some(_))))
        {
            self.emit_app(AppAction::DetachCameraTake { camera, gesture });
            self.evaluated_world = None;
        }
        match update.finished {
            Some(Ok(take)) => {
                self.emit_app(AppAction::RecordedTake(crate::takes::Take {
                    source: camera,
                    ..take
                }));
            }
            Some(Err(error)) => self.status = format!("Recorded take dropped: {error}"),
            None => {}
        }
        // The settle time runs on frames: keep them coming until the gesture is cut.
        if self.takes_recorder.recording() {
            ctx.request_repaint();
        }
    }

    /// The OCIO config file dialog (Settings > Color > File…): a picked file becomes the
    /// working config. Runs where the panels do, before the end-of-frame diff authors it.
    fn update_config_picker(&mut self, ctx: &egui::Context) {
        self.config_picker.update(ctx);
        let before = self.world_ui.file_dialogs.clone();
        let mut history = before.clone();
        history.observe(crate::file_dialogs::OCIO, &self.config_picker);
        if history != before {
            self.emit_app(AppAction::SettingsPatch {
                before: serde_json::json!({"file_dialogs": before}),
                after: serde_json::json!({"file_dialogs": history}),
            });
        }
        if let Some(path) = self.config_picker.take_picked() {
            self.emit_app(AppAction::ColourConfig(path.to_string_lossy().into_owned()));
        }
    }

    /// Hand changed preferences to the file worker (coalesced there).
    fn save_settings(&mut self) {
        match self.changed_settings_json() {
            Ok(Some(json)) => {
                self.io.settings(settings_path(), json);
            }
            Ok(None) => {}
            Err(error) => self.status = error,
        }
    }

    /// The preferences as JSON when they differ from the last ones handed out, else None.
    fn changed_settings_json(&mut self) -> Result<Option<String>, String> {
        self.layouts
            .cache
            .refresh(&self.dock)
            .map_err(|error| format!("Layout save failed: {error}"))?;
        let settings = self.current_settings();
        if self.persisted_settings.as_ref() == Some(&settings) {
            return Ok(None);
        }
        let json = serde_json::to_string_pretty(&settings).map_err(|error| error.to_string())?;
        #[cfg(test)]
        {
            self.settings_serializations += 1;
        }
        self.persisted_settings = Some(settings);
        Ok(Some(json))
    }

    /// `FRAC_SNAP_PRESET=i` opens gallery preset i, `FRAC_SNAP_MATERIAL=name` applies that
    /// library preset to the selection's material, `FRAC_SNAP_TAB=materials` shows the
    /// Materials panel: the documentation screenshots' scene (`App::handle_snap`).
    fn with_snap_preset(mut self) -> Self {
        match env_number::<usize>("FRAC_SNAP_PRESET") {
            Ok(None) => {}
            Ok(Some(i)) => match self.gallery.get(i) {
                Some(e) => {
                    let scene = e.scene.clone();
                    self.load(scene, Load::New);
                    self.card = Some(Card::Gallery(i));
                }
                None => {
                    self.status = format!(
                        "FRAC_SNAP_PRESET={i}: the gallery has {} presets",
                        self.gallery.len()
                    );
                }
            },
            Err(error) => self.status = error,
        }
        if let Ok(name) = std::env::var("FRAC_SNAP_MATERIAL") {
            match crate::materials::PRESETS.iter().find(|p| p.name() == name) {
                Some(p) => {
                    let before = self.scene.clone();
                    p.apply(&mut self.scene.material);
                    if let Err(error) = self.world.edit_snapshot(
                        self.world.active(),
                        &before,
                        &self.scene,
                        f64::from(self.world_ui.playhead),
                    ) {
                        self.status = format!("FRAC_SNAP_MATERIAL {name}: {error}");
                    }
                }
                None => self.status = format!("FRAC_SNAP_MATERIAL {name}: no such material preset"),
            }
        }
        if std::env::var("FRAC_SNAP_TAB").is_ok_and(|t| t == "materials") {
            self.panels_to_open.push(dock::Panel::Materials);
        }
        self
    }

    /// Arm the `FRAC_SNAP` window shot once the viewport shows its samples (the snap's count,
    /// at most the profile's target, which is where adaptive sampling stops) and, unless told
    /// not to, every card thumbnail is done (a failed one counts as done).
    fn handle_snap(&mut self, thumbs_pending: bool) {
        let Some(snap) = &self.snap else {
            return;
        };
        let spp = snap.spp.min(self.target_spp);
        let ready = !snap.requested
            && (!snap.wait_thumbs || !thumbs_pending)
            && !self.showing_preview
            && self.frame.as_ref().is_some_and(|t| t.complete(spp));
        if !ready {
            return;
        }
        // The window's own signal through the one window-screenshot path; the window closes
        // once the file is written (`window_event`). Behind an armed user shot it waits.
        let shot = crate::window_shot::WindowShot {
            kind: crate::window_shot::ShotKind::Displayed {
                path: snap.path.clone(),
            },
            quit: true,
        };
        if self.arm_window_shot(shot)
            && let Some(snap) = &mut self.snap
        {
            snap.requested = true;
        }
    }

    fn refresh_templates(&mut self) {
        match self.io.send(crate::io_service::Command::RefreshTemplates {
            directory: crate::templates::directory(),
        }) {
            Ok(()) => {
                self.templates_requested = true;
                self.templates_pending = true;
            }
            Err(error) => self.status = error,
        }
    }
    /// The file dialog of a scene or template `action`; the picked path goes to
    /// `submit_scene_action` (`update_scene_picker`).
    fn scene_action_dialog(&mut self, action: crate::templates::FileAction) {
        let save = action.save();
        let template_directory = crate::templates::directory();
        let mut dialog = egui_file_dialog::FileDialog::new()
            .add_file_filter_extensions("Fractal scene", vec!["json"])
            .default_file_name(&crate::fs_name::frame_file(
                &crate::fs_name::stem(&self.scene.name),
                None,
                crate::fs_name::SCENE_SUFFIX,
            ));
        dialog = self.world_ui.file_dialogs.prepare(
            dialog,
            action.history_key(),
            if action.template() {
                Some(template_directory.as_path())
            } else {
                self.scene_file_path.as_ref().and_then(|path| path.parent())
            },
            "Fractal scene",
        );
        if save {
            dialog.save_file();
        } else {
            dialog.pick_file();
        }
        self.scene_picker = Some((action, dialog));
    }

    /// Send a scene or template `action` on `path` to the file worker; its answer comes back by
    /// id (`scene_file_event`).
    fn submit_scene_action(&mut self, mut path: PathBuf, action: crate::templates::FileAction) {
        let save = action.save();
        self.world.finish_edit();
        if save && path.extension().is_none() {
            path.set_extension(crate::fs_name::SCENE_SUFFIX);
        }
        // Saved under another name the scene becomes a copy: the file it was opened from keeps
        // the old identity, this document (and its new file) gets its own.
        if action == crate::templates::FileAction::Save
            && self.scene_file_path.as_ref() != Some(&path)
        {
            self.world.renew_identity();
        }
        if save {
            // The scene file carries the CamClip as it is now.
            self.world.document.camera_slots = self.camera_slots;
        }
        let id = self.next_file_id();
        let command = if save {
            crate::io_service::Command::SaveScene {
                id,
                path: path.clone(),
                document: Box::new(self.world.document.clone()),
            }
        } else {
            crate::io_service::Command::OpenScene {
                id,
                path: path.clone(),
            }
        };
        match self.io.send(command) {
            Ok(()) => {
                self.scene_file_pending = Some((id, self.load_revision, action));
                self.status = format!(
                    "{} {}",
                    if save { "Saving" } else { "Opening" },
                    path.display()
                );
            }
            Err(error) => self.status = error,
        }
    }

    fn update_scene_picker(&mut self, ctx: &egui::Context) {
        let before = self.world_ui.file_dialogs.clone();
        let mut history = before.clone();
        let picked = if let Some((save, picker)) = &mut self.scene_picker {
            picker.update(ctx);
            history.observe(save.history_key(), picker);
            picker.take_picked().map(|path| (*save, path))
        } else {
            None
        };
        if history != before {
            self.emit_app(AppAction::SettingsPatch {
                before: serde_json::json!({"file_dialogs": before}),
                after: serde_json::json!({"file_dialogs": history}),
            });
        }
        if let Some((save, path)) = picked {
            self.scene_picker = None;
            self.emit_app(AppAction::SceneFile { path, action: save });
        }
    }

    fn scene_file_event(&mut self, event: crate::io_service::SceneEvent) {
        let Some((id, revision, action)) = self.scene_file_pending else {
            return;
        };
        if id != event.id || revision != self.load_revision {
            return;
        }
        self.scene_file_pending = None;
        match event.result {
            Ok(Some(scene)) => {
                // A template opens as a new scene; a scene file is that document again.
                self.load(
                    *scene,
                    if action.template() {
                        Load::New
                    } else {
                        Load::Same
                    },
                );
                if !action.template() {
                    self.scene_file_path = Some(event.path.clone());
                }
                self.status = format!("Opened {}", event.path.display());
            }
            Ok(None) => {
                if action.template() {
                    self.refresh_templates();
                } else {
                    self.scene_file_path = Some(event.path.clone());
                }
                self.status = format!("Saved {}", event.path.display());
            }
            Err(error) => self.status = format!("Scene file {}: {error}", event.path.display()),
        }
    }

    /// A finished bookmark save or delete (`BookmarkOp`): the list follows the file only now.
    fn bookmark_event(&mut self, op: BookmarkOp, event: crate::io_service::SceneEvent) {
        if let Err(error) = event.result {
            self.status = format!("Bookmark {}: {error}", event.path.display());
            return;
        }
        match op {
            BookmarkOp::Save(mut entry) => {
                entry.path = Some(event.path.clone());
                self.bookmarks.push(*entry);
                self.panels_to_open.push(dock::Panel::Bookmarks);
                self.status = format!("Saved bookmark {}", event.path.display());
            }
            BookmarkOp::Delete => {
                self.bookmarks
                    .retain(|entry| entry.path.as_ref() != Some(&event.path));
                self.status = format!("Deleted bookmark {}", event.path.display());
            }
        }
        // Bookmark thumbnail ids are list indices: a changed list makes their failures stale.
        self.thumb_failed.retain(|id| {
            !matches!(
                crate::material_gallery::ThumbId::decode(*id),
                crate::material_gallery::ThumbId::Bookmark(_)
            )
        });
    }

    /// The id of the next file-worker request answered by id (`SceneEvent::id`).
    fn next_file_id(&mut self) -> u64 {
        self.scene_file_sequence = self.scene_file_sequence.wrapping_add(1);
        self.scene_file_sequence
    }

    /// File > Save: to the scene's file, or through the dialog for a scene that has none.
    fn save_scene(&mut self) {
        if let Some(path) = self.scene_file_path.clone() {
            self.submit_scene_action(path, crate::templates::FileAction::Save);
        } else {
            self.scene_action_dialog(crate::templates::FileAction::Save);
        }
    }

    /// File > New: everything cleared, the scene the app starts with loaded as a new document.
    fn new_scene(&mut self) {
        let scene = self.start_scene();
        self.load(scene, Load::New);
        self.status = "New scene".into();
    }

    /// Edit > Reset scene: the document back to how it was loaded (`origin`), as one undoable
    /// step; the scene's file and the history stay.
    fn reset_scene(&mut self) {
        self.world.finish_edit();
        let document = crate::world::WorldDocument::from_scene(&self.origin);
        self.status = match self.world.reset(document) {
            Ok(()) => "Scene reset".into(),
            Err(error) => error,
        };
    }

    /// Apply a document history step (`History`) at the frame's history point (`ui`).
    fn step_history(&mut self, step: History) {
        match step {
            History::Undo => {
                self.world.undo();
            }
            History::Redo => {
                self.world.redo();
            }
            History::Reset => self.reset_scene(),
        }
    }

    /// Load `scene` as the editor's document: the same document again (`Load::Same`: a file
    /// opened) or a new one made from it (`Load::New`: a template, preset or bookmark, which
    /// gets its own identity: `WorldDocument::renew_identity`). A new history starts; the
    /// colour state follows the document at the next `refresh_scene`.
    fn load(&mut self, mut scene: Scene, mode: Load) {
        self.takes_recorder.cancel();
        self.preview.cancel();
        self.preview_key = None;
        self.card = None;
        self.scene_file_path = None;
        self.scene_file_pending = None;
        self.pick_pending = None;
        self.evaluated_world = None;
        self.viewport_scene = None;
        self.viewport_render = None;
        self.load_revision = self.load_revision.wrapping_add(1);
        let mut document = crate::world::WorldDocument::from_scene(&scene);
        // A scene that carries a CamClip replaces the application's; one without keeps it.
        if !document.camera_slots.is_empty() {
            self.camera_slots = document.camera_slots;
        }
        if mode == Load::New {
            document.renew_identity();
            // "Reset" goes back to this document, not to the template it was made from.
            scene.document = Some(Box::new(document.clone()));
        }
        self.world_ui.reset();
        self.world_ui.seek(document.first);
        self.world = crate::world::WorldEditor::new(document.clone());
        self.material_gallery.invalidate();
        self.origin = scene.clone();
        scene.document = Some(Box::new(document));
        self.scene = match self
            .world
            .document
            .snapshot(f64::from(self.world_ui.playhead))
        {
            Ok(evaluated) => evaluated,
            Err(error) => {
                // Shown as loaded until an edit makes it evaluate (`refresh_scene` retries).
                self.status = format!("Scene not evaluated: {error}");
                scene
            }
        };
    }

    /// Send the next card thumbnail to the render worker, one at a time: material cards first,
    /// then gallery, bookmark and swatch cards without a thumbnail that did not fail
    /// (`thumb_failed`). Returns whether thumbnail work is still waiting (`FRAC_SNAP`; a failed
    /// card counts as done).
    fn render_one_thumbnail(&mut self) -> bool {
        if let Err(error) = self
            .material_gallery
            .refresh(&self.world, f64::from(self.world_ui.playhead))
        {
            self.status = error;
        }
        if self.thumb_pending.is_some() {
            return true;
        }
        use crate::material_gallery::ThumbId;
        use crate::render_service::CommandError;
        // Every card renders at the same quality: this seed, and at most this many bounces on
        // the evaluated scene (the worker caps after evaluation: a bookmark's frozen world
        // carries its own render settings).
        const SEED: u32 = 7;
        const MAX_BOUNCES: u32 = 3;
        let failed = &self.thumb_failed;
        // The first card of `entries` still to render, as (encoded id, scene).
        let next_card = |entries: &[Entry], card: fn(usize) -> ThumbId| {
            entries.iter().enumerate().find_map(|(i, e)| {
                let id = card(i).encode();
                (e.thumb.is_none() && !failed.contains(&id)).then(|| (id, e.scene.clone()))
            })
        };
        let item = self
            .material_gallery
            .next_thumbnail()
            .map(|request| {
                (
                    request.id,
                    material_preview_scene(request.material),
                    SWATCH,
                    SWATCH,
                    SWATCH_SPP,
                )
            })
            .or_else(|| {
                next_card(&self.gallery, ThumbId::Gallery)
                    .or_else(|| next_card(&self.bookmarks, ThumbId::Bookmark))
                    .map(|(id, scene)| (id, scene, THUMB_W, THUMB_H, THUMB_SPP))
            })
            .or_else(|| {
                crate::materials::PRESETS
                    .iter()
                    .zip(&self.swatches)
                    .enumerate()
                    .find(|(i, (_, slot))| {
                        slot.is_none() && !failed.contains(&ThumbId::Swatch(*i).encode())
                    })
                    .map(|(i, (preset, _))| {
                        (
                            ThumbId::Swatch(i).encode(),
                            swatch_scene(preset),
                            SWATCH,
                            SWATCH,
                            SWATCH_SPP,
                        )
                    })
            });
        let Some((id, scene, width, height, spp)) = item else {
            return false;
        };
        let material = matches!(ThumbId::decode(id), ThumbId::Material(_));
        match self.renderer.try_command(Command::Thumbnail {
            id,
            scene: Box::new(scene.clone()),
            width,
            height,
            spp,
            seed: SEED,
            budget_ms: crate::render_service::BACKGROUND_BUDGET_MS,
            max_bounces: MAX_BOUNCES,
        }) {
            Ok(()) => self.thumb_pending = Some((id, scene)),
            // A full queue is retried on the next frame, which no worker result may bring.
            Err(CommandError::Full) if material => {
                self.material_gallery.retry_thumbnail(id);
                self.wake.wake();
            }
            Err(CommandError::Full) => self.wake.wake(),
            Err(error) => self.fail_thumbnail(id, error.to_string()),
        }
        true
    }

    /// A card's thumbnail failed: the card keeps its placeholder and is not asked again
    /// (material cards keep the error on their card), and the status line says why.
    fn fail_thumbnail(&mut self, id: u64, error: String) {
        if matches!(
            crate::material_gallery::ThumbId::decode(id),
            crate::material_gallery::ThumbId::Material(_)
        ) {
            self.material_gallery.fail_thumbnail(id, error.clone());
        } else {
            self.thumb_failed.insert(id);
        }
        self.status = format!("Thumbnail: {error}");
    }

    /// Selection and assignment are separate: cards refer to authored node UUIDs.
    fn select_material_node(&mut self, id: crate::world::NodeId) {
        self.remember_material_targets();
        self.world.finish_edit();
        self.world.click_select(Some(id), false);
        self.panels_to_open.push(dock::Panel::Inspector);
    }

    /// Selecting a material changes the editor's node selection, but keeps the
    /// last consumer selection as an explicit assignment context. Other node
    /// selections clear it; document changes cannot leak targets across projects.
    fn remember_material_targets(&mut self) {
        use std::hash::{Hash, Hasher};
        let mut hash = std::collections::hash_map::DefaultHasher::new();
        self.world.selected().hash(&mut hash);
        let stamp = (
            playa_graph::NodeId(self.world.document.graph.id),
            self.world.revision(),
            self.world.active(),
            hash.finish(),
        );
        if self.material_selection_stamp == Some(stamp) {
            return;
        }
        if self
            .material_selection_stamp
            .is_none_or(|old| old.0 != stamp.0)
        {
            self.material_targets.clear();
        }
        self.material_selection_stamp = Some(stamp);
        let selected = self.world.targets(None);
        let eligible = selected
            .iter()
            .copied()
            .filter(|id| self.world.document.supports_material(*id));
        if eligible.clone().next().is_some() {
            self.material_targets.clear();
            self.material_targets.extend(eligible);
        } else if selected.is_empty()
            || selected.iter().any(|id| {
                self.world
                    .document
                    .info(*id)
                    .is_ok_and(|n| n.kind != crate::world::WorldKind::Material)
            })
        {
            self.material_targets.clear();
        } else {
            self.material_targets
                .retain(|id| self.world.document.supports_material(*id));
        }
    }

    /// The objects an "Assign to …" action addresses.
    fn assign_targets(&mut self, to: AssignTo) -> Vec<crate::world::NodeId> {
        match to {
            AssignTo::Selected => {
                self.remember_material_targets();
                self.material_targets.clone()
            }
            AssignTo::AllFractals => self.world.document.material_consumers(),
        }
    }

    /// Assign a work-area Material node: one Batch, one undo step.
    #[cfg(test)]
    fn assign_gallery_material(
        &mut self,
        material: crate::world::NodeId,
        to: AssignTo,
    ) -> Result<(), String> {
        let commands = self
            .assign_targets(to)
            .into_iter()
            .map(|id| crate::world::WorldCommand::AssignMaterial {
                id,
                material: Some(material),
            })
            .collect::<Vec<_>>();
        if commands.is_empty() {
            return Err("Select an object that supports materials".into());
        }
        self.world.finish_edit();
        self.world
            .execute(crate::world::WorldCommand::Batch(commands))
    }

    /// Instantiate a library preset as a Material node and assign it, as one undo step.
    fn assign_library_material(&mut self, index: usize, to: AssignTo) -> Result<(), String> {
        let preset = &crate::materials::PRESETS[index];
        let mut material = Material::default();
        preset.apply(&mut material);
        let targets = self.assign_targets(to);
        self.world.finish_edit();
        self.world
            .execute(crate::world::WorldCommand::CreateMaterialFor {
                material,
                name: preset.name().into(),
                targets,
            })?;
        self.material_gallery.invalidate();
        Ok(())
    }

    /// "Assign to selected" / "Assign to all fractals": the shared context-menu items.
    fn assign_menu(ui: &mut egui::Ui, selected: bool, fractals: bool) -> Option<AssignTo> {
        let mut picked = None;
        if ui
            .add_enabled(selected, egui::Button::new("Assign to selected"))
            .clicked()
        {
            picked = Some(AssignTo::Selected);
        }
        if ui
            .add_enabled(fractals, egui::Button::new("Assign to all fractals"))
            .clicked()
        {
            picked = Some(AssignTo::AllFractals);
        }
        if picked.is_some() {
            ui.close();
        }
        picked
    }

    fn materials_tab(&mut self, ui: &mut egui::Ui) {
        self.remember_material_targets();
        use crate::materials::{CATEGORIES, PRESETS};
        use crate::world::WorldCommand;
        if let Err(error) = self
            .material_gallery
            .refresh(&self.world, f64::from(self.world_ui.playhead))
        {
            self.status = error;
        }
        let mut create = None;
        let mut create_default = false;
        ui.horizontal_wrapped(|ui| {
            create_default = ui.button("+ New material").clicked();
            if ui.button("Library…").clicked() {
                self.emit_app(AppAction::OpenPanel(dock::Panel::MaterialLibrary));
            }
            ui.menu_button("Create from preset", |ui| {
                for category in CATEGORIES {
                    ui.menu_button(category, |ui| {
                        for (index, preset) in PRESETS
                            .iter()
                            .enumerate()
                            .filter(|(_, preset)| preset.category == category)
                        {
                            if ui.button(preset.name()).clicked() {
                                create = Some(index);
                                ui.close();
                            }
                        }
                    });
                }
            });
        });
        ui.label(
            RichText::new("Click to edit · right-click to assign to selected objects")
                .small()
                .weak(),
        );
        let mut select = None;
        let selected_targets = !self.material_targets.is_empty();
        let any_fractal = !self.world.document.material_consumers().is_empty();
        let mut assign = None;
        let mut refresh = None;
        let mut apply_preset = None;
        let mut duplicate = None;
        let mut command = None;
        egui::ScrollArea::vertical()
            .id_salt("material_nodes")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                let gap = 6.0;
                let (columns, size) =
                    thumbnail_grid(ui.available_width(), Vec2::splat(SWATCH as f32), gap);
                egui::Grid::new("material-node-grid")
                    .spacing([gap, gap])
                    .show(ui, |ui| {
                        for (index, entry) in self.material_gallery.entries.iter().enumerate() {
                            let response = ui
                                .push_id(entry.id, |ui| {
                                    ui.vertical(|ui| {
                                        ui.set_width(size.x);
                                        if let Some(texture) = &entry.thumb {
                                            ui.add(
                                                egui::Image::new((texture.id(), size))
                                                    .corner_radius(6.0),
                                            );
                                        } else {
                                            let (rect, _) =
                                                ui.allocate_exact_size(size, Sense::hover());
                                            ui.painter().rect_filled(
                                                rect,
                                                6.0,
                                                ui.visuals().extreme_bg_color,
                                            );
                                        }
                                        let mut label = RichText::new(&entry.name).small();
                                        if self.world.active() == Some(entry.id) {
                                            label = label
                                                .strong()
                                                .color(ui.visuals().selection.stroke.color);
                                        }
                                        ui.add(egui::Label::new(label).truncate());
                                    })
                                    .response
                                    .interact(Sense::click())
                                })
                                .inner;
                            if response.clicked() {
                                select = Some(entry.id);
                            }
                            if let Some(error) = &entry.error {
                                response.clone().on_hover_text(error);
                            }
                            response.context_menu(|ui| {
                                if let Some(to) =
                                    Self::assign_menu(ui, selected_targets, any_fractal)
                                {
                                    assign = Some((entry.id, to));
                                }
                                if ui.button("Select / Edit").clicked() {
                                    select = Some(entry.id);
                                    ui.close();
                                }
                                if ui.button("Refresh preview").clicked() {
                                    refresh = Some(entry.id);
                                    ui.close();
                                }
                                ui.menu_button("Apply preset to this material", |ui| {
                                    for category in CATEGORIES {
                                        ui.menu_button(category, |ui| {
                                            for preset in PRESETS
                                                .iter()
                                                .filter(|preset| preset.category == category)
                                            {
                                                if ui.button(preset.name()).clicked() {
                                                    let mut material = entry.material.clone();
                                                    preset.apply(&mut material);
                                                    apply_preset =
                                                        Some((entry.id, material, preset.name()));
                                                    ui.close();
                                                }
                                            }
                                        });
                                    }
                                });
                                if ui.button("Duplicate").clicked() {
                                    duplicate = Some(entry.id);
                                    ui.close();
                                }
                                if ui.button("Delete").clicked() {
                                    command = Some(WorldCommand::Delete(entry.id));
                                    ui.close();
                                }
                            });
                            if (index + 1) % columns == 0 {
                                ui.end_row();
                            }
                        }
                    });
            });
        if let Some(id) = refresh {
            self.emit_app(AppAction::RefreshMaterial(id));
        }
        if let Some((id, material, name)) = apply_preset {
            command = Some(WorldCommand::ApplyMaterial {
                id,
                material,
                name: name.into(),
                frame: f64::from(self.world_ui.playhead),
            });
        }
        if let Some(id) = select {
            self.emit_app(AppAction::SelectMaterial(id));
        }
        if let Some((id, to)) = assign {
            let targets = self.assign_targets(to);
            self.emit_app(AppAction::AssignMaterial {
                material: id,
                targets,
            });
        }
        if let Some(command) = command {
            self.emit_app(AppAction::Edit {
                command,
                gesture: None,
            });
        }
        // The one Duplicate path of the UI (`WorldEditor::duplicate`): the card's material, or
        // the whole selection when the card is part of it, as the outliner does.
        if let Some(id) = duplicate {
            self.emit_app(AppAction::Edit {
                command: WorldCommand::Duplicate(self.world.targets(Some(id))),
                gesture: None,
            });
        }
        if create_default || create.is_some() {
            let mut material = Material::default();
            let name = if let Some(index) = create {
                PRESETS[index].apply(&mut material);
                PRESETS[index].name()
            } else {
                "Material"
            };
            self.emit_app(AppAction::CreateMaterial {
                material,
                name: name.into(),
                targets: None,
                open: true,
            });
        }
    }

    /// Presets are reusable sources. Only this explicit action creates a
    /// workspace Material node; creation never changes object assignments.
    #[cfg(test)]
    fn add_library_material(&mut self, index: usize) {
        let preset = &crate::materials::PRESETS[index];
        let mut material = Material::default();
        preset.apply(&mut material);
        self.world.finish_edit();
        match self
            .world
            .execute(crate::world::WorldCommand::CreateMaterial {
                material,
                name: preset.name().into(),
            }) {
            Ok(()) => {
                if let Some(id) = self.world.active() {
                    self.select_material_node(id);
                }
                self.panels_to_open.push(dock::Panel::Materials);
            }
            Err(error) => self.status = error,
        }
    }

    fn material_library(&mut self, ui: &mut egui::Ui) {
        use crate::materials::{CATEGORIES, PRESETS};
        ui.label("Click a preset to add a Material node to the work area. Assign it separately.");
        self.remember_material_targets();
        let selected_targets = !self.material_targets.is_empty();
        let any_fractal = !self.world.document.material_consumers().is_empty();
        let mut add = None;
        let mut assign = None;
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                for category in CATEGORIES {
                    egui::CollapsingHeader::new(category)
                        .default_open(true)
                        .show(ui, |ui| {
                            let (columns, size) = thumbnail_grid(
                                ui.available_width(),
                                Vec2::splat(SWATCH as f32),
                                6.0,
                            );
                            egui::Grid::new(("library", category))
                                .spacing([6.0, 6.0])
                                .show(ui, |ui| {
                                    let mut column = 0;
                                    for (index, preset) in PRESETS
                                        .iter()
                                        .enumerate()
                                        .filter(|(_, p)| p.category == category)
                                    {
                                        let response = ui
                                            .push_id(index, |ui| {
                                                ui.vertical(|ui| {
                                                    ui.set_width(size.x);
                                                    if let Some(texture) = &self.swatches[index] {
                                                        ui.add(
                                                            egui::Image::new((texture.id(), size))
                                                                .corner_radius(6.0),
                                                        );
                                                    } else {
                                                        let (rect, _) = ui.allocate_exact_size(
                                                            size,
                                                            Sense::hover(),
                                                        );
                                                        ui.painter().rect_filled(
                                                            rect,
                                                            6.0,
                                                            ui.visuals().extreme_bg_color,
                                                        );
                                                    }
                                                    ui.add(
                                                        egui::Label::new(
                                                            RichText::new(preset.name()).small(),
                                                        )
                                                        .truncate(),
                                                    );
                                                })
                                                .response
                                                .interact(Sense::click())
                                            })
                                            .inner;
                                        if response.clicked() {
                                            add = Some(index);
                                        }
                                        response.context_menu(|ui| {
                                            if ui.button("Add to work area").clicked() {
                                                add = Some(index);
                                                ui.close();
                                            }
                                            ui.separator();
                                            if let Some(to) =
                                                Self::assign_menu(ui, selected_targets, any_fractal)
                                            {
                                                assign = Some((index, to));
                                            }
                                        });
                                        column += 1;
                                        if column == columns {
                                            ui.end_row();
                                            column = 0;
                                        }
                                    }
                                });
                        });
                }
            });
        if let Some(index) = add {
            let preset = &crate::materials::PRESETS[index];
            let mut material = Material::default();
            preset.apply(&mut material);
            self.emit_app(AppAction::CreateMaterial {
                material,
                name: preset.name().into(),
                targets: None,
                open: true,
            });
        }
        if let Some((index, to)) = assign {
            let preset = &crate::materials::PRESETS[index];
            let mut material = Material::default();
            preset.apply(&mut material);
            let targets = self.assign_targets(to);
            self.emit_app(AppAction::CreateMaterial {
                material,
                name: preset.name().into(),
                targets: Some(targets),
                open: false,
            });
        }
    }

    /// File > Save bookmark: the document into `~/.warpbro/bookmarks/<time>-<name>.json`. The
    /// card joins the Bookmarks panel once the file is written (`bookmark_event`).
    fn save_bookmark(&mut self) {
        let stamp = match SystemTime::now().duration_since(UNIX_EPOCH) {
            Ok(time) => time.as_secs(),
            Err(error) => {
                self.status = format!("Bookmark not saved: the clock is before 1970 ({error})");
                return;
            }
        };
        let path = bookmarks_dir().join(format!(
            "{stamp}-{}.json",
            crate::fs_name::stem(&self.scene.name)
        ));
        let id = self.next_file_id();
        match self.io.send(crate::io_service::Command::SaveScene {
            id,
            path,
            document: Box::new(self.world.document.clone()),
        }) {
            Ok(()) => {
                let entry = Entry {
                    scene: self.frozen_world_scene(),
                    thumb: None,
                    path: None,
                };
                self.bookmark_ops
                    .push((id, BookmarkOp::Save(Box::new(entry))));
                self.status = "Saving bookmark…".into();
            }
            Err(error) => self.status = error,
        }
    }

    /// The snapshot choices, shared by the File menu and the viewport toolbar.
    pub(super) fn snapshot_menu(&mut self, ui: &mut egui::Ui) {
        use crate::render_service::{FrameFile, PngEncoding};
        let monitor = Monitor::read(ui.ctx());
        let displayed = if monitor.hdr {
            "HDR10 PQ PNG: the monitor shows HDR"
        } else {
            "8-bit sRGB PNG: the monitor shows SDR"
        };
        let items = [
            ("Save image (as displayed)", displayed, None),
            (
                "Save SDR PNG",
                "8-bit sRGB / BT.709: the SDR rendering (an HDR view's SDR preview)",
                Some(FrameFile::Png(PngEncoding::Sdr8)),
            ),
            (
                "Save HDR10 PQ PNG",
                "16-bit PQ / BT.2020 with cICP: what an HDR monitor shows for this view (an SDR view's white at the monitor's, else BT.2408's 203 nits); needs an HDR-aware viewer",
                Some(FrameFile::Png(PngEncoding::Hdr10)),
            ),
            (
                "Save display EXR",
                "Linear display light, float, display primaries",
                Some(FrameFile::DisplayExr),
            ),
        ];
        for (label, hint, file) in items {
            if ui.button(label).on_hover_text(hint).clicked() {
                self.emit_app(AppAction::SaveFrame { monitor, file });
                ui.close();
            }
        }
    }

    /// Save the viewport's current frame as `monitor` shows it. `None` saves it as displayed:
    /// an HDR10 PNG when the monitor shows HDR (relative light at the monitor's SDR white, so
    /// the file is as bright as the screen), an SDR PNG otherwise. One path for the File menu
    /// and the toolbar.
    pub(super) fn save_frame(
        &mut self,
        monitor: Monitor,
        file: Option<crate::render_service::FrameFile>,
    ) {
        use crate::render_service::{FrameFile, PngEncoding};
        let Some(frame) = self.frame.clone() else {
            self.status = "Nothing rendered yet".into();
            return;
        };
        let file = file.unwrap_or(FrameFile::Png(PngEncoding::displayed(monitor.hdr)));
        let dir = match crate::new_out_dir(&crate::out_root()) {
            Ok(dir) => dir,
            Err(error) => {
                self.status = error;
                return;
            }
        };
        let path = dir.join(crate::fs_name::frame_file(
            &crate::fs_name::stem(&self.scene.name),
            None,
            file.suffix(),
        ));
        self.status = match self.io.send(crate::io_service::Command::SaveFrame {
            frame,
            path,
            file,
            sdr_white_nits: monitor.sdr_white_nits(),
        }) {
            Ok(()) => "Saving image…".into(),
            Err(e) => e,
        };
    }
    /// The window-screenshot choices of the File menu (`F12`: EXR + PQ PNG).
    fn window_shot_menu(&mut self, ui: &mut egui::Ui) {
        let hint = "The whole window as presented (viewport, panels, menus) at its physical \
            resolution. EXR: linear BT.709 light, 1.0 = SDR white, whiteLuminance in nits. PQ PNG: \
            16-bit ST 2084 / BT.2020 with cICP; SDR white at the monitor's (HDR) or 203 nits.";
        for files in crate::window_shot::WindowFiles::ALL {
            use crate::hotkeys::{Command as Hotkey, Scope, menu_item};
            let response = if files == crate::window_shot::WindowFiles::ExrAndPq {
                menu_item(ui, files.label(), Scope::Global, Hotkey::WindowShot)
            } else {
                ui.button(files.label())
            };
            if response.on_hover_text(hint).clicked() {
                self.emit_app(AppAction::WindowShot {
                    files,
                    monitor: Monitor::read(ui.ctx()),
                });
                ui.close();
            }
        }
    }

    /// Arm a whole-window screenshot of `files` into a new folder under `~/.warpbro/out`, with
    /// SDR white at `monitor`'s reference white (`Monitor::sdr_white_nits`). The window captures
    /// the frame after this one (`take_window_shot`).
    pub(crate) fn request_window_shot(
        &mut self,
        files: crate::window_shot::WindowFiles,
        monitor: Monitor,
    ) {
        self.arm_window_shot(crate::window_shot::WindowShot {
            kind: crate::window_shot::ShotKind::Linear {
                root: crate::out_root(),
                stem: crate::fs_name::stem(&self.scene.name),
                files,
                sdr_white_nits: monitor.sdr_white_nits(),
            },
            quit: false,
        });
    }

    /// Arm `shot` for the next frame. One shot is armed at a time: while one waits for its
    /// frame, a new request is refused in the status line rather than replacing it (a
    /// `FRAC_SNAP` quit shot in particular). Returns whether `shot` was armed.
    fn arm_window_shot(&mut self, shot: crate::window_shot::WindowShot) -> bool {
        if self.window_shot.is_some() {
            log::warn!("Window screenshot refused: another one is waiting for its frame");
            self.status = "A window screenshot is already pending".into();
            return false;
        }
        self.window_shot = Some((shot, false));
        true
    }

    /// The error the window must exit with, once (a failed `FRAC_SNAP`).
    pub(crate) fn take_fatal(&mut self) -> Option<String> {
        self.fatal.take()
    }

    /// The armed window screenshot, once a frame has been drawn since it was requested. The
    /// window calls this after presenting a frame and captures that frame.
    pub(crate) fn take_window_shot(&mut self) -> Option<crate::window_shot::WindowShot> {
        match self.window_shot.take() {
            Some((shot, true)) => Some(shot),
            pending => {
                self.window_shot = pending;
                None
            }
        }
    }

    /// Hand a submitted window capture to the file worker, which waits for the readback and
    /// writes the files. A failed capture or a refused request is reported at once, through
    /// the same `window_event` as a finished one.
    pub(crate) fn submit_window_shot(
        &mut self,
        ctx: &egui::Context,
        shot: crate::window_shot::WindowShot,
        capture: Result<
            soluwell_viewport::display::PendingCapture,
            soluwell_viewport::display::CaptureError,
        >,
        device: &wgpu::Device,
    ) {
        use crate::window_shot::ShotError;
        let quit = shot.quit;
        let sent = capture.map_err(ShotError::from).and_then(|capture| {
            self.io
                .send(crate::io_service::Command::SaveWindow {
                    shot,
                    capture,
                    device: device.clone(),
                })
                .map_err(ShotError::Worker)
        });
        match sent {
            Ok(()) => self.status = "Saving window screenshot…".into(),
            Err(error) => self.window_event(
                ctx,
                crate::io_service::WindowEvent {
                    quit,
                    result: Err(error),
                },
            ),
        }
    }

    /// A finished window screenshot: the files or the error in the status line (and the log).
    /// A quit shot (`FRAC_SNAP`) closes the window when it succeeded and makes the window exit
    /// with its error (`take_fatal`) when it failed.
    fn window_event(&mut self, ctx: &egui::Context, event: crate::io_service::WindowEvent) {
        match event.result {
            Ok(paths) => {
                let names: Vec<String> = paths.iter().map(|p| p.display().to_string()).collect();
                log::info!("Window screenshot: {}", names.join(", "));
                self.status = format!("Saved {}", names.join(", "));
                if event.quit {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
            }
            Err(error) => {
                log::error!("Window screenshot: {error}");
                self.status = format!("Window screenshot failed: {error}");
                if event.quit {
                    self.fatal = Some(format!("FRAC_SNAP: {error}"));
                }
            }
        }
    }
    fn step_viewport(&mut self, w: usize, h: usize, output_hdr: bool, white_nits: f32) {
        let policy = match self
            .world
            .document
            .viewport_policy(f64::from(self.world_ui.playhead))
        {
            Ok(policy) => policy,
            Err(error) => {
                self.viewport_render = None;
                self.status = error;
                return;
            }
        };
        // Frozen keeps the previous request, so the previous selection stays the displayed one.
        if policy.frozen {
            if let Some(request) = &mut self.request
                && (!request.paused || request.interactive)
            {
                request.paused = true;
                request.interactive = false;
                self.renderer.request_viewport(request.clone());
            }
            return;
        }
        if self.scene != self.last_scene {
            let mut traced = self.scene.clone();
            traced.animation = self.last_scene.animation.clone();
            traced.render.exposure_stops = self.last_scene.render.exposure_stops;
            traced.render.saturation = self.last_scene.render.saturation;
            traced.render.reinhard = self.last_scene.render.reinhard;
            traced.render.denoise = self.last_scene.render.denoise;
            traced.colour = self.last_scene.colour.clone();
            if traced != self.last_scene {
                self.last_change = Instant::now();
            }
            // Compare the authored output snapshot, never the previously selected moving profile.
            self.last_scene = self.scene.clone();
        }
        let moving = self.world_ui.playing
            || self.last_change.elapsed().as_secs_f32() * 1000.0 < policy.settle_delay_ms;
        let selection = match self
            .world
            .document
            .viewport_render(f64::from(self.world_ui.playhead), moving)
        {
            Ok(selection) => selection,
            Err(error) => {
                self.viewport_render = None;
                self.status = error;
                return;
            }
        };
        let effective = &selection.effective;
        let profile_key = (
            effective.profile,
            self.world.revision(),
            f64::from(self.world_ui.playhead).to_bits(),
        );
        let cached = self
            .viewport_scene
            .as_ref()
            .filter(|cached| (cached.0, cached.1, cached.2) == profile_key)
            .map(|cached| cached.3.clone());
        let mut snapshot = match cached {
            Some(scene) => scene,
            None => match self.world.document.snapshot_with_render_profile(
                effective.profile,
                f64::from(self.world_ui.playhead),
                crate::world::View::Viewport,
            ) {
                Ok(scene) => {
                    self.viewport_scene =
                        Some((profile_key.0, profile_key.1, profile_key.2, scene.clone()));
                    scene
                }
                Err(error) => {
                    self.viewport_render = None;
                    self.status = error;
                    return;
                }
            },
        };
        // Navigation is authored at the end of this UI frame; use the current pose immediately.
        snapshot.camera = self.scene.camera;
        snapshot.colour = self.scene.colour.clone();
        snapshot.animation = Default::default();
        // The one profile application (idempotent on the cached snapshot): it keeps the scene's
        // iterations and Direct's adaptive-off, which a plain `render` overwrite undid.
        effective.apply_to(&mut snapshot);
        let (width, height) = scaled_extent((w, h), effective.resolution_scale);
        self.target_spp = effective.samples;
        // Only the Moving binding is a disposable proxy; Locked motion keeps Manual's film.
        let preview = selection.target == crate::render_profiles::ProfileTarget::Moving;
        let mut req = ViewportRequest {
            active: true,
            generation: self.generation,
            scene: snapshot,
            width,
            height,
            target_spp: effective.samples,
            paused: policy.paused,
            raw: self.raw_view,
            interactive: moving,
            preview,
            batch_budget_ms: policy.batch_budget_ms.min(1000.0 / policy.target_fps),
            target_fps: policy.target_fps,
            seed: self.seed,
            output_hdr,
            white_nits,
        };
        if self.request.as_ref().is_none_or(|old| {
            old.scene != req.scene
                || old.width != width
                || old.height != height
                || old.seed != req.seed
                || old.raw != req.raw
                || old.target_spp != req.target_spp
                || old.output_hdr != output_hdr
                || old.white_nits != white_nits
        }) {
            self.generation = self.generation.wrapping_add(1);
            req.generation = self.generation;
        }
        self.viewport_stamp = Some((
            req.generation,
            playa_graph::NodeId(self.world.document.graph.id),
            self.world.revision(),
            self.world_ui.playhead,
        ));
        self.viewport_render = Some(selection);
        if self.request.as_ref() != Some(&req) {
            self.renderer.request_viewport(req.clone());
            self.request = Some(req);
        }
    }
    fn poll_events(&mut self, ctx: &egui::Context) {
        let frozen = self
            .world
            .document
            .viewport_policy(f64::from(self.world_ui.playhead))
            .is_ok_and(|policy| policy.frozen);
        while let Some(result) = self.io.poll_templates() {
            self.templates_pending = false;
            match result {
                Ok(entries) => self.templates = entries,
                Err(error) => self.status = format!("Templates: {error}"),
            }
            ctx.request_repaint();
        }
        if let Some(frame) = self.renderer.take_latest_frame()
            && !frozen
            && !self.preview.displaying_preview()
            && !(self.frame_from_cache && self.preview.presents())
        {
            self.cache_completed_viewport(&frame, ctx);
            self.showing_preview = frame.preview;
            self.spp_per_frame = frame.last_spp;
            if let Some(old) = self.frame.replace(frame)
                && self.frame_from_cache
            {
                self.preview.recycle(old);
            }
            self.frame_from_cache = false;
        }
        for event in self.renderer.drain_events() {
            if let Some(frame) = self.preview.handle(&event) {
                if frozen {
                    self.preview.recycle(frame);
                    continue;
                }
                self.showing_preview = true;
                if let Some(old) = self.frame.replace(frame)
                    && self.frame_from_cache
                {
                    self.preview.recycle(old);
                }
                self.frame_from_cache = true;
            } else if let RenderEvent::PreviewFrame { frame, .. } = &event {
                self.preview.recycle(frame.clone());
            }
            match event {
                RenderEvent::Ready { name } => self.gpu_name = name,
                // Viewport and worker errors: never a thumbnail's (that is `ThumbnailFailed`).
                RenderEvent::Error(error) => self.status = error,
                RenderEvent::Picked { id, hit } => self.picked(id, hit),
                RenderEvent::ThumbnailFailed { id, error } => {
                    if self
                        .thumb_pending
                        .as_ref()
                        .is_some_and(|(pending_id, _)| *pending_id == id)
                    {
                        self.thumb_pending = None;
                    }
                    self.fail_thumbnail(id, error);
                }
                RenderEvent::Thumbnail { id, image } => {
                    use crate::material_gallery::ThumbId;
                    let pending = if self
                        .thumb_pending
                        .as_ref()
                        .is_some_and(|(pending_id, _)| *pending_id == id)
                    {
                        self.thumb_pending.take()
                    } else {
                        None
                    };
                    let card = ThumbId::decode(id);
                    if let ThumbId::Material(_) = card {
                        if let Err(error) = self
                            .material_gallery
                            .refresh(&self.world, f64::from(self.world_ui.playhead))
                        {
                            self.material_gallery.retry_thumbnail(id);
                            self.status = error;
                            continue;
                        }
                        if !self.material_gallery.is_pending(id) {
                            continue;
                        }
                    }
                    let texture = ctx.load_texture(
                        format!("thumb-{id}"),
                        ColorImage::from_rgba_unmultiplied(
                            [image.width, image.height],
                            bytemuck::cast_slice(&image.pixels),
                        ),
                        TextureOptions::LINEAR,
                    );
                    let entries = match card {
                        ThumbId::Material(_) => {
                            self.material_gallery.accept_thumbnail(id, texture);
                            continue;
                        }
                        ThumbId::Swatch(index) => {
                            if let Some(slot) = self.swatches.get_mut(index) {
                                *slot = Some(texture);
                            }
                            continue;
                        }
                        ThumbId::Bookmark(_) => &mut self.bookmarks,
                        ThumbId::Gallery(_) => &mut self.gallery,
                    };
                    if let Some((pending_id, scene)) = pending
                        && pending_id == id
                        && let Some(index) = pending_thumbnail_entry(entries, &scene)
                    {
                        entries[index].thumb = Some(texture);
                    }
                }
                _ => {}
            }
        }
        while let Some(event) = self.io.poll_scene() {
            match self.bookmark_ops.iter().position(|(id, _)| *id == event.id) {
                Some(index) => {
                    let (_, op) = self.bookmark_ops.remove(index);
                    self.bookmark_event(op, event);
                }
                None => self.scene_file_event(event),
            }
        }
        while let Some(result) = self.io.poll() {
            self.status = result.unwrap_or_else(|error| {
                log::error!("{error}");
                error
            });
        }
        while let Some(event) = self.io.poll_window() {
            self.window_event(ctx, event);
        }
        while let Some(result) = self.io.poll_take() {
            self.take_import_result(result);
        }
    }
    pub(crate) fn gui_fps(&self) -> u32 {
        self.gui_fps.clamp(15, 240)
    }
    fn frozen_world_scene(&self) -> Scene {
        freeze_world_scene(&self.scene, &self.world.document)
    }
    /// Ctrl+D duplicates, Ctrl+C copies the selected nodes (with descendants) to the system
    /// clipboard, Ctrl+V pastes WarpBro nodes from it. Each edit is one world command / undo step.
    fn node_clipboard(&mut self, ctx: &egui::Context) {
        use crate::hotkeys::{self, Command as Hotkey, Scope};
        // Takes paste wherever the pointer is (the text says what it is); the panel under the
        // pointer owns Copy / Duplicate / Delete.
        self.paste_takes(ctx);
        if hotkeys::active(ctx) == Some(Scope::Takes) {
            self.takes_hotkeys(ctx);
            return;
        }
        let selected = self.world.targets(None);
        if hotkeys::consume(ctx, Scope::Global, Hotkey::Duplicate) && !selected.is_empty() {
            self.emit_app(AppAction::Edit {
                command: crate::world::WorldCommand::Duplicate(selected.clone()),
                gesture: None,
            });
        }
        if hotkeys::consume(ctx, Scope::Outliner, Hotkey::Delete) && !selected.is_empty() {
            self.world_ui.delete(&self.world, None);
        }
        if hotkeys::take_copy(ctx, !selected.is_empty()) {
            self.emit_app(AppAction::CopyNodes(selected.clone()));
        }
        if let Some(text) =
            hotkeys::take_paste(ctx, |text| crate::world::parse_clipboard(text).is_some())
        {
            self.emit_app(AppAction::PasteNodes(text));
        }
    }

    /// Render / Encode: the export bindings, then the job form. The form edits a copy of the
    /// bound OutputSettings recipe; a change is authored back to that node as one undoable
    /// edit per gesture, so the document stays the only store of the recipe.
    fn export_ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Render / Encode");
        let frame = f64::from(self.world_ui.playhead);
        self.export.quality = self
            .world
            .document
            .output_render_profile()
            .and_then(|id| {
                self.world
                    .document
                    .effective_render(id, f64::from(self.export.job.first))
            })
            .and_then(|effective| effective.export_quality());
        let recipe = self
            .world
            .document
            .output_settings_id()
            .and_then(|id| Ok((id, self.world.document.output_settings_of(id)?)));
        match &recipe {
            Ok((_, output)) => self.export.output = output.clone(),
            Err(error) => {
                ui.colored_label(egui::Color32::LIGHT_RED, error);
                self.status = error.clone();
            }
        }
        let timeline = (
            self.world.document.first,
            self.world.document.last,
            self.world.document.fps,
            self.world_ui.playhead,
        );
        let emitter = self.event_emitter();
        let world = &self.world;
        let mut profiles = crate::render_profiles_ui::Actions::default();
        let actions = self.export.ui(
            ui,
            timeline,
            self.colour.config(),
            &self.scene.colour,
            |ui| {
                profiles =
                    crate::render_profiles_ui::output_binding_rows(ui, world, &emitter, frame);
            },
        );
        self.profile_ui_actions(profiles);
        if let Some(job) = actions.job {
            self.emit_app(AppAction::ExportJob(job));
        }
        if let (Ok((id, before)), Some(after)) = (recipe, actions.output) {
            self.emit_app(AppAction::OutputRecipe {
                id,
                before,
                after,
                gesture: soluwell_panels::attr_grid::edit_gesture(ui.ctx()),
            });
        }
        if let Some((save, path)) = actions.preset_file {
            self.emit_app(AppAction::ExportPreset { save, path });
        }
        if let Some(request) = actions.preset_dialog {
            self.emit_app(AppAction::ExportPresetDialog(request));
        }
        for request in actions.preset_requests {
            self.emit_app(AppAction::ExportPresetRequest(request));
        }
        if actions.start {
            self.emit_app(AppAction::StartExport);
        }
        if actions.cancel {
            self.emit_app(AppAction::CancelExport);
        }
    }

    /// Unreal-style flight: hold RMB in the viewport, mouse looks, WASD moves, R/Space up, C down,
    /// Q/E roll, Shift fast, Alt slow, Ctrl aligns to the nearest plane or axis, the wheel moves along the view. Integrated by cam-controls `SpaceFlight` (thrust,
    /// inertia, damping); on release the orbit pivot is placed in front of the camera at the
    /// current orbit distance, so orbiting continues from where you flew.
    fn fly_camera(&mut self, ui: &egui::Ui, resp: &egui::Response) {
        use crate::hotkeys::{self, Command as Hotkey, Scope};
        hotkeys::register(ui, Scope::Viewport, resp.rect);
        if hotkeys::consume(ui.ctx(), Scope::Viewport, Hotkey::Flight) {
            self.emit_app(AppAction::ToggleFlight);
        }
        let held = flight_held(ui, resp);
        if hotkeys::consume(ui.ctx(), Scope::Viewport, Hotkey::Home) {
            self.emit_app(AppAction::RestoreCamera);
            return;
        }
        if !held && hotkeys::consume(ui.ctx(), Scope::Viewport, Hotkey::Fit) {
            self.emit_app(AppAction::FrameCamera {
                width: resp.rect.width().max(1.0) as u32,
                height: resp.rect.height().max(1.0) as u32,
            });
            return;
        }
        if !held && hotkeys::consume(ui.ctx(), Scope::Viewport, Hotkey::Play) {
            self.emit_app(AppAction::SetPlaying(!self.world_ui.playing));
        }
        if !held && self.fly.is_none() {
            return;
        }
        let inertia = self.controls.inertia(self.scene.camera_radius());
        let mut speed_scale = self.fly_speed_scale;
        let intents = if held {
            cam_controls_egui::gather_fly_intents(
                ui,
                &inertia,
                &mut speed_scale,
                true,
                true,
                resp.rect,
                resp.rect.center(),
            )
        } else {
            Vec::new()
        };
        self.emit_app(AppAction::Fly(FlyInput {
            intents,
            wheel: ui.input(|i| i.smooth_scroll_delta.y),
            held,
            align: ui.input(|i| i.modifiers.ctrl),
            dt: nav_dt(ui.ctx()),
            width: resp.rect.width().max(1.0) as u32,
            height: resp.rect.height().max(1.0) as u32,
            speed_scale,
        }));
    }

    /// Houdini orbit through the shared rig: LMB tumbles about world up (Ctrl: smart 2-plane /
    /// axis lock-on eased by `align_seconds`; Shift: plane; Ctrl + Shift: axis), MMB pans, the
    /// wheel zooms toward the target, and released drags coast. The camera roll is an overlay the
    /// turntable leaves alone (except under Ctrl smart alignment, which eases roll to 0°); pan
    /// turns the cursor motion by it so the view follows the cursor.
    fn orbit_camera(&mut self, ui: &egui::Ui, resp: &egui::Response, over_toolbar: bool) {
        let navigation = self.controls.navigation(
            self.scene.camera.fov_y_degrees.to_radians(),
            resp.rect.height(),
        );
        let buttons = cam_controls_egui::OrbitButtons {
            dolly: None,
            wheel: !over_toolbar,
            shift_snap: true,
            ..cam_controls_egui::OrbitButtons::HOUDINI
        };
        let frame = cam_controls_egui::gather_orbit_intents(
            ui,
            resp,
            &navigation,
            &mut self.orbit_drag,
            buttons,
        );
        if !frame.intents.is_empty() || self.orbit.rig.has_inertia() {
            self.emit_app(AppAction::Orbit(OrbitInput {
                intents: frame.intents,
                dt: nav_dt(ui.ctx()),
                width: resp.rect.width().max(1.0) as u32,
                height: resp.rect.height().max(1.0) as u32,
            }));
        }
        if frame.request_repaint || self.orbit.rig.has_inertia() {
            ui.ctx().request_repaint();
        }
    }

    fn frame_camera(&mut self, width: u32, height: u32) {
        use cam_controls::{CameraController, CameraPose, SpaceFlight};
        use glam::{Mat4, Vec3};
        let reference = self.scene.camera_radius().max(1e-4);
        if self.scene.world_render {
            let selected = self.world.targets(None);
            let matches_selection = |node: Option<crate::world::NodeId>| {
                let Some(node) = node else {
                    return false;
                };
                selected
                    .iter()
                    .any(|&sel| sel == node || self.world.document.descends(node, sel))
            };
            let object_sphere = |object: &Scene| -> (Vec3, f32) {
                let matrix = object
                    .object_world
                    .map(|m| Mat4::from_cols_array_2d(&m))
                    .unwrap_or_else(|| {
                        Mat4::from_scale_rotation_translation(
                            Vec3::splat(object.object.scale),
                            glam::Quat::IDENTITY,
                            Vec3::from_array(object.object.offset),
                        )
                    });
                let local_radius = if object.inversion_strength != 0.0 {
                    let mut local = object.clone();
                    local.object = crate::scene::ObjectTransform {
                        offset: [0.0; 3],
                        rotation_degrees: [0.0; 3],
                        scale: 1.0,
                    };
                    local.pack(1, 1)[crate::params::P_CLIP_RADIUS]
                } else {
                    object.formula.framing_radius()
                };
                (
                    matrix.transform_point3(Vec3::ZERO),
                    (local_radius * max_scale(&matrix)).max(1e-4),
                )
            };
            let bounds_sphere = |b: &crate::scene::Bounds| -> (Vec3, f32) {
                let matrix = Mat4::from_cols_array_2d(&b.world);
                (
                    matrix.transform_point3(Vec3::ZERO),
                    (b.half * max_scale(&matrix)).max(1e-4),
                )
            };
            let mut spheres: Vec<(Vec3, f32)> = self
                .scene
                .objects
                .iter()
                .filter(|o| matches_selection(o.node))
                .map(object_sphere)
                .chain(
                    self.scene
                        .bounds
                        .iter()
                        .filter(|b| matches_selection(b.node))
                        .map(bounds_sphere),
                )
                .collect();
            if spheres.is_empty() {
                spheres.extend(self.scene.objects.iter().map(object_sphere));
                spheres.extend(self.scene.bounds.iter().map(bounds_sphere));
            }
            let (center, radius) = if spheres.is_empty() {
                (Vec3::ZERO, 1.0)
            } else {
                let mut lower = Vec3::splat(f32::INFINITY);
                let mut upper = Vec3::splat(f32::NEG_INFINITY);
                for &(c, r) in &spheres {
                    lower = lower.min(c - Vec3::splat(r));
                    upper = upper.max(c + Vec3::splat(r));
                }
                let center = (lower + upper) * 0.5;
                let radius = spheres
                    .iter()
                    .map(|&(c, r)| c.distance(center) + r)
                    .fold(0.0f32, f32::max)
                    .max(1e-4);
                (center, radius)
            };
            let cam = &mut self.scene.camera;
            let half_y = (cam.fov_y_degrees.to_radians() * 0.5).tan().max(1e-4);
            let aspect = (width.max(1) as f32) / (height.max(1) as f32);
            let half_min = half_y.min((half_y * aspect).max(1e-4));
            cam.target = center.to_array();
            cam.distance = (1.1 * radius / half_min).max(0.0001) / reference;
            self.fly = None;
            self.status = "Camera framed to bounds".into();
            return;
        }
        let (min, max) = self.scene.framing_bounds();
        let cam = &mut self.scene.camera;
        let mut fly = SpaceFlight::from_pose(CameraPose {
            orientation: cam.orientation(),
            ..Default::default()
        });
        fly.projection.fov_y = cam.fov_y_degrees.to_radians();
        fly.projection.distance_min = 0.0001;
        fly.projection.distance_max = f32::MAX;
        let mut controller = CameraController::Space(fly);
        // The shared fitter uses radius/tan(FOV/2); allow for the sphere's depth,
        // especially with wide FOVs, so the entire box stays inside the frustum.
        let margin = 1.1 / (cam.fov_y_degrees.to_radians() * 0.5).cos();
        controller.frame_bounds(min, max, width, height, margin);
        let center = (min + max) * 0.5;
        cam.target = center.to_array();
        cam.distance = controller.pose().eye.distance(center) / reference;
        self.fly = None;
        self.status = "Camera framed to bounds".into();
    }

    /// View > Restore camera (H): the camera as the scene was loaded (`origin`); a flight
    /// stops and releases the cursor. Authored by the end-of-frame diff like any navigation.
    fn restore_camera(&mut self, ctx: &egui::Context) {
        self.scene.camera = self.origin.camera;
        self.fly = None;
        ctx.send_viewport_cmd(egui::ViewportCommand::CursorGrab(egui::CursorGrab::None));
        self.status = "Camera restored".into();
    }

    /// Seed the flight rig from the scene camera. `fly_camera` then drives it while RMB is
    /// held and lets its momentum and lock spring settle after release.
    fn start_flight(&mut self) {
        use cam_controls::{CameraPose, SpaceFlight};
        use glam::Vec3;
        let radius = self.scene.camera_radius();
        let cam = &self.scene.camera;
        let orientation = cam.orientation();
        let eye = Vec3::from_array(cam.target) - (orientation * -Vec3::Z) * (cam.distance * radius);
        let mut fly = SpaceFlight::from_pose(CameraPose {
            eye,
            orientation,
            ..CameraPose::default()
        });
        fly.inertia = self.controls.inertia(radius);
        // The lock takes the world plane nearest to the current roll (a camera left on its
        // side by an earlier flip keeps that plane).
        fly.set_horizon_lock((!cam.free_flight).then_some(Vec3::Y));
        self.fly = Some(fly);
    }

    /// Free 6-DoF flight or horizon lock. A live flight follows the flag in `fly_camera`; a
    /// camera at rest gets a coasting rig, so the lock spring levels it smoothly either way.
    fn toggle_flight_mode(&mut self) {
        self.scene.camera.free_flight = !self.scene.camera.free_flight;
        if !self.scene.camera.free_flight && self.fly.is_none() {
            self.start_flight();
        }
        let cam = &self.scene.camera;
        self.status = if cam.free_flight {
            "Flight: free 6-DoF · Q/E roll · R/Space up, C down · Shift fast · Alt slow"
        } else {
            "Flight: horizon lock · Q/E tilt, hold to flip the plane · R/Space up, C down · Shift fast · Alt slow"
        }
        .into();
    }

    fn current(&self) -> Option<&Frame> {
        self.frame.as_deref()
    }

    // -------------------------------------------------------------------------
    // panels
    // -------------------------------------------------------------------------

    fn top_bar(&mut self, ui: &mut egui::Ui) {
        egui::MenuBar::new().ui(ui, |ui| {
            ui.menu_button("File", |ui| {
                if !self.templates_requested {
                    self.emit_app(AppAction::RefreshTemplates);
                }
                use crate::hotkeys::{Command as Hotkey, Scope, menu_item};
                if menu_item(ui, "New scene", Scope::Global, Hotkey::NewScene)
                    .on_hover_text("Start over: the scene WarpBro opens with, nothing kept")
                    .clicked()
                {
                    self.emit_app(AppAction::NewScene);
                    ui.close();
                }
                if menu_item(ui, "Open scene…", Scope::Global, Hotkey::OpenScene).clicked() {
                    self.emit_app(AppAction::SceneDialog(crate::templates::FileAction::Open));
                    ui.close();
                }
                if menu_item(ui, "Save scene", Scope::Global, Hotkey::SaveScene).clicked() {
                    self.emit_app(AppAction::SaveScene);
                    ui.close();
                }
                if menu_item(ui, "Save scene as…", Scope::Global, Hotkey::SaveSceneAs).clicked() {
                    self.emit_app(AppAction::SceneDialog(crate::templates::FileAction::Save));
                    ui.close();
                }
                ui.menu_button("Templates", |ui| {
                    if ui.button("Save current scene as template…").clicked() {
                        self.emit_app(AppAction::SceneDialog(
                            crate::templates::FileAction::SaveTemplate,
                        ));
                        ui.close();
                    }
                    if ui.button("Open template file…").clicked() {
                        self.emit_app(AppAction::SceneDialog(
                            crate::templates::FileAction::OpenTemplate,
                        ));
                        ui.close();
                    }
                    if ui
                        .add_enabled(
                            !self.templates_pending,
                            egui::Button::new("Refresh templates"),
                        )
                        .clicked()
                    {
                        self.emit_app(AppAction::RefreshTemplates);
                    }
                    ui.separator();
                    if self.templates_pending {
                        ui.weak("Loading templates…");
                        ui.ctx().request_repaint();
                    }
                    let mut picked = None;
                    for entry in &self.templates {
                        let origin = match &entry.source {
                            crate::templates::Source::Builtin(_) => "Built-in".to_owned(),
                            crate::templates::Source::File(path) if entry.overrides => {
                                format!("{} (overrides the built-in)", path.display())
                            }
                            crate::templates::Source::File(path) => path.display().to_string(),
                        };
                        let hint = match entry.description {
                            Some(description) => format!(
                                "{description}

{origin}"
                            ),
                            None => origin,
                        };
                        if ui.button(&entry.name).on_hover_text(hint).clicked() {
                            picked = Some(entry.source.clone());
                            ui.close();
                        }
                    }
                    match picked {
                        Some(crate::templates::Source::Builtin(index)) => {
                            self.emit_app(AppAction::BuiltinTemplate(index));
                        }
                        Some(crate::templates::Source::File(path)) => {
                            self.emit_app(AppAction::SceneFile {
                                path,
                                action: crate::templates::FileAction::OpenTemplate,
                            });
                        }
                        None => {}
                    }
                });
                ui.separator();
                if ui.button("Save bookmark").clicked() {
                    self.emit_app(AppAction::SaveBookmark);
                    ui.close();
                }
                self.snapshot_menu(ui);
                ui.separator();
                self.window_shot_menu(ui);
            });
            ui.menu_button("Edit", |ui| {
                use crate::hotkeys::{Command as Hotkey, Scope, menu_item};
                // History steps wait for the next frame's history point (`History`).
                let mut history = None;
                if menu_item(ui, "Undo", Scope::Global, Hotkey::Undo).clicked() {
                    history = Some(History::Undo);
                }
                if menu_item(ui, "Redo", Scope::Global, Hotkey::Redo).clicked() {
                    history = Some(History::Redo);
                }
                ui.separator();
                if ui
                    .button("Reset scene")
                    .on_hover_text("Back to the scene as it was opened: one undoable step")
                    .clicked()
                {
                    history = Some(History::Reset);
                }
                if history.is_some() {
                    if let Some(step) = history {
                        self.emit_app(AppAction::History(step));
                    }
                    ui.ctx().request_repaint();
                    ui.close();
                }
                if ui.button("Copy scene JSON").clicked() {
                    self.emit_app(AppAction::CopyScene);
                    ui.close();
                }
                ui.separator();
                if ui.button("Settings…").clicked() {
                    self.emit_app(AppAction::OpenPanel(dock::Panel::Settings));
                    ui.close();
                }
            });
            ui.menu_button("View", |ui| {
                use crate::hotkeys::{Command as Hotkey, Scope, menu_item};
                if let Ok(policy) = self
                    .world
                    .document
                    .viewport_policy(f64::from(self.world_ui.playhead))
                {
                    let mut paused = policy.paused;
                    if ui.checkbox(&mut paused, "Pause rendering").changed() {
                        self.emit_app(AppAction::ViewportFlag {
                            path: "/viewport/paused".into(),
                            value: paused,
                        });
                    }
                }
                if ui.button("Render & Viewport settings…").clicked() {
                    self.emit_app(AppAction::OpenSettings(SettingsPage::Render));
                    ui.close();
                }
                if ui
                    .selectable_label(self.scene.camera.free_flight, "Free flight")
                    .clicked()
                {
                    self.emit_app(AppAction::ToggleFlight);
                    ui.close();
                }
                if menu_item(ui, "Restore camera", Scope::Viewport, Hotkey::Home).clicked() {
                    self.emit_app(AppAction::RestoreCamera);
                    ui.close();
                }
                if crate::hotkeys::menu_item(
                    ui,
                    "Hide UI",
                    crate::hotkeys::Scope::Global,
                    crate::hotkeys::Command::ToggleUi,
                )
                .clicked()
                {
                    self.emit_app(AppAction::ShowUi(false));
                    ui.close();
                }
            });
            ui.menu_button("Render", |ui| {
                if ui.button("Render / Encode…").clicked() {
                    self.emit_app(AppAction::OpenPanel(dock::Panel::Export));
                    ui.close();
                }
            });
            self.window_menu(ui);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                self.layout_manager_ui(ui)
            });
        });
    }

    /// The Gallery or the Bookmarks dock `panel`: scene cards; a click opens the card's scene as
    /// a new document, a bookmark's context menu deletes its file.
    fn browser(&mut self, ui: &mut egui::Ui, panel: dock::Panel) {
        let mut load: Option<(Scene, Card)> = None;
        let mut delete: Option<PathBuf> = None;
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                let entries = if panel == dock::Panel::Bookmarks {
                    &self.bookmarks
                } else {
                    &self.gallery
                };
                if entries.is_empty() {
                    ui.label("No bookmarks yet: Save bookmark stores the current scene.");
                }
                let gap = 6.0;
                let padding = egui::Frame::group(ui.style()).inner_margin.sum();
                let (columns, cell) =
                    thumbnail_grid(ui.available_width(), Vec2::new(168.0 + padding.x, 1.0), gap);
                let image_width = (cell.x - padding.x).max(1.0);
                let size = Vec2::new(image_width, image_width * THUMB_H as f32 / THUMB_W as f32);
                egui::Grid::new("scene-card-grid")
                    .spacing([gap, gap])
                    .show(ui, |ui| {
                        for (i, e) in entries.iter().enumerate() {
                            let card = match &e.path {
                                Some(path) => Card::Bookmark(path.clone()),
                                None => Card::Gallery(i),
                            };
                            let selected = self.card.as_ref() == Some(&card);
                            let frame = egui::Frame::group(ui.style()).stroke(if selected {
                                egui::Stroke::new(2.0, ui.visuals().selection.stroke.color)
                            } else {
                                ui.visuals().widgets.noninteractive.bg_stroke
                            });
                            let resp = ui
                                .push_id(i, |ui| {
                                    ui.vertical(|ui| {
                                        ui.set_width(cell.x);
                                        frame
                                            .show(ui, |ui| {
                                                ui.set_width(image_width);
                                                match &e.thumb {
                                                    Some(t) => {
                                                        ui.add(egui::Image::new((t.id(), size)));
                                                    }
                                                    None => {
                                                        let (r, _) = ui.allocate_exact_size(
                                                            size,
                                                            Sense::hover(),
                                                        );
                                                        ui.painter().rect_filled(
                                                            r,
                                                            4.0,
                                                            ui.visuals().extreme_bg_color,
                                                        );
                                                    }
                                                }
                                                ui.add(
                                                    egui::Label::new(
                                                        RichText::new(&e.scene.name).strong(),
                                                    )
                                                    .truncate(),
                                                );
                                                ui.add(
                                                    egui::Label::new(
                                                        RichText::new(e.scene.formula.name())
                                                            .small()
                                                            .weak(),
                                                    )
                                                    .truncate(),
                                                );
                                            })
                                            .response
                                            .interact(Sense::click())
                                    })
                                    .inner
                                })
                                .inner;
                            if let Some(path) = &e.path {
                                resp.context_menu(|ui| {
                                    if ui.button("Delete bookmark").clicked() {
                                        delete = Some(path.clone());
                                        ui.close();
                                    }
                                });
                            }
                            if resp.clicked() {
                                load = Some((e.scene.clone(), card));
                            }
                            if (i + 1) % columns == 0 {
                                ui.end_row();
                            }
                        }
                    });
            });
        if let Some((scene, card)) = load {
            self.emit_app(AppAction::LoadScene {
                scene: Box::new(scene),
                new_document: true,
                card: Some(card),
            });
        }
        if let Some(path) = delete {
            self.emit_app(AppAction::DeleteBookmark(path));
        }
    }

    fn inspector(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui.button("Render & Viewport profiles…").clicked() {
                self.emit_app(AppAction::OpenSettings(SettingsPage::Render));
            }
            if ui.button("New noise seed").clicked() {
                self.emit_app(AppAction::NoiseSeed);
            }
        });
        self.remember_material_targets();
        if let Some(material) = self.world.active()
            && self
                .world
                .document
                .info(material)
                .is_ok_and(|node| node.kind == crate::world::WorldKind::Material)
        {
            let apply = ui.add_enabled(!self.material_targets.is_empty(), egui::Button::new("Apply to object"))
                .on_hover_ui(|ui| {
                    if self.material_targets.is_empty() {
                        ui.label("Select an object with a Material field, then select a material to edit.");
                    } else {
                        ui.label("Assign this material to:");
                        for &id in &self.material_targets {
                            if let Ok(node) = self.world.document.info(id) { ui.label(node.name); }
                        }
                    }
                }).clicked();
            if apply {
                let targets = self.assign_targets(AssignTo::Selected);
                self.emit_app(AppAction::AssignMaterial { material, targets });
            }
        }
        self.world_ui.inspector(ui, &self.world);
        if std::mem::take(&mut self.world_ui.material_library_requested) {
            self.emit_app(AppAction::OpenPanel(dock::Panel::Materials));
        }
    }

    /// The status bar in its resizable (`soluwell_chrome::statusbar`) or fixed layout: the same
    /// sections (`StatusItem`) drawn by one `status_item`.
    fn status_bar(&mut self, ui: &mut egui::Ui) {
        let mut status_layout = self.status_layout.clone();
        let mut resizable = self.status_resizable;
        ui.scope(|ui| {
            if resizable {
                // Fixed section identity / count: transient frame and OIDN states never move
                // widths onto another indicator.
                let mut layout = std::mem::take(&mut status_layout);
                soluwell_chrome::statusbar::StatusBar::new().show_with(
                    ui,
                    &mut layout,
                    &StatusItem::WIDTHS,
                    |index, ui| {
                        ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Truncate);
                        if let Some(item) = StatusItem::ALL.get(index) {
                            self.status_item(ui, *item, false);
                        }
                    },
                );
                status_layout = layout;
            } else {
                ui.horizontal(|ui| {
                    ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Truncate);
                    let mut separate = false;
                    for item in StatusItem::ALL {
                        separate |= self.status_item(ui, item, separate);
                    }
                });
            }
        })
        .response
        .context_menu(|ui| {
            ui.checkbox(&mut resizable, "Resizable sections");
            if ui.button("Reset section widths").clicked() {
                status_layout = Default::default();
                ui.close();
            }
        });
        if status_layout != self.status_layout || resizable != self.status_resizable {
            self.emit_app(AppAction::PatchSettings(serde_json::json!({ "status_layout": status_layout, "status_resizable": resizable })));
        }
    }

    fn status_gpu_label(&self) -> String {
        let backend = self
            .viewport_render
            .as_ref()
            .map(|s| s.effective.render.backend)
            .unwrap_or(self.scene.render.backend);
        let backend = if backend == crate::render_profiles::Backend::Auto
            && (self
                .request
                .as_ref()
                .map_or(&self.scene, |r| &r.scene)
                .uses_extended_formula()
                || self.gpu_name.ends_with("(WGPU)"))
        {
            crate::render_profiles::Backend::Wgpu
        } else {
            backend
        };
        match backend {
            crate::render_profiles::Backend::Wgpu => format!("WGPU · {}", self.gpu_name),
            crate::render_profiles::Backend::Cuda | crate::render_profiles::Backend::Auto => {
                if self.gpu_name.starts_with("Initializing") {
                    self.gpu_name.clone()
                } else {
                    format!("CUDA · {}", self.gpu_name)
                }
            }
        }
    }

    /// Draw status bar section `item` when it has something to show; returns whether it did.
    /// `separate` puts a separator before a shown section (the fixed layout; the resizable one
    /// draws its own dividers).
    fn status_item(&self, ui: &mut egui::Ui, item: StatusItem, separate: bool) -> bool {
        let frame = self.current();
        let shown = match item {
            StatusItem::Gpu | StatusItem::Flight | StatusItem::Batch | StatusItem::UiFps => true,
            StatusItem::Size | StatusItem::Samples | StatusItem::Speed => frame.is_some(),
            StatusItem::Denoise => {
                frame.is_some_and(|f| f.denoise_error.is_some() || f.denoised_samples > 0)
            }
            StatusItem::March => frame.is_some_and(|f| f.unresolved > 0.0),
            StatusItem::Message => {
                !self.status.is_empty() || crate::render_diagnostics::compiling().is_some()
            }
        };
        if !shown {
            return false;
        }
        if separate {
            ui.separator();
        }
        match (item, frame) {
            (StatusItem::Gpu, _) => {
                let label = self.status_gpu_label();
                ui.label(RichText::new(&label).weak()).on_hover_text(&label);
            }
            (StatusItem::Flight, _) => {
                ui.label(if self.scene.camera.free_flight {
                    "Free flight · `"
                } else {
                    "Horizon · `"
                });
            }
            (StatusItem::Size, Some(frame)) => {
                ui.label(format!("{}×{}", frame.width, frame.height));
            }
            (StatusItem::Samples, Some(frame)) => {
                sample_status_ui(ui, frame, self.target_spp, self.showing_preview, 120.0);
            }
            (StatusItem::Speed, Some(frame)) => {
                ui.label(format!("{:.1} Msamples/s", frame.msamples_per_s()));
            }
            (StatusItem::Batch, _) => {
                ui.label(format!("{} spp/batch", self.spp_per_frame));
            }
            (StatusItem::Denoise, Some(frame)) => match &frame.denoise_error {
                Some(error) => {
                    ui.colored_label(WARNING, "OIDN failed")
                        .on_hover_text(error);
                }
                None => {
                    ui.label(format!(
                        "OIDN {} spp · {:.1} ms",
                        frame.denoised_samples, frame.denoise_ms
                    ));
                }
            },
            (StatusItem::UiFps, _) => {
                ui.label(format!("UI {:.0} fps", 1000.0 / self.frame_ms.max(0.1)));
            }
            (StatusItem::March, Some(frame)) => march_limit_label(ui, frame.unresolved),
            (StatusItem::Message, _) => {
                if let Some((entry, seconds)) = crate::render_diagnostics::compiling() {
                    let message = format!("Preparing shader · {seconds:.0}s · {entry}");
                    ui.colored_label(WARNING, &message).on_hover_text(&message);
                    ui.ctx()
                        .request_repaint_after(std::time::Duration::from_millis(250));
                } else {
                    ui.label(RichText::new(&self.status).weak())
                        .on_hover_text(&self.status);
                }
            }
            // A frame section without a frame is not shown (`shown`).
            (_, None) => {}
        }
        true
    }

    /// A viewport click (Blender / Maya): the extras' wire under the cursor first (cameras,
    /// lights, masks, groups, Bounds cubes), then a Bounds display's cube the cursor's ray
    /// enters, else the fractal under it through one GPU ray of the shown frame
    /// (`Command::Pick`, answered in [`Self::picked`]). Shift adds or removes; Ctrl focuses the
    /// camera at the fractal under the cursor instead (depth of field); a click on nothing
    /// clears the selection. Only what the viewport may select is picked.
    fn viewport_click(&mut self, ui: &egui::Ui, resp: &egui::Response, rect: egui::Rect) {
        let Some(pos) = resp.interact_pointer_pos() else {
            return;
        };
        // A played preview shows cached frames of other moments than the live request.
        if self.preview.displaying_preview() {
            self.status = "Picking needs the live viewport: stop the preview playback".into();
            return;
        }
        let Some(request) = self.request.as_ref() else {
            return;
        };
        let (add, focus) = ui.input(|i| (i.modifiers.shift, i.modifiers.command));
        let aspect = rect.width() / rect.height().max(1.0);
        let (view, projection) = self.scene.camera_matrices(aspect);
        let ndc = [
            2.0 * (pos.x - rect.left()) / rect.width().max(1.0) - 1.0,
            1.0 - 2.0 * (pos.y - rect.top()) / rect.height().max(1.0),
        ];
        if !focus {
            let frame = f64::from(self.world_ui.playhead);
            let document = &self.world.document;
            let policy = match document.viewport_policy(frame) {
                Ok(policy) => policy,
                Err(error) => {
                    self.status = error;
                    return;
                }
            };
            // A node whose visibility cannot be evaluated is not picked; the error is shown.
            let failed = std::cell::Cell::new(None);
            let selects = |node| {
                viewport_selects(document, &policy, frame, node).unwrap_or_else(|error| {
                    failed.set(Some(error));
                    false
                })
            };
            let bounds: Vec<_> = crate::overlay::bounds(&request.scene.bounds, &[]).collect();
            let extras = self.overlays.iter().flat_map(|(_, wires)| wires.iter());
            let wire = crate::overlay::wire_at(
                extras.chain(bounds.iter()),
                projection * view,
                rect,
                pos,
                PICK_REACH,
                selects,
            );
            // The cursor's world ray: through the near plane and a point beyond it.
            let unproject = (projection * view).inverse();
            let near = unproject.project_point3(glam::Vec3::new(ndc[0], ndc[1], 0.0));
            let beyond = unproject.project_point3(glam::Vec3::new(ndc[0], ndc[1], 0.5));
            let cube = request
                .scene
                .bounds
                .iter()
                .filter_map(|b| {
                    let node = b.node.filter(|node| selects(*node))?;
                    Some((b.hit(near, (beyond - near).normalize_or_zero())?, node))
                })
                .min_by(|a, b| a.0.total_cmp(&b.0))
                .map(|(_, node)| node);
            if let Some(error) = failed.into_inner() {
                self.status = error;
            }
            if let Some(node) = wire.or(cube) {
                self.emit_app(AppAction::Select {
                    id: Some(node),
                    add,
                });
                return;
            }
        }
        if request.scene.objects.is_empty() {
            if focus {
                self.status = "Focus: no traced fractal in this display".into();
            } else {
                self.emit_app(AppAction::Select { id: None, add });
            }
            return;
        }
        self.emit_app(AppAction::Pick { ndc, add, focus });
    }

    /// A GPU pick's answer ([`Self::viewport_click`]); a stale one (a newer click) is dropped.
    fn picked(&mut self, id: u64, hit: Result<Option<(usize, f32)>, String>) {
        let Some(pending) = self.pick_pending.take_if(|p| p.id == id) else {
            return;
        };
        if pending.document != playa_graph::NodeId(self.world.document.graph.id)
            || pending.camera != self.world.document.active_camera
            || pending.frame != f64::from(self.world_ui.playhead)
            || (pending.focus
                && self
                    .request
                    .as_ref()
                    .is_none_or(|request| request.generation != pending.generation))
        {
            return;
        }
        let hit = match hit {
            Ok(hit) => hit,
            Err(error) => {
                self.status = error;
                return;
            }
        };
        if pending.focus {
            match hit {
                Some((_, depth)) if pending.camera_radius > 0.0 => {
                    // Authored with the camera pose at the end of the frame (navigation).
                    self.emit_app(AppAction::FocusDistance(depth / pending.camera_radius));
                    self.status = if self.scene.camera.lens_radius() > 0.0 {
                        format!("Focus at {depth:.3}")
                    } else {
                        format!("Focus at {depth:.3} (no depth of field: the aperture is closed)")
                    };
                }
                _ => self.status = "Focus: no fractal under the cursor".into(),
            }
            return;
        }
        let frame = f64::from(self.world_ui.playhead);
        let document = &self.world.document;
        let node = match hit.and_then(|(index, _)| pending.nodes.get(index).copied().flatten()) {
            Some(node) => match document
                .viewport_policy(frame)
                .and_then(|policy| viewport_selects(document, &policy, frame, node))
            {
                Ok(selects) => selects.then_some(node),
                Err(error) => {
                    self.status = error;
                    return;
                }
            },
            None => None,
        };
        if hit.is_some() && node.is_none() {
            // A fractal the viewport does not let pick: the click changes nothing.
            return;
        }
        self.emit_app(AppAction::Select {
            id: node,
            add: pending.add,
        });
    }

    fn viewport(&mut self, ui: &mut egui::Ui) {
        self.viewport_visible = true;
        let avail = ui.available_size();
        let (rect, resp) = ui.allocate_exact_size(avail, Sense::click_and_drag());
        self.viewport_rect = Some(rect);
        let toolbar = self.viewport_toolbar(ui, rect);
        if !toolbar.contains_pointer || self.fly.is_some() {
            self.fly_camera(ui, &resp);
            self.handle_events(ui.ctx());
        }
        // A hovered or dragged handle owns the left button (last frame's hover: the gizmo
        // is drawn over the image, after the camera rigs run).
        let gizmo_owns = self.gizmo.is_hovered() || self.gizmo_gesture.is_some();
        if !flight_held(ui, &resp) && !gizmo_owns {
            self.orbit_camera(ui, &resp, toolbar.contains_pointer);
            self.handle_events(ui.ctx());
        }
        // Held RMB gives the keyboard to the flight rig (W, Q, E, R are thrust and roll there);
        // released, Q W E R pick tools and + / − resize the manipulator, also while the flight
        // still coasts. Gated on the button, not on `self.fly`, which outlives the release.
        use crate::hotkeys::{self, Command as Hotkey, Scope};
        if !flight_held(ui, &resp) {
            for tool in ViewTool::ALL {
                if hotkeys::consume(ui.ctx(), Scope::Viewport, Hotkey::Tool(tool)) {
                    self.emit_app(AppAction::SetTool(tool));
                }
            }
            if self.tool != ViewTool::Select {
                let larger = hotkeys::consume(ui.ctx(), Scope::Viewport, Hotkey::GizmoLarger);
                let smaller = hotkeys::consume(ui.ctx(), Scope::Viewport, Hotkey::GizmoSmaller);
                if larger || smaller {
                    self.emit_app(AppAction::GizmoSize(larger));
                }
            }
        }
        if resp.double_clicked() {
            self.emit_app(AppAction::RecenterCamera);
        }

        self.handle_events(ui.ctx());
        let (w, h) = viewport_pixels(avail, ui.ctx().pixels_per_point());
        let monitor = Monitor::read(ui.ctx());
        let (output_hdr, white) = (monitor.hdr, monitor.white_nits);
        // A pass egui discards may lay the viewport out at another size: only the shown pass
        // decides that the cached preview no longer fits.
        if !ui.ctx().will_discard()
            && self
                .preview_key
                .is_some_and(|key| key.width != w || key.height != h)
        {
            self.emit_app(AppAction::CancelPreview);
        }
        self.handle_events(ui.ctx());
        if !self.preview.displaying_preview() && !(self.frame_from_cache && self.preview.presents())
        {
            self.step_viewport(w, h, output_hdr, white);
        }
        if let Some(t) = self.frame.clone() {
            if (!output_hdr || !t.hdr_bytes.is_empty())
                && (self.staged_output != Some((output_hdr, white))
                    || self
                        .staged_frame
                        .as_ref()
                        .is_none_or(|old| !Arc::ptr_eq(old, &t)))
            {
                let mvp = soluwell_viewport::hdr_view::Mvp {
                    model: glam::Mat4::from_scale(glam::vec3(2.0, 2.0, 1.0)).to_cols_array_2d(),
                    ..Default::default()
                };
                // A paint callback that panicked mid-draw leaves the view usable: take it back.
                let mut view = self
                    .hdr_view
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                view.set_output_format(soluwell_viewport::display::CANVAS_FORMAT);
                let (format, bytes) = if output_hdr {
                    (
                        soluwell_viewport::hdr_view::HdrFormat::Rgba32F,
                        t.hdr_bytes.clone(),
                    )
                } else {
                    (
                        soluwell_viewport::hdr_view::HdrFormat::Rgba8,
                        t.sdr_bytes.clone(),
                    )
                };
                // Straight: the blend the view always drew these frames with.
                let staged = view.stage_frame(
                    format,
                    soluwell_viewport::hdr_view::AlphaMode::Straight,
                    bytes,
                    t.width,
                    t.height,
                    mvp,
                );
                // On an error the previous frame stays; the same input is not retried.
                if let Err(error) = staged {
                    self.status = format!("Viewport frame not shown: {error}");
                }
                self.staged_frame = Some(t);
                self.staged_output = Some((output_hdr, white));
            }
            ui.painter().add(egui_wgpu::Callback::new_paint_callback(
                rect,
                soluwell_viewport::hdr_view::HdrPaintCallback {
                    inner: self.hdr_view.clone(),
                },
            ));
        }
        self.draw_overlays(ui, rect);
        self.manipulate(ui, rect);
        // The second click of a double click recentres instead.
        if resp.clicked()
            && !resp.double_clicked()
            && !gizmo_owns
            && !flight_held(ui, &resp)
            && !toolbar.contains_pointer
        {
            self.viewport_click(ui, &resp, rect);
        }
        if self.export.is_running() {
            let text = format!("Rendering · {}", self.export.status);
            ui.painter().text(
                rect.left_top() + Vec2::new(12.0, 12.0),
                egui::Align2::LEFT_TOP,
                text,
                egui::FontId::proportional(14.0),
                Color32::WHITE,
            );
        }
    }
}

/// A viewport click waiting for its GPU pick: what each object index of the picked scene
/// selects, and what the click asked for (`App::viewport_click`).
struct PendingPick {
    id: u64,
    document: crate::world::NodeId,
    camera: Option<crate::world::NodeId>,
    frame: f64,
    generation: u64,
    nodes: Vec<Option<playa_graph::NodeId>>,
    add: bool,
    focus: bool,
    /// The picked scene's `Scene::camera_radius`: a focus distance is in its units.
    camera_radius: f32,
}

/// How far (points) from a wire of the viewport extras a click still picks its node.
const PICK_REACH: f32 = 6.0;

/// Whether the viewport lets `node` be picked and manipulated at `frame`: shown, and of a kind
/// the viewport's filter lets pick (`/viewport/select/<kind>`). A node the document no longer
/// has is not selectable; an evaluation error is returned.
fn viewport_selects(
    document: &WorldDocument,
    policy: &crate::render_profiles::ViewportPolicy,
    frame: f64,
    node: crate::world::NodeId,
) -> Result<bool, String> {
    let Ok(info) = document.info(node) else {
        return Ok(false);
    };
    Ok(policy.selectable.contains(info.kind)
        && document.shows(node, info.kind, frame, policy.shown)?)
}

/// Whether a loaded scene is the same document again or a new one made from it (`App::load`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Load {
    /// A scene file opened (again): its identity stays.
    Same,
    /// A template, preset or bookmark instantiated: a document of its own.
    New,
}

/// What a cached preview was rendered for: any difference makes the cache stale. Compared
/// on repaint, so it holds identities and bits rather than the document itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PreviewKey {
    document: playa_graph::NodeId,
    revision: u64,
    output_hdr: bool,
    /// `f32::to_bits` of the display's SDR reference white (nits).
    white_nits_bits: u32,
    seed: u32,
    /// The viewport in physical pixels (`viewport_pixels`), as the viewport compares it.
    width: usize,
    height: usize,
    /// `f32::to_bits` of the Still profile's resolution scale (the render size per viewport pixel).
    scale_bits: u32,
    /// The Still profile's sample target.
    samples: u32,
}

impl PreviewKey {
    /// The key without the viewport size: what the frames show, not the size they fit.
    fn content(self) -> Self {
        Self {
            width: 0,
            height: 0,
            ..self
        }
    }
}

/// The render size of a viewport of `pixels` (`viewport_pixels`) at a profile's resolution
/// `scale`: what the viewport and the cached preview render at.
fn scaled_extent(pixels: (usize, usize), scale: f32) -> (usize, usize) {
    (
        ((pixels.0 as f32 * scale).round() as usize).max(1),
        ((pixels.1 as f32 * scale).round() as usize).max(1),
    )
}

/// The viewport's size in physical pixels: the one measure the viewport renders at (scaled by the
/// profile's resolution) and the cached preview is keyed by (`PreviewKey`).
fn viewport_pixels(size: egui::Vec2, ppp: f32) -> (usize, usize) {
    (
        ((size.x * ppp) as usize).max(16),
        ((size.y * ppp) as usize).max(16),
    )
}

impl App {
    /// Identify cached render content without serializing the document on repaint: the
    /// document state, the monitor, the viewport's size (as laid out, else as last requested)
    /// and the Still profile's scale and samples. Errors when the profile does not evaluate or
    /// the viewport has no size yet (a preview renders at the viewport's size).
    fn preview_identity(&self, ctx: &egui::Context) -> Result<PreviewKey, String> {
        let monitor = Monitor::read(ctx);
        let effective = self
            .world
            .document
            .viewport_render(f64::from(self.world_ui.playhead), false)?
            .effective;
        let (width, height) = match (self.viewport_rect, &self.request) {
            (Some(rect), _) => viewport_pixels(rect.size(), ctx.pixels_per_point()),
            (None, Some(request)) => (request.width, request.height),
            (None, None) => {
                return Err("The viewport has no size yet: a preview renders at its size".into());
            }
        };
        Ok(PreviewKey {
            document: playa_graph::NodeId(self.world.document.graph.id),
            revision: self.world.revision(),
            output_hdr: monitor.hdr,
            white_nits_bits: monitor.white_nits.to_bits(),
            seed: self.seed,
            width: width.max(1),
            height: height.max(1),
            scale_bits: effective.resolution_scale.to_bits(),
            samples: effective.samples,
        })
    }

    /// The stamp rejects late frames after scrubbing or authoring; interactive
    /// proxies and incomplete accumulations are presentation-only, never cache entries.
    fn cache_completed_viewport(&mut self, frame: &Arc<Frame>, ctx: &egui::Context) {
        let Some(viewport) = &self.request else {
            return;
        };
        if frame.preview
            || !frame.complete(viewport.target_spp)
            || frame.width != viewport.width
            || frame.height != viewport.height
            || self.viewport_stamp
                != Some((
                    frame.generation,
                    playa_graph::NodeId(self.world.document.graph.id),
                    self.world.revision(),
                    self.world_ui.playhead,
                ))
            || frame.colour_error.is_some()
        {
            return;
        }
        let key = match self.preview_identity(ctx) {
            Ok(key) => key,
            Err(error) => {
                self.status = error;
                return;
            }
        };
        if viewport.target_spp != key.samples
            || viewport.output_hdr != key.output_hdr
            || viewport.white_nits.to_bits() != key.white_nits_bits
            || viewport.seed != key.seed
        {
            return;
        }
        let request = if self.preview_key == Some(key) {
            let Some(request) = self.preview.cached_request() else {
                return;
            };
            request
        } else {
            self.preview_sequence = self.preview_sequence.wrapping_add(1);
            match self.make_preview_request(
                self.world.document.first,
                self.world.document.last,
                key.samples,
                ctx,
            ) {
                Ok(request) => request,
                Err(error) => {
                    self.status = error;
                    return;
                }
            }
        };
        match self
            .preview
            .cache_viewport(request, self.world_ui.playhead, frame.clone())
        {
            Ok(()) => self.preview_key = Some(key),
            Err(error) => self.status = error,
        }
    }

    /// The preview render of frames `first..=last` at `spp` samples: the frozen document (the
    /// cached preview's while its identity holds), the Still profile, the viewport's size and
    /// the preview cache budget of the settings (`preview_cache`).
    fn make_preview_request(
        &self,
        first: u32,
        last: u32,
        spp: u32,
        ctx: &egui::Context,
    ) -> Result<crate::preview::PreviewRequest, String> {
        let key = self.preview_identity(ctx)?;
        let scene = if self.preview_key == Some(key) {
            self.preview.cached_request().map(|r| r.scene)
        } else {
            None
        }
        .unwrap_or_else(|| Arc::new(self.frozen_world_scene()));
        let size = scaled_extent((key.width, key.height), f32::from_bits(key.scale_bits));
        let policy = self.world.document.viewport_policy(f64::from(first))?;
        Ok(crate::preview::PreviewRequest {
            generation: self.preview_sequence,
            scene,
            render_profile: Some(policy.selected(false).1),
            first,
            last,
            fps: self.world.document.fps as f32,
            width: size.0,
            height: size.1,
            spp,
            seed: self.seed,
            budget_ms: crate::render_service::BACKGROUND_BUDGET_MS,
            output_hdr: key.output_hdr,
            white_nits: f32::from_bits(key.white_nits_bits),
            cache_fraction: f64::from(self.preview_cache.fraction),
            reserve_gb: f64::from(self.preview_cache.reserve_gb),
        })
    }

    fn begin_preview(
        &mut self,
        first: u32,
        last: u32,
        mode: crate::preview::PreviewMode,
        ctx: &egui::Context,
    ) {
        self.world.finish_edit();
        self.preview_sequence = self.preview_sequence.wrapping_add(1);
        let prepared = self.preview_identity(ctx).and_then(|key| {
            let request = self.make_preview_request(first, last, mode.samples(key.samples), ctx)?;
            Ok((request, key))
        });
        match prepared {
            Ok((request, key)) => self.start_preview_request(request, mode.cache_all(), key),
            Err(error) => {
                self.world_ui.playing = false;
                self.status = error;
            }
        }
    }

    fn start_preview_request(
        &mut self,
        request: crate::preview::PreviewRequest,
        cache_all: bool,
        key: PreviewKey,
    ) {
        let first = request.first;
        match self.preview.start(request, cache_all) {
            Ok(()) => {
                self.preview_key = Some(key);
                self.preview_progress = None;
                self.world_ui.seek(first);
                self.world_ui.playing = true;
                if let Some(request) = &mut self.request {
                    request.active = false;
                    self.renderer.request_viewport(request.clone());
                }
            }
            Err(error) => {
                self.world_ui.playing = false;
                self.status = error;
            }
        }
    }

    fn sync_preview(&mut self, dt: f32, ctx: &egui::Context) {
        self.preview.set_loop(self.world_ui.looping);
        // Navigation over the viewport ends a played preview: a press or the wheel there, or a
        // viewport key that moves the camera (`hotkeys`, under the viewport's scope rules).
        use crate::hotkeys::{Command as Hotkey, Scope};
        let navigating = self.viewport_rect.is_some_and(|rect| {
            ctx.input(|input| {
                input.pointer.hover_pos().is_some_and(|p| rect.contains(p))
                    && (input.pointer.any_down() || input.smooth_scroll_delta.y != 0.0)
            })
        }) || crate::hotkeys::pressed(
            ctx,
            Scope::Viewport,
            &[Hotkey::Fit, Hotkey::Home, Hotkey::Flight],
        );
        if navigating && self.preview.position().is_some() {
            self.emit_app(AppAction::CancelPreview);
        }
        // The viewport alone decides whether the size still fits (on the pass that is shown):
        // a discarded pass's layout must not cancel playback here. A preview whose identity
        // cannot be told any more is stale too.
        let stale = self.preview_key.map(|key| {
            self.preview_identity(ctx)
                .map(|current| key.content() != current.content())
        });
        if let Some(Err(error)) = &stale {
            self.status = error.clone();
        }
        if matches!(stale, Some(Ok(true) | Err(_))) {
            self.preview.cancel();
            self.preview_key = None;
            self.showing_preview = false;
            self.world_ui.playing = false;
        }
        if let Some(position) = self.preview.position() {
            if self.world_ui.playhead != position {
                self.preview.scrub(self.world_ui.playhead);
            }
            if self.world_ui.playing != self.preview.running() {
                if self.world_ui.playing {
                    self.preview.resume();
                } else {
                    self.preview.pause();
                }
            }
        }
        if self.preview.displaying_preview()
            && let Some(request) = &mut self.request
            && request.active
        {
            request.active = false;
            self.renderer.request_viewport(request.clone());
        }
        self.world_ui.cached_frames = self.preview.resident.clone();
        self.world_ui.cache_draft = self
            .preview
            .cached_spp()
            .is_some_and(|s| s < self.target_spp);
        if let Some(number) = self.preview.update(dt, &self.renderer) {
            self.world_ui.seek(number);
        }
        if self.preview.position().is_some() {
            self.world_ui.playing = self.preview.running();
        }
        if self.preview.caching() {
            let progress = self.preview.progress();
            if self.preview_progress != Some(progress) {
                self.preview_progress = Some(progress);
                self.status = format!(
                    "Caching preview ({} spp): {}/{} frames · {} MiB",
                    self.preview.cached_spp().unwrap_or(self.target_spp),
                    progress.0,
                    progress.1,
                    progress.2 / 1048576
                );
            }
        }
        if let Some(error) = self.preview.error() {
            if self.status != error {
                self.status.clear();
                self.status.push_str(error);
            }
            self.world_ui.playing = false;
        }
    }

    /// A take picked on the timeline (its bar) becomes the Takes panel's selection.
    fn process_picked_take(&mut self) {
        if let Some(id) = self.world_ui.take_picked() {
            self.emit_take(takes_ui::Action::Select {
                id,
                ctrl: false,
                shift: false,
            });
        }
    }

    fn process_preview_intent(&mut self, ctx: &egui::Context) {
        // A discarded pass's layout is not the one shown: start on the pass that sizes the viewport.
        if ctx.will_discard() {
            return;
        }
        match self.world_ui.take_preview_action() {
            Some(action) => self.emit_app(AppAction::Preview {
                first: action.first,
                last: action.last,
                mode: action.mode,
            }),
            None if self.preview.position().is_none() && self.world_ui.playing => {
                let (first, last) = self
                    .world_ui
                    .playback_range
                    .unwrap_or((self.world.document.first, self.world.document.last));
                self.emit_app(AppAction::PlayPreview {
                    first,
                    last,
                    head: self.world_ui.playhead,
                });
            }
            None => {}
        }
        self.sync_preview(0.0, ctx);
    }

    fn refresh_scene(&mut self) -> Result<(), String> {
        // Cached playback presents worker-owned frames; keep the frozen scene instead
        // of rebuilding serialized world data on every clock tick. An edit or paused
        // navigation evaluates the current pose before any authoring takes place.
        if self.preview.running()
            && self.preview_key.is_some_and(|key| {
                key.document == playa_graph::NodeId(self.world.document.graph.id)
                    && key.revision == self.world.revision()
            })
        {
            return Ok(());
        }
        let frame = f64::from(self.world_ui.playhead);
        let key = (
            playa_graph::NodeId(self.world.document.graph.id),
            self.world.revision(),
            frame.to_bits(),
        );
        if self.evaluated_world != Some(key) {
            self.scene = self.world.document.snapshot(frame)?;
            self.evaluated_world = Some(key);
            self.snapshot_before = Some(self.scene.clone());
            #[cfg(test)]
            {
                self.snapshot_clones += 1;
            }
        }
        // The document owns the colour selection: the colour controls show its selection
        // (after a load, undo, redo or any edit); a rebuild happens only when it differs.
        self.colour.set_sel(&self.scene.colour);
        Ok(())
    }

    /// Whether something advances without input or a worker's result: worker results (frames,
    /// thumbnails, picks, file answers, colour configs, export progress) wake the UI themselves
    /// (`render_service::Wake`). What is left: the clock (playback, a played preview, the
    /// interactive viewport settling into its still profile), motion that coasts, a drag,
    /// a preview cache that retries refused commands, a video encode (no worker events) and an
    /// armed window shot (captured after the next frame). While it holds, every frame asks for
    /// the next one (the window caps them at `gui_fps`); otherwise the window sleeps until
    /// input or a wake (`window.rs` honours egui's repaint requests).
    fn animating(&self) -> bool {
        self.request
            .as_ref()
            .is_some_and(|request| request.active && request.interactive)
            || self.world_ui.playing
            || self.preview.caching()
            || self.preview.running()
            || self.export.is_encoding()
            || self.fly.is_some()
            || self.orbit.rig.has_inertia()
            || self.takes_recorder.recording()
            || self.gizmo_gesture.is_some()
            || self.window_shot.is_some()
    }

    pub(crate) fn ui(&mut self, root: &mut egui::Ui) {
        let ctx = root.ctx().clone();
        self.wake.attach(&ctx);
        self.navigated = false;
        self.apply_fonts(&ctx);
        self.update_scene_picker(&ctx);
        self.update_take_picker(&ctx);
        self.colour.poll();
        self.world.new_key = self.world_ui.new_key;
        let dt = ctx.input(|i| i.stable_dt).max(1.0e-4);
        self.frame_ms = self.frame_ms * 0.9 + dt * 1000.0 * 0.1;
        self.sync_preview(dt, &ctx);
        if self.preview.position().is_none() {
            self.world_ui.advance(dt, &self.world);
        }
        if let Err(error) = self.refresh_scene() {
            self.world_ui.playing = false;
            self.status = error;
        }
        self.poll_control(&ctx);
        self.handle_events(&ctx);
        use crate::hotkeys::{self, Command as Hotkey, Scope};
        if hotkeys::consume(&ctx, Scope::Global, Hotkey::ToggleUi) {
            self.emit_app(AppAction::ShowUi(!self.show_ui));
        }
        if hotkeys::consume(&ctx, Scope::Global, Hotkey::AddNode) {
            let at = ctx
                .pointer_hover_pos()
                .unwrap_or_else(|| ctx.content_rect().center());
            self.world_ui.open_add_menu(at);
        }
        // A window shot armed in an earlier frame is captured after this one.
        if let Some((_, drawn)) = &mut self.window_shot {
            *drawn = true;
        }
        if hotkeys::consume(&ctx, Scope::Global, Hotkey::NewScene) {
            self.emit_app(AppAction::NewScene);
        }
        if hotkeys::consume(&ctx, Scope::Global, Hotkey::OpenScene) {
            self.emit_app(AppAction::SceneDialog(crate::templates::FileAction::Open));
        }
        if hotkeys::consume(&ctx, Scope::Global, Hotkey::SaveScene) {
            self.emit_app(AppAction::SaveScene);
        }
        if hotkeys::consume(&ctx, Scope::Global, Hotkey::SaveSceneAs) {
            self.emit_app(AppAction::SceneDialog(crate::templates::FileAction::Save));
        }
        if hotkeys::consume(&ctx, Scope::Global, Hotkey::WindowShot) {
            self.emit_app(AppAction::WindowShot {
                files: crate::window_shot::WindowFiles::ExrAndPq,
                monitor: Monitor::read(&ctx),
            });
        }
        // Global intents are accepted before panels capture their edit context.
        let history = if hotkeys::consume(&ctx, Scope::Global, Hotkey::Redo) {
            Some(History::Redo)
        } else if hotkeys::consume(&ctx, Scope::Global, Hotkey::Undo) {
            Some(History::Undo)
        } else {
            None
        };
        if let Some(step) = history {
            self.emit_app(AppAction::History(step));
        }
        self.node_clipboard(&ctx);
        self.handle_events(&ctx);

        if let Err(error) = self.refresh_scene() {
            self.status = error;
        }
        let before_camera = self.scene.camera;
        let edit_origin = self.load_revision;
        self.update_config_picker(&ctx);
        let thumbs_pending = self.render_one_thumbnail();
        self.poll_events(&ctx);
        self.export.update();
        self.viewport_visible = false;
        self.remember_material_targets();

        if self.show_ui {
            egui::Panel::top("top").show(root, |ui| self.top_bar(ui));
            egui::Panel::bottom("status").show(root, |ui| self.status_bar(ui));
        }
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(root, |ui| {
                if self.show_ui {
                    self.dock_ui(ui);
                } else {
                    self.viewport(ui);
                }
            });
        self.world_ui.add_menu_ui(root.ctx(), &self.world);
        self.world_ui.mark_shortcuts(root.ctx(), &self.world);
        self.handle_events(&ctx);

        if !self.viewport_visible {
            if let Some(request) = &mut self.request
                && request.active
            {
                request.active = false;
                self.renderer.request_viewport(request.clone());
            }
            self.fly = None;
            ctx.send_viewport_cmd(egui::ViewportCommand::CursorGrab(egui::CursorGrab::None));
        }
        // A held button is a camera gesture unless the manipulator owns the drag: its own gesture
        // must stay the open undo step until release, not be closed by this one every frame.
        let camera_gesture = (self.fly.is_some()
            || self.orbit.rig.has_inertia()
            || (self.gizmo_gesture.is_none() && ctx.input(|i| i.pointer.any_down())))
        .then(|| self.camera_gesture());
        if self.load_revision == edit_origin {
            self.record_take(&ctx, before_camera, camera_gesture);
        }
        self.remember_material_targets();
        self.process_preview_intent(&ctx);
        self.process_picked_take();
        self.handle_events(&ctx);
        let held = self
            .world_ui
            .held_gesture(&ctx)
            .or(camera_gesture)
            .or(self.gizmo_gesture);
        if let Err(error) = crate::world_events::emit_action(
            &self.event_emitter(),
            &self.world,
            f64::from(self.world_ui.playhead),
            crate::world_events::WorldAction::FinishEdit { held },
        ) {
            self.status = error;
        }
        self.handle_events(&ctx);
        if let Err(error) = self.refresh_scene() {
            self.status = error;
        }
        self.save_settings();
        if let Some(error) = self.current().and_then(|t| t.colour_error.as_deref()) {
            self.status = format!("Colour output failed: {error}");
        }
        self.handle_snap(thumbs_pending);
        if self.animating() {
            ctx.request_repaint();
        }
    }
}

/// The viewport tool (Maya / Playa Q W E R): Select shows no manipulator and leaves the mouse
/// to navigation; Move, Rotate and Scale show the transform gizmo on the selected objects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewTool {
    Select,
    Move,
    Rotate,
    Scale,
}
impl ViewTool {
    pub const ALL: [Self; 4] = [Self::Select, Self::Move, Self::Rotate, Self::Scale];
    pub fn label(self) -> (&'static str, &'static str) {
        use soluwell_config::icons as ph;
        match self {
            Self::Select => (ph::SELECT, "Select · Q (no manipulator)"),
            Self::Move => (ph::MOVE, "Move · W · +/− gizmo size"),
            Self::Rotate => (ph::ROTATE, "Rotate · E · +/− gizmo size"),
            Self::Scale => (ph::SCALE, "Scale · R · +/− gizmo size"),
        }
    }
}

/// The largest axis scale of `matrix`: the radius factor of a sphere it transforms.
fn max_scale(matrix: &glam::Mat4) -> f32 {
    matrix
        .x_axis
        .truncate()
        .length()
        .max(matrix.y_axis.truncate().length())
        .max(matrix.z_axis.truncate().length())
}

/// The navigation clock step: what the flight and orbit rigs integrate with, and what takes
/// are timed by, so a played take moves at the speed it was flown.
fn nav_dt(ctx: &egui::Context) -> f32 {
    ctx.input(|i| i.stable_dt).clamp(1.0e-4, 0.1)
}

/// RMB held on the viewport: flight steers and owns the keyboard (`App::fly_camera`), so
/// the viewport's tool keys stand down until the button is released.
fn flight_held(ui: &egui::Ui, resp: &egui::Response) -> bool {
    resp.is_pointer_button_down_on() && ui.input(|i| i.focused && i.pointer.secondary_down())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn queued_camera_edit_uses_the_emission_frame_and_one_undo() {
        let mut app = App::new();
        let ctx = egui::Context::default();
        let camera = app.world.document.active_camera.unwrap();
        app.world_ui.auto_key = true;
        app.world_ui.playhead = 10;
        let before = serde_json::to_value(&app.world.document).unwrap();
        let fov = app
            .world
            .document
            .snapshot(10.0)
            .unwrap()
            .camera
            .fov_y_degrees
            + 5.0;
        app.emit_app(AppAction::CameraPatch(
            serde_json::json!({"fov_y_degrees": fov}),
        ));
        assert_eq!(serde_json::to_value(&app.world.document).unwrap(), before);
        app.world_ui.playhead = 20;
        app.handle_events(&ctx);
        let lens = app
            .world
            .document
            .attributes(camera, 10.0)
            .unwrap()
            .into_iter()
            .find(|attribute| attribute.path == "/camera/fov_y_degrees")
            .unwrap();
        assert!(lens.frames.contains(&10.0));
        assert!(!lens.frames.contains(&20.0));
        assert!(
            (app.world
                .document
                .snapshot(10.0)
                .unwrap()
                .camera
                .fov_y_degrees
                - fov)
                .abs()
                < 1e-4
        );
        assert_eq!(
            app.world_ui.playhead, 20,
            "dispatch must not seek the current playhead"
        );
        let edited = serde_json::to_value(&app.world.document).unwrap();
        app.emit_app(AppAction::History(History::Undo));
        assert_eq!(serde_json::to_value(&app.world.document).unwrap(), edited);
        app.handle_events(&ctx);
        assert_eq!(serde_json::to_value(&app.world.document).unwrap(), before);
    }

    #[test]
    fn gizmo_size_steps_stay_inside_the_settings_span() {
        assert_eq!(step_gizmo_size(120.0, true), 130.0);
        assert_eq!(step_gizmo_size(120.0, false), 110.0);
        assert_eq!(step_gizmo_size(GIZMO_SIZE_MIN, false), GIZMO_SIZE_MIN);
        assert_eq!(step_gizmo_size(GIZMO_SIZE_MAX, true), GIZMO_SIZE_MAX);
    }

    #[test]
    fn library_preset_assigns_to_all_fractals_in_one_undo_step() {
        let _gpu_test = crate::test_gpu::lock();
        let mut app = App::new();
        app.world
            .execute(crate::world::WorldCommand::Create {
                kind: crate::world::WorldKind::Fractal,
                name: "Second".into(),
                parent: None,
            })
            .unwrap();
        let fractals = app.world.document.material_consumers();
        assert!(fractals.len() >= 2, "a world holds several fractal nodes");
        let before: Vec<_> = fractals
            .iter()
            .map(|id| app.world.document.assigned_material(*id).unwrap())
            .collect();
        let materials = app.world.document.nodes().len();
        app.assign_library_material(0, AssignTo::AllFractals)
            .unwrap();
        let assigned: Vec<_> = fractals
            .iter()
            .map(|id| app.world.document.assigned_material(*id).unwrap())
            .collect();
        assert!(
            assigned.iter().all(|m| m.is_some() && *m == assigned[0]),
            "one new material on every fractal"
        );
        assert_eq!(app.world.document.nodes().len(), materials + 1);
        assert!(app.world.undo());
        let undone: Vec<_> = fractals
            .iter()
            .map(|id| app.world.document.assigned_material(*id).unwrap())
            .collect();
        assert_eq!(
            undone, before,
            "one undo removes the material and every assignment"
        );
        assert_eq!(app.world.document.nodes().len(), materials);
    }

    #[test]
    fn library_is_read_only_until_explicit_creation_and_never_assigns_objects() {
        let _gpu_test = crate::test_gpu::lock();
        let mut app = App::new();
        let before = serde_json::to_string(&app.world.document).unwrap();
        let object = app.world.active().unwrap();
        let assignment = app.world.document.assigned_material(object).unwrap();
        let ctx = egui::Context::default();
        for _ in 0..3 {
            let _ = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(850.0, 600.0),
                    )),
                    ..Default::default()
                },
                |root| app.material_library(root),
            );
        }
        assert_eq!(serde_json::to_string(&app.world.document).unwrap(), before);
        let preset = &crate::materials::PRESETS[0];
        let mut material = Material::default();
        preset.apply(&mut material);
        app.emit_app(AppAction::CreateMaterial {
            material,
            name: preset.name().into(),
            targets: None,
            open: true,
        });
        assert_eq!(serde_json::to_string(&app.world.document).unwrap(), before);
        app.handle_events(&ctx);
        let material = app.world.active().unwrap();
        assert_ne!(material, object);
        assert_eq!(
            app.world.document.assigned_material(object).unwrap(),
            assignment
        );
        assert!(
            app.world
                .document
                .nodes()
                .iter()
                .any(|node| node.id == material && node.kind == crate::world::WorldKind::Material)
        );
        assert!(app.world.undo());
        assert_eq!(serde_json::to_string(&app.world.document).unwrap(), before);
    }

    #[test]
    fn material_library_adapts_columns_to_available_panel_width() {
        let _gpu_test = crate::test_gpu::lock();
        fn labels(shape: &egui::epaint::Shape, found: &mut Vec<(String, egui::Pos2)>) {
            match shape {
                egui::epaint::Shape::Text(text) => {
                    found.push((text.galley.text().to_owned(), text.pos))
                }
                egui::epaint::Shape::Vec(shapes) => {
                    for shape in shapes {
                        labels(shape, found);
                    }
                }
                _ => {}
            }
        }
        for (width, columns) in [(145.0, 1), (245.0, 2), (545.0, 5)] {
            let mut app = App::new();
            let previous: Vec<_> = app
                .world
                .document
                .nodes()
                .into_iter()
                .filter(|node| node.kind == crate::world::WorldKind::Material)
                .map(|node| node.id)
                .collect();
            for id in previous {
                app.world
                    .execute(crate::world::WorldCommand::Delete(id))
                    .unwrap();
            }
            for name in [
                "Metal",
                "MetalChrome",
                "MetalGold",
                "MetalCopper",
                "MetalBrass",
            ] {
                app.world
                    .execute(crate::world::WorldCommand::CreateMaterial {
                        material: Material::default(),
                        name: name.into(),
                    })
                    .unwrap();
            }
            app.world.clear_selection();
            let ctx = egui::Context::default();
            let mut found = Vec::new();
            for _ in 0..3 {
                let output = ctx.run_ui(
                    egui::RawInput {
                        screen_rect: Some(egui::Rect::from_min_size(
                            egui::Pos2::ZERO,
                            egui::vec2(width, 900.0),
                        )),
                        ..Default::default()
                    },
                    |ui| app.materials_tab(ui),
                );
                found.clear();
                for shape in &output.shapes {
                    labels(&shape.shape, &mut found);
                }
            }
            let positions: Vec<_> = ["Metal", "MetalChrome", "MetalGold"]
                .iter()
                .map(|name| found.iter().find(|(text, _)| text == name).unwrap().1)
                .collect();
            assert!(positions[0].x < width);
            if columns == 1 {
                assert!(
                    positions[1].y > positions[0].y,
                    "width {width}, positions {positions:?}"
                );
            } else {
                assert!((positions[1].y - positions[0].y).abs() < 1.0);
                assert!(positions[1].x > positions[0].x);
            }
            if columns == 2 {
                assert!(positions[2].y > positions[0].y);
            } else if columns > 2 {
                assert!((positions[2].y - positions[0].y).abs() < 1.0);
            }
        }
    }

    #[test]
    fn scene_gallery_cards_reflow_to_multiple_columns() {
        let _gpu_test = crate::test_gpu::lock();
        for (width, columns) in [(200.0, 1), (600.0, 3)] {
            let mut app = App::new();
            app.gallery.truncate(4);
            for (index, entry) in app.gallery.iter_mut().enumerate() {
                entry.scene.name = format!("Gallery-{index}");
            }
            let ctx = egui::Context::default();
            let mut positions = Vec::new();
            for _ in 0..3 {
                let output = ctx.run_ui(
                    egui::RawInput {
                        screen_rect: Some(egui::Rect::from_min_size(
                            egui::Pos2::ZERO,
                            egui::vec2(width, 900.0),
                        )),
                        ..Default::default()
                    },
                    |root| app.browser(root, dock::Panel::Gallery),
                );
                positions = output
                    .shapes
                    .iter()
                    .filter_map(|shape| match &shape.shape {
                        egui::epaint::Shape::Text(text)
                            if text.galley.text().starts_with("Gallery-") =>
                        {
                            Some((text.galley.text().to_owned(), text.pos))
                        }
                        _ => None,
                    })
                    .collect();
            }
            positions.sort_by(|a, b| a.0.cmp(&b.0));
            assert_eq!(positions.len(), 4);
            if columns == 1 {
                assert!(positions[1].1.y > positions[0].1.y);
            } else {
                assert!((positions[2].1.y - positions[0].1.y).abs() < 1.0);
                assert!(positions[2].1.x > positions[1].1.x);
                assert!(positions[3].1.y > positions[0].1.y);
            }
        }
    }

    #[test]
    fn material_card_selects_existing_attribute_editor_without_assignment() {
        let _gpu_test = crate::test_gpu::lock();
        let mut app = App::new();
        let object = app.world.active().unwrap();
        let assignment = app.world.document.assigned_material(object).unwrap();
        app.world
            .execute(crate::world::WorldCommand::CreateMaterial {
                material: Material::default(),
                name: "Gallery test material".into(),
            })
            .unwrap();
        let material = app.world.active().unwrap();
        app.world.select_only(object);
        app.panels_to_open.clear();
        let before = serde_json::to_value(&app.world.document).unwrap();
        let ctx = egui::Context::default();
        let mut point = egui::Pos2::ZERO;
        let mut frame = |app: &mut App, events| {
            let output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(420.0, 400.0),
                    )),
                    events,
                    ..Default::default()
                },
                |ui| app.materials_tab(ui),
            );
            for shape in output.shapes {
                if let egui::epaint::Shape::Text(text) = shape.shape
                    && text.galley.text() == "Gallery test material"
                {
                    point = text.pos + egui::vec2(20.0, -40.0);
                }
            }
            point
        };
        frame(&mut app, vec![]);
        let point = frame(&mut app, vec![]);
        frame(&mut app, vec![egui::Event::PointerMoved(point)]);
        for pressed in [true, false] {
            frame(
                &mut app,
                vec![egui::Event::PointerButton {
                    pos: point,
                    button: egui::PointerButton::Primary,
                    pressed,
                    modifiers: egui::Modifiers::NONE,
                }],
            );
        }
        assert_ne!(app.world.active(), Some(material));
        assert!(!app.panels_to_open.contains(&dock::Panel::Inspector));
        app.handle_events(&ctx);
        assert_eq!(app.world.active(), Some(material));
        assert_eq!(app.world.selected(), [material]);
        assert!(app.panels_to_open.contains(&dock::Panel::Inspector));
        assert_eq!(
            app.world.document.assigned_material(object).unwrap(),
            assignment
        );
        assert_eq!(serde_json::to_value(&app.world.document).unwrap(), before);
    }

    #[test]
    fn material_editor_apply_uses_remembered_consumers_and_one_undo() {
        let _gpu_test = crate::test_gpu::lock();
        use crate::world::{WorldCommand, WorldKind};
        let mut app = App::new();
        let object = app.world.active().unwrap();
        let original = app.world.document.assigned_material(object).unwrap();
        app.remember_material_targets();
        app.world
            .execute(WorldCommand::CreateMaterial {
                material: Material::default(),
                name: "Apply test".into(),
            })
            .unwrap();
        let material = app.world.active().unwrap();
        app.select_material_node(material);
        app.remember_material_targets();
        assert_eq!(app.material_targets, [object]);
        assert!(app.world.document.supports_material(object));
        assert!(!app.world.document.supports_material(material));
        app.assign_gallery_material(material, AssignTo::Selected)
            .unwrap();
        assert_eq!(
            app.world.active(),
            Some(material),
            "Apply must retain the material in the editor"
        );
        assert_eq!(
            app.world.document.assigned_material(object).unwrap(),
            Some(material)
        );
        app.world.undo();
        assert_eq!(
            app.world.document.assigned_material(object).unwrap(),
            original
        );
        let camera = app
            .world
            .document
            .nodes()
            .iter()
            .find(|n| n.kind == WorldKind::Camera)
            .unwrap()
            .id;
        app.world.select_only(camera);
        app.remember_material_targets();
        assert!(app.material_targets.is_empty());
        assert!(
            app.assign_gallery_material(material, AssignTo::Selected)
                .is_err()
        );
    }

    #[test]
    fn reopening_same_document_rebuilds_selected_profile_scene() {
        let _gpu_test = crate::test_gpu::lock();
        let mut app = App::new();
        let viewport = app.world.document.viewport_settings_id().unwrap();
        app.world
            .execute(crate::world::WorldCommand::SetAttribute {
                id: viewport,
                path: "/viewport/mode".into(),
                value: serde_json::json!("Locked"),
                frame: 0.0,
            })
            .unwrap();
        let mut initial = app.scene.clone();
        initial.document = Some(Box::new(app.world.document.clone()));
        app.load(initial, Load::Same);
        app.last_change = Instant::now() - std::time::Duration::from_secs(2);
        app.step_viewport(1, 1, false, 100.0);
        let previous = app.request.as_ref().unwrap().scene.objects.clone();
        let original_id = app.world.document.graph.id;
        assert_eq!(app.world.revision(), 0);
        let mut edited = crate::world::WorldEditor::new(app.world.document.clone());
        let fractal = edited
            .document
            .nodes()
            .into_iter()
            .find(|node| node.kind == crate::world::WorldKind::Fractal)
            .unwrap()
            .id;
        edited
            .execute(crate::world::WorldCommand::SetAttribute {
                id: fractal,
                path: "/transform/position/0".into(),
                value: serde_json::json!(3.0),
                frame: 0.0,
            })
            .unwrap();
        let manual = edited.document.viewport_policy(0.0).unwrap().manual_id;
        let expected = edited
            .document
            .snapshot_with_render_profile(manual, 0.0, crate::world::View::Viewport)
            .unwrap()
            .objects;
        assert_ne!(expected, previous);
        let mut reopened = app.scene.clone();
        reopened.document = Some(Box::new(edited.document));
        app.load(reopened, Load::Same);
        assert_eq!(app.world.document.graph.id, original_id);
        assert_eq!(app.world.revision(), 0);
        app.last_change = Instant::now() - std::time::Duration::from_secs(2);
        app.step_viewport(1, 1, false, 100.0);
        assert_eq!(app.request.as_ref().unwrap().scene.objects, expected);
    }

    #[test]
    fn node_profiles_route_runtime_and_cache_without_mutating_output() {
        let _gpu_test = crate::test_gpu::lock();
        use crate::world::WorldCommand;
        use serde_json::json;
        let mut app = App::new();
        let policy = app.world.document.viewport_policy(0.0).unwrap();
        app.world
            .execute(WorldCommand::CreateRenderProfile {
                name: "Output".into(),
                role: crate::render_profiles::CatalogRole::Profile,
                source: Some(policy.still_id),
                target: Some(crate::render_profiles::ProfileTarget::Output),
            })
            .unwrap();
        let output = app.world.document.output_render_profile().unwrap();
        for (id, path, value) in [
            (policy.moving_id, "/quality/samples", json!(19)),
            (policy.moving_id, "/quality/max_bounces", json!(1)),
            (policy.moving_id, "/quality/resolution_scale", json!(0.25)),
            (policy.still_id, "/quality/samples", json!(41)),
            (policy.still_id, "/quality/max_bounces", json!(6)),
            (policy.still_id, "/quality/resolution_scale", json!(0.75)),
            (output, "/quality/samples", json!(997)),
        ] {
            app.world
                .execute(WorldCommand::SetAttribute {
                    id,
                    path: path.into(),
                    value,
                    frame: 0.0,
                })
                .unwrap();
        }
        let authored = app.world.document.clone();
        app.last_change = Instant::now();
        app.step_viewport(128, 64, false, 100.0);
        let moving = app.request.as_ref().unwrap();
        assert_eq!(
            (
                moving.width,
                moving.height,
                moving.target_spp,
                moving.scene.render.max_bounces
            ),
            (32, 16, 19, 1)
        );
        assert!(moving.preview);
        assert_eq!(app.world.document, authored);
        let ctx = egui::Context::default();
        app.viewport_rect = Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(128.0, 64.0),
        ));
        let key = app.preview_identity(&ctx).unwrap();
        let cached = app.make_preview_request(0, 10, key.samples, &ctx).unwrap();
        assert_eq!((cached.width, cached.height, cached.spp), (96, 48, 41));
        assert_eq!(cached.render_profile, Some(policy.still_id));
        app.last_change = Instant::now() - std::time::Duration::from_secs(2);
        app.step_viewport(128, 64, false, 100.0);
        let still = app.request.as_ref().unwrap();
        assert_eq!(
            (
                still.width,
                still.height,
                still.target_spp,
                still.scene.render.max_bounces
            ),
            (96, 48, 41, 6)
        );
        assert!(!still.preview);
        assert_eq!(app.world.document, authored);
        let generation = app.generation;
        app.world
            .execute(WorldCommand::Rename {
                id: policy.still_id,
                name: "Renamed still".into(),
            })
            .unwrap();
        app.step_viewport(128, 64, false, 100.0);
        assert_eq!(
            app.generation, generation,
            "A catalog name does not change effective film content"
        );
        let viewport = app.world.document.viewport_settings_id().unwrap();
        app.world
            .execute(WorldCommand::Batch(vec![
                WorldCommand::SetAttribute {
                    id: viewport,
                    path: "/viewport/mode".into(),
                    value: json!("Locked"),
                    frame: 0.0,
                },
                WorldCommand::SetAttribute {
                    id: viewport,
                    path: "/viewport/manual_id".into(),
                    value: json!(policy.moving_id),
                    frame: 0.0,
                },
            ]))
            .unwrap();
        app.step_viewport(128, 64, false, 100.0);
        assert_eq!(app.request.as_ref().unwrap().target_spp, 19);
        assert!(!app.request.as_ref().unwrap().preview);
        assert_eq!(
            app.world
                .document
                .effective_render(output, 0.0)
                .unwrap()
                .samples,
            997
        );
        app.set_viewport_flag("/viewport/frozen", true);
        let frozen = app.request.clone().unwrap();
        app.scene.camera.yaw_degrees += 20.0;
        app.step_viewport(512, 256, false, 100.0);
        let request = app.request.as_ref().unwrap();
        assert_eq!(
            (request.width, request.height, &request.scene),
            (frozen.width, frozen.height, &frozen.scene)
        );
        assert!(request.paused);
    }

    /// Navigation (flight or orbit) writes the camera pose before `step_viewport`; that change
    /// alone must route Auto to Moving, back to Still after the settle delay, and Locked to Manual.
    #[test]
    fn camera_motion_routes_moving_until_settled() {
        let _gpu_test = crate::test_gpu::lock();
        use crate::render_profiles::ProfileTarget;
        use crate::world::WorldCommand;
        use serde_json::json;
        let mut app = App::new();
        let viewport = app.world.document.viewport_settings_id().unwrap();
        app.world
            .execute(WorldCommand::SetAttribute {
                id: viewport,
                path: "/viewport/settle_delay_ms".into(),
                value: json!(40.0),
                frame: 0.0,
            })
            .unwrap();
        let policy = app.world.document.viewport_policy(0.0).unwrap();
        let settle = || std::thread::sleep(std::time::Duration::from_millis(120));
        let routed = |app: &App| {
            let selection = app.viewport_render.as_ref().unwrap();
            let request = app.request.as_ref().unwrap();
            (
                selection.target,
                selection.effective.profile,
                request.interactive,
                request.preview,
            )
        };
        app.step_viewport(64, 32, false, 100.0);
        settle();
        app.step_viewport(64, 32, false, 100.0);
        assert_eq!(
            routed(&app),
            (ProfileTarget::Still, policy.still_id, false, false),
            "an unchanged scene settles"
        );
        app.scene.camera.yaw_degrees += 5.0;
        app.step_viewport(64, 32, false, 100.0);
        assert_eq!(
            routed(&app),
            (ProfileTarget::Moving, policy.moving_id, true, true)
        );
        let moving = app.request.clone().unwrap();
        assert_eq!(
            (
                moving.width,
                moving.height,
                moving.target_spp,
                moving.scene.render.adaptive.enabled
            ),
            (64, 32, 8, false)
        );
        app.step_viewport(64, 32, false, 100.0);
        assert_eq!(
            routed(&app).0,
            ProfileTarget::Moving,
            "a still frame inside the settle delay keeps the moving proxy"
        );
        settle();
        app.step_viewport(64, 32, false, 100.0);
        assert_eq!(
            routed(&app),
            (ProfileTarget::Still, policy.still_id, false, false)
        );
        app.world
            .execute(WorldCommand::SetAttribute {
                id: viewport,
                path: "/viewport/mode".into(),
                value: json!("Locked"),
                frame: 0.0,
            })
            .unwrap();
        app.scene.camera.yaw_degrees += 5.0;
        app.step_viewport(64, 32, false, 100.0);
        assert_eq!(
            routed(&app),
            (ProfileTarget::Manual, policy.manual_id, true, false),
            "Locked motion keeps Manual's film instead of a proxy"
        );
    }

    #[test]
    fn viewport_cache_accepts_only_final_current_authoring_frames() {
        let _gpu_test = crate::test_gpu::lock();
        let ctx = egui::Context::default();
        let mut app = App::new();
        let still = app.world.document.viewport_policy(0.0).unwrap().still_id;
        app.world
            .execute(crate::world::WorldCommand::SetAttribute {
                id: still,
                path: "/quality/samples".into(),
                value: serde_json::json!(8),
                frame: 0.0,
            })
            .unwrap();
        app.last_change = Instant::now() - std::time::Duration::from_secs(2);
        app.step_viewport(1, 1, false, 100.0);
        let make_frame = |samples, preview| {
            Arc::new(Frame {
                generation: app.generation,
                preview,
                width: 1,
                height: 1,
                pixels: vec![0xff112233],
                light: vec![[0.1, 0.2, 0.3, 1.0]],
                radiance: vec![],
                light_kind: crate::color::DisplayLight::Relative,
                colour_error: None,
                denoised_samples: 0,
                denoise_ms: 0.0,
                denoise_error: None,
                samples,
                active_tiles: (0, 0),
                converged: false,
                last_ms: 0.0,
                last_spp: 1,
                unresolved: 0.0,
                sdr_bytes: Arc::new(vec![]),
                hdr_bytes: Arc::new(vec![]),
            })
        };
        let partial = make_frame(7, false);
        let proxy = make_frame(8, true);
        let complete = make_frame(8, false);
        app.cache_completed_viewport(&partial, &ctx);
        assert!(app.preview.position().is_none());
        app.cache_completed_viewport(&proxy, &ctx);
        assert!(app.preview.position().is_none());
        app.world_ui.seek(1);
        app.cache_completed_viewport(&complete, &ctx);
        assert!(
            app.preview.position().is_none(),
            "late frame must not be relabelled after scrubbing"
        );
        app.world_ui.seek(0);
        app.cache_completed_viewport(&complete, &ctx);
        assert_eq!(app.preview.position(), Some(0));
        assert!(
            !app.preview.displaying_preview(),
            "adoption must not start transport"
        );
        assert_eq!(app.preview.cached_spp(), Some(8));
        let key = app.preview_key;
        app.world
            .execute(crate::world::WorldCommand::SetAttribute {
                id: still,
                path: "/quality/samples".into(),
                value: serde_json::json!(16),
                frame: 0.0,
            })
            .unwrap();
        assert_ne!(key, Some(app.preview_identity(&ctx).unwrap()));
    }

    #[test]
    fn gallery_assignment_and_object_material_field_share_uuid_and_undo() {
        let _gpu_test = crate::test_gpu::lock();
        use crate::world::WorldCommand;
        let mut app = App::new();
        let object = app.world.active().unwrap();
        app.world
            .execute(WorldCommand::CreateMaterial {
                material: Material::default(),
                name: "Shared gallery material".into(),
            })
            .unwrap();
        let material = app.world.active().unwrap();
        let count = app.world.document.nodes().len();
        app.world.select_only(object);
        let old = app.world.document.assigned_material(object).unwrap();
        app.assign_gallery_material(material, AssignTo::Selected)
            .unwrap();
        assert_eq!(
            app.world.document.assigned_material(object).unwrap(),
            Some(material)
        );
        assert_eq!(app.world.active(), Some(object));
        assert_eq!(app.world.document.nodes().len(), count);
        assert!(app.world.undo());
        assert_eq!(app.world.document.assigned_material(object).unwrap(), old);
        app.world
            .execute(WorldCommand::SetAttribute {
                id: object,
                path: "/material_id".into(),
                value: serde_json::json!(material.0.to_string()),
                frame: 0.0,
            })
            .unwrap();
        assert_eq!(
            app.world.document.assigned_material(object).unwrap(),
            Some(material)
        );
        assert_eq!(app.world.document.nodes().len(), count);
    }

    #[test]
    fn auto_key_preference_serializes_once_and_defaults_off() {
        let _gpu_test = crate::test_gpu::lock();
        let mut app = App::new();
        assert!(!Settings::default().auto_key);
        assert!(serde_json::from_str::<Settings>("{}").is_err());
        app.changed_settings_json().unwrap();
        app.world_ui.auto_key = true;
        let json = app.changed_settings_json().unwrap().unwrap();
        assert!(serde_json::from_str::<Settings>(&json).unwrap().auto_key);
        for _ in 0..16 {
            assert!(app.changed_settings_json().unwrap().is_none());
        }
    }

    #[test]
    fn cached_preview_skips_world_evaluation_until_pause() {
        let _gpu_test = crate::test_gpu::lock();
        let mut app = App::new();
        let ctx = egui::Context::default();
        app.refresh_scene().unwrap();
        let key = app.evaluated_world;
        let objects = app.scene.objects.as_ptr();
        let snapshots = app.snapshot_clones;
        assert!(app.request.is_none());
        // A preview renders at the viewport's size: without one it is refused, not guessed.
        app.begin_preview(0, 10, crate::preview::PreviewMode::Play, &ctx);
        assert!(app.preview.position().is_none());
        assert!(!app.status.is_empty());
        app.viewport_rect = Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(640.0, 360.0),
        ));
        app.begin_preview(0, 10, crate::preview::PreviewMode::Play, &ctx);
        for frame in 1..=10 {
            app.world_ui.seek(frame);
            app.refresh_scene().unwrap();
            assert_eq!(app.evaluated_world, key);
            assert_eq!(app.scene.objects.as_ptr(), objects);
            assert_eq!(app.snapshot_clones, snapshots);
        }
        app.preview.pause();
        app.world_ui.playing = false;
        app.refresh_scene().unwrap();
        assert_ne!(app.evaluated_world, key);
    }

    #[test]
    fn file_load_completion_preserves_document_and_discards_stale_requests() {
        let _gpu_test = crate::test_gpu::lock();
        let mut app = App::new();
        let mut scene = Scene::preset(Formula::Mandelbox(MandelBox::PRESET));
        let document = crate::world::WorldDocument::from_scene(&scene);
        let expected = serde_json::to_value(&document).unwrap();
        scene.document = Some(Box::new(document));
        app.scene_file_pending = Some((7, app.load_revision, crate::templates::FileAction::Open));
        app.scene_file_event(crate::io_service::SceneEvent {
            id: 7,
            path: PathBuf::from("opened.frac.json"),
            result: Ok(Some(Box::new(scene))),
        });
        assert_eq!(serde_json::to_value(&app.world.document).unwrap(), expected);
        assert_eq!(
            app.scene_file_path.as_deref(),
            Some(Path::new("opened.frac.json"))
        );
        app.refresh_scene().unwrap();
        let revision = app.world.revision();
        let clones = app.snapshot_clones;
        app.refresh_scene().unwrap();
        assert_eq!(app.world.revision(), revision);
        assert_eq!(app.snapshot_clones, clones);
        app.scene_file_pending = Some((8, app.load_revision, crate::templates::FileAction::Open));
        app.load(Scene::preset(Formula::Mandelbulb(Bulb::PRESET)), Load::New);
        app.scene_file_event(crate::io_service::SceneEvent {
            id: 8,
            path: PathBuf::from("stale.frac.json"),
            result: Ok(Some(Box::new(Scene::preset(Formula::Mandelbox(
                MandelBox::PRESET,
            ))))),
        });
        assert!(matches!(
            app.scene.formula,
            crate::scene::Formula::Mandelbulb(_)
        ));
        assert!(app.scene_file_path.is_none());
    }

    #[test]
    fn export_panel_does_not_freeze_scene_during_idle_or_playback() {
        let _gpu_test = crate::test_gpu::lock();
        let mut app = App::new();
        let ctx = egui::Context::default();
        for frame in 0..32 {
            let _ = ctx.run_ui(egui::RawInput::default(), |root| {
                egui::CentralPanel::default().show(root, |ui| {
                    app.export.ui(
                        ui,
                        (frame, frame + 250, 24.0, frame),
                        Err("No OCIO config in this test".into()),
                        &crate::color::default_selection(),
                        |_| {},
                    );
                });
            });
        }
    }

    #[test]
    fn settings_and_layouts_serialize_only_after_actual_changes() {
        let _gpu_test = crate::test_gpu::lock();
        let mut app = App::new();
        app.refresh_scene().unwrap();
        assert!(app.changed_settings_json().unwrap().is_some());
        let serializations = app.settings_serializations;
        let dock_serializations = app.layouts.cache.serializations;
        let baseline = app.snapshot_before.as_ref().unwrap().objects.as_ptr();
        let snapshots = app.snapshot_clones;
        for _ in 0..32 {
            app.refresh_scene().unwrap();
            assert!(app.changed_settings_json().unwrap().is_none());
            assert_eq!(
                app.snapshot_before.as_ref().unwrap().objects.as_ptr(),
                baseline
            );
        }
        assert_eq!(app.settings_serializations, serializations);
        assert_eq!(app.layouts.cache.serializations, dock_serializations);
        assert_eq!(app.snapshot_clones, snapshots);
        app.world_ui.attribute_metrics.numeric_width = 64.0;
        app.fonts.face = "Custom face".into();
        app.status_layout.widths = vec![210.0, 100.0];
        app.gui_fps = 144;
        let mut export = app.export.job.clone();
        export.name = "new".into();
        export.last = 19;
        app.export.restore(export);
        let json = app.changed_settings_json().unwrap().unwrap();
        assert!(
            !json.contains("png_peak_nits"),
            "the encode recipe lives in the document, not in preferences"
        );
        let saved: Settings = serde_json::from_str(&json).unwrap();
        assert_eq!(saved.attribute_metrics.numeric_width, 64.0);
        assert_eq!(saved.fonts.face, "Custom face");
        assert_eq!(saved.status_layout.widths, [210.0, 100.0]);
        assert_eq!(saved.gui_fps, 144);
        assert_eq!(saved.export.name, "new");
        assert_eq!(saved.export.last, 19);
        assert!(app.changed_settings_json().unwrap().is_none());
        let current = app.colour.sel.clone();
        assert!(app.colour.presets.store(1, "Custom colour", &current));
        let json = app.changed_settings_json().unwrap().unwrap();
        let saved: Settings = serde_json::from_str(&json).unwrap();
        assert_eq!(saved.colour_presets, app.colour.presets);
        assert_eq!(saved.colour_presets.slots[1].name, "Custom colour");
        assert!(app.changed_settings_json().unwrap().is_none());
        assert!(serde_json::from_str::<Settings>("{}").is_err());
        app.layouts
            .store
            .save(
                "Edited workspace",
                app.layouts.cache.blob.as_ref().unwrap().clone(),
            )
            .unwrap();
        let json = app.changed_settings_json().unwrap().unwrap();
        let saved: Settings = serde_json::from_str(&json).unwrap();
        assert_eq!(saved.layouts.current(), "Edited workspace");
        assert!(app.changed_settings_json().unwrap().is_none());
    }

    #[test]
    fn cached_scene_reuses_idle_vectors_and_invalidates_edit_seek_undo_and_load() {
        let _gpu_test = crate::test_gpu::lock();
        let mut app = App::new();
        app.refresh_scene().unwrap();
        let objects = app.scene.objects.as_ptr();
        let key = app.evaluated_world;
        for _ in 0..32 {
            app.refresh_scene().unwrap();
            assert_eq!(app.scene.objects.as_ptr(), objects);
            assert_eq!(app.evaluated_world, key);
        }
        let camera = app.world.document.active_camera.unwrap();
        let old_fov = app.scene.camera.fov_y_degrees;
        app.world
            .execute(crate::world::WorldCommand::SetAttribute {
                id: camera,
                path: "/camera/fov_y_degrees".into(),
                value: serde_json::json!(75.0),
                frame: 0.0,
            })
            .unwrap();
        app.refresh_scene().unwrap();
        assert_eq!(app.scene.camera.fov_y_degrees, 75.0);
        assert_ne!(app.evaluated_world, key);
        assert!(app.world.undo());
        app.refresh_scene().unwrap();
        assert_eq!(app.scene.camera.fov_y_degrees, old_fov);
        assert!(app.world.redo());
        app.world
            .execute(crate::world::WorldCommand::Key {
                id: camera,
                path: "/camera/fov_y_degrees".into(),
                frame: 0.0,
            })
            .unwrap();
        app.world
            .execute(crate::world::WorldCommand::SetAttribute {
                id: camera,
                path: "/camera/fov_y_degrees".into(),
                value: serde_json::json!(100.0),
                frame: 10.0,
            })
            .unwrap();
        app.world_ui.seek(5);
        app.refresh_scene().unwrap();
        assert!((app.scene.camera.fov_y_degrees - 87.5).abs() < 1e-5);
        let revision = app.load_revision;
        let mut reload = app.scene.clone();
        reload.document = Some(Box::new(app.world.document.clone()));
        app.load(reload, Load::Same);
        assert!(app.evaluated_world.is_none());
        assert_ne!(app.load_revision, revision);
        app.refresh_scene().unwrap();
    }

    #[test]
    fn status_text_never_changes_workspace_height() {
        let _gpu_test = crate::test_gpu::lock();
        let mut app = App::new();
        let ctx = egui::Context::default();
        for width in [320.0, 800.0, 1600.0] {
            let mut heights = Vec::new();
            for status in [
                String::new(),
                "Saved".into(),
                "A long render status with changing counters ".repeat(30),
            ] {
                app.status = status;
                let mut height = 0.0;
                let _ = ctx.run_ui(
                    egui::RawInput {
                        screen_rect: Some(egui::Rect::from_min_size(
                            egui::Pos2::ZERO,
                            egui::vec2(width, 200.0),
                        )),
                        ..Default::default()
                    },
                    |ui| {
                        height = ui.scope(|ui| app.status_bar(ui)).response.rect.height();
                    },
                );
                heights.push(height);
            }
            assert!(
                heights.iter().all(|h| (h - heights[0]).abs() <= 1.0),
                "{width}: {heights:?}"
            );
        }
    }

    #[test]
    fn settings_pages_are_one_list_in_tab_order() {
        let _gpu_test = crate::test_gpu::lock();
        for (index, page) in SettingsPage::ALL.into_iter().enumerate() {
            assert_eq!(page as usize, index, "the tab index is the discriminant");
            assert_eq!(SettingsPage::named(page.category().label), Some(page));
        }
        assert_eq!(SettingsPage::named("COLOR"), Some(SettingsPage::Color));
        assert_eq!(SettingsPage::named("nope"), None);
    }
    #[test]
    fn orbit_tumbles_coasts_to_rest_and_pans_with_the_cursor_under_roll() {
        let _gpu_test = crate::test_gpu::lock();
        let mut app = App::new();
        let ctx = egui::Context::default();
        let mut time = 0.0;
        let mut frame = |app: &mut App, events| {
            time += 1.0 / 60.0;
            ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(640.0, 400.0),
                    )),
                    time: Some(time),
                    events,
                    ..Default::default()
                },
                |root| {
                    egui::CentralPanel::default().show(root, |ui| app.viewport(ui));
                },
            )
        };
        let button = |button, pressed, pos| egui::Event::PointerButton {
            pos,
            button,
            pressed,
            modifiers: Default::default(),
        };
        let mut pos = egui::pos2(300.0, 200.0);
        frame(&mut app, vec![egui::Event::PointerMoved(pos)]);
        frame(
            &mut app,
            vec![button(egui::PointerButton::Primary, true, pos)],
        );
        let yaw = app.scene.camera.yaw_degrees;
        for _ in 0..6 {
            pos.x += 10.0;
            frame(&mut app, vec![egui::Event::PointerMoved(pos)]);
        }
        let dragged = app.scene.camera.yaw_degrees;
        assert!(
            dragged < yaw - 1.0,
            "dragging right turns the view: {yaw} -> {dragged}"
        );
        frame(
            &mut app,
            vec![button(egui::PointerButton::Primary, false, pos)],
        );
        let released = app.scene.camera.yaw_degrees;
        frame(&mut app, Vec::new());
        assert!(
            app.scene.camera.yaw_degrees < released,
            "a released drag coasts on"
        );
        for _ in 0..600 {
            frame(&mut app, Vec::new());
        }
        assert!(!app.orbit.rig.has_inertia(), "the coast decays to rest");
        let rest = app.scene.camera;
        frame(&mut app, Vec::new());
        assert_eq!(
            app.scene.camera, rest,
            "a resting orbit leaves the camera alone"
        );

        // MMB pan with the camera rolled 90 degrees: the target moves against the
        // cursor along the rolled screen axis, the roll itself stays.
        app.scene.camera.roll_degrees = 90.0;
        app.world
            .navigate_camera(&app.scene, 0.0, false, None)
            .unwrap();
        let orientation = app.scene.camera.orientation();
        let (right, up) = (orientation * glam::Vec3::X, orientation * glam::Vec3::Y);
        let target = glam::Vec3::from_array(app.scene.camera.target);
        frame(
            &mut app,
            vec![button(egui::PointerButton::Middle, true, pos)],
        );
        for _ in 0..6 {
            pos.x += 10.0;
            frame(&mut app, vec![egui::Event::PointerMoved(pos)]);
        }
        frame(
            &mut app,
            vec![button(egui::PointerButton::Middle, false, pos)],
        );
        let moved = glam::Vec3::from_array(app.scene.camera.target) - target;
        assert!(
            moved.dot(right) < 0.0,
            "pan follows the rolled cursor axis: {moved}"
        );
        assert!(
            moved.dot(up).abs() < 1e-3 * moved.length(),
            "no drift across the rolled axis: {moved}"
        );
        assert_eq!(app.scene.camera.roll_degrees, 90.0);

        // Shift + LMB keeps the nearest world plane while allowing continuous yaw.
        app.scene.camera.roll_degrees = 0.0;
        app.scene.camera.yaw_degrees = 20.0;
        app.scene.camera.pitch_degrees = 10.0;
        app.world
            .navigate_camera(&app.scene, 0.0, false, None)
            .unwrap();
        frame(
            &mut app,
            vec![
                egui::Event::ModifiersChanged(egui::Modifiers::SHIFT),
                egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::SHIFT,
                },
            ],
        );
        for _ in 0..4 {
            pos.x += 2.0;
            frame(&mut app, vec![egui::Event::PointerMoved(pos)]);
        }
        assert!(app.scene.camera.pitch_degrees.abs() < 1e-3);
        assert!(
            app.scene.camera.yaw_degrees.abs() > 1.0,
            "plane orbit must not snap yaw to an axis"
        );
        frame(
            &mut app,
            vec![
                button(egui::PointerButton::Primary, false, pos),
                egui::Event::ModifiersChanged(egui::Modifiers::NONE),
            ],
        );
        assert!(!app.orbit.rig.has_inertia());

        // Ctrl + Shift + LMB: the view snaps parallel to the nearest world axis; dragging up far
        // enough gives the top view, square to the world, which a plain orbit then continues
        // from without losing its heading.
        app.scene.camera.roll_degrees = 0.0;
        app.scene.camera.yaw_degrees = 20.0;
        app.scene.camera.pitch_degrees = 10.0;
        app.world
            .navigate_camera(&app.scene, 0.0, false, None)
            .unwrap();
        let shift = egui::Modifiers::CTRL.plus(egui::Modifiers::SHIFT);
        let shifted = |button, pressed, pos| egui::Event::PointerButton {
            pos,
            button,
            pressed,
            modifiers: shift,
        };
        frame(
            &mut app,
            vec![
                egui::Event::ModifiersChanged(shift),
                shifted(egui::PointerButton::Primary, true, pos),
            ],
        );
        for _ in 0..4 {
            pos.x += 2.0;
            frame(&mut app, vec![egui::Event::PointerMoved(pos)]);
        }
        let cam = app.scene.camera;
        assert_eq!(
            (cam.yaw_degrees.round(), cam.pitch_degrees.round()),
            (0.0, 0.0),
            "snapped to -Z"
        );
        // 1 degree per point keeps the 120 degree drag inside the window.
        app.controls.look_sensitivity = 10.0;
        for _ in 0..12 {
            pos.y += 10.0;
            frame(&mut app, vec![egui::Event::PointerMoved(pos)]);
        }
        frame(
            &mut app,
            vec![
                shifted(egui::PointerButton::Primary, false, pos),
                egui::Event::ModifiersChanged(egui::Modifiers::NONE),
            ],
        );
        let top = app.scene.camera;
        assert!(
            (top.pitch_degrees - 90.0).abs() < 1e-3,
            "top view: pitch {}",
            top.pitch_degrees
        );
        assert!(
            (top.yaw_degrees.rem_euclid(90.0)).min(90.0 - top.yaw_degrees.rem_euclid(90.0)) < 1e-3,
            "square yaw {}",
            top.yaw_degrees
        );
        frame(
            &mut app,
            vec![button(egui::PointerButton::Primary, true, pos)],
        );
        pos.y -= 10.0;
        frame(&mut app, vec![egui::Event::PointerMoved(pos)]);
        frame(
            &mut app,
            vec![button(egui::PointerButton::Primary, false, pos)],
        );
        assert!(
            (app.scene.camera.yaw_degrees - top.yaw_degrees).abs() < 1e-2,
            "leaving the top view keeps the heading: {} -> {}",
            top.yaw_degrees,
            app.scene.camera.yaw_degrees
        );

        // Ctrl + LMB: Smart 2-plane / axis lock-on eases smoothly toward the nearest coordinate
        // plane or world axis (within 15° of two planes) and levels camera roll to 0°.
        app.controls.look_sensitivity = 0.5;
        app.controls.align_seconds = 0.05;
        app.scene.camera.yaw_degrees = 8.0;
        app.scene.camera.pitch_degrees = 6.0;
        app.scene.camera.roll_degrees = 25.0;
        app.world
            .navigate_camera(&app.scene, 0.0, false, None)
            .unwrap();
        let ctrl = egui::Modifiers::CTRL;
        frame(
            &mut app,
            vec![
                egui::Event::ModifiersChanged(ctrl),
                egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: ctrl,
                },
            ],
        );
        for i in 0..40 {
            if i < 4 {
                pos.x += 2.0;
            }
            frame(&mut app, vec![egui::Event::PointerMoved(pos)]);
        }
        frame(
            &mut app,
            vec![
                egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: ctrl,
                },
                egui::Event::ModifiersChanged(egui::Modifiers::NONE),
            ],
        );
        assert!(
            app.scene.camera.yaw_degrees.abs() < 1e-2
                && app.scene.camera.pitch_degrees.abs() < 1e-2
                && app.scene.camera.roll_degrees.abs() < 1e-2,
            "Ctrl + LMB within 15° of -Z must ease yaw, pitch, and roll to 0: {:?}",
            app.scene.camera
        );
    }

    #[test]
    fn camclip_toolbar_has_a_label_and_keeps_the_last_button_inset() {
        let _gpu_test = crate::test_gpu::lock();
        let mut app = App::new();
        let ctx = egui::Context::default();
        for width in [1000.0, 800.0] {
            for _ in 0..3 {
                let mut toolbar_rect = egui::Rect::NOTHING;
                let mut gap = 0.0;
                let mut minimum_height = 0.0;
                let mut output = ctx.run_ui(
                    egui::RawInput {
                        screen_rect: Some(egui::Rect::from_min_size(
                            egui::Pos2::ZERO,
                            egui::vec2(width, 400.0),
                        )),
                        ..Default::default()
                    },
                    |root| {
                        egui::CentralPanel::default().show(root, |ui| {
                            gap = ui.spacing().item_spacing.x;
                            minimum_height = ui.spacing().interact_size.y.max(
                                ui.text_style_height(&egui::TextStyle::Body)
                                    + 2.0 * ui.spacing().button_padding.y,
                            ) + egui::style::ScrollStyle::solid()
                                .allocated_width();
                            toolbar_rect = app.viewport_toolbar(ui, ui.max_rect()).rect;
                        });
                    },
                );
                assert!(
                    toolbar_rect.height() >= minimum_height,
                    "the viewport must retain room for controls and the separate scroll track"
                );
                let text_rect = |wanted: &str| {
                    output.shapes.iter().find_map(|shape| {
                        if let egui::epaint::Shape::Text(text) = &shape.shape {
                            (text.galley.job.text == wanted)
                                .then(|| text.galley.rect.translate(text.pos.to_vec2()))
                        } else {
                            None
                        }
                    })
                };
                if let (Some(label), Some(first), Some(last)) =
                    (text_rect("CamClip:"), text_rect("1"), text_rect("5"))
                {
                    assert!(
                        label.right() < first.left(),
                        "label must precede the camera slots"
                    );
                    assert!(
                        first.left() < last.left(),
                        "slot order must remain 1 through 5"
                    );
                    assert!(
                        last.right() <= toolbar_rect.right() - gap,
                        "last slot must stay inset from the toolbar end"
                    );
                } else {
                    assert!(!output.shapes.is_empty());
                    // The Area needs its sizing pass before content shapes are emitted.
                    if ctx.cumulative_frame_nr() > 1 {
                        panic!("CamClip label or buttons are missing");
                    }
                }
                output.textures_delta.clear();
            }
        }
    }

    #[test]
    fn viewport_toolbar_preserves_rmb_flight_and_continuous_camera_updates() {
        let _gpu_test = crate::test_gpu::lock();
        let mut app = App::new();
        let ctx = egui::Context::default();
        let mut time = 0.0;
        let mut frame = |app: &mut App, events| {
            time += 1.0 / 60.0;
            ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(640.0, 400.0),
                    )),
                    time: Some(time),
                    events,
                    ..Default::default()
                },
                |root| {
                    egui::CentralPanel::default().show(root, |ui| app.viewport(ui));
                },
            )
        };
        let pos = egui::pos2(300.0, 200.0);
        frame(&mut app, vec![egui::Event::PointerMoved(pos)]);
        frame(
            &mut app,
            vec![egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Secondary,
                pressed: true,
                modifiers: Default::default(),
            }],
        );
        assert!(
            app.fly.is_some(),
            "the real viewport toolbar must not swallow RMB outside its strip"
        );
        let before = app.fly.as_ref().unwrap().pose();
        let generation = app.generation;
        app.tool = ViewTool::Select;
        frame(
            &mut app,
            vec![
                egui::Event::Key {
                    key: egui::Key::W,
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers: Default::default(),
                },
                egui::Event::MouseMoved(egui::vec2(20.0, -10.0)),
            ],
        );
        for _ in 0..30 {
            frame(&mut app, vec![]);
        }
        let after = app.fly.as_ref().unwrap().pose();
        assert!(after.eye.distance(before.eye) > 0.01);
        assert!(after.orientation.angle_between(before.orientation) > 0.001);
        assert!(
            app.generation > generation + 10,
            "flight must continuously submit changing camera poses"
        );
        assert_eq!(app.tool, ViewTool::Select, "W in flight thrusts, not Move");
        let w = |pressed| egui::Event::Key {
            key: egui::Key::W,
            physical_key: None,
            pressed,
            repeat: false,
            modifiers: Default::default(),
        };
        let output = frame(
            &mut app,
            vec![
                w(false),
                egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Secondary,
                    pressed: false,
                    modifiers: Default::default(),
                },
            ],
        );
        assert!(
            output.viewport_output[&egui::ViewportId::ROOT]
                .commands
                .contains(&egui::ViewportCommand::CursorGrab(egui::CursorGrab::None))
        );
        assert!(app.fly.is_some(), "the release coasts");
        frame(&mut app, vec![w(true)]);
        assert_eq!(
            app.tool,
            ViewTool::Move,
            "released, W picks Move even while coasting"
        );
    }

    #[test]
    fn duplicate_bookmark_thumbnails_fill_the_next_empty_entry() {
        let _gpu_test = crate::test_gpu::lock();
        let scene = Scene::preset(Formula::Mandelbulb(Bulb::PRESET));
        // The scene as sent: the worker caps the bounces itself, so the card's scene is unchanged.
        let rendered = scene.clone();
        let ctx = egui::Context::default();
        let thumb = ctx.load_texture(
            "duplicate",
            ColorImage::new([1, 1], vec![Color32::WHITE]),
            TextureOptions::LINEAR,
        );
        let mut entries = vec![
            Entry {
                scene: scene.clone(),
                thumb: Some(thumb.clone()),
                path: None,
            },
            Entry {
                scene,
                thumb: None,
                path: None,
            },
        ];
        assert_eq!(pending_thumbnail_entry(&entries, &rendered), Some(1));
        entries[1].thumb = Some(thumb);
        assert_eq!(pending_thumbnail_entry(&entries, &rendered), None);
    }

    #[test]
    fn right_button_enters_flight_moves_and_releases_capture() {
        let _gpu_test = crate::test_gpu::lock();
        let mut app = App::new();
        let ctx = egui::Context::default();
        let mut time = 0.0;
        let mut frame = |app: &mut App, events: Vec<egui::Event>| {
            time += 1.0 / 60.0;
            ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(800.0, 600.0),
                    )),
                    time: Some(time),
                    events,
                    ..Default::default()
                },
                |root| {
                    egui::CentralPanel::default().show(root, |ui| {
                        ui.add(
                            egui::TextEdit::singleline(&mut app.scene.name)
                                .id(egui::Id::new("flight-test-text")),
                        );
                        let (_, response) =
                            ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
                        app.fly_camera(ui, &response);
                        app.handle_events(ui.ctx());
                    });
                },
            )
        };
        let pos = egui::pos2(400.0, 300.0);
        let button = |pressed| egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Secondary,
            pressed,
            modifiers: Default::default(),
        };
        let key = |pressed| egui::Event::Key {
            key: egui::Key::W,
            physical_key: None,
            pressed,
            repeat: false,
            modifiers: Default::default(),
        };
        frame(&mut app, vec![egui::Event::PointerMoved(pos)]);
        assert!(app.fly.is_none());
        let output = frame(&mut app, vec![button(true)]);
        assert!(app.fly.is_some(), "RMB in the viewport must enter flight");
        assert!(
            output.viewport_output[&egui::ViewportId::ROOT]
                .commands
                .contains(&egui::ViewportCommand::CursorGrab(egui::CursorGrab::Locked))
        );
        let before = app.scene.camera;
        let eye = app.fly.as_ref().unwrap().pose().eye;
        frame(
            &mut app,
            vec![key(true), egui::Event::MouseMoved(egui::vec2(40.0, -20.0))],
        );
        let first_yaw = app.scene.camera.yaw_degrees;
        assert!(
            app.fly.as_ref().unwrap().has_motion(),
            "The shared controller must take the mouse-look turn to ease in"
        );
        frame(&mut app, vec![]);
        assert!(
            (app.scene.camera.yaw_degrees - first_yaw).abs() > 0.1,
            "The shared controller must continue easing rotation without new mouse events"
        );
        for _ in 0..30 {
            frame(&mut app, vec![]);
        }
        assert!((app.scene.camera.yaw_degrees - before.yaw_degrees).abs() > 1.0);
        assert!(
            (app.fly.as_ref().unwrap().pose().eye - eye).length() > 0.01,
            "W must translate the camera independently of mouse rotation"
        );
        let output = frame(&mut app, vec![button(false), key(false)]);
        assert!(
            output.viewport_output[&egui::ViewportId::ROOT]
                .commands
                .contains(&egui::ViewportCommand::CursorGrab(egui::CursorGrab::None))
        );
        for _ in 0..600 {
            frame(&mut app, vec![]);
        }
        assert!(
            app.fly.is_none(),
            "Flight must end once release inertia settles"
        );
        let toggle = |pressed, shift| egui::Event::Key {
            key: egui::Key::L,
            physical_key: Some(egui::Key::L),
            pressed,
            repeat: false,
            modifiers: egui::Modifiers {
                shift,
                ..Default::default()
            },
        };
        frame(&mut app, vec![toggle(true, false), button(true)]);
        assert!(
            app.scene.camera.free_flight,
            "L must enable free flight while RMB is held"
        );
        let roll = |pressed| egui::Event::Key {
            key: egui::Key::E,
            physical_key: None,
            pressed,
            repeat: false,
            modifiers: Default::default(),
        };
        frame(&mut app, vec![toggle(false, false), roll(true)]);
        for _ in 0..30 {
            frame(&mut app, vec![]);
        }
        assert!(
            app.scene.camera.roll_degrees.abs() > 1.0,
            "Q/E must roll the free-flight camera"
        );
        frame(&mut app, vec![roll(false), button(false)]);
        for _ in 0..600 {
            frame(&mut app, vec![]);
        }
        let saved_roll = app.scene.camera.roll_degrees;
        assert!(
            saved_roll.abs() > 1.0,
            "Release must retain the free-flight roll"
        );
        let saved = serde_json::to_string(&app.scene).unwrap();
        let loaded: Scene = serde_json::from_str(&saved).unwrap();
        assert_eq!(loaded.camera, app.scene.camera);
        let p = app.scene.pack(80, 60);
        let right = app.scene.camera.orientation() * glam::Vec3::X;
        assert!(
            (glam::Vec3::from_slice(&p[crate::params::P_CAM_RIGHT..]) - right).length() < 1e-5,
            "CUDA camera basis must retain free-flight roll"
        );
        frame(&mut app, vec![toggle(true, true)]);
        assert!(
            !app.scene.camera.free_flight,
            "Shift+L must restore horizon mode"
        );
        // The levelling spring's time constant is the rotate decay: ten of them settle it.
        for _ in 0..(10.0 * 60.0 * app.controls.rotate_decay) as usize {
            frame(&mut app, vec![]);
        }
        // Horizon mode levels the resting camera smoothly onto the nearest world plane.
        assert!(app.fly.is_none(), "the levelling rig settles and lets go");
        let right = app.scene.camera.orientation() * glam::Vec3::X;
        assert!(
            [glam::Vec3::X, glam::Vec3::Y, glam::Vec3::Z]
                .iter()
                .any(|axis| right.dot(*axis).abs() < 2e-3),
            "level to a world plane: right {right}"
        );
        frame(&mut app, vec![toggle(false, true)]);
        let shortcut = |key| egui::Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: Default::default(),
        };
        // F frames the world's objects: move the fractal through the document.
        let fractal = app
            .world
            .document
            .nodes()
            .into_iter()
            .find(|node| node.kind == crate::world::WorldKind::Fractal)
            .unwrap()
            .id;
        app.world
            .execute(crate::world::WorldCommand::SetAttribute {
                id: fractal,
                path: "/transform/position".into(),
                value: serde_json::json!([7.0, -3.0, 2.0]),
                frame: 0.0,
            })
            .unwrap();
        app.refresh_scene().unwrap();
        frame(&mut app, vec![shortcut(egui::Key::F)]);
        assert!(
            (glam::Vec3::from_array(app.scene.camera.target) - glam::Vec3::new(7.0, -3.0, 2.0))
                .length()
                < 1e-5
        );
        assert!(app.fly.is_none());
        frame(&mut app, vec![shortcut(egui::Key::H)]);
        assert_eq!(app.scene.camera, app.origin.camera);
        let mut home_release = shortcut(egui::Key::H);
        if let egui::Event::Key { pressed, .. } = &mut home_release {
            *pressed = false;
        }
        frame(&mut app, vec![home_release]);
        assert_eq!(
            app.world
                .document
                .attribute_value(fractal, "/transform/position", 0.0)
                .unwrap(),
            serde_json::json!([7.0, -3.0, 2.0]),
            "H must reset only the camera"
        );
        let flight_key = |key, pressed| egui::Event::Key {
            key,
            physical_key: None,
            pressed,
            repeat: false,
            modifiers: Default::default(),
        };
        frame(&mut app, vec![button(true)]);
        let eye = app.fly.as_ref().unwrap().eye;
        frame(&mut app, vec![flight_key(egui::Key::R, true)]);
        for _ in 0..30 {
            frame(&mut app, vec![]);
        }
        assert!(
            app.fly.as_ref().unwrap().eye.y > eye.y,
            "R must strafe upward"
        );
        let raised = app.fly.as_ref().unwrap().eye;
        frame(
            &mut app,
            vec![
                flight_key(egui::Key::R, false),
                flight_key(egui::Key::C, true),
            ],
        );
        for _ in 0..90 {
            frame(&mut app, vec![]);
        }
        assert!(
            app.fly.as_ref().unwrap().eye.y < raised.y,
            "C must strafe downward"
        );
        let lowered = app.fly.as_ref().unwrap().eye;
        let playing = app.world_ui.playing;
        frame(
            &mut app,
            vec![
                flight_key(egui::Key::C, false),
                flight_key(egui::Key::Space, true),
            ],
        );
        for _ in 0..90 {
            frame(&mut app, vec![]);
        }
        assert!(
            app.fly.as_ref().unwrap().eye.y > lowered.y,
            "Space must strafe upward like R"
        );
        assert_eq!(app.world_ui.playing, playing, "Space in flight is not Play");
        frame(
            &mut app,
            vec![
                flight_key(egui::Key::Space, false),
                flight_key(egui::Key::Q, true),
            ],
        );
        for _ in 0..30 {
            frame(&mut app, vec![]);
        }
        assert!(
            !app.scene.camera.free_flight,
            "Q/E tilt against the horizon lock instead of releasing it"
        );
        assert!(
            app.scene.camera.roll_degrees.abs() > 1.0,
            "Q must tilt the camera"
        );
        frame(
            &mut app,
            vec![flight_key(egui::Key::Q, false), button(false)],
        );
        // Ten time constants of the levelling spring (the rotate decay).
        for _ in 0..(10.0 * 60.0 * app.controls.rotate_decay) as usize {
            frame(&mut app, vec![]);
        }
        assert!(
            app.scene.camera.roll_degrees.abs() < 0.1,
            "a short tilt springs back level: {}",
            app.scene.camera.roll_degrees
        );
        frame(&mut app, vec![shortcut(egui::Key::H)]);
        ctx.memory_mut(|m| m.request_focus(egui::Id::new("flight-test-text")));
        frame(&mut app, vec![]);
        frame(
            &mut app,
            vec![toggle(true, false), egui::Event::Text("`".into())],
        );
        assert!(
            !app.scene.camera.free_flight,
            "Typing backtick in a text field must not switch mode"
        );
    }

    #[test]
    fn frame_bounds_fit_transformed_fractal_and_fallback_in_both_aspects() {
        let _gpu_test = crate::test_gpu::lock();
        use glam::Vec3;
        let mut app = App::new();
        // The single-object scene (no world document objects): `framing_bounds`' fallback box.
        app.scene.world_render = false;
        for formula in [
            crate::scene::Formula::Mandelbulb(crate::scene::Bulb::PRESET),
            crate::scene::Formula::Mandelbox(crate::scene::MandelBox::PRESET),
        ] {
            app.scene.formula = formula;
            app.scene.object.offset = [12.0, -4.0, 3.0];
            app.scene.object.scale = 2.0;
            app.scene.object.rotation_degrees = [0.0, 45.0, 0.0];
            let (min, max) = app.scene.framing_bounds();
            assert!(
                ((min + max) * 0.5 - Vec3::from_array(app.scene.object.offset)).length() < 1e-5
            );
            if matches!(formula, crate::scene::Formula::Mandelbox(_)) {
                assert!(
                    ((max - min).y - 20.0).abs() < 1e-4,
                    "Fallback box must be 10 units before scaling"
                );
                assert!((max - min).x > 28.0, "Bounds must include object rotation");
            } else {
                assert!(
                    (max - min).y < 9.0,
                    "Explicit fractal bounds must replace the fallback"
                );
            }
            app.scene.camera.free_flight = true;
            app.scene.camera.roll_degrees = 35.0;
            let orientation = app.scene.camera.orientation();
            for (width, height, fov) in [(800, 600, 40.0), (400, 900, 40.0), (800, 600, 120.0)] {
                app.scene.camera.fov_y_degrees = fov;
                app.frame_camera(width, height);
                assert_eq!(
                    app.scene.camera.orientation(),
                    orientation,
                    "Framing must retain rotation and roll"
                );
                let cam = app.scene.camera;
                let eye = Vec3::from_array(cam.target)
                    - (orientation * -Vec3::Z) * cam.distance * formula.framing_radius();
                let half_y = (cam.fov_y_degrees.to_radians() * 0.5).tan();
                let half_x = half_y * width as f32 / height as f32;
                for x in [min.x, max.x] {
                    for y in [min.y, max.y] {
                        for z in [min.z, max.z] {
                            let p = orientation.inverse() * (Vec3::new(x, y, z) - eye);
                            assert!(
                                p.z < 0.0 && p.x.abs() < -p.z * half_x && p.y.abs() < -p.z * half_y,
                                "Every bounding box corner must fit the viewport"
                            );
                        }
                    }
                }
            }
        }
        assert_eq!(Settings::default().controls, Controls::default());
        app.controls.look_sensitivity = 0.4;
        app.controls.fly_speed = 3.0;
        let loaded: Controls =
            serde_json::from_str(&serde_json::to_string(&app.controls).unwrap()).unwrap();
        assert_eq!(loaded.look_sensitivity, 0.4);
        assert_eq!(loaded.fly_speed, 3.0);
    }

    #[test]
    fn world_frame_camera_fits_visual_sphere_and_respects_selection() {
        let _gpu_test = crate::test_gpu::lock();
        let mut app = App::new();
        app.refresh_scene().unwrap();
        assert!(app.scene.world_render);
        app.frame_camera(1600, 900);
        assert!(
            app.scene.camera.distance > 2.5 && app.scene.camera.distance < 3.5,
            "World Mandelbulb framing distance should match ~3.0 visual radii, got {}",
            app.scene.camera.distance
        );
        app.world
            .execute(crate::world::WorldCommand::Create {
                kind: crate::world::WorldKind::Fractal,
                name: "Fractal 2".into(),
                parent: None,
            })
            .unwrap();
        let second = app.world.active().expect("created fractal selected");
        app.world
            .execute(crate::world::WorldCommand::SetAttribute {
                id: second,
                path: "/transform/position".into(),
                value: serde_json::json!([25.0, 0.0, 0.0]),
                frame: 0.0,
            })
            .unwrap();
        app.refresh_scene().unwrap();
        app.world.select_only(second);
        app.frame_camera(1600, 900);
        assert!(
            (app.scene.camera.target[0] - 25.0).abs() < 1e-4,
            "Selected fractal must be framed at its world position, got {:?}",
            app.scene.camera.target
        );
        assert!(
            app.scene.camera.distance > 2.5 && app.scene.camera.distance < 3.5,
            "Selected fractal framing distance should stay tight, got {}",
            app.scene.camera.distance
        );
    }

    #[test]
    fn shared_flight_inertia_coasts_and_stops_on_all_six_axes() {
        let _gpu_test = crate::test_gpu::lock();
        use cam_controls::{CameraIntent, SpaceFlight};
        let cases = [
            (
                "forward",
                CameraIntent::Thrust {
                    forward: 1.0,
                    right: 0.0,
                    up: 0.0,
                },
            ),
            (
                "right",
                CameraIntent::Thrust {
                    forward: 0.0,
                    right: 1.0,
                    up: 0.0,
                },
            ),
            (
                "up",
                CameraIntent::Thrust {
                    forward: 0.0,
                    right: 0.0,
                    up: 1.0,
                },
            ),
            (
                "yaw",
                CameraIntent::Look {
                    dyaw: 0.001,
                    dpitch: 0.0,
                },
            ),
            (
                "pitch",
                CameraIntent::Look {
                    dyaw: 0.0,
                    dpitch: 0.001,
                },
            ),
            ("roll", CameraIntent::Roll { d: 1.0 }),
        ];
        let viewport = cam_viewport::ViewportSize::new(800, 600);
        for (name, intent) in cases {
            let mut fly = SpaceFlight::default();
            // The rig exactly as the viewport configures it.
            fly.inertia = Controls::default().inertia(1.0);
            for _ in 0..30 {
                fly.apply_intent(intent, viewport);
                fly.update_dynamics(1.0 / 60.0);
            }
            fly.apply_intent(
                CameraIntent::Thrust {
                    forward: 0.0,
                    right: 0.0,
                    up: 0.0,
                },
                viewport,
            );
            fly.apply_intent(CameraIntent::Roll { d: 0.0 }, viewport);
            let before = fly.pose();
            let momentum = fly.momentum;
            assert!(
                fly.update_dynamics(1.0 / 60.0),
                "{name} must coast after release"
            );
            if matches!(intent, CameraIntent::Thrust { .. }) {
                assert!((fly.eye - before.eye).length() > 0.0, "{name}");
                assert!(
                    fly.momentum.linear.length() < momentum.linear.length(),
                    "{name} must damp"
                );
            } else if matches!(intent, CameraIntent::Roll { .. }) {
                assert!(
                    fly.orientation.angle_between(before.orientation) > 1e-4,
                    "{name}"
                );
                assert!(
                    fly.momentum.angular.length() < momentum.angular.length(),
                    "{name} must damp"
                );
            } else {
                // Smoothed look: the turn the mouse asked for keeps easing in, without momentum.
                assert!(
                    fly.orientation.angle_between(before.orientation) > 1e-6,
                    "{name}"
                );
                assert_eq!(fly.momentum.angular, glam::Vec3::ZERO, "{name}");
            }
            for _ in 0..600 {
                fly.update_dynamics(1.0 / 60.0);
            }
            assert!(!fly.update_dynamics(1.0 / 60.0), "{name} must settle");
        }
        let mut old =
            serde_json::to_value(Scene::preset(Formula::Mandelbulb(Bulb::PRESET))).unwrap();
        old["camera"]
            .as_object_mut()
            .unwrap()
            .remove("roll_degrees");
        old["camera"].as_object_mut().unwrap().remove("free_flight");
        assert!(serde_json::from_value::<Scene>(old).is_err());
    }

    #[test]
    fn cuda_play_attaches_the_take_and_the_camera_follows_it_while_playing() {
        let _gpu_test = crate::test_gpu::lock();
        let mut app = App::new();
        // A Still profile that renders fast, at a resolution scale below 1: the scale once made
        // the cached preview's size key disagree with the viewport and cancel playback at once.
        let policy = app.world.document.viewport_policy(0.0).unwrap();
        let still = policy.still_id;
        for (path, value) in [
            ("/quality/samples", serde_json::json!(1)),
            ("/quality/resolution_scale", serde_json::json!(0.75)),
        ] {
            app.world
                .execute(crate::world::WorldCommand::SetAttribute {
                    id: still,
                    path: path.into(),
                    value,
                    frame: 0.0,
                })
                .unwrap();
        }
        let camera = app.world.document.active_camera.unwrap();
        let base = app.world.document.snapshot(0.0).unwrap().camera;
        let mut turned = base;
        turned.yaw_degrees += 40.0;
        let id = app.world.add_take(
            crate::takes::Take::from_frames(10.0, 1.0, Some(camera), vec![base, turned]).unwrap(),
        );
        let ctx = egui::Context::default();
        let frame = |app: &mut App, i: u32| {
            let _ = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(1280.0, 720.0),
                    )),
                    time: Some(f64::from(i) / 30.0),
                    predicted_dt: 1.0 / 30.0,
                    ..Default::default()
                },
                |ui| app.ui(ui),
            );
        };
        // The window's layout settles first, as it has long before anyone presses Play.
        for i in 0..10 {
            frame(&mut app, i);
        }
        app.emit_take(takes_ui::Action::Play(id));
        assert_eq!(app.world.document.take_clip(camera).unwrap(), None);
        app.handle_events(&ctx);
        assert_eq!(app.world.document.take_clip(camera).unwrap(), Some(id));
        assert_eq!(app.world_ui.playhead, 10);
        assert!(app.world_ui.playing);
        // Play advances as fast as the preview renders its frames: real time, bounded.
        let started = std::time::Instant::now();
        let mut i = 10;
        while app.world_ui.playhead < 13 && started.elapsed().as_secs() < 300 {
            assert!(app.world_ui.playing, "playback stopped: {}", app.status);
            frame(&mut app, i);
            i += 1;
            std::thread::sleep(std::time::Duration::from_millis(30));
        }
        assert!(
            app.world_ui.playhead >= 13,
            "the playhead advanced: {}",
            app.world_ui.playhead
        );
        assert_eq!(
            app.world.document.take_clip(camera).unwrap(),
            Some(id),
            "still attached"
        );
        // Cached playback presents the worker's frames (rendered through the same snapshot
        // hook); paused, the viewport evaluates the playhead itself.
        app.world_ui.playing = false;
        frame(&mut app, i);
        let head = f64::from(app.world_ui.playhead);
        let expected = app.world.document.snapshot(head).unwrap().camera;
        assert_ne!(expected, base, "the take moved the camera");
        assert_eq!(app.scene.camera, expected, "the viewport shows the take");
    }

    #[test]
    fn scenes_made_from_one_template_and_save_as_copies_are_distinct_documents() {
        let _gpu_test = crate::test_gpu::lock();
        let mut app = App::new();
        let preset = Scene::preset(Formula::Mandelbulb(Bulb::PRESET));
        let mut template = preset.clone();
        template.document = Some(Box::new(crate::world::WorldDocument::from_scene(&preset)));
        app.load(template.clone(), Load::New);
        let first = app.world.document.graph.id;
        app.load(template.clone(), Load::New);
        let second = app.world.document.graph.id;
        assert_ne!(first, second, "two scenes from one template");
        assert_ne!(Some(first), template.document.as_ref().map(|d| d.graph.id));
        // A reset returns to this scene, not to the template, as one undoable step.
        let moved = app.origin.animation.first + 1;
        app.world
            .execute(crate::world::WorldCommand::SetMark {
                slot: 1,
                frame: Some(moved),
            })
            .unwrap();
        app.scene_file_path = Some(PathBuf::from("kept.frac.json"));
        app.reset_scene();
        assert_eq!(app.world.document.graph.id, second);
        assert_eq!(
            app.world.document.marks.get(&1),
            Some(&app.origin.animation.first)
        );
        assert_eq!(app.scene_file_path, Some(PathBuf::from("kept.frac.json")));
        assert!(app.world.undo(), "the reset is undoable");
        assert_eq!(app.world.document.marks.get(&1), Some(&moved));
        assert!(app.world.redo());
        // An edit then Save As: the copy is a new document, and undo keeps it so.
        app.world
            .execute(crate::world::WorldCommand::SetMark {
                slot: 1,
                frame: Some(3),
            })
            .unwrap();
        app.submit_scene_action(
            std::env::temp_dir().join("warpbro-identity-copy.json"),
            crate::templates::FileAction::Save,
        );
        let copy = app.world.document.graph.id;
        assert_ne!(copy, second);
        assert!(app.world.undo());
        assert_eq!(app.world.document.graph.id, copy);
    }

    #[test]
    fn cuda_scrubbing_after_a_completed_frame_renders_the_new_frame_live() {
        let _gpu_test = crate::test_gpu::lock();
        let mut app = App::new();
        let policy = app.world.document.viewport_policy(0.0).unwrap();
        let still = policy.still_id;
        app.world
            .execute(crate::world::WorldCommand::SetAttribute {
                id: still,
                path: "/quality/samples".into(),
                value: serde_json::json!(2),
                frame: 0.0,
            })
            .unwrap();
        let ctx = egui::Context::default();
        let frame = |app: &mut App, i: u32| {
            let _ = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(1280.0, 720.0),
                    )),
                    time: Some(f64::from(i) / 30.0),
                    predicted_dt: 1.0 / 30.0,
                    ..Default::default()
                },
                |ui| app.ui(ui),
            );
        };
        // Render until the completed (denoised) frame is cached: the cache now has a position.
        let started = std::time::Instant::now();
        let mut i = 0;
        while app.preview.position().is_none() && started.elapsed().as_secs() < 300 {
            frame(&mut app, i);
            i += 1;
            std::thread::sleep(std::time::Duration::from_millis(30));
        }
        assert!(
            app.preview.position().is_some(),
            "the completed frame was cached"
        );
        // Scrub, paused, to a frame the cache does not hold: the live viewport renders it.
        app.world_ui.seek(30);
        frame(&mut app, i);
        frame(&mut app, i + 1);
        assert!(!app.preview.presents(), "the cache does not hold frame 30");
        assert!(
            app.request.as_ref().is_some_and(|r| r.active),
            "the viewport renders live"
        );
        assert_eq!(app.viewport_stamp.map(|s| s.3), Some(30));
    }

    #[test]
    fn camclip_is_app_state_saved_with_a_scene_and_replaced_by_one_that_carries_it() {
        let _gpu_test = crate::test_gpu::lock();
        let mut app = App::new();
        let mut stored = app.scene.camera;
        stored.yaw_degrees += 33.0;
        app.camera_slots.slots[0] = Some(stored);
        // A new scene keeps the CamClip.
        app.new_scene();
        assert_eq!(app.camera_slots.slots[0], Some(stored));
        // Saving writes it into the scene.
        app.submit_scene_action(
            std::env::temp_dir().join("warpbro-camclip.json"),
            crate::templates::FileAction::Save,
        );
        assert_eq!(app.world.document.camera_slots.slots[0], Some(stored));
        // A scene that carries a CamClip replaces the application's.
        let mut other = app.scene.clone();
        let mut document = app.world.document.clone();
        let mut theirs = stored;
        theirs.yaw_degrees += 10.0;
        document.camera_slots = crate::camera_slots::CameraSlots::default();
        document.camera_slots.slots[2] = Some(theirs);
        other.document = Some(Box::new(document));
        app.load(other, Load::Same);
        assert_eq!(app.camera_slots.slots[0], None);
        assert_eq!(app.camera_slots.slots[2], Some(theirs));
    }

    #[test]
    fn output_profile_refuses_bounds_display_in_form_and_export_start() {
        let _gpu_test = crate::test_gpu::lock();
        let mut app = App::new();
        let output = app.world.document.output_render_profile().unwrap();
        app.world
            .execute(crate::world::WorldCommand::SetAttribute {
                id: output,
                path: "/render/display".into(),
                value: serde_json::json!("Bounds"),
                frame: 0.0,
            })
            .unwrap();
        let ctx = egui::Context::default();
        let _ = ctx.run_ui(egui::RawInput::default(), |root| {
            egui::CentralPanel::default().show(root, |ui| app.export_ui(ui));
        });
        let expected =
            "The Output profile shows Bounds: nothing to render; pick Path traced or Direct";
        assert_eq!(app.export.quality, Err(expected.to_string()));
        let frozen = freeze_world_scene(&app.scene, &app.world.document);
        assert_eq!(
            app.export
                .start(&frozen, &app.renderer, app.colour.config()),
            Err(expected.to_string())
        );
    }

    #[test]
    fn add_menu_memory_persists_in_settings_and_survives_scene_reset() {
        let _gpu_test = crate::test_gpu::lock();
        use crate::world::WorldKind;
        let mut app = App::new();
        app.changed_settings_json().unwrap();
        let memory = crate::world_ui::AddMenuMemory {
            history: vec![WorldKind::Mask, WorldKind::Fractal],
            favorites: vec![WorldKind::Camera],
            usage: vec![(WorldKind::Fractal, 4), (WorldKind::Mask, 2)],
        };
        app.world_ui.set_add_menu_memory(memory.clone());
        let json = app.changed_settings_json().unwrap().unwrap();
        let saved: Settings = serde_json::from_str(&json).unwrap();
        assert_eq!(saved.add_menu, memory);
        for _ in 0..8 {
            assert!(app.changed_settings_json().unwrap().is_none());
        }
        app.new_scene();
        assert_eq!(
            app.world_ui.add_menu_memory(),
            memory,
            "new scene / WorldUi::reset must preserve the Tab menu memory"
        );
    }

    #[test]
    fn status_bar_reports_adaptive_tile_convergence() {
        let _gpu_test = crate::test_gpu::lock();
        let ctx = egui::Context::default();
        let mut frame = Frame {
            generation: 1,
            preview: false,
            width: 64,
            height: 64,
            pixels: vec![0; 64 * 64],
            light: vec![[0.0; 4]; 64 * 64],
            radiance: Vec::new(),
            light_kind: crate::color::DisplayLight::Relative,
            colour_error: None,
            denoised_samples: 0,
            denoise_ms: 0.0,
            denoise_error: None,
            samples: 12,
            active_tiles: (32, 128),
            converged: false,
            last_ms: 1.0,
            last_spp: 1,
            unresolved: 0.0,
            sdr_bytes: Arc::new(Vec::new()),
            hdr_bytes: Arc::new(Vec::new()),
        };
        let texts = |frame: &Frame| {
            let output = ctx.run_ui(egui::RawInput::default(), |root| {
                egui::CentralPanel::default().show(root, |ui| {
                    ui.horizontal(|ui| sample_status_ui(ui, frame, 64, false, 120.0));
                });
            });
            fn collect(shape: &egui::epaint::Shape, out: &mut Vec<String>) {
                match shape {
                    egui::epaint::Shape::Text(text) => out.push(text.galley.text().to_owned()),
                    egui::epaint::Shape::Vec(shapes) => {
                        for s in shapes {
                            collect(s, out);
                        }
                    }
                    _ => {}
                }
            }
            let mut out = Vec::new();
            for s in &output.shapes {
                collect(&s.shape, &mut out);
            }
            out
        };
        let partial = texts(&frame);
        assert!(partial.iter().any(|t| t == "12 / 64 spp"), "{partial:?}");
        assert!(partial.iter().any(|t| t == "75% converged"), "{partial:?}");
        frame.active_tiles = (0, 128);
        frame.converged = true;
        let done = texts(&frame);
        assert!(done.iter().any(|t| t == "converged"), "{done:?}");
    }

    /// The document owns the colour selection: the colour controls follow it through edits,
    /// undo and redo, and the saved default seeds only new documents, never a loaded one.
    #[test]
    fn colour_selection_follows_the_document_and_seeds_only_new_scenes() {
        let _gpu_test = crate::test_gpu::lock();
        let mut app = App::new();
        app.refresh_scene().unwrap();
        let original = app.scene.colour.clone();
        let before = app.scene.clone();
        let mut edited = app.scene.clone();
        edited.colour.look = "Edited look".into();
        let frame = f64::from(app.world_ui.playhead);
        app.world
            .edit_snapshot(None, &before, &edited, frame)
            .unwrap();
        app.refresh_scene().unwrap();
        assert_eq!(app.colour.sel, edited.colour);
        assert!(app.world.undo());
        app.refresh_scene().unwrap();
        assert_eq!(app.colour.sel, original);
        assert!(app.world.redo());
        app.refresh_scene().unwrap();
        assert_eq!(app.colour.sel, edited.colour);

        app.default_colour.look = "Default look".into();
        let mut own = Scene::preset(Formula::Mandelbox(MandelBox::PRESET));
        own.colour.look = "Own look".into();
        app.load(own, Load::New);
        app.refresh_scene().unwrap();
        assert_eq!(app.colour.sel.look, "Own look");
        app.new_scene();
        app.refresh_scene().unwrap();
        assert_eq!(app.scene.colour.look, "Default look");
        assert_eq!(app.colour.sel.look, "Default look");
    }

    /// A gallery, bookmark or swatch card whose render failed is not asked for again.
    #[test]
    fn failed_card_thumbnails_are_not_requested_again() {
        use crate::material_gallery::ThumbId;
        let _gpu_test = crate::test_gpu::lock();
        let mut app = App::new();
        let cards: Vec<ThumbId> = (0..app.gallery.len())
            .map(ThumbId::Gallery)
            .chain((0..app.bookmarks.len()).map(ThumbId::Bookmark))
            .chain((0..app.swatches.len()).map(ThumbId::Swatch))
            .collect();
        for card in cards {
            app.fail_thumbnail(card.encode(), "test failure".into());
        }
        for _ in 0..4 {
            app.render_one_thumbnail();
            if let Some((id, _)) = app.thumb_pending.take() {
                assert!(matches!(ThumbId::decode(id), ThumbId::Material(_)));
            }
        }
    }

    /// Saving and deleting a bookmark change the list only once the file worker confirms.
    #[test]
    fn bookmark_list_changes_only_after_the_file_does() {
        let _gpu_test = crate::test_gpu::lock();
        let mut app = App::new();
        let count = app.bookmarks.len();
        let path = PathBuf::from("bookmark-test.json");
        let event = |result: Result<Option<Box<Scene>>, String>| crate::io_service::SceneEvent {
            id: 1,
            path: path.clone(),
            result,
        };
        let card = |app: &App| Entry {
            scene: app.frozen_world_scene(),
            thumb: None,
            path: None,
        };
        app.bookmark_event(
            BookmarkOp::Save(Box::new(card(&app))),
            event(Err("disk full".into())),
        );
        assert_eq!(app.bookmarks.len(), count);
        app.bookmark_event(BookmarkOp::Save(Box::new(card(&app))), event(Ok(None)));
        assert_eq!(app.bookmarks.len(), count + 1);
        assert_eq!(
            app.bookmarks.last().and_then(|entry| entry.path.clone()),
            Some(path.clone())
        );
        assert!(app.panels_to_open.contains(&dock::Panel::Bookmarks));
        app.bookmark_event(BookmarkOp::Delete, event(Err("locked".into())));
        assert_eq!(app.bookmarks.len(), count + 1);
        app.bookmark_event(BookmarkOp::Delete, event(Ok(None)));
        assert_eq!(app.bookmarks.len(), count);
    }
}
