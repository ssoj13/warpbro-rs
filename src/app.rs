//! The browser UI (egui): gallery + bookmarks with GPU thumbnails, the progressive viewport,
//! the scene inspector and viewport screenshots. Renders (PNG / EXR / video) go through the
//! Render / Encode panel (`export.rs`); all output lands in `~/.warpbro/out/<timestamp>`.

use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use egui::{self, Color32, ColorImage, RichText, Sense, TextureHandle, TextureOptions, Vec2};

use crate::render_service::{Command, Frame, RenderEvent, RenderService, ViewportRequest};
use crate::scene::*;
use crate::world::WorldDocument;
use egui_attr_grid::{AttrField, AttrGridHooks, AttrGridState, AttrValue, render_grid_with_config};
use egui_widgets_config::AttrMetrics;
use std::collections::HashSet;
use std::sync::Arc;

#[path = "dock.rs"]
mod dock;

const THUMB_W: usize = 224;
const THUMB_H: usize = 126;
const THUMB_SPP: u32 = 24;
/// After the last camera / parameter change, keep the low-resolution preview this long.
const PREVIEW_HOLD_S: f32 = 0.18;

pub fn run() -> anyhow::Result<()> {
    crate::window::run()
}

struct Entry {
    scene: Scene,
    thumb: Option<TextureHandle>,
    path: Option<PathBuf>,
}

#[derive(PartialEq)]
enum Tab {
    Gallery,
    Bookmarks,
    Materials,
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
    let mut s = Scene::preset(crate::params::FAMILY_QUAT);
    if let Formula::QuaternionJulia(q) = &mut s.formula {
        q.constant = [0.0; 4];
    }
    s.camera.distance = 2.3;
    s.camera.fov_y_degrees = 30.0;
    s.camera.yaw_degrees = 20.0;
    s.camera.pitch_degrees = 15.0;
    s.render.max_bounces = 4;
    s.render.iterations = 8;
    s.render.exposure_stops = -0.5;
    // Neutral grey "studio" sky so metals show their own tint, not the blue sky's.
    s.lighting.sun_azimuth = 45.0;
    s.lighting.sun_elevation = 40.0;
    s.lighting.sun_intensity = 3.0;
    s.lighting.sky_horizon = [0.03, 0.03, 0.03];
    s.lighting.sky_zenith = [0.14, 0.14, 0.14];
    s.lighting.sky_intensity = 5.0;
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
    gui_fps: u32,
    scene: Scene,
    world: crate::world::WorldEditor,
    world_ui: crate::world_ui::WorldUi,
    preview: crate::preview::PreviewController,
    preview_key: Option<(playa_graph::NodeId, u64, bool, u32, u32, usize, usize, u32)>,
    preview_sequence: u64,
    material_targets: Vec<crate::world::NodeId>,
    material_selection_stamp: Option<(playa_graph::NodeId, u64, Option<crate::world::NodeId>, u64)>,
    preview_progress: Option<(u32, u32, usize)>,
    evaluated_world: Option<(playa_graph::NodeId, u64, u64)>,
    load_revision: u64,
    /// The preset / bookmark the scene came from, for "Reset".
    origin: Scene,

    showing_preview: bool,
    hdr_view: std::sync::Arc<std::sync::Mutex<egui_hdr_view::HdrView>>,
    pub display: egui_display::DisplayPrefs,
    colour: crate::ocio::State,
    prefs: egui_prefs2::PrefsPanelState,
    recorder_options: crate::camera_recorder::Options,
    camera_recording: Option<(Instant, crate::camera_recorder::Recording)>,
    /// Splitter state of the three Settings > Controls grids.
    prefs_grids: [AttrGridState; 3],
    dock: egui_dock::DockState<dock::Panel>,
    panels_to_open: Vec<dock::Panel>,
    toolbar: egui_viewport_toolbar::ToolbarState,
    camera_slots: crate::camera_slots::CameraSlots,
    fonts: dock::Fonts,
    applied_fonts: Option<dock::Fonts>,
    viewport_visible: bool,
    viewport_rect: Option<egui::Rect>,
    config_picker: egui_file_dialog::FileDialog,
    scene_picker: Option<(crate::templates::FileAction, egui_file_dialog::FileDialog)>,
    scene_file_path: Option<PathBuf>,
    scene_file_sequence: u64,
    scene_file_pending: Option<(u64, u64, crate::templates::FileAction)>,
    templates: Vec<crate::templates::Entry>,
    templates_requested: bool,
    templates_pending: bool,
    persisted_settings: Option<Settings>,
    persisted_layout_revision: u64,
    snapshot_before: Option<Scene>,
    #[cfg(test)]
    settings_serializations: usize,
    #[cfg(test)]
    snapshot_clones: usize,
    status_layout: egui_statusbar::StatusBarLayout,
    status_resizable: bool,
    render_editor: crate::inspector::RenderEditor,
    gallery: Vec<Entry>,
    bookmarks: Vec<Entry>,

    swatches: Vec<Option<TextureHandle>>,
    material_gallery: crate::material_gallery::MaterialGallery,
    tab: Tab,
    target_spp: u32,
    spp_per_frame: u32,
    last_change: Instant,
    last_scene: Scene,
    resolution: f32,
    paused: bool,
    /// Viewport A/B: raw samples instead of the denoised image (view state, not the scene).
    raw_view: bool,
    /// Exposure parked by the toolbar's EV bypass (`exposure_control`).
    exposure_hold: f32,
    show_ui: bool,
    status: String,
    frame_ms: f32,
    seed: u32,
    /// `FRAC_SNAP=out.png [FRAC_SNAP_PRESET=i] [FRAC_SNAP_SPP=n]`: screenshot the window once the
    /// thumbnails and n viewport samples are done, then quit (for docs).
    snap: Option<(PathBuf, u32, bool)>,
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
    /// Persistent mouse sensitivity and flight speed (also adjusted by the wheel).
    controls: Controls,
}

/// Shared fitted tile geometry for scene, workspace-material and library cards.
fn thumbnail_grid(width: f32, maximum: Vec2, gap: f32) -> (usize, Vec2) {
    let width = width.max(1.0);
    let columns = ((width + gap) / (maximum.x + gap)).floor().max(1.0) as usize;
    let cell = ((width - gap * (columns - 1) as f32) / columns as f32).clamp(1.0, maximum.x);
    (columns, Vec2::new(cell, cell * maximum.y / maximum.x))
}

fn now_stamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn load_bookmarks() -> Vec<Entry> {
    let dir = crate::warpbro_dir().join("bookmarks");
    let mut v: Vec<(PathBuf, Scene)> = std::fs::read_dir(&dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .filter_map(|p| {
            let s = std::fs::read_to_string(&p).ok()?;
            let scene = crate::io_service::decode_scene(&s).ok()?;
            Some((p, scene))
        })
        .collect();
    v.sort_by(|a, b| a.0.cmp(&b.0));
    v.into_iter()
        .map(|(p, scene)| Entry {
            scene,
            thumb: None,
            path: Some(p),
        })
        .collect()
}

fn pending_thumbnail_entry(entries: &[Entry], scene: &Scene) -> Option<usize> {
    entries.iter().position(|entry| {
        if entry.thumb.is_some() {
            return false;
        }
        let mut rendered = entry.scene.clone();
        rendered.render.max_bounces = rendered.render.max_bounces.min(3);
        &rendered == scene
    })
}

/// What the window shows the viewport on, read from `egui_display::DisplayState` by everything
/// that depends on it (the viewport canvas and its snapshots): HDR output or not, and the SDR
/// white in nits the present pass maps canvas 1.0 to. No window state yet: SDR, 100 nits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Monitor {
    pub hdr: bool,
    pub white_nits: f32,
}
impl Monitor {
    pub(crate) fn read(ctx: &egui::Context) -> Self {
        let state =
            ctx.data(|d| d.get_temp::<egui_display::DisplayState>(egui_display::state_id()));
        Self {
            hdr: state.as_ref().is_some_and(|s| s.output.is_hdr()),
            white_nits: state.as_ref().map_or(100.0, |s| s.target.white),
        }
    }
}

/// The colour of every status warning (preview, OIDN failure, out of steps).
const WARNING: Color32 = Color32::from_rgb(230, 180, 60);

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

fn to_image(t: &Frame) -> ColorImage {
    ColorImage::new(
        [t.width, t.height],
        t.pixels
            .iter()
            .map(|&p| {
                let [r, g, b, _] = p.to_le_bytes();
                Color32::from_rgb(r, g, b)
            })
            .collect(),
    )
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
    CameraRecorder,
}
impl SettingsPage {
    const ALL: [Self; 6] = [
        Self::Display,
        Self::Color,
        Self::Controls,
        Self::Fonts,
        Self::Animation,
        Self::CameraRecorder,
    ];
    fn category(self) -> egui_prefs2::Category<'static> {
        use egui_widgets_config::icons as ph;
        match self {
            Self::Display => egui_prefs2::Category::new(ph::MONITOR, "Display"),
            Self::Color => egui_prefs2::Category::new(ph::MONITOR, "Color"),
            Self::Controls => egui_prefs2::Category::new(ph::MONITOR, "Controls"),
            Self::Fonts => egui_prefs2::Category::new(ph::TEXT, "Fonts"),
            Self::Animation => egui_prefs2::Category::new(ph::MEDIA, "Animation"),
            Self::CameraRecorder => {
                egui_prefs2::Category::new(egui_widgets_config::icons::RECORD, "Camera recorder")
            }
        }
    }
    /// The page named `name` (case-insensitive tab label).
    fn named(name: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|page| page.category().label.eq_ignore_ascii_case(name))
    }
}

/// The Settings > Controls page: viewport flight preferences (`inertia` turns them into the
/// shared `cam_controls` flight settings, the one place WarpBro configures the flight rig) and
/// the mouse mapping of the slot strips.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Controls {
    look_sensitivity: f32,
    fly_speed: f32,
    /// Seconds for the translation speed to fall to 1/e once thrust stops.
    translate_decay: f32,
    /// Seconds for the rotation rate (look, roll) to fall to 1/e; also the horizon-lock
    /// levelling speed.
    rotate_decay: f32,
    /// Mouse look coasts and eases out instead of turning 1:1.
    inertial_look: bool,
    /// Horizon lock: roll angle (degrees) past which the lock flips to the next plane.
    flip_degrees: f32,
    /// Thrust multipliers while Shift (fast) / Alt (slow) are held.
    fast_multiplier: f32,
    slow_multiplier: f32,
    /// Slot strips (CamClip, colour presets): the left button stores and the right recalls,
    /// instead of the default left = recall, right = store (`hotkeys::slot_click`).
    swap_slot_buttons: bool,
}
impl Default for Controls {
    fn default() -> Self {
        Self {
            look_sensitivity: 1.0,
            fly_speed: 1.0,
            translate_decay: 0.6,
            rotate_decay: 0.25,
            inertial_look: true,
            flip_degrees: 60.0,
            fast_multiplier: 4.0,
            slow_multiplier: 0.1,
            swap_slot_buttons: false,
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
            // Base rates: look 1/3 mrad per pixel, roll 2 rad/s², thrust 2 radii/s² (cruise
            // 2 radii x translate decay per second).
            look_sensitivity: 0.001 / 3.0 * self.look_sensitivity,
            roll_sensitivity: 2.0,
            thrust_sensitivity: 2.0 * radius * self.fly_speed,
            linear_damping: 1.0 / self.translate_decay.max(0.02),
            angular_damping: 1.0 / self.rotate_decay.max(0.02),
            inertial_look: self.inertial_look,
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

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Settings {
    display: egui_display::DisplayPrefs,
    colour: crate::ocio::Sel,
    colour_presets: crate::ocio::ColourPresets,
    panel: egui_prefs2::PrefsPanelState,
    controls: Controls,
    recorder: crate::camera_recorder::Options,
    fonts: dock::Fonts,
    layout: Option<String>,
    toolbar: egui_viewport_toolbar::ToolbarState,
    camera_slots: crate::camera_slots::CameraSlots,
    layouts: egui_layout_manager::LayoutStore,
    export: crate::export::ExportSettings,
    gui_fps: u32,
    status_layout: egui_statusbar::StatusBarLayout,
    status_resizable: bool,
    attribute_metrics: egui_widgets_config::AttrMetrics,
    auto_key: bool,
    new_key: curves::Tan,
    timeline_outline_width: f32,
    file_dialogs: crate::file_dialogs::History,
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
            fonts: Default::default(),
            layout: None,
            toolbar: Default::default(),
            camera_slots: Default::default(),
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
        }
    }
}
/// A numeric Settings row: slider over `range`, Ctrl-click / row menu resets to `default`.
fn f_row(label: &str, v: f32, range: std::ops::RangeInclusive<f32>, default: f32) -> AttrField {
    AttrField::new(label, AttrValue::Float(v))
        .with_ui_options(vec![range.start().to_string(), range.end().to_string()])
        .with_default(AttrValue::Float(default))
}
/// A checkbox Settings row.
fn b_row(label: &str, v: bool, default: bool) -> AttrField {
    AttrField::new(label, AttrValue::Bool(v)).with_default(AttrValue::Bool(default))
}
/// Draws Settings `rows` as an attribute grid with the Attribute Editor's metrics and hands every
/// edit (row label, new value) to `apply`.
fn prefs_grid(
    ui: &mut egui::Ui,
    metrics: AttrMetrics,
    state: &mut AttrGridState,
    mut rows: Vec<AttrField>,
    mut apply: impl FnMut(&str, AttrValue),
) {
    struct Plain;
    impl AttrGridHooks for Plain {}
    let metrics = metrics.normalized();
    // Preferences are scalar rows with no animation controls: the editor needs one field, not the
    // attribute grid's three, and the label column keeps the room (a label never collapses to "...").
    let mut config = egui_attr_grid::AttrGridConfig {
        prefix_width: 0.0,
        action_width: 0.0,
        min_editor_width: metrics.numeric_width + metrics.component_gap + metrics.field_height,
        ..crate::world_ui::grid_config(metrics)
    };
    ui.scope(|ui| {
        metrics.apply(ui);
        config.value_box_width = Some(egui_attr_grid::value_box_width(
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
        for (label, value) in
            render_grid_with_config(ui, &mut rows, state, &HashSet::new(), &config, &mut Plain)
        {
            apply(&label, value);
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
        let scene = gallery[0].scene.clone();

        Self {
            renderer: RenderService::spawn(),
            io: crate::io_service::IoService::spawn(),
            gpu_name: "Initializing CUDA…".into(),
            frame: None,
            frame_from_cache: false,
            request: None,
            viewport_stamp: None,
            generation: 0,
            staged_frame: None,
            staged_output: None,
            layouts: Default::default(),
            export: Default::default(),
            thumb_pending: None,
            gui_fps: 60,
            origin: scene.clone(),
            last_scene: scene.clone(),
            scene: scene.clone(),
            world: crate::world::WorldEditor::new(crate::world::WorldDocument::from_scene(&scene)),
            world_ui: Default::default(),
            preview: Default::default(),
            preview_key: None,
            preview_sequence: 0,
            material_targets: Vec::new(),
            material_selection_stamp: None,
            preview_progress: None,
            evaluated_world: None,
            load_revision: 0,

            showing_preview: false,
            hdr_view: std::sync::Arc::new(std::sync::Mutex::new(egui_hdr_view::HdrView::new())),
            display: Default::default(),
            colour: crate::ocio::State::new(scene.colour.clone()),
            prefs: Default::default(),
            recorder_options: Default::default(),
            camera_recording: None,
            prefs_grids: Default::default(),
            dock: dock::default_layout(),
            panels_to_open: Vec::new(),
            toolbar: Default::default(),
            camera_slots: Default::default(),
            fonts: Default::default(),
            applied_fonts: None,
            viewport_visible: false,
            viewport_rect: None,
            config_picker: egui_file_dialog::FileDialog::new(),
            scene_picker: None,
            scene_file_path: None,
            scene_file_sequence: 0,
            scene_file_pending: None,
            templates: Vec::new(),
            templates_requested: false,
            templates_pending: false,
            persisted_settings: None,
            persisted_layout_revision: 0,
            snapshot_before: None,
            #[cfg(test)]
            settings_serializations: 0,
            #[cfg(test)]
            snapshot_clones: 0,
            status_layout: Default::default(),
            status_resizable: true,
            render_editor: Default::default(),
            gallery,
            bookmarks: load_bookmarks(),

            swatches: (0..crate::materials::PRESETS.len()).map(|_| None).collect(),
            material_gallery: Default::default(),
            tab: Tab::Gallery,
            target_spp: 1024,
            spp_per_frame: 1,
            last_change: Instant::now(),
            resolution: 1.0,
            paused: false,
            raw_view: false,
            exposure_hold: 0.0,
            show_ui: true,
            status: String::new(),
            frame_ms: 16.0,
            seed: 0,
            fly: None,
            fly_speed_scale: 1.0,
            orbit: cam_controls::HoudiniOrbit::default(),
            orbit_drag: cam_controls_egui::OrbitDragState::default(),
            orbit_written: None,
            controls: Default::default(),
            snap: std::env::var("FRAC_SNAP").ok().map(|p| {
                let spp = std::env::var("FRAC_SNAP_SPP")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(256);
                (PathBuf::from(p), spp, false)
            }),
        }
        .with_snap_preset()
        .with_settings()
    }

    fn with_settings(mut self) -> Self {
        if let Ok(data) = std::fs::read(settings_path())
            && let Ok(settings) = serde_json::from_slice::<Settings>(&data)
        {
            self.display = settings.display;
            self.prefs = settings.panel;
            self.controls = settings.controls;
            self.recorder_options = settings.recorder;
            self.toolbar = settings.toolbar;
            self.camera_slots = settings.camera_slots;
            self.layouts.store = settings.layouts;
            self.export.restore(settings.export);
            self.gui_fps = settings.gui_fps.clamp(15, 240);
            self.status_layout = settings.status_layout;
            self.status_resizable = settings.status_resizable;
            self.world_ui.attribute_metrics = settings.attribute_metrics.normalized();
            self.world_ui.auto_key = settings.auto_key;
            self.world_ui.new_key = settings.new_key;
            self.world_ui.file_dialogs = settings.file_dialogs;
            if settings.timeline_outline_width.is_finite() {
                self.world_ui.timeline_outline_width = settings.timeline_outline_width.max(80.0);
            }
            self.fonts = settings.fonts;
            if let Some(blob) = settings.layout {
                match egui_dock_layout::from_blob(&blob) {
                    Ok(layout) if dock::valid_layout(&layout) => self.dock = layout,
                    _ => self.status = "Saved layout could not be restored; using default".into(),
                }
            }
            self.colour = crate::ocio::State::new(settings.colour.clone());
            self.colour.presets = settings.colour_presets;
            self.scene.colour = settings.colour;
            self.origin.colour = self.scene.colour.clone();
        }
        if let Ok(category) = std::env::var("FRAC_SETTINGS") {
            self.panels_to_open
                .push(if category.eq_ignore_ascii_case("export") {
                    dock::Panel::Export
                } else {
                    dock::Panel::Settings
                });
            self.prefs.selected =
                SettingsPage::named(&category).unwrap_or(SettingsPage::Display) as usize;
        }
        if let Ok(before) = self
            .world
            .document
            .snapshot(f64::from(self.world_ui.playhead))
        {
            let _ = self.world.edit_snapshot(
                None,
                &before,
                &self.scene,
                f64::from(self.world_ui.playhead),
            );
        }
        self
    }

    /// Open the Settings panel on `page`.
    pub(crate) fn open_settings(&mut self, page: SettingsPage) {
        self.prefs.selected = page as usize;
        self.panels_to_open.push(dock::Panel::Settings);
    }

    /// Settings layout and category bodies adapted directly from exr-view::ui_settings.
    fn settings_ui(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        let state =
            ctx.data(|d| d.get_temp::<egui_display::DisplayState>(egui_display::state_id()));
        // Colour chooses the rendering/export target; SDR presentation has its own preview.
        self.colour.set_hdr(true);
        self.colour
            .set_filterable(gpu_info::shared_device().is_some_and(|g| {
                g.device
                    .features()
                    .contains(wgpu::Features::FLOAT32_FILTERABLE)
            }));
        let mut browse = false;
        let mut changed = false;
        let mut reset = false;
        let mut category = self.prefs.selected;
        let mut destination = None;
        let mut prefs = std::mem::take(&mut self.prefs);
        ui.push_id("settings", |ui| {
            let categories = SettingsPage::ALL.map(SettingsPage::category);
            egui_prefs2::draw(ui, &mut prefs, &categories, |ui, idx| {
                category = idx;
                match SettingsPage::ALL.get(idx).copied().unwrap_or(SettingsPage::Display) {
                    SettingsPage::Display => {
                        egui_display::settings_ui(ui, &mut self.display, state.as_ref());
                        if state.as_ref().is_some_and(|s| !s.available.iter().any(|o| o.is_hdr())) {
                            ui.label("This window surface offers SDR only. PQ/HDR targets in Color still render and export HDR; the screen uses an SDR preview.");
                        }
                        ui.add(egui::DragValue::new(&mut self.gui_fps).range(15..=240).suffix(" GUI FPS"));
                        ui.add_space(8.0);
                        if ui.button("Colour management & monitor presets…").clicked() { destination = Some(SettingsPage::Color); }
                    }
                    SettingsPage::Color => {
                        egui_prefs2::section_header(ui, "Colour management");
                        if let Some(state) = &state { ui.label(format!("Window output: {}.", state.output.label())); }
                        ui.label("Monitor presets choose rendering and export. PQ/HDR remains available on SDR screens using an SDR preview. HDR 1000 nits is the rendering peak; SDR reference white controls UI brightness.");
                        if ui.button("Display output & reference white…").clicked() { destination = Some(SettingsPage::Display); }
                        ui.add_space(8.0);
                        changed = self.colour.ui(ui, &mut browse, self.controls.swap_slot_buttons);
                    }
                    SettingsPage::Fonts => self.fonts_ui(ui),
                    SettingsPage::CameraRecorder => self.camera_recorder_ui(ui),
                    SettingsPage::Animation => {
                        egui_prefs2::section_header(ui, "Keys");
                        ui.horizontal(|ui| {
                            ui.label("New key type")
                                .on_hover_text("Interpolation of every key created by Key, Auto Key and edits of animated values.");
                            for tan in [curves::Tan::Linear, curves::Tan::Smooth] {
                                ui.selectable_value(&mut self.world_ui.new_key, tan, tan.label());
                            }
                        });
                    }
                    SettingsPage::Controls => {
                        egui_prefs2::section_header(ui, "Camera controls");
                        let am = self.world_ui.attribute_metrics;
                        let c = &mut self.controls;
                        prefs_grid(ui, am, &mut self.prefs_grids[0], vec![
                            f_row("Mouse sensitivity", c.look_sensitivity, 0.1..=5.0, 1.0),
                            f_row("Flight speed ×", c.fly_speed, 0.02..=50.0, 1.0),
                            f_row("Translate decay, s", c.translate_decay, 0.02..=5.0, 0.6),
                            f_row("Rotate decay, s", c.rotate_decay, 0.02..=5.0, 0.25),
                            b_row("Inertial look", c.inertial_look, true),
                            f_row("Horizon flip, °", c.flip_degrees, 20.0..=85.0, 60.0),
                            f_row("Shift fast ×", c.fast_multiplier, 1.0..=20.0, 4.0),
                            f_row("Alt slow ×", c.slow_multiplier, 0.01..=1.0, 0.1),
                        ], |label, v| match (label, v) {
                            ("Mouse sensitivity", AttrValue::Float(v)) => c.look_sensitivity = v,
                            ("Flight speed ×", AttrValue::Float(v)) => c.fly_speed = v,
                            ("Translate decay, s", AttrValue::Float(v)) => c.translate_decay = v,
                            ("Rotate decay, s", AttrValue::Float(v)) => c.rotate_decay = v,
                            ("Inertial look", AttrValue::Bool(v)) => c.inertial_look = v,
                            ("Horizon flip, °", AttrValue::Float(v)) => c.flip_degrees = v,
                            ("Shift fast ×", AttrValue::Float(v)) => c.fast_multiplier = v,
                            ("Alt slow ×", AttrValue::Float(v)) => c.slow_multiplier = v,
                            _ => {}
                        });
                        egui_prefs2::section_header(ui, "Slot buttons");
                        prefs_grid(ui, am, &mut self.prefs_grids[1], vec![
                            b_row("Swap copy/paste mouse buttons", c.swap_slot_buttons, false),
                        ], |_, v| {
                            if let AttrValue::Bool(v) = v { c.swap_slot_buttons = v }
                        });
                        ui.label(format!(
                            "CamClip and colour presets · {}",
                            crate::hotkeys::slot_hint(self.controls.swap_slot_buttons, "copy", "paste")
                        ));
                        egui_prefs2::section_header(ui, "Attribute controls");
                        let defaults = AttrMetrics::default();
                        let metrics = &mut self.world_ui.attribute_metrics;
                        let m = *metrics;
                        prefs_grid(ui, am, &mut self.prefs_grids[2], vec![
                            f_row("Field height", m.field_height, AttrMetrics::FIELD_HEIGHT_RANGE, defaults.field_height),
                            f_row("Numeric field width", m.numeric_width, AttrMetrics::NUMERIC_WIDTH_RANGE, defaults.numeric_width),
                            f_row("Icon size", m.icon_side, AttrMetrics::ICON_SIDE_RANGE, defaults.icon_side),
                            f_row("Row spacing", m.row_gap, AttrMetrics::ROW_GAP_RANGE, defaults.row_gap),
                            f_row("Component spacing", m.component_gap, AttrMetrics::COMPONENT_GAP_RANGE, defaults.component_gap),
                        ], |label, v| {
                            let AttrValue::Float(v) = v else { return };
                            match label {
                                "Field height" => metrics.field_height = v,
                                "Numeric field width" => metrics.numeric_width = v,
                                "Icon size" => metrics.icon_side = v,
                                "Row spacing" => metrics.row_gap = v,
                                "Component spacing" => metrics.component_gap = v,
                                _ => {}
                            }
                        });
                        *metrics = metrics.normalized();
                        ui.label("Attribute Editor, Timeline and Render Settings share these sizes.");
                        ui.label("RMB: fly · wheel: flight speed · `: horizon/free flight");
                        ui.label("H: restore camera · F: frame bounds (without RMB)");
                    }
                }
            }, Some(|| reset = true));
        });
        self.prefs = prefs;
        if reset {
            if category == 0 {
                self.display = Default::default();
            } else if category == 1 {
                self.colour.sel = crate::color::default_selection();
                self.colour.reload();
                changed = true;
            } else if category == 3 {
                self.fonts = Default::default();
            } else if category == SettingsPage::CameraRecorder as usize {
                self.recorder_options = Default::default();
            } else if category == SettingsPage::Animation as usize {
                self.world_ui.new_key = curves::Tan::Smooth;
            } else {
                self.controls = Default::default();
                self.world_ui.attribute_metrics = Default::default();
            }
        }
        if let Some(page) = destination {
            self.prefs.selected = page as usize;
        }
        if browse {
            self.config_picker = self.world_ui.file_dialogs.prepare(
                egui_file_dialog::FileDialog::new()
                    .add_file_filter_extensions("OCIO config", vec!["ocio"]),
                crate::file_dialogs::OCIO,
                std::path::Path::new(&self.colour.sel.config).parent(),
                "OCIO config",
            );
            self.config_picker.pick_file();
        }
        self.apply_colour_change(changed);
    }

    fn camera_recorder_ui(&mut self, ui: &mut egui::Ui) {
        use crate::camera_recorder::{Recording, Timing};
        egui_prefs2::section_header(ui, "Capture channels");
        ui.add_enabled_ui(self.camera_recording.is_none(), |ui| {
            ui.checkbox(
                &mut self.recorder_options.transform,
                "Translate / Rotate / Scale",
            );
            ui.checkbox(&mut self.recorder_options.focus, "Focus");
            ui.checkbox(&mut self.recorder_options.zoom, "Zoom");
            ui.checkbox(&mut self.recorder_options.f_number, "f-number");
            egui_prefs2::section_header(ui, "Timing");
            ui.radio_value(
                &mut self.recorder_options.timing,
                Timing::KeepSpeed,
                "Keep speed - extend work area",
            );
            ui.radio_value(
                &mut self.recorder_options.timing,
                Timing::FitWorkArea,
                "Fit to work area",
            );
            ui.horizontal(|ui| {
                ui.label("Reduction error");
                ui.add(
                    egui::DragValue::new(&mut self.recorder_options.tolerance)
                        .range(0.0..=1.0)
                        .speed(0.0001),
                );
            })
            .response
            .on_hover_text(
                "Maximum absolute error at captured samples, in each channel's authored units.",
            );
        });
        if let Some((start, recording)) = &self.camera_recording {
            ui.label(format!(
                "Recording {:.1} s - {} samples",
                start.elapsed().as_secs_f64(),
                recording.samples.len()
            ));
            ui.horizontal(|ui| {
                ui.label("Focus");
                ui.add(
                    egui::DragValue::new(&mut self.scene.camera.focus_distance)
                        .range(0.0..=f32::MAX)
                        .speed(0.01),
                );
            });
            ui.horizontal(|ui| {
                ui.label("Vertical FOV, degrees");
                ui.add(
                    egui::DragValue::new(&mut self.scene.camera.fov_y_degrees).range(1.0..=179.0),
                );
            });
            ui.horizontal(|ui| {
                ui.label("f-number (0 = pinhole)");
                ui.add(
                    egui::DragValue::new(&mut self.scene.camera.f_number)
                        .range(0.0..=64.0)
                        .speed(0.1),
                );
            });
            if ui.button("Stop and apply").clicked() {
                let (start, mut recording) = self.camera_recording.take().unwrap();
                let result = recording
                    .capture(start.elapsed().as_secs_f64(), self.scene.camera)
                    .and_then(|_| self.world.record_camera(&recording));
                self.status = match result {
                    Ok((samples, keys, error, last)) => {
                        self.world_ui.seek(last);
                        format!(
                            "Camera recorded: {samples} samples -> {keys} keys (sample error {error:.6})"
                        )
                    }
                    Err(error) => format!("Camera recording failed: {error}"),
                };
                self.fly = None;
                self.orbit = Default::default();
                self.orbit_written = None;
                self.evaluated_world = None;
            }
            if ui.button("Cancel recording").clicked() {
                self.camera_recording = None;
                self.fly = None;
                self.orbit = Default::default();
                self.orbit_written = None;
                self.evaluated_world = None;
                self.status = "Camera recording cancelled".into();
            }
        } else if ui.button("Start recording").clicked() {
            self.preview.cancel();
            self.preview_key = None;
            self.world_ui.playing = false;
            self.world.finish_edit();
            self.evaluated_world = None;
            if let Err(error) = self.refresh_scene() {
                self.status = error;
                return;
            }
            match Recording::new(
                &self.world.document,
                self.world.revision(),
                f64::from(self.world_ui.playhead),
                &self.scene,
                self.recorder_options,
            ) {
                Ok(recording) => {
                    self.camera_recording = Some((Instant::now(), recording));
                    self.status = "Recording camera navigation".into();
                }
                Err(error) => self.status = error,
            }
        }
        ui.label("Selected channels replace keys in the captured interval; keys outside it remain. One undo restores the capture.");
    }

    fn apply_colour_change(&mut self, changed: bool) {
        if changed {
            self.scene.render.reinhard = false;
            self.scene.colour = self.colour.sel.clone();
            let _ = self.renderer.try_command(Command::ReloadColour);
        }
    }

    fn save_settings(&mut self, ctx: &egui::Context) {
        self.config_picker.update(ctx);
        self.world_ui
            .file_dialogs
            .observe(crate::file_dialogs::OCIO, &self.config_picker);
        let mut changed = false;
        if let Some(path) = self.config_picker.take_picked() {
            self.colour.set_config(path.to_string_lossy().into_owned());
            changed = true;
        }
        if changed {
            let before = self.scene.clone();
            self.scene.render.reinhard = false;
            self.scene.colour = self.colour.sel.clone();
            if let Err(error) = self.world.edit_snapshot(
                None,
                &before,
                &self.scene,
                f64::from(self.world_ui.playhead),
            ) {
                self.status = error;
            }
            let _ = self.renderer.try_command(Command::ReloadColour);
        }
        match self.changed_settings_json(ctx) {
            Ok(Some(json)) => {
                self.io.settings(settings_path(), json);
            }
            Ok(None) => {}
            Err(error) => self.status = error,
        }
    }

    fn settings_match(&self, saved: &Settings) -> bool {
        let export = self.export.settings();
        let previous = &saved.export;
        saved.display == self.display
            && saved.colour == self.scene.colour
            && saved.colour_presets == self.colour.presets
            && saved.panel == self.prefs
            && saved.controls == self.controls
            && saved.recorder == self.recorder_options
            && saved.toolbar == self.toolbar
            && saved.camera_slots == self.camera_slots
            && saved.fonts == self.fonts
            && saved.layouts == self.layouts.store
            && saved.gui_fps == self.gui_fps
            && saved.status_layout == self.status_layout
            && saved.status_resizable == self.status_resizable
            && saved.attribute_metrics == self.world_ui.attribute_metrics
            && saved.auto_key == self.world_ui.auto_key
            && saved.new_key == self.world_ui.new_key
            && saved.file_dialogs == self.world_ui.file_dialogs
            && saved.timeline_outline_width == self.world_ui.timeline_outline_width
            // Every field, so a new setting is saved without being listed here.
            && export == previous
    }

    fn changed_settings_json(&mut self, ctx: &egui::Context) -> Result<Option<String>, String> {
        self.layouts
            .cache
            .refresh(&self.dock, ctx)
            .map_err(|error| format!("Layout save failed: {error}"))?;
        if self.persisted_layout_revision == self.layouts.cache.revision
            && self
                .persisted_settings
                .as_ref()
                .is_some_and(|saved| self.settings_match(saved))
        {
            return Ok(None);
        }
        let layout = self.layouts.cache.blob.clone();
        let settings = Settings {
            display: self.display,
            colour: self.scene.colour.clone(),
            colour_presets: self.colour.presets.clone(),
            panel: self.prefs.clone(),
            controls: self.controls,
            recorder: self.recorder_options,
            layout,
            toolbar: self.toolbar,
            camera_slots: self.camera_slots,
            fonts: self.fonts.clone(),
            layouts: self.layouts.store.clone(),
            export: self.export.settings().clone(),
            gui_fps: self.gui_fps,
            status_layout: self.status_layout.clone(),
            status_resizable: self.status_resizable,
            attribute_metrics: self.world_ui.attribute_metrics,
            auto_key: self.world_ui.auto_key,
            new_key: self.world_ui.new_key,
            file_dialogs: self.world_ui.file_dialogs.clone(),
            timeline_outline_width: self.world_ui.timeline_outline_width,
        };
        let json = serde_json::to_string_pretty(&settings).map_err(|error| error.to_string())?;
        #[cfg(test)]
        {
            self.settings_serializations += 1;
        }
        self.persisted_settings = Some(settings);
        self.persisted_layout_revision = self.layouts.cache.revision;
        Ok(Some(json))
    }

    fn with_snap_preset(mut self) -> Self {
        if let Some(i) = std::env::var("FRAC_SNAP_PRESET")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            && let Some(e) = self.gallery.get(i)
        {
            let scene = e.scene.clone();
            self.load(scene);
        }
        if let Ok(name) = std::env::var("FRAC_SNAP_MATERIAL")
            && let Some(p) = crate::materials::PRESETS.iter().find(|p| p.name() == name)
        {
            let before = self.scene.clone();
            p.apply(&mut self.scene.material);
            let _ = self.world.edit_snapshot(
                self.world.selection,
                &before,
                &self.scene,
                f64::from(self.world_ui.playhead),
            );
        }
        if std::env::var("FRAC_SNAP_TAB").is_ok_and(|t| t == "materials") {
            self.tab = Tab::Materials;
            self.panels_to_open.push(dock::Panel::Materials);
        }
        self
    }

    fn handle_snap(&mut self, ctx: &egui::Context, thumbs_pending: bool) {
        let Some((_path, spp, requested)) = self.snap.clone() else {
            return;
        };
        let wait_thumbs = !std::env::var("FRAC_SNAP_WAIT_THUMBS").is_ok_and(|value| value == "0");
        let ready = (!wait_thumbs || !thumbs_pending)
            && !self.showing_preview
            && self.frame.as_ref().is_some_and(|t| t.complete(spp));
        if ready && !requested {
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(Default::default()));
            self.snap = self.snap.take().map(|(path, spp, _)| (path, spp, true));
        }
        ctx.request_repaint();
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
    fn scene_file_dialog(&mut self, save: bool) {
        self.scene_action_dialog(if save {
            crate::templates::FileAction::Save
        } else {
            crate::templates::FileAction::Open
        });
    }
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

    fn submit_scene_file(&mut self, path: PathBuf, save: bool) {
        self.submit_scene_action(
            path,
            if save {
                crate::templates::FileAction::Save
            } else {
                crate::templates::FileAction::Open
            },
        );
    }
    fn submit_scene_action(&mut self, mut path: PathBuf, action: crate::templates::FileAction) {
        let save = action.save();
        self.world.finish_edit();
        if save && path.extension().is_none() {
            path.set_extension(crate::fs_name::SCENE_SUFFIX);
        }
        let id = self.scene_file_sequence.wrapping_add(1);
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
                self.scene_file_sequence = id;
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
        let picked = if let Some((save, picker)) = &mut self.scene_picker {
            picker.update(ctx);
            self.world_ui
                .file_dialogs
                .observe(save.history_key(), picker);
            picker.take_picked().map(|path| (*save, path))
        } else {
            None
        };
        if let Some((save, path)) = picked {
            self.scene_picker = None;
            self.submit_scene_action(path, save);
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
                self.load(*scene);
                if !action.template() {
                    self.scene_file_path = Some(event.path.clone());
                }
                self.status = format!("Opened {}", event.path.display());
            }
            Ok(None) => {
                if !action.template() {
                    self.scene_file_path = Some(event.path.clone());
                }
                if action.template() {
                    self.refresh_templates();
                }
                self.status = format!("Saved {}", event.path.display());
            }
            Err(error) => self.status = format!("Scene file {}: {error}", event.path.display()),
        }
    }

    fn load(&mut self, mut scene: Scene) {
        self.camera_recording = None;
        self.preview.cancel();
        self.preview_key = None;
        self.scene_file_path = None;
        self.scene_file_pending = None;
        self.evaluated_world = None;
        self.load_revision = self.load_revision.wrapping_add(1);
        let document = crate::world::WorldDocument::from_scene(&scene);
        self.world_ui.reset();
        self.world_ui.seek(document.first);
        self.world = crate::world::WorldEditor::new(document.clone());
        self.material_gallery.invalidate();
        self.origin = scene.clone();
        scene.document = Some(Box::new(document));
        self.colour.sel = scene.colour.clone();
        self.colour.rebuild();
        self.scene = self
            .world
            .document
            .snapshot(f64::from(self.world_ui.playhead))
            .unwrap_or(scene);
    }

    fn render_one_thumbnail(&mut self, _ctx: &egui::Context) -> bool {
        if let Err(error) = self
            .material_gallery
            .refresh(&self.world, f64::from(self.world_ui.playhead))
        {
            self.status = error;
        }
        if self.thumb_pending.is_some() {
            return true;
        }
        let material_request = self.material_gallery.next_thumbnail();
        let item = material_request
            .map(|request| {
                let _node = request.node;
                (
                    request.id,
                    material_preview_scene(request.material),
                    SWATCH,
                    SWATCH,
                    SWATCH_SPP,
                )
            })
            .or_else(|| {
                self.gallery
                    .iter()
                    .enumerate()
                    .find(|(_, e)| e.thumb.is_none())
                    .map(|(i, e)| (i as u64, e.scene.clone(), THUMB_W, THUMB_H, THUMB_SPP))
                    .or_else(|| {
                        self.bookmarks
                            .iter()
                            .enumerate()
                            .find(|(_, e)| e.thumb.is_none())
                            .map(|(i, e)| {
                                (
                                    1_000_000 + i as u64,
                                    e.scene.clone(),
                                    THUMB_W,
                                    THUMB_H,
                                    THUMB_SPP,
                                )
                            })
                    })
                    .or_else(|| {
                        self.swatches.iter().position(Option::is_none).map(|i| {
                            (
                                2_000_000 + i as u64,
                                swatch_scene(&crate::materials::PRESETS[i]),
                                SWATCH,
                                SWATCH,
                                SWATCH_SPP,
                            )
                        })
                    })
            });
        let Some((id, mut scene, width, height, spp)) = item else {
            return false;
        };
        if id < 2_000_000 {
            scene.render.max_bounces = scene.render.max_bounces.min(3);
        }
        if self
            .renderer
            .try_command(Command::Thumbnail {
                id,
                scene: scene.clone(),
                width,
                height,
                spp,
            })
            .is_ok()
        {
            self.thumb_pending = Some((id, scene));
        } else if id & crate::material_gallery::THUMBNAIL_ID_MASK != 0 {
            self.material_gallery.retry_thumbnail(id);
        }
        true
    }

    /// Selection and assignment are separate: cards refer to authored node UUIDs.
    fn select_material_node(&mut self, id: crate::world::NodeId) {
        self.remember_material_targets();
        self.world.finish_edit();
        self.world.selection = Some(id);
        self.world.selected.clear();
        self.world.selected.push(id);
        self.panels_to_open.push(dock::Panel::Inspector);
    }

    /// Selecting a material changes the editor's node selection, but keeps the
    /// last consumer selection as an explicit assignment context. Other node
    /// selections clear it; document changes cannot leak targets across projects.
    fn remember_material_targets(&mut self) {
        use std::hash::{Hash, Hasher};
        let mut hash = std::collections::hash_map::DefaultHasher::new();
        self.world.selected.hash(&mut hash);
        let stamp = (
            playa_graph::NodeId(self.world.document.graph.id),
            self.world.revision(),
            self.world.selection,
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
        let selected = if self.world.selected.is_empty() {
            self.world.selection.as_slice()
        } else {
            &self.world.selected
        };
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

    fn material_assignment_targets(&self) -> &[crate::world::NodeId] {
        &self.material_targets
    }

    /// The objects an "Assign to …" action addresses.
    fn assign_targets(&mut self, to: AssignTo) -> Vec<crate::world::NodeId> {
        match to {
            AssignTo::Selected => {
                self.remember_material_targets();
                self.material_assignment_targets().to_vec()
            }
            AssignTo::AllFractals => self.world.document.material_consumers(),
        }
    }

    /// Assign a work-area Material node: one Batch, one undo step.
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

    fn assign_status(result: Result<(), String>, to: AssignTo) -> String {
        match (result, to) {
            (Ok(()), AssignTo::Selected) => "Material assigned to selected objects".into(),
            (Ok(()), AssignTo::AllFractals) => "Material assigned to all fractals".into(),
            (Err(error), _) => error,
        }
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
                self.panels_to_open.push(dock::Panel::MaterialLibrary);
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
        let selected_targets = {
            self.remember_material_targets();
            !self.material_assignment_targets().is_empty()
        };
        let any_fractal = !self.world.document.material_consumers().is_empty();
        let mut assign = None;
        let mut refresh = None;
        let mut apply_preset = None;
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
                                        if self.world.selection == Some(entry.id) {
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
                                    command = Some(WorldCommand::Duplicate(vec![entry.id]));
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
            self.material_gallery.invalidate_thumbnail(id);
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
            self.select_material_node(id);
        }
        if let Some((id, to)) = assign {
            let result = self.assign_gallery_material(id, to);
            self.status = Self::assign_status(result, to);
        }
        if let Some(command) = command
            && let Err(error) = self.world.execute(command)
        {
            self.status = error;
        }
        if create_default || create.is_some() {
            let mut material = Material::default();
            let name = if let Some(index) = create {
                PRESETS[index].apply(&mut material);
                PRESETS[index].name()
            } else {
                "Material"
            };
            match self.world.execute(WorldCommand::CreateMaterial {
                material,
                name: name.into(),
            }) {
                Ok(()) => {
                    if let Some(id) = self.world.selection {
                        self.select_material_node(id);
                    }
                }
                Err(error) => self.status = error,
            }
        }
    }

    /// Presets are reusable sources. Only this explicit action creates a
    /// workspace Material node; creation never changes object assignments.
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
                if let Some(id) = self.world.selection {
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
        let selected_targets = {
            self.remember_material_targets();
            !self.material_assignment_targets().is_empty()
        };
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
            self.add_library_material(index);
        }
        if let Some((index, to)) = assign {
            let result = self.assign_library_material(index, to);
            self.status = Self::assign_status(result, to);
        }
    }

    fn save_bookmark(&mut self) {
        let dir = crate::warpbro_dir().join("bookmarks");
        let path = dir.join(format!(
            "{}-{}.json",
            now_stamp(),
            crate::fs_name::stem(&self.scene.name)
        ));
        match serde_json::to_string_pretty(&self.world.document)
            .map_err(|e| e.to_string())
            .and_then(|text| {
                self.io.send(crate::io_service::Command::Write {
                    path: path.clone(),
                    text,
                })
            }) {
            Ok(()) => {
                self.bookmarks.push(Entry {
                    scene: self.frozen_world_scene(),
                    thumb: None,
                    path: Some(path),
                });
                self.tab = Tab::Bookmarks;
                self.status = "Saving bookmark…".into();
            }
            Err(e) => self.status = e,
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
                self.save_frame(monitor, file);
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
        let sdr_white_nits = if monitor.hdr {
            monitor.white_nits
        } else {
            crate::color::BT2408_SDR_WHITE_NITS
        };
        self.status = match self.io.send(crate::io_service::Command::SaveFrame {
            frame,
            path,
            file,
            sdr_white_nits,
        }) {
            Ok(()) => "Saving image…".into(),
            Err(e) => e,
        };
    }
    fn step_viewport(&mut self, w: usize, h: usize, output_hdr: bool, white_nits: f32) {
        let mut snapshot = self.scene.clone();
        snapshot.animation = Default::default();
        let mut traced = snapshot.clone();
        traced.render.exposure_stops = self.last_scene.render.exposure_stops;
        traced.render.saturation = self.last_scene.render.saturation;
        traced.render.reinhard = self.last_scene.render.reinhard;
        traced.render.denoise = self.last_scene.render.denoise;
        traced.colour = self.last_scene.colour.clone();
        if traced != self.last_scene {
            self.last_change = Instant::now();
        }
        self.last_scene = snapshot.clone();
        if self.request.as_ref().is_none_or(|r| {
            r.scene != snapshot
                || r.width != w
                || r.height != h
                || r.seed != self.seed
                || r.raw != self.raw_view
                || r.target_spp != self.target_spp
                || r.output_hdr != output_hdr
                || r.white_nits != white_nits
        }) {
            self.generation = self.generation.wrapping_add(1);
        }
        let req = ViewportRequest {
            active: true,
            generation: self.generation,
            scene: snapshot,
            width: w,
            height: h,
            target_spp: self.target_spp,
            paused: self.paused,
            raw: self.raw_view,
            interactive: self.world_ui.playing
                || self.last_change.elapsed().as_secs_f32() < PREVIEW_HOLD_S,
            seed: self.seed,
            output_hdr,
            white_nits,
        };
        self.viewport_stamp = Some((
            req.generation,
            playa_graph::NodeId(self.world.document.graph.id),
            self.world.revision(),
            self.world_ui.playhead,
        ));
        self.renderer.request_viewport(req.clone());
        self.request = Some(req);
    }
    fn poll_events(&mut self, ctx: &egui::Context) {
        while let Some(result) = self.io.poll_templates() {
            self.templates_pending = false;
            match result {
                Ok(entries) => self.templates = entries,
                Err(error) => self.status = format!("Templates: {error}"),
            }
            ctx.request_repaint();
        }
        if let Some(frame) = self.renderer.take_latest_frame()
            && !self.preview.displaying_preview()
            && !(self.frame_from_cache && self.preview.position().is_some())
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
            self.export.handle(&event, &self.renderer);
            match event {
                RenderEvent::Ready { name } => self.gpu_name = name,
                RenderEvent::Error(error) => {
                    if let Some((id, _)) = self.thumb_pending.take() {
                        self.material_gallery.fail_thumbnail(id, error.clone());
                    }
                    self.status = error;
                }
                RenderEvent::Thumbnail { id, frame } => {
                    let pending = if self
                        .thumb_pending
                        .as_ref()
                        .is_some_and(|(pending_id, _)| *pending_id == id)
                    {
                        self.thumb_pending.take()
                    } else {
                        None
                    };
                    let material_thumb = id & crate::material_gallery::THUMBNAIL_ID_MASK != 0;
                    if material_thumb {
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
                        if let Some(error) = &frame.colour_error {
                            self.material_gallery.fail_thumbnail(id, error.clone());
                            self.status = error.clone();
                            continue;
                        }
                    }
                    let texture = ctx.load_texture(
                        format!("thumb-{id}"),
                        to_image(&frame),
                        TextureOptions::LINEAR,
                    );
                    if material_thumb {
                        self.material_gallery.accept_thumbnail(id, texture);
                    } else if id >= 2_000_000 {
                        if let Some(slot) = self.swatches.get_mut((id - 2_000_000) as usize) {
                            *slot = Some(texture);
                        }
                    } else {
                        let entries = if id >= 1_000_000 {
                            &mut self.bookmarks
                        } else {
                            &mut self.gallery
                        };
                        if let Some((pending_id, scene)) = pending
                            && pending_id == id
                            && let Some(index) = pending_thumbnail_entry(entries, &scene)
                        {
                            entries[index].thumb = Some(texture);
                        }
                    }
                }
                _ => {}
            }
        }
        while let Some(event) = self.io.poll_scene() {
            self.scene_file_event(event);
        }
        while let Some(result) = self.io.poll() {
            self.status = result.unwrap_or_else(|e| e);
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
        let selected = if self.world.selected.is_empty() {
            self.world.selection.into_iter().collect()
        } else {
            self.world.selected.clone()
        };
        if hotkeys::consume(ctx, Scope::Global, Hotkey::Duplicate) && !selected.is_empty() {
            self.world.finish_edit();
            let count = selected.len();
            self.status = match self
                .world
                .execute(crate::world::WorldCommand::Duplicate(selected.clone()))
            {
                Ok(()) => format!("Duplicated {count} node(s)"),
                Err(error) => error,
            };
        }
        if hotkeys::take_copy(ctx, !selected.is_empty()) {
            self.status = match self.world.document.copy_fragment(&selected) {
                Ok(text) => {
                    ctx.copy_text(text);
                    format!("Copied {} node(s)", selected.len())
                }
                Err(error) => error,
            };
        }
        if let Some(text) =
            hotkeys::take_paste(ctx, |text| crate::world::parse_clipboard(text).is_some())
        {
            let unresolved = crate::world::parse_clipboard(&text)
                .map(|nodes| self.world.document.unresolved_references(&nodes))
                .unwrap_or_default();
            self.world.finish_edit();
            self.status = match self.world.execute(crate::world::WorldCommand::Paste(text)) {
                Ok(()) if unresolved.is_empty() => {
                    format!("Pasted {} node(s)", self.world.selected.len())
                }
                Ok(()) => {
                    let materials = unresolved
                        .iter()
                        .filter(|(_, field)| *field == "material")
                        .count();
                    let parents = unresolved.len() - materials;
                    format!(
                        "Pasted {} node(s); not in this scene, cleared: {materials} material and {parents} parent reference(s)",
                        self.world.selected.len()
                    )
                }
                Err(error) => error,
            };
        }
    }

    fn export_ui(&mut self, ui: &mut egui::Ui) {
        let timeline = (
            self.world.document.first,
            self.world.document.last,
            self.world.document.fps,
            self.world_ui.playhead,
        );
        let scene = &self.scene;
        let world = &mut self.world;
        let ocio = self.colour.config().ok();
        self.export.ui(
            ui,
            timeline,
            &self.renderer,
            ocio,
            &self.scene.colour,
            || {
                world.finish_edit();
                freeze_world_scene(scene, &world.document)
            },
        );
    }

    /// Unreal-style flight: hold RMB in the viewport, mouse looks, WASD moves, R/Space up, C down,
    /// Q/E roll, Shift fast, Alt slow, the wheel scales the speed. Integrated by cam-controls `SpaceFlight` (thrust,
    /// inertia, damping); on release the orbit pivot is placed in front of the camera at the
    /// current orbit distance, so orbiting continues from where you flew.
    fn fly_camera(&mut self, ui: &egui::Ui, resp: &egui::Response) {
        use crate::hotkeys::{self, Command as Hotkey, Scope};
        use cam_controls::CameraIntent;
        use glam::Vec3;
        hotkeys::register(ui, Scope::Viewport, resp.rect);
        if hotkeys::consume(ui.ctx(), Scope::Viewport, Hotkey::Flight) {
            self.toggle_flight_mode();
        }
        let held = resp.is_pointer_button_down_on()
            && ui.input(|i| i.focused && i.pointer.secondary_down());
        if hotkeys::consume(ui.ctx(), Scope::Viewport, Hotkey::Home) {
            self.scene.camera = self.origin.camera;
            self.fly = None;
            ui.ctx()
                .send_viewport_cmd(egui::ViewportCommand::CursorGrab(egui::CursorGrab::None));
            self.status = "Camera restored".into();
            return;
        }
        if !held && hotkeys::consume(ui.ctx(), Scope::Viewport, Hotkey::Fit) {
            self.frame_camera(
                resp.rect.width().max(1.0) as u32,
                resp.rect.height().max(1.0) as u32,
            );
            return;
        }
        if !held && hotkeys::consume(ui.ctx(), Scope::Viewport, Hotkey::Play) {
            self.world_ui.playing = !self.world_ui.playing;
        }
        let radius = self
            .scene
            .camera_reference
            .unwrap_or(self.scene.formula.framing_radius());
        if held && self.fly.is_none() {
            self.start_flight();
        }
        let cam = &mut self.scene.camera;
        let dist = cam.distance * radius;
        let lock = (!cam.free_flight).then_some(Vec3::Y);
        let Some(fly) = &mut self.fly else { return };
        // Settings act live; the toolbar mode switch engages or releases the lock mid-flight.
        fly.inertia = self.controls.inertia(radius);
        if fly.horizon_lock().is_some() != lock.is_some() && !fly.set_horizon_lock(lock) {
            self.status = "Horizon lock needs a view that is not straight up or down".into();
        }
        let viewport = cam_viewport::ViewportSize::new(
            resp.rect.width().max(1.0) as u32,
            resp.rect.height().max(1.0) as u32,
        );
        let dt = ui.input(|i| i.stable_dt).clamp(1.0e-4, 0.1);
        if held {
            ui.ctx().set_cursor_icon(egui::CursorIcon::None);
            ui.ctx()
                .send_viewport_cmd(egui::ViewportCommand::CursorGrab(egui::CursorGrab::Locked));
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                self.controls.fly_speed =
                    (self.controls.fly_speed * (scroll * 0.003).exp()).clamp(0.02, 50.0);
                self.status = format!("Flight speed ×{:.2}", self.controls.fly_speed);
            }
            // WASD / R-Space-C thrust, Q/E roll, Shift / Alt speed, mouse look: the shared bindings.
            let intents = cam_controls_egui::gather_fly_intents(
                ui,
                &fly.inertia,
                &mut self.fly_speed_scale,
                true,
                // Space is free in flight: Play only takes it while RMB is up.
                true,
                resp.rect,
                resp.rect.center(),
            );
            for intent in intents {
                fly.apply_intent(intent, viewport);
            }
        } else {
            // Released: drop held axes and modifiers; the momentum coasts and damps out.
            ui.ctx()
                .send_viewport_cmd(egui::ViewportCommand::CursorGrab(egui::CursorGrab::None));
            for intent in [
                CameraIntent::Thrust {
                    forward: 0.0,
                    right: 0.0,
                    up: 0.0,
                },
                CameraIntent::Roll { d: 0.0 },
                CameraIntent::SpeedScale(1.0),
            ] {
                fly.apply_intent(intent, viewport);
            }
            self.fly_speed_scale = 1.0;
        }
        let moving = fly.update_dynamics(dt) || fly.has_motion();

        // The authored camera follows the flight pose (Euler YXZ keeps any roll).
        let pose = fly.pose();
        let (yaw, pitch, roll) = pose.orientation.to_euler(glam::EulerRot::YXZ);
        cam.yaw_degrees = yaw.to_degrees();
        cam.pitch_degrees = -pitch.to_degrees();
        cam.roll_degrees = roll.to_degrees();
        let target = pose.eye + pose.forward() * dist;
        cam.target = [target.x, target.y, target.z];
        if !held && !moving {
            self.fly = None;
        }
        ui.ctx().request_repaint();
    }

    /// Houdini orbit through the shared rig: LMB tumbles about world up (Shift: snapped to the
    /// views along the world axes), MMB pans, the wheel zooms toward the target, and released
    /// drags coast. The camera roll is an overlay the
    /// turntable leaves alone (as do pan and zoom); pan turns the cursor motion by it so the
    /// view follows the cursor.
    fn orbit_camera(&mut self, ui: &egui::Ui, resp: &egui::Response, over_toolbar: bool) {
        use cam_controls::{CameraIntent, OrbitPose};
        use glam::Vec3;
        let radius = self
            .scene
            .camera_reference
            .unwrap_or(self.scene.formula.framing_radius());
        let cam = &mut self.scene.camera;
        let rig = &mut self.orbit.rig;
        if rig.has_inertia() && self.orbit_written != Some(*cam) {
            rig.stop_inertia();
        }
        let nav = self
            .controls
            .navigation(cam.fov_y_degrees.to_radians(), resp.rect.height());
        rig.apply_navigation(nav);
        rig.pitch_limit = 89.0_f32.to_radians();
        rig.projection.distance_min = 0.05 * radius;
        rig.projection.distance_max = f32::MAX;
        // Seeded from yaw / pitch, not the view vector: straight up or down keeps its heading.
        // The rig's turntable yaw is the camera yaw minus a quarter turn (its offset axis is +X).
        let seed = OrbitPose::from_yaw_pitch(
            (cam.yaw_degrees - 90.0).to_radians(),
            cam.pitch_degrees.to_radians(),
            cam.distance * radius,
            Vec3::from_array(cam.target),
        );
        rig.pose = seed;

        let buttons = cam_controls_egui::OrbitButtons {
            // RMB belongs to flight.
            dolly: None,
            wheel: !over_toolbar,
            // Shift + tumble: nearest world plane; Ctrl + Shift: nearest axis view.
            shift_snap: true,
            ..cam_controls_egui::OrbitButtons::HOUDINI
        };
        let frame =
            cam_controls_egui::gather_orbit_intents(ui, resp, &nav, &mut self.orbit_drag, buttons);
        let viewport = cam_viewport::ViewportSize::new(
            resp.rect.width().max(1.0) as u32,
            resp.rect.height().max(1.0) as u32,
        );
        // The rig pans in the unrolled view plane: turn screen motion back by the roll.
        let (sin, cos) = cam.roll_degrees.to_radians().sin_cos();
        let unroll = |x: f32, y: f32| (x * cos + y * sin, y * cos - x * sin);
        for intent in frame.intents {
            let intent = match intent {
                CameraIntent::Pan { dx_px, dy_px } => {
                    let (dx_px, dy_px) = unroll(dx_px, dy_px);
                    CameraIntent::Pan { dx_px, dy_px }
                }
                CameraIntent::PanInertia { dx_rate, dy_rate } => {
                    let (dx_rate, dy_rate) = unroll(dx_rate, dy_rate);
                    CameraIntent::PanInertia { dx_rate, dy_rate }
                }
                other => other,
            };
            self.orbit.apply_intent(intent, viewport);
        }
        let rig = &mut self.orbit.rig;
        let coasting = rig.has_inertia();
        if coasting {
            rig.update_dynamics(ui.input(|i| i.stable_dt).clamp(1.0e-4, 0.1));
        }
        // Write back only real changes: a still drag must not restart the progressive render.
        let pose = rig.pose;
        if pose != seed {
            let (yaw, pitch) = cam_controls::yaw_pitch_from_orientation(pose.orientation);
            cam.yaw_degrees = yaw.to_degrees() + 90.0;
            cam.pitch_degrees = pitch.to_degrees();
            cam.target = pose.target.to_array();
            cam.distance = pose.distance / radius;
        }
        self.orbit_written = rig.has_inertia().then_some(*cam);
        if coasting || frame.request_repaint {
            ui.ctx().request_repaint();
        }
    }

    fn frame_camera(&mut self, width: u32, height: u32) {
        use cam_controls::{CameraController, CameraPose, SpaceFlight};
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
        cam.distance = controller.pose().eye.distance(center)
            / self
                .scene
                .camera_reference
                .unwrap_or(self.scene.formula.framing_radius());
        self.fly = None;
        self.status = "Camera framed to bounds".into();
    }

    /// Seed the flight rig from the scene camera. `fly_camera` then drives it while RMB is
    /// held and lets its momentum and lock spring settle after release.
    fn start_flight(&mut self) {
        use cam_controls::{CameraPose, SpaceFlight};
        use glam::Vec3;
        let radius = self
            .scene
            .camera_reference
            .unwrap_or(self.scene.formula.framing_radius());
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
                    self.refresh_templates();
                }
                if ui.button("Open scene…").clicked() {
                    self.scene_file_dialog(false);
                    ui.close();
                }
                if ui.button("Save scene").clicked() {
                    if let Some(path) = self.scene_file_path.clone() {
                        self.submit_scene_file(path, true);
                    } else {
                        self.scene_file_dialog(true);
                    }
                    ui.close();
                }
                if ui.button("Save scene as…").clicked() {
                    self.scene_file_dialog(true);
                    ui.close();
                }
                ui.menu_button("Templates", |ui| {
                    if ui.button("Save current scene as template…").clicked() {
                        self.scene_action_dialog(crate::templates::FileAction::SaveTemplate);
                        ui.close();
                    }
                    if ui.button("Open template file…").clicked() {
                        self.scene_action_dialog(crate::templates::FileAction::OpenTemplate);
                        ui.close();
                    }
                    if ui
                        .add_enabled(
                            !self.templates_pending,
                            egui::Button::new("Refresh templates"),
                        )
                        .clicked()
                    {
                        self.refresh_templates();
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
                            match crate::presets::scene(index) {
                                Ok(scene) => {
                                    self.status = format!("Opened template {}", scene.name);
                                    self.load(scene);
                                }
                                Err(error) => self.status = error,
                            }
                        }
                        Some(crate::templates::Source::File(path)) => {
                            self.submit_scene_action(
                                path,
                                crate::templates::FileAction::OpenTemplate,
                            );
                        }
                        None => {}
                    }
                });
                ui.separator();
                if ui.button("Save bookmark").clicked() {
                    self.save_bookmark();
                    ui.close();
                }
                self.snapshot_menu(ui);
            });
            ui.menu_button("Edit", |ui| {
                if ui.button("Undo    Ctrl+Z").clicked() {
                    self.world.undo();
                    ui.close();
                }
                if ui.button("Redo    Ctrl+Shift+Z").clicked() {
                    self.world.redo();
                    ui.close();
                }
                ui.separator();
                if ui.button("Reset scene").clicked() {
                    self.load(self.origin.clone());
                    self.colour.sel = self.scene.colour.clone();
                    self.colour.rebuild();
                    ui.close();
                }
                if ui.button("Copy scene JSON").clicked()
                    && let Ok(json) = serde_json::to_string_pretty(&self.world.document)
                {
                    ui.ctx().copy_text(json);
                    ui.close();
                }
                ui.separator();
                if ui.button("Settings…").clicked() {
                    self.panels_to_open.push(dock::Panel::Settings);
                    ui.close();
                }
            });
            ui.menu_button("View", |ui| {
                ui.checkbox(&mut self.paused, "Pause rendering");
                if ui
                    .selectable_label(self.scene.camera.free_flight, "Free flight")
                    .clicked()
                {
                    self.toggle_flight_mode();
                    ui.close();
                }
                if ui.button("Restore camera    H").clicked() {
                    self.scene.camera = self.origin.camera;
                    self.fly = None;
                    ui.close();
                }
                if ui.button("Hide UI    Tab").clicked() {
                    self.show_ui = false;
                    ui.close();
                }
            });
            ui.menu_button("Render", |ui| {
                if ui.button("Render / Encode…").clicked() {
                    self.panels_to_open.push(dock::Panel::Export);
                    ui.close();
                }
            });
            self.window_menu(ui);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                self.layout_manager_ui(ui)
            });
        });
    }

    fn browser(&mut self, ui: &mut egui::Ui) {
        if self.tab == Tab::Materials {
            self.materials_tab(ui);
            return;
        }
        let mut load: Option<Scene> = None;
        let mut delete: Option<usize> = None;
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                let entries = if self.tab == Tab::Gallery {
                    &self.gallery
                } else {
                    &self.bookmarks
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
                            let selected = e.scene == self.origin;
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
                            if resp.clicked() {
                                load = Some(e.scene.clone());
                            }
                            if e.path.is_some() {
                                resp.context_menu(|ui| {
                                    if ui.button("Delete bookmark").clicked() {
                                        delete = Some(i);
                                        ui.close();
                                    }
                                });
                            }
                            if (i + 1) % columns == 0 {
                                ui.end_row();
                            }
                        }
                    });
            });
        if let Some(s) = load {
            self.load(s);
        }
        if let Some(i) = delete {
            let e = self.bookmarks.remove(i);
            if let Some(p) = e.path {
                let _ = self.io.send(crate::io_service::Command::Delete(p));
            }
        }
    }

    fn inspector(&mut self, ui: &mut egui::Ui) {
        let section_id = ui.id().with("render_settings_open");
        let open = ui.data(|data| data.get_temp::<bool>(section_id).unwrap_or(false));
        let response = egui_titlebar::CollapsingSection::new("Render settings")
            .id_salt("render_settings")
            .open(open)
            .tint(crate::world_ui::section_color("Render"), 0.28)
            .show(ui, |ui| {
                crate::inspector::render(
                    ui,
                    &mut self.render_editor,
                    &mut self.scene.render,
                    &mut self.target_spp,
                    &mut self.resolution,
                    &mut self.seed,
                    &mut self.world_ui.attribute_label_width,
                    self.world_ui.attribute_metrics,
                );
                if ui.button("New noise seed").clicked() {
                    self.seed = self.seed.wrapping_add(7920);
                }
            });
        ui.data_mut(|data| data.insert_temp(section_id, response.header.open));
        self.remember_material_targets();
        if let Some(material) = self.world.selection
            && self
                .world
                .document
                .info(material)
                .is_ok_and(|node| node.kind == crate::world::WorldKind::Material)
        {
            let apply = ui.add_enabled(!self.material_assignment_targets().is_empty(), egui::Button::new("Apply to object"))
                .on_hover_ui(|ui| {
                    if self.material_assignment_targets().is_empty() {
                        ui.label("Select an object with a Material field, then select a material to edit.");
                    } else {
                        ui.label("Assign this material to:");
                        for &id in self.material_assignment_targets() {
                            if let Ok(node) = self.world.document.info(id) { ui.label(node.name); }
                        }
                    }
                }).clicked();
            if apply {
                self.status = match self.assign_gallery_material(material, AssignTo::Selected) {
                    Ok(()) => "Material assigned".into(),
                    Err(error) => error,
                };
            }
        }
        self.world_ui.inspector(ui, &mut self.world);
        if std::mem::take(&mut self.world_ui.material_library_requested) {
            self.panels_to_open.push(dock::Panel::Materials);
        }
    }

    fn status_bar(&mut self, ui: &mut egui::Ui) {
        ui.scope(|ui| {
            if self.status_resizable {
                self.resizable_status_bar(ui);
            } else {
                self.fixed_status_bar(ui);
            }
        })
        .response
        .context_menu(|ui| {
            ui.checkbox(&mut self.status_resizable, "Resizable sections");
            if ui.button("Reset section widths").clicked() {
                self.status_layout = Default::default();
                ui.close();
            }
        });
    }

    fn resizable_status_bar(&mut self, ui: &mut egui::Ui) {
        // Fixed section identity/count: transient frame and OIDN states never
        // move widths onto another indicator. Rendering uses only mailbox data.
        const WIDTHS: [f32; 10] = [
            200.0, 95.0, 95.0, 220.0, 130.0, 100.0, 170.0, 80.0, 150.0, 0.0,
        ];
        let frame = self.frame.as_deref();
        let preview = self.showing_preview;
        let target_spp = self.target_spp;
        egui_statusbar::StatusBar::new().show_with(
            ui,
            &mut self.status_layout,
            &WIDTHS,
            |index, ui| {
                ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Truncate);
                match index {
                    0 => {
                        ui.label(RichText::new(&self.gpu_name).weak())
                            .on_hover_text(&self.gpu_name);
                    }
                    1 => {
                        ui.label(if self.scene.camera.free_flight {
                            "Free flight · `"
                        } else {
                            "Horizon · `"
                        });
                    }
                    2 => {
                        if let Some(frame) = frame {
                            ui.label(format!("{}×{}", frame.width, frame.height));
                        }
                    }
                    3 => {
                        if let Some(frame) = frame {
                            if preview {
                                ui.colored_label(WARNING, "preview");
                            } else {
                                ui.label(format!("{} / {} spp", frame.samples, target_spp));
                                ui.add(
                                    egui::ProgressBar::new(
                                        frame.samples as f32 / target_spp.max(1) as f32,
                                    )
                                    .desired_width((ui.available_width() - 4.0).max(0.0)),
                                );
                            }
                        }
                    }
                    4 => {
                        if let Some(frame) = frame {
                            ui.label(format!("{:.1} Msamples/s", frame.msamples_per_s()));
                        }
                    }
                    5 => {
                        ui.label(format!("{} spp/batch", self.spp_per_frame));
                    }
                    6 => {
                        if let Some(frame) = frame {
                            if let Some(error) = &frame.denoise_error {
                                ui.colored_label(WARNING, "OIDN failed")
                                    .on_hover_text(error);
                            } else if frame.denoised_samples > 0 {
                                ui.label(format!(
                                    "OIDN {} spp · {:.1} ms",
                                    frame.denoised_samples, frame.denoise_ms
                                ));
                            }
                        }
                    }
                    7 => {
                        ui.label(format!("UI {:.0} fps", 1000.0 / self.frame_ms.max(0.1)));
                    }
                    8 => {
                        if let Some(frame) = frame {
                            march_limit_label(ui, frame.unresolved);
                        }
                    }
                    _ => {
                        ui.label(RichText::new(&self.status).weak())
                            .on_hover_text(&self.status);
                    }
                }
            },
        );
    }

    fn fixed_status_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Truncate);
            ui.label(RichText::new(&self.gpu_name).weak());
            ui.label(if self.scene.camera.free_flight {
                "Free flight · `"
            } else {
                "Horizon · `"
            });
            ui.separator();
            if let Some(t) = self.current() {
                ui.label(format!("{}×{}", t.width, t.height));
                ui.separator();
                if self.showing_preview {
                    ui.label(RichText::new("preview").color(WARNING));
                } else {
                    ui.label(format!("{} / {} spp", t.samples, self.target_spp));
                    ui.add(
                        egui::ProgressBar::new(t.samples as f32 / self.target_spp as f32)
                            .desired_width(120.0),
                    );
                }
                ui.separator();
                ui.label(format!("{:.1} Msamples/s", t.msamples_per_s()));
                ui.separator();
                ui.label(format!("{} spp/batch", self.spp_per_frame));
                if t.denoised_samples > 0 {
                    ui.separator();
                    ui.label(format!(
                        "OIDN {} spp · {:.1} ms",
                        t.denoised_samples, t.denoise_ms
                    ));
                }
                if let Some(error) = &t.denoise_error {
                    ui.separator();
                    ui.label(RichText::new("OIDN failed").color(WARNING))
                        .on_hover_text(error);
                }
            }
            ui.separator();
            ui.label(format!("UI {:.0} fps", 1000.0 / self.frame_ms.max(0.1)));
            if let Some(t) = self.current()
                && t.unresolved > 0.0
            {
                ui.separator();
                march_limit_label(ui, t.unresolved);
            }
            if !self.status.is_empty() {
                ui.separator();
                ui.label(RichText::new(&self.status).weak())
                    .on_hover_text(&self.status);
            }
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
        }
        if self.fly.is_none() {
            self.orbit_camera(ui, &resp, toolbar.contains_pointer);
        }
        if resp.double_clicked() {
            self.scene.camera.target = [0.0; 3];
        }

        let ppp = ui.ctx().pixels_per_point() * self.resolution;
        let (w, h) = (
            ((avail.x * ppp) as usize).max(16),
            ((avail.y * ppp) as usize).max(16),
        );
        let monitor = Monitor::read(ui.ctx());
        let (output_hdr, white) = (monitor.hdr, monitor.white_nits);
        if self.preview_key.is_some_and(|key| key.5 != w || key.6 != h) {
            self.preview.cancel();
            self.preview_key = None;
            self.world_ui.playing = false;
            self.showing_preview = false;
        }
        if !self.preview.displaying_preview()
            && !(self.frame_from_cache && self.preview.position().is_some())
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
                let mvp = egui_hdr_view::Mvp {
                    model: glam::Mat4::from_scale(glam::vec3(2.0, 2.0, 1.0)).to_cols_array_2d(),
                    ..Default::default()
                };
                let mut view = self.hdr_view.lock().unwrap();
                view.set_output_format(egui_display::CANVAS_FORMAT);
                let (format, bytes) = if output_hdr {
                    (egui_hdr_view::HdrFormat::Rgba32F, t.hdr_bytes.clone())
                } else {
                    (egui_hdr_view::HdrFormat::Rgba8, t.sdr_bytes.clone())
                };
                view.stage_frame(format, bytes, t.width, t.height, mvp);
                self.staged_frame = Some(t);
                self.staged_output = Some((output_hdr, white));
            }
            ui.painter().add(egui_wgpu::Callback::new_paint_callback(
                rect,
                egui_hdr_view::HdrPaintCallback {
                    inner: self.hdr_view.clone(),
                },
            ));
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

impl App {
    /// Identify cached render content without serializing the document on repaint.
    fn preview_identity(
        &self,
        ctx: &egui::Context,
    ) -> (playa_graph::NodeId, u64, bool, u32, u32, usize, usize, u32) {
        let state =
            ctx.data(|d| d.get_temp::<egui_display::DisplayState>(egui_display::state_id()));
        (
            playa_graph::NodeId(self.world.document.graph.id),
            self.world.revision(),
            state.as_ref().is_some_and(|s| s.output.is_hdr()),
            state.as_ref().map_or(100.0, |s| s.target.white).to_bits(),
            self.seed,
            self.request.as_ref().map_or(640, |r| r.width),
            self.request.as_ref().map_or(360, |r| r.height),
            self.target_spp,
        )
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
        let key = self.preview_identity(ctx);
        if viewport.target_spp != self.target_spp
            || viewport.output_hdr != key.2
            || viewport.white_nits.to_bits() != key.3
            || viewport.seed != key.4
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
            self.make_preview_request(
                self.world.document.first,
                self.world.document.last,
                self.target_spp,
                ctx,
            )
        };
        match self
            .preview
            .cache_viewport(request, self.world_ui.playhead, frame.clone())
        {
            Ok(()) => self.preview_key = Some(key),
            Err(error) => self.status = error,
        }
    }

    fn make_preview_request(
        &self,
        first: u32,
        last: u32,
        spp: u32,
        ctx: &egui::Context,
    ) -> crate::preview::PreviewRequest {
        let key = self.preview_identity(ctx);
        let scene = if self.preview_key == Some(key) {
            self.preview.cached_request().map(|r| r.scene)
        } else {
            None
        }
        .unwrap_or_else(|| Arc::new(self.frozen_world_scene()));
        crate::preview::PreviewRequest {
            generation: self.preview_sequence,
            scene,
            first,
            last,
            fps: self.world.document.fps as f32,
            width: key.5,
            height: key.6,
            spp,
            seed: self.seed,
            output_hdr: key.2,
            white_nits: f32::from_bits(key.3),
            cache_fraction: 0.05,
            reserve_gb: 2.0,
        }
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
        let key = self.preview_identity(ctx);
        let request = self.make_preview_request(first, last, mode.samples(self.target_spp), ctx);
        self.start_preview_request(request, mode.cache_all(), key);
    }

    fn start_preview_request(
        &mut self,
        request: crate::preview::PreviewRequest,
        cache_all: bool,
        key: (playa_graph::NodeId, u64, bool, u32, u32, usize, usize, u32),
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
        let navigating = self.viewport_rect.is_some_and(|rect| {
            ctx.input(|input| {
                input.pointer.hover_pos().is_some_and(|p| rect.contains(p))
                    && (input.pointer.any_down()
                        || input.smooth_scroll_delta.y != 0.0
                        || [egui::Key::G, egui::Key::H, egui::Key::F]
                            .iter()
                            .any(|k| input.key_pressed(*k)))
            })
        });
        if navigating && self.preview.position().is_some() {
            self.preview.cancel();
            self.preview_key = None;
            self.showing_preview = false;
            self.world_ui.playing = false;
        }
        if self
            .preview_key
            .is_some_and(|key| key != self.preview_identity(ctx))
        {
            self.preview.cancel();
            self.preview_key = None;
            self.showing_preview = false;
            self.world_ui.playing = false;
        }
        if let Some(position) = self.preview.position() {
            if self.world_ui.playhead != position {
                self.preview.seek(self.world_ui.playhead);
            }
            if self.world_ui.playing != self.preview.running() {
                if self.world_ui.playing {
                    self.preview.resume();
                } else {
                    self.preview.pause();
                }
            }
        }
        if self.preview.displaying_preview() {
            if let Some(request) = &mut self.request {
                if request.active {
                    request.active = false;
                    self.renderer.request_viewport(request.clone());
                }
            }
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

    fn process_preview_intent(&mut self, ctx: &egui::Context) {
        if self.camera_recording.is_some() {
            self.world_ui.playing = false;
            self.world_ui.take_preview_action();
            return;
        }
        match self.world_ui.take_preview_action() {
            Some(action) => self.begin_preview(action.first, action.last, action.mode, ctx),
            None if self.preview.position().is_none() && self.world_ui.playing => {
                let head = self.world_ui.playhead;
                let (first, last) = self
                    .world_ui
                    .playback_range
                    .unwrap_or((self.world.document.first, self.world.document.last));
                self.begin_preview(first, last, crate::preview::PreviewMode::Play, ctx);
                self.preview.seek(head.clamp(first, last));
                self.world_ui.seek(head.clamp(first, last));
            }
            None => {}
        }
        self.sync_preview(0.0, ctx);
    }

    fn refresh_scene(&mut self) -> Result<(), String> {
        if let Some((_, recording)) = &self.camera_recording {
            if self.world.revision() == recording.revision
                && self.world.document.graph.id == recording.document.graph.id
            {
                return Ok(());
            }
            self.camera_recording = None;
            self.fly = None;
            self.orbit = Default::default();
            self.orbit_written = None;
            self.evaluated_world = None;
            self.status = "Camera recording cancelled because the document changed".into();
        }
        // Cached playback presents worker-owned frames; keep the frozen scene instead
        // of rebuilding serialized world data on every clock tick. An edit or paused
        // navigation evaluates the current pose before any authoring takes place.
        if self.preview.running()
            && self.preview_key.is_some_and(|key| {
                key.0 == playa_graph::NodeId(self.world.document.graph.id)
                    && key.1 == self.world.revision()
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
        Ok(())
    }
    pub(crate) fn ui(&mut self, root: &mut egui::Ui) {
        let ctx = root.ctx().clone();
        let recording_frame = self.camera_recording.is_some();
        self.apply_fonts(&ctx);
        self.update_scene_picker(&ctx);
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
        let edit_frame = f64::from(self.world_ui.playhead);
        let edit_origin = self.load_revision;
        use crate::hotkeys::{self, Command as Hotkey, Scope};
        if hotkeys::consume(&ctx, Scope::Global, Hotkey::ToggleUi) {
            self.show_ui = !self.show_ui;
        }
        if hotkeys::consume(&ctx, Scope::Global, Hotkey::Redo) {
            self.world.redo();
        } else if hotkeys::consume(&ctx, Scope::Global, Hotkey::Undo) {
            self.world.undo();
        }
        self.node_clipboard(&ctx);

        if let Err(error) = self.refresh_scene() {
            self.status = error;
        }
        let before = self.snapshot_before.take();
        let thumbs_pending = self.render_one_thumbnail(&ctx);
        self.poll_events(&ctx);
        self.export.update(&self.renderer);
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

        if !self.viewport_visible {
            if let Some(request) = &mut self.request {
                request.active = false;
                self.renderer.request_viewport(request.clone());
            }
            self.fly = None;
            ctx.send_viewport_cmd(egui::ViewportCommand::CursorGrab(egui::CursorGrab::None));
        }
        let camera_gesture =
            (self.fly.is_some() || ctx.input(|i| i.pointer.any_down())).then(|| {
                egui::Id::new(("viewport_camera_edit", self.world.document.active_camera)).value()
            });
        if let Some(before) = before.as_ref()
            && self.load_revision == edit_origin
            && !recording_frame
            && self.camera_recording.is_none()
            && (self.scene.camera != before.camera
                || self.scene.render != before.render
                || self.scene.colour != before.colour)
        {
            if self.scene.camera != before.camera {
                if let Err(error) = self.world.navigate_camera(
                    &self.scene,
                    edit_frame,
                    self.world_ui.auto_key,
                    camera_gesture,
                ) {
                    // Rejected navigation must not leave an unauthored viewport pose or inertia.
                    self.evaluated_world = None;
                    self.fly = None;
                    self.status = error;
                }
            }
            let gesture = egui_attr_grid::edit_gesture(&ctx)
                .or(camera_gesture)
                .or_else(|| {
                    ctx.input(|input| input.pointer.primary_released())
                        .then(|| self.world.active_edit())
                        .flatten()
                });
            let result = if gesture.is_some() {
                self.world.edit_snapshot_with_gesture(
                    self.world.selection,
                    before,
                    &self.scene,
                    edit_frame,
                    gesture,
                )
            } else {
                self.world
                    .edit_snapshot(self.world.selection, before, &self.scene, edit_frame)
            };
            if let Err(error) = result {
                self.status = error;
            }
        }
        self.snapshot_before = before;
        if let Some((start, recording)) = &mut self.camera_recording {
            if let Err(error) = recording.capture(start.elapsed().as_secs_f64(), self.scene.camera)
            {
                self.status = error;
                self.camera_recording = None;
                self.evaluated_world = None;
            }
            ctx.request_repaint();
        }
        self.remember_material_targets();
        self.process_preview_intent(&ctx);
        self.world
            .finish_edit_unless(egui_attr_grid::edit_gesture(&ctx).or(camera_gesture));
        if let Err(error) = self.refresh_scene() {
            self.status = error;
        }
        self.save_settings(&ctx);
        if let Some(error) = self.current().and_then(|t| t.colour_error.as_deref()) {
            self.status = format!("Colour output failed: {error}");
        }
        self.handle_snap(&ctx, thumbs_pending);
        ctx.request_repaint_after(std::time::Duration::from_secs_f64(
            1.0 / self.gui_fps as f64,
        ));
    }
}

#[allow(dead_code)]
fn exists(p: &Path) -> bool {
    p.exists()
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let object = app.world.selection.unwrap();
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
        app.add_library_material(0);
        let material = app.world.selection.unwrap();
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
            app.world.selection = None;
            app.world.selected.clear();
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
            app.tab = Tab::Gallery;
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
                    |root| app.browser(root),
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
        let object = app.world.selection.unwrap();
        let assignment = app.world.document.assigned_material(object).unwrap();
        app.world
            .execute(crate::world::WorldCommand::CreateMaterial {
                material: Material::default(),
                name: "Gallery test material".into(),
            })
            .unwrap();
        let material = app.world.selection.unwrap();
        app.world.selection = Some(object);
        app.world.selected = vec![object];
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
        assert_eq!(app.world.selection, Some(material));
        assert_eq!(app.world.selected, [material]);
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
        let object = app.world.selection.unwrap();
        let original = app.world.document.assigned_material(object).unwrap();
        app.remember_material_targets();
        app.world
            .execute(WorldCommand::CreateMaterial {
                material: Material::default(),
                name: "Apply test".into(),
            })
            .unwrap();
        let material = app.world.selection.unwrap();
        app.select_material_node(material);
        app.remember_material_targets();
        assert_eq!(app.material_assignment_targets(), &[object]);
        assert!(app.world.document.supports_material(object));
        assert!(!app.world.document.supports_material(material));
        app.assign_gallery_material(material, AssignTo::Selected)
            .unwrap();
        assert_eq!(
            app.world.selection,
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
        app.world.selection = Some(camera);
        app.world.selected = vec![camera];
        app.remember_material_targets();
        assert!(app.material_assignment_targets().is_empty());
        assert!(
            app.assign_gallery_material(material, AssignTo::Selected)
                .is_err()
        );
    }

    #[test]
    fn viewport_cache_accepts_only_final_current_authoring_frames() {
        let _gpu_test = crate::test_gpu::lock();
        let ctx = egui::Context::default();
        let mut app = App::new();
        app.target_spp = 8;
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
        app.target_spp = 16;
        assert_ne!(key, Some(app.preview_identity(&ctx)));
    }

    #[test]
    fn gallery_assignment_and_object_material_field_share_uuid_and_undo() {
        let _gpu_test = crate::test_gpu::lock();
        use crate::world::WorldCommand;
        let mut app = App::new();
        let object = app.world.selection.unwrap();
        app.world
            .execute(WorldCommand::CreateMaterial {
                material: Material::default(),
                name: "Shared gallery material".into(),
            })
            .unwrap();
        let material = app.world.selection.unwrap();
        let count = app.world.document.nodes().len();
        app.world.selection = Some(object);
        app.world.selected = vec![object];
        let old = app.world.document.assigned_material(object).unwrap();
        app.assign_gallery_material(material, AssignTo::Selected)
            .unwrap();
        assert_eq!(
            app.world.document.assigned_material(object).unwrap(),
            Some(material)
        );
        assert_eq!(app.world.selection, Some(object));
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
        let ctx = egui::Context::default();
        assert!(!Settings::default().auto_key);
        assert!(serde_json::from_str::<Settings>("{}").is_err());
        app.changed_settings_json(&ctx).unwrap();
        app.world_ui.auto_key = true;
        let json = app.changed_settings_json(&ctx).unwrap().unwrap();
        assert!(serde_json::from_str::<Settings>(&json).unwrap().auto_key);
        for _ in 0..16 {
            assert!(app.changed_settings_json(&ctx).unwrap().is_none());
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
        let mut scene = Scene::preset(crate::params::FAMILY_BOX);
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
        app.load(Scene::preset(crate::params::FAMILY_BULB));
        app.scene_file_event(crate::io_service::SceneEvent {
            id: 8,
            path: PathBuf::from("stale.frac.json"),
            result: Ok(Some(Box::new(Scene::preset(crate::params::FAMILY_BOX)))),
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
                        &app.renderer,
                        None,
                        &crate::color::default_selection(),
                        || panic!("An idle export panel must not clone the authoring document"),
                    );
                });
            });
        }
    }

    #[test]
    fn settings_and_layouts_serialize_only_after_actual_changes() {
        let _gpu_test = crate::test_gpu::lock();
        let mut app = App::new();
        let ctx = egui::Context::default();
        app.refresh_scene().unwrap();
        assert!(app.changed_settings_json(&ctx).unwrap().is_some());
        let serializations = app.settings_serializations;
        let dock_serializations = app.layouts.cache.serializations;
        let baseline = app.snapshot_before.as_ref().unwrap().objects.as_ptr();
        let snapshots = app.snapshot_clones;
        for _ in 0..32 {
            app.refresh_scene().unwrap();
            assert!(app.changed_settings_json(&ctx).unwrap().is_none());
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
        let mut export = app.export.settings().clone();
        export.name = "new".into();
        export.qp = 19;
        app.export.restore(export);
        let json = app.changed_settings_json(&ctx).unwrap().unwrap();
        let saved: Settings = serde_json::from_str(&json).unwrap();
        assert_eq!(saved.attribute_metrics.numeric_width, 64.0);
        assert_eq!(saved.fonts.face, "Custom face");
        assert_eq!(saved.status_layout.widths, [210.0, 100.0]);
        assert_eq!(saved.gui_fps, 144);
        assert_eq!(saved.export.name, "new");
        assert_eq!(saved.export.qp, 19);
        assert!(app.changed_settings_json(&ctx).unwrap().is_none());
        let current = app.colour.sel.clone();
        assert!(app.colour.presets.store(1, "Custom colour", &current));
        let json = app.changed_settings_json(&ctx).unwrap().unwrap();
        let saved: Settings = serde_json::from_str(&json).unwrap();
        assert_eq!(saved.colour_presets, app.colour.presets);
        assert_eq!(saved.colour_presets.slots[1].name, "Custom colour");
        assert!(app.changed_settings_json(&ctx).unwrap().is_none());
        assert!(serde_json::from_str::<Settings>("{}").is_err());
        app.layouts.store.save(
            "Edited workspace",
            app.layouts.cache.blob.as_ref().unwrap().clone(),
        );
        let json = app.changed_settings_json(&ctx).unwrap().unwrap();
        let saved: Settings = serde_json::from_str(&json).unwrap();
        assert_eq!(saved.layouts.current(), "Edited workspace");
        assert!(app.changed_settings_json(&ctx).unwrap().is_none());
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
        app.load(reload);
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
                            toolbar_rect = app.viewport_toolbar(ui, ui.max_rect()).rect;
                        });
                    },
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
        let output = frame(
            &mut app,
            vec![egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Secondary,
                pressed: false,
                modifiers: Default::default(),
            }],
        );
        assert!(
            output.viewport_output[&egui::ViewportId::ROOT]
                .commands
                .contains(&egui::ViewportCommand::CursorGrab(egui::CursorGrab::None))
        );
    }

    #[test]
    fn duplicate_bookmark_thumbnails_fill_the_next_empty_entry() {
        let _gpu_test = crate::test_gpu::lock();
        let scene = Scene::preset(crate::params::FAMILY_BULB);
        let mut rendered = scene.clone();
        rendered.render.max_bounces = rendered.render.max_bounces.min(3);
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
            app.fly.as_ref().unwrap().momentum.angular.length() > 0.0,
            "The shared controller must receive angular momentum from mouse-look"
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
            key: egui::Key::Backtick,
            physical_key: Some(egui::Key::Backtick),
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
            "Backtick must enable free flight while RMB is held"
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
            "Shift+backtick (tilde) must restore horizon mode"
        );
        for _ in 0..240 {
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
        app.scene.object.offset = [7.0, -3.0, 2.0];
        frame(&mut app, vec![shortcut(egui::Key::F)]);
        assert!(
            (glam::Vec3::from_array(app.scene.camera.target)
                - glam::Vec3::from_array(app.scene.object.offset))
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
            app.scene.object.offset,
            [7.0, -3.0, 2.0],
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
        for _ in 0..240 {
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
        assert_eq!(Settings::default().controls.look_sensitivity, 1.0);
        app.controls.look_sensitivity = 0.4;
        app.controls.fly_speed = 3.0;
        let loaded: Controls =
            serde_json::from_str(&serde_json::to_string(&app.controls).unwrap()).unwrap();
        assert_eq!(loaded.look_sensitivity, 0.4);
        assert_eq!(loaded.fly_speed, 3.0);
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
            } else {
                assert!(
                    fly.orientation.angle_between(before.orientation) > 1e-4,
                    "{name}"
                );
                assert!(
                    fly.momentum.angular.length() < momentum.angular.length(),
                    "{name} must damp"
                );
            }
            for _ in 0..600 {
                fly.update_dynamics(1.0 / 60.0);
            }
            assert!(!fly.update_dynamics(1.0 / 60.0), "{name} must settle");
        }
        let mut old = serde_json::to_value(Scene::preset(0)).unwrap();
        old["camera"]
            .as_object_mut()
            .unwrap()
            .remove("roll_degrees");
        old["camera"].as_object_mut().unwrap().remove("free_flight");
        assert!(serde_json::from_value::<Scene>(old).is_err());
    }
}
