//! The browser UI (egui): gallery + bookmarks with GPU thumbnails, the progressive viewport,
//! the scene inspector, screenshots and final PNG renders.

use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use egui::{self, Color32, ColorImage, RichText, Sense, TextureHandle, TextureOptions, Vec2};

use crate::render_service::{Command, Frame, RenderEvent, RenderService, ViewportRequest};
use crate::scene::*;
use crate::world::WorldDocument;
use std::sync::Arc;

#[path = "dock.rs"]
mod dock;

const THUMB_W: usize = 224;
const THUMB_H: usize = 126;
const THUMB_SPP: u32 = 24;
/// After the last camera / parameter change, keep the low-resolution preview this long.
const PREVIEW_HOLD_S: f32 = 0.18;

const FINAL_SIZES: [(u32, u32, &str); 5] = [
    (1280, 720, "720p"),
    (1920, 1080, "1080p"),
    (2560, 1440, "1440p"),
    (3840, 2160, "4K"),
    (2048, 2048, "2048²"),
];

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

struct Job {
    width: usize,
    height: usize,
    samples: u32,
    spp: u32,
    path: PathBuf,
}

pub(crate) struct App {
    renderer: RenderService,
    io: crate::io_service::IoService,
    gpu_name: String,
    frame: Option<Arc<Frame>>,
    frame_from_cache: bool,
    request: Option<ViewportRequest>,
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
    preview_key: Option<(playa_graph::NodeId, u64, bool, u32, u32)>,
    preview_sequence: u64,
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
    dock: egui_dock::DockState<dock::Panel>,
    panels_to_open: Vec<dock::Panel>,
    toolbar: egui_viewport_toolbar::ToolbarState,
    fonts: dock::Fonts,
    applied_fonts: Option<dock::Fonts>,
    viewport_visible: bool,
    viewport_rect: Option<egui::Rect>,
    config_picker: egui_file_dialog::FileDialog,
    scene_picker: Option<(bool, egui_file_dialog::FileDialog)>,
    scene_file_path: Option<PathBuf>,
    scene_file_sequence: u64,
    scene_file_pending: Option<(u64, u64)>,
    persisted_settings: Option<Settings>,
    persisted_layout_revision: u64,
    legacy_before: Option<Scene>,
    #[cfg(test)]
    settings_serializations: usize,
    #[cfg(test)]
    legacy_snapshots: usize,
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
    show_ui: bool,
    job: Option<Job>,
    final_size: usize,
    final_spp: u32,
    status: String,
    frame_ms: f32,
    seed: u32,
    /// `FRAC_SNAP=out.png [FRAC_SNAP_PRESET=i] [FRAC_SNAP_SPP=n]`: screenshot the window once the
    /// thumbnails and n viewport samples are done, then quit (for docs).
    snap: Option<(PathBuf, u32, bool)>,
    /// Flight via cam-controls' inertial `SpaceFlight`, alive while RMB is held and
    /// while its momentum coasts after release.
    fly: Option<cam_controls::SpaceFlight>,
    /// Persistent mouse sensitivity and flight speed (also adjusted by the wheel).
    controls: Controls,
}

fn now_stamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn data_dir() -> PathBuf {
    if let Some(root) = std::env::var_os("FRAC_PROFILE_DIR") {
        return PathBuf::from(root).join("data");
    }
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("frac-rs")
}

fn pictures_dir() -> PathBuf {
    dirs::picture_dir()
        .or_else(dirs::home_dir)
        .unwrap_or_else(|| PathBuf::from("."))
        .join("frac-rs")
}

fn load_bookmarks() -> Vec<Entry> {
    let dir = data_dir().join("bookmarks");
    let mut v: Vec<(PathBuf, Scene)> = std::fs::read_dir(&dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .filter_map(|p| {
            let s = std::fs::read_to_string(&p).ok()?;
            let scene =
                if let Ok(document) = serde_json::from_str::<crate::world::WorldDocument>(&s) {
                    let mut snapshot = document.snapshot(f64::from(document.first)).ok()?;
                    snapshot.document = Some(Box::new(document));
                    snapshot
                } else {
                    serde_json::from_str::<Scene>(&s).ok()?
                };
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

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(default)]
struct Controls {
    look_sensitivity: f32,
    fly_speed: f32,
}
impl Default for Controls {
    fn default() -> Self {
        Self {
            look_sensitivity: 1.0,
            fly_speed: 1.0,
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(default)]
struct Settings {
    display: egui_display::DisplayPrefs,
    colour: crate::ocio::Sel,
    panel: egui_prefs2::PrefsPanelState,
    controls: Controls,
    fonts: dock::Fonts,
    layout: Option<String>,
    toolbar: egui_viewport_toolbar::ToolbarState,
    layouts: egui_layout_manager::LayoutStore,
    export: crate::export::ExportSettings,
    gui_fps: u32,
    status_layout: egui_statusbar::StatusBarLayout,
    status_resizable: bool,
    attribute_metrics: crate::ui_style::AttributeMetrics,
    auto_key: bool,
    #[serde(default)]
    timeline_initialized: bool,
    #[serde(default)]
    world_layout_initialized: bool,
}
impl Default for Settings {
    fn default() -> Self {
        Self {
            display: Default::default(),
            colour: crate::color::default_selection(),
            panel: Default::default(),
            controls: Default::default(),
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
            timeline_initialized: true,
            world_layout_initialized: true,
        }
    }
}
fn settings_path() -> PathBuf {
    if let Some(root) = std::env::var_os("FRAC_PROFILE_DIR") {
        return PathBuf::from(root).join("settings.json");
    }
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("frac-rs/settings.json")
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
            preview_progress: None,
            evaluated_world: None,
            load_revision: 0,

            showing_preview: false,
            hdr_view: std::sync::Arc::new(std::sync::Mutex::new(egui_hdr_view::HdrView::new())),
            display: Default::default(),
            colour: crate::ocio::State::new(scene.colour.clone()),
            prefs: Default::default(),
            dock: dock::default_layout(),
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
            persisted_settings: None,
            persisted_layout_revision: 0,
            legacy_before: None,
            #[cfg(test)]
            settings_serializations: 0,
            #[cfg(test)]
            legacy_snapshots: 0,
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
            show_ui: true,
            job: None,
            final_size: 1,
            final_spp: 512,
            status: String::new(),
            frame_ms: 16.0,
            seed: 0,
            fly: None,
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
            self.toolbar = settings.toolbar;
            self.layouts.store = settings.layouts;
            self.export.restore(settings.export);
            self.gui_fps = settings.gui_fps.clamp(15, 240);
            self.status_layout = settings.status_layout;
            self.status_resizable = settings.status_resizable;
            self.world_ui.attribute_metrics = settings.attribute_metrics.normalized();
            self.world_ui.auto_key = settings.auto_key;
            self.fonts = settings.fonts;
            if let Some(blob) = settings.layout {
                match egui_dock_layout::from_blob(&blob) {
                    Ok(layout) if dock::valid_layout(&layout) => self.dock = layout,
                    _ => self.status = "Saved layout could not be restored; using default".into(),
                }
            }
            if !settings.timeline_initialized {
                dock::add_timeline(&mut self.dock);
            }
            if !settings.world_layout_initialized {
                dock::add_outliner(&mut self.dock);
            }
            self.colour = crate::ocio::State::new(settings.colour.clone());
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
            self.prefs.selected = if category.eq_ignore_ascii_case("fonts") {
                3
            } else if category.eq_ignore_ascii_case("controls") {
                2
            } else {
                usize::from(category.eq_ignore_ascii_case("color"))
            };
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
            let categories = [egui_prefs2::Category::new(egui_phosphor::regular::MONITOR, "Display"), egui_prefs2::Category::new(egui_phosphor::regular::MONITOR, "Color"), egui_prefs2::Category::new(egui_phosphor::regular::MONITOR, "Controls"), egui_prefs2::Category::new(egui_phosphor::regular::TEXT_T, "Fonts")];
            egui_prefs2::draw(ui, &mut prefs, &categories, |ui, idx| {
                category = idx;
                match idx {
                    0 => {
                        egui_display::settings_ui(ui, &mut self.display, state.as_ref());
                        if state.as_ref().is_some_and(|s| !s.available.iter().any(|o| o.is_hdr())) {
                            ui.label("This window surface offers SDR only. PQ/HDR targets in Color still render and export HDR; the screen uses an SDR preview.");
                        }
                        ui.add(egui::DragValue::new(&mut self.gui_fps).range(15..=240).suffix(" GUI FPS"));
                        ui.add_space(8.0);
                        if ui.button("Colour management & monitor presets…").clicked() { destination = Some(1); }
                    }
                    1 => {
                        egui_prefs2::section_header(ui, "Colour management");
                        if let Some(state) = &state { ui.label(format!("Window output: {}.", state.output.label())); }
                        ui.label("Monitor presets choose rendering and export. PQ/HDR remains available on SDR screens using an SDR preview. HDR 1000 nits is the rendering peak; SDR reference white controls UI brightness.");
                        if ui.button("Display output & reference white…").clicked() { destination = Some(0); }
                        ui.add_space(8.0);
                        changed = self.colour.ui(ui, &mut browse);
                    }
                    3 => self.fonts_ui(ui),
                    _ => {
                        egui_prefs2::section_header(ui, "Camera controls");
                        egui_attr_table::attr_table(ui, |t| {
                            t.row("Mouse sensitivity").default(1.0).slider(&mut self.controls.look_sensitivity, 0.1..=5.0);
                            t.row("Flight speed ×").default(1.0).slider(&mut self.controls.fly_speed, 0.02..=50.0);
                        });
                        egui_prefs2::section_header(ui, "Attribute controls");
                        let defaults = crate::ui_style::AttributeMetrics::default();
                        let metrics = &mut self.world_ui.attribute_metrics;
                        egui_attr_table::attr_table(ui, |table| {
                            table.row("Field height").default(defaults.field_height).slider(&mut metrics.field_height, crate::ui_style::AttributeMetrics::FIELD_HEIGHT_RANGE);
                            table.row("Numeric field width").default(defaults.numeric_width).slider(&mut metrics.numeric_width, crate::ui_style::AttributeMetrics::NUMERIC_WIDTH_RANGE);
                            table.row("Icon size").default(defaults.icon_side).slider(&mut metrics.icon_side, crate::ui_style::AttributeMetrics::ICON_SIDE_RANGE);
                            table.row("Row spacing").default(defaults.row_gap).slider(&mut metrics.row_gap, crate::ui_style::AttributeMetrics::ROW_GAP_RANGE);
                            table.row("Component spacing").default(defaults.component_gap).slider(&mut metrics.component_gap, crate::ui_style::AttributeMetrics::COMPONENT_GAP_RANGE);
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
            } else {
                self.controls = Default::default();
                self.world_ui.attribute_metrics = Default::default();
            }
        }
        if let Some(idx) = destination {
            self.prefs.selected = idx;
        }
        if browse {
            self.config_picker.pick_file();
        }
        self.apply_colour_change(changed);
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
            && saved.panel == self.prefs
            && saved.controls.look_sensitivity == self.controls.look_sensitivity
            && saved.controls.fly_speed == self.controls.fly_speed
            && saved.toolbar == self.toolbar
            && saved.fonts == self.fonts
            && saved.layouts == self.layouts.store
            && saved.gui_fps == self.gui_fps
            && saved.status_layout == self.status_layout
            && saved.status_resizable == self.status_resizable
            && saved.attribute_metrics == self.world_ui.attribute_metrics
            && saved.auto_key == self.world_ui.auto_key
            && export.format == previous.format
            && export.output == previous.output
            && export.width == previous.width
            && export.height == previous.height
            && export.samples == previous.samples
            && export.first == previous.first
            && export.last == previous.last
            && export.fps_num == previous.fps_num
            && export.fps_den == previous.fps_den
            && export.qp == previous.qp
            && export.preset == previous.preset
            && export.overwrite == previous.overwrite
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
            panel: self.prefs.clone(),
            controls: Controls {
                look_sensitivity: self.controls.look_sensitivity,
                fly_speed: self.controls.fly_speed,
            },
            layout,
            toolbar: self.toolbar,
            fonts: self.fonts.clone(),
            layouts: self.layouts.store.clone(),
            export: self.export.settings().clone(),
            gui_fps: self.gui_fps,
            status_layout: self.status_layout.clone(),
            status_resizable: self.status_resizable,
            attribute_metrics: self.world_ui.attribute_metrics,
            auto_key: self.world_ui.auto_key,
            timeline_initialized: true,
            world_layout_initialized: true,
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
            && self.frame.as_ref().is_some_and(|t| t.samples >= spp);
        if ready && !requested {
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(Default::default()));
            self.snap = self.snap.take().map(|(path, spp, _)| (path, spp, true));
        }
        ctx.request_repaint();
    }

    fn scene_file_dialog(&mut self, save: bool) {
        let mut dialog = egui_file_dialog::FileDialog::new()
            .add_file_filter_extensions("Fractal scene", vec!["json"])
            .default_file_name(&format!("{}.frac.json", crate::slug(&self.scene.name)));
        if let Some(parent) = self.scene_file_path.as_ref().and_then(|path| path.parent()) {
            dialog = dialog.initial_directory(parent.to_path_buf());
        }
        if save {
            dialog.save_file();
        } else {
            dialog.pick_file();
        }
        self.scene_picker = Some((save, dialog));
    }

    fn submit_scene_file(&mut self, mut path: PathBuf, save: bool) {
        self.world.finish_edit();
        if save && path.extension().is_none() {
            path.set_extension("frac.json");
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
                self.scene_file_pending = Some((id, self.load_revision));
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
            picker.take_picked().map(|path| (*save, path))
        } else {
            None
        };
        if let Some((save, path)) = picked {
            self.scene_picker = None;
            self.submit_scene_file(path, save);
        }
    }

    fn scene_file_event(&mut self, event: crate::io_service::SceneEvent) {
        if self.scene_file_pending != Some((event.id, self.load_revision)) {
            return;
        }
        self.scene_file_pending = None;
        match event.result {
            Ok(Some(scene)) => {
                self.load(*scene);
                self.scene_file_path = Some(event.path.clone());
                self.status = format!("Opened {}", event.path.display());
            }
            Ok(None) => {
                self.scene_file_path = Some(event.path.clone());
                self.status = format!("Saved {}", event.path.display());
            }
            Err(error) => self.status = format!("Scene file {}: {error}", event.path.display()),
        }
    }

    fn load(&mut self, mut scene: Scene) {
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
        self.world.finish_edit();
        self.world.selection = Some(id);
        self.world.selected.clear();
        self.world.selected.push(id);
        self.panels_to_open.push(dock::Panel::Inspector);
    }

    fn material_assignment_targets(&self) -> Vec<crate::world::NodeId> {
        let selected: &[crate::world::NodeId] = if self.world.selected.is_empty() {
            self.world.selection.as_slice()
        } else {
            &self.world.selected
        };
        selected
            .iter()
            .copied()
            .filter(|id| {
                self.world
                    .document
                    .info(*id)
                    .is_ok_and(|node| node.kind == crate::world::WorldKind::Fractal)
            })
            .collect()
    }

    fn assign_gallery_material(&mut self, material: crate::world::NodeId) -> Result<(), String> {
        let commands = self
            .material_assignment_targets()
            .into_iter()
            .map(|id| crate::world::WorldCommand::AssignMaterial {
                id,
                material: Some(material),
            })
            .collect::<Vec<_>>();
        if commands.is_empty() {
            return Err("Select an object that supports materials".into());
        }
        self.world
            .execute(crate::world::WorldCommand::Batch(commands))
    }

    fn materials_tab(&mut self, ui: &mut egui::Ui) {
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
        let mut assign = None;
        let mut refresh = None;
        let mut apply_preset = None;
        let mut command = None;
        egui::ScrollArea::vertical()
            .id_salt("material_nodes")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                let gap = 6.0;
                let columns = ((ui.available_width() + gap) / (SWATCH as f32 + gap))
                    .floor()
                    .max(1.0) as usize;
                let size = Vec2::splat(
                    ((ui.available_width() - gap * (columns - 1) as f32) / columns as f32)
                        .clamp(1.0, SWATCH as f32),
                );
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
                                let eligible = !self.material_assignment_targets().is_empty();
                                if ui
                                    .add_enabled(
                                        eligible,
                                        egui::Button::new("Assign to selected objects"),
                                    )
                                    .clicked()
                                {
                                    assign = Some(entry.id);
                                    ui.close();
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
                                    command = Some(WorldCommand::Duplicate(entry.id));
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
        if let Some(id) = assign {
            self.status = match self.assign_gallery_material(id) {
                Ok(()) => "Material assigned to selected objects".into(),
                Err(error) => error,
            };
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

    fn save_bookmark(&mut self) {
        let dir = data_dir().join("bookmarks");
        let path = dir.join(format!(
            "{}-{}.json",
            now_stamp(),
            crate::slug(&self.scene.name)
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

    fn screenshot(&mut self) {
        self.save_frame(false);
    }
    fn screenshot_exr(&mut self) {
        self.save_frame(true);
    }
    fn save_frame(&mut self, exr: bool) {
        let Some(frame) = self.frame.clone() else {
            self.status = "Nothing rendered yet".into();
            return;
        };
        let path = pictures_dir().join(format!(
            "{}-{}.{}",
            crate::slug(&self.scene.name),
            now_stamp(),
            if exr { "display.exr" } else { "png" }
        ));
        self.status = match self
            .io
            .send(crate::io_service::Command::SaveFrame { frame, path, exr })
        {
            Ok(()) => "Saving image…".into(),
            Err(e) => e,
        };
    }
    fn start_job(&mut self) {
        let (w, h, _) = FINAL_SIZES[self.final_size];
        let path = pictures_dir().join(format!(
            "{}-{}x{}-{}spp-{}.png",
            crate::slug(&self.scene.name),
            w,
            h,
            self.final_spp,
            now_stamp()
        ));
        let scene = match self
            .world
            .document
            .snapshot(f64::from(self.world_ui.playhead))
        {
            Ok(scene) => scene,
            Err(error) => {
                self.status = error;
                return;
            }
        };
        match self.renderer.try_command(Command::RenderExport {
            reply: None,
            id: 1,
            scene,
            width: w as usize,
            height: h as usize,
            spp: self.final_spp,
        }) {
            Ok(()) => {
                self.job = Some(Job {
                    width: w as usize,
                    height: h as usize,
                    samples: 0,
                    spp: self.final_spp,
                    path,
                })
            }
            Err(e) => self.status = e,
        }
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
            r.scene != snapshot || r.width != w || r.height != h || r.seed != self.seed
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
            interactive: self.world_ui.playing
                || self.last_change.elapsed().as_secs_f32() < PREVIEW_HOLD_S,
            seed: self.seed,
            output_hdr,
            white_nits,
        };
        self.renderer.request_viewport(req.clone());
        self.request = Some(req);
    }
    fn poll_events(&mut self, ctx: &egui::Context) {
        if let Some(frame) = self.renderer.take_latest_frame()
            && self.preview.position().is_none()
        {
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
                RenderEvent::ExportFailed { id: 1, error } => {
                    self.job = None;
                    self.status = format!("Render failed: {error}");
                }
                RenderEvent::Error(error) => {
                    if let Some((id, _)) = self.thumb_pending.take() {
                        self.material_gallery.fail_thumbnail(id, error.clone());
                    }
                    self.status = error;
                    self.job = None;
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
                RenderEvent::ExportProgress { id: 1, samples, .. } => {
                    if let Some(job) = &mut self.job {
                        job.samples = samples;
                    }
                }
                RenderEvent::ExportFrame { id: 1, frame } => {
                    if let Some(job) = self.job.take()
                        && let Err(error) = self.io.send(crate::io_service::Command::SaveFrame {
                            frame,
                            path: job.path,
                            exr: false,
                        })
                    {
                        self.status = error;
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
    fn export_ui(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("Still image · PNG")
            .default_open(true)
            .show(ui, |ui| self.still_image_ui(ui));
        ui.separator();
        let timeline = (
            self.world.document.first,
            self.world.document.last,
            self.world.document.fps,
        );
        let scene = &self.scene;
        let world = &mut self.world;
        ui.add_enabled_ui(self.job.is_none(), |ui| {
            self.export.ui(ui, timeline, &self.renderer, || {
                world.finish_edit();
                freeze_world_scene(scene, &world.document)
            });
        });
    }

    fn still_image_ui(&mut self, ui: &mut egui::Ui) {
        egui::ComboBox::from_id_salt("final_size")
            .selected_text(FINAL_SIZES[self.final_size].2)
            .show_ui(ui, |ui| {
                for (i, size) in FINAL_SIZES.iter().enumerate() {
                    ui.selectable_value(&mut self.final_size, i, size.2);
                }
            });
        ui.add(
            egui::DragValue::new(&mut self.final_spp)
                .range(1..=65536)
                .suffix(" spp"),
        );
        if let Some(job) = &self.job {
            ui.add(egui::ProgressBar::new(job.samples as f32 / job.spp as f32).show_percentage());
            if ui.button("Cancel image render").clicked() {
                let _ = self.renderer.try_command(Command::CancelExport);
                self.job = None;
            }
        } else if ui
            .add_enabled(!self.export.is_running(), egui::Button::new("Render PNG"))
            .clicked()
        {
            self.start_job();
        }
    }

    /// Unreal-style flight: hold RMB in the viewport, mouse looks, WASD moves, R/C up/down, Q/E rolls,
    /// Shift boosts, the wheel scales the speed. Integrated by cam-controls `SpaceFlight` (thrust,
    /// inertia, damping); on release the orbit pivot is placed in front of the camera at the
    /// current orbit distance, so orbiting continues from where you flew.
    fn fly_camera(&mut self, ui: &egui::Ui, resp: &egui::Response) {
        use cam_controls::{CameraIntent, CameraPose, InertiaSettings, SpaceFlight};
        use glam::{Quat, Vec3};
        if !ui.ctx().text_edit_focused()
            && ui.input(|i| {
                i.events.iter().any(|e| {
                    matches!(e,
            egui::Event::Key { key, physical_key, pressed: true, repeat: false, modifiers }
            if (*key == egui::Key::Backtick || *physical_key == Some(egui::Key::Backtick))
                && !modifiers.command && !modifiers.ctrl && !modifiers.alt)
                })
            })
        {
            self.toggle_flight_mode();
        }
        let held = resp.is_pointer_button_down_on()
            && ui.input(|i| i.focused && i.pointer.secondary_down());
        if !ui.ctx().text_edit_focused() {
            let shortcut = |key| {
                ui.input(|i| {
                    !i.modifiers.command
                        && !i.modifiers.ctrl
                        && !i.modifiers.alt
                        && i.key_pressed(key)
                })
            };
            if shortcut(egui::Key::H) {
                self.scene.camera = self.origin.camera;
                self.fly = None;
                ui.ctx()
                    .send_viewport_cmd(egui::ViewportCommand::CursorGrab(egui::CursorGrab::None));
                self.status = "Camera restored".into();
                return;
            }
            if !held && shortcut(egui::Key::F) {
                self.frame_camera(
                    resp.rect.width().max(1.0) as u32,
                    resp.rect.height().max(1.0) as u32,
                );
                return;
            }
        }
        let radius = self
            .scene
            .camera_reference
            .unwrap_or(self.scene.formula.framing_radius());
        let cam = &mut self.scene.camera;
        // Roll releases the horizon lock rather than being silently discarded.
        if held && ui.input(|i| i.key_down(egui::Key::Q) || i.key_down(egui::Key::E)) {
            cam.free_flight = true;
        }
        let dist = cam.distance * radius;
        if held && self.fly.is_none() {
            let orientation = cam.orientation();
            let eye = Vec3::from_array(cam.target) - (orientation * -Vec3::Z) * dist;
            let pose = CameraPose {
                eye,
                orientation,
                ..CameraPose::default()
            };
            let mut fly = SpaceFlight::from_pose(pose);
            fly.inertia = InertiaSettings::fps();
            self.fly = Some(fly);
        }
        let Some(fly) = &mut self.fly else { return };
        let viewport = cam_viewport::ViewportSize::new(
            resp.rect.width().max(1.0) as u32,
            resp.rect.height().max(1.0) as u32,
        );
        let dt = ui.input(|i| i.stable_dt).clamp(1.0e-4, 0.1);
        if held {
            ui.ctx().set_cursor_icon(egui::CursorIcon::None);
            ui.ctx()
                .send_viewport_cmd(egui::ViewportCommand::CursorGrab(egui::CursorGrab::Locked));
            let (delta, scroll, keys) = ui.input(|i| {
                let k = |key| if i.key_down(key) { 1.0f32 } else { 0.0 };
                (
                    i.pointer.motion().unwrap_or_else(|| i.pointer.delta()),
                    i.smooth_scroll_delta.y,
                    [
                        k(egui::Key::W) - k(egui::Key::S),
                        k(egui::Key::D) - k(egui::Key::A),
                        k(egui::Key::R) - k(egui::Key::C),
                        if i.modifiers.shift { 1.0 } else { 0.0 },
                        k(egui::Key::E) - k(egui::Key::Q),
                    ],
                )
            });
            if scroll != 0.0 {
                self.controls.fly_speed =
                    (self.controls.fly_speed * (scroll * 0.003).exp()).clamp(0.02, 50.0);
                self.status = format!("Flight speed ×{:.2}", self.controls.fly_speed);
            }
            let sens = 0.001 * self.controls.look_sensitivity;
            fly.apply_intent(
                CameraIntent::Look {
                    dyaw: -delta.x * sens,
                    dpitch: -delta.y * sens,
                },
                viewport,
            );
            fly.apply_intent(
                CameraIntent::Thrust {
                    forward: keys[0],
                    right: keys[1],
                    up: keys[2],
                },
                viewport,
            );
            fly.apply_intent(CameraIntent::Boost(keys[3] > 0.0), viewport);
            fly.apply_intent(CameraIntent::Roll { d: keys[4] }, viewport);
        } else {
            ui.ctx()
                .send_viewport_cmd(egui::ViewportCommand::CursorGrab(egui::CursorGrab::None));
            fly.apply_intent(
                CameraIntent::Thrust {
                    forward: 0.0,
                    right: 0.0,
                    up: 0.0,
                },
                viewport,
            );
            fly.apply_intent(CameraIntent::Boost(false), viewport);
            fly.apply_intent(CameraIntent::Roll { d: 0.0 }, viewport);
        }
        // Acceleration scales with the formula's size and the speed multiplier.
        fly.inertia.thrust_sensitivity = 6.0 * radius * self.controls.fly_speed;
        let moving = fly.update_dynamics(dt);

        let pose = fly.pose();
        if cam.free_flight {
            let (yaw, pitch, roll) = pose.orientation.to_euler(glam::EulerRot::YXZ);
            cam.yaw_degrees = yaw.to_degrees();
            cam.pitch_degrees = -pitch.to_degrees();
            cam.roll_degrees = roll.to_degrees();
        } else {
            let f = pose.forward();
            cam.pitch_degrees = (-f.y)
                .clamp(-1.0, 1.0)
                .asin()
                .to_degrees()
                .clamp(-89.0, 89.0);
            cam.yaw_degrees = (-f.x).atan2(-f.z).to_degrees();
            cam.roll_degrees = 0.0;
            fly.orientation = Quat::from_axis_angle(Vec3::Y, cam.yaw_degrees.to_radians())
                * Quat::from_axis_angle(Vec3::X, -cam.pitch_degrees.to_radians());
        }
        let f = fly.pose().forward();
        let target = pose.eye + f * dist;
        cam.target = [target.x, target.y, target.z];
        if !held && !moving {
            self.fly = None;
        }
        ui.ctx().request_repaint();
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

    fn toggle_flight_mode(&mut self) {
        use glam::{Quat, Vec3};
        let cam = &mut self.scene.camera;
        let dist = cam.distance
            * self
                .scene
                .camera_reference
                .unwrap_or(self.scene.formula.framing_radius());
        let eye = self.fly.as_ref().map_or_else(
            || Vec3::from_array(cam.target) - (cam.orientation() * -Vec3::Z) * dist,
            |fly| fly.eye,
        );
        cam.free_flight = !cam.free_flight;
        if !cam.free_flight {
            let f = cam.orientation() * -Vec3::Z;
            cam.yaw_degrees = (-f.x).atan2(-f.z).to_degrees();
            cam.pitch_degrees = (-f.y)
                .clamp(-1.0, 1.0)
                .asin()
                .to_degrees()
                .clamp(-89.0, 89.0);
            cam.roll_degrees = 0.0;
            let orientation: Quat = cam.orientation();
            cam.target = (eye + (orientation * -Vec3::Z) * dist).to_array();
            if let Some(fly) = &mut self.fly {
                fly.orientation = orientation;
                fly.momentum.angular.z = 0.0;
                fly.apply_intent(
                    cam_controls::CameraIntent::Roll { d: 0.0 },
                    cam_viewport::ViewportSize::new(1, 1),
                );
            }
        }
        self.status = if cam.free_flight {
            "Flight: free · Q/E roll · R/C up/down"
        } else {
            "Flight: horizon · R/C up/down · Q/E enables roll"
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
                ui.menu_button("Animation presets", |ui| {
                    for (index, preset) in crate::presets::ANIMATED.iter().enumerate() {
                        if ui
                            .button(preset.name)
                            .on_hover_text(preset.description)
                            .clicked()
                        {
                            match crate::presets::scene(index) {
                                Ok(scene) => self.load(scene),
                                Err(error) => self.status = error,
                            }
                            ui.close();
                        }
                    }
                });
                ui.separator();
                if ui.button("Save bookmark").clicked() {
                    self.save_bookmark();
                    ui.close();
                }
                if ui.button("Save image…").clicked() {
                    self.screenshot();
                    ui.close();
                }
                if ui.button("Save display EXR…").clicked() {
                    self.screenshot_exr();
                    ui.close();
                }
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
        let size = Vec2::new(ui.available_width().min(THUMB_W as f32), 0.0);
        let size = Vec2::new(size.x, size.x * THUMB_H as f32 / THUMB_W as f32);
        egui::ScrollArea::vertical().show(ui, |ui| {
            let entries = if self.tab == Tab::Gallery {
                &self.gallery
            } else {
                &self.bookmarks
            };
            if entries.is_empty() {
                ui.label("No bookmarks yet: ★ Bookmark saves the current scene.");
            }
            for (i, e) in entries.iter().enumerate() {
                let selected = e.scene == self.origin;
                let frame = egui::Frame::group(ui.style()).stroke(if selected {
                    egui::Stroke::new(2.0, ui.visuals().selection.stroke.color)
                } else {
                    ui.visuals().widgets.noninteractive.bg_stroke
                });
                let resp = frame
                    .show(ui, |ui| {
                        ui.vertical(|ui| {
                            match &e.thumb {
                                Some(t) => {
                                    ui.add(egui::Image::new((t.id(), size)));
                                }
                                None => {
                                    let (r, _) = ui.allocate_exact_size(size, Sense::hover());
                                    ui.painter()
                                        .rect_filled(r, 4.0, ui.visuals().extreme_bg_color);
                                    ui.painter().text(
                                        r.center(),
                                        egui::Align2::CENTER_CENTER,
                                        "rendering…",
                                        egui::FontId::proportional(12.0),
                                        ui.visuals().weak_text_color(),
                                    );
                                }
                            }
                            ui.label(RichText::new(&e.scene.name).strong());
                            ui.label(
                                RichText::new(format!(
                                    "{} · {}",
                                    e.scene.formula.name(),
                                    if e.scene.material.model == MaterialModel::StandardSurface {
                                        "Standard Surface"
                                    } else {
                                        "fast"
                                    }
                                ))
                                .small()
                                .weak(),
                            );
                        });
                    })
                    .response
                    .interact(Sense::click());
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
            }
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
            .tint(Color32::from_rgb(70, 130, 200), 0.28)
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
        const WIDTHS: [f32; 9] = [200.0, 95.0, 95.0, 220.0, 130.0, 100.0, 170.0, 80.0, 0.0];
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
                                ui.colored_label(Color32::from_rgb(230, 180, 60), "preview");
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
                                ui.colored_label(Color32::from_rgb(230, 180, 60), "OIDN failed")
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
                    ui.label(RichText::new("preview").color(Color32::from_rgb(230, 180, 60)));
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
                    ui.label(RichText::new("OIDN failed").color(Color32::from_rgb(230, 180, 60)))
                        .on_hover_text(error);
                }
            }
            ui.separator();
            ui.label(format!("UI {:.0} fps", 1000.0 / self.frame_ms.max(0.1)));
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
        let cam = &mut self.scene.camera;
        if resp.dragged_by(egui::PointerButton::Primary) {
            let d = resp.drag_delta();
            let sensitivity = 0.1 * self.controls.look_sensitivity;
            cam.yaw_degrees = (cam.yaw_degrees - d.x * sensitivity) % 360.0;
            cam.pitch_degrees = (cam.pitch_degrees + d.y * sensitivity).clamp(-89.0, 89.0);
        }
        if resp.dragged_by(egui::PointerButton::Middle) {
            // Pan the orbit target in the camera plane.
            let d = resp.drag_delta();
            let radius = self
                .scene
                .camera_reference
                .unwrap_or(self.scene.formula.framing_radius());
            let orientation = cam.orientation();
            let right = (orientation * glam::Vec3::X).to_array();
            let up = (orientation * glam::Vec3::Y).to_array();
            let k = cam.distance * radius * 2.0 * (cam.fov_y_degrees.to_radians() * 0.5).tan()
                / avail.y.max(1.0);
            for i in 0..3 {
                cam.target[i] += (-d.x * right[i] + d.y * up[i]) * k;
            }
        }
        if resp.hovered() && !toolbar.contains_pointer && self.fly.is_none() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                cam.distance = (cam.distance * (-scroll * 0.0015).exp()).max(0.05);
            }
        }
        if resp.double_clicked() {
            cam.target = [0.0; 3];
        }

        let ppp = ui.ctx().pixels_per_point() * self.resolution;
        let (w, h) = (
            ((avail.x * ppp) as usize).max(16),
            ((avail.y * ppp) as usize).max(16),
        );
        let state = ui
            .ctx()
            .data(|d| d.get_temp::<egui_display::DisplayState>(egui_display::state_id()));
        let output_hdr = state.as_ref().is_some_and(|s| s.output.is_hdr());
        let white = state.as_ref().map_or(100.0, |s| s.target.white);
        if self.preview.position().is_none() {
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
        if let Some(job) = &self.job {
            let text = format!(
                "Rendering {}×{} · {}/{} spp",
                job.width, job.height, job.samples, job.spp
            );
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
    fn preview_identity(&self, ctx: &egui::Context) -> (playa_graph::NodeId, u64, bool, u32, u32) {
        let state =
            ctx.data(|d| d.get_temp::<egui_display::DisplayState>(egui_display::state_id()));
        (
            playa_graph::NodeId(self.world.document.graph.id),
            self.world.revision(),
            state.as_ref().is_some_and(|s| s.output.is_hdr()),
            state.as_ref().map_or(100.0, |s| s.target.white).to_bits(),
            self.seed,
        )
    }

    fn begin_preview(&mut self, first: u32, last: u32, cache_all: bool, ctx: &egui::Context) {
        self.world.finish_edit();
        self.preview_sequence = self.preview_sequence.wrapping_add(1);
        let key = self.preview_identity(ctx);
        let (width, height) = self.request.as_ref().map_or((640, 360), |r| {
            let scale = (640.0 / r.width.max(r.height) as f64).min(1.0);
            (
                (r.width as f64 * scale).round().max(16.0) as usize,
                (r.height as f64 * scale).round().max(16.0) as usize,
            )
        });
        let request = crate::preview::PreviewRequest {
            generation: self.preview_sequence,
            scene: Arc::new(self.frozen_world_scene()),
            first,
            last,
            fps: self.world.document.fps as f32,
            width,
            height,
            spp: 8,
            seed: self.seed,
            output_hdr: key.2,
            white_nits: f32::from_bits(key.3),
            cache_fraction: 0.05,
            reserve_gb: 2.0,
        };
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
            self.preview.pause();
            self.world_ui.playing = false;
        }
        if self
            .preview_key
            .is_some_and(|key| key != self.preview_identity(ctx))
        {
            self.preview.cancel();
            self.preview_key = None;
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
                    "Caching preview: {}/{} frames · {} MiB",
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
        match self.world_ui.take_preview_action() {
            Some(crate::world_ui::PreviewAction::Play { first, last }) => {
                self.begin_preview(first, last, false, ctx)
            }
            Some(crate::world_ui::PreviewAction::CacheThenPlay { first, last }) => {
                self.begin_preview(first, last, true, ctx)
            }
            None if self.preview.position().is_none() && self.world_ui.playing => {
                let head = self.world_ui.playhead;
                let (first, last) = self
                    .world_ui
                    .playback_range
                    .unwrap_or((self.world.document.first, self.world.document.last));
                self.begin_preview(first, last, false, ctx);
                self.preview.seek(head.clamp(first, last));
                self.world_ui.seek(head.clamp(first, last));
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
            self.legacy_before = Some(self.scene.clone());
            #[cfg(test)]
            {
                self.legacy_snapshots += 1;
            }
        }
        Ok(())
    }
    pub(crate) fn ui(&mut self, root: &mut egui::Ui) {
        let ctx = root.ctx().clone();
        self.apply_fonts(&ctx);
        self.update_scene_picker(&ctx);
        self.colour.poll();
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
        if !ctx.text_edit_focused() {
            if ctx.input(|i| i.key_pressed(egui::Key::Tab)) {
                self.show_ui = !self.show_ui;
            }
            if ctx.input(|i| i.modifiers.command && i.key_pressed(egui::Key::Z)) {
                if ctx.input(|i| i.modifiers.shift) {
                    self.world.redo();
                } else {
                    self.world.undo();
                }
            }
        }

        if let Err(error) = self.refresh_scene() {
            self.status = error;
        }
        let before = self.legacy_before.take();
        let thumbs_pending = self.render_one_thumbnail(&ctx);
        self.poll_events(&ctx);
        self.export.update(&self.renderer);
        self.viewport_visible = false;

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

        if !ctx.text_edit_focused()
            && !ctx.input(|i| i.pointer.secondary_down())
            && ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Space))
        {
            self.world_ui.playing = !self.world_ui.playing;
        }

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
            let gesture = crate::world_ui::parameter_gesture(&ctx)
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
        self.legacy_before = before;
        self.process_preview_intent(&ctx);
        self.world
            .finish_edit_unless(crate::world_ui::parameter_gesture(&ctx).or(camera_gesture));
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
    fn material_library_adapts_columns_to_available_panel_width() {
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
    fn material_card_selects_existing_attribute_editor_without_assignment() {
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
    fn gallery_assignment_and_object_material_field_share_uuid_and_undo() {
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
        app.assign_gallery_material(material).unwrap();
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
        let mut app = App::new();
        let ctx = egui::Context::default();
        assert!(!serde_json::from_str::<Settings>("{}").unwrap().auto_key);
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
        let mut app = App::new();
        let ctx = egui::Context::default();
        app.refresh_scene().unwrap();
        let key = app.evaluated_world;
        let objects = app.scene.objects.as_ptr();
        let snapshots = app.legacy_snapshots;
        assert!(app.request.is_none());
        app.begin_preview(0, 10, false, &ctx);
        for frame in 1..=10 {
            app.world_ui.seek(frame);
            app.refresh_scene().unwrap();
            assert_eq!(app.evaluated_world, key);
            assert_eq!(app.scene.objects.as_ptr(), objects);
            assert_eq!(app.legacy_snapshots, snapshots);
        }
        app.preview.pause();
        app.world_ui.playing = false;
        app.refresh_scene().unwrap();
        assert_ne!(app.evaluated_world, key);
    }

    #[test]
    fn file_load_completion_preserves_document_and_discards_stale_requests() {
        let mut app = App::new();
        let mut scene = Scene::preset(crate::params::FAMILY_BOX);
        let document = crate::world::WorldDocument::from_scene(&scene);
        let expected = serde_json::to_value(&document).unwrap();
        scene.document = Some(Box::new(document));
        app.scene_file_pending = Some((7, app.load_revision));
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
        let clones = app.legacy_snapshots;
        app.refresh_scene().unwrap();
        assert_eq!(app.world.revision(), revision);
        assert_eq!(app.legacy_snapshots, clones);
        app.scene_file_pending = Some((8, app.load_revision));
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
        let mut app = App::new();
        let ctx = egui::Context::default();
        for frame in 0..32 {
            let _ = ctx.run_ui(egui::RawInput::default(), |root| {
                egui::CentralPanel::default().show(root, |ui| {
                    app.export
                        .ui(ui, (frame, frame + 250, 24.0), &app.renderer, || {
                            panic!("An idle export panel must not clone the authoring document")
                        });
                });
            });
        }
    }

    #[test]
    fn settings_and_layouts_serialize_only_after_actual_changes() {
        let mut app = App::new();
        let ctx = egui::Context::default();
        app.refresh_scene().unwrap();
        assert!(app.changed_settings_json(&ctx).unwrap().is_some());
        let serializations = app.settings_serializations;
        let dock_serializations = app.layouts.cache.serializations;
        let baseline = app.legacy_before.as_ref().unwrap().objects.as_ptr();
        let snapshots = app.legacy_snapshots;
        for _ in 0..32 {
            app.refresh_scene().unwrap();
            assert!(app.changed_settings_json(&ctx).unwrap().is_none());
            assert_eq!(
                app.legacy_before.as_ref().unwrap().objects.as_ptr(),
                baseline
            );
        }
        assert_eq!(app.settings_serializations, serializations);
        assert_eq!(app.layouts.cache.serializations, dock_serializations);
        assert_eq!(app.legacy_snapshots, snapshots);
        app.world_ui.attribute_metrics.numeric_width = 64.0;
        app.fonts.face = "Custom face".into();
        app.status_layout.widths = vec![210.0, 100.0];
        app.gui_fps = 144;
        let mut export = app.export.settings().clone();
        export.output = "renders/new.exr".into();
        export.qp = 19;
        app.export.restore(export);
        let json = app.changed_settings_json(&ctx).unwrap().unwrap();
        let saved: Settings = serde_json::from_str(&json).unwrap();
        assert_eq!(saved.attribute_metrics.numeric_width, 64.0);
        assert_eq!(saved.fonts.face, "Custom face");
        assert_eq!(saved.status_layout.widths, [210.0, 100.0]);
        assert_eq!(saved.gui_fps, 144);
        assert_eq!(saved.export.output, "renders/new.exr");
        assert_eq!(saved.export.qp, 19);
        assert!(app.changed_settings_json(&ctx).unwrap().is_none());
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
    fn viewport_toolbar_preserves_rmb_flight_and_continuous_camera_updates() {
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
        assert_eq!(app.scene.camera.roll_degrees, 0.0);
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
        frame(
            &mut app,
            vec![
                flight_key(egui::Key::C, false),
                flight_key(egui::Key::Q, true),
            ],
        );
        for _ in 0..30 {
            frame(&mut app, vec![]);
        }
        assert!(
            app.scene.camera.free_flight,
            "Q/E must release the horizon lock"
        );
        assert!(
            app.scene.camera.roll_degrees.abs() > 1.0,
            "Q must roll the camera"
        );
        frame(
            &mut app,
            vec![flight_key(egui::Key::Q, false), button(false)],
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
        let legacy: Settings = serde_json::from_str("{}").unwrap();
        assert_eq!(legacy.controls.look_sensitivity, 1.0);
        app.controls.look_sensitivity = 0.4;
        app.controls.fly_speed = 3.0;
        let loaded: Controls =
            serde_json::from_str(&serde_json::to_string(&app.controls).unwrap()).unwrap();
        assert_eq!(loaded.look_sensitivity, 0.4);
        assert_eq!(loaded.fly_speed, 3.0);
    }

    #[test]
    fn shared_flight_inertia_coasts_and_stops_on_all_six_axes() {
        use cam_controls::{CameraIntent, InertiaSettings, SpaceFlight};
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
            fly.inertia = InertiaSettings::fps();
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
        let loaded: Scene = serde_json::from_value(old).unwrap();
        assert!(!loaded.camera.free_flight);
        assert_eq!(loaded.camera.roll_degrees, 0.0);
    }
}
