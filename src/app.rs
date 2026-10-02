//! The browser UI (egui): gallery + bookmarks with GPU thumbnails, the progressive viewport,
//! the scene inspector, screenshots and final PNG renders.

use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use egui::{self, Color32, ColorImage, RichText, Sense, TextureHandle, TextureOptions, Vec2};

use crate::palette::{PaletteScheme, build_lut};
use crate::render::{Gpu, Target};
use crate::scene::*;

const THUMB_W: usize = 224;
const THUMB_H: usize = 126;
const THUMB_SPP: u32 = 24;
/// GPU time a frame may spend tracing, in ms (keeps the UI responsive).
const FRAME_BUDGET_MS: f32 = 28.0;
/// After the last camera / parameter change, keep the low-resolution preview this long.
const PREVIEW_HOLD_S: f32 = 0.18;

const FINAL_SIZES: [(u32, u32, &str); 5] = [
    (1280, 720, "720p"),
    (1920, 1080, "1080p"),
    (2560, 1440, "1440p"),
    (3840, 2160, "4K"),
    (2048, 2048, "2048²"),
];

pub fn run() -> anyhow::Result<()> { crate::window::run() }

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
    s.material.model = MaterialModel::StandardSurface;
    preset.apply(&mut s.material);
    s.material.model = MaterialModel::StandardSurface;
    s
}

struct Job {
    scene: Scene,
    target: Target,
    spp: u32,
    path: PathBuf,
    started: Instant,
}

pub(crate) struct App {
    gpu: Gpu,
    scene: Scene,
    /// The preset / bookmark the scene came from, for "Reset".
    origin: Scene,
    full: Option<Target>,
    preview: Option<Target>,
    showing_preview: bool,
    hdr_view: std::sync::Arc<std::sync::Mutex<egui_hdr_view::HdrView>>,
    pub display: egui_display::DisplayPrefs,
    colour: crate::ocio::State,
    prefs: egui_prefs2::PrefsPanelState,
    settings_open: bool,
    config_picker: egui_file_dialog::FileDialog,
    saved_settings: String,
    gallery: Vec<Entry>,
    bookmarks: Vec<Entry>,
    thumb_target: Target,
    swatch_target: Target,
    swatches: Vec<Option<TextureHandle>>,
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
    /// Unreal-style flight (gitnexus-rs cam-controls `FpsFly`), alive while RMB is held and
    /// while its momentum coasts after release.
    fly: Option<cam_controls::FpsFly>,
    /// Flight speed multiplier (mouse wheel while flying).
    fly_speed: f32,
}

fn now_stamp() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn data_dir() -> PathBuf {
    dirs::data_dir().unwrap_or_else(|| PathBuf::from(".")).join("frac-rs")
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
            Some((p, serde_json::from_str::<Scene>(&s).ok()?))
        })
        .collect();
    v.sort_by(|a, b| a.0.cmp(&b.0));
    v.into_iter().map(|(p, scene)| Entry { scene, thumb: None, path: Some(p) }).collect()
}

fn to_image(t: &Target) -> ColorImage {
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
struct Settings {
    display: egui_display::DisplayPrefs,
    colour: crate::ocio::Sel,
    panel: egui_prefs2::PrefsPanelState,
}
impl Default for Settings {
    fn default() -> Self { Self { display: Default::default(), colour: crate::color::default_selection(), panel: Default::default() } }
}
fn settings_path() -> PathBuf { dirs::config_dir().unwrap_or_else(|| PathBuf::from(".")).join("frac-rs/settings.json") }

impl App {
    pub(crate) fn new() -> Self {
        let gpu = Gpu::new().unwrap_or_else(|e| panic!("CUDA init failed: {e}"));
        let gallery: Vec<Entry> =
            Scene::gallery().into_iter().map(|scene| Entry { scene, thumb: None, path: None }).collect();
        let scene = gallery[0].scene.clone();
        let thumb_target = gpu.target(THUMB_W, THUMB_H);
        let swatch_target = gpu.target(SWATCH, SWATCH);
        Self {
            gpu,
            origin: scene.clone(),
            last_scene: scene.clone(),
            scene: scene.clone(),
            full: None,
            preview: None,
            showing_preview: false,
            hdr_view: std::sync::Arc::new(std::sync::Mutex::new(egui_hdr_view::HdrView::new())),
            display: Default::default(),
            colour: crate::ocio::State::new(scene.colour.clone()),
            prefs: Default::default(), settings_open: false,
            config_picker: egui_file_dialog::FileDialog::new(), saved_settings: String::new(),
            gallery,
            bookmarks: load_bookmarks(),
            thumb_target,
            swatch_target,
            swatches: (0..crate::materials::PRESETS.len()).map(|_| None).collect(),
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
            fly_speed: 1.0,
            snap: std::env::var("FRAC_SNAP").ok().map(|p| {
                let spp = std::env::var("FRAC_SNAP_SPP").ok().and_then(|v| v.parse().ok()).unwrap_or(256);
                (PathBuf::from(p), spp, false)
            }),
        }
        .with_snap_preset().with_settings()
    }

    fn with_settings(mut self) -> Self {
        if let Ok(data) = std::fs::read(settings_path())
            && let Ok(settings) = serde_json::from_slice::<Settings>(&data) {
                self.display = settings.display;
                self.prefs = settings.panel;
                self.colour = crate::ocio::State::new(settings.colour.clone());
                self.scene.colour = settings.colour;
                self.origin.colour = self.scene.colour.clone();
        }
        if let Ok(category) = std::env::var("FRAC_SETTINGS") {
            self.settings_open = true;
            self.prefs.selected = usize::from(category.eq_ignore_ascii_case("color"));
        }
        self
    }

    /// Settings layout and category bodies adapted directly from exr-view::ui_settings.
    fn settings_ui(&mut self, ctx: &egui::Context) {
        let state = ctx.data(|d| d.get_temp::<egui_display::DisplayState>(egui_display::state_id()));
        self.colour.set_hdr(state.as_ref().is_some_and(|s| s.output.is_hdr()));
        self.colour.set_filterable(gpu_info::shared_device().is_some_and(|g| g.device.features().contains(wgpu::Features::FLOAT32_FILTERABLE)));
        let mut open = self.settings_open;
        let mut browse = false;
        let mut changed = false;
        let mut reset = false;
        let mut category = self.prefs.selected;
        let mut destination = None;
        egui::Window::new("Settings").open(&mut open).default_size([850.0, 600.0]).show(ctx, |ui| {
            let categories = [egui_prefs2::Category::new(egui_phosphor::regular::MONITOR, "Display"), egui_prefs2::Category::new(egui_phosphor::regular::MONITOR, "Color")];
            egui_prefs2::draw(ui, &mut self.prefs, &categories, |ui, idx| {
                category = idx;
                match idx {
                    0 => {
                        egui_display::settings_ui(ui, &mut self.display, state.as_ref());
                        ui.add_space(8.0);
                        if ui.button("Colour management & monitor presets…").clicked() { destination = Some(1); }
                    }
                    _ => {
                        egui_prefs2::section_header(ui, "Colour management");
                        if let Some(state) = &state { ui.label(format!("Window output: {}.", state.output.label())); }
                        ui.label("Monitor presets choose the image rendering. HDR 1000 nits is a rendering peak; SDR reference white controls the brightness of the UI and relative-white image values.");
                        if ui.button("Display output & reference white…").clicked() { destination = Some(0); }
                        ui.add_space(8.0);
                        changed = self.colour.ui(ui, &mut browse);
                    }
                }
            }, Some(|| reset = true));
        });
        self.settings_open = open;
        if reset {
            if category == 0 { self.display = Default::default(); }
            else { self.colour.sel = crate::color::default_selection(); self.colour.reload(); changed = true; }
        }
        if let Some(idx) = destination { self.prefs.selected = idx; }
        if browse { self.config_picker.pick_file(); }
        self.config_picker.update(ctx);
        if let Some(path) = self.config_picker.take_picked() {
            self.colour.set_config(path.to_string_lossy().into_owned());
            changed = true;
        }
        if changed {
            self.scene.render.reinhard = false;
            self.scene.colour = self.colour.sel.clone();
            self.gpu.invalidate_colour();
        }
        let settings = Settings { display: self.display, colour: self.scene.colour.clone(), panel: self.prefs.clone() };
        if let Ok(json) = serde_json::to_string_pretty(&settings)
            && self.saved_settings != json {
                let path = settings_path();
                let result = std::fs::create_dir_all(path.parent().unwrap()).and_then(|_| std::fs::write(&path, &json));
                match result { Ok(()) => self.saved_settings = json, Err(e) => self.status = format!("Settings save failed: {e}") }
        }
    }

    fn with_snap_preset(mut self) -> Self {
        if let Some(i) = std::env::var("FRAC_SNAP_PRESET").ok().and_then(|v| v.parse::<usize>().ok()) {
            if let Some(e) = self.gallery.get(i) {
                let scene = e.scene.clone();
                self.load(scene);
            }
        }
        if let Ok(name) = std::env::var("FRAC_SNAP_MATERIAL") {
            if let Some(p) = crate::materials::PRESETS.iter().find(|p| p.name() == name) {
                p.apply(&mut self.scene.material);
            }
        }
        if std::env::var("FRAC_SNAP_TAB").is_ok_and(|t| t == "materials") {
            self.tab = Tab::Materials;
        }
        self
    }

    fn handle_snap(&mut self, ctx: &egui::Context, thumbs_pending: bool) {
        let Some((_path, spp, requested)) = self.snap.clone() else { return };
        let ready = !thumbs_pending && !self.showing_preview && self.full.as_ref().is_some_and(|t| t.samples >= spp);
        if ready && !requested {
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(Default::default()));
            self.snap = self.snap.take().map(|(path, spp, _)| (path, spp, true));
        }
        ctx.request_repaint();
    }

    fn load(&mut self, scene: Scene) {
        self.origin = scene.clone();
        self.colour.sel = scene.colour.clone();
        self.colour.rebuild();
        self.scene = scene;
    }

    /// Render one pending thumbnail (gallery first, then bookmarks). Returns true if one ran.
    fn render_one_thumbnail(&mut self, ctx: &egui::Context) -> bool {
        let pending = self
            .gallery
            .iter_mut()
            .chain(self.bookmarks.iter_mut())
            .find(|e| e.thumb.is_none());
        let Some(entry) = pending else { return self.render_one_swatch(ctx) };
        let mut s = entry.scene.clone();
        s.render.max_bounces = s.render.max_bounces.min(3);
        let t = &mut self.thumb_target;
        self.gpu.step(t, &s, THUMB_SPP / 2, 7, None);
        self.gpu.step(t, &s, THUMB_SPP / 2, 7, None);
        entry.thumb = Some(ctx.load_texture(format!("thumb-{}", s.name), to_image(t), TextureOptions::LINEAR));
        true
    }

    /// Render one pending material swatch. Returns true if one ran.
    fn render_one_swatch(&mut self, ctx: &egui::Context) -> bool {
        let Some(i) = self.swatches.iter().position(Option::is_none) else { return false };
        let preset = &crate::materials::PRESETS[i];
        let s = swatch_scene(preset);
        let t = &mut self.swatch_target;
        self.gpu.step(t, &s, SWATCH_SPP / 2, 3, None);
        self.gpu.step(t, &s, SWATCH_SPP / 2, 3, None);
        self.swatches[i] = Some(ctx.load_texture(format!("swatch-{}", preset.name()), to_image(t), TextureOptions::LINEAR));
        true
    }

    fn materials_tab(&mut self, ui: &mut egui::Ui) {
        use crate::materials::{CATEGORIES, PRESETS};
        ui.label(RichText::new("usd-rs material library · click to apply").small().weak());
        let current = self.scene.material.preset.clone();
        let mut apply: Option<usize> = None;
        let size = Vec2::splat(((ui.available_width() - 12.0) / 2.0).min(SWATCH as f32));
        egui::ScrollArea::vertical().show(ui, |ui| {
            for cat in CATEGORIES {
                let items: Vec<usize> = (0..PRESETS.len()).filter(|&i| PRESETS[i].category == cat).collect();
                if items.is_empty() {
                    continue;
                }
                egui::CollapsingHeader::new(format!("{cat} ({})", items.len())).default_open(true).show(ui, |ui| {
                    egui::Grid::new(("mat-grid", cat)).spacing([6.0, 6.0]).show(ui, |ui| {
                        for (k, &i) in items.iter().enumerate() {
                            let p = &PRESETS[i];
                            let selected = current.as_deref() == Some(p.name());
                            let resp = ui
                                .vertical(|ui| {
                                    match &self.swatches[i] {
                                        Some(t) => {
                                            ui.add(egui::Image::new((t.id(), size)).corner_radius(6.0));
                                        }
                                        None => {
                                            let (r, _) = ui.allocate_exact_size(size, Sense::hover());
                                            ui.painter().rect_filled(r, 6.0, ui.visuals().extreme_bg_color);
                                        }
                                    }
                                    let mut label = RichText::new(p.name()).small();
                                    if selected {
                                        label = label.strong().color(ui.visuals().selection.stroke.color);
                                    }
                                    ui.label(label);
                                })
                                .response
                                .interact(Sense::click());
                            let mut tip = format!(
                                "{}\nroughness {:.2} · metallic {:.0} · IOR {:.2}",
                                p.name(),
                                p.roughness,
                                p.metallic,
                                p.ior
                            );
                            if p.sheen.is_some() {
                                tip += "\n+ sheen (Standard Surface)";
                            }
                            if p.anisotropy.is_some() {
                                tip += "\n+ anisotropy (Standard Surface)";
                            }
                            if p.facing.is_some() {
                                tip += "\n+ facing mix";
                            }
                            if p.opacity < 1.0 {
                                tip += "\nglass: rendered opaque (no refraction on DE fractals)";
                            }
                            if resp.on_hover_text(tip).clicked() {
                                apply = Some(i);
                            }
                            if k % 2 == 1 {
                                ui.end_row();
                            }
                        }
                    });
                });
            }
        });
        if let Some(i) = apply {
            PRESETS[i].apply(&mut self.scene.material);
            self.status = format!("Material: {}", PRESETS[i].name());
        }
    }

    fn save_bookmark(&mut self) {
        let dir = data_dir().join("bookmarks");
        let path = dir.join(format!("{}-{}.json", now_stamp(), crate::slug(&self.scene.name)));
        let res = std::fs::create_dir_all(&dir)
            .map_err(|e| e.to_string())
            .and_then(|_| serde_json::to_string_pretty(&self.scene).map_err(|e| e.to_string()))
            .and_then(|json| std::fs::write(&path, json).map_err(|e| e.to_string()));
        self.status = match res {
            Ok(()) => {
                self.bookmarks.push(Entry { scene: self.scene.clone(), thumb: None, path: Some(path.clone()) });
                self.tab = Tab::Bookmarks;
                format!("Bookmark saved: {}", path.display())
            }
            Err(e) => format!("Bookmark failed: {e}"),
        };
    }

    fn screenshot(&mut self) {
        let path = pictures_dir().join(format!("{}-{}.png", crate::slug(&self.scene.name), now_stamp()));
        let t = if self.showing_preview { self.preview.as_ref() } else { self.full.as_ref() };
        self.status = match t.map(|t| t.save_png(&path)) {
            Some(Ok(())) => format!("Saved {}", path.display()),
            Some(Err(e)) => format!("Screenshot failed: {e}"),
            None => "Nothing rendered yet".into(),
        };
    }

    fn screenshot_exr(&mut self) {
        let path = pictures_dir().join(format!("{}-{}.display.exr", crate::slug(&self.scene.name), now_stamp()));
        self.status = match self.current().map(|t| t.save_display_exr(&path)) {
            Some(Ok(())) => format!("Saved {}", path.display()),
            Some(Err(e)) => format!("EXR failed: {e}"),
            None => "Nothing to save yet".into(),
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
        self.job = Some(Job {
            scene: self.scene.clone(),
            target: self.gpu.target(w as usize, h as usize),
            spp: self.final_spp,
            path,
            started: Instant::now(),
        });
    }

    /// Advance the final render by about one frame budget; returns true while it runs.
    fn step_job(&mut self) -> bool {
        let Some(job) = &mut self.job else { return false };
        let per = (job.target.last_ms / job.target.last_spp.max(1) as f32).max(0.01);
        let n = if job.target.samples == 0 { 1 } else { ((FRAME_BUDGET_MS / per) as u32).clamp(1, 64) };
        let n = n.min(job.spp - job.target.samples);
        self.gpu.step(&mut job.target, &job.scene, n, 0, None);
        if job.target.samples >= job.spp {
            let secs = job.started.elapsed().as_secs_f32();
            self.status = match job.target.save_png(&job.path) {
                Ok(()) => format!("Rendered {} in {:.1} s", job.path.display(), secs),
                Err(e) => format!("Render failed: {e}"),
            };
            self.job = None;
            return false;
        }
        true
    }

    /// Trace the viewport for this frame (preview while the scene is changing).
    fn step_viewport(&mut self, w: usize, h: usize) {
        let mut traced_scene = self.scene.clone();
        traced_scene.render.exposure_stops = self.last_scene.render.exposure_stops;
        traced_scene.render.saturation = self.last_scene.render.saturation;
        traced_scene.render.reinhard = self.last_scene.render.reinhard;
        traced_scene.colour = self.last_scene.colour.clone();
        if traced_scene != self.last_scene {
            self.last_change = Instant::now();
        }
        self.last_scene = self.scene.clone();
        let moving = self.last_change.elapsed().as_secs_f32() < PREVIEW_HOLD_S;
        if moving {
            let (pw, ph) = ((w / 2).max(8), (h / 2).max(8));
            if self.preview.as_ref().is_none_or(|t| t.width != pw || t.height != ph) {
                self.preview = Some(self.gpu.target(pw, ph));
            }
            let t = self.preview.as_mut().unwrap();
            self.gpu.step(t, &self.scene, 1, self.seed, Some(2));
            self.showing_preview = true;
            return;
        }
        if self.full.as_ref().is_none_or(|t| t.width != w || t.height != h) {
            self.full = Some(self.gpu.target(w, h));
        }
        let t = self.full.as_mut().unwrap();
        let remaining = self.target_spp.saturating_sub(t.samples);
        let spp = if self.paused { 0 } else { self.spp_per_frame.min(remaining) };
        if spp == 0 && t.samples > 0 && !self.showing_preview {
            // Converged or paused: re-tonemap only (exposure etc. may have changed).
            self.gpu.step(t, &self.scene, 0, self.seed, None);
            return;
        }
        self.gpu.step(t, &self.scene, spp, self.seed, None);
        if spp > 0 && t.samples > spp {
            let per = t.last_ms / spp as f32;
            self.spp_per_frame = ((FRAME_BUDGET_MS / per.max(0.01)) as u32).clamp(1, 64);
        } else if t.samples == spp {
            self.spp_per_frame = 1;
        }
        self.showing_preview = false;
    }

    /// Unreal-style flight: hold RMB in the viewport, mouse looks, WASD moves, Q/E down/up,
    /// Shift boosts, the wheel scales the speed. Integrated by cam-controls `FpsFly` (thrust,
    /// inertia, damping); on release the orbit pivot is placed in front of the camera at the
    /// current orbit distance, so orbiting continues from where you flew.
    fn fly_camera(&mut self, ui: &egui::Ui, resp: &egui::Response) {
        use cam_controls::{CameraIntent, CameraPose, FpsFly};
        use glam::{Quat, Vec3};
        let held = resp.is_pointer_button_down_on() && ui.input(|i| i.pointer.secondary_down());
        let radius = self.scene.formula.framing_radius();
        let cam = &mut self.scene.camera;
        let (yaw, pitch) = (cam.yaw_degrees.to_radians(), cam.pitch_degrees.to_radians());
        let dist = cam.distance * radius;
        if held && self.fly.is_none() {
            // Orbit -> free pose. Our forward is (-cosP sinY, -sinP, -cosP cosY); FpsFly's is
            // (-cos p sin y, sin p, -cos p cos y): same yaw, pitch negated.
            let eye = Vec3::new(
                cam.target[0] + dist * pitch.cos() * yaw.sin(),
                cam.target[1] + dist * pitch.sin(),
                cam.target[2] + dist * pitch.cos() * yaw.cos(),
            );
            let orientation = Quat::from_axis_angle(Vec3::Y, yaw) * Quat::from_axis_angle(Vec3::X, -pitch);
            let pose = CameraPose { eye, orientation, ..CameraPose::default() };
            self.fly = Some(FpsFly::from_pose(pose));
        }
        let Some(fly) = &mut self.fly else { return };
        let viewport = cam_viewport::ViewportSize::new(resp.rect.width().max(1.0) as u32, resp.rect.height().max(1.0) as u32);
        let dt = ui.input(|i| i.stable_dt).clamp(1.0e-4, 0.1);
        if held {
            ui.ctx().set_cursor_icon(egui::CursorIcon::None);
            let (delta, scroll, keys) = ui.input(|i| {
                let k = |key| if i.key_down(key) { 1.0f32 } else { 0.0 };
                (
                    i.pointer.delta(),
                    i.smooth_scroll_delta.y,
                    [
                        k(egui::Key::W) - k(egui::Key::S),
                        k(egui::Key::D) - k(egui::Key::A),
                        k(egui::Key::E) - k(egui::Key::Q),
                        if i.modifiers.shift { 1.0 } else { 0.0 },
                    ],
                )
            });
            if scroll != 0.0 {
                self.fly_speed = (self.fly_speed * (scroll * 0.003).exp()).clamp(0.02, 50.0);
                self.status = format!("Flight speed ×{:.2}", self.fly_speed);
            }
            let sens = 0.0035;
            fly.apply_intent(CameraIntent::Look { dyaw: -delta.x * sens, dpitch: -delta.y * sens }, viewport);
            fly.apply_intent(CameraIntent::Thrust { forward: keys[0], right: keys[1], up: keys[2] }, viewport);
            fly.apply_intent(CameraIntent::Boost(keys[3] > 0.0), viewport);
        } else {
            fly.apply_intent(CameraIntent::Thrust { forward: 0.0, right: 0.0, up: 0.0 }, viewport);
            fly.apply_intent(CameraIntent::Boost(false), viewport);
        }
        // Acceleration scales with the formula's size and the speed multiplier.
        fly.inertia.thrust_sensitivity = 6.0 * radius * self.fly_speed;
        let moving = fly.update_dynamics(dt);

        let pose = fly.pose();
        let f = pose.forward();
        cam.pitch_degrees = (-f.y).clamp(-1.0, 1.0).asin().to_degrees();
        cam.yaw_degrees = (-f.x).atan2(-f.z).to_degrees();
        let target = pose.eye + f * dist;
        cam.target = [target.x, target.y, target.z];
        if !held && !moving {
            self.fly = None;
        }
        ui.ctx().request_repaint();
    }

    fn current(&self) -> Option<&Target> {
        if self.showing_preview { self.preview.as_ref() } else { self.full.as_ref() }
    }

    // -------------------------------------------------------------------------
    // panels
    // -------------------------------------------------------------------------

    fn top_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new("frac-rs").strong().size(16.0));
            ui.separator();
            ui.label(RichText::new(&self.scene.name).italics());
            ui.separator();
            if ui.button("★ Bookmark").on_hover_text("Save the scene (JSON) to bookmarks").clicked() {
                self.save_bookmark();
            }
            if ui.button("Settings…").clicked() { self.settings_open = true; }
            if ui.button("📷 Screenshot").on_hover_text("SDR sRGB PNG or HDR10 PQ PNG with cICP/mDCV/cLLI metadata").clicked() {
                self.screenshot();
            }
            if ui.button("Display EXR").on_hover_text("Save linear Rec.709 display light; 1 = 100 nits").clicked() { self.screenshot_exr(); }
            if ui.button("⟲ Reset").on_hover_text("Back to the loaded preset / bookmark").clicked() {
                self.scene = self.origin.clone();
                self.colour.sel = self.scene.colour.clone();
                self.colour.rebuild();
            }
            if ui.button("⧉ Copy JSON").on_hover_text("Copy the scene as JSON").clicked() {
                if let Ok(json) = serde_json::to_string_pretty(&self.scene) {
                    ui.ctx().copy_text(json);
                    self.status = "Scene JSON copied to the clipboard".into();
                }
            }
            ui.separator();
            ui.toggle_value(&mut self.paused, "⏸ Pause");
            ui.separator();
            egui::ComboBox::from_id_salt("final_size")
                .width(80.0)
                .selected_text(FINAL_SIZES[self.final_size].2)
                .show_ui(ui, |ui| {
                    for (i, s) in FINAL_SIZES.iter().enumerate() {
                        ui.selectable_value(&mut self.final_size, i, s.2);
                    }
                });
            ui.add(egui::DragValue::new(&mut self.final_spp).range(1..=65536).suffix(" spp"));
            match &self.job {
                Some(job) => {
                    let f = job.target.samples as f32 / job.spp as f32;
                    ui.add(egui::ProgressBar::new(f).desired_width(140.0).show_percentage());
                    if ui.button("✖").on_hover_text("Cancel the render").clicked() {
                        self.job = None;
                        self.status = "Render cancelled".into();
                    }
                }
                None => {
                    if ui.button("🎞 Render PNG").on_hover_text("Final render to ~/Pictures/frac-rs").clicked() {
                        self.start_job();
                    }
                }
            }
        });
    }

    fn browser(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.tab, Tab::Gallery, format!("Gallery ({})", self.gallery.len()));
            ui.selectable_value(&mut self.tab, Tab::Bookmarks, format!("Bookmarks ({})", self.bookmarks.len()));
            ui.selectable_value(&mut self.tab, Tab::Materials, "Materials");
        });
        ui.separator();
        if self.tab == Tab::Materials {
            self.materials_tab(ui);
            return;
        }
        let mut load: Option<Scene> = None;
        let mut delete: Option<usize> = None;
        let size = Vec2::new(ui.available_width().min(THUMB_W as f32), 0.0);
        let size = Vec2::new(size.x, size.x * THUMB_H as f32 / THUMB_W as f32);
        egui::ScrollArea::vertical().show(ui, |ui| {
            let entries = if self.tab == Tab::Gallery { &self.gallery } else { &self.bookmarks };
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
                                    ui.painter().rect_filled(r, 4.0, ui.visuals().extreme_bg_color);
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
                let _ = std::fs::remove_file(p);
            }
        }
    }

    fn inspector(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            let s = &mut self.scene;
            ui.horizontal(|ui| {
                ui.label("Name");
                ui.text_edit_singleline(&mut s.name);
            });
            egui::CollapsingHeader::new("Formula").default_open(true).show(ui, |ui| formula_ui(ui, s));
            egui::CollapsingHeader::new("Camera").default_open(true).show(ui, |ui| camera_ui(ui, s));
            egui::CollapsingHeader::new("Light").show(ui, |ui| light_ui(ui, &mut s.lighting));
            egui::CollapsingHeader::new("Material").default_open(true).show(ui, |ui| material_ui(ui, &mut s.material));
            egui::CollapsingHeader::new("Colour").default_open(true).show(ui, |ui| colour_ui(ui, s));
            egui::CollapsingHeader::new("Render").default_open(true).show(ui, |ui| {
                render_ui(ui, &mut s.render);
                ui.add(egui::Slider::new(&mut self.target_spp, 1..=65536).logarithmic(true).text("target spp"));
                ui.add(egui::Slider::new(&mut self.resolution, 0.25..=2.0).text("viewport scale"));
                if ui.button("New noise seed").clicked() {
                    self.seed = self.seed.wrapping_add(7919);
                    self.full = None;
                }
            });
        });
    }

    fn status_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(RichText::new(&self.gpu.name).weak());
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
                ui.label(format!("{} spp/frame", self.spp_per_frame));
            }
            ui.separator();
            ui.label(format!("UI {:.0} fps", 1000.0 / self.frame_ms.max(0.1)));
            if !self.status.is_empty() {
                ui.separator();
                ui.label(RichText::new(&self.status).weak());
            }
        });
    }

    fn viewport(&mut self, ui: &mut egui::Ui) {
        let avail = ui.available_size();
        let (rect, resp) = ui.allocate_exact_size(avail, Sense::click_and_drag());
        self.fly_camera(ui, &resp);
        let cam = &mut self.scene.camera;
        if resp.dragged_by(egui::PointerButton::Primary) {
            let d = resp.drag_delta();
            cam.yaw_degrees = (cam.yaw_degrees - d.x * 0.3) % 360.0;
            cam.pitch_degrees = (cam.pitch_degrees + d.y * 0.3).clamp(-89.0, 89.0);
        }
        if resp.dragged_by(egui::PointerButton::Middle) {
            // Pan the orbit target in the camera plane.
            let d = resp.drag_delta();
            let radius = self.scene.formula.framing_radius();
            let (yaw, pitch) = (cam.yaw_degrees.to_radians(), cam.pitch_degrees.to_radians());
            let fwd = [-pitch.cos() * yaw.sin(), -pitch.sin(), -pitch.cos() * yaw.cos()];
            let right = normalize(cross(fwd, [0.0, 1.0, 0.0]));
            let up = cross(right, fwd);
            let k = cam.distance * radius * 2.0 * (cam.fov_y_degrees.to_radians() * 0.5).tan() / avail.y.max(1.0);
            for i in 0..3 {
                cam.target[i] += (-d.x * right[i] + d.y * up[i]) * k;
            }
        }
        if resp.hovered() && self.fly.is_none() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                cam.distance = (cam.distance * (-scroll * 0.0015).exp()).clamp(0.05, 20.0);
            }
        }
        if resp.double_clicked() {
            cam.target = [0.0; 3];
        }

        let ppp = ui.ctx().pixels_per_point() * self.resolution;
        let (w, h) = (((avail.x * ppp) as usize).max(16), ((avail.y * ppp) as usize).max(16));
        if !self.step_job() {
            self.step_viewport(w, h);
        }

        if let Some(t) = self.current() {
            let state = ui.ctx().data(|d| d.get_temp::<egui_display::DisplayState>(egui_display::state_id()));
            let output_hdr = state.as_ref().is_some_and(|s| s.output.is_hdr());
            let white = state.as_ref().map_or(100.0, |s| s.target.white);
            let gain = if t.hdr && output_hdr { 100.0 / white } else { 1.0 };
            if !output_hdr {
                let bytes = std::sync::Arc::new(t.pixels.iter().flat_map(|p| p.to_le_bytes()).collect::<Vec<_>>());
                let mut view = self.hdr_view.lock().unwrap();
                view.set_output_format(egui_display::CANVAS_FORMAT);
                view.stage_frame(egui_hdr_view::HdrFormat::Rgba8, bytes, t.width, t.height, Default::default());
            } else {
            let canvas: Vec<[f32; 4]> = t.light.iter().map(|p| {
                let f = |v: f32| crate::color::oetf(if output_hdr { v * gain } else { v.clamp(0.0, 1.0) });
                [f(p[0]), f(p[1]), f(p[2]), 1.0]
            }).collect();
            let bytes = std::sync::Arc::new(bytemuck::cast_slice::<[f32; 4], u8>(&canvas).to_vec());
            let mut view = self.hdr_view.lock().unwrap();
            view.set_output_format(egui_display::CANVAS_FORMAT);
            view.stage_frame(egui_hdr_view::HdrFormat::Rgba32F, bytes, t.width, t.height, Default::default());
            drop(view);
            }
            ui.painter().add(egui_wgpu::Callback::new_paint_callback(rect, egui_hdr_view::HdrPaintCallback { inner: self.hdr_view.clone() }));
        }
        if let Some(job) = &self.job {
            let text = format!(
                "Rendering {}×{} · {}/{} spp",
                job.target.width, job.target.height, job.target.samples, job.spp
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
    pub(crate) fn ui(&mut self, root: &mut egui::Ui) {
        let ctx = root.ctx().clone();
        let dt = ctx.input(|i| i.stable_dt).max(1.0e-4);
        self.frame_ms = self.frame_ms * 0.9 + dt * 1000.0 * 0.1;
        if !ctx.text_edit_focused() {
            if ctx.input(|i| i.key_pressed(egui::Key::Tab)) {
                self.show_ui = !self.show_ui;
            }
            if ctx.input(|i| i.key_pressed(egui::Key::Space)) {
                self.paused = !self.paused;
            }
        }

        let thumbs_pending = self.render_one_thumbnail(&ctx);

        if self.show_ui {
            egui::Panel::top("top").show(root, |ui| self.top_bar(ui));
            egui::Panel::bottom("status").show(root, |ui| self.status_bar(ui));
            egui::Panel::left("browser").default_size(250.0).min_size(180.0).show(root, |ui| self.browser(ui));
            egui::Panel::right("inspector").default_size(330.0).min_size(260.0).show(root, |ui| self.inspector(ui));
        }
        egui::CentralPanel::default().frame(egui::Frame::NONE).show(root, |ui| self.viewport(ui));

        self.settings_ui(&ctx);
        if let Some(error) = self.current().and_then(|t| t.colour_error.as_deref()) {
            self.status = format!("Colour output failed: {error}");
        }
        self.handle_snap(&ctx, thumbs_pending);
        let converged = self.full.as_ref().is_some_and(|t| t.samples >= self.target_spp) && !self.showing_preview;
        if !converged || thumbs_pending || self.job.is_some() {
            ctx.request_repaint();
        }
    }
}

// =============================================================================
// inspector sections
// =============================================================================

fn slider(ui: &mut egui::Ui, v: &mut f32, range: std::ops::RangeInclusive<f32>, text: &str) {
    ui.add(egui::Slider::new(v, range).text(text));
}

fn vec3(ui: &mut egui::Ui, v: &mut [f32; 3], range: std::ops::RangeInclusive<f32>, text: &str) {
    ui.horizontal(|ui| {
        for c in v.iter_mut() {
            ui.add(egui::DragValue::new(c).speed(0.005).range(range.clone()));
        }
        ui.label(text);
    });
}

fn formula_ui(ui: &mut egui::Ui, s: &mut Scene) {
    let mut code = s.formula.code();
    egui::ComboBox::from_label("family").selected_text(s.formula.name()).show_ui(ui, |ui| {
        for (i, n) in Formula::NAMES.iter().enumerate() {
            ui.selectable_value(&mut code, i as u32, *n);
        }
    });
    if code != s.formula.code() {
        let fresh = Scene::preset(code);
        s.formula = fresh.formula;
        s.render.iterations = fresh.render.iterations;
        s.render.max_steps = fresh.render.max_steps;
        s.render.hit_epsilon = fresh.render.hit_epsilon;
        s.julia = None;
    }
    ui.add(egui::Slider::new(&mut s.render.iterations, 1..=64).text("iterations"));
    match &mut s.formula {
        Formula::Mandelbulb(b) => bulb_ui(ui, b),
        Formula::Mandelbox(b) => box_ui(ui, b),
        Formula::QuaternionJulia(q) => {
            for (i, n) in ["c.x", "c.y", "c.z", "c.w"].iter().enumerate() {
                slider(ui, &mut q.constant[i], -1.5..=1.5, n);
            }
            slider(ui, &mut q.slice_w, -1.5..=1.5, "slice w");
            vec3(ui, &mut q.rotation_degrees, -360.0..=360.0, "4D rotation °");
            slider(ui, &mut q.bailout, 2.0..=16.0, "bailout");
        }
        Formula::Kifs(k) => kifs_ui(ui, k),
        Formula::Kleinian(k) => {
            slider(ui, &mut k.a, 1.0..=2.2, "a");
            slider(ui, &mut k.b, -1.0..=1.0, "b");
            slider(ui, &mut k.bound_radius, 0.0..=4.0, "ball bound (0 = none)");
        }
        Formula::PseudoKleinian(k) => {
            vec3(ui, &mut k.box_size, 0.1..=2.0, "box");
            slider(ui, &mut k.size, 0.1..=2.0, "size");
            vec3(ui, &mut k.c, -2.0..=2.0, "c");
            vec3(ui, &mut k.offset, -2.0..=2.0, "offset");
            slider(ui, &mut k.thickness, 0.0..=0.2, "thickness");
            slider(ui, &mut k.bound_radius, 0.0..=4.0, "ball bound (0 = none)");
        }
        Formula::Apollonian(a) => {
            slider(ui, &mut a.scale, 1.0..=2.0, "scale");
            slider(ui, &mut a.bound_radius, 0.0..=4.0, "ball bound (0 = none)");
        }
        Formula::Hybrid(h) => {
            ui.label("steps (repeated in order)");
            for (i, st) in h.steps.iter_mut().enumerate() {
                egui::ComboBox::from_id_salt(("hstep", i)).selected_text(format!("{st:?}")).show_ui(ui, |ui| {
                    for v in [HybridStep::Off, HybridStep::Mandelbulb, HybridStep::Mandelbox, HybridStep::KifsFold, HybridStep::Inversion] {
                        ui.selectable_value(st, v, format!("{v:?}"));
                    }
                });
            }
            slider(ui, &mut h.bailout, 2.0..=16.0, "bailout");
            ui.collapsing("bulb step", |ui| bulb_ui(ui, &mut h.bulb));
            ui.collapsing("box step", |ui| box_ui(ui, &mut h.mandelbox));
            ui.collapsing("KIFS step", |ui| kifs_ui(ui, &mut h.kifs));
            slider(ui, &mut h.apollonian_scale, 1.0..=2.0, "inversion scale");
        }
    }
    if s.formula.supports_julia() {
        let mut on = s.julia.is_some();
        ui.checkbox(&mut on, "Julia mode");
        if on && s.julia.is_none() {
            s.julia = Some([0.35, 0.35, -0.4]);
        } else if !on {
            s.julia = None;
        }
        if let Some(c) = &mut s.julia {
            vec3(ui, c, -2.0..=2.0, "Julia c");
        }
    }
    ui.collapsing("object transform", |ui| {
        vec3(ui, &mut s.object.offset, -4.0..=4.0, "offset");
        vec3(ui, &mut s.object.rotation_degrees, -360.0..=360.0, "rotation °");
        slider(ui, &mut s.object.scale, 0.1..=4.0, "scale");
    });
}

fn bulb_ui(ui: &mut egui::Ui, b: &mut Bulb) {
    slider(ui, &mut b.power, 2.0..=16.0, "power");
    slider(ui, &mut b.bailout, 2.0..=16.0, "bailout");
    slider(ui, &mut b.angle_scale[0], -4.0..=4.0, "θ scale");
    slider(ui, &mut b.angle_scale[1], -4.0..=4.0, "φ scale");
    slider(ui, &mut b.angle_phase_degrees[0], -360.0..=360.0, "θ phase °");
    slider(ui, &mut b.angle_phase_degrees[1], -360.0..=360.0, "φ phase °");
    vec3(ui, &mut b.rotation_degrees, -360.0..=360.0, "iter rotation °");
}

fn box_ui(ui: &mut egui::Ui, b: &mut MandelBox) {
    slider(ui, &mut b.scale, -4.0..=4.0, "scale");
    slider(ui, &mut b.min_radius_ratio, 0.05..=1.0, "min radius ratio");
    slider(ui, &mut b.fixed_radius, 0.25..=2.0, "fixed radius");
    slider(ui, &mut b.fold_limit, 0.25..=2.0, "fold limit");
    vec3(ui, &mut b.rotation_degrees, -360.0..=360.0, "iter rotation °");
}

fn kifs_ui(ui: &mut egui::Ui, k: &mut Kifs) {
    egui::ComboBox::from_label("kind").selected_text(format!("{:?}", k.kind)).show_ui(ui, |ui| {
        for v in [KifsKind::Tetrahedron, KifsKind::Octahedron, KifsKind::Menger] {
            if ui.selectable_value(&mut k.kind, v, format!("{v:?}")).changed() {
                k.scale = v.preset_scale();
            }
        }
    });
    slider(ui, &mut k.scale, 1.2..=4.0, "scale");
    vec3(ui, &mut k.offset, -2.0..=2.0, "centre offset");
    vec3(ui, &mut k.rotation_degrees, -360.0..=360.0, "iter rotation °");
}

fn camera_ui(ui: &mut egui::Ui, s: &mut Scene) {
    let c = &mut s.camera;
    slider(ui, &mut c.yaw_degrees, -180.0..=180.0, "yaw °");
    slider(ui, &mut c.pitch_degrees, -89.0..=89.0, "pitch °");
    ui.add(egui::Slider::new(&mut c.distance, 0.05..=12.0).logarithmic(true).text("distance (framing radii)"));
    slider(ui, &mut c.fov_y_degrees, 5.0..=120.0, "FOV °");
    vec3(ui, &mut c.target, -10.0..=10.0, "target");
    slider(ui, &mut c.aperture, 0.0..=1.0, "aperture (DOF)");
    if c.aperture > 0.0 {
        ui.add(egui::Slider::new(&mut c.focus_distance, 0.0..=12.0).text("focus (0 = target)"));
    }
    ui.label(RichText::new("LMB orbit · MMB pan · wheel zoom · RMB fly (WASD, Q/E, Shift, wheel = speed) · double-click recentre · Tab hides UI").small().weak());
}

fn light_ui(ui: &mut egui::Ui, l: &mut Lighting) {
    slider(ui, &mut l.sun_azimuth, -180.0..=180.0, "sun azimuth °");
    slider(ui, &mut l.sun_elevation, -30.0..=90.0, "sun elevation °");
    ui.horizontal(|ui| {
        ui.color_edit_button_rgb(&mut l.sun_color);
        ui.add(egui::Slider::new(&mut l.sun_intensity, 0.0..=16.0).text("sun"));
    });
    ui.add(egui::Slider::new(&mut l.sun_angle, 0.05..=30.0).logarithmic(true).text("sun angle °"));
    slider(ui, &mut l.sky_intensity, 0.0..=16.0, "sky intensity");
    ui.horizontal(|ui| {
        ui.color_edit_button_rgb(&mut l.sky_horizon);
        ui.label("horizon");
        ui.color_edit_button_rgb(&mut l.sky_zenith);
        ui.label("zenith");
    });
    ui.checkbox(&mut l.background, "sky visible behind the fractal");
}

fn material_ui(ui: &mut egui::Ui, m: &mut Material) {
    ui.horizontal(|ui| {
        ui.selectable_value(&mut m.model, MaterialModel::Fast, "Fast");
        ui.selectable_value(&mut m.model, MaterialModel::StandardSurface, "Standard Surface");
    });
    let presets = crate::materials::PRESETS;
    egui::ComboBox::from_label("library")
        .selected_text(m.preset.clone().unwrap_or_else(|| "—".into()))
        .height(400.0)
        .show_ui(ui, |ui| {
            for p in presets {
                if ui.selectable_label(m.preset.as_deref() == Some(p.name()), format!("{} · {}", p.category, p.name())).clicked() {
                    p.apply(m);
                }
            }
        });
    ui.horizontal(|ui| {
        ui.label("colour from");
        ui.selectable_value(&mut m.color_source, ColorSource::Palette, "palette");
        ui.selectable_value(&mut m.color_source, ColorSource::Material, "material");
        if m.color_source == ColorSource::Material {
            ui.color_edit_button_rgb(&mut m.base_color);
        }
    });
    ui.horizontal(|ui| {
        ui.color_edit_button_rgb(&mut m.base_tint);
        ui.add(egui::Slider::new(&mut m.base, 0.0..=1.0).text("base (× palette)"));
    });
    slider(ui, &mut m.metalness, 0.0..=1.0, "metalness");
    ui.horizontal(|ui| {
        ui.color_edit_button_rgb(&mut m.specular_color);
        ui.add(egui::Slider::new(&mut m.specular, 0.0..=1.0).text("specular"));
    });
    slider(ui, &mut m.specular_roughness, 0.0..=1.0, "roughness");
    slider(ui, &mut m.specular_ior, 1.0..=3.0, "IOR");
    ui.horizontal(|ui| {
        ui.color_edit_button_rgb(&mut m.emission_color);
        ui.add(egui::Slider::new(&mut m.emission, 0.0..=4.0).text("emission"));
    });
    let mut facing_on = m.facing.is_some();
    if ui.checkbox(&mut facing_on, "facing mix (pearlescent)").changed() {
        m.facing = facing_on.then_some(Facing { color: [0.18, 0.10, 0.65], roughness: 0.22, metallic: 0.0, exponent: 3.0 });
    }
    if let Some(f) = &mut m.facing {
        ui.horizontal(|ui| {
            ui.color_edit_button_rgb(&mut f.color);
            ui.add(egui::Slider::new(&mut f.exponent, 0.5..=8.0).text("grazing colour · exponent"));
        });
        slider(ui, &mut f.roughness, 0.0..=1.0, "grazing roughness");
        slider(ui, &mut f.metallic, 0.0..=1.0, "grazing metallic");
    }
    if m.model == MaterialModel::StandardSurface {
        slider(ui, &mut m.diffuse_roughness, 0.0..=1.0, "diffuse roughness");
        slider(ui, &mut m.specular_anisotropy, 0.0..=1.0, "anisotropy");
        slider(ui, &mut m.specular_rotation, 0.0..=1.0, "aniso rotation");
        ui.horizontal(|ui| {
            ui.color_edit_button_rgb(&mut m.coat_color);
            ui.add(egui::Slider::new(&mut m.coat, 0.0..=1.0).text("coat"));
        });
        slider(ui, &mut m.coat_roughness, 0.0..=1.0, "coat roughness");
        slider(ui, &mut m.coat_ior, 1.0..=3.0, "coat IOR");
        slider(ui, &mut m.coat_affect_color, 0.0..=1.0, "coat affect colour");
        slider(ui, &mut m.coat_affect_roughness, 0.0..=1.0, "coat affect roughness");
        ui.horizontal(|ui| {
            ui.color_edit_button_rgb(&mut m.sheen_color);
            ui.add(egui::Slider::new(&mut m.sheen, 0.0..=1.0).text("sheen"));
        });
        slider(ui, &mut m.sheen_roughness, 0.0..=1.0, "sheen roughness");
        slider(ui, &mut m.thin_film_thickness, 0.0..=2000.0, "thin film nm");
        slider(ui, &mut m.thin_film_ior, 1.0..=3.0, "thin film IOR");
    } else {
        ui.label(RichText::new("Fast: Lambert + GGX. Standard Surface adds coat, sheen, thin film, anisotropy.").small().weak());
    }
}

fn colour_ui(ui: &mut egui::Ui, s: &mut Scene) {
    egui::ComboBox::from_label("palette").selected_text(s.palette.label()).show_ui(ui, |ui| {
        for p in PaletteScheme::ALL {
            ui.selectable_value(&mut s.palette, p, p.label());
        }
    });
    palette_strip(ui, s.palette);
    egui::ComboBox::from_label("colouring").selected_text(format!("{:?}", s.coloring)).show_ui(ui, |ui| {
        for c in [Coloring::Radius, Coloring::TrapOrigin, Coloring::TrapPlane, Coloring::TrapPoint] {
            ui.selectable_value(&mut s.coloring, c, format!("{c:?}"));
        }
    });
    if s.coloring != Coloring::Radius {
        ui.add(egui::Slider::new(&mut s.trap_scale, 0.05..=20.0).logarithmic(true).text("trap scale"));
        if s.coloring == Coloring::TrapPlane {
            ui.horizontal(|ui| {
                ui.label("plane normal");
                for (i, a) in ["X", "Y", "Z"].iter().enumerate() {
                    ui.selectable_value(&mut s.trap_axis, i as u32, *a);
                }
            });
        }
        if s.coloring != Coloring::TrapOrigin {
            vec3(ui, &mut s.trap_point, -4.0..=4.0, "trap point");
        }
    }
}

fn palette_strip(ui: &mut egui::Ui, scheme: PaletteScheme) {
    let lut = build_lut(scheme);
    let (rect, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 12.0), Sense::hover());
    let n = 64;
    for i in 0..n {
        let c = lut[i * (lut.len() - 2) / (n - 1)];
        let to8 = |v: f32| (v.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8;
        let x0 = rect.left() + rect.width() * i as f32 / n as f32;
        let x1 = rect.left() + rect.width() * (i + 1) as f32 / n as f32;
        ui.painter().rect_filled(
            egui::Rect::from_min_max(egui::pos2(x0, rect.top()), egui::pos2(x1, rect.bottom())),
            0.0,
            Color32::from_rgb(to8(c[0]), to8(c[1]), to8(c[2])),
        );
    }
}

fn render_ui(ui: &mut egui::Ui, r: &mut Render) {
    ui.add(egui::Slider::new(&mut r.max_bounces, 0..=16).text("bounces"));
    ui.add(egui::Slider::new(&mut r.max_steps, 32..=2048).logarithmic(true).text("march steps"));
    ui.add(egui::Slider::new(&mut r.hit_epsilon, 0.0001..=0.01).logarithmic(true).text("hit epsilon"));
    ui.add(egui::Slider::new(&mut r.step_factor, 0.3..=1.0).text("step factor (0.5 = reference)"));
    slider(ui, &mut r.exposure_stops, -6.0..=6.0, "exposure EV");
    slider(ui, &mut r.saturation, 0.0..=2.0, "saturation");
    ui.horizontal(|ui| {
        ui.selectable_value(&mut r.reinhard, false, "ACES 2.0");
        ui.selectable_value(&mut r.reinhard, true, "Reinhard");
    });
}

#[allow(dead_code)]
fn exists(p: &Path) -> bool {
    p.exists()
}
