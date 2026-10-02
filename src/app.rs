//! The browser UI (egui): gallery + bookmarks with GPU thumbnails, the progressive viewport,
//! the scene inspector, screenshots and final PNG renders.

use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use eframe::egui::{self, Color32, ColorImage, RichText, Sense, TextureHandle, TextureOptions, Vec2};

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

pub fn run() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("frac-rs — path-traced fractals on CUDA (Rust)")
            .with_inner_size([1600.0, 940.0])
            .with_min_inner_size([900.0, 560.0]),
        ..Default::default()
    };
    eframe::run_native("frac-rs", options, Box::new(|cc| Ok(Box::new(App::new(cc)))))
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
}

struct Job {
    scene: Scene,
    target: Target,
    spp: u32,
    path: PathBuf,
    started: Instant,
}

struct App {
    gpu: Gpu,
    scene: Scene,
    /// The preset / bookmark the scene came from, for "Reset".
    origin: Scene,
    full: Option<Target>,
    preview: Option<Target>,
    showing_preview: bool,
    tex: Option<TextureHandle>,
    tex_stamp: (u32, bool, usize),
    gallery: Vec<Entry>,
    bookmarks: Vec<Entry>,
    thumb_target: Target,
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

impl App {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let gpu = Gpu::new().unwrap_or_else(|e| panic!("CUDA init failed: {e}"));
        let gallery: Vec<Entry> =
            Scene::gallery().into_iter().map(|scene| Entry { scene, thumb: None, path: None }).collect();
        let scene = gallery[0].scene.clone();
        let thumb_target = gpu.target(THUMB_W, THUMB_H);
        Self {
            gpu,
            origin: scene.clone(),
            last_scene: scene.clone(),
            scene,
            full: None,
            preview: None,
            showing_preview: false,
            tex: None,
            tex_stamp: (u32::MAX, false, 0),
            gallery,
            bookmarks: load_bookmarks(),
            thumb_target,
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
            snap: std::env::var("FRAC_SNAP").ok().map(|p| {
                let spp = std::env::var("FRAC_SNAP_SPP").ok().and_then(|v| v.parse().ok()).unwrap_or(256);
                (PathBuf::from(p), spp, false)
            }),
        }
        .with_snap_preset()
    }

    fn with_snap_preset(mut self) -> Self {
        if let Some(i) = std::env::var("FRAC_SNAP_PRESET").ok().and_then(|v| v.parse::<usize>().ok()) {
            if let Some(e) = self.gallery.get(i) {
                let scene = e.scene.clone();
                self.load(scene);
            }
        }
        self
    }

    fn handle_snap(&mut self, ctx: &egui::Context, thumbs_pending: bool) {
        let Some((path, spp, requested)) = self.snap.clone() else { return };
        for ev in ctx.input(|i| i.events.clone()) {
            if let egui::Event::Screenshot { image, .. } = ev {
                let rgb: Vec<u8> = image.pixels.iter().flat_map(|c| [c.r(), c.g(), c.b()]).collect();
                let file = std::fs::File::create(&path).expect("snap file");
                let mut enc = png::Encoder::new(std::io::BufWriter::new(file), image.size[0] as u32, image.size[1] as u32);
                enc.set_color(png::ColorType::Rgb);
                enc.set_depth(png::BitDepth::Eight);
                enc.write_header().and_then(|mut w| w.write_image_data(&rgb)).expect("snap png");
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                return;
            }
        }
        let ready = !thumbs_pending && !self.showing_preview && self.full.as_ref().is_some_and(|t| t.samples >= spp);
        if ready && !requested {
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(Default::default()));
            self.snap = Some((path, spp, true));
        }
        ctx.request_repaint();
    }

    fn load(&mut self, scene: Scene) {
        self.origin = scene.clone();
        self.scene = scene;
    }

    /// Render one pending thumbnail (gallery first, then bookmarks). Returns true if one ran.
    fn render_one_thumbnail(&mut self, ctx: &egui::Context) -> bool {
        let pending = self
            .gallery
            .iter_mut()
            .chain(self.bookmarks.iter_mut())
            .find(|e| e.thumb.is_none());
        let Some(entry) = pending else { return false };
        let mut s = entry.scene.clone();
        s.render.max_bounces = s.render.max_bounces.min(3);
        let t = &mut self.thumb_target;
        self.gpu.step(t, &s, THUMB_SPP / 2, 7, None);
        self.gpu.step(t, &s, THUMB_SPP / 2, 7, None);
        entry.thumb = Some(ctx.load_texture(format!("thumb-{}", s.name), to_image(t), TextureOptions::LINEAR));
        true
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
        if self.scene != self.last_scene {
            self.last_change = Instant::now();
            self.last_scene = self.scene.clone();
        }
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
            if ui.button("📷 Screenshot").on_hover_text("Save the viewport as PNG").clicked() {
                self.screenshot();
            }
            if ui.button("⟲ Reset").on_hover_text("Back to the loaded preset / bookmark").clicked() {
                self.scene = self.origin.clone();
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
        });
        ui.separator();
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
        let cam = &mut self.scene.camera;
        if resp.dragged_by(egui::PointerButton::Primary) {
            let d = resp.drag_delta();
            cam.yaw_degrees = (cam.yaw_degrees - d.x * 0.3) % 360.0;
            cam.pitch_degrees = (cam.pitch_degrees + d.y * 0.3).clamp(-89.0, 89.0);
        }
        if resp.dragged_by(egui::PointerButton::Secondary) || resp.dragged_by(egui::PointerButton::Middle) {
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
        if resp.hovered() {
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
            let stamp = (t.samples, self.showing_preview, t.width * t.height);
            let image = (stamp != self.tex_stamp || self.paused).then(|| to_image(t));
            if let Some(image) = image {
                match &mut self.tex {
                    Some(tex) => tex.set(image, TextureOptions::LINEAR),
                    None => self.tex = Some(ui.ctx().load_texture("viewport", image, TextureOptions::LINEAR)),
                }
                self.tex_stamp = stamp;
            }
        }
        if let Some(tex) = &self.tex {
            ui.painter().image(
                tex.id(),
                rect,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                Color32::WHITE,
            );
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

impl eframe::App for App {
    fn ui(&mut self, root: &mut egui::Ui, _frame: &mut eframe::Frame) {
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
    ui.label(RichText::new("LMB orbit · RMB/MMB pan · wheel zoom · double-click recentre · Tab hides UI").small().weak());
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
        ui.selectable_value(&mut r.reinhard, false, "ACES");
        ui.selectable_value(&mut r.reinhard, true, "Reinhard");
    });
}

#[allow(dead_code)]
fn exists(p: &Path) -> bool {
    p.exists()
}
