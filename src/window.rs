//! Native window shell. egui renders to egui-display's float canvas; presentation
//! selects BOTH the swapchain format and colour space (eframe only offered SDR).
//! Frames are drawn on demand: on input and when egui asks for one (`repaint_delay`, which the
//! app requests while something animates: `App::animating`), at most `App::gui_fps` per
//! second; an idle window sleeps.
use anyhow::{Context, Result};
use soluwell_viewport::display::{Output, PresentPass};
use std::sync::Arc;
use std::time::{Duration, Instant};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

pub fn run() -> Result<()> {
    let event_loop = EventLoop::new()?;
    let mut host = Host {
        window: None,
        error: None,
    };
    event_loop.run_app(&mut host)?;
    if let Some(e) = host.error {
        return Err(e);
    }
    Ok(())
}
struct Host {
    window: Option<Native>,
    error: Option<anyhow::Error>,
}
struct Native {
    window: Arc<Window>,
    app: crate::app::App,
    ctx: egui::Context,
    input: egui_winit::State,
    renderer: egui_wgpu::Renderer,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    present: PresentPass,
    output: Output,
    error: Option<String>,
    /// When the next frame is due (input, or egui's repaint request); None: idle until input.
    next_redraw: Option<Instant>,
    /// When the last frame was drawn: frames are at least one GUI frame (`App::gui_fps`) apart.
    last_redraw: Instant,
    cursor_grab: egui::CursorGrab,
}
impl Native {
    fn new(events: &ActiveEventLoop) -> Result<Self> {
        let window = Arc::new(
            events.create_window(
                Window::default_attributes()
                    .with_title("WarpBro")
                    .with_theme(Some(winit::window::Theme::Dark))
                    .with_inner_size(winit::dpi::LogicalSize::new(1600.0, 940.0))
                    .with_min_inner_size(winit::dpi::LogicalSize::new(900.0, 560.0)),
            )?,
        );
        let gpu = gpu_info::shared_device().context("wgpu: shared GPU unavailable")?;
        log::info!("Display/OCIO adapter: {:?}", gpu.adapter.get_info());
        let adapter = gpu.adapter.clone();
        // Surface configuration waits for this device to become idle. OCIO's
        // continuously submitting offscreen queue must never own that wait.
        // Keep the negotiated instance/physical adapter, with a private UI queue.
        let (device, queue) =
            gpu_info::request_max_device_blocking(&adapter, gpu.device.features())
                .context("wgpu: private presentation device unavailable")?;
        let surface = gpu.instance.create_surface(window.clone())?;
        let caps = surface.get_capabilities(&adapter);
        let (format, color_space) = Output::Sdr8
            .surface(&caps)
            .context("surface has no SDR output")?;
        let size = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            color_space,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);
        let ctx = egui::Context::default();
        let mut fonts = egui::FontDefinitions::default();
        soluwell_config::add_icon_font(&mut fonts);
        ctx.set_fonts(fonts);
        // Solid scroll bars reserve their width: egui's default floating bars paint over the
        // right edge of every scroll area (e.g. the Attribute Editor's per-row expand buttons).
        ctx.all_styles_mut(|style| style.spacing.scroll = egui::style::ScrollStyle::solid());
        let input = egui_winit::State::new(
            ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            window.theme(),
            Some(device.limits().max_texture_dimension_2d as usize),
        );
        let renderer = egui_wgpu::Renderer::new(
            &device,
            soluwell_viewport::display::CANVAS_FORMAT,
            Default::default(),
        );
        let present = PresentPass::new(&device, format, Output::Sdr8);
        Ok(Self {
            window,
            app: crate::app::App::new(),
            ctx,
            input,
            renderer,
            adapter,
            device,
            queue,
            surface,
            config,
            present,
            output: Output::Sdr8,
            error: None,
            next_redraw: None,
            last_redraw: Instant::now(),
            cursor_grab: egui::CursorGrab::None,
        })
    }

    /// The shortest time between two frames: one GUI frame (`App::gui_fps`).
    fn frame_interval(&self) -> Duration {
        Duration::from_secs_f64(1.0 / f64::from(self.app.gui_fps()))
    }

    /// Ask for a frame `delay` from now (egui's `Duration::MAX`: none), no sooner than one GUI
    /// frame after the last one; the earliest request wins.
    fn schedule(&mut self, delay: Duration) {
        let Some(at) = Instant::now().checked_add(delay) else {
            return;
        };
        let at = at.max(self.last_redraw + self.frame_interval());
        self.next_redraw = Some(self.next_redraw.map_or(at, |due| due.min(at)));
    }

    fn redraw(&mut self, events: &ActiveEventLoop) -> Result<()> {
        let tick = Instant::now();
        // A redraw the system asked for sooner than the GUI rate waits for its turn.
        if tick < self.last_redraw + self.frame_interval() {
            self.schedule(Duration::ZERO);
            return Ok(());
        }
        self.next_redraw = None;
        self.last_redraw = tick;
        let size = self.window.inner_size();
        if size.width == 0 || size.height == 0 {
            return Ok(());
        }
        let caps = self.surface.get_capabilities(&self.adapter);
        let available: Vec<_> = Output::ALL
            .into_iter()
            .filter(|o| o.surface(&caps).is_some())
            .collect();
        if !available.contains(&self.output) {
            self.app.display.output = Output::Sdr8;
        }
        let requested = self.app.display.output;
        if requested != self.output {
            if let Some((format, color_space)) = requested.surface(&caps) {
                self.config.format = format;
                self.config.color_space = color_space;
                self.surface.configure(&self.device, &self.config);
                let mut next = PresentPass::new(&self.device, format, requested);
                next.inherit_canvas(&self.device, &mut self.present);
                self.present = next;
                self.output = requested;
                self.error = None;
            } else {
                self.error = Some(format!(
                    "{} is unavailable on this display",
                    requested.label()
                ));
                self.app.display.output = self.output;
            }
        }
        // A surface larger than the device's textures is a validation error (a panic in wgpu's
        // default handler): clamp it; the canvas then reports `TooLarge` and skips the frame.
        let max = self.device.limits().max_texture_dimension_2d;
        let (width, height) = (size.width.min(max), size.height.min(max));
        if width != self.config.width || height != self.config.height {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
        }
        let info = self.surface.display_hdr_info(&self.adapter);
        let target = self.app.display.target(self.output, &info);
        self.ctx.data_mut(|d| {
            d.insert_temp(
                soluwell_viewport::display::state_id(),
                soluwell_viewport::display::DisplayState {
                    output: self.output,
                    target,
                    info,
                    error: self.error.clone(),
                    available,
                },
            )
        });
        let raw = self.input.take_egui_input(&self.window);
        let full = self.ctx.run_ui(raw, |root| self.app.ui(root));
        if let Some(error) = self.app.take_fatal() {
            anyhow::bail!(error);
        }
        // The one place the next frame is planned: what egui (and through it the app) asked for.
        if let Some(output) = full.viewport_output.get(&egui::ViewportId::ROOT) {
            self.schedule(output.repaint_delay);
        }
        self.input
            .handle_platform_output(&self.window, full.platform_output);
        for out in full.viewport_output.values() {
            for cmd in &out.commands {
                match cmd {
                    egui::ViewportCommand::Close => events.exit(),
                    egui::ViewportCommand::CursorGrab(mode) if self.cursor_grab != *mode => {
                        use winit::window::CursorGrabMode;
                        let grab = match mode {
                            egui::CursorGrab::None => CursorGrabMode::None,
                            egui::CursorGrab::Confined => CursorGrabMode::Confined,
                            egui::CursorGrab::Locked => CursorGrabMode::Locked,
                        };
                        let result = self.window.set_cursor_grab(grab).or_else(|e| {
                            if grab == CursorGrabMode::Locked {
                                self.window.set_cursor_grab(CursorGrabMode::Confined)
                            } else {
                                Err(e)
                            }
                        });
                        if let Err(e) = result {
                            log::warn!("Mouse capture: {e}");
                        }
                        self.cursor_grab = *mode;
                    }
                    _ => {}
                }
            }
        }
        for (id, deltas) in &full.textures_delta.set {
            for delta in deltas {
                self.renderer
                    .update_texture(&self.device, &self.queue, *id, delta);
            }
        }
        let jobs = self.ctx.tessellate(full.shapes, full.pixels_per_point);
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [size.width, size.height],
            pixels_per_point: full.pixels_per_point,
        };
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
            // Not presented: the frame is drawn again (an occluded window waits for an event).
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.config);
                self.schedule(Duration::ZERO);
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Timeout => {
                self.schedule(Duration::ZERO);
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Occluded => return Ok(()),
            other => anyhow::bail!("surface acquisition: {other:?}"),
        };
        let mut encoder = self.device.create_command_encoder(&Default::default());
        let buffers = match paint_canvas(
            &self.device,
            &self.queue,
            &mut self.renderer,
            &mut self.present,
            &mut encoder,
            &jobs,
            &screen,
            wgpu::Color::BLACK,
        ) {
            Ok(buffers) => buffers,
            // The window outgrew the device's texture limit: no frame until it fits again.
            Err(error @ soluwell_viewport::display::CanvasError::TooLarge { .. }) => {
                log::error!("{error}");
                return Ok(());
            }
            Err(error) => anyhow::bail!("window canvas: {error}"),
        };
        let view = frame.texture.create_view(&Default::default());
        self.present.draw(&self.queue, &mut encoder, &view, target);
        // This queue belongs exclusively to Native. The shared submit helper's
        // process-wide mutex would serialize presentation behind OCIO workers.
        self.queue
            .submit(buffers.into_iter().chain([encoder.finish()]));
        self.queue.present(frame);
        for id in &full.textures_delta.free {
            self.renderer.free_texture(id);
        }
        // A window screenshot captures the canvas of the frame just presented, through the
        // present pass, and is read back and written on the file worker (`window_shot`).
        if let Some(shot) = self.app.take_window_shot() {
            let capture = self.present.capture(
                &self.device,
                &self.queue,
                shot.output(self.output),
                shot.target(target),
            );
            self.app
                .submit_window_shot(&self.ctx, shot, capture, &self.device);
            if let Some(error) = self.app.take_fatal() {
                anyhow::bail!(error);
            }
        }
        Ok(())
    }
}

/// Paint tessellated egui `jobs` into `present`'s float canvas, cleared to `clear` (canvas
/// values: extended sRGB, so the window's black, or HDR and negative values in the headless
/// test), at the size of `screen`: buffer uploads and the egui pass, recorded into `encoder`.
/// Returns egui's own command buffers, to submit before `encoder`, or why the canvas could not
/// be (re)allocated at that size (nothing was recorded then). The window's redraw and the
/// headless window-screenshot test draw through here, so a test frame is composited exactly
/// like a presented one.
#[allow(
    clippy::too_many_arguments,
    reason = "the distinct GPU objects of one pass; the window owns them as separate fields"
)]
pub(crate) fn paint_canvas(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer: &mut egui_wgpu::Renderer,
    present: &mut PresentPass,
    encoder: &mut wgpu::CommandEncoder,
    jobs: &[egui::ClippedPrimitive],
    screen: &egui_wgpu::ScreenDescriptor,
    clear: wgpu::Color,
) -> Result<Vec<wgpu::CommandBuffer>, soluwell_viewport::display::CanvasError> {
    let [width, height] = screen.size_in_pixels;
    present.prepare_canvas(device, queue, width, height)?;
    let canvas = present.canvas(device, width, height)?;
    let buffers = renderer.update_buffers(device, queue, encoder, jobs, screen);
    let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("frac.egui"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: canvas,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(clear),
                store: wgpu::StoreOp::Store,
            },
        })],
        ..Default::default()
    });
    renderer.render(&mut pass.forget_lifetime(), jobs, screen);
    Ok(buffers)
}
impl ApplicationHandler for Host {
    fn device_event(
        &mut self,
        _: &ActiveEventLoop,
        _: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        // Raw motion drives flight look while the cursor is locked: egui takes it when the
        // pointer is in the window or a button is held, and a frame consumes it.
        if let Some(native) = &mut self.window
            && let winit::event::DeviceEvent::MouseMotion { delta } = event
            && native.input.on_mouse_motion(delta)
        {
            native.schedule(Duration::ZERO);
        }
    }
    fn resumed(&mut self, events: &ActiveEventLoop) {
        if self.window.is_none() {
            match Native::new(events) {
                Ok(native) => {
                    native.window.request_redraw();
                    self.window = Some(native);
                }
                Err(e) => {
                    self.error = Some(e);
                    events.exit();
                }
            }
        }
    }
    fn about_to_wait(&mut self, events: &ActiveEventLoop) {
        let Some(native) = &self.window else {
            return;
        };
        match native.next_redraw {
            Some(due) if Instant::now() >= due => {
                native.window.request_redraw();
                events.set_control_flow(ControlFlow::Wait);
            }
            Some(due) => events.set_control_flow(ControlFlow::WaitUntil(due)),
            // Idle: nothing animates, the next input wakes the loop.
            None => events.set_control_flow(ControlFlow::Wait),
        }
    }
    fn window_event(&mut self, events: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let Some(native) = &mut self.window else {
            return;
        };
        // egui says whether the event needs a frame (input, resize, focus, ...).
        if native.input.on_window_event(&native.window, &event).repaint
            || matches!(event, WindowEvent::Occluded(false))
        {
            native.schedule(Duration::ZERO);
        }
        match event {
            WindowEvent::CloseRequested => events.exit(),
            WindowEvent::RedrawRequested => {
                if let Err(e) = native.redraw(events) {
                    self.error = Some(e);
                    events.exit();
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::window_shot::{ShotKind, WindowFiles, WindowShot};

    /// Headless whole-window shot: egui draws a small frame (a mid-grey block and a label) into
    /// the float canvas through the window's own `paint_canvas`, the present pass captures it the
    /// way the window does after presenting, and the file writer saves the EXR and the PQ PNG.
    /// Oracles: the grey block (egui sRGB code 128) decodes to linear 0.2159 in the EXR and to
    /// 0.2159 x 203 = 43.8 nits in the PQ PNG. The uncovered canvas holds the clear colour
    /// (4.0, -0.1, 1.0) in canvas (extended sRGB) values: it must come through the real present
    /// shader unclipped in the EXR (about 25 and -0.01) and as the PQ of its BT.2020 nits in the
    /// PNG. The PNG carries cICP 9/16/0/1 and both files have the window's size.
    #[test]
    fn gpu_window_shot_writes_exr_and_pq_png() {
        let _gpu = crate::test_gpu::lock();
        let gpu = gpu_info::shared_device().expect("wgpu adapter");
        let (device, queue) =
            gpu_info::request_max_device_blocking(&gpu.adapter, gpu.device.features())
                .expect("wgpu device");
        let (w, h) = (96u32, 64u32);
        let clear = [4.0f32, -0.1, 1.0];
        let ctx = egui::Context::default();
        let block = egui::Rect::from_min_max(egui::pos2(56.0, 16.0), egui::pos2(96.0, 64.0));
        let raw = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(w as f32, h as f32),
            )),
            ..Default::default()
        };
        let full = ctx.run_ui(raw, |ui| {
            ui.painter()
                .rect_filled(block, 0.0, egui::Color32::from_gray(128));
            ui.label("W");
        });
        assert_eq!(full.pixels_per_point, 1.0);

        let mut renderer = egui_wgpu::Renderer::new(
            &device,
            soluwell_viewport::display::CANVAS_FORMAT,
            Default::default(),
        );
        let mut present = PresentPass::new(&device, wgpu::TextureFormat::Rgba8Unorm, Output::Sdr8);
        for (id, deltas) in &full.textures_delta.set {
            for delta in deltas {
                renderer.update_texture(&device, &queue, *id, delta);
            }
        }
        let jobs = ctx.tessellate(full.shapes, full.pixels_per_point);
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [w, h],
            pixels_per_point: full.pixels_per_point,
        };
        let mut encoder = device.create_command_encoder(&Default::default());
        let buffers = paint_canvas(
            &device,
            &queue,
            &mut renderer,
            &mut present,
            &mut encoder,
            &jobs,
            &screen,
            wgpu::Color {
                r: f64::from(clear[0]),
                g: f64::from(clear[1]),
                b: f64::from(clear[2]),
                a: 1.0,
            },
        )
        .unwrap();
        queue.submit(buffers.into_iter().chain([encoder.finish()]));

        let root = std::env::temp_dir().join(format!("warpbro-window-gpu-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let white = crate::color::BT2408_SDR_WHITE_NITS;
        let shot = WindowShot {
            kind: ShotKind::Linear {
                root: root.clone(),
                stem: "ui".into(),
                files: WindowFiles::ExrAndPq,
                sdr_white_nits: white,
            },
            quit: false,
        };
        // An HDR window's levels: the capture must not inherit its white (canvas 1.0 stays 1.0).
        let shown = soluwell_viewport::display::Target {
            hdr: true,
            white: 240.0,
            peak: 1000.0,
        };
        let capture = present
            .capture(
                &device,
                &queue,
                shot.output(Output::Sdr8),
                shot.target(shown),
            )
            .expect("capture")
            .wait(&device)
            .expect("readback");
        assert_eq!((capture.width, capture.height), (w, h));
        let paths = crate::window_shot::save(&shot, &capture).expect("window shot files");
        assert_eq!(paths.len(), 2);
        let (exr, png) = (&paths[0], &paths[1]);
        assert!(exr.ends_with("ui.window.exr") && png.ends_with("ui.window.pq.png"));

        let grey = soluwell_viewport::display::transfer::eotf(128.0 / 255.0);
        let at = |x: u32, y: u32| (y * w + x) as usize;
        let image = crate::exr_io::read_rgb(exr).unwrap();
        assert_eq!((image.width, image.height), (w, h));
        for c in image.pixels[at(80, 40)] {
            assert!((c - grey).abs() < 0.005, "grey block {c}, want {grey}");
        }
        // The clear colour, decoded on the CPU: f16 canvas storage bounds the error.
        let hdr = clear
            .map(|c| soluwell_viewport::display::transfer::eotf(half::f16::from_f32(c).to_f32()));
        let got = image.pixels[at(4, 60)];
        assert!(
            hdr[0] > 20.0 && hdr[1] < 0.0,
            "the oracle is HDR and negative: {hdr:?}"
        );
        for (g, want) in got.iter().zip(hdr) {
            assert!(
                (g - want).abs() <= want.abs() * 2e-3 + 1e-5,
                "clear colour {got:?}, want {hdr:?} unclipped"
            );
        }

        let decoder = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(png).unwrap()));
        let mut reader = decoder.read_info().unwrap();
        let cicp = reader.info().coding_independent_code_points.expect("cICP");
        assert_eq!(
            [
                cicp.color_primaries,
                cicp.transfer_function,
                cicp.matrix_coefficients,
                u8::from(cicp.is_video_full_range_image)
            ],
            [9, 16, 0, 1]
        );
        assert_eq!((reader.info().width, reader.info().height), (w, h));
        let mut buf = vec![0; reader.output_buffer_size().unwrap()];
        reader.next_frame(&mut buf).unwrap();
        let sample = |i: usize| u16::from_be_bytes([buf[2 * i], buf[2 * i + 1]]);
        let nits = soluwell_viewport::display::pq_nits(f32::from(sample(4 * at(80, 40))) / 65535.0);
        assert!(
            (nits - grey * white).abs() < 1.5,
            "grey block at {nits} nits, want {}",
            grey * white
        );
        // The HDR clear colour: PQ of its BT.2020 nits (negative ones black), not clipped at white.
        let want = soluwell_viewport::display::rec2020_nits(hdr, white).map(|n| {
            (soluwell_viewport::display::pq(n.clamp(0.0, 10_000.0)) * 65535.0).round() as u16
        });
        let got: Vec<u16> = (0..3).map(|c| sample(4 * at(4, 60) + c)).collect();
        // SDR white (203 nits) is code 38055; red sits near 3200 nits.
        assert!(
            want[0] > 50_000,
            "the oracle is far above SDR white: {want:?}"
        );
        for (g, want) in got.iter().zip(want) {
            assert!(
                g.abs_diff(want) <= 8,
                "clear colour codes {got:?}, want {want:?}"
            );
        }
        std::fs::remove_dir_all(root).unwrap();
    }
}
