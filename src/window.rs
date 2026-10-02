//! Native window shell. egui renders to egui-display's float canvas; presentation
//! selects BOTH the swapchain format and colour space (eframe only offered SDR).
use anyhow::{Context, Result};
use egui_display::{Output, PresentPass};
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
    next_redraw: Instant,
    cursor_grab: egui::CursorGrab,
}
impl Native {
    fn new(events: &ActiveEventLoop) -> Result<Self> {
        let window = Arc::new(
            events.create_window(
                Window::default_attributes()
                    .with_title("frac-rs — path-traced fractals on CUDA (Rust)")
                    .with_theme(Some(winit::window::Theme::Dark))
                    .with_inner_size(winit::dpi::LogicalSize::new(1600.0, 940.0))
                    .with_min_inner_size(winit::dpi::LogicalSize::new(900.0, 560.0)),
            )?,
        );
        let gpu = gpu_info::shared_device().context("wgpu: shared GPU unavailable")?;
        log::info!("Display/OCIO adapter: {:?}", gpu.adapter.get_info());
        let adapter = gpu.adapter.clone();
        let device = gpu.device.clone();
        let queue = gpu.queue.clone();
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
        egui_phosphor::add_to_fonts(&mut fonts, egui_phosphor::Variant::Regular);
        ctx.set_fonts(fonts);
        egui_attr_table::set_label_width(&ctx, 130.0);
        let input = egui_winit::State::new(
            ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            window.theme(),
            Some(device.limits().max_texture_dimension_2d as usize),
        );
        let renderer =
            egui_wgpu::Renderer::new(&device, egui_display::CANVAS_FORMAT, Default::default());
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
            next_redraw: Instant::now(),
            cursor_grab: egui::CursorGrab::None,
        })
    }
    fn redraw(&mut self, events: &ActiveEventLoop) -> Result<()> {
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
        if size.width != self.config.width || size.height != self.config.height {
            self.config.width = size.width;
            self.config.height = size.height;
            self.surface.configure(&self.device, &self.config);
        }
        let info = self.surface.display_hdr_info(&self.adapter);
        let target = self.app.display.target(self.output, &info);
        self.ctx.data_mut(|d| {
            d.insert_temp(
                egui_display::state_id(),
                egui_display::DisplayState {
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
        let delay = full
            .viewport_output
            .get(&egui::ViewportId::ROOT)
            .map_or(Duration::from_secs(60), |v| v.repaint_delay);
        self.next_redraw = Instant::now() + delay.min(Duration::from_secs(60));
        self.input
            .handle_platform_output(&self.window, full.platform_output);
        let mut screenshot = false;
        for out in full.viewport_output.values() {
            for cmd in &out.commands {
                match cmd {
                    egui::ViewportCommand::Close => events.exit(),
                    egui::ViewportCommand::Screenshot(_) => screenshot = true,
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
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.config);
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return Ok(());
            }
            other => anyhow::bail!("surface acquisition: {other:?}"),
        };
        self.present
            .prepare_canvas(&self.device, &self.queue, size.width, size.height);
        let mut encoder = self.device.create_command_encoder(&Default::default());
        let buffers =
            self.renderer
                .update_buffers(&self.device, &self.queue, &mut encoder, &jobs, &screen);
        {
            let canvas = self.present.canvas(&self.device, size.width, size.height);
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("frac.egui"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: canvas,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            self.renderer
                .render(&mut pass.forget_lifetime(), &jobs, &screen);
        }
        let view = frame.texture.create_view(&Default::default());
        self.present.draw(&self.queue, &mut encoder, &view, target);
        let _ = gpu_info::submit(
            &self.queue,
            "frac-rs present",
            buffers.into_iter().chain([encoder.finish()]),
        );
        self.queue.present(frame);
        for id in &full.textures_delta.free {
            self.renderer.free_texture(id);
        }
        if screenshot {
            // FRAC_SNAP captures the canvas using the shared output writer below.
            self.capture_snap(target)?;
        }
        Ok(())
    }
    fn capture_snap(&mut self, target: egui_display::Target) -> Result<()> {
        let Some(path) = std::env::var_os("FRAC_SNAP") else {
            return Ok(());
        };
        let capture = self
            .present
            .capture(&self.device, &self.queue, self.output, target)?
            .wait(&self.device)?;
        capture.save(std::path::Path::new(&path))?;
        self.ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        Ok(())
    }
}
impl ApplicationHandler for Host {
    fn device_event(
        &mut self,
        _: &ActiveEventLoop,
        _: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        if let Some(native) = &mut self.window
            && let winit::event::DeviceEvent::MouseMotion { delta } = event
        {
            let _ = native.input.on_mouse_motion(delta);
            if native.cursor_grab != egui::CursorGrab::None {
                native.window.request_redraw();
            }
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
        if let Some(native) = &self.window {
            if Instant::now() >= native.next_redraw {
                native.window.request_redraw();
                events.set_control_flow(ControlFlow::Wait);
            } else {
                events.set_control_flow(ControlFlow::WaitUntil(native.next_redraw));
            }
        }
    }
    fn window_event(&mut self, events: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let Some(native) = &mut self.window else {
            return;
        };
        let _ = native.input.on_window_event(&native.window, &event);
        match event {
            WindowEvent::CloseRequested => events.exit(),
            WindowEvent::RedrawRequested => {
                if let Err(e) = native.redraw(events) {
                    self.error = Some(e);
                    events.exit();
                }
            }
            _ => native.window.request_redraw(),
        }
    }
}
