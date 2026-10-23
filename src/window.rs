//! Native window: WarpBro on the shared `soluwell-window` host (egui-widgets-rs), which owns the
//! winit window, the surface and egui-winit, draws egui into the float display canvas and presents
//! it in the output the user picked (SDR, HDR10 PQ, HLG, scRGB). Frames are drawn on demand: on
//! input and when egui asks for one (`repaint_delay`, requested while something animates:
//! `App::animating`), at most `App::gui_fps` per second; an idle window sleeps.
use anyhow::Context;

/// Open the window and run the app until it closes.
pub fn run() -> anyhow::Result<()> {
    let gpu = gpu_info::shared_device().context("wgpu: shared GPU unavailable")?;
    log::info!("Display/OCIO adapter: {:?}", gpu.adapter.get_info());
    // Surface configuration waits for its device to become idle. OCIO's continuously submitting
    // offscreen queue must never own that wait: the window gets a private device and queue on
    // the shared instance and adapter.
    let (device, queue) =
        gpu_info::request_max_device_blocking(&gpu.adapter, gpu.device.features())
            .context("wgpu: private presentation device unavailable")?;
    let ctx = egui::Context::default();
    let mut fonts = egui::FontDefinitions::default();
    soluwell_config::add_icon_font(&mut fonts);
    ctx.set_fonts(fonts);
    // Solid scroll bars reserve their width: egui's default floating bars paint over the
    // right edge of every scroll area (e.g. the Attribute Editor's per-row expand buttons).
    ctx.all_styles_mut(|style| style.spacing.scroll = egui::style::ScrollStyle::solid());
    // A dark title bar, applied by the host with the first frame's viewport commands.
    ctx.send_viewport_cmd(egui::ViewportCommand::SetTheme(egui::SystemTheme::Dark));
    let options = soluwell_window::Options {
        viewport: egui::ViewportBuilder::default()
            .with_title("WarpBro")
            .with_inner_size([1600.0, 940.0])
            .with_min_inner_size([900.0, 560.0]),
        wgpu: egui_wgpu::WgpuConfiguration {
            wgpu_setup: egui_wgpu::WgpuSetup::Existing(egui_wgpu::WgpuSetupExisting {
                instance: gpu.instance.clone(),
                adapter: gpu.adapter.clone(),
                device,
                queue,
            }),
            ..Default::default()
        },
        ..Default::default()
    };
    soluwell_window::run(options, ctx, |_, _| Window {
        app: crate::app::App::new(),
    })?;
    Ok(())
}

/// The app as the host drives it.
struct Window {
    app: crate::app::App,
}

impl soluwell_window::App for Window {
    fn ui(&mut self, ui: &mut egui::Ui, _: &soluwell_window::RenderState) {
        self.app.ui(ui);
    }

    fn display_prefs(&self) -> soluwell_window::DisplayPrefs {
        self.app.display
    }

    fn max_fps(&self) -> Option<f32> {
        Some(self.app.gui_fps() as f32)
    }

    fn take_fatal(&mut self) -> Option<String> {
        self.app.take_fatal()
    }

    /// A window screenshot captures the canvas of the frame just presented, through the present
    /// pass, and is read back and written on the file worker (`window_shot`).
    fn presented(&mut self, ctx: &egui::Context, frame: &soluwell_window::Presented<'_>) {
        if let Some(shot) = self.app.take_window_shot() {
            let capture = frame.capture(shot.output(frame.output), shot.target(frame.target));
            self.app
                .submit_window_shot(ctx, shot, capture, &frame.rs.device);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::window_shot::{ShotKind, WindowFiles, WindowShot};
    use soluwell_viewport::display::{Output, PresentPass};

    /// Paint tessellated egui `jobs` into `present`'s float canvas (as the host's frame does), cleared to `clear` (canvas
    /// values: extended sRGB, so the window's black, or HDR and negative values in the headless
    /// test), at the size of `screen`: buffer uploads and the egui pass, recorded into `encoder`.
    /// Returns egui's own command buffers, to submit before `encoder`, or why the canvas could not
    /// be (re)allocated at that size (nothing was recorded then).
    #[allow(
        clippy::too_many_arguments,
        reason = "the distinct GPU objects of one pass; the window owns them as separate fields"
    )]
    fn paint_canvas(
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
