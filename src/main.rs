#![recursion_limit = "256"]
//! WarpBro: a path-traced 3D fractal browser. Rust CUDA kernels (cuda-oxide) port ofx-rs
//! `ofx-fractal` (eight distance-estimated families, DE sphere tracing) and render-rs
//! (`pt-integrator` path tracing, `standard-surface-bsdf`); the UI is egui.
//!
//!   WarpBro                                  the browser
//!   WarpBro --gallery DIR [W H SPP]          render every gallery preset to DIR/*.png
//!   WarpBro --bench [W H SPP]                time every preset

mod animation;
mod app;
mod camera_orbit;
mod color;
mod denoise;
mod environment;
mod inspector;
#[cfg(test)]
mod timeline;
// Keep the copied viewer API intact, including its CPU oracle used by tests.
mod export;
mod file_dialogs;
mod gpu;
mod hotkeys;
mod io_service;
mod material_gallery;
mod materials;
#[allow(dead_code)]
mod ocio;
mod palette;
mod params;
mod path_sampling;
mod presets;
mod preview;
mod render;
mod render_bench;
mod render_service;
mod scene;
mod transfer;
mod transmission;
mod templates;
mod ui_style;
mod window;
mod world;
mod world_ui;

use std::time::Instant;

fn arg<T: std::str::FromStr>(args: &[String], i: usize, default: T) -> T {
    args.get(i).and_then(|v| v.parse().ok()).unwrap_or(default)
}

pub fn slug(name: &str) -> String {
    let mut s: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect();
    while s.contains("--") {
        s = s.replace("--", "-");
    }
    s.trim_matches('-').to_string()
}

/// Render (or only time) every gallery preset offscreen.
fn headless(out: Option<&str>, w: usize, h: usize, spp: u32, hdr: bool, display_exr: bool) {
    let mut gpu = render::Gpu::new().unwrap_or_else(|e| panic!("{e}"));
    println!("GPU: {}  ·  {w}x{h}, {spp} spp", gpu.name);
    println!(
        "{:<28} {:>16} {:>9} {:>11}",
        "preset", "model", "ms/spp", "Msamples/s"
    );
    for mut scene in scene::Scene::gallery() {
        if hdr {
            scene.colour.config = "ocio://studio-config-latest".into();
            scene.colour.display = "Rec.2100-PQ - Display".into();
            scene.colour.view = color::HDR_VIEW.into();
        }
        let mut t = gpu.target(w, h);
        gpu.step(&mut t, &scene, 1, 0, None, spp <= 1); // warm-up
        let batch = 8u32;
        let t0 = Instant::now();
        while t.samples < spp {
            let n = batch.min(spp - t.samples);
            let final_pass = t.samples + n >= spp;
            gpu.step(&mut t, &scene, n, 0, None, final_pass);
        }
        let secs = t0.elapsed().as_secs_f64();
        let traced = (spp - 1).max(1) as f64;
        println!(
            "{:<28} {:>16} {:>9.2} {:>11.1}",
            scene.name,
            format!("{:?}", scene.material.model),
            secs * 1e3 / traced,
            (w * h) as f64 * traced / secs / 1e6
        );
        if let Some(dir) = out {
            let path = std::path::Path::new(dir).join(format!("{}.png", slug(&scene.name)));
            t.save_png(&path).expect("save png");
            if display_exr {
                t.save_display_exr(&path.with_extension("display.exr"))
                    .expect("save display EXR");
            }
        }
    }
}

fn animated_fixtures(
    dir: &str,
    width: usize,
    height: usize,
    samples: u32,
    all_frames: bool,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        width > 0
            && height > 0
            && width <= 16384
            && height <= 16384
            && width.checked_mul(height).is_some_and(|n| n <= 67_108_864),
        "Invalid fixture resolution"
    );
    anyhow::ensure!(
        samples > 0 && samples <= 1_000_000,
        "Invalid fixture samples"
    );
    std::fs::create_dir_all(dir)?;
    let mut gpu = render::Gpu::new().map_err(anyhow::Error::msg)?;
    for (index, descriptor) in presets::ANIMATED.iter().enumerate() {
        let scene = presets::scene(index).map_err(anyhow::Error::msg)?;
        let document = scene
            .document
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Preset has no document"))?;
        let output = std::path::Path::new(dir).join(slug(descriptor.name));
        std::fs::create_dir_all(&output)?;
        serde_json::to_writer_pretty(
            std::fs::File::create(output.join("scene.frac.json"))?,
            document,
        )?;
        let frames: Vec<u32> = if all_frames {
            (document.first..=document.last).collect()
        } else {
            vec![
                document.first,
                (document.first + document.last) / 2,
                document.last,
            ]
        };
        let mut target = gpu.target(width, height);
        for frame in frames {
            let evaluated = document
                .snapshot(f64::from(frame))
                .map_err(anyhow::Error::msg)?;
            let start = Instant::now();
            // Fixed seed keeps regression images repeatable. One target is reused between frames.
            for first_sample in (0..samples).step_by(8) {
                let count = 8.min(samples - first_sample);
                gpu.step(
                    &mut target,
                    &evaluated,
                    count,
                    0,
                    None,
                    first_sample + count == samples,
                );
                if let Some(error) = &target.colour_error {
                    anyhow::bail!("{error}");
                }
            }
            anyhow::ensure!(
                target.samples == samples,
                "Fixture accumulation did not complete"
            );
            target
                .save_png(&output.join(format!("frame.{frame:06}.png")))
                .map_err(anyhow::Error::msg)?;
            println!(
                "{} frame {}: {:.3}s",
                descriptor.name,
                frame,
                start.elapsed().as_secs_f64()
            );
        }
    }
    println!("Preparation: {:?}", gpu.preparation_stats());
    Ok(())
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        // Build-time warmup initializes the exact embedded module without opening
        // a window or authoring scene state. The CUDA driver owns its JIT cache;
        // this does not populate the Timeline frame cache or wait on the UI thread.
        Some("--warmup-cuda") => {
            let started = std::time::Instant::now();
            let gpu = render::Gpu::new().map_err(anyhow::Error::msg)?;
            println!("CUDA ready: {} ({:?})", gpu.name, started.elapsed());
            Ok(())
        }
        Some("--world-bench") => render_bench::run(&args),
        Some("--animated-fixtures") => animated_fixtures(
            args.get(2)
                .map(String::as_str)
                .unwrap_or("animated-fixtures"),
            arg(&args, 3, 640),
            arg(&args, 4, 360),
            arg(&args, 5, 32),
            args.iter().any(|a| a == "--all-frames"),
        ),
        Some("--gallery") => {
            let dir = args.get(2).cloned().unwrap_or_else(|| "gallery".into());
            headless(
                Some(&dir),
                arg(&args, 3, 960),
                arg(&args, 4, 540),
                arg(&args, 5, 64),
                args.iter().any(|a| a == "--hdr"),
                args.iter().any(|a| a == "--display-exr"),
            );
            Ok(())
        }
        Some("--bench") => {
            headless(
                None,
                arg(&args, 2, 960),
                arg(&args, 3, 540),
                arg(&args, 4, 32),
                false,
                false,
            );
            Ok(())
        }
        _ => app::run(),
    }
}
