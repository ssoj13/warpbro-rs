#![recursion_limit = "256"]
//! frac-rs: a path-traced 3D fractal browser. Rust CUDA kernels (cuda-oxide) port ofx-rs
//! `ofx-fractal` (eight distance-estimated families, DE sphere tracing) and render-rs
//! (`pt-integrator` path tracing, `standard-surface-bsdf`); the UI is egui.
//!
//!   frac-rs                                  the browser
//!   frac-rs --gallery DIR [W H SPP]          render every gallery preset to DIR/*.png
//!   frac-rs --bench [W H SPP]                time every preset

mod animation;
mod app;
mod color;
mod denoise;
mod environment;
mod inspector;
#[cfg(test)]
mod timeline;
// Keep the copied viewer API intact, including its CPU oracle used by tests.
mod export;
mod gpu;
mod io_service;
mod materials;
#[allow(dead_code)]
mod ocio;
mod palette;
mod params;
mod render;
mod render_service;
mod scene;
mod transfer;
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

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
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
