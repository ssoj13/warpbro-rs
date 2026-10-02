//! frac-rs: a path-traced 3D fractal browser. Rust CUDA kernels (cuda-oxide) port ofx-rs
//! `ofx-fractal` (eight distance-estimated families, DE sphere tracing) and render-rs
//! (`pt-integrator` path tracing, `standard-surface-bsdf`); the UI is egui.
//!
//!   frac-rs                                  the browser
//!   frac-rs --gallery DIR [W H SPP]          render every gallery preset to DIR/*.png
//!   frac-rs --bench [W H SPP]                time every preset

mod app;
mod gpu;
mod palette;
mod params;
mod render;
mod scene;

use std::time::Instant;

fn arg<T: std::str::FromStr>(args: &[String], i: usize, default: T) -> T {
    args.get(i).and_then(|v| v.parse().ok()).unwrap_or(default)
}

pub fn slug(name: &str) -> String {
    let mut s: String = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '-' })
        .collect();
    while s.contains("--") {
        s = s.replace("--", "-");
    }
    s.trim_matches('-').to_string()
}

/// Render (or only time) every gallery preset offscreen.
fn headless(out: Option<&str>, w: usize, h: usize, spp: u32) {
    let mut gpu = render::Gpu::new().unwrap_or_else(|e| panic!("{e}"));
    println!("GPU: {}  ·  {w}x{h}, {spp} spp", gpu.name);
    println!("{:<28} {:>16} {:>9} {:>11}", "preset", "model", "ms/spp", "Msamples/s");
    for scene in scene::Scene::gallery() {
        let mut t = gpu.target(w, h);
        gpu.step(&mut t, &scene, 1, 0, None); // warm-up
        let batch = 8u32;
        let t0 = Instant::now();
        while t.samples < spp {
            let n = batch.min(spp - t.samples);
            gpu.step(&mut t, &scene, n, 0, None);
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
        }
    }
}

fn main() -> eframe::Result {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("--gallery") => {
            let dir = args.get(2).cloned().unwrap_or_else(|| "gallery".into());
            headless(Some(&dir), arg(&args, 3, 960), arg(&args, 4, 540), arg(&args, 5, 64));
            Ok(())
        }
        Some("--bench") => {
            headless(None, arg(&args, 2, 960), arg(&args, 3, 540), arg(&args, 4, 32));
            Ok(())
        }
        _ => app::run(),
    }
}
