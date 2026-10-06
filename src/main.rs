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
mod camera_slots;
mod color;
mod denoise;
mod environment;
mod fs_name;
mod inspector;
// Keep the copied viewer API intact, including its CPU oracle used by tests.
mod export;
mod exr_io;
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
mod sampler;
mod scene;
mod transfer;
mod transmission;
mod templates;
mod window;
mod world;
mod world_ui;

use std::time::Instant;

fn arg<T: std::str::FromStr>(args: &[String], i: usize, default: T) -> T {
    args.get(i).and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// The WarpBro profile, `~/.warpbro` (or `$WARPBRO_HOME`): `settings.json`, `bookmarks/`,
/// `templates/` and every render / screenshot under `out/`.
pub fn warpbro_dir() -> std::path::PathBuf {
    if let Some(root) = std::env::var_os("WARPBRO_HOME") {
        return std::path::PathBuf::from(root);
    }
    // Tests never touch the operator's real profile: each test process gets its own empty one.
    #[cfg(test)]
    return std::env::temp_dir().join(format!("warpbro-test-{}", std::process::id()));
    #[cfg(not(test))]
    dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".warpbro")
}

/// `~/.warpbro/out`: every export and screenshot gets its own timestamped folder here.
pub fn out_root() -> std::path::PathBuf {
    warpbro_dir().join("out")
}

/// A new, unique `<root>/<local timestamp>` folder for one export or screenshot.
pub fn new_out_dir(root: &std::path::Path) -> Result<std::path::PathBuf, String> {
    std::fs::create_dir_all(root).map_err(|e| format!("{}: {e}", root.display()))?;
    let stamp = jiff::Zoned::now().strftime("%Y-%m-%d_%H-%M-%S").to_string();
    // Two outputs within one second get -2, -3, ...: `create_dir` is the atomic claim.
    for n in 1..1000 {
        let dir = root.join(if n == 1 { stamp.clone() } else { format!("{stamp}-{n}") });
        match std::fs::create_dir(&dir) {
            Ok(()) => return Ok(dir),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(format!("{}: {e}", dir.display())),
        }
    }
    Err(format!("{}: no free folder name for {stamp}", root.display()))
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
        while !t.complete(spp) {
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
            let encoding = crate::render_service::PngEncoding::displayed(t.light_kind.hdr());
            let dir = std::path::Path::new(dir);
            let stem = fs_name::stem(&scene.name);
            let path = dir.join(fs_name::frame_file(&stem, None, encoding.suffix()));
            t.save_png(&path, encoding, crate::color::BT2408_SDR_WHITE_NITS, true).expect("save png");
            if display_exr {
                let exr = crate::render_service::FrameFile::DisplayExr.suffix();
                t.save_display_exr(&dir.join(fs_name::frame_file(&stem, None, exr)))
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
        let output = std::path::Path::new(dir).join(fs_name::stem(descriptor.name));
        std::fs::create_dir_all(&output)?;
        serde_json::to_writer_pretty(
            std::fs::File::create(output.join(fs_name::frame_file("scene", None, fs_name::SCENE_SUFFIX)))?,
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
            let encoding = crate::render_service::PngEncoding::displayed(target.light_kind.hdr());
            target
                .save_png(
                    &output.join(fs_name::frame_file("frame", Some(frame), encoding.suffix())),
                    encoding,
                    crate::color::BT2408_SDR_WHITE_NITS,
                    true,
                )
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
