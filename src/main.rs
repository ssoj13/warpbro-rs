#![recursion_limit = "256"]
//! WarpBro: a path-traced 3D fractal browser. The render worker drives the fractal-rs CUDA
//! kernels (cuda-oxide) or their WGSL port: the distance-estimated families of
//! `formulas::FORMULAS` (the classic ones on both backends, the extended ones on WGPU only), DE
//! sphere tracing, `pt-integrator` path tracing and `standard-surface-bsdf`. The UI is egui.
//!
//!   WarpBro                                          the browser
//!   WarpBro --gallery DIR [W H SPP] [--hdr] [--display-exr]
//!                                                    render every gallery preset to DIR
//!   WarpBro --bench [W H SPP]                        time every gallery preset
//!   WarpBro --world-bench [DIR] [W H SPP] [--case NAME] [--seed N] [--batch 4|8] [--adaptive T]
//!                         [--backend auto|cuda|wgpu]
//!                                                    measure evaluated worlds (`render_bench`)
//!   WarpBro --animated-fixtures [DIR] [W H SPP] [--all-frames]
//!                                                    render the animated presets' frames
//!   WarpBro --warmup                                 initialize the embedded CUDA module and
//!                                                    the WGPU device, module and post pipelines

mod animation;
// The parameter ABI is shared with the CUDA kernels and with ofx-rs through fractal-rs; host code
// keeps addressing it as crate::params.
use fractal_core::params;
mod app;
mod camera_orbit;
mod camera_recorder;
mod camera_slots;
mod color;
mod denoise;
mod environment;
mod formulas;
mod fs_name;
mod export;
mod exr_io;
mod file_dialogs;
mod hotkeys;
mod io_service;
mod material_gallery;
mod materials;
mod ocio;
mod overlay;
mod palette;
#[cfg(test)]
mod parity_tests;
// Unwired roulette / clamp policies awaiting a decision (plan1 Q4): no kernel uses them yet.
#[allow(dead_code)]
mod path_convergence;
mod presets;
mod preview;
mod render;
mod render_bench;
mod render_prep;
mod render_profiles;
mod render_profiles_ui;
mod render_service;
mod render_wgpu;
mod scene;
mod takes;
mod templates;
#[cfg(test)]
mod test_gpu;
mod window;
mod window_shot;
mod world;
mod world_ui;

use std::time::Instant;

/// Optional positional argument `i`: absent, or a `--flag` in its place (`--gallery DIR --hdr`),
/// takes `default`; anything else must parse, so `--bench 96O` is an error, not 960 silently.
fn arg<T: std::str::FromStr>(args: &[String], i: usize, default: T) -> anyhow::Result<T>
where
    T::Err: std::fmt::Display,
{
    match args.get(i).filter(|v| !v.starts_with("--")) {
        None => Ok(default),
        Some(v) => v
            .parse()
            .map_err(|e| anyhow::anyhow!("argument {i} ({v:?}): {e}")),
    }
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
    video_io::job::new_out_dir(root).map_err(|e| e.to_string())
}

/// Render (or only time) every gallery preset offscreen.
fn headless(
    out: Option<&str>,
    w: usize,
    h: usize,
    spp: u32,
    hdr: bool,
    display_exr: bool,
) -> anyhow::Result<()> {
    render::validate_size(w, h, spp)?;
    let mut gpu = render::Gpu::new()?;
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
        let mut t = render::Target::new(w, h);
        gpu.step(&mut t, &scene, 1, 0, spp <= 1)?; // warm-up
        let t0 = Instant::now();
        while !gpu.advance(&mut t, &scene, spp, 0, render::Batch::Fixed(8))? {}
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
            let frame = render_service::Frame::snapshot(&t, 0, false, false, 100.0);
            let encoding = render_service::PngEncoding::displayed(frame.light_kind.hdr());
            let dir = std::path::Path::new(dir);
            let stem = fs_name::stem(&scene.name);
            let path = dir.join(fs_name::frame_file(&stem, None, encoding.suffix()));
            frame
                .save_png(&path, encoding, color::BT2408_SDR_WHITE_NITS, true)
                .map_err(anyhow::Error::msg)?;
            if display_exr {
                let exr = render_service::FrameFile::DisplayExr.suffix();
                frame
                    .save_display_exr(
                        &dir.join(fs_name::frame_file(&stem, None, exr)),
                        color::BT2408_SDR_WHITE_NITS,
                    )
                    .map_err(anyhow::Error::msg)?;
            }
        }
    }
    Ok(())
}

fn animated_fixtures(
    dir: &str,
    width: usize,
    height: usize,
    samples: u32,
    all_frames: bool,
) -> anyhow::Result<()> {
    render::validate_size(width, height, samples)?;
    std::fs::create_dir_all(dir)?;
    let mut gpu = render::Gpu::new()?;
    let animated = templates::BUILTINS
        .iter()
        .enumerate()
        .filter(|(index, _)| templates::ANIMATED.contains(index));
    for (index, descriptor) in animated {
        let scene = templates::builtin_scene(index).map_err(anyhow::Error::msg)?;
        let document = scene
            .document
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Preset has no document"))?;
        let output = std::path::Path::new(dir).join(fs_name::stem(descriptor.name));
        std::fs::create_dir_all(&output)?;
        serde_json::to_writer_pretty(
            std::fs::File::create(output.join(fs_name::frame_file(
                "scene",
                None,
                fs_name::SCENE_SUFFIX,
            )))?,
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
        for frame in frames {
            let evaluated = document
                .snapshot(f64::from(frame))
                .map_err(anyhow::Error::msg)?;
            let start = Instant::now();
            // A fixed seed keeps regression images repeatable. Every frame gets its own target,
            // so identical consecutive frames never accumulate into one another.
            let mut target = render::Target::new(width, height);
            while !gpu.advance(&mut target, &evaluated, samples, 0, render::Batch::Fixed(8))? {}
            let image = render_service::Frame::snapshot(&target, 0, false, false, 100.0);
            let encoding = render_service::PngEncoding::displayed(image.light_kind.hdr());
            image
                .save_png(
                    &output.join(fs_name::frame_file("frame", Some(frame), encoding.suffix())),
                    encoding,
                    color::BT2408_SDR_WHITE_NITS,
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
        // Build-time warmup, without a window or scene state: the CUDA driver JIT-caches the
        // embedded module, and the WGPU device opens with its dynamic module and post pipelines
        // (trace pipelines compile per family module on first use, then come from disk).
        Some("--warmup") => {
            let started = std::time::Instant::now();
            let gpu = render::Gpu::new()?;
            println!("CUDA ready: {} ({:?})", gpu.name, started.elapsed());
            let started = std::time::Instant::now();
            let backend = render_wgpu::warmup().map_err(anyhow::Error::msg)?;
            println!("WGPU ready on {backend:?} ({:?})", started.elapsed());
            Ok(())
        }
        Some("--world-bench") => render_bench::run(&args),
        Some("--animated-fixtures") => animated_fixtures(
            args.get(2)
                .map(String::as_str)
                .unwrap_or("animated-fixtures"),
            arg(&args, 3, 640)?,
            arg(&args, 4, 360)?,
            arg(&args, 5, 32)?,
            args.iter().any(|a| a == "--all-frames"),
        ),
        Some("--gallery") => {
            let dir = args.get(2).cloned().unwrap_or_else(|| "gallery".into());
            headless(
                Some(&dir),
                arg(&args, 3, 960)?,
                arg(&args, 4, 540)?,
                arg(&args, 5, 64)?,
                args.iter().any(|a| a == "--hdr"),
                args.iter().any(|a| a == "--display-exr"),
            )
        }
        Some("--bench") => headless(
            None,
            arg(&args, 2, 960)?,
            arg(&args, 3, 540)?,
            arg(&args, 4, 32)?,
            false,
            false,
        ),
        _ => app::run(),
    }
}
