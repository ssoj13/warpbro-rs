#![recursion_limit = "256"]
//! WarpBro: a path-traced 3D fractal browser built on top of `sl-studio`.
//!
//! WarpBro registers its `"Fractal"` scene node type ([`fractal_node::FractalNodeSpec`]),
//! 28-family WGSL distance estimators (`formulas.wgsl`), built-in animated/static templates,
//! 38 gallery presets, and dual CUDA (`fractal-kernels` via `cuda-oxide`) + WGPU compute
//! engine ([`engine::WarpBroComputeEngine`]) into `sl-studio`.
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
//!
//! Any other `--option` is an error (a typo must not open the GUI from a script).

// The parameter ABI is shared with the CUDA kernels and with ofx-rs through fractal-rs; host code
// keeps addressing it as crate::params.
use fractal_core::params;

mod engine;
mod formulas;
mod fractal_node;
#[cfg(test)]
mod parity_tests;
mod presets;
mod render_bench;
mod templates;
#[cfg(test)]
mod test_gpu;

use std::sync::Arc;
use std::time::Instant;

use fractal_node::{Bulb, Formula, FractalSceneExt};
use sl_studio::color::{BT2408_SDR_WHITE_NITS, HDR_VIEW};
use sl_studio::fs_name;
use sl_studio::render::{Batch, Target, validate_size};
use sl_studio::render_service::{Frame, FrameFile, PngEncoding};
use sl_studio::render_wgpu;
use sl_studio::scene::Scene;

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

/// Optional directory positional `i` with the same `--flag` rule as [`arg`]: `--gallery --hdr`
/// writes to `default`, not to a directory named `--hdr`.
fn dir_arg<'a>(args: &'a [String], i: usize, default: &'a str) -> &'a str {
    args.get(i)
        .filter(|v| !v.starts_with("--"))
        .map_or(default, String::as_str)
}

/// Reject what a mode does not read: an unknown `--flag`, or more than `positional` plain
/// arguments after the mode, so a typo fails instead of being ignored.
fn check_args(args: &[String], positional: usize, flags: &[&str]) -> anyhow::Result<()> {
    let mode = &args[1];
    let mut plain = 0;
    for a in &args[2..] {
        if a.starts_with("--") {
            anyhow::ensure!(flags.contains(&a.as_str()), "{mode}: unknown option {a}");
        } else {
            plain += 1;
        }
    }
    anyhow::ensure!(
        plain <= positional,
        "{mode}: expected at most {positional} positional arguments, got {plain}"
    );
    Ok(())
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
    validate_size(w, h, spp)?;
    // The first sample is an untimed warm-up step: with one sample nothing is left to time.
    anyhow::ensure!(
        out.is_some() || spp >= 2,
        "--bench needs at least 2 spp (the first sample is the untimed warm-up)"
    );
    if let Some(dir) = out {
        // Fail before rendering anything, not after the first preset.
        std::fs::create_dir_all(dir)?;
    }
    let mut gpu = engine::new_gpu()?;
    // The renderer is known once a scene resolved its backend: printed after the first warm-up.
    let mut header = false;
    for mut scene in Scene::fractal_gallery().map_err(anyhow::Error::msg)? {
        if hdr {
            scene.colour.config = "ocio://studio-config-latest".into();
            scene.colour.display = "Rec.2100-PQ - Display".into();
            scene.colour.view = HDR_VIEW.into();
        }
        let mut t = Target::new(w, h);
        gpu.step(&mut t, &scene, 1, 0, spp <= 1)?; // warm-up
        if !header {
            let (backend, device) = render_bench::renderer(&t)?;
            println!("GPU: {} · {device}  ·  {w}x{h}, {spp} spp", backend.label());
            println!(
                "{:<28} {:>16} {:>9} {:>11}",
                "preset", "model", "ms/spp", "Msamples/s"
            );
            header = true;
        }
        let t0 = Instant::now();
        while !gpu.advance(&mut t, &scene, spp, 0, Batch::Fixed(8))? {}
        let secs = t0.elapsed().as_secs_f64();
        let traced = spp.saturating_sub(1).max(1) as f64;
        println!(
            "{:<28} {:>16} {:>9.2} {:>11.1}",
            scene.name,
            format!("{:?}", scene.material.model),
            secs * 1e3 / traced,
            (w * h) as f64 * traced / secs / 1e6
        );
        if let Some(dir) = out {
            let frame = Frame::snapshot(&t, 0, false, false, 100.0);
            let encoding = PngEncoding::displayed(frame.light_kind.hdr());
            let dir = std::path::Path::new(dir);
            let stem = fs_name::stem(&scene.name);
            let path = dir.join(fs_name::frame_file(&stem, None, encoding.suffix()));
            frame
                .save_png(&path, encoding, BT2408_SDR_WHITE_NITS, true)
                .map_err(anyhow::Error::msg)?;
            if display_exr {
                let exr = FrameFile::DisplayExr.suffix();
                frame
                    .save_display_exr(
                        &dir.join(fs_name::frame_file(&stem, None, exr)),
                        BT2408_SDR_WHITE_NITS,
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
    validate_size(width, height, samples)?;
    std::fs::create_dir_all(dir)?;
    let mut gpu = engine::new_gpu()?;
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
            let mut picks = vec![
                document.first,
                document.first.midpoint(document.last),
                document.last,
            ];
            // Sorted (`first <= last` is a document invariant): a one- or two-frame range
            // renders each frame once.
            picks.dedup();
            picks
        };
        for frame in frames {
            let evaluated = document
                .snapshot(f64::from(frame))
                .map_err(anyhow::Error::msg)?;
            let start = Instant::now();
            // A fixed seed keeps regression images repeatable. Every frame gets its own target,
            // so identical consecutive frames never accumulate into one another.
            let mut target = Target::new(width, height);
            while !gpu.advance(&mut target, &evaluated, samples, 0, Batch::Fixed(8))? {}
            let image = Frame::snapshot(&target, 0, false, false, 100.0);
            let encoding = PngEncoding::displayed(image.light_kind.hdr());
            image
                .save_png(
                    &output.join(fs_name::frame_file("frame", Some(frame), encoding.suffix())),
                    encoding,
                    BT2408_SDR_WHITE_NITS,
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
    let _crash_guard = crashdump::init!();
    env_logger::init();
    engine::init_warpbro();
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        // Build-time warmup, without a window or scene state: the CUDA driver JIT-caches the
        // embedded module, and the WGPU device opens with its dynamic module and post pipelines
        // (trace pipelines compile per family module on first use, then come from disk).
        // A machine without CUDA (another vendor's GPU) still warms WGPU, so a missing CUDA
        // is reported, not fatal; it is never reported as ready.
        Some("--warmup") => {
            check_args(&args, 0, &[])?;
            let started = std::time::Instant::now();
            match engine::warmup_cuda() {
                Ok(name) => println!("CUDA ready: {name} ({:?})", started.elapsed()),
                Err(error) => println!("CUDA unavailable, not warmed: {error}"),
            }
            let started = std::time::Instant::now();
            let backend = render_wgpu::warmup().map_err(anyhow::Error::msg)?;
            println!("WGPU ready on {backend:?} ({:?})", started.elapsed());
            Ok(())
        }
        Some("--world-bench") => render_bench::run(&args),
        Some("--animated-fixtures") => {
            check_args(&args, 4, &["--all-frames"])?;
            animated_fixtures(
                dir_arg(&args, 2, "animated-fixtures"),
                arg(&args, 3, 640)?,
                arg(&args, 4, 360)?,
                arg(&args, 5, 32)?,
                args.iter().any(|a| a == "--all-frames"),
            )
        }
        Some("--gallery") => {
            check_args(&args, 4, &["--hdr", "--display-exr"])?;
            headless(
                Some(dir_arg(&args, 2, "gallery")),
                arg(&args, 3, 960)?,
                arg(&args, 4, 540)?,
                arg(&args, 5, 64)?,
                args.iter().any(|a| a == "--hdr"),
                args.iter().any(|a| a == "--display-exr"),
            )
        }
        Some("--bench") => {
            check_args(&args, 3, &[])?;
            headless(
                None,
                arg(&args, 2, 960)?,
                arg(&args, 3, 540)?,
                arg(&args, 4, 32)?,
                false,
                false,
            )
        }
        Some(option) if option.starts_with("--") => anyhow::bail!(
            "unknown option {option}; modes: --gallery, --bench, --world-bench, \
             --animated-fixtures, --warmup (see the header of src/main.rs)"
        ),
        _ => {
            let bulb =
                Scene::fractal_preset(Formula::Mandelbulb(Bulb::PRESET)).map_err(anyhow::Error::msg)?;
            let swatch = bulb.clone();
            sl_studio::run_with(sl_studio::StudioConfig {
                title: "WarpBro".into(),
                initial_scene: Some(bulb),
                gallery: Some(Scene::fractal_gallery().map_err(anyhow::Error::msg)?),
                engine_factory: Some(Arc::new(engine::new_gpu)),
                material_swatch_base: Some(Arc::new(move || swatch.clone())),
                gpu_init_label: "Initializing GPU (CUDA / WGPU)…".into(),
                modules: vec![Arc::new(engine::WarpBroFractalModule)],
                // Profile root and encode-preset dir follow sl-studio's own defaults
                // (`~/.warpbro`, shared video-io presets); new fields stay opt-in here.
                ..Default::default()
            })
        }
    }
}

#[cfg(test)]
mod cli_tests {
    use super::*;

    fn argv(rest: &[&str]) -> Vec<String> {
        std::iter::once("WarpBro")
            .chain(rest.iter().copied())
            .map(String::from)
            .collect()
    }

    /// plan1 L1/L6: a flag is never taken for a directory, and a typo fails.
    #[test]
    fn cli_flags_never_become_paths_and_typos_fail() {
        let args = argv(&["--gallery", "--hdr"]);
        assert_eq!(dir_arg(&args, 2, "gallery"), "gallery");
        assert_eq!(dir_arg(&argv(&["--gallery", "out"]), 2, "gallery"), "out");
        assert!(check_args(&args, 4, &["--hdr", "--display-exr"]).is_ok());
        assert!(check_args(&argv(&["--gallery", "--hrd"]), 4, &["--hdr"]).is_err());
        assert!(check_args(&argv(&["--bench", "1", "2", "3", "4"]), 3, &[]).is_err());
    }
}
