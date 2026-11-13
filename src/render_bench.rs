//! Deterministic measurements of evaluated editor worlds, rather than standalone gallery kernels.
//!
//! Raw RGB artifacts deliberately bypass denoising, exposure and display transforms. Comparing
//! independent seeds against a high-sample reference measures convergence as well as throughput;
//! a faster image alone is insufficient evidence that a sampling change is an improvement.

use std::fmt::Write as _;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, ensure};
use serde::Serialize;
use serde_json::json;
use sl_studio::render::{Batch, Plane, Target, validate_size};
use sl_studio::render_profiles::{Backend, DirectRender, DirectShading};
use sl_studio::scene::{Material, MaterialModel, Scene};

use crate::engine::{WarpBroGpuExt, init_warpbro, new_gpu};
use crate::fractal_node::{Formula, FractalSceneExt};

const CASES: [(&str, usize, u32, bool); 10] = [
    ("ember-early", 0, 0, false),
    ("ember-mid", 0, 124, false),
    ("ember-late", 0, 249, false),
    ("chrome-mid", 1, 124, false),
    ("diffuse-mid", 0, 124, true),
    ("opal-mid", 2, 124, false),
    ("fast-metal", 0, 124, false),
    ("fast-dielectric", 0, 124, false),
    // Sampling-plan cases: most pixels miss the fractal (sky converges at once) and a close
    // camera where light reaches surfaces through narrow gaps (adaptive sampling targets).
    ("sky-wide", 0, 124, false),
    ("cavity", 1, 124, false),
];

fn has_families_flag(args: &[String]) -> bool {
    let mut idx = 2usize;
    while idx < args.len() && !args[idx].starts_with("--") {
        idx += 1;
    }
    while idx < args.len() {
        match args[idx].as_str() {
            "--families" => return true,
            "--case" | "--seed" | "--batch" | "--backend" | "--adaptive" => idx += 2,
            "--scope" | "--shade" | "--repeats" => idx += 2,
            _ => idx += 1,
        }
    }
    false
}

fn parse_bench_prefix<'a>(
    args: &'a [String],
    default_dir: &'a str,
    mut dimensions: [usize; 3],
) -> anyhow::Result<(&'a str, [usize; 3], usize)> {
    let mut position = 2usize;
    let directory = match args.get(position).filter(|s| !s.starts_with("--")) {
        Some(dir) => {
            position += 1;
            dir.as_str()
        }
        None => default_dir,
    };
    for slot in &mut dimensions {
        let Some(text) = args.get(position).filter(|s| !s.starts_with("--")) else {
            break;
        };
        *slot = text
            .parse()
            .with_context(|| format!("Invalid benchmark number: {text}"))?;
        position += 1;
    }
    Ok((directory, dimensions, position))
}

fn take_flag_value<'a>(args: &'a [String], pos: usize, flag: &str) -> anyhow::Result<&'a str> {
    args.get(pos + 1)
        .filter(|value| !value.starts_with("--"))
        .map(String::as_str)
        .with_context(|| format!("Missing value for {flag}"))
}

/// Run a bounded world benchmark. A separate warm-up target removes driver compilation and
/// preparation from the timed samples. Every measured sample uses the same seed/dimension
/// sequence as normal world rendering, and each output records its exact evaluated inputs.
pub fn run(args: &[String]) -> anyhow::Result<()> {
    init_warpbro();
    if has_families_flag(args) {
        return run_families(args);
    }
    let (directory, dimensions, mut position) =
        parse_bench_prefix(args, "world-bench", [256, 256, 32])?;
    let mut case = None;
    let mut seed = 0u32;
    let mut batch = 4u32;
    // `--adaptive THRESHOLD`: measure adaptive sampling (stops converged tiles; `samples` is the
    // budget). Off by default so estimator comparisons get exactly `samples` per pixel.
    let mut adaptive_threshold: Option<f32> = None;
    // `--backend cuda|wgpu`: which tracer the cases run on (default: the profile's Auto choice).
    let mut backend = Backend::Auto;
    while position < args.len() {
        let flag = &args[position];
        match flag.as_str() {
            "--case" => case = Some(take_flag_value(args, position, flag)?),
            "--seed" => {
                let value = take_flag_value(args, position, flag)?;
                seed = value.parse().context("Invalid benchmark seed")?;
            }
            "--batch" => {
                let value = take_flag_value(args, position, flag)?;
                batch = value.parse().context("Invalid benchmark batch")?;
            }
            "--backend" => {
                backend = match take_flag_value(args, position, flag)? {
                    "auto" => Backend::Auto,
                    "cuda" => Backend::Cuda,
                    "wgpu" => Backend::Wgpu,
                    _ => anyhow::bail!("Benchmark backend must be auto, cuda or wgpu"),
                };
            }
            "--adaptive" => {
                let value = take_flag_value(args, position, flag)?;
                adaptive_threshold = Some(value.parse().context("Invalid adaptive threshold")?);
            }
            _ => anyhow::bail!("Unknown world benchmark option: {flag}"),
        }
        position += 2;
    }
    let [width, height, samples] = dimensions;
    let samples = u32::try_from(samples).context("Invalid benchmark samples")?;
    validate_size(width, height, samples)?;
    ensure!(batch == 4 || batch == 8, "Benchmark batch must be 4 or 8");
    ensure!(
        seed <= 16_777_216,
        "Benchmark seed must be exactly representable in the f32 GPU parameter block"
    );
    ensure!(
        case.is_none_or(|name| CASES.iter().any(|c| c.0 == name)),
        "Unknown benchmark case"
    );
    if let Some(threshold) = adaptive_threshold {
        ensure!(
            threshold.is_finite() && (0.0005..=1.0).contains(&threshold),
            "Adaptive threshold must be finite and in 0.0005..=1.0"
        );
    }
    std::fs::create_dir_all(directory)?;
    let base = Path::new(directory);
    let mut gpu = new_gpu()?;
    // The renderer is reported per case (`renderer`): each scene resolves its own backend.
    println!("World benchmark · {width}x{height} · {samples} spp · seed {seed} · batch {batch}");
    // Hoist warmup/measured targets and readback scratch outside the case loop so repeated
    // benchmarks reuse GPU film allocations and warm host pages across cases.
    let mut warmup = Target::new(width, height);
    let mut target = Target::new(width, height);
    let mut batches = Vec::with_capacity(samples.div_ceil(batch) as usize);
    let mut tiled = Vec::new();
    let mut pixels = Vec::new();
    for (name, preset, frame, diffuse) in CASES {
        if case.is_some_and(|chosen| chosen != name) {
            continue;
        }
        let template = crate::templates::BUILTINS
            .get(preset)
            .context("Unknown built-in template")?;
        let authored = crate::templates::builtin_scene(preset).map_err(anyhow::Error::msg)?;
        let document = authored
            .document
            .as_ref()
            .context("Preset has no world document")?;
        let mut scene = document
            .snapshot(f64::from(frame))
            .map_err(anyhow::Error::msg)?;
        ensure!(
            scene.world_render && !scene.objects.is_empty(),
            "Benchmark must use evaluated world objects"
        );
        scene.render.denoise.enabled = false;
        scene.render.backend = backend;
        // Estimator comparisons need exactly `samples` per pixel; adaptive sampling would stop
        // converged tiles early and change both the error and the time of every run.
        scene.render.adaptive.enabled = adaptive_threshold.is_some();
        if let Some(threshold) = adaptive_threshold {
            scene.render.adaptive.noise_threshold = threshold;
        }
        scene.render.exposure_stops = 0.0;
        scene.render.saturation = 1.0;
        if diffuse {
            for object in &mut scene.objects {
                object.material = Material {
                    model: MaterialModel::StandardSurface,
                    metalness: 0.0,
                    specular: 0.0,
                    coat: 0.0,
                    sheen: 0.0,
                    emission: 0.0,
                    ..Material::default()
                };
            }
        }
        // The bulb silhouette and curved folds cover normal-to-grazing view angles;
        // these paired cases isolate the viewport GGX model from full-material controls.
        if name == "fast-metal" || name == "fast-dielectric" {
            let metallic = name == "fast-metal";
            for object in &mut scene.objects {
                object.material.model = MaterialModel::Fast;
                object.material.facing = None;
                object.material.metalness = if metallic { 1.0 } else { 0.0 };
                object.material.specular_roughness = if metallic { 0.3 } else { 0.55 };
                object.material.specular = 1.0;
                object.material.specular_ior = 1.5;
                object.material.emission = 0.0;
            }
        }
        match name {
            "sky-wide" => scene.camera.distance *= 3.0,
            "cavity" => scene.camera.distance *= 0.45,
            _ => {}
        }
        gpu.prepare_scene(&scene, width, height)?;
        warmup.requested_backend = None;
        warmup.colour_error = None;
        gpu.step(&mut warmup, &scene, 1, seed, false)?;
        ensure!(
            warmup.colour_error.is_none(),
            "Warm-up display error: {:?}",
            warmup.colour_error
        );
        // Reset and clear the measured film outside the timed steps, synchronizing async clears
        // before starting the stopwatch.
        target.requested_backend = None;
        target.colour_error = None;
        gpu.prepare(&mut target, &scene)?;
        gpu.step(&mut target, &scene, 0, seed, false)?;
        ensure!(target.samples == 0, "Preparation consumed traced samples");
        batches.clear();
        let start = Instant::now();
        loop {
            let complete = gpu.advance(&mut target, &scene, samples, seed, Batch::Fixed(batch))?;
            batches.push((target.last_spp, target.last_ms));
            if complete {
                break;
            }
        }
        let elapsed = start.elapsed().as_secs_f64();
        let safe_elapsed = elapsed.max(1.0e-12);
        ensure!(
            target.colour_error.is_none(),
            "Display error: {:?}",
            target.colour_error
        );
        let readback_start = Instant::now();
        gpu.read_film(&target, Plane::Accum, true, &mut tiled, &mut pixels)?;
        let readback_seconds = readback_start.elapsed().as_secs_f64();
        ensure!(
            pixels.iter().all(|p| p[..3].iter().all(|c| c.is_finite())),
            "Nonfinite raw radiance in {name}"
        );
        let stem = format!("{name}.seed-{seed}.spp-{samples}");
        let raw_name = format!("{stem}.rgb.f32");
        let raw_path = base.join(&raw_name);
        write_raw_rgb(&raw_path, &pixels)?;
        let shot =
            sl_studio::render_service::Frame::snapshot(&target, 0, false, false, 100.0);
        let encoding =
            sl_studio::render_service::PngEncoding::displayed(shot.light_kind.hdr());
        shot.save_png(
            &base.join(sl_studio::fs_name::frame_file(
                &stem,
                None,
                encoding.suffix(),
            )),
            encoding,
            sl_studio::color::BT2408_SDR_WHITE_NITS,
            true,
        )
        .map_err(anyhow::Error::msg)?;
        let pixel_count = (width * height) as f64;
        let pixel_samples = traced_pixel_samples(&tiled, width, height);
        let mean_spp = pixel_samples / pixel_count;
        let ms_per_sample = elapsed * 1000.0 / mean_spp.max(1.0e-12);
        let msamples_per_s = pixel_samples / safe_elapsed / 1.0e6;
        let (used_backend, device) = renderer(&target)?;
        let metadata = json!({
            "schema": 1,
            "case": name, "preset": template.name, "frame": frame,
            "backend": used_backend.label(), "gpu": device, "width": width, "height": height, "samples": samples,
            "seed": seed, "batch": batch, "world_render": true,
            "adaptive_threshold": adaptive_threshold, "samples_taken": target.samples,
            "pixel_samples_taken": pixel_samples, "mean_spp": mean_spp,
            "active_tiles": [target.active_tiles.0, target.active_tiles.1],
            "denoise_enabled": false, "exposure_multiplier": 1.0, "saturation": 1.0,
            "elapsed_seconds": elapsed, "raw_readback_seconds": readback_seconds,
            "ms_per_sample": ms_per_sample,
            "million_samples_per_second": msamples_per_s,
            "timing_scope": "Gpu::advance batches, display/readback on the completing batch only; excludes warmup, target allocation, artifact writes and final raw readback",
            "raw": {"file": raw_name,
                "format": "little-endian f32 RGB, row-major top-to-bottom, scene-linear ACEScg, no header"},
            "batches": batches.iter().map(|(count, ms)| json!({"samples": count, "step_ms": ms})).collect::<Vec<_>>(),
            "authoring_world": document,
            "evaluated_scene": &scene,
            "evaluated_objects": &scene.objects,
            "evaluated_object_matrices": scene.objects.iter().map(|o| o.object_world).collect::<Vec<_>>(),
            "camera_reference": scene.camera_reference,
            "evaluated_lights": &scene.lights,
            "diffuse_override": diffuse
        });
        write_json(&base.join(format!("{stem}.json")), &metadata)?;
        println!(
            "{name} ({} · {device}): {elapsed:.3}s · {ms_per_sample:.2} ms/spp · {msamples_per_s:.2} Msamples/s",
            used_backend.label()
        );
    }
    Ok(())
}

/// The backend and device that rendered `target` (`Target::device`). Read from the target,
/// not the engine: the scene picks the backend, so one engine renders on several devices.
pub fn renderer(target: &Target) -> anyhow::Result<(Backend, &str)> {
    match (&target.film, target.device.as_deref()) {
        (Some((backend, _)), Some(device)) => Ok((*backend, device)),
        _ => anyhow::bail!("The target was never rendered: no backend or device to report"),
    }
}

/// Build a controlled standalone, homogeneous editor world or genuinely mixed-family world.
/// Family benchmarks share geometry and materials across backends; only the engine differs.
fn family_scene(code: u32, scope: &str, shade: &str, samples: u32) -> anyhow::Result<Scene> {
    let mut scene =
        Scene::fractal_preset(Formula::PRESETS[code as usize]).map_err(anyhow::Error::msg)?;
    scene.render.denoise.enabled = false;
    scene.render.adaptive.enabled = false;
    scene.render.max_bounces = 2;
    scene.colour.on = false;
    scene.material = Material {
        model: if shade == "fast" {
            MaterialModel::Fast
        } else {
            MaterialModel::StandardSurface
        },
        ..Material::default()
    };
    if shade == "direct" {
        scene.direct = Some(DirectRender {
            shading: DirectShading::default(),
            grid: (f64::from(samples).sqrt().ceil() as u32).clamp(1, DirectShading::MAX_GRID),
        });
    }
    if scope != "single" {
        let radius = scene.framing_radius();
        let mut first = scene.clone();
        first.direct = None;
        let mut matrix = glam::Mat4::from_scale(glam::Vec3::new(0.65, 0.7, 0.6));
        matrix.y_axis += matrix.x_axis * 0.15;
        matrix.w_axis.x = -0.38 * radius;
        first.object_world = Some(matrix.to_cols_array_2d());
        let mut second = first.clone();
        matrix.w_axis.x = 0.38 * radius;
        second.object_world = Some(matrix.to_cols_array_2d());
        if scope == "mixed" {
            let partner = if code == 0 { 1 } else { 0 };
            second
                .edit_fractal(|f| f.formula = Formula::PRESETS[partner])
                .map_err(anyhow::Error::msg)?;
        }
        scene.world_render = true;
        scene.objects = vec![first, second];
        scene.lights = vec![scene.sun];
    }
    Ok(scene)
}

fn traced_pixel_samples(tiled: &[[f32; 4]], width: usize, height: usize) -> f64 {
    let tiles_per_row = width.div_ceil(8);
    let mut total = 0.0f64;
    for y in 0..height {
        let tile_row = (y / 4) * tiles_per_row;
        let in_tile_y = (y % 4) * 8;
        for tx in 0..tiles_per_row {
            let cols = (width - tx * 8).min(8);
            let base = (tile_row + tx) * 32 + in_tile_y;
            if let Some(slice) = tiled.get(base..base + cols) {
                for pixel in slice {
                    total += f64::from(pixel[3].max(0.0));
                }
            }
        }
    }
    total
}

fn write_raw_rgb(path: &Path, pixels: &[[f32; 4]]) -> anyhow::Result<()> {
    let mut writer = BufWriter::with_capacity(64 * 1024, File::create(path)?);
    for pixel in pixels {
        let mut rgb = [0u8; 12];
        rgb[0..4].copy_from_slice(&pixel[0].to_le_bytes());
        rgb[4..8].copy_from_slice(&pixel[1].to_le_bytes());
        rgb[8..12].copy_from_slice(&pixel[2].to_le_bytes());
        writer.write_all(&rgb)?;
    }
    writer.flush()?;
    Ok(())
}

fn write_json(path: &Path, value: &impl Serialize) -> anyhow::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    serde_json::to_writer_pretty(&mut writer, value)?;
    writer.flush()?;
    Ok(())
}

/// Explicit all-family benchmark mode. Backend order alternates each round to expose thermal
/// or clock drift; five or more odd repetitions produce an observed median rather than a best run.
/// Allocation, preparation, warm-up, raw readback and artifact I/O stay outside timed steps.
fn run_families(args: &[String]) -> anyhow::Result<()> {
    let (directory, dimensions, mut position) =
        parse_bench_prefix(args, "family-bench", [256, 192, 16])?;
    let mut families: Option<Vec<u32>> = None;
    let mut scopes = vec!["world"];
    let mut shades = vec!["fast"];
    let mut backends = vec![Backend::Cuda, Backend::Wgpu];
    let mut repeats = 5usize;
    let mut seed = 17u32;
    let mut batch = 4u32;
    while position < args.len() {
        let flag = &args[position];
        match flag.as_str() {
            "--families" => {
                if let Some(value) = args.get(position + 1).filter(|v| !v.starts_with("--")) {
                    position += 2;
                    families = if value == "all" {
                        Some((0..Formula::COUNT).collect())
                    } else {
                        let parsed = value
                            .split(',')
                            .map(|code| code.parse().context("Invalid comma-separated family code"))
                            .collect::<anyhow::Result<Vec<_>>>()?;
                        Some(parsed)
                    };
                } else {
                    position += 1;
                    families = Some((0..Formula::COUNT).collect());
                }
                continue;
            }
            "--scope" => {
                scopes = match take_flag_value(args, position, flag)? {
                    "all" => vec!["single", "world", "mixed"],
                    "single" => vec!["single"],
                    "world" => vec!["world"],
                    "mixed" => vec!["mixed"],
                    _ => anyhow::bail!("Family scope must be single, world, mixed or all"),
                };
            }
            "--shade" => {
                shades = match take_flag_value(args, position, flag)? {
                    "all" => vec!["fast", "full", "direct"],
                    "fast" => vec!["fast"],
                    "full" => vec!["full"],
                    "direct" => vec!["direct"],
                    _ => anyhow::bail!("Family shade must be fast, full, direct or all"),
                };
            }
            "--backend" => {
                backends = match take_flag_value(args, position, flag)? {
                    "both" => vec![Backend::Cuda, Backend::Wgpu],
                    "cuda" => vec![Backend::Cuda],
                    "wgpu" => vec![Backend::Wgpu],
                    _ => anyhow::bail!("Family backend must be cuda, wgpu or both"),
                };
            }
            "--repeats" => {
                let value = take_flag_value(args, position, flag)?;
                repeats = value.parse().context("Invalid family repetition count")?;
            }
            "--seed" => {
                let value = take_flag_value(args, position, flag)?;
                seed = value.parse().context("Invalid family seed")?;
            }
            "--batch" => {
                let value = take_flag_value(args, position, flag)?;
                batch = value.parse().context("Invalid family batch")?;
            }
            _ => anyhow::bail!("Unknown family benchmark option: {flag}"),
        }
        position += 2;
    }
    let mut families = families.unwrap_or_else(|| (0..Formula::COUNT).collect());
    let [width, height, samples] = dimensions;
    let samples = u32::try_from(samples).context("Invalid family sample count")?;
    validate_size(width, height, samples)?;
    ensure!(!families.is_empty(), "Select at least one family");
    ensure!(
        families.iter().all(|&family| family < Formula::COUNT),
        "Family code must be in 0..28"
    );
    families.sort_unstable();
    families.dedup();
    ensure!(
        (5..=101).contains(&repeats) && !repeats.is_multiple_of(2),
        "Family repetitions must be odd and between 5 and 101"
    );
    ensure!(batch == 4 || batch == 8, "Family batch must be 4 or 8");
    ensure!(
        seed <= 16_777_216,
        "Family seed must fit the f32 parameter block exactly"
    );
    ensure!(
        !shades.contains(&"direct") || samples <= 256,
        "Direct benchmarks support at most 256 exact strata samples"
    );
    std::fs::create_dir_all(directory)?;
    let directory = Path::new(directory);
    let mut gpu = new_gpu()?;
    let total_cases = families
        .len()
        .saturating_mul(scopes.len())
        .saturating_mul(shades.len());
    let mut summaries = Vec::with_capacity(total_cases);
    // Reuse one progressive Target per backend and shared readback/metric buffers across all
    // warmup rounds, measurement repetitions, and families so GPU film allocations and host page
    // faults never churn inside the benchmark loops.
    let mut targets: Vec<Target> = (0..backends.len())
        .map(|_| Target::new(width, height))
        .collect();
    let mut backend_scenes: Vec<Scene> = Vec::with_capacity(backends.len());
    let mut times = vec![Vec::<f64>::with_capacity(repeats); backends.len()];
    let mut sorted = Vec::<f64>::with_capacity(repeats);
    let mut artifacts = vec![String::new(); backends.len()];
    let mut first_pixels = vec![Vec::<[f32; 4]>::new(); backends.len()];
    let mut tiled = Vec::<[f32; 4]>::new();
    let mut cuda_resources = vec![serde_json::Value::Null; backends.len()];
    let mut warmup_samples = vec![0u64; backends.len()];
    let mut order: Vec<Vec<&'static str>> = Vec::with_capacity(repeats);
    for code in families {
        for &scope in &scopes {
            for &shade in &shades {
                let scene = family_scene(code, scope, shade, samples)?;
                let stem = format!("family-{code:02}-{scope}-{shade}");
                backend_scenes.clear();
                for &backend in &backends {
                    let mut backend_scene = scene.clone();
                    backend_scene.render.backend = backend;
                    backend_scenes.push(backend_scene);
                }
                for slot in &mut times {
                    slot.clear();
                }
                for slot in &mut artifacts {
                    slot.clear();
                }
                cuda_resources.fill(serde_json::Value::Null);
                warmup_samples.fill(0);
                order.clear();
                for (index, &backend) in backends.iter().enumerate() {
                    let warm_scene = &backend_scenes[index];
                    if backend == Backend::Cuda {
                        cuda_resources[index] =
                            gpu.cuda_kernel_resources(warm_scene, width, height)?;
                    }
                    gpu.prepare_scene(warm_scene, width, height)?;
                    let warm_target = &mut targets[index];
                    warm_target.requested_backend = None;
                    warm_target.colour_error = None;
                    while !gpu.advance(warm_target, warm_scene, 1, seed, Batch::Fixed(1))? {}
                    ensure!(
                        warm_target.samples == 1,
                        "Warmup did not trace exactly one sample"
                    );
                    ensure!(warm_target.colour_error.is_none(), "Warmup display failed");
                }
                // Pipeline compilation can leave the GPU idle for a long time. Warm the
                // actual sample workload after every backend is ready, before timed rounds.
                let warmup_started = Instant::now();
                while warmup_started.elapsed().as_secs_f64() < 2.0 {
                    for index in 0..backends.len() {
                        let warm_scene = &backend_scenes[index];
                        let warm_target = &mut targets[index];
                        warm_target.requested_backend = None;
                        warm_target.colour_error = None;
                        while !gpu.advance(
                            warm_target,
                            warm_scene,
                            samples,
                            seed,
                            Batch::Fixed(batch),
                        )? {}
                        ensure!(
                            warm_target.samples == samples,
                            "Workload warmup sample count"
                        );
                        ensure!(
                            warm_target.colour_error.is_none(),
                            "Workload warmup display failed"
                        );
                        warmup_samples[index] += u64::from(warm_target.samples);
                    }
                }
                let warmup_seconds = warmup_started.elapsed().as_secs_f64();
                for round in 0..repeats {
                    let mut round_labels = Vec::with_capacity(backends.len());
                    for step in 0..backends.len() {
                        let index = if round.is_multiple_of(2) {
                            step
                        } else {
                            backends.len() - 1 - step
                        };
                        let backend = backends[index];
                        round_labels.push(backend.label());
                        let measured_scene = &backend_scenes[index];
                        let target = &mut targets[index];
                        target.requested_backend = None;
                        target.colour_error = None;
                        gpu.prepare(target, measured_scene)?;
                        // Presenting the empty film completes asynchronous clears/uploads before
                        // timing begins; it does not trace or consume any sample dimensions.
                        gpu.step(target, measured_scene, 0, seed, false)?;
                        ensure!(target.samples == 0, "Preparation consumed traced samples");
                        let started = Instant::now();
                        while !gpu.advance(
                            target,
                            measured_scene,
                            samples,
                            seed,
                            Batch::Fixed(batch),
                        )? {}
                        let elapsed = started.elapsed().as_secs_f64();
                        ensure!(
                            target.samples == samples,
                            "Measured target has the wrong sample count"
                        );
                        ensure!(
                            target.colour_error.is_none() && target.denoise.error.is_none(),
                            "Measured target reported a display/denoise error"
                        );
                        times[index].push(elapsed);
                        if round == 0 {
                            let pixels = &mut first_pixels[index];
                            gpu.read_film(target, Plane::Accum, true, &mut tiled, pixels)?;
                            ensure!(
                                pixels
                                    .iter()
                                    .all(|pixel| pixel[..3].iter().all(|c| c.is_finite())),
                                "Nonfinite raw output: family {code}, {scope}, {shade}, {backend:?}"
                            );
                            let label = backend.label().to_ascii_lowercase();
                            artifacts[index].clear();
                            write!(&mut artifacts[index], "{stem}-{label}.rgb.f32")?;
                            write_raw_rgb(&directory.join(&artifacts[index]), pixels)?;
                        }
                    }
                    order.push(round_labels);
                }
                let results: Vec<_> = backends
                    .iter()
                    .enumerate()
                    .map(|(index, backend)| {
                        sorted.clear();
                        sorted.extend_from_slice(&times[index]);
                        sorted.sort_by(f64::total_cmp);
                        let median = sorted[repeats / 2];
                        let safe_median = median.max(1.0e-12);
                        json!({
                            "backend": backend.label(), "seconds": times[index],
                            "median_seconds": median, "min_seconds": sorted[0],
                            "max_seconds": sorted[repeats - 1],
                            "median_ms_per_sample": median * 1000.0 / f64::from(samples),
                            "median_million_samples_per_second":
                                (width * height) as f64 * f64::from(samples) / safe_median / 1.0e6,
                            "gpu": targets[index].device.as_deref(),
                            "raw_file": artifacts[index],
                            "cuda_resources": cuda_resources[index],
                        })
                    })
                    .collect();
                let comparison = if first_pixels.len() == 2 {
                    let mut squared_error = 0.0f64;
                    let mut reference_energy = 0.0f64;
                    let mut max_absolute = 0.0f64;
                    for (a, b) in first_pixels[0].iter().zip(&first_pixels[1]) {
                        for channel in 0..3 {
                            let difference = f64::from(a[channel]) - f64::from(b[channel]);
                            squared_error += difference.powi(2);
                            reference_energy += f64::from(b[channel]).powi(2);
                            max_absolute = max_absolute.max(difference.abs());
                        }
                    }
                    json!({
                        "max_absolute_rgb": max_absolute,
                        "relative_l2_rgb": (squared_error / reference_energy.max(1.0e-30)).sqrt(),
                        "note": "Multibounce radiance diagnostic; acceptance uses the numerical/guide parity tests."
                    })
                } else {
                    serde_json::Value::Null
                };
                let report = json!({
                    "schema": 1, "case": stem, "family": code, "formula": scene.summary_label(),
                    "scope": scope, "shade": shade, "width": width, "height": height,
                    "samples": samples, "seed": seed, "batch": batch, "repeats": repeats,
                    "round_backend_order": order,
                    "workload_warmup_seconds": warmup_seconds,
                    "workload_warmup_samples_per_backend": warmup_samples,
                    "timing_scope": "Synchronized Gpu::advance including final display; excludes JIT, warmup, allocation, preparation, raw readback and artifact I/O. Not isolated kernel-event timing.",
                    "results": results, "radiance_comparison": comparison,
                    "raw_format": "little-endian f32 RGB, row-major, scene-linear ACEScg, no header",
                    "scene": scene,
                });
                write_json(&directory.join(format!("{stem}.json")), &report)?;
                println!(
                    "{}",
                    json!({
                        "case": stem, "results": report["results"],
                        "radiance_comparison": report["radiance_comparison"],
                    })
                );
                summaries.push(report);
            }
        }
    }
    write_json(&directory.join("family-summary.json"), &summaries)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct TempDirGuard(Option<PathBuf>);

    impl Drop for TempDirGuard {
        fn drop(&mut self) {
            if let Some(directory) = self.0.take()
                && directory.exists()
                && let Err(err) = std::fs::remove_dir_all(&directory)
            {
                if std::thread::panicking() {
                    eprintln!("Failed to remove {}: {err}", directory.display());
                } else {
                    panic!("Failed to remove {}: {err}", directory.display());
                }
            }
        }
    }

    #[test]
    #[ignore = "runs the GPU world benchmark; optional WARP_BRO_WORLD_BENCH_ARGS JSON argv"]
    fn gpu_paired_world_de_benchmark() {
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        let _gpu_test = crate::test_gpu::lock();
        let mut cleanup = TempDirGuard(None);
        let args = match std::env::var("WARP_BRO_WORLD_BENCH_ARGS") {
            Ok(json) => serde_json::from_str::<Vec<String>>(&json).expect("benchmark JSON argv"),
            Err(_) => {
                let pid = std::process::id();
                let seq = NEXT_ID.fetch_add(1, Ordering::Relaxed);
                let name = format!("warpbro-world-de-bench-{pid}-{seq}");
                let directory = std::env::temp_dir().join(name);
                let arg = directory.to_string_lossy().into_owned();
                cleanup.0 = Some(PathBuf::from(&arg));
                vec![
                    "WarpBro".into(),
                    "--world-bench".into(),
                    arg,
                    "128".into(),
                    "128".into(),
                    "8".into(),
                    "--case".into(),
                    "fast-metal".into(),
                    "--seed".into(),
                    "17".into(),
                ]
            }
        };
        super::run(&args).expect("world DE benchmark");
        drop(cleanup);
    }
}
