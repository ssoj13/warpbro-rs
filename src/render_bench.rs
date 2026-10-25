//! Deterministic measurements of evaluated editor worlds, rather than standalone gallery kernels.
//!
//! Raw RGB artifacts deliberately bypass denoising, exposure and display transforms. Comparing
//! independent seeds against a high-sample reference measures convergence as well as throughput;
//! a faster image alone is insufficient evidence that a sampling change is an improvement.
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, ensure};
use serde_json::json;

use crate::render::{Batch, Gpu, Target};
use crate::scene::{Material, MaterialModel};

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

/// Run a bounded world benchmark. A separate warm-up target removes driver compilation and
/// preparation from the timed samples. Every measured sample uses the same seed/dimension
/// sequence as normal world rendering, and each output records its exact evaluated inputs.
pub fn run(args: &[String]) -> anyhow::Result<()> {
    if args.iter().any(|argument| argument == "--families") {
        return run_families(args);
    }
    let directory = args
        .get(2)
        .filter(|s| !s.starts_with("--"))
        .map(String::as_str)
        .unwrap_or("world-bench");
    let mut dimensions = [256usize, 256, 32];
    let mut position = if args.get(2).is_some_and(|s| !s.starts_with("--")) {
        3usize
    } else {
        2usize
    };
    for value in &mut dimensions {
        if let Some(text) = args.get(position).filter(|s| !s.starts_with("--")) {
            *value = text
                .parse()
                .with_context(|| format!("Invalid benchmark number: {text}"))?;
            position += 1;
        } else {
            break;
        }
    }
    let mut case = None;
    let mut seed = 0u32;
    let mut batch = 4u32;
    // `--adaptive THRESHOLD`: measure adaptive sampling (stops converged tiles; `samples` is the
    // budget). Off by default so estimator comparisons get exactly `samples` per pixel.
    let mut adaptive_threshold: Option<f32> = None;
    // `--backend cuda|wgpu`: which tracer the cases run on (default: the profile's Auto choice).
    let mut backend = crate::render_profiles::Backend::Auto;
    while position < args.len() {
        let flag = &args[position];
        let value = args
            .get(position + 1)
            .with_context(|| format!("Missing value for {flag}"))?;
        match flag.as_str() {
            "--case" => case = Some(value.as_str()),
            "--seed" => seed = value.parse().context("Invalid benchmark seed")?,
            "--batch" => batch = value.parse().context("Invalid benchmark batch")?,
            "--backend" => {
                backend = match value.as_str() {
                    "auto" => crate::render_profiles::Backend::Auto,
                    "cuda" => crate::render_profiles::Backend::Cuda,
                    "wgpu" => crate::render_profiles::Backend::Wgpu,
                    _ => anyhow::bail!("Benchmark backend must be auto, cuda or wgpu"),
                }
            }
            "--adaptive" => {
                adaptive_threshold = Some(value.parse().context("Invalid adaptive threshold")?)
            }
            _ => anyhow::bail!("Unknown world benchmark option: {flag}"),
        }
        position += 2;
    }
    let [width, height, samples] = dimensions;
    let samples = u32::try_from(samples).context("Invalid benchmark samples")?;
    crate::render::validate_size(width, height, samples)?;
    ensure!(batch == 4 || batch == 8, "Benchmark batch must be 4 or 8");
    ensure!(
        seed <= 16_777_216,
        "Benchmark seed must be exactly representable in the f32 GPU parameter block"
    );
    ensure!(
        case.is_none_or(|name| CASES.iter().any(|c| c.0 == name)),
        "Unknown benchmark case"
    );
    std::fs::create_dir_all(directory)?;
    let mut gpu = Gpu::new()?;
    println!(
        "World benchmark GPU: {} · {width}x{height} · {samples} spp · seed {seed} · batch {batch}",
        gpu.name
    );
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
        {
            let mut warmup = Target::new(width, height);
            gpu.step(&mut warmup, &scene, 1, seed, false)?;
            ensure!(
                warmup.colour_error.is_none(),
                "Warm-up display error: {:?}",
                warmup.colour_error
            );
        }
        // The film is allocated outside the timed steps.
        let mut target = Target::new(width, height);
        gpu.prepare(&mut target, &scene)?;
        let mut batches = Vec::with_capacity(samples.div_ceil(batch) as usize);
        let start = Instant::now();
        loop {
            let complete = gpu.advance(&mut target, &scene, samples, seed, Batch::Fixed(batch))?;
            batches.push((target.last_spp, target.last_ms));
            if complete {
                break;
            }
        }
        let elapsed = start.elapsed().as_secs_f64();
        ensure!(
            target.colour_error.is_none(),
            "Display error: {:?}",
            target.colour_error
        );
        let readback_start = Instant::now();
        let pixels = gpu.raw_scene_linear(&target)?;
        let readback_seconds = readback_start.elapsed().as_secs_f64();
        ensure!(
            pixels.iter().all(|p| p[..3].iter().all(|c| c.is_finite())),
            "Nonfinite raw radiance in {name}"
        );
        let stem = format!("{name}.seed-{seed}.spp-{samples}");
        let base = Path::new(directory);
        let raw_name = format!("{stem}.rgb.f32");
        let raw_path = base.join(&raw_name);
        let mut raw = BufWriter::new(File::create(&raw_path)?);
        for pixel in &pixels {
            for channel in &pixel[..3] {
                raw.write_all(&channel.to_le_bytes())?;
            }
        }
        raw.flush()?;
        let shot = crate::render_service::Frame::snapshot(&target, 0, false, false, 100.0);
        let encoding = crate::render_service::PngEncoding::displayed(shot.light_kind.hdr());
        shot.save_png(
            &base.join(crate::fs_name::frame_file(&stem, None, encoding.suffix())),
            encoding,
            crate::color::BT2408_SDR_WHITE_NITS,
            true,
        )
        .map_err(anyhow::Error::msg)?;
        let metadata = json!({
            "schema": 1,
            "case": name, "preset": template.name, "frame": frame,
            "gpu": &gpu.name, "width": width, "height": height, "samples": samples,
            "seed": seed, "batch": batch, "world_render": true,
            "adaptive_threshold": adaptive_threshold, "samples_taken": target.samples,
            "active_tiles": [target.active_tiles.0, target.active_tiles.1],
            "denoise_enabled": false, "exposure_multiplier": 1.0, "saturation": 1.0,
            "elapsed_seconds": elapsed, "raw_readback_seconds": readback_seconds,
            "ms_per_sample": elapsed * 1000.0 / f64::from(samples),
            "million_samples_per_second": (width * height) as f64 * f64::from(samples) / elapsed / 1e6,
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
        let mut report = BufWriter::new(File::create(base.join(format!("{stem}.json")))?);
        serde_json::to_writer_pretty(&mut report, &metadata)?;
        report.flush()?;
        println!(
            "{name}: {:.3}s · {:.2} ms/spp · {:.2} Msamples/s",
            elapsed,
            elapsed * 1000.0 / f64::from(samples),
            (width * height) as f64 * f64::from(samples) / elapsed / 1e6
        );
    }
    Ok(())
}

/// Build a controlled standalone, homogeneous editor world or genuinely mixed-family world.
/// Family benchmarks share geometry and materials between backends; only the requested engine differs.
fn family_scene(code: u32, scope: &str, shade: &str, samples: u32) -> crate::scene::Scene {
    use crate::render_profiles::{DirectRender, DirectShading};
    use crate::scene::{Formula, Scene};
    let mut scene = Scene::preset(Formula::PRESETS[code as usize]);
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
        let radius = scene.formula.framing_radius();
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
            second.formula = Formula::PRESETS[0];
        }
        scene.world_render = true;
        scene.objects = vec![first, second];
        scene.lights = vec![scene.sun];
    }
    scene
}

fn write_raw_rgb(path: &Path, pixels: &[[f32; 4]]) -> anyhow::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    for pixel in pixels {
        for channel in &pixel[..3] {
            writer.write_all(&channel.to_le_bytes())?;
        }
    }
    writer.flush()?;
    Ok(())
}

/// Explicit all-family benchmark mode. Backend order alternates each round to expose thermal
/// or clock drift; five or more odd repetitions produce an observed median rather than a best run.
/// Allocation, preparation, warm-up, final raw readback and artifact I/O are outside the timed steps.
fn run_families(args: &[String]) -> anyhow::Result<()> {
    use crate::render_profiles::Backend;
    let directory = args
        .get(2)
        .filter(|value| !value.starts_with("--"))
        .map(String::as_str)
        .unwrap_or("family-bench");
    let mut position = if args.get(2).is_some_and(|s| !s.starts_with("--")) {
        3usize
    } else {
        2usize
    };
    let mut dimensions = [256usize, 192, 16];
    for dimension in &mut dimensions {
        if let Some(value) = args.get(position).filter(|s| !s.starts_with("--")) {
            *dimension = value
                .parse()
                .context("Invalid family benchmark dimension/sample count")?;
            position += 1;
        }
    }
    let mut families: Vec<u32> = (8..crate::scene::Formula::COUNT).collect();
    let mut scopes = vec!["world"];
    let mut shades = vec!["fast"];
    let mut backends = vec![Backend::Cuda, Backend::Wgpu];
    let mut repeats = 5usize;
    let mut seed = 17u32;
    let mut batch = 4u32;
    while position < args.len() {
        let flag = &args[position];
        let value = args
            .get(position + 1)
            .with_context(|| format!("Missing value for {flag}"))?;
        match flag.as_str() {
            "--families" => {
                if value != "all" {
                    families = value
                        .split(',')
                        .map(|code| code.parse().context("Invalid comma-separated family code"))
                        .collect::<anyhow::Result<Vec<_>>>()?;
                }
            }
            "--scope" => {
                scopes = match value.as_str() {
                    "all" => vec!["single", "world", "mixed"],
                    "single" => vec!["single"],
                    "world" => vec!["world"],
                    "mixed" => vec!["mixed"],
                    _ => anyhow::bail!("Family scope must be single, world, mixed or all"),
                };
            }
            "--shade" => {
                shades = match value.as_str() {
                    "all" => vec!["fast", "full", "direct"],
                    "fast" => vec!["fast"],
                    "full" => vec!["full"],
                    "direct" => vec!["direct"],
                    _ => anyhow::bail!("Family shade must be fast, full, direct or all"),
                };
            }
            "--backend" => {
                backends = match value.as_str() {
                    "both" => vec![Backend::Cuda, Backend::Wgpu],
                    "cuda" => vec![Backend::Cuda],
                    "wgpu" => vec![Backend::Wgpu],
                    _ => anyhow::bail!("Family backend must be cuda, wgpu or both"),
                };
            }
            "--repeats" => repeats = value.parse().context("Invalid family repetition count")?,
            "--seed" => seed = value.parse().context("Invalid family seed")?,
            "--batch" => batch = value.parse().context("Invalid family batch")?,
            _ => anyhow::bail!("Unknown family benchmark option: {flag}"),
        }
        position += 2;
    }
    let [width, height, samples] = dimensions;
    let samples = u32::try_from(samples).context("Invalid family sample count")?;
    crate::render::validate_size(width, height, samples)?;
    ensure!(!families.is_empty(), "Select at least one family");
    ensure!(
        families
            .iter()
            .all(|&family| family < crate::scene::Formula::COUNT),
        "Family code must be in 0..28"
    );
    families.sort_unstable();
    families.dedup();
    ensure!(
        (5..=101).contains(&repeats) && repeats % 2 == 1,
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
    let mut gpu = Gpu::new()?;
    let mut summaries = Vec::new();
    for code in families {
        for &scope in &scopes {
            for &shade in &shades {
                let scene = family_scene(code, scope, shade, samples);
                let stem = format!("family-{code:02}-{scope}-{shade}");
                let mut times = vec![Vec::<f64>::with_capacity(repeats); backends.len()];
                let mut artifacts = vec![String::new(); backends.len()];
                let mut first_pixels = vec![Vec::new(); backends.len()];
                let mut cuda_resources = vec![serde_json::Value::Null; backends.len()];
                let mut order = Vec::with_capacity(repeats);
                for (index, &backend) in backends.iter().enumerate() {
                    let mut warm_scene = scene.clone();
                    warm_scene.render.backend = backend;
                    if backend == Backend::Cuda {
                        cuda_resources[index] =
                            gpu.cuda_kernel_resources(&warm_scene, width, height)?;
                    }
                    gpu.prepare_scene(&warm_scene, width, height)?;
                    let mut warm_target = Target::new(width, height);
                    while !gpu.advance(&mut warm_target, &warm_scene, 1, seed, Batch::Fixed(1))? {}
                    ensure!(
                        warm_target.samples == 1,
                        "Warmup did not trace exactly one sample"
                    );
                    ensure!(warm_target.colour_error.is_none(), "Warmup display failed");
                }
                for round in 0..repeats {
                    let indices: Vec<_> = if round % 2 == 0 {
                        (0..backends.len()).collect()
                    } else {
                        (0..backends.len()).rev().collect()
                    };
                    order.push(
                        indices
                            .iter()
                            .map(|&i| backends[i].label())
                            .collect::<Vec<_>>(),
                    );
                    for index in indices {
                        let backend = backends[index];
                        let mut measured_scene = scene.clone();
                        measured_scene.render.backend = backend;
                        let mut target = Target::new(width, height);
                        gpu.prepare(&mut target, &measured_scene)?;
                        // Presenting the empty film completes asynchronous clears/uploads before
                        // timing begins; it does not trace or consume any sample dimensions.
                        gpu.step(&mut target, &measured_scene, 0, seed, false)?;
                        ensure!(target.samples == 0, "Preparation consumed traced samples");
                        let started = Instant::now();
                        while !gpu.advance(
                            &mut target,
                            &measured_scene,
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
                            let pixels = gpu.raw_scene_linear(&target)?;
                            ensure!(
                                pixels
                                    .iter()
                                    .all(|pixel| pixel[..3].iter().all(|c| c.is_finite())),
                                "Nonfinite raw output: family {code}, {scope}, {shade}, {backend:?}"
                            );
                            artifacts[index] =
                                format!("{stem}-{}.rgb.f32", backend.label().to_ascii_lowercase());
                            write_raw_rgb(&directory.join(&artifacts[index]), &pixels)?;
                            first_pixels[index] = pixels;
                        }
                    }
                }
                let results: Vec<_> = backends
                    .iter()
                    .enumerate()
                    .map(|(index, backend)| {
                        let mut sorted = times[index].clone();
                        sorted.sort_by(f64::total_cmp);
                        let median = sorted[repeats / 2];
                        json!({
                            "backend": backend.label(), "seconds": times[index],
                            "median_seconds": median, "min_seconds": sorted[0],
                            "max_seconds": sorted[repeats - 1],
                            "median_ms_per_sample": median * 1000.0 / f64::from(samples),
                            "median_million_samples_per_second":
                                (width * height) as f64 * f64::from(samples) / median / 1.0e6,
                            "raw_file": artifacts[index],
                            "cuda_resources": cuda_resources[index],
                        })
                    })
                    .collect();
                let comparison = if first_pixels.len() == 2 {
                    let mut squared_error = 0.0f64;
                    let mut reference_energy = 0.0f64;
                    let mut max_absolute = 0.0f32;
                    for (a, b) in first_pixels[0].iter().zip(&first_pixels[1]) {
                        for channel in 0..3 {
                            let difference = a[channel] - b[channel];
                            squared_error += f64::from(difference).powi(2);
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
                    "schema": 1, "case": stem, "family": code, "formula": scene.formula.name(),
                    "scope": scope, "shade": shade, "width": width, "height": height,
                    "samples": samples, "seed": seed, "batch": batch, "repeats": repeats,
                    "gpu": gpu.name, "round_backend_order": order,
                    "timing_scope": "Synchronized Gpu::advance including final display; excludes JIT, warmup, allocation, preparation, raw readback and artifact I/O. Not isolated kernel-event timing.",
                    "results": results, "radiance_comparison": comparison,
                    "raw_format": "little-endian f32 RGB, row-major, scene-linear ACEScg, no header",
                    "scene": scene,
                });
                serde_json::to_writer_pretty(
                    BufWriter::new(File::create(directory.join(format!("{stem}.json")))?),
                    &report,
                )?;
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
    serde_json::to_writer_pretty(
        BufWriter::new(File::create(directory.join("family-summary.json"))?),
        &summaries,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    #[ignore = "runs the GPU world benchmark; optional WARP_BRO_WORLD_BENCH_ARGS JSON argv"]
    fn paired_world_de_benchmark() {
        let _gpu_test = crate::test_gpu::lock();
        let args = match std::env::var("WARP_BRO_WORLD_BENCH_ARGS") {
            Ok(json) => serde_json::from_str::<Vec<String>>(&json).expect("benchmark JSON argv"),
            Err(_) => {
                let stamp = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos();
                let directory = std::env::temp_dir().join(format!(
                    "warpbro-world-de-bench-{}-{stamp}",
                    std::process::id()
                ));
                vec![
                    "WarpBro".into(),
                    "--world-bench".into(),
                    directory.to_string_lossy().into_owned(),
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
    }
}
