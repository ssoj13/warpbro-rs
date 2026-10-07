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

use crate::render::Gpu;
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
    while position < args.len() {
        let flag = &args[position];
        let value = args
            .get(position + 1)
            .with_context(|| format!("Missing value for {flag}"))?;
        match flag.as_str() {
            "--case" => case = Some(value.as_str()),
            "--seed" => seed = value.parse().context("Invalid benchmark seed")?,
            "--batch" => batch = value.parse().context("Invalid benchmark batch")?,
            "--adaptive" => {
                adaptive_threshold = Some(value.parse().context("Invalid adaptive threshold")?)
            }
            _ => anyhow::bail!("Unknown world benchmark option: {flag}"),
        }
        position += 2;
    }
    let [width, height, samples] = dimensions;
    ensure!(
        width > 0
            && height > 0
            && width <= 16384
            && height <= 16384
            && width.checked_mul(height).is_some_and(|n| n <= 67_108_864),
        "Invalid benchmark resolution"
    );
    ensure!(
        (1..=1_000_000).contains(&samples),
        "Invalid benchmark samples"
    );
    ensure!(batch == 4 || batch == 8, "Benchmark batch must be 4 or 8");
    ensure!(
        seed <= 16_777_216,
        "Benchmark seed must be exactly representable in the f32 GPU parameter block"
    );
    ensure!(
        case.is_none_or(|name| CASES.iter().any(|c| c.0 == name)),
        "Unknown benchmark case"
    );
    let samples = samples as u32;
    std::fs::create_dir_all(directory)?;
    let mut gpu = Gpu::new().map_err(anyhow::Error::msg)?;
    println!(
        "World benchmark GPU: {} · {width}x{height} · {samples} spp · seed {seed} · batch {batch}",
        gpu.name
    );
    for (name, preset, frame, diffuse) in CASES {
        if case.is_some_and(|chosen| chosen != name) {
            continue;
        }
        let authored = crate::presets::scene(preset).map_err(anyhow::Error::msg)?;
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
        gpu.prepare_scene(&scene, width, height)
            .map_err(anyhow::Error::msg)?;
        {
            let mut warmup = gpu.target(width, height);
            gpu.step(&mut warmup, &scene, 1, seed, None, false);
            ensure!(
                warmup.colour_error.is_none(),
                "Warm-up display error: {:?}",
                warmup.colour_error
            );
        }
        let mut target = gpu.target(width, height);
        gpu.prepare_target(&mut target, &scene, None);
        let mut batches = Vec::with_capacity(samples.div_ceil(batch) as usize);
        let start = Instant::now();
        while !target.complete(samples) {
            let count = batch.min(samples - target.samples);
            let final_pass = target.samples + count == samples;
            gpu.step(&mut target, &scene, count, seed, None, final_pass);
            ensure!(
                target.colour_error.is_none(),
                "Display error: {:?}",
                target.colour_error
            );
            batches.push((count, target.last_ms));
        }
        let elapsed = start.elapsed().as_secs_f64();
        let readback_start = Instant::now();
        let pixels = gpu.raw_scene_linear(&target);
        let readback_seconds = readback_start.elapsed().as_secs_f64();
        ensure!(
            pixels.iter().all(|p| p[..3].iter().all(|c| c.is_finite())),
            "Nonfinite raw radiance in {name}"
        );
        let stem = format!("{name}.seed-{seed}.spp-{samples}");
        let base = Path::new(directory);
        let raw_path = base.join(format!("{stem}.rgb.f32"));
        let mut raw = BufWriter::new(File::create(&raw_path)?);
        for pixel in &pixels {
            for channel in &pixel[..3] {
                raw.write_all(&channel.to_le_bytes())?;
            }
        }
        raw.flush()?;
        let encoding = crate::render_service::PngEncoding::displayed(target.light_kind.hdr());
        target
            .save_png(
                &base.join(crate::fs_name::frame_file(&stem, None, encoding.suffix())),
                encoding,
                crate::color::BT2408_SDR_WHITE_NITS,
                true,
            )
            .map_err(anyhow::Error::msg)?;
        let metadata = json!({
            "schema": 1,
            "case": name, "preset": crate::presets::ANIMATED[preset].name, "frame": frame,
            "gpu": &gpu.name, "width": width, "height": height, "samples": samples,
            "seed": seed, "batch": batch, "world_render": true,
            "adaptive_threshold": adaptive_threshold, "samples_taken": target.samples,
            "active_tiles": [target.active_tiles.0, target.active_tiles.1],
            "denoise_enabled": false, "exposure_multiplier": 1.0, "saturation": 1.0,
            "elapsed_seconds": elapsed, "raw_readback_seconds": readback_seconds,
            "ms_per_sample": elapsed * 1000.0 / f64::from(samples),
            "million_samples_per_second": (width * height) as f64 * f64::from(samples) / elapsed / 1e6,
            "timing_scope": "Gpu::step including per-batch display/readback; excludes warmup, target allocation, artifact writes and final raw readback",
            "raw": {"file": raw_path.file_name().unwrap().to_string_lossy(),
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
