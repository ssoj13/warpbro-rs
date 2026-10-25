//! CUDA vs WGPU parity (plan1 2.0) and the WGPU dispatch past the per-dimension workgroup limit
//! (plan1 1.3), through the public render API: `Gpu::advance` and the film readback.
//!
//! The WGSL kernels are a port of fractal-kernels (`render_wgpu.wgsl`): the same estimates,
//! march, normals, sample layout (Sobol dimensions per pixel and sample) and integrator. One
//! sample per pixel with one seed therefore traces the same camera ray on both backends, and the
//! primary guides (albedo, normal) of the two films must agree pixel by pixel. They cannot agree
//! to the bit: NVPTX and the WGPU shader compiler fuse different multiply-adds, and a fractal
//! amplifies a last-bit difference near its silhouettes and in its finest folds, so a few pixels
//! hit a different fold or miss. The tolerance below is per pixel, plus a small share of pixels
//! allowed beyond it; the test reports every case's largest errors either way.

use crate::render::{Batch, Gpu, Plane, Target};
use crate::render_profiles::Backend;
use crate::scene::{
    ArrayAxis, ArrayKind, Coloring, Formula, Mask, MaskArray, MaskInstance, MaskShape, Scene,
};

const WIDTH: usize = 96;
const HEIGHT: usize = 64;
const SEED: u32 = 7;
/// Largest per-channel difference of a pixel's normal guide that counts as agreeing.
const NORMAL_TOLERANCE: f32 = 0.05;
/// Largest per-channel difference of a pixel's albedo guide that counts as agreeing.
const ALBEDO_TOLERANCE: f32 = 0.02;
/// Share of pixels allowed beyond the tolerances (silhouettes and the finest folds).
const MISMATCH_SHARE: f32 = 0.03;

/// The primary guides of one frame, row major, normalized by the sample count.
struct Guides {
    albedo: Vec<[f32; 4]>,
    normal: Vec<[f32; 4]>,
}

/// `scene` on `backend` at one sample per pixel with [`SEED`].
fn guides(gpu: &mut Gpu, scene: &Scene, backend: Backend) -> Guides {
    let mut scene = scene.clone();
    scene.render.backend = backend;
    let mut target = Target::new(WIDTH, HEIGHT);
    while !gpu
        .advance(&mut target, &scene, 1, SEED, Batch::Fixed(1))
        .unwrap_or_else(|e| panic!("{backend:?} render failed: {e}"))
    {}
    let mut tiled = Vec::new();
    let mut albedo = Vec::new();
    let mut normal = Vec::new();
    gpu.read_film(&target, Plane::Albedo, true, &mut tiled, &mut albedo)
        .unwrap();
    gpu.read_film(&target, Plane::Normal, true, &mut tiled, &mut normal)
        .unwrap();
    Guides { albedo, normal }
}

/// The largest per-channel difference of two pixels' rgb.
fn channel_error(a: [f32; 4], b: [f32; 4]) -> f32 {
    (0..3).map(|c| (a[c] - b[c]).abs()).fold(0.0, f32::max)
}

/// Per-case agreement of the two films: the largest normal and albedo errors and the share of
/// pixels beyond the tolerances.
struct Agreement {
    normal_max: f32,
    albedo_max: f32,
    mismatch: f32,
}

fn compare(cuda: &Guides, wgpu: &Guides) -> Agreement {
    let pixels = cuda.normal.len();
    assert_eq!(pixels, WIDTH * HEIGHT);
    assert_eq!(wgpu.normal.len(), pixels);
    let mut agreement = Agreement {
        normal_max: 0.0,
        albedo_max: 0.0,
        mismatch: 0.0,
    };
    let mut beyond = 0usize;
    for i in 0..pixels {
        let normal = channel_error(cuda.normal[i], wgpu.normal[i]);
        let albedo = channel_error(cuda.albedo[i], wgpu.albedo[i]);
        agreement.normal_max = agreement.normal_max.max(normal);
        agreement.albedo_max = agreement.albedo_max.max(albedo);
        // A NaN error counts as beyond the tolerance.
        if !(normal <= NORMAL_TOLERANCE && albedo <= ALBEDO_TOLERANCE) {
            beyond += 1;
        }
    }
    agreement.mismatch = beyond as f32 / pixels as f32;
    agreement
}

/// The share of pixels with a primary hit (a nonzero normal guide): a case must show its
/// fractal for the comparison to mean anything.
fn coverage(g: &Guides) -> f32 {
    let hits = g
        .normal
        .iter()
        .filter(|n| n[0] * n[0] + n[1] * n[1] + n[2] * n[2] > 0.25)
        .count();
    hits as f32 / g.normal.len() as f32
}

/// A scene whose guides are cheap and deterministic: no denoise, no adaptive stop, no bounce.
fn quiet(mut scene: Scene) -> Scene {
    scene.render.denoise.enabled = false;
    scene.render.adaptive.enabled = false;
    scene.render.max_bounces = 0;
    scene.colour.on = false;
    scene
}

/// A world of one Mandelbulb at the origin intersected with `mask` (identity placement).
fn masked_world(mask: Mask) -> Scene {
    let bulb = Formula::PRESETS[crate::params::FAMILY_BULB as usize];
    let mut world = quiet(Scene::preset(bulb));
    world.world_render = true;
    world.camera.distance = 4.0;
    world.camera.fov_y_degrees = 55.0;
    let mut object = Scene::preset(bulb);
    object.object_world = Some(glam::Mat4::IDENTITY.to_cols_array_2d());
    object.masks = vec![MaskInstance {
        mask,
        world: glam::Mat4::IDENTITY.to_cols_array_2d(),
    }];
    world.objects = vec![object];
    world
}

/// Every classic family (Mandelbulb through Hybrid), each orbit trap mode on the Mandelbulb, a
/// Cells mask and a full-circle radial mask array (sweep 0) on a world Mandelbulb: rendered at
/// one sample per pixel with one seed on CUDA and on WGPU, their primary guides agree per pixel
/// within [`NORMAL_TOLERANCE`] / [`ALBEDO_TOLERANCE`] except at most [`MISMATCH_SHARE`] of the
/// pixels. Reports each case's largest errors.
#[test]
fn cuda_wgpu_primary_guides_agree_per_pixel() {
    let _gpu_test = crate::test_gpu::lock();
    let mut gpu = Gpu::new().unwrap();
    let classic = crate::params::FAMILY_BULB..=crate::params::FAMILY_HYBRID;
    let mut cases: Vec<(String, Scene)> = classic
        .map(|code| {
            let formula = Formula::PRESETS[code as usize];
            (
                format!("family {code} {}", formula.name()),
                quiet(Scene::preset(formula)),
            )
        })
        .collect();
    for coloring in Coloring::ALL {
        let mut scene = quiet(Scene::preset(
            Formula::PRESETS[crate::params::FAMILY_BULB as usize],
        ));
        scene.coloring = coloring;
        cases.push((format!("Mandelbulb trap {coloring:?}"), scene));
    }
    cases.push((
        "world Cells mask".into(),
        masked_world(Mask {
            shape: MaskShape::Cells,
            size: [0.25; 3],
            density: 0.5,
            seed: 3,
            ..Mask::default()
        }),
    ));
    cases.push((
        "world radial mask array (sweep 0)".into(),
        masked_world(Mask {
            shape: MaskShape::Sphere,
            size: [0.35; 3],
            array: MaskArray {
                kind: ArrayKind::Radial,
                count: 6,
                radius: 0.7,
                axis: ArrayAxis::Y,
                sweep_degrees: 0.0,
                ..MaskArray::default()
            },
            ..Mask::default()
        }),
    ));

    let mut failures = Vec::new();
    for (name, scene) in &cases {
        let cuda = guides(&mut gpu, scene, Backend::Cuda);
        let wgpu = guides(&mut gpu, scene, Backend::Wgpu);
        let shown = coverage(&cuda);
        let a = compare(&cuda, &wgpu);
        eprintln!(
            "{name}: coverage {:.3} (WGPU {:.3}), normal max {:.4}, albedo max {:.4}, beyond tolerance {:.4}",
            shown,
            coverage(&wgpu),
            a.normal_max,
            a.albedo_max,
            a.mismatch,
        );
        if shown < 0.02 {
            failures.push(format!(
                "{name}: the fractal covers only {shown:.3} of the frame"
            ));
        }
        if a.mismatch > MISMATCH_SHARE {
            failures.push(format!(
                "{name}: {:.4} of the pixels differ (normal max {:.4}, albedo max {:.4})",
                a.mismatch, a.normal_max, a.albedo_max
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "CUDA / WGPU parity:\n{}",
        failures.join("\n")
    );
}

/// A DCI 4K frame (4096x2160, 8.8 M samples: more than 65,535 workgroups of 128) dispatches in
/// two dimensions, and the WGSL index (`linear_id`) covers every pixel exactly once: one sample
/// per pixel in the film's count, and a display code (opaque alpha) for every pixel from
/// `tonemap_display`. A distant camera keeps most rays on the cheap bounding-sphere miss.
#[test]
fn gpu_wgpu_frame_past_the_dispatch_limit_covers_every_pixel() {
    let _gpu_test = crate::test_gpu::lock();
    let (width, height) = (4096, 2160);
    assert!(crate::render_prep::padded_len(width, height) > 65_535 * 128);
    let mut gpu = Gpu::new().unwrap();
    let mut scene = quiet(Scene::preset(
        Formula::PRESETS[crate::params::FAMILY_BULB as usize],
    ));
    scene.render.backend = Backend::Wgpu;
    scene.render.max_steps = 32;
    scene.camera.distance *= 20.0;
    let mut target = Target::new(width, height);
    while !gpu
        .advance(&mut target, &scene, 1, SEED, Batch::Fixed(1))
        .unwrap()
    {}
    let mut tiled = Vec::new();
    let mut radiance = Vec::new();
    gpu.read_film(&target, Plane::Accum, false, &mut tiled, &mut radiance)
        .unwrap();
    assert_eq!(radiance.len(), width * height);
    let wrong = radiance.iter().filter(|p| p[3] != 1.0).count();
    assert_eq!(
        wrong, 0,
        "{wrong} pixels did not receive exactly one sample"
    );
    let blank = target.pixels.iter().filter(|&&p| p >> 24 != 0xff).count();
    assert_eq!(blank, 0, "{blank} pixels have no display code");
}

/// Object-local samples include negative cells, axes, the origin and subnormal-scale offsets.
/// The off-grid lattice avoids making every point a fold symmetry case.
fn formula_probe_points(radius: f32) -> Vec<[f32; 4]> {
    let mut points = vec![
        [0.0; 4],
        [1.0e-7, -2.0e-7, 3.0e-7, 0.0],
        [0.5, -0.5, 1.5, 0.0],
        [-0.5, 0.5, -1.5, 0.0],
    ];
    for axis in 0..3 {
        for sign in [-1.0, 1.0] {
            let mut point = [0.0; 4];
            point[axis] = sign * radius * 0.6;
            points.push(point);
        }
    }
    for z in -3..=3 {
        for y in -3..=3 {
            for x in -3..=3 {
                points.push([
                    (x as f32 + 0.173) * radius / 3.7,
                    (y as f32 - 0.287) * radius / 3.7,
                    (z as f32 + 0.391) * radius / 3.7,
                    0.0,
                ]);
            }
        }
    }
    points
}

/// Change one numeric parameter at a time using the registry's supported slider/range.
/// This checks every packed slot without coupling failures to a simultaneous extreme preset.
fn parameter_probe_formulas(formula: Formula) -> Vec<(String, Formula)> {
    use serde_json::{Value, json};
    let meta = &crate::formulas::FORMULAS[formula.code() as usize];
    let original = serde_json::to_value(formula).unwrap();
    let mut result = Vec::new();
    for parameter in meta.params {
        let value = &original[meta.variant][parameter.name];
        if value.is_string() {
            continue;
        }
        let Some((lo, hi)) = parameter
            .slider
            .map(|slider| (slider.min, slider.max))
            .or(parameter.range)
        else {
            continue;
        };
        let mut variant = original.clone();
        let changed = if let Some(array) = value.as_array() {
            Value::Array(
                array
                    .iter()
                    .enumerate()
                    .map(|(i, _)| json!(lo + (hi - lo) * (0.31 + 0.11 * i as f64)))
                    .collect(),
            )
        } else if value.is_number() {
            let number = lo + (hi - lo) * 0.37;
            if value.is_u64() {
                json!(number.round().max(0.0) as u64)
            } else if value.is_i64() {
                json!(number.round() as i64)
            } else {
                json!(number)
            }
        } else {
            continue;
        };
        variant[meta.variant][parameter.name] = changed;
        let changed: Formula = serde_json::from_value(variant)
            .unwrap_or_else(|error| panic!("{} {}: {error}", meta.variant, parameter.name));
        result.push((parameter.name.into(), changed));
    }
    use crate::formulas::{AttractorKind, ComplexTpmsKind, PolyhedralSymmetry, TpmsKind};
    let modes: &[&str] = match formula {
        Formula::PolyhedralKifs(_) => &["Dodecahedron"],
        Formula::GyroidTpms(_) => &["SchwarzP", "SchwarzD", "Neovius"],
        Formula::LidinoidTpms(_) => &["FischerKochS", "SplitP"],
        Formula::AttractorTube(_) => &["Halvorsen", "Thomas"],
        _ => &[],
    };
    for mode in modes {
        let mut changed = formula;
        match &mut changed {
            Formula::PolyhedralKifs(f) => f.symmetry = PolyhedralSymmetry::Dodecahedron,
            Formula::GyroidTpms(f) => {
                f.surface = match *mode {
                    "SchwarzP" => TpmsKind::SchwarzP,
                    "SchwarzD" => TpmsKind::SchwarzD,
                    _ => TpmsKind::Neovius,
                };
            }
            Formula::LidinoidTpms(f) => {
                f.surface = if *mode == "SplitP" {
                    ComplexTpmsKind::SplitP
                } else {
                    ComplexTpmsKind::FischerKochS
                };
            }
            Formula::AttractorTube(f) => {
                f.system = if *mode == "Thomas" {
                    AttractorKind::Thomas
                } else {
                    AttractorKind::Halvorsen
                };
            }
            _ => unreachable!(),
        }
        result.push(((*mode).into(), changed));
    }
    result
}

/// Distance and trap parity before ray marching: each mismatch identifies the family,
/// parameter case and exact object-local point. No silhouette mismatch allowance applies here.
#[test]
fn cuda_extended_formula_estimates_and_traps_match_wgpu() {
    let _gpu_test = crate::test_gpu::lock();
    let mut gpu = Gpu::new().unwrap();
    let mut failures = Vec::new();
    for code in 8..Formula::COUNT {
        let formula = Formula::PRESETS[code as usize];
        let points = formula_probe_points(formula.framing_radius());
        let base = quiet(Scene::preset(formula));
        let mut cases = vec![("preset".to_string(), base.clone())];
        for coloring in Coloring::ALL {
            let mut scene = base.clone();
            scene.coloring = coloring;
            scene.trap_point = [0.13, -0.21, 0.17];
            scene.trap_axis = 1;
            cases.push((format!("trap-{coloring:?}"), scene));
        }
        let mut fractional = base.clone();
        fractional.render.iterations = 7.375;
        cases.push(("fractional-iterations".into(), fractional));
        let mut inverted = base.clone();
        inverted.inversion_strength = 0.37;
        inverted.inversion_radius = 1.13;
        inverted.inversion_center = [0.11, -0.17, 0.23];
        inverted.julia_mix = 0.35;
        inverted.julia = [-0.18, 0.27, 0.09];
        inverted.orbit_offset = [0.03, -0.04, 0.07];
        cases.push(("inversion-and-orbit".into(), inverted));
        for (name, formula) in parameter_probe_formulas(formula) {
            let mut scene = base.clone();
            scene.formula = formula;
            scene.coloring = Coloring::TrapPoint;
            scene.trap_point = [0.13, -0.21, 0.17];
            cases.push((format!("parameter-{name}"), scene));
        }
        for (name, mut scene) in cases {
            scene.render.backend = Backend::Cuda;
            let cuda = gpu.probe_formula(&scene, &points).unwrap();
            scene.render.backend = Backend::Wgpu;
            let wgpu = gpu.probe_formula(&scene, &points).unwrap();
            assert_eq!(cuda.len(), points.len());
            assert_eq!(wgpu.len(), points.len());
            let mut max_error = [0.0f32; 2];
            let mut beyond = [0usize; 2];
            let mut worst = [0usize; 2];
            for (index, (a, b)) in cuda.iter().zip(&wgpu).enumerate() {
                for channel in 0..2 {
                    let error = (a[channel] - b[channel]).abs();
                    if error > max_error[channel] {
                        max_error[channel] = error;
                        worst[channel] = index;
                    }
                    let absolute = if channel == 0 { 5.0e-5 } else { 1.0e-4 };
                    let tolerance = absolute + 2.0e-4 * b[channel].abs();
                    if !a[channel].is_finite() || !b[channel].is_finite() || error > tolerance {
                        beyond[channel] += 1;
                    }
                }
            }
            eprintln!(
                "{}",
                serde_json::json!({
                    "check": "formula-probe", "family": code, "case": name,
                    "points": points.len(), "max_error": max_error, "beyond": beyond,
                    "worst_points": worst.map(|i| points[i]),
                    "worst_cuda": worst.map(|i| cuda[i]),
                    "worst_wgpu": worst.map(|i| wgpu[i]),
                })
            );
            if beyond != [0, 0] {
                failures.push(format!(
                    "family={code} case={name} beyond={beyond:?} max={max_error:?}"
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "DE parity failures:\n{}",
        failures.join("\n")
    );
}

/// Analytical/procedural families ignore P_ITERATIONS, including fractional values.
/// This checks the field itself separately from the fractional shading-normal path.
#[test]
fn cuda_noniterating_formulas_ignore_iteration_count_on_both_backends() {
    let _gpu_test = crate::test_gpu::lock();
    let mut gpu = Gpu::new().unwrap();
    for meta in &crate::formulas::FORMULAS {
        if meta.code < 8 || meta.iterates {
            continue;
        }
        let formula = Formula::PRESETS[meta.code as usize];
        let points = formula_probe_points(formula.framing_radius());
        for backend in [Backend::Cuda, Backend::Wgpu] {
            let mut scene = quiet(Scene::preset(formula));
            scene.render.backend = backend;
            scene.render.iterations = 3.0;
            let reference = gpu.probe_formula(&scene, &points).unwrap();
            for iterations in [3.375, 7.0, 17.875] {
                scene.render.iterations = iterations;
                assert_eq!(
                    gpu.probe_formula(&scene, &points).unwrap(),
                    reference,
                    "family={} backend={backend:?} iterations={iterations}",
                    meta.code
                );
            }
        }
    }
}

/// Extended families traverse the same primary march and normal path on both backends.
/// The established classic guide tolerances remain unchanged.
#[test]
fn cuda_extended_primary_guides_match_wgpu_with_fractional_iterations_and_inversion() {
    let _gpu_test = crate::test_gpu::lock();
    let mut gpu = Gpu::new().unwrap();
    let mut failures = Vec::new();
    for code in 8..Formula::COUNT {
        for modified in [false, true] {
            let mut scene = quiet(Scene::preset(Formula::PRESETS[code as usize]));
            if modified {
                scene.render.iterations = 7.375;
                scene.inversion_strength = 0.12;
                scene.inversion_center = [0.07, -0.03, 0.05];
            }
            let cuda = guides(&mut gpu, &scene, Backend::Cuda);
            let wgpu = guides(&mut gpu, &scene, Backend::Wgpu);
            let agreement = compare(&cuda, &wgpu);
            eprintln!(
                "{}",
                serde_json::json!({
                    "check": "extended-guides", "family": code, "modified": modified,
                    "coverage_cuda": coverage(&cuda), "coverage_wgpu": coverage(&wgpu),
                    "normal_max": agreement.normal_max, "albedo_max": agreement.albedo_max,
                    "mismatch": agreement.mismatch,
                })
            );
            if coverage(&cuda) < 0.02
                || coverage(&wgpu) < 0.02
                || agreement.mismatch > MISMATCH_SHARE
            {
                failures.push(format!(
                    "family={code} modified={modified} mismatch={}",
                    agreement.mismatch
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "Extended guide parity:\n{}",
        failures.join("\n")
    );
}

/// A real primary hit belonging to one of the two solid-colour oracle objects.
fn oracle_object(guides: &Guides, index: usize) -> Option<usize> {
    let n = guides.normal[index];
    if n[0] * n[0] + n[1] * n[1] + n[2] * n[2] <= 0.25 {
        return None;
    }
    let color = guides.albedo[index];
    if color[0] > color[2] * 2.0 {
        Some(0)
    } else if color[2] > color[0] * 2.0 {
        Some(1)
    } else {
        None
    }
}

/// Prefer the interior of the observed object, including its actual sampling jitter.
/// Picking must use the same ray rather than assume a torus or porous field hits its centre.
fn observed_pick(
    gpu: &mut Gpu,
    scene: &Scene,
    guides: &Guides,
    size: (usize, usize),
    object: usize,
    seed: u32,
) -> Result<(), String> {
    let (width, height) = size;
    let index = (0..guides.normal.len())
        .filter(|&index| oracle_object(guides, index) == Some(object))
        .max_by_key(|&index| {
            let (x, y) = (index % width, index / width);
            let mut same = 0;
            for py in y.saturating_sub(2)..=(y + 2).min(height - 1) {
                for px in x.saturating_sub(2)..=(x + 2).min(width - 1) {
                    same += (oracle_object(guides, py * width + px) == Some(object)) as usize;
                }
            }
            same
        })
        .ok_or_else(|| format!("object {object} has no observed primary hit"))?;
    let (x, y) = (index % width, index / width);
    let [u, v] = if let Some(direct) = scene.direct {
        // The first Direct stratum uses the same deterministic subpixel position.
        [0.5 / direct.grid as f32; 2]
    } else {
        std::array::from_fn(|dimension| {
            standard_surface_bsdf::sampling::sobol(x as u32 ^ seed, y as u32, 0, dimension as u32)
        })
    };
    let ndc = [
        2.0 * (x as f32 + u) / width as f32 - 1.0,
        1.0 - 2.0 * (y as f32 + v) / height as f32,
    ];
    let mut reference_depth = None;
    for (backend, force_dynamic) in [
        (Backend::Cuda, false),
        (Backend::Cuda, true),
        (Backend::Wgpu, false),
    ] {
        let mut selected = scene.clone();
        selected.render.backend = backend;
        gpu.set_force_mixed_world(force_dynamic);
        let picked = gpu
            .pick(&selected, width, height, ndc)
            .map_err(|error| error.to_string());
        gpu.set_force_mixed_world(false);
        let Some((actual_object, depth)) = picked? else {
            return Err(format!(
                "{backend:?} dynamic={force_dynamic} missed object {object}, pixel=({x},{y}), ndc={ndc:?}"
            ));
        };
        if actual_object != object || !depth.is_finite() || depth <= 0.0 {
            return Err(format!(
                "{backend:?} dynamic={force_dynamic} pixel=({x},{y}) expected={object}, picked={actual_object}, depth={depth}"
            ));
        }
        if let Some(reference) = reference_depth {
            // Each independent sphere trace accepts a finite pixel-footprint band.
            // Four projected footprints cover both contact bands and final refinement.
            let slope = selected.pack(width as u32, height as u32)[crate::params::P_FOOTPRINT];
            let tolerance = 4.0 * slope * depth.max(reference)
                + 32.0 * f32::EPSILON * selected.formula.framing_radius();
            if (depth - reference).abs() > tolerance {
                return Err(format!(
                    "{backend:?} dynamic={force_dynamic} object={object} depth={depth}, reference={reference}, footprint tolerance={tolerance}"
                ));
            }
        } else {
            reference_depth = Some(depth);
        }
    }
    Ok(())
}

struct OracleFilm {
    radiance: Vec<[f32; 4]>,
    albedo: Vec<[f32; 4]>,
    normal: Vec<[f32; 4]>,
    samples: Vec<[f32; 4]>,
    first: Guides,
}

fn oracle_plane(gpu: &Gpu, target: &Target, plane: Plane) -> Vec<[f32; 4]> {
    let mut tiled = Vec::new();
    let mut output = Vec::new();
    gpu.read_film(target, plane, false, &mut tiled, &mut output)
        .unwrap();
    output
}

/// Progressive 1+3+4 samples exercise the launch offsets as well as the material path.
fn oracle_film(gpu: &mut Gpu, scene: &Scene) -> OracleFilm {
    let mut target = Target::new(48, 32);
    gpu.step(&mut target, scene, 1, 17, false).unwrap();
    let first = Guides {
        albedo: oracle_plane(gpu, &target, Plane::Albedo),
        normal: oracle_plane(gpu, &target, Plane::Normal),
    };
    for count in [3, 4] {
        gpu.step(&mut target, scene, count, 17, false).unwrap();
    }
    assert_eq!(target.samples, 8);
    assert!(target.colour_error.is_none(), "{:?}", target.colour_error);
    OracleFilm {
        radiance: gpu.raw_scene_linear(&target).unwrap(),
        albedo: oracle_plane(gpu, &target, Plane::Albedo),
        normal: oracle_plane(gpu, &target, Plane::Normal),
        samples: oracle_plane(gpu, &target, Plane::Accum),
        first,
    }
}

/// Reuse the existing CUDA material oracle's 3e-5 absolute + relative numerical bound.
/// Exact differences are reported too; a failing family is investigated rather than hidden
/// behind the broader cross-backend silhouette mismatch allowance.
fn compare_oracle_plane(
    actual: &[[f32; 4]],
    expected: &[[f32; 4]],
    family: u32,
    shade: &str,
    plane: &str,
) -> Option<String> {
    assert_eq!(actual.len(), expected.len());
    let mut max_absolute = 0.0f32;
    let mut max_relative = 0.0f32;
    let mut exact_differences = 0;
    let mut beyond = 0;
    let mut worst = (0, 0);
    for (index, (a, b)) in actual.iter().zip(expected).enumerate() {
        for channel in 0..4 {
            let error = (a[channel] - b[channel]).abs();
            if a[channel] != b[channel] {
                exact_differences += 1;
            }
            if error > max_absolute {
                max_absolute = error;
                worst = (index, channel);
            }
            max_relative = max_relative.max(error / b[channel].abs().max(1.0e-12));
            if !a[channel].is_finite()
                || !b[channel].is_finite()
                || error > 3.0e-5 + 3.0e-5 * b[channel].abs()
            {
                beyond += 1;
            }
        }
    }
    eprintln!(
        "{}",
        serde_json::json!({
            "check": "world-radiance-oracle", "family": family, "shade": shade, "plane": plane,
            "max_absolute": max_absolute, "max_relative": max_relative,
            "exact_differences": exact_differences, "beyond": beyond,
            "worst_pixel": worst.0, "worst_channel": worst.1,
            "actual": actual[worst.0], "reference": expected[worst.0],
        })
    );
    (beyond != 0).then(|| format!(
        "family={family} shade={shade} plane={plane} beyond={beyond} max={max_absolute:e} pixel={}",
        worst.0
    ))
}

/// The production family/model specializations preserve real multibounce radiance and
/// progressive guides for two same-family objects, including mixed Fast/Full materials.
#[test]
fn cuda_extended_specialized_worlds_match_dynamic_dispatch() {
    use crate::render_profiles::{DirectRender, DirectShading};
    use crate::scene::{ColorSource, Facing, Material, MaterialModel};
    let _gpu_test = crate::test_gpu::lock();
    let mut gpu = Gpu::new().unwrap();
    let mut failures = Vec::new();
    for code in 8..Formula::COUNT {
        for shade in ["fast", "full", "mixed", "direct"] {
            let mut scene = quiet(Scene::preset(Formula::PRESETS[code as usize]));
            scene.render.backend = Backend::Cuda;
            scene.render.max_bounces = 2;
            scene.camera.f_number = 0.0;
            scene.world_render = true;
            scene.lights = vec![scene.sun];
            // The root material deliberately contradicts a homogeneous world's objects.
            scene.material.model = if shade == "fast" {
                MaterialModel::StandardSurface
            } else {
                MaterialModel::Fast
            };
            let radius = scene.formula.framing_radius();
            let mut objects = Vec::new();
            for object_index in 0..2 {
                let mut object = scene.clone();
                object.world_render = false;
                object.lights.clear();
                let full =
                    shade == "full" || shade == "direct" || (shade == "mixed" && object_index == 1);
                let color = if object_index == 0 {
                    [0.8, 0.12, 0.04]
                } else {
                    [0.04, 0.12, 0.8]
                };
                object.material = Material {
                    model: if full {
                        MaterialModel::StandardSurface
                    } else {
                        MaterialModel::Fast
                    },
                    color_source: ColorSource::Material,
                    base_color: color,
                    metalness: if object_index == 0 { 0.35 } else { 0.0 },
                    diffuse_roughness: 0.25,
                    specular_roughness: 0.34,
                    specular_anisotropy: 0.32,
                    specular_rotation: 0.17,
                    coat: 0.3,
                    coat_roughness: 0.24,
                    sheen: 0.15,
                    sheen_roughness: 0.5,
                    thin_film_thickness: 320.0,
                    thin_film_ior: 1.35,
                    facing: Some(Facing {
                        color: color.map(|c| c * 0.85),
                        roughness: 0.45,
                        metallic: 0.2,
                        exponent: 2.0,
                    }),
                    ..Material::default()
                };
                let mut matrix = glam::Mat4::from_scale(glam::Vec3::new(0.5, 0.55, 0.45));
                matrix.y_axis += matrix.x_axis * 0.12;
                matrix.w_axis.x = if object_index == 0 {
                    -0.6 * radius
                } else {
                    0.6 * radius
                };
                object.object_world = Some(matrix.to_cols_array_2d());
                object.masks = vec![MaskInstance {
                    mask: Mask {
                        shape: MaskShape::Sphere,
                        size: [radius * 1.5; 3],
                        ..Mask::default()
                    },
                    world: matrix.to_cols_array_2d(),
                }];
                objects.push(object);
            }
            scene.objects = objects;
            if shade == "direct" {
                scene.direct = Some(DirectRender {
                    shading: DirectShading::default(),
                    grid: 3,
                });
            }
            gpu.set_force_mixed_world(false);
            let specialized = oracle_film(&mut gpu, &scene);
            gpu.set_force_mixed_world(true);
            let dynamic = oracle_film(&mut gpu, &scene);
            gpu.set_force_mixed_world(false);
            for (plane, actual, reference) in [
                ("radiance", &specialized.radiance, &dynamic.radiance),
                ("albedo", &specialized.albedo, &dynamic.albedo),
                ("normal", &specialized.normal, &dynamic.normal),
            ] {
                failures.extend(compare_oracle_plane(actual, reference, code, shade, plane));
            }
            for (index, (a, b)) in specialized.samples.iter().zip(&dynamic.samples).enumerate() {
                if a[3] != 8.0 || b[3] != 8.0 {
                    failures.push(format!(
                        "family={code} shade={shade} pixel={index} sample counts={}/{}",
                        a[3], b[3]
                    ));
                    break;
                }
            }
            for object in 0..2 {
                let observed = specialized
                    .first
                    .normal
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| oracle_object(&specialized.first, *index) == Some(object))
                    .count();
                let lit = specialized
                    .first
                    .normal
                    .iter()
                    .enumerate()
                    .any(|(index, _)| {
                        oracle_object(&specialized.first, index) == Some(object)
                            && specialized.radiance[index][..3]
                                .iter()
                                .any(|value| *value > 1.0e-6)
                    });
                if observed == 0 || !lit {
                    failures.push(format!(
                        "family={code} shade={shade} object={object} observed={observed} lit={lit}"
                    ));
                }
                // One geometry pick check per family suffices; all material variants share placement.
                if shade == "fast"
                    && let Err(error) =
                        observed_pick(&mut gpu, &scene, &specialized.first, (48, 32), object, 17)
                {
                    failures.push(format!("family={code} homogeneous pick: {error}"));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "Specialization parity:\n{}",
        failures.join("\n")
    );
}

/// Mixed worlds must dispatch each object's own family rather than interpreting an
/// extended code as the old world sentinel or silently substituting Hybrid.
#[test]
fn cuda_extended_mixed_world_guides_match_wgpu() {
    use crate::scene::{ColorSource, Material, MaterialModel};
    let _gpu_test = crate::test_gpu::lock();
    let mut gpu = Gpu::new().unwrap();
    let mut failures = Vec::new();
    for code in 8..Formula::COUNT {
        let mut scene = quiet(Scene::preset(Formula::PRESETS[code as usize]));
        let radius = scene.formula.framing_radius();
        let mut extended = scene.clone();
        extended.material = Material {
            model: MaterialModel::Fast,
            color_source: ColorSource::Material,
            base_color: [0.8, 0.1, 0.05],
            ..Material::default()
        };
        let mut matrix = glam::Mat4::from_scale(glam::Vec3::splat(0.5));
        matrix.w_axis.x = -0.6 * radius;
        extended.object_world = Some(matrix.to_cols_array_2d());
        let mut classic = extended.clone();
        classic.formula = Formula::PRESETS[0];
        classic.material.base_color = [0.05, 0.1, 0.8];
        matrix.w_axis.x = 0.6 * radius;
        classic.object_world = Some(matrix.to_cols_array_2d());
        scene.world_render = true;
        scene.objects = vec![extended, classic];
        scene.lights = vec![scene.sun];
        let cuda = guides(&mut gpu, &scene, Backend::Cuda);
        let wgpu = guides(&mut gpu, &scene, Backend::Wgpu);
        let agreement = compare(&cuda, &wgpu);
        let object_hits = |g: &Guides| {
            [0, 1].map(|object| {
                (0..g.normal.len())
                    .filter(|&index| oracle_object(g, index) == Some(object))
                    .count()
            })
        };
        let cuda_hits = object_hits(&cuda);
        let wgpu_hits = object_hits(&wgpu);
        for object in 0..2 {
            if let Err(error) =
                observed_pick(&mut gpu, &scene, &cuda, (WIDTH, HEIGHT), object, SEED)
            {
                failures.push(format!("family={code} mixed pick: {error}"));
            }
        }
        eprintln!(
            "{}",
            serde_json::json!({
                "check": "mixed-world", "family": code, "cuda_object_hits": cuda_hits,
                "wgpu_object_hits": wgpu_hits, "normal_max": agreement.normal_max,
                "albedo_max": agreement.albedo_max, "mismatch": agreement.mismatch,
            })
        );
        if cuda_hits.contains(&0) || wgpu_hits.contains(&0) || agreement.mismatch > MISMATCH_SHARE {
            failures.push(format!(
                "family={code} hits={cuda_hits:?}/{wgpu_hits:?} mismatch={}",
                agreement.mismatch
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "Mixed-world parity:\n{}",
        failures.join("\n")
    );
}

/// WGSL round uses ties-to-even; Rust round uses ties-away-from-zero. Periodic
/// half-cells and authored half-integer counts therefore need explicit regression coverage.
#[test]
fn cuda_extended_rounding_matches_wgpu_at_positive_and_negative_half_cells() {
    use crate::formulas::{AstrolabeGimbal, HopfFibration, MobiusIfs, PhyllotaxisShell};
    let _gpu_test = crate::test_gpu::lock();
    let mut gpu = Gpu::new().unwrap();
    let trap_point = [0.13f32, -0.21, 0.17];
    let mut scene = quiet(Scene::preset(Formula::MobiusIfs(MobiusIfs {
        scale: 1.0,
        min_radius: 0.25,
        fixed_radius: 1.0,
        lattice_pitch: [1.0; 3],
        twist: 0.0,
        rotation_degrees: [0.0; 3],
        ..MobiusIfs::PRESET
    })));
    scene.render.iterations = 1.0;
    scene.coloring = Coloring::TrapPoint;
    scene.trap_point = trap_point;
    let mut points = Vec::new();
    for axis in 0..3 {
        for value in [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5] {
            let mut point = [0.23, -0.31, 0.17, 0.0];
            point[axis] = value;
            points.push(point);
        }
    }
    for backend in [Backend::Cuda, Backend::Wgpu] {
        scene.render.backend = backend;
        let result = gpu.probe_formula(&scene, &points).unwrap();
        assert_eq!(result.len(), points.len());
        for (point, actual) in points.iter().zip(result) {
            // One iteration has a closed-form trap. Its asymmetric point distinguishes the
            // signs of ±half-cell wraps even when the radial distance estimate is symmetric.
            let wrapped: [f32; 3] =
                std::array::from_fn(|axis| point[axis] - point[axis].round_ties_even());
            let radius2: f32 = wrapped.iter().map(|value| value * value).sum();
            let factor = (1.0 / radius2.max(0.0625)).clamp(1.0, 16.0);
            let expected = wrapped
                .iter()
                .zip(trap_point)
                .map(|(&value, trap)| (value * factor - trap).powi(2))
                .sum::<f32>()
                .sqrt();
            assert!(
                actual[1].is_finite() && (actual[1] - expected).abs() <= 1.0e-4,
                "Mobius half-cell point={point:?} backend={backend:?} trap={} expected={expected}",
                actual[1],
            );
        }
    }

    let mut cases = Vec::new();
    for half_count in [8.5, 10.5, 12.5] {
        cases.push(Scene::preset(Formula::PhyllotaxisShell(PhyllotaxisShell {
            count: half_count,
            ..PhyllotaxisShell::PRESET
        })));
        cases.push(Scene::preset(Formula::AstrolabeGimbal(AstrolabeGimbal {
            gear_teeth: half_count,
            ..AstrolabeGimbal::PRESET
        })));
    }
    cases.push(Scene::preset(Formula::HopfFibration(HopfFibration {
        fibers: 2,
        shells: 1,
        twist: 0.0,
        scale: 1.0,
        ..HopfFibration::PRESET
    })));
    for mut scene in cases {
        let mut points = formula_probe_points(scene.formula.framing_radius());
        // xi1=±pi/2 with two fibers gives an exact ±half-sector phase in Hopf's field.
        points.extend([[0.0, 1.0, 0.3, 0.0], [0.0, -1.0, 0.3, 0.0]]);
        scene.coloring = Coloring::TrapPoint;
        scene.trap_point = trap_point;
        scene.render.backend = Backend::Cuda;
        let cuda = gpu.probe_formula(&scene, &points).unwrap();
        scene.render.backend = Backend::Wgpu;
        let wgpu = gpu.probe_formula(&scene, &points).unwrap();
        assert_eq!(cuda.len(), points.len());
        assert_eq!(wgpu.len(), points.len());
        for (index, (a, b)) in cuda.iter().zip(&wgpu).enumerate() {
            for channel in 0..2 {
                let absolute = if channel == 0 { 5.0e-5 } else { 1.0e-4 };
                let tolerance = absolute + 2.0e-4 * b[channel].abs();
                assert!(
                    a[channel].is_finite()
                        && b[channel].is_finite()
                        && (a[channel] - b[channel]).abs() <= tolerance,
                    "half-count/sector family={} point={:?} channel={channel} cuda={a:?} wgpu={b:?}",
                    scene.formula.code(),
                    points[index],
                );
            }
        }
    }
}
