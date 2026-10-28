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

use soluwell_studio::render::{Batch, Gpu, Target};
use soluwell_studio::render_prep::{Plane, WorldUpload, padded_len};
use soluwell_studio::render_profiles::{Backend, DirectRender, DirectShading};
use soluwell_studio::scene::{
    ArrayAxis, ArrayKind, ColorSource, Coloring, Facing, Mask, MaskArray, MaskInstance, MaskShape,
    Material, MaterialModel, Scene,
};

use crate::engine::{WarpBroGpuExt, init_warpbro, new_gpu};
use crate::fractal_node::{Formula, FractalSceneExt};

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
    let mut world = quiet(Scene::preset_fractal(bulb));
    world.world_render = true;
    world.camera.distance = 4.0;
    world.camera.fov_y_degrees = 55.0;
    let mut object = Scene::preset_fractal(bulb);
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
    let mut gpu = new_gpu().unwrap();
    let classic = crate::params::FAMILY_BULB..=crate::params::FAMILY_HYBRID;
    let mut cases: Vec<(String, Scene)> = classic
        .map(|code| {
            let formula = Formula::PRESETS[code as usize];
            (
                format!("family {code} {}", formula.name()),
                quiet(Scene::preset_fractal(formula)),
            )
        })
        .collect();
    for coloring in Coloring::ALL {
        let mut scene = quiet(Scene::preset_fractal(
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
    assert!(padded_len(width, height) > 65_535 * 128);
    let mut gpu = new_gpu().unwrap();
    let mut scene = quiet(Scene::preset_fractal(
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

fn formula_probe_cases(formula: Formula) -> Vec<(String, Scene)> {
    let base = quiet(Scene::preset_fractal(formula));
    let mut cases = vec![("preset".to_string(), base.clone())];
    for coloring in Coloring::ALL {
        let mut scene = base.clone();
        scene.coloring = coloring;
        scene.set_trap_point([0.13, -0.21, 0.17]);
        scene.set_trap_axis(1);
        cases.push((format!("trap-{coloring:?}"), scene));
    }
    let mut fractional = base.clone();
    fractional.render.iterations = 7.375;
    cases.push(("fractional-iterations".into(), fractional));
    let mut inverted = base.clone();
    inverted.set_inversion_strength(0.37);
    inverted.set_inversion_radius(1.13);
    inverted.set_inversion_center([0.11, -0.17, 0.23]);
    inverted.set_julia_mix(0.35);
    inverted.set_julia([-0.18, 0.27, 0.09]);
    inverted.set_orbit_offset([0.03, -0.04, 0.07]);
    cases.push(("inversion-and-orbit".into(), inverted));
    for (name, formula) in parameter_probe_formulas(formula) {
        let mut scene = base.clone();
        scene.set_formula(formula);
        scene.coloring = Coloring::TrapPoint;
        scene.set_trap_point([0.13, -0.21, 0.17]);
        cases.push((format!("parameter-{name}"), scene));
    }
    cases
}

/// On an exact X axis, the knot has reflected branches with exactly equal geometric
/// distance. Evaluate each reflection class once, then expose both trap vectors only
/// for the globally nearest class. No epsilon can admit a nonminimum branch.
/// Negative X pairs k with p-1-k; positive X pairs k with (p-k) mod p. Integer q and
/// strand count preserve the braid cosine under (qt, pol_angle) -> (-qt, -pol_angle).
fn knot_axis_minimum_traps(scene: &Scene, point: [f32; 4]) -> Option<Vec<f64>> {
    let Formula::TorusKnotBraid(f) = scene.formula() else {
        return None;
    };
    if scene.inversion_strength() != 0.0
        || point[0] == 0.0
        || point[1] != 0.0
        || point[2] != 0.0
        || scene.coloring == Coloring::Radius
    {
        return None;
    }
    // Match the packed scene ABI; these authored counts are quantized before dispatch.
    let p = f.winds_p.round().max(1.0) as usize;
    let q = f.winds_q.round().max(1.0) as usize;
    let count = p.clamp(1, 12);
    let strands = f.sub_strands.clamp(1, 12) as f64;
    let negative_x = point[0] < 0.0;
    let major = f64::from(f.major_radius);
    let minor = f64::from(f.minor_radius);
    let tube = f64::from(f.tube_radius);
    let braid = f64::from(f.braid_depth);
    let slope = ((p as f64 * major).powi(2) + (q as f64 * minor).powi(2)).sqrt()
        / (p as f64 * major).max(0.1);
    let mut visited = vec![false; count];
    let mut minimum = f64::INFINITY;
    let mut vectors = Vec::new();
    for k in 0..count {
        if visited[k] {
            continue;
        }
        let mirror = if negative_x { p - 1 - k } else { (p - k) % p };
        if mirror >= count {
            return None; // A truncated branch inventory need not contain its reflection.
        }
        visited[k] = true;
        visited[mirror] = true;
        let numerator = (q * (usize::from(negative_x) + 2 * k)) % (2 * p);
        let (cos_qt, sin_qt) = if numerator == 0 {
            (1.0, 0.0)
        } else if numerator == p {
            (-1.0, 0.0)
        } else {
            let angle = std::f64::consts::PI * numerator.min(2 * p - numerator) as f64 / p as f64;
            (
                angle.cos(),
                if numerator < p {
                    angle.sin()
                } else {
                    -angle.sin()
                },
            )
        };
        let qt = std::f64::consts::PI * numerator as f64 / p as f64;
        let radial = f64::from(point[0]).abs() - (major + minor * cos_qt);
        let vertical = -minor * sin_qt;
        let polar = vertical.atan2(radial);
        let distance = ((radial * radial + vertical * vertical).sqrt()
            - tube
            - braid * (strands * (polar - qt)).cos())
            / slope.max(1.0);
        let vector = [radial, vertical, sin_qt];
        if distance < minimum {
            minimum = distance;
            vectors.clear();
        }
        if distance == minimum {
            vectors.push(vector);
            if mirror != k {
                vectors.push([radial, -vertical, -sin_qt]);
            }
        }
    }
    let trap_point = scene.trap_point();
    let trap_axis = scene.trap_axis();
    Some(
        vectors
            .into_iter()
            .map(|vector| match scene.coloring {
                Coloring::TrapOrigin => vector.iter().map(|v| v * v).sum::<f64>().sqrt(),
                Coloring::TrapPlane => {
                    let axis = trap_axis.min(2) as usize;
                    (vector[axis] - f64::from(trap_point[axis])).abs()
                }
                Coloring::TrapPoint => (0..3)
                    .map(|axis| (vector[axis] - f64::from(trap_point[axis])).powi(2))
                    .sum::<f64>()
                    .sqrt(),
                Coloring::Radius => unreachable!(),
            })
            .collect(),
    )
}

/// Acceptance covers full-iteration geometry, traps at fixed short orbit lengths1..3,
/// and full traps of noniterating formulas. Full iterative color equivalence is NOT
/// claimed: the separate explicitly ignored precision diagnostic retains that hypothesis.
/// Numerical tolerances are identical to the original probe, without point allowances.
#[test]
fn cuda_extended_full_de_short_orbits_and_noniterating_traps_match_wgpu() {
    let _gpu_test = crate::test_gpu::lock();
    let mut gpu = new_gpu().unwrap();
    let mut failures = Vec::new();
    for meta in crate::formulas::FORMULAS
        .iter()
        .filter(|meta| meta.code >= 8)
    {
        let formula = Formula::PRESETS[meta.code as usize];
        let points = formula_probe_points(formula.framing_radius());
        for (name, scene) in formula_probe_cases(formula) {
            let mut stages = vec![("full".to_string(), scene.clone(), !meta.iterates)];
            if meta.iterates {
                for iterations in 1..=3 {
                    let mut short = scene.clone();
                    short.render.iterations = iterations as f32;
                    stages.push((format!("short-{iterations}"), short, true));
                }
            }
            for (stage, mut scene, compare_trap) in stages {
                scene.render.backend = Backend::Cuda;
                let cuda = gpu.probe_formula(&scene, &points).unwrap();
                scene.render.backend = Backend::Wgpu;
                let wgpu = gpu.probe_formula(&scene, &points).unwrap();
                assert_eq!(cuda.len(), points.len());
                assert_eq!(wgpu.len(), points.len());
                let mut maximum = [0.0f32; 2];
                let mut beyond = [0usize; 2];
                let mut worst = [0usize; 2];
                let mut branch_sets = 0;
                for (index, (a, b)) in cuda.iter().zip(&wgpu).enumerate() {
                    let errors = [(a[0] - b[0]).abs(), (a[1] - b[1]).abs()];
                    for channel in 0..2 {
                        if errors[channel] > maximum[channel] {
                            maximum[channel] = errors[channel];
                            worst[channel] = index;
                        }
                    }
                    if !a[0].is_finite()
                        || !b[0].is_finite()
                        || errors[0] > 5.0e-5 + 2.0e-4 * b[0].abs()
                    {
                        beyond[0] += 1;
                    }
                    let valid_traps =
                        a[1].is_finite() && b[1].is_finite() && a[1] >= 0.0 && b[1] >= 0.0;
                    if !valid_traps {
                        beyond[1] += 1;
                    } else if compare_trap {
                        if let Some(expected) = knot_axis_minimum_traps(&scene, points[index]) {
                            branch_sets += 1;
                            for (backend, value) in [(Backend::Cuda, a[1]), (Backend::Wgpu, b[1])] {
                                if !expected.iter().any(|trap| {
                                    (f64::from(value) - trap).abs() <= 1.0e-4 + 2.0e-4 * trap.abs()
                                }) {
                                    beyond[1] += 1;
                                    failures.push(format!(
                                        "family={} case={name} stage={stage} backend={backend:?} point={:?} trap={value} nearest_branch_traps={expected:?}",
                                        meta.code, points[index]
                                    ));
                                }
                            }
                        } else if errors[1] > 1.0e-4 + 2.0e-4 * b[1].abs() {
                            beyond[1] += 1;
                        }
                    }
                }
                eprintln!(
                    "{}",
                    serde_json::json!({
                        "check": "full-de-short-orbits-noniterating-traps", "family": meta.code,
                        "case": name, "stage": stage, "points": points.len(),
                        "full_trap_comparison": compare_trap,
                        "uncompared_iterative_traps": if compare_trap {0} else {points.len()},
                        "analytic_closest_branch_sets": branch_sets,
                        "max_observed_error": maximum, "beyond": beyond,
                        "worst_points": worst.map(|i| points[i]),
                        "worst_cuda": worst.map(|i| cuda[i]),
                        "worst_wgpu": worst.map(|i| wgpu[i]),
                    })
                );
                if beyond != [0, 0] {
                    failures.push(format!(
                        "family={} case={name} stage={stage} beyond={beyond:?} max={maximum:?}",
                        meta.code
                    ));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "Geometry/short-orbit/noniterating-trap failures:\n{}",
        failures.join("\n")
    );
}

/// Precision diagnostic, including full-orbit colors: the original strict hypothesis is
/// retained unchanged. Measured one-ULP input changes can alter long-orbit traps by order1;
/// this diagnostic does not define a full-color equivalence acceptance contract.
#[test]
#[ignore = "Full-orbit precision diagnostic; long iterative traps are ill-conditioned. Run explicitly with --ignored --nocapture"]
fn cuda_extended_formula_estimates_and_traps_match_wgpu() {
    let _gpu_test = crate::test_gpu::lock();
    let mut gpu = new_gpu().unwrap();
    let mut failures = Vec::new();
    for code in 8..Formula::COUNT {
        let formula = Formula::PRESETS[code as usize];
        let points = formula_probe_points(formula.framing_radius());
        let cases = formula_probe_cases(formula);
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
    let mut gpu = new_gpu().unwrap();
    for meta in &crate::formulas::FORMULAS {
        if meta.code < 8 || meta.iterates {
            continue;
        }
        let formula = Formula::PRESETS[meta.code as usize];
        let points = formula_probe_points(formula.framing_radius());
        for backend in [Backend::Cuda, Backend::Wgpu] {
            let mut scene = quiet(Scene::preset_fractal(formula));
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
    let mut gpu = new_gpu().unwrap();
    let mut failures = Vec::new();
    for code in 8..Formula::COUNT {
        for modified in [false, true] {
            let mut scene = quiet(Scene::preset_fractal(Formula::PRESETS[code as usize]));
            if modified {
                scene.render.iterations = 7.375;
                scene.set_inversion_strength(0.12);
                scene.set_inversion_center([0.07, -0.03, 0.05]);
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
                + 32.0 * f32::EPSILON * selected.formula().framing_radius();
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
    let _gpu_test = crate::test_gpu::lock();
    let mut gpu = new_gpu().unwrap();
    let mut failures = Vec::new();
    for code in 8..Formula::COUNT {
        for shade in ["fast", "full", "mixed", "direct"] {
            let mut scene = quiet(Scene::preset_fractal(Formula::PRESETS[code as usize]));
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
            let radius = scene.formula().framing_radius();
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
    let _gpu_test = crate::test_gpu::lock();
    let mut gpu = new_gpu().unwrap();
    let mut failures = Vec::new();
    for code in 8..Formula::COUNT {
        let mut scene = quiet(Scene::preset_fractal(Formula::PRESETS[code as usize]));
        let radius = scene.formula().framing_radius();
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
        classic.set_formula(Formula::PRESETS[0]);
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
    let mut gpu = new_gpu().unwrap();
    let trap_point = [0.13f32, -0.21, 0.17];
    let mut scene = quiet(Scene::preset_fractal(Formula::MobiusIfs(MobiusIfs {
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
    scene.set_trap_point(trap_point);
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
        cases.push(Scene::preset_fractal(Formula::PhyllotaxisShell(
            PhyllotaxisShell {
                count: half_count,
                ..PhyllotaxisShell::PRESET
            },
        )));
        cases.push(Scene::preset_fractal(Formula::AstrolabeGimbal(
            AstrolabeGimbal {
                gear_teeth: half_count,
                ..AstrolabeGimbal::PRESET
            },
        )));
    }
    cases.push(Scene::preset_fractal(Formula::HopfFibration(
        HopfFibration {
            fibers: 2,
            shells: 1,
            twist: 0.0,
            scale: 1.0,
            ..HopfFibration::PRESET
        },
    )));
    for mut scene in cases {
        let mut points = formula_probe_points(scene.formula().framing_radius());
        // xi1=±pi/2 with two fibers gives an exact ±half-sector phase in Hopf's field.
        points.extend([[0.0, 1.0, 0.3, 0.0], [0.0, -1.0, 0.3, 0.0]]);
        scene.coloring = Coloring::TrapPoint;
        scene.set_trap_point(trap_point);
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
                    scene.formula().code(),
                    points[index],
                );
            }
        }
    }
}

/// The exact input and its six adjacent f32 coordinates, without decimal re-quantization.
fn orbit_probe_ulp_neighbours(anchor: [f32; 4]) -> [[f32; 4]; 7] {
    let mut points = [anchor; 7];
    for axis in 0..3 {
        points[1 + axis * 2][axis] = anchor[axis].next_down();
        points[2 + axis * 2][axis] = anchor[axis].next_up();
    }
    points
}

/// Attribute orbit divergence before changing either arithmetic or parity thresholds.
/// Each line preserves both backends' full answers and bits at one iteration endpoint.
#[test]
fn cuda_extended_orbit_rounding_diagnostic() {
    use crate::params::*;

    let _gpu_test = crate::test_gpu::lock();
    let mut gpu = new_gpu().unwrap();
    let mut nonfinite = Vec::new();
    for (family, anchor) in [
        (
            FAMILY_MOBIUS_IFS,
            [0xbfab_25ea, 0x3eac_a960, 0xbf42_d1c7, 0].map(f32::from_bits),
        ),
        (
            FAMILY_AMAZING_SURFACE,
            [0x3de5_d19d, 0xbe3e_a14c, 0xbeca_40fb, 0].map(f32::from_bits),
        ),
    ] {
        let base = quiet(Scene::preset_fractal(Formula::PRESETS[family as usize]));
        let points = orbit_probe_ulp_neighbours(anchor);
        let ablations: &[(&str, bool, bool)] = if family == FAMILY_MOBIUS_IFS {
            &[
                ("preset", false, false),
                ("rotation-zero", true, false),
                ("twist-zero", false, true),
                ("rotation-and-twist-zero", true, true),
            ]
        } else {
            &[("preset", false, false), ("rotation-zero", true, false)]
        };
        for &(case, rotation_zero, twist_zero) in ablations {
            let mut scene = base.clone();
            scene.coloring = Coloring::TrapOrigin;
            scene.modify_formula(|f| match f {
                Formula::MobiusIfs(formula) => {
                    if rotation_zero {
                        formula.rotation_degrees = [0.0; 3];
                    }
                    if twist_zero {
                        formula.twist = 0.0;
                    }
                }
                Formula::AmazingSurface(formula) => {
                    if rotation_zero {
                        formula.rotation_degrees = [0.0; 3];
                    }
                }
                _ => unreachable!(),
            });
            let packed = scene.pack(1, 1);
            eprintln!(
                "{}",
                serde_json::json!({
                    "check": "orbit-rounding-input",
                    "family": family,
                    "case": case,
                    "points": points,
                    "point_bits": points.map(|point| point.map(f32::to_bits)),
                    "formula_slots": &packed[P_BULB_POWER..P_COLOR_MODE],
                    "rotation_enabled": packed[P_ITER_ROTATE],
                    "rotation_rows": &packed[P_ITER_ROT..P_ITER_ROT + 9],
                    "julia_mix": packed[P_JULIA],
                    "julia": &packed[P_JULIA_C..P_JULIA_C + 3],
                    "orbit_offset": &packed[P_ORBIT_OFFSET..P_ORBIT_OFFSET + 3],
                    "inversion_strength": packed[P_INVERT_STRENGTH],
                    "color_mode": packed[P_COLOR_MODE],
                    "trap_point": &packed[P_TRAP_POINT..P_TRAP_POINT + 3],
                    "trap_normal": &packed[P_TRAP_NORMAL..P_TRAP_NORMAL + 3],
                })
            );
            for iterations in 1..=16 {
                scene.render.iterations = iterations as f32;
                scene.render.backend = Backend::Cuda;
                let cuda = gpu.probe_formula(&scene, &points).unwrap();
                scene.render.backend = Backend::Wgpu;
                let wgpu = gpu.probe_formula(&scene, &points).unwrap();
                assert_eq!(cuda.len(), points.len());
                assert_eq!(wgpu.len(), points.len());
                let delta: Vec<[f32; 2]> = cuda
                    .iter()
                    .zip(&wgpu)
                    .map(|(a, b)| [(a[0] - b[0]).abs(), (a[1] - b[1]).abs()])
                    .collect();
                let trap_range = |values: &[[f32; 4]]| {
                    values.iter().fold([f32::MAX, -f32::MAX], |range, row| {
                        [range[0].min(row[1]), range[1].max(row[1])]
                    })
                };
                eprintln!(
                    "{}",
                    serde_json::json!({
                        "check": "orbit-rounding-endpoint",
                        "family": family,
                        "case": case,
                        "iterations": iterations,
                        "cuda": cuda,
                        "wgpu": wgpu,
                        "cuda_bits": cuda.iter().map(|row| row.map(f32::to_bits)).collect::<Vec<_>>(),
                        "wgpu_bits": wgpu.iter().map(|row| row.map(f32::to_bits)).collect::<Vec<_>>(),
                        "absolute_difference": delta,
                        "cuda_neighbour_trap_range": trap_range(&cuda),
                        "wgpu_neighbour_trap_range": trap_range(&wgpu),
                    })
                );
                for (backend, values) in [(Backend::Cuda, &cuda), (Backend::Wgpu, &wgpu)] {
                    for (index, row) in values.iter().enumerate() {
                        if row.iter().any(|value| !value.is_finite()) {
                            nonfinite.push(format!(
                                "family={family} case={case} iterations={iterations} backend={backend:?} point={index} value={row:?}"
                            ));
                        }
                    }
                }
            }
        }
    }
    assert!(
        nonfinite.is_empty(),
        "Nonfinite orbit diagnostics:\n{}",
        nonfinite.join("\n")
    );
}

/// Targeted numerical attribution, separate from parity acceptance: output intermediate
/// one-iteration trap projections and exact failing points without changing any tolerance.
#[test]
#[ignore = "GPU axis/branch attribution; run explicitly with --ignored --nocapture"]
fn cuda_extended_axis_and_nearest_branch_diagnostic() {
    use crate::formulas::{HarmonicBulb, PolyhedralKifs, Superformula3D};
    let _gpu_test = crate::test_gpu::lock();
    let mut gpu = new_gpu().unwrap();
    let harmonic_q: [f32; 4] = [0.0, 0.0, -0.69, 0.0];
    let mut harmonic_points = vec![harmonic_q];
    for axis in 0..3 {
        for direction in [f32::NEG_INFINITY, f32::INFINITY] {
            let mut point = harmonic_q;
            point[axis] = if direction.is_sign_negative() {
                point[axis].next_down()
            } else {
                point[axis].next_up()
            };
            harmonic_points.push(point);
        }
    }
    for iterations in 1..=14 {
        let mut scene = quiet(Scene::preset_fractal(Formula::HarmonicBulb(
            HarmonicBulb::PRESET,
        )));
        scene.render.iterations = iterations as f32;
        for axis in 0..3 {
            scene.coloring = Coloring::TrapPlane;
            scene.set_trap_axis(axis);
            scene.set_trap_point([1.0; 3]);
            for backend in [Backend::Cuda, Backend::Wgpu] {
                scene.render.backend = backend;
                let values = gpu.probe_formula(&scene, &harmonic_points).unwrap();
                assert!(values.iter().flatten().all(|value| value.is_finite()));
                eprintln!(
                    "{}",
                    serde_json::json!({
                        "check": "harmonic-axis-attribution", "iterations": iterations,
                        "axis": axis, "backend": backend.label(), "points": harmonic_points,
                        "values": values, "one_minus_trap": values[0][1].mul_add(-1.0, 1.0),
                    })
                );
            }
        }
    }
    let mut superformula = quiet(Scene::preset_fractal(Formula::Superformula3D(
        Superformula3D {
            pore_frequency: 7.4,
            ..Superformula3D::PRESET
        },
    )));
    let super_points = [[0.0, 0.0, -0.99, 0.0], [0.0, 0.0, 0.99, 0.0]];
    let frequency = f64::from(7.4f32);
    let exact_pole_pore = ((frequency * std::f64::consts::FRAC_PI_2).cos() - 0.45) / frequency;
    for backend in [Backend::Cuda, Backend::Wgpu] {
        superformula.render.backend = backend;
        eprintln!(
            "{}",
            serde_json::json!({
                "check": "superformula-pole-attribution", "backend": backend.label(),
                "points": super_points, "closed_form_pole_pore": exact_pole_pore,
                "values": gpu.probe_formula(&superformula, &super_points).unwrap(),
            })
        );
    }
    let mut knot = quiet(Scene::preset_fractal(Formula::PRESETS[19]));
    knot.coloring = Coloring::TrapPoint;
    knot.set_trap_point([0.13, -0.21, 0.17]);
    let knot_points: [[f32; 4]; 6] = [
        [-0.87000006, 0.0, 0.0, 0.0],
        [-0.87000006, -0.0, 0.0, 0.0],
        [-0.87000006, 1.0e-7, 0.0, 0.0],
        [-0.87000006, -1.0e-7, 0.0, 0.0],
        [-0.87000006, 0.0, 1.0e-7, 0.0],
        [-0.87000006, 0.0, -1.0e-7, 0.0],
    ];
    for backend in [Backend::Cuda, Backend::Wgpu] {
        knot.render.backend = backend;
        eprintln!(
            "{}",
            serde_json::json!({
                "check": "knot-symmetric-branch-attribution", "backend": backend.label(),
                "points": knot_points, "point_bits": knot_points.map(|p| p.map(f32::to_bits)),
                "values": gpu.probe_formula(&knot, &knot_points).unwrap(),
            })
        );
    }
    for offset in [[0.8506508, 0.5257311, 0.0], [0.5257311, 0.0, 0.8506508]] {
        let scene = quiet(Scene::preset_fractal(Formula::PolyhedralKifs(
            PolyhedralKifs {
                offset,
                ..PolyhedralKifs::PRESET
            },
        )));
        for backend in [Backend::Cuda, Backend::Wgpu] {
            let rendered = guides(&mut gpu, &scene, backend);
            eprintln!(
                "{}",
                serde_json::json!({
                    "check": "polyhedral-offset-attribution", "backend": backend.label(),
                    "offset": offset, "coverage": coverage(&rendered),
                })
            );
        }
    }
}

/// Independent forward construction: no inverse lattice basis or four-candidate search.
/// Quantize the authored golden angle and phase multiplication to the scene/device f32 ABI,
/// then evaluate the unit-sphere geometry in f64 for an O(n) nearest-neighbour oracle.
fn fibonacci_reference_seed(index: usize, count: usize) -> [f64; 3] {
    let z = f64::from(1.0 - (2 * index + 1) as f32 / count as f32);
    let radius = (1.0 - z * z).max(0.0).sqrt();
    let phase = f64::from(2.399_963_1f32 * index as f32);
    [radius * phase.cos(), radius * phase.sin(), z]
}

fn fibonacci_nearest_reference(point: [f32; 4], seeds: &[[f64; 3]]) -> (usize, f64, f64) {
    let radius = point[..3]
        .iter()
        .map(|&x| f64::from(x).powi(2))
        .sum::<f64>()
        .sqrt();
    let direction: [f64; 3] = std::array::from_fn(|axis| f64::from(point[axis]) / radius);
    let mut nearest = (0, f64::INFINITY, f64::INFINITY);
    for (index, seed) in seeds.iter().enumerate() {
        let distance = (0..3)
            .map(|axis| (direction[axis] - seed[axis]).powi(2))
            .sum::<f64>();
        if distance < nearest.1 {
            nearest = (index, distance, nearest.1);
        } else {
            nearest.2 = nearest.2.min(distance);
        }
    }
    nearest
}

/// Actual production GPU nearest seed against an independent O(n) oracle. Every authored
/// seed is queried, alongside off-grid directions, poles, and the original failing point.
/// Three displaced trap planes expose the selected seed vector without a diagnostic kernel.
#[test]
fn cuda_phyllotaxis_nearest_seed_matches_brute_force_on_both_backends() {
    use crate::formulas::PhyllotaxisShell;
    let _gpu_test = crate::test_gpu::lock();
    let mut gpu = new_gpu().unwrap();
    let mut failures = Vec::new();
    for count in [
        8.0f32, 12.0, 34.0, 132.0, 144.0, 300.0, 600.0, 8.5, 10.5, 132.5,
    ] {
        let n = count.round_ties_even().max(8.0) as usize;
        let seeds: Vec<_> = (0..n).map(|i| fibonacci_reference_seed(i, n)).collect();
        let mut points: Vec<[f32; 4]> = seeds
            .iter()
            .map(|seed| [seed[0] as f32, seed[1] as f32, seed[2] as f32, 0.0])
            .collect();
        for latitude in 0..7 {
            let z = 2.0 * (latitude as f64 + 0.37) / 7.0 - 1.0;
            let radius = (1.0 - z * z).sqrt();
            for longitude in 0..11 {
                let phi = std::f64::consts::TAU * (longitude as f64 + 0.19) / 11.0;
                points.push([
                    (radius * phi.cos()) as f32,
                    (radius * phi.sin()) as f32,
                    z as f32,
                    0.0,
                ]);
            }
        }
        points.extend([
            [1.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [-0.324_094_62, -0.112_472_974, -0.238_662_15, 0.0],
        ]);
        let expected: Vec<_> = points
            .iter()
            .map(|&p| fibonacci_nearest_reference(p, &seeds))
            .collect();
        // Each seed has a unique nearest neighbour: itself. This also checks the oracle's
        // normalization and the f32 query conversion independently of the inverse formula.
        for (index, nearest) in expected.iter().take(n).enumerate() {
            assert_eq!(nearest.0, index, "forward seed oracle n={n} index={index}");
            assert!(nearest.2 > nearest.1);
        }
        for backend in [Backend::Cuda, Backend::Wgpu] {
            let mut scene = quiet(Scene::preset_fractal(Formula::PhyllotaxisShell(
                PhyllotaxisShell {
                    count,
                    sub_florets: 0.0,
                    ..PhyllotaxisShell::PRESET
                },
            )));
            scene.render.backend = backend;
            scene.coloring = Coloring::TrapPlane;
            scene.set_trap_point([3.0; 3]);
            let mut observed = vec![[0.0f64; 3]; points.len()];
            for axis in 0..3 {
                scene.set_trap_axis(axis);
                let values = gpu.probe_formula(&scene, &points).unwrap();
                for (selected, value) in observed.iter_mut().zip(values) {
                    assert!(value.iter().all(|v| v.is_finite()));
                    selected[axis as usize] = 3.0 - f64::from(value[1]);
                }
            }
            let mut mismatches = 0;
            for (index, selected) in observed.iter().enumerate() {
                let norm = selected.iter().map(|x| x * x).sum::<f64>().sqrt();
                assert!(norm > 0.0 && norm.is_finite());
                let recovered = ((n as f64 * (1.0 - selected[2] / norm) - 1.0) * 0.5)
                    .round()
                    .clamp(0.0, (n - 1) as f64) as usize;
                let reference = expected[index];
                // Index equality is exact. Only a mathematically exact f64 distance tie
                // permits another index; there is no floating mismatch allowance.
                let chosen = fibonacci_nearest_reference(points[index], &[seeds[recovered]]).1;
                if recovered != reference.0 && chosen != reference.1 {
                    mismatches += 1;
                    failures.push(format!(
                        "backend={backend:?} count={count} point={:?} expected={} selected={recovered} vector={selected:?} d2={} selected_d2={chosen} runner_up_d2={}",
                        points[index], reference.0, reference.1, reference.2
                    ));
                }
            }
            eprintln!(
                "{}",
                serde_json::json!({
                    "check": "phyllotaxis-brute-force-nearest", "backend": backend.label(),
                    "authored_count": count, "realized_count": n, "queries": points.len(),
                    "seed_queries": n, "mismatches": mismatches,
                    "smallest_runner_up_gap_squared": expected.iter().map(|v| v.2 - v.1).fold(f64::INFINITY, f64::min),
                })
            );
        }
    }
    assert!(
        failures.is_empty(),
        "Nearest-seed failures:\n{}",
        failures.join("\n")
    );
}

#[test]
fn cuda_extended_pole_estimates_match_analytic_geometry_on_both_backends() {
    use crate::formulas::{HarmonicBulb, Superformula3D};
    let _gpu_test = crate::test_gpu::lock();
    let mut gpu = new_gpu().unwrap();
    for backend in [Backend::Cuda, Backend::Wgpu] {
        let harmonic = HarmonicBulb::PRESET;
        let points = [[0.0, 0.0, -0.69, 0.0], [0.0, 0.0, 0.69, 0.0]];
        let mut scene = quiet(Scene::preset_fractal(Formula::HarmonicBulb(harmonic)));
        scene.render.backend = backend;
        scene.render.iterations = 1.0;
        scene.coloring = Coloring::TrapOrigin;
        let values = gpu.probe_formula(&scene, &points).unwrap();
        for (point, value) in points.iter().zip(values) {
            // power=7, l=4, phi=0: both exact poles stay on their signed axis after
            // one iteration, with magnitude r^7 * (1 + amplitude) + r.
            let radius = f64::from(point[2].abs());
            let expected = radius.powf(f64::from(harmonic.power))
                * (1.0 + f64::from(harmonic.amplitude))
                + radius;
            let error = (f64::from(value[1]) - expected).abs();
            assert!(
                error <= 1.0e-4 + 2.0e-4 * expected,
                "{backend:?} Harmonic pole {point:?}: {value:?}, expected trap {expected}"
            );
        }
        for frequency in [7.4f32, 9.25, 11.5] {
            let mut scene = quiet(Scene::preset_fractal(Formula::Superformula3D(
                Superformula3D {
                    pore_frequency: frequency,
                    ..Superformula3D::PRESET
                },
            )));
            scene.render.backend = backend;
            let points = [[0.0, 0.0, -0.99, 0.0], [0.0, 0.0, 0.99, 0.0]];
            let values = gpu.probe_formula(&scene, &points).unwrap();
            let f = f64::from(frequency);
            let pore = ((f * std::f64::consts::FRAC_PI_2).cos() - 0.45) / f;
            // Both Gielis radii equal1 at these poles. Evaluate the geometric shell
            // and pore independently; these frequencies exercise both max arms.
            let p = Superformula3D::PRESET;
            let shell = ((f64::from(0.99f32) - 1.0) / (1.0 + 0.12 * f64::from(p.m1 + p.m2))).abs()
                - f64::from(p.shell_thickness);
            let expected = shell.max(pore);
            for (point, value) in points.iter().zip(values) {
                let error = (f64::from(value[0]) - expected).abs();
                assert!(
                    error <= 5.0e-5 + 2.0e-4 * expected.abs(),
                    "{backend:?} Superformula pole {point:?} f={frequency}: {value:?}, expected {expected}"
                );
            }
        }
    }
}

#[test]
fn cuda_polyhedral_preset_has_visible_surface_on_both_backends() {
    let _gpu_test = crate::test_gpu::lock();
    let mut gpu = new_gpu().unwrap();
    let scene = quiet(Scene::preset_fractal(Formula::PRESETS[12]));
    for backend in [Backend::Cuda, Backend::Wgpu] {
        let value = coverage(&guides(&mut gpu, &scene, backend));
        // The corrected folded vertex measured .103841 coverage on both backends;
        // retain the existing primary-guide minimum rather than accepting an empty oracle.
        assert!(
            value >= 0.02,
            "{backend:?} Polyhedral preset coverage {value}"
        );
    }
}

/// A normal stencil's signal versus one-coordinate-ULP changes at its six endpoints.
fn orbit_normal_probe_report(values: &[[f32; 4]]) -> serde_json::Value {
    assert_eq!(values.len(), 49);
    let gradient: [f32; 3] =
        std::array::from_fn(|axis| values[(1 + 2 * axis) * 7][0] - values[(2 + 2 * axis) * 7][0]);
    let scale = gradient.iter().map(|v| v.abs()).fold(0.0, f32::max);
    let normal = if scale > 0.0 {
        let g = gradient.map(|v| v / scale);
        let inverse = 1.0 / (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
        Some(g.map(|v| v * inverse))
    } else {
        None
    };
    let endpoint_spans: Vec<f64> = values
        .as_chunks::<7>()
        .0
        .iter()
        .map(|samples| {
            let low = samples.iter().map(|v| v[0]).fold(f32::INFINITY, f32::min);
            let high = samples
                .iter()
                .map(|v| v[0])
                .fold(f32::NEG_INFINITY, f32::max);
            f64::from(high) - f64::from(low)
        })
        .collect();
    let signal = gradient
        .map(f64::from)
        .iter()
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt();
    let uncertainty = (0..3)
        .map(|axis| (endpoint_spans[1 + 2 * axis] + endpoint_spans[2 + 2 * axis]).powi(2))
        .sum::<f64>()
        .sqrt();
    serde_json::json!({
        "center": values[0],
        "stencil": (1..7).map(|index| values[index * 7]).collect::<Vec<_>>(),
        "gradient": gradient,
        "normal": normal,
        "gradient_norm": signal,
        "endpoint_ulp_spans": endpoint_spans,
        "ulp_span_over_gradient": (signal > 0.0).then_some(uncertainty / signal),
    })
}

/// Diagnose the existing Mobius preset at a measured worst normal pixel, then at a common
/// CUDA pick contact on that pixel's Sobol ray. This reports conditioning, not parity success.
#[test]
fn cuda_mobius_primary_normal_conditioning_diagnostic() {
    use crate::params::*;
    let _gpu_test = crate::test_gpu::lock();
    let mut gpu = new_gpu().unwrap();
    let scene = quiet(Scene::preset_fractal(
        Formula::PRESETS[FAMILY_MOBIUS_IFS as usize],
    ));
    // The object-local probe contract has no authored/world transform.
    assert_eq!(scene.object.offset, [0.0; 3]);
    assert_eq!(scene.object.rotation_degrees, [0.0; 3]);
    assert_eq!(scene.object.scale, 1.0);
    assert!(scene.masks.is_empty());
    assert_eq!(scene.inversion_strength(), 0.0);
    let packed = scene.pack(WIDTH as u32, HEIGHT as u32);
    assert_eq!(packed[P_APERTURE], 0.0);
    let cuda = guides(&mut gpu, &scene, Backend::Cuda);
    let wgpu = guides(&mut gpu, &scene, Backend::Wgpu);
    let pixel = (0..WIDTH * HEIGHT)
        .filter(|&i| {
            [cuda.normal[i], wgpu.normal[i]].iter().all(|n| {
                n[..3].iter().all(|v| v.is_finite())
                    && n[0] * n[0] + n[1] * n[1] + n[2] * n[2] > 0.25
            })
        })
        .max_by(|&a, &b| {
            channel_error(cuda.normal[a], wgpu.normal[a])
                .total_cmp(&channel_error(cuda.normal[b], wgpu.normal[b]))
        })
        .expect("Mobius has no common primary hit");
    let (x, y) = (pixel % WIDTH, pixel / WIDTH);
    let jitter: [f32; 2] = std::array::from_fn(|dimension| {
        standard_surface_bsdf::sampling::sobol(x as u32 ^ SEED, y as u32, 0, dimension as u32)
    });
    let ndc = [
        2.0 * (x as f32 + jitter[0]) / WIDTH as f32 - 1.0,
        1.0 - 2.0 * (y as f32 + jitter[1]) / HEIGHT as f32,
    ];
    let v3 = |slot| [packed[slot], packed[slot + 1], packed[slot + 2]];
    let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let origin = v3(P_CAM_ORIGIN);
    let forward = v3(P_CAM_FORWARD);
    let right = v3(P_CAM_RIGHT);
    let up = v3(P_CAM_UP);
    // Same nested add/multiply order as trace_pixel and pick_point; GPU contraction can
    // still change the last bit. Raw coordinates plus ULP neighbours remain in the receipt.
    let direction: [f32; 3] = std::array::from_fn(|axis| {
        forward[axis]
            + (right[axis] * (ndc[0] * packed[P_HALF_W]) + up[axis] * (ndc[1] * packed[P_HALF_H]))
    });
    let inverse_length = 1.0 / dot(direction, direction).sqrt();
    let direction = direction.map(|v| v * inverse_length);
    let mut pick_scene = scene.clone();
    pick_scene.world_render = true;
    pick_scene.objects = vec![scene.clone()];
    let upload = WorldUpload::new(&pick_scene, WIDTH as u32, HEIGHT as u32).unwrap();
    assert_eq!(upload.objects[O_DISTANCE_SCALE], 1.0);
    assert_eq!(
        &upload.objects[O_INVERSE..O_INVERSE + 12],
        &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    );
    for slot in [P_CAM_ORIGIN, P_CAM_FORWARD, P_CAM_RIGHT, P_CAM_UP] {
        assert_eq!(&upload.params[slot..slot + 3], &packed[slot..slot + 3]);
    }
    let mut depths = [0.0; 2];
    for (i, backend) in [Backend::Cuda, Backend::Wgpu].into_iter().enumerate() {
        pick_scene.render.backend = backend;
        let (object, depth) = gpu
            .pick(&pick_scene, WIDTH, HEIGHT, ndc)
            .unwrap()
            .expect("The measured Mobius sample ray must pick its object");
        assert_eq!(object, 0);
        assert!(depth.is_finite() && depth > 0.0);
        depths[i] = depth;
    }
    // pick returns view-axis depth, not ray length. Its contact is reconstructed from
    // depth = t * dot(dir, forward), as defined by the device pick_point implementation.
    let distance = depths[0] / dot(direction, forward);
    let point: [f32; 3] = std::array::from_fn(|axis| origin[axis] + direction[axis] * distance);
    let point_length = dot(point, point).sqrt();
    // footprint(0, P_FOOTPRINT, t, x), then endpoint_normal's signed-family quarter step.
    // FOOTPRINT_RELATIVE_FLOOR in fractal-kernels is exactly 16 * f32::EPSILON.
    let eps = (packed[P_FOOTPRINT] * distance).max(point_length * (16.0 * f32::EPSILON));
    let h = 0.25 * eps;
    assert!(h.is_finite() && h > 0.0);
    let mut centers = vec![[point[0], point[1], point[2], 0.0]];
    for axis in 0..3 {
        for sign in [1.0, -1.0] {
            let mut endpoint = centers[0];
            endpoint[axis] += sign * h;
            centers.push(endpoint);
        }
    }
    let queries: Vec<[f32; 4]> = centers
        .iter()
        .flat_map(|&p| orbit_probe_ulp_neighbours(p))
        .collect();
    let agreement = compare(&cuda, &wgpu);
    eprintln!(
        "{}",
        serde_json::json!({
            "check": "mobius-primary-normal-input", "pixel": [x, y], "seed": SEED,
            "iterations": scene.render.iterations, "guide_cuda": cuda.normal[pixel],
            "guide_wgpu": wgpu.normal[pixel], "guide_normal_max": agreement.normal_max,
            "guide_mismatch": agreement.mismatch, "jitter": jitter, "ndc": ndc,
            "origin": origin, "direction": direction, "pick_view_depths": depths,
            "cuda_pick_ray_distance": distance, "common_pick_point": point,
            "epsilon": eps, "h": h, "footprint_slope": packed[P_FOOTPRINT],
            "queries": queries, "query_bits": queries.iter().map(|q| q.map(f32::to_bits)).collect::<Vec<_>>(),
            "contact_contract": "same Sobol sample ray; CUDA world-pick contact reconstructed on CPU; pick cone starts on the sample ray, render cone starts on the pixel center; not the exact render contact",
        })
    );
    for iterations in 1..=16 {
        let mut selected = scene.clone();
        selected.render.iterations = iterations as f32;
        selected.render.backend = Backend::Cuda;
        let c = gpu.probe_formula(&selected, &queries).unwrap();
        selected.render.backend = Backend::Wgpu;
        let w = gpu.probe_formula(&selected, &queries).unwrap();
        assert_eq!(c.len(), queries.len());
        assert_eq!(w.len(), queries.len());
        assert!(c.iter().chain(&w).flatten().all(|v| v.is_finite()));
        let error: [f64; 3] = std::array::from_fn(|axis| {
            let plus = (1 + 2 * axis) * 7;
            let minus = (2 + 2 * axis) * 7;
            (f64::from(c[plus][0]) - f64::from(c[minus][0]))
                - (f64::from(w[plus][0]) - f64::from(w[minus][0]))
        });
        eprintln!(
            "{}",
            serde_json::json!({
                "check": "mobius-primary-normal-endpoint", "iterations": iterations,
                "cuda": orbit_normal_probe_report(&c), "wgpu": orbit_normal_probe_report(&w),
                "gradient_delta": error, "cuda_raw": c, "wgpu_raw": w,
                "cuda_bits": c.iter().map(|q| q.map(f32::to_bits)).collect::<Vec<_>>(),
                "wgpu_bits": w.iter().map(|q| q.map(f32::to_bits)).collect::<Vec<_>>(),
            })
        );
    }
}

/// Independent f64 first-contact oracle for the beam and sampled camera rays.
fn beam_entry_reference(
    offset: [f32; 3],
    direction: [f32; 3],
    radius: f32,
    slope: f32,
) -> Option<f64> {
    let o = offset.map(f64::from);
    let d = direction.map(f64::from);
    let r = f64::from(radius);
    let s = f64::from(slope);
    let dot = |a: [f64; 3], b: [f64; 3]| a.into_iter().zip(b).map(|(a, b)| a * b).sum::<f64>();
    let c = dot(o, o) - r * r;
    if c <= 0.0 {
        return Some(0.0);
    }
    let a = dot(d, d) - s * s;
    let b = dot(o, d) - r * s;
    let disc = b * b - a * c;
    if disc < 0.0 {
        return None;
    }
    let denominator = disc.sqrt() - b;
    (denominator > 0.0).then(|| c / denominator)
}

#[test]
fn pixel_beam_start_fixes_recorded_mobius_surface_skip() {
    init_warpbro();
    use fractal_core::ray_bounds::beam_sphere_entry;
    // The actual 96x64, seed7, pixel(62,6) Mobius camera, packed by the host.
    let scene = Scene::preset_fractal(Formula::PRESETS[crate::params::FAMILY_MOBIUS_IFS as usize]);
    let p = scene.pack(WIDTH as u32, HEIGHT as u32);
    use crate::params::*;
    let origin = [p[P_CAM_ORIGIN], p[P_CAM_ORIGIN + 1], p[P_CAM_ORIGIN + 2]];
    let ray = |ndc: [f32; 2]| {
        let d: [f64; 3] = std::array::from_fn(|axis| {
            f64::from(p[P_CAM_FORWARD + axis])
                + f64::from(p[P_CAM_RIGHT + axis]) * f64::from(ndc[0]) * f64::from(p[P_HALF_W])
                + f64::from(p[P_CAM_UP + axis]) * f64::from(ndc[1]) * f64::from(p[P_HALF_H])
        });
        let length = d.iter().map(|v| v * v).sum::<f64>().sqrt();
        d.map(|v| (v / length) as f32)
    };
    let center = ray([
        2.0 * 62.5 / WIDTH as f32 - 1.0,
        1.0 - 2.0 * 6.5 / HEIGHT as f32,
    ]);
    let jitter: [f32; 2] = std::array::from_fn(|dim| {
        standard_surface_bsdf::sampling::sobol(62 ^ SEED, 6, 0, dim as u32)
    });
    let sample = ray([
        2.0 * (62.0 + jitter[0]) / WIDTH as f32 - 1.0,
        1.0 - 2.0 * (6.0 + jitter[1]) / HEIGHT as f32,
    ]);
    let radius = p[P_CLIP_RADIUS];
    assert_eq!(radius, 1.75);
    let basis =
        [P_CAM_FORWARD, P_CAM_RIGHT, P_CAM_UP].map(|slot| [p[slot], p[slot + 1], p[slot + 2]]);
    let slope = fractal_core::ray_bounds::pixel_beam_slope(
        basis,
        p[P_HALF_W],
        p[P_HALF_H],
        p[P_WIDTH],
        p[P_HEIGHT],
    );
    let old_start = beam_entry_reference(origin, center, radius, 0.0).unwrap();
    let sample_entry = beam_entry_reference(origin, sample, radius, 0.0).unwrap();
    assert!(old_start > sample_entry + 0.03);
    let skipped: [f64; 3] =
        std::array::from_fn(|i| f64::from(origin[i]) + old_start * f64::from(sample[i]));
    let skipped_radius = skipped.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(skipped_radius < f64::from(radius) - 0.01);
    let safe = beam_sphere_entry(origin, center, radius, slope);
    assert!(
        safe > 4.45 && safe < 4.46,
        "shortcut must retain useful progress"
    );
    assert!(f64::from(safe) <= beam_entry_reference(origin, center, radius, slope).unwrap());
    assert!(f64::from(safe) <= sample_entry);
}

#[test]
fn pixel_beam_start_precedes_all_sampled_corner_and_interior_ray_contacts() {
    use fractal_core::ray_bounds::beam_sphere_entry;
    let normalize = |v: [f64; 3]| {
        let length = v.iter().map(|v| v * v).sum::<f64>().sqrt();
        v.map(|v| (v / length) as f32)
    };
    let mut contacts = 0;
    // Independent intersections of camera sample rays across origins, image-plane
    // positions and pixel sizes. No tolerance is added to the lower-bound property.
    for offset in [
        [0.0, 0.0, 5.0],
        [2.923_996_4, 1.855_459_1, 4.175_899_5],
        [0.2, -0.1, 1.001],
    ] {
        for nx in [-1.5, -0.3, 0.0, 0.7, 1.5] {
            for ny in [-0.8, 0.0, 0.8] {
                for pixel_half in [0.000_01, 0.007_7, 0.3, 1.0, 2.0] {
                    let center = normalize([nx, ny, -1.0]);
                    let mut samples = Vec::new();
                    let mut separation = 0.0f64;
                    for dx in [-pixel_half, 0.0, pixel_half] {
                        for dy in [-pixel_half, 0.0, pixel_half] {
                            let d = normalize([nx + dx, ny + dy, -1.0]);
                            let delta: [f64; 3] =
                                std::array::from_fn(|i| f64::from(d[i]) - f64::from(center[i]));
                            separation =
                                separation.max(delta.iter().map(|v| v * v).sum::<f64>().sqrt());
                            samples.push(d);
                        }
                    }
                    // This measured bundle radius includes float ray-normalization error.
                    let slope = (separation as f32).next_up();
                    let safe = beam_sphere_entry(offset, center, 1.0, slope);
                    assert!(safe.is_finite() && safe >= 0.0);
                    for sample in samples {
                        if let Some(entry) = beam_entry_reference(offset, sample, 1.0, 0.0) {
                            assert!(
                                f64::from(safe) <= entry,
                                "offset={offset:?} center={center:?} slope={slope} sample={sample:?} safe={safe} entry={entry}"
                            );
                            contacts += 1;
                        }
                    }
                }
            }
        }
    }
    assert!(contacts > 500, "fixtures must intersect the sphere");
}

#[test]
fn pixel_beam_lower_root_covers_tangency_inside_wide_beams_and_extreme_scales() {
    use fractal_core::ray_bounds::beam_sphere_entry;
    let mut positive = 0;
    for scale in [1.0e-12f32, 1.0e-4, 1.0, 1.0e4, 1.0e12] {
        for offset in [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, scale],
            [0.0, 0.0, 5.0 * scale],
            [scale, 0.0, 3.0 * scale],
            [scale.next_down(), 0.0, 3.0 * scale],
            [scale.next_up(), 0.0, 3.0 * scale],
        ] {
            for slope in [
                0.0,
                1.0e-20,
                0.001,
                0.5,
                0.999_999_94,
                1.0,
                1.000_000_1,
                4.0,
                1.0e10,
            ] {
                let dir = [0.0, 0.0, -1.0];
                let safe = beam_sphere_entry(offset, dir, scale, slope);
                assert!(safe.is_finite() && safe >= 0.0);
                if let Some(reference) = beam_entry_reference(offset, dir, scale, slope) {
                    assert!(
                        f64::from(safe) <= reference,
                        "offset={offset:?} radius={scale} slope={slope} safe={safe} reference={reference}"
                    );
                }
                positive += usize::from(safe > 0.0);
            }
        }
    }
    assert!(positive > 100, "finite fixtures must retain the shortcut");
}

#[test]
fn pixel_beam_slope_encloses_float_camera_corner_directions() {
    init_warpbro();
    use crate::params::*;
    use fractal_core::ray_bounds::pixel_beam_slope;
    for fov in [0.01, 1.0, 38.0, 120.0, 179.0] {
        for (width, height) in [(3, 3), (96, 64), (16_384, 8_192)] {
            for [yaw, pitch, roll] in [[0.0, 0.0, 0.0], [35.0, 20.0, 0.0], [179.0, 89.0, 172.0]] {
                let mut scene = Scene::preset_fractal(Formula::PRESETS[FAMILY_MOBIUS_IFS as usize]);
                scene.camera.fov_y_degrees = fov;
                scene.camera.yaw_degrees = yaw;
                scene.camera.pitch_degrees = pitch;
                scene.camera.roll_degrees = roll;
                let p = scene.pack(width, height);
                let basis = [P_CAM_FORWARD, P_CAM_RIGHT, P_CAM_UP]
                    .map(|slot| [p[slot], p[slot + 1], p[slot + 2]]);
                let slope =
                    pixel_beam_slope(basis, p[P_HALF_W], p[P_HALF_H], p[P_WIDTH], p[P_HEIGHT]);
                assert!(slope.is_finite() && slope > 0.0);
                let ray = |x: u32, y: u32, u: f32, v: f32| {
                    // Actual float order of CUDA trace_pixel and WGSL camera_dir.
                    let nx = 2.0 * (x as f32 + u) / width as f32 - 1.0;
                    let ny = 1.0 - 2.0 * (y as f32 + v) / height as f32;
                    let d: [f32; 3] = std::array::from_fn(|axis| {
                        p[P_CAM_FORWARD + axis]
                            + (p[P_CAM_RIGHT + axis] * (nx * p[P_HALF_W])
                                + p[P_CAM_UP + axis] * (ny * p[P_HALF_H]))
                    });
                    let inverse = 1.0 / (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                    d.map(|v| v * inverse)
                };
                for x in [0, width / 2, width - 1] {
                    for y in [0, height / 2, height - 1] {
                        let center = ray(x, y, 0.5, 0.5);
                        for u in [0.0, 0.5, 1.0f32.next_down()] {
                            for v in [0.0, 0.5, 1.0f32.next_down()] {
                                let sample = ray(x, y, u, v);
                                let separation = (0..3)
                                    .map(|axis| {
                                        (f64::from(sample[axis]) - f64::from(center[axis])).powi(2)
                                    })
                                    .sum::<f64>()
                                    .sqrt();
                                assert!(
                                    separation <= f64::from(slope),
                                    "fov={fov} size={width}x{height} rotation={yaw}/{pitch}/{roll} pixel={x}/{y} uv={u}/{v} slope={slope} separation={separation}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn pixel_beam_slope_bounds_sheared_bases_and_disables_degenerate_shortcuts() {
    use fractal_core::ray_bounds::{beam_sphere_entry, pixel_beam_slope};
    for basis in [
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [[1.0, 0.0, 0.1], [0.0, 1.04, 0.0], [0.1, 0.0, 0.94]],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.01]],
    ] {
        let (hw, hh, width, height) = (0.5, 0.3, 128.0, 96.0);
        let slope = pixel_beam_slope(basis, hw, hh, width, height);
        assert!(slope > 0.0 && slope < f32::MAX, "usable basis={basis:?}");
        let ray = |nx: f32, ny: f32| {
            let d: [f32; 3] = std::array::from_fn(|axis| {
                basis[0][axis] + (basis[1][axis] * (nx * hw) + basis[2][axis] * (ny * hh))
            });
            let inverse = 1.0 / (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            d.map(|v| v * inverse)
        };
        for nx in [-0.9, 0.0, 0.9] {
            for ny in [-0.9, 0.0, 0.9] {
                let center = ray(nx, ny);
                for dx in [-1.0 / width, 0.0, 1.0 / width] {
                    for dy in [-1.0 / height, 0.0, 1.0 / height] {
                        let sample = ray(nx + dx, ny + dy);
                        let separation = (0..3)
                            .map(|axis| (f64::from(sample[axis]) - f64::from(center[axis])).powi(2))
                            .sum::<f64>()
                            .sqrt();
                        assert!(
                            separation <= f64::from(slope),
                            "basis={basis:?} ndc={nx}/{ny} separation={separation} slope={slope}"
                        );
                    }
                }
            }
        }
    }
    let degenerate = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
    let slope = pixel_beam_slope(degenerate, 0.5, 0.3, 128.0, 96.0);
    assert_eq!(slope, f32::MAX);
    assert_eq!(
        beam_sphere_entry([0.0, 0.0, 5.0], [0.0, 0.0, -1.0], 1.0, slope),
        0.0
    );
}
