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
