//! Host side of the GPU: one CUDA context, the loaded kernel module, and progressive render
//! targets (accumulator + float display light) that the viewport, the gallery thumbnails and final renders use.

use std::sync::Arc;
use std::time::Instant;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};

use crate::gpu::kernels;
use crate::palette::{PaletteScheme, build_lut};
use crate::params::*;
use crate::scene::{MaterialModel, Scene};

const BLOCK: u32 = 128;

/// Slots that only the tonemap reads: changing them re-tonemaps without restarting the samples.
const TONEMAP_ONLY: [usize; 3] = [P_EXPOSURE, P_SATURATION, P_TONEMAP];
/// Slots that change every launch.
const PER_LAUNCH: [usize; 3] = [P_SAMPLE_BEGIN, P_SPP, P_SEED];

pub struct Gpu {
    _ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: kernels::LoadedModule,
    lut: DeviceBuffer<[f32; 4]>,
    objects: DeviceBuffer<f32>,
    lights: DeviceBuffer<f32>,
    lut_scheme: Option<PaletteScheme>,
    lut_environment: Option<(String, u64)>,
    environments:
        std::collections::BTreeMap<(String, u64), Result<Arc<crate::environment::Map>, String>>,
    pub name: String,
    colour: crate::color::ColorPipeline,
    colour_revision: u64,
    denoiser: Option<Result<crate::denoise::Processor, String>>,
}

/// A progressive render of one scene at one size.
pub struct Target {
    pub width: usize,
    pub height: usize,
    accum: DeviceBuffer<[f32; 4]>,
    albedo: DeviceBuffer<[f32; 4]>,
    normal: DeviceBuffer<[f32; 4]>,
    pub denoise: crate::denoise::State,
    denoise_selected: bool,
    out: DeviceBuffer<[f32; 4]>,
    /// SDR monitor codes (R in the low byte), row major, after the last `step`.
    pub pixels: Vec<u32>,
    /// Linear display Rec.709 (HDR: 1 = 100 nits); never quantized.
    pub light: Vec<[f32; 4]>,
    raw: Vec<[f32; 4]>,
    pub hdr: bool,
    pub colour_error: Option<String>,
    display_key: Option<(f32, f32, bool, String)>,
    pub samples: u32,
    key: Vec<f32>,
    palette: Option<PaletteScheme>,
    environment: Option<(String, u64)>,
    /// GPU milliseconds of the last traced batch and its sample count.
    pub last_ms: f32,
    pub last_spp: u32,
}

impl Gpu {
    pub fn invalidate_colour(&mut self) {
        self.colour = crate::color::ColorPipeline::new();
        self.colour_revision = self.colour_revision.wrapping_add(1);
    }
    pub fn new() -> Result<Self, String> {
        let ctx = CudaContext::new(0).map_err(|e| format!("CUDA context: {e:?}"))?;
        let stream = ctx.default_stream();
        // SAFETY: this package owns the embedded device bundle for `kernels`.
        let module = unsafe { kernels::load(&ctx) }.map_err(|e| format!("load module: {e:?}"))?;
        let name = ctx.device_name().unwrap_or_else(|_| "CUDA GPU".into());
        let lut =
            DeviceBuffer::zeroed(&stream, PALETTE_SAMPLES + 1).map_err(|e| format!("{e:?}"))?;
        Ok(Self {
            _ctx: ctx,
            stream: stream.clone(),
            module,
            lut,
            objects: DeviceBuffer::zeroed(&stream, 1).map_err(|e| format!("objects: {e:?}"))?,
            lights: DeviceBuffer::zeroed(&stream, 1).map_err(|e| format!("lights: {e:?}"))?,
            lut_scheme: None,
            lut_environment: None,
            environments: Default::default(),
            name,
            colour: crate::color::ColorPipeline::new(),
            colour_revision: 0,
            denoiser: None,
        })
    }

    pub fn target(&self, width: usize, height: usize) -> Target {
        let padded = width.div_ceil(8) * height.div_ceil(4) * 32;
        Target {
            width,
            height,
            accum: DeviceBuffer::zeroed(&self.stream, padded).expect("accumulator"),
            albedo: DeviceBuffer::zeroed(&self.stream, padded).expect("albedo guides"),
            normal: DeviceBuffer::zeroed(&self.stream, padded).expect("normal guides"),
            denoise: Default::default(),
            denoise_selected: false,
            out: DeviceBuffer::zeroed(&self.stream, width * height).expect("output"),
            pixels: vec![0; width * height],
            light: vec![[0.0; 4]; width * height],
            raw: vec![[0.0; 4]; width * height],
            hdr: false,
            colour_error: None,
            display_key: None,
            samples: 0,
            key: Vec::new(),
            palette: None,
            environment: None,
            last_ms: 0.0,
            last_spp: 0,
        }
    }

    /// Read the accumulated scene-linear radiance without exposure, saturation or OCIO.
    /// Only final exports need this additional readback; CUDA uses padded 8×4 tiles.
    pub fn scene_linear(&self, target: &Target) -> Vec<[f32; 4]> {
        if target.denoise_selected {
            if let Some(output) = &target.denoise.output {
                return output.clone();
            }
        }
        self.raw_scene_linear(target)
    }

    pub fn raw_scene_linear(&self, target: &Target) -> Vec<[f32; 4]> {
        let mut tiled = vec![[0.0; 4]; target.accum.len()];
        target
            .accum
            .copy_to_host(&self.stream, &mut tiled)
            .expect("scene-linear readback");
        let mut radiance = vec![[0.0; 4]; target.width * target.height];
        for y in 0..target.height {
            for x in 0..target.width {
                let tile = (y / 4) * target.width.div_ceil(8) + x / 8;
                let value = tiled[tile * 32 + (y % 4) * 8 + x % 8];
                let inv = if value[3] > 0.0 {
                    value[3].recip()
                } else {
                    0.0
                };
                radiance[y * target.width + x] =
                    [value[0] * inv, value[1] * inv, value[2] * inv, 1.0];
            }
        }
        radiance
    }

    /// Reset accumulation for changed tracing parameters without rendering or CPU readback.
    /// This lets schedulers calculate remaining samples before selecting a single traced batch.
    pub fn prepare_target(
        &mut self,
        target: &mut Target,
        scene: &Scene,
        bounces_cap: Option<u32>,
    ) -> bool {
        let upload = match WorldUpload::new(scene, target.width as u32, target.height as u32) {
            Ok(upload) => upload,
            Err(error) => {
                target.colour_error = Some(error);
                return false;
            }
        };
        let mut p = upload.params.clone();
        if let Some(cap) = bounces_cap {
            p[P_MAX_BOUNCES] = p[P_MAX_BOUNCES].min(cap as f32);
        }
        // Environment dimensions affect the launch, but source identity belongs to
        // the accumulation key even before the worker has decoded that source.
        self.reset_target_for_params(target, scene, &p)
    }

    fn reset_target_for_params(&self, target: &mut Target, scene: &Scene, params: &[f32]) -> bool {
        let mut key = params.to_vec();
        key.extend(scene_trace_data(scene));
        for i in TONEMAP_ONLY.iter().chain(PER_LAUNCH.iter()) {
            key[*i] = 0.0;
        }
        if key == target.key
            && target.palette == Some(scene.palette)
            && target.environment == scene.environment.key()
        {
            return false;
        }
        target
            .accum
            .zero_async(&self.stream)
            .expect("clear accumulator");
        target
            .albedo
            .zero_async(&self.stream)
            .expect("clear albedo guides");
        target
            .normal
            .zero_async(&self.stream)
            .expect("clear normal guides");
        target.denoise.reset();
        target.denoise_selected = false;
        target.samples = 0;
        target.display_key = None;
        target.key = key;
        target.palette = Some(scene.palette);
        target.environment = scene.environment.key();
        true
    }

    /// Called on the CUDA worker, never the GUI thread. Cache failures too so a bad
    /// file cannot make every repaint start another decode; Reload bumps revision.
    pub fn ensure_environment(
        &mut self,
        scene: &Scene,
    ) -> Result<Option<Arc<crate::environment::Map>>, String> {
        validate_world(scene)?;
        let Some(key) = scene.environment.key() else {
            return Ok(None);
        };
        if !self.environments.contains_key(&key) {
            if self.environments.len() >= 2 {
                self.environments.clear();
            }
            let loaded = crate::environment::Map::load(std::path::Path::new(&key.0)).map(Arc::new);
            self.environments.insert(key.clone(), loaded);
        }
        self.environments[&key].clone().map(Some)
    }

    /// Add `spp` samples (0 = only re-tonemap) and refresh `target.pixels`. Restarts the
    /// accumulation when anything but the tonemap changed. `bounces_cap` limits bounces (preview).
    pub fn step(
        &mut self,
        target: &mut Target,
        scene: &Scene,
        spp: u32,
        seed: u32,
        bounces_cap: Option<u32>,
        final_pass: bool,
    ) {
        let upload = match WorldUpload::new(scene, target.width as u32, target.height as u32) {
            Ok(upload) => upload,
            Err(error) => {
                target.colour_error = Some(error);
                return;
            }
        };
        let mut p = upload.params.clone();
        if let Some(cap) = bounces_cap {
            p[P_MAX_BOUNCES] = p[P_MAX_BOUNCES].min(cap as f32);
        }
        self.reset_target_for_params(target, scene, &p);
        let env = match self.ensure_environment(scene) {
            Ok(env) => env,
            Err(error) => {
                target.colour_error = Some(error);
                return;
            }
        };
        let environment_key = scene.environment.key();
        if scene.world_render
            || self.lut_scheme != Some(scene.palette)
            || self.lut_environment != environment_key
        {
            let mut table = build_lut(scene.palette);
            if scene.world_render {
                for object in &scene.objects {
                    table.extend(build_lut(object.palette));
                }
            }
            if let Some(env) = &env {
                table.extend_from_slice(&env.texels);
            }
            if self.lut.len() != table.len() {
                self.lut = DeviceBuffer::zeroed(&self.stream, table.len())
                    .expect("environment allocation");
            }
            self.lut
                .copy_from_host(&self.stream, &table)
                .expect("palette / environment upload");
            self.lut_scheme = if scene.world_render {
                None
            } else {
                Some(scene.palette)
            };
            self.lut_environment = environment_key;
        }
        if let Some(env) = env {
            p[P_ENV_WIDTH] = env.width as f32;
            p[P_ENV_HEIGHT] = env.height as f32;
            p[P_ENV_MEAN] = env.mean_luminance;
        }
        if scene.world_render {
            if self.objects.len() != upload.objects.len().max(1) {
                self.objects = DeviceBuffer::zeroed(&self.stream, upload.objects.len().max(1))
                    .expect("world object allocation");
            }
            if self.lights.len() != upload.lights.len().max(1) {
                self.lights = DeviceBuffer::zeroed(&self.stream, upload.lights.len().max(1))
                    .expect("world light allocation");
            }
            if !upload.objects.is_empty() {
                self.objects
                    .copy_from_host(&self.stream, &upload.objects)
                    .expect("world objects upload");
            }
            if !upload.lights.is_empty() {
                self.lights
                    .copy_from_host(&self.stream, &upload.lights)
                    .expect("world lights upload");
            }
        }
        p[P_SAMPLE_BEGIN] = target.samples as f32;
        p[P_SPP] = spp as f32;
        p[P_SEED] = seed as f32;
        let block: [f32; P_COUNT] = p.as_slice().try_into().expect("parameter block size");
        self.module
            .set_params(&self.stream, &block)
            .expect("params upload");

        let t0 = Instant::now();
        if spp > 0 {
            let padded = target.accum.len() as u32;
            let cfg = LaunchConfig1D::new(padded.div_ceil(BLOCK), BLOCK, 0);
            let full = scene.material.model == MaterialModel::StandardSurface;
            let (m, s, lut, acc) = (&self.module, &self.stream, &self.lut, &mut target.accum);
            macro_rules! launch {
                ($prep:ident, $run:ident) => {{
                    let prepared = m.$prep(cfg).expect(stringify!($prep));
                    m.$run(
                        s,
                        &prepared,
                        lut,
                        &self.objects,
                        &self.lights,
                        acc,
                        &mut target.albedo,
                        &mut target.normal,
                    )
                    .expect(stringify!($run));
                }};
            }
            if scene.world_render {
                launch!(prepare_world, world);
            } else {
                match (full, scene.formula.code()) {
                    (false, FAMILY_BULB) => launch!(prepare_fast_bulb, fast_bulb),
                    (false, FAMILY_BOX) => launch!(prepare_fast_box, fast_box),
                    (false, FAMILY_QUAT) => launch!(prepare_fast_quat, fast_quat),
                    (false, FAMILY_KIFS) => launch!(prepare_fast_kifs, fast_kifs),
                    (false, FAMILY_KLEINIAN) => launch!(prepare_fast_kleinian, fast_kleinian),
                    (false, FAMILY_PSEUDO_KLEINIAN) => {
                        launch!(prepare_fast_pseudo_kleinian, fast_pseudo_kleinian)
                    }
                    (false, FAMILY_APOLLONIAN) => launch!(prepare_fast_apollonian, fast_apollonian),
                    (false, _) => launch!(prepare_fast_hybrid, fast_hybrid),
                    (true, FAMILY_BULB) => launch!(prepare_full_bulb, full_bulb),
                    (true, FAMILY_BOX) => launch!(prepare_full_box, full_box),
                    (true, FAMILY_QUAT) => launch!(prepare_full_quat, full_quat),
                    (true, FAMILY_KIFS) => launch!(prepare_full_kifs, full_kifs),
                    (true, FAMILY_KLEINIAN) => launch!(prepare_full_kleinian, full_kleinian),
                    (true, FAMILY_PSEUDO_KLEINIAN) => {
                        launch!(prepare_full_pseudo_kleinian, full_pseudo_kleinian)
                    }
                    (true, FAMILY_APOLLONIAN) => launch!(prepare_full_apollonian, full_apollonian),
                    (true, _) => launch!(prepare_full_hybrid, full_hybrid),
                }
            }
            target.samples += spp;
        }
        let denoise_changed = self.process_denoise(target, scene, final_pass);
        let display_key = (
            scene.render.exposure_stops,
            scene.render.saturation,
            scene.render.reinhard,
            format!(
                "{}:{}",
                self.colour_revision,
                serde_json::to_string(&scene.colour).expect("colour settings")
            ),
        );
        if spp == 0 && !denoise_changed && target.display_key.as_ref() == Some(&display_key) {
            return;
        }
        if target.denoise_selected {
            let output = target
                .denoise
                .output
                .as_ref()
                .expect("selected denoised output");
            for (raw, pixel) in target.raw.iter_mut().zip(output) {
                let c = [
                    pixel[0] * p[P_EXPOSURE],
                    pixel[1] * p[P_EXPOSURE],
                    pixel[2] * p[P_EXPOSURE],
                ];
                let l = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
                *raw = [
                    l + (c[0] - l) * p[P_SATURATION],
                    l + (c[1] - l) * p[P_SATURATION],
                    l + (c[2] - l) * p[P_SATURATION],
                    1.0,
                ];
            }
        } else {
            let n = (target.width * target.height) as u32;
            let cfg = LaunchConfig1D::new(n.div_ceil(BLOCK), BLOCK, 0);
            let prepared = self.module.prepare_tonemap(cfg).expect("prepare tonemap");
            self.module
                .tonemap(&self.stream, &prepared, &target.accum, &mut target.out)
                .expect("tonemap");
            target
                .out
                .copy_to_host(&self.stream, &mut target.raw)
                .expect("readback");
        }
        match self.colour.apply(
            target.width,
            target.height,
            &target.raw,
            &scene.colour,
            scene.render.reinhard,
        ) {
            Ok((light, encoded, absolute)) => {
                target.light = light;
                target.hdr = absolute;
                target.colour_error = None;
                target.display_key = Some(display_key);
                for (p, l) in target.pixels.iter_mut().zip(&encoded) {
                    let code = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                    *p = u32::from_le_bytes([code(l[0]), code(l[1]), code(l[2]), 255]);
                }
            }
            Err(e) => target.colour_error = Some(e),
        }
        if spp > 0 {
            target.last_ms = t0.elapsed().as_secs_f32() * 1000.0;
            target.last_spp = spp;
        }
    }

    fn guide_sums(&self, target: &Target, buffer: &DeviceBuffer<[f32; 4]>) -> Vec<[f32; 4]> {
        let mut tiled = vec![[0.0; 4]; buffer.len()];
        buffer
            .copy_to_host(&self.stream, &mut tiled)
            .expect("guide readback");
        (0..target.height)
            .flat_map(|y| {
                let tiled = &tiled;
                (0..target.width).map(move |x| {
                    let tile = (y / 4) * target.width.div_ceil(8) + x / 8;
                    tiled[tile * 32 + (y % 4) * 8 + x % 8]
                })
            })
            .collect()
    }

    fn process_denoise(&mut self, target: &mut Target, scene: &Scene, final_pass: bool) -> bool {
        let settings = &scene.render.denoise;
        let previous = target.denoise_selected;
        let due = target.denoise.due(settings, target.samples, final_pass);
        if due {
            let color = self.raw_scene_linear(target);
            let albedo = (settings.mode != crate::denoise::Mode::Color)
                .then(|| self.guide_sums(target, &target.albedo));
            let normal = (settings.mode == crate::denoise::Mode::ColorAlbedoNormal)
                .then(|| self.guide_sums(target, &target.normal));
            let processor = self
                .denoiser
                .get_or_insert_with(crate::denoise::Processor::new);
            let result = match processor {
                Ok(processor) => processor.process(
                    target.width,
                    target.height,
                    &color,
                    albedo.as_deref(),
                    normal.as_deref(),
                    settings,
                    target.samples,
                ),
                Err(error) => Err(error.clone()),
            };
            target.denoise.complete(target.samples, result);
        }
        target.denoise_selected = settings.enabled && target.denoise.output.is_some();
        due || previous != target.denoise_selected
    }
}

/// Render-relevant runtime data only; hierarchy names and UUIDs never enter this signature.
pub(crate) fn scene_trace_data(scene: &Scene) -> Vec<f32> {
    let mut values = vec![
        scene.world_render as u8 as f32,
        scene.objects.len() as f32,
        scene.lights.len() as f32,
    ];
    if !scene.world_render {
        return values;
    }
    for object in &scene.objects {
        let p = object.pack(1, 1);
        values.extend_from_slice(&p[P_FAMILY..P_LIGHT_DIR]);
        values.extend_from_slice(&p[P_BASE..P_EXPOSURE]);
        values.push(p[P_MATERIAL_MODEL]);
        if let Some(matrix) = object.object_world {
            values.push(1.0);
            values.extend(matrix.into_iter().flatten());
        } else {
            values.push(0.0);
        }
        values.extend(build_lut(object.palette).into_iter().flatten());
    }
    for light in &scene.lights {
        let mut packed = Scene::preset(FAMILY_BULB);
        packed.lighting = *light;
        let p = packed.pack(1, 1);
        values.extend_from_slice(&p[P_LIGHT_DIR..P_SKY_INTENSITY]);
    }
    values
}

struct WorldUpload {
    params: Vec<f32>,
    objects: Vec<f32>,
    lights: Vec<f32>,
}

fn object_matrix(object: &Scene) -> glam::Mat4 {
    object
        .object_world
        .map(|columns| glam::Mat4::from_cols_array_2d(&columns))
        .unwrap_or_else(|| {
            // Legacy ObjectTransform uses the same XYZ Euler convention as Scene::pack.
            let p = object.pack(1, 1);
            let s = p[P_OBJ_SCALE];
            glam::Mat4::from_cols(
                glam::Vec4::new(
                    p[P_OBJ_AXES] * s,
                    p[P_OBJ_AXES + 1] * s,
                    p[P_OBJ_AXES + 2] * s,
                    0.0,
                ),
                glam::Vec4::new(
                    p[P_OBJ_AXES + 3] * s,
                    p[P_OBJ_AXES + 4] * s,
                    p[P_OBJ_AXES + 5] * s,
                    0.0,
                ),
                glam::Vec4::new(
                    p[P_OBJ_AXES + 6] * s,
                    p[P_OBJ_AXES + 7] * s,
                    p[P_OBJ_AXES + 8] * s,
                    0.0,
                ),
                glam::Vec4::from((glam::Vec3::from_array(object.object.offset), 1.0)),
            )
        })
}

fn inverse_object(object: &Scene, index: usize) -> Result<glam::Mat4, String> {
    let matrix = object_matrix(object);
    let columns = matrix.to_cols_array_2d();
    if !matrix.is_finite()
        || columns[0][3] != 0.0
        || columns[1][3] != 0.0
        || columns[2][3] != 0.0
        || columns[3][3] != 1.0
        || matrix.determinant() == 0.0
    {
        return Err(format!(
            "World object {index} has a nonfinite, nonaffine or singular transform"
        ));
    }
    let inverse = matrix.inverse();
    if !inverse.is_finite() {
        return Err(format!(
            "World object {index} transform has no finite inverse"
        ));
    }
    Ok(inverse)
}

pub(crate) fn validate_world(scene: &Scene) -> Result<(), String> {
    if scene.world_render {
        WorldUpload::new(scene, 1, 1)?;
    }
    Ok(())
}

impl WorldUpload {
    fn new(scene: &Scene, width: u32, height: u32) -> Result<Self, String> {
        let mut params = scene.pack(width, height);
        let mut objects = Vec::new();
        let mut lights = Vec::new();
        if scene.world_render {
            params[P_WORLD] = WORLD_ABI_VERSION as f32;
            params[P_OBJECT_COUNT] = scene.objects.len() as f32;
            params[P_LIGHT_COUNT] = scene.lights.len() as f32;
            let mut extent = 0.0f32;
            for (index, object) in scene.objects.iter().enumerate() {
                let inverse = inverse_object(object, index)?;
                let matrix = object_matrix(object);
                let mut local = object.clone();
                local.object.offset = [0.0; 3];
                local.object.rotation_degrees = [0.0; 3];
                local.object.scale = 1.0;
                let p = local.pack(width, height);
                let local_radius = p[P_CLIP_RADIUS];
                objects.extend_from_slice(&p);
                let cols = inverse.to_cols_array_2d();
                for row in 0..3 {
                    for column in &cols {
                        objects.push(column[row]);
                    }
                }
                // ||A^-1||2 <= sqrt(||A^-1||1 ||A^-1||inf), so this is a
                // conservative lower bound on sigma_min(A), including parent shear.
                let norm1 = (0..3)
                    .map(|c| (0..3).map(|r| cols[c][r].abs()).sum::<f32>())
                    .fold(0.0f32, f32::max);
                let norm_inf = (0..3)
                    .map(|r| (0..3).map(|c| cols[c][r].abs()).sum::<f32>())
                    .fold(0.0f32, f32::max);
                let scale = 1.0 / (norm1.sqrt() * norm_inf.sqrt());
                if !scale.is_finite() || scale <= 0.0 {
                    return Err(format!(
                        "World object {index} has an unusable distance scale"
                    ));
                }
                objects.push(scale);
                objects.push(local_radius);
                let tangent = matrix.transform_vector3(glam::Vec3::Y).normalize();
                if !tangent.is_finite() {
                    return Err(format!(
                        "World object {index} has an unusable material tangent"
                    ));
                }
                objects.extend(tangent.to_array());
                let cols = matrix.to_cols_array_2d();
                let upper = (0..3)
                    .map(|c| (0..3).map(|r| cols[c][r] * cols[c][r]).sum::<f32>())
                    .sum::<f32>()
                    .sqrt();
                let center = matrix.transform_point3(glam::Vec3::ZERO);
                extent = extent.max(center.length() + local_radius * upper);
            }
            if !extent.is_finite() {
                return Err("World geometry bounds exceed finite render range".into());
            }
            params[P_CLIP_CENTER..P_CLIP_CENTER + 3].fill(0.0);
            params[P_CLIP_RADIUS] = extent;
            let camera = glam::Vec3::from_slice(&params[P_CAM_ORIGIN..P_CAM_ORIGIN + 3]);
            params[P_MAX_DISTANCE] = params[P_MAX_DISTANCE].max(camera.length() + 2.0 * extent);
            for light in &scene.lights {
                let mut packed = Scene::preset(FAMILY_BULB);
                packed.lighting = *light;
                let mut p = packed.pack(1, 1);
                // Stable for zero/tiny angular radii; 1-cos(theta) cancels in f32.
                let half_angle = (light.sun_angle * 0.5)
                    .to_radians()
                    .clamp(1e-4, std::f32::consts::PI);
                let sine = (0.5 * half_angle).sin();
                p[P_SUN_ONE_MINUS_COS] = 2.0 * sine * sine;
                p[P_SUN_CONE_PDF] = 1.0 / (2.0 * std::f32::consts::PI * p[P_SUN_ONE_MINUS_COS]);
                if p[P_LIGHT_DIR..P_SKY_INTENSITY]
                    .iter()
                    .any(|v| !v.is_finite())
                {
                    return Err("World directional light has nonfinite parameters".into());
                }
                lights.extend_from_slice(&p[P_LIGHT_DIR..P_SKY_INTENSITY]);
            }
        }
        Ok(Self {
            params,
            objects,
            lights,
        })
    }
}

impl Target {
    /// Samples per second of the last traced batch, in millions.
    #[allow(dead_code)] // Retained for standalone CUDA render consumers; GUI uses CPU Frame.
    pub fn msamples_per_s(&self) -> f64 {
        if self.last_ms <= 0.0 {
            return 0.0;
        }
        (self.width * self.height) as f64 * self.last_spp as f64
            / (self.last_ms as f64 / 1000.0)
            / 1.0e6
    }

    pub fn save_png(&self, path: &std::path::Path) -> Result<(), String> {
        if let Some(e) = &self.colour_error {
            return Err(format!("Colour transform failed: {e}"));
        }
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        }
        let pixels = if self.hdr {
            // OCIO display light is absolute, 1 = 100 nits. HDR10 is BT.2020 + PQ.
            let codes = self
                .light
                .iter()
                .flat_map(|p| {
                    let nits = egui_display::rec2020_nits([p[0], p[1], p[2]], 100.0);
                    [pq16(nits[0]), pq16(nits[1]), pq16(nits[2]), 65535]
                })
                .collect();
            egui_display::screenshot::Pixels::Rgba16(codes)
        } else {
            egui_display::screenshot::Pixels::Rgba8(
                self.pixels.iter().flat_map(|p| p.to_le_bytes()).collect(),
            )
        };
        let capture = egui_display::screenshot::Capture {
            output: if self.hdr {
                egui_display::Output::Hdr10
            } else {
                egui_display::Output::Sdr8
            },
            width: self.width as u32,
            height: self.height as u32,
            white_nits: 100.0,
            peak_nits: if self.hdr { 1000.0 } else { 100.0 },
            pixels,
        };
        capture.save(path).map(|_| ()).map_err(|e| e.to_string())
    }

    /// Display light, linear Rec.709, normalized to 100 nits (matching exr-view).
    pub fn save_display_exr(&self, path: &std::path::Path) -> Result<(), String> {
        if let Some(e) = &self.colour_error {
            return Err(format!("Colour transform failed: {e}"));
        }
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        }
        use exr::prelude::*;
        let mut image = Image::from_channels(
            (self.width, self.height),
            SpecificChannels::rgb(|pos: Vec2<usize>| {
                let p = self.light[pos.y() * self.width + pos.x()];
                (p[0], p[1], p[2])
            }),
        );
        image.attributes.chromaticities = Some(attribute::Chromaticities {
            red: Vec2(0.64, 0.33),
            green: Vec2(0.30, 0.60),
            blue: Vec2(0.15, 0.06),
            white: Vec2(0.3127, 0.3290),
        });
        image.layer_data.attributes.white_luminance = Some(100.0);
        image.write().to_file(path).map_err(|e| e.to_string())
    }
}

fn pq16(nits: f32) -> u16 {
    (egui_display::pq(nits.clamp(0.0, 10000.0)) * 65535.0 + 0.5) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    fn world_object(family: u32, translation: [f32; 3], scale: [f32; 3], color: [f32; 3]) -> Scene {
        let mut object = Scene::preset(family);
        object.object_world = Some(
            glam::Mat4::from_scale_rotation_translation(
                glam::Vec3::from_array(scale),
                glam::Quat::from_rotation_y(0.31),
                glam::Vec3::from_array(translation),
            )
            .to_cols_array_2d(),
        );
        object.material.color_source = crate::scene::ColorSource::Material;
        object.material.base_color = color;
        object.material.emission = 2.0;
        object.material.emission_color = color;
        object.material.specular = 0.0;
        object
    }
    fn dark_world() -> Scene {
        let mut scene = Scene::preset(FAMILY_BULB);
        scene.world_render = true;
        scene.camera.distance = 4.0;
        scene.camera.fov_y_degrees = 55.0;
        scene.lighting.sky_intensity = 0.0;
        scene.lighting.background = false;
        scene.colour.on = false;
        scene.render.denoise.enabled = false;
        scene.render.max_bounces = 0;
        scene.render.max_steps = 256;
        scene
    }
    #[test]
    fn cuda_primary_guides_capture_material_world_normal_and_miss_counts() {
        let mut gpu = Gpu::new().unwrap();
        let mut object = Scene::preset(FAMILY_KIFS);
        object.formula =
            crate::scene::Formula::Kifs(crate::scene::Kifs::preset(crate::scene::KifsKind::Menger));
        object.render.iterations = 0;
        object.material.color_source = crate::scene::ColorSource::Material;
        object.material.base_color = [0.2, 0.4, 0.6];
        object.material.base = 0.8;
        object.material.base_tint = [1.0, 0.5, 0.25];
        object.material.metalness = 0.25;
        object.material.emission = 0.7;
        object.material.emission_color = [0.1, 0.2, 0.3];
        let matrix = glam::Mat4::from_cols(
            glam::Vec4::new(0.7, 0.0, 0.2, 0.0),
            glam::Vec4::new(0.3, 0.9, 0.0, 0.0),
            glam::Vec4::new(0.2, 0.0, 1.2, 0.0),
            glam::Vec4::W,
        );
        for world in [false, true] {
            for model in [MaterialModel::Fast, MaterialModel::StandardSurface] {
                let mut scene = object.clone();
                scene.material.model = model;
                scene.colour.on = false;
                scene.render.denoise.enabled = false;
                scene.render.max_bounces = 0;
                scene.camera.distance = 4.0;
                scene.camera.fov_y_degrees = 55.0;
                if world {
                    scene.world_render = true;
                    let mut child = scene.clone();
                    child.object_world = Some(matrix.to_cols_array_2d());
                    scene.objects = vec![child];
                }
                let mut target = gpu.target(33, 25);
                gpu.step(&mut target, &scene, 4, 19, None, false);
                let albedo = gpu.guide_sums(&target, &target.albedo);
                let normal = gpu.guide_sums(&target, &target.normal);
                assert!(albedo.iter().all(|p| p[3] == 4.0));
                assert!(normal.iter().all(|p| p[3] == 4.0));
                assert!(
                    albedo
                        .iter()
                        .zip(&normal)
                        .any(|(a, n)| a[..3] == [0.0; 3] && n[..3] == [0.0; 3])
                );
                let center = 12 * 33 + 16;
                for (actual, expected) in albedo[center][..3].iter().zip([0.19, 0.26, 0.3]) {
                    assert!(
                        (actual / 4.0 - expected).abs() < 1e-5,
                        "{:?}",
                        albedo[center]
                    );
                }
                let expected = if world {
                    matrix
                        .inverse()
                        .transpose()
                        .transform_vector3(glam::Vec3::Z)
                        .normalize()
                } else {
                    glam::Vec3::Z
                };
                let actual = glam::Vec3::from_slice(&normal[center]) / 4.0;
                assert!(
                    actual.distance(expected) < 0.01,
                    "world={world}: {actual:?} vs {expected:?}"
                );
                scene.camera.yaw_degrees += 0.1;
                gpu.step(&mut target, &scene, 1, 19, None, false);
                assert_eq!(target.samples, 1);
                assert!(
                    gpu.guide_sums(&target, &target.albedo)
                        .iter()
                        .all(|p| p[3] == 1.0)
                );
                if world {
                    scene.objects.clear();
                    gpu.step(&mut target, &scene, 1, 19, None, false);
                    assert!(
                        gpu.guide_sums(&target, &target.albedo)
                            .iter()
                            .all(|p| *p == [0.0, 0.0, 0.0, 1.0])
                    );
                    assert!(
                        gpu.guide_sums(&target, &target.normal)
                            .iter()
                            .all(|p| *p == [0.0, 0.0, 0.0, 1.0])
                    );
                }
            }
        }
    }

    #[test]
    #[ignore = "requires actual CUDA and shared wgpu OIDN inference"]
    fn cuda_oidn_final_output_is_linear_and_never_changes_raw_samples() {
        let mut gpu = Gpu::new().unwrap();
        let mut scene = dark_world();
        scene.objects = vec![world_object(
            FAMILY_BULB,
            [0.0; 3],
            [1.0; 3],
            [3.0, 1.0, 0.25],
        )];
        scene.render.denoise.enabled = true;
        scene.render.denoise.interval = 128;
        scene.render.exposure_stops = 0.0;
        scene.render.saturation = 1.0;
        scene.render.reinhard = false;
        let mut target = gpu.target(33, 25);
        gpu.step(&mut target, &scene, 4, 0, None, false);
        assert!(target.denoise.output.is_none());
        let raw = gpu.raw_scene_linear(&target);
        gpu.step(&mut target, &scene, 0, 0, None, true);
        assert!(target.denoise.error.is_none(), "{:?}", target.denoise.error);
        assert_eq!(target.denoise.last_samples, 4);
        let denoised = target
            .denoise
            .output
            .clone()
            .expect("final pass must run below interval");
        assert_eq!(gpu.scene_linear(&target), denoised);
        assert!(denoised.iter().flatten().all(|v| v.is_finite()));
        assert!(
            denoised.iter().any(|p| p[0] > 1.0),
            "HDR inference must retain headroom"
        );
        assert_eq!(gpu.raw_scene_linear(&target), raw);
        assert_eq!(target.samples, 4);
        scene.render.exposure_stops = 2.0;
        scene.render.saturation = 0.5;
        gpu.step(&mut target, &scene, 0, 0, None, true);
        assert_eq!(
            gpu.scene_linear(&target),
            denoised,
            "Export ignores display adjustments"
        );
        for (display, p) in target.light.iter().zip(&denoised) {
            let c = [p[0] * 4.0, p[1] * 4.0, p[2] * 4.0];
            let l = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
            for k in 0..3 {
                assert!((display[k] - (l + (c[k] - l) * 0.5)).abs() < 1e-5);
            }
        }
        scene.render.denoise.enabled = false;
        gpu.step(&mut target, &scene, 0, 0, None, true);
        assert_eq!(target.samples, 4);
        assert_eq!(gpu.scene_linear(&target), raw);
        assert_eq!(gpu.raw_scene_linear(&target), raw);
    }

    #[test]
    fn cuda_denoise_failure_cadence_final_and_settings_preserve_raw_accumulation() {
        let mut gpu = Gpu::new().unwrap();
        gpu.denoiser = Some(Err("controlled initialization failure".into()));
        let mut scene = dark_world();
        scene.objects = vec![world_object(
            FAMILY_BULB,
            [0.0; 3],
            [1.0; 3],
            [1.0, 0.2, 0.1],
        )];
        scene.render.denoise.enabled = true;
        scene.render.denoise.interval = 128;
        let mut target = gpu.target(17, 19);
        gpu.step(&mut target, &scene, 3, 0, None, false);
        assert!(target.denoise.error.is_none());
        let raw = gpu.raw_scene_linear(&target);
        gpu.step(&mut target, &scene, 0, 0, None, true);
        assert_eq!(
            target.denoise.error.as_deref(),
            Some("controlled initialization failure")
        );
        assert_eq!(target.samples, 3);
        assert_eq!(gpu.raw_scene_linear(&target), raw);
        assert_eq!(gpu.scene_linear(&target), raw);
        assert!(!target.denoise.due(&scene.render.denoise, 3, true));
        scene.render.denoise.enabled = false;
        scene.render.denoise.mode = crate::denoise::Mode::Color;
        scene.render.denoise.quality = crate::denoise::Quality::High;
        scene.render.exposure_stops = 2.0;
        gpu.step(&mut target, &scene, 0, 0, None, true);
        assert_eq!(target.samples, 3);
        assert_eq!(gpu.raw_scene_linear(&target), raw);
        scene.render.denoise.enabled = true;
        scene.render.denoise.interval = 4;
        gpu.step(&mut target, &scene, 1, 0, None, false);
        assert_eq!(target.samples, 4);
        assert!(target.denoise.error.is_some());
        scene.camera.yaw_degrees += 1.0;
        gpu.step(&mut target, &scene, 1, 0, None, false);
        assert_eq!(target.samples, 1);
        assert!(target.denoise.error.is_none());
    }

    #[test]
    fn world_upload_preserves_affine_shear_and_rejects_singular_transforms() {
        let mut scene = dark_world();
        let mut object = world_object(
            FAMILY_BULB,
            [1.0, -2.0, 3.0],
            [0.4, 1.0, 1.7],
            [1.0, 0.0, 0.0],
        );
        let mut matrix = glam::Mat4::from_cols_array_2d(&object.object_world.unwrap());
        matrix.y_axis += matrix.x_axis * 0.6;
        object.object_world = Some(matrix.to_cols_array_2d());
        scene.objects.push(object);
        let upload = WorldUpload::new(&scene, 16, 16).unwrap();
        let inverse = &upload.objects[O_INVERSE..O_INVERSE + 12];
        let local = glam::Vec3::new(0.23, 0.41, -0.52);
        let world = matrix.transform_point3(local);
        for row in 0..3 {
            let actual = inverse[4 * row] * world.x
                + inverse[4 * row + 1] * world.y
                + inverse[4 * row + 2] * world.z
                + inverse[4 * row + 3];
            assert!((actual - local[row]).abs() < 1e-5);
        }
        let scale = upload.objects[O_DISTANCE_SCALE];
        for direction in [
            glam::Vec3::X,
            glam::Vec3::Y,
            glam::Vec3::Z,
            glam::Vec3::ONE.normalize(),
        ] {
            assert!(scale <= matrix.transform_vector3(direction).length());
        }
        scene.objects[0].object_world =
            Some(glam::Mat4::from_scale(glam::Vec3::new(1.0, 0.0, 1.0)).to_cols_array_2d());
        assert!(validate_world(&scene).unwrap_err().contains("singular"));
    }
    #[test]
    fn cuda_world_two_formulas_materials_occlusion_visibility_and_empty_geometry() {
        let mut gpu = Gpu::new().unwrap();
        let mut target = gpu.target(32, 24);
        let mut scene = dark_world();
        scene.objects = vec![
            world_object(
                FAMILY_BULB,
                [-0.7, 0.0, 0.0],
                [0.6, 0.8, 0.5],
                [1.0, 0.0, 0.0],
            ),
            world_object(
                FAMILY_QUAT,
                [0.7, 0.0, 0.0],
                [0.45, 0.65, 0.4],
                [0.0, 0.0, 1.0],
            ),
        ];
        scene.objects[1].material.model = MaterialModel::StandardSurface;
        gpu.step(&mut target, &scene, 4, 0, None, false);
        assert!(target.colour_error.is_none(), "{:?}", target.colour_error);
        let both = gpu.scene_linear(&target);
        assert!(both.iter().any(|p| p[0] > 0.5 && p[2] < 0.01));
        assert!(both.iter().any(|p| p[2] > 0.5 && p[0] < 0.01));
        scene.objects[0] = world_object(FAMILY_BULB, [0.0, 0.0, 0.7], [1.0; 3], [1.0, 0.0, 0.0]);
        scene.objects[1] = world_object(FAMILY_BULB, [0.0, 0.0, -0.7], [1.0; 3], [0.0, 0.0, 1.0]);
        gpu.step(&mut target, &scene, 4, 0, None, false);
        let center = gpu.scene_linear(&target)[12 * 32 + 16];
        assert!(
            center[0] > center[2],
            "Front object's material must win: {center:?}"
        );
        scene.objects.remove(0);
        gpu.step(&mut target, &scene, 4, 0, None, false);
        let center = gpu.scene_linear(&target)[12 * 32 + 16];
        assert!(
            center[2] > center[0],
            "Hidden front object reveals the rear: {center:?}"
        );
        assert_eq!(target.samples, 4);
        scene.objects.clear();
        gpu.step(&mut target, &scene, 2, 0, None, false);
        assert!(gpu.scene_linear(&target).iter().all(|p| p[..3] == [0.0; 3]));
    }
    #[test]
    fn cuda_world_directional_lights_sum_and_object_edits_reset_accumulation() {
        let mut gpu = Gpu::new().unwrap();
        let mut target = gpu.target(24, 16);
        let mut scene = dark_world();
        let mut object = world_object(FAMILY_BULB, [0.0; 3], [1.0; 3], [1.0; 3]);
        object.material.emission = 0.0;
        scene.objects.push(object);
        let mut light = scene.lighting;
        light.sun_azimuth = 0.0;
        light.sun_elevation = 35.0;
        light.sun_angle = 40.0;
        light.sun_intensity = 4.0;
        light.sun_color = [1.0, 0.0, 0.0];
        scene.lights.push(light);
        gpu.step(&mut target, &scene, 32, 0, None, false);
        let first = gpu.scene_linear(&target);
        let red: f32 = first.iter().map(|p| p[0]).sum();
        assert!(red > 0.1);
        light.sun_color = [0.0, 0.0, 1.0];
        scene.lights.push(light);
        gpu.step(&mut target, &scene, 32, 0, None, false);
        let both = gpu.scene_linear(&target);
        let blue: f32 = both.iter().map(|p| p[2]).sum();
        let red: f32 = both.iter().map(|p| p[0]).sum();
        assert!(blue > 0.1 && red > 0.1);
        assert!(
            (blue - red).abs() < 1e-4 * red.max(1.0),
            "Overlapping sun radiance and mixture PDFs must sum"
        );
        scene.objects[0].material.base_color = [0.5; 3];
        assert!(gpu.prepare_target(&mut target, &scene, None));
        assert_eq!(target.samples, 0);
        gpu.step(&mut target, &scene, 1, 0, None, false);
        scene.objects[0].name = "metadata only".into();
        scene.objects[0].material.preset = Some("UI label only".into());
        assert!(!gpu.prepare_target(&mut target, &scene, None));
        scene.render.exposure_stops += 1.0;
        assert!(!gpu.prepare_target(&mut target, &scene, None));
    }
    #[test]
    fn cuda_world_secondary_rays_reach_another_objects_emission() {
        let mut gpu = Gpu::new().unwrap();
        let mut target = gpu.target(24, 24);
        let mut scene = dark_world();
        let mut mirror = world_object(FAMILY_QUAT, [0.0; 3], [0.75; 3], [1.0; 3]);
        mirror.material.emission = 0.0;
        mirror.material.metalness = 1.0;
        mirror.material.specular = 1.0;
        mirror.material.specular_roughness = 0.45;
        scene.objects = vec![
            mirror,
            world_object(FAMILY_BULB, [2.5, 0.0, 2.8], [0.8; 3], [1.0, 0.2, 0.0]),
        ];
        gpu.step(&mut target, &scene, 64, 0, None, false);
        let direct: f32 = gpu.scene_linear(&target).iter().map(|p| p[0]).sum();
        scene.render.max_bounces = 1;
        gpu.step(&mut target, &scene, 64, 0, None, false);
        let reflected: f32 = gpu.scene_linear(&target).iter().map(|p| p[0]).sum();
        assert!(
            reflected > direct + 0.01,
            "Secondary rays must see emissive sibling: {direct} -> {reflected}"
        );
    }

    #[test]
    fn cuda_world_sibling_casts_a_shadow_outside_its_camera_silhouette() {
        let mut gpu = Gpu::new().unwrap();
        let mut target = gpu.target(32, 24);
        let mut scene = dark_world();
        let mut receiver = world_object(FAMILY_BULB, [0.0; 3], [1.0; 3], [1.0; 3]);
        receiver.material.emission = 0.0;
        scene.objects.push(receiver);
        let mut light = scene.lighting;
        light.sun_azimuth = 45.0;
        light.sun_elevation = 0.0;
        light.sun_angle = 8.0;
        light.sun_intensity = 4.0;
        scene.lights.push(light);
        gpu.step(&mut target, &scene, 64, 0, None, false);
        let clear = gpu.scene_linear(&target);
        let mut blocker = world_object(FAMILY_BULB, [1.0, 0.0, 1.7], [0.45; 3], [0.0; 3]);
        blocker.material.emission = 0.0;
        scene.objects.push(blocker);
        gpu.step(&mut target, &scene, 64, 0, None, false);
        let shadowed = gpu.scene_linear(&target);
        let mut mask_scene = scene.clone();
        mask_scene.lights.clear();
        mask_scene.objects[1].material.emission = 1.0;
        mask_scene.objects[1].material.emission_color = [1.0; 3];
        gpu.step(&mut target, &mask_scene, 64, 0, None, false);
        let silhouette = gpu.scene_linear(&target);
        assert!(
            clear
                .iter()
                .zip(&shadowed)
                .zip(&silhouette)
                .any(|((clear, shadow), mask)| mask[0] == 0.0
                    && clear[0] > 0.1
                    && shadow[0] < 0.7 * clear[0]),
            "Sibling must block NEE rays on pixels where it is absent from the camera silhouette"
        );
    }

    #[test]
    fn hdr_environment_lights_surfaces_and_preserves_background_radiance() {
        let path = std::env::temp_dir().join(format!("frac-env-gpu-{}.exr", std::process::id()));
        exr::prelude::write_rgb_file(&path, 8, 4, |_, _| (4.0f32, 2.0f32, 1.0f32)).unwrap();
        let mut gpu = Gpu::new().unwrap();
        let mut target = gpu.target(16, 8);
        let mut scene = Scene::preset(FAMILY_QUAT);
        scene.environment.path = path.to_string_lossy().into_owned();
        scene.lighting.sun_intensity = 0.0;
        scene.colour.on = false;
        scene.render.exposure_stops = 0.0;
        scene.render.denoise.enabled = false;
        scene.render.max_bounces = 0;
        scene.camera.target = [1000.0, 0.0, 0.0];
        gpu.step(&mut target, &scene, 2, 0, None, false);
        let raw = gpu.scene_linear(&target);
        for p in &raw {
            assert_eq!(&p[..3], &[4.0, 2.0, 1.0]);
        }
        // The shared HDR map follows every per-object palette in the world LUT.
        scene.world_render = true;
        scene.objects = vec![Scene::preset(FAMILY_BULB), Scene::preset(FAMILY_QUAT)];
        gpu.step(&mut target, &scene, 2, 0, None, false);
        assert_eq!(gpu.scene_linear(&target), raw);
        scene.objects.clear();
        gpu.step(&mut target, &scene, 2, 0, None, false);
        assert_eq!(
            gpu.scene_linear(&target),
            raw,
            "Empty worlds retain the active HDR background"
        );
        scene.world_render = false;
        scene.environment.intensity = 2.0;
        gpu.step(&mut target, &scene, 2, 0, None, false);
        assert_eq!(target.samples, 2, "Environment edits reset accumulation");
        for p in gpu.scene_linear(&target) {
            assert_eq!(&p[..3], &[8.0, 4.0, 2.0]);
        }
        scene.camera.target = [0.0; 3];
        scene.lighting.background = false;
        scene.environment.intensity = 1.0;
        gpu.step(&mut target, &scene, 8, 0, None, false);
        let lit = gpu.scene_linear(&target);
        assert!(
            lit.iter().any(|p| p[0] > 0.01),
            "Map must illuminate hits even with background hidden"
        );
        scene.environment.enabled = false;
        scene.lighting.sky_intensity = 0.0;
        gpu.step(&mut target, &scene, 8, 0, None, false);
        assert!(gpu.scene_linear(&target).iter().all(|p| p[..3] == [0.0; 3]));
        std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn cuda_float_output_reuses_samples_and_exports_hdr_metadata() {
        let mut gpu = Gpu::new().unwrap();
        let mut target = gpu.target(17, 9); // partial CUDA tiles
        let mut scene = Scene::preset(FAMILY_KIFS);
        scene.render.denoise.enabled = false;
        gpu.step(&mut target, &scene, 2, 0, None, false);
        assert_eq!(target.samples, 2);
        assert!(target.colour_error.is_none(), "{:?}", target.colour_error);
        let old = target.light.clone();
        let radiance = gpu.scene_linear(&target);
        assert_eq!(radiance.len(), 17 * 9);
        assert!(radiance.iter().all(|p| p.iter().all(|v| v.is_finite())));
        assert!(
            radiance
                .iter()
                .any(|p| p[0] > 0.0 || p[1] > 0.0 || p[2] > 0.0)
        );
        scene.render.exposure_stops += 1.0;
        assert!(!gpu.prepare_target(&mut target, &scene, None));
        assert_eq!(
            target.samples, 2,
            "Display-only preparation must preserve accumulation"
        );
        gpu.step(&mut target, &scene, 0, 0, None, false);
        assert_eq!(target.samples, 2);
        assert_ne!(target.light, old);
        assert_eq!(
            gpu.scene_linear(&target),
            radiance,
            "Exposure must not alter exported scene radiance"
        );
        scene.colour.display = "Rec.2100-PQ - Display".into();
        scene.colour.view = crate::color::HDR_VIEW.into();
        gpu.step(&mut target, &scene, 0, 0, None, false);
        assert_eq!(target.samples, 2);
        assert!(
            target.hdr && target.colour_error.is_none(),
            "{:?}",
            target.colour_error
        );
        assert_eq!(
            gpu.scene_linear(&target),
            radiance,
            "OCIO must not alter exported scene radiance"
        );
        let dir = std::env::temp_dir().join(format!("frac-hdr-test-{}", std::process::id()));
        let png = dir.join("hdr.png");
        target.save_png(&png).unwrap();
        let bytes = std::fs::read(&png).unwrap();
        for chunk in [b"cICP", b"mDCV", b"cLLI"] {
            assert!(bytes.windows(4).any(|b| b == chunk));
        }
        let mut decoder = png::Decoder::new(std::io::Cursor::new(&bytes));
        decoder.set_transformations(png::Transformations::IDENTITY);
        let reader = decoder.read_info().unwrap();
        assert_eq!(reader.info().bit_depth, png::BitDepth::Sixteen);
        assert_eq!(reader.info().width, 17);
        let exr_path = dir.join("display.exr");
        target.save_display_exr(&exr_path).unwrap();
        let image = exr::prelude::read_all_flat_layers_from_file(&exr_path).unwrap();
        assert!(image.attributes.chromaticities.is_some());
        assert_eq!(image.layer_data[0].attributes.white_luminance, Some(100.0));
        scene.colour.view = "missing view".into();
        gpu.step(&mut target, &scene, 0, 0, None, false);
        assert!(target.colour_error.is_some());
        assert!(target.save_png(&png).is_err());
        let display_before_reset = target.light.clone();
        scene.camera.yaw_degrees += 5.0;
        assert!(gpu.prepare_target(&mut target, &scene, None));
        assert_eq!(target.samples, 0);
        assert_eq!(
            target.light, display_before_reset,
            "Preparation must not run tonemap, OCIO or readback"
        );
        scene.colour.view = crate::color::HDR_VIEW.into();
        gpu.step(&mut target, &scene, 1, 0, None, false);
        assert_eq!(
            target.samples, 1,
            "One traced batch must follow the lightweight reset"
        );
        assert!(target.colour_error.is_none());
        std::fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn old_bookmarks_default_to_linear_rec709_aces2() {
        let scene = Scene::preset(FAMILY_BULB);
        let mut json = serde_json::to_value(&scene).unwrap();
        json.as_object_mut().unwrap().remove("colour");
        let restored: Scene = serde_json::from_value(json).unwrap();
        assert_eq!(restored.colour, crate::color::default_selection());
    }
}
