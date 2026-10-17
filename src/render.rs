//! Host side of the GPU: one CUDA context, the loaded kernel module, and progressive render
//! targets (accumulator + float display light) that the viewport, the gallery thumbnails and final renders use.

use std::sync::Arc;
use std::time::Instant;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};

use crate::palette::{PaletteScheme, build_lut};
use crate::params::*;
#[cfg(test)]
use crate::scene::MaterialModel;
use crate::scene::Scene;
use fractal_kernels::kernels;

pub use crate::render_prep::PreparationStats;
pub(crate) use crate::render_prep::*;

fn read_tiled_into(
    stream: &Arc<CudaStream>,
    buffer: &DeviceBuffer<[f32; 4]>,
    width: usize,
    height: usize,
    normalize: bool,
    tiled: &mut Vec<[f32; 4]>,
    output: &mut Vec<[f32; 4]>,
) {
    tiled.resize(buffer.len(), [0.0; 4]);
    buffer
        .copy_to_host(stream, tiled)
        .expect("scene / guide readback");
    untile_into(tiled, width, height, normalize, output);
}

struct CudaEngine {
    _ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: kernels::LoadedModule,
    lut: DeviceBuffer<[f32; 4]>,
    objects: DeviceBuffer<f32>,
    lights: DeviceBuffer<f32>,
    lut_resident: Resident<LutResidentKey>,
    object_resident: Resident<ObjectResidentKey>,
    light_resident: Resident<LightResidentKey>,
    name: String,
}

impl CudaEngine {
    fn new() -> Result<Self, String> {
        let started = std::time::Instant::now();
        log::info!("CUDA startup: creating context");
        let ctx = CudaContext::new(0).map_err(|e| format!("CUDA context: {e:?}"))?;
        log::info!("CUDA startup: context ready in {:?}", started.elapsed());
        let stream = ctx.default_stream();
        let module_started = std::time::Instant::now();
        log::info!("CUDA startup: loading embedded kernels");
        // SAFETY: this package owns the embedded device bundle for `kernels`.
        let module = unsafe { kernels::load(&ctx) }.map_err(|e| format!("load module: {e:?}"))?;
        // Validate the host/device constant-memory contract before reporting Ready.
        // A stripped/mismatched artifact must surface as a startup error, not kill
        // the shared worker when its first viewport or thumbnail request arrives.
        module
            .set_params_blocking(&[0.0; P_COUNT])
            .map_err(|e| format!("CUDA parameter block: {e:?}"))?;
        log::info!(
            "CUDA startup: kernels ready in {:?}",
            module_started.elapsed()
        );
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
            lut_resident: Default::default(),
            object_resident: Default::default(),
            light_resident: Default::default(),
            name,
        })
    }

    fn create_target(&self, width: usize, height: usize) -> CudaTarget {
        let padded = width.div_ceil(8) * height.div_ceil(4) * 32;
        let mut active = DeviceBuffer::zeroed(&self.stream, padded / 32).expect("tile flags");
        active
            .copy_from_host(&self.stream, &vec![1u32; padded / 32])
            .expect("activate tiles");
        CudaTarget {
            accum: DeviceBuffer::zeroed(&self.stream, padded).expect("accumulator"),
            albedo: DeviceBuffer::zeroed(&self.stream, padded).expect("albedo guides"),
            normal: DeviceBuffer::zeroed(&self.stream, padded).expect("normal guides"),
            stats: DeviceBuffer::zeroed(&self.stream, padded).expect("sample statistics"),
            tally: DeviceBuffer::zeroed(&self.stream, TALLY_THREADS).expect("tally sums"),
            active,
            out: DeviceBuffer::zeroed(&self.stream, width * height).expect("output"),
            padded,
        }
    }

    fn clear_target(&self, ct: &mut CudaTarget) {
        ct.accum
            .zero_async(&self.stream)
            .expect("clear accumulator");
        ct.albedo
            .zero_async(&self.stream)
            .expect("clear albedo guides");
        ct.normal
            .zero_async(&self.stream)
            .expect("clear normal guides");
        ct.stats
            .zero_async(&self.stream)
            .expect("clear sample statistics");
    }

    fn upload_prepared(
        &mut self,
        prepared: &Arc<PreparedScene>,
        env: Option<&crate::environment::Map>,
        world: bool,
        table: &mut Vec<[f32; 4]>,
        stats: &mut PreparationStats,
    ) -> Result<(), String> {
        let stream = &self.stream;
        let lut = &mut self.lut;
        let changed = self
            .lut_resident
            .update(LutResidentKey(prepared.clone()), || {
                table.clear();
                for palette in &prepared.palettes {
                    table.extend(build_lut(*palette));
                }
                if let Some(env) = env {
                    table.extend(env.map.gpu_entries());
                }
                if lut.len() != table.len() {
                    *lut = DeviceBuffer::zeroed(stream, table.len())
                        .map_err(|e| format!("environment allocation: {e:?}"))?;
                }
                lut.copy_from_host(stream, table)
                    .map_err(|e| format!("palette / environment upload: {e:?}"))
            })?;
        if changed {
            stats.lut_uploads += 1;
            stats.upload_bytes += (table.len() * std::mem::size_of::<[f32; 4]>()) as u64;
        }
        if world {
            let objects = &mut self.objects;
            let data = &prepared.upload.objects;
            let changed =
                self.object_resident
                    .update(ObjectResidentKey(prepared.clone()), || {
                        if objects.len() != data.len().max(1) {
                            *objects = DeviceBuffer::zeroed(stream, data.len().max(1))
                                .map_err(|e| format!("world object allocation: {e:?}"))?;
                        }
                        if !data.is_empty() {
                            objects
                                .copy_from_host(stream, data)
                                .map_err(|e| format!("world objects upload: {e:?}"))?;
                        }
                        Ok(())
                    })?;
            if changed && !data.is_empty() {
                stats.object_uploads += 1;
                stats.upload_bytes += (data.len() * std::mem::size_of::<f32>()) as u64;
            }
            let lights = &mut self.lights;
            let data = &prepared.upload.lights;
            let changed = self
                .light_resident
                .update(LightResidentKey(prepared.clone()), || {
                    if lights.len() != data.len().max(1) {
                        *lights = DeviceBuffer::zeroed(stream, data.len().max(1))
                            .map_err(|e| format!("world light allocation: {e:?}"))?;
                    }
                    if !data.is_empty() {
                        lights
                            .copy_from_host(stream, data)
                            .map_err(|e| format!("world lights upload: {e:?}"))?;
                    }
                    Ok(())
                })?;
            if changed && !data.is_empty() {
                stats.light_uploads += 1;
                stats.upload_bytes += (data.len() * std::mem::size_of::<f32>()) as u64;
            }
        }
        Ok(())
    }
}

struct CudaTarget {
    accum: DeviceBuffer<[f32; 4]>,
    albedo: DeviceBuffer<[f32; 4]>,
    normal: DeviceBuffer<[f32; 4]>,
    /// Per pixel: [sum of squared sample luminance (adaptive sampling variance), samples with
    /// an unresolved march, the camera-ray start the kernels cache (fractal-kernels
    /// `cone_start`), 0]. Zeroed with every accumulation reset, which the start cache relies on.
    stats: DeviceBuffer<[f32; 4]>,
    /// [out-of-steps samples, samples] partial sums (the `tally` kernel).
    tally: DeviceBuffer<[f64; 2]>,
    /// Per 8x4 tile: 1 = keep sampling, 0 = converged (written by the `adapt` kernel).
    active: DeviceBuffer<u32>,
    out: DeviceBuffer<[f32; 4]>,
    padded: usize,
}

pub struct Gpu {
    cuda: Result<CudaEngine, String>,
    wgpu: Option<Result<crate::render_wgpu::WgpuEngine, String>>,
    cuda_unavailable: Option<String>,
    preparation: PreparationCache,
    lut_staging: Vec<[f32; 4]>,
    denoise_scratch: DenoiseScratch,
    stats: PreparationStats,
    environments:
        std::collections::BTreeMap<(String, u64), Result<Arc<crate::environment::Map>, String>>,
    pub name: String,
    colour: crate::color::ColorPipeline,
    colour_revision: u64,
    denoiser: Option<Result<crate::denoise::Processor, String>>,
    /// Regression oracle only: exercise the unchanged runtime material dispatcher.
    #[cfg(test)]
    force_mixed_world: bool,
}

/// A progressive render of one scene at one size.
pub struct Target {
    pub width: usize,
    pub height: usize,
    cuda: Option<CudaTarget>,
    wgpu: Option<crate::render_wgpu::WgpuTarget>,
    /// Host copy of [out-of-steps samples, samples] partial sums (`tally` kernel).
    tally_host: Vec<[f64; 2]>,
    /// Share of the accumulated samples with an unresolved march (camera, bounce or shadow ray
    /// out of steps, `gpu::Outcome`): a visible share means holes or missing light - raise the
    /// step budget.
    pub unresolved: f32,
    active_host: Vec<u32>,
    /// Tiles still sampling / all tiles, after the last adaptive decision.
    pub active_tiles: (usize, usize),
    /// Every tile is under the adaptive noise threshold: further samples change nothing.
    pub converged: bool,
    pub denoise: crate::denoise::State,
    denoise_selected: bool,
    /// The viewport shows the raw samples; denoising still runs, so switching back is instant.
    pub raw_view: bool,
    pub active_backend: crate::render_profiles::Backend,
    requested_backend: Option<crate::render_profiles::Backend>,
    /// SDR monitor codes (R in the low byte), row major, after the last `step`.
    pub pixels: Vec<u32>,
    /// Linear display Rec.709 (HDR: 1 = 100 nits); never quantized.
    pub light: Vec<[f32; 4]>,
    raw: Vec<[f32; 4]>,
    /// Relative (SDR view) or absolute (HDR view, with its peak) display light.
    pub light_kind: crate::color::DisplayLight,
    pub colour_error: Option<String>,
    display_key: Option<DisplayKey>,
    pub samples: u32,
    key: Vec<f32>,
    palette: Option<PaletteScheme>,
    environment: Option<(String, u64)>,
    /// GPU milliseconds of the last traced batch and its sample count.
    pub last_ms: f32,
    pub last_spp: u32,
}

impl Target {
    /// The image shows the denoised result (not the raw samples).
    pub fn denoise_shown(&self) -> bool {
        self.denoise_selected
    }
    /// Finished for a target of `spp`: all samples taken, or adaptive sampling converged.
    pub fn complete(&self, spp: u32) -> bool {
        self.samples >= spp || self.converged
    }
}

impl Gpu {
    /// Resolve `requested` (`Auto`, `Cuda`, `Wgpu`) to a concrete engine (`Cuda` or `Wgpu`).
    /// Never falls back silently when an explicit backend (`Cuda` or `Wgpu`) is requested.
    pub fn resolve_backend(
        &mut self,
        requested: crate::render_profiles::Backend,
    ) -> Result<crate::render_profiles::Backend, String> {
        use crate::render_profiles::Backend;
        let cuda_err = self
            .cuda_unavailable
            .as_ref()
            .or_else(|| self.cuda.as_ref().err());
        match requested {
            Backend::Cuda => {
                if let Some(err) = cuda_err {
                    Err(err.clone())
                } else {
                    Ok(Backend::Cuda)
                }
            }
            Backend::Wgpu => match self
                .wgpu
                .get_or_insert_with(crate::render_wgpu::WgpuEngine::new)
            {
                Ok(_) => Ok(Backend::Wgpu),
                Err(err) => Err(err.clone()),
            },
            Backend::Auto => {
                if cuda_err.is_none() {
                    Ok(Backend::Cuda)
                } else {
                    match self
                        .wgpu
                        .get_or_insert_with(crate::render_wgpu::WgpuEngine::new)
                    {
                        Ok(_) => Ok(Backend::Wgpu),
                        Err(err) => Err(err.clone()),
                    }
                }
            }
        }
    }

    /// Every tile samples again (new accumulation, or adaptive sampling switched off).
    fn activate_all(&self, target: &mut Target) {
        target.active_host.fill(1);
        if target.active_backend == crate::render_profiles::Backend::Wgpu {
            if let (Some(Ok(wgpu)), Some(wt)) = (&self.wgpu, &target.wgpu) {
                wgpu.upload_active(wt, &target.active_host);
            }
        } else if let (Ok(cuda), Some(ct)) = (&self.cuda, &mut target.cuda) {
            ct.active
                .copy_from_host(&cuda.stream, &target.active_host)
                .expect("activate tiles");
        }
        target.active_tiles = (target.active_host.len(), target.active_host.len());
        target.converged = false;
    }

    /// The share of accumulated samples that ran out of march steps ([`Target::unresolved`]).
    fn tally(&self, target: &mut Target) {
        if target.active_backend == crate::render_profiles::Backend::Wgpu {
            if let (Some(Ok(wgpu)), Some(wt)) = (&self.wgpu, &target.wgpu) {
                target.unresolved = wgpu.tally(wt);
            }
            return;
        }
        let (Ok(cuda), Some(ct)) = (&self.cuda, &mut target.cuda) else {
            return;
        };
        let cfg = LaunchConfig1D::new(TALLY_THREADS as u32 / BLOCK, BLOCK, 0);
        let prepared = cuda.module.prepare_tally(cfg).expect("prepare tally");
        cuda.module
            .tally(&cuda.stream, &prepared, &ct.accum, &ct.stats, &mut ct.tally)
            .expect("tally");
        ct.tally
            .copy_to_host(&cuda.stream, &mut target.tally_host)
            .expect("tally readback");
        let (unresolved, samples) = target
            .tally_host
            .iter()
            .fold((0.0f64, 0.0f64), |(l, s), t| (l + t[0], s + t[1]));
        target.unresolved = if samples > 0.0 {
            (unresolved / samples) as f32
        } else {
            0.0
        };
    }

    /// Adaptive sampling after a traced batch: the `adapt` kernel decides per tile, the host reads
    /// the flags back for the scheduler (`converged`) and the status line (`active_tiles`).
    fn adapt(&self, target: &mut Target, scene: &Scene) {
        let adaptive = scene.render.adaptive;
        if !adaptive.enabled {
            if target.active_tiles.0 != target.active_tiles.1 {
                self.activate_all(target);
            }
            return;
        }
        if target.samples < adaptive.min() {
            return;
        }
        if target.active_backend == crate::render_profiles::Backend::Wgpu {
            if let (Some(Ok(wgpu)), Some(wt)) = (&self.wgpu, &target.wgpu) {
                let (active, converged) = wgpu.adapt(wt, &mut target.active_host);
                target.active_tiles = (active, target.active_host.len());
                target.converged = converged;
            }
            return;
        }
        let (Ok(cuda), Some(ct)) = (&self.cuda, &mut target.cuda) else {
            return;
        };
        let tiles = target.active_host.len() as u32;
        let cfg = LaunchConfig1D::new(tiles.div_ceil(BLOCK), BLOCK, 0);
        let prepared = cuda.module.prepare_adapt(cfg).expect("prepare adapt");
        cuda.module
            .adapt(
                &cuda.stream,
                &prepared,
                &ct.accum,
                &ct.stats,
                &mut ct.active,
            )
            .expect("adapt");
        ct.active
            .copy_to_host(&cuda.stream, &mut target.active_host)
            .expect("tile flags readback");
        let active = target.active_host.iter().filter(|&&a| a != 0).count();
        target.active_tiles = (active, target.active_host.len());
        target.converged = active == 0;
    }

    pub fn invalidate_colour(&mut self) {
        self.colour = crate::color::ColorPipeline::new();
        self.colour_revision = self.colour_revision.wrapping_add(1);
    }

    pub fn new() -> Result<Self, String> {
        let cuda = CudaEngine::new();
        let cuda_unavailable = cuda.as_ref().err().cloned();
        let (name, wgpu) = match &cuda {
            Ok(c) => (c.name.clone(), None),
            Err(cuda_err) => {
                log::warn!("CUDA unavailable ({cuda_err}); initializing WGPU compute backend");
                let wgpu = crate::render_wgpu::WgpuEngine::new().map_err(|wgpu_err| {
                    format!("{cuda_err}; WGPU fallback also failed: {wgpu_err}")
                })?;
                let name = wgpu.name().to_owned();
                (name, Some(Ok(wgpu)))
            }
        };
        Ok(Self {
            cuda,
            wgpu,
            cuda_unavailable,
            preparation: Default::default(),
            lut_staging: Vec::new(),
            denoise_scratch: Default::default(),
            stats: Default::default(),
            environments: Default::default(),
            name,
            colour: crate::color::ColorPipeline::new(),
            colour_revision: 0,
            denoiser: None,
            #[cfg(test)]
            force_mixed_world: false,
        })
    }

    pub fn target(&self, width: usize, height: usize) -> Target {
        let padded = width.div_ceil(8) * height.div_ceil(4) * 32;
        let active_backend = if self.cuda.is_ok() && self.cuda_unavailable.is_none() {
            crate::render_profiles::Backend::Cuda
        } else {
            crate::render_profiles::Backend::Wgpu
        };
        Target {
            width,
            height,
            cuda: None,
            wgpu: None,
            tally_host: vec![[0.0; 2]; TALLY_THREADS],
            unresolved: 0.0,
            active_host: vec![1; padded / 32],
            active_tiles: (padded / 32, padded / 32),
            converged: false,
            denoise: Default::default(),
            denoise_selected: false,
            raw_view: false,
            active_backend,
            requested_backend: None,
            pixels: vec![0; width * height],
            light: vec![[0.0; 4]; width * height],
            raw: vec![[0.0; 4]; width * height],
            light_kind: crate::color::DisplayLight::Relative,
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
        if target.denoise_selected
            && let Some(output) = &target.denoise.output
        {
            return output.clone();
        }
        self.raw_scene_linear(target)
    }

    pub fn raw_scene_linear(&self, target: &Target) -> Vec<[f32; 4]> {
        let mut tiled = Vec::new();
        let mut radiance = Vec::new();
        if target.active_backend == crate::render_profiles::Backend::Wgpu {
            if let (Some(Ok(wgpu)), Some(wt)) = (&self.wgpu, &target.wgpu) {
                wgpu.read_tiled_into(
                    wt,
                    &wt.accum,
                    (target.width, target.height),
                    true,
                    &mut tiled,
                    &mut radiance,
                );
            }
            return radiance;
        }
        if let (Ok(cuda), Some(ct)) = (&self.cuda, &target.cuda) {
            read_tiled_into(
                &cuda.stream,
                &ct.accum,
                target.width,
                target.height,
                true,
                &mut tiled,
                &mut radiance,
            );
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
        let resolved_backend = match self.resolve_backend(scene.render.backend) {
            Ok(backend) => backend,
            Err(error) => {
                target.colour_error = Some(error);
                return false;
            }
        };
        let prepared = match self
            .preparation
            .get(scene, target.width as u32, target.height as u32)
        {
            Ok(prepared) => prepared,
            Err(error) => {
                target.colour_error = Some(error);
                return false;
            }
        };
        let params = prepared.params(scene, bounces_cap);
        self.reset_target_for_params(target, scene, &prepared, &params, resolved_backend)
    }

    fn reset_target_for_params(
        &self,
        target: &mut Target,
        scene: &Scene,
        prepared: &PreparedScene,
        params: &[f32; P_COUNT],
        resolved_backend: crate::render_profiles::Backend,
    ) -> bool {
        let mut key = *params;
        for i in TONEMAP_ONLY
            .iter()
            .chain(PER_LAUNCH.iter())
            .chain(SCHEDULE_ONLY.iter())
        {
            key[*i] = 0.0;
        }
        if target.key.len() == P_COUNT + prepared.trace.len()
            && target.key[..P_COUNT] == key
            && target.key[P_COUNT..] == prepared.trace
            && target.palette == Some(scene.palette)
            && target.environment == prepared.environment
            && target.requested_backend == Some(scene.render.backend)
            && target.active_backend == resolved_backend
        {
            return false;
        }
        target.requested_backend = Some(scene.render.backend);
        target.active_backend = resolved_backend;
        let padded = target.width.div_ceil(8) * target.height.div_ceil(4) * 32;
        if resolved_backend == crate::render_profiles::Backend::Wgpu {
            if let Some(Ok(wgpu)) = &self.wgpu {
                let needs_alloc = target
                    .wgpu
                    .as_ref()
                    .map(|wt| wt.padded != padded)
                    .unwrap_or(true);
                if needs_alloc {
                    target.wgpu = Some(wgpu.create_target(target.width, target.height));
                }
                if let Some(wt) = &target.wgpu {
                    wgpu.clear_target(wt);
                }
            }
        } else if let Ok(cuda) = &self.cuda {
            let needs_alloc = target
                .cuda
                .as_ref()
                .map(|ct| ct.padded != padded)
                .unwrap_or(true);
            if needs_alloc {
                target.cuda = Some(cuda.create_target(target.width, target.height));
            } else if let Some(ct) = &mut target.cuda {
                cuda.clear_target(ct);
            }
        }
        target.unresolved = 0.0;
        self.activate_all(target);
        target.denoise.reset();
        target.denoise_selected = false;
        target.samples = 0;
        target.display_key = None;
        target.key.clear();
        target.key.extend_from_slice(&key);
        target.key.extend_from_slice(&prepared.trace);
        target.palette = Some(scene.palette);
        target.environment.clone_from(&prepared.environment);
        true
    }

    /// Validate and prepare the exact requested dimensions on the worker. Subsequent
    /// scheduler and launch preparation share this entry instead of packing it again.
    pub fn prepare_scene(
        &mut self,
        scene: &Scene,
        width: usize,
        height: usize,
    ) -> Result<(), String> {
        let prepared = self.preparation.get(scene, width as u32, height as u32)?;
        self.environment_for(prepared.environment.as_ref())?;
        Ok(())
    }

    /// The fractal under the normalized device point `ndc` (-1..1, y up) of the world `scene`
    /// seen in a `width` x `height` frame: its index in `scene.objects` and the hit's depth
    /// along the view axis, or None on a miss. One camera ray through the `pick` kernel,
    /// marched like the frame's own (the same estimators, transforms and masks), on this
    /// worker between launches: it uploads the scene's parameters as a launch does.
    pub fn pick(
        &mut self,
        scene: &Scene,
        width: usize,
        height: usize,
        ndc: [f32; 2],
    ) -> Result<Option<(usize, f32)>, String> {
        if !scene.world_render || scene.objects.is_empty() {
            return Ok(None);
        }
        let resolved_backend = self.resolve_backend(scene.render.backend)?;
        let prepared = self.preparation.get(scene, width as u32, height as u32)?;
        let env = self.environment_for(prepared.environment.as_ref())?;
        let p = prepared.params(scene, None);
        if resolved_backend == crate::render_profiles::Backend::Wgpu {
            let stats = &mut self.stats;
            let wgpu = self
                .wgpu
                .as_mut()
                .and_then(|r| r.as_mut().ok())
                .ok_or_else(|| "WGPU backend unavailable".to_string())?;
            wgpu.upload_prepared(&prepared, env.as_deref(), true, stats)?;
            wgpu.set_params(&p);
            return wgpu.pick(ndc);
        }
        let cuda = self
            .cuda
            .as_mut()
            .map_err(|e| format!("CUDA backend unavailable: {e}"))?;
        cuda.upload_prepared(
            &prepared,
            env.as_deref(),
            true,
            &mut self.lut_staging,
            &mut self.stats,
        )?;
        cuda.module
            .set_params(&cuda.stream, &p)
            .map_err(|e| format!("CUDA parameter upload: {e:?}"))?;
        let mut query = DeviceBuffer::zeroed(&cuda.stream, 1).map_err(|e| format!("{e:?}"))?;
        query
            .copy_from_host(&cuda.stream, &[[ndc[0], ndc[1], 0.0, 0.0]])
            .map_err(|e| format!("pick upload: {e:?}"))?;
        let mut out = DeviceBuffer::zeroed(&cuda.stream, 1).map_err(|e| format!("{e:?}"))?;
        let launch = cuda
            .module
            .prepare_pick(LaunchConfig1D::new(1, 32, 0))
            .map_err(|e| format!("pick: {e:?}"))?;
        cuda.module
            .pick(
                &cuda.stream,
                &launch,
                &cuda.objects,
                &cuda.lights,
                &query,
                &mut out,
            )
            .map_err(|e| format!("pick: {e:?}"))?;
        let mut hit = [[0.0f32; 4]];
        out.copy_to_host(&cuda.stream, &mut hit)
            .map_err(|e| format!("pick readback: {e:?}"))?;
        let [object, _, depth, _] = hit[0];
        Ok((object >= 1.0).then(|| (object as usize - 1, depth)))
    }

    pub fn preparation_stats(&self) -> PreparationStats {
        PreparationStats {
            builds: self.preparation.builds,
            hits: self.preparation.hits,
            ..self.stats
        }
    }

    fn environment_for(
        &mut self,
        key: Option<&(String, u64)>,
    ) -> Result<Option<Arc<crate::environment::Map>>, String> {
        let Some(key) = key else {
            return Ok(None);
        };
        if !self.environments.contains_key(key) {
            if self.environments.len() >= 2 {
                self.environments.clear();
            }
            self.stats.environment_decodes += 1;
            let loaded = crate::environment::Map::load(&key.0).map(Arc::new);
            self.environments.insert(key.clone(), loaded);
        }
        self.environments[key].clone().map(Some)
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
        self.step_with_display(target, scene, spp, seed, bounces_cap, final_pass, |_| true);
    }

    /// Like [`Self::step`], but runs OIDN, tonemap, readback and OCIO only when `display(target)`
    /// asks to present the updated film after sample accumulation and convergence checks.
    #[allow(clippy::too_many_arguments)]
    pub fn step_with_display(
        &mut self,
        target: &mut Target,
        scene: &Scene,
        spp: u32,
        seed: u32,
        bounces_cap: Option<u32>,
        final_pass: bool,
        display: impl FnOnce(&Target) -> bool,
    ) {
        let resolved_backend = match self.resolve_backend(scene.render.backend) {
            Ok(backend) => backend,
            Err(error) => {
                target.colour_error = Some(error);
                return;
            }
        };
        let prepared = match self
            .preparation
            .get(scene, target.width as u32, target.height as u32)
        {
            Ok(prepared) => prepared,
            Err(error) => {
                target.colour_error = Some(error);
                return;
            }
        };
        let mut p = prepared.params(scene, bounces_cap);
        self.reset_target_for_params(target, scene, &prepared, &p, resolved_backend);
        let env = match self.environment_for(prepared.environment.as_ref()) {
            Ok(env) => env,
            Err(error) => {
                target.colour_error = Some(error);
                return;
            }
        };
        // The environment block's slots, through the one packer ofx-rs uses too. The sun's
        // radiance is the packed light's, so the sun/map balance follows what the kernels light.
        let sun = std::array::from_fn(|c| p[P_LIGHT_COLOR + c] * p[P_LIGHT_INTENSITY]);
        fractal_env::pack(
            &mut p,
            env.as_deref().map(|env| &env.map),
            scene.environment.intensity.max(0.0),
            scene.environment.rotation_degrees,
            sun,
        );
        p[P_SAMPLE_BEGIN] = target.samples as f32;
        p[P_SPP] = spp as f32;
        p[P_SEED] = seed as f32;

        if resolved_backend == crate::render_profiles::Backend::Wgpu {
            let upload_res = {
                let stats = &mut self.stats;
                let wgpu = self
                    .wgpu
                    .as_mut()
                    .and_then(|r| r.as_mut().ok())
                    .expect("resolved WGPU backend");
                let res =
                    wgpu.upload_prepared(&prepared, env.as_deref(), scene.world_render, stats);
                if res.is_ok() {
                    wgpu.set_params(&p);
                }
                res
            };
            if let Err(error) = upload_res {
                target.colour_error = Some(error);
                return;
            }
        } else {
            let cuda = self.cuda.as_mut().expect("resolved CUDA backend");
            if let Err(error) = cuda.upload_prepared(
                &prepared,
                env.as_deref(),
                scene.world_render,
                &mut self.lut_staging,
                &mut self.stats,
            ) {
                target.colour_error = Some(error);
                return;
            }
            if let Err(error) = cuda.module.set_params(&cuda.stream, &p) {
                target.colour_error = Some(format!("CUDA parameter upload: {error:?}"));
                return;
            }
        }

        let t0 = Instant::now();
        if spp > 0 {
            let full = p[P_MATERIAL_MODEL] != 0.0;
            if resolved_backend == crate::render_profiles::Backend::Wgpu {
                #[cfg(test)]
                let force_mixed = self.force_mixed_world;
                #[cfg(not(test))]
                let force_mixed = false;
                if let (Some(Ok(wgpu)), Some(wt)) = (&self.wgpu, &target.wgpu) {
                    wgpu.launch_trace(wt, scene, &prepared, full, force_mixed);
                }
            } else if let (Ok(cuda), Some(ct)) = (&self.cuda, &mut target.cuda) {
                let padded = ct.accum.len() as u32;
                let cfg = LaunchConfig1D::new(padded.div_ceil(BLOCK), BLOCK, 0);
                let (m, s, lut, acc) = (&cuda.module, &cuda.stream, &cuda.lut, &mut ct.accum);
                macro_rules! launch {
                    ($prep:ident, $run:ident) => {{
                        let prepared = m.$prep(cfg).expect(stringify!($prep));
                        m.$run(
                            s,
                            &prepared,
                            lut,
                            &cuda.objects,
                            &cuda.lights,
                            acc,
                            &mut ct.albedo,
                            &mut ct.normal,
                            &ct.active,
                            &mut ct.stats,
                        )
                        .expect(stringify!($run));
                    }};
                }
                if scene.direct.is_some() {
                    if scene.world_render {
                        launch!(prepare_world_direct, world_direct)
                    } else {
                        match scene.formula.code() {
                            FAMILY_BULB => launch!(prepare_direct_bulb, direct_bulb),
                            FAMILY_BOX => launch!(prepare_direct_box, direct_box),
                            FAMILY_QUAT => launch!(prepare_direct_quat, direct_quat),
                            FAMILY_KIFS => launch!(prepare_direct_kifs, direct_kifs),
                            FAMILY_KLEINIAN => launch!(prepare_direct_kleinian, direct_kleinian),
                            FAMILY_PSEUDO_KLEINIAN => {
                                launch!(prepare_direct_pseudo_kleinian, direct_pseudo_kleinian)
                            }
                            FAMILY_APOLLONIAN => {
                                launch!(prepare_direct_apollonian, direct_apollonian)
                            }
                            _ => launch!(prepare_direct_hybrid, direct_hybrid),
                        }
                    }
                } else if scene.world_render {
                    #[cfg(test)]
                    let world_material = if self.force_mixed_world {
                        WorldMaterial::Mixed
                    } else {
                        prepared.world_material
                    };
                    #[cfg(not(test))]
                    let world_material = prepared.world_material;
                    match world_material {
                        WorldMaterial::Fast if prepared.single_fast_bulb => {
                            launch!(prepare_world_fast_bulb, world_fast_bulb)
                        }
                        WorldMaterial::Fast => launch!(prepare_world_fast, world_fast),
                        WorldMaterial::Full | WorldMaterial::Mixed => launch!(prepare_world, world),
                    }
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
                        (false, FAMILY_APOLLONIAN) => {
                            launch!(prepare_fast_apollonian, fast_apollonian)
                        }
                        (false, _) => launch!(prepare_fast_hybrid, fast_hybrid),
                        (true, FAMILY_BULB) => launch!(prepare_full_bulb, full_bulb),
                        (true, FAMILY_BOX) => launch!(prepare_full_box, full_box),
                        (true, FAMILY_QUAT) => launch!(prepare_full_quat, full_quat),
                        (true, FAMILY_KIFS) => launch!(prepare_full_kifs, full_kifs),
                        (true, FAMILY_KLEINIAN) => launch!(prepare_full_kleinian, full_kleinian),
                        (true, FAMILY_PSEUDO_KLEINIAN) => {
                            launch!(prepare_full_pseudo_kleinian, full_pseudo_kleinian)
                        }
                        (true, FAMILY_APOLLONIAN) => {
                            launch!(prepare_full_apollonian, full_apollonian)
                        }
                        (true, _) => launch!(prepare_full_hybrid, full_hybrid),
                    }
                }
            }
            target.samples += spp;
            target.display_key = None;
            self.tally(target);
            self.adapt(target, scene);
        }
        if !display(target) {
            if spp > 0 {
                target.last_ms = t0.elapsed().as_secs_f32() * 1000.0;
                target.last_spp = spp;
            }
            return;
        }
        // Converging early ends the render: run the final-pass work (OIDN at completion) now.
        let final_pass = final_pass || target.converged;
        let denoise_changed = self.process_denoise(target, scene, final_pass);
        let display_changed = !target
            .display_key
            .as_ref()
            .is_some_and(|key| key.matches(scene, self.colour_revision));
        if spp == 0 && !denoise_changed && !display_changed {
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
                let l = crate::color::luma(c);
                *raw = [
                    l + (c[0] - l) * p[P_SATURATION],
                    l + (c[1] - l) * p[P_SATURATION],
                    l + (c[2] - l) * p[P_SATURATION],
                    1.0,
                ];
            }
        } else if resolved_backend == crate::render_profiles::Backend::Wgpu {
            if let (Some(Ok(wgpu)), Some(wt)) = (&self.wgpu, &target.wgpu) {
                wgpu.tonemap(wt, target.width, target.height, &mut target.raw);
            }
        } else if let (Ok(cuda), Some(ct)) = (&self.cuda, &mut target.cuda) {
            let n = (target.width * target.height) as u32;
            let cfg = LaunchConfig1D::new(n.div_ceil(BLOCK), BLOCK, 0);
            let prepared = cuda.module.prepare_tonemap(cfg).expect("prepare tonemap");
            cuda.module
                .tonemap(&cuda.stream, &prepared, &ct.accum, &mut ct.out)
                .expect("tonemap");
            ct.out
                .copy_to_host(&cuda.stream, &mut target.raw)
                .expect("readback");
        }
        match self.colour.apply(
            target.width,
            target.height,
            &target.raw,
            &scene.colour,
            scene.render.reinhard,
        ) {
            Ok((light, encoded, kind)) => {
                target.light = light;
                target.light_kind = kind;
                target.colour_error = None;
                if display_changed {
                    target.display_key = Some(DisplayKey::new(scene, self.colour_revision));
                }
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

    #[cfg(test)]
    fn read_guide_sums(
        &self,
        target: &Target,
        cuda_buf: impl FnOnce(&CudaTarget) -> &DeviceBuffer<[f32; 4]>,
        wgpu_buf: impl FnOnce(&crate::render_wgpu::WgpuTarget) -> &wgpu::Buffer,
    ) -> Vec<[f32; 4]> {
        let mut tiled = Vec::new();
        let mut guides = Vec::new();
        if target.active_backend == crate::render_profiles::Backend::Wgpu {
            if let (Some(Ok(wgpu)), Some(wt)) = (&self.wgpu, &target.wgpu) {
                wgpu.read_tiled_into(
                    wt,
                    wgpu_buf(wt),
                    (target.width, target.height),
                    false,
                    &mut tiled,
                    &mut guides,
                );
            }
            return guides;
        }
        if let (Ok(cuda), Some(ct)) = (&self.cuda, &target.cuda) {
            read_tiled_into(
                &cuda.stream,
                cuda_buf(ct),
                target.width,
                target.height,
                false,
                &mut tiled,
                &mut guides,
            );
        }
        guides
    }

    #[cfg(test)]
    fn accum_sums(&self, target: &Target) -> Vec<[f32; 4]> {
        self.read_guide_sums(target, |ct| &ct.accum, |wt| &wt.accum)
    }

    #[cfg(test)]
    fn albedo_sums(&self, target: &Target) -> Vec<[f32; 4]> {
        self.read_guide_sums(target, |ct| &ct.albedo, |wt| &wt.albedo)
    }

    #[cfg(test)]
    fn normal_sums(&self, target: &Target) -> Vec<[f32; 4]> {
        self.read_guide_sums(target, |ct| &ct.normal, |wt| &wt.normal)
    }

    fn process_denoise(&mut self, target: &mut Target, scene: &Scene, final_pass: bool) -> bool {
        let settings = &scene.render.denoise;
        let previous = target.denoise_selected;
        let due = target.denoise.due(settings, target.samples, final_pass);
        if due {
            let scratch = &mut self.denoise_scratch;
            if target.active_backend == crate::render_profiles::Backend::Wgpu {
                if let (Some(Ok(wgpu)), Some(wt)) = (&self.wgpu, &target.wgpu) {
                    let dims = (target.width, target.height);
                    wgpu.read_tiled_into(
                        wt,
                        &wt.accum,
                        dims,
                        true,
                        &mut scratch.tiled,
                        &mut scratch.color,
                    );
                    if settings.mode != crate::denoise::Mode::Color {
                        wgpu.read_tiled_into(
                            wt,
                            &wt.albedo,
                            dims,
                            false,
                            &mut scratch.tiled,
                            &mut scratch.albedo,
                        );
                    }
                    if settings.mode == crate::denoise::Mode::ColorAlbedoNormal {
                        wgpu.read_tiled_into(
                            wt,
                            &wt.normal,
                            dims,
                            false,
                            &mut scratch.tiled,
                            &mut scratch.normal,
                        );
                    }
                }
            } else if let (Ok(cuda), Some(ct)) = (&self.cuda, &target.cuda) {
                read_tiled_into(
                    &cuda.stream,
                    &ct.accum,
                    target.width,
                    target.height,
                    true,
                    &mut scratch.tiled,
                    &mut scratch.color,
                );
                if settings.mode != crate::denoise::Mode::Color {
                    read_tiled_into(
                        &cuda.stream,
                        &ct.albedo,
                        target.width,
                        target.height,
                        false,
                        &mut scratch.tiled,
                        &mut scratch.albedo,
                    );
                }
                if settings.mode == crate::denoise::Mode::ColorAlbedoNormal {
                    read_tiled_into(
                        &cuda.stream,
                        &ct.normal,
                        target.width,
                        target.height,
                        false,
                        &mut scratch.tiled,
                        &mut scratch.normal,
                    );
                }
            }
            let albedo =
                (settings.mode != crate::denoise::Mode::Color).then_some(scratch.albedo.as_slice());
            let normal = (settings.mode == crate::denoise::Mode::ColorAlbedoNormal)
                .then_some(scratch.normal.as_slice());
            let processor = self
                .denoiser
                .get_or_insert_with(crate::denoise::Processor::new);
            let result = match processor {
                Ok(processor) => processor.process(
                    crate::denoise::Image {
                        width: target.width,
                        height: target.height,
                        color: &scratch.color,
                        albedo,
                        normal,
                    },
                    settings,
                    target.samples,
                ),
                Err(error) => Err(error.clone()),
            };
            target.denoise.complete(target.samples, result);
        }
        target.denoise_selected =
            settings.enabled && target.denoise.output.is_some() && !target.raw_view;
        due || previous != target.denoise_selected
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

    pub fn save_png(
        &self,
        path: &std::path::Path,
        encoding: crate::render_service::PngEncoding,
        sdr_white_nits: f32,
        overwrite: bool,
    ) -> Result<(), String> {
        if let Some(e) = &self.colour_error {
            return Err(format!("Colour transform failed: {e}"));
        }
        let sdr = || self.pixels.iter().flat_map(|p| p.to_le_bytes()).collect();
        let scale = crate::render_service::hdr_scale(
            self.light_kind,
            &self.light,
            sdr_white_nits,
            encoding,
        );
        crate::render_service::write_png(
            path,
            self.width,
            self.height,
            &self.light,
            sdr,
            encoding,
            scale,
            overwrite,
        )
        .map(|_| ())
        .map_err(|e| e.to_string())
    }

    /// Display light, linear Rec.709 (`render_service::write_display_exr`): `whiteLuminance` 100
    /// for an HDR view's absolute light, `sdr_white_nits` for relative light.
    pub fn save_display_exr(
        &self,
        path: &std::path::Path,
        sdr_white_nits: f32,
    ) -> Result<(), String> {
        if let Some(e) = &self.colour_error {
            return Err(format!("Colour transform failed: {e}"));
        }
        crate::render_service::write_display_exr(
            path,
            self.width,
            self.height,
            &self.light,
            self.light_kind,
            sdr_white_nits,
            true,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Accumulated radiance, albedo and normal guides of `scene` at 96x64, one sample.
    fn frame_guides(gpu: &mut Gpu, scene: &Scene) -> [Vec<[f32; 4]>; 3] {
        let mut target = gpu.target(96, 64);
        gpu.step(&mut target, scene, 1, 19, None, false);
        [
            gpu.accum_sums(&target),
            gpu.albedo_sums(&target),
            gpu.normal_sums(&target),
        ]
    }

    /// How far two normal guides are apart: the share of pixels hit in one and missed in the
    /// other, and the mean normal difference over the pixels both hit.
    fn normal_gap(a: &[[f32; 4]], b: &[[f32; 4]]) -> (f32, f32) {
        let hit = |n: &[f32; 4]| n[..3] != [0.0; 3];
        let (mut flips, mut both, mut sum) = (0usize, 0usize, 0.0f32);
        for (x, y) in a.iter().zip(b) {
            match (hit(x), hit(y)) {
                (true, true) => {
                    both += 1;
                    sum += (0..3).map(|i| (x[i] - y[i]).abs()).sum::<f32>();
                }
                (false, false) => {}
                _ => flips += 1,
            }
        }
        (flips as f32 / a.len() as f32, sum / both.max(1) as f32)
    }

    /// The neutral defaults (Julia mix 0 / 1 as the former switch, integer iterations, no offset,
    /// no inversion) render as the kernels before these parameters did. `golden/
    /// neutral_defaults.bin` holds that build's guides (96x64, 2 samples, seed 19) of every
    /// gallery scene on both routes: an FNV-1a hash of the albedo and normal guides per route,
    /// then the normal guides of the Hybrid world routes. Every route matches bit for bit except
    /// Hybrid on the world route: NVPTX fuses multiply-adds per basic block, and the dispatch
    /// layout around the estimates changed (fractal-kernels `signed_distance_and_trap`), so its
    /// normals match within NEUTRAL_HYBRID_NORMAL (measured 2.6e-4) and its radiance, whose
    /// paths diverge from there, is not compared. The golden pins a GPU / driver: re-record it
    /// with a build of the former kernels if either changes.
    #[test]
    fn cuda_neutral_defaults_render_as_before_the_new_parameters() {
        const NEUTRAL_HYBRID_NORMAL: f32 = 1.0e-3;
        let golden = include_bytes!("golden/neutral_defaults.bin");
        let word = |at: usize| u32::from_le_bytes(golden[at..at + 4].try_into().unwrap());
        let routes = word(0) as usize;
        let hash_at =
            |r: usize| u64::from_le_bytes(golden[4 + 8 * r..12 + 8 * r].try_into().unwrap());
        let mut normals = golden[4 + 8 * routes..]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c));
        let fnv = |buffers: &[&Vec<[f32; 4]>]| {
            let mut h = 0xcbf2_9ce4_8422_2325u64;
            for v in buffers.iter().flat_map(|b| b.iter().flatten()) {
                for byte in v.to_le_bytes() {
                    h = (h ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3);
                }
            }
            h
        };
        let _gpu_test = crate::test_gpu::lock();
        let mut gpu = Gpu::new().unwrap();
        let gallery = Scene::gallery();
        assert_eq!(2 * gallery.len(), routes, "the golden covers the gallery");
        for (index, scene) in gallery.iter().enumerate() {
            for world in [false, true] {
                let s = if world {
                    crate::world::WorldDocument::from_scene(scene)
                        .snapshot(0.0)
                        .unwrap()
                } else {
                    scene.clone()
                };
                let mut target = gpu.target(96, 64);
                gpu.step(&mut target, &s, 2, 19, None, false);
                let albedo = gpu.albedo_sums(&target);
                let normal = gpu.normal_sums(&target);
                let name = format!("{} (world {world})", scene.name);
                if world && matches!(scene.formula, crate::scene::Formula::Hybrid(_)) {
                    for (k, n) in normal.iter().flatten().enumerate() {
                        let before = normals.next().unwrap();
                        assert!(
                            (n - before).abs() <= NEUTRAL_HYBRID_NORMAL,
                            "{name}: normal component {k} {n} was {before}"
                        );
                    }
                } else {
                    assert_eq!(
                        fnv(&[&albedo, &normal]),
                        hash_at(2 * index + world as usize),
                        "{name}"
                    );
                }
            }
        }
        assert!(normals.next().is_none(), "every golden normal was compared");
    }

    /// Values the kernels must not read change nothing, bit for bit, on every gallery scene and
    /// both routes: the inversion sphere at strength 0, the Julia constant at mix 0, the orbit
    /// controls of the families without an orbit constant.
    #[test]
    fn cuda_inactive_orbit_and_inversion_values_change_nothing() {
        let _gpu_test = crate::test_gpu::lock();
        let mut gpu = Gpu::new().unwrap();
        for scene in Scene::gallery() {
            let mut idle = scene.clone();
            idle.inversion_center = [0.4, -0.3, 2.5];
            idle.inversion_radius = 0.7;
            if idle.julia_mix == 0.0 {
                idle.julia = [0.6, -0.2, 0.1];
            }
            if !idle.formula.has_orbit_constant() {
                idle.julia_mix = 0.5;
                idle.orbit_offset = [0.3, 0.0, -0.2];
            }
            for world in [false, true] {
                let route = |s: &Scene| {
                    if world {
                        crate::world::WorldDocument::from_scene(s)
                            .snapshot(0.0)
                            .unwrap()
                    } else {
                        s.clone()
                    }
                };
                assert!(
                    frame_guides(&mut gpu, &route(&scene)) == frame_guides(&mut gpu, &route(&idle)),
                    "{} (world {world}): an unread value changed the render",
                    scene.name
                );
            }
        }
    }

    /// Fractional iterations, the Julia mix, the orbit offset and the inversion strength morph
    /// continuously: a step of 1e-3 moves the normals far less than the whole unit step does.
    /// The iteration regression also checks individual pixels when leaving an integer: a
    /// frame average alone hid the zero-gradient endpoint discontinuity.
    #[test]
    fn cuda_animated_formula_controls_are_continuous() {
        let _gpu_test = crate::test_gpu::lock();
        let _ = env_logger::try_init();
        let initialized = std::time::Instant::now();
        println!("fractional: initializing CUDA module");
        let mut gpu = Gpu::new().unwrap();
        println!(
            "fractional: CUDA initialized in {:?}",
            initialized.elapsed()
        );
        let mut base = Scene::preset(FAMILY_BULB);
        base.render.iterations = 4.0;
        let normals = |gpu: &mut Gpu, edit: &dyn Fn(&mut Scene)| {
            let mut s = base.clone();
            edit(&mut s);
            let [_, _, normal] = frame_guides(gpu, &s);
            normal
        };
        // (control, its value at v in 0..1, the starts of the 1e-3 steps probed)
        type Edit = Box<dyn Fn(&mut Scene, f32)>;
        let controls: [(&str, Edit, [f32; 2]); 4] = [
            (
                "iterations",
                Box::new(|s, v| s.render.iterations = 4.0 + v),
                [0.5, 0.999],
            ),
            (
                "julia mix",
                Box::new(|s, v| {
                    s.julia = [0.35, 0.35, -0.4];
                    s.julia_mix = v;
                }),
                [0.0, 0.999],
            ),
            (
                "orbit offset",
                Box::new(|s, v| s.orbit_offset = [0.3 * v, 0.0, 0.0]),
                [0.0, 0.999],
            ),
            (
                "inversion",
                Box::new(|s, v| {
                    s.inversion_center = [0.0, 0.0, 3.0];
                    s.inversion_radius = 2.0;
                    s.inversion_strength = v;
                }),
                [0.0, 0.999],
            ),
        ];
        for (name, edit, starts) in &controls {
            let whole = normal_gap(
                &normals(&mut gpu, &|s| edit(s, 0.0)),
                &normals(&mut gpu, &|s| edit(s, 1.0)),
            );
            assert!(whole.1 > 0.0, "{name}: the unit step must change the shape");
            for start in *starts {
                let a = normals(&mut gpu, &|s| edit(s, start));
                let b = normals(&mut gpu, &|s| edit(s, start + 0.001));
                let (flips, mean) = normal_gap(&a, &b);
                assert!(
                    flips <= whole.0.max(0.005) && 10.0 * mean < whole.1,
                    "{name} at {start}: a 1e-3 step jumped (flips {flips}, normals {mean},                      unit step {whole:?})"
                );
            }
        }
        // Probe both sides of the integer on the single and world routes. Bound
        // each pixel's change so a frame average cannot hide discontinuities.
        let camera = base.pack(96, 64);
        let v = |i| glam::Vec3::from_slice(&camera[i..i + 3]);
        let (forward, right, up) = (v(P_CAM_FORWARD), v(P_CAM_RIGHT), v(P_CAM_UP));
        for world in [false, true] {
            let sample = |gpu: &mut Gpu, iterations: f32| {
                let mut scene = base.clone();
                scene.render.iterations = iterations;
                if world {
                    scene = crate::world::WorldDocument::from_scene(&scene)
                        .snapshot(0.0)
                        .unwrap();
                }
                let started = std::time::Instant::now();
                let normal = frame_guides(gpu, &scene)[2].clone();
                println!(
                    "fractional: world={world}, n={iterations}, render {:?}",
                    started.elapsed()
                );
                assert!(normal.iter().flatten().all(|n| n.is_finite()));
                normal
            };
            let unit = normal_gap(&sample(&mut gpu, 4.0), &sample(&mut gpu, 5.0));
            for (a, b) in [(4.0, 4.000001), (4.0, 4.001), (4.999, 5.0), (5.0, 5.001)] {
                let x = sample(&mut gpu, a);
                let y = sample(&mut gpu, b);
                let mut both = 0;
                let mut jumps = 0;
                let mut maximum = 0.0f32;
                let mut fallback = 0;
                let mut fallback_maximum = 0.0f32;
                for (pixel, (x, y)) in x.iter().zip(&y).enumerate() {
                    if x[..3] != [0.0; 3] && y[..3] != [0.0; 3] {
                        both += 1;
                        let gap = glam::Vec3::from_slice(x).distance(glam::Vec3::from_slice(y));
                        maximum = maximum.max(gap);
                        jumps += (gap > 0.25) as usize;
                        let (px, py) = ((pixel % 96) as u32, (pixel / 96) as u32);
                        let nx = 2.0
                            * (px as f32
                                + standard_surface_bsdf::sampling::sobol(px ^ 19, py, 0, 0))
                            / 96.0
                            - 1.0;
                        let ny = 1.0
                            - 2.0
                                * (py as f32
                                    + standard_surface_bsdf::sampling::sobol(px ^ 19, py, 0, 1))
                                / 64.0;
                        let ray =
                            (forward + right * nx * camera[P_HALF_W] + up * ny * camera[P_HALF_H])
                                .normalize();
                        if a.fract() == 0.0 && glam::Vec3::from_slice(x).dot(-ray) > 0.999999 {
                            fallback += 1;
                            fallback_maximum = fallback_maximum.max(gap);
                        }
                    }
                }
                println!(
                    "fractional world={world}, {a}->{b}: hits={both}, jumps={jumps}, max={maximum}, fallback={fallback}, fallback_max={fallback_maximum}"
                );
                assert!(both > 3000, "fixture must exercise the interior endpoint");
                let (flips, mean) = normal_gap(&x, &y);
                assert!(
                    flips <= unit.0.max(0.005) && 10.0 * mean < unit.1,
                    "world={world}, {a}->{b}: flips={flips}, mean={mean}, unit={unit:?}"
                );
                assert!(
                    maximum < 0.25,
                    "world={world}, {a}->{b}: maximum per-pixel normal change {maximum}"
                );
            }
        }
        // Mix 1 is the Julia form: the constant alone, whatever the point.
        let julia = normals(&mut gpu, &|s| {
            s.julia = [0.35, 0.35, -0.4];
            s.julia_mix = 1.0;
        });
        assert!(julia.iter().any(|n| n[..3] != [0.0; 3]));
    }

    /// A camera inside the finite four-iteration set but outside the fifth probes
    /// the actual zero-gradient endpoint, independently of cone-start hit positions.
    #[test]
    fn cuda_fractional_normals_leave_zero_gradient_endpoints_continuously() {
        let _gpu_test = crate::test_gpu::lock();
        let mut gpu = Gpu::new().unwrap();
        let mut scene = Scene::preset(FAMILY_BULB);
        let p = scene.pack(96, 64);
        let v = |i| glam::Vec3::from_slice(&p[i..i + 3]);
        let (forward, right, up) = (v(P_CAM_FORWARD), v(P_CAM_RIGHT), v(P_CAM_UP));
        let nx =
            2.0 * (45.0 + standard_surface_bsdf::sampling::sobol(45 ^ 19, 16, 0, 0)) / 96.0 - 1.0;
        let ny =
            1.0 - 2.0 * (16.0 + standard_surface_bsdf::sampling::sobol(45 ^ 19, 16, 0, 1)) / 64.0;
        let ray = (forward + right * nx * p[P_HALF_W] + up * ny * p[P_HALF_H]).normalize();
        // CUDA pick measured this point at n=4.000001 on the original kernels.
        let point = v(P_CAM_ORIGIN) + ray * (2.4022722 / forward.dot(ray));
        scene.camera.distance = 0.1;
        scene.camera.target =
            (point + forward * (scene.camera.distance * scene.formula.framing_radius())).to_array();
        scene.camera.fov_y_degrees = 0.5;
        scene.camera.f_number = 0.0;
        scene.render.iterations = 4.0;
        scene.render.max_bounces = 0;
        scene.render.adaptive.enabled = false;
        scene.render.denoise.enabled = false;
        for world in [true, false] {
            let route = |n| {
                let mut s = scene.clone();
                s.render.iterations = n;
                if world {
                    crate::world::WorldDocument::from_scene(&s)
                        .snapshot(0.0)
                        .unwrap()
                } else {
                    s
                }
            };
            let a = frame_guides(&mut gpu, &route(4.0))[2].clone();
            let p = route(4.0).pack(96, 64);
            let v = |i| glam::Vec3::from_slice(&p[i..i + 3]);
            let (forward, right, up) = (v(P_CAM_FORWARD), v(P_CAM_RIGHT), v(P_CAM_UP));
            let mut fallback = 0;
            for (i, a) in a.iter().enumerate() {
                let (x, y) = ((i % 96) as u32, (i / 96) as u32);
                let nx = 2.0 * (x as f32 + standard_surface_bsdf::sampling::sobol(x ^ 19, y, 0, 0))
                    / 96.0
                    - 1.0;
                let ny = 1.0
                    - 2.0 * (y as f32 + standard_surface_bsdf::sampling::sobol(x ^ 19, y, 0, 1))
                        / 64.0;
                let ray = (forward + right * nx * p[P_HALF_W] + up * ny * p[P_HALF_H]).normalize();
                fallback += (glam::Vec3::from_slice(a).dot(-ray) > 0.999999) as usize;
            }
            println!(
                "zero-gradient world={world}: fallback={fallback}/{}",
                a.len()
            );
            assert!(
                fallback * 100 > 99 * a.len(),
                "fixture must exercise actual integer fallback normals"
            );
            // This camera lies inside the zero-gradient integer endpoint. The
            // smallest positive fraction exercises shading while retaining hits;
            // larger steps also exercise the unsigned DE's march convergence.
            let n = 4.000001;
            let b = frame_guides(&mut gpu, &route(n))[2].clone();
            let mut maximum = 0.0f32;
            let mut both = 0;
            for (a, b) in a.iter().zip(&b) {
                if a[..3] != [0.0; 3] && b[..3] != [0.0; 3] {
                    maximum =
                        maximum.max(glam::Vec3::from_slice(a).distance(glam::Vec3::from_slice(b)));
                    both += 1;
                }
            }
            println!("zero-gradient world={world}, 4->{n}: hits={both}, max={maximum}");
            assert!(both * 100 > 99 * a.len());
            assert!(
                maximum < 0.01,
                "zero-gradient endpoint jumped: world={world}, 4->{n}, max={maximum}"
            );
        }
    }

    #[test]
    fn fractional_endpoint_margin_preserves_a_grazing_weighted_sphere_band() {
        let (weight, eps, y, start) = (0.001f64, 0.0001, 1.09999, -0.04);
        let sphere = |x: f64| x.hypot(y) - 1.0;
        let raw_next = start + 0.8 * sphere(start);
        assert!(weight * sphere(start) > eps && weight * sphere(raw_next) > eps);
        assert!(
            weight * sphere(0.0) < eps,
            "raw endpoint maximum skipped the entire hit band"
        );
        let boundary = -((1.0 + eps / weight).powi(2) - y * y).sqrt();
        let mut x = start;
        let mut accelerated = 0;
        for _ in 0..4096 {
            let weighted = weight * sphere(x);
            if weighted <= eps {
                break;
            }
            let margin = ((weighted - eps) / weight).max(0.0);
            let next = x + (0.8 * weighted.max(margin)).max(0.5 * eps);
            if margin > weighted {
                assert!(
                    next <= boundary,
                    "an accelerated step crossed the analytic band entry"
                );
                accelerated += 1;
            }
            x = next;
        }
        assert!(accelerated > 0, "fixture must exercise acceleration");
        assert!(
            weight * sphere(x) <= eps,
            "margin marcher must resolve the grazing band"
        );
    }

    #[test]
    fn cuda_fractional_bulb_march_converges_without_erasing_the_weighted_hit_band() {
        let _gpu_test = crate::test_gpu::lock();
        let mut gpu = Gpu::new().unwrap();
        let mut scene = Scene::preset(FAMILY_BULB);
        let p = scene.pack(96, 64);
        let v = |i| glam::Vec3::from_slice(&p[i..i + 3]);
        let (forward, right, up) = (v(P_CAM_FORWARD), v(P_CAM_RIGHT), v(P_CAM_UP));
        let nx =
            2.0 * (45.0 + standard_surface_bsdf::sampling::sobol(45 ^ 19, 16, 0, 0)) / 96.0 - 1.0;
        let ny =
            1.0 - 2.0 * (16.0 + standard_surface_bsdf::sampling::sobol(45 ^ 19, 16, 0, 1)) / 64.0;
        let ray = (forward + right * nx * p[P_HALF_W] + up * ny * p[P_HALF_H]).normalize();
        let point = v(P_CAM_ORIGIN) + ray * (2.4022722 / forward.dot(ray));
        scene.camera.distance = 0.1;
        scene.camera.target =
            (point + forward * (scene.camera.distance * scene.formula.framing_radius())).to_array();
        scene.camera.fov_y_degrees = 0.5;
        scene.camera.f_number = 0.0;
        scene.render.iterations = 4.001;
        scene.render.max_bounces = 0;
        scene.render.adaptive.enabled = false;
        scene.render.denoise.enabled = false;
        // Captured on f3e55bd with 16,384 steps, before the march enhancement.
        let expected = include_bytes!("golden/fractional_bulb_march.bin")
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| f32::from_le_bytes(*v))
            .collect::<Vec<_>>();
        let reference_depths = [
            0.012887325,
            0.012751972,
            0.0126214465,
            0.012838403,
            0.012704739,
            0.012574547,
            0.012789463,
            0.01265746,
            0.012528922,
        ];
        let mut failures = Vec::new();
        for world in [false, true] {
            let mut actual = if world {
                crate::world::WorldDocument::from_scene(&scene)
                    .snapshot(0.0)
                    .unwrap()
            } else {
                scene.clone()
            };
            actual.render.max_steps = 4096;
            for object in &mut actual.objects {
                object.render.max_steps = 4096;
            }
            let mut reference = actual.clone();
            reference.render.max_steps = 16384;
            for object in &mut reference.objects {
                object.render.max_steps = 16384;
            }
            let started = Instant::now();
            let high = frame_guides(&mut gpu, &reference)[2].clone();
            let reference_time = started.elapsed();
            let started = Instant::now();
            let low = frame_guides(&mut gpu, &actual)[2].clone();
            println!(
                "geometry world={world}: reference {reference_time:?}, default {:?}",
                started.elapsed()
            );
            let ref_hits = high.iter().filter(|n| n[..3] != [0.0; 3]).count();
            let hits = low.iter().filter(|n| n[..3] != [0.0; 3]).count();
            println!("geometry world={world}: default hits={hits}, 4x reference hits={ref_hits}");
            assert_eq!(
                ref_hits,
                96 * 64,
                "fixture's reference must converge every primary ray"
            );
            if hits != ref_hits {
                failures.push(format!(
                    "world={world}: default {hits}, reference {ref_hits}"
                ));
            }
            let mut normal_error = 0.0f32;
            for (a, b) in low.iter().zip(expected.as_chunks::<4>().0) {
                if a[..3] != [0.0; 3] {
                    normal_error = normal_error
                        .max(glam::Vec3::from_slice(a).distance(glam::Vec3::from_slice(b)));
                    assert!(
                        glam::Vec3::from_slice(a).distance(glam::Vec3::from_slice(b)) < 1.0e-3,
                        "world={world}: weighted-hit normal changed"
                    );
                }
            }
            println!(
                "geometry world={world}: maximum original-reference normal error={normal_error}"
            );
            let picks = if world {
                actual.clone()
            } else {
                crate::world::WorldDocument::from_scene(&actual)
                    .snapshot(0.0)
                    .unwrap()
            };
            let packed = picks.pack(96, 64);
            let origin = glam::Vec3::from_slice(&packed[P_CAM_ORIGIN..P_CAM_ORIGIN + 3]);
            let mut i = 0;
            for ny in [-1.0, 0.0, 1.0] {
                for nx in [-1.0, 0.0, 1.0] {
                    let direction =
                        (forward + right * nx * packed[P_HALF_W] + up * ny * packed[P_HALF_H])
                            .normalize();
                    let reference_depth = reference_depths[i];
                    let distance = reference_depth / forward.dot(direction);
                    let eps = (packed[P_FOOTPRINT] * distance)
                        .max((origin + direction * distance).length() * 0.0000019073486);
                    match gpu.pick(&picks, 96, 64, [nx, ny]).unwrap() {
                        Some((_, depth)) => {
                            let error = (depth - reference_depth).abs() / forward.dot(direction);
                            println!(
                                "world={world}, ray={i}: depth={depth}, original={reference_depth}, error={error}, footprint={eps}"
                            );
                            if error > eps {
                                failures.push(format!(
                                    "world={world}, ray={i}: error={error}, footprint={eps}"
                                ));
                            }
                        }
                        None => failures.push(format!(
                            "world={world}, ray={i}: missed original weighted hit"
                        )),
                    }
                    i += 1;
                }
            }
        }
        // The weighted-nearest Bulb is not necessarily the object that bounds an
        // accelerated step: this tiny exact cube lies in front of its hit band.
        let mut union = crate::world::WorldDocument::from_scene(&scene)
            .snapshot(0.0)
            .unwrap();
        let packed = union.pack(96, 64);
        let origin = glam::Vec3::from_slice(&packed[P_CAM_ORIGIN..P_CAM_ORIGIN + 3]);
        let scale = 0.0001;
        let center_depth = 0.0008;
        let center = origin + forward * center_depth;
        let mut blocker = Scene::preset(FAMILY_KIFS);
        if let crate::scene::Formula::Kifs(k) = &mut blocker.formula {
            k.kind = crate::scene::KifsKind::Menger;
        }
        blocker.render.iterations = 0.0;
        blocker.material.model = crate::scene::MaterialModel::Fast;
        blocker.object_world = Some(
            glam::Mat4::from_cols(
                (right * scale).extend(0.0),
                (up * scale).extend(0.0),
                (forward * scale).extend(0.0),
                center.extend(1.0),
            )
            .to_cols_array_2d(),
        );
        union.objects.push(blocker);
        let upload = WorldUpload::new(&union, 96, 64).unwrap();
        let estimate_scale = upload.objects[OBJECT_STRIDE + O_DISTANCE_SCALE] / scale;
        let front = center_depth - scale;
        let eps = (packed[P_FOOTPRINT] * front)
            .max((origin + forward * front).length() * 0.0000019073486);
        let expected_depth = front - eps / estimate_scale;
        let (object, depth) = gpu
            .pick(&union, 96, 64, [0.0, 0.0])
            .unwrap()
            .expect("near cube must not be skipped");
        println!(
            "union blocker: object={object}, depth={depth}, analytic band={expected_depth}, footprint={eps}"
        );
        assert_eq!(
            object, 1,
            "step must include the unsupported neighboring object's estimate"
        );
        assert!(
            (depth - expected_depth).abs() <= eps,
            "neighbor weighted band moved beyond its footprint"
        );
        assert!(
            failures.is_empty(),
            "fractional march understeps: {failures:?}"
        );
    }

    /// Explicit visual probe; artifacts remain under target and are not scene templates.
    #[test]
    #[ignore = "renders CUDA glass comparison PNGs; run explicitly with --ignored --nocapture"]
    fn cuda_glass_visual_probe() {
        let _gpu_test = crate::test_gpu::lock();
        let _ = env_logger::try_init();
        let output = std::env::current_dir().unwrap().join("target/glass-probe");
        std::fs::create_dir_all(&output).unwrap();
        let environment = output.join("studio-checker.exr");
        let checker: Vec<[f32; 4]> = (0..256u32)
            .flat_map(|y| (0..512u32).map(move |x| (x, y)))
            .map(|(x, y)| {
                let band = (x / 32 + y / 32) % 2;
                if y < 48 {
                    [2.0, 2.0, 2.0, 1.0]
                } else if band == 0 {
                    [0.16, 0.32, 0.70, 1.0]
                } else {
                    [0.85, 0.60, 0.28, 1.0]
                }
            })
            .collect();
        crate::exr_io::write_rgb(
            &environment,
            512,
            256,
            &checker,
            &crate::color::DISPLAY_PRIMS,
            None,
            true,
        )
        .unwrap();
        let mut gpu = Gpu::new().unwrap();
        for (family, shape) in [(FAMILY_KIFS, "sphere"), (FAMILY_BULB, "bulb")] {
            for preset in ["GlassClear", "GlassBottleGreen", "GlassWaterGreen"] {
                let mut object = Scene::preset(family);
                if family == FAMILY_KIFS {
                    object.render.iterations = 0.0;
                }
                crate::materials::PRESETS
                    .iter()
                    .find(|p| p.path == preset)
                    .unwrap()
                    .apply(&mut object.material);
                object.render.max_bounces = 12;
                object.render.max_steps = 384;
                object.render.denoise.enabled = false;
                object.colour.on = false;
                object.camera.distance = 3.8;
                object.camera.pitch_degrees = 10.0;
                object.environment.enabled = true;
                object.environment.path = environment.to_string_lossy().into_owned();
                object.environment.intensity = 1.0;
                object.lighting.sky_intensity = 0.0;
                object.lighting.sun_intensity = 0.0;
                object.world_render = true;
                let mut child = object.clone();
                child.object_world = Some(glam::Mat4::IDENTITY.to_cols_array_2d());
                object.objects = vec![child];
                let mut target = gpu.target(192, 128);
                for _ in 0..8 {
                    gpu.step(&mut target, &object, 8, 29, None, false);
                }
                assert!(target.colour_error.is_none(), "{:?}", target.colour_error);
                let path = output.join(format!("{shape}-{preset}.png"));
                target
                    .save_png(
                        &path,
                        crate::render_service::PngEncoding::displayed(target.light_kind.hdr()),
                        crate::color::BT2408_SDR_WHITE_NITS,
                        true,
                    )
                    .unwrap();
                eprintln!("glass_visual {} samples={}", path.display(), target.samples);
            }
        }
    }

    #[test]
    fn transmission_packing_promotes_fast_and_invalidates_preparation() {
        let _gpu_test = crate::test_gpu::lock();
        let mut scene = dark_world();
        scene
            .objects
            .push(world_object(FAMILY_BULB, [0.0; 3], [1.0; 3], [1.0; 3]));
        let mut cache = PreparationCache::default();
        let opaque = cache.get(&scene, 17, 17).unwrap();
        scene.objects[0].material.transmission = 1.0;
        let glass = cache.get(&scene, 17, 17).unwrap();
        assert_eq!(glass.world_material, WorldMaterial::Full);
        assert!(!Arc::ptr_eq(&opaque, &glass));
        scene.objects[0].material.transmission_depth = 0.5;
        scene.objects[0].material.transmission_color = [0.12, 0.82, 0.25];
        let absorbed = cache.get(&scene, 17, 17).unwrap();
        assert!(!Arc::ptr_eq(&glass, &absorbed));
        let p = scene.objects[0].pack(17, 17);
        assert_eq!(p[P_TRANSMISSION], 1.0);
        assert_eq!(p[P_TRANSMISSION_DEPTH], 0.5);
        assert_eq!(
            &p[P_TRANSMISSION_COLOR..P_TRANSMISSION_COLOR + 3],
            &crate::color::to_working([0.12, 0.82, 0.25])
        );
        assert_eq!(cache.builds, 3);
    }

    #[test]
    fn cuda_glass_exits_signed_and_unsigned_solids_and_absorbs_green_by_depth() {
        let _gpu_test = crate::test_gpu::lock();
        let mut gpu = Gpu::new().unwrap();
        for family in [FAMILY_KIFS, FAMILY_BULB] {
            let mut object = Scene::preset(family);
            if family == FAMILY_KIFS {
                object.formula = crate::scene::Formula::Kifs(crate::scene::Kifs::preset(
                    crate::scene::KifsKind::Menger,
                ));
                object.render.iterations = 0.0;
            }
            object.material = crate::scene::Material {
                // Prove the dispatch also promotes manually edited Fast materials.
                model: MaterialModel::Fast,
                color_source: crate::scene::ColorSource::Material,
                transmission: 1.0,
                specular_ior: 1.0,
                specular_roughness: 0.0,
                ..Default::default()
            };
            object.camera.pitch_degrees = 0.0;
            object.camera.yaw_degrees = 0.0;
            object.camera.target = [0.0; 3];
            object.camera.distance = 4.0;
            object.colour.on = false;
            object.render.denoise.enabled = false;
            object.render.max_bounces = 8;
            object.render.max_steps = 512;
            object.lighting.sun_intensity = 0.0;
            object.lighting.sky_intensity = 1.0;
            object.lighting.sky_horizon = [1.0; 3];
            object.lighting.sky_zenith = [1.0; 3];
            // A uniform white sky behind the glass: no environment map.
            object.environment.enabled = false;
            for world in [false, true] {
                let mut scene = object.clone();
                if world {
                    scene.world_render = true;
                    let mut child = object.clone();
                    child.object_world = Some(glam::Mat4::IDENTITY.to_cols_array_2d());
                    scene.objects = vec![child];
                }
                let mut target = gpu.target(17, 17);
                gpu.step(&mut target, &scene, 16, 19, None, false);
                let raw = gpu.raw_scene_linear(&target);
                let center = 8 * 17 + 8;
                assert!(
                    raw[center][..3].iter().all(|v| (*v - 1.0).abs() < 0.03),
                    "clear family={family} world={world}: {:?}",
                    raw[center]
                );
                assert!(
                    gpu.normal_sums(&target)[center][..3]
                        .iter()
                        .any(|v| v.abs() > 0.1),
                    "fixture must actually intersect the solid"
                );
                if family == FAMILY_KIFS {
                    let material = if world {
                        &mut scene.objects[0].material
                    } else {
                        &mut scene.material
                    };
                    material.transmission_color = [0.12, 0.82, 0.25];
                    material.transmission_depth = 2.0;
                    gpu.step(&mut target, &scene, 256, 19, None, false);
                    let green = gpu.raw_scene_linear(&target)[center];
                    // Hue in the authored (Rec.709) primaries, magnitude in the working space.
                    let hue = crate::color::to_709([green[0], green[1], green[2]]);
                    assert!(
                        hue[1] > hue[0] * 5.0 && hue[1] > hue[2] * 2.0,
                        "{green:?} = Rec.709 {hue:?}"
                    );
                    for (value, expected) in green[..3]
                        .iter()
                        .zip(crate::color::to_working([0.12, 0.82, 0.25]))
                    {
                        assert!(
                            (*value - expected).abs() < expected * 0.15 + 0.01,
                            "depth family={family} world={world}: {green:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn prepared_world_material_tracks_packed_objects_and_model_edits() {
        let _gpu_test = crate::test_gpu::lock();
        let mut cache = PreparationCache::default();
        let mut scene = dark_world();
        scene.material.model = MaterialModel::StandardSurface;
        scene.objects = vec![
            world_object(
                FAMILY_BULB,
                [-0.6, 0.0, 0.0],
                [0.6, 0.8, 0.5],
                [0.8, 0.2, 0.1],
            ),
            world_object(
                FAMILY_QUAT,
                [0.6, 0.0, 0.0],
                [0.5, 0.65, 0.4],
                [0.1, 0.2, 0.8],
            ),
        ];
        for object in &mut scene.objects {
            object.material.model = MaterialModel::Fast;
        }
        let fast = cache.get(&scene, 32, 24).unwrap();
        assert_eq!(fast.world_material, WorldMaterial::Fast);
        assert!(Arc::ptr_eq(&fast, &cache.get(&scene, 32, 24).unwrap()));
        scene.objects[1].material.model = MaterialModel::StandardSurface;
        let mixed = cache.get(&scene, 32, 24).unwrap();
        assert_eq!(mixed.world_material, WorldMaterial::Mixed);
        assert!(!Arc::ptr_eq(&fast, &mixed));
        scene.objects[0].material.model = MaterialModel::StandardSurface;
        assert_eq!(
            cache.get(&scene, 32, 24).unwrap().world_material,
            WorldMaterial::Full
        );
        scene.objects.clear();
        assert_eq!(
            cache.get(&scene, 32, 24).unwrap().world_material,
            WorldMaterial::Mixed
        );
        assert_eq!(cache.builds, 4);
        scene.objects.push(world_object(
            FAMILY_BULB,
            [0.0; 3],
            [1.0; 3],
            [0.8, 0.2, 0.1],
        ));
        scene.objects[0].material.model = MaterialModel::StandardSurface;
        assert!(!cache.get(&scene, 32, 24).unwrap().single_fast_bulb);
        scene.objects[0].material.model = MaterialModel::Fast;
        let single = cache.get(&scene, 32, 24).unwrap();
        assert!(single.single_fast_bulb);
        assert!(Arc::ptr_eq(&single, &cache.get(&scene, 32, 24).unwrap()));
        // Masks are part of the upload: a new mask must rebuild, never reuse the unmasked world.
        scene.objects[0].masks.push(crate::scene::MaskInstance {
            mask: crate::scene::Mask::default(),
            world: glam::Mat4::IDENTITY.to_cols_array_2d(),
        });
        assert!(!Arc::ptr_eq(&single, &cache.get(&scene, 32, 24).unwrap()));
        scene.objects[0].masks.clear();
        scene.objects[0].formula = crate::scene::Formula::preset(FAMILY_QUAT);
        assert!(!cache.get(&scene, 32, 24).unwrap().single_fast_bulb);
        scene.objects.push(world_object(
            FAMILY_BULB,
            [1.0, 0.0, 0.0],
            [1.0; 3],
            [0.1, 0.2, 0.8],
        ));
        assert!(!cache.get(&scene, 32, 24).unwrap().single_fast_bulb);
    }

    /// A compiler-specialized material path must preserve the complete world estimator,
    /// transformed normals, direct/secondary lighting and progressive guide accumulation.
    /// Inlining the single-bulb estimator can reorder f32 transcendental arithmetic;
    /// that route has a tight numerical contract, while the other routes remain exact.
    #[test]
    fn cuda_specialized_world_materials_preserve_radiance_and_affine_guides() {
        let _gpu_test = crate::test_gpu::lock();
        fn compare_pixels(
            actual: &[[f32; 4]],
            expected: &[[f32; 4]],
            approximate: bool,
            label: &str,
        ) -> Option<String> {
            assert_eq!(actual.len(), expected.len());
            let mut max_absolute = 0.0f32;
            let mut max_relative = 0.0f32;
            let mut worst = (0, 0);
            let mut passed = true;
            for (index, (a, b)) in actual.iter().zip(expected).enumerate() {
                for channel in 0..4 {
                    let difference = (a[channel] - b[channel]).abs();
                    if difference > max_absolute {
                        max_absolute = difference;
                        worst = (index, channel);
                    }
                    max_relative = max_relative.max(difference / b[channel].abs().max(1e-12));
                    passed &= if approximate {
                        a[channel].is_finite()
                            && b[channel].is_finite()
                            && difference <= 3e-5 + 3e-5 * b[channel].abs()
                    } else {
                        a[channel] == b[channel]
                    };
                }
            }
            let diagnostic = format!(
                "{label}: max_absolute={max_absolute:e}, max_relative={max_relative:e}, approximate={approximate}, pixel={}, channel={}, actual={:?}, expected={:?}",
                worst.0, worst.1, actual[worst.0], expected[worst.0]
            );
            println!("{diagnostic}, passed={passed}");
            (!passed).then_some(diagnostic)
        }
        fn compare_guides(
            actual: &[[f32; 4]],
            expected: &[[f32; 4]],
            approximate: bool,
            label: &str,
        ) -> Option<String> {
            assert_eq!(actual.len(), expected.len());
            // Alpha stores the accumulated sample count. Zero RGB identifies a miss
            // for the nonblack materials/normals used by this fixture.
            for (index, (a, b)) in actual.iter().zip(expected).enumerate() {
                assert!(a[3] == b[3], "{label}: count differs at pixel {index}");
                let hit = |p: &[f32; 4]| p[..3].iter().any(|channel| *channel != 0.0);
                assert!(
                    hit(a) == hit(b),
                    "{label}: hit/miss differs at pixel {index}"
                );
            }
            compare_pixels(actual, expected, approximate, label)
        }
        let mut gpu = Gpu::new().unwrap();
        let mut errors = Vec::new();
        for model in [MaterialModel::Fast, MaterialModel::StandardSurface] {
            for object_count in [1usize, 2] {
                let mut scene = dark_world();
                // Deliberately contradict the root material: the world objects own the BSDFs.
                scene.material.model = if model == MaterialModel::Fast {
                    MaterialModel::StandardSurface
                } else {
                    MaterialModel::Fast
                };
                scene.lighting.sky_intensity = 2.0;
                scene.lighting.background = true;
                scene.lights.push(scene.lighting);
                scene.render.max_bounces = 2;
                scene.objects = vec![
                    world_object(
                        FAMILY_BULB,
                        [-0.6, 0.0, 0.0],
                        [0.55, 0.8, 0.5],
                        [0.8, 0.2, 0.1],
                    ),
                    world_object(
                        FAMILY_QUAT,
                        [0.65, 0.0, 0.0],
                        [0.5, 0.65, 0.4],
                        [0.1, 0.2, 0.8],
                    ),
                ];
                scene.objects.truncate(object_count);
                for (index, object) in scene.objects.iter_mut().enumerate() {
                    object.material.model = model;
                    object.material.emission = 0.0;
                    object.material.specular = 1.0;
                    object.material.specular_roughness = 0.35;
                    object.material.metalness = if index == 0 { 0.8 } else { 0.0 };
                    let mut matrix = glam::Mat4::from_cols_array_2d(&object.object_world.unwrap());
                    matrix.y_axis += matrix.x_axis * 0.35;
                    object.object_world = Some(matrix.to_cols_array_2d());
                }
                let mut specialized = gpu.target(48, 32);
                let mut reference = gpu.target(48, 32);
                gpu.force_mixed_world = false;
                for count in [1, 3, 4] {
                    gpu.step(&mut specialized, &scene, count, 17, None, false);
                }
                gpu.force_mixed_world = true;
                for count in [1, 3, 4] {
                    gpu.step(&mut reference, &scene, count, 17, None, false);
                }
                assert!(
                    specialized.colour_error.is_none(),
                    "{:?}",
                    specialized.colour_error
                );
                assert!(
                    reference.colour_error.is_none(),
                    "{:?}",
                    reference.colour_error
                );
                assert_eq!(specialized.samples, 8);
                assert_eq!(specialized.samples, reference.samples);
                let specialized_radiance = gpu.raw_scene_linear(&specialized);
                let reference_radiance = gpu.raw_scene_linear(&reference);
                assert!(specialized_radiance.iter().any(|p| p[0] > 0.0));
                let approximate = model == MaterialModel::Fast && object_count == 1;
                errors.extend(compare_pixels(
                    &specialized_radiance,
                    &reference_radiance,
                    approximate,
                    &format!("radiance for {model:?}, objects={object_count}"),
                ));
                let specialized_albedo = gpu.albedo_sums(&specialized);
                assert!(specialized_albedo.iter().any(|p| p[0] > 0.0 && p[0] > p[2]));
                if object_count == 2 {
                    assert!(specialized_albedo.iter().any(|p| p[2] > 0.0 && p[2] > p[0]));
                }
                errors.extend(compare_guides(
                    &specialized_albedo,
                    &gpu.albedo_sums(&reference),
                    approximate,
                    &format!("albedo for {model:?}, objects={object_count}"),
                ));
                errors.extend(compare_guides(
                    &gpu.normal_sums(&specialized),
                    &gpu.normal_sums(&reference),
                    approximate,
                    &format!("normals for {model:?}, objects={object_count}"),
                ));
            }
        }
        gpu.force_mixed_world = false;
        assert!(errors.is_empty(), "specialized world errors: {errors:#?}");
    }

    #[test]
    fn preparation_reuses_display_edits_and_invalidates_runtime_geometry() {
        let _gpu_test = crate::test_gpu::lock();
        let mut cache = PreparationCache::default();
        let mut scene = dark_world();
        scene.objects.push(world_object(
            FAMILY_BULB,
            [0.0; 3],
            [1.0; 3],
            [0.8, 0.2, 0.1],
        ));
        scene.lights.push(scene.lighting);
        let first = cache.get(&scene, 320, 180).unwrap();
        scene.name = "renamed".into();
        scene.render.exposure_stops += 1.25;
        scene.render.saturation += 0.2;
        scene.render.reinhard = !scene.render.reinhard;
        scene.render.denoise.enabled = !scene.render.denoise.enabled;
        scene.render.denoise.interval = 7;
        scene.render.denoise.mode = crate::denoise::Mode::Color;
        scene.colour.on = !scene.colour.on;
        scene.material.preset = Some("authoring label".into());
        let display = cache.get(&scene, 320, 180).unwrap();
        assert!(Arc::ptr_eq(&first, &display));
        assert_eq!(
            display.params(&scene, None).as_slice(),
            WorldUpload::new(&scene, 320, 180).unwrap().params
        );
        let original = scene.clone();
        for change in 0..9 {
            let mut scene = original.clone();
            match change {
                0 => scene.objects[0].object_world.as_mut().unwrap()[3][0] += 0.25,
                1 => scene.objects[0].material.base_color[0] += 0.1,
                2 => scene.objects[0].formula = crate::scene::Formula::preset(FAMILY_BOX),
                3 => scene.lights[0].sun_azimuth += 1.0,
                4 => scene.camera.yaw_degrees += 2.0,
                5 => scene.camera_reference = Some(7.0),
                6 => scene.objects[0].render.iterations += 1.0,
                7 => scene.environment.revision += 1,
                _ => scene.objects.clear(),
            }
            let prepared = cache.get(&scene, 320, 180).unwrap();
            assert!(!Arc::ptr_eq(&first, &prepared), "change={change}");
            let fresh = WorldUpload::new(&scene, 320, 180).unwrap();
            assert_eq!(prepared.params(&scene, None).as_slice(), fresh.params);
            assert_eq!(prepared.upload.objects, fresh.objects);
            assert_eq!(prepared.upload.lights, fresh.lights);
        }
        assert_eq!(cache.builds, 10);
        assert_eq!(cache.hits, 1);
        assert_eq!(cache.entries.iter().flatten().count(), 4);
    }

    #[test]
    fn preparation_preview_full_and_all_formula_packs_remain_exact() {
        let _gpu_test = crate::test_gpu::lock();
        let mut cache = PreparationCache::default();
        for family in 0..=FAMILY_HYBRID {
            let mut scene = Scene::preset(family);
            scene.render.max_bounces = 4;
            for world in [false, true] {
                scene.world_render = world;
                scene.objects = if world {
                    vec![world_object(
                        family,
                        [0.2, -0.1, 0.3],
                        [0.8, 1.2, 0.6],
                        [0.3; 3],
                    )]
                } else {
                    Vec::new()
                };
                let full = cache.get(&scene, 320, 180).unwrap();
                let preview = cache.get(&scene, 160, 90).unwrap();
                let mut expected = WorldUpload::new(&scene, 160, 90).unwrap().params;
                expected[P_MAX_BOUNCES] = expected[P_MAX_BOUNCES].min(2.0);
                assert_eq!(preview.params(&scene, Some(2)).as_slice(), expected);
                assert!(Arc::ptr_eq(&full, &cache.get(&scene, 320, 180).unwrap()));
                assert_eq!(
                    full.params(&scene, None).as_slice(),
                    WorldUpload::new(&scene, 320, 180).unwrap().params
                );
            }
        }
    }

    #[test]
    fn preparation_failures_are_cached_until_actual_inputs_change() {
        let _gpu_test = crate::test_gpu::lock();
        let mut scene = dark_world();
        scene
            .objects
            .push(world_object(FAMILY_BULB, [0.0; 3], [0.0; 3], [0.4; 3]));
        let mut cache = PreparationCache::default();
        let first_error = cache.get(&scene, 320, 180).err().unwrap();
        assert_eq!(cache.get(&scene, 320, 180).err().unwrap(), first_error);
        assert_eq!((cache.builds, cache.hits), (1, 1));
        scene.objects[0].object_world = Some(glam::Mat4::IDENTITY.to_cols_array_2d());
        assert!(cache.get(&scene, 320, 180).is_ok());
        assert_eq!((cache.builds, cache.hits), (2, 1));
    }

    #[test]
    fn resident_upload_identity_restores_a_after_b_and_retries_failure() {
        let _gpu_test = crate::test_gpu::lock();
        let mut resident = Resident::<u32>::default();
        let mut copies = Vec::new();
        for key in [1, 1, 2, 2, 1] {
            resident
                .update(key, || {
                    copies.push(key);
                    Ok(())
                })
                .unwrap();
        }
        assert_eq!(copies, [1, 2, 1]);
        assert!(
            resident
                .update(2, || Err("failed partial upload".into()))
                .is_err()
        );
        assert_eq!(resident.key, None);
        assert!(
            resident
                .update(1, || {
                    copies.push(1);
                    Ok(())
                })
                .unwrap()
        );
        assert!(
            resident
                .update(2, || {
                    copies.push(2);
                    Ok(())
                })
                .unwrap()
        );
        assert_eq!(copies, [1, 2, 1, 1, 2]);
    }

    #[test]
    fn texel_identity_ignores_environment_gain_rotation_but_tracks_reload_and_palettes() {
        let _gpu_test = crate::test_gpu::lock();
        let mut scene = dark_world();
        scene.environment.enabled = true;
        scene.environment.path = "fixture.exr".into();
        scene.objects.push(Scene::preset(FAMILY_BULB));
        let first = Arc::new(PreparedScene::new(&scene, 320, 180).unwrap());
        scene.environment.intensity += 0.5;
        scene.environment.rotation_degrees += 19.0;
        let adjusted = Arc::new(PreparedScene::new(&scene, 320, 180).unwrap());
        assert!(LutResidentKey(first.clone()) == LutResidentKey(adjusted));
        scene.environment.revision += 1;
        assert!(
            LutResidentKey(first.clone())
                != LutResidentKey(Arc::new(PreparedScene::new(&scene, 320, 180).unwrap()))
        );
        scene.environment.revision -= 1;
        scene.objects.clear();
        assert!(
            LutResidentKey(first)
                != LutResidentKey(Arc::new(PreparedScene::new(&scene, 320, 180).unwrap()))
        );
    }

    #[test]
    fn padded_readback_reuses_storage_and_preserves_normalization_and_guide_counts() {
        let _gpu_test = crate::test_gpu::lock();
        let (width, height) = (9usize, 5usize);
        let mut tiled = vec![[f32::NAN; 4]; width.div_ceil(8) * height.div_ceil(4) * 32];
        let expected: Vec<_> = (0..width * height)
            .map(|i| [i as f32, -2.0, 4.0, if i == 0 { 0.0 } else { 2.0 }])
            .collect();
        for y in 0..height {
            for x in 0..width {
                let tile = (y / 4) * width.div_ceil(8) + x / 8;
                tiled[tile * 32 + (y % 4) * 8 + x % 8] = expected[y * width + x];
            }
        }
        let mut output = Vec::new();
        untile_into(&tiled, width, height, false, &mut output);
        assert_eq!(output, expected);
        let pointer = output.as_ptr();
        let capacity = output.capacity();
        untile_into(&tiled, width, height, true, &mut output);
        assert_eq!(output[0], [0.0, -0.0, 0.0, 1.0]);
        for (i, pixel) in output.iter().enumerate().skip(1) {
            assert_eq!(*pixel, [i as f32 * 0.5, -1.0, 2.0, 1.0]);
        }
        assert_eq!(output.as_ptr(), pointer);
        assert_eq!(output.capacity(), capacity);
        untile_into(&tiled, width, height, false, &mut output);
        assert_eq!(output, expected);
        assert_eq!(output.as_ptr(), pointer);
    }

    #[test]
    fn typed_display_key_tracks_colour_and_revision_without_denoise() {
        let _gpu_test = crate::test_gpu::lock();
        let mut scene = Scene::preset(FAMILY_BULB);
        let key = DisplayKey::new(&scene, 0);
        scene.render.denoise.interval += 1;
        assert!(key.matches(&scene, 0));
        assert!(!key.matches(&scene, 1));
        scene.render.exposure_stops += 0.25;
        assert!(!key.matches(&scene, 0));
        scene.render.exposure_stops -= 0.25;
        scene.colour.on = !scene.colour.on;
        assert!(!key.matches(&scene, 0));
    }

    #[test]
    #[ignore = "CPU preparation timing; run explicitly with --ignored --nocapture"]
    fn preparation_benchmark() {
        let _gpu_test = crate::test_gpu::lock();
        const ITERATIONS: u64 = 10_000;
        let mut scene = dark_world();
        scene.objects = vec![
            world_object(FAMILY_BULB, [-0.8, 0.0, 0.0], [0.7; 3], [0.8, 0.2, 0.1]),
            world_object(FAMILY_BOX, [0.8, 0.0, 0.0], [0.3; 3], [0.1, 0.3, 0.8]),
        ];
        scene.lights.push(scene.lighting);
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            std::hint::black_box(PreparedScene::new(&scene, 320, 180).unwrap());
        }
        let uncached = start.elapsed();
        let mut cache = PreparationCache::default();
        cache.get(&scene, 320, 180).unwrap();
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            std::hint::black_box(cache.get(&scene, 320, 180).unwrap());
        }
        let cached = start.elapsed();
        assert_eq!((cache.builds, cache.hits), (1, ITERATIONS));
        eprintln!(
            "preparation_cpu iterations={ITERATIONS} uncached_us={} cached_us={} builds={} hits={}",
            uncached.as_micros(),
            cached.as_micros(),
            cache.builds,
            cache.hits
        );
    }

    #[test]
    fn cuda_pick_finds_the_fractal_under_the_cursor_and_misses_the_sky() {
        let _gpu_test = crate::test_gpu::lock();
        let mut gpu = Gpu::new().unwrap();
        let scene = crate::world::WorldDocument::from_scene(&Scene::preset(FAMILY_BULB))
            .snapshot(0.0)
            .unwrap();
        assert!(scene.world_render && scene.objects[0].node.is_some());
        let (w, h) = (64usize, 48usize);
        let centre = gpu.pick(&scene, w, h, [0.0, 0.0]).unwrap();
        let (index, depth) = centre.expect("the framed fractal fills the centre");
        assert_eq!(index, 0);
        let (eye, _, _) = scene.camera_eye();
        assert!(depth > 0.0 && depth < eye.length(), "depth {depth}");
        assert_eq!(gpu.pick(&scene, w, h, [0.999, 0.999]).unwrap(), None);
        // Between viewport launches: a launch after a pick sets its own parameters back, and a
        // pick after a launch answers the same.
        let mut target = gpu.target(w, h);
        gpu.step(&mut target, &scene, 1, 19, None, false);
        assert_eq!(gpu.pick(&scene, w, h, [0.0, 0.0]).unwrap(), centre);
        // The index is the hit object's: a box off to the side first, the bulb in the middle.
        let mut world = dark_world();
        world.objects.push(world_object(
            FAMILY_BOX,
            [3.0, 0.0, 0.0],
            [0.2; 3],
            [1.0; 3],
        ));
        world
            .objects
            .push(world_object(FAMILY_BULB, [0.0; 3], [1.0; 3], [1.0; 3]));
        world.lights.push(world.lighting);
        let hit = gpu.pick(&world, w, h, [0.0, 0.0]).unwrap();
        assert_eq!(hit.map(|(index, _)| index), Some(1));
    }

    #[test]
    fn cuda_resident_world_restore_matches_uninterrupted_reference() {
        let _gpu_test = crate::test_gpu::lock();
        let mut gpu = Gpu::new().unwrap();
        let mut a = dark_world();
        a.objects.push(world_object(
            FAMILY_BULB,
            [0.0; 3],
            [1.0; 3],
            [0.8, 0.1, 0.2],
        ));
        a.lights.push(a.lighting);
        let mut b = a.clone();
        b.objects[0].object_world.as_mut().unwrap()[3][0] += 0.25;
        b.objects[0].material.base_color = [0.1, 0.8, 0.2];
        b.objects.push(world_object(
            FAMILY_BOX,
            [0.6, 0.0, 0.0],
            [0.2; 3],
            [0.1, 0.2, 0.9],
        ));
        b.lights[0].sun_azimuth += 20.0;
        let mut target_a = gpu.target(33, 25);
        let mut target_b = gpu.target(33, 25);
        gpu.step(&mut target_a, &a, 1, 19, None, false);
        let before = gpu.preparation_stats();
        gpu.step(&mut target_a, &a, 1, 19, None, false);
        let idle = gpu.preparation_stats();
        assert_eq!(
            (idle.lut_uploads, idle.object_uploads, idle.light_uploads),
            (
                before.lut_uploads,
                before.object_uploads,
                before.light_uploads
            )
        );
        gpu.step(&mut target_b, &b, 1, 19, None, false);
        gpu.step(&mut target_a, &a, 1, 19, None, false);
        let restored = gpu.raw_scene_linear(&target_a);
        let stats = gpu.preparation_stats();
        assert_eq!(
            (stats.lut_uploads, stats.object_uploads, stats.light_uploads),
            (3, 3, 3)
        );
        assert_eq!(stats.builds, 2);
        let mut reference = gpu.target(33, 25);
        for _ in 0..3 {
            gpu.step(&mut reference, &a, 1, 19, None, false);
        }
        assert_eq!(restored, gpu.raw_scene_linear(&reference));
        assert_eq!(target_a.samples, 3);
        assert!(target_a.colour_error.is_none());
    }

    #[test]
    fn cuda_failed_environment_reload_keeps_resident_identity_and_caches_decode_error() {
        let _gpu_test = crate::test_gpu::lock();
        let mut gpu = Gpu::new().unwrap();
        let mut scene = dark_world();
        scene
            .objects
            .push(world_object(FAMILY_BULB, [0.0; 3], [1.0; 3], [0.4; 3]));
        let mut target = gpu.target(9, 5);
        gpu.step(&mut target, &scene, 1, 1, None, false);
        let resident = gpu
            .cuda
            .as_ref()
            .unwrap()
            .lut_resident
            .key
            .as_ref()
            .unwrap()
            .0
            .clone();
        let before = gpu.preparation_stats();
        scene.environment.enabled = true;
        scene.environment.path = std::env::temp_dir()
            .join("frac-missing-preparation-fixture.exr")
            .to_string_lossy()
            .into_owned();
        assert!(!std::path::Path::new(&scene.environment.path).exists());
        for _ in 0..2 {
            gpu.step(&mut target, &scene, 1, 1, None, false);
        }
        assert!(target.colour_error.is_some());
        assert!(Arc::ptr_eq(
            &resident,
            &gpu.cuda
                .as_ref()
                .unwrap()
                .lut_resident
                .key
                .as_ref()
                .unwrap()
                .0
        ));
        let failed = gpu.preparation_stats();
        assert_eq!(failed.environment_decodes, before.environment_decodes + 1);
        assert_eq!(failed.lut_uploads, before.lut_uploads);
        scene.environment.revision += 1;
        gpu.step(&mut target, &scene, 1, 1, None, false);
        assert_eq!(
            gpu.preparation_stats().environment_decodes,
            failed.environment_decodes + 1
        );
        scene.environment.enabled = false;
        gpu.step(&mut target, &scene, 1, 1, None, false);
        assert!(target.colour_error.is_none());
        assert_eq!(gpu.preparation_stats().lut_uploads, before.lut_uploads);
    }

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
    #[test]
    fn cuda_adaptive_sampling_stops_converged_tiles_and_keeps_sky_exact() {
        let _gpu_test = crate::test_gpu::lock();
        let mut gpu = Gpu::new().unwrap();
        let mut scene = Scene::preset(FAMILY_BULB);
        scene.camera.distance *= 4.0; // mostly sky: those tiles converge at the minimum
        scene.render.denoise.enabled = false;
        scene.render.adaptive = crate::scene::Adaptive {
            enabled: true,
            noise_threshold: 0.05,
            min_samples: 16,
        };
        let budget = 4096;
        let mut target = gpu.target(64, 48);
        while !target.complete(budget) {
            gpu.step(&mut target, &scene, 16, 0, None, false);
        }
        assert!(
            target.converged && target.samples < budget,
            "{} samples",
            target.samples
        );
        assert_eq!(target.active_tiles.0, 0);
        let counts: Vec<f32> = gpu.accum_sums(&target).iter().map(|p| p[3]).collect();
        let (low, high) = counts
            .iter()
            .fold((f32::MAX, 0.0f32), |(l, h), &c| (l.min(c), h.max(c)));
        assert_eq!(low, 16.0, "sky tiles stop at the minimum");
        assert!(high > low, "noisy fractal tiles keep sampling");
        // Sky radiance is deterministic: per-pixel normalisation must reproduce the uniform render.
        let adaptive = gpu.raw_scene_linear(&target);
        scene.render.adaptive.enabled = false;
        let mut uniform = gpu.target(64, 48);
        gpu.step(&mut uniform, &scene, 16, 0, None, false);
        let reference = gpu.raw_scene_linear(&uniform);
        let sky = counts.iter().position(|&c| c == 16.0).unwrap();
        for k in 0..3 {
            assert!(
                (adaptive[sky][k] - reference[sky][k]).abs() < 1e-5,
                "{:?} vs {:?}",
                adaptive[sky],
                reference[sky]
            );
        }
    }

    /// Working-space pixels in Rec.709, the primaries test colours are authored in.
    fn rec709(pixels: &[[f32; 4]]) -> Vec<[f32; 3]> {
        pixels
            .iter()
            .map(|p| crate::color::to_709([p[0], p[1], p[2]]))
            .collect()
    }
    fn dark_world() -> Scene {
        let mut scene = Scene::preset(FAMILY_BULB);
        scene.world_render = true;
        scene.camera.distance = 4.0;
        scene.camera.fov_y_degrees = 55.0;
        scene.lighting.sky_intensity = 0.0;
        scene.lighting.background = false;
        // Dark: no environment either (a new scene has the built-in map).
        scene.environment.enabled = false;
        scene.colour.on = false;
        scene.render.denoise.enabled = false;
        scene.render.max_bounces = 0;
        scene.render.max_steps = 256;
        scene
    }

    /// BUG1 frame 27 (copper turbine: Hybrid Mandelbulb + octahedral KIFS), self-contained:
    /// grazing rays that need hundreds of march steps, primary rays only.
    fn bug1_frame27() -> Scene {
        let mut scene = Scene::preset(crate::params::FAMILY_HYBRID);
        if let crate::scene::Formula::Hybrid(h) = &mut scene.formula {
            h.bailout = 2.0;
            h.bulb.power = 1.829205;
            h.bulb.angle_scale = [2.43, 0.49];
            h.bulb.angle_phase_degrees = [-7.74, 0.0];
            h.kifs.kind = crate::scene::KifsKind::Octahedron;
            h.steps = [
                crate::scene::HybridStep::Mandelbulb,
                crate::scene::HybridStep::KifsFold,
                crate::scene::HybridStep::Off,
                crate::scene::HybridStep::Off,
            ];
        }
        scene.camera.target = [-0.3617138, 0.04296637, -0.083121695];
        scene.camera.yaw_degrees = 63.69792;
        scene.camera.pitch_degrees = -0.55681777;
        scene.camera.roll_degrees = 0.00025211525;
        scene.camera.distance = 2.017553;
        scene.camera.fov_y_degrees = 38.0;
        scene.camera.f_number = 0.0;
        scene.camera_reference = Some(2.0);
        scene.render.iterations = 12.0;
        scene.render.step_factor = 0.85;
        scene.render.hit_epsilon = 0.001;
        scene.render.max_bounces = 0;
        scene.render.adaptive.enabled = false;
        scene.render.denoise.enabled = false;
        scene.environment.enabled = false;
        scene.colour.on = false;
        scene.world_render = true;
        scene.objects = vec![scene.clone()];
        scene
    }

    /// Primary albedo and normal guides of one sample per pixel at `max_steps`, with the share
    /// of samples that ran out of steps.
    fn frame27_guides(gpu: &mut Gpu, max_steps: u32) -> (Vec<[f32; 4]>, Vec<[f32; 4]>, f32) {
        let mut scene = bug1_frame27();
        scene.render.max_steps = max_steps;
        scene.objects[0].render.max_steps = max_steps;
        let mut target = gpu.target(384, 384);
        gpu.step(&mut target, &scene, 1, 19, None, false);
        (
            gpu.albedo_sums(&target),
            gpu.normal_sums(&target),
            target.unresolved,
        )
    }

    /// A march that runs out of steps is unresolved (a camera ray: the background), never the
    /// closest sample as a hit (BUG1): at the default
    /// budget no ray of frame 27 runs out and the hits equal a 4x budget's bit for bit; at 256
    /// steps the shortfall is reported and every remaining hit is a converged one.
    #[test]
    fn cuda_march_out_of_steps_is_a_reported_miss_not_a_phantom_hit() {
        let _gpu_test = crate::test_gpu::lock();
        let mut gpu = Gpu::new().unwrap();
        let budget = crate::scene::DEFAULT_MAX_STEPS;
        let (albedo, normal, unresolved) = frame27_guides(&mut gpu, budget);
        let (ref_albedo, ref_normal, _) = frame27_guides(&mut gpu, 4 * budget);
        assert_eq!(unresolved, 0.0, "the default budget converges every ray");
        assert!(albedo == ref_albedo && normal == ref_normal);
        let hits = normal.iter().filter(|n| n[..3] != [0.0; 3]).count();
        assert!(hits > 60_000, "{hits} primary hits");

        let (short_albedo, short_normal, short) = frame27_guides(&mut gpu, 256);
        assert!(
            short > 0.1,
            "frame 27 at 256 steps leaves many rays out of steps: {short}"
        );
        let mut kept = 0;
        for k in 0..normal.len() {
            if short_normal[k][..3] != [0.0; 3] {
                assert_eq!(
                    (short_albedo[k], short_normal[k]),
                    (ref_albedo[k], ref_normal[k]),
                    "pixel {k}: a hit at 256 steps must be the converged hit"
                );
                kept += 1;
            }
        }
        assert!(kept > 0 && kept < hits);
    }

    #[test]
    fn cuda_curved_hybrid_normals_match_converged_field_gradient() {
        let _gpu_test = crate::test_gpu::lock();
        let scene = bug1_frame27();
        // Independent f64 field gradients, converged by halving the central difference step
        // from eps/2 to eps/64, at converged hits (4096 march steps) whose neighborhoods keep
        // the same escape iteration count. The old tetrahedral stencil errs 15..40 degrees here.
        // (The former points were closest samples of rays out of steps, not surface points.)
        // The pixel-cone camera start (fractal-kernels `cone_start`) moves hits within their
        // footprint band: over this frame every one of 69,529 hits stays within 2 eps of the
        // march from t_enter (measured 2026-10-11, no skipped feature). Two former probes,
        // (140,189) and (180,268), sit where the surface turns by 3.2 and 11.5 degrees inside
        // that band, so they no longer probe the stencil, only the hit point; they are dropped.
        let references = [
            (316, 125, [-0.08490141, 0.18438966, -0.979_179_3]),
            (215, 198, [0.212_273_2, 0.17521111, 0.96137462]),
            (155, 208, [0.67414312, -0.606_544_8, -0.42146708]),
            (259, 232, [0.81196077, 0.44119635, 0.38218516]),
            (280, 246, [0.850_794_4, -0.521_097_2, -0.067_872_08]),
            (366, 263, [0.028329306, -0.4778096, 0.878_006_5]),
        ];
        let mut gpu = Gpu::new().unwrap();
        let mut target = gpu.target(384, 384);
        gpu.step(&mut target, &scene, 1, 19, None, false);
        let normals = gpu.normal_sums(&target);
        for (x, y, expected) in references {
            let actual = glam::Vec3::from_slice(&normals[y * 384 + x]);
            let expected = glam::Vec3::from_array(expected);
            assert!(actual.is_finite() && (actual.length() - 1.0).abs() < 1e-4);
            // Guides orient the normal toward the incoming ray.
            let cosine = actual.dot(expected).abs().clamp(0.0, 1.0);
            let degrees = cosine.acos().to_degrees();
            println!("({x},{y}): {degrees} degrees");
            assert!(degrees < 1.0, "({x},{y}): {actual:?} vs {expected:?}");
        }
    }

    #[test]
    fn cuda_tiny_hybrid_gradients_keep_surface_normal_direction() {
        let _gpu_test = crate::test_gpu::lock();
        let mut gpu = Gpu::new().unwrap();
        let mut authored = Scene::preset(crate::params::FAMILY_HYBRID);
        if let crate::scene::Formula::Hybrid(h) = &mut authored.formula {
            h.bailout = 2.0;
            h.bulb.power = 1.0;
            h.bulb.angle_scale = [2.43, 0.49];
            h.bulb.angle_phase_degrees = [-7.74, 0.0];
            h.kifs.kind = crate::scene::KifsKind::Octahedron;
            h.steps = [
                crate::scene::HybridStep::Mandelbulb,
                crate::scene::HybridStep::KifsFold,
                crate::scene::HybridStep::Off,
                crate::scene::HybridStep::Off,
            ];
        }
        authored.camera.target = [-0.3617138, 0.04296637, -0.083121695];
        authored.camera.yaw_degrees = 63.69792;
        authored.camera.pitch_degrees = -0.55681777;
        authored.camera.distance = 2.017553;
        authored.camera.fov_y_degrees = 38.0;
        authored.camera.f_number = 0.0;
        authored.render.iterations = 64.0;
        authored.render.hit_epsilon = 0.001;
        authored.render.denoise.enabled = false;
        authored.render.adaptive.enabled = false;
        authored.render.max_bounces = 0;
        authored.colour.on = false;
        authored.environment.enabled = false;
        for world in [false, true] {
            let scene = if world {
                crate::world::WorldDocument::from_scene(&authored)
                    .snapshot(0.0)
                    .unwrap()
            } else {
                authored.clone()
            };
            let (w, h) = (128usize, 128usize);
            let mut target = gpu.target(w, h);
            gpu.step(&mut target, &scene, 1, 19, None, false);
            let normals = gpu.normal_sums(&target);
            let p = scene.pack(w as u32, h as u32);
            let v = |i| glam::Vec3::from_slice(&p[i..i + 3]);
            let (forward, right, up) = (v(P_CAM_FORWARD), v(P_CAM_RIGHT), v(P_CAM_UP));
            let mut hits = 0;
            let mut fallback = 0;
            for (i, n) in normals.iter().enumerate() {
                let n = glam::Vec3::from_slice(n);
                assert!(n.is_finite());
                if n.length_squared() < 0.1 {
                    continue;
                }
                assert!((n.length() - 1.0).abs() < 1e-4);
                hits += 1;
                let (x, y) = ((i % w) as u32, (i / w) as u32);
                let nx = 2.0 * (x as f32 + standard_surface_bsdf::sampling::sobol(x ^ 19, y, 0, 0))
                    / w as f32
                    - 1.0;
                let ny = 1.0
                    - 2.0 * (y as f32 + standard_surface_bsdf::sampling::sobol(x ^ 19, y, 0, 1))
                        / h as f32;
                let ray = (forward + right * nx * p[P_HALF_W] + up * ny * p[P_HALF_H]).normalize();
                if n.dot(-ray) > 0.999999 {
                    fallback += 1;
                }
            }
            println!("world={world}, hits={hits}, view-facing normals={fallback}");
            assert!(hits > 100);
            // These tiny DE gradients used to trip the absolute 1e-7 cutoff,
            // replacing thousands of genuine normals with the view direction.
            assert!(fallback * 100 < hits, "world={world}: {fallback}/{hits}");
        }
    }

    #[test]
    fn cuda_primary_guides_capture_material_world_normal_and_miss_counts() {
        let _gpu_test = crate::test_gpu::lock();
        let mut gpu = Gpu::new().unwrap();
        let mut object = Scene::preset(FAMILY_KIFS);
        object.formula =
            crate::scene::Formula::Kifs(crate::scene::Kifs::preset(crate::scene::KifsKind::Menger));
        object.render.iterations = 0.0;
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
                let albedo = gpu.albedo_sums(&target);
                let normal = gpu.normal_sums(&target);
                assert!(albedo.iter().all(|p| p[3] == 4.0));
                assert!(normal.iter().all(|p| p[3] == 4.0));
                // Misses: no normal, and the albedo of what the camera ray shows, clamped to
                // [0, 1] per sample (the denoiser keeps the sky as detail).
                let misses: Vec<_> = albedo
                    .iter()
                    .zip(&normal)
                    .filter(|(_, n)| n[..3] == [0.0; 3])
                    .collect();
                assert!(!misses.is_empty());
                assert!(
                    misses
                        .iter()
                        .all(|(a, _)| a[..3].iter().all(|v| (0.0..=4.0).contains(v)))
                );
                let center = 12 * 33 + 16;
                // The kernel's primary albedo: render-rs `SurfaceInputs::guide_albedo`, base * colour *
                // tint (a metal's F0 too: metalness does not enter) plus the emission, clamped to
                // [0, 1], over working-space inputs.
                let w = crate::color::to_working;
                let (c, t, e) = (w([0.2, 0.4, 0.6]), w([1.0, 0.5, 0.25]), w([0.1, 0.2, 0.3]));
                let want: [f32; 3] =
                    std::array::from_fn(|k| (c[k] * 0.8 * t[k] + e[k] * 0.7).clamp(0.0, 1.0));
                for (actual, expected) in albedo[center][..3].iter().zip(want) {
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
                assert!(gpu.albedo_sums(&target).iter().all(|p| p[3] == 1.0));
                if world {
                    scene.objects.clear();
                    gpu.step(&mut target, &scene, 1, 19, None, false);
                    // Every ray misses: the albedo is the sky it shows, clamped; no normal.
                    assert!(gpu.albedo_sums(&target).iter().all(|p| {
                        p[3] == 1.0 && p[..3].iter().all(|v| (0.0..=1.0).contains(v))
                    }));
                    assert!(
                        gpu.normal_sums(&target)
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
        let _gpu_test = crate::test_gpu::lock();
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
        // OCIO is off: exposure and saturation in the working space, then Rec.709 display light.
        for (display, p) in target.light.iter().zip(&denoised) {
            let c = [p[0] * 4.0, p[1] * 4.0, p[2] * 4.0];
            let l = crate::color::luma(c);
            let want = crate::color::to_709(c.map(|v| l + (v - l) * 0.5));
            for k in 0..3 {
                assert!((display[k] - want[k]).abs() < 1e-5);
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
        let _gpu_test = crate::test_gpu::lock();
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
        scene.render.denoise.quality = crate::denoise::Quality::Fast;
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
        let _gpu_test = crate::test_gpu::lock();
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
        let _gpu_test = crate::test_gpu::lock();
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
        // Pure Rec.709 red / blue materials are pure only in Rec.709.
        let both = rec709(&gpu.scene_linear(&target));
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
        let _gpu_test = crate::test_gpu::lock();
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
        let first = rec709(&gpu.scene_linear(&target));
        let red: f32 = first.iter().map(|p| p[0]).sum();
        assert!(red > 0.1);
        light.sun_color = [0.0, 0.0, 1.0];
        scene.lights.push(light);
        gpu.step(&mut target, &scene, 32, 0, None, false);
        let both = rec709(&gpu.scene_linear(&target));
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
        let _gpu_test = crate::test_gpu::lock();
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
    fn cuda_world_masks_section_keep_or_cut_and_cells_thin_out() {
        use crate::scene::{Mask, MaskInstance, MaskShape};
        let _gpu_test = crate::test_gpu::lock();
        let mut gpu = Gpu::new().unwrap();
        let mut target = gpu.target(32, 24);
        let mut scene = dark_world();
        scene
            .objects
            .push(world_object(FAMILY_BULB, [0.0; 3], [1.0; 3], [1.0; 3]));
        let mut lit = |scene: &Scene| {
            gpu.step(&mut target, scene, 4, 0, None, false);
            assert!(target.colour_error.is_none(), "{:?}", target.colour_error);
            gpu.scene_linear(&target)
                .iter()
                .filter(|p| p[0] > 0.01)
                .count()
        };
        let whole = lit(&scene);
        assert!(whole > 50, "the bulb must fill part of the frame: {whole}");
        // A box keeping the y > 0 half space (across the image, so the cut face never fills the old
        // silhouette), then the same box inverted: each keeps part of the bulb.
        let half = MaskInstance {
            mask: Mask {
                shape: MaskShape::Box,
                size: [10.0; 3],
                ..Mask::default()
            },
            world: glam::Mat4::from_translation(glam::Vec3::Y * 10.0).to_cols_array_2d(),
        };
        let mut masked = scene.clone();
        masked.objects[0].masks = vec![half];
        let kept = lit(&masked);
        masked.objects[0].masks[0].mask.invert = true;
        let cut = lit(&masked);
        for part in [kept, cut] {
            assert!(
                part > whole / 5 && part < whole * 4 / 5,
                "a half-space mask keeps part of the bulb: {kept} / {cut} of {whole}"
            );
        }
        // Cells removing every cell leave nothing.
        masked.objects[0].masks = vec![MaskInstance {
            mask: Mask {
                shape: MaskShape::Cells,
                size: [0.25; 3],
                density: 1.0,
                ..Mask::default()
            },
            world: glam::Mat4::IDENTITY.to_cols_array_2d(),
        }];
        assert_eq!(lit(&masked), 0, "density 1 removes every cell");
    }

    /// The Direct display on a world (`world_direct`): each object is shaded with its own
    /// material under the world's lights, the image is deterministic (one sample per stratum
    /// of the grid) and it is not the path-traced image.
    #[test]
    fn cuda_direct_shades_each_world_object_with_its_material() {
        use crate::render_profiles::{DirectRender, DirectShading};
        let _gpu_test = crate::test_gpu::lock();
        let mut gpu = Gpu::new().unwrap();
        let mut scene = dark_world();
        scene.lighting.sky_intensity = 1.0;
        scene.lights = vec![scene.lighting];
        scene.objects = [
            ([-1.1, 0.0, 0.0], [1.0, 0.05, 0.05]),
            ([1.1, 0.0, 0.0], [0.05, 0.05, 1.0]),
        ]
        .into_iter()
        .map(|(at, color)| {
            let mut object = world_object(FAMILY_BULB, at, [0.9; 3], color);
            object.material.emission = 0.0;
            object
        })
        .collect();
        let grid = 2;
        let mut render = |scene: &Scene| {
            let mut target = gpu.target(64, 40);
            gpu.step(&mut target, scene, grid * grid, 0, None, false);
            assert!(target.colour_error.is_none(), "{:?}", target.colour_error);
            rec709(&gpu.scene_linear(&target))
        };
        let traced = render(&scene);
        scene.direct = Some(DirectRender {
            shading: DirectShading::default(),
            grid,
        });
        let direct = render(&scene);
        assert!(direct.iter().flatten().all(|v| v.is_finite()));
        assert_eq!(render(&scene), direct, "Direct is deterministic");
        assert_ne!(direct, traced, "Direct is not the path-traced image");
        let red = direct
            .iter()
            .filter(|p| p[0] > 0.02 && p[0] > 3.0 * p[2])
            .count();
        let blue = direct
            .iter()
            .filter(|p| p[2] > 0.02 && p[2] > 3.0 * p[0])
            .count();
        assert!(
            red > 20 && blue > 20,
            "each object keeps its own colour: {red} red, {blue} blue"
        );
        scene.lights.clear();
        scene.objects[0].material.base = 0.0;
        scene.objects[0].material.specular = 1.0;
        scene.objects[0].material.transmission = 1.0;
        scene.objects[0].material.transmission_color = [0.05, 1.0, 0.05];
        let glass = render(&scene);
        let green = glass
            .iter()
            .filter(|p| p[1] > 0.02 && p[1] > 2.0 * p[0] && p[1] > 2.0 * p[2])
            .count();
        assert!(
            green > 20,
            "transmissive object refracts sky with its transmission tint in Direct mode: {green} green"
        );
    }

    #[test]
    fn cuda_mask_arrays_union_their_copies() {
        use crate::scene::{ArrayAxis, ArrayKind, Mask, MaskArray, MaskInstance, MaskShape};
        let _gpu_test = crate::test_gpu::lock();
        let mut gpu = Gpu::new().unwrap();
        let mut target = gpu.target(48, 36);
        let mut scene = dark_world();
        scene
            .objects
            .push(world_object(FAMILY_BULB, [0.0; 3], [1.0; 3], [1.0; 3]));
        let mut render = |array: MaskArray, at: glam::Vec3| {
            let mut masked = scene.clone();
            masked.objects[0].masks = vec![MaskInstance {
                mask: Mask {
                    shape: MaskShape::Sphere,
                    size: [0.3; 3],
                    array,
                    ..Mask::default()
                },
                world: glam::Mat4::from_translation(at).to_cols_array_2d(),
            }];
            gpu.step(&mut target, &masked, 4, 0, None, false);
            assert!(target.colour_error.is_none(), "{:?}", target.colour_error);
            gpu.scene_linear(&target)
        };
        let lit = |pixels: &[[f32; 4]]| pixels.iter().filter(|p| p[0] > 0.01).count();
        let left = glam::Vec3::new(-0.6, 0.0, 0.0);
        // One copy of a linear array is the shape itself, to the bit.
        let single = render(MaskArray::default(), left);
        let one_copy = MaskArray {
            offset: [1.2, 0.0, 0.0],
            ..MaskArray::default()
        };
        assert_eq!(
            render(one_copy, left),
            single,
            "a one-copy array is the plain mask"
        );
        let piece = lit(&single);
        assert!(piece > 10, "the sphere keeps a piece of the bulb: {piece}");
        // A second copy one step along +X keeps the mirror piece too.
        let two = lit(&render(
            MaskArray {
                count: 2,
                ..one_copy
            },
            left,
        ));
        assert!(
            two > piece * 3 / 2,
            "two linear copies keep two pieces: {two} vs {piece}"
        );
        // A ring of four around Z through the bulb keeps more than one copy of it does.
        let ring = |count| MaskArray {
            kind: ArrayKind::Radial,
            count,
            radius: 0.6,
            axis: ArrayAxis::Z,
            ..MaskArray::default()
        };
        let one = lit(&render(ring(1), glam::Vec3::ZERO));
        let four = lit(&render(ring(4), glam::Vec3::ZERO));
        assert!(
            one > 10 && four > one * 2,
            "a radial ring of four: {four} vs {one}"
        );
    }

    /// Cost of a Cells mask on the world march: the same bulb unmasked, with a Cells mask
    /// (density 0.5) and with a Sphere mask, median of five 32-spp renders at 512x384.
    #[test]
    #[ignore = "GPU timing; run explicitly with --ignored --nocapture"]
    fn cuda_mask_cost_benchmark() {
        use crate::scene::{Mask, MaskInstance, MaskShape};
        let _gpu_test = crate::test_gpu::lock();
        let mut gpu = Gpu::new().unwrap();
        let mut target = gpu.target(512, 384);
        let mut scene = dark_world();
        scene.render.max_bounces = 2;
        scene
            .objects
            .push(world_object(FAMILY_BULB, [0.0; 3], [1.0; 3], [1.0; 3]));
        let mask = |shape, size: f32| MaskInstance {
            mask: Mask {
                shape,
                size: [size; 3],
                density: 0.5,
                ..Mask::default()
            },
            world: glam::Mat4::IDENTITY.to_cols_array_2d(),
        };
        for (name, masks) in [
            ("none", vec![]),
            ("cells", vec![mask(MaskShape::Cells, 0.15)]),
            ("sphere", vec![mask(MaskShape::Sphere, 0.9)]),
        ] {
            scene.objects[0].masks = masks;
            let mut times: Vec<f64> = (0..6)
                .map(|_| {
                    gpu.prepare_target(&mut target, &scene, None);
                    target.samples = 0;
                    let started = std::time::Instant::now();
                    gpu.step(&mut target, &scene, 32, 0, None, false);
                    assert!(!gpu.scene_linear(&target).is_empty());
                    started.elapsed().as_secs_f64() * 1000.0
                })
                .skip(1)
                .collect();
            times.sort_by(f64::total_cmp);
            eprintln!("mask {name}: {:.1} ms", times[2]);
        }
    }

    #[test]
    fn cuda_world_sibling_casts_a_shadow_outside_its_camera_silhouette() {
        let _gpu_test = crate::test_gpu::lock();
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
    fn cuda_hdr_background_interpolates_texels_instead_of_showing_nearest_blocks() {
        let _gpu_test = crate::test_gpu::lock();
        let path = std::env::temp_dir().join(format!("frac-env-filter-{}.exr", std::process::id()));
        let texels: Vec<[f32; 4]> = (0..4)
            .map(|i| [8.0 * (i % 2) as f32, 4.0 * (i / 2) as f32, 2.0, 1.0])
            .collect();
        crate::exr_io::write_rgb(
            &path,
            2,
            2,
            &texels,
            &crate::color::WORKING_PRIMS,
            None,
            true,
        )
        .unwrap();
        let mut gpu = Gpu::new().unwrap();
        let mut target = gpu.target(64, 32);
        let mut scene = Scene::preset(FAMILY_QUAT);
        scene.environment.path = path.to_string_lossy().into_owned();
        scene.lighting.sun_intensity = 0.0;
        scene.render.denoise.enabled = false;
        scene.render.max_bounces = 0;
        scene.camera.target = [1000.0, 0.0, 0.0];
        gpu.step(&mut target, &scene, 1, 0, None, false);
        let raw = gpu.scene_linear(&target);
        assert!(raw.iter().any(|p| p[0] > 1.0 && p[0] < 7.0));
        assert!(raw.iter().any(|p| p[1] > 0.2 && p[1] < 3.8));
        assert!(raw.iter().all(|p| p[0] >= 0.0
            && p[0] <= 8.0
            && p[1] >= 0.0
            && p[1] <= 4.0
            && p[2] == 2.0));
        scene.world_render = true;
        scene.objects.clear();
        gpu.step(&mut target, &scene, 1, 0, None, false);
        assert_eq!(gpu.scene_linear(&target), raw);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn cuda_hdr_environment_lights_surfaces_and_preserves_background_radiance() {
        let _gpu_test = crate::test_gpu::lock();
        let path = std::env::temp_dir().join(format!("frac-env-gpu-{}.exr", std::process::id()));
        crate::exr_io::write_rgb(
            &path,
            8,
            4,
            &[[4.0, 2.0, 1.0, 1.0]; 32],
            &crate::color::WORKING_PRIMS,
            None,
            true,
        )
        .unwrap();
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
        let _gpu_test = crate::test_gpu::lock();
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
            target.light_kind.hdr() && target.colour_error.is_none(),
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
        // Every encoding from one HDR frame: HDR10 / HLG are 16-bit BT.2020 with their cICP
        // transfer (16 = PQ, 18 = HLG), SDR is the 8-bit sRGB rendering.
        use crate::render_service::PngEncoding;
        for (encoding, depth, cicp) in [
            (PngEncoding::Hdr10, png::BitDepth::Sixteen, Some(16u8)),
            (PngEncoding::Hlg, png::BitDepth::Sixteen, Some(18)),
            (PngEncoding::Sdr8, png::BitDepth::Eight, None),
        ] {
            target
                .save_png(&png, encoding, crate::color::BT2408_SDR_WHITE_NITS, true)
                .unwrap();
            let bytes = std::fs::read(&png).unwrap();
            let at = bytes.windows(4).position(|b| b == b"cICP");
            assert_eq!(at.map(|i| bytes[i + 5]), cicp, "{encoding:?} transfer");
            assert_eq!(
                bytes.windows(4).any(|b| b == b"cLLI"),
                encoding == PngEncoding::Hdr10
            );
            let mut decoder = png::Decoder::new(std::io::Cursor::new(&bytes));
            decoder.set_transformations(png::Transformations::IDENTITY);
            let reader = decoder.read_info().unwrap();
            assert_eq!(reader.info().bit_depth, depth, "{encoding:?}");
            assert_eq!(reader.info().width, 17);
        }
        assert!(
            target
                .save_png(
                    &png,
                    PngEncoding::Sdr8,
                    crate::color::BT2408_SDR_WHITE_NITS,
                    false
                )
                .is_err(),
            "existing file without overwrite"
        );
        let exr_path = dir.join("display.exr");
        // An HDR view's absolute light: 1.0 = 100 nits whatever the SDR white.
        target
            .save_display_exr(&exr_path, crate::color::BT2408_SDR_WHITE_NITS)
            .unwrap();
        assert_eq!(
            crate::exr_io::read_attr::<exr_core::attr::Chromaticities>(&exr_path, "chromaticities"),
            Some(exr_core::attr::Chromaticities::default())
        );
        assert_eq!(
            crate::exr_io::read_attr::<f32>(&exr_path, "whiteLuminance"),
            Some(100.0)
        );
        scene.colour.view = "missing view".into();
        gpu.step(&mut target, &scene, 0, 0, None, false);
        assert!(target.colour_error.is_some());
        assert!(
            target
                .save_png(
                    &png,
                    crate::render_service::PngEncoding::Sdr8,
                    crate::color::BT2408_SDR_WHITE_NITS,
                    true
                )
                .is_err()
        );
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
    fn obsolete_bookmark_colour_payloads_are_rejected() {
        let _gpu_test = crate::test_gpu::lock();
        let scene = Scene::preset(FAMILY_BULB);
        let mut json = serde_json::to_value(&scene).unwrap();
        json.as_object_mut().unwrap().remove("colour");
        assert!(serde_json::from_value::<Scene>(json).is_err());
        // Obsolete colour fields cannot silently reinterpret scene input.
        let mut json = serde_json::to_value(&scene).unwrap();
        json["colour"]["input"] = "Linear Rec.709 (sRGB)".into();
        assert!(serde_json::from_value::<Scene>(json).is_err());
    }

    #[test]
    fn cuda_backend_resolution_respects_explicit_cuda_and_wgpu_without_silent_fallback() {
        use crate::render_profiles::Backend;
        let _gpu_test = crate::test_gpu::lock();
        let mut gpu = Gpu::new().unwrap();
        assert_eq!(gpu.resolve_backend(Backend::Auto).unwrap(), Backend::Cuda);
        assert_eq!(gpu.resolve_backend(Backend::Cuda).unwrap(), Backend::Cuda);
        assert_eq!(gpu.resolve_backend(Backend::Wgpu).unwrap(), Backend::Wgpu);

        // When CUDA is marked unavailable, explicit Backend::Cuda must fail rather than fall back to WGPU.
        gpu.cuda_unavailable = Some("CUDA driver missing".into());
        assert_eq!(
            gpu.resolve_backend(Backend::Cuda).unwrap_err(),
            "CUDA driver missing"
        );
        // Meanwhile Backend::Auto falls back to WGPU.
        assert_eq!(gpu.resolve_backend(Backend::Auto).unwrap(), Backend::Wgpu);
        gpu.cuda_unavailable = None;

        // When WGPU is marked unavailable, explicit Backend::Wgpu must fail rather than fall back to CUDA.
        let saved_wgpu = gpu.wgpu.take();
        gpu.wgpu = Some(Err("WGPU adapter unavailable".into()));
        assert_eq!(
            gpu.resolve_backend(Backend::Wgpu).unwrap_err(),
            "WGPU adapter unavailable"
        );
        assert_eq!(gpu.resolve_backend(Backend::Auto).unwrap(), Backend::Cuda);
        gpu.wgpu = saved_wgpu;
    }

    #[test]
    fn cuda_wgpu_renders_mandelbulb_world_and_picks_object() {
        use crate::render_profiles::Backend;
        let _gpu_test = crate::test_gpu::lock();
        let mut gpu = Gpu::new().unwrap();
        let mut scene = dark_world();
        scene.render.backend = Backend::Wgpu;
        scene.lighting.sky_intensity = 2.0;
        scene.lighting.background = true;
        scene.lights.push(scene.lighting);
        scene.objects.push(world_object(
            FAMILY_BULB,
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.85, 0.25, 0.15],
        ));

        let hit = gpu.pick(&scene, 64, 64, [0.0, 0.0]).unwrap();
        assert!(hit.is_some(), "WGPU pick must hit centered Mandelbulb");
        assert_eq!(hit.unwrap().0, 0);

        let mut target = gpu.target(32, 24);
        gpu.step(&mut target, &scene, 4, 7, None, false);
        assert!(target.colour_error.is_none(), "{:?}", target.colour_error);
        assert_eq!(target.active_backend, Backend::Wgpu);
        assert_eq!(target.samples, 4);
        let radiance = gpu.raw_scene_linear(&target);
        assert_eq!(radiance.len(), 32 * 24);
        assert!(
            radiance.iter().any(|p| p[0] > 0.05 && p[0].is_finite()),
            "WGPU must accumulate non-black finite scene radiance"
        );
        assert!(
            target.pixels.iter().any(|&px| (px & 0x00ff_ffff) != 0),
            "WGPU tonemap + OCIO must produce visible pixels"
        );
    }

    #[test]
    fn wgsl_compute_shader_validates_all_entry_points() {
        let src = crate::render_wgpu::shader_source();
        let module = naga::front::wgsl::parse_str(&src)
            .unwrap_or_else(|e| panic!("WGSL parse error:\n{}", e.emit_to_string(&src)));
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        );
        validator
            .validate(&module)
            .unwrap_or_else(|e| panic!("WGSL validation error:\n{}", e.emit_to_string(&src)));

        let expected = [
            "world",
            "world_fast",
            "world_fast_bulb",
            "world_direct",
            "fast_bulb",
            "fast_box",
            "fast_quat",
            "fast_kifs",
            "fast_kleinian",
            "fast_pseudo_kleinian",
            "fast_apollonian",
            "fast_hybrid",
            "full_bulb",
            "full_box",
            "full_quat",
            "full_kifs",
            "full_kleinian",
            "full_pseudo_kleinian",
            "full_apollonian",
            "full_hybrid",
            "direct_bulb",
            "direct_box",
            "direct_quat",
            "direct_kifs",
            "direct_kleinian",
            "direct_pseudo_kleinian",
            "direct_apollonian",
            "direct_hybrid",
            "tally",
            "adapt",
            "tonemap",
            "pick",
        ];
        for name in expected {
            assert!(
                module
                    .entry_points
                    .iter()
                    .any(|ep| ep.name == name && ep.stage == naga::ShaderStage::Compute),
                "missing compute entry point {name}"
            );
        }
    }
}
