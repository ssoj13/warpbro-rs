//! WarpBro's dual CUDA (`fractal-kernels`) + 28-family WGPU (`formulas.wgsl`) compute engine
//! implementing [`soluwell_studio::render::ComputeEngine`], plus one-time host registration
//! (`init_warpbro`).

use std::sync::{Arc, OnceLock};

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DeviceCopy, LaunchConfig1D};
use fractal_core::params::*;
use fractal_kernels::kernels;
use sha2::{Digest, Sha256};
use soluwell_studio::palette::build_lut;
use soluwell_studio::render::{
    BackendFilm, ComputeEngine, Gpu, RenderError, StepOutcome, StepSpec, Target, denoise_due,
    film_mismatch, fused_display,
};
use soluwell_studio::render_prep::{
    BLOCK, LightResidentKey, LutResidentKey, ObjectResidentKey, Plane, PreparationStats,
    PreparedScene, Resident, TALLY_THREADS, WorldMaterial, padded_len, untile_into,
};
use soluwell_studio::render_profiles::Backend;
use soluwell_studio::render_wgpu::{TraceDispatch, WgpuEngine, WgpuTarget};
use soluwell_studio::scene::{MaterialModel, Scene};

use crate::formulas::{FORMULAS, ITERATING_MASK, SIGNED_MASK};
use crate::fractal_node::{FractalNodeSpec, FractalSceneExt};

/// Register WarpBro's `"Fractal"` node spec, 28-family WGSL shader hook, extra thumbnail
/// cache fingerprint, and built-in scene templates into `soluwell-studio`.
/// Safe and idempotent to call from `main`, CLI subcommands, and tests.
pub fn init_warpbro() {
    static INIT: OnceLock<()> = OnceLock::new();
    INIT.get_or_init(|| {
        soluwell_studio::node_spec::register_node_type(Arc::new(FractalNodeSpec));
        soluwell_studio::render_wgpu::set_custom_shader_hook(warpbro_shader_hook);
        static FINGERPRINT: OnceLock<[u8; 32]> = OnceLock::new();
        let fp = FINGERPRINT.get_or_init(|| {
            let mut h = Sha256::new();
            for bytes in [
                include_bytes!("../Cargo.toml").as_slice(),
                include_bytes!("../Cargo.lock").as_slice(),
                include_bytes!("formulas.rs").as_slice(),
                include_bytes!("formulas.wgsl").as_slice(),
                include_bytes!("fractal_node.rs").as_slice(),
                include_bytes!("engine.rs").as_slice(),
            ] {
                h.update(bytes);
            }
            h.finalize().into()
        });
        soluwell_studio::thumbnail_cache::set_extra_renderer_fingerprint(fp);
        crate::templates::register_warpbro_templates();
    });
}

/// Construct a [`Gpu`] configured with [`WarpBroComputeEngine`] (dual CUDA + 28-family WGPU).
pub fn new_gpu() -> Result<Gpu, RenderError> {
    init_warpbro();
    Gpu::with_engine(Box::new(WarpBroComputeEngine::new()?))
}

/// WGSL shader hook injected into `soluwell-studio-render`'s `shader_source(family)`.
pub fn warpbro_shader_hook(family: Option<u32>) -> (&'static str, String) {
    use std::fmt::Write;
    let defs = include_str!("formulas.wgsl");
    let mut s = String::with_capacity(4096);
    let _ = writeln!(s, "const SIGNED_MASK: u32 = {SIGNED_MASK}u;");
    let _ = writeln!(s, "const ITERATING_MASK: u32 = {ITERATING_MASK}u;");
    for &(name, code) in &[
        ("FAMILY_BOX", FAMILY_BOX),
        ("FAMILY_QUAT", FAMILY_QUAT),
        ("FAMILY_KIFS", FAMILY_KIFS),
        ("FAMILY_KLEINIAN", FAMILY_KLEINIAN),
        ("FAMILY_PSEUDO_KLEINIAN", FAMILY_PSEUDO_KLEINIAN),
        ("FAMILY_APOLLONIAN", FAMILY_APOLLONIAN),
        ("FAMILY_HYBRID", FAMILY_HYBRID),
        ("FAMILY_AMAZING_SURFACE", FAMILY_AMAZING_SURFACE),
        ("FAMILY_BRISTORBROT", FAMILY_BRISTORBROT),
        ("FAMILY_BURNING_SHIP", FAMILY_BURNING_SHIP),
        ("FAMILY_KALEIDO_TWIST", FAMILY_KALEIDO_TWIST),
        ("FAMILY_POLYHEDRAL_KIFS", FAMILY_POLYHEDRAL_KIFS),
        ("FAMILY_HARMONIC_BULB", FAMILY_HARMONIC_BULB),
        ("FAMILY_MANDELBAR", FAMILY_MANDELBAR),
        ("FAMILY_MOBIUS_IFS", FAMILY_MOBIUS_IFS),
        ("FAMILY_GYROID_TPMS", FAMILY_GYROID_TPMS),
        ("FAMILY_LIDINOID_TPMS", FAMILY_LIDINOID_TPMS),
        ("FAMILY_ATTRACTOR_TUBE", FAMILY_ATTRACTOR_TUBE),
        ("FAMILY_TORUS_KNOT", FAMILY_TORUS_KNOT),
        ("FAMILY_HOPF_FIBRATION", FAMILY_HOPF_FIBRATION),
        ("FAMILY_DUPIN_CYCLIDE", FAMILY_DUPIN_CYCLIDE),
        ("FAMILY_SUPERFORMULA", FAMILY_SUPERFORMULA),
        ("FAMILY_PHYLLOTAXIS", FAMILY_PHYLLOTAXIS),
        ("FAMILY_NAUTILUS", FAMILY_NAUTILUS),
        ("FAMILY_VORONOI_FOAM", FAMILY_VORONOI_FOAM),
        ("FAMILY_CRYSTAL_GEODE", FAMILY_CRYSTAL_GEODE),
        ("FAMILY_ASTROLABE", FAMILY_ASTROLABE),
    ] {
        let _ = writeln!(s, "const {name}: u32 = {code}u;");
    }
    s.push_str(
        "fn family_distance(is_world: bool, obj: u32, family_override: u32, p: vec3<f32>, trap_mode: u32) -> vec2<f32> {\n",
    );
    match family {
        Some(code) => {
            let _ = writeln!(
                s,
                "    return {}(is_world, obj, p, trap_mode);",
                FORMULAS[code as usize].wgsl
            );
        }
        None => {
            s.push_str("    switch resolved_family(is_world, obj, family_override) {\n");
            for meta in &FORMULAS {
                let label = if meta.code == FAMILY_BULB {
                    "default".to_string()
                } else {
                    format!("case {}u", meta.code)
                };
                let _ = writeln!(
                    s,
                    "        {label}: {{ return {}(is_world, obj, p, trap_mode); }}",
                    meta.wgsl
                );
            }
            s.push_str("    }\n");
        }
    }
    s.push_str("}\n");
    (defs, s)
}

fn cuda_trace_entry(
    scene: &Scene,
    prepared: &PreparedScene,
    force_dynamic: bool,
) -> (fractal_kernels::TraceMaterial, Option<u32>) {
    use fractal_kernels::TraceMaterial;
    let shade = if scene.direct.is_some() {
        TraceMaterial::Direct
    } else if scene.world_render {
        match if force_dynamic {
            WorldMaterial::Mixed
        } else {
            prepared.world_material
        } {
            WorldMaterial::Fast => TraceMaterial::Fast,
            WorldMaterial::Full => TraceMaterial::Full,
            WorldMaterial::Mixed => TraceMaterial::Mixed,
        }
    } else if scene.material.model == MaterialModel::StandardSurface {
        TraceMaterial::Full
    } else {
        TraceMaterial::Fast
    };
    let family = if !scene.world_render {
        Some(scene.formula().code())
    } else if force_dynamic {
        None
    } else {
        prepared.homogeneous_family
    };
    (shade, family)
}

fn device(what: &str, error: impl std::fmt::Debug) -> RenderError {
    RenderError::Device(format!("{what}: {error:?}"))
}

fn zeroed<T: DeviceCopy>(
    stream: &CudaStream,
    len: usize,
    what: &str,
) -> Result<DeviceBuffer<T>, RenderError> {
    DeviceBuffer::zeroed(stream, len).map_err(|e| device(what, e))
}

fn read_tiled_into(
    stream: &Arc<CudaStream>,
    buffer: &DeviceBuffer<[f32; 4]>,
    width: usize,
    height: usize,
    normalize: bool,
    tiled: &mut Vec<[f32; 4]>,
    output: &mut Vec<[f32; 4]>,
) -> Result<(), RenderError> {
    tiled.resize(buffer.len(), [0.0; 4]);
    buffer
        .copy_to_host(stream, tiled)
        .map_err(|e| device("scene / guide readback", e))?;
    untile_into(tiled, width, height, normalize, output);
    Ok(())
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

    fn create_target(&self, width: usize, height: usize) -> Result<CudaTarget, RenderError> {
        let padded = padded_len(width, height);
        let s = &self.stream;
        let mut active = zeroed(s, padded / 32, "tile flags")?;
        active
            .copy_from_host(s, &vec![1u32; padded / 32])
            .map_err(|e| device("activate tiles", e))?;
        Ok(CudaTarget {
            accum: zeroed(s, padded, "accumulator")?,
            albedo: zeroed(s, padded, "albedo guides")?,
            normal: zeroed(s, padded, "normal guides")?,
            stats: zeroed(s, padded, "sample statistics")?,
            tally: zeroed(s, TALLY_THREADS, "tally sums")?,
            active,
            out: zeroed(s, width * height, "output")?,
            padded,
        })
    }

    fn clear_target(&self, ct: &mut CudaTarget) -> Result<(), RenderError> {
        let s = &self.stream;
        ct.accum
            .zero_async(s)
            .map_err(|e| device("clear accumulator", e))?;
        ct.albedo
            .zero_async(s)
            .map_err(|e| device("clear albedo guides", e))?;
        ct.normal
            .zero_async(s)
            .map_err(|e| device("clear normal guides", e))?;
        ct.stats
            .zero_async(s)
            .map_err(|e| device("clear sample statistics", e))
    }

    fn upload_prepared(
        &mut self,
        prepared: &Arc<PreparedScene>,
        env: Option<&soluwell_studio::environment::Map>,
        world: bool,
        table: &mut Vec<[f32; 4]>,
        stats: &mut PreparationStats,
    ) -> Result<(), RenderError> {
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
            })
            .map_err(RenderError::Device)?;
        if changed {
            stats.lut_uploads += 1;
            stats.upload_bytes += (table.len() * std::mem::size_of::<[f32; 4]>()) as u64;
        }
        if world {
            let objects = &mut self.objects;
            let data = &prepared.upload.objects;
            let changed = self
                .object_resident
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
                })
                .map_err(RenderError::Device)?;
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
                })
                .map_err(RenderError::Device)?;
            if changed && !data.is_empty() {
                stats.light_uploads += 1;
                stats.upload_bytes += (data.len() * std::mem::size_of::<f32>()) as u64;
            }
        }
        Ok(())
    }
}

pub struct CudaTarget {
    accum: DeviceBuffer<[f32; 4]>,
    albedo: DeviceBuffer<[f32; 4]>,
    normal: DeviceBuffer<[f32; 4]>,
    stats: DeviceBuffer<[f32; 4]>,
    tally: DeviceBuffer<[f64; 2]>,
    active: DeviceBuffer<u32>,
    out: DeviceBuffer<[f32; 4]>,
    padded: usize,
}
impl CudaTarget {
    fn plane(&self, plane: Plane) -> &DeviceBuffer<[f32; 4]> {
        match plane {
            Plane::Accum => &self.accum,
            Plane::Albedo => &self.albedo,
            Plane::Normal => &self.normal,
        }
    }
}

pub struct WarpBroComputeEngine {
    cuda: Result<CudaEngine, String>,
    wgpu: Option<Result<WgpuEngine, String>>,
    lut_staging: Vec<[f32; 4]>,
    tally_host: Vec<[f64; 2]>,
    name: String,
}

impl WarpBroComputeEngine {
    pub fn new() -> Result<Self, RenderError> {
        init_warpbro();
        let cuda = CudaEngine::new();
        let (name, wgpu) = match &cuda {
            Ok(c) => (c.name.clone(), None),
            Err(cuda_err) => {
                log::warn!("CUDA unavailable ({cuda_err}); initializing WGPU compute backend");
                let wgpu = WgpuEngine::new().map_err(|wgpu_err| {
                    RenderError::Backend(format!(
                        "{cuda_err}; WGPU fallback also failed: {wgpu_err}"
                    ))
                })?;
                let name = wgpu.name().to_owned();
                (name, Some(Ok(wgpu)))
            }
        };
        Ok(Self {
            cuda,
            wgpu,
            lut_staging: Vec::new(),
            tally_host: vec![[0.0; 2]; TALLY_THREADS],
            name,
        })
    }

    fn cuda(&self) -> Result<&CudaEngine, RenderError> {
        self.cuda
            .as_ref()
            .map_err(|e| RenderError::Backend(format!("CUDA backend unavailable: {e}")))
    }

    fn cuda_mut(&mut self) -> Result<&mut CudaEngine, RenderError> {
        self.cuda
            .as_mut()
            .map_err(|e| RenderError::Backend(format!("CUDA backend unavailable: {e}")))
    }

    fn wgpu(&self) -> Result<&WgpuEngine, RenderError> {
        match &self.wgpu {
            Some(Ok(wgpu)) => Ok(wgpu),
            Some(Err(error)) => Err(RenderError::Backend(error.clone())),
            None => Err(RenderError::Backend(
                "WGPU backend is not initialized".into(),
            )),
        }
    }

    fn wgpu_mut(&mut self) -> Result<&mut WgpuEngine, RenderError> {
        match &mut self.wgpu {
            Some(Ok(wgpu)) => Ok(wgpu),
            Some(Err(error)) => Err(RenderError::Backend(error.clone())),
            None => Err(RenderError::Backend(
                "WGPU backend is not initialized".into(),
            )),
        }
    }

    fn init_wgpu(&mut self) -> Result<Backend, RenderError> {
        match self.wgpu.get_or_insert_with(WgpuEngine::new) {
            Ok(_) => Ok(Backend::Wgpu),
            Err(error) => Err(RenderError::Backend(error.clone())),
        }
    }

    #[cfg(test)]
    pub fn probe_formula(
        &mut self,
        backend: Backend,
        scene: &Scene,
        points: &[[f32; 4]],
    ) -> Result<Vec<[f32; 4]>, RenderError> {
        if points.is_empty() {
            return Ok(Vec::new());
        }
        let p: [f32; P_COUNT] = scene.pack(1, 1).try_into().expect("packed parameter ABI");
        if backend == Backend::Wgpu {
            let wgpu = self.wgpu_mut()?;
            wgpu.set_params(&p)?;
            return wgpu.probe_formula(scene.formula().code(), points);
        }
        let cuda = self.cuda_mut()?;
        cuda.module
            .set_params(&cuda.stream, &p)
            .map_err(|e| device("probe parameters", e))?;
        let mut query = zeroed(&cuda.stream, points.len(), "formula probe queries")?;
        query
            .copy_from_host(&cuda.stream, points)
            .map_err(|e| device("formula probe upload", e))?;
        let mut out = zeroed(&cuda.stream, points.len(), "formula probe output")?;
        let cfg = LaunchConfig1D::new((points.len() as u32).div_ceil(128), 128, 0);
        let launch = cuda
            .module
            .prepare_probe_formula(cfg)
            .map_err(|e| device("prepare formula probe", e))?;
        cuda.module
            .probe_formula(
                &cuda.stream,
                &launch,
                &cuda.objects,
                &cuda.lights,
                &query,
                &mut out,
            )
            .map_err(|e| device("formula probe", e))?;
        let mut result = vec![[0.0; 4]; points.len()];
        out.copy_to_host(&cuda.stream, &mut result)
            .map_err(|e| device("formula probe readback", e))?;
        Ok(result)
    }

    pub fn cuda_kernel_resources(
        &self,
        scene: &Scene,
        prepared: &PreparedScene,
    ) -> Result<serde_json::Value, RenderError> {
        let (shade, family) = cuda_trace_entry(scene, prepared, false);
        let module = &self.cuda()?.module;
        macro_rules! resources {
            ($prepare:ident, $run:ident) => {{
                let function = module
                    .as_cuda_module()
                    .load_function(stringify!($run))
                    .map_err(|e| device("query CUDA kernel", e))?;
                Ok(serde_json::json!({
                    "entry": stringify!($run),
                    "registers": function
                        .num_registers()
                        .map_err(|e| device("CUDA register count", e))?,
                    "local_bytes": function
                        .local_size_bytes()
                        .map_err(|e| device("CUDA local size", e))?,
                }))
            }};
        }
        fractal_kernels::dispatch_trace!(
            resources,
            scene.world_render,
            shade,
            family,
            Err(RenderError::Scene("unsupported CUDA resource entry".into()))
        )
    }
}

impl ComputeEngine for WarpBroComputeEngine {
    fn name(&self) -> &str {
        &self.name
    }

    fn resolve_backend(&mut self, scene: &Scene) -> Result<Backend, RenderError> {
        match scene.render.backend {
            Backend::Cuda => match &self.cuda {
                Ok(_) => Ok(Backend::Cuda),
                Err(error) => Err(RenderError::Backend(error.clone())),
            },
            Backend::Wgpu => self.init_wgpu(),
            Backend::Auto if self.cuda.is_err() => self.init_wgpu(),
            Backend::Auto => Ok(Backend::Cuda),
        }
    }

    fn pipeline_ready(
        &mut self,
        backend: Backend,
        scene: &Scene,
        prepared: &PreparedScene,
        pick: bool,
        force_mixed_world: bool,
    ) -> Result<bool, RenderError> {
        if pick && (!scene.world_render || scene.objects.is_empty()) {
            return Ok(true);
        }
        if backend != Backend::Wgpu {
            return Ok(true);
        }
        self.init_wgpu()?;
        self.wgpu()?
            .pipeline_ready(scene, prepared, pick, force_mixed_world)
    }

    fn create_film(
        &mut self,
        backend: Backend,
        width: usize,
        height: usize,
    ) -> Result<BackendFilm, RenderError> {
        match backend {
            Backend::Cuda => Ok(Box::new(self.cuda()?.create_target(width, height)?)),
            Backend::Wgpu => {
                self.init_wgpu()?;
                Ok(Box::new(self.wgpu()?.create_target(width, height)?))
            }
            Backend::Auto => Err(RenderError::Backend(
                "The Auto backend was not resolved before allocating a film".into(),
            )),
        }
    }

    fn can_reuse_film(&self, film: &BackendFilm, backend: Backend, padded: usize) -> bool {
        match backend {
            Backend::Cuda => film
                .downcast_ref::<CudaTarget>()
                .is_some_and(|ct| ct.padded == padded),
            Backend::Wgpu => film
                .downcast_ref::<WgpuTarget>()
                .is_some_and(|wt| wt.padded() == padded),
            Backend::Auto => false,
        }
    }

    fn clear_film(&mut self, backend: Backend, film: &mut BackendFilm) -> Result<(), RenderError> {
        match backend {
            Backend::Cuda => {
                let ct = film
                    .downcast_mut::<CudaTarget>()
                    .ok_or_else(film_mismatch)?;
                self.cuda()?.clear_target(ct)
            }
            Backend::Wgpu => {
                let wt = film
                    .downcast_mut::<WgpuTarget>()
                    .ok_or_else(film_mismatch)?;
                self.wgpu()?.clear_target(wt)
            }
            Backend::Auto => Err(film_mismatch()),
        }
    }

    fn activate_all(
        &mut self,
        backend: Backend,
        film: &mut BackendFilm,
        active_host: &[u32],
    ) -> Result<(), RenderError> {
        match backend {
            Backend::Cuda => {
                let ct = film
                    .downcast_mut::<CudaTarget>()
                    .ok_or_else(film_mismatch)?;
                ct.active
                    .copy_from_host(&self.cuda()?.stream, active_host)
                    .map_err(|e| device("activate tiles", e))
            }
            Backend::Wgpu => {
                let wt = film
                    .downcast_mut::<WgpuTarget>()
                    .ok_or_else(film_mismatch)?;
                self.wgpu()?.upload_active(wt, active_host)
            }
            Backend::Auto => Err(film_mismatch()),
        }
    }

    fn upload_and_set_params(
        &mut self,
        backend: Backend,
        prepared: &Arc<PreparedScene>,
        env: Option<&soluwell_studio::environment::Map>,
        world_render: bool,
        params: &[f32; P_COUNT],
        stats: &mut PreparationStats,
    ) -> Result<(), RenderError> {
        if backend == Backend::Wgpu {
            self.init_wgpu()?;
            let wgpu = self.wgpu_mut()?;
            wgpu.upload_prepared(prepared, env, world_render, stats)?;
            wgpu.set_params(params)
        } else {
            let mut staging = std::mem::take(&mut self.lut_staging);
            let res = {
                let cuda = self.cuda_mut()?;
                cuda.upload_prepared(prepared, env, world_render, &mut staging, stats)
                    .and_then(|()| {
                        cuda.module
                            .set_params(&cuda.stream, params)
                            .map_err(|e| device("CUDA parameter upload", e))
                    })
            };
            self.lut_staging = staging;
            res
        }
    }

    fn execute_step(
        &mut self,
        backend: Backend,
        target: &mut Target,
        trace: TraceDispatch<'_>,
        step: StepSpec,
        display: &mut dyn FnMut(&Target) -> bool,
    ) -> Result<StepOutcome, RenderError> {
        let scene = trace.scene;
        let prepared = trace.prepared;
        let full = trace.full_material;
        let force_mixed_world = trace.force_mixed_world;
        let spp = step.spp;
        let final_pass = step.final_pass;
        if backend == Backend::Wgpu {
            let adaptive = scene.render.adaptive;
            let next_samples = target.samples + spp;
            let run_adapt = adaptive.enabled && next_samples >= adaptive.min();
            let total_tiles = target.active_host.len();
            let wgpu = self.wgpu()?;
            if !run_adapt {
                target.samples = next_samples;
                target.tally_dirty = true;
                if !display(target) {
                    let wt = target
                        .film
                        .as_ref()
                        .and_then(|(_, f)| f.downcast_ref::<WgpuTarget>())
                        .ok_or_else(film_mismatch)?;
                    wgpu.launch_trace(wt, scene, prepared, full, force_mixed_world)?;
                    return Ok(StepOutcome::CompletedNoDisplay);
                }
                let due = denoise_due(target, scene, final_pass);
                if fused_display(target, scene, due) {
                    let wt = target
                        .film
                        .as_ref()
                        .and_then(|(_, f)| f.downcast_ref::<WgpuTarget>())
                        .ok_or_else(film_mismatch)?;
                    target.unresolved = wgpu.launch_fused_step_display(
                        wt,
                        trace,
                        target.width,
                        target.height,
                        &mut target.light,
                        &mut target.pixels,
                    )?;
                    target.tally_dirty = false;
                    return Ok(StepOutcome::FusedDisplayDone);
                }
                let wt = target
                    .film
                    .as_ref()
                    .and_then(|(_, f)| f.downcast_ref::<WgpuTarget>())
                    .ok_or_else(film_mismatch)?;
                let (unresolved, _, _) =
                    wgpu.launch_trace_and_tally(wt, trace, false, total_tiles)?;
                target.unresolved = unresolved;
                target.tally_dirty = false;
                Ok(StepOutcome::NeedsDisplay {
                    decided_due: Some(due),
                })
            } else {
                let wt = target
                    .film
                    .as_ref()
                    .and_then(|(_, f)| f.downcast_ref::<WgpuTarget>())
                    .ok_or_else(film_mismatch)?;
                let (unresolved, active, converged) =
                    wgpu.launch_trace_and_tally(wt, trace, true, total_tiles)?;
                target.unresolved = unresolved;
                target.tally_dirty = false;
                target.active_tiles = (active, total_tiles);
                target.converged = converged;
                target.samples = next_samples;
                if !display(target) {
                    return Ok(StepOutcome::CompletedNoDisplay);
                }
                Ok(StepOutcome::NeedsDisplay { decided_due: None })
            }
        } else {
            {
                let cuda = self.cuda()?;
                let ct = target
                    .film
                    .as_mut()
                    .and_then(|(_, f)| f.downcast_mut::<CudaTarget>())
                    .ok_or_else(film_mismatch)?;
                let padded = ct.accum.len() as u32;
                let cfg = LaunchConfig1D::new(padded.div_ceil(BLOCK), BLOCK, 0);
                let (m, s, lut, acc) = (&cuda.module, &cuda.stream, &cuda.lut, &mut ct.accum);
                macro_rules! launch {
                    ($prep:ident, $run:ident) => {{
                        let prep = m.$prep(cfg).map_err(|e| device(stringify!($prep), e))?;
                        m.$run(
                            s,
                            &prep,
                            lut,
                            &cuda.objects,
                            &cuda.lights,
                            acc,
                            &mut ct.albedo,
                            &mut ct.normal,
                            &ct.active,
                            &mut ct.stats,
                        )
                        .map_err(|e| device(stringify!($run), e))?;
                    }};
                }
                let (shade, family) = cuda_trace_entry(scene, prepared, force_mixed_world);
                fractal_kernels::dispatch_trace!(
                    launch,
                    scene.world_render,
                    shade,
                    family,
                    return Err(RenderError::Scene("unsupported CUDA trace entry".into()))
                );
            }
            target.samples += spp;
            target.tally_dirty = true;
            let adaptive = scene.render.adaptive;
            if adaptive.enabled && target.samples >= adaptive.min() {
                let tiles = target.active_host.len();
                let cuda = self.cuda()?;
                let ct = target
                    .film
                    .as_mut()
                    .and_then(|(_, f)| f.downcast_mut::<CudaTarget>())
                    .ok_or_else(film_mismatch)?;
                let cfg = LaunchConfig1D::new((tiles as u32).div_ceil(BLOCK), BLOCK, 0);
                let prep = cuda
                    .module
                    .prepare_adapt(cfg)
                    .map_err(|e| device("prepare adapt", e))?;
                cuda.module
                    .adapt(&cuda.stream, &prep, &ct.accum, &ct.stats, &mut ct.active)
                    .map_err(|e| device("adapt", e))?;
                ct.active
                    .copy_to_host(&cuda.stream, &mut target.active_host)
                    .map_err(|e| device("tile flags readback", e))?;
                let active = target.active_host.iter().filter(|&&a| a != 0).count();
                target.active_tiles = (active, tiles);
                target.converged = active == 0;
            }
            if !display(target) {
                return Ok(StepOutcome::CompletedNoDisplay);
            }
            Ok(StepOutcome::NeedsDisplay { decided_due: None })
        }
    }

    fn tally(&mut self, backend: Backend, film: &mut BackendFilm) -> Result<f32, RenderError> {
        if backend == Backend::Wgpu {
            let wt = film
                .downcast_mut::<WgpuTarget>()
                .ok_or_else(film_mismatch)?;
            self.wgpu()?.tally(wt)
        } else {
            let ct = film
                .downcast_mut::<CudaTarget>()
                .ok_or_else(film_mismatch)?;
            let mut tally_host = std::mem::take(&mut self.tally_host);
            let res = (|| -> Result<f32, RenderError> {
                let cuda = self.cuda()?;
                let cfg = LaunchConfig1D::new(TALLY_THREADS as u32 / BLOCK, BLOCK, 0);
                let prepared = cuda
                    .module
                    .prepare_tally(cfg)
                    .map_err(|e| device("prepare tally", e))?;
                cuda.module
                    .tally(&cuda.stream, &prepared, &ct.accum, &ct.stats, &mut ct.tally)
                    .map_err(|e| device("tally", e))?;
                ct.tally
                    .copy_to_host(&cuda.stream, &mut tally_host)
                    .map_err(|e| device("tally readback", e))?;
                let (unresolved, samples) = tally_host
                    .iter()
                    .fold((0.0f64, 0.0f64), |(l, s), t| (l + t[0], s + t[1]));
                Ok(if samples > 0.0 {
                    (unresolved / samples) as f32
                } else {
                    0.0
                })
            })();
            self.tally_host = tally_host;
            res
        }
    }

    fn tonemap_display_fast(
        &mut self,
        backend: Backend,
        film: &mut BackendFilm,
        width: usize,
        height: usize,
        light: &mut Vec<[f32; 4]>,
        pixels: &mut Vec<u32>,
    ) -> Result<bool, RenderError> {
        if backend == Backend::Wgpu {
            let wt = film
                .downcast_mut::<WgpuTarget>()
                .ok_or_else(film_mismatch)?;
            self.wgpu()?
                .tonemap_display(wt, width, height, light, pixels)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn tonemap_raw(
        &mut self,
        backend: Backend,
        film: &mut BackendFilm,
        width: usize,
        height: usize,
        raw: &mut Vec<[f32; 4]>,
    ) -> Result<(), RenderError> {
        if backend == Backend::Wgpu {
            let wt = film
                .downcast_mut::<WgpuTarget>()
                .ok_or_else(film_mismatch)?;
            self.wgpu()?.tonemap(wt, width, height, raw)
        } else {
            let ct = film
                .downcast_mut::<CudaTarget>()
                .ok_or_else(film_mismatch)?;
            let cuda = self.cuda()?;
            let n = (width * height) as u32;
            let cfg = LaunchConfig1D::new(n.div_ceil(BLOCK), BLOCK, 0);
            let prepared = cuda
                .module
                .prepare_tonemap(cfg)
                .map_err(|e| device("prepare tonemap", e))?;
            cuda.module
                .tonemap(&cuda.stream, &prepared, &ct.accum, &mut ct.out)
                .map_err(|e| device("tonemap", e))?;
            ct.out
                .copy_to_host(&cuda.stream, raw)
                .map_err(|e| device("display readback", e))
        }
    }

    fn read_film(
        &self,
        film: &BackendFilm,
        plane: Plane,
        dims: (usize, usize),
        normalize: bool,
        tiled: &mut Vec<[f32; 4]>,
        output: &mut Vec<[f32; 4]>,
    ) -> Result<(), RenderError> {
        if let Some(wt) = film.downcast_ref::<WgpuTarget>() {
            self.wgpu()?
                .read_tiled_into(wt, plane, dims, normalize, tiled, output)
        } else {
            let ct = film
                .downcast_ref::<CudaTarget>()
                .ok_or_else(film_mismatch)?;
            read_tiled_into(
                &self.cuda()?.stream,
                ct.plane(plane),
                dims.0,
                dims.1,
                normalize,
                tiled,
                output,
            )
        }
    }

    fn pick(
        &mut self,
        backend: Backend,
        prepared: &Arc<PreparedScene>,
        env: Option<&soluwell_studio::environment::Map>,
        params: &[f32; P_COUNT],
        (ndc, family): ([f32; 2], Option<u32>),
        stats: &mut PreparationStats,
    ) -> Result<Option<(usize, f32)>, RenderError> {
        if backend == Backend::Wgpu {
            self.init_wgpu()?;
            let wgpu = self.wgpu_mut()?;
            wgpu.upload_prepared(prepared, env, true, stats)?;
            wgpu.set_params(params)?;
            return wgpu.pick(ndc, family);
        }
        let mut staging = std::mem::take(&mut self.lut_staging);
        let res = (|| -> Result<Option<(usize, f32)>, RenderError> {
            let cuda = self.cuda_mut()?;
            cuda.upload_prepared(prepared, env, true, &mut staging, stats)?;
            cuda.module
                .set_params(&cuda.stream, params)
                .map_err(|e| device("CUDA parameter upload", e))?;
            let mut query = zeroed(&cuda.stream, 1, "pick query")?;
            query
                .copy_from_host(&cuda.stream, &[[ndc[0], ndc[1], 0.0, 0.0]])
                .map_err(|e| device("pick upload", e))?;
            let mut out = zeroed(&cuda.stream, 1, "pick result")?;
            macro_rules! pick_entry {
                ($prepare:ident, $run:ident) => {{
                    let launch = cuda
                        .module
                        .$prepare(LaunchConfig1D::new(1, 32, 0))
                        .map_err(|e| device("prepare pick", e))?;
                    cuda.module
                        .$run(
                            &cuda.stream,
                            &launch,
                            &cuda.objects,
                            &cuda.lights,
                            &query,
                            &mut out,
                        )
                        .map_err(|e| device("pick", e))?;
                }};
            }
            fractal_kernels::dispatch_pick!(
                pick_entry,
                family,
                return Err(RenderError::Scene("unsupported CUDA pick entry".into()))
            );
            let mut hit = [[0.0f32; 4]];
            out.copy_to_host(&cuda.stream, &mut hit)
                .map_err(|e| device("pick readback", e))?;
            let [[object, _, depth, _]] = hit;
            Ok((object >= 1.0).then(|| (object as usize - 1, depth)))
        })();
        self.lut_staging = staging;
        res
    }

    fn finish(&self, backend: Backend) -> Result<(), RenderError> {
        if backend == Backend::Wgpu {
            self.wgpu()?.finish()
        } else {
            self.cuda()?
                .stream
                .synchronize()
                .map_err(|e| device("CUDA stream synchronize", e))
        }
    }
}

pub trait WarpBroGpuExt {
    #[cfg(test)]
    fn set_force_mixed_world(&mut self, force: bool);
    #[cfg(test)]
    fn probe_formula(
        &mut self,
        scene: &Scene,
        points: &[[f32; 4]],
    ) -> Result<Vec<[f32; 4]>, RenderError>;
    fn cuda_kernel_resources(
        &mut self,
        scene: &Scene,
        width: usize,
        height: usize,
    ) -> Result<serde_json::Value, RenderError>;
}

impl WarpBroGpuExt for Gpu {
    #[cfg(test)]
    fn set_force_mixed_world(&mut self, force: bool) {
        self.force_mixed_world = force;
    }

    #[cfg(test)]
    fn probe_formula(
        &mut self,
        scene: &Scene,
        points: &[[f32; 4]],
    ) -> Result<Vec<[f32; 4]>, RenderError> {
        let backend = self.resolve_backend(scene)?;
        let wb = (&mut *self.engine as &mut dyn std::any::Any)
            .downcast_mut::<WarpBroComputeEngine>()
            .ok_or_else(|| RenderError::Backend("Not a WarpBroComputeEngine".into()))?;
        wb.probe_formula(backend, scene, points)
    }

    fn cuda_kernel_resources(
        &mut self,
        scene: &Scene,
        width: usize,
        height: usize,
    ) -> Result<serde_json::Value, RenderError> {
        if self.resolve_backend(scene)? != Backend::Cuda {
            return Err(RenderError::Backend(
                "CUDA resources require the CUDA backend".into(),
            ));
        }
        let prepared = self
            .preparation
            .get(scene, width as u32, height as u32)
            .map_err(RenderError::Scene)?;
        let wb = (&*self.engine as &dyn std::any::Any)
            .downcast_ref::<WarpBroComputeEngine>()
            .ok_or_else(|| RenderError::Backend("Not a WarpBroComputeEngine".into()))?;
        wb.cuda_kernel_resources(scene, &prepared)
    }
}
