//! WarpBro's dual CUDA (`fractal-kernels`) + 28-family WGPU (`formulas.wgsl`) compute engine
//! implementing [`sl_studio::render::ComputeEngine`], plus one-time host registration
//! ([`init_warpbro`]) and [`WarpBroFractalModule`] (`StudioModule`).
//!
//! Exists so `sl-studio` remains fractal-agnostic while WarpBro drives both the embedded
//! `cuda-oxide` PTX kernels (`fractal_kernels::kernels`) and the per-family WGSL shader hook
//! ([`warpbro_shader_hook`]) over the shared `P_COUNT` studio ABI. Consumed by `src/main.rs`,
//! `src/render_bench.rs`, and `src/parity_tests.rs`.

use std::sync::{Arc, OnceLock};

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DeviceCopy, LaunchConfig1D};
use fractal_core::params::*;
use fractal_kernels::kernels;
use sha2::{Digest, Sha256};
use sl_studio::palette::build_lut;
use sl_studio::render::{
    BackendFilm, ComputeEngine, Gpu, RenderError, StepOutcome, StepSpec, Target, WgpuComputeEngine,
    film_mismatch, validate_size,
};
use sl_studio::render_prep::{
    BLOCK, LightResidentKey, LutResidentKey, ObjectResidentKey, Plane, PreparationStats,
    PreparedScene, Resident, TALLY_THREADS, WorldMaterial, padded_len, tile_count, untile_into,
};
use sl_studio::render_profiles::Backend;
use sl_studio::render_wgpu::{TraceDispatch, WgpuTarget};
use sl_studio::scene::{MaterialModel, Scene};

use crate::formulas::{FORMULAS, ITERATING_MASK, SIGNED_MASK};
use crate::fractal_node::FractalNodeSpec;

/// Hot-pluggable [`sl_studio::StudioModule`] ("organ") registering WarpBro's `"Fractal"`
/// SDF scene node (`FractalNodeSpec`), 28-family WGSL shader hook, thumbnail fingerprint, and
/// built-in scene templates into `sl-studio`.
pub struct WarpBroFractalModule;

impl sl_studio::StudioModule for WarpBroFractalModule {
    fn id(&self) -> &'static str {
        "warpbro.fractals"
    }

    fn title(&self) -> &'static str {
        "WarpBro 3D Fractals"
    }

    fn node_specs(&self) -> Vec<std::sync::Arc<dyn sl_studio::node_spec::NodeTypeSpec>> {
        vec![std::sync::Arc::new(crate::fractal_node::FractalNodeSpec)]
    }

    fn on_register(
        &self,
        _bus: &sl_studio::app_control::appbus::EventBus,
        _hotkeys: &mut sl_studio::HotkeyRegistry,
    ) {
        init_warpbro();
    }
}

/// Register WarpBro's `"Fractal"` node spec, 28-family WGSL shader hook, extra thumbnail
/// cache fingerprint, and built-in scene templates into `sl-studio`.
/// Safe and idempotent to call from `main`, CLI subcommands, and tests.
///
/// `register_node_type` replaces by name, so the second registration through
/// `WarpBroFractalModule::node_specs` is the same spec, not a duplicate; CLI paths that never
/// build a `StudioConfig` (`--bench`, `--gallery`, tests) rely on this one.
pub fn init_warpbro() {
    static INIT: OnceLock<()> = OnceLock::new();
    INIT.get_or_init(|| {
        sl_studio::node_spec::register_node_type(Arc::new(FractalNodeSpec));
        sl_studio::render_wgpu::set_custom_shader_hook(warpbro_shader_hook);
        // `set_extra_renderer_fingerprint` keeps a `&'static [u8]`: the static owns the digest.
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
        sl_studio::thumbnail_cache::set_extra_renderer_fingerprint(fp);
        crate::templates::register_warpbro_templates();
    });
}

/// Load the embedded CUDA module on device 0, which makes the driver JIT-compile and cache
/// the PTX, and return the device name. `--warmup` calls this directly instead of
/// [`new_gpu`]: that falls back to WGPU when CUDA is missing, which would make a CUDA warmup
/// report success without loading anything.
pub fn warmup_cuda() -> Result<String, String> {
    CudaEngine::new().map(|cuda| cuda.name)
}

/// Construct a [`Gpu`] configured with [`WarpBroComputeEngine`] (dual CUDA + 28-family WGPU).
pub fn new_gpu() -> Result<Gpu, RenderError> {
    init_warpbro();
    Gpu::with_engine(Box::new(WarpBroComputeEngine::new()?))
}

/// WGSL shader hook injected into `sl-studio-render`'s `shader_source(family)`.
pub fn warpbro_shader_hook(family: Option<u32>) -> (&'static str, String) {
    use std::fmt::Write;
    let defs = include_str!("formulas.wgsl");
    let mut s = String::with_capacity(4096);
    // `fmt::Write` for `String` never fails: the `let _` discards an always-`Ok`.
    // Only the masks are emitted: `render_wgpu.wgsl` reads them, and soluwell emits
    // `FAMILY_BULB` itself; the estimators in `formulas.wgsl` take no family constants.
    let _ = writeln!(s, "const SIGNED_MASK: u32 = {SIGNED_MASK}u;");
    let _ = writeln!(s, "const ITERATING_MASK: u32 = {ITERATING_MASK}u;");
    s.push_str(
        "fn family_distance(is_world: bool, obj: u32, family_override: u32, p: vec3<f32>, trap_mode: u32) -> vec2<f32> {\n",
    );
    // A family code outside `FORMULAS` is a miss, as in CUDA's `family_distance`
    // (`(f32::MAX, TRAP_START)`): both backends then render nothing for it instead of WGPU
    // drawing a Mandelbulb. Scenes never carry one (`FractalNodeSpec::prepare` parses it).
    const MISS: &str = "return vec2<f32>(F32_MAX, TRAP_START);";
    match family {
        Some(code) => match FORMULAS.get(code as usize) {
            Some(meta) => {
                let _ = writeln!(s, "    return {}(is_world, obj, p, trap_mode);", meta.wgsl);
            }
            None => {
                let _ = writeln!(s, "    {MISS}");
            }
        },
        None => {
            s.push_str("    switch resolved_family(is_world, obj, family_override) {\n");
            for meta in &FORMULAS {
                let _ = writeln!(
                    s,
                    "        case {}u: {{ return {}(is_world, obj, p, trap_mode); }}",
                    meta.code, meta.wgsl
                );
            }
            let _ = writeln!(s, "        default: {{ {MISS} }}");
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
        Some(scene.family_code())
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

fn unresolved_auto() -> RenderError {
    RenderError::Backend("The Auto backend was not resolved before dispatch".into())
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
    pick_query: DeviceBuffer<[f32; 4]>,
    pick_out: DeviceBuffer<[f32; 4]>,
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
        // SAFETY: `kernels::load` trusts that the embedded PTX matches the host launch
        // signatures it generated. Both come from the same `fractal-kernels` build (the
        // `#[cuda_module]` bundle compiled into this binary, one `P_COUNT` ABI from
        // `fractal_core::params`), so they cannot disagree. The module runs in `ctx`'s CUDA
        // context, which this engine keeps in `_ctx`.
        let module = unsafe { kernels::load(&ctx) }.map_err(|e| format!("load module: {e:?}"))?;
        module
            .set_params_blocking(&[0.0; P_COUNT])
            .map_err(|e| format!("CUDA parameter block: {e:?}"))?;
        log::info!(
            "CUDA startup: kernels ready in {:?}",
            module_started.elapsed()
        );
        // The name is a label only (status bar, bench header): log a failed query, keep going.
        let name = ctx.device_name().unwrap_or_else(|e| {
            log::warn!("CUDA device name query failed: {e:?}");
            "CUDA GPU".into()
        });
        let lut =
            DeviceBuffer::zeroed(&stream, PALETTE_SAMPLES + 1).map_err(|e| format!("{e:?}"))?;
        Ok(Self {
            _ctx: ctx,
            stream: stream.clone(),
            module,
            lut,
            objects: DeviceBuffer::zeroed(&stream, 1).map_err(|e| format!("objects: {e:?}"))?,
            lights: DeviceBuffer::zeroed(&stream, 1).map_err(|e| format!("lights: {e:?}"))?,
            pick_query: DeviceBuffer::zeroed(&stream, 1)
                .map_err(|e| format!("pick query: {e:?}"))?,
            pick_out: DeviceBuffer::zeroed(&stream, 1)
                .map_err(|e| format!("pick result: {e:?}"))?,
            lut_resident: Default::default(),
            object_resident: Default::default(),
            light_resident: Default::default(),
            name,
        })
    }

    fn create_target(&self, width: usize, height: usize) -> Result<CudaTarget, RenderError> {
        validate_size(width, height, 1)?;
        let pixels = width * height;
        let padded = padded_len(width, height);
        let s = &self.stream;
        // Zeroed here; `Gpu::prepare` calls `activate_all` right after `create_film`, which
        // uploads the flags from `Target::active_host` (one source for the initial state).
        let active = zeroed(s, tile_count(width, height), "tile flags")?;
        Ok(CudaTarget {
            accum: zeroed(s, padded, "accumulator")?,
            albedo: zeroed(s, padded, "albedo guides")?,
            normal: zeroed(s, padded, "normal guides")?,
            stats: zeroed(s, padded, "sample statistics")?,
            tally: zeroed(s, TALLY_THREADS, "tally sums")?,
            active,
            out: zeroed(s, pixels, "output")?,
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
        env: Option<&sl_studio::environment::Map>,
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

/// CUDA (`fractal-kernels`) plus WGPU. The WGPU backend is soluwell's own
/// [`WgpuComputeEngine`]: every `Backend::Wgpu` arm delegates to it, so step pacing, adaptive
/// sampling and display bookkeeping have one implementation; WarpBro contributes only the
/// shader hook ([`warpbro_shader_hook`], installed globally by [`init_warpbro`]).
pub struct WarpBroComputeEngine {
    cuda: Result<CudaEngine, String>,
    /// Created on first WGPU use. A failed creation is kept, not retried: a missing adapter
    /// would otherwise be re-probed on every render step; the error reaches each requester.
    wgpu: Option<Result<WgpuComputeEngine, String>>,
    lut_staging: Vec<[f32; 4]>,
    tally_host: Vec<[f64; 2]>,
}

impl WarpBroComputeEngine {
    pub fn new() -> Result<Self, RenderError> {
        init_warpbro();
        let cuda = CudaEngine::new();
        let wgpu = match &cuda {
            Ok(_) => None,
            Err(cuda_err) => {
                log::warn!("CUDA unavailable ({cuda_err}); initializing WGPU compute backend");
                let wgpu = WgpuComputeEngine::new().map_err(|wgpu_err| {
                    RenderError::Backend(format!(
                        "{cuda_err}; WGPU fallback also failed: {wgpu_err}"
                    ))
                })?;
                Some(Ok(wgpu))
            }
        };
        Ok(Self {
            cuda,
            wgpu,
            lut_staging: Vec::new(),
            tally_host: vec![[0.0; 2]; TALLY_THREADS],
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

    /// The WGPU backend if it was created; `&self` trait methods cannot create it.
    fn wgpu(&self) -> Result<&WgpuComputeEngine, RenderError> {
        match &self.wgpu {
            Some(Ok(wgpu)) => Ok(wgpu),
            Some(Err(error)) => Err(RenderError::Backend(error.clone())),
            None => Err(RenderError::Backend(
                "WGPU backend is not initialized".into(),
            )),
        }
    }

    /// The WGPU backend, created on first use (see the `wgpu` field).
    fn wgpu_mut(&mut self) -> Result<&mut WgpuComputeEngine, RenderError> {
        let created = self
            .wgpu
            .get_or_insert_with(|| WgpuComputeEngine::new().map_err(|e| e.to_string()));
        match created {
            Ok(wgpu) => Ok(wgpu),
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
        let backend = match backend {
            Backend::Auto => self.resolve_backend(scene)?,
            Backend::Wgpu => {
                self.wgpu_mut()?;
                Backend::Wgpu
            }
            Backend::Cuda => {
                self.cuda()?;
                Backend::Cuda
            }
        };
        let p: [f32; P_COUNT] = scene.pack(1, 1).try_into().expect("packed parameter ABI");
        if backend == Backend::Wgpu {
            let wgpu = self.wgpu_mut()?.wgpu_mut()?;
            wgpu.set_params(&p)?;
            return wgpu.probe_formula(scene.family_code(), points);
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
    fn device_name(&self, backend: Backend) -> Result<&str, RenderError> {
        match backend {
            Backend::Cuda => Ok(self.cuda()?.name.as_str()),
            Backend::Wgpu => self.wgpu()?.device_name(Backend::Wgpu),
            Backend::Auto => Err(RenderError::Backend(
                "Auto names no device: it is resolved to CUDA or WGPU first".into(),
            )),
        }
    }

    fn resolve_backend(&mut self, scene: &Scene) -> Result<Backend, RenderError> {
        match scene.render.backend {
            Backend::Cuda => match &self.cuda {
                Ok(_) => Ok(Backend::Cuda),
                Err(error) => Err(RenderError::Backend(error.clone())),
            },
            Backend::Wgpu => self.wgpu_mut().map(|_| Backend::Wgpu),
            Backend::Auto if self.cuda.is_err() => self.wgpu_mut().map(|_| Backend::Wgpu),
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
        match backend {
            Backend::Cuda => {
                self.cuda()?;
                Ok(true)
            }
            Backend::Wgpu => {
                self.wgpu_mut()?
                    .pipeline_ready(backend, scene, prepared, pick, force_mixed_world)
            }
            Backend::Auto => Err(unresolved_auto()),
        }
    }

    fn create_film(
        &mut self,
        backend: Backend,
        width: usize,
        height: usize,
    ) -> Result<BackendFilm, RenderError> {
        match backend {
            Backend::Cuda => Ok(Box::new(self.cuda()?.create_target(width, height)?)),
            Backend::Wgpu => self.wgpu_mut()?.create_film(backend, width, height),
            Backend::Auto => Err(unresolved_auto()),
        }
    }

    fn can_reuse_film(&self, film: &BackendFilm, backend: Backend, padded: usize) -> bool {
        match backend {
            Backend::Cuda => film
                .downcast_ref::<CudaTarget>()
                .is_some_and(|ct| ct.padded == padded),
            Backend::Wgpu => self
                .wgpu()
                .is_ok_and(|wgpu| wgpu.can_reuse_film(film, backend, padded)),
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
            Backend::Wgpu => self.wgpu_mut()?.clear_film(backend, film),
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
            Backend::Wgpu => self.wgpu_mut()?.activate_all(backend, film, active_host),
            Backend::Auto => Err(film_mismatch()),
        }
    }

    fn upload_and_set_params(
        &mut self,
        backend: Backend,
        prepared: &Arc<PreparedScene>,
        env: Option<&sl_studio::environment::Map>,
        world_render: bool,
        params: &[f32; P_COUNT],
        stats: &mut PreparationStats,
    ) -> Result<(), RenderError> {
        match backend {
            Backend::Wgpu => self.wgpu_mut()?.upload_and_set_params(
                backend,
                prepared,
                env,
                world_render,
                params,
                stats,
            ),
            Backend::Cuda => {
                // The closure keeps every early return inside it, so the staging buffer is
                // always put back (its capacity is reused by the next palette upload).
                let mut staging = std::mem::take(&mut self.lut_staging);
                let res = (|| -> Result<(), RenderError> {
                    let cuda = self.cuda_mut()?;
                    cuda.upload_prepared(prepared, env, world_render, &mut staging, stats)?;
                    cuda.module
                        .set_params(&cuda.stream, params)
                        .map_err(|e| device("CUDA parameter upload", e))
                })();
                self.lut_staging = staging;
                res
            }
            Backend::Auto => Err(unresolved_auto()),
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
        match backend {
            Backend::Wgpu => self
                .wgpu_mut()?
                .execute_step(backend, target, trace, step, display),
            Backend::Cuda => {
                let scene = trace.scene;
                let prepared = trace.prepared;
                let force_mixed_world = trace.force_mixed_world;
                let spp = step.spp;
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
                    if tiles > 0 {
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
                }
                if !display(target) {
                    return Ok(StepOutcome::CompletedNoDisplay);
                }
                Ok(StepOutcome::NeedsDisplay { decided_due: None })
            }
            Backend::Auto => Err(unresolved_auto()),
        }
    }

    fn tally(&mut self, backend: Backend, film: &mut BackendFilm) -> Result<f32, RenderError> {
        match backend {
            Backend::Wgpu => self.wgpu_mut()?.tally(backend, film),
            Backend::Cuda => {
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
            Backend::Auto => Err(unresolved_auto()),
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
        match backend {
            Backend::Wgpu => self
                .wgpu_mut()?
                .tonemap_display_fast(backend, film, width, height, light, pixels),
            Backend::Cuda => Ok(false),
            Backend::Auto => Err(unresolved_auto()),
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
        match backend {
            Backend::Wgpu => self
                .wgpu_mut()?
                .tonemap_raw(backend, film, width, height, raw),
            Backend::Cuda => {
                let pixels = width.checked_mul(height).ok_or_else(|| {
                    RenderError::Scene("Render target dimensions overflow".into())
                })?;
                if pixels == 0 {
                    raw.clear();
                    return Ok(());
                }
                raw.resize(pixels, [0.0; 4]);
                let ct = film
                    .downcast_mut::<CudaTarget>()
                    .ok_or_else(film_mismatch)?;
                let cuda = self.cuda()?;
                if ct.out.len() != pixels {
                    ct.out = zeroed(&cuda.stream, pixels, "output")?;
                }
                let n = u32::try_from(pixels).map_err(|_| {
                    RenderError::Scene("Render target pixel count exceeds u32".into())
                })?;
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
            Backend::Auto => Err(unresolved_auto()),
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
        if film.downcast_ref::<WgpuTarget>().is_some() {
            self.wgpu()?
                .read_film(film, plane, dims, normalize, tiled, output)
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
        env: Option<&sl_studio::environment::Map>,
        params: &[f32; P_COUNT],
        (ndc, family): ([f32; 2], Option<u32>),
        stats: &mut PreparationStats,
    ) -> Result<Option<(usize, f32)>, RenderError> {
        match backend {
            Backend::Wgpu => {
                self.wgpu_mut()?
                    .pick(backend, prepared, env, params, (ndc, family), stats)
            }
            Backend::Cuda => {
                let mut staging = std::mem::take(&mut self.lut_staging);
                let res = (|| -> Result<Option<(usize, f32)>, RenderError> {
                    let cuda = self.cuda_mut()?;
                    cuda.upload_prepared(prepared, env, true, &mut staging, stats)?;
                    let CudaEngine {
                        ref stream,
                        ref module,
                        ref objects,
                        ref lights,
                        ref mut pick_query,
                        ref mut pick_out,
                        ..
                    } = *cuda;
                    module
                        .set_params(stream, params)
                        .map_err(|e| device("CUDA parameter upload", e))?;
                    pick_query
                        .copy_from_host(stream, &[[ndc[0], ndc[1], 0.0, 0.0]])
                        .map_err(|e| device("pick upload", e))?;
                    pick_out
                        .zero_async(stream)
                        .map_err(|e| device("clear pick result", e))?;
                    macro_rules! pick_entry {
                        ($prepare:ident, $run:ident) => {{
                            let launch = module
                                .$prepare(LaunchConfig1D::new(1, 32, 0))
                                .map_err(|e| device("prepare pick", e))?;
                            module
                                .$run(stream, &launch, objects, lights, &*pick_query, pick_out)
                                .map_err(|e| device("pick", e))?;
                        }};
                    }
                    fractal_kernels::dispatch_pick!(
                        pick_entry,
                        family,
                        return Err(RenderError::Scene("unsupported CUDA pick entry".into()))
                    );
                    let mut hit = [[0.0f32; 4]];
                    pick_out
                        .copy_to_host(stream, &mut hit)
                        .map_err(|e| device("pick readback", e))?;
                    let [[object, _, depth, _]] = hit;
                    Ok((object >= 1.0).then(|| (object as usize - 1, depth)))
                })();
                self.lut_staging = staging;
                res
            }
            Backend::Auto => Err(unresolved_auto()),
        }
    }

    fn finish(&self, backend: Backend) -> Result<(), RenderError> {
        match backend {
            Backend::Wgpu => self.wgpu()?.finish(backend),
            Backend::Cuda => self
                .cuda()?
                .stream
                .synchronize()
                .map_err(|e| device("CUDA stream synchronize", e)),
            Backend::Auto => Err(unresolved_auto()),
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
        validate_size(width, height, 1)?;
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

#[cfg(test)]
mod hook_tests {
    /// plan1 S2/S3/S4: an unknown family code is a miss on both WGSL paths (as in CUDA), the
    /// Mandelbulb has its own case instead of the default, and no unused constants are emitted.
    #[test]
    fn unknown_family_is_a_miss_on_both_wgsl_paths() {
        const MISS: &str = "return vec2<f32>(F32_MAX, TRAP_START);";
        let (_, dynamic) = super::warpbro_shader_hook(None);
        assert!(dynamic.contains(&format!("default: {{ {MISS} }}")));
        assert!(dynamic.contains(&format!("case {}u:", fractal_core::params::FAMILY_BULB)));
        assert!(!dynamic.contains("const FAMILY_"));
        let unknown = crate::formulas::FORMULAS.len() as u32;
        let (_, specialized) = super::warpbro_shader_hook(Some(unknown));
        assert!(specialized.contains(MISS));
    }
}
