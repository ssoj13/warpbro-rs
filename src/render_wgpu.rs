//! WGPU compute backend for WarpBro, sharing `gpu_info::shared_device()` and the 8×4 tiled
//! target contract of `render.rs`.

use std::future::Future;
use std::pin::pin;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};
use std::task::{Context, Poll, Waker};

use crate::formulas::{
    FAMILY_AMAZING_SURFACE, FAMILY_ASTROLABE, FAMILY_ATTRACTOR_TUBE, FAMILY_BRISTORBROT,
    FAMILY_BURNING_SHIP, FAMILY_CRYSTAL_GEODE, FAMILY_DUPIN_CYCLIDE, FAMILY_GYROID_TPMS,
    FAMILY_HARMONIC_BULB, FAMILY_HOPF_FIBRATION, FAMILY_KALEIDO_TWIST, FAMILY_LIDINOID_TPMS,
    FAMILY_MANDELBAR, FAMILY_MOBIUS_IFS, FAMILY_NAUTILUS, FAMILY_PHYLLOTAXIS,
    FAMILY_POLYHEDRAL_KIFS, FAMILY_SUPERFORMULA, FAMILY_TORUS_KNOT, FAMILY_VORONOI_FOAM,
    ITERATING_MASK, SIGNED_MASK,
};
use crate::palette::build_lut;
use crate::params::*;
use crate::render::RenderError;
use crate::render_prep::{
    BLOCK, LightResidentKey, LutResidentKey, ObjectResidentKey, Plane, PreparedScene, Resident,
    TALLY_THREADS, WorldMaterial, padded_len, untile_into,
};
use crate::scene::Scene;

const UNIFORM_FLOATS: usize = 256;

/// Formula families, each with its own specialized trace entries.
const FAMILIES: usize = crate::scene::Formula::COUNT as usize;

/// How a trace entry point shades its hits: `trace_pixel`'s material spec and Direct flag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Shade {
    /// Each world object's own material model (`MAT_DISPATCH`): objects that mix models.
    Mixed,
    Fast,
    Full,
    /// The Direct display (`trace_direct`), on the full material inputs.
    Direct,
}
impl Shade {
    const ALL: [Self; 4] = [Self::Mixed, Self::Fast, Self::Full, Self::Direct];

    fn name(self) -> &'static str {
        match self {
            Self::Mixed => "mixed",
            Self::Fast => "fast",
            Self::Full => "full",
            Self::Direct => "direct",
        }
    }

    /// `trace_pixel`'s `mat_spec, is_direct` arguments.
    fn args(self) -> &'static str {
        match self {
            Self::Mixed => "MAT_DISPATCH, false",
            Self::Fast => "MAT_FAST, false",
            Self::Full => "MAT_FULL, false",
            Self::Direct => "MAT_FULL, true",
        }
    }
}

/// Slots of the trace pipeline cache ([`TraceEntry::slot`]): scope x shade x (family or dynamic).
const TRACE_SLOTS: usize = 2 * Shade::ALL.len() * (FAMILIES + 1);

/// One trace entry point: the one table [`shader_source`] generates the WGSL entries from and
/// the pipeline cache ([`WgpuShared::trace_pipe`]) is keyed by, so a name or a slot cannot
/// drift from its entry. `family: None` reads each object's `P_FAMILY` (the WGSL
/// `FAMILY_DYNAMIC` sentinel); `Some(code)` inlines that family's estimate. Family codes and
/// the dynamic case never share a value (plan1 2.8: fractal-core's `FAMILY_WORLD` equals the
/// extended family code 8, so a sentinel inside the family range would misroute it).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TraceEntry {
    pub(crate) world: bool,
    pub(crate) shade: Shade,
    pub(crate) family: Option<u32>,
}

impl TraceEntry {
    /// Every entry [`shader_source`] emits. World scenes: each shade dynamic, and except Mixed
    /// also specialized to each family (`PreparedScene::homogeneous_family`). Single scenes
    /// know their one family, so they are only specialized.
    pub(crate) fn all() -> impl Iterator<Item = Self> {
        let world = Shade::ALL.into_iter().flat_map(|shade| {
            let specialized = if shade == Shade::Mixed {
                0
            } else {
                FAMILIES as u32
            };
            std::iter::once(None)
                .chain((0..specialized).map(Some))
                .map(move |family| Self {
                    world: true,
                    shade,
                    family,
                })
        });
        let single = [Shade::Fast, Shade::Full, Shade::Direct]
            .into_iter()
            .flat_map(|shade| {
                (0..FAMILIES as u32).map(move |code| Self {
                    world: false,
                    shade,
                    family: Some(code),
                })
            });
        world.chain(single)
    }

    /// The WGSL entry point name: `{world|single}_{shade}` plus `_f{code}` when specialized.
    pub(crate) fn name(self) -> String {
        let scope = if self.world { "world" } else { "single" };
        match self.family {
            Some(code) => format!("{scope}_{}_f{code}", self.shade.name()),
            None => format!("{scope}_{}", self.shade.name()),
        }
    }

    /// The entry's cell in [`WgpuShared::trace`] (below [`TRACE_SLOTS`] for a family code
    /// below `Formula::COUNT`).
    fn slot(self) -> usize {
        let family = self.family.map_or(FAMILIES, |code| code as usize);
        (usize::from(self.world) * Shade::ALL.len() + self.shade as usize) * (FAMILIES + 1) + family
    }

    /// The WGSL source of the entry: 128 invocations per workgroup over the 2D dispatch of
    /// [`WgpuEngine::groups`], flattened by `linear_id`.
    fn wgsl(self) -> String {
        let family = self
            .family
            .map_or_else(|| "FAMILY_DYNAMIC".to_string(), |code| format!("{code}u"));
        format!(
            "@compute @workgroup_size(128, 1, 1) fn {}(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{ trace_pixel(linear_id(gid, nwg), {}, {family}, {}); }}\n",
            self.name(),
            self.world,
            self.shade.args()
        )
    }
}

/// The trace entry of one launch. Direct wins over the material; a world takes its objects'
/// common material model (`force_mixed_world`: the per-object dispatcher, a test oracle) and,
/// unless Mixed, the entry specialized to its objects' common family; a single scene always
/// takes its family's entry.
fn trace_entry(
    scene: &Scene,
    prepared: &PreparedScene,
    full_material: bool,
    force_mixed_world: bool,
) -> TraceEntry {
    let shade = if scene.direct.is_some() {
        Shade::Direct
    } else if !scene.world_render {
        if full_material {
            Shade::Full
        } else {
            Shade::Fast
        }
    } else if force_mixed_world {
        Shade::Mixed
    } else {
        match prepared.world_material {
            WorldMaterial::Fast => Shade::Fast,
            WorldMaterial::Full => Shade::Full,
            WorldMaterial::Mixed => Shade::Mixed,
        }
    };
    let family = if !scene.world_render {
        Some(scene.formula.code())
    } else if force_mixed_world || shade == Shade::Mixed {
        None
    } else {
        prepared.homogeneous_family
    };
    TraceEntry {
        world: scene.world_render,
        shade,
        family,
    }
}

/// Assemble the complete WGSL source: `standard_surface_bsdf::wgsl_source()`, the `P_*` / `O_*` /
/// `M_*` constants from `fractal_core::params`, the family registry's codes and masks,
/// `render_wgpu.wgsl`, and the trace entry points of [`TraceEntry::all`].
pub fn shader_source() -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(65536);
    s.push_str(standard_surface_bsdf::wgsl_source());
    s.push('\n');
    macro_rules! consts {
        ($($name:ident),+ $(,)?) => {{
            $(
                let _ = writeln!(s, "const {}: u32 = {}u;", stringify!($name), $name as u32);
            )+
        }};
    }
    consts!(
        P_WIDTH,
        P_HEIGHT,
        P_SAMPLE_BEGIN,
        P_SPP,
        P_SEED,
        P_CAM_ORIGIN,
        P_CAM_RIGHT,
        P_CAM_UP,
        P_CAM_FORWARD,
        P_HALF_W,
        P_HALF_H,
        P_APERTURE,
        P_FOCUS_DISTANCE,
        P_MAX_STEPS,
        P_MAX_DISTANCE,
        P_FOOTPRINT,
        P_SAMPLE_CONE,
        P_MAX_BOUNCES,
        P_SECONDARY_STEPS,
        P_SECONDARY_EPS,
        P_GLASS_PROBES,
        P_STEP_FACTOR,
        P_MATERIAL_MODEL,
        P_FAMILY,
        P_ITERATIONS,
        P_BAILOUT,
        P_JULIA,
        P_JULIA_C,
        P_ORBIT_OFFSET,
        P_INVERT_STRENGTH,
        P_INVERT_CENTER,
        P_INVERT_RADIUS,
        P_OBJ_AXES,
        P_OBJ_OFFSET,
        P_OBJ_SCALE,
        P_BOUND_RADIUS,
        P_ITER_ROT,
        P_ITER_ROTATE,
        P_BULB_POWER,
        P_BULB_THETA_POWER,
        P_BULB_PHI_POWER,
        P_BULB_THETA_PHASE,
        P_BULB_PHI_PHASE,
        P_BULB_GROWTH,
        P_BULB_FAST8,
        P_BOX_SCALE,
        P_BOX_FOLD,
        P_BOX_MIN_R2,
        P_BOX_FIXED_R2,
        P_QUAT_ROWS,
        P_QUAT_OFFSET,
        P_QUAT_C,
        P_KIFS_KIND,
        P_KIFS_SCALE,
        P_KIFS_SHIFT,
        P_KIFS_FOLD_HEIGHT,
        P_KIFS_BOUND,
        P_KLEIN_A,
        P_KLEIN_B,
        P_KLEIN_SKEW,
        P_KLEIN_LINE_AMP,
        P_KLEIN_LINE_RATE,
        P_PK_BOX,
        P_PK_SIZE,
        P_PK_C,
        P_PK_OFFSET,
        P_PK_THICKNESS,
        P_APOLLO_SCALE,
        P_HYBRID_STEPS,
        P_HYBRID_COUNT,
        P_COLOR_MODE,
        P_TRAP_POINT,
        P_TRAP_NORMAL,
        P_TRAP_SCALE,
        P_LIGHT_DIR,
        P_LIGHT_COLOR,
        P_LIGHT_INTENSITY,
        P_SUN_ONE_MINUS_COS,
        P_SUN_CONE_PDF,
        P_SKY_INTENSITY,
        P_SKY_HORIZON,
        P_SKY_ZENITH,
        P_BACKGROUND,
        P_CAMERA_SUN,
        P_ENV_WIDTH,
        P_ENV_HEIGHT,
        P_ENV_LEVELS,
        P_ENV_MEAN,
        P_ENV_INTENSITY,
        P_ENV_COS,
        P_ENV_SIN,
        P_ENV_SUN_SELECT,
        P_BASE,
        P_BASE_TINT,
        P_DIFFUSE_ROUGHNESS,
        P_METALNESS,
        P_SPECULAR,
        P_SPECULAR_COLOR,
        P_SPECULAR_ROUGHNESS,
        P_SPECULAR_IOR,
        P_SPECULAR_ANISOTROPY,
        P_SPECULAR_ROTATION,
        P_EXPOSURE,
        P_SATURATION,
        P_TONEMAP,
        P_CLIP_CENTER,
        P_CLIP_RADIUS,
        P_TRANSMISSION,
        P_TRANSMISSION_COLOR,
        P_TRANSMISSION_EXTRA_ROUGHNESS,
        P_TRANSMISSION_DEPTH,
        P_SHEEN,
        P_SHEEN_COLOR,
        P_SHEEN_ROUGHNESS,
        P_COAT,
        P_COAT_COLOR,
        P_COAT_ROUGHNESS,
        P_COAT_IOR,
        P_COAT_AFFECT_COLOR,
        P_COAT_AFFECT_ROUGHNESS,
        P_THIN_FILM_THICKNESS,
        P_THIN_FILM_IOR,
        P_EMISSION,
        P_EMISSION_COLOR,
        P_COLOR_SOURCE,
        P_BASE_COLOR,
        P_FACING_EXPONENT,
        P_FACING_COLOR,
        P_FACING_ROUGHNESS,
        P_FACING_METALLIC,
        P_ADAPT_THRESHOLD,
        P_ADAPT_MIN,
        P_LUMA,
        P_WORLD,
        P_OBJECT_COUNT,
        P_LIGHT_COUNT,
        P_DIRECT,
        P_DIRECT_GRID,
        P_SHADOW_STRENGTH,
        P_SHADOW_STEPS,
        P_AO_STRENGTH,
        P_AO_STEPS,
        P_AO_RADIUS,
        P_LIGHT_HALF_ANGLE,
        P_OFX,
        P_TILE_X,
        P_TILE_Y,
        P_TILE_WIDTH,
        P_TILE_HEIGHT,
        P_OFX_NO_CLIP,
        P_OFX_THIN_FILM_ENERGY,
        P_OFX_SEED_HIGH,
        P_OFX_SKY_ALPHA,
        P_COUNT,
        O_INVERSE,
        O_DISTANCE_SCALE,
        O_CLIP_RADIUS,
        O_TANGENT,
        O_MASK_COUNT,
        O_MASKS,
        OBJECT_STRIDE,
        LIGHT_STRIDE,
        M_INVERSE,
        M_SCALE,
        M_SHAPE,
        M_INVERT,
        M_SIZE,
        M_DENSITY,
        M_SEED,
        M_ARRAY,
        M_ARRAY_COUNT,
        M_ARRAY_OFFSET,
        M_ARRAY_RADIUS,
        M_ARRAY_AXIS,
        M_ARRAY_SWEEP,
        M_ARRAY_ALIGN,
        MASK_STRIDE,
        MAX_MASKS,
        MASK_SPHERE,
        MASK_BOX,
        MASK_CELLS,
        ARRAY_LINEAR,
        ARRAY_RADIAL,
        ARRAY_MAX,
        FAMILY_BULB,
        FAMILY_BOX,
        FAMILY_QUAT,
        FAMILY_KIFS,
        FAMILY_KLEINIAN,
        FAMILY_PSEUDO_KLEINIAN,
        FAMILY_APOLLONIAN,
        FAMILY_HYBRID,
        FAMILY_AMAZING_SURFACE,
        FAMILY_BRISTORBROT,
        FAMILY_BURNING_SHIP,
        FAMILY_KALEIDO_TWIST,
        FAMILY_POLYHEDRAL_KIFS,
        FAMILY_HARMONIC_BULB,
        FAMILY_MANDELBAR,
        FAMILY_MOBIUS_IFS,
        FAMILY_GYROID_TPMS,
        FAMILY_LIDINOID_TPMS,
        FAMILY_ATTRACTOR_TUBE,
        FAMILY_TORUS_KNOT,
        FAMILY_HOPF_FIBRATION,
        FAMILY_DUPIN_CYCLIDE,
        FAMILY_SUPERFORMULA,
        FAMILY_PHYLLOTAXIS,
        FAMILY_NAUTILUS,
        FAMILY_VORONOI_FOAM,
        FAMILY_CRYSTAL_GEODE,
        FAMILY_ASTROLABE,
        SIGNED_MASK,
        ITERATING_MASK,
        PALETTE_SAMPLES,
        TALLY_THREADS,
    );
    let c0 = crate::color::to_709([1.0, 0.0, 0.0]);
    let c1 = crate::color::to_709([0.0, 1.0, 0.0]);
    let c2 = crate::color::to_709([0.0, 0.0, 1.0]);
    let _ = writeln!(
        s,
        "const TO_709_ROW0: vec3<f32> = vec3<f32>({:.9}, {:.9}, {:.9});",
        c0[0], c1[0], c2[0]
    );
    let _ = writeln!(
        s,
        "const TO_709_ROW1: vec3<f32> = vec3<f32>({:.9}, {:.9}, {:.9});",
        c0[1], c1[1], c2[1]
    );
    let _ = writeln!(
        s,
        "const TO_709_ROW2: vec3<f32> = vec3<f32>({:.9}, {:.9}, {:.9});",
        c0[2], c1[2], c2[2]
    );
    s.push_str(include_str!("formulas.wgsl"));
    s.push_str(include_str!("render_wgpu.wgsl"));
    for entry in TraceEntry::all() {
        s.push_str(&entry.wgsl());
    }
    s
}

fn poll_ready<F: Future>(fut: F) -> Option<F::Output> {
    let mut fut = pin!(fut);
    let mut cx = Context::from_waker(Waker::noop());
    match fut.as_mut().poll(&mut cx) {
        Poll::Ready(val) => Some(val),
        Poll::Pending => None,
    }
}

/// Execute `op` inside `OutOfMemory` and `Validation` error scopes on `device`. An error outside
/// any scope reaches wgpu's default handler, which panics and takes the render worker with it,
/// so every resource creation, write, submission and lazy pipeline compile of this backend runs
/// in here. wgpu-core validates on the CPU: a native scope holds its error when it is popped,
/// without waiting for the device.
pub fn with_error_scopes<T>(device: &wgpu::Device, op: impl FnOnce() -> T) -> Result<T, String> {
    let oom_scope = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let val_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let output = op();
    let validation = scope_error(val_scope.pop());
    let oom = scope_error(oom_scope.pop());
    if let Some(err) = validation {
        return Err(format!("WGPU validation error: {err}"));
    }
    if let Some(err) = oom {
        return Err(format!("WGPU out-of-memory error: {err}"));
    }
    Ok(output)
}

/// The error a popped scope caught. A pop that has not resolved (never on native wgpu-core) is
/// reported as an error instead of being taken for success.
fn scope_error(pop: impl Future<Output = Option<wgpu::Error>>) -> Option<String> {
    match poll_ready(pop) {
        Some(error) => error.map(|e| e.to_string()),
        None => Some("the error scope did not resolve".into()),
    }
}

struct WgpuShared {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_name: String,
    /// The device's limits, read once: dispatch sizes (`WgpuEngine::groups`) and film buffer
    /// sizes (`WgpuEngine::create_target`) are checked against them.
    limits: wgpu::Limits,
    module: wgpu::ShaderModule,
    trace_bgl: wgpu::BindGroupLayout,
    post_bgl: wgpu::BindGroupLayout,
    pick_bgl: wgpu::BindGroupLayout,
    trace_layout: wgpu::PipelineLayout,
    /// The trace pipelines, compiled on first use: one cell per [`TraceEntry::slot`].
    trace: Box<[OnceLock<wgpu::ComputePipeline>]>,
    tally: wgpu::ComputePipeline,
    tally_sum: wgpu::ComputePipeline,
    adapt: wgpu::ComputePipeline,
    tonemap: wgpu::ComputePipeline,
    tonemap_display: wgpu::ComputePipeline,
    pick: wgpu::ComputePipeline,
}

impl WgpuShared {
    /// The trace pipeline of `entry`, compiled on first use inside error scopes: a WGSL entry
    /// that fails to compile is an `Err` of the launch, not a panic.
    fn trace_pipe(&self, entry: TraceEntry) -> Result<&wgpu::ComputePipeline, RenderError> {
        let cell = self.trace.get(entry.slot()).ok_or_else(|| {
            RenderError::Device(format!("{} has no trace pipeline slot", entry.name()))
        })?;
        if let Some(pipe) = cell.get() {
            return Ok(pipe);
        }
        let name = entry.name();
        let pipe = with_error_scopes(&self.device, || {
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(&name),
                    layout: Some(&self.trace_layout),
                    module: &self.module,
                    entry_point: Some(&name),
                    compilation_options: Default::default(),
                    cache: None,
                })
        })
        .map_err(|e| RenderError::Device(format!("{name}: {e}")))?;
        Ok(cell.get_or_init(|| pipe))
    }

    /// The two `tally` passes after a trace or `adapt` dispatch of `pass` (post bind group set):
    /// `TALLY_THREADS` invocations add strided partial sums per workgroup, then one workgroup
    /// adds the partials into the 16-byte summary (CUDA `tally` + its host sum).
    fn record_tally(&self, pass: &mut wgpu::ComputePass<'_>) {
        pass.set_pipeline(&self.tally);
        pass.dispatch_workgroups(TALLY_THREADS as u32 / BLOCK, 1, 1);
        pass.set_pipeline(&self.tally_sum);
        pass.dispatch_workgroups(1, 1, 1);
    }
}

static SHARED_WGPU: OnceLock<Result<Arc<WgpuShared>, String>> = OnceLock::new();

fn shared_wgpu() -> Result<Arc<WgpuShared>, String> {
    SHARED_WGPU
        .get_or_init(|| {
            let shared = gpu_info::shared_device()
                .ok_or_else(|| "WGPU shared device unavailable".to_string())?;
            let info = shared.adapter.get_info();
            let lower = info.name.to_ascii_lowercase();
            if info.device_type == wgpu::DeviceType::Cpu
                || lower.contains("llvmpipe")
                || lower.contains("lavapipe")
                || lower.contains("swiftshader")
                || lower.contains("microsoft basic render driver")
            {
                return Err(format!(
                    "WGPU software adapter '{}' is not a hardware GPU",
                    info.name
                ));
            }
            let device = shared.device.clone();
            let queue = shared.queue.clone();
            let adapter_name = format!("{} (WGPU)", info.name);
            with_error_scopes(&shared.device, || {
                let source = shader_source();
                let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("warpbro_render_wgpu"),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                });

                let bgl_entry = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                };
                let uniform_entry = wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                };

                let trace_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("warpbro_trace_bgl"),
                    entries: &[
                        uniform_entry,
                        bgl_entry(1, true),
                        bgl_entry(2, true),
                        bgl_entry(3, true),
                        bgl_entry(4, true),
                        bgl_entry(5, false),
                        bgl_entry(6, false),
                        bgl_entry(7, false),
                        bgl_entry(8, false),
                    ],
                });

                let post_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("warpbro_post_bgl"),
                    entries: &[
                        uniform_entry,
                        bgl_entry(5, false),
                        bgl_entry(8, false),
                        bgl_entry(9, false),
                        bgl_entry(10, false),
                        bgl_entry(11, false),
                        bgl_entry(12, false),
                        bgl_entry(13, false),
                    ],
                });

                let pick_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("warpbro_pick_bgl"),
                    entries: &[uniform_entry, bgl_entry(2, true), bgl_entry(10, false)],
                });

                let trace_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("warpbro_trace_layout"),
                    bind_group_layouts: &[Some(&trace_bgl)],
                    immediate_size: 0,
                });
                let post_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("warpbro_post_layout"),
                    bind_group_layouts: &[Some(&post_bgl)],
                    immediate_size: 0,
                });
                let pick_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("warpbro_pick_layout"),
                    bind_group_layouts: &[Some(&pick_bgl)],
                    immediate_size: 0,
                });

                let make_pipe = |layout: &wgpu::PipelineLayout, entry: &str| {
                    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(entry),
                        layout: Some(layout),
                        module: &module,
                        entry_point: Some(entry),
                        compilation_options: Default::default(),
                        cache: None,
                    })
                };

                Arc::new(WgpuShared {
                    trace: (0..TRACE_SLOTS).map(|_| OnceLock::new()).collect(),
                    tally: make_pipe(&post_layout, "tally"),
                    tally_sum: make_pipe(&post_layout, "tally_sum"),
                    adapt: make_pipe(&post_layout, "adapt"),
                    tonemap: make_pipe(&post_layout, "tonemap"),
                    tonemap_display: make_pipe(&post_layout, "tonemap_display"),
                    pick: make_pipe(&pick_layout, "pick"),
                    module,
                    trace_layout,
                    limits: device.limits(),
                    device,
                    queue,
                    adapter_name,
                    trace_bgl,
                    post_bgl,
                    pick_bgl,
                })
            })
        })
        .clone()
}

/// One WGPU film: the tiled accumulation planes ([`padded_len`] samples), the display outputs
/// (`width * height`), and one staging buffer that holds the largest readback.
pub struct WgpuTarget {
    padded: usize,
    accum: wgpu::Buffer,
    albedo: wgpu::Buffer,
    normal: wgpu::Buffer,
    stats: wgpu::Buffer,
    active: wgpu::Buffer,
    out: wgpu::Buffer,
    sdr: wgpu::Buffer,
    /// The `tally` pass's per-workgroup partial sums (`TALLY_THREADS / BLOCK` vec4s).
    tally_parts: wgpu::Buffer,
    summary: wgpu::Buffer,
    staging: wgpu::Buffer,
    bind_groups: Mutex<Option<(u64, wgpu::BindGroup, wgpu::BindGroup)>>,
}

impl WgpuTarget {
    /// Samples of the tiled film ([`padded_len`] of its size).
    pub fn padded(&self) -> usize {
        self.padded
    }

    fn plane(&self, plane: Plane) -> &wgpu::Buffer {
        match plane {
            Plane::Accum => &self.accum,
            Plane::Albedo => &self.albedo,
            Plane::Normal => &self.normal,
        }
    }
}

pub struct WgpuEngine {
    shared: Arc<WgpuShared>,
    params_buf: wgpu::Buffer,
    lut_buf: wgpu::Buffer,
    lut_len: usize,
    objects_buf: wgpu::Buffer,
    objects_len: usize,
    lights_buf: wgpu::Buffer,
    lights_len: usize,
    buf_generation: u64,
    pick_aux: wgpu::Buffer,
    pick_staging: wgpu::Buffer,
    lut_resident: Resident<LutResidentKey>,
    object_resident: Resident<ObjectResidentKey>,
    light_resident: Resident<LightResidentKey>,
    lut_staging: Vec<[f32; 4]>,
}

/// The compute pass of one submission ([`WgpuEngine::run`]).
fn compute<'e>(encoder: &'e mut wgpu::CommandEncoder, label: &str) -> wgpu::ComputePass<'e> {
    encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(label),
        timestamp_writes: None,
    })
}

/// `len` bytes at `at` of a mapped readback; a short mapping is an error, never a panic.
fn bytes_at(bytes: &[u8], at: usize, len: usize) -> Result<&[u8], RenderError> {
    at.checked_add(len)
        .and_then(|end| bytes.get(at..end))
        .ok_or_else(|| RenderError::Device("WGPU readback is shorter than its copies".into()))
}

/// The `[unresolved, samples, active tiles, 0]` summary the `tally` entry writes, at the front of
/// a readback.
fn summary(bytes: &[u8]) -> Result<[f32; 4], RenderError> {
    Ok(bytemuck::pod_read_unaligned(bytes_at(bytes, 0, 16)?))
}

/// The share of samples with an unresolved march ([`crate::render::Target::unresolved`]).
fn unresolved_share(unresolved: f32, samples: f32) -> f32 {
    if samples > 0.0 {
        unresolved / samples
    } else {
        0.0
    }
}

/// Grow `buf` (`len` elements of `T`, at least `min`) when `data` changes its size and write
/// `data` into it: the one storage upload of the palette / environment LUT, the world objects and
/// the world lights. A reallocation bumps `generation`, so cached bind groups are rebuilt; a
/// failed one leaves `len` unchanged, so the next upload retries it.
fn upload_storage<T: bytemuck::Pod>(
    shared: &WgpuShared,
    label: &'static str,
    buf: &mut wgpu::Buffer,
    len: &mut usize,
    generation: &mut u64,
    min: usize,
    data: &[T],
) -> Result<(), String> {
    let needed = data.len().max(min);
    if *len != needed {
        *buf = with_error_scopes(&shared.device, || {
            shared.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (needed * std::mem::size_of::<T>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        })?;
        *len = needed;
        *generation = generation.wrapping_add(1);
    }
    if !data.is_empty() {
        with_error_scopes(&shared.device, || {
            shared
                .queue
                .write_buffer(buf, 0, bytemuck::cast_slice(data))
        })?;
    }
    Ok(())
}

impl WgpuEngine {
    pub fn new() -> Result<Self, String> {
        let shared = shared_wgpu()?;
        let device = shared.device.clone();
        with_error_scopes(&device, || {
            let storage_ro = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
            let storage_rw = wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC;
            let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("warpbro_wgpu_params"),
                size: (UNIFORM_FLOATS * std::mem::size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let lut_len = PALETTE_SAMPLES + 1;
            let lut_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("warpbro_wgpu_lut"),
                size: (lut_len * std::mem::size_of::<[f32; 4]>()) as u64,
                usage: storage_ro,
                mapped_at_creation: false,
            });
            let objects_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("warpbro_wgpu_objects"),
                size: 16,
                usage: storage_ro,
                mapped_at_creation: false,
            });
            let lights_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("warpbro_wgpu_lights"),
                size: 16,
                usage: storage_ro,
                mapped_at_creation: false,
            });
            let pick_aux = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("warpbro_wgpu_pick_aux"),
                size: 16,
                usage: storage_rw,
                mapped_at_creation: false,
            });
            let pick_staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("warpbro_wgpu_pick_staging"),
                size: 16,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let engine = Self {
                shared,
                params_buf,
                lut_buf,
                lut_len,
                objects_buf,
                objects_len: 4,
                lights_buf,
                lights_len: 4,
                buf_generation: 1,
                pick_aux,
                pick_staging,
                lut_resident: Default::default(),
                object_resident: Default::default(),
                light_resident: Default::default(),
                lut_staging: Vec::new(),
            };
            log::info!("WGPU compute ready on {}", engine.name());
            engine
        })
    }

    pub fn name(&self) -> &str {
        &self.shared.adapter_name
    }

    /// Allocate a zeroed film for `width` x `height`. Every buffer is checked against the
    /// device's storage-binding and buffer-size limits first, so an oversized export fails with
    /// a typed error instead of a wgpu panic.
    pub fn create_target(&self, width: usize, height: usize) -> Result<WgpuTarget, RenderError> {
        let padded = padded_len(width, height);
        let tiles = padded / 32;
        let pixels = (width * height).max(1);
        let vec4_bytes = std::mem::size_of::<[f32; 4]>() as u64;
        let film_bytes = padded.max(1) as u64 * vec4_bytes;
        let out_bytes = pixels as u64 * vec4_bytes;
        let sdr_bytes = (pixels * std::mem::size_of::<u32>()) as u64;
        // The largest readback: one tiled plane, or the fused summary + display light + codes.
        let staging_bytes = film_bytes.max(16 + out_bytes + sdr_bytes);
        let limits = &self.shared.limits;
        let storage = limits
            .max_storage_buffer_binding_size
            .min(limits.max_buffer_size);
        for (label, bytes, limit) in [
            ("film plane", film_bytes, storage),
            ("display light", out_bytes, storage),
            ("display codes", sdr_bytes, storage),
            ("readback staging", staging_bytes, limits.max_buffer_size),
        ] {
            if bytes > limit {
                return Err(RenderError::Device(format!(
                    "{width}x{height} needs a WGPU {label} buffer of {bytes} bytes; this device allows {limit}"
                )));
            }
        }
        let device = &self.shared.device;
        let rw = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let make = |label: &str, size: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: rw,
                mapped_at_creation: false,
            })
        };
        with_error_scopes(device, || {
            let active = make(
                "warpbro_wgpu_active",
                (tiles.max(1) * std::mem::size_of::<u32>()) as u64,
            );
            let ones = vec![1u32; tiles.max(1)];
            self.shared
                .queue
                .write_buffer(&active, 0, bytemuck::cast_slice(&ones));
            WgpuTarget {
                padded,
                accum: make("warpbro_wgpu_accum", film_bytes),
                albedo: make("warpbro_wgpu_albedo", film_bytes),
                normal: make("warpbro_wgpu_normal", film_bytes),
                stats: make("warpbro_wgpu_stats", film_bytes),
                active,
                out: make("warpbro_wgpu_out", out_bytes),
                sdr: make("warpbro_wgpu_sdr", sdr_bytes),
                tally_parts: make(
                    "warpbro_wgpu_tally_parts",
                    (TALLY_THREADS as u64 / u64::from(BLOCK)) * vec4_bytes,
                ),
                summary: make("warpbro_wgpu_summary", vec4_bytes),
                staging: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("warpbro_wgpu_staging"),
                    size: staging_bytes,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                bind_groups: Mutex::new(None),
            }
        })
        .map_err(|e| RenderError::Device(format!("{width}x{height} WGPU film: {e}")))
    }

    /// Zero the accumulation planes for a new render.
    pub fn clear_target(&self, wt: &WgpuTarget) -> Result<(), RenderError> {
        self.run(
            "warpbro_wgpu_clear",
            |encoder| {
                encoder.clear_buffer(&wt.accum, 0, None);
                encoder.clear_buffer(&wt.albedo, 0, None);
                encoder.clear_buffer(&wt.normal, 0, None);
                encoder.clear_buffer(&wt.stats, 0, None);
            },
            &wt.staging,
            &[],
            |_| Ok(()),
        )
    }

    pub fn upload_active(&self, wt: &WgpuTarget, active_host: &[u32]) -> Result<(), RenderError> {
        with_error_scopes(&self.shared.device, || {
            self.shared
                .queue
                .write_buffer(&wt.active, 0, bytemuck::cast_slice(active_host))
        })
        .map_err(|e| RenderError::Device(format!("tile flags upload: {e}")))
    }

    pub fn set_params(&self, params: &[f32; P_COUNT]) -> Result<(), RenderError> {
        let mut padded = [0.0f32; UNIFORM_FLOATS];
        padded[..P_COUNT].copy_from_slice(params);
        with_error_scopes(&self.shared.device, || {
            self.shared
                .queue
                .write_buffer(&self.params_buf, 0, bytemuck::cast_slice(&padded))
        })
        .map_err(|e| RenderError::Device(format!("WGPU parameter upload: {e}")))
    }

    pub fn upload_prepared(
        &mut self,
        prepared: &Arc<PreparedScene>,
        env: Option<&crate::environment::Map>,
        world: bool,
        stats: &mut crate::render_prep::PreparationStats,
    ) -> Result<(), RenderError> {
        let shared = &self.shared;
        let table = &mut self.lut_staging;
        let (buf, len, generation) = (
            &mut self.lut_buf,
            &mut self.lut_len,
            &mut self.buf_generation,
        );
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
                upload_storage(shared, "warpbro_wgpu_lut", buf, len, generation, 1, table)
            })
            .map_err(|e| RenderError::Device(format!("palette / environment upload: {e}")))?;
        if changed {
            stats.lut_uploads += 1;
            stats.upload_bytes += (table.len() * std::mem::size_of::<[f32; 4]>()) as u64;
        }

        if world {
            let (buf, len, generation) = (
                &mut self.objects_buf,
                &mut self.objects_len,
                &mut self.buf_generation,
            );
            let data = &prepared.upload.objects;
            let changed = self
                .object_resident
                .update(ObjectResidentKey(prepared.clone()), || {
                    upload_storage(
                        shared,
                        "warpbro_wgpu_objects",
                        buf,
                        len,
                        generation,
                        4,
                        data,
                    )
                })
                .map_err(|e| RenderError::Device(format!("world objects upload: {e}")))?;
            if changed && !data.is_empty() {
                stats.object_uploads += 1;
                stats.upload_bytes += (data.len() * std::mem::size_of::<f32>()) as u64;
            }

            let (buf, len, generation) = (
                &mut self.lights_buf,
                &mut self.lights_len,
                &mut self.buf_generation,
            );
            let data = &prepared.upload.lights;
            let changed = self
                .light_resident
                .update(LightResidentKey(prepared.clone()), || {
                    upload_storage(shared, "warpbro_wgpu_lights", buf, len, generation, 4, data)
                })
                .map_err(|e| RenderError::Device(format!("world lights upload: {e}")))?;
            if changed && !data.is_empty() {
                stats.light_uploads += 1;
                stats.upload_bytes += (data.len() * std::mem::size_of::<f32>()) as u64;
            }
        }
        Ok(())
    }

    /// One submission of this backend, the only place that encodes, submits and reads back:
    /// `record` encodes the work, then each `(buffer, bytes)` of `reads` is copied into `staging`
    /// back to back, mapped once and handed to `read` as one byte slice. All of it runs inside
    /// error scopes, so a bad dispatch, binding or copy is an `Err`. Without `reads` nothing is
    /// waited for: the GPU runs on while the host prepares the next launch.
    fn run<R>(
        &self,
        label: &str,
        record: impl FnOnce(&mut wgpu::CommandEncoder),
        staging: &wgpu::Buffer,
        reads: &[(&wgpu::Buffer, u64)],
        read: impl FnOnce(&[u8]) -> Result<R, RenderError>,
    ) -> Result<R, RenderError> {
        let device = &self.shared.device;
        let mapped: u64 = reads.iter().map(|(_, bytes)| bytes).sum();
        with_error_scopes(device, || {
            let mut encoder = device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
            record(&mut encoder);
            let mut offset = 0;
            for (src, bytes) in reads {
                encoder.copy_buffer_to_buffer(src, 0, staging, offset, *bytes);
                offset += bytes;
            }
            self.shared.queue.submit(Some(encoder.finish()));
        })
        .map_err(|e| RenderError::Device(format!("{label}: {e}")))?;
        if reads.is_empty() {
            return read(&[]);
        }
        let slice = staging.slice(..mapped);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            if sender.send(result).is_err() {
                log::warn!("WGPU readback completed after its wait gave up");
            }
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| RenderError::Device(format!("{label}: device poll: {e}")))?;
        match receiver.try_recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(RenderError::Device(format!("{label}: readback map: {e}"))),
            Err(_) => {
                return Err(RenderError::Device(format!(
                    "{label}: readback did not complete"
                )));
            }
        }
        let output = match slice.get_mapped_range() {
            Ok(view) => read(&view),
            Err(e) => Err(RenderError::Device(format!("{label}: mapped range: {e}"))),
        };
        staging.unmap();
        output
    }

    /// Workgroups `(x, y)` of a 128-wide entry point over `threads` invocations. One dimension
    /// is capped by the device (`max_compute_workgroups_per_dimension`, 65,535 on DX12 and
    /// Vulkan), so a frame above ~8.4 MP (4096x2160) spills into `y`. The WGSL entries index
    /// `gid.x + gid.y * nwg.x * 128` and drop the invocations past `threads`.
    fn groups(&self, threads: usize) -> Result<(u32, u32), RenderError> {
        let max = u64::from(
            self.shared
                .limits
                .max_compute_workgroups_per_dimension
                .max(1),
        );
        let total = (threads as u64).div_ceil(u64::from(BLOCK)).max(1);
        let x = total.min(max);
        let y = total.div_ceil(x);
        if y > max {
            return Err(RenderError::Device(format!(
                "{threads} invocations exceed the device's {max} x {max} workgroups"
            )));
        }
        Ok((x as u32, y as u32))
    }

    /// Read one tiled film plane back and untile it (normalized by the sample count in `w` when
    /// `normalize`): radiance for exports and the denoiser, guides for the denoiser.
    pub fn read_tiled_into(
        &self,
        wt: &WgpuTarget,
        plane: Plane,
        (width, height): (usize, usize),
        normalize: bool,
        tiled: &mut Vec<[f32; 4]>,
        output: &mut Vec<[f32; 4]>,
    ) -> Result<(), RenderError> {
        tiled.resize(wt.padded, [0.0; 4]);
        let bytes = (wt.padded * std::mem::size_of::<[f32; 4]>()) as u64;
        self.run(
            "warpbro_wgpu_readback",
            |_| {},
            &wt.staging,
            &[(wt.plane(plane), bytes)],
            |mapped| {
                let dst = bytemuck::cast_slice_mut::<[f32; 4], u8>(tiled);
                dst.copy_from_slice(bytes_at(mapped, 0, dst.len())?);
                Ok(())
            },
        )?;
        untile_into(tiled, width, height, normalize, output);
        Ok(())
    }

    fn create_trace_bind_group(&self, wt: &WgpuTarget) -> wgpu::BindGroup {
        self.shared
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("warpbro_trace_bg"),
                layout: &self.shared.trace_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.lut_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.objects_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.lights_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wt.active.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wt.accum.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wt.albedo.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: wt.normal.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: wt.stats.as_entire_binding(),
                    },
                ],
            })
    }

    fn create_post_bind_group(&self, wt: &WgpuTarget) -> wgpu::BindGroup {
        self.shared
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("warpbro_post_bg"),
                layout: &self.shared.post_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wt.accum.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: wt.stats.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: wt.active.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: wt.out.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 11,
                        resource: wt.sdr.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 12,
                        resource: wt.summary.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 13,
                        resource: wt.tally_parts.as_entire_binding(),
                    },
                ],
            })
    }

    /// The film's (trace, post) bind groups, rebuilt when an engine buffer was reallocated.
    fn bind_groups(
        &self,
        wt: &WgpuTarget,
    ) -> Result<(wgpu::BindGroup, wgpu::BindGroup), RenderError> {
        let mut guard = wt
            .bind_groups
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if let Some((cached_gen, trace_bg, post_bg)) = guard.as_ref()
            && *cached_gen == self.buf_generation
        {
            return Ok((trace_bg.clone(), post_bg.clone()));
        }
        let (trace_bg, post_bg) = with_error_scopes(&self.shared.device, || {
            (
                self.create_trace_bind_group(wt),
                self.create_post_bind_group(wt),
            )
        })
        .map_err(|e| RenderError::Device(format!("WGPU bind groups: {e}")))?;
        *guard = Some((self.buf_generation, trace_bg.clone(), post_bg.clone()));
        Ok((trace_bg, post_bg))
    }

    fn select_trace_pipeline(
        &self,
        scene: &Scene,
        prepared: &PreparedScene,
        full_material: bool,
        force_mixed_world: bool,
    ) -> Result<&wgpu::ComputePipeline, RenderError> {
        self.shared.trace_pipe(trace_entry(
            scene,
            prepared,
            full_material,
            force_mixed_world,
        ))
    }

    /// Trace one batch without a readback: the GPU runs on while the host returns.
    pub(crate) fn launch_trace(
        &self,
        wt: &WgpuTarget,
        scene: &Scene,
        prepared: &PreparedScene,
        full_material: bool,
        force_mixed_world: bool,
    ) -> Result<(), RenderError> {
        let pipe = self.select_trace_pipeline(scene, prepared, full_material, force_mixed_world)?;
        let (trace_bg, _) = self.bind_groups(wt)?;
        let (x, y) = self.groups(wt.padded)?;
        self.run(
            "warpbro_wgpu_trace",
            |encoder| {
                let mut pass = compute(encoder, "warpbro_wgpu_trace_pass");
                pass.set_pipeline(pipe);
                pass.set_bind_group(0, &trace_bg, &[]);
                pass.dispatch_workgroups(x, y, 1);
            },
            &wt.staging,
            &[],
            |_| Ok(()),
        )
    }

    /// Fused single-encoder `trace` + optional `adapt` + the two-pass `tally` reduction,
    /// reading back a single 16-byte `[unresolved, samples, active_tiles, 0]` summary:
    /// `(unresolved share, active tiles, converged)`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn launch_trace_and_tally(
        &self,
        wt: &WgpuTarget,
        scene: &Scene,
        prepared: &PreparedScene,
        full_material: bool,
        force_mixed_world: bool,
        run_adapt: bool,
        total_tiles: usize,
    ) -> Result<(f32, usize, bool), RenderError> {
        let pipe = self.select_trace_pipeline(scene, prepared, full_material, force_mixed_world)?;
        let (trace_bg, post_bg) = self.bind_groups(wt)?;
        let (tx, ty) = self.groups(wt.padded)?;
        let (ax, ay) = self.groups(total_tiles)?;
        let [unresolved, samples, active, _] = self.run(
            "warpbro_wgpu_trace_tally",
            |encoder| {
                let mut pass = compute(encoder, "warpbro_wgpu_trace_tally_pass");
                pass.set_pipeline(pipe);
                pass.set_bind_group(0, &trace_bg, &[]);
                pass.dispatch_workgroups(tx, ty, 1);
                pass.set_bind_group(0, &post_bg, &[]);
                if run_adapt {
                    pass.set_pipeline(&self.shared.adapt);
                    pass.dispatch_workgroups(ax, ay, 1);
                }
                self.shared.record_tally(&mut pass);
            },
            &wt.staging,
            &[(&wt.summary, 16)],
            summary,
        )?;
        let active = (active.round().max(0.0) as usize).min(total_tiles);
        Ok((unresolved_share(unresolved, samples), active, active == 0))
    }

    /// Fused single-encoder `trace` + `tally` + `tonemap_display` with a single `map_async`
    /// readback of `summary` (16 B) + `light` (`vec4<f32>`) + `pixels` (`u32` sRGB); returns the
    /// unresolved share.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn launch_fused_step_display(
        &self,
        wt: &WgpuTarget,
        scene: &Scene,
        prepared: &PreparedScene,
        full_material: bool,
        force_mixed_world: bool,
        width: usize,
        height: usize,
        light: &mut Vec<[f32; 4]>,
        pixels: &mut [u32],
    ) -> Result<f32, RenderError> {
        let n = width * height;
        if pixels.len() != n {
            return Err(RenderError::Device(format!(
                "display codes hold {} pixels, the film {n}",
                pixels.len()
            )));
        }
        light.resize(n, [0.0; 4]);
        let pipe = self.select_trace_pipeline(scene, prepared, full_material, force_mixed_world)?;
        let (trace_bg, post_bg) = self.bind_groups(wt)?;
        let (tx, ty) = self.groups(wt.padded)?;
        let (dx, dy) = self.groups(n)?;
        let out_bytes = std::mem::size_of_val(light.as_slice());
        let sdr_bytes = std::mem::size_of_val(pixels);
        self.run(
            "warpbro_wgpu_fused_step",
            |encoder| {
                let mut pass = compute(encoder, "warpbro_wgpu_fused_step_pass");
                pass.set_pipeline(pipe);
                pass.set_bind_group(0, &trace_bg, &[]);
                pass.dispatch_workgroups(tx, ty, 1);
                pass.set_bind_group(0, &post_bg, &[]);
                self.shared.record_tally(&mut pass);
                pass.set_pipeline(&self.shared.tonemap_display);
                pass.dispatch_workgroups(dx, dy, 1);
            },
            &wt.staging,
            &[
                (&wt.summary, 16),
                (&wt.out, out_bytes as u64),
                (&wt.sdr, sdr_bytes as u64),
            ],
            |mapped| {
                let [unresolved, samples, _, _] = summary(mapped)?;
                bytemuck::cast_slice_mut::<[f32; 4], u8>(light)
                    .copy_from_slice(bytes_at(mapped, 16, out_bytes)?);
                bytemuck::cast_slice_mut::<u32, u8>(pixels).copy_from_slice(bytes_at(
                    mapped,
                    16 + out_bytes,
                    sdr_bytes,
                )?);
                Ok(unresolved_share(unresolved, samples))
            },
        )
    }

    /// The share of accumulated samples with an unresolved march.
    pub fn tally(&self, wt: &WgpuTarget) -> Result<f32, RenderError> {
        let (_, post_bg) = self.bind_groups(wt)?;
        let [unresolved, samples, _, _] = self.run(
            "warpbro_wgpu_tally",
            |encoder| {
                let mut pass = compute(encoder, "warpbro_wgpu_tally_pass");
                pass.set_bind_group(0, &post_bg, &[]);
                self.shared.record_tally(&mut pass);
            },
            &wt.staging,
            &[(&wt.summary, 16)],
            summary,
        )?;
        Ok(unresolved_share(unresolved, samples))
    }

    /// The adaptive decision over `tiles` 8x4 tiles: `(tiles still sampling, all converged)`.
    pub fn adapt(&self, wt: &WgpuTarget, tiles: usize) -> Result<(usize, bool), RenderError> {
        let (_, post_bg) = self.bind_groups(wt)?;
        let (x, y) = self.groups(tiles)?;
        let [_, _, active, _] = self.run(
            "warpbro_wgpu_adapt",
            |encoder| {
                let mut pass = compute(encoder, "warpbro_wgpu_adapt_pass");
                pass.set_pipeline(&self.shared.adapt);
                pass.set_bind_group(0, &post_bg, &[]);
                pass.dispatch_workgroups(x, y, 1);
                self.shared.record_tally(&mut pass);
            },
            &wt.staging,
            &[(&wt.summary, 16)],
            summary,
        )?;
        let active = (active.round().max(0.0) as usize).min(tiles);
        Ok((active, active == 0))
    }

    /// Exposed, saturated working-space radiance (`width * height`) into `raw` for OCIO.
    pub fn tonemap(
        &self,
        wt: &WgpuTarget,
        width: usize,
        height: usize,
        raw: &mut [[f32; 4]],
    ) -> Result<(), RenderError> {
        let (_, post_bg) = self.bind_groups(wt)?;
        let (x, y) = self.groups(width * height)?;
        let bytes = std::mem::size_of_val(raw);
        self.run(
            "warpbro_wgpu_tonemap",
            |encoder| {
                let mut pass = compute(encoder, "warpbro_wgpu_tonemap_pass");
                pass.set_pipeline(&self.shared.tonemap);
                pass.set_bind_group(0, &post_bg, &[]);
                pass.dispatch_workgroups(x, y, 1);
            },
            &wt.staging,
            &[(&wt.out, bytes as u64)],
            |mapped| {
                bytemuck::cast_slice_mut::<[f32; 4], u8>(raw)
                    .copy_from_slice(bytes_at(mapped, 0, bytes)?);
                Ok(())
            },
        )
    }

    /// The fast display (no OCIO, or Reinhard): linear display light and sRGB codes.
    pub fn tonemap_display(
        &self,
        wt: &WgpuTarget,
        width: usize,
        height: usize,
        light: &mut Vec<[f32; 4]>,
        pixels: &mut [u32],
    ) -> Result<(), RenderError> {
        let n = width * height;
        if pixels.len() != n {
            return Err(RenderError::Device(format!(
                "display codes hold {} pixels, the film {n}",
                pixels.len()
            )));
        }
        light.resize(n, [0.0; 4]);
        let (_, post_bg) = self.bind_groups(wt)?;
        let (x, y) = self.groups(n)?;
        let out_bytes = std::mem::size_of_val(light.as_slice());
        let sdr_bytes = std::mem::size_of_val(pixels);
        self.run(
            "warpbro_wgpu_tonemap_display",
            |encoder| {
                let mut pass = compute(encoder, "warpbro_wgpu_tonemap_display_pass");
                pass.set_pipeline(&self.shared.tonemap_display);
                pass.set_bind_group(0, &post_bg, &[]);
                pass.dispatch_workgroups(x, y, 1);
            },
            &wt.staging,
            &[(&wt.out, out_bytes as u64), (&wt.sdr, sdr_bytes as u64)],
            |mapped| {
                bytemuck::cast_slice_mut::<[f32; 4], u8>(light)
                    .copy_from_slice(bytes_at(mapped, 0, out_bytes)?);
                bytemuck::cast_slice_mut::<u32, u8>(pixels)
                    .copy_from_slice(bytes_at(mapped, out_bytes, sdr_bytes)?);
                Ok(())
            },
        )
    }

    pub fn pick(&self, ndc: [f32; 2]) -> Result<Option<(usize, f32)>, RenderError> {
        let query = [[ndc[0], ndc[1], 0.0f32, 0.0f32]];
        let bg = with_error_scopes(&self.shared.device, || {
            self.shared
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("warpbro_wgpu_pick_bg"),
                    layout: &self.shared.pick_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.params_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self.objects_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 10,
                            resource: self.pick_aux.as_entire_binding(),
                        },
                    ],
                })
        })
        .map_err(|e| RenderError::Device(format!("WGPU pick bind group: {e}")))?;
        let [object, _, depth, _] = self.run(
            "warpbro_wgpu_pick",
            |encoder| {
                self.shared
                    .queue
                    .write_buffer(&self.pick_aux, 0, bytemuck::cast_slice(&query));
                let mut pass = compute(encoder, "warpbro_wgpu_pick_pass");
                pass.set_pipeline(&self.shared.pick);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            },
            &self.pick_staging,
            &[(&self.pick_aux, 16)],
            summary,
        )?;
        Ok((object >= 1.0).then(|| (object as usize - 1, depth)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wgpu_shader_source_contains_all_compute_entry_points() {
        let src = shader_source();
        for entry in [
            "fn tally(",
            "fn tally_sum(",
            "fn adapt(",
            "fn tonemap(",
            "fn tonemap_display(",
            "fn pick(",
        ] {
            assert!(src.contains(entry), "missing entry point {entry}");
        }
        for entry in TraceEntry::all() {
            let name = format!("fn {}(", entry.name());
            assert_eq!(src.matches(&name).count(), 1, "{name} must be emitted once");
        }
        assert!(
            !src.contains("FAMILY_WORLD"),
            "the dynamic family is FAMILY_DYNAMIC, never fractal-core's FAMILY_WORLD (= family 8)"
        );
    }

    /// One generated entry per (scope, shade, family), each with its own pipeline slot: a
    /// single scene of every family and a world of every shade, specialized and dynamic.
    #[test]
    fn trace_entries_have_unique_names_and_slots() {
        let entries: Vec<TraceEntry> = TraceEntry::all().collect();
        assert_eq!(entries.len(), 1 + 3 * (FAMILIES + 1) + 3 * FAMILIES);
        let mut names: Vec<String> = entries.iter().map(|e| e.name()).collect();
        let mut slots: Vec<usize> = entries.iter().map(|e| e.slot()).collect();
        names.sort();
        names.dedup();
        slots.sort_unstable();
        slots.dedup();
        assert_eq!(names.len(), entries.len());
        assert_eq!(slots.len(), entries.len());
        assert!(slots.iter().all(|&slot| slot < TRACE_SLOTS));
        assert!(names.contains(&"single_fast_f8".to_string()));
        assert!(names.contains(&"world_direct_f27".to_string()));
        assert!(names.contains(&"world_mixed".to_string()));
    }

    /// The launch routing (GPU-free): single scenes take their family's entry in every shade
    /// (an extended family included), a homogeneous Fast world (also one Fast Mandelbulb, the
    /// old `world_fast_bulb`) takes `world_fast_f{family}`, a Mixed or forced-mixed world the
    /// dynamic `world_mixed`.
    #[test]
    fn trace_entry_routes_each_scene_to_its_specialization() {
        use crate::scene::{Bulb, Formula};
        let direct = crate::render_profiles::DirectRender {
            shading: Default::default(),
            grid: 1,
        };
        let name = |scene: &Scene, full: bool, mixed: bool| {
            let prepared = PreparedScene::new(scene, 8, 8).unwrap();
            trace_entry(scene, &prepared, full, mixed).name()
        };
        let mut single = Scene::preset(Formula::PRESETS[FAMILY_AMAZING_SURFACE as usize]);
        assert_eq!(name(&single, false, false), "single_fast_f8");
        assert_eq!(name(&single, true, false), "single_full_f8");
        single.direct = Some(direct);
        assert_eq!(name(&single, true, false), "single_direct_f8");

        let mut world = Scene::preset(Formula::Mandelbulb(Bulb::PRESET));
        world.world_render = true;
        let mut object = Scene::preset(Formula::Mandelbulb(Bulb::PRESET));
        object.material.model = crate::scene::MaterialModel::Fast;
        world.objects = vec![object];
        assert_eq!(name(&world, false, false), "world_fast_f0");
        assert_eq!(name(&world, false, true), "world_mixed");
        let mut amazing = Scene::preset(Formula::PRESETS[FAMILY_AMAZING_SURFACE as usize]);
        amazing.material.model = crate::scene::MaterialModel::StandardSurface;
        world.objects = vec![amazing.clone(), amazing.clone()];
        assert_eq!(name(&world, false, false), "world_full_f8");
        world.objects[1].material.model = crate::scene::MaterialModel::Fast;
        assert_eq!(name(&world, false, false), "world_mixed");
        world.objects[1] = Scene::preset(Formula::Mandelbulb(Bulb::PRESET));
        world.objects[1].material.model = crate::scene::MaterialModel::StandardSurface;
        assert_eq!(name(&world, false, false), "world_full");
        world.direct = Some(direct);
        assert_eq!(name(&world, false, false), "world_direct");
    }

    #[test]
    fn gpu_wgpu_shader_compiles_without_validation_errors() {
        let _gpu_test = crate::test_gpu::lock();
        let engine =
            WgpuEngine::new().expect("WGPU engine and shader pipelines must compile cleanly");
        assert!(!engine.name().is_empty());
    }

    /// A frame above the per-dimension workgroup limit spills into y instead of failing
    /// validation (plan1 1.3); the grid covers every invocation with less than one row spare.
    #[test]
    fn gpu_wgpu_dispatch_grid_covers_dci_4k_within_the_limit() {
        let _gpu_test = crate::test_gpu::lock();
        let engine = WgpuEngine::new().expect("WGPU engine");
        let max = engine.shared.limits.max_compute_workgroups_per_dimension;
        let threads = padded_len(4096, 2160);
        let (x, y) = engine.groups(threads).unwrap();
        assert!(x <= max && y <= max);
        let covered = u64::from(x) * u64::from(y) * u64::from(BLOCK);
        assert!(covered >= threads as u64);
        assert!(covered - (threads as u64) < u64::from(x) * u64::from(BLOCK));
    }
}
