//! WarpBro's dual CUDA (`fractal-kernels`) + 28-family WGPU (`formulas.wgsl`) compute engine
//! implementing [`sl_studio::render::ComputeEngine`], plus one-time host registration
//! ([`init_warpbro`]) and [`WarpBroFractalModule`] (`StudioModule`).
//!
//! Exists so `sl-studio` remains fractal-agnostic while WarpBro drives both the embedded
//! `cuda-oxide` PTX kernels (`fractal_kernels::kernels`) and the per-family WGSL shader hook
//! ([`warpbro_shader_hook`]) over the shared `P_COUNT` studio ABI. Consumed by `src/main.rs`,
//! `src/render_bench.rs`, and `src/parity_tests.rs`.
//!
//! M1-bridge: Host CUDA types (`Context`, `Stream`, `Event`, `DeviceBuffer`, `PinnedHostBuffer`)
//! use official `NVIDIA/cuda-rust` (`cuda-core`), bridging to `fractal-kernels` via embedded PTX
//! loading and raw `CUdeviceptr`/`CUstream` kernel launches.
//! M2: Two-batch `CudaEvent` ring buffer with true GPU `elapsed_ms` timing and pinned async
//! `cuMemcpyDtoHAsync_v2` readbacks for `active` and `tally`.
//! M3: In-VRAM OptiX AI denoiser (`optix::Denoiser`) with GPU untile/normalize (zero PCIe
//! roundtrips when `readback == false`).
//! M4: Zero-copy Vulkan/DX12 <-> CUDA presentation via `pt_nv_interop::PairedGpuContext` and
//! `InteropFrameTargets` (`SharedImageTarget`) + CUDA `surf2d_write` tonemap kernel.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use cuda_core::{
    ConstantHandle, Context as CudaContext, DeviceBuffer, DeviceCopy, Event as CudaEvent,
    Function as CudaFunction, Module as CudaModule, PinnedHostBuffer, Stream as CudaStream,
    launch_kernel_on_stream, sys,
};
use fractal_core::params::*;
use pt_nv_interop::{InteropFrameTargets, InteropTargetConfig, PairedGpuContext};
use sha2::{Digest, Sha256};
use sl_studio::denoise::{self, DenoiseEngine, Mode};
use sl_studio::palette::build_lut;
use sl_studio::render::{
    BackendFilm, ComputeEngine, DenoiseOutcome, Gpu, RenderError, StepOutcome, StepSpec, Target,
    WgpuComputeEngine, film_mismatch, validate_size,
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

/// Embedded PTX helper module for M3 (GPU untile/normalize in VRAM) and M4 (zero-copy CUDA
/// tonemap + `sust.b.2d.b32` surface write into `SharedImageTarget` `Rgba8Unorm`).
const POST_PTX: &str = r#"
.version 7.0
.target sm_70
.address_size 64

.visible .entry untile_normalize_f4(
    .param .u64 p_tiled,
    .param .u64 p_linear,
    .param .u32 p_width,
    .param .u32 p_height,
    .param .u32 p_tile_cols,
    .param .u32 p_normalize
)
{
    .reg .pred %p<3>;
    .reg .u32 %r<16>;
    .reg .u64 %rd<8>;
    .reg .f32 %f<8>;

    ld.param.u32 %r1, [p_width];
    ld.param.u32 %r2, [p_height];
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.u32 %r6, %r3, %r4, %r5;
    mul.lo.u32 %r7, %r1, %r2;
    setp.ge.u32 %p0, %r6, %r7;
    @%p0 bra DONE_UNTILE;

    div.u32 %r8, %r6, %r1;   // y = idx / width
    rem.u32 %r9, %r6, %r1;   // x = idx % width

    ld.param.u32 %r10, [p_tile_cols];
    shr.u32 %r11, %r8, 2;    // y / 4
    mul.lo.u32 %r11, %r11, %r10;
    shr.u32 %r12, %r9, 3;    // x / 8
    add.u32 %r11, %r11, %r12; // tile_idx
    and.b32 %r13, %r8, 3;    // y % 4
    shl.b32 %r13, %r13, 3;   // (y % 4) * 8
    and.b32 %r14, %r9, 7;    // x % 8
    add.u32 %r13, %r13, %r14;
    shl.b32 %r11, %r11, 5;   // tile_idx * 32
    add.u32 %r11, %r11, %r13; // src_idx

    ld.param.u64 %rd1, [p_tiled];
    mul.wide.u32 %rd2, %r11, 16;
    add.u64 %rd3, %rd1, %rd2;
    ld.global.v4.f32 {%f0, %f1, %f2, %f3}, [%rd3];

    ld.param.u32 %r15, [p_normalize];
    setp.ne.u32 %p1, %r15, 0;
    @!%p1 bra STORE_UNTILE;

    setp.gt.f32 %p2, %f3, 0f00000000;
    mov.f32 %f4, 0f00000000;
    @%p2 div.rn.f32 %f4, 0f3f800000, %f3;
    mul.f32 %f0, %f0, %f4;
    mul.f32 %f1, %f1, %f4;
    mul.f32 %f2, %f2, %f4;
    mov.f32 %f3, 0f3f800000;

STORE_UNTILE:
    ld.param.u64 %rd4, [p_linear];
    mul.wide.u32 %rd5, %r6, 16;
    add.u64 %rd6, %rd4, %rd5;
    st.global.v4.f32 [%rd6], {%f0, %f1, %f2, %f3};

DONE_UNTILE:
    ret;
}

.visible .entry tonemap_to_rgba8_surface(
    .param .u64 p_src,
    .param .u64 p_surf,
    .param .u32 p_width,
    .param .u32 p_height,
    .param .u32 p_tile_cols,
    .param .u32 p_is_tiled,
    .param .f32 p_exposure,
    .param .f32 p_saturation,
    .param .u32 p_reinhard,
    .param .f32 p_luma_x,
    .param .f32 p_luma_y,
    .param .f32 p_luma_z,
    .param .f32 p_r0x,
    .param .f32 p_r0y,
    .param .f32 p_r0z,
    .param .f32 p_r1x,
    .param .f32 p_r1y,
    .param .f32 p_r1z,
    .param .f32 p_r2x,
    .param .f32 p_r2y,
    .param .f32 p_r2z
)
{
    .reg .pred %p<6>;
    .reg .u32 %r<24>;
    .reg .u64 %rd<6>;
    .reg .f32 %f<40>;

    ld.param.u32 %r1, [p_width];
    ld.param.u32 %r2, [p_height];
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.u32 %r6, %r3, %r4, %r5;
    mul.lo.u32 %r7, %r1, %r2;
    setp.ge.u32 %p0, %r6, %r7;
    @%p0 bra DONE_TM;

    div.u32 %r8, %r6, %r1;   // y
    rem.u32 %r9, %r6, %r1;   // x

    ld.param.u32 %r10, [p_is_tiled];
    setp.ne.u32 %p1, %r10, 0;
    mov.u32 %r11, %r6;
    @!%p1 bra LOAD_TM;

    ld.param.u32 %r12, [p_tile_cols];
    shr.u32 %r13, %r8, 2;
    mul.lo.u32 %r13, %r13, %r12;
    shr.u32 %r14, %r9, 3;
    add.u32 %r13, %r13, %r14;
    and.b32 %r15, %r8, 3;
    shl.b32 %r15, %r15, 3;
    and.b32 %r16, %r9, 7;
    add.u32 %r15, %r15, %r16;
    shl.b32 %r13, %r13, 5;
    add.u32 %r11, %r13, %r15;

LOAD_TM:
    ld.param.u64 %rd1, [p_src];
    mul.wide.u32 %rd2, %r11, 16;
    add.u64 %rd3, %rd1, %rd2;
    ld.global.v4.f32 {%f0, %f1, %f2, %f3}, [%rd3];

    ld.param.f32 %f4, [p_exposure];
    @!%p1 bra APPLY_EXP;
    setp.gt.f32 %p2, %f3, 0f00000000;
    mov.f32 %f5, 0f00000000;
    @%p2 div.rn.f32 %f5, %f4, %f3;
    mov.f32 %f4, %f5;

APPLY_EXP:
    mul.f32 %f0, %f0, %f4;
    mul.f32 %f1, %f1, %f4;
    mul.f32 %f2, %f2, %f4;

    ld.param.f32 %f6, [p_luma_x];
    ld.param.f32 %f7, [p_luma_y];
    ld.param.f32 %f8, [p_luma_z];
    mul.f32 %f9, %f0, %f6;
    fma.rn.f32 %f9, %f1, %f7, %f9;
    fma.rn.f32 %f9, %f2, %f8, %f9; // luma

    ld.param.f32 %f10, [p_saturation];
    sub.f32 %f0, %f0, %f9;
    fma.rn.f32 %f0, %f0, %f10, %f9;
    sub.f32 %f1, %f1, %f9;
    fma.rn.f32 %f1, %f1, %f10, %f9;
    sub.f32 %f2, %f2, %f9;
    fma.rn.f32 %f2, %f2, %f10, %f9;

    // ACEScg -> Rec.709 linear
    ld.param.f32 %f11, [p_r0x];
    ld.param.f32 %f12, [p_r0y];
    ld.param.f32 %f13, [p_r0z];
    mul.f32 %f20, %f0, %f11;
    fma.rn.f32 %f20, %f1, %f12, %f20;
    fma.rn.f32 %f20, %f2, %f13, %f20;

    ld.param.f32 %f14, [p_r1x];
    ld.param.f32 %f15, [p_r1y];
    ld.param.f32 %f16, [p_r1z];
    mul.f32 %f21, %f0, %f14;
    fma.rn.f32 %f21, %f1, %f15, %f21;
    fma.rn.f32 %f21, %f2, %f16, %f21;

    ld.param.f32 %f17, [p_r2x];
    ld.param.f32 %f18, [p_r2y];
    ld.param.f32 %f19, [p_r2z];
    mul.f32 %f22, %f0, %f17;
    fma.rn.f32 %f22, %f1, %f18, %f22;
    fma.rn.f32 %f22, %f2, %f19, %f22;

    ld.param.u32 %r17, [p_reinhard];
    setp.ne.u32 %p3, %r17, 0;
    @!%p3 bra SRGB;

    max.f32 %f20, %f20, 0f00000000;
    add.f32 %f23, %f20, 0f3f800000;
    div.rn.f32 %f20, %f20, %f23;

    max.f32 %f21, %f21, 0f00000000;
    add.f32 %f23, %f21, 0f3f800000;
    div.rn.f32 %f21, %f21, %f23;

    max.f32 %f22, %f22, 0f00000000;
    add.f32 %f23, %f22, 0f3f800000;
    div.rn.f32 %f22, %f22, %f23;

SRGB:
    // Channel R (%f20 -> %r18)
    min.f32 %f24, %f20, 0f3f800000;
    max.f32 %f24, %f24, 0f00000000;
    setp.le.f32 %p4, %f24, 0f3b4d2e1c; // 0.0031308
    mul.f32 %f25, %f24, 0f454de99a;    // 12.92 * 255.0 = 3294.6
    @%p4 bra R_QUANT;
    lg2.approx.f32 %f26, %f24;
    mul.f32 %f26, %f26, 0f3ed55555;    // 1.0 / 2.4
    ex2.approx.f32 %f26, %f26;
    fma.rn.f32 %f25, %f26, 0f3f870a3d, 0fbd6147ae; // 1.055 * pow - 0.055
    mul.f32 %f25, %f25, 0f437f0000;    // * 255.0
R_QUANT:
    add.f32 %f25, %f25, 0f3f000000;
    cvt.rzi.u32.f32 %r18, %f25;
    min.u32 %r18, %r18, 255;

    // Channel G (%f21 -> %r19)
    min.f32 %f24, %f21, 0f3f800000;
    max.f32 %f24, %f24, 0f00000000;
    setp.le.f32 %p4, %f24, 0f3b4d2e1c;
    mul.f32 %f25, %f24, 0f454de99a;
    @%p4 bra G_QUANT;
    lg2.approx.f32 %f26, %f24;
    mul.f32 %f26, %f26, 0f3ed55555;
    ex2.approx.f32 %f26, %f26;
    fma.rn.f32 %f25, %f26, 0f3f870a3d, 0fbd6147ae;
    mul.f32 %f25, %f25, 0f437f0000;
G_QUANT:
    add.f32 %f25, %f25, 0f3f000000;
    cvt.rzi.u32.f32 %r19, %f25;
    min.u32 %r19, %r19, 255;

    // Channel B (%f22 -> %r20)
    min.f32 %f24, %f22, 0f3f800000;
    max.f32 %f24, %f24, 0f00000000;
    setp.le.f32 %p4, %f24, 0f3b4d2e1c;
    mul.f32 %f25, %f24, 0f454de99a;
    @%p4 bra B_QUANT;
    lg2.approx.f32 %f26, %f24;
    mul.f32 %f26, %f26, 0f3ed55555;
    ex2.approx.f32 %f26, %f26;
    fma.rn.f32 %f25, %f26, 0f3f870a3d, 0fbd6147ae;
    mul.f32 %f25, %f25, 0f437f0000;
B_QUANT:
    add.f32 %f25, %f25, 0f3f000000;
    cvt.rzi.u32.f32 %r20, %f25;
    min.u32 %r20, %r20, 255;

    shl.b32 %r19, %r19, 8;
    shl.b32 %r20, %r20, 16;
    or.b32 %r21, %r18, %r19;
    or.b32 %r21, %r21, %r20;
    or.b32 %r21, %r21, 0xff000000;

    shl.b32 %r22, %r9, 2; // x * 4 bytes
    ld.param.u64 %rd4, [p_surf];
    sust.b.2d.b32.trap [%rd4, {%r22, %r8}], %r21;

DONE_TM:
    ret;
}
"#;

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
        _bus: &sl_studio::node_editor_panel::EventBus,
        _hotkeys: &mut sl_studio::HotkeyRegistry,
    ) {
        init_warpbro();
    }
}

/// Register WarpBro's `"Fractal"` node spec, 28-family WGSL shader hook, extra thumbnail
/// cache fingerprint, and built-in scene templates into `sl-studio`.
/// Safe and idempotent to call from `main`, CLI subcommands, and tests.
pub fn init_warpbro() {
    static INIT: OnceLock<()> = OnceLock::new();
    INIT.get_or_init(|| {
        sl_studio::node_spec::register_node_type(Arc::new(FractalNodeSpec));
        sl_studio::custom_branches::register_scene_packer::<crate::fractal_node::FractalParams>();
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
/// the PTX, and return the device name.
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
    let _ = writeln!(s, "const SIGNED_MASK: u32 = {SIGNED_MASK}u;");
    let _ = writeln!(s, "const ITERATING_MASK: u32 = {ITERATING_MASK}u;");
    s.push_str(
        "fn family_distance(is_world: bool, obj: u32, family_override: u32, p: vec3<f32>, trap_mode: u32) -> vec2<f32> {\n",
    );
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

/// Thin zero-copy raw-handle bridge over `fractal-kernels`' embedded PTX module using
/// `NVIDIA/cuda-rust` (`cuda_core::Module`, `Function`, `ConstantHandle`) plus the M3/M4
/// `POST_PTX` helper module.
struct FractalModule {
    module: Arc<CudaModule>,
    params_const: ConstantHandle,
    params_pinned: [PinnedHostBuffer<f32>; 2],
    params_slot: usize,
    functions: HashMap<&'static str, CudaFunction>,
    #[expect(
        dead_code,
        reason = "keeps CUDA post-processing module loaded for untile/tonemap functions"
    )]
    post_module: Arc<CudaModule>,
    untile_fn: CudaFunction,
    tonemap_surf_fn: CudaFunction,
}

fn find_ptx_const_symbol<'a>(ptx: &'a [u8], suffix: &str) -> Option<&'a str> {
    let text = std::str::from_utf8(ptx).ok()?;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("//") {
            continue;
        }
        if !trimmed.split_whitespace().any(|t| t == ".const") {
            continue;
        }
        let Some(decl) = trimmed.split(['[', ';', '=']).next() else {
            continue;
        };
        let Some(sym) = decl.split_whitespace().last() else {
            continue;
        };
        if sym == suffix || sym.ends_with(&format!("_{suffix}")) {
            return Some(sym);
        }
    }
    None
}

impl FractalModule {
    fn load(ctx: &Arc<CudaContext>) -> Result<Self, String> {
        let embedded = cuda_core::embedded::embedded_modules_from_current_exe()
            .map_err(|e| format!("embedded CUDA modules: {e:?}"))?;
        let bundle = embedded
            .iter()
            .find(|m| {
                m.bundle()
                    .entries
                    .iter()
                    .any(|e| e.symbol == "world_fast_f0")
            })
            .ok_or_else(|| {
                "no embedded fractal-kernels CUDA module found (missing world_fast_f0)".to_string()
            })?;
        let ptx = bundle
            .payload(cuda_core::embedded::ArtifactPayloadKind::Ptx)
            .ok_or_else(|| format!("embedded module '{}' has no PTX payload", bundle.name()))?;
        let params_symbol = find_ptx_const_symbol(ptx, "PARAMS").ok_or_else(|| {
            format!(
                "PARAMS constant symbol not found in embedded module '{}'",
                bundle.name()
            )
        })?;
        log::info!(
            "CUDA startup: loading module '{}' (PARAMS = '{}')",
            bundle.name(),
            params_symbol
        );
        let module = bundle
            .load(ctx)
            .map_err(|e| format!("load fractal-kernels module: {e:?}"))?;
        let (params_dptr, params_size) = module
            .get_global(params_symbol)
            .map_err(|e| format!("resolve {params_symbol} global: {e:?}"))?;
        let expected_bytes = std::mem::size_of::<[f32; P_COUNT]>();
        if params_size != expected_bytes {
            return Err(format!(
                "PARAMS size mismatch: device has {params_size} B, host expects {expected_bytes} B"
            ));
        }
        // SAFETY: `params_dptr` was just resolved from `module`'s `"PARAMS"` constant symbol and
        // its driver-reported byte size equals `size_of::<[f32; P_COUNT]>()`.
        let params_const = unsafe { ConstantHandle::from_raw(params_dptr) };
        let params_pinned_0 = PinnedHostBuffer::<f32>::zeroed(ctx, P_COUNT)
            .map_err(|e| format!("pinned params 0: {e:?}"))?;
        let params_pinned_1 = PinnedHostBuffer::<f32>::zeroed(ctx, P_COUNT)
            .map_err(|e| format!("pinned params 1: {e:?}"))?;
        // SAFETY: `params_pinned_0` holds `P_COUNT * 4 == expected_bytes` readable zeroed bytes
        // and `module` stays loaded for the lifetime of `FractalModule`.
        unsafe {
            module
                .copy_bytes_to_device_global_sync(
                    params_dptr,
                    params_pinned_0.as_ptr().cast(),
                    expected_bytes,
                )
                .map_err(|e| format!("init PARAMS: {e:?}"))?;
        }

        let post_module = ctx
            .load_module_from_image(POST_PTX.as_bytes())
            .map_err(|e| format!("load post PTX: {e:?}"))?;
        let untile_fn = post_module
            .load_function("untile_normalize_f4")
            .map_err(|e| format!("load untile_normalize_f4: {e:?}"))?;
        let tonemap_surf_fn = post_module
            .load_function("tonemap_to_rgba8_surface")
            .map_err(|e| format!("load tonemap_to_rgba8_surface: {e:?}"))?;

        Ok(Self {
            module,
            params_const,
            params_pinned: [params_pinned_0, params_pinned_1],
            params_slot: 0,
            functions: HashMap::new(),
            post_module,
            untile_fn,
            tonemap_surf_fn,
        })
    }

    fn set_params_async(
        &mut self,
        stream: &CudaStream,
        params: &[f32; P_COUNT],
    ) -> Result<(), RenderError> {
        let slot = self.params_slot;
        self.params_slot = (slot + 1) % 2;
        self.params_pinned[slot].copy_from_slice(params);
        // SAFETY: `params_pinned[slot]` is page-locked host memory of `P_COUNT` `f32`s and
        // `params_const` points to `PARAMS` in `self.module` of identical size; 2-slot rotation
        // prevents overwriting an in-flight batch's staging buffer.
        unsafe {
            self.params_const.write_async(
                stream,
                self.params_pinned[slot].as_ptr().cast(),
                std::mem::size_of::<[f32; P_COUNT]>(),
            )
        }
        .map_err(|e| device("CUDA parameter upload", e))
    }

    fn function(&mut self, name: &'static str) -> Result<CudaFunction, RenderError> {
        if let Some(f) = self.functions.get(name) {
            return Ok(f.clone());
        }
        let f = self
            .module
            .load_function(name)
            .map_err(|e| device(name, e))?;
        self.functions.insert(name, f.clone());
        Ok(f)
    }

    fn launch_slices(
        &mut self,
        stream: &CudaStream,
        name: &'static str,
        grid_x: u32,
        block_x: u32,
        slices: &[(sys::CUdeviceptr, usize)],
    ) -> Result<(), RenderError> {
        let func = self.function(name)?;
        // Each `#[kernel]` `&[T]` / `DisjointSlice<T>` parameter expands to `(CUdeviceptr, u64)`.
        let mut storage: [(sys::CUdeviceptr, u64); 8] = [(0, 0); 8];
        if slices.len() > storage.len() {
            return Err(RenderError::Device(format!(
                "kernel {name} slice count {} exceeds 8",
                slices.len()
            )));
        }
        let mut args: [*mut c_void; 16] = [std::ptr::null_mut(); 16];
        for (i, &(ptr, len)) in slices.iter().enumerate() {
            storage[i] = (ptr, len as u64);
            args[i * 2] = (&mut storage[i].0 as *mut sys::CUdeviceptr).cast();
            args[i * 2 + 1] = (&mut storage[i].1 as *mut u64).cast();
        }
        // SAFETY: `func` belongs to `self.module` on `stream`'s context; `args` matches
        // `fractal-kernels`' `(CUdeviceptr, u64)` slice ABI; all device buffers remain live until
        // stream completion.
        unsafe {
            launch_kernel_on_stream(
                &func,
                (grid_x, 1, 1),
                (block_x, 1, 1),
                0,
                stream,
                &mut args[..slices.len() * 2],
            )
        }
        .map_err(|e| device(name, e))
    }

    fn launch_untile_normalize(
        &self,
        stream: &CudaStream,
        tiled: &DeviceBuffer<[f32; 4]>,
        linear: &mut DeviceBuffer<[f32; 4]>,
        width: u32,
        height: u32,
        normalize: bool,
    ) -> Result<(), RenderError> {
        let pixels = width.saturating_mul(height);
        if pixels == 0 {
            return Ok(());
        }
        let tile_cols = width.div_ceil(8);
        let norm_flag: u32 = u32::from(normalize);
        let mut p_tiled = tiled.cu_deviceptr();
        let mut p_linear = linear.cu_deviceptr();
        let mut p_w = width;
        let mut p_h = height;
        let mut p_tc = tile_cols;
        let mut p_norm = norm_flag;
        let mut args: [*mut c_void; 6] = [
            (&mut p_tiled as *mut sys::CUdeviceptr).cast(),
            (&mut p_linear as *mut sys::CUdeviceptr).cast(),
            (&mut p_w as *mut u32).cast(),
            (&mut p_h as *mut u32).cast(),
            (&mut p_tc as *mut u32).cast(),
            (&mut p_norm as *mut u32).cast(),
        ];
        let grid = (pixels.div_ceil(128), 1, 1);
        // SAFETY: `self.untile_fn` is loaded from `self.post_module` on `stream`'s context;
        // `tiled` has `padded_len(width, height)` elements and `linear` has `width * height`.
        unsafe { launch_kernel_on_stream(&self.untile_fn, grid, (128, 1, 1), 0, stream, &mut args) }
            .map_err(|e| device("untile_normalize_f4", e))
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "PTX kernel launch marshals 10 scalar/handle parameters for zero-copy tonemap"
    )]
    fn launch_tonemap_to_surface(
        &self,
        stream: &CudaStream,
        src_ptr: sys::CUdeviceptr,
        surf_obj: u64,
        width: u32,
        height: u32,
        is_tiled: bool,
        exposure: f32,
        saturation: f32,
        reinhard: bool,
        luma: [f32; 3],
    ) -> Result<(), RenderError> {
        let pixels = width.saturating_mul(height);
        if pixels == 0 {
            return Ok(());
        }
        let tile_cols = width.div_ceil(8);
        let c0 = sl_studio::color::to_709([1.0, 0.0, 0.0]);
        let c1 = sl_studio::color::to_709([0.0, 1.0, 0.0]);
        let c2 = sl_studio::color::to_709([0.0, 0.0, 1.0]);
        let mut p_src = src_ptr;
        let mut p_surf = surf_obj;
        let mut p_w = width;
        let mut p_h = height;
        let mut p_tc = tile_cols;
        let mut p_tiled = u32::from(is_tiled);
        let mut p_exp = exposure;
        let mut p_sat = saturation;
        let mut p_rein = u32::from(reinhard);
        let mut p_lx = luma[0];
        let mut p_ly = luma[1];
        let mut p_lz = luma[2];
        let mut p_r0x = c0[0];
        let mut p_r0y = c1[0];
        let mut p_r0z = c2[0];
        let mut p_r1x = c0[1];
        let mut p_r1y = c1[1];
        let mut p_r1z = c2[1];
        let mut p_r2x = c0[2];
        let mut p_r2y = c1[2];
        let mut p_r2z = c2[2];
        let mut args: [*mut c_void; 21] = [
            (&mut p_src as *mut sys::CUdeviceptr).cast(),
            (&mut p_surf as *mut u64).cast(),
            (&mut p_w as *mut u32).cast(),
            (&mut p_h as *mut u32).cast(),
            (&mut p_tc as *mut u32).cast(),
            (&mut p_tiled as *mut u32).cast(),
            (&mut p_exp as *mut f32).cast(),
            (&mut p_sat as *mut f32).cast(),
            (&mut p_rein as *mut u32).cast(),
            (&mut p_lx as *mut f32).cast(),
            (&mut p_ly as *mut f32).cast(),
            (&mut p_lz as *mut f32).cast(),
            (&mut p_r0x as *mut f32).cast(),
            (&mut p_r0y as *mut f32).cast(),
            (&mut p_r0z as *mut f32).cast(),
            (&mut p_r1x as *mut f32).cast(),
            (&mut p_r1y as *mut f32).cast(),
            (&mut p_r1z as *mut f32).cast(),
            (&mut p_r2x as *mut f32).cast(),
            (&mut p_r2y as *mut f32).cast(),
            (&mut p_r2z as *mut f32).cast(),
        ];
        let grid = (pixels.div_ceil(128), 1, 1);
        // SAFETY: `self.tonemap_surf_fn` is loaded from `self.post_module`; `src_ptr` holds the
        // frame pixels and `surf_obj` is a valid `Rgba8Unorm` CUDA surface owned by `InteropFrameTargets`
        // under `TargetOwner::Cuda`.
        unsafe {
            launch_kernel_on_stream(
                &self.tonemap_surf_fn,
                grid,
                (128, 1, 1),
                0,
                stream,
                &mut args,
            )
        }
        .map_err(|e| device("tonemap_to_rgba8_surface", e))
    }
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

struct OptixDenoiserState {
    denoiser: optix::Denoiser,
    width: u32,
    height: u32,
    mode: Mode,
}

struct CudaEngine {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    paired: Option<PairedGpuContext>,
    module: FractalModule,
    lut: DeviceBuffer<[f32; 4]>,
    objects: DeviceBuffer<f32>,
    lights: DeviceBuffer<f32>,
    pick_query: DeviceBuffer<[f32; 4]>,
    pick_out: DeviceBuffer<[f32; 4]>,
    lut_resident: Resident<LutResidentKey>,
    object_resident: Resident<ObjectResidentKey>,
    light_resident: Resident<LightResidentKey>,
    optix_ctx: Option<Result<Arc<optix::DeviceContext>, String>>,
    optix_denoiser: Option<OptixDenoiserState>,
    last_exposure: f32,
    last_saturation: f32,
    last_reinhard: bool,
    last_luma: [f32; 3],
    name: String,
}

impl CudaEngine {
    fn new() -> Result<Self, String> {
        let started = std::time::Instant::now();
        log::info!("CUDA startup: creating context");
        let (ctx, stream, paired) = match gpu_info::shared_device() {
            Some(shared) if shared.has_external_interop() => {
                match PairedGpuContext::from_device_queue(&shared.device, &shared.queue) {
                    Ok(paired) => {
                        let ctx = Arc::clone(paired.cuda_context());
                        let stream = Arc::clone(paired.cuda_stream());
                        log::info!(
                            "CUDA startup: paired with wgpu on ordinal {}",
                            paired.cuda_ordinal()
                        );
                        (ctx, stream, Some(paired))
                    }
                    Err(e) => {
                        log::warn!("CUDA/wgpu pairing unavailable ({e}); using CUDA ordinal 0");
                        let ctx =
                            CudaContext::new(0).map_err(|e| format!("CUDA context: {e:?}"))?;
                        let stream = ctx.default_stream();
                        (ctx, stream, None)
                    }
                }
            }
            _ => {
                let ctx = CudaContext::new(0).map_err(|e| format!("CUDA context: {e:?}"))?;
                let stream = ctx.default_stream();
                (ctx, stream, None)
            }
        };
        log::info!("CUDA startup: context ready in {:?}", started.elapsed());
        let module_started = std::time::Instant::now();
        log::info!("CUDA startup: loading embedded kernels");
        let module = FractalModule::load(&ctx)?;
        log::info!(
            "CUDA startup: kernels ready in {:?}",
            module_started.elapsed()
        );
        let name = ctx.name().unwrap_or_else(|e| {
            log::warn!("CUDA device name query failed: {e:?}");
            "CUDA GPU".into()
        });
        let lut =
            DeviceBuffer::zeroed(&stream, PALETTE_SAMPLES + 1).map_err(|e| format!("{e:?}"))?;
        Ok(Self {
            ctx,
            stream: stream.clone(),
            paired,
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
            optix_ctx: None,
            optix_denoiser: None,
            last_exposure: 1.0,
            last_saturation: 1.0,
            last_reinhard: false,
            last_luma: [0.272_228_7, 0.674_081_8, 0.053_689_5],
            name,
        })
    }

    fn new_batch_slot(&self, tiles: usize) -> Result<BatchSlot, RenderError> {
        let timing_flag = Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT);
        let start_event = self
            .ctx
            .new_event_with_flags(timing_flag)
            .map_err(|e| device("create start event", e))?;
        let end_event = self
            .ctx
            .new_event_with_flags(timing_flag)
            .map_err(|e| device("create end event", e))?;
        let active_pinned = PinnedHostBuffer::zeroed(&self.ctx, tiles.max(1))
            .map_err(|e| device("pinned active buffer", e))?;
        Ok(BatchSlot {
            start_event,
            end_event,
            active_pinned,
            has_active: false,
            in_flight: false,
            spp: 0,
        })
    }

    fn create_target(&self, width: usize, height: usize) -> Result<CudaTarget, RenderError> {
        validate_size(width, height, 1)?;
        let pixels = width * height;
        let padded = padded_len(width, height);
        let tiles = tile_count(width, height);
        let s = &self.stream;
        let active = zeroed(s, tiles, "tile flags")?;
        let ring = [self.new_batch_slot(tiles)?, self.new_batch_slot(tiles)?];
        let tally_pinned = PinnedHostBuffer::zeroed(&self.ctx, TALLY_THREADS)
            .map_err(|e| device("pinned tally buffer", e))?;
        Ok(CudaTarget {
            accum: zeroed(s, padded, "accumulator")?,
            albedo: zeroed(s, padded, "albedo guides")?,
            normal: zeroed(s, padded, "normal guides")?,
            stats: zeroed(s, padded, "sample statistics")?,
            tally: zeroed(s, TALLY_THREADS, "tally sums")?,
            active,
            out: zeroed(s, pixels, "output")?,
            padded,
            ring,
            ring_head: 0,
            in_flight: 0,
            tally_pinned,
            linear_beauty: None,
            linear_albedo: None,
            linear_normal: None,
            denoised_beauty: None,
            interop: None,
        })
    }

    fn clear_target(&self, ct: &mut CudaTarget) -> Result<(), RenderError> {
        for slot in &mut ct.ring {
            if slot.in_flight {
                slot.end_event
                    .synchronize()
                    .map_err(|e| device("drain in-flight batch", e))?;
                slot.in_flight = false;
                slot.has_active = false;
                slot.spp = 0;
            }
        }
        ct.in_flight = 0;
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

    fn ensure_optix_context(&mut self) -> Result<&Arc<optix::DeviceContext>, String> {
        let ctx = &self.ctx;
        let entry = self.optix_ctx.get_or_insert_with(|| {
            optix::DeviceContext::new(ctx, None, 0, false)
                .map_err(|e| format!("OptiX context init failed: {e:?}"))
        });
        entry.as_ref().map_err(Clone::clone)
    }

    fn ensure_optix_denoiser(
        &mut self,
        width: u32,
        height: u32,
        mode: Mode,
    ) -> Result<&mut optix::Denoiser, String> {
        let needs_rebuild = self.optix_denoiser.as_ref().is_none_or(|state| {
            state.width != width || state.height != height || state.mode != mode
        });
        if needs_rebuild {
            self.optix_denoiser = None;
            let optix_ctx = self.ensure_optix_context()?.clone();
            let opts = optix::DenoiserOptions {
                model: optix::DenoiserModel::Aov,
                albedo: mode != Mode::Color,
                normal: mode == Mode::ColorAlbedoNormal,
                alpha: optix::AlphaMode::Copy,
                tile: None,
                budget: None,
            };
            let denoiser = optix::Denoiser::new(&optix_ctx, &self.stream, opts, width, height)
                .map_err(|e| format!("OptiX denoiser setup failed: {e:?}"))?;
            self.optix_denoiser = Some(OptixDenoiserState {
                denoiser,
                width,
                height,
                mode,
            });
        }
        self.optix_denoiser
            .as_mut()
            .map(|state| &mut state.denoiser)
            .ok_or_else(|| "OptiX denoiser missing".to_string())
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

struct BatchSlot {
    start_event: CudaEvent,
    end_event: CudaEvent,
    active_pinned: PinnedHostBuffer<u32>,
    has_active: bool,
    in_flight: bool,
    spp: u32,
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
    ring: [BatchSlot; 2],
    ring_head: usize,
    in_flight: usize,
    tally_pinned: PinnedHostBuffer<[f64; 2]>,
    linear_beauty: Option<DeviceBuffer<[f32; 4]>>,
    linear_albedo: Option<DeviceBuffer<[f32; 4]>>,
    linear_normal: Option<DeviceBuffer<[f32; 4]>>,
    denoised_beauty: Option<DeviceBuffer<[f32; 4]>>,
    interop: Option<InteropFrameTargets>,
}

impl CudaTarget {
    fn plane(&self, plane: Plane) -> &DeviceBuffer<[f32; 4]> {
        match plane {
            Plane::Accum => &self.accum,
            Plane::Albedo => &self.albedo,
            Plane::Normal => &self.normal,
        }
    }

    fn harvest_oldest(
        &mut self,
        target_active_host: &mut [u32],
    ) -> Result<Option<(f32, u32)>, RenderError> {
        if self.in_flight == 0 {
            return Ok(None);
        }
        let oldest_idx = (self.ring_head + 2 - self.in_flight) % 2;
        let slot = &mut self.ring[oldest_idx];
        let elapsed_ms = slot
            .start_event
            .elapsed_ms(&slot.end_event)
            .map_err(|e| device("batch elapsed_ms", e))?;
        if slot.has_active && !target_active_host.is_empty() {
            let n = target_active_host.len().min(slot.active_pinned.len());
            target_active_host[..n].copy_from_slice(&slot.active_pinned[..n]);
        }
        let spp = slot.spp;
        slot.in_flight = false;
        slot.has_active = false;
        slot.spp = 0;
        self.in_flight -= 1;
        Ok(Some((elapsed_ms, spp)))
    }
}

/// CUDA (`fractal-kernels`) plus WGPU. The WGPU backend is soluwell's own
/// [`WgpuComputeEngine`]: every `Backend::Wgpu` arm delegates to it, so step pacing, adaptive
/// sampling and display bookkeeping have one implementation; WarpBro contributes only the
/// shader hook ([`warpbro_shader_hook`], installed globally by [`init_warpbro`]).
pub struct WarpBroComputeEngine {
    cuda: Result<CudaEngine, String>,
    wgpu: Option<Result<WgpuComputeEngine, String>>,
    lut_staging: Vec<[f32; 4]>,
    optix_warned: bool,
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
            optix_warned: false,
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

    fn wgpu(&self) -> Result<&WgpuComputeEngine, RenderError> {
        match &self.wgpu {
            Some(Ok(wgpu)) => Ok(wgpu),
            Some(Err(error)) => Err(RenderError::Backend(error.clone())),
            None => Err(RenderError::Backend(
                "WGPU backend is not initialized".into(),
            )),
        }
    }

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
        let p: [f32; P_COUNT] = scene
            .pack(1, 1)
            .try_into()
            .map_err(|_| RenderError::Scene("packed parameter ABI size mismatch".into()))?;
        if backend == Backend::Wgpu {
            let wgpu = self.wgpu_mut()?.wgpu_mut()?;
            wgpu.set_params(&p)?;
            return wgpu.probe_formula(scene.family_code(), points);
        }
        let cuda = self.cuda_mut()?;
        cuda.module.set_params_async(&cuda.stream, &p)?;
        let mut query = zeroed(&cuda.stream, points.len(), "formula probe queries")?;
        query
            .copy_from_host(&cuda.stream, points)
            .map_err(|e| device("formula probe upload", e))?;
        let out = zeroed(&cuda.stream, points.len(), "formula probe output")?;
        cuda.module.launch_slices(
            &cuda.stream,
            "probe_formula",
            (points.len() as u32).div_ceil(128),
            128,
            &[
                (cuda.objects.cu_deviceptr(), cuda.objects.len()),
                (cuda.lights.cu_deviceptr(), cuda.lights.len()),
                (query.cu_deviceptr(), query.len()),
                (out.cu_deviceptr(), out.len()),
            ],
        )?;
        let mut result = vec![[0.0; 4]; points.len()];
        out.copy_to_host(&cuda.stream, &mut result)
            .map_err(|e| device("formula probe readback", e))?;
        Ok(result)
    }

    pub fn cuda_kernel_resources(
        &mut self,
        scene: &Scene,
        prepared: &PreparedScene,
    ) -> Result<serde_json::Value, RenderError> {
        let (shade, family) = cuda_trace_entry(scene, prepared, false);
        let module = &mut self.cuda_mut()?.module;
        macro_rules! resources {
            ($prepare:ident, $run:ident) => {{
                let function = module.function(stringify!($run))?;
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
                let mut staging = std::mem::take(&mut self.lut_staging);
                let res = (|| -> Result<(), RenderError> {
                    let cuda = self.cuda_mut()?;
                    cuda.upload_prepared(prepared, env, world_render, &mut staging, stats)?;
                    cuda.module.set_params_async(&cuda.stream, params)?;
                    cuda.last_exposure = params[P_EXPOSURE];
                    cuda.last_saturation = params[P_SATURATION];
                    cuda.last_reinhard = params[P_TONEMAP] != 0.0;
                    cuda.last_luma = [params[P_LUMA], params[P_LUMA + 1], params[P_LUMA + 2]];
                    Ok(())
                })();
                self.lut_staging = staging;
                res
            }
            Backend::Auto => Err(unresolved_auto()),
        }
    }

    fn poll_or_wait_inflight(
        &mut self,
        backend: Backend,
        target: &mut Target,
    ) -> Result<Option<(f32, u32)>, RenderError> {
        match backend {
            Backend::Wgpu => self.wgpu_mut()?.poll_or_wait_inflight(backend, target),
            Backend::Cuda => {
                let Some((_, film)) = target.film.as_mut() else {
                    return Ok(None);
                };
                let ct = film
                    .downcast_mut::<CudaTarget>()
                    .ok_or_else(film_mismatch)?;
                if ct.in_flight == 0 {
                    return Ok(None);
                }
                let oldest_idx = (ct.ring_head + 2 - ct.in_flight) % 2;
                let ready = if ct.in_flight >= 2 {
                    ct.ring[oldest_idx]
                        .end_event
                        .synchronize()
                        .map_err(|e| device("wait in-flight batch", e))?;
                    true
                } else {
                    // Non-blocking poll when only 1 batch is in flight.
                    // SAFETY: `end_event` is a valid `CUevent` created on `cuda.ctx`.
                    let status =
                        unsafe { sys::cuEventQuery(ct.ring[oldest_idx].end_event.cu_event()) };
                    match status {
                        sys::cudaError_enum_CUDA_SUCCESS => true,
                        sys::cudaError_enum_CUDA_ERROR_NOT_READY => false,
                        other => {
                            return Err(device("cuEventQuery", other));
                        }
                    }
                };
                if !ready {
                    return Ok(None);
                }
                let had_active = ct.ring[oldest_idx].has_active;
                let harvested = ct.harvest_oldest(&mut target.active_host)?;
                if had_active && !target.active_host.is_empty() {
                    let tiles = target.active_host.len();
                    let active = target.active_host.iter().filter(|&&a| a != 0).count();
                    target.active_tiles = (active, tiles);
                    target.converged = active == 0;
                }
                Ok(harvested)
            }
            Backend::Auto => Err(unresolved_auto()),
        }
    }

    fn end_batch(
        &mut self,
        backend: Backend,
        target: &mut Target,
        started: Instant,
        spp: u32,
    ) -> Result<Option<(f32, u32)>, RenderError> {
        match backend {
            Backend::Wgpu => self.wgpu_mut()?.end_batch(backend, target, started, spp),
            Backend::Cuda => {
                if spp == 0 {
                    return Ok(None);
                }
                let cuda = self.cuda()?;
                let ct = target
                    .film
                    .as_mut()
                    .and_then(|(_, f)| f.downcast_mut::<CudaTarget>())
                    .ok_or_else(film_mismatch)?;
                let slot = &mut ct.ring[ct.ring_head % 2];
                slot.end_event
                    .record(&cuda.stream)
                    .map_err(|e| device("record end event", e))?;
                slot.in_flight = true;
                slot.spp = spp;
                ct.ring_head = (ct.ring_head + 1) % 2;
                ct.in_flight += 1;
                Ok(None)
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
                    let cuda = self.cuda_mut()?;
                    let ct = target
                        .film
                        .as_mut()
                        .and_then(|(_, f)| f.downcast_mut::<CudaTarget>())
                        .ok_or_else(film_mismatch)?;
                    let slot = &mut ct.ring[ct.ring_head % 2];
                    slot.has_active = false;
                    slot.start_event
                        .record(&cuda.stream)
                        .map_err(|e| device("record start event", e))?;

                    let padded = ct.accum.len() as u32;
                    let grid_x = padded.div_ceil(BLOCK);
                    let slices = [
                        (cuda.lut.cu_deviceptr(), cuda.lut.len()),
                        (cuda.objects.cu_deviceptr(), cuda.objects.len()),
                        (cuda.lights.cu_deviceptr(), cuda.lights.len()),
                        (ct.accum.cu_deviceptr(), ct.accum.len()),
                        (ct.albedo.cu_deviceptr(), ct.albedo.len()),
                        (ct.normal.cu_deviceptr(), ct.normal.len()),
                        (ct.active.cu_deviceptr(), ct.active.len()),
                        (ct.stats.cu_deviceptr(), ct.stats.len()),
                    ];
                    macro_rules! launch {
                        ($prep:ident, $run:ident) => {{
                            cuda.module.launch_slices(
                                &cuda.stream,
                                stringify!($run),
                                grid_x,
                                BLOCK,
                                &slices,
                            )?;
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
                        let cuda = self.cuda_mut()?;
                        let ct = target
                            .film
                            .as_mut()
                            .and_then(|(_, f)| f.downcast_mut::<CudaTarget>())
                            .ok_or_else(film_mismatch)?;
                        cuda.module.launch_slices(
                            &cuda.stream,
                            "adapt",
                            (tiles as u32).div_ceil(BLOCK),
                            BLOCK,
                            &[
                                (ct.accum.cu_deviceptr(), ct.accum.len()),
                                (ct.stats.cu_deviceptr(), ct.stats.len()),
                                (ct.active.cu_deviceptr(), ct.active.len()),
                            ],
                        )?;
                        let slot = &mut ct.ring[ct.ring_head % 2];
                        let copy_bytes = tiles * std::mem::size_of::<u32>();
                        // SAFETY: `slot.active_pinned` is page-locked host memory with at least
                        // `tiles` elements; `ct.active` has `tiles` device elements; copy is
                        // stream-ordered on `cuda.stream`.
                        let err = unsafe {
                            sys::cuMemcpyDtoHAsync_v2(
                                slot.active_pinned.as_mut_ptr().cast(),
                                ct.active.cu_deviceptr(),
                                copy_bytes,
                                cuda.stream.cu_stream(),
                            )
                        };
                        if err != sys::cudaError_enum_CUDA_SUCCESS {
                            return Err(device("tile flags async readback", err));
                        }
                        slot.has_active = true;
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
                let cuda = self.cuda_mut()?;
                cuda.module.launch_slices(
                    &cuda.stream,
                    "tally",
                    TALLY_THREADS as u32 / BLOCK,
                    BLOCK,
                    &[
                        (ct.accum.cu_deviceptr(), ct.accum.len()),
                        (ct.stats.cu_deviceptr(), ct.stats.len()),
                        (ct.tally.cu_deviceptr(), ct.tally.len()),
                    ],
                )?;
                let bytes = TALLY_THREADS * std::mem::size_of::<[f64; 2]>();
                // SAFETY: `ct.tally_pinned` is page-locked host memory of `TALLY_THREADS` `[f64; 2]`
                // entries; `ct.tally` has `TALLY_THREADS` elements; synchronized right below.
                let err = unsafe {
                    sys::cuMemcpyDtoHAsync_v2(
                        ct.tally_pinned.as_mut_ptr().cast(),
                        ct.tally.cu_deviceptr(),
                        bytes,
                        cuda.stream.cu_stream(),
                    )
                };
                if err != sys::cudaError_enum_CUDA_SUCCESS {
                    return Err(device("tally async readback", err));
                }
                cuda.stream
                    .synchronize()
                    .map_err(|e| device("tally synchronize", e))?;
                let (unresolved, samples) = ct
                    .tally_pinned
                    .iter()
                    .fold((0.0f64, 0.0f64), |(l, s), t| (l + t[0], s + t[1]));
                Ok(if samples > 0.0 {
                    (unresolved / samples) as f32
                } else {
                    0.0
                })
            }
            Backend::Auto => Err(unresolved_auto()),
        }
    }

    fn denoise_film(
        &mut self,
        backend: Backend,
        film: &mut BackendFilm,
        width: usize,
        height: usize,
        settings: &denoise::Settings,
        _samples: u32,
        readback: bool,
    ) -> Result<Option<DenoiseOutcome>, RenderError> {
        if backend != Backend::Cuda {
            if settings.engine == DenoiseEngine::Optix {
                return Err(RenderError::Backend(
                    "OptiX denoiser requires a CUDA/NVIDIA backend".into(),
                ));
            }
            return Ok(None);
        }
        if settings.engine == DenoiseEngine::Oidn || width == 0 || height == 0 {
            return Ok(None);
        }
        let w = u32::try_from(width)
            .map_err(|_| RenderError::Scene("Denoise width exceeds u32".into()))?;
        let h = u32::try_from(height)
            .map_err(|_| RenderError::Scene("Denoise height exceeds u32".into()))?;
        let pixels = width
            .checked_mul(height)
            .ok_or_else(|| RenderError::Scene("Denoise dimensions overflow".into()))?;

        let t0 = Instant::now();
        let cuda = self.cuda_mut()?;
        if let Err(err) = cuda.ensure_optix_denoiser(w, h, settings.mode) {
            if settings.engine == DenoiseEngine::Optix {
                return Err(RenderError::Backend(err));
            }
            if !self.optix_warned {
                log::warn!("OptiX denoiser unavailable ({err}); falling back to OIDN");
                self.optix_warned = true;
            }
            return Ok(None);
        }

        let ct = film
            .downcast_mut::<CudaTarget>()
            .ok_or_else(film_mismatch)?;
        let stream = Arc::clone(&cuda.stream);
        let ensure_buf =
            |slot: &mut Option<DeviceBuffer<[f32; 4]>>, what: &str| -> Result<(), RenderError> {
                if slot.as_ref().is_none_or(|b| b.len() != pixels) {
                    *slot = Some(zeroed(&stream, pixels, what)?);
                }
                Ok(())
            };
        ensure_buf(&mut ct.linear_beauty, "linear beauty")?;
        ensure_buf(&mut ct.denoised_beauty, "denoised beauty")?;
        let linear_beauty = ct
            .linear_beauty
            .as_mut()
            .ok_or_else(|| RenderError::Device("linear beauty missing".into()))?;
        cuda.module
            .launch_untile_normalize(&stream, &ct.accum, linear_beauty, w, h, true)?;

        let use_albedo = settings.mode != Mode::Color;
        let use_normal = settings.mode == Mode::ColorAlbedoNormal;
        if use_albedo {
            ensure_buf(&mut ct.linear_albedo, "linear albedo")?;
            let linear_albedo = ct
                .linear_albedo
                .as_mut()
                .ok_or_else(|| RenderError::Device("linear albedo missing".into()))?;
            cuda.module
                .launch_untile_normalize(&stream, &ct.albedo, linear_albedo, w, h, false)?;
        }
        if use_normal {
            ensure_buf(&mut ct.linear_normal, "linear normal")?;
            let linear_normal = ct
                .linear_normal
                .as_mut()
                .ok_or_else(|| RenderError::Device("linear normal missing".into()))?;
            cuda.module
                .launch_untile_normalize(&stream, &ct.normal, linear_normal, w, h, false)?;
        }

        let linear_beauty_ref = ct
            .linear_beauty
            .as_ref()
            .ok_or_else(|| RenderError::Device("linear beauty missing".into()))?;
        let denoised_beauty_ref = ct
            .denoised_beauty
            .as_ref()
            .ok_or_else(|| RenderError::Device("denoised beauty missing".into()))?;
        let input_img = optix::Image::new(linear_beauty_ref, w, h)
            .map_err(|e| device("OptiX beauty image", e))?;
        let output_img = optix::Image::new(denoised_beauty_ref, w, h)
            .map_err(|e| device("OptiX output image", e))?;
        let albedo_img = if use_albedo {
            let buf = ct
                .linear_albedo
                .as_ref()
                .ok_or_else(|| RenderError::Device("linear albedo missing".into()))?;
            Some(optix::Image::new(buf, w, h).map_err(|e| device("OptiX albedo image", e))?)
        } else {
            None
        };
        let normal_img = if use_normal {
            let buf = ct
                .linear_normal
                .as_ref()
                .ok_or_else(|| RenderError::Device("linear normal missing".into()))?;
            Some(optix::Image::new(buf, w, h).map_err(|e| device("OptiX normal image", e))?)
        } else {
            None
        };
        let guides = optix::Guides {
            albedo: albedo_img,
            normal: normal_img,
            flow: None,
            flow_trust: None,
        };
        let layer = optix::Layer {
            input: input_img,
            output: output_img,
            previous: None,
            aov: optix::AovType::Beauty,
        };
        let denoiser = cuda
            .optix_denoiser
            .as_mut()
            .map(|s| &mut s.denoiser)
            .ok_or_else(|| RenderError::Device("OptiX denoiser missing".into()))?;
        // SAFETY: `linear_beauty`, `linear_albedo`, `linear_normal`, and `denoised_beauty` are
        // owned by `CudaTarget`, sized `w * h`, and remain live on `stream` until synchronized or
        // consumed by subsequent stream-ordered work (`present_shared` / `copy_to_host`).
        unsafe {
            denoiser.invoke(
                &stream,
                &guides,
                &[layer],
                optix::Params {
                    blend: 0.0,
                    flow_mul: [0.0, 0.0],
                },
            )
        }
        .map_err(|e| device("OptiX denoiser invoke", e))?;

        let output = if readback {
            let mut host = vec![[0.0f32; 4]; pixels];
            denoised_beauty_ref
                .copy_to_host(&stream, &mut host)
                .map_err(|e| device("OptiX denoised readback", e))?;
            let latency_ms = t0.elapsed().as_secs_f32() * 1000.0;
            Some(denoise::Output {
                pixels: host,
                latency_ms,
            })
        } else {
            None
        };
        let latency_ms = t0.elapsed().as_secs_f32() * 1000.0;
        Ok(Some(DenoiseOutcome {
            engine: DenoiseEngine::Optix,
            output,
            latency_ms,
        }))
    }

    fn present_shared(
        &mut self,
        backend: Backend,
        film: &mut BackendFilm,
        width: usize,
        height: usize,
        denoised: bool,
    ) -> Result<Option<wgpu::TextureView>, RenderError> {
        if backend != Backend::Cuda || width == 0 || height == 0 {
            return Ok(None);
        }
        let cuda = self.cuda_mut()?;
        let Some(paired) = cuda.paired.as_ref() else {
            return Ok(None);
        };
        let w = u32::try_from(width)
            .map_err(|_| RenderError::Scene("Interop width exceeds u32".into()))?;
        let h = u32::try_from(height)
            .map_err(|_| RenderError::Scene("Interop height exceeds u32".into()))?;
        let ct = film
            .downcast_mut::<CudaTarget>()
            .ok_or_else(film_mismatch)?;

        let cfg = InteropTargetConfig {
            enable_albedo: false,
            enable_normal: false,
            enable_preview: true,
            ..Default::default()
        };
        let interop = match ct.interop.as_mut() {
            Some(interop) => {
                interop
                    .resize(w, h)
                    .map_err(|e| device("interop resize", e))?;
                interop
            }
            None => ct.interop.insert(
                InteropFrameTargets::new(paired, w, h, cfg)
                    .map_err(|e| device("interop targets", e))?,
            ),
        };
        interop
            .wait_cuda_for_wgpu(&cuda.stream, wgpu::TextureUses::RESOURCE)
            .map_err(|e| device("wait_cuda_for_wgpu", e))?;
        let preview = interop
            .preview()
            .ok_or_else(|| RenderError::Device("Interop preview surface missing".into()))?;
        let surf_obj = preview.surface_handle();
        let (src_ptr, is_tiled) = if denoised {
            let buf = ct
                .denoised_beauty
                .as_ref()
                .ok_or_else(|| RenderError::Device("VRAM denoised beauty buffer missing".into()))?;
            (buf.cu_deviceptr(), false)
        } else {
            (ct.accum.cu_deviceptr(), true)
        };
        cuda.module.launch_tonemap_to_surface(
            &cuda.stream,
            src_ptr,
            surf_obj,
            w,
            h,
            is_tiled,
            cuda.last_exposure,
            cuda.last_saturation,
            cuda.last_reinhard,
            cuda.last_luma,
        )?;
        interop
            .wait_wgpu_for_cuda(&cuda.stream)
            .map_err(|e| device("wait_wgpu_for_cuda", e))?;
        let view = interop
            .preview()
            .ok_or_else(|| RenderError::Device("Interop preview surface missing".into()))?
            .wgpu_view()
            .clone();
        Ok(Some(view))
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
                let cuda = self.cuda_mut()?;
                if ct.out.len() != pixels {
                    ct.out = zeroed(&cuda.stream, pixels, "output")?;
                }
                let n = u32::try_from(pixels).map_err(|_| {
                    RenderError::Scene("Render target pixel count exceeds u32".into())
                })?;
                cuda.module.launch_slices(
                    &cuda.stream,
                    "tonemap",
                    n.div_ceil(BLOCK),
                    BLOCK,
                    &[
                        (ct.accum.cu_deviceptr(), ct.accum.len()),
                        (ct.out.cu_deviceptr(), ct.out.len()),
                    ],
                )?;
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
                        ref mut module,
                        ref objects,
                        ref lights,
                        ref mut pick_query,
                        ref mut pick_out,
                        ..
                    } = *cuda;
                    module.set_params_async(stream, params)?;
                    pick_query
                        .copy_from_host(stream, &[[ndc[0], ndc[1], 0.0, 0.0]])
                        .map_err(|e| device("pick upload", e))?;
                    pick_out
                        .zero_async(stream)
                        .map_err(|e| device("clear pick result", e))?;
                    macro_rules! pick_entry {
                        ($prepare:ident, $run:ident) => {{
                            module.launch_slices(
                                stream,
                                stringify!($run),
                                1,
                                32,
                                &[
                                    (objects.cu_deviceptr(), objects.len()),
                                    (lights.cu_deviceptr(), lights.len()),
                                    (pick_query.cu_deviceptr(), pick_query.len()),
                                    (pick_out.cu_deviceptr(), pick_out.len()),
                                ],
                            )?;
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
        let wb = (&mut *self.engine as &mut dyn std::any::Any)
            .downcast_mut::<WarpBroComputeEngine>()
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
