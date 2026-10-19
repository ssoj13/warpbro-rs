//! WGPU compute backend for WarpBro, sharing `gpu_info::shared_device()` and the 8×4 tiled
//! target contract of `render.rs`.

use std::future::Future;
use std::pin::pin;
use std::sync::{Arc, OnceLock};
use std::task::{Context, Poll, Waker};

use crate::palette::build_lut;
use crate::params::*;
use crate::render_prep::{
    BLOCK, LightResidentKey, LutResidentKey, ObjectResidentKey, PreparedScene, Resident,
    TALLY_THREADS, WorldMaterial, untile_into,
};
use crate::scene::Scene;

const UNIFORM_FLOATS: usize = 256;

/// Assemble the complete WGSL source: `standard_surface_bsdf::wgsl_source()`, the `P_*` / `O_*` /
/// `M_*` constants from `fractal_core::params`, and `render_wgpu.wgsl`.
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
        FAMILY_WORLD,
        WORLD_ABI_VERSION,
        PALETTE_SAMPLES,
    );
    s.push_str(include_str!("render_wgpu.wgsl"));
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

/// Execute `op` inside `OutOfMemory` and `Validation` error scopes on `device`.
pub fn with_error_scopes<T>(device: &wgpu::Device, op: impl FnOnce() -> T) -> Result<T, String> {
    let oom_scope = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let val_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let output = op();
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    if let Some(Some(err)) = poll_ready(val_scope.pop()) {
        let _ = poll_ready(oom_scope.pop());
        return Err(format!("WGPU validation error: {err}"));
    }
    if let Some(Some(err)) = poll_ready(oom_scope.pop()) {
        return Err(format!("WGPU out-of-memory error: {err}"));
    }
    Ok(output)
}

struct WgpuShared {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_name: String,
    module: wgpu::ShaderModule,
    trace_bgl: wgpu::BindGroupLayout,
    post_bgl: wgpu::BindGroupLayout,
    pick_bgl: wgpu::BindGroupLayout,
    trace_layout: wgpu::PipelineLayout,
    world: OnceLock<wgpu::ComputePipeline>,
    world_fast: OnceLock<wgpu::ComputePipeline>,
    world_fast_bulb: OnceLock<wgpu::ComputePipeline>,
    world_direct: OnceLock<wgpu::ComputePipeline>,
    fast_family: [OnceLock<wgpu::ComputePipeline>; 8],
    full_family: [OnceLock<wgpu::ComputePipeline>; 8],
    direct_family: [OnceLock<wgpu::ComputePipeline>; 8],
    tally: wgpu::ComputePipeline,
    adapt: wgpu::ComputePipeline,
    tonemap: wgpu::ComputePipeline,
    pick: wgpu::ComputePipeline,
}

impl WgpuShared {
    fn trace_pipe<'a>(
        &self,
        cell: &'a OnceLock<wgpu::ComputePipeline>,
        entry: &'static str,
    ) -> &'a wgpu::ComputePipeline {
        cell.get_or_init(|| {
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&self.trace_layout),
                    module: &self.module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
        })
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
                        bgl_entry(2, true),
                        bgl_entry(5, false),
                        bgl_entry(8, false),
                        bgl_entry(9, false),
                        bgl_entry(10, false),
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
                    world: OnceLock::new(),
                    world_fast: OnceLock::new(),
                    world_fast_bulb: OnceLock::new(),
                    world_direct: OnceLock::new(),
                    fast_family: std::array::from_fn(|_| OnceLock::new()),
                    full_family: std::array::from_fn(|_| OnceLock::new()),
                    direct_family: std::array::from_fn(|_| OnceLock::new()),
                    tally: make_pipe(&post_layout, "tally"),
                    adapt: make_pipe(&post_layout, "adapt"),
                    tonemap: make_pipe(&post_layout, "tonemap"),
                    pick: make_pipe(&pick_layout, "pick"),
                    module,
                    trace_layout,
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

pub struct WgpuTarget {
    pub padded: usize,
    pub accum: wgpu::Buffer,
    pub albedo: wgpu::Buffer,
    pub normal: wgpu::Buffer,
    pub stats: wgpu::Buffer,
    pub active: wgpu::Buffer,
    pub out: wgpu::Buffer,
    pub staging: wgpu::Buffer,
    bind_groups: std::sync::Mutex<Option<(u64, wgpu::BindGroup, wgpu::BindGroup)>>,
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

    pub fn create_target(&self, width: usize, height: usize) -> WgpuTarget {
        let padded = width.div_ceil(8) * height.div_ceil(4) * 32;
        let tiles = padded / 32;
        let out_len = (width * height).max(TALLY_THREADS);
        let vec4_bytes = std::mem::size_of::<[f32; 4]>() as u64;
        let device = &self.shared.device;
        let rw = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let make_vec4 = |label: &str, count: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (count.max(1) as u64) * vec4_bytes,
                usage: rw,
                mapped_at_creation: false,
            })
        };
        let active = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("warpbro_wgpu_active"),
            size: (tiles.max(1) * std::mem::size_of::<u32>()) as u64,
            usage: rw,
            mapped_at_creation: false,
        });
        let ones = vec![1u32; tiles.max(1)];
        self.shared
            .queue
            .write_buffer(&active, 0, bytemuck::cast_slice(&ones));
        let staging_bytes = (padded.max(out_len).max(1) as u64) * vec4_bytes;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("warpbro_wgpu_staging"),
            size: staging_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        WgpuTarget {
            padded,
            accum: make_vec4("warpbro_wgpu_accum", padded),
            albedo: make_vec4("warpbro_wgpu_albedo", padded),
            normal: make_vec4("warpbro_wgpu_normal", padded),
            stats: make_vec4("warpbro_wgpu_stats", padded),
            active,
            out: make_vec4("warpbro_wgpu_out", out_len),
            staging,
            bind_groups: std::sync::Mutex::new(None),
        }
    }

    pub fn clear_target(&self, wt: &WgpuTarget) {
        let mut encoder =
            self.shared
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("warpbro_wgpu_clear"),
                });
        encoder.clear_buffer(&wt.accum, 0, None);
        encoder.clear_buffer(&wt.albedo, 0, None);
        encoder.clear_buffer(&wt.normal, 0, None);
        encoder.clear_buffer(&wt.stats, 0, None);
        self.shared.queue.submit(Some(encoder.finish()));
    }

    pub fn upload_active(&self, wt: &WgpuTarget, active_host: &[u32]) {
        self.shared
            .queue
            .write_buffer(&wt.active, 0, bytemuck::cast_slice(active_host));
    }

    pub fn set_params(&self, params: &[f32; P_COUNT]) {
        let mut padded = [0.0f32; UNIFORM_FLOATS];
        padded[..P_COUNT].copy_from_slice(params);
        self.shared
            .queue
            .write_buffer(&self.params_buf, 0, bytemuck::cast_slice(&padded));
    }

    pub fn upload_prepared(
        &mut self,
        prepared: &Arc<PreparedScene>,
        env: Option<&crate::environment::Map>,
        world: bool,
        stats: &mut crate::render_prep::PreparationStats,
    ) -> Result<(), String> {
        let device = &self.shared.device;
        let queue = &self.shared.queue;
        let storage_ro = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;

        let table = &mut self.lut_staging;
        let lut_buf = &mut self.lut_buf;
        let lut_len = &mut self.lut_len;
        let buf_gen = &mut self.buf_generation;
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
                let needed = table.len().max(1);
                if *lut_len != needed {
                    *lut_buf = with_error_scopes(device, || {
                        device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some("warpbro_wgpu_lut"),
                            size: (needed * std::mem::size_of::<[f32; 4]>()) as u64,
                            usage: storage_ro,
                            mapped_at_creation: false,
                        })
                    })?;
                    *lut_len = needed;
                    *buf_gen = buf_gen.wrapping_add(1);
                }
                queue.write_buffer(lut_buf, 0, bytemuck::cast_slice(table));
                Ok(())
            })?;
        if changed {
            stats.lut_uploads += 1;
            stats.upload_bytes += (table.len() * std::mem::size_of::<[f32; 4]>()) as u64;
        }

        if world {
            let objects_buf = &mut self.objects_buf;
            let objects_len = &mut self.objects_len;
            let buf_gen = &mut self.buf_generation;
            let data = &prepared.upload.objects;
            let changed =
                self.object_resident
                    .update(ObjectResidentKey(prepared.clone()), || {
                        let needed = data.len().max(4);
                        if *objects_len != needed {
                            *objects_buf = with_error_scopes(device, || {
                                device.create_buffer(&wgpu::BufferDescriptor {
                                    label: Some("warpbro_wgpu_objects"),
                                    size: (needed * std::mem::size_of::<f32>()) as u64,
                                    usage: storage_ro,
                                    mapped_at_creation: false,
                                })
                            })?;
                            *objects_len = needed;
                            *buf_gen = buf_gen.wrapping_add(1);
                        }
                        if !data.is_empty() {
                            queue.write_buffer(objects_buf, 0, bytemuck::cast_slice(data));
                        }
                        Ok(())
                    })?;
            if changed && !data.is_empty() {
                stats.object_uploads += 1;
                stats.upload_bytes += (data.len() * std::mem::size_of::<f32>()) as u64;
            }

            let lights_buf = &mut self.lights_buf;
            let lights_len = &mut self.lights_len;
            let buf_gen = &mut self.buf_generation;
            let data = &prepared.upload.lights;
            let changed = self
                .light_resident
                .update(LightResidentKey(prepared.clone()), || {
                    let needed = data.len().max(4);
                    if *lights_len != needed {
                        *lights_buf = with_error_scopes(device, || {
                            device.create_buffer(&wgpu::BufferDescriptor {
                                label: Some("warpbro_wgpu_lights"),
                                size: (needed * std::mem::size_of::<f32>()) as u64,
                                usage: storage_ro,
                                mapped_at_creation: false,
                            })
                        })?;
                        *lights_len = needed;
                        *buf_gen = buf_gen.wrapping_add(1);
                    }
                    if !data.is_empty() {
                        queue.write_buffer(lights_buf, 0, bytemuck::cast_slice(data));
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

    fn readback_bytes(
        &self,
        src: &wgpu::Buffer,
        staging: &wgpu::Buffer,
        byte_len: u64,
        dst: &mut [u8],
    ) -> Result<(), String> {
        let device = &self.shared.device;
        let queue = &self.shared.queue;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("warpbro_wgpu_readback"),
        });
        encoder.copy_buffer_to_buffer(src, 0, staging, 0, byte_len);
        queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..byte_len);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let mapped = slice
            .get_mapped_range()
            .map_err(|e| format!("WGPU staging map failed: {e:?}"))?;
        dst.copy_from_slice(&mapped[..dst.len()]);
        drop(mapped);
        staging.unmap();
        Ok(())
    }

    pub fn read_tiled_into(
        &self,
        wt: &WgpuTarget,
        src: &wgpu::Buffer,
        (width, height): (usize, usize),
        normalize: bool,
        tiled: &mut Vec<[f32; 4]>,
        output: &mut Vec<[f32; 4]>,
    ) {
        tiled.resize(wt.padded, [0.0; 4]);
        let bytes = (wt.padded * std::mem::size_of::<[f32; 4]>()) as u64;
        self.readback_bytes(src, &wt.staging, bytes, bytemuck::cast_slice_mut(tiled))
            .expect("WGPU tiled readback");
        untile_into(tiled, width, height, normalize, output);
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
                        binding: 2,
                        resource: self.objects_buf.as_entire_binding(),
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
                ],
            })
    }

    fn bind_groups(&self, wt: &WgpuTarget) -> (wgpu::BindGroup, wgpu::BindGroup) {
        let mut guard = wt.bind_groups.lock().expect("bind_groups lock");
        if let Some((cached_gen, trace_bg, post_bg)) = guard.as_ref()
            && *cached_gen == self.buf_generation
        {
            return (trace_bg.clone(), post_bg.clone());
        }
        let trace_bg = self.create_trace_bind_group(wt);
        let post_bg = self.create_post_bind_group(wt);
        *guard = Some((self.buf_generation, trace_bg.clone(), post_bg.clone()));
        (trace_bg, post_bg)
    }

    fn trace_bind_group(&self, wt: &WgpuTarget) -> wgpu::BindGroup {
        self.bind_groups(wt).0
    }

    fn post_bind_group(&self, wt: &WgpuTarget) -> wgpu::BindGroup {
        self.bind_groups(wt).1
    }

    pub(crate) fn launch_trace(
        &self,
        wt: &WgpuTarget,
        scene: &Scene,
        prepared: &PreparedScene,
        full_material: bool,
        force_mixed_world: bool,
    ) {
        const FAST_NAMES: [&str; 8] = [
            "fast_bulb",
            "fast_box",
            "fast_quat",
            "fast_kifs",
            "fast_kleinian",
            "fast_pseudo_kleinian",
            "fast_apollonian",
            "fast_hybrid",
        ];
        const FULL_NAMES: [&str; 8] = [
            "full_bulb",
            "full_box",
            "full_quat",
            "full_kifs",
            "full_kleinian",
            "full_pseudo_kleinian",
            "full_apollonian",
            "full_hybrid",
        ];
        const DIRECT_NAMES: [&str; 8] = [
            "direct_bulb",
            "direct_box",
            "direct_quat",
            "direct_kifs",
            "direct_kleinian",
            "direct_pseudo_kleinian",
            "direct_apollonian",
            "direct_hybrid",
        ];
        let fam = (scene.formula.code() as usize).min(7);
        let pipe = if scene.direct.is_some() {
            if scene.world_render {
                self.shared
                    .trace_pipe(&self.shared.world_direct, "world_direct")
            } else {
                self.shared
                    .trace_pipe(&self.shared.direct_family[fam], DIRECT_NAMES[fam])
            }
        } else if scene.world_render {
            let wm = if force_mixed_world {
                WorldMaterial::Mixed
            } else {
                prepared.world_material
            };
            match wm {
                WorldMaterial::Fast if prepared.single_fast_bulb => self
                    .shared
                    .trace_pipe(&self.shared.world_fast_bulb, "world_fast_bulb"),
                WorldMaterial::Fast => self
                    .shared
                    .trace_pipe(&self.shared.world_fast, "world_fast"),
                WorldMaterial::Full | WorldMaterial::Mixed => {
                    self.shared.trace_pipe(&self.shared.world, "world")
                }
            }
        } else if full_material {
            self.shared
                .trace_pipe(&self.shared.full_family[fam], FULL_NAMES[fam])
        } else {
            self.shared
                .trace_pipe(&self.shared.fast_family[fam], FAST_NAMES[fam])
        };

        let bg = self.trace_bind_group(wt);
        let mut encoder =
            self.shared
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("warpbro_wgpu_trace"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("warpbro_wgpu_trace_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups((wt.padded as u32).div_ceil(BLOCK), 1, 1);
        }
        self.shared.queue.submit(Some(encoder.finish()));
    }

    pub fn tally(&self, wt: &WgpuTarget) -> f32 {
        let bg = self.post_bind_group(wt);
        let mut encoder =
            self.shared
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("warpbro_wgpu_tally"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("warpbro_wgpu_tally_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.shared.tally);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups((TALLY_THREADS as u32).div_ceil(BLOCK), 1, 1);
        }
        self.shared.queue.submit(Some(encoder.finish()));

        let mut sums = vec![[0.0f32; 4]; TALLY_THREADS];
        let bytes = (TALLY_THREADS * std::mem::size_of::<[f32; 4]>()) as u64;
        self.readback_bytes(
            &wt.out,
            &wt.staging,
            bytes,
            bytemuck::cast_slice_mut(&mut sums),
        )
        .expect("WGPU tally readback");
        let (unresolved, samples) = sums.iter().fold((0.0f64, 0.0f64), |(u, s), v| {
            (u + v[0] as f64, s + v[1] as f64)
        });
        if samples > 0.0 {
            (unresolved / samples) as f32
        } else {
            0.0
        }
    }

    pub fn adapt(&self, wt: &WgpuTarget, active_host: &mut [u32]) -> (usize, bool) {
        let tiles = active_host.len() as u32;
        let bg = self.post_bind_group(wt);
        let mut encoder =
            self.shared
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("warpbro_wgpu_adapt"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("warpbro_wgpu_adapt_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.shared.adapt);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(tiles.div_ceil(BLOCK), 1, 1);
        }
        self.shared.queue.submit(Some(encoder.finish()));

        let bytes = std::mem::size_of_val(active_host) as u64;
        self.readback_bytes(
            &wt.active,
            &wt.staging,
            bytes,
            bytemuck::cast_slice_mut(active_host),
        )
        .expect("WGPU active tiles readback");
        let active = active_host.iter().filter(|&&a| a != 0).count();
        (active, active == 0)
    }

    pub fn tonemap(&self, wt: &WgpuTarget, width: usize, height: usize, raw: &mut [[f32; 4]]) {
        let n = (width * height) as u32;
        let bg = self.post_bind_group(wt);
        let mut encoder =
            self.shared
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("warpbro_wgpu_tonemap"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("warpbro_wgpu_tonemap_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.shared.tonemap);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(n.div_ceil(BLOCK), 1, 1);
        }
        self.shared.queue.submit(Some(encoder.finish()));

        let bytes = std::mem::size_of_val(raw) as u64;
        self.readback_bytes(&wt.out, &wt.staging, bytes, bytemuck::cast_slice_mut(raw))
            .expect("WGPU tonemap readback");
    }

    pub fn pick(&self, ndc: [f32; 2]) -> Result<Option<(usize, f32)>, String> {
        let query = [[ndc[0], ndc[1], 0.0f32, 0.0f32]];
        self.shared
            .queue
            .write_buffer(&self.pick_aux, 0, bytemuck::cast_slice(&query));
        let bg = self
            .shared
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
            });
        let mut encoder =
            self.shared
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("warpbro_wgpu_pick"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("warpbro_wgpu_pick_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.shared.pick);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        self.shared.queue.submit(Some(encoder.finish()));

        let mut hit = [[0.0f32; 4]];
        self.readback_bytes(
            &self.pick_aux,
            &self.pick_staging,
            16,
            bytemuck::cast_slice_mut(&mut hit),
        )?;
        let [object, _, depth, _] = hit[0];
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
            "fn world(",
            "fn world_fast(",
            "fn world_fast_bulb(",
            "fn world_direct(",
            "fn tally(",
            "fn adapt(",
            "fn tonemap(",
            "fn pick(",
        ] {
            assert!(src.contains(entry), "missing entry point {entry}");
        }
    }

    #[test]
    fn gpu_wgpu_shader_compiles_without_validation_errors() {
        let _gpu_test = crate::test_gpu::lock();
        let engine =
            WgpuEngine::new().expect("WGPU engine and shader pipelines must compile cleanly");
        assert!(!engine.name().is_empty());
    }
}
