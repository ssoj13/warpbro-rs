# BUG1: Unified `wgpu` / `cuda` Compute Backend Architecture (Parity with `ofx-rs`)

## Problem Statement

`WarpBro` currently couples its entire build and tracing pipeline to `cuda-oxide` (`rustc_codegen_cuda`) and `cuda-core`, while simultaneously running OIDN denoising (`pt-denoise-oidn`), colour management (`vfx-ocio`), and window presentation on `wgpu`:

1. **Hard build-time lock to `cargo-oxide` (`build.rs`, `Cargo.toml`)**
   - `fractal-kernels` is linked directly as a `#[cuda_module]` crate, requiring `rustc_codegen_cuda`, CUDA 13+, and LLVM/`llc` 21+ for every binary build and test run (`build.rs:16-29`).
   - Unlike `ofx-rs` (where ordinary builds use standard `cargo build` and load CUDA dynamically or fall back to `wgpu`), `WarpBro` cannot be built or run on machines without the full `cuda-oxide` toolchain and an NVIDIA GPU.

2. **Monolithic `src/render.rs` (4,272 lines)**
   - `src/render.rs` interleaves three distinct responsibilities in `Gpu` and `Target`:
     1. **Scene & parameter preparation** (`PreparationInputs`, `PreparationCache`, `Resident`, `fractal_core::params` packing, palette LUTs, HDRI environment importance CDFs).
     2. **CUDA execution** (`CudaContext`, `CudaStream`, `DeviceBuffer`, `kernels::LoadedModule`, `tally`, `adapt`, `world` / `world_direct` / specialized kernel launches, `pick`).
     3. **Post-processing & display orchestration** (`denoise::Processor`, `color::ColorPipeline`, SDR/HDR readback).

3. **Redundant GPU ↔ CPU ↔ GPU memory transfers (`README.md:532-535`)**
   - Every progressive viewport update or denoised pass copies float buffers across APIs:
     `CUDA DeviceBuffer (accum, albedo, normal)` → **CPU readback (`DenoiseScratch` / `raw`)** → **`wgpu` upload (`gpu_info::shared_device()` for OIDN + `vfx-ocio`)** → **CPU readback (`light`, `pixels`)** → **`wgpu` presentation upload (`src/window.rs`)**.
   - When running on a `wgpu` compute backend sharing `gpu_info::shared_device()`, accumulator and guide buffers already reside on the same `wgpu::Device` as `pt-denoise-oidn` and `vfx-ocio`, allowing an in-device GPU pipeline up to final frame readback/presentation.

---

## Reference Pattern in `ofx-rs`

`ofx-rs` (`crates/ofx-gen`, `crates/ofx-cuda`, `crates/ofx-fractal`) already solves multi-backend execution for the shared `fractal-rs` / `render-rs` ecosystem:

| Layer | `ofx-rs` Implementation | How it applies to `WarpBro` |
| --- | --- | --- |
| **Backend Contract** | `Backend { Auto = 0, Cpu = 1, Wgpu = 2, Cuda = 3 }` (`crates/ofx-gen/src/ofx_gen.rs:282-288`). `Auto` selects `Cuda` when a supported NVIDIA driver (`>= 13.0`) and device are available, otherwise `Wgpu`. Explicit backend selection never silently falls back on error. | Add `Backend { Auto, Cuda, Wgpu }` (and optional `Cpu` for oracle tests) to `RenderSettings` (`src/render_profiles.rs`). |
| **CUDA Runtime** | `ofx-cuda` (`crates/ofx-cuda/src/{lib.rs,driver.rs}`) loads `nvcuda` dynamically via `cudarc` (`dynamic-loading`, `cuda-13030`), checks `MIN_DRIVER = 13000`, preserves caller `CUcontext` (`capture_context` / `restore_context`), caches allocations in `IdlePool`, and loads embedded `Artifact { ptx, cubin, kernels }`. | Allows decoupling default Cargo builds from `cargo-oxide` when using pre-built/embedded fatbin+PTX artifacts, while keeping `cargo oxide` for kernel development. |
| **WGPU Runtime** | `ofx-gen::device` (`crates/ofx-gen/src/device.rs`) wraps `gpu_info::compute_device()` / `shared_device()` with `OutOfMemory` + `Validation` error scopes (`scoped()`) and a thread-safe single-flight compute pipeline cache. | `WarpBro` already links `gpu-info-rs` (`features = ["wgpu"]`) and initializes `gpu_info::shared_device()` in `src/denoise.rs:374` and `vfx-ocio` in `src/color.rs:151`. |
| **WGSL Kernels** | `ofx-fractal` (`crates/ofx-fractal/src/{lib.rs,uniform3d.rs,pathtrace.rs}`) composes WGSL shaders with `standard-surface-bsdf`, `pt-integrator`, and `fractal-core` parameter slots, validated offline with `naga`. | Port/share the 3D WGSL world tracer, direct shading (`world_direct`), `pick`, `tally`, and `adapt` compute shaders against `fractal_core::params`. |

---

## Architecture & Implementation Plan

### Phase 1: Split `src/render.rs` into Preparation, Backend Trait, and Post-Processing

Extract backend-independent logic out of `src/render.rs` so CUDA and WGPU share a single preparation and post-processing path:

1. **`src/render_prep.rs` (Backend-agnostic preparation)**
   - Move `PreparationInputs`, `PreparationStats`, `PreparationCache`, `Resident<K>`, `same_material`, `preparation_render`, parameter array packing (`[f32; P_COUNT]`), object/mask/light table packing (`Vec<f32>`), palette LUT building (`build_lut`), and environment map decoding (`BTreeMap<(String, u64), ...>`).
   - Output a plain host diff/update struct (`PreparedScene`) describing which slots (`params`, `lut`, `objects`, `lights`, `env`) changed since the last launch.

2. **`src/render_backend.rs` (Compute backend interface)**
   - Define `Backend`:
     ```rust
     #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
     pub enum Backend {
         #[default]
         Auto,
         Cuda,
         Wgpu,
     }
     ```
   - Define a worker-owned backend abstraction (`ComputeBackend` / `ComputeTarget`) for:
     - `upload_prepared(&mut self, prepared: &PreparedScene) -> Result<(), String>`
     - `create_target(&self, width: usize, height: usize) -> Result<TargetBuffers, String>`
     - `reset_target(&self, target: &mut TargetBuffers) -> Result<(), String>`
     - `step_batch(&mut self, target: &mut TargetBuffers, launch: &BatchLaunch) -> Result<BatchStats, String>` (runs trace + `tally` + `adapt` + tonemap/normalize)
     - `pick(&mut self, prepared: &PreparedScene, uv: [f32; 2]) -> Result<Option<u32>, String>`
     - `read_accum_normalized(&self, target: &TargetBuffers, raw: &mut [[f32; 4]], albedo: &mut [[f32; 4]], normal: &mut [[f32; 4]]) -> Result<(), String>`

3. **`src/render_cuda.rs` (Existing CUDA implementation)**
   - Move `CudaContext`, `CudaStream`, `kernels::LoadedModule`, and `DeviceBuffer` management from `src/render.rs` into `CudaBackend` without changing kernel launch semantics or PTX generation.

---

### Phase 2: WGPU Compute Backend (`src/render_wgpu.rs`)

1. **Shared `wgpu::Device` Reuse (`gpu_info::shared_device()`)**
   - Acquire the same headless `gpu_info::shared_device()` already used by `src/denoise.rs` (`Processor`) and `src/color.rs` (`ColorPipeline`).
   - Wrap compute dispatches in `wgpu::ErrorFilter::OutOfMemory` and `wgpu::ErrorFilter::Validation` error scopes (as in `ofx_gen::device::scoped`).

2. **Target Storage Buffers & 8×4 Tile Layout**
   - Match the 8×4 tile layout (`padded = width.div_ceil(8) * height.div_ceil(4) * 32`) used by CUDA:
     - `accum`, `albedo`, `normal`, `stats`: `wgpu::Buffer` (`STORAGE | COPY_SRC | COPY_DST`).
     - `active`: `wgpu::Buffer` (`u32` per 8×4 tile).
     - `tally`: `wgpu::Buffer` (atomic/workgroup reduction of `[unresolved, samples]`).
     - Uniform / storage buffers for `params` (`[f32; P_COUNT]`), `lut`, `objects`, `lights`, and `environment`.

3. **WGSL Compute Pipelines**
   - Compose WGSL modules using `fractal-core` parameter offsets and `render-rs`'s `standard-surface-bsdf` WGSL library (following `ofx-fractal/src/uniform3d.rs`):
     - `world_trace` / `world_direct` / specialized family entry points.
     - `adapt` (8×4 tile variance threshold check against `stats` and `accum`).
     - `tally` (workgroup reduction of unresolved march counts).
     - `pick` (single-ray scene intersection returning hit object index/distance).
   - Add a `naga` validation unit test in `dev-dependencies` (like `ofx-fractal`) so WGSL syntax and binding layouts are verified even on headless CPU-only CI runners.

4. **Zero-Copy In-Device Handoff to OIDN and OCIO**
   - When `WgpuBackend` is active, keep `accum`, `albedo`, and `normal` on `gpu_info::shared_device()` so normalization, `pt-denoise-oidn`, and `vfx-ocio` `GpuBufferExecutor` can operate without intermediate host roundtrips where supported.

---

### Phase 3: Optional / Dynamic CUDA Loading (Build Decoupling like `ofx-rs`)

To allow building and developing `WarpBro` with plain `cargo build` when `cuda-oxide` is not invoked:

1. **Feature-gate live `cuda-oxide` compilation vs embedded artifact**
   - Introduce Cargo features:
     - `wgpu` (enabled by default).
     - `cuda-oxide` (compiles `fractal-kernels` from Rust source via `rustc_codegen_cuda`, used by `python bootstrap.py b`).
     - Optional `cuda-runtime` (loads pre-built fatbin/PTX via `cudarc` `dynamic-loading`, matching `ofx-cuda`).
   - Update `build.rs` so plain `cargo build` succeeds when building with `wgpu` (or `cuda-runtime`), and only panics when `cuda-oxide` is active without `rustc_codegen_cuda` in `CARGO_ENCODED_RUSTFLAGS`.

---

### Phase 4: `RenderSettings`, UI, and Worker Routing

1. **Graph & Settings Schema (`src/render_profiles.rs`, `src/world.rs`)**
   - Add `/render/backend` (`Auto`, `Cuda`, `Wgpu`) to `RenderSettings` nodes (`Moving`, `Still`, `Manual`, `Output`).
   - Expose **Backend** in the Attribute Editor (`Render` section) and status bar (displaying active resolved backend, e.g. `CUDA (RTX 3080 Ti)` vs `WGPU (DX12 / Vulkan)`).
   - Follow the `ofx-rs` rule:
     - `Backend::Auto`: prefers `Cuda` if initialization succeeds, otherwise selects `Wgpu`.
     - `Backend::Cuda` or `Backend::Wgpu`: if the explicitly requested backend fails or is unavailable, surface the exact diagnostic error in the viewport/status bar and refuse silent fallback.

2. **Worker Switching (`src/render_service.rs`)**
   - Hold both lazily initialized backends (`Option<Result<CudaBackend, String>>` and `Option<Result<WgpuBackend, String>>`) inside the render worker.
   - Switching `/render/backend` invalidates the target's accumulation key and dispatches on the selected backend.

---

## Verification Plan

1. **Shader Validation (No GPU required)**:
   - `naga` unit test validating all composed WGSL entry points (`world`, `world_direct`, `adapt`, `tally`, `pick`).
2. **Backend Resolution & Error Contract Tests**:
   - Verify `Backend::Auto` selection and verify that forced `Backend::Cuda` / `Backend::Wgpu` errors are reported verbatim without silent fallback.
3. **CUDA vs WGPU Numerical Parity (`cuda_*` and `gpu_*` tests)**:
   - Compare single-bounce `world_direct`, `pick`, `tally`, `adapt`, and multi-bounce `world` radiance + albedo/normal guides between `CudaBackend` and `WgpuBackend` within f32 tolerance on fixed seeds.
4. **Throughput & Convergence Benchmark**:
   - Run `--world-bench` across `fast-metal`, `fast-dielectric`, and `diffuse-mid` on both `CUDA` and `WGPU` backends and record timing/MSE tables.

---

## Implementation Status (2026-10-17)

- **Scene Preparation Split (`src/render_prep.rs`)**: Extracted backend-agnostic parameter packing, object/light/mask table construction, palette LUT generation, HDRI environment importance sampling CDFs, and `PreparationCache` / `Resident<K>` diffing from `src/render.rs`.
- **WGPU Compute Backend (`src/render_wgpu.rs`, `src/render_wgpu.wgsl`)**: Implemented `WgpuEngine` and `WgpuTarget` sharing `gpu_info::shared_device()` with `OutOfMemory` and `Validation` error scopes (`with_error_scopes`), 8×4 tile storage buffers, and full WGSL compute kernels (`world`, `world_fast`, `world_fast_bulb`, `world_direct`, `fast_<family>`, `full_<family>`, `direct_<family>`, `tally`, `adapt`, `tonemap`, `pick`).
- **Backend Contract & UI Wiring (`src/render_profiles.rs`, `src/scene.rs`, `src/world.rs`, `src/world_ui.rs`, `src/render_profiles_ui.rs`, `src/render_service.rs`, `src/app.rs`, `src/render.rs`)**: Added `Backend { Auto, Cuda, Wgpu }` wired to `/render/backend`, included backend in `RenderService::trace_key`, surfaced active backend in the status bar (`CUDA · <gpu>` / `WGPU · <gpu>`), and enforced strict no-silent-fallback when `Backend::Cuda` or `Backend::Wgpu` is explicitly requested.
