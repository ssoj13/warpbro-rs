# WarpBro PLAN

The authoritative working document: rules, state, open work, durable findings. Keep it current:
drop done items (details go to CHANGELOG.md, history stays in git), add new requests, keep open work
sorted by priority. Code, comments and documentation in English; chat in Russian. Reusable UI and
studio infrastructure live in `soluwell-studio` (`egui-widgets-rs/crates/soluwell/studio`). Topic
docs live in `docs/`; this file tracks architecture, open work, and technical invariants.

---

## Rules that bite

- **Plan first**: wait for approval before non-trivial changes. Choose systemic fixes, SSOT, zero
  hacks, no backward-compatibility shims, no silent fallbacks; verify every claim.
- **Never `git reset` / checkout-rollback**: restore a file with `git show <rev>:<path> > <path>`.
- **No `Co-Authored-By` lines**: commit only your own files. Parallel sessions and WIP auto-commits
  share these repos. `cargo fmt` may touch foreign code: inspect `git diff -U0`.
- **One independent review per finished branch**: plus visual check of screenshots; no review per
  small fix. Tests sparingly: analyse, trace blast radius, compile once, targeted test at the end.
- **Atomic push to branch AND `main`**: own deps via SSH on `main`, third-party HTTPS, no `[patch]`.
  Bump an own git dep by editing the lock's rev by hand (never `cargo update -p` which downgrades
  `windows-sys`), then build `--locked`.
- **Shared machine**: every build / check / test through
  `python C:/projects/projects.rust.cg/cglibs/.buildlock/buildlock.py <cmd>` (never nested), only
  after edits are finished. CUDA / WGPU tests and benchmarks only with the operator's approval.
- **Build and test only through cuda-oxide**: `python bootstrap.py b`, `cargo oxide test -- --release
  --locked -- <filter>` (tests need `--release`); type-check with
  `FRAC_ALLOW_PLAIN_CARGO=1 cargo clippy --all-targets --locked`.
- **Strict test naming**: a test requiring a CUDA device is named `cuda_*`, one requiring another GPU
  API (wgpu, Vulkan Video) `gpu_*`, and no other test name may contain these (`test_gpu.rs`, enforced by
  `device_prefixes_only_start_test_names`). The GPU-free suite runs with `--skip cuda_ --skip gpu_`.
- **Zero legacy compatibility**: no legacy loaders, serde aliases or compat migrations. Convert
  checked-in or operator files once with a throwaway script and backup.
- **Reference code archive**: `D:\Projects\vfx.ref`. Shared CG crates: `C:\projects\projects.rust.cg\cglibs`.

---

## Current Architecture & State (`5afff8a`)

WarpBro is an egui desktop browser and animator for path-traced 3D fractals (28 DE families) built on
top of the shared `soluwell-studio` widget (`egui-widgets-rs/crates/soluwell/studio`):

- **Domain Separation**: `soluwell-studio` owns the dock panels (Outliner, Timeline, Attribute Editor,
  Curve Editor), Playa `EventBus` dispatch, loopback HTTP REST API, `WorldDocument` graph/undo,
  `RenderService` worker, `PreparationCache`, OCIO display pipeline, OIDN denoiser,
  `PreviewController`, `ThumbnailCache`, and `ExportController`.
- **Fractal Registration**: WarpBro registers:
  1. `"Fractal"` scene node (`src/fractal_node.rs`, `FractalNodeSpec` implementing `NodeTypeSpec`).
     Evaluates into `scene.objects` with 28-variant `Formula` and `pack_object` (`P_COUNT` 128-float ABI).
  2. Pluggable `WarpBroComputeEngine` (`src/engine.rs` implementing `soluwell_studio::render::ComputeEngine`):
     - **CUDA**: 234 compile-time specialized kernel entries from `fractal-rs` (`fractal-kernels`),
       compiled to PTX by `cuda-oxide` and linked via `#[cuda_module]`.
     - **WGPU**: 28 per-family WGSL shader modules + 1 dynamic 28-way module (`src/formulas.wgsl`
       injected into `soluwell-studio-render` via `render_wgpu::set_custom_shader_hook`).
  3. Built-in templates (`src/templates.rs`, 14 templates) and procedural animated presets
     (`src/presets.rs`, `ANIMATED = 0..5`), plus 38 curated gallery scenes (`Scene::fractal_gallery`).
- **Inline Compute Backend Selection**: `Backend { Auto, Cuda, Wgpu }` selectable per render binding
  (`Moving`, `Still`, `Manual`, `Output`) in `Settings -> Render & Viewport`, the viewport toolbar profile
  dropdown, and `Render / Encode`. Backed by `/render/backend` on `RenderSettings` nodes with single-step Undo.
- **Central EventBus Dispatch**: UI widgets, keyboard shortcuts, and the loopback REST API publish typed
  `app_control::Event<UiCommand>` into App's Playa `EventBus`. Central deferred dispatch enforces document,
  frame, active camera, and revision guards before mutation. `emit_retained` prevents dropped terminal DTOs.
- **Content-Addressed Thumbnail Cache**: `<profile>/cache/thumbnails` (256 MiB budget), validated by binary
  headers and SHA-256 checksums, content-addressed by packed scene ABI, quality, backend, SPP, seed,
  environment map, OCIO processor, and source fingerprint.
- **Strict Document Decode**: `WorldDocument::decode_json` strictly verifies node shape, format, and attributes
  without legacy migrations.
- **Pinned Dependencies**:
  - `cuda-oxide`: `CUDA_OXIDE_REV = 9370f8c91eba499a880a5769ecd11f39fa65ffbb`.
  - `fractal-rs`: `5a21813e0f8ae8f4d49a423d56564f84e5629dfd` (all 28 families on CUDA, 234 kernel entries,
    conservative `cone_start` beam-root).
  - `ffmpeg-rs` / OxideAV: git dependencies locked in `Cargo.lock` (`ssoj13/oxideav-av1`,
    `ssoj13/oxideav-vp9`, `ssoj13/oxideav-core`).
  - `gpu-info`: `f845ac889738e159191eb9f71cfabc8831ce134f` (persistent pipeline cache fix).
  - `OIDN`: `pt-denoise-oidn` `HostDenoiser::new(ctx, &work)` on `gpu_info::WorkContext`.

---

## Open Work (Priority Order)

### Section A: CUDA / WGPU 28-Family Parity & Kernel Performance

1. **Parity suite execution on real GPU (`cuda_*` / `gpu_*`)**:
   - Run `parity_tests::cuda_wgpu_primary_guides_agree_per_pixel` on real hardware across all 28 families;
     measure and lock down tolerances from actual empirical distributions instead of guesses.
   - Run the revised numerical QA contract (`cuda_extended_full_de_short_orbits_and_noniterating_traps_match_wgpu`),
     verifying full DE for all cases, traps at iterations 1..3 for iterative families, and full traps for
     non-iterative families.
   - Verify specific corrected formulas on GPU:
     - Family 12 (Polyhedral KIFS): verify golden-ratio vertex offset fold produces $\ge 10\%$ coverage on both backends.
     - Families 13 & 22: verify stable `atan2` polar angles eliminate normalization error amplification at poles.
     - Family 19: verify analytic closest-branch resolution at mathematical distance ties.
     - Family 23 (Phyllotaxis): verify corrected inverse Fibonacci basis matches exhaustive nearest-seed oracle.
     - Conservative ray-beam bounding-sphere entry (`cone_start`): run the 5 CPU fixtures and device checks on
       diagnostic pixel `[62,6]` with depth-of-field guards.
2. **Zero-spill compiler verification & launch bounds tuning**:
   - `ptxas` / Nsight Compute (NCU) analysis: investigate register pressure under `#[launch_bounds(128)]`.
     The baseline `world_full_f9` (Bristorbrot) on sm_86 reported 255 registers per thread and 2.88 MB of
     local-memory register spill traffic.
   - Separate array/call stack frame allocations from true register spills; verify zero spill stores and
     spill loads across all 234 generated kernel entries.
   - Tune launch geometry (`launch_bounds`, thread block sizing) and reduce live ranges across ray marching,
     DE loops, and shading to eliminate spilling without sacrificing occupancy.
3. **Warm steady throughput & A/B benchmarks**:
   - Interleaved CUDA vs WGPU benchmarks via `--world-bench --backend <backend>` with 2 seconds of untimed
     warmup across all 20 extended families and 8 classic families.
   - Measure and record median timings, variation, active warps, and memory throughput on clean GPU state.

---

### Section B: Render Pacing & GPU Display Path (Stages 0..4)

4. **Stage 0: Per-stage timing instrumentation & headless viewport benchmark**:
   - Instrument the render worker: submit time, GPU time (CUDA events / wgpu timestamps), host wait duration,
     readback bytes and bandwidth, OCIO execution, and UI upload time via `tracing` spans.
   - Add a headless "viewport pattern" to `--world-bench` (display every 1/60 s, adaptive sampling active,
     periodic camera navigation resets) to evaluate interactive pacing without GUI window overhead.
   - Measure baseline performance on an idle machine across CUDA and WGPU at 1080p and 1440p; enforce golden
     check that accumulated film after $N$ samples is bit-identical across pacing changes.
5. **Stage 1: Two-batch in-flight ring buffer & true GPU cost pacing**:
   - Replace the synchronous stop-gap wait (`Gpu::stamp`) with a two-deep ring buffer:
     record a GPU fence after each batch (CUDA event / wgpu submission index); wait for batch $N$'s fence
     only before submitting batch $N+2$. The GPU queue never starves, while host latency remains $\le 2$ batches.
   - Size batches via `Batch::Budget` using true GPU execution time (non-blocking query of completed event
     `elapsed_ms` / timestamp queries), eliminating CPU submit-time skew.
   - Decouple viewport budget (frame budget minus measured display latency) from export budget (large batches
     bounded safely below Windows TDR limit of 1–2 s).
6. **Stage 2: Asynchronous adaptive sampling & tally readbacks**:
   - Keep adaptive tile flags and sample tally on the GPU; perform asynchronous small readbacks (CUDA: async
     D2H into pinned host memory + event; WGPU: `map_async` + `PollType::Poll`).
   - Consume flags one batch later in the scheduler: at most one extra batch past convergence, zero per-step
     host stalls.
7. **Stage 3: GPU display path & fused OCIO**:
   - WGPU film: fuse tonemapping and OCIO into a single compute pass (`OcioShader` / `GpuTextureExecutor`),
     writing linear display light and SDR codes directly into a shared texture; read back at most once per
     displayed UI frame (RGBA16F for HDR, 8-bit for SDR).
   - CUDA film: asynchronous D2H of tonemapped film into pinned double buffers overlapped with the next ray-march
     batch; execute OCIO on the shared wgpu device texture-to-texture (`GpuTextureExecutor`). Coordinate with
     `oiio-rs` CUDA OCIO on the film (`plan-cuda-oxide.md`).
8. **Stage 4: Zero-copy present**:
   - Render (WGPU film), OCIO, and the window presentation texture on a single shared wgpu device: compute pass
     writes directly to the texture egui renders, eliminating GPU-to-CPU-to-GPU readback cycles.
   - For CUDA film: evaluate CUDA <-> Vulkan external memory interop (`wgpu-hal` level) if pinned host staging
     proves to be the measured bottleneck.

---

### Section C: Document & Error Propagation Hardening

9. **Complete `.ok()` sweep part 2 (`nodes() -> Result`)**:
   - Audit remaining `.ok()` conversions in document traversal: `WorldDocument::nodes_at` drops a node whose
     `info_at` fails, and `info_at_with` silently defaults flags whose values fail.
   - Convert `nodes()` and related traversal methods across `soluwell-studio-core` to return `Result<...>`
     so data corruptions or invalid node states surface as typed errors rather than vanishing.
   - Audit error handling in `warpbro_dir()` fallback, `preview.rs` `frame_count().ok()`, and `exr_io.rs`
     header inspection.
10. **Operator decisions**:
    - **Q2: Attribute connections (`conn`)**: evaluated in 7 graph locations but currently authored only by
      tests. Operator decision: either complete a first-class "Connect Attribute" UI in the Attribute Editor
      or cleanly remove connection resolver branches.
    - **Q4: `path_convergence.rs`**: uncommitted prototype for firefly AP1 luminance clamping and Russian
      roulette. Operator decision: evaluate on GPU with `--world-bench` + `tools/convergence.py`, migrate into
      `fractal-core`, or remove.

---

### Section D: Rendering Backlog & UI Polish

11. **Dielectric multiple-scattering energy compensation**:
    - `standard-surface-bsdf` in `render-rs` currently lacks multiple-scattering compensation for rough
      dielectric transmission (furnace test drops to 0.81–0.93 at roughness 0.5).
    - Plan: furnace grid baseline $\rightarrow$ reference tables (Turquin 2019, Cycles multiscatter GGX glass,
      OpenPBR) $\rightarrow$ compensation LUT with Rust + WGSL parity $\rightarrow$ furnace test verification
      ($\ge 0.99$).
12. **Hybrid DE surface transitions**:
    - Address residual fold/escape boundary discontinuities and 6-probe normal evaluation cost; profile
      over-relaxed sphere tracing with Lipschitz safety bounds.
13. **Render / Encode colour output consistency**:
    - Standardize output space options across EXR (ACEScg / ACES2065-1 / linear Rec.709 / Rec.2020),
      PNG (SDR / HDR10 / HLG), and MP4 (SDR / HDR10).
14. **In-process HDR video via ffmpeg-rs**:
    - Replace the temporary external ffmpeg CLI sequence encoder (`PngVideo`) with native in-process encoding
      via `ffmpeg-rs` once the OxideAV encoder matures.
15. **GPU test isolation**:
    - Resolve Vulkan Video export test concurrency collision ("Posix(38)") when running alongside the render worker.
16. **Slider range tuning & visual verification**:
    - Review soft slider bounds (aperture, focus, orbit speed, thin film, sun angle, hit epsilon) with operator.
    - Native verification pass: High-DPI scaling (100%/150%), drag-and-drop layer reordering, material library
      layout, HDR playback cache, keyed camera flight.

---

## Key Files

### WarpBro Domain Files (`cglibs/warpbro-rs/src/`)

| File | Role |
|---|---|
| `src/main.rs` | Entry point, CLI flag dispatch (`--gallery`, `--bench`, `--world-bench`, `--warmup`), `soluwell_studio::run_with(StudioConfig)` |
| `src/engine.rs` | `WarpBroComputeEngine`, `init_warpbro()`, dual CUDA + WGPU dispatch, custom shader hook injection |
| `src/fractal_node.rs` | `"Fractal"` `NodeTypeSpec`, 28-variant `Formula`, `pack_object` (`P_COUNT` ABI), `fractal_gallery()` |
| `src/formulas.rs` | Declarative metadata table `FORMULAS: [FormulaMeta; 28]`, `EXT_SLOTS`, parameter definitions |
| `src/formulas.wgsl` | WGSL distance estimators, raymarching, normal evaluation, and materials for all 28 families |
| `src/templates.rs` | 14 built-in world templates (`BUILTINS`) |
| `src/presets.rs` | 6 procedural 250-frame animated world templates (`ANIMATED`) |
| `src/render_bench.rs` | Headless evaluated world and family A/B benchmarks |
| `src/parity_tests.rs` | Numerical and rendered parity suite between CUDA and WGPU across all 28 families |
| `src/test_gpu.rs` | Device test harness and strict prefix validator (`cuda_` / `gpu_`) |

### Shared Studio Infrastructure (`egui-widgets-rs/crates/soluwell/studio/`)

| Subsystem | Role |
|---|---|
| `studio/src/app.rs` | Main application shell, window lifecycle, frame event loop, layout persistence |
| `studio/src/world.rs` | `WorldDocument`, `WorldEditor`, graph commands, undo/redo stack, evaluated scene snapshots |
| `studio/src/render_service.rs` | Dedicated `RenderService` worker, `Gpu` lifecycle, batch budget pacing, OIDN integration |
| `studio/src/render_prep.rs` | `PreparationCache`, parameter packing, mask arrays, light uploads |
| `studio/src/color.rs` | OpenColorIO (OCIO) pipeline, display transforms, tonemapping, HDR/EDR views |
| `studio/src/export.rs` | Multi-threaded export coordinator and video/still image writers |
| `studio/src/preview.rs` | Timeline RAM preview caching controller |
| `studio/src/thumbnail_cache.rs` | Content-addressed disk cache for gallery, bookmark, and material thumbnails |
| `studio/src/app_control/` | Playa `EventBus` integration, central deferred action dispatch, loopback HTTP REST API |

---

## Findings Worth Keeping

### WGPU Trace Code Size & Call-Site Reduction (2026-10-23)
- Naga and GPU drivers inline distance estimators at every call site. In the unoptimized trace pipeline,
  169 call sites per trace entry expanded to ~110k statements, causing 15+ minute driver compilation stalls.
- Restructuring loops in `formulas.wgsl` so each loop has exactly ONE estimator call site (`march` unified
  phase loop, 2-pass `march_leaving`, 6-tap normal stencil loop, unified `march_sample`) reduced inlined
  estimator copies from 169 to 22 in `trace_pixel` and from 18 to 3 in `pick_ray`, cutting driver compile times
  to seconds.

### GPU Pipeline Disk Cache Persistence (2026-10-24)
- Driver-side Vulkan pipeline cache persistence was blocked by a header validation mismatch. Pinning `gpu-info`
  to `f845ac8` corrected disk cache publication after pipeline creation: trace pipelines (`world_fast_f25`,
  `world_full_f26`) prepare in 50–60 ms on warm restarts compared to 65+ s cold compiles.

### EventBus Deferred Dispatch & Terminal Backpressure (2026-10-25)
- UI widgets, hotkeys, and REST API commands share App's Playa `EventBus` (`try_emit`).
- Central deferred dispatch validates document ID, frame, camera identity, and revision before mutating state,
  preventing stale-context corruption.
- Terminal backpressure handling via `emit_retained` buffers rejected terminal DTOs (scene save, bookmark
  completion, take imports) and retries admission without evicting accepted actions or creating secondary queues.
- Dock layout drawing clones an immutable layout tree so App retains authoritative ownership and avoids
  overwriting restored layouts during dispatch.

### DE Boundary and Inlining (2026-10-09)
- Narrowing the Mandelbulb DE boundary to `inverse_affine` alone delivered ~27% speedup on single-bulb scenes,
  enabled by the `cuda-oxide` fork's `#[inline(never)]` support (`noinline` attribute lowering).
- Specialized versus dynamic worlds contract arithmetic identically; verified via
  `cuda_specialized_world_materials_preserve_radiance_and_affine_guides`.

### Cubin Constant Memory & Whole-Program LTO (2026-10-10)
- In cubin generation (`--materialize-cubin`), libNVVM/nvJitLink whole-program LTO folded `#[constant] PARAMS`
  to zero and stripped the symbol because it was emitted as an unreferenced zero-initialized global.
- Fork fix (`750e60b`): `create_device_global` marks AS4 globals retained and host-written, emitting
  `externally_initialized`.

### Sphere Tracing Normals, Phantom Hits & Glass Energy (2026-10-05)
- **Normal Stencil**: 4-probe tetrahedral normals exhibited direction bias. Replaced with 6 central probes
  at $\epsilon/4$ (median error dropped from 0.45° to 0.016° on converged hits).
- **Phantom Hits**: Rays exhausting their step budget were formerly misidentified as hits. Fixed with
  `Outcome::Unresolved` policy (camera rays miss to background; bounce rays terminate; shadow rays block).
  Default step budget raised to 4096.
- **Glass Energy**: Clear glass at grazing angles lost energy under MaterialX albedo approximations. Fixed
  in `render-rs` with `exact_interface` (albedo = exact Fresnel for smooth interfaces; furnace test 1.00003).

### OxideAV Codec Transition (2026-10-24)
- CPU-based AV1 and VP9 decoders in `ffmpeg-rs` were replaced with spec-compliant OxideAV clean-room decoders
  (`oxideav-av1`, `oxideav-vp9`, `oxideav-core`).
- Dependencies are consumed as SSH git dependencies and locked in `Cargo.lock`. Software decoders return
  `AVERROR(ENOSYS)` until OxideAV wrapper integration is fully complete.
