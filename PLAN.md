# WarpBro PLAN

The one working document: rules, state, open work, findings. Keep it current: drop done items
(their details go to CHANGELOG.md, history stays in git), add new requests, keep open work
sorted by priority. Chat in Russian; code, comments and files in English. Reusable UI goes into
egui-widgets-rs. Topic docs (how things work) live in docs/; this file only tracks what is next.

## Rules that bite

- Plan first; wait for approval before non-trivial changes. Systemic fixes, SSOT, no hacks, no
  silent fallbacks; verify every claim (agents included).
- Never `git reset` / checkout-rollback; restore a file with `git show <rev>:<path> > <path>`.
- No `Co-Authored-By` lines. Commit only your own files: parallel sessions and WIP auto-commits
  share these repos. `cargo fmt` may touch foreign code: check `git diff -U0`.
- One independent review per finished branch plus a look at screenshots; no review per small fix.
  Tests sparingly: analyse, then one targeted run at the end.
- Push when told (branch and main). Own deps via SSH on `main`, third-party HTTPS, no `[patch]`.
- Build and test only through cuda-oxide: `python bootstrap.py b`, `cargo oxide test -- --release
  -- <filter>`; type-check with `FRAC_ALLOW_PLAIN_CARGO=1 cargo check --all-targets`.
- A test that needs a CUDA device is named `cuda_*`, one that needs another GPU API (wgpu,
  Vulkan Video) `gpu_*` (src/test_gpu.rs); hosted CI skips both.
- No compatibility with old data: no legacy loaders, serde aliases or migrations. Convert
  checked-in or operator files once with a throwaway script (and a backup).
- References: D:\Projects\vfx.ref.

## State (2026-10-09)

- **Render settings are graph nodes**: RenderSettings, QualitySettings, ViewportSettings and
  OutputSettings (the export recipe), with Profile / Template catalogs; preferences keep only the
  export job (name, range, cadence). Viewport routing has one producer (`App::step_viewport` ->
  `ViewportRender`) and the toolbar shows the active binding. Details: docs/render-profiles.md.
- **cuda-oxide fork** `ssoj13/cuda-rust-windows` `314033b` (renamed from cuda-oxide-windows,
  synced with the NVIDIA/cuda-rust monorepo via ansidium) carries every `#[inline]` intent to LLVM
  (`noinline` included) and pins release codegen in the Cargo profile instead of rustflags, so
  `test -- --release` then `build` compiles 1 crate instead of ~680 (CI built everything twice). cuda-core, cuda-host and cuda-device all come from it; the lock,
  bootstrap's cargo-oxide pin and the installed tool match it. `.cargo/cuda-oxide.toml` fixes
  `default-arch = "sm_75"` for every build and test.
- **DE boundary** narrowed to `inverse_affine`: fast-metal +27.5%, exact routes bit-identical.
- **Lint**: `cargo fmt --check` and `clippy --all-targets -D warnings` pass on the whole crate.
- **CI** (branch `ci/github-actions`, not merged yet): `python bootstrap.py ci` on Windows and
  Linux runners (no GPU): toolchain, fmt + clippy, tests without cuda_/gpu_, release build for
  sm_75, `dist/warpbro-<version>-<platform>.zip`; a `v*` tag publishes both archives. Both
  platforms passed (253 tests each). The repo is public, so runner minutes are free.

## Open work (priority order)

1. **CI finish**: with the profile-pin fork (one compile of the graph per job) confirm both Rust
   caches plus the trimmed LLVM cache fit the 10 GB repo limit (before: Windows 5.8 + Linux 5.1 GB
   holding two variants of every crate), then merge into main.
2. **First launch of a release build JIT-compiles ~9 MB of sm_75 PTX** (minutes on a cold driver
   cache; README tells users to run `WarpBro --warmup-cuda` once). Fix: shipping cubins needs the
   cuda-oxide constant-memory contract (see Findings: cubin) fixed in the fork, then CI
   materializes cubins for the supported architectures. Re-test first: since the 2026-10 upstream
   sync, device globals are emitted as `__device_global_<hash>_N`.
3. **Kernel entry points**: 20 near-identical `#[kernel]` bodies with an 8-parameter launch ABI
   (one documented `#[expect(clippy::too_many_arguments)]`). Generate them from one
   `macro_rules!` and pass a `#[repr(C)]` parameter block (grid constant); embedded PTX must stay
   byte-identical or be re-benchmarked.
4. **Render settings phases**: OutputSettings resize / crop and a render queue held in the
   document (Playa Output Module parity); Viewer / Output nodes; external catalog of settings
   subgraphs (export / import with explicit membership and UUID remap); quality adaptation
   within explicit bounds (a profile now fixes samples and resolution scale); WorldDirect
   (shared world geometry, transforms, HDR light, then frame / input latency numbers).
5. **Rough glass energy** (render-rs standard-surface-bsdf): no multiple-scattering compensation
   for dielectric transmission (furnace 0.81-0.93 inside at roughness 0.5, not re-verified). Plan:
   furnace grid baseline -> references (Turquin 2019, Cycles multiscatter GGX glass, OpenPBR) ->
   R/T compensation table, Rust+WGSL parity -> furnace tests (>= 0.99 rough, 1 smooth) -> render.
6. **Hybrid DE surface** (BUG1 leftovers): discontinuities at fold / escape transitions and the
   6-probe normal's p99 11 deg on converged hits; isolated cost of the 6-probe normal. Over-relaxed
   sphere tracing only with a safety check (Hybrid DEs are not Lipschitz).
7. **Render / Encode colour output**: one option set across EXR (ACEScg / ACES2065-1 / linear 709 /
   2020 + display-referred), PNG (SDR / HDR10 / HLG), MP4 (SDR [+ HDR10]). Plan first.
8. **In-process HDR video** via ffmpeg-rs (requirements: ffmpeg-rs/BUG3.md); then delete the ffmpeg
   stopgap (`PngVideo`, `encode_video`).
9. **GPU test isolation**: Vulkan Video export tests fail with "Posix(38)" beside the render_service /
   colour GPU tests and pass alone; `vulkan_video()` serializes only within export.rs. Find the
   conflicting GPU use. Related flake: `cuda_export_coordinator_samples_animation_and_writes_each_frame_once`
   ("autonomous export timed out").
10. Discarded sends outside io_service (io_service now logs through `deliver`): `app.rs`
    `let _ = self.io.send(Command::Delete(..))` loses the error when the IO queue is full (a real
    loss: report it); `denoise.rs` result and six `export.rs` writer events (`Written`,
    `Finished`, `Cancelled`, `Failed`) - classify each as "receiver gone after cancel" (expected)
    or a lost result, and handle it explicitly.
11. Six soft slider ranges were picked without measurement (aperture, focus, orbit speed, thin film,
    sun angle, hit epsilon) - operator to check.
12. **Convergence**, each step measured with `--world-bench` + `tools/convergence.py`: firefly clamp
    of indirect contributions by AP1 luminance (off by default, labelled biased, report the energy
    loss); Russian roulette from AP1 luminance with a minimum survival probability and start depth
    2-3 (gpu.rs still uses `max3(throughput)` from bounce 1); then profile secondary-ray marching
    before any path guiding / light cache.
13. Viewport status bar: adaptive sampling progress (active tiles / converged %).
14. ACEScg golden check: a fixed grey scene under a white sky before (b4fd7a5^) and after must match
    within noise; record the numbers in CHANGELOG.
15. **Native verification pass** (tests cover logic only): narrow widths and 100/150% scale, layer
    drag/drop and RMB menus, Material Library layout, cached HDR playback, keyed and unkeyed camera
    flight, File Open/Save during rendering, HDR environment comparison, UI latency under heavy
    render + OIDN. Screenshots to ~/.warpbro/diagnostics.
16. **Reusable workspace extraction**: Outliner and Timeline adapters, adaptive Gallery, Material
    Library browser + publish fractal-materials, shared hotkey dispatch / file-dialog history,
    Curve Editor on the shared keys. Specify the host contracts (stable ids, select vs assign
    intents, one transaction per gesture) first; pilot in Playa.
17. CUDA startup: a reusable application module cache and a viewport initialization overlay.
18. Dead per-object `render` fields on fractal nodes (only `/render/iterations` is per object) -
    drop when the object model is split (low).
19. ofx-rs fractal kernel copy still uses pcg4d white noise and no adaptive sampling: resync with the
    Owen-Sobol sampler and `active` / `moment` buffers, or record that ofx-rs keeps its own (low).
20. Kvazaar inter-prediction corruption (repro: `gpu_hevc_motion_fixture_encodes_every_source_frame`)
    is mitigated by I-frames only; file it in ffmpeg-rs (low).
21. Deferred (operator decision): HDR environment blending, area lights.

## Related work in other repos

- curves-rs: remove `curves::legacy` / `Track::from_legacy` once Playa no longer uses it
  (WarpBro does not).
- Playa: Output / Render Queue adaptation of the settings-node contract; Attribute Editor
  acceptance against WarpBro; `playa-io` still requests the removed `av-player/simd` feature.
- SquareBob: encoder / HDR / display parity through the shared egui-display export code; toolbar,
  camera slots and named presets acceptance; port of the settings nodes.
- Apps (Watermark, COLMAP, GitNexus, RV, EXV, OTIO): native UI acceptance is still open.

## Gotchas

- egui: nested `ui.input` inside `hotkeys::active` deadlocks; read `active` first. Hotkeys need
  exact modifier matching; egui tests send key releases (a second press is a repeat).
- `cargo oxide` keeps one backend cache per machine (`~/.cargo/cuda-oxide`): two concurrent builds
  against different cuda-oxide revisions overwrite each other's backend. Build them in sequence.
- `--arch` changes the backend flags, so every crate's fingerprint: tests and builds that should
  share artifacts need the same arch. `default-arch` in `.cargo/cuda-oxide.toml` gives them one;
  without it builds follow the backend default, which moved from sm_75 to sm_80 in the 2026-10
  upstream sync.
- The Windows LLVM installer has no `llc`; the clang+llvm release archive does. cuda-oxide also
  runs `opt` and `llvm-link` from llc's directory.
- Many sources here are CRLF on disk: scripted replacements must match `\r\n` (or use Edit).
- render-rs: fetch before committing (local main was once behind origin).
- Playa: `cargo test -p playa-app <filter>` (not `--bin playa`); the OpenFX host needs
  `--features ofx`; debug builds overflow the main stack at start - run release.

## Key files

- WarpBro: src/world.rs (graph document, commands, settings nodes), src/world_ui.rs (outliner,
  timeline, Attribute Editor), src/render_profiles*.rs (settings contracts and catalog UI),
  src/app.rs (viewport routing, Render / Encode), src/export.rs, src/render.rs, src/gpu.rs
  (kernels), src/render_service.rs, bootstrap.py (every build / check / CI entry), docs/.
- render-rs BSDF: crates/render-engine-pt/standard-surface-bsdf/src/{microfacet.rs,sample.rs,wgsl/microfacet.wgsl}.

## Findings worth keeping

### DE boundary and inlining (2026-10-09)

- `bulb_estimate` was `#[inline(never)]` so specialized and mixed-world routes contract its
  arithmetic identically; the call in the hot march cost ~27%. The fork used to drop
  `#[inline(never)]` (only `alwaysinline` reached LLVM), which is why a narrower boundary failed the
  oracle: opt inlined the small helper anyway. With `noinline` honoured, `inverse_affine` alone is
  enough. Oracle: `cuda_specialized_world_materials_preserve_radiance_and_affine_guides`.
- Measure kernels by A/B with alternating runs and medians (`--world-bench`), and compare embedded
  PTX bytes to prove a source change did not touch device code.

### Cubin (2026-10-04)

- `--materialize-cubin --arch sm_86` loaded in 82 ms, then the first constant-memory parameter
  upload failed with `DriverError(500, "named symbol not found")`: the embedded cubin had no
  parameter symbol. A volatile-read workaround triggered a very slow full NVVM compile. The route
  needs a proper constant-memory contract in cuda-oxide and a regression before adoption.

### BUG1 (2026-10-05; scene copper-turbine-kifs, frame 27)

- **Normals**: tiny gradients lost their direction (fixed 4f66e83: scale by the max component,
  retry smaller steps). The 4-probe tetrahedral normal was biased; now 6 central probes at eps/4.
  On converged hits (4096 steps, 19 311 stable neighbourhoods): old tetra median 0.45 deg / p90
  2.96 deg; 6-probe median 0.016 deg / p90 0.043 deg / p99 11.1 deg. Primary hits and their
  positions unchanged bit for bit. Regression: `cuda_curved_hybrid_normals_match_converged_field_gradient`.
- **Phantom hits**: at 256 steps 37% of primary hits were rays that ran out of steps (half were
  misses, half 0.30 units too deep, normals off by a median 48 deg). Fixed: `Outcome::Unresolved`,
  one policy per ray kind (camera: background; bounce: path ends; shadow: blocked; Direct: Keinert's
  half-pixel rule), unresolved marches counted in the status bar, presets at 4096 steps, step cap
  independent of the budget, glass interior probes a setting (256). Cost: primary rays 37.1 ms
  (256) vs 47.4 ms (8192); full 6-bounce 16 spp render 978 ms vs 958 ms. As Mandelbulber2 and
  Fragmentarium do (a ray out of steps is a miss). Regression:
  `cuda_march_out_of_steps_is_a_reported_miss_not_a_phantom_hit`.
- **Glass energy**: clear glass lost energy at grazing angles (exact Fresnel for the delta lobe but
  the MaterialX albedo approximation for its throughput; IOR 1 gave reflection weight 0). Fixed in
  render-rs 207473e (`exact_interface`: albedo = exact F for smooth / IOR-1 interfaces, Rust+WGSL).
  Scene furnace 1.00003.
- **Not the cause**: material (matte shows the same facets), denoiser (raw accumulation unchanged),
  iterations (orbits escape after 4-9 of 12). bailout and step settings change the surface itself.
- Sources: Hart, Sphere Tracing; Christensen (Fragmentarium) lighting and Mandelbulb DE;
  Mandelbulber2 shader_calculate_normals.cpp / compute_fractal.cpp / ray_recursion.cl; Keinert et
  al. 2014 Enhanced Sphere Tracing.
