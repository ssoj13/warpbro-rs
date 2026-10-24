# HANDOFF (2026-10-24)

Read first: `PLAN.md` (rules, open work), `plan1.md` (bug hunt: remaining items in "Audit of the
open checkboxes"), `plan2.md` (render pacing / display speed), `CHANGELOG.md` "Unreleased",
`AGENTS.md` (module map, build rules). Cross-repo: oiio-rs `plan-cuda-oxide.md`.

## State

- Correction to the previous branch note: work is on `fix/plan1`, based on HEAD `792e6b5`.
  Local `main` is still `4cfa5eb`, 22 commits behind `origin/main` at the branch check. The user
  authorized pushing to both `main` and `fix/plan1`; publishing these changes is pending.
- Viewport stalls fixed (52f6545): `Gpu::stamp` waits for the backend's work before recording a
  batch's time, so `Batch::Budget` sizes batches from the GPU's real cost. Colour changes reuse
  the cached OCIO transforms (`DisplayKey` keyed by `Ocio::serial`). Correction to the previous
  handoff: the checked installed `~/.local/bin/WarpBro.exe` had modification time
  `2026-10-24T17:54:51Z` and size 171,435,008 bytes, so the old October 16 date was wrong.
  Its source commit was not embedded or established; modification time and size do not prove
  which fixes it contained. The synchronous shader-compilation freeze was reproduced in a
  newly rebuilt `792e6b5`-derived build whose source included `52f6545`, independently of the
  installed binary's provenance.
- The stop-gap costs GPU idle time between steps; plan2.md removes it (two batches in flight,
  GPU timers, no synchronous per-step readbacks, display on the GPU).
- In progress elsewhere: oiio-rs `plan-cuda-oxide.md` (approved 2026-10-24): CUDA for
  vfx-compute and vfx-ocio on cuda-oxide, the cudarc/NVRTC engine removed. WarpBro then runs
  tonemap + OCIO on its CUDA film without leaving the GPU (plan2 stage 3). Stages 0, 0b and the
  first slice of stage 1 are done: see oiio-rs `HANDOFF.md` (2026-10-24 section).
- d9980e1: Cargo.lock on cuda-rust-windows fbb7df9 (committed CUDA driver bindings: no CUDA
  toolkit needed to build; `cuda_toolkit_dir()` removed; `links = "cuda_bindings"` forbids two
  fork revisions in one graph). Only `cuda-bindings` / `cuda-core` were checked (`--locked`); the
  full app was not rebuilt. Details: `nv/cuda-oxide-windows/HANDOFF.md`.
- New local control: the separate `app-control` crate supplies the loopback HTTP transport and
  bounded command bus; `src/app_control.rs` dispatches on the UI thread through WorldEditor and
  camera navigation. See [the REST reference](docs/reference/rest-api.md), including orbit, pan,
  zoom, fly, camera patches, queried attribute edits, screenshots, and queued-result semantics.
- WGPU trace/pick preflight now polls deduplicated compile tickets serviced by a separate
  `warpbro-shader-compiler` thread. Render targets remain owned by one RenderWorker. The UI
  reports `Preparing shader` with elapsed time/entry and labels extended-family Auto scenes WGPU.
- One `ThumbnailCache` / `DiskCache` implementation serves gallery, bookmarks and material
  previews under `<profile>/cache/thumbnails`, with a 256 MiB budget, validated binary entries,
  atomic publication, content-addressed evaluated inputs, resident environment fingerprints,
  OCIO processor identities and source fingerprints. Terminal thumbnail completion survives
  event-queue backpressure. This does not add disk-backed timeline preview caching.
- Measured baseline (before accepting the final new build): `world_fast_f12` compiled in
  25.984 s and `world_fast_f13` in 27.792 s. The Bristorbrot request waited behind another
  thumbnail pipeline. At 932×683 its warm 32-SPP render was about 8 s; the 24.093 s whole-load
  measurement included about 16 s of waiting. OIDN was 195 ms. These are separate costs, not a
  single viewport-render timing or a frame-rate guarantee.
- Pipeline-cache investigation: the old 61,024,499-byte seed carried a 616.x driver validation
  header while the current driver was 621.01. The accepted cache was only 100 bytes; a tiny
  compile produced 7,211 bytes in memory while the disk file remained unchanged. Upstream
  gpu-info guard fix `f845ac889738e159191eb9f71cfabc8831ce134f` is committed/pushed, and only
  that owned Git revision was manually advanced in Cargo.lock. With the new build, pipeline
  persistence across restart is verified: `world_fast_f25` prepared in 0.050 s and
  `world_full_f26` in 0.055 s, against the latter's 65.499 s cold first-session compile.
  This verifies reuse of those pipelines; it is not a general render-speed measurement.
- Final build with the new dependency exited 0. The WarpBro checkpoint advanced from 346 to
  347 passing tests, with 7 ignored and 47 filtered; all 3 REST-crate tests passed. The app
  completed all 38 gallery thumbnails and 69 preset swatches with 0 failures; thumbnail-cache
  counters were 6 hits, 111 misses and 111 writes. After restart, 110 thumbnail cache hits,
  2 misses and 2 writes were recorded, with all 38 gallery cards and 69 swatches ready and no failures.
  A final stable Bristorbrot run (revision 0, default camera, 932×683) reached 32 spp in 8.662 s;
  its reused pipeline prepared in 0.0615 s.
- The upstream cache regression failed under the old guard (exit 101) and passed with the fix;
  upstream library tests had 62 passed / 18 ignored, and all three pipeline-cache tests
  (including two GPU tests) passed. Upstream Clippy exited 101 on the preexisting
  `src/pool.rs:221` `unnecessary_sort_by` finding; that unrelated code was not changed.
- The 57 s Bristorbrot smoke run and the later controlled log that switched to Torus Knot were
  affected by manual interaction and cannot establish pure timing. Keep the earlier stable
  baseline above separate. The user's report that rendering felt substantially livelier is
  subjective feedback, not a measured speedup.

## Next

1. Operator explicitly requested publishing this checkpoint, then replacing the separate
   REST command queue with Playa's EventBus and routing the entire UI through that same bus.
   The current app-control queue is a checkpoint, not the accepted final control architecture.
2. oiio-rs plan-cuda-oxide.md: rest of stage 1, then stages 2-4 (see oiio-rs HANDOFF.md).
   Then plan2 stage 3 here.
3. plan2.md stage 0 (measurements; GPU runs approved 2026-10-24) -> stage 1 pacing.
4. plan1.md leftovers: the `.ok()` sweep part 2 (`nodes()` -> `Result`); operator decisions
   Q2 (`conn`), Q4 (`path_convergence.rs`), 7.3 (GPU->CPU->GPU display, now plan2 stage 4).
5. Pending GPU checks of plan1 7.5: the WGPU world test compile time (was 4946 s before 7.5),
   `parity_tests::cuda_wgpu_primary_guides_agree_per_pixel`, an A/B `--world-bench --backend
   wgpu`. Then delete `target/release` (kept for these).

## Traps

- Build / test only through `python C:/projects/projects.rust.cg/cglibs/.buildlock/buildlock.py
  <cmd>` (machine-wide lock, never nested), and only after the edits are finished.
- Tests need `--release` (debug device codegen of fractal-kernels fails):
  `buildlock.py cargo oxide test -- --release --locked -- <filters> --skip cuda_ --skip gpu_`.
  Type-check: `FRAC_ALLOW_PLAIN_CARGO=1 buildlock.py cargo clippy --all-targets --locked`.
- Device tests start with `cuda_` / `gpu_`; no other test name may contain them
  (`test_gpu::device_prefixes_only_start_test_names`).
- CUDA / WGPU runs: only with the operator's go, on a quiet machine.
- Own git deps: edit the lock's rev by hand, never `cargo update -p` (it downgrades windows-sys).
- `BUG1.md` is an untracked operator file in the current workspace. Preserve it and unrelated backlog.
- A git commit from a pwsh stdin script stops after `git add` (CRLF warnings on stderr):
  commit with `git commit -F <file>` as its own command.
