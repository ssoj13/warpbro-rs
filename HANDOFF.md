# HANDOFF (2026-10-24)

## WIP checkpoint for urgent push

The user requested an immediate push before the session limit. This migration is unfinished;
the architecture descriptions below describe source changes, not accepted runtime behavior.

- First migration release compile exited 101 with four integration errors. All four reported
  errors have source fixes, but compilation has not been rerun. Migration CPU tests and new
  real-API acceptance have not run. The available app binary is checkpoint `47509bb`, not
  a verified migration build.
- Upstream video-io `90477c3689ab97a9f1b419c8ae9b5b7f79ab75b5` is pushed and pinned;
  its 26 release tests passed. This does not establish WarpBro integration acceptance.
- **P1 review follow-up:** `dock_ui` uses `mem::take(&mut self.dock)` while viewport code may
  dispatch events. Save can capture an empty layout; Restore, Reset or DockAction can be
  overwritten when the borrowed layout returns. Prevent dispatch while that layout is borrowed,
  or correct ownership before applying those actions.
- **P2 review follow-up:** `scene_file_event` loads directly outside the shared bus.
  `submit_scene_action` swallows I/O queue failures, allowing HTTP success, and renews identity
  before sending. Route the completion through the bus; stage changes and return a `Result`
  so failed admission cannot acknowledge or apply the operation.
- **P2 review follow-up:** `queue_pick` can use stale `self.request` scene data and does not
  cancel playback. Pick the accepted current-frame scene/profile after cancelling preview.
- Source fixes already made, still unverified in the migration build: raw-input preview
  cancellation emits `CancelPreview`; material Duplicate captures `world.targets(Some(id))`;
  `apply_settings` synchronizes `world.new_key`.
- Resume with the review fixes, then the machine-locked release compile/tests and real API/UI
  checks. Preserve the operator's untracked `BUG1.md` and unrelated backlog.

Read first: `PLAN.md` (rules, open work), `plan1.md` (bug hunt: remaining items in "Audit of the
open checkboxes"), `plan2.md` (render pacing / display speed), `CHANGELOG.md` "Unreleased",
`AGENTS.md` (module map, build rules). Cross-repo: oiio-rs `plan-cuda-oxide.md`.

## State

- Checkpoint `47509bb` was pushed atomically to `origin/main` and `origin/fix/plan1` as requested.
  The current Playa EventBus migration follows that checkpoint; its final build, tests and push are
  not yet accepted. The previous assertion that all work was already on local `main` was wrong.
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
- Local control now shares App's actual Playa `EventBus` with widgets and hotkeys. Both publish
  `app_control::Event<UiCommand>` with `Action::{App, World, Take, Query}`; `app_events.rs` drains
  and centrally dispatches, `app_actions.rs` applies app actions, and `world_events.rs` applies
  World intents. UI document/frame/camera and concrete targets are captured before deferral;
  stale-context actions fail, and widget drafts remain transient. The `app-control` crate retains
  only HTTP transport/typed decoding and acknowledgement channels; its parser channel carries
  sockets, not application commands. See [the REST reference](docs/reference/rest-api.md).
- Playa's reliable `EventBus::try_emit` / emitter admission is published upstream as `72ed08b`:
  overflow refuses the new event without evicting an accepted event. Migration verification is
  pending; do not treat the pre-migration checkpoint's receipts below as tests of this architecture.
  Video-io deferred preset-request support is also under upstream testing.
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
- Checkpoint `47509bb` build with the new dependency exited 0. The WarpBro checkpoint advanced from 346 to
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

1. Finish and verify the UI/REST Playa EventBus migration, including deferred video-io preset
   requests, document/frame/camera/target capture, gesture Undo and reliable overflow handling.
   Run final locked checks and app acceptance, then record receipts and publish the migration;
   checkpoint `47509bb` is already pushed to both branches.
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
