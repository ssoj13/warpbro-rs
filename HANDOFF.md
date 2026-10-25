# HANDOFF (2026-10-25)

## EventBus migration completed and verified

The migration is complete. Source review, formatting, strict Clippy, final release tests/build
and real API/UI acceptance passed. Git history and remote branch refs provide the commit identity
and publication status.

- Final machine-locked pipeline exited 0: WarpBro CPU suite **365 passed, 0 failed, 7 ignored,
  47 filtered** in 45.76 s; app-control **4 passed**; release build passed in 57.39 s.
  The CPU suite excluded device tests. Separate authorized CUDA Play and Scrub regressions
  passed in 2.90 s and 2.30 s. Receipt:
  `C:/Users/joss1/.filesystem-mcp-rs/tmp/run_command_1792913341583_083b9c3c-d882-4027-a23f-5d4e8fc64011_stdout.log`.
- Strict `FRAC_ALLOW_PLAIN_CARGO=1 cargo clippy --workspace --all-targets --locked -- -D warnings`
  passed in 23.49 s; the full-entry formatting check passed.
- Real API/UI acceptance passed; receipts and before/after PNGs are under
  `C:/Users/joss1/.warpbro/diagnostics/eventbus-20261025/` (`result.json`). It checked the actual
  Playa EventBus and enumerated 39 advertised commands; all five navigation actions plus Undo had camera
  acknowledgements matching queries; picking stopped playback; Save/New/Open restored document
  identity; invalid zoom returned HTTP 400; and colour slot 3 selected the exact 1000-nit HDR view.
  The app exited cleanly through `app.exit` with code 0 (PID 49440).
- The accepted Bristorbrot frame at 932×683 reached 32 spp in 8.385 s (OIDN 242.5 ms).
  Orbit changed generation 19 to 21; the new frame reached 32 spp (OIDN 205.4 ms), and its PNG
  differed. Both PNGs were visually inspected. Cached WGPU `world_fast_f9` prepared in
  0.0475857 s. These are workload-specific receipts, not a general frame-rate guarantee.
- Installed `~/.local/bin/WarpBro.exe` matches the built binary: 162,332,672 bytes, SHA-256
  `9a331e1594e604f92c400cb69089771241174b624f8cc1b74b9928134394439e`.
  The installed binary's real main profile was set through REST on port 19878 to
  **ACES 2.0 - HDR 1000 nits (P3 D65)** on **Display P3 HDR - Display**. The installed-profile
  receipt (`installed-hdr-result.json`, PID 18452) reports success and exit code 0; controls,
  fonts, presets, recording, camera slots and GUI FPS preferences were preserved. Previous
  executable/settings backups remain in the diagnostics directory. All acceptance app processes
  and HTTP listeners are closed.
- Earlier failures are resolved by this CPU run: the first compile exited 101 with four
  integration errors, intermediate runs were 357 passed / 4 failed and 360 passed / 2 failed,
  and app-control previously had one port-reuse failure.
- Upstream video-io `90477c3689ab97a9f1b419c8ae9b5b7f79ab75b5` is pushed and pinned;
  its 26 release tests passed; the separate WarpBro acceptance receipts are above.
- **P1 dock ownership fixed in source:** `dock_ui` renders an immutable layout clone while App
  retains the authoritative layout, with event drains around rendering. Save no longer observes
  a temporary empty dock, and returning a borrowed layout cannot overwrite Restore/Reset/actions.
- **P2 file completion/admission fixed in source:** Scene and Bookmark completion DTOs enter
  the central EventBus dispatcher. Scene save stages identity/document changes and applies them
  only after I/O admission; `submit_scene_action` returns a `Result`, propagating queue failures
  instead of reporting HTTP success.
- **P2 picking fixed in source:** picks cancel playback/preview and evaluate the accepted frame
  with its viewport profile, rather than copying stale `self.request` data. Pending results carry
  document/camera/frame/revision guards.
- Verified source regressions: raw-input preview cancellation emits `CancelPreview`; material
  Duplicate captures `world.targets(Some(id))`; preferences and scene load immediately synchronize
  `world.new_key`. A captured no-op ColourSelection no longer assigns a whole stale-frame Scene.
  `idle_event_dispatch_keeps_the_evaluated_scene_cached` and the UI no-op cache regression pass;
  FinishEdit/UI no-ops preserve evaluated-scene caching while camera authoring/error recovery
  still canonicalize.
- Orbit-pole correction retains the rig quaternion while its canonical orbit marker matches,
  including at rest. External camera edits reseed the controller; a new document clears it.
  Existing coast/rest assertions pass with the corrected post-dispatch navigation marking.
- **Terminal backpressure fixed in source:** generic `emit_retained` keeps a rejected typed
  event, drains the actual shared EventBus and retries. Scene/Bookmark completions and Take
  imports survive a full bus; Take imports retain their original request document. No separate
  application-command queue is added. Review confirmed the seven prior findings and this
  backpressure issue resolved in source; that review is not a passing test receipt.
- **Windows HTTP dependency fix published:** tiny_http 0.12 detached its accept thread, so the
  port could remain bound after server Drop. The fork
  [ssoj13/tiny-http](https://github.com/ssoj13/tiny-http) is pushed on `main` at
  `69b64907d74968ad520124306a856432be7e6f47`, based on official 0.12 tag commit
  `212b1c45852fef2093dc1374875a9393c55eb4b9`; Cargo.toml and Cargo.lock are manually pinned.
  The fix joins the accept thread, wakes wildcard listeners through loopback, and moves TLS
  handshakes to the existing connection pool. Public API, dependencies/version and socket
  reuse flags are unchanged. Default release checks had 50 passes including 9 doc tests;
  Rustls checks had 51 passes including 9 doc tests. These are upstream receipts, not final
  WarpBro integration results. The shared cglibs catalog now lists the fork.
- Acceptance history: the first harness attempt used an incorrect asynchronous generation wait,
  which was corrected. The user confirmed that the intermediate diagnostic window was closed
  manually. The instrumented third run passed.
- The broader GPU/parity checks in the backlog are separate from the completed migration's two
  CUDA regressions and real WGPU acceptance above. Preserve the operator's untracked `BUG1.md`
  and unrelated backlog.

Read first: `PLAN.md` (rules, open work), `plan1.md` (bug hunt: remaining items in "Audit of the
open checkboxes"), `plan2.md` (render pacing / display speed), `CHANGELOG.md` "Unreleased",
`AGENTS.md` (module map, build rules). Cross-repo: oiio-rs `plan-cuda-oxide.md`.

## State

- Checkpoint `47509bb` was pushed atomically to `origin/main` and `origin/fix/plan1` as requested.
  The completed Playa EventBus migration follows that checkpoint; its final checks, release build
  and real API/UI acceptance pass. Use Git history and remote branch refs for publication status.
  The previous assertion that all work was already on local `main` was wrong.
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
  overflow refuses the new event without evicting an accepted event. Migration verification and
  real API/UI acceptance pass as recorded above; older checkpoint receipts remain historical.
  Video-io deferred preset-request support is pushed/pinned at `90477c3` with 26 upstream release
  tests passed; WarpBro integration is covered by the separate acceptance receipts above.
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

1. For migration provenance, inspect `git log` and
   `git ls-remote origin refs/heads/main refs/heads/fix/plan1`. Checkpoint `47509bb` is historical;
   the actual migration commit and publication status come from those records.
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
