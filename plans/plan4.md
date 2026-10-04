# Plan 4: Scene files, animated presets and orbit animation

Current status (2026-10-03): [workspace guide](../docs/workspace.md), [startup investigation](../docs/cuda-startup.md), and [changelog](../CHANGELOG.md) describe the current host/toolkit integration. The dated checkpoints below remain historical evidence; unchecked native checks are not implied complete.

Date: 2026-10-03.
Status: File menus, five animated presets and orbit evaluation implemented; final system suite passed 144 tests. Final tuned build, two preset tests, 15 key shots and all 1,250 GPU frames passed; five low-resolution 32-frame MP4 comparisons passed; manual native File/UI checks remain pending.

## Goals and ownership

Add File → Open, Save and Save As for the persistent World document. Provide five visually distinct animated presets, each with 250 frames (0–249 inclusive) at 24 FPS. Restore camera orbit angular speed as an animatable property. Preserve shared selection, transactional undo, stable camera references and asynchronous rendering.

Root owns `src/presets.rs`, World factory/evaluation and orbit controls in `src/world.rs`/`src/world_ui.rs`. The UI implementation owns File menus and asynchronous IO in `src/app.rs`, `src/dock.rs` and `src/io_service.rs`, plus idle-cache work. Renderer/service cache validation belongs to Plan 2; export/ffmpeg changes belong to Plan 3. Concurrent work must preserve these boundaries and existing changes.

## Document and IO contract

- [ ] Open a Playa World document through the existing migration/validation path; preserve UUIDs, node order, hierarchy, metadata, material references, keys and the active camera.
- [ ] Track the current file path; Save uses it, while Save As chooses a new destination. A document without a path uses Save As.
- [ ] Capture a coherent document snapshot for background writing; filesystem work does not block the UI or render worker.
- [ ] Apply a completed Open through one explicit document replacement path; invalidate stale render generations and cached editor projections.
- [ ] Keep the existing document on cancellation or failed reads; report IO/validation errors visibly.
- [ ] Write atomically and preserve an existing file on failure. Complete the save status only after publication succeeds.
- [ ] Save persistent nodes/animation rather than evaluated Scene objects or device buffers.

## Presets and deterministic evaluation

- [x] Build five authored World documents with deliberate composition, lighting and materials; retain the existing gallery/bookmark workflow.
- [x] Set an inclusive export range 0–249 at rational 24/1 FPS. Use deterministic timeline evaluation at frame time `frame / 24`, independent of playback wall-clock rate.
- [ ] Animate camera and/or object transforms, plus a couple of meaningful fractal-parameter transitions per preset. Select parameters and amplitudes that preserve a readable subject.
- [x] Store animation in the World/Playa attributes so Timeline, preview and export evaluate the same document.
- [x] Restore orbit speed (degrees/second) and phase as World host attributes, with frame-domain integration supporting animated speed and connected numeric drivers. Independent seeking and FPS changes do not accumulate UI deltas.
- [ ] Keep manual camera editing and orbit animation coherent; preserve the stable active-camera UUID and existing camera migration.

## Meaningful verification

- [ ] Semantic save/open/save roundtrip fixtures retain node UUIDs, saved order, hierarchy, metadata, materials, spans/visibility/solo/lock, keys and active camera. Compare document meaning rather than JSON formatting.
- [ ] Exercise legacy migration, malformed/missing files, cancelled operations and atomic-write failures; verify the old document/file survives failures.
- [ ] Evaluate each preset at frames 0, 124 and 249, including out-of-order/repeated evaluation; camera, transforms and parameter values remain deterministic.
- [x] Render all 250 frames for each of five presets (1,250 total); rational video timing regression passed.
- [ ] Test orbit speed interpolation/time evaluation, stable camera identity and save/open preservation of its animation.
- [x] Render and inspect all 15 first/middle/last GPU fixtures; subjects remain contained at frames 0/124/249.
- [ ] Inspect native preset interaction and playback; key-shot inspection does not establish every transition visually.
- [x] Reuse initial 32-frame raw RGB sequences for all five MP4 source/decode comparisons in Plan 3; full-length/high-resolution encoding remains pending.
- [x] Run 144 host CPU/CUDA/OIDN tests; after the final distance constants, two targeted preset tests and release build passed.
- [ ] Inspect File menu operation during rendering and record manual native interaction results.

## Implementation and validation checkpoint

File Open/Save/Save As and five 250-frame presets are implemented. Orbit speed defaults to neutral zero for old documents; the evaluator integrates in the frame domain. Manual viewport yaw uses a delta that excludes orbit, avoiding double application. Tests cover neutral defaults, migration, static negative and animated speed, independent seeking and manual-camera undo.

The [integration suite](../target/verification/integration-tests.out) passed 143 tests, including the normally ignored GPU tests, in 23.02 s with success marker 0. A subsequent `decode_scene` fix restores nondefault timeline settings and adds a regression; unused `ensure_environment` was removed. An initial release build passed in approximately 200 s; its obsolete `validate_world` warning was subsequently addressed with test-only gating. The [final system suite](../target/verification/final-system-tests.out) passed 144 tests in 55.50 s without compiler warnings, `CDX_FINAL_SYSTEM_TEST_EXIT=0`, including the latest Chrome tuning, camera/file timeline fix and production MP4 rational-timing regression. After the final Chrome/Apollonian distance constants, two targeted preset tests passed in 16.28 s and release build passed in 46.14 s; all tuned validation markers are 0 in [the checkpoint log](../target/verification/tuned-final-presets.out). The 144-test suite preceded those last constants. Native File interactions remain pending.

The implemented CLI `--animated-fixtures DIR [W H SPP]` writes first/middle/last PNGs and `scene.frac.json`; `--all-frames` renders all 250 frames with deterministic seed 0 and a reused target. GPU fixture rendering produced 15 first/middle/last images at 480×270 and 32 SPP. Inspection found cropping and small subject framing; the presets were tuned. The final 15 shots were rerendered at 480×270/32 SPP and the regenerated [contact sheet](../target/verification/animated-presets/contact-sheet.png) was inspected: all five subjects are contained at 0/124/249. Chrome's silver appearance is intentionally low contrast. All 1,250 frames rendered successfully at 160×90/4 SPP. The initial 32-frame raw sequences from all five fractals passed MP4 source/decode comparison, including frame count, zero-origin timing and improved RGB PSNR with QP 18/medium (see Plan 3). These low-resolution results do not validate full-length or 4K encoding. Previous UI snapshots do not establish universal responsiveness or final 150% layout validation.
