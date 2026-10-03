# Plan 5: Materials, Timeline controls and cached playback

Date: 2026-10-03.
Status: WarpBro rename and current Materials/Timeline/HDR/shortcut/cache/Camera changes implemented; all 181 Final5 tests passed, including GPU tests. Release build and revised 1,250-frame/15-still fixtures passed. Final production build and Chrome start-frame inspection passed; final 15-still rerun passed; native/DPI/latency verification pending.

## Shared interface and document actions

- [x] Material Gallery shows actual World material nodes; clicking selects the UUID and opens the existing Attribute Editor. No separate material editor and no implicit assign/create.
- [x] Assign through the object's Material attribute (`WorldCommand::AssignMaterial`) or the card context menu's Assign to selected objects; shared material UUID edits update all consumers and asynchronous gallery thumbnails.
- [x] Use separate sibling toolbar buttons: + New material creates a default node; Create from preset opens its preset submenu and creates a preset-based node. Neither assigns it.
- [x] Add card RMB Apply preset to this material, updating the existing UUID/all consumers; ordinary click remains selection only.
- [x] Add Refresh preview retry for terminal colour/render failures and reject stale results.
- [x] Complete the preview-error API regression in Final5: 181 passed, zero failed/ignored.
- [x] Implement adaptive material columns and `auto_shrink(false)` scrolling; automated layout checks passed. Native narrow-width scrollbar inspection remains pending.
- [x] Timeline names use fixed painting/alignment and layer drag/drop explicitly responds to the primary button.
- [x] Context menus provide Duplicate and Delete Selected through transactional undo; retain UUID selection and coherent layer order.
- [x] Move Quick PNG from the menu into the Render / Encode panel; preserve the existing asynchronous render job.
- [ ] Verify actual pointer drag/drop, right-click menus, duplicate/delete/undo and narrow Materials layout in the native app.

## WarpBro branding

- [x] Rename and verify the GitHub repository [WarpBro](https://github.com/ssoj13/WarpBro).
- [x] Align Cargo/bootstrap/window branding and Windows binary `WarpBro.exe`; preserve legacy local folder/profile/data paths for compatibility.
- [x] Verify the updated GitHub SSH remote `ssh://git@github.com/ssoj13/WarpBro.git`.
- [x] Complete the production release build at the 180-test checkpoint.
- [x] Repeat all tests after the preview-error API and Chrome tuning: Final5 passed.
- [x] Complete the final production build: 1m43s, no Rust warnings, `CDX_WARPBRO5_STEP_2_EXIT=0`.
- [x] Inspect/accept the bounded Chrome start-frame rerender at 640×360/32 SPP; a coherent visible cube replaces the sparse flecks.
- [x] Complete the final 15-still rerun: `CDX_WARPBRO5_STILLS_EXIT=0`; inspect all five presets in the final contact sheet.

## HDR environment filtering

- [x] Replace nearest environment lookup with bilinear scene-linear float sampling, longitude wrap and pole clamp.
- [x] Preserve HDR range and keep the importance-sampling CDF independent of the filtered lookup.
- [x] Pass automated HDR lookup/filtering and stale-preview regressions in the full suite.
- [ ] Compare representative HDR environment images manually.

## AE shortcuts and playback

The shortcut implementation uses shared Attribute Editor/selection state: P/T selects Translate, alongside R/S/U/I/O and existing Space behaviour. Automated key/focus regressions passed; native shortcut interaction under rendering load remains pending.

- [x] Implement playback with PlayaClock and HDRGlobalFrameCache using native HDR frames and a worker-owned three-slot pool. The checked CPU constructor and rustdoc were published at Playa SSH revision `00d90428d6c931b69b7117fd2a4cdf143d075413`; the host pin is updated.
- [x] Validate cache integration, memory budget and invalidation through the 180-test checkpoint.
- [ ] Verify cached HDR frame presentation in the native app.
- [x] Define shared events for shortcut intent, selected playback range, cache progress, completion/cancellation and playback state.
- [x] Implement Ins selection-range playback and Shift+Ins cache-then-play intent; automated suite passed, native interaction pending.
- [x] Resolve selection range, cache identity/invalidation, sample target, seek/end behaviour and memory limits from the actual reused contract.
- [x] Keep frame generation, caching and GPU work asynchronous; UI handlers submit intent and consume ready progress/frames.
- [x] Pass automated key/focus checks; preserve text/numeric-editor focus, undo, selection and render generations when invoking shortcuts or cancelling cache preparation.
- [x] Test shortcut dispatch, selection-range endpoints, cache-before-play ordering, cancellation and invalidation in the 180-test checkpoint.
- [ ] Inspect native playback and interaction under rendering load.

## Monotonic preset development

Root owns the new preset revision. This requirement supersedes the earlier oscillating motion choices; the previous 1,250-frame renders validate the previous preset version only.

- [x] Unfold each preset monotonically from low to higher spatial frequencies across frames 0–249.
- [x] Give Mandelbulb `angle_scale` a 0.5→1.5 transition; power and phase use their own deliberate ranges.
- [x] Remove oscillating TRS and camera motion; inspect the progression rather than relying only on endpoint framing.
- [x] Pass the 180-test checkpoint and render all revised 1,250 frames at 160×90/4 SPP, plus 15 stills at 640×360/32 SPP.
- [x] Rerender/review the latest Chrome bounded-range start frame; accepted at 640×360/32 SPP.
- [x] Complete and visually inspect the final key-shot rerun. The previous 1,250-frame GPU run predates only this bounded Chrome tune; Final5 CPU tests cover all 250 final curves.

## Camera flight audit and accepted policy

The earlier audit found flight updating target/yaw/pitch while skipping visible Camera TRS, and animated-property writes inserting keys even when `key` was false. The new source edits visible Camera TRS. With Auto Key off, serialized hidden static offsets retain the existing animation curves byte-for-byte; with Auto Key on, only changed components receive keys. Automated Camera-node/curve/Auto Key regressions passed in the 180-test suite; native flight interaction remains pending.

- [x] Confirm the user-selected policy: edit the Camera node; create/update keys only when Auto Key is enabled.
- [x] Implement Camera-node edits, serialized static offsets and Auto Key-only changed-component keys through World commands/shared evaluation.
- [x] Verify Camera TRS, curve preservation/changed-component keys, evaluation and undo through automated regressions.
- [ ] Verify keyed/unkeyed camera flight manually in the native viewport.

## Verification checkpoint

[The verified combined run](../target/verification/warpbro-final3.out) passed 180 tests, zero failed/ignored, with normally ignored tests included, in 51.79 s (`CDX_WARPBRO3_TEST_EXIT=0`). It covers material UUID selection/open-existing-AE, assignment field/context actions, preview staleness, HDR, keys/focus and Camera TRS. The production release build passed; all 1,250 revised GPU frames at 160×90/4 SPP and 15 key shots at 640×360/32 SPP passed. The [contact sheet](../target/verification/unfolding/contact.png) was visually inspected. Chrome's sparse first frame prompted a final bounded-range tune. [Final5](../target/verification/warpbro-final5.out) subsequently passed 181 tests, zero failed/ignored, in 58.81 s (`CDX_WARPBRO5_STEP_1_EXIT=0`), completing the preview-error API regression after the explicit-click fix. The final production build passed in 1m43s without Rust warnings (`CDX_WARPBRO5_STEP_2_EXIT=0`). Chrome's new first frame at 640×360/32 SPP was inspected and accepted as a coherent visible cube; the final 15-still rerun passed (`CDX_WARPBRO5_STILLS_EXIT=0`) and the [new contact sheet](../target/verification/unfolding-v2/contact.png) was inspected for all five presets. The earlier 1,250-frame GPU run predates only the last bounded Chrome tune; Final5 CPU tests cover all 250 final preset curves. Native drag/drop/cache/shortcut interactions remain pending.

Current source work replaces Materials Parameters with World-node gallery selection into the existing Attribute Editor and explicit assignment/creation, plus adaptive layout, Timeline name painting/primary-button drag/drop and undo menus, Quick PNG relocation, and bilinear HDR lookup. The source now reuses the researched Playa clock/cache and implements the accepted Camera-node/Auto Key policy. Revised presets follow monotonic unfolding with no oscillating TRS/camera motion. The 180-test suite, release build and revised preset fixture checkpoint passed. Final5 tests/build passed and the tuned Chrome first frame was accepted. Final 15-still rerun and visual inspection passed. Native interaction validation remains pending.

Record final test/build artifacts and observed native behaviour before marking checklist items complete. Manual UI latency, the final scaled-window check and full-length/high-resolution MP4 acceptance remain pending from earlier phases; these changes do not establish universal responsiveness or whole-GPU speedup.
