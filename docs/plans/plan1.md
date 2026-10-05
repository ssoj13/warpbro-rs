# Plan 1: Consistent interface using Playa's original widgets

Current status (2026-10-03): [workspace guide](../docs/workspace.md), [startup investigation](../docs/cuda-startup.md), and [changelog](../CHANGELOG.md) describe the current host/toolkit integration. The dated checkpoints below remain historical evidence; unchecked native checks are not implied complete.

Date: 2026-10-03.
Status: toolkit revision e953b2c published; 40 toolkit tests and strict Clippy passed. Final e953 validation passed all 118 CPU/GPU tests and the release build; scaled native inspection and manual latency checks pending.

## Scope and reference

Preserve Playa's original Attribute Editor composition: egui-attr-grid inside egui-titlebar::CollapsingSection. Improve those same shared widgets through optional configuration rather than replacing the editor. Keep the World command/undo model and the existing asynchronous render pipeline.

Reference sources:

- playa/crates/playa-app/src/app/tabs.rs: render_layer_attributes
- playa/crates/playa-ui/src/widgets/ae/ae_ui.rs: render_with_mixed_natural
- playa/crates/playa-ui/src/widgets/timeline/timeline_ui.rs
- egui-widgets-rs/crates/egui-titlebar: CollapsingSection
- egui-widgets-rs/crates/egui-attr-grid: render_grid/render_grid_virtualized
- egui-widgets-rs/crates/egui-statusbar: StatusBar

## Phase 1: Shared widget contract

- [x] Identify the original Playa editor and section widgets.
- [x] Audit the affected shared widget symbols before editing. AttrRow.emit has CRITICAL impact (9 dependents); preserve its old defaults and test compatibility. The grid renderer has LOW impact.
- [>] Extend the existing grid with optional animation/key-control gutter, display-label and custom-editor hooks. Existing consumers get no new buttons by default.
- [x] Define shared row height, label/value origins, fixed numeric component geometry, vertical centring and compact square icon controls.
- [ ] Preserve grid filtering, mixed values, splitter state and existing property editors.
- [x] Test fixed label/splitter geometry in the toolkit; 40 tests passed, including actual narrow/wide rendering with minimum editor width and unchanged saved splitter width.
- [ ] Verify native narrow-panel clipping and all default consumer layouts.

## Phase 2: Apply the original composition in frac-rs

- [x] Restore section chrome through CollapsingSection, keeping stable IDs and open state; checked in the default native-window snapshot.
- [x] Use short display labels and full-path tooltips without changing stored property addresses; compact rows checked in the default snapshot.
- [x] Name transform controls Translate, Rotate and Scale, as in Maya/Houdini; Rotate maps to the actual `rotation_degrees` property. Preserve XYZ identity and units.
- [x] Share geometry between Attribute Editor and Timeline through `AttributeMetrics::grid_config`; indentation affects names, not value columns. Controls preferences own the shared metrics.
- [ ] Rename the visible panel Attribute Editor in tabs, menus and layouts.
- [ ] Audit remaining Outliner/settings/toolbars/dialogs for alignment and spacing.

## Timeline interaction

- [ ] Replace manual Move up/down items with layer drag/drop and clear insertion feedback.
- [ ] Keep UUID selection, expansion, attribute/component addresses and keys stable during reorder.
- [x] Implement and unit-test atomic layer drag/drop and deferred numeric undo gestures; manual native interaction remains pending.
- [ ] Verify completed time gestures manually through the undo path.
- [ ] Reuse egui-track-timeline canvas actions. Playa's removed outline DnD is not a working reusable implementation.
- [ ] Keep vector parents compact and expand component lanes deliberately.

## Material library and assignment

- [ ] Keep the existing Materials library visible and explain its current assignment target.
- [x] Implement material assignment through validated UUID references and WorldCommand::AssignMaterial; assignment unit tests passed.
- [ ] Preserve selected object while browsing, and keep shared-material edits distinct from assignment.
- [ ] Test missing/deleted references, locked targets and undo.
- [ ] Do not claim a material-library workflow exists in Playa; source research found none.

## Status bar

- [x] Provide one resizable-section status bar through egui-statusbar's borrowed-section API, alongside the toolkit's fixed variant.
- [ ] Persist widths; retain clipped contents and double-click reset.
- [ ] Avoid per-frame boxed section callbacks/default-width allocations in the new rendering path.
- [ ] Keep fixed height at narrow widths and with long error/status text.
- [ ] If section ordering is added, use stable section IDs so widths follow their own indicator.

## UI performance and responsiveness invariants

- [x] Implement and unit-test descriptor/projection cache identity using document identity, editor revision and relevant evaluation state.
- [ ] Reuse buffers; use iterator/fixed-array numeric geometry instead of per-row Vec allocation.
- [ ] Refresh only dirty schema/order work and evaluation-dependent values.
- [ ] Do not serialize/clone the full document or create undo snapshots on idle repaint.
- [ ] Preserve existing asynchronous render workers, nonblocking command submission, backpressure, coalescing and latest-frame delivery.
- [ ] Never wait for GPU completion, OIDN or renderer work from the UI thread.
- [ ] Audit renderer locks and upload work; keep existing responsiveness measures intact.
- [ ] Verify input, resize, scrubbing and parameter edits during active heavy rendering and denoising. Record UI latency/frame-time variation, not only average FPS.

## Verification and publication

- [x] Run toolkit compatibility and meaningful geometry/interaction tests: 40 tests passed; strict Clippy passed for five crates.
- [ ] Check property editing, animation, drag/drop, undo and material assignment.
- [x] Publish toolkit improvements on GitHub SSH at `e953b2cc6836db2bb47aa44bb7995d9696636034`.
- [x] Confirm the preceding 8aad Cargo.lock used one SSH toolkit revision for all 21 widget sources.
- [x] Confirm final Cargo.lock uses one e953 SSH revision for all 21 widget sources.
- [x] Build the earlier c96 revision through bootstrap: release application build passed.
- [x] Complete the 8aad release build: 47.54 s, `CDX_SSOT_BUILD_EXIT=0`.
- [x] Complete the final e953 combined tests with `--include-ignored`: 118 passed, zero failed/ignored in 36.99 s; `CDX_FINAL_TEST_EXIT=0`.
- [x] Complete the final e953 release build: 49.19 s; `CDX_FINAL_BUILD_EXIT=0`.
- [x] Run host CPU tests against c96: 115 passed, three GPU tests ignored; exit code 0.
- [x] Complete all host CPU tests against 8aad: 115 passed, three GPU tests ignored in 36.22 s; `CDX_SSOT_TEST_EXIT=0`.
- [x] Run the earlier c96 GPU/OIDN checks explicitly against the existing test executable: all three passed.
- [x] Complete the 8aad GPU/OIDN checks: all three passed in 2.46 s, `CDX_SSOT_GPU_EXIT=0`.
- [x] Inspect the actual default native window: compact aligned Attribute Editor and one status bar.
- [ ] Inspect narrow widths and multiple UI scales after final fixes.
- [ ] Compare screenshots against Playa sections and aligned controls; inspect Timeline drop feedback and status splitters.
- [ ] Update README and record exact revisions, commands, test results and artifacts.

## Implementation update

The host source now uses Playa-style Attribute Editor sections and one shared `AttributeMetrics` configuration in Controls preferences. It adds a single status bar with resizable sections, Materials-library opening/assignment, and deferred numeric gestures that share one undo transaction. These source changes still need host interaction and visual checks.

The E0502 UI-closure issue is fixed. Host CPU tests against toolkit revision `c96ec7d19ab3c768b3917d499f0c3d5eafc8a11b` completed with exit code 0: [115 passed, three GPU tests ignored](../target/verification/attribute-layout-tests.out) in 37.57 s. The [three GPU tests](../target/verification/attribute-layout-gpu.out) then passed explicitly through `bootstrap.run` against the existing test executable in 3.98 s. Both success markers are 0; 118 host/GPU tests passed in total. The toolkit has 39 passing tests and strict Clippy passed for five crates. Its glyph-paint overload avoids cloning icon style; `AttributeMetrics::grid_config` supplies one shared grid configuration.

The earlier [release build](../target/verification/attribute-layout-build.out) passed with `CDX_BUILD_EXIT=0` in 2m56s. The [default native-window snapshot](../target/verification/attribute-ui-default.png) was inspected: it shows compact aligned Attribute Editor rows and one status bar. That inspection identified Timeline name centring and Rotate mapping issues; both are fixed, with 19 UI tests passing afterward.

Toolkit revision `8aad144ca69c4f02868c7cb2080d11f4d8184360` makes `AttrGridConfig::label_cells`, `value_cells` and `row_rects` the common geometry for the actual grid and Timeline. `SplitterTable::CELL_PADDING` and `DEFAULT_SPLITTER_WIDTH` are exposed from the renderer's own values; 36 lines of mirrored host layout were removed. The toolkit's 39 tests and strict Clippy for five crates passed. Final [host CPU tests on this revision](../target/verification/attribute-ssot-tests.out) passed: 115 passed, three ignored in 36.22 s, `CDX_SSOT_TEST_EXIT=0`. The 19 UI tests cover deferred undo, atomic drag/drop, cache identity and material assignment. These unit checks establish the command/layout behaviour; they do not replace manual native interactions. The [8aad release build](../target/verification/attribute-ssot-build.out) passed in 47.54 s (`CDX_SSOT_BUILD_EXIT=0`); its [three GPU checks](../target/verification/attribute-ssot-gpu.out) passed in 2.46 s (`CDX_SSOT_GPU_EXIT=0`). The inspected [100% native snapshot](../target/verification/attribute-ui-ssot.png) confirms corrected Rotate mapping, left-aligned layer names, compact controls and one bottom status bar. Its displayed 59 FPS during rendering does not establish input latency. Narrow/multiple-scale visuals, interaction under heavy rendering and UI latency still need verification. Idle settings serialization remains a known limitation to audit; these tests do not prove an allocation-free or serialization-free idle repaint.

Native inspection at 150% scale then exposed clipped numerical values in a narrow Attribute Editor. Published toolkit revision `e953b2cc6836db2bb47aa44bb7995d9696636034` adds opt-in `AttrGridConfig::min_editor_width`: `fitted_label_width` prioritizes editor space while retaining the saved splitter width for narrow-to-wide resizing. Timeline `row_rects` uses the same fit. The host sets a minimum for three TRS numeric components plus two gaps. All 40 toolkit tests and strict Clippy for five crates passed; an actual-render regression checks 300/460-pixel widths, editor width ≥112 and unchanged saved label width 210. Final [host validation](../target/verification/attribute-final.out) passed all 118 tests with `--include-ignored` (115 CPU + three CUDA/OIDN), zero failed/ignored in 36.99 s, `CDX_FINAL_TEST_EXIT=0`; the release build passed in 49.19 s, `CDX_FINAL_BUILD_EXIT=0`. Cargo.lock uses one e953 SSH revision for all 21 widget sources. The replacement 150% snapshot remains pending. Public API changes are additive; the original toolkit/Playa directories remain untouched.

Renderer optimization and MP4 work remain queued; there is no measured speedup or quality claim for them. A read-only Nsight baseline of the old `13b1caa` kernel completed 45 passes: [`fast_bulb` at 320×180](../target/verification/render-baseline-13b1caa-ncu.ncu-repz) took 27.85 ms under instrumentation, with 162 registers/thread, theoretical/achieved occupancy 25%/16.33%, 7.8 average active warp threads, 1.5% DRAM throughput, no spills and 45.7% dependency stalls. These profiler timings are baseline diagnostics, not a benchmark or speedup.

Implementation is not complete until tests and visual/responsiveness checks pass. Previous World/OIDN results do not validate these UI changes.
