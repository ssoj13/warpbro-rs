# Changelog

## Unreleased — 2026-10-04

### Glass and absorption

- Add keyable Material-node transmission, transmission color, extra roughness, and world-space absorption depth.
- Correct glass preset translation, promote transmissive Fast materials to Full dispatch, and connect existing Standard Surface refraction with enter/exit tracking.
- Add bounded interior exit search for signed and exterior-only fractal fields; unresolved exits terminate rather than leaking sky.
- Apply Beer-Lambert absorption to traveled interior segments, including internal reflections, without double-tinting interfaces.
- Add GlassBottleGreen and GlassWaterGreen; the catalog now has 69 presets in 12 categories.
- Invalidate preparation, accumulation, and preview identities for every new material field.
- Add missing controls to older node schemas while preserving existing authored values and keys. Reapply old glass presets explicitly to adopt corrected transmission.
- Document single-medium and probe-resolution limits in [the glass guide](docs/glass.md).

### Render startup

- Warm the CUDA driver's JIT cache for the exact executable during bootstrap builds, before reporting success; add `--warmup-cuda` and a packaging opt-out.
- Validate the constant-memory parameter block before reporting the CUDA worker ready. Initialization failures now fail the default bootstrap build.
- Keep the tested PTX path: cubin materialization was rejected after a missing-parameter-symbol failure and an excessively slow compiler workaround.
- Return parameter-upload failures as render-target errors instead of panicking the shared worker.

The accepted production build and CUDA warmup passed. The following native launch loaded/validated kernels in 87.5814 ms; viewport, material preview, Gallery, and OIDN rendered. The baseline preset regression failed as expected. The release suite passed 211 tests with 8 ignored probes (62.07 s); the separate glass visual probe passed and its six images were inspected. The targeted CUDA glass regression covers signed/unsigned geometry, legacy/World dispatch, and green absorption.

## 2026-10-03 workspace checkpoint

### Changed

- Keep the Timeline ruler and outline header fixed during vertical scrolling; layer names, bars, and property lanes remain synchronized beneath them.
- Pin every egui-widgets-rs dependency to the same SSH revision, `afad4c31f5066f34eca69994248a33735874fb12`, which combines aligned Attribute Editor APIs and the pinned-ruler extension.
- Add separate CUDA-context and embedded-kernel startup timings. Successful render checks reproduced long module loading under concurrent system load; startup optimization remains pending.
- Add current workspace and startup guides and reconcile README references with the implementation.

### Existing workspace features documented in this update

- Unified final/draft cached preview commands: Ins, Shift+Ins, and Ctrl+Shift+Ins; completed native viewport frames populate the final RAM cache.
- Green final and blue draft cache coverage in Timeline, with bounded memory and distinct quality modes.
- Universal node Attribute Editor and explicit material assignment through object properties, card menus, or **Apply to object**.
- Adaptive workspace/preset galleries, shared compact layout metrics, scoped hotkeys, and one Undo transaction per numeric gesture.
- Shared renderer-independent material catalog with 67 presets across 12 categories, including the expanded metals.
- Graceful partial-movie finalization on cancellation and explicit GPU/software encoder selection, as recorded in the export follow-up.

### Validation

The production release build passed. The release test executable passed **199 tests, 0 failed, 7 ignored** on the final repeat (72.71 s). The first run had 198 passes and one 90-second export-cancellation timeout under load; its isolated repeat passed in 1.39 s. The seven ignored GPU tests were not rerun in this update. The rebuilt and retried 32×32 legacy headless checks completed successfully. Shared Timeline, attribute-grid, and layout/configuration tests passed after combining the toolkit branches. See [CUDA startup](docs/cuda-startup.md) for exact timings and load caveats. Earlier full-suite counts and native-check limits remain in the dated plans.

Disk preview caching, Curve Editor integration, and CUDA compiled-module caching remain open. The subsequent glass update is described above.
