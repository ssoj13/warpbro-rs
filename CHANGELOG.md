# Changelog

## Unreleased — 2026-10-03

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

Disk preview caching, Curve Editor integration, true refractive glass, and CUDA compiled-module caching remain open.
