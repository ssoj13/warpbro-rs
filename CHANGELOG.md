# Changelog

## Unreleased — 2026-10-04

### Snapshots, slot buttons, Render / Encode panel

- Viewport snapshot: a camera button on the toolbar (click: as the monitor shows it; right click and
  File: SDR PNG, HDR10 PQ PNG, display EXR), one `Frame::save` path. Every PNG writer (snapshot,
  export, CLI) names HDR PNGs `*.pq.png` / `*.hlg.png` (`PngEncoding::suffix`) and takes the peak from
  one constant; a snapshot records its HDR view's own peak.
- **Changed default:** CamClip and colour presets now restore on left click and store on right
  click; Settings → Controls → Swap copy/paste mouse buttons restores the old layout for both.
- Render / Encode: shared settings once above the format tabs; file names fall back to "untitled".

### Sphere tracing: out of steps is a miss (BUG1)

- A path-traced camera or bounce ray that runs out of march steps is a miss, as in Mandelbulber and
  Fragmentarium. It used to take its closest sample within a pixel (Keinert et al. 2014, Enhanced
  Sphere Tracing 3.2, a real-time technique): on BUG1 frame 27, 37% of the primary hits were such
  samples - 52% of them rays passing the surface, 48% lying 0.3 scene units before the real hit,
  normals ~48 degrees off. WarpBro's opt-in `ofx-direct` kernels keep Keinert's rule at his half
  pixel. The OFX plug-in builds from its own copy (`ofx-fractal/kernels/source`), not changed here.
- The longest step is a fixed share of the ray's interval (`STEP_CAP_STEPS = 256`, the former
  default exactly) instead of `2 span / max_steps`, so the step budget no longer shortens steps.
- Every preset's step budget is `DEFAULT_MAX_STEPS = 4096` (was 256, KIFS 128). Only rays that
  need the steps pay for them: on frame 27 primary rays alone take 47.4 ms against 37.1 ms at 256,
  the full render with 6 bounces 958 ms against 978 ms (no phantom paths to shade). The March steps
  slider goes to 16384. The status bar reports the share of samples out of steps (`Target::limited`,
  per-tile `tally` kernel); the per-pixel `moment` buffer became `stats` [luma², out-of-steps].
- The interior exit probe of unsigned fields (`exit_distance`) uses `EXIT_PROBES = 256` instead of
  the step budget, so the larger budget does not make refraction exits 16x more expensive.
- The out-of-steps share is reduced on the GPU to 4096 f64 partial sums (64 KiB readback per batch
  at any resolution).
- Saved scenes keep their budget and now show misses where they showed phantoms (the status bar
  warns). The local user templates in `~/.warpbro/templates` were raised from 256 to 4096 by hand.

### Rendering in ACEScg

- The tracer works in linear ACEScg (AP1, ACES white) instead of linear Rec.709. Authored colours
  (materials, palettes, sky, sun, lights) stay Rec.709 and are converted once at upload
  (`Scene::pack`, `palette::build_lut`); the Rec.709->AP1 matrix is built by vfx-ocio (Bradford) and
  matches the OCIO studio config. The library's emissive preset (authored in ACEScg) is stored as
  its Rec.709 equivalent, so it reaches the kernel unchanged.
- Environment EXRs are converted from their `chromaticities` (BT.709 when untagged); `.hdr` is Rec.709.
- Luminance (light selection, specular probability, saturation, environment importance sampling)
  uses the AP1 weights (`SS_LUMA`, the same as the Standard Surface lobe selection).
- OCIO input is always `ACEScg`; the Colour panel's Input choice and the persisted `input` field are
  gone (old scenes still load). OCIO-off / Reinhard paths convert AP1->Rec.709 before the sRGB OETF.
- Scene-linear EXR sequences are tagged with AP1 chromaticities; the display-light EXR stays BT.709.
- Expected differences: saturated colours under GI, saturation != 1, and the noise pattern (light /
  lobe probabilities) change; neutral scenes under a white sky render the same within noise.

### OpenEXR through exr-core

- Every EXR read/write goes through `src/exr_io.rs` over our `exr-core` (exr-rs, 1:1 OpenEXR port)
  instead of crates.io `exr`: sequence frames, the display-light "Save view" EXR (now one writer
  shared by `render.rs` and `render_service.rs`), lat-long environment maps and the CUDA glass probe's
  checker environment. Files are float32 R,G,B, ZIP, tagged with `chromaticities`
  (+ `whiteLuminance` = 100 nits on the display export); environment maps are size-checked from the
  header before any pixel is read; an existing file is refused unless overwrite is granted.
- egui-widgets-rs pinned to `6afc5cb` (egui-display writes EXR through exr-core). `cargo tree -i exr` is
  empty.

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

- Exclude the eight unused OFX Direct CUDA entry points from default WarpBro builds with the opt-in `ofx-direct` feature; enable them with `cargo oxide build --features ofx-direct`. Preserve the separate OFX plugin sources and artifacts. See [CUDA startup](docs/cuda-startup.md#exclude-unused-ofx-direct-kernels) for scope and validation limits.
- Verify both Direct feature variants: default PTX has 20 entries and no Direct entries; opt-in PTX has 28 entries with eight Direct entries. Isolated cold-cache loading fell from 215.7338802 s to 76.6563534 s in this run (about 2.8×); concurrent load differed, and cold startup remains above 5–10 seconds.
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
