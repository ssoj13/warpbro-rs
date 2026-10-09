# Changelog

## Unreleased — 2026-10-09

### Render settings as graph nodes

- **RenderSettings / QualitySettings / ViewportSettings / OutputSettings** are document nodes
  with Profile / Template catalogs (Settings → Render & Viewport), edited in the Attribute
  Editor. Moving, Still, Manual and Output bind render profiles; Auto switches Moving / Still
  on motion with a settle delay, Locked uses Manual; Pause and Freeze hold the viewport.
- The viewport toolbar labels the profile menu with the binding it renders (Moving / Still /
  Manual) and lists profile, method, samples and resolution in its tooltip.
- **OutputSettings** holds the export recipe (format, encoder, size, QP / CRF, PNG encoding,
  HDR peak, PNG video, OCIO override, denoise at completion); Render / Encode binds Output render
  and Output file and authors its edits into the node with Undo. Preferences keep only the job
  (file name, frame range, frame rate). An OCIO override of the wrong output kind is refused.
- Details: [docs/render-profiles.md](docs/render-profiles.md).

### Animation and authoring

- Keys live on curves-rs Tracks (per-side tangents, a *New key type* setting); the camera orbit
  integrates its animated speed exactly. Preferences moved onto the attribute grid; Reset to
  default works for host-edited rows. Old-scene loaders and migrations were removed.
- Camera recording (transform, focus, zoom, f-number) and five camera slots on the toolbar.

### Renderer

- Mandelbulb DE boundary narrowed to the world inverse affine: single-bulb scenes about 27%
  faster (needs the cuda-oxide fork's `noinline` support, `be40bf2`); other routes unchanged.

### Build and CI

- `python bootstrap.py ci`: toolchain, `cargo fmt --check`, `clippy --all-targets -D warnings`,
  tests without a GPU, a release build for sm_75 and `dist/warpbro-<version>-<platform>.zip`.
  GitHub Actions runs it on Windows and Linux; a `v*` tag publishes both archives.
- Tests that need a CUDA device are named `cuda_*`, other GPU APIs `gpu_*`.
- A release archive needs one `WarpBro --warmup-cuda` after unpacking (README, *CUDA startup*).

## 2026-10-04

### PNG export video (ffmpeg stopgap)

- The PNG export's **Video** option encodes the finished sequence with the `ffmpeg` on PATH:
  ProRes 4444 XQ `.mov` (10-bit, the most `prores_ks` takes) or HEVC 10-bit `.mp4`, tagged
  like the PNGs. An HDR10 HEVC carries the
  mastering display and the clip's MaxCLL / MaxFALL, aggregated from what the PNGs measured
  (`render_service::HdrLevels`, `egui_display::screenshot::Capture::content_light`). Written
  through `AtomicOut` (temp sibling, published when ffmpeg succeeds).
- Saved settings compare the whole `ExportSettings`: a change of any export field is saved. Until ffmpeg-rs carries HDR (`ffmpeg-rs/BUG3.md`).
- FPS and Quality are one grid row each for the Video format and the PNG's video.
- SDR video (the built-in HEVC and the PNG export's video) is encoded for BT.1886 and tagged
  BT.709 (1/1/1): `color::bt1886_code` (`L^(1/2.4)` of relative display light) replaces the sRGB
  codes the HEVC carried under an sRGB tag (13), which QuickTime and ProRes headers do not know.
  A video display now shows the light the viewport shows; QuickTime shows BT.709 at about
  gamma 1.96 (lighter).
- Test: an HDR10 PNG of the ACES 2.0 500-nit P3 view records the measured peak in `mDCV` and keeps
  colour outside BT.709.

### Snapshots, slot buttons, Render / Encode panel

- Viewport snapshot: a camera button on the toolbar (click: as the monitor shows it; right click and
  File: SDR PNG, HDR10 PQ PNG, display EXR), one `Frame::save` path. The monitor is read from
  `egui_display::DisplayState` when saving (`Monitor`): on an HDR monitor an SDR view's white is
  written at the monitor's SDR white, so the file is as bright as the screen (no HDR monitor:
  BT.2408's 203 nits).
- One display-light model (`color::DisplayLight`): an HDR view's light is absolute with its peak
  measured on the OCIO transform itself (a very bright probe through the view's tone curve), not
  parsed from the view's name; `render_service::hdr_scale` turns it into file nits for every PNG
  writer. An HDR PNG export records the rendered view's peak. The export renders through the
  scene's view when it is an HDR view of the right kind, else through the view whose measured peak
  is nearest the export's HDR peak (no view-name parsing). An OCIO export turns the legacy Reinhard
  curve off, which would bypass the output transform.
- One file-name module (`fs_name`): stems keep any script ("Медная турбина" -> `медная-турбина`),
  Windows device names get `_`, a typed export name is checked like the filesystem would; every
  sequence numbers before the whole suffix (`shot.000042.pq.png`).
- Attribute hover hints: `egui-attr-grid` `AttrField::hint`, filled from one table
  (`world::attribute_hint`) for the Attribute Editor and Render settings; a test fails for any
  attribute without one.
- **Changed default:** CamClip and colour presets now restore on left click and store on right
  click; Settings → Controls → Swap copy/paste mouse buttons restores the old layout for both.
- Render / Encode: one grid, the settings every format shares once, then the Format row and that
  format's options.

### Sphere tracing: an unresolved march has one policy per ray (BUG1)

- `march` reports `Hit`, `Miss` or `Unresolved` (out of steps inside the interval); the integrator
  decides: a camera ray shows the background (as in Mandelbulber and Fragmentarium), a bounce ray
  ends the path (an unknown direction brings no light - its environment would leak into crevices),
  a shadow ray counts as blocked. Every unresolved march is counted for the status bar. A camera
  ray used to take its closest sample within a pixel (Keinert et al. 2014, Enhanced
  Sphere Tracing 3.2, a real-time technique): on BUG1 frame 27, 37% of the primary hits were such
  samples - 52% of them rays passing the surface, 48% lying 0.3 scene units before the real hit,
  normals ~48 degrees off. WarpBro's opt-in `ofx-direct` kernels keep Keinert's rule at his half
  pixel. The OFX plug-in builds from its own copy (`ofx-fractal/kernels/source`), not changed here.
- The longest step is a fixed share of the ray's interval (`STEP_CAP_STEPS = 256`, the former
  default exactly) instead of `2 span / max_steps`, so the step budget no longer shortens steps.
- Every preset's step budget is `DEFAULT_MAX_STEPS = 4096` (was 256, KIFS 128). Only rays that
  need the steps pay for them: on frame 27 primary rays alone take 47.4 ms against 37.1 ms at 256,
  the full render with 6 bounces 958 ms against 978 ms (no phantom paths to shade). The March steps
  slider goes to 16384. The status bar reports the share of samples out of steps (`Target::unresolved`,
  `tally` kernel); the per-pixel `moment` buffer became `stats` [luma², unresolved samples].
- Glass interior probes (`Render::glass_probes`, Render settings, default 256 as before): the
  probe count of an exit through glass whose field is zero inside. It is the resolution of
  interior walls, not a step budget: tied to the 4096-step budget a glass Mandelbulb rendered
  10-30x slower and visibly darker (more walls found), so the trade-off is an explicit setting.
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

- Exclude the eight unused OFX Direct CUDA entry points from default WarpBro builds with the opt-in `ofx-direct` feature; enable them with `cargo oxide build --features ofx-direct`. Preserve the separate OFX plugin sources and artifacts. See README, *CUDA startup*.
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

The production release build passed. The release test executable passed **199 tests, 0 failed, 7 ignored** on the final repeat (72.71 s). The first run had 198 passes and one 90-second export-cancellation timeout under load; its isolated repeat passed in 1.39 s. The seven ignored GPU tests were not rerun in this update. The rebuilt and retried 32×32 legacy headless checks completed successfully. Shared Timeline, attribute-grid, and layout/configuration tests passed after combining the toolkit branches. See CUDA startup (README, *CUDA startup*) for exact timings and load caveats. Earlier full-suite counts and native-check limits remain in the dated plans.

Disk preview caching, Curve Editor integration, and CUDA compiled-module caching remain open. The subsequent glass update is described above.
