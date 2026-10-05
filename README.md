# WarpBro

A browser for path-traced 3D fractals. The tracing kernels are **Rust compiled straight to PTX** by
NVIDIA's [cuda-oxide](https://github.com/NVlabs/cuda-oxide) rustc backend. The tracer uses no CUDA C++
or shader DSL: host code and tracing kernels live in the same crate and are built in a single
`cargo oxide build`. The UI is [egui](https://github.com/emilk/egui).
The project repository is [WarpBro](https://github.com/ssoj13/WarpBro), with SSH remote
`ssh://git@github.com/ssoj13/WarpBro.git`.

![WarpBro browsing the Menger sponge](docs/ui-menger.jpg)

Current usage and verification: [documentation index](docs/README.md), [workspace and cached playback](docs/workspace.md), [CUDA startup investigation](docs/cuda-startup.md), and [changelog](CHANGELOG.md).

## What it does

- **Eight distance-estimated families:** Mandelbulb (Mandelbrot or Julia form, angle scales,
  phases and per-iteration rotation), Mandelbox, quaternion Julia (a rotated 4D slice), KIFS
  (tetrahedron, octahedron, Menger), Kleinian, pseudo-Kleinian, Apollonian, and Hybrid (up to
  four bulb / box / KIFS-fold / inversion steps, repeated).
- **World objects:** Playa UUID nodes for fractals, cameras, directional lights, HDR environments,
  groups and materials, with parent transforms, visibility, layer spans, lock and solo.
  Multiple fractals take part in primary rays, shadows and reflections; multiple directional
  lights and one active HDR/EXR environment illuminate the same world.
- **Object animation:** an Outliner, Attribute Editor and AE-style Timeline share selection and
  transactional undo/redo. Numeric components and discrete properties use Playa animation.
- **Unidirectional path tracing:** sun-cone, sky and HDR next-event estimation, MIS with the
  power heuristic, and Russian roulette. A thin-lens camera gives depth of field.
- **GPU OIDN:** periodic scene-linear HDR denoising with primary-hit albedo/normal guides,
  followed by exposure and the display transform. Raw samples remain available.
- **Two material models**, supported by both the world tracer and legacy specialized kernels:
  - _Fast_: Lambert plus GGX.
  - _Standard Surface_: the full Autodesk Standard Surface (MaterialX port) with coat, sheen,
    thin film, anisotropy, and dielectric transmission. The [`standard-surface-bsdf`](https://github.com/ssoj13/render-rs/tree/cd72eb3ac4ad28b7b31c78826f192f383f4b6989) crate
    is called directly from the kernel.
- **14 palettes** and orbit-trap colouring (origin, plane, point).
- **Material Gallery:** actual World material nodes appear as asynchronous GPU-rendered sphere cards.
  The separate **Material Library** window shows the 69 curated presets from `fractal-materials` (derived from usd-rs `usd-mat-lib`) in adaptive grids across 12 categories (metals,
  brushed metals, plastics, car paint, ceramic, stone, wood, leather, velvet and fabric, rubber,
  paper, glass, emissive).
  - Presets are translated the way usd-rs `usd-hd-pt` maps UsdPreviewSurface to Standard
    Surface. Sheen (velvet) and anisotropy (brushed metal) select the Standard Surface kernels.
  - The `pt-material-ext` facing mix (pearlescent, oil slick) works in both models.
  - The fractal colour comes from either the palette or the material.
  - Path-traced glass uses Fresnel reflection, refraction, and depth-dependent absorption.
    Bottle-green glass and green water are available as presets. See [glass controls and geometry limits](docs/glass.md).
- **Unreal-style flight:** hold the right mouse button in the viewport to fly.
  - The mouse looks around; WASD moves; R or Space moves up, C down; Q/E rolls; Shift flies x4, Alt x0.1;
    the wheel sets the speed. Translation, look and roll all carry inertia and ease out
    (Settings > Camera controls: translate / rotate decay, inertial look, flip angle, multipliers).
  - The toolbar airplane switches free 6-DoF flight and horizon lock. Locked, the horizon stays
    level to a world plane: a short Q/E press tilts and springs back, holding past the flip angle
    turns the camera onto the next world plane, where it levels again.
  - The flight rig, its horizon lock and the key bindings come from the shared gitnexus-rs
    `cam-controls` / `cam-controls-egui` crates, like the Houdini orbit (left drag tumbles,
    middle drag pans, the wheel zooms; released drags coast and ease out with the rotate decay).
  - When you release the button, the orbit pivot sits in front of the camera, so orbiting
    continues from where you flew.
  - Backtick / tilde switches between horizon lock and free flight; switching to the lock
    levels a resting camera smoothly too. Roll and mode are saved with the camera and survive
    releasing RMB; the orbit keeps the roll.
  - H restores the loaded preset/bookmark's camera; F frames the fractal's declared bounds
    (or a 10×10×10 box), including object scale/offset/rotation.
  - Settings → Controls adjusts mouse sensitivity and flight speed, saved between runs.
- **The browser:**
  - an 18-preset gallery with GPU-rendered thumbnails;
  - bookmarks (scenes saved as JSON);
  - an Attribute Editor for object parameters;
  - a progressive viewport that switches to a half-resolution, 2-bounce preview while you drag;
  - screenshots;
  - final PNG renders up to 4K.

![gallery](docs/gallery.jpg)

![material library](docs/ui-materials.jpg)

## Origin

WarpBro is a CUDA port of `ofx-fractal`, the fractal engine of the ofx-rs OpenFX plug-ins
(WGPU/WGSL):

| ofx-rs / render-rs                                                                 | WarpBro                                                              |
| ---------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `fractal3d.wgsl`: estimates, march, normals, palette, sky                          | `src/gpu.rs`                                                         |
| `pathtrace.wgsl` + render-rs `pt-integrator`                                       | `src/gpu.rs`, `trace_path`                                           |
| `fractal3d.rs`: presets, `Frames`, `max_distance`, footprint                       | `src/scene.rs`                                                       |
| `uniform3d.rs`: the uniform layout                                                 | `src/params.rs`                                                      |
| `ofx-gen/palette.rs`                                                               | `src/palette.rs`                                                     |
| render-rs `standard-surface-bsdf`                                                  | SSH crate with opt-in `cuda-math` (`libm::*` → `f32` methods)        |
| usd-rs `usd-mat-lib` presets, `usd-hd-pt` translator, `pt-material-ext` facing mix | `src/materials.rs`, `src/gpu.rs`                                     |
| gitnexus-rs `cam-controls` / `cam-viewport` / `cam-controls-egui`                  | SSH crates (flight rig, horizon lock, shared fly bindings)           |

## Requirements

- An NVIDIA GPU (developed on an RTX 3080 Ti, `sm_86`) on Windows, Linux or WSL2.
  - On WSL2, install only the CUDA **toolkit**; the driver comes from Windows. Never install
    `cuda-drivers` or `nvidia-driver-*` inside WSL.
- CUDA Toolkit 13.x, LLVM/`llc` 21 or newer, and clang (for bindgen).
- Rust stable (currently **1.99**), with compiler-internal APIs enabled through `.cargo/config.toml` (see `rust-toolchain.toml`).
- CUDA crates and the backend come from [our Windows fork](https://github.com/ssoj13/cuda-oxide-windows),
  pinned to `f3f1098a776a630d8003f2229934e0563cdec252` with the Rust 1.99 fixes. Cargo fetches the
  checkout automatically; GitHub SSH access is required.
- `cargo-oxide` must come from the same fork and revision:

  ```sh
  cargo +stable install --force --locked --git ssh://git@github.com/ssoj13/cuda-oxide-windows.git --rev f3f1098a776a630d8003f2229934e0563cdec252 cargo-oxide
  cargo oxide doctor
  ```

  `python bootstrap.py d --fix` also installs or migrates the CLI to this revision; regular builds
  check its source without reinstalling it.

- Git dependencies resolve through GitHub SSH and track the latest `main` of every repository;
  Cargo.lock records the exact commits (`cargo update` moves them forward). OIDN is render-rs
  `pt-denoise-oidn` (ACEScg input), `standard-surface-bsdf` comes from the same render-rs with
  `cuda-math`. Our own repositories are SSH Git dependencies; third-party Git sources
  (burn, cubecl, cutile-rs, ...) stay on the HTTPS URLs their crates declare.
  The workspace material catalog lives in `crates/fractal-materials`.

## Build and run

On Windows, `python bootstrap.py d --fix` sets up the Rust tools, then `python bootstrap.py b`
builds the release binary `target/release/WarpBro.exe`. The bootstrap discovers the MSVC, Windows SDK and CUDA environment
through `vcv-rs`, so a Developer Command Prompt is unnecessary. After linking, it initializes
the exact executable's CUDA module and parameter block to warm the driver's JIT cache.
A module/ABI initialization failure now fails the build instead of appearing only in the UI.
Use `--skip-cuda-warmup` for packaging/cross-builds without a GPU. For a direct
`cargo oxide build`, run `target/release/WarpBro --warmup-cuda` afterward.

On Linux / WSL2:

```sh
export CUDA_HOME=/usr/local/cuda CUDA_OXIDE_LLC=/usr/bin/llc-22
python bootstrap.py r                                  # build, warm the CUDA cache, then open
./target/release/WarpBro --gallery out 1920 1080 256    # every preset to out/*.png
./target/release/WarpBro --bench 960 540 32             # timing table
```

**Controls:**

| input             | action                                                                                                     |
| ----------------- | ---------------------------------------------------------------------------------------------------------- |
| left drag         | orbit (coasts on release); Shift: snap to the views along the world axes                                  |
| middle drag       | pan (coasts on release)                                                                                    |
| wheel             | zoom                                                                                                       |
| hold right button | fly: mouse looks, WASD moves, R/Space up, C down, Q/E rolls (tilts under the lock), Shift x4, Alt x0.1, wheel speed |
| double-click      | recentre                                                                                                   |
| `Tab`             | hide the panels                                                                                            |
| `Space`           | pause                                                                                                      |
| backtick / tilde  | switch horizon lock / free flight                                                                          |

The interface uses `egui-dock`: drag tabs to rearrange, split or float panels, including Settings.
The top bar uses **File**, **Edit**, **View**, **Render** and **Window** menus, following Playa.
Open Settings through **Edit → Settings…**; use **Window** to reopen panels or **Reset layout**
to restore their arrangement. The floating toolbar sits against the viewport's top edge, with
direct exposure (EV, with a reset to 0), colour view (the gear opens Settings → Color), proxy
resolution, target samples, pause, the denoise switch (off shows the raw samples at once) and
free-flight controls.
The top-right layout manager uses Playa's shared `egui-layout-manager`: save, select, rename
or delete named workspaces; **Layout → Update selected** replaces the selected preset and
**Layout → Reset to default** restores the default arrangement. Panel layout, named workspaces
and toolbar position are saved between runs. **Settings → Fonts** (also **Window →
Fonts…**) selects the built-in font family, body/control, small, heading and monospace text sizes,
and UI scale.

One universal **Attribute Editor** edits the selected node of any supported type, including materials.
Every number has a slider over its useful span (type past it; hard limits hold in the document),
colours use the HDR picker, vectors expand into per-channel sliders, parameters the current
formula or colouring does not use are greyed with the reason, sections are tinted by kind, and
right-click > **Show in timeline** expands and scrolls the timeline to the property.
It follows Playa's original composition: `egui-attr-grid` inside
`egui-titlebar::CollapsingSection`, with typed property controls from `egui-widgets-rs`.
Short labels keep rows compact; tooltips retain full property paths. Shared attribute metrics
set label/value columns, numeric component widths, square icon buttons and vertical alignment
across the editor and Timeline. **Settings → Controls** adjusts these compact layout metrics.
Numeric gestures defer their undo transaction until completion. Drag the vertical divider between Timeline names and time bars to resize the outline; its width is saved in settings, and double-click restores the default.

One status bar provides resizable sections. The current Materials revision makes the gallery
show actual World material nodes. Clicking a card selects its UUID and opens the existing
Attribute Editor. Assignment uses the object's Material attribute or **Assign to selected objects**
in the card context menu. The sibling toolbar buttons **+ New material** and **Create from preset**
create a default node or a preset-based node respectively, without assigning it. The card's **Apply preset to this material** action updates the existing UUID
and every consumer. **Library…** opens the separate preset library; clicking a preset creates and selects a workspace node without assigning it. Gallery and Bookmarks also adapt their columns to panel width. **Refresh preview** retries terminal rendering/colour failures; stale
thumbnail results are rejected. Ordinary clicks only select the material for the shared editor.
**Apply to object** in the material Attribute Editor assigns to the remembered eligible object
selection through one validated Undo command. Material preview identity follows evaluated
parameters; renames and unchanged values retain previews, while parameter changes reject
stale pending results.

The shared toolkit combines aligned Attribute Editor APIs and the pinned Timeline ruler.
The production release build and headless render checks passed after updating the host pin.
Native narrow-panel, cached-playback and latency checks remain pending; see
[workspace behavior](docs/workspace.md) and [Plan 1](plans/plan1.md).

The **Outliner** uses `egui-outliner` for the parent tree. The **Timeline** shows the same
objects as layers with spans, property groups and component lanes; stopwatch and diamond
controls enable animation and add/remove keys. Rename, reparent and layer order preserve
UUID-based property addresses. During vertical scrolling the ruler and outline header stay
fixed while layer names, property lanes and bars scroll together. Green/blue coverage marks
resident final/draft RAM frames. Ins plays the selection; Shift+Ins caches at target samples
then plays; Ctrl+Shift+Ins caches at 1 SPP then plays. Only completed native viewport frames
enter the automatic final cache. Disk caching remains pending. Selection chooses the editing target; visibility and solo
filter the rendered world. Solo is saved and applies equally to preview and export.
Visibility and the half-open span `[start, end)` inherit through parents; a locked ancestor
prevents edits. The active camera remains active when its layer is hidden.

Bookmarks save a Playa `SubnetFile` with node attributes, material UUID assignments and
arbitrary JSON metadata. Each node separates its GPU baseline (`gpu`) from Playa attributes
and animation (`host`), discrete-value dictionaries and metadata. Evaluated `Scene` objects
and device buffers are temporary. Legacy scenes migrate their transforms and keys, including
the rotation convention; layer order is separate from hierarchy and physical occlusion.

**Settings → Display / Color** uses the same `egui-prefs2` layout as exr-view. Display uses
`egui-display::settings_ui` directly: output, SDR reference white and HLG display peak, with
OS values used automatically unless overridden. Color is the copied exr-view OCIO panel:
config (built-in, `$OCIO`, `.ocio`/`.ocioz` file), Display, View, Look, Reload and monitor
presets. The tracer renders in **linear ACEScg** (AP1, ACES white), which is always the OCIO
input (a config without an `ACEScg` colour space is reported, not substituted). Material,
palette, sky and light colours are authored in linear Rec.709 / sRGB primaries and converted to
ACEScg once at GPU upload; environment EXRs are converted from their `chromaticities` (BT.709
when untagged), `.hdr` maps are read as Rec.709. With OCIO off or Reinhard, the image is
converted to Rec.709 before the sRGB encoding.

For HDR on screen, enable HDR in the OS, choose an available HDR output in Display, then
select **HDR · 1000 nits** in Color. The ACES rendering peak and UI reference white are
independent. A saved HDR scene on an SDR output gets a separate SDR ACES preview.
PQ/HDR render targets remain selectable in Color on SDR screens, including for HDR export;
only the actual window-output modes in Display depend on the connected display.

**Settings → Display → GUI FPS** sets the interface refresh rate independently of rendering
(15–240 FPS, default 60). A dedicated CUDA worker owns all progressive targets, thumbnails
and final renders. Commands and completion events use bounded queues; viewport requests and
frames use replaceable slots, so the GUI takes the newest ready frame without waiting for
CUDA. Scene changes invalidate older generations. Settings writes, screenshot output and
OCIO config loading also run in background workers. Window presentation uses a separate wgpu
device from offscreen OCIO processing, so surface reconfiguration does not wait for that
worker's queue. The GUI still performs drawing and GPU presentation; the configured FPS is a
target, not a guarantee under GPU or system load.

### OIDN denoising

In the render controls, **OIDN denoise** is enabled by default. **Denoise every N samples**
defaults to 128; 0 disables periodic passes while retaining the final pass. **Denoise guides**
selects Color, Color + Albedo, or Color + Albedo + Normal (default). **Denoise quality** offers
Fast, Balanced (default), and High. Turning denoising off shows raw radiance without discarding
samples. Changing mode or quality refilters current samples; changing the interval preserves
accumulation and the last usable result.

CUDA accumulates primary-hit albedo and world-normal sums with their counts. The worker
passes normalized scene-linear ACEScg HDR to render-rs `pt-denoise-oidn` (OIDN autoexposure on
ACEScg luminance) before exposure, saturation
or OCIO. No firefly clamp changes the HDR input range; NaN protection acts on the denoiser's
input only. Raw radiance and guide accumulators remain untouched. Until the next pass,
the viewport can show the last denoised result; its status reports that result's sample count
and elapsed milliseconds. An OIDN error shows **OIDN failed** with diagnostic details and
falls back to raw output, separately from colour-processing errors.

A persistent worker processor reuses the existing offscreen wgpu device and embedded model
weights. CUDA readback, wgpu uploads, inference and result readback run on workers; the GUI
receives ready frames. New render generations and dimensions invalidate target results.
Screenshots save the already computed frame; final renders and each export frame request
a final denoise pass when enabled, even below the periodic threshold. Release validation
passed 92 regular tests and all three explicitly enabled GPU tests (95 total) after the
final SSH dependency switch, including
HDR preservation, unchanged raw samples and final denoising below the interval.
The final release test logs are [the regular suite](target/verification/ssh-final-tests.out)
and [the explicitly enabled GPU suite](target/verification/ssh-final-gpu-tests.out).

All output goes to `~/.warpbro/out/<local date_time>/`, one new folder per export or screenshot.
**Snapshots** (the camera button of the viewport toolbar, or File): a click saves the viewport as
the monitor shows it - 8-bit sRGB PNG on an SDR monitor, HDR10 BT.2020/PQ 16-bit PNG with `cICP`,
`mDCV` (the view's peak) and measured `cLLI` on an HDR one. Right click (or File) also offers the SDR
PNG and the HDR10 PNG explicitly, and **Display EXR**: unquantized linear Rec.709 display light with
chromaticities and `whiteLuminance = 100` (display-referred, not a scene-linear master). HDR PNGs are
named `*.pq.png` / `*.hlg.png`: in a viewer that ignores `cICP` they look washed out.

```sh
./target/release/WarpBro --gallery out 1920 1080 256 --hdr --display-exr
CUDA_HOME=/usr/local/cuda CUDA_OXIDE_LLC=/usr/bin/llc-22 cargo oxide test -- --release
```

### Render / Encode

Open **Render → Render / Encode…** or **Window → Render / Encode**. This dockable panel
uses Playa's shared encoder schema. Pick the format tab, a file **Name**, resolution,
**Samples / frame** and the inclusive frame range (**Current frame** renders the frame under the
playhead). A one-frame range writes `name.ext`; a longer range writes `name.000001.ext`, ...

- **PNG:** the monitor rendering baked in, as **SDR · 8-bit sRGB / BT.709**, **HDR10 · 16-bit PQ /
  BT.2020** (`cICP`, `mDCV`, `cLLI`) or **HLG · 16-bit BT.2020** (`cICP`, `mDCV`), with the HDR
  mastering peak in nits. HDR keeps SDR white at 100 nits.

Every output renders through its **own output transform** from ACEScg, independent of the viewport:
EXR stays scene-linear; SDR PNG and video use the scene's SDR view (or the config's first SDR
display); HDR10 PNG uses a PQ display and the HDR view nearest the chosen peak; HLG PNG an HLG
display. The panel shows the choice under **Output transform** and lets you override display / view.
- **EXR sequence:** float RGB scene-linear ACEScg tagged with AP1 chromaticities; exposure and the
  display transform are excluded.
- **HEVC / ffmpeg-rs:** hardware **GPU · Vulkan Video** encoding to MP4 or MOV by default, with rational FPS and QP 0–51. **CPU · Kvazaar (I-frames)** is an explicit software alternative using independent frames to avoid reproduced corruption in the pinned inter-prediction path; hardware initialization failures are reported instead of silently switching encoders. Output is SDR 8-bit YUV 4:2:0 with Rec.709 primaries and sRGB transfer;
  the display transform is baked in. Width and height must be even. HDR video is unavailable.

Each frame receives the requested sample count. The World document is frozen at export start;
each frame evaluates its Playa animation at that frame's time, including transforms, lights,
visibility and discrete keys. When OIDN is enabled, the scene-linear EXR output contains the
final denoised radiance; exposure and OCIO remain excluded.
An autonomous coordinator advances rendering and a bounded writer queue handles encoding
and file output even when the GUI stops updating. **Cancel** stops sampling new frames, drains completed frames and flushes delayed codec packets to publish a playable partial movie. Completed frames remain; cancellation before the first complete video frame creates no movie. Every export
writes into its own new folder, and finished outputs are published atomically.

**Templates** (File → Templates) are built into the binary; a `*.frac.json` in
`~/.warpbro/templates` with the same name overrides a built-in, any other file is added. The
binary needs no external files.

**Camera slots:** the five buttons at the right of the viewport toolbar restore a camera on
left click and store the viewport camera on right click (paste / copy); the colour presets work the
same way, and Settings → Controls → *Swap copy/paste mouse buttons* exchanges the buttons for both.
The slots are kept with the application settings, and a restore is authored like any viewport
navigation (Auto Key, undo).

For compatibility, the local checkout folder, profile and data locations retain the legacy
`frac-rs` name during the WarpBro rename.
Settings are saved in the platform config directory (`~/.config/frac-rs/settings.json` on
Linux); the colour selection also travels with scene bookmarks. Older bookmarks default
to the ACES 2.0 SDR view.

The root `frac_rs.ll`, `frac_rs.linked.ll`, `frac_rs.linked.opt.ll` and `frac_rs.ptx` are
cuda-oxide build intermediates: LLVM IR, linked IR, optimized IR and NVIDIA PTX. The
backend defaults to the current directory; `.cargo/config.toml` now redirects these files
to `target/oxide-ptx` through `CUDA_OXIDE_PTX_DIR`. They are ignored by Git and can be
regenerated by a build; the runtime loads the embedded device bundle from the executable.

Bookmarks are stored in `~/.local/share/frac-rs/bookmarks`. Screenshots and renders go to
`~/Pictures/frac-rs`.

Current follow-up work is tracked in [render optimization](plans/plan2.md),
[MP4 quality](plans/plan3.md), and [scene files, animated presets and orbit animation](plans/plan4.md).
File Open/Save/Save As and five 250-frame presets (frames 0–249 at 24 FPS) are implemented.
Camera orbit speed (degrees/second) and phase are animatable World attributes; old scenes default
to zero speed. Timeline evaluation supports independent seeking and animated speed.
The current release suite passed 211 tests with 8 ignored GPU/visual probes (62.07 s).
The earlier final production release build passed without Rust warnings; the latest startup build
passed with two dead-code warnings. All 250 final preset curves
are covered by CPU tests. The 1,250-frame GPU run at 160×90/4 SPP preceded only the last bounded
Chrome tune; the final 15-still rerun at 640×360/32 SPP passed and its
[contact sheet](target/verification/unfolding-v2/contact.png) was inspected, including Chrome's accepted first frame. Native File interactions and manual UI latency remain pending.
The captured Nsight kernel baseline and CPU preparation measurements do not establish a GPU speedup.

For reproducible preset fixtures, use `--animated-fixtures DIR [W H SPP]`: it writes first,
middle and last PNGs plus `scene.frac.json`. Add `--all-frames` for all 250 frames using seed 0
and a reused render target. Initial 32-frame MP4 comparisons passed for all five presets at 160×90/4 SPP: QP 18/medium
improved RGB PSNR over QP 27/veryfast, with correct frame count and zero-origin timing.
Full 250-frame/high-resolution encoded output remains unverified.
The current hardware backend, safe CPU mode, partial-video cancellation and independent decoder investigation are documented in [the export follow-up](plans/plan6.md). Its moving-source comparison supersedes the earlier inter-preset export configuration.

The [MP4 quality harness](plans/plan3.md#reproduce-the-quality-comparison) reproduces the measured
QP/preset tradeoff and verified zero-origin rational video timing on deterministic CPU patterns.

## Startup diagnostics

The default bootstrap build warms the CUDA driver's JIT cache before reporting success.
This moves cold compilation out of the next workspace launch. Clearing/disabling the cache,
changing drivers, or moving to another GPU may require warming again. Context and kernel-load timings remain
logged separately. See [startup measurements and build choices](docs/cuda-startup.md).
A reusable runtime module cache and a viewport initialization overlay remain unimplemented.

## Performance

Current production-world CUDA profiling and convergence work is tracked in [Plan 7](plans/plan7.md). The new `--world-bench DIR W H SPP --case fast-metal --seed 0 --batch 4` route retains raw linear RGB f32, PNG and frozen-scene JSON. Other cases are `fast-dielectric`, `chrome-mid`, `diffuse-mid` and `opal-mid`. `tools/convergence.py` compares multiple independent seeds against a common high-SPP reference and reports error and elapsed time. The validated base passed 193 CPU/CUDA tests. On five-seed 256×256/32-SPP paired runs, Fast metal/dielectric throughput improved about 1.61×; Full cases were 1–6% slower after the corrected final environment ray. At 128×128, metal error fell about 11.6% at fixed SPP; MSE×time estimates improved 1.63× for metal and 1.52× for dielectric. These are workload-specific measurements, not a universal convergence guarantee. A Full shading helper experiment was rejected because its paired speed changes were inconclusive; Plan 7 records the accepted base, experiment and exact limits.

These numbers are from an RTX 3080 Ti, 1280×720, 6 bounces, at the preset views (`--bench`;
the full table is in [docs/bench-1280x720.txt](docs/bench-1280x720.txt)):

| preset                                      | ms / spp | Msamples/s |
| ------------------------------------------- | -------: | ---------: |
| KIFS (tetrahedron)                          |      5.2 |        177 |
| Octahedron KIFS                             |      8.6 |        107 |
| Quaternion Julia                            |     12.6 |         73 |
| Apollonian                                  |     12.2 |         76 |
| Kleinian                                    |     13.3 |         69 |
| Mandelbulb                                  |     35.4 |         26 |
| Menger sponge, Standard Surface             |     20.0 |         46 |
| Mandelbulb power 12, gold, Standard Surface |     69.2 |         13 |

**What makes it fast:**

- **World and specialized kernels.** The world tracer dispatches each hit object's formula
  and material at runtime and checks all objects for primary, shadow and reflection rays.
  The legacy single-object path retains one specialized kernel per (family, material model).
  GPU inlining is tuned per path, including outlined world-estimate functions.
- **Parameters in `#[constant]` memory.** Every lane reads the same slot, so the value is
  broadcast to the whole warp.
- **Bounding-sphere ray clipping.** Rays march only inside the escape radius or ball bound. This
  is exact: outside it the estimate is "outside" anyway. Without it, every camera and escaping
  bounce ray spent about 40 capped steps crossing empty space.
- **A trig-free power-8 Mandelbulb.** Three angle doublings replace `acos`/`atan2`/`pow`.
- **Step factor 0.85** (ofx-fractal uses 0.5). On the presets it measured unbiased (mean
  luminance unchanged) and is 1.5× faster. A slider brings back 0.5.
- **8×4 pixel tiles per warp,** so neighbouring rays share their march paths.
- **Float colour pipeline.** Normalized scene-linear ACEScg RGBA32F passes through optional
  OIDN, then exposure and saturation. The shared `vfx-ocio` GPU runtime applies the complete
  ACES 2.0 output transform from oiio-rs; no Narkowicz approximation. Display and denoise
  setting changes preserve accumulated samples.
- **Shared SDR/HDR presentation.** `egui-display` renders an extended-sRGB float canvas to a
  supported SDR 8/10-bit, HDR10/PQ, HLG or scRGB surface. HDR output negotiates both the pixel
  format and colour space; unavailable modes are disabled in Settings.

Under WSL2 CUDA↔graphics interop is unavailable (WSLg is Mesa d3d12). The current bridge
reads back CUDA float data, uploads it to the shared offscreen wgpu device for OIDN/OCIO,
reads back the result and display light/codes, then uploads the viewport. Pipelines and the
OIDN processor are reused, but these transfers cost more than the former RGBA8-only path.
The timings above predate this bridge, the World tracer and OIDN integration.

## Layout

```text
src/gpu.rs        the kernels (#[cuda_module]): estimates, march, normals, lighting, integrator, tonemap
src/scene.rs      evaluated Scene, legacy serde, formulas, presets and parameter packing
src/world.rs      Playa World document, migration, evaluator, commands and undo/redo
src/world_ui.rs   Outliner, object Attribute Editor and layered component Timeline
src/presets.rs    five authored 250-frame animated World presets
src/camera_orbit.rs   deterministic timeline integration of camera orbit speed
src/params.rs     parameter-block slots shared by host and device
src/render.rs     CUDA context, progressive targets, kernel dispatch, PNG output
src/app.rs        the egui browser, shared Settings panel and viewport toolbar
src/dock.rs       dockable panels, named layout manager and viewport toolbar
src/render_service.rs   CUDA worker, bounded command/event bus and latest-frame mailbox
src/io_service.rs       asynchronous full-scene open/save, settings, bookmarks and screenshots
src/export.rs     autonomous render coordinator, EXR sink and ffmpeg-rs HEVC writer
src/inspector.rs  scene controls from egui-widgets-rs attribute editors
src/denoise.rs    worker OIDN settings, target cadence and shared-device processor
src/color.rs      working space (ACEScg) matrices and luma, cached vfx-ocio GPU colour processing
src/ocio.rs       OCIO controls and background config loader, adapted from exr-view (BSD-3-Clause)
src/window.rs     winit/wgpu shell + egui-display float canvas and SDR/HDR swapchain
src/palette.rs    the 14 palettes
src/materials.rs  host adapter for the shared fractal-materials catalog
crates/fractal-materials   renderer-independent material preset definitions
src/preview.rs    bounded final/draft Playa RAM cache and playback state
src/hotkeys.rs    global-first, panel-scoped command bindings
standard-surface-bsdf   SSH dependency: Autodesk Standard Surface (MaterialX port, Apache-2.0)
cam-controls, cam-viewport   SSH dependencies: gitnexus-rs camera rigs (PolyForm-Noncommercial-1.0.0)
```

## Licences

`standard-surface-bsdf` is a derivative of MaterialX (Apache-2.0); its source checkout
contains `LICENSE` and `NOTICE`. `cam-controls` and `cam-viewport` come from gitnexus-rs
and are under PolyForm-Noncommercial-1.0.0. The retained vendor copies are historical;
Cargo uses the pinned SSH dependencies above. The fractal formulas credit their sources in the ofx-rs code they were ported from
(Knighty's KIFS, Leys' Kleinian, Mandelbulber's pseudo-Kleinian, and others).

The OCIO panel in `src/ocio.rs` is adapted from exr-view. The ACES 2.0 presets follow the [Academy output transform parameters](https://docs.acescentral.com/system-components/output-transforms/parameters/).
