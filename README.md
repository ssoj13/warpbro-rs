# frac-rs

A browser for path-traced 3D fractals. The GPU kernels are **Rust compiled straight to PTX** by
NVIDIA's [cuda-oxide](https://github.com/NVlabs/cuda-oxide) rustc backend. There is no CUDA C++,
no WGSL and no DSL: host code and kernels live in the same crate and are built in a single
`cargo oxide build`. The UI is [egui](https://github.com/emilk/egui).

![frac-rs browsing the Menger sponge](docs/ui-menger.jpg)

## What it does

- **Eight distance-estimated families:** Mandelbulb (Mandelbrot or Julia form, angle scales,
  phases and per-iteration rotation), Mandelbox, quaternion Julia (a rotated 4D slice), KIFS
  (tetrahedron, octahedron, Menger), Kleinian, pseudo-Kleinian, Apollonian, and Hybrid (up to
  four bulb / box / KIFS-fold / inversion steps, repeated).
- **Unidirectional path tracing:** sun-cone and gradient-sky next-event estimation, MIS with the
  power heuristic, and Russian roulette. A thin-lens camera gives depth of field.
- **Two material models**, each compiled as its own set of kernels:
  - *Fast*: Lambert plus GGX.
  - *Standard Surface*: the full Autodesk Standard Surface (MaterialX port) with coat, sheen,
    thin film and anisotropy. The [`standard-surface-bsdf`](vendor/standard-surface-bsdf) crate
    is called directly from the kernel.
- **14 palettes** and orbit-trap colouring (origin, plane, point).
- **Material library:** the 50 curated presets of usd-rs `usd-mat-lib` in 12 categories (metals,
  brushed metals, plastics, car paint, ceramic, stone, wood, leather, velvet and fabric, rubber,
  paper, glass, emissive), each shown as a GPU-rendered sphere swatch.
  - Presets are translated the way usd-rs `usd-hd-pt` maps UsdPreviewSurface to Standard
    Surface. Sheen (velvet) and anisotropy (brushed metal) select the Standard Surface kernels.
  - The `pt-material-ext` facing mix (pearlescent, oil slick) works in both models.
  - The fractal colour comes from either the palette or the material.
  - Glass renders opaque: a distance-estimated fractal has no interior to refract through, so
    glass becomes a clear-coated smooth dielectric.
- **Unreal-style flight:** hold the right mouse button in the viewport to fly.
  - The mouse looks around; WASD moves; Q and E go down and up; Shift boosts; the wheel sets
    the speed.
  - It uses `cam-controls` `SpaceFlight` with FPS damping and a level horizon.
    Inertial mouse-look is copied from the ready `nodes-rs` implementation of the shared crate.
  - When you release the button, the orbit pivot sits in front of the camera, so orbiting
    continues from where you flew.
  - Backtick / tilde switches between a level horizon and free flight. In free flight,
    Q/E rolls and R/F moves up/down; all six flight axes have damped inertia. Roll and mode
    are saved with the camera and survive releasing RMB.
  - H restores the loaded preset/bookmark's camera; F frames the fractal's declared bounds
    (or a 10×10×10 box), including object scale/offset/rotation. F without RMB frames;
    RMB+F still moves down in free flight.
  - Settings → Controls adjusts mouse sensitivity and flight speed, saved between runs.
- **The browser:**
  - an 18-preset gallery with GPU-rendered thumbnails;
  - bookmarks (scenes saved as JSON);
  - an inspector for every parameter;
  - a progressive viewport that switches to a half-resolution, 2-bounce preview while you drag;
  - screenshots;
  - final PNG renders up to 4K.

![gallery](docs/gallery.jpg)

![material library](docs/ui-materials.jpg)

## Origin

frac-rs is a CUDA port of `ofx-fractal`, the fractal engine of the ofx-rs OpenFX plug-ins
(WGPU/WGSL):

| ofx-rs / render-rs | frac-rs |
|---|---|
| `fractal3d.wgsl`: estimates, march, normals, palette, sky | `src/gpu.rs` |
| `pathtrace.wgsl` + render-rs `pt-integrator` | `src/gpu.rs`, `trace_path` |
| `fractal3d.rs`: presets, `Frames`, `max_distance`, footprint | `src/scene.rs` |
| `uniform3d.rs`: the uniform layout | `src/params.rs` |
| `ofx-gen/palette.rs` | `src/palette.rs` |
| render-rs `standard-surface-bsdf` | `vendor/standard-surface-bsdf` (used unchanged, except `libm::*` → `f32` methods) |
| usd-rs `usd-mat-lib` presets, `usd-hd-pt` translator, `pt-material-ext` facing mix | `src/materials.rs`, `src/gpu.rs` |
| gitnexus-rs `cam-controls` / `cam-viewport` | `vendor/cam-controls`, `vendor/cam-viewport`; inertial `SpaceFlight::Look` from nodes-rs |

## Requirements

- An NVIDIA GPU (developed on an RTX 3080 Ti, `sm_86`) on Linux or WSL2.
  - On WSL2, install only the CUDA **toolkit**; the driver comes from Windows. Never install
    `cuda-drivers` or `nvidia-driver-*` inside WSL.
- CUDA Toolkit 13.x, LLVM/`llc` 21 or newer, and clang (for bindgen).
- The pinned nightly `nightly-2026-08-28` (see `rust-toolchain.toml`).
- `cargo-oxide`:
  ```sh
  cargo +nightly-2026-08-28 install --git https://github.com/NVlabs/cuda-oxide.git cargo-oxide
  cargo oxide doctor
  ```

## Build and run

```sh
export CUDA_HOME=/usr/local/cuda CUDA_OXIDE_LLC=/usr/bin/llc-22
cargo oxide run                                         # the browser
./target/release/frac-rs --gallery out 1920 1080 256    # every preset to out/*.png
./target/release/frac-rs --bench 960 540 32             # timing table
```

**Controls:**

| input | action |
|---|---|
| left drag | orbit |
| middle drag | pan |
| wheel | zoom |
| hold right button | fly: mouse looks, WASD moves, Q/E down/up, Shift boosts, wheel sets speed |
| double-click | recentre |
| `Tab` | hide the panels |
| `Space` | pause |
| backtick / tilde | switch horizon / free flight (Q/E roll, R/F up/down in free mode) |

The scene inspector uses `egui-widgets-rs`: `egui-attr-table` for typed controls and
reset/copy/paste actions, `egui-attr-grid` for vectors, and `egui-titlebar` for sections.

**Settings → Display / Color** uses the same `egui-prefs2` layout as exr-view. Display uses
`egui-display::settings_ui` directly: output, SDR reference white and HLG display peak, with
OS values used automatically unless overridden. Color is the copied exr-view OCIO panel:
config (built-in, `$OCIO`, `.ocio`/`.ocioz` file), Input, Display, View, Look, Reload and monitor
presets. The default input is **Linear Rec.709 (sRGB)**, matching this tracer's material,
palette and lighting values; it must not be interpreted as ACEScg.

For HDR on screen, enable HDR in the OS, choose an available HDR output in Display, then
select **HDR · 1000 nits** in Color. The ACES rendering peak and UI reference white are
independent. A saved HDR scene on an SDR output gets a separate SDR ACES preview.
PQ/HDR render targets remain selectable in Color on SDR screens, including for HDR export;
only the actual window-output modes in Display depend on the connected display.

Screenshot / Render PNG saves the selected rendering: SDR sRGB/selected monitor codes as
8-bit PNG, or HDR10 BT.2020/PQ as 16-bit PNG with `cICP`, `mDCV`, and measured `cLLI` metadata.
**Display EXR** saves unquantized linear Rec.709 display light with chromaticities and
`whiteLuminance = 100`; this is display-referred light, not a scene-linear master.

```sh
./target/release/frac-rs --gallery out 1920 1080 256 --hdr --display-exr
CUDA_HOME=/usr/local/cuda CUDA_OXIDE_LLC=/usr/bin/llc-22 cargo oxide test -- --release
```

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

## Performance

These numbers are from an RTX 3080 Ti, 1280×720, 6 bounces, at the preset views (`--bench`;
the full table is in [docs/bench-1280x720.txt](docs/bench-1280x720.txt)):

| preset | ms / spp | Msamples/s |
|---|---:|---:|
| KIFS (tetrahedron) | 5.2 | 177 |
| Octahedron KIFS | 8.6 | 107 |
| Quaternion Julia | 12.6 | 73 |
| Apollonian | 12.2 | 76 |
| Kleinian | 13.3 | 69 |
| Mandelbulb | 35.4 | 26 |
| Menger sponge, Standard Surface | 20.0 | 46 |
| Mandelbulb power 12, gold, Standard Surface | 69.2 | 13 |

**What makes it fast:**

- **One kernel per (family, material model).** Both are const generics, so each of the 16
  kernels contains only its own estimate and its own BSDF. There is no runtime dispatch, and the
  kernels need fewer registers. All device functions are force-inlined, so there are no ABI
  spills.
- **Parameters in `#[constant]` memory.** Every lane reads the same slot, so the value is
  broadcast to the whole warp.
- **Bounding-sphere ray clipping.** Rays march only inside the escape radius or ball bound. This
  is exact: outside it the estimate is "outside" anyway. Without it, every camera and escaping
  bounce ray spent about 40 capped steps crossing empty space.
- **A trig-free power-8 Mandelbulb.** Three angle doublings replace `acos`/`atan2`/`pow`.
- **Step factor 0.85** (ofx-fractal uses 0.5). On the presets it measured unbiased (mean
  luminance unchanged) and is 1.5× faster. A slider brings back 0.5.
- **8×4 pixel tiles per warp,** so neighbouring rays share their march paths.
- **Float colour pipeline.** CUDA resolves accumulation, exposure and saturation to linear
  Rec.709 RGBA32F. The shared `vfx-ocio` GPU runtime applies the complete ACES 2.0 output
  transform from oiio-rs; no Narkowicz approximation. Display changes preserve accumulated samples.
- **Shared SDR/HDR presentation.** `egui-display` renders an extended-sRGB float canvas to a
  supported SDR 8/10-bit, HDR10/PQ, HLG or scRGB surface. HDR output negotiates both the pixel
  format and colour space; unavailable modes are disabled in Settings.

Under WSL2 CUDA↔graphics interop is unavailable (WSLg is Mesa d3d12). The current bridge
reads back the CUDA float resolve, uploads it to the shared wgpu device for OCIO, reads back
its display light/codes, then uploads the viewport. OCIO pipelines are cached, but these
transfers cost more than the former RGBA8-only path. The timings above predate this change.

## Layout

```text
src/gpu.rs        the kernels (#[cuda_module]): estimates, march, normals, lighting, integrator, tonemap
src/scene.rs      Scene (serde), formulas, presets, gallery, packing into the parameter block
src/params.rs     parameter-block slots shared by host and device
src/render.rs     CUDA context, progressive targets, kernel dispatch, PNG output
src/app.rs        the egui browser + shared Settings panel
src/inspector.rs  scene controls from egui-widgets-rs attribute editors
src/color.rs      cached vfx-ocio GPU colour processing (linear Rec.709 input)
src/ocio.rs       OCIO controls adapted directly from exr-view (BSD-3-Clause)
src/window.rs     winit/wgpu shell + egui-display float canvas and SDR/HDR swapchain
src/palette.rs    the 14 palettes
src/materials.rs  the usd-rs material library and its Standard Surface translation
vendor/standard-surface-bsdf   Autodesk Standard Surface (MaterialX port, Apache-2.0, see its NOTICE)
vendor/cam-controls, vendor/cam-viewport   camera rigs from gitnexus-rs (PolyForm-Noncommercial-1.0.0)
```

## Licences

`vendor/standard-surface-bsdf` is a derivative of MaterialX (Apache-2.0); see its `LICENSE` and
`NOTICE`. `vendor/cam-controls` and `vendor/cam-viewport` come from gitnexus-rs and are under
PolyForm-Noncommercial-1.0.0. The fractal formulas credit their sources in the ofx-rs code they were ported from
(Knighty's KIFS, Leys' Kleinian, Mandelbulber's pseudo-Kleinian, and others).

The OCIO panel in `src/ocio.rs` is adapted from exr-view; its BSD-3-Clause notice is in
`vendor/EXR-VIEW-LICENSE`. The ACES 2.0 presets follow the [Academy output transform parameters](https://docs.acescentral.com/system-components/output-transforms/parameters/).
