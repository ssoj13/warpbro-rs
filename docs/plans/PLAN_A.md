# PLAN1 — render in ACEScg (AP1)

> Living plan. Status markers: `[ ]` todo, `[~]` in progress, `[x]` done. Started 2026-10-04.
> Revised 2026-10-04 after a code audit (table corrected, missing call sites and the `Sel.input` problem added),
> then after an independent review (Fable) — see "Review corrections". Implementation started the same day.

## Review corrections (verified in code, 2026-10-04)
- `render_service.rs:57`, `preview.rs:570`, `export.rs:928` consume `target.light`, which is **OCIO display
  light** when OCIO is on (`render.rs:879-887`); raw tracer RGB reaches them only via the `!sel.on || reinhard`
  branch of `ColorPipeline::apply`. So the ONLY AP1->Rec.709 point is that branch; per-site conversion would
  double-convert. `ocio.rs:413` runs after `proc.apply_rgba` (display light, test-only path): unchanged.
  `render_service.rs:1793`, `export.rs:1332` are tests.
- The CUDA kernel (cuda-oxide) cannot reach `color.rs` (it links vfx-ocio). The device-safe luma source is
  `standard_surface_bsdf::consts::SS_LUMA` (MaterialX/ACEScg, already used by the BSDF lobe selection, i.e.
  the Full model mixed AP1 and Rec.709 luma before this change). `color::LUMA` re-exports it; a test pins it
  to the AP1 Y row from vfx-ocio.
- `pt-denoise-oidn` (render-rs) has **no `acescg` feature**: it is AP1-only, and its API changed
  (explicit device handles instead of `render_core::gpu::GpuContext`), so step 7 is a port, not a flag.
- `fractal-materials` emissive is documented as ACEScg energy: the preset is stored as `to_709(emissive)`
  so upload restores the exact AP1 value.
- Matrix facts: Rec.709->AP1 (Bradford) has all-positive entries and unit row sums, so [0,1] albedo stays
  in [0,1] (no gain, no negatives; Beer-Lambert `powf` stays finite). Per-channel products (palette x tint,
  Beer-Lambert) do not commute with the matrix: an intended convention change.
- `ocio::Sel` is used only for the tracer image: `input` deleted outright. No `deny_unknown_fields`
  anywhere, so old JSON with `"input"` loads.

## Goal
The tracer works in **linear ACEScg (AP1, D60)** instead of linear Rec.709. Every colour that enters the
renderer is converted to AP1 once, at its single entry point; everything that leaves it says it is AP1.
OCIO (ACES 2.0 views) stays the only display path.

## Why
- Wide-gamut GI: multi-bounce products of saturated colours stay in gamut; Rec.709 rendering produces
  hue shifts and clipped/negative channels on saturated bounces. AP1 is the production rendering space.
- The display side is already ACES 2.0 via OCIO; today OCIO converts `Linear Rec.709 (sRGB)` → ACES at the
  end, i.e. the gamut is lost before rendering, not after.
- The shared OIDN glue (`pt-denoise-oidn` in render-rs) is AP1-only: its firefly clamp and oidn-rs
  autoexposure (`acescg-autoexposure`) weigh luminance in ACEScg.

## State before the change (verified in code, 2026-10-04; historical, line numbers as of `1b0b440`)
| Where | What | Before |
|---|---|---|
| `src/color.rs:9` | `INPUT` (OCIO input colour space) | `"Linear Rec.709 (sRGB)"`; module doc: "The tracer's RGB is linear Rec.709, NOT ACEScg" |
| `src/color.rs:32` `default_selection` | default `ocio::Sel` with `input: INPUT` | Rec.709 |
| `src/ocio.rs:52` `Sel.input` | input colour space, **user-selectable** (combo `ocio.input`, `ocio.rs:771`), resolved at `ocio.rs:166`, fed to OCIO at `:216` | persisted in `Scene.colour` (`scene.rs:458`, i.e. saved scenes / bookmarks) and app `Settings.colour` (`app.rs:286`) |
| `src/color.rs:48` `ColorPipeline::apply` (`:54`) | OCIO off / Reinhard fallback | raw tracer RGB → `oetf` (assumes Rec.709 primaries) |
| `src/ocio.rs:413` | built-in (non-OCIO) display path: `transfer::oetf(transfer::dial(c * gain, gamma))` | assumes Rec.709 primaries |
| `src/render_service.rs:57`, `:1793`; `src/preview.rs:570` | direct `oetf(pixel * gain)` display paths | assume Rec.709 primaries |
| `src/export.rs:928`, `:1332` | 8-bit / video export: `oetf(p).clamp(0,1)` | assume Rec.709 primaries |
| `src/gpu.rs:78` `luminance` | kernel luminance (Rec.709 weights `0.2126/0.7152/0.0722`); used by light selection (`:1266–1301`), fast specular probability (`:1494–1495`, `path_sampling::fast_spec_probability`) and the tonemap saturation (`:2973`) | Rec.709 |
| `src/render.rs:859` | CPU saturation after denoise (`l + (c - l) * P_SATURATION`) | Rec.709 weights, duplicate of `gpu.rs` saturation |
| `src/render.rs:2128` | test of the above saturation | Rec.709 weights |
| `src/environment.rs:79` | environment importance-sampling CDF weights **and** `mean_luminance` (→ `P_ENV_MEAN`, `render.rs:765`) | Rec.709 weights |
| `src/palette.rs` `build_lut` / `PaletteScheme` | fractal palette LUTs (escape / trap colouring) | authored as Rec.709 values |
| `src/materials.rs` presets (`diffuse`, `emissive`) → `Material::{base_color, emission_color}` | material colours | Rec.709 values |
| `src/scene.rs` `Lighting::{sky_horizon, sky_zenith, sun_color}`, `src/presets.rs` | procedural sky / sun | Rec.709 values |
| `src/environment.rs` `Map::load` → `exr_io::read_rgb` | lat-long EXR / HDR environments | read as-is (no chromaticities honoured) |
| `src/exr_io.rs` `write_rgb` (`:59`, test `:128`) | EXR export | tagged BT.709/D65 `chromaticities` |
| `src/denoise.rs` → `pt-denoise-oidn` (squarebob copy today) | OIDN | Rec.709 weights |
| Metadata / hints: `export.rs:194` ("Scene-linear Rec.709"), `export.rs:900` (YUV comment), `render_bench.rs:202` (`"format"` field), `render_service.rs:39`, `render.rs:430`, `:1231`, `exr_io.rs:6`, `gpu.rs:2953` | text describing the working space | say Rec.709 |
| Test `render.rs:2541` `old_bookmarks_default_to_linear_rec709_aces2` | missing `colour` falls back to `default_selection()` | name and meaning tied to Rec.709 |

**Not in scope (stays Rec.709 by design):** `ocio.rs` `to_display_light` (`:257–308`, `:351`) — the
display-light output (HDR export, tile shader) is *display* linear Rec.709 / XYZ→Rec.709 after the view;
it is downstream of OCIO and independent of the working space. `export.rs:900` YUV encoding is a display
encoding too; only its comment changes if it mentions the working space.

## Design
1. **One colour-space module** (`src/color.rs`): `WORKING = "ACEScg"`, the Rec.709→AP1 and AP1→Rec.709
   3x3 matrices (Bradford D65→D60, built from `vfx_ocio::color_matrix`, which `color.rs`/`ocio.rs`
   already import — not hand-typed), AP1 luminance weights (the Y row of AP1→XYZ:
   `0.2722287, 0.6740818, 0.0536895`), `to_working([f32;3])`, `to_709([f32;3])` and `luma([f32;3])`.
   No other file contains a colour matrix or luminance weights. The CUDA kernel cannot link `color.rs`
   (it pulls vfx-ocio), so it reads the same weights from `standard_surface_bsdf::consts::SS_LUMA`, which
   `color::LUMA` re-exports; a test pins both to the AP1 Y row.
2. **Authoring stays Rec.709/sRGB; conversion happens at upload.** UI colour pickers, presets, palettes and
   saved scenes keep their current (Rec.709) values — they are user-facing. `scene.rs` parameter packing
   (`put3` for colours) and `palette::build_lut` convert to AP1 when filling GPU buffers. This keeps every
   existing scene and preset valid without a migration.
3. **The tracer's input space is a renderer fact, not a user setting.** `Sel.input` is removed from the
   tracer's selection: OCIO always receives `color::WORKING` as the source. The `ocio.input` combo is not
   shown for the tracer image (it stays available only where `ocio::Sel` is used for foreign images, if
   anywhere — check during step 5). Saved scenes / settings that still contain `"input"` are ignored on
   load (serde skips the unknown field); no compatibility shim. Rationale: if the input remained a
   persisted field, every old scene/bookmark would keep feeding `"Linear Rec.709 (sRGB)"` to OCIO while
   the pixels are AP1 — a silent, per-file colour shift.
4. **Images carry their space.** `exr_io::read_rgb` returns the file's `chromaticities` (default BT.709 when
   absent, per OpenEXR); `environment::Map::load` converts to AP1 with the matching matrix (built from the
   file's primaries/white via `vfx_ocio::color_matrix`, not limited to two hard-coded cases). `.hdr`
   (Radiance) is Rec.709. `exr_io::write_rgb` tags AP1 chromaticities (ACES AP1, white D60) for
   scene-linear output; the display-light export keeps its display tagging.
5. **Luminance uses the working space**: `gpu.rs luminance` (light selection, specular probability,
   saturation), `environment.rs` CDF + `mean_luminance`, and the CPU saturation in `render.rs` call the one
   AP1 `color::luma`. The CPU saturation and the GPU saturation must stay one formula (currently
   duplicated; keep them sharing the weights at minimum).
6. **Display**: OCIO source = `color::WORKING`. The only place raw tracer RGB meets `oetf` is the OCIO-off /
   Reinhard branch of `ColorPipeline::apply`: it converts with `color::to_709`, then Reinhard, then `oetf`.
   Every other `oetf` site (`render_service`, `preview`, `export` HEVC) consumes `target.light`, which is
   already display light — converting there would double-convert.
7. **Denoise**: `pt-denoise-oidn` from render-rs `main` (AP1-only; no feature flag exists).

## Expected behaviour changes (by design, record in step 9)
- Saturation ≠ 1 changes slightly (AP1 luma pivots on different weights).
- Light selection / specular probabilities change → different noise pattern, same expectation.
  Golden comparisons are statistical (mean/variance within noise), never bit-exact.
- Saturated materials / palettes under GI change hue/brightness (that is the point).
- Neutral (grey) scenes under a white sky render the same within noise: Rec.709 white (D65) maps to AP1
  white (D60) via Bradford, and both luminance vectors sum to 1.

## Steps
- [x] 0. Prerequisite: `pt-denoise-oidn` unified in render-rs (with the `acescg` feature); warpbro moved off
       the squarebob copy. **Blocks only step 7**; steps 1–6 can proceed in parallel.
- [x] 1. `color.rs`: `WORKING`, matrices from `vfx_ocio::color_matrix`, AP1 luma, `to_working`,
       `to_display_709`; unit tests (matrix round trip, white maps to white, `luma(white) == 1`, AP1 matrix
       matches the OCIO studio config's `ACEScg` ↔ `Linear Rec.709 (sRGB)` processor within 1e-5).
- [x] 2. Upload-time conversion: scene parameter packing (material base/emission/tints, sky horizon/zenith,
       sun), palette LUTs (`build_lut` output), interior colour. Grep gate: no raw Rec.709 colour reaches a
       GPU buffer except through `to_working`.
- [x] 3. Environments: `read_rgb` reports chromaticities; `Map::load` converts; CDF + `mean_luminance` use
       `color::luma`; tests with a BT.709-tagged and an AP1-tagged EXR (same light, same result).
- [x] 4. Luminance: `gpu.rs luminance` (all uses) and `render.rs:859` CPU saturation switch to the AP1
       weights from `color.rs`; update the `render.rs:2128` test.
- [x] 5. Display / OCIO input: remove `Sel.input` from the tracer path (design 3); OCIO source =
       `WORKING`; hide the `ocio.input` combo for the tracer; one `to_display_709` in front of every non-OCIO
       `oetf` path (`color.rs:54`, `ocio.rs:413`, `render_service.rs:57`, `:1793`, `preview.rs:570`,
       `export.rs:928`, `:1332`). Rename/replace test `old_bookmarks_default_to_linear_rec709_aces2`
       (a scene JSON with a stale `"input"` loads and renders in ACEScg).
- [x] 6. EXR export tags AP1 for scene-linear output; read-back test checks the chromaticities attribute
       (update `exr_io.rs:128`). Display-light export tagging unchanged.
- [x] 7. Denoise: `pt-denoise-oidn` from render-rs (AP1-only).
- [ ] 8. Golden check: render a fixed scene before/after; neutral (grey) materials under a white sky must
       match within noise; saturated scenes differ by design. Record the numbers here.
- [~] 9. Docs and metadata: module docs (`color.rs`, `exr_io.rs`, `gpu.rs:2953`), comments
       (`render_service.rs:39`, `render.rs:430`, `:1231`), UI hint `export.rs:194`, `render_bench.rs:202`
       `"format"`, README (`:240`, `:368`), CLAUDE.md, CHANGELOG — describe the working space, the authoring
       convention and the behaviour changes above.

## Open questions
- Should the UI offer an "author in ACEScg" switch for colour pickers (pick directly in AP1)? Default plan:
  no — author in Rec.709/sRGB, render in AP1.
- Palettes: keep their Rec.709 definitions (converted at upload) or re-author in AP1 for wider colours?
  Default plan: convert at upload; re-authoring is a separate creative change.
- ~~Is `ocio::Sel` used for anything besides the tracer image?~~ No: `input` deleted outright.

## Progress log
- 2026-10-04: steps 1-6 implemented. `color.rs`: `WORKING`, `WORKING_PRIMS`/`DISPLAY_PRIMS`, `LUMA`
  (= `SS_LUMA`), `working_from(&Primaries)`, `mul`, `to_working`, `to_709`, `luma`. `Scene::pack` routes the
  11 RGB slots through `put_rgb`; `build_lut` converts samples + interior. `exr_io::read_rgb` returns the
  file's primaries, `write_rgb` takes them. `Sel.input` / `Ocio::inputs` / the Input combo removed.
  Step 5 conversion point: `ColorPipeline::apply` non-OCIO branch only (to_709 -> Reinhard -> oetf).
- 2026-10-04: second Fable review of the diff: no product bugs. Applied: typed error for a
  `chromaticities` attribute of the wrong type (no silent BT.709), the two `#[ignore]` OIDN tests compare
  OCIO-off display light as `to_709(...)`, stale "log input" note in `ocio.rs` doc.
- Tests updated to compare in the right primaries: `render.rs` glass (hue in Rec.709, magnitude vs
  `to_working`), primary albedo (kernel formula over `to_working` inputs), two-formula world and directional
  lights (`rec709()` helper), packed transmission colour, environment load (BT.709-tagged, AP1-tagged and
  .hdr give the same AP1 texels), EXR tags, bookmark with a stale `"input"`.
- Verification: `cargo oxide test -- --release`: 214 passed, 8 ignored; the two OIDN `--ignored` tests pass.
  HEVC tests (`hevc_motion_fixture…`, `cancel_controller…`) fail intermittently with
  `Vulkan Video unavailable: Posix(38)` in full parallel runs — reproduced on clean HEAD (1 of 3 runs), pass
  in isolation on both trees: pre-existing, unrelated.
- 2026-10-04: step 7 done. `pt-denoise-oidn` now comes from render-rs `main` (AP1 firefly clamp, oidn-rs
  `acescg-autoexposure`); squarebob `pt-denoise-oidn` / `render-core` and the direct `oidn-rs` dep are gone.
  `denoise.rs`: `GpuContext` from `pt_denoise_oidn` (no `gpu_info`). Same API otherwise.
- 2026-10-04: every own Git dependency tracks `branch = "main"` (latest revisions). Three old pins lived only
  on unmerged side branches and were brought to `main`: render-rs `cuda-math` (`d46df73`), gitnexus-rs
  `inertial-look` (`a06c4d0`), egui-file-dialog host-persistence API (fast-forward to `8659643`). `vendor/` removed.
  (Later the same day: `[patch]` removed — own repos are SSH, third-party git sources stay HTTPS.)
- Verification: 215 passed, 8 ignored; the 3 OIDN `--ignored` tests pass; `--features ofx-direct` builds.
  Only warning left: `presets.rs` `AnimatedPreset::description` (hover text lost in 7bc48c2; decision pending).
- Open: step 8 (golden numbers), step 9 CLAUDE.md note + CHANGELOG entry for the later features
  (PNG export, `~/.warpbro` profile, built-in templates, camera slots, material assign menus).

## TODO found on the way (out of scope)
- [x] `P_SHADOW_*` / `P_AO_*` / `P_LIGHT_HALF_ANGLE`: OFX Direct ABI slots read only by `ofx-direct` kernels;
      `slots!` now takes attributes, names exist under the feature, offsets pinned by a const assert.
- [x] `presets.rs` `description`: restored as the Templates menu hover text (built-in template catalog).
- [~] HEVC export tests contend for the hardware Vulkan Video session in parallel runs (`Posix(38)`):
      the tests that may open a session now share `vulkan_video()` lock; not yet confirmed in a full run.
- [x] `vendor/` (dead copies + exr-view licence) removed.
- [ ] gitnexus index (re-analyzed as `warpbro-rs`) is stale after the later commits: `reanalyze`.
