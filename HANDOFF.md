# HANDOFF (2026-10-06, end of the curves / widgets / render-settings wave)

Cross-repo state for the next session. Repos live in `C:\projects\projects.rust.cg\cglibs`.
Per-repo detail: WarpBro `PLAN.md` ("After the 2026-10-06 merge" section), Playa `todo.md`
("Open after the 2026-10-06 merge"), each repo's `CHANGELOG.md`.

## Current continuation (2026-10-06, systematic update in progress)

### Latest checkpoint — 2026-10-07

This checkpoint supersedes the runtime/publication statuses in the 2026-10-06 table below;
that table and the previous shift's entries remain history. The whole scope remains authorized.

| Work | Verified receipt | Still open |
|---|---|---|
| GitNexus | Published `ec11c6a2ad54eee9b27f477c531b6b9625ec18aa`; six Git + 87 MCP tests passed, native GUI + CLI build passed in 42m38s, two actual help calls exited 0, post-commit reindex passed | Foreign installed MCP was not replaced; these receipts do not refresh other repositories' graphs |
| Shared toolbar / output policy | Solid toolbar and mandatory ToolbarState persistence source; 15 ordinary tests passed on Rust 1.96 after the natural 44px-button/font fixture correction. Pure OutputKind/view resolver extracts actual WarpBro policy in egui-display::export. Sole producer review closed with no findings | Producer is not published. Six-test policy run `21437` stopped on disk-full regex-syntax IO before tests, not a source/test failure. No files were deleted; about 45 GB subsequently became free. Current coherent-8c4-lock run `82388` is active |
| SquareBob UI | Foundation: 24 UI tests passed; one GPU gate produced seven screenshots | Visual inspection found narrow overlap/floating toolbar and fade-fixture issues. Fixes are source-ready, but regenerated shots are not accepted. Five camera slots and named preset buttons are not full WarpBro toolbar parity |
| SquareBob export freeze | FrozenRenderSession source complete across nine files plus ColorPipeline; it owns frozen DirEntry/camera/quality/options, its ColorPipeline and existing 3D/2D renderer resources moved from author fields | Eight ordinary + one ignored GPU tests are source-only; no compile/test/reindex receipt yet, awaiting published shared API. No authored time/flag mutation or zero-copy encoder claim. Full EXR metadata and remaining OCIO/proxy/denoise/snapshot/toolbar features remain open |
| WarpBro | Default-schema run: 256 passed, one failed, nine ignored. The remaining lens fixture was then corrected; four targeted lens tests passed. Actual OIDN HDR/all-quality, PNG-video HDR/SDR tags+CLL and Vulkan partial `24000/1001` each passed one gate | Specialized Fast one-object radiance still fails: max_abs `5.9247017e-5`. Fast two-object and StandardSurface one/two-object radiance/albedo/normals are exact. Owner traces the shared field boundary; no tolerance/kernel change yet. New shared-view policy and cache-key migration are pending; current key omits current input/look |
| App queue | SquareBob floating Git dependencies audited against actual remotes; explicit fscan pin moved `5f57` → `94dd173` | COLMAP native `75655` is active on 8c4; B fixture is 7406 bytes / 48 frames / stride 3 / expected 16, not yet run. EXV/RV wait for the new widgets SHA |

FrozenRenderSession preserves actual GPU causes through worker errors and a final readback
drain; its 3D SDR path uses RGBA16F float rather than pre-gamma 8-bit data. Configuration/LUT
bake failures are strict. These are current source contracts, not passed consumer gates.

Retained video evidence is under `C:/Temp/bob`:
- `delivery-20261007-0615`: twelve technical three-frame 64x64 clips.
- `warpbro-8c4-a9623942eb974620a00de18fd1bbc4e2`: Kvazaar/Vulkan motion, 51 frames,
  2.125s, 256x256; fractional clip, 32 frames, 66x50.
- `warpbro-schema-76890017f5f64347b7b90b6cd7d03291`: schema-run artifacts.
- `warpbro-explicit-aeba34931e6e40a39c3ab6bbb291d0b3`: actual HDR fixtures, two frames,
  16x16; Vulkan partial clip, nine frames, 256x256.
- `ui-20261007`: older camera/preset/wide screenshots; narrow UI acceptance is still open.

Graph evidence is scoped: SquareBob owned mirror full-force `54164` passed (5861 nodes,
13403 relationships), then serialized `94939` and five-file reindex passed. The canonical
SquareBob DB is still foreign-locked and is not fresh. Shared CLI ec11 one-file reindex passed
before the new policy helper; another reindex is needed. WarpBro's isolated registry graph has
3556 nodes / 9918 relationships; detect reported 503 symbols / 42 files, CRITICAL combined
expected scope, and passed inspection. No original/global freshness is claimed.

The operator authorized the whole remaining handoff: update dependencies, verify behavior,
fix shared contracts rather than adding compatibility shims, and compare with references in
`D:\Projects\vfx.ref` where available. This supersedes the previous hold on the ffmpeg update.
The merge table and OPEN list below describe the previous shift; this ledger records the
continuation. A published dependency or a passing focused test does not certify an application's
full build or native UI.

| Work | Current state | Evidence / next gate |
|---|---|---|
| Shared widgets | Main `3443b6f46a2a72526c9542b1e7d8e528c88142bc` published; foundation includes `9d459d2` | Attr-grid foundation gate at `9d459d2`: 50 passed, one ignored. Earlier config/ramp gates and independent screenshot review passed. Latest export evidence is recorded below; full consumer/native AE acceptance remains open. |
| Shared export SSOT | Published `3443b6f46a2a72526c9542b1e7d8e528c88142bc` in existing egui-display | Recorded source gates: 31 tests passed, including measured cLLI, real FFmpeg/ffprobe checks of six SDR/PQ/HLG HEVC+ProRes combinations and cancellation; strict release all-target clippy `-D warnings` passed. Extracted actual WarpBro PNG/HDR/FFmpeg/view-peak/BT.1886 primitives preserve alpha, atomic publication and true 16-bit PNG. Consumer compilation/acceptance remains open. |
| Audio dependency | Published on main, `76b66b0` in audio-rs | Removed obsolete av-decode SIMD feature request; audio-decode checked against ffmpeg-rs main `a6676574`. This does not certify Playa video decode. |
| FFmpeg color producer | Canonical metadata checkpoint `8c4` published | Canonical `bt709_limited` and checked nclx passed six release tests. SquareBob/WarpBro consume this metadata; those producer receipts do not certify the newest application launch/frame/UI changes. |
| ofx-host-egui | Main `9377dce` published; production unchanged from `402e90a` | New commit changes only fixtures, lock and changelog. Host-egui passed 58 tests on final widgets `9d459d2`: 52 unit + four instance + one param_interacts + one doctest. All-numeric descriptor-span audit and final 23 GPU screenshots passed and were visually inspected. Backend-hint overlap, RGB R/G/B labels, and status wrapping are verified. Consumer/native application acceptance remains open. |
| Playa foundation | Published `8fa38c4d0483945613fc1ca5b6428344b7225df9`; final foundation native/build gates passed | Native IO plus 347 engine/entities, 82 app, 125 UI and nine cache tests passed; native binary/build gates passed. Earlier seven GPU tests remain outside the claimed executed scope. Measured sysinfo memory-query cost improved from 106s to 0.56ms. Output/Queue backlog and native UI acceptance remain open; PQ-tag evidence alone does not certify 10-bit encoding precision. |
| USD / Alembic dependencies | USD published `eb92860f9c55281c67bcee6985dce622f649f81e`; latest Alembic `7538a727` | USD fallible compound creation and all seven sample admissions are included. Four native gates passed; clippy reported zero errors. WarpBro's coherent lock is updated, but its own source/build validation remains open. |
| WarpBro | Native CFR baseline and updated all-target gate passed; ordinary oxide active | Earlier all-target `31284` passed in 12.73s. Actual oxide native CFR baseline `74654` passed one test (52m45s reported). Updated canonical `8c4` CFR/coded-metadata all-target `57365` passed in 2m55s with zero warnings. Full ordinary oxide session `71257` is active, not passed. Remaining behavior/build, old-data/lens/recorder and native acceptance remain open. |
| Watermark | Published `c4040f0`; release and icon gates passed | Latest release passed in 9m59s and three icon tests passed. Native UI acceptance remains open. |
| Colmap | av_graph video port and native gates active | Application owner is implementing/verifying the video port and native gates. Earlier shared-icon migration and clippy receipts do not certify the current port; no completed-app claim. |
| GitNexus | Integrity tests and focused review passed; native build active | Supported-SDK rename `39631` passed five tests. Latest integrity gates passed six Git + 87 MCP tests; sole focused review closed with no findings. Default native GUI + CLI session `39070` is still building. Canonical SquareBob MCP graph remains locked. Concurrent writes on the owned SHA-verified mirror encountered a duplicate File primary key; cause is not proven. Serialized full analysis `54164` is active; no completed fresh-index receipt is claimed for that run. Foreign processes/caches remain untouched. |
| Squarebob | Native file/source gates passed; newest viewport UI unverified | Production `73811` passed on Rust 1.96 (24m49s bootstrap / 24m34s Cargo). Earlier library/native gates passed 21 + four tests. Latest four native gates passed in 2.51s: alpha, six SDR variants, four PQ/HLG HEVC/ProRes variants, unsupported settings. Twelve retained files include two ProRes-alpha clips. Source-error/extent transaction gate `98764` passed three tests, with three ignored (3m20s). New viewport toolbar, five camera slots and Settings integration remain source-only: app `1177` is compiling, UI tests/GPU screenshots are not passed. Full renderer freeze, independent SDR still-view and remaining WarpBro OCIO/proxy/denoise/snapshot parity remain open. |
| EXIF producer | Canonical source identity published `77c56ae`, following `6960603` | Canonical EXR/JPG/JPH source identity retains the same SHAs. Consumer dependency/native acceptance remains separate. |
| RV / EXV | RV full dependency update complete; native gate pending | RV dependency checkpoint includes `8fa` / `73a` / `9d`; native binary is pending. EXV actual executable `--help` earlier passed in 3m57s after the library-only pass; own view tests and native UI acceptance remain open. |
| OTIO | Main `73a074f` published; final demo release and tests passed | Final demo release gate passed in 29.24s and 21 tests passed. No final-lock refresh remains active at this checkpoint; native demo-window acceptance remains open. |

Native UI gate limitation: Computer Use initialization succeeded, but two `sky.list_apps()`
calls failed because the native pipe is unavailable (Windows os error 2). The exact reproduction
and permitted fallback are recorded in oh-my-harness `BUG3.md` (2026-10-06). Use existing
egui_kittest GPU screenshots and supported release builds; keep native acceptance open.
No custom helper or PowerShell UI automation was used.

Latest requested UI/video work: match WarpBro's viewport toolbar and camera slots, and retain
all video artifacts under `C:/Temp/bob`. Verified SquareBob outputs are in
`C:/Temp/bob/delivery-20261007-0615`: six SDR, four PQ/HLG HEVC/ProRes and two ProRes-alpha
files. These are three-frame technical clips, not real application-motion recordings.
New UI source has five nullable persisted full OrbitCamera/PhysicalCamera/DoF slots:
primary click recalls, secondary click copies, and an empty slot is a no-op. It uses the shared
egui-viewport-toolbar Top geometry, 2D/PBR/PT selection, physical EV, spp, Settings gear via
OpenSettingsEvent/focus, primary SDR PNG through the existing capture handler and context
Export settings. New mandatory persistence fields have no old-JSON fallback. Named preset
and new camera/toolbar UI acceptance remain open; source presence is not a screenshot gate.
WarpBro oxide `71257` is still building. Its new videos are retained in the unique owned
`C:/Temp/bob/warpbro-8c4-a9623942eb974620a00de18fd1bbc4e2` destination, routed before
writes through a previously nonexistent TEMP junction; no completed new-video gate is claimed.

Render-profile proposal: [docs/render-profiles.md](docs/render-profiles.md) is a Russian
DRAFT for review. It specifies canonical settings nodes, profile/template semantics,
Viewport routing, WorldDirect requirements, and independent Output configuration.
The implementation-start gate has not passed; the existing backlog below remains open.

Current explicit priority: complete SquareBob's full Encoder/HDR/Display parity by reusing
actual WarpBro export/color/display code through one SSOT in the existing egui-display
export module, with thin WarpBro/SquareBob source adapters. The user explicitly requested
deduplication, including selected-view peak measurement and canonical BT.1886 math.
The user rejected the proposed common DisplayColorPass, shared Playa NativeEncoder migration
and new encoding/interop architecture; those plans are superseded and no longer active.
No new raw GPU target or GPU interop. Shared export is published at `3443b6f46a2a72526c9542b1e7d8e528c88142bc`;
SquareBob production/metadata library receipts exclude newest launch/frame/UI source;
WarpBro updated all-target/native CFR passed, ordinary oxide remains active. Preserve full behavior, controls, precision, persistence,
formats/codecs and decoded-file acceptance. Revised directive and preserved audit:
[SquareBob HDR/encoder/display](../squarebob-rs/docs/hdr-encoder-display.md).
The existing WarpBro implementation is the reference; the render-node proposal above remains
a separate draft.

Ordered next steps:
1. Complete consumer checks of the published actual WarpBro extraction in egui-display:
   SquareBob adapters/gamma/files/UI-persistence and WarpBro all-target/CFR/native/oxide.
2. Keep published Playa foundation `8fa38c4d0483945613fc1ca5b6428344b7225df9`, widgets `9d459d2`, ofx `9377dce` and coherent
   native dependencies aligned; remaining integration/Output/Queue scope stays open.
3. Finish WarpBro scene/lens/recorder work, then check all targets and run oxide tests/build.
4. Inspect native Attribute Editors at narrow widths and 100/150% scale; compare Playa/ofx with
   WarpBro and the existing parity spec below. Logic tests do not replace these screenshots.
5. Complete application build prerequisites and icon migrations; only then remove the remaining
   curves compatibility surface after confirming all consumers compile.
6. Review the node/UI/preview proposal, then WorldDirect benchmarks and Playa Output/Queue
   adaptation. Continue Output Module templates and the rendering/export backlog from
   [PLAN.md](PLAN.md). These have not been completed by this wave.

Shared ownership: egui-widgets-rs owns row metrics, numeric measurement, icons, and curve-domain
geometry; ofx-host-egui owns the reusable effect-stack and parameter/curve presentation. Each
application keeps stable effect IDs, plugin discovery, backend choice policy, status data,
Undo transactions, and model mutation. `EffectStack` emits actions; its host applies them.
Keep application state out of shared widgets and do not recreate the same widget in Playa.

## Operator rules learned this session (also in memory)

- No compatibility with old data, ever: no legacy loaders, serde aliases/defaults for old formats,
  format markers, `migrate_*`, `legacy` modules. Convert checked-in files once with a throwaway script.
- Icons: phosphor through `egui_widgets_config::icons` only. Never Unicode glyphs or letters as icons.
  A removal migrates every consumer; a missing icon is added to the icons module, never the UI
  element dropped.
- Verification budget: ONE independent review per finished branch + my own look at screenshots.
  No review rounds per small fix (operator: "заебал проверками").
- Decide design details systemically myself; ask only real product forks. Push when told
  (branch + main, don't wait for tests). Never `git reset`. No Co-Authored-By.
- Clean only caches of repos this session works in and only when no cargo runs there; other
  sessions work in ffmpeg-rs, oiio-rs, exr-rs, h264-rs, agent-worktrees.
- WarpBro builds/tests only via cuda-oxide (`cargo oxide test -- --release -- <filter>`);
  type-check with `FRAC_ALLOW_PLAIN_CARGO=1 cargo check --all-targets`.

## Merged to main and pushed (2026-10-06)

| Repo | main | Content | Verified |
|---|---|---|---|
| curves-rs | 169ded7 | Track model (Tan per side, Bezier segments, extrapolation, fitter) | yes |
| egui-widgets-rs | 4139221 | one attribute editor, icon value actions, egui-ramp in grid style, value box width from span, segmented -> combo when it doesn't fit, AttrMetrics/segmented/icon_button in egui-widgets-config | tests + my screenshot check |
| ofx-rs | 91eccca | Parametric on Track (Linear point = segment it leaves), params panel on the attr grid, status icons, `EffectStack` | stack is WIP, unverified |
| render-rs | 73871a6 | no old-data shims, no compat aliases | agent tests |
| vfx-view | b13cd72 | no old-data shims/aliases, lock aligned (oiio-rs + exr-rs together) | cargo check clean |
| playa | 149e03c | RenderSettings/Tier + templates, progressive viewer, one background gate, Track keys, attribute schema (labels/spans/defaults/enums + guard test), icons, LNK1140 profile fix | WIP parts unverified, no full build after the last commits |
| warpbro-rs | 6f9b788 | keys on Track, camera orbit via integrate, Preferences on grid, Playa tier cache, reset defaults, no old scenes | last commit WIP, NOT built (target was cleaned) |
| watermark-rs / colmap-rs / gitnexus-rs / squarebob-rs | 7f05e81 / 19147c8 / d500651 / efe1448 | icons + attr-grid | WIP, unverified |

## OPEN: ffmpeg-rs `av-player/simd` (diagnosed, not fixed)

- ffmpeg-rs `b8636ea` ("codec-simd mandatory for prores/h26x") made SIMD mandatory and REMOVED the
  `simd` feature from `av-player`. Playa `crates/playa-io/Cargo.toml:49` still enables
  `"av-player/simd"` (comment above it, lines ~40-48), so Playa only resolves at the old
  av-player `97ddb21`; WarpBro's lock is pinned to that same rev to resolve.
- Fix (in Playa, ffmpeg-rs is fine): drop `"av-player/simd"` from the `av-player` feature and fix
  the comment; `cargo update -p av-player` in Playa (pulls the whole ffmpeg-rs set to current main;
  ffmpeg-rs is under active PLAN10 work by another session - expect API drift to fix); build,
  check video decode; then `cargo update -p av-player` in WarpBro to drop the pin.
- Operator has not yet said "go" for this.

## OPEN: build and finish (priority)

1. WarpBro main: full `FRAC_ALLOW_PLAIN_CARGO=1 cargo check --all-targets` + targeted oxide tests;
   finish the WIP commit (attribute labels without the redundant family prefix + label-width test;
   remaining old-data compat: `upgrade_material_schema`, orbit controls added to old documents,
   dock.rs layout upgrades, export.rs fields kept for old settings, `legacy_*` names,
   thumbnail_snapshot no-document path). Context-menu reset (fe032fa) needs a test.
2. Playa main: build + full test run with `ofx` (one combined run works now); migrate to
   egui-widgets 4139221 (`grid_config` free fn, `AttrMetrics`/`segmented` from egui-widgets-config,
   `AttrGridConfig::value_box_width` across AE sections); Attribute Editor screenshots vs WarpBro
   (spec: scratchpad `ae-parity-spec.md`, copied below); effect stack onto ofx-host-egui
   `EffectStack` (drop Playa's own Fx stack).
3. ofx-rs: finish/verify `EffectStack` slots (backend choice, app "+" menu, timing line);
   `ofx-host-egui/src/curves.rs` onto restyled egui-ramp (drop `width: Some(260.0)`, pass
   AttrMetrics, short curve labels R/G/B/A); `grid_config` free fn.
4. Apps (watermark-rs, colmap-rs, gitnexus-rs, squarebob-rs): build/clippy/icon scan test on the
   merged WIP; rv-rs, otio-rs, exv-rs icons not started. Consumer list: scratchpad
   `w1-consumers.md` (sections 1-8). gitnexus-rs has its own egui-widgets fork; squarebob-rs has
   its own copies of pt-mats (with the compat layers render-rs dropped) and color-pipeline.
5. After Playa + WarpBro build: remove `curves::legacy` / `Track::from_legacy` from curves-rs.
6. Camera recorder for WarpBro (own egui-prefs2 panel: TRS / Focus / Zoom / f-number checkboxes;
   "keep speed" extends the work area vs "fit to work area"; overrides existing keys; simplify with
   the curves fitter).
7. Playa render settings phases 3-5: Output Module templates (format, resolution, resize, crop),
   Viewer / Output nodes, Render Queue.
8. TODO: oiio-rs `from_legacy_matrix_filter` (two filter numberings for one thing; repo owned by
   another session); flaky WarpBro CUDA export test; vfx-view clippy pedantic backlog.

## AE parity spec (Playa / ofx panel must match WarpBro, reference scratchpad after_main.png)

1. Every numeric scalar: explicit slider span from ONE schema table (lin/log, soft vs hard), slider
   rail + value box. Guard test fails on a displayed numeric attr without a span.
2. Vectors: X/Y/Z cells + channel caret expanding to per-component slider rows.
3. Prefix gutter: key + reset icons left of the label.
4. Value actions: phosphor icons (done in egui-widgets).
5. Defaults on every attr (Reset everywhere). 6. Human labels. 7. Enums as dropdowns.
8. No full-width scalar boxes.

## Gotchas found

- Cargo.lock consistency: oiio-rs main needs the matching exr-rs main (exr-core `api::planar`);
  update them together (`cargo update -p vfx-io -p exr-core`), or align revs with Playa's lock.
- `cargo update` with an empty `-p` list updates EVERYTHING: build the list first.
- vfx-view pushes need `git -c lfs.locksverify=false push` (GitHub LFS locking API).
- Python `Path.write_text` keeps CRLF on Windows (diffs stay clean); bash `>` from `git show` writes LF.
- Scratchpad: `C:\Users\joss1\AppData\Local\Temp\claude\C--projects-projects-rust-cg-cglibs-warpbro-rs\3473b2bc-e01c-4760-80ab-c58e476664ca\scratchpad`
  (ae-parity-spec.md, w1-consumers.md, playa-cd-fixes.md, curves-review.md, screenshots).
