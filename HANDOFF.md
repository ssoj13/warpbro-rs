# HANDOFF (2026-10-06, end of the curves / widgets / render-settings wave)

Cross-repo state for the next session. Repos live in `C:\projects\projects.rust.cg\cglibs`.
Per-repo detail: WarpBro `PLAN.md` ("After the 2026-10-06 merge" section), Playa `todo.md`
("Open after the 2026-10-06 merge"), each repo's `CHANGELOG.md`.

## Current continuation (2026-10-06, systematic update in progress)

### Render/quality nodes checkpoint — 2026-10-07, published source

User priority is completing canonical render/quality profiles and templates. Compiler and
cross-repo work below is retained history; it is not a substitute for this deliverable.
Current source passed the final test/build gates below and was published as
[d3eccd229c6fad6d81d5ef70fb867cef5821a78a](https://github.com/ssoj13/warpbro-rs/commit/d3eccd229c6fad6d81d5ef70fb867cef5821a78a).
Exact origin/main was verified by git ls-remote at 17:02:33; this receipt covers the source
commit, not the subsequent documentation-only checkpoint commit.

- `WorldKind::RenderSettings / QualitySettings / ViewportSettings` are canonical graph
  nodes. Profile/Template is `metadata.catalog_role`; settings values are not duplicated in
  a persisted preset DTO. Local catalog entries save with WorldDocument.
- Settings → Render & Viewport has named buttons. LMB recalls to Moving/Still/Manual/Output; quality
  recall chooses a RenderSettings node. RMB opens Edit, Rename, Save as profile/template.
  Creation, independent copies and template instantiation use new UUIDs. A render template
  creates a render/quality pair and remaps its quality reference in one Undo operation.
- Existing Attribute Editor owns parameters, animation, resets and typed UUID choices.
  Toolbar and Settings bind the same ViewportSettings node. The quality UUID reference
  and all /viewport/* fields are static; commands/loading reject persisted animation and
  connections on them. Numeric render/quality parameters remain animatable at frame time.
  Auto chooses Moving while active and Still after settle delay; Locked uses Manual.
  Pause stops new samples; Freeze holds the previous request and pauses it, preserves the
  displayed image and ignores late frames.
- Progressive viewport uses one Target, sharing film for identical effective tracing inputs
  and extent. Cache preview freezes an explicit Still (Auto) or Manual (Locked) profile UUID
  with that quality's samples/scale; worker evaluates each frame using that chosen profile.
- Fast remains World path tracing with approximate opaque materials, preserving transmitting
  models. Full retains authored models. No WorldDirect or real-time performance acceptance.
  Moving defaults are editable Fast / 64 samples / 0.5 scale / two bounces.
- Output is an independent render reference. At this checkpoint the export freezes the
  document and fixes samples/resolution scale from Output Quality at the first job frame;
  other scene/render attributes evaluate per frame. OutputModule nodes/templates and
  external subgraph catalogs remain future work.
- Missing/invalid references, wrong kinds, invalid settings and deletion of referenced
  settings are rejected. No compatibility migration or default repair is introduced.
- Final ordinary cuda-oxide suite: **271 passed, zero failed, ten ignored, four filtered**,
  42.40s. Release compile took 5m28s including a Cargo build-directory lock wait.
  The four previously certified native-movie fixtures were intentionally filtered;
  no extra demo videos or renders were generated.
- Production `python bootstrap.py b` passed: release compile 4m45s, bootstrap build 4m49s;
  actual NVIDIA GeForce RTX 3080 Ti CUDA readiness took 75.41s. The installed
  `f3f1098a77` backend was unchanged; this deliverable makes no compiler-producer changes.
- Final plain locked/offline all-target check passed in 5.79s with no warnings.
  Owned rustfmt check and staged Git whitespace check passed.
- The sole independent branch review closed three P2 findings with source fixes and
  regressions: same-UUID reload cache invalidation, static typed-reference/viewport policy
  rejection, and rejection of a late Auto preview result after switching to Locked.
- GitNexus evidence covers the owned mirror `C:/Temp/warpbro-profile-audit-47dbf83e061c2`:
  12 current Rust files SHA-verified; incremental refresh at 16:49:42; detect_changes(all)
  reported 258 symbols / 14 files, HIGH, zero flows. Two main-name aliases in tools/mp4-quality
  and xtask were unchanged by actual Git/SHA checks; the actual 13 source paths include
  inspector deletion. The canonical/global graph was not repaired or certified.
- Direct pushes to main returned remote InternalServerError. Publication used the owned
  temporary branch `codex/render-quality-profiles-d3eccd2`, then a GitHub REST update of
  main with `force=false`; git ls-remote confirmed the exact source SHA above.
  Receipt: `run_command_1791392552766_6b35899f-2a34-4e48-9af0-5f45dd1a56c8_stdout.log`.
- Postcommit owned-mirror HEAD and index both match d3eccd2: is_stale=false and
  unstaged changes=false after incremental refresh at 16:59:19. This does not certify
  the canonical/global graph.
- WorldDirect, external catalog, OutputModule presets and the SquareBob node-profile
  port remain open.

Final logs under `C:/Users/joss1/.filesystem-mcp-rs/tmp`:

- tests: `run_command_1791391699187_43d47637-99ab-4e6b-a237-e7d9b6d4b6e5_stdout.log`
  and its `_stderr.log` companion;
- production build: `run_command_1791391667484_4fb51ea3-4985-4f3b-90c9-7f3a72c1db1f_stdout.log`
  and its `_stderr.log` companion;
- plain check: `run_command_1791392027817_66e7d804-69c4-4757-9481-c1ce3007fb36_stderr.log`.

Concrete behavior and remaining work: [docs/render-profiles.md](docs/render-profiles.md).
Implementation: world.rs, render_profiles.rs, render_profiles_ui.rs, app.rs,
render_service.rs and export.rs. Preserve unrelated edits and earlier receipts below.

### Latest checkpoint — 2026-10-07

This checkpoint supersedes the runtime/publication statuses in the 2026-10-06 table below;
that table and the previous shift's entries remain history. The whole scope remains authorized.

| Work | Verified receipt | Still open |
|---|---|---|
| GitNexus | Published coherent-d411 lock follow-up `0aa9f32`, following `ec11c6a2ad54eee9b27f477c531b6b9625ec18aa`; earlier six Git + 87 MCP tests, native GUI + CLI build (42m38s), two actual help calls and post-commit reindex passed | Foreign installed MCP was not replaced; no new native receipt is inferred from the lock publication |
| Shared toolbar / output policy | Published `d4117f50`: 15 toolbar tests, six output-policy tests and strict Rust 1.96 release all-target clippy passed. Pure OutputKind/view resolver shares actual WarpBro policy; sole Apps producer review closed with no findings. Latest `8127947` is docs-only, consumed source byte-identical to d411 | Earlier disk-full `21437` stopped before tests and was not a source failure. No rebuild is needed for the docs-only follow-up; consumer GPU/native color parity remains outside ordinary-test receipts |
| SquareBob UI/build | Main `8ac9d58f6f4245d431ed7f892e34ad1d9c58d511` verified against exact remote, source clean/current. Final ordinary `48040`: compile 1m40s, 57 passed / zero failed / three ignored in 0.98s. Production `21158` passed in 1m59s; actual squarebob.exe --help exited 0. Release locked workspace/all-targets `33869` passed in 4m55s without warnings/errors | Narrow-toolbar measurement fixture is corrected; production UI unchanged. Historical visual rejection stands; no new screenshots accepted. Ignored GPU/demo tests were not run; no extra demos/movies/screenshots after the passing ordinary suite |
| SquareBob export freeze | Published `b487bbb` → camera restore `6c1e4edbcd470aac466b77c245acbdca2023d350` → final `8ac9d58f6f4245d431ed7f892e34ad1d9c58d511`. All nine ordinary Frozen session tests and camera regressions passed. Resource invalidation is separate from authored-camera action; sole Plan review P2 corrected | No zero-copy encoder claim. Full EXR metadata and remaining OCIO/proxy/denoise/snapshot/toolbar features remain open; passing ordinary/build gates do not execute ignored GPU freeze evidence |
| WarpBro | Published docs `d1bc4ba8`, production source `cd505435cfafff60102f6038e55fb44d148f8e36`; plain all-target 28.01s passed. Ordinary oxide `27252` passed 254 / zero failed / ten intentionally ignored, four previously certified native-movie fixtures filtered, in 35.80s. Numerical four routes/all guides exact with unchanged tolerance; 15 OCIO tests passed, including invalid-input/look warm-cache | Full DE boundary is about 27–28% slower. Narrow affine attempt failed the original oracle because inline-never intent was dropped by the compiler; attempt was reverted, published correct source frozen. Authorized cuda-oxide inline-intent propagation fix uses an owned producer checkout/tool build, pending plan/compile; no global tool replacement or performance acceptance. Earlier strict clippy stopped before WarpBro linting at the existing fractal-materials eight-argument factory |
| App queue | COLMAP seven/seven actual ingest tests and native release (35m30s) passed at 8c4+9d; coherent-d411 follow-up `8d44976db7e76f3a1f34ab535d72eb3cdd365d7c` published. EXV `2ef5db6`, RV `d1cafd2`, Watermark `9628dcb`, GitNexus `0aa9f32`, OTIO `229dfb4` published coherent d411 locks. Audio `db929c8` published coherent 8c4 lock and locked offline metadata passed | COLMAP cached native `43593` is rebuilding dependencies. Audio has no current new native receipt. OTIO's 21 tests/demo receipts stand by byte-identical used config-source proof; lock publication alone does not certify every app's native UI |

FrozenRenderSession preserves actual GPU causes through worker errors and a final readback
drain; its 3D SDR path uses RGBA16F float rather than pre-gamma 8-bit data. Configuration/LUT
bake failures are strict. Final ordinary consumer suite and production/workspace gates passed.
The narrow-toolbar fixture now measures natural ordering at 1200px with both labels visible,
then actual 280px scrolling, camera-button visibility and the separate scrollbar row; its
former hidden-label failure diagnostics are retained. This fix changes no production UI.
The user requested push of all authorized work to main. Preserve the existing videos below;
do not run gratuitous extra videos/renders after their tests pass.
Additional review/demo movies and screenshot regeneration are skipped once relevant tests
pass. Historical narrow visual rejection remains recorded; it is not a claim of accepted
new screenshots or an optional-artifact blocker. Necessary consumer logic/runtime tests continue.

Compiler work is planned against clean tracked `f3f1098` in
`C:/projects/projects.rust.cg/nv/cuda-oxide-windows`, using an isolated candidate worktree.
The proposed canonical InlineIntent is Default/Hint/Always/Never, with Rust Always/Force
mapping to Always and attributes carried through MIR to LLVM. It is not implemented or accepted
by these documentation/test receipts.

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
SquareBob DB is still foreign-locked and was not repaired. The owned isolated GITNEXUS_HOME
`analyze --out` wrote a fresh graph at the original current checkout, but CLI graph_status/
list_repos loaded the stale canonical `.gitnexus` instead. Registry/meta freshness does not
certify tool-query freshness; a new test was absent from impact. Exact reproduction is in
oh-my-harness BUG3.md, 2026-10-07. Use a SHA-verified owned mirror with default local
`.gitnexus`, without `--out`. Authoritative pre-8ac mirror is
`C:/projects/projects.rust.cg/.codex-worktrees/squarebob-main-gate-20261007-0407c6ea38691`:
two latest files SHA-verified, local DB 5968 nodes / 13831 edges, HEAD/index `6c1` fresh
at 08:21:08; precommit detect covered nine symbols / two files, LOW / zero flows.
These receipts cover that owned mirror, not original/global graph repair. GitNexus's canonical
resolved-storage architecture fix is underway in Apps, with source/tests still pending.
Earlier CRITICAL camera-path impact/scoped two-file LOW detect and exact remote commit/push
receipts remain historical; do not reuse them as proof of the newer test's graph coverage.
Shared widgets and Audio post-push full indexes passed and are fresh. WarpBro's earlier isolated registry graph has
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
