# WarpBro PLAN

The one working document: rules, state, open work, findings. Keep it current: tick or drop done
items (details go to CHANGELOG.md), add new requests, keep open work sorted by priority. Chat in
Russian; code, comments and files in English. Reusable UI goes into egui-widgets-rs.

## Rules that bite

- Plan first; wait for approval before non-trivial changes.
- Never `git reset` / checkout-rollback; restore a file with `git show <rev>:<path> > <path>`.
- No `Co-Authored-By` lines. Commit only your own files: parallel sessions and WIP auto-commits
  share these repos. `cargo fmt` reformats foreign code: check `git diff -U0`.
- Systemic fixes, SSOT, no hacks, no silent fallbacks; verify everything; two reviewers before
  committing; tests sparingly (one targeted run at the end).
- Push when told. Own deps via SSH on `main`, third-party HTTPS, no `[patch]`.
- Build / test only through cuda-oxide: `cargo oxide build`, `cargo oxide test -- --release -- <filter>`.
- References: D:\Projects\vfx.ref.

## State (2026-10-06)

This table is the previous shift's snapshot. The authorized continuation and current gates are
recorded below; do not use the old branch names as the current publication state.

| Repo | HEAD | In flight |
|---|---|---|
| warpbro-rs | a707c9b | - |
| egui-widgets-rs | 55d1fbd | `feat/one-attr-editor`: one attribute-editor widget set |
| playa | ba49d14 | `feat/render-settings` (tier cache, background gate, templates, progressive viewer); `feat/nested-effects-menu` d1e8236 (awaits a visual check) |
| ofx-rs | bce0b90 | - |
| render-rs | 207473e | - |

## Authorized continuation (2026-10-06)

### Latest gates and order — 2026-10-07

The older receipts below remain historical; use this checkpoint for current publication and
unfinished gates. No extra approval is needed for the already authorized scope.

- [x] Publish GitNexus `ec11c6a2ad54eee9b27f477c531b6b9625ec18aa`, followed by coherent-d411
  lock `0aa9f32`: earlier six Git + 87 MCP
  tests, native GUI/CLI build (42m38s), two actual help exit-0 calls and post-commit reindex
  passed. The foreign installed MCP was not replaced.
- [x] Publish shared toolbar/output policy `d4117f50`. Solid toolbar/
  mandatory ToolbarState passed 15 ordinary tests on Rust 1.96; the actual 44px
  natural-button/font fixture is corrected. Pure OutputKind/view resolver in the existing
  egui-display export module shares actual WarpBro policy; one producer review closed with
  no findings. Six policy tests did not execute in `21437`: disk-full regex-syntax IO stopped
  compilation. No files were deleted; later six output-policy tests and strict Rust 1.96
  release all-target clippy passed. Sole Apps producer review is closed without findings.
- [x] Verify scoped SquareBob FrozenRenderSession/camera logic after `b487bbb` and restore fix
  `6c1e4edbcd470aac466b77c245acbdca2023d350` against d411. Nine files plus ColorPipeline
  now own the frozen DirEntry/camera/quality/options, own color pipeline and existing 3D/2D
  renderer resources moved from author fields. Author time/flags are not mutated; GPU error
  causes survive worker failure/final readback drain. 3D SDR uses RGBA16F float, strict config/
  LUT failures. Resource invalidation is separate from an authored-camera action; four
  finish/cancel × failed-path camera regressions and all nine ordinary Frozen session tests
  passed in latest-source `95027`. Sole Plan review closed with
  one P2, corrected in `6c1`; no repeated review was requested. Release bin suite `95027`
  completed its release build in 19m16s: runtime 56 passed, one failed, three ignored.
  Full suite is not passed. No zero-copy encoder claim; full EXR metadata remains open.
- [ ] Finish SquareBob toolbar/CamClip/preset acceptance. Foundation 24 UI + one GPU tests
  passed and seven screenshots were generated. Root visual inspection found narrow overlap/
  floating-bar and fade-fixture issues. Existing camera/preset/store/pointer UI tests passed.
  Remaining latest-suite failure is
  `app::viewport_toolbar::tests::narrow_viewport_toolbar_horizontal_scroll_reaches_camera_slots`:
  `toolbar text geometry` while measuring a hidden clipped label with no emitted painter
  geometry. Root is diagnosing before a fix; correction/rerun is still open.
  No new screenshots are accepted. Per the latest user instruction, extra screenshots/demo
  movies are skipped after relevant tests pass and are not mandatory completion blockers.
  Five mandatory nullable LMB-recall/RMB-store camera slots and named preset buttons remain
  incomplete WarpBro OCIO/proxy/denoise/snapshot/full-toolbar parity.
- [ ] Finish runtime WarpBro DE/cache/toolbar/paired-benchmark verification after published
  `cd505435cfafff60102f6038e55fb44d148f8e36` (four files: lock/app/gpu/ocio), following
  baseline `301b61e`. d411 integration passed plain all-target in 28.01s, no warnings.
  Strict clippy stopped at the existing fractal-materials eight-argument factory before
  WarpBro linting; no strict WarpBro clippy PASS. Earlier schema gate recorded 256
  passed / one failed / nine ignored; its remaining lens fixture was fixed and four targeted
  lens tests passed. Actual OIDN HDR/all-quality, PNG-video PQ HEVC/PQ ProRes/HLG HEVC/SDR
  ProRes tags+CLL, and Vulkan partial exact `24000/1001` each passed one gate. Specialized
  Fast one-object radiance still fails at max_abs `5.9247017e-5`; Fast two-object and
  StandardSurface one/two-object radiance/albedo/normals are exact. Trace the shared field
  boundary; no tolerance change is claimed. Earlier display/view key omitted current input/
  look; latest DE/cache runtime behavior and paired benchmark remain pending.
- [ ] Finish the native app queue: COLMAP seven/seven actual ingest tests and 35m30s native
  release passed at 8c4+9d. Its coherent-d411 follow-up
  `8d44976db7e76f3a1f34ab535d72eb3cdd365d7c` is published; cached native `43593` is
  rebuilding dependencies. EXV `2ef5db6`, RV `d1cafd2`, Watermark `9628dcb`, GitNexus
  `0aa9f32` and OTIO `229dfb4` published coherent d411 locks. OTIO 21 tests/demo receipts
  stand by byte-identical used config-source proof. Audio `db929c8` published coherent 8c4
  lock; locked offline metadata passed, with no current new native receipt. SquareBob floating
  Git refs were checked against actual remotes; fscan pin is `94dd173`.

All videos remain under `C:/Temp/bob`; exact retained folders/dimensions are in
[HANDOFF.md](HANDOFF.md). Twelve SquareBob technical three-frame 64x64 clips are distinct
from the actual WarpBro Kvazaar/Vulkan motion and fractional/HDR fixtures. Old screenshots
in `ui-20261007` do not establish narrow acceptance.
Push all authorized work to main as requested; preserve existing artifacts and avoid gratuitous
new videos/renders after their tests pass. Publication is not all-scope completion.
Skip additional review/demo movies and screenshot regeneration after relevant tests pass;
continue necessary logic/runtime tests and retain the historical narrow visual rejection.

Scoped graph receipts: owned SquareBob mirror full-force `54164` passed (5861 nodes / 13403
relationships), then serialized `94939` and five-file reindex passed; original canonical DB
remains foreign-locked and was not repaired. Isolated GITNEXUS_HOME `analyze --out` wrote a
fresh graph, but CLI graph_status/list_repos read the stale canonical `.gitnexus`; impact
could not find a new test. See BUG3.md, 2026-10-07: fresh registry/meta is not fresh tool-query
evidence. Root is preparing a SHA-verified owned mirror using default local `.gitnexus`,
without `--out`; no recovery is claimed yet. Earlier CRITICAL camera impact/two-file LOW
detect and exact remote commit/push receipts are historical, not newer-test graph coverage.
Shared widgets and Audio full post-push
indexes passed and are fresh. WarpBro's earlier isolated registry has 3556 nodes / 9918
relationships; combined expected detect scope, 503 symbols / 42 files, was CRITICAL and
inspected successfully. No global/original graph freshness claim.

The operator authorized systematic completion of [HANDOFF.md](HANDOFF.md): update the whole
dependency chain, verify every claimed result, fix architecture and shared ownership, and use
the references in `D:\Projects\vfx.ref` when they cover the behavior. No further approval is
needed for the formerly deferred av-player feature fix. The earlier rule requiring two reviewers
is superseded by ONE independent review per finished branch plus native screenshot inspection;
do not repeat reviews for every small correction. WarpBro build/test commands remain cuda-oxide.

- [x] Publish shared curve-domain geometry and phosphor vocabulary: egui-widgets-rs main
  `c2bae44`; 28 config/icon tests including docs and 12 ramp tests plus docs passed. Independent
  review found no defects; generated ramp screenshots were inspected and passed.
  Application native screenshots are still a separate open gate.
  Final shared vocabulary is published in main `a6090f0`, including `920f71d` COLMAP/WarpBro
  icons and application controls; its config gate passed 28 tests. Final widgets main
  `9d459d2` passed the attr-grid gate: 50 passed, one ignored.
  Shared export is now published in `3443b6f46a2a72526c9542b1e7d8e528c88142bc`:
  recorded source gates passed 31 tests (measured cLLI, real FFmpeg/ffprobe six SDR/PQ/HLG
  HEVC+ProRes combinations, cancellation) and strict release all-target clippy `-D warnings`.
  Consumer compilation/acceptance remains open.
- [x] Publish audio-rs `76b66b0`: remove obsolete av-decode SIMD feature request; audio-decode
  checked against ffmpeg-rs main `a6676574`. Video evidence belongs to the Playa native IO gate below.
- [x] Publish FFmpeg canonical metadata checkpoint `8c4`: `bt709_limited` and checked
  nclx passed six release tests. Application launch/source/UI gates remain separate.
- [x] Publish ofx-host-egui main `9377dce` (following `402e90a` / `0a3149e`), including `EffectStack`, common numeric-column width, live `AttrMetrics`,
  responsive curves, and background coordinates from the widget's data-domain callback rect.
  Latest host-egui release suite passed: 52 unit + four instance + one param_interacts + one
  doctest. The old fixed 260x150 assumption and the adapter's second inset are corrected
  and verified by the real parametric-interact gate. Widget owns
  inner geometry; host consumes the callback rect directly and owns parameter/stack edits.
  Consumer/native integration remains pending under the Playa and screenshot gates below.
  All-numeric OFX descriptor-span audit and final 23 GPU screenshots passed and were
  visually inspected. Backend-hint overlap, RGB R/G/B labels, and status wrapping
  are verified; the final follow-up is published. Native application acceptance remains open.
  `9377dce` changes fixtures/lock/changelog only; production is unchanged from `402e90a`.
  Final widgets `9d459d2` gate passed 58 tests (52 + four + one + one).
- [x] Publish Playa foundation `8fa38c4d0483945613fc1ca5b6428344b7225df9`; use the published ofx contracts,
  shared `EffectStack`, and one measured numeric column across ordinary and OFX sections.
  347 engine/entities tests passed, seven GPU tests ignored. Independent review findings
  for nonfinite matrices/scalars, empty typed arrays, and non-EXR export validation were
  fixed. Native IO passed 46/46 after canonical ffmpeg CFR and AAC time-base fixes;
  earlier app 82 and UI 125 gates passed, and the actual native binary gate passed.
  App all-target check and fractional-rate encoded-file regression at 29.97 fps passed.
  Existing accessibility regression passed on widgets `9d459d2`. Final native IO, 347 engine/entities,
  82 app, 125 UI and nine cache gates plus native/build now passed and are published.
  Measured sysinfo memory-query cost improved from 106s to 0.56ms. Native UI/AE acceptance
  and Output/Queue remain open. PQ-tag tests do not certify 10-bit precision.
- [ ] Finish WarpBro old-data removal, physical f-number lens model, and camera recorder;
  check all targets and run targeted oxide tests plus production release/native gates.
  Current working-tree code is not yet a verified publication.
  The rejected GPU architecture refactor was restored only in its owner's eight files
  and new display file; parent MaterialSource, codec and dependency edits were preserved.
  Diff check passed; this restoration does not establish build or HDR acceptance.
  Coherent lock is updated to published USD `eb92860f9c55281c67bcee6985dce622f649f81e`
  (fallible compound creation/all seven sample admissions; four native gates passed,
  clippy zero errors) and latest Alembic `7538a727`. Shared-source all-target check passed
  in 12.73s (session `31284`) after precise curves accessor / `sample_at` consumer fixes.
  Actual oxide native CFR baseline `74654` passed one test (52m45s reported). Updated
  canonical `8c4` CFR/coded-metadata all-target `57365` passed in 2m55s, zero warnings.
  Full ordinary oxide `71257` is active, not passed. Remaining behavior/build/native gates stay open.
- [ ] Finish apps: Watermark `c4040f0` is published; latest release passed in 9m59s and
  three icon tests passed, with native UI still open. Colmap's av_graph video port/native
  gates are active under its application owner; no completed-app claim. GitNexus's supported
  SDK rename xtask `39631` passed five tests (23m33s build / 10.08s runtime). Latest integrity
  gates passed six Git + 87 MCP tests; sole focused review closed with no findings. Default
  native GUI + CLI build `39070` is active. Canonical SquareBob MCP graph remains locked.
  Concurrent reanalysis of the owned SHA-verified mirror encountered a duplicate File key;
  overlapping writers are a plausible, unproven cause. Serialized full analysis `54164` is
  active. Preserve foreign processes/caches; do not treat the unfinished run as a fresh index.
  Squarebob's first production bootstrap passed in 59m21s (115390976-byte artifact),
  earlier all-target locked check passed in 4m50s, and 15 pt-mats tests passed.
  Three remaining glyphs are fixed and full `cargo update` passed with widgets `9d459d2`;
  SquareBob CFR repair passed four video tests/six encoder variants and final production
  build passed in 5m57s, before HDR changes. New float capture/output color-white/strict
  DisplayLight/HDR PNG-video adapter/decode-back source consumes shared SHA
  `3443b6f46a2a72526c9542b1e7d8e528c88142bc`. Rust 1.96 production `73811` passed
  (24m49s bootstrap / 24m34s Cargo); canonical `8c4` metadata library `57163` passed
  19 tests in 7m, excluding newest launch/test changes. Later library/native gates passed
  21 + four tests. Latest four native gates passed in 2.51s (alpha, six SDR variants,
  four PQ/HLG HEVC/ProRes variants, unsupported settings); twelve retained files in
  `C:/Temp/bob/delivery-20261007-0615` include two ProRes-alpha clips. These three-frame
  technical fixtures do not certify real application motion. Source-error/extent transactional
  gate `98764` passed three tests, three ignored, in 3m20s.
  New immutable post-edit EncodeLaunchRequest/event, FrameSourceResult errors/Cancelled,
  explicit extent rejection, best-effort post-commit completion and typed SDR conversion
  are covered only by their scoped file/source receipts. Named Settings presets use the existing HashMap SSOT, buttons/
  New/RMB Save-Rename-Delete and atomic av-util-core writer; raw-pointer/persistence/GPU-shot
  tests are unrun, and the direct dependency was added after the compile closed.
  Actual canvas is RGBA16Float extended-sRGB Rec.709 display light, not raw PT/AP1;
  frozen output color and SDR still-view independent of the monitor remain open.
  Latest request adds WarpBro-style viewport toolbar/CamClip and retains all videos in
  `C:/Temp/bob`. Current UI source has five nullable full OrbitCamera/PhysicalCamera/DoF
  slots, primary recall/secondary copy/empty no-op, shared Top toolbar geometry, 2D/PBR/PT,
  physical EV, spp, Settings event/focus, existing primary SDR PNG capture and context Export
  settings. New required persistence fields have no old-JSON fallback. App gate `1177` is
  compiling; new UI tests and GPU screenshots have not passed. Full WarpBro OCIO/proxy/
  denoise/snapshot-format parity and full renderer freeze remain open. WarpBro oxide `71257`
  is still building; its new-video destination is the unique owned
  `C:/Temp/bob/warpbro-8c4-a9623942eb974620a00de18fd1bbc4e2`, routed before writes through
  a previously nonexistent TEMP junction; no completed new-video gate is claimed.
  RV full dependency refresh includes `8fa` / `73a` / `9d`, native gate pending.
  EXIF producer `77c56ae` (after `6960603`) publishes canonical EXR/JPG/JPH identity with
  the same SHAs. EXV actual
  executable `--help` passed in 3m57s after the library-only pass; own view tests are pending.
  OTIO main `73a074f` is published: final demo release passed in 29.24s and 21 tests passed.
  No final-lock refresh remains active. Native application acceptance remains open for every app.
- [ ] Inspect AE/screenshots against WarpBro at narrow widths and 100/150% scale. Confirm
  aligned rails/value boxes, channels, reset/key gutters, defaults, labels, and dropdowns.
  Native Computer Use is unavailable: initialization succeeded, then two `sky.list_apps()`
  calls failed on the native pipe (Windows os error 2); exact evidence is in oh-my-harness
  `BUG3.md`, 2026-10-06. Continue GPU kittest screenshots and supported release builds;
  keep native acceptance open. No custom helper or PowerShell UI automation was used.
- [ ] Remove curves legacy API only after Playa and WarpBro consume the new APIs successfully.
- [ ] Continue render settings phases 3-5 (Output Module templates, Viewer / Output nodes,
  Render Queue) and the remaining render/export backlog below.

The earlier unchecked items below remain historical context until their corresponding gate
passes. Dependency updates and source edits alone do not complete build or native verification.

Render-profile proposal: [docs/render-profiles.md](docs/render-profiles.md) is a Russian
DRAFT for review, covering canonical nodes, AttributeEditor flows, Viewport policy,
WorldDirect, and Output independence. No implementation-start gate has passed;
this proposal does not replace or complete the current backlog.

Overall order: published shared contracts → Playa foundation → WarpBro verification →
apps / AE acceptance → curves legacy removal after consumer checks → proposed nodes /
UI preview → WorldDirect benchmarks → Playa Output / Queue. The node design is not implemented.

Latest explicit priority: full SquareBob Encoder/HDR/Display parity by reusing actual existing
WarpBro export/color/display code deduplicated into one SSOT in the existing egui-display
export module, with thin application source adapters. The user rejected
the proposed common DisplayColorPass, shared NativeEncoder migration and new encoding/interop
architecture. Those plans are superseded; they are not active implementation authorization.
[Revised directive and preserved audit](../squarebob-rs/docs/hdr-encoder-display.md) keep the
full format/codec, precision, crop/resize, cancellation/publication, persistence and decode-back
requirements. Extraction source (PngEncoding/DisplayLight/HdrScale/HdrLevels, scale/PNG/
PngVideo/FFmpeg, view-peak measurement and BT.1886 math) is published as
`3443b6f46a2a72526c9542b1e7d8e528c88142bc`, with recorded 31-test and strict clippy gates.
Preserve alpha, atomic publication and true 16-bit PNG. SquareBob production/metadata library
receipts exclude newest UI changes; scoped native file/source gates passed, app/UI gates remain pending.
WarpBro updated all-target and actual native CFR passed; ordinary oxide is active. Full render
freeze and monitor-independent SDR view remain open. Native SDR source uses canonical BT.1886.
No duplicate math, new raw GPU target or GPU interop. The render-node
proposal remains a draft, and the existing backlog remains open.

## After the 2026-10-06 merge of feat/keys-track (verify first)

Merged with a stopped agent's WIP and WITHOUT a full build (usage limit): first job is
`FRAC_ALLOW_PLAIN_CARGO=1 cargo check --all-targets` + targeted `cargo oxide test` and fixing.
- [ ] Build check on main: Playa now main 149e03c (tier cache API), av-player pinned to Playa's
  rev 97ddb21 (ffmpeg-rs main dropped the `simd` feature Playa's playa-io asks for - tell the
  ffmpeg-rs owner or drop the feature request in Playa), exr-core 6856661 aligned with oiio-rs.
- [ ] egui-widgets main 4139221: `grid_config` is a free fn; `AttrMetrics` / `segmented` live in
  egui-widgets-config; pass `AttrGridConfig::value_box_width` across the AE sections.
- [ ] WIP last commit: attribute labels without the redundant family prefix (test: every label fits
  the default 180 px column), removal of the remaining old-data compat (`upgrade_material_schema`,
  orbit controls added to old documents, dock.rs layout upgrades, export.rs settings kept for old
  files, `legacy_*` names, thumbnail_snapshot no-document path).
- [ ] Context-menu "Reset to default" (fe032fa) has no test.
- [ ] Flaky `export::tests::cuda_export_coordinator_samples_animation_and_writes_each_frame_once`
  ("autonomous export timed out"): isolate GPU tests (see open work).
- [ ] Camera recorder (own egui-prefs2 panel: TRS / Focus / Zoom / f-number; keep speed or fit to
  work area; overrides existing keys; fitter simplification).

## Open work (priority order)

1. **Attribute Editor parity** (in progress). The old attr-table rows (fill-width sliders, aligned
   value boxes, reset / copy / paste / random, segmented strips) were lost in 13b1caa / 9416673 /
   f2c73ab. egui-widgets-rs `feat/one-attr-editor` restores them in egui-attr-grid (attr-table,
   egui-attr, egui-vector deleted). Then: WarpBro (field defaults for reset, row actions on,
   Preferences onto the grid), Playa layer AE and ofx-rs `ofx-host-egui` onto the same grid.
   Screenshots before / after.
2. **Rough glass energy** (render-rs standard-surface-bsdf): no multiple-scattering compensation for
   dielectric transmission (furnace 0.81-0.93 inside at roughness 0.5, not re-verified). Plan:
   furnace grid baseline -> references (Turquin 2019, Cycles multiscatter GGX glass, OpenPBR) ->
   R/T compensation table, Rust+WGSL parity -> furnace tests (>= 0.99 rough, 1 smooth) -> render.
3. **Hybrid DE surface** (BUG1 leftovers): discontinuities at fold / escape transitions and the
   6-probe normal's p99 11 deg on converged hits; isolated cost of the 6-probe normal. Over-relaxed
   sphere tracing only with a safety check (Hybrid DEs are not Lipschitz).
4. **Render / Encode colour output**: one option set across EXR (ACEScg / ACES2065-1 / linear 709 /
   2020 + display-referred), PNG (SDR / HDR10 / HLG), MP4 (SDR [+ HDR10]). Plan first.
5. **In-process HDR video** via ffmpeg-rs (requirements: ffmpeg-rs/BUG3.md); then delete the ffmpeg
   stopgap (`PngVideo`, `encode_video`).
6. **GPU test isolation**: Vulkan Video export tests fail with "Posix(38)" when run beside the
   render_service / color GPU tests (6/6 on clean HEAD), pass alone; `vulkan_video()` serializes only
   within export.rs. Find the conflicting GPU use; serialize or fix.
7. `cuda_specialized_world_materials_preserve_radiance_and_affine_guides` (ignored): radiance
   max_absolute 5.92e-5 > 3e-5 since b5d562a - investigate.
8. io_service: four `let _ = done.send(..)` drop a result when the app has gone - log it.
9. Six soft slider ranges were picked without measurement (aperture, focus, orbit speed, thin film,
   sun angle, hit epsilon) - operator to check.
10. Dead per-object `render` fields on fractal nodes (only `/render/iterations` is per object) -
    drop when the object model is split (low).
11. **Convergence**, each step measured with `--world-bench` + `tools/convergence.py`: firefly clamp of
    indirect contributions by AP1 luminance (off by default, labelled biased, report the energy
    loss); Russian roulette from AP1 luminance with a minimum survival probability and start depth
    2-3 (gpu.rs still uses `max3(throughput)` from bounce 1); then profile secondary-ray marching
    before any path guiding / light cache.
12. Viewport status bar: adaptive sampling progress (active tiles / converged %).
13. ACEScg golden check: a fixed grey scene under a white sky before (b4fd7a5^) and after must match
    within noise; record the numbers in CHANGELOG.
14. CHANGELOG catch-up: PNG export, ~/.warpbro profile, built-in templates, camera slots, material
    assign menus (2480a63, a871707, 96d17b7), Owen-Sobol sampler (e0aa6bb), adaptive sampling
    (51b1d1b), Kvazaar I-frames, Material Library, Timeline divider.
15. **Native verification pass** (never done; tests cover logic only): narrow widths and 100/150%
    scale, layer drag/drop and RMB menus, Material Library layout, cached HDR playback, keyed and
    unkeyed camera flight, File Open/Save during rendering, HDR environment comparison, UI latency /
    frame-time variation under heavy render + OIDN. Screenshots to ~/.warpbro/diagnostics.
16. Reusable workspace extraction (after item 1): Outliner and Timeline adapters, adaptive Gallery,
    Material Library browser + publish fractal-materials, shared hotkey dispatch / file-dialog
    history / compact metrics, Curve Editor on the shared keys. Specify the host contracts (stable
    ids, select vs assign intents, one transaction per gesture) first; pilot in Playa.
17. ofx-rs fractal kernel copy still uses pcg4d white noise and no adaptive sampling: resync with the
    Owen-Sobol sampler and `active` / `moment` buffers, or record that ofx-rs keeps its own (low).
18. Kvazaar inter-prediction corruption (repro: export.rs `hevc_motion_fixture_encodes_every_source_frame`)
    is mitigated by I-frames only; file it in ffmpeg-rs (low).
19. Deferred: HDR environment blending, area lights (operator decision; low).

## Gotchas

- egui: nested `ui.input` inside `hotkeys::active` deadlocks; read `active` first.
- Hotkeys need exact modifier matching. egui tests: send key releases (a second press is a repeat).
- render-rs: fetch before committing (local main was once behind origin).
- Playa: `cargo test -p playa-app <filter>` (not `--bin playa`); the OpenFX host needs
  `--features ofx`; debug builds overflow the main stack at start - run release.

## Key files

- WarpBro: src/world_ui.rs (AE), src/render.rs, src/gpu.rs, src/export.rs, src/render_service.rs,
  docs/color.md.
- render-rs BSDF: crates/render-engine-pt/standard-surface-bsdf/src/{microfacet.rs,sample.rs,wgsl/microfacet.wgsl}.

## Findings worth keeping (BUG1, 2026-10-05; scene: copper-turbine-kifs, frame 27)

- **Normals**: tiny gradients lost their direction (fixed 4f66e83: scale by the max component,
  retry smaller steps). The 4-probe tetrahedral normal was biased; now 6 central probes at eps/4.
  On converged hits (4096 steps, 19 311 stable neighbourhoods): old tetra median 0.45 deg / p90
  2.96 deg; 6-probe median 0.016 deg / p90 0.043 deg / p99 11.1 deg. Primary hits and their
  positions unchanged bit for bit. Regression: `cuda_curved_hybrid_normals_match_converged_field_gradient`.
- **Phantom hits**: at 256 steps 37% of primary hits were rays that ran out of steps (half were
  misses, half 0.30 units too deep, normals off by a median 48 deg). Fixed: `Outcome::Unresolved`,
  one policy per ray kind (camera: background; bounce: path ends; shadow: blocked; Direct: Keinert's
  half-pixel rule), unresolved marches counted in the status bar, presets at 4096 steps, step cap
  independent of the budget, glass interior probes a setting (256). Cost: primary rays 37.1 ms
  (256) vs 47.4 ms (8192); full 6-bounce 16 spp render 978 ms vs 958 ms. As Mandelbulber2 and
  Fragmentarium do (a ray out of steps is a miss). Ported to ofx-rs (bce0b90). Regression:
  `cuda_march_out_of_steps_is_a_reported_miss_not_a_phantom_hit`.
- **Glass energy**: clear glass lost energy at grazing angles (exact Fresnel for the delta lobe but
  the MaterialX albedo approximation for its throughput; IOR 1 gave reflection weight 0). Fixed in
  render-rs 207473e (`exact_interface`: albedo = exact F for smooth / IOR-1 interfaces, Rust+WGSL).
  Scene furnace 1.00003.
- **Not the cause**: material (matte shows the same facets), denoiser (raw accumulation unchanged),
  iterations (orbits escape after 4-9 of 12). bailout and step settings change the surface itself.
- Artifacts (not in the repo): ~/.warpbro/diagnostics/{surface-audit,surface-central}; the diag
  branch `diag/bug1-march` (worktree ../warpbro-diag).
- Sources: Hart, Sphere Tracing; Christensen (Fragmentarium) lighting and Mandelbulb DE;
  Mandelbulber2 shader_calculate_normals.cpp / compute_fractal.cpp / ray_recursion.cl; Keinert et
  al. 2014 Enhanced Sphere Tracing.
