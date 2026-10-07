# HANDOFF (2026-10-06, end of the curves / widgets / render-settings wave)

Cross-repo state for the next session. Repos live in `C:\projects\projects.rust.cg\cglibs`.
Per-repo detail: WarpBro `PLAN.md` ("After the 2026-10-06 merge" section), Playa `todo.md`
("Open after the 2026-10-06 merge"), each repo's `CHANGELOG.md`.

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
