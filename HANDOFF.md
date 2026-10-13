# HANDOFF (2026-10-13, ~00:00)

Current resume state (2026-10-13): animation parameters were merged in `fractal-rs/main`
`1f0d122` and `warpbro-rs/main` `059984d`. The accepted normal correction is now pushed on
fractal-rs main as `f3e55bd`; WarpBro `6289cc8` is now pushed on main. Its four Git-pin lines
change without a local patch and preserve `windows-sys`. The published-candidate 326-test
suite passes: 312 passed, 0 failed, 14 ignored in 366.81s (compiled in 3m1s), recorded in
`published-full.log`.
The dated snapshots below preserve the interrupted session's history; use the
[resume completion plan](plans/20261013-resume/plan.md) for the current acceptance ledger.

- Three operator template JSON files (amethyst / box / mandelwhit) are converted, with `.bak`
  files retained, and now pass Rust scene-schema validation. The existing default-feature host
  `WarpBro-a14b1f1310ab8852.exe` ran
  `operator_templates_decode_with_current_scene_schema --ignored --nocapture` with
  `WARPBRO_TEMPLATE_PROBE_DIR=C:/Users/joss1/.warpbro/templates`: 1 passed in 0.06s, loading all
  three templates. No schema change was needed for the CUDA correction.
- EXR main/tag now points to `fa8aa3376f78a72794785af347b1a1733d68ff87` (docs commit
  `fa8aa33`), annotated tag object `d65355fe9e1492117b540274ee813371e415f744`. Six local
  bootstrap stages pass. Release CI `38382275628` completed successfully (10 jobs, 0 failed);
  v0.1.0 is public/non-draft, published 2026-10-13 11:04:59 UTC with 3 ZIPs/5 wheels. All asset
  sizes/SHA-256, ZIP CRCs, wheel RECORD hashes, and architectures/tags pass. The downloaded
  Windows CLI passes version/help/info and identifies the real PQ PNG as 1920x1080 RGBA.
  Linux/macOS binaries were not executed, wheels not installed, no release-artifact GPU shot.
  Old tag `1671769` is preserved locally as
  `backup/v0.1.0-before-color-resume`; owned duplicate branch/main runs were cancelled because
  the tag covers the same production code.
- WarpBro/fractal dependency reachability: 34 distinct own Git pins verified, 34/34 GitHub API
  checks passed. The existing EXR `7db` pin is reachable and was retained.
- Larger fractional-step unsigned-DE geometry diagnostics are isolated in `fractal-geometry`
  and `warp-geometry`, branch `fix/fractional-bulb-march`. Those experimental changes are not
  in the accepted main shader. Zero fallback pixels and the observed pick-depth jump
  `2.3983283` -> `2.4022722` disproved the earlier fallback-trigger explanation.

## Running agents (background, both commit to branches only, no push, no merge)

1. **Mandelbulb animation parameters** - repos fractal-rs and warpbro-rs, branch
   `mandelbulb-params` in both. The six asked for:
   1. Julia mix t (c = mix(p, J, t), dc/dp = 1 - t) in the shared `orbit_constant`;
   2. fractional iterations (smooth blend between floor(n) and floor(n)+1);
   3. c offset vec3 (z = z^p + c + k);
   4. spherical inversion pre-transform (centre, radius, enable/strength) with DE correction;
   5. separate theta / phi powers - check `angle_scale` already does it, no duplicate control;
   6. orbit-trap params (point, axis, scale) animatable.
   Rules given: append ABI slots with neutral defaults (bit-identical old scenes, ofx-rs
   packs the same slots), fast paths disabled when a new value is non-neutral, a TEMPORARY
   `[patch]` in warpbro-rs/Cargo.toml for building the kernels (backup of Cargo.toml / .lock
   in the agent's scratch dir; must be restored before its commit), `cuda_*` tests.
   State at 23:55: 3 files changed in fractal-rs (params.rs, fractal_core.rs,
   fractal_kernels.rs), 9 in warpbro-rs (scene, world, presets, render, app, tests,
   Cargo.toml / .lock with the patch), cargo oxide building; no commits yet.
   After it: review (one reviewer), push fractal-rs main, bump warpbro-rs Cargo.lock rev by
   hand (never `cargo update`), check `git diff Cargo.lock | grep windows-sys` is empty,
   merge to main, push.

2. **exr-view COLOR_PLAN** - repo exr-rs, branch `color-plan`. Spec:
   exr-rs/COLOR_PLAN.md (pushed on main as a784528): frames keep pixels as read + a colour tag
   (oiio:ColorSpace / cICP -> interop ID, EXR colorInteropID / chromaticities, then the
   config's file rules, then integer -> srgb_rec709_display, float -> lin_rec709_scene); one OCIO
   processor per frame (input = frame space), views filtered by viewing rules
   (`get_views_for_colorspace`: a PQ PNG gets "Video (colorimetric)", PQ -> PQ identity); remove
   the import-time linearisation, the /203 and light_gain of 8e7f166. Visual check: the PQ PNG
   C:\Users\joss1\.warpbro\out\2026-10-12_23-11-51\frame.000000.pq.png must look like WarpBro's
   viewport in exr-view (it was near black: ACES view applied to an ACES-rendered picture).
   State at 23:55: 8 files changed (exr-import, exr-view image.rs / transfer.rs, cli convert,
   Cargo.lock), no commits, no [x] in the plan yet.
   After it: review, check Cargo.lock change, merge, push.

## State update ~01:00 (2026-10-13)

- **color-plan** (exr-rs): first pass done and committed on branch `color-plan` (40a9322,
  649ff08; not pushed). Review = REQUEST CHANGES. The agent (`colorplan`) is now doing, on the
  same branch:
  1. **Architecture (operator: systemic, no crutches): two stages for every frame.** DECODE:
     frame space -> `scene_linear` (OCIO ColorSpaceTransform; no-OCIO built-in curves) - the
     ONLY input of effects, pyramid levels, scopes, inspector, exposure. DISPLAY:
     `scene_linear` -> display/view (+look), view and look chosen by the SOURCE class (scene /
     video, remembered per class). PQ / sRGB stills stay an identity because OCIO's display ->
     scene bridge (inverse default VT, studio Un-tone-mapped) and the Video view's scene ->
     display bridge (same VT forward) cancel; must be proven by tests; a non-invertible default
     VT = typed error. One decode fn + one display fn (WGSL + CPU twin), composed in one
     shader. COLOR_PLAN.md "The model" to be rewritten first.
  2. Review fixes: looks per class (scene look like ACES RGC must not touch PQ frames); EDR
     displays (Display P3 HDR) not SDR, SDR headless PNG only on an SDR display or typed error;
     whiteLuminance EXR round trip exact on relative displays (test 203 nits); processor cache
     key before build (sequences rebuilt per frame); unmatched EXR chromaticities = pre-matrix
     or typed error; LOWs (premul doc, explicit builtin under OCIO, presets for video frames,
     with_capacity clone).
  3. Effects / pyramid on linear light = merge blocker (it was a regression of the branch).
  - Needs exr-rs Cargo.lock oiio-rs rev bumped by hand to fcad4c0 (vfx-ocio Encoding::Edr,
    pushed on oiio-rs main by the lead).
  After it: re-review, my screenshot check of
  C:\Users\joss1\.warpbro\out\2026-10-12_23-11-51\frame.000000.pq.png, merge to exr-rs main,
  push, then the exr-rs release: SAME version, delete the old GitHub release and create it anew
  (operator's order).
- **mandelbulb-params**: agent `mandelbulb` was INTERRUPTED at ~00:56. The lead committed its
  work as WIP checkpoints (unreviewed): fractal-rs ecb3919, warpbro-rs 9d62773 (both on branch
  `mandelbulb-params`). warpbro-rs Cargo.toml / Cargo.lock still carry the local [patch]
  (uncommitted; restore from main before switching branches: `git show main:Cargo.toml >
  Cargo.toml`, same for Cargo.lock). Agent resumed with orders to write
  MANDELBULB_PROGRESS.md and finish.

## WIP backups (agents' uncommitted work, taken by the lead)

00:16: C:\Users\joss1\AppData\Local\Temp\claude\C--projects-projects-rust-cg-cglibs-warpbro-rs\7d561394-85ba-41bf-95b3-a7611591945c\scratchpad\wip-backup-0016
(`<repo>.diff` = `git diff HEAD` on the branch tip named above, `<repo>-untracked/` = new
files). Restore: `git checkout -b <branch> <base>` then `git apply <repo>.diff` and copy the
untracked files back. Bases: exr-rs a784528, fractal-rs 75e8412, warpbro-rs 8535906. Both
agents were told at ~00:12 to commit WIP checkpoints and keep progress files (COLOR_PLAN.md
marks; warpbro-rs/MANDELBULB_PROGRESS.md).

## Done today (pushed)

warpbro-rs main: picking (0150925, fe4c152), Tab menu via egui-tab-menu (86def17), ` hides
panels / L flight lock (68924d2), default layout + controls (1d1a44c), smooth flight look
(c124dad), Butterworth Smooth / pinned Simplify (9dd00ca), docs + CHANGELOG 0.1.0 (02c5a6c),
new-scene marks 1 and 2 (02ce96e), built-in Venice env default (8535906).
fractal-rs: pick kernel (c80f708), fractal-env builtin maps (75e8412).
egui-widgets-rs: egui-tab-menu (25313cb, README 96fc02f, docs a830c26).
curves-rs 2d82329, gitnexus-rs cam-controls cb9dee0.
Release v0.1.0 tag at 86def17 (CI built both archives; local dist/ zip 86def17). Commits after
the tag (docs, marks, Venice env) are not in the release.

## Open (not started)

- Tab menu History / favorites / usage saved in settings.
- ofx-rs should take its built-in HDRI from fractal-env (BUILTINS there now; ofx-rs still has
  its own copy) - another session's repo, coordinate.
- Gallery and README screenshots refresh with environments (operator may supply scenes).
- WarpBro HDR10 PNGs are 1.0 = 100 nits (OCIO); exr-rs after COLOR_PLAN reads them through
  OCIO, so no change needed in WarpBro unless the operator wants BT.2408 203-nit files.
- Check the flight look feel on screen; Cells mask benchmark; 1-ULP cuda_specialized mismatch.

## State ~01:25 (2026-10-13) - merged and pushed

- Mandelbulb parameters MERGED: fractal-rs main 1f0d122, warpbro-rs main 26c69f9 (merge +
  Cargo.lock bump). Operator templates converted (amethyst / box / mandelwhit, .bak beside).
  Not reviewed, WarpBro not built after the merge: build and run first.
- exr-rs main 197417f = color-plan + review fixes + the two-stage decision in COLOR_PLAN.md;
  tag v0.1.0 moved there, old release deleted (CI release not checked). Agent `colorplan` is
  implementing the two-stage model on branch `color-plan` (not merged): merge it, re-tag.

## Separate follow-up: fractional unsigned-DE geometry

The earlier hypothesis attributed the jump (~2% of hit pixels on the bulb preset) to
DE(n) = 0, a fallback normal, and normalisation of w grad DE(n+1) for any w > 0. It proposed
`normalize((1-w) N_n + w N_(n+1))`, or a weighted fallback correction, to make normals
continuous at the integer endpoint. This is a historical proposal, not a verified cause.

The resume measurements found zero fallback pixels and a pick-depth jump from `2.3983283`
to `2.4022722`. They contradict the claimed fallback trigger. The accepted endpoint-normal correction
is `f3e55bd`; strict targeted tests pass for eight endpoint pairs and the neutral golden.
A larger fractional change still exposes a separate unsigned-DE step-budget limitation.
Default 4096-step inside 4.001 produced 0 primary hits with unresolved 1; 16384 steps produced
6144 primary hits with unresolved 0.09798177 (possibly secondary rays; attribution is not
established). Keep that geometry work isolated and do not claim all animation is continuous.
The final published-candidate suite passes (312 passed, 0 failed, 14 ignored). The separate
RAW-max-step prototype has a counterexample and is not merged; bounded band-margin
investigation remains pending. Endpoint-normal correction and published integration are done.
