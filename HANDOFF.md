# HANDOFF (2026-10-13, ~00:00)

Active release/upstream follow-up (2026-10-13): see
[RELEASE_PROGRESS.md](RELEASE_PROGRESS.md) before resuming. NVIDIA PRs
[#1509](https://github.com/NVIDIA/cuda-rust/pull/1509),
[#1510](https://github.com/NVIDIA/cuda-rust/pull/1510),
[#1511](https://github.com/NVIDIA/cuda-rust/pull/1511), and
[#1512](https://github.com/NVIDIA/cuda-rust/pull/1512) are OPEN/non-draft and MERGEABLE.
All four were checked locally on the pinned current NVIDIA toolchain before submission;
upstream hosted checks have not started. No direct upstream merge/push was attempted.
Our compiler main is 758c069c0 (functional backport/pin 5e1510934 plus a documentation
provenance guard correction); status-guard now passes. ansidium PR #14 remains open.
WarpBro's release candidate pins compiler 5e1510934: full locked CUDA suite 315 passed,
0 failed, 13 ignored; four-architecture cubin release build, clippy with tests and
-D warnings, fmt and RTX 3080 Ti warmup pass. Package version remains 0.1.0.
The original release archives pass backup digest/size/CRC checks; its exact annotated
tag is preserved on GitHub as archive/v0.1.0-before-20261013. Candidate publication and
v0.1.0 tag CI are the remaining steps; do not claim the new archives published yet.

MIS follow-up (2026-10-13): fractal main/origin is now
`c4222f4a31d61f92cead18a8fbb69d89c29f2690`. `power_heuristic` uses explicit
`a.mul_add(a, b * b)` so homogeneous Fast and mixed-world kernels keep the same FMA arm.
The material oracle is enabled in the normal CUDA suite; no tolerance changed. Full source
verification: 315 passed, 0 failed, 13 ignored (`target/mis-fix/source-full.log`). WarpBro's
four Git pins are updated, with Cargo.toml and windows-sys unchanged. The published dependency
passes the strict oracle with `--locked` and no patch (`published-oracle.log`), release build
(35.35s), clippy with tests and `-D warnings` (13.83s), and fmt. All receipts are in
`target/mis-fix/`. The generated cuda-device scalar intrinsic sm_80 restriction is a separate
compiler issue: [upstream PR #14](https://github.com/ansidium/cuda-rust-windows/pull/14),
linked to issue #13, is open from `ssoj13:fix/scalar-fp-target-floors` against ansidium/main
(not NVIDIA). Commits `d39feb6eb` and book-stamp follow-up `4891e74ea` are signed off and pushed.
It separates native PTX 2.0/sm_20, LLVM PTX 3.2/sm_20 and CUDA 13.3 libNVVM PTX 6.3/sm_75
contracts for all 64 scalar variants. Proof: 192 LLVM codegen/assembly captures, 64 typed
declaration canonicalizations, and 64 libNVVM operations compiled/assembled/LTO-linked with
all 64 cubin entries retained. Runtime execution of those fixtures is not claimed. The other
965 catalog records and all 1029 ABI identities are preserved. 319 generator and 1221 codegen
tests, regeneration/ABI checks, clippy, formatting, Ruff and SPDX pass. CI is running on the
PR; its initial book catalog-stamp failure was corrected after verifying 576 operations across
42 modules, with local book-stamp check passing. The installed compiler and our CUDA fork's
main are untouched. Isolated compiler checkout: C:/Users/joss1/AppData/Local/Temp/
codex-mis-20261013/cuda-upstream. Receipts: target/mis-fix/. WarpBro main `9f7ef5b` and fractal
main `c4222f4` are pushed; WarpBro CI 38430261140 passes Windows and Linux.

Current resume state (2026-10-13): geometry and normal corrections are integrated and pushed:
fractal main/origin `55d8134f093e6b3a291a0d811e7a07ab1424dd5c`; WarpBro source integration
revision `d67d3fed05641bea133a2ce4876e68cbd6dea7de` is on main. A docs-only checkpoint may
advance WarpBro HEAD without changing that source. Root verified real Git dependencies with
`--locked`, no patch: 314 passed, 0 failed, 14 ignored (328 total), compile 2m53s, runtime
322.31s (`target/fractional-normal-backup/published-geometry-full.log`). Production release
build passes in 35.48s, clippy `--release --locked --tests -D warnings` in 12.44s, fmt passes.
The production executable renders 18/18 valid, nonconstant 384x216 PNGs at 4 spp on RTX
3080 Ti; the Mandelbulb image was visually inspected. A temporary profile leaves the operator
profile untouched. Receipt: `published-geometry-gallery.log` and temporary gallery
`verification.json`. WarpBro changes are four lock pins and two tests/golden only;
`windows-sys` is preserved and Cargo.toml is unchanged. CI `38395622388` on source `d67d3fe`
completed SUCCESS: both Linux and Windows build jobs passed. Each reports 280 passed, 0 failed,
7 ignored, 41 filtered (Linux 21.25s, Windows 35.45s), with release build/archive success.
Windows test build took 20m10s after backend 3m29s; release build 2m32s, archive upload passed
at 13:05:03 UTC, job completed at 13:13:42 UTC. Receipts: `ci-geometry-linux.log` and
`target/fractional-normal-backup/ci-geometry-windows.log`. Publish release archives was
expectedly skipped for the main push (not a tag).
The dated snapshots below preserve the interrupted session's history; use the
[resume completion plan](plans/20261013-resume/plan.md) for the current acceptance ledger.

- Three operator template JSON files (amethyst / box / mandelwhit) are converted, with `.bak`
  files retained, and now pass Rust scene-schema validation. The existing default-feature host
  `WarpBro-a14b1f1310ab8852.exe` ran
  `operator_templates_decode_with_current_scene_schema --ignored --nocapture` with
  `WARPBRO_TEMPLATE_PROBE_DIR=C:/Users/joss1/.warpbro/templates`: 1 passed in 0.06s, loading all
  three templates. No schema change was needed for the CUDA correction. The final published
  test executable `WarpBro-e96469ec1c702b2b` repeats the same filter/environment successfully:
  1 passed in 0.06s, all three templates loaded (`published-geometry-templates.log`).
- EXR current main is `68483c7` (documentation); the released source tag stays at
  `fa8aa3376f78a72794785af347b1a1733d68ff87` (`fa8aa33`), annotated tag object
  `d65355fe9e1492117b540274ee813371e415f744`. Six local bootstrap stages pass. Release CI
  `38382275628` completed successfully (10 jobs, 0 failed);
  v0.1.0 is public/non-draft, published 2026-10-13 11:04:59 UTC with 3 ZIPs/5 wheels. All asset
  sizes/SHA-256, ZIP CRCs, wheel RECORD hashes, and architectures/tags pass. The downloaded
  Windows CLI passes version/help/info and identifies the real PQ PNG as 1920x1080 RGBA.
  Linux/macOS binaries were not executed, wheels not installed, no release-artifact GPU shot.
  Old tag `1671769` is preserved locally as
  `backup/v0.1.0-before-color-resume`; owned duplicate branch/main runs were cancelled because
  the tag covers the same production code.
- Current WarpBro/fractal dependency audit (including `55d8134`):
  `python target/fractional-normal-backup/remote-pins.py` passes 35/35 distinct own
  (repo, revision) pairs (`published-geometry-pins.log`). The earlier 34/34 GitHub API result
  is historical; reachable EXR `7db` was retained.
- Geometry work was isolated in `fractal-geometry` and `warp-geometry` on
  `fix/fractional-bulb-march`; the accepted guarded band-margin correction is now integrated
  in the revisions above. The rejected RAW-max-step prototype was not merged. Zero fallback pixels and the observed pick-depth jump
  `2.3983283` -> `2.4022722` disproved the earlier fallback-trigger explanation.

- Independent review verifies all six requested parameters and ABI 5 neutral defaults are
  wired, with no concrete missing connection. Production main `5a496c6` builds in 35.34s
  after a 408s lock wait; the real release executable renders 18 nonconstant 96x64 gallery
  PNGs at 1 spp on RTX 3080 Ti, exit 0. A temporary `WARPBRO_HOME` leaves the operator profile
  untouched. Logs: `published-build.log` and `published-gallery.log`; this is smoke, not a benchmark.
- Historical geometry worktree checks, before final integration: full default suite 314 passed/0 failed/14 ignored, and final guard
  regression 1 passed. Both routes have 6144/6144 primary hits, normal-reference error 0,
  worst depth error 0.0000007264541 below about 0.00000152194 footprint. Nearby cube object 1
  is picked correctly. Final published dependency checks and integration now pass as recorded
  above. CI `38395622388`: Linux and Windows succeeded; earlier `38391295641` was
  auto-cancelled by the new
  main push after its Linux job passed. The cancelled Windows log still shows compilation
  of `nvjitlink-sys` at 12:13:17 UTC and no `Running unittests`; this is not a test hang or failure.

## Historical agent tasks (both originally committed to branches only)

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

## Verified recovery branches and attribution (2026-10-13)

The old snapshots name agents `mandelbulb` and `colorplan`. The inspected commits are authored
by `ssoj13`; Git author metadata cannot reliably identify the individual agent that wrote a
change. Branch receipts below were verified with `git branch -avv` by the coordinator.

- WarpBro `mandelbulb-params` and its remote tip remain at WIP `8e70078`, including the local
  patch, preserved on `backup/limited-agent-20261013`. Main already contains the merged
  parameters and accepted normal correction; that WIP branch is not the production candidate.
- Fractal recovery baseline is `1f0d122`, prior accepted normal revision `f3e55bd`, and
  current integrated main/origin `55d8134`. WarpBro source integration `d67d3fe` is on main. Geometry
  worktree history on `fix/fractional-bulb-march` remains preserved.
- EXR recovery baseline `234f3bb` and the saved binary diff are retained. Local `color-plan`
  is `fa8aa33`, its remote branch is `bbea08c`, current main is `68483c7` (documentation), and
  release tag v0.1.0 remains at source `fa8aa33`.
- Preserve branches from other work, including `acescg`, `diag`, `feat/keys`, `takes`, and
  `mergeofx` work. Do not clean up or delete them as part of this recovery.

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

## Geometry correction and retained limits

The earlier hypothesis attributed the jump (~2% of hit pixels on the bulb preset) to
DE(n) = 0, a fallback normal, and normalisation of w grad DE(n+1) for any w > 0. It proposed
`normalize((1-w) N_n + w N_(n+1))`, or a weighted fallback correction, to make normals
continuous at the integer endpoint. This is a historical proposal, not a verified cause.

The resume measurements found zero fallback pixels and a pick-depth jump from `2.3983283`
to `2.4022722`. They contradict the claimed fallback trigger. The accepted endpoint-normal correction
is `f3e55bd`; strict targeted tests pass for eight endpoint pairs and the neutral golden.
Before the guarded correction, a larger fractional change exposed an unsigned-DE step-budget
limitation: default 4096-step inside 4.001 produced 0 primary hits with unresolved 1;
16384 steps produced 6144 primary hits with unresolved 0.09798177 (possibly secondary rays;
attribution was not established). Those are historical measurements, preceding integrated
geometry verification.
The earlier normal-only suite passed (312 passed, 0 failed, 14 ignored). The RAW-max-step
prototype failed a counterexample and was not merged. Guarded band-margin geometry is now
integrated (`55d8134` / `d67d3fe`) and its published-dependency suite passes (314/0/14).

Endpoint band-margin correction uses the tracer's existing distance-estimate assumption.
Tests establish the documented fixtures, not a global Lipschitz bound for finite-iteration
Mandelbulb DE; no formal/global guarantee is claimed. Shadow `march_visibility` is unchanged.
The existing specialized-world one-ULP radiance failure remains verified as pre-existing via
captured pre-fix PTX; assertions were not relaxed. No native HDR viewport screenshot was
captured. CI `38395622388` completed SUCCESS for Linux and Windows; release publishing was
expectedly skipped on this main push.
