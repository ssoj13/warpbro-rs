# Resume completion plan

Updated: 2026-10-13. Scope: finish the interrupted Mandelbulb and EXR color work across
`fractal-rs`, `warpbro-rs`, and `exr-rs`. This is an execution checkpoint; the existing
[WarpBro plan](../../PLAN.md) remains the project backlog, and [HANDOFF](../../HANDOFF.md)
contains the original requirements and recovery history.

## Verified starting state (historical resume checkpoint)

- Coordinator verified remote `fractal-rs/main` at `1f0d122` and `warpbro-rs/main` at
  `059984d`: the animation parameters are already merged. WarpBro's current local HEAD is
  `059984d` after the coordinator's fast-forward.
- WarpBro WIP `8e70078` accidentally includes a local dependency patch. It is preserved on
  `backup/limited-agent-20261013`; retain that recovery branch and land only reviewed changes.
- Coordinator verified `exr-rs/main` at `197417f`: the first-pass color fixes landed, while the
  production two-stage decode/display implementation remains unfinished. Local `color-plan`
  is `234f3bb`; its 16 dirty files were saved to a temporary diff before further work.
- The integer-iteration normal discontinuity remains the first CUDA task. EXR effects,
  pyramid levels, scopes, inspector, and exposure must consume scene-linear pixels.
- The missing EXR GitNexus skill references are recorded in the shared `BUG3.md` report.
  Use available WarpBro skill guidance with explicit repository binding and callable tool
  schemas; obtain impact before symbol edits and refresh the changed-source index afterward.

## Current verified checkpoint

- Earlier normal correction `f3e55bd` was accepted and pushed. Strict eight-pair and
  inside tiny-step checks pass; neutral golden, CUDA default 26 passed/6 ignored, and core
  12 passed. Geometry work was isolated on `fix/fractional-bulb-march`; the later accepted
  guarded margin is now integrated, while rejected RAW-max-step stays unmerged.
- WarpBro `6289cc8` is pushed on main: four Git-pin lines, no patch, `windows-sys` preserved.
  The full published-candidate suite passes: 312 passed, 0 failed, 14 ignored (326 total),
  366.81s after 3m1s compile (`published-full.log`). The separate RAW-max-step geometry
  prototype failed a counterexample and remained unmerged; the later bounded band-margin
  correction is now integrated as recorded below.
- Operator templates amethyst/box/mandelwhit pass the current Rust schema: existing
  `WarpBro-a14b1f1310ab8852.exe operator_templates_decode_with_current_scene_schema --ignored
  --nocapture`, with `WARPBRO_TEMPLATE_PROBE_DIR=C:/Users/joss1/.warpbro/templates`, 1 passed
  in 0.06s. Converted JSON and `.bak` files retained.
- Current original WarpBro/fractal lockfile audit including `55d8134`:
  `python target/fractional-normal-backup/remote-pins.py` passes 35/35 distinct own
  (repo, revision) pairs (`published-geometry-pins.log`). Earlier 34/34 GitHub API checks
  remain historical; reachable EXR `7db` is retained.
- EXR's six local bootstrap stages pass. Current main is `68483c7` (documentation), while
  release tag v0.1.0 points to `fa8aa3376f78a72794785af347b1a1733d68ff87`; annotated tag object
  `d65355fe9e1492117b540274ee813371e415f744`. Old tag `1671769` is backed up locally as
  `backup/v0.1.0-before-color-resume`. Tag CI `38382275628`: all 10 jobs succeeded, 0 failed.
  The public/non-draft release was published at 2026-10-13 11:04:59 UTC with 3 ZIPs/5 wheels;
  downloaded sizes/SHA-256, ZIP CRCs, wheel RECORD hashes, architecture/tags, and Windows
  CLI/PQ-info smoke pass. Linux/macOS execution, wheel installation, and release-artifact GPU
  shots were not performed. Owned duplicate branch/main CI runs were cancelled because the
  tag covers the same production code.

- Independent parameter audit verifies all six controls and ABI 5 neutral defaults wired,
  with no concrete missing connection. Production main `5a496c6` builds successfully
  (35.34s after 408s queue) and renders 18 nonconstant 96x64 gallery PNGs at 1 spp using
  a temporary profile. This is render smoke, not a performance benchmark.
- Historical guarded band-margin geometry worktree checks: 314 passed/0 failed/14 ignored (328 total),
  318.93s including cold CUDA; final guard-focused test 1 passed in 266.06s. Both routes
  6144/6144 primary hits, normal-reference error 0, worst depth error 0.0000007264541 below
  minimum footprint about 0.00000152194; nearby cube guard passes. These preceded the final
  integrated published-dependency verification below.
- Recovery branch and author attribution is recorded in HANDOFF: `ssoj13` commit authors do
  not reliably identify agents. Preserve WIP backups, saved EXR binary diff, and unrelated
  branches; no branch cleanup is part of this recovery.

- Geometry source integration revisions: fractal main/origin
  `55d8134f093e6b3a291a0d811e7a07ab1424dd5c`; WarpBro
  `d67d3fed05641bea133a2ce4876e68cbd6dea7de` is on main. Docs-only checkpoints can advance
  WarpBro HEAD without changing the source verified by CI. Real Git dependency
  `--locked` suite, no patch: 314 passed/0 failed/14 ignored (328 total), compile 2m53s,
  tests 322.31s (`published-geometry-full.log`). Production build 35.48s, clippy 12.44s, fmt
  pass; 18/18 nonconstant, decoded 384x216 4-spp gallery PNGs pass on RTX 3080 Ti, with the
  Mandelbulb image visually inspected and isolated temporary profile.
- Final published template-schema probe `WarpBro-e96469ec1c702b2b` repeats the same filter
  and `WARPBRO_TEMPLATE_PROBE_DIR` successfully: 1 passed in 0.06s, all three templates
  (`published-geometry-templates.log`). Prior normal-only receipts remain historical.
- Endpoint band-margin correction uses the tracer's existing distance-estimate assumption;
  fixture tests do not prove a global Lipschitz bound for finite-iteration Mandelbulb DE.
  Shadow `march_visibility` is unchanged; pre-existing one-ULP oracle failure and missing
  native HDR screenshot remain explicit. CI `38395622388` completed SUCCESS: both build jobs
  pass, Linux/Windows each 280 passed, 0 failed, 7 ignored, 41 filtered (21.25s/35.45s),
  release builds/archive uploads passed. Windows test build 20m10s after backend 3m29s,
  release build 2m32s; upload success 13:05:03 UTC, job complete 13:13:42 UTC. Receipts:
  `ci-geometry-linux.log` and `target/fractional-normal-backup/ci-geometry-windows.log`.
  Publish release archives was skipped as expected for main push (not tag). Earlier
  `38391295641` auto-cancelled on the new main push after Linux passed. Its Windows job
  was still compiling `nvjitlink-sys` at 12:13:17 UTC, before unit tests; no test hang/failure.

## Completion sequence

1. **Preserve and review the recovered work.** Confirm branch tips, backups, exact dirty-file
   inventories, and dependency diffs. Keep each worker's file ownership explicit. Serialize
   CUDA builds and final gates; do not mix another worker's partial edits into acceptance.
2. **Fix fractional-iteration endpoint normals.** Make normals continuous from integer `n`
   toward `n + epsilon`, including the inside-set fixture. The earlier fallback-trigger
   hypothesis was contradicted by zero fallback pixels; keep cause claims tied to diagnostics.
   Extend the CUDA continuity regression, preserve neutral/integer behavior, and review both
   specialized and world routes against the shared kernel contract.
3. **Complete EXR decode and display.** Implement one source-space-to-scene-linear decode
   stage and one scene-linear-to-display/view/look stage, with matching CPU/GPU behavior.
   Preserve source tags for scene/video view and look selection; cache processors by the
   complete transform inputs. Handle EDR, unsupported chromaticities, non-invertible view
   transforms, and relative-display white luminance through explicit contracts and errors.
4. **Prove the color pipeline.** Cover PQ and sRGB decode/display identity, linear input to
   effects and pyramid reduction, per-source-class looks, whiteLuminance round-trip, and
   invalid transform cases. Review the known PQ PNG against WarpBro's viewport; record the
   actual image and numeric evidence before claiming visual parity.
5. **Run final gates on frozen candidates.** Use the repository's CUDA/bootstrap paths and
   focused EXR tests, then the required build/lint gates. Record commands, exact candidate
   revisions, outcomes, skips, and limitations. Obtain independent review and GitNexus
   change-scope evidence, checking the native Git inventory for false-positive paths.
6. **Integrate dependencies and release.** Land reviewed fractal work first, update WarpBro's
   dependency revision explicitly, preserve `windows-sys` resolution, and remove temporary
   local patches from publication candidates. Integrate EXR only after the two-stage gates
   pass; verify the requested same-version release replacement, tag, CI, and assets. Update
   HANDOFF, MANDELBULB_PROGRESS, COLOR_PLAN, and changelog from completed receipts.

## Acceptance ledger

| Item | Current state | Required evidence |
| --- | --- | --- |
| Recovered WIP preserved | Coordinator receipt; WarpBro backup branch confirmed | Exact backup refs and EXR diff retained |
| Integer-endpoint normals | Complete; incorporated in fractal `55d8134` / WarpBro `d67d3fe` | Eight strict pairs, inside tiny-step, neutral golden, final published-candidate suite |
| EXR two-stage production path | Six local gates pass; source tag released | Current docs main `68483c7`, release source tag `fa8aa33`; viewer 337 passed, 0 failed, 23 ignored |
| PQ output parity | Targeted EXR checks pass; native viewport comparison pending | Real PQ worst/mean code error 0, CPU/GPU twins, visually inspected SDR proof; no native HDR viewport screenshot |
| Operator templates | Complete on final published executable | Three converted JSONs load; repeated final schema probe 1 passed in 0.06s |
| Dependency reachability | Current audit complete | 35/35 own repo/revision pairs reachable, including `55d8134`; `d67d3fe` no patch / `windows-sys` preserved |
| Final WarpBro local gates and integration | Complete | Final published dependency 314/0/14; `d67d3fe` pushed; build/clippy/fmt and 4-spp gallery pass |
| Independent parameter audit | Complete | All six controls and ABI 5 defaults wired; no concrete missing connection |
| Current production build/gallery smoke | Complete on `d67d3fe` | Build/clippy/fmt pass; 18/18 valid nonconstant 384x216 PNGs, 4 spp, isolated profile |
| Historical production smoke | Earlier `5a496c6` passed | 18 nonconstant 96x64 PNGs at 1 spp; render smoke, not a benchmark |
| Guarded unsigned-DE geometry | Integrated | `55d8134` / `d67d3fe`; published 314/0/14 and focused guard fixtures pass |
| Final integrated-source CI | Complete: SUCCESS | `38395622388` on source `d67d3fe`; Linux/Windows 280/0/7, release builds/archives pass; publish skipped for main push |
| EXR release replacement | Published and verified | CI `38382275628`: 10 successful jobs, 3 ZIPs/5 wheels integrity verified, Windows CLI smoke passes |

No pending item is complete until its evidence is recorded. Existing historical test results
in the progress files are not acceptance evidence for the new candidates.
