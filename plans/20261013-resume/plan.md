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

- Normal correction `fractal-rs/main` `f3e55bd` is accepted and pushed. Strict eight-pair and
  inside tiny-step checks pass; neutral golden, CUDA default 26 passed/6 ignored, and core
  12 passed. Larger unsigned-DE geometry/step-budget work remains isolated on
  `fix/fractional-bulb-march` in separate worktrees and is not in accepted main.
- WarpBro `6289cc8` is pushed on main: four Git-pin lines, no patch, `windows-sys` preserved.
  The full published-candidate suite passes: 312 passed, 0 failed, 14 ignored (326 total),
  366.81s after 3m1s compile (`published-full.log`). The separate RAW-max-step geometry
  prototype has a counterexample, remains unmerged, and is superseded by an open bounded
  band-margin investigation.
- Operator templates amethyst/box/mandelwhit pass the current Rust schema: existing
  `WarpBro-a14b1f1310ab8852.exe operator_templates_decode_with_current_scene_schema --ignored
  --nocapture`, with `WARPBRO_TEMPLATE_PROBE_DIR=C:/Users/joss1/.warpbro/templates`, 1 passed
  in 0.06s. Converted JSON and `.bak` files retained.
- All 34 distinct WarpBro/fractal own Git pins are reachable (34/34 GitHub API checks);
  existing reachable EXR `7db` is retained.
- EXR's six local bootstrap stages pass. Main/tag points to
  `fa8aa3376f78a72794785af347b1a1733d68ff87`; annotated tag object
  `d65355fe9e1492117b540274ee813371e415f744`. Old tag `1671769` is backed up locally as
  `backup/v0.1.0-before-color-resume`. Tag CI `38382275628`: all 10 jobs succeeded, 0 failed.
  The public/non-draft release was published at 2026-10-13 11:04:59 UTC with 3 ZIPs/5 wheels;
  downloaded sizes/SHA-256, ZIP CRCs, wheel RECORD hashes, architecture/tags, and Windows
  CLI/PQ-info smoke pass. Linux/macOS execution, wheel installation, and release-artifact GPU
  shots were not performed. Owned duplicate branch/main CI runs were cancelled because the
  tag covers the same production code.

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
| Integer-endpoint normals | Complete; `f3e55bd` and WarpBro `6289cc8` pushed | Eight strict pairs, inside tiny-step, neutral golden, final published-candidate suite |
| EXR two-stage production path | Six local gates pass; main/tag integrated | `fa8aa33`; final viewer 337 passed, 0 failed, 23 ignored; real PQ exact |
| PQ output parity | Targeted EXR checks pass; native viewport comparison pending | Real PQ worst/mean code error 0, CPU/GPU twins, visually inspected SDR proof; no native HDR viewport screenshot |
| Operator templates | Complete | Three converted JSONs load through current Rust schema; 1 test passed in 0.06s |
| Dependency reachability | Complete | 34/34 own Git pins reachable; no patch and `windows-sys` preserved in `6289cc8` |
| Final WarpBro gates and integration | Complete | 312 passed, 0 failed, 14 ignored; `6289cc8` pushed main |
| Separate unsigned-DE geometry | Open | RAW-max-step prototype rejected by counterexample; bounded band-margin investigation unmerged |
| EXR release replacement | Published and verified | CI `38382275628`: 10 successful jobs, 3 ZIPs/5 wheels integrity verified, Windows CLI smoke passes |

No pending item is complete until its evidence is recorded. Existing historical test results
in the progress files are not acceptance evidence for the new candidates.
