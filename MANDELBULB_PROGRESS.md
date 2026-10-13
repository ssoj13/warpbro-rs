# Mandelbulb animation parameters: progress

The parameters are merged and pushed: `fractal-rs/main` `1f0d122`, `warpbro-rs/main`
`059984d`. The accepted normal correction `f3e55bd` is pushed on fractal-rs main; WarpBro
normal-only `6289cc8` passed and was pushed. Guarded geometry is now integrated and pushed
as fractal `55d8134` / WarpBro `d67d3fe`, with real published-dependency tests passing.
Template Rust-schema validation passes. Final integrated-source CI completed SUCCESS for
both Linux and Windows build jobs.

## Parameters

| # | Parameter | State |
|---|---|---|
| 1 | Julia mix `julia_mix` (0..1), c = mix(p, J, t) + k, dc/dp = 1 - t | done (kernel `orbit_constant`, scene, attributes); the Julia switch is removed, `julia` is a plain vec3 |
| 2 | Fractional iterations (`render.iterations` f32) | done in all 8 families (`Fraction`: estimate and trap after floor(n) and floor(n)+1 iterations, lerped by the fraction; an integer runs the former loop) |
| 3 | Orbit offset `orbit_offset` (vec3), z <- f(z) + c + k | done (shared `orbit_constant`: Mandelbulb, Mandelbox) |
| 4 | Spherical inversion `inversion_strength` / `inversion_center` / `inversion_radius` | done (fractal-core `inversion`: map, stretch g = (1-s) + s rho^2/r^2, DE / g; host clip sphere `inversion::reach`, unbounded inside-out set marched to `INVERSION_FAR_REACH` = 1e4) |
| 5 | Separate theta / phi powers | already exist: `angle_scale` [theta, phi] multiplies the power per angle (P_BULB_THETA_POWER / PHI_POWER); keyable, slider -8..8, hint present. No new control |
| 6 | Orbit-trap colouring params | keyable already (trap_point vec3 -16..16, trap_scale log 0.001..32, trap_axis 0..2); added the missing hard limit trap_scale >= 0 (a negative scale pins the palette at 0) |

## ABI (fractal-core params.rs)

- Appended: `P_ORBIT_OFFSET` (3), `P_INVERT_STRENGTH` (1), `P_INVERT_CENTER` (3),
  `P_INVERT_RADIUS` (1). Zeros are neutral, so ofx-rs (zero-filled `[0.0; P_COUNT]`) renders as
  before.
- `P_JULIA` is now the mix t (0 / 1 = the former switch: no new slot); `P_ITERATIONS` may be
  fractional.
- `WORLD_ABI_VERSION` 4 -> 5 (object rows are longer).

## Bit-identity evidence (A/B, one-off)

A temporary probe (removed) rendered every gallery scene, single and world route, 96x64 2 spp,
and compared accum, albedo and normal guides bit for bit with a baseline recorded from the
pre-change code (two baseline runs identical: deterministic). Final kernel: 34 of 36 routes
bit-identical; Hybrid and "Hybrid · bulb + KIFS" on the world route differ by ULPs (normals
within 2.6e-4, albedo 4e-5, radiance then diverges per path). Cause: NVPTX fuses multiply-adds
per basic block, so the changed dispatch layout moves the fusions (a shared closure wrapper had
changed every Mandelbulb). Lead's decision: not a regression, no layout tricks; documented at
`signed_distance_and_trap` and in fractal-rs AGENTS.md. A cross-build golden would pin the
layout and the driver, so the permanent tests check in-build invariants instead.

## Historical parameter tests (first pass, 2026-10-13)

- fractal-core: 12 (3 new `inversion`: involution, stretch bound, reach holds the set).
- `cuda_inactive_orbit_and_inversion_values_change_nothing`: every gallery scene, both routes,
  bit-identical with the inversion sphere changed at strength 0, J changed at mix 0, orbit
  controls on families without an orbit constant.
- `cuda_animated_formula_controls_are_continuous`: iterations, Julia mix, offset, inversion;
  a 1e-3 step moves normals < 1/10 of the unit step.
- World / UI: `julia_mix_and_orbit_controls_are_keyable_clamped_numbers`,
  `inactive_parameters_follow_what_the_kernel_reads`, hints, slider endpoints, presets
  monotonic, frame cache, per-object iterations, `cuda_pick`, `cuda_march_out_of_steps`.
- clippy -D warnings clean in both repos (fractal-core; warpbro-rs with the local patch).

## Findings

- First-pass measurements found integer-count jumps on ~2% of hit pixels (66 of 3366 on the
  bulb preset at n = 4), with other pixels continuous (4.999 -> 5: normals 5e-5). The earlier
  explanation blamed a zero gradient and fallback normal. Resume measurements found zero
  fallback pixels and a pick-depth jump `2.3983283` -> `2.4022722`, contradicting that trigger.
  The accepted normal correction is `f3e55bd`, with historical targeted checks below.
  The larger fractional-step marching limitation was subsequently addressed by integrated
  guarded margin `55d8134`; the initial fallback-trigger explanation remains unestablished.
- Julia mix 0 / 1 and offset 0 take no arithmetic: the former constants bit for bit.

## Historical accepted normal candidate (2026-10-13)

- Eight strict endpoint pairs pass, with zero normal jumps above 0.25. Shared hits: 3366/3365.
  Maximum normal differences for the representative single/world bulb checks:

  | Step | Single | World |
  | --- | --- | --- |
  | Integer 4 plus tiny increment | 2.947e-6 | 2.921e-6 |
  | 4 -> 4.001 | 0.00307656 | 0.00305181 |
  | 4.999 -> 5 | 0.00307657 | 0.00305165 |
  | 5 -> 5.001 | 0.00015452 | 0.00015014 |

- Inside fixture: 6144 hits, tiny-step maximum 1.1515e-6; captured baseline 0.80643 fails.
- Neutral golden passes (3.05s): 34 exact routes plus the existing Hybrid world tolerance.
- Default CUDA suite: 26 passed, 6 existing ignored (18.82s); fractal-core: 12 passed.
- The separately run existing specialized-world oracle still fails by one ULP at Fast/two
  objects pixel 643, green radiance: delta 2.9802322e-8, relative 8.56848e-8. Captured pre-fix
  PTX reproduces exactly the same vectors; this is a verified pre-existing failure, with
  assertions unchanged.
- Cold JIT: 303.54s. Warm integer frames: 4.5-5.5ms; fractional frames: 6.2-7.5ms, a measured
  35-45% reported ratio in that run. These are fixture-specific timings, not a controlled
  benchmark against the final geometry.
- Larger inside 4.001 geometry is not certified: at 4096 steps, 0 primary hits/unresolved 1;
  at 16384, 6144 primary hits/unresolved 0.09798177. This proves understepping; the remaining
  unresolved share may be secondary rays. Bounded geometry work lives only in separate
  `fractal-geometry`/`warp-geometry` worktrees on `fix/fractional-bulb-march`.
- WarpBro `6289cc8`: pushed main, four Git-pin lines, no local patch, `windows-sys` preserved.
  Full published-candidate suite: 312 passed, 0 failed, 14 ignored (326 total), 366.81s after
  a 3m1s compile; receipt `published-full.log`.
- The separate RAW-max-step geometry prototype has a proven counterexample and is not merged.
  New bounded band-margin investigation remains pending.

## Independent parameter audit and production smoke (2026-10-13)

- Independent source review found all six requested parameters wired, with no concrete missing
  connection or error: `orbit_constant` uses c = (1-t)p + tJ + k, bulb dc/dp = 1-t;
  fractional iterations run the extra iteration and blend without host rounding; orbit offset
  is connected; inversion applies its DE/stretch correction and bypasses at strength 0;
  theta/phi use separate `angle_scale[0/1] * power`; trap point/normal and color scale reach
  world branches and keyable numeric attributes through `resolve_attribute`.
- ABI 5 neutral defaults: Julia mix 0, orbit offset 0, inversion strength 0/radius 1, angles
  [1,1]. Reviewed source: fractal-kernels `fractal_kernels.rs`, fractal-core `params.rs` and
  `inversion.rs`, WarpBro `scene.rs`, `world.rs`, and `world_tests.rs`. No additional build was
  part of that independent source review.
- Original production main `5a496c6`: `cargo oxide build -- --release` passes in 35.34s after
  a 408s lock queue. Receipt: `target/fractional-normal-backup/published-build.log`.
- The real `target/release/WarpBro.exe` gallery command at 96x64, 1 spp passes with exit 0 on
  RTX 3080 Ti: 18 PNGs, each 96x64 with nonconstant RGB, in the temporary
  `warp-production-gallery` directory. `WARPBRO_HOME` points to temporary
  `warp-production-profile`; the operator profile was untouched. Receipt: `published-gallery.log`.
  This is production render smoke; 1-spp ms/spp/inf readings are not a benchmark.
- Earlier CI `38391295641`: Linux passed; it was later auto-cancelled by the geometry push.

## Historical guarded band-margin geometry worktree checks

These checks preceded final integration. The rejected RAW-max-step prototype was not merged;
final guarded-margin integration and production receipts follow below.

- Full candidate default suite: 314 passed, 0 failed, 14 ignored (328 total), optimized compile
  2m53s, runtime 318.93s including cold CUDA.
- Final guard-focused regression: 1 passed in 266.06s including cold loading. Both routes
  produce 6144/6144 primary hits; original-reference normal error is 0. Worst pick-depth error
  is 0.0000007264541, below minimum pixel footprint about 0.00000152194.
- Nearby cube: object 1 is picked at depth 0.0006979555 versus analytic band 0.0006974757;
  footprint 1.5419748e-6. This validates the nearby-object guard for that fixture.
- Warm five-frame ranges for the 96x64 inside fixture:

  | Iterations | Single | World | World + cube |
  | --- | --- | --- | --- |
  | 4.001 | 23.44-24.86ms | 28.83-29.59ms | 29.69-30.34ms |
  | 4.5 | 4.12-4.30ms | 4.64-5.32ms | 7.18-7.95ms |

  The newly valid 4.001 surface replaces a previous zero-hit result, so runtimes are not
  comparable to that baseline or to the earlier normal fixtures. Receipt:
  `final-geometry-timings.log`. Temporary timing instrumentation is awaiting removal.
- Geometry PTX: 15,883,396 bytes versus the normal-only shader's 15,322,376. The final
  focused guard check passed; final revision, integration, and the published-dependency
  gate were pending at this worktree checkpoint and later passed below.

## Integrated geometry and production verification (2026-10-13)

- Source integration revisions: fractal main/origin
  `55d8134f093e6b3a291a0d811e7a07ab1424dd5c`; WarpBro
  `d67d3fed05641bea133a2ce4876e68cbd6dea7de` is on main. Docs-only commits may advance
  WarpBro HEAD without changing the verified source.
- Root independently tested real published Git dependencies with `--locked`, no patch:
  314 passed, 0 failed, 14 ignored (328 total); compile 2m53s, tests 322.31s. Receipt:
  `target/fractional-normal-backup/published-geometry-full.log`.
- Production `build --release --locked`: passed, 35.48s. Production clippy
  `--release --locked --tests -D warnings`: passed, 12.44s. fmt passed.
- Final published test executable `WarpBro-e96469ec1c702b2b`:
  `operator_templates_decode_with_current_scene_schema --ignored --nocapture`, with
  `WARPBRO_TEMPLATE_PROBE_DIR=C:/Users/joss1/.warpbro/templates`: 1 passed in 0.06s, loading
  amethyst/box/mandelwhit. Receipt: `published-geometry-templates.log`; the earlier normal-only
  template probe is retained above as history.
- Real production executable gallery: 18/18 PNGs decode and have nonconstant RGB, 384x216,
  4 spp, RTX 3080 Ti; Mandelbulb image visually inspected. Temporary profile, operator
  profile untouched. Receipts: `published-geometry-gallery.log` and `verification.json`
  in temporary `warp-geometry-gallery`.
- Current dependency audit: `python target/fractional-normal-backup/remote-pins.py` passes
  35/35 distinct own (repo, revision) pairs on the current original WarpBro/fractal locks,
  including `55d8134`. Receipt: `published-geometry-pins.log`; the earlier 34/34 is history.
- WarpBro integration changes four lock pins and two tests/golden only; `windows-sys`
  preserved, Cargo.toml unchanged. Refreshed graph scopes: fractal 5 symbols/5 flows, MEDIUM;
  WarpBro 14 symbols/0 flows across 7 expected files, MEDIUM. Zero flows is not no callers.
- Endpoint band-margin correction uses the tracer's existing distance-estimate assumption.
  Tests establish these fixtures, not a global Lipschitz bound for finite-iteration Mandelbulb
  DE; no formal/global guarantee. Shadow `march_visibility` is unchanged. The pre-existing
  one-ULP specialized-world failure and lack of native HDR screenshot remain recorded.
- Integrated-source CI `38395622388` on `d67d3fe` completed SUCCESS. Linux: 280 passed,
  0 failed, 7 ignored, 41 filtered (21.25s), release build/archive passed
  (`ci-geometry-linux.log`). Windows: all checks passed, 280 passed, 0 failed, 7 ignored,
  41 filtered (35.45s); test build 20m10s after backend 3m29s, release build 2m32s, archive
  upload success 13:05:03 UTC, job completed 13:13:42 UTC. Receipt:
  `target/fractional-normal-backup/ci-geometry-windows.log`. Publish release archives was
  skipped as expected (main push, not tag). Earlier run
  `38391295641` auto-cancelled on the new main push after Linux passed. Its Windows log was
  still compiling `nvjitlink-sys` at 12:13:17 UTC, with no `Running unittests`; no test hang
  or test failure is established.

## Remaining

- [x] Operator templates amethyst / box / mandelwhit converted once (`"julia": null`,
  `/julia` Json, `/render/iterations` Int64 replaced), `.bak` files retained. All three JSON
  files were structurally validated; do not rerun conversion.
- [x] Rust schema: `WarpBro-a14b1f1310ab8852.exe`
  `operator_templates_decode_with_current_scene_schema --ignored --nocapture`, with
  `WARPBRO_TEMPLATE_PROBE_DIR=C:/Users/joss1/.warpbro/templates`: 1 passed in 0.06s; loaded
  amethyst, box, and mandelwhit.
- [x] Parameters and dependency revision integrated in the main revisions above.
- [x] Accepted endpoint-normal correction `f3e55bd` and strict targeted CUDA regressions.
- [x] Earlier normal-only suite 312 passed / 0 failed / 14 ignored; `6289cc8` pushed main.
- [x] Independently audit all six parameters and run production build/gallery smoke.
- [x] Guarded geometry integrated (`55d8134` / `d67d3fe`), published-dependency 314/0/14,
  production build/clippy/fmt and 384x216 4-spp gallery pass.
- [x] Final integrated-source CI `38395622388`: SUCCESS, Linux and Windows build jobs pass;
  release publishing skipped as expected on main push.

## Build setup

The original parameter branch used a temporary
`[patch."ssh://git@github.com/ssoj13/fractal-rs.git"]` in Cargo.toml. A later interrupted WIP
`8e70078` accidentally committed that local patch; it is preserved on
`backup/limited-agent-20261013` and is not a publication candidate. Backup of the original
Cargo.toml / Cargo.lock:
`C:\Users\joss1\AppData\Local\Temp\claude\C--projects-projects-rust-cg-cglibs-warpbro-rs\7d561394-85ba-41bf-95b3-a7611591945c\scratchpad\backup\`.
The parameter dependency revision is already integrated on main. Current continuity work
used a local patch for testing; accepted WarpBro candidate `6289cc8` has no patch, updates
four Git-pin lines, and preserves `windows-sys`.
