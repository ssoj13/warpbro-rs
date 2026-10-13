# Mandelbulb animation parameters: progress

The parameters are merged and pushed: `fractal-rs/main` `1f0d122`, `warpbro-rs/main`
`059984d`. The accepted normal correction `f3e55bd` is pushed on fractal-rs main; WarpBro
`6289cc8` is pushed on main, with its full published-candidate suite passing. Template
Rust-schema validation passes. Keep this record for the separate open geometry investigation.

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
  The accepted normal correction is `f3e55bd`, with strict targeted checks below. A larger
  fractional change still has a separate unsigned-DE marching limitation; the initial
  `Fraction` explanation must not be treated as established.
- Julia mix 0 / 1 and offset 0 take no arithmetic: the former constants bit for bit.

## Accepted normal candidate (2026-10-13)

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
  35-45% overhead in this run.
- Larger inside 4.001 geometry is not certified: at 4096 steps, 0 primary hits/unresolved 1;
  at 16384, 6144 primary hits/unresolved 0.09798177. This proves understepping; the remaining
  unresolved share may be secondary rays. Bounded geometry work lives only in separate
  `fractal-geometry`/`warp-geometry` worktrees on `fix/fractional-bulb-march`.
- WarpBro `6289cc8`: pushed main, four Git-pin lines, no local patch, `windows-sys` preserved.
  Full published-candidate suite: 312 passed, 0 failed, 14 ignored (326 total), 366.81s after
  a 3m1s compile; receipt `published-full.log`.
- The separate RAW-max-step geometry prototype has a proven counterexample and is not merged.
  New bounded band-margin investigation remains pending.

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
- [x] Final published-candidate suite 312 passed / 0 failed / 14 ignored; `6289cc8` pushed main.
- [ ] Resolve the separately isolated unsigned-DE geometry/step-budget limitation.

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
