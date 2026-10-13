# Mandelbulb animation parameters: progress

Branch `mandelbulb-params` in warpbro-rs and fractal-rs. Not pushed, not merged (lead reviews).
Keep this file current after every finished parameter; delete it when the branch lands.

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

## Tests (run 2026-10-13, all pass)

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

- Fractional iterations jump at integer counts on ~2% of hit pixels (66 of 3366 on the bulb
  preset at n = 4): where the n-iteration estimate is 0 its gradient is 0 (fallback normal),
  and any fraction above 0 adds the (n+1)-th estimate's gradient. Every other pixel is
  continuous (4.999 -> 5: normals 5e-5). Documented in the kernel's `Fraction`.
- Julia mix 0 / 1 and offset 0 take no arithmetic: the former constants bit for bit.

## Remaining

- [ ] Operator templates (~/.warpbro/templates: `"julia": null`, the `/julia` Json attr,
      `/render/iterations` Int64) need the one-off conversion; script in the scratch dir
      (`convert_templates.py`, writes .bak), to run after the lead's review and the merge
      (main still reads the old format).
- [ ] Bump WarpBro's lock to the fractal-rs branch commit when landing (this branch does not
      build against fractal-rs main).

## Build setup

Temporary `[patch."ssh://git@github.com/ssoj13/fractal-rs.git"]` in Cargo.toml (never
committed). Backup of the original Cargo.toml / Cargo.lock:
`C:\Users\joss1\AppData\Local\Temp\claude\C--projects-projects-rust-cg-cglibs-warpbro-rs\7d561394-85ba-41bf-95b3-a7611591945c\scratchpad\backup\`.
WarpBro's branch builds against main fractal-rs only after the lock is bumped to the fractal-rs
branch commit.
