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

## Bit-identity evidence (A/B probe)

Temporary test `render::tests::cuda_tmp_ab_probe` (in src/render.rs, REMOVE before the final
commit) renders every gallery scene (single and world route, 96x64, 2 spp) and compares accum,
albedo and normal guides bit for bit against `ab_baseline.bin` recorded from the pre-change code
(two baseline runs were identical: deterministic).

- First attempt (inversion through a shared closure wrapper): every Mandelbulb differed by ULPs;
  the wrapper changed inlining / FMA contraction.
- Current (inverted route out of line, `inverted_estimate`): all scenes identical except Hybrid
  and "Hybrid · bulb + KIFS" in the world route (normals ~1e-4). Hybrid's fractional code is not
  the cause (reverted it: same diff). Open: find the structure that keeps world Hybrid identical,
  or accept and document the ULP drift.

## Remaining

- [ ] world Hybrid bit-identity (above)
- [ ] permanent `cuda_*` tests: inversion strength 0 ignores centre/radius; mix 0 ignores J;
      fractional continuity (n vs n + 1e-3, n + 0.999 vs n + 1); mix and inversion continuity
- [ ] remove `cuda_tmp_ab_probe` from src/render.rs
- [ ] conversion script for operator templates (~/.warpbro/templates: `"julia": null` and the
      `/julia` Json attr -> vec3 + julia_mix and the new fields, `/render/iterations` Int64 ->
      Float); written, run only after the lead's review (main still reads the old format)
- [ ] clippy (both repos), cargo fmt on changed files only, README / CHANGELOG / PLAN lines
- [ ] restore Cargo.toml / Cargo.lock from the backup

## Build setup

Temporary `[patch."ssh://git@github.com/ssoj13/fractal-rs.git"]` in Cargo.toml (never
committed). Backup of the original Cargo.toml / Cargo.lock:
`C:\Users\joss1\AppData\Local\Temp\claude\C--projects-projects-rust-cg-cglibs-warpbro-rs\7d561394-85ba-41bf-95b3-a7611591945c\scratchpad\backup\`.
WarpBro's branch builds against main fractal-rs only after the lock is bumped to the fractal-rs
branch commit.
