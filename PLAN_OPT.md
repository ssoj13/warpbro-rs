# PLAN_OPT — faster convergence (sampling)

> Living plan. Status markers: `[ ]` todo, `[~]` in progress, `[x]` done. Started 2026-10-04.
> Goal: less noise per second, the way V-Ray / Redshift do it, without bias unless a control says so.

## Where we start (verified in code, 2026-10-04)

| Area | Today | Where |
|---|---|---|
| Sample generator | `pcg4d(px, py, sample, dim)`: white noise, every dimension independent | `gpu.rs` `Rng` / `rand` -> `standard_surface_bsdf::sampling::rng` |
| Light sampling | one sun / sky / environment sample per vertex, power-heuristic MIS, light choice by AP1 luminance | `gpu.rs` `sun_select`, `env_sample`, `trace_path` |
| Russian roulette | from bounce 1, `p = max(throughput)` | `trace_path` |
| Adaptive sampling | none: every pixel gets the same SPP, converged sky included | — |
| Firefly control | none (renderer and OIDN) | — |
| Accumulation | per-pixel sample count in `acc[3]`; tonemap divides by it | `gpu.rs` accumulate / `tonemap` |
| Work unit | 8x4 pixel tile per warp (shared march paths) | `gpu.rs` module doc |
| Cost | dominated by distance-estimator marching (primary, shadow, bounce rays) | plan7 Nsight |

The per-pixel count already in `acc[3]` means adaptive sampling needs no new buffer format.

## Measurement (shared by every phase)

- Tools: `WarpBro --world-bench DIR W H SPP --case C --seed S` (raw scene-linear HDR + timings) and
  `tools/convergence.py BASELINE CANDIDATE --output REPORT.json` (linear RGB MSE against independent
  high-SPP references, MSE x time). Method as in plan7: >= 5 independent seeds at low SPP, >= 3 references,
  alternated before/after runs, release build, same frozen scenes.
- Cases: the existing `render_bench::CASES` plus two this plan needs: a **sky-heavy** framing (most pixels
  miss the fractal) and a **cavity** framing (light enters through narrow gaps).
- Report per case: ms/SPP, MSE at 1 / 4 / 16 / 64 SPP, MSE x time, and for adaptive sampling the
  time to reach a fixed MSE.
- Every sampler change must keep the converged image: candidate references equal baseline references
  within their measured noise (unbiased). A biased option is measured with its bias reported.

## Phase 0 — baseline
- [ ] Add the sky-heavy and cavity cases to `render_bench::CASES`.
- [ ] Record baseline numbers (all cases) in this file.

## Phase 1 — low-discrepancy sampler (V-Ray DMC analogue)
- [ ] Device-safe `sampler.rs` (like `path_sampling.rs`): Sobol with hash-based Owen scrambling
      (Burley 2020, "Practical Hash-based Owen Scrambling"), ZSobol-style pixel decorrelation (Morton
      index of the pixel, as in pbrt-v4), padded 2D dimension pairs so each sampling decision
      (lens, pixel, BSDF lobe + direction, light, roulette) gets its own stratified pair.
- [ ] `Rng` / `rand` in `gpu.rs` switch to it: one call site. CPU oracles use the same module.
- [ ] Unit tests: values in [0,1), per-dimension uniformity, 2D stratification of the first pairs,
      different pixels decorrelated.
- [ ] Measure: expected lower MSE at equal SPP mostly at low SPP and low bounce depth; blue-noise-like
      error distribution (better for OIDN). Keep only if unbiased and MSE x time improves.

## Phase 2 — adaptive sampling (V-Ray noise threshold / Redshift adaptive error)
- [ ] Per-pixel error estimate from data the accumulator already has plus one value: accumulate
      luminance and luminance squared (or a second half-sample buffer, Cycles style). Error metric:
      relative standard error of the pixel mean, with a floor for dark pixels.
- [ ] Decide per 8x4 tile (warp-uniform): a tile stops when all its pixels are under the threshold
      after `min_samples`; converged tiles are skipped by the launch, not branched inside a warp.
- [ ] Controls in Render settings: Adaptive on/off, Noise threshold, Min samples; the existing target
      SPP stays the maximum. Export and preview cache use the same rule.
- [ ] Viewport status shows active tiles / converged %. Optional heat-map overlay later.
- [ ] Measure: time to a fixed MSE on every case; check tile borders and dark regions for bias.

## Phase 3 — firefly clamp (Max ray intensity, Clamp secondary)
- [ ] Optional clamp of indirect (bounce >= 1) path contributions by AP1 luminance, off by default,
      labelled as biased. Primary hits and direct light unclamped.
- [ ] Measure energy loss against the unclamped reference and the MSE gain.

## Phase 4 — Russian roulette
- [ ] Continue probability from AP1 luminance of throughput (and of the BSDF albedo estimate),
      with a minimum survival probability and a start depth of 2-3.
- [ ] Measure: unbiased, MSE x time.

## Phase 5 — research gate (only if the numbers ask for it)
- [ ] If the cavity case still dominates after phases 1-2: path guiding (spatial-directional radiance
      cache) or a light cache. Decide from measurements, not by default.
- [ ] If marching dominates time, sampling is not the lever: profile secondary-ray stepping instead.

## Progress log
- 2026-10-04: plan written from the code survey above.
