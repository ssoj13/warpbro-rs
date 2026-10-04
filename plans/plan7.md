# Plan 7: CUDA throughput and convergence

Date: 2026-10-03. Base: WarpBro main `bd7bf07`.
Status: Bounded optimization and convergence phase complete. Accepted base has paired timings/quality, Nsight evidence, repaired CUDA BSDF defect and passing tests. The inconclusive Full helper was rejected; restored shipping release build and raw-output verification pass. Publication is pending.

The objective is a shorter time to a clean image in the current world-node renderer. Samples per second alone is insufficient: a faster kernel with higher variance can take longer to converge. Measure throughput and image error separately, then compare quality at equal elapsed time.

## Previous evidence

[Plan 2](plan2.md) retains the earlier `13b1caa` instrumented `fast_bulb` profile and CPU scene-preparation cache measurements. That profile described a legacy kernel at 320×180, and the cache microbenchmark measured CPU preparation only. Neither establishes the current world renderer's end-to-end performance. The video encoder work in [Plan 6](plan6.md) is also separate from CUDA tracing performance.

## Invariants

Keep the existing render worker, bounded command queues, cancellation and progressive publication. GPU synchronization, denoising, profiling and reference generation stay off the UI thread. Keep UI access nonblocking; preserve scrubbing, navigation, material edits, preview and export behavior.

Do not trade away estimator correctness, fractal detail or authored scene settings to claim a speedup. Retain a change only after its expected numerical effect and measured benefit are understood.

## Completed bounded phase

- [x] Preserve the original release binary, scene snapshots, seeds, toolchain and hardware information.
- [x] Exercise the production world route separately from legacy `--bench`; retain raw HDR, scene JSON and paired production timings.
- [x] Capture baseline/accepted Fast Nsight profiles and separate replay timings from ordinary runs.
- [x] Compare five independent low-SPP seeds against three independent high-SPP references with a reliability gate.
- [x] Validate world specialization numerically, including transformed/multiple objects, secondary rays, guide buffers and progressive accumulation.
- [x] Correct finite-depth MIS and reproduce/repair the pinned Standard Surface support defect without radiance clipping.
- [x] Run full CPU/CUDA tests, responsive worker regressions, cancellation/recovery checks and production release builds.
- [x] Report workload-specific throughput and raw-image error, including the measured Full cost.
- [x] Decide the bounded Standard Surface helper experiment: reject GPU integration because paired results are inconclusive.
- [ ] Publish the final selected revision and confirm the repository state.

The phase profiles and measures the existing asynchronous architecture; it does not implement wavefront scheduling or a general renderer redesign. Dedicated timing decomposition of every CPU/transfer/OIDN stage and a broad multi-object performance matrix are outside the bounded measured result. Multi-object scenes are covered by correctness regressions, not the reported five-case performance table.

## Results

### Workload and hardware

Baseline revision: `bd7bf07`; the unchanged executable is retained as `target/release/WarpBro-baseline.exe`. The current toolchain is Rust stable 1.99, CUDA 13.3 and MSVC 14.51. GPU: RTX 3080 Ti, driver 616.64. Profiler: Nsight Compute 2026.3.1.

The new `--world-bench DIR W H SPP --case CASE --seed N --batch 4` path exercises the production world renderer, including per-batch presentation. Cases are `fast-metal`, `fast-dielectric`, `chrome-mid`, `diffuse-mid` and `opal-mid`. Each run retains raw RGB f32 data, PNG and JSON with the frozen evaluated objects, lights and document. Legacy `--bench` remains a different route.

The preliminary baseline suite at 128×128 and 32 SPP used seeds 0, 1 and 2, with a 1024-SPP seed-101 reference; its artifacts remain under `target/verification/convergence/baseline`. These small seeds were later found to produce correlated permutations and are excluded from the final comparison. The final paired suite covers 128×128 and 256×256 at 32 SPP with five seeds: 65537, 131075, 262151, 524303 and 1048607. Before/after execution order alternates. Independent 1024-SPP reference seeds are 101, 4194305 and 8388611. Binary hashes are retained in `target/verification/convergence/final-manifest.json`.

| Case | Preliminary baseline median ms/SPP |
|---|---:|
| Fast metal | 4.247 |
| Fast dielectric | 4.492 |
| Chrome mid | 4.298 |
| Diffuse mid | 4.806 |
| Opal mid | 4.171 |

These preliminary production timings include the selected batching/presentation policy and are not Nsight replay timings. The final accepted-base comparison appears below.

`tools/convergence.py` assesses the retained raw linear images with NumPy. It uses a common independent candidate high-SPP reference, at least three seeds, and reports MSE, variance and elapsed time. Reference noise, finite sample count and the tested scene coverage limit the interpretation; raw-image error is separate from display-transform or OIDN quality.

### Current Nsight baseline

The representative `fast-metal` world kernel at 256×256/4 SPP required 45 profiler passes. Instrumented duration was 49.05 ms, with 255 registers per thread, 462,029 spill requests, theoretical occupancy 16.67%, achieved occupancy 12.06% and 7.93 average active warp threads. Memory throughput was small. Artifacts: `target/verification/world-fast-baseline.ncu-repz` and its details log.

This supports testing material-model specialization to remove unused branches and reduce register pressure. It does not prove a production speedup. An earlier 128×96 Ember first-frame capture took 3.65 ms under instrumentation (`world-baseline-bd7bf07`); that grid is too small to generalize.

### Retained Fast kernel profile

The guarded `world_fast_bulb` kernel was profiled at the same 256×256/4-SPP, seed-0, batch-4 workload. Instrumented duration fell from 49.05 to 32.90 ms, registers per thread from 255 to 163, and spill requests from 462,029 to zero. Theoretical/achieved occupancy changed from 16.67%/12.06% to 25%/16.34%; average active warp threads changed from 7.93 to 7.02. Cycles were 47,980,438 versus 75,315,796. GPU clocks were not locked: 1.46 GHz during this capture versus 1.54 GHz at baseline. Artifacts: `target/verification/world-fast-bulb-final.ncu-repz` and `world-fast-bulb-final-details.out`.

This is a roughly 33% lower instrumented duration for one kernel/workload, not a blanket render or convergence claim. The final ordinary paired measurements below substantiate the Fast improvement and expose the Full costs.

### Candidate implementation and correctness

Prepared world state caches the material-model classification. Homogeneous Fast scenes use a specialized material path; qualifying single Fast Mandelbulbs also use a guarded DE specialization. Full and Mixed scenes retain the generic world kernel. The separate Full specialization was removed after it failed the exact-output oracle; the oracle was not loosened. Raw radiance and primary-hit guide comparisons cover retained specialization against the generic route, including sheared transforms, multiple objects, secondary bounces and progressive accumulation. Single-object Fast float components use the documented bound `3e-5 + 3e-5 × abs(reference)`; observed maximum absolute/relative differences were 1.93e-5/1.72e-5. Fast multiple-object and Full one/two-object comparisons require exact floats. Alpha/sample counts and zero-RGB masks are exact across all routes.

The accepted finite-depth correction permits the final BSDF ray to reach the environment with its original complementary MIS weight. A surface hit beyond the permitted depth receives neither shading nor emission: the shared `allows_surface_vertex` check occurs after the miss/environment branch and before surface shading. An analytic Lambert check verifies that the NEE contribution ln(17)/16 plus the BSDF contribution 1 − ln(17)/16 totals 1. The experimental NEE-only terminal estimator was rejected because it performed poorly for glossy surfaces; its preliminary assessment is retained as `quality-nee-only.json`.

Fast shading uses the shared visible-normal GGX sampling/PDF pair; pure metals use specular selection probability 1. These estimator changes require the independent-reference quality comparison; specialized-versus-generic comparisons alone cannot establish convergence improvement. The single-Fast-Mandelbulb DE specialization (`world_fast_bulb`) retains the world context, affine transforms, clipping and world lighting. Focused and full regression checks pass; final production measurements are reported below.

The release test suite passed **193 tests**, including GPU tests with `--include-ignored`, after the revised finite-depth correction and guarded Fast specialization. The latest accepted-source full suite completed in 41.74 s (`target/verification/convergence-accepted-tests.out`); the focused oracle suite completed in 46.11 s. Earlier checkpoints remain in `convergence-final-tests.*` and `convergence-tests.*`. Responsive worker regressions passed. Existing worker scheduling is unchanged. A pre-dependency-fix release build passed in 49.81 s. The final CUDA suite with the pinned BSDF repair passed **193 tests in 79.43 s**, after 52.73 s compilation (`target/verification/convergence-bsdf-final-tests.*`). The final release build passed in 48.89 s (`convergence-final-pipeline.out/.err`). Ordinary paired throughput and quality/time measurements are complete for this validated base. The later helper experiment passed separate checks but was rejected; those checks do not replace the accepted-base suite. The restored shipping release build passed in 1m 03s (`convergence-shipping-build.*`). Shipping Fast metal at 128×128/32 SPP, seed 65537, and Opal at 128×128/1024 SPP, seed 101, are bitwise identical to the accepted final outputs and `opal-post-support-fix`. Opal output is finite, with maximum RGB 8.48473358 and mean 0.28781459. Only publication remains pending.

### Rejected hypotheses and reference limits

The first material-only specialization, profiled with the rejected NEE-only terminal estimator, lowered register use from 255 to 168 and raised theoretical occupancy from 16.67% to 25%. However, spill requests increased from approximately 0.46 million to 1.43 million, while instrumented duration increased from 49.05 to 50.88 ms. This profile does not establish an improvement; lower register count alone is not an acceptance criterion. Final production measurements determine which specializations remain.

The original Opal seed-101 reference at 1024 SPP contained a rare outlier with maximum RGB 1,316,556 and mean 38.26, compared with an ordinary mean near 0.287. The outlier was identical in the original baseline and pre-dependency-fix candidate, so it was not introduced by specialization. That historical reference was rejected for convergence claims. The assessor marks a reference unreliable when its independently estimated MSE exceeds 20% of the low-SPP image MSE; final measurements use fresh references after the repair.

An independent CPU reproduction against the actual pinned Standard Surface dependency (`13c757e`) found a grazing f32 support failure: for an above-horizon sample, the local half-vector dot product dot(v,h) can round negative. The GGX density is then rejected while evaluation stays positive; a reproduced sample had weight 9,949,037. A standalone one-million-sample endpoint stress scan with repaired support reduced weights above 1000 from five to zero, with maximum weight 1.276. The CPU investigation initially did not prove the Opal link, because that pixel's traced RNG contained no exact-zero variate. The later isolated CUDA rerender establishes the dependency repair's effect on the actual scene. The repair was published on the exact-old-pin CUDA-compatible branch at render-rs `cd72eb3`, and separately cherry-picked onto render-rs main as `c913657`. The dedicated pin passed 71 default and 71 `cuda-math` tests (one GPU test ignored in each run), with WGSL/Naga validation; main passed 71 default tests. WarpBro now pins `cd72eb3` over GitHub SSH; exactly one locked dependency package changed. An identical CUDA Opal rerender at 128×128/1024 SPP, seed 101, changes only this dependency revision; all earlier integrator changes are present on both sides. Before the repair, maximum RGB was 1,316,556 and mean 38.263462; pixel (35,55) was [207476.75, 342546.59375, 1316556]. After the repair, maximum RGB was 8.484734 and mean 0.2878146; that pixel was [0.04481322, 0.04479841, 0.27129424]. Artifacts are retained as `opal-pre-support-fix` and `opal-post-support-fix`, with `convergence-final-pipeline.out`. This causally isolates the BSDF dependency fix; no radiance clamp was added. The completed five-case quality/performance comparison is reported below.

### Final paired production results for the validated base

At 256×256, 32 SPP and five independent seeds, before/after runs alternate order and use the same frozen scenes and batch-4 presentation policy. `target/verification/convergence/timing-paired-report.json` retains the per-run data. The ratio column is the median paired baseline/new ratio, so it need not equal the ratio of the two separate medians.

| Case | Baseline median ms/SPP | New median ms/SPP | Paired throughput ratio |
|---|---:|---:|---:|
| Fast metal | 10.9387 | 6.7927 | 1.609× |
| Fast dielectric | 10.9686 | 6.7299 | 1.618× |
| Chrome mid | 9.0512 | 9.4819 | 0.956× |
| Diffuse mid | 11.2526 | 11.9687 | 0.942× |
| Opal mid | 8.5860 | 8.7109 | 0.986× |

Fast cases improve by approximately 61% in samples/time on these workloads. Full material cases are slightly slower: the corrected finite-depth estimator actually traces the final environment ray instead of discarding it. These results do not establish a speedup for every world or material.

At 128×128/32 SPP, all five final references pass the reliability gate; independently estimated reference noise contributes approximately 1% of the measured low-SPP error. `quality-final.json` records the full assessment.

| Case | Baseline linear RGB MSE | New linear RGB MSE | Estimated MSE×time efficiency gain |
|---|---:|---:|---:|
| Fast metal | 0.012940688 | 0.011435995 | 1.625× |
| Fast dielectric | 0.000290187 | 0.000284789 | 1.524× |
| Chrome mid | 0.001699349 | 0.001697601 | 0.946× |
| Diffuse mid | 0.000261949 | 0.000263724 | 0.936× |
| Opal mid | 0.002954307 | 0.002952560 | 0.964× |

Metal MSE is about 11.6% lower at fixed SPP. The dielectric reduction is small (about 1.9%); Full-case errors are approximately unchanged. MSE×time is an efficiency estimate, not a directly measured stopping time to a chosen visual threshold. The lower noise depends on the tested workload and finite independent sample set.

A post-dependency-fix Nsight capture (`world-fast-bulb-accepted.ncu-repz` and its details log) measured 29.98 ms at 1.55 GHz versus the baseline 49.05 ms at 1.54 GHz, with 46,538,518 versus 75,315,796 cycles. Registers remain 163 and spill requests zero. It confirms the earlier Fast profile finding; production conclusions use the paired uninstrumented runs above.

### Rejected bounded Full helper

A shared Standard Surface NEE evaluation/PDF helper avoids repeated layer preparation while retaining the same shading model. Its functional API experiment remains published at `56d290f` on render-rs `perf/short-lived-nee-preparation`; it was not promoted to main and is not active in WarpBro. Shipping source and the dependency pin were restored to the accepted `cd72eb3` base.

The helper passed 73 CPU tests per backend, then 193 WarpBro CPU/CUDA tests in 77.37 s after 53.10 s compilation, plus a 49.33 s release build. Fifty raw comparisons (128×128 and 256×256, five cases, five seeds, 32 SPP) and 28 additional 512×512 comparisons are bitwise identical to the validated base: **78 comparisons total**. Results are in `prepared-raw-comparison.json` and `prepared-512-raw.json`.

Against the validated base at 256×256, median paired ratios are Chrome 1.01562, Diffuse 1.00909 and Opal 1.00260, with Fast controls 0.99632 (metal) and 0.99092 (dielectric). At 128×128 the corresponding ratios are 1.01975, 1.00526, 1.00710, 0.99166 and 0.98816. These small differences and noisy outlier pairs do not establish a blanket gain.

Chrome Nsight captures at 256×256/4 SPP, seed 65537, show corrected-base 44.35 ms at 1.51 GHz / 66,939,998 cycles versus helper 41.64 ms at 1.55 GHz / 64,685,268 cycles. Both use 255 registers and have 404,014 spill requests. Artifacts are `world-full-corrected-base` and `world-full-prepared-candidate`. The final 512×512/32-SPP comparison used seven seeds, three Full cases and a Fast dielectric control (`convergence-prepared-512.out`). Results are:

| Case | Paired median ratio | Geometric mean ratio | Log standard error |
|---|---:|---:|---:|
| Chrome | 1.01463 | 1.02745 | 0.01138 |
| Diffuse | 1.01245 | 1.01096 | 0.00613 |
| Opal | 1.00716 | 1.00245 | 0.01687 |
| Fast dielectric control | 0.99634 | 0.99068 | 0.00963 |

Broad overlap, outliers and inconsistent Full improvements do not demonstrate a coherent application gain. GPU integration was therefore rejected despite correct raw outputs and the isolated profile improvement. The accepted base is retained; this experiment adds no speedup claim.

### Research and iteration

The working loop is profile → formulate a narrow hypothesis → reproduce its numerical behavior → test exact outputs or estimator expectations → repeat paired production and quality measurements. Rejected candidates and noisy references remain visible so the conclusions can be checked.

Primary references used for the numerical and architectural review:

- [PBRT: A Better Path Tracer](https://pbr-book.org/4ed/Light_Transport_I_Surface_Reflection/A_Better_Path_Tracer): complementary MIS and environment evaluation before the surface-depth guard.
- [Sampling Visible GGX Normals with Spherical Caps](https://arxiv.org/abs/2306.05044): visible-normal sampling and its probability density.
- [NVIDIA Compute Triage](https://docs.nvidia.com/nsight-compute/ComputeTriage/): relate occupancy and spills to measured bottlenecks; occupancy alone is not the objective.
- [PBRT: Monte Carlo Basics](https://pbr-book.org/4ed/Monte_Carlo_Integration/Monte_Carlo_Basics): estimator variance and computational cost.
- [Cycles kernel scheduling](https://developer.blender.org/docs/features/cycles/kernel_scheduling/) and [PBRT: Mapping Path Tracing to the GPU](https://pbr-book.org/4ed/Wavefront_Rendering_on_GPUs/Mapping_Path_Tracing_to_the_GPU): wavefront scheduling benefits and queue/launch costs. A wavefront renderer has not been implemented in this phase.
