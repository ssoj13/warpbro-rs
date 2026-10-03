# Plan 2: Measured render optimization

Date: 2026-10-03.
Status: Nsight baseline captured; CPU scene-preparation cache measured and integration checkpoint passed. Final system tests and tuned build passed; end-to-end GPU performance measurements pending.

This is a separate phase following UI commit `9416673` and toolkit `e953b2cc6836db2bb47aa44bb7995d9696636034` (118 host tests, 40 toolkit tests and release build passed). Optimize measured bottlenecks; do not claim a speedup before evidence exists.

## Nonblocking UI invariant

Preserve the current asynchronous renderer architecture and its existing responsiveness measures. Profiling runs offline or in the render worker. Do not introduce GPU/OIDN waits, blocking command submission, renderer locks or expensive preprocessing on the UI thread. Verify responsiveness during rendering after each accepted change.

## Baseline

- [x] Verify the local profiler: NVIDIA Nsight Compute CLI 2026.3.1.0, available through ncu.bat.
- [ ] Record toolchain, driver, GPU, dependency revisions and exact commands.
- [ ] Define representative fixed fractal/material/environment scenes, resolution, SPP, seed, ray settings and denoise settings.
- [ ] Warm up and repeat measurements; record variation and distributions.
- [ ] Measure CPU preparation, GPU kernels, transfers, OIDN and display/upload separately.
- [ ] Preserve baseline images, raw radiance and profiling artifacts.
- [ ] Record UI responsiveness under simultaneous rendering.

## Nsight analysis

- [x] Verify CUDA profiling access and kernel capture: 45 Nsight passes completed.
- [ ] Identify kernels dominating total render time.
- [x] Inspect baseline registers, spills, occupancy, divergence, memory traffic and dependency stalls.
- [ ] Distinguish compute, transfer, synchronization and allocation bottlenecks.
- [ ] Separate profiler replay overhead from production render timing.
- [x] Record kernel baseline findings before choosing changes.

Baseline: RTX 3080 Ti (CC 8.6), Nsight Compute 2026.3.1, Rust stable 1.99 and CUDA 13.3. The old `13b1caa` `fast_bulb` kernel at 320×180 took 27.85 ms under instrumentation: 162 registers/thread, no spills, theoretical/achieved occupancy 25%/16.33%, 7.8 average active warp threads, 1.5% DRAM throughput and 45.7% dependency stalls. Artifacts: [report](../target/verification/render-baseline-13b1caa-ncu.ncu-repz) and [details](../target/verification/render-baseline-ncu-details.out). These measurements describe one instrumented kernel; end-to-end warm timings and before/after correctness remain pending.

## Evidence-based implementation

- [ ] Optimize only hypotheses supported by profiles.
- [ ] Evaluate cached scene uploads, transform/material/environment preparation and reusable GPU/CPU buffers where repeated work is measured.
- [ ] Use explicit invalidation for scene, animation, camera, material, visibility, environment and resolution changes.
- [ ] Evaluate kernel/mathematical changes with numerical error bounds and representative coverage.
- [ ] Keep progressive accumulation, cancellation, previews, export and OIDN cadence correct.
- [ ] Re-profile each meaningful change with identical settings.

The CPU preparation microbenchmark repeated the same two-object scene 10,000 times: uncached 150,542 µs, cached 1,529 µs, with one build and 10,000 cache hits. This measures preparation only; it is not a whole-render/GPU speedup. The integration checkpoint passed 143 tests including GPU tests in 23.02 s ([log](../target/verification/integration-tests.out), success marker 0); the [final system suite](../target/verification/final-system-tests.out) subsequently passed 144 tests in 55.50 s without compiler warnings (`CDX_FINAL_SYSTEM_TEST_EXIT=0`). The final tuned release build subsequently passed in 46.14 s ([log](../target/verification/tuned-final-presets.out)); end-to-end render performance measurements remain pending.

Current work owns renderer/service cache statistics and their validation. The parallel UI phase owns idle-cache and asynchronous scene IO work. Exact cache invalidation, unchanged raw radiance and measured benefit must be verified before marking implementation complete.

## Correctness and report

- [ ] Run render, animation, HDR/EXR environment, material, transform, visibility, export and OIDN regressions.
- [ ] Compare images/raw HDR radiance against the baseline; define tolerances before accepting approximations.
- [ ] Measure responsiveness under heavy rendering, cancellation, scrubbing and denoising.
- [ ] Report before/after timings, variance, image differences, hardware/tool versions and limitations.
- [ ] Retain or revert each change according to measured benefit and correctness.
