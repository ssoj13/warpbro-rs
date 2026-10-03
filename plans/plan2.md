# Plan 2: Measured render optimization

Date: 2026-10-03.
Status: profiler availability verified; baseline and optimization pending.

This is a separate task after the unified UI work. Optimize measured bottlenecks; do not claim a speedup before evidence exists.

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

- [ ] Verify CUDA profiling access and kernel capture on this GPU/driver.
- [ ] Identify kernels dominating total render time.
- [ ] Inspect registers, spills, occupancy, divergence, memory traffic/access patterns and execution limits.
- [ ] Distinguish compute, transfer, synchronization and allocation bottlenecks.
- [ ] Separate profiler replay overhead from production render timing.
- [ ] Record findings before choosing changes.

## Evidence-based implementation

- [ ] Optimize only hypotheses supported by profiles.
- [ ] Evaluate cached scene uploads, transform/material/environment preparation and reusable GPU/CPU buffers where repeated work is measured.
- [ ] Use explicit invalidation for scene, animation, camera, material, visibility, environment and resolution changes.
- [ ] Evaluate kernel/mathematical changes with numerical error bounds and representative coverage.
- [ ] Keep progressive accumulation, cancellation, previews, export and OIDN cadence correct.
- [ ] Re-profile each meaningful change with identical settings.

## Correctness and report

- [ ] Run render, animation, HDR/EXR environment, material, transform, visibility, export and OIDN regressions.
- [ ] Compare images/raw HDR radiance against the baseline; define tolerances before accepting approximations.
- [ ] Measure responsiveness under heavy rendering, cancellation, scrubbing and denoising.
- [ ] Report before/after timings, variance, image differences, hardware/tool versions and limitations.
- [ ] Retain or revert each change according to measured benefit and correctness.
