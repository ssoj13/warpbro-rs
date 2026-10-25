# CUDA coverage and kernel performance (2026-10-25)

Status: source ports and host dispatch are implemented; verification is in progress. The new
CUDA code has not yet passed device compilation, parity, resource or performance acceptance.

## Work order and completion criteria

1. [x] Source: port the 20 extended distance estimators (family IDs 8..27) into the shared
   fractal-rs CUDA implementation, using the existing WGSL formulas and packed parameter ABI
   as the behavioural reference. Done when every parameter, orbit trap, inversion and fractional
   iteration behaviour is covered and all 28 families have an explicit CUDA route.
2. [x] Source: extend compile-time family specialization through the fused trace/shade kernels:
   single-object Fast/Full/Direct and homogeneous-family world Fast/Full/Direct, including worlds
   with different materials. Preserve the dynamic world route for genuinely mixed families and
   cover picking. Done when generated entry points and host dispatch agree and no supported
   family is routed to WGPU merely because its CUDA implementation was missing.
3. [>] Verify compiled resources before making a no-spill claim. Done when all expected PTX
   entries have complete diagnostics and every reported old/new entry and helper has zero
   compiler spill stores and loads for each reported architecture. Record local frames separately,
   inspect surviving helper calls and actual loaded-module register/local bytes on the test GPU.
4. [ ] Run numerical and rendered CUDA/WGPU parity for every extended family. Done when the
   agreed deterministic DE/hit/normal/trap checks and rendered-image comparisons pass, including
   representative parameter changes, transformed objects and mixed-family worlds. Any tolerance
   must follow an attributed numerical/noise measurement rather than hide a mismatch.
5. [ ] Tune launch geometry and live ranges with interleaved CUDA/WGPU measurements. Done when
   the accepted variant has reproducible warm GPU timings, unchanged correctness and zero
   register spill traffic; record family, shading path, workload, GPU/toolchain, resource counts
   and the measured tradeoff for every retained optimization.
6. [ ] Complete the machine-locked release build, relevant host/device regressions and real
   application acceptance through REST/EventBus. Update the implementation status and publish
   tested dependency revisions with the consuming application's lockfile.

## Implemented source, awaiting acceptance

All 20 estimator ports are present. The source inventory has 234 generated kernel entries and
229 host-dispatched entry names; these are source counts, not compiled-PTX acceptance. Family
IDs now come from fractal-core::params, including TOTAL_FORMULA_COUNT and the out-of-range
FAMILY_WORLD = u32::MAX sentinel. Fourteen WGSL round sites use canonical f32::round_ties_even;
the pinned cuda-oxide 5e151093 maps that intrinsic to round-even lowering. Robust trigonometry
is retained until parity and profiling justify bounded fast paths.

Root will test the shared changes from the fractal-rs WIP branch cuda-extended: Cargo.toml keeps
its SSH main source and Cargo.lock records the exact pushed commit. The sibling fractal-rs
lockfile has stale, different CUDA pins and is not being copied into the consuming app. WarpBro
retains its single fbb7df CUDA dependency graph and tested compiler setup. Build and device
results must be recorded before promoting this work as an accepted release.

## Architecture decisions

The shared fractal-rs crates own CUDA mathematics and entry points because WarpBro and OFX
consume the same kernels. Extend their existing estimator/tracing abstractions and specialization
macros rather than duplicate a CUDA implementation in the application. The host packs the same
parameter ABI for both backends; WGPU remains a correctness reference and the backend for other
GPU vendors.

A homogeneous family is a compile-time specialization even when its objects have different
materials. A world with different geometric families still needs dynamic family selection.
Keep that distinction explicit so material diversity does not accidentally defeat distance-estimator
specialization.

Fusion retains ray marching, distance estimation, shading and sample accumulation in the existing
kernel path. Optimize the live set and the code actually generated for each family; changing the
appearance, iteration budget or sampling quality is not a performance optimization for this task.

## Resource evidence

The kernel report in bootstrap.py is authoritative for the command-line interface. It writes
per-architecture raw ptxas output, a cubin, and a JSON report with the PTX SHA-256, entry coverage,
reported helper resources and pre-optimization call relationships under target/oxide-ptx/kernels.

A local stack frame is not the same as compiler register spilling: arrays and call ABI storage can
produce a frame even with zero spill counters. The spill-only gate rejects spill loads/stores;
the stronger historical strict gate also rejects local frames. Missing or incomplete entry
diagnostics fail either invocation, because absence of a report cannot establish zero usage.
Helpers optimized away by ptxas have no separate resource figure and must not be assigned an
invented zero. Check all reported functions even when a helper is absent from the parsed call graph.

Do not force a low register cap to manufacture occupancy. Measure the register/occupancy/time
tradeoff and reduce unnecessary live ranges or specialization barriers in the source. A native
sm_86 assembly report and an sm_75 PTX driver JIT are different compiler products: identify which
one was executed and record loaded-module resources. GPU timings must come from completed GPU
work, not asynchronous submit duration.

## Performance acceptance

Use identical evaluated scenes, resolution, sample counts, seeds, materials, ray-march budgets
and display mode for the compared backends. Separate shader/module preparation, tracing,
denoising and colour/display costs. Warm both variants before sampling; alternate their order
across repeated runs to reduce thermal and scheduling bias. Record median and variation, not
only the best sample. Use device timing for the kernel and wall time for the user-visible frame.

Nsight Compute replay perturbs execution, so use it to attribute bottlenecks and local-memory
traffic; measure throughput separately without the profiler. The local RTX 3080 Ti establishes
performance only for that device. Offline assembly of other supported architectures establishes
resource counts, not runtime speed on untested GPUs.

## Tooling verification receipt

CPU-only parser/gate verification passed 11 fixtures on 2026-10-25: missing entries/fields,
empty diagnostics, helper-only spills, local frames without spills, repeated/quoted resource
sections and multiline/recursive direct calls. The pre-migration PTX parsed as 32 entries,
81 defined functions and 266 direct call edges; its independently scanned entry set agreed.
This verifies the reporting tool, not the new kernels or their resource usage. Receipt:
C:/Users/joss1/.filesystem-mcp-rs/tmp/run_command_1792917080448_6b5a41cc-4f50-46ab-ae82-f09ff91c8f45_stderr.log.

## Existing backlog outside this work

Render pacing and GPU display transfer remain in plan2.md; CUDA OCIO depends on the corresponding
oiio-rs plan. The remaining document error-propagation/operator decisions are in plan1.md.
Revisit those after the CUDA coverage/performance acceptance above; this plan does not mark them
complete. Preserve the unrelated operator-owned BUG1.md codec note.

Next action: run the machine-locked WarpBro build on the exact WIP dependency revision, then
apply the resource gate to its resulting PTX before the all-family CUDA/WGPU parity matrix.
