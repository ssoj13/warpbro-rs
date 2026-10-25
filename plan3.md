# CUDA coverage and kernel performance (2026-10-25)

Status: the first CUDA candidate compiled and passed host checks. Subsequent formula and
material-phase source changes are not yet native-verified. GPU parity, no-spill and performance
acceptance remain open; the original Full Bristorbrot kernel has measured register spilling.
The installed WarpBro release remains `46a485a` with HDR 1000; this work is isolated on the
`cuda-extended` branches.

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
229 host-dispatched entry names. The built sm_86 fat binary contains all 234 kernel entries;
complete compiler spill diagnostics and device correctness acceptance remain open. Family
IDs now come from fractal-core::params, including TOTAL_FORMULA_COUNT and the out-of-range
FAMILY_WORLD = u32::MAX sentinel. Fourteen WGSL round sites use canonical f32::round_ties_even;
the first candidate's cuda-oxide 5e151093 maps that intrinsic to round-even lowering. The
installed compiler is now 3a73b5e, as recorded below. Robust trigonometry is retained until
parity and profiling justify bounded fast paths.

The first compiled candidate uses fractal-rs `6d1235d4ba8d038d1c57bb7f8bf50159663e591e`
and WarpBro `f9d33d969408c65b48cb92e4ceb2bd3eb3f73a24`, published only on their WIP
`cuda-extended` branches. Cargo.toml keeps its SSH main source; Cargo.lock records the exact
pushed dependency commit. The sibling fractal-rs lockfile is not being copied into the app.
That baseline used the fbb7df CUDA dependency graph; the current candidate's pins are recorded
below. Build and device results must be recorded before promoting an accepted release.

Later fractal-rs source commits `411d8216299f2d00f5efb94bc734355b0372fa32` (polar angles
and spherical-seed lookup) and `761ae026898fb5e8a15ffd02b4907155b455346c` (Full material
phase isolation), followed by conservative cone-start commit
`5a21813e0f8ae8f4d49a423d56564f84e5629dfd`, are pushed on `origin/cuda-extended`.
They have not passed a native rebuild or GPU acceptance. WarpBro's current lockfile selects
5a21813e for all four fractal crates and 3a73b5e for all 13 CUDA crates. Offline Cargo metadata
reconciled only cuda-bindings' dependency list: remove serde_json and add bindgen, prettyplease,
proc-macro2, quote and syn. No package/version was added or removed, and no unrelated dependency
was updated.

## Verification checkpoint (2026-10-25)

The machine-locked release test build completed in 89m 08s. Host tests passed: 365 WarpBro
checks and four app-control checks, with no failures. Strict Clippy passed. The subsequent
release executable build passed in 59.05s. These receipts establish compilation and host
regressions, not CUDA rendering correctness. Build receipt:
C:/Users/joss1/.filesystem-mcp-rs/tmp/run_command_1792920260121_2aa17379-3936-4578-bb29-b598a05798e1_stdout.log
(and its stderr log); Clippy receipt:
C:/Users/joss1/.filesystem-mcp-rs/tmp/run_command_1792920194262_1c070b1b-48fe-4130-ade4-e107382d8bf0_stdout.log.

GPU results for that candidate:

- Half-cell/half-count round-to-even checks passed. Noniterating families' independence from
  iteration count passed on both backends.
- Strict DE/trap parity failed and remains under investigation. Thresholds have not been
  widened to accept those failures. The focused orbit/axis diagnostics passed their finite-output
  assertions; those diagnostic passes are not strict parity acceptance.
- All 240 raw radiance/albedo/normal plane comparisons for specialized versus dynamic CUDA
  worlds were bit-identical: 20 families, four shading modes and three planes. The test still
  failed because family 12 rendered no objects; identical empty images are not acceptance.
- Cross-backend primary guides failed for family 12's empty preset, family 15's default
  mismatch share of 0.28271484, and family 26's modified-case coverage of 0.00244140625.
- The coordinator stopped the baseline mixed-family WGPU run during cold compilation after
  the process reached approximately 18 GB resident memory. The pending shared formula fixes
  will invalidate that shader. This stopped run is not a pass; repeat mixed-family acceptance
  after the fixes.

Numerical receipts:
C:/Users/joss1/.filesystem-mcp-rs/tmp/run_command_1792925902691_6117b59f-144a-4f72-aa17-38f19c4e6387_stderr.log;
rendered receipts:
C:/Users/joss1/.filesystem-mcp-rs/tmp/run_command_1792926588748_e9866bc2-67e7-432f-8ced-85addacfacd5_stderr.log
(and its stdout log).

Confirmed findings and pending device regressions:

- Family 12, Polyhedral KIFS: the shared default offset lies outside its folding cone and
  produces an empty shape. Folding that golden-ratio vertex raises rendered coverage from zero
  to 0.103841 on both backends. The preset source is corrected in WarpBro's working tree;
  rebuilt regression acceptance remains open.
- Families 13 and 22: polar `acos/asin` calculations amplify normalization error. Stable
  `atan2` source corrections are committed in CUDA and mirrored in the WGSL working tree;
  rebuilt exact-pole and one-ULP-neighbour regression remains open.
- Family 19: symmetric nearest branches have equal mathematical distance but different traps;
  backend rounding selects opposite branches. Distance agreement does not establish a stable
  coloring rule at that tie.
- Family 23, Phyllotaxis: the shared inverse Fibonacci basis can omit the globally nearest
  seed. The corrected basis agrees with an independent exhaustive nearest-seed oracle across
  the investigated counts and queries. Its CUDA source correction is committed and WGSL is
  mirrored in the working tree; rebuilt CUDA/WGPU regression remains pending.
- Cone start: diagnostic pixel `[62,6]` has a center-ray bound entry of 4.5441432547 but a
  jittered-ray entry of 4.5129545580. Reusing the center-ray start skips 0.0311887 and lands
  0.0116106 inside the bound, exceeding the 0.001528 contact band. The conservative beam
  root is 4.4552378831. The shared pure helper and CUDA correction are committed in 5a21813e;
  WGSL mirrors the correction in the working tree. Independent source/math review passed,
  including Naga validation of nine WGSL helpers. The bound derives `rho`/`kappa` from the
  actual pixel-basis Gram matrix and floating-point envelopes `gamma(32N)`/`gamma(16)`, with
  a depth-of-field guard; it does not use an arbitrary factor of 128. Five CPU fixtures still
  await execution, and native/GPU acceptance remains open. The contact diagnostic passed
  finiteness assertions, but picking used a different start path and did not prove the
  rendered-guide condition.

The proposed numerical QA contract is now split explicitly. The nonignored
`cuda_extended_full_de_short_orbits_and_noniterating_traps_match_wgpu` checks full DE for all
cases, traps at iterations 1..3 for iterative families, and full traps for noniterative families.
Family 19 uses an analytic closest-branch set at mathematical ties. The original strict
full-orbit probe was a new, unmeasured contract; it remains as an ignored precision diagnostic.
No numerical tolerance was widened, and no full-orbit color-parity claim follows from this
change. The established 3% guide criterion and the 240 raw CUDA oracle comparisons are
unchanged. The revised contract has not yet passed the rebuilt device suite.

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

### Compiled sm_86 baseline

Resource extraction used the actual fat binary embedded in the built executable, not a fresh
assembly with different flags. All 234 entries were found. `world_full_f9` (Bristorbrot)
reports 255 registers per thread and a 2240-byte stack/local footprint; `world_fast_f9`
reports 255 registers and 1720 bytes. These figures are not a zero-spill certificate and do
not distinguish call/array storage from register spilling.

The baseline binary SHA-256 is
`7a9c1eea8b9e82cb0cad8ebdca2dca248e08dcda14895b3d026005cf3f5dc614`;
its embedded PTX SHA-256 is
`194ac956d82ab6c42395d0d1af5905cb9f5941c0338cdd692d6f721224876dc7`.
Extraction receipts are under `target/cuda-diagnostics/embedded-baseline`.
Nsight Compute captured the actual `world_full_f9` launch successfully in
`target/cuda-diagnostics/baseline-bristor-profile.ncu-repz`: 59.093632 ms for a 256x192,
four-sample batch, a register limit of two blocks per SM and 9.76% active warps. Hardware
counters recorded 307,004,867 local read sectors and 232,045,942 local write sectors. The
first capture did not collect the software spill metrics; missing values are unknown, not zero.
A second InstructionStats/source-counter capture at 64x48 and four samples, using the same
original Full kernel, is saved as
`target/cuda-diagnostics/baseline-bristor-spill-source.ncu-repz`. It reports 2.88 Mbyte of
local-memory register spilling and zero shared-memory spilling. The baseline therefore fails
the no-spill requirement; the new material-phase source still needs compiled/profiled comparison.

The first interleaved five-run throughput baseline was not steady: CUDA times fell from
0.4558 to 0.1946 seconds, and WGPU times from 0.9301 to 0.3781 seconds. Strengthen warm-up
before accepting a throughput ratio; these descending samples are not a maximum-speed result.
The benchmark working tree now adds two seconds of untimed warm-up. Formatting passed;
rebuilt benchmark acceptance has not run. Raw Full-world CUDA baselines for all 20 extended
families are saved under `target/cuda-diagnostics/baseline-all-world-full`, using the original
6d1235d4 dependency/f9d33d9 executable, including family 12's known-empty preset.

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

The isolated compiler-report candidate also passed three finalizer split-budget CPU tests
and ten backend materialization/publication CPU tests. The successful receipts cover exact
assembly identity, bounded concurrency, collision retry, atomic JSON publication and raw
resource evidence. Its actual tiny-PTX assembly smoke then passed on sm_86 with Toolkit
13.3.73 and `--split-compile=4`, using the production assembly/receipt path. Independent checks
validated the PTX/tool hashes, exact flags, entry inventory and raw helper diagnostics. This
validates the tooling on that fixture, not all WarpBro kernels. Compiler commit
`3a73b5eedfb638db601fe6b704b3a57dce4ff5e9` is published on the fork's
`warpbro-resource-report` branch and is now installed through machine-locked
`bootstrap.py doctor --fix`. The first doctor result still accepted the old fbb7df cached
backend DLL; the coordinator caught that before the application build. Official isolated
`cargo oxide setup` then populated the 3a73b5e backend cache. The actual selected DLL SHA-256 is
`e8dc2d7b9994701c3e9863cba32ae3a0812a63098f47583a5274b24783f6aec9`.
Backend selection follows the precise cuda-host/device revision, so the consuming lockfile's
13 CUDA pins were advanced together to 3a73b5e. CLI version/doctor success alone was not used
as backend identity evidence. Receipts:
C:/Users/joss1/.filesystem-mcp-rs/tmp/run_command_1792926636010_780ea698-d543-4581-9620-4899caa585db_stdout.log;
C:/Users/joss1/.filesystem-mcp-rs/tmp/run_command_1792927615501_7047e148-9fa3-40b9-b9c4-48653ecde486_stdout.log.

The portable bootstrap receipt-consumer suite in `tests/test_bootstrap_resources.py` passed
35 tests with one optional real-receipt test skipped; supplying the retained tiny-PTX JSON
receipt passed all 36 in 0.251s. These are CPU-only parser tests and invoke no compiler or GPU.
Receipt:
C:/Users/joss1/.filesystem-mcp-rs/tmp/run_command_1792929299543_2878a27e-2fe9-46b1-b477-eae8c31f7c22_stderr.log.
They do not establish the new kernels' resources or GPU correctness.

Current-candidate formatting passed. The first locked Clippy attempt stopped before compilation
because the lockfile dependency list needed the reconciliation above. The repeat is active;
no current-candidate Clippy pass or post-fix native build is claimed. The five cone CPU fixtures,
full device suite, zero-spill gate, steady throughput measurements and application installation
remain open.

## Existing backlog outside this work

Render pacing and GPU display transfer remain in plan2.md; CUDA OCIO depends on the corresponding
oiio-rs plan. The remaining document error-propagation/operator decisions are in plan1.md.
Revisit those after the CUDA coverage/performance acceptance above; this plan does not mark them
complete. Preserve the unrelated operator-owned BUG1.md codec note.

Next action: finish the current locked Clippy run, execute the cone CPU fixtures, then build
the exact promoted dependency graph and repeat numerical, rendered and mixed-world acceptance.
Compare the material-phase candidate with the saved raw baselines and source-correlated spill
profile before publishing any no-spill or throughput result.
