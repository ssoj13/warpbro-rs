# Diagnose CUDA startup before changing the loader

## Current result

The rebuilt release executable renders successfully. The grey startup viewport is consistent with waiting for the CUDA worker to finish initialization; it does not by itself establish a broken tracing kernel.

The host now logs context creation and embedded-module loading separately in `Gpu::new`. Both remain in the existing CUDA worker for interactive use.

| Run                                      |                       CUDA context |        Embedded kernels | Whole headless check | Load caveat                                                 |
| ---------------------------------------- | ---------------------------------: | ----------------------: | -------------------: | ----------------------------------------------------------- |
| Initial executable                       |                          Not split |               Not split |            114.085 s | Contended machine; CPU saturation was subsequently observed |
| Rebuilt executable                       | Logging filter excluded the module | Not captured separately |            104.152 s | Not an idle-machine baseline                                |
| Retry, 2026-10-03 local / 2026-10-04 UTC |                        122.5224 ms |           152.3668334 s |            156.041 s | CPU rose from 33% to 76%; concurrent atlas_chans_bench      |

The retry started with GPU utilization at 34% and 8,739 MiB allocated. All three checks completed successfully. The measured delay is inside embedded-kernel loading. Its split between compilation, linking, and driver loading has not been measured yet.

These numbers do not establish the normal startup time, a GPU tracing regression, or a speedup. The tiny 32×32, 1-SPP legacy benchmark is a startup smoke check; its throughput table is not a production-world convergence benchmark.

## Reproduce with visible logging

Build through the project bootstrap:

```powershell
python -u bootstrap.py b
$env:RUST_LOG = "info"
.\target\release\WarpBro.exe --bench 32 32 1
```

The module path is `WarpBro::render`. A lowercase `warpbro=info` filter does not select those messages. The log includes `context ready in` and `kernels ready in`.

Record CPU/GPU utilization before and during the run. Avoid overlapping builds, profilers, and benchmarks for the idle baseline. Keep executable/dependency revisions and distinguish first-process and repeated-process starts. Do not terminate other applications just to obtain a clean result.

Local retry artifacts are `target/startup-retry.stdout.log` and `target/startup-retry.stderr.log`; they are generated files, not repository fixtures.

## Pending work

- Measure compilation, linking, and CUDA driver loading independently inside the module loader.
- Repeat first/repeated-process checks on an idle machine with the same executable.
- If runtime compilation is confirmed as the avoidable cost, implement a reusable compiled-module cache in cuda-host with source, architecture, compile-option, and toolchain identity plus atomic publication.
- Evaluate build-time cubin materialization separately. The existing backend offers it for an explicit architecture; it is not currently the project's default build route.
- Show initialization and terminal failures clearly in the viewport rather than presenting an unexplained empty frame.

No loader cache, precompiled-cubin default, or new startup overlay is implemented by the timing instrumentation. Preserve worker ownership, bounded queues, cancellation, and nonblocking UI polling when addressing startup.
