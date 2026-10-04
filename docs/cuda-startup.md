# Warm CUDA before opening the workspace

## Build and validate the exact executable

`python bootstrap.py b` builds the existing embedded-PTX renderer, then runs the resulting executable with `--warmup-cuda`. That headless command creates the CUDA context, loads the exact module, validates the host-visible parameter block, and exits without opening a window or authoring a scene.

A failed module/parameter initialization now fails the default bootstrap build. For packaging or cross-building without a GPU, use `--skip-cuda-warmup`. A direct `cargo oxide build` does not perform this extra step:

```powershell
python -u bootstrap.py b
$env:RUST_LOG = "info"
.\target\release\WarpBro.exe
```

For an already-built executable:

```powershell
.\target\release\WarpBro.exe --warmup-cuda
```

This warms the CUDA driver's existing disk JIT cache. It does not add a separate application module cache or populate the Timeline frame cache. Driver changes, cache eviction/clearing, another GPU, or a disabled cache can require another cold compilation. NVIDIA documents `CUDA_CACHE_DISABLE`, `CUDA_CACHE_PATH`, and `CUDA_CACHE_MAXSIZE` in [the CUDA environment-variable guide](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html). Keep build and runtime cache settings consistent.

The interactive worker still owns CUDA. Bounded queues, cancellation, and nonblocking UI polling are unchanged. Parameter upload failures now become target errors instead of panicking the shared worker, and startup validates the block before reporting Ready.

## Verified release result

The accepted production build completed in 1m 12s of Cargo time. Its CUDA warmup initialized successfully in 187.3112 ms (304 ms for the complete process). The following native GUI launch created the context in 106.3973 ms and loaded/validated kernels in **87.5814 ms**. The viewport, material preview, and Gallery cards all rendered; OIDN also completed. These are warm-cache startup measurements on this machine, not an idle cold-cache benchmark or a tracing-throughput speedup.

## Why every render panel looked broken

Viewport, material previews, and Gallery use the same worker. It cannot process requests until `Gpu::new` finishes loading the module. The reproduced native launch took 277.2692 ms to create the context and **173.177592 s** to load kernels. After that delay, the viewport, material preview, and Gallery thumbnails rendered.

A separate repeated launch of the glass test module loaded kernels in 239.2474 ms, showing that cache state matters. Compare first and repeated processes rather than calling a warm repeat a cold-start speedup. The new build route performs that cold work before the next workspace launch; it does not make compilation itself faster.

## Cubin experiment was rejected

The explicit `--materialize-cubin --arch sm_86` trial loaded in 82.0381 ms, then failed its first constant-memory parameter upload with `DriverError(500, "named symbol not found")`. Inspection of the embedded cubin found no parameter symbol. The resulting worker panic made every render panel unavailable.

A volatile-read workaround triggered a very slow full NVVM compilation and was stopped before completion. Neither that workaround nor cubin materialization is part of the accepted renderer/build path. The toolchain needs a proper externally updated constant-memory contract and a focused regression before adopting that route. Early parameter validation prevents this particular incompatibility from first appearing halfway through a render request.

## Earlier measurements

| Run                                      |                       CUDA context |        Embedded kernels | Whole headless check | Load caveat                                            |
| ---------------------------------------- | ---------------------------------: | ----------------------: | -------------------: | ------------------------------------------------------ |
| Initial executable                       |                          Not split |               Not split |            114.085 s | Contended machine                                      |
| Rebuilt executable                       | Logging filter excluded the module | Not captured separately |            104.152 s | Not an idle baseline                                   |
| Retry, 2026-10-03 local / 2026-10-04 UTC |                        122.5224 ms |           152.3668334 s |            156.041 s | CPU rose from 33% to 76%; concurrent atlas_chans_bench |

The retry began at 34% GPU utilization with 8,739 MiB allocated. These tiny 32×32, 1-SPP checks establish successful startup, not production-world convergence or a GPU tracing speedup.

## Capture useful evidence

Use `RUST_LOG=info`. The module path is `WarpBro::render`; lowercase `warpbro=info` does not select it. Record `context ready in` and `kernels ready in` separately, along with architecture, artifact route, executable/dependency revisions, cache settings, system load, and first versus repeated-process starts. Do not terminate other applications to create an idle baseline.

Generated local logs include `target/gui-diagnostic.stderr.log`, `target/startup-fixed-build.stdout.log`, and the earlier `target/startup-retry.stderr.log`.

A reusable application module cache and a viewport initialization overlay remain future work.
