# plan2: render pacing and the display path, as fast as the GPU allows

Status: PROPOSED 2026-10-23, waiting for the operator's go. Nothing below is implemented yet
except stage 1's stop-gap (`Gpu::stamp` waits for the GPU, see "Current state").

## Why (verified in code, not measured yet)

- **The stall (fixed by the stop-gap).** Launches are asynchronous. A step that publishes no
  frame stamped its CPU time = the submit time (~0), `Batch::Budget` turned 8 ms into thousands
  of samples, the next publishing step waited for all of them (multi-second freezes, colour
  changes stuck behind it). plan1 wave 1 (e75a8c6) had removed the `clamp(1, 4)` that hid it.
- **The stop-gap costs GPU idle time.** Every step now waits for its own batch; while the CPU
  prepares the next launch the GPU has nothing queued.
- **Per-step host round trips.** Adaptive sampling reads the tile flags back synchronously every
  step once `samples >= adaptive.min()` (`Gpu::adapt`, CUDA `copy_to_host`; WGPU
  `launch_trace_and_tally(.., true)` maps). The tally (`Gpu::tally`) also reads back.
- **The display path copies every frame ~4 times through the CPU.** CUDA: tonemap ->
  `copy_to_host` (16 B/px) -> `ColorPipeline::apply_into` uploads to the wgpu shared device ->
  `GpuBufferExecutor::execute` -> readback (16 B/px light + 4 B/px codes) -> the UI uploads
  again to the present device's texture (`hdr_view` `stage_frame`). WGPU film: the same minus
  the CUDA copy. At 2560x1440 one f32 frame is 59 MB; this is likely the viewport's real
  ceiling (to be measured in stage 0).

## Tools available (checked in the pinned dependencies)

- cuda-core e78756c: `CudaEvent` (`record`, non-blocking `query`, `synchronize`,
  `elapsed_ms` with `CU_EVENT_DEFAULT`), `CudaStream::record_event` / `wait`.
- wgpu 30.0.1: `queue.submit` returns a `SubmissionIndex`;
  `PollType::Wait { submission_index: Some(i), timeout }` waits for one submission only;
  `PollType::Poll` never blocks; `Features::TIMESTAMP_QUERY` for GPU timestamps (the shared
  device comes from `gpu_info::shared_device`: whether it requests the feature is to check).
- vfx-ocio (oiio-rs 302a88d): `GpuTextureExecutor::execute(input, output)` = texture -> texture
  on the shared device, submit only, no readback; `wgsl_prelude` / render-rs
  `render_color_gpu::OcioShader` = the processor as WGSL to fuse into our own pass.

## Stages (each: implement -> GPU-free tests + clippy -> measured A/B with the operator's go)

0. **Measure first (no behaviour change).**
   - Per-stage timing on the worker: submit, GPU time (CUDA events / wgpu timestamps where
     available), every host wait (which one, how long), readback bytes and time, OCIO time,
     UI upload time; frames/s and Msamples/s. Through `tracing` spans, off by default.
   - A headless "viewport pattern" in `--world-bench` (display every 1/60 s, adaptive on,
     camera moves = restarts) next to the existing throughput cases, so A/B runs need no window.
   - Baseline on an idle machine: 0318557-equivalent pacing vs today, CUDA and WGPU, 1080p and
     1440p. Golden check: the accumulated film after N samples is bit-identical across pacing
     changes (same sample begin / seed per batch).

1. **Pacing: true GPU cost, two batches in flight, no idle.**
   - Replace the stop-gap's "wait for my own batch" with a two-deep ring: record a fence after
     each batch (CUDA event; wgpu submission index), before submitting batch N+2 wait for batch
     N's fence. The GPU always has the next batch queued; host latency <= 2 batches.
   - Cost per sample from the GPU itself (events' `elapsed_ms` / timestamps, read when the
     fence has passed, never blocking), not CPU wall time. `Batch::Budget` then sizes exactly.
   - Separate budgets: viewport = frame time minus measured display cost; export = large
     batches (latency irrelevant), capped below the Windows TDR limit (~1-2 s per launch).

2. **No synchronous per-step readbacks.**
   - Adaptive flags and the tally stay on the GPU; their small readbacks become asynchronous
     (CUDA: async copy into pinned memory + event; wgpu: `map_async` + `PollType::Poll`) and
     are consumed one batch later. The scheduler decides on the previous batch's result: at
     most one extra batch past convergence, no stall.

3. **Display on the GPU, one readback at UI rate (or none).**
   - WGPU film: tonemap and OCIO fused into one pass (OCIO WGSL from the processor, the
     render-rs `OcioShader` route) writing display light + SDR codes; read back once per
     displayed frame, as RGBA16F where the window is HDR, 8-bit codes for SDR (4x less).
   - CUDA film: async D2H of the tonemapped film into pinned double buffers overlapped with the
     next batch; OCIO on the shared wgpu device texture -> texture (`GpuTextureExecutor`).
   - Display only at the UI's cadence (gui fps), never per step.

4. **Zero-copy present (largest; decide after 0-3 are measured).**
   - Render (WGPU film), OCIO and the window's texture on ONE wgpu device, so the compute pass
     writes the texture the window shows: no readback at all (plan1 7.3). Today the window uses
     a private present device so surface reconfiguration does not wait for the render queue;
     with stage 1's bounded queue that reason has to be re-measured.
   - CUDA film: needs CUDA <-> Vulkan external-memory interop (wgpu hal level); only if stage
     3's pinned path is measured as the bottleneck.

## Order and rules

- 0 -> 1 -> 2 -> 3 -> (4 if the numbers say so). Every stage keeps the film bit-identical and is
  measured against the previous stage; a change that is not faster is reverted (by edit, never
  git reset).
- GPU runs (benches, CUDA/WGPU tests) only with the operator's go, through buildlock, on an
  idle machine; A/B interleaved, median of >= 5.

## Current state

- Stop-gap in the working tree (not committed yet): `Gpu::stamp` waits for the backend's work
  before stamping (correct, no stalls, GPU idles between steps). Colour: a selection change no
  longer drops the OCIO transform cache (`DisplayKey` keyed by `Ocio::serial`, `ReloadColour`
  only on a real config reload).
