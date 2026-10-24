# HANDOFF (2026-10-24)

Read first: `PLAN.md` (rules, open work), `plan1.md` (bug hunt: remaining items in "Audit of the
open checkboxes"), `plan2.md` (render pacing / display speed), `CHANGELOG.md` "Unreleased",
`AGENTS.md` (module map, build rules). Cross-repo: oiio-rs `plan-cuda-oxide.md`.

## State

- All work is on `main` (fast-forwarded from `fix/plan1`; every finished commit goes to both).
- Viewport stalls fixed (52f6545): `Gpu::stamp` waits for the backend's work before recording a
  batch's time, so `Batch::Budget` sizes batches from the GPU's real cost. Colour changes reuse
  the cached OCIO transforms (`DisplayKey` keyed by `Ocio::serial`). Not yet seen by the
  operator in a rebuilt app: the installed `~/.local/bin/WarpBro.exe` is from 2026-10-16;
  `python bootstrap.py b` rebuilds it.
- The stop-gap costs GPU idle time between steps; plan2.md removes it (two batches in flight,
  GPU timers, no synchronous per-step readbacks, display on the GPU).
- In progress elsewhere: oiio-rs `plan-cuda-oxide.md` (approved 2026-10-24): CUDA for
  vfx-compute and vfx-ocio on cuda-oxide, the cudarc/NVRTC engine removed. Stage 0 spike in the
  session scratchpad (`oxide-probe/`: math, kernels, host). WarpBro then runs tonemap + OCIO on
  its CUDA film without leaving the GPU (plan2 stage 3).

## Next

1. oiio-rs plan-cuda-oxide.md stage 0 results -> stages 1.. (shared no_std math, kernel crate,
   vfx-ocio CudaExecutor, old engine removal).
2. plan2.md stage 0 (measurements; GPU runs approved 2026-10-24) -> stage 1 pacing.
3. plan1.md leftovers: the `.ok()` sweep part 2 (`nodes()` -> `Result`); operator decisions
   Q2 (`conn`), Q4 (`path_convergence.rs`), 7.3 (GPU->CPU->GPU display, now plan2 stage 4).
4. Pending GPU checks of plan1 7.5: the WGPU world test compile time (was 4946 s before 7.5),
   `parity_tests::cuda_wgpu_primary_guides_agree_per_pixel`, an A/B `--world-bench --backend
   wgpu`. Then delete `target/release` (kept for these).

## Traps

- Build / test only through `python C:/projects/projects.rust.cg/cglibs/.buildlock/buildlock.py
  <cmd>` (machine-wide lock, never nested), and only after the edits are finished.
- Tests need `--release` (debug device codegen of fractal-kernels fails):
  `buildlock.py cargo oxide test -- --release --locked -- <filters> --skip cuda_ --skip gpu_`.
  Type-check: `FRAC_ALLOW_PLAIN_CARGO=1 buildlock.py cargo clippy --all-targets --locked`.
- Device tests start with `cuda_` / `gpu_`; no other test name may contain them
  (`test_gpu::device_prefixes_only_start_test_names`).
- CUDA / WGPU runs: only with the operator's go, on a quiet machine.
- Own git deps: edit the lock's rev by hand, never `cargo update -p` (it downgrades windows-sys).
- A git commit from a pwsh stdin script stops after `git add` (CRLF warnings on stderr):
  commit with `git commit -F <file>` as its own command.
