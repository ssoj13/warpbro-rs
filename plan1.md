# plan1: bug hunt findings and remediation (2026-11-11)

Scope: `src/*.rs`, `build.rs`, `xtask/`, `examples/`, `tools/mp4-quality/` at `a5ee386`.
Method: two independent read-only reviewers (engine/main/build/tools and
fractal_node/formulas/presets/templates) plus my own re-check of the key claims against the code,
plus `cargo clippy --all-targets --release` (section 0).
Why this file exists: it is the single surviving record of the audit, so it survives context
compaction. Every finding has a file:line and a failure scenario; the checkbox list at the end is
the proposed order of work. Nothing has been changed in the code yet. It waits for operator approval.

Tags: **C** = confirmed by reading the code, **P** = plausible, needs a runtime or dependency check.
No HIGH-severity bug was found. The packed parameter ABI (P_COUNT, extended slots 0..83, hybrid
step bits, family codes, masks) was traced slot by slot against `formulas.wgsl` and the pinned
`pt_integrator::studio_abi::params`, and it matches.

## 0. Compiler / clippy

`cargo clippy --all-targets --release` at `a5ee386`: 0 warnings (run took 15 min because a
concurrent `cargo check` from another session held the build lock).

## 1. Silent fallbacks and swallowed errors

| ID | Sev | Tag | Location | Problem | Failure scenario |
|---|---|---|---|---|---|
| F1 | MED | C | `src/fractal_node.rs:758-771` | `formula()` turns a deserialize error into `log::warn!` + Mandelbulb; a missing key gives Mandelbulb with no log. It is called at least 3 times per object per `Scene::pack` (`family_code` 1628, `bound_radius` 1613, `pack_object` 1665). | A corrupt or float-in-u32 formula renders a Mandelbulb instead of the user's object and spams 3+ warnings per frame. |
| F2 | MED | C | `src/fractal_node.rs:536-559`, getters 786-858 | `get_f32` / `get_u32` / `get_vec3` fall back to hard-coded defaults for missing or mistyped keys; `get_vec3` maps non-numbers to 0.0, `get_u32` accepts floats. Every shipped template and `default_fractal_custom_branches` always writes all keys, so the defaults only serve malformed or old data. This breaks the no-legacy rule. | `inversion_radius: "x"` silently becomes 1.0. |
| F3 | MED | P | `src/formulas.rs:360,471,492-493,627,648,672` | Integer formula leaves (`octaves`, `sub_strands`, `fibers`, `shells`, `crystal_count`, `ring_count`) have float ranges and linear sliders; `validate_attribute` (`fractal_node.rs:1117-1123`) does not check them. | The UI or key interpolation writes `2.5`; `Formula::deserialize` fails; F1 renders a Mandelbulb. |
| F4 | MED | C | `src/main.rs:198-205`, `src/engine.rs:436-448` | `--warmup` calls `new_gpu()`, which falls back to WGPU when CUDA fails, then prints `CUDA ready: <name>` and exits 0. | The build-time warmup reports success on a machine where the CUDA PTX was never JIT-cached. |
| F5 | LOW | C | `src/fractal_node.rs:1151,1159,977,979,1589-1603,1662-1664` | UI-path `.ok()` / `Null` fallbacks: a broken label shows the raw key, a tooltip disappears, a Hybrid step edit with invalid JSON or a length mismatch is dropped silently, and a short pack buffer returns silently in release. (1336 is benign: an empty path segment.) | The user's edit is ignored with no message. |
| F6 | LOW | C | `src/engine.rs:266` | `device_name().unwrap_or_else(\|_\| "CUDA GPU")` drops the error; `CudaContext::new(0)` hard-codes device 0. | Multi-GPU machine: the wrong device is used and the name is wrong, with no log. |
| F7 | LOW | C | `src/engine.rs:492` | `get_or_insert_with` caches a failed WGPU init forever. | A transient adapter failure is never retried. This may be intended; it is undocumented. |
| F8 | LOW | C | `xtask/src/main.rs:18-22` | When no CUDA-compatible Visual Studio is found, it falls back to any VS <= 2026 without a warning. `to_string_lossy` at 59-60 can corrupt non-UTF-8 paths. | A confusing nvcc or host-compiler error later. |

## 2. CUDA vs WGPU divergence and SSOT

| ID | Sev | Tag | Location | Problem | Failure scenario |
|---|---|---|---|---|---|
| S1 | MED | C | `src/engine.rs:747-814` (and the WGPU arms of every `ComputeEngine` method) | Hand copy of soluwell `WgpuComputeEngine::execute_step` (`render.rs:344-424` @0f63f61). It has already drifted: upstream clears `target.display_key`, WarpBro cannot because the field is private. That is harmless only because `Gpu::step` clears it (`render.rs:1015`). | Any upstream fix to WGPU pacing or adaptive sampling silently misses WarpBro. |
| S2 | LOW-MED | C | `src/engine.rs:146-160` | In the dynamic WGSL `switch`, Mandelbulb is `default`, so an unknown family code renders a bulb. CUDA `family_distance` (fractal-kernels `fractal_kernels.rs:1074`) returns a miss. | A bad `family_override` or a newer template shows a Mandelbulb on WGPU and nothing on CUDA. |
| S3 | LOW | C | `src/engine.rs:139-143` | A specialized module with an unknown code emits a call to the non-existent `unknown_fractal_family_{code}`, which gives an opaque naga compile error. CUDA returns a typed `RenderError::Scene` (engine.rs:850). | An unreadable shader error instead of a scene error. |
| S4 | LOW | C | `src/engine.rs:100-130` | 27 hand-copied `FAMILY_*` WGSL constants. Verified: `formulas.wgsl` contains `FAMILY_` only once, in a comment, and soluwell emits the 3 it needs itself. | Dead duplication of `fractal_core::params`, which must be hand-edited for a 29th family. |
| S5 | MED | C | `src/formulas.rs:703-708` | `*_CHOICES` string tables duplicate the variant names of `KifsKind`, `HybridStep`, `PolyhedralSymmetry`, `TpmsKind`, `ComplexTpmsKind` and `AttractorKind`. `formula_param_choices` takes precedence (`fractal_node.rs:1137`), so the serde-derived branch at 1142 is reachable only for `/formula/Hybrid/kifs/kind`. No test ties the strings to serde. | Rename or add a variant: `validate_attribute` accepts a choice that `Formula::deserialize` rejects, then F1 applies. |
| S6 | MED | C | `src/fractal_node.rs:1309`, `883-947`; `src/formulas.rs:723-1093` | Labels and hints of the classic families (codes 0..7) in `FORMULAS` are dead: `attribute_label` uses the table only for `code >= 8`, and `formula_hint` hard-codes the classic hints. | The table and the UI disagree. Example: "Rotation (°)" is shown as "Rotation degrees". |
| S7 | LOW | C | `src/engine.rs:291` | The 8x4 tile size is the bare literal `32`, duplicated with soluwell `Target::new` (`render.rs:563`). | If the two diverge, `active.copy_to_host` lengths mismatch and every adaptive step fails. |
| S8 | LOW | C | `src/formulas.rs:2764-2780` vs `src/fractal_node.rs:1717-1725,1878`; `fractal_node.rs:950-999` vs soluwell `world.rs:3046-3090` | `put_rot`/`put_vec3` duplicate `set_rotation`/`put3`; `to_attr`/`attr_json`/`discover_formula` duplicate private studio helpers. | Fixes land in one copy only. |
| S9 | LOW | C | `src/fractal_node.rs:635,751,1309` | Magic `8` instead of `FAMILY_AMAZING_SURFACE` (line 596 uses the constant). | Wrong split between classic and extended families if the code order changes. |
| S10 | LOW | C | `src/engine.rs:62-84` | `init_warpbro` registers `FractalNodeSpec`, and `WarpBroFractalModule::node_specs` registers it again; there is also a nested `OnceLock` for `FINGERPRINT` inside an init that already runs once. | Harmless (register replaces by name), but there are two registration paths. |
| S11 | LOW | C | `examples/pipeline_cache_probe.rs:22-34` | Re-implements the `gpu_info` cache-directory resolution and asserts nothing; its fixed WGSL is a cache hit on every run after the first. | The probe reports `None` or a stale "hit" without failing. |

## 3. Logic and parameter-range bugs

| ID | Sev | Tag | Location | Problem | Failure scenario |
|---|---|---|---|---|---|
| L1 | MED | C | `src/main.rs:209-211,218` | The DIR positional is `args.get(2)` with no `--` filter, unlike `arg()` (main.rs:53). | `WarpBro --gallery --hdr` writes into a directory named `--hdr`; `--animated-fixtures --all-frames` creates `./--all-frames/`. |
| L2 | MED | P | `src/main.rs:96-113` | `headless` never runs `create_dir_all(dir)` (`animated_fixtures` does at 127). | `--gallery newdir` renders the first preset fully, then fails to save. |
| L3 | MED | P | `src/fractal_node.rs:356-358,1698-1699` | The clip sphere uses fixed radii: Mandelbox `4.0637`, KIFS `kind.bounding_radius()`, Kleinian and Apollonian `PRESET.bound_radius`. User scale and offset are ignored, but the sliders allow Mandelbox scale -16..16 (`formulas.rs:784`) and KIFS offset +/-16 (`formulas.rs:906`). | Scale near +/-1 or a large offset: the fractal is cut off by the clip sphere. Confirm in a render first. |
| L4 | LOW | C | `src/formulas.rs` ranges vs `src/formulas.wgsl:1055,625,1172` | UI ranges disagree with WGSL clamps: `VoronoiFoam.jitter` 1.0 vs 0.95, `AttractorTube.step_dt` 0.2 vs 0.15. The Astrolabe inactive check `gear_teeth < 1.0` (`formulas.rs:2739`) disagrees with the WGSL rounding. Mandelbox `min_radius_ratio` can be 0 with no host `max(1e-6)` guard (AmazingSurface and Mobius have one: 2793, 2857). | Part of the slider range does nothing; a wrong "inactive" hint; possible division by zero in the DE. |
| L5 | LOW | C | `src/main.rs:149-157` | `first == last` renders the same frame 3 times; `first > last` with `--all-frames` renders nothing and gives no error. | Wasted work or an empty fixture run. |
| L6 | LOW | C | `src/main.rs:236` | Unknown `--flags` fall through to the GUI. | `--galery` silently opens a window in CI or a script. |
| L7 | LOW | P | `src/main.rs:84-94` | At `spp == 1` the warm-up step finishes the render, so the timed loop measures about 0 s. | `--bench` prints an absurd Msamples/s. |
| L8 | LOW | C | `src/engine.rs:716-726` | `self.cuda_mut()?` returns early before `self.lut_staging = staging` runs, so the staging capacity is lost. `pick` (1057) uses the correct closure pattern. | One reallocation after an error. Not hot. |
| L9 | LOW | C | `src/engine.rs:293-296` | `create_target` uploads `vec![1u32; tiles]`, then `reset_target_for_params` calls `activate_all` and uploads the same flags again. | An extra allocation and upload per film (not per frame). |
| L10 | LOW | C | `src/engine.rs:905` | `TALLY_THREADS as u32 / BLOCK` relies on an unchecked divisibility (4096/128 today). | A future constant change silently skips tally threads. |
| L11 | LOW | C | `src/engine.rs:437-438`, `src/main.rs:72` | `Gpu.name` is fixed to the CUDA device even when the scene resolves to WGPU. | The UI and bench label name the wrong device. |
| L12 | LOW | P | `build.rs:6-15` | The clippy / rust-analyzer early return has no `rerun-if-env-changed` for `CLIPPY_ARGS` / `RA_RUSTC_WRAPPER`. | A cached "ok" from a clippy run skips the backend guard in a later plain build. |
| L13 | LOW | C | `src/fractal_node.rs:1437` | `!matches!(family, "Mandelbulb" \| "Mandelbox")` is dead: those variants are always found in `FORMULAS`. | None. Dead branch. |

## 4. Performance (churn)

| ID | Sev | Tag | Location | Problem |
|---|---|---|---|---|
| P1 | LOW | C | `src/fractal_node.rs:1613,1628,1665` | Formula JSON is deserialized at least 3 times per object per pack, plus about 10 BTreeMap lookups. Parse once per pack. |
| P2 | LOW | C | `src/fractal_node.rs:880,1132,1197-1203,1322-1339` | `attribute_choices("/formula")` deep-clones 28 formula `Value`s per call; `attribute_label` and `attribute_slider` run `format!` and allocate a `Vec` per row per UI frame. |
| P3 | LOW | P | WGSL module build | Each of the 28 per-family modules includes all of `formulas.wgsl`, so naga validates all 28 estimators 28 times. Measure the compile time before acting. |

## 5. Tests and tools

| ID | Sev | Tag | Location | Problem |
|---|---|---|---|---|
| T1 | LOW | C | `src/presets.rs:171-197` vs `312-319` | Constant keyframe tracks (position and scale never change) count toward the `>= 6 animated attrs` assertion. |
| T2 | LOW | C | `tools/mp4-quality/src/main.rs:443-447,554,618-628` | The edge-column PSNR is computed but never asserted; `--rgb-sequence` skips every gate; running with no arguments exits 0; magic `Posix(11)`. |
| T3 | LOW | C | `src/engine.rs:257-258` | The `// SAFETY:` comment on `kernels::load` does not list the real preconditions: PTX/host signature match, the `P_COUNT` ABI, and context lifetime via `_ctx`. |

## 6. Docs drift

- `src/formulas.rs:79` says the estimators live in `render_wgpu.wgsl`; they are in `formulas.wgsl`.
- `AGENTS.md` "Ownership and per-frame loop" and "World document" anchors (`app.rs`, `world.rs:356`, `render_service.rs:393`, ...) describe files that now live in `soluwell-studio`, and the line numbers are from `4cfa5eb`. Keep them as cross-repo pointers, marked as such.
- `DIAGRAMS.md` section 5 refers to a "WGSL consts! macro" emitting `FAMILY_*` that nothing consumes (S4).

## Remediation plan (proposed order, awaiting approval)

Why this order: first the bugs that hide user data (F1/F2/F3/S5 form one failure chain), then the
backend divergence (S1/S2/S3), then CLI correctness, then cleanups. Each item names its
verification gate. Per the operator rules: no build while edits are in flight, one targeted test
run at the end of a wave, GPU tests only with the operator's yes.

### Wave A: formula error surface (fractal_node.rs, formulas.rs)
- [x] A1 (F1; P1 left: the trait methods are separate calls, see Status): parse the formula once per `Scene::pack` / `evaluate` and return `Result<Formula, _>`; surface a parse failure as a typed scene or validation error instead of a Mandelbulb. | verify: unit test where a corrupt `/formula` gives `Err`, not a bulb.
- [x] A2 (F2): turn the custom-branch getters into `Result`, or validate the custom map once at load (strict, no defaults). | verify: unit test where a mistyped key gives `Err`; all 11 templates still load (`templates` tests).
- [x] A3 (F3): validate `u32` formula leaves in `validate_attribute` (integer-valued, inside the range), and give them integer sliders. | verify: unit test where `2.5` on `/formula/GyroidTpms/octaves` is rejected.
- [x] A4 (S5): derive the `*_CHOICES` tables from the enums' `ALL` arrays through serde (one source), and delete the dead branch at `fractal_node.rs:1142` or make it reachable. | verify: test that every choice round-trips through `Formula::deserialize`.
- [x] A5 (S6, S9): use the `FORMULAS` labels and hints for all 28 families; replace the magic `8` with `FAMILY_AMAZING_SURFACE`. | verify: label test for one classic and one extended family.
- [x] A6 (F5, L13; P2 partly: choices are built once, the trait still returns an owned Vec): log or return the UI-path errors at 977/979/1151/1159/1601-1603/1662; remove the dead `matches!`; make `attribute_choices` return a cached `&'static`/`Arc`.

### Wave B: backend parity (engine.rs)
- [x] B1 (S1): hold a soluwell `WgpuComputeEngine` and delegate every `Backend::Wgpu` arm to it; keep only the shader hook and the CUDA arms in WarpBro. This needs `WgpuComputeEngine` to be constructible with a custom hook upstream; check that before starting (ask first if it needs an API change in egui-widgets-rs).
- [x] B2 (S2, S3): give the WGSL `switch` an explicit `case FAMILY_BULB` and a `default` that returns the miss value; reject an unknown code before building a specialized shader, with a typed error.
- [x] B3 (S4): delete the 27 unused `FAMILY_*` constants from the hook (keep the masks if the WGSL uses them; verify with grep first).
- [x] B4 (S7, L8, L9, L10, F6, T3): shared tile constant from `render_prep`; the closure pattern for `lut_staging`; drop the duplicate tile upload; `const _: () = assert!(TALLY_THREADS % BLOCK == 0)`; log the device-name error; write the real `// SAFETY:` preconditions.
- [ ] B5 (L11, deferred: `ComputeEngine::name` is static per engine; a per-backend name needs a soluwell API change): report the device name per resolved backend.
- [ ] Gate: `python bootstrap.py c` + GPU-free tests `--skip cuda_ --skip gpu_` (release). GPU parity run only with the operator's yes.

### Wave C: CLI (main.rs, build.rs, xtask)
- [x] C1 (F4): `--warmup` constructs CUDA explicitly and fails if CUDA is unavailable; print the real backend.
- [x] C2 (L1, L2, L6): shared positional parser that skips `--flags`; `create_dir_all` in `headless`; reject unknown `--` flags with a usage error.
- [x] C3 (L5, L7): dedupe the fixture frames and reject `first > last`; reject `--bench` with `spp < 2`.
- [x] C4 (L12, F8): `rerun-if-env-changed` for `CLIPPY_ARGS` / `RA_RUSTC_WRAPPER`; warn on the VS fallback in xtask.

### Wave D: ranges and clipping (needs renders, operator look)
- [ ] D1 (L3): derive the Mandelbox / KIFS / Kleinian / Apollonian clip radius from the instance parameters. First confirm the cut-off in a render at Mandelbox scale near 1.
- [ ] D2 (L4): align the slider ranges with the WGSL clamps (jitter 0.95, step_dt 0.15); host guard on `min_radius_ratio`; the Astrolabe inactive check uses the same rounding as the WGSL.

### Wave E: tests, tools, docs
- [ ] E1 (T1): make the preset animation test count only tracks that actually change.
- [ ] E2 (T2, S11): mp4-quality asserts the edge PSNR and gates `--rgb-sequence`, and a no-argument run exits non-zero; the pipeline probe gets pass/fail checks and the cache path from `gpu_info`.
- [ ] E3 (S8, S10): one helper for rotation/vec3 put; export the studio JSON helpers instead of copying them; single `FractalNodeSpec` registration path.
- [ ] E4 (docs): fix `formulas.rs:79`; refresh the AGENTS.md cross-repo anchors.

### Status after waves A-C (2026-11-11)

Why the fixes look the way they do:
- F1/F2 needed a validation point the render path cannot skip, so it went upstream:
  egui-widgets-rs `13d6f8d` adds `NodeTypeSpec::validate_node`, called on decode, after every
  edit (rollback) and per shown node in each snapshot. WarpBro's `validate_node` runs the same
  strict readers the accessors use. A numeric edit out of range is clamped by the editor
  (`attribute_range`) before validation, so the test that the hook rejects a node uses a string.
- The same commit makes `WgpuComputeEngine::wgpu_mut` public (B1 delegation) and adds
  `render_prep::tile_count` / `TILE_PIXELS` plus a `TALLY_THREADS % BLOCK` const-assert (S7, L10).
- Correction to the audit: the `FINGERPRINT` `OnceLock` (S10) is not dead.
  `set_extra_renderer_fingerprint` needs `&'static`, so it stays, with a comment. Also, L13's
  `None` arm was only partly dead: it handled unknown variants, which validation now rejects.
- Not done: B5 (needs an API change), P1 (formula parsed once per trait call), wave D (needs
  renders), wave E.

Out of scope: soluwell-studio / render-rs / fractal-kernels internals (cross-repo items B1 and E3
only touch them through public API requests), the PLAN.md Sections A-D roadmap, and any GPU benchmark.
