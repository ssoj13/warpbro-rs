# plan1: bug hunt 2026-10-22 (HEAD 4cfa5eb)

Status: **REPORT + PLAN, waiting for operator approval. No source file was changed.**

How this was made: five read-only reviewers split the crate without overlap (app shell; world +
takes; CUDA render path; WGPU + shaders; IO / colour / scene). Two independent adversarial
verifiers then re-checked every HIGH / MED finding and every proposed deletion against the code
(and against fractal-rs `b39c64f`, the revision pinned in Cargo.lock). The lead re-checked the key
items by hand. Only verified items are listed as bugs; refuted ones are listed at the end.
Raw reports (session scratchpad, not in the repo): `review-{app,world,render,wgpu,io}.md`,
`verify-A.md`, `verify-B.md`.

Anchors are `file:line` at 4cfa5eb. Severity: H = wrong output / hang / crash in normal use,
M = wrong behaviour in a reachable case, L = smell, latent or cosmetic.

Companion docs: [AGENTS.md](AGENTS.md) (ASCII dataflow, module map, traps) and
[DIAGRAMS.md](DIAGRAMS.md) (mermaid).

---

## 0. Decisions needed from the operator

- [ ] **Q1. Legacy migrations in `WorldDocument::hydrate_schema`** (`world.rs:5639-6346`, ~580
  lines, added 2026-10-16 in `edd91fa` / `46fc274`): pre-Track channels (5643-5685),
  `settings_node` / `world.render` (5703-5797), QualitySettings folding (5799-5922, 6164-6220),
  `/camera/aperture` (6022-6039), legacy `/julia` (6040-6047). This contradicts the
  "no legacy compat" rule. Proposal: convert checked-in templates and `~/.warpbro` templates /
  bookmarks once with a throwaway script (backups), add a test that every shipped template decodes
  with no migration step, then delete the migration steps and their tests. Separate question: keep
  the generic "fill missing params from schema / drop removed params" merge or not (it is what
  makes appended ABI slots work on older files).
- [ ] **Q2. Attribute connections (`conn`)**: evaluated in 7 places (`world.rs:4469-4477`,
  `5338-5351`, `3665-3679`, `3724-3735`, `5041`, `1481`, `2985/3367`) but created only by tests and
  by the Q1 migrations. Finish (a "connect attribute" UI) or remove. Decide together with Q1.
- [ ] **Q3. Explicit `Backend::Cuda` + extended family (8..27)**: CUDA has no kernel for them and
  silently draws Hybrid. Proposal: a typed error ("family X needs WGPU") at admission, consistent
  with "explicit backend never falls back". Alternative: port the 20 families to fractal-rs.
- [ ] **Q4. `path_convergence.rs`** (unwired roulette / clamp policies): move to fractal-rs
  (fractal-core), wire into both kernels and A/B with `--world-bench` (PLAN.md item 12), or delete.
- [ ] **Q5. Undo history bound** (`world.rs` undo stack of full document clones, each node holding a
  full-scene `gpu` blob): cap by count or by bytes? Proposal: byte budget + slimmer `gpu` blobs (5.4).

---

## 1. Render robustness (one change, verifier B: must not be split)

Root cause: GPU failures are panics or a side channel (`Target.colour_error`) instead of
`Result`s, and worker events carry no request id.

- [ ] **1.1 H. Step errors never fail the job.** `Gpu::step_with_display` (`render.rs:779`) records
  the error in `colour_error` and returns without samples. Export loop
  (`render_service.rs:1460-1503`) keeps sending `ExportProgress`; the coordinator
  (`export.rs:1323-1357`) waits forever; one failed thumbnail blocks every later thumbnail
  (`render_service.rs:1174`). `prepare_scene` never calls `resolve_backend_for_scene`, so an
  unavailable explicit backend passes admission. Fix: `step_with_display -> Result<StepOutcome, RenderError>`;
  keep `colour_error` for non-fatal display / OCIO messages only; resolve the backend in
  admission (`prepare_scene`).
- [ ] **1.2 H. CUDA allocation / launch failures panic** (`.expect` on alloc and launch in
  render.rs) and kill the worker for the session ("CUDA worker is unavailable"). A 64 MP export
  needs ~5.4 GB VRAM. Fix: propagate `Result` from target creation, upload, launch, readback.
- [ ] **1.3 H. WGPU dispatch > 65535 workgroups** for frames above ~8.39 MP (e.g. 4096x2160):
  `dispatch_workgroups(padded/128)` in render_wgpu.rs; no uncaptured-error handler, so wgpu panics
  and the worker dies. Fix: 2D dispatch (x, y) from `device.limits().max_compute_workgroups_per_dimension`,
  WGSL index from `global_id.x + global_id.y * stride`; check buffer binding sizes against limits
  in `create_target` (inside an error scope, returning `Result`).
- [ ] **1.4 M. WGPU readback `.expect` (7) and `let _ = device.poll(..)` (5)**, lazy pipeline
  compilation outside error scopes (render_wgpu.rs `:1122-1130`, `:1191-1203`, `:1340-1350`). Fix:
  one `WgpuEngine::run(label, pass_fn, copies) -> Result<Mapped>` helper (also dedups U8 below).
- [ ] **1.5 H. OIDN never runs on the WGPU path when adaptive is off or below `adaptive.min()`.**
  `denoise::State::due` consumes the attempt; it is called at `render.rs:899` and again in
  `process_denoise` at `render.rs:1231` with the same sample count, so the second call is always
  false. Affects viewport, previews, thumbnails and exports (Auto routes extended families to WGPU).
  Fix: decide `due` once per step and pass it into `process_denoise(target, scene, due)`.
  Regression test on the WGPU target (no CUDA needed: `gpu_*`).
- [ ] **1.6 M. Worker `continue` drops a dequeued command** (`render_service.rs:1224`, `:1327`,
  also `:1436`): a lost Pick / RenderExport hangs its requester. Fix: handle the command before
  `continue`, or answer it with its typed failure.
- [ ] **1.7 M. Thumbnail failures are untyped.** `RenderEvent::Error` has no id
  (`render_service.rs:312`); `app.rs:2903` fails whatever thumbnail is pending, including on
  viewport errors; a failing gallery thumbnail is re-requested every frame (a `FRAC_SNAP` run that
  waits for thumbnails never ends). Fix: `RenderEvent::ThumbnailFailed { id, error }` like
  `PreviewFailed` / `ExportFailed`; plus `enum ThumbId` (5.9).
- [ ] **1.8 M. Viewport Direct mode re-enables adaptive.** `app.rs:2807`
  (`snapshot.render = effective.render`) overwrites what `EffectiveRender::apply_to` did. Fix: call
  `effective.apply_to(&mut snapshot)` there (idempotent), so one function applies a profile.
- [ ] **1.9 L. `try_command` returns `Result<(), String>`**; export retries on
  `error.contains("queue is full")` (`export.rs:1318`). Fix: `enum CommandError { Full, Stopped, Invalid(String) }`.
- [ ] **1.10 L. Export ignores `Frame.denoise_error`** (`export.rs:1720`, `:1861` check only
  `colour_error`): a denoise-enabled export silently writes raw frames. Fix: fail the frame.
- [ ] **1.11 L. `CancelExport` is global with no id** (verifier: unreachable today, single export).
  Fold into 1.1's typed job ids when touching it.
- [ ] **1.12 L. `animated_fixtures` reuses one target** (`main.rs:139-229`): fails for a single-frame
  preset (`first == last`); fixed by 5.1 (`Gpu::advance` resets per frame).
- [ ] **1.13 L. Poisoned locks**: `lock().unwrap()` in the worker and `hdr_view.lock().unwrap()`
  (`app.rs:4242`); use `PoisonError::into_inner` consistently.

---

## 2. CUDA vs WGPU parity (WGSL is a hand port and has diverged)

First build the safety net, then fix each divergence against the pinned fractal-rs revision.

- [ ] **2.0 Parity test (`cuda_*`).** For every classic family, each trap mode, Hybrid, a Cells
  mask and a radial mask array: render 1 spp, fixed seed, small size on both backends; compare
  primary-hit depth, guide albedo / normal and DE samples within a tolerance. Today
  `render.rs:4003` only checks "Mandelbulb is not black". This test must fail on 2.1-2.7 first.
- [ ] **2.1 H. Radial mask arrays**: WGSL `mask_array_angle` (`render_wgpu.wgsl:1114-1121`) reads
  degrees; host packs radians with 0 = full circle (`render_prep.rs:684`). Default sweep 0 stacks
  every copy on copy 0. Fix: port `fractal_core::mask_array` exactly (radians, 0 = full circle).
- [ ] **2.2 H. Cells mask** (`render_wgpu.wgsl:1071-1076`, `1186-1216`): different hash order and
  width, seed decoded by `bitcast` instead of `as u32` (host writes `seed as f32`), a WGSL-only fill
  box from `size.y`, `M_INVERT` ignored. Viewport thinning uses Cells, so it is visible. Fix: port
  the CUDA cells function 1:1.
- [ ] **2.3 H. Kleinian** (`render_wgpu.wgsl:704-766`): different wrap, separation line and
  inversion than CUDA (CUDA and ofx WGSL agree with each other). Fix: port from CUDA.
- [ ] **2.4 H. Pseudo-Kleinian offset** read as a scalar on Z (`render_wgpu.wgsl:785`); host packs
  a vec3 (`scene.rs:1607`). Fix: read the vec3.
- [ ] **2.5 H. Explicit CUDA + extended family draws Hybrid** (`_ => hybrid_distance` in the kernel,
  `render.rs:301` switches only under Auto). Fix per Q3.
- [ ] **2.6 M. TrapPoint (mode 3)** measures distance to an axis in WGSL (`render_wgpu.wgsl:194-217`);
  CUDA, ofx and the doc (`world.rs:2581`) use the point. Fix: port.
- [ ] **2.7 M. Hybrid** (`render_wgpu.wgsl:869-925`, `822`): DE zeroed while `r <= bailout`, extra
  stop at zero Apollonian scale, Apollonian step rotated; none of it in CUDA. Fix: port.
- [ ] **2.8 L. `FAMILY_WORLD == FAMILY_AMAZING_SURFACE == 8`**: `world_*_f8` specialization falls
  into the generic switch (output correct). Fix: move `FAMILY_WORLD` outside the family range in
  fractal-core (coordinate with fractal-rs / ofx-rs).
- [ ] **2.9 M (perf). `tally` runs in one workgroup** over the whole frame on every fused and every
  adaptive step (`render_wgpu.wgsl:3559`). Fix: two-level reduction (per-workgroup partials, then
  one small pass), as CUDA does with `TALLY_THREADS`.

---

## 3. Export and colour

- [ ] **3.1 H. Export panel rewrites OutputSettings every UI frame through a lossy round trip.**
  `ExportController::ui` does `preset = output.to_preset()` (`export.rs:1105`) and
  `output.apply_preset(&preset)` (`export.rs:1124`); `App::export_ui` (`app.rs:3063`) stores any
  difference as an undoable edit. Lost or rewritten (verifier A, field by field): container,
  video depth, encoder (Auto after ProRes / VP9 / EXV), qp (clamped to 51), channels (JPEG forces
  RGB, later PNG / EXR lose alpha), and PNG encoding copied into `video_signal` on PNG -> Video.
  Root cause: `ExportSettings` / `OutputSettings` re-implement `video_io::job::ExportSettings` /
  `Preset` (`export.rs:422-491`, `624-726`, see 5.7). Fix: `OutputSettings` holds a
  `video_io::job::Preset` plus WarpBro extras (samples, transform, denoise_at_completion, output
  override); the dialog edits that preset in place; no conversion, no per-frame write-back;
  round-trip test.
- [ ] **3.2 M. Colour selection has 4-6 sources of truth**: `App.colour.sel` (`ocio::State`),
  `Scene.colour`, `bus_slots["world"]["colour"]` (`world.rs:1623`, `5292`), `Settings.colour`
  (`app.rs:561`, applied over every loaded scene incl. File > New at `872-875`), `ColourPresets`,
  export override. Undo / redo changes the document but not `colour.sel`; the next colour edit
  writes the stale selection back; export uses a stale config (`app.rs:3099` vs `3107`). Fix: the
  document owns the selection; `refresh_scene` pushes it into `ocio::State` through one
  `State::set_sel` (rebuild only on change); `Settings.colour` becomes only the default for new
  documents; delete the manual copies (`app.rs:1593`, `1611`, `1956`, `3660`).
- [ ] **3.3 M. One OCIO config, loaded once.** Loaded independently by the colour worker
  (`ocio.rs:660`), `ColorPipeline::ensure_cached` (`color.rs:175`, per new `Sel`) and export start
  (`export.rs:872`). Fix: `ocio::State` owns `Arc<Ocio>` per (source, reload serial) and hands it
  to the renderer (`ReloadColour`) and to export.
- [ ] **3.4 M. HDR view breaks on configs without ACES 2.0 names**: `color.rs:180-185` hard-codes
  `"sRGB - Display"` and the ACES 2.0 SDR view for the SDR preview; other configs error in the
  viewport and refuse every HDR export frame. Fix: resolve the SDR display / view from the config
  (`video_io::output_view::resolve_output_view` / viewing rules), typed error only when none exists.
- [ ] **3.5 M. 16-bit SDR stills differ from 8-bit** (`export.rs:1754`): 8-bit = OCIO display
  codes, 16-bit = sRGB OETF of linear Rec.709; differs for P3 / Rec.1886 displays; 16-bit silently
  falls back to 8-bit when `light` is empty. Fix: 16-bit codes from the same OCIO display transform
  (float executor output quantized to u16); error instead of fallback.
- [ ] **3.6 L-M. HDR video mastering peak = requested `png_peak_nits`** (`export.rs:1811`), not the
  view's peak used by PNG (`export.rs:95-97`). MaxCLL / MaxFALL are measured by video-io (verifier
  correction). Fix: same peak source as PNG.
- [ ] **3.7 M. IO worker never joined** (`io_service.rs`): the final settings write can race exit; a
  panic in it kills all file IO ("File writer busy"); temp file left behind when rename fails.
  Fix: `IoService: Drop` closes the channel and joins; `catch_unwind` per command with an error
  event; remove the temp on failure; tmp name `<file>.tmp` instead of `with_extension("json.tmp")`.
- [ ] **3.8 L.** `exr_io::read_rgb` does not check plane lengths / multi-part files; DWA level range
  accepts 0 (video-io rejects); `write_rgb` hard-codes DWA 45 while `DWA_LEVEL` exists
  (`exr_io.rs:46`); empty dated folder left when the video encoder fails to open.

---

## 4. Document and UI behaviour

- [ ] **4.1 M. Timeline reorder moves hidden Shy nodes (settings) to the end** (`world.rs:983-1001`,
  `world_ui.rs:2366-2383`, `2025-2039`). Fix: `Reorder` permutes only the listed ids within the
  positions they occupy.
- [ ] **4.2 M. Sun / Environment fields edit and key but do nothing.** `branches()`
  (`world.rs:2716-2717`) gives both kinds the whole `Lighting`; render reads only its part.
  `sun_disc` works only for the first Sun. Fix: split `scene::Lighting` into `Sun` and `Sky` structs;
  each node kind exposes only its own.
- [ ] **4.3 L-M. Rename makes one undo step per keystroke and cannot be cleared to retype**
  (`world_ui.rs:2783`, `world.rs:973`). Fix: edit buffer, commit on focus loss / Enter (one step).
- [ ] **4.4 L-M. Transport In / Out / FPS drags make one undo step per frame**
  (`world_ui.rs:2542`). Fix: `panel_command("timeline_work_area")` gesture, as the work-area drag.
- [ ] **4.5 L-M. Malformed documents panic**: `serde_json` index assignment (`v[i] = ..`) at
  `world.rs:5986`, `4647`, `4592`, `877`, `5063/5070`, and `u32` subtraction at `3554` when
  `first > last`. Fix: `get_mut` + typed error; validate `first <= last` in `decode_json`.
- [ ] **4.6 L. Keyed `/visible`** shows visible in the outliner (`info_with` `world.rs:4072`) while
  the timeline evaluates it at the frame. Fix: one evaluated value for both.
- [ ] **4.7 L. Takes Play** seeks to `take.start`, ignoring FitWorkArea and `/take/offset`. Fix: use
  `take_span`.
- [ ] **4.8 L. Paste status** counts unresolved take refs as parent refs (`app.rs:3049`); take refs
  cleared in two places (`world.rs:963-968`, `749-753`). Fix: `insert_fragment` alone decides and
  returns counts per field.
- [ ] **4.9 L. Delete** leaves stale ids in `layer_order` and deletes locked descendants.
- [ ] **4.10 L. Duplicate has four entry points with different semantics**; one `WorldEditor::duplicate`.
- [ ] **4.11 L. Gallery "selected" highlight never shows** (`app.rs:3756`, scene equality includes
  the renewed document id); **Save bookmark** sets `App::tab` (`app.rs:2515`) which the dock
  overwrites; **Reset scene** drops the file path and the undo history.
- [ ] **4.12 L. Window ignores egui repaint requests** (`window.rs:355-363`): fixed `1/gui_fps`
  redraw even when idle; every `request_repaint` is a no-op; `next_redraw` set twice
  (`window.rs:129`, `185`). Fix: honour `viewport_output.repaint_delay` (idle at 0 % CPU); keep
  continuous redraw only while rendering / playing.
- [ ] **4.13 L. Menu Undo / Redo runs after `snapshot_before`** (mid-frame); route through the same
  path as the hotkey.
- [ ] **4.14 L. Preview-cancel uses raw keys G / H / F** (`app.rs:4548`, G unbound): use `hotkeys`.
- [ ] **4.15 L. `--bench` / `--gallery` with 0 spp underflows** (`main.rs:112`); fixed by 5.1.
- [ ] **4.16 L. Silently discarded errors**: `let _ =` at `app.rs:892`, `1594`, `1620`, `1722`;
  `unwrap_or` at `app.rs:1962`; unreadable bookmarks vanish; hydrate `let _ =` (`world.rs:5753`,
  `5787`); failed take filter plays raw motion (`takes.rs:577`); evaluation error shows an empty AE
  (`world_ui.rs:1215`); `catalog_role(..).ok().flatten()` (`render_profiles_ui.rs:650`, `744`);
  `export.rs:1021`; `ocio.rs:317-324`; `mask_row(..).unwrap_or(NaN)` (`render_prep.rs:559`).
  Each becomes a status message or a propagated error.
- [ ] **4.17 L. Settings compat defaults** `#[serde(default)]` on `min_take_secs` and
  `Controls::selection_bounds` (`app.rs:484`, `569`) in a `deny_unknown_fields` struct: remove
  (no-compat rule). Preview cache invents `cache_fraction 0.05`, `reserve_gb 2.0`
  (`app.rs:4497-4498`): take the budget from gpu-info / an explicit setting.

---

## 5. Single sources of truth and deduplication

- [ ] **5.1 One accumulate loop.** Six copies with different batch rules: `main.rs:102-110`,
  `main.rs:192-205`, `render_bench.rs:176-190`, `render_service.rs:1037-1048` (preview),
  `1460-1474` (export), `1510-1524` (thumbnail). Add
  `Gpu::advance(target, scene, goal_spp, seed, budget_ms) -> Result<bool, RenderError>` owning batch
  size, `final_pass` and completion; CLI / bench loop on it.
- [ ] **5.2 `Frame` vs `Target` twins**: `complete`, `msamples_per_s`, `save_png`,
  `save_display_exr` (`render.rs:279`, `1331`, `1340-1390` vs `render_service.rs:193-243`). Keep the
  `Frame` versions; CLI saves through `Frame::snapshot`.
- [ ] **5.3 `enum Film { Cuda(CudaTarget), Wgpu(WgpuTarget) }`** replaces `Target.cuda`, `.wgpu`,
  `.active_backend` and ~9 dispatch blocks (`render.rs:346-433`, `521-549`, `611-636`,
  `1139-1153`, `1178-1298`); fixes the film leak on backend switch. Promote the test-only
  `read_guide_sums` to `Gpu::read_film(target, Plane, ..)` used by `raw_scene_linear` and
  `process_denoise`.
- [ ] **5.4 Render settings: `EffectiveRender` is the only producer** of per-render quality
  (samples, budget, seed, bounces). Remove worker-side patches: 8 ms background budget
  (`render_service.rs:1563`, `1566`), batch clamp 4 (`1578`), thumbnail `max_bounces.min(3)` and
  seed 7 (`1553`, `1518`), export seed = id (`1469`) - thumbnails get a profile.
- [ ] **5.5 Formula registry.** `Formula::NAMES` (`scene.rs:262-291`) vs
  `FORMULAS[].display_name` (5 differ; dropdown and header show different names,
  `world_ui.rs:3381` / `3437`); `has_orbit_constant` twice (`scene.rs:402-413`, `FormulaMeta`);
  classic slider table duplicated in `world.rs:1981-2011` (no `log` flag in `ParamMeta`);
  extended family codes hand-copied into `formulas.wgsl:6-25`; signed / unsigned DE only in WGSL /
  CUDA (`formulas.wgsl:35-43`). Fix: `FORMULAS` is the one table (name, orbit, slider{min,max,log},
  signed, iterates); `Formula` methods read it; `shader_source` emits `FAMILY_*`, `SIGNED_MASK`,
  `ITERATING_MASK` via `consts!`; const assert `FORMULAS[i].code == i`.
- [ ] **5.6 WGSL specialization finished**: generate `single_{fast,full,direct}_f{code}` for all 28
  families in the loop that already generates the world entries (`render_wgpu.rs:254-267`); delete
  24 hand-written entries, `FAST/FULL/DIRECT_NAMES`, the `[OnceLock; 8]` arrays and
  `world_fast_bulb` (identical to `world_fast_f0`). Merge `distance_and_trap` into
  `clipped_distance_and_trap_q` (one estimate path). WGSL helpers: `facing_mix` (3 copies,
  `2481`, `2565`, `3300`), `nee` (sun + env NEE twice in `trace_path`, `2845-2899` / `2936-2994`),
  `display_raw` (tonemap twins `3664-3711`).
- [ ] **5.7 Export settings = video-io types** (root of 3.1): compose
  `video_io::job::ExportSettings { job, preset, dir }` + WarpBro extras; delete the re-implemented
  `resolve / output / video_output / frame_path / sequence / frame_count / validate`
  (`export.rs:624-726`), `OutputSettings::validate` (`422-491`), duplicated constants
  (`HIGH_QUALITY_QP`, `DEFAULT_PNG_PEAK_NITS`, `MAX_PIXELS` also in `main.rs:151`). Snapshots and
  exports write PNG / EXR through one `write_still` path (`render_service.rs:206-243` vs
  `export.rs:1688-1791`). Rename `ExportFormat::Hevc` / `HevcSink` to `Video` / `VideoSink`.
- [ ] **5.8 Editor selection**: `WorldEditor::targets()` replaces the "selected else single" fallback
  in 8 places (`app.rs:1280`, `1374`, `2077`, `3004`, `3375`, `takes_ui.rs:771`,
  `world_ui.rs:81`, `725`); then one selection field.
- [ ] **5.9 App helpers**: `Scene::camera_radius()` instead of 4 hand copies (`app.rs:3166`, `3263`,
  `3369`, `3505`); `current_settings()` + `#[derive(PartialEq)]` instead of 4 field lists
  (`app.rs:559-587`, `836-875`, `1631-1657`, `1673-1697`); `enum ThumbId` instead of magic
  1_000_000 / 2_000_000 ranges (`app.rs:2000-2023`, `2946-2955`); one `restore_camera` for hotkey
  and menu (`3148`, `3698`); one status-item renderer for both status bars (`3916-4038`);
  `SettingsPage` instead of magic indices (`app.rs:1235-1241`, `dock.rs:353`); one camera gesture
  id fn (`app.rs:4780`, `dock.rs:481`); `Monitor::read` everywhere (`app.rs:4376-4400`, `937`).
- [ ] **5.10 World helpers**: one `numeric_like` (5 copies: `world.rs:2674`, `2791-2800`,
  `3347-3356`, `6318-6324`, `world_ui.rs:3363`); one component/channel walker for `SetAnimation`,
  `RemoveKey`, `MoveKeys`, `Interpolation`, `curve_effect`, `set_attribute` (`world.rs:866`,
  `1257-1450`, `4582`); one `base_attrs(kind)` (`3975-3993` vs `6056-6074`); enum choices from
  `T::ALL` (`attribute_choices` `1686-1712`); one time-warp integral for `take_frame` / `take_span`
  (`3642-3796`); `static_settings_reference` via `_with`.
- [ ] **5.11 Slim node `gpu` blobs**: store only `branches(kind)`; make the camera framing radius an
  explicit attribute (today a frozen copy of an old formula, `camera_pose` `world.rs:5172-5174`).
  Smaller files and undo snapshots (Q5).
- [ ] **5.12 Small render dups**: `stretch_scale` (`render_prep.rs:655-661` = `770-776`),
  `PreviewProgress` built 4x (`render_service.rs:1071`, `1235`, `1260`, `1337`), slot recycle 3x
  (`534-549`, `636-652`, `867-879`), size validation 3x (`render_service.rs:1090-1105`,
  `render_bench.rs:80-91`, `main.rs:146-157`), WGPU upload blocks 3x (`render_wgpu.rs:697-787`).
- [ ] **5.13 Built-in templates**: `templates.rs:28-50` repeats names / descriptions of
  `presets::ANIMATED` (`presets.rs:14-35`); "Ember Bloom" / "Luminous Web" are JSON in the Templates
  menu but generated code in the gallery CLI. One table, one payload kind.

---

## 6. Dead code (all verified: zero non-test uses)

- [ ] `app.rs`: `fn exists` (`4855`, since the initial import), `let _node` (`1978`), `_ctx` param
  (`1965`), unreachable bookmark-delete `else` (`3833-3835`), shadowed `let d` (`1119`), second
  `remember_material_targets` at `2228` (keep `4832`). `App::tab` / `enum Tab`: finish the dock
  refactor (pass the tab to `browser`), then delete the field.
- [ ] `export.rs`: no-op `ExportController::handle` and its call (`811`, `app.rs:2900`), unused
  `_service` param of `update`, `ExportFormat::Hevc.extension()`, test-only
  `OutputSettings::save_preset_file` (`335`).
- [ ] `render.rs`: `Target::msamples_per_s` (`1328-1338`), `bounces_cap` param (`557`, `770`, `785`,
  `render_prep.rs:342-354`, test `2489`), `Gpu.cuda_unavailable` (test-only override ->
  `#[cfg(test)]`). `render_service.rs`: `RenderPort::is_stopped` (`571`), `Viewport.key` +
  `trace_key` (`761-821`, test `2404`; keep only "seed changed -> drop target").
- [ ] `render_wgpu.*`: `primary_hits` (`wgsl:3375`, `3384`), unused `WORLD_ABI_VERSION` emit
  (`rs:227`), unused post-BGL binding 2 (`rs:416`, `893-896`), `adapt(active_host)` -> `tiles: usize`,
  `out_len` padding (`rs:599`), over-exposed `pub` fields of `WgpuTarget` (`rs:492-504`).
- [ ] `ocio.rs` (copied from exr-view in `bbb97fb`): `Transform::display`, `display_scratch`,
  `display_kernel`, `gain`, `linear`, `key`, `prelude` and the per-transform WGSL build;
  `State.hdr` false branch (always set true); use `ColourPresets::restore` in the UI recall instead
  of the inline copy. Keep a cheap config validation in the colour worker (it is the UI's error
  source). Then drop `#[allow(dead_code)]` on `mod ocio` (`main.rs:31`) and let the compiler list
  the rest. `transfer.rs` (`dial`, `oetf`) dies with `display_kernel`.
- [ ] `world.rs`: `"/transform/rotation"` arms (`1807`, `6517`), `#[cfg(test)]` command variants
  `SetVisible / SetLocked / SetSolo` (`320-334`, `814-834`; tests use `SetAttribute`).
  `world_ui.rs`: `category()` arms `fractal` / `light` / `colour` (`589`, `592`, `605`).
- [ ] `path_convergence.rs` + its `#[allow(dead_code)]` (`main.rs:35`): per Q4.
- [ ] Not dead (verifier): `camera_orbit::integrate` (used `world.rs:5357`),
  `denoise::Processor::new -> Result` (test hook `render.rs:3262`), `process(!enabled)` guard.

---

## 7. Performance (measure first, per the memory rules)

- [ ] **7.1 M. `Frame::snapshot` allocates ~0.25-0.3 GB per 4K viewport publish**
  (`render_service.rs:144-191`: clones `pixels`, `light`, copies `pixels` again into `sdr_bytes`).
  Fix: `snapshot_into(&mut Frame)` with the existing `PresentationPool` for viewport publishes;
  export reads `radiance` only for EXR (`render_service.rs:1479`).
- [ ] **7.2 L.** Two deep `PreparationInputs::matches` per step (`render_service.rs:1598`,
  `render.rs:789-809`); `trace_key` allocations (gone with 6); HDRI / LUT residency thrash when jobs
  alternate; `validate_render_profiles` evaluates the whole document per keystroke
  (`world.rs:481-483`).
- [ ] **7.3 L.** WGPU display does GPU -> CPU -> GPU (`render_wgpu.rs:1186-1201`, `app.rs:4255`);
  same device as egui. Displaying from the GPU buffer removes 2 copies: design change, measure.
- [ ] **7.4 M. WGPU first-use pipeline compile takes minutes** (found in wave 2 integration,
  2026-10-23). `render::tests::cuda_wgpu_renders_mandelbulb_world_and_picks_object`: the first
  `gpu.pick` took 908.8 s with one CPU core at 100 % and the GPU at 0-1 % (driver compiling the
  dynamic-family pick pipeline); the following `step` (trace pipelines) did not finish within the
  remaining run. The same test at e75a8c6 had not finished its pick after ~49 min, but both runs
  were on a heavily loaded machine, so the absolute times prove nothing. A static estimate of the
  inlined WGSL (call graph expanded per call site) puts the new pick at ~73k statements vs ~110k
  at e75a8c6: the module is not larger. Fix directions: a persistent wgpu `PipelineCache`
  (compiled pipelines stored on disk, reused across runs); measure the pick module size (naga ->
  SPIR-V per entry) and `create_compute_pipeline` time per entry; suspect constructs are the
  constant-trip refinement loops in `march` that inline the 28-family dynamic union. **Measure
  on an idle machine.**
  **Root cause (measured 2026-10-23, RTX 3080 Ti, Vulkan, driver 616.64):** naga does not fold
  the constant family argument, so every `*_f{code}` entry carried the whole 28-family
  `family_distance` union (~30.9k inlined statements; reached from 169 call sites per trace
  entry, 18 in pick); pick compile 17-20 s with 1 family, 55 s with 4, 137 s with 8, ~15 min
  with 28. The NVIDIA driver's own disk cache already makes an unchanged SPIR-V recompile ~7 ms.
  **Fix (done, uncommitted):** one shader module per family (`shader_source(family)`,
  `family_distance_wgsl` generated from `FORMULAS[..].wgsl`), the dynamic module only for
  mixed-family worlds; Mixed material x one family now specialized too; pick per module (lazy,
  `homogeneous_family`); every pipeline via `gpu_info::create_compute_pipeline`; `--warmup` opens
  the device + post pipelines only. Shader code paths, inlining and unrolling unchanged.
  **Still slow (2026-10-23):** `cuda_wgpu_renders_mandelbulb_world_and_picks_object` passed but
  took 4946 s on an overloaded machine: one family's estimator is still inlined at ~169 call
  sites per trace pipeline. Next: 7.5.
- [ ] **7.5 M. WGPU trace code size: one estimator call site per loop** (operator approved
  2026-10-23; all tests at the very end, GPU tests only with the operator's go).
  1. [x] Static breakdown (CPU only): session scratchpad `callsites.py TARGET ENTRY.. -- FILES`
     (regex call graph, copies = sum of sites x callee copies). Before: trace_pixel 169, pick_ray
     18. Main multipliers: surface_normal -> endpoint_ray_normal x2 (82), endpoint_ray_normal 41
     (march_leaving 28 + endpoint_normal 6 + cone_start 4 + march_exit 3), march_leaving ->
     march x2, march -> march_sample x3 (+ fractional 2), march_sample 4, stencil_normal 6,
     visibility_leaving -> march_visibility x2, trace_direct 38 (march 14, stencil x2, direct_shade
     12). Target after step 2: ~25.
     **WIP 2026-10-23 (uncommitted -> committed as WIP on fix/plan1 only):** `march_sample` done
     (4 -> 1 site; `world_sample` removed; k==0 always takes its sample, k>0 reach-skip; same
     order and ties). NEXT, in order, each bit-identical:
     a. `march`: one `loop` with `phase` 0 step / 1 accelerated bisection / 2 inside bisection,
        ONE `march_sample` (point = origin + dir * st, st = t or 0.5*(lo+hi)); every phase-0 path
        that loops again does `i = i + 1u` (the old `continue` too); phases 1/2 count `k` against
        HIT_REFINEMENTS and return `best` on `k >= HIT_REFINEMENTS || hi - lo <= 0.5*best.eps`.
     b. `march_leaving` / `visibility_leaving`: a 2-pass loop, origin then retreat, one call.
     c. `surface_normal`: `for k in 0..2` filling `array<vec3<f32>,2>` from ONE
        `endpoint_ray_normal(.., whole + k)`.
     d. `stencil_normal`: per halving, a loop over 6 taps j: axis j/2, `off[axis] = h`,
        `p = select(center + off, center - off, (j & 1u) == 1u)` (exactly the old +/- forms,
        signed zeros included), values into `array<f32,6>`, then the old checks.
     e. Re-run callsites.py (record), naga validate via the GPU-free test, clippy; GPU tests
        only with the operator's go (see step 4).
     **Done 2026-10-23:** a-d plus `fractional_march_distance` (2 sites -> 1, same order as
     `march_sample`). Copies of `family_distance`: trace_pixel 169 -> 22, pick_ray 18 -> 3.
     Remaining multipliers (small, left as is): march_exit 3 sites, trace_direct stencil x2,
     direct_shade -> direct_shadow x2, surface_normal's own endpoint_normal next to the two
     re-marched ones.
  2. [x] For the largest callers, restructure so each loop has ONE estimator call site, with
     bit-identical math (same points, same order, same results): the normal stencil (6 taps x
     halvings -> a loop over the 6 offsets), `march` (step + two bisection loops -> one loop
     with a phase state), shadow / AO / light visibility marches likewise. No unrolling or
     inlining is forbidden; no algorithm changes. Mirror nothing in CUDA (fractal-rs unchanged).
  3. [x] Re-count (step 1) and record before/after call sites (above).
  4. [ ] End: GPU-free suite + clippy; then, with the operator's go and an idle machine: the
     WGPU world test (compile time), `parity_tests::cuda_wgpu_primary_guides_agree_per_pixel`,
     and one A/B `--world-bench --backend wgpu --case <one case>` before/after (the "before"
     pipelines are in the driver cache from the earlier test). Any slowdown -> that change is
     dropped.
  Fallbacks, not planned: SPIR-V `DontInline` on the estimator via passthrough (may cost call
  overhead, needs a probe); wavefront split of the megakernel (CUDA parity, big refactor).
- [ ] **7.6 L. CUDA per-family world kernels** (fractal-rs): only `world_fast_bulb` is
  specialized; homogeneous worlds of other families run the universal dispatcher. Unmeasured.

---

## 8. Docs and comments

**Done 2026-10-23:** most items were already fixed by the waves; left: main.rs top doc (families
from `formulas::FORMULAS`, stale `mod export` comment), `decode_json` doc (same JSON value, not
bytes), README / `test_gpu.rs` "CUDA worker" -> render worker. Not found (no longer present):
`bounces_cap`, the `render.rs` false comment, the `world.rs` doc on the wrong fn.
- [x] `main.rs:2-8` ("eight families", 3 of 8 CLI modes), `main.rs:34` false "Integrated into CUDA";
  render worker named "CUDA worker" / thread `frac-cuda` though it drives both backends
  (`render_service.rs:1-2`, `401`, `410`, `601`, `export.rs:1330`, `render.rs:1-2`, `preview.rs`);
  `render.rs:762` `bounces_cap` doc; `render.rs:1330` false comment;
  `render_wgpu.wgsl:1` "1:1 port" (false until section 2 is done); slot comments
  `formulas.wgsl:3`, `formulas.rs:9`, `2615`, `2644` (real block 69..=125; add a const assert);
  `ParamMeta.order` doc (`formulas.rs:58`); `world.rs:2653` doc on the wrong fn; `/end` hint is
  exclusive (`world.rs:2523`); `world.rs:5933` "8 families"; `decode_json` byte-for-byte claim;
  ocio.rs module doc and `Sel` doc describe exr-view; manipulator doc on `draw_overlays`
  (`app.rs:1268`); `app.rs:151` grid count; `xtask` has no module doc.

---

## Refuted or corrected by the verifiers (do not act)

- Manipulator `targets` / `edited` mismatch (`app.rs:1421-1470`): **refuted**. `manip_frame` already
  runs `take_trs` and the same paths as `authored_trs`; a failing node is skipped before
  `targets.push`. Kept only as an L smell: build one record list so the two cannot drift.
- `HevcSink::finish` dropping output on cancel: **refuted** (video-io documents that an unfinished
  encoder publishes nothing).
- "MaxCLL / MaxFALL not measured": **wrong**, video-io measures them (3.6 corrected).
- `CancelExport` global: unreachable today (single export) - L.

## Not done in this pass

- `cargo clippy --all-targets -- --force-warn dead_code --force-warn unused` (overrides the
  `#[allow]`s, lists what the compiler sees as dead): **run 2026-10-23**, the only dead code is
  `path_convergence.rs` (5 items), the open Q4 draft; nothing else.

## Execution state (2026-10-23)

**Current:** wave 1 = e75a8c6, wave 2 = 621baa6 on `fix/plan1`. GPU-free suite green (333/0/7),
clippy `-D warnings` and fmt clean. **Wave 2's render changes are NOT verified on the GPU:** the
WGSL rewrite (CUDA ports, two-pass tally, glass absorption, generated entries) has not run, and
`parity_tests::cuda_wgpu_primary_guides_agree_per_pixel` has never run (its tolerances are a
guess). Blocked by 7.4: every WGPU test first compiles pipelines for 15+ min. Order: (1) CUDA-only
`cuda_` tests (`--skip wgpu`): done, 31/31; (2) 7.4 fixed by per-family shader modules (see 7.4;
shader code paths, inlining and unrolling unchanged); (3) the WGPU and parity tests, tolerances
tuned from measured numbers. Wave 2 does not merge to main before (3).
window.rs migrated onto `soluwell-window` (egui-widgets-rs f5c71b3 added `App::presented`,
Locked->Confined grab, surface clamp); CUDA has no per-family world kernels except
`world_fast_bulb` (fractal-rs handoff, unmeasured).

Approved by the operator ("unification, dedup, SSOT everywhere, fix everything systemically").
Decisions taken by the lead: Q1 delete migrations after a one-time conversion (`#[ignore]` test
`convert_documents_once`, backups `*.bak`); Q2 keep `conn`, one resolver; Q3 typed admission
error; Q4 deferred to PLAN.md item 12 (comments fixed only); Q5 byte-budget undo + slim `gpu`
blobs. Presets take a typed `Formula` (`Formula::from_code` only at boundaries).
Branch `fix/plan1`, checkpoint d1c9cae. Wave 1 agents (render-core, shaders, export-colour,
world-doc) were PAUSED mid-work on the operator's order (usage limit): the tree is uncommitted
and not type-checked. Their status files and handoffs: session scratchpad `status-*.md`, brief
`brief.md`. Resume: finish wave 1, apply handoffs, `cargo check` via the machine-wide lock
(`python C:/projects/projects.rust.cg/cglibs/.buildlock/buildlock.py cargo ...`), then wave 2
(app shell, WGSL entry-point generator + parity test, migration deletion after conversion).

## Execution order (after approval)

1. Clippy with forced dead-code warnings (baseline list).
2. Section 1 (robustness) as one branch, then 2.0 parity test, then 2.1-2.9.
3. Section 3 (export / colour SSOT), section 4, section 5 in dependency order
   (5.3 before 5.1; 5.7 together with 3.1), section 6 after its owners are refactored.
4. One targeted test run at the end of each branch (`cargo oxide test -- --release -- <filter>`),
   one independent review per finished branch, screenshots of viewport / exports.
5. Q1-Q5 answers gate their items.
