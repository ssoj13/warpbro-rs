# WarpBro: agent notebook

> Instructions to myself: keep this file current. Record here every significant fact about the
> crate (what it is, how it works, where things live, traps, open TODOs). Update it in place,
> compress outdated parts into a short changelog, delete what no longer applies. Line anchors are
> as of commit `4cfa5eb` (2026-10-22); re-check them with grep before trusting a number.
> Backlog: [PLAN.md](PLAN.md). Current bug hunt and fix plan: [plan1.md](plan1.md).
> Mermaid versions of the diagrams: [DIAGRAMS.md](DIAGRAMS.md).

## What it is

WarpBro is an egui desktop browser / animator for path-traced 3D fractals (28 DE families).
Two compute backends render the same packed scene ABI (`fractal_core::params`, `P_COUNT`):

- **CUDA**: Rust kernels from the fractal-rs git dependency (`fractal-kernels`), compiled to PTX by
  the cuda-oxide rustc backend and linked in with `#[cuda_module]`. Classic families 0..8 only.
- **WGPU**: a WGSL port (`src/render_wgpu.wgsl` + `src/formulas.wgsl`) on the shared
  `gpu_info::shared_device()`. All 28 families. Extended families 8..27 exist only here.
  One shader module per family plus a dynamic one (`render_wgpu::shader_source(family)`): a
  family module's `family_distance` is that family's estimator alone, generated from
  `formulas::FORMULAS[..].wgsl`; the dynamic module switches over all 28 (mixed-family worlds
  only). Reason: naga does not fold a constant family argument, so the old one-module design
  gave every "specialized" pipeline the whole union, inlined by the driver at ~170 call sites
  (minutes per pipeline). Pipelines go through `gpu_info::create_compute_pipeline` (disk cache).

Build only through cuda-oxide (`cargo oxide build`, `python bootstrap.py b`); type-check with
`FRAC_ALLOW_PLAIN_CARGO=1 cargo check --all-targets` or `cargo clippy` (build.rs skips the backend
check for clippy / rust-analyzer).

The machine is shared: every cargo build / check / clippy / test goes through the machine-wide
lock, `python C:/projects/projects.rust.cg/cglibs/.buildlock/buildlock.py <cmd>` (its `SLOTS`
is set by the operator; never nest it; `d:\temp\buildsem.ps1` only forwards to it). Build once after the edits are finished, never
while still editing. Tests need `--release` (debug device codegen of fractal-kernels fails):
`buildlock.py cargo oxide test -- --release --locked -- <filters>`. libtest `--skip gpu_` is a
substring match and also drops GPU-free tests named `wgpu_*` / `*_gpu_*`: run those by exact
name. CUDA / WGPU tests and benches only with the operator's go.

## Module map

| Area | Files | Role |
|---|---|---|
| Entry | `main.rs` | CLI modes (`--gallery`, `--bench`, `--world-bench`, `--animated-fixtures`, `--warmup`), else `app::run` |
| Window | `window.rs`, `window_shot.rs` | adapter onto the shared `soluwell-window` host (egui-widgets-rs): private present queue, `max_fps` = `gui_fps`, HDR window shots via `App::presented` |
| App shell | `app.rs` (7k), `dock.rs`, `hotkeys.rs`, `file_dialogs.rs` | `App` owns everything; per-frame `App::ui`; dock panels; scoped hotkeys |
| Document | `world.rs` (6.5k), `world_ui.rs` (6k), `takes*.rs`, `camera_*.rs`, `animation.rs` | Playa-graph world document, commands, undo, evaluation; outliner / timeline / Attribute Editor; recorded takes |
| Scene | `scene.rs`, `formulas.rs`, `presets.rs`, `templates.rs`, `materials.rs`, `palette.rs`, `environment.rs` | Evaluated `Scene` (typed), `Scene::pack` to the kernel ABI, formula registry, built-in presets/templates |
| Render | `render.rs` (4.9k), `render_prep.rs`, `render_wgpu.rs`, `render_service.rs`, `render_profiles*.rs`, `denoise.rs` | `Gpu` (both backends), preparation cache, worker thread, profiles -> `EffectiveRender`, OIDN |
| Preview | `preview.rs`, `overlay.rs`, `material_gallery.rs` | Cached timeline playback, viewport wires/bounds, material thumbnails |
| Colour / IO | `color.rs`, `ocio.rs`, `transfer.rs`, `export.rs` (3.2k), `exr_io.rs`, `io_service.rs`, `fs_name.rs` | OCIO display pipeline, export coordinator + writers, file worker |
| Unused draft | `path_convergence.rs` | Roulette / clamp policies, not wired (belongs in fractal-rs) |

## Ownership and per-frame loop (app shell)

```
main.rs:main --> app::run --> window::run
  soluwell_window::run: ctx.run_ui -> Window::ui -> App::ui -> paint -> present -> Window::presented (window shots)

App owns:
  world: WorldEditor (world.rs:356)   document + undo/redo     <- the ONLY authored state
  scene: Scene = document.snapshot(playhead)                  <- evaluated per-frame scratch copy
  renderer: RenderService (render_service.rs:393, worker thread owns Gpu)
  io: IoService (io_service.rs:102, file worker)   preview: PreviewController   export: ExportController
  colour: ocio::State (+ copies of the selection, see plan1.md: colour SSOT)

App::ui per frame:
  sync_preview -> world_ui.advance (playhead) -> refresh_scene (snapshot if (graph id, revision, frame) changed)
  hotkeys::consume (GLOBAL table first, then the panel scope)
  before = snapshot_before.take()                                   -- edit transaction opens
  poll_events <- renderer (Frame | Picked | Thumbnail | Error) ; io polls
  panels: viewport (fly/orbit mutate self.scene.camera; step_viewport -> ViewportRequest;
          manipulate -> world.execute_edit(gesture); viewport_click -> Pick) | AE | timeline | export_ui ...
  diff self.scene vs before: camera -> world.navigate_camera ; render/colour -> world.edit_snapshot
  world.finish_edit_unless(held gesture)                            -- edit transaction closes
  save_settings -> io
```

Rule: `self.scene` is scratch. Only the end-of-frame diff turns its mutations into document edits.
Anything that edits `self.scene` outside that window, or keeps a parallel copy of document state, desyncs.

## World document and edits

```
FILE --io_service::decode_scene--> WorldDocument::decode_json (world.rs:5602) --hydrate_schema (5639)
NEW  --WorldDocument::from_scene--> seed settings nodes + Fractal, Camera, Sun, Environment, Material

WorldDocument: graph (playa_graph::SubnetFile: nodes{uuid -> type, parent, host: Attrs, gpu: JSON template,
               discrete, material, take}), bus_slots{world, layer_order, output_render, viewport/output settings},
               active_camera / active_environment, first/last/fps, marks; takes + camera_slots outside undo.

EDIT: WorldCommand (world.rs:188) -> WorldEditor::execute_edit(cmd, gesture) (415) -> apply_edit (471)
      -> apply (812) -> set_attribute (4550) -> validate_render_profiles (3361, Err = rollback)
      -> record_edit (453): gesture = one pending step, else undo.push(before)
EVAL: snapshot(frame) (5393) = snapshot_with_render_profile(output_render, frame, View::Output) (5426)
      attribute_value (4390) / world_matrix (5122) <- take_trs <- take_frame (time warp)
      effective_render (3104) = RenderSettings profile + ViewportSettings look
```

## Render dataflow

```
WorldDocument --snapshot_with_render_profile--> Scene --Scene::pack--> [f32; P_COUNT]
  viewport: app.rs step_viewport -> ViewportRequest -> RenderService::request_viewport (single-slot mailbox)
  export:   export.rs coordinate_export -> Command::RenderExport{reply}     thumbnails: Command::Thumbnail
  preview:  preview.rs -> BeginPreview / CacheViewport / Seek
run_worker (render_service.rs:1126) owns Gpu:
  Gpu::step_with_display (render.rs:779)
    resolve_backend_for_scene (295): Auto -> CUDA, WGPU for extended families / no CUDA
    PreparationCache::get (render_prep.rs:374) -> PreparedScene (WorldUpload: params, objects, masks, lights)
    trace: CUDA kernels (world / world_fast / world_direct / per family) | WgpuEngine launch_* (render_wgpu.rs:
           trace_entry -> TraceEntry {world, shade, family} -> the family's module, compiled on first use)
    adapt (active tiles) / tally (unresolved) / process_denoise (OIDN, denoise::State::due cadence)
    display: tonemap -> ColorPipeline::apply_into (OCIO) -> Target.light / Target.pixels
  Frame::snapshot (render_service.rs:144) -> publish_frame -> UI texture | ExportFrame -> writer
```

## Colour and export

```
Scene.colour (ocio::Sel) -> color::ColorPipeline (render worker) -> light (linear 709 display) + sdr codes
ExportController::start (export.rs:849) -> coordinate_export thread -> RenderExport per frame
  -> ExportWriter thread: EXR (exr_io::write_exr, ACEScg radiance) | PNG/JPEG/TIFF (video_io::still)
     | video (HevcSink -> video_io::native encoders) | PNG sequence + ffmpeg (video_io::video::encode)
IoService thread: settings / scenes / bookmarks (tmp + rename), frame and window shots, templates, takes
```

## Invariants and traps

- No legacy data compatibility (operator rule): no migrations, serde aliases or compat defaults.
  `hydrate_schema` currently violates it (plan1.md decides).
- WGSL is a hand port of the CUDA kernels; there is no per-family CUDA vs WGPU parity test yet, and
  several families diverge (plan1.md). Any kernel change must land in both.
- `denoise::State::due` mutates the cadence: call it exactly once per step.
- Render errors on the worker must reach the requester as a typed event; never leave a job waiting.
- Undo granularity is one open edit per gesture id; text fields and drag values need a gesture.
- egui: nested `ui.input` inside `hotkeys::active` deadlocks; read `active` first.
- Many sources are CRLF; scripted replacements must match `\r\n`.
- Tests: `cuda_*` need a CUDA device, `gpu_*` another GPU API; hosted CI skips both.

## TODO

- [ ] Execute plan1.md after approval (bug hunt 2026-10-22).

<!-- gitnexus-rs:start -->
# GitNexus-rs — Code Intelligence

This project is indexed by gitnexus-rs as **warpbro-rs** (3512 symbols, 9699 relationships, 300 execution flows). Use the gitnexus-rs MCP tools to understand code, assess impact, and navigate safely.

> Call `graph_status` when freshness matters. Use `reanalyze` (incremental) or `gitnexus-rs analyze` (full). `detect_changes` does **not** re-index.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `detect_changes({scope: "all"})` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `context({name: "symbolName"})`.
- **Check graph freshness** with `graph_status` before trusting query/impact results on a repo you have been editing.
- **Update the graph** with `reanalyze` (incremental) or `gitnexus-rs analyze` (full). `detect_changes` does **not** re-index.
- **Uncommitted edits** while commit is fresh: `reanalyze` with `scope: "unstaged"`, or run `gitnexus-rs watch` in a terminal.

## Never Do

- NEVER edit a function, class, or method without first running `impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `rename`, which understands the call graph (it previews by default).
- NEVER commit changes without running `detect_changes` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/warpbro-rs/context` | Codebase overview |
| `gitnexus://repo/warpbro-rs/clusters` | All functional areas |
| `gitnexus://repo/warpbro-rs/processes` | All execution flows |
| `gitnexus://repo/warpbro-rs/process/{name}` | Step-by-step execution trace |

## Skills

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus-rs-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus-rs-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus-rs-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus-rs-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus-rs-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus-rs-cli/SKILL.md` |

<!-- gitnexus-rs:end -->
