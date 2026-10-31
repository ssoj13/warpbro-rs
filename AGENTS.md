# WarpBro: agent notebook

> Instructions to myself: keep this file current. Record here every significant fact about the
> crate (what it is, how it works, where things live, traps, open TODOs). Update it in place,
> compress outdated parts into a short changelog, delete what no longer applies. Line anchors are
> as of commit `4cfa5eb` (2026-10-22); re-check them with grep before trusting a number.
> Backlog & active roadmap: [PLAN.md](PLAN.md).
> Mermaid versions of the diagrams: [DIAGRAMS.md](DIAGRAMS.md).

## What it is

WarpBro is an egui desktop browser / animator for path-traced 3D fractals (28 DE families) built on
top of the shared `soluwell-studio` widget (`egui-widgets-rs/crates/soluwell/studio`) and the
shared `animation-rs` animation/attribute substrate (`animation` crate: `time`, `anim`, `attrs`
(`AttrValue`, `Attrs`), `transform`, `keys`, `curves`, re-exported via `soluwell_studio::world`).

`soluwell-studio` knows nothing about fractals or fractal formulas (`soluwell-studio` has zero
`fractal-*` dependencies; shared studio ABI, environment, and Standard Surface material presets live
in `render-rs`: `pt_integrator::studio_abi`, `render_env::studio`, and
`standard_surface_bsdf::presets`). WarpBro plugs in as a hot-pluggable
`soluwell_studio::StudioModule` (`WarpBroFractalModule` in `src/engine.rs`, passed via
`StudioConfig { modules: vec![Arc::new(engine::WarpBroFractalModule)], .. }`) and registers:
- **`"Fractal"` scene node (`src/fractal_node.rs`, `FractalNodeSpec`)**: implements
  `NodeTypeSpec` v2 (`payload_kind() -> NodePayloadKind::SdfField`), registered via
  `WarpBroFractalModule::node_specs` and `soluwell_studio::node_spec::register_node_type`. When a
  `"Fractal"` node is present and visible in the graph, it evaluates into `scene.objects` and
  renders; when absent or hidden, `scene.objects` is empty (`P_OBJECT_COUNT == 0`) and
  `soluwell-studio` renders only the 3D studio environment (sky/HDRI + 3D grid + gizmos).
- **Two compute backends (`src/engine.rs`, `WarpBroComputeEngine`)** rendering the same packed scene
  ABI (`fractal_core::params`, `P_COUNT`, all 28 DE families `0..27`):
  - **CUDA**: Rust kernels from the `fractal-rs` git dependency (`fractal-kernels`), compiled to PTX
    by the `cuda-oxide` rustc backend and linked in with `#[cuda_module]`. All 28 families.
  - **WGPU**: WGSL port (`src/formulas.wgsl` injected into `soluwell-studio-render` via
    `render_wgpu::set_custom_shader_hook`) on the shared `gpu_info::shared_device()`. All 28
    families. One shader module per family plus a dynamic 28-way module for mixed-family worlds.
- **Built-in templates (`src/templates.rs`, `src/presets.rs`) and 38 gallery presets
  (`Scene::fractal_gallery`)**: injected into `soluwell_studio::StudioConfig` and
  `soluwell_studio::templates::register_builtins`.

Build only through cuda-oxide (`cargo oxide build`, `python bootstrap.py b`); type-check with
`FRAC_ALLOW_PLAIN_CARGO=1 cargo check --all-targets` or `python bootstrap.py c` (build.rs skips the
backend check for clippy / rust-analyzer). Tests need `--release` (debug device codegen of
`fractal-kernels` fails): `cargo oxide test -- --release --locked -- <filters>`. GPU-free suite:
`--skip cuda_ --skip gpu_` (libtest `--skip` is a substring match; device tests start with
`cuda_` / `gpu_` and no other test name may contain them, enforced by
`test_gpu::device_prefixes_only_start_test_names`).

## Module map

All shared studio UI, dock panels, Outliner/Timeline/Attribute Editor/Curve Editor, `appbus`
two-layer nervous system (`EventBus` + `StateSlot<T>`), loopback HTTP control, `WorldDocument`
graph/undo, `RenderService`, `PreparationCache`, OCIO display pipeline, OIDN denoiser,
`PreviewController`, `ThumbnailCache`, and `ExportController` live in
`egui-widgets-rs/crates/soluwell/studio` (`soluwell-studio`, `soluwell-studio-core`,
`soluwell-studio-scene`, `soluwell-studio-render`), backed by `render-rs` (`pt-integrator`,
`render-env`, `standard-surface-bsdf`) and `animation-rs` (`animation` crate: `time`, `anim`,
`attrs`, `transform`, `keys`, `curves`). `warpbro-rs` keeps a clean direct dependency graph
(`soluwell-studio`, `animation`, `fractal-core`, `fractal-kernels`,
`cuda-core`/`cuda-host`/`cuda-device`, `curves`, `egui`, `soluwell-config`, `glam`, `log`,
`anyhow`, `env_logger`, `serde`, `serde_json`, `sha2`) after dropping unused direct `playa-*`,
`video-io`, `vfx-ocio`, `exr-core`, `wgpu`, `bytemuck`, `gpu-info`, etc. dependencies.
`warpbro-rs/src/` contains only the 10 fractal domain files:

| Area | Files | Role |
|---|---|---|
| Entry & CLI | `main.rs` | `engine::init_warpbro()`, CLI modes (`--gallery`, `--bench`, `--world-bench`, `--animated-fixtures`, `--warmup`), else `soluwell_studio::run_with(StudioConfig { modules: vec![Arc::new(engine::WarpBroFractalModule)], ... })` |
| Compute engine & module | `engine.rs` | `WarpBroFractalModule` (`StudioModule`), `init_warpbro()`, `WarpBroComputeEngine` (dual CUDA `fractal-kernels` + 28-family WGPU `WgpuEngine` implementing `soluwell_studio::render::ComputeEngine`), `warpbro_shader_hook` |
| `"Fractal"` node & ABI | `fractal_node.rs` | `FractalNodeSpec` (`NodeTypeSpec` v2, `NodePayloadKind::SdfField`), 28-variant `Formula`, `FractalSceneExt` over `Scene::custom`, 38-preset `fractal_gallery()`, and `pack_object` (`P_COUNT` kernel ABI) |
| Formula registry & WGSL | `formulas.rs`, `formulas.wgsl` | Declarative 28-family metadata table (`FORMULAS`), extended parameter structs/packing (`EXT_SLOTS`), and 28-family WGSL distance estimators |
| Templates & presets | `templates.rs`, `presets.rs` | 14 built-in templates (`BUILTINS`, `ANIMATED = 0..5`) and procedural 250-frame animated `WorldDocument` builders |
| Benchmarks & parity tests | `render_bench.rs`, `parity_tests.rs`, `test_gpu.rs` | Evaluated world/family A/B benchmarks, CUDA vs WGPU per-pixel/ray-beam parity suite, and `cuda_`/`gpu_` test-prefix guard |

## Ownership and per-frame loop (app shell)

```
main.rs:main --> app::run --> window::run
  soluwell_window::run: ctx.run_ui -> Window::ui -> App::ui -> paint -> present -> Window::presented (window shots)

App owns:
  world: WorldEditor (world.rs:356)   document + undo/redo     <- the ONLY authored state
  scene: Scene = document.snapshot(playhead)                  <- evaluated per-frame scratch copy
  renderer: RenderService (render_service.rs:393, worker thread owns Gpu)
  io: IoService (io_service.rs:102, file worker)   preview: PreviewController   export: ExportController
  colour: ocio::State (UI; the document owns the selection, refresh_scene mirrors it with set_sel)
  event_bus: playa_events::EventBus; UI and REST publish app_control::Event<UiCommand>

App::ui per frame:
  sync_preview -> world_ui.advance (playhead) -> refresh_scene (snapshot if (graph id, revision, frame) changed)
  poll_control starts HTTP transport -> handle_events drains the App-owned Playa EventBus
  hotkeys::consume (GLOBAL table first, then panel scope) -> emit_app -> handle_events
  poll_events <- renderer (Frame | Picked | Thumbnail | Error) ; io polls
  panels read evaluated state and edit transient widget drafts -> typed UiCommand intents
  handle_events at explicit frame boundaries -> dispatch_command:
    Action::App -> handle_app_action; Action::World -> world_events::apply;
    Action::Take -> handle_take_action; Action::Query -> control_query
  accepted actions -> WorldEditor / navigate_camera / worker requests -> refresh_scene
  record_take / preview intents -> bus; WorldAction::FinishEdit {held} -> bus -> handle_events
  save_settings -> io
```

Rule: `self.scene` is evaluated scratch; widget drafts are transient. Widgets, hotkeys and REST
publish the same `app_control::Event<UiCommand>` to App's Playa EventBus. `UiCommand` carries
`Action::{App, World, Take, Query}` plus captured document/frame and, for camera intents, active-camera
identity. Deferred dispatch validates that context before applying state; delayed actions for a closed
document or replaced camera fail instead of editing a different target. Concrete node/take/material
targets are captured with the intent. HTTP commands default to current dispatch-time context.

`app_control` supplies HTTP parsing, typed decoding and one-shot acknowledgements, not another
application-command queue. Its parser `mpsc` carries HTTP sockets only; reply `mpsc` carries results
only. Both UI and HTTP use Playa `try_emit`: a full EventBus rejects admission without evicting an
older accepted action. Apply mutations in the central handlers, never in a widget or an immediate
EventBus subscriber. Draft deltas merge onto current accepted state rather than replacing unrelated
intervening edits.

`emit_retained` holds a rejected terminal event, drains this same EventBus and retries admission;
it does not apply the action directly or create a second command queue. Scene/Bookmark completion
DTOs use this path; Take imports also retain the request's original document. Scene-save identity
and document changes are staged until the I/O worker accepts the request, and admission errors
propagate to HTTP. The dock renders an immutable layout clone while App keeps authoritative layout
ownership, so dispatch cannot save a temporary empty tree or overwrite a newly restored layout.
Picks evaluate the accepted frame and effective viewport profile after cancelling playback/preview;
late results are guarded by document, active camera, frame and revision.

## World document and edits

```
FILE --io_service::decode_scene--> WorldDocument::decode_json (strict: node_shape, check_format, validate)
NEW  --WorldDocument::from_scene--> seed settings nodes + Fractal, Camera, Sun, Environment, Material

WorldDocument: graph (playa_graph::SubnetFile: nodes{uuid -> type, parent, host: Attrs, gpu: JSON template,
               discrete, material, take}), bus_slots{world, layer_order, output_render, viewport/output settings},
               active_camera / active_environment, first/last/fps, marks; takes + camera_slots outside undo.
               Animation & attribute primitives (`time`, `anim`, `attrs` (`AttrValue`, `Attrs`), `transform`,
               `keys`, `curves`) come from `animation-rs` (`animation` crate) and are re-exported via
               `soluwell_studio::world`.

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
run_worker (render_service.rs; line anchors above predate the compiler/cache additions) owns Gpu:
  pipeline_ready -> WgpuShared::ready -> CompileTicket -> warpbro-shader-compiler
    pending: keep the render job; compile once, retain errors, wake app when finished
    ready: continue on the sole render worker (no second render worker)
  Gpu::step_with_display (render.rs:779)
    resolve_backend_for_scene (295): Auto -> CUDA, WGPU for extended families / no CUDA
    PreparationCache::get (render_prep.rs:374) -> PreparedScene (WorldUpload: params, objects, masks, lights)
    trace: CUDA kernels (world / world_fast / world_direct / per family) | WgpuEngine launch_* (render_wgpu.rs:
           trace_entry -> TraceEntry {world, shade, family} -> the family's module, compiled on first use)
    adapt (active tiles) / tally (unresolved) / process_denoise (OIDN, denoise::State::due cadence)
    display: tonemap -> ColorPipeline::apply_into (OCIO) -> Target.light / Target.pixels
             (skipped while Target.display_key matches: exposure, saturation, reinhard, Sel and
             the config load Ocio::serial; transforms cached per selection + load)
    end of step: Gpu::stamp waits for the backend's work, then records last_ms / last_spp:
             Batch::Budget sizes the next batch from that GPU time (launches are async; stamping
             the submit time made thousands-of-samples batches and multi-second stalls)
  Frame::snapshot (render_service.rs:144) -> publish_frame -> UI texture | ExportFrame -> writer
```

The status bar shows `Preparing shader · <seconds>s · <entry>` during WGPU pipeline preparation;
Auto scenes using extended families are labelled WGPU. This separates compilation waits from sample
progress; it does not promise a navigation frame rate. `render_diagnostics` exposes active/recent
compiles and thumbnail-cache counters through the [local REST API](docs/reference/rest-api.md).

All gallery, bookmark and material thumbnails use `thumbnail_cache::ThumbnailCache` with `DiskCache`
at `<profile>/cache/thumbnails` (`~/.warpbro` or `WARPBRO_HOME`), bounded to 256 MiB by oldest-write
eviction. Entries contain CPU display pixels, with a versioned binary header, extent and SHA-256
validation; writes publish through a temporary file and rename. Invalid entries regenerate.
Content-addressed keys include the exact packed scene ABI, transforms/masks/lights, palettes, quality,
backend, SPP/seed, the actual resident environment-map fingerprint, OCIO processor cache IDs and the
renderer/source fingerprint (including Cargo.lock). Labels, document UUIDs and GPU handles are not
pixel identities. These are thumbnail entries, separate from the RAM-only timeline preview cache.
Completed thumbnail jobs retain completion under event-queue backpressure; cached-hit and failure
terminal events have a pending slot rather than disappearing when the event queue is full.

## Colour and export

```
Scene.colour (ocio::Sel) -> color::ColorPipeline (render worker) -> light (linear 709 display) + sdr codes
ExportController::start (export.rs:849) -> coordinate_export thread -> RenderExport per frame
  -> ExportWriter thread: EXR (exr_io::write_exr, ACEScg radiance) | PNG/JPEG/TIFF (video_io::still)
     | video (HevcSink -> video_io::native encoders) | PNG sequence + ffmpeg (video_io::video::encode)
IoService thread: settings / scenes / bookmarks (tmp + rename), frame and window shots, templates, takes
```

## Invariants and traps

- No legacy data compatibility (operator rule): no migrations, serde aliases or compat defaults
  (the migrations were removed after converting the shipped templates once).
- WGSL is a hand port of the CUDA kernels; `parity_tests::cuda_wgpu_primary_guides_agree_per_pixel`
  compares them but has not run on a GPU yet (tolerances unmeasured). Any kernel change must land
  in both.
- `denoise::State::due` mutates the cadence: call it exactly once per step.
- Render errors on the worker must reach the requester as a typed event; never leave a job waiting.
- Diagnostic wake clones the shared wake handle before calling it; never wake the UI while holding
  the diagnostics wake mutex.
- A pending shader ticket must not consume the render worker with an inline driver compile; failures
  remain terminal for that ticket. The compiler is separate from the sole owner of render targets.
- Pipeline-cache persistence is driver-specific. The gpu-info dependency is pinned to the guard fix
  `f845ac889738e159191eb9f71cfabc8831ce134f`, which corrects cache persistence after pipeline creation.
  Verify accepted cache contents and reuse across restart; a large old disk file alone proves neither.
- UI/REST admission must use reliable Playa EventBus `try_emit` (upstream `72ed08b`); do not add a
  parallel application queue or fall back to an emitter that evicts accepted actions on overflow.
- Undo granularity is one open edit per gesture id; text fields and drag values need a gesture.
- egui: nested `ui.input` inside `hotkeys::active` deadlocks; read `active` first.
- Many sources are CRLF; scripted replacements must match `\r\n`.
- Tests: `cuda_*` need a CUDA device, `gpu_*` another GPU API, and no other test name contains
  them (`test_gpu::device_prefixes_only_start_test_names`); hosted CI skips both.

## TODO

- [x] Migrate `warpbro-rs` onto `soluwell-studio` with extensible `WorldKind::Custom("Fractal")` (`FractalNodeSpec`) and pluggable `ComputeEngine` (`WarpBroComputeEngine`).
- [ ] CUDA / WGPU parity verification & zero-spill launch bounds tuning across all 28 families (see [PLAN.md](PLAN.md) Section A).
- [ ] Render pacing and the GPU display path stages 0..4 (two-batch ring, async readbacks, GPU tonemap/OCIO, zero-copy present; see [PLAN.md](PLAN.md) Section B).
- [ ] oiio-rs CUDA OCIO on the film (`plan-cuda-oxide.md` in `oiio-rs`, feeds into Stage 3).
- [ ] Complete `.ok()` sweep part 2 (`nodes() -> Result`) & resolve operator decisions Q2 `conn` and Q4 `path_convergence.rs` (see [PLAN.md](PLAN.md) Section C).

## Changelog

- **2026-10-30**: Synchronized `warpbro-rs` with `soluwell-studio` (`NodeTypeSpec v2`, `StudioModule`, `appbus` two-layer nervous system) and `render-rs` shared studio SSOT (`pt_integrator::studio_abi`, `render_env::studio`, `standard_surface_bsdf::presets`): added `WarpBroFractalModule` (`StudioModule`) in `src/engine.rs`, wired `StudioConfig::modules` in `src/main.rs`, and implemented `FractalNodeSpec::payload_kind() -> NodePayloadKind::SdfField` in `src/fractal_node.rs`.
- **2026-10-29**: Documented `animation-rs` (`animation` crate: `time`, `anim`, `attrs` (`AttrValue`, `Attrs`), `transform`, `keys`, `curves` re-exported via `soluwell_studio::world`) across `soluwell-studio` and cleaned up the `warpbro-rs` dependency graph by dropping unused direct `playa-*`, `video-io`, `vfx-ocio`, `exr-core`, `wgpu`, etc. dependencies.
- **2026-10-28**: Consolidated all root plan/handoff/bug documents into `PLAN.md`. Added inline `Backend` (`Auto / CUDA / WGPU`) selector across RenderSettings bindings, synced OIDN `WorkContext`, updated `CUDA_OXIDE_REV`, and locked OxideAV git dependencies in `Cargo.lock`. All 28 families ported to CUDA with 234 specialized kernel entries (`fractal-rs` `5a21813e`).
- **2026-10-27**: Migrated `warpbro-rs` onto `soluwell-studio` (`egui-widgets-rs` `a8b4e935`). Removed all fractal/formula knowledge from `soluwell-studio`, extracted `"Fractal"` into `src/fractal_node.rs` (`FractalNodeSpec`) and the dual CUDA+WGPU backend into `src/engine.rs` (`WarpBroComputeEngine`), and deleted 45 duplicated UI/scene/render files and `crates/app-control`.

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
