# WarpBro diagrams

Mermaid views of the main system dataflows, reflecting the `soluwell-studio` architecture (`5afff8a`).
Companion architecture notebooks: [PLAN.md](PLAN.md) and [AGENTS.md](AGENTS.md).

---

## 1. Ownership & Application Shell

```mermaid
flowchart TD
  main[main.rs] -->|engine::init_warpbro| reg[Register FractalNodeSpec<br/>& WarpBroComputeEngine]
  main -->|soluwell_studio::run_with| win[soluwell_window::run]
  win -->|ctx.run_ui| app[soluwell_studio::StudioApp]

  subgraph StudioApp[StudioApp - soluwell-studio]
    world[WorldEditor<br/>SubnetFile graph + undo/redo]
    scene[Scene<br/>evaluated scratch snapshot]
    colour[ocio::State<br/>OCIO display pipeline]
    bus[Playa EventBus<br/>central typed command bus]
    preview[PreviewController<br/>RAM timeline preview cache]
    export[ExportController<br/>multi-format export]
    thumb[ThumbnailCache<br/>content-addressed disk cache]
  end

  app --> rs[RenderService<br/>dedicated worker thread]
  app --> io[IoService<br/>background file worker]
  rs --> gpu[Gpu<br/>WarpBroComputeEngine]

  subgraph WarpBroComputeEngine[WarpBroComputeEngine - src/engine.rs]
    cuda[CUDA: 234 specialized kernels<br/>fractal-kernels sm_75..sm_120]
    wgpu[WGPU: 28 per-family modules<br/>+ dynamic 28-way module]
  end

  gpu --> cuda
  gpu --> wgpu
  export --> coord[coordinate_export thread] --> writer[ExportWriter thread]
  coord -->|RenderExport| rs
```

---

## 2. Per-Frame Edit Transaction & Playa EventBus

```mermaid
sequenceDiagram
  participant UI as UI Panels / Hotkeys / REST
  participant Bus as Playa EventBus
  participant App as StudioApp::ui
  participant W as WorldEditor
  participant S as Scene (Scratch)
  participant R as RenderService

  UI->>Bus: try_emit(app_control::Event<UiCommand>)
  Note over Bus: Reliable admission: refuses overflow without evicting accepted events
  App->>Bus: handle_events drains queue at frame boundary
  App->>App: dispatch_command validates context (doc ID, frame, camera, revision)
  alt World Action (AE, Outliner, Timeline, Manipulator)
    App->>W: execute_edit(cmd, gesture)
    W->>W: apply -> set_attribute -> validate_render_profiles
    W->>W: record_edit: gesture step or undo.push(before)
  else App Action / Query
    App->>App: handle_app_action / control_query
  end
  App->>S: refresh_scene: snapshot_with_render_profile if dirty
  App->>R: step_viewport: ViewportRequest (single-slot mailbox)
  R-->>App: poll_events: Frame / Picked / Thumbnail / Error
```

---

## 3. World Document & Scene Evaluation

```mermaid
flowchart TD
  file[(scene .json)] -->|decode_scene| dj[WorldDocument::decode_json<br/>strict format & shape validation]
  dj --> doc[WorldDocument]
  new[File > New] -->|from_scene| doc
  
  subgraph Edits[Edit Pipeline]
    cmd[WorldCommand] --> ex[execute_edit] --> ap[apply_edit] --> sa[set_attribute]
    sa --> val[validate_render_profiles]
    val -->|Err: rollback| doc
    val -->|Ok: record_edit| doc
  end

  subgraph Evaluation[Evaluation Pipeline]
    doc -->|snapshot_with_render_profile| sc[Scene]
    doc -->|effective_render| er[EffectiveRender]
    er --> sc
    takes[Takes] -->|take_frame / take_trs| sc
    fn[FractalNodeSpec::evaluate] -->|pack_object| sc
    sc --> pack[Scene::pack: P_COUNT 128-float ABI]
  end
```

---

## 4. Render Worker & Dual Compute Backends

```mermaid
flowchart TD
  vp[ViewportRequest] --> mb[Mailbox]
  ex[RenderExport] --> mb
  th[Thumbnail] --> mb
  pv[Preview intents] --> mb
  mb --> wk[run_worker]

  wk --> prep[PreparationCache::get<br/>WorldUpload: params, objects, masks, lights]
  prep --> step[Gpu::step_with_display]

  step --> rb{resolve_backend_for_scene<br/>Auto / CUDA / WGPU}
  rb -->|CUDA| cuda[CUDA Kernels<br/>fractal-kernels: 234 specialized entries<br/>Fast / Full / Direct / Dynamic]
  rb -->|WGPU| wgpu[WgpuEngine<br/>formulas.wgsl: 28 family modules<br/>+ dynamic 28-way module]

  cuda --> film[Target Film: Film::Cuda / Film::Wgpu]
  wgpu --> film

  film --> adapt[adapt: active tile flags]
  film --> tally[tally: unresolved samples]
  film --> dn[process_denoise: OIDN HostDenoiser<br/>WorkContext, due decided once per step]
  dn --> tm[tonemap]
  tm --> cp[ColorPipeline OCIO<br/>cached per Ocio::serial]
  cp --> stamp[Gpu::stamp: waits for GPU batch<br/>accurate Batch::Budget pacing]

  stamp --> fr[Frame::snapshot]
  fr --> ui[UI Viewport Texture]
  fr --> xw[ExportWriter]
  step -->|Result<StepOutcome, RenderError>| ev[Typed RenderEvent to Requester]
```

---

## 5. Formula Registry & Compute Engine ABI

```mermaid
flowchart LR
  subgraph SSOT[Single Source of Truth]
    params[fractal-core::params]
    reg[formulas.rs: FORMULAS [FormulaMeta; 28]]
  end

  SSOT --> form[Formula: 28-variant enum]
  SSOT --> ui[Attribute Editor<br/>ParamMeta sliders, tooltips, validation]
  SSOT --> pack[FractalNodeSpec::pack_object<br/>P_COUNT 128-float ABI]
  SSOT --> consts[WGSL consts! macro<br/>FAMILY_*, SIGNED_MASK, ITERATING_MASK]

  pack --> cuda[CUDA: fractal-kernels<br/>234 specialized kernel entries]
  consts --> wgsl[WGPU: formulas.wgsl<br/>28 distance estimators]

  cuda <-->|parity_tests.rs| wgsl
  subgraph Parity[Parity Verification]
    num[Numerical DE / trap / guide oracle]
    img[Rendered image pixel comparison]
  end
  parity -.-> num
  parity -.-> img
```

---

## 6. Multi-Threaded Export Pipeline

```mermaid
flowchart TD
  out[OutputSettings Node<br/>video_io::job::Preset + WarpBro extras] --> ui[ExportController::ui<br/>edits preset in-place without writeback]
  ui --> start[ExportController::start]
  start --> val[export_quality + validation + OCIO view resolution]
  val --> coord[coordinate_export thread]

  coord -->|per-frame RenderExport| rs[RenderService]
  rs -->|ExportFrame: radiance / light / pixels| coord
  coord --> writer[ExportWriter thread]

  subgraph Writers[Writers & Sinks]
    writer --> exr[exr_io::write_exr<br/>ACEScg radiance OpenEXR]
    writer --> still[video_io::still::write_still<br/>PNG / JPEG / TIFF]
    writer --> vid[VideoSink: native encoders<br/>HEVC / ProRes]
    writer --> seq[PNG sequence + ffmpeg CLI]
  end
```
