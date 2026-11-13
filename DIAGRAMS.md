# WarpBro diagrams

Mermaid views of the main system dataflows, reflecting the `sl-studio` architecture (`5afff8a`).
Companion architecture notebooks: [PLAN.md](PLAN.md) and [AGENTS.md](AGENTS.md).

---

## 1. Ownership & Crate Dependency Architecture

```mermaid
flowchart TD
  subgraph AnimationRs["animation-rs (animation crate)"]
    anim_mods["curves / keys / time / anim / attrs (AttrValue, Attrs) / transform"]
  end

  subgraph SoluwellStudioCrates["sl-studio workspace crates"]
    ss_core["sl-studio-core (time, attrs, node_spec, params)"]
    ss_scene["sl-studio-scene (WorldDocument, Scene, takes, presets)"]
    ss_render["sl-studio-render (Gpu, RenderService, WgpuEngine, OCIO, OIDN)"]
    ss_ui["sl-studio (StudioApp, dock panels, EventBus, ExportController)"]
  end

  anim_mods --> ss_core
  anim_mods --> ss_scene
  ss_core --> ss_scene --> ss_render --> ss_ui

  main["warpbro-rs (main.rs)"] -->|"engine::init_warpbro"| reg["Register FractalNodeSpec & WarpBroComputeEngine"]
  ss_ui -->|"sl_studio::world re-exports animation-rs"| main
  main -->|"sl_studio::run_with"| win["sl_window::run"]
  win -->|"ctx.run_ui"| app["sl_studio::StudioApp"]

  subgraph StudioApp["StudioApp (sl-studio)"]
    world["WorldEditor (SubnetFile graph + undo/redo)"]
    scene["Scene (evaluated scratch snapshot)"]
    colour["ocio::State (OCIO display pipeline)"]
    bus["Playa EventBus (central typed command bus)"]
    preview["PreviewController (RAM timeline preview cache)"]
    export["ExportController (multi-format export)"]
    thumb["ThumbnailCache (content-addressed disk cache)"]
  end

  app --> rs["RenderService (dedicated worker thread)"]
  app --> io["IoService (background file worker)"]
  rs --> gpu["Gpu (WarpBroComputeEngine)"]

  subgraph WarpBroComputeEngine["WarpBroComputeEngine (src/engine.rs)"]
    cuda["CUDA: 234 specialized kernels (fractal-kernels sm_75..sm_120)"]
    wgpu["WGPU: 28 per-family modules + dynamic 28-way module"]
  end

  gpu --> cuda
  gpu --> wgpu
  export --> coord["coordinate_export thread"] --> writer["ExportWriter thread"]
  coord -->|"RenderExport"| rs
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
  file[("scene .json")] -->|"decode_scene"| dj["WorldDocument::decode_json (strict format & shape validation)"]
  dj --> doc["WorldDocument"]
  new["File > New"] -->|"from_scene"| doc

  subgraph AnimSubstrate["animation-rs (re-exported via sl_studio::world)"]
    anim_types["time / anim / attrs (AttrValue, Attrs) / transform / keys / curves"]
  end
  anim_types -->|"host Attrs & keyframe curves"| doc

  subgraph Edits["Edit Pipeline"]
    cmd["WorldCommand"] --> ex["execute_edit"] --> ap["apply_edit"] --> sa["set_attribute"]
    sa --> val["validate_render_profiles"]
    val -->|"Err: rollback"| doc
    val -->|"Ok: record_edit"| doc
  end

  subgraph Evaluation["Evaluation Pipeline (sl-studio-core -> sl-studio-scene -> warpbro-rs)"]
    doc -->|"snapshot_with_render_profile"| sc["Scene"]
    doc -->|"effective_render"| er["EffectiveRender"]
    er --> sc
    anim_types -->|"curve evaluation & TRS composition"| sc
    takes["Takes"] -->|"take_frame / take_trs"| sc
    fn["FractalNodeSpec::evaluate"] -->|"pack_object"| sc
    sc --> pack["Scene::pack (P_COUNT 128-float ABI)"]
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
  SSOT --> consts["engine.rs warpbro_shader_hook<br/>SIGNED_MASK, ITERATING_MASK + family_distance switch"]

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

## 6. Backend Resolution, Shader Hook, Fractal Validation & Clip (after plan1 waves A-D)

Real codepaths in `src/engine.rs` / `src/fractal_node.rs` and soluwell `world.rs`
(egui-widgets-rs `13d6f8d`). See [plan1.md](plan1.md) for why each path is shaped this way.

```mermaid
flowchart TD
  ng[engine::new_gpu] --> wce[WarpBroComputeEngine::new]
  wce -->|CudaEngine::new ok| cu[CUDA available]
  wce -->|CUDA fails: log::warn| wonly[WGPU only]
  warm[main --warmup] -->|engine::warmup_cuda| wc{CUDA loads?}
  wc -->|yes| ready[CUDA ready: device]
  wc -->|no| unav[CUDA unavailable, not warmed]

  step[execute_step scene] --> rb{resolve_backend}
  rb -->|Auto with CUDA, or explicit Cuda| cuk["CUDA kernels<br/>unknown family gives RenderError::Scene"]
  rb -->|Auto without CUDA, or explicit Wgpu| wg["soluwell WgpuComputeEngine<br/>(delegated, one implementation)"]
  rb -->|explicit Cuda, no device| err[typed error, no downgrade]

  wg --> hook[warpbro_shader_hook family]
  hook -->|Some code in FORMULAS| spec[specialized module: one estimator call]
  hook -->|Some unknown code| miss1["return miss (F32_MAX, TRAP_START)"]
  hook -->|None| dyn["dynamic switch: 28 cases + default miss"]

  load[WorldDocument::decode_json] --> vcn[validate_custom_nodes at first frame]
  edit[WorldEditor::apply_edit] --> vcn
  snap[snapshot_with_render_profile] -->|each shown node, at the frame| ov
  vcn --> ov["overlay_scene: Scene deserialize<br/>CustomBranches::new"]
  ov --> vn["FractalNodeSpec::prepare<br/>FractalParams::parse (once per scene build)"]
  vn -->|Err| rej["decode fails / edit rolled back / snapshot Err"]
  vn -->|Ok| pack["Scene::pack, radii, family, label:<br/>read the cached FractalParams (no JSON)"]
  pack --> clip{"clip radius (P_CLIP_RADIUS)"}
  clip -->|Mandelbox| mb["max(framed, mandelbox_reach: sqrt3 * 2F(s+1)/(s-1) or sqrt3 * 2F)"]
  clip -->|KIFS| kr["max(framed, kifs_reach: (|shift| + 2|h|s)/(s-1))"]
  clip -->|bulb-like| be["bailout * 1.001"]
  clip -->|own bound_radius| br["instance bound_radius"]
```

---

## 7. Multi-Threaded Export Pipeline

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
