# WarpBro diagrams

Mermaid views of the main dataflows. ASCII versions with line anchors: [AGENTS.md](AGENTS.md).
As of commit `4cfa5eb` (2026-10-22). Red edges mark the defects tracked in [plan1.md](plan1.md).

## Ownership

```mermaid
flowchart LR
  main[main.rs] --> run[app::run] --> win[window::run / Native]
  win -->|ctx.run_ui| App
  subgraph App[App - app.rs]
    world[WorldEditor<br/>document + undo]
    scene[Scene<br/>evaluated scratch]
    colour[ocio::State]
    export[ExportController]
    preview[PreviewController]
  end
  App --> rs[RenderService<br/>worker thread]
  App --> io[IoService<br/>file worker]
  rs --> gpu[Gpu<br/>CUDA + WGPU]
  export --> coord[coordinate_export thread] --> writer[ExportWriter thread]
  coord -->|RenderExport| rs
```

## Per-frame edit transaction

```mermaid
sequenceDiagram
  participant UI as App::ui
  participant S as self.scene (scratch)
  participant W as WorldEditor
  participant R as RenderService
  UI->>W: snapshot(playhead) if (id, revision, frame) changed
  W-->>S: Scene
  UI->>UI: before = snapshot_before.take()
  UI->>R: poll_events (Frame / Picked / Thumbnail / Error)
  UI->>S: navigation and settings panels mutate
  UI->>W: manipulate / AE / timeline: execute_edit(cmd, gesture)
  UI->>W: diff(S, before): navigate_camera / edit_snapshot
  UI->>W: finish_edit_unless(held gesture)
  UI->>R: step_viewport: ViewportRequest
```

## World document

```mermaid
flowchart TD
  file[(scene .json)] -->|decode_scene| dj[WorldDocument::decode_json]
  dj --> hy[hydrate_schema<br/>legacy migrations]:::bad
  hy --> doc[WorldDocument]
  new[File > New] -->|from_scene| doc
  cmd[WorldCommand] --> ex[execute_edit] --> ap[apply_edit] --> sa[set_attribute]
  sa --> val[validate_render_profiles] -->|Err| rb[rollback]
  val -->|Ok| rec[record_edit: gesture step or undo.push]
  rec --> doc
  doc -->|snapshot_with_render_profile| sc[Scene]
  doc -->|effective_render| er[EffectiveRender]
  er --> sc
  takes[Takes] -->|take_frame / take_trs| sc
  classDef bad stroke:#d33,stroke-width:2px
```

## Render worker

```mermaid
flowchart TD
  vp[ViewportRequest] --> mb[Mailbox]
  ex[RenderExport] --> mb
  th[Thumbnail] --> mb
  pv[Preview intents] --> mb
  mb --> wk[run_worker]
  wk --> step[Gpu::step_with_display]
  step --> rb{resolve_backend}
  rb -->|Auto, classic| cuda[CUDA kernels<br/>fractal-kernels]
  rb -->|extended family / no CUDA| wgpu[WgpuEngine<br/>render_wgpu.wgsl]
  cuda --> film[Target film]
  wgpu --> film
  film --> adapt[adapt / tally]
  film --> dn[process_denoise<br/>OIDN]
  dn -.->|due called twice on WGPU| skip[denoise skipped]:::bad
  film --> tm[tonemap] --> cp[ColorPipeline OCIO] --> fr[Frame::snapshot]
  fr --> ui[UI texture]
  fr --> xw[ExportWriter]
  step -.->|error only in colour_error| hang[job waits forever]:::bad
  classDef bad stroke:#d33,stroke-width:2px
```

## Formula registry (target state after plan1.md)

```mermaid
flowchart LR
  reg[fractal-core params<br/>+ formulas.rs FORMULAS] --> names[Formula::name]
  reg --> ui[AE sliders / hints]
  reg --> pack[Scene::pack]
  reg --> consts[WGSL consts! emit<br/>FAMILY_*, SIGNED_MASK]
  pack --> cuda[CUDA kernels]
  pack --> wgsl[WGSL kernels]
  cuda <-->|parity test| wgsl
```

## Export

```mermaid
flowchart TD
  out[OutputSettings node] --> ui[ExportController::ui]
  ui -.->|to_preset / apply_preset each frame| out
  ui --> start[ExportController::start]
  start --> q[export_quality + validate + OCIO resolve]
  q --> co[coordinate_export]
  co -->|per frame RenderExport| rs[RenderService]
  rs -->|ExportFrame| co --> w[ExportWriter]
  w --> exr[EXR ACEScg]
  w --> png[PNG / JPEG / TIFF]
  w --> vid[Video sink]
  w --> seq[PNG seq + ffmpeg]
```
