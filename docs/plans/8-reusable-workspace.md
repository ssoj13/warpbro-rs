# Reusable workspace components — future TODO

Requested 2026-10-03. This is a future extraction plan, not implemented functionality.

- [ ] Extract the unified Outliner to egui-widgets-rs: hierarchy, selection, visibility, lock/solo, drag ordering, context commands.
- [ ] Extract the unified Attribute Editor: sections, adjustable label split, compact metrics, schema values, optional animation controls.
- [ ] Extract the application-independent Timeline adapter: node layers, property lanes, shared selection/key identities, cache coverage, panel command bindings.
- [ ] Extract Curve Editor after its animation model and interactions stabilize. Timeline and Curve Editor must edit the same keys and use host Undo transactions.
- [ ] Extract adaptive Gallery: responsive thumbnail columns, stable selection, drag-and-drop, menus, dirty preview scheduling.
- [ ] Extract Material Library as both a complete data catalog and a reusable browser/assignment widget. Preserve all existing material identities and the curated metallic, glass, fabric, ceramic and other looks.
- [ ] Place material definitions in a renderer-independent crate with no egui dependency. Provide host adapters for Standard Surface and scene material nodes.
- [ ] Share global/panel hotkey dispatch, keyed file-dialog history, compact UI metrics and cache coverage presentation.
- [ ] Evaluate scene templates and reusable preview-cache transport as separate host services.

## Implemented foundations (2026-10-03)

Shared attribute-row geometry and optional editor hooks already live in egui-widgets-rs. Its Timeline now exposes a pinned-ruler host API with clipping/input regression coverage. These are reusable primitives, not the complete World-panel adapters listed above.

The renderer-independent material catalog was extracted into WarpBro's `crates/fractal-materials` and reused by the OFX integration. Moving/publishing the complete catalog and generic browser in the toolkit remains pending.

## Architecture constraints

Widgets receive borrowed, stable projections and emit edit intents. Applications retain scene ownership, animation evaluation, renderer scheduling, Undo/Redo and I/O. No WarpBro node types, CUDA dependencies or file-system operations in generic widgets. Rebuild projections only on model revisions; retain scratch storage between repaints. Dependency pins remain SSH Git refs.

## Shared contracts and extraction order

- [ ] Specify stable node, property, material and key identities. All panels observe the host's selection; clicking a material selects its node for the same Attribute Editor.
- [ ] Keep material selection and assignment as separate intents. Support assignment through an object's material property, a library/gallery context menu and host-defined drag-and-drop.
- [ ] Define one editing transaction per gesture, with preview updates during dragging and a single Undo commit on release.
- [ ] Define versioned material catalog data, serialization and preview cache keys. Parameter changes dirty previews and notify every material consumer; widgets never launch render jobs themselves.
- [ ] Expose shared compact layout metrics, movable column splits and panel-scoped input ownership. Include empty-panel click handling to prevent input falling through to another dock panel.
- [ ] Stabilize and document host contracts first; extract existing reusable primitives before the larger panels; move the complete material catalog with adapters; then migrate applications one at a time.

This is deduplication of the working system, not a second implementation beside WarpBro. Existing egui-widgets-rs attribute editors and timeline widgets should be extended where appropriate rather than replaced by parallel APIs.

## Adoption

Pilot the extracted components in Playa and one other cglibs application. Keep the existing WarpBro behavior as the reference and test clipping/layer ownership, keyboard scope, selection synchronization, gesture Undo grouping and idle allocation behavior before switching other applications.
