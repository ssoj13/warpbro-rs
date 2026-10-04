# Work with nodes, materials, and cached playback

## Edit nodes through one Attribute Editor

The Outliner, Materials gallery, Timeline, and Attribute Editor share UUID-based node selection. Fractals, cameras, lights, environments, groups, and materials belong to the same World document. Material edits use the same Attribute Editor as other node types.

Translate, Rotate, and Scale edit the node's authored transform. Viewport flight edits the active Camera node; navigation records keys only when Auto Key is enabled. Numeric dragging updates the preview during the gesture and creates one Undo transaction when the gesture completes.

The Attribute Editor retains Playa-style collapsible sections and the existing shared attribute grid. Settings → Controls owns compact row, icon, and numeric-field metrics. The shared layout aligns value columns and preserves the saved label divider when a panel narrows.

## Select and assign materials separately

Click a workspace material card to select that Material node for the Attribute Editor. Editing its parameters updates every object that references its UUID and invalidates its preview when evaluated material values change.

Assign a material through any of these paths:

- Select a fractal, then choose its **Material** attribute.
- Right-click a material card and choose **Assign to selected objects**.
- Select an eligible object, select a material, and use **Apply to object** in the material Attribute Editor. The command remembers the last eligible object selection, validates it, and groups assignment into one Undo transaction.

Only nodes that support material assignment are receivers; a placeholder metadata field does not make a camera or light a material consumer. Unsupported selection clears the remembered target.

The separate **Material Library** window shows 69 presets across 12 categories from the shared `fractal-materials` catalog. Opening it does not create nodes. Clicking a preset creates and selects a workspace Material node without assigning it. **Apply preset to this material** updates an existing material UUID. **Refresh preview** retries a failed preview. Galleries adapt their column count to panel width.

Transparent and absorbing materials use the same node editor and assignment commands. See [glass, refraction, and depth-dependent color](glass.md) for parameters and presets.

## Save scenes and templates

File actions save/open the persistent World document, including node UUIDs, metadata, material references, animation, and the active camera. Templates are ordinary scene JSON files under `~/.warpbro/templates/`. The five bundled 250-frame scenes seed that directory once; later scans preserve user edits and deliberately deleted templates. Saving the current scene as a template uses the same background I/O and scene decoder as normal projects.

File dialogs retain directories and selected filters by control/action key. HDR environment selection supports Radiance HDR and OpenEXR. Camera orbit speed and phase remain animatable attributes; scene evaluation derives the orbit from Timeline time.

## Render final images and animation

Render / Encode owns image and movie settings and progress. Its **Once at completion** denoise option performs one final pass for each completed output frame, independently of World Settings' periodic cadence. EXR sequence output stays scene-linear; video uses the display-transformed SDR image.

Vulkan Video is the default GPU HEVC route. The explicit software choice uses Kvazaar I-frames to avoid the reproduced inter-prediction corruption. Cancelling a movie drains complete frames and finalizes the partial container; incomplete frames are discarded. Initialization and finalization stay on workers.

## Use shortcuts in the active panel

Global shortcuts resolve first, then the active panel's bindings. Text and numeric editing retain their input. A pointer in Render / Encode must not trigger Timeline gestures underneath it.

| Scope    | Input                        | Action                                                |
| -------- | ---------------------------- | ----------------------------------------------------- |
| Global   | Tab                          | Toggle panels                                         |
| Global   | Ctrl/Cmd+Z; Ctrl/Cmd+Shift+Z | Undo; redo                                            |
| Timeline | F                            | Fit layer spans into the visible time range           |
| Viewport | F                            | Frame the fractal bounds                              |
| Timeline | P or T; R; S                 | Show Translate; Rotate; Scale properties              |
| Timeline | U                            | Show keyed properties                                 |
| Timeline | Shift + property shortcut    | Add or remove a property filter                       |
| Timeline | I; O                         | Set selection In; Out                                 |
| Timeline | Space                        | Play or pause                                         |
| Timeline | Ins                          | Play the selected range                               |
| Timeline | Shift+Ins                    | Cache the selected range at target samples, then play |
| Timeline | Ctrl+Shift+Ins               | Cache the selected range at 1 sample, then play       |

The Timeline toolbar and time ruler remain fixed during vertical scrolling. Only the layer names, property lanes, and corresponding bars scroll. The names/bars divider is draggable and persists its width; double-click resets it.

## Cache completed frames

A stationary viewport frame enters the RAM cache after reaching the requested final sample count at native output dimensions. Incomplete accumulation, interactive proxies, and stale generations are excluded.

The Timeline coverage strip shows resident cached frames: green for final quality, blue for draft. Final and 1-SPP draft playback use distinct cache modes, so a draft cannot satisfy a final-quality request. Playa supplies the frame cache and playback clock; CUDA rendering and frame preparation remain in the render worker.

Ins plays immediately and can reuse valid resident frames. Shift+Ins and Ctrl+Shift+Ins prepare their requested cache mode before playing. Cache validity follows the document/evaluation identity and render settings. Memory is bounded, so eviction can require rerendering a frame.

This preview cache stores frames in RAM. Disk-backed EXR preview caching and progressive partial-frame persistence are not implemented. The proposed Curve Editor must share the existing animation keys and Undo commands; it is still pending.
