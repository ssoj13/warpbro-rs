# WarpBro TODO

Working list of operator requests. Keep it current: tick done items, move finished work to
CHANGELOG.md, add new requests as they come. Reusable UI goes into egui-widgets-rs crates
(operator, 2026-10-04: "все эти фичи в крейтах egui-widgets-rs, чтобы другие могли пользоваться").

## Timeline
- [x] Home / End: time cursor to work area start / end (Timeline scope, Playa bindings)
- [x] B / N: work area start / end at the time cursor
- [x] Shift+0-9 set time mark, 0-9 jump (physical keys; `WorldDocument.marks`, undoable);
      Ctrl+click on a ruler mark clears it (as in Playa); slot numbers drawn on the ruler
- [x] U (and P/R/S) expand caret desync: caret follows the track's real expanded state;
      toggling the last filter off collapses
- [x] Zoom slider twice as long

## Viewport / camera
- [x] Shift + LMB orbit snaps the view parallel to the nearest world axis (X/Y/Z)
- [x] Toolbar: gear right of the view transform ("ACES") opens Settings > Colour
- [x] Toolbar: EV reset-to-0 button left of EV
- [x] Toolbar: denoiser quick toggle (off = raw radiance, on = current denoised result)

## Decisions (operator, 2026-10-04)
- Order: Attribute Editor first, then Render / Encode.
- Soft slider ranges: an explicit table for every numeric parameter; doubtful ones listed here.
- EXR colour: linear ACEScg / ACES2065-1 / Rec.709 / Rec.2020 and a display-referred variant.
- HDR10 MP4 (Main10, PQ, BT.2020): a separate stage after the panel unification.

## Attribute Editor (shared widgets in egui-widgets-rs)
- [x] Grey out parameters inactive for the current fractal type (e.g. Mandelbox)
- [x] Section colour coding: tint title bars (render, material, custom, transform, fractal)
- [x] Colour picker: the newer egui-widgets-rs picker (delayed hide)
- [x] Vec3 colour everywhere a parameter is a colour
- [x] Every numeric parameter: label | slider | value | swatch | expand toggle (design first)
- [x] RMB > Show in timeline: expand the layer and scroll to the attribute

## Render / Encode panel
- [ ] Deduplicate the three format tabs (shared settings once; per-format codec options);
      the egui-encode-dialog schema half-duplicates the hand-written UI (SSOT)
- [ ] Unify colour output options across EXR / PNG / MP4 (EXR: ACEScg / ACES2065-1 /
      linear Rec.709 primaries; PNG: SDR / HDR10 / HLG; MP4: SDR [+ HDR10 if the encoders do 10-bit])
      — plan for operator approval first

## Bugs
- [x] Clear glass lost energy at grazing angles (BUG1.md; render-rs 207473e, exact smooth interfaces)
- [ ] Rough dielectric transmission: no multiple-scattering compensation (furnace 0.81-0.93 inside
      at roughness 0.5) — compensation table for dielectrics
- [x] BUG1 march: out of steps is a miss (was 37% phantom hits in frame 27), step cap decoupled
      from the budget, presets 4096 steps, status reports the out-of-steps share
- [ ] BUG1 open: Hybrid DE discontinuities at fold / escape transitions (re-measure on converged
      hits: the old statistics mixed in the phantoms); 6-probe normal cost (isolated benchmark)
- [ ] Port the march fix to the OFX plug-in's kernel copy (`ofx-rs/crates/ofx-fractal/kernels/source/src/gpu.rs`
      still has the full-pixel closest-sample rule and `2 span / max_steps`), then check silhouettes
- [ ] Fractal nodes carry a full `render` block but only `/render/iterations` is per object (the
      Attribute Editor hides the rest, world.rs `attributes`; march settings live on the settings
      node, one march traces the union of objects). Not user-visible; drop the dead per-object
      fields when the object model is split (low priority)
- [ ] Over-relaxation sphere tracing (Keinert 3.1) for grazing rays: only with a safety check,
      Hybrid DEs are not Lipschitz bounds
- [ ] `cuda_specialized_world_materials_preserve_radiance_and_affine_guides` (ignored test) fails on
      b5d562a too: Fast objects=1 radiance max_absolute 5.92e-5 > 3e-5 tolerance - investigate
- [x] Adaptive sampling: with 1 sample per frame it renders that sample and stops
- [ ] Playa: `playa-engine/tests/ofx_builtin.rs` (feature ofx) expects 8 built-in plug-ins, ofx-effects
      links 29 now (fails on fce3d93 too, before the per-channel work) — update the test to the set
- [ ] HEVC export tests flaky under the full parallel suite (Vulkan Video `Posix(38)`);
      pass alone

## Done (2026-10-04)
- [x] Camera: horizon lock, inertia everywhere, finite braking, Space up, Shift fast / Alt slow
- [x] Orbit on the shared Houdini rig; orbit coast frame-rate independent
- [x] PreviewCache frees discarded rasters off-thread (dead `invalidate` warning)

