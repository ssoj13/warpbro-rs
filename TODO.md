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
- [ ] Toolbar: gear right of the view transform ("ACES") opens Settings > Colour
- [ ] Toolbar: EV reset-to-0 button left of EV
- [ ] Toolbar: denoiser quick toggle (off = raw radiance, on = current denoised result)

## Attribute Editor (shared widgets in egui-widgets-rs)
- [ ] Grey out parameters inactive for the current fractal type (e.g. Mandelbox)
- [ ] Section colour coding: tint title bars (render, material, custom, transform, fractal)
- [ ] Colour picker: the newer egui-widgets-rs picker (delayed hide)
- [ ] Vec3 colour everywhere a parameter is a colour
- [ ] Every numeric parameter: label | slider | value | swatch | expand toggle (design first)
- [ ] RMB > Show in timeline: expand the layer and scroll to the attribute

## Render / Encode panel
- [ ] Deduplicate the three format tabs (shared settings once; per-format codec options);
      the egui-encode-dialog schema half-duplicates the hand-written UI (SSOT)
- [ ] Unify colour output options across EXR / PNG / MP4 (EXR: ACEScg / ACES2065-1 /
      linear Rec.709 primaries; PNG: SDR / HDR10 / HLG; MP4: SDR [+ HDR10 if the encoders do 10-bit])
      — plan for operator approval first

## Bugs
- [x] Adaptive sampling: with 1 sample per frame it renders that sample and stops
- [ ] HEVC export tests flaky under the full parallel suite (Vulkan Video `Posix(38)`);
      pass alone

## Done (2026-10-04)
- [x] Camera: horizon lock, inertia everywhere, finite braking, Space up, Shift fast / Alt slow
- [x] Orbit on the shared Houdini rig; orbit coast frame-rate independent
- [x] PreviewCache frees discarded rasters off-thread (dead `invalidate` warning)
