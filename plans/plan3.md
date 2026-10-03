# Plan 3: MP4 export quality

Date: 2026-10-03.
Status: queued after the current UI and render-optimization tasks.

User reports moving block artifacts and unstable image detail in MP4 exported through ffmpeg-rs. Do not assume bitrate is the only cause.

## Diagnose

- [ ] Trace the actual codec and encoder settings from frac-rs through ffmpeg-rs.
- [ ] Inspect bitrate/quality control, GOP/keyframe interval, B/P-frame and motion settings, pixel format, bit depth, color tags and frame timestamps.
- [ ] Produce a short reproducible export from deterministic frames and compare it with the source images.
- [ ] Verify image dimensions/stride, frame delivery, timebase and encoder flush; distinguish compression damage from conversion or sequencing defects.

## Improve

- [ ] Fix any confirmed conversion/timing/encoder defect in its owning module.
- [ ] Provide appropriate quality-oriented defaults and clear quality controls.
- [ ] Compare supported codec/quality presets on high-detail fractals and motion.
- [ ] Keep HDR/SDR conversion and color metadata correct.
- [ ] Preserve asynchronous export, bounded queues, cancellation and UI responsiveness.

## Verify and publish

- [ ] Decode representative output and compare frames against source; inspect motion and block artifacts.
- [ ] Record actual codec/settings, bitrate/file size, quality and encode time.
- [ ] Run ffmpeg-rs and frac-rs export regressions.
- [ ] Publish shared fixes and consume a pinned GitHub SSH revision.
- [ ] Document final controls and measured tradeoffs.
