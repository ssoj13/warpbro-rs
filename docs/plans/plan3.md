# Plan 3: MP4 export quality

Current status (2026-10-03): [workspace guide](../docs/workspace.md), [startup investigation](../docs/cuda-startup.md), and [changelog](../CHANGELOG.md) describe the current host/toolkit integration. The dated checkpoints below remain historical evidence; unchecked native checks are not implied complete.

Date: 2026-10-03.
Status: QP 18/medium defaults and zero-origin rational video timing implemented and tested; relocated CPU harness passed. Final tuned release build passed; five-fractal low-resolution MP4 comparison passed.

User reports moving block artifacts and unstable image detail in MP4 exported through ffmpeg-rs. Do not assume bitrate is the only cause. Root owns `src/export.rs` and any isolated ffmpeg-rs changes; publish shared fixes through a pinned GitHub SSH source after validation. The new 250-frame animated presets in [Plan 4](plan4.md) will provide camera/object motion and fractal-parameter transitions for reproducible comparisons.

## Diagnose

- [ ] Trace the actual codec and encoder settings from frac-rs through ffmpeg-rs.
- [ ] Inspect bitrate/quality control, GOP/keyframe interval, B/P-frame and motion settings, pixel format, bit depth, color tags and frame timestamps.
- [x] Compare deterministic 32-frame CPU motion patterns through the current encoder; five real-fractal sequences also passed the source/decode comparison.
- [ ] Verify image dimensions/stride, frame delivery, timebase and encoder flush; distinguish compression damage from conversion or sequencing defects.

## Improve

- [x] Fix production VideoTiming zero-origin/rational timing; regression and relocated harness passed.
- [x] Implement SSOT QP 18/medium defaults and a High quality button for existing preferences; 144-test host suite and final tuned release build passed.
- [x] Compare QP 27/veryfast and QP 18/medium on five low-resolution real-fractal motion sequences.
- [ ] Validate high-resolution/full-length animated output before extending the quality claim.
- [ ] Keep HDR/SDR conversion and color metadata correct.
- [ ] Preserve asynchronous export, bounded queues, cancellation and UI responsiveness.

## Verify and publish

- [x] Decode all five 32-frame real-fractal outputs and compare RGB PSNR and timing against source.
- [ ] Record actual codec/settings, bitrate/file size, quality and encode time.
- [ ] Run ffmpeg-rs and frac-rs export regressions.
- [ ] Publish shared fixes and consume a pinned GitHub SSH revision.
- [ ] Document final controls and measured tradeoffs.

## Measured checkpoint

The current SSH ffmpeg revision `7524123` was exercised with 32 deterministic motion-pattern frames at 320×180, plus a separate 66×50 RGB fixture with padded stride 288 bytes (dense row size 198 bytes). All 32 frames decoded; dimensions/stride, color and relative DTS/PTS cadence were valid. Comparing QP 27/veryfast with QP 18/medium raised RGB PSNR from 21.1174 to 31.9081 dB and luma PSNR from 28.1435 to 38.8945 dB. File size rose from 50,851 to 229,747 bytes (4.52×). These are CPU synthetic-pattern results, not final animated-fractal video acceptance.

The earlier mux introduced a leading empty edit of approximately 125–166 ms. Production `VideoTiming` now uses zero-origin rational timing. The relocated harness generated all four files with zero first PTS/start, 32 decoded frames spanning 32032 ticks and no empty edit. Fixed QP 27/veryfast and QP 18/medium files measured 50,839 and 229,735 bytes; RGB PSNR remained 21.1174 and 31.9081 dB. No codec source update was needed: the new `av-format-mov` dev dependency uses the existing SSH branch source locked at `7524123`.

Host `ExportSettings` supplies QP 18/medium defaults, cached encoder schema uses `OnceLock`, and High quality updates saved preferences. The [final system suite](../target/verification/final-system-tests.out) passed 144 tests in 55.50 s without compiler warnings, `CDX_FINAL_SYSTEM_TEST_EXIT=0`, including the production timing regression. The final tuned release build passed in 46.14 s ([log](../target/verification/tuned-final-presets.out), marker 0). All five real-fractal source/decode comparisons and the synthetic regression completed with marker 0. Each sequence uses the initial 32 frames at 160×90/4 SPP; all decoded frame counts are 32, first PTS is 0, duration is 32032 ticks and no empty edit remains. RGB PSNR (QP 27/veryfast → QP 18/medium, dB): Chrome 39.7928→40.6054, Ember 30.7657→35.0739, Web 34.0105→36.6380, Opal 34.1753→37.2935 and Verdant 34.0105→37.6529. Artifacts are `target/verification/mp4-quality/fractal-{slug}/summary.csv`, MP4 files and timing CSVs. These checks do not establish 4K quality or a full 250-frame mux run; encode-time comparison and broader visual acceptance remain pending.

## Reproduce the quality comparison

The tracked [harness](../tools/mp4-quality/src/main.rs) uses its own manifest and lockfile. Run:

```sh
cargo +stable run --release --offline --locked --manifest-path tools/mp4-quality/Cargo.toml --target-dir target/verification/mp4-quality/build -- target/verification/mp4-quality/results-fixed
```

Append `--legacy` to reproduce the old leading-empty-edit behaviour. To compare a real-fractal raw RGB sequence, run:

```sh
cargo +stable run --release --offline --locked --manifest-path tools/mp4-quality/Cargo.toml --target-dir target/verification/mp4-quality/build -- OUTPUT --rgb-sequence target/verification/fractal-rgb/PRESET 160 90
```

The verified real sequences cover low-resolution initial 32-frame segments. Full 250-frame/high-resolution encoded output remains outside this checkpoint.
