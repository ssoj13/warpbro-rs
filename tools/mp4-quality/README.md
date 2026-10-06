# mp4-quality

Reproduces the measured MP4 QP / preset trade-off and verifies zero-origin rational video timing
on deterministic CPU patterns. It has its own manifest and lockfile.

```sh
cargo +stable run --release --offline --locked --manifest-path tools/mp4-quality/Cargo.toml --target-dir target/verification/mp4-quality/build -- target/verification/mp4-quality/results-fixed
```

Append `--legacy` to reproduce the old leading-empty-edit behaviour. To compare a real-fractal raw
RGB sequence:

```sh
cargo +stable run --release --offline --locked --manifest-path tools/mp4-quality/Cargo.toml --target-dir target/verification/mp4-quality/build -- OUTPUT --rgb-sequence target/verification/fractal-rgb/PRESET 160 90
```

Last measured (initial 32 frames of each preset at 160x90 / 4 spp; RGB PSNR, QP 27 veryfast ->
QP 18 medium, dB): Chrome 39.79 -> 40.61, Ember 30.77 -> 35.07, Web 34.01 -> 36.64, Opal 34.18 ->
37.29, Verdant 34.01 -> 37.65. All decoded frame counts 32, first PTS 0, no empty edit. Full
250-frame and high-resolution output were not verified.
