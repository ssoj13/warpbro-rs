# Systemic `/sysgo` & `/mastermind` Plan — Ecosystem SSH Git Ref Upgrade (`2026-11-11`)

## Active `/mastermind` Wave (`box-rs` + `warpbro-rs` Lock & Seam Upgrade)
- [x] **D00 (`box-rs`)**: Restore complete `//!` crate header in `box-rs/src/lib.rs` + fix Rust 1.99 clippy lint in `src/container.rs`, pass `cargo fmt --check`, `cargo clippy --all-targets --all-features -- -D warnings`, `cargo doc`, and `cargo test --all-features` (26 passed), committed `b99c8be` and pushed to `origin/main`.
- [x] **D01 (All 3 Lockfiles in `warpbro-rs`)**: Upgraded all SSH git refs (`ssh://git@github.com/ssoj13/*`) across `Cargo.lock`, `xtask/Cargo.lock`, and `tools/mp4-quality/Cargo.lock`.
- [x] **D02 (`engine` / `pipeline_cache_probe` / `xtask` / `mp4-quality`)**: Audited 91/91 symbols + Tier 2 Inquisitor (`0` defects): fixed `examples/pipeline_cache_probe.rs` post-compile reload for `gpu-info-rs@25e69fd`, migrated `tools/mp4-quality/src/main.rs` to `ffmpeg-rs@bdbfdea` (`set_video_color_for`, zero-churn `read_sample_into`), fixed `CudaTarget::out` reuse resize + persistent `pick_query`/`pick_out` buffers + exhaustive `Backend` match in `src/engine.rs`, and aligned `xtask/src/main.rs` with `vcv-rs@d74f16e` CUDA MSVC range.
- [ ] **D03 (`fractal_node` / `presets` / `templates`)**: Audit `FractalNodeSpec` (`NodeTypeSpec` v2), `pack_object` kernel ABI, and `animation` / `curves` contracts.
- [x] **D04 (`parity_tests` / `render_bench` / `test_gpu`)**: Audited 49/49 (`D04a`) + 23/23 (`D04b`) symbols through 3 Inquisitor passes (`0` defects): aligned `pixel_beam_start_fixes_recorded_mobius_surface_skip` (`sample = 12`) with `standard-surface-bsdf@61c7fc3`, fixed quantized `parameter_probe_formulas` & `oracle_film` readback churn, fixed `render_bench` `"mixed"` `code == 0` homogeneity bug + adaptive sample accounting + `gpu_paired_world_de_benchmark` prefix, and upgraded `test_gpu.rs` workspace lock/prefix ratchet.
- [ ] **Inquisitor & Seam Gate**: Independent Tier 2 Inquisitor audit (7 Mandatory Questions) + `SEAM_MATRIX.md` + `python bootstrap.py c` + `cargo oxide test` + seeded mutant check + `python bootstrap.py b` GPU warmup + `AGENTS.md` changelog update.

---

# Systemic `/sysgo` & `/mastermind` Plan — `animation-rs` & `sl-studio` Ecosystem Unification

## Completed Milestones
- [x] **M0**: Extract standalone `animation-rs` (`animation` crate: `time`, `anim`, `attrs`, `transform`, `keys`, `curves`) with `#![forbid(unsafe_code)]`, 97 unit tests, and NaN/Inf/empty-slice boundary guards (`607e380`).
- [x] **M1**: Rewire `playa` (`playa-time`, `playa-engine`) to re-export `animation::*` (`-5,213` duplicate lines deleted, `818a26f`).
- [x] **M2**: Rewire `egui-widgets-rs` (`sl-studio-core`, `sl-studio-scene`, `sl-studio-render`, `sl-studio`) directly to `animation`, fix camera keyframe navigation + 3-pose hold-key seeding (`84f39a3`).
- [x] **M3**: Rewire `warpbro-rs` onto `sl-studio` + `animation-rs`, prune 33 unused dependencies (`597b60c`).

## Completed `/mastermind` Wave — Consumer Ecosystem Unification (`D01`..`D04`)
- [x] **D01 (`animation-rs` `607e380` & `playa` `818a26f` — SSOT Deduplication)**:
  - Cleaned up legacy `playa` prose references in `animation-rs/src/{time/mod.rs, attrs.rs, keys.rs}` and exported `conform` and `rescale` from `animation::time`.
  - Removed unused `glam` & `serde` from `playa/crates/playa-time/Cargo.toml` and unused `bitflags` from `playa/crates/playa-engine/Cargo.toml`.
  - Deduplicated `bar_span`, `conform`, `rescale` in `playa/crates/playa-engine/src/entities/timing.rs` to re-export/delegate to `animation::time::{bar_span, conform, rescale}` (Single Source of Truth).
  - **Verified**: `cargo clippy --all-targets -- -D warnings && cargo test --quiet` in `animation-rs` (97 passed); `cargo check -p playa-time -p playa-engine --no-default-features --quiet` + `cargo test -p playa-engine --no-default-features --lib entities::timing::tests --quiet` in `playa` (22 passed).
- [x] **D02 (`openvdb-rs/crates/vdb-view` `3765fca` — `animation-rs` & Camera Animation Fix)**:
  - Switched `openvdb-rs/crates/vdb-view` directly to `animation` (`animation-rs`), dropping `playa-time` from `crates/vdb-view/Cargo.toml`.
  - Updated `world.rs`, `camera_orbit.rs`, `camera_recorder.rs`, `world_tests.rs` from `playa_engine::entities::*` to `animation::*`.
  - Ported the exact camera animation fix (`POSE_PATHS`, `seed_frames`, `write_navigation_value` with 3-pose hold-key seeding, and `camera_pose` distance scaling) into `crates/vdb-view/src/world.rs`, eliminating the broken `/_navigation/*` offset writes in `vdb-view`, and added the 4-stage regression test in `crates/vdb-view/src/world_tests.rs`.
  - **Verified**: `cargo check -p vdb-view --lib --quiet`, `cargo test -p vdb-view --lib manual_camera_keying_and_viewport_navigation_animate_pose_and_distance --quiet` (1 passed + seeded mutant check verified RED then GREEN), and `cargo test -p vdb-view --lib camera_navigation_ --quiet` (3 passed).
- [x] **D03 (`rv-rs` `c875447` — Dead Dependency Pruning & `animation::time::Fps` Migration)**:
  - Removed dead/unused workspace dependencies `playa-engine`, `playa-ui`, and `curves` from `rv-rs/Cargo.toml`.
  - Replaced `playa-time` with `animation` in `rv-rs/Cargo.toml` and `crates/rv-app/Cargo.toml`, and updated `crates/rv-app/src/playback.rs:2465` to `animation::time::Fps::from_f32_lossy`.
  - **Verified**: `cargo check -p rv-app --tests --quiet` (0 errors).
- [x] **D04 (`skelly-rs` — `animation-rs` Substrate Adoption)**:
  - Added `animation` (`animation-rs`) to `skelly-rs/Cargo.toml`, `crates/skelly/Cargo.toml`, `crates/skelly-rig/Cargo.toml`, and `crates/skelly-nodes/Cargo.toml`.
  - Re-exported `pub use animation;` from `skelly-rig` and `skelly-nodes`.
  - Upgraded `Host` in `crates/skelly/src/app.rs` to use `animation::time::Fps` (`Fps::from_f32_lossy`) for frame-rate sanitized playback stepping, with unit test coverage (`host_time_and_step_normal_fps`, `host_time_and_step_degenerate_fps`, `host_step_wraps_around_frame_range`).
  - **Verified**: `cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace --quiet` in `skelly-rs` (16 passed; 0 failed).

## Out of Scope (Deferred to Dedicated Repo Sessions)
- Full 57-file shell deletion in `openvdb-rs/crates/vdb-view` (requires extracting `VolumeNodeSpec` + `VdbComputeEngine` + custom EXR/Scope dock tabs into `sl-studio`).
- `usd-rs` internal `usd-ts` / `usd-vt::spline` consolidation.
