# Systemic `/sysgo` & `/mastermind` Plan — `animation-rs` & `soluwell-studio` Ecosystem Unification

## Completed Milestones
- [x] **M0**: Extract standalone `animation-rs` (`animation` crate: `time`, `anim`, `attrs`, `transform`, `keys`, `curves`) with `#![forbid(unsafe_code)]`, 97 unit tests, and NaN/Inf/empty-slice boundary guards (`607e380`).
- [x] **M1**: Rewire `playa` (`playa-time`, `playa-engine`) to re-export `animation::*` (`-5,213` duplicate lines deleted, `818a26f`).
- [x] **M2**: Rewire `egui-widgets-rs` (`soluwell-studio-core`, `soluwell-studio-scene`, `soluwell-studio-render`, `soluwell-studio`) directly to `animation`, fix camera keyframe navigation + 3-pose hold-key seeding (`84f39a3`).
- [x] **M3**: Rewire `warpbro-rs` onto `soluwell-studio` + `animation-rs`, prune 33 unused dependencies (`597b60c`).

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
- Full 57-file shell deletion in `openvdb-rs/crates/vdb-view` (requires extracting `VolumeNodeSpec` + `VdbComputeEngine` + custom EXR/Scope dock tabs into `soluwell-studio`).
- `usd-rs` internal `usd-ts` / `usd-vt::spline` consolidation.
