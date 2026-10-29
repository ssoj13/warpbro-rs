# Domain Ledger `D02` — `openvdb-rs/crates/vdb-view` (`animation-rs` & Camera Animation Fix)

- **Status**: `[x] VERIFIED (0 defects after 2 Inquisitor passes)`
- **Sub-Orchestrator**: Tier 1 Domain Sub-Orchestrator (`D02`)
- **Touched Scope**: `<= 450 LOC` across 5 source files in `C:\projects\projects.rust.cg\cglibs\openvdb-rs\crates\vdb-view`
- **Reference Implementation**: `C:\projects\projects.rust.cg\cglibs\egui-widgets-rs\crates\soluwell\studio\scene\src\world.rs` (`L1830-1895`, `L5426-5526`, `L5626-5674`, `L6551-6668`)

---

## 1. Micro-Unit & Symbol Ledger (`1..8`)

### Micro-Unit Partition (`U01..U03`)
- **`U01` (`Cargo.toml`, `camera_orbit.rs`, `camera_recorder.rs`)**: Crate dependency rewiring (`animation-rs`, dropping `playa-time`, retaining `playa-engine` for `preview.rs`) and `animation::anim::{Animation, Channel, Keyframe}` migration (`~90 LOC`).
- **`U02` (`src/world.rs`)**: `animation::{anim, attrs, transform}` migration, `WorldEditor::navigate_camera` (`POSE_PATHS`, `seed_frames`, `pose_animated`), `WorldDocument::write_navigation_value` (3-pose `should_seed` hold-key seeding, zero `/_navigation/*` writes), and `WorldDocument::camera_pose` distance scaling parity (`~240 LOC`).
- **`U03` (`src/world_tests.rs`)**: `animation::{attrs::AttrConnection, transform::build_model_matrix}` test migration, updated camera navigation test assertions (verifying zero `/_navigation/*` writes and offset compensation), and the 4-stage `manual_camera_keying_and_viewport_navigation_animate_pose_and_distance` regression test (`~220 LOC`).

### Symbol-by-Symbol Ledger (`N = 8 In -> 8 Out`)

| # | Micro-Unit | Symbol / Item | Exact Location (`file:line`) | Parity & 6-Axis Verification | Verdict |
|---|---|---|---|---|---|
| **1** | `U01` | `vdb-view` crate dependencies (`animation`, `playa-time`, `playa-engine`) | `openvdb-rs/crates/vdb-view/Cargo.toml:22-28` | Replaced `playa-time` with `animation = { git = "ssh://git@github.com/ssoj13/animation-rs.git", branch = "main" }` (`L26`). Retained `playa-engine` (`default-features = false`, `L23`) alongside `playa-player` (`L22`) because `src/preview.rs:7-14` directly imports `playa_engine::{colour, core, entities::frame}` (matching `soluwell-studio-render/Cargo.toml:29`). Zero `playa_time` imports remain anywhere in `crates/vdb-view/src/**`. | **VERIFIED** |
| **2** | `U01` | `camera_orbit::integrate` & `Channel` / `Keyframe` imports | `openvdb-rs/crates/vdb-view/src/camera_orbit.rs:3, 7-12, 18` | Switched `playa_engine::entities::anim::{Channel, Keyframe}` to `animation::anim::Channel` (`L3`) and `animation::anim::Keyframe` (`L18`). | **VERIFIED** |
| **3** | `U01` | `camera_recorder` `Animation` & `Channel` imports | `openvdb-rs/crates/vdb-view/src/camera_recorder.rs:9` | Switched `playa_engine::entities::anim::{Animation, Channel}` to `animation::anim::{Animation, Channel}` (`L9`). | **VERIFIED** |
| **4** | `U02` | `world.rs` `animation::{anim, attrs, transform}` imports, `local_pose`, `world_matrix` | `openvdb-rs/crates/vdb-view/src/world.rs:7-8, 22-27, 5154-5192` | Switched top-level imports to `animation::anim::{Animation, Channel, Keyframe}` (`L7`) and `animation::attrs::{AttrValue, Attrs}` (`L8`); updated `local_pose` (`L24`) to `animation::transform::decompose_pose` and `world_matrix` (`L5184`) to `animation::transform::build_model_matrix`. | **VERIFIED** |
| **5** | `U02` | `WorldEditor::navigate_camera` | `openvdb-rs/crates/vdb-view/src/world.rs:1458-1525` | 1:1 parity with `soluwell/studio/scene/src/world.rs:1830-1895`: defines `POSE_PATHS` (`["/transform/position", "/transform/rotation_degrees", "/camera/distance"]`, `L1465-1469`), gathers and deduplicates `seed_frames` across all 3 pose paths with `1e-6` epsilon (`L1475-1480`), computes `pose_animated = !seed_frames.is_empty()` (`L1481`), sets `force_key = auto_key \|\| (is_pose_path && pose_animated)` (`L1485`), and passes `(force_key, editor.new_key, seeds)` to `write_navigation_value` (`L1491-1498`). | **VERIFIED** |
| **6** | `U02` | `WorldDocument::write_navigation_value` | `openvdb-rs/crates/vdb-view/src/world.rs:5014-5114` | 1:1 parity with `soluwell/studio/scene/src/world.rs:5426-5526`: accepts `nav_key: (bool, Tan, &[f64])`; compensates any pre-existing legacy `offset_path` (`L5030-5041`) without ever writing `/_navigation/*` attributes; seeds hold keys `(seed_t, raw as f32, seed_tan)` for newly animated channels (`(None, None) => true`) and exterior intervals (`frame > t1 + 1e-6 && seed_t > t1 + 1e-6 && seed_t < frame - 1e-6` or `frame < t0 - 1e-6 && seed_t < t0 - 1e-6 && seed_t > frame + 1e-6`, `L5077-5095`); writes `(wanted - added) as f32` to the animation channel (`L5097`) or static `base` (`L5099-5112`). | **VERIFIED** |
| **7** | `U02` | `WorldDocument::camera_pose` | `openvdb-rs/crates/vdb-view/src/world.rs:5196-5238` | 1:1 parity with `soluwell/studio/scene/src/world.rs:5626-5674`: `evaluated_node(id, frame)` (`L5200`) evaluates `/camera/distance` at `frame` into `camera.distance`; when `m == glam::Mat4::IDENTITY` (`L5209`), `camera.distance` is preserved as evaluated; when `m != glam::Mat4::IDENTITY`, `camera.distance *= scale` (`L5225`), matching the inverse `after.camera.distance / back_scale` in `camera_navigation_pose` (`L4911`). | **VERIFIED** |
| **8** | `U03` | `world_tests.rs` `animation::*` migration, `/_navigation/*` test updates, and 4-stage camera animation test | `openvdb-rs/crates/vdb-view/src/world_tests.rs:334, 480-616, 1203-1215, 1465-1474, 2699, 2749, 4012-4118` | Migrated all `AttrConnection` (`L334, 2699, 2749`) and `build_model_matrix` (`L1203, 1209, 1465`) references to `animation::*`; updated `camera_navigation_without_auto_key_keys_animated_pose_and_leaves_no_navigation_offsets` (`L480-541`) and `camera_navigation_auto_key_compensates_existing_offsets_and_keys_changed_components_only` (`L544-616`); added `manual_camera_keying_and_viewport_navigation_animate_pose_and_distance` (all 4 stages including 3-pose hold-key seeding, `L4012-4118`). | **VERIFIED** |

---

## 2. Files Inspected

- `C:\projects\projects.rust.cg\cglibs\openvdb-rs\crates\vdb-view\Cargo.toml` (`L1-90`)
- `C:\projects\projects.rust.cg\cglibs\openvdb-rs\crates\vdb-view\src\camera_orbit.rs` (`L1-46`)
- `C:\projects\projects.rust.cg\cglibs\openvdb-rs\crates\vdb-view\src\camera_recorder.rs` (`L1-45`)
- `C:\projects\projects.rust.cg\cglibs\openvdb-rs\crates\vdb-view\src\preview.rs` (`L1-60`, `L400-480`)
- `C:\projects\projects.rust.cg\cglibs\openvdb-rs\crates\vdb-view\src\world.rs` (`L1-50`, `L520-600`, `L1450-1535`, `L4700-5240`, `L5440-5520`)
- `C:\projects\projects.rust.cg\cglibs\openvdb-rs\crates\vdb-view\src\world_tests.rs` (`L1-100`, `L300-660`, `L1020-1070`, `L1200-1250`, `L1460-1540`, `L2695-2775`, `L2900-4118`)
- `C:\projects\projects.rust.cg\cglibs\egui-widgets-rs\crates\soluwell\studio\scene\src\world.rs` (`L1-80`, `L1825-1925`, `L3250-3300`, `L4830-4860`, `L5300-5680`, `L5900-5980`, `L6550-6668`)

---

## 3. Files Changed

1. `C:\projects\projects.rust.cg\cglibs\openvdb-rs\crates\vdb-view\Cargo.toml`:
   - Added `animation = { git = "ssh://git@github.com/ssoj13/animation-rs.git", branch = "main" }`.
   - Removed `playa-time`.
   - Kept `playa-engine = { git = "ssh://git@github.com/ssoj13/playa.git", branch = "main", default-features = false }` alongside `playa-player` for `src/preview.rs`.
2. `C:\projects\projects.rust.cg\cglibs\openvdb-rs\crates\vdb-view\src\camera_orbit.rs`:
   - Switched `playa_engine::entities::anim::{Channel, Keyframe}` to `animation::anim::{Channel, Keyframe}`.
3. `C:\projects\projects.rust.cg\cglibs\openvdb-rs\crates\vdb-view\src\camera_recorder.rs`:
   - Switched `playa_engine::entities::anim::{Animation, Channel}` to `animation::anim::{Animation, Channel}`.
4. `C:\projects\projects.rust.cg\cglibs\openvdb-rs\crates\vdb-view\src\world.rs`:
   - Switched `playa_engine::entities::*` imports (`Animation, Channel, Keyframe, AttrValue, Attrs, transform::{decompose_pose, build_model_matrix}`) to `animation::{anim, attrs, transform}`.
   - Ported `WorldEditor::navigate_camera` (`POSE_PATHS`, `seed_frames`, `pose_animated`, `write_navigation_value` call) and `WorldDocument::write_navigation_value` (3-pose `should_seed` hold-key seeding, zero `/_navigation/*` writes) from `soluwell-studio-scene`.
5. `C:\projects\projects.rust.cg\cglibs\openvdb-rs\crates\vdb-view\src\world_tests.rs`:
   - Switched `playa_engine::entities::*` references to `animation::*`.
   - Updated `camera_navigation_without_auto_key_keys_animated_pose_and_leaves_no_navigation_offsets` and `camera_navigation_auto_key_compensates_existing_offsets_and_keys_changed_components_only`.
   - Added `manual_camera_keying_and_viewport_navigation_animate_pose_and_distance` (4 stages).

---

## 4. Inquisitor Ratchet Proof (`Pass 1 -> Repair -> Pass 2`)

- **Pass 1 (`0d0bae7b-0a04-4527-8231-df0bc0d70b4e`)**:
  - **Verdict**: `REJECTED (1 defect found)`
  - **Defect 1 (`Q2` / `Q7` — Broken Crate Wiring)**: Removing `playa-engine` from `crates/vdb-view/Cargo.toml` broke `crates/vdb-view/src/preview.rs:7-14`, which still imports `playa_engine::{colour::{ColorSpaceId, Tier}, core::{CacheManager, GlobalFrameCache, global_cache::TierCache}, entities::{CacheStrategy, frame::{Frame as PlayaFrame, PixelBuffer, PixelFormat, Premult}}}` (unlike `warpbro-rs`, `vdb-view` still compiles its own `preview.rs`, just like `soluwell-studio-render/Cargo.toml:29`).
- **Repair Applied**:
  - Restored `playa-engine = { git = "ssh://git@github.com/ssoj13/playa.git", branch = "main", default-features = false }` in `crates/vdb-view/Cargo.toml:23` next to `playa-player`, while keeping `animation` (`L26`) and keeping `playa-time` removed.
- **Pass 2 (`97a3a0ca-c7ae-4d61-8a8a-621f54c97baf`)**:
  - **Verdict**: `APPROVED (0 defects across all 8 symbols and Q1..Q7)`
  - **Q1 (Unfinished Code & Stubs)**: 0 defects.
  - **Q2 (Dead Wiring & Uncalled Code)**: 0 defects (`Cargo.toml` matches all crate imports across `src/**`; `offset_path` in `write_navigation_value` is read-only for legacy offset compensation and tested at `world_tests.rs:544-573`).
  - **Q3 (Swallowed Errors & Panic Traps)**: 0 defects (all fallible ops return `Result<_, String>` inside `apply_edit` rollback guards).
  - **Q4 (Logic, Math & Boundary Flaws)**: 0 defects (`seed_frames` uses `f64::total_cmp` + `1e-6` dedup; `should_seed` anchors unkeyed channels and exterior intervals without clobbering interior keys; `camera_pose` and `camera_navigation_pose` are exact inverses for `camera.distance`).
  - **Q5 (Hot-Path Memory Churn)**: 0 defects (`POSE_PATHS` is `const`; `seed_frames` is allocated once per interactive edit outside the channel loop).
  - **Q6 (Memory Layout & Cache/SIMD Hazards)**: 0 defects.
  - **Q7 (Author Uncertainty)**: 0 defects.

---

## 5. Cross-Domain Seam Questions (For Tier 0 `SEAM_MATRIX.md`)

1. **`animation` Git Dependency Lock (`openvdb-rs/Cargo.lock`)**: When Tier 0 runs `cargo check -p vdb-view --config "profile.dev.debug=0" --quiet` in `openvdb-rs`, Cargo will resolve `animation` (`ssh://git@github.com/ssoj13/animation-rs.git#main`) alongside `playa-engine` (`ssh://git@github.com/ssoj13/playa.git#main`). Because `playa-engine` (`9b60383` / `D01`) re-exports `animation::*` from the same `animation-rs` `main` branch, verify that `openvdb-rs/Cargo.lock` resolves a single unified `animation` crate version.
2. **Test Verification Command**: Run `cargo test -p vdb-view --config "profile.dev.debug=0" --lib manual_camera_keying_and_viewport_navigation_animate_pose_and_distance --quiet` as well as `camera_navigation_without_auto_key_keys_animated_pose_and_leaves_no_navigation_offsets` and `camera_navigation_auto_key_compensates_existing_offsets_and_keys_changed_components_only` to confirm green execution.
