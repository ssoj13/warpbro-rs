# Domain `D04`: `skelly-rs` — `animation-rs` Substrate Adoption

## 1. Micro-Unit & Symbol Ledger (`1..7`)

1. **`skelly-rs/Cargo.toml:1-9` (`[workspace.dependencies] animation`)**:
   - Added `animation = { git = "ssh://git@github.com/ssoj13/animation-rs.git", branch = "main" }` under `[workspace.dependencies]` (`C:\projects\projects.rust.cg\cglibs\skelly-rs\Cargo.toml:8-9`).
2. **`skelly-rs/crates/skelly-rig/Cargo.toml:1-10` & `src/lib.rs:1-99` (`animation` re-export)**:
   - Added `animation.workspace = true` to `[dependencies]` in `C:\projects\projects.rust.cg\cglibs\skelly-rs\crates\skelly-rig\Cargo.toml:8`.
   - Added `pub use animation;` at `C:\projects\projects.rust.cg\cglibs\skelly-rs\crates\skelly-rig\src\lib.rs:19`, exposing `animation::{anim, attrs, transform, time, curves}` to rig consumers alongside `Skeleton` (`lib.rs:28-72`), `flat_unit` (`lib.rs:75-82`), `lateral` (`lib.rs:85-92`), and `biped` exports (`crates/skelly-rig/src/biped.rs:1-281`: `Bone`, `bones`, `Pose`, `LegJoints`, `SpineJoints`, `ArmJoints`, `legs`, `spine`, `arms`, `bind_pose`, `WalkParams`, `simulate`).
3. **`skelly-rs/crates/skelly-nodes/Cargo.toml:1-12` & `src/lib.rs:1-53` (`animation` re-export)**:
   - Added `animation.workspace = true` to `[dependencies]` in `C:\projects\projects.rust.cg\cglibs\skelly-rs\crates\skelly-nodes\Cargo.toml:8`.
   - Added `pub use animation;` at `C:\projects\projects.rust.cg\cglibs\skelly-rs\crates\skelly-nodes\src\lib.rs:21`, exposing `animation::{anim, attrs, transform, time, curves}` for keyframed node attributes and timing alongside `PREVIEW_KEY` (`lib.rs:28`) and `Playhead` (`lib.rs:33-53`).
4. **`skelly-rs/crates/skelly/Cargo.toml:1-25` (`animation` dependency)**:
   - Added `animation.workspace = true` to `[dependencies]` in `C:\projects\projects.rust.cg\cglibs\skelly-rs\crates\skelly\Cargo.toml:12`.
5. **`skelly-rs/crates/skelly/src/app.rs:22-224` (`Host`, `Host::new`, `Host::effective_fps`, `Host::time`, `Host::step`, `Host::advance`, `Host::wrap`, `Host::frame_index`, `Host::cook`, `Host::transport`)**:
   - Exposed `pub fps: f32`, `pub first_frame: i64`, `pub last_frame: i64`, `pub playing: bool` on `Host` (`crates/skelly/src/app.rs:22-39`) for egui `DragValue` compatibility.
   - Added `Host::effective_fps(&self) -> f64` (`crates/skelly/src/app.rs:94-96`) using `f64::from(Fps::from_f32_lossy(self.fps).as_f32())`.
   - Added `Host::time(&self) -> f64` (`crates/skelly/src/app.rs:100-113`), sanitizing `fps` via `effective_fps()`, pinning to `start` when `last_frame <= first_frame` or `!self.seconds.is_finite()`, and wrapping via `rem_euclid(span)`.
   - Added `Host::step(&mut self, dt: f64)` (`crates/skelly/src/app.rs:116-123`), filtering negative/non-finite `dt`, advancing from `self.time()`, and wrapping via `self.wrap()` (`app.rs:132-134`).
   - Updated `Host::frame_index(&self) -> i64` (`app.rs:136-138`), `Host::cook(&mut self)` (`app.rs:140-165`, also fixing initial `shown: f64::NAN` IEEE-754 comparison trap), and `Host::transport(&mut self, ui: &mut egui::Ui)` (`app.rs:191-224`) to use `Fps::from_f32_lossy` and `effective_fps()`.
6. **`skelly-rs/crates/skelly/src/app.rs:248-329` (Unit tests for `Host::time()` and `Host::step()`)**:
   - Added `#[cfg(test)] fn new_headless() -> Self` (`app.rs:75-92`) with `editor: Option<EditorWidget>` so `Host` timing can be unit-tested without requiring a live GPU `egui_wgpu::RenderState`.
   - Added `host_time_and_step_normal_fps` (`app.rs:252-271`, testing `24.0` and `60.0` FPS), `host_time_and_step_degenerate_fps` (`app.rs:274-294`, testing `0.0`, `-10.0`, `f32::NAN`, `f32::INFINITY`, `f32::NEG_INFINITY`), and `host_step_wraps_around_frame_range` (`app.rs:297-328`, testing `[0..48]` wrap, non-zero `[24..72]` wrap, and single-frame `[10..10]` pinning).
7. **Module-level `//!` documentation headers**:
   - Updated `C:\projects\projects.rust.cg\cglibs\skelly-rs\crates\skelly-rig\src\lib.rs:1-12`, `C:\projects\projects.rust.cg\cglibs\skelly-rs\crates\skelly-nodes\src\lib.rs:1-16`, and `C:\projects\projects.rust.cg\cglibs\skelly-rs\crates\skelly\src\app.rs:1-7` with complete `//!` headers explaining what each module is, why it exists, how it works, what it uses, and where it is used.

---

## 2. Files Inspected

- `C:\projects\projects.rust.cg\cglibs\skelly-rs\Cargo.toml`
- `C:\projects\projects.rust.cg\cglibs\skelly-rs\crates\skelly\Cargo.toml`
- `C:\projects\projects.rust.cg\cglibs\skelly-rs\crates\skelly\src\app.rs`
- `C:\projects\projects.rust.cg\cglibs\skelly-rs\crates\skelly-rig\Cargo.toml`
- `C:\projects\projects.rust.cg\cglibs\skelly-rs\crates\skelly-rig\src\lib.rs`
- `C:\projects\projects.rust.cg\cglibs\skelly-rs\crates\skelly-rig\src\biped.rs`
- `C:\projects\projects.rust.cg\cglibs\skelly-rs\crates\skelly-nodes\Cargo.toml`
- `C:\projects\projects.rust.cg\cglibs\skelly-rs\crates\skelly-nodes\src\lib.rs`

---

## 3. Files Changed

- `C:\projects\projects.rust.cg\cglibs\skelly-rs\Cargo.toml`
- `C:\projects\projects.rust.cg\cglibs\skelly-rs\crates\skelly\Cargo.toml`
- `C:\projects\projects.rust.cg\cglibs\skelly-rs\crates\skelly\src\app.rs`
- `C:\projects\projects.rust.cg\cglibs\skelly-rs\crates\skelly-rig\Cargo.toml`
- `C:\projects\projects.rust.cg\cglibs\skelly-rs\crates\skelly-rig\src\lib.rs`
- `C:\projects\projects.rust.cg\cglibs\skelly-rs\crates\skelly-nodes\Cargo.toml`
- `C:\projects\projects.rust.cg\cglibs\skelly-rs\crates\skelly-nodes\src\lib.rs`
- `C:\projects\projects.rust.cg\cglibs\warpbro-rs\docs\mastermind\domains\D04_skelly_rs.md`

---

## 4. Inquisitor Ratchet Proof (`Q1..Q7`)

Tier 2 Adversarial Inquisitor ([29a96c79-9fa5-4834-ac23-79248347eeb0](conversation://29a96c79-9fa5-4834-ac23-79248347eeb0)) audited `D04` across two passes:
- **Pass 1 Findings & Repairs**:
  1. `Q2 (Dead / Unwired Code)`: `Host::new_headless()` in `[[bin]] skelly` was un-gated and would trigger `dead_code` in non-test builds. Repaired by adding `#[cfg(test)]` at `crates/skelly/src/app.rs:75`.
  2. `Q4 (Boundary & Numeric Edge Cases)`:
     - Single-frame or inverted frame ranges (`last_frame <= first_frame`) previously wrapped across `1.0 / fps` instead of pinning to `start`. Repaired at `crates/skelly/src/app.rs:103-105` and verified with unit test at `app.rs:321-327`.
     - Non-finite `self.seconds` recovery in `Host::step()` previously reset to `0.0` instead of `self.time()` (`start`), causing a backward jump when `first_frame > 0`. Repaired at `crates/skelly/src/app.rs:119-121`.
     - Clamped `first_frame` and `last_frame` at `crates/skelly/src/app.rs:191-192` before passing to `egui::DragValue::range(...)`.
- **Pass 2 Final Verdicts (`Q1..Q7`)**:
  - `Q1 (Unfinished Stubs / Placeholders / TODOs)`: **PASS** (0 stubs/TODOs).
  - `Q2 (Dead / Unwired Code)`: **PASS** (0 unused items in bin or test targets).
  - `Q3 (Swallowed Errors / Panics)`: **PASS** (0 panics or swallowed errors in `ALLOWED_WRITE_FILES`; replaced `editor: EditorWidget` with `Option<EditorWidget>` handled via `if let Some` / `let Some(...) else` without `.unwrap()`).
  - `Q4 (Boundary & Numeric Edge Cases)`: **PASS** (`fps <= 0.0`, `NaN`, `Inf`, `last_frame <= first_frame`, negative/NaN `dt`, and `shown = f64::NAN` all handled and unit-tested).
  - `Q5 (Hot-Path Memory Churn)`: **PASS** (no new allocations introduced).
  - `Q6 (Memory Layout & Thread Safety)`: **PASS**.
  - `Q7 (Documentation & Architectural Invariants)`: **PASS** (complete `//!` headers in all 3 modified source files).

---

## 5. Cross-Domain Seam Questions

1. **`crates/skelly-rig/src/biped.rs:48-53` (`Pose::joint` unchecked index)**: `Pose` exposes `pub joints: Vec<Vec3>` (`biped.rs:45`), and `Pose::joint` indexes `self.joints[i]` directly (`biped.rs:51`) after finding `i` in `bones()` (`0..18`). Because `biped.rs` was in `READ SCOPE` only, a future pass on `skelly-rig` can replace `.map(|i| self.joints[i])` with `.and_then(|i| self.joints.get(i).copied())`.
2. **`crates/skelly-nodes/src/lib.rs:30-53` (`Playhead` time representation)**: `Playhead` currently stores `seconds: Mutex<f64>`. With `animation` now re-exported from both `skelly-rig` and `skelly-nodes`, downstream node evaluation can adopt `animation::time::FrameTime` / `Fps` or `animation::attrs::Attrs` without changing workspace dependencies.
