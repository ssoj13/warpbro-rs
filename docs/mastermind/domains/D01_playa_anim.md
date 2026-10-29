# Domain D01 Ledger: `animation-rs` & `playa` SSOT Deduplication

## 1. Micro-Unit & Symbol Ledger (`1..7`)

### Micro-Unit `U01` (`animation-rs` Standalone Ownership & Doc Hygiene)
1. **`animation-rs/src/time/mod.rs:1-40` — Module doc header & SSOT exports (`bar_span`, `conform`, `rescale`)**
   - **Action**: Rewrote module header (`animation-rs/src/time/mod.rs:1-26`) to document `animation::time` as the standalone Single Source of Truth for timeline frame arithmetic, rational frame rates (`Fps`), speed clamping (`Speed`), `TimeMap`, `Timecode`, and explicit `Round` conversions, removing all stale `playa-time` / `playa-coord` self-references. Exported `pub fn conform` (`animation-rs/src/time/mod.rs:77-79`) and `pub fn rescale` (`animation-rs/src/time/mod.rs:85-96`) alongside `pub fn bar_span` (`animation-rs/src/time/mod.rs:48-70`).
2. **`animation-rs/src/attrs.rs:1140-1155` (and surrounding lines `524-546`, `1089-1093`, `1716-1724`, `1909-1910`) — Doc comments referencing `playa` instead of `animation`**
   - **Action**: Removed stale `playa` / `Comp::set_layer_attr` / `playa_io::AttrKv` / `crate::ofx` / `timing::bar_span` references across `Attrs::key_at` (`animation-rs/src/attrs.rs:1140-1149`), `Attrs::eval_at` (`1089-1093`), `AttrValue` (`524-546`), and `Attrs::bar` (`1716-1724`, pointing intra-doc links to `[`crate::time::bar_span`]`).
3. **`animation-rs/src/keys.rs:85-120` (and surrounding lines `1-7`, `56-64`) — Doc comments referencing `playa` instead of `animation`**
   - **Action**: Cleaned `animation-rs/src/keys.rs:89-115` (`A_SHY`, `A_QUALITY`, `A_FX`, `A_MOTION_BLUR`, `A_ADJUSTMENT`, `A_3D`, `A_PRESERVE_TRANSP`, `A_PARENT`), `keys.rs:1-7`, and `keys.rs:56-64` (`A_TRIM_OUT`, `A_FPS` referencing `[`crate::time::bar_span`]`) so all attribute key constants document `animation` domain semantics with zero stale `playa` milestone notes.

### Micro-Unit `U02` (`playa-time` & `playa-engine` Dependency & Timing SSOT Deduplication)
4. **`playa/crates/playa-time/Cargo.toml:1-10` — Unused `glam` and `serde` dependencies**
   - **Action**: Removed unused `glam` and `serde` dependencies from `playa/crates/playa-time/Cargo.toml:8-10`, leaving only `animation = { workspace = true }` (`playa-time/src/lib.rs:36-39` is a pure `#![forbid(unsafe_code)]` re-export of `animation::time::*`).
5. **`playa/crates/playa-engine/Cargo.toml:68-75` — Unused `bitflags` dependency**
   - **Action**: Verified via `grep_files` across `playa/crates/playa-engine/src/**` that `bitflags` has `0` matches (`AttrFlags` lives in `animation::attrs`), and removed `bitflags = "2"` from `playa/crates/playa-engine/Cargo.toml:70-73`.
6. **`playa/crates/playa-engine/src/entities/timing.rs:624-627` — `bar_span` SSOT re-export**
   - **Action**: Replaced duplicate `bar_span` implementation with `pub use animation::time::bar_span;` (`timing.rs:627`), making `animation::time::bar_span` the Single Source of Truth.
7. **`playa/crates/playa-engine/src/entities/timing.rs:629-656` — `conform` and `rescale` SSOT delegation**
   - **Action**: Verified identical parameter types, units, and `Round` policies against `animation::time::{conform, rescale}` (`animation-rs/src/time/mod.rs:77-96`) and delegated `timing::conform` (`timing.rs:632-635`) and `timing::rescale` (`timing.rs:653-656`) directly to `animation::time::{conform, rescale}`.

---

## 2. Files Inspected
- `C:\projects\projects.rust.cg\cglibs\animation-rs\src\time\mod.rs` (`1..117`)
- `C:\projects\projects.rust.cg\cglibs\animation-rs\src\attrs.rs` (`1..1950`)
- `C:\projects\projects.rust.cg\cglibs\animation-rs\src\keys.rs` (`1..260`)
- `C:\projects\projects.rust.cg\cglibs\playa\crates\playa-time\Cargo.toml` (`1..10`)
- `C:\projects\projects.rust.cg\cglibs\playa\crates\playa-time\src\lib.rs` (`1..39`)
- `C:\projects\projects.rust.cg\cglibs\playa\crates\playa-engine\Cargo.toml` (`1..104`)
- `C:\projects\projects.rust.cg\cglibs\playa\crates\playa-engine\src\entities\timing.rs` (`1..1040`)

---

## 3. Files Changed
- `C:\projects\projects.rust.cg\cglibs\animation-rs\src\time\mod.rs`
- `C:\projects\projects.rust.cg\cglibs\animation-rs\src\attrs.rs`
- `C:\projects\projects.rust.cg\cglibs\animation-rs\src\keys.rs`
- `C:\projects\projects.rust.cg\cglibs\playa\crates\playa-time\Cargo.toml`
- `C:\projects\projects.rust.cg\cglibs\playa\crates\playa-engine\Cargo.toml`
- `C:\projects\projects.rust.cg\cglibs\playa\crates\playa-engine\src\entities\timing.rs`
- `C:\projects\projects.rust.cg\cglibs\warpbro-rs\docs\mastermind\domains\D01_playa_anim.md`

---

## 4. Inquisitor Ratchet Proof (`Q1..Q7` Verification)
Audited by independent read-only Tier 2 Inquisitor (`conversation://5b1d0f67-8990-473c-845d-257fe0d33b76`):
- **`Q1` (Completeness & Anti-Stubbing)**: **PASS.** Zero `todo!()`, `unimplemented!()`, `// TODO`, placeholder constants, or half-wired branches across all 7 files.
- **`Q2` (Dead Code & Phantom Symbols)**: **PASS.** `animation::time::{bar_span, conform, rescale}` (`animation-rs/src/time/mod.rs:48-96`) are public SSOT primitives consumed by `playa-engine/src/entities/timing.rs:627-656` (`bar_span` re-exported at `timing.rs:627`; `conform` called by `conform_frames` at `timing.rs:677` and unit tests at `timing.rs:854-882`; `rescale` called by `to_local` at `timing.rs:480`, `to_parent` at `timing.rs:508`, `key_frame_for` at `timing.rs:556`, `key_frame_to_parent` at `timing.rs:574`, `rescale_pos` at `timing.rs:666`, and `conform_to_source` at `timing.rs:706`).
- **`Q3` (Error & Panic Discipline)**: **PASS.** Zero `.unwrap()`, `.expect()`, `.ok()`, or `let _ =` in production code across `animation-rs/src/time/mod.rs:1-97` and `playa-engine/src/entities/timing.rs:1-719`.
- **`Q4` (Boundary, Math & State Transitions)**: **PASS.** `animation::time::bar_span` (`mod.rs:48-70`) floors `visible_src` and `visible` at `1` and uses `saturating_add` / `saturating_sub(1)` to prevent `i32` overflow and inverted spans. `animation::time::rescale` (`mod.rs:85-96`) preserves exact identity (`return n`) when `from_fps == to_fps || !from_fps.is_finite() || !to_fps.is_finite() || from_fps <= 0.0 || to_fps <= 0.0` and evaluates in `f64` before `policy.to_i32(...)`.
- **`Q5` (Hot-Path Churn & Blocking)**: **PASS.** `bar_span`, `conform`, and `rescale` are allocation-free, lock-free stack scalar operations (`#[inline]` delegates in `timing.rs:632, 653`).
- **`Q6` (Memory Layout & Cache Locality)**: **PASS.** `Round`, `Speed`, `Fps`, and `LayerTiming` remain compact `Copy` value types; no layout changes.
- **`Q7` (Seam & Contract Integrity)**: **PASS.** `playa_time` re-exports `animation::time::*` (`playa-time/src/lib.rs:38`), so `playa_time::Round` and `animation::time::Round` are identical types, and `bar_span` / `conform` / `rescale` preserve identical parameter order, inclusive frame conventions, and `Round::Ceil` semantics across `animation` and `playa-engine`.

---

## 5. Cross-Domain Seam Questions
- None. All `animation::time` ↔ `playa-time` ↔ `playa-engine::entities::timing` contracts are unified under `animation::time` SSOT with zero open seam questions.
