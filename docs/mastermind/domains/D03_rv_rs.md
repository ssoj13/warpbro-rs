# Domain D03 Ledger: `rv-rs` — Dead Dependency Pruning & `animation::time::Fps` Migration

## 1. Micro-Unit & Symbol Ledger (`1..5`)

1. **`C:\projects\projects.rust.cg\cglibs\rv-rs\Cargo.toml:73-75` (`[workspace.dependencies]`)**:
   - Pruned dead workspace dependencies `playa-engine` and `playa-ui`.
   - Replaced `playa-time = { git = "ssh://git@github.com/ssoj13/playa.git", branch = "main", package = "playa-time" }` with `animation = { git = "ssh://git@github.com/ssoj13/animation-rs.git", branch = "main" }` (`line 75`), retaining `playa-io` (`line 74`).
   - Verified via `grep_files` across all `crates/*/Cargo.toml` in `rv-rs` that no member crate depends on `playa-engine`, `playa-ui`, or `playa-time`.
2. **`C:\projects\projects.rust.cg\cglibs\rv-rs\Cargo.toml:155-160` (`[workspace.dependencies]`)**:
   - Removed unused `curves = { git = "ssh://git@github.com/ssoj13/curves-rs.git", branch = "main", package = "curves" }` workspace dependency.
   - Verified via `grep_files` across `rv-rs` (`*.toml` and `*.rs`) that no crate or source file references `curves`.
3. **`C:\projects\projects.rust.cg\cglibs\rv-rs\crates\rv-app\Cargo.toml:100-104` (`[dependencies]`)**:
   - Replaced `playa-time = { workspace = true }` with `animation = { workspace = true }` (`line 103`).
4. **`C:\projects\projects.rust.cg\cglibs\rv-rs\crates\rv-app\src\playback.rs:2460-2468` (`movie_tests::fake_meta`)**:
   - Replaced `frame_rate: playa_time::Fps::from_f32_lossy(fps as f32)` with `frame_rate: animation::time::Fps::from_f32_lossy(fps as f32)` (`line 2465`).
5. **Workspace-wide `grep_files` Verification (`C:\projects\projects.rust.cg\cglibs\rv-rs`)**:
   - `*.toml`: 0 active dependency entries remain for `playa-time`, `playa-engine`, `playa-ui`, or `curves` (only historical provenance comments in `Cargo.toml:53,169`, `crates/workpool-rs/Cargo.toml:6`, `crates/nodes-rv/Cargo.toml:17`, and `crates/rv-app/Cargo.toml:169`).
   - `*.rs`: 0 active code references remain to `playa_time`, `playa_engine`, `playa_ui`, or `curves` (only rustdoc provenance comments in `crates/workpool-rs/src/workpool.rs:9-10`, `crates/nodes-rv/src/nodes_rv.rs:29`, and `crates/rv-app/src/playback.rs:7`).

---

## 2. Files Inspected

- `C:\projects\projects.rust.cg\cglibs\rv-rs\Cargo.toml`
- `C:\projects\projects.rust.cg\cglibs\rv-rs\crates\rv-app\Cargo.toml`
- `C:\projects\projects.rust.cg\cglibs\rv-rs\crates\rv-app\src\playback.rs`
- `C:\projects\projects.rust.cg\cglibs\rv-rs\crates\nodes-rv\Cargo.toml`

---

## 3. Files Changed

- `C:\projects\projects.rust.cg\cglibs\rv-rs\Cargo.toml`
- `C:\projects\projects.rust.cg\cglibs\rv-rs\crates\rv-app\Cargo.toml`
- `C:\projects\projects.rust.cg\cglibs\rv-rs\crates\rv-app\src\playback.rs`
- `C:\projects\projects.rust.cg\cglibs\warpbro-rs\docs\mastermind\domains\D03_rv_rs.md`

---

## 4. Inquisitor Ratchet Proof (`Q1..Q7` Verification)

Audited by fresh read-only Tier 2 Inquisitor subagent (`7e7ae8e6-8364-47e0-9382-613235f9786a`, `Defect count = 0`):

- **Q1 (Completeness & Zero Stubs): `PASS`** — All 5 manifest items are completely implemented across `Cargo.toml:74-75,155-160`, `crates/rv-app/Cargo.toml:103`, `crates/rv-app/src/playback.rs:2465`, and `crates/nodes-rv/Cargo.toml:1-25` with zero stubs, `TODO`s, or placeholders.
- **Q2 (Dead Code & Unused Dependencies): `PASS`** — `playa-engine`, `playa-ui`, `playa-time`, and `curves` are completely pruned from `[workspace.dependencies]` and all crate `Cargo.toml` manifests in `rv-rs`.
- **Q3 (Error Handling & Panic Safety): `PASS`** — `animation::time::Fps::from_f32_lossy(fps as f32)` at `crates/rv-app/src/playback.rs:2465` is an infallible constructor inside `#[cfg(test)] mod movie_tests::fake_meta`; zero `.unwrap()`, `.expect()`, panics, or swallowed errors were introduced.
- **Q4 (Type & Contract Parity): `PASS`** — `VideoMetadata.frame_rate` in `playa-io` uses `animation::time::Fps` (which `playa-time` re-exports from the identical git dependency `ssh://git@github.com/ssoj13/animation-rs.git`, `branch = "main"`), so `animation::time::Fps` is nominally identical to `VideoMetadata.frame_rate`.
- **Q5 (Hot-Path Memory Churn): `PASS`** — Zero allocations or runtime hot-path changes introduced.
- **Q6 (Memory Layout & Cache Alignment): `PASS`** — Zero struct layout or ABI modifications.
- **Q7 (Cross-Module Seam Integrity): `PASS`** — `rv-rs/Cargo.toml:75` and `crates/rv-app/Cargo.toml:103` point to the canonical `animation-rs` git repository (`branch = "main"`), unifying with `playa-io`'s `animation` dependency in Cargo's dependency graph.

---

## 5. Cross-Domain Seam Questions

- **`playa-io` ↔ `rv-app` `Fps` type unification**: `rv-rs` and `playa` both consume `animation = { git = "ssh://git@github.com/ssoj13/animation-rs.git", branch = "main" }`. Once `D01` (`playa`) completes any `playa-io` migration from `playa-time` to `animation`, `cargo update -p playa-io -p animation` in `rv-rs` during Tier 0 verification will keep the lockfile unified on the latest `animation-rs` commit.
