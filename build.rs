//! The kernels in `src/gpu.rs` are compiled to PTX by the cuda-oxide rustc backend, which only
//! `cargo oxide` enables. A plain `cargo build` would otherwise fail much later, at link time,
//! with "undefined symbol: cuda_oxide_artifact_anchor_*"; stop early with the fix instead.

fn main() {
    println!("cargo::rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");
    println!("cargo::rerun-if-env-changed=FRAC_ALLOW_PLAIN_CARGO");
    // Type-checking front ends never link, so they need no backend: clippy, rust-analyzer, and
    // an explicit opt-out for plain `cargo check`.
    if ["CLIPPY_ARGS", "RA_RUSTC_WRAPPER", "FRAC_ALLOW_PLAIN_CARGO"].iter().any(|v| std::env::var_os(v).is_some()) {
        return;
    }
    let flags = std::env::var("CARGO_ENCODED_RUSTFLAGS").unwrap_or_default();
    let oxide = flags
        .split('\u{1f}')
        .any(|f| f.contains("codegen-backend") && f.contains("rustc_codegen_cuda"));
    if !oxide {
        panic!(
            "\n\nfrac-rs must be built with cuda-oxide (its GPU kernels are Rust compiled to PTX):\n\
             \n    cargo oxide build     # -> target/release/frac-rs   (alias: cargo ob)\
             \n    cargo oxide run       #                             (alias: cargo or)\
             \n    python bootstrap.py b # checks the toolchain first\n\
             \nPlain `cargo build` / `cargo run` does not enable the CUDA codegen backend.\n\
             Setup: see README.md (Requirements).\n"
        );
    }
}
