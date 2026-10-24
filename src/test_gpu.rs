//! CUDA, Vulkan Video, OCIO and OIDN tests share one physical adapter.
//!
//! Naming contract: a test that needs a CUDA device is named `cuda_*`; one that needs another
//! GPU API (a wgpu adapter, Vulkan Video) is named `gpu_*`. Hosted CI has no GPU and runs the
//! ordinary suite with `--skip cuda_ --skip gpu_` (`bootstrap.py` GPU_TEST_PREFIXES); a device
//! test under another name fails there ("Render worker is unavailable", "Vulkan Video
//! unavailable", a wgpu validation error). libtest's `--skip` is a substring match, so the
//! prefixes must not occur anywhere else in a test name either (`wgpu_*` contains `gpu_`):
//! [`device_prefixes_only_start_test_names`] enforces both rules.
use std::sync::{Mutex, MutexGuard};

/// Hold until all GPU objects and background render workers from a test have been dropped.
/// A per-module lock cannot isolate CUDA/render workers from Vulkan Video device creation.
pub fn lock() -> MutexGuard<'static, ()> {
    static ADAPTER: Mutex<()> = Mutex::new(());
    ADAPTER.lock().unwrap_or_else(|error| error.into_inner())
}

/// The naming contract over every `#[test]` in `src/`: `gpu_` / `cuda_` start a device test's
/// name and occur in no other test name, so `--skip cuda_ --skip gpu_` drops exactly the device
/// tests (a GPU-free `wgpu_*` test used to be skipped with them and never ran).
#[test]
fn device_prefixes_only_start_test_names() {
    const PREFIXES: [&str; 2] = ["gpu_", "cuda_"];
    let mut bad = Vec::new();
    let mut dirs = vec![std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src")];
    while let Some(dir) = dirs.pop() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                dirs.push(path);
                continue;
            }
            if path.extension().is_none_or(|ext| ext != "rs") {
                continue;
            }
            let text = std::fs::read_to_string(&path).unwrap();
            // Attributes and doc comments may sit between `#[test]` and its `fn`.
            let mut test_next = false;
            for line in text.lines().map(str::trim) {
                if line == "#[test]" {
                    test_next = true;
                } else if test_next && let Some(rest) = line.strip_prefix("fn ") {
                    test_next = false;
                    let name = rest.split('(').next().unwrap_or(rest);
                    let device = PREFIXES.iter().any(|p| name.starts_with(p));
                    if !device && PREFIXES.iter().any(|p| name.contains(p)) {
                        bad.push(format!("{}: {name}", path.display()));
                    }
                }
            }
        }
    }
    assert!(
        bad.is_empty(),
        "a GPU-free test name contains a device prefix (`--skip` would drop it): {bad:#?}"
    );
}
