//! CUDA, Vulkan Video, OCIO and OIDN tests share one physical adapter.
//!
//! Naming contract: a test that needs a CUDA device is named `cuda_*`; one that needs another
//! GPU API (a wgpu adapter, Vulkan Video) is named `gpu_*`. Hosted CI has no GPU and runs the
//! ordinary suite with `--skip cuda_ --skip gpu_` (`bootstrap.py` GPU_TEST_PREFIXES); a device
//! test under another name fails there ("CUDA worker is unavailable", "Vulkan Video
//! unavailable", a wgpu validation error).
use std::sync::{Mutex, MutexGuard};

/// Hold until all GPU objects and background render workers from a test have been dropped.
/// A per-module lock cannot isolate CUDA/render workers from Vulkan Video device creation.
pub fn lock() -> MutexGuard<'static, ()> {
    static ADAPTER: Mutex<()> = Mutex::new(());
    ADAPTER.lock().unwrap_or_else(|error| error.into_inner())
}
