//! CUDA, Vulkan Video, OCIO and OIDN tests share one physical adapter.
use std::sync::{Mutex, MutexGuard};

/// Hold until all GPU objects and background render workers from a test have been dropped.
/// A per-module lock cannot isolate CUDA/render workers from Vulkan Video device creation.
pub fn lock() -> MutexGuard<'static, ()> {
    static ADAPTER: Mutex<()> = Mutex::new(());
    ADAPTER.lock().unwrap_or_else(|error| error.into_inner())
}
