//! Diagnostic probe for `gpu_info`'s persistent Vulkan compute-pipeline cache.
//!
//! Exists to verify (`AGENTS.md` pipeline-cache invariant) that compiling a compute shader via
//! [`gpu_info::create_compute_pipeline`] persists the driver's cache blob to disk AND that a
//! freshly opened [`gpu_info::pipeline_cache`] handle accepts those persisted bytes as live driver
//! seed data (a large disk file alone does not prove driver acceptance).
//!
//! Opens a pre-compile [`gpu_info::pipeline_cache`] handle on [`gpu_info::shared_device`], compiles
//! one tiny WGSL compute shader through [`gpu_info::create_compute_pipeline`] (which opens its own
//! `DiskPipelineCache` and flushes to `<cache_dir>/gpu-info/wgpu-pipelines/<key>`), and then
//! re-opens [`gpu_info::pipeline_cache`] so `live_after` measures the newly persisted seed rather
//! than the stale pre-compile `wgpu::PipelineCache` handle.
//!
//! Run manually under the machine build lock:
//! `cargo oxide run --release --example pipeline_cache_probe`

use std::path::PathBuf;
use std::time::{Instant, SystemTime};

const WGSL: &str = "@group(0) @binding(0) var<storage,read_write> p:array<f32>; @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) i:vec3<u32>) {p[i.x]=sqrt(p[i.x])+0.3125;}";

/// User cache root matching `gpu_info::pipeline_cache`'s platform directory resolution.
fn cache_dir() -> Option<PathBuf> {
    let var = |name: &str| std::env::var_os(name).filter(|value| !value.is_empty());
    if cfg!(windows) {
        var("LOCALAPPDATA").map(PathBuf::from)
    } else if cfg!(target_os = "macos") {
        var("HOME").map(|home| PathBuf::from(home).join("Library").join("Caches"))
    } else {
        var("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .or_else(|| var("HOME").map(|home| PathBuf::from(home).join(".cache")))
    }
}

fn main() -> Result<(), String> {
    let gpu = gpu_info::shared_device().ok_or("hardware device unavailable")?;
    let info = gpu.adapter.get_info();
    let key = wgpu::util::pipeline_cache_key(&info)
        .ok_or("persistent Vulkan pipeline cache unsupported on adapter")?;
    let root = cache_dir().ok_or("user cache directory unavailable")?;
    let path = root.join("gpu-info").join("wgpu-pipelines").join(key);
    let disk = || -> Result<Option<(u64, SystemTime)>, String> {
        match std::fs::metadata(&path) {
            Ok(meta) => {
                let modified = meta
                    .modified()
                    .map_err(|err| format!("modified {}: {err}", path.display()))?;
                Ok(Some((meta.len(), modified)))
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(err) => Err(format!("stat {}: {err}", path.display())),
        }
    };
    let before = gpu_info::pipeline_cache(&gpu.device, &info)
        .ok_or("pipeline cache feature disabled on device")?;
    println!(
        "adapter={} driver={} disk_before={:?} accepted_live_bytes={:?}",
        info.name,
        info.driver_info,
        disk()?,
        before.cache().get_data().map(|d| d.len()),
    );
    drop(before);

    let module = gpu
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pipeline cache probe"),
            source: wgpu::ShaderSource::Wgsl(WGSL.into()),
        });
    let started = Instant::now();
    let _pipeline = gpu_info::create_compute_pipeline(
        &gpu.device,
        &info,
        &wgpu::ComputePipelineDescriptor {
            label: Some("pipeline cache probe"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        },
    );
    let compile_seconds = started.elapsed().as_secs_f64();
    let after = gpu_info::pipeline_cache(&gpu.device, &info)
        .ok_or("pipeline cache feature disabled on device after compile")?;
    println!(
        "compile_seconds={compile_seconds:.6} live_after={:?} disk_after={:?}",
        after.cache().get_data().map(|d| d.len()),
        disk()?,
    );
    Ok(())
}
