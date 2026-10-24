//! Inspect the persisted Vulkan cache against the driver's accepted live cache.
//! Run through the machine build lock; this creates one tiny compute pipeline.
fn main() {
    let gpu = gpu_info::shared_device().expect("hardware device");
    let info = gpu.adapter.get_info();
    let key = wgpu::util::pipeline_cache_key(&info).expect("persistent Vulkan cache");
    let root = std::env::var_os("LOCALAPPDATA")
        .map(std::path::PathBuf::from)
        .expect("LOCALAPPDATA");
    let path = root.join("gpu-info/wgpu-pipelines").join(key);
    let disk = || {
        std::fs::metadata(&path)
            .ok()
            .map(|m| (m.len(), m.modified().ok()))
    };
    let cache = gpu_info::pipeline_cache(&gpu.device, &info).expect("pipeline cache enabled");
    println!(
        "adapter={} driver={} disk_before={:?} accepted_live_bytes={:?}",
        info.name,
        info.driver_info,
        disk(),
        cache.cache().get_data().map(|d| d.len())
    );
    let module=gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label:Some("pipeline cache probe"),
        source:wgpu::ShaderSource::Wgsl("@group(0) @binding(0) var<storage,read_write> p:array<f32>; @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) i:vec3<u32>) {p[i.x]=sqrt(p[i.x])+0.3125;}".into()),
    });
    let started = std::time::Instant::now();
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
    println!(
        "compile_seconds={:.6} live_after={:?} disk_after={:?}",
        started.elapsed().as_secs_f64(),
        cache.cache().get_data().map(|d| d.len()),
        disk()
    );
}
