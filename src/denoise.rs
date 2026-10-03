//! Worker-side OIDN processing of normalized scene-linear render samples.
#[cfg(test)]
mod tests {
    use super::*;
    fn success(samples: u32) -> Result<Output, String> {
        Ok(Output {
            pixels: vec![[samples as f32, 0.0, 0.0, 1.0]],
            latency_ms: 3.5,
        })
    }
    #[test]
    fn cadence_counts_samples_and_runs_final_only_once() {
        let settings = Settings::default();
        let mut state = State::default();
        assert!(!state.due(&settings, 0, true));
        assert!(!state.due(&settings, 127, false));
        assert!(state.due(&settings, 128, false));
        state.complete(128, success(128));
        assert!(!state.due(&settings, 128, true));
        assert!(!state.due(&settings, 255, false));
        assert!(state.due(&settings, 256, false));
        state.complete(256, success(256));
        assert!(state.due(&settings, 300, true));
        state.complete(300, success(300));
        assert!(!state.due(&settings, 300, true));
        assert_eq!(state.last_samples, 300);
    }
    #[test]
    fn failures_clear_stale_output_and_do_not_retry_every_frame() {
        let settings = Settings::default();
        let mut state = State::default();
        assert!(state.due(&settings, 128, false));
        state.complete(128, success(128));
        assert!(state.due(&settings, 256, false));
        state.complete(256, Err("test failure".into()));
        assert!(state.output.is_none());
        assert_eq!(state.error.as_deref(), Some("test failure"));
        assert!(!state.due(&settings, 256, true));
        assert!(!state.due(&settings, 300, false));
        assert!(state.due(&settings, 384, false));
    }
    #[test]
    fn new_generation_discards_old_output_and_cadence() {
        let settings = Settings::default();
        let mut state = State::default();
        assert!(state.due(&settings, 512, false));
        state.complete(512, success(512));
        assert!(!state.due(&settings, 1, false));
        assert!(state.output.is_none());
        assert_eq!(state.last_samples, 0);
        assert!(state.due(&settings, 128, false));
        state.complete(128, success(128));
        state.reset();
        assert!(state.output.is_none());
        assert!(!state.due(&settings, 10, false));
    }
    #[test]
    fn settings_refilter_current_samples_without_trace_reset() {
        let mut settings = Settings::default();
        let mut state = State::default();
        assert!(state.due(&settings, 128, false));
        state.complete(128, success(128));
        settings.interval = 256;
        assert!(!state.due(&settings, 128, false));
        assert!(state.output.is_some());
        settings.quality = Quality::High;
        assert!(state.due(&settings, 128, false));
        state.complete(128, success(128));
        settings.mode = Mode::Color;
        assert!(state.due(&settings, 128, false));
        state.complete(128, Err("mode failure".into()));
        settings.enabled = false;
        assert!(!state.due(&settings, 128, true));
        assert!(state.output.is_none());
        settings.enabled = true;
        assert!(state.due(&settings, 128, false));
    }
    #[test]
    fn interval_zero_keeps_final_pass_and_partial_settings_defaults() {
        let settings: Settings = serde_json::from_str(r#"{"interval":0}"#).unwrap();
        assert!(settings.enabled);
        assert_eq!(settings.mode, Mode::ColorAlbedoNormal);
        let mut state = State::default();
        assert!(!state.due(&settings, 4096, false));
        assert!(state.due(&settings, 4096, true));
        state.complete(4096, success(4096));
        assert!(!state.due(&settings, 4096, true));
    }
    #[test]
    fn invalid_inputs_fail_before_initializing_gpu() {
        let mut processor = Processor::new().unwrap();
        assert!(
            processor
                .process(2, 2, &[[0.0; 4]; 3], None, None, &Settings::default(), 1)
                .is_err()
        );
        assert!(
            processor
                .process(0, 2, &[], None, None, &Settings::default(), 1)
                .is_err()
        );
        assert!(processor.inner.is_none());
    }
    #[test]
    #[ignore = "requires a real shared wgpu adapter and embedded OIDN HDR weights"]
    fn gpu_quality_modes_preserve_hdr_reduce_noise_and_reuse_processor() {
        let (width, height) = (65, 48); // Padded copy rows, not a 256-byte-aligned width.
        let target = [2.5f32, 1.8, 1.2];
        let pixels = (0..width * height)
            .map(|i| {
                let noise = (((i * 1664525 + 1013904223) % 997) as f32 / 996.0 - 0.5) * 1.2;
                [target[0] + noise, target[1] + noise, target[2] + noise, 1.0]
            })
            .collect::<Vec<_>>();
        let albedo = vec![[2.0, 2.0, 2.0, 4.0]; pixels.len()];
        let normal = vec![[0.0, 0.0, 4.0, 4.0]; pixels.len()];
        let mse = |p: &[[f32; 4]]| {
            p.iter()
                .map(|p| (0..3).map(|c| (p[c] - target[c]).powi(2)).sum::<f32>())
                .sum::<f32>()
                / (p.len() * 3) as f32
        };
        let raw_error = mse(&pixels);
        let mut processor = Processor::new().unwrap();
        for quality in Quality::ALL {
            let settings = Settings {
                quality,
                ..Settings::default()
            };
            let result = processor
                .process(
                    width,
                    height,
                    &pixels,
                    Some(&albedo),
                    Some(&normal),
                    &settings,
                    128,
                )
                .unwrap();
            assert_eq!(result.pixels.len(), pixels.len());
            assert!(
                result.pixels.iter().flatten().all(|v| v.is_finite()),
                "{quality:?}"
            );
            assert!(result.pixels.iter().all(|p| p[3] == 1.0));
            assert!(
                mse(&result.pixels) < raw_error,
                "{quality:?}: {} >= {raw_error}",
                mse(&result.pixels)
            );
            let mean_red = result.pixels.iter().map(|p| p[0]).sum::<f32>() / pixels.len() as f32;
            assert!(
                mean_red > 1.5 && mean_red < 3.5,
                "{quality:?}: HDR red={mean_red}"
            );
            assert!(result.latency_ms >= 0.0);
        }
        let settings = Settings {
            mode: Mode::Color,
            ..Settings::default()
        };
        let resized = processor
            .process(
                17,
                19,
                &vec![[2.0, 1.5, 1.0, 1.0]; 17 * 19],
                None,
                None,
                &settings,
                128,
            )
            .unwrap();
        assert_eq!(resized.pixels.len(), 17 * 19);
        assert!(resized.pixels.iter().flatten().all(|v| v.is_finite()));
    }
}

use pt_denoise_oidn::{OidnDenoiser, OidnMode};
use render_core::gpu::GpuContext;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, mpsc};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Mode {
    Color,
    ColorAlbedo,
    #[default]
    ColorAlbedoNormal,
}
impl Mode {
    pub const ALL: [Self; 3] = [Self::Color, Self::ColorAlbedo, Self::ColorAlbedoNormal];
    pub fn label(self) -> &'static str {
        match self {
            Self::Color => "Color",
            Self::ColorAlbedo => "Color + Albedo",
            Self::ColorAlbedoNormal => "Color + Albedo + Normal",
        }
    }
    fn native(self) -> OidnMode {
        match self {
            Self::Color => OidnMode::Color,
            Self::ColorAlbedo => OidnMode::ColorAlbedo,
            Self::ColorAlbedoNormal => OidnMode::ColorAlbedoNormal,
        }
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Quality {
    Fast,
    #[default]
    Balanced,
    High,
}
impl Quality {
    pub const ALL: [Self; 3] = [Self::Fast, Self::Balanced, Self::High];
    pub fn label(self) -> &'static str {
        match self {
            Self::Fast => "Fast",
            Self::Balanced => "Balanced",
            Self::High => "High",
        }
    }
    fn native(self) -> pt_denoise_oidn::Quality {
        match self {
            Self::Fast => pt_denoise_oidn::Quality::Fast,
            Self::Balanced => pt_denoise_oidn::Quality::Balanced,
            Self::High => pt_denoise_oidn::Quality::High,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub enabled: bool,
    pub interval: u32,
    pub mode: Mode,
    pub quality: Quality,
}
impl Default for Settings {
    fn default() -> Self {
        Self {
            enabled: true,
            interval: 128,
            mode: Mode::default(),
            quality: Quality::default(),
        }
    }
}
#[derive(Debug)]
pub struct Output {
    pub pixels: Vec<[f32; 4]>,
    pub latency_ms: f32,
}

/// Independent cadence and output ownership for each viewport or export target.
#[derive(Default)]
pub struct State {
    pub output: Option<Vec<[f32; 4]>>,
    pub last_samples: u32,
    pub last_ms: f32,
    pub error: Option<String>,
    observed: u32,
    attempted: Option<u32>,
    config: Option<(bool, Mode, Quality)>,
}
impl State {
    pub fn reset(&mut self) {
        *self = Self::default();
    }
    /// Consume an attempt before executing OIDN, including failed attempts.
    pub fn due(&mut self, settings: &Settings, samples: u32, final_pass: bool) -> bool {
        if samples < self.observed {
            self.reset();
        }
        self.observed = samples;
        let config = (settings.enabled, settings.mode, settings.quality);
        let changed = self.config.is_some_and(|previous| previous != config);
        self.config = Some(config);
        if changed || !settings.enabled {
            self.output = None;
            self.error = None;
        }
        if !settings.enabled || samples == 0 {
            return false;
        }
        let since_attempt = samples.saturating_sub(self.attempted.unwrap_or(0));
        let periodic = settings.interval > 0 && since_attempt >= settings.interval;
        let final_due = final_pass && self.attempted != Some(samples);
        if changed || periodic || final_due {
            self.attempted = Some(samples);
            return true;
        }
        false
    }
    pub fn complete(&mut self, samples: u32, result: Result<Output, String>) {
        self.last_samples = samples;
        match result {
            Ok(output) => {
                self.output = Some(output.pixels);
                self.last_ms = output.latency_ms;
                self.error = None;
            }
            Err(error) => {
                self.output = None;
                self.error = Some(error);
            }
        }
    }
}

/// A single persistent processor belongs to the GPU worker, never the GUI.
pub struct Processor {
    inner: Option<Inner>,
}
struct Inner {
    ctx: GpuContext,
    denoiser: OidnDenoiser,
    width: u32,
    height: u32,
    color: wgpu::Texture,
    albedo: wgpu::Buffer,
    normal: wgpu::Buffer,
    readback: wgpu::Buffer,
    padded_row: u32,
}
impl Processor {
    pub fn new() -> Result<Self, String> {
        Ok(Self { inner: None })
    }
    pub fn process(
        &mut self,
        width: usize,
        height: usize,
        color: &[[f32; 4]],
        albedo: Option<&[[f32; 4]]>,
        normal: Option<&[[f32; 4]]>,
        settings: &Settings,
        spp: u32,
    ) -> Result<Output, String> {
        let count = width
            .checked_mul(height)
            .ok_or("OIDN dimensions overflow")?;
        if count == 0 || width > u32::MAX as usize || height > u32::MAX as usize {
            return Err("OIDN dimensions must be nonzero and fit u32".into());
        }
        if color.len() != count
            || albedo.is_some_and(|v| v.len() != count)
            || normal.is_some_and(|v| v.len() != count)
        {
            return Err("OIDN input dimensions do not match pixel buffers".into());
        }
        if !settings.enabled {
            return Ok(Output {
                pixels: color.to_vec(),
                latency_ms: 0.0,
            });
        }
        if self.inner.is_none() {
            // shared_device uses OnceLock::get_or_init: this also negotiates the
            // headless GPU on first use, then adopts that same device everywhere.
            let shared = gpu_info::shared_device().ok_or("OIDN shared GPU unavailable")?;
            let ctx = GpuContext {
                instance: Arc::new(shared.instance.clone()),
                adapter: Arc::new(shared.adapter.clone()),
                device: Arc::new(shared.device.clone()),
                queue: Arc::new(shared.queue.clone()),
                gpu_info: None,
            };
            self.inner = Some(Inner::new(ctx, width as u32, height as u32)?);
        }
        let inner = self.inner.as_mut().unwrap();
        inner.resize(width as u32, height as u32)?;
        inner.denoiser.set_mode(settings.mode.native());
        inner.denoiser.set_quality(settings.quality.native());
        // No firefly clamp: preserve the scene's HDR range and color.
        inner.denoiser.set_input_clamp(0.0);
        inner.denoiser.set_external_input_scale(None);
        inner.denoiser.set_nan_protect(true);
        inner.ctx.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &inner.color,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(color),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(inner.width * 16),
                rows_per_image: Some(inner.height),
            },
            wgpu::Extent3d {
                width: inner.width,
                height: inner.height,
                depth_or_array_layers: 1,
            },
        );
        if let Some(aov) = albedo {
            inner
                .ctx
                .queue
                .write_buffer(&inner.albedo, 0, bytemuck::cast_slice(aov));
        }
        if let Some(aov) = normal {
            inner
                .ctx
                .queue
                .write_buffer(&inner.normal, 0, bytemuck::cast_slice(aov));
        }
        let encoder = inner.ctx.device.create_command_encoder(&Default::default());
        inner
            .denoiser
            .denoise(
                &inner.ctx,
                encoder,
                &inner.color,
                albedo.map(|_| &inner.albedo),
                normal.map(|_| &inner.normal),
                spp,
            )
            .map_err(|e| format!("OIDN inference: {e}"))?;
        let pixels = inner.read_pixels()?;
        Ok(Output {
            pixels,
            latency_ms: inner.denoiser.last_latency_ms().unwrap_or(0.0),
        })
    }
}
impl Inner {
    fn new(ctx: GpuContext, width: u32, height: u32) -> Result<Self, String> {
        let (color, albedo, normal, readback, padded_row) = Self::resources(&ctx, width, height)?;
        let denoiser = OidnDenoiser::new(&ctx, width, height, None);
        Ok(Self {
            ctx,
            denoiser,
            width,
            height,
            color,
            albedo,
            normal,
            readback,
            padded_row,
        })
    }
    fn resources(
        ctx: &GpuContext,
        width: u32,
        height: u32,
    ) -> Result<(wgpu::Texture, wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, u32), String> {
        let limits = ctx.device.limits();
        if width > limits.max_texture_dimension_2d || height > limits.max_texture_dimension_2d {
            return Err("OIDN resolution exceeds shared device limits".into());
        }
        let row = width.checked_mul(16).ok_or("OIDN row size overflow")?;
        let padded_row = row.checked_add(255).ok_or("OIDN row padding overflow")? & !255;
        let aov_bytes = u64::from(row) * u64::from(height);
        let readback_bytes = u64::from(padded_row) * u64::from(height);
        if aov_bytes > limits.max_buffer_size || readback_bytes > limits.max_buffer_size {
            return Err("OIDN image exceeds shared device buffer limits".into());
        }
        let color = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("frac OIDN linear input"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let aov = |label| {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: aov_bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frac OIDN linear readback"),
            size: readback_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Ok((
            color,
            aov("frac OIDN albedo sums"),
            aov("frac OIDN normal sums"),
            readback,
            padded_row,
        ))
    }
    fn resize(&mut self, width: u32, height: u32) -> Result<(), String> {
        if (width, height) == (self.width, self.height) {
            return Ok(());
        }
        let (color, albedo, normal, readback, padded_row) =
            Self::resources(&self.ctx, width, height)?;
        self.denoiser.resize(&self.ctx, width, height);
        self.width = width;
        self.height = height;
        self.color = color;
        self.albedo = albedo;
        self.normal = normal;
        self.readback = readback;
        self.padded_row = padded_row;
        Ok(())
    }
    fn read_pixels(&self) -> Result<Vec<[f32; 4]>, String> {
        let mut encoder = self.ctx.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: self.denoiser.result_texture(),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.padded_row),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        self.ctx.queue.submit([encoder.finish()]);
        let slice = self.readback.slice(..);
        let (tx, rx) = mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let result = (|| {
            self.ctx
                .device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|e| format!("OIDN readback poll: {e}"))?;
            rx.recv()
                .map_err(|e| format!("OIDN readback callback: {e}"))?
                .map_err(|e| format!("OIDN readback map: {e}"))?;
            let mapped = slice
                .get_mapped_range()
                .map_err(|e| format!("OIDN mapped readback view: {e}"))?;
            let mut pixels = Vec::with_capacity(self.width as usize * self.height as usize);
            for row in mapped.chunks_exact(self.padded_row as usize) {
                for pixel in row[..self.width as usize * 16].chunks_exact(16) {
                    pixels.push(std::array::from_fn(|channel| {
                        f32::from_le_bytes(pixel[channel * 4..channel * 4 + 4].try_into().unwrap())
                    }));
                }
            }
            Ok(pixels)
        })();
        // Drop the BufferView before unmapping, on success or any readback error.
        self.readback.unmap();
        result
    }
}
