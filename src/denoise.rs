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
        let _gpu_test = crate::test_gpu::lock();
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
        let _gpu_test = crate::test_gpu::lock();
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
        let _gpu_test = crate::test_gpu::lock();
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
        let _gpu_test = crate::test_gpu::lock();
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
        let _gpu_test = crate::test_gpu::lock();
        let settings = Settings {
            interval: 0,
            ..Settings::default()
        };
        assert!(serde_json::from_str::<Settings>(r#"{"interval":0}"#).is_err());
        assert!(settings.enabled);
        assert_eq!(settings.mode, Mode::ColorAlbedoNormal);
        let mut state = State::default();
        assert!(!state.due(&settings, 4096, false));
        assert!(state.due(&settings, 4096, true));
        state.complete(4096, success(4096));
        assert!(!state.due(&settings, 4096, true));
    }
    /// The positional form the tests read best; production code names the fields.
    fn image<'a>(
        width: usize,
        height: usize,
        color: &'a [[f32; 4]],
        albedo: Option<&'a [[f32; 4]]>,
        normal: Option<&'a [[f32; 4]]>,
    ) -> Image<'a> {
        Image {
            width,
            height,
            color,
            albedo,
            normal,
        }
    }
    #[test]
    fn invalid_inputs_are_rejected() {
        let _gpu_test = crate::test_gpu::lock();
        let mut processor = Processor::new().unwrap();
        assert!(
            processor
                .process(
                    image(2, 2, &[[0.0; 4]; 3], None, None),
                    &Settings::default(),
                    1
                )
                .is_err()
        );
        assert!(
            processor
                .process(image(0, 2, &[], None, None), &Settings::default(), 1)
                .is_err()
        );
        // The default mode needs both guides; a render that lacks one is an error, never a
        // silently poorer denoise.
        assert!(
            processor
                .process(
                    image(1, 1, &[[0.0; 4]], None, None),
                    &Settings::default(),
                    1
                )
                .is_err()
        );
    }
    #[test]
    #[ignore = "requires a real shared wgpu adapter and embedded OIDN HDR weights"]
    fn gpu_quality_modes_preserve_hdr_reduce_noise_and_reuse_processor() {
        let _gpu_test = crate::test_gpu::lock();
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
                    image(width, height, &pixels, Some(&albedo), Some(&normal)),
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
                image(17, 19, &vec![[2.0, 1.5, 1.0, 1.0]; 17 * 19], None, None),
                &settings,
                128,
            )
            .unwrap();
        assert_eq!(resized.pixels.len(), 17 * 19);
        assert!(resized.pixels.iter().flatten().all(|v| v.is_finite()));
    }
}

use pt_denoise_oidn::host::{HostDenoiser, Params};
pub use pt_denoise_oidn::host::{Image, Output};
use pt_denoise_oidn::{GpuContext, OidnMode, Primaries};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Mode {
    Color,
    ColorAlbedo,
    #[default]
    ColorAlbedoNormal,
}
impl Mode {
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
    #[cfg(test)]
    pub const ALL: [Self; 3] = [Self::Fast, Self::Balanced, Self::High];
    fn native(self) -> pt_denoise_oidn::Quality {
        match self {
            Self::Fast => pt_denoise_oidn::Quality::Fast,
            Self::Balanced => pt_denoise_oidn::Quality::Balanced,
            Self::High => pt_denoise_oidn::Quality::High,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]

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

/// A single persistent processor belongs to the GPU worker, never the GUI. The staging around
/// the network (upload, readback, size-dependent resources) is render-rs's shared
/// `pt_denoise_oidn::host::HostDenoiser`; this adds WarpBro's settings and its working space.
pub struct Processor {
    inner: Option<HostDenoiser>,
}

impl Processor {
    pub fn new() -> Result<Self, String> {
        Ok(Self { inner: None })
    }
    /// Denoise `image` under `settings` (disabled: the colour unchanged). The colour is ACEScg,
    /// so OIDN's autoexposure measures luminance in ACEScg primaries.
    pub fn process(
        &mut self,
        image: Image<'_>,
        settings: &Settings,
        spp: u32,
    ) -> Result<Output, String> {
        if !settings.enabled {
            return Ok(Output {
                pixels: image.color.to_vec(),
                latency_ms: 0.0,
            });
        }
        let denoiser = match &mut self.inner {
            Some(denoiser) => denoiser,
            slot => {
                // shared_device uses OnceLock::get_or_init: this also negotiates the
                // headless GPU on first use, then adopts that same device everywhere.
                let shared = gpu_info::shared_device().ok_or("OIDN shared GPU unavailable")?;
                slot.insert(HostDenoiser::new(GpuContext {
                    instance: Arc::new(shared.instance.clone()),
                    adapter: Arc::new(shared.adapter.clone()),
                    device: Arc::new(shared.device.clone()),
                    queue: Arc::new(shared.queue.clone()),
                }))
            }
        };
        denoiser.process(
            image,
            &Params {
                mode: settings.mode.native(),
                quality: settings.quality.native(),
                primaries: Primaries::AcesCg,
                spp,
            },
        )
    }
}
