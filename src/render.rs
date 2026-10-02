//! Host side of the GPU: one CUDA context, the loaded kernel module, and progressive render
//! targets (accumulator + RGBA8) that the viewport, the gallery thumbnails and final renders use.

use std::sync::Arc;
use std::time::Instant;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};

use crate::gpu::kernels;
use crate::palette::{PaletteScheme, build_lut};
use crate::params::*;
use crate::scene::{MaterialModel, Scene};

const BLOCK: u32 = 128;

/// Slots that only the tonemap reads: changing them re-tonemaps without restarting the samples.
const TONEMAP_ONLY: [usize; 3] = [P_EXPOSURE, P_SATURATION, P_TONEMAP];
/// Slots that change every launch.
const PER_LAUNCH: [usize; 3] = [P_SAMPLE_BEGIN, P_SPP, P_SEED];

pub struct Gpu {
    _ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: kernels::LoadedModule,
    lut: DeviceBuffer<[f32; 4]>,
    lut_scheme: Option<PaletteScheme>,
    pub name: String,
}

/// A progressive render of one scene at one size.
pub struct Target {
    pub width: usize,
    pub height: usize,
    accum: DeviceBuffer<[f32; 4]>,
    out: DeviceBuffer<u32>,
    /// RGBA8 (R in the low byte), row major, after the last `step`.
    pub pixels: Vec<u32>,
    pub samples: u32,
    key: Vec<f32>,
    palette: Option<PaletteScheme>,
    /// GPU milliseconds of the last traced batch and its sample count.
    pub last_ms: f32,
    pub last_spp: u32,
}

impl Gpu {
    pub fn new() -> Result<Self, String> {
        let ctx = CudaContext::new(0).map_err(|e| format!("CUDA context: {e:?}"))?;
        let stream = ctx.default_stream();
        // SAFETY: this package owns the embedded device bundle for `kernels`.
        let module = unsafe { kernels::load(&ctx) }.map_err(|e| format!("load module: {e:?}"))?;
        let name = ctx.device_name().unwrap_or_else(|_| "CUDA GPU".into());
        let lut = DeviceBuffer::zeroed(&stream, PALETTE_SAMPLES + 1).map_err(|e| format!("{e:?}"))?;
        Ok(Self { _ctx: ctx, stream, module, lut, lut_scheme: None, name })
    }

    pub fn target(&self, width: usize, height: usize) -> Target {
        let padded = width.div_ceil(8) * height.div_ceil(4) * 32;
        Target {
            width,
            height,
            accum: DeviceBuffer::zeroed(&self.stream, padded).expect("accumulator"),
            out: DeviceBuffer::zeroed(&self.stream, width * height).expect("output"),
            pixels: vec![0; width * height],
            samples: 0,
            key: Vec::new(),
            palette: None,
            last_ms: 0.0,
            last_spp: 0,
        }
    }

    /// Add `spp` samples (0 = only re-tonemap) and refresh `target.pixels`. Restarts the
    /// accumulation when anything but the tonemap changed. `bounces_cap` limits bounces (preview).
    pub fn step(&mut self, target: &mut Target, scene: &Scene, spp: u32, seed: u32, bounces_cap: Option<u32>) {
        let mut p = scene.pack(target.width as u32, target.height as u32);
        if let Some(cap) = bounces_cap {
            p[P_MAX_BOUNCES] = p[P_MAX_BOUNCES].min(cap as f32);
        }
        let mut key = p.clone();
        for i in TONEMAP_ONLY.iter().chain(PER_LAUNCH.iter()) {
            key[*i] = 0.0;
        }
        if key != target.key || target.palette != Some(scene.palette) {
            target.accum.zero_async(&self.stream).expect("clear accumulator");
            target.samples = 0;
            target.key = key;
            target.palette = Some(scene.palette);
        }
        if self.lut_scheme != Some(scene.palette) {
            self.lut.copy_from_host(&self.stream, &build_lut(scene.palette)).expect("palette upload");
            self.lut_scheme = Some(scene.palette);
        }
        p[P_SAMPLE_BEGIN] = target.samples as f32;
        p[P_SPP] = spp as f32;
        p[P_SEED] = seed as f32;
        let block: [f32; P_COUNT] = p.as_slice().try_into().expect("parameter block size");
        self.module.set_params(&self.stream, &block).expect("params upload");

        let t0 = Instant::now();
        if spp > 0 {
            let padded = target.accum.len() as u32;
            let cfg = LaunchConfig1D::new(padded.div_ceil(BLOCK), BLOCK, 0);
            let full = scene.material.model == MaterialModel::StandardSurface;
            let (m, s, lut, acc) = (&self.module, &self.stream, &self.lut, &mut target.accum);
            macro_rules! launch {
                ($prep:ident, $run:ident) => {{
                    let prepared = m.$prep(cfg).expect(stringify!($prep));
                    m.$run(s, &prepared, lut, acc).expect(stringify!($run));
                }};
            }
            match (full, scene.formula.code()) {
                (false, FAMILY_BULB) => launch!(prepare_fast_bulb, fast_bulb),
                (false, FAMILY_BOX) => launch!(prepare_fast_box, fast_box),
                (false, FAMILY_QUAT) => launch!(prepare_fast_quat, fast_quat),
                (false, FAMILY_KIFS) => launch!(prepare_fast_kifs, fast_kifs),
                (false, FAMILY_KLEINIAN) => launch!(prepare_fast_kleinian, fast_kleinian),
                (false, FAMILY_PSEUDO_KLEINIAN) => launch!(prepare_fast_pseudo_kleinian, fast_pseudo_kleinian),
                (false, FAMILY_APOLLONIAN) => launch!(prepare_fast_apollonian, fast_apollonian),
                (false, _) => launch!(prepare_fast_hybrid, fast_hybrid),
                (true, FAMILY_BULB) => launch!(prepare_full_bulb, full_bulb),
                (true, FAMILY_BOX) => launch!(prepare_full_box, full_box),
                (true, FAMILY_QUAT) => launch!(prepare_full_quat, full_quat),
                (true, FAMILY_KIFS) => launch!(prepare_full_kifs, full_kifs),
                (true, FAMILY_KLEINIAN) => launch!(prepare_full_kleinian, full_kleinian),
                (true, FAMILY_PSEUDO_KLEINIAN) => launch!(prepare_full_pseudo_kleinian, full_pseudo_kleinian),
                (true, FAMILY_APOLLONIAN) => launch!(prepare_full_apollonian, full_apollonian),
                (true, _) => launch!(prepare_full_hybrid, full_hybrid),
            }
            target.samples += spp;
        }
        let n = (target.width * target.height) as u32;
        let cfg = LaunchConfig1D::new(n.div_ceil(BLOCK), BLOCK, 0);
        let prepared = self.module.prepare_tonemap(cfg).expect("prepare tonemap");
        self.module
            .tonemap(&self.stream, &prepared, &target.accum, &mut target.out)
            .expect("tonemap");
        target.out.copy_to_host(&self.stream, &mut target.pixels).expect("readback");
        if spp > 0 {
            target.last_ms = t0.elapsed().as_secs_f32() * 1000.0;
            target.last_spp = spp;
        }
    }
}

impl Target {
    /// Samples per second of the last traced batch, in millions.
    pub fn msamples_per_s(&self) -> f64 {
        if self.last_ms <= 0.0 {
            return 0.0;
        }
        (self.width * self.height) as f64 * self.last_spp as f64 / (self.last_ms as f64 / 1000.0) / 1.0e6
    }

    pub fn rgb8(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(self.pixels.len() * 3);
        for p in &self.pixels {
            let [r, g, b, _] = p.to_le_bytes();
            v.extend_from_slice(&[r, g, b]);
        }
        v
    }

    pub fn save_png(&self, path: &std::path::Path) -> Result<(), String> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        }
        let file = std::fs::File::create(path).map_err(|e| e.to_string())?;
        let mut enc = png::Encoder::new(std::io::BufWriter::new(file), self.width as u32, self.height as u32);
        enc.set_color(png::ColorType::Rgb);
        enc.set_depth(png::BitDepth::Eight);
        let mut w = enc.write_header().map_err(|e| e.to_string())?;
        w.write_image_data(&self.rgb8()).map_err(|e| e.to_string())
    }
}
