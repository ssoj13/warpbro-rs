//! Host side of the GPU: one CUDA context, the loaded kernel module, and progressive render
//! targets (accumulator + float display light) that the viewport, the gallery thumbnails and final renders use.

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
    colour: crate::color::ColorPipeline,
    colour_revision: u64,
}

/// A progressive render of one scene at one size.
pub struct Target {
    pub width: usize,
    pub height: usize,
    accum: DeviceBuffer<[f32; 4]>,
    out: DeviceBuffer<[f32; 4]>,
    /// SDR monitor codes (R in the low byte), row major, after the last `step`.
    pub pixels: Vec<u32>,
    /// Linear display Rec.709 (HDR: 1 = 100 nits); never quantized.
    pub light: Vec<[f32; 4]>,
    raw: Vec<[f32; 4]>,
    pub hdr: bool,
    pub colour_error: Option<String>,
    display_key: Option<(f32, f32, bool, String)>,
    pub samples: u32,
    key: Vec<f32>,
    palette: Option<PaletteScheme>,
    /// GPU milliseconds of the last traced batch and its sample count.
    pub last_ms: f32,
    pub last_spp: u32,
}

impl Gpu {
    pub fn invalidate_colour(&mut self) { self.colour = crate::color::ColorPipeline::new(); self.colour_revision = self.colour_revision.wrapping_add(1); }
    pub fn new() -> Result<Self, String> {
        let ctx = CudaContext::new(0).map_err(|e| format!("CUDA context: {e:?}"))?;
        let stream = ctx.default_stream();
        // SAFETY: this package owns the embedded device bundle for `kernels`.
        let module = unsafe { kernels::load(&ctx) }.map_err(|e| format!("load module: {e:?}"))?;
        let name = ctx.device_name().unwrap_or_else(|_| "CUDA GPU".into());
        let lut = DeviceBuffer::zeroed(&stream, PALETTE_SAMPLES + 1).map_err(|e| format!("{e:?}"))?;
        Ok(Self { _ctx: ctx, stream, module, lut, lut_scheme: None, name, colour: crate::color::ColorPipeline::new(), colour_revision: 0 })
    }

    pub fn target(&self, width: usize, height: usize) -> Target {
        let padded = width.div_ceil(8) * height.div_ceil(4) * 32;
        Target {
            width,
            height,
            accum: DeviceBuffer::zeroed(&self.stream, padded).expect("accumulator"),
            out: DeviceBuffer::zeroed(&self.stream, width * height).expect("output"),
            pixels: vec![0; width * height],
            light: vec![[0.0; 4]; width * height],
            raw: vec![[0.0; 4]; width * height],
            hdr: false,
            colour_error: None,
            display_key: None,
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
            target.display_key = None;
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
        let display_key = (scene.render.exposure_stops, scene.render.saturation, scene.render.reinhard, format!("{}:{}", self.colour_revision, serde_json::to_string(&scene.colour).expect("colour settings")));
        if spp == 0 && target.display_key.as_ref() == Some(&display_key) { return; }
        let n = (target.width * target.height) as u32;
        let cfg = LaunchConfig1D::new(n.div_ceil(BLOCK), BLOCK, 0);
        let prepared = self.module.prepare_tonemap(cfg).expect("prepare tonemap");
        self.module
            .tonemap(&self.stream, &prepared, &target.accum, &mut target.out)
            .expect("tonemap");
        target.out.copy_to_host(&self.stream, &mut target.raw).expect("readback");
        match self.colour.apply(target.width, target.height, &target.raw, &scene.colour, scene.render.reinhard) {
            Ok((light, encoded, absolute)) => {
                target.light = light;
                target.hdr = absolute;
                target.colour_error = None;
                target.display_key = Some(display_key);
                for (p, l) in target.pixels.iter_mut().zip(&encoded) {
                    let code = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                    *p = u32::from_le_bytes([code(l[0]), code(l[1]), code(l[2]), 255]);
                }
            }
            Err(e) => target.colour_error = Some(e),
        }
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

    pub fn save_png(&self, path: &std::path::Path) -> Result<(), String> {
        if let Some(e) = &self.colour_error { return Err(format!("Colour transform failed: {e}")); }
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        }
        let pixels = if self.hdr {
            // OCIO display light is absolute, 1 = 100 nits. HDR10 is BT.2020 + PQ.
            let codes = self.light.iter().flat_map(|p| {
                let nits = egui_display::rec2020_nits([p[0], p[1], p[2]], 100.0);
                [pq16(nits[0]), pq16(nits[1]), pq16(nits[2]), 65535]
            }).collect();
            egui_display::screenshot::Pixels::Rgba16(codes)
        } else {
            egui_display::screenshot::Pixels::Rgba8(self.pixels.iter().flat_map(|p| p.to_le_bytes()).collect())
        };
        let capture = egui_display::screenshot::Capture {
            output: if self.hdr { egui_display::Output::Hdr10 } else { egui_display::Output::Sdr8 },
            width: self.width as u32, height: self.height as u32,
            white_nits: 100.0, peak_nits: if self.hdr { 1000.0 } else { 100.0 }, pixels,
        };
        capture.save(path).map(|_| ()).map_err(|e| e.to_string())
    }

    /// Display light, linear Rec.709, normalized to 100 nits (matching exr-view).
    pub fn save_display_exr(&self, path: &std::path::Path) -> Result<(), String> {
        if let Some(e) = &self.colour_error { return Err(format!("Colour transform failed: {e}")); }
        if let Some(dir) = path.parent() { std::fs::create_dir_all(dir).map_err(|e| e.to_string())?; }
        use exr::prelude::*;
        let mut image = Image::from_channels((self.width, self.height), SpecificChannels::rgb(|pos: Vec2<usize>| {
            let p = self.light[pos.y() * self.width + pos.x()]; (p[0], p[1], p[2])
        }));
        image.attributes.chromaticities = Some(attribute::Chromaticities { red: Vec2(0.64, 0.33), green: Vec2(0.30, 0.60), blue: Vec2(0.15, 0.06), white: Vec2(0.3127, 0.3290) });
        image.layer_data.attributes.white_luminance = Some(100.0);
        image.write().to_file(path).map_err(|e| e.to_string())
    }
}

fn pq16(nits: f32) -> u16 { (egui_display::pq(nits.clamp(0.0, 10000.0)) * 65535.0 + 0.5) as u16 }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cuda_float_output_reuses_samples_and_exports_hdr_metadata() {
        let mut gpu = Gpu::new().unwrap();
        let mut target = gpu.target(17, 9); // partial CUDA tiles
        let mut scene = Scene::preset(FAMILY_KIFS);
        gpu.step(&mut target, &scene, 2, 0, None);
        assert_eq!(target.samples, 2);
        assert!(target.colour_error.is_none(), "{:?}", target.colour_error);
        let old = target.light.clone();
        scene.render.exposure_stops += 1.0;
        gpu.step(&mut target, &scene, 0, 0, None);
        assert_eq!(target.samples, 2);
        assert_ne!(target.light, old);
        scene.colour.display = "Rec.2100-PQ - Display".into();
        scene.colour.view = crate::color::HDR_VIEW.into();
        gpu.step(&mut target, &scene, 0, 0, None);
        assert_eq!(target.samples, 2);
        assert!(target.hdr && target.colour_error.is_none(), "{:?}", target.colour_error);
        let dir = std::env::temp_dir().join(format!("frac-hdr-test-{}", std::process::id()));
        let png = dir.join("hdr.png");
        target.save_png(&png).unwrap();
        let bytes = std::fs::read(&png).unwrap();
        for chunk in [b"cICP", b"mDCV", b"cLLI"] { assert!(bytes.windows(4).any(|b| b == chunk)); }
        let mut decoder = png::Decoder::new(std::io::Cursor::new(&bytes));
        decoder.set_transformations(png::Transformations::IDENTITY);
        let reader = decoder.read_info().unwrap();
        assert_eq!(reader.info().bit_depth, png::BitDepth::Sixteen);
        assert_eq!(reader.info().width, 17);
        let exr_path = dir.join("display.exr");
        target.save_display_exr(&exr_path).unwrap();
        let image = exr::prelude::read_all_flat_layers_from_file(&exr_path).unwrap();
        assert!(image.attributes.chromaticities.is_some());
        assert_eq!(image.layer_data[0].attributes.white_luminance, Some(100.0));
        scene.colour.view = "missing view".into();
        gpu.step(&mut target, &scene, 0, 0, None);
        assert!(target.colour_error.is_some());
        assert!(target.save_png(&png).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn old_bookmarks_default_to_linear_rec709_aces2() {
        let scene = Scene::preset(FAMILY_BULB);
        let mut json = serde_json::to_value(&scene).unwrap();
        json.as_object_mut().unwrap().remove("colour");
        let restored: Scene = serde_json::from_value(json).unwrap();
        assert_eq!(restored.colour, crate::color::default_selection());
    }
}
