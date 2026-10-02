//! The same OCIO ACES 2.0 display transforms as Playa and exr-view.
//! The tracer's RGB is linear Rec.709, NOT ACEScg. OCIO performs the input
//! conversion, full ACES rendering, gamut compression and display conversion.
#[cfg(test)]
use vfx_ocio::{Config, DisplayViewTransform, GroupTransform, MatrixTransform, Processor, Transform, TransformDirection};
#[cfg(test)]
use vfx_ocio::color_matrix::{Adaptation, REC709, conversion_matrix_from_xyz_d65};

pub const INPUT: &str = "Linear Rec.709 (sRGB)";
pub const SDR_VIEW: &str = "ACES 2.0 - SDR 100 nits (Rec.709)";
pub const HDR_VIEW: &str = "ACES 2.0 - HDR 1000 nits (P3 D65)";

#[cfg(test)]
pub fn processor(hdr: bool) -> Result<Processor, String> {
    let cfg = Config::from_file("ocio://studio-config-latest").map_err(|e| e.to_string())?;
    let (display, view) = if hdr { ("Rec.2100-PQ - Display", HDR_VIEW) } else { ("sRGB - Display", SDR_VIEW) };
    let v = cfg.get_views(display).into_iter().find(|v| v.name() == view).ok_or("ACES 2.0 view missing")?;
    let cs = cfg.colorspace(v.effective_colorspace(display)).ok_or("display colour space missing")?;
    let mut transforms = vec![Transform::DisplayView(DisplayViewTransform {
        src: INPUT.into(), display: display.into(), view: view.into(), ..Default::default()
    })];
    transforms.extend(cs.to_display_reference().cloned().or_else(|| cs.from_display_reference().map(|t| t.clone().inverse())));
    transforms.push(Transform::Matrix(MatrixTransform {
        name: String::new(), matrix: conversion_matrix_from_xyz_d65(&REC709, Adaptation::None).map_err(|e| e.to_string())?,
        offset: [0.0; 4], direction: TransformDirection::Forward,
    }));
    cfg.processor_from_transform(&Transform::Group(GroupTransform {
        name: String::new(), transforms, direction: TransformDirection::Forward,
    }), TransformDirection::Forward).map_err(|e| e.to_string())
}

pub fn default_selection() -> crate::ocio::Sel {
    crate::ocio::Sel { on: true, config: "ocio://studio-config-latest".into(), input: INPUT.into(),
        display: "sRGB - Display".into(), view: SDR_VIEW.into(), ..Default::default() }
}

struct Cached {
    selection: crate::ocio::Sel,
    linear: vfx_ocio::GpuBufferExecutor,
    encoded: vfx_ocio::GpuBufferExecutor,
    absolute: bool,
}
type DisplayPixels = (Vec<[f32; 4]>, Vec<[f32; 4]>, bool);
pub struct ColorPipeline { cached: Vec<Cached> }
impl ColorPipeline {
    pub fn new() -> Self { Self { cached: Vec::new() } }
    /// The selected monitor rendering in display light, and its SDR preview codes.
    pub fn apply(&mut self, width: usize, height: usize, pixels: &[[f32; 4]], sel: &crate::ocio::Sel, reinhard: bool) -> Result<DisplayPixels, String> {
        if !sel.on || reinhard {
            let light: Vec<_> = pixels.iter().map(|p| {
                let f = |v: f32| if reinhard { v.max(0.0)/(1.0+v.max(0.0)) } else { v };
                [f(p[0]), f(p[1]), f(p[2]), 1.0]
            }).collect();
            let encoded = light.iter().map(|p| [oetf(p[0]), oetf(p[1]), oetf(p[2]), 1.0]).collect();
            return Ok((light, encoded, false));
        }
        if !self.cached.iter().any(|c| c.selection == *sel) {
            let ocio = crate::ocio::Ocio::load(&crate::ocio::source(&sel.config)).map_err(|e| e.to_string())?;
            let names = ocio.resolve(sel, true).map_err(|e| e.to_string())?;
            let light = ocio.transform(&names, true).map_err(|e| e.to_string())?;
            let absolute = light.absolute();
            let preview_names = if absolute {
                let mut sdr = sel.clone();
                sdr.display = "sRGB - Display".into(); sdr.view = SDR_VIEW.into();
                ocio.resolve(&sdr, false).map_err(|e| e.to_string())?
            } else { names };
            let encoded = ocio.transform(&preview_names, false).map_err(|e| e.to_string())?;
            if self.cached.len() == 8 { self.cached.remove(0); }
            self.cached.push(Cached { selection: sel.clone(),
                linear: vfx_ocio::GpuBufferExecutor::compile(light.processor()).map_err(|e| e.to_string())?,
                encoded: vfx_ocio::GpuBufferExecutor::compile(encoded.processor()).map_err(|e| e.to_string())?, absolute });
        }
        let c = self.cached.iter().find(|c| c.selection == *sel).unwrap();
        let light = c.linear.execute(width as u32, height as u32, pixels).map_err(|e| e.to_string())?;
        let encoded = c.encoded.execute(width as u32, height as u32, pixels).map_err(|e| e.to_string())?;
        Ok((light, encoded, c.absolute))
    }
}

/// Extended sRGB canvas encoding; preserves negative values and values above 1.
pub fn oetf(v: f32) -> f32 {
    let a = v.abs();
    v.signum() * if a <= 0.003_130_8 { 12.92 * a } else { 1.055 * a.powf(1.0 / 2.4) - 0.055 }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn aces2_gray_and_hdr_headroom() {
        let mut sdr = [[0.18, 0.18, 0.18, 1.0], [64.0, 64.0, 64.0, 1.0]];
        let mut hdr = sdr;
        processor(false).unwrap().apply_rgba(&mut sdr);
        processor(true).unwrap().apply_rgba(&mut hdr);
        assert!((oetf(sdr[0][1]) - 0.349).abs() < 0.002, "{sdr:?}");
        assert!(sdr[1][1] <= 1.001);
        assert!(hdr[1][1] > 5.0 && hdr[1][1] <= 10.001, "{hdr:?}");
    }
    #[test]
    fn extended_canvas_keeps_headroom() {
        assert!(oetf(10.0) > 1.0);
        assert!(oetf(-0.01) < 0.0);
        assert!((egui_display::pq(100.0) - 0.5080784).abs() < 1e-5);
        assert!((egui_display::pq(1000.0) - 0.7518271).abs() < 1e-5);
    }
    #[test]
    fn shared_gpu_matches_ocio_cpu() {
        let _ = env_logger::try_init();
        let mut pipeline = ColorPipeline::new();
        for hdr in [false, true] {
            let mut sel = default_selection();
            if hdr { sel.display = "Rec.2100-PQ - Display".into(); sel.view = HDR_VIEW.into(); }
            let pixels: Vec<_> = (0..64).map(|i| { let x = 2.0f32.powf(i as f32 / 4.0 - 8.0); [x, x * 0.6, x * 0.1, 1.0] }).collect();
            let mut expected = pixels.clone();
            processor(hdr).unwrap().apply_rgba(&mut expected);
            let (actual, _, absolute) = pipeline.apply(8, 8, &pixels, &sel, false).unwrap();
            assert_eq!(absolute, hdr);
            for (got, want) in actual.iter().zip(expected) {
                for c in 0..3 { assert!((got[c]-want[c]).abs() < 0.004 * want[c].abs().max(1.0), "HDR {hdr}: {got:?} vs {want:?}"); }
            }
        }
    }

}
