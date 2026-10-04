//! The same OCIO ACES 2.0 display transforms as Playa and exr-view.
//! The tracer renders in linear ACEScg ([`WORKING`]); colours are authored in Rec.709 and
//! converted once at upload ([`to_working`]). OCIO performs full ACES rendering, gamut
//! compression and display conversion; the paths without OCIO go through [`to_709`].
#[cfg(test)]
use vfx_ocio::{Config, DisplayViewTransform, GroupTransform, MatrixTransform, Processor, Transform, TransformDirection};
#[cfg(test)]
use vfx_ocio::color_matrix::conversion_matrix_from_xyz_d65;
use std::sync::LazyLock;
use vfx_ocio::color_matrix::{ACES_AP1, Adaptation, Primaries, REC709, conversion_matrix};

/// The tracer's working space (linear AP1, ACES white), as named in the OCIO studio config.
pub const WORKING: &str = "ACEScg";
/// Primaries of [`WORKING`]: the tag of scene-linear output.
pub const WORKING_PRIMS: Primaries = ACES_AP1;
/// Primaries of display light (`ocio::Transform` ends in linear Rec.709 / D65).
pub const DISPLAY_PRIMS: Primaries = REC709;

/// Working-space luminance weights. The CUDA kernel cannot reach this host module (it links
/// `vfx-ocio`), so the device-safe source is the BSDF crate's MaterialX/ACEScg constant, which
/// the BSDF already uses for lobe selection; `luma_is_ap1_y_row` pins it to the AP1 Y row.
pub const LUMA: [f32; 3] = standard_surface_bsdf::consts::SS_LUMA;

/// Row-major 3x3 RGB matrix, applied as `M * rgb`.
pub type M3 = [[f32; 3]; 3];

/// Rec.709 (D65) -> working: authored colours (pickers, presets, palettes) enter the tracer here.
static TO_WORKING: LazyLock<M3> = LazyLock::new(|| working_from(&REC709).expect("builtin Rec.709 primaries"));
/// Working -> Rec.709 (D65): the non-OCIO display paths, in front of `oetf`.
static TO_709: LazyLock<M3> = LazyLock::new(|| {
    m3(conversion_matrix(&ACES_AP1, &REC709, Adaptation::Bradford).expect("builtin AP1 primaries"))
});

/// The matrix from RGB with chromaticities `src` to the working space. Bradford adaptation, as
/// ACES and the OCIO studio config use between D65 and the ACES white. Errors for chromaticities
/// that do not define an RGB space (e.g. a corrupt EXR `chromaticities` attribute).
pub fn working_from(src: &Primaries) -> Result<M3, String> {
    conversion_matrix(src, &ACES_AP1, Adaptation::Bradford).map(m3).map_err(|e| e.to_string())
}

/// Apply a [`M3`] to an RGB triple.
pub fn mul(m: &M3, c: [f32; 3]) -> [f32; 3] {
    m.map(|r| r[0] * c[0] + r[1] * c[1] + r[2] * c[2])
}
/// Rec.709 -> working.
pub fn to_working(c: [f32; 3]) -> [f32; 3] {
    mul(&TO_WORKING, c)
}
/// Working -> Rec.709, for display paths that bypass OCIO.
pub fn to_709(c: [f32; 3]) -> [f32; 3] {
    mul(&TO_709, c)
}
/// Working-space luminance.
pub fn luma(c: [f32; 3]) -> f32 {
    LUMA[0] * c[0] + LUMA[1] * c[1] + LUMA[2] * c[2]
}

/// The RGB block of a vfx-ocio row-major 4x4, narrowed to f32 as OCIO's matrix op does.
fn m3(m: vfx_ocio::color_matrix::M44) -> M3 {
    std::array::from_fn(|i| std::array::from_fn(|j| m[i * 4 + j] as f32))
}
pub const SDR_VIEW: &str = "ACES 2.0 - SDR 100 nits (Rec.709)";
pub const HDR_VIEW: &str = "ACES 2.0 - HDR 1000 nits (P3 D65)";

#[cfg(test)]
pub fn processor(hdr: bool) -> Result<Processor, String> {
    let cfg = Config::from_file("ocio://studio-config-latest").map_err(|e| e.to_string())?;
    let (display, view) = if hdr { ("Rec.2100-PQ - Display", HDR_VIEW) } else { ("sRGB - Display", SDR_VIEW) };
    let v = cfg.get_views(display).into_iter().find(|v| v.name() == view).ok_or("ACES 2.0 view missing")?;
    let cs = cfg.colorspace(v.effective_colorspace(display)).ok_or("display colour space missing")?;
    let mut transforms = vec![Transform::DisplayView(DisplayViewTransform {
        src: WORKING.into(), display: display.into(), view: view.into(), ..Default::default()
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
    crate::ocio::Sel { on: true, config: "ocio://studio-config-latest".into(),
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
            // Display primaries first: Reinhard is a display operator, and `oetf` encodes Rec.709.
            let light: Vec<_> = pixels.iter().map(|p| {
                let f = |v: f32| if reinhard { v.max(0.0)/(1.0+v.max(0.0)) } else { v };
                let [r, g, b] = to_709([p[0], p[1], p[2]]);
                [f(r), f(g), f(b), 1.0]
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
    fn working_matrices_round_trip_and_keep_white() {
        for c in [[1.0, 1.0, 1.0], [0.8, 0.1, 0.05], [0.02, 0.6, 0.9]] {
            let back = to_709(to_working(c));
            for k in 0..3 { assert!((back[k] - c[k]).abs() < 1e-5, "{c:?} -> {back:?}"); }
        }
        // Bradford maps D65 white onto the ACES white, so neutrals stay neutral.
        for v in to_working([1.0; 3]) { assert!((v - 1.0).abs() < 1e-5); }
        assert!((luma([1.0; 3]) - 1.0).abs() < 1e-6);
    }
    #[test]
    fn luma_is_ap1_y_row() {
        let m = vfx_ocio::color_matrix::conversion_matrix_to_xyz_d65(&ACES_AP1, Adaptation::None).unwrap();
        for k in 0..3 { assert!((LUMA[k] as f64 - m[4 + k]).abs() < 1e-6, "{LUMA:?} vs {:?}", &m[4..7]); }
    }
    #[test]
    fn working_matrix_matches_studio_config() {
        let cfg = Config::from_file("ocio://studio-config-latest").unwrap();
        let p = cfg.processor("Linear Rec.709 (sRGB)", WORKING).unwrap();
        let mut px = [[0.8f32, 0.1, 0.05], [0.02, 0.6, 0.9], [0.18, 0.18, 0.18]];
        let want = px;
        p.apply_rgb(&mut px);
        for (got, c) in px.iter().zip(want) {
            let ours = to_working(c);
            for k in 0..3 { assert!((got[k] - ours[k]).abs() < 1e-5, "{got:?} vs {ours:?}"); }
        }
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
