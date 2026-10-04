//! The one place WarpBro reads and writes OpenEXR, over our own `exr-core` (exr-rs: a 1:1 port of
//! the OpenEXR C++ library, verified against it). Scene-linear sequence frames, the display-light
//! "Save view" EXR and lat-long environment maps all go through here, so the colour tags, the
//! compression and the overwrite rules exist once.
//!
//! Every file is tagged with its `chromaticities`: ACES AP1 for scene-linear (working-space)
//! output, BT.709/D65 for display light. Readers that colour-manage (OCIO, ffmpeg-rs, Nuke) then
//! know the primaries instead of guessing; `read_rgb` honours the tag the same way.

use std::path::Path;

use exr_core::attr::Attribute;
use exr_core::attr::{Chromaticities, ChromaticitiesAttribute, Compression, FloatAttribute};
use exr_core::{ChannelData, Image};
use imath_rs::{Box2i, V2i};
use vfx_ocio::color_matrix::Primaries;

/// OpenEXR's own default compression (`Header` default = ZIP): lossless, and the usual choice for
/// float renders.
const COMPRESSION: Compression = Compression::Zip;

/// Largest environment map accepted, checked from the header BEFORE any pixel is allocated.
const MAX_ENV_PIXELS: u64 = 16_777_216;
const MAX_ENV_AXIS: i64 = 16_384;

/// Write the RGB of `pixels` (alpha ignored) as a float32 EXR tagged with `prims`.
///
/// `white_nits` adds OpenEXR's `whiteLuminance` (the display-light export: 1.0 == that many nits).
/// The file is published atomically by `exr-core` (sibling temp + rename); an existing file is
/// refused unless `overwrite`, through the same gate as every other ffmpeg-rs output.
pub fn write_rgb(
    path: &Path,
    width: usize,
    height: usize,
    pixels: &[[f32; 4]],
    prims: &Primaries,
    white_nits: Option<f32>,
    overwrite: bool,
) -> Result<(), String> {
    if width == 0 || height == 0 || pixels.len() != width * height {
        return Err(format!(
            "EXR {}: {} pixels for a {width}x{height} image",
            path.display(),
            pixels.len()
        ));
    }
    let max = |n: usize| i32::try_from(n - 1).map_err(|_| format!("EXR {}: size {width}x{height} too large", path.display()));
    let window = Box2i {
        min: V2i { x: 0, y: 0 },
        max: V2i { x: max(width)?, y: max(height)? },
    };
    let channel = |c: usize| ChannelData::Float(pixels.iter().map(|p| p[c]).collect());
    let mut image = Image::new(window)
        .with_channel("R", channel(0))
        .with_channel("G", channel(1))
        .with_channel("B", channel(2));
    let attrs = image.attributes_mut();
    attrs
        .insert("chromaticities", Box::new(ChromaticitiesAttribute::new(chroma(prims))))
        .map_err(|e| e.to_string())?;
    if let Some(nits) = white_nits {
        attrs
            .insert("whiteLuminance", Box::new(FloatAttribute::new(nits)))
            .map_err(|e| e.to_string())?;
    }
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    av_util_core::outfile::check_overwrite(path, overwrite).map_err(|e| e.to_string())?;
    image
        .write(path, COMPRESSION)
        .map_err(|e| format!("EXR {}: {e}", path.display()))
}

/// Read the R, G, B channels of a single-part flat EXR as linear float RGB, raster order over the
/// data window. Any channel type is widened to f32; a file without all three is an error, never a
/// guessed fill. The size is checked from the header before the pixels are read. Also returns the
/// file's primaries: its `chromaticities`, or OpenEXR's BT.709/D65 default when untagged.
pub fn read_rgb(path: &Path) -> Result<(u32, u32, Vec<[f32; 3]>, Primaries), String> {
    let err = |e: exr_core::ExrError| format!("EXR {}: {e}", path.display());
    let header = Image::read_header_only(path).map_err(err)?;
    let (w, h) = (header.width(), header.height());
    if w <= 0 || h <= 0 || w > MAX_ENV_AXIS || h > MAX_ENV_AXIS || (w * h) as u64 > MAX_ENV_PIXELS {
        return Err(format!(
            "EXR {}: {w}x{h} exceeds the 16 megapixel / 16384 per axis limit",
            path.display()
        ));
    }
    let image = Image::read(path).map_err(err)?;
    // Untagged is BT.709 by the OpenEXR spec; a "chromaticities" of another type is a broken file.
    let tag = match image.attributes().get("chromaticities") {
        None => Chromaticities::default(),
        Some(_) => attr::<Chromaticities>(&image, "chromaticities")
            .ok_or_else(|| format!("EXR {}: \"chromaticities\" has the wrong attribute type", path.display()))?,
    };
    let plane = |name: &str| -> Result<Vec<f32>, String> {
        if image.sampling(name).is_some_and(|s| s != (1, 1)) {
            return Err(format!("EXR {}: channel {name} is subsampled", path.display()));
        }
        image
            .channel(name)
            .map(ChannelData::to_f32)
            .ok_or_else(|| format!("EXR {}: no {name} channel (an RGB image is required)", path.display()))
    };
    let (r, g, b) = (plane("R")?, plane("G")?, plane("B")?);
    let pixels = r.iter().zip(&g).zip(&b).map(|((r, g), b)| [*r, *g, *b]).collect();
    // Bounded by the checks above, so the casts cannot truncate.
    Ok((w as u32, h as u32, pixels, primaries(&tag)))
}

/// The EXR attribute for `p` (narrowed to the attribute's f32).
fn chroma(p: &Primaries) -> Chromaticities {
    let xy = |v: [f64; 2]| v.map(|c| c as f32);
    Chromaticities { red: xy(p.red), green: xy(p.grn), blue: xy(p.blu), white: xy(p.wht) }
}
/// The primaries an EXR attribute declares; validity is checked by the matrix builder.
fn primaries(c: &Chromaticities) -> Primaries {
    let xy = |v: [f32; 2]| v.map(f64::from);
    Primaries { red: xy(c.red), grn: xy(c.green), blu: xy(c.blue), wht: xy(c.white) }
}

/// One typed header attribute of `path` (tests: the tags the writer must stamp).
#[cfg(test)]
pub(crate) fn read_attr<T: exr_core::attr::AttrValue + Clone + 'static>(path: &Path, name: &str) -> Option<T> {
    attr(&Image::read_header_only(path).ok()?, name)
}

/// One typed header attribute of `image`; None when absent or of another type.
fn attr<T: exr_core::attr::AttrValue + Clone + 'static>(image: &Image, name: &str) -> Option<T> {
    let attr: &dyn Attribute = image.attributes().get(name)?;
    attr.as_any()
        .downcast_ref::<exr_core::attr::TypedAttribute<T>>()
        .map(|a| a.value.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_rgb_round_trips_with_primaries_tags_and_refuses_overwrite() {
        let dir = std::env::temp_dir().join(format!("frac-exr-io-{}", std::process::id()));
        let path = dir.join("rt.exr");
        // Values above 1, negative and tiny: scene-linear data must survive bit-exactly.
        let pixels = vec![[19.43, -0.25, 1e-6, 1.0], [0.5, 2.0, 0.125, 1.0]];
        let display = &crate::color::DISPLAY_PRIMS;
        write_rgb(&path, 2, 1, &pixels, display, Some(100.0), true).unwrap();
        let (w, h, back, prims) = read_rgb(&path).unwrap();
        assert_eq!((w, h), (2, 1));
        assert_eq!(back, vec![[19.43, -0.25, 1e-6], [0.5, 2.0, 0.125]]);
        // BT.709 is OpenEXR's default attribute value; the f32 tag reads back within f32.
        assert_eq!(read_attr::<Chromaticities>(&path, "chromaticities"), Some(Chromaticities::default()));
        assert_eq!(chroma(&prims), chroma(display));
        assert_eq!(read_attr::<f32>(&path, "whiteLuminance"), Some(100.0));
        write_rgb(&path, 2, 1, &pixels, &crate::color::WORKING_PRIMS, None, true).unwrap();
        assert_eq!(read_attr::<Chromaticities>(&path, "chromaticities"), Some(chroma(&crate::color::WORKING_PRIMS)));
        assert_eq!(read_rgb(&path).unwrap().3.wht, chroma(&crate::color::WORKING_PRIMS).white.map(f64::from));
        assert!(write_rgb(&path, 2, 1, &pixels, display, None, false).is_err(), "existing file without overwrite");
        assert!(write_rgb(&path, 3, 1, &pixels, display, None, true).is_err(), "size mismatch");
        std::fs::remove_dir_all(dir).unwrap();
    }
}
