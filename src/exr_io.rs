//! The one place WarpBro reads and writes OpenEXR, over our own `exr-core` (exr-rs: a 1:1 port of
//! the OpenEXR C++ library, verified against it). Scene-linear sequence frames, the display-light
//! "Save view" EXR and lat-long environment maps all go through here, so the colour tags, the
//! compression and the overwrite rules exist once.
//!
//! The tracer's RGB is linear Rec.709 (`color::INPUT`), so every file is tagged with BT.709/D65
//! `chromaticities`: readers that colour-manage (OCIO, ffmpeg-rs, Nuke) then know the primaries
//! instead of guessing.

use std::path::Path;

#[cfg(test)]
use exr_core::attr::Attribute;
use exr_core::attr::{Chromaticities, ChromaticitiesAttribute, Compression, FloatAttribute};
use exr_core::{ChannelData, Image};
use imath_rs::{Box2i, V2i};

/// OpenEXR's own default compression (`Header` default = ZIP): lossless, and the usual choice for
/// float renders.
const COMPRESSION: Compression = Compression::Zip;

/// Largest environment map accepted, checked from the header BEFORE any pixel is allocated.
const MAX_ENV_PIXELS: u64 = 16_777_216;
const MAX_ENV_AXIS: i64 = 16_384;

/// Write the RGB of `pixels` (alpha ignored) as a float32 EXR tagged linear Rec.709.
///
/// `white_nits` adds OpenEXR's `whiteLuminance` (the display-light export: 1.0 == that many nits).
/// The file is published atomically by `exr-core` (sibling temp + rename); an existing file is
/// refused unless `overwrite`, through the same gate as every other ffmpeg-rs output.
pub fn write_rgb(
    path: &Path,
    width: usize,
    height: usize,
    pixels: &[[f32; 4]],
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
    // `Chromaticities::default()` is OpenEXR's Rec. ITU-R BT.709-3 / D65 default.
    attrs
        .insert("chromaticities", Box::new(ChromaticitiesAttribute::new(Chromaticities::default())))
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
/// guessed fill. The size is checked from the header before the pixels are read.
pub fn read_rgb(path: &Path) -> Result<(u32, u32, Vec<[f32; 3]>), String> {
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
    Ok((w as u32, h as u32, pixels))
}

/// One typed header attribute of `path` (tests: the tags the writer must stamp).
#[cfg(test)]
pub(crate) fn read_attr<T: exr_core::attr::AttrValue + Clone + 'static>(path: &Path, name: &str) -> Option<T> {
    let image = Image::read_header_only(path).ok()?;
    let attr: &dyn Attribute = image.attributes().get(name)?;
    attr.as_any()
        .downcast_ref::<exr_core::attr::TypedAttribute<T>>()
        .map(|a| a.value.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_rgb_round_trips_with_rec709_tags_and_refuses_overwrite() {
        let dir = std::env::temp_dir().join(format!("frac-exr-io-{}", std::process::id()));
        let path = dir.join("rt.exr");
        // Values above 1, negative and tiny: scene-linear data must survive bit-exactly.
        let pixels = vec![[19.43, -0.25, 1e-6, 1.0], [0.5, 2.0, 0.125, 1.0]];
        write_rgb(&path, 2, 1, &pixels, Some(100.0), true).unwrap();
        let (w, h, back) = read_rgb(&path).unwrap();
        assert_eq!((w, h), (2, 1));
        assert_eq!(back, vec![[19.43, -0.25, 1e-6], [0.5, 2.0, 0.125]]);
        assert_eq!(read_attr::<Chromaticities>(&path, "chromaticities"), Some(Chromaticities::default()));
        assert_eq!(read_attr::<f32>(&path, "whiteLuminance"), Some(100.0));
        assert!(write_rgb(&path, 2, 1, &pixels, None, false).is_err(), "existing file without overwrite");
        assert!(write_rgb(&path, 3, 1, &pixels, None, true).is_err(), "size mismatch");
        std::fs::remove_dir_all(dir).unwrap();
    }
}
