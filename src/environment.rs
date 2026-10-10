//! Linear lat-long HDR/EXR environments and a solid-angle weighted importance CDF.
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]

pub struct Environment {
    pub path: String,
    pub enabled: bool,
    pub intensity: f32,
    /// Turns the environment about +Y; positive values make the image content appear to turn
    /// left in the view (see `path_sampling::env_uv`).
    pub rotation_degrees: f32,
    #[serde(skip)]
    pub revision: u64,
}
impl Default for Environment {
    fn default() -> Self {
        Self {
            path: String::new(),
            enabled: true,
            intensity: 1.0,
            rotation_degrees: 0.0,
            revision: 0,
        }
    }
}
impl Environment {
    pub fn key(&self) -> Option<(String, u64)> {
        (self.enabled && !self.path.is_empty()).then(|| (self.path.clone(), self.revision))
    }
}
pub struct Map {
    pub width: u32,
    pub height: u32,
    /// RGB radiance + cumulative pixel probability; appended after the palette LUT.
    pub texels: Vec<[f32; 4]>,
    pub mean_luminance: f32,
}
impl Map {
    pub fn load(path: &Path) -> Result<Self, String> {
        let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if !extension.eq_ignore_ascii_case("hdr") && !extension.eq_ignore_ascii_case("exr") {
            return Err("Environment must be a lat-long .hdr or .exr image".into());
        }
        if extension.eq_ignore_ascii_case("exr") {
            let crate::exr_io::RgbImage {
                width: w,
                height: h,
                pixels,
                primaries: prims,
            } = crate::exr_io::read_rgb(path)?;
            let m = crate::color::working_from(&prims)
                .map_err(|e| format!("Environment {}: chromaticities: {e}", path.display()))?;
            return Self::from_pixels(
                w,
                h,
                pixels
                    .into_iter()
                    .map(|p| crate::color::mul(&m, p))
                    .collect(),
            );
        }
        let reader = image::ImageReader::open(path)
            .map_err(|e| format!("Environment {}: {e}", path.display()))?
            .with_guessed_format()
            .map_err(|e| e.to_string())?;
        let img = reader
            .decode()
            .map_err(|e| format!("Environment {}: {e}", path.display()))?
            .into_rgb32f();
        let (w, h) = img.dimensions();
        if w == 0 || h == 0 || w > 16384 || h > 16384 || u64::from(w) * u64::from(h) > 16_777_216 {
            return Err("Environment is limited to 16 megapixels / 16384 per axis".into());
        }
        // Radiance .hdr has no primaries tag in practice: it is read as Rec.709.
        let pixels = img
            .pixels()
            .map(|p| crate::color::to_working(p.0))
            .collect();
        Self::from_pixels(w, h, pixels)
    }
    /// `pixels` are working-space (ACEScg) radiance; `load` converts files into it.
    pub fn from_pixels(width: u32, height: u32, pixels: Vec<[f32; 3]>) -> Result<Self, String> {
        if width == 0 || height == 0 || pixels.len() != width as usize * height as usize {
            return Err("Invalid environment dimensions".into());
        }
        let mut texels = Vec::with_capacity(pixels.len());
        let mut weights = Vec::with_capacity(pixels.len());
        let mut total = 0.0f64;
        for (i, pixel) in pixels.into_iter().enumerate() {
            if pixel.iter().any(|v| !v.is_finite()) {
                return Err("Environment contains NaN or infinite pixels".into());
            }
            let p = pixel.map(|v| v.max(0.0));
            let row = i / width as usize;
            let solid_angle = 2.0 * std::f64::consts::PI / f64::from(width)
                * ((std::f64::consts::PI * row as f64 / f64::from(height)).cos()
                    - (std::f64::consts::PI * (row + 1) as f64 / f64::from(height)).cos());
            let l = f64::from(crate::color::luma(p));
            let weight = l * solid_angle;
            weights.push((weight, solid_angle));
            total += weight;
            texels.push([p[0], p[1], p[2], 0.0]);
        }
        let mean_luminance = (total / (4.0 * std::f64::consts::PI)) as f32;
        // A small uniform-sphere mixture keeps every direction sampleable and avoids
        // f32 CDF plateaus starving dim pixels beside a tiny, extremely bright sun.
        let mut cumulative = 0.0f64;
        for (texel, (weight, omega)) in texels.iter_mut().zip(weights) {
            let probability = if total > 0.0 {
                0.99 * weight / total + 0.01 * omega / (4.0 * std::f64::consts::PI)
            } else {
                omega / (4.0 * std::f64::consts::PI)
            };
            cumulative += probability;
            texel[3] = cumulative.min(1.0) as f32;
        }
        texels.last_mut().unwrap()[3] = 1.0;
        Ok(Self {
            width,
            height,
            texels,
            mean_luminance,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hdr_and_exr_load_into_the_working_space_by_their_primaries() {
        let dir = std::env::temp_dir().join(format!("frac-env-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let hdr = dir.join("light.hdr");
        let exr709 = dir.join("light709.exr");
        let exr_ap1 = dir.join("light_ap1.exr");
        let rec709 = [4.0, 2.0, 1.0];
        let ap1 = crate::color::to_working(rec709);
        let pixels = vec![image::Rgb(rec709); 8];
        image::codecs::hdr::HdrEncoder::new(std::fs::File::create(&hdr).unwrap())
            .encode(&pixels, 4, 2)
            .unwrap();
        let rgba = |c: [f32; 3]| [[c[0], c[1], c[2], 1.0]; 8];
        crate::exr_io::write_rgb(
            &exr709,
            4,
            2,
            &rgba(rec709),
            &crate::color::DISPLAY_PRIMS,
            None,
            true,
        )
        .unwrap();
        crate::exr_io::write_rgb(
            &exr_ap1,
            4,
            2,
            &rgba(ap1),
            &crate::color::WORKING_PRIMS,
            None,
            true,
        )
        .unwrap();
        // The same light, tagged either way, loads as the same working-space radiance (above one).
        for path in [&hdr, &exr709, &exr_ap1] {
            let map = Map::load(path).unwrap();
            assert_eq!((map.width, map.height), (4, 2));
            for k in 0..3 {
                assert!(
                    (map.texels[0][k] - ap1[k]).abs() < 1e-5,
                    "{}: {:?} vs {ap1:?}",
                    path.display(),
                    map.texels[0]
                );
            }
        }
        std::fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn constant_map_is_uniform_on_the_sphere() {
        let m = Map::from_pixels(8, 4, vec![[2.0; 3]; 32]).unwrap();
        assert!((m.mean_luminance - 2.0).abs() < 1e-5);
        let mut previous = 0.0;
        for (i, p) in m.texels.iter().enumerate() {
            let row = i / 8;
            let omega = 2.0 * std::f32::consts::PI / 8.0
                * ((std::f32::consts::PI * row as f32 / 4.0).cos()
                    - (std::f32::consts::PI * (row + 1) as f32 / 4.0).cos());
            assert!(((p[3] - previous) / omega - 1.0 / (4.0 * std::f32::consts::PI)).abs() < 1e-5);
            previous = p[3];
        }
        assert_eq!(previous, 1.0);
    }
    #[test]
    fn bright_pixel_gets_most_samples_and_bad_pixels_are_refused() {
        let mut p = vec![[1.0; 3]; 8];
        p[3] = [1000.0; 3];
        let m = Map::from_pixels(4, 2, p).unwrap();
        assert!(m.texels[3][3] - m.texels[2][3] > 0.98);
        assert!(Map::from_pixels(1, 1, vec![[f32::NAN; 3]]).is_err());
        let black = Map::from_pixels(2, 2, vec![[0.0; 3]; 4]).unwrap();
        assert_eq!(black.texels[3][3], 1.0);
    }
}
