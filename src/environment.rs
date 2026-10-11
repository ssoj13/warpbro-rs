//! Linear lat-long HDR/EXR environments: files decode here into working-space (ACEScg)
//! radiance; the map the kernels sample (CDFs, mean, harmonics, mips and its GPU block) is
//! fractal-rs `fractal_env::EnvMap`, shared with ofx-rs.
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]

pub struct Environment {
    pub path: String,
    pub enabled: bool,
    pub intensity: f32,
    /// Turns the environment about +Y; positive values make the image content appear to turn
    /// left in the view (`fractal_env::turn`).
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
/// A decoded environment: the shared map built from working-space texels.
pub struct Map {
    pub map: fractal_env::EnvMap,
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
            )
            .map_err(|e| format!("Environment {}: {e}", path.display()));
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
        // Radiance .hdr has no primaries tag in practice: it is read as Rec.709.
        let pixels = img
            .pixels()
            .map(|p| crate::color::to_working(p.0))
            .collect();
        Self::from_pixels(w, h, pixels).map_err(|e| format!("Environment {}: {e}", path.display()))
    }
    /// `pixels` are working-space (ACEScg) radiance; `load` converts files into it. Rec.709 lies
    /// inside AP1, so a converted texel is negative only by rounding, but a map in wider primaries
    /// (AP0) has real negative AP1 components; neither carries light here, so both clamp to zero
    /// (the shared map refuses negatives). NaN and infinities stay refused.
    pub fn from_pixels(width: u32, height: u32, pixels: Vec<[f32; 3]>) -> Result<Self, String> {
        let pixels = pixels
            .into_iter()
            .map(|p| {
                if p.iter().all(|v| v.is_finite()) {
                    p.map(|v| v.max(0.0))
                } else {
                    p
                }
            })
            .collect();
        fractal_env::EnvMap::new(width, height, pixels, fractal_env::Luma::AcesCg)
            .map(|map| Self { map })
            .map_err(|e| e.to_string())
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
            assert_eq!((map.map.width(), map.map.height()), (4, 2));
            let texel = map.map.equirect().texels()[0];
            for k in 0..3 {
                assert!(
                    (texel[k] - ap1[k]).abs() < 1e-5,
                    "{}: {texel:?} vs {ap1:?}",
                    path.display()
                );
            }
        }
        std::fs::remove_dir_all(dir).unwrap();
    }
    /// Working-space texels build the shared map: its mean is the ACEScg luminance, rounding
    /// residues below zero clamp, non-finite texels are refused.
    #[test]
    fn texels_build_the_shared_map_in_the_working_space() {
        let m = Map::from_pixels(8, 4, vec![[2.0; 3]; 32]).unwrap();
        assert!((m.map.mean() - 2.0).abs() < 1e-5);
        assert_eq!(m.map.luma(), fractal_env::Luma::AcesCg);
        let mut p = vec![[1.0; 3]; 8];
        p[3] = [-1e-7, 0.5, 0.25];
        let m = Map::from_pixels(4, 2, p).unwrap();
        assert_eq!(m.map.equirect().texels()[3], [0.0, 0.5, 0.25]);
        assert!(Map::from_pixels(1, 1, vec![[f32::NAN; 3]]).is_err());
        assert!(Map::from_pixels(2, 2, vec![[0.0; 3]; 4]).is_ok());
    }
}
