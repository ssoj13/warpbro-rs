//! Content-addressed thumbnails shared by gallery, bookmarks and material previews.
//! Only CPU display pixels are persisted; documents, UUIDs and GPU resources are not cache keys.
use crate::{render_prep::PreparedScene, scene::Scene};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{LazyLock, Mutex, PoisonError},
};
#[derive(Clone)]
pub struct ThumbnailImage {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<u32>,
}
pub trait ThumbnailCache {
    fn get(&self, key: &str, width: usize, height: usize)
    -> Result<Option<ThumbnailImage>, String>;
    fn put(&self, key: &str, image: &ThumbnailImage) -> Result<(), String>;
}
pub struct DiskCache {
    root: PathBuf,
}
impl DiskCache {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }
}
const MAGIC: &[u8; 8] = b"WBTHMB01";
const BUDGET: u64 = 256 * 1024 * 1024;
impl ThumbnailCache for DiskCache {
    fn get(
        &self,
        key: &str,
        width: usize,
        height: usize,
    ) -> Result<Option<ThumbnailImage>, String> {
        let path = self.root.join(format!("{key}.thumb"));
        let file = match std::fs::File::open(&path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e.to_string()),
        };
        use std::io::Read;
        let expected = 48usize
            .checked_add(
                width
                    .checked_mul(height)
                    .and_then(|n| n.checked_mul(4))
                    .ok_or("Thumbnail size overflow")?,
            )
            .ok_or("Thumbnail size overflow")?;
        let mut data = Vec::new();
        file.take(expected as u64 + 1)
            .read_to_end(&mut data)
            .map_err(|e| e.to_string())?;
        let valid = data.len() == expected
            && &data[..8] == MAGIC
            && u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize == width
            && u32::from_le_bytes(data[12..16].try_into().unwrap()) as usize == height
            && Sha256::digest(&data[48..])[..] == data[16..48];
        if !valid {
            return Err(format!("Invalid thumbnail cache entry: {}", path.display()));
        }
        Ok(Some(ThumbnailImage {
            width,
            height,
            pixels: data[48..]
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                .collect(),
        }))
    }
    fn put(&self, key: &str, image: &ThumbnailImage) -> Result<(), String> {
        if image.pixels.len() != image.width * image.height {
            return Err("Thumbnail extent mismatch".into());
        }
        std::fs::create_dir_all(&self.root).map_err(|e| e.to_string())?;
        let mut pixels = Vec::with_capacity(image.pixels.len() * 4);
        for p in &image.pixels {
            pixels.extend_from_slice(&p.to_le_bytes());
        }
        let mut data = Vec::with_capacity(48 + pixels.len());
        data.extend_from_slice(MAGIC);
        data.extend_from_slice(&(image.width as u32).to_le_bytes());
        data.extend_from_slice(&(image.height as u32).to_le_bytes());
        data.extend_from_slice(&Sha256::digest(&pixels));
        data.extend_from_slice(&pixels);
        let path = self.root.join(format!("{key}.thumb"));
        let tmp = self.root.join(format!("{key}.{}.tmp", std::process::id()));
        std::fs::write(&tmp, data)
            .and_then(|_| std::fs::rename(&tmp, &path))
            .map_err(|e| e.to_string())?;
        let mut entries = Vec::new();
        let mut total = 0;
        for entry in std::fs::read_dir(&self.root).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            if entry.path().extension().and_then(|s| s.to_str()) != Some("thumb") {
                continue;
            }
            let meta = entry.metadata().map_err(|e| e.to_string())?;
            total += meta.len();
            entries.push((meta.modified().ok(), meta.len(), entry.path()));
        }
        entries.sort_by_key(|e| e.0);
        for (_, size, path) in entries {
            if total <= BUDGET {
                break;
            }
            std::fs::remove_file(path).map_err(|e| e.to_string())?;
            total -= size;
        }
        Ok(())
    }
}
static RENDERER: LazyLock<[u8; 32]> = LazyLock::new(|| {
    let mut h = Sha256::new();
    for source in [
        include_bytes!("../Cargo.lock").as_slice(),
        include_bytes!("render.rs").as_slice(),
        include_bytes!("render_wgpu.rs").as_slice(),
        include_bytes!("render_wgpu.wgsl").as_slice(),
        include_bytes!("formulas.wgsl").as_slice(),
        include_bytes!("formulas.rs").as_slice(),
        include_bytes!("render_prep.rs").as_slice(),
        include_bytes!("scene.rs").as_slice(),
        include_bytes!("color.rs").as_slice(),
        include_bytes!("ocio.rs").as_slice(),
        include_bytes!("denoise.rs").as_slice(),
        include_bytes!("environment.rs").as_slice(),
        include_bytes!("thumbnail_cache.rs").as_slice(),
    ] {
        h.update(source);
    }
    h.finalize().into()
});
type ColourKeys = HashMap<(u64, String), String>;
static COLOUR: LazyLock<Mutex<ColourKeys>> = LazyLock::new(Default::default);
fn colour_key(scene: &Scene) -> Result<String, String> {
    if !scene.colour.on || scene.render.reinhard {
        return Ok(String::new());
    }
    let config = crate::ocio::shared(&crate::ocio::source(&scene.colour.config))?;
    let sel = serde_json::to_string(&scene.colour).map_err(|e| e.to_string())?;
    let key = (config.serial(), sel);
    let mut cache = COLOUR.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(id) = cache.get(&key) {
        return Ok(id.clone());
    }
    let names = config
        .resolve(&scene.colour, true)
        .map_err(|e| e.to_string())?;
    let linear = config.transform(&names, true).map_err(|e| e.to_string())?;
    let preview_names = if linear.absolute() {
        let sdr = config
            .output_transform(
                &scene.colour,
                crate::ocio::OutputKind::Sdr,
                crate::color::BT2408_SDR_WHITE_NITS,
            )
            .map_err(|e| e.to_string())?;
        config
            .resolve(
                &crate::ocio::Sel {
                    display: sdr.display,
                    view: sdr.view,
                    ..scene.colour.clone()
                },
                false,
            )
            .map_err(|e| e.to_string())?
    } else {
        names
    };
    let encoded = config
        .transform(&preview_names, false)
        .map_err(|e| e.to_string())?;
    let id = format!(
        "{}:{}",
        encoded.processor().cache_id(),
        linear.processor().cache_id()
    );
    cache.insert(key, id.clone());
    Ok(id)
}
pub fn key(
    scene: &Scene,
    width: usize,
    height: usize,
    spp: u32,
    seed: u32,
    backend: crate::render_profiles::Backend,
    environment: Option<[u8; 32]>,
) -> Result<String, String> {
    let prepared = PreparedScene::new(scene, width as u32, height as u32)?;
    let mut h = Sha256::new();
    h.update(*RENDERER);
    // Float bytes preserve the exact packed ABI, including object transforms and masks
    // that Scene's document serializer deliberately skips.
    for floats in [
        &prepared.params(scene)[..],
        &prepared.upload.objects,
        &prepared.upload.lights,
        &prepared.trace,
    ] {
        h.update((floats.len() as u64).to_le_bytes());
        for value in floats {
            h.update(value.to_bits().to_le_bytes());
        }
    }
    h.update(
        serde_json::to_vec(&(
            prepared.palettes,
            scene.render,
            scene.world_render,
            &scene.colour,
            backend,
            spp,
            seed,
        ))
        .map_err(|e| e.to_string())?,
    );
    h.update(colour_key(scene)?);
    if prepared.environment.is_some() {
        h.update(environment.ok_or("Loaded thumbnail environment fingerprint is missing")?);
    }
    Ok(format!("{:x}", h.finalize()))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cache_survives_reopen_and_rejects_corruption() {
        let root = std::env::temp_dir().join(format!("warpbro-thumb-{}", std::process::id()));
        let cache = DiskCache::new(root.clone());
        let image = ThumbnailImage {
            width: 2,
            height: 1,
            pixels: vec![0xff123456, 0xffabcdef],
        };
        assert!(cache.get("roundtrip", 2, 1).unwrap().is_none());
        cache.put("roundtrip", &image).unwrap();
        let reopened = DiskCache::new(root.clone());
        assert_eq!(
            reopened.get("roundtrip", 2, 1).unwrap().unwrap().pixels,
            image.pixels
        );
        assert!(reopened.get("roundtrip", 1, 1).is_err());
        std::fs::write(root.join("roundtrip.thumb"), b"broken").unwrap();
        assert!(reopened.get("roundtrip", 2, 1).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn key_tracks_evaluated_world_and_quality_but_not_document_identity() {
        let scene = Scene::preset(crate::scene::Formula::Mandelbulb(
            crate::scene::Bulb::PRESET,
        ));
        let doc = crate::world::WorldDocument::from_scene(&scene);
        let mut scene = doc.snapshot(0.0).unwrap();
        scene.colour.on = false;
        scene.environment.enabled = false;
        let key_of = |s: &Scene, spp, seed| {
            key(
                s,
                8,
                8,
                spp,
                seed,
                crate::render_profiles::Backend::Wgpu,
                None,
            )
            .unwrap()
        };
        let original = key_of(&scene, 4, 1);
        let mut changed = scene.clone();
        changed.name = "Other label".into();
        assert_eq!(original, key_of(&changed, 4, 1));
        assert_ne!(original, key_of(&scene, 8, 1));
        assert_ne!(original, key_of(&scene, 4, 2));
        changed.objects[0].node = None; // Authored identity does not affect rendered pixels.
        assert_eq!(original, key_of(&changed, 4, 1));
        changed.objects[0].material.base_color[0] *= 0.5;
        assert_ne!(original, key_of(&changed, 4, 1));
        changed = scene.clone();
        changed.camera.yaw_degrees += 1.0;
        assert_ne!(original, key_of(&changed, 4, 1));
        // A resident environment, rather than the potentially changed file on disk, is keyed.
        changed = scene.clone();
        changed.environment.enabled = true;
        let a = crate::environment::Map::from_pixels(2, 2, vec![[1.0; 3]; 4]).unwrap();
        let b = crate::environment::Map::from_pixels(2, 2, vec![[2.0; 3]; 4]).unwrap();
        let env_key = |map: &crate::environment::Map| {
            key(
                &changed,
                8,
                8,
                4,
                1,
                crate::render_profiles::Backend::Wgpu,
                Some(map.fingerprint()),
            )
            .unwrap()
        };
        assert_ne!(env_key(&a), env_key(&b));
        assert_eq!(env_key(&a), env_key(&a));
    }
}
