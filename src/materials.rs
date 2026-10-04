//! WarpBro adapter for the shared, renderer-independent material catalog.
use crate::scene::{ColorSource, Facing, Material, MaterialModel};
use fractal_materials::MaterialTarget;
pub use fractal_materials::{CATEGORIES, MaterialPreset, PRESETS};

impl MaterialTarget for Material {
    fn apply_material_preset(&mut self, preset: &MaterialPreset) {
        let m = self;
        let defaults = Material::default();
        let model = m.model;
        *m = Material { model, ..defaults };
        m.color_source = ColorSource::Material;
        m.base_color = preset.diffuse;
        m.metalness = preset.metallic;
        m.specular_roughness = preset.roughness;
        m.specular_ior = preset.ior;
        if preset.emissive.iter().any(|&c| c > 0.0) {
            m.emission = 1.0;
            m.emission_color = preset.emissive;
        }
        if preset.opacity < 1.0 {
            m.base = preset.opacity;
            m.coat = 1.0;
            m.coat_roughness = preset.roughness;
            m.coat_ior = preset.ior;
        }
        if let Some((color, roughness)) = preset.sheen {
            m.sheen = 1.0;
            m.sheen_color = color;
            m.sheen_roughness = roughness;
        }
        if let Some((amount, _brush_dir)) = preset.anisotropy {
            // The brush direction follows the object's up axis (the shading tangent hint).
            m.specular_anisotropy = amount.abs();
        }
        m.facing = preset
            .facing
            .map(|(color, roughness, metallic, exponent)| Facing {
                color,
                roughness,
                metallic,
                exponent,
            });
        if preset.needs_standard_surface() {
            m.model = MaterialModel::StandardSurface;
        }
        m.preset = Some(preset.path.to_string());
    }
}
