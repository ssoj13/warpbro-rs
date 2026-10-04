//! WarpBro adapter for the shared, renderer-independent material catalog.

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn glass_uses_inverse_ior_on_exit_and_preserves_total_internal_reflection() {
        use standard_surface_bsdf::sample::sample_with;
        use standard_surface_bsdf::surface::Lobe;
        use standard_surface_bsdf::surface::{ShadingFrame, SurfaceInputs, lobe_weights};
        let inputs = SurfaceInputs {
            transmission: 1.0,
            specular_roughness: 0.0,
            specular_ior: 1.5,
            ..SurfaceInputs::MATERIALX_DEFAULT
        };
        for (inside, sin_theta) in [(false, 0.6f32), (true, 0.4)] {
            let frame = ShadingFrame {
                n: [0.0, 0.0, 1.0],
                tangent: [1.0, 0.0, 0.0],
                inside,
                curvature: 0.0,
            };
            let wo = [sin_theta, 0.0, (1.0 - sin_theta * sin_theta).sqrt()];
            let weights = lobe_weights(&inputs, &frame, wo);
            let sample = sample_with(&inputs, &frame, wo, [0.9999, 0.5, 0.5], &weights);
            assert!(sample.valid && sample.delta);
            assert_eq!(sample.lobe, Lobe::Transmission);
            let expected = if inside {
                sin_theta * 1.5
            } else {
                sin_theta / 1.5
            };
            assert!((sample.wi[0].abs() - expected).abs() < 1e-6);
            assert!(sample.wi[2] < 0.0);
        }
        let frame = ShadingFrame {
            n: [0.0, 0.0, 1.0],
            tangent: [1.0, 0.0, 0.0],
            inside: true,
            curvature: 0.0,
        };
        let wo = [0.8, 0.0, 0.6];
        let weights = lobe_weights(&inputs, &frame, wo);
        for u in [0.0, 0.3, 0.9, 0.9999] {
            let sample = sample_with(&inputs, &frame, wo, [u, 0.5, 0.5], &weights);
            assert!(sample.valid && sample.wi[2] > 0.0, "{sample:?}");
            assert_eq!(sample.lobe, Lobe::Specular);
        }
    }

    #[test]
    fn glass_old_scene_fields_default_to_opaque_and_white_without_rewriting_authoring() {
        let original = Material::default();
        let mut old = serde_json::to_value(&original).unwrap();
        for key in [
            "transmission",
            "transmission_color",
            "transmission_depth",
            "transmission_extra_roughness",
        ] {
            old.as_object_mut().unwrap().remove(key);
        }
        let loaded: Material = serde_json::from_value(old).unwrap();
        assert_eq!(loaded, original);
    }

    #[test]
    fn glass_presets_use_transmission_model_instead_of_a_coat_approximation() {
        for preset in PRESETS.iter().filter(|preset| preset.category == "Glass") {
            let mut material = Material::default();
            preset.apply(&mut material);
            assert_eq!(
                material.model,
                MaterialModel::StandardSurface,
                "{}",
                preset.path
            );
            assert_eq!(material.coat, 0.0, "{}", preset.path);
            assert_eq!(material.transmission, 1.0 - preset.opacity);
            assert_eq!(material.transmission_color, preset.diffuse);
            assert_eq!(material.transmission_depth, preset.transmission_depth);
        }
    }
}
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
            m.transmission = 1.0 - preset.opacity;
            m.transmission_color = preset.diffuse;
            m.transmission_depth = preset.transmission_depth;
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
