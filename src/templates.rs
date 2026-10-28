//! Scene templates: built into the binary, overridable by files in `~/.warpbro/templates`.
//! Registered into `soluwell-studio` via [`register_warpbro_templates`].

use soluwell_studio::scene::Scene;
pub use soluwell_studio::templates::{BuiltinTemplate, builtin_scene};
use soluwell_studio::world::WorldDocument;

pub fn decode_scene(text: &str) -> Result<Scene, String> {
    let document = WorldDocument::decode_json(text).map_err(|e| format!("Invalid scene: {e}"))?;
    let mut scene = document.snapshot(f64::from(document.first))?;
    scene.animation.first = document.first;
    scene.animation.last = document.last;
    scene.animation.fps = document.fps;
    scene.document = Some(Box::new(document));
    Ok(scene)
}

fn decode_named_json(text: &'static str, name: &'static str) -> Result<Scene, String> {
    let mut scene = decode_scene(text)?;
    scene.name = name.to_owned();
    Ok(scene)
}

fn build_ember_bloom(name: &'static str) -> Result<Scene, String> {
    decode_named_json(include_str!("../templates/ember-bloom.frac.json"), name)
}
fn build_chrome_cathedral(name: &'static str) -> Result<Scene, String> {
    crate::presets::scene(crate::presets::Animated::ChromeCathedral, name)
}
fn build_opal_julia(name: &'static str) -> Result<Scene, String> {
    crate::presets::scene(crate::presets::Animated::OpalJulia, name)
}
fn build_verdant_fold(name: &'static str) -> Result<Scene, String> {
    crate::presets::scene(crate::presets::Animated::VerdantFold, name)
}
fn build_luminous_web(name: &'static str) -> Result<Scene, String> {
    decode_named_json(include_str!("../templates/luminous-web.frac.json"), name)
}
fn build_amethyst(name: &'static str) -> Result<Scene, String> {
    decode_named_json(include_str!("../templates/amethyst.json"), name)
}
fn build_box(name: &'static str) -> Result<Scene, String> {
    decode_named_json(include_str!("../templates/box.json"), name)
}
fn build_mandelbulb(name: &'static str) -> Result<Scene, String> {
    decode_named_json(include_str!("../templates/mandelbulb.frac.json"), name)
}
fn build_mandelwhit(name: &'static str) -> Result<Scene, String> {
    decode_named_json(include_str!("../templates/mandelwhit.json"), name)
}
fn build_metal_frac(name: &'static str) -> Result<Scene, String> {
    decode_named_json(include_str!("../templates/metal_frac.json"), name)
}
fn build_metal_frac2(name: &'static str) -> Result<Scene, String> {
    decode_named_json(include_str!("../templates/metal_frac2.json"), name)
}
fn build_pinky(name: &'static str) -> Result<Scene, String> {
    decode_named_json(include_str!("../templates/pinky.json"), name)
}
fn build_q_julia1(name: &'static str) -> Result<Scene, String> {
    decode_named_json(include_str!("../templates/q-julia1.json"), name)
}
fn build_q_julia2(name: &'static str) -> Result<Scene, String> {
    decode_named_json(include_str!("../templates/q-julia2.json"), name)
}

/// The animated showcase scenes among [`BUILTINS`] (its first entries): what the gallery
/// fixtures (`--animated-fixtures`) and the world benchmark render.
pub const ANIMATED: std::ops::Range<usize> = 0..5;

pub const BUILTINS: [BuiltinTemplate; 14] = [
    BuiltinTemplate {
        name: "Ember Bloom",
        description: "A copper Mandelbulb opens and twists under warm light.",
        builder: build_ember_bloom,
    },
    BuiltinTemplate {
        name: "Chrome Cathedral",
        description: "Cool metallic Mandelbox folds breathe through a slow camera arc.",
        builder: build_chrome_cathedral,
    },
    BuiltinTemplate {
        name: "Opal Julia",
        description: "A pearlescent quaternion Julia shifts through violet and blue.",
        builder: build_opal_julia,
    },
    BuiltinTemplate {
        name: "Verdant Fold",
        description: "A jade octahedral KIFS unfolds against a deep green sky.",
        builder: build_verdant_fold,
    },
    BuiltinTemplate {
        name: "Luminous Web",
        description: "Golden Apollonian filaments change density under a dusk gradient.",
        builder: build_luminous_web,
    },
    BuiltinTemplate {
        name: "amethyst",
        description: "Authored Mandelbulb scene template.",
        builder: build_amethyst,
    },
    BuiltinTemplate {
        name: "box",
        description: "Authored Mandelbox scene template.",
        builder: build_box,
    },
    BuiltinTemplate {
        name: "mandelbulb",
        description: "Multi-object Mandelbulb scene template.",
        builder: build_mandelbulb,
    },
    BuiltinTemplate {
        name: "mandelwhit",
        description: "Authored Mandelbulb scene template.",
        builder: build_mandelwhit,
    },
    BuiltinTemplate {
        name: "metal_frac",
        description: "Metallic Quaternion Julia scene template.",
        builder: build_metal_frac,
    },
    BuiltinTemplate {
        name: "metal_frac2",
        description: "Metallic Quaternion Julia variation template.",
        builder: build_metal_frac2,
    },
    BuiltinTemplate {
        name: "pinky",
        description: "Authored Mandelbulb scene template.",
        builder: build_pinky,
    },
    BuiltinTemplate {
        name: "q-julia1",
        description: "Quaternion Julia animated scene template.",
        builder: build_q_julia1,
    },
    BuiltinTemplate {
        name: "q-julia2",
        description: "Quaternion Julia animated variation template.",
        builder: build_q_julia2,
    },
];

/// Register WarpBro's 14 built-in templates and animated showcase range with `soluwell-studio`.
pub fn register_warpbro_templates() {
    soluwell_studio::templates::register_builtins(&BUILTINS, ANIMATED);
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use soluwell_studio::templates::{Source, catalog};

    use super::*;

    #[test]
    #[ignore = "reads operator templates from WARPBRO_TEMPLATE_PROBE_DIR"]
    fn operator_templates_decode_with_current_scene_schema() {
        crate::engine::init_warpbro();
        let path =
            std::env::var_os("WARPBRO_TEMPLATE_PROBE_DIR").expect("set WARPBRO_TEMPLATE_PROBE_DIR");
        let entries = catalog(Path::new(&path)).unwrap();
        let mut files = 0;
        for entry in entries {
            if let Source::File(path) = entry.source {
                let scene = decode_scene(&std::fs::read_to_string(&path).unwrap())
                    .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
                assert!(scene.render.iterations.is_finite());
                eprintln!("loaded {}", path.display());
                files += 1;
            }
        }
        assert!(files > 0, "the probe must exercise real templates");
    }

    #[test]
    fn builtins_need_no_directory_and_files_override_by_name() {
        crate::engine::init_warpbro();
        let dir =
            std::env::temp_dir().join(format!("warpbro-templates-test-{}", std::process::id()));
        let builtin = catalog(&dir).unwrap();
        assert_eq!(builtin.len(), BUILTINS.len());
        assert!(
            builtin
                .iter()
                .all(|e| matches!(e.source, Source::Builtin(_)) && e.description.is_some())
        );
        for (index, entry) in BUILTINS.iter().enumerate() {
            let scene = builtin_scene(index).unwrap();
            assert_eq!(scene.name, entry.name);
            assert!(scene.document.is_some());
        }
        assert!(!dir.exists(), "the catalog never creates the directory");
        std::fs::create_dir_all(&dir).unwrap();
        let first = BUILTINS[0].name;
        let text = serde_json::to_string(&builtin_scene(1).unwrap().document).unwrap();
        let replaced = dir.join(soluwell_studio::fs_name::frame_file(
            &soluwell_studio::fs_name::stem(first),
            None,
            soluwell_studio::fs_name::SCENE_SUFFIX,
        ));
        let user = dir.join("user-scene.json");
        std::fs::write(&replaced, &text).unwrap();
        std::fs::write(&user, &text).unwrap();
        std::fs::write(dir.join("notes.txt"), "ignored").unwrap();
        let merged = catalog(&dir).unwrap();
        assert_eq!(merged.len(), builtin.len() + 1);
        assert_eq!(merged[0].name, first);
        assert_eq!(merged[0].source, Source::File(replaced));
        assert!(merged[0].overrides && merged[0].description.is_some());
        let added = merged.last().unwrap();
        assert_eq!(
            (added.name.as_str(), &added.source),
            ("user-scene", &Source::File(user.clone()))
        );
        let scene = decode_scene(&std::fs::read_to_string(&user).unwrap()).unwrap();
        assert_eq!(scene.document.unwrap().last, 249);
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// One table: every built-in has its own name (a file overrides by it), and each animated
    /// showcase scene is an animated document of that name.
    #[test]
    fn builtin_names_are_unique_and_the_animated_scenes_animate() {
        crate::engine::init_warpbro();
        let mut names: Vec<_> = BUILTINS
            .iter()
            .map(|b| soluwell_studio::fs_name::slug(b.name))
            .collect();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), BUILTINS.len());
        for index in ANIMATED {
            let scene = builtin_scene(index).unwrap();
            assert_eq!(scene.name, BUILTINS[index].name);
            let document = scene.document.unwrap();
            assert!(document.last > document.first, "{}", scene.name);
        }
    }
}
