//! Portable animated worlds authored through the same commands as the editor.
use crate::params::{FAMILY_APOLLONIAN, FAMILY_BOX, FAMILY_BULB, FAMILY_KIFS, FAMILY_QUAT};
use crate::scene::{Coloring, Facing, Formula, Kifs, KifsKind, MaterialModel, Scene};
use crate::world::{
    CAMERA_ORBIT_PHASE, CAMERA_ORBIT_SPEED, NodeId, WorldCommand, WorldDocument, WorldEditor,
};
use curves::CurveKind;
use serde_json::{Value, json};

pub struct AnimatedPreset {
    pub name: &'static str,
    pub description: &'static str,
}
pub const ANIMATED: [AnimatedPreset; 5] = [
    AnimatedPreset {
        name: "Ember Bloom",
        description: "A copper Mandelbulb opens and twists under warm light.",
    },
    AnimatedPreset {
        name: "Chrome Cathedral",
        description: "Cool metallic Mandelbox folds breathe through a slow camera arc.",
    },
    AnimatedPreset {
        name: "Opal Julia",
        description: "A pearlescent quaternion Julia shifts through violet and blue.",
    },
    AnimatedPreset {
        name: "Verdant Fold",
        description: "A jade octahedral KIFS unfolds against a deep green sky.",
    },
    AnimatedPreset {
        name: "Luminous Web",
        description: "Golden Apollonian filaments change density under a dusk gradient.",
    },
];
const TIMES: [f64; 3] = [0.0, 124.0, 249.0];

fn track(commands: &mut Vec<WorldCommand>, id: NodeId, path: &str, values: [Value; 3]) {
    for (frame, value) in TIMES.into_iter().zip(values) {
        commands.push(WorldCommand::SetAttribute {
            id,
            path: path.into(),
            value,
            frame,
        });
        commands.push(WorldCommand::Key {
            id,
            path: path.into(),
            frame,
        });
    }
    commands.push(WorldCommand::Interpolation {
        id,
        path: path.into(),
        frames: TIMES.to_vec(),
        kind: CurveKind::Smooth,
    });
}

pub fn scene(index: usize) -> Result<Scene, String> {
    let preset = ANIMATED.get(index).ok_or("Unknown animation preset")?;
    let family = [
        FAMILY_BULB,
        FAMILY_BOX,
        FAMILY_QUAT,
        FAMILY_KIFS,
        FAMILY_APOLLONIAN,
    ][index];
    let mut base = Scene::preset(family);
    base.name = preset.name.into();
    base.animation.first = 0;
    base.animation.last = 249;
    base.animation.fps = 24.0;
    base.camera.distance = [3.8, 3.1, 2.6, 3.8, 3.1][index];
    base.camera.pitch_degrees = 14.0;
    base.camera.aperture = 0.0;
    base.render.max_bounces = 4;
    base.render.exposure_stops = 0.0;
    base.material.model = MaterialModel::StandardSurface;
    base.material.specular_roughness = [0.27, 0.30, 0.23, 0.3, 0.25][index];
    base.material.metalness = [0.75, 0.9, 0.12, 0.28, 0.65][index];
    base.material.coat = [0.25, 0.3, 0.8, 0.55, 0.2][index];
    base.material.coat_roughness = 0.14;
    base.coloring = Coloring::TrapOrigin;
    base.trap_scale = 1.25;
    let (palette, horizon, zenith, sun) = match index {
        0 => (
            crate::palette::PaletteScheme::Ember,
            [0.045, 0.014, 0.007],
            [0.035, 0.055, 0.11],
            [1.0, 0.68, 0.42],
        ),
        1 => (
            crate::palette::PaletteScheme::Ice,
            [0.12, 0.16, 0.22],
            [0.008, 0.014, 0.025],
            [0.78, 0.88, 1.0],
        ),
        2 => (
            crate::palette::PaletteScheme::Amethyst,
            [0.055, 0.025, 0.09],
            [0.045, 0.09, 0.16],
            [1.0, 0.8, 0.95],
        ),
        3 => (
            crate::palette::PaletteScheme::Verdant,
            [0.014, 0.04, 0.027],
            [0.055, 0.105, 0.09],
            [0.86, 1.0, 0.72],
        ),
        _ => (
            crate::palette::PaletteScheme::Copper,
            [0.05, 0.022, 0.065],
            [0.035, 0.07, 0.14],
            [1.0, 0.82, 0.52],
        ),
    };
    base.palette = palette;
    base.lighting.sky_horizon = horizon;
    base.lighting.sky_zenith = zenith;
    base.lighting.sun_color = sun;
    base.lighting.sky_intensity = 4.5;
    base.lighting.sun_intensity = if index == 1 { 4.0 } else { 2.5 };
    base.lighting.sun_angle = 3.0;
    base.lighting.sun_azimuth = -35.0;
    base.lighting.sun_elevation = 48.0;
    base.lighting.background = true;
    if index == 1
        && let Formula::Mandelbox(parameters) = &mut base.formula
    {
        parameters.scale = -1.5;
    }
    if index == 4 {
        base.material.base_tint = [1.0, 0.88, 0.64];
    }
    if index == 2 {
        base.material.facing = Some(Facing {
            color: [0.18, 0.65, 0.9],
            roughness: 0.28,
            metallic: 0.35,
            exponent: 2.3,
        });
        base.material.thin_film_thickness = 420.0;
    }
    if index == 3 {
        base.formula = Formula::Kifs(Kifs::preset(KifsKind::Octahedron));
    }
    let mut editor = WorldEditor::new(WorldDocument::from_scene(&base));
    let object = editor.selection.ok_or("Preset has no fractal")?;
    let camera = editor
        .document
        .active_camera
        .ok_or("Preset has no camera")?;
    let mut commands = vec![WorldCommand::SetTimeRange {
        first: 0,
        last: 249,
        fps: 24.0,
    }];
    for node in editor.document.nodes() {
        commands.push(WorldCommand::SetSpan {
            id: node.id,
            start: 0.0,
            end: 250.0,
        });
    }
    for (path, value) in [
        (CAMERA_ORBIT_SPEED, 6.0 + index as f64),
        (CAMERA_ORBIT_PHASE, 0.0),
    ] {
        commands.push(WorldCommand::SetAttribute {
            id: camera,
            path: path.into(),
            value: json!(value),
            frame: 0.0,
        });
    }
    track(
        &mut commands,
        object,
        "/transform/position",
        [
            json!([0.0, -0.025, 0.0]),
            json!([0.035, 0.035, -0.02]),
            json!([-0.02, 0.0, 0.025]),
        ],
    );
    track(
        &mut commands,
        object,
        "/transform/rotation_degrees",
        [
            json!([0.0, -8.0, 0.0]),
            json!([4.0, 5.0, -3.0]),
            json!([-2.0, 18.0, 2.0]),
        ],
    );
    track(
        &mut commands,
        object,
        "/transform/scale",
        [
            json!([0.98, 0.98, 0.98]),
            json!([1.03, 1.03, 1.03]),
            json!([1.0, 1.0, 1.0]),
        ],
    );
    let distance = f64::from(base.camera.distance);
    track(
        &mut commands,
        camera,
        "/camera/distance",
        [
            json!(distance),
            json!(distance * 0.95),
            json!(distance * 1.02),
        ],
    );
    track(
        &mut commands,
        camera,
        "/camera/pitch_degrees",
        [json!(14.0), json!(22.0), json!(17.0)],
    );
    match index {
        0 => {
            track(
                &mut commands,
                object,
                "/formula/Mandelbulb/power",
                [json!(7.4), json!(8.5), json!(7.8)],
            );
            track(
                &mut commands,
                object,
                "/formula/Mandelbulb/angle_phase_degrees/0",
                [json!(-5.0), json!(8.0), json!(2.0)],
            );
        }
        1 => {
            track(
                &mut commands,
                object,
                "/formula/Mandelbox/scale",
                [json!(-1.55), json!(-1.45), json!(-1.5)],
            );
            track(
                &mut commands,
                object,
                "/formula/Mandelbox/min_radius_ratio",
                [json!(0.46), json!(0.53), json!(0.49)],
            );
        }
        2 => {
            track(
                &mut commands,
                object,
                "/formula/QuaternionJulia/constant/0",
                [json!(-0.2), json!(-0.26), json!(-0.22)],
            );
            track(
                &mut commands,
                object,
                "/formula/QuaternionJulia/slice_w",
                [json!(-0.07), json!(0.09), json!(0.02)],
            );
        }
        3 => {
            track(
                &mut commands,
                object,
                "/formula/Kifs/scale",
                [json!(1.96), json!(2.05), json!(2.01)],
            );
            track(
                &mut commands,
                object,
                "/formula/Kifs/rotation_degrees/1",
                [json!(-3.0), json!(4.0), json!(1.5)],
            );
        }
        _ => {
            track(
                &mut commands,
                object,
                "/formula/Apollonian/scale",
                [json!(1.27), json!(1.34), json!(1.3)],
            );
            track(
                &mut commands,
                object,
                "/trap_scale",
                [json!(1.0), json!(1.65), json!(1.3)],
            );
        }
    }
    editor.execute(WorldCommand::Batch(commands))?;
    let mut result = editor.document.snapshot(0.0)?;
    result.animation.first = editor.document.first;
    result.animation.last = editor.document.last;
    result.animation.fps = editor.document.fps;
    result.document = Some(Box::new(editor.document));
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::WorldKind;

    #[test]
    fn animated_presets_remain_finite_visible_and_authored_for_all_250_frames() {
        for index in 0..ANIMATED.len() {
            let preset = scene(index).unwrap();
            let document = preset.document.as_ref().unwrap();
            assert_eq!(
                (document.first, document.last, document.fps),
                (0, 249, 24.0)
            );
            assert!(
                document
                    .nodes()
                    .iter()
                    .all(|n| n.start == 0.0 && n.end == 250.0)
            );
            let first = document.snapshot(0.0).unwrap();
            let last = document.snapshot(249.0).unwrap();
            assert_ne!(first.camera, last.camera);
            assert_ne!(first.objects[0].object_world, last.objects[0].object_world);
            assert_ne!(first.formula, last.formula);
            let object = document
                .nodes()
                .into_iter()
                .find(|n| n.kind == WorldKind::Fractal)
                .unwrap()
                .id;
            assert!(
                document
                    .attributes(object, 0.0)
                    .unwrap()
                    .iter()
                    .filter(|a| !a.frames.is_empty())
                    .count()
                    >= 6
            );
            for frame in 0..250 {
                let snapshot = document.snapshot(f64::from(frame)).unwrap();
                assert_eq!(snapshot.objects.len(), 1, "{}", ANIMATED[index].name);
                assert!(snapshot.pack(320, 180).iter().all(|v| v.is_finite()));
                assert!(
                    snapshot.objects[0]
                        .pack(320, 180)
                        .iter()
                        .all(|v| v.is_finite())
                );
                assert!(
                    snapshot.objects[0]
                        .object_world
                        .unwrap()
                        .iter()
                        .flatten()
                        .all(|v| v.is_finite())
                );
            }
        }
        assert!(scene(ANIMATED.len()).is_err());
    }

    #[test]
    fn preset_documents_roundtrip_and_evaluate_identically_in_reverse_and_at_subframes() {
        for index in 0..ANIMATED.len() {
            let preset = scene(index).unwrap();
            let document = preset.document.unwrap();
            let restored: WorldDocument =
                serde_json::from_str(&serde_json::to_string(&document).unwrap()).unwrap();
            let frames = [0.0, 13.25, 62.5, 124.0, 186.75, 249.0];
            let expected: Vec<_> = frames
                .iter()
                .map(|&frame| document.snapshot(frame).unwrap())
                .collect();
            for (at, &frame) in frames.iter().enumerate().rev() {
                let actual = restored.snapshot(frame).unwrap();
                assert_eq!(actual, expected[at]);
                assert_eq!(actual.pack(320, 180), expected[at].pack(320, 180));
                assert_eq!(
                    actual.objects[0].pack(320, 180),
                    expected[at].objects[0].pack(320, 180)
                );
            }
        }
    }
}
