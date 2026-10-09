//! Portable animated worlds authored through the same commands as the editor.
use crate::params::{FAMILY_APOLLONIAN, FAMILY_BOX, FAMILY_BULB, FAMILY_KIFS, FAMILY_QUAT};
use crate::scene::{Coloring, Facing, Formula, Kifs, KifsKind, MaterialModel, Scene};
use crate::world::{
    CAMERA_ORBIT_PHASE, CAMERA_ORBIT_SPEED, NodeId, WorldCommand, WorldDocument, WorldEditor,
};
use curves::Tan;
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
        kind: Tan::Smooth,
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
    base.camera.f_number = 0.0;
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
            json!([0.0, 0.0, 0.0]),
            json!([0.0, 0.0, 0.0]),
            json!([0.0, 0.0, 0.0]),
        ],
    );
    track(
        &mut commands,
        object,
        "/transform/rotation_degrees",
        [
            json!([0.0, -8.0, 0.0]),
            json!([0.0, 5.0, 0.0]),
            json!([0.0, 18.0, 0.0]),
        ],
    );
    track(
        &mut commands,
        object,
        "/transform/scale",
        [
            json!([1.0, 1.0, 1.0]),
            json!([1.0, 1.0, 1.0]),
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
            json!(distance * 1.05),
            json!(distance * 1.10),
        ],
    );
    track(
        &mut commands,
        camera,
        "/camera/pitch_degrees",
        [json!(14.0), json!(17.0), json!(20.0)],
    );
    let iterations = match index {
        0 => [6, 13, 20],
        1 => [6, 10, 14],
        _ => [3, 8, 14],
    };
    track(
        &mut commands,
        object,
        "/render/iterations",
        iterations.map(|n| json!(n)),
    );
    match index {
        0 => {
            track(
                &mut commands,
                object,
                "/formula/Mandelbulb/angle_scale",
                [json!([0.5, 0.5]), json!([1.0, 1.0]), json!([1.5, 1.5])],
            );
            track(
                &mut commands,
                object,
                "/formula/Mandelbulb/power",
                [json!(3.0), json!(5.5), json!(8.0)],
            );
            track(
                &mut commands,
                object,
                "/formula/Mandelbulb/angle_phase_degrees/0",
                [json!(0.0), json!(3.0), json!(6.0)],
            );
        }
        1 => {
            track(
                &mut commands,
                object,
                "/formula/Mandelbox/scale",
                [json!(-1.45), json!(-1.5), json!(-1.55)],
            );
            track(
                &mut commands,
                object,
                "/formula/Mandelbox/min_radius_ratio",
                [json!(0.5), json!(0.4), json!(0.3)],
            );
        }
        2 => {
            track(
                &mut commands,
                object,
                "/formula/QuaternionJulia/constant/0",
                [json!(-0.12), json!(-0.20), json!(-0.28)],
            );
            track(
                &mut commands,
                object,
                "/formula/QuaternionJulia/slice_w",
                [json!(-0.07), json!(0.0), json!(0.07)],
            );
        }
        3 => {
            track(
                &mut commands,
                object,
                "/formula/Kifs/scale",
                [json!(1.75), json!(1.9), json!(2.05)],
            );
            track(
                &mut commands,
                object,
                "/formula/Kifs/rotation_degrees/1",
                [json!(-3.0), json!(0.5), json!(4.0)],
            );
        }
        _ => {
            track(
                &mut commands,
                object,
                "/formula/Apollonian/scale",
                [json!(1.2), json!(1.3), json!(1.4)],
            );
            track(
                &mut commands,
                object,
                "/trap_scale",
                [json!(1.0), json!(1.325), json!(1.65)],
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
        for (index, animated) in ANIMATED.iter().enumerate() {
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
                assert_eq!(snapshot.objects.len(), 1, "{}", animated.name);
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
    fn unfolding_controls_are_monotonic_across_the_whole_work_area() {
        for index in 0..ANIMATED.len() {
            let preset = scene(index).unwrap();
            let document = preset.document.unwrap();
            let mut previous = document.snapshot(0.0).unwrap();
            for frame in 1..250 {
                let current = document.snapshot(f64::from(frame)).unwrap();
                assert!(current.render.iterations >= previous.render.iterations);
                assert!(current.camera.distance >= previous.camera.distance);
                assert!(current.camera.pitch_degrees >= previous.camera.pitch_degrees);
                match (&previous.formula, &current.formula) {
                    (Formula::Mandelbulb(a), Formula::Mandelbulb(b)) => {
                        assert!(b.power >= a.power);
                        for axis in 0..2 {
                            assert!(b.angle_scale[axis] >= a.angle_scale[axis]);
                        }
                    }
                    (Formula::Mandelbox(a), Formula::Mandelbox(b)) => {
                        assert!(b.scale <= a.scale);
                        assert!(b.min_radius_ratio <= a.min_radius_ratio);
                    }
                    (Formula::QuaternionJulia(a), Formula::QuaternionJulia(b)) => {
                        assert!(b.constant[0] <= a.constant[0]);
                        assert!(b.slice_w >= a.slice_w);
                    }
                    (Formula::Kifs(a), Formula::Kifs(b)) => assert!(b.scale >= a.scale),
                    (Formula::Apollonian(a), Formula::Apollonian(b)) => assert!(b.scale >= a.scale),
                    _ => panic!("Preset changed formula family"),
                }
                previous = current;
            }
        }
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
