//! Portable animated worlds authored through the same commands as the editor. The built-in
//! template table (`templates::BUILTINS`) names them and is their only entry point; the other
//! animated showcase scenes there are authored documents (JSON).
use crate::scene::{
    Coloring, Facing, Formula, Kifs, KifsKind, MandelBox, MaterialModel, Quat, Scene,
};
use crate::world::{
    CAMERA_ORBIT_PHASE, CAMERA_ORBIT_SPEED, NodeId, WorldCommand, WorldDocument, WorldEditor,
};
use curves::Tan;
use serde_json::{Value, json};

/// The animated worlds this module authors (`templates::BUILTINS` gives their names and
/// descriptions).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Animated {
    /// Cool metallic Mandelbox folds.
    ChromeCathedral,
    /// A pearlescent quaternion Julia.
    OpalJulia,
    /// A jade octahedral KIFS.
    VerdantFold,
}
impl Animated {
    #[cfg(test)]
    pub const ALL: [Self; 3] = [Self::ChromeCathedral, Self::OpalJulia, Self::VerdantFold];
}
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

/// The world `preset` named `name`: 250 frames at 24 fps, the camera orbiting and pulling back
/// while the fractal unfolds.
pub fn scene(preset: Animated, name: &str) -> Result<Scene, String> {
    let mut base = Scene::preset(match preset {
        Animated::ChromeCathedral => Formula::Mandelbox(MandelBox::PRESET),
        Animated::OpalJulia => Formula::QuaternionJulia(Quat::PRESET),
        Animated::VerdantFold => Formula::Kifs(Kifs::preset(KifsKind::Tetrahedron)),
    });
    base.name = name.into();
    base.animation.first = 0;
    base.animation.last = 249;
    base.animation.fps = 24.0;
    base.camera.distance = match preset {
        Animated::ChromeCathedral => 3.1,
        Animated::OpalJulia => 2.6,
        Animated::VerdantFold => 3.8,
    };
    base.camera.pitch_degrees = 14.0;
    base.camera.f_number = 0.0;
    base.render.max_bounces = 4;
    base.render.exposure_stops = 0.0;
    base.material.model = MaterialModel::Fast;
    (base.material.specular_roughness, base.material.metalness) = match preset {
        Animated::ChromeCathedral => (0.30, 0.9),
        Animated::OpalJulia => (0.23, 0.12),
        Animated::VerdantFold => (0.3, 0.28),
    };
    base.coloring = Coloring::TrapOrigin;
    base.trap_scale = 1.25;
    let (palette, horizon, zenith, sun) = match preset {
        Animated::ChromeCathedral => (
            crate::palette::PaletteScheme::Ice,
            [0.12, 0.16, 0.22],
            [0.008, 0.014, 0.025],
            [0.78, 0.88, 1.0],
        ),
        Animated::OpalJulia => (
            crate::palette::PaletteScheme::Amethyst,
            [0.055, 0.025, 0.09],
            [0.045, 0.09, 0.16],
            [1.0, 0.8, 0.95],
        ),
        Animated::VerdantFold => (
            crate::palette::PaletteScheme::Verdant,
            [0.014, 0.04, 0.027],
            [0.055, 0.105, 0.09],
            [0.86, 1.0, 0.72],
        ),
    };
    base.palette = palette;
    base.sky.horizon = horizon;
    base.sky.zenith = zenith;
    base.sun.color = sun;
    base.sky.intensity = 4.5;
    base.sun.intensity = if preset == Animated::ChromeCathedral {
        4.0
    } else {
        2.5
    };
    base.sun.angle = 3.0;
    base.sun.azimuth = -35.0;
    base.sun.elevation = 48.0;
    base.sky.background = true;
    match preset {
        Animated::ChromeCathedral => {
            if let Formula::Mandelbox(parameters) = &mut base.formula {
                parameters.scale = -1.5;
            }
        }
        Animated::OpalJulia => {
            base.material.facing = Some(Facing {
                color: [0.18, 0.65, 0.9],
                roughness: 0.28,
                metallic: 0.35,
                exponent: 2.3,
            });
            base.material.thin_film_thickness = 420.0;
        }
        Animated::VerdantFold => base.formula = Formula::Kifs(Kifs::preset(KifsKind::Octahedron)),
    }
    let mut editor = WorldEditor::new(WorldDocument::from_scene(&base));
    let object = editor.active().ok_or("Preset has no fractal")?;
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
    let orbit_speed = match preset {
        Animated::ChromeCathedral => 7.0,
        Animated::OpalJulia => 8.0,
        Animated::VerdantFold => 9.0,
    };
    for (path, value) in [(CAMERA_ORBIT_SPEED, orbit_speed), (CAMERA_ORBIT_PHASE, 0.0)] {
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
    let iterations = match preset {
        Animated::ChromeCathedral => [6.0, 10.0, 14.0],
        Animated::OpalJulia | Animated::VerdantFold => [3.0, 8.0, 14.0],
    };
    track(
        &mut commands,
        object,
        "/render/iterations",
        iterations.map(|n| json!(n)),
    );
    match preset {
        Animated::ChromeCathedral => {
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
        Animated::OpalJulia => {
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
        Animated::VerdantFold => {
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
        for preset in Animated::ALL {
            let name = format!("{preset:?}");
            let authored = scene(preset, &name).unwrap();
            assert_eq!(authored.name, name);
            let document = authored.document.as_ref().unwrap();
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
                assert_eq!(snapshot.objects.len(), 1, "{name}");
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
    }

    #[test]
    fn unfolding_controls_are_monotonic_across_the_whole_work_area() {
        for preset in Animated::ALL {
            let document = scene(preset, "unfolding").unwrap().document.unwrap();
            let mut previous = document.snapshot(0.0).unwrap();
            for frame in 1..250 {
                let current = document.snapshot(f64::from(frame)).unwrap();
                assert!(current.render.iterations >= previous.render.iterations);
                assert!(current.camera.distance >= previous.camera.distance);
                assert!(current.camera.pitch_degrees >= previous.camera.pitch_degrees);
                match (&previous.formula, &current.formula) {
                    (Formula::Mandelbox(a), Formula::Mandelbox(b)) => {
                        assert!(b.scale <= a.scale);
                        assert!(b.min_radius_ratio <= a.min_radius_ratio);
                    }
                    (Formula::QuaternionJulia(a), Formula::QuaternionJulia(b)) => {
                        assert!(b.constant[0] <= a.constant[0]);
                        assert!(b.slice_w >= a.slice_w);
                    }
                    (Formula::Kifs(a), Formula::Kifs(b)) => assert!(b.scale >= a.scale),
                    _ => panic!("Preset changed formula family"),
                }
                previous = current;
            }
        }
    }

    #[test]
    fn preset_documents_roundtrip_and_evaluate_identically_in_reverse_and_at_subframes() {
        for preset in Animated::ALL {
            let document = scene(preset, "roundtrip").unwrap().document.unwrap();
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
