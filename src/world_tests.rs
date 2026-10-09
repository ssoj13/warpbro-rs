use super::*;

#[test]
fn glass_node_transmission_is_keyable_shared_and_roundtrips() {
    let mut e = editor();
    let first = find(&e, WorldKind::Fractal);
    e.execute(WorldCommand::Duplicate(vec![first])).unwrap();
    let second = e.selection.unwrap();
    let material = find(&e, WorldKind::Material);
    e.execute(WorldCommand::AssignMaterial {
        id: second,
        material: Some(material),
    })
    .unwrap();
    set(&mut e, material, "/material/transmission", json!(1.0), 0.0);
    set(
        &mut e,
        material,
        "/material/transmission_color",
        json!([0.12, 0.82, 0.25]),
        0.0,
    );
    e.document
        .set_attribute(
            material,
            "/material/transmission_depth",
            json!(0.5),
            0.0,
            true,
            Tan::Linear,
        )
        .unwrap();
    e.document
        .set_attribute(
            material,
            "/material/transmission_depth",
            json!(1.5),
            20.0,
            true,
            Tan::Linear,
        )
        .unwrap();
    let attrs = e.document.attributes(material, 10.0).unwrap();
    let depth = attrs
        .iter()
        .find(|a| a.path == "/material/transmission_depth")
        .unwrap();
    assert!(depth.keyable);
    assert_eq!(depth.frames, vec![0.0, 20.0]);
    let scene = e.document.snapshot(10.0).unwrap();
    assert_eq!(scene.objects.len(), 2);
    for object in &scene.objects {
        assert_eq!(object.material.transmission, 1.0);
        assert!((object.material.transmission_depth - 1.0).abs() < 1e-6);
        assert_eq!(object.material.transmission_color, [0.12, 0.82, 0.25]);
    }
    let loaded: WorldDocument =
        serde_json::from_str(&serde_json::to_string(&e.document).unwrap()).unwrap();
    assert_eq!(
        loaded.material(material, 10.0).unwrap(),
        scene.objects[0].material
    );
}

#[test]
fn incomplete_material_payload_is_rejected_without_schema_repair() {
    let mut e = editor();
    let id = find(&e, WorldKind::Material);
    let node = e.document.node_mut(id).unwrap();
    node["gpu"]["material"]
        .as_object_mut()
        .unwrap()
        .remove("transmission");
    let document = e.document.clone();
    let loaded = WorldEditor::new(document.clone());
    assert_eq!(loaded.document, document);
    assert!(loaded.document.material(id, 0.0).is_err());
}

#[test]
fn camera_recording_replaces_interval_preserves_outside_keys_and_undoes_atomically() {
    use crate::camera_recorder::{Options, Recording, Timing};
    let mut e = editor();
    let id = find(&e, WorldKind::Camera);
    for (frame, value) in [(0.0, 40.0), (20.0, 80.0), (100.0, 60.0)] {
        e.document
            .set_attribute(
                id,
                "/camera/fov_y_degrees",
                json!(value),
                frame,
                true,
                Tan::Linear,
            )
            .unwrap();
    }
    let original = e.document.clone();
    let scene = e.document.snapshot(10.0).unwrap();
    let options = Options {
        transform: false,
        focus: false,
        f_number: false,
        zoom: true,
        timing: Timing::KeepSpeed,
        ..Options::default()
    };
    let mut recording = Recording::new(&e.document, e.revision(), 10.0, &scene, options).unwrap();
    let mut camera = scene.camera;
    camera.fov_y_degrees = 44.0;
    recording.capture(1.0, camera).unwrap();
    e.record_camera(&recording).unwrap();
    let channel = &e
        .document
        .attrs(id)
        .unwrap()
        .anim("/camera/fov_y_degrees")
        .unwrap()
        .channels[0]
        .clone();
    assert_eq!(
        channel.keys().iter().map(|k| k.t()).collect::<Vec<_>>(),
        vec![0.0, 10.0, 34.0, 100.0]
    );
    assert_eq!(channel.sample_at(34.0), 44.0);
    assert_eq!(channel.sample_at(0.0), 40.0);
    assert_eq!(channel.sample_at(100.0), 60.0);
    assert!(e.undo());
    assert_eq!(e.document, original);
    assert!(
        !e.undo(),
        "the complete capture creates exactly one undo entry"
    );
    let serialized = serde_json::to_string(&e.document).unwrap();
    assert_eq!(
        serde_json::from_str::<WorldDocument>(&serialized).unwrap(),
        original
    );
}

#[test]
fn camera_recording_rejects_locked_and_changed_documents_without_partial_edits() {
    use crate::camera_recorder::{Options, Recording};
    let mut e = editor();
    let scene = e.document.snapshot(0.0).unwrap();
    let mut recording =
        Recording::new(&e.document, e.revision(), 0.0, &scene, Options::default()).unwrap();
    recording.capture(1.0, scene.camera).unwrap();
    let id = recording.camera;
    e.execute(WorldCommand::SetLocked { id, locked: true })
        .unwrap();
    let locked = e.document.clone();
    assert!(e.record_camera(&recording).is_err());
    assert_eq!(e.document, locked);
    assert!(Recording::new(&e.document, e.revision(), 0.0, &scene, Options::default()).is_err());
    assert_eq!(e.document, locked);
}

#[test]
fn standalone_material_creation_selects_one_node_without_assigning_and_undo_restores_selection() {
    let mut e = editor();
    let fractal = find(&e, WorldKind::Fractal);
    let before = e.document.clone();
    let previous_selection = e.selection;
    let mut surface = crate::scene::Material::default();
    crate::materials::PRESETS[1].apply(&mut surface);
    e.execute(WorldCommand::CreateMaterial {
        material: surface.clone(),
        name: "Shared copper".into(),
    })
    .unwrap();
    let id = e.selection.unwrap();
    assert_eq!(e.selected, vec![id]);
    assert_eq!(e.document.info(id).unwrap().kind, WorldKind::Material);
    assert_eq!(e.document.info(id).unwrap().name, "Shared copper");
    assert_eq!(e.document.material(id, 0.0).unwrap(), surface);
    assert_eq!(e.document.nodes().len(), before.nodes().len() + 1);
    assert_eq!(
        e.document.assigned_material(fractal).unwrap(),
        before.assigned_material(fractal).unwrap()
    );
    let created = e.document.clone();
    assert!(e.undo());
    assert_eq!(e.document, before);
    assert_eq!(e.selection, previous_selection);
    assert!(e.redo());
    assert_eq!(e.document, created);
    assert_eq!(e.selection, Some(id));
    assert_eq!(e.selected, vec![id]);
    let mut isolated = e.document.clone();
    isolated.active_camera = Some(NodeId::new());
    assert!(isolated.snapshot(0.0).is_err());
    assert_eq!(isolated.material(id, 0.0).unwrap(), surface);
    assert!(isolated.material(fractal, 0.0).is_err());
    assert!(isolated.material(id, f64::NAN).is_err());
}

#[test]
fn shared_material_assignments_reuse_uuid_and_animated_edits_roundtrip_for_all_consumers() {
    let mut e = editor();
    let first = find(&e, WorldKind::Fractal);
    e.execute(WorldCommand::Duplicate(vec![first])).unwrap();
    let second = e.selection.unwrap();
    e.execute(WorldCommand::CreateMaterial {
        material: crate::scene::Material::default(),
        name: "Animated shared surface".into(),
    })
    .unwrap();
    let material = e.selection.unwrap();
    let before_assignment = e.document.clone();
    e.execute(WorldCommand::Batch(
        vec![first, second]
            .into_iter()
            .map(|id| WorldCommand::AssignMaterial {
                id,
                material: Some(material),
            })
            .collect(),
    ))
    .unwrap();
    let assigned = e.document.clone();
    assert!(e.undo());
    assert_eq!(e.document, before_assignment);
    assert!(e.redo());
    assert_eq!(e.document, assigned);
    let revision = e.revision();
    let node_count = e.document.nodes().len();
    e.execute(WorldCommand::AssignMaterial {
        id: first,
        material: Some(material),
    })
    .unwrap();
    assert_eq!(e.revision(), revision);
    assert_eq!(e.document.nodes().len(), node_count);
    set(
        &mut e,
        material,
        "/material/specular_roughness",
        json!(0.2),
        0.0,
    );
    e.execute(WorldCommand::Key {
        id: material,
        path: "/material/specular_roughness".into(),
        frame: 0.0,
    })
    .unwrap();
    set(
        &mut e,
        material,
        "/material/specular_roughness",
        json!(0.8),
        20.0,
    );
    let saved: WorldDocument =
        serde_json::from_str(&serde_json::to_string(&e.document).unwrap()).unwrap();
    for frame in [0.0, 5.5, 10.0, 20.0] {
        let surface = e.document.material(material, frame).unwrap();
        assert_eq!(saved.material(material, frame).unwrap(), surface);
        for id in [first, second] {
            assert_eq!(saved.assigned_material(id).unwrap(), Some(material));
            assert_eq!(saved.node_scene(id, frame).unwrap().material, surface);
        }
        let snapshot = saved.snapshot(frame).unwrap();
        assert_eq!(snapshot.objects.len(), 2);
        assert!(
            snapshot
                .objects
                .iter()
                .all(|object| object.material == surface)
        );
    }
    let surface = e.document.material(material, 10.0).unwrap();
    assert!((surface.specular_roughness - 0.5).abs() < 1e-6);
}

#[test]
fn material_assignment_batch_rejects_locked_or_invalid_targets_without_partial_edits() {
    let mut e = editor();
    let first = find(&e, WorldKind::Fractal);
    e.execute(WorldCommand::Duplicate(vec![first])).unwrap();
    let second = e.selection.unwrap();
    e.execute(WorldCommand::CreateMaterial {
        material: crate::scene::Material::default(),
        name: "Candidate".into(),
    })
    .unwrap();
    let material = e.selection.unwrap();
    set(&mut e, second, "/locked", json!(true), 0.0);
    let before = e.document.clone();
    let revision = e.revision();
    assert!(
        e.execute(WorldCommand::Batch(vec![
            WorldCommand::AssignMaterial {
                id: first,
                material: Some(material)
            },
            WorldCommand::AssignMaterial {
                id: second,
                material: Some(material)
            },
        ]))
        .is_err()
    );
    assert_eq!(e.document, before);
    assert_eq!(e.revision(), revision);
    for (receiver, target) in [
        (first, e.document.active_camera.unwrap()),
        (material, material),
        (first, NodeId::new()),
    ] {
        assert!(
            e.execute(WorldCommand::AssignMaterial {
                id: receiver,
                material: Some(target)
            })
            .is_err()
        );
        assert_eq!(e.document, before);
        assert_eq!(e.revision(), revision);
    }
}

#[test]
fn unchanged_connected_camera_lens_allows_navigation_but_requested_lens_edit_rolls_back() {
    let mut e = editor();
    let camera = find(&e, WorldKind::Camera);
    let source = find(&e, WorldKind::Fractal);
    let mut source_attrs = e.document.attrs(source).unwrap();
    source_attrs.set("/custom/fov", to_attr(&json!(32.0)));
    e.document.store_attrs(source, &source_attrs).unwrap();
    let mut camera_attrs = e.document.attrs(camera).unwrap();
    camera_attrs.set_conn(
        "/camera/fov_y_degrees",
        Some(playa_engine::entities::attrs::AttrConnection {
            source_layer: source.0,
            source_key: "/custom/fov".into(),
        }),
    );
    e.document.store_attrs(camera, &camera_attrs).unwrap();
    let mut desired = e.document.snapshot(17.0).unwrap();
    assert_eq!(desired.camera.fov_y_degrees, 32.0);
    desired.camera.target[0] += 0.7;
    e.navigate_camera(&desired, 17.0, false, None).unwrap();
    assert_camera_navigation_pose(&e.document.snapshot(17.0).unwrap(), &desired);
    assert_eq!(
        e.document.snapshot(17.0).unwrap().camera.fov_y_degrees,
        32.0
    );
    let before = e.document.clone();
    let revision = e.revision();
    desired.camera.target[1] += 0.5;
    desired.camera.fov_y_degrees = 45.0;
    assert!(
        e.navigate_camera(&desired, 17.0, false, None)
            .unwrap_err()
            .contains("connected")
    );
    assert_eq!(e.document, before);
    assert_eq!(e.revision(), revision);
}

fn assert_camera_navigation_pose(actual: &Scene, expected: &Scene) {
    let actual_pack = actual.pack(640, 360);
    let expected_pack = expected.pack(640, 360);
    for start in [
        crate::params::P_CAM_ORIGIN,
        crate::params::P_CAM_FORWARD,
        crate::params::P_CAM_UP,
    ] {
        for lane in 0..3 {
            assert!(
                (actual_pack[start + lane] - expected_pack[start + lane]).abs() < 0.0004,
                "camera lane {}: actual {}, expected {}",
                start + lane,
                actual_pack[start + lane],
                expected_pack[start + lane]
            );
        }
    }
    assert_eq!(actual.camera.free_flight, expected.camera.free_flight);
}

#[test]
fn camera_navigation_solves_affine_parent_scale_pivot_and_orbit_into_visible_trs() {
    let mut e = editor();
    let camera = find(&e, WorldKind::Camera);
    e.execute(WorldCommand::Create {
        kind: WorldKind::Group,
        name: "Camera parent".into(),
        parent: None,
    })
    .unwrap();
    let parent = e.selection.unwrap();
    e.execute(WorldCommand::Reparent {
        id: camera,
        parent: Some(parent),
    })
    .unwrap();
    for (id, path, value) in [
        (parent, "/transform/position", json!([2.0, -1.0, 0.4])),
        (
            parent,
            "/transform/rotation_degrees",
            json!([14.0, -22.0, 31.0]),
        ),
        (parent, "/transform/scale", json!([2.0, 0.7, 1.4])),
        (camera, "/transform/scale", json!([0.8, 1.3, 1.1])),
        (camera, "/transform/pivot", json!([0.2, -0.3, 0.1])),
        (camera, CAMERA_ORBIT_SPEED, json!(12.0)),
    ] {
        set(&mut e, id, path, value, 0.0);
    }
    let original = e.document.clone();
    let canonical_camera = e.document.attrs(camera).unwrap();
    let mut desired = e.document.snapshot(37.5).unwrap();
    desired.camera.target[0] += 0.6;
    desired.camera.target[1] -= 0.2;
    desired.camera.yaw_degrees += 9.0;
    desired.camera.pitch_degrees -= 6.0;
    desired.camera.roll_degrees += 4.0;
    desired.camera.distance *= 0.9;
    desired.camera.free_flight = true;
    e.navigate_camera(&desired, 37.5, false, Some(88)).unwrap();
    assert_camera_navigation_pose(&e.document.snapshot(37.5).unwrap(), &desired);
    let attrs = e.document.attrs(camera).unwrap();
    for path in [
        "/camera/target",
        "/camera/yaw_degrees",
        "/camera/pitch_degrees",
        "/camera/roll_degrees",
        CAMERA_ORBIT_SPEED,
    ] {
        assert_eq!(attrs.get(path), canonical_camera.get(path));
        assert_eq!(
            serde_json::to_value(attrs.anim(path)).unwrap(),
            serde_json::to_value(canonical_camera.anim(path)).unwrap()
        );
    }
    assert_ne!(
        e.document
            .attribute_value(camera, "/transform/position", 37.5)
            .unwrap(),
        original
            .attribute_value(camera, "/transform/position", 37.5)
            .unwrap()
    );
    assert_ne!(
        e.document
            .attribute_value(camera, "/transform/rotation_degrees", 37.5)
            .unwrap(),
        original
            .attribute_value(camera, "/transform/rotation_degrees", 37.5)
            .unwrap()
    );
    assert_eq!(
        e.document
            .attribute_value(camera, "/transform/scale", 37.5)
            .unwrap(),
        original
            .attribute_value(camera, "/transform/scale", 37.5)
            .unwrap()
    );
    assert_eq!(
        e.document
            .attribute_value(camera, "/transform/pivot", 37.5)
            .unwrap(),
        original
            .attribute_value(camera, "/transform/pivot", 37.5)
            .unwrap()
    );
    let saved: WorldDocument =
        serde_json::from_str(&serde_json::to_string(&e.document).unwrap()).unwrap();
    assert_camera_navigation_pose(&saved.snapshot(37.5).unwrap(), &desired);
    e.finish_edit();
    assert!(e.undo());
    assert_eq!(e.document, original);
}

#[test]
fn camera_navigation_without_auto_key_preserves_curve_bytes_and_uses_static_offsets() {
    let mut e = editor();
    let camera = find(&e, WorldKind::Camera);
    for path in [
        "/transform/position/0",
        "/transform/rotation_degrees/1",
        "/camera/distance",
    ] {
        e.execute(WorldCommand::Key {
            id: camera,
            path: path.into(),
            frame: 0.0,
        })
        .unwrap();
        e.execute(WorldCommand::Key {
            id: camera,
            path: path.into(),
            frame: 40.0,
        })
        .unwrap();
    }
    let paths = [
        "/transform/position",
        "/transform/rotation_degrees",
        "/camera/distance",
    ];
    let before_attrs = e.document.attrs(camera).unwrap();
    let before_keys: Vec<_> = paths
        .iter()
        .map(|path| serde_json::to_value(before_attrs.anim(path)).unwrap())
        .collect();
    let original = e.document.clone();
    e.undo.clear();
    for delta in [0.3, 0.6, 0.9] {
        let mut desired = e.document.snapshot(12.5).unwrap();
        desired.camera.target[0] += delta;
        desired.camera.yaw_degrees += 5.0;
        desired.camera.distance *= 0.95;
        e.navigate_camera(&desired, 12.5, false, Some(91)).unwrap();
        assert_camera_navigation_pose(&e.document.snapshot(12.5).unwrap(), &desired);
        let current = e.document.attrs(camera).unwrap();
        for (path, keys) in paths.iter().zip(&before_keys) {
            assert_eq!(&serde_json::to_value(current.anim(path)).unwrap(), keys);
        }
        assert_eq!(e.undo.len(), 0);
    }
    assert!(
        e.document
            .attrs(camera)
            .unwrap()
            .contains("/_navigation/transform/position")
    );
    assert!(
        e.document
            .attributes(camera, 12.5)
            .unwrap()
            .iter()
            .all(|a| !a.path.starts_with("/_navigation/"))
    );
    let saved: WorldDocument =
        serde_json::from_str(&serde_json::to_string(&e.document).unwrap()).unwrap();
    for frame in [40.0, 12.5, 0.0, 19.75] {
        assert_eq!(
            saved.snapshot(frame).unwrap(),
            e.document.snapshot(frame).unwrap()
        );
    }
    e.finish_edit();
    assert_eq!(e.undo.len(), 1);
    assert!(e.undo());
    assert_eq!(e.document, original);
}

#[test]
fn camera_navigation_auto_key_compensates_existing_offsets_and_keys_changed_components_only() {
    let mut e = editor();
    let camera = find(&e, WorldKind::Camera);
    e.execute(WorldCommand::Key {
        id: camera,
        path: "/transform/position/0".into(),
        frame: 0.0,
    })
    .unwrap();
    let mut desired = e.document.snapshot(10.0).unwrap();
    desired.camera.target[0] += 0.5;
    e.navigate_camera(&desired, 10.0, false, None).unwrap();
    let offset = e
        .document
        .navigation_offset(camera, "/transform/position")
        .unwrap();
    let original = e.document.clone();
    desired = e.document.snapshot(10.0).unwrap();
    desired.camera.target[0] += 0.25;
    e.navigate_camera(&desired, 10.0, true, None).unwrap();
    assert_camera_navigation_pose(&e.document.snapshot(10.0).unwrap(), &desired);
    assert_eq!(
        e.document
            .navigation_offset(camera, "/transform/position")
            .unwrap(),
        offset
    );
    let attrs = e.document.attrs(camera).unwrap();
    let animation = attrs.anim("/transform/position").unwrap();
    assert_eq!(
        animation.channels[0]
            .keys()
            .iter()
            .map(|k| k.t())
            .collect::<Vec<_>>(),
        vec![0.0, 10.0]
    );
    assert!(animation.channels[1].is_empty());
    assert!(animation.channels[2].is_empty());
    e.execute(WorldCommand::RemoveKey {
        id: camera,
        path: "/transform/position/0".into(),
        frame: 10.0,
    })
    .unwrap();
    e.execute(WorldCommand::SetAnimation {
        id: camera,
        path: "/transform/position/0".into(),
        enabled: false,
        frame: 0.0,
    })
    .unwrap();
    let held = e
        .document
        .attribute_value(camera, "/transform/position/0", 0.0)
        .unwrap();
    e.execute(WorldCommand::Key {
        id: camera,
        path: "/transform/position/0".into(),
        frame: 15.0,
    })
    .unwrap();
    assert_eq!(
        e.document
            .attribute_value(camera, "/transform/position/0", 15.0)
            .unwrap(),
        held
    );
    assert_ne!(e.document, original);
}

#[test]
fn snapshot_bridge_never_generates_camera_keys_and_navigation_noop_is_clean() {
    let mut e = editor();
    let original = e.document.clone();
    let revision = e.revision();
    let before = e.document.snapshot(14.0).unwrap();
    e.navigate_camera(&before, 14.0, false, None).unwrap();
    assert_eq!(e.document, original);
    assert_eq!(e.revision(), revision);
    let mut after = before.clone();
    after.camera.yaw_degrees += 20.0;
    e.edit_snapshot(None, &before, &after, 14.0).unwrap();
    assert_eq!(e.document, original);
    assert_eq!(e.revision(), revision);
}

#[test]
fn camera_orbit_defaults_are_authored_at_creation_and_preserve_camera_pose() {
    let scene = Scene::preset(0);
    let e = editor();
    let id = find(&e, WorldKind::Camera);
    let before = serde_json::to_value(&e.document).unwrap();
    for frame in [0.0, 12.5, 200.0] {
        assert_eq!(e.document.snapshot(frame).unwrap().camera, scene.camera);
        assert_eq!(
            e.document
                .attribute_value(id, CAMERA_ORBIT_SPEED, frame)
                .unwrap(),
            json!(0.0)
        );
        assert!(
            e.document
                .attributes(id, frame)
                .unwrap()
                .iter()
                .any(|a| a.path == CAMERA_ORBIT_SPEED && a.keyable)
        );
    }
    assert_eq!(serde_json::to_value(&e.document).unwrap(), before);
}

#[test]
fn camera_orbit_is_frame_based_signed_and_survives_save_load() {
    let mut e = editor();
    let id = find(&e, WorldKind::Camera);
    e.execute(WorldCommand::SetTimeRange {
        first: 12,
        last: 249,
        fps: 24.0,
    })
    .unwrap();
    set(&mut e, id, CAMERA_ORBIT_SPEED, json!(-12.0), 12.0);
    set(&mut e, id, CAMERA_ORBIT_PHASE, json!(7.0), 12.0);
    let saved: WorldDocument =
        serde_json::from_str(&serde_json::to_string(&e.document).unwrap()).unwrap();
    for frame in [249.0, 12.0, 36.5, 0.0, 36.5] {
        let camera = e.document.snapshot(frame).unwrap().camera;
        let expected = 35.0 + 7.0 - 12.0 * ((frame - 12.0) / 24.0) as f32;
        assert!((camera.yaw_degrees - expected).abs() < 0.0001);
        assert_eq!(camera, saved.snapshot(frame).unwrap().camera);
    }
    e.document.fps = 48.0;
    assert!((e.document.snapshot(60.0).unwrap().camera.yaw_degrees - 30.0).abs() < 0.0001);
}

#[test]
fn camera_animated_orbit_speed_integrates_the_curve_instead_of_multiplying_current_speed() {
    let mut e = editor();
    let id = find(&e, WorldKind::Camera);
    set(&mut e, id, CAMERA_ORBIT_SPEED, json!(0.0), 0.0);
    e.execute(WorldCommand::Key {
        id,
        path: CAMERA_ORBIT_SPEED.into(),
        frame: 0.0,
    })
    .unwrap();
    set(&mut e, id, CAMERA_ORBIT_SPEED, json!(90.0), 24.0);
    assert!((e.document.snapshot(24.0).unwrap().camera.yaw_degrees - 80.0).abs() < 0.0001);
    assert!((e.document.snapshot(12.0).unwrap().camera.yaw_degrees - 46.25).abs() < 0.0001);
    assert_eq!(
        e.document.snapshot(12.0).unwrap().camera,
        e.document.snapshot(12.0).unwrap().camera
    );
}

#[test]
fn viewport_yaw_edits_preserve_authored_yaw_under_active_orbit_and_undo() {
    let mut e = editor();
    let camera = find(&e, WorldKind::Camera);
    set(&mut e, camera, CAMERA_ORBIT_SPEED, json!(12.0), 0.0);
    let original = e.document.clone();
    let before = e.document.snapshot(48.0).unwrap();
    let mut after = before.clone();
    after.camera.yaw_degrees += 10.0;
    e.navigate_camera(&after, 48.0, false, None).unwrap();
    assert_eq!(
        e.document
            .attribute_value(camera, "/camera/yaw_degrees", 48.0)
            .unwrap(),
        json!(35.0)
    );
    assert_camera_navigation_pose(&e.document.snapshot(48.0).unwrap(), &after);
    assert!(e.undo());
    assert_eq!(e.document, original);
}

fn drag_value(id: NodeId, value: f64) -> WorldCommand {
    WorldCommand::SetAttribute {
        id,
        path: "/transform/position/0".into(),
        value: json!(value),
        frame: 0.0,
    }
}

#[test]
fn parameter_drag_is_live_but_commits_one_undo_on_release() {
    let mut e = editor();
    let id = find(&e, WorldKind::Fractal);
    let original = e.document.clone();
    for value in [1.0, 2.0, 3.0, 4.0] {
        let revision = e.revision();
        e.execute_edit(drag_value(id, value), Some(7)).unwrap();
        assert!(e.revision() > revision);
        assert_eq!(
            e.document
                .attribute_value(id, "/transform/position/0", 0.0)
                .unwrap(),
            json!(value)
        );
        assert_eq!(e.undo.len(), 0);
        assert_eq!(e.active_edit(), Some(7));
    }
    e.finish_edit_unless(Some(7));
    assert_eq!(e.undo.len(), 0);
    e.finish_edit_unless(None);
    assert_eq!(e.undo.len(), 1);
    let final_document = e.document.clone();
    assert!(e.undo());
    assert_eq!(e.document, original);
    assert!(!e.undo());
    assert!(e.redo());
    assert_eq!(e.document, final_document);
}

#[test]
fn parameter_drag_returning_to_original_preserves_redo() {
    let mut e = editor();
    let id = find(&e, WorldKind::Fractal);
    e.execute(drag_value(id, 9.0)).unwrap();
    assert!(e.undo());
    let original = e.document.clone();
    let value = e
        .document
        .attribute_value(id, "/transform/position/0", 0.0)
        .unwrap();
    e.execute_edit(drag_value(id, 3.0), Some(11)).unwrap();
    e.execute_edit(
        WorldCommand::SetAttribute {
            id,
            path: "/transform/position/0".into(),
            value,
            frame: 0.0,
        },
        Some(11),
    )
    .unwrap();
    assert!(!e.finish_edit());
    assert_eq!(e.document, original);
    assert_eq!(e.undo.len(), 0);
    assert!(e.redo());
    assert_eq!(
        e.document
            .attribute_value(id, "/transform/position/0", 0.0)
            .unwrap(),
        json!(9.0)
    );
}

#[test]
fn failed_live_edit_rolls_back_and_unrelated_command_stays_separate() {
    let mut e = editor();
    let id = find(&e, WorldKind::Fractal);
    let original = e.document.clone();
    e.execute_edit(drag_value(id, 2.0), Some(17)).unwrap();
    let valid = e.document.clone();
    assert!(
        e.execute_edit(
            WorldCommand::Batch(vec![
                drag_value(id, 4.0),
                WorldCommand::SetSpan {
                    id,
                    start: 5.0,
                    end: 4.0
                },
            ]),
            Some(17)
        )
        .is_err()
    );
    assert_eq!(e.document, valid);
    assert_eq!(e.undo.len(), 0);
    assert_eq!(e.active_edit(), Some(17));
    e.execute(WorldCommand::Rename {
        id,
        name: "Separate rename".into(),
    })
    .unwrap();
    assert_eq!(e.undo.len(), 2);
    assert!(e.undo());
    assert_eq!(e.document, valid);
    assert!(e.undo());
    assert_eq!(e.document, original);
}

#[test]
fn render_settings_drag_shares_deferred_history_and_undo() {
    let mut e = editor();
    let original = e.document.clone();
    for exposure in [1.0, 2.0, 3.0] {
        let before = e.document.snapshot(0.0).unwrap();
        let mut after = before.clone();
        after.render.exposure_stops = exposure;
        e.edit_snapshot_with_gesture(e.selection, &before, &after, 0.0, Some(23))
            .unwrap();
        assert_eq!(e.undo.len(), 0);
        assert_eq!(
            e.document.snapshot(0.0).unwrap().render.exposure_stops,
            exposure
        );
    }
    assert!(e.finish_edit());
    assert_eq!(e.undo.len(), 1);
    assert!(e.undo());
    assert_eq!(e.document, original);
}

#[test]
fn material_library_assignment_is_atomic_and_keeps_object_selection() {
    let mut e = editor();
    let fractal = find(&e, WorldKind::Fractal);
    let original = e.document.clone();
    let old_material = e.document.assigned_material(fractal).unwrap().unwrap();
    let old_surface = e.document.node_scene(old_material, 0.0).unwrap().material;
    let mut material = old_surface.clone();
    crate::materials::PRESETS[1].apply(&mut material);
    e.execute(WorldCommand::ApplyMaterial {
        id: fractal,
        material: material.clone(),
        name: "Library metal".into(),
        frame: 0.0,
    })
    .unwrap();
    let assigned = e.document.assigned_material(fractal).unwrap().unwrap();
    assert_eq!(assigned, old_material);
    assert_eq!(e.selection, Some(fractal));
    assert_eq!(
        e.document.node_scene(old_material, 0.0).unwrap().material,
        material
    );
    assert_eq!(
        e.document.snapshot(0.0).unwrap().objects[0].material,
        material
    );
    assert_eq!(e.document.nodes().len(), original.nodes().len());
    assert!(e.undo());
    assert_eq!(e.document, original);
    assert!(e.redo());
    assert_eq!(
        e.document.assigned_material(fractal).unwrap(),
        Some(assigned)
    );

    // Editing a Material node keeps its identity and existing animation channels.
    e.execute(WorldCommand::Key {
        id: assigned,
        path: "/material/specular_roughness".into(),
        frame: 0.0,
    })
    .unwrap();
    let mut edited = material.clone();
    edited.specular_roughness = 0.8;
    e.execute(WorldCommand::ApplyMaterial {
        id: assigned,
        material: edited.clone(),
        name: "Unused rename".into(),
        frame: 10.0,
    })
    .unwrap();
    assert_eq!(
        e.document.assigned_material(fractal).unwrap(),
        Some(assigned)
    );
    let middle = e
        .document
        .attribute_value(assigned, "/material/specular_roughness", 5.0)
        .unwrap()
        .as_f64()
        .unwrap();
    assert!((middle - f64::from((material.specular_roughness + 0.8) * 0.5)).abs() < 1e-6);

    set(&mut e, fractal, "/locked", json!(true), 0.0);
    let locked = e.document.clone();
    assert!(
        e.execute(WorldCommand::ApplyMaterial {
            id: fractal,
            material: edited,
            name: "Rejected".into(),
            frame: 0.0
        })
        .is_err()
    );
    assert_eq!(e.document, locked);
}
#[test]
fn editor_revision_invalidates_caches_on_edit_undo_and_redo_only() {
    let mut e = editor();
    let id = find(&e, WorldKind::Fractal);
    assert_eq!(e.revision(), 0);
    e.execute(WorldCommand::Batch(vec![])).unwrap();
    assert_eq!(e.revision(), 0);
    assert!(
        e.execute(WorldCommand::SetSpan {
            id,
            start: 10.0,
            end: 5.0
        })
        .is_err()
    );
    assert_eq!(e.revision(), 0);
    e.execute(WorldCommand::Rename {
        id,
        name: "Changed".into(),
    })
    .unwrap();
    let revision = e.revision();
    assert!(revision > 0);
    assert!(e.undo());
    assert!(e.revision() > revision);
    let revision = e.revision();
    assert!(e.redo());
    assert!(e.revision() > revision);
    let before = e.document.snapshot(0.0).unwrap();
    let mut after = before.clone();
    after.render.max_steps += 1;
    let revision = e.revision();
    e.edit_snapshot(Some(id), &before, &after, 0.0).unwrap();
    assert!(e.revision() > revision);
}
#[test]
fn component_stopwatch_disables_only_selected_channel_and_holds_value() {
    let mut e = editor();
    let camera = find(&e, WorldKind::Camera);
    set(
        &mut e,
        camera,
        "/camera/target",
        json!([3.0, 4.0, 5.0]),
        0.0,
    );
    e.execute(WorldCommand::SetAnimation {
        id: camera,
        path: "/camera/target/0".into(),
        enabled: true,
        frame: 0.0,
    })
    .unwrap();
    set(&mut e, camera, "/camera/target/0", json!(13.0), 10.0);
    set(&mut e, camera, "/camera/target/1", json!(7.0), 10.0);
    assert_eq!(
        e.document.snapshot(5.0).unwrap().camera.target,
        [8.0, 7.0, 5.0]
    );
    e.execute(WorldCommand::SetAnimation {
        id: camera,
        path: "/camera/target/0".into(),
        enabled: false,
        frame: 5.0,
    })
    .unwrap();
    assert_eq!(
        e.document.snapshot(10.0).unwrap().camera.target,
        [8.0, 7.0, 5.0]
    );
    assert!(
        e.document
            .attributes(camera, 5.0)
            .unwrap()
            .iter()
            .find(|a| a.path == "/camera/target/0")
            .unwrap()
            .frames
            .is_empty()
    );
}
#[test]
fn environment_reload_revision_is_authoritative_and_undoable() {
    let mut e = editor();
    let env = find(&e, WorldKind::Environment);
    e.execute(WorldCommand::ReloadEnvironment(env)).unwrap();
    assert_eq!(e.document.snapshot(0.0).unwrap().environment.revision, 1);
    assert!(e.undo());
    assert_eq!(e.document.snapshot(0.0).unwrap().environment.revision, 0);
}
fn editor() -> WorldEditor {
    WorldEditor::new(WorldDocument::from_scene(&Scene::preset(0)))
}
fn find(e: &WorldEditor, kind: WorldKind) -> NodeId {
    e.document
        .nodes()
        .iter()
        .find(|n| n.kind == kind)
        .unwrap()
        .id
}
fn set(e: &mut WorldEditor, id: NodeId, path: &str, value: Value, frame: f64) {
    e.execute(WorldCommand::SetAttribute {
        id,
        path: path.into(),
        value,
        frame,
    })
    .unwrap();
}
#[test]
fn constructs_all_formula_families_without_dropping_parameters() {
    for family in 0..8 {
        let s = Scene::preset(family);
        let world = WorldDocument::from_scene(&s);
        let snapshot = world.snapshot(0.0).unwrap();
        assert_eq!(snapshot.formula, s.formula);
        assert_eq!(snapshot.material, s.material);
        let mut expected_render = s.render.clone();
        expected_render.iterations = Scene::preset(crate::params::FAMILY_BULB).render.iterations;
        assert_eq!(snapshot.render, expected_render);
        assert_eq!(snapshot.camera, s.camera);
        assert_eq!(snapshot.environment, s.environment);
        assert_eq!(snapshot.objects.len(), 1);
        assert_eq!(snapshot.objects[0].render.iterations, s.render.iterations);
        assert_eq!(snapshot.lights.len(), 1);
        assert_eq!(
            world.runtime_graph().unwrap().nodes.len(),
            world.nodes().len()
        );
    }
}
#[test]
fn numeric_component_keys_and_render_keys_share_playa_evaluation() {
    let mut s = Scene::preset(0);
    s.camera.target = [10.0, 20.0, 30.0];
    let w = WorldDocument::from_scene(&s);
    let mut e = WorldEditor::new(w);
    let camera = find(&e, WorldKind::Camera);
    e.execute(WorldCommand::Key {
        id: camera,
        path: "/camera/target/0".into(),
        frame: 12.0,
    })
    .unwrap();
    set(&mut e, camera, "/camera/target/0", json!(100.0), 12.0);
    let sampled = e.document.snapshot(12.0).unwrap();
    assert_eq!(sampled.camera.target, [100.0, 20.0, 30.0]);
    e.execute(WorldCommand::MoveKeys {
        id: camera,
        path: "/camera/target/0".into(),
        frames: vec![12.0],
        delta: 1.25,
    })
    .unwrap();
    let attrs = e.document.attributes(camera, 13.25).unwrap();
    assert!(
        attrs
            .iter()
            .find(|a| a.path == "/camera/target/0")
            .unwrap()
            .frames
            .contains(&13.25)
    );
    assert!(
        !attrs
            .iter()
            .find(|a| a.path == "/camera/target/1")
            .unwrap()
            .frames
            .contains(&13.25)
    );
}
#[test]
fn metadata_and_unknown_gpu_data_survive_roundtrip_with_stable_ids() {
    let mut e = editor();
    let id = find(&e, WorldKind::Fractal);
    let metadata = json!({"notes":["arbitrary",{"nested":true}],"shot":42,"nullable":null});
    e.execute(WorldCommand::SetMetadata {
        id,
        path: "".into(),
        value: metadata.clone(),
    })
    .unwrap();
    e.document.node_mut(id).unwrap()["gpu"]["unknown_vendor_extension"] =
        json!({"scalar":1.75,"channels":[1,2,3]});
    set(&mut e, id, "/custom/gain", json!(2.5), 0.0);
    e.execute(WorldCommand::Key {
        id,
        path: "/custom/gain".into(),
        frame: 0.25,
    })
    .unwrap();
    let restored: WorldDocument =
        serde_json::from_value(serde_json::to_value(&e.document).unwrap()).unwrap();
    assert_eq!(restored, e.document);
    assert_eq!(restored.metadata(id).unwrap(), metadata);
    assert_eq!(
        restored.runtime_graph().unwrap().nodes[&id]
            .data
            .to_json_value()["gpu"]["unknown_vendor_extension"]["scalar"],
        json!(1.75)
    );
    assert_eq!(
        restored.attribute_value(id, "/custom/gain", 0.25).unwrap(),
        json!(2.5)
    );
}
#[test]
fn parent_nonuniform_rotation_chain_is_full_matrix_and_span_half_open() {
    let mut e = editor();
    let child = find(&e, WorldKind::Fractal);
    e.execute(WorldCommand::Create {
        kind: WorldKind::Group,
        name: "Parent".into(),
        parent: None,
    })
    .unwrap();
    let parent = e.selection.unwrap();
    set(
        &mut e,
        parent,
        "/transform/position",
        json!([3.0, 4.0, 5.0]),
        0.0,
    );
    set(
        &mut e,
        parent,
        "/transform/rotation_degrees",
        json!([0.0, 0.0, 37.0]),
        0.0,
    );
    set(
        &mut e,
        parent,
        "/transform/scale",
        json!([2.0, 3.0, 4.0]),
        0.0,
    );
    set(
        &mut e,
        child,
        "/transform/position",
        json!([1.0, 2.0, 3.0]),
        0.0,
    );
    set(
        &mut e,
        child,
        "/transform/rotation_degrees",
        json!([20.0, 30.0, 0.0]),
        0.0,
    );
    e.execute(WorldCommand::Reparent {
        id: child,
        parent: Some(parent),
    })
    .unwrap();
    e.execute(WorldCommand::SetSpan {
        id: parent,
        start: 2.0,
        end: 8.0,
    })
    .unwrap();
    assert!(e.document.snapshot(1.99).unwrap().objects.is_empty());
    assert_eq!(e.document.snapshot(2.0).unwrap().objects.len(), 1);
    assert!(e.document.snapshot(8.0).unwrap().objects.is_empty());
    let matrix = e.document.snapshot(4.0).unwrap().objects[0]
        .object_world
        .unwrap();
    let p = playa_engine::entities::transform::build_model_matrix(
        [3.0, 4.0, 5.0],
        [0.0, 0.0, 37_f32.to_radians()],
        [2.0, 3.0, 4.0],
        [0.0; 3],
    );
    let c = playa_engine::entities::transform::build_model_matrix(
        [1.0, 2.0, 3.0],
        [20_f32.to_radians(), 30_f32.to_radians(), 0.0],
        [1.0; 3],
        [0.0; 3],
    );
    assert_eq!(matrix, (p * c).to_cols_array_2d());
    let snapshot = e.document.snapshot(4.0).unwrap();
    let bounds = snapshot.framing_bounds();
    let radius = Scene::preset(0).pack(1, 1)[crate::params::P_CLIP_RADIUS];
    let matrix = glam::Mat4::from_cols_array_2d(&matrix);
    for x in [-1.0, 1.0] {
        for y in [-1.0, 1.0] {
            for z in [-1.0, 1.0] {
                let corner =
                    matrix.transform_point3(glam::Vec3::new(x, y, z) * radius / 3.0_f32.sqrt());
                assert!(corner.cmpge(bounds.0 - glam::Vec3::splat(1e-4)).all());
                assert!(corner.cmple(bounds.1 + glam::Vec3::splat(1e-4)).all());
            }
        }
    }
    e.selection = Some(parent);
    assert_eq!(bounds, e.document.snapshot(4.0).unwrap().framing_bounds());
}
#[test]
fn graph_transactions_rollback_cycles_and_undo_atomically() {
    let mut e = editor();
    let id = find(&e, WorldKind::Fractal);
    let camera = find(&e, WorldKind::Camera);
    e.selected = vec![id, camera];
    let before = e.document.clone();
    assert!(
        e.execute(WorldCommand::Batch(vec![
            WorldCommand::Rename {
                id,
                name: "Changed".into()
            },
            WorldCommand::Reparent {
                id,
                parent: Some(id)
            }
        ]))
        .is_err()
    );
    assert_eq!(e.document, before);
    assert_eq!(e.selected, vec![id, camera]);
    assert!(!e.can_undo());
    e.execute(WorldCommand::Batch(vec![
        WorldCommand::Rename {
            id,
            name: "Changed".into(),
        },
        WorldCommand::SetVisible { id, visible: false },
    ]))
    .unwrap();
    assert_eq!(e.selected, vec![id, camera]);
    assert!(e.document.snapshot(0.0).unwrap().objects.is_empty());
    assert!(e.undo());
    assert_eq!(e.document, before);
    assert!(e.redo());
    assert_eq!(e.document.info(id).unwrap().name, "Changed");
}
#[test]
fn duplicate_retains_independent_keys_and_all_visible_objects() {
    let mut e = editor();
    let original = find(&e, WorldKind::Fractal);
    e.execute(WorldCommand::Key {
        id: original,
        path: "/transform/position".into(),
        frame: 0.0,
    })
    .unwrap();
    set(
        &mut e,
        original,
        "/transform/position",
        json!([2.0, 3.0, 4.0]),
        10.0,
    );
    e.execute(WorldCommand::Duplicate(vec![original])).unwrap();
    let duplicate = e.selection.unwrap();
    assert_ne!(original, duplicate);
    assert_eq!(e.document.snapshot(5.0).unwrap().objects.len(), 2);
    set(
        &mut e,
        duplicate,
        "/transform/position/0",
        json!(20.0),
        10.0,
    );
    assert_ne!(
        e.document
            .attribute_value(original, "/transform/position", 10.0)
            .unwrap(),
        e.document
            .attribute_value(duplicate, "/transform/position", 10.0)
            .unwrap()
    );
    e.execute(WorldCommand::SetSolo {
        id: duplicate,
        solo: true,
    })
    .unwrap();
    assert_eq!(e.document.snapshot(5.0).unwrap().objects.len(), 1);
}
#[test]
fn export_frozen_document_uses_identical_preview_snapshot() {
    let mut e = editor();
    let camera = find(&e, WorldKind::Camera);
    e.execute(WorldCommand::Key {
        id: camera,
        path: "/camera/distance".into(),
        frame: 0.0,
    })
    .unwrap();
    set(&mut e, camera, "/camera/distance", json!(5.0), 24.0);
    let mut exported = Scene::preset(0);
    exported.document = Some(Box::new(e.document.clone()));
    for frame in [0.0, 3.5, 12.0, 24.0] {
        assert_eq!(
            exported.evaluated(frame).unwrap(),
            e.document.snapshot(frame).unwrap()
        );
    }
}
#[test]
fn material_uuid_assignments_are_validated_and_snapshot_resolved() {
    let mut e = editor();
    let id = find(&e, WorldKind::Fractal);
    let material = find(&e, WorldKind::Material);
    let camera = find(&e, WorldKind::Camera);
    set(
        &mut e,
        material,
        "/material/specular_roughness",
        json!(0.125),
        0.0,
    );
    e.execute(WorldCommand::AssignMaterial {
        id,
        material: Some(material),
    })
    .unwrap();
    assert_eq!(
        e.document.snapshot(0.0).unwrap().objects[0]
            .material
            .specular_roughness,
        0.125
    );
    assert!(
        e.execute(WorldCommand::AssignMaterial {
            id,
            material: Some(camera)
        })
        .is_err()
    );
    let attr = e
        .document
        .attributes(id, 0.0)
        .unwrap()
        .into_iter()
        .find(|a| a.path == "/material_id")
        .unwrap();
    assert_eq!(attr.value, json!(material));
    assert!(!attr.keyable);
}
#[test]
fn camera_reference_survives_visibility_reorder_and_formula_differences() {
    let mut e = editor();
    let first = find(&e, WorldKind::Fractal);
    let baseline = e.document.snapshot(0.0).unwrap().pack(8, 8);
    e.execute(WorldCommand::Create {
        kind: WorldKind::Fractal,
        name: "Second".into(),
        parent: None,
    })
    .unwrap();
    let second = e.selection.unwrap();
    set(
        &mut e,
        second,
        "/formula",
        serde_json::to_value(Scene::preset(1).formula).unwrap(),
        0.0,
    );
    e.execute(WorldCommand::Reorder {
        ids: vec![second, first],
    })
    .unwrap();
    let reordered = e.document.snapshot(0.0).unwrap().pack(8, 8);
    e.execute(WorldCommand::SetVisible {
        id: first,
        visible: false,
    })
    .unwrap();
    let hidden = e.document.snapshot(0.0).unwrap().pack(8, 8);
    let origin = crate::params::P_CAM_ORIGIN;
    assert_eq!(
        &baseline[origin..origin + 3],
        &reordered[origin..origin + 3]
    );
    assert_eq!(&baseline[origin..origin + 3], &hidden[origin..origin + 3]);
    let restored: WorldDocument =
        serde_json::from_value(serde_json::to_value(&e.document).unwrap()).unwrap();
    assert_eq!(restored.nodes()[0].id, second);
    assert!(e.undo());
    assert!(e.undo());
    assert_ne!(e.document.nodes()[0].id, second);
}
#[test]
fn parent_camera_pose_environment_visibility_and_ancestor_lock() {
    let mut e = editor();
    let camera = find(&e, WorldKind::Camera);
    let env = find(&e, WorldKind::Environment);
    let initial = e.document.snapshot(0.0).unwrap();
    e.execute(WorldCommand::Create {
        kind: WorldKind::Group,
        name: "Rig".into(),
        parent: None,
    })
    .unwrap();
    let parent = e.selection.unwrap();
    set(
        &mut e,
        parent,
        "/transform/position",
        json!([4.0, 5.0, 6.0]),
        0.0,
    );
    set(
        &mut e,
        parent,
        "/transform/rotation_degrees",
        json!([0.0, 90.0, 0.0]),
        0.0,
    );
    set(
        &mut e,
        parent,
        "/transform/scale",
        json!([2.0, 2.0, 2.0]),
        0.0,
    );
    e.execute(WorldCommand::Reparent {
        id: camera,
        parent: Some(parent),
    })
    .unwrap();
    e.execute(WorldCommand::Reparent {
        id: env,
        parent: Some(parent),
    })
    .unwrap();
    let rigged = e.document.snapshot(0.0).unwrap();
    assert_eq!(rigged.camera.target, [4.0, 5.0, 6.0]);
    assert!((rigged.camera.distance - 2.0 * initial.camera.distance).abs() < 1e-5);
    let expected = playa_engine::entities::transform::build_model_matrix(
        [4.0, 5.0, 6.0],
        [0.0, 90_f32.to_radians(), 0.0],
        [2.0; 3],
        [0.0; 3],
    );
    let a = rigged.camera.orientation() * glam::Vec3::Z;
    let b = glam::Mat4::from_cols_array_2d(&expected.to_cols_array_2d())
        .transform_vector3(initial.camera.orientation() * glam::Vec3::Z)
        .normalize();
    assert!((a - b).length() < 1e-5);
    e.execute(WorldCommand::SetVisible {
        id: parent,
        visible: false,
    })
    .unwrap();
    assert!(!e.document.snapshot(0.0).unwrap().environment.enabled);
    assert_eq!(
        e.document.snapshot(0.0).unwrap().lighting.sky_intensity,
        0.0
    );
    assert!(!e.document.snapshot(0.0).unwrap().lighting.background);
    e.execute(WorldCommand::SetLocked {
        id: parent,
        locked: true,
    })
    .unwrap();
    assert!(
        e.execute(WorldCommand::Rename {
            id: camera,
            name: "Blocked".into()
        })
        .is_err()
    );
    e.execute(WorldCommand::SetLocked {
        id: parent,
        locked: false,
    })
    .unwrap();
    e.execute(WorldCommand::Rename {
        id: camera,
        name: "Allowed".into(),
    })
    .unwrap();
}
#[test]
fn per_object_gpu_iterations_are_independent_of_world_ray_settings() {
    let mut e = editor();
    let first = find(&e, WorldKind::Fractal);
    e.execute(WorldCommand::Create {
        kind: WorldKind::Fractal,
        name: "Second".into(),
        parent: None,
    })
    .unwrap();
    let second = e.selection.unwrap();
    set(&mut e, first, "/render/iterations", json!(7), 0.0);
    set(&mut e, second, "/render/iterations", json!(19), 0.0);
    let snapshot = e.document.snapshot(0.0).unwrap();
    assert_eq!(
        snapshot
            .objects
            .iter()
            .map(|o| o.render.iterations)
            .collect::<Vec<_>>(),
        vec![7, 19]
    );
    assert_ne!(snapshot.render.iterations, 7);
    assert_ne!(snapshot.render.iterations, 19);
}
#[test]
fn fractal_schema_offers_all_formula_defaults_and_keeps_custom_values() {
    let mut e = editor();
    let id = find(&e, WorldKind::Fractal);
    let descriptor = e
        .document
        .attributes(id, 0.0)
        .unwrap()
        .into_iter()
        .find(|a| a.path == "/formula")
        .unwrap();
    assert_eq!(descriptor.choices.len(), 8);
    for (family, choice) in descriptor.choices.into_iter().enumerate() {
        assert_eq!(
            choice,
            serde_json::to_value(Scene::preset(family as u32).formula).unwrap()
        );
        set(&mut e, id, "/formula", choice, 0.0);
        assert_eq!(
            e.document.node_scene(id, 0.0).unwrap().formula,
            Scene::preset(family as u32).formula
        );
    }
    let mut custom = serde_json::to_value(Scene::preset(0).formula).unwrap();
    let variant = custom.as_object_mut().unwrap().values_mut().next().unwrap();
    variant["power"] = json!(11.0);
    set(&mut e, id, "/formula", custom.clone(), 0.0);
    assert_eq!(
        serde_json::to_value(e.document.node_scene(id, 0.0).unwrap().formula).unwrap(),
        custom
    );
}
#[test]
fn julia_schema_toggle_accepts_custom_constants_and_is_undoable() {
    let mut e = editor();
    let id = find(&e, WorldKind::Fractal);
    let descriptor = e
        .document
        .attributes(id, 0.0)
        .unwrap()
        .into_iter()
        .find(|a| a.path == "/julia")
        .unwrap();
    assert_eq!(
        descriptor.choices,
        vec![Value::Null, json!([0.0, 0.0, 0.0])]
    );
    set(&mut e, id, "/julia", descriptor.choices[1].clone(), 0.0);
    let lanes: Vec<_> = e
        .document
        .attributes(id, 0.0)
        .unwrap()
        .into_iter()
        .filter(|a| a.path.starts_with("/julia/"))
        .collect();
    assert_eq!(lanes.len(), 3);
    assert!(lanes.iter().all(|a| a.choices.is_empty()));
    set(&mut e, id, "/julia", json!([0.3, 0.4, 0.5]), 0.0);
    assert_eq!(
        e.document.node_scene(id, 0.0).unwrap().julia,
        Some([0.3, 0.4, 0.5])
    );
    set(&mut e, id, "/julia", Value::Null, 0.0);
    assert_eq!(e.document.node_scene(id, 0.0).unwrap().julia, None);
    assert!(e.undo());
    assert_eq!(
        e.document.node_scene(id, 0.0).unwrap().julia,
        Some([0.3, 0.4, 0.5])
    );
}
#[test]
fn render_schema_filters_fractal_controls_but_keeps_world_settings_keys() {
    let scene = Scene::preset(0);
    let mut editor = WorldEditor::new(WorldDocument::from_scene(&scene));
    let settings = editor.document.output_render_profile().unwrap();
    for (frame, value) in [(0.0, 0.0), (12.0, 2.5)] {
        editor
            .execute(WorldCommand::Key {
                id: settings,
                path: "/render/exposure_stops".into(),
                frame,
            })
            .unwrap();
        set(
            &mut editor,
            settings,
            "/render/exposure_stops",
            json!(value),
            frame,
        );
    }
    let world = editor.document;
    let fractal = world
        .nodes()
        .into_iter()
        .find(|n| n.kind == WorldKind::Fractal)
        .unwrap()
        .id;
    let descriptors = world.attributes(fractal, 0.0).unwrap();
    assert_eq!(
        world.node(fractal).unwrap()["gpu"]["render"],
        json!({"iterations":scene.render.iterations})
    );
    assert!(
        !world
            .attrs(fractal)
            .unwrap()
            .contains("/render/exposure_stops")
    );
    assert!(
        !world
            .attrs(fractal)
            .unwrap()
            .contains("/render/max_bounces")
    );
    assert_eq!(
        descriptors
            .iter()
            .filter(|a| a.path.starts_with("/render/"))
            .map(|a| a.path.as_str())
            .collect::<Vec<_>>(),
        vec!["/render/iterations"]
    );
    assert!(
        world
            .attributes(settings, 0.0)
            .unwrap()
            .iter()
            .any(|a| a.path == "/render/exposure_stops")
    );
    assert!(
        world
            .attrs(settings)
            .unwrap()
            .is_animated("/render/exposure_stops")
    );
    // Two keys: a straight ramp 0 -> 2.5 over 12 frames.
    for frame in [0.0, 6.0, 12.0] {
        let want = 2.5 * (frame as f32) / 12.0;
        assert!((world.snapshot(frame).unwrap().render.exposure_stops - want).abs() < 1e-5);
    }
}
#[test]
fn clipboard_copy_paste_and_duplicate_remap_ids_in_one_undo_step() {
    let mut e = editor();
    let fractal = find(&e, WorldKind::Fractal);
    let material = find(&e, WorldKind::Material);
    e.execute(WorldCommand::AssignMaterial {
        id: fractal,
        material: Some(material),
    })
    .unwrap();
    let count = e.document.nodes().len();

    // Ctrl+D on two nodes: both copied, one undo step, copies selected.
    e.execute(WorldCommand::Duplicate(vec![fractal, material]))
        .unwrap();
    assert_eq!(e.document.nodes().len(), count + 2);
    assert_eq!(e.selected.len(), 2);
    let copy = *e
        .selected
        .iter()
        .find(|id| e.document.supports_material(**id))
        .unwrap();
    let copied_material = e.document.assigned_material(copy).unwrap().unwrap();
    assert_ne!(
        copied_material, material,
        "a reference inside the fragment follows the remap"
    );
    assert!(e.selected.contains(&copied_material));
    assert!(e.undo());
    assert_eq!(e.document.nodes().len(), count);

    // Ctrl+C / Ctrl+V: fresh UUIDs; an outside reference that exists is kept.
    let text = e.document.copy_fragment(&[fractal]).unwrap();
    e.execute(WorldCommand::Delete(fractal)).unwrap();
    e.execute(WorldCommand::Paste(text.clone())).unwrap();
    let pasted = e.selection.unwrap();
    assert_ne!(pasted, fractal, "paste never reuses the source UUID");
    assert_eq!(
        e.document.assigned_material(pasted).unwrap(),
        Some(material)
    );
    e.execute(WorldCommand::Paste(text)).unwrap();
    assert_ne!(e.selection.unwrap(), pasted, "every paste is a new node");

    // A copy holds the material's UUID, not the material: once the material is gone the
    // reference is reported as unresolved and the paste clears it.
    let text = e.document.copy_fragment(&[pasted]).unwrap();
    let nodes = parse_clipboard(&text).unwrap();
    assert!(e.document.unresolved_references(&nodes).is_empty());
    e.execute(WorldCommand::Delete(material)).unwrap();
    assert_eq!(e.document.unresolved_references(&nodes).len(), 1);
    e.execute(WorldCommand::Paste(text)).unwrap();
    assert_eq!(
        e.document.assigned_material(e.selection.unwrap()).unwrap(),
        None
    );

    // Foreign clipboard text is not WarpBro nodes and changes nothing.
    assert!(parse_clipboard("hello").is_none());
    let before = e.document.nodes().len();
    assert!(
        e.execute(WorldCommand::Paste("{\"nodes\":{}}".into()))
            .is_err()
    );
    assert_eq!(e.document.nodes().len(), before);
}

#[test]
fn fractal_slider_endpoints_pack_finite_and_hybrid_spans_match_standalone() {
    for family in 0..8 {
        let scene = crate::scene::Scene::preset(family);
        let document = WorldDocument::from_scene(&scene);
        let id = document
            .nodes()
            .into_iter()
            .find(|n| n.kind == WorldKind::Fractal)
            .unwrap()
            .id;
        for attr in document.attributes(id, 0.0).unwrap() {
            if attr.component.is_some()
                || !(attr.path.starts_with("/formula/")
                    || attr.path == "/julia"
                    || attr.path == "/render/iterations")
            {
                continue;
            }
            let Some(slider) = attr.slider else { continue };
            for endpoint in [slider.min, slider.max] {
                let value = if attr.value.is_u64() {
                    json!(endpoint as u64)
                } else if let Some(values) = attr.value.as_array() {
                    json!(vec![endpoint; values.len()])
                } else if attr.value.is_number() {
                    json!(endpoint)
                } else {
                    continue;
                };
                let mut e = WorldEditor::new(document.clone());
                set(&mut e, id, &attr.path, value, 0.0);
                let evaluated = e.document.node_scene(id, 0.0).unwrap();
                assert!(
                    evaluated.pack(128, 128).iter().all(|v| v.is_finite()),
                    "family {family}: {} = {endpoint}",
                    attr.path
                );
            }
        }
    }
    for (nested, standalone) in [
        ("bulb", "Mandelbulb"),
        ("mandelbox", "Mandelbox"),
        ("kifs", "Kifs"),
    ] {
        let prefix = format!("/formula/Hybrid/{nested}/");
        for param in ["power", "scale", "bailout", "offset", "rotation_degrees"] {
            let a = attribute_slider(&format!("{prefix}{param}"));
            let b = attribute_slider(&format!("/formula/{standalone}/{param}"));
            assert_eq!(
                a.map(|s| (s.min, s.max, s.log)),
                b.map(|s| (s.min, s.max, s.log))
            );
        }
    }
}

#[test]
fn every_numeric_parameter_has_a_slider_and_hard_limits_hold_in_the_document() {
    let mut colors = std::collections::BTreeSet::new();
    for family in 0..8 {
        let doc = WorldDocument::from_scene(&crate::scene::Scene::preset(family));
        for node in doc.nodes() {
            for a in doc.attributes(node.id, 0.0).unwrap() {
                let numeric = a.value.is_number()
                    || a.value
                        .as_array()
                        .is_some_and(|v| v.iter().all(|x| x.is_number()));
                if numeric && a.choices.is_empty() {
                    assert!(a.slider.is_some(), "{} has no slider span", a.path);
                    if let (Some(s), Some((min, max))) = (a.slider, a.range) {
                        assert!(
                            min <= s.min && s.max <= max,
                            "{}: slider outside its hard limits",
                            a.path
                        );
                    }
                }
                if a.color && a.component.is_none() {
                    colors.insert(a.path.clone());
                }
            }
        }
    }
    // Every RGB slot `Scene::pack` writes with put_rgb that the presets expose.
    assert!(colors.contains("/lighting/sky_horizon") && colors.contains("/material/base_tint"));
    assert!(colors.len() >= 10, "{colors:?}");

    let mut e = WorldEditor::new(WorldDocument::from_scene(&crate::scene::Scene::preset(0)));
    let material = e
        .document
        .nodes()
        .into_iter()
        .find(|n| n.kind == WorldKind::Material)
        .unwrap()
        .id;
    e.execute(WorldCommand::SetAttribute {
        id: material,
        path: "/material/transmission".into(),
        value: serde_json::json!(2.5),
        frame: 0.0,
    })
    .unwrap();
    let value = e
        .document
        .attribute_value(material, "/material/transmission", 0.0)
        .unwrap();
    assert_eq!(
        value.as_f64(),
        Some(1.0),
        "hard limit [0, 1] clamps any editor's value"
    );
}

#[test]
fn inactive_parameters_follow_what_the_kernel_reads() {
    use serde_json::json;
    let doc = |formula: Value, coloring: &str| {
        let values = [
            ("/formula".to_owned(), formula),
            ("/coloring".to_owned(), json!(coloring)),
        ];
        move |path: &str| {
            inactive_reason(path, |p| {
                values.iter().find(|(k, _)| k == p).map(|(_, v)| v)
            })
        }
    };
    let bulb = doc(json!({"Mandelbulb": {}}), "Radius");
    assert!(bulb("/julia").is_none());
    assert!(
        bulb("/trap_scale").is_some() && bulb("/trap_point/1").is_some(),
        "Radius reads no trap"
    );
    let kifs = doc(json!({"Kifs": {}}), "TrapPoint");
    assert_eq!(
        kifs("/julia").as_deref(),
        Some("The Kifs formula has no Julia mode")
    );
    assert!(kifs("/trap_point").is_none() && kifs("/trap_scale").is_none());
    assert!(kifs("/trap_axis").is_some(), "a point trap has no axis");
    let origin = doc(json!({"Kifs": {}}), "TrapOrigin");
    assert!(origin("/trap_scale").is_none() && origin("/trap_point").is_some());
    let plane = doc(json!({"Kifs": {}}), "TrapPlane");
    assert!(plane("/trap_axis").is_none() && plane("/trap_point/0").is_none());

    let hybrid = doc(
        json!({"Hybrid": {"steps": ["Mandelbox", "Off", "Off", "Off"]}}),
        "Radius",
    );
    assert!(hybrid("/formula/Hybrid/mandelbox/scale").is_none());
    assert!(hybrid("/formula/Hybrid/kifs/scale").is_some());
    assert!(hybrid("/formula/Hybrid/bulb/power").is_some());
    assert!(
        hybrid("/formula/Hybrid/bulb/rotation_degrees/2").is_none(),
        "turns the whole hybrid"
    );
    assert!(hybrid("/formula/Hybrid/apollonian_scale").is_some());
    assert!(hybrid("/formula/Hybrid/bailout").is_none());
    let idle = doc(
        json!({"Hybrid": {"steps": ["Off", "Off", "Off", "Off"]}}),
        "Radius",
    );
    assert!(
        idle("/formula/Hybrid/bulb/power").is_none(),
        "no step on runs a Mandelbulb step"
    );
}

#[test]
fn hybrid_marks_unpacked_sub_formula_parameters_and_limits_keep_their_floor() {
    use serde_json::json;
    let formula = json!({"Hybrid": {"steps": ["Mandelbox", "KifsFold", "Mandelbulb", "Off"]}});
    let value = |p: &str| (p == "/formula").then_some(&formula);
    for path in [
        "/formula/Hybrid/mandelbox/rotation_degrees/1",
        "/formula/Hybrid/kifs/rotation_degrees",
        "/formula/Hybrid/bulb/bailout",
    ] {
        assert!(
            inactive_reason(path, value).is_some(),
            "{path} is never packed"
        );
    }
    assert!(inactive_reason("/formula/Hybrid/mandelbox/scale", value).is_none());
    assert_eq!(
        numeric_like(&json!(0), 0.0005),
        json!(0.0005),
        "a float floor never rounds away"
    );
    assert_eq!(numeric_like(&json!(9), 2.0), json!(2));
}

/// Every attribute the Attribute Editor shows (any family, any node kind) explains itself on
/// hover: a new parameter without a line in `attribute_hint` fails here.
#[test]
fn every_attribute_has_a_hover_hint() {
    let mut missing = std::collections::BTreeSet::new();
    // Optional attributes appear only when set: a facing blend and a Julia constant.
    let mut optional = Scene::preset(crate::params::FAMILY_BULB);
    optional.julia = Some([0.1, 0.2, 0.3]);
    optional.material.facing = Some(crate::scene::Facing {
        color: [1.0, 0.5, 0.2],
        roughness: 0.3,
        metallic: 0.5,
        exponent: 2.0,
    });
    for scene in (0..8).map(Scene::preset).chain([optional]) {
        let doc = WorldDocument::from_scene(&scene);
        for node in doc.nodes() {
            for attr in doc.attributes(node.id, 0.0).unwrap() {
                if attribute_hint(&attr.path).is_none() {
                    missing.insert(attr.path);
                }
            }
        }
    }
    assert!(missing.is_empty(), "attributes without a hint: {missing:?}");
    // A channel row explains its vector; the Hybrid's sub-formulas share their family's text.
    assert_eq!(
        attribute_hint("/camera/target/1"),
        attribute_hint("/camera/target")
    );
    assert_eq!(
        attribute_hint("/formula/Hybrid/bulb/power"),
        attribute_hint("/formula/Mandelbulb/power")
    );
    assert_ne!(
        attribute_hint("/formula/Apollonian/scale"),
        attribute_hint("/formula/Mandelbox/scale")
    );
}
#[test]
fn attributes_carry_the_fresh_world_value_as_their_reset_default() {
    let mut e = editor();
    let camera = find(&e, WorldKind::Camera);
    let pick = |e: &WorldEditor, path: &str| {
        e.document
            .attributes(camera, 0.0)
            .unwrap()
            .into_iter()
            .find(|a| a.path == path)
            .unwrap()
    };
    let fresh = pick(&e, "/camera/distance").value;
    let target = pick(&e, "/camera/target").value;
    set(&mut e, camera, "/camera/distance", json!(7.5), 0.0);
    set(&mut e, camera, "/camera/target/0", json!(99.0), 0.0);
    let distance = pick(&e, "/camera/distance");
    assert_eq!(distance.value, json!(7.5));
    assert_eq!(distance.default, Some(fresh));
    // A component's default is its element of the vector's default.
    assert_eq!(
        pick(&e, "/camera/target/0").default,
        Some(target[0].clone())
    );
    // Locks and time range are not resettable parameters.
    assert_eq!(pick(&e, "/locked").default, None);
}

#[test]
fn camera_recording_replays_visible_pose_with_animated_nonuniform_camera_scale() {
    use crate::camera_recorder::{Options, Recording};
    let mut e = editor();
    let id = find(&e, WorldKind::Camera);
    for (frame, scale) in [(0.0, [1.0, 1.0, 1.0]), (24.0, [2.0, 0.8, 1.4])] {
        e.document
            .set_attribute(
                id,
                "/transform/scale",
                json!(scale),
                frame,
                true,
                Tan::Linear,
            )
            .unwrap();
    }
    let start = e.document.snapshot(0.0).unwrap();
    let mut recording =
        Recording::new(&e.document, e.revision(), 0.0, &start, Options::default()).unwrap();
    let mut expected = vec![(0.0, start.camera)];
    for index in 1..=4 {
        let seconds = f64::from(index) / 4.0;
        let mut camera = start.camera;
        camera.target[0] += index as f32 * 0.05;
        camera.yaw_degrees += index as f32 * 2.0;
        recording.capture(seconds, camera).unwrap();
        expected.push((seconds * e.document.fps, camera));
    }
    e.record_camera(&recording).unwrap();
    for (frame, expected) in expected {
        let actual = e.document.snapshot(frame).unwrap().camera;
        let position_error =
            (glam::Vec3::from(actual.target) - glam::Vec3::from(expected.target)).length();
        assert!(
            position_error < 0.001,
            "frame {frame}: target error {position_error}"
        );
        assert!(
            (actual.distance - expected.distance).abs() < 0.001,
            "frame {frame}: distance"
        );
        assert!(
            actual.orientation().dot(expected.orientation()).abs() > 0.99999,
            "frame {frame}: orientation"
        );
    }
}

#[test]
fn snapshot_and_gpu_upload_reject_invalid_lens_payload_without_repair() {
    let mut e = editor();
    let camera = find(&e, WorldKind::Camera);
    // Malformed persisted authoring values must reach the evaluated lens boundary.
    // The gpu template is overridden by these authoritative host attributes.
    let mut attrs = e.document.attrs(camera).unwrap();
    attrs.set("/camera/sensor_height", AttrValue::Float(-0.024));
    e.document.store_attrs(camera, &attrs).unwrap();
    let before = e.document.clone();
    assert!(
        e.document
            .snapshot(0.0)
            .unwrap_err()
            .contains("sensor height")
    );
    assert_eq!(e.document, before);
    let mut scene = Scene::preset(0);
    scene.world_render = true;
    scene.camera.f_number = -1.0;
    assert!(
        crate::render::validate_world(&scene)
            .unwrap_err()
            .contains("f-number")
    );
}

#[test]
fn current_world_format_roundtrips_but_old_and_future_graph_versions_are_rejected() {
    let mut scene = Scene::preset(0);
    scene.animation = crate::animation::Animation {
        first: 17,
        last: 91,
        fps: 24000.0 / 1001.0,
    };
    let original = WorldDocument::from_scene(&scene);
    let json = serde_json::to_string(&original).unwrap();
    let decoded = crate::io_service::decode_scene(&json).unwrap();
    assert_eq!(decoded.animation, scene.animation);
    let loaded: WorldDocument = serde_json::from_str(&json).unwrap();
    assert_eq!(loaded, original);
    assert!(loaded.graph.bus_slots["world"].get("animation").is_none());
    assert_eq!(loaded.snapshot(37.5).unwrap().animation, Default::default());
    for version in [
        0,
        playa_graph::SUBNET_FORMAT_VERSION - 1,
        playa_graph::SUBNET_FORMAT_VERSION + 1,
    ] {
        let mut document = original.clone();
        document.graph.format_version = version;
        let serialized = serde_json::to_string(&document).unwrap();
        assert!(
            crate::io_service::decode_scene(&serialized)
                .unwrap_err()
                .contains("Unsupported world graph version")
        );
        assert_eq!(
            document.graph.format_version, version,
            "decoding never repairs a document"
        );
    }
}

#[test]
fn render_profiles_route_auto_locked_and_output_without_mutating_document() {
    use crate::render_profiles::{ProfileTarget, RenderMethod, ViewportMode};
    let mut e = editor();
    let saved = e.document.clone();
    let output = e.document.output_render_profile().unwrap();
    let policy = e.document.viewport_policy(0.0).unwrap();
    let moving = e.document.viewport_render(0.0, true).unwrap();
    let still = e.document.viewport_render(0.0, false).unwrap();
    assert_eq!(
        (moving.target, still.target),
        (ProfileTarget::Moving, ProfileTarget::Still)
    );
    let (moving, still) = (moving.effective, still.effective);
    assert_eq!(moving.profile, policy.moving_id);
    assert_eq!(moving.method, RenderMethod::Fast);
    assert_eq!(
        (
            moving.samples,
            moving.resolution_scale,
            moving.render.max_bounces
        ),
        (64, 0.5, 2)
    );
    assert_eq!(still.profile, policy.still_id);
    assert_eq!(still.method, RenderMethod::Full);
    assert_ne!(output, policy.moving_id);
    assert_ne!(output, policy.still_id);
    assert_eq!(
        e.document, saved,
        "routing is transient and authors no data"
    );
    let viewport = e.document.viewport_settings_id().unwrap();
    set(
        &mut e,
        viewport,
        "/viewport/mode",
        json!(ViewportMode::Locked),
        0.0,
    );
    set(
        &mut e,
        viewport,
        "/viewport/manual_id",
        json!(policy.moving_id),
        0.0,
    );
    for moving in [true, false] {
        let locked = e.document.viewport_render(0.0, moving).unwrap();
        assert_eq!(
            (locked.target, locked.effective.profile),
            (ProfileTarget::Manual, policy.moving_id)
        );
    }
    let exported = e.document.snapshot(0.0).unwrap();
    assert_eq!(exported.render, saved.snapshot(0.0).unwrap().render);
    assert!(
        !e.document.graph.bus_slots["world"]
            .as_object()
            .unwrap()
            .contains_key("render")
    );
    assert!(!e.document.graph.bus_slots.contains_key("settings_node"));
}

#[test]
fn render_template_instantiation_remaps_pair_and_undoes_assignment_atomically() {
    use crate::render_profiles::{CatalogRole, ProfileTarget};
    let mut e = editor();
    e.execute(WorldCommand::CreateRenderProfile {
        name: "Studio".into(),
        role: CatalogRole::Template,
        source: None,
        target: None,
    })
    .unwrap();
    let template = e.selection.unwrap();
    let template_quality = e.document.render_quality(template).unwrap();
    assert_eq!(
        e.document.catalog_role(template_quality).unwrap(),
        Some(CatalogRole::Template)
    );
    set(
        &mut e,
        template_quality,
        "/quality/samples",
        json!(123),
        0.0,
    );
    let before = e.document.clone();
    let count = before.nodes().len();
    e.execute(WorldCommand::InstantiateRenderTemplate {
        id: template,
        name: "Studio still".into(),
        target: Some(ProfileTarget::Still),
    })
    .unwrap();
    let instance = e.selection.unwrap();
    let quality = e.document.render_quality(instance).unwrap();
    assert_ne!(instance, template);
    assert_ne!(quality, template_quality);
    assert_eq!(e.document.nodes().len(), count + 2);
    assert_eq!(e.document.viewport_policy(0.0).unwrap().still_id, instance);
    assert_eq!(
        e.document.effective_render(instance, 0.0).unwrap().samples,
        123
    );
    let applied = e.document.clone();
    assert!(e.undo());
    assert_eq!(e.document, before);
    assert!(e.redo());
    assert_eq!(e.document, applied);
    set(
        &mut e,
        template_quality,
        "/quality/samples",
        json!(777),
        0.0,
    );
    assert_eq!(
        e.document.effective_render(instance, 0.0).unwrap().samples,
        123
    );
    assert_eq!(
        e.document.effective_render(template, 0.0).unwrap().samples,
        777
    );
}

#[test]
fn live_render_profile_edits_reach_all_uuid_consumers_and_roundtrip() {
    use crate::render_profiles::ProfileTarget;
    let mut e = editor();
    let policy = e.document.viewport_policy(0.0).unwrap();
    let viewport = e.document.viewport_settings_id().unwrap();
    let output = e.document.output_render_profile().unwrap();
    set(
        &mut e,
        viewport,
        ProfileTarget::Still.viewport_path().unwrap(),
        json!(policy.moving_id),
        0.0,
    );
    e.execute(WorldCommand::SetOutputRender(policy.moving_id))
        .unwrap();
    let quality = e.document.render_quality(policy.moving_id).unwrap();
    set(&mut e, quality, "/quality/max_bounces", json!(5), 0.0);
    for moving in [true, false] {
        assert_eq!(
            e.document
                .viewport_render(0.0, moving)
                .unwrap()
                .effective
                .render
                .max_bounces,
            5
        );
    }
    assert_eq!(e.document.snapshot(0.0).unwrap().render.max_bounces, 5);
    let encoded = serde_json::to_string(&e.document).unwrap();
    let restored: WorldDocument = serde_json::from_str(&encoded).unwrap();
    restored.validate_render_profiles(0.0).unwrap();
    assert_eq!(restored, e.document);
    assert_eq!(restored.render_quality(policy.moving_id).unwrap(), quality);
    assert_ne!(output, restored.output_render_profile().unwrap());
}

#[test]
fn profile_reference_validation_rejects_missing_wrong_template_and_animated_refs() {
    use crate::render_profiles::CatalogRole;
    let mut e = editor();
    let viewport = e.document.viewport_settings_id().unwrap();
    let fractal = e.selection.unwrap();
    let saved = e.document.clone();
    for target in [fractal, NodeId::new()] {
        assert!(
            e.execute(WorldCommand::SetAttribute {
                id: viewport,
                path: "/viewport/moving_id".into(),
                value: json!(target),
                frame: 0.0,
            })
            .is_err()
        );
        assert_eq!(e.document, saved);
    }
    e.execute(WorldCommand::CreateRenderProfile {
        name: "Draft".into(),
        role: CatalogRole::Template,
        source: None,
        target: None,
    })
    .unwrap();
    let template = e.selection.unwrap();
    let before = e.document.clone();
    assert!(e.execute(WorldCommand::SetOutputRender(template)).is_err());
    assert_eq!(e.document, before);
    assert!(
        e.execute(WorldCommand::SetAttribute {
            id: viewport,
            path: "/viewport/manual_id".into(),
            value: json!(template),
            frame: 0.0
        })
        .is_err()
    );
    assert!(
        e.execute(WorldCommand::Key {
            id: viewport,
            path: "/viewport/still_id".into(),
            frame: 0.0
        })
        .is_err()
    );
    assert_eq!(e.document, before);
    let mut invalid = saved;
    invalid
        .graph
        .bus_slots
        .insert("output_render".into(), json!(NodeId::new()));
    assert!(
        invalid.snapshot(0.0).is_err(),
        "load/evaluation never repairs missing refs"
    );
}

#[test]
fn assigned_profiles_guard_delete_and_explicit_reassignment_allows_atomic_delete() {
    let mut e = editor();
    let output = e.document.output_render_profile().unwrap();
    let quality = e.document.render_quality(output).unwrap();
    let still = e.document.viewport_policy(0.0).unwrap().still_id;
    let before = e.document.clone();
    assert!(e.execute(WorldCommand::Delete(output)).is_err());
    assert!(e.execute(WorldCommand::Delete(quality)).is_err());
    assert_eq!(e.document, before);
    e.execute(WorldCommand::Batch(vec![
        WorldCommand::SetOutputRender(still),
        WorldCommand::Delete(output),
        WorldCommand::Delete(quality),
    ]))
    .unwrap();
    assert!(!e.document.graph.nodes.contains_key(&output.to_string()));
    assert!(!e.document.graph.nodes.contains_key(&quality.to_string()));
    assert!(e.undo());
    assert_eq!(e.document, before);
}

#[test]
fn standalone_quality_templates_are_independent_atomic_instances() {
    use crate::render_profiles::CatalogRole;
    let mut e = editor();
    let output = e.document.output_render_profile().unwrap();
    e.execute(WorldCommand::CreateQualityProfile {
        name: "Draft quality".into(),
        role: CatalogRole::Template,
        source: None,
        target: None,
    })
    .unwrap();
    let template = e.selection.unwrap();
    set(&mut e, template, "/quality/samples", json!(31), 0.0);
    let before = e.document.clone();
    e.execute(WorldCommand::InstantiateQualityTemplate {
        id: template,
        name: "Delivery quality".into(),
        target: Some(output),
    })
    .unwrap();
    let quality = e.selection.unwrap();
    assert_ne!(quality, template);
    assert_eq!(e.document.render_quality(output).unwrap(), quality);
    assert_eq!(
        e.document.effective_render(output, 0.0).unwrap().samples,
        31
    );
    let applied = e.document.clone();
    assert!(e.undo());
    assert_eq!(e.document, before);
    assert!(e.redo());
    assert_eq!(e.document, applied);
    set(&mut e, template, "/quality/samples", json!(59), 0.0);
    assert_eq!(
        e.document.effective_render(output, 0.0).unwrap().samples,
        31
    );
}

#[test]
fn profile_fast_materials_are_evaluated_only_and_preserve_transmission_and_geometry() {
    use crate::scene::MaterialModel;
    let mut scene = Scene::preset(0);
    scene.material.model = MaterialModel::StandardSurface;
    scene.material.transmission = 0.0;
    let mut e = WorldEditor::new(WorldDocument::from_scene(&scene));
    let fractal = e.selection.unwrap();
    let material = e.document.assigned_material(fractal).unwrap().unwrap();
    let policy = e.document.viewport_policy(0.0).unwrap();
    let saved = e.document.clone();
    let fast = e
        .document
        .snapshot_with_render_profile(policy.moving_id, 0.0)
        .unwrap();
    assert_eq!(fast.objects[0].material.model, MaterialModel::Fast);
    assert_eq!(fast.objects[0].render.iterations, scene.render.iterations);
    assert_eq!(fast.objects[0].render.max_bounces, 2);
    let full = e
        .document
        .snapshot_with_render_profile(policy.still_id, 0.0)
        .unwrap();
    assert_eq!(
        full.objects[0].material.model,
        MaterialModel::StandardSurface
    );
    assert_eq!(e.document, saved);
    e.execute(WorldCommand::SetOutputRender(policy.moving_id))
        .unwrap();
    assert_eq!(
        e.document.snapshot(0.0).unwrap().objects[0].material.model,
        MaterialModel::Fast
    );
    assert_eq!(
        e.document
            .snapshot_with_render_profile(policy.still_id, 0.0)
            .unwrap()
            .objects[0]
            .material
            .model,
        MaterialModel::StandardSurface
    );
    set(&mut e, material, "/material/transmission", json!(0.75), 0.0);
    assert_eq!(
        e.document
            .snapshot_with_render_profile(policy.moving_id, 0.0)
            .unwrap()
            .objects[0]
            .material
            .model,
        MaterialModel::StandardSurface
    );
    assert_eq!(
        e.document.material(material, 0.0).unwrap().model,
        MaterialModel::StandardSurface
    );
}

#[test]
fn canonical_settings_validation_rejects_missing_fields_and_invalid_unused_templates() {
    use crate::render_profiles::CatalogRole;
    let mut e = editor();
    e.execute(WorldCommand::CreateQualityProfile {
        name: "Unused".into(),
        role: CatalogRole::Template,
        source: None,
        target: None,
    })
    .unwrap();
    let template = e.selection.unwrap();
    let mut broken = e.document.clone();
    let mut attrs = broken.attrs(template).unwrap();
    attrs.set("/quality/samples", to_attr(&json!("not samples")));
    broken.store_attrs(template, &attrs).unwrap();
    assert!(
        broken.snapshot(0.0).is_err(),
        "unused templates obey the same schema"
    );
    let mut missing = e.document;
    let mut attrs = missing.attrs(template).unwrap();
    attrs.remove("/quality/step_factor");
    missing.store_attrs(template, &attrs).unwrap();
    assert!(
        missing.snapshot(0.0).is_err(),
        "GPU schema defaults never replace missing authored fields"
    );
}

#[test]
fn persisted_settings_references_reject_animation_and_connections_instead_of_sampling_first() {
    let e = editor();
    let render = e.document.output_render_profile().unwrap();
    let first_quality = e.document.render_quality(render).unwrap();
    let still = e.document.viewport_policy(0.0).unwrap().still_id;
    let later_quality = e.document.render_quality(still).unwrap();
    assert_ne!(first_quality, later_quality);
    let mut animated = e.document.clone();
    let path = "/render/quality_id";
    let mut attrs = animated.attrs(render).unwrap();
    // Discrete animation channels evaluate numeric dictionary indices, not UUID strings.
    attrs.set(path, AttrValue::Float(0.0));
    let mut animation = Animation::with_arity(1);
    animation.channels[0].upsert_key(Keyframe::with_tan(0.0, 0.0, Tan::Constant));
    animation.channels[0].upsert_key(Keyframe::with_tan(10.0, 1.0, Tan::Constant));
    attrs.set_anim(path, Some(animation));
    animated.store_attrs(render, &attrs).unwrap();
    animated.node_mut(render).unwrap()["discrete"][path] = json!([first_quality, later_quality]);
    assert_eq!(
        animated.attribute_value(render, path, 0.0).unwrap(),
        json!(first_quality)
    );
    assert_eq!(
        animated.attribute_value(render, path, 20.0).unwrap(),
        json!(later_quality)
    );
    let restored: WorldDocument =
        serde_json::from_str(&serde_json::to_string(&animated).unwrap()).unwrap();
    assert!(restored.render_quality(render).is_err());
    assert!(restored.effective_render(render, 20.0).is_err());
    assert!(restored.snapshot(0.0).is_err());
    let mut connected = e.document;
    let mut attrs = connected.attrs(render).unwrap();
    attrs.set_conn(
        path,
        Some(playa_engine::entities::attrs::AttrConnection {
            source_layer: still.0,
            source_key: path.into(),
        }),
    );
    connected.store_attrs(render, &attrs).unwrap();
    let restored: WorldDocument =
        serde_json::from_str(&serde_json::to_string(&connected).unwrap()).unwrap();
    assert!(restored.snapshot(0.0).is_err());
    assert!(restored.render_quality(render).is_err());
}

#[test]
fn viewport_policy_is_static_in_commands_and_persisted_graphs() {
    let mut e = editor();
    let viewport = e.document.viewport_settings_id().unwrap();
    let original = e.document.clone();
    for descriptor in e
        .document
        .attributes(viewport, 0.0)
        .unwrap()
        .into_iter()
        .filter(|attr| attr.path.starts_with("/viewport/"))
    {
        assert!(!descriptor.keyable);
        assert!(
            e.execute(WorldCommand::Key {
                id: viewport,
                path: descriptor.path,
                frame: 10.0
            })
            .is_err()
        );
        assert_eq!(e.document, original);
    }
    let path = "/viewport/target_fps";
    let mut animated = original.clone();
    let mut attrs = animated.attrs(viewport).unwrap();
    let mut animation = Animation::with_arity(1);
    animation.channels[0].upsert_key(Keyframe::with_tan(0.0, 30.0, Tan::Constant));
    attrs.set_anim(path, Some(animation));
    animated.store_attrs(viewport, &attrs).unwrap();
    let restored: WorldDocument =
        serde_json::from_str(&serde_json::to_string(&animated).unwrap()).unwrap();
    assert!(restored.snapshot(0.0).is_err());
    assert!(restored.viewport_policy(0.0).is_err());
    let mut connected = original;
    let mut attrs = connected.attrs(viewport).unwrap();
    attrs.set_conn(
        path,
        Some(playa_engine::entities::attrs::AttrConnection {
            source_layer: viewport.0,
            source_key: "/viewport/batch_budget_ms".into(),
        }),
    );
    connected.store_attrs(viewport, &attrs).unwrap();
    let restored: WorldDocument =
        serde_json::from_str(&serde_json::to_string(&connected).unwrap()).unwrap();
    assert!(restored.snapshot(0.0).is_err());
    assert!(restored.viewport_policy(0.0).is_err());
}

#[test]
fn output_settings_node_is_the_typed_static_export_recipe() {
    use crate::export::{ExportFormat, OutputSettings};
    use crate::render_profiles::CatalogRole;
    let mut e = editor();
    let recipe = e.document.output_settings_id().unwrap();
    assert_eq!(
        e.document.output_settings().unwrap(),
        OutputSettings::default()
    );
    // Every field is a static row: no keys, enums as choices, numbers inside their limits.
    let original = e.document.clone();
    for attr in e
        .document
        .attributes(recipe, 0.0)
        .unwrap()
        .into_iter()
        .filter(|attr| attr.path.starts_with("/output/"))
    {
        assert!(!attr.keyable, "{} must be static", attr.path);
        assert!(
            e.execute(WorldCommand::Key {
                id: recipe,
                path: attr.path.clone(),
                frame: 5.0
            })
            .is_err()
        );
        if attr.path == "/output/format" {
            assert_eq!(attr.choices.len(), ExportFormat::ALL.len());
        }
    }
    assert_eq!(e.document, original);
    // The panel's edits: one command per changed field, and the document reads them back.
    let png = OutputSettings {
        format: ExportFormat::Png,
        png: egui_display::export::PngEncoding::Hdr10,
        width: 640,
        ..OutputSettings::default()
    };
    let edits =
        crate::render_profiles::output_edits(recipe, &OutputSettings::default(), &png).unwrap();
    assert_eq!(edits.len(), 3);
    e.execute(WorldCommand::Batch(edits)).unwrap();
    assert_eq!(e.document.output_settings().unwrap(), png);
    // Hard limits hold on edit; a cross-field rule waits for the export start.
    set(&mut e, recipe, "/output/width", json!(99999), 0.0);
    assert_eq!(e.document.output_settings().unwrap().width, 16384);
    set(&mut e, recipe, "/output/width", json!(17), 0.0);
    set(
        &mut e,
        recipe,
        "/output/format",
        json!(ExportFormat::Hevc),
        0.0,
    );
    let odd = e.document.output_settings().unwrap();
    assert!(odd.validate().is_err(), "HEVC needs even sizes");
    // Catalog: a bound copy is one Undo step with a new UUID; templates must be instantiated.
    let before = e.document.clone();
    e.execute(WorldCommand::CreateOutputSettings {
        name: "Review".into(),
        role: CatalogRole::Profile,
        source: None,
        assign: true,
    })
    .unwrap();
    let copy = e.document.output_settings_id().unwrap();
    assert_ne!(copy, recipe);
    assert_eq!(e.document.output_settings().unwrap(), odd);
    assert!(e.undo());
    assert_eq!(e.document, before);
    e.execute(WorldCommand::CreateOutputSettings {
        name: "Delivery".into(),
        role: CatalogRole::Template,
        source: None,
        assign: false,
    })
    .unwrap();
    let template = e.selection.unwrap();
    assert!(
        e.execute(WorldCommand::SetOutputSettings(template))
            .is_err()
    );
    assert!(
        e.execute(WorldCommand::CreateOutputSettings {
            name: "Bound template".into(),
            role: CatalogRole::Template,
            source: None,
            assign: true,
        })
        .is_err()
    );
    e.execute(WorldCommand::InstantiateOutputTemplate {
        id: template,
        name: "Delivery shot".into(),
        assign: true,
    })
    .unwrap();
    let instance = e.document.output_settings_id().unwrap();
    assert_ne!(instance, template);
    assert_eq!(
        e.document.catalog_role(instance).unwrap(),
        Some(CatalogRole::Profile)
    );
    // The bound recipe cannot be deleted out from under the export.
    assert!(e.execute(WorldCommand::Delete(instance)).is_err());
    // A persisted animation on a recipe field is refused on load.
    let mut animated = e.document.clone();
    let mut attrs = animated.attrs(instance).unwrap();
    let mut animation = Animation::with_arity(1);
    animation.channels[0].upsert_key(Keyframe::with_tan(0.0, 18.0, Tan::Constant));
    attrs.set_anim("/output/qp", Some(animation));
    animated.store_attrs(instance, &attrs).unwrap();
    let restored: WorldDocument =
        serde_json::from_str(&serde_json::to_string(&animated).unwrap()).unwrap();
    assert!(restored.snapshot(0.0).is_err());
}
