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
            CurveKind::Linear,
        )
        .unwrap();
    e.document
        .set_attribute(
            material,
            "/material/transmission_depth",
            json!(1.5),
            20.0,
            true,
            CurveKind::Linear,
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
fn old_material_schema_gains_glass_controls_without_losing_existing_keys() {
    fn strip_transmission(value: &mut Value) {
        match value {
            Value::Object(map) => {
                map.retain(|key, _| {
                    !key.starts_with("/material/transmission") && !key.starts_with("transmission")
                });
                for value in map.values_mut() {
                    strip_transmission(value);
                }
            }
            Value::Array(values) => {
                for value in values {
                    strip_transmission(value);
                }
            }
            _ => {}
        }
    }
    let mut e = editor();
    let id = find(&e, WorldKind::Material);
    e.document
        .set_attribute(
            id,
            "/material/specular_roughness",
            json!(0.2),
            0.0,
            true,
            CurveKind::Linear,
        )
        .unwrap();
    e.document
        .set_attribute(
            id,
            "/material/specular_roughness",
            json!(0.8),
            20.0,
            true,
            CurveKind::Linear,
        )
        .unwrap();
    let expected = e.document.material(id, 10.0).unwrap();
    let mut old = serde_json::to_value(&e.document).unwrap();
    strip_transmission(&mut old);
    let old: WorldDocument = serde_json::from_value(old).unwrap();
    assert!(
        old.attrs(id)
            .unwrap()
            .get("/material/transmission")
            .is_none()
    );
    let upgraded = WorldEditor::new(old);
    assert_eq!(upgraded.document.material(id, 10.0).unwrap(), expected);
    let attrs = upgraded.document.attributes(id, 10.0).unwrap();
    assert!(
        attrs
            .iter()
            .any(|a| a.path == "/material/transmission_depth")
    );
    assert_eq!(
        attrs
            .iter()
            .find(|a| a.path == "/material/specular_roughness")
            .unwrap()
            .frames,
        vec![0.0, 20.0]
    );
    let once = upgraded.document.clone();
    assert_eq!(WorldEditor::new(once.clone()).document, once);
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
            .map(|k| k.frame)
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
fn legacy_snapshot_bridge_never_generates_camera_keys_and_navigation_noop_is_clean() {
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
fn camera_orbit_defaults_preserve_old_documents_and_camera_pose() {
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
fn legacy_object_rotation_migrates_to_playa_cw_axes_without_changing_pose() {
    let mut scene = Scene::preset(0);
    scene.object.offset = [1.0, 2.0, 3.0];
    scene.object.scale = 2.5;
    scene.object.rotation_degrees = [15.0, 30.0, 45.0];
    scene.key_parameter("/object/rotation_degrees", 0.0);
    scene.object.rotation_degrees = [30.0, 60.0, 90.0];
    scene.key_parameter("/object/rotation_degrees", 10.0);
    let world = WorldDocument::from_scene(&scene);
    for frame in [0.0, 3.5, 10.0] {
        let legacy = scene.evaluated(frame).unwrap();
        let pack = legacy.pack(1, 1);
        let index = crate::params::P_OBJ_AXES;
        let rows: [[f32; 3]; 3] =
            std::array::from_fn(|i| std::array::from_fn(|j| pack[index + 3 * i + j]));
        let rotation = glam::Mat3::from_cols_array_2d(&rows);
        let point = glam::Vec3::new(0.3, 0.7, 1.1);
        let expected =
            glam::Vec3::from(legacy.object.offset) + rotation * point * legacy.object.scale;
        let snapshot = world.snapshot(frame).unwrap();
        let matrix = glam::Mat4::from_cols_array_2d(&snapshot.objects[0].object_world.unwrap());
        assert!((expected - matrix.transform_point3(point)).length() < 1e-5);
    }
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
fn migrates_all_formula_families_without_dropping_parameters() {
    for family in 0..8 {
        let s = Scene::preset(family);
        let world = WorldDocument::from_scene(&s);
        let snapshot = world.snapshot(0.0).unwrap();
        assert_eq!(snapshot.formula, s.formula);
        assert_eq!(snapshot.material, s.material);
        assert_eq!(snapshot.render, s.render);
        assert_eq!(snapshot.camera, s.camera);
        assert_eq!(snapshot.environment, s.environment);
        assert_eq!(snapshot.objects.len(), 1);
        assert_eq!(snapshot.lights.len(), 1);
        assert_eq!(
            world.runtime_graph().unwrap().nodes.len(),
            world.nodes().len()
        );
    }
}
#[test]
fn legacy_structural_and_discrete_keys_hold_until_boundary() {
    let mut s = Scene::preset(0);
    s.key_parameter("/formula", 0.0);
    s.formula = Scene::preset(7).formula;
    s.key_parameter("/formula", 10.25);
    s.key_parameter("/julia", 0.0);
    s.julia = Some([0.3, 0.4, 0.5]);
    s.key_parameter("/julia", 10.25);
    s.key_parameter("/lighting/background", 0.0);
    s.lighting.background = false;
    s.key_parameter("/lighting/background", 10.25);
    let w = WorldDocument::from_scene(&s);
    for frame in [0.0, 5.0, 10.24, 10.25, 12.0] {
        let legacy = s.evaluated(frame).unwrap();
        let world = w.snapshot(frame).unwrap();
        assert_eq!(world.formula, legacy.formula, "frame={frame}");
        assert_eq!(world.julia, legacy.julia);
        assert_eq!(world.lighting.background, legacy.lighting.background);
    }
}
#[test]
fn numeric_component_keys_and_render_keys_share_playa_evaluation() {
    let mut s = Scene::preset(0);
    s.key_parameter("/camera/target", 0.0);
    s.camera.target = [10.0, 20.0, 30.0];
    s.key_parameter("/camera/target", 10.5);
    s.key_parameter("/render/step_factor", 0.0);
    s.render.step_factor = 0.9;
    s.key_parameter("/render/step_factor", 10.5);
    let w = WorldDocument::from_scene(&s);
    for frame in [0.0, 1.25, 5.25, 10.5] {
        let a = s.evaluated(frame).unwrap();
        let b = w.snapshot(frame).unwrap();
        assert_eq!(a.camera.target, b.camera.target);
        assert!((a.render.step_factor - b.render.step_factor).abs() < 1e-6);
    }
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
fn render_schema_filters_fractal_controls_without_dropping_stored_tracks() {
    let mut scene = Scene::preset(0);
    scene.key_parameter("/render/exposure_stops", 0.0);
    scene.render.exposure_stops = 2.5;
    scene.key_parameter("/render/exposure_stops", 12.0);
    let world = WorldDocument::from_scene(&scene);
    let fractal = world
        .nodes()
        .into_iter()
        .find(|n| n.kind == WorldKind::Fractal)
        .unwrap()
        .id;
    let settings = world
        .nodes()
        .into_iter()
        .find(|n| n.name == "World Settings")
        .unwrap()
        .id;
    let descriptors = world.attributes(fractal, 0.0).unwrap();
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
            .attrs(fractal)
            .unwrap()
            .is_animated("/render/exposure_stops")
    );
    for frame in [0.0, 6.0, 12.0] {
        assert_eq!(
            world
                .node_scene(fractal, frame)
                .unwrap()
                .render
                .exposure_stops,
            scene.evaluated(frame).unwrap().render.exposure_stops
        );
        assert_eq!(
            world.snapshot(frame).unwrap().render.exposure_stops,
            scene.evaluated(frame).unwrap().render.exposure_stops
        );
    }
}

#[test]
fn clipboard_copy_paste_and_duplicate_remap_ids_in_one_undo_step() {
    let mut e = editor();
    let fractal = find(&e, WorldKind::Fractal);
    let material = find(&e, WorldKind::Material);
    e.execute(WorldCommand::AssignMaterial { id: fractal, material: Some(material) }).unwrap();
    let count = e.document.nodes().len();

    // Ctrl+D on two nodes: both copied, one undo step, copies selected.
    e.execute(WorldCommand::Duplicate(vec![fractal, material])).unwrap();
    assert_eq!(e.document.nodes().len(), count + 2);
    assert_eq!(e.selected.len(), 2);
    let copy = *e.selected.iter().find(|id| e.document.supports_material(**id)).unwrap();
    let copied_material = e.document.assigned_material(copy).unwrap().unwrap();
    assert_ne!(copied_material, material, "a reference inside the fragment follows the remap");
    assert!(e.selected.contains(&copied_material));
    assert!(e.undo());
    assert_eq!(e.document.nodes().len(), count);

    // Ctrl+C / Ctrl+V: fresh UUIDs; an outside reference that exists is kept.
    let text = e.document.copy_fragment(&[fractal]).unwrap();
    e.execute(WorldCommand::Delete(fractal)).unwrap();
    e.execute(WorldCommand::Paste(text.clone())).unwrap();
    let pasted = e.selection.unwrap();
    assert_ne!(pasted, fractal, "paste never reuses the source UUID");
    assert_eq!(e.document.assigned_material(pasted).unwrap(), Some(material));
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
    assert_eq!(e.document.assigned_material(e.selection.unwrap()).unwrap(), None);

    // Foreign clipboard text is not WarpBro nodes and changes nothing.
    assert!(parse_clipboard("hello").is_none());
    let before = e.document.nodes().len();
    assert!(e.execute(WorldCommand::Paste("{\"nodes\":{}}".into())).is_err());
    assert_eq!(e.document.nodes().len(), before);
}

#[test]
fn fractal_slider_endpoints_pack_finite_and_hybrid_spans_match_standalone() {
    for family in 0..8 {
        let scene = crate::scene::Scene::preset(family);
        let document = WorldDocument::from_scene(&scene);
        let id = document.nodes().into_iter().find(|n| n.kind == WorldKind::Fractal).unwrap().id;
        for attr in document.attributes(id, 0.0).unwrap() {
            if attr.component.is_some() || !(attr.path.starts_with("/formula/") || attr.path == "/julia" || attr.path == "/render/iterations") {
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
                assert!(evaluated.pack(128, 128).iter().all(|v| v.is_finite()), "family {family}: {} = {endpoint}", attr.path);
            }
        }
    }
    for (nested, standalone) in [("bulb", "Mandelbulb"), ("mandelbox", "Mandelbox"), ("kifs", "Kifs")] {
        let prefix = format!("/formula/Hybrid/{nested}/");
        for param in ["power", "scale", "bailout", "offset", "rotation_degrees"] {
            let a = attribute_slider(&format!("{prefix}{param}"));
            let b = attribute_slider(&format!("/formula/{standalone}/{param}"));
            assert_eq!(a.map(|s| (s.min, s.max, s.log)), b.map(|s| (s.min, s.max, s.log)));
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
                    || a.value.as_array().is_some_and(|v| v.iter().all(|x| x.is_number()));
                if numeric && a.choices.is_empty() {
                    assert!(a.slider.is_some(), "{} has no slider span", a.path);
                    if let (Some(s), Some((min, max))) = (a.slider, a.range) {
                        assert!(min <= s.min && s.max <= max, "{}: slider outside its hard limits", a.path);
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
    let value = e.document.attribute_value(material, "/material/transmission", 0.0).unwrap();
    assert_eq!(value.as_f64(), Some(1.0), "hard limit [0, 1] clamps any editor's value");
}

#[test]
fn inactive_parameters_follow_what_the_kernel_reads() {
    use serde_json::json;
    let doc = |formula: Value, coloring: &str| {
        let values = [("/formula".to_owned(), formula), ("/coloring".to_owned(), json!(coloring))];
        move |path: &str| inactive_reason(path, |p| values.iter().find(|(k, _)| k == p).map(|(_, v)| v))
    };
    let bulb = doc(json!({"Mandelbulb": {}}), "Radius");
    assert!(bulb("/julia").is_none());
    assert!(bulb("/trap_scale").is_some() && bulb("/trap_point/1").is_some(), "Radius reads no trap");
    let kifs = doc(json!({"Kifs": {}}), "TrapPoint");
    assert_eq!(kifs("/julia").as_deref(), Some("The Kifs formula has no Julia mode"));
    assert!(kifs("/trap_point").is_none() && kifs("/trap_scale").is_none());
    assert!(kifs("/trap_axis").is_some(), "a point trap has no axis");
    let origin = doc(json!({"Kifs": {}}), "TrapOrigin");
    assert!(origin("/trap_scale").is_none() && origin("/trap_point").is_some());
    let plane = doc(json!({"Kifs": {}}), "TrapPlane");
    assert!(plane("/trap_axis").is_none() && plane("/trap_point/0").is_none());

    let hybrid = doc(json!({"Hybrid": {"steps": ["Mandelbox", "Off", "Off", "Off"]}}), "Radius");
    assert!(hybrid("/formula/Hybrid/mandelbox/scale").is_none());
    assert!(hybrid("/formula/Hybrid/kifs/scale").is_some());
    assert!(hybrid("/formula/Hybrid/bulb/power").is_some());
    assert!(hybrid("/formula/Hybrid/bulb/rotation_degrees/2").is_none(), "turns the whole hybrid");
    assert!(hybrid("/formula/Hybrid/apollonian_scale").is_some());
    assert!(hybrid("/formula/Hybrid/bailout").is_none());
    let idle = doc(json!({"Hybrid": {"steps": ["Off", "Off", "Off", "Off"]}}), "Radius");
    assert!(idle("/formula/Hybrid/bulb/power").is_none(), "no step on runs a Mandelbulb step");
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
        assert!(inactive_reason(path, value).is_some(), "{path} is never packed");
    }
    assert!(inactive_reason("/formula/Hybrid/mandelbox/scale", value).is_none());
    assert_eq!(numeric_like(&json!(0), 0.0005), json!(0.0005), "a float floor never rounds away");
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
    assert_eq!(attribute_hint("/camera/target/1"), attribute_hint("/camera/target"));
    assert_eq!(
        attribute_hint("/formula/Hybrid/bulb/power"),
        attribute_hint("/formula/Mandelbulb/power")
    );
    assert_ne!(
        attribute_hint("/formula/Apollonian/scale"),
        attribute_hint("/formula/Mandelbox/scale")
    );
}
