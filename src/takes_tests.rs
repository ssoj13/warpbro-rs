use super::*;

fn cam(yaw: f32) -> Camera {
    Camera {
        yaw_degrees: yaw,
        ..Camera::default_fov(50.0)
    }
}
fn take(yaws: &[f32]) -> Take {
    let samples: Vec<Sample<Camera>> = yaws
        .iter()
        .enumerate()
        .map(|(i, y)| Sample {
            seconds: i as f64,
            pose: cam(*y),
        })
        .collect();
    Take::new(10.0, None, samples).unwrap()
}
fn trs(z: f32, x: f32) -> Trs {
    Trs {
        position: [x, 0.0, 0.0],
        rotation_degrees: [0.0, 0.0, z],
        scale: [1.0; 3],
    }
}
fn node(i: u8) -> NodeId {
    NodeId::parse(&format!("00000000-0000-4000-8000-0000000000{i:02x}")).unwrap()
}

#[test]
fn camera_poses_interpolate_rotation_by_slerp_across_the_wrap_and_hold_outside() {
    let t = take(&[170.0, -170.0]);
    let mid = t.camera_at(10.0 + 0.5 * 24.0, 0, 239, 24.0).unwrap();
    // Shortest arc through 180, written nearest the first pose (180, not -180).
    assert!(
        (mid.yaw_degrees - 180.0).abs() < 1e-3,
        "{}",
        mid.yaw_degrees
    );
    assert_eq!(t.camera_at(0.0, 0, 239, 24.0).unwrap(), cam(170.0));
    assert_eq!(t.camera_at(1000.0, 0, 239, 24.0).unwrap(), cam(-170.0));
    assert!(
        t.trs_at(10.0, 0, 239, 24.0).is_none(),
        "a camera take drives no transform"
    );
}

#[test]
fn transform_poses_interpolate_and_keep_rotation_continuous() {
    let samples = vec![
        Sample {
            seconds: 0.0,
            pose: trs(170.0, 0.0),
        },
        Sample {
            seconds: 1.0,
            pose: trs(-170.0, 2.0),
        },
    ];
    let t = Take::new(0.0, Some(node(1)), samples).unwrap();
    assert_eq!(t.kind(), TakeKind::Transform);
    let mid = t.trs_at(12.0, 0, 239, 24.0).unwrap();
    assert!((mid.position[0] - 1.0).abs() < 1e-6);
    assert!(
        (mid.rotation_degrees[2] - 180.0).abs() < 1e-3,
        "{:?}",
        mid.rotation_degrees
    );
    assert!(t.camera_at(12.0, 0, 239, 24.0).is_none());
}

#[test]
fn frame_mapping_keeps_speed_or_fits_the_work_area() {
    let mut t = take(&[0.0, 90.0]);
    assert_eq!(t.seconds_at(34.0, 0, 239, 24.0), 1.0);
    assert_eq!(t.seconds_at(22.0, 0, 239, 24.0), 0.5);
    t.timing = Timing::FitWorkArea;
    assert_eq!(t.seconds_at(239.0, 0, 239, 24.0), 1.0);
    assert!((t.seconds_at(119.5, 0, 239, 24.0) - 0.5).abs() < 1e-9);
}

#[test]
fn retention_keeps_pinned_takes_and_the_takes_nodes_play() {
    let mut takes = Takes::default();
    let first = takes.add(take(&[0.0, 1.0]), &[]);
    let second = takes.add(take(&[0.0, 2.0]), &[]);
    assert!(takes.edit(first, &[], |t| t.pinned = true));
    for i in 0..(MAX_UNPINNED + 5) {
        takes.add(take(&[0.0, 3.0 + i as f32]), &[second]);
    }
    assert!(takes.get(first).is_some(), "pinned take survives");
    assert!(takes.get(second).is_some(), "a played take survives");
    assert_eq!(
        takes.list().iter().filter(|t| !t.pinned).count(),
        MAX_UNPINNED
    );
    assert_eq!(takes.list()[2].name, "Take 9");
    assert!(takes.remove(first));
    assert!(!takes.remove(first));
}

#[test]
fn takes_roundtrip_through_json_and_reject_inconsistent_data() {
    let mut takes = Takes::default();
    let id = takes.add(take(&[0.0, 45.0, 90.0]), &[]);
    let json = serde_json::to_string(&takes).unwrap();
    let back: Takes = serde_json::from_str(&json).unwrap();
    assert_eq!(back.get(id), takes.get(id));
    assert_eq!(back.list().len(), 1);
    let future = json.replace("\"next\":1", "\"next\":0");
    assert_ne!(future, json);
    assert!(
        serde_json::from_str::<Takes>(&future).is_err(),
        "an id beyond next"
    );
}

#[test]
fn clip_and_clipboard_roundtrip_both_kinds_and_reject_other_versions() {
    let mut t = take(&[0.0, 30.0]);
    t.name = "Fly-by".into();
    t.source = Some(node(4));
    let text = t.to_clip(24.0).unwrap();
    let back = Take::from_clip(&text).unwrap();
    assert_eq!(
        (back.name.as_str(), back.id, back.pinned, back.source),
        ("Fly-by", 0, false, None)
    );
    assert_eq!(back.poses, t.poses);
    let future = text.replace("\"version\": 2", "\"version\": 3");
    assert!(Take::from_clip(&future).unwrap_err().contains("version"));
    assert!(Take::from_clip("{}").is_err());
    let moved = Take::still(trs(10.0, 1.0), 5.0, 24.0, Some(node(5))).unwrap();
    let text = to_clipboard(&[&t, &moved], 24.0).unwrap();
    let back = from_clipboard(&text).unwrap().unwrap();
    assert_eq!(
        back.iter().map(Take::kind).collect::<Vec<_>>(),
        vec![TakeKind::Camera, TakeKind::Transform]
    );
    assert!(from_clipboard("{\"warpbro_nodes\":1}").is_none());
    assert!(from_clipboard("plain text").is_none());
}

#[test]
fn trim_split_retime_and_smooth_work_on_either_kind() {
    let t = take(&[0.0, 10.0, 20.0, 30.0, 40.0]);
    let cut = t.trimmed(1.5, 3.0, 24.0).unwrap();
    assert_eq!(cut.duration(), 1.5);
    assert_eq!(cut.start, 10.0 + 1.5 * 24.0);
    assert!(t.trimmed(3.0, 3.0, 24.0).is_err());
    let (head, tail) = t.split(1.0, 24.0).unwrap();
    assert_eq!((head.duration(), tail.duration()), (1.0, 3.0));
    assert_eq!(t.retimed(2.0).unwrap().duration(), 2.0);
    assert!(t.retimed(0.0).is_err());
    let jitter: Vec<Sample<Trs>> = (0..41)
        .map(|i| Sample {
            seconds: i as f64,
            pose: trs(0.0, i as f32 + if i % 2 == 0 { 0.5 } else { -0.5 }),
        })
        .collect();
    let shaky = Take::new(0.0, None, jitter).unwrap();
    let smooth = shaky
        .effected(&curves::Effect::Smooth {
            cutoff: 0.1,
            order: 4,
            blend: 0.0,
        })
        .unwrap();
    let roughness = |t: &Take| match &t.poses {
        Poses::Transform(s) => s[5..36]
            .windows(3)
            .map(|w| {
                (w[0].pose.position[0] + w[2].pose.position[0] - 2.0 * w[1].pose.position[0]).abs()
            })
            .fold(0.0f32, f32::max),
        Poses::Camera(_) => unreachable!(),
    };
    assert!(roughness(&smooth) < 0.25 * roughness(&shaky));
}

fn tick(recorder: &mut Recorder, before: Camera, after: Camera, held: bool) -> Update {
    recorder.tick(&Tick {
        dt: 1.0 / 60.0,
        before,
        after,
        frame: 7.0,
        held,
    })
}

#[test]
fn a_navigation_gesture_runs_through_the_coast_and_trims_the_still_tail() {
    let mut r = Recorder::default();
    let mut yaw = 0.0;
    assert!(!tick(&mut r, cam(0.0), cam(0.0), false).started);
    let mut started = false;
    for _ in 0..30 {
        let next = yaw + 1.0;
        let up = tick(&mut r, cam(yaw), cam(next), true);
        started |= up.started;
        assert!(up.finished.is_none());
        yaw = next;
    }
    assert!(started);
    let mut finished = None;
    for i in 0..60 {
        let up = tick(&mut r, cam(yaw), cam(yaw), false);
        if up.finished.is_some() {
            assert!(i >= 17, "ended after the 0.3 s settle, not before");
            finished = up.finished;
            break;
        }
    }
    let take = finished.expect("take finished");
    assert_eq!(
        (take.start, take.len(), take.kind()),
        (7.0, 31, TakeKind::Camera)
    );
    assert!((take.duration() - 30.0 / 60.0).abs() < 1e-9);
    assert!(tick(&mut r, cam(30.0), cam(31.0), true).started);
    // A single-frame jump (Home / Fit) is shorter than the minimum.
    let mut r = Recorder::default();
    tick(&mut r, cam(0.0), cam(90.0), false);
    let mut finished = None;
    for _ in 0..30 {
        finished = finished.or(tick(&mut r, cam(90.0), cam(90.0), false).finished);
    }
    assert!(finished.is_none());
}

#[test]
fn a_drag_records_one_transform_take_per_moved_node() {
    let mut d = DragRecorder::default();
    let (a, b) = (node(1), node(2));
    for i in 0..30 {
        // Node a moves after a still head; node b never moves.
        let x = if i < 5 { 0.0 } else { (i - 5) as f32 };
        d.sample(1.0 / 60.0, 12.0, &[(a, trs(0.0, x)), (b, trs(0.0, 0.0))]);
    }
    let takes = d.finish();
    assert_eq!(takes.len(), 1, "a still node makes no take");
    let t = &takes[0];
    assert_eq!(
        (t.source, t.start, t.kind()),
        (Some(a), 12.0, TakeKind::Transform)
    );
    assert!(
        (t.duration() - 24.0 / 60.0).abs() < 1e-9,
        "{}",
        t.duration()
    );
    assert!(d.finish().is_empty(), "finish clears the drag");
}

#[test]
fn the_euler_filter_and_simplify_edit_a_takes_motion_like_keys() {
    // A turn written with a decomposer's wrap at +-180: 170, 180, -170, -160.
    let turn: Vec<Sample<Trs>> = [170.0, 180.0, -170.0, -160.0]
        .into_iter()
        .enumerate()
        .map(|(i, z)| Sample {
            seconds: i as f64,
            pose: Trs {
                position: [i as f32, 0.0, 0.0],
                rotation_degrees: [0.0, 0.0, z],
                scale: [1.0; 3],
            },
        })
        .collect();
    let take = Take::new(0.0, None, turn).unwrap();
    let filtered = take
        .effected(&curves::Effect::EulerFilter { middle: None })
        .unwrap();
    let Poses::Transform(s) = &filtered.poses else {
        unreachable!()
    };
    let z: Vec<f32> = s.iter().map(|x| x.pose.rotation_degrees[2]).collect();
    assert!(
        z.windows(2).all(|w| (w[1] - w[0] - 10.0).abs() < 1e-3),
        "{z:?}"
    );
    // Smoothing a wrapped turn keeps it a 30 degree turn, never a spin through 360.
    let smooth = take
        .effected(&curves::Effect::Smooth {
            cutoff: 0.2,
            order: 4,
            blend: 0.0,
        })
        .unwrap();
    let Poses::Transform(s) = &smooth.poses else {
        unreachable!()
    };
    let z: Vec<f32> = s.iter().map(|x| x.pose.rotation_degrees[2]).collect();
    assert!(z.windows(2).all(|w| (w[1] - w[0]).abs() < 20.0), "{z:?}");
    // Simplify keeps the samples, now on a curve within the error of the old one.
    let simple = take
        .effected(&curves::Effect::Simplify {
            tolerance: 0.01,
            max_keys: None,
        })
        .unwrap();
    assert_eq!(simple.len(), take.len());
}
