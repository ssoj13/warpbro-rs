//! Camera takes: every viewport navigation gesture is recorded as a take on the navigation clock.
//!
//! A [`Recorder`] watches the viewport camera frame by frame and cuts a [`Take`] from the first
//! move until the input has ended and the camera has come to rest. Finished takes live in the
//! document's [`Takes`] store (saved with the scene, outside undo/redo and outside document
//! equality: a take is a recording, not an edit). The store can bind one take as *current*; the
//! document then resolves the active camera from it (`WorldDocument::snapshot`, the single pose
//! hook shared by viewport, playback cache and export). "Bake to keys" turns a take into camera
//! keyframes through `camera_recorder::Recording`.
use crate::camera_recorder::{Sample, Timing};
use crate::scene::Camera;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Unpinned takes kept; the oldest unpinned take beyond this is dropped.
pub const MAX_UNPINNED: usize = 20;
/// Version of the single-take clip file (`Take::to_clip`).
pub const CLIP_VERSION: u32 = 1;
/// A gesture ends once no input is held and the camera has not moved for this long. It also
/// merges the separate ticks of a mouse wheel into one take.
const SETTLE_SECS: f64 = 0.3;
/// Shorter gestures (a Home / Fit jump, a stray click) are not worth a take.
const MIN_SECS: f64 = 0.1;
/// A runaway gesture is cut here (about ten minutes at 60 Hz) and the next move starts a new take.
const MAX_SAMPLES: usize = 36_000;

/// One recorded navigation gesture.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Take {
    /// Stable key within the store; 0 until the take is added.
    pub id: u64,
    pub name: String,
    /// Document frame the gesture began on: where a bound take starts playing.
    pub start: f64,
    pub timing: Timing,
    /// Pinned takes are never dropped by retention.
    pub pinned: bool,
    /// Strictly increasing seconds from 0, every pose the visible camera had.
    pub samples: Vec<Sample>,
}

impl Take {
    pub fn duration(&self) -> f64 {
        self.samples.last().map_or(0.0, |s| s.seconds)
    }

    /// Reject samples that cannot drive a camera: too few, unordered or non-finite.
    pub fn checked(self) -> Result<Self, String> {
        let bad = |what: &str| Err(format!("Invalid take \"{}\": {what}", self.name));
        if self.samples.len() < 2 {
            return bad("needs at least two samples");
        }
        if !self.start.is_finite() {
            return bad("start frame is not finite");
        }
        if self.samples.first().is_some_and(|s| s.seconds != 0.0) {
            return bad("time must start at 0");
        }
        if self.samples.windows(2).any(|w| {
            w[0].seconds.partial_cmp(&w[1].seconds) != Some(std::cmp::Ordering::Less)
                || !w[1].seconds.is_finite()
        }) {
            return bad("sample times must increase");
        }
        if self.samples.iter().any(|s| !camera_finite(&s.camera)) {
            return bad("camera values must be finite");
        }
        Ok(self)
    }

    /// Seconds into the take that document `frame` shows: wall speed from `start` at the document
    /// rate (`KeepSpeed`) or the whole take stretched over the work area (`FitWorkArea`). The same
    /// mapping as the bake (`Recording::prepare`).
    pub fn seconds_at(&self, frame: f64, first: u32, last: u32, fps: f64) -> f64 {
        match self.timing {
            Timing::KeepSpeed => (frame - self.start) / fps,
            Timing::FitWorkArea => {
                let span = f64::from(last) - f64::from(first);
                if span > 0.0 {
                    (frame - f64::from(first)) / span * self.duration()
                } else {
                    0.0
                }
            }
        }
    }

    /// The camera at `seconds`: rotation by quaternion slerp (no Euler wrap jumps), target and
    /// lens linearly. Held at the first / last pose outside the take.
    pub fn pose_at(&self, seconds: f64) -> Option<Camera> {
        let (first, last) = (self.samples.first()?, self.samples.last()?);
        if seconds <= first.seconds {
            return Some(first.camera);
        }
        if seconds >= last.seconds {
            return Some(last.camera);
        }
        let hi = self.samples.partition_point(|s| s.seconds <= seconds);
        let a = self.samples.get(hi.checked_sub(1)?)?;
        let b = self.samples.get(hi)?;
        let t = ((seconds - a.seconds) / (b.seconds - a.seconds)) as f32;
        Some(blend(&a.camera, &b.camera, t))
    }

    /// The camera a document at (`first`..`last`, `fps`) shows on `frame`.
    pub fn pose_at_frame(&self, frame: f64, first: u32, last: u32, fps: f64) -> Option<Camera> {
        self.pose_at(self.seconds_at(frame, first, last, fps))
    }

    /// Versioned single-take clip file for exchange between scenes.
    pub fn to_clip(&self, fps: f64) -> Result<String, String> {
        serde_json::to_string_pretty(&Clip {
            version: CLIP_VERSION,
            fps,
            name: self.name.clone(),
            start: self.start,
            timing: self.timing,
            samples: self.samples.clone(),
        })
        .map_err(|e| e.to_string())
    }

    /// Parse a clip file; the take is unpinned and has no id until a store adds it.
    pub fn from_clip(text: &str) -> Result<Self, String> {
        let clip: Clip =
            serde_json::from_str(text).map_err(|e| format!("Invalid take clip: {e}"))?;
        if clip.version != CLIP_VERSION {
            return Err(format!(
                "Unsupported take clip version {} (expected {CLIP_VERSION})",
                clip.version
            ));
        }
        Take {
            id: 0,
            name: clip.name,
            start: clip.start,
            timing: clip.timing,
            pinned: false,
            samples: clip.samples,
        }
        .checked()
    }
}

/// The on-disk single-take exchange format.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Clip {
    version: u32,
    /// Document rate the take was recorded under (informational: samples are in seconds).
    fps: f64,
    name: String,
    start: f64,
    timing: Timing,
    samples: Vec<Sample>,
}

fn camera_finite(c: &Camera) -> bool {
    c.target.iter().all(|v| v.is_finite())
        && [
            c.yaw_degrees,
            c.pitch_degrees,
            c.roll_degrees,
            c.distance,
            c.fov_y_degrees,
            c.f_number,
            c.sensor_height,
            c.focus_distance,
        ]
        .iter()
        .all(|v| v.is_finite())
}

/// Interpolate two poses; an unchanged orientation keeps its exact Euler angles.
fn blend(a: &Camera, b: &Camera, t: f32) -> Camera {
    let lerp = |x: f32, y: f32| x + (y - x) * t;
    let mut c = *a;
    c.target = std::array::from_fn(|i| lerp(a.target[i], b.target[i]));
    c.distance = lerp(a.distance, b.distance);
    c.fov_y_degrees = lerp(a.fov_y_degrees, b.fov_y_degrees);
    c.f_number = lerp(a.f_number, b.f_number);
    c.sensor_height = lerp(a.sensor_height, b.sensor_height);
    c.focus_distance = lerp(a.focus_distance, b.focus_distance);
    if (a.yaw_degrees, a.pitch_degrees, a.roll_degrees)
        != (b.yaw_degrees, b.pitch_degrees, b.roll_degrees)
    {
        let q = a.orientation().slerp(b.orientation(), t);
        let (yaw, pitch, roll) = q.to_euler(glam::EulerRot::YXZ);
        c.yaw_degrees = yaw.to_degrees();
        c.pitch_degrees = -pitch.to_degrees();
        c.roll_degrees = roll.to_degrees();
    }
    c
}

/// The takes of one document plus the binding of the current one.
#[derive(Clone, Debug, Default)]
pub struct Takes {
    /// Oldest first. `Arc` keeps document clones (undo snapshots, export jobs) cheap.
    list: Vec<Arc<Take>>,
    current: Option<u64>,
    /// Next id: ids are never reused, so a stale reference cannot hit another take.
    next: u64,
}

impl Takes {
    pub fn is_empty(&self) -> bool {
        self.list.is_empty() && self.current.is_none() && self.next == 0
    }
    pub fn list(&self) -> &[Arc<Take>] {
        &self.list
    }
    pub fn get(&self, id: u64) -> Option<&Arc<Take>> {
        self.list.iter().find(|t| t.id == id)
    }
    pub fn current_id(&self) -> Option<u64> {
        self.current
    }
    /// The take that drives the active camera, if one is bound.
    pub fn current(&self) -> Option<&Take> {
        self.current.and_then(|id| self.get(id)).map(Arc::as_ref)
    }

    /// Add a take (named "Take N" when unnamed) and apply retention. Returns its id.
    pub fn add(&mut self, mut take: Take) -> u64 {
        self.next += 1;
        take.id = self.next;
        if take.name.trim().is_empty() {
            take.name = format!("Take {}", take.id);
        }
        self.list.push(Arc::new(take));
        self.trim();
        self.next
    }

    /// Drop the oldest unpinned takes beyond [`MAX_UNPINNED`]. The bound take is kept too: dropping
    /// it would silently change the camera of the scene.
    fn trim(&mut self) {
        while self.list.iter().filter(|t| !t.pinned).count() > MAX_UNPINNED {
            let Some(index) = self
                .list
                .iter()
                .position(|t| !t.pinned && Some(t.id) != self.current)
            else {
                break;
            };
            self.list.remove(index);
        }
    }

    /// Remove a take; unbinds it when current.
    pub fn remove(&mut self, id: u64) -> bool {
        let before = self.list.len();
        self.list.retain(|t| t.id != id);
        if self.current == Some(id) {
            self.current = None;
        }
        self.list.len() != before
    }

    pub fn set_current(&mut self, id: Option<u64>) -> Result<(), String> {
        if let Some(id) = id
            && self.get(id).is_none()
        {
            return Err(format!("No take {id}"));
        }
        self.current = id;
        Ok(())
    }

    /// Mutate one take in place; false when it does not exist.
    pub fn edit(&mut self, id: u64, apply: impl FnOnce(&mut Take)) -> bool {
        let Some(take) = self.list.iter_mut().find(|t| t.id == id) else {
            return false;
        };
        apply(Arc::make_mut(take));
        self.trim();
        true
    }
}

#[derive(Serialize)]
struct TakesOut<'a> {
    list: Vec<&'a Take>,
    current: Option<u64>,
    next: u64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TakesIn {
    list: Vec<Take>,
    current: Option<u64>,
    next: u64,
}
impl Serialize for Takes {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        TakesOut {
            list: self.list.iter().map(Arc::as_ref).collect(),
            current: self.current,
            next: self.next,
        }
        .serialize(serializer)
    }
}
impl<'de> Deserialize<'de> for Takes {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error;
        let input = TakesIn::deserialize(deserializer)?;
        let mut list: Vec<Arc<Take>> = Vec::with_capacity(input.list.len());
        for take in input.list {
            let take = take.checked().map_err(D::Error::custom)?;
            if take.id == 0 || take.id > input.next || list.iter().any(|t| t.id == take.id) {
                return Err(D::Error::custom(format!("Invalid take id {}", take.id)));
            }
            list.push(Arc::new(take));
        }
        if input
            .current
            .is_some_and(|id| list.iter().all(|t| t.id != id))
        {
            return Err(D::Error::custom("The current take does not exist"));
        }
        Ok(Self {
            list,
            current: input.current,
            next: input.next,
        })
    }
}

/// What the recorder needs from one UI frame.
pub struct Tick {
    /// Navigation clock step: the dt the flight and orbit rigs integrate with.
    pub dt: f64,
    /// The viewport camera before / after this frame's navigation.
    pub before: Camera,
    pub after: Camera,
    /// Document frame under the playhead (where a new take starts).
    pub frame: f64,
    /// Navigation input is still held (button down, rig coasting).
    pub held: bool,
}

/// What a tick changed.
#[derive(Default)]
pub struct Update {
    /// A gesture began this frame.
    pub started: bool,
    /// A gesture ended and produced this take.
    pub finished: Option<Take>,
}

struct Live {
    take: Take,
    clock: f64,
    idle: f64,
    /// Index of the last sample that differs from its predecessor: the take ends there.
    moved: usize,
}

/// Cuts navigation gestures into takes; one instance per app.
#[derive(Default)]
pub struct Recorder {
    live: Option<Live>,
}

impl Recorder {
    /// Abandon a gesture in progress (the document was replaced).
    pub fn cancel(&mut self) {
        self.live = None;
    }

    /// A gesture is being recorded.
    pub fn recording(&self) -> bool {
        self.live.is_some()
    }

    pub fn tick(&mut self, t: &Tick) -> Update {
        let moved = !still(&t.before, &t.after);
        let mut update = Update::default();
        let Some(live) = &mut self.live else {
            if moved {
                self.live = Some(Live {
                    take: Take {
                        id: 0,
                        name: String::new(),
                        start: t.frame,
                        timing: Timing::KeepSpeed,
                        pinned: false,
                        samples: vec![
                            Sample {
                                seconds: 0.0,
                                camera: t.before,
                            },
                            Sample {
                                seconds: t.dt,
                                camera: t.after,
                            },
                        ],
                    },
                    clock: t.dt,
                    idle: 0.0,
                    moved: 1,
                });
                update.started = true;
            }
            return update;
        };
        live.clock += t.dt;
        live.take.samples.push(Sample {
            seconds: live.clock,
            camera: t.after,
        });
        if moved {
            live.idle = 0.0;
            live.moved = live.take.samples.len() - 1;
        } else {
            live.idle += t.dt;
        }
        if (!t.held && live.idle >= SETTLE_SECS) || live.take.samples.len() >= MAX_SAMPLES {
            update.finished = self.finish();
        }
        update
    }

    /// End the gesture now, dropping the still tail. None when it was too short to keep.
    fn finish(&mut self) -> Option<Take> {
        let live = self.live.take()?;
        let mut take = live.take;
        take.samples.truncate(live.moved + 1);
        (take.duration() >= MIN_SECS).then_some(take)
    }
}

/// Rest test with a small epsilon: relative to the view distance for positions (deep zooms move
/// by tiny absolute amounts), a fraction of a degree for angles.
fn still(a: &Camera, b: &Camera) -> bool {
    let eps = 1.0e-6 * a.distance.abs().max(f32::MIN_POSITIVE);
    let near = |x: f32, y: f32, tol: f32| (x - y).abs() <= tol;
    a.free_flight == b.free_flight
        && a.target
            .iter()
            .zip(&b.target)
            .all(|(x, y)| near(*x, *y, eps))
        && near(a.distance, b.distance, eps)
        && near(a.yaw_degrees, b.yaw_degrees, 1.0e-4)
        && near(a.pitch_degrees, b.pitch_degrees, 1.0e-4)
        && near(a.roll_degrees, b.roll_degrees, 1.0e-4)
        && near(a.fov_y_degrees, b.fov_y_degrees, 1.0e-4)
        && near(a.f_number, b.f_number, 1.0e-4)
        && near(a.sensor_height, b.sensor_height, 0.0)
        && near(a.focus_distance, b.focus_distance, eps)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cam(yaw: f32) -> Camera {
        Camera {
            yaw_degrees: yaw,
            ..Camera::default_fov(50.0)
        }
    }
    fn take(yaws: &[f32]) -> Take {
        Take {
            id: 0,
            name: String::new(),
            start: 10.0,
            timing: Timing::KeepSpeed,
            pinned: false,
            samples: yaws
                .iter()
                .enumerate()
                .map(|(i, y)| Sample {
                    seconds: i as f64,
                    camera: cam(*y),
                })
                .collect(),
        }
    }

    #[test]
    fn pose_interpolates_rotation_by_slerp_across_the_wrap_and_holds_outside() {
        let t = take(&[170.0, -170.0]);
        let mid = t.pose_at(0.5).unwrap();
        // Shortest arc passes through 180 degrees, never through 0.
        assert!(mid.orientation().dot(cam(180.0).orientation()).abs() > 0.99999);
        assert_eq!(t.pose_at(-1.0).unwrap(), cam(170.0));
        assert_eq!(t.pose_at(9.0).unwrap(), cam(-170.0));
        let mut moved = take(&[0.0, 0.0]);
        moved.samples[1].camera.distance = 4.0;
        moved.samples[0].camera.distance = 2.0;
        assert_eq!(moved.pose_at(0.5).unwrap().distance, 3.0);
    }

    #[test]
    fn frame_mapping_keeps_speed_or_fits_the_work_area() {
        let mut t = take(&[0.0, 90.0]);
        // KeepSpeed: one second at 24 fps after the start frame.
        assert_eq!(t.seconds_at(34.0, 0, 239, 24.0), 1.0);
        assert_eq!(t.seconds_at(22.0, 0, 239, 24.0), 0.5);
        t.timing = Timing::FitWorkArea;
        assert_eq!(t.seconds_at(239.0, 0, 239, 24.0), 1.0);
        assert!((t.seconds_at(119.5, 0, 239, 24.0) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn retention_keeps_pinned_and_current_and_drops_the_oldest_unpinned() {
        let mut takes = Takes::default();
        let first = takes.add(take(&[0.0, 1.0]));
        let second = takes.add(take(&[0.0, 2.0]));
        assert!(takes.edit(first, |t| t.pinned = true));
        takes.set_current(Some(second)).unwrap();
        for i in 0..(MAX_UNPINNED + 5) {
            takes.add(take(&[0.0, 3.0 + i as f32]));
        }
        assert!(takes.get(first).is_some(), "pinned take survives");
        assert!(takes.get(second).is_some(), "bound take survives");
        assert_eq!(
            takes.list().iter().filter(|t| !t.pinned).count(),
            MAX_UNPINNED
        );
        // The six oldest droppable takes (ids 3..=8) went first.
        assert_eq!(takes.list()[2].name, "Take 9");
        assert!(takes.remove(second));
        assert_eq!(takes.current_id(), None);
        assert!(takes.set_current(Some(second)).is_err());
    }

    #[test]
    fn takes_roundtrip_through_json_and_reject_inconsistent_data() {
        let mut takes = Takes::default();
        let id = takes.add(take(&[0.0, 45.0, 90.0]));
        takes.set_current(Some(id)).unwrap();
        let json = serde_json::to_string(&takes).unwrap();
        let back: Takes = serde_json::from_str(&json).unwrap();
        assert_eq!(back.current(), takes.current());
        assert_eq!(back.list().len(), 1);
        let broken = json.replace("\"current\":1", "\"current\":7");
        assert!(serde_json::from_str::<Takes>(&broken).is_err());
        let one_sample = serde_json::to_string(&TakesOut {
            list: vec![&take(&[0.0])],
            current: None,
            next: 1,
        })
        .unwrap();
        assert!(serde_json::from_str::<Takes>(&one_sample).is_err());
    }

    #[test]
    fn clip_roundtrips_and_rejects_other_versions() {
        let mut t = take(&[0.0, 30.0]);
        t.name = "Fly-by".into();
        let text = t.to_clip(24.0).unwrap();
        let back = Take::from_clip(&text).unwrap();
        assert_eq!(
            (back.name.as_str(), back.id, back.pinned),
            ("Fly-by", 0, false)
        );
        assert_eq!(back.samples, t.samples);
        let future = text.replace("\"version\": 1", "\"version\": 2");
        assert!(Take::from_clip(&future).unwrap_err().contains("version"));
        assert!(Take::from_clip("{}").is_err());
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
    fn a_gesture_runs_from_the_first_move_through_the_coast_and_trims_the_still_tail() {
        let mut r = Recorder::default();
        let mut yaw = 0.0;
        // Idle frames never start a take.
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
        // Released: input is gone but the camera has to settle first.
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
        assert_eq!(take.start, 7.0);
        assert_eq!(
            take.samples.len(),
            31,
            "one initial pose + 30 moves; tail trimmed"
        );
        assert_eq!(take.samples[0].camera, cam(0.0));
        assert_eq!(take.samples[30].camera, cam(30.0));
        assert!((take.duration() - 30.0 / 60.0).abs() < 1e-9);
        // The recorder is free for the next gesture.
        assert!(tick(&mut r, cam(30.0), cam(31.0), true).started);
    }

    #[test]
    fn held_input_keeps_a_still_gesture_alive_and_tiny_jumps_are_discarded() {
        let mut r = Recorder::default();
        tick(&mut r, cam(0.0), cam(1.0), true);
        for _ in 0..120 {
            assert!(tick(&mut r, cam(1.0), cam(1.0), true).finished.is_none());
        }
        // A single-frame jump (Home / Fit) is shorter than the minimum.
        let mut r = Recorder::default();
        tick(&mut r, cam(0.0), cam(90.0), false);
        let mut finished = None;
        for _ in 0..30 {
            finished = finished.or(tick(&mut r, cam(90.0), cam(90.0), false).finished);
        }
        assert!(finished.is_none());
    }
}
