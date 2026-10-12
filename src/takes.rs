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
/// Marker of WarpBro takes on the system clipboard, so other text is never pasted as takes.
const CLIPBOARD_KEY: &str = "warpbro_takes";

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
        serde_json::to_string_pretty(&Clip::of(self, fps)).map_err(|e| e.to_string())
    }

    /// Parse a clip file; the take is unpinned and has no id until a store adds it.
    pub fn from_clip(text: &str) -> Result<Self, String> {
        let clip: Clip =
            serde_json::from_str(text).map_err(|e| format!("Invalid take clip: {e}"))?;
        clip.take()
    }
}

impl Take {
    /// A still take of `camera` from frame `start`: a camera bookmark (two equal poses one
    /// frame apart, the least a take holds).
    pub fn still(camera: Camera, start: f64, fps: f64) -> Result<Self, String> {
        Self::from_frames(start, fps, vec![camera, camera])
    }

    /// A take through `poses`, one per frame from `start` at `fps`: the timeline animation of a
    /// camera as a take.
    pub fn from_frames(start: f64, fps: f64, poses: Vec<Camera>) -> Result<Self, String> {
        if !(fps.is_finite() && fps > 0.0) {
            return Err("Invalid frame rate".into());
        }
        Take {
            id: 0,
            name: String::new(),
            start,
            timing: Timing::KeepSpeed,
            pinned: false,
            samples: poses
                .into_iter()
                .enumerate()
                .map(|(i, camera)| Sample {
                    seconds: i as f64 / fps,
                    camera,
                })
                .collect(),
        }
        .checked()
    }

    /// The part between `from` and `to` seconds, rebased to 0. The cut poses are interpolated,
    /// and the start frame moves with the cut (`from` seconds at `fps`), so the kept motion
    /// still plays on the frames it played on.
    pub fn trimmed(&self, from: f64, to: f64, fps: f64) -> Result<Self, String> {
        let from = from.max(0.0);
        let to = to.min(self.duration());
        if !(to > from) {
            return Err("Trim keeps nothing".into());
        }
        let first = self.pose_at(from).ok_or("Empty take")?;
        let last = self.pose_at(to).ok_or("Empty take")?;
        let mut samples = vec![Sample {
            seconds: 0.0,
            camera: first,
        }];
        samples.extend(
            self.samples
                .iter()
                .filter(|s| s.seconds > from && s.seconds < to)
                .map(|s| Sample {
                    seconds: s.seconds - from,
                    camera: s.camera,
                }),
        );
        samples.push(Sample {
            seconds: to - from,
            camera: last,
        });
        Take {
            start: self.start + from * fps,
            samples,
            ..self.clone()
        }
        .checked()
    }

    /// Played `speed` times as fast (2 = half the duration).
    pub fn retimed(&self, speed: f64) -> Result<Self, String> {
        if !(speed.is_finite() && speed > 0.0) {
            return Err("Speed must be positive".into());
        }
        let mut take = self.clone();
        for sample in &mut take.samples {
            sample.seconds /= speed;
        }
        take.checked()
    }

    /// Hand shake removed: every pose becomes the mean of the poses within `window` seconds
    /// around it (rotation by the normalized mean of hemisphere-aligned quaternions, accurate
    /// for the small angles one window spans). The first and last poses stay.
    pub fn smoothed(&self, window: f64) -> Self {
        let mut take = self.clone();
        let half = window.max(0.0) * 0.5;
        let n = self.samples.len();
        if half <= 0.0 || n < 3 {
            return take;
        }
        for (i, out) in take.samples.iter_mut().enumerate().take(n - 1).skip(1) {
            let t = self.samples[i].seconds;
            let lo = self.samples.partition_point(|s| s.seconds < t - half);
            let hi = self.samples.partition_point(|s| s.seconds <= t + half);
            out.camera = mean(&self.samples[lo..hi], &self.samples[i].camera);
        }
        take
    }

    /// Cut at `seconds` into two takes; the second starts on the frame the cut played on.
    pub fn split(&self, seconds: f64, fps: f64) -> Result<(Self, Self), String> {
        let head = self.trimmed(0.0, seconds, fps)?;
        let tail = self.trimmed(seconds, self.duration(), fps)?;
        Ok((head, tail))
    }
}

/// The mean pose of `samples` (non-empty); discrete fields come from `like`.
fn mean(samples: &[Sample], like: &Camera) -> Camera {
    let n = samples.len().max(1) as f32;
    let avg = |f: &dyn Fn(&Camera) -> f32| samples.iter().map(|s| f(&s.camera)).sum::<f32>() / n;
    let reference = like.orientation();
    let q = samples
        .iter()
        .map(|s| {
            let q = s.camera.orientation();
            if q.dot(reference) < 0.0 { -q } else { q }
        })
        .fold(glam::Quat::from_xyzw(0.0, 0.0, 0.0, 0.0), |a, q| a + q)
        .normalize();
    let mut out = Camera {
        target: std::array::from_fn(|k| avg(&|c| c.target[k])),
        distance: avg(&|c| c.distance),
        fov_y_degrees: avg(&|c| c.fov_y_degrees),
        f_number: avg(&|c| c.f_number),
        sensor_height: avg(&|c| c.sensor_height),
        focus_distance: avg(&|c| c.focus_distance),
        ..*like
    };
    set_orientation(&mut out, q, like);
    out
}

/// Write the rotation `q` into `camera`'s yaw / pitch / roll (Euler YXZ) as the angles nearest
/// `near`'s by whole turns: no flip at +-180 between neighbouring poses, the turns of a yaw past
/// +-180 kept (`curves::nearest_euler`, the Euler filter's rule).
fn set_orientation(camera: &mut Camera, q: glam::Quat, near: &Camera) {
    let (yaw, pitch, roll) = q.to_euler(glam::EulerRot::YXZ);
    let [yaw, pitch, roll] = curves::nearest_euler(
        [yaw.to_degrees(), -pitch.to_degrees(), roll.to_degrees()],
        [near.yaw_degrees, near.pitch_degrees, near.roll_degrees],
        // Whole turns only: the pitch is limited to +-90, the second solution would leave it.
        None,
    );
    (
        camera.yaw_degrees,
        camera.pitch_degrees,
        camera.roll_degrees,
    ) = (yaw, pitch, roll);
}

/// Takes on the system clipboard: the clip format, many at once.
pub fn to_clipboard(takes: &[&Take], fps: f64) -> Result<String, String> {
    let clips: Vec<Clip> = takes.iter().map(|t| Clip::of(t, fps)).collect();
    serde_json::to_string(&serde_json::json!({ CLIPBOARD_KEY: CLIP_VERSION, "clips": clips }))
        .map_err(|e| e.to_string())
}

/// The takes of a [`to_clipboard`] text; None for any other text, an error for a WarpBro
/// takes text that does not parse.
pub fn from_clipboard(text: &str) -> Option<Result<Vec<Take>, String>> {
    let value: serde_json::Value = serde_json::from_str(text.trim()).ok()?;
    let version = value.get(CLIPBOARD_KEY)?.as_u64();
    if version != Some(u64::from(CLIP_VERSION)) {
        return Some(Err(format!(
            "Unsupported takes clipboard version {version:?}"
        )));
    }
    Some(
        serde_json::from_value::<Vec<Clip>>(value["clips"].clone())
            .map_err(|e| format!("Invalid takes clipboard: {e}"))
            .and_then(|clips| clips.into_iter().map(Clip::take).collect()),
    )
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

impl Clip {
    fn of(take: &Take, fps: f64) -> Self {
        Clip {
            version: CLIP_VERSION,
            fps,
            name: take.name.clone(),
            start: take.start,
            timing: take.timing,
            samples: take.samples.clone(),
        }
    }

    /// The take of a clip: unpinned, without an id until a store adds it.
    fn take(self) -> Result<Take, String> {
        if self.version != CLIP_VERSION {
            return Err(format!(
                "Unsupported take clip version {} (expected {CLIP_VERSION})",
                self.version
            ));
        }
        Take {
            id: 0,
            name: self.name,
            start: self.start,
            timing: self.timing,
            pinned: false,
            samples: self.samples,
        }
        .checked()
    }
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
        set_orientation(&mut c, a.orientation().slerp(b.orientation(), t), a);
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

    #[test]
    fn trim_split_and_retime_keep_the_motion_where_it_played() {
        // Yaw 0, 10, 20, 30, 40 at 0..4 s from frame 10.
        let t = take(&[0.0, 10.0, 20.0, 30.0, 40.0]);
        let cut = t.trimmed(1.5, 3.0, 24.0).unwrap();
        assert_eq!(cut.duration(), 1.5);
        assert!(
            (cut.samples[0].camera.yaw_degrees - 15.0).abs() < 1e-3,
            "interpolated cut pose"
        );
        assert_eq!(
            cut.start,
            10.0 + 1.5 * 24.0,
            "the kept motion stays on its frames"
        );
        assert!((cut.samples.last().unwrap().camera.yaw_degrees - 30.0).abs() < 1e-3);
        assert!(t.trimmed(3.0, 3.0, 24.0).is_err());
        let (head, tail) = t.split(1.0, 24.0).unwrap();
        assert_eq!((head.duration(), tail.duration()), (1.0, 3.0));
        assert_eq!(tail.start, 10.0 + 24.0);
        assert_eq!(tail.samples[0].camera, head.samples.last().unwrap().camera);
        let fast = t.retimed(2.0).unwrap();
        assert_eq!(fast.duration(), 2.0);
        assert!(t.retimed(0.0).is_err());
    }

    #[test]
    fn smoothing_removes_jitter_and_keeps_the_ends() {
        let yaws: Vec<f32> = (0..41)
            .map(|i| i as f32 + if i % 2 == 0 { 0.5 } else { -0.5 })
            .collect();
        let t = take(&yaws);
        let s = t.smoothed(8.0);
        assert_eq!(s.samples[0].camera, t.samples[0].camera);
        assert_eq!(s.samples[40].camera, t.samples[40].camera);
        let jitter = |k: &Take| {
            k.samples[5..36]
                .windows(3)
                .map(|w| {
                    (w[0].camera.yaw_degrees + w[2].camera.yaw_degrees
                        - 2.0 * w[1].camera.yaw_degrees)
                        .abs()
                })
                .fold(0.0f32, f32::max)
        };
        assert!(
            jitter(&s) < 0.25 * jitter(&t),
            "{} vs {}",
            jitter(&s),
            jitter(&t)
        );
    }

    #[test]
    fn stills_and_frames_make_takes_and_the_clipboard_roundtrips() {
        let still = Take::still(cam(30.0), 5.0, 24.0).unwrap();
        assert_eq!(still.samples.len(), 2);
        assert_eq!(still.pose_at(0.5).unwrap(), cam(30.0));
        let anim = Take::from_frames(0.0, 24.0, vec![cam(0.0), cam(1.0), cam(2.0)]).unwrap();
        assert!((anim.duration() - 2.0 / 24.0).abs() < 1e-12);
        let mut named = take(&[0.0, 9.0]);
        named.name = "Orbit".into();
        let text = to_clipboard(&[&named, &still], 24.0).unwrap();
        let back = from_clipboard(&text).unwrap().unwrap();
        assert_eq!(back.len(), 2);
        assert_eq!((back[0].name.as_str(), back[0].id), ("Orbit", 0));
        assert!(
            from_clipboard("{\"warpbro_nodes\":1}").is_none(),
            "other text is not takes"
        );
        assert!(from_clipboard("plain text").is_none());
        assert!(
            from_clipboard(&text.replace("\"warpbro_takes\":1", "\"warpbro_takes\":9"))
                .unwrap()
                .is_err()
        );
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
