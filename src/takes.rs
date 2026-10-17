//! Takes: recorded motion of any node, on the clock it was performed at.
//!
//! A take is a list of timed poses of one kind ([`Poses`]): a camera's visible pose (viewport
//! navigation, recorded by [`Recorder`]) or a node's local transform ([`Trs`]: manipulator drags,
//! recorded by [`DragRecorder`]). Every operation on a take (interpolation, trim, speed, curve effects,
//! split) is written once over the [`Pose`] trait. Finished takes live in the document's [`Takes`]
//! store, saved with the scene but outside undo/redo and outside document equality: a take is a
//! recording, not an edit. A node plays a take through its own attributes (`/take/clip`, the
//! take's id; `/take/offset` and `/take/speed`, keyable: the time warp, `WorldDocument::take_frame`),
//! so attaching is an ordinary undoable edit saved with the node. The poses replace the node's
//! keys in `WorldDocument::snapshot` (the active camera) and `world_matrix` (transforms): the
//! single pose hooks shared by viewport, playback cache and export. "Bake to keys" turns a take
//! into keyframes (`camera_recorder`).
use crate::camera_recorder::Timing;
use crate::scene::Camera;
use crate::world::NodeId;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Unpinned takes kept; the oldest unpinned take no node plays beyond this is dropped.
pub const MAX_UNPINNED: usize = 20;
/// Version of the single-take clip file (`Take::to_clip`) and of the clipboard text.
pub const CLIP_VERSION: u32 = 2;
/// A gesture ends once no input is held and the camera has not moved for this long. It also
/// merges the separate ticks of a mouse wheel into one take.
const SETTLE_SECS: f64 = 0.3;
/// Shorter gestures (a Home / Fit jump, a stray click) are not worth a take.
pub const MIN_SECS: f64 = 0.1;
/// A runaway gesture is cut here (about ten minutes at 60 Hz) and the next move starts a new take.
const MAX_SAMPLES: usize = 36_000;
/// Marker of WarpBro takes on the system clipboard, so other text is never pasted as takes.
const CLIPBOARD_KEY: &str = "warpbro_takes";

/// A pose a take records: what interpolation, averaging and validation need.
pub trait Pose: Copy + PartialEq + std::fmt::Debug + Serialize + DeserializeOwned {
    /// All values finite.
    fn finite(&self) -> bool;
    /// The pose `t` of the way from `self` to `other`: rotation by quaternion slerp, written as
    /// the Euler angles nearest `self`'s (no flip at +-180), the rest linearly.
    fn blend(&self, other: &Self, t: f32) -> Self;
    /// The pose's continuous values: the channels a curve effect edits (`Take::effected`).
    fn channels(&self) -> Vec<f32>;
    /// This pose with `values` (as [`Pose::channels`] orders them) written back.
    fn with_channels(&self, values: &[f32]) -> Self;
    /// The first of its three rotation channels and their Euler order's middle
    /// (`curves::Effect::EulerFilter`): the timeline's choice for the same attributes.
    const ROTATION: (usize, Option<usize>);
}

/// One pose at `seconds` of the take's clock.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sample<P> {
    pub seconds: f64,
    pub pose: P,
}

/// A node's local transform (`/transform/*`; rotation in the attribute convention: degrees,
/// Playa's clockwise-positive ZYX).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Trs {
    pub position: [f32; 3],
    pub rotation_degrees: [f32; 3],
    pub scale: [f32; 3],
}

impl Trs {
    /// The rotation as a quaternion: `playa_engine`'s `build_model_matrix` convention.
    fn quat(&self) -> glam::Quat {
        let [x, y, z] = self.rotation_degrees.map(f32::to_radians);
        glam::Quat::from_euler(glam::EulerRot::ZYX, -z, -y, -x)
    }

    /// `q` written as rotation degrees nearest `near` (both Euler solutions, whole turns).
    fn with_quat(mut self, q: glam::Quat, near: [f32; 3]) -> Self {
        let (z, y, x) = q.to_euler(glam::EulerRot::ZYX);
        self.rotation_degrees = curves::nearest_euler(
            [-x.to_degrees(), -y.to_degrees(), -z.to_degrees()],
            near,
            Some(1),
        );
        self
    }
}

impl Pose for Trs {
    fn finite(&self) -> bool {
        [self.position, self.rotation_degrees, self.scale]
            .iter()
            .flatten()
            .all(|v| v.is_finite())
    }
    fn blend(&self, other: &Self, t: f32) -> Self {
        let lerp = |a: [f32; 3], b: [f32; 3]| std::array::from_fn(|i| a[i] + (b[i] - a[i]) * t);
        let out = Trs {
            position: lerp(self.position, other.position),
            scale: lerp(self.scale, other.scale),
            ..*self
        };
        if self.rotation_degrees == other.rotation_degrees {
            return out;
        }
        out.with_quat(self.quat().slerp(other.quat(), t), self.rotation_degrees)
    }
    fn channels(&self) -> Vec<f32> {
        [self.position, self.rotation_degrees, self.scale]
            .into_iter()
            .flatten()
            .collect()
    }
    fn with_channels(&self, values: &[f32]) -> Self {
        let three = |at: usize| std::array::from_fn(|i| values[at + i]);
        Trs {
            position: three(0),
            rotation_degrees: three(3),
            scale: three(6),
        }
    }
    const ROTATION: (usize, Option<usize>) = (3, Some(1));
}

impl Pose for Camera {
    fn finite(&self) -> bool {
        self.target.iter().all(|v| v.is_finite())
            && [
                self.yaw_degrees,
                self.pitch_degrees,
                self.roll_degrees,
                self.distance,
                self.fov_y_degrees,
                self.f_number,
                self.sensor_height,
                self.focus_distance,
            ]
            .iter()
            .all(|v| v.is_finite())
    }
    fn blend(&self, other: &Self, t: f32) -> Self {
        let lerp = |x: f32, y: f32| x + (y - x) * t;
        let mut c = *self;
        c.target = std::array::from_fn(|i| lerp(self.target[i], other.target[i]));
        c.distance = lerp(self.distance, other.distance);
        c.fov_y_degrees = lerp(self.fov_y_degrees, other.fov_y_degrees);
        c.f_number = lerp(self.f_number, other.f_number);
        c.sensor_height = lerp(self.sensor_height, other.sensor_height);
        c.focus_distance = lerp(self.focus_distance, other.focus_distance);
        if (self.yaw_degrees, self.pitch_degrees, self.roll_degrees)
            != (other.yaw_degrees, other.pitch_degrees, other.roll_degrees)
        {
            set_orientation(
                &mut c,
                self.orientation().slerp(other.orientation(), t),
                self,
            );
        }
        c
    }
    fn channels(&self) -> Vec<f32> {
        let mut values = self.target.to_vec();
        values.extend([
            self.yaw_degrees,
            self.pitch_degrees,
            self.roll_degrees,
            self.distance,
            self.fov_y_degrees,
            self.f_number,
            self.sensor_height,
            self.focus_distance,
        ]);
        values
    }
    fn with_channels(&self, v: &[f32]) -> Self {
        Camera {
            target: [v[0], v[1], v[2]],
            yaw_degrees: v[3],
            pitch_degrees: v[4],
            roll_degrees: v[5],
            distance: v[6],
            fov_y_degrees: v[7],
            f_number: v[8],
            sensor_height: v[9],
            focus_distance: v[10],
            ..*self
        }
    }
    // Pitch is limited: whole turns only, as the timeline filters a camera's keys.
    const ROTATION: (usize, Option<usize>) = (3, None);
}

/// Write the rotation `q` into `camera`'s yaw / pitch / roll (Euler YXZ) as the angles nearest
/// `near`'s by whole turns: no flip at +-180 between neighbouring poses, the turns of a yaw past
/// +-180 kept (`curves::nearest_euler`; turns only, the pitch is limited to +-90).
fn set_orientation(camera: &mut Camera, q: glam::Quat, near: &Camera) {
    let (yaw, pitch, roll) = q.to_euler(glam::EulerRot::YXZ);
    let [yaw, pitch, roll] = curves::nearest_euler(
        [yaw.to_degrees(), -pitch.to_degrees(), roll.to_degrees()],
        [near.yaw_degrees, near.pitch_degrees, near.roll_degrees],
        None,
    );
    (
        camera.yaw_degrees,
        camera.pitch_degrees,
        camera.roll_degrees,
    ) = (yaw, pitch, roll);
}

/// The poses of a take, one kind per take.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "samples", rename_all = "snake_case")]
pub enum Poses {
    /// A camera's visible pose and lens (viewport navigation).
    Camera(Vec<Sample<Camera>>),
    /// A node's local transform (manipulator drags, a node's animation).
    Transform(Vec<Sample<Trs>>),
}

/// What a take can drive: a camera, or the transform of a placed node.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TakeKind {
    Camera,
    Transform,
}

impl From<Vec<Sample<Camera>>> for Poses {
    fn from(samples: Vec<Sample<Camera>>) -> Self {
        Self::Camera(samples)
    }
}
impl From<Vec<Sample<Trs>>> for Poses {
    fn from(samples: Vec<Sample<Trs>>) -> Self {
        Self::Transform(samples)
    }
}

/// `$e` over the samples of either kind, its result rewrapped in the same kind.
macro_rules! map_poses {
    ($poses:expr, $s:ident => $e:expr) => {
        match $poses {
            Poses::Camera($s) => Poses::Camera($e),
            Poses::Transform($s) => Poses::Transform($e),
        }
    };
}
/// `$e` over the samples of either kind.
macro_rules! on_poses {
    ($poses:expr, $s:ident => $e:expr) => {
        match $poses {
            Poses::Camera($s) => $e,
            Poses::Transform($s) => $e,
        }
    };
}

fn duration<P>(s: &[Sample<P>]) -> f64 {
    s.last().map_or(0.0, |s| s.seconds)
}

fn check<P: Pose>(s: &[Sample<P>]) -> Result<(), &'static str> {
    if s.len() < 2 {
        return Err("needs at least two samples");
    }
    if s.first().is_some_and(|s| s.seconds != 0.0) {
        return Err("time must start at 0");
    }
    if s.windows(2).any(|w| {
        w[0].seconds.partial_cmp(&w[1].seconds) != Some(std::cmp::Ordering::Less)
            || !w[1].seconds.is_finite()
    }) {
        return Err("sample times must increase");
    }
    if s.iter().any(|s| !s.pose.finite()) {
        return Err("values must be finite");
    }
    Ok(())
}

/// The pose at `seconds`, held at the first / last pose outside the take.
fn pose_at<P: Pose>(s: &[Sample<P>], seconds: f64) -> Option<P> {
    let (first, last) = (s.first()?, s.last()?);
    if seconds <= first.seconds {
        return Some(first.pose);
    }
    if seconds >= last.seconds {
        return Some(last.pose);
    }
    let hi = s.partition_point(|x| x.seconds <= seconds);
    let a = s.get(hi.checked_sub(1)?)?;
    let b = s.get(hi)?;
    let t = ((seconds - a.seconds) / (b.seconds - a.seconds)) as f32;
    Some(a.pose.blend(&b.pose, t))
}

/// The part between `from` and `to` seconds, rebased to 0, the cut poses interpolated.
fn trim<P: Pose>(s: &[Sample<P>], from: f64, to: f64) -> Result<Vec<Sample<P>>, String> {
    let (from, to) = (from.max(0.0), to.min(duration(s)));
    // Not greater includes NaN: nothing to keep.
    if to.partial_cmp(&from) != Some(std::cmp::Ordering::Greater) {
        return Err("Trim keeps nothing".into());
    }
    let first = pose_at(s, from).ok_or("Empty take")?;
    let last = pose_at(s, to).ok_or("Empty take")?;
    let mut out = vec![Sample {
        seconds: 0.0,
        pose: first,
    }];
    out.extend(
        s.iter()
            .filter(|x| x.seconds > from && x.seconds < to)
            .map(|x| Sample {
                seconds: x.seconds - from,
                pose: x.pose,
            }),
    );
    out.push(Sample {
        seconds: to - from,
        pose: last,
    });
    Ok(out)
}

fn retime<P: Pose>(s: &[Sample<P>], speed: f64) -> Vec<Sample<P>> {
    s.iter()
        .map(|x| Sample {
            seconds: x.seconds / speed,
            pose: x.pose,
        })
        .collect()
}

/// `effects` on the samples: each channel ([`Pose::channels`]) a curve of linear keys at the
/// sample times, edited by the same `curves::Effect` the timeline applies to keys (the Euler
/// filter to the rotation's three channels, [`Pose::ROTATION`]), and read back at those times.
fn apply_effects<P: Pose>(
    s: &[Sample<P>],
    effects: &[curves::Effect<f64>],
) -> Result<Vec<Sample<P>>, String> {
    if effects.is_empty() {
        return Ok(s.to_vec());
    }
    let rows: Vec<Vec<f32>> = s.iter().map(|x| x.pose.channels()).collect();
    let count = rows.first().map_or(0, Vec::len);
    let mut tracks = (0..count)
        .map(|c| {
            curves::Track::from_keys(
                s.iter()
                    .zip(&rows)
                    .map(|(x, row)| {
                        curves::Key::new(x.seconds, f64::from(row[c])).with_tan(curves::Tan::Linear)
                    })
                    .collect(),
            )
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    // The rotation's angles are made continuous first (the Euler filter), whatever the effect:
    // a recorded wrap at +-180 would otherwise be smoothed or fitted through as a full turn.
    let (first, middle) = P::ROTATION;
    let mut rotation = tracks[first..first + 3].to_vec();
    curves::Effect::EulerFilter { middle }
        .apply(&mut rotation)
        .map_err(|e| e.to_string())?;
    tracks.splice(first..first + 3, rotation);
    for effect in effects {
        if !matches!(effect, curves::Effect::EulerFilter { .. }) {
            effect.apply(&mut tracks).map_err(|e| e.to_string())?;
        }
    }
    Ok(s.iter()
        .map(|x| {
            let values: Vec<f32> = tracks.iter().map(|t| t.eval(x.seconds) as f32).collect();
            Sample {
                seconds: x.seconds,
                pose: x.pose.with_channels(&values),
            }
        })
        .collect())
}

#[cfg(test)]
fn effected<P: Pose>(
    s: &[Sample<P>],
    effect: &curves::Effect<f64>,
) -> Result<Vec<Sample<P>>, String> {
    apply_effects(s, std::slice::from_ref(effect))
}

fn filtered_samples<P: Pose>(
    s: &[Sample<P>],
    filters: &TakeFilters,
) -> Result<Vec<Sample<P>>, String> {
    if !filters.any_active() {
        return Ok(s.to_vec());
    }
    let mut effects = Vec::new();
    if filters.euler {
        effects.push(curves::Effect::EulerFilter {
            middle: P::ROTATION.1,
        });
    }
    if filters.smooth {
        let order = filters.order.clamp(2, 16);
        let order = order + (order % 2);
        effects.push(curves::Effect::Smooth {
            cutoff: filters.cutoff.clamp(0.01, 100.0),
            order,
            blend: filters.blend.clamp(0.0, 60.0),
        });
    }
    if filters.simplify {
        effects.push(curves::Effect::Simplify {
            tolerance: filters.tolerance.max(0.0),
            max_keys: (filters.max_keys > 0).then_some(filters.max_keys),
        });
    }
    apply_effects(s, &effects)
}

/// Non-destructive curve filters and their parameters on a [`Take`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TakeFilters {
    pub euler: bool,
    pub smooth: bool,
    pub simplify: bool,
    /// Simplify: largest allowed error in channel units, and key limit (0 = unlimited).
    pub tolerance: f64,
    pub max_keys: usize,
    /// Smooth: cutoff in Hz, poles (even, 2..=16), blend in seconds.
    pub cutoff: f64,
    pub order: u32,
    pub blend: f64,
}
impl Default for TakeFilters {
    fn default() -> Self {
        Self {
            euler: false,
            smooth: false,
            simplify: false,
            tolerance: 0.01,
            max_keys: 0,
            cutoff: 3.0,
            order: 4,
            blend: 0.0,
        }
    }
}
impl TakeFilters {
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }
    pub fn any_active(&self) -> bool {
        self.euler || self.smooth || self.simplify
    }
}

/// Cached filtered poses on a [`Take`], ignored by equality and serialization.
#[derive(Clone, Debug, Default)]
pub struct FilteredCache(Option<Poses>);
impl PartialEq for FilteredCache {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

/// One recorded motion.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Take {
    /// Stable key within the store; 0 until the take is added.
    pub id: u64,
    pub name: String,
    /// Document frame the motion began on: where an attached take starts playing (before the
    /// node's `/take/offset`).
    pub start: f64,
    pub timing: Timing,
    /// Pinned takes are never dropped by retention.
    pub pinned: bool,
    /// The node it was recorded on (shown in the Takes panel; other nodes may play it).
    #[serde(default)]
    pub source: Option<NodeId>,
    /// Non-destructive curve filters active on this take.
    #[serde(default, skip_serializing_if = "TakeFilters::is_default")]
    pub filters: TakeFilters,
    /// Strictly increasing seconds from 0.
    pub poses: Poses,
    /// Cached filtered poses when `filters.any_active()` is true.
    #[serde(skip)]
    pub filtered: FilteredCache,
}

impl Take {
    /// A take of `samples`, unnamed and without an id; validated.
    pub fn new(
        start: f64,
        source: Option<NodeId>,
        poses: impl Into<Poses>,
    ) -> Result<Self, String> {
        Take {
            id: 0,
            name: String::new(),
            start,
            timing: Timing::KeepSpeed,
            pinned: false,
            source,
            filters: TakeFilters::default(),
            poses: poses.into(),
            filtered: FilteredCache::default(),
        }
        .checked()
    }

    /// A take through `poses`, one per frame from `start` at `fps` (a node's timeline animation).
    pub fn from_frames<P: Pose>(
        start: f64,
        fps: f64,
        source: Option<NodeId>,
        poses: Vec<P>,
    ) -> Result<Self, String>
    where
        Poses: From<Vec<Sample<P>>>,
    {
        if !(fps.is_finite() && fps > 0.0) {
            return Err("Invalid frame rate".into());
        }
        let samples: Vec<Sample<P>> = poses
            .into_iter()
            .enumerate()
            .map(|(i, pose)| Sample {
                seconds: i as f64 / fps,
                pose,
            })
            .collect();
        Self::new(start, source, samples)
    }

    /// A still take of `pose` from frame `start`: a pose bookmark (two equal poses one frame
    /// apart, the least a take holds).
    pub fn still<P: Pose>(
        pose: P,
        start: f64,
        fps: f64,
        source: Option<NodeId>,
    ) -> Result<Self, String>
    where
        Poses: From<Vec<Sample<P>>>,
    {
        Self::from_frames(start, fps, source, vec![pose, pose])
    }

    pub fn kind(&self) -> TakeKind {
        match self.poses {
            Poses::Camera(_) => TakeKind::Camera,
            Poses::Transform(_) => TakeKind::Transform,
        }
    }

    pub fn duration(&self) -> f64 {
        on_poses!(&self.poses, s => duration(s))
    }

    pub fn len(&self) -> usize {
        on_poses!(&self.poses, s => s.len())
    }

    /// Recompute the cached filtered poses from `self.poses` and `self.filters`.
    pub fn rebuild_filtered(&mut self) {
        if !self.filters.any_active() {
            self.filtered.0 = None;
            return;
        }
        self.filtered.0 = Some(map_poses!(
            &self.poses,
            s => filtered_samples(s, &self.filters).unwrap_or_else(|_| s.clone())
        ));
    }

    /// The poses this take plays and bakes: the non-destructively filtered poses when any filter
    /// is active, otherwise the raw recorded poses.
    pub fn effective_poses(&self) -> &Poses {
        self.filtered.0.as_ref().unwrap_or(&self.poses)
    }

    /// Reject samples that cannot drive anything: too few, unordered or non-finite.
    pub fn checked(mut self) -> Result<Self, String> {
        let problem = if !self.start.is_finite() {
            Some("start frame is not finite")
        } else if ![
            self.filters.tolerance,
            self.filters.cutoff,
            self.filters.blend,
        ]
        .iter()
        .all(|v| v.is_finite())
        {
            Some("filter parameters must be finite")
        } else {
            on_poses!(&self.poses, s => check(s).err())
        };
        match problem {
            Some(what) => Err(format!("Invalid take \"{}\": {what}", self.name)),
            None => {
                self.rebuild_filtered();
                Ok(self)
            }
        }
    }

    /// Seconds into the take that document `frame` shows: wall speed from `start` at the document
    /// rate (`KeepSpeed`) or the whole take stretched over the work area (`FitWorkArea`). The same
    /// mapping as the bake (`camera_recorder`).
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

    /// The document frame the take begins playing on: its start frame (`KeepSpeed`) or the work
    /// area's first frame (`FitWorkArea`). A node's time warp pivots here
    /// (`WorldDocument::take_frame`).
    pub fn play_start(&self, first: u32) -> f64 {
        match self.timing {
            Timing::KeepSpeed => self.start,
            Timing::FitWorkArea => f64::from(first),
        }
    }

    /// The camera a document at (`first`..`last`, `fps`) shows on `frame`; None for a transform take.
    pub fn camera_at(&self, frame: f64, first: u32, last: u32, fps: f64) -> Option<Camera> {
        match self.effective_poses() {
            Poses::Camera(s) => pose_at(s, self.seconds_at(frame, first, last, fps)),
            Poses::Transform(_) => None,
        }
    }

    /// The local transform on `frame`; None for a camera take.
    pub fn trs_at(&self, frame: f64, first: u32, last: u32, fps: f64) -> Option<Trs> {
        match self.effective_poses() {
            Poses::Transform(s) => pose_at(s, self.seconds_at(frame, first, last, fps)),
            Poses::Camera(_) => None,
        }
    }

    /// The part between `from` and `to` seconds, rebased to 0. The cut poses are interpolated,
    /// and the start frame moves with the cut (`from` seconds at `fps`), so the kept motion
    /// still plays on the frames it played on.
    pub fn trimmed(&self, from: f64, to: f64, fps: f64) -> Result<Self, String> {
        Take {
            start: self.start + from.max(0.0) * fps,
            poses: map_poses!(&self.poses, s => trim(s, from, to)?),
            ..self.clone()
        }
        .checked()
    }

    /// Played `speed` times as fast (2 = half the duration).
    pub fn retimed(&self, speed: f64) -> Result<Self, String> {
        if !(speed.is_finite() && speed > 0.0) {
            return Err("Speed must be positive".into());
        }
        Take {
            poses: map_poses!(&self.poses, s => retime(s, speed)),
            ..self.clone()
        }
        .checked()
    }

    /// A curve effect (Simplify, Smooth, Euler filter) on the take's motion, as the timeline
    /// applies it to keys ([`effected`]).
    #[cfg(test)]
    pub fn effected(&self, effect: &curves::Effect<f64>) -> Result<Self, String> {
        Take {
            poses: map_poses!(&self.poses, s => effected(s, effect)?),
            ..self.clone()
        }
        .checked()
    }

    /// Cut at `seconds` into two takes; the second starts on the frame the cut played on.
    pub fn split(&self, seconds: f64, fps: f64) -> Result<(Self, Self), String> {
        let head = self.trimmed(0.0, seconds, fps)?;
        let tail = self.trimmed(seconds, self.duration(), fps)?;
        Ok((head, tail))
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

/// The exchange format of one take (file and clipboard): no id or source node, which
/// belong to the scene it came from.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Clip {
    version: u32,
    /// Document rate the take was recorded under (informational: samples are in seconds).
    fps: f64,
    name: String,
    start: f64,
    timing: Timing,
    #[serde(default, skip_serializing_if = "TakeFilters::is_default")]
    filters: TakeFilters,
    poses: Poses,
}

impl Clip {
    fn of(take: &Take, fps: f64) -> Self {
        Clip {
            version: CLIP_VERSION,
            fps,
            name: take.name.clone(),
            start: take.start,
            timing: take.timing,
            filters: take.filters,
            poses: take.poses.clone(),
        }
    }

    fn take(self) -> Result<Take, String> {
        if self.version != CLIP_VERSION {
            return Err(format!(
                "Unsupported take clip version {} (expected {CLIP_VERSION})",
                self.version
            ));
        }
        Take {
            name: self.name,
            timing: self.timing,
            filters: self.filters,
            ..Take::new(self.start, None, self.poses)?
        }
        .checked()
    }
}

/// The takes of one document. Which nodes play them is node data (`/take/clip`), so every
/// operation that can drop a take is told the ids in use (`WorldDocument::takes_in_use`).
#[derive(Clone, Debug, Default)]
pub struct Takes {
    /// Oldest first. `Arc` keeps document clones (undo snapshots, export jobs) cheap.
    list: Vec<Arc<Take>>,
    /// Next id: ids are never reused, so a stale reference cannot hit another take.
    next: u64,
}

impl Takes {
    pub fn is_empty(&self) -> bool {
        self.list.is_empty() && self.next == 0
    }
    pub fn list(&self) -> &[Arc<Take>] {
        &self.list
    }
    pub fn get(&self, id: u64) -> Option<&Arc<Take>> {
        self.list.iter().find(|t| t.id == id)
    }
    /// Add a take (named "Take N" when unnamed) and apply retention, keeping the takes in
    /// `used`. Returns its id.
    pub fn add(&mut self, mut take: Take, used: &[u64]) -> u64 {
        self.next += 1;
        take.id = self.next;
        if take.name.trim().is_empty() {
            take.name = format!("Take {}", take.id);
        }
        take.rebuild_filtered();
        self.list.push(Arc::new(take));
        self.trim(used);
        self.next
    }

    /// Drop the oldest unpinned takes beyond [`MAX_UNPINNED`]. Takes in `used` (played by a node)
    /// are kept too: dropping one would silently change the motion of the scene.
    fn trim(&mut self, used: &[u64]) {
        while self.list.iter().filter(|t| !t.pinned).count() > MAX_UNPINNED {
            let Some(index) = self
                .list
                .iter()
                .position(|t| !t.pinned && !used.contains(&t.id))
            else {
                break;
            };
            self.list.remove(index);
        }
    }

    /// Remove a take. The caller makes sure no node plays it (`WorldEditor::delete_take`).
    pub fn remove(&mut self, id: u64) -> bool {
        let before = self.list.len();
        self.list.retain(|t| t.id != id);
        self.list.len() != before
    }

    /// Mutate one take in place (retention keeps `used`); false when it does not exist.
    pub fn edit(&mut self, id: u64, used: &[u64], apply: impl FnOnce(&mut Take)) -> bool {
        let Some(take) = self.list.iter_mut().find(|t| t.id == id) else {
            return false;
        };
        let take = Arc::make_mut(take);
        apply(take);
        take.rebuild_filtered();
        self.trim(used);
        true
    }
}

#[derive(Serialize)]
struct TakesOut<'a> {
    list: Vec<&'a Take>,
    next: u64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TakesIn {
    list: Vec<Take>,
    next: u64,
}
impl Serialize for Takes {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        TakesOut {
            list: self.list.iter().map(Arc::as_ref).collect(),
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
        Ok(Self {
            list,
            next: input.next,
        })
    }
}

/// What the navigation recorder needs from one UI frame.
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
    /// A gesture ended and produced this take (a camera take, its source unset).
    pub finished: Option<Take>,
}

struct Live {
    start: f64,
    samples: Vec<Sample<Camera>>,
    clock: f64,
    idle: f64,
    /// Index of the last sample that differs from its predecessor: the take ends there.
    moved: usize,
}

/// Cuts viewport navigation gestures into camera takes; one instance per app. A gesture runs
/// from the first move until the input has ended and the camera has come to rest.
pub struct Recorder {
    /// Minimum gesture duration (seconds) required to keep a recorded take.
    pub min_secs: f64,
    live: Option<Live>,
}

impl Default for Recorder {
    fn default() -> Self {
        Self {
            min_secs: MIN_SECS,
            live: None,
        }
    }
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
                    start: t.frame,
                    samples: vec![
                        Sample {
                            seconds: 0.0,
                            pose: t.before,
                        },
                        Sample {
                            seconds: t.dt,
                            pose: t.after,
                        },
                    ],
                    clock: t.dt,
                    idle: 0.0,
                    moved: 1,
                });
                update.started = true;
            }
            return update;
        };
        live.clock += t.dt;
        live.samples.push(Sample {
            seconds: live.clock,
            pose: t.after,
        });
        if moved {
            live.idle = 0.0;
            live.moved = live.samples.len() - 1;
        } else {
            live.idle += t.dt;
        }
        if (!t.held && live.idle >= SETTLE_SECS) || live.samples.len() >= MAX_SAMPLES {
            update.finished = self.finish();
        }
        update
    }

    /// End the gesture now, dropping the still tail. None when it was too short to keep.
    fn finish(&mut self) -> Option<Take> {
        let mut live = self.live.take()?;
        live.samples.truncate(live.moved + 1);
        let min_secs = self.min_secs.max(MIN_SECS);
        (duration(&live.samples) >= min_secs)
            .then(|| Take::new(live.start, None, live.samples).ok())
            .flatten()
    }
}

/// Records manipulator drags: each dragged node's local transform every frame of the drag,
/// cut into one transform take per node when the drag ends.
pub struct DragRecorder {
    /// Minimum gesture duration (seconds) required to keep a recorded take.
    pub min_secs: f64,
    start: f64,
    clock: f64,
    live: Vec<(NodeId, Vec<Sample<Trs>>)>,
}

impl Default for DragRecorder {
    fn default() -> Self {
        Self {
            min_secs: MIN_SECS,
            start: 0.0,
            clock: 0.0,
            live: Vec::new(),
        }
    }
}

impl DragRecorder {
    /// One frame of a drag: the dragged nodes' local transforms after this frame's edit. The
    /// first call starts the drag at document frame `frame`.
    pub fn sample(&mut self, dt: f64, frame: f64, poses: &[(NodeId, Trs)]) {
        if self.live.is_empty() {
            self.start = frame;
            self.clock = 0.0;
        } else {
            self.clock += dt;
        }
        for &(node, pose) in poses {
            let sample = Sample {
                seconds: self.clock,
                pose,
            };
            match self.live.iter_mut().find(|(n, _)| *n == node) {
                Some((_, samples)) => samples.push(sample),
                None => self.live.push((node, vec![sample])),
            }
        }
    }

    /// The drag ended: one take per node that moved for at least the minimum gesture time
    /// (rebased to its own first sample, its still head and tail dropped).
    pub fn finish(&mut self) -> Vec<Take> {
        let start = self.start;
        let min_secs = self.min_secs.max(MIN_SECS);
        std::mem::take(&mut self.live)
            .into_iter()
            .filter_map(|(node, samples)| {
                let first = samples.windows(2).position(|w| w[0].pose != w[1].pose)?;
                let last = samples.windows(2).rposition(|w| w[0].pose != w[1].pose)? + 1;
                let t0 = samples[first].seconds;
                let kept: Vec<_> = samples[first..=last]
                    .iter()
                    .map(|s| Sample {
                        seconds: s.seconds - t0,
                        pose: s.pose,
                    })
                    .collect();
                (duration(&kept) >= min_secs)
                    .then(|| Take::new(start, Some(node), kept).ok())
                    .flatten()
            })
            .collect()
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
#[path = "takes_tests.rs"]
mod tests;
