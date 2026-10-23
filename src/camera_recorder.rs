//! Bake of a take into keyframes: samples are mapped to document frames and reduced with the
//! shared fitter. A camera take goes through the camera's navigation pose solver ([`Recording`]),
//! a transform take writes its node's `/transform` channels ([`bake_transform`]). The takes
//! themselves live in `takes`.
use crate::scene::{Bulb, Camera, Formula, Scene};
use crate::takes::{Poses, Sample, Take};
use crate::world::{NodeId, WorldDocument};
use curves::{Fitter, Track, nearest_euler};
use playa_engine::entities::anim::{Animation, Channel};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Timing {
    KeepSpeed,
    FitWorkArea,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Options {
    pub transform: bool,
    pub focus: bool,
    pub zoom: bool,
    pub f_number: bool,
    pub timing: Timing,
    /// Absolute error at captured samples, in each channel's authored units.
    pub tolerance: f64,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            transform: true,
            focus: true,
            zoom: true,
            f_number: true,
            timing: Timing::KeepSpeed,
            tolerance: 0.0001,
        }
    }
}

pub struct Recording {
    pub document: WorldDocument,
    pub camera: NodeId,
    pub revision: u64,
    pub start_frame: f64,
    pub options: Options,
    pub samples: Vec<Sample<Camera>>,
}

/// Keys ready to write onto `node` over `[first, last]` (`WorldEditor::record_prepared`).
pub struct Prepared {
    pub node: NodeId,
    pub channels: BTreeMap<&'static str, Animation>,
    pub first: f64,
    pub last: f64,
    pub extend_last: u32,
    pub sample_count: usize,
    pub key_count: usize,
    pub max_error: f64,
}

impl Recording {
    pub fn new(
        document: &WorldDocument,
        revision: u64,
        start_frame: f64,
        scene: &Scene,
        options: Options,
    ) -> Result<Self, String> {
        let camera = document.active_camera.ok_or("No active camera to record")?;
        if !(options.transform || options.focus || options.zoom || options.f_number) {
            return Err("Select at least one camera channel".into());
        }
        if !document.fps.is_finite()
            || document.fps <= 0.0
            || !start_frame.is_finite()
            || !options.tolerance.is_finite()
            || options.tolerance < 0.0
        {
            return Err("Invalid camera recording timing or tolerance".into());
        }
        document.assert_unlocked(camera)?;
        // Validate parent transforms and lens payload before capturing any frames.
        document.camera_navigation_pose(camera, scene, start_frame)?;
        Ok(Self {
            document: document.clone(),
            camera,
            revision,
            start_frame,
            options,
            samples: vec![Sample {
                seconds: 0.0,
                pose: scene.camera,
            }],
        })
    }

    /// A bake of the camera take `take` over `document`: the take's timing overrides
    /// `options.timing`.
    pub fn from_take(
        document: &WorldDocument,
        revision: u64,
        take: &Take,
        options: Options,
    ) -> Result<Self, String> {
        let Poses::Camera(samples) = take.effective_poses() else {
            return Err("Not a camera take".into());
        };
        let first = samples.first().ok_or("The take has no samples")?;
        let mut scene = Scene::preset(Formula::Mandelbulb(Bulb::PRESET));
        scene.camera = first.pose;
        let options = Options {
            timing: take.timing,
            ..options
        };
        let mut recording = Self::new(document, revision, take.start, &scene, options)?;
        for sample in samples.iter().skip(1) {
            recording.capture(sample.seconds, sample.pose)?;
        }
        Ok(recording)
    }

    pub fn capture(&mut self, seconds: f64, camera: Camera) -> Result<(), String> {
        let previous = self.samples.last().ok_or("Empty camera recording")?;
        if !seconds.is_finite() || seconds < previous.seconds {
            return Err("Camera recording time must increase".into());
        }
        if seconds == previous.seconds {
            self.samples.last_mut().unwrap().pose = camera;
        } else {
            self.samples.push(Sample {
                seconds,
                pose: camera,
            });
        }
        Ok(())
    }

    pub fn prepare(&self) -> Result<Prepared, String> {
        let duration = self.samples.last().ok_or("Empty camera recording")?.seconds;
        if self.samples.len() < 2 || duration <= 0.0 {
            return Err("Record more than one camera sample".into());
        }
        let (first, last) = frame_span(
            self.options.timing,
            self.start_frame,
            duration,
            &self.document,
        )?;
        let mut data: BTreeMap<&'static str, Vec<Vec<(f64, f64)>>> = BTreeMap::new();
        let mut scene = Scene::preset(Formula::Mandelbulb(Bulb::PRESET));
        for sample in &self.samples {
            let frame = first + (last - first) * (sample.seconds / duration);
            scene.camera = sample.pose;
            for (path, value) in self
                .document
                .camera_navigation_pose(self.camera, &scene, frame)?
            {
                let selected = match path {
                    "/transform/position" | "/transform/rotation_degrees" | "/camera/distance" => {
                        self.options.transform
                    }
                    "/camera/focus_distance" => self.options.focus,
                    "/camera/fov_y_degrees" => self.options.zoom,
                    "/camera/f_number" => self.options.f_number,
                    _ => false,
                };
                if selected {
                    append(&mut data, path, frame, &value)?;
                }
            }
            if self.options.transform {
                // Replay the same local scale used by the pose solver at this mapped frame.
                // Freezing the start scale would invalidate its distance and basis conversion.
                let scale =
                    self.document
                        .attribute_value(self.camera, "/transform/scale", frame)?;
                append(&mut data, "/transform/scale", frame, &scale)?;
            }
        }
        let (channels, key_count, max_error) = fit_channels(data, self.options.tolerance)?;
        Ok(Prepared {
            node: self.camera,
            channels,
            first,
            last,
            extend_last: self.document.last.max(last.ceil() as u32),
            sample_count: self.samples.len(),
            key_count,
            max_error,
        })
    }
}

/// Keys of the transform take `take` on `node`'s `/transform` position, rotation and scale (all
/// three: the bake's Translate / Rotate / Scale switch), mapped to frames like a camera take.
pub fn bake_transform(
    document: &WorldDocument,
    node: NodeId,
    take: &Take,
    options: Options,
) -> Result<Prepared, String> {
    let Poses::Transform(samples) = take.effective_poses() else {
        return Err("Not a transform take".into());
    };
    if !options.transform {
        return Err("Select Translate / Rotate / Scale to bake a transform take".into());
    }
    if !options.tolerance.is_finite() || options.tolerance < 0.0 {
        return Err("Invalid bake tolerance".into());
    }
    document.assert_unlocked(node)?;
    let duration = take.duration();
    let (first, last) = frame_span(take.timing, take.start, duration, document)?;
    let mut data: BTreeMap<&'static str, Vec<Vec<(f64, f64)>>> = BTreeMap::new();
    for sample in samples {
        let frame = first + (last - first) * (sample.seconds / duration);
        let p = sample.pose;
        for (path, value) in [
            ("/transform/position", p.position),
            ("/transform/rotation_degrees", p.rotation_degrees),
            ("/transform/scale", p.scale),
        ] {
            append(&mut data, path, frame, &serde_json::json!(value))?;
        }
    }
    let (channels, key_count, max_error) = fit_channels(data, options.tolerance)?;
    Ok(Prepared {
        node,
        channels,
        first,
        last,
        extend_last: document.last.max(last.ceil() as u32),
        sample_count: samples.len(),
        key_count,
        max_error,
    })
}

/// The document frames a take of `duration` seconds from frame `start` plays over: at recorded
/// speed (`KeepSpeed`) or stretched over the work area (`FitWorkArea`); `Take::seconds_at` is
/// its inverse.
fn frame_span(
    timing: Timing,
    start: f64,
    duration: f64,
    document: &WorldDocument,
) -> Result<(f64, f64), String> {
    if duration <= 0.0 || !document.fps.is_finite() || document.fps <= 0.0 {
        return Err("A take needs a positive duration and frame rate".into());
    }
    let (first, last) = match timing {
        Timing::KeepSpeed => (start, start + duration * document.fps),
        Timing::FitWorkArea => (f64::from(document.first), f64::from(document.last)),
    };
    if !last.is_finite() || last > f64::from(u32::MAX - 1) || last <= first {
        return Err("A take needs a nonempty work area within the frame limit".into());
    }
    Ok((first, last))
}

/// Per-frame samples (path -> components -> (frame, value)) fitted to animations within
/// `tolerance`, rotations kept continuous first. Returns the animations, the key count and the
/// largest sample error.
fn fit_channels(
    data: BTreeMap<&'static str, Vec<Vec<(f64, f64)>>>,
    tolerance: f64,
) -> Result<(BTreeMap<&'static str, Animation>, usize, f64), String> {
    let mut fitter = Fitter::new();
    let mut channels = BTreeMap::new();
    let mut key_count = 0;
    let mut max_error: f64 = 0.0;
    for (path, mut components) in data {
        if path == "/transform/rotation_degrees" {
            continuous_rotation(&mut components)?;
        }
        let mut animation = Animation::with_arity(components.len());
        for (index, samples) in components.into_iter().enumerate() {
            let mut track = Track::new();
            let stats = fitter
                .fit(&samples, tolerance, &mut track)
                .map_err(|e| e.to_string())?;
            max_error = max_error.max(stats.max_err);
            key_count += stats.keys;
            animation.channels[index] = Channel::from_track(track);
        }
        channels.insert(path, animation);
    }
    Ok((channels, key_count, max_error))
}

fn append(
    data: &mut BTreeMap<&'static str, Vec<Vec<(f64, f64)>>>,
    path: &'static str,
    frame: f64,
    value: &Value,
) -> Result<(), String> {
    let components: Vec<f64> = if let Some(vector) = value.as_array() {
        vector
            .iter()
            .map(|v| v.as_f64().ok_or("Invalid recorded camera vector"))
            .collect::<Result<_, _>>()?
    } else {
        vec![value.as_f64().ok_or("Invalid recorded camera value")?]
    };
    let samples = data
        .entry(path)
        .or_insert_with(|| vec![Vec::new(); components.len()]);
    if samples.len() != components.len() {
        return Err("Camera channel shape changed".into());
    }
    for (samples, value) in samples.iter_mut().zip(components) {
        if !value.is_finite() {
            return Err("Recorded camera values must be finite".into());
        }
        samples.push((frame, value));
    }
    Ok(())
}

/// Rotation samples (`/transform/rotation_degrees`: x, y, z, Playa's ZYX order, y the middle
/// rotation) rewritten so each sample is the equivalent of its angles nearest the previous one
/// (`curves::nearest_euler`: both Euler solutions and whole turns), the same rule as the Euler
/// filter: no 180 / 360 degree flips between samples. The components share their sample times.
fn continuous_rotation(components: &mut [Vec<(f64, f64)>]) -> Result<(), String> {
    let [x, y, z] = components else {
        return Err("Rotation needs three components".into());
    };
    if x.len() != y.len() || x.len() != z.len() {
        return Err("Rotation components have different samples".into());
    }
    for i in 1..x.len() {
        let previous = [x[i - 1].1, y[i - 1].1, z[i - 1].1];
        let [a, b, c] = nearest_euler([x[i].1, y[i].1, z[i].1], previous, Some(1));
        (x[i].1, y[i].1, z[i].1) = (a, b, c);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{Bulb, Formula};
    fn recording(timing: Timing) -> Recording {
        let scene = Scene::preset(Formula::Mandelbulb(Bulb::PRESET));
        let document = WorldDocument::from_scene(&scene);
        let scene = document.snapshot(0.0).unwrap();
        Recording::new(
            &document,
            0,
            10.0,
            &scene,
            Options {
                timing,
                ..Options::default()
            },
        )
        .unwrap()
    }
    #[test]
    fn capture_preserves_speed_or_fits_work_area_and_selects_channels() {
        let mut r = recording(Timing::KeepSpeed);
        r.options = Options {
            transform: false,
            focus: false,
            zoom: true,
            f_number: false,
            ..r.options
        };
        let mut camera = r.samples[0].pose;
        for i in 1..=24 {
            camera.fov_y_degrees += 1.0;
            r.capture(f64::from(i) / 24.0, camera).unwrap();
        }
        let prepared = r.prepare().unwrap();
        assert_eq!((prepared.first, prepared.last), (10.0, 34.0));
        assert_eq!(prepared.channels.len(), 1);
        assert_eq!(prepared.key_count, 2);
        assert!(prepared.max_error <= r.options.tolerance);
        r.options.timing = Timing::FitWorkArea;
        let prepared = r.prepare().unwrap();
        assert_eq!((prepared.first, prepared.last), (0.0, 239.0));
        let track = prepared.channels["/camera/fov_y_degrees"].channels[0].track();
        assert!((track.eval(239.0) - f64::from(camera.fov_y_degrees)).abs() < 1e-6);
    }
    #[test]
    fn keep_speed_extends_work_area_and_rotation_crosses_wrap_continuously() {
        let mut r = recording(Timing::KeepSpeed);
        let mut camera = r.samples[0].pose;
        for i in 1..=12 {
            camera.roll_degrees = 170.0 + i as f32 * 3.0;
            r.capture(f64::from(i), camera).unwrap();
        }
        let prepared = r.prepare().unwrap();
        assert_eq!(prepared.last, 298.0);
        assert_eq!(prepared.extend_last, 298);
        let roll = prepared.channels["/transform/rotation_degrees"].channels[2].track();
        for pair in roll.keys().windows(2) {
            assert!((pair[1].v() - pair[0].v()).abs() < 180.0);
        }
    }
}
