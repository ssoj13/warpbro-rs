//! Wall-clock camera capture, mapped to document frames and reduced with the shared fitter.
use crate::scene::{Camera, Scene};
use crate::world::{NodeId, WorldDocument};
use curves::{Fitter, Track, unwind_slice};
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

#[derive(Clone, Copy, Debug)]
pub struct Sample {
    pub seconds: f64,
    pub camera: Camera,
}

pub struct Recording {
    pub document: WorldDocument,
    pub camera: NodeId,
    pub revision: u64,
    pub start_frame: f64,
    pub options: Options,
    pub samples: Vec<Sample>,
}

pub struct Prepared {
    pub camera: NodeId,
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
                camera: scene.camera,
            }],
        })
    }

    pub fn capture(&mut self, seconds: f64, camera: Camera) -> Result<(), String> {
        let previous = self.samples.last().ok_or("Empty camera recording")?;
        if !seconds.is_finite() || seconds < previous.seconds {
            return Err("Camera recording time must increase".into());
        }
        if seconds == previous.seconds {
            self.samples.last_mut().unwrap().camera = camera;
        } else {
            self.samples.push(Sample { seconds, camera });
        }
        Ok(())
    }

    pub fn prepare(&self) -> Result<Prepared, String> {
        let duration = self.samples.last().ok_or("Empty camera recording")?.seconds;
        if self.samples.len() < 2 || duration <= 0.0 {
            return Err("Record more than one camera sample".into());
        }
        let (first, last) = match self.options.timing {
            Timing::KeepSpeed => (
                self.start_frame,
                self.start_frame + duration * self.document.fps,
            ),
            Timing::FitWorkArea => (
                f64::from(self.document.first),
                f64::from(self.document.last),
            ),
        };
        if !last.is_finite() || last > f64::from(u32::MAX - 1) || last <= first {
            return Err(
                "Camera recording needs a nonempty work area within the frame limit".into(),
            );
        }
        let mut data: BTreeMap<&'static str, Vec<Vec<(f64, f64)>>> = BTreeMap::new();
        let mut scene = Scene::preset(0);
        for sample in &self.samples {
            let frame = first + (last - first) * (sample.seconds / duration);
            scene.camera = sample.camera;
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
        let mut fitter = Fitter::new();
        let mut channels = BTreeMap::new();
        let mut key_count = 0;
        let mut max_error: f64 = 0.0;
        for (path, components) in data {
            let mut animation = Animation::with_arity(components.len());
            for (index, mut samples) in components.into_iter().enumerate() {
                if path == "/transform/rotation_degrees" {
                    let mut values: Vec<_> = samples.iter().map(|s| s.1).collect();
                    unwind_slice(&mut values, 360.0).map_err(|e| e.to_string())?;
                    for (sample, value) in samples.iter_mut().zip(values) {
                        sample.1 = value;
                    }
                }
                let mut track = Track::new();
                let stats = fitter
                    .fit(&samples, self.options.tolerance, &mut track)
                    .map_err(|e| e.to_string())?;
                max_error = max_error.max(stats.max_err);
                key_count += stats.keys;
                animation.channels[index] = Channel::from_track(track);
            }
            channels.insert(path, animation);
        }
        Ok(Prepared {
            camera: self.camera,
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

#[cfg(test)]
mod tests {
    use super::*;
    fn recording(timing: Timing) -> Recording {
        let scene = Scene::preset(0);
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
        let mut camera = r.samples[0].camera;
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
        let mut camera = r.samples[0].camera;
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
