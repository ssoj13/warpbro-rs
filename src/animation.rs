//! Scene parameter curves. JSON pointers are stable parameter identities, including
//! formula variants and vector components; rendering still receives a typed Scene.
use crate::scene::Scene;
use box_rs::{BoxValue, RustBox};
use curves::{CurveKind, Knot, eval_segment};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

fn box_path(pointer: &str) -> String {
    pointer
        .trim_start_matches('/')
        .split('/')
        .map(|s| {
            format!(
                "[{}]",
                Value::String(s.replace("~1", "/").replace("~0", "~"))
            )
        })
        .collect()
}

pub fn set_parameter(scene: &mut Scene, path: &str, value: Value) -> Result<(), String> {
    let mut doc = RustBox::from_json_value(
        serde_json::to_value(&*scene).map_err(|e| e.to_string())?,
        Default::default(),
    );
    let path = box_path(path);
    if doc.get_path(&path).is_none() {
        return Err("Parameter is unavailable in this scene".into());
    }
    doc.set_path(&path, BoxValue::from_json_value(value))
        .map_err(|e| e.to_string())?;
    let mut edited: Scene = serde_json::from_value(doc.to_json_value())
        .map_err(|e| format!("Invalid parameter: {e}"))?;
    edited.environment.revision = scene.environment.revision;
    *scene = edited;
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Key {
    pub frame: f64,
    pub value: Value,
    pub interpolation: CurveKind,
}

#[derive(Clone, Debug, PartialEq, Default, Serialize)]
pub struct Track {
    keys: Vec<Key>,
}

impl<'de> Deserialize<'de> for Track {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Stored {
            keys: Vec<Key>,
        }
        let mut track = Self::default();
        for key in Stored::deserialize(d)?.keys {
            if !track.upsert(key) {
                return Err(serde::de::Error::custom("Invalid animation key"));
            }
        }
        Ok(track)
    }
}

impl Track {
    pub fn keys(&self) -> &[Key] {
        &self.keys
    }
    pub fn upsert(&mut self, mut key: Key) -> bool {
        if !key.frame.is_finite() || !key_value(&key.value) {
            return false;
        }
        if key.frame == 0.0 {
            key.frame = 0.0;
        }
        if self
            .keys
            .first()
            .is_some_and(|k| !same_shape(&k.value, &key.value))
        {
            return false;
        }
        match self
            .keys
            .binary_search_by(|k| k.frame.total_cmp(&key.frame))
        {
            Ok(i) => self.keys[i] = key,
            Err(i) => self.keys.insert(i, key),
        }
        true
    }
    pub fn set(&mut self, frame: f64, value: Value) {
        let interpolation = self
            .keys
            .iter()
            .find(|k| k.frame == frame)
            .map_or(CurveKind::Linear, |k| k.interpolation);
        self.upsert(Key {
            frame,
            value,
            interpolation,
        });
    }
    pub fn remove(&mut self, frame: f64) {
        self.keys.retain(|k| k.frame != frame);
    }
    pub fn move_keys(&mut self, frames: &[f64], delta: f64) {
        if !delta.is_finite() || frames.iter().any(|f| !(f + delta).is_finite()) {
            return;
        }
        let moved: Vec<_> = self
            .keys
            .iter()
            .filter(|k| frames.contains(&k.frame))
            .cloned()
            .map(|mut k| {
                k.frame += delta;
                k
            })
            .collect();
        self.keys.retain(|k| !frames.contains(&k.frame));
        for k in moved {
            self.upsert(k);
        }
    }
    pub fn sample(&self, frame: f64) -> Option<Value> {
        let first = self.keys.first()?;
        if !frame.is_finite() {
            return None;
        }
        let i = self.keys.partition_point(|k| k.frame <= frame);
        if i == 0 {
            return Some(first.value.clone());
        }
        let a = &self.keys[i - 1];
        let Some(b) = self.keys.get(i) else {
            return Some(a.value.clone());
        };
        Some(interpolate(
            &a.value,
            &b.value,
            frame,
            a.frame,
            b.frame,
            a.interpolation,
        ))
    }
}

fn same_shape(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Null, _) | (_, Value::Null) | (Value::Object(_), Value::Object(_)) => true,
        (Value::Number(_), Value::Number(_))
        | (Value::Bool(_), Value::Bool(_))
        | (Value::String(_), Value::String(_)) => true,
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| same_shape(a, b))
        }
        _ => false,
    }
}
fn key_value(v: &Value) -> bool {
    match v {
        Value::Null => true,
        Value::Object(fields) => fields.values().all(key_value),
        Value::Array(values) => values.iter().all(key_value),
        _ => parameter_value(v),
    }
}
fn parameter_value(v: &Value) -> bool {
    match v {
        Value::Number(n) => n.as_f64().is_some_and(f64::is_finite),
        Value::Bool(_) | Value::String(_) => true,
        Value::Array(v) => !v.is_empty() && v.iter().all(parameter_value),
        _ => false,
    }
}
fn interpolate(a: &Value, b: &Value, frame: f64, start: f64, end: f64, kind: CurveKind) -> Value {
    match (a, b) {
        (Value::Number(a), Value::Number(b)) => {
            let av = a.as_f64().unwrap_or(0.0);
            let bv = b.as_f64().unwrap_or(av);
            let value = eval_segment(
                kind,
                Knot::new(start, av),
                Knot::new(end, bv),
                av,
                bv,
                frame,
            );
            if a.is_u64() {
                Value::from(value.round().clamp(0.0, u32::MAX as f64) as u64)
            } else if a.is_i64() {
                Value::from(value.round() as i64)
            } else {
                Value::from(value)
            }
        }
        (Value::Array(a), Value::Array(b)) if a.len() == b.len() => Value::Array(
            a.iter()
                .zip(b)
                .map(|(a, b)| interpolate(a, b, frame, start, end, kind))
                .collect(),
        ),
        _ => a.clone(), // switches and enums hold the left key
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Animation {
    pub first: u32,
    pub last: u32,
    pub fps: f64,
    pub tracks: BTreeMap<String, Track>,
}
impl Default for Animation {
    fn default() -> Self {
        Self {
            first: 0,
            last: 239,
            fps: 24.0,
            tracks: BTreeMap::new(),
        }
    }
}

/// Parameter discovery follows the scene's serialization rather than a hand-maintained
/// list. New render parameters automatically become keyable. Config and labels are not keys.
pub fn parameters(scene: &Scene) -> Vec<(String, Value)> {
    fn visit(v: &Value, path: &str, out: &mut Vec<(String, Value)>) {
        if parameter_value(v) {
            out.push((path.to_owned(), v.clone()));
        } else if let Value::Object(fields) = v {
            for (k, v) in fields {
                if matches!(
                    k.as_str(),
                    "animation" | "name" | "colour" | "preset" | "path"
                ) {
                    continue;
                }
                visit(
                    v,
                    &format!("{path}/{}", k.replace('~', "~0").replace('/', "~1")),
                    out,
                );
            }
        }
    }
    let mut out = Vec::new();
    if let Ok(v) = serde_json::to_value(scene) {
        visit(&v, "", &mut out);
        // Structural switches are discrete tracks; their numeric children remain
        // separately keyable. The parent is applied first, then child overrides.
        for path in ["/formula", "/julia", "/material/facing"] {
            if !out.iter().any(|(p, _)| p == path) {
                if let Some(value) = v.pointer(path) {
                    out.push((path.into(), value.clone()));
                }
            }
        }
    }
    out
}

impl Scene {
    /// A render snapshot at any frame. Animation metadata is removed from worker requests:
    /// editing a key that does not change this frame must not restart its accumulation.
    pub fn evaluated(&self, frame: f64) -> Result<Self, String> {
        if let Some(document) = &self.document {
            return document.snapshot(frame);
        }
        let mut base = self.clone();
        base.animation = Animation::default();
        if self.animation.tracks.is_empty() {
            return Ok(base);
        }
        let mut doc = RustBox::from_json_value(
            serde_json::to_value(&base).map_err(|e| e.to_string())?,
            Default::default(),
        );
        for (path, track) in &self.animation.tracks {
            let bp = box_path(path);
            if let Some(slot) = doc.get_path(&bp) {
                if let Some(value) = track.sample(frame) {
                    if !same_shape(&slot.to_json_value(), &value) {
                        return Err(format!("Animation type mismatch: {path}"));
                    }
                    doc.set_path(&bp, BoxValue::from_json_value(value))
                        .map_err(|e| e.to_string())?;
                }
            } // another formula variant's tracks remain dormant
        }
        let mut evaluated: Scene = serde_json::from_value(doc.to_json_value())
            .map_err(|e| format!("Invalid animated scene: {e}"))?;
        evaluated.environment.revision = self.environment.revision;
        Ok(evaluated)
    }
    pub fn apply_animation(&mut self, frame: f64) -> Result<(), String> {
        let animation = self.animation.clone();
        let mut evaluated = self.evaluated(frame)?;
        evaluated.animation = animation;
        *self = evaluated;
        Ok(())
    }
    pub fn key_parameter(&mut self, path: &str, frame: f64) {
        if let Some((_, value)) = parameters(self).into_iter().find(|(p, _)| p == path) {
            self.animation
                .tracks
                .entry(path.into())
                .or_default()
                .set(frame, value);
        }
    }
    /// Editing an animated value (including viewport camera gestures) creates/updates
    /// its key at the playhead. Unanimated values stay static.
    pub fn record_edits(&mut self, before: &[(String, Value)], frame: f64) {
        for (path, value) in parameters(self) {
            if let Some(track) = self.animation.tracks.get_mut(&path) {
                if before.iter().any(|(p, v)| p == &path && v != &value) {
                    track.set(frame, value);
                }
            }
        }
    }
}

pub fn label(path: &str) -> String {
    path.trim_start_matches('/')
        .split('/')
        .map(|part| {
            let s = part.replace('_', " ");
            let mut chars = s.chars();
            chars
                .next()
                .map(|c| c.to_uppercase().to_string() + chars.as_str())
                .unwrap_or_default()
        })
        .collect::<Vec<_>>()
        .join(" / ")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn environment_reload_survives_parameter_edits_and_evaluation() {
        let mut scene = Scene::preset(0);
        scene.environment.revision = 7;
        scene.key_parameter("/environment/intensity", 0.0);
        set_parameter(&mut scene, "/environment/intensity", Value::from(2.0)).unwrap();
        assert_eq!(scene.environment.revision, 7);
        scene.key_parameter("/environment/intensity", 10.0);
        let sample = scene.evaluated(5.0).unwrap();
        assert_eq!(sample.environment.revision, 7);
        assert_eq!(sample.environment.intensity, 1.5);
    }
    #[test]
    fn family_and_optional_switches_are_discrete_and_children_override() {
        let mut s = Scene::preset(0);
        s.key_parameter("/formula", 0.0);
        s.formula = Scene::preset(1).formula;
        s.key_parameter("/formula", 10.0);
        s.key_parameter("/julia", 0.0);
        s.julia = Some([0.3, 0.4, 0.5]);
        s.key_parameter("/julia", 10.0);
        assert_eq!(s.evaluated(9.0).unwrap().formula.code(), 0);
        assert_eq!(s.evaluated(10.0).unwrap().formula.code(), 1);
        assert!(s.evaluated(9.0).unwrap().julia.is_none());
        assert_eq!(s.evaluated(10.0).unwrap().julia, Some([0.3, 0.4, 0.5]));
        let doc = serde_json::to_string(&s).unwrap();
        assert_eq!(serde_json::from_str::<Scene>(&doc).unwrap(), s);
    }
    #[test]
    fn sample_vectors_switches_and_roundtrip() {
        let mut s = Scene::preset(0);
        s.key_parameter("/camera/target", 0.0);
        s.camera.target = [10.0, 20.0, 30.0];
        s.key_parameter("/camera/target", 10.0);
        s.key_parameter("/lighting/background", 0.0);
        s.lighting.background = false;
        s.key_parameter("/lighting/background", 10.0);
        let restored: Scene = serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
        assert_eq!(
            restored.evaluated(5.0).unwrap().camera.target,
            [5.0, 10.0, 15.0]
        );
        assert!(restored.evaluated(9.9).unwrap().lighting.background);
        assert!(!restored.evaluated(10.0).unwrap().lighting.background);
        assert_eq!(restored.evaluated(-1.0).unwrap().camera.target, [0.0; 3]);
    }
    #[test]
    fn moving_overlapping_keys_is_atomic() {
        let mut t = Track::default();
        t.set(10.25, Value::from(1.0));
        t.set(11.25, Value::from(2.0));
        t.move_keys(&[10.25, 11.25], 1.0);
        assert_eq!(
            t.keys().iter().map(|k| k.frame).collect::<Vec<_>>(),
            vec![11.25, 12.25]
        );
        assert_eq!(t.sample(11.25), Some(Value::from(1.0)));
    }
    #[test]
    fn every_formula_and_material_parameter_is_discovered() {
        for family in 0..8 {
            let s = Scene::preset(family);
            let params = parameters(&s);
            assert!(params.iter().any(|(p, _)| p.starts_with("/formula/")));
            assert!(params.iter().any(|(p, _)| p == "/material/coat_roughness"));
            assert!(
                !params
                    .iter()
                    .any(|(p, _)| p.starts_with("/animation") || p.starts_with("/colour"))
            );
        }
    }
    #[test]
    fn legacy_scene_and_edit_recording() {
        let mut s = Scene::preset(0);
        let mut doc = serde_json::to_value(&s).unwrap();
        doc.as_object_mut().unwrap().remove("animation");
        assert!(
            serde_json::from_value::<Scene>(doc)
                .unwrap()
                .animation
                .tracks
                .is_empty()
        );
        s.key_parameter("/camera/distance", 0.0);
        let before = parameters(&s);
        s.camera.distance = 4.0;
        s.record_edits(&before, 12.0);
        assert_eq!(s.evaluated(12.0).unwrap().camera.distance, 4.0);
        assert_eq!(s.animation.tracks["/camera/distance"].keys().len(), 2);
    }
}
