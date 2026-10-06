//! Scene parameter curves. JSON pointers are stable parameter identities, including
//! formula variants and vector components; rendering still receives a typed Scene.
use crate::scene::Scene;
use box_rs::{BoxValue, RustBox};
use curves::Tan;
use curves::legacy::CurveKind;
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

#[cfg(test)]
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

/// One scene key. `interpolation` shapes the segment that STARTS at this key (the same meaning the
/// old per-key `CurveKind` had); the arriving side of the next key follows it, see [`Track::rebuild`].
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Key {
    pub frame: f64,
    pub value: Value,
    #[serde(rename = "tan")]
    pub interpolation: Tan,
}

/// On-disk key: current scenes store `tan`; scenes saved before the curves `Track` model store the
/// old `interpolation` kind, which is mapped by [`legacy_tan`].
#[derive(Deserialize)]
struct StoredKey {
    frame: f64,
    value: Value,
    tan: Option<Tan>,
    interpolation: Option<CurveKind>,
}

/// The `Tan` that evaluates like an old key kind. Exact for Linear / Smooth (smoothstep = zero-slope
/// Hermite) / Step (hold-left); Hermite is exact because old keys carried no handles (zero slope);
/// Bezier with its zero default handles becomes the same flat Hermite; CatmullRom / Monotone become
/// the auto kinds (curves KEYS.md section 5, "semantic only").
fn legacy_tan(kind: CurveKind) -> Tan {
    match kind {
        CurveKind::Linear => Tan::Linear,
        CurveKind::Smooth | CurveKind::Hermite | CurveKind::Bezier => Tan::Flat,
        CurveKind::Step => Tan::Constant,
        CurveKind::CatmullRom => Tan::CatmullRom,
        CurveKind::Monotone => Tan::Smooth,
    }
}

/// Keys sorted by frame plus one `curves::Track<f64>` per numeric leaf of the value (derived from
/// `keys`, rebuilt on every edit). Non-numeric parts of a value (strings, bools, objects, enums)
/// hold the LEFT key's value; a track whose keys disagree on the numeric leaf count is entirely
/// hold-left.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Track {
    keys: Vec<Key>,
    #[serde(skip)]
    comps: Vec<curves::Track<f64>>,
}

impl PartialEq for Track {
    fn eq(&self, other: &Self) -> bool {
        self.keys == other.keys
    }
}

impl<'de> Deserialize<'de> for Track {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Stored {
            keys: Vec<StoredKey>,
        }
        let mut track = Self::default();
        for k in Stored::deserialize(d)?.keys {
            let interpolation = k
                .tan
                .or(k.interpolation.map(legacy_tan))
                .ok_or_else(|| serde::de::Error::custom("Animation key without interpolation"))?;
            let key = Key {
                frame: k.frame,
                value: k.value,
                interpolation,
            };
            if !track.upsert(key) {
                return Err(serde::de::Error::custom("Invalid animation key"));
            }
        }
        Ok(track)
    }
}

/// Numeric leaves of a value in depth-first order (numbers inside arrays only: objects hold).
fn leaves(v: &Value, out: &mut Vec<f64>) {
    match v {
        Value::Number(n) => out.push(n.as_f64().unwrap_or(0.0)),
        Value::Array(a) => a.iter().for_each(|v| leaves(v, out)),
        _ => {}
    }
}

/// `like` with its numeric leaves replaced by `vals` (same order as [`leaves`]); integer leaves
/// stay integers (rounded, unsigned ones clamped to `u32`).
fn fill(like: &Value, vals: &mut impl Iterator<Item = f64>) -> Value {
    match like {
        Value::Number(n) => {
            let v = vals.next().unwrap_or(0.0);
            if n.is_u64() {
                Value::from(v.round().clamp(0.0, u32::MAX as f64) as u64)
            } else if n.is_i64() {
                Value::from(v.round() as i64)
            } else {
                Value::from(v)
            }
        }
        Value::Array(a) => Value::Array(a.iter().map(|v| fill(v, vals)).collect()),
        other => other.clone(),
    }
}

impl Track {
    pub fn keys(&self) -> &[Key] {
        &self.keys
    }

    /// Rebuilds the per-leaf tracks. Segment `i` takes its kind from key `i` on both its ends
    /// (`set_seg`); a Constant (hold) kind only sets the out side, so the segment arriving at the
    /// next key is never turned into a reverse step (KEYS.md 3.3).
    fn rebuild(&mut self) {
        self.comps.clear();
        let cols: Vec<Vec<f64>> = self
            .keys
            .iter()
            .map(|k| {
                let mut col = Vec::new();
                leaves(&k.value, &mut col);
                col
            })
            .collect();
        let n = cols.first().map_or(0, Vec::len);
        if cols.iter().any(|c| c.len() != n) {
            return;
        }
        let segs = self.keys.len().saturating_sub(1);
        for leaf in 0..n {
            let mut t = curves::Track::new();
            let built = self
                .keys
                .iter()
                .zip(&cols)
                .try_for_each(|(k, c)| t.add(k.frame, c[leaf], k.interpolation).map(drop))
                .and_then(|()| {
                    (0..segs)
                        .filter(|&i| self.keys[i].interpolation != Tan::Constant)
                        .try_for_each(|i| t.set_seg(i, self.keys[i].interpolation))
                });
            // Keys are validated finite and strictly increasing on entry; only an overflowing
            // derived value can fail here, and then the whole track holds its left values.
            if built.is_err() {
                self.comps.clear();
                return;
            }
            self.comps.push(t);
        }
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
        self.rebuild();
        true
    }
    /// Auto-key: an existing key keeps its interpolation, a new one is linear.
    #[cfg(test)]
    pub fn set(&mut self, frame: f64, value: Value) {
        let interpolation = self
            .keys
            .iter()
            .find(|k| k.frame == frame)
            .map_or(Tan::Linear, |k| k.interpolation);
        self.upsert(Key {
            frame,
            value,
            interpolation,
        });
    }
    /// Moves keys by re-upserting them (not `curves::Track::shift`): `comps` is a cache rebuilt on
    /// every edit, so the key list is the single source of truth.
    #[cfg(test)]
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
        let left = if i == 0 { first } else { &self.keys[i - 1] };
        if self.comps.is_empty() {
            return Some(left.value.clone());
        }
        Some(fill(&left.value, &mut self.comps.iter().map(|c| c.eval(frame))))
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
#[cfg(test)]
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
    #[cfg(test)]
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
    #[cfg(test)]
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
    #[test]
    fn pre_migration_scene_keys_evaluate_as_before() {
        // Saved before the curves Track model: one `interpolation` kind per key, values as in the file.
        let old: Track = serde_json::from_value(serde_json::json!({ "keys": [
            { "frame": 0.0,  "value": [0.0, 10.0], "interpolation": "Linear" },
            { "frame": 10.0, "value": [10.0, 20.0], "interpolation": "Smooth" },
            { "frame": 20.0, "value": [20.0, 40.0], "interpolation": "Step" },
            { "frame": 30.0, "value": [30.0, 50.0], "interpolation": "Linear" },
        ]}))
        .unwrap();
        let at = |f| old.sample(f).unwrap();
        let close = |v: Value, want: [f64; 2]| {
            (0..2).for_each(|i| assert!((v[i].as_f64().unwrap() - want[i]).abs() < 1e-9, "{v} vs {want:?}"));
        };
        // Linear segment: straight line.
        close(at(5.0), [5.0, 15.0]);
        // Smooth segment: smoothstep 3t^2 - 2t^3 on the segment, the legacy `Smooth`.
        let s = 0.25_f64 * 0.25 * (3.0 - 2.0 * 0.25);
        let v = at(12.5);
        assert!((v[0].as_f64().unwrap() - (10.0 + 10.0 * s)).abs() < 1e-9);
        assert!((v[1].as_f64().unwrap() - (20.0 + 20.0 * s)).abs() < 1e-9);
        // Step holds the LEFT value and jumps at the next key; the segment after it is untouched.
        assert_eq!(at(19.999), serde_json::json!([20.0, 40.0]));
        assert_eq!(at(20.0), serde_json::json!([20.0, 40.0]));
        close(at(25.0), [25.0, 45.0]);
        // Constant extrapolation on both ends.
        assert_eq!(at(-5.0), serde_json::json!([0.0, 10.0]));
        assert_eq!(at(99.0), serde_json::json!([30.0, 50.0]));
        assert_eq!(
            old.keys().iter().map(|k| k.interpolation).collect::<Vec<_>>(),
            [Tan::Linear, Tan::Flat, Tan::Constant, Tan::Linear]
        );
        // Re-saved scenes use the new `tan` field and read back identically.
        let saved = serde_json::to_string(&old).unwrap();
        assert!(saved.contains("\"tan\"") && !saved.contains("interpolation"));
        assert_eq!(serde_json::from_str::<Track>(&saved).unwrap(), old);
    }
    #[test]
    fn legacy_keys_match_the_curves_legacy_loader() {
        use curves::legacy::LegacyKey;
        let kinds = [CurveKind::Linear, CurveKind::Smooth, CurveKind::Step, CurveKind::Linear];
        let vals = [0.0, 10.0, 25.0, 4.0];
        let json: Vec<_> = kinds
            .iter()
            .zip(vals)
            .enumerate()
            .map(|(i, (k, v))| {
                serde_json::json!({ "frame": i as f64 * 7.0, "value": v, "interpolation": k })
            })
            .collect();
        let ours: Track = serde_json::from_value(serde_json::json!({ "keys": json })).unwrap();
        let reference = curves::Track::from_legacy(kinds.iter().zip(vals).enumerate().map(
            |(i, (&interp, v))| LegacyKey {
                t: i as f64 * 7.0,
                v,
                interp,
                tan_in: Default::default(),
                tan_out: Default::default(),
            },
        ))
        .unwrap();
        for i in 0..=210 {
            let f = -3.0 + f64::from(i) * 0.1;
            let got = ours.sample(f).unwrap().as_f64().unwrap();
            assert!((got - reference.eval(f)).abs() < 1e-12, "frame {f}");
        }
    }
    #[test]
    fn non_numeric_parts_hold_the_left_key_and_ints_stay_ints() {
        let mut t = Track::default();
        let key = |frame, value| Key { frame, value, interpolation: Tan::Linear };
        assert!(t.upsert(key(0.0, serde_json::json!({ "n": 0, "mode": "a" }))));
        assert!(t.upsert(key(10.0, serde_json::json!({ "n": 10, "mode": "b" }))));
        // Objects hold the left key whole.
        assert_eq!(t.sample(9.0).unwrap()["mode"], "a");
        let mut t = Track::default();
        assert!(t.upsert(key(0.0, serde_json::json!([1, "a", 0.0]))));
        assert!(t.upsert(key(10.0, serde_json::json!([5, "b", 10.0]))));
        let mid = t.sample(5.0).unwrap();
        assert_eq!((&mid[0], &mid[1]), (&serde_json::json!(3), &serde_json::json!("a")));
        assert!((mid[2].as_f64().unwrap() - 5.0).abs() < 1e-9);
        assert_eq!(t.sample(10.0).unwrap(), serde_json::json!([5, "b", 10.0]));
    }
}
