//! The scene's timeline range and attribute labels. Keys live in the world document (Playa
//! channels, `curves::Track`), never on the scene.
use crate::scene::Scene;
use serde::{Deserialize, Serialize};

/// First / last frame and frame rate of a scene.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Animation {
    pub first: u32,
    pub last: u32,
    pub fps: f64,
}
impl Default for Animation {
    fn default() -> Self {
        Self {
            first: 0,
            last: 239,
            fps: 24.0,
        }
    }
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
        Ok(base)
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
