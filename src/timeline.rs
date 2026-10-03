//! Test-only transport for legacy scene animation compatibility.
use crate::scene::Scene;

pub struct Timeline {
    pub frame: u32,
    pub playing: bool,
    pub looping: bool,
    elapsed: f64,
}
impl Default for Timeline {
    fn default() -> Self {
        Self {
            frame: 0,
            playing: false,
            looping: true,
            elapsed: 0.0,
        }
    }
}
impl Timeline {
    pub fn reset(&mut self, scene: &Scene) {
        *self = Self::default();
        self.frame = scene.animation.first;
    }
    pub fn advance(&mut self, dt: f64, scene: &Scene) {
        if !self.playing {
            self.elapsed = 0.0;
            return;
        }
        let a = &scene.animation;
        if !a.fps.is_finite() || a.fps <= 0.0 {
            self.playing = false;
            return;
        }
        let first = u64::from(a.first);
        let last = u64::from(a.last.max(a.first));
        self.elapsed += dt.clamp(0.0, 0.25) * a.fps;
        let steps = self.elapsed.floor() as u64;
        self.elapsed -= steps as f64;
        if steps == 0 {
            return;
        }
        let next = u64::from(self.frame) + steps;
        if next > last {
            if self.looping {
                self.frame = (first + (next - first) % (last - first + 1)) as u32;
            } else {
                self.frame = last as u32;
                self.playing = false;
            }
        } else {
            self.frame = next as u32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn playback_steps_and_wraps_inclusive_range() {
        let mut scene = Scene::preset(0);
        scene.animation.first = 10;
        scene.animation.last = 12;
        let mut timeline = Timeline::default();
        timeline.reset(&scene);
        timeline.playing = true;
        timeline.advance(2.0 / 24.0, &scene);
        assert_eq!(timeline.frame, 12);
        timeline.advance(1.0 / 24.0, &scene);
        assert_eq!(timeline.frame, 10);
        timeline.looping = false;
        timeline.advance(3.0 / 24.0, &scene);
        assert_eq!(timeline.frame, 12);
        assert!(!timeline.playing);
    }
}
