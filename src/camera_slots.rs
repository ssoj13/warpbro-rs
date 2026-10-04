//! Camera slots: five viewport-toolbar buttons that copy (left click) and paste (right click)
//! the viewport camera. A paste only assigns the camera; the viewport's ordinary edit path
//! (`WorldEditor::navigate_camera`) authors it, so Auto Key, undo gestures and locks apply.
//! The slots are an application clipboard persisted with the settings, not part of a scene.
use crate::scene::Camera;
use serde::{Deserialize, Serialize};

pub const COUNT: usize = 5;

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CameraSlots {
    pub slots: [Option<Camera>; COUNT],
}

/// What a click on the slot strip did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotAction {
    Stored(usize),
    Restored(usize),
}

impl CameraSlots {
    pub fn store(&mut self, index: usize, camera: &Camera) {
        if let Some(slot) = self.slots.get_mut(index) {
            *slot = Some(*camera);
        }
    }
    /// Write slot `index` into `camera`; false for an empty slot.
    pub fn restore(&self, index: usize, camera: &mut Camera) -> bool {
        match self.slots.get(index).copied().flatten() {
            Some(stored) => {
                *camera = stored;
                true
            }
            None => false,
        }
    }

    /// The strip of slot buttons: LMB stores `camera`, RMB restores into it.
    pub fn ui(&mut self, ui: &mut egui::Ui, camera: &mut Camera) -> Option<SlotAction> {
        let mut action = None;
        // Keep 1..5 left to right inside a right-aligned (right-to-left) toolbar block.
        let order: Vec<usize> = if ui.layout().prefer_right_to_left() {
            (0..COUNT).rev().collect()
        } else {
            (0..COUNT).collect()
        };
        for index in order {
            let stored = self.slots[index];
            let hint = match stored {
                Some(c) => format!(
                    "Camera {}\nLMB: store the current camera · RMB: restore\n\nTarget {:.3} {:.3} {:.3}\nYaw {:.1}° Pitch {:.1}° Roll {:.1}°\nDistance {:.3} · FOV {:.1}°",
                    index + 1,
                    c.target[0],
                    c.target[1],
                    c.target[2],
                    c.yaw_degrees,
                    c.pitch_degrees,
                    c.roll_degrees,
                    c.distance,
                    c.fov_y_degrees
                ),
                None => format!("Camera {} (empty)\nLMB: store the current camera", index + 1),
            };
            let response = ui
                .add(egui::Button::new((index + 1).to_string()).selected(stored.is_some()))
                .on_hover_text(hint);
            if response.clicked() {
                self.store(index, camera);
                action = Some(SlotAction::Stored(index));
            } else if response.secondary_clicked() && self.restore(index, camera) {
                action = Some(SlotAction::Restored(index));
            }
        }
        action
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slots_copy_and_paste_the_whole_camera_and_survive_serialization() {
        let mut slots = CameraSlots::default();
        let mut camera = crate::scene::Scene::preset(crate::params::FAMILY_BULB).camera;
        let original = camera;
        assert!(!slots.restore(2, &mut camera), "an empty slot leaves the camera");
        assert_eq!(camera, original);
        camera.yaw_degrees = 37.0;
        camera.target = [0.1, 0.2, 0.3];
        slots.store(2, &camera);
        let stored = camera;
        camera = original;
        assert!(slots.restore(2, &mut camera));
        assert_eq!(camera, stored);
        slots.store(COUNT, &camera); // out of range is ignored
        let json = serde_json::to_string(&slots).unwrap();
        assert_eq!(serde_json::from_str::<CameraSlots>(&json).unwrap(), slots);
    }
}
