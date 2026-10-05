//! Camera slots: five viewport-toolbar buttons that copy and paste the viewport camera (the
//! mouse mapping is `hotkeys::slot_click`, shared with the colour presets). A paste only assigns the camera; the viewport's ordinary edit path
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
    /// A recall on a slot that holds no camera yet.
    Empty(usize),
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

    /// The strip of slot buttons: one click stores `camera`, the other restores into it
    /// (`hotkeys::slot_click`; `swap` exchanges the buttons).
    pub fn ui(&mut self, ui: &mut egui::Ui, camera: &mut Camera, swap: bool) -> Option<SlotAction> {
        use crate::hotkeys::{SlotClick, slot_click, slot_hint};
        let mut action = None;
        let right_to_left = ui.layout().prefer_right_to_left();
        if !right_to_left {
            ui.label("CamClip:");
        }
        // Keep 1..5 left to right inside a right-aligned (right-to-left) toolbar block.
        let order: Vec<usize> = if right_to_left {
            (0..COUNT).rev().collect()
        } else {
            (0..COUNT).collect()
        };
        for index in order {
            let stored = self.slots[index];
            let mapping = slot_hint(swap, "copy the current camera", "paste this camera");
            let hint = match stored {
                Some(c) => format!(
                    "CamClip {}\n{mapping}\n\nTarget {:.3} {:.3} {:.3}\nYaw {:.1}° Pitch {:.1}° Roll {:.1}°\nDistance {:.3} · FOV {:.1}°",
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
                None => format!(
                    "CamClip {} (empty)\n{}",
                    index + 1,
                    slot_hint(swap, "copy the current camera", "nothing to paste yet")
                ),
            };
            let response = ui
                .add(egui::Button::new((index + 1).to_string()).selected(stored.is_some()))
                .on_hover_text(hint);
            match slot_click(&response, swap) {
                Some(SlotClick::Store) => {
                    self.store(index, camera);
                    action = Some(SlotAction::Stored(index));
                }
                Some(SlotClick::Recall) => {
                    action = Some(if self.restore(index, camera) {
                        SlotAction::Restored(index)
                    } else {
                        SlotAction::Empty(index)
                    });
                }
                None => {}
            }
        }
        if right_to_left {
            ui.label("CamClip:");
        }
        action
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_clicks_follow_the_shared_mapping_in_both_layouts() {
        use egui::PointerButton::{Primary, Secondary};
        for swap in [false, true] {
            let (store, recall) = if swap { (Primary, Secondary) } else { (Secondary, Primary) };
            let ctx = egui::Context::default();
            let mut slots = CameraSlots::default();
            let mut camera = crate::scene::Scene::preset(crate::params::FAMILY_BULB).camera;
            let frame = |slots: &mut CameraSlots, camera: &mut Camera, events: Vec<egui::Event>| {
                let mut action = None;
                let output = ctx.run_ui(
                    egui::RawInput {
                        screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(400.0, 100.0))),
                        events,
                        ..Default::default()
                    },
                    |root| {
                        egui::CentralPanel::default().show(root, |ui| {
                            ui.horizontal(|ui| action = slots.ui(ui, camera, swap));
                        });
                    },
                );
                (action, output)
            };
            let click = |pos: egui::Pos2, button| {
                vec![
                    egui::Event::PointerMoved(pos),
                    egui::Event::PointerButton { pos, button, pressed: true, modifiers: Default::default() },
                    egui::Event::PointerButton { pos, button, pressed: false, modifiers: Default::default() },
                ]
            };
            frame(&mut slots, &mut camera, vec![]);
            let (_, output) = frame(&mut slots, &mut camera, vec![]);
            let pos = output
                .shapes
                .iter()
                .find_map(|s| match &s.shape {
                    egui::epaint::Shape::Text(t) if t.galley.text() == "2" => {
                        Some(t.pos + t.galley.rect.center().to_vec2())
                    }
                    _ => None,
                })
                .expect("slot 2");
            // Recalling an empty slot reports it and leaves the camera.
            let before = camera;
            let (action, _) = frame(&mut slots, &mut camera, click(pos, recall));
            assert_eq!(action, Some(SlotAction::Empty(1)), "swap={swap}");
            assert_eq!(camera, before);
            let (action, _) = frame(&mut slots, &mut camera, click(pos, store));
            assert_eq!(action, Some(SlotAction::Stored(1)), "swap={swap}");
            let stored = camera;
            camera.yaw_degrees += 30.0;
            let (action, _) = frame(&mut slots, &mut camera, click(pos, recall));
            assert_eq!(action, Some(SlotAction::Restored(1)), "swap={swap}");
            assert_eq!(camera, stored);
        }
    }

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
