//! Global render controls using the same Playa attribute grid as world objects.
use crate::scene::Render;
use egui::Ui;
use egui_attr_grid::{AttrField, AttrGridHooks, AttrGridState, AttrValue, render_grid_with_config};
use egui_widgets_config::ValueEditorLayout;
use std::collections::HashSet;

/// Persistent descriptors and splitter state; repaint updates only scalar values.
pub struct RenderEditor {
    fields: [AttrField; 14],
    state: AttrGridState,
    mixed: HashSet<String>,
}
impl Default for RenderEditor {
    fn default() -> Self {
        const LABELS: [&str; 14] = [
            "Bounces",
            "March steps",
            "Hit epsilon",
            "Step factor",
            "Exposure EV",
            "Saturation",
            "Legacy Reinhard",
            "OIDN denoise",
            "Denoise every N samples",
            "Denoise guides",
            "Denoise quality",
            "Target samples",
            "Viewport scale",
            "Noise seed",
        ];
        let mut fields = std::array::from_fn(|index| {
            AttrField::new(LABELS[index], AttrValue::UInt(0)).with_order(index as f32)
        });
        for (index, options) in [
            (2, &["0.0001", "0.01", "log"][..]),
            (3, &["0.3", "1.0"][..]),
            (4, &["-6.0", "6.0"][..]),
            (5, &["0.0", "2.0"][..]),
            (12, &["0.25", "2.0"][..]),
        ] {
            fields[index].ui_options = options.iter().map(|value| (*value).into()).collect();
        }
        Self {
            fields,
            state: Default::default(),
            mixed: HashSet::new(),
        }
    }
}
struct RenderHooks {
    denoise: bool,
}
impl AttrGridHooks for RenderHooks {
    fn editor(
        &mut self,
        ui: &mut Ui,
        field: &mut AttrField,
        _: bool,
        _: &ValueEditorLayout,
    ) -> Option<bool> {
        let index = field.order as usize;
        let AttrValue::UInt(value) = &mut field.value else {
            return None;
        };
        let enabled = self.denoise || !matches!(index, 8..=10);
        Some(
            ui.add_enabled_ui(enabled, |ui| {
                if matches!(index, 9 | 10) {
                    let label = if index == 9 {
                        crate::denoise::Mode::ALL
                            .get(*value as usize)
                            .map(|mode| mode.label())
                            .unwrap_or("Color")
                    } else {
                        crate::denoise::Quality::ALL
                            .get(*value as usize)
                            .map(|quality| quality.label())
                            .unwrap_or("High")
                    };
                    let mut edited = false;
                    egui::ComboBox::from_id_salt(&field.key)
                        .selected_text(label)
                        .show_ui(ui, |ui| {
                            if index == 9 {
                                for (i, mode) in crate::denoise::Mode::ALL.iter().enumerate() {
                                    edited |= ui
                                        .selectable_value(value, i as u32, mode.label())
                                        .changed();
                                }
                            } else {
                                for (i, quality) in crate::denoise::Quality::ALL.iter().enumerate()
                                {
                                    edited |= ui
                                        .selectable_value(value, i as u32, quality.label())
                                        .changed();
                                }
                            }
                        });
                    edited
                } else {
                    let range = match index {
                        0 => 0..=16,
                        1 => 32..=2048,
                        8 => 0..=65536,
                        11 => 1..=65536,
                        _ => 0..=u32::MAX,
                    };
                    ui.add(egui::Slider::new(value, range).integer()).changed()
                }
            })
            .inner,
        )
    }
}
pub fn render(
    ui: &mut Ui,
    editor: &mut RenderEditor,
    r: &mut Render,
    target: &mut u32,
    resolution: &mut f32,
    seed: &mut u32,
    label_width: &mut f32,
    metrics: crate::ui_style::AttributeMetrics,
) {
    let metrics = metrics.normalized();
    metrics.apply(ui);
    let values = [
        AttrValue::UInt(r.max_bounces),
        AttrValue::UInt(r.max_steps),
        AttrValue::Float(r.hit_epsilon),
        AttrValue::Float(r.step_factor),
        AttrValue::Float(r.exposure_stops),
        AttrValue::Float(r.saturation),
        AttrValue::Bool(r.reinhard),
        AttrValue::Bool(r.denoise.enabled),
        AttrValue::UInt(r.denoise.interval),
        AttrValue::UInt(
            crate::denoise::Mode::ALL
                .iter()
                .position(|mode| *mode == r.denoise.mode)
                .unwrap_or(0) as u32,
        ),
        AttrValue::UInt(
            crate::denoise::Quality::ALL
                .iter()
                .position(|quality| *quality == r.denoise.quality)
                .unwrap_or(0) as u32,
        ),
        AttrValue::UInt(*target),
        AttrValue::Float(*resolution),
        AttrValue::UInt(*seed),
    ];
    for (field, value) in editor.fields.iter_mut().zip(values) {
        field.value = value;
    }
    let config = metrics.grid_config();
    let mut hooks = RenderHooks {
        denoise: r.denoise.enabled,
    };
    // Only the label column stores a width; the value column flexes.
    editor.state.table.widths.resize(1, 0.0);
    editor.state.table.widths[0] = *label_width;
    let changed = render_grid_with_config(
        ui,
        &mut editor.fields,
        &mut editor.state,
        &editor.mixed,
        &config,
        &mut hooks,
    );
    if let Some(width) = editor.state.table.widths.first() {
        *label_width = *width;
    }
    for (key, value) in changed {
        match (key.as_str(), value) {
            ("Bounces", AttrValue::UInt(value)) => r.max_bounces = value,
            ("March steps", AttrValue::UInt(value)) => r.max_steps = value,
            ("Hit epsilon", AttrValue::Float(value)) => r.hit_epsilon = value,
            ("Step factor", AttrValue::Float(value)) => r.step_factor = value,
            ("Exposure EV", AttrValue::Float(value)) => r.exposure_stops = value,
            ("Saturation", AttrValue::Float(value)) => r.saturation = value,
            ("Legacy Reinhard", AttrValue::Bool(value)) => r.reinhard = value,
            ("OIDN denoise", AttrValue::Bool(value)) => r.denoise.enabled = value,
            ("Denoise every N samples", AttrValue::UInt(value)) => r.denoise.interval = value,
            ("Denoise guides", AttrValue::UInt(value)) => {
                if let Some(mode) = crate::denoise::Mode::ALL.get(value as usize) {
                    r.denoise.mode = *mode;
                }
            }
            ("Denoise quality", AttrValue::UInt(value)) => {
                if let Some(quality) = crate::denoise::Quality::ALL.get(value as usize) {
                    r.denoise.quality = *quality;
                }
            }
            ("Target samples", AttrValue::UInt(value)) => *target = value,
            ("Viewport scale", AttrValue::Float(value)) => *resolution = value,
            ("Noise seed", AttrValue::UInt(value)) => *seed = value,
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opening_render_section_preserves_shared_width_and_idle_storage() {
        let ctx = egui::Context::default();
        let mut editor = RenderEditor::default();
        let mut settings = crate::scene::Scene::gallery().remove(0).render;
        let original = settings.clone();
        let mut target = 1024;
        let mut resolution = 1.0;
        let mut seed = 0;
        let mut shared_width = 210.0;
        let mut storage = None;
        for panel_width in [430.0, 430.0, 300.0, 430.0] {
            let _ = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(panel_width, 800.0),
                    )),
                    ..Default::default()
                },
                |ui| {
                    render(
                        ui,
                        &mut editor,
                        &mut settings,
                        &mut target,
                        &mut resolution,
                        &mut seed,
                        &mut shared_width,
                        Default::default(),
                    )
                },
            );
            assert_eq!(editor.state.table.widths.len(), 1);
            assert!((shared_width - 210.0).abs() < 0.01);
            let current = editor.state.table.widths.as_ptr();
            if let Some(previous) = storage {
                assert_eq!(current, previous);
            }
            storage = Some(current);
        }
        assert_eq!(settings, original);
        assert_eq!((target, resolution, seed), (1024, 1.0, 0));
    }
}
