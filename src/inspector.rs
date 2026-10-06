//! Global render controls using the same Playa attribute grid as world objects. Every row is
//! one `Row`: its label, the world attribute it mirrors (hint, slider span and hard limits come
//! from the world's tables, `world::attribute_hint` / `world::path_slider_options` /
//! `world::attribute_range`), and how it reads and writes the settings.
use crate::scene::Render;
use egui::Ui;
use egui_attr_grid::{AttrField, AttrGridHooks, AttrGridState, AttrValue, render_grid_with_config};
use egui_widgets_config::ValueEditorLayout;
use std::collections::HashSet;

/// One Render settings row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Row {
    Bounces,
    MarchSteps,
    GlassProbes,
    HitEpsilon,
    StepFactor,
    Exposure,
    Saturation,
    Reinhard,
    Denoise,
    DenoiseInterval,
    DenoiseMode,
    DenoiseQuality,
    TargetSamples,
    ViewportScale,
    Seed,
}

/// What the rows edit: the scene's render settings and the viewport's own.
struct Settings<'a> {
    render: &'a mut Render,
    target: &'a mut u32,
    resolution: &'a mut f32,
    seed: &'a mut u32,
}

impl Row {
    const ALL: [Row; 15] = [
        Row::Bounces,
        Row::MarchSteps,
        Row::GlassProbes,
        Row::HitEpsilon,
        Row::StepFactor,
        Row::Exposure,
        Row::Saturation,
        Row::Reinhard,
        Row::Denoise,
        Row::DenoiseInterval,
        Row::DenoiseMode,
        Row::DenoiseQuality,
        Row::TargetSamples,
        Row::ViewportScale,
        Row::Seed,
    ];

    fn label(self) -> &'static str {
        match self {
            Row::Bounces => "Bounces",
            Row::MarchSteps => "March steps",
            Row::GlassProbes => "Glass interior probes",
            Row::HitEpsilon => "Hit epsilon",
            Row::StepFactor => "Step factor",
            Row::Exposure => "Exposure EV",
            Row::Saturation => "Saturation",
            Row::Reinhard => "Legacy Reinhard",
            Row::Denoise => "OIDN denoise",
            Row::DenoiseInterval => "Denoise every N samples",
            Row::DenoiseMode => "Denoise guides",
            Row::DenoiseQuality => "Denoise quality",
            Row::TargetSamples => "Target samples",
            Row::ViewportScale => "Viewport scale",
            Row::Seed => "Noise seed",
        }
    }

    /// The world attribute this row mirrors; None for the viewport's own settings.
    fn path(self) -> Option<&'static str> {
        Some(match self {
            Row::Bounces => "/render/max_bounces",
            Row::MarchSteps => "/render/max_steps",
            Row::GlassProbes => "/render/glass_probes",
            Row::HitEpsilon => "/render/hit_epsilon",
            Row::StepFactor => "/render/step_factor",
            Row::Exposure => "/render/exposure_stops",
            Row::Saturation => "/render/saturation",
            Row::Reinhard => "/render/reinhard",
            Row::Denoise => "/render/denoise/enabled",
            Row::DenoiseInterval => "/render/denoise/interval",
            Row::DenoiseMode => "/render/denoise/mode",
            Row::DenoiseQuality => "/render/denoise/quality",
            Row::TargetSamples | Row::ViewportScale | Row::Seed => return None,
        })
    }

    fn hint(self) -> Option<&'static str> {
        match self {
            Row::TargetSamples => Some("Samples per pixel the viewport renders before it stops."),
            Row::ViewportScale => Some("Viewport resolution relative to the window: lower renders faster."),
            Row::Seed => Some("Random sequence of the samples; a new seed gives a different noise pattern."),
            _ => self.path().and_then(crate::world::attribute_hint),
        }
    }

    /// The grid's slider span: the world's for a world attribute.
    fn ui_options(self) -> Vec<String> {
        match (self, self.path()) {
            (Row::ViewportScale, _) => vec!["0.25".into(), "2.0".into()],
            (Row::DenoiseMode | Row::DenoiseQuality, _) => Vec::new(),
            (_, Some(path)) => crate::world::path_slider_options(path, self.integer()),
            (_, None) => Vec::new(),
        }
    }

    fn integer(self) -> bool {
        matches!(self.get_default(), AttrValue::UInt(_))
    }

    /// The row's value in a preset's settings: its value kind (and the field's first value).
    fn get_default(self) -> AttrValue {
        let mut render = crate::scene::Scene::preset(0).render;
        let (mut target, mut resolution, mut seed) = (0, 1.0, 0);
        self.get(&Settings {
            render: &mut render,
            target: &mut target,
            resolution: &mut resolution,
            seed: &mut seed,
        })
    }

    fn get(self, s: &Settings) -> AttrValue {
        let r = &*s.render;
        match self {
            Row::Bounces => AttrValue::UInt(r.max_bounces),
            Row::MarchSteps => AttrValue::UInt(r.max_steps),
            Row::GlassProbes => AttrValue::UInt(r.glass_probes),
            Row::HitEpsilon => AttrValue::Float(r.hit_epsilon),
            Row::StepFactor => AttrValue::Float(r.step_factor),
            Row::Exposure => AttrValue::Float(r.exposure_stops),
            Row::Saturation => AttrValue::Float(r.saturation),
            Row::Reinhard => AttrValue::Bool(r.reinhard),
            Row::Denoise => AttrValue::Bool(r.denoise.enabled),
            Row::DenoiseInterval => AttrValue::UInt(r.denoise.interval),
            Row::DenoiseMode => AttrValue::UInt(
                crate::denoise::Mode::ALL.iter().position(|m| *m == r.denoise.mode).unwrap_or(0) as u32,
            ),
            Row::DenoiseQuality => AttrValue::UInt(
                crate::denoise::Quality::ALL.iter().position(|q| *q == r.denoise.quality).unwrap_or(0) as u32,
            ),
            Row::TargetSamples => AttrValue::UInt(*s.target),
            Row::ViewportScale => AttrValue::Float(*s.resolution),
            Row::Seed => AttrValue::UInt(*s.seed),
        }
    }

    /// Write an edited value, held to the world attribute's hard limits.
    fn set(self, s: &mut Settings, value: AttrValue) {
        let limit = |v: f64| match self.path().and_then(crate::world::attribute_range) {
            Some((min, max)) => v.clamp(min, max),
            None => v,
        };
        let r = &mut *s.render;
        match (self, value) {
            (Row::Bounces, AttrValue::UInt(v)) => r.max_bounces = limit(v.into()) as u32,
            (Row::MarchSteps, AttrValue::UInt(v)) => r.max_steps = limit(v.into()) as u32,
            (Row::GlassProbes, AttrValue::UInt(v)) => r.glass_probes = limit(v.into()) as u32,
            (Row::HitEpsilon, AttrValue::Float(v)) => r.hit_epsilon = limit(v.into()) as f32,
            (Row::StepFactor, AttrValue::Float(v)) => r.step_factor = limit(v.into()) as f32,
            (Row::Exposure, AttrValue::Float(v)) => r.exposure_stops = limit(v.into()) as f32,
            (Row::Saturation, AttrValue::Float(v)) => r.saturation = limit(v.into()) as f32,
            (Row::Reinhard, AttrValue::Bool(v)) => r.reinhard = v,
            (Row::Denoise, AttrValue::Bool(v)) => r.denoise.enabled = v,
            (Row::DenoiseInterval, AttrValue::UInt(v)) => r.denoise.interval = limit(v.into()) as u32,
            (Row::DenoiseMode, AttrValue::UInt(v)) => {
                if let Some(mode) = crate::denoise::Mode::ALL.get(v as usize) {
                    r.denoise.mode = *mode;
                }
            }
            (Row::DenoiseQuality, AttrValue::UInt(v)) => {
                if let Some(quality) = crate::denoise::Quality::ALL.get(v as usize) {
                    r.denoise.quality = *quality;
                }
            }
            (Row::TargetSamples, AttrValue::UInt(v)) => *s.target = v.max(1),
            (Row::ViewportScale, AttrValue::Float(v)) => *s.resolution = v,
            (Row::Seed, AttrValue::UInt(v)) => *s.seed = v,
            _ => {}
        }
    }

    fn of(field: &AttrField) -> Option<Row> {
        Row::ALL.get(field.order as usize).copied()
    }
}

/// Persistent descriptors and splitter state; repaint updates only scalar values.
pub struct RenderEditor {
    fields: [AttrField; 15],
    state: AttrGridState,
    mixed: HashSet<String>,
}
impl Default for RenderEditor {
    fn default() -> Self {
        let fields = std::array::from_fn(|index| {
            let row = Row::ALL[index];
            let mut field = AttrField::new(row.label(), row.get_default())
                .with_order(index as f32)
                .with_ui_options(row.ui_options());
            field.hint = row.hint().map(str::to_owned);
            // Render rows reset to what a new scene starts with; the viewport's own rows have no
            // canonical default.
            if row.path().is_some() {
                field = field.with_default(row.get_default());
            }
            field
        });
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
    /// Only the denoise guides and quality are choices (combo boxes) and the denoise rows go
    /// inactive with denoising off; every other row is the grid's own editor.
    fn editor(
        &mut self,
        ui: &mut Ui,
        field: &mut AttrField,
        _: bool,
        _: &ValueEditorLayout,
        _: &egui_attr_grid::EditorCtx,
    ) -> Option<bool> {
        let row = Row::of(field)?;
        let choice: &[&str] = match row {
            Row::DenoiseMode => &crate::denoise::Mode::ALL.map(|m| m.label()),
            Row::DenoiseQuality => &crate::denoise::Quality::ALL.map(|q| q.label()),
            _ => return None,
        };
        let AttrValue::UInt(value) = &mut field.value else {
            return None;
        };
        Some(
            ui.add_enabled_ui(self.denoise, |ui| {
                let mut edited = false;
                egui::ComboBox::from_id_salt(&field.key)
                    .selected_text(choice.get(*value as usize).copied().unwrap_or_default())
                    .show_ui(ui, |ui| {
                        for (i, label) in choice.iter().enumerate() {
                            edited |= ui.selectable_value(value, i as u32, *label).changed();
                        }
                    });
                edited
            })
            .inner,
        )
    }

    fn disabled(&self, field: &AttrField) -> Option<String> {
        match Row::of(field)? {
            Row::DenoiseInterval | Row::DenoiseMode | Row::DenoiseQuality if !self.denoise => {
                Some("Denoising is off".into())
            }
            _ => None,
        }
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
    metrics: egui_attr_grid::AttrMetrics,
) {
    let metrics = metrics.normalized();
    metrics.apply(ui);
    let mut settings = Settings {
        render: r,
        target,
        resolution,
        seed,
    };
    for (field, row) in editor.fields.iter_mut().zip(Row::ALL) {
        field.value = row.get(&settings);
    }
    let config = crate::world_ui::grid_config(metrics);
    let mut hooks = RenderHooks {
        denoise: settings.render.denoise.enabled,
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
        if let Some(row) = Row::ALL.into_iter().find(|row| row.label() == key) {
            row.set(&mut settings, value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each row's value round-trips through its own field, and the world rows take their hint
    /// and slider span from the world's tables (March steps reaches the 4096 default).
    #[test]
    fn rows_read_write_and_explain_themselves_from_the_world_tables() {
        let mut render = crate::scene::Scene::gallery().remove(0).render;
        let (mut target, mut resolution, mut seed) = (64, 1.0, 3);
        let mut s = Settings {
            render: &mut render,
            target: &mut target,
            resolution: &mut resolution,
            seed: &mut seed,
        };
        for row in Row::ALL {
            let value = row.get(&s);
            row.set(&mut s, value.clone());
            assert_eq!(row.get(&s), value, "{row:?}");
            assert!(row.hint().is_some(), "{row:?} has no hint");
        }
        Row::GlassProbes.set(&mut s, AttrValue::UInt(0));
        assert_eq!(s.render.glass_probes, 1, "the world's hard limit holds here too");
        let span = Row::MarchSteps.ui_options();
        let max: f64 = span[1].parse().unwrap();
        assert!(max >= f64::from(crate::scene::DEFAULT_MAX_STEPS), "{span:?}");
    }

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
