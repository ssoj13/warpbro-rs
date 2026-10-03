use egui_widgets_config::ValueEditorLayout;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Weak};

#[derive(Clone)]
struct PreparedStyleEntry {
    metrics: AttributeMetrics,
    source: Weak<egui::Style>,
    prepared: Arc<egui::Style>,
}

#[derive(Clone, Default)]
struct PreparedStyleCache {
    entries: [Option<PreparedStyleEntry>; 4],
    next: usize,
}

impl PreparedStyleCache {
    fn get(&mut self, metrics: AttributeMetrics, base: &Arc<egui::Style>) -> Arc<egui::Style> {
        // Resolve an already prepared child back to its original style so larger
        // field heights can restore the original font and reuse earlier cache entries.
        let source = self
            .entries
            .iter()
            .flatten()
            .find(|entry| Arc::ptr_eq(&entry.prepared, base))
            .and_then(|entry| entry.source.upgrade())
            .unwrap_or_else(|| base.clone());
        if let Some(entry) = self.entries.iter().flatten().find(|entry| {
            entry.metrics == metrics
                && (Arc::ptr_eq(&entry.prepared, base)
                    || entry
                        .source
                        .upgrade()
                        .is_some_and(|cached_source| Arc::ptr_eq(&cached_source, &source)))
        }) {
            return entry.prepared.clone();
        }
        let mut style = (*source).clone();
        let mut font = style
            .override_font_id
            .clone()
            .unwrap_or_else(|| egui::TextStyle::Body.resolve(&style));
        font.size = font.size.min((metrics.field_height - 2.0).max(6.0));
        style.override_font_id = Some(font);
        style.spacing.button_padding = egui::vec2(2.0, 0.0);
        style.spacing.interact_size = egui::vec2(metrics.numeric_width, metrics.field_height);
        style.spacing.item_spacing = egui::vec2(metrics.component_gap, metrics.row_gap);
        style.spacing.icon_width = metrics.icon_side.min(metrics.field_height);
        style.spacing.icon_width_inner = style.spacing.icon_width * 0.6;
        style.spacing.icon_spacing = metrics.component_gap;
        let prepared = Arc::new(style);
        self.entries[self.next] = Some(PreparedStyleEntry {
            metrics,
            source: Arc::downgrade(&source),
            prepared: prepared.clone(),
        });
        self.next = (self.next + 1) % self.entries.len();
        prepared
    }
}

/// Persisted geometry for attribute controls; apply only inside an editor scope.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct AttributeMetrics {
    pub field_height: f32,
    pub numeric_width: f32,
    pub icon_side: f32,
    pub row_gap: f32,
    pub component_gap: f32,
}

impl Default for AttributeMetrics {
    fn default() -> Self {
        Self {
            field_height: 14.0,
            numeric_width: 36.0,
            icon_side: 12.0,
            row_gap: 4.0,
            component_gap: 2.0,
        }
    }
}

impl AttributeMetrics {
    pub const FIELD_HEIGHT_RANGE: std::ops::RangeInclusive<f32> = 8.0..=48.0;
    pub const NUMERIC_WIDTH_RANGE: std::ops::RangeInclusive<f32> = 24.0..=160.0;
    pub const ICON_SIDE_RANGE: std::ops::RangeInclusive<f32> = 8.0..=48.0;
    pub const ROW_GAP_RANGE: std::ops::RangeInclusive<f32> = 0.0..=16.0;
    pub const COMPONENT_GAP_RANGE: std::ops::RangeInclusive<f32> = 0.0..=16.0;

    pub fn normalized(self) -> Self {
        let defaults = Self::default();
        let clamp = |value: f32, fallback: f32, range: std::ops::RangeInclusive<f32>| {
            if value.is_finite() {
                value.clamp(*range.start(), *range.end())
            } else {
                fallback
            }
        };
        Self {
            field_height: clamp(
                self.field_height,
                defaults.field_height,
                Self::FIELD_HEIGHT_RANGE,
            ),
            numeric_width: clamp(
                self.numeric_width,
                defaults.numeric_width,
                Self::NUMERIC_WIDTH_RANGE,
            ),
            icon_side: clamp(self.icon_side, defaults.icon_side, Self::ICON_SIDE_RANGE),
            row_gap: clamp(self.row_gap, defaults.row_gap, Self::ROW_GAP_RANGE),
            component_gap: clamp(
                self.component_gap,
                defaults.component_gap,
                Self::COMPONENT_GAP_RANGE,
            ),
        }
    }

    pub fn row_height(self) -> f32 {
        let metrics = self.normalized();
        metrics.field_height.max(metrics.icon_side) + metrics.row_gap
    }

    pub fn gutter(self) -> f32 {
        let metrics = self.normalized();
        2.0 * metrics.icon_side + metrics.component_gap
    }

    pub fn value_layout(self) -> ValueEditorLayout {
        let metrics = self.normalized();
        ValueEditorLayout {
            numeric_width: metrics.numeric_width,
            row_height: Some(metrics.field_height),
            gap: metrics.component_gap,
        }
    }

    pub fn grid_config(self) -> egui_attr_grid::AttrGridConfig {
        egui_attr_grid::AttrGridConfig {
            prefix_width: self.gutter(),
            action_width: self.gutter(),
            min_editor_width: 3.0 * self.normalized().numeric_width
                + 2.0 * self.normalized().component_gap,
            row_height: Some(self.row_height()),
            value_layout: self.value_layout(),
            fields_sorted: true,
        }
    }

    /// Apply to a scoped child Ui. The Context/global style remains unchanged.
    /// Plain TextEdit builders should also use this Ui's button_padding as their margin.
    pub fn apply(self, ui: &mut egui::Ui) {
        let metrics = self.normalized();
        let base = ui.style().clone();
        let prepared = ui.data_mut(|data| {
            data.get_temp_mut_or_default::<PreparedStyleCache>(egui::Id::new(
                "frac_attribute_styles",
            ))
            .get(metrics, &base)
        });
        ui.set_style(prepared);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepared_style_reuses_idle_arc_and_invalidates_metrics_or_source_font() {
        let ctx = egui::Context::default();
        let source = Arc::new(egui::Style {
            override_font_id: Some(egui::FontId::proportional(20.0)),
            ..Default::default()
        });
        let original = (*source).clone();
        let metrics = AttributeMetrics::default();
        let mut previous: Option<Arc<egui::Style>> = None;
        for _ in 0..3 {
            let mut output = ctx.run_ui(Default::default(), |ui| {
                ui.set_style(source.clone());
                ui.scope(|ui| {
                    metrics.apply(ui);
                    let prepared = ui.style().clone();
                    if let Some(previous) = &previous {
                        assert!(Arc::ptr_eq(previous, &prepared));
                    }
                    previous = Some(prepared.clone());
                    metrics.apply(ui);
                    assert!(Arc::ptr_eq(&prepared, ui.style()));
                    let changed = AttributeMetrics {
                        field_height: 24.0,
                        ..metrics
                    };
                    changed.apply(ui);
                    assert!(!Arc::ptr_eq(&prepared, ui.style()));
                    assert_eq!(ui.style().spacing.interact_size.y, 24.0);
                    assert_eq!(ui.style().override_font_id.as_ref().unwrap().size, 20.0);
                    metrics.apply(ui);
                    assert!(Arc::ptr_eq(&prepared, ui.style()));
                });
                assert!(Arc::ptr_eq(ui.style(), &source));
            });
            output.textures_delta.clear();
        }
        assert_eq!(*source, original);
        let mut changed_source = (*source).clone();
        changed_source.override_font_id = Some(egui::FontId::proportional(9.0));
        let changed_source = Arc::new(changed_source);
        let mut output = ctx.run_ui(Default::default(), |ui| {
            ui.set_style(changed_source.clone());
            metrics.apply(ui);
            assert!(!Arc::ptr_eq(previous.as_ref().unwrap(), ui.style()));
            assert_eq!(ui.style().override_font_id.as_ref().unwrap().size, 9.0);
        });
        output.textures_delta.clear();
    }

    #[test]
    fn absent_preference_fields_keep_compact_defaults() {
        let metrics: AttributeMetrics = serde_json::from_str("{}").unwrap();
        assert_eq!(metrics, AttributeMetrics::default());
        let partial: AttributeMetrics = serde_json::from_str(r#"{"numeric_width":52.0}"#).unwrap();
        assert_eq!(partial.numeric_width, 52.0);
        assert_eq!(partial.field_height, 14.0);
        assert_eq!(metrics.row_height(), 18.0);
        assert_eq!(metrics.gutter(), 26.0);
        assert_eq!(metrics.value_layout().row_height, Some(14.0));
    }

    #[test]
    fn invalid_and_extreme_metrics_produce_finite_bounded_geometry() {
        let metrics = AttributeMetrics {
            field_height: f32::NAN,
            numeric_width: f32::INFINITY,
            icon_side: -10.0,
            row_gap: 1000.0,
            component_gap: -1.0,
        }
        .normalized();
        assert_eq!(metrics.field_height, 14.0);
        assert_eq!(metrics.numeric_width, 36.0);
        assert_eq!(metrics.icon_side, 8.0);
        assert_eq!(metrics.row_gap, 16.0);
        assert_eq!(metrics.component_gap, 0.0);
        assert_eq!(metrics.normalized(), metrics);
        assert_eq!(metrics.row_height(), 30.0);
        assert_eq!(metrics.gutter(), 16.0);
        assert_eq!(
            serde_json::from_str::<AttributeMetrics>(&serde_json::to_string(&metrics).unwrap())
                .unwrap(),
            metrics
        );
    }
}
