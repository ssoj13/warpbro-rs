//! One dock workspace for every tool panel, plus viewport-local chrome.
use super::{App, Tab};
use egui_dock::{DockArea, DockState, NodeIndex, TabViewer};
use egui_layout_manager::{LayoutAction, LayoutManager, LayoutStore};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub(super) enum Panel {
    Viewport,
    Gallery,
    Bookmarks,
    Materials,
    Inspector,
    Settings,
    Export,
    Timeline,
    Outliner,
    MaterialLibrary,
}
impl Panel {
    const ALL: [Self; 10] = [
        Self::Viewport,
        Self::Gallery,
        Self::Bookmarks,
        Self::Materials,
        Self::Inspector,
        Self::Settings,
        Self::Export,
        Self::Timeline,
        Self::Outliner,
        Self::MaterialLibrary,
    ];
    fn title(self) -> &'static str {
        match self {
            Self::Viewport => "Viewport",
            Self::Gallery => "Gallery",
            Self::Bookmarks => "Bookmarks",
            Self::Materials => "Materials",
            Self::MaterialLibrary => "Material Library",
            Self::Inspector => "Attribute Editor",
            Self::Settings => "Settings",
            Self::Export => "Render / Encode",
            Self::Timeline => "Timeline",
            Self::Outliner => "Outliner",
        }
    }
}

pub(super) fn default_layout() -> DockState<Panel> {
    let mut state = DockState::new(vec![Panel::Viewport]);
    let [workspace, _] =
        state
            .main_surface_mut()
            .split_below(NodeIndex::root(), 0.64, vec![Panel::Timeline]);
    let [workspace, _] = state.main_surface_mut().split_left(
        workspace,
        0.18,
        vec![
            Panel::Outliner,
            Panel::Gallery,
            Panel::Bookmarks,
            Panel::Materials,
        ],
    );
    state
        .main_surface_mut()
        .split_right(workspace, 0.64, vec![Panel::Inspector]);
    state
}

pub(super) fn valid_layout(state: &DockState<Panel>) -> bool {
    let mut seen = std::collections::HashSet::new();
    state.iter_all_tabs().all(|(_, tab)| seen.insert(*tab)) && seen.contains(&Panel::Viewport)
}

/// One-time upgrade preserves the user's existing panes and floating tools.
pub(super) fn add_timeline(state: &mut DockState<Panel>) {
    if state.find_tab(&Panel::Timeline).is_none() {
        if let Some(path) = state.find_tab(&Panel::Viewport) {
            state[path.surface].split_below(path.node, 0.74, vec![Panel::Timeline]);
        }
    }
}

pub(super) fn add_outliner(state: &mut DockState<Panel>) {
    if state.find_tab(&Panel::Outliner).is_none() {
        if let Some(path) = state.find_tab(&Panel::Viewport) {
            state[path.surface].split_left(path.node, 0.22, vec![Panel::Outliner]);
        }
    }
}

// egui_dock 0.21 does not update WindowState::screen_rect. Preserve each
// floating window's drawn outer size and position as next-frame requests.
pub(super) fn layout_blob(
    state: &DockState<Panel>,
    ctx: &egui::Context,
) -> Result<String, egui_dock_layout::Error> {
    let mut portable = state.clone();
    for (surface, _) in state.iter_surfaces_indexed() {
        if let Some(window) = portable.get_window_state_mut(surface) {
            // The id is the one used by egui_dock's window_surface.
            if let Some(area) =
                egui::AreaState::load(ctx, egui::Id::new(format!("window {surface:?}")))
                && area.rect().is_finite()
                && area.rect().width() > 0.0
                && area.rect().height() > 0.0
            {
                window.set_size(area.rect().size());
                window.set_position(area.left_top_pos());
            }
        }
    }
    egui_dock_layout::to_blob(&portable)
}

fn open_panel(state: &mut DockState<Panel>, panel: Panel) {
    if let Some(path) = state.find_tab(&panel) {
        let _ = state.set_active_tab(path);
        state.set_focused_node_and_surface(path.node_path());
    } else if matches!(
        panel,
        Panel::Settings | Panel::Export | Panel::MaterialLibrary
    ) {
        let surface = state.add_window(vec![panel]);
        if let Some(window) = state.get_window_state_mut(surface) {
            window.set_size(egui::vec2(850.0, 600.0));
        }
    } else {
        state.push_to_focused_leaf(panel);
    }
}

struct Viewer<'a> {
    app: &'a mut App,
}
impl TabViewer for Viewer<'_> {
    type Tab = Panel;
    fn id(&mut self, tab: &mut Panel) -> egui::Id {
        egui::Id::new(("frac-panel", *tab))
    }
    fn title(&mut self, tab: &mut Panel) -> egui::WidgetText {
        tab.title().into()
    }
    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Panel) {
        use crate::hotkeys::Scope;
        let scope = match tab {
            Panel::Viewport => Scope::Viewport,
            Panel::Timeline => Scope::Timeline,
            Panel::Gallery => Scope::Gallery,
            Panel::Bookmarks => Scope::Bookmarks,
            Panel::Materials => Scope::Materials,
            Panel::Inspector => Scope::AttributeEditor,
            Panel::Settings => Scope::Settings,
            Panel::Export => Scope::Export,
            Panel::Outliner => Scope::Outliner,
            Panel::MaterialLibrary => Scope::MaterialLibrary,
        };
        crate::hotkeys::register(ui, scope, ui.max_rect());
        ui.push_id(*tab, |ui| match tab {
            Panel::Viewport => self.app.viewport(ui),
            Panel::Inspector => self.app.inspector(ui),
            Panel::MaterialLibrary => self.app.material_library(ui),
            Panel::Settings => self.app.settings_ui(ui),
            Panel::Export => self.app.export_ui(ui),
            Panel::Timeline => self.app.world_ui.timeline(ui, &mut self.app.world),
            Panel::Outliner => self.app.world_ui.outliner(ui, &mut self.app.world),
            Panel::Gallery | Panel::Bookmarks | Panel::Materials => {
                self.app.tab = match tab {
                    Panel::Bookmarks => Tab::Bookmarks,
                    Panel::Materials => Tab::Materials,
                    _ => Tab::Gallery,
                };
                self.app.browser(ui);
            }
        });
    }
    fn is_closeable(&self, tab: &Panel) -> bool {
        *tab != Panel::Viewport
    }
    fn scroll_bars(&self, _: &Panel) -> [bool; 2] {
        [false, false]
    }
    fn context_menu(&mut self, ui: &mut egui::Ui, _: &mut Panel, _: egui_dock::NodePath) {
        for panel in Panel::ALL {
            if ui.button(panel.title()).clicked() {
                self.app.panels_to_open.push(panel);
                ui.close();
            }
        }
    }
}

/// A reusable, exact stamp of authored layout state. Pixel rectangles of docked
/// leaves are derived during painting; floating outer rectangles are authored.
#[derive(Default)]
pub(super) struct LayoutCache {
    stamp: Vec<u64>,
    scratch: Vec<u64>,
    window_ids: Vec<egui::Id>,
    pub blob: Option<String>,
    pub revision: u64,
    #[cfg(test)]
    pub serializations: usize,
}
impl LayoutCache {
    pub fn refresh(
        &mut self,
        state: &DockState<Panel>,
        ctx: &egui::Context,
    ) -> Result<(), egui_dock_layout::Error> {
        let stamp = &mut self.scratch;
        stamp.clear();
        let focused = state.focused_leaf();
        stamp.extend([
            focused.map_or(u64::MAX, |p| p.surface.0 as u64),
            focused.map_or(u64::MAX, |p| p.node.0 as u64),
        ]);
        for (index, surface) in state.iter_surfaces_indexed() {
            stamp.extend([
                index.0 as u64,
                match surface {
                    egui_dock::Surface::Empty => 0,
                    egui_dock::Surface::Main(_) => 1,
                    egui_dock::Surface::Window(_, _) => 2,
                },
            ]);
            let Some(tree) = surface.node_tree() else {
                continue;
            };
            stamp.push(tree.focused_leaf().map_or(u64::MAX, |p| p.0 as u64));
            for node in surface.iter_nodes() {
                match node {
                    egui_dock::Node::Empty => stamp.push(0),
                    egui_dock::Node::Leaf(leaf) => {
                        stamp.extend([
                            1,
                            leaf.tabs.len() as u64,
                            leaf.active.0 as u64,
                            leaf.scroll.to_bits() as u64,
                            leaf.collapsed as u64,
                            leaf.tab_bar_hidden as u64,
                        ]);
                        stamp.extend(leaf.tabs.iter().map(|panel| *panel as u64));
                    }
                    egui_dock::Node::Horizontal(split) | egui_dock::Node::Vertical(split) => {
                        stamp.extend([
                            if matches!(node, egui_dock::Node::Horizontal(_)) {
                                2
                            } else {
                                3
                            },
                            split.fraction.to_bits() as u64,
                            split.fully_collapsed as u64,
                            split.collapsed_leaf_count as u64,
                        ]);
                    }
                }
            }
            stamp.push(u64::MAX - 1);
            if let egui_dock::Surface::Window(_, window) = surface {
                while self.window_ids.len() <= index.0 {
                    let surface = egui_dock::SurfaceIndex(self.window_ids.len());
                    self.window_ids
                        .push(egui::Id::new(format!("window {surface:?}")));
                }
                let rect = egui::AreaState::load(ctx, self.window_ids[index.0])
                    .map(|area| area.rect())
                    .filter(|rect| rect.is_finite() && rect.is_positive())
                    .unwrap_or_else(|| window.rect());
                stamp.extend([
                    rect.min.x.to_bits() as u64,
                    rect.min.y.to_bits() as u64,
                    rect.max.x.to_bits() as u64,
                    rect.max.y.to_bits() as u64,
                ]);
            }
        }
        if self.blob.is_some() && self.stamp == self.scratch {
            return Ok(());
        }
        let blob = layout_blob(state, ctx)?;
        #[cfg(test)]
        {
            self.serializations += 1;
        }
        if self.blob.as_ref() != Some(&blob) {
            self.revision = self.revision.wrapping_add(1);
            self.blob = Some(blob);
        }
        std::mem::swap(&mut self.stamp, &mut self.scratch);
        Ok(())
    }
    pub fn invalidate(&mut self) {
        self.stamp.clear();
    }
}

/// Persistent presets are separate from transient widget state.
pub(super) struct Layouts {
    pub store: LayoutStore,
    manager: LayoutManager,
    pub cache: LayoutCache,
}
impl Default for Layouts {
    fn default() -> Self {
        Self {
            store: LayoutStore::new(),
            manager: LayoutManager::new().with_combo_width(140.0),
            cache: LayoutCache::default(),
        }
    }
}

fn restore_layout(blob: &str) -> Result<DockState<Panel>, String> {
    let state = egui_dock_layout::from_blob(blob).map_err(|e| e.to_string())?;
    if valid_layout(&state) {
        Ok(state)
    } else {
        Err("Layout must contain one viewport and no duplicate panels".into())
    }
}

impl App {
    /// The host places this inline in the top-right of its main toolbar.
    pub(super) fn layout_manager_ui(&mut self, ui: &mut egui::Ui) {
        let mut cache = std::mem::take(&mut self.layouts.cache);
        if let Err(error) = cache.refresh(&self.dock, ui.ctx()) {
            self.layouts.cache = cache;
            ui.label("Layouts unavailable")
                .on_hover_text(error.to_string());
            return;
        }
        let blob = cache.blob.as_deref().unwrap_or_default();
        let mut restored = false;
        let previous = self.layouts.store.current().to_owned();
        ui.push_id("frac-layout-manager", |ui| {
            if let Some(LayoutAction::Restore(saved)) =
                self.layouts.manager.show(ui, &mut self.layouts.store, blob)
            {
                match restore_layout(&saved) {
                    Ok(state) => {
                        self.dock = state;
                        self.panels_to_open.clear();
                        restored = true;
                    }
                    Err(error) => {
                        if !previous.is_empty() {
                            self.layouts.store.select(&previous);
                        }
                        self.status = format!("Layout restore failed: {error}");
                    }
                }
            }
            ui.menu_button("Layout", |ui| {
                let selected = self.layouts.store.current().to_owned();
                if ui
                    .add_enabled(!selected.is_empty(), egui::Button::new("Update selected"))
                    .clicked()
                {
                    self.layouts.store.save(selected, blob.to_owned());
                    ui.close();
                }
                if ui.button("Reset to default").clicked() {
                    self.dock = default_layout();
                    self.panels_to_open.clear();
                    restored = true;
                    ui.close();
                }
            });
        });
        if restored {
            cache.invalidate();
        }
        self.layouts.cache = cache;
    }

    pub(super) fn window_menu(&mut self, ui: &mut egui::Ui) {
        ui.menu_button("Window", |ui| {
            for panel in Panel::ALL {
                let open = self.dock.find_tab(&panel).is_some();
                if ui.selectable_label(open, panel.title()).clicked() {
                    self.panels_to_open.push(panel);
                    ui.close();
                }
            }
            ui.separator();
            if ui.button("Reset layout").clicked() {
                self.dock = default_layout();
                self.panels_to_open.clear();
                ui.close();
            }
            if ui.button("Fonts…").clicked() {
                self.prefs.selected = 3;
                self.panels_to_open.push(Panel::Settings);
                ui.close();
            }
        });
    }

    pub(super) fn dock_ui(&mut self, ui: &mut egui::Ui) {
        for panel in self.panels_to_open.drain(..) {
            open_panel(&mut self.dock, panel);
        }
        let mut state = std::mem::replace(&mut self.dock, DockState::new(Vec::new()));
        DockArea::new(&mut state)
            .id(egui::Id::new("frac-workspace"))
            .style(egui_dock::Style::from_egui(ui.style().as_ref()))
            .show_inside(ui, &mut Viewer { app: self });
        for panel in self.panels_to_open.drain(..) {
            open_panel(&mut state, panel);
        }
        self.dock = state;
    }

    pub(super) fn viewport_toolbar(
        &mut self,
        ui: &mut egui::Ui,
        rect: egui::Rect,
    ) -> egui_viewport_toolbar::ToolbarResponse {
        let mut state = self.toolbar;
        let response = egui_viewport_toolbar::ViewportToolbar {
            edge: egui_viewport_toolbar::ToolbarEdge::Top,
            margin: 0.0,
            height: (self.fonts.body + 12.0).max(34.0),
            ..Default::default()
        }
        .show(ui, rect, &mut state, |ui| {
            ui.add(
                egui::DragValue::new(&mut self.scene.render.exposure_stops)
                    .speed(0.05)
                    .range(-10.0..=10.0)
                    .fixed_decimals(2)
                    .suffix(" EV"),
            )
            .on_hover_text("Exposure");
            ui.separator();
            self.colour.set_hdr(true);
            let changed = self.colour.quick_view_ui(ui);
            self.apply_colour_change(changed);
            ui.separator();
            ui.label("Proxy:");
            egui::ComboBox::from_id_salt("viewport_proxy")
                .width(60.0)
                .selected_text(if self.resolution >= 0.99 {
                    "1:1"
                } else if self.resolution >= 0.49 {
                    "1/2"
                } else {
                    "1/4"
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.resolution, 1.0, "1:1");
                    ui.selectable_value(&mut self.resolution, 0.5, "1/2");
                    ui.selectable_value(&mut self.resolution, 0.25, "1/4");
                });
            ui.add(
                egui::DragValue::new(&mut self.target_spp)
                    .range(1..=65536)
                    .suffix(" spp"),
            )
            .on_hover_text("Samples to converge");
            ui.separator();
            ui.toggle_value(&mut self.paused, egui_phosphor::regular::PAUSE)
                .on_hover_text("Pause rendering");
            if ui
                .selectable_label(
                    self.scene.camera.free_flight,
                    egui_phosphor::regular::AIRPLANE,
                )
                .on_hover_text("Free flight / horizon lock · RMB + WASD, R/C, Q/E")
                .clicked()
            {
                self.toggle_flight_mode();
            }
        });
        self.toolbar = state;
        response
    }
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub(super) struct Fonts {
    pub face: String,
    pub body: f32,
    pub small: f32,
    pub heading: f32,
    pub monospace: f32,
    pub zoom: f32,
}
impl Default for Fonts {
    fn default() -> Self {
        Self {
            face: "Default".into(),
            body: 13.0,
            small: 10.0,
            heading: 18.0,
            monospace: 13.0,
            zoom: 1.0,
        }
    }
}
impl App {
    pub(super) fn fonts_ui(&mut self, ui: &mut egui::Ui) {
        egui_prefs2::section_header(ui, "Fonts & UI scale");
        let definitions = egui::FontDefinitions::default();
        egui::ComboBox::from_id_salt("ui-font")
            .selected_text(&self.fonts.face)
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut self.fonts.face, "Default".into(), "Default");
                for name in definitions.font_data.keys() {
                    ui.selectable_value(&mut self.fonts.face, name.clone(), name);
                }
            });
        egui::Grid::new("font-sizes").num_columns(2).show(ui, |ui| {
            for (label, value) in [
                ("Body / controls", &mut self.fonts.body),
                ("Small text", &mut self.fonts.small),
                ("Headings", &mut self.fonts.heading),
                ("Monospace", &mut self.fonts.monospace),
            ] {
                ui.label(label);
                ui.add(egui::DragValue::new(value).range(8.0..=36.0).suffix(" pt"));
                ui.end_row();
            }
            ui.label("UI scale");
            ui.add(egui::Slider::new(&mut self.fonts.zoom, 0.75..=2.0));
            ui.end_row();
        });
        ui.separator();
        ui.heading("Heading preview");
        ui.label("Body text · Формула и камера");
        ui.small("Small text preview");
        ui.monospace("Monospace · 1024 spp");
    }

    pub(super) fn apply_fonts(&mut self, ctx: &egui::Context) {
        if self.applied_fonts.as_ref() == Some(&self.fonts) {
            return;
        }
        self.fonts.body = self.fonts.body.clamp(8.0, 36.0);
        self.fonts.small = self.fonts.small.clamp(8.0, 36.0);
        self.fonts.heading = self.fonts.heading.clamp(8.0, 36.0);
        self.fonts.monospace = self.fonts.monospace.clamp(8.0, 36.0);
        self.fonts.zoom = self.fonts.zoom.clamp(0.75, 2.0);
        self.prefs.fonts.body = self.fonts.body;
        self.prefs.fonts.sidebar = self.fonts.body;
        self.prefs.fonts.heading = self.fonts.heading;
        let mut definitions = egui::FontDefinitions::default();
        if definitions.font_data.contains_key(&self.fonts.face) {
            definitions
                .families
                .entry(egui::FontFamily::Proportional)
                .or_default()
                .insert(0, self.fonts.face.clone());
        }
        egui_phosphor::add_to_fonts(&mut definitions, egui_phosphor::Variant::Regular);
        ctx.set_fonts(definitions);
        ctx.set_zoom_factor(self.fonts.zoom);
        egui_attr_table::set_label_width(ctx, 130.0 * (self.fonts.body / 13.0).max(1.0));
        ctx.all_styles_mut(|style| {
            for (kind, size, family) in [
                (
                    egui::TextStyle::Body,
                    self.fonts.body,
                    egui::FontFamily::Proportional,
                ),
                (
                    egui::TextStyle::Button,
                    self.fonts.body,
                    egui::FontFamily::Proportional,
                ),
                (
                    egui::TextStyle::Small,
                    self.fonts.small,
                    egui::FontFamily::Proportional,
                ),
                (
                    egui::TextStyle::Heading,
                    self.fonts.heading,
                    egui::FontFamily::Proportional,
                ),
                (
                    egui::TextStyle::Monospace,
                    self.fonts.monospace,
                    egui::FontFamily::Monospace,
                ),
            ] {
                style
                    .text_styles
                    .insert(kind, egui::FontId::new(size, family));
            }
        });
        self.applied_fonts = Some(self.fonts.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn layout_cache_reuses_idle_blob_and_observes_authored_edits() {
        let ctx = egui::Context::default();
        let mut state = default_layout();
        let mut cache = LayoutCache::default();
        cache.refresh(&state, &ctx).unwrap();
        let pointer = cache.blob.as_ref().unwrap().as_ptr();
        let revision = cache.revision;
        for _ in 0..32 {
            cache.refresh(&state, &ctx).unwrap();
        }
        assert_eq!(cache.serializations, 1);
        assert_eq!(cache.revision, revision);
        assert_eq!(cache.blob.as_ref().unwrap().as_ptr(), pointer);
        let path = state.find_tab(&Panel::Timeline).unwrap();
        state[path.surface][path.node]
            .get_leaf_mut()
            .unwrap()
            .tab_bar_hidden = true;
        cache.refresh(&state, &ctx).unwrap();
        assert_eq!(cache.serializations, 2);
        assert_ne!(cache.revision, revision);
        open_panel(&mut state, Panel::Export);
        cache.refresh(&state, &ctx).unwrap();
        let restored = restore_layout(cache.blob.as_deref().unwrap()).unwrap();
        assert!(restored.find_tab(&Panel::Export).is_some());
        let count = cache.serializations;
        cache.refresh(&state, &ctx).unwrap();
        assert_eq!(cache.serializations, count);
    }

    #[test]
    fn fresh_and_floating_layouts_restore_and_tabs_do_not_duplicate() {
        let mut state = default_layout();
        open_panel(&mut state, Panel::Settings);
        open_panel(&mut state, Panel::Settings);
        assert_eq!(
            state
                .iter_all_tabs()
                .filter(|(_, p)| **p == Panel::Settings)
                .count(),
            1
        );
        let blob = layout_blob(&state, &egui::Context::default()).unwrap();
        let restored = egui_dock_layout::from_blob::<Panel>(&blob).unwrap();
        assert!(valid_layout(&restored));
        assert_eq!(restored.iter_all_tabs().count(), 8);
    }
    #[test]
    fn floating_geometry_survives_a_new_egui_context() {
        struct Placeholder;
        impl TabViewer for Placeholder {
            type Tab = Panel;
            fn id(&mut self, tab: &mut Panel) -> egui::Id {
                egui::Id::new(tab)
            }
            fn title(&mut self, tab: &mut Panel) -> egui::WidgetText {
                tab.title().into()
            }
            fn ui(&mut self, ui: &mut egui::Ui, _: &mut Panel) {
                ui.allocate_space(ui.available_size());
            }
        }
        fn draw(ctx: &egui::Context, state: &mut DockState<Panel>) {
            for _ in 0..3 {
                let _ = ctx.run_ui(
                    egui::RawInput {
                        screen_rect: Some(egui::Rect::from_min_size(
                            egui::Pos2::ZERO,
                            egui::vec2(1600.0, 900.0),
                        )),
                        ..Default::default()
                    },
                    |root| {
                        egui::CentralPanel::default().show(root, |ui| {
                            DockArea::new(state).show_inside(ui, &mut Placeholder);
                        });
                    },
                );
            }
        }
        let mut state = default_layout();
        open_panel(&mut state, Panel::Settings);
        let surface = state.find_tab(&Panel::Settings).unwrap().surface;
        state
            .get_window_state_mut(surface)
            .unwrap()
            .set_position(egui::pos2(100.0, 120.0));
        let ctx = egui::Context::default();
        draw(&ctx, &mut state);
        let before = state[surface].root_node().unwrap().rect().unwrap();
        let blob = layout_blob(&state, &ctx).unwrap();
        let mut restored = egui_dock_layout::from_blob::<Panel>(&blob).unwrap();
        draw(&egui::Context::default(), &mut restored);
        let after = restored[surface].root_node().unwrap().rect().unwrap();
        assert!(
            (after.min - before.min).length() <= 1.0,
            "{before:?} -> {after:?}"
        );
        assert!(
            (after.size() - before.size()).length() <= 1.0,
            "{before:?} -> {after:?}"
        );
    }

    #[test]
    fn named_layout_preserves_floating_export_and_legacy_workspaces() {
        let ctx = egui::Context::default();
        let legacy = layout_blob(&default_layout(), &ctx).unwrap();
        let mut state = default_layout();
        open_panel(&mut state, Panel::Settings);
        open_panel(&mut state, Panel::Export);
        let surface = state.find_tab(&Panel::Export).unwrap().surface;
        let window = state.get_window_state_mut(surface).unwrap();
        window.set_position(egui::pos2(125.0, 80.0));
        window.set_size(egui::vec2(700.0, 500.0));
        let mut store = LayoutStore::new();
        store.save("Default", legacy);
        store.save("Render", layout_blob(&state, &ctx).unwrap());
        assert!(store.rename("Render", "Export workspace"));
        let json = serde_json::to_string(&store).unwrap();
        let mut restored_store: LayoutStore = serde_json::from_str(&json).unwrap();
        let mut restored =
            restore_layout(&restored_store.select("Export workspace").unwrap()).unwrap();
        assert_eq!(restored.iter_all_tabs().count(), 9);
        assert!(restored.find_tab(&Panel::Export).is_some());
        let window = restored.get_window_state_mut(surface).unwrap();
        let geometry = serde_json::to_value(window).unwrap();
        assert_eq!(
            geometry["next_size"],
            serde_json::to_value(egui::vec2(700.0, 500.0)).unwrap()
        );
        assert_eq!(
            geometry["next_position"],
            serde_json::to_value(egui::pos2(125.0, 80.0)).unwrap()
        );
        let legacy_restored = restore_layout(&restored_store.select("Default").unwrap()).unwrap();
        assert_eq!(legacy_restored.iter_all_tabs().count(), 7);
        assert!(restored_store.delete("Export workspace"));
        assert_eq!(restored_store.len(), 1);
    }

    #[test]
    fn timeline_upgrade_preserves_existing_panels_and_is_idempotent() {
        let mut state = DockState::new(vec![Panel::Viewport, Panel::Inspector]);
        add_timeline(&mut state);
        add_timeline(&mut state);
        assert!(valid_layout(&state));
        assert_eq!(state.iter_all_tabs().count(), 3);
        for panel in [Panel::Viewport, Panel::Inspector, Panel::Timeline] {
            assert!(state.find_tab(&panel).is_some());
        }
    }
    #[test]
    fn world_upgrade_preserves_existing_panels_and_is_idempotent() {
        let mut state = DockState::new(vec![Panel::Viewport, Panel::Inspector, Panel::Timeline]);
        add_outliner(&mut state);
        add_outliner(&mut state);
        assert!(valid_layout(&state));
        assert_eq!(state.iter_all_tabs().count(), 4);
        for panel in [
            Panel::Viewport,
            Panel::Inspector,
            Panel::Timeline,
            Panel::Outliner,
        ] {
            assert!(state.find_tab(&panel).is_some());
        }
    }

    #[test]
    fn incomplete_or_duplicate_layouts_are_rejected() {
        assert!(!valid_layout(&DockState::new(vec![Panel::Inspector])));
        assert!(!valid_layout(&DockState::new(vec![
            Panel::Viewport,
            Panel::Viewport
        ])));
    }
}
