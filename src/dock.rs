//! One dock workspace for every tool panel, plus viewport-local chrome.
//!
//! The workspace is an `soluwell_chrome::dock_layout::Layout`: plain data the dock draws and reports
//! actions for; every change goes through `Layout::apply_with_permissions` (atomic, validated).
use super::App;
use soluwell_chrome::dock_layout::{
    Action, DockArea, DockContent, DockError, DockLayoutBar, DockStyle, Layout, LayoutStore,
    Location, Node, Permissions, Placement, SplitAxis, SplitSize,
};

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
    Takes,
}
impl Panel {
    const ALL: [Self; 11] = [
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
        Self::Takes,
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
            Self::Takes => "Takes",
        }
    }
    /// The Viewport is the workspace's one fixed panel: it never closes.
    pub(super) fn permissions(&self) -> Permissions {
        Permissions {
            closable: *self != Self::Viewport,
            ..Permissions::default()
        }
    }
}

/// A tab group of `panels`, the first one shown.
fn tabs(panels: &[Panel]) -> Node<Panel> {
    Node::Tabs {
        panels: panels.to_vec(),
        active: 0,
    }
}

/// `first` and `second` divided along `axis`.
fn split(axis: SplitAxis, size: SplitSize, first: Node<Panel>, second: Node<Panel>) -> Node<Panel> {
    Node::Split {
        axis,
        size,
        first: Box::new(first),
        second: Box::new(second),
    }
}

/// The start layout: the Attribute Editor over Settings / Render on the right, the Timeline
/// under the workspace, and Outliner / Gallery / Materials over Takes left of the Viewport.
/// Secondary panels use fixed pixel sizes so resizing the window resizes the Viewport.
pub(super) fn default_layout() -> Layout<Panel> {
    use SplitAxis::{Horizontal, Vertical};
    let panels = split(
        Vertical,
        SplitSize::FixedSecond(260.0),
        tabs(&[Panel::Outliner, Panel::Gallery, Panel::Materials]),
        tabs(&[Panel::Takes]),
    );
    let workspace = split(
        Vertical,
        SplitSize::FixedSecond(180.0),
        split(
            Horizontal,
            SplitSize::FixedFirst(280.0),
            panels,
            tabs(&[Panel::Viewport]),
        ),
        tabs(&[Panel::Timeline]),
    );
    let right = split(
        Vertical,
        SplitSize::FixedSecond(320.0),
        tabs(&[Panel::Inspector]),
        tabs(&[Panel::Settings, Panel::Export]),
    );
    Layout {
        root: Some(split(
            Horizontal,
            SplitSize::FixedSecond(380.0),
            workspace,
            right,
        )),
        floating: Vec::new(),
    }
}

/// A usable workspace: valid (every panel once, finite geometry) and showing the Viewport.
pub(super) fn valid_layout(layout: &Layout<Panel>) -> bool {
    layout.validate().is_ok() && layout.contains(&Panel::Viewport)
}

/// Size of a tool window opened floating (Settings, Render / Encode, Material Library).
const WINDOW_SIZE: egui::Vec2 = egui::vec2(850.0, 600.0);

/// Bring `panel` forward: show its tab where it is, else open it. Settings, Render / Encode and
/// the Material Library open as a floating window centred on `area` (the dock's rect); the
/// others join the `focused` panel's group (else the Viewport's, else the first group).
fn open_panel(
    layout: &mut Layout<Panel>,
    panel: Panel,
    focused: Option<Panel>,
    area: egui::Rect,
) -> Result<(), DockError> {
    if layout.contains(&panel) {
        return layout.apply(Action::Activate { panel });
    }
    if matches!(
        panel,
        Panel::Settings | Panel::Export | Panel::MaterialLibrary
    ) {
        let min = (area.center() - WINDOW_SIZE * 0.5).max(area.min);
        let location = Location {
            floating: Some([min.x, min.y, WINDOW_SIZE.x, WINDOW_SIZE.y]),
            anchor: None,
            placement: Placement::Tab { before: None },
            size: None,
            stack: None,
            sibling: Vec::new(),
            window: None,
        };
        return layout.restore(panel, &location);
    }
    let anchor = focused
        .filter(|p| layout.contains(p))
        .or_else(|| layout.contains(&Panel::Viewport).then_some(Panel::Viewport))
        .or_else(|| layout.panels().first().map(|p| **p));
    layout.apply_with_permissions(Action::Open { panel, anchor }, Panel::permissions)
}

struct Viewer<'a> {
    app: &'a mut App,
}
impl DockContent<Panel> for Viewer<'_> {
    fn title(&self, panel: &Panel) -> String {
        panel.title().into()
    }
    fn body(&mut self, ui: &mut egui::Ui, panel: &Panel) {
        use crate::hotkeys::Scope;
        let scope = match panel {
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
            Panel::Takes => Scope::Takes,
        };
        crate::hotkeys::register(ui, scope, ui.max_rect());
        ui.push_id(*panel, |ui| match panel {
            Panel::Viewport => self.app.viewport(ui),
            Panel::Inspector => self.app.inspector(ui),
            Panel::MaterialLibrary => self.app.material_library(ui),
            Panel::Settings => self.app.settings_ui(ui),
            Panel::Export => self.app.export_ui(ui),
            Panel::Timeline => self.app.world_ui.timeline(ui, &mut self.app.world),
            Panel::Outliner => self.app.world_ui.outliner(ui, &mut self.app.world),
            Panel::Takes => self.app.takes_panel(ui),
            Panel::Materials => self.app.materials_tab(ui),
            Panel::Gallery | Panel::Bookmarks => self.app.browser(ui, *panel),
        });
    }
    fn permissions(&self, panel: &Panel) -> Permissions {
        panel.permissions()
    }
    fn panel_menu(&mut self, ui: &mut egui::Ui, _: &Panel) {
        for panel in Panel::ALL {
            if ui.button(panel.title()).clicked() {
                self.app.emit_app(super::AppAction::OpenPanel(panel));
                ui.close();
            }
        }
        ui.separator();
    }
}

/// The settings autosave's view of the workspace: its blob, serialised again only when the
/// layout changed (the autosave asks every frame). `revision` counts blob changes.
#[derive(Default)]
pub(super) struct LayoutCache {
    layout: Option<Layout<Panel>>,
    pub blob: Option<String>,
    pub revision: u64,
    #[cfg(test)]
    pub serializations: usize,
}
impl LayoutCache {
    pub fn refresh(
        &mut self,
        layout: &Layout<Panel>,
    ) -> Result<(), soluwell_chrome::dock_layout::Error> {
        if self.blob.is_some() && self.layout.as_ref() == Some(layout) {
            return Ok(());
        }
        let blob = soluwell_chrome::dock_layout::to_blob(layout)?;
        #[cfg(test)]
        {
            self.serializations += 1;
        }
        if self.blob.as_ref() != Some(&blob) {
            self.revision = self.revision.wrapping_add(1);
            self.blob = Some(blob);
        }
        self.layout = Some(layout.clone());
        Ok(())
    }
}

/// Persistent presets are separate from transient widget state.
pub(super) struct Layouts {
    pub store: LayoutStore,
    bar: DockLayoutBar,
    pub cache: LayoutCache,
}
impl Default for Layouts {
    fn default() -> Self {
        Self {
            store: LayoutStore::new(),
            bar: DockLayoutBar::new().with_combo_width(140.0),
            cache: LayoutCache::default(),
        }
    }
}

/// A stored layout blob, read and checked ([`valid_layout`]).
pub(super) fn restore_layout(blob: &str) -> Result<Layout<Panel>, String> {
    let layout = soluwell_chrome::dock_layout::from_blob(blob).map_err(|e| e.to_string())?;
    if valid_layout(&layout) {
        Ok(layout)
    } else {
        Err(INVALID_LAYOUT.into())
    }
}

/// Why a stored layout is refused ([`valid_layout`]).
const INVALID_LAYOUT: &str = "Layout must contain one viewport and no duplicate panels";

impl App {
    /// The host places this inline in the top-right of its main toolbar.
    pub(super) fn layout_manager_ui(&mut self, ui: &mut egui::Ui) {
        let previous = self.layouts.store.current().to_owned();
        let mut store = self.layouts.store.clone();
        let mut save_selected = None;
        ui.push_id("frac-layout-manager", |ui| {
            // The bar reads a picked layout before selecting it; WarpBro also needs the Viewport.
            match self.layouts.bar.show(ui, &mut store, &self.dock) {
                Ok(output) => match output.restored {
                    Some(layout) if valid_layout(&layout) => {
                        self.emit_app(super::AppAction::RestoreLayout(layout));
                    }
                    Some(_) => {
                        if !previous.is_empty() {
                            store.select(&previous);
                        }
                        self.status = format!("Layout restore failed: {INVALID_LAYOUT}");
                    }
                    None => {}
                },
                Err(error) => self.status = format!("Layout: {error}"),
            }
            ui.menu_button("Layout", |ui| {
                let selected = store.current().to_owned();
                if ui
                    .add_enabled(
                        !selected.is_empty() && !store.is_builtin(&selected),
                        egui::Button::new("Update selected"),
                    )
                    .clicked()
                {
                    save_selected = Some(selected);
                    ui.close();
                }
                if ui.button("Reset to default").clicked() {
                    self.emit_app(super::AppAction::ResetLayout);
                    ui.close();
                }
            });
        });
        if store != self.layouts.store {
            self.emit_app(super::AppAction::LayoutStore(store));
        }
        if let Some(selected) = save_selected {
            self.emit_app(super::AppAction::SaveLayout(selected));
        }
    }

    /// The default workspace (`default_layout`): Layout > Reset to default and Window > Reset
    /// layout. Panels queued to open are dropped with the old layout.
    pub(super) fn reset_layout(&mut self) {
        self.dock = default_layout();
        self.panels_to_open.clear();
    }

    pub(super) fn window_menu(&mut self, ui: &mut egui::Ui) {
        ui.menu_button("Window", |ui| {
            for panel in Panel::ALL {
                let open = self.dock.contains(&panel);
                if ui.selectable_label(open, panel.title()).clicked() {
                    self.emit_app(super::AppAction::OpenPanel(panel));
                    ui.close();
                }
            }
            ui.separator();
            if ui.button("Reset layout").clicked() {
                self.emit_app(super::AppAction::ResetLayout);
                ui.close();
            }
            if ui.button("Fonts…").clicked() {
                self.emit_app(super::AppAction::OpenSettings(super::SettingsPage::Fonts));
                ui.close();
            }
        });
    }

    /// Open the requested panels (`panels_to_open`) in the workspace.
    fn open_panels(&mut self, area: egui::Rect) {
        for panel in std::mem::take(&mut self.panels_to_open) {
            if let Err(error) = open_panel(&mut self.dock, panel, self.dock_focus, area) {
                self.status = format!("Cannot open {}: {error}", panel.title());
            }
        }
    }

    pub(super) fn dock_ui(&mut self, ui: &mut egui::Ui) {
        let area = ui.available_rect_before_wrap();
        let style = DockStyle::from_ui(ui);
        self.open_panels(area);
        self.dock
            .pin_around(|p| *p == Panel::Viewport, area, style.gap);
        // The bodies borrow the whole App; the layout is only read while drawing.
        let layout = std::mem::take(&mut self.dock);
        let output = DockArea::new(egui::Id::new("frac-workspace")).show_with(
            ui,
            &layout,
            &style,
            &mut Viewer { app: self },
        );
        self.dock = layout;
        if let Some(error) = output.error {
            self.status = format!("Dock layout not shown: {error}");
        }
        if output.focused.is_some() {
            self.dock_focus = output.focused;
        }
        for action in output.actions {
            self.emit_app(super::AppAction::DockAction(action));
        }
        // Attribute Editor > Show in timeline brings the Timeline forward.
        if self.world_ui.take_timeline_request() {
            self.emit_app(super::AppAction::OpenPanel(Panel::Timeline));
        }
        self.open_panels(area);
    }

    pub(super) fn viewport_toolbar(
        &mut self,
        ui: &mut egui::Ui,
        rect: egui::Rect,
    ) -> soluwell_viewport::toolbar::ToolbarResponse {
        let mut state = self.toolbar;
        let response = soluwell_viewport::toolbar::ViewportToolbar {
            edge: soluwell_viewport::toolbar::ToolbarEdge::Top,
            margin: 0.0,
            height: (self.fonts.body + 12.0).max(34.0),
            ..Default::default()
        }
        .show(ui, rect, &mut state, |ui| {
            // CamClip claims the right end first; the other controls lay out left to right in
            // what remains, so a narrow viewport never pushes CamClip off the toolbar.
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.add_space(ui.spacing().item_spacing.x);
                match self.camera_slots.ui(
                    ui,
                    self.controls.swap_slot_buttons,
                ) {
                    Some(crate::camera_slots::SlotAction::Stored(i)) => {
                        self.emit_app(super::AppAction::StoreCameraSlot(i));
                    }
                    Some(crate::camera_slots::SlotAction::Restored(i)) => {
                        // The viewport commit authors the pasted camera; flight inertia must not
                        // carry on from the previous pose.
                        self.emit_app(super::AppAction::RecallCameraSlot(i));
                    }
                    Some(crate::camera_slots::SlotAction::Empty(i)) => {
                        self.status = format!("CamClip {} is empty: store a camera first", i + 1);
                    }
                    None => {}
                }
                ui.separator();
                ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                    use soluwell_config::icons as ph;
                    for tool in super::ViewTool::ALL {
                        let (icon, tip) = tool.label();
                        if ui
                            .selectable_label(self.tool == tool, icon)
                            .on_hover_text(tip)
                            .clicked()
                        {
                            self.emit_app(super::AppAction::SetTool(tool));
                        }
                    }
                    // Twenty toggles: the menu stays open until a click elsewhere or on its button.
                    egui::containers::menu::MenuButton::new(ph::VISIBLE)
                        .config(
                            egui::containers::menu::MenuConfig::new()
                                .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside),
                        )
                        .ui(ui, |ui| self.visibility_menu(ui))
                        .0
                        .on_hover_text(
                            "Visibility and selectability of node kinds in the viewport",
                        );
                    ui.separator();
                    let frame = f64::from(self.world_ui.playhead);
                    // The binding the viewport actually renders; step_viewport is its only producer.
                    let active = self.viewport_render.clone();
                    if let Some(effective) = active.as_ref().map(|selection| &selection.effective) {
                        let mut exposure = effective.render.exposure_stops;
                        soluwell_viewport::toolbar::exposure_control(
                            ui,
                            &mut exposure,
                            &mut self.exposure_hold,
                            -10.0..=10.0,
                            None,
                        );
                        if exposure != effective.render.exposure_stops
                            && self.world.document.viewport_settings_id().is_ok()
                        {
                            // The camera's gesture: the drag shares the navigation's undo step.
                            let gesture = ui
                                .input(|input| input.pointer.primary_down())
                                .then(|| self.camera_gesture())
                                .or_else(|| self.world.active_edit());
                            self.emit_app(super::AppAction::Exposure { value: exposure, frame, gesture });
                        }
                    }
                    ui.separator();
                    if let Some(selection) = self.colour.quick_view_ui(ui) { self.emit_app(super::AppAction::ColourSelection(selection)); }
                    if ui
                        .button(ph::GEAR)
                        .on_hover_text("Colour management settings")
                        .clicked()
                    {
                        self.emit_app(super::AppAction::OpenSettings(super::SettingsPage::Color));
                    }
                    ui.separator();
                    let actions = crate::render_profiles_ui::toolbar(
                        ui,
                        &self.world,
                        &self.event_emitter(),
                        f64::from(self.world_ui.playhead),
                        active.as_ref(),
                    );
                    self.profile_ui_actions(actions);
                    ui.separator();
                    // A/B of the viewport only: the scene's denoise settings (World Settings) and exports
                    // stay as authored, and denoising keeps running so switching back is instant.
                    let denoising = active
                        .as_ref()
                        .is_some_and(|selection| selection.effective.render.denoise.enabled);
                    let mut shown = denoising && !self.raw_view;
                    if ui
                        .add_enabled(denoising, egui::Button::selectable(shown, ph::PATH_TRACE))
                        .on_hover_text("Viewport: denoised / raw samples")
                        .on_disabled_hover_text("Denoising is off in the selected viewport profile")
                        .clicked()
                    {
                        shown = !shown;
                        self.emit_app(super::AppAction::SetRaw(!shown));
                    }
                    if ui
                        .selectable_label(self.scene.camera.free_flight, ph::FLIGHT)
                        .on_hover_text(
                            "Free flight / horizon lock (L) · RMB + WASD, R / Space up, C down, Q/E roll, Shift fast, Alt slow, Ctrl align",
                        )
                        .clicked()
                    {
                        self.emit_app(super::AppAction::ToggleFlight);
                    }
                    // One button keeps the toolbar narrow: click saves as displayed, the context menu
                    // (right click) offers the other snapshot files, as File does.
                    let snapshot = ui.button(ph::CAMERA).on_hover_text(
                        "Save the viewport as displayed (PNG) · right click: more formats",
                    );
                    if snapshot.clicked() {
                        self.emit_app(super::AppAction::SaveFrame { monitor: super::Monitor::read(ui.ctx()), file: None });
                    }
                    snapshot.context_menu(|ui| self.snapshot_menu(ui));
                });
            });
        });
        if self.toolbar != state {
            self.emit_app(super::AppAction::PatchSettings(
                serde_json::json!({ "toolbar": state }),
            ));
        }
        response
    }
}

impl super::App {
    /// The viewport's kind filter (Blender's "Selectability & Visibility" popover): an eye and a
    /// pick toggle per node kind and the extras switch. It holds nothing itself: each toggle
    /// edits the ViewportSettings node's attribute (`set_viewport_flag`), one undo step, as the
    /// Attribute Editor does.
    fn visibility_menu(&mut self, ui: &mut egui::Ui) {
        use soluwell_config::icons as ph;
        let frame = f64::from(self.world_ui.playhead);
        let policy = match self.world.document.viewport_policy(frame) {
            Ok(policy) => policy,
            Err(error) => {
                ui.label(error);
                return;
            }
        };
        let mut edit = None;
        egui::Grid::new("viewport_visibility").show(ui, |ui| {
            for (kind, name, label) in crate::render_profiles::VIEW_KINDS {
                ui.label(label);
                let shown = policy.shown.contains(kind);
                if ui
                    .selectable_label(shown, if shown { ph::VISIBLE } else { ph::HIDDEN })
                    .on_hover_text("Show in the viewport")
                    .clicked()
                {
                    edit = Some((format!("/viewport/show/{name}"), !shown));
                }
                let pick = policy.selectable.contains(kind);
                if ui
                    .add_enabled(shown, egui::Button::selectable(pick, ph::SELECT))
                    .on_hover_text("Selectable in the viewport (the manipulator)")
                    .clicked()
                {
                    edit = Some((format!("/viewport/select/{name}"), !pick));
                }
                ui.end_row();
            }
        });
        ui.separator();
        let mut extras = policy.extras;
        if ui
            .checkbox(&mut extras, "Extras")
            .on_hover_text("Mask shapes, group locators, the other cameras and the lights")
            .changed()
        {
            edit = Some(("/viewport/extras".into(), extras));
        }
        if let Some((path, value)) = edit {
            self.emit_app(super::AppAction::ViewportFlag { path, value });
        }
    }
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
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
            body: 11.0,
            small: 10.0,
            heading: 18.0,
            monospace: 13.0,
            zoom: 1.0,
        }
    }
}
impl App {
    pub(super) fn fonts_ui(&mut self, ui: &mut egui::Ui, fonts: &mut Fonts) {
        soluwell_panels::prefs::section_header(ui, "Fonts & UI scale");
        let definitions = egui::FontDefinitions::default();
        egui::ComboBox::from_id_salt("ui-font")
            .selected_text(&fonts.face)
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut fonts.face, "Default".into(), "Default");
                for name in definitions.font_data.keys() {
                    ui.selectable_value(&mut fonts.face, name.clone(), name);
                }
            });
        egui::Grid::new("font-sizes").num_columns(2).show(ui, |ui| {
            for (label, value) in [
                ("Body / controls", &mut fonts.body),
                ("Small text", &mut fonts.small),
                ("Headings", &mut fonts.heading),
                ("Monospace", &mut fonts.monospace),
            ] {
                ui.label(label);
                ui.add(egui::DragValue::new(value).range(8.0..=36.0).suffix(" pt"));
                ui.end_row();
            }
            ui.label("UI scale");
            ui.add(egui::Slider::new(&mut fonts.zoom, 0.75..=2.0));
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
        soluwell_config::add_icon_font(&mut definitions);
        ctx.set_fonts(definitions);
        ctx.set_zoom_factor(self.fonts.zoom);
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

    /// A 1600 x 900 dock, as the window gives it.
    fn area() -> egui::Rect {
        egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1600.0, 900.0))
    }

    /// The floating group holding `panel`.
    fn floating(
        layout: &Layout<Panel>,
        panel: Panel,
    ) -> Option<&soluwell_chrome::dock_layout::Floating<Panel>> {
        layout.floating.iter().find(|g| g.panels.contains(&panel))
    }

    #[test]
    fn layout_cache_reuses_idle_blob_and_observes_authored_edits() {
        let mut layout = default_layout();
        let mut cache = LayoutCache::default();
        cache.refresh(&layout).unwrap();
        let pointer = cache.blob.as_ref().unwrap().as_ptr();
        let revision = cache.revision;
        for _ in 0..32 {
            cache.refresh(&layout).unwrap();
        }
        assert_eq!(cache.serializations, 1);
        assert_eq!(cache.revision, revision);
        assert_eq!(cache.blob.as_ref().unwrap().as_ptr(), pointer);
        layout
            .apply(Action::Activate {
                panel: Panel::Gallery,
            })
            .unwrap();
        cache.refresh(&layout).unwrap();
        assert_eq!(cache.serializations, 2);
        assert_ne!(cache.revision, revision);
        layout
            .apply(Action::Close {
                panel: Panel::Export,
            })
            .unwrap();
        open_panel(&mut layout, Panel::Export, None, area()).unwrap();
        cache.refresh(&layout).unwrap();
        let restored = restore_layout(cache.blob.as_deref().unwrap()).unwrap();
        assert!(floating(&restored, Panel::Export).is_some());
        let count = cache.serializations;
        cache.refresh(&layout).unwrap();
        assert_eq!(cache.serializations, count);
    }

    #[test]
    fn fresh_and_floating_layouts_restore_and_tabs_do_not_duplicate() {
        let mut layout = default_layout();
        open_panel(&mut layout, Panel::Settings, None, area()).unwrap();
        open_panel(&mut layout, Panel::Settings, None, area()).unwrap();
        assert_eq!(
            layout
                .panels()
                .into_iter()
                .filter(|p| **p == Panel::Settings)
                .count(),
            1
        );
        let blob = soluwell_chrome::dock_layout::to_blob(&layout).unwrap();
        let restored = soluwell_chrome::dock_layout::from_blob::<Panel>(&blob).unwrap();
        assert!(valid_layout(&restored));
        assert_eq!(restored.panels().len(), 9);
    }

    /// A tool panel the focused group lacks joins that group; a tool window floats centred.
    #[test]
    fn panels_open_beside_the_focus_or_float() {
        let mut layout = default_layout();
        open_panel(&mut layout, Panel::Bookmarks, Some(Panel::Timeline), area()).unwrap();
        let blob = soluwell_chrome::dock_layout::to_blob(&layout).unwrap();
        assert!(blob.contains(r#"["Timeline","Bookmarks"]"#), "{blob}");
        open_panel(&mut layout, Panel::MaterialLibrary, None, area()).unwrap();
        let group = floating(&layout, Panel::MaterialLibrary).unwrap();
        assert_eq!(group.rect, [375.0, 150.0, 850.0, 600.0]);
    }

    #[test]
    fn floating_geometry_survives_a_blob() {
        let mut layout = Layout {
            root: Some(tabs(&[Panel::Viewport])),
            floating: Vec::new(),
        };
        open_panel(&mut layout, Panel::Settings, None, area()).unwrap();
        layout
            .apply(Action::MoveFloating {
                panel: Panel::Settings,
                rect: [100.0, 120.0, 700.0, 500.0],
            })
            .unwrap();
        let blob = soluwell_chrome::dock_layout::to_blob(&layout).unwrap();
        let restored = soluwell_chrome::dock_layout::from_blob::<Panel>(&blob).unwrap();
        assert_eq!(
            floating(&restored, Panel::Settings).unwrap().rect,
            [100.0, 120.0, 700.0, 500.0]
        );
    }

    #[test]
    fn named_layout_preserves_floating_export_and_default_workspaces() {
        let default_blob = soluwell_chrome::dock_layout::to_blob(&Layout {
            root: Some(tabs(&[
                Panel::Viewport,
                Panel::Timeline,
                Panel::Outliner,
                Panel::Gallery,
                Panel::Bookmarks,
                Panel::Materials,
                Panel::Inspector,
            ])),
            floating: Vec::new(),
        })
        .unwrap();
        let mut layout = Layout {
            root: Some(tabs(&[Panel::Viewport])),
            floating: Vec::new(),
        };
        open_panel(&mut layout, Panel::Settings, None, area()).unwrap();
        open_panel(&mut layout, Panel::Export, None, area()).unwrap();
        layout
            .apply(Action::MoveFloating {
                panel: Panel::Export,
                rect: [125.0, 80.0, 700.0, 500.0],
            })
            .unwrap();
        let mut store = LayoutStore::new();
        store.save("Default", default_blob).unwrap();
        store
            .save(
                "Render",
                soluwell_chrome::dock_layout::to_blob(&layout).unwrap(),
            )
            .unwrap();
        store.rename("Render", "Export workspace").unwrap();
        let json = serde_json::to_string(&store).unwrap();
        let mut restored_store: LayoutStore = serde_json::from_str(&json).unwrap();
        let restored = restore_layout(&restored_store.select("Export workspace").unwrap()).unwrap();
        assert_eq!(restored.panels().len(), 3);
        assert_eq!(
            floating(&restored, Panel::Export).unwrap().rect,
            [125.0, 80.0, 700.0, 500.0]
        );
        let default_restored = restore_layout(&restored_store.select("Default").unwrap()).unwrap();
        assert_eq!(default_restored.panels().len(), 7);
        restored_store.delete("Export workspace").unwrap();
        assert_eq!(restored_store.len(), 1);
    }

    #[test]
    fn incomplete_or_duplicate_layouts_are_rejected() {
        let only = |panels: &[Panel]| Layout {
            root: Some(tabs(panels)),
            floating: Vec::new(),
        };
        assert!(valid_layout(&default_layout()));
        assert!(!valid_layout(&only(&[Panel::Inspector])));
        assert!(!valid_layout(&only(&[Panel::Viewport, Panel::Viewport])));
    }

    /// A layout saved by the egui_dock-era build is refused, not misread: the app falls back to
    /// its default workspace once.
    #[test]
    fn pre_engine_blobs_are_refused() {
        assert!(restore_layout(r#"{"surfaces":[]}"#).is_err());
    }

    #[test]
    fn window_resize_grows_viewport_and_keeps_secondary_panels_fixed() {
        struct Measure(std::collections::HashMap<Panel, egui::Rect>);
        impl DockContent<Panel> for Measure {
            fn title(&self, panel: &Panel) -> String {
                panel.title().into()
            }
            fn permissions(&self, panel: &Panel) -> Permissions {
                panel.permissions()
            }
            fn body(&mut self, ui: &mut egui::Ui, panel: &Panel) {
                self.0.insert(*panel, ui.max_rect());
            }
        }

        let mut layout = default_layout();
        let ctx = egui::Context::default();
        let mut run_at = |w: f32, h: f32| {
            let mut m = Measure(Default::default());
            let raw = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(w, h),
                )),
                ..Default::default()
            };
            let mut out = ctx.run_ui(raw, |root| {
                egui::CentralPanel::default().show(root, |ui| {
                    let style = DockStyle::from_ui(ui);
                    let area = ui.available_rect_before_wrap();
                    layout.pin_around(|p| *p == Panel::Viewport, area, style.gap);
                    let dock_out = DockArea::new(egui::Id::new("test-warpbro-dock"))
                        .show_with(ui, &layout, &style, &mut m);
                    for action in dock_out.actions {
                        layout
                            .apply_with_permissions(action, Panel::permissions)
                            .unwrap();
                    }
                });
            });
            out.textures_delta.clear();
            m.0
        };

        let small = run_at(1400.0, 900.0);
        let large = run_at(2200.0, 1300.0);

        for sec in [
            Panel::Outliner,
            Panel::Takes,
            Panel::Inspector,
            Panel::Settings,
        ] {
            assert!(
                (large[&sec].width() - small[&sec].width()).abs() < 0.5,
                "{sec:?} width changed on window resize"
            );
        }
        for sec_h in [Panel::Timeline, Panel::Takes, Panel::Settings] {
            assert!(
                (large[&sec_h].height() - small[&sec_h].height()).abs() < 0.5,
                "{sec_h:?} height changed on window resize"
            );
        }
        assert_eq!(
            (large[&Panel::Viewport].width() - small[&Panel::Viewport].width()).round(),
            800.0
        );
        assert_eq!(
            (large[&Panel::Viewport].height() - small[&Panel::Viewport].height()).round(),
            400.0
        );
    }
}
