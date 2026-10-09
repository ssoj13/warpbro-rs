//! Render/quality catalog controls. All settings and references live in World nodes.
use crate::render_profiles::{
    CatalogRole, ProfileTarget, RenderMethod, ViewportMode, ViewportRender,
};
use crate::world::{NodeId, WorldCommand, WorldEditor, WorldKind, WorldNodeInfo};
use egui_widgets_config::icons as ph;
use serde_json::json;

#[derive(Default)]
pub(crate) struct Actions {
    pub edit: Option<NodeId>,
    pub error: Option<String>,
}
impl Actions {
    fn execute(&mut self, world: &mut WorldEditor, command: WorldCommand) -> bool {
        match world.execute(command) {
            Ok(()) => true,
            Err(error) => {
                self.error = Some(error);
                false
            }
        }
    }
    fn merge(&mut self, other: Self) {
        self.edit = other.edit.or(self.edit);
        self.error = other.error.or(self.error.take());
    }
}

#[derive(Clone)]
enum NameOperation {
    Create {
        source: Option<NodeId>,
        role: CatalogRole,
        target: Option<ProfileTarget>,
    },
    Instantiate {
        id: NodeId,
        target: ProfileTarget,
    },
    CreateQuality {
        source: Option<NodeId>,
        role: CatalogRole,
        target: Option<NodeId>,
    },
    InstantiateQuality {
        id: NodeId,
        target: Option<NodeId>,
    },
    CreateOutput {
        source: Option<NodeId>,
        role: CatalogRole,
        assign: bool,
    },
    InstantiateOutput {
        id: NodeId,
        assign: bool,
    },
    Rename(NodeId),
}
#[derive(Clone)]
struct NameDialog {
    operation: NameOperation,
    name: String,
    error: Option<String>,
}
fn dialog_id(world: &WorldEditor) -> egui::Id {
    egui::Id::new(("render_profile_name", world.document.graph.id))
}
fn request_name(ui: &egui::Ui, world: &WorldEditor, operation: NameOperation, name: String) {
    ui.data_mut(|data| {
        data.insert_temp(
            dialog_id(world),
            NameDialog {
                operation,
                name,
                error: None,
            },
        )
    });
}
fn name_command(dialog: &NameDialog) -> Result<WorldCommand, String> {
    let name = dialog.name.trim();
    if name.is_empty() {
        return Err("Enter a name.".into());
    }
    Ok(match dialog.operation {
        NameOperation::Create {
            source,
            role,
            target,
        } => WorldCommand::CreateRenderProfile {
            name: name.into(),
            role,
            source,
            target,
        },
        NameOperation::Instantiate { id, target } => WorldCommand::InstantiateRenderTemplate {
            id,
            name: name.into(),
            target: Some(target),
        },
        NameOperation::CreateQuality {
            source,
            role,
            target,
        } => WorldCommand::CreateQualityProfile {
            name: name.into(),
            role,
            source,
            target,
        },
        NameOperation::InstantiateQuality { id, target } => {
            WorldCommand::InstantiateQualityTemplate {
                id,
                name: name.into(),
                target,
            }
        }
        NameOperation::CreateOutput {
            source,
            role,
            assign,
        } => WorldCommand::CreateOutputSettings {
            name: name.into(),
            role,
            source,
            assign,
        },
        NameOperation::InstantiateOutput { id, assign } => {
            WorldCommand::InstantiateOutputTemplate {
                id,
                name: name.into(),
                assign,
            }
        }
        NameOperation::Rename(id) => WorldCommand::Rename {
            id,
            name: name.into(),
        },
    })
}
fn dialogs(ui: &egui::Ui, world: &mut WorldEditor) -> Actions {
    let mut actions = Actions::default();
    let id = dialog_id(world);
    let frame = ui.ctx().cumulative_frame_nr();
    let drawn = id.with("drawn");
    if ui.data(|data| data.get_temp::<u64>(drawn)) == Some(frame) {
        return actions;
    }
    ui.data_mut(|data| data.insert_temp(drawn, frame));
    let Some(mut dialog) = ui.data(|data| data.get_temp::<NameDialog>(id)) else {
        return actions;
    };
    let mut accepted = false;
    let mut cancelled = false;
    egui::Window::new(match dialog.operation {
        NameOperation::Rename(_) => "Rename settings",
        NameOperation::Instantiate { .. }
        | NameOperation::InstantiateQuality { .. }
        | NameOperation::InstantiateOutput { .. } => "Apply independent copy",
        NameOperation::Create {
            role: CatalogRole::Template,
            ..
        } => "Save render template",
        NameOperation::Create { .. } => "New render profile",
        NameOperation::CreateQuality {
            role: CatalogRole::Template,
            ..
        } => "Save quality template",
        NameOperation::CreateQuality { .. } => "New quality profile",
        NameOperation::CreateOutput {
            role: CatalogRole::Template,
            ..
        } => "Save output template",
        NameOperation::CreateOutput { .. } => "New output profile",
    })
    .id(id.with("window"))
    .collapsible(false)
    .resizable(false)
    .show(ui.ctx(), |ui| {
        ui.label("Name");
        let response = ui.add(egui::TextEdit::singleline(&mut dialog.name).desired_width(280.0));
        if let Some(error) = &dialog.error {
            ui.colored_label(egui::Color32::LIGHT_RED, error);
        }
        ui.horizontal(|ui| {
            accepted = ui.button("Save").clicked()
                || (response.has_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)));
            cancelled =
                ui.button("Cancel").clicked() || ui.input(|i| i.key_pressed(egui::Key::Escape));
        });
    });
    if accepted {
        match name_command(&dialog).and_then(|command| world.execute(command)) {
            Ok(()) => {
                actions.edit = world.selection;
                cancelled = true;
            }
            Err(error) => dialog.error = Some(error),
        }
    }
    ui.data_mut(|data| {
        if cancelled {
            data.remove::<NameDialog>(id);
        } else {
            data.insert_temp(id, dialog);
        }
    });
    actions
}

fn profile_name(nodes: &[WorldNodeInfo], id: NodeId) -> &str {
    nodes
        .iter()
        .find(|node| node.id == id)
        .map(|node| node.name.as_str())
        .unwrap_or("Missing profile")
}
fn binding_command(
    viewport: NodeId,
    target: ProfileTarget,
    id: NodeId,
    frame: f64,
) -> WorldCommand {
    match target {
        ProfileTarget::Output => WorldCommand::SetOutputRender(id),
        _ => WorldCommand::SetAttribute {
            id: viewport,
            path: target.viewport_path().expect("viewport target").into(),
            value: json!(id),
            frame,
        },
    }
}
fn target_name(target: ProfileTarget) -> &'static str {
    match target {
        ProfileTarget::Moving => "Moving",
        ProfileTarget::Still => "Still",
        ProfileTarget::Manual => "Manual",
        ProfileTarget::Output => "Output",
    }
}
/// How one settings catalog names its operations in one context. Selectors, menus and button
/// rows of every catalog (render, quality, output) are built from it, so they share one UI.
struct CatalogOps<'a> {
    /// A copy of `source` with `role`; None copies the catalog's default source (the
    /// assigned Output render, its quality, or the assigned output recipe).
    create: &'a dyn Fn(Option<NodeId>, CatalogRole) -> NameOperation,
    /// An independent profile instantiated from template `id`.
    instantiate: &'a dyn Fn(NodeId) -> NameOperation,
}
/// One binding row: `label`, the bound profile's combo (picking executes `bind`), the
/// Attribute Editor gear and the create / copy / template menu of `kind`'s catalog.
fn catalog_selector(
    ui: &mut egui::Ui,
    world: &mut WorldEditor,
    label: &str,
    kind: WorldKind,
    selected: NodeId,
    bind: &dyn Fn(NodeId) -> WorldCommand,
    ops: &CatalogOps,
) -> Actions {
    let mut actions = Actions::default();
    let nodes = world
        .document
        .settings_nodes(kind, Some(CatalogRole::Profile));
    ui.push_id(("profile_binding", label), |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.label(label);
            let mut candidate = selected;
            egui::ComboBox::from_id_salt("profile")
                .width(160.0)
                .selected_text(profile_name(&nodes, selected))
                .show_ui(ui, |ui| {
                    for node in &nodes {
                        ui.selectable_value(&mut candidate, node.id, &node.name);
                    }
                });
            if candidate != selected {
                actions.execute(world, bind(candidate));
            }
            if ui
                .button(ph::GEAR)
                .on_hover_text("Edit profile in Attribute Editor")
                .clicked()
            {
                actions.edit = Some(candidate);
            }
            ui.menu_button(ph::ADD, |ui| {
                if ui.button("New profile…").clicked() {
                    request_name(
                        ui,
                        world,
                        (ops.create)(None, CatalogRole::Profile),
                        format!("{label} profile"),
                    );
                    ui.close();
                }
                if ui.button("Independent copy…").clicked() {
                    request_name(
                        ui,
                        world,
                        (ops.create)(Some(candidate), CatalogRole::Profile),
                        format!("{} copy", profile_name(&nodes, candidate)),
                    );
                    ui.close();
                }
                let templates = world
                    .document
                    .settings_nodes(kind, Some(CatalogRole::Template));
                if !templates.is_empty() {
                    ui.separator();
                    ui.label("Apply template as independent copy");
                    for template in templates {
                        if ui.button(&template.name).clicked() {
                            request_name(ui, world, (ops.instantiate)(template.id), template.name);
                            ui.close();
                        }
                    }
                }
            });
        });
    });
    actions
}
/// A render-profile binding (Moving / Still / Manual / Output); copies made here are bound.
fn profile_selector(
    ui: &mut egui::Ui,
    world: &mut WorldEditor,
    viewport: NodeId,
    target: ProfileTarget,
    selected: NodeId,
    frame: f64,
) -> Actions {
    catalog_selector(
        ui,
        world,
        target_name(target),
        WorldKind::RenderSettings,
        selected,
        &|id| binding_command(viewport, target, id, frame),
        &CatalogOps {
            create: &|source, role| NameOperation::Create {
                source,
                role,
                target: Some(target),
            },
            instantiate: &|id| NameOperation::Instantiate { id, target },
        },
    )
}
/// The export bindings for the Render / Encode panel: `output_rows` plus the name dialog,
/// which every top-level entry draws once, after all its rows could request it.
pub(crate) fn output_bindings(ui: &mut egui::Ui, world: &mut WorldEditor, frame: f64) -> Actions {
    let mut actions = output_rows(ui, world, frame);
    actions.merge(dialogs(ui, world));
    actions
}
/// The Output render profile and the OutputSettings recipe rows, shared by the Render /
/// Encode panel and Settings → Render & Viewport.
fn output_rows(ui: &mut egui::Ui, world: &mut WorldEditor, frame: f64) -> Actions {
    let mut actions = Actions::default();
    match (
        world.document.viewport_settings_id(),
        world.document.output_render_profile(),
    ) {
        (Ok(viewport), Ok(output)) => actions.merge(profile_selector(
            ui,
            world,
            viewport,
            ProfileTarget::Output,
            output,
            frame,
        )),
        (Err(error), _) | (_, Err(error)) => actions.error = Some(error),
    }
    match world.document.output_settings_id() {
        Ok(recipe) => actions.merge(catalog_selector(
            ui,
            world,
            "Output file",
            WorldKind::OutputSettings,
            recipe,
            &WorldCommand::SetOutputSettings,
            &CatalogOps {
                create: &|source, role| NameOperation::CreateOutput {
                    source,
                    role,
                    assign: true,
                },
                instantiate: &|id| NameOperation::InstantiateOutput { id, assign: true },
            },
        )),
        Err(error) => actions.error = Some(error),
    }
    actions
}
fn binding_controls(
    ui: &mut egui::Ui,
    world: &mut WorldEditor,
    include_output: bool,
    frame: f64,
) -> Actions {
    let mut actions = Actions::default();
    let policy = match world.document.viewport_policy(frame) {
        Ok(policy) => policy,
        Err(error) => {
            actions.error = Some(error);
            return actions;
        }
    };
    let viewport = match world.document.viewport_settings_id() {
        Ok(viewport) => viewport,
        Err(error) => {
            actions.error = Some(error);
            return actions;
        }
    };
    for (target, selected) in [
        (ProfileTarget::Moving, policy.moving_id),
        (ProfileTarget::Still, policy.still_id),
        (ProfileTarget::Manual, policy.manual_id),
    ] {
        actions.merge(profile_selector(
            ui, world, viewport, target, selected, frame,
        ));
    }
    if include_output {
        actions.merge(output_rows(ui, world, frame));
    }
    actions
}

/// One line naming what the viewport renders, so motion switching is visible while it happens.
fn active_summary(nodes: &[WorldNodeInfo], active: &ViewportRender) -> String {
    let effective = &active.effective;
    format!(
        "{}: {} · {} · {} spp · {:.0}% resolution",
        target_name(active.target),
        profile_name(nodes, effective.profile),
        match effective.method {
            RenderMethod::Fast => "Fast",
            RenderMethod::Full => "Full",
        },
        effective.samples,
        effective.resolution_scale * 100.0
    )
}

/// Compact viewport controls share the same node references as the Settings panel. The
/// profile menu is labelled with the binding the viewport rendered last (`active`).
pub(crate) fn toolbar(
    ui: &mut egui::Ui,
    world: &mut WorldEditor,
    frame: f64,
    active: Option<&ViewportRender>,
) -> Actions {
    let mut actions = Actions::default();
    match (
        world.document.viewport_settings_id(),
        world.document.viewport_policy(frame),
    ) {
        (Ok(viewport), Ok(policy)) => {
            let mut mode = policy.mode;
            egui::ComboBox::from_id_salt("viewport_profile_mode")
                .width(76.0)
                .selected_text(match mode {
                    ViewportMode::Auto => "Auto",
                    ViewportMode::Locked => "Locked",
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut mode, ViewportMode::Auto, "Auto");
                    ui.selectable_value(&mut mode, ViewportMode::Locked, "Locked");
                });
            if mode != policy.mode {
                actions.execute(
                    world,
                    WorldCommand::SetAttribute {
                        id: viewport,
                        path: "/viewport/mode".into(),
                        value: json!(mode),
                        frame,
                    },
                );
            }
            const ROUTING: &str = "Auto uses Moving and Still profiles; Locked always uses Manual. Fast uses path tracing with approximate opaque materials and the selected quality.";
            let (label, hover) = match active {
                Some(active) => {
                    let nodes = world.document.render_profiles(Some(CatalogRole::Profile));
                    (
                        target_name(active.target),
                        format!("{}\n\n{ROUTING}", active_summary(&nodes, active)),
                    )
                }
                None => ("Profiles", ROUTING.to_owned()),
            };
            ui.menu_button(label, |ui| {
                actions.merge(binding_controls(ui, world, false, frame))
            })
            .response
            .on_hover_text(hover);
            for (path, before, label, tooltip) in [
                (
                    "/viewport/paused",
                    policy.paused,
                    ph::PAUSE,
                    "Pause accumulation",
                ),
                (
                    "/viewport/frozen",
                    policy.frozen,
                    ph::LOCK,
                    "Freeze the displayed image",
                ),
            ] {
                if ui
                    .selectable_label(before, label)
                    .on_hover_text(tooltip)
                    .clicked()
                {
                    actions.execute(
                        world,
                        WorldCommand::SetAttribute {
                            id: viewport,
                            path: path.into(),
                            value: json!(!before),
                            frame,
                        },
                    );
                }
            }
        }
        (Err(error), _) | (_, Err(error)) => actions.error = Some(error),
    }
    actions.merge(dialogs(ui, world));
    actions
}

fn render_catalog_target(ui: &mut egui::Ui, world: &WorldEditor) -> ProfileTarget {
    let id = egui::Id::new(("render_catalog_target", world.document.graph.id));
    let mut target = ui
        .data(|data| data.get_temp::<ProfileTarget>(id))
        .unwrap_or(ProfileTarget::Output);
    ui.horizontal_wrapped(|ui| {
        ui.label("Recall to");
        for choice in [
            ProfileTarget::Moving,
            ProfileTarget::Still,
            ProfileTarget::Manual,
            ProfileTarget::Output,
        ] {
            ui.selectable_value(&mut target, choice, target_name(choice));
        }
    });
    ui.data_mut(|data| data.insert_temp(id, target));
    target
}
fn quality_catalog_target(ui: &mut egui::Ui, world: &WorldEditor) -> Result<NodeId, String> {
    let output = world.document.output_render_profile()?;
    let nodes = world.document.render_profiles(Some(CatalogRole::Profile));
    let id = egui::Id::new(("quality_catalog_target", world.document.graph.id));
    let mut target = ui
        .data(|data| data.get_temp::<NodeId>(id))
        .filter(|id| nodes.iter().any(|node| node.id == *id))
        .unwrap_or(output);
    ui.horizontal_wrapped(|ui| {
        ui.label("Recall quality to");
        egui::ComboBox::from_id_salt("quality_catalog_target")
            .selected_text(profile_name(&nodes, target))
            .show_ui(ui, |ui| {
                for node in &nodes {
                    ui.selectable_value(&mut target, node.id, &node.name);
                }
            });
    });
    ui.data_mut(|data| data.insert_temp(id, target));
    Ok(target)
}
fn recall_render(
    ui: &egui::Ui,
    world: &mut WorldEditor,
    node: &WorldNodeInfo,
    target: ProfileTarget,
    frame: f64,
    actions: &mut Actions,
) {
    match world.document.catalog_role(node.id) {
        Ok(Some(CatalogRole::Template)) => request_name(
            ui,
            world,
            NameOperation::Instantiate {
                id: node.id,
                target,
            },
            node.name.clone(),
        ),
        Ok(_) => match world.document.viewport_settings_id() {
            Ok(viewport) => {
                actions.execute(world, binding_command(viewport, target, node.id, frame));
            }
            Err(error) => actions.error = Some(error),
        },
        Err(error) => actions.error = Some(error),
    }
}
fn recall_quality(
    ui: &egui::Ui,
    world: &mut WorldEditor,
    node: &WorldNodeInfo,
    target: NodeId,
    frame: f64,
    actions: &mut Actions,
) {
    match world.document.catalog_role(node.id) {
        Ok(Some(CatalogRole::Template)) => request_name(
            ui,
            world,
            NameOperation::InstantiateQuality {
                id: node.id,
                target: Some(target),
            },
            node.name.clone(),
        ),
        Ok(_) => {
            actions.execute(
                world,
                WorldCommand::SetAttribute {
                    id: target,
                    path: "/render/quality_id".into(),
                    value: json!(node.id),
                    frame,
                },
            );
        }
        Err(error) => actions.error = Some(error),
    }
}

fn recall_output(
    ui: &egui::Ui,
    world: &mut WorldEditor,
    node: &WorldNodeInfo,
    actions: &mut Actions,
) {
    match world.document.catalog_role(node.id) {
        Ok(Some(CatalogRole::Template)) => request_name(
            ui,
            world,
            NameOperation::InstantiateOutput {
                id: node.id,
                assign: true,
            },
            node.name.clone(),
        ),
        Ok(_) => {
            actions.execute(world, WorldCommand::SetOutputSettings(node.id));
        }
        Err(error) => actions.error = Some(error),
    }
}
/// An unbound copy of `source` (None: the catalog's default source) in `kind`'s catalog.
fn copy_operation(
    kind: WorldKind,
    source: Option<NodeId>,
    role: CatalogRole,
) -> Option<NameOperation> {
    Some(match kind {
        WorldKind::RenderSettings => NameOperation::Create {
            source,
            role,
            target: None,
        },
        WorldKind::QualitySettings => NameOperation::CreateQuality {
            source,
            role,
            target: None,
        },
        WorldKind::OutputSettings => NameOperation::CreateOutput {
            source,
            role,
            assign: false,
        },
        _ => return None,
    })
}
/// "New … profile / template" buttons of one catalog (`noun` names it, "" for render).
fn catalog_new_buttons(
    ui: &mut egui::Ui,
    world: &WorldEditor,
    kind: WorldKind,
    noun: &str,
    default_name: &str,
) {
    ui.horizontal_wrapped(|ui| {
        for (role, label) in [
            (CatalogRole::Profile, format!("New {noun}profile…")),
            (CatalogRole::Template, format!("New {noun}template…")),
        ] {
            if ui.button(label).clicked()
                && let Some(operation) = copy_operation(kind, None, role)
            {
                request_name(ui, world, operation, default_name.into());
            }
        }
    });
}
/// One catalog's named buttons, sorted by name: LMB `recall`s, RMB opens `settings_menu`.
/// A template carries the file icon.
fn catalog_buttons(
    ui: &mut egui::Ui,
    world: &mut WorldEditor,
    kind: WorldKind,
    hover: &str,
    recall: &mut dyn FnMut(&egui::Ui, &mut WorldEditor, &WorldNodeInfo, &mut Actions),
) -> Actions {
    let mut actions = Actions::default();
    let mut nodes = world.document.settings_nodes(kind, None);
    nodes.sort_by(|a, b| {
        a.name
            .cmp(&b.name)
            .then_with(|| a.id.to_string().cmp(&b.id.to_string()))
    });
    ui.horizontal_wrapped(|ui| {
        for node in nodes {
            let template =
                world.document.catalog_role(node.id).ok().flatten() == Some(CatalogRole::Template);
            let label = if template {
                format!("{} {}", ph::FILE, node.name)
            } else {
                node.name.clone()
            };
            let response = ui.button(label).on_hover_text(hover);
            if response.clicked() {
                recall(ui, world, &node, &mut actions);
            }
            response.context_menu(|ui| {
                actions.merge(settings_menu(ui, world, node.id, kind, &node.name))
            });
        }
    });
    actions
}

/// Viewport policy and profile parameters use the existing node Attribute Editor.
pub(crate) fn settings(
    ui: &mut egui::Ui,
    world: &mut WorldEditor,
    world_ui: &mut crate::world_ui::WorldUi,
) -> Actions {
    let frame = f64::from(world_ui.playhead);
    let mut actions = binding_controls(ui, world, true, frame);
    ui.separator();
    ui.label("Render profiles and templates");
    ui.weak(
        "Profiles are shared live. Applying a template creates an independent render/quality pair.",
    );
    catalog_new_buttons(ui, world, WorldKind::RenderSettings, "", "Render settings");
    let render_target = render_catalog_target(ui, world);
    actions.merge(catalog_buttons(
        ui,
        world,
        WorldKind::RenderSettings,
        "Recall to the selected target; right-click to edit or save settings",
        &mut |ui, world, node, actions| {
            recall_render(ui, world, node, render_target, frame, actions)
        },
    ));
    ui.separator();
    ui.label("Quality profiles and templates");
    catalog_new_buttons(
        ui,
        world,
        WorldKind::QualitySettings,
        "quality ",
        "Quality settings",
    );
    let quality_target = quality_catalog_target(ui, world);
    actions.merge(catalog_buttons(
        ui,
        world,
        WorldKind::QualitySettings,
        "Recall quality to the selected render profile; right-click to edit or save settings",
        &mut |ui, world, node, actions| match &quality_target {
            Ok(target) => recall_quality(ui, world, node, *target, frame, actions),
            Err(error) => actions.error = Some(error.clone()),
        },
    ));
    ui.separator();
    ui.label("Output file profiles and templates");
    ui.weak("The recipe Render / Encode writes: format, codec, colour and size.");
    catalog_new_buttons(
        ui,
        world,
        WorldKind::OutputSettings,
        "output ",
        "Output settings",
    );
    actions.merge(catalog_buttons(
        ui,
        world,
        WorldKind::OutputSettings,
        "Use as the export recipe; right-click to edit or save settings",
        &mut recall_output,
    ));
    ui.separator();
    match world.document.viewport_settings_id() {
        Ok(viewport) => world_ui.attribute_editor(ui, world, viewport),
        Err(error) => actions.error = Some(error),
    }
    actions.merge(dialogs(ui, world));
    actions
}
/// The right-click menu of a catalog settings node: edit, rename, save as profile / template
/// and, for a template, the independent copies its catalog can apply.
fn settings_menu(
    ui: &mut egui::Ui,
    world: &mut WorldEditor,
    id: NodeId,
    kind: WorldKind,
    name: &str,
) -> Actions {
    let mut actions = Actions::default();
    if ui.button("Edit in Attribute Editor").clicked() {
        actions.edit = Some(id);
        ui.close();
    }
    if ui.button("Rename…").clicked() {
        request_name(ui, world, NameOperation::Rename(id), name.into());
        ui.close();
    }
    for (role, label) in [
        (CatalogRole::Profile, "Save as profile…"),
        (CatalogRole::Template, "Save as template…"),
    ] {
        if ui.button(label).clicked()
            && let Some(operation) = copy_operation(kind, Some(id), role)
        {
            request_name(ui, world, operation, format!("{name} copy"));
            ui.close();
        }
    }
    if world.document.catalog_role(id).ok().flatten() != Some(CatalogRole::Template) {
        return actions;
    }
    let apply: Vec<(String, NameOperation)> = match kind {
        WorldKind::RenderSettings => [
            ProfileTarget::Moving,
            ProfileTarget::Still,
            ProfileTarget::Manual,
            ProfileTarget::Output,
        ]
        .into_iter()
        .map(|target| {
            (
                format!("Apply independent copy to {}", target_name(target)),
                NameOperation::Instantiate { id, target },
            )
        })
        .collect(),
        WorldKind::QualitySettings => std::iter::once((
            "Create independent quality profile…".to_owned(),
            NameOperation::InstantiateQuality { id, target: None },
        ))
        .chain(
            world
                .document
                .render_profiles(Some(CatalogRole::Profile))
                .into_iter()
                .map(|render| {
                    (
                        format!("Apply independent copy to {}", render.name),
                        NameOperation::InstantiateQuality {
                            id,
                            target: Some(render.id),
                        },
                    )
                }),
        )
        .collect(),
        WorldKind::OutputSettings => vec![
            (
                "Create independent output profile…".to_owned(),
                NameOperation::InstantiateOutput { id, assign: false },
            ),
            (
                "Apply independent copy as the export recipe".to_owned(),
                NameOperation::InstantiateOutput { id, assign: true },
            ),
        ],
        _ => vec![],
    };
    ui.separator();
    for (label, operation) in apply {
        if ui.button(label).clicked() {
            request_name(ui, world, operation, name.into());
            ui.close();
        }
    }
    actions
}

/// Catalog role is metadata on the canonical settings node, independent of its schema.
pub(crate) fn node_actions(
    ui: &mut egui::Ui,
    world: &mut WorldEditor,
    id: NodeId,
    kind: WorldKind,
) -> Actions {
    let mut actions = Actions::default();
    if !crate::render_profiles::setting_kind(kind) {
        return actions;
    }
    let before = world.document.catalog_role(id).ok().flatten();
    let mut role = before;
    ui.horizontal_wrapped(|ui| {
        egui::ComboBox::from_id_salt(("catalog_role", id))
            .selected_text(match role {
                Some(CatalogRole::Profile) => "Profile",
                Some(CatalogRole::Template) => "Template",
                None => "Settings node",
            })
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut role, Some(CatalogRole::Profile), "Profile");
                ui.selectable_value(&mut role, Some(CatalogRole::Template), "Template");
            });
        if kind == WorldKind::RenderSettings
            && let Ok(quality) = world.document.render_quality(id)
            && ui.button("Edit quality").clicked()
        {
            actions.edit = Some(quality);
        }
        if kind != WorldKind::ViewportSettings {
            let name = world
                .document
                .nodes()
                .into_iter()
                .find(|node| node.id == id)
                .map(|node| node.name)
                .unwrap_or_default();
            ui.menu_button("Catalog actions", |ui| {
                actions.merge(settings_menu(ui, world, id, kind, &name))
            });
        }
    });
    if role != before
        && let Some(role) = role
    {
        actions.execute(world, WorldCommand::SetCatalogRole { id, role });
    }
    actions.merge(dialogs(ui, world));
    actions
}

#[cfg(test)]
mod tests {
    use super::*;
    fn text_center(output: &egui::FullOutput, label: &str) -> egui::Pos2 {
        fn find(shape: &egui::epaint::Shape, label: &str) -> Option<egui::Pos2> {
            match shape {
                egui::epaint::Shape::Text(text) if text.galley.job.text == label => {
                    Some(text.pos + text.galley.rect.center().to_vec2())
                }
                egui::epaint::Shape::Vec(shapes) => {
                    shapes.iter().find_map(|shape| find(shape, label))
                }
                _ => None,
            }
        }
        output
            .shapes
            .iter()
            .find_map(|shape| find(&shape.shape, label))
            .unwrap_or_else(|| panic!("Missing UI text: {label}"))
    }
    fn pointer(pos: egui::Pos2, pressed: bool) -> Vec<egui::Event> {
        vec![
            egui::Event::PointerMoved(pos),
            egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: egui::Modifiers::NONE,
            },
        ]
    }
    fn frame(
        ctx: &egui::Context,
        events: Vec<egui::Event>,
        draw: impl FnOnce(&mut egui::Ui),
    ) -> egui::FullOutput {
        let mut draw = Some(draw);
        ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::Vec2::new(800.0, 900.0),
                )),
                events,
                ..Default::default()
            },
            |root| {
                egui::CentralPanel::default().show(root, |ui| draw.take().unwrap()(ui));
            },
        )
    }
    #[test]
    fn toolbar_pause_uses_node_command_and_preserves_output() {
        let at = 0.0;
        let mut world = WorldEditor::new(crate::world::WorldDocument::from_scene(
            &crate::scene::Scene::preset(crate::params::FAMILY_BULB),
        ));
        let output = world.document.output_render_profile().unwrap();
        let original = world.document.clone();
        let ctx = egui::Context::default();
        // The profile menu is labelled with the binding the viewport rendered last.
        let moving = world.document.viewport_render(at, true).unwrap();
        let draw = |ui: &mut egui::Ui, world: &mut WorldEditor| {
            ui.horizontal(|ui| {
                assert!(toolbar(ui, world, 0.0, Some(&moving)).error.is_none());
            });
        };
        let painted = frame(&ctx, vec![], |ui| draw(ui, &mut world));
        text_center(&painted, "Moving");
        let position = text_center(&painted, ph::PAUSE);
        frame(&ctx, pointer(position, true), |ui| draw(ui, &mut world));
        frame(&ctx, pointer(position, false), |ui| draw(ui, &mut world));
        assert!(world.document.viewport_policy(at).unwrap().paused);
        assert_eq!(world.document.output_render_profile().unwrap(), output);
        assert!(world.undo());
        assert_eq!(world.document, original);
    }
    #[test]
    fn settings_new_profile_button_creates_pair_in_one_undo() {
        let mut world = WorldEditor::new(crate::world::WorldDocument::from_scene(
            &crate::scene::Scene::preset(crate::params::FAMILY_BULB),
        ));
        let original = world.document.clone();
        let ctx = egui::Context::default();
        let mut world_ui = crate::world_ui::WorldUi::default();
        let mut draw = |events, world: &mut WorldEditor| {
            frame(&ctx, events, |ui| {
                assert!(settings(ui, world, &mut world_ui).error.is_none());
            })
        };
        let painted = draw(vec![], &mut world);
        let create = text_center(&painted, "New profile…");
        draw(pointer(create, true), &mut world);
        draw(pointer(create, false), &mut world);
        let dialog = draw(vec![], &mut world);
        let save = text_center(&dialog, "Save");
        draw(pointer(save, true), &mut world);
        draw(pointer(save, false), &mut world);
        let id = world.selection.unwrap();
        assert!(!original.nodes().iter().any(|node| node.id == id));
        let quality = world.document.render_quality(id).unwrap();
        assert!(!original.nodes().iter().any(|node| node.id == quality));
        assert!(world.undo());
        assert_eq!(world.document, original);
    }
    #[test]
    fn named_profile_button_recalls_live_output_with_one_undo() {
        let mut world = WorldEditor::new(crate::world::WorldDocument::from_scene(
            &crate::scene::Scene::preset(crate::params::FAMILY_BULB),
        ));
        world
            .execute(WorldCommand::CreateRenderProfile {
                name: "Recall custom render".into(),
                role: CatalogRole::Profile,
                source: None,
                target: None,
            })
            .unwrap();
        let profile = world.selection.unwrap();
        let original = world.document.clone();
        let viewport_before = world.document.viewport_policy(0.0).unwrap();
        let ctx = egui::Context::default();
        let mut world_ui = crate::world_ui::WorldUi::default();
        let mut draw = |events, world: &mut WorldEditor| {
            frame(&ctx, events, |ui| {
                assert!(settings(ui, world, &mut world_ui).error.is_none());
            })
        };
        let painted = draw(vec![], &mut world);
        let recall = text_center(&painted, "Recall custom render");
        draw(pointer(recall, true), &mut world);
        draw(pointer(recall, false), &mut world);
        assert_eq!(world.document.output_render_profile().unwrap(), profile);
        assert_eq!(
            world.document.viewport_policy(0.0).unwrap(),
            viewport_before
        );
        assert!(world.undo());
        assert_eq!(world.document, original);
    }
    #[test]
    fn named_template_button_applies_independent_pair_with_one_undo() {
        let mut world = WorldEditor::new(crate::world::WorldDocument::from_scene(
            &crate::scene::Scene::preset(crate::params::FAMILY_BULB),
        ));
        world
            .execute(WorldCommand::CreateRenderProfile {
                name: "Recall independent template".into(),
                role: CatalogRole::Template,
                source: None,
                target: None,
            })
            .unwrap();
        let template = world.selection.unwrap();
        let template_quality = world.document.render_quality(template).unwrap();
        let original = world.document.clone();
        let ctx = egui::Context::default();
        let mut world_ui = crate::world_ui::WorldUi::default();
        let mut draw = |events, world: &mut WorldEditor| {
            frame(&ctx, events, |ui| {
                assert!(settings(ui, world, &mut world_ui).error.is_none());
            })
        };
        let painted = draw(vec![], &mut world);
        let recall = text_center(
            &painted,
            &format!("{} Recall independent template", ph::FILE),
        );
        draw(pointer(recall, true), &mut world);
        draw(pointer(recall, false), &mut world);
        let dialog = draw(vec![], &mut world);
        let save = text_center(&dialog, "Save");
        draw(pointer(save, true), &mut world);
        draw(pointer(save, false), &mut world);
        let instance = world.document.output_render_profile().unwrap();
        assert_ne!(instance, template);
        assert_ne!(
            world.document.render_quality(instance).unwrap(),
            template_quality
        );
        assert_eq!(
            world.document.catalog_role(instance).unwrap(),
            Some(CatalogRole::Profile)
        );
        assert!(world.undo());
        assert_eq!(world.document, original);
    }
    #[test]
    fn names_are_trimmed_and_empty_names_are_rejected() {
        let dialog = NameDialog {
            operation: NameOperation::Create {
                source: None,
                role: CatalogRole::Profile,
                target: Some(ProfileTarget::Still),
            },
            name: "   Custom still   ".into(),
            error: None,
        };
        assert!(
            matches!(name_command(&dialog).unwrap(), WorldCommand::CreateRenderProfile { name, target: Some(ProfileTarget::Still), .. } if name == "Custom still")
        );
        assert!(
            name_command(&NameDialog {
                name: "  ".into(),
                ..dialog
            })
            .is_err()
        );
    }
    #[test]
    fn profile_assignments_use_typed_node_refs_and_output_is_separate() {
        let at = 0.0;
        let mut world = WorldEditor::new(crate::world::WorldDocument::from_scene(
            &crate::scene::Scene::preset(crate::params::FAMILY_BULB),
        ));
        let viewport = world.document.viewport_settings_id().unwrap();
        let output = world.document.output_render_profile().unwrap();
        let original = world.document.viewport_policy(at).unwrap();
        world
            .execute(WorldCommand::CreateRenderProfile {
                name: "Custom".into(),
                role: CatalogRole::Profile,
                source: Some(output),
                target: None,
            })
            .unwrap();
        let created = world.selection.unwrap();
        world
            .execute(binding_command(
                viewport,
                ProfileTarget::Still,
                created,
                0.0,
            ))
            .unwrap();
        assert_eq!(
            world.document.viewport_policy(at).unwrap().still_id,
            created
        );
        assert_eq!(world.document.output_render_profile().unwrap(), output);
        world.undo();
        assert_eq!(
            world.document.viewport_policy(at).unwrap().still_id,
            original.still_id
        );
        world
            .execute(binding_command(
                viewport,
                ProfileTarget::Output,
                created,
                0.0,
            ))
            .unwrap();
        assert_eq!(world.document.output_render_profile().unwrap(), created);
        assert_eq!(
            world.document.viewport_policy(at).unwrap().moving_id,
            original.moving_id
        );
    }
}
