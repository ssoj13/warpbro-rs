//! Typed world-panel intents shared by widgets and external control.
use crate::world::{NodeId, WorldAttribute, WorldCommand, WorldEditor};
use crate::world_ui::{AddMenuMemory, EffectParams, PreviewAction, WorldUi};
use playa_events::EventEmitter;
use serde_json::Value;

pub(crate) type KeyAddress = (NodeId, String, u64);

#[derive(Clone, Debug)]
pub(crate) enum WorldAction {
    Edit {
        command: WorldCommand,
        gesture: Option<u64>,
    },
    Metadata {
        id: NodeId,
        value: Value,
    },
    EditKeys {
        command: WorldCommand,
        keys: Vec<KeyAddress>,
    },
    Selection {
        ids: Vec<NodeId>,
        lead: Option<NodeId>,
    },
    ClickSelection {
        id: Option<NodeId>,
        add: bool,
    },
    SelectKeys {
        keys: Vec<KeyAddress>,
        add: bool,
        order: Vec<NodeId>,
        lead: Option<NodeId>,
    },
    ClearKeys,
    Channels {
        id: NodeId,
        expansion: soluwell_panels::attr_grid::ChannelExpansion,
    },
    PickTake(Option<u64>),
    Seek(u32),
    Playing(bool),
    Looping(bool),
    Preview(PreviewAction),
    TimelineOptions {
        auto_key: Option<bool>,
        new_key: Option<curves::Tan>,
        snap: Option<bool>,
        hide_shy: Option<bool>,
    },
    TimelineWidths {
        attribute: Option<f32>,
        outline: Option<f32>,
    },
    Effects(EffectParams),
    AddMenuMemory(AddMenuMemory),
    FileDialogs {
        directories: std::collections::HashMap<String, std::path::PathBuf>,
        filters: std::collections::HashMap<String, Option<String>>,
    },
    PropertyFilter {
        ids: Vec<NodeId>,
        bit: u8,
        additive: bool,
    },
    Expanded {
        id: NodeId,
        open: bool,
    },
    Group {
        id: NodeId,
        path: String,
        open: bool,
    },
    Channel {
        id: NodeId,
        path: String,
        open: bool,
    },
    Section {
        id: NodeId,
        group: String,
        open: bool,
    },
    Reveal {
        id: NodeId,
        attribute: WorldAttribute,
    },
    FitTimeline {
        width: f32,
    },
    OpenMaterialLibrary,
    FinishEdit {
        held: Option<u64>,
    },
}

pub(crate) fn emit_action(
    emitter: &EventEmitter,
    world: &WorldEditor,
    frame: f64,
    action: WorldAction,
) -> Result<(), String> {
    crate::events::emit(
        emitter,
        crate::events::UiCommand {
            document: Some(NodeId(world.document.graph.id)),
            frame: Some(frame),
            camera: None,
            action: crate::events::Action::World(action),
        },
    )
}

pub(crate) fn emit_edit(
    emitter: &EventEmitter,
    world: &WorldEditor,
    frame: f64,
    command: WorldCommand,
    gesture: Option<u64>,
) -> Result<(), String> {
    emit_action(
        emitter,
        world,
        frame,
        WorldAction::Edit { command, gesture },
    )
}

/// The only application point for world-panel events; widgets only receive a read view.
pub(crate) fn apply(
    world: &mut WorldEditor,
    ui: &mut WorldUi,
    action: WorldAction,
    frame: f64,
) -> Result<Value, String> {
    ui.apply_world_action(world, action, frame)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> (WorldEditor, WorldUi) {
        let scene = crate::scene::Scene::preset(crate::scene::Formula::Mandelbulb(
            crate::scene::Bulb::PRESET,
        ));
        (
            WorldEditor::new(crate::world::WorldDocument::from_scene(&scene)),
            WorldUi::default(),
        )
    }
    #[test]
    fn widget_command_is_deferred_until_the_shared_bus_dispatches() {
        let (mut world, mut ui) = fixture();
        let id = world.active().unwrap();
        let before = world.document.clone();
        emit_edit(
            ui.emitter(),
            &world,
            0.0,
            WorldCommand::Rename {
                id,
                name: "Through bus".into(),
            },
            None,
        )
        .unwrap();
        assert_eq!(world.document, before);
        ui.dispatch_events(&mut world);
        assert_eq!(world.document.info(id).unwrap().name, "Through bus");
        assert!(world.undo());
        assert_eq!(world.document, before);
    }
    #[test]
    fn separate_bus_drains_preserve_one_gesture_undo() {
        let (mut world, mut ui) = fixture();
        let id = world.active().unwrap();
        let before = world.document.clone();
        for name in ["Drag one", "Drag two"] {
            emit_edit(
                ui.emitter(),
                &world,
                0.0,
                WorldCommand::Rename {
                    id,
                    name: name.into(),
                },
                Some(91),
            )
            .unwrap();
            ui.dispatch_events(&mut world);
        }
        assert_eq!(world.active_edit(), Some(91));
        emit_action(
            ui.emitter(),
            &world,
            0.0,
            WorldAction::FinishEdit { held: None },
        )
        .unwrap();
        ui.dispatch_events(&mut world);
        assert!(world.active_edit().is_none());
        assert!(world.undo());
        assert_eq!(world.document, before);
        assert!(!world.undo());
    }
    #[test]
    fn selection_and_transport_use_the_same_deferred_bus() {
        let (mut world, mut ui) = fixture();
        let camera = world.document.active_camera.unwrap();
        let selection = world.selected().to_vec();
        let before = world.document.clone();
        emit_action(
            ui.emitter(),
            &world,
            0.0,
            WorldAction::Selection {
                ids: vec![camera],
                lead: Some(camera),
            },
        )
        .unwrap();
        emit_action(ui.emitter(), &world, 0.0, WorldAction::Seek(27)).unwrap();
        emit_action(ui.emitter(), &world, 0.0, WorldAction::Playing(true)).unwrap();
        assert_eq!(world.selected(), selection);
        assert_eq!(ui.playhead, 0);
        assert!(!ui.playing);
        ui.dispatch_events(&mut world);
        assert_eq!(world.selected(), &[camera]);
        assert_eq!(ui.playhead, 27);
        assert!(ui.playing);
        assert_eq!(world.document, before);
    }
}
