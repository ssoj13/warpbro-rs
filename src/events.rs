//! The single typed action protocol shared by widgets, hotkeys and REST.
use crate::world::NodeId;
use playa_events::EventEmitter;
#[derive(Clone)]
pub(crate) enum Action {
    App(Box<crate::app::AppAction>),
    World(crate::world_events::WorldAction),
    Take(crate::app::takes_ui::Action),
    Query(String),
}
#[derive(Clone)]
pub(crate) struct UiCommand {
    /// A UI intent is bound to the document it observed. HTTP may explicitly address
    /// current state by leaving this absent; concrete IDs still undergo domain validation.
    pub document: Option<NodeId>,
    pub frame: Option<f64>,
    /// Reject a delayed viewport intent if another camera became active.
    pub camera: Option<Option<NodeId>>,
    pub action: Action,
}
pub(crate) type UiEvent = app_control::Event<UiCommand>;
pub(crate) fn emit(emitter: &EventEmitter, command: UiCommand) -> Result<(), String> {
    emitter
        .try_emit(UiEvent::ui(command))
        .map_err(|_| "EventBus is full; action was not published".into())
}
