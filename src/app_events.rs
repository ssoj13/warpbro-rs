//! One deferred mutation boundary for the actual Playa EventBus.
use super::*;
use crate::events::{Action, UiCommand, UiEvent};
use crate::world::NodeId;

impl App {
    pub(crate) fn event_emitter(&self) -> playa_events::EventEmitter {
        self.event_bus.emitter()
    }
    pub(crate) fn emit_app(&mut self, action: AppAction) {
        let camera = matches!(
            &action,
            AppAction::CameraPatch(_)
                | AppAction::CameraOrbit { .. }
                | AppAction::CameraPan { .. }
                | AppAction::CameraZoom { .. }
                | AppAction::CameraFly { .. }
                | AppAction::CameraPose { .. }
                | AppAction::FrameCamera { .. }
                | AppAction::FrameCurrentCamera
                | AppAction::RestoreCamera
                | AppAction::ToggleFlight
                | AppAction::RecenterCamera
                | AppAction::Fly(_)
                | AppAction::Orbit(_)
                | AppAction::FocusDistance(_)
                | AppAction::Exposure { .. }
                | AppAction::RecallCameraSlot(_)
                | AppAction::StoreCameraSlot(_)
        )
        .then_some(self.world.document.active_camera);
        let document = match &action {
            AppAction::SettingsPatch { .. }
            | AppAction::PatchSettings(_)
            | AppAction::ShowUi(_)
            | AppAction::OpenSettings(_)
            | AppAction::OpenPanel(_)
            | AppAction::NewScene
            | AppAction::ResetLayout
            | AppAction::SaveLayout(_)
            | AppAction::LayoutStore(_)
            | AppAction::RestoreLayout(_)
            | AppAction::DockAction(_)
            | AppAction::Exit => None,
            _ => Some(NodeId(self.world.document.graph.id)),
        };
        if let Err(error) = crate::events::emit(
            &self.event_emitter(),
            UiCommand {
                document,
                frame: Some(f64::from(self.world_ui.playhead)),
                camera,
                action: Action::App(action),
            },
        ) {
            self.status = error;
        }
    }
    pub(crate) fn emit_take(&mut self, action: takes_ui::Action) {
        self.emit_take_for_document(action, NodeId(self.world.document.graph.id));
    }
    pub(crate) fn emit_take_for_document(&mut self, action: takes_ui::Action, document: NodeId) {
        let action = self.capture_take_action(action);
        if let Err(error) = crate::events::emit(
            &self.event_emitter(),
            UiCommand {
                document: Some(document),
                frame: Some(f64::from(self.world_ui.playhead)),
                camera: None,
                action: Action::Take(action),
            },
        ) {
            self.status = error;
        }
    }
    /// Drain at explicit frame boundaries, never in an immediate EventBus subscriber.
    pub(crate) fn handle_events(&mut self, ctx: &egui::Context) {
        // Handlers may publish a follow-up result. Each pass consumes the one shared bus;
        // no private command queue or alternate state-application path is involved.
        loop {
            let events = self.event_bus.poll();
            if events.is_empty() {
                break;
            }
            for boxed in events {
                let Some(event) = playa_events::downcast_event::<UiEvent>(&boxed) else {
                    self.status = format!("Unhandled EventBus event: {}", (*boxed).type_name());
                    continue;
                };
                if event.expired() {
                    event.respond(Err("Event expired before execution".into()));
                    continue;
                }
                let outcome = self.dispatch_command(event.command.clone(), ctx);
                if let Err(error) = &outcome {
                    self.status = error.clone();
                    log::error!("EventBus: {error}");
                }
                event.respond(outcome);
            }
        }
    }
    fn dispatch_command(
        &mut self,
        command: UiCommand,
        ctx: &egui::Context,
    ) -> Result<app_control::Reply, String> {
        if command
            .document
            .is_some_and(|id| id.0 != self.world.document.graph.id)
        {
            return Err("Event belongs to a document that is no longer open".into());
        }
        if let Some(camera) = command.camera
            && camera != self.world.document.active_camera
        {
            return Err("Event belongs to a camera that is no longer active".into());
        }
        let frame = command
            .frame
            .unwrap_or_else(|| f64::from(self.world_ui.playhead));
        let value = match command.action {
            Action::App(action) => self.handle_app_action(action, command.frame, ctx)?,
            Action::World(action) => {
                if matches!(
                    action,
                    crate::world_events::WorldAction::Seek(_)
                        | crate::world_events::WorldAction::Playing(false)
                ) {
                    self.preview.cancel();
                    self.preview_key = None;
                    self.showing_preview = false;
                }
                let value =
                    crate::world_events::apply(&mut self.world, &mut self.world_ui, action, frame)?;
                self.evaluated_world = None;
                value
            }
            Action::Take(action) => self.handle_take_action(action, frame, ctx)?,
            Action::Query(query) => return self.control_query(&query, ctx),
        };
        self.refresh_scene()?;
        ctx.request_repaint();
        Ok(app_control::Reply::Json(value))
    }
}
