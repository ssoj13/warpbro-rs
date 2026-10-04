//! Shared DCC command routing. Global bindings reserve a chord before any panel
//! sees it. Panel focus survives pointer exit; a press in another panel changes
//! the owner. Text editors keep their keys. Static tables avoid per-frame maps.
use egui::{Context, Id, Key, Modifiers, Rect, Ui};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scope {
    Global,
    Viewport,
    Timeline,
    Gallery,
    Bookmarks,
    Materials,
    AttributeEditor,
    Settings,
    Export,
    Outliner,
    MaterialLibrary,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Command {
    ToggleUi,
    Undo,
    Redo,
    Fit,
    Home,
    Flight,
    Translate,
    Rotate,
    Scale,
    Keyed,
    In,
    Out,
    Play,
    Preview,
    CachePreview,
    DraftCachePreview,
}

#[derive(Clone, Copy)]
pub struct Binding {
    pub command: Command,
    pub key: Key,
    pub modifiers: Modifiers,
    additive: bool,
}
const fn binding(command: Command, key: Key, modifiers: Modifiers) -> Binding {
    Binding {
        command,
        key,
        modifiers,
        additive: false,
    }
}
const fn property(command: Command, key: Key) -> Binding {
    Binding {
        command,
        key,
        modifiers: Modifiers::NONE,
        additive: true,
    }
}
pub const GLOBAL: &[Binding] = &[
    binding(Command::ToggleUi, Key::Tab, Modifiers::NONE),
    binding(Command::Undo, Key::Z, Modifiers::COMMAND),
    binding(
        Command::Redo,
        Key::Z,
        Modifiers::COMMAND.plus(Modifiers::SHIFT),
    ),
];
pub const VIEWPORT: &[Binding] = &[
    binding(Command::Fit, Key::F, Modifiers::NONE),
    binding(Command::Home, Key::H, Modifiers::NONE),
    property(Command::Flight, Key::Backtick),
    binding(Command::Play, Key::Space, Modifiers::NONE),
];
pub const TIMELINE: &[Binding] = &[
    binding(Command::Fit, Key::F, Modifiers::NONE),
    property(Command::Translate, Key::P),
    property(Command::Translate, Key::T),
    property(Command::Rotate, Key::R),
    property(Command::Scale, Key::S),
    property(Command::Keyed, Key::U),
    binding(Command::In, Key::I, Modifiers::NONE),
    binding(Command::Out, Key::O, Modifiers::NONE),
    binding(Command::Play, Key::Space, Modifiers::NONE),
    binding(Command::Preview, Key::Insert, Modifiers::NONE),
    binding(Command::CachePreview, Key::Insert, Modifiers::SHIFT),
    binding(
        Command::DraftCachePreview,
        Key::Insert,
        Modifiers::CTRL.plus(Modifiers::SHIFT),
    ),
];
pub fn bindings(scope: Scope) -> &'static [Binding] {
    match scope {
        Scope::Global => GLOBAL,
        Scope::Viewport => VIEWPORT,
        Scope::Timeline => TIMELINE,
        _ => &[],
    }
}
fn matches(binding: &Binding, key: Key, mut modifiers: Modifiers) -> bool {
    if binding.additive {
        modifiers.shift = false;
    }
    binding.key == key && binding.modifiers.matches_logically(modifiers)
}
/// Global-first lookup is shared by every consumer, independent of panel draw order.
pub fn resolve(scope: Scope, key: Key, modifiers: Modifiers) -> Option<(Scope, Command)> {
    GLOBAL
        .iter()
        .find(|b| matches(b, key, modifiers))
        .map(|b| (Scope::Global, b.command))
        .or_else(|| {
            bindings(scope)
                .iter()
                .find(|b| matches(b, key, modifiers))
                .map(|b| (scope, b.command))
        })
}
fn owner_id() -> Id {
    Id::new("dcc-hotkey-owner")
}
pub fn active(ctx: &Context) -> Option<Scope> {
    ctx.data(|data| data.get_temp(owner_id()))
}
/// Call for every panel, including panels with empty command tables. Clip and
/// layer checks prevent an obscured panel from acquiring ownership.
pub fn register(ui: &Ui, scope: Scope, rect: Rect) {
    let inside = ui.rect_contains_pointer(rect);
    let pressed = ui.input(|input| input.pointer.any_pressed());
    let owner = active(ui.ctx());
    if inside && (pressed || owner.is_none()) {
        ui.ctx()
            .data_mut(|data| data.insert_temp(owner_id(), scope));
    } else if pressed && !inside && owner == Some(scope) {
        ui.ctx().data_mut(|data| data.remove::<Scope>(owner_id()));
    }
}
/// Consume exactly one non-repeated command event, only in its owning scope.
pub fn consume(ctx: &Context, scope: Scope, command: Command) -> bool {
    if ctx.text_edit_focused() || (scope != Scope::Global && active(ctx) != Some(scope)) {
        return false;
    }
    ctx.input_mut(|input| {
        let chord = input.events.iter().find_map(|event| match event {
            egui::Event::Key {
                key,
                pressed: true,
                repeat: false,
                modifiers,
                ..
            } if resolve(scope, *key, *modifiers) == Some((scope, command)) => {
                Some((*key, *modifiers))
            }
            _ => None,
        });
        chord.is_some_and(|(key, modifiers)| input.consume_key(modifiers, key))
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn one_press_routes_only_to_the_active_panel_and_is_consumed_once() {
        let ctx = Context::default();
        let key = egui::Event::Key {
            key: Key::F,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: Modifiers::NONE,
        };
        let draw = |events, owner, expected| {
            let mut count = 0;
            let mut output = ctx.run_ui(
                egui::RawInput {
                    events,
                    ..Default::default()
                },
                |root| {
                    egui::CentralPanel::default().show(root, |ui| {
                        ui.ctx()
                            .data_mut(|data| data.insert_temp(owner_id(), owner));
                        for scope in [Scope::Viewport, Scope::Timeline, Scope::Export] {
                            let fired = consume(ui.ctx(), scope, Command::Fit);
                            assert_eq!(fired, scope == expected && expected != Scope::Export);
                            count += usize::from(fired);
                            assert!(!consume(ui.ctx(), scope, Command::Fit));
                        }
                    });
                },
            );
            output.textures_delta = Default::default();
            assert_eq!(count, usize::from(expected != Scope::Export));
        };
        draw(vec![key.clone()], Scope::Timeline, Scope::Timeline);
        let mut release = key.clone();
        if let egui::Event::Key { pressed, .. } = &mut release {
            *pressed = false;
        }
        let mut output = ctx.run_ui(
            egui::RawInput {
                events: vec![release.clone()],
                ..Default::default()
            },
            |_| {},
        );
        output.textures_delta = Default::default();
        draw(vec![key.clone()], Scope::Viewport, Scope::Viewport);
        let mut output = ctx.run_ui(
            egui::RawInput {
                events: vec![release],
                ..Default::default()
            },
            |_| {},
        );
        output.textures_delta = Default::default();
        draw(vec![key], Scope::Export, Scope::Export);
    }
    #[test]
    fn global_commands_reserve_chords_and_panel_tables_are_distinct() {
        assert_eq!(
            resolve(Scope::Timeline, Key::Z, Modifiers::COMMAND),
            Some((Scope::Global, Command::Undo))
        );
        assert_eq!(
            resolve(
                Scope::Viewport,
                Key::Z,
                Modifiers::COMMAND.plus(Modifiers::SHIFT)
            ),
            Some((Scope::Global, Command::Redo))
        );
        assert_eq!(
            resolve(Scope::Timeline, Key::F, Modifiers::NONE),
            Some((Scope::Timeline, Command::Fit))
        );
        assert_eq!(
            resolve(Scope::Viewport, Key::F, Modifiers::NONE),
            Some((Scope::Viewport, Command::Fit))
        );
        assert_eq!(resolve(Scope::Export, Key::F, Modifiers::NONE), None);
        assert_eq!(resolve(Scope::Timeline, Key::F, Modifiers::COMMAND), None);
        assert_eq!(
            resolve(Scope::Timeline, Key::Insert, Modifiers::SHIFT),
            Some((Scope::Timeline, Command::CachePreview))
        );
    }
}
