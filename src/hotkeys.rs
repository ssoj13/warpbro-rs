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
    Duplicate,
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
    /// Time cursor to the work area start / end.
    Start,
    End,
    /// Work area start / end at the time cursor.
    SetStart,
    SetEnd,
    /// Time mark slot 0-9: set at the time cursor, or jump to it.
    SetMark(u8),
    Mark(u8),
}

#[derive(Clone, Copy)]
pub struct Binding {
    pub command: Command,
    pub key: Key,
    pub modifiers: Modifiers,
    additive: bool,
    /// Match the key position, not the character: Shift+1 types "!" on a US layout and
    /// "!" / "№" elsewhere, but is always the physical 1 key.
    physical: bool,
}
const fn binding(command: Command, key: Key, modifiers: Modifiers) -> Binding {
    Binding {
        command,
        key,
        modifiers,
        additive: false,
        physical: false,
    }
}
const fn property(command: Command, key: Key) -> Binding {
    Binding {
        command,
        key,
        modifiers: Modifiers::NONE,
        additive: true,
        physical: false,
    }
}
const fn physical(command: Command, key: Key, modifiers: Modifiers) -> Binding {
    Binding {
        command,
        key,
        modifiers,
        additive: false,
        physical: true,
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
    binding(Command::Duplicate, Key::D, Modifiers::COMMAND),
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
    binding(Command::Start, Key::Home, Modifiers::NONE),
    binding(Command::End, Key::End, Modifiers::NONE),
    binding(Command::SetStart, Key::B, Modifiers::NONE),
    binding(Command::SetEnd, Key::N, Modifiers::NONE),
    physical(Command::SetMark(0), Key::Num0, Modifiers::SHIFT),
    physical(Command::SetMark(1), Key::Num1, Modifiers::SHIFT),
    physical(Command::SetMark(2), Key::Num2, Modifiers::SHIFT),
    physical(Command::SetMark(3), Key::Num3, Modifiers::SHIFT),
    physical(Command::SetMark(4), Key::Num4, Modifiers::SHIFT),
    physical(Command::SetMark(5), Key::Num5, Modifiers::SHIFT),
    physical(Command::SetMark(6), Key::Num6, Modifiers::SHIFT),
    physical(Command::SetMark(7), Key::Num7, Modifiers::SHIFT),
    physical(Command::SetMark(8), Key::Num8, Modifiers::SHIFT),
    physical(Command::SetMark(9), Key::Num9, Modifiers::SHIFT),
    physical(Command::Mark(0), Key::Num0, Modifiers::NONE),
    physical(Command::Mark(1), Key::Num1, Modifiers::NONE),
    physical(Command::Mark(2), Key::Num2, Modifiers::NONE),
    physical(Command::Mark(3), Key::Num3, Modifiers::NONE),
    physical(Command::Mark(4), Key::Num4, Modifiers::NONE),
    physical(Command::Mark(5), Key::Num5, Modifiers::NONE),
    physical(Command::Mark(6), Key::Num6, Modifiers::NONE),
    physical(Command::Mark(7), Key::Num7, Modifiers::NONE),
    physical(Command::Mark(8), Key::Num8, Modifiers::NONE),
    physical(Command::Mark(9), Key::Num9, Modifiers::NONE),
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
fn matches(binding: &Binding, key: Key, physical_key: Option<Key>, mut modifiers: Modifiers) -> bool {
    if binding.additive {
        modifiers.shift = false;
    }
    let key = if binding.physical { physical_key.unwrap_or(key) } else { key };
    // Exact chords (Ctrl / Cmd equivalent): the pressed modifiers are `self`, the binding the
    // pattern. The reversed `matches_logically` let a Shift binding fire without Shift, so
    // plain 3 hit "set mark 3" and table order alone kept Insert apart from Shift + Insert.
    binding.key == key && modifiers.matches_exact(binding.modifiers)
}
/// Global-first lookup is shared by every consumer, independent of panel draw order.
/// `physical_key` is the key position an event reports (see [`Binding::physical`]).
fn resolve(
    scope: Scope,
    key: Key,
    physical_key: Option<Key>,
    modifiers: Modifiers,
) -> Option<(Scope, Command)> {
    GLOBAL
        .iter()
        .find(|b| matches(b, key, physical_key, modifiers))
        .map(|b| (Scope::Global, b.command))
        .or_else(|| {
            bindings(scope)
                .iter()
                .find(|b| matches(b, key, physical_key, modifiers))
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
        // Remove the exact event: a physical binding's logical key may be any character.
        let index = input.events.iter().position(|event| {
            matches!(event, egui::Event::Key {
                key,
                physical_key,
                pressed: true,
                repeat: false,
                modifiers,
            } if resolve(scope, *key, *physical_key, *modifiers) == Some((scope, command)))
        });
        index.map(|i| input.events.remove(i)).is_some()
    })
}
/// The platform Copy request (Ctrl+C / Cmd+C, Ctrl+Insert). egui-winit turns it into
/// `Event::Copy`, not a key event. Consumed only when `wanted` and no text editor has focus,
/// so text fields and selectable labels keep their own copy.
pub fn take_copy(ctx: &Context, wanted: bool) -> bool {
    if !wanted || ctx.text_edit_focused() {
        return false;
    }
    ctx.input_mut(|input| {
        let before = input.events.len();
        input.events.retain(|event| !matches!(event, egui::Event::Copy));
        input.events.len() != before
    })
}
/// The platform Paste (`Event::Paste`, clipboard text) when `accept` claims it and no text
/// editor has focus; other text stays for the widgets.
pub fn take_paste(ctx: &Context, accept: impl Fn(&str) -> bool) -> Option<String> {
    if ctx.text_edit_focused() {
        return None;
    }
    ctx.input_mut(|input| {
        let index = input
            .events
            .iter()
            .position(|event| matches!(event, egui::Event::Paste(text) if accept(text)))?;
        match input.events.remove(index) {
            egui::Event::Paste(text) => Some(text),
            _ => None,
        }
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
    fn chords_match_exactly_and_digits_by_key_position() {
        let t = Scope::Timeline;
        // A Shift binding never fires without Shift, a plain one never with it.
        assert_eq!(resolve(t, Key::Insert, None, Modifiers::NONE), Some((t, Command::Preview)));
        assert_eq!(resolve(t, Key::Insert, None, Modifiers::SHIFT), Some((t, Command::CachePreview)));
        assert_eq!(resolve(t, Key::Num3, Some(Key::Num3), Modifiers::NONE), Some((t, Command::Mark(3))));
        // Shift + 3 types "!" (US) or "№" (no egui key) but is the physical 3.
        let shifted = resolve(t, Key::Exclamationmark, Some(Key::Num3), Modifiers::SHIFT);
        assert_eq!(shifted, Some((t, Command::SetMark(3))));
        // Property filters keep their additive Shift.
        assert_eq!(resolve(t, Key::U, None, Modifiers::SHIFT), Some((t, Command::Keyed)));
    }
    #[test]
    fn global_commands_reserve_chords_and_panel_tables_are_distinct() {
        assert_eq!(
            resolve(Scope::Timeline, Key::Z, None, Modifiers::COMMAND),
            Some((Scope::Global, Command::Undo))
        );
        assert_eq!(
            resolve(Scope::Viewport, Key::Z, None, Modifiers::COMMAND.plus(Modifiers::SHIFT)),
            Some((Scope::Global, Command::Redo))
        );
        assert_eq!(
            resolve(Scope::Timeline, Key::F, None, Modifiers::NONE),
            Some((Scope::Timeline, Command::Fit))
        );
        assert_eq!(
            resolve(Scope::Viewport, Key::F, None, Modifiers::NONE),
            Some((Scope::Viewport, Command::Fit))
        );
        assert_eq!(resolve(Scope::Export, Key::F, None, Modifiers::NONE), None);
        assert_eq!(resolve(Scope::Timeline, Key::F, None, Modifiers::COMMAND), None);
        assert_eq!(
            resolve(Scope::Timeline, Key::Insert, None, Modifiers::SHIFT),
            Some((Scope::Timeline, Command::CachePreview))
        );
    }
}
