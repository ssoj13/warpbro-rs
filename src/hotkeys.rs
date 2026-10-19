//! Shared DCC command routing. Global bindings reserve a chord before any panel
//! sees it. The visible panel under the pointer owns panel commands; keyboard
//! focus survives pointer exit. Text editors keep their keys.
use egui::{Context, Id, Key, Modifiers, Rect, Ui};

/// What a click on a slot strip (CamClip, colour presets) asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotClick {
    /// Put the current value into the slot.
    Store,
    /// Bring the slot's value back.
    Recall,
}

/// The one mouse mapping of every slot strip: the left button recalls and the right stores, or
/// the reverse with `swap` (Settings > Controls > Swap copy/paste mouse buttons).
pub fn slot_click(response: &egui::Response, swap: bool) -> Option<SlotClick> {
    let (left, right) = if swap {
        (SlotClick::Store, SlotClick::Recall)
    } else {
        (SlotClick::Recall, SlotClick::Store)
    };
    if response.clicked() {
        Some(left)
    } else if response.secondary_clicked() {
        Some(right)
    } else {
        None
    }
}

/// The hover line that names the mapping, e.g. "Left click: paste · Right click: copy".
pub fn slot_hint(swap: bool, store: &str, recall: &str) -> String {
    let (left, right) = if swap {
        (store, recall)
    } else {
        (recall, store)
    };
    format!("Left click: {left} · Right click: {right}")
}

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
    Takes,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Command {
    /// Hide / show every panel: the viewport alone, full window.
    ToggleUi,
    /// The Add menu at the cursor: type to filter, Enter creates (Houdini / Nuke Tab menu).
    AddNode,
    /// Save a screenshot of the whole window (EXR + PQ PNG, `window_shot`).
    WindowShot,
    /// File > New scene / Open / Save / Save as.
    NewScene,
    OpenScene,
    SaveScene,
    SaveSceneAs,
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
    /// Delete the panel's selection (Takes, Outliner).
    Delete,
    /// Timeline: show the modified properties of the selected layers (After Effects U U).
    Modified,
    /// Viewport tools: Q select (no manipulator), W move, E rotate, R scale.
    Tool(crate::app::ViewTool),
    /// Viewport manipulator size (Settings > Controls), while a gizmo tool is on.
    GizmoLarger,
    GizmoSmaller,
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
    /// Fires on a double tap of the chord ([`DOUBLE_TAP_SECONDS`]) instead of a press.
    double: bool,
}

/// Two presses of one chord within this many seconds are a double tap: a binding made with
/// [`double_tap`] fires on the second press (the first fired the chord's own binding).
pub const DOUBLE_TAP_SECONDS: f64 = 0.4;
const fn binding(command: Command, key: Key, modifiers: Modifiers) -> Binding {
    Binding {
        command,
        key,
        modifiers,
        additive: false,
        physical: false,
        double: false,
    }
}
const fn property(command: Command, key: Key) -> Binding {
    Binding {
        command,
        key,
        modifiers: Modifiers::NONE,
        additive: true,
        physical: false,
        double: false,
    }
}
const fn physical(command: Command, key: Key, modifiers: Modifiers) -> Binding {
    Binding {
        command,
        key,
        modifiers,
        additive: false,
        physical: true,
        double: false,
    }
}
/// `binding` fired by a double tap of its chord (`property` matching: Shift is additive).
const fn double_tap(binding: Binding) -> Binding {
    Binding {
        double: true,
        ..binding
    }
}
pub const GLOBAL: &[Binding] = &[
    // The key left of 1 on every layout (` on US, ё on Russian).
    physical(Command::ToggleUi, Key::Backtick, Modifiers::NONE),
    binding(Command::AddNode, Key::Tab, Modifiers::NONE),
    binding(Command::WindowShot, Key::F12, Modifiers::NONE),
    binding(Command::NewScene, Key::N, Modifiers::COMMAND),
    binding(Command::OpenScene, Key::O, Modifiers::COMMAND),
    binding(Command::SaveScene, Key::S, Modifiers::COMMAND),
    binding(
        Command::SaveSceneAs,
        Key::S,
        Modifiers::COMMAND.plus(Modifiers::SHIFT),
    ),
    binding(Command::Undo, Key::Z, Modifiers::COMMAND),
    binding(
        Command::Redo,
        Key::Z,
        Modifiers::COMMAND.plus(Modifiers::SHIFT),
    ),
    binding(Command::Duplicate, Key::D, Modifiers::COMMAND),
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
];
pub const VIEWPORT: &[Binding] = &[
    binding(Command::Fit, Key::F, Modifiers::NONE),
    binding(Command::Home, Key::H, Modifiers::NONE),
    property(Command::Flight, Key::L),
    binding(Command::Play, Key::Space, Modifiers::NONE),
    binding(
        Command::Tool(crate::app::ViewTool::Select),
        Key::Q,
        Modifiers::NONE,
    ),
    binding(
        Command::Tool(crate::app::ViewTool::Move),
        Key::W,
        Modifiers::NONE,
    ),
    binding(
        Command::Tool(crate::app::ViewTool::Rotate),
        Key::E,
        Modifiers::NONE,
    ),
    binding(
        Command::Tool(crate::app::ViewTool::Scale),
        Key::R,
        Modifiers::NONE,
    ),
    // + is Plus on the numpad and Shift+= on the main row; shift is ignored so both land.
    property(Command::GizmoLarger, Key::Plus),
    property(Command::GizmoLarger, Key::Equals),
    property(Command::GizmoSmaller, Key::Minus),
];
pub const TAKES: &[Binding] = &[binding(Command::Delete, Key::Delete, Modifiers::NONE)];
pub const OUTLINER: &[Binding] = &[binding(Command::Delete, Key::Delete, Modifiers::NONE)];
pub const TIMELINE: &[Binding] = &[
    binding(Command::Fit, Key::F, Modifiers::NONE),
    property(Command::Translate, Key::P),
    property(Command::Translate, Key::T),
    property(Command::Rotate, Key::R),
    property(Command::Scale, Key::S),
    property(Command::Keyed, Key::U),
    double_tap(property(Command::Modified, Key::U)),
    binding(Command::In, Key::I, Modifiers::NONE),
    binding(Command::Out, Key::O, Modifiers::NONE),
    binding(Command::Play, Key::Space, Modifiers::NONE),
    binding(Command::Start, Key::Home, Modifiers::NONE),
    binding(Command::End, Key::End, Modifiers::NONE),
    binding(Command::SetStart, Key::B, Modifiers::NONE),
    binding(Command::SetEnd, Key::N, Modifiers::NONE),
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
        Scope::Takes => TAKES,
        Scope::Outliner => OUTLINER,
        _ => &[],
    }
}
fn matches(
    binding: &Binding,
    key: Key,
    physical_key: Option<Key>,
    mut modifiers: Modifiers,
) -> bool {
    if binding.additive {
        modifiers.shift = false;
    }
    let key = if binding.physical {
        physical_key.unwrap_or(key)
    } else {
        key
    };
    // Exact chords (Ctrl / Cmd equivalent): the pressed modifiers are `self`, the binding the
    // pattern. The reversed `matches_logically` let a Shift binding fire without Shift, so
    // plain 3 hit "set mark 3" and table order alone kept Insert apart from Shift + Insert.
    binding.key == key && modifiers.matches_exact(binding.modifiers)
}
/// Global-first lookup is shared by every consumer, independent of panel draw order.
/// `physical_key` is the key position an event reports (see [`Binding::physical`]).
/// A `double` press (the second of a double tap, [`double_taps`]) takes a double-tap binding
/// first and is an ordinary press where there is none.
fn resolve(
    scope: Scope,
    key: Key,
    physical_key: Option<Key>,
    modifiers: Modifiers,
    double: bool,
) -> Option<(Scope, Command)> {
    let find = |double: bool| {
        GLOBAL
            .iter()
            .find(|b| b.double == double && matches(b, key, physical_key, modifiers))
            .map(|b| (Scope::Global, b.command))
            .or_else(|| {
                bindings(scope)
                    .iter()
                    .find(|b| b.double == double && matches(b, key, physical_key, modifiers))
                    .map(|b| (scope, b.command))
            })
    };
    double.then(|| find(true)).flatten().or_else(|| find(false))
}

/// The chord presses of this frame that are double taps, and the last press before them.
#[derive(Clone, Default)]
struct Taps {
    frame: u64,
    last: Option<(Key, Modifiers, f64)>,
    doubles: Vec<(Key, Modifiers)>,
}
fn taps_id() -> Id {
    Id::new("dcc-hotkey-taps")
}
/// The chords pressed this frame as the second press of a double tap: the chord pressed last,
/// again within [`DOUBLE_TAP_SECONDS`] (a third press starts over). Settled once per frame,
/// on the first ask, before any consumer removes the frame's events.
fn double_taps(ctx: &Context) -> Vec<(Key, Modifiers)> {
    let frame = ctx.cumulative_frame_nr();
    let mut taps = ctx
        .data(|d| d.get_temp::<Taps>(taps_id()))
        .unwrap_or_default();
    if taps.frame != frame || frame == 0 && taps.last.is_none() && taps.doubles.is_empty() {
        taps.frame = frame;
        taps.doubles.clear();
        let (now, presses) = ctx.input(|input| {
            let presses: Vec<(Key, Modifiers)> = input
                .events
                .iter()
                .filter_map(|event| match event {
                    egui::Event::Key {
                        key,
                        pressed: true,
                        repeat: false,
                        modifiers,
                        ..
                    } => Some((*key, *modifiers)),
                    _ => None,
                })
                .collect();
            (input.time, presses)
        });
        for (key, modifiers) in presses {
            if taps.last.is_some_and(|(k, m, at)| {
                k == key && m == modifiers && now - at <= DOUBLE_TAP_SECONDS
            }) {
                taps.doubles.push((key, modifiers));
                taps.last = None;
            } else {
                taps.last = Some((key, modifiers, now));
            }
        }
        ctx.data_mut(|d| d.insert_temp(taps_id(), taps.clone()));
    }
    taps.doubles
}
fn owner_id() -> Id {
    Id::new("dcc-hotkey-owner")
}
#[derive(Clone)]
struct PanelRegion {
    id: Id,
    scope: Scope,
    rect: Rect,
    layer: egui::LayerId,
    viewport: egui::ViewportId,
    frame: u64,
}
fn regions_id() -> Id {
    Id::new("dcc-hotkey-regions")
}
pub fn active(ctx: &Context) -> Option<Scope> {
    let regions = ctx
        .data(|data| data.get_temp::<Vec<PanelRegion>>(regions_id()))
        .unwrap_or_default();
    let frame = ctx.cumulative_frame_nr();
    let viewport = ctx.viewport_id();
    // Previous-frame geometry is available before any panel draws. Resolve using
    // this frame's pointer and egui's layer hit test, independent of draw order.
    let hovered = regions.iter().rev().find(|region| {
        region.viewport == viewport
            && region.frame.saturating_add(1) >= frame
            && ctx.rect_contains_pointer(region.layer, region.rect)
    });
    hovered.map(|region| region.scope).or_else(|| {
        // A click outside every panel clears retained keyboard focus immediately.
        let obscured = ctx.pointer_hover_pos().is_some_and(|pointer| {
            regions.iter().any(|region| {
                region.viewport == viewport
                    && region.frame.saturating_add(1) >= frame
                    && ctx
                        .layer_transform_to_global(region.layer)
                        .map_or(region.rect, |transform| transform * region.rect)
                        .contains(pointer)
            })
        });
        if obscured || ctx.input(|input| input.pointer.any_pressed()) {
            None
        } else {
            ctx.data(|data| data.get_temp(owner_id()))
        }
    })
}
/// Call for every panel, including panels with empty command tables. Clip and
/// layer checks prevent an obscured panel from acquiring ownership.
pub fn register(ui: &Ui, scope: Scope, rect: Rect) {
    let frame = ui.ctx().cumulative_frame_nr();
    let viewport = ui.ctx().viewport_id();
    let region = PanelRegion {
        id: ui.id(),
        scope,
        rect: rect.intersect(ui.clip_rect()),
        layer: ui.layer_id(),
        viewport,
        frame,
    };
    ui.ctx().data_mut(|data| {
        let regions = data.get_temp_mut_or_default::<Vec<PanelRegion>>(regions_id());
        regions.retain(|r| r.viewport != viewport || r.frame.saturating_add(1) >= frame);
        if let Some(previous) = regions
            .iter_mut()
            .find(|r| r.id == region.id && r.scope == scope && r.viewport == viewport)
        {
            *previous = region;
        } else {
            regions.push(region);
        }
    });
    let inside = ui.rect_contains_pointer(rect);
    let pressed = ui.input(|input| input.pointer.any_pressed());
    let owner = ui.ctx().data(|data| data.get_temp::<Scope>(owner_id()));
    if inside && pressed {
        ui.ctx()
            .data_mut(|data| data.insert_temp(owner_id(), scope));
    } else if pressed && !inside && owner == Some(scope) && active(ui.ctx()).is_none() {
        ui.ctx().data_mut(|data| data.remove::<Scope>(owner_id()));
    }
}
/// The shortcut text of `command`'s first binding in `scope` ("Ctrl+S" / "⌘S"): menus show the
/// binding table itself, never a copy of it.
pub fn shortcut(ctx: &Context, scope: Scope, command: Command) -> Option<String> {
    bindings(scope)
        .iter()
        .find(|b| b.command == command)
        .map(|b| {
            let chord = ctx.format_shortcut(&egui::KeyboardShortcut::new(b.modifiers, b.key));
            if b.double {
                format!("{chord} {chord}")
            } else {
                chord
            }
        })
}

/// A menu item for `command`, labelled with its shortcut ([`shortcut`]).
pub fn menu_item(ui: &mut Ui, text: &str, scope: Scope, command: Command) -> egui::Response {
    let mut button = egui::Button::new(text);
    if let Some(keys) = shortcut(ui.ctx(), scope, command) {
        button = button.shortcut_text(keys);
    }
    ui.add(button)
}

/// True when any `TextEdit` or `DragValue` numeric field owns keyboard focus on this frame.
/// Latching per `cumulative_frame_nr()` ensures that if `Enter` or `Escape` surrenders focus
/// mid-frame inside a widget, any key events from that same frame are not consumed as hotkeys
/// after the UI pass, while the very next frame immediately accepts hotkeys again.
fn text_input_active(ctx: &Context) -> bool {
    let frame = ctx.cumulative_frame_nr();
    let latch_id = Id::new("dcc-hotkey-text-active-frame");
    let active_now = ctx.memory(|m| m.focused().is_some());
    if active_now {
        ctx.data_mut(|d| d.insert_temp(latch_id, frame));
        true
    } else {
        ctx.data(|d| d.get_temp::<u64>(latch_id)) == Some(frame)
    }
}

/// Consume exactly one non-repeated command event, only in its owning scope.
pub fn consume(ctx: &Context, scope: Scope, command: Command) -> bool {
    if text_input_active(ctx) || (scope != Scope::Global && active(ctx) != Some(scope)) {
        return false;
    }
    let doubles = double_taps(ctx);
    ctx.input_mut(|input| {
        // Remove the exact event: a physical binding's logical key may be any character.
        let index = input.events.iter().position(|event| {
            matches!(event, egui::Event::Key {
                key,
                physical_key,
                pressed: true,
                repeat: false,
                modifiers,
            } if resolve(
                scope,
                *key,
                *physical_key,
                *modifiers,
                doubles.contains(&(*key, *modifiers)),
            ) == Some((scope, command)))
        });
        index.map(|i| input.events.remove(i)).is_some()
    })
}
/// The platform Copy request (Ctrl+C / Cmd+C, Ctrl+Insert). egui-winit turns it into
/// `Event::Copy`, not a key event. Consumed only when `wanted` and no text editor has focus,
/// so text fields and selectable labels keep their own copy.
pub fn take_copy(ctx: &Context, wanted: bool) -> bool {
    if !wanted || text_input_active(ctx) {
        return false;
    }
    ctx.input_mut(|input| {
        let before = input.events.len();
        input
            .events
            .retain(|event| !matches!(event, egui::Event::Copy));
        input.events.len() != before
    })
}
/// The platform Paste (`Event::Paste`, clipboard text) when `accept` claims it and no text
/// editor has focus; other text stays for the widgets.
pub fn take_paste(ctx: &Context, accept: impl Fn(&str) -> bool) -> Option<String> {
    if text_input_active(ctx) {
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
    fn pointer_routes_commands_before_the_hovered_panel_draws() {
        for order in [
            [Scope::Viewport, Scope::Timeline],
            [Scope::Timeline, Scope::Viewport],
        ] {
            let ctx = Context::default();
            let viewport = Rect::from_min_max(egui::pos2(20.0, 20.0), egui::pos2(300.0, 200.0));
            let timeline = Rect::from_min_max(egui::pos2(20.0, 220.0), egui::pos2(600.0, 400.0));
            let draw = |pointer, key, expected| {
                let mut events = vec![egui::Event::PointerMoved(pointer)];
                if key {
                    events.push(egui::Event::Key {
                        key: Key::F,
                        physical_key: None,
                        pressed: true,
                        repeat: false,
                        modifiers: Modifiers::NONE,
                    });
                } else {
                    events.push(egui::Event::Key {
                        key: Key::F,
                        physical_key: None,
                        pressed: false,
                        repeat: false,
                        modifiers: Modifiers::NONE,
                    });
                }
                let mut output = ctx.run_ui(
                    egui::RawInput {
                        screen_rect: Some(Rect::from_min_size(
                            egui::Pos2::ZERO,
                            egui::vec2(800.0, 600.0),
                        )),
                        events,
                        ..Default::default()
                    },
                    |root| {
                        egui::CentralPanel::default().show(root, |ui| {
                            for scope in order {
                                let rect = if scope == Scope::Viewport {
                                    viewport
                                } else {
                                    timeline
                                };
                                register(ui, scope, rect);
                                assert_eq!(
                                    consume(ui.ctx(), scope, Command::Fit),
                                    key && scope == expected
                                );
                            }
                        });
                    },
                );
                output.textures_delta.clear();
            };
            draw(viewport.center(), false, Scope::Viewport);
            // Move and press in one frame, without clicking to switch ownership.
            draw(timeline.center(), true, Scope::Timeline);
            draw(timeline.center(), false, Scope::Timeline);
            draw(viewport.center(), true, Scope::Viewport);
        }
    }
    #[test]
    fn overlays_and_panels_without_bindings_block_the_previous_owner() {
        for registered in [false, true] {
            let ctx = Context::default();
            let draw = |events: Vec<egui::Event>, covered: bool| {
                let mut fired = false;
                let mut output = ctx.run_ui(
                    egui::RawInput {
                        screen_rect: Some(Rect::from_min_size(
                            egui::Pos2::ZERO,
                            egui::vec2(800.0, 600.0),
                        )),
                        events,
                        ..Default::default()
                    },
                    |root| {
                        egui::CentralPanel::default().show(root, |ui| {
                            register(ui, Scope::Viewport, ui.max_rect());
                            fired = consume(ui.ctx(), Scope::Viewport, Command::Fit);
                        });
                        if covered {
                            egui::Area::new(Id::new("hotkey-overlay"))
                                .order(egui::Order::Foreground)
                                .fixed_pos(egui::pos2(40.0, 40.0))
                                .show(&ctx, |ui| {
                                    let (rect, _) = ui.allocate_exact_size(
                                        egui::vec2(200.0, 200.0),
                                        egui::Sense::hover(),
                                    );
                                    if registered {
                                        register(ui, Scope::Settings, rect);
                                    }
                                });
                        }
                    },
                );
                output.textures_delta.clear();
                fired
            };
            let point = egui::pos2(80.0, 80.0);
            draw(vec![egui::Event::PointerMoved(point)], false);
            draw(
                vec![egui::Event::PointerButton {
                    pos: point,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: Modifiers::NONE,
                }],
                false,
            );
            draw(
                vec![egui::Event::PointerButton {
                    pos: point,
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: Modifiers::NONE,
                }],
                true,
            );
            draw(vec![], true);
            assert!(!draw(
                vec![egui::Event::Key {
                    key: Key::F,
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers: Modifiers::NONE
                }],
                true
            ));
            assert_eq!(
                active(&ctx),
                if registered {
                    Some(Scope::Settings)
                } else {
                    None
                }
            );
        }
    }
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
            output.textures_delta.clear();
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
        output.textures_delta.clear();
        draw(vec![key.clone()], Scope::Viewport, Scope::Viewport);
        let mut output = ctx.run_ui(
            egui::RawInput {
                events: vec![release],
                ..Default::default()
            },
            |_| {},
        );
        output.textures_delta.clear();
        draw(vec![key], Scope::Export, Scope::Export);
    }
    #[test]
    fn chords_match_exactly_and_digits_by_key_position() {
        let t = Scope::Timeline;
        // A Shift binding never fires without Shift, a plain one never with it.
        assert_eq!(
            resolve(t, Key::Insert, None, Modifiers::NONE, false),
            Some((t, Command::Preview))
        );
        assert_eq!(
            resolve(t, Key::Insert, None, Modifiers::SHIFT, false),
            Some((t, Command::CachePreview))
        );
        assert_eq!(
            resolve(t, Key::Num3, Some(Key::Num3), Modifiers::NONE, false),
            Some((Scope::Global, Command::Mark(3)))
        );
        assert_eq!(
            resolve(
                Scope::Viewport,
                Key::Num3,
                Some(Key::Num3),
                Modifiers::NONE,
                false
            ),
            Some((Scope::Global, Command::Mark(3)))
        );
        // Shift + 3 types "!" (US) or "№" (no egui key) but is the physical 3.
        let shifted = resolve(
            Scope::AttributeEditor,
            Key::Exclamationmark,
            Some(Key::Num3),
            Modifiers::SHIFT,
            false,
        );
        assert_eq!(shifted, Some((Scope::Global, Command::SetMark(3))));
        // Property filters keep their additive Shift.
        assert_eq!(
            resolve(t, Key::U, None, Modifiers::SHIFT, false),
            Some((t, Command::Keyed))
        );
    }
    #[test]
    fn global_digit_marks_never_steal_keys_from_focused_numeric_fields() {
        let ctx = Context::default();
        let mut value = 1.0_f64;
        let mut field_rect = Rect::NOTHING;
        let step = |ctx: &Context, events: Vec<egui::Event>, value: &mut f64, rect: &mut Rect| {
            let mut mark_fired = false;
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(400.0, 200.0),
                    )),
                    events,
                    ..Default::default()
                },
                |root| {
                    // Poll global marks both before and after the UI to ensure neither steals input.
                    mark_fired |= consume(root, Scope::Global, Command::Mark(5));
                    mark_fired |= consume(root, Scope::Global, Command::SetMark(5));
                    egui::CentralPanel::default().show(root, |ui| {
                        let resp = ui.add(soluwell_config::num::drag(value));
                        *rect = resp.rect;
                    });
                    mark_fired |= consume(root, Scope::Global, Command::Mark(5));
                    mark_fired |= consume(root, Scope::Global, Command::SetMark(5));
                },
            );
            output.textures_delta.clear();
            mark_fired
        };

        // 1. Initial layout pass.
        assert!(!step(&ctx, vec![], &mut value, &mut field_rect));

        // 2. Click the numeric field to enter text editing mode.
        let center = field_rect.center();
        assert!(!step(
            &ctx,
            vec![
                egui::Event::PointerMoved(center),
                egui::Event::PointerButton {
                    pos: center,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: Modifiers::NONE,
                },
                egui::Event::PointerButton {
                    pos: center,
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: Modifiers::NONE,
                },
            ],
            &mut value,
            &mut field_rect,
        ));

        // 3. Type "5" and commit with Enter while the numeric field is focused.
        assert!(
            !step(
                &ctx,
                vec![
                    egui::Event::Key {
                        key: Key::Num5,
                        physical_key: Some(Key::Num5),
                        pressed: true,
                        repeat: false,
                        modifiers: Modifiers::NONE,
                    },
                    egui::Event::Text("5".into()),
                    egui::Event::Key {
                        key: Key::Num5,
                        physical_key: Some(Key::Num5),
                        pressed: false,
                        repeat: false,
                        modifiers: Modifiers::NONE,
                    },
                    egui::Event::Key {
                        key: Key::Enter,
                        physical_key: Some(Key::Enter),
                        pressed: true,
                        repeat: false,
                        modifiers: Modifiers::NONE,
                    },
                    egui::Event::Key {
                        key: Key::Enter,
                        physical_key: Some(Key::Enter),
                        pressed: false,
                        repeat: false,
                        modifiers: Modifiers::NONE,
                    },
                ],
                &mut value,
                &mut field_rect,
            ),
            "typing 5 in a focused numeric field must never trigger Command::Mark(5)"
        );
        assert_eq!(value, 5.0);

        // 4. Once committed (no longer editing), pressing 5 fires Command::Mark(5) globally.
        assert!(step(
            &ctx,
            vec![egui::Event::Key {
                key: Key::Num5,
                physical_key: Some(Key::Num5),
                pressed: true,
                repeat: false,
                modifiers: Modifiers::NONE,
            }],
            &mut value,
            &mut field_rect,
        ));
    }
    #[test]
    fn delete_belongs_to_the_outliner_and_the_takes() {
        for scope in [Scope::Outliner, Scope::Takes] {
            assert_eq!(
                resolve(scope, Key::Delete, None, Modifiers::NONE, false),
                Some((scope, Command::Delete))
            );
        }
        assert_eq!(
            resolve(Scope::Viewport, Key::Delete, None, Modifiers::NONE, false),
            None
        );
    }
    #[test]
    fn gizmo_size_keys_belong_to_the_viewport() {
        let v = Scope::Viewport;
        assert_eq!(
            resolve(v, Key::Plus, None, Modifiers::NONE, false),
            Some((v, Command::GizmoLarger))
        );
        assert_eq!(
            resolve(v, Key::Equals, None, Modifiers::SHIFT, false),
            Some((v, Command::GizmoLarger))
        );
        assert_eq!(
            resolve(v, Key::Minus, None, Modifiers::NONE, false),
            Some((v, Command::GizmoSmaller))
        );
        assert_eq!(
            resolve(Scope::Timeline, Key::Plus, None, Modifiers::NONE, false),
            None
        );
    }
    #[test]
    fn global_commands_reserve_chords_and_panel_tables_are_distinct() {
        assert_eq!(
            resolve(Scope::Timeline, Key::Z, None, Modifiers::COMMAND, false),
            Some((Scope::Global, Command::Undo))
        );
        assert_eq!(
            resolve(
                Scope::Viewport,
                Key::Z,
                None,
                Modifiers::COMMAND.plus(Modifiers::SHIFT),
                false
            ),
            Some((Scope::Global, Command::Redo))
        );
        assert_eq!(
            resolve(Scope::Timeline, Key::F, None, Modifiers::NONE, false),
            Some((Scope::Timeline, Command::Fit))
        );
        assert_eq!(
            resolve(Scope::Viewport, Key::F, None, Modifiers::NONE, false),
            Some((Scope::Viewport, Command::Fit))
        );
        assert_eq!(
            resolve(Scope::Export, Key::F, None, Modifiers::NONE, false),
            None
        );
        // The window screenshot and the scene file chords win in every panel.
        for (key, modifiers, command) in [
            (Key::F12, Modifiers::NONE, Command::WindowShot),
            (Key::N, Modifiers::COMMAND, Command::NewScene),
            (Key::O, Modifiers::COMMAND, Command::OpenScene),
            (Key::S, Modifiers::COMMAND, Command::SaveScene),
            (
                Key::S,
                Modifiers::COMMAND.plus(Modifiers::SHIFT),
                Command::SaveSceneAs,
            ),
        ] {
            assert_eq!(
                resolve(Scope::Timeline, key, None, modifiers, false),
                Some((Scope::Global, command))
            );
        }
        assert_eq!(
            resolve(Scope::Timeline, Key::S, None, Modifiers::NONE, false),
            Some((Scope::Timeline, Command::Scale))
        );
        assert_eq!(
            resolve(Scope::Timeline, Key::F, None, Modifiers::COMMAND, false),
            None
        );
        assert_eq!(
            resolve(Scope::Timeline, Key::Insert, None, Modifiers::SHIFT, false),
            Some((Scope::Timeline, Command::CachePreview))
        );
    }
}
