# Этап 3: Outliner и слои Timeline

Статус: реализация и итоговая проверка выполнены. [Общий план](plan.md). Зависит от [миграции](phase-02-world-migration.md).

## Реализовано 2026-10-02

[world_ui.rs](../../src/world_ui.rs) содержит Outliner на `egui-outliner`, Inspector и объектный Timeline на `egui-track-timeline`. Панели используют общий `WorldEditor.selection/selected` с UUID узлов; create, duplicate, delete, rename, reparent, reorder, свойства, metadata и ключи проходят через команды и атомарные транзакции undo/redo.

Timeline строит слой из узла, показывает его span, раскрывает группы свойств и отдельные компоненты. Stopwatch включает/выключает анимацию, diamond добавляет/удаляет ключ на playhead; поддерживаются выбор, перенос, удаление ключей и смена interpolation. UI identity ключа состоит из UUID узла, property path и битового представления времени `f64`; отдельные persistent KeyId/TrackId в документе не создаются.

`[start, end)` и visibility наследуются через parent; lock предка блокирует редактирование. UI playback использует целочисленный playhead, хотя сохранённые ключи и evaluator поддерживают `f64`. Типизированные inspector/camera controls передают изменённые значения через `edit_snapshot` после сообщения об изменении; сравнение не выполняется безусловно каждый кадр.

**Принятое решение:** solo — сохраняемое свойство слоя, как в AE. Host `/solo` одинаково фильтрует вычисленную сцену в preview и export; visibility также влияет на рендер. Selection определяет цель редактирования и не фильтрует геометрию.

Ниже сохранён исходный перечень UI-сценариев.

## Изменения

- [Cargo.toml](C:/projects/projects.rust.cg/cglibs/frac-rs/Cargo.toml): подключить egui-outliner совместимой версии, как в colmap.
- Новый C:/projects/projects.rust.cg/cglibs/frac-rs/src/outliner.rs: проекция parent tree, создание/duplicate/delete, rename, reparent, reorder; stable NodeId в selection.
- [dock.rs](C:/projects/projects.rust.cg/cglibs/frac-rs/src/dock.rs): панель Outliner и миграция сохранённых layouts.
- [inspector.rs](C:/projects/projects.rust.cg/cglibs/frac-rs/src/inspector.rs): свойства выбранного типа из schema + Playa Attrs, metadata editor на RustBox, stopwatch/diamond у каждой строки.
- [timeline.rs](C:/projects/projects.rust.cg/cglibs/frac-rs/src/timeline.rs): слои объектов с clip span, раскрытие свойств и компонентных channels, адаптер к shared egui-track-timeline canvas.
- [app.rs](C:/projects/projects.rust.cg/cglibs/frac-rs/src/app.rs): общий selection/commands/undo, camera gestures в тот же edit pathway.

## Поведение

Outliner, Inspector и Timeline показывают один документ и синхронизируют выбор. Timeline имеет объектную строку, её span и дочерние свойства: например Fractal → Transform → Position X/Y/Z. Нет общего dropdown всех параметров сцены.

Lanes строятся из schema и анимированных runtime attrs, включая custom numeric attrs. UI key identity использует Playa KeyId/TrackId и stable NodeId; keymove/delete/interpolation, multi-selection и undo не теряют компоненты.

Span объекта полуоткрытый [start, end); вне него объект неактивен и в preview, и в export. Неактивность parent распространяется на descendants. По умолчанию обычный объект действует весь диапазон мира.

Stopwatch включает канал у свойства; diamond добавляет/удаляет ключ текущего времени. Поведение редактирования при включённой анимации берётся из Playa, явно показывается состояние keyed/interpolated.

Visibility управляет присутствием в мире, lock предотвращает редакторские команды, solo фильтрует мир одинаково в preview и export и сохраняется в документе. Reparent не переставляет wires, порядок слоёв не меняет физическое перекрытие.

## Проверка и выход

UI-сценарий: создать второй Fractal, выделить в Outliner, изменить TRS/material в Inspector, добавить component key в Timeline, переименовать/reparent/reorder, scrub, undo/redo. Во всех панелях остаются те же UUID и key targets. Проверить lock, visibility, solo и сохранённый dock layout.
