# Исследование Playa и colmap-rs

Дата: 2026-10-02. Основание: локальные исходники и исследовательские отчёты. [Общий план](../plan.md).

## Что переиспользуем

| Источник | Найденный контракт | Применение |
|---|---|---|
| [Graph node](C:/projects/projects.rust.cg/cglibs/playa/crates/playa-graph/src/node.rs:271) | Node с данными RustBox | Авторитетные объекты документа |
| [Graph container](C:/projects/projects.rust.cg/cglibs/playa/crates/playa-graph/src/graph_container.rs:23) | Graph | Хранение узлов и связей |
| [Attrs](C:/projects/projects.rust.cg/cglibs/playa/crates/playa-engine/src/entities/attrs.rs:511), [runtime keys](C:/projects/projects.rust.cg/cglibs/playa/crates/playa-engine/src/entities/attrs.rs:596) | Общие атрибуты; numeric runtime attrs поддерживают add_key | Динамические свойства и анимация |
| [Animation](C:/projects/projects.rust.cg/cglibs/playa/crates/playa-engine/src/entities/anim.rs:241) | Компонентные каналы и ключи | Сохранение и вычисление через Playa |
| [Transform](C:/projects/projects.rust.cg/cglibs/playa/crates/playa-engine/src/entities/transform.rs:1) | 3D T*R*Shear*S*T(-pivot), AE clockwise ZYX | Трансформации и совместимость |
| [Timing](C:/projects/projects.rust.cg/cglibs/playa/crates/playa-engine/src/entities/timing.rs:51) | Время и границы объекта | Layer clip span |
| [Scene evaluation](C:/projects/projects.rust.cg/cglibs/playa/crates/playa-engine/src/entities/scene.rs:578) | Существующий путь вычисления | Общий snapshot для всех render entrypoints |
| [AE inspector](C:/projects/projects.rust.cg/cglibs/playa/crates/playa-ui/src/widgets/ae/ae_ui.rs:106) | Свойства объекта и keyframe-поведение | Stopwatch у строки свойства |
| [KeyId](C:/projects/projects.rust.cg/cglibs/playa/crates/playa-ui/src/widgets/timeline/timeline.rs:50), [lanes](C:/projects/projects.rust.cg/cglibs/playa/crates/playa-ui/src/widgets/timeline/timeline_ui.rs:230) | Адреса ключей и дорожки UI | Адаптер слоёв frac |
| [TrackId](C:/projects/projects.rust.cg/cglibs/playa/crates/playa-events/src/ids.rs:39) | Идентификатор слоя | Отдельно от NodeId |
| [colmap outliner](C:/projects/projects.rust.cg/cglibs/colmap-rs/crates/colmap-ui/src/panels/recon_tree.rs:801) | egui-outliner | Дерево мировых объектов |

## Вывод

Прямое подключение существующих crates — baseline. Shared widgets получают адаптеры frac и динамические lanes из схемы и анимированных runtime attrs; фиксированный список Playa LANE_KEYS не ограничивает свойства frac. Модель и evaluator не копируются.

Текущий frac хранит один Scene и анимирует JSON pointer paths: [scene.rs](C:/projects/projects.rust.cg/cglibs/frac-rs/src/scene.rs), [animation.rs](C:/projects/projects.rust.cg/cglibs/frac-rs/src/animation.rs), [timeline.rs](C:/projects/projects.rust.cg/cglibs/frac-rs/src/timeline.rs). Это исходный формат для миграции, а не будущая модель мира.

CUDA сейчас упаковывает одну сцену: [params.rs](C:/projects/projects.rust.cg/cglibs/frac-rs/src/params.rs), [gpu.rs](C:/projects/projects.rust.cg/cglibs/frac-rs/src/gpu.rs), [render.rs](C:/projects/projects.rust.cg/cglibs/frac-rs/src/render.rs). Добавления дерева UI недостаточно: финальный этап должен трассировать несколько объектов.

## Проверки перед изменением модели

- Выбрать общую Playa git revision и проверить версии egui, serde, box-rs, curves, time/coord; записать реальные результаты сборки.
- Проверить сохранение произвольных RustBox полей при обновлении Attrs и расширяемость общей дискретной анимации.
- Зафиксировать migration mapping старых путей и единицы времени f64, без округления ключей.
- Подтвердить old yaw/rotation → Playa clockwise ZYX и конвенции камеры golden fixtures.
- Проверить нормали, singular TRS, nonuniform scale, parent composition и консервативный DE-bound.
