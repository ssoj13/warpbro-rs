# Этап 1: подключить модель Playa

Статус: реализация и итоговая проверка выполнены. [Общий план](plan.md).

## Реализовано 2026-10-02

Модель, схема, миграция и команды объединены в [world.rs](../../src/world.rs). Отдельные файлы из исходного перечня ниже не создавались. Постоянный `WorldDocument` хранит `playa_graph::SubnetFile`, UUID активной камеры и окружения, диапазон кадров и FPS; `runtime_graph()` восстанавливает Playa `Graph` и `Node.data: RustBox`. Версию задаёт `SubnetFile.format_version`.

Данные узла разделены по ключам: `gpu` — сохраняемая авторитетная базовая конфигурация для построения GPU-параметров; `host` — сериализованные Playa `Attrs` и `Animation`; `discrete` — словари произвольных JSON-значений; `metadata` — пользовательский JSON. Вычисление начинается с `gpu` и накладывает значения `host` на заданном времени; CUDA device buffers создаются отдельно.

Числовые ключи и компоненты вычисляет Playa. Для bool, enum, string и структурных переключений host-анимация хранит числовой индекс словаря с `CurveKind::Step`; после sampling индекс декодируется в исходный JSON. Собственной реализации интерполяции нет, расширения Playa не потребовались. Динамические custom attrs и компонентные descriptors формируются из Attrs.

`WorldEditor::execute` атомарно применяет команду или `Batch`, откатывает документ и selection при ошибке и хранит снимки для undo/redo. Parent cycles запрещены; lock проверяется также у предков. Material assignment адресует Material-узел по UUID. Сохранённый `bus_slots.layer_order` хранит UUID узлов отдельно от parent/children и wires; отдельный persistent TrackId не введён.

Прямые Playa dependencies в Cargo.toml подключены по `ssh://git@github.com/ssoj13/playa.git` с общей revision `6b1c6c7d53522b696f859af400e0b141aaab2ba5`; `playa-engine` использует `default-features = false`. Локальные репозитории служат источником исследования; финальная политика всех Git dependencies — GitHub SSH и закреплённый commit. У widgets и box-rs текущий Cargo.toml ещё использует SSH `branch = "main"`; их точные commits закрепляет Cargo.lock. Camera crates и Standard Surface переведены с vendor paths на закреплённые SSH commits. Итоговый аудит подтвердил 214 Git packages по SSH без локальных path dependencies; повторные release build и 95 тестов прошли. Revisions и точные результаты описаны в [плане 1](../../plan1.md).

Ниже сохранён исходный перечень работ и контрактов для сопоставления с реализацией. Указанные в нём локальные пути Playa — исследовательские ссылки, а не выбранный путь подключения зависимости.

## Изменения

- [Cargo.toml](C:/projects/projects.rust.cg/cglibs/frac-rs/Cargo.toml), [Cargo.lock](C:/projects/projects.rust.cg/cglibs/frac-rs/Cargo.lock): подключить playa-graph, playa-engine, playa-time, playa-coord на одной revision; проверить совместимость имеющихся widgets и box-rs.
- Новый C:/projects/projects.rust.cg/cglibs/frac-rs/src/world.rs: WorldDocument с schema_version, Graph, активными camera/environment UUID и порядком TrackId; Node.data остаётся RustBox.
- Новый C:/projects/projects.rust.cg/cglibs/frac-rs/src/world_schema.rs: registry типов Fractal, Camera, DirectionalLight, Environment, Group, Material; descriptors атрибутов, единиц, defaults и применимости.
- Новый C:/projects/projects.rust.cg/cglibs/frac-rs/src/world_commands.rs: атомарные команды, события и undo/redo для свойств, ключей, создания/удаления, parent и порядка.
- [animation.rs](C:/projects/projects.rust.cg/cglibs/frac-rs/src/animation.rs): заменить прежний evaluator адаптером к Playa Attrs/Animation; старый формат оставить только входом migration.
- [main.rs](C:/projects/projects.rust.cg/cglibs/frac-rs/src/main.rs): зарегистрировать модули.

## Контракты

Сериализованные Playa Attrs и arbitrary metadata имеют одного владельца в Node.data. Декодированные views и caches производные; обратная запись сохраняет неизвестные поля. Адрес анимации содержит NodeId + attr key + component; f64 время не округляется.

Parent tree хранится отдельно от wires. Создание parent cycles запрещено. Переименование не меняет identity; layer TrackId и NodeId не смешиваются. Material assignment использует UUID, отсутствующая ссылка даёт определённый fallback.

Для custom numeric attrs descriptors строятся динамически. Для bool, enum, string и structural switches проверяем shared Playa API; при недостатке расширяем именно Playa Attrs/Animation/evaluator с совместимой serde-схемой и тестами.

Возможные изменения общей библиотеки: [attrs.rs](C:/projects/projects.rust.cg/cglibs/playa/crates/playa-engine/src/entities/attrs.rs) — discrete attribute/key API и тесты roundtrip; [anim.rs](C:/projects/projects.rust.cg/cglibs/playa/crates/playa-engine/src/entities/anim.rs) — hold sampling для bool/enum/string/structural values и компонентные тесты. Сначала проверяем существующий контракт; это пункты будущей реализации, код Playa при подготовке плана не меняем.

## Проверка и выход

Сначала compile spike на прямых dependencies. Проверить roundtrip неизвестных metadata и numeric attrs, component keys, f64 sampling и identity после rename/reorder. Перед кодовыми изменениями проверить GitNexus freshness и impact, после — reanalyze.

Только при подтверждённом конфликте dependency graph выделить существующие entities в lightweight crate внутри C:/projects/projects.rust.cg/cglibs/playa/crates, сохранить reexports из playa-engine. Не копировать реализацию в frac и не начинать этап с обязательного extraction.
