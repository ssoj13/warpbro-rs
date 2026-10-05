# Этап 2: миграция и общий snapshot

Статус: реализация и итоговая проверка выполнены. [Общий план](plan.md). Зависит от [этапа 1](phase-01-model.md).

## Реализовано 2026-10-02

`WorldDocument::from_scene` в [world.rs](../../src/world.rs) создаёт World Settings, Fractal, Camera, DirectionalLight, Environment и Material; переносит значения, старые animation paths, компоненты, interpolation и дробные времена ключей. Legacy object rotation и её ключи меняют знак при переносе в Playa transform; прежний scalar scale разворачивается в XYZ. Исходная legacy animation сохраняется в расширении документа для диагностики и совместимости.

Preview, frozen export и thumbnails используют `WorldDocument::snapshot(frame: f64)`; адаптер `Scene::evaluate` обращается к тому же пути. Результат — временный `Scene` с вычисленными objects, lights и полными world matrices. Он не владеет редакторским документом: `document = None`, прежняя animation очищается. Замороженная копия WorldDocument переносится только в bookmark/export carrier; renderer получает вычисленные значения и создаёт device buffers.

Камера использует стабильный `camera_reference` из своего `gpu.formula`; добавление, скрытие и reorder фракталов не меняют масштаб орбиты. Активная камера продолжает использоваться при скрытии её слоя. Скрытое или неактивное HDR-окружение выключает environment, sky illumination и background.

Полная матрица строится через Playa `build_model_matrix` и `parent * local`, включая pivot и nonuniform scale. Fractal сохраняет собственные DE iterations; глобальные ray settings остаются у World Settings.

Ниже сохранён исходный перечень миграции и проверок. Итоговая проверка данных и GPU parity приведена в [этапе 5](phase-05-validation.md).

## Изменения

- [scene.rs](C:/projects/projects.rust.cg/cglibs/frac-rs/src/scene.rs): сохранить LegacyScene decoder; перевести persistence на versioned WorldDocument.
- Новый C:/projects/projects.rust.cg/cglibs/frac-rs/src/world_migration.rs: Scene → Fractal/Camera/DirectionalLight/Environment/default Material, root world settings; таблица старых animation paths → NodeId/attrs/components.
- [app.rs](C:/projects/projects.rust.cg/cglibs/frac-rs/src/app.rs), [io_service.rs](C:/projects/projects.rust.cg/cglibs/frac-rs/src/io_service.rs): загрузка/сохранение мира, bookmarks и presets, выбор активных объектов, единый command pathway.
- [render.rs](C:/projects/projects.rust.cg/cglibs/frac-rs/src/render.rs), [render_service.rs](C:/projects/projects.rust.cg/cglibs/frac-rs/src/render_service.rs), [export.rs](C:/projects/projects.rust.cg/cglibs/frac-rs/src/export.rs): единый evaluate_world(time f64) → immutable RenderSnapshot для viewport, export и headless.
- [environment.rs](C:/projects/projects.rust.cg/cglibs/frac-rs/src/environment.rs), [materials.rs](C:/projects/projects.rust.cg/cglibs/frac-rs/src/materials.rs): ресурсы адресовать через свойства объектов/UUID; runtime decode caches оставить производными.

## Миграция

Старый bookmark становится миром с теми же параметрами и изображением. Сохраняются pending tracks из текущей реализации, их значения, interpolation и f64 frames. Inline material допустим как migration default с последующим material UUID assignment.

Сохраняются неизвестные поля и нерешённые старые paths как migration extensions с диагностикой; нельзя молча выбросить данные или привязать ключ к другому объекту. schema_version определяет decoder, новые файлы не перезаписывают исходный формат без успешного decode.

Transform adapter документирует старую камеру/yaw и Playa 3D conventions. Selection, thumbnails и export используют документ, а не mutable legacy Scene.

## Проверка и выход

Fixture старого Scene без animation, bookmark с HDR и текущего Scene с keys → новый мир → save/load: параметры, metadata, значения ключей и изображение сохраняются. Один и тот же кадр совпадает в preview/export/headless.

Временный single-object renderer допустим только для parity во время миграции; завершённая задача требует [полного мира CUDA](phase-04-cuda-world.md). Удалить per-frame JSON diff; редактирование генерирует команды сразу.
