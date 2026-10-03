# Этап 5: проверка перехода парадигмы

Статус: выполнен; полный release harness, холодный export и проверка реального окна прошли. [Общий план](plan.md). Завершает этапы [1](phase-01-model.md), [2](phase-02-world-migration.md), [3](phase-03-object-ui.md), [4](phase-04-cuda-world.md).

## Текущее подтверждение 2026-10-02

- `cargo check --tests` прошёл.
- Release-сборка приложения через `python -B bootstrap.py b` с оптимизированным GPU-кодом прошла.
- Предыдущий release harness: 74 из 75 тестов прошли, включая CUDA multiobject, shadows, reflections и несколько lights. Один export test завершился по таймауту при холодном PTX JIT; повтор без кеша занял около 90 секунд, с кешем — около 1,1 секунды. Это не успешный полный прогон.
- Полный release harness прошёл: **80 passed, 0 failed**, время тестов 66,54 секунды; лог — [world-test-final.out](../../target/world-test-final.out).
- После расширения UI/schema полный release harness повторён: **84 passed, 0 failed**, 3,17 секунды; лог — [world-test-schema.out](../../target/world-test-schema.out).
- Принудительно холодный export после настройки CUDA Context прошёл за **48,70 секунды** при неизменном тестовом лимите 90 секунд; лог — [export-cold-optimized.out](../../target/export-cold-optimized.out).
- Реальное окно приложения проверено визуально: Outliner и AE-подобные object/component layers присутствуют; screenshot — [world-ui.png](../../target/verification/world-ui.png).
- Проверки roundtrip, компонентов, discrete values, parent transforms, camera reference и hidden environment находятся в [world_tests.rs](../../src/world_tests.rs), UI-проверки — рядом с [world_ui.rs](../../src/world_ui.rs), CUDA-проверки — в [render.rs](../../src/render.rs).
- Принятое поведение solo: сохраняется в host Attrs и одинаково фильтрует preview/export; selection не меняет состав геометрии.

Этап World завершён: release-сборка, полный прогон 84/84, холодный export и реальный UI подтверждены. Интеграция OIDN по образцу squarebob-rs с запуском через каждые N samples — следующий запрос; в выполненную World-проверку она пока не входит.

Ниже сохранены исходные сценарии и условия завершения.

## Значимые сценарии

1. Legacy Scene/bookmark/preset с HDR и animation → WorldDocument → save/load: ключи, произвольные metadata, ресурсы и прежний single-fractal результат сохраняются.
2. Два Fractal-узла, разные TRS и materials: оба реально участвуют в CUDA primary/shadow/secondary rays. Timeline reorder не меняет изображение.
3. Group с движущимся transform, дочерняя Camera и HDR intensity keys: viewport и экспорт на одинаковом f64 времени совпадают; interpolation/component channels вычисляет Playa.
4. Rename/reparent/reorder не меняют адреса keys; create/delete/duplicate и все key edits корректно undo/redo. Clone получает новый UUID и собственные ключи.
5. Custom numeric attr анимируется; unknown nested metadata и discrete values проходят roundtrip без потерь; structural switch не применяет несовместимые channels.
6. Metadata-only edit не сбрасывает accumulation. Видимость, material/transform/light changes и active camera/environment корректно обновляют render.
7. Outliner selection совпадает с Timeline/Inspector; lock не позволяет edit, solo одинаково фильтрует preview/export и сохраняется в документе. Сохранённая раскладка получает Outliner.

## Файлы и команды

Тесты разместить рядом с C:/projects/projects.rust.cg/cglibs/frac-rs/src/world.rs, world_migration.rs, world_commands.rs, timeline.rs, render.rs и export.rs; end-to-end fixtures — в C:/projects/projects.rust.cg/cglibs/frac-rs/tests/fixtures/world/.

Базовый отчёт предыдущей реализации: 51 тест и CUDA release-сборка. Это контрольная точка, а не доказательство проверки будущего мира.

Из C:/projects/projects.rust.cg/cglibs/frac-rs выполнить:

```powershell
$env:FRAC_ALLOW_PLAIN_CARGO='1'
cargo check --tests
python -B bootstrap.py b
python -B -c "import bootstrap, sys; sys.exit(bootstrap.run(['cargo', 'oxide', 'test', '--', '--release', '--', '--test-threads=1'])[0])"
```

Debug oxide backend ранее имел address-space compile failure; release harness остаётся рабочим путём CUDA-тестов. Не подменять GPU E2E только headless UI-тестами.

Проверить окно с изолированным FRAC_PROFILE_DIR: default и migrated layouts, Outliner, раскрытые component lanes, ключи и scrub. Сохранить screenshots и тестовые world files вне пользовательского профиля.

## Условия завершения

Все прежние релевантные тесты и новые E2E проходят. Данные мигрируют без потерь. Мир действительно содержит и рендерит несколько объектов. Viewport/export/headless используют общий Playa evaluator. GitNexus reanalyze отражает изменения; проверить detect_changes и git diff --check. README/CHANGELOG обновить после реализации фактическим поведением.
