# Этап 4: CUDA трассирует мир объектов

Статус: реализация и итоговая проверка выполнены. [Общий план](plan.md). Зависит от [snapshot](phase-02-world-migration.md).

## Реализовано 2026-10-02

[render.rs](../../src/render.rs) создаёт `WorldUpload` с версионированным world ABI, массивами объектов и directional lights; материал кодируется в записи соответствующего объекта. UUID ссылки разрешаются до загрузки. Отдельный GPU-массив материалов из исходного плана не введён.

[gpu.rs](../../src/gpu.rs) трассирует несколько Fractal-объектов с индивидуальным formula dispatch и материалом hit-объекта. Primary, shadow и reflection rays работают с одним миром; перестановка слоёв не определяет физическое перекрытие. Legacy single-object путь сохранён.

GPU получает полную обратную affine-матрицу, включая parent composition и shear. Нормали преобразуются через inverse transpose; консервативный distance scale учитывает nonuniform scale. Невырожденный negative scale допустим, singular/nonfinite/nonaffine matrices отклоняются с ошибкой. Object DE iterations остаются локальными, ray marching и bounce settings — глобальными.

Несколько directional lights и одно активное HDR-окружение участвуют в освещении. Float HDR/EXR decode и importance sampling сохранены. `scene_trace_data` содержит render-relevant параметры, matrices, палитры и lights; имена, UUID и metadata в подпись не входят. Preview/export используют общий вычисленный Scene; runtime/device buffers не сохраняются в WorldDocument.

Release-сборка приложения и полный CUDA harness прошли: 84/84 теста. Принудительно холодный export после настройки CUDA Context прошёл за 48,70 секунды с прежним лимитом 90 секунд. Итоги и проверка реального окна записаны в [этапе 5](phase-05-validation.md).

Ниже сохранён исходный перечень геометрических и световых проверок.

## Изменения

- [params.rs](C:/projects/projects.rust.cg/cglibs/frac-rs/src/params.rs): snapshot buffers объектов, материалов, directional lights и active camera/environment; версионировать ABI вместо единственного packed Scene.
- [gpu.rs](C:/projects/projects.rust.cg/cglibs/frac-rs/src/gpu.rs): hit/object/material identity, world distance/intersection, shadows, secondary rays и environment/light MIS.
- [render.rs](C:/projects/projects.rust.cg/cglibs/frac-rs/src/render.rs): upload immutable snapshot и cache signatures по render-relevant данным.
- [render_service.rs](C:/projects/projects.rust.cg/cglibs/frac-rs/src/render_service.rs): resource lifetime и invalidation между jobs/preview.
- [environment.rs](C:/projects/projects.rust.cg/cglibs/frac-rs/src/environment.rs): active map resource по Environment UUID, intensity/rotation через evaluated attrs.
- [export.rs](C:/projects/projects.rust.cg/cglibs/frac-rs/src/export.rs): те же buffers и sampling, без отдельного evaluator.

## Геометрия и освещение

Каждый луч проверяет все видимые Fractal-узлы; ближайший hit хранит object index и определяет material index. Formula dispatch выполняется для каждого объекта во время трассировки, вместо одной глобальной const F специализации. Single-object specialized fast path сохраняется лишь после измерения parity и производительности. Тени, отражения и прочие secondary rays учитывают все объекты. Порядок Timeline не влияет на intersection.

Использовать Playa полный 3D transform и parent composition; позиция world → local для DE, нормаль через inverse transpose. Для nonuniform scale нужен консервативный bound шага; singular/negative scale имеют проверенные правила и диагностику.

В snapshot отдельные material records и ссылки UUID → indices. Directional lights имеют собственные направление, интенсивность и цвет; выбор источника и MIS PDF учитывают число источников и вероятности. Active HDR environment участвует в background, NEE и BSDF miss с согласованным PDF.

Прежние HDR/EXR float decode и importance sampling сохраняются. Несколько Environment можно авторить, но активна одна map; HDR blending и area lights в следующей задаче.

## Проверка и выход

Два фрактала с разными TRS и материалами видны одновременно; один затеняет другой; вторичный луч попадает в другой объект. Проверить transformed normals, parent motion, nonuniform scale и reorder invariance.

Две directional lights + HDR дают ожидаемое изменение освещения без неверного MIS-веса. Metadata edit не сбрасывает accumulation; render-affecting attrs и active resource изменения сбрасывают.
