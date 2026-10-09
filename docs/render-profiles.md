# Render/quality profiles: канонические узлы и viewport

Состояние на 2026-10-07: узловая модель, каталог в Settings и маршрутизация viewport
прошли финальные проверки: ordinary cuda-oxide suite — 271 passed / zero failed,
десять ignored и четыре ранее сертифицированных movie-fixtures filtered; production
cuda-oxide build и фактическая CUDA readiness прошли. Единственная независимая
проверка ветки закрыта с тремя исправленными P2 и регрессиями.
Точные результаты и границы проверки записаны в [HANDOFF](../HANDOFF.md).
OutputSettings (рецепт файла экспорта) добавлен 2026-10-08. WorldDirect, внешний
каталог подграфов и перенос узловых профилей в SquareBob остаются планом.

## Данные и связи

Параметры сохраняются только в графе `WorldDocument`. Профиль и шаблон используют
одинаковые виды узлов; `metadata.catalog_role = Profile / Template` задаёт роль
в каталоге. Имя, UUID и роль не выбирают алгоритм рендера.

| Вид узла | Собственные параметры | Связи |
| --- | --- | --- |
| `RenderSettings` | Метод `Fast / Full`, exposure, saturation, Reinhard, OIDN denoise | `/render/quality_id → QualitySettings` |
| `QualitySettings` | Samples, resolution scale, max steps, hit epsilon, step factor, max bounces, glass probes, adaptive sampling | Без обратной ссылки |
| `ViewportSettings` | `Auto / Locked`, target FPS, settle delay, batch budget, pause, freeze | Moving, Still, Manual → RenderSettings |
| `OutputSettings` | Формат (EXR / PNG / Video), encoder, ширина и высота, QP / CRF, denoise at completion, PNG encoding, HDR peak, видео из PNG, override OCIO display / view | Без ссылок |

Слоты графа `viewport_settings`, `output_render` и `output_settings` определяют текущий
viewport, независимо назначенный выходной RenderSettings и рецепт файла экспорта. Камера, геометрия, свет и
материал остаются в своих узлах. Преобразование OCIO и физический режим дисплея
остаются в существующих настройках Display / Color.

Численные render/quality-параметры остаются анимируемыми и оцениваются на текущем
кадре. Ссылка `/render/quality_id` и все поля `/viewport/*` статичны: загрузка
и команды отклоняют сохранённые animation/connections для этих полей. Назначения
Moving/Still/Manual и политика viewport не анимируются по времени.
Renderer получает временный `EffectiveRender` и снимок сцены; они производны
от графа и не являются второй сохраняемой конфигурацией. Старый Group
«World Settings» больше не дублирует сохраняемые render-параметры.

У свежего документа есть отдельные пары Moving, Still и Output. Moving начинает
с Fast, 64 samples, половинного разрешения и двух bounce. Still и Output получают
свои QualitySettings; Manual первоначально ссылается на Still. Начальная политика
viewport: Auto, 30 FPS, settle delay 180 мс, batch budget 8 мс. Это редактируемые
узлы, а не жёсткие ограничения preview.

UUID, вид узла, роль и значения проверяются. Live-потребитель принимает профиль;
шаблон сначала нужно применить как независимый экземпляр. Ссылка на отсутствующий
UUID или неподходящий вид узла, недопустимый диапазон и удаление используемых
настроек дают ошибку. Проверяется и постоянная часть документа, и сохранённые
значения ссылок. Типизированная схема связей не допускает обратной связи
QualitySettings → RenderSettings. Нет автоматической миграции, aliases или
восстановления отсутствующих полей старого формата.

## Создание и именованные кнопки

Откройте **Settings → Render & Viewport**. Панель показывает назначение Moving, Still, Manual
и Output, затем каталоги render- и quality-настроек. Параметры редактируются
существующим **Attribute Editor**: те же диапазоны, enums, UUID choices,
reset и анимация. Кнопка шестерёнки рядом с назначением открывает нужный узел.

- **New profile… / New template…** создают render/quality-пару с новым именем
  и UUID. Начальные параметры копируются из текущего Output, а не из отдельного
  глобального набора defaults.
- **New quality profile… / New quality template…** создают самостоятельный
  QualitySettings, начиная с текущего Output Quality.
- Для render-кнопок **Recall to** выбирает Moving, Still, Manual или Output.
  ЛКМ на профиле назначает его UUID этой цели. ЛКМ на шаблоне предлагает имя
  и создаёт независимую render/quality-пару.
- Для quality-кнопок **Recall quality to** выбирает RenderSettings.
  ЛКМ на quality-профиле меняет его quality-ссылку; шаблон создаёт отдельный
  QualitySettings и назначает новый UUID.
- ПКМ по именованной кнопке открывает **Edit in Attribute Editor**, **Rename…**,
  **Save as profile…**, **Save as template…**. Save as копирует текущие параметры
  в новую именованную запись. Меню шаблона также позволяет выбрать цель
  применения независимой копии.
- В Attribute Editor render-узла есть **Edit quality** и **Catalog actions**.
  Роль Profile / Template редактируется как metadata того же settings-узла.

Назначенный профиль — общая live-ссылка внутри документа. Изменение его
RenderSettings или QualitySettings видно всем потребителям этой ссылки.
**Independent copy…** копирует render и его quality, переназначая внутреннюю
ссылку на новый UUID. Применение шаблона делает такую же независимую копию:
последующее редактирование шаблона не меняет уже созданный экземпляр.

Создание, копирование и назначение одной render/quality-пары проходят через одну
команду WorldEditor и один шаг Undo. Quality-копия с назначением тоже является
одной операцией. Undo/Redo восстанавливает UUID, значения и назначения вместе.
Повседневные изменения Attribute Editor используют существующий undo-контракт.

Каталог находится в текущем WorldDocument и сохраняется вместе с ним. В этой
реализации нет скрытой синхронизации между файлами или отдельного глобального
словаря render/quality-параметров. **File → Templates** по-прежнему работает с
целыми сценами; это другой каталог.

## Viewport и переключение при движении

Toolbar viewport использует те же ссылки, что Settings → Render & Viewport. В нём доступны
**Auto / Locked**, меню профилей, pause и freeze. Через меню профилей можно
назначить Moving, Still и Manual, открыть профиль в Attribute Editor, создать
новый или независимую копию и применить шаблон.

В **Auto** playback или изменение вычисленной сцены выбирает Moving. После
последнего изменения выдерживается статичный settle delay из ViewportSettings,
затем выбирается Still. Orbit, fly и scrubbing меняют сцену
и используют этот же путь. В **Locked** всегда используется Manual, в том числе
при движении камеры.

Выбор делается в одном месте (`App::step_viewport` → `WorldDocument::viewport_render`)
и сохраняется как `ViewportRender` (ветвь + вычисленный профиль), отправленный worker.
Toolbar не пересчитывает выбор: кнопка меню профилей подписана активной ветвью
(Moving / Still / Manual), подсказка показывает профиль, метод, samples и масштаб
разрешения. Тест `camera_motion_routes_moving_until_settled` проверяет, что одна
смена позы камеры включает Moving, после settle delay — Still, а в Locked — Manual.

Runtime выбирает ветвь без перезаписи authored-ссылок и без новых Undo-записей.
Samples и resolution scale берутся из выбранного QualitySettings. Размер
viewport масштабируется до отправки запроса worker; worker не добавляет
скрытое половинное разрешение, лимит samples или bounce.

**Pause** запрещает новые sampling-батчи; новый запрос и обработка отображения
могут обновить видимый кадр без добавления samples. **Freeze** сохраняет показанное
изображение, удерживает прежний запрос сцены и приостанавливает его накопление
до разморозки. Поздние viewport-кадры не заменяют замороженное изображение;
пришедшие кадры cache preview возвращаются в пул. Уже выполнявшаяся GPU-работа
завершается штатно; это не возможность прервать CUDA kernel.

Бюджет viewport-батча ограничивается меньшим из batch budget и периода target FPS.
Батч использует измеренную стоимость sample и существующий предел 1–4 samples.
Target FPS задаёт частоту публикации и бюджет исполнения, а не гарантированный FPS.
Отдельная политика фоновой работы остаётся у render service. Полёт в реальном
времени для произвольной сцены этой реализацией не сертифицирован.

Progressive viewport использует один Target. Профили с совпадающими эффективными
параметрами накопления и размером изображения сохраняют тот же film; имя, UUID
и метка moving/still сами по себе его не сбрасывают. Изменение метода, геометрии,
tracing-параметров или extent проверяется renderer по вычисленному состоянию.
Target samples и параметры планирования не подменяют параметры estimator.
Запросы имеют поколения, а очередь и доставка viewport-кадров остаются ограниченными.

Cache preview фиксирует явный render-profile UUID в PreviewRequest: Still в Auto
или Manual в Locked, независимо от текущего Moving-запроса. Его samples и
resolution scale вычисляются из выбранного QualitySettings для зафиксированного
задания. Worker оценивает каждый кадр через snapshot_with_render_profile,
сохраняя выбранный профиль при воспроизведении кэша.

## Что означают Fast и Full

Оба метода используют существующий **World path tracer**.

**Fast** применяет приближённую модель `MaterialModel::Fast` к непрозрачным
материалам вычисленного снимка сцены, включая дочерние объекты. Передающие
материалы сохраняют свою модель: переключение не превращает стекло в
непрозрачный объект. Authored-материалы документа остаются неизменными.

**Full** сохраняет authored-модели материалов. Samples, глубина и точность
в обоих методах определяются связанным QualitySettings. Fast — явный выбор
приближения материалов, а не название отдельного world-raymarch renderer.

Существующий OFX Direct использует контекст одного объекта, корневой материал,
одно солнце и процедурное небо. Готового WorldDirect с мировыми transforms,
HDR-окружением и несколькими объектами здесь нет. Он остаётся отдельным
расширением общего world-контракта; переключатель Fast не объявляет его готовым.

## Output и Render / Encode

Output назначается отдельно от Moving, Still и Manual. Движение мыши,
Auto / Locked, pause и freeze viewport не меняют выбранный выходной профиль.

Панель **Render / Encode** начинается с двух привязок: **Output** (RenderSettings)
и **Output file** (OutputSettings) — те же строки, что в Settings → Render & Viewport,
с выбором, шестерёнкой Attribute Editor и меню New / Independent copy / шаблоны.
Форма ниже редактирует копию назначенного OutputSettings; каждое изменение
записывается обратно в узел командами `/output/*` (одна на изменённое поле,
`render_profiles::output_edits`), один шаг Undo на жест. Рецепт хранится только
в документе; preferences хранят лишь задание (`ExportJob`: имя файла, диапазон
кадров, FPS) — его не несёт ни один пресет.

Поля OutputSettings статичны, как `/viewport/*`: ключи и сохранённые
animation/connections отклоняются (`render_profiles::static_path`). Документ
проверяет тип и диапазон каждого поля; межполевые правила (чётный размер для
HEVC, лимит 64 Мпикс) проверяются при старте экспорта и показываются формой,
чтобы правка могла пройти через промежуточное состояние. Каталог Output
(**Settings → Output file profiles and templates**) работает как render/quality:
ЛКМ назначает профиль или создаёт независимую копию шаблона, ПКМ — Edit,
Rename, Save as profile / template.

При запуске экспорта WorldDocument фиксируется; рецепт читается из зафиксированного
документа (`ExportController::start`), а не из формы. Samples и resolution scale
вычисляются по Output Quality на первом кадре задания и фиксируются для задания;
заданный в панели размер масштабируется один раз. Остальные параметры сцены,
включая render/quality-атрибуты, оцениваются из зафиксированного документа
на времени каждого кадра. Изменение текущего документа после запуска
не меняет задание. Анимированные samples и resolution scale пока не задают
покадровый sample target или размер экспортируемого изображения.

Display / Color и существующий output-transform контракт описаны в
[README](../README.md). Назначение viewport-профиля не заменяет независимый
выбор преобразования выходного файла.

## Следующие этапы

| Работа | Граница текущей реализации |
| --- | --- |
| Проверенная реализация | Финальные ordinary tests, production cuda-oxide build/CUDA readiness и одна независимая проверка ветки прошли; receipts и ограничения — в HANDOFF |
| Внешний каталог | Экспорт/импорт канонических подграфов с явным составом и remap; сейчас каталог локален документу |
| OutputSettings | Сделано 2026-10-08: канонический узел рецепта, каталог, привязка в Render / Encode. Дальше — resize/crop (Playa Output Module) и очередь рендера с заданиями в документе |
| Дополнительная адаптация | Динамическое изменение качества в явных границах; сейчас профиль задаёт фиксированные samples и resolution scale |
| WorldDirect | Общая world-геометрия, object context, transforms, материалы и HDR-свет; затем измерения frame/input latency |
| Playa / очередь | Адаптация подтверждённого узлового контракта, явная фиксация и обновление задания |
| SquareBob | Перенос канонических render/quality-узлов и их UI; текущая проверка WarpBro не сертифицирует этот потребитель |

Проверки и benchmark должны отличать корректность данных от скорости.
Результат обычных тестов не заменяет измерение frame time или input latency;
снижение samples не доказывает 60 FPS. После прохождения необходимых тестов
дополнительные демонстрационные рендеры и видео не требуются.

История предыдущей смены и остальных репозиториев сохранена в
[HANDOFF](../HANDOFF.md); текущий render/export backlog — в [PLAN](../PLAN.md).
