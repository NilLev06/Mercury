# Что где лежит в Digital Fly Car

Актуально на 10 октября 2026 года. Для повседневного использования достаточно запуска галереи; остальные файлы нужны для разработки, обучения и сохранения истории.

## Что открывать

| Файл | Назначение |
|---|---|
| [Fly Map Gallery.command](Fly%20Map%20Gallery.command) | Основной просмотр: локальный веб-пульт с 231 картой, камерой, траекторией и телеметрией. |
| [Digital Fly Car.command](Digital%20Fly%20Car.command) | Лаборатория с режимами запуска, проверки и обучения. |
| [Try Complex Fly.command](Try%20Complex%20Fly.command) | Прежний просмотр 32 сложных карт в отдельном окне MuJoCo. |
| [README.md](README.md) | Подробная инструкция, команды и история возможностей. Старые разделы описывают также прежние режимы. |
| [run.py](run.py) | Общая точка входа командной строки и маршрутизация режимов. |
| [requirements-imitation.txt](requirements-imitation.txt) | Дополнительные зависимости для imitation-обучения; это не полный список окружения проекта. |

## Где сама муха и управление

Рабочая цепочка галереи:

`камера и датчики MuJoCo → признаки наблюдения → рекуррентная сеть → моторный адаптер → газ и руль → MuJoCo`

| Файл | За что отвечает |
|---|---|
| [fly_brain_rc/imitation/connectome.py](fly_brain_rc/imitation/connectome.py) | Подготовка анатомического подграфа из данных соединений FlyWire. |
| [fly_brain_rc/imitation/model.py](fly_brain_rc/imitation/model.py) | Архитектура сети: входной адаптер, рекуррентные связи с анатомической маской, состояние и моторный выход. |
| [fly_brain_rc/imitation/observations.py](fly_brain_rc/imitation/observations.py) | Преобразование сенсорных данных в числовой вход сети. |
| [fly_brain_rc/perception/target.py](fly_brain_rc/perception/target.py) | Детекторы цели и препятствий по изображению. В симуляции используется обработка цветов, не YOLO. |
| [fly_brain_rc/imitation/runner.py](fly_brain_rc/imitation/runner.py) | Цикл проезда: получение наблюдения, вызов сети, передача команд машине и сбор результата. |
| [fly_brain_rc/imitation/arrival.py](fly_brain_rc/imitation/arrival.py) | Расстояние до цели и критерий устойчивой остановки. |
| [fly_brain_rc/imitation/teacher.py](fly_brain_rc/imitation/teacher.py) | Учитель для получения обучающих команд. В галерее учитель не управляет машиной. |
| [fly_brain_rc/imitation/training.py](fly_brain_rc/imitation/training.py) | Обновление параметров по обучающим траекториям. |

Фаворит содержит **255 нейронов и 4088 анатомических связей**. Это обучаемая модель на небольшом подграфе, а не полный цифровой мозг мухи. Во время просмотра веса фиксированы; меняется внутреннее состояние сети.

## Симулятор и карты

| Файл или папка | Назначение |
|---|---|
| [fly_brain_rc/sim/env.py](fly_brain_rc/sim/env.py) | Среда MuJoCo: камера, дальномеры, столкновения, движение и шаг симуляции. |
| [fly_brain_rc/sim/car_model.py](fly_brain_rc/sim/car_model.py) | Построение модели машины и сцены. |
| [fly_brain_rc/sim/kinematics.py](fly_brain_rc/sim/kinematics.py) | Кинематика движения машины. |
| [fly_brain_rc/config.py](fly_brain_rc/config.py) | Общие параметры машины, датчиков, арены и прежних режимов. |
| [fly_brain_rc/sim/scenarios.py](fly_brain_rc/sim/scenarios.py) | Базовые сценарии и их геометрия. |
| [fly_brain_rc/imitation/curriculum.py](fly_brain_rc/imitation/curriculum.py) | Учебные сценарии и старые проверочные карты. |
| [fly_brain_rc/imitation/navigation_cases.py](fly_brain_rc/imitation/navigation_cases.py) | Воспроизводимая генерация смешанных и сложных карт по seed. |
| `scenes/` | Сохранённые XML-сцены MuJoCo. Галерея строит карты из сценариев; эти XML не являются полным каталогом её карт. |

## Веб-пульт

| Файл | Назначение |
|---|---|
| [scripts/run_map_gallery.py](scripts/run_map_gallery.py) | Запуск сервера галереи на порту 8766. |
| [scripts/run_gallery_episode.py](scripts/run_gallery_episode.py) | Отдельный процесс одного настоящего проезда; загрузка фаворита, камера, телеметрия и проверка хеша модели. |
| [fly_brain_rc/dashboard/gallery.py](fly_brain_rc/dashboard/gallery.py) | Каталог 231 карты, исторические результаты, запуск и остановка процесса симуляции. |
| [fly_brain_rc/dashboard/lab.py](fly_brain_rc/dashboard/lab.py) | HTTP-сервер и API лаборатории и галереи. |
| [fly_brain_rc/dashboard/static/gallery.html](fly_brain_rc/dashboard/static/gallery.html) | Структура страницы галереи. |
| [fly_brain_rc/dashboard/static/gallery.css](fly_brain_rc/dashboard/static/gallery.css) | Внешний вид и адаптация под размер окна. |
| [fly_brain_rc/dashboard/static/gallery.js](fly_brain_rc/dashboard/static/gallery.js) | Выбор карт, фильтры, кнопки, получение состояния и отрисовка. Не рассчитывает физику и не выбирает моторные команды. |
| `dashboard/static/lab.html`, `dashboard/server.py`, `dashboard/static/index.html` | Страница лаборатории и прежний пульт отдельной симуляции. Полные пути находятся внутри `fly_brain_rc/`. |

## Модели и результаты — `artifacts/`

| Путь внутри `artifacts/imitation/` | Значение |
|---|---|
| `versions/main-23of32.pt` | Защищённая копия выбранного фаворита. Именно её читает галерея. |
| `connectome.pt`, `connectome.best.pt` | Рабочие копии модели для лаборатории; на момент уборки совпадают с фаворитом. При обучении рабочие копии могут меняться. |
| `versions/original-pre-compass.pt` | Сохранённая исходная модель до экспериментов с компасом. |
| `versions/development-broad-round1.pt` | Промежуточная версия широкого обучения; не основная. |
| `versions/manifest.json` | Реестр версий, контрольные суммы и история выбора основной модели. Некоторые записи относятся к ранее удалённым версиям и явно помечены. |
| `backups/` | Резервные копии перед важными изменениями. |
| `experiments/` | Прогоны обучения, промежуточные кандидаты, учебные данные и оценки. Не всё содержимое является моделью. |
| `retained-training-data/` | Учебные траектории, оставленные после прежней уборки весов. |
| `reports/`, `*.json`, `*.jsonl` | Отчёты, проверки и журналы. |

Кандидат `experiments/favourite-recovery-distillation-20261009/live-3.pt` сохранён для сравнения и **не заменяет фаворита**. История карт галереи читается из отчётов `independent-paired-navigation-20261010` и `frozen-navigation-final2-20261009`: удалять эти отчёты как мусор не следует.

В `artifacts/diagnostics/`, `artifacts/evolution/` и `artifacts/rl/` находятся результаты диагностик и прежних подходов. Они не являются весами текущего фаворита. `.pt` может содержать как веса, так и учебные данные — расширение само по себе не определяет назначение.

## Разработка и история

- [scripts/README.md](scripts/README.md) — список служебных сценариев. Большинство предназначено для экспериментов, а не обычного запуска.
- `tests/` — автоматические проверки кода и контрактов; это не учебные карты и не обучающие данные.
- [docs/README.md](docs/README.md) — указатель отчётов и планов.
- `docs/superpowers/specs/` и `docs/superpowers/plans/` — проекты решений и планы реализации; наличие плана не означает успешность эксперимента.
- `docs/archive/early-task-reports/` — ранние отчёты Task 9, перенесённые из корня. Их команды и метрики исторические.
- `figures/` — сохранённые иллюстрации и диагностические графики.
- `config/` — JSON-настройки прежних геномов и эволюционных экспериментов.
- `fly_brain_rc/neural/`, `runtime.py`, `rl*.py`, `bypass_learning.py`, `navigation_memory.py`, `recovery.py` — другие режимы и исследования, в том числе компас. Они сохранены для истории и отдельных запусков; галерея использует цепочку из `imitation/`.

## Скрытые папки

- `.local/connectome/` — исходные данные коннектома и аннотации (около 711 МиБ).
- `.local/heading-deps/` — зависимости прежних исследований компаса (около 100 МиБ).
- `.local/familiarity.json` — память знакомых видов для прежнего режима; не веса фаворита.
- `.local/gallery/` и `.local/lab/` — служебные файлы локальных запусков; не очищать при активном процессе.
- `.git/` — история Git. `.superpowers/` — сохранённые рабочие материалы, включая прежние черновики.

В Git включены выбранные модели из `versions/`, две рабочие копии и отчёты для галереи и независимого сравнения. Остальные учебные данные, резервные копии, промежуточные эксперименты и `.local/` остаются локальными. Для полного переноса исследовательского окружения их нужно копировать отдельно.

Уборка 10 октября: ранние отчёты перенесены в архив; удалены генерируемые Python-кэши в коде и `.DS_Store`. Исходники, модели, данные экспериментов и зависимости сохранены. Подробный журнал: [2026-10-10-folder-cleanup.json](docs/2026-10-10-folder-cleanup.json).
