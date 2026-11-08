# Устройство Mercury

Основной вход — [README.md](README.md). Для просмотра используйте [Mercury Gallery.command](Mercury%20Gallery.command) или `python scripts/run_map_gallery.py`.

## Рабочая цепочка

| Файл | Назначение |
|---|---|
| [imitation/connectome.py](fly_brain_rc/imitation/connectome.py) | Вычислительный отбор подграфа FlyWire |
| [imitation/model.py](fly_brain_rc/imitation/model.py) | Рекуррентная сеть и моторный адаптер |
| [imitation/observations.py](fly_brain_rc/imitation/observations.py) | Обработка сенсорных наблюдений |
| [imitation/runner.py](fly_brain_rc/imitation/runner.py) | Цикл самостоятельной поездки |
| [imitation/teacher.py](fly_brain_rc/imitation/teacher.py) | Получение демонстраций для обучения |
| [imitation/training.py](fly_brain_rc/imitation/training.py) | Обучение модели |
| [sim/env.py](fly_brain_rc/sim/env.py) | Сцена, сенсоры и движение MuJoCo |
| [dashboard/gallery.py](fly_brain_rc/dashboard/gallery.py) | Каталог карт и запуск поездок |
| [dashboard/static/gallery.html](fly_brain_rc/dashboard/static/gallery.html) | Страница локальной демонстрации |

Все пути в таблице находятся внутри `fly_brain_rc/`.

## Модели

Галерея читает `artifacts/imitation/versions/main-23of32.pt`. Имя обозначает результат исторической проверки на 32 сложных сценах, а не размер актуального сравнительного набора. Контрольная сумма проверяется до и после поездки.

`connectome.pt` и `connectome.best.pt` — рабочие копии для лаборатории. История других сохранённых моделей находится в `artifacts/imitation/versions/manifest.json`.

## Результаты

Актуальное сравнение Mercury и реактивного контроллера: [96 сцен, 192 отчёта](results/2026-10-14-classical-comparison/README.md). Проверка без запуска симулятора: `python3 scripts/verify_saved_results.py`.

231 карта живой галереи относится к прежним наборам. Её исторические результаты читаются из `artifacts/imitation/experiments/independent-paired-navigation-20261010/` и `frozen-navigation-final2-20261009/`.

## Дополнительные материалы

- [scripts/README.md](scripts/README.md) — сценарии запуска и экспериментов.
- [docs/README.md](docs/README.md) — указатель исторических исследований.
- `tests/` — автоматические проверки.
- `.local/` — создаваемые локально данные галереи, лаборатории и исходный архив графа; не включаются в Git.
- `fly_brain_rc/neural/`, `rl*.py`, `evolution.py` — прежние исследовательские режимы.

Старые имена `.command` и Python-пакета сохранены для совместимости с командами и результатами экспериментов. Публичное название проекта — Mercury.
