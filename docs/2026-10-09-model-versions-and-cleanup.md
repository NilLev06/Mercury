# Сохранённые модели после очистки — 2026-10-09

Оставлены три версии в `artifacts/imitation/versions`:

| Версия | Назначение |
|---|---|
| main-23of32.pt | Основная, 23/32 сложных независимого final2 без контактов |
| original-pre-compass.pt | Исходная до компаса, точка отката |
| development-broad-round1.pt | Перспективная ветка, 51/64 сложных на development |

Основная также доступна через `connectome.pt`, `connectome.best.pt` и прежний frozen-final2 путь. Исходные копии и резервные каталоги сохранены.

Удалено 260 устаревших файлов весов и оптимизаторов; освобождено 304.9 MiB по сумме размеров файлов. Все 111 сохранённых прежних PT-файлов проверены по SHA-256. Из 156 удалённых checkpoint-файлов учебные проезды выделены в `artifacts/imitation/retained-training-data/`. Отдельные replay, anchors, код и текстовые отчёты не удалялись.

Подробный журнал: `artifacts/imitation/versions/cleanup-20261009.json`. Исторические ссылки на удалённые веса в отчётах больше не открываются. Точные удалённые веса и состояния оптимизаторов для всех старых промежуточных раундов восстановить из одних учебных данных нельзя.

Ниже сохранён **исторический инвентарь до очистки**; перечисленные в нём версии 27/32, complex-30 и safe-balanced удалены. Размеры в старой таблице относятся к состоянию до очистки.

---

# Сохранённые модели и очистка — 2026-10-09

Основная версия: `artifacts/imitation/connectome.best.pt` и `connectome.pt`.
Неизменяемая копия: `artifacts/imitation/versions/main-23of32.pt`.
Откат: `artifacts/imitation/backups/before-main-23of32-20261009/`.

| Версия | Сложные карты без касаний | Простые без касаний | Набор |
|---|---:|---:|---|
| Основная main-23of32 | 23/32 | 13/16 | Независимый final2 |
| Исходная до компаса | 1/32 | 14/16 | Тот же final2 |
| Предыдущая previous-final-27of32 | 27/32 | 10/16 | Другой независимый final1 |
| development-complex-30 | 30/32 | 12/16 | Проверочный development |
| development-safe-balanced | 21/32 | 16/16 | Тот же development |
| development-broad-round1 | 51/64 | 29/32 | Новый development, 96 эпизодов |

Эти результаты нельзя ранжировать только по числителю: наборы различаются. Все шесть версий сохраняют 7/7 старых тестов без контактов. Основная выбрана пользователем; строгий прежний критерий приёмки она не прошла.

## Что занимает место

Ключевые веса малы. Основное место занимают replay-наборы и промежуточные раунды. Ниже полный инвентарь; эти эксперименты пока не удалены. Для продолжения обучения нужны исходные наборы, для езды нужны только веса. Удаление исходных каталогов сделает пути происхождения в старых отчётах недоступными.

| Каталог/файл | Размер MiB |
|---|---:|
| mixed-neural-rehearsal-fit-20261009 | 451.8 |
| precontact-dagger-20261009 | 421.1 |
| neural-label-fit-20261009 | 209.3 |
| neural-label-continued-20261009 | 206.9 |
| broad-neural-dagger-20261009 | 162.4 |
| broad-neural-replay-20261009 | 141.0 |
| broad-student-corrections-20261009 | 102.8 |
| sequential-transfer-20261008 | 101.4 |
| neural-teacher-rehearsal-20261009 | 84.8 |
| broad-corrections-20261009 | 69.0 |
| full-replay-20261009 | 58.3 |
| mixed-curriculum-20261009 | 46.3 |
| mixed-neural-rehearsal-20261009 | 45.0 |
| sector-assisted-20261009 | 42.7 |
| gentle-corrected-mastery-20261009 | 32.5 |
| sector-navigation-20261009 | 32.1 |
| gentle-mastery-20261009 | 28.4 |
| mastery-curriculum-20261009 | 23.1 |
| mirror-augmented-20261009 | 20.9 |
| teacher-consistent-20261009 | 18.2 |
| live-core-rehearsal-20261009 | 15.5 |
| sector-teacher-only-20261009 | 14.5 |
| navigation-20261008 | 14.5 |
| sector-frozen-core-20261009 | 14.4 |
| mirror-control-20261009 | 14.3 |
| live-core-teacher-warm-20261009 | 13.9 |
| live-core-20261009 | 12.8 |
| neural-ppo-protected-20261009 | 11.9 |
| neural-ppo-smoke-20261009 | 7.4 |
| opening-control-20261009 | 5.8 |
| opening-focus-20261009 | 5.8 |
| single-route-core-20261009 | 5.4 |
| throttle-calibration-20261009 | 4.1 |
| neural-scope-live-continued-20261009 | 4.0 |
| neural-scope-live-20261009 | 4.0 |
| physical-mirror-replay-20261009 | 3.3 |
| neural-scope-frozen-continued-20261009 | 2.3 |
| neural-scope-frozen-20261009 | 2.3 |
| cached-anchor-1-20261009 | 2.2 |
| cached-anchor-32-20261009 | 2.2 |
| rehearsal-between-rounds-20261009 | 2.1 |
| neural-label-final-blend-20261009 | 2.0 |
| rehearsal-static-blend-20261009 | 2.0 |
| gentle-teacher-20261009 | 2.0 |
| neural-scope-full-blend-20261009 | 1.4 |
| navigation-milestones-20261009 | 1.4 |
| frozen-navigation-final2-20261009 | 0.8 |
| frozen-navigation-final-20261009 | 0.8 |
| neural-blend-neighbors-20261009 | 0.7 |
| sector-initial-20261009.pt | 0.7 |
| motor-capacity-probe-matched-20261009 | 0.2 |
| motor-capacity-probe-20261009 | 0.2 |
| gentle-corrected-mastery-20261009.log | 0.1 |
| sector-navigation-20261009.log | 0.1 |
| sector-assisted-20261009.log | 0.1 |
| gentle-mastery-20261009.log | 0.1 |
| mastery-curriculum-20261009.log | 0.1 |
| live-core-rehearsal-20261009.log | 0.0 |
| live-core-20261009.log | 0.0 |
| sequential-transfer-20261008.log | 0.0 |
| sector-frozen-core-20261009.log | 0.0 |
| sector-teacher-only-20261009.log | 0.0 |
| full-replay-20261009.log | 0.0 |
| live-core-teacher-warm-20261009.log | 0.0 |
| teacher-consistent-20261009.log | 0.0 |
| neural-recovery-audit-balanced-20261009 | 0.0 |
| opening-control-20261009.log | 0.0 |
| opening-focus-20261009.log | 0.0 |
| gentle-teacher-20261009.log | 0.0 |
| neural-recovery-audit-20261009 | 0.0 |
| single-route-core-20261009.log | 0.0 |
| gentle-teacher-old-variations-20261009.log | 0.0 |

Всего экспериментов: 2.42 GiB.

Рекомендовано сохранить `versions`, `backups`, оба `frozen-navigation-final*` с отчётами и `navigation-20261008`. Старые промежуточные эксперименты можно сначала перенести в архив; replay-наборы удалять, только если они больше не нужны для продолжения обучения.

## Попробовать

Дважды открыть `Try Complex Fly.command`. Откроется MuJoCo с основной моделью на карте 1 из сохранённого final2. После закрытия окна в терминале можно выбрать карту 1–32, Enter — следующую, q — выход. Это самостоятельные проезды без обучения и без учителя. Не все карты модель проходит — набор содержит и её неудачи.

При желании из терминала:

```sh
/opt/miniconda3/envs/flybrain/bin/mjpython scripts/try_navigation.py --map 2
```
