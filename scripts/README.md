# Служебные сценарии

Для обычного просмотра запускайте `Fly Map Gallery.command` в корне проекта. Эти файлы оставлены на прежних путях: их используют команды, тесты и сохранённые отчёты.

Сценарии обучения создают результаты экспериментов; перед ручным запуском проверьте `--help`, параметры модели и каталога вывода. Здесь перечислено назначение, а не рекомендация запускать все сценарии.

| Файл | Назначение |
|---|---|
| [ablate_frozen_recurrence.py](ablate_frozen_recurrence.py) | Проверяет влияние рекуррентных весов без изменения файла модели. |
| [audit_neural_recovery.py](audit_neural_recovery.py) | Диагностика обучающих восстановлений из неудачных состояний ученика. |
| [calibrate_neural_throttle.py](calibrate_neural_throttle.py) | Дообучает нейронный выход газа с сохранением рулевого выхода. |
| [collect_mirror_replay.py](collect_mirror_replay.py) | Собирает отражённые сенсорные траектории. |
| [compare_neural_scope.py](compare_neural_scope.py) | Сравнивает обучение с замороженным и изменяемым ядром сети. |
| [diagnose_training.py](diagnose_training.py) | Печатает диагностику датчиков и моторных команд учебной сцены. |
| [drive_manual.py](drive_manual.py) | Ручное управление машиной в MuJoCo. |
| [evaluate.py](evaluate.py) | Структурированная оценка прежнего нейронного режима. |
| [evaluate_frozen_navigation.py](evaluate_frozen_navigation.py) | Проверяет фиксированного кандидата и исходную модель на заранее заданных новых картах. |
| [evaluate_paired_navigation.py](evaluate_paired_navigation.py) | Сравнивает модели на одних и тех же независимых картах без обучения. |
| [export_scene.py](export_scene.py) | Экспортирует самостоятельную XML-сцену. |
| [fit_full_replay.py](fit_full_replay.py) | Диагностическое обучение на полном сохранённом наборе траекторий. |
| [fly_mode.py](fly_mode.py) | Запускает прежний нейронный runtime с окном MuJoCo. |
| [improve_navigation.py](improve_navigation.py) | Изолированный эксперимент с обучением остаточного моторного адаптера. |
| [memory_store.py](memory_store.py) | Загрузка и сохранение памяти знакомых видов для прежних режимов. |
| [open_scene.py](open_scene.py) | Открывает XML-сцену в MuJoCo. |
| [prepare_neural_rehearsal.py](prepare_neural_rehearsal.py) | Записывает успешные проезды нейронных учителей для повторного обучения. |
| [probe_motor_capacity.py](probe_motor_capacity.py) | Проверяет ёмкость моторного адаптера вне среды; не оценка реального проезда. |
| [refine_cached_motor.py](refine_cached_motor.py) | Дообучает моторный адаптер на фиксированных траекториях с опорой на прежние успешные выходы. |
| [refine_navigation.py](refine_navigation.py) | Обучает на сохранённом наборе траекторий. |
| [run_gallery_episode.py](run_gallery_episode.py) | Выполняет один проезд фаворита и передаёт кадры и телеметрию в галерею. |
| [run_map_gallery.py](run_map_gallery.py) | Запускает локальную веб-галерею. |
| [run_sim.py](run_sim.py) | Совместимая точка входа прежнего оценщика. |
| [select_neural_blend.py](select_neural_blend.py) | Выбирает статическую смесь параметров моделей по разработческим проверкам. |
| [select_rehearsal_blend.py](select_rehearsal_blend.py) | Сравнивает статические смеси моторных адаптеров с фиксированным ядром. |
| [train_broad_neural_dagger.py](train_broad_neural_dagger.py) | Обучение по широкому набору преимущественно нейронных демонстраций. |
| [train_live_core.py](train_live_core.py) | Эксперимент с обновлением ядра при сохранении топологии коннектома. |
| [train_mastery_curriculum.py](train_mastery_curriculum.py) | Расширяет учебную программу после самостоятельного освоения этапа. |
| [train_mixed_curriculum.py](train_mixed_curriculum.py) | Смешанная программа проездов учителя и ученика. |
| [train_neural_ppo.py](train_neural_ppo.py) | Эксперимент PPO для моторного адаптера. |
| [train_precontact_dagger.py](train_precontact_dagger.py) | DAgger с фокусом на предотвращение первого касания. |
| [train_recovery_distillation.py](train_recovery_distillation.py) | Обучение восстановлению с повторением поведения фаворита. |
| [train_sector_navigation.py](train_sector_navigation.py) | Эксперимент с дополнительными секторами зрительного входа. |
| [train_sequential.py](train_sequential.py) | Последовательное обучение объезду с опорой на исходные проезды. |
| [train_validated_recovery.py](train_validated_recovery.py) | Сравнивает восстановление с замороженным и обучаемым ядром по проверенным траекториям. |
| [try_navigation.py](try_navigation.py) | Интерактивно показывает сохранённый набор сложных карт в MuJoCo. |

[Общая карта проекта](../PROJECT_MAP.md).
