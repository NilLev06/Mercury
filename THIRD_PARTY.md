# Источники и права

Автор реализации Mercury — Нильмаер Лев Владимирович.

## Анатомические данные

Источник графа: [FLYNN Connectome data and trained policy checkpoints](https://huggingface.co/datasets/benquan1/fly-gym-trained-policies), реконструкция FlyWire FAFB v783. В манифесте используемого графа указана лицензия **CC BY-NC 4.0**. Название лицензии и происхождение сохраняются внутри checkpoint. Исходный архив полного коннектома не включён в репозиторий.

Обученные веса Mercury, анатомический подграф и программный код имеют разное происхождение. Публикация исходников не отменяет условий использования сторонних данных. Лицензия на авторский код пока отдельно не назначена; наличие публичного доступа само по себе не является предоставлением открытой лицензии.

## Научные работы

- Wang B., Chen J. [FLYNN Robust Neural Network for Robot Navigation using Fly Brain Topology](https://arxiv.org/abs/2607.00025).
- Ross S., Gordon G., Bagnell D. [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://proceedings.mlr.press/v15/ross11a.html), DAgger.

## Библиотеки

Mercury использует PyTorch, MuJoCo, NumPy, OpenCV, NetworkX и py7zr. Эти зависимости устанавливаются отдельно и сохраняют собственные лицензии и авторство.

## Иллюстрации

`assets/demo.gif` и `assets/demo-preview.png` показывают воспроизведение сохранённых поездок Mercury и реактивного контроллера. Это кадры симуляции, не фотография физического устройства.
