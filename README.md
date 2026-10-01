# Digital Fly Car

Симуляция RC-машинки с bicycle-кинематикой и зрительным контроллером, вдохновлённым реакциями мухи. Машинка ищет красную цель камерой, сближается с ней и объезжает препятствия. Контроллер не получает координаты цели из MuJoCo.

## Запуск двойным кликом

Открой Finder и дважды нажми [Digital Fly Car.command](</Users/leonard/Desktop/DigitalFlyCar/Digital Fly Car.command>). Запустится карта с прямым коридором. После достижения цели окно MuJoCo останется открытым до нажатия Esc.

При первом запуске macOS может заблокировать файл. Тогда нажми по нему правой кнопкой, выбери «Открыть» и подтверди запуск.

## Единая команда запуска

```bash
conda activate flybrain
cd /Users/leonard/Desktop/DigitalFlyCar
python run.py sim --scenario corridor
```

`run.py` сам переключается на `mjpython`, который нужен интерактивному viewer на macOS. Доступные карты:

- `corridor` — прямой коридор и цель в конце;
- `turn_corridor` — цель требует поворота;
- `slalom` — чередующиеся препятствия;
- `blocked_goal` — цель закрыта препятствием;
- `random` — процедурно созданная арена.

## Оценка без окна viewer

```bash
conda activate flybrain
cd /Users/leonard/Desktop/DigitalFlyCar
python run.py evaluate --suite scenarios
python run.py evaluate --scenario corridor
python run.py evaluate --suite random --episodes 20
```

Команда возвращает ненулевой код, если успешность ниже 75%.
`--episodes` задаёт число только для случайных арен; каждая фиксированная карта запускается один раз.

## Тесты

```bash
conda activate flybrain
cd /Users/leonard/Desktop/DigitalFlyCar
python run.py test
```
