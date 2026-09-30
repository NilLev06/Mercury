# Digital Fly Car

Симуляция RC-машинки с bicycle-кинематикой и зрительным контроллером, вдохновлённым реакциями мухи. Машинка ищет красную цель камерой, сближается с ней и объезжает препятствия. Контроллер не получает координаты цели из MuJoCo.

## Автономный запуск на macOS

```bash
conda activate flybrain
cd /Users/leonard/Desktop/DigitalFlyCar
PYTHONPATH=. mjpython -m scripts.fly_mode --seed 0 --max_episodes 1
```

MuJoCo требует `mjpython` для интерактивного viewer на macOS. Закрыть окно можно клавишей Esc.

## Оценка без интерактивного окна

```bash
conda activate flybrain
cd /Users/leonard/Desktop/DigitalFlyCar
PYTHONPATH=. python -m scripts.evaluate --episodes 20
```

Команда возвращает ненулевой код, если успешность ниже 75%.

## Тесты

```bash
conda activate flybrain
cd /Users/leonard/Desktop/DigitalFlyCar
PYTHONPATH=. python -m unittest discover -s tests -v
```
