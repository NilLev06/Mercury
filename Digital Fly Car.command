#!/bin/zsh
cd "$(dirname "$0")"

choice=$(osascript <<'APPLESCRIPT'
set options to {"Коридор — прямая цель|corridor", "Поворот|turn_corridor", "Слалом|slalom", "Цель за препятствием|blocked_goal", "Случайная арена|random"}
set chosen to choose from list options with title "Digital Fly Car" with prompt "Выбери карту для мухи:" default items {item 1 of options} OK button name "Запустить" cancel button name "Отмена"
if chosen is false then return ""
return item 1 of chosen
APPLESCRIPT
)

[[ -z "$choice" ]] && exit 0
scenario="${choice##*|}"
exec /opt/miniconda3/envs/flybrain/bin/python run.py sim --scenario "$scenario" --dashboard --memory .local/familiarity.json
