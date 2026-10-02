#!/bin/zsh
cd "$(dirname "$0")"

choice=$(osascript <<'APPLESCRIPT'
set options to {"Коридор — прямая цель|corridor", "Поворот|turn_corridor", "Слалом|slalom", "Цель за препятствием|blocked_goal", "Случайная арена|random", "Обучить RL — 120 эпизодов|train_rl", "RL-муха: коридор|rl_corridor"}
set chosen to choose from list options with title "Digital Fly Car" with prompt "Выбери карту для мухи:" default items {item 1 of options} OK button name "Запустить" cancel button name "Отмена"
if chosen is false then return ""
return item 1 of chosen
APPLESCRIPT
)

[[ -z "$choice" ]] && exit 0
scenario="${choice##*|}"
if [[ "$scenario" == "train_rl" ]]; then
  exec /opt/miniconda3/envs/flybrain/bin/python run.py train-rl --episodes 120 --out artifacts/rl/q-policy.json
fi
if [[ "$scenario" == "rl_corridor" ]]; then
  if [[ ! -f artifacts/rl/q-policy.json ]]; then
    osascript -e 'display alert "Сначала выбери «Обучить RL — 120 эпизодов»."'
    exit 1
  fi
  exec /opt/miniconda3/envs/flybrain/bin/python run.py sim --scenario corridor --dashboard --memory .local/familiarity.json --rl-policy artifacts/rl/q-policy.json
fi
exec /opt/miniconda3/envs/flybrain/bin/python run.py sim --scenario "$scenario" --dashboard --memory .local/familiarity.json
