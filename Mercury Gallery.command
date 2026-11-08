#!/bin/zsh
cd "$(dirname "$0")" || exit 1
python_bin="${MERCURY_PYTHON:-${DIGITAL_FLY_PYTHON:-}}"
if [[ -z "$python_bin" ]]; then
  if [[ -x .venv/bin/python ]]; then
    python_bin="$PWD/.venv/bin/python"
  else
    python_bin="$(command -v python3)"
  fi
fi
if [[ -z "$python_bin" || ! -x "$python_bin" ]]; then
  print -u2 'Python не найден. Создайте .venv по инструкции README.md.'
  exit 1
fi
exec "$python_bin" scripts/run_map_gallery.py
