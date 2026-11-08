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
if [[ "$(uname -s)" == Darwin ]]; then
  viewer_python="${python_bin:h}/mjpython"
  if [[ ! -x "$viewer_python" ]]; then
    print -u2 'Для окна MuJoCo нужен mjpython из выбранного Python-окружения.'
    exit 1
  fi
  python_bin="$viewer_python"
fi
exec "$python_bin" scripts/try_navigation.py --map 1
