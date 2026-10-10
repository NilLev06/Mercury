#!/bin/zsh
cd "$(dirname "$0")"
exec "${DIGITAL_FLY_PYTHON:-/opt/miniconda3/envs/flybrain/bin/python}" scripts/run_map_gallery.py
