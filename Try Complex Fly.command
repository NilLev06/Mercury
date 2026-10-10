#!/bin/zsh
cd "$(dirname "$0")"
exec /opt/miniconda3/envs/flybrain/bin/mjpython scripts/try_navigation.py --map 1
