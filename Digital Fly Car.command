#!/bin/zsh
cd "$(dirname "$0")"
exec /opt/miniconda3/envs/flybrain/bin/python run.py sim --scenario corridor
