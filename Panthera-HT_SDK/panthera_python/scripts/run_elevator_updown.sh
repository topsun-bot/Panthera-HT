#!/usr/bin/env bash
# 电梯 UP/DOWN 点击（LocateAnything + Panthera）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
export LA_PYTHON="${LA_PYTHON:-/home/lenovo/miniconda3/envs/locateanything/bin/python}"
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source "$HOME/venvs/panthera/bin/activate"
cd "$ROOT/Panthera-HT_SDK/panthera_python/scripts"
exec python3 11_elevator_updown_click.py "$@"
