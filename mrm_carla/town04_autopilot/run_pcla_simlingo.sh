#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/tmo/carla_tools"
PCLA_ROOT="${ROOT}/PCLA"
PYTHON="/home/tmo/yes/envs/simlingo-carla015/bin/python"
CARLA_EGG="/home/tmo/carla/PythonAPI/carla/dist/carla-0.9.15-py3.8-linux-x86_64.egg"

export PYTHONNOUSERSITE=1
export PYTHONPATH="${CARLA_EGG}:${PCLA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${ROOT}/models/huggingface"
export HF_MODULES_CACHE="/tmp/pcla_simlingo_hf_modules"

exec "${PYTHON}" "${ROOT}/town04_autopilot/run_pcla_simlingo.py" "$@"
