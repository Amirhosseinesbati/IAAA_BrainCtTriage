#!/usr/bin/env bash
set -Eeuo pipefail
cd /workspace/IAAA_BrainCtTriage_mls_da
test "$(stat -c '%a' /root/.config/iaaa/secrets.env)" = 600
set -a
source /root/.config/iaaa/secrets.env
set +a
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec .venv/bin/python scripts/run_mls_a7_sequence.py
