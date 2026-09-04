#!/bin/bash
set -euo pipefail
cd /workspace/IAAA_BrainCtTriage_mls_da
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
set -a
source /root/.config/iaaa/secrets.env
set +a
exec .venv/bin/python -u scripts/run_mls_a8_pair.py
