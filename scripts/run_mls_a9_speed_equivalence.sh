#!/bin/bash
set -euo pipefail
cd /workspace/IAAA_BrainCtTriage_mls_da
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
set -a
. /root/.config/iaaa/secrets.env
set +a
exec .venv/bin/python scripts/benchmark_mls_a9_speed_equivalence_cuda.py
