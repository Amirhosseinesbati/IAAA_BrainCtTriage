#!/bin/bash
set -euo pipefail
cd /workspace/IAAA_BrainCtTriage_mls_da
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
set -a
. /root/.config/iaaa/secrets.env
set +a
exec .venv/bin/python scripts/train_mls_a9_frozen_refiner_cuda.py
