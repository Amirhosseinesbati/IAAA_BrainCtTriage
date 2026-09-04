#!/bin/bash
set -euo pipefail
cd /workspace/IAAA_BrainCtTriage_mls_da
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
exec .venv/bin/python scripts/evaluate_mls_a9_frozen_refiner.py
