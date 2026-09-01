#!/usr/bin/env bash
set -Eeuo pipefail

# Idempotent workspace bootstrap for a persistent MLS Vast instance.
# This script never starts training and never stops/destroys the instance.

PROJECT_DIR="${IAAA_PROJECT_DIR:-/workspace/IAAA_BrainCtTriage}"
SECRETS_FILE="${IAAA_SECRETS_FILE:-/root/.config/iaaa/secrets.env}"

if [[ ! -f "$SECRETS_FILE" ]]; then
    echo "Missing root-only secrets file: $SECRETS_FILE" >&2
    exit 1
fi
if [[ "$(stat -c '%a' "$SECRETS_FILE")" != "600" ]]; then
    echo "Secrets file permissions must be 600: $SECRETS_FILE" >&2
    exit 1
fi

set -a
# shellcheck disable=SC1090
source "$SECRETS_FILE"
set +a

# Some Vast CUDA images register the toolkit's forward-compatibility libcuda
# ahead of the driver library mounted from the host.  That can make nvidia-smi
# succeed while PyTorch fails with CUDA error 803.  Prefer the real host driver
# library for bootstrap and all child processes.
HOST_DRIVER_LIB_DIR="${IAAA_HOST_DRIVER_LIB_DIR:-/usr/lib/x86_64-linux-gnu}"
if [[ -e "$HOST_DRIVER_LIB_DIR/libcuda.so.1" ]]; then
    export LD_LIBRARY_PATH="$HOST_DRIVER_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

required=(
    DAGSHUB_REPO_OWNER DAGSHUB_USER_TOKEN DAGSHUB_TRACKING_URI
    DAGSHUB_REPO_ENDPOINT
)
for name in "${required[@]}"; do
    if [[ -z "${!name:-}" ]]; then
        echo "Required variable is missing from secrets file: $name" >&2
        exit 1
    fi
done

cd "$PROJECT_DIR"
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export IAAA_RUN_SOURCE=vast_persistent

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="/root/.local/bin:$PATH"
fi

# vastai is an orchestration CLI and is intentionally not installed into the
# remote training environment. The local controller owns the instance lifecycle.
uv sync --frozen --no-install-package vastai
uv run python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print({"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)})'

uv run dvc remote modify origin --local endpointurl "$DAGSHUB_REPO_ENDPOINT"
uv run dvc remote modify origin --local access_key_id "$DAGSHUB_USER_TOKEN"
uv run dvc remote modify origin --local secret_access_key "$DAGSHUB_USER_TOKEN"
chmod 600 .dvc/config.local
uv run dvc pull -r origin Data/raw.dvc

test -d Data/raw
actual_files="$(find Data/raw -type f | wc -l)"
actual_bytes="$(find Data/raw -type f -printf '%s\n' | awk '{sum += $1} END {printf "%.0f", sum}')"
expected_files="$(awk '/nfiles:/ {print $2}' Data/raw.dvc)"
expected_bytes="$(awk '/size:/ {print $2}' Data/raw.dvc)"
if [[ "$actual_files" != "$expected_files" || "$actual_bytes" != "$expected_bytes" ]]; then
    echo "DVC raw-data size/count mismatch: actual=$actual_files/$actual_bytes expected=$expected_files/$expected_bytes" >&2
    exit 1
fi

echo "Workspace bootstrap complete; no experiment was started."
echo "project=$PROJECT_DIR raw_files=$actual_files raw_bytes=$actual_bytes"
df -h "$PROJECT_DIR"
