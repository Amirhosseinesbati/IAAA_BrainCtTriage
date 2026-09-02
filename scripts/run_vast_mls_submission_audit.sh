#!/usr/bin/env bash
set -Eeuo pipefail

# Apply the same host-driver and secret-loading contract used by Vast training
# before importing PyTorch.  All arguments are forwarded to the CUDA audit.

PROJECT_DIR="${IAAA_PROJECT_DIR:-/workspace/IAAA_BrainCtTriage}"
SECRETS_FILE="${IAAA_SECRETS_FILE:-/root/.config/iaaa/secrets.env}"
HOST_DRIVER_LIB_DIR="${IAAA_HOST_DRIVER_LIB_DIR:-/usr/lib/x86_64-linux-gnu}"

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

if [[ -e "$HOST_DRIVER_LIB_DIR/libcuda.so.1" ]]; then
    export LD_LIBRARY_PATH="$HOST_DRIVER_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

cd "$PROJECT_DIR"
exec uv run python scripts/evaluate_mls_submission_ensemble_cuda.py "$@"
