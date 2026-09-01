#!/usr/bin/env bash
set -Eeuo pipefail

# Persistent-server wrapper for the explicit Python training gate.  It loads
# only the root-owned server secrets file and also applies the host-libcuda fix
# used by bootstrap_vast_workspace.sh before Python imports torch.

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

HOST_DRIVER_LIB_DIR="${IAAA_HOST_DRIVER_LIB_DIR:-/usr/lib/x86_64-linux-gnu}"
if [[ -e "$HOST_DRIVER_LIB_DIR/libcuda.so.1" ]]; then
    export LD_LIBRARY_PATH="$HOST_DRIVER_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

cd "$PROJECT_DIR"
exec uv run python scripts/run_vast_mls_experiment.py "$@"
