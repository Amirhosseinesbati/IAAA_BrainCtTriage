#!/usr/bin/env bash
set -euo pipefail

secret_file="${FRACTURE_MLFLOW_ENV_FILE:-/workspace/.secrets/iaaa_fracture_mlflow.env}"
if [[ ! -r "${secret_file}" ]]; then
  echo "Fracture MLflow environment file is not readable: ${secret_file}" >&2
  exit 2
fi
if [[ "$#" -eq 0 ]]; then
  echo "Usage: $0 COMMAND [ARG ...]" >&2
  exit 2
fi

set -a
# shellcheck disable=SC1090
source "${secret_file}"
set +a

exec "$@"
