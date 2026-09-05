#!/usr/bin/env bash
# Load only the dedicated remote-MLflow credentials needed by MLS runs.
set -Eeuo pipefail

secret_file="${IAAA_MLS_MLFLOW_ENV_FILE:-/workspace/.secrets/iaaa_mls_mlflow.env}"
if [[ ! -r "$secret_file" ]]; then
  echo "MLS remote-MLflow secret file is not readable: $secret_file" >&2
  exit 2
fi
for key in DAGSHUB_TRACKING_URI DAGSHUB_REPO_OWNER DAGSHUB_USER_TOKEN; do
  if ! grep -q "^${key}=." "$secret_file"; then
    echo "MLS remote-MLflow secret file is missing required key: $key" >&2
    exit 2
  fi
done

set -a
# shellcheck disable=SC1090
source "$secret_file"
set +a
exec "$@"
