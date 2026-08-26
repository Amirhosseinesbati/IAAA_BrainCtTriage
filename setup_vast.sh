#!/usr/bin/env bash
set -Eeuo pipefail

cleanup() {
    exit_code=$?
    trap - EXIT
    if [[ "${AUTO_DESTROY:-true}" == "true" ]]; then
        contract_id="${VAST_CONTAINERLABEL//[!0-9]/}"
        contract_id="${contract_id:-${INSTANCE_ID:-}}"
        if [[ -n "$contract_id" ]]; then
            echo "Destroying Vast.ai instance ${contract_id} after job exit=${exit_code}"
            uv run vastai destroy instance "$contract_id" -y --api-key "$VAST_API_KEY" || true
        fi
    fi
    exit "$exit_code"
}
trap cleanup EXIT

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export IAAA_RUN_SOURCE=vast

echo "Starting reproducible Vast.ai environment setup"
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
apt-get update
apt-get install -y git awscli libgl1-mesa-glx libglib2.0-0

git clone --branch "$GIT_BRANCH" --single-branch "$GIT_REPO_URL" /workspace/project
cd /workspace/project
uv sync --frozen

uv run dvc remote remove origin 2>/dev/null || true
uv run dvc remote add -d origin s3://dvc
# Repository ownership and authentication username are not necessarily equal.
# Use the explicit, locally verified DagsHub repository endpoint.
uv run dvc remote modify origin endpointurl "$DAGSHUB_REPO_ENDPOINT"
uv run dvc remote modify origin --local access_key_id "$DAGSHUB_TOKEN"
uv run dvc remote modify origin --local secret_access_key "$DAGSHUB_TOKEN"
uv run dvc pull -r origin
[[ -d Data/raw ]] || { echo "Data/raw is missing after DVC pull"; exit 1; }

export AWS_ACCESS_KEY_ID="$DAGSHUB_TOKEN"
export AWS_SECRET_ACCESS_KEY="$DAGSHUB_TOKEN"
export AWS_DEFAULT_REGION=us-east-1
export MLFLOW_S3_ENDPOINT_URL="$DAGSHUB_REPO_ENDPOINT"
export MLFLOW_TRACKING_USERNAME="$DAGSHUB_USERNAME"
export MLFLOW_TRACKING_PASSWORD="$DAGSHUB_TOKEN"
export MLFLOW_TRACKING_URI="$DAGSHUB_TRACKING_URI"

mkdir -p config/runtime
echo "$IAAA_EXPERIMENT_MANIFEST_B64" | base64 -d > config/runtime/active_run.yaml
echo "Launching validated manifest on branch $GIT_BRANCH"
uv run python -m src.pipelines.run_pipeline --manifest config/runtime/active_run.yaml
echo "Experiment completed successfully"
