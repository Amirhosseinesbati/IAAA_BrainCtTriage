"""Create a reproducible Vast.ai training instance from an experiment manifest."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from urllib.parse import urlparse

from dotenv import load_dotenv

from src.config import PROJECT_ROOT, config_section
from src.deploy.experiment import ExperimentManifest


@dataclass(frozen=True)
class Credentials:
    vast_api_key: str
    dagshub_token: str
    dagshub_username: str
    dagshub_repo_name: str
    tracking_uri: str
    repo_endpoint: str
    git_repo_url: str


def _validate_dagshub_url(value: str, repo_name: str, suffix: str, env_name: str) -> None:
    parsed = urlparse(value)
    expected_tail = f"/{repo_name}{suffix}"
    if parsed.scheme != "https" or parsed.netloc != "dagshub.com" or not parsed.path.rstrip("/").endswith(expected_tail):
        raise RuntimeError(
            f"{env_name} must be an explicit DagsHub URL ending in {expected_tail}; got {value!r}"
        )


def load_credentials() -> Credentials:
    load_dotenv(PROJECT_ROOT / ".env")
    names = {
        "vast_api_key": "VAST_API_KEY",
        "dagshub_token": "DAGSHUB_USER_TOKEN",
        "dagshub_username": "DAGSHUB_REPO_OWNER",
        "dagshub_repo_name": "DAGSHUB_REPO_NAME",
        "tracking_uri": "DAGSHUB_TRACKING_URI",
        "repo_endpoint": "DAGSHUB_REPO_ENDPOINT",
        "git_repo_url": "GIT_REPO_URL",
    }
    values = {field: os.getenv(env_name, "").strip() for field, env_name in names.items()}
    missing = [names[field] for field, value in values.items() if not value]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")
    credentials = Credentials(**values)
    _validate_dagshub_url(
        credentials.tracking_uri, credentials.dagshub_repo_name,
        ".mlflow", "DAGSHUB_TRACKING_URI",
    )
    _validate_dagshub_url(
        credentials.repo_endpoint, credentials.dagshub_repo_name,
        ".s3", "DAGSHUB_REPO_ENDPOINT",
    )
    return credentials


def run_command(command: Sequence[str], *, redact: Sequence[str] = ()) -> str:
    display = " ".join("***" if part in redact else part for part in command)
    completed = subprocess.run(
        list(command), cwd=PROJECT_ROOT, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    if completed.returncode:
        raise RuntimeError(f"Command failed ({display}):\n{completed.stdout}\n{completed.stderr}")
    return completed.stdout.strip()


def build_offer_query(manifest: ExperimentManifest) -> str:
    profiles = config_section("deployment", "gpu_profiles")
    gpu = profiles[manifest.hardware.gpu_profile]["query_name"]
    min_disk = max(manifest.hardware.disk_gb, int(config_section("deployment", "min_disk_gb")))
    return (
        f"gpu_name={gpu} num_gpus=1 "
        f"reliability>={manifest.hardware.min_reliability} disk_space>={min_disk} "
        f"dph_total>={manifest.hardware.min_price_per_hour} "
        f"dph_total<={manifest.hardware.max_price_per_hour} "
        f"inet_down>={manifest.hardware.min_download_mbps} "
        f"inet_down<={manifest.hardware.max_download_mbps} "
        f"cpu_cores_effective>={manifest.hardware.min_cpu_cores} "
        f"cpu_cores_effective<={manifest.hardware.max_cpu_cores} rented=false"
    )


def select_offer(
    offers: list[dict],
    max_price: float,
    *,
    min_price: float = 0.0,
    top_k_enabled: bool = False,
    top_k: int = 10,
) -> dict:
    eligible = [
        offer for offer in offers
        if min_price <= float(offer.get("dph_total", float("inf"))) <= max_price
    ]
    if not eligible:
        raise RuntimeError(
            f"No eligible Vast.ai offer between ${min_price:.3f} and ${max_price:.3f}/hour"
        )
    cheapest = sorted(
        eligible,
        key=lambda offer: (float(offer["dph_total"]), -float(offer.get("reliability", 0))),
    )
    if not top_k_enabled:
        return cheapest[0]

    candidates = cheapest[:top_k]
    return max(
        candidates,
        key=lambda offer: (
            float(offer.get("score") or 0),
            float(offer.get("dlperf_usd") or 0),
            float(offer.get("reliability") or 0),
            -float(offer["dph_total"]),
        ),
    )


def load_manifest(path: Path | None) -> ExperimentManifest:
    if path:
        return ExperimentManifest.from_yaml(path.read_text(encoding="utf-8"))
    encoded = os.getenv("IAAA_EXPERIMENT_MANIFEST_B64", "")
    if not encoded:
        raise RuntimeError("Provide --manifest or IAAA_EXPERIMENT_MANIFEST_B64")
    return ExperimentManifest.from_base64(encoded)


def build_remote_environment(
    manifest: ExperimentManifest,
    credentials: Credentials,
    instance_id: str,
) -> str:
    values = {
        "VAST_API_KEY": credentials.vast_api_key,
        "INSTANCE_ID": instance_id,
        "DAGSHUB_TOKEN": credentials.dagshub_token,
        "DAGSHUB_USERNAME": credentials.dagshub_username,
        "DAGSHUB_REPO_NAME": credentials.dagshub_repo_name,
        "DAGSHUB_TRACKING_URI": credentials.tracking_uri,
        "DAGSHUB_REPO_ENDPOINT": credentials.repo_endpoint,
        "GIT_REPO_URL": credentials.git_repo_url,
        "GIT_BRANCH": manifest.runtime.git_branch,
        "IAAA_EXPERIMENT_MANIFEST_B64": manifest.to_base64(),
        "AUTO_DESTROY": str(manifest.runtime.auto_destroy).lower(),
    }
    return " ".join(f"-e {key}={value}" for key, value in values.items())


def deploy(manifest: ExperimentManifest, *, dry_run: bool = False) -> dict:
    credentials = load_credentials()
    run_command(["vastai", "set", "api-key", credentials.vast_api_key], redact=[credentials.vast_api_key])
    query = build_offer_query(manifest)
    offers = json.loads(run_command(["vastai", "search", "offers", query, "-o", "dph_total", "--raw"]))
    offer = select_offer(
        offers,
        manifest.hardware.max_price_per_hour,
        min_price=manifest.hardware.min_price_per_hour,
        top_k_enabled=manifest.hardware.top_k_enabled,
        top_k=manifest.hardware.top_k,
    )
    summary = {
        "dry_run": dry_run,
        "offer_id": str(offer["id"]),
        "price_per_hour": float(offer["dph_total"]),
        "download_mbps": float(offer.get("inet_down") or 0),
        "cpu_cores_effective": float(offer.get("cpu_cores_effective") or 0),
        "offer_score": float(offer.get("score") or 0),
        "dlperf_per_usd": float(offer.get("dlperf_usd") or 0),
        "gpu": manifest.hardware.gpu_profile,
        "task": manifest.task,
        "strategy": manifest.strategy,
        "run_name": manifest.run_name,
        "git_branch": manifest.runtime.git_branch,
        "selection_mode": "best_of_top_k_cheapest" if manifest.hardware.top_k_enabled else "cheapest",
        "top_k": manifest.hardware.top_k if manifest.hardware.top_k_enabled else None,
    }
    if dry_run:
        return summary

    deployment = config_section("deployment")
    environment = build_remote_environment(manifest, credentials, str(offer["id"]))
    command = [
        "vastai", "create", "instance", str(offer["id"]),
        "--image", deployment["docker_image"],
        "--disk", str(manifest.hardware.disk_gb),
        "--env", environment,
        "--onstart", "setup_vast.sh",
        "--ssh", "--direct", "--cancel-unavail",
        "--raw",
    ]
    response_text = run_command(
        command,
        redact=[credentials.vast_api_key, credentials.dagshub_token, manifest.to_base64(), environment],
    )
    response = json.loads(response_text) if response_text else {}
    if response.get("error"):
        raise RuntimeError(response.get("msg", "Vast.ai instance creation failed"))
    summary["instance_id"] = str(response.get("new_contract") or response.get("id") or offer["id"])
    summary["response"] = response
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(deploy(load_manifest(args.manifest), dry_run=args.dry_run), ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
