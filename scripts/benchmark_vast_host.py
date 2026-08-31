"""Measure a rented Vast worker before accepting it for ICH experiments."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any


MIB = 1024 * 1024


def _memory_gib() -> float:
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / (1024 * 1024)
    return 0.0


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor()


def _disk_benchmark(workspace: Path, size_mib: int) -> dict[str, float]:
    workspace.mkdir(parents=True, exist_ok=True)
    block = bytes(8 * MIB)
    blocks = max(1, size_mib // 8)
    path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=".vast-disk-bench-", dir=workspace, delete=False
        ) as stream:
            path = Path(stream.name)
            started = time.perf_counter()
            for _ in range(blocks):
                stream.write(block)
            stream.flush()
            os.fsync(stream.fileno())
            write_seconds = time.perf_counter() - started

        started = time.perf_counter()
        read_bytes = 0
        with path.open("rb", buffering=0) as stream:
            for chunk in iter(lambda: stream.read(8 * MIB), b""):
                read_bytes += len(chunk)
        read_seconds = time.perf_counter() - started
        written_mib = blocks * 8
        return {
            "size_mib": float(written_mib),
            "write_mb_s": written_mib / max(write_seconds, 1e-9),
            "read_mb_s": (read_bytes / MIB) / max(read_seconds, 1e-9),
        }
    finally:
        if path is not None:
            path.unlink(missing_ok=True)


def _network_benchmark(url: str) -> dict[str, float | str]:
    started = time.perf_counter()
    total = 0
    request = urllib.request.Request(url, headers={"User-Agent": "iaaa-vast-benchmark/1"})
    with urllib.request.urlopen(request, timeout=90) as response:
        while chunk := response.read(8 * MIB):
            total += len(chunk)
    seconds = time.perf_counter() - started
    return {
        "url": url,
        "bytes": total,
        "seconds": seconds,
        "download_mbps": (total * 8 / 1_000_000) / max(seconds, 1e-9),
    }


def _gpu_benchmark(matrix_size: int, iterations: int) -> dict[str, Any]:
    import torch

    result: dict[str, Any] = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if not torch.cuda.is_available():
        return result

    properties = torch.cuda.get_device_properties(0)
    result.update(
        {
            "name": torch.cuda.get_device_name(0),
            "vram_gib": properties.total_memory / (1024**3),
            "compute_capability": f"{properties.major}.{properties.minor}",
        }
    )
    left = torch.randn((matrix_size, matrix_size), device="cuda", dtype=torch.float16)
    right = torch.randn((matrix_size, matrix_size), device="cuda", dtype=torch.float16)
    for _ in range(3):
        torch.mm(left, right)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        torch.mm(left, right)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    average = seconds / iterations
    result.update(
        {
            "matrix_size": matrix_size,
            "iterations": iterations,
            "average_seconds": average,
            "fp16_matmul_tflops": (2 * matrix_size**3) / average / 1e12,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
        }
    )
    return result


def _nvidia_smi() -> str:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,power.limit,temperature.gpu", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else completed.stderr.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--disk-mib", type=int, default=512)
    parser.add_argument(
        "--network-url",
        default="https://proof.ovh.net/files/100Mb.dat",
    )
    parser.add_argument("--matrix-size", type=int, default=8192)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    result: dict[str, Any] = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": {
            "cpu_model": _cpu_model(),
            "logical_cores": os.cpu_count() or 0,
            "ram_gib": _memory_gib(),
            "platform": platform.platform(),
        },
        "nvidia_smi": _nvidia_smi(),
    }
    errors: dict[str, str] = {}
    for name, operation in (
        ("disk", lambda: _disk_benchmark(args.workspace, args.disk_mib)),
        ("network", lambda: _network_benchmark(args.network_url)),
        ("gpu", lambda: _gpu_benchmark(args.matrix_size, args.iterations)),
    ):
        try:
            result[name] = operation()
        except Exception as exc:  # benchmark failures belong in the report
            errors[name] = f"{type(exc).__name__}: {exc}"
    result["errors"] = errors

    gpu = result.get("gpu", {})
    checks = {
        "rtx_3090": "RTX 3090" in str(gpu.get("name", "")),
        "cuda_available": bool(gpu.get("cuda_available")),
        "vram_at_least_23_gib": float(gpu.get("vram_gib", 0)) >= 23.0,
        "fp16_at_least_40_tflops": float(gpu.get("fp16_matmul_tflops", 0)) >= 40.0,
        "cpu_at_least_8_threads": int(result["host"]["logical_cores"]) >= 8,
        "ram_at_least_30_gib": float(result["host"]["ram_gib"]) >= 30.0,
        "disk_write_at_least_300_mb_s": float(result.get("disk", {}).get("write_mb_s", 0)) >= 300.0,
        "download_at_least_150_mbps": float(result.get("network", {}).get("download_mbps", 0)) >= 150.0,
    }
    result["checks"] = checks
    result["accepted"] = not errors and all(checks.values())

    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if args.strict and not result["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
