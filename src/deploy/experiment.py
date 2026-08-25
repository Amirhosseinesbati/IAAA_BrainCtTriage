"""Versioned experiment manifest shared by UI, Vast and MLflow."""

from __future__ import annotations

import base64
import re
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from src.config import config_section

TaskName = Literal["ich", "fracture", "mls", "triage_calibration"]


class HardwareSpec(BaseModel):
    gpu_profile: str
    disk_gb: int = Field(ge=20, le=500)
    max_price_per_hour: float = Field(gt=0, le=20)
    min_reliability: float = Field(ge=0.0, le=1.0)

    @field_validator("gpu_profile")
    @classmethod
    def known_gpu(cls, value: str) -> str:
        if value not in config_section("deployment", "gpu_profiles"):
            raise ValueError(f"Unknown GPU profile: {value}")
        return value


class RuntimeSpec(BaseModel):
    git_branch: str
    prepare_data: bool = True
    auto_destroy: bool = True

    @field_validator("git_branch")
    @classmethod
    def safe_branch(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9._/-]+", value) or ".." in value:
            raise ValueError("Unsafe git branch name")
        return value


class ExperimentManifest(BaseModel):
    schema_version: Literal[1] = 1
    task: TaskName
    strategy: str
    run_name: str = Field(min_length=3, max_length=120)
    notes: str = Field(default="", max_length=5000)
    tags: dict[str, str | int | float | bool] = Field(default_factory=dict)
    training_config: dict[str, Any] = Field(default_factory=dict)
    hardware: HardwareSpec
    runtime: RuntimeSpec

    @field_validator("run_name")
    @classmethod
    def safe_run_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not re.fullmatch(r"[A-Za-z0-9_. -]+", cleaned):
            raise ValueError("Run name may contain letters, numbers, spaces, dot, dash and underscore")
        return cleaned

    @model_validator(mode="after")
    def validate_task_strategy(self):
        if self.task == "fracture" and self.strategy != "yolo":
            raise ValueError("Fracture task currently requires strategy='yolo'")
        if self.task == "mls" and self.strategy != "mls_heatmap":
            raise ValueError("MLS task currently requires strategy='mls_heatmap'")
        return self

    @property
    def task_key(self) -> str:
        if self.task == "ich":
            return f"ich_{self.strategy}"
        if self.task == "fracture":
            return "fracture"
        if self.task == "mls":
            return "mls_heatmap"
        return "triage_calibration"

    @property
    def target_pipeline(self) -> str:
        return {"ich": "ich", "fracture": "yolo", "mls": "mls", "triage_calibration": "triage_calibration"}[self.task]

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False, allow_unicode=True)

    @classmethod
    def from_yaml(cls, payload: str) -> "ExperimentManifest":
        return cls.model_validate(yaml.safe_load(payload))

    def to_base64(self) -> str:
        return base64.b64encode(self.to_yaml().encode("utf-8")).decode("ascii")

    @classmethod
    def from_base64(cls, payload: str) -> "ExperimentManifest":
        return cls.from_yaml(base64.b64decode(payload.encode("ascii"), validate=True).decode("utf-8"))


def default_hardware() -> HardwareSpec:
    cfg = config_section("deployment")
    return HardwareSpec(
        gpu_profile=cfg["default_gpu_profile"], disk_gb=cfg["disk_gb"],
        max_price_per_hour=cfg["max_price_per_hour"], min_reliability=cfg["min_reliability"],
    )


def default_runtime() -> RuntimeSpec:
    cfg = config_section("deployment")
    return RuntimeSpec(
        git_branch=cfg["default_git_branch"], auto_destroy=cfg["auto_destroy"], prepare_data=True,
    )
