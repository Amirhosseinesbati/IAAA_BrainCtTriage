"""Streamlit control center for reproducible competition experiments."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import YOLO_DEFAULTS, config_section
from src.deploy.experiment import ExperimentManifest
from src.deploy.ui_helpers import build_manifest, parse_tags, save_manifest
from src.strategies import list_mls_strategies, list_strategies

load_dotenv(PROJECT_ROOT / ".env")
st.set_page_config(page_title="IAAA Experiment Control Center", page_icon="🧠", layout="wide")
st.title("🧠 IAAA Brain CT — Experiment Control Center")
st.caption("یک manifest واحد از UI تا Vast.ai، pipeline و MLflow؛ بدون اختلاف کانفیگ بین مراحل.")


def _resolve_schema(info: dict, root: dict) -> dict:
    ref = info.get("$ref")
    if not ref and "anyOf" in info:
        ref = next((item.get("$ref") for item in info["anyOf"] if item.get("$ref")), None)
    if not ref:
        return info
    value = root
    for part in ref.removeprefix("#/").split("/"):
        value = value[part]
    return {**value, **{key: value for key, value in info.items() if key != "$ref"}}


def _schema_type(info: dict) -> str:
    if info.get("type"):
        return info["type"]
    for choice in info.get("anyOf", []):
        if choice.get("type") not in (None, "null"):
            return choice["type"]
    return "string"


def render_schema(schema: dict, defaults: dict, *, root: dict | None = None, path: tuple[str, ...] = ()) -> dict:
    root = root or schema
    result: dict[str, Any] = {}
    for name, raw_info in schema.get("properties", {}).items():
        info = _resolve_schema(raw_info, root)
        default = defaults.get(name, info.get("default"))
        label = raw_info.get("description") or info.get("description") or name.replace("_", " ").title()
        key = "cfg_" + "_".join((*path, name))
        if info.get("properties"):
            with st.expander(label, expanded=False):
                result[name] = render_schema(info, default or {}, root=root, path=(*path, name))
            continue

        enum = info.get("enum")
        if not enum:
            enum = [choice["const"] for choice in raw_info.get("anyOf", []) if "const" in choice]
        if enum:
            index = enum.index(default) if default in enum else 0
            result[name] = st.selectbox(label, enum, index=index, key=key)
        elif _schema_type(raw_info) == "boolean":
            result[name] = st.checkbox(label, value=bool(default), key=key)
        elif _schema_type(raw_info) in ("integer", "number"):
            numeric_type = int if _schema_type(raw_info) == "integer" else float
            minimum = info.get("minimum", 0)
            maximum = info.get("maximum", 1_000_000)
            step = 1 if numeric_type is int else (1e-5 if abs(float(default or 0)) < 1 else 0.01)
            result[name] = st.number_input(
                label, min_value=numeric_type(minimum), max_value=numeric_type(maximum),
                value=numeric_type(default if default is not None else minimum), step=numeric_type(step), key=key,
            )
        elif _schema_type(raw_info) == "array":
            result[name] = st.text_input(label, value=json.dumps(default or []), key=key)
            try:
                result[name] = json.loads(result[name])
            except json.JSONDecodeError:
                st.error(f"{label}: آرایه JSON معتبر نیست")
        else:
            result[name] = st.text_input(label, value=str(default or ""), key=key)
    return result


def schema_from_defaults(defaults: dict) -> dict:
    types = {bool: "boolean", int: "integer", float: "number"}
    return {
        "type": "object",
        "properties": {
            key: {"type": types.get(type(value), "string"), "default": value, "description": key.replace("_", " ").title()}
            for key, value in defaults.items()
        },
    }


def run_deployer(manifest: ExperimentManifest, *, dry_run: bool) -> dict:
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "experiment.yaml"
        path.write_text(manifest.to_yaml(), encoding="utf-8")
        command = [sys.executable, "-m", "src.deploy.deploy", "--manifest", str(path)]
        if dry_run:
            command.append("--dry-run")
        completed = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if completed.returncode:
            try:
                message = json.loads(completed.stderr.strip().splitlines()[-1])["error"]
            except Exception:
                message = completed.stderr or completed.stdout
            raise RuntimeError(message)
        return json.loads(completed.stdout.strip().splitlines()[-1])


deployment = config_section("deployment")
identity_tab, model_tab, infra_tab, review_tab = st.tabs([
    "۱) هویت آزمایش", "۲) مدل و آموزش", "۳) Vast.ai", "۴) بازبینی و اجرا",
])

with identity_tab:
    left, right = st.columns(2)
    with left:
        task = st.selectbox("Task", ["ich", "fracture", "mls"], format_func={
            "ich": "ICH segmentation / volumes", "fracture": "Skull fracture", "mls": "Midline shift",
        }.get)
        default_name = f"{task}-{datetime.now().strftime('%Y%m%d-%H%M')}"
        run_name = st.text_input("نام Run", value=default_name, help="در experiment مخصوص همین task ثبت می‌شود.")
    with right:
        tags_text = st.text_area("Tags (هر خط key=value)", value="fold=0\nstage=baseline", height=105)
    notes = st.text_area("فرضیه و توضیح آزمایش", placeholder="چه چیزی تغییر کرده و معیار موفقیت چیست؟")

with model_tab:
    if task == "ich":
        choices = list_strategies()
        strategy = st.selectbox("استراتژی", [item["name"] for item in choices], format_func=lambda name: next(item["display_name"] for item in choices if item["name"] == name))
        selected = next(item for item in choices if item["name"] == strategy)
        st.info(selected["description"])
        training_config = render_schema(selected["config_schema"], selected["default_config"])
    elif task == "mls":
        choices = list_mls_strategies()
        strategy = st.selectbox("استراتژی", [item["name"] for item in choices], format_func=lambda name: next(item["display_name"] for item in choices if item["name"] == name))
        selected = next(item for item in choices if item["name"] == strategy)
        st.info(selected["description"])
        training_config = render_schema(selected["config_schema"], selected["default_config"])
    else:
        strategy = "yolo"
        training_config = render_schema(schema_from_defaults(YOLO_DEFAULTS), YOLO_DEFAULTS)
    st.markdown("##### Resolved training config")
    st.json(training_config)

with infra_tab:
    left, middle, right = st.columns(3)
    profiles = deployment["gpu_profiles"]
    with left:
        gpu_profile = st.selectbox("GPU profile", list(profiles), index=list(profiles).index(deployment["default_gpu_profile"]))
        disk_gb = st.number_input("Disk (GB)", min_value=20, max_value=500, value=int(deployment["disk_gb"]))
    with middle:
        max_price = st.number_input("حداکثر قیمت $/hour", min_value=0.05, max_value=20.0, value=float(deployment["max_price_per_hour"]), step=0.05)
        reliability = st.slider("حداقل reliability", 0.80, 1.0, float(deployment["min_reliability"]), 0.01)
    with right:
        git_branch = st.text_input("Git branch", value=deployment["default_git_branch"])
        prepare_data = st.checkbox("اجرای preprocessing", value=True)
        auto_destroy = st.checkbox("نابودی خودکار instance", value=bool(deployment["auto_destroy"]))
    estimated_hours = st.number_input("برآورد مدت اجرا (ساعت)", min_value=0.25, max_value=72.0, value=4.0, step=0.25)
    st.metric("حداکثر هزینه تخمینی", f"${max_price * estimated_hours:.2f}")

with review_tab:
    manifest = None
    try:
        manifest = build_manifest(
            task=task, strategy=strategy, run_name=run_name, notes=notes,
            tags=parse_tags(tags_text), training_config=training_config,
            gpu_profile=gpu_profile, disk_gb=int(disk_gb), max_price_per_hour=float(max_price),
            min_reliability=float(reliability), git_branch=git_branch,
            prepare_data=prepare_data, auto_destroy=auto_destroy,
        )
        st.success(f"Manifest معتبر است — MLflow experiment: {manifest.task_key}")
        st.code(manifest.to_yaml(), language="yaml")
        st.download_button("دانلود manifest", manifest.to_yaml(), file_name=f"{run_name}.yaml", mime="application/yaml")
    except Exception as exc:
        st.error(f"Manifest نامعتبر است: {exc}")

    required_env = ["VAST_API_KEY", "DAGSHUB_USER_TOKEN", "DAGSHUB_REPO_OWNER", "DAGSHUB_REPO_NAME", "DAGSHUB_TRACKING_URI", "GIT_REPO_URL"]
    missing_env = [name for name in required_env if not os.getenv(name)]
    if missing_env:
        st.warning("متغیرهای لازم در .env موجود نیستند: " + ", ".join(missing_env))

    col_save, col_dry, col_launch = st.columns(3)
    if col_save.button("💾 ذخیره محلی", use_container_width=True, disabled=manifest is None):
        path = save_manifest(manifest, PROJECT_ROOT / "config" / "experiments")
        st.success(f"ذخیره شد: {path.relative_to(PROJECT_ROOT)}")
    if col_dry.button("🔎 Dry-run و قیمت", use_container_width=True, disabled=manifest is None or bool(missing_env)):
        with st.spinner("در حال جست‌وجوی offer مناسب..."):
            try:
                st.json(run_deployer(manifest, dry_run=True))
            except Exception as exc:
                st.error(str(exc))
    if col_launch.button("🔥 Launch", type="primary", use_container_width=True, disabled=manifest is None or bool(missing_env)):
        with st.spinner("در حال ایجاد instance..."):
            try:
                result = run_deployer(manifest, dry_run=False)
                st.success(f"Instance ساخته شد: {result.get('instance_id', result['offer_id'])}")
                st.json(result)
                st.info("Run در experiment مخصوص task ثبت می‌شود؛ instance پس از پایان یا خطا طبق manifest نابود خواهد شد.")
            except Exception as exc:
                st.error(str(exc))

st.divider()
with st.expander("📚 تاریخچه manifestهای ذخیره‌شده"):
    directory = PROJECT_ROOT / "config" / "experiments"
    files = sorted(directory.glob("*.yaml"), key=lambda path: path.stat().st_mtime, reverse=True) if directory.exists() else []
    if files:
        st.dataframe([{"name": path.name, "modified": datetime.fromtimestamp(path.stat().st_mtime), "size": path.stat().st_size} for path in files], use_container_width=True)
    else:
        st.caption("هنوز manifest محلی ذخیره نشده است.")
