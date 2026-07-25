"""
deployApp.py — Streamlit UI for cloud training orchestration on Vast.ai.

Dynamically renders config forms from Pydantic model JSON schemas with
support for nested / recursive models.  The dynamic rendering means adding
a new field to ``config_models.py`` automatically surfaces it in the UI.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import streamlit as st

# ── Path setup ──────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

st.set_page_config(page_title="Medical AI Orchestrator", page_icon="🏥", layout="centered")

st.title("🏥 Medical AI MLOps Center")
st.markdown("پایپ‌لاین مورد نظر خود را انتخاب کرده و با یک کلیک روی Vast.ai اجرا کنید.")

# ═════════════════════════════════════════════════════════════════════════
# 1. Pipeline Selection
# ═════════════════════════════════════════════════════════════════════════

st.subheader("تنظیمات آموزش")

PIPELINE_OPTIONS = {
    "nnunet": "🩸 nnU-Net (Hemorrhage Segmentation) [Legacy]",
    "yolo": "🦴 YOLO (Skull Fracture Detection)",
    "mls": "🧠 Midline Shift (Keypoints)",
    "ich": "🧪 ICH Strategy Selector ⭐ NEW",
    "all": "🚀 Run ALL Pipelines Sequentially",
}

pipeline_choice = st.selectbox(
    "کدام پایپ‌لاین اجرا شود؟",
    options=list(PIPELINE_OPTIONS.keys()),
    format_func=lambda x: PIPELINE_OPTIONS[x],
)

# ═════════════════════════════════════════════════════════════════════════
# 2. Recursive Config Form Renderer
# ═════════════════════════════════════════════════════════════════════════


def _resolve_ref(ref: str, schema: dict) -> dict:
    """Resolve a JSON Schema ``$ref`` like ``#/\\$defs/LossConfig``."""
    if not ref.startswith("#/"):
        return {}
    parts = ref[2:].split("/")
    obj = schema
    for part in parts:
        if part == "$defs":
            part = "$defs"
        obj = obj.get(part, {})
    return obj


def _render_config_fields(
    schema: dict,
    defaults: dict,
    *,
    prefix: str = "",
    nested_path: str = "",
) -> dict:
    """
    Recursively render Streamlit widgets from a JSON Schema subtree.

    Parameters
    ----------
    schema : dict
        The full JSON schema (used for ``$defs`` resolution).
    defaults : dict
        Default values for this level (from ``.model_dump()``).
    prefix : str
        Key prefix for nested fields (e.g. ``loss_config_``).
    nested_path : str
        Human-readable path for expander labels.

    Returns
    -------
    dict
        Flat dict of ``{field_name: value}`` collected from all widgets.
    """
    values = {}
    properties = schema.get("properties", {})

    for field_name, field_info in properties.items():
        qual_name = f"{prefix}{field_name}"
        raw_default = defaults.get(field_name)

        # ── Resolve $ref ────────────────────────────────────────────
        ref = field_info.get("$ref") or (
            field_info.get("anyOf", [{}])[0].get("$ref") if "anyOf" in field_info else None
        )
        resolved = _resolve_ref(ref, schema) if ref else {}

        field_type = field_info.get("type") or resolved.get("type", "string")
        nested_props = resolved.get("properties") or field_info.get("properties")

        # ── Determine label ─────────────────────────────────────────
        description = (
            field_info.get("description")
            or resolved.get("description")
            or field_name.replace("_", " ").title()
        )
        label = f"{description}"

        # ── Object / nested model → recursive expander ──────────────
        if nested_props and field_type in ("object", None):
            sub_defaults = raw_default if isinstance(raw_default, dict) else {}
            expander_label = f"⚙️  {label}"
            with st.expander(expander_label, expanded=False):
                sub_values = _render_config_fields(
                    resolved if ref else field_info,
                    sub_defaults,
                    prefix=f"{qual_name}_",
                    nested_path=qual_name,
                )
                values.update(sub_values)
                if sub_values:
                    st.caption("مقادیر انتخاب‌شده:")
                    st.code(json.dumps(sub_values, indent=2, default=str), language="json")
            values[qual_name] = sub_values
            continue

        # ── Enum / Literal → selectbox ──────────────────────────────
        enum_vals = field_info.get("enum") or resolved.get("enum")

        # Handle anyOf with enum + null (e.g. Optional[Literal[3, 5]])
        if enum_vals is None and "anyOf" in field_info:
            all_enum = []
            for opt in field_info["anyOf"]:
                opt_enum = opt.get("enum")
                if opt_enum:
                    all_enum.extend(opt_enum)
            if all_enum:
                enum_vals = all_enum

        # Handle anyOf with const values (OpenAPI 3.1 style)
        if enum_vals is None and "anyOf" in field_info:
            const_vals = []
            for opt in field_info["anyOf"]:
                if "const" in opt:
                    const_vals.append(opt["const"])
            if const_vals:
                enum_vals = const_vals

        if enum_vals:
            try:
                default_idx = enum_vals.index(raw_default) if raw_default in enum_vals else 0
            except (ValueError, TypeError):
                default_idx = 0
            # Show selectbox, but if value is None-replacement, show human-readable
            options_display = [str(v) if v is not None else "None" for v in enum_vals]
            display_idx = default_idx if raw_default in enum_vals else 0
            selected_display = st.selectbox(
                label, options=options_display, index=display_idx,
                key=f"field_{qual_name}",
            )
            # Map back to original value (reverse display)
            val_idx = options_display.index(selected_display)
            values[qual_name] = enum_vals[val_idx]
            continue

        # ── Boolean → checkbox ──────────────────────────────────────
        if field_type == "boolean":
            val = bool(raw_default) if raw_default is not None else False
            values[qual_name] = st.checkbox(label, value=val, key=f"field_{qual_name}")
            continue

        # ── Integer → number_input ──────────────────────────────────
        if field_type == "integer":
            minimum = field_info.get("minimum") or resolved.get("minimum", 0)
            maximum = field_info.get("maximum") or resolved.get("maximum", 10000)
            val = int(raw_default) if raw_default is not None else minimum
            values[qual_name] = st.number_input(
                label, min_value=minimum, max_value=maximum,
                value=val, step=1, key=f"field_{qual_name}",
            )
            continue

        # ── Number / Float → number_input ───────────────────────────
        if field_type == "number":
            minimum = field_info.get("minimum") or resolved.get("minimum", 0.0)
            maximum = field_info.get("maximum") or resolved.get("maximum", 1e6)
            val = float(raw_default) if raw_default is not None else minimum
            step = 1e-5 if abs(val) < 1 else 0.01
            values[qual_name] = st.number_input(
                label, min_value=minimum, max_value=maximum,
                value=val, step=step, format="%.5f" if step < 0.001 else "%.4f",
                key=f"field_{qual_name}",
            )
            continue

        # ── Array (e.g. Tuple for var_limit) → multi-number_input ───
        if field_type == "array":
            items = field_info.get("items", {})
            item_type = items.get("type", "number")
            min_items = field_info.get("minItems", field_info.get("prefixItems", None))
            prefix_items = field_info.get("prefixItems", [])

            arr_val = list(raw_default) if isinstance(raw_default, (list, tuple)) else []
            cols = st.columns(len(arr_val) if arr_val else 2)
            arr_inputs = []
            for i, col in enumerate(cols):
                pitem = prefix_items[i] if i < len(prefix_items) else items
                pmin = pitem.get("minimum", 0.0)
                pmax = pitem.get("maximum", 1.0)
                pstep = 0.001
                pval = arr_val[i] if i < len(arr_val) else pmin
                with col:
                    arr_inputs.append(
                        st.number_input(
                            f"{i}", min_value=pmin, max_value=pmax,
                            value=float(pval), step=pstep, format="%.4f",
                            key=f"field_{qual_name}_{i}", label_visibility="collapsed",
                        )
                    )
            values[qual_name] = arr_inputs
            st.caption(label)
            continue

        # ── String (fallback) → text_input ──────────────────────────
        val = str(raw_default) if raw_default is not None else ""
        values[qual_name] = st.text_input(label, value=val, key=f"field_{qual_name}")

    return values


def _reconstruct_nested(flat: dict, nested_path: str) -> dict:
    """
    Convert a flat dict (``loss_config_use_dice`` → True) back into a
    nested structure (``{"loss_config": {"use_dice": True}}``).

    Fields without an underscore separator are kept as top-level keys.
    """
    result: dict = {}
    for key, value in flat.items():
        if isinstance(value, dict) and all(
            k.startswith(f"{key}_") for k in flat if k != key
        ):
            # Already processed
            result[key] = value
            continue
        parts = key.split("_", 1)
        if len(parts) == 2 and parts[0] in (
            "loss_config",
            "augmentation_config",
            "top_bottom_flip",
            "left_right_flip",
            "rotate90",
            "shift_scale_rotate",
            "brightness_contrast",
            "gauss_noise",
            "scale_intensity",
            "adjust_contrast",
            "flip_axis_0",
            "flip_axis_1",
            "flip_axis_2",
            "shift_intensity",
            "gaussian_noise",
        ):
            segment, rest = parts
            if segment not in result:
                result[segment] = {}
            result[segment][rest] = value
        else:
            result[key] = value
    return result


# ═════════════════════════════════════════════════════════════════════════
# 3. ICH Strategy Selection & Dynamic Config Form
# ═════════════════════════════════════════════════════════════════════════

config_json_str = "{}"
strategy_choice = "nnunet"

if pipeline_choice == "ich":
    st.markdown("---")
    st.subheader("🧬 انتخاب استراتژی سگمنتیشن خونریزی (ICH)")

    @st.cache_data(ttl=3600)
    def load_strategies() -> list[dict]:
        from src.strategies import list_strategies
        return list_strategies()

    strategies = load_strategies()

    if not strategies:
        st.warning("⚠️ هیچ استراتژی ثبت نشده است. مطمئن شوید پکیج src/strategies سالم است.")
    else:
        strategy_choice = st.selectbox(
            "استراتژی",
            options=[s["name"] for s in strategies],
            format_func=lambda x: next(
                (s["display_name"] for s in strategies if s["name"] == x), x
            ),
            help="استراتژی سگمنتیشن خونریزی مغزی را انتخاب کنید",
        )

        # Show description
        selected = next(s for s in strategies if s["name"] == strategy_choice)
        with st.expander("📖 توضیحات استراتژی", expanded=False):
            st.info(selected["description"])

        # ── Dynamic Config Form ──────────────────────────────────────
        st.markdown("### ⚙️ پیکربندی پارامترها")

        schema = selected["config_schema"]
        defaults = selected["default_config"]

        # Render all fields recursively
        raw_values = _render_config_fields(schema, defaults)

        # Reconstruct nested JSON from flat keys
        def _build_nested(flat_dict: dict) -> dict:
            """Convert flat prefixed keys back to nested dict."""
            nested: dict = {}
            # Collect top-level keys vs nested groups
            nested_groups: dict[str, dict] = {}
            for key, value in flat_dict.items():
                # Check if this key belongs to a known nested group
                known_groups = [
                    "loss_config", "augmentation_config",
                    "top_bottom_flip", "left_right_flip", "rotate90",
                    "shift_scale_rotate", "brightness_contrast",
                    "gauss_noise", "scale_intensity", "adjust_contrast",
                    "flip_axis_0", "flip_axis_1", "flip_axis_2",
                    "shift_intensity", "gaussian_noise",
                ]
                found_group = None
                for group in known_groups:
                    if key.startswith(f"{group}_"):
                        rest = key[len(group) + 1:]
                        if group not in nested_groups:
                            nested_groups[group] = {}
                        nested_groups[group][rest] = value
                        found_group = group
                        break
                if found_group is None:
                    nested[key] = value

            # Merge nested groups back
            for group, group_dict in nested_groups.items():
                nested[group] = group_dict
            return nested

        config_values = _build_nested(raw_values)

        # ── Warning for risky augmentations ──────────────────────────
        if "augmentation_config" in config_values:
            aug = config_values["augmentation_config"]
            if isinstance(aug, dict):
                # Check SMP left_right_flip
                lr = aug.get("left_right_flip", {})
                if isinstance(lr, dict) and lr.get("enabled", True):
                    st.warning(
                        "⚠️ **Left-Right Flip فعال است.** اگر خروجی MLS (Midline Shift) "
                        "نیز به اشتراک گذاشته می‌شود، این Augmentation می‌تواند علامت "
                        "MLS را معکوس کند. در صورت نیاز به MLS، این گزینه را غیرفعال کنید.",
                        icon="⚠️",
                    )
                # Check MONAI flip_axis_0 (coronal = left-right)
                fa0 = aug.get("flip_axis_0", {})
                if isinstance(fa0, dict) and fa0.get("enabled", True):
                    st.warning(
                        "⚠️ **Flip along axis 0 (left-right) فعال است.** "
                        "می‌تواند علامت MLS را معکوس کند.",
                        icon="⚠️",
                    )

        # ── JSON Config Preview ──────────────────────────────────────
        config_json_str = json.dumps(config_values, default=str)

        with st.expander("📄 مشاهده کانفیگ کامل (JSON)", expanded=False):
            st.code(config_json_str, language="json")

# ═════════════════════════════════════════════════════════════════════════
# 4. GPU Selection
# ═════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.subheader("🖥️ انتخاب سخت‌افزار")

gpu_choice = st.selectbox(
    "کدام گرافیک (GPU) اجاره شود؟",
    options=["RTX_3060", "RTX_3090", "RTX_4090", "A5000"],
)

# ═════════════════════════════════════════════════════════════════════════
# 5. Launch Button
# ═════════════════════════════════════════════════════════════════════════

DEPLOY_SCRIPT_PATH = Path("src/deploy/deploy.py")

st.markdown("---")

if st.button("🔥 Launch on Vast.ai", type="primary", use_container_width=True):
    st.info(f"در حال ارتباط با Vast.ai برای اجاره {gpu_choice} و اجرای {pipeline_choice} ...")

    os.environ["TARGET_PIPELINE"] = pipeline_choice
    os.environ["GPU_TARGET"] = gpu_choice

    if pipeline_choice == "ich":
        os.environ["ICH_STRATEGY"] = strategy_choice
        os.environ["ICH_CONFIG"] = config_json_str

    try:
        with st.spinner("کمی صبر کنید... عملیات سرور در حال انجام است."):
            result = subprocess.run(
                ["uv", "run", "python", str(DEPLOY_SCRIPT_PATH)],
                capture_output=True, text=True, check=True,
            )
            st.success(f"✅ سرور با موفقیت ایجاد شد! پایپ‌لاین {pipeline_choice} در حال اجراست.")

            with st.expander("مشاهده لاگ‌های سیستم Vast.ai"):
                st.code(result.stdout)

            st.markdown("### 📊 مانیتورینگ زنده")
            st.markdown("- [داشبورد ZenML (لوکال)](http://127.0.0.1:8237)")
            st.markdown("- داشبورد MLflow در پروژه DagsHub شما")

            if pipeline_choice == "ich":
                st.info(
                    f"🧪 استراتژی: **{strategy_choice}** | "
                    f"کانفیگ به MLflow لاگ می‌شود."
                )

            st.warning("⚠️ نیازی نیست کاری انجام دهید. سرور پس از اتمام کار، به صورت خودکار خودش را نابود می‌کند.")

    except subprocess.CalledProcessError as e:
        st.error("❌ خطایی در اجاره سرور رخ داد!")
        with st.expander("جزئیات خطا"):
            st.code(e.stdout)
        if e.stderr:
            st.code(e.stderr)
