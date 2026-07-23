import sys
from pathlib import Path

# Ensure project root is on sys.path so `from src.xxx` imports work
# regardless of where Streamlit launches the script from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
import subprocess
import json
import os

st.set_page_config(page_title="Medical AI Orchestrator", page_icon="🏥", layout="centered")

st.title("🏥 Medical AI MLOps Center")
st.markdown("پایپ‌لاین مورد نظر خود را انتخاب کرده و با یک کلیک روی Vast.ai اجرا کنید.")

# ═════════════════════════════════════════════════════════════════════
# 1. Pipeline Selection
# ═════════════════════════════════════════════════════════════════════

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

# ═════════════════════════════════════════════════════════════════════
# 2. ICH Strategy Selection (only when pipeline == "ich")
# ═════════════════════════════════════════════════════════════════════

config_json_str = "{}"
strategy_choice = "nnunet"

if pipeline_choice == "ich":
    st.markdown("---")
    st.subheader("🧬 انتخاب استراتژی سگمنتیشن خونریزی (ICH)")

    # Import strategies (cached to avoid re-import on every rerun)
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

        # ── Dynamic Config Form ──────────────────────────────────
        st.markdown("### ⚙️ پیکربندی پارامترها")

        schema = selected["config_schema"]
        defaults = selected["default_config"]

        config_values = {}

        # Render each property as an appropriate Streamlit widget
        for field_name, field_info in schema.get("properties", {}).items():
            field_type = field_info.get("type", "string")
            description = field_info.get("description", field_name)
            default = defaults.get(field_name, field_info.get("default", None))
            is_required = field_name in schema.get("required", [])

            label = f"{description} {'*' if is_required else ''}"

            # ── Enum / Literal → selectbox ──
            if "enum" in field_info or "anyOf" in field_info:
                enum_vals = field_info.get("enum")
                if enum_vals is None and "anyOf" in field_info:
                    # OpenAPI 3.1 anyOf with const
                    enum_vals = []
                    for item in field_info["anyOf"]:
                        if "const" in item:
                            enum_vals.append(item["const"])
                        elif "type" in item and item["type"] == "string":
                            pass  # skip generic fallback
                    enum_vals = [v for v in enum_vals if v is not None]

                if enum_vals:
                    # Find default index
                    try:
                        default_idx = enum_vals.index(default) if default in enum_vals else 0
                    except (ValueError, TypeError):
                        default_idx = 0
                    config_values[field_name] = st.selectbox(
                        label, options=enum_vals, index=default_idx,
                    )
                    continue

            # ── Integer → number_input ──
            if field_type == "integer":
                minimum = field_info.get("minimum", 0)
                maximum = field_info.get("maximum", 10000)
                val = int(default) if default is not None else minimum
                config_values[field_name] = st.number_input(
                    label, min_value=minimum, max_value=maximum,
                    value=val, step=1,
                )
                continue

            # ── Number / Float → number_input ──
            if field_type == "number":
                minimum = field_info.get("minimum", 0.0)
                maximum = field_info.get("maximum", 1e6)
                val = float(default) if default is not None else minimum
                config_values[field_name] = st.number_input(
                    label, min_value=minimum, max_value=maximum,
                    value=val, step=1e-5 if val < 1 else 0.01,
                    format="%.5f" if val < 1e-3 else "%.4f",
                )
                continue

            # ── Boolean → checkbox ──
            if field_type == "boolean":
                val = bool(default) if default is not None else False
                config_values[field_name] = st.checkbox(label, value=val)
                continue

            # ── String (default) → text_input ──
            val = str(default) if default is not None else ""
            config_values[field_name] = st.text_input(label, value=val)

        # ── JSON Config Preview ──────────────────────────────────
        config_json_str = json.dumps(config_values, default=str)

        with st.expander("📄 مشاهده کانفیگ کامل (JSON)", expanded=False):
            st.code(config_json_str, language="json")

# ═════════════════════════════════════════════════════════════════════
# 3. GPU Selection
# ═════════════════════════════════════════════════════════════════════

st.markdown("---")
st.subheader("🖥️ انتخاب سخت‌افزار")

gpu_choice = st.selectbox(
    "کدام گرافیک (GPU) اجاره شود؟",
    options=["RTX_3060", "RTX_3090", "RTX_4090", "A5000"],
)

# ═════════════════════════════════════════════════════════════════════
# 4. Launch Button
# ═════════════════════════════════════════════════════════════════════

DEPLOY_SCRIPT_PATH = Path("src/deploy/deploy.py")

st.markdown("---")

if st.button("🔥 Launch on Vast.ai", type="primary", use_container_width=True):
    st.info(f"در حال ارتباط با Vast.ai برای اجاره {gpu_choice} و اجرای {pipeline_choice} ...")

    # Inject selections into environment variables for deploy.py
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
