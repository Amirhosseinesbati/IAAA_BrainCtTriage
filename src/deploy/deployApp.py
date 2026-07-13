import streamlit as st
import subprocess
import os
from pathlib import Path

st.set_page_config(page_title="Medical AI Orchestrator", page_icon="🏥", layout="centered")

st.title("🏥 Medical AI MLOps Center")
st.markdown("پایپ‌لاین مورد نظر خود را انتخاب کرده و با یک کلیک روی Vast.ai اجرا کنید.")

# 1. انتخاب پایپ‌لاین
st.subheader("تنظیمات آموزش")
pipeline_choice = st.selectbox(
    "کدام پایپ‌لاین اجرا شود؟",
    options=["nnunet", "yolo", "mls", "all"],
    format_func=lambda x: {
        "nnunet": "🩸 nnU-Net (Hemorrhage Segmentation)",
        "yolo": "🦴 YOLO (Skull Fracture Detection)",
        "mls": "🧠 Midline Shift (Keypoints)",
        "all": "🚀 Run ALL Pipelines Sequentially"
    }[x]
)

# 2. انتخاب پردازنده گرافیکی
gpu_choice = st.selectbox(
    "کدام گرافیک (GPU) اجاره شود؟",
    options=["RTX_3060", "RTX_3090", "RTX_4090", "A5000"]
)

# مسیر دقیق فایل دیپلوی نسبت به روت پروژه
DEPLOY_SCRIPT_PATH = Path("src/deploy/deploy.py")

# 3. دکمه اجرا
if st.button("🔥 Launch on Vast.ai", type="primary", use_container_width=True):
    st.info(f"در حال ارتباط با Vast.ai برای اجاره {gpu_choice} و اجرای {pipeline_choice} ...")
    
    # تزریق انتخاب‌ها به متغیرهای محیطی تا deploy.py آنها را بخواند
    os.environ["TARGET_PIPELINE"] = pipeline_choice
    os.environ["GPU_TARGET"] = gpu_choice
    
    try:
        with st.spinner('کمی صبر کنید... عملیات سرور در حال انجام است.'):
            # اجرای deploy.py
            # توجه: ما فرض میکنیم کاربر این صفحه را در مسیر اصلی (Root) اجرا کرده است
            result = subprocess.run(
                ["uv", "run", "python", str(DEPLOY_SCRIPT_PATH)], 
                capture_output=True, text=True, check=True
            )
            st.success(f"✅ سرور با موفقیت ایجاد شد! پایپ‌لاین {pipeline_choice} در حال اجراست.")
            
            with st.expander("مشاهده لاگ‌های سیستم Vast.ai"):
                st.code(result.stdout)
                
            st.markdown("### 📊 مانیتورینگ زنده")
            st.markdown("- [داشبورد ZenML (لوکال)](http://127.0.0.1:8237)")
            st.markdown("- داشبورد MLflow در پروژه DagsHub شما")
            st.warning("⚠️ نیازی نیست کاری انجام دهید. سرور پس از اتمام کار، به صورت خودکار خودش را نابود می‌کند.")
            
    except subprocess.CalledProcessError as e:
        st.error("❌ خطایی در اجاره سرور رخ داد!")
        with st.expander("جزئیات خطا"):
            st.code(e.stdout)
        if e.stderr:
            st.code(e.stderr)