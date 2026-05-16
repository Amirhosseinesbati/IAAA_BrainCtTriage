# app.py
import streamlit as st
import os
import sys
from pathlib import Path
import time

# --- Setup Paths ---
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from src.preprocessing.core.dicom_reader import BrainDicomReader
from src.inference.main_predict import load_all_models
from src.inference.triage_rules import apply_triage_rules

# --- Page Configuration ---
st.set_page_config(
    page_title="Brain CT Triage AI | هوش مصنوعی تریاژ مغز",
    page_icon="🧠",
    layout="wide"
)

# --- Translation Dictionary ---
lang_dict = {
    "en": {
        "title": "🧠 AI Brain CT Triage System",
        "subtitle": "Automated detection of Hemorrhage, Fracture, and Midline Shift",
        "sidebar_title": "⚙️ Settings & Inputs",
        "language": "Language / زبان",
        "folder_path": "Patient DICOM Directory Path:",
        "folder_help": "e.g., ./Data/raw/training/1952",
        "run_btn": "🚀 Run AI Analysis",
        "clear_btn": "🧹 Clear Screen & Reset",
        "loading_models": "Loading AI Models into memory... Please wait.",
        "models_loaded": "All models loaded successfully!",
        "analyzing": "Analyzing scan... This may take a few moments.",
        "error_path": "❌ The specified directory does not exist!",
        "results_title": "📊 Analysis Results",
        "triage_decision": "FINAL TRIAGE DECISION",
        "level_1": "CRITICAL (LEVEL 1)",
        "level_2": "ATTENTION NEEDED (LEVEL 2)",
        "normal": "NORMAL",
        "ich_title": "🩸 Hemorrhage (Volumes in mL)",
        "fracture_title": "💀 Skull Fracture",
        "fracture_detected": "⚠️ DETECTED",
        "fracture_not_detected": "✅ Not Detected",
        "mls_title": "📏 Midline Shift (mm)",
        "total_ich": "Total Volume:",
        "footer": "IAAA 2026 Challenge System",
        "waiting_msg": "👈 Please enter a DICOM directory path in the sidebar and run the analysis."
    },
    "fa": {
        "title": "🧠 سیستم هوش مصنوعی تریاژ سی‌تی اسکن مغز",
        "subtitle": "تشخیص خودکار خونریزی، شکستگی جمجمه و انحراف خط میانی",
        "sidebar_title": "⚙️ تنظیمات و ورودی‌ها",
        "language": "زبان / Language",
        "folder_path": "مسیر پوشه دایکام بیمار:",
        "folder_help": "مثال: ./Data/raw/training/1952",
        "run_btn": "🚀 شروع تحلیل هوش مصنوعی",
        "clear_btn": "🧹 پاک کردن صفحه و ریست",
        "loading_models": "در حال بارگذاری مدل‌های هوش مصنوعی... لطفا صبر کنید.",
        "models_loaded": "تمام مدل‌ها با موفقیت بارگذاری شدند!",
        "analyzing": "در حال تحلیل اسکن... این فرآیند ممکن است کمی طول بکشد.",
        "error_path": "❌ مسیر مشخص شده وجود ندارد!",
        "results_title": "📊 نتایج تحلیل",
        "triage_decision": "تصمیم نهایی تریاژ",
        "level_1": "بحرانی (سطح ۱)",
        "level_2": "نیازمند توجه (سطح ۲)",
        "normal": "نرمال (طبیعی)",
        "ich_title": "🩸 خونریزی (حجم به میلی‌لیتر)",
        "fracture_title": "💀 شکستگی جمجمه",
        "fracture_detected": "⚠️ تشخیص داده شد",
        "fracture_not_detected": "✅ مشاهده نشد",
        "mls_title": "📏 انحراف خط میانی (میلی‌متر)",
        "total_ich": "حجم کل:",
        "footer": "سیستم مسابقات IAAA 2026",
        "waiting_msg": "👈 لطفاً مسیر پوشه دایکام را در منوی کناری وارد کرده و تحلیل را شروع کنید."
    }
}

# --- State Management (جادوی حفظ اطلاعات) ---
if 'lang' not in st.session_state:
    st.session_state['lang'] = 'en'
if 'analysis_done' not in st.session_state:
    st.session_state['analysis_done'] = False # بررسی اینکه آیا تحلیلی انجام شده یا نه
if 'results' not in st.session_state:
    st.session_state['results'] = {} # ذخیره نتایج تحلیل
if 'study_dir_input' not in st.session_state:
    st.session_state['study_dir_input'] = "" # ذخیره مسیر ورودی

def toggle_language():
    st.session_state['lang'] = 'fa' if st.session_state['lang'] == 'en' else 'en'

def clear_screen():
    # این تابع وضعیت را به حالت اول برمی‌گرداند
    st.session_state['analysis_done'] = False
    st.session_state['results'] = {}
    st.session_state['study_dir_input'] = ""

# Helper function to get text
def t(key):
    return lang_dict[st.session_state['lang']][key]

# --- CSS Injection for RTL ---
if st.session_state['lang'] == 'fa':
    st.markdown("""
        <style>
        body, .stApp {
            direction: rtl;
            text-align: right;
            font-family: 'Vazirmatn', Tahoma, sans-serif;
        }
        .stMetric, .stMarkdown, .stText { direction: rtl; text-align: right; }
        div[data-testid="stSidebar"] { direction: rtl; text-align: right; }
        </style>
    """, unsafe_allow_html=True)

# --- Model Loading (Cached) ---
@st.cache_resource(show_spinner=False)
def init_models():
    return load_all_models(device='cuda')

def predict_for_ui(study_dir, models):
    reader = BrainDicomReader(study_dir).load_and_sort()
    ich_volumes = models["ich"].predict(reader)
    has_fracture = models["fracture"].predict(reader)
    mls_mm = models["mls"].predict(reader)
    final_label = apply_triage_rules(ich_volumes, has_fracture, mls_mm)
    return final_label, ich_volumes, has_fracture, mls_mm, reader.metadata.get('patient_id', 'Unknown')

# ==========================================
#                  UI LAYOUT
# ==========================================

# 1. Main Page Header
st.title(t("title"))
st.markdown(f"*{t('subtitle')}*")
st.divider()

# 2. Model Loading Process
with st.spinner(t("loading_models")):
    models = init_models()

# 3. Sidebar
with st.sidebar:
    st.title(t("sidebar_title"))
    
    # Language Toggle
    st.button("🌐 English / فارسی", on_click=toggle_language, use_container_width=True)
    st.divider()
    
    # Input Field (connected to session_state)
    st.session_state['study_dir_input'] = st.text_input(
        t("folder_path"), 
        value=st.session_state['study_dir_input'], 
        help=t("folder_help")
    )
    
    # Action Buttons
    analyze_btn = st.button(t("run_btn"), type="primary", use_container_width=True)
    clear_btn = st.button(t("clear_btn"), on_click=clear_screen, use_container_width=True)
    
    st.divider()
    st.caption(t("footer"))


# 4. Analysis Execution (ذخیره نتایج در Session State)
if analyze_btn:
    if not os.path.isdir(st.session_state['study_dir_input']):
        st.error(t("error_path"))
    else:
        with st.spinner(t("analyzing")):
            try:
                start_time = time.time()
                final_label, ich_vols, has_frac, mls_val, p_id = predict_for_ui(st.session_state['study_dir_input'], models)
                calc_time = time.time() - start_time
                
                # ذخیره نتایج در Session State تا با تغییر زبان پاک نشوند
                st.session_state['results'] = {
                    'final_label': final_label,
                    'ich_vols': ich_vols,
                    'has_frac': has_frac,
                    'mls_val': mls_val,
                    'p_id': p_id,
                    'calc_time': calc_time
                }
                st.session_state['analysis_done'] = True
                
            except Exception as e:
                st.error(f"An error occurred during analysis:\n{str(e)}")


# 5. Display Results (خواندن نتایج از Session State)
if st.session_state['analysis_done']:
    # خواندن مقادیر از حافظه
    res = st.session_state['results']
    
    st.subheader(f"{t('results_title')} (Patient: {res['p_id']})")
    
    # --- Triage Decision ---
    if res['final_label'] == "Level 1":
        st.error(f"### {t('triage_decision')}: {t('level_1')} 🚨")
    elif res['final_label'] == "Level 2":
        st.warning(f"### {t('triage_decision')}: {t('level_2')} ⚠️")
    else:
        st.success(f"### {t('triage_decision')}: {t('normal')} ✅")
    
    st.markdown("---")
    
    # --- Metrics ---
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"**{t('ich_title')}**")
        total_vol = 0
        for ich_type, vol in res['ich_vols'].items():
            st.metric(ich_type, f"{vol:.2f} mL")
            total_vol += vol
        st.divider()
        st.metric(t("total_ich"), f"{total_vol:.2f} mL")
    
    with col2:
        st.markdown(f"**{t('fracture_title')}**")
        if res['has_frac']:
            st.error(t("fracture_detected"))
        else:
            st.success(t("fracture_not_detected"))
            
    with col3:
        st.markdown(f"**{t('mls_title')}**")
        if res['mls_val'] > 5:
            st.error(f"{res['mls_val']:.2f} mm")
        elif res['mls_val'] > 0:
            st.warning(f"{res['mls_val']:.2f} mm")
        else:
            st.success(f"{res['mls_val']:.2f} mm")
    
    st.caption(f"⏱️ Analysis completed in {res['calc_time']:.2f} seconds.")

elif not st.session_state['study_dir_input'] or not st.session_state['analysis_done']:
    st.info(t("waiting_msg"))