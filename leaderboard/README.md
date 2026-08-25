# 🏆 Personal Leaderboard — IAAA 2026 Brain CT Triage

یک لیدربورد شخصی (شبیه‌سازی شده) برای ارزیابی مدل‌های آموزش‌دیده روی دیتای خودمون.

این ماژول مدل‌های قرارگرفته در `submission/models/` رو می‌گیره، روی کل مطالعات DICOM موجود در
`data/raw/training/` اینفرنس می‌کنه، و با ground truth فایل `Data/metadata/training_df.csv`
مقایسه می‌کنه. معیار اصلی **Quadratic Weighted Kappa (QWK)** هستش — دقیقاً همون معیار رسمی مسابقه.

**ویژگی کلیدی:** پشتیبانی کامل از **Strategy Pattern** — می‌تونین هر استراتژی ICH که آموزش دادین
(nnU-Net, SMP, MONAI, YOLO Seg) رو به راحتی تست کنین.

---

## 📁 ساختار

```
leaderboard/
├── __init__.py              # Package init
├── evaluate.py              # اسکریپت اصلی (CLI) — triage pipeline (QWK)
├── ground_truth.py          # استخراج label های study-level از CSV
├── scorer.py                # محاسبه QWK + متریک‌های جانبی
├── task_fracture.py         # 🦴 ارزیابی تسک شکستگی جمجمه
├── task_mls.py              # 📏 ارزیابی تسک انحراف خط میانی
├── task_hemorrhage.py       # 🩸 ارزیابی تسک خونریزی (تشخیص + حجم)
├── README.md                # همین فایل
├── results.csv              # نتایج per-study از evaluate.py
├── metrics.json             # متریک‌های QWK
├── fracture_results.csv     # نتایج per-study از task_fracture
├── fracture_metrics.json    # متریک‌های شکستگی
├── mls_results.csv          # نتایج per-study از task_mls
├── mls_metrics.json         # متریک‌های MLS
├── hemorrhage_results.csv   # نتایج per-study از task_hemorrhage
└── hemorrhage_metrics.json  # متریک‌های خونریزی
```

---

## 🚀 نحوه استفاده

### ۱. قرار دادن مدل‌ها

مدل‌های آموزش‌دیده رو توی `submission/models/` با این ساختار قرار بده:

```
submission/models/
├── nnunet/                          # ICH strategy: nnU-Net (default)
│   ├── checkpoint_best.pth
│   ├── dataset.json
│   ├── plans.json
│   └── dataset_fingerprint.json
├── smp/                             # ICH strategy: SMP (alternative)
│   └── best.ckpt
├── monai/                           # ICH strategy: MONAI (alternative)
│   └── UNETR_best.pth               # or SwinUNETR_best.pth / SegResNet_best.pth / DynUNet_best.pth
├── yolo_seg/                        # ICH strategy: YOLO Seg (alternative)
│   └── best.pt
├── yolo/                            # Fracture detection (shared across all strategies)
│   └── best.pt
├── mls/                             # Midline shift (shared across all strategies)
│   ├── slice_selector_best.ckpt
│   └── keypoint_best.ckpt           # legacy MLS (only used if mls_heatmap/ absent)
└── mls_heatmap/                     # [NEW] heatmap MLS strategy (auto-detected)
    └── mls_heatmap_best.pth         # ← مدل آموزش‌دیده با استراتژی mls_heatmap
```

> **نکته:** مدل‌های `yolo/` و `mls/` برای همه استراتژی‌های ICH یکسان هستند و فقط
> استراتژی ICH فرق می‌کند.
>
> **تشخیص خودکار MLS:** اگر `mls_heatmap/mls_heatmap_best.pth` موجود باشد، لیدربورد
> به‌طور خودکار از pipeline جدید heatmap (HRNet + DARK + Top-K) استفاده می‌کند؛
> backbone و تعداد کانال ورودی از `config` داخل checkpoint خوانده می‌شوند. در غیر این
> صورت fallback به مدل قدیمی `mls/keypoint_best.ckpt` می‌شود.

### ۲. اجرای لیدربورد با استراتژی‌های مختلف

```bash
# از روت پروژه اجرا کن:

# nnU-Net (پیش‌فرض)
python -m leaderboard.evaluate

# SMP
python -m leaderboard.evaluate --ich-strategy smp

# MONAI (SegResNet, UNETR, SwinUNETR, ...)
python -m leaderboard.evaluate --ich-strategy monai --device cuda

# YOLO Segmentation
python -m leaderboard.evaluate --ich-strategy yolo_seg

# روی CPU
python -m leaderboard.evaluate --ich-strategy smp --device cpu

# با مسیر دلخواه برای مدل‌ها
python -m leaderboard.evaluate --ich-strategy monai --models-dir experiments/monai_v2/models

# نمایش استراتژی‌های موجود
python -m leaderboard.evaluate --list-strategies

# ⭐ مقایسه همه استراتژی‌ها یکجا (--compare-all)
python -m leaderboard.evaluate --compare-all
python -m leaderboard.evaluate --compare-all --device cpu
```

### ۳. آپشن‌های قابل تنظیم

| آپشن | پیش‌فرض | توضیح |
|------|---------|-------|
| `--ich-strategy` | `nnunet` | استراتژی ICH: یکی از `nnunet`, `smp`, `monai`, `yolo_seg` |
| `--compare-all` | - | اجرای **تمامی** استراتژی‌های ICH و مقایسه کنار هم |
| `--list-strategies` | - | نمایش استراتژی‌های موجود و خروج |
| `--models-dir` | `submission/models` | مسیر پوشه مدل‌ها |
| `--data-dir` | `data/raw/training` | مسیر پوشه دیتای خام DICOM |
| `--csv-path` | `Data/metadata/training_df.csv` | مسیر فایل CSV metadata |
| `--device` | `cuda` | دستگاه اجرا: `cuda` یا `cpu` |
| `--output-csv` | `leaderboard/results.csv` | خروجی CSV نتایج به ازای هر مطالعه |
| `--output-json` | `leaderboard/metrics.json` | خروجی JSON متریک‌ها |
| `--no-export` | `False` | ذخیره نکردن فایل‌های خروجی |

---

---

## 🧪 تسک لیدربوردهای مجزا (Task-Specific Leaderboards)

علاوه بر ارزیابی کلی triage با QWK، می‌تونین هر کدوم از سه تسک رو **مستقل** ارزیابی کنین تا ببینین هر کدوم جداگانه چقدر دقیق عمل می‌کنن.

### قابلیت‌های مشترک

- **حالت CSV:** از روی فایل `leaderboard/results.csv` که توسط `evaluate.py` تولید شده ارزیابی می‌کنه (سریع، نیازی به لود مجدد مدل‌ها نیست)
- **حالت inference:** مدل‌ها رو لود می‌کنه و مستقیم روی DICOM اینفرنس اجرا می‌کنه
- **Strategy-aware:** برای تسک hemorrhage می‌تونه استراتژی‌های مختلف ICH رو مقایسه کنه
- **قابلیت گسترش:** افزودن استراتژی یا معیار جدید نیاز به تغییرات حداقلی داره

---

### 🦴 ۱. Skull Fracture Detection — `task_fracture.py`

ارزیابی تشخیص شکستگی جمجمه به صورت باینری.

| معیار | توضیح |
|-------|-------|
| **AUC-ROC** | معیار اصلی — توانایی تفکیک مثبت/منفی |
| **Default threshold (0.5)** | Accuracy, Precision, Recall, F1, Confusion Matrix |
| **Optimal threshold** | آستانه بهینه با معیار Youden's J |
| **Prediction distribution** | توزیع fracture_prob در کلاس‌های مثبت و منفی |

**نحوه استفاده:**
```bash
# از روی CSV موجود
python -m leaderboard.task_fracture

# اجرای مستقیم inference
python -m leaderboard.task_fracture --run-inference --ich-strategy nnunet

# با CSV دلخواه
python -m leaderboard.task_fracture --input-csv results_monai.csv
```

**خروجی نمونه:**
```
╔══════════════════════════════════════════════════════════╗
║  🦴  FRACTURE DETECTION — Task Leaderboard              ║
╠══════════════════════════════════════════════════════════╣
║  Studies evaluated: 288                                  ║
║  Prevalence: 55/288 (19.1%)                              ║
║  AUC-ROC:     0.8745                                     ║
╠══════════════════════════════════════════════════════════╣
║  At default threshold (0.5):                             ║
║    Accuracy : 0.8438  (84.4%)                            ║
║    Precision: 0.7200                                     ║
║    Recall   : 0.6545                                     ║
║    F1-score : 0.6857                                     ║
╠══════════════════════════════════════════════════════════╣
║  At optimal threshold (Youden J=0.65):                   ║
║    Threshold: 0.3200                                     ║
║    ...                                                   ║
╚══════════════════════════════════════════════════════════╝
```

**خروجی فایل‌ها:**
- `leaderboard/fracture_results.csv` — پیش‌بینی به ازای هر study
- `leaderboard/fracture_metrics.json` — متریک‌های جامع

---

### 📏 ۲. Midline Shift Estimation — `task_mls.py`

ارزیابی تخمین انحراف خط میانی (MLS) به صورت رگرسیون + طبقه‌بندی بالینی.

| معیار | توضیح |
|-------|-------|
| **MAE / RMSE / R²** | معیارهای اصلی رگرسیون |
| **Pearson / Spearman** | همبستگی بین GT و prediction |
| **Bland-Altman** | بایاس، SD اختلافات، 95% Limits of Agreement |
| **≥3mm (Urgent)** | Accuracy, Precision, Recall, F1 در آستانه اورژانس |
| **≥5mm (Critical)** | Accuracy, Precision, Recall, F1 در آستانه بحرانی |
| **Error distribution** | درصد خطاهای کمتر از 1mm, 2mm, 5mm |

**نحوه استفاده:**
```bash
# از روی CSV موجود
python -m leaderboard.task_mls

# اجرای مستقیم inference — فقط مدل‌های MLS (سبک، مناسب GTX 1660 Ti)
python -m leaderboard.task_mls --run-inference

# با CSV دلخواه
python -m leaderboard.task_mls --input-csv results_monai.csv

# اجرای inference با کل pipeline تریاژ (ICH + Fracture + MLS — سنگین)
python -m leaderboard.task_mls --run-inference --full-pipeline
```

**🤖 تشخیص خودکار مدل `mls_heatmap`:**

مدل آموزش‌دیده با استراتژی `mls_heatmap` را در مسیر زیر قرار دهید — به‌طور خودکار تشخیص داده شده و ارزیابی شروع می‌شود:

```
submission/models/
├── mls/
│   └── slice_selector_best.ckpt          # همیشه لازم است
└── mls_heatmap/
    └── mls_heatmap_best.pth              # ← مدل آموزش‌دیده (خروجی train_mls_heatmap)
```

- اگر `mls_heatmap_best.pth` موجود باشد → **حالت heatmap** (HRNet + DARK + Top-K) فعال می‌شود؛
  backbone و تعداد کانال ورودی به‌صورت خودکار از `config` داخل خود checkpoint خوانده می‌شوند
  (پشتیبانی از `hrnet_w32`/`hrnet_w18` و `input_channels=1`/`3`).
- در غیر این صورت → fallback به مدل قدیمی (`mls/keypoint_best.ckpt`).
- خروجی ارزیابی شامل `MAE / RMSE / R²`، Bland-Altman، و طبقه‌بندی در آستانه‌های ۳mm و ۵mm است.

**🎮 نکات GTX 1660 Ti (6 GB VRAM):**

- `--run-inference` به‌صورت پیش‌فرض **فقط مدل‌های MLS** را لود می‌کند (`load_mls_models` + `predict_mls_only`)
  — بدون نیاز به مدل‌های سنگین ICH (nnU-Net) و Fracture (YOLO) و بدون ریسک OOM.
- حافظه GPU بعد از هر study آزاد می‌شود (`torch.cuda.empty_cache()`).
- 1660 Ti از CUDA 11.x پشتیبانی می‌کند؛ در صورت نداشتن CUDA یا OOM از `--device cpu` استفاده کنید.

**خروجی نمونه:**
```
╔══════════════════════════════════════════════════════════╗
║  📏  MIDLINE SHIFT — Task Leaderboard                    ║
╠══════════════════════════════════════════════════════════╣
║  Studies evaluated: 288                                  ║
╠══════════════════════════════════════════════════════════╣
║  Regression Metrics:                                     ║
║    MAE  :   1.2345 mm                                    ║
║    RMSE :   2.3456 mm                                    ║
║    R²   :   0.6789                                       ║
║    Pearson r : 0.8234  (p=0.0000)                        ║
╠══════════════════════════════════════════════════════════╣
║  Bland-Altman Analysis:                                  ║
║    Bias (mean diff) :  0.1234 mm                         ║
║    SD of differences:  1.5678 mm                         ║
║    95% LOA         : [-2.95, 3.20] mm                    ║
╠══════════════════════════════════════════════════════════╣
║  Classification at Clinical Thresholds:                   ║
║    URGENT (≥ 3.0 mm) — F1: 0.7456                        ║
║    CRITICAL (≥ 5.0 mm) — F1: 0.8123                      ║
╚══════════════════════════════════════════════════════════╝
```

**خروجی فایل‌ها:**
- `leaderboard/mls_results.csv` — پیش‌بینی و خطا به ازای هر study
- `leaderboard/mls_metrics.json` — متریک‌های جامع

---

### 🩸 ۳. Hemorrhage Detection & Volume — `task_hemorrhage.py`

ارزیابی کامل تشخیص خونریزی (باینری) + تخمین حجم (رگرسیون).

| سطح ارزیابی | معیارها |
|-------------|---------|
| **AnyICH detection** | AUC-ROC, Accuracy, Precision, Recall, F1 |
| **Per-type detection** (IVH/IPH/SDH/EDH/SAH) | AUC-ROC, Accuracy, Precision, Recall, F1 |
| **Per-type volume** | MAE, RMSE, R², Pearson r |
| **Total volume** | MAE, RMSE, R², Pearson r |

**نحوه استفاده:**
```bash
# از روی CSV موجود
python -m leaderboard.task_hemorrhage

# یک استراتژی خاص (inference)
python -m leaderboard.task_hemorrhage --run-inference --ich-strategy monai

# ⭐ مقایسه همه استراتژی‌ها
python -m leaderboard.task_hemorrhage --run-inference --compare-all
```

**خروجی نمونه (تک استراتژی):**
```
╔════════════════════════════════════════════════════════════════╗
║  🩸  HEMORRHAGE — Task Leaderboard  [nnunet]                  ║
╠════════════════════════════════════════════════════════════════╣
║  Detection Metrics (threshold: volume >= 0.1 mL):              ║
║  Type         AUC-ROC   Acc      Prec     Rec      F1    Prev ║
║  ──────────────────────────────────────────────────────────── ║
║  IVH          0.9213   0.8924   0.7812   0.6543   0.7123  0.123║
║  IPH          0.8934   0.8456   0.7234   0.7123   0.7178  0.234║
║  ...                                                           ║
╠════════════════════════════════════════════════════════════════╣
║  Volume Regression Metrics (mL):                               ║
║  Type         MAE      RMSE     R²       GT-mean  Pred-mean   ║
║  ──────────────────────────────────────────────────────────── ║
║  IVH          0.8923   2.3456   0.5123    1.2345   1.3456     ║
║  ...                                                           ║
╚════════════════════════════════════════════════════════════════╝
```

**خروجی نمونه (مقایسه استراتژی‌ها):**
```
╔══════════════════════════════════════════════════════════════════════╗
║  📊  HEMORRHAGE STRATEGY COMPARISON                                  ║
╠══════════════════════════════════════════════════════════════════════╣
║  Strategy    AnyICH-AUC  AnyICH-F1 AnyICH-Acc Vol-MAE  Vol-RMSE Vol-R² Time║
║  ────────────────────────────────────────────────────────────────── ║
║ 👑nnunet      0.9234     0.8123    0.8543   1.2345   3.4567  0.5234  320s║
║   monai       0.9156     0.8034    0.8456   1.3456   3.5678  0.5012  450s║
╚══════════════════════════════════════════════════════════════════════╝
```

**خروجی فایل‌ها:**
- `leaderboard/hemorrhage_results.csv` — ۵ حجم + AnyICH به ازای هر study
- `leaderboard/hemorrhage_metrics.json` — متریک‌های جامع (تشخیص + حجم)

---

## 📊 خروجی نمونه

```
============================================================
  🏆  PERSONAL LEADERBOARD — IAAA 2026 Brain CT Triage
============================================================
  ICH strategy : smp
  Models dir   : submission\models
  Data dir     : data\raw\training
  CSV path     : Data\metadata\training_df.csv
  Device       : cuda
============================================================

╔══════════════════════════════════════════════════════════╗
║  🏆  PERSONAL LEADERBOARD — IAAA 2026 Brain CT Triage   ║
╠══════════════════════════════════════════════════════════╣
║  Studies evaluated: 320                                  ║
║  QWK Score:    0.8234                                    ║
║  Accuracy:     0.8938  (89.4%)                           ║
╠══════════════════════════════════════════════════════════╣
║  Interpretation: Almost perfect agreement                ║
╠══════════════════════════════════════════════════════════╣
║  Confusion Matrix (rows=GT, cols=Pred):                  ║
║           Pred N   Pred E   Pred C                       ║
║  GT Normal      110       8       2                     ║
║  GT Emerg        12      58       5                     ║
║  GT Critical      3       4     118                     ║
╠══════════════════════════════════════════════════════════╣
║  Per-Class Metrics:                                      ║
║           Precision  Recall    F1       Support          ║
║  Normal     0.8800    0.9167    0.8980      120          ║
║  Emergency  0.8286    0.7733    0.8000       75          ║
║  Critical   0.9440    0.9440    0.9440      125          ║
╠══════════════════════════════════════════════════════════╣
║  Class Distribution (GT → Pred):                         ║
║    Normal    :  120 →  125                               ║
║    Emergency :   75 →   70                               ║
║    Critical  :  125 →  125                               ║
╚══════════════════════════════════════════════════════════╝
```

### تفسیر QWK

| بازه QWK | تفسیر |
|----------|-------|
| 0.81 – 1.00 | Almost perfect agreement ✅ |
| 0.61 – 0.80 | Substantial agreement 🟡 |
| 0.41 – 0.60 | Moderate agreement 🟠 |
| 0.21 – 0.40 | Fair agreement 🔴 |
| 0.00 – 0.20 | Slight agreement 🔴 |
| < 0.00 | Worse than chance ❌ |

---

## 🧪 مقایسه استراتژی‌ها

### روش ۱: دستی (خروجی CSV جداگانه)

می‌تونین چندین استراتژی مختلف رو آموزش بدین و نتایج رو مقایسه کنین:

```bash
# nnU-Net
python -m leaderboard.evaluate --ich-strategy nnunet --output-csv results_nnunet.csv

# SMP (U-Net با EfficientNet)
python -m leaderboard.evaluate --ich-strategy smp   --output-csv results_smp.csv

# MONAI (SwinUNETR)
python -m leaderboard.evaluate --ich-strategy monai --output-csv results_monai.csv

# YOLO Seg
python -m leaderboard.evaluate --ich-strategy yolo_seg --output-csv results_yolo_seg.csv
```

### روش ۲: خودکار با `--compare-all` ⭐

تنها با یک دستور، همه استراتژی‌هایی که مدل دارن رو اجرا می‌کنه و جدول مقایسه چاپ می‌کنه:

```bash
python -m leaderboard.evaluate --compare-all
```

خروجی نمونه:
```
╔══════════════════════════════════════════════════════════════════════╗
║  📊  STRATEGY COMPARISON — IAAA 2026 Brain CT Triage                ║
╠══════════════════════════════════════════════════════════════════════╣
║  Strategy           QWK  Accuracy    F1-N      F1-E    F1-C    Time(s)║
║  ──────────────────────────────────────────────────────────────────  ║
║ 👑monai         0.8450    0.9062   0.912   0.815   0.951    450.0s  ║
║   nnunet        0.8234    0.8938   0.898   0.800   0.944    320.0s  ║
║   smp           0.8156    0.8875   0.891   0.792   0.938    180.0s  ║
║   yolo_seg      0.7812    0.8625   0.875   0.768   0.920     95.0s  ║
╠══════════════════════════════════════════════════════════════════════╣
║  👑 Best QWK: monai = 0.8450                                        ║
╚══════════════════════════════════════════════════════════════════════╝
```

> **نکته:** فقط استراتژی‌هایی که مدل‌شون در `submission/models/` وجود داره اجرا می‌شن.
> بقیه اسکیپ می‌شن با پیام warning.

---

## 🔍 منطق محاسبات

### Ground Truth
از فایل `training_df.csv`:
- داده‌ها در سطح slice هستن → aggregate به سطح study
- `triage_class` از CSV (توسط تیم annotator محاسبه شده)
- حجم خون‌ریزی‌ها: sum مساحت × spacing ÷ 1000 = mL
- `MLS_mm`: max در کل اسلایس‌های study
- `SkullFracture`: max (boolean OR)

### Prediction
با `submission/model.py` و `submission/triage.py`:
- `load_models(ich_strategy=...)` ← load مدل با استراتژی انتخابی
- `predict(study_dir)` ← ۷ مقدار intermediate
- `triage_from_intermediates()` ← کلاس 0/1/2

### Matching
فقط مطالعاتی که **هم** توی CSV هستن **و هم** پوشه DICOM دارن ارزیابی می‌شن.
بقیه با پیام warning رد می‌شن.

---

## 📦 وابستگی‌ها

- `pandas`, `numpy`, `scikit-learn`
- `pydicom`, `nibabel`, `torch`
- `nnunetv2`, `ultralytics`, `segmentation-models-pytorch`, `monai`
- (اختیاری) `tqdm` — برای progress bar
