# 🏆 Personal Leaderboard — IAAA 2026 Brain CT Triage

یک لیدربورد شخصی (شبیه‌سازی شده) برای ارزیابی مدل‌های آموزش‌دیده روی دیتای خودمون.

این ماژول مدل‌های قرارگرفته در `submission/models/` رو می‌گیره، روی کل مطالعات DICOM موجود در
`data/raw/training/` اینفرنس می‌کنه، و با ground truth فایل `Data/metadata/training_df.csv`
مقایسه می‌کنه. معیار اصلی **Quadratic Weighted Kappa (QWK)** هستش — دقیقاً همون معیار رسمی مسابقه.

---

## 📁 ساختار

```
leaderboard/
├── __init__.py          # Package init
├── evaluate.py          # اسکریپت اصلی (CLI)
├── ground_truth.py      # استخراج label های study-level از CSV
├── scorer.py            # محاسبه QWK + متریک‌های جانبی
└── README.md            # همین فایل
```

---

## 🚀 نحوه استفاده

### ۱. قرار دادن مدل‌ها

اول مدل‌های آموزش‌دیده رو توی پوشه `submission/models/` با این ساختار قرار بده:

```
submission/models/
├── nnunet/
│   └── checkpoint_best.pth       # مدل ICH segmentation
├── yolo/
│   └── best.pt                   # مدل fracture detection
└── mls/
    ├── slice_selector_best.ckpt  # مدل انتخاب اسلایس MLS
    └── keypoint_best.ckpt        # مدل keypoint detection MLS
```

### ۲. اجرای لیدربورد

```bash
# از روت پروژه اجرا کن:
python -m leaderboard.evaluate

# یا با آپشن‌های دلخواه:
python -m leaderboard.evaluate --device cuda
python -m leaderboard.evaluate --device cpu          # اگه GPU نداری/نمی‌خوای
python -m leaderboard.evaluate --models-dir submission/models --device cuda
```

### ۳. آپشن‌های قابل تنظیم

| آپشن | پیش‌فرض | توضیح |
|------|---------|-------|
| `--models-dir` | `submission/models` | مسیر پوشه مدل‌ها |
| `--data-dir` | `data/raw/training` | مسیر پوشه دیتای خام DICOM |
| `--csv-path` | `Data/metadata/training_df.csv` | مسیر فایل CSV metadata |
| `--device` | `cuda` | دستگاه اجرا: `cuda` یا `cpu` |
| `--output-csv` | `leaderboard/results.csv` | خروجی CSV نتایج به ازای هر مطالعه |
| `--output-json` | `leaderboard/metrics.json` | خروجی JSON متریک‌ها |
| `--no-export` | `False` | ذخیره نکردن فایل‌های خروجی |

---

## 📊 خروجی نمونه

```
============================================================
  🏆  PERSONAL LEADERBOARD — IAAA 2026 Brain CT Triage
============================================================
  Models dir : submission\models
  Data dir   : data\raw\training
  CSV path   : Data\metadata\training_df.csv
  Device     : cuda
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
- `predict(study_dir)` → ۷ مقدار intermediate
- `triage_from_intermediates()` → کلاس 0/1/2

### Matching
فقط مطالعاتی که **هم** توی CSV هستن **و هم** پوشه DICOM دارن ارزیابی می‌شن.
بقیه با پیام warning رد می‌شن.

---

## 📦 وابستگی‌ها

همون وابستگی‌های اصلی پروژه:
- `pandas`, `numpy`, `scikit-learn`
- `pydicom`, `nibabel`, `torch`
- `nnunetv2`, `ultralytics`
- (اختیاری) `tqdm` — برای progress bar
