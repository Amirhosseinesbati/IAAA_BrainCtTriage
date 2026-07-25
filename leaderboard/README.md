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
├── __init__.py          # Package init
├── evaluate.py          # اسکریپت اصلی (CLI) — strategy-aware
├── ground_truth.py      # استخراج label های study-level از CSV
├── scorer.py            # محاسبه QWK + متریک‌های جانبی
└── README.md            # همین فایل
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
└── mls/                             # Midline shift (shared across all strategies)
    ├── slice_selector_best.ckpt
    └── keypoint_best.ckpt
```

> **نکته:** مدل‌های `yolo/` و `mls/` برای همه استراتژی‌های ICH یکسان هستند و فقط
> استراتژی ICH فرق می‌کند.

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
