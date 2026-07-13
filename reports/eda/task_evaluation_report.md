# گزارش ارزیابی معماری سه‌تسک + ناهماهنگی با زیرساخت جدید

**تاریخ:** 2026-07-13  
**نگارش:** پس از اتمام فاز ۰ (EDA) و فاز ۱ (Shared Preprocessing)

---

## فهرست
1. [بررسی کلی معماری سه‌تسک](#1-بررسی-کلی-معماری-سه‌تسک)
2. [تسک ۱: ICH Segmentation (nnU-Net)](#2-تسک-۱-ich-segmentation-nnu-net)
3. [تسک ۲: Skull Fracture Detection (YOLO)](#3-تسک-۲-skull-fracture-detection-yolo)
4. [تسک ۳: Midline Shift Estimation (Custom CNN)](#4-تسک-۳-midline-shift-estimation-custom-cnn)
5. [ناهماهنگی‌های بین تسک‌ها و زیرساخت جدید](#5-ناهماهنگی‌های-بین-تسک‌ها-و-زیرساخت-جدید)
6. [جدول اولویت‌بندی اصلاحات](#6-جدول-اولویت‌بندی-اصلاحات)

---

## 1. بررسی کلی معماری سه‌تسک

### دیاگرام معماری فعلی

```
                       ┌──────────────────────────────────────────────┐
                       │             BrainDicomReader                  │
                       │     (خواندن DICOM، تبدیل HU، Windowing)       │
                       └──────────────────────┬───────────────────────┘
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    ▼                         ▼                         ▼
       ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
       │   ICH Predictor      │  │  Fracture Predictor  │  │   MLS Predictor      │
       │   (nnU-Net 3D)       │  │  (YOLOv8s 2D)        │  │  (ResNet18+ResNet34) │
       │                      │  │                      │  │                      │
       │  خروجی: ۵ حجم ICH    │  │  خروجی: bool         │  │  خروجی: float (mm)   │
       │  IVH,IPH,SDH,EDH,SAH │  │  Fracture Detected   │  │  Midline Shift       │
       └──────────┬───────────┘  └──────────┬───────────┘  └──────────┬───────────┘
                  │                         │                         │
                  └─────────────────────────┼─────────────────────────┘
                                            ▼
                               ┌─────────────────────────┐
                               │   apply_triage_rules()   │
                               │   (فعلاً: قوانین قدیمی)│
                               └──────────┬──────────────┘
                                          ▼
                               ┌─────────────────────────┐
                               │  Level 1 / Level 2 /    │
                               │  Normal                 │
                               └─────────────────────────┘
```

### خلاصه داده‌های آموزشی هر تسک (از EDA)

| معیار | ICH (nnU-Net) | Fracture (YOLO) | MLS (Custom) |
|-------|---------------|-----------------|--------------|
| **تعداد نمونه (سری)** | ۱۹۸ بیمار | ۲۸ بیمار | ۱۹۸ بیمار |
| **تعداد اسلایس** | ۵۱۷۶ اسلایس | ۳۵۶ bounding box | ۱۷۸۱ اسلایس (keypoint) |
| **نوع داده** | RLE Mask 3D | Bounding Box 2D | Keypoint 2D |
| **تعداد کلاس** | ۶ (bg + ۵ ICH) | ۱ (fracture) | ۳ keypoint regression |
| **Problem Type** | Semantic Segmentation | Object Detection | Binary Classification + Regression |
| **فایل آموزشی** | NIfTI (.nii.gz) | JPG + .txt (YOLO) | PNG + CSV |

> **نکته بحرانی:** اسلایس‌های دارای annotation ICH و MLS در اکثر بیماران یکسان هستند، اما در سطح سری (patient) متفاوت. این یعنی سه تسک از منابع DICOM یکسان استفاده می‌کنند، اما labelهای متفاوتی دارند.

---

## 2. تسک ۱: ICH Segmentation (nnU-Net)

### معماری

```
DICOM Patient
    │
    ▼
BrainDicomReader ──► HU Volume (H, W, D)
    │
    ▼
AnnotationParser ──► Mask Volume (D, H, W) از RLE
    │
    ▼
NNUnetDatasetBuilder ──► NIfTI files
    │
    ▼
nnUNetv2_train ──► checkpoint_best.pth
    │
    ▼
ICHPredictor (inference)
    │
    ▼
5 hemorrhage volumes (mL)
```

### نقاط قوت ✅
1. **انتخاب nnU-Net کاملاً درست است** — این مدل به طور خاص برای segmentation تصاویر پزشکی طراحی شده و در مسابقات متعدد (Medical Segmentation Decathlon) برتر بوده است.
2. **پیکربندی ۲D + ۳D_fullres** — هر دو کانفیگ در plans.json ذخیره شده که امکان مقایسه را فراهم می‌کند.
3. **Data augmentation خودکار** — nnU-Net به طور پیش‌فرض augmentationهای مؤثری اعمال می‌کند (rotation, scaling, noise, gamma correction).
4. **Loss تخصصی** — ترکیب Dice + CrossEntropy برای داده‌های نامتوازن مناسب است.
5. **Pipeline خودکار** — `nnUNetv2_plan_and_preprocess` به طور خودکار normalization و preprocessing را بر اساس آمار داده تنظیم می‌کند.

### نقاط ضعف ❌
| مشکل | شدت | توضیح |
|------|------|-------|
| **فقط Fold 0 آموزش دیده** | 🔴 بحرانی | `range(0)` در `__main__` باعث می‌شود هیچ foldی اجرا نشود. nnU-Net ذاتاً ۵-fold است و ensemble همه foldها دقت را بالا می‌برد |
| **Temp files در Inference** | 🟡 متوسط | ICHPredictor هر بار فایل NIfTI موقت می‌سازد و بعد پاک می‌کند. کند و شکننده |
| **ICH_MODEL_PATH از .env** | 🟡 متوسط | اگر متغیر env ست نشده باشد، fallback وجود ندارد |
| **عدم استفاده از config.py** | 🟡 متوسط | مسیرها در train_nnunet.py همچنان hardcoded هستند |
| **EDH در RLE masks دیده نشد** | 🟢 کم | EDA نشان داد EDH در نمونه ۲۰ تایی بررسی شده وجود نداشت — احتمالاً کلاس کمیاب است |
| **Overlap ۱۲.۸٪ بین کلاس‌ها** | 🟢 کم | در ۶۶ اسلایس بیش از یک کلاس ICH در mask وجود دارد. nnU-Net احتمالاً با softmax این را مدیریت می‌کند |

### جزئیات فنی کلیدی (از debug.json)
- **Patch size 2D:** (512, 512) — کل تصویر
- **Patch size 3D:** (16, 320, 320) — محدود به حافظه GPU
- **Optimizer:** SGD, Nesterov momentum 0.99, lr=0.01, PolyLR
- **Loss:** DC_and_CE_loss
- **Epochs:** 1000 (scheduled)
- **Iterations per epoch:** 250
- **Oversampling foreground:** 33%

### آنالیز حجم داده

| آمار | مقدار |
|------|-------|
| Median shape اصلی | (29, 512, 512) |
| Median spacing | (5.27, 0.482, 0.482) mm |
| تعداد training samples | ۱۹۸ |
| Max total ICH volume (per-series) | 488.4 mL |
| Mean total ICH volume (per-series) | 28.4 mL |

---

## 3. تسک ۲: Skull Fracture Detection (YOLO)

### معماری

```
DICOM Patient
    │
    ▼
BrainDicomReader ──► Per-slice HU
    │
    ▼
Bone Window (W:1000, L:400)
    │
    ▼
JPEG Image (512×512, grayscale→RGB)
    │
    ▼
YOLOv8s ──► Bounding Boxes
    │
    ▼
Any box with conf>0.5? ──► True/False
```

### نقاط قوت ✅
1. **YOLOv8s انتخاب مناسبی** — یکی از سریع‌ترین و دقیق‌ترین مدل‌های detection برای اشیاء کوچک.
2. **تنظیمات augmentation پزشکی** — mosaic=0, mixup=0, rotation محدود (۱۰°) — کار درستی برای تصاویر پزشکی.
3. **AdamW optimizer + lr=0.001** — مدرن و پایدار.
4. **Bone window** — شکستگی‌ها در bone window بهترین visibility را دارند.

### نقاط ضعف ❌
| مشکل | شدت | توضیح |
|------|------|-------|
| **۲۸ بیمار fracture داریم** | 🔴 بحرانی | فقط ۲۸ بیمار از ۳۳۸ دارای fracture bounding box هستند. این برای training یک مدل عمیق کافی نیست |
| **عدم توازن شدید کلاس** | 🔴 بحرانی | ۹۶.۵٪ منفی، ۳.۵٪ مثبت. بدون استراتژی خاص، مدل همیشه «بدون شکستگی» پیش‌بینی می‌کند |
| **تعداد bounding box کم** | 🟡 متوسط | ۳۵۶ bounding box برای کل دیتاست — یعنی ~۱۲ box per patient |
| **هیچ Data Augmentation خاصی** | 🟡 متوسط | YOLO خودش augmentation دارد، اما برای fracture شاید نیاز به flipping افقی بیشتر و rotation کمتر باشد |
| **تشخیص روی همه اسلایس‌ها** | 🟢 کم | `FracturePredictor` همه اسلایس‌ها را بررسی می‌کند که کند است. می‌توان فقط ۳-۵ اسلایس میانی را بررسی کرد |
| **هیچ ارزیابی (Precision/Recall)** | 🟡 متوسط | فعلاً راهی برای اندازه‌گیری دقت مدل روی validation set وجود ندارد |

### استراتژی Sampling فعلی
- **Fractured patients:** همه اسلایس‌های positive + ۱ اسلایس negative تصادفی
- **Healthy patients:** ۲۰٪ شانس انتخاب ۱ اسلایس
- **Train/Val split:** 80/20 با random seed=42

### تحلیل مشکل داده
داده‌های fracture شدیداً محدود هستند. استراتژی‌های ممکن:
1. **Data augmentation سنگین** — rotation تا ±۲۰°، scale، translation
2. **Transfer learning از fractureهای non-skull** — شاید از مدل‌های fracture استخوان‌های دیگر
3. **Pseudo-labeling** — استفاده از مدل روی ۱۴۰ patient بدون annotation
4. **قطعاً نیاز به oversampling** از کلاس minority

---

## 4. تسک ۳: Midline Shift Estimation (Custom CNN)

### معماری

```
DICOM Patient
    │
    ▼
BrainDicomReader ──► 3D HU Volume
    │
    ▼
Stage 1: Slice Selector (ResNet18)
    │   ورودی: 3-channel (Brain, Subdural, Bone)
    │   خروجی: best_z (اسلایس بهینه)
    ▼
Stage 2: Keypoint Detector (ResNet34)
    │   ورودی: 3-channel of best_z
    │   خروجی: 3 keypoints (x1,y1,x2,y2,x3,y3)
    ▼
MLS Calculation
    │   perpendicular distance formula
    ▼
MLS in mm
```

### نقاط قوت ✅
1. **معماری دو مرحله‌ای هوشمندانه است** — اول بهترین اسلایس را پیدا می‌کند، سپس روی آن کی‌پوینت می‌زند. شبیه به روش‌های SOTA.
2. **۳-channel windowing عالی است** — Brain, Subdural, Bone هر کدام اطلاعات متفاوتی می‌دهند و به مدل کمک می‌کنند.
3. **PyTorch Lightning** — کد تمیز، log خودکار، checkpointing.
4. **MLflow integration** — tracking metrics و model registry.
5. **SmoothL1Loss برای Keypoint** — نسبت به MSE به outliers مقاوم‌تر است.

### نقاط ضعف ❌
| مشکل | شدت | توضیح |
|------|------|-------|
| **فقط ۱۷۸۱ اسلایس مثبت** | 🔴 بحرانی | از ۱۹۸ بیمار، فقط ۱۷۸۱ اسلایس هر ۳ کی‌پوینت را دارند. تنوع آناتومیکی محدود |
| **۲ اسلایس منفی per patient** | 🟡 متوسط | در MLS builder فقط ۲ negative sample per patient. این باعث می‌شود SliceSelector همه چیز را مثبت پیش‌بینی کند |
| **Sigmoid در KeypointModel** | 🟡 متوسط | Sigmoid خروجی را به [0,1] محدود می‌کند. اگر کی‌پوینت‌ها نزدیک لبه تصویر باشند (x~0 یا x~512) مشکل ایجاد می‌کند. ReLU + Tanh یا hard constraints بهتر است |
| **هیچ قید آناتومیکی** | 🟡 متوسط | مدل می‌تواند کی‌پوینت‌ها را در هر جایی پیش‌بینی کند. روابط فضایی (Anterior بالاتر از Posterior) enforced نیست |
| **256×256 برای SliceSelector** | 🟢 کم | کاهش رزولوشن ممکن است جزئیات ظریف خط میانی را از بین ببرد |
| **MLS تابع دستی** | 🟢 کم | فرمول `perpendicular distance` درست است، اما فقط روی مختصات پیکسل اعمال می‌شود. نیاز به تبدیل به mm با spacing_x دارد که در کد انجام شده |
| **عدم استفاده از base_dataset** | 🟡 متوسط | train_mls.py هنوز از FastMlsDataset خودش استفاده می‌کند نه BrainCTDataset |

### فرمول MLS (در predictors.py)
```python
num = abs((x2 - x1)*(y1 - y3) - (x1 - x3)*(y2 - y1))
den = sqrt((x2 - x1)**2 + (y2 - y1)**2)
mls_px = num / den                     # فاصله عمود از نقطه ۳ به خط ۱-۲
mls_mm = mls_px * spacing_x
```
✅ این فرمول از نظر هندسی درست است: فاصله عمود از نقطه `(x3,y3)` به خط عبوری از `(x1,y1)` و `(x2,y2)`.

### تحلیل Coordinates کی‌پوینت‌ها (از EDA)
| Keypoint | محدوده X | محدوده Y | میانگین X | میانگین Y |
|----------|----------|----------|-----------|-----------|
| AnteriorFalxAttachment | ۱۰۱-۳۵۲ | ۳۰-۲۶۶ | ~۲۲۰ | ~۱۲۰ |
| PosteriorFalxAttachment | ۱۵۹-۳۳۳ | ۳۲۱-۴۷۴ | ~۲۵۰ | ~۴۰۰ |
| OutermostPointOfTheFalx | ۱۴۴-۴۷۸ | ۱۰۹-۳۱۰ | ~۲۸۰ | ~۲۰۰ |

> Point 1 (Anterior) در بخش بالایی و قدامی مغز است (Y کم)
> Point 2 (Posterior) در بخش پایینی و خلفی است (Y زیاد)
> Point 3 (Outermost) منحنی‌ترین نقطه فالکس است

---

## 5. ناهماهنگی‌های بین تسک‌ها و زیرساخت جدید

### 5.1. ناهماهنگی‌های بحرانی 🔴

| # | ناهماهنگی | فایل‌های affected | راهکار |
|---|-----------|------------------|--------|
| ۱ | **triage_rules.py هنوز قدیمی است** | `src/inference/triage_rules.py` | باید با تابع رسمی `triage_from_intermediates()` جایگزین شود. هم‌اکنون `src/analysis/eda_triage_simulation.py` کپی تابع رسمی را دارد |
| ۲ | **Training scripts از config.py استفاده نمی‌کنند** | `train_nnunet.py`, `train_yolo.py`, `train_mls.py` | مسیرها در این فایل‌ها hardcoded باقی مانده‌اند |
| ۳ | **train_nnunet.py: range(0)** | `train_nnunet.py:108` | `for i in range(0)` یعنی هیچ آموزشی هرگز اجرا نمی‌شود |
| ۴ | **Predictors از temp files استفاده می‌کنند** | `predictors.py: ICHPredictor` | به‌جای ذخیره موقت NIfTI، می‌توان مستقیماً آرایه numpy به nnUNetPredictor داد |
| ۵ | **main_predict.py به config.py وصل نیست** | `main_predict.py` | از مسیرهای hardcoded برای مدل‌ها استفاده می‌کند |

### 5.2. ناهماهنگی‌های متوسط 🟡

| # | ناهماهنگی | توضیح |
|---|-----------|--------|
| ۶ | **FastMlsDataset مجزاست** | train_mls.py Dataset خودش را دارد، از BrainCTDataset استفاده نمی‌کند |
| ۷ | **YOLO builder bone window hardcoded** | در yolo_builder.py قدیمی hardcoded بود. در بازسازی شده از config.WINDOWS استفاده می‌کند ✅ |
| ۸ | **MLFow tracking در training scripts** | train_nnunet.py از mlflow.set_tracking_uri محلی استفاده می‌کند، اما ZenML pipeline از DagsHub tracker. inconsistency |
| ۹ | **app.py مسیرها را hardcoded کرده** | ندارد -- از sys.path.append استفاده می‌کند که configuration متمرکز را دور می‌زند |

### 5.3. ناهماهنگی‌های غیرتطابق با مسابقه 🚨

| # | مشکل | توضیح |
|---|-------|--------|
| ۱۰ | **model.py وجود ندارد** | مسابقه یک ماژول استاندارد `model.py` با متد `predict(study_dir)` می‌خواهد. فعلاً چنین فایلی نداریم |
| ۱۱ | **خروجی predictors با intermediate schema مسابقه هماهنگ نیست** | مسابقه دیکشنری با کلیدهای `V_EDH, V_SDH, ..., fracture_prob, MLS_mm` می‌خواهد. ICHPredictor دیکشنری با کلیدهای کوتاه برمی‌گرداند |
| ۱۲ | **triage_rules.py از intermediate dict استفاده نمی‌کند** | تابع فعلی `apply_triage_rules(ich_volumes, has_fracture, mls_mm)` با signature مسابقه فرق دارد |

---

## 6. جدول اولویت‌بندی اصلاحات

| اولویت | تسک | اصلاح | زمان تخمینی | وابستگی |
|--------|-----|-------|------------|---------|
| **P0** | همه | جایگزینی `triage_rules.py` با تابع رسمی مسابقه | ۱ ساعت | — |
| **P0** | همه | ایجاد `model.py` با `predict(study_dir)` استاندارد | ۲ ساعت | P0 triage |
| **P0** | nnU-Net | رفع `range(0)` در `train_nnunet.py` | ۵ دقیقه | — |
| **P1** | همه | به‌روزرسانی training scripts برای استفاده از `config.py` | ۲ ساعت | — |
| **P1** | nnU-Net | بهبود ICHPredictor (حذف temp files) | ۲ ساعت | — |
| **P1** | MLS | بهبود negative sampling (بیشتر از ۲ اسلایس) | ۱ ساعت | — |
| **P1** | YOLO | افزودن Data Augmentation قوی‌تر | ۱ ساعت | — |
| **P2** | همه | هماهنگ‌سازی MLflow tracking بین local و DagsHub | ۱ ساعت | — |
| **P2** | Fracture | استراتژی برای داده محدود (Pseudo-labeling, augment) | ۳ ساعت | P1 YOLO |
| **P2** | MLS | افزودن قیود آناتومیکی به KeypointModel | ۲ ساعت | — |
| **P3** | همه | یکپارچه‌سازی با `BrainCTDataset` | ۳ ساعت | — |
| **P3** | nnU-Net | Cross-validation کامل ۵-fold | ۴ ساعت | P1 nnU-Net |

### خلاصه

```
وضعیت فعلی:  ۳ تسک مجزا با preprocessing ناهماهنگ
              ⬇
فاز ۰ + ۱:   Unified config + shared DICOM reader + shared AnnotationParser ✅
              ⬇
گام بعدی:    یکپارچه‌سازی training scripts + model.py استاندارد 
              + تابع رسمی تریاژ (اولویت P0)
```

---

*تاریخ نگارش: ۲۰۲۶-۰۷-۱۳ — پس از تکمیل فاز ۰ (EDA) و فاز ۱ (Shared Preprocessing)*
