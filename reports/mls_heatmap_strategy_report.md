# گزارش معماری و استراتژی تشخیص انحراف خط میانی (MLS) — استراتژی `mls_heatmap`

> **تاریخ:** ۲۰۲۶-۰۸-۰۵  
> **شاخه:** `feature/upgrade-mls`  
> **خلاصه:** گزارش جامع معماری، الگوریتم، داده، آموزش، inference، تست و یکپارچه‌سازی استراتژی Heatmap-based Keypoint Regression برای تخمین Midline Shift در مسابقه IAAA 2026 Brain CT Triage.

---

## ۱. مسئله و زمینه مسابقه

### ۱.۱ تعریف MLS

انحراف خط میانی (Midline Shift) از سه کی‌پوینت آناتومیک در اسلایس سطح **foramen of Monro** اندازه‌گیری می‌شود:

| # | نام کی‌پوینت | نقش |
|---|--------------|-----|
| ۱ | `AnteriorFalxAttachment` | اتصال قدامی فاکس — نقطه انتهایی خط فاکس ایده‌آل |
| ۲ | `PosteriorFalxAttachment` | اتصال خلفی فاکس — نقطه انتهایی دیگر خط فاکس |
| ۳ | `OutermostPointOfTheFalx` | بیرونی‌ترین نقطه فاکس — محل اندازه‌گیری انحراف |

**تعریف ریاضی:** MLS فاصله عمود از کی‌پوینت ۳ به خط فاکس ایده‌آل (خط گذرنده از کی‌پوینت‌های ۱ و ۲) است که از پیکسل به میلی‌متر با `PixelSpacing` دیکام تبدیل می‌شود.

### ۱.۲ اهمیت آستانه‌های تریاژ

مقدار نهایی `MLS_mm` در سطح سری است و مستقیماً در تابع رسمی تریاژ وارد می‌شود:

| آستانه | معنای بالینی | نقش در تریاژ |
|--------|--------------|--------------|
| `EPS_MLS = 1.0` mm | شیفت کمتر از ۱mm = نویز | به‌عنوان «شیفت معنادار» محسوب نمی‌شود |
| `MLS_URGENT_LOW = 3.0` mm | شیفت متوسط | شاخص Urgent؛ در قواعد ترکیبی با حجم خونریزی |
| `MLS_CRITICAL = 5.0` mm | شیفت بحرانی | شاخص Critical (به‌تنهایی یا همراه با خونریزی/شکستگی) |

به همین دلیل **دقت زیر-میلی‌متری هندسی** (به‌ویژه نزدیک آستانه‌های ۳ و ۵ میلی‌متر) حیاتی است — متریک نهایی مسابقه QWK روی کلاس تریاژ است و یک خطای کوچک در MLS می‌تواند کلاس را تغییر دهد.

---

## ۲. نمای کلی معماری — الگوی Strategy

### ۲.۱ چرا الگوی Strategy؟

پروژه سه تسک دارد (ICH Segmentation، Fracture Detection، MLS Estimation) و برای هر کدام چند روش ممکن است. معماری پروژه یک سیستم **پلاگین‌پذیر** ساخته که هر «استراتژی» یک پکیج مستقل با رابط یکسان است:

```
src/strategies/
├── base.py            → ICHStrategy (ABC)      ──┐  دو سیستم موازی
├── registry.py        → StrategyRegistry         │
├── mls_base.py        → MLSStrategy (ABC)     ──┘  (ICH و MLS)
├── mls_registry.py    → MLSStrategyRegistry
├── config_models.py   → تمام مدل‌های Pydantic (شامل MLSHeatmapConfig)
├── mls_heatmap/       → استراتژی جدید MLS (مورد این گزارش)
├── nnunet/ smp/ monai/ yolo_seg/  → استراتژی‌های ICH
└── __init__.py        → API عمومی + ثبت خودکار
```

### ۲.۲ قرارداد `MLSStrategy`

هر استراتژی MLS باید این اعضا را پیاده کند:

| عضو | امضا | توضیح |
|-----|------|-------|
| `name` / `display_name` / `description` | `ClassVar[str]` | شناسه و فراداده برای UI |
| `prepare_data()` | `-> bool` | ساخت/آماده‌سازی دیتاست |
| `train(config: BaseModel)` | `-> bool` | آموزش با کانفیگ اعتبارسنجی‌شده Pydantic |
| `predict(study_dir: str)` | `-> float` | خروجی `MLS_mm` |
| `get_config_class()` | `-> type[BaseModel]` | مدل Pydantic (موتور فرم پویای UI) |

ثبت خودکار: انتهای ماژول استراتژی `MLSStrategyRegistry.register(MLSHeatmapStrategy())` صدا زده می‌شود و `src/strategies/__init__.py` با ایمپورت ماژول آن را فعال می‌کند.

### ۲.۳ طراحی افزودنی (Add-on)

استراتژی جدید **کاملاً افزودنی** است — ساختار قدیمی آموزش MLS (‏`src/training/train_mls.py` + `mls_models.py`: SliceSelector ResNet18 + KeypointModel ResNet34) دست‌نخورده باقی می‌ماند و به‌عنوان fallback در `load_all_models()` و پکیج `submission` عمل می‌کند. تشخیص خودکار بر اساس وجود `models/checkpoints/mls_heatmap/mls_heatmap_best.pth` انجام می‌شود.

---

## ۳. معماری مدل — HRNet Heatmap

### ۳.۱ اجزای مدل (`model.py`)

```
ورودی (B, C, H, W)        C ∈ {1, 3}, H = W = 512
        │
        ▼
HRNet backbone (timm)     hrnet_w32 | hrnet_w18, features_only=True, out_indices=(1,)
        │
        ▼
Feature map 1/4           (B, 128, 128, 128)
        │
        ▼
HeatmapHead               Conv2d(128,64,3) → BN → ReLU → Conv2d(64,3,1)
        │
        ▼
خروجی (B, 3, 128, 128)    ۳ کانال گاوسی (یک کانال برای هر کی‌پوینت)
```

### ۳.۲ انتخاب backbone (قابل تنظیم)

| backbone | پارامتر تقریبی | کاربرد |
|----------|---------------|--------|
| `hrnet_w32` (پیش‌فرض) | ~۲۸.۵M | دقت بالاتر |
| `hrnet_w18` | ~۲۱.۳M | سریع‌تر و سبک‌تر (برای بودجه زمانی inference) |

### ۳.۳ سازگاری کانال ورودی

اگر `input_channels ≠ 3` باشد، اولین کانولوشن backbone جایگزین می‌شود:
- **۱ کانال:** میانگین وزن‌های RGB → تک‌کاناله
- **۲ کانال:** گرفتن دو کانال اول
- **>۳ کانال:** تکرار وزن‌ها

### ۳.۴ چرا Heatmap به‌جای رگرسیون مستقیم مختصات؟

| معیار | رگرسیون مستقیم (روش قبلی) | رگرسیون Heatmap (روش جدید) |
|-------|--------------------------|----------------------------|
| خروجی | ۶ عدد با Sigmoid (محدود به [0,1]) | هیت‌مپ فضایی (اطلاعات آماری توزیع) |
| دقت sub-pixel | ندارد (خروجی مستقیم) | دارد (DARK) |
| آموزش | یک مسیر برای همه کی‌پوینت‌ها | هر کانال توزیع مستقل + mask برای کی‌پوینت گمشده |
| کی‌پوینت گمشده | غیرقابل مدیریت | mask → حذف سهم loss |

---

## ۴. تولید Heatmap و استخراج مختصات Sub-Pixel

### ۴.۱ تولید گاوسی (`utils.generate_gaussian_heatmap`)

برای هر کی‌پوینت، یک گاوسی ۲بعدی در مختصات هیت‌مپ (مقیاس ۱/۴) قرار می‌گیرد:

```
heatmap[i] = exp( -‖(x,y) − (x_c, y_c)‖² / (2σ²) )
```

- **σ = 3.5 پیکسل هیت‌مپ** (پیش‌فرض، قابل تنظیم) — گاوسی پهن‌تر هدف آسان‌تری برای یادگیری روی داده کم می‌سازد و DARK دقت sub-pixel را در دیکد بازیابی می‌کند (خطای round-trip با σ=3.5: **0.0139px**)
- **کی‌پوینت گمشده (None):** کانال تمام‌صفر + `mask[i] = 0` — هیچ هدف مصنوعی ساخته نمی‌شود
- خروجی: `(K, H, W)` هیت‌مپ + `(K,)` ماسک حضور

### ۴.۲ دیکدینگ DARK — قلب دقت زیر-پیکسل

روش DARK (Distribution-Aware coordinate Representation, CVPR 2020):

1. یافتن پیکسل بیشینه هیت‌مپ `(x₀, y₀)`
2. محاسبه گرادیان و هسین در آن نقطه (مشتق مرکزی):
   - `g = (∂f/∂x, ∂f/∂y)`
   - `H = [[Hₓₓ, Hₓᵧ], [Hᵧₓ, Hᵧᵧ]]`
3. حل `H·Δ = −g` → افست زیر-پیکسل `Δ = (Δx, Δy)`
4. محدود کردن افست به `[−0.5, 0.5]` برای پایداری عددی
5. مقیاس به مختصات تصویر: `(x₀ + Δx) × (img_size / heatmap_size)`

**نتیجه عملی:** خطای گردشی (round-trip) حداکثر **0.0435 پیکسل** (آستانه موردنیاز 0.5px) — یعنی کی‌پوینت‌ها با دقت بسیار بالاتر از وضوح پیکسل استخراج می‌شوند. این همان دقتی است که برای آستانه‌های ۳ و ۵ میلی‌متر حیاتی است.

### ۴.۳ جایگزین Soft-argmax

`decode_soft_argmax` به‌عنوان روش جایگزین (با نرمال‌سازی softmax و میانگین موزون) ارائه شده؛ دقت کمتری از DARK دارد اما پایدارتر در هیت‌مپ‌های پراکنده.

---

## ۵. هندسه محاسبه MLS

### ۵.۱ فرمول

```python
dx = x₂ − x₁ ; dy = y₂ − y₁            # جهت خط فاکس
denom = √(dx² + dy²)
mls_px = |(x₂−x₁)(y₁−y₃) − (x₁−x₃)(y₂−y₁)| / denom
mls_mm = mls_px × spacing_x             # PixelSpacing دیکام (mm/px)
```

- اگر خط فاکس منحط باشد (نقاط ۱ و ۲ بر هم) → `0.0`
- نسخه برداری `compute_mls_batch` برای بچ

### ۵.۲ خاصیت مهم: نامتغیری نسبت به ترجمه صلب

جابه‌جایی یکسان هر سه کی‌پوینت مقدار MLS را تغییر نمی‌دهد (چون انحراف عمود به خط است). این در تست‌ها تأیید شده — برای ایجاد خطای MLS باید فقط یک کی‌پوینت جابه‌جا شود.

### ۵.۳ دقت سرتاسری تأییدشده

در تست زنجیره کامل (هیت‌مپ ثابت → دیکد → MLS): خروجی `3.3005mm` در برابر مقدار موردانتظار `3.3076mm` — **خطای 0.007 میلی‌متر**.

---

## ۶. پایپ‌لاین داده

### ۶.۱ منبع داده (رویکرد هیبرید)

دیتاست از خروجی `MlsDatasetBuilder` موجود (رویکرد هیبرید) مصرف می‌شود:

- **تصاویر:** PNG سه‌کاناله 512×512 (پنجره‌های brain 80/40، subdural 200/80، bone 1000/400)
- **برچسب:** CSV شامل مختصات کی‌پوینت‌ها

**آمار دیتاست فعلی:**

| معیار | مقدار |
|-------|-------|
| کل اسلایس‌ها | ۲۱۳۵ |
| اسلایس‌های هدف (`is_target=1`) | ۱۷۸۱ |
| اسلایس‌های منفی | ۳۵۴ |
| بیماران | ۱۷۷ |

### ۶.۲ `MLSHeatmapDataset`

- فیلتر به اسلایس‌های هدف (`is_target=1`)
- تولید هیت‌مپ **به‌صورت آنلاین** در `__getitem__`
- خروجی ۴-تایی: `(image, heatmap_target, mask, keypoints_true)` — کی‌پوینت واقعی برای محاسبه متریک validation واقعی

### ۶.۳ Augmentation با تبدیل سازگار کی‌پوینت

| تبدیل | پارامتر | اعمال هم‌زمان روی کی‌پوینت |
|-------|---------|----------------------------|
| چرخش حول مرکز | ±۱۵° (پیش‌فرض) | با ماتریس چرخش Affine |
| انتقال | تا ۵٪ اندازه تصویر | با بردار انتقال |
| نویز شدت (brightness/contrast) | ±۵٪ | فقط تصویر (هندسه حفظ می‌شود) |

احتمال اعمال مجموع تبدیل‌ها: `augment_prob = 0.9`

### ۶.۴ مدیریت کی‌پوینت گمشده (Masking)

مکانیزم به‌طور کامل پیاده و تست شده:

```
loss = Σ_k  MSE( heatmap_pred[k] × mask[k] ,  heatmap_target[k] × mask[k] )
```

برای کی‌پوینت گمشده (`mask=0`) سهم loss و گرادیان دقیقاً صفر است (تست `test_gradient_zero_for_missing_keypoint`).

### ۶.۵ تقسیم بیمارمحور (Patient-Level Split)

به‌جای تقسیم تصادفی ردیف‌ها، **کل بیمار** به train یا val تخصیص می‌یابد تا از نشت داده بین اسلایس‌های همبسته یک مطالعه جلوگیری شود:

| مجموعه | نمونه | بیمار |
|--------|-------|-------|
| Train | ۱۴۲۸ | ۱۴۲ |
| Val | ۳۵۳ | ۳۵ |
| تداخل بیماران | — | **۰** ✅ |

---

## ۷. حلقه آموزش (پایتورچ خالص)

### ۷.۱ چرا پایتورچ خالص؟

برخلاف پایتورچ لایتینگ، استراتژی `mls_heatmap` با **پایتورچ خالص** نوشته شده (هم‌الگو با `train_monai` در استراتژی‌های ICH) — کنترل کامل، وابستگی کمتر و لاگ مستقیم به MLflow.

### ۷.۲ پیکربندی (`MLSHeatmapConfig` — ۱۹ فیلد)

| دسته | فیلدهای کلیدی | پیش‌فرض |
|------|---------------|---------|
| معماری | `backbone`, `input_channels`, `image_size`, `head_dropout` | `hrnet_w32`, `3`, `512`, `0.1` |
| هیت‌مپ | `heatmap_sigma` | `3.5` |
| آموزش | `learning_rate`, `weight_decay`, `epochs`, `batch_size`, `val_split` | `1e-4`, `1e-3`, `100`, `8`, `0.2` |
| اسلایس | `top_k_slices`, `aggregation` | `3`, `max` |
| Augmentation | `rotation_deg`, `translation`, `intensity_jitter`, `augment_prob` | `15`, `0.05`, `0.05`, `0.9` |
| ابزار | `early_stopping_patience`, `lr_scheduler_patience`, `use_amp`, `num_workers`, `seed` | `15`, `5`, `True`, `4`, `42` |

### ۷.۳ جریان آموزش (`train_mls_heatmap`)

1. **Loss:** MSE ماسک‌شده روی هیت‌مپ
2. **بهینه‌ساز:** AdamW با `weight_decay=1e-3` (از کانفیگ — ضد overfitting)
3. **Scheduler:** Warmup خطی (۵ epoch) + Cosine Annealing (LambdaLR) — مستقل از نویز val_loss (به‌جای ReduceLROnPlateau که LR را خیلی زود و تهاجمی نصف می‌کرد)
4. **Regularization:** `Dropout2d(0.1)` در Head (در eval بدون اثر)
5. **AMP:** بله (روی CUDA)
6. **Early Stopping:** بر اساس `val_mls_mae_mm` (patience=15)
7. **Logging:** MLflow — همه پارامترها + متریک‌های هر epoch + آپلود `mls_heatmap_best.pth` در `artifact_path="models"` + اسنپ‌شات کد (`log_src_snapshot`)
8. **Checkpoint:** `models/checkpoints/mls_heatmap/mls_heatmap_best.pth` + `_final.pth` (شامل state_dict، کانفیگ و متریک‌ها)

### ۷.۴ متریک‌های Validation (به میلی‌متر، نه فقط loss هیت‌مپ)

| متریک | معنا |
|-------|------|
| `val_loss` | MSE هیت‌مپ (برای پایش) |
| `kp_mae_px` | میانگین خطای کی‌پوینت (پیکسل) — **مقایسه با کی‌پوینت واقعی** |
| `mls_mae_mm` | **MAE مقدار نهایی MLS به میلی‌متر — متریک اصلی checkpoint** |
| `mls_rmse_mm` | RMSE |
| `mls_bin_acc` | دقت دسته‌بندی در bins تریاژی `<1 / 1–3 / 3–5 / ≥5` mm |
| `bin_acc_per_bin` | دقت **تک‌تکِ** bins (`val_bin_acc_0..3` در MLflow) — محل تمرکز خطا را نشان می‌دهد |
| `mls_mae_critical` / `mls_mae_low` | MAE در ناحیه بحرانی (≥3mm) و غیربحرانی |

**نکته مهم:** مقدار مرجع MLS از **کی‌پوینت واقعی** دیتاست محاسبه می‌شود، نه با دیکد مجدد هیت‌مپ GT — تا خطای decode در متریک دو بار محاسبه نشود (رفع‌شده در فاز ۲).

---

## ۸. پایپ‌لاین Inference

### ۸.۱ `predict_mls(study_dir) -> float`

```
study_dir
   │
   ▼  BrainDicomReader.load_and_sort()
  volume (H, W, D) + spacing_x
   │
   ▼  ۱. SliceSelector (ResNet18) روی همه اسلایس‌ها (بچ‌های ۳۲، ورودی 256×256)
   │
   ▼  ۲. انتخاب Top-K اسلایس (K=3 پیش‌فرض)
   │
   ▼  ۳. مدل HRNet روی K اسلایس (بچ) → (K, 3, 128, 128)
   │
   ▼  ۴. دیکد DARK → کی‌پوینت‌های sub-pixel در فضای تصویر 512
   │
   ▼  ۵. محاسبه MLS هر اسلایس (spacing_x واقعی دیکام)
   │
   ▼  ۶. تجمیع: max (محافظه‌کارانه) یا p90
   │
   ▼
  MLS_mm (float)
```

### ۸.۲ چرا Top-K به‌جای یک اسلایس؟

اگر SliceSelector اسلایس اشتباهی را انتخاب کند، MLS حاصل از یک اسلایس خطای بزرگ خواهد داشت. با Top-K و تجمیع `max`، اثر خطای انتخاب اسلایس کاهش می‌یابد (بیشترین شیفت در میان K اسلایس کاندید — نگاه محافظه‌کارانه برای تریاژ). K قابل تنظیم است (`top_k_slices`).

### ۸.۳ مسیرهای checkpoint (چند لایه)

```
۱. آرگومان‌های صریح تابع
۲. متغیرهای محیطی MLS_SLICE_SELECTOR_PATH / MLS_HEATMAP_MODEL_PATH
۳. پیش‌فرض: models/checkpoints/slice_selector_best.ckpt
              models/checkpoints/mls_heatmap/mls_heatmap_best.pth
(خطای واضح FileNotFoundError اگر فایل نباشد)
```

### ۸.۴ `MLSHeatmapPredictor` (برای UI)

مدل‌ها را **یک‌بار** در `__init__` لود و کش می‌کند و رابط duck-typed `predict(reader)` دارد — سازگار با الگوی `models["mls"].predict(reader)` در `app.py` و مناسب `@st.cache_resource` استریم‌لیت. منطق مشترک با `predict_mls` از طریق `_run_pipeline()` در یک‌جا نگه‌داری می‌شود.

---

## ۹. یکپارچه‌سازی با پروژه

### ۹.۱ اتصال به UI

| مؤلفه | وضعیت |
|-------|-------|
| `app.py` (دمو) | بدون تغییر؛ `load_all_models()` خودکار heatmap را تشخیص می‌دهد (fallback به مدل قدیمی) |
| `src/deploy/deployApp.py` | سکشن MLS کامل: انتخاب استراتژی → فرم پویا از JSON Schema → `MLS_STRATEGY` + `MLS_CONFIG` |
| `src/deploy/deploy.py` | ارسال `MLS_STRATEGY` و `MLS_CONFIG_B64` (base64) به نمونه Vast.ai |
| `setup_vast.sh` | بلوک `TARGET_PIPELINE=mls`: decode کانفیگ → اجرای `--run mls-strategy --strategy mls_heatmap` |

### ۹.۲ پایپ‌لاین ZenML و CLI

- `mls_strategy_pipeline(strategy_name, config_json)` — آینه کامل `ich_pipeline`
- استپ‌ها: `prepare_mls_strategy_step` + `train_mls_strategy_step` (ولیدیشن Pydantic قبل از آموزش)
- CLI: `--run mls-strategy --strategy mls_heatmap --config '{...}'` + `--list-mls-strategies`

### ۹.۳ پکیج Submission (برای ارسال مسابقه)

`submission/model.py` شامل پیاده‌سازی خودکفای heatmap است:
- `_MLSHeatmapModel` (backbone timm + fallback به ResNet34 اگر timm نباشد)
- `_decode_heatmap_dark` / `_predict_mls_heatmap_pipeline`
- **تشخیص خودکار:** وجود `models/mls_heatmap/mls_heatmap_best.pth` → حالت heatmap، وگرنه حالت legacy
- `download_models.py`: `MODEL_TARGETS["mls_heatmap"] = "mls_heatmap"`

---

## ۱۰. تست‌ها (۲۷ تست واحد)

### `tests/test_heatmap_utils.py` — ۱۷ تست
تولید گاوسی (حضور کامل، گمشده، همه گمشده) • دیکد DARK (مختصات صحیح، sub-pixel، لبه تصویر، هیت‌مپ خالی، **round-trip < 0.5px**) • Soft-argmax • محاسبه MLS (مقادیر معلوم، افست، خط منحط، بچ) • binning تریاژی • مقایسه DARK vs Soft-argmax

### `tests/test_mls_integration.py` — ۱۰ تست
resolve مسیر checkpoint (صریح، env، خطا) • پنجره تک‌کاناله/سه‌کاناله • **loss ماسک‌شده: گرادیان صفر برای کی‌پوینت گمشده** • متریک‌های validation با کی‌پوینت واقعی (مدل کامل → خطای ~۰؛ مدل با کی‌پوینت جابه‌جا → خطای مثبت) • **زنجیره عددی `_run_pipeline` (خطای < 0.3mm)**

### نتایج کلیدی
- خطای DARK round-trip: **0.0435px**
- خطای زنجیره کامل MLS: **0.007mm**
- زمان اجرای کل تست‌ها: ~۰.۲ ثانیه

---

## ۱۱. ساختار فایل‌ها

```
src/strategies/mls_heatmap/
├── _strategy.py     استراتژی + ثبت خودکار
├── model.py         HRNetHeatmapModel + HeatmapHead
├── utils.py         گاوسی، DARK، MLS، binning
├── dataset.py       دیتاست + augmentation + split بیمارمحور
├── train.py         حلقه آموزش پایتورچ خالص + متریک‌ها
├── predict.py       predict_mls + _run_pipeline + MLSHeatmapPredictor
└── __init__.py      re-export

src/strategies/config_models.py    MLSHeatmapConfig (Pydantic)
src/pipelines/...                  mls_strategy_pipeline + CLI
setup_vast.sh                      دیپلوی Vast.ai
tests/test_heatmap_utils.py        تست‌های الگوریتم
tests/test_mls_integration.py      تست‌های یکپارچه‌سازی
```

---

## ۱۲. ارزیابی، مزیت‌ها و گام‌های بعدی

### مزیت‌های طراحی
1. **دقت sub-pixel** با DARK — حیاتی برای آستانه‌های ۳/۵mm
2. **متریک هدف واقعی** (`mls_mae_mm` + `mls_bin_acc`) به‌جای صرفاً loss هیت‌مپ
3. **پایداری در برابر خطای انتخاب اسلایس** با Top-K + تجمیع
4. **الگوی Strategy** — افزودنی، قابل تعویض، قابل اجرا از UI و Vast.ai
5. **پایتورچ خالص** — وابستگی کمتر و کنترل کامل

### گام‌های بعدی
1. **آموزش واقعی روی Vast.ai** (دانلود وزن pretrained HRNet، ~۲-۳ ساعت روی GPU 24GB)
2. ارزیابی `mls_mae_mm` و `mls_bin_acc` روی دیتای validation و مقایسه با pipeline قدیمی
3. (اختیاری) گنجاندن اسلایس‌های با کی‌پوینت ناقص در دیتاست برای استفاده بیشتر از مکانیزم masking
4. بازآموزی SliceSelector با منفی‌سایمینگ بهتر (پارامتر `negative_ratio` جدید)
5. ثبت `MLS_mm` واقعی هر بیمار (با `spacing_x` واقعی) در CSV برای بهبود متریک validation در آموزش نهایی

---

## ۱۳. نتیجه آموزش اول و اقدامات انجام‌شده (۲۰۲۶-۰۸-۰۵)

### ۱۳.۱ نتیجه اولین اجرا روی Vast.ai

| متریک | مقدار |
|-------|-------|
| بهترین `val_MLS_MAE` | **1.679 mm** (epoch 10) |
| `val_bin_acc` | ۶۱-۶۶٪ (نوسانی) |
| `kp_MAE` | ~۶.۱-۶.۵ px |
| Early stopping | epoch 25 |

### ۱۳.۲ تفسیر (مقایسه با بازلاین‌های واقعی)

| متریک | مدل | بازلاین | نتیجه |
|-------|-----|---------|-------|
| `val_bin_acc` | ۶۳-۶۵٪ | ۴۵.۳٪ (کلاس اکثریت) | +۱۸ تا ۲۰ واحد |
| `val_MLS_MAE` | 1.68mm | 3.94mm (پیش‌بینی میانه) | ~۲.۳ برابر بهتر |
| `val_loss` | 0.0006 | 0.0023 (پیش‌بینی صفر) | ~۴ برابر بهتر |

توزیع val (۳۵۳ اسلایس، split بیمارمحور) به سمت شیفت بالا است: ۴۵.۳٪ در bin ≥5mm، میانه 3.73mm — به همین دلیل بازلاین اکثریت فقط ۴۵٪ است.

### ۱۳.۳ تشخیص: Overfitting روی داده کوچک

- `train_loss` → 0 در ~۷ epoch اما `val_loss` از 0.0006 پایین نیامد → نشانه کلاسیک overfitting
- علت: ۱۴۲۸ اسلایس train + HRNet-W32 (۲۸.۵M پارامتر)
- ظرفیت داده اضافه ≈ صفر است (در کل ۵۱۷۶ JSON فقط ۳ اسلایس با کی‌پوینت ناقص وجود دارد)
- خطای کی‌پوینت ~۶px عمدتاً در جهت طول خط فاکس است (اثر اندک روی MLS) — بررسی شد

### ۱۳.۴ بهبودهای اعمال‌شده (کامیت `a27fcdd`)

1. `heatmap_sigma`: 2 → **3.5** (هدف نرم‌تر؛ DARK حتی بهتر شد: round-trip 0.0435 → **0.0139px**)
2. `augment_prob`: 0.5 → **0.9** و `rotation_deg`: 10 → **15** (تعمیم بهتر)
3. `weight_decay`: 1e-4 → **1e-3** (فیلد جدید کانفیگ)
4. `head_dropout`: **0.1** (Dropout2d در Head — بدون اثر در eval)
5. سچدولر: ReduceLROnPlateau → **Warmup(5) + Cosine** (LR تهاجمی نصف نمی‌شود)
6. متریک جدید: `bin_acc_per_bin` (دقت هر bin تریاژی به‌صورت جداگانه)

### ۱۳.۵ توصیه برای اجرای بعدی

```bash
uv run python -m src.pipelines.run_pipeline --run mls-strategy \
    --strategy mls_heatmap \
    --config '{"backbone":"hrnet_w32","epochs":120,"batch_size":8}'
```

نکات:
- اگر زمان مهم است، `hrnet_w18` را امتحان کنید (پارامتر کمتر → overfit کمتر)
- متریک `val_bin_acc_2` و `val_bin_acc_3` (bins حساس ۳-۵ و ≥۵) را در داشبورد MLflow دنبال کنید
- معیار موفقیت: `val_bin_acc` بالاتر از ~۷۰٪ و `val_MLS_MAE` زیر ۱.۲mm
