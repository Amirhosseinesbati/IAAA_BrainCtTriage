# گزارش تصمیم مدل شکستگی YOLOv8s + Smooth-Attention MIL

**تاریخ ارزیابی:** 2026-08-31

**وضعیت:** بهترین کاندید فعلی، قابل اجرا و منتقل‌شده؛ هنوز تأییدشده با لیدربورد واقعی نیست

**MLflow meta-run:** `680232c493b2405288d73bdfd1a405fd`

## جمع‌بندی اجرایی

پنج مدل YOLOv8s مربوط به foldهای مستقل در epoch 10 با سه head کوچک Smooth-Attention MIL برای هر fold ترکیب شدند. SA-MIL به‌تنهایی از pooling مجاور detector ضعیف‌تر بود، اما اطلاعات آن مکمل detector بود. ترکیب ثابت و قابل‌استقرار با وزن `0.45` برای MIL، پس از empirical-CDFای که فقط روی outer-train هر مدل ساخته شد، Macro AUC را از `0.87370` به `0.90781` رساند. بهبود در هر پنج fold مشاهده شد.

از آن‌جا که قانون رسمی triage شکستگی را با `fracture_prob >= 0.5` مصرف می‌کند، استفاده مستقیم از CDF نامعتبر بود. threshold به‌صورت leave-one-outer-fold-out انتخاب شد و برای deployment، score نهایی با نگاشت یکنواخت piecewise به probability تبدیل شد تا score برابر `0.8363212059916904` دقیقاً به probability برابر `0.5` نگاشت شود.

در ارزیابی تصمیمی cross-fit، F1 از `0.4000` برای detector مرجع به `0.5484` رسید. این پیشرفت واقعی و معنادار است، ولی هنوز از هدف داخلی `F1 > 0.70` پایین‌تر است؛ بنابراین package فعلی باید «بهترین checkpoint عملی موجود» تلقی شود، نه پایان پژوهش.

## داده و پروتکل اعتبارسنجی

- 338 مطالعه، 320 بیمار و 7683 اسلایس؛ 28 مطالعه شکستگی‌دار.
- پنج outer fold در سطح بیمار؛ هیچ بیمار یا مطالعه‌ای بین train و validation تداخل ندارد.
- validation هر fold تمام اسلایس‌های مطالعه را شامل می‌شود.
- feature extractor هر fold فقط detector همان fold است.
- انتخاب alpha و epoch مدل MIL داخل outer-train و با splitهای patient-disjoint انجام شد.
- تمام نتایج OOF دقیقاً یک پیش‌بینی held-out برای هر مطالعه دارند.
- bootstrap زوجی با 20,000 تکرار انجام شد.

## نتایج مدل‌های پایه و مکمل

| مدل/روش | Macro AUC | Worst-fold AUC | تصمیم |
|---|---:|---:|---|
| YOLOv8s adjacent-pair pooling | 0.87370 | 0.79157 | مرجع |
| SA-MIL به‌تنهایی | 0.84529 | 0.77344 | رد به‌عنوان جایگزین |
| cohort-rank blend | 0.91290 | 0.85714 | فقط diagnostic؛ غیرقابل‌استقرار مستقل |
| train-CDF deployable، وزن‌های cross-fit | 0.90677 | 0.85317 | امیدوارکننده |
| train-CDF deployable، وزن ثابت 0.45 | **0.90781** | **0.85317** | انتخاب فعلی |

برای وزن ثابت `0.45`:

- اختلاف Macro AUC نسبت به مرجع: `+0.03411`.
- بازه 95 درصد bootstrap اختلاف: `[-0.00503, +0.07453]`.
- احتمال بهتر نبودن کاندید: `0.0436`.
- AUC foldها: `0.91927`, `0.85317`, `0.94068`, `0.95238`, `0.87354`.
- تمام پنج fold نسبت به مرجع بهتر شدند.

## نتیجه‌ی تصمیم در آستانه رسمی 0.5

threshold برای هر fold فقط با چهار fold دیگر انتخاب شد. نتیجه pooled OOF:

| روش | F1 | Precision | Recall | Specificity | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| detector مرجع با threshold cross-fit | 0.4000 | 0.3077 | 0.5714 | 0.8839 | 16 | 36 | 12 | 274 |
| detector + MIL ثابت 0.45 | **0.5484** | **0.5000** | **0.6071** | **0.9452** | **17** | **17** | **11** | **293** |

- اختلاف F1: `+0.14839`.
- CI 95 درصد اختلاف F1: `[+0.02312, +0.27112]`.
- احتمال بهتر نبودن کاندید: `0.01075`.
- threshold نهایی fit‌شده روی تمام OOF برای deployment: `0.8363212059916904`.

grid هم‌زمان وزن و threshold روی development foldها آزمایش شد، اما F1 held-out به `0.4516` افت کرد و bootstrap معنادار نبود. این مسیر به‌دلیل overfitting رد شد و وزن ساده‌ی ثابت `0.45` حفظ شد.

## صحت مسیر deployment

package واقعی با پنج detector و پانزده MIL head ساخته و روی RTX 3060 اجرا شد.

- تصاویر ساخته‌شده از مسیر مستقیم DICOM → HU → bone window → JPEG با cache مرجع در سه مطالعه نماینده کاملاً یکسان بودند: `max_abs_difference = 0`.
- بیشینه خطای adjacent score در برابر cache: کمتر از `9e-6`.
- بیشینه خطای MIL: کمتر از `7.7e-5`.
- خطای blend: عملاً صفر.
- peak VRAM: حدود `1.69 GB`.
- load اولیه: حدود `2–3 s`.
- runtime گرم برای مطالعات 16 و 39 اسلایسی: حدود `0.80 s` و `2.59 s`.
- مطالعه 50 اسلایسی با cold initialization: حدود `10.52 s`.

checkpointهای detector با ابزار استاندارد Ultralytics از optimizer و stateهای آموزشی strip شدند. حجم package از حدود `430 MB` به `115,038,454` بایت کاهش یافت و parity عددی بعد از strip دوباره پاس شد.

## artifact محلی

package در مسیر زیر قرار دارد:

`checkpoint/ich/fracture-yolov8s5fold-sa-mil-fixed045-v2-20260831`

محتوا:

- پنج `detector.pt` inference-only؛
- سه `model_seed*.pt` برای هر fold؛
- `manifest.json` شامل calibration، threshold، provenance و SHA-256؛
- `README.md` کوتاه.

پس از انتقال، هر 20 artifact روی سیستم محلی با SHA-256 manifest بررسی شد و `20/20` فایل بدون mismatch بودند.

## محدودیت‌ها و ریسک‌های باقی‌مانده

1. تنها 28 نمونه مثبت وجود دارد؛ CIهای AUC هنوز پهن‌اند.
2. F1 برابر `0.5484` برای مدل شکستگی بهتر از baseline است، ولی برای راهکار برنده کافی نیست.
3. برآورد OOF مربوط به یک مدل held-out برای هر مطالعه است؛ در test، میانگین پنج مدل استفاده می‌شود. ensemble باید پایدارتر باشد، ولی اثر آن بدون داده test مستقل مستقیماً قابل اندازه‌گیری نیست.
4. runtime fracture به‌تنهایی مناسب است، اما باید همراه ICH و MLS در package کامل و روی محدودیت 15 دقیقه benchmark شود.
5. pretrained بودن YOLOv8s/COCO باید از نظر قوانین external weights پیش از submission رسمی روشن شود.
6. هیچ ادعایی درباره leaderboard واقعی تا قبل از submission رسمی وجود ندارد.

## مسیر بعدی پیشنهادی

1. benchmark بزرگ‌تر runtime با نمونه‌ی طولی متوازن و سپس benchmark package کامل ICH+fracture+MLS.
2. تولید hard-negative taxonomy از 17 FP cross-fit و بررسی skull base، sutures و artefactها.
3. آزمایش detector/embeddingهای آموزش‌دیده با hard-negative mining، ابتدا روی دو fold برای کنترل هزینه.
4. بررسی classifier سطح مطالعه با featureهای frozen قوی‌تر و regularization مناسب داده کم.
5. ارزیابی downstream triage با OOFهای معتبر ICH و MLS؛ threshold شکستگی ممکن است برای Macro-F1 نهایی با threshold بهینه‌ی task-level متفاوت باشد، اما انتخاب باید nested/cross-fit باقی بماند.
6. پس از عبور package کامل از zero-failure، size و runtime gate، یک submission رسمی کنترل‌شده و تحلیل domain shift لیدربورد.

## فایل‌های مرجع

- `oof_v2/summary.json`: SA-MIL standalone.
- `crossfit_blend_v2/summary.json`: blend رتبه‌ای diagnostic.
- `deployable_fixed045_v2/summary.json`: AUC روش قابل‌استقرار.
- `threshold_crossfit_fixed045_v2/summary.json`: معیار تصمیمی cross-fit.
- `decision_crossfit_grid01_v2/summary.json`: آزمایش joint tuning ردشده.
- `package_validation_fixed045_v2_stripped/benchmark.json`: parity و runtime package نهایی.
- `package_validation_fixed045_v2_stripped/package_manifest.json`: provenance و hashها.
