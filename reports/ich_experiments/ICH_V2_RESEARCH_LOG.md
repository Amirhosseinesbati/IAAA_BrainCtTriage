# دفترچهٔ پژوهش ICH-v2 — مسابقه IAAA Brain CT Triage 2026

آخرین به‌روزرسانی: ۲۰۲۶-۰۸-۳۱  
وضعیت: پژوهش فعال روی Vast.ai؛ هیچ مدل جدیدی هنوز جایگزین baseline نشده است.

## ۱. هدف و معیار تصمیم

هدف این مسیر، بهبود جزء خونریزی داخل‌جمجمه‌ای (ICH) در سامانهٔ نهایی triage است؛ نه صرفاً افزایش Dice یک segmentation. معیار رسمی مسابقه Macro-F1 کلاس‌های triage صفر، یک و دو است. بنابراین checkpointها در این مرحله با یک ارزیابی study-level انتخاب می‌شوند که حجم فیزیکی پنج زیرنوع خونریزی را به rule نهایی triage می‌دهد.

در ارزیابی فعلی ICH، مقادیر واقعی MLS و fracture ثابت نگه داشته می‌شوند تا اثر جزء ICH از بقیهٔ pipeline جدا شود. نام این معیار در کد `oracle_context_macro_f1` است. این عدد برای مقایسهٔ مدل‌های ICH معتبر است، اما امتیاز قابل‌ارسال نهایی نیست؛ submission واقعی باید هر سه جزء پیش‌بینی‌شده را کنار هم اجرا کند.

امتیاز حدود `0.914` فقط مرجع لیدربردی گزارش‌شده توسط کاربر است و هنوز submission رسمی از این پروژه ثبت نشده است.

## ۲. ممیزی داده و supervision

### ۲.۱ آمار قطعی

| مورد | مقدار |
|---|---:|
| کل studyها | 338 |
| کل بیمارها | 320 |
| studyهای دارای JSON جزئی | 198 |
| clean-negativeهای معتبر | 140 |
| sliceهای برچسب‌دار در 198 study | 7,653 |
| sliceهای واقعاً فاقد JSON پس از تطبیق دقیق SOP | 30 |
| voxelهای foreground | 7,171,440 |
| foreground داخل ناحیهٔ unknown | 0 |
| حجم cache پردازش‌شده | 2,409,194,824 bytes |

تعریف clean-negative فقط وقتی پذیرفته می‌شود که `triage_class=0` و مساحت هر پنج زیرنوع EDH، IPH، IVH، SAH و SDH صفر باشد. ستون `AnyICH` برای این کار قابل اعتماد نیست، چون IVH-only را جا می‌اندازد.

در 198 مطالعهٔ JSONدار، slice فاقد JSON «پس‌زمینه» فرض نمی‌شود؛ supervision mask آن صفر است و هیچ loss یا distillation از آن عبور نمی‌کند. در 140 مطالعهٔ clean-negative، تمام voxelها پس‌زمینهٔ شناخته‌شده‌اند.

### ۲.۲ یکپارچگی و بازتولیدپذیری

- مسیر دادهٔ پردازش‌شده روی سرور: `/workspace/project/Data/processed/ich_v2/BrainICHPartial`
- SHA256 مانیفست ICH-v2: `e1d6294de29166ca011bf67e8b3c847af9106e5e59f51681dc6408f97ee87c6f`
- محدودهٔ حجم voxel: `0.0005526` تا `0.0030676` میلی‌لیتر
- ممیزی integrity: معتبر؛ هیچ foreground در ناحیهٔ unknown وجود ندارد.
- SHA256 مانیفست دادهٔ raw: `b7ba4b0d332d27738d21fae774dd776e8241182af6f8386ea4d4d7218b482271`
- raw روی سیستم محلی و Vast: 12,860 فایل و 2,908,071,754 bytes؛ تطبیق تأیید شده است.

## ۳. split و کنترل leakage

مانیفست fold شامل 338 study و 320 بیمار است. جداسازی patient-level انجام شده و اشتراک بیمار بین train و validation صفر است. تمام مقایسه‌های فعلی روی fold0 با 70 study انجام شده‌اند.

هشدار مهم: checkpoint قدیمی با folds 1–4 آموزش دیده و فقط برای validation روی fold0 بدون leakage است. استفاده از همان checkpoint برای مقداردهی اولیهٔ آزمایش validation روی fold1 تا fold4 نشتی مستقیم ایجاد می‌کند. برای آن foldها باید مدل متناظر از scratch یا checkpointی که آن fold را ندیده ساخته شود.

همچنین thresholdهایی که با خروجی fold0 انتخاب شوند، برای گزارش deployable روی همان fold معتبر نیستند. threshold یا calibration باید با cross-fitting یا fold جدا انتخاب شود.

## ۴. اصلاح هندسه و محاسبهٔ حجم

pipeline قدیمی پس از crop/resample، labelmap را با `Resize` ساده به ابعاد DICOM برمی‌گرداند. این تبدیل از نظر هندسی درست نیست و می‌تواند حجم خونریزی را تحریف کند.

ICH-v2 این موارد را اصلاح کرده است:

- affine از DICOM LPS برای آرایهٔ `(row, column, slice)` ساخته و به NIfTI RAS تبدیل می‌شود.
- حجم voxel از قدرمطلق دترمینان affine محاسبه می‌شود.
- حجم زیرنوع‌ها مستقیماً در فضای فیزیکی prediction محاسبه می‌شود؛ inverse resize وجود ندارد.
- حذف component کوچک با حجم واقعی میلی‌لیتر و connectivity بیست‌وشش‌تایی انجام می‌شود.
- حد فعلی component filter برابر `0.1 mL` است.

تست‌های واحد هندسه، حجم، component filtering، partial supervision و loss در `tests/test_ich_v2.py` ثبت شده‌اند.

## ۵. baseline معتبر فعلی

checkpoint اولیه:

- مسیر سرور: `/workspace/project/checkpoint/ich/monai/3d/segresnet-baseline-20260826/SegResNet_best.pth`
- SHA256: `5859c3dd6ebe4101a8169b0f5b940c23b48eba298110dbf9a731d01271839788`
- معماری: MONAI SegResNet سه‌بعدی، یک کانال ورودی، شش کلاس خروجی، `init_filters=16`
- تعداد پارامترها: 4,700,982

### ۵.۱ نتیجهٔ fold0 با هندسهٔ اصلاح‌شده

| معیار | بدون filter | component filter = 0.1 mL |
|---|---:|---:|
| Macro-F1 triage | 0.7056 | **0.7177** |
| normal FPR | 0.4865 | **0.4595** |
| presence F1 | 0.7407 | **0.7500** |
| total-volume MAE | 11.525 mL | 11.538 mL |
| total-volume bias | -8.494 mL | -8.533 mL |

confusion matrix پس از filter:

```text
[[20, 14,  0],
 [ 1, 10,  1],
 [ 0,  3, 21]]
```

F1 کلاس‌ها:

- کلاس 0: `0.7273`
- کلاس 1: `0.5128`
- کلاس 2: `0.9130`

F1 حضور زیرنوع‌ها:

- EDH: `0.0000`؛ در fold0 مثبت واقعی وجود ندارد و یک false positive ثبت شده است، پس توان تشخیص EDH از این fold قابل نتیجه‌گیری نیست.
- IPH: `0.8387`
- IVH: `0.6500`
- SAH: `0.0000`؛ شش false negative و recall صفر
- SDH: `0.2778`؛ 22 false positive و چهار false negative

تفسیر: ضعف اصلی baseline در تفکیک normal از خونریزی کم‌حجم و به‌خصوص false positiveهای SDH است. کلاس بحرانی خوب مانده، اما bias منفی نشان می‌دهد حجم موارد مثبت معمولاً کم‌برآورد می‌شود.

MLflow: run `b41db371cdfa4fc2bd375fd24b2d9b1c` در experiment شمارهٔ 19.

## ۶. دفتر آزمایش‌ها

| اجرا | تنظیم کلیدی | نتیجه | تصمیم |
|---|---|---|---|
| exp00 smoke | ROI 96، 12 train/6 val، یک epoch | Macro-F1=1.0 روی زیرمجموعهٔ بسیار کوچک؛ VRAM=2.35GB | فقط گیت فنی؛ ادعای کیفیت ندارد |
| exp01 full | LR=3e-5، Dice/Focal=0.6/0.4، background weight=0.2، crop تهاجمی | epoch1: Macro-F1=0.3936، FPR=1.0، bias=+8.98mL | در epoch2 متوقف و رد شد |
| exp02 full | LR=1e-5، Dice/Focal=0.3/0.7، background weight=1، crop=4/2 | Macro-F1=0.5061، FPR=0.8649، presence F1=0.6735، bias=-2.21mL، MAE=12.37mL | raw checkpoint رد شد |
| exp03 smoke distill | teacher ثابت روی partial-json، clean-negative آزاد، weight=1 | گیت فنی پاس؛ Macro-F1 کوچک=0.6667، VRAM=2.45GB، 36.7s | فقط صحت پیاده‌سازی؛ weight برای full به 10 افزایش یافت |
| exp03 full distill w10 | ROI 128، LR=1e-5، loss محافظه‌کارانه، یک epoch | Macro-F1=0.6843، FPR=0.5676، presence F1=0.7294، MAE=11.29mL | checkpoint خام رد شد؛ ranking برای gate حفظ شود |

شناسه‌های مهم MLflow:

- exp00 smoke: `39ac3f3f3fb44402ad9d1764f24813bd`
- exp02 full: `10d449699a4b4b11b479829059cfb35f`
- exp03 smoke: `505667ab072346cba4070c26226a8ddf`
- exp03 full: `bfc33803a25d499ab7b20efcc086c0b9`

### ۶.۱ نتیجهٔ دقیق exp03 full

distillation وزن 10 سقوط exp02 را تا حد زیادی مهار کرد، اما baseline را شکست نداد:

| معیار | baseline | exp03 | تغییر exp03 |
|---|---:|---:|---:|
| Macro-F1 | 0.7177 | 0.6843 | -0.0334 |
| normal FPR | 0.4595 | 0.5676 | +0.1081 |
| presence F1 | 0.7500 | 0.7294 | -0.0206 |
| MAE حجم کل | 11.538mL | 11.292mL | -0.247mL |
| bias حجم کل | -8.533mL | -7.763mL | +0.769mL |

confusion matrix خام exp03:

```text
[[16, 18,  0],
 [ 0, 11,  1],
 [ 0,  3, 21]]
```

مدل هیچ خطای catastrophic بین کلاس صفر و دو نداشت، recall کلاس یک به `0.9167` رسید، اما 18 مورد normal را کلاس یک اعلام کرد. false positiveهای زیرنوعی شامل SDH=27، IVH=14، IPH=8 و EDH=2 بود؛ SAH همچنان شش false negative و recall صفر دارد.

نکتهٔ مهم این است که حجم‌های کوچک کاذب بخش بزرگی از افت را ساخته‌اند. thresholdهای diagnostic روی همان fold چنین رفتاری داشتند:

| حداقل حجم کل | Macro-F1 | normal FPR |
|---:|---:|---:|
| 0.1mL | 0.6843 | 0.5676 |
| 0.5mL | 0.7781 | 0.3784 |
| 1mL | 0.8237 | 0.2162 |
| 2mL | 0.8577 | 0.1081 |
| 5mL | 0.8549 | 0.0541 |
| 10mL | 0.8688 | 0.0270 |

این جدول اثبات می‌کند که ranking/presence signal ارزشمند است، اما thresholdها چون روی همان fold انتخاب شده‌اند امتیاز deployable نیستند. نتیجهٔ صحیح، pivot به calibration یا presence gate با OOF است؛ نه استفادهٔ مستقیم از threshold 10mL.

### ۶.۲ failureهای فنی و درس آن‌ها

1. MONAI `MetaTensor` هنگام slicing loss روی transform historyهای ناهمگن خطا می‌داد. در مرز loss به tensor ساده تبدیل شد.
2. loader با workerهای دو و چهار و pin-memory دچار خطای IPC با متن `received 0 items of ancdata` شد. `/dev/shm` کافی بود، ولی `NOFILE=1024` و MetaTensor batching ناپایدار بودند. workers=0 هم پایدارتر و هم در sweep سریع‌تر بود؛ پیش‌فرض رسمی workers=0 شد.
3. افزودن نام metadata به‌شکل `supervision_type` و سپس `label_scope` با رفتار MONAI `Spacingd` تداخل داشت، زیرا این transform هر کلید شروع‌شونده با `<data-key>_` را metadata فرض می‌کند. نام بی‌تداخل `annotation_scope` انتخاب و batch واقعی بررسی شد.
4. نخستین invocation اسکریپت با مسیر فایل به‌علت import path شکست خورد؛ entrypoint رسمی `python -m scripts.train_ich_v2` است.

این failureها نتیجهٔ کیفیتی محسوب نمی‌شوند و checkpoint ناقص promote نشده است.

## ۷. تحلیل exp02 و ممنوعیت threshold leakage

exp02 مدل خوبی نشد، ولی ranking حجم کل بین negative و positive بهتر از raw decision rule بود:

- quantileهای حجم پیش‌بینی‌شده در GT-negative: `[0, 0.304, 2.073, 4.659, 65.102]`
- quantileهای حجم پیش‌بینی‌شده در GT-positive: `[1.278, 15.264, 29.138, 57.035, 157.896]`

گیت‌های 10 و 20 میلی‌لیتر روی همان fold به Macro-F1 حدود `0.800` و `0.811` رسیدند؛ این اعداد diagnostic و آلوده به انتخاب threshold روی validation هستند و نباید به‌عنوان مدل deployable گزارش شوند. پیام درست این تحلیل آن است که presence/calibration، نه صرفاً segmentation، اهرم مهمی است و باید با OOF یا مدل 2.5D مستقل حل شود.

## ۸. راهبرد teacher-distillation در exp03

student از baseline مقداردهی اولیه می‌شود. teacher نسخهٔ frozen همان baseline است.

- روی `partial_json`: segmentation loss ماسک‌شده به‌اضافهٔ KL distillation فقط در voxelهای شناخته‌شده اعمال می‌شود.
- روی `clean_negative`: KL صفر است تا student بتواند false positiveهای baseline را سرکوب کند.
- sliceهای unknown نه segmentation gradient دارند و نه distillation gradient.
- temperature فعلی `2.0` است.
- smoke نشان داد KL با weight=1 نسبت به loss اصلی بسیار ضعیف است؛ آزمایش کامل با weight=10 اجرا می‌شود.

فرضیهٔ قابل ابطال: distillation باید بخش مهمی از رفتار baseline روی مثبت‌ها را حفظ کند و clean-negative training باید FPR را کاهش دهد. اگر Macro-F1 خام یا حساسیت زیرنوع‌ها افت کند، checkpoint رد می‌شود و بدون تکرار بی‌هدف به presence gate مستقل می‌رویم.

## ۹. پژوهش خارجی و معماری پیشنهادی

نتیجهٔ مشترک پژوهش‌ها این است که context بین sliceها و multi-window برای تشخیص ICH مهم است، در حالی که segmentation سه‌بعدی برای حجم فیزیکی مناسب باقی می‌ماند.

- راه‌حل اول RSNA 2019 از adjacent slices، windowهای متعدد و مدل ترتیبی استفاده کرد: <https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection/writeups/seutao-1st-place-solution-sequential-model-wins>
- صفحهٔ رسمی چالش RSNA: <https://www.rsna.org/artificial-intelligence/ai-image-challenge/RSNA-Intracranial-Hemorrhage-Detection-Challenge-2019>
- مطالعهٔ 3D nnU-Net نشان می‌دهد focal loss برای imbalance و IVH مفید است: <https://pmc.ncbi.nlm.nih.gov/articles/PMC9745441/>
- ارزیابی cross-institutional nnU-Net اهمیت volume reliability را تأیید می‌کند: <https://pmc.ncbi.nlm.nih.gov/articles/PMC12429176/>
- منابع partial-label: arXiv `2206.09148` و `2007.03868`.

نتیجهٔ معماری: SegResNet سه‌بعدی فعلاً baseline حجم است و کنار گذاشته نشده؛ یک مدل 2.5D multi-window/adjacent-slice به‌عنوان presence/calibration gate مکمل طراحی می‌شود، نه جایگزین فوری segmentation. این ترکیب باید false positive نرمال و ضعف SAH/SDH را هدف بگیرد.

پلاگین SciSpace در context ابزار فعلی callable نبود؛ پژوهش با منابع اولیه و صفحات رسمی انجام شد. در صورت فعال‌شدن connector، مرور نظام‌مند مقالات از همان نقطه ادامه می‌یابد.

## ۱۰. زیرساخت و عملیات

- Vast instance: `49378919`
- GPU: RTX 3090 24GB
- مسیر پروژه: `/workspace/project`
- branch: `codex/competition-winning-pipeline`
- MLflow: DagsHub، experiment `IAAA_BrainCT-ich-v2`
- داده: DVC؛ secretها داخل `.env` با permission برابر 600
- Telegram: همهٔ پیام‌های این مسیر با عنوان ثابت مسابقه/ICH، متن فارسی، تحلیل کوتاه و اقدام بعدی ارسال می‌شوند.

اینستنس تا پایان goal و اعتبارسنجی واقعی leaderboard بدون اجازهٔ کاربر stop یا destroy نمی‌شود. اینستنس دیگر حساب خارج از scope و دست‌نخورده است.

## ۱۱. گیت‌های تصمیم بعدی

1. exp03 خام رد شده است؛ distillation سه‌بعدی فعلاً ادامه داده نمی‌شود، چون معیار اصلی و FPR را بهتر نکرد.
2. مدل 2.5D multi-window با adjacent slices برای presence/subtype ساخته شود. آموزش و threshold آن fold-safe و OOF باشد.
3. برای calibration حجم، thresholdها فقط با cross-fitting یا validation مجزا انتخاب شوند؛ جدول fold0 صرفاً hypothesis generator است.
4. ترکیب 2.5D gate و حجم 3D ارزیابی شود. هدف کوتاه‌مدت عبور معتبر از `0.7177` و سپس نزدیک‌شدن به بازهٔ diagnostic `0.82–0.87` بدون leakage است.
5. اگر gate مستقل ارزش افزوده نداد، مسیر بعدی segmentation از scratch با foldهای متعدد و loss/crop بازطراحی‌شده است؛ نه ادامه از checkpoint نشت‌دار.
6. فقط مدل دارای بهبود معتبر به مسیر محلی طبقه‌بندی‌شدهٔ `checkpoint/ich` منتقل شود و کنار آن README، config، hash، commit و metrics قرار گیرد.
7. قبل از ادعای نهایی، pipeline submission کامل با پکیج‌های مجاز leaderboard benchmark و سپس submission واقعی اعتبارسنجی شود.

## ۱۲. مرجع کدهای اصلی

- `src/strategies/ich_v2/dataset_builder.py`: ساخت dataset با partial supervision
- `src/strategies/ich_v2/geometry.py`: affine و حجم فیزیکی
- `src/strategies/ich_v2/data.py`: split و loader پایدار
- `src/strategies/ich_v2/losses.py`: masked Dice/Focal و teacher KL
- `src/strategies/ich_v2/train.py`: loop آموزش، MLflow و Telegram
- `src/strategies/ich_v2/evaluation.py`: معیارهای study-level
- `scripts/train_ich_v2.py`: CLI آموزش
- `scripts/analyze_ich_experiment.py`: مقایسهٔ هم‌تراز آزمایش‌ها
- `tests/test_ich_v2.py`: تست‌های correctness
