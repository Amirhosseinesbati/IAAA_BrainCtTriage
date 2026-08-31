# دفترچهٔ پژوهش ICH-v2 — مسابقه IAAA Brain CT Triage 2026

آخرین به‌روزرسانی: ۲۰۲۶-۰۸-۳۱  
وضعیت: پژوهش فعال روی Vast.ai؛ بهترین مدل مستقیم و مستقل فعلی ICH، exp02 دوبعدونیم
با outer Any-ICH AUC=`0.9681` و volume MAE=`10.18mL` است. ترکیب قدیمی 2.5D gate
و SegResNet exp03 با Macro-F1 برابر `0.8498` فقط مرجع تاریخی است؛ در این مسیر
هیچ خروجی MLS یا شکستگی وارد آموزش/انتخاب ICH نمی‌شود و هنوز submission رسمی نشده است.

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
| 2.5D exp01 | EfficientNet-B0، سه slice × سه window، train folds2–4، calibration fold1، outer fold0 | outer AUC=0.9320، presence F1=0.8571 | gate پذیرفته شد |
| 2.5D exp01 + baseline 3D | rule ثابت fold1 | Macro-F1=0.8308، FPR=0.1351 | از baseline خام بهتر |
| 2.5D exp01 + exp03 3D | rule ثابت fold1 | **Macro-F1=0.8498**، FPR=0.1351، MAE=11.136mL | بهترین ترکیب معتبر فعلی |

شناسه‌های مهم MLflow:

- exp00 smoke: `39ac3f3f3fb44402ad9d1764f24813bd`
- exp02 full: `10d449699a4b4b11b479829059cfb35f`
- exp03 smoke: `505667ab072346cba4070c26226a8ddf`
- exp03 full: `bfc33803a25d499ab7b20efcc086c0b9`
- 2.5D exp01 full: `3b62ac04f05949b6b02fbb9a26e26ac3`
- 2.5D outer1/cal2: `584370aada734955a4fc0c9456320a79`
- 2.5D outer2/cal3: `8111a2a0640c46febde312828962a47b`
- 2.5D outer3/cal4: `bd20c867505b4b44aa9806a79806ccfa`
- 2.5D outer4/cal0: `4d75986ff0be4e638100b54de44c772d`

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

### ۹.۱ نتیجهٔ معماری 2.5D

پیاده‌سازی نهایی exp01 از سه slice مجاور و سه window مغز `(WL=40, WW=80)`، ساب‌دورال `(75, 215)` و استخوان/context `(600, 2800)` استفاده می‌کند؛ ورودی 9کاناله با اندازهٔ 320×320 به EfficientNet-B0 pretrained داده می‌شود. خروجی‌ها `AnyICH` و پنج subtype هستند.

برای جلوگیری از leakage:

- training فقط folds 2، 3 و 4 را می‌بیند؛
- checkpoint، pooling و threshold فقط با fold1 انتخاب می‌شوند؛
- fold0 تا پایان بهترین checkpoint دست‌نخورده می‌ماند؛
- threshold ثابت `0.744656` و pooling نوع `max` از fold1 به fold0 منتقل شده‌اند.

نتیجهٔ fold1: study AUC=`0.9440`، F1=`0.8696`، sensitivity=`0.9677` و specificity=`0.7778`.

نتیجهٔ fold0: study AUC=`0.9320`، F1=`0.8571`، sensitivity=`0.9091` و specificity=`0.8108`.

اعمال همین rule ثابت روی exp03 سه‌بعدی:

```text
confusion = [[29, 5, 0],
             [ 1,11, 0],
             [ 0, 3,21]]
Macro-F1 = 0.8497536
normal FPR = 0.1351351
catastrophic 0↔2 errors = 0
```

paired patient bootstrap با 5000 نمونه در برابر baseline خام:

- mean delta: `+0.13399`
- CI95 delta: `[+0.04460, +0.22747]`
- probability of improvement: `0.998`
- CI95 خود candidate: `[0.7500, 0.9300]`

بنابراین بهبود از گیت آماری پروژه عبور کرده است. threshold حجمی 2mL روی همان fold عدد `0.8718` می‌دهد، اما چون پس از مشاهدهٔ fold0 است فقط diagnostic باقی می‌ماند.

ضعف subtype مدل 2.5D روی outer: IVH AUC=`0.9163`، IPH=`0.9388`، SDH=`0.6275` و SAH=`0.6224`. در fold0 مثبت EDH وجود ندارد. هدف پژوهشی بعدی SDH/SAH و برآورد شدت/حجم است.

### ۹.۲ تأیید OOF پنج‌فولد 2.5D

همان hyperparameterها برای هر outer fold تکرار شدند. در هر اجرا یک fold مستقل برای calibration کنار گذاشته شد و سه fold باقی‌مانده training بودند؛ بنابراین هر 338 study دقیقاً یک‌بار outer prediction دارد.

| outer fold | AUC | F1 | sensitivity | specificity |
|---:|---:|---:|---:|---:|
| 0 | 0.9320 | 0.8571 | 0.9091 | 0.8108 |
| 1 | 0.8468 | 0.7179 | 0.9032 | 0.4722 |
| 2 | 0.8647 | 0.7353 | 0.8065 | 0.6667 |
| 3 | 0.9382 | 0.8333 | 0.8065 | 0.8857 |
| 4 | 0.9227 | 0.8000 | 1.0000 | 0.5556 |

تجمیع بی‌طرفانهٔ ruleهای per-fold:

- macro outer AUC: `0.9009`
- worst-fold AUC: `0.8468`
- presence F1: `0.7865`
- sensitivity: `0.8861`
- specificity: `0.6778`
- confusion: `[[122,58],[18,140]]`

CDF normalization با reference همان calibration fold، AUC تجمیعی `0.8998` دارد. threshold انتخاب‌شده روی کل OOF با sensitivity `0.9684` فقط diagnostic است، چون labelهای همان OOF در انتخاب threshold استفاده شده‌اند.

OOF subtype macro AUC:

- IPH: `0.9141`؛ worst fold=`0.8594`
- IVH: `0.8487`؛ worst fold=`0.6530`
- SDH: `0.7463`؛ worst fold=`0.6275`
- SAH: `0.6181`؛ worst fold=`0.4974`
- EDH: `0.4746` روی foldهای دارای مثبت؛ فقط 16 study مثبت و بسیار ناپایدار

نتیجه: presence representation در تمام foldها معتبر است، اما probability calibration و subtypeهای کم‌نمونه پایدار نیستند. برای امتیاز ترکیبی OOF روی 338 study هنوز 3D volume OOF لازم است؛ checkpoint legacy فقط برای fold0 leakage-safe بود.

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

1. checkpointهای پذیرفته‌شدهٔ 2.5D و exp03 سه‌بعدی همراه hash/config/README به ساختار محلی `checkpoint/ich` منتقل شوند.
2. OOF پنج‌فولد 2.5D کامل شده است؛ score normalization، calibration مشترک و ensemble باید با protocol بدون leakage تثبیت شوند.
3. سه false-negative گیت و هفت false-positive fold0 تحلیل شده‌اند. تنها study `3604` از false-negativeها triage را خراب کرد؛ `270845` و `3416` با MLS/fracture درست ماندند.
4. SDH و SAH با sampling/loss هدفمند، resolution یا encoder قوی‌تر بهبود داده شوند. EDH به‌دلیل فقط 16 study به sampling و augmentation محتاطانه نیاز دارد.
5. features ترتیبی sliceها برای volume/severity regression توسعه یابد تا SDHهای `465` و `272689` که 3D کاملاً صفر داده و critical case `2034` که کم‌برآورد شده نجات داده شوند.
6. 3D volume OOF با training مستقل هر fold ساخته شود؛ بدون آن Macro-F1 ترکیبی کل 338 study قابل ادعا نیست.
7. calibration حجم فقط با cross-fitting یا validation مجزا انتخاب شود؛ thresholdهای 2/10mL fold0 صرفاً hypothesis generator هستند.
8. مدل نهایی 2.5D با 3D volume، MLS و fracture در package inference benchmark شود و محدودیت 15/30 دقیقه و 1GB رعایت شود.
9. قبل از ادعای رتبه، submission واقعی leaderboard لازم است.

## ۱۲. مرجع کدهای اصلی

- `src/strategies/ich_v2/dataset_builder.py`: ساخت dataset با partial supervision
- `src/strategies/ich_v2/geometry.py`: affine و حجم فیزیکی
- `src/strategies/ich_v2/data.py`: split و loader پایدار
- `src/strategies/ich_v2/losses.py`: masked Dice/Focal و teacher KL
- `src/strategies/ich_v2/train.py`: loop آموزش، MLflow و Telegram
- `src/strategies/ich_v2/evaluation.py`: معیارهای study-level
- `src/strategies/ich_2p5d/cache.py`: cache سه‌پنجره‌ای
- `src/strategies/ich_2p5d/data.py`: split سه‌گانهٔ بدون leakage و adjacent slices
- `src/strategies/ich_2p5d/train.py`: آموزش EfficientNet و calibration مستقل
- `src/strategies/ich_2p5d/gating.py`: اعمال rule ثابت روی حجم سه‌بعدی
- `scripts/train_ich_v2.py`: CLI آموزش
- `scripts/train_ich_2p5d.py`: CLI آموزش gate
- `scripts/apply_ich_presence_gate.py`: ترکیب 2.5D و 3D
- `scripts/analyze_ich_experiment.py`: مقایسهٔ هم‌تراز آزمایش‌ها
- `tests/test_ich_v2.py`: تست‌های correctness

## ۱۳. ممیزی supervision و مدل مستقیم 2.5D (2026-08-31)

### ۱۳.۱ نقص رسمی annotation و رد فرض خرابی انتقال

ممیزی از universe واقعی DICOM شروع شد، نه فقط ردیف‌های metadata:

- 338 مطالعه و 7683 برش DICOM وجود دارد؛
- `training_df.pkl` فقط 7508 برش را پوشش می‌دهد؛ 175 برش در 22 مطالعه هیچ ردیف
  metadata و هیچ JSON متناظر ندارند؛
- از 7508 برش دارای metadata، تعداد 1660 برش حداقل یک subtype مثبت دارند، اما
  ماسک decodeشده فقط در 1580 برش مثبت است؛
- هر 80 اختلاف متعلق به IVH و در 19 مطالعه است: metadata مثبت ولی ماسک فضایی
  کاملاً خالی؛ IPH/SDH/EDH/SAH هیچ اختلاف presence ندارند؛
- checksum تجمیعی همان 80 JSON روی local و Vast یکسان بود:
  `b4abc9df42399a2daafeab6b331cb4f9398abbc51780293e41777f8f27dd4fbe`؛
- checksum metadata در هر دو محیط یکسان بود:
  `0e00255ce7dcd6963a00db6c4d5a5dfdc5cff5a6fa8aec6ead5cc5da185cfe7d`.

پس مشکل از DVC/SCP نیست و در دادهٔ رسمی وجود دارد. schema نسخهٔ 3 دو نوع known را
جدا می‌کند: `classification_known=7508` و `segmentation_known=7428`. آن 80 برش
IVH برای classification حفظ ولی از voxel loss حذف می‌شوند؛ 175 برش فاقد metadata
از هر دو loss کنار گذاشته می‌شوند و هرگز background فرض نمی‌شوند. مجموع برش‌های
فضایی ناشناخته 255 است. تست نماینده روی مطالعات `2068`، `270872` و `1451` و سپس
بازسازی هر 338 مطالعه موفق بود. 28 تست ICH روی local و server پاس شدند.

اعتبارسنجی cache 320×320:

- manifest SHA-256:
  `d63fc4f5ffe1cde00ecfe2326f03b8d63727ab06d926b610c8bade8774391eae`؛
- total-volume MAE=`0.3943mL`، bias=`-0.3787mL` و Pearson=`0.9992`؛
- اختلاف بزرگ IVH مطالعهٔ `2068` برابر `25.01mL` مستقیماً ناشی از 11 ماسک IVH
  خالی رسمی است و به‌عنوان خطای cache تفسیر نمی‌شود؛
- حجم voxel از مساحت affine درون‌صفحه‌ای × `SliceThickness` ساخته می‌شود؛ 12
  مطالعه فاصلهٔ slice با thickness بیش از 5٪ اختلاف دارند و spacing برای حجم
  canonical استفاده نمی‌شود.

### ۱۳.۲ معماری مستقیم و protocol

مدل جدید `segmentation_models_pytorch U-Net++` با encoder نوع EfficientNet-B2،
ImageNet initialization، ورودی 9کاناله (سه slice مجاور × brain/subdural/bone
window) و خروجی categorical شش‌کلاسه به‌همراه auxiliary head شش‌خروجی است.
folds 2/3/4 training، fold1 calibration و fold0 outer untouched هستند. checkpoint
با `0.55×Dice + 0.30×Any-AUC + 0.15×macro-subtype-AUC` فقط روی fold1 انتخاب و
سپس یک‌بار روی fold0 ارزیابی می‌شود. حجم از تعداد پیکسل × حجم فیزیکی voxel cache
محاسبه می‌شود. ارزیابی فقط ICH است.

| آزمایش | تغییر | Cal selection | Outer Any AUC | Outer mean Dice | Outer volume MAE | Outer FPR |
|---|---|---:|---:|---:|---:|---:|
| exp00 smoke | 20 step | 0.2953 | 0.7658 | 0.0373 | 1805.38 | 1.000 |
| exp01 | loss هم‌وزن subtype | 0.6104 | 0.9296 | 0.3308 | 11.25 | 0.351 |
| exp02 | class-aware Dice/Focal، power=1، cap=8 | **0.6404** | **0.9681** | **0.3772** | **10.18** | **0.162** |
| exp03 | همان exp02 در رزولوشن 384×384 | 0.6194 | 0.9349 | 0.3870 | 9.91 | 0.243 |

جزئیات outer exp02: presence F1=`0.8696`، macro subtype AUC=`0.8280`،
IPH Dice=`0.7950`، IVH Dice=`0.5385`، SAH Dice=`0.0981` و SDH Dice=`0.0771`.
fold0 نمونهٔ EDH مثبت ندارد. checkpoint exp02 با SHA-256 زیر پذیرفته و محلی شد:

`32ee13c1fc3b4ea6b3c981e869f254271ba91a2a07561fa24355a3e96a0572d9`

مسیر محلی:
`checkpoint/ich/smp/2p5d/unetplusplus-efficientnet-b2-classweighted-exp02-20260831`

MLflow run id: `a16e0e8cd441442281891e85a95865fd`.

### ۱۳.۳ نتیجهٔ تحلیل imbalance و post-processing

از 7428 برش spatial-known، سهم پیکسل subtypeها: IPH=`0.2072%`، SDH=`0.0879%`،
IVH=`0.0330%`، EDH=`0.0243%` و SAH فقط `0.0149%` است. SAH فقط 23 بیمار مثبت
و median area برابر 377 پیکسل از 102400 پیکسل cache دارد. SDH نیز فقط 44 بیمار
مثبت دارد و افت calibration→outer آن نشانهٔ ناپایداری fold است.

threshold مستقل softmax فقط روی fold1 انتخاب شد: EDH=.25، IPH=.50، IVH=.50،
SAH=.03 و SDH=.05. این کار outer SAH Dice را به `0.0645` رساند، اما normal FPR
را 1.0 و volume MAE را `15.39mL` کرد؛ بنابراین candidate رد شد. این نتیجه نشان
می‌دهد ضعف SAH صرفاً argmax نیست و class logit آن واقعاً ضعیف است. exp02 با
class-aware loss بدون این post-processing، همزمان تمام معیارهای اصلی exp01 را
بهتر کرد و candidate فعلی شد.

### ۱۳.۴ آزمایش کنترل‌شدهٔ رزولوشن 384×384

برای آزمون فرضیهٔ «ضایعات نازک SAH/SDH در 320 پیکسل بیش‌ازحد کوچک می‌شوند»، cache
مستقل 384 ساخته شد؛ cache 320 دست‌نخورده ماند. تعداد study/slice و supervision با
نسخهٔ 320 دقیقاً یکسان بود: 338 مطالعه، 7683 برش، 7428 برش spatial-known و 255
برش spatial-unknown. SHA-256 manifest جدید برابر بود با:

`7dc4efcaa801425eab9f6c90c343e99fe4aa8c4cebcb0b88e41c898e45f53ce8`

validator مستقل روی cache جدید Pearson حجم کل=`0.999225` و MAE=`0.4164mL` را ثبت
کرد. اختلاف عمده همچنان مطالعهٔ `2068` و همان IVH رسمیِ فاقد ماسک بود. exp03 همهٔ
تنظیمات exp02، از جمله split، seed، optimizer، sampler، loss، class weights و
معیار انتخاب را ثابت نگه داشت و فقط ورودی را از 320 به 384 تغییر داد. بهترین epoch
برابر 8، MLflow run id برابر `22c497042643464da16bd37510fbcf6b` و SHA-256
checkpoint برابر بود با:

`323608179ae8fcd55092d8f186d423a21edb65450d4fd5eaebf21dc44a81ce373`

نتیجهٔ outer exp03:

- selection=`0.61596`، Any-AUC=`0.93489` و macro subtype AUC=`0.81756`؛
- mean Dice=`0.38702`، presence F1=`0.8800` و normal FPR=`0.24324`؛
- total-volume MAE=`9.9071mL` و bias=`-5.6522mL`؛
- IPH Dice=`0.80279`، IVH=`0.55886`، SAH=`0.14326` و SDH=`0.04318`؛
- مصرف اوج VRAM=`5.47GB` و زمان آموزش=`12.97min`.

در مقایسهٔ paired روی همان outer، 384 نسبت به exp02 باعث `+0.00985` Dice میانگین،
`-0.276mL` MAE حجم و بهبود Dice در IPH/IVH/SAH شد، اما Any-AUC را `0.03317` و
selection را `0.00609` کاهش داد، FPR را `0.08108` افزایش داد و SDH Dice را
`0.03392` پایین آورد. بنابراین فرضیه فقط برای بخشی از subtypeها تأیید شد: exp03
به‌عنوان شاهد مکمل حفظ می‌شود، اما checkpoint اصلی را جایگزین نمی‌کند و به سیستم
محلی promote نشده است. exp02 همچنان candidate اصلی standalone ICH است.

EDA موجود همچنین ثابت می‌کند RLE یک ماسک semantic تک‌کاناله با کلاس‌های mutually
exclusive است؛ وجود چند subtype روی یک برش به معنی overlap پیکسلی نیست. بنابراین
softmax شش‌کلاسه از نظر نمایش target درست است. رفتن به sigmoid پنج‌کاناله فقط اگر
با loss مستقل rare-class مزیت تجربی نشان دهد توجیه دارد، نه با فرض همپوشانی برچسب.

### ۱۳.۵ وزن‌دهی بر اساس فراوانی پیکسل و ارزیابی OOF کامل

در exp02 وزن rare class از تعداد برش‌های مثبت به‌دست می‌آمد. EDA پیکسلی نشان داد
این تقریب شدت imbalance فضایی را کم‌برآورد می‌کند. وزن‌های واقعی training در fold0
برای ترتیب `[IVH, IPH, SDH, EDH, SAH]` چنین بودند:

- slice-frequency: `[2.2296, 1.0000, 2.3243, 8.0000, 6.1429]`؛
- pixel-frequency: `[6.6427, 1.0000, 3.5322, 8.0000, 8.0000]`.

محاسبهٔ pixel-frequency فقط از `segmentation_known=1` و cache ماسک foldهای training
انجام می‌شود؛ در نتیجه 80 ماسک رسمی خالی و 175 برش بدون metadata هیچ false-negative
فضایی وارد وزن یا loss نمی‌کنند. همهٔ hyperparameterهای دیگر ثابت ماندند. برای حذف
اثر خوش‌شانسی fold0، reference و candidate روی هر پنج outer fold آموزش مستقل دیدند.

| fold | reference selection | pixel selection | reference Dice | pixel Dice | reference FPR | pixel FPR |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.6221 | 0.6327 | 0.3772 | 0.3940 | 0.1622 | 0.4324 |
| 1 | 0.6422 | 0.6546 | 0.4320 | 0.4535 | 0.2222 | 0.3333 |
| 2 | 0.5921 | 0.6014 | 0.3786 | 0.4028 | 0.3889 | 0.3333 |
| 3 | 0.5810 | 0.5878 | 0.3312 | 0.3326 | 0.5714 | 0.7429 |
| 4 | 0.6566 | 0.6689 | 0.4533 | 0.4886 | 0.3056 | 0.3056 |

هر دو OOF شامل 338 مطالعه، 320 بیمار، 7683 برش و 7428 برش spatial-known هستند؛
هر مطالعه دقیقاً یک‌بار outer prediction دارد و هیچ بیمار بین outer foldها مشترک
نیست. تجمیع از prediction خام انجام شد، نه میانگین سادهٔ metricهای fold:

| معیار OOF | slice-frequency | pixel-frequency | اختلاف candidate-reference |
|---|---:|---:|---:|
| selection | 0.62078 | **0.63213** | +0.01135 |
| mean Dice | 0.40351 | **0.42233** | +0.01881 |
| Any-ICH AUC | **0.92504** | 0.92400 | -0.00104 |
| macro subtype AUC | 0.80894 | **0.81770** | +0.00876 |
| presence F1 در 0.1mL | **0.81744** | 0.79177 | -0.02566 |
| normal FPR در 0.1mL | **0.32778** | 0.42778 | +0.10000 |
| total-volume MAE | **8.62313mL** | 8.77215mL | +0.14901mL |

نتایج subtype Dice برای reference→pixel:

- EDH: `0.30763→0.34939`؛ IVH: `0.55016→0.56792`؛
- SAH: `0.06458→0.08260`؛ SDH: `0.31827→0.34453`؛
- IPH تنها افت کوچک داشت: `0.77693→0.76719`.

bootstrap جفت‌شده در سطح بیمار با 2000 نمونه و seed=42 نشان داد:

- Dice delta CI95=`[-0.00930,+0.04236]` و احتمال برتری candidate=`0.8945`؛
- selection delta CI95=`[-0.00529,+0.02520]` و احتمال برتری=`0.9085`؛
- macro-AUC delta CI95=`[-0.01692,+0.03691]`؛ Any-AUC تفاوت معنادار ندارد؛
- FPR delta CI95=`[+0.03172,+0.17143]` و احتمال بهتر بودن candidate فقط `0.002`؛
- F1 delta CI95=`[-0.05786,+0.00678]` و احتمال بهتر بودن candidate=`0.06`؛
- volume-MAE delta CI95=`[-0.43316,+0.71812]` و تفاوت قطعی نیست.

نتیجهٔ انتقادی: pixel-frequency یک spatial specialist معتبر است و در هر پنج fold
selection و Dice را بهتر کرد، ولی مدل نهایی standalone را به‌طور همه‌جانبه بهتر
نکرد. افت FPR قطعی است و ناشی از پیش‌بینی بیش‌ازحد rare classهاست. بنابراین exp04
در کنار exp02 حفظ و محلی شد، اما exp02 به‌عنوان checkpoint محافظه‌کارِ کم-FPR حذف
یا جایگزین نشد. مسیر محلی spatial specialist:

`checkpoint/ich/smp/2p5d/unetplusplus-efficientnet-b2-pixelweighted-exp04-20260831`

SHA-256 آن:
`5a826155a0ec7857b96a894ae6d1302aac8701ef850efbeb35079faa46174b51`.

ابزار بازتولیدپذیر `scripts/compare_ich_2p5d_segmentation_oof.py` علاوه بر کنترل
پوشش fold و patient leakage، Dice را از sufficient statistics پیکسلی بازسازی و
bootstrap جفت‌شده را اجرا می‌کند. خروجی سرور در مسیر زیر ثبت شده است:

`reports/ich_experiments/2p5d_segmentation/oof_slice_vs_pixel_p1_audited_v3`

provenance اجراهای افزوده‌شده:

| exp | fold/method | MLflow run id | checkpoint SHA-256 |
|---|---|---|---|
| 04 | f0 pixel | `acd1fe4833ab4f608b905f1b1c41a2aa` | `5a826155a0ec7857b96a894ae6d1302aac8701ef850efbeb35079faa46174b51` |
| 07 | f1 slice | `35200dec54db4919961a8a4eeae58a34` | `5ed6c983064ece95e44e7b7a85d4aeae2807a3f756e01a14697cd964407ce08ee` |
| 08 | f1 pixel | `17f73d8985e6471485889d4bc9b27205` | `ff10aaa039b9c653d589278528324efc66d3e84d4347222cfef7ea044b7e06e6d` |
| 05 | f2 slice | `c14d210daeb8450b93f558826e216b19` | `9ce08cfaba739bcb270f320832489426ef456f4c804d7507e1cf23a99e59a01e1` |
| 06 | f2 pixel | `a03427fa66714c89af677f1c6df2a71a` | `7b91203875d642e7b5c499cd7e1abb04f5992a4ba1a6357ae462ec76632bfa710` |
| 09 | f3 slice | `e1e9442dda6942d78a5b8fcda4e5724f` | `281e14eaa98850c0eaf0f2447788955e521ae1a5803dcc6d7352011dd9225af37` |
| 10 | f3 pixel | `e68e69900bb6432799dabfeadec53701` | `62ffa8271a4ab12c8613323fa31b9e21346aa38b488a0f2aea84c9466c306d876` |
| 11 | f4 slice | `c098524436c64fffac010072addcc2d8` | `12c78e6a3ef56e0039bb90465420b4d7259ead80c768fef39439190d3870a5c9` |
| 12 | f4 pixel | `ed9b062b3748467fa5f4221b0efb4de1` | `03cc3ba42fdea681f5f38d91ccf3be8781ac07a4b91a89792dfb7208bfe719a2` |

### ۱۳.۶ بازتنظیم background و آزمون تکرار مستقل

تحلیل loss نشان داد background فقط در focal CE وزن `0.15` دارد و Dice تنها subtypeهای
حاضر در batch را لحاظ می‌کند؛ بنابراین rare-class weighting می‌تواند بدون جریمهٔ
Dice برای کلاس غایب، false-positive بسازد. بر اساس نسبت جرم وزنی foreground در
focal، افزایش background از `0.15` به `0.20` به‌عنوان تنها تغییر exp13 تعریف شد.

exp13 روی outer fold0 و calibration fold1:

- بهترین epoch=7؛ calibration selection=`0.62602`، Dice=`0.39536`، FPR=`0.36111`؛
- outer selection=`0.65101`، Any-AUC=`0.96396`، macro subtype AUC=`0.85658`؛
- outer Dice=`0.42425`، FPR=`0.37838`، F1=`0.79487` و MAE=`11.1392mL`؛
- در برابر exp04 همان fold، selection، Dice، FPR و F1 بهتر شدند، ولی MAE از
  `10.1730` به `11.1392mL` بدتر شد؛ در برابر exp02 هنوز FPR و F1 ضعیف‌تر بود.

فرایند اصلی exp13 پس از پایان epoch10 و پیش از outer evaluation بدون traceback
خاتمه یافت. checkpoint انتخاب‌شده سالم بود و هیچ process/GPU job باقی نمانده بود.
به‌جای تکرار آموزش، evaluator بازیابی مستقل در
`scripts/evaluate_ich_2p5d_segmentation_checkpoint.py` اضافه شد؛ outer فقط از
checkpoint epoch7 بازسازی و در MLflow run `ea816734a6d3454ca562c49b26eab4f1`
ثبت شد. SHA-256 checkpoint:
`11557ff98240d4e85508f90aa6a15e003112652cab18370781d4877b0a235986`.

برای تأیید مستقل، exp14 با همان تغییر واحد روی outer fold2 اجرا شد:

- MLflow run=`b9919de855ec42738ca39bbf2f37001a`، best epoch=8 و checkpoint SHA-256=
  `9a948cd583770ca7fea28ba16fc5d3cde573ac7415c6e71f539c9f63ff545bd3`؛
- calibration selection=`0.67614` و Dice=`0.47795` از exp06 همان split اندکی بهتر
  بود، اما outer بهبود منتقل نشد؛
- نسبت به exp06، selection=`0.60138→0.57622`، Dice=`0.40276→0.36806`،
  Any-AUC=`0.86066→0.85439`، macro-AUC=`0.81107→0.78312`،
  FPR=`0.33333→0.38889`، F1=`0.82192→0.76712` و
  MAE=`8.8860→9.2485mL` همگی بدتر شدند.

نتیجه: background=`0.20` روی fold0 امیدوارکننده ولی روی fold مستقل شکست خورد؛
بنابراین به پنج fold گسترش یا promote نمی‌شود. این شکست همچنین نشان می‌دهد انتخاب
checkpoint با score فاقد FPR/MAE می‌تواند calibration ظاهراً بهتر ولی outer ضعیف‌تر
بسازد. جهت بعدی باید خود objective انتخاب/آموزش یا تفکیک presence از segmentation را
اصلاح کند، نه اینکه background weight بیشتری sweep شود.

سیاست Telegram نیز اصلاح شد: رویدادهای routine نوع `checkpoint/progress` به‌طور
پیش‌فرض خاموش‌اند و فقط start، success، failure، warning و گزارش‌های milestone
ارسال می‌شوند. پیام پایان اکنون selection، Dice، AUC، FPR و MAE را همراه با محدودیت
اعتبار و تصمیم بعدی توضیح می‌دهد. در صورت نیاز می‌توان با متغیر
`IAAA_TELEGRAM_EVENTS` این فیلتر را تغییر داد.

### ۱۳.۷ گیت‌های بعدی مستقل ICH

1. exp02 هنوز مدل نهایی نیست؛ ولی افزایش سادهٔ rare-class weight دیگر اولویت ندارد،
   چون هزینهٔ FPR آن روی OOF اثبات شد.
2. جهت بعدی باید spatial gain را از normal/presence decision جدا کند: head یا gate
   مستقل Any-ICH، کنترل gradient بین segmentation و classification، یا ensemble
   exp02/exp04 با rule کاملاً cross-fitted؛ thresholdگذاری روی همین OOF فقط diagnostic
   است و نباید به‌عنوان برآورد بی‌طرف گزارش شود.
3. sigmoid/Tversky یا encoder قوی‌تر فقط با protocol پنج‌fold یا غربال calibration
   اجرا شود؛ هدف صریح، بهبود SAH/SDH بدون افزایش FPR است.
4. EDH اکنون با 16 مطالعهٔ مثبت در OOF قابل مشاهده است ولی CI همچنان پهن است؛ هر
   تصمیم EDH باید patient-bootstrap و worst-fold را گزارش کند.
5. calibration/ensemble با MLS و شکستگی و بسته‌بندی leaderboard خارج از این task
   و متعلق به task تجمیع نهایی است.

### ۱۳.۸ پژوهش empty-label و exp15: جریمهٔ پایدار false-positive

جست‌وجوی هدفمند SciSpace در سه محور loss ضایعات کوچک، کنترل false-positive روی
تصاویر بدون ضایعه و joint classification/segmentation انجام شد. مقالهٔ MICCAI 2022
دربارهٔ Dice در حضور label خالی نشان می‌دهد reduction dimensions و smoothing رفتار
گرادیان را به‌طور بنیادی عوض می‌کنند و تنظیم متداول Dice لزوماً روی target خالی
سیگنال مفیدی نمی‌دهد. مقالهٔ Unified Focal و Focal-Tversky نیز برای imbalance مفیدند،
اما جهت اصلی آن‌ها افزایش sensitivity/recall است؛ با توجه به اینکه مشکل اثبات‌شدهٔ
مدل pixel-weighted ما افزایش FPR است، جایگزینی مستقیم loss با Tversky فعلاً فرضیهٔ
هم‌راستا با خطا نیست. منابع اصلی:

- Tilborghs et al., MICCAI 2022،
  `https://doi.org/10.1007/978-3-031-16443-9_51`؛
- Yeung et al., Unified Focal Loss،
  `https://doi.org/10.1016/j.compmedimag.2021.102026`؛
- Abraham and Khan, Focal Tversky Loss،
  `https://doi.org/10.1109/ISBI.2019.8759329`؛
- Ren et al., joint classification/segmentation with uncertainty،
  `https://doi.org/10.1007/978-3-031-43901-8_4`.

کالبدشکافی implementation نشان داد Dice فعلی تنها foreground classهایی را لحاظ
می‌کند که حداقل در batch حاضرند. برش کاملاً سالم فقط focal-CE background می‌گیرد؛
به علت عامل `(1-p)^2`، همین گرادیان روی negativeهای نسبتاً آسان سریع خاموش می‌شود.
این مسئله با صرفاً بالا بردن background weight حل قابل‌تکرار نداشت. در splitهای
outer0/cal1 و outer2/cal1، حدود `78.3%` و `78.8%` برش‌های training خام منفی‌اند، اما
subtype-aware sampling سهم مؤثر منفی را به `50.65%` و `51.39%` کاهش می‌دهد. بنابراین
negative data کافی است ولی objective از آن سیگنال پایدار نمی‌گیرد.

کاندید exp15 یک مؤلفهٔ `empty_foreground` اضافه می‌کند: میانگین CE بدون focal برای
background، فقط روی نمونه‌هایی که `segmentation_known=1` و target آن‌ها کاملاً خالی
است. 80 mismatch رسمی و 175 برش فاقد metadata هرگز وارد آن نمی‌شوند. default این
وزن صفر است تا checkpointها و آزمایش‌های قبلی بازتولیدپذیر بمانند. وزن آزمایشی
`0.05` است: در logits یکنواخت حدود 20% loss فضایی اولیه و در negativeهای آسان یک
فشار کوچک اما غیرخاموش ایجاد می‌کند. همهٔ متغیرهای exp04 ثابت می‌مانند.

گیت‌های ازپیش‌ثبت‌شده برای رفتن از fold0 به تأیید مستقل fold2:

1. در برابر exp04 همان split، FPR حداقل `0.05` مطلق کاهش یابد و F1 حضور کم نشود؛
2. Dice بیش از `0.01` و selection بیش از `0.005` افت نکند؛
3. Any-ICH AUC بیش از `0.01` افت نکند و MAE حجم علامت هشدار مستقل باقی بماند؛
4. هر NaN، گرادیان روی spatial-unknown، یا failure فنی کاندید را پیش از full run رد
   می‌کند؛
5. فقط اگر جهت اثر روی fold2 نیز تکرار شد، گسترش پنج‌fold یا promotion مجاز است.

تست‌های واحد مؤلفهٔ جدید ثابت می‌کنند negative معلوم گرادیان افزایش background و
کاهش foreground می‌گیرد، batch فقط مثبت جریمهٔ empty ندارد و classification-only
هیچ گرادیان فضایی دریافت نمی‌کند. 30 تست مرتبط محلی پس از تغییر پاس شدند.

### ۱۳.۹ نتیجهٔ exp15 و جداسازی خطای loss از خطای checkpoint selection

smoke دو-step سالم بود: loss و gradient محدود، peak VRAM=`3.86GB`، checkpoint و
MLflow موفق. full exp15 با تنها تغییر `empty_foreground_weight=0.05` نسبت به exp04
روی outer0/calibration1 اجرا شد. run در MLflow با شناسهٔ
`5e8d0c55b5234697b4c4741c93c95b12` و checkpoint SHA-256 زیر ثبت شد:

`a07405bab4febfd6505df6f6c89c3b0543f638197b1a5803a2f56d3aa20b2b86`

بهترین checkpoint با معیار قدیمی epoch3 بود. مقایسهٔ outer با exp04 همان split:

| معیار | exp04 | exp15 | delta candidate-reference |
|---|---:|---:|---:|
| selection | 0.63266 | 0.59777 | -0.03489 |
| Dice | 0.39401 | 0.35638 | -0.03764 |
| Any-AUC | 0.95741 | 0.92916 | -0.02826 |
| macro subtype AUC | 0.85819 | 0.82010 | -0.03809 |
| FPR | 0.43243 | 0.97297 | +0.54054 |
| presence F1 | 0.77500 | 0.64706 | -0.12794 |
| volume MAE | 10.173mL | 23.187mL | +13.014mL |

ابزار `scripts/evaluate_ich_segmentation_promotion.py` برابری manifest و config را
کنترل و پنج گیت ازپیش‌ثبت‌شده را ماشینی اعمال کرد. همهٔ گیت‌ها شکست خوردند؛ artifact
در `exp15.../promotion_gate.json` ثبت شد و exp15 به fold2 گسترش نمی‌یابد.

بااین‌حال trajectory یک failure mode قابل‌اقدام نشان داد. در epoch3، calibration
selection=`0.61619` ولی FPR=`0.97222` بود؛ در epoch6، selection تنها به `0.60930`
کاهش یافت، اما FPR=`0.36111`، F1=`0.81081` و MAE=`11.843mL` شد. چون selection فعلی
FPR/MAE را نمی‌بیند، epoch3 ذخیره و epoch6 دور انداخته شد. همچنین empty loss میانگین
از `0.431` در epoch1 به `0.00089` در epoch6 رسید؛ این مؤلفه غالب نشد، ولی average
روی 102400 پیکسل می‌تواند چند پیکسل بسیار سخت را پنهان کند. فقط 13 تا 66 پیکسل با
هندسهٔ داده برای عبور یک مطالعه از آستانهٔ `0.1mL` کافی است.

پیش از تغییر loss به top-k، exp16 فقط خطای انتخاب checkpoint را جدا می‌کند. score
جدید برابر `selection - 0.10 * normal_FPR` است. ضریب 0.10 از گیت قبلی مشتق می‌شود:
کاهش `0.05` در FPR می‌تواند حداکثر افت مجاز `0.005` در selection را جبران کند.
این قاعده اگر روی history exp15 اعمال می‌شد epoch6 را به epoch3 ترجیح می‌داد
(`0.57319` در برابر `0.51897`). training، seed، loss و split ثابت می‌مانند. فقط اگر
checkpoint ریسک‌آگاه outer را بهتر کند، سپس دربارهٔ hard-pixel loss تصمیم می‌گیریم.

### ۱۳.۱۰ exp16: بازیابی spatial gain با checkpoint ریسک‌آگاه

exp16 همان trajectory قطعی exp15 را بازتولید کرد و تنها selection strategy را به
`selection - 0.10*FPR` تغییر داد. برخلاف legacy selection، epoch3 با FPR=`0.97222`
promote نشد. best نهایی epoch8 با calibration selection=`0.62549`، Dice=`0.40387`،
FPR=`0.27778`، F1=`0.84507` و MAE=`11.186mL` بود. run در MLflow:

`5558a615f12c444e98bc0b0f4d68c74d`

checkpoint SHA-256:

`6b2a8f0a1a0fcae80bcbb33f7cb3fb8c01b05d7fc82747391c4a190d7f4989c9`

outer0 در مقایسه با exp04:

| معیار | exp04 | exp16 | delta candidate-reference |
|---|---:|---:|---:|
| selection | 0.63266 | **0.65324** | +0.02058 |
| Dice | 0.39401 | **0.43413** | +0.04012 |
| Any-AUC | 0.95741 | **0.96028** | +0.00287 |
| macro subtype AUC | **0.85819** | 0.84257 | -0.01562 |
| FPR | 0.43243 | **0.37838** | -0.05405 |
| presence F1 | 0.77500 | **0.79487** | +0.01987 |
| volume MAE | 10.173mL | **9.742mL** | -0.431mL |

هر پنج gate ماشینی پاس شدند و MAE نیز بدتر نشد؛ artifact در
`exp16.../promotion_gate.json` ثبت شد. افت کوچک macro subtype AUC هشدار باقی می‌ماند،
اما در گیت اولیه selection، Dice، Any-AUC، FPR و F1 هم‌زمان بهتر شده‌اند. بنابراین
exp17 بدون تغییر روش روی outer2/calibration1 آغاز شد؛ baseline مستقل آن exp06 است و
هیچ نتیجهٔ fold0 برای تغییر hyperparameter exp17 استفاده نمی‌شود.

### ۱۳.۱۱ exp17: شکست تأیید مستقل و توقف گسترش

exp17 روی outer2/calibration1 با همان loss، selection strategy، seed و hyperparameter
های exp16 اجرا شد. best ریسک‌آگاه epoch6 بود: calibration selection=`0.66756`،
Dice=`0.47922`، FPR=`0.33333`، F1=`0.82192` و MAE=`9.348mL`. run شناسهٔ زیر را دارد:

`fe1b276180244925be9a4365aecfd433`

checkpoint SHA-256:

`9b85659e94722f321cbb87e7e5699bee65b0ba12313085eb12ee301750556fb3`

outer2 در مقایسه با exp06 همان split:

| معیار | exp06 | exp17 | delta candidate-reference |
|---|---:|---:|---:|
| selection | **0.60138** | 0.57663 | -0.02475 |
| Dice | **0.40276** | 0.36637 | -0.03639 |
| Any-AUC | 0.86066 | **0.86470** | +0.00403 |
| macro subtype AUC | **0.81107** | 0.77141 | -0.03966 |
| FPR | **0.33333** | 0.41667 | +0.08333 |
| presence F1 | **0.82192** | 0.77333 | -0.04858 |
| volume MAE | **8.886mL** | 8.962mL | +0.076mL |

گیت ماشینی فقط شرط Any-AUC را پاس کرد و چهار شرط اصلی دیگر شکست خوردند. بنابراین
موفقیت exp16 روی fold0 قابل‌تکرار نیست، exp17 promote یا محلی نمی‌شود و این روش روی
foldهای 1/3/4 اجرا نخواهد شد. این نتیجه همچنین نشان می‌دهد selection ریسک‌آگاه
می‌تواند checkpoint داخلی معقول‌تری انتخاب کند، اما به‌تنهایی generalization loss
را اصلاح نمی‌کند.

جهت بعدی باید با خطای واقعی هم‌مقیاس شود: آستانهٔ 0.1mL فقط معادل 13 تا 66 پیکسل
در کل مطالعه است، درحالی‌که average empty CE روی 102400 پیکسل هر slice میانگین
می‌گیرد. بنابراین hard-pixel/CVaR یا soft-volume penalty با تمرکز بر چند پیکسل سخت،
کاندید منطقی‌تر از افزایش مجدد وزن average است. ابتدا gradient scale و smoke، سپس
یک fold توسعه و فقط در صورت عبور گیت، یک fold مستقل اجرا می‌شود.

### ۱۳.۱۲ exp18: empty hard-pixel mining با split چرخشی

OHEM در منبع اصلی CVPR 2016 برای موقعیتی معرفی شده که تعداد بسیار زیادی مثال آسان
و تعداد کمی مثال سخت وجود دارد؛ انتخاب آنلاین مثال‌های سخت سیگنال آموزشی را روی
خطاهای تعیین‌کننده متمرکز می‌کند:

`https://openaccess.thecvf.com/content_cvpr_2016/html/Shrivastava_Training_Region-Based_Object_CVPR_2016_paper.html`

این منطق با failure exp15/17 منطبق است، اما فقط روی negativeهای spatial-known اعمال
می‌شود تا حساسیت ضایعات مثبت مستقیماً محدود نشود. پارامتر جدید
`empty_foreground_top_fraction` با default=`1.0` اضافه شد؛ بنابراین همهٔ اجراهای قبلی
بازتولیدپذیرند. exp18 از fraction=`0.001` استفاده می‌کند: در ماسک 320×320 تقریباً
103 سخت‌ترین پیکسل هر برش سالم. focal-CE همچنان همهٔ پیکسل‌ها را می‌بیند و hard CE
با وزن `0.05` فقط false-positiveهای موضعی را هدف می‌گیرد. در logits یکنواخت مقیاس
loss با exp15 یکسان است و hyperparameter جدید صرفاً نحوهٔ aggregation را تغییر می‌دهد.

برای جلوگیری از ادامهٔ tuning روی outer0/2 که اکنون دیده شده‌اند، غربال exp18 روی
outer3/calibration1 و baseline هم‌fold exp10 انجام می‌شود. اگر پنج گیت پاس شوند،
تأیید بدون تغییر روی outer4/calibration1 در برابر exp12 خواهد بود. full OOF تنها پس
از تأیید outer4 مجاز است. 36 تست مرتبط محلی پس از پیاده‌سازی پاس شدند؛ تست hard-pixel
نشان داد یک لکهٔ منفرد سخت نسبت به average loss بیش از 20 برابر برجسته می‌شود.

### ۱۳.۱۳ نتیجهٔ exp18: عبور کامل گیت روی outer3

exp18 با `empty_foreground_weight=0.05`،
`empty_foreground_top_fraction=0.001` و checkpoint selection ریسک‌آگاه روی
outer3/calibration1 اجرا شد. early stopping پس از epoch7 رخ داد و checkpoint منتخب
epoch4 بود. provenance اجرای کامل:

- MLflow run id: `73f6355f22cd4a50b090ac49697ff614`؛
- checkpoint SHA-256:
  `cef60d76040c22196511c5b6671c5bd057f4a9b3badbd2a5ba3d1149e6289084`؛
- manifest SHA-256:
  `d63fc4f5ffe1cde00ecfe2326f03b8d63727ab06d926b610c8bade8774391eae`.

مقایسهٔ outer با exp10، یعنی baseline هم‌fold و pixel-weighted:

| معیار | exp10 | exp18 | delta candidate-reference |
|---|---:|---:|---:|
| selection | 0.58782 | **0.62661** | +0.03879 |
| Dice | 0.33259 | **0.38722** | +0.05463 |
| Any-AUC | 0.97327 | **0.99355** | +0.02028 |
| macro subtype AUC | 0.75279 | **0.77049** | +0.01770 |
| FPR | 0.74286 | **0.14286** | -0.60000 |
| presence F1 | 0.70455 | **0.89231** | +0.18776 |
| volume MAE | 8.921mL | **7.326mL** | -1.595mL |

هر پنج promotion gate پاس شد و MAE نیز بهتر شد. کاهش 60 واحد درصدی FPR همراه با
افزایش Dice، F1، AUC و selection نشان می‌دهد hard-pixel aggregation فقط مدل را
محافظه‌کار نکرده، بلکه خطاهای کوچک و تعیین‌کنندهٔ مطالعه را هدف گرفته است. این همان
تفاوتی است که average empty loss در exp15 نتوانست ایجاد کند: averaging روی 102400
پیکسل سیگنال چند ده پیکسل خطرناک را رقیق می‌کرد، اما top-0.1% آن‌ها را مستقیماً در
گرادیان نگه می‌دارد.

با وجود بزرگی اثر، outer3 یک fold غربال است و پذیرش نهایی مجاز نیست. exp19 بدون
هیچ تغییر در loss، معماری، seed یا hyperparameter روی outer4/calibration1 آغاز شد و
baseline آن exp12 است. فقط اگر همان پنج گیت روی outer4 نیز پاس شوند، روش برای تکمیل
OOF و انتقال checkpointهای برتر promote خواهد شد؛ در غیر این صورت موفقیت exp18
fold-specific تلقی می‌شود.

### ۱۳.۱۴ exp19: تأیید مستقل و قفل‌کردن روش

exp19 همان پیکربندی exp18 را بدون تغییر روی outer4/calibration1 اجرا کرد. آموزش تا
epoch10 ادامه یافت و checkpoint همان epoch با provenance زیر انتخاب شد:

- MLflow run id: `fd3a4d16c5ac401198f00f8e4d673e3e`؛
- checkpoint SHA-256:
  `a5c9688563455048b47a790c09e641f50bc1267583030767d37025c806f8f02e`؛
- manifest همان SHA-256 ثابت exp18 و baselineها را داشت.

مقایسهٔ outer با exp12 همان split:

| معیار | exp12 | exp19 | delta candidate-reference |
|---|---:|---:|---:|
| selection | 0.66886 | **0.68430** | +0.01544 |
| Dice | 0.48862 | **0.49206** | +0.00343 |
| Any-AUC | 0.94792 | **0.95312** | +0.00521 |
| macro subtype AUC | 0.77160 | **0.85155** | +0.07995 |
| FPR | 0.30556 | **0.16667** | -0.13889 |
| presence F1 | 0.85333 | **0.89855** | +0.04522 |
| volume MAE | **5.666mL** | 6.114mL | +0.448mL |

هر پنج گیت اصلی برای دومین fold متوالی پاس شد. بزرگی اثر FPR نسبت به exp18 کمتر
است، اما جهت اثر در selection، Dice، Any-AUC، macro-AUC، FPR و F1 همگی یکسان و
مثبت است؛ بنابراین موفقیت exp18 را نمی‌توان صرفاً fold-specific دانست. تنها هشدار
exp19 افزایش `0.448mL` در MAE حجم است. این مقدار مانع promotion اصلی نیست، ولی در
OOF کامل باید هم میانگین و هم worst-fold آن گزارش شود.

تصمیم: روش hard-pixel با fraction=`0.001`، وزن `0.05` و checkpoint selection
ریسک‌آگاه قفل می‌شود. تغییر hyperparameter در ادامه مجاز نیست. foldهای 0، 1 و 2
برای تکمیل OOF با همین recipe آموزش می‌بینند؛ exp18/19 foldهای 3 و 4 همان OOF خواهند
بود. پس از ساخت OOF کامل، patient-level coverage/leakage، paired bootstrap،
worst-fold و MAE بررسی می‌شوند و checkpointهای تأییدشده با provenance محلی خواهند
شد.

### ۱۳.۱۵ تکمیل OOF، exp20 روی outer0

پس از قفل‌شدن recipe، exp20 روی outer0/calibration1 اجرا شد. بهترین checkpoint
epoch7، MLflow run id برابر `4707b32b0d1643d99c054b4cc3f070fb` و SHA-256 آن:

`62d696d4d5f45f83c8a477893718142d643614203e025f55a929b9bc3539f1ef`

مقایسه با exp04 همان split:

| معیار | exp04 | exp20 | delta candidate-reference |
|---|---:|---:|---:|
| selection | 0.63266 | **0.64211** | +0.00945 |
| Dice | 0.39401 | **0.40372** | +0.00971 |
| Any-AUC | 0.95741 | **0.96724** | +0.00983 |
| macro subtype AUC | 0.85819 | **0.86592** | +0.00773 |
| FPR | 0.43243 | **0.10811** | -0.32432 |
| presence F1 | 0.77500 | **0.87879** | +0.10379 |
| volume MAE | **10.173mL** | 10.761mL | +0.588mL |

هر پنج گیت برای سومین fold پاس شدند. FPR بیش از 32 واحد درصد کاهش یافت و تمام
معیارهای classification/spatial بهتر شدند. افزایش کوچک MAE در exp19 و exp20 تکرار
شده و یک trade-off واقعی احتمالی است؛ نتیجهٔ نهایی باید با تجمیع patient-level پنج
fold تعیین کند آیا بهبود بزرگ FPR/F1 این هزینه را جبران می‌کند. exp21 روی outer1 با
calibration0 و سپس exp22 روی outer2/calibration1 بدون تغییر recipe اجرا می‌شوند.

اصلاح protocol: exp21 اولیه با outer1/calibration0 اجرا و outer summary آن سالم ثبت
شد، اما promotion checker پیش از مقایسه خطا داد، چون exp08 هم‌fold در واقع از
`calibration_fold=2` استفاده کرده بود. تغییر calibration fold مجموعهٔ train را نیز
عوض می‌کند؛ بنابراین exp21 اولیه نه خراب است و نه برای مقایسهٔ paired با exp08 مجاز.
به‌جای نادیده‌گرفتن guardrail، exp21b با outer1/calibration2 و همان recipe قفل‌شده
آغاز شد. فقط exp21b وارد OOF مقایسه‌ای نهایی می‌شود؛ exp21 به‌عنوان اجرای اضافی با
split متفاوت حفظ می‌شود. این اصلاح هیچ hyperparameter جدیدی معرفی نمی‌کند.

### ۱۳.۱۶ exp21b روی outer1: چهارمین fold موفق

exp21b با split صحیح outer1/calibration2 تا epoch10 اجرا شد و checkpoint منتخب
epoch9 بود. provenance:

- MLflow run id: `805b67a3c12449cebdd14f36a276a680`؛
- checkpoint SHA-256:
  `b3125e8c8a0994875b218d6dff5085dc63d619aa04660d68717fb354e7be9d4a`.

مقایسهٔ هم‌شرط با exp08:

| معیار | exp08 | exp21b | delta candidate-reference |
|---|---:|---:|---:|
| selection | 0.65462 | **0.66420** | +0.00959 |
| Dice | 0.45347 | **0.46343** | +0.00996 |
| Any-AUC | 0.90278 | **0.92384** | +0.02106 |
| macro subtype AUC | **0.89585** | 0.88112 | -0.01473 |
| FPR | 0.33333 | **0.25000** | -0.08333 |
| presence F1 | 0.82192 | **0.85714** | +0.03523 |
| volume MAE | 10.201mL | **10.157mL** | -0.045mL |

هر پنج گیت اصلی برای چهارمین fold پاس شدند و MAE اندکی بهتر شد. افت macro subtype
AUC به‌اندازهٔ `0.0147` تنها هشدار این fold است و نشان می‌دهد نتیجهٔ نهایی نباید فقط
به Any-ICH/FPR خلاصه شود. exp22 روی outer2/calibration1 آخرین fold recipe قفل‌شده
است؛ پس از آن OOF patient-level با همان پنج خروجی outer ساخته می‌شود.

### ۱۳.۱۷ exp22 روی outer2: تنها fold ناموفق در گیت سخت

exp22 با outer2/calibration1 تا epoch10 اجرا شد. provenance:

- MLflow run id: `f10d3e53a21f47aabcb5f2aa3e34d052`؛
- checkpoint SHA-256:
  `c63e609c652c2c15c2051c0c8da58a8060de31a7f67ecfa0f48d5e6d4338191d`؛
- best epoch: 10.

مقایسه با exp06 همان split:

| معیار | exp06 | exp22 | delta candidate-reference |
|---|---:|---:|---:|
| selection | **0.60138** | 0.58536 | -0.01602 |
| Dice | **0.40276** | 0.36618 | -0.03658 |
| Any-AUC | 0.86066 | **0.87231** | +0.01165 |
| macro subtype AUC | 0.81107 | **0.81513** | +0.00406 |
| FPR | 0.33333 | **0.19444** | -0.13889 |
| presence F1 | **0.82192** | 0.81250 | -0.00942 |
| volume MAE | **8.886mL** | 9.164mL | +0.277mL |

گیت FPR و Any-AUC پاس شدند، اما Dice، F1 و selection شکست خوردند؛ MAE نیز اندکی
بدتر شد. بنابراین recipe در چهار fold از پنج fold همهٔ گیت‌ها را پاس کرده و در
outer2 trade-off نامطلوب فضایی نشان داده است. این fold حذف، retune یا با checkpoint
دیگری جایگزین نمی‌شود، چون چنین کاری OOF را خوش‌بینانه می‌کند. تصمیم promotion کلی
فقط از OOF کامل 338 مطالعه و paired patient bootstrap گرفته خواهد شد.

### ۱۳.۱۸ OOF پنج‌fold: promotion کلی با محدودیت فضایی روشن

پنج outer prediction کاندید (`exp20`, `exp21b`, `exp22`, `exp18`, `exp19`) با پنج
baseline pixel-weighted هم‌fold (`exp04`, `exp08`, `exp06`, `exp10`, `exp12`) روی
338 مطالعه و 320 بیمار تجمیع شدند. کنترل ابزار تأیید کرد هر مطالعه و هر بیمار دقیقاً
در یک outer fold حضور دارد؛ 7428 برش spatial-known ارزیابی شدند. paired bootstrap
با 5000 بازنمونه و واحد بیمار اجرا شد. artifact کامل:

`reports/ich_experiments/2p5d_segmentation/oof_pixel_vs_hardpixel001_fprselect_audited_v3`

| معیار OOF | pixel baseline | hard-pixel | delta | CI95 delta | احتمال برتری کاندید |
|---|---:|---:|---:|---:|---:|
| selection | 0.63213 | **0.64402** | +0.01188 | [-0.00393, +0.03258] | 0.9280 |
| Dice | 0.42233 | **0.43506** | +0.01273 | [-0.01365, +0.04822] | 0.8096 |
| Any-AUC | 0.92400 | **0.93453** | +0.01053 | [+0.00031, +0.02139] | 0.9780 |
| macro subtype AUC | 0.81770 | **0.82916** | +0.01146 | [-0.01414, +0.03966] | 0.8054 |
| FPR | 0.42778 | **0.17222** | -0.25556 | [-0.32276, -0.19126] | 1.0000 |
| presence F1 | 0.79177 | **0.86826** | +0.07649 | [+0.04066, +0.11440] | 1.0000 |
| volume MAE | 8.772mL | **8.719mL** | -0.053mL | [-0.740, +0.682] | 0.5746 |

نتیجهٔ قطعی آماری، کاهش شدید FPR و افزایش F1 است؛ Any-AUC نیز CI کاملاً مثبت دارد.
selection، Dice و macro-AUC نقطه‌ای بهترند اما CI آن‌ها صفر را قطع می‌کند، پس نباید
به‌عنوان برد فضایی قطعی معرفی شوند. MAE حجم عملاً خنثی است. بااین‌حال total volume
bias از `-1.118mL` به `-3.144mL` منفی‌تر شده است؛ یعنی بخشی از کاهش FPR همراه با
محافظه‌کارترشدن حجم رخ داده، هرچند افزایش هم‌زمان F1 و Any-AUC ثابت می‌کند مدل صرفاً
همه‌چیز را خاموش نکرده است.

تحلیل زیرنوع OOF:

- EDH: Dice `0.349→0.393`، AUC `0.747→0.815` و MAE `2.416→1.959mL`؛ بهبود روشن
  نقطه‌ای، ولی فقط 16 مطالعهٔ مثبت دارد و uncertainty پهن است.
- IPH: Dice `0.767→0.762` تقریباً ثابت، AUC `0.945→0.953` بهتر و MAE تقریباً ثابت.
- IVH: AUC `0.932→0.943` بهتر، اما Dice `0.568→0.516` و MAE
  `1.187→1.360mL` بدتر؛ mismatchهای شناخته‌شدهٔ IVH همچنان یک محدودیت مهم‌اند.
- SAH: Dice `0.083→0.126` بهتر ولی AUC `0.712→0.676` و MAE کمی بدتر؛ presence
  و localization این subtype هنوز ناپایدار است.
- SDH: Dice `0.345→0.379`، AUC `0.752→0.758` و MAE `6.245→5.189mL` بهتر.

تصمیم: hard-pixel/fpr-select کاندید اصلی فعلی ICH است و جای pixel baseline را برای
ادامهٔ تحقیق می‌گیرد. بااین‌حال مدل نهایی بی‌نقص یا leaderboard-validated نیست.
مرحلهٔ بعد باید failureهای outer2 و افت IVH Dice/SAH AUC را بدون پس‌دادن برد قطعی
FPR/F1 هدف بگیرد. calibration با taskهای دیگر و بسته‌بندی leaderboard خارج از scope
این task مستقل ICH باقی می‌ماند.
