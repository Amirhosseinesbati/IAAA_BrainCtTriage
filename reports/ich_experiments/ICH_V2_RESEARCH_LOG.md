# دفترچهٔ پژوهش ICH-v2 — مسابقه IAAA Brain CT Triage 2026

آخرین به‌روزرسانی: ۲۰۲۶-۰۹-۰۲
وضعیت: متوقف به درخواست کاربر در ۲۰۲۶-۰۹-۰۲؛ قابل ادامه و بدون اجرای فعال. snapshot
جامع توقف، مصنوعات نجات‌یافته و دستور ادامه در
`reports/ich_experiments/ICH_STOP_AND_RESUME_2026-09-02_FA.md` ثبت شده است. بهترین
recipe مستقیم و مستقل فعلی ICH، پنج-fold
hard-pixel/fpr-select است. روی OOF کامل 338 مطالعه selection=`0.6440`، Any-ICH
AUC=`0.9345`، normal FPR=`0.1722`، presence F1=`0.8683` و volume MAE=`8.719mL`
دارد. checkpoint محلی candidate با پنج fold نگهداری شده، اما هنوز leaderboard-validated
نیست. ترکیب قدیمی 2.5D gate و SegResNet exp03 با Macro-F1=`0.8498` فقط مرجع تاریخی
است؛ هیچ خروجی MLS یا شکستگی وارد آموزش/انتخاب ICH نمی‌شود و هنوز submission رسمی
ثبت نشده است.

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

### ۱۳.۱۹ آزمون hybrid: رد blanket ensemble و حفظ سیگنال IVH

برای سنجش اینکه آیا می‌توان برد presence مدل hard-pixel را با maskهای spatial
baseline ترکیب کرد، یک hybrid کاملاً OOF و deterministic ساخته شد. gate هر مطالعه
همان شرط ثابت `candidate total volume >= 0.1mL` بود؛ در مطالعات gate-positive،
maskهای pixel baseline حفظ و در بقیه صفر شدند. scoreهای Any/subtype از hard-pixel
آمدند. هیچ threshold جدیدی fit نشد. ابزار و artifact:

- `scripts/analyze_ich_hardpixel_hybrid_oof.py`؛
- `reports/ich_experiments/2p5d_segmentation/oof_hardpixel_presence_gated_reference_spatial_v1`.

در برابر pixel baseline، hybrid در 2000 paired patient bootstrap همهٔ معیارهای
اصلی را بهتر کرد: FPR=`0.1667`، F1=`0.8709`، selection=`0.6380` و
MAE=`8.555mL`. CI selection `[+0.00039,+0.01136]` و MAE
`[-0.364,-0.107]mL` بود. confusion از `TP=154, FP=77, FN=4, TN=103` در baseline
به `TP=145, FP=30, FN=13, TN=150` رسید.

اما مقایسهٔ تصمیم‌ساز با خود hard-pixel نتیجهٔ blanket ensemble را رد کرد:

| معیار | hard-pixel | gated reference-mask hybrid | delta hybrid-hard-pixel |
|---|---:|---:|---:|
| selection | **0.64402** | 0.63797 | -0.00605 |
| Dice | **0.43506** | 0.42406 | -0.01099 |
| FPR | 0.17222 | **0.16667** | -0.00556 |
| F1 | 0.86826 | **0.87087** | +0.00261 |
| MAE | 8.719mL | 8.555mL | -0.164mL |

بهبود FPR/F1 فقط ناشی از نجات یک FP بود و MAE CI صفر را قطع کرد؛ در مقابل Dice و
selection نقطه‌ای بدتر شدند. پس اجرای کامل دو مجموعه checkpoint و جایگزینی همهٔ
maskها، هزینهٔ inference را توجیه نمی‌کند.

سیگنال مکانیکی hybrid همچنان مهم است: IVH Dice=`0.5685` و MAE=`1.182mL` تقریباً
به baseline بازگشت، درحالی‌که hard-pixel IVH Dice=`0.5156` داشت. اما این سود با از
دست‌دادن بهبودهای EDH/SAH/SDH همراه شد. بنابراین blanket hybrid رد می‌شود؛ جهت بعدی
باید branch/auxiliary objective یا ensemble هدفمند IVH را بررسی کند و SAH AUC و
outer2 را نیز زیر گیت عدم‌پسرفت FPR/F1 نگه دارد. هر انتخاب per-subtype روی همین OOF
صرفاً exploratory است و برای ادعای نهایی به تأیید جدا نیاز دارد.

### ۱۳.۲۰ تشخیص هدفمند IVH/SAH پیش از آزمایش بعدی

قبل از مصرف GPU برای یک معماری یا optimizer جدید، جهت افت زیرنوع‌ها در سه fold که
خلاصهٔ کاملشان محلی است جدا شد. مقادیر زیر delta مدل hard-pixel نسبت به pixel
baseline هم‌fold هستند:

| outer fold | delta Dice کل | delta IVH Dice | delta SAH AUC | FPR baseline→hard-pixel |
|---:|---:|---:|---:|---:|
| 0 | +0.0097 | +0.0067 | -0.0286 | 0.432→0.108 |
| 1 | +0.0100 | -0.0244 | -0.0556 | 0.333→0.250 |
| 2 | -0.0366 | -0.0532 | -0.0807 | 0.333→0.194 |

افت SAH-AUC در هر سه fold هم‌جهت است؛ IVH-Dice نیز در foldهای 1 و 2 افت می‌کند،
درحالی‌که FPR در هر سه بهتر است. OOF کامل همین trade-off را تأیید می‌کند: IVH-AUC
بهتر ولی IVH-Dice/MAE بدتر شده و SAH-Dice بهتر ولی SAH-AUC بدتر شده است. این الگو
با نوسان تصادفی یک fold توضیح داده نمی‌شود، اما هنوز اثبات نمی‌کند که علت حتماً
gradient conflict است.

مدل یک encoder مشترک دارد: decoder ماسک شش‌کلاسه و auxiliary classification head
هر دو از آن گرادیان می‌گیرند. hard-empty loss نیز فقط spatial-known maskهای کاملاً
خالی را هدف می‌گیرد. بنابراین سه سازوکار محتمل باید از هم جدا شوند:

1. فشار hard-negative روی background ممکن است نمایش ضایعات بسیار کوچک IVH/SAH را
   در encoder ضعیف کند؛
2. گرادیان segmentation و classification ممکن است برای SAH/IVH در جهت مخالف باشند؛
3. 80 برش IVH metadata-positive/mask-empty که در schema-v3 فقط classification-known
   هستند، سیگنال classification معتبر ولی بدون spatial target می‌دهند؛ کم‌بودن وزن
   head (`0.25`) ممکن است نتواند این سیگنال را در برابر objective فضایی حفظ کند.

PCGrad در مقالهٔ اصلی NeurIPS 2020 فقط وقتی گرادیان taskها cosine منفی دارد آن‌ها را
projection می‌کند و model-agnostic است، اما اجرای آن بدون اندازه‌گیری conflict یک
پیچیدگی بی‌دلیل خواهد بود:

`https://papers.neurips.cc/paper_files/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf`

از سوی دیگر یک مطالعهٔ مستقیم segmentation خونریزی/IVH روی CT گزارش کرده که focal
loss برای عدم‌توازن IVH از چند loss/معماری مقایسه‌شده بهتر بوده است. این شاهد، حفظ
focal objective فعلی را منطقی می‌کند و از تعویض فوری کل loss پشتیبانی نمی‌کند:

`https://pmc.ncbi.nlm.nih.gov/articles/PMC9745441/`

برای تصمیم evidence-first ابزار
`scripts/diagnose_ich_multitask_gradient_conflict.py` افزوده شد. این ابزار هیچ
پارامتری را update نمی‌کند و روی encoder مشترک موارد زیر را در batchهای واقعی train
اندازه می‌گیرد:

- cosine و نسبت norm گرادیان segmentation کامل در برابر classification کل؛
- همان مقادیر برای Any-ICH، IVH و SAH به‌صورت جدا؛
- جداسازی base Dice/Focal از hard-empty term؛
- سهم batchهای دارای spatial-empty mask و کسر batchهای دارای cosine منفی.

`py_compile`، import، آزمون مصنوعی cosine و 46 تست مرتبط ICH/loss/promotion پاس
شدند. اجرای smoke محلی به‌علت نبودن
cache مستقل `Data/processed/ich_2p5d` متوقف شد؛ raw data و checkpointها سالم‌اند و
این خطا کیفیت مدل نیست. cache کامل روی سرور موجود است. در همان زمان SSH سرور با
`Permission denied (publickey)` پاسخ داد و CLI رسمی `vastai` نیز در PATH این محیط
نبود؛ طبق guardrail افزونه هیچ نصب مجدد، API جایگزین، reboot، stop یا destroy انجام
نشد. پس از بازگشت دسترسی، اول diagnostic روی exp22 با batch size اصلی و حداقل 24
batch اجرا می‌شود. تنها اگر conflict منفی پایدار و از نظر norm معنادار باشد PCGrad
یا decoupling آزمایش می‌شود؛ در غیر این صورت جهت بعدی class-aware spatial objective
برای IVH/SAH خواهد بود.

### ۱۳.۲۱ منشأ outer2: shift ضایعات کوچک و sampler برش‌محور

برای جلوگیری از نسبت‌دادن شتاب‌زدهٔ افت IVH/SAH به gradient conflict، پنج checkpoint
محلی و سپس توزیع labelهای foldها مستقل از prediction مدل بررسی شد. checkpoint exp22
روی calibration1، IVH-Dice=`0.6836` و SAH-AUC=`0.9286` داشت، اما روی outer2 به
`0.5451` و `0.6224` رسید. در مقایسهٔ هم‌split calibration1 نیز exp20 hard-pixel
نسبت به exp04، SAH-AUC را از `0.5238` به `0.6270` بهتر کرده بود؛ بنابراین افت outer
به‌تنهایی تداخل قطعی loss یا انتخاب بد checkpoint را اثبات نمی‌کند.

ابزار label-only زیر بدون خواندن هیچ prediction مدل ساخته و اجرا شد:

- `scripts/analyze_ich_fold_subtype_shift.py`؛
- `reports/ich_experiments/fold_subtype_shift_audited_v1`.

کنترل ابزار 338 مطالعه، 320 بیمار، پوشش دقیق foldهای 0 تا 4 و patient-disjoint بودن
آن‌ها را تأیید کرد. نتیجهٔ اصلی این است که outer2 یک fold واقعاً متفاوت برای IVH
کوچک است:

| ویژگی IVH مثبت | outer2 | چهار fold دیگر |
|---|---:|---:|
| تعداد مطالعه | 12 | 41 |
| میانهٔ حجم | **3.118mL** | 13.856mL |
| delta میانه outer2-other | -10.739mL | CI95 bootstrap=[-17.729,-3.092] |
| حجم زیر 1mL | 25.0% | بسیار نادر |
| حجم زیر 2mL | 33.3% | بسیار نادر |
| میانهٔ برش مثبت | 3.5 | در foldهای بزرگ‌تر تا 13 |

outer2 هیچ برش metadata-positive/mask-empty نداشت؛ پس افت IVH آن از mismatchهای 80
برشی شناخته‌شده ناشی نمی‌شود. در مقابل fold2 شامل یک IVH فقط `0.305mL` و سه مطالعهٔ
isolated-IVH است. bias حجم IVH نیز از `-0.034mL` در pixel baseline به `-0.147mL`
در hard-pixel منفی‌تر شده است. مجموعهٔ این شواهد failure را به under-segmentation
ضایعات کوچک نزدیک می‌کند. SAH outer2 فقط سه مطالعهٔ مثبت دارد؛ بنابراین AUC fold-level
آن uncertainty بسیار بزرگی دارد و نباید به‌تنهایی مبنای تغییر معماری باشد.

تحلیل sampler نقص مکانیکی متناظر را نشان داد. sampler فعلی هر «برش مثبت» را وزن
می‌دهد، نه هر «مطالعهٔ مثبت» را. در train دقیق exp22، یعنی foldهای 0/3/4:

- 36 مطالعه و 330 برش IVH-positive وجود دارد؛
- فقط دو مطالعهٔ IVH زیر 2mL و مجموعاً چهار برش مثبت‌اند؛
- ضایعات بزرگ‌تر از 10mL، `75.28%` جرم sampling مثبت IVH را می‌گیرند؛
- ضایعات زیر 2mL فقط `1.35%` جرم را می‌گیرند؛
- Spearman حجم IVH با draw مورد انتظار هر مطالعه `0.763` است.

این همان lesion-size imbalance است که مقالهٔ inverse weighting گزارش می‌کند: lesion
بزرگ، loss/sampling را تحت سلطه می‌گیرد و وزن‌دهی معکوس می‌تواند recall ضایعات کوچک
را بهتر کند، هرچند ممکن است delineation را قربانی کند و باید با گیت سنجیده شود:

`https://arxiv.org/abs/2007.10033`

کار مستقل ICI loss نیز در MIDL/PMLR 2024 نشان داده objectiveهای instance-wise برای
small-instance detection می‌توانند Dice را نسبت به Dice استاندارد بهتر کنند:

`https://proceedings.mlr.press/v227/rachmadi24a.html`

به‌جای تعویض هم‌زمان loss و معماری، sampler مطالعه‌محور با پارامتر
`sampler_study_balance_power` پیاده شد. default=`0.0` رفتار تمام اجراهای قبلی را
دقیقاً حفظ می‌کند. در power>0 وزن هر برش مثبت با تعداد برش‌های مثبت همان subtype در
مطالعه تعدیل و سپس normalize می‌شود؛ بنابراین جرم کل مثبت و نسبت exposure مثبت/منفی
ثابت می‌ماند. شبیه‌سازی پیش از آموزش:

| strategy | سهم IVH کوچک ≤2mL | corr حجم↔draw | بیشینهٔ وزن یک برش |
|---|---:|---:|---:|
| current/p0 | 1.35% | 0.763 | 5.47 |
| p0.50 | 2.35% | 0.741 | 11.28 |
| **p0.75** | **3.29%** | **0.671** | 18.88 |
| p1.00 | 4.59% | 0.391 | 30.90 |

p1 تقریباً مطالعه‌ها را برابر می‌کند، اما تنها دو small-IVH train دارد و وزن 30.9
ریسک overfit به همان دو نمونه را بالا می‌برد. بنابراین p0.75 به‌عنوان اولین تغییر
واحد و محافظه‌کارانه انتخاب شد؛ p1 هم‌زمان اجرا نمی‌شود. ترتیب preregistered پس از
بازگشت سرور: ابتدا diagnostic بدون update گرادیان، سپس smoke فنی p0.75، سپس screening
کامل فقط روی calibration1 در برابر exp22. تنها در صورت حفظ FPR/selection و بهبود
IVH کوچک، outer2 یک‌بار دیده می‌شود. 50 تست ICH/loss/sampler/promotion پاس شدند و
هیچ آموزش جدیدی تا این نقطه انجام نشده است.

### ۱۳.۲۲ گیت اندازهٔ ضایعه: جلوگیری از پنهان‌شدن شکست IVH کوچک

تحلیل بخش ۱۳.۲۱ نشان داد معیارهای کلی به‌تنهایی برای داوری آزمایش p0.75 کافی
نیستند: outer2 نسبت نامتعارفی از IVHهای کم‌حجم دارد و یک بهبود Dice سراسری می‌تواند
هم‌زمان با افت ضایعات کوچک رخ دهد. بنابراین پیش از مصرف GPU، ارزیابی segmentation
به سه stratum ازپیش‌تعیین‌شده برای هر زیرنوع گسترش یافت:

- کوچک: `0 < volume ≤ 2mL`؛
- متوسط: `2 < volume ≤ 10mL`؛
- بزرگ: `volume > 10mL`.

برای هر stratum تعداد مطالعهٔ مثبت، Dice روی پیکسل‌های spatial-known، حساسیت تشخیص
حجم در آستانهٔ `0.1mL`، MAE، median absolute error و median relative absolute error
محاسبه می‌شود. مرز 2mL پس از دیدن prediction جدید انتخاب نشده؛ مستقیماً از audit
label-only و فرضیهٔ preregistered بخش ۱۳.۲۱ آمده است. این معیارها در JSON/CSV خروجی،
MLflow و فقط پیام‌های مهم Telegram ثبت می‌شوند. پیام checkpoint اکنون تعداد، Dice و
حساسیت IVH کوچک را همراه تحلیل کوتاه فارسی نشان می‌دهد.

این گیت عمداً وارد `checkpoint_selection_score` نشده است. calibration1 تعداد کمی
IVH کوچک دارد و بهینه‌کردن مستقیم checkpoint بر این زیرگروه می‌تواند بیش‌برازش شدید
ایجاد کند. انتخاب checkpoint همچنان با راهبرد preregistered
`selection - 0.10 × normal-FPR` انجام می‌شود؛ metricهای size-stratified نقش diagnostic
و شرط عدم‌پسرفت دارند. برای promotion آزمایش p0.75 باید هم‌زمان:

1. امتیاز و FPR calibration در برابر exp22 حفظ شوند؛
2. IVH کوچک از نظر Dice یا sensitivity بهبود نشان دهد و معیار دیگر افت شدید نکند؛
3. MAE حجم کل و IVH بدتر نشود؛
4. فقط پس از عبور از این گیت، outer2 یک‌بار و بدون تنظیم مجدد دیده شود.

پیاده‌سازی backward-compatible است و selection تاریخی را تغییر نمی‌دهد. 57 تست
مرتبط ICH، loss، sampler، fold-shift، gradient diagnostic، promotion و Telegram
پاس شدند. اجرای GPU هنوز آغاز نشده، زیرا executable مستقیم `vastai` در PATH محیط
Codex دیده نمی‌شود و بعد از شکست SSH قبلی، guardrail افزونه خواندن
`vastai logs 49378919` را پیش از هر تلاش SSH الزامی می‌کند؛ هیچ مسیر جایگزین، stop،
reboot یا destroy استفاده نشده است.

### ۱۳.۲۳ نتیجهٔ diagnostic گرادیان: رد PCGrad سراسری در مرحلهٔ فعلی

پس از بازیابی دسترسی، علت شکست SSH به‌صورت قطعی مشخص شد: sshd سرور سالم بود و
fingerprint پذیرفته‌شده دقیقاً با کلید RSA محلی تطبیق داشت؛ تلاش قبلی کلید دیگری را
ارائه می‌کرد. اینستنس `49378919` در وضعیت `actual/intended/cur/next=running`، با RTX
3090 بیکار، 24GB VRAM، 25GB فضای آزاد و manifest دقیقاً با SHA256 مورد انتظار
`d63fc4f...391eae` تأیید شد. سرور با fast-forward به commitهای فعلی رسید و همان
57 تست ICH را پاس کرد.

diagnostic بدون optimizer step روی checkpoint دقیق exp22 با SHA256
`c63e609c...8191d`، تعداد 24 batch واقعی train، batch size=16، BF16 و 299 tensor
مشترک encoder اجرا شد. artifact بازتولیدپذیر هم روی سرور و هم محلی ذخیره شد:

`reports/ich_experiments/2p5d_segmentation/diagnostics/exp22_train_gradconflict_24b_v2`

در 384 ردیف نمونه‌گیری‌شده، 370 ردیف spatial-known بود و 192 مورد از آن‌ها
(`51.89%`) ماسک خالی داشتند. خلاصهٔ کل:

| جفت گرادیان روی encoder | cosine میانگین | میانه | کسر منفی | میانهٔ نسبت norm به segmentation |
|---|---:|---:|---:|---:|
| segmentation ↔ classification کل | +0.0392 | +0.0305 | 33.3% | 34.47% |
| segmentation ↔ Any-ICH | +0.0512 | +0.0399 | 25.0% | 14.26% |
| segmentation ↔ IVH | +0.0105 | -0.0006 | 50.0% | 9.40% |
| segmentation ↔ SAH | -0.0160 | -0.0253 | 75.0% | 7.02% |

تحلیل شرطی اهمیت بیشتری دارد. IVH در 23/24 batch و 66 ردیف مثبت حاضر بود؛ در همین
batchهای مثبت cosine میانگین `+0.0102`، میانه `-0.0037` و کسر منفی `52.2%` بود.
پس conflict IVH جهت پایدار ندارد و افت outer2 را بهتر است با shift ضایعات کوچک و
sampling توضیح دهیم. SAH در 16 batch و 25 ردیف مثبت حاضر بود؛ حتی در همین subset،
cosine میانگین `-0.0105`، میانه `-0.0253` و کسر منفی `75%` باقی ماند. بنابراین
یک conflict ضعیف ولی تکرارشونده برای SAH وجود دارد، اما norm آن در میانه فقط
`5.73%` گرادیان segmentation است.

hard-empty علت conflict نیست. نسبت norm گرادیان hard-empty به segmentation پایه در
میانه فقط `0.265%` بود؛ در batchهای SAH-positive نیز cosine آن با SAH میانگین
`+0.0286`، میانه `+0.0439` و فقط 25% منفی بود. در نتیجه اعمال PCGrad روی تمام
classification در این مرحله می‌تواند یک سیگنال کلی عمدتاً هم‌جهت را به‌خاطر conflict
کوچک SAH بی‌دلیل دست‌کاری کند. تصمیم preregistered:

- PCGrad سراسری فعلاً اجرا نمی‌شود؛
- sampler مطالعه‌محور p0.75 همچنان تغییر واحد بعدی است؛
- اگر SAH پس از آن همچنان افت کرد، projection فقط برای شاخهٔ SAH یا decoder/head
  جداگانه به‌عنوان آزمایش ثانویه بررسی می‌شود؛
- نتیجه و تحلیل فشردهٔ فارسی با پیشوند ثابت مسابقه در Telegram ارسال شد.

در بازبینی مسیر اجرا یک guardrail دیگر نیز اضافه شد: runهای دارای
`max_train_steps` اکنون smoke فنی محسوب می‌شوند و هرگز outer fold را inference
نمی‌کنند. smoke فقط forward/backward، calibration، checkpoint round-trip، MLflow و
Telegram را می‌آزماید؛ outer صرفاً در اجرای کامل و یک‌بار خوانده خواهد شد. این
guardrail همراه کل مسیر ICH با 59 تست مرتبط پاس شد.

### ۱۳.۲۴ نتیجهٔ p0.75: تأیید مکانیسم کوچک‌ضایعه، رد مدل و اصلاح پروتکل outer

پس از diagnostic، گیت فنی چهار-step با خروجی
`exp23_smoke_studybalanced075_hardempty001_fprselect_p1_audited_v3_f2` کامل شد.
مدت اجرا `28.3s` و peak VRAM برابر `3.865GB` بود؛ sampler جرم مثبت `0.4861`،
بیشینهٔ وزن `18.88` و ESS برابر `2791.5` داشت. فایل `run_summary.json` به‌طور صریح
`outer_evaluation_performed=false` ثبت کرد و هیچ artifact مربوط به outer ساخته نشد.
پس forward/backward، BF16، checkpoint round-trip، MLflow و Telegram سالم بودند،
ولی مطابق تعریف smoke هیچ ادعای کیفیتی از اعداد آن استخراج نشد.

آموزش کامل p0.75 با MLflow run
`ecba13a8b452404f8ff85d6d0e3d2cb9` در `687.1s`، peak VRAM=`3.868GB` و best
epoch=`8` تمام شد. checkpoint آن SHA256
`f3661441...92392` دارد. مقایسهٔ calibration1 با exp22:

| معیار calibration1 | exp22 / p0 | exp23 / p0.75 | delta |
|---|---:|---:|---:|
| selection | 0.66143 | **0.66406** | +0.00263 |
| Dice کل foreground | 0.45576 | **0.46766** | +0.01189 |
| Any-ICH AUC | **0.92025** | 0.91398 | -0.00627 |
| macro subtype AUC | **0.89789** | 0.88434 | **-0.01354** |
| normal FPR | 0.19444 | 0.19444 | 0.00000 |
| total-volume MAE | 10.2678 | **9.9683** | -0.2994mL |
| IVH Dice | 0.68365 | **0.70761** | +0.02397 |
| IVH AUC | **0.98387** | 0.96129 | **-0.02258** |
| IVH MAE | **0.22937** | 0.33087 | **+0.10150mL** |
| IVH کوچک، n=1: Dice / sensitivity / MAE | 0.1113 / 0 / 1.1297 | **0.1940 / 1 / 1.0446** | بهتر |

نتیجهٔ مکانیکی دوگانه است: study balancing واقعاً exposure ضایعهٔ کوچک را به یک
سیگنال قابل‌اندازه‌گیری تبدیل کرد، اما وزن p0.75 بیش از حد تهاجمی بود. سه گیت
ازپیش‌تعیین‌شده روی calibration شکست خوردند: macro-AUC بیش از `0.01` افت کرد، IVH-AUC
بیش از `0.01` افت کرد و IVH-MAE بدتر شد. ابزار نسخه‌بندی‌شدهٔ
`scripts/evaluate_ich_sampler_screen.py` بنابراین تصمیم رسمی
`reject_before_outer` را ثبت کرد.

outer2 با وجود این شکست مشاهده شده بود. این انحراف از ترتیب preregistered باید
شفاف ثبت شود: نتیجهٔ outer2 دیگر تأیید مستقل نیست و صرفاً تحلیل اکتشافی مکانیسم است.
روی آن fold، selection و Dice به‌ترتیب `+0.00402` و `+0.00591` بهتر شدند، اما normal
FPR از `0.19444` به `0.22222`، total-volume MAE به اندازهٔ `+0.6041mL`، IVH-MAE
به اندازهٔ `+0.0560mL` و IVH-AUC به اندازهٔ `-0.03333` بدتر شدند. F1 حضور نیز
`-0.0125` افت کرد. درون IVH:

| stratum outer2 | n | Dice exp22 → p0.75 | MAE exp22 → p0.75 | تفسیر |
|---|---:|---:|---:|---|
| کوچک، ≤2mL | 4 | 0.0000 → **0.1871** | 0.9088 → **0.7410** | مکانیسم هدف بهتر شد؛ sensitivity هر دو 0.25 |
| متوسط، 2–10mL | 7 | **0.5286** → 0.4583 | **2.5159** → 3.4332 | گروه غالب آسیب جدی دید |
| بزرگ، >10mL | 1 | 0.6373 → **0.6895** | 4.8409 → **0.5359** | بهتر، ولی n=1 |

بنابراین بهبود ضایعات کوچک واقعی اما برای promotion ناکافی است؛ هزینهٔ آن روی گروه
متوسط، FPR و خطای حجم بیشتر از منفعت بوده است. checkpoint ردشده به پوشهٔ مدل‌های
محلی منتقل نشد. فقط config، history، predictionها، summaryها و تصمیم رسمی با checksum
یکسان سرور/محلی در مسیر زیر نگهداری شدند:

`reports/ich_experiments/2p5d_segmentation/exp23_studybalanced075_hardempty001_fprselect_p1_audited_v3_f2`

برای جلوگیری از تکرار انحراف پروتکل، commit `0a5908d` گزینهٔ صریح
`--skip-outer-evaluation` را افزود. این حالت smoke نیست: آموزش کامل و انتخاب checkpoint
روی calibration انجام می‌شود، اما outer حتی inference هم نمی‌شود. پیام Telegram نیز
آن را با نوع `calibration_screen` و تحلیل فارسی متمایز گزارش می‌کند. 26 تست مستقیم
گیت، training و promotion روی سرور پاس شدند.

آزمایش بعدی p0.50 است، نه p1.00. شبیه‌سازی پیش از آموزش نشان داده بیشینهٔ وزن آن
`11.28` در برابر `18.88` برای p0.75 است و سهم IVH کوچک را از `1.35%` به `2.35%`
می‌رساند؛ پس همان جهت مفید را با variance کمتر امتحان می‌کند. این آزمایش با همان
train/calibration، seed، loss، معماری و checkpoint selection و فقط با تغییر power
اجرا می‌شود. ابتدا تنها calibration1 دیده خواهد شد. شرط ادامه، عبور هم‌زمان همهٔ
گیت‌های selection، FPR، Dice، Any-AUC، macro-AUC، F1، total MAE، IVH Dice/AUC/MAE و
سیگنال IVH کوچک است. در صورت شکست، شاخهٔ generic study balancing بسته می‌شود؛ در صورت
عبور، به‌جای تنظیم دوباره روی outer2 مستقیماً پنج-fold OOF patient-disjoint اجرا خواهد
شد. p1 بدون شواهد تازه اجرا نمی‌شود.

### ۱۳.۲۵ نتیجهٔ p0.50 calibration-only: رد generic study balancing

پیش از اجرا، کد سرور به commitهای `0a5908d` و سپس `e824d4b` رسید. گارد جدید
`--skip-outer-evaluation`، ارزیاب sampler، قالب Telegram و سیاست عدم‌انتشار CSVهای
ردیفی در Git فعال بودند. 54 تست ICH و 6 تست Telegram روی سرور پاس شدند و RTX 3090
با 17MiB VRAM مصرفی بیکار بود. تلاش اول launcher پیش از import مدل با
`ModuleNotFoundError: src` متوقف شد؛ هیچ output/checkpoint/MLflow run یا مصرف GPU
ایجاد نکرد. همان فرمان با تنها اصلاح فنی `PYTHONPATH=.` و بدون تغییر علمی اجرا شد.

exp24 با run id=`511ae883ecde4119a2d9b56d34a043b7`، مدت `540.49s`، peak
VRAM=`3.868GB` و early stopping پس از epoch8 کامل شد. بهترین checkpoint epoch5 و
SHA256 آن `6ac7d59d...e85f98` است. sampler دقیقاً جرم مثبت p0/p0.75 را حفظ کرد،
بیشینهٔ وزن=`11.2813` و ESS=`3045.67` داشت. `run_summary.json` صریحاً
`run_kind=calibration_screen`، `outer_evaluation_performed=false` و
`outer_summary=null` ثبت کرد؛ نبود `outer_summary.json` و predictionهای outer نیز
با gate فایل‌سیستم تأیید شد.

مقایسهٔ بهترین calibration checkpoint با exp22 هم‌split:

| معیار calibration1 | exp22 / p0 | exp24 / p0.50 | delta |
|---|---:|---:|---:|
| selection | **0.66143** | 0.64872 | **-0.01271** |
| Dice کل foreground | **0.45576** | 0.44427 | **-0.01149** |
| Any-ICH AUC | **0.92025** | 0.91756 | -0.00269 |
| macro subtype AUC | **0.89789** | 0.86066 | **-0.03723** |
| presence F1 | **0.88235** | 0.86957 | **-0.01279** |
| normal FPR | **0.19444** | 0.22222 | **+0.02778** |
| total-volume MAE | 10.2678 | **9.6595** | -0.6082mL |
| IVH Dice | **0.68365** | 0.63757 | **-0.04607** |
| IVH AUC | **0.98387** | 0.97742 | -0.00645 |
| IVH MAE | **0.22937** | 0.65179 | **+0.42242mL** |
| IVH کوچک، n=1: Dice / sensitivity / MAE | 0.1113 / 0 / 1.1297 | **0.1603 / 1 / 1.0755** | بهتر |

گیت رسمی فقط Any-AUC، IVH-AUC، total MAE و small-IVH signal را پاس کرد؛ هفت گیت
selection، Dice، macro-AUC، FPR، F1، IVH-Dice و IVH-MAE شکست خوردند. تصمیم artifact
`reject_before_outer` است و protocol note تأیید می‌کند outer استفاده نشده. checkpoint
ردشده به پوشهٔ model candidate محلی منتقل نشد. پنج artifact غیرردیفی با checksum
دقیقاً یکسان سرور/محلی ذخیره شدند؛ predictionهای study/slice به دلیل امکان linkage
فقط در MLflow و storage محلی/سرور باقی ماندند و `.gitignore` مانع انتشار تصادفی
آن‌ها شد.

نتیجهٔ دو power مستقل p0.75 و p0.50 یکسان است: study balancing سیگنال IVH کوچک را
بهبود می‌دهد، اما چون وزن sampling همهٔ subtypeهای حاضر در مطالعه را با قاعدهٔ max
تغییر می‌دهد، distribution shift و variance کافی برای آسیب به FPR، macro-AUC و IVH
متوسط ایجاد می‌کند. کاهش power شدت مشکل را کم نکرد و p1.00 از نظر علمی توجیه ندارد؛
شاخهٔ generic study balancing بسته شد.

جهت بعدی sampling اصلی p0 را بازمی‌گرداند و فقط objective فضایی IVH را هدف می‌گیرد.
مقالهٔ ICI در MIDL/PMLR، loss سراسری پیکسلی را با instance-wise و center-of-instance
ترکیب می‌کند؛ مؤلفهٔ instance برای کاهش false-negative ضایعات کوچک و مؤلفهٔ center
برای حفظ مرکز ضایعه و مهار instanceهای کاذب طراحی شده است. مقاله روی دادهٔ ATLAS
بهبود Dice و lesion-wise detection گزارش کرده، اما وابستگی به وزن‌ها و هزینهٔ CCA را
نیز صریحاً محدودیت می‌داند:

- `https://proceedings.mlr.press/v227/rachmadi24a.html`؛
- `https://github.com/BrainImageAnalysis/ICI-loss`.

به‌جای واردکردن کور کامل ICI چندکلاسه، ابتدا EDA component-level روی maskهای IVH
train انجام می‌شود تا تعداد component، اندازه و شعاع مرکز مناسب مشخص شود. سپس یک
auxiliary loss فقط-IVH با weight کوچک و preregistered، sampler p0 و همان hard-pixel
baseline غربال خواهد شد. شرط ادامه، بهبود IVH کوچک/کلی بدون پس‌دادن FPR، F1،
macro-AUC و total MAE است؛ outer در این غربال خوانده نمی‌شود.

### ۱۳.۲۶ loss مرکز IVH: گیت فنی، دوز ۰٫۱۰ و رد پیش از outer

EDA component-level روی ماسک‌های IVH نشان داد supervision پیکسلی به‌طور طبیعی
تحت سلطهٔ componentهای بزرگ است. به‌جای واردکردن کامل ICI loss، یک auxiliary loss
کم‌هزینه پیاده شد که برای هر component متصل IVH یک مربع مرکزی ۱۱×۱۱ می‌سازد و
میانگین `-log(p_IVH)` را فقط روی آن نقاط اضافه می‌کند. این تغییر فقط در صورت
`ivh_center_loss_weight > 0` فعال است و حالت صفر رفتار baseline را دقیقاً حفظ می‌کند.

در smoke اولیه یک نقص اجرایی کشف شد: `max_train_steps=4` در هر epoch چهار step اجرا
می‌کرد، نه چهار step برای کل run. guardrail اصلاح شد تا smoke دقیقاً یک partial epoch،
یک calibration pass و بدون هیچ outer inference اجرا کند. پس از commit `2112545`،
مجموع ۶۷ تست مرتبط پاس شد. smoke اصلاح‌شده در `18.26s` با peak VRAM=`3.872GB`
تمام شد؛ loss مرکز IVH متناهی (`1.4116`) و `outer_evaluation_performed=false` بود.

آزمایش کامل وزن `0.10` با run id=`319a02b6c57141e5bcf4daf34839eef0`، best epoch=10،
مدت `677.58s` و peak VRAM=`3.875GB` تمام شد. مقایسهٔ calibration هم‌split با exp22:

| معیار | delta کاندیدا نسبت به exp22 |
|---|---:|
| selection | -0.00840 |
| Dice کل | -0.00784 |
| Any-ICH AUC | -0.00941 |
| macro subtype AUC | -0.00843 |
| normal FPR | **+0.02778** |
| presence F1 | **-0.01279** |
| total-volume MAE | **+0.28018mL** |
| IVH Dice | -0.00258 |
| IVH MAE | **+0.17745mL** |

در تنها نمونهٔ IVH کوچک، Dice به‌اندازهٔ `+0.1281` و sensitivity از صفر به یک بهتر
شد، اما این منفعت کوچک به قیمت افت کلی و حجم تمام شد. تصمیم رسمی
`reject_before_outer` ثبت شد؛ checkpoint به پوشهٔ مدل‌های محلی منتقل نشد.

### ۱۳.۲۷ دوز ۰٫۰۳ و فرضیهٔ channel-safe hybrid

برای تفکیک «ایدهٔ غلط» از «دوز بیش‌ازحد»، همان آزمایش با تنها تغییر وزن به `0.03`
اجرا شد. exp26 با run id=`e82eb6311dad44699b136acbee0564db`، best epoch=8، مدت
`690.81s` و peak VRAM=`3.875GB` کامل شد. در calibration1، selection=`+0.01750`،
Dice=`+0.02710`، Any-AUC=`+0.00538`، macro-AUC=`+0.00653` و total MAE=`-0.47167mL`
بهتر شدند و FPR/F1 بدون تغییر ماندند. بااین‌حال IVH Dice=`-0.02309`، IVH
MAE=`+0.10482mL` و معیارهای IVH کوچک بدتر شدند. پس مدل به‌عنوان standalone رد شد.

این الگو نشان داد auxiliary IVH loss به‌طور غیرمستقیم بعضی کانال‌های غیر-IVH را
بهبود داده است. ابزار hybrid کانال‌به‌کانال با کنترل alignment، SHA manifest و
ممنوعیت انتشار predictionهای ردیفی در Git ساخته شد. hybrid اولیه که تمام غیر-IVHها
را از exp26 می‌گرفت به‌علت افت پنهان SAH رد شد. نگاشت محافظه‌کارانهٔ exp28 فقط
IPH/SDH/EDH را از exp26 و IVH/SAH را دقیقاً از exp22 گرفت؛ score مربوط به Any-ICH
نیز از exp26 بود. روی calibration1 همهٔ گیت‌ها عبور کردند:

| معیار exp28 نسبت به exp22 | delta |
|---|---:|
| selection | **+0.02227** |
| Dice کل | **+0.03409** |
| Any-ICH AUC | +0.00538 |
| macro subtype AUC | +0.01273 |
| normal FPR | **-0.02778** |
| presence F1 | +0.01317 |
| total-volume MAE | **-0.88862mL** |

پس از قفل‌شدن نگاشت فقط از روی calibration، outer2 یک‌بار ارزیابی شد. exp29 روی آن
fold selection=`+0.00474`، Dice=`+0.00736`، Any-AUC=`+0.00179` و F1=`+0.03598`
بهتر داشت، اما gain انتخاب اندکی کمتر از حد `0.005` بود و total MAE=`+0.20176mL`
بدتر شد. گیت strict آن را تأیید نکرد. نتیجه نه شکست قطعی فرضیه است و نه مجوز promotion:
یک fold برای یک hybrid پرنوسان کافی نیست. نگاشت پس از دیدن outer تغییر نکرد.

### ۱۳.۲۸ پروتکل cross-fitted پنج‌fold و نتیجهٔ fold صفر

برای حذف cherry-picking، پیش از foldهای باقی‌مانده یک قاعدهٔ انتخاب قطعی commit شد:
برای هر subtype، کاندیدا فقط وقتی انتخاب می‌شود که Dice و AUC کمتر، MAE بیشتر نباشد
و حداقل یک معیار واقعاً بهتر باشد؛ score Any فقط با AUC بهتر عوض می‌شود. انتخاب برای
هر outer fold صرفاً روی calibration متناظر انجام می‌شود و سپس outer حداکثر یک‌بار
خوانده می‌شود. کانال فاقد metric معتبر یا فاقد نمونهٔ مثبت الزاماً روی reference
می‌ماند. بعد از ساخت hybrid نیز گیت‌های کلی FPR/F1/MAE باید پاس شوند؛ بهبود محلی یک
کانال به‌تنهایی کافی نیست.

exp30 برای `(outer=0, calibration=1)` با run id=`205f7f402163414fae76720adb0840e7`
در `659.55s` و peak VRAM=`3.877GB` تمام شد. بهترین checkpoint epoch7،
selection=`0.63745` و Dice=`0.42212` داشت. قاعدهٔ ثابت فقط EDH را از کاندیدا انتخاب
کرد: Dice=`+0.00858`، AUC=`+0.01437` و MAE=`-0.16431mL`. با وجود این، hybrid نهایی
FPR را از `0.19444` به `0.22222` رساند، F1 را `-0.01279` و total MAE را
`+0.38089mL` بدتر کرد. بنابراین گیت رسمی `reject_before_outer` ثبت شد و outer0
عمداً خوانده نشد.

نکتهٔ محاسباتی: جفت‌های `(outer0, cal1)` و `(outer1, cal0)` هر دو روی foldهای 2/3/4
آموزش می‌بینند و با seed یکسان trajectory آموزشی برابر دارند؛ تفاوت در epoch منتخب
از calibration متفاوت است. در دورهای بعد نگه‌داری snapshotهای epoch می‌تواند اجرای
تکراری را حذف کند، بدون آن‌که استقلال انتخاب checkpoint قربانی شود. exp32 برای
calibration0 لازم بود، زیرا run قبلی فقط بهترین checkpoint calibration1 را حفظ کرده
و snapshot همهٔ epochها موجود نبود.

سه split باقی‌مانده نیز با همان rule و بدون مشاهدهٔ outer داوری شدند:

| outer / calibration | کانال‌های انتخاب‌شده | نقاط مثبت hybrid | علت رد |
|---|---|---|---|
| 1 / 0 | IVH, IPH, SAH + Any | selection `+0.00836`، Dice `+0.01337`، MAE کل `-1.4025mL` | FPR `+0.08108` و F1 `-0.02165` |
| 3 / 1 | IVH | IVH Dice `+0.10607`، AUC `+0.04839`، MAE `-0.15278mL` | MAE کل `+0.59531mL` |
| 4 / 1 | IVH, IPH, SDH, EDH | selection `+0.01674`، Dice `+0.02647`، FPR `-0.02778` | MAE کل `+0.04690mL` |

در calibration0 هیچ EDH مثبتی وجود نداشت. selector و gate اصلاح شدند تا metricهای
`null` را به‌عنوان «عدم پشتیبانی» ثبت کنند و فقط در صورت حفظ دقیق reference آن کانال
را neutral/pass بدانند؛ نبود نمونه هرگز به‌عنوان بهبود تفسیر نمی‌شود. outerهای 0، 1،
3 و 4 به‌دلیل شکست گیت خوانده نشدند. برای OOF نهایی این چهار fold دقیقاً reference
ماندند و فقط fold2 از hybridی استفاده کرد که پیش‌تر calibration متناظر را پاس کرده
بود. این fallback بخشی از قانون calibration-only است، نه تصمیمی پس از دیدن outer.

مقایسهٔ نهایی روی ۳۳۸ مطالعه، ۳۲۰ بیمار و ۷۴۲۸ برش با ۵۰۰۰ bootstrap بیمارمحور:

| معیار OOF | reference | cross-fit fallback | delta | P(بهتر) / CI95 delta |
|---|---:|---:|---:|---|
| selection | 0.63112 | 0.63366 | +0.00254 | 0.7936 / [-0.00225, +0.00835] |
| Dice کل | 0.41393 | 0.41791 | +0.00399 | 0.7114 / [-0.00277, +0.01393] |
| Any-ICH AUC | 0.93578 | 0.93778 | +0.00200 | 0.7268 / [-0.00468, +0.00883] |
| macro subtype AUC | 0.81818 | 0.81651 | **-0.00168** | 0.2658 / [-0.00692, +0.00299] |
| presence F1 | 0.87349 | 0.88024 | +0.00675 | 0.9312 / [0, +0.01742] |
| normal FPR | 0.16111 | 0.16111 | 0 | 0.5000 / [0, 0] |
| total-volume MAE | **9.00221** | 9.04220 | **+0.03999mL** | 0.2940 / [-0.09192, +0.18514] |

بهبود F1 تکرارپذیرتر از بقیه است، اما endpoint اصلی selection CI شامل صفر دارد،
macro-AUC و MAE جهت نامطلوب دارند و تنها یک fold از پنج fold از مدل جدید استفاده
می‌کند. نتیجهٔ رسمی: loss مرکز IVH و channel-wise hybrid برای promotion رد و این شاخه
بسته شد. هیچ checkpoint ردشده‌ای به مسیر مدل‌های پذیرفته‌شده منتقل نمی‌شود؛ baseline
hard-pixel/FPR-select همچنان incumbent است. artifact نهایی در مسیر
`reports/ich_experiments/2p5d_segmentation/oof_crossfit_ivhcenter003_channelsafe_fallback_v1`
ثبت شده است.

### ۱۳.۲۹ ممیزی supervision و اصلاح clean-negativeها در schema v4

ممیزی خط‌به‌خط سازندهٔ dataset یک خطای معنایی مهم، ولی بدون نشت، را آشکار کرد.
مطالعات strict clean-negative که JSON ضایعه نداشتند به‌درستی تصویر و mask صفر
داشتند، اما به‌علت `metadata_missing` بودن همان برش‌ها، supervision طبقه‌بندی و
segmentation نیز خاموش می‌شد. در نتیجه ۱۴۵ برش واقعاً منفی از ۱۷ مطالعه در loss
فضایی و ارزیابی Dice شرکت نمی‌کردند. commit `a9ad0eb` فقط برای clean-negativeهای
سخت‌گیرانه supervision را فعال کرد و schema را از ۳ به ۴ رساند؛ برش‌های partial
واقعاً نامعلوم همچنان masked باقی ماندند.

نسخه‌های جدید در مسیرهای جداگانه ساخته شدند و نسخه‌های قدیمی دست‌نخورده ماندند:

- `Data/processed/ich_v2/BrainICHPartial_v4`؛
- `Data/processed/ich_2p5d_v4` با SHA256 manifest برابر
  `e54d94be...70198`.

اعتبارسنجی v4 روی ۳۳۸ مطالعه، ۷۶۸۳ برش و ۳۲۰ بیمار نشان داد ۷۶۵۳ برش
classification-known و ۷۵۷۳ برش spatial-known هستند. هر ۲۴۷۷ برش clean-negative
اکنون supervision معتبر دارند؛ فقط ۳۰ برش classification و ۱۱۰ برش spatial از
مطالعات partial نامعلوم مانده‌اند. diff بازگشتی byte-for-byte تأیید کرد image و mask
نسخهٔ 2.5D قدیم و جدید کاملاً یکسان‌اند؛ تنها semantics supervision تغییر کرده است.

برای جلوگیری از اشتباه گرفتن تغییر معیار با تغییر مدل، predictionهای incumbent بدون
اجرای inference دوباره و فقط با manifest v4 بازامتیازدهی شدند. در OOF پنج‌fold، ۱۴۵
برش منفی وارد Dice شدند و Dice از `0.43506` به `0.43074` و selection از `0.64402`
به `0.64164` رسید؛ Any-AUC=`0.93453`، macro-AUC=`0.82916`، FPR=`0.17222`،
F1=`0.86826` و MAE=`8.71892mL` دقیقاً ثابت ماندند. این افت کوچک پس‌رفت مدل نیست؛
برآورد سخت‌گیرانه‌تر baseline است. روی calibration متناظر exp22 نیز ۴۸ برش از ۵
مطالعه اضافه و baseline معتبر v4 به شکل زیر قفل شد:

| معیار calibration1 | baseline v4 |
|---|---:|
| selection / checkpoint score | 0.65995 / 0.64051 |
| Dice کل foreground | 0.45308 |
| Any-ICH / macro subtype AUC | 0.92025 / 0.89789 |
| normal FPR / presence F1 | 0.19444 / 0.88235 |
| total-volume MAE | 10.26777mL |

artifactهای خلاصهٔ قابل‌انتشار در
`diagnostics/incumbent_oof_schema_v4_rescore_v1` و
`diagnostics/exp22_calibration_schema_v4_rescore_v1` نگهداری شدند. predictionهای
ردیفی به‌علت قابلیت linkage وارد Git نمی‌شوند.

### ۱۳.۳۰ آموزش از صفر روی v4 و warm-start ایمن

exp39 همان recipe موفق exp22 را از ImageNet و روی v4 آموزش داد تا مشخص شود آیا
supervision اصلاح‌شده به‌تنهایی trajectory بهتری می‌سازد. run
`4462b221f00c41169f4d9fd079b97bbc` در `683.75s` و best epoch=8 تمام شد، اما نسبت
به baseline v4، selection=`-0.01292`، Dice=`-0.00274`، Any-AUC=`-0.02957`،
macro-AUC=`-0.01696` و MAE=`+1.09124mL` داشت. FPR به‌اندازهٔ `-0.02778` و F1
به‌اندازهٔ `+0.01317` بهتر شدند، ولی trade-off کلی نامطلوب بود؛ بنابراین پیش از outer
رد شد و checkpoint آن منتقل نشد.

برای اینکه fine-tuning هیچ‌گاه baseline را تصادفاً overwrite نکند، commit `8b71e3b`
گزینهٔ leakage-safe `--initial-checkpoint` را افزود. معماری، encoder، split، کانال‌ها
و labelها پیش از load اعتبارسنجی می‌شوند؛ checkpoint اولیه و SHA آن در MLflow ثبت و
قبل از هر gradient به‌عنوان epoch صفر ارزیابی و ذخیره می‌شود. تنها epochی جای آن را
می‌گیرد که checkpoint score واقعاً بهتر باشد.

exp40 با چهار step صحت مکانیسم را ثابت کرد: epoch صفر تمام معیارهای baseline v4 را
دقیقاً بازتولید کرد و stepهای smoke آن را بدتر کردند. exp41 سپس fine-tuning کامل با
LR=`5e-6` را آزمود؛ پس از دو epoch patience فعال شد و best epoch همان صفر باقی ماند
(run `d239ee8965af4467925a8cfeb7d54a98`). نتیجهٔ علّی این است که fine-tuning عمومی
روی supervision جدید، بدون objective هدفمند، ارزش ندارد. برای کاهش نویز عملیاتی،
commit `30cd084` اعلان‌های Telegram مربوط به شروع/موفقیت smoke و checkpointهای
میانی calibration را حذف کرد؛ failure smoke و رخدادهای مهم همچنان گزارش می‌شوند.

### ۱۳.۳۱ غربال hard-negative: سیگنال یک-step، رد آموزش کامل

ممیزی OOF incumbent روی ۱۸۰ مطالعهٔ normal، ۳۱ false-positive در سطح مطالعه و ۱۳۳
برش دارای foreground کاذب پیدا کرد. sampler جدید فقط hard-negativeهای foldهای مجاز
train را می‌پذیرد و patient/study/source-fold را کنترل می‌کند؛ بنابراین از calibration
یا outer نشت نمی‌کند. با multiplier=2 روی split `(outer=2, calibration=1)` فقط ۵۰
برش از ۱۴ مطالعه match شد. جرم احتمال hard-negative برابر `1.3318%` و ESS برابر
`3254.18` در برابر `3263.47` baseline بود؛ جرم positive نیز دقیقاً `0.48300` ماند.
پس مداخله از نظر توزیعی کوچک و کنترل‌شده بود.

exp42 یک smoke تک-step از checkpoint exp22 بود. checkpoint score فقط
`+0.000057` و macro-AUC=`+0.00153` بهتر شد، اما Dice=`-0.00031` و MAE=
`+0.00459mL` بدتر شدند. این فقط یک سیگنال جهت بود، نه ادعای کیفیت؛ به همین دلیل
exp43 به‌عنوان calibration screen کامل با LR=`5e-6`، پنج epoch، patience=2 و محافظ
epoch صفر اجرا شد.

در exp43، epoch1 checkpoint score را از `0.64051` به `0.63480` و Dice را از
`0.45308` به `0.44305` کاهش داد؛ MAE نیز از `10.26777` به `10.95684mL` رسید.
epoch2 اندکی برگشت، اما checkpoint score=`0.63751`، Dice=`0.44619` و MAE=
`10.72652mL` هنوز بدتر بودند. run `3419743324f743c1af41f25e8728223e` پس از
`160.22s` متوقف و best epoch=0 باقی ماند. بنابراین oversampling عمومی hard-negative
برای promotion رد شد، outer خوانده نشد و checkpoint جدید به مسیر مدل‌های محلی منتقل
نشد.

این نتیجه به معنی بی‌ارزش بودن hard-negativeها نیست؛ نشان می‌دهد تزریق آن‌ها به loss
مشترک segmentation، حتی با جرم کم، مرز subtypeها را جابه‌جا می‌کند. مسیر بعدی باید
FPR را در لایهٔ تصمیم‌گیری/score study یا با objective تفکیک‌شدهٔ foreground-presence
هدف بگیرد، نه با تکرار کور multiplierهای ۱٫۵، ۳ یا ۴. هر کاندیدا ابتدا روی همان
calibration و با baseline v4 قفل‌شده غربال می‌شود و تنها پس از عبور هم‌زمان Dice،
AUC، FPR، F1 و MAE اجازهٔ یک ارزیابی outer خواهد داشت.

### ۱۳.۳۲ رد TTA افقی و طراحی adapter تقارن صفرآغاز

پیش از آموزش معماری تازه، horizontal-flip TTA به‌عنوان یک inference intervention
کم‌هزینه بررسی شد. پیاده‌سازی، probabilityهای softmax ماسک را پس از برگرداندن نمای
قرینه به مختصات اصلی و probabilityهای sigmoid auxiliary head را بدون تغییر مختصات
میانگین می‌گیرد. baseline و TTA در یک اجرا و روی calibration1 یکسان محاسبه شدند؛
outer2 نه load و نه inference شد. ۴۶ تست محلی و سرور، از جمله بازگرداندن محور فضایی،
alignment برش‌ها و گیت عدم‌پسرفت پاس شدند.

exp44 با MLflow run `6f7108f4a93043f89ff2d634a19d99f9` در `27.32s` و peak
VRAM=`1.144GB` کامل شد. نسبت به inference سادهٔ exp22 روی schema v4:

| معیار calibration1 | baseline | hflip TTA | delta |
|---|---:|---:|---:|
| selection | **0.65995** | 0.65477 | -0.00518 |
| Dice کل | **0.45308** | 0.44558 | -0.00751 |
| Any-AUC | **0.92025** | 0.91667 | -0.00358 |
| macro subtype AUC | 0.89789 | **0.89803** | +0.00014 |
| FPR / F1 | 0.19444 / 0.88235 | 0.19444 / 0.88235 | بدون تغییر |
| total-volume MAE | **10.26777mL** | 10.43205mL | +0.16428mL |

TTA روی EDH و IPH Dice اندکی بهتر و IVH کوچک را از `0.1113` به `0.1336` رساند،
اما IVH کلی، SAH و SDH افت کردند. این الگو با smoothing ساختارهای نازک سازگار است.
سه گیت checkpoint-score، Dice و MAE شکست خوردند؛ تصمیم رسمی
`reject_before_outer` است. sweep وزن‌های دلخواه برای blend قرینه توجیه ندارد.

رد averaging به معنی بی‌ارزش بودن تقارن نیست. Li et al. در یک چارچوب اختصاصی CT
خونریزی، تصویر اصلی و flipped را در ورودی concatenate کردند تا شبکه تفاوت دو نیمکره
را *یاد بگیرد*، نه اینکه خروجی‌های آن‌ها را کورکورانه صاف کند
(`https://doi.org/10.1109/JBHI.2020.3028243`). برای آزمون کم‌ریسک این مکانیسم، یک
adapter ورودی ۱×۱ با ۱۶۲ پارامتر طراحی شد: `[x, flip(x)]` را به residual نه‌کاناله
تبدیل و به `x` اضافه می‌کند. وزن adapter در صفر آغاز می‌شود؛ در نتیجه epoch0 باید
خروجی checkpoint exp22 را دقیقاً بازتولید کند. کل U-Net++/EfficientNet-B2، شامل
BatchNorm و dropout، freeze/eval می‌ماند و فقط adapter آموزش می‌بیند. این طراحی اثر
تقارن را از تغییر میلیون‌ها وزن مدل جدا می‌کند و اگر موفق نبود با هزینهٔ بسیار کم
شاخه را می‌بندد.

پروتکل exp45: ابتدا smoke چند-step برای اثبات برابری epoch0، جریان گرادیان فقط در
۱۶۲ پارامتر و round-trip checkpoint؛ سپس فقط در صورت سلامت، calibration screen با
محافظ epoch0 و بدون outer. معیار ادامه همان checkpoint score به‌همراه عدم‌پسرفت Dice،
Any/macro AUC، FPR، F1 و MAE است. تنها پس از عبور، outer یا OOF مجاز خواهد بود.

### ۱۳.۳۳ رد adapter تقارن پس از smoke و screen کامل

exp45 سلامت فنی adapter را تأیید کرد: مدل wrapperشده در epoch صفر دقیقاً همهٔ معیارهای
checkpoint قدیمی را بازتولید کرد، فقط ۱۶۲ پارامتر trainable بود و یک به‌روزرسانی
چهار-step بدون خطای gradient یا checkpoint انجام شد. این smoke با LR=`1e-4` و MLflow
run `f58202632cd74842bf93e80465e4362d` در `28.07s` تمام شد. checkpoint score فقط
`+0.000229` بهتر شد، اما Dice=`-0.001485` و MAE=`+0.18570mL` بدتر شدند. Any-AUC و
macro-AUC به‌ترتیب `+0.002688` و `+0.001597` رشد کردند و FPR/F1 ثابت ماندند. در سطح
زیرنوع، IPH و IVH بهتر ولی EDH، SAH و SDH ضعیف‌تر شدند؛ بنابراین این فقط سیگنال
جهتی مختلط بود، نه کاندیدای قابل promotion.

برای تفکیک شکست فرضیه از بزرگ بودن نرخ یادگیری، exp46 با LR=`1e-5`، پنج epoch،
patience=2، همان split و همان محافظ epoch صفر اجرا شد. run
`42775ff16d9f4be896f793081339735a` پس از `121.74s` در epoch دوم متوقف شد و best
epoch=0 باقی ماند. مسیر دو epoch آموزشی چنین بود:

| epoch | checkpoint score | Dice | Any-AUC | macro-AUC | FPR | F1 | MAE (mL) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | **0.64051** | **0.45308** | 0.92025 | **0.89789** | 0.19444 | 0.88235 | **10.26777** |
| 1 | 0.63663 | 0.44685 | **0.92115** | 0.89311 | 0.19444 | 0.88235 | 10.70858 |
| 2 | 0.63459 | 0.44489 | 0.91846 | 0.89201 | 0.19444 | 0.88235 | 10.79894 |

کاهش LR نه‌تنها افت را رفع نکرد، بلکه همان الگوی علّی را تکرار کرد: اصلاح ورودی برای
مدل freezeشده به‌سرعت Dice و برآورد حجم را خراب می‌کند، در حالی که تغییرات AUC کوچک
و ناپایدارند. بنابراین `horizontal_symmetry_adapter` رسماً
`reject_before_outer` شد؛ outer load نشد، checkpoint جدید به کتابخانهٔ مدل‌های محلی
منتقل نشد و sweep نرخ‌های یادگیری/تعداد step بیشتر توجیه ندارد. نتیجهٔ فنی adapter و
تست‌های آن در کد باقی می‌ماند تا بازتولیدپذیری حفظ شود، اما مسیر پژوهش از تقارن افقی
به فرضیه‌ای مستقل دربارهٔ استفادهٔ ساختاریافته‌تر از context بین‌برشی منتقل می‌شود.

### ۱۳.۳۴ EDA تداوم محور z و طراحی adapter پنج‌برشی

پیش از افزودن context، یک ممیزی بازتولیدپذیر روی manifest v4 انجام شد؛ ابزار
`scripts/analyze_ich_slice_context.py` طول runهای مثبت، فاصلهٔ فیزیکی برش‌ها و میزان
padding لبه را بدون بازکردن outer predictionها محاسبه می‌کند. روی ۳۳۸ مطالعه و ۷۶۸۳
برش، median فاصلهٔ مرکز برش‌ها `8.0mm`، چارک اول `5.0mm` و صدک ۹۰ برابر
`8.19mm` است. بنابراین پنج برش شعاع فیزیکی median=`16mm` و span کامل median=`32mm`
می‌دهند. فقط `6.60%` جایگاه‌های همسایه در context شعاع دو به‌علت مرز حجم edge-pad
می‌شوند.

تداوم labelها در محور z بسیار قوی است:

| subtype | برش مثبت با همسایهٔ مثبت | برش مثبت با همسایه در فاصلهٔ ۲ | run تک‌برشی | median طول run | median span |
|---|---:|---:|---:|---:|---:|
| IVH | 99.27% | 95.64% | 5.08% | 7 برش | 35.0mm |
| IPH | 99.89% | 97.67% | 0.86% | 7 برش | 40.75mm |
| SDH | 99.54% | 98.16% | 4.55% | 9 برش | 53.61mm |
| EDH | 100% | 100% | 0% | 9 برش | 71.22mm |
| SAH | 97.93% | 93.78% | 12.90% | 4 برش | 20.0mm |

این آمار با دو شاهد مستقل هم‌جهت است: CSA-Net از attention بین center slice و برش‌های
مجاور برای بازیابی وابستگی‌های سه‌بعدی در داده‌های anisotropic استفاده می‌کند
(`<https://pubmed.ncbi.nlm.nih.gov/39317055/>`) و مدل sequential کم‌برچسب ICH نیز
صراحتاً وابستگی interslice برش‌های پیوسته را برای activation map پایدار به‌کار گرفته
است (`<https://pubmed.ncbi.nlm.nih.gov/39962740/>`). بااین‌حال سهم runهای تک‌برشی SAH
هشدار می‌دهد که smoothing یا رأی‌گیری سخت بین برش‌ها مناسب نیست.

برای آزمون کم‌ریسک، `FiveSliceContextInputAdapter` طراحی شد. dataset به‌جای
`[-1,0,+1]` برش‌های `[-2,-1,0,+1,+2]` را با همان سه window تولید می‌کند. مدل exp22
هنوز دقیقاً نه کانال می‌گیرد: نه کانال میانی بدون تغییر انتخاب و یک convolution
سه‌درسهٔ صفرآغاز، از هر ۱۵ کانال residual نه‌کاناله می‌سازد. در epoch صفر residual
دقیقاً صفر است و خروجی باید bit-identical با baseline باشد. backbone، BatchNorm و
dropout freeze/eval می‌مانند و فقط ۱۲۱۵ پارامتر adapter آموزش می‌بیند.

پروتکل ازپیش‌ثبت‌شدهٔ exp48 ابتدا چهار step smoke با LR=`1e-4`، محافظ epoch صفر،
outer خاموش و کنترل دقیق ۱۵ کانال/gradient/checkpoint است. smoke فقط سلامت و جهت اولیه
را می‌سنجد. اگر جریان فنی سالم باشد و فروپاشی فوری Dice/MAE رخ ندهد، exp49 با
LR=`2e-5` و patience=2 روی همان calibration1 اجرا می‌شود. گیت promotion همان baseline
v4 قفل‌شده است: checkpoint score باید بهتر شود، Dice بیش از `0.002` افت نکند، افت
Any/macro AUC یا F1 از `0.005` بیشتر نباشد، FPR بدتر نشود و MAE حداکثر `0.1mL`
افزایش یابد. outer تنها پس از عبور هم‌زمان این قیود مجاز است.

### ۱۳.۳۵ exp48: سلامت فنی، رد residual context و لغو exp49

smoke پنج‌برشی با MLflow run `8d9bcaa6ea7e4bb081768ba342c54df6` در `30.11s`
و peak VRAM=`3.93GB` کامل شد. تعداد پارامتر trainable دقیقاً ۱۲۱۵ بود و epoch صفر
تمام معیارهای exp22/v4 را بدون اختلاف بازتولید کرد؛ بنابراین ترتیب ۱۵ کانال، انتخاب
نه کانال میانی، warm-start و checkpoint round-trip تأیید شدند. چهار step با LR=`1e-4`
این تغییرات را ساخت:

| معیار calibration1 | baseline epoch0 | پس از ۴ step | delta |
|---|---:|---:|---:|
| checkpoint score | **0.64051** | 0.63613 | -0.00438 |
| selection | **0.65995** | 0.65280 | -0.00715 |
| Dice | **0.45308** | 0.44732 | -0.00576 |
| Any-AUC | **0.92025** | 0.91219 | -0.00806 |
| macro subtype AUC | **0.89789** | 0.88744 | -0.01044 |
| FPR | 0.19444 | **0.16667** | -0.02778 |
| F1 | 0.88235 | **0.89552** | +0.01317 |
| MAE | **10.26777mL** | 10.61706mL | +0.34930mL |

بهبود FPR/F1 واقعی ولی با افت هم‌زمان segmentation، ranking و حجم خریده شده است.
IPH و IVH Dice بهتر شدند، اما EDH، SAH و مخصوصاً SDH افت کردند؛ macro-AUC نیز در
چهار subtype از پنج subtype ضعیف‌تر شد. این الگو از گیت ازپیش‌ثبت‌شده فاصلهٔ زیادی
دارد و «فروپاشی فوری Dice/MAE» محسوب می‌شود. در نتیجه exp49 با LR پایین‌تر لغو شد،
outer خوانده نشد و checkpoint adapter منتقل نمی‌شود. EDA تداوم بین‌برشی رد نشده؛
فقط روش تزریق residual در ورودی یک backbone freezeشده رد شده است. اگر context دوباره
آزموده شود باید در سطح feature یا sequence و با head مستقل انجام شود تا مرز پیکسلی
incumbent مستقیماً جابه‌جا نشود.

### ۱۳.۳۶ پیش‌ثبت exp50: pooling ترتیبی روی scoreهای OOF

در evaluator فعلی، احتمال auxiliary هر subtype و Any-ICH با `max` ساده روی تمام
برش‌های مطالعه جمع می‌شود. این score فقط AUC را می‌سازد و در Dice، حجم، FPR یا MAE
دخالت ندارد؛ بنابراین می‌توان dependence بین برش‌ها را بدون دست‌زدن به segmentation
آزمود. فرضیهٔ exp50 این است که یک maximum تک‌برشی ممکن است به artifact نویزی حساس
باشد، درحالی‌که وجود حمایت در برش مجاور با EDA بخش ۱۳.۳۴ سازگار است.

پیش از دیدن نتیجه، شش rule ثابت در کد قفل شدند: max، میانگین top-2، بیشینهٔ میانگین
جفت مجاور، بیشینهٔ geometric-mean جفت مجاور، بیشینهٔ میانگین سه‌برشی و ترکیب برابر
`0.5×max + 0.5×adjacent-pair-mean-max`. روش آخر primary است، چون هم حساسیت runهای
تک‌برشی SAH را از max حفظ می‌کند و هم به تداوم مجاور پاداش می‌دهد؛ هیچ وزن یا
thresholdی روی labelها fit نشده است.

screen فقط predictionهای OOF پنج fold incumbent و ground truth همان ۳۳۸ مطالعه را
می‌خواند؛ inference یا training جدید و مشاهدهٔ تازهٔ outer رخ نمی‌دهد. علاوه بر
مقایسهٔ primary ثابت، یک تحلیل ثانویه کاملاً cross-fitted برای هر held-out fold، روش
pooling هر label را فقط روی چهار fold دیگر انتخاب و روی fold پنجم اعمال می‌کند. این
بخش exploratory است و به‌تنهایی مجوز deploy نیست. bootstrap با ۲۰۰۰ resample در سطح
study عدم‌قطعیت delta Any-AUC و macro-AUC را گزارش می‌کند و score ردیفی persist یا
وارد Git نمی‌شود.

چون معیارهای spatial ثابت‌اند، delta امتیاز proxy دقیقاً برابر
`0.30×ΔAny-AUC + 0.15×Δmacro-AUC` است. primary فقط اگر proxy مثبت، macro-AUC بهتر،
Any-AUC حداکثر `0.002` پایین‌تر و جهت foldها فاقد شکست آشکار باشد به مرحلهٔ پیاده‌سازی
inference می‌رود. در غیر این صورت rule ثابت رد می‌شود؛ موفقیت احتمالی انتخاب
cross-fitted فقط فرضیه‌ای برای یک sequence head مستقل تولید خواهد کرد.

### ۱۳.۳۷ exp50: بهبود کوچک subtype، شواهد ناکافی برای تغییر default

screen روی هر ۳۳۸ مطالعهٔ OOF کامل شد و با MLflow run
`d8d969f53c5140489b293559f2928a82` ثبت شد. روش primary ازپیش‌ثبت‌شده یعنی
`max_pair_equal_blend`، Any-AUC را از `0.934529` به `0.933615` کاهش داد
(`-0.000914`) و macro subtype AUC را از `0.829159` به `0.832217` افزایش داد
(`+0.003057`). چون معیارهای spatial ثابت‌اند، selection proxy فقط `+0.000184`
بهبود یافت: `0.641638 → 0.641822`.

bootstrap جفت‌شدهٔ ۲۰۰۰تایی در سطح مطالعه برای delta macro-AUC میانگین
`+0.003010`، بازهٔ ۹۵٪ `[-0.000105,+0.006065]` و احتمال مثبت `97.1%` داد؛ برای
Any-AUC میانگین delta برابر `-0.000891`، بازهٔ ۹۵٪
`[-0.003131,+0.001190]` و احتمال مثبت فقط `20.2%` بود. جهت foldها نیز یکنواخت
نیست: foldهای ۰ و ۴ در Any-AUC افت داشتند، fold ۲ اندکی بهتر شد و foldهای ۱ و ۳
بدون تغییر ماندند؛ macro-AUC فقط در foldهای ۰ و ۳ بهتر شد.

انتخاب exploratory و cross-fitted یک pooler مستقل برای هر label نیز موفق نبود:
Any-AUC=`0.931523` و macro-AUC=`0.830953` شد. rule سه‌برشی در تحلیل ثانویه Any-AUC
را فقط `+0.000035` و macro-AUC را `+0.004545` بهتر کرد، اما چون primary نبود و روی
همان OOF دیده شده است، مجوز انتخاب یا deploy محسوب نمی‌شود.

جمع‌بندی: exp50 یک سیگنال واقعی اما بسیار کوچک نشان می‌دهد که تداوم برش‌ها برای
ranking subtypeها مفید است، ولی شواهد برای جایگزینی `max` به‌عنوان default کافی
نیست. `max_pair_equal_blend` فقط به‌عنوان گزینهٔ کم‌ریسک برای مرحلهٔ نهایی نگه داشته
می‌شود و جهش مدل محسوب نمی‌شود. گام بعدی exp51 است: یک logistic meta-head با ظرفیت
کم، featureهای ثابت و ارزیابی پنج‌fold کاملاً cross-fitted؛ تنها در صورت بهبود
معنادار، پایدار و bootstrap-supported به inference منتقل خواهد شد.

### ۱۳.۳۸ پیش‌ثبت exp51: logistic sequence meta-head دو-لایه OOF

exp51 پیش از مشاهدهٔ نتیجه به یک کاندید واحد محدود شد؛ هیچ sweep روی feature، C،
وزن blend یا label انجام نمی‌شود. برای هر یک از شش label، هشت feature threshold-free
و ازپیش‌ثابت استفاده می‌شود: `max`، میانگین top-2، بیشینهٔ میانگین جفت مجاور،
بیشینهٔ geometric-mean جفت مجاور، بیشینهٔ میانگین سه‌برشی، mean و standard deviation
کل sequence و `log(1+slice_count)`. هر label فقط featureهای خودش را می‌بیند؛ هیچ
feature متقاطع subtype/Any وارد مدل نمی‌شود.

مدل یک `StandardScaler` و logistic regression با L2، `C=0.1`،
`class_weight=balanced`، solver=`lbfgs` و حداکثر ۲۰۰۰ iteration است. ظرفیت هر head
فقط هشت coefficient و یک intercept است. base slice scoreها خودشان OOF هستند؛ سپس
برای هر held-out outer fold، scaler و meta-head فقط روی چهار fold دیگر fit و روی fold
پنجم اعمال می‌شوند. بنابراین هیچ مطالعه‌ای نه در base model و نه در meta-head سازندهٔ
score خودش دیده نشده است. score ردیفی ذخیره یا commit نمی‌شود.

گیت promotion عمداً سخت‌تر از exp50 است: delta proxy حداقل `+0.002`، delta macro-AUC
حداقل `+0.005`، افت Any-AUC حداکثر `0.002`، proxy غیرمنفی در حداقل سه fold، افت هیچ
subtype بیش از `0.02` و احتمال bootstrap مثبت برای هر دو proxy و macro-AUC حداقل
`90%`. عبور cross-fit فقط اجازهٔ پیاده‌سازی می‌دهد، نه پذیرش نهایی. در deployment
احتمالی، head نهایی که فقط روی کل OOF fit شده باید جداگانه روی sequence خروجی هر یک
از پنج base fold model اعمال و سپس scoreهای مطالعه average شوند؛ fit یا انتخاب بر
اساس leaderboard ممنوع است. شکست هر گیت این کاندید ثابت را می‌بندد و مجوز tune
پس‌نگر C یا feature روی همان ۳۳۸ مطالعه نیست.

### ۱۳.۳۹ exp51: رد قاطع meta-fit و بستن pooling پس‌آموزشی

exp51 روی ۳۳۸ مطالعه کامل و با MLflow run
`71013357df7f42c09bd087994efe335a` ثبت شد. نتیجه نسبت به `max` نه‌تنها بهبود
نداشت، بلکه Any-AUC از `0.934529` به `0.924508` (`-0.010021`)، macro subtype AUC
از `0.829159` به `0.814997` (`-0.014163`) و selection proxy به‌اندازهٔ
`-0.005131` افت کرد. فقط دو fold از پنج fold proxy غیرمنفی داشتند و هر هفت شرط
promotion شکست خورد.

delta زیرنوع‌ها علت افت را روشن می‌کند: IVH=`-0.030851`، SDH=`-0.044643`،
EDH=`-0.005823`، IPH=`-0.001023` و فقط SAH=`+0.011525`. bootstrap جفت‌شدهٔ ۲۰۰۰
مطالعه‌ای برای proxy بازهٔ ۹۵٪ `[-0.010147,-0.000374]` و احتمال مثبت تنها `1.85%`
داد. برای Any-AUC نیز بازهٔ ۹۵٪ `[-0.020469,-0.000033]` بود؛ پس افت صرفاً نوسان
نقطه‌ای نیست.

coefficientهای مدل نهایی نشان دادند آمارهای pooling شدیداً هم‌بسته‌اند و برای
زیرنوع‌های کم‌نمونه جهت‌های ناپایدار می‌سازند؛ نمونهٔ روشن، coefficient منفی `max`
برای SDH و ترکیب علامت‌های متضاد برای SAH/EDH است. بنابراین هیچ tune پس‌نگر C، حذف
feature انتخابی یا نجات تک‌برچسبی روی همین OOF انجام نمی‌شود. نتیجهٔ مشترک exp50 و
exp51 این است که scoreهای خروجی فعلی مقداری سیگنال تداوم دارند، اما pooling ثابت فقط
سود ناچیز می‌دهد و meta-fit روی ۳۳۸ مطالعه بیش‌برازش می‌کند. مسیر aggregation
پس‌آموزشی بسته می‌شود؛ تغییر بعدی باید از featureهای encoder و supervision برشی
استفاده کند و mask incumbent را دست‌نخورده نگه دارد.

### ۱۳.۴۰ پیش‌ثبت exp52: بازآموزی study-balanced فقط برای classification head

پیش از ساخت GRU یا attention، یک آزمون کم‌هزینه‌تر و علت‌محور تعریف شد. checkpoint
دقیق exp22 بارگذاری و تمام encoder، BatchNorm، decoder و segmentation head در حالت
`eval` و `requires_grad=False` قفل می‌شوند؛ فقط auxiliary classification head با
۲۱۱۸ پارامتر trainable می‌ماند. در نتیجه mask، حجم، Dice، FPR، F1 و MAE باید در هر
epoch bit-identical با epoch صفر بمانند و تنها Any/subtype ranking اجازهٔ تغییر دارد.

فرضیه این است که embeddingهای encoder اطلاعات classification بیشتری از head فعلی
دارند، اما head در آموزش joint با وزن `0.25` و sampling برش‌محور بهینه شده است.
کاندید واحد exp52 از همان focal-BCE و pos-weight موجود، اما با
`classification_loss_weight=1.0` و sampler مطالعه‌محور ثابت `power=0.75` استفاده
می‌کند؛ LR=`1e-4`، weight decay=`1e-3`، batch=`16`، حداکثر ۶ epoch و patience=2
است. هیچ encoder feature، معماری، augmentation، pooling یا threshold دیگری تغییر
نمی‌کند و sweep مجاز نیست.

ابتدا چهار optimizer step فقط گیت سلامت/هویت فضایی است. سپس screen کامل فقط روی
calibration1 انجام می‌شود و outer2 خوانده نمی‌شود. promotion به outer2 مستلزم delta
selection حداقل `+0.002`، delta macro subtype AUC حداقل `+0.005`، افت Any-AUC حداکثر
`0.002`، افت هیچ subtype بیش از `0.01` و هویت دقیق تمام معیارهای spatial/volume است.
اگر پاس شد، outer2 یک‌بار ارزیابی و فقط در صورت proxy مثبت و نبود پسرفت جدی ranking
مسیر پنج‌fold دنبال می‌شود. شکست calibration این linear refit را می‌بندد؛ GRU
feature-level فرضیه‌ای مستقل است و فقط با شواهد نیاز به context اجرا خواهد شد.

### ۱۳.۴۱ exp52: هویت فضایی تأیید، سود خطی ناکافی و outer بسته

smoke چهار-step با run `e0af5e09f0034a24a5bdd3910ddd4f54` در `27.32s` و
peak VRAM=`1.11GB` کامل شد. تعداد پارامتر trainable دقیقاً ۲۱۱۸ بود، epoch صفر
baseline v4 را بازتولید کرد و پس از update نیز تمام معیارهای mask/volume ثابت ماندند؛
پس freeze encoder/decoder/BatchNorm و جداسازی head تأیید شد.

screen کامل calibration-only با run `5e3de5ca78974c50bb1ee4207bef8e37` در
`131.37s` پایان یافت. best epoch=1 بود و patience پس از epoch3 متوقف کرد. مقایسه با
epoch صفر هم‌split:

| معیار calibration1 | baseline | exp52 | delta |
|---|---:|---:|---:|
| selection | 0.659954 | 0.660856 | +0.000901 |
| Any-AUC | 0.920251 | 0.922939 | +0.002688 |
| macro subtype AUC | 0.897887 | 0.898521 | +0.000634 |
| Dice | 0.453083 | 0.453083 | 0 |
| FPR / F1 | 0.194444 / 0.882353 | 0.194444 / 0.882353 | 0 / 0 |
| total-volume MAE | 10.267766mL | 10.267766mL | 0 |

delta AUC زیرنوعی عبارت بود از IVH=`0`، IPH=`-0.001348`، SDH=`+0.006705`،
EDH=`+0.005747` و SAH=`-0.007937`. بنابراین هویت فضایی دقیق و بهبود Any واقعی است،
اما هر دو گیت اصلی selection=`+0.002` و macro=`+0.005` شکست خوردند؛ سود میان
زیرنوع‌ها نیز یکنواخت نیست. outer2 خوانده نشد، checkpoint منتقل نمی‌شود و تنظیم
پس‌نگر LR/power مجاز نیست.

برداشت مکانیکی: encoder frozen سیگنال استفاده‌نشده دارد، چون تنها ۲۱۱۸ پارامتر
خطی Any/SDH/EDH را بهتر کردند؛ بااین‌حال head مستقل هر برش، تداوم محور z را نمی‌بیند
و SAH را پس می‌دهد. همراه با سود کوچک exp50، این نتیجه اجرای یک temporal residual
head کم‌ظرفیت روی featureهای frozen encoder را توجیه می‌کند. آن head باید در
ابتدا logits incumbent را دقیقاً حفظ کند، تنها classification scoreها را تغییر دهد
و قبل از هر outer روی calibration1 گیت سخت داشته باشد.

### ۱۳.۴۲ پیش‌ثبت exp53: temporal residual روی featureهای frozen encoder

exp53 یک کاندید واحد و بدون sweep است. برای split `(outer2, calibration1)`، مدل
exp22 کاملاً frozen/eval می‌ماند و از عمیق‌ترین feature map encoder برای هر برش
global-average pooling گرفته می‌شود. cache فقط برای سه fold train و calibration1
ساخته می‌شود؛ outer2 نه feature extraction و نه inference می‌شود. cache شامل feature،
logit پایه و label برشی است، خارج Git می‌ماند و با SHA checkpoint/manifest قفل
می‌شود.

head ترتیبی از `LayerNorm → Linear(352→64) → GELU → BiGRU(hidden=32)` و یک linear
residual شش‌خروجی تشکیل می‌شود. وزن و bias آخر صفرآغاز هستند؛ پس در epoch صفر تمام
logitها و AUCها باید دقیقاً با exp22/v4 یکسان باشند. مدل حدود ۴۳هزار پارامتر دارد،
dropout=`0.2` است و mask/decoder اصلاً در graph آموزش حضور ندارند. score مطالعه همان
`max(sigmoid(slice-logit))` باقی می‌ماند.

loss برشی همان focal-BCE با gamma=`1` و pos-weight capped=`20` است، اما ابتدا داخل
هر مطالعه average و سپس بین مطالعه‌ها average می‌شود تا طول ضایعه وزن مطالعه را
تعیین نکند. یک study-level focal-BCE با وزن ثابت `0.5` روی max logits افزوده می‌شود؛
truth آن مستقیماً از حجم‌های واقعی مطالعه می‌آید. AdamW با LR=`5e-4`، weight
decay=`1e-3`، batch هشت مطالعه، حداکثر ۲۰ epoch و patience=4 استفاده می‌شود.

ابتدا smoke چهار-step فقط cache، zero-identity، packed sequence، gradient، checkpoint
و MLflow را می‌سنجد. screen کامل فقط calibration1 را می‌بیند. گیت همان exp52 است:
delta proxy حداقل `+0.002`، macro-AUC حداقل `+0.005`، افت Any حداکثر `0.002` و افت
هیچ subtype بیش از `0.01`. spatial/volume به‌طور معماری ثابت است. فقط عبور هم‌زمان
همهٔ گیت‌ها اجازهٔ استخراج و ارزیابی یک‌بارهٔ outer2 را می‌دهد؛ شکست، این معماری
ثابت را بدون tune پس‌نگر می‌بندد.

### ۱۳.۴۳ exp53 calibration: عبور همهٔ گیت‌ها و پیش‌ثبت outer2

اجرای نخست cache با batch=`32` پیش از training توسط guardrail متوقف شد: Any-AUC
دقیقاً برابر بود، اما macro-AUC به‌اندازهٔ `0.000794` با baseline قفل‌شده فرق داشت.
علت، تفاوت شکل batch در BF16 نسبت به evaluator مرجع با batch=`16` بود. tolerance شل
نشد؛ batch extraction وارد هویت SHA cache شد و cache تازه با ۱۶ ساخته شد. baseline
جدید همهٔ AUCهای v4 را دقیقاً بازتولید کرد. این رخداد نتیجهٔ کیفیتی نبود و outer
خوانده نشد.

smoke سالم با run `e64b8b561cff4a969d95dbbe784289f1`، feature-dim=`352`، تعداد
پارامتر trainable=`42,502`، ۲۰۴ مطالعهٔ train، ۶۷ مطالعهٔ calibration و zero-init
bit-identical کامل شد. چهار step فقط proxy را `-0.000119` تغییر داد و فروپاشی نداشت.

screen کامل با MLflow run `8baf49f0919f4f8e9c3484372f9146e6` در `16.35s` آموزش
head پایان یافت. بهترین epoch=12 بود و از epoch8 تا epoch16 proxy در تمام epochها
مثبت ماند؛ نتیجه spike تک‌epoch نیست:

| معیار calibration1 | baseline | exp53 | delta |
|---|---:|---:|---:|
| Any-AUC | 0.920251 | **0.930108** | **+0.009857** |
| macro subtype AUC | 0.897887 | **0.914583** | **+0.016695** |
| selection proxy | 0.659954 | **0.665416** | **+0.005461** |

delta زیرنوع‌ها: IVH=`+0.012903`، SDH=`+0.030651`، EDH=`+0.026820`،
SAH=`+0.019841` و IPH=`-0.006739`. هر چهار گیت ازپیش‌ثبت‌شده پاس شدند. mask، Dice،
FPR، F1، حجم و MAE به‌طور معماری ثابت‌اند. این الگو با فرضیه سازگار است: embedding
incumbent اطلاعات مفیدی داشت که نه refit خطی exp52 و نه pooling score exp50/51 قادر
به استخراج پایدارش نبودند؛ context دوسویهٔ feature-level آن را آشکار کرده است.

پیش از دیدن outer2، گیت تکرار مستقل قفل می‌شود. checkpoint epoch12 و base exp22 فقط
روی outer2 inference می‌شوند؛ هیچ training، threshold، pooling یا انتخاب epoch جدید
مجاز نیست. کاندید برای گسترش پنج‌fold باید روی outer2 هم‌زمان delta proxy حداقل
`+0.001`، delta macro-AUC غیرمنفی، افت Any-AUC حداکثر `0.002` و افت هیچ subtype بیش
از `0.02` داشته باشد. به‌علت فقط یک EDH و سه SAH مثبت در outer2، subtypeهای فاقد
پشتیبانی کافی فقط شرط ایمنی‌اند، نه هدف انتخاب. شکست هر شرط، exp53 را بدون دست‌کاری
پس از outer می‌بندد؛ عبور همهٔ شروط اجازهٔ آموزش cross-fitted چهار split باقی‌مانده
با همین معماری/hyperparameter را می‌دهد، نه promotion نهایی خودکار.

### ۱۳.۴۴ exp53 outer2: بهبود قوی subtype اما شکست گیت Any و رد گسترش

ارزیابی یک‌بارهٔ قفل‌شده روی ۶۷ مطالعهٔ outer2 با MLflow run
`c9d8a44e9c5a4ed8bafc43fcab01cc57` انجام شد. checkpoint temporal با
SHA-256=`1cfdc05e483760232c1664c3c9298c3e063554b85edbeb9bf41365c7f422a6a6`،
checkpoint پایهٔ exp22 و manifest دقیقاً همان SHAهای calibration بودند. baseline
feature-cache نیز Any-AUC=`0.872311827957` و macro-AUC=`0.815132281668` قفل‌شده
را با tolerance برابر `1e-12` بازتولید کرد؛ بنابراین نتیجه ناشی از drift در batch،
checkpoint یا داده نیست. هیچ score ردیفی persist نشد و outer در training، انتخاب
epoch یا تنظیم hyperparameter دخالت نداشت.

| معیار outer2 | baseline | exp53 | delta | گیت |
|---|---:|---:|---:|---|
| Any-ICH AUC | 0.872312 | 0.869176 | **-0.003136** | رد؛ حداقل -0.002 |
| macro subtype AUC | 0.815132 | **0.844210** | **+0.029078** | پاس |
| selection proxy | 0.585364 | **0.588785** | **+0.003421** | پاس |

delta زیرنوع‌ها IVH=`+0.007576`، IPH=`-0.012422`، SDH=`-0.009091`،
EDH=`-0.015152` و SAH=`+0.174479` بود. شرط ایمنی همهٔ زیرنوع‌ها پاس شد، اما شرط
non-inferiority مربوط به Any شکست خورد؛ پس `expansion_allowed=false` و exp53 طبق
پیش‌ثبت بدون تغییر post-hoc بسته می‌شود. checkpoint به `checkpoint/ich` منتقل
نمی‌شود و چهار split دیگر با این recipe آموزش داده نخواهند شد.

برداشت مکانیکی دو بخش دارد. نخست، جهش macro روی split مستقل تأیید می‌کند که context
ترتیبی feature-level واقعاً اطلاعات subtype—به‌خصوص الگوی چندبرشی SAH—را بازیابی
می‌کند و موفقیت calibration صرفاً spike نبوده است. دوم، shared temporal trunk و loss
شش‌برچسبی ranking عمومی Any را اندکی جابه‌جا کرده‌اند؛ با توجه به فقط ۶۷ مطالعه، این
افت کوچک است اما از حد ایمنی ازپیش‌تعیین‌شده عبور کرده و قابل نادیده‌گرفتن نیست.
بهبود بسیار بزرگ SAH نیز به‌علت تنها سه مطالعهٔ SAH-positive در outer2 نباید به‌تنهایی
برای انتخاب معماری استفاده شود.

فایل aggregate نتیجه در
`reports/ich_experiments/2p5d_segmentation/exp53_temporal_residual_outer2_one_shot_v1/outer_evaluation_summary.json`
ثبت شده است. این outer دیگر برای اصلاح exp53 یا انتخاب وزن/کانال استفاده نمی‌شود.
فرضیهٔ بعدی، اگر اجرا شود، باید پیش از training مستقل و معماری‌محور تعریف شود و
نتیجهٔ OOF آن adaptive تلقی شود؛ اعتبار نهایی همچنان فقط از leaderboard واقعی
خواهد آمد.

### ۱۳.۴۵ پیش‌ثبت exp54: temporal area/volume residual با خروجی رسمی حجم

بازبینی اثر exp53 نشان داد ادامهٔ صرفِ classification-AUC کافی نیست: مسابقه از شاخهٔ
ICH پنج حجم فیزیکی می‌گیرد، درحالی‌که exp50 تا exp53 عمداً mask و حجم را ثابت نگه
می‌داشتند. ممیزی OOF incumbent گلوگاه مستقیم‌تری نشان می‌دهد: total-volume bias برابر
`-3.15mL` و MAE برابر `8.72mL` است؛ در calibration1 این دو مقدار به‌ترتیب
`-6.06mL` و `10.27mL` هستند. روی OOF فقط یک مورد از چهار ضایعهٔ `<=2mL` در آستانهٔ
حضور بازیابی می‌شود و SDH/SAH به‌ترتیب Dice=`0.379/0.126` و bias منفی دارند. این
همان نیاز قدیمیِ ثبت‌شده به area/severity head است و مستقل از دست‌کاری exp53 پس از
outer محسوب می‌شود.

exp54 checkpoint exp22 را کاملاً frozen/eval نگه می‌دارد و برای هر برش سه چیز را
خارج Git cache می‌کند: deepest pooled encoder feature، حجم پنج‌کلاسهٔ حاصل از argmax
incumbent و حجم هدف همان برش در فضای فیزیکی resized. head ثابت از
`LayerNorm → Linear(352→64) → GELU`، ده signal پایه
(`log1p` پنج حجم برشی + احتمال پنج subtype) و یک BiGRU با hidden=`32` استفاده
می‌کند. linear پنج‌خروجی آخر صفرآغاز است و residual را در فضای لگاریتمی اعمال می‌کند:
`candidate+1=(base+1)×exp(residual)`. بنابراین در epoch صفر حجم هر برش و مطالعه باید
bit-identical با incumbent باشد؛ residual مثبت می‌تواند ضایعهٔ missed را از حجم صفر
بازیابی کند و residual منفی false positive را سرکوب کند. mask فضایی تغییر نمی‌کند،
اما خروجی رسمی حجم تغییر می‌کند.

loss واحد و بدون sweep است: Smooth-L1 روی `log1p` حجم برشی فقط برای
spatial-knownها، Smooth-L1 مطالعه‌ای روی مجموع پنج subtype با وزن `0.75` و loss
`log1p(total-volume)` با وزن `0.25`. وزن مثبت‌ها از split train با square-root
imbalance و سقف ۸ محاسبه می‌شود. AdamW با LR=`2e-4`، weight decay=`1e-3`، batch
هشت مطالعه، حداکثر ۲۰ epoch و patience=4 استفاده می‌شود. استخراج حتماً batch=`16`
است تا baseline BF16 همان evaluator مرجع را بازتولید کند.

checkpoint فقط میان epochهایی انتخاب می‌شود که FPR حداکثر `+0.02` و presence-F1
حداکثر `0.01` بدتر از epoch صفر باشند؛ در میان آن‌ها کمترین total-volume MAE برنده
است و در غیر این صورت epoch صفر حفظ می‌شود. promotion calibration مستلزم هم‌زمان
این شروط است: MAE حداقل `0.5mL` کمتر، قدرمطلق bias حداقل `0.5mL` بهتر، FPR/F1 در
قیود بالا، افت critical-trigger macro-F1 حداکثر `0.02` و افزایش MAE هیچ subtype بیش
از `0.5mL` نباشد. ابتدا smoke چهار-step و سپس calibration1 اجرا می‌شود؛ outer2 که
برای exp53 مصرف شده دوباره خوانده نخواهد شد.

critical-trigger به‌طور صریح از تمام مرزهای حجمی rule رسمی ساخته می‌شود: EDH=`30`،
SDH=`70`، IPH=`70` و total در `15/40/60mL` (ترکیب شکستگی، ترکیب MLS و critical
مستقیم). هر trigger فاقد نمونهٔ مثبت در calibration از macro کنار گذاشته می‌شود و
FPR/F1 حضور در آستانهٔ `0.1mL` جداگانه gate می‌شوند. این تعریف پیش از نخستین اجرای
smoke ثبت شده تا بهبود MAE نتواند با عبور مخرب از مرزهای تصمیم مسابقه خریداری شود.

اگر calibration پاس شود، نخستین replication ازپیش‌ثابت روی مدل fold0 یعنی exp20
با split `(outer0, calibration1)` و همین recipe انجام می‌شود. outer0 فقط یک‌بار و
بدون تنظیم جدید مصرف می‌شود. عبور replication اجازهٔ OOF پنج‌fold می‌دهد. چون foldها
در پژوهش‌های تاریخی دیده شده‌اند، این OOF صادقانه `adaptive development OOF` نامیده
می‌شود و تأیید نهایی همچنان leaderboard واقعی است؛ هیچ ادعای confirmatory از آن
ساخته نخواهد شد.

### ۱۳.۴۶ exp54 smoke attempt 1: توقف فنی پیش از training

اولین اجرای smoke روی commit `797fbae` پیش از ساخت cache و پیش از هر optimizer step
با `TypeError` متوقف شد. علت، قرارداد decoder در نسخهٔ نصب‌شدهٔ
`segmentation_models_pytorch` بود: `UnetPlusPlusDecoder.forward(features)` یک لیست
feature می‌گیرد، اما extractor اولیه آن را با `decoder(*features)` فراخوانی کرده بود.
GPU پس از خطا idle بود، پوشهٔ cache هیچ فایلی نداشت، تنها artifact اجرا
`resolved_config.json` بود و outer ارزیابی نشد. این رخداد هیچ معنای کیفیتی ندارد؛
اصلاح فقط قرارداد forward را با forward مرجع SMP یکسان می‌کند، تست regression مستقل
برای list-decoder اضافه می‌شود و attempt بعدی دقیقاً با همان hyperparameterهای
پیش‌ثبت‌شده و یک output directory تازه اجرا خواهد شد.

### ۱۳.۴۷ نتیجهٔ exp54 و ممیزی علت شکست

attempt دوم smoke روی commit `406a353` از نظر فنی کامل شد. forward دستی با forward
واقعی U-Net++ به‌صورت bit-exact برابر بود، cache train با SHA-256 برابر
`2fbf26ea53d774f7631ae04612ed4dcea36cf0a09a7a46a1826729451d94814e` و cache
calibration با SHA-256 برابر
`77210e4b0dc89d55961bd1ee94c2a5532d6c2894f26c6ffe6c0078ab70c8303e`
ساخته شد. baseline ۶۷ مطالعه دقیقاً MAE=`10.267765mL`، bias=`-6.061514mL`،
FPR=`0.19444` و F1=`0.88235` را بازتولید کرد و epoch صفر bit-identical بود. چهار
step smoke مقدار MAE را `+0.064mL` بدتر کرد، پس checkpoint به‌درستی epoch صفر ماند.
MLflow smoke run برابر `98e0643215514ffe8248a6d4bb104365` است.

calibration کامل با MLflow run=`cfea23233fec4f0ead4cfcb9ae7b380e` در چهار epoch
بدون بهبود متوقف شد. delta MAE در epochهای ۱ تا ۴ به‌ترتیب
`+0.605/+1.380/+1.535/+1.387mL` بود و قدرمطلق bias نیز تا `+2.83mL` بدتر شد؛ پس
best epoch صفر، خروجی رسمی بدون تغییر و `promotion_allowed=false` است. هیچ checkpoint
exp54 به سیستم محلی منتقل نشد؛ فقط `resolved_config.json`، `history.csv` و
`run_summary.json` برای provenance نگه داشته شدند.

ممیزی cache علت را از نبود mask جدا کرد: نسبت مجموع supervision فضایی به حجم مطالعه
روی مثبت‌ها در train میانه=`0.995` و در calibration میانه=`0.997` بود. مشکل اصلی
stacking in-sample است. checkpoint exp22 روی همان ۲۰۴ مطالعهٔ train مقدار
MAE=`4.924mL` و bias=`+3.608mL` داشت، ولی روی calibration واقعاً ندیده MAE=`10.268mL`
و bias=`-6.062mL` داشت. بنابراین residual head اصلاح مثبت‌بودن bias دادهٔ in-sample
را آموخت و روی held-out کم‌برآوردی را تشدید کرد. کاهش LR یا sweep همین split از نظر
علمی توجیه ندارد.

### ۱۳.۴۸ پیش‌ثبت exp55: پنج‌fold meta-OOF حجمی

exp55 همان head/loss/hyperparameter قفل‌شدهٔ exp54 را نگه می‌دارد، اما feature هر
مطالعه فقط با checkpointی استخراج می‌شود که outer fold آن مطالعه را در training
ندیده باشد: exp20/exp21b/exp22/exp18/exp19 برای foldهای ۰ تا ۴. برای هر meta-heldout
fold، inner-validation با policy ثابت «اولین fold مجاز از `[3,4,0]`» انتخاب می‌شود؛
نگاشت دقیق `0→3, 1→3, 2→3, 3→4, 4→3` است و سه fold باقی‌مانده head را آموزش
می‌دهند. inner فقط checkpoint را با قیود FPR/F1 و کمترین MAE انتخاب می‌کند؛ metaheldout
تا پس از انتخاب checkpoint خوانده نمی‌شود.

ابتدا heldout0 با چهار optimizer step smoke می‌شود و سپس هر پنج meta-fold کامل اجرا
می‌شوند. baseline تجمیعی باید summary قفل‌شدهٔ schema-v4 را بازتولید کند:
MAE=`8.718919mL`، bias=`-3.144161mL`، FPR=`0.172222` و F1=`0.868263`. promotion
علاوه‌بر تمام gateهای exp54، به ۵۰۰۰ paired patient-bootstrap نیاز دارد:
احتمال بهترشدن MAE حداقل `0.95` و کران بالای CI95 delta MAE حداکثر صفر. به‌علت اینکه
checkpointهای پایه در پژوهش تاریخی با calibration1/2 انتخاب شده‌اند و foldها قبلاً
دیده شده‌اند، این ارزیابی صادقانه `adaptive development OOF` است، نه nested
confirmatory؛ تنها leaderboard واقعی می‌تواند تأیید نهایی بدهد.

### ۱۳.۴۹ exp55 smoke attempt 1: اصلاح قرارداد نام fold

attempt اول smoke روی commit `d1b6e7a` پیش از ساخت cache با
`KeyError: outer_fold` متوقف شد. manifest canonical ستون split را `fold` می‌نامد و
`outer_fold` نام پارامتر/مفهوم evaluator است؛ اسکریپت جدید این دو را یکسان فرض کرده
بود. MLflow run شکست‌خورده `2d7872d6ed6749f49d0a922beb16961f` است. هیچ feature،
optimizer step یا metric کیفیتی تولید نشد. اصلاح فقط استفاده از ستون canonical
`fold` است و attempt دوم با همان mapping، gate و hyperparameter در output تازه اجرا
می‌شود.

### ۱۳.۵۰ exp55 کامل: سود حجمی ناچیز، FPR بدتر و رد promotion

attempt دوم smoke با MLflow run=`8beb74cfdb234da4999cdde9ae0e5789` از نظر
فنی سالم بود: پنج cache مستقل OOF ساخته شدند، epoch صفر روی heldout0 دقیقاً با
incumbent برابر بود و چهار optimizer step هیچ checkpoint ایمنِ بهتر از identity
نساخت. سپس اجرای کامل cross-fit با run=`c5c5d40ca3a943b18c5f811bfca6e510`
روی commit=`69497584025e31a6ed87848f9c3044ce94ee9e7e`، ۳۳۸ مطالعه و ۳۲۰ بیمار
انجام شد. هر meta-heldout فقط پس از انتخاب checkpoint روی inner fold خوانده شد.

از پنج head، تنها meta-fold0 در epoch4 نسبت به inner baseline checkpoint ایمنِ
بهتری انتخاب کرد؛ foldهای ۱ تا ۴ epoch صفر را حفظ کردند. نتیجهٔ تجمیعی:

| معیار OOF | incumbent | exp55 | delta |
|---|---:|---:|---:|
| total-volume MAE | 8.718919mL | 8.707533mL | -0.011386mL |
| total-volume bias | -3.144161mL | -3.164856mL | قدرمطلق +0.020695mL بدتر |
| normal FPR | 0.172222 | 0.188889 | +0.016667 |
| presence F1 | 0.868263 | 0.863905 | -0.004358 |
| critical-trigger macro-F1 | 0.730918 | 0.730918 | 0 |

paired patient bootstrap با ۵۰۰۰ نمونه برای delta MAE، CI95 برابر
`[-0.044509,+0.016109]mL` و احتمال بهترشدن فقط `0.7562` داد. هر دو شرط bootstrap،
گیت بهبود حداقل `0.5mL` در MAE و گیت بهبود قدرمطلق bias شکست خوردند؛ بنابراین
`promotion_allowed=false` است. checkpointهای head روی سرور فقط برای provenance
حفظ شدند و به `checkpoint/ich` منتقل نشدند. artifactهای تجمیعی غیرمدلی در
`reports/ich_experiments/2p5d_segmentation/exp55_crossfit_temporal_volume_oof_v1`
محلی آرشیو شدند.

ممیزی ردیفی ۱۸۰ مطالعهٔ truth-normal نشان داد incumbent روی ۳۱ و exp55 روی ۳۴
مطالعه FPR داشت. هر سه عبور جدید از مرز `0.1mL` در meta-fold0 رخ دادند و حجم پایهٔ
هر سه دقیقاً صفر بود، ولی exp55 آن‌ها را به حدود `0.21–0.22mL` رساند. در کل ۱۹
مطالعهٔ نرمال با base دقیقاً صفر، candidate مثبت گرفتند. پس failure mode اصلیِ
تغییر non-identity، جملهٔ `+1` در تبدیل log1p است که اجازه می‌دهد residual مثبت از
support صفر حجم بسازد؛ ادامهٔ sweep روی همان فرمول توجیه ندارد.

### ۱۳.۵۱ پیش‌ثبت exp56: residual ضربی با قفل support صفر

exp56 فقط یک تغییر معماریِ برگرفته از failure mode exp55 دارد و هیچ hyperparameter
را sweep نمی‌کند. به‌جای
`candidate = base + (base+1)×expm1(residual)` از
`candidate = base + base×expm1(residual) = base×exp(residual)` استفاده می‌شود.
بنابراین هر subtype در هر برشی که incumbent حجم دقیقاً صفر دارد، برای تمام وزن‌های
head دقیقاً صفر می‌ماند؛ گرادیان آن نقطه نیز صفر است. این محدودیت عمداً توان بازیابی
ضایعهٔ کاملاً missed را قربانی می‌کند تا head فقط کالیبراسیون شدت support موجود را
بیاموزد. حجم‌های مثبت کوچک همچنان ممکن است از مرز `0.1mL` عبور کنند، بنابراین گیت
FPR حذف یا شل نمی‌شود.

تمام اجزای دیگر ثابت‌اند: همان پنج checkpoint و cache با batch extraction=16،
همان نگاشت inner-fold، feature/projection/BiGRU، LR=`2e-4`، weight decay=`1e-3`،
lossهای slice/study/total، batch هشت مطالعه، patience=4 و حداکثر ۲۰ epoch. ابتدا
تست unit باید هم هویت bit-exact epoch صفر و هم صفرماندن support پس از residual مثبت
را اثبات کند. سپس یک smoke چهار-step روی meta-heldout0 اجرا می‌شود و در صورت سلامت
فنی، پنج‌fold کامل با همان ۵۰۰۰ patient-bootstrap اجرا خواهد شد.

معیار promotion بدون تغییر از exp55 باقی می‌ماند: بهبود MAE و قدرمطلق bias هرکدام
حداقل `0.5mL`، FPR حداکثر `+0.02`، افت F1 حداکثر `0.01`، افت critical-trigger
macro-F1 حداکثر `0.02`، افزایش MAE هیچ subtype بیش از `0.5mL`، احتمال bootstrap
بهترشدن MAE حداقل `0.95` و کران بالای CI95 delta MAE حداکثر صفر. این نیز adaptive
development OOF است و promotion تنها کاندید پژوهشی می‌سازد؛ تأیید نهایی leaderboard
واقعی باقی می‌ماند.

### ۱۳.۵۲ exp56 کامل: حفظ کامل FPR، بهبود جهت‌دار MAE و شکست گیت bias/CI

smoke با MLflow run=`e6597894ff31446aa141e4feb3d7e4d5` از نظر فنی سالم بود؛
چهار step روی meta-fold0 هیچ checkpoint بهتر از identity نساخت. اجرای کامل پنج‌fold
روی commit=`982947e483aee72bb582466ff3a490d16ef559a5` و MLflow
run=`6a46beba166e468f95217b58f509f925` در حدود ۷۱ ثانیه تمام شد. قفل support دقیقاً
failure mode exp55 را حذف کرد: FPR=`0.172222`، F1=`0.868263` و sensitivity حضور
با incumbent bit-identical ماندند.

| معیار OOF | incumbent | exp56 | delta |
|---|---:|---:|---:|
| total-volume MAE | 8.718919mL | 8.538286mL | **-0.180633mL** |
| total-volume bias | -3.144161mL | -3.393235mL | قدرمطلق **+0.249074mL بدتر** |
| normal FPR | 0.172222 | 0.172222 | 0 |
| presence F1 | 0.868263 | 0.868263 | 0 |
| critical-trigger macro-F1 | 0.730918 | 0.747749 | **+0.016831** |

bootstrap بیمارمحور ۵۰۰۰ نمونه احتمال بهترشدن MAE را `0.9586` برآورد کرد، اما CI95
delta MAE برابر `[-0.414285,+0.019348]mL` بود و کران بالا هنوز از صفر عبور کرد.
CI95 تغییر قدرمطلق bias نیز `[+0.04495,+0.49235]mL` بود و فقط احتمال `0.008` برای
بهترشدن bias داد. تنها meta-foldهای ۰ و ۳ non-identity شدند؛ هر دو روی heldout، bias
را بدتر کردند. delta MAE زیرنوع‌ها EDH=`-0.0073`، IPH=`-0.1194`،
IVH=`+0.0481`، SAH=`+0.0084` و SDH=`+0.0250mL` بود. بنابراین سود تجمیعی بیشتر از
IPH می‌آید و کالیبراسیون همهٔ زیرنوع‌ها نیست. تمام گیت‌های safety پاس شدند، اما
حداقل بهبود، bias و CI شکست خوردند؛ `promotion_allowed=false` و هیچ headی به
`checkpoint/ich` منتقل نشد.

یک probe تحلیلی بعدی بدون آموزش، چهار کالیبراتور صفرنگهدار را cross-fit کرد: ضریب
میانگین کلی، ضریب میانگین زیرنوعی، weighted-median L1 زیرنوعی و grid لگاریتمی
زیرنوعی. ضریب کلی bias را بهتر ولی MAE را از `8.7189` به `8.8159mL` بدتر کرد
(`P=0.3192`)؛ همهٔ سیاست‌های زیرنوعی به‌دلیل ضرایب ناپایدار—به‌ویژه SAH تا حدود
چهار برابر—روی inner به identity برگشتند. بنابراین exp57 scalar پیش از مصرف GPU
رد شد و sweep ضریب توجیه علمی ندارد.

### ۱۳.۵۳ exp57/exp58: threshold فضایی، کشف drift در manifest و اصلاح rare-EDH

فرضیهٔ بعدی مستقیماً کم‌برآوردی مرزهای ضایعه را هدف گرفت: برای هر checkpoint پنج‌fold،
آستانهٔ softmax هر subtype فقط روی calibration-fold همان checkpoint با بیشینه‌سازی
pixel-Dice انتخاب و یک‌بار روی outer-fold اعمال شد. تداخل subtypeها با بیشترین نسبت
`probability/threshold` حل می‌شود. نخستین اجرای exp57 روی manifest قدیمی schema3
با SHA=`d63fc4...` انجام شد. هنگام مقایسه با OOF محلی آشکار شد سرور فقط ۷۴۲۸ slice
spatial-known دارد، درحالی‌که manifest معتبر محلی schema4 با SHA محلی=`6b7439...`
۷۵۷۳ slice دارد؛ اختلاف دقیقاً ۱۴۵ clean-negative تازه‌اعتبارسنجی‌شده بود. بنابراین
Dice exp57 غیرقابل‌مقایسه اعلام و برای promotion مصرف نشد.

manifest schema4 به‌جای overwrite کردن ورودی مشترک، پس از تبدیل مکانیکی pathها به
Linux و کنترل کامل hash/تعداد ردیف، در مسیر جداگانهٔ سرور قرار گرفت. SHA نسخهٔ
Linux=`0455e6d24590a652b324c58730d81750d675b8c8d2442e67c12f11c16531ec37`
و تعداد ردیف ۷۶۸۳ است. cache تصویر/label موجود دوباره استفاده شد؛ چون ۱۴۵ promotion
فقط clean-negative با mask صفر هستند. یک ضعف دیگر ابزار نیز آشکار شد: calibration0
هیچ پیکسل EDH مثبت ندارد و اسکریپت قدیمی crash می‌کرد. policy ثابت و محافظه‌کارانهٔ
`threshold=0.5` برای subtype بدون observed pixel اضافه شد، دلیل انتخاب داخل curve
ثبت شد و همراه provenance fold/commit با ۶۷ تست نهایی پوشش داده شد.

exp58 هر پنج fold را روی همین schema4 بازاجرا کرد. thresholdهای انتخاب‌شده:

| outer/calibration | EDH | IPH | IVH | SAH | SDH |
|---|---:|---:|---:|---:|---:|
| 0/1 | 0.03 | 0.50 | 0.50 | 0.005 | 0.005 |
| 1/2 | 0.005 | 0.40 | 0.25 | 0.01 | 0.03 |
| 2/3 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 |
| 3/4 | 0.50 | 0.50 | 0.50 | 0.50 | 0.25 |
| 4/0 | 0.50 fallback | 0.50 | 0.50 | 0.50 | 0.50 |

threshold خام Dice/selection را بهتر کرد، اما FPR را از `0.1722` به حدود `0.2333`
برد؛ پس deployable نبود. یک gate کاملاً label-free و دوطرفه از پیش قفل شد: mask
threshold فقط وقتی استفاده می‌شود که هم hard incumbent و هم thresholded total-volume
در مرز `0.1mL` مثبت باشند؛ در غیر این صورت تمام spatial statistics از hard گرفته
می‌شود. scoreهای classification نیز همیشه از incumbent می‌آیند. این ساختار به‌طور
جبری تصمیم حضور، FPR، F1 و AUC را ثابت نگه می‌دارد و از label outer استفاده نمی‌کند.

### ۱۳.۵۴ exp58 رسمی: کاندید ایمن و جهت‌دار، اما رد promotion به‌علت ناپایداری fold

ارزیاب رسمی روی commit=`e6bd0ba` با MLflow
run=`7f7375a2cf4341999827dac5da5f3483`، ۳۳۸ مطالعه، ۳۲۰ بیمار و ۵۰۰۰ paired
patient-bootstrap اجرا شد. ابتدا OOF قدیمی به‌صورت deterministic با همان ۱۴۵
clean-negative schema4 rescore شد؛ سپس کلید slice/patient/fold، voxel volume و
known flag بین hard و exp58 یک‌به‌یک تطبیق داده شدند. از ۳۳۸ مطالعه، ۱۷۶ مطالعه
threshold و ۱۶۲ مطالعه hard را از gate گرفتند.

| معیار OOF schema4 | hard | gated exp58 | delta |
|---|---:|---:|---:|
| mean foreground Dice | 0.430737 | **0.442940** | **+0.012203** |
| selection proxy | 0.641638 | **0.648349** | **+0.006712** |
| total-volume MAE | 8.718919mL | **8.488571mL** | **-0.230348mL** |
| total-volume bias | -3.144161mL | **-1.392495mL** | قدرمطلق **-1.751667mL** |
| normal FPR | 0.172222 | 0.172222 | 0 |
| presence F1 | 0.868263 | 0.868263 | 0 |

bootstrap برای Dice و selection احتمال بهترشدن `0.9224` داد؛ CI95 دلتاها به‌ترتیب
`[-0.00298,+0.02356]` و `[-0.00164,+0.01296]` بود. برای MAE احتمال بهترشدن
`0.8232` و CI95 برابر `[-0.74466,+0.22731]mL` بود. فقط foldهای ۰ و ۱ در selection
بردند؛ foldهای ۲ تا ۴ پس‌رفت کوچک داشتند. در سطح subtype، بیشترین سود Dice از
SAH حدود `+0.05` و سپس SDH حدود `+0.01` آمد، اما MAE زیرنوعی EDH و SDH به‌ترتیب
حدود `+0.13` و `+0.29mL` بدتر شد. پس بهبود total-volume را نمی‌توان به‌عنوان
بهبود یکنواخت پنج حجم تعبیر کرد.

گیت‌های ایمنی FPR/F1/Any-AUC/macro-AUC و بهبود قدرمطلق bias همگی پاس شدند؛ هفت
شرط `dice/selection/MAE probability`، CIهای متناظر و حداقل سه برد fold شکست خوردند.
نتیجه `promotion_allowed=false` است: checkpoint یا پکیج inference تازه‌ای به
`checkpoint/ich` منتقل نشد. CSVهای OOF پزشکی فقط روی سرور/محیط محلی باقی ماندند؛
MLflow صرفاً متریک‌های تجمیعی و summary بدون شناسه دریافت کرد و تلگرام نیز فقط
گزارش عددی تجمیعی گرفت. exp58 یک ایدهٔ قابل بازگشت برای بسته‌بندی نهایی است، اما
مسیر مدل‌سازی بعدی باید representation فضایی SAH/SDH/EDH را بهبود دهد، نه اینکه
thresholdهای همین checkpoint را بیشتر sweep کند.

### ۱۳.۵۵ پیش‌ثبت exp59: EfficientNetV2-S به‌عنوان تغییر تک‌عاملی encoder

پس از رد شدن calibration حجمی و threshold sweep، فرضیهٔ exp59 این است که محدودیت
بعدی از ظرفیت بازنمایی فضایی encoder می‌آید. غربال فنی روی RTX 3090 با قرارداد واقعی
ورودی ۹کانالهٔ `320×320`، BF16، backward و اولین step از AdamW انجام شد. کنترل
`EfficientNet-B2 + U-Net++` در batch=8 با ۱۰٬۴۰۵٬۸۷۰ پارامتر و peak allocated
برابر `2.544GiB` سالم بود. `EfficientNetV2-S + U-Net++` نیز با وزن ImageNet،
۲۴٬۴۹۷٬۳۴۸ پارامتر و peakهای `1.753/2.886GiB` برای batchهای ۴/۸ سالم اجرا شد.
در مقابل `ConvNeXt-Tiny + U-Net++` در SMP نصب‌شده واقعاً ناسازگار بود: encoder یک
stage صفرکاناله به decoder داد و forward با وزن convolution به شکل `[0,96,3,3]`
متوقف شد. تغییر هم‌زمان decoder به FPN attribution را دو عاملی و ریسک آزمایش را
بیشتر می‌کرد؛ بنابراین ConvNeXt در exp59 وارد نمی‌شود.

exp59 فقط encoder را از `efficientnet-b2` به `tu-efficientnetv2_rw_s` عوض می‌کند.
معماری U-Net++، ورودی سه برش مجاور × سه window، ImageNet pretraining، loss، sampler،
انتخاب checkpoint و تمام hyperparameterهای exp39 ثابت می‌مانند: batch=16،
LR=`2e-4`، weight decay=`1e-4`، حداکثر ۱۰ epoch، patience=3، classification loss
weight=`0.25`، background=`0.15`، focal gamma=`1`، pixel class-weight power=`1`
با سقف ۸، و empty-foreground CE با وزن `0.05` روی سخت‌ترین `0.1%` پیکسل‌های
ماسک خالی. انتخاب checkpoint همچنان
`selection_score - 0.10×normal-FPR@0.1mL` است و hard-negative oversampling و
study rebalancing غیرفعال می‌مانند.

manifest مستقل schema4 سرور با SHA
`0455e6d24590a652b324c58730d81750d675b8c8d2442e67c12f11c16531ec37` استفاده
می‌شود. split برابر `(outer=2, calibration=1)` است؛ train فقط foldهای ۰/۳/۴ را
می‌بیند. ابتدا smoke چهاراoptimizer-step با `--skip-outer-evaluation` انجام می‌شود
و فقط سلامت pretrained adaptation، finite loss، BF16 و VRAM را می‌سنجد. سپس یک
calibration screen کامل اجرا می‌شود و outer2 پیش از عبور گیت زیر خوانده نخواهد شد.

baseline قفل‌شدهٔ incumbent روی calibration1/schema4 عبارت است از checkpoint score
`0.64051`، selection=`0.65995`، Dice=`0.45308`، Any-AUC=`0.92025`، macro-AUC=
`0.89789`، FPR=`0.19444`، F1=`0.88235` و MAE=`10.26777mL`. مجوز یک ارزیابی
one-shot روی outer2 فقط در صورت برقراری هم‌زمان این شروط صادر می‌شود:

1. checkpoint score حداقل `0.64351` باشد؛
2. حداقل یکی از Dice=`0.45808` یا selection=`0.66495` عبور کند؛
3. FPR از `0.19444` بیشتر و F1 از `0.87235` کمتر نشود؛
4. Any-AUC و macro-AUC به‌ترتیب از `0.91025/0.88789` کمتر نشوند؛
5. total-volume MAE از `10.26777mL` بدتر نشود.

Diceهای SAH و SDH جداگانه گزارش می‌شوند، ولی به‌دلیل فقط ۴ و ۹ مطالعهٔ مثبت در این
calibration به‌تنهایی gate نیستند. در صورت شکست هر شرط، checkpoint exp59 به سیستم
محلی منتقل نمی‌شود و outer2 مصرف تازه‌ای نخواهد داشت. در صورت عبور، outer2 فقط
یک‌بار با همین checkpoint و بدون تنظیم جدید ارزیابی می‌شود؛ سپس تصمیم OOF پنج‌fold
با bootstrap بیمارمحور گرفته خواهد شد. این مسیر همچنان adaptive development است و
تأیید نهایی فقط از leaderboard واقعی می‌آید.

### ۱۳.۵۶ نتیجهٔ exp59: backbone قوی‌تر، Pareto بهتر در حضور و افت در حجم

پیش از آموزش، یک ممیزی ایمنی نشان داد `segmentation_train.py` کل پوشهٔ خروجی را با
`mlflow.log_artifacts` ارسال می‌کرد و بنابراین CSVهای ردیفی calibration/outer نیز
ممکن بود خارج از سرور بروند. commit=`e2cac00` logging را به allow-list صریح
`best.pth/resolved_config/history/summary` محدود کرد. چهار CSV ردیفی از allow-list
حذف شدند و تست regression عملکرد را بررسی می‌کند. ۶۴ تست متمرکز هم روی سیستم محلی
و هم سرور پاس شدند. از این نقطه predictionهای ردیفی پزشکی فقط روی سرور/محیط محلی
می‌مانند؛ MLflow مدل و artifactهای تجمیعی را می‌گیرد.

smoke چهار-step روی همان commit و manifest schema4 با run=
`8488b4e399c645bf99f7413f4457ee59` در `17.59s` تمام شد. loss متناهی، shape خروجی
درست، outer-evaluation=false و peak VRAM=`4.481GiB` بود. ضعف کیفیت پس از فقط ۶۴
برش معنای مقایسه‌ای نداشت و smoke صرفاً گیت فنی را پاس کرد.

غربال کامل calibration-only با run=`20a45531455a4c208ba9dc5ee7bc5d5c`
در `702.38s` انجام شد. early stopping در پایان epoch9 فعال شد و checkpoint منتخب
epoch6 با SHA=`c94de064bbad5816701f1788877ac64134eabb3a5171b7152149d06f2075f5ab`
است. outer2 در هیچ مرحله خوانده نشد. مقایسهٔ دقیق با incumbent exp22 که روی همان
schema4 rescore شده بود:

| معیار calibration1/schema4 | incumbent B2 | exp59 V2-S | delta |
|---|---:|---:|---:|
| checkpoint score | 0.64051 | **0.64943** | **+0.00892** |
| selection | 0.65995 | **0.66609** | **+0.00614** |
| mean foreground Dice | 0.45308 | **0.45567** | **+0.00259** |
| Any-ICH AUC | 0.92025 | **0.93011** | **+0.00986** |
| macro subtype AUC | 0.89789 | **0.90961** | **+0.01172** |
| normal FPR | 0.19444 | **0.16667** | **-0.02778** |
| presence F1 | 0.88235 | **0.89552** | **+0.01317** |
| total-volume MAE | **10.26777mL** | 11.51566mL | **+1.24790mL بدتر** |
| total-volume bias | **-6.06151mL** | -8.40827mL | قدرمطلق **+2.34676mL بدتر** |

تحلیل زیرنوع نشان می‌دهد نتیجه صرفاً «backbone بدتر» نیست. EDH Dice به‌اندازهٔ
`+0.13348` و MAE به‌اندازهٔ `-1.11496mL` بهتر شد؛ SDH MAE نیز `-0.91033mL`
بهتر شد، هرچند Dice آن `-0.03478` افت کرد. در مقابل Diceهای IPH/IVH به‌ترتیب
`-0.02263/-0.06286` و MAE آن‌ها `+0.37967/+0.44363mL` بدتر شدند. SAH تقریباً
خنثی بود. بنابراین V2-S اطلاعات مکمل واقعی دارد، اما checkpoint score فعلی حجم کل
را مستقیم نمی‌بیند و epoch6 را با کم‌برآوردی شدیدتر انتخاب کرده است.

epoch9 به‌صورت تشخیصی Dice=`0.45911`، selection=`0.66616` و MAE=`10.76272mL`
داشت؛ یعنی trajectory دیرتر trade-off حجم را کم کرد، ولی checkpoint آن طبق policy
ازپیش‌ثبت‌شده برنده نبود و وزن epoch9 ذخیره نشد. انتخاب post-hoc آن مجاز نیست. این
مشاهده فقط فرضیهٔ بعدی را می‌سازد: پیش از هزینهٔ پنج‌fold باید یک screen قفل‌شدهٔ
مکمل/ترکیبی روی predictionهای calibration انجام شود یا checkpoint objective آینده
به‌طور صریح قید MAE/bias داشته باشد.

exp59 چهار گیت حضور/AUC/selection را پاس کرد، اما شرط MAE را شکست؛ در نتیجه
`promotion_allowed=false`، outer2 دست‌نخورده، و هیچ checkpointی به `checkpoint/ich`
منتقل نشد. فقط چهار artifact تجمیعی در مسیر محلی
`reports/ich_experiments/2p5d_segmentation/exp59_effnetv2s_hardempty001_fprselect_p1_schema4_calonly_f2`
آرشیو شدند. پیام تلگرام جداگانه با همین تصمیم و تحلیل trade-off ارسال شد.

### ۱۳.۵۷ پیش‌ثبت exp60: ترکیب کانالی محافظه‌کارانهٔ B2/V2-S روی calibration1

exp59 نشان داد تغییر encoder یک برد یکنواخت نیست، ولی EDH و Any-ICH اطلاعات مکمل
قابل‌توجهی دارند. پیش از هر آموزش یا خواندن outer2، exp60 یک آزمون deterministic
و فقط calibration است. predictionهای منجمد incumbent B2 و exp59 V2-S روی همان
manifest schema4 با SHA=`0455e6...` استفاده می‌شوند؛ هیچ threshold، وزن blend یا
پارامتر پیوسته‌ای جست‌وجو نخواهد شد.

برای هر subtype، خروجی V2-S فقط وقتی انتخاب می‌شود که Dice و AUC و MAE همگی نسبت
به B2 non-inferior باشند و دست‌کم یکی بهبود سخت داشته باشد؛ در غیر این صورت کانال
B2 دقیقاً حفظ می‌شود. منبع Any-ICH فقط در صورت AUC بهتر به V2-S تغییر می‌کند. این
policy با توجه به اعداد exp59 به‌طور قابل‌پیش‌بینی EDH و Any را نامزد V2-S می‌کند،
اما تصمیم نهایی توسط همان قواعد عمومی و بدون override دستی ساخته می‌شود.

گیت advance از پیش قفل است: selection حداقل `+0.005` بهتر شود؛ Dice، Any-AUC،
macro-AUC، F1 و تمام معیارهای subtype افت نکنند؛ FPR و total-volume MAE بدتر نشوند؛
و کانال‌های مرجع تا سطح strata دقیقاً ثابت بمانند. شکست هر شرط یعنی رد exp60 بدون
خواندن outer2. عبور صرفاً مجوز تکرار همین انتخاب در چارچوب cross-fit پنج‌fold است،
نه promotion نهایی؛ چون انتخاب کانال روی calibration1 به‌تنهایی برآورد OOF نیست.
CSVهای ردیفی روی سرور می‌مانند و فقط خلاصه‌های تجمیعی قابل انتقال/ثبت بیرونی‌اند.

### ۱۳.۵۸ نتیجهٔ exp60: مکمل EDH واقعی است، ولی hybrid جمع حجم را خراب می‌کند

exp60 روی commit=`08a406f` و همان calibration1/schema4 اجرا شد. برای رفع drift
provenance، prediction منجمد exp22 با ۴۸ clean-negative promotion دوباره rescore و
به SHA manifest جدید، SHA prediction، run و checkpoint اصلی متصل شد. invariantهای
Any-AUC، macro-AUC، FPR، F1، MAE و bias دقیقاً ثابت ماندند. سپس selector عمومی،
بدون override دستی، فقط EDH را از V2-S و IVH/IPH/SDH/SAH را از B2 انتخاب کرد؛
Any-ICH نیز به‌دلیل AUC بهتر از V2-S آمد.

| معیار calibration1/schema4 | B2 | exp60 hybrid | delta |
|---|---:|---:|---:|
| selection | 0.659954 | **0.678686** | **+0.018732** |
| mean foreground Dice | 0.453083 | **0.479780** | **+0.026696** |
| Any-ICH AUC | 0.920251 | **0.930108** | **+0.009857** |
| macro subtype AUC | 0.897887 | **0.905167** | **+0.007280** |
| normal FPR | 0.194444 | 0.194444 | 0 |
| presence F1 | 0.882353 | 0.882353 | 0 |
| total-volume MAE | **10.267766mL** | 11.003898mL | **+0.736132mL بدتر** |

کانال EDH به‌تنهایی برد منسجم داشت: Dice=`+0.133482`، AUC=`+0.036398` و MAE=
`-1.114963mL`. بااین‌حال total-volume MAE بدتر شد، چون معیار حجم کل خطای مجموع پنج
زیرنوع را می‌سنجد و جایگزینی EDH بخشی از جبران خطاهای علامت‌دار بین کانال‌ها را از
بین برد. این تفاوت مهم است: بهبود MAE یک زیرنوع الزاماً MAE مجموع را بهتر نمی‌کند.

تمام گیت‌های کیفیت، حضور، AUC، FPR/F1، حفظ دقیق کانال مرجع و حداقل selection gain
پاس شدند؛ تنها `total_volume_mae_ml_not_worse` شکست خورد. تصمیم رسمی
`reject_before_outer` است و outer2 همچنان خوانده نشده. هیچ checkpoint یا hybrid
جدیدی به `checkpoint/ich` منتقل نشد. خلاصهٔ بدون شناسه با MLflow run=
`7b8a683847114355b0f89bbbb4d983a0` ثبت و سه artifact تجمیعی در پوشهٔ محلی exp60
آرشیو شد؛ CSVهای ردیفی فقط روی سرور باقی ماندند.

نتیجهٔ پژوهشی exp59/60 این است که V2-S representation مکمل مفیدی دارد، اما مصرف آن
در inference با channel replacement کافی نیست. آزمایش بعدی باید در زمان آموزش یا
انتخاب checkpoint، قید مستقیم total-volume MAE/bias داشته باشد تا همان برد EDH/Any
بدون کم‌برآوردی مجموع حفظ شود؛ sweep وزن blend روی همین calibration توجیه ندارد.

### ۱۳.۵۹ پیش‌ثبت exp61: انتخاب checkpoint حجمی برای EfficientNetV2-S

exp59/60 نشان داد V2-S از نظر EDH، Any-AUC، macro-AUC و FPR اطلاعات بهتری دارد،
اما checkpoint منتخب epoch6 مجموع حجم را شدیدتر کم‌برآورد می‌کند. history منجمد
exp59 نیز یک trajectory معنادار دارد: از epoch6 تا epoch9 MAE از `11.5157` به
`10.7627mL` و قدرمطلق bias از `8.4083` به `6.2364mL` بهتر شد، درحالی‌که selection
تقریباً ثابت و Dice کمی بهتر شد؛ ولی امتیاز FPR-only به‌علت FPR=`0.1944` آن وزن را
ذخیره نکرد و patience آموزش را پیش از epoch10 بست.

exp61 فقط معیار انتخاب checkpoint را عوض می‌کند. فرمول از پیش قفل‌شده عبارت است از:

`selection - 0.10×FPR - 0.005×total-MAE(mL) - 0.001×|total-bias(mL)|`

ضریب‌ها sweep نمی‌شوند. جریمهٔ `0.005/mL` خطای حجمی ۱۰mL را معادل ۰٫۰۵ امتیاز
می‌کند و جریمهٔ کوچک bias، drift جهت‌دار را بدون غالب‌شدن بر selection کنترل می‌کند.
بازمحاسبهٔ صرفاً تشخیصی روی history exp59 امتیاز epoch6/8/9 را به‌ترتیب
`0.58344/0.58383/0.58667` می‌دهد؛ بنابراین فرضیهٔ قابل‌ابطال این است که ذخیرهٔ
trajectory دیرتر و رسیدن به epoch10، ضعف حجم را کم می‌کند بی‌آنکه برد EDH/Any از
بین برود.

encoder=`tu-efficientnetv2_rw_s`، U-Net++، ورودی ۹کاناله، seed=42، pretrained،
manifest schema4، split `(outer=2, calibration=1)`، batch=16، LR=`2e-4`، WD=
`1e-4`، ۱۰ epoch، patience=3، loss و sampler دقیقاً exp59 می‌مانند. ابتدا smoke
چهار-step و سپس calibration-only اجرا می‌شود؛ outer2 در هر دو ممنوع است.

گیت نسبت به incumbent B2/schema4 قفل است: امتیاز همین فرمول برای baseline برابر
حدود `0.58311` است و checkpoint score حجمی باید با حاشیهٔ ۰٫۰۰۳ حداقل `0.58611`
باشد؛ selection حداقل
`0.66495` یا Dice حداقل `0.45808`؛ FPR حداکثر `0.19444` و F1 حداقل `0.87235`؛
Any/macro-AUC حداقل `0.91025/0.88789`؛ MAE حداکثر `10.26777mL`؛ و قدرمطلق bias
حداکثر `6.06151mL`. برای حفظ برد representation، EDH Dice نباید از baseline
`0.44314` کمتر شود. شکست هر شرط یعنی رد پیش از outer و عدم انتقال checkpoint.

### ۱۳.۶۰ نتیجهٔ exp61: checkpoint دیرتر ذخیره شد، اما شکاف حجم کاملاً بسته نشد

پیاده‌سازی `fpr_volume_penalized` با commit=`b512c44` و ۶۱ تست متمرکز سالم شد.
smoke چهار-step با run=`a8a9180b528848c7b12f9fec9f655ab3` در `17.22s`،
peak VRAM=`4.481GiB` و بدون outer سالم بود. اجرای کامل calibration-only با run=
`14b15a032e4949dc8916a78670d9e32b` در `779.96s` تمام شد. همان seed، trajectory
epochهای ۱ تا ۹ exp59 را دقیقاً بازتولید کرد؛ معیار جدید epoch9 را ذخیره و اجازهٔ
اجرای epoch10 را صادر کرد. epoch10 ضعیف‌تر شد، پس epoch9 با checkpoint SHA=
`5f304018340e88e1d858d0842e432d8a163c96ed26117796c7068b6d399c18d4` برنده ماند.

| معیار calibration1/schema4 | incumbent B2 | exp61 epoch9 | delta |
|---|---:|---:|---:|
| volume-aware checkpoint score | 0.58311 | **0.58667** | **+0.00356** |
| selection | 0.65995 | **0.66616** | **+0.00621** |
| mean foreground Dice | 0.45308 | **0.45911** | **+0.00602** |
| Any-ICH AUC | 0.92025 | **0.92339** | **+0.00314** |
| macro subtype AUC | 0.89789 | **0.91092** | **+0.01303** |
| normal FPR | 0.19444 | 0.19444 | 0 |
| presence F1 | 0.88235 | 0.88235 | 0 |
| total-volume MAE | **10.26777mL** | 10.76272mL | **+0.49495mL بدتر** |
| total-volume bias | **-6.06151mL** | -6.23636mL | قدرمطلق **+0.17484mL بدتر** |

EDH هنوز بهتر از baseline است: Dice=`+0.09484`، AUC=`+0.04885` و MAE=
`-0.74603mL`. SDH نیز Dice=`+0.01659` و MAE=`-1.09952mL` بهتر شد. در مقابل IPH
Dice=`-0.02434` و MAE=`+1.20380mL`، IVH Dice=`-0.03563` و MAE=`+0.36285mL`،
و SAH Dice/AUC/MAE هر سه اندکی بدتر شدند. پس criterion جدید بخشی از مشکل حجم exp59
را حل و SDH را بهتر کرد، اما خطای IPH/IVH و ضعف SAH مانع عبور کامل شد.

گیت checkpoint score، selection/Dice، FPR/F1، Any/macro-AUC و EDH همگی پاس شدند؛
فقط MAE کل و قدرمطلق bias شکست خوردند. تصمیم `reject_before_outer` است: outer2
خوانده نشد، checkpoint به سیستم محلی منتقل نشد و تنها summary/history/config/run
summary بدون ردیف پزشکی محلی آرشیو شدند. نتیجهٔ علّی این است که انتخاب checkpoint
لازم بود ولی کافی نیست؛ آزمایش بعدی باید مستقیماً گرادیان آموزش را با حجم فیزیکی
هم‌راستا کند یا head حجمی‌ای بسازد که representation V2-S را بدون اتکا به جمع hard
mask تصحیح کند. تنظیم بیشتر ضرایب همین criterion روی calibration1 ممنوع است.

### ۱۳.۶۱ پیش‌ثبت exp62: هم‌راستاسازی مستقیم گرادیان با حجم فیزیکی

exp61 ثابت کرد انتخاب checkpoint حجمی شکاف MAE را از `+1.24790mL` به
`+0.49495mL` کم می‌کند، اما چون خود تابع آموزش حجم را مستقیم نمی‌بیند از baseline
عبور نکرد. exp62 یک تغییر علّی مستقل و تک‌عاملی است: برای هر برش دارای supervision
فضایی، جرم softmax پنج کلاس خونریزی در `resized_voxel_volume_ml` ضرب می‌شود؛ سپس
Smooth-L1 روی `log1p(volume)` برای حجم پنج زیرنوع و حجم کل، با سهم برابر، به loss
اضافه می‌شود. ردیف‌های classification-only کاملاً mask می‌شوند و بنابراین ماسک خالی
فرض نمی‌شوند. log-transform اجازه نمی‌دهد چند خونریزی بزرگ بر ضایعات کوچک غالب شوند.

پیاده‌سازی عمومی با commit=`84d40c4` و ۶۵ تست محلی/سرور سالم شد. پیش از تعیین وزن،
probe تشخیصی و بدون هیچ parameter update روی checkpoint منجمد exp61، ۲۴ batch
آموزشی، ۳۷۴ ردیف دارای ماسک و seed=42 اجرا شد. SHA checkpoint برابر
`5f304018...c18d4` و SHA manifest برابر `0455e6d...ec37` بود. cosine حجم با
segmentation میانگین/میانه=`0.2597/0.2423`، سهم batchهای متضاد=`20.83%` و نسبت
گرادیان خام volume/segmentation میانگین/میانه=`0.8249/0.6544` شد. بنابراین وزن
`0.15` پیش از آموزش قفل می‌شود: سهم موردانتظار گرادیان حجمی تقریباً `12.4%` در
میانگین و `9.8%` در میانه است؛ نه sweep انجام می‌شود و نه وزن با نتیجهٔ calibration
تنظیم خواهد شد. خروجی probe در
`diagnostics/exp61_train_physicalvolume_grad24b_v1` روی سرور نگه‌داری می‌شود.

تمام تنظیمات دیگر دقیقاً exp61 می‌مانند: U-Net++/`tu-efficientnetv2_rw_s`، ورودی
۹کاناله، ImageNet initialization، manifest schema4، split `(outer=2,
calibration=1)`، batch=16، LR=`2e-4`، WD=`1e-4`، seed=42، ۱۰ epoch، patience=3،
pixel class weighting، hard-empty=`0.05` روی top=`0.001` و checkpoint criterion=
`fpr_volume_penalized`. ابتدا smoke چهار-step و سپس فقط calibration-only اجرا
می‌شود؛ outer2 تا عبور کامل گیت ممنوع است.

گیت همان exp61 و بدون تغییر پس از مشاهدهٔ نتیجه است: volume-aware score حداقل
`0.58611`؛ selection حداقل `0.66495` یا Dice حداقل `0.45808`؛ FPR حداکثر
`0.19444`، F1 حداقل `0.87235`، Any/macro-AUC حداقل `0.91025/0.88789`، EDH Dice
حداقل `0.44314`، total-volume MAE حداکثر `10.26777mL` و قدرمطلق bias حداکثر
`6.06151mL`. شکست هر شرط به معنی رد پیش از outer و عدم انتقال checkpoint است؛
عبور فقط مجوز پنج-fold OOF همین recipe خواهد بود، نه promotion یا ادعای leaderboard.

### ۱۳.۶۲ نتیجهٔ exp62: حجم compact بهتر شد، اما SDH/SAH و FPR هزینه دادند

probe ازپیش‌ثبت‌شده وزن `0.15` را قفل کرد. smoke چهار-step با MLflow run=
`54faabc9b77f480b91b208e7777372a2` در `21.53s`، peak VRAM=`4.481GiB`، loss
متناهی و `outer_evaluation_performed=false` سالم بود. کیفیت smoke تفسیر نشد.
اجرای کامل calibration-only با run=`61e17fe36e3c4a54a1690e6127085b16`
در `809.34s` و peak VRAM=`4.483GiB` تمام شد. criterion حجمی epoch9 را با SHA=
`0ccc5aecc4ee26c525144eaf7ec6264eada32db0603e682e5ad83b2d6bd4955b` انتخاب
کرد؛ epoch10 score پایین‌تری داشت و outer2 در کل اجرا خوانده نشد.

| معیار calibration1/schema4 | B2 incumbent | exp61 V2-S | exp62 volume-loss | delta exp62-B2 |
|---|---:|---:|---:|---:|
| volume-aware checkpoint score | **0.58311** | **0.58667** | 0.56813 | -0.01498 |
| selection | **0.65995** | **0.66616** | 0.65501 | -0.00495 |
| mean foreground Dice | **0.45308** | **0.45911** | 0.44993 | -0.00315 |
| Any-ICH AUC | **0.92025** | **0.92339** | 0.90591 | -0.01434 |
| macro subtype AUC | 0.89789 | **0.91092** | 0.90515 | **+0.00726** |
| normal FPR | **0.19444** | **0.19444** | 0.22222 | +0.02778 بدتر |
| presence F1 | **0.88235** | **0.88235** | 0.86957 | -0.01279 |
| total-volume MAE | **10.26777mL** | 10.76272mL | 11.32951mL | +1.06175mL بدتر |
| total-volume bias | **-6.06151mL** | -6.23636mL | -8.01292mL | قدرمطلق +1.95141mL بدتر |

اثر loss یکنواخت نبود و همین نکته از رد سادهٔ ایده مهم‌تر است. نسبت به exp61، IVH
Dice=`+0.01937` و MAE=`-0.19149mL`، IPH Dice=`+0.00663` و MAE=
`-0.97977mL`، و EDH Dice/AUC/MAE به‌ترتیب `+0.00435/+0.00862/-0.13149mL`
بهتر شدند. در مقابل SDH Dice=`-0.05522` و MAE=`+0.95183mL` و SAH
Dice/AUC=`-0.02100/-0.01587` افت کردند. تفسیر محتمل این است که loss حجم soft
برش‌محور برای خونریزی‌های compact مفید است، اما میانگین‌گیری روی تعداد زیاد زوج
slice/subtype با target صفر، فشار shrinkage نامناسبی بر morphologyهای نازک و پخش
SDH/SAH وارد می‌کند. همچنین per-slice alignment با MAE مجموع study-level یکسان
نیست و جبران خطاهای علامت‌دار میان زیرنوع‌ها را از بین می‌برد.

exp62 گیت macro-AUC و EDH را پاس کرد، اما score، selection/Dice، Any-AUC، FPR،
F1، MAE و bias را شکست. تصمیم رسمی `reject_before_outer` است؛ checkpoint به
`checkpoint/ich` منتقل نشد. فقط summary/history/config/run-summary و خلاصهٔ probe
بدون ردیف پزشکی محلی آرشیو شدند و پیام تحلیلی فارسی تلگرام ارسال شد. sweep وزن
`0.15` روی همین calibration ممنوع است. فرضیهٔ بعدی، در صورت اجرا، باید به‌جای
فشار symmetric روی همهٔ slice/subtypeها، supervision مثبت/مورفولوژی‌آگاه یا یک
objective واقعاً study-level داشته باشد و پیش از مشاهدهٔ outer قفل شود.

### ۱۳.۶۳ پیش‌ثبت exp63: Tversky مثبت‌محور و محافظه‌کار برای SDH/SAH

شکست exp62 نشان داد loss حجمی متقارن، گرادیان مفیدی برای IPH/IVH/EDH ایجاد می‌کند
اما targetهای صفر فراوان، morphology نازک و پخش SDH/SAH را shrink می‌کنند. exp63
در نتیجه warm-start مستقیم از checkpoint منجمد exp61 است و فقط یک عامل تازه دارد:
Tversky روی کانال‌های SDH/SAH با `alpha=0.3` و `beta=0.7`. این loss فقط روی ردیف‌های
دارای supervision فضایی و فقط وقتی همان زیرنوع واقعاً در mask حضور دارد محاسبه
می‌شود؛ ردیف‌های empty و classification-only هیچ فشار Tversky نمی‌گیرند. loss حجم
فیزیکی exp62 کاملاً غیرفعال است.

پروب گرادیان تشخیصی، بدون parameter update و بدون outer، روی SHA checkpoint=
`5f304018340e88e1d858d0842e432d8a163c96ed26117796c7068b6d399c18d4`، SHA
manifest=`0455e6d24590a652b324c58730d81750d675b8c8d2442e67c12f11c16531ec37`،
۲۴ batch، ۳۷۴ ردیف spatially-known و seed=42 انجام شد. cosine گرادیان diffuse
Tversky با segmentation میانگین/میانه=`0.58473/0.59780`، سهم تعارض=`4.35%` و
نسبت norm خام Tversky/segmentation میانگین/میانه=`1.71037/1.42083` بود. نسخهٔ اول
گزارش، پیشنهاد وزن diffuse را به‌اشتباه با فرمول physical-volume overwrite می‌کرد؛
هندسهٔ خام صحیح بود. commit=`22843ac` محاسبهٔ دو خانوادهٔ وزن را مستقل و با تست
regression اصلاح کرد. بازاجرای دقیق v2 هندسهٔ خام را bit-for-bit بازتولید کرد و
وزن‌های متناظر با فشار ۵٪ را mean/median=`0.03525/0.03156` داد. بنابراین پیش از
آموزش وزن `diffuse_tversky_loss_weight=0.03` قفل می‌شود؛ فشار میانهٔ موردانتظار
حدود `4.26%` است و هیچ sweep وزنی روی calibration انجام نخواهد شد.

تنظیمات ثابت exp63: U-Net++/`tu-efficientnetv2_rw_s`، warm-start از exp61، ورودی
۹کاناله، manifest schema4، split `(outer=2, calibration=1)`، batch=16، workers=4،
seed=42، LR=`2e-5`، WD=`1e-4`، حداکثر ۴ epoch، patience=2، classification loss=
`0.25`، background=`0.15`، focal gamma=`1`، pixel class weighting power=`1` با
سقف ۸، hard-empty=`0.05` روی top=`0.001`، بدون hard-negative oversampling و بدون
study rebalance. checkpoint criterion همان `fpr_volume_penalized` است. LR ده برابر
کمتر از آموزش اولیه انتخاب شد تا representation موفق exp61 حفظ شود، ولی همهٔ
پارامترها trainable می‌مانند تا decoder بتواند morphology پخش را اصلاح کند.

ابتدا smoke یک‌epoch/چهار-step با همین recipe و `--skip-outer-evaluation` اجرا
می‌شود؛ گیت آن صرفاً provenance صحیح warm-start، loss متناهی، component متناهی
Tversky، سلامت BF16/VRAM، `outer_evaluation_performed=false` و checkpoint epoch0
است. کیفیت smoke تفسیر نمی‌شود. سپس calibration-only کامل، باز هم بدون outer،
اجرا می‌شود. baseline exp61 در epoch0 ذخیره می‌شود و candidate فقط با epoch بزرگ‌تر
از صفر قابل قبول است.

گیت advance قبل از مشاهدهٔ نتیجه قفل است:

1. `best_epoch >= 1` و volume-aware checkpoint score حداقل `0.58767`، یعنی دست‌کم
   `+0.001` نسبت به exp61=`0.586668`؛
2. میانگین Dice دو subtype پخش SAH/SDH حداقل `0.22234`، یعنی `+0.005` نسبت به
   exp61=`0.21734`، و هیچ‌کدام بیش از `0.01` افت نکند: SAH حداقل `0.04302` و SDH
   حداقل `0.37166`؛
3. subtypeهای compact نسبت به exp61 حفظ شوند: EDH/IPH/IVH Dice به‌ترتیب حداقل
   `0.52798/0.66484/0.63802`؛
4. گیت سراسری قبلی B2 حفظ شود: selection حداقل `0.66495` یا mean Dice حداقل
   `0.45808`؛ FPR حداکثر `0.19444`، F1 حداقل `0.87235`، Any/macro-AUC حداقل
   `0.91025/0.88789`؛
5. شکاف حجم واقعاً بسته شود: total-volume MAE حداکثر `10.26777mL` و قدرمطلق bias
   حداکثر `6.06151mL`.

شکست هر شرط یعنی `reject_before_outer`، عدم اجرای OOF و عدم انتقال checkpoint.
عبور همهٔ شروط فقط مجوز اجرای همان recipe در چارچوب OOF پنج‌fold با bootstrap
بیمارمحور است؛ نه promotion نهایی و نه ادعای leaderboard. CSVهای ردیفی پزشکی روی
سرور می‌مانند و MLflow/تلگرام فقط artifact و معیارهای تجمیعی دریافت می‌کنند.

### ۱۳.۶۴ نتیجهٔ exp63: سیگنال SAH واقعی بود، اما SDH و حجم مانع ادامه شدند

smoke چهار-step پس از اصلاح صریح `PYTHONPATH` با MLflow run=
`7d3934db4fdf47eab05301eaa44aa039` در `31.38s`، peak VRAM=`4.484GiB`، loss
متناهی، diffuse-Tversky متناهی و `outer_evaluation_performed=false` سالم شد. خطای
اول launcher پیش از import مدل و داده رخ داده بود و هیچ update یا خواندن outer انجام
نداد. اجرای کامل calibration-only با run=`afbe6cf814e244cab1afde708db42939`
در `169.51s` و peak VRAM=`4.486GiB` پایان یافت. manifest SHA و warm-start SHA
دقیقاً با پیش‌ثبت برابر بودند و outer2 در هیچ مرحله خوانده نشد.

checkpoint اولیهٔ exp61 در epoch0 با score=`0.586668` بهترین باقی ماند. epoch1 و
epoch2 به‌ترتیب score=`0.565760/0.573030` گرفتند و patience=2 اجرا را متوقف کرد؛
بنابراین `best_epoch=0` و تمام معیارهای best candidate دقیقاً با exp61 برابر است.
گیت ماشینی تمام provenance checks را پاس کرد، اما شرط `best_epoch>=1`، score حداقل
`0.58767`، diffuse-mean Dice حداقل `0.22234`، MAE حداکثر `10.26777mL` و قدرمطلق
bias حداکثر `6.06151mL` شکست خوردند. تصمیم رسمی `reject_before_outer` است: OOF
اجرا نشد، checkpoint به `checkpoint/ich` منتقل نشد و فقط artifactهای تجمیعی محلی
آرشیو شدند.

trajectory دو epoch نشان می‌دهد ایده کاملاً بی‌سیگنال نبود، اما هدف مشترک SAH/SDH
نامتوازن عمل کرد. نسبت به epoch0، SAH Dice در epoch1/2 از `0.05302` به
`0.06556/0.06456` بهتر شد؛ در مقابل SDH از `0.38166` به `0.28151/0.32155` افت
کرد. در epoch2، EDH/IPH به `0.55440/0.68681` بهتر از exp61 بودند، IVH تقریباً روی
مرز حفظ (`0.63773`) بود و macro-AUC=`0.91120` حفظ شد؛ اما Any-AUC به `0.91398`،
Dice کل به `0.45301`، MAE به `11.82047mL` و bias به `-8.45252mL` بدتر شدند.
کاهش train loss از `0.21760` به `0.21199` هم‌زمان با بدترشدن معیارهای calibration
نشان می‌دهد مسئله کمبود همگرایی ساده نیست و ادامهٔ epoch یا sweep وزن ۰٫۰۳ توجیه
علمی ندارد.

برداشت علّی: positive-only Tversky برای morphology بسیار نازک SAH recall مفید ایجاد
کرد، ولی ادغام SAH و SDH زیر یک وزن/فرمول واحد، با وجود تفاوت شدید اندازه و شکل،
مرزهای SDH را خراب و کم‌برآوردی حجم را تشدید کرد. آزمایش بعدی نباید تکرار وزنی همین
loss باشد؛ باید ابتدا خطای calibration را در سطح study/slice و به تفکیک subtype،
حجم و confidence کالبدشکافی کند و سپس یک مداخلهٔ تک‌عاملی مستقل—مانند supervision
تفکیک‌شدهٔ SAH بدون دست‌زدن به SDH یا اصلاح study-level حجم پس از segmentation—را
پیش از outer و با گیت تازه قفل کند.

### ۱۳.۶۵ پیش‌ثبت exp64: خوانش حجم نرمِ دما‌دار با قفل حضور

exp61 هم‌زمان Dice/AUC بهتری از B2 و MAE/bias ضعیف‌تری دارد؛ بنابراین پیش از خرج
GPU برای معماری تازه باید مشخص شود bottleneck از representation است یا از تبدیل
`argmax mask → volume`. exp64 هیچ وزنی را آموزش نمی‌دهد و checkpoint منجمد exp61
با SHA=`5f304018340e88e1d858d0842e432d8a163c96ed26117796c7068b6d399c18d4`
را فقط روی calibration1/schema4 یک بار inference می‌کند. outer2 نه loader inference
می‌گیرد، نه summary و نه artifact.

برای هر پیکسل، حجم پنج subtype از جرم `softmax(logits / T)` در فضای فیزیکی resized
محاسبه می‌شود. grid پیش از اجرا و بدون sweep تطبیقی برابر
`T={0.05,0.10,0.20,0.35,0.50,0.75,1.00}` است. mask hard، Dice پیکسلی، scoreهای
classification و AUCها مطلقاً تغییر نمی‌کنند. قفل حضور دوطرفه نیز study-level است:
soft volume فقط وقتی استفاده می‌شود که هم total hard و هم total soft حداقل `0.1mL`
باشند؛ در غیر این صورت hard volume حفظ می‌شود. پس تصمیم حضور، FPR و F1 باید با
tolerance=`1e-12` دقیقاً invariant بماند. هیچ prediction ردیفی persist یا به MLflow
ارسال نمی‌شود؛ فقط curve هفت‌ردیفی و summary تجمیعی ثبت خواهد شد.

temperature برنده با بیشینه‌کردن همان score ازپیش‌تعریف‌شدهٔ
`fpr_volume_penalized` انتخاب می‌شود و tie به temperature کوچک‌تر/نزدیک‌تر به hard
می‌رسد. گیت advance هم‌زمان همهٔ شروط زیر را می‌خواهد:

1. checkpoint score حداقل `0.58900` و MAE نسبت به exp61 حداقل `0.50mL` بهتر شود؛
2. MAE حداکثر B2 یعنی `10.26777mL` و قدرمطلق bias حداکثر `6.06151mL` باشد؛
3. حداقل یکی از دو temperature مجاور نیز MAE را حداقل `0.25mL` بهتر و قدرمطلق bias
   را بهتر از exp61 کند تا optimum تک‌نقطه‌ای نباشد؛
4. selection/Dice/Any-AUC/macro-AUC/FPR/F1 دقیقاً invariant بمانند؛
5. MAE هیچ subtype بیش از `0.25mL` بدتر نشود.

عبور فقط مجوز پیاده‌سازی و ارزیابی cross-fitted پنج‌fold با temperature منتخب در
هر calibration fold است، نه خواندن outer2 این checkpoint، promotion یا leaderboard
claim. شکست یعنی رد readout نرم و انتقال بودجه به معماری‌ای که SAH/SDH یا حجم
study-level را مستقل مدل کند؛ grid یا threshold پس‌نگر روی calibration1 گسترش داده
نخواهد شد.

### ۱۳.۶۶ نتیجهٔ exp64: اصلاح بایاس واقعی، اما MAE ناکافی و رد readout نرم

غربال calibration-only روی commit=`1d22a0a360fafa6f1f3a43c9b447508bba32a5f4`
و MLflow run=`37b962030add45929f612b7cc4630b7a` در `16.41s` با peak
VRAM=`0.981GiB` کامل شد. SHA checkpoint و manifest دقیقاً با پیش‌ثبت برابر بودند،
۶۷ مطالعهٔ calibration و ۱۳۳۵ برش spatial-known ارزیابی شدند و
`outer_evaluation_performed=false` ماند. قفل دوطرفه در تمام temperatureها ۳۷ مطالعه
را به soft readout داد و ۳۰ مطالعه را روی hard نگه داشت؛ FPR=`0.19444`، F1=
`0.88235`، Dice، Any-AUC و macro-AUC تا tolerance `1e-12` ثابت ماندند.

curve از `T=0.05` تا `T=1.00` به‌طور کلی بایاس را کمتر منفی کرد، اما اثر MAE کوچک
بود. بهترین نقطهٔ ازپیش‌تعریف‌شده `T=1.00` شد:

| معیار calibration1 | hard exp61 | soft T=1 | delta |
|---|---:|---:|---:|
| volume-aware checkpoint score | 0.586668 | 0.587866 | +0.001197 |
| total-volume MAE | 10.762716mL | 10.627109mL | **-0.135606mL** |
| total-volume bias | -6.236356mL | -5.716894mL | **+0.519462mL** |
| normal FPR | 0.19444 | 0.19444 | 0 |
| presence F1 | 0.88235 | 0.88235 | 0 |

MAE زیرنوعی فقط SAH به‌اندازهٔ ناچیز `-0.00039mL` بهتر شد؛ IVH/IPH/SDH/EDH
به‌ترتیب `+0.02380/+0.14684/+0.07195/+0.08458mL` بدتر شدند، هرچند همگی گیت
non-inferiority برابر `+0.25mL` را پاس کردند. بهترین temperature روی لبهٔ بالایی grid
قرار گرفت، ولی همسایهٔ `T=0.75` فقط `0.03306mL` MAE را بهتر کرد و شرط پایداری
`0.25mL` را پاس نکرد. score حداقل `0.58900`، بهبود MAE حداقل `0.50mL`، MAE مرجع
B2=`10.26777mL` و neighborhood stability همگی شکست خوردند؛ فقط bias، invariance و
subtype safety پاس شدند. تصمیم رسمی `reject_before_outer` است و grid به دماهای بیشتر
گسترش داده نمی‌شود.

تفسیر علّی: جرم احتمال نرم واقعاً بخشی از کم‌برآوردی سیستماتیک hard mask را برمی‌گرداند،
اما بیشتر این افزایش، error را میان مطالعه/زیرنوع‌ها جابه‌جا می‌کند و absolute error را
به‌اندازهٔ لازم کم نمی‌کند. پس bottleneck غالب صرفاً quantization مرز argmax نیست؛
representation یا سر خروجی باید morphology/حجم را بهتر مدل کند. ادامهٔ post-processing
روی همین checkpoint ارزش GPU/ریسک overfit ندارد. مرحلهٔ بعد باید یک معماری مستقل و
قابل‌برگشت برای subtypeهای diffuse یا یک head حجمی end-to-end با آموزش patient-safe
باشد، نه sweep temperature، threshold یا scalar calibration.

### ۱۳.۶۷ پیش‌ثبت exp65: گسترش پس‌زمینه به SAH با head ایزوله و صفرمقدار

ممیزی foldهای بیمارمحور نشان می‌دهد train مجاز 0/3/4 فقط ۱۶ مطالعهٔ SAH مثبت دارد؛
calibration1 نیز تنها چهار SAH مثبت دارد و هیچ‌کدام isolated-SAH نیست. بنابراین
adapter پرظرفیت، sweep وزن یا sampler مطالعه‌محور توجیه ندارد. p0.50 و p0.75 قبلاً
distribution shift و افت AUC/FPR ایجاد کرده‌اند، پس `sampler_study_balance_power=0`
حفظ می‌شود.

exp65 مدل exp61 هم‌split را فریز می‌کند و head حدود سه‌هزارپارامتری از decoder feature
و mask logits جداشده استفاده می‌کند. Conv نهایی صفر است و خروجی اولیه دقیقاً هویت
exp61 خواهد بود. residual محدود `8*tanh` فقط به logit کلاس SAH در پیکسل‌هایی افزوده
می‌شود که argmax اولیه پس‌زمینه است؛ بنابراین IVH/IPH/SDH/EDH و SAH موجود نه حذف و
نه جابه‌جا می‌شوند. این مداخله فقط false-negativeهای SAH از نوع background را هدف
می‌گیرد—همان بخشی که exp63 با بهترکردن SAH اما خراب‌کردن SDH نتوانست ایزوله کند.

recipe قفل‌شده: `epochs=6`، `patience=2`، `lr=5e-4`، hidden=16، loss طبقه‌بندی=0،
Tversky مثبت‌محور فقط SAH با وزن `0.03`، بدون diffuse-Tversky/physical-volume loss و
بدون outer. ابتدا smoke چهار-step و بعد یک calibration-only اجرا می‌شود. گیت عبور
هم‌زمان SAH-Dice حداقل `+0.01`، SAH-MAE حداقل `-0.10mL`، checkpoint score حداقل
`+0.001`، عدم‌افت FPR/F1/total-MAE/abs-bias، و invariance دقیق تمام معیارهای چهار
زیرنوع غیرهدف و AUCها تا `1e-10` را می‌خواهد. شکست هر شرط یعنی رد پیش از outer؛
عبور فقط مجوز OOF پنج‌fold recipe قفل‌شده است.

### ۱۳.۶۸ نتیجهٔ exp65: invariance کامل، اما residual از مرز argmax عبور نکرد

کد exp65 در commitهای `a55d413` و launcher بازتولیدپذیر در `0827154` ثبت و روی
سرور fast-forward شد. ۷۸ تست مستقیم segmentation/gate در `10.27s` پاس شدند. smoke
چهار-step و سپس calibration-only کامل اجرا شد؛ run اصلی MLflow برابر
`046aaf1f2ec94ab09cd06d513e390563`، مدت `109.49s`، peak VRAM=`1.028GiB` و تعداد
پارامتر trainable فقط `3217` بود. SHA مدل خروجی
`98d2939d03e096cbf97bed5118a96399b5da89af1b5d1e1013ae01dc958cdae1` است و
`outer_evaluation_performed=false` ماند.

identity اولیه و محدودیت معماری دقیقاً عمل کردند: در epochهای ۱ و ۲ تمام Dice/AUC/
MAE/bias زیرنوع‌های غیرهدف، FPR، F1 و حجم با exp61 بیت‌به‌بیت برابر ماندند. بااین‌حال
هیچ پیکسل calibration از background به SAH عبور نکرد؛ SAH-Dice=`0.053024`،
SAH-MAE=`1.845278mL` و checkpoint score=`0.586668` در هر سه ردیف epoch0/1/2 ثابت
بودند. patience پس از epoch2 متوقف کرد و best epoch صفر ماند. loss آموزشی SAH-Tversky
از `0.23139` در epoch1 به `0.24120` در epoch2 بدتر شد، بنابراین صرفاً ادامهٔ epoch
یا افزایش تصادفی وزن توجیه ندارد.

گیت رسمی فقط چهار شرطِ trained-epoch، SAH-Dice gain، SAH-MAE improvement و score
gain را رد کرد؛ تمام provenance، outer isolation، FPR/volume safety و invarianceهای
معماری پاس شدند. تصمیم `reject_before_outer` است، checkpoint به `checkpoint/ich`
منتقل نمی‌شود و OOF اجرا نخواهد شد. برداشت علّی محتمل این است که head یا گرادیان
نتوانسته margin پس‌زمینه–SAH را تا سقف residual=8 رد کند، یا decoder feature منجمد
برای missed-SAH قابل‌تفکیک نیست. قدم بعدی پیش از هر آموزش تازه باید اندازه‌گیری
تجمیعی margin روی true-SAHهای background و یک update-probe روی خود head باشد؛ فقط
اگر cap/optimization مانع باشد exp66 اصلاح محدود می‌گیرد، وگرنه مسیر frozen adapter
بسته و معماری multi-label مستقل بررسی می‌شود.

### ۱۳.۶۹ postmortem margin exp65: سقف ۸ واقعاً محدودکننده است، اما تنها مانع نیست

diagnostic تجمیعی calibration-only با MLflow run
`cd4e2dc8c71a4010a6aeeff76ed19462` در `11.93s` و peak VRAM=`1.069GiB` کامل شد؛
outer استفاده نشد و هیچ prediction ردیفی persist/ارسال نشد. در چهار مطالعهٔ SAH
مثبت، ۳۱٬۷۲۴ پیکسل true-SAH وجود داشت. مدل exp61 فقط ۸۹۲ را SAH تشخیص داد؛
۱۸٬۸۸۷ پیکسل (`59.54%`) را background و ۱۱٬۵۳۷ پیکسل (`36.37%`) را IPH خواند.
پس adapter background-only از نظر support به بیش از نصف false-negativeها دسترسی دارد،
اما بخش بزرگ IPH-confusion ذاتاً بیرون دامنهٔ آن است.

در true-SAHهای background، margin=`logit(background)-logit(SAH)` دارای q25=`8.125`،
median=`9.719`، q75=`11.344`، q90=`12.688` و q99=`14.5` بود. بنابراین سقف
`8*tanh` exp65 فقط `23.44%` پیکسل‌های eligible و سه مطالعه از چهار مطالعه را حتی در
حالت اشباع نظری می‌توانست برگرداند. cap=12 دسترسی نظری را به `82.15%` و هر چهار
مطالعه می‌رساند، ولی هم‌زمان `6.57%` پیکسل‌های true-background نیز از نظر margin
آسیب‌پذیر می‌شوند؛ cap=16 تقریباً همهٔ SAH را reachable می‌کند اما `98.76%`
پس‌زمینه را هم در صورت اشباع آسیب‌پذیر می‌سازد و غیرقابل‌دفاع است.

این نتیجه شکست exp65 را صرفاً به «feature بد» نسبت نمی‌دهد: cap=8 واقعاً زیر q25
بود. بااین‌حال حتی حدود `23%` reachable هیچ تغییر hard-mask نداد، پس optimization/
گرادیان نیز مشکوک است. افزایش کور cap به 12 ریسک FPR بالایی دارد و هنوز 36% confusion
با IPH را حل نمی‌کند. پیش از exp66 باید gradient norm/cosine objective اصلی و
SAH-only Tversky روی همان ۳۲۱۷ پارامتر head اندازه‌گیری شود. فقط اگر وزن `0.03`
واقعاً سیگنال بسیار ضعیف یا متضاد باشد، cap میانی همراه وزن مبتنی بر گرادیان یک
آزمایش محدود می‌گیرد؛ در غیر این صورت frozen background-expansion بسته می‌شود.

### ۱۳.۷۰ diagnostic گرادیان exp65: سیگنال SAH مادی و تقریباً کاملاً هم‌جهت است

diagnostic train-only در commit=`0845579` و MLflow run=
`4b0ab211fc4b48fbaeaa1d506f8ae8fc` اجرا شد. checkpoint/manifest دقیقاً همان SHAهای
audited exp61 بودند. فقط ۱۲ batch دارای SAH فضایی از train مجاز 0/3/4 استفاده شد؛
برای یافتن آن‌ها ۱۶ batch پیمایش شد، هیچ optimizer step، calibration یا outer اجرا
نشد و هیچ سطر patient/prediction ذخیره یا ارسال نشد. زمان اجرا `8.46s`، peak VRAM
برابر `1.086GiB` و شمار پارامترهای head دقیقاً `3217` بود. ۸۷ تست مسیر SAH پیش از
اجرا پاس شدند.

روی پارامترهای head صفرمقدار، cosine گرادیان loss اصلی segmentation با SAH-Tversky
دارای median=`0.99787` و q10=`0.99273` بود؛ بنابراین دو objective نه‌تنها متعارض
نیستند، بلکه تقریباً یک جهت را پیشنهاد می‌کنند. نسبت خام norm گرادیان SAH به loss
اصلی median=`2.5843` بود و با وزن واقعی `0.03` به median=`0.07753` رسید. وزن متناظر
با سهم هدف ۱۰٪، ۲۵٪ و ۵۰٪ به‌ترتیب medianهای `0.03875`، `0.09686` و `0.19373`
است. میانگین این وزن‌ها به‌علت یک batch با norm اصلی بسیار کوچک outlier بزرگی دارد؛
برای تصمیم recipe از median و quantileها استفاده می‌شود، نه mean.

پس فرض «exp65 به‌خاطر گرادیان بسیار ضعیف یا conflict شکست خورد» رد می‌شود. افزایش
وزن از `0.03` به حدود `0.10` می‌تواند سهم SAH را به حدود ۲۵٪ برساند، اما به‌تنهایی
توجیه کافی برای cap=12 نیست: margin diagnostic نشان داده cap=8 زیر q25 است و
background-only support نیز ۳۶٪ خطاهای SAH→IPH را نمی‌بیند. قدم علّی بعد یک
update-probe کوتاه train-only با optimizer واقعی exp65 است تا norm جابه‌جایی پارامتر،
توزیع residual پس از چند step، fraction اشباع و عبور margin روی SAH مثبت/پس‌زمینه
اندازه‌گیری شود. اگر update با lr فعلی کوچک بماند، فقط یک exp66 محدود با وزن نزدیک
`0.10` و cap میانی/کنترل FPR پیش‌ثبت می‌شود؛ اگر update مادی است ولی support/مرز
تغییر نمی‌کند، مسیر frozen adapter بسته و معماری multi-label مستقل در اولویت قرار
می‌گیرد.

### ۱۳.۷۱ update-probe exp65: optimizer حرکت می‌کند، اما head یک suppressor می‌آموزد

update-probe علّی در commit=`291aade` و MLflow run=
`a7d5c70a6ebb4814a4d6ed65ee63e415` recipe دقیق exp65 را از head صفر برای یک epoch
کامل و فقط روی train بازسازی کرد: AdamW با `lr=5e-4`، weight decay=`1e-4`،
SAH-Tversky=`0.03` و cap=`8`. در ۳۰۳ step، ۲۱۸ batch دارای SAH مثبت بودند. هیچ
calibration/outer خوانده نشد، هیچ checkpoint یا prediction ردیفی ذخیره نشد و فقط
aggregate در MLflow/تلگرام ثبت شد. زمان `48.62s` و peak VRAM=`1.106GiB` بود.

optimizer واقعاً فعال بود: norm تغییر پارامترها `0.58057` یا `12.56%` norm اولیه
شد و mean loss از ربع اول `0.21425` به ربع آخر `0.19245` کاهش یافت. پس صفرماندن
معیارهای exp65 از freeze اشتباه، zero-gradient یا تعداد step کم ناشی نیست. بااین‌حال
روی probe مستقل train با ۱۲ batch SAH مثبت و ۱۲ batch منفی، از ۱٬۶۹۹ پیکسل
true-SAH که incumbent آن‌ها را background می‌دید حتی یک پیکسل به SAH تبدیل نشد.
residual این پیکسل‌ها به‌جای مثبت‌شدن median=`-1.003`، q90=`-0.787` و
q99=`-0.705` داشت. روی ۳۷٬۲۶۵٬۳۷۹ پیکسل true-background و ۳۰٬۹۰۷ پیکسل دیگر
خونریزی نیز تبدیل صفر بود؛ یعنی head محافظه‌کار و ایمن ماند، اما دقیقاً خلاف مأموریت
خود SAH را suppress کرد.

کد loss علت را تأیید می‌کند: `positive_sah_tversky_loss` فقط *ردیف* مثبت را انتخاب
می‌کند، ولی در همان ردیف‌ها `alpha * false_positive` را روی میلیون‌ها پیکسل غیرSAH
جمع می‌زند. در برابر تعداد اندک true-SAH، این فشار همراه loss اصلی، راه کم‌هزینهٔ
کاهش SAH در کل تصویر را غالب می‌کند. cosine تقریباً +۱ diagnostic قبلی اکنون معنای
روشنی دارد: دو loss واقعاً هم‌جهت بودند، اما جهت مشترک در پارامترهای head عمدتاً
سرکوب residual بود، نه recovery.

بنابراین افزایش epoch، lr، وزن همان Tversky یا cap با objective فعلی رد می‌شود.
مداخلهٔ بعدی باید objective را اصلاح کند: یک loss مثبت‌محور واقعی فقط روی پیکسل‌های
true-SAH (برای مثال NLL/margin بین logit SAH و رقیب background) تا recovery را هل
دهد، درحالی‌که loss اصلی segmentation به‌تنهایی false-positiveهای background را
کنترل کند. پیش از calibration، gradient geometry و یک update-probe train-only برای
این loss تازه اجرا می‌شود. اگر head بتواند بخشی معنادار از SAH train را با فشار
background ناچیز برگرداند، exp66 با cap=12 و گیت سخت FPR/SAH/MAE پیش‌ثبت می‌شود؛
در غیر این صورت مسیر adapter background-only بسته خواهد شد.

### ۱۳.۷۲ غربال loss مثبت‌پیکسلی: جهت درست و fit انتخابی train تأیید شد

objective تازه `positive_sah_pixel_nll_loss` برخلاف Tversky فقط در مختصات
spatial-known true-SAH، منفی log-probability کلاس SAH را میانگین می‌گیرد و هیچ
false-positive term از background ندارد. loss اصلی Dice/Focal بدون تغییر مسئول
مهار false-positive است. پیاده‌سازی config/CLI/history/MLflow و دو تست صریحِ
گرادیان SAH و صفر بودن گرادیان background/unknown در commit=`4d947de` ثبت شد؛ کل
مجموعهٔ ۹۵ تست مرتبط در `10.07s` پاس شد.

gradient diagnostic train-only با cap=12 و MLflow run=
`c2354ba9a7684658b568ec0b88af3bc0` روی همان ۱۲ batch مثبت نشان داد objective جدید
واقعاً مسئلهٔ مطلوب را می‌بیند: cosine با loss اصلی median=`-0.99193` است، درحالی‌که
Tversky قدیمی median=`+0.99787` داشت. norm خام objective مثبت‌پیکسلی median=
`36.99×` segmentation بود؛ وزن `0.03` آن را به median=`1.1097×` رساند و cosine
ترکیب با objective جدید median=`+0.8233` شد. وزن متناظر با سهم صرفاً ۲۵٪ median=
`0.00676` است، اما چون دو جهت تقریباً مخالف‌اند، چنین وزن کوچکی هنوز suppressor را
غالب می‌گذارد؛ بنابراین برای probe علّی وزن `0.03`—نقطه‌ای که جهت ترکیبی median را
برمی‌گرداند—ازپیش انتخاب شد، نه sweep calibration.

update-probe یک‌epochی با cap=12، Tversky=0 و positive-pixel=`0.03` در MLflow run=
`8bcf99171a9f4021948448af3eb1ddcd` کامل شد. در ۳۰۳ step، تغییر پارامتر `13.06%`
بود. روی probe ثابت train، ۲۱۹ از ۱٬۶۹۹ true-SAH eligible یعنی `12.89%` واقعاً از
background به SAH برگشتند؛ recipe قدیمی صفر بود. هم‌زمان فقط ۱٬۴۴۹ از
۳۷٬۲۶۵٬۳۷۹ true-background (`0.00389%`) و ۶ از ۳۰٬۹۰۷ پیکسل سایر خونریزی‌ها
تبدیل شدند. residual true-SAH median=`+0.965` و q99=`+1.162`، در برابر background
median=`+0.100` و q99=`+1.023` بود. saturation صفر ماند، پس cap=12 هنوز اشباع نشده؛
نقش آن فعلاً افزایش دامنه/مشتق اولیه و حفظ reachability marginهای بزرگ‌تر است.

این دو diagnostic شرط لازم exp66 را پاس می‌کنند: head frozen توان fit انتخابی دارد و
فشار background در train پایین است. اما این شواهد اثبات generalization نیست. exp66
باید یک recipe واحد و پیش‌ثبت‌شده باشد: warm-start همان exp61، hidden=16، cap=12،
positive-pixel NLL=`0.03`، SAH-Tversky=0، lr=`5e-4`، sampler power=0، outer خاموش
و patience=3. عبور فقط با بهبود هم‌زمان SAH-Dice/SAH-MAE/checkpoint score، عدم‌افت
FPR/F1/total-volume و invariance دقیق چهار subtype غیرهدف مجاز است؛ در غیر این صورت
قبل از OOF رد می‌شود.

### ۱۳.۷۳ نتیجهٔ exp66: recovery ایمن تعمیم یافت، اما اثر برای promotion کوچک بود

launcher پیش‌ثبت‌شده روی commit=`0639fa7` پس از ۹۷ تست سبز، smoke چهار-step و سپس
calibration-only شش‌epochی را کامل کرد. run اصلی MLflow برابر
`639f6535939548d3a0b97da66c7e8871`، زمان `300.00s`، peak VRAM=`1.028GiB` و
پارامتر trainable=`3217` بود. best epoch=3 و SHA checkpoint برابر
`4146926a395b66652623c3900c82d9f1b473b9b16380ee56066c76de830de269` شد؛
`outer_evaluation_performed=false` و تمام checksum/config checks پاس ماندند.

برخلاف exp65، اثر calibration صفر نبود. trajectory SAH-Dice از `0.053024` در epoch0
به `0.054350/0.054731/0.057467` در epochهای ۱/۲/۳ رسید، سپس در epoch4 افت و در
epoch5/6 فقط بخشی از آن را بازیافت. بهترین epoch3 چنین بود:

| معیار calibration1 | exp61 | exp66 best | delta |
|---|---:|---:|---:|
| SAH Dice | 0.053024 | 0.057467 | **+0.004443** |
| SAH MAE | 1.845278mL | 1.837739mL | **-0.007539mL** |
| SAH bias | -1.797290mL | -1.779458mL | +0.017833mL |
| checkpoint score | 0.586668 | 0.587206 | **+0.000538** |
| total-volume MAE | 10.762716mL | 10.756333mL | -0.006383mL |
| normal FPR | 0.194444 | 0.194444 | 0 |
| presence F1 | 0.882353 | 0.882353 | 0 |

EDH/IPH/IVH/SDH Dice/AUC/MAE/bias و Any/macro AUC تا `1e-10` دقیقاً invariant
ماندند؛ total bias نیز `0.01783mL` بهتر شد. بنابراین loss تازه recovery را با آسیب
جانبی نخرید. بااین‌حال گیت سه شکست داشت: SAH-Dice gain فقط `+0.00444` در برابر
`+0.01`، SAH-MAE improvement فقط `0.00754mL` در برابر `0.10mL` و score gain فقط
`+0.000538` در برابر `+0.001` بود. تصمیم رسمی `reject_before_outer` است؛ OOF و
انتقال checkpoint انجام نمی‌شود.

trajectory غیرتک‌نواخت و بهترین epoch3 نشان می‌دهد صرف epoch بیشتر راه‌حل نیست.
cap نیز در update-probe اشباع نشده بود، پس افزایش cap/weight بر اساس feedback همین
calibration به sweep پس‌نگر و ریسک overfit تبدیل می‌شود. نتیجهٔ علّی این است که
objective اصلاح شد اما support معماری هنوز محدود است: adapter فقط background→SAH
را می‌بیند، درحالی‌که postmortem exp61 نشان داد `36.37%` true-SAHها به IPH اشتباه
شده‌اند. قدم بعدی مجاز باید پیش از آموزش، margin و خطر false relabel مسیر IPH→SAH
را به‌صورت aggregate اندازه بگیرد. فقط اگر correct-IPHها از SAH margin ایمن و بخش
قابل‌توجهی از SAHهای اشتباه‌شده reachable باشند، یک relabeler ایزولهٔ
background-or-IPH→SAH پیش‌ثبت می‌شود؛ وگرنه frozen adapter بسته و مدل segmentation
چندبرچسبی مستقل آغاز خواهد شد.

### ۱۳.۷۴ ممیزی margin مسیر IPH→SAH: فرصت واقعی است، اما relabel ساده پرریسک است

diagnostic توسعه‌یافته روی commit=`e4df4bc` پس از ۸۶ تست سبز با MLflow run=
`785d05b2253f4bdcb10cb610d817d895` در `12.48s` و peak VRAM=`1.075GiB` اجرا شد.
فقط aggregate calibration1 ثبت شد، outer خوانده نشد و checkpoint/manifest همان
SHAهای audited exp61 ماندند.

در ۳۱٬۷۲۴ پیکسل true-SAH، تعداد ۱۱٬۵۳۷ پیکسل (`36.37%`) به IPH اشتباه شده بود.
margin=`logit(IPH)-logit(SAH)` برای این خطاها q25=`8.891`، median=`11.688`،
q75=`14.344` و q90=`15.5` بود. cap=8 فقط ۱٬۸۵۴ پیکسل (`16.07%`) و cap=12
تعداد ۶٬۰۰۱ پیکسل (`52.02%`) را از نظر نظری reachable می‌کند؛ این فرصت فقط در دو
مطالعه از چهار مطالعهٔ SAH متمرکز است.

در مقابل، ۱۲۳٬۸۰۳ پیکسل true-IPH که مدل درست IPH تشخیص داده دارای margin q10=
`8.438`، median=`11.469` و q90=`14.188` بودند. اگر residual به‌طور یکنواخت اشباع
شود، cap=8 حدود ۸٬۴۴۲ پیکسل (`6.82%`) و cap=12 حدود ۷۲٬۰۳۸ پیکسل (`58.19%`) از
true-IPHهای درست را در معرض تبدیل غلط به SAH می‌گذارد. بنابراین decision ماشینی
`iph_support_has_high_theoretical_true_iph_relabel_risk` است.

این نتیجه فرصت IPH→SAH را انکار نمی‌کند، ولی relabeler ساده با support گسترده یا
cap=12 را رد می‌کند: نسبت negative درست به positive قابل‌بازیابی بزرگ و تعداد
مطالعات SAH بسیار کم است. ادامهٔ کم‌ریسک فقط در قالب یک probe train-only با negative
control صریح true-IPH و cap محافظه‌کار مجاز است؛ عبور آن باید هم recovery true-SAH
و هم نرخ تبدیل true-IPH را مستقیماً گزارش کند. اگر انتخاب‌پذیری train نیز ضعیف باشد،
شاخهٔ frozen relabel بسته و head چندبرچسبی/دو‌مرحله‌ای با ارزیابی patient-safe جایگزین
می‌شود. calibration دیگری صرفاً برای sweep cap/weight مجاز نیست.

### ۱۳.۷۵ اثبات انتخاب‌پذیری train برای IPH→SAH: امیدبخش، اما لب‌مرزی

probe ازپیش‌ثبت‌شدهٔ background-or-IPH support با cap=`8`، hidden=`16`،
positive-pixel NLL=`0.03` و یک epoch کامل روی train اجرا شد. نخستین اجرای فنی پس از
آموزش به‌علت batch بزرگِ probe نتوانست سهمیهٔ همهٔ strata را پر کند؛ چون هیچ metric،
checkpoint، calibration یا outer از آن تولید نشده بود، فقط batch probe از `16` به
`4` کاهش یافت و recipe آموزش، seed و gateها دست‌نخورده ماند. پس از این اصلاح، ۸۹
تست پاس شد و اجرای موفق MLflow با run=`f9eda9a494724ea39af284da483ec1f1`
در `45.82s`، ۳۰۳ step و peak VRAM=`0.93GiB` کامل شد. شمار پارامترهای trainable
`3217` و تغییر نسبی پارامترها `14.44%` بود.

هر چهار gate train-only پاس شدند: از ۸۰ پیکسل true-SAH که incumbent آن‌ها را IPH
می‌دید، ۹ پیکسل (`11.25%`) بازیابی شد؛ هیچ‌کدام از ۴۳٬۳۰۷ پیکسل true-IPH درست به
SAH تبدیل نشد؛ precision تبدیل‌های IPH-support برابر ۹ از ۱۸، دقیقاً `50%`، و فشار
پس‌زمینه ۹۰۴ از ۱۴٬۱۸۹٬۳۲۸ (`0.006371%`) بود. در کل، ۱۸۷ از ۱٬۱۳۵ پیکسل
true-SAH (`16.48%`) بازیابی شد که ۱۷۸ مورد آن از support پس‌زمینه و ۹ مورد از
support IPH بود.

این نتیجه نشان می‌دهد IPH support الزاماً IPHهای درست را خراب نمی‌کند و head روی
train توان انتخاب‌پذیری دارد؛ بنابراین یک calibration واحد را توجیه می‌کرد. بااین‌حال
precision دقیقاً روی مرز gate و denominator مثبت فقط ۸۰ پیکسل بود، پس شواهد شکننده
و صرفاً شرط لازم بودند، نه اثبات generalization. aggregate رسمی در
`diagnostics/exp67_pre_iph_support_update_epoch1_v1/update_probe.json` ثبت شد و
هیچ خروجی ردیفی یا checkpoint probe نگهداری نشد.

### ۱۳.۷۶ نتیجهٔ exp67: Dice هدف پاس شد، اما بهبود حجمی ناکافی و شاخه بسته شد

exp67 تنها calibration مجاز این شاخه را با warm-start دقیق exp61، base منجمد،
support پس‌زمینه یا IPH، cap=`8`، positive-pixel NLL=`0.03`، lr=`5e-4`، شش epoch،
patience=`3`، batch=`16`، seed=`42` و outer خاموش اجرا کرد. ۹۲ تست پیش از اجرا
پاس شد. MLflow run=`c77d0f7633a14d5c88e3fbc71c3ef1aa` در `294.64s` با
peak VRAM=`1.028GiB` و `3217` پارامتر trainable کامل شد؛ best epoch=`3` و SHA
checkpoint برابر `fdc5bb24c953450deab9577b9d681775f1452e24cbf0ad81d8526fab3756214b`
بود. تمام checksum، provenance و recipe checks پاس شدند و outer خوانده نشد.

| معیار calibration1 | exp61 | exp67 best | delta |
|---|---:|---:|---:|
| SAH Dice | 0.053024 | 0.063168 | **+0.010144** |
| SAH MAE | 1.845278mL | 1.823664mL | **-0.021614mL** |
| checkpoint score | 0.586668 | 0.587852 | **+0.001184** |
| selection score | 0.666162 | 0.667290 | **+0.001128** |
| mean foreground Dice | 0.459106 | 0.461156 | **+0.002050** |
| IPH Dice | 0.674845 | 0.674952 | +0.000107 |
| IPH MAE | 5.170219mL | 5.161328mL | -0.008891mL |
| total-volume MAE | 10.762716mL | 10.755757mL | -0.006959mL |
| normal FPR | 0.194444 | 0.194444 | 0 |
| presence F1 | 0.882353 | 0.882353 | 0 |

SAH Dice و checkpoint score حداقل‌های ازپیش‌ثبت‌شده را پاس کردند؛ IPH نیز اندکی
بهتر شد و EDH/IVH/SDH، Any/macro AUC، FPR و F1 دقیقاً invariant ماندند. شکست یگانه
گیت این بود که SAH MAE فقط `0.0216mL` بهتر شد، نه حداقل `0.10mL`. بنابراین تصمیم
رسمی `reject_before_outer_close_frozen_relabel_branch` است: OOF اجرا نشد، checkpoint
در پوشهٔ accepted محلی قرار نگرفت و sweep پس‌نگر cap/weight/epoch مجاز نیست.

برداشت علّی روشن است: residual head کوچک و منجمد می‌تواند چند پیکسل SAH را با
ایمنی بسیار خوب بازبرچسب‌گذاری کند، اما ظرفیت نمایشی کافی برای اصلاح حجم‌های SAH
ندارد. ادامهٔ علمی باید از relabel محلی عبور کند و به مدل دو‌مرحله‌ای conditional
subtype برسد: مرحلهٔ اول support خونریزی/پس‌زمینهٔ exp61 را حفظ کند و مرحلهٔ دوم با
ظرفیت واقعی، فقط داخل foreground میان پنج subtype—به‌ویژه SAH در برابر IPH—تصمیم
بگیرد. نخست باید یک fit/selectivity proof فقط روی train اجرا شود و تنها در صورت
بهبود معنادار و بدون افت سایر subtypeها، یک calibration patient-safe پیش‌ثبت شود.

### ۱۳.۷۷ نتیجهٔ exp68: ظرفیت relabel اثبات شد، اما train اشباع و دو gate نامعتبر بود

معماری دو‌مرحله‌ای در commit نهایی `e24813c` پیاده شد: incumbent مالک دائمی
foreground/background و تمام classification logits است و کپی decoder/mask-head فقط
داخل hard foreground میان پنج subtype تصمیم می‌گیرد. دو اجرای فنی پیش از optimizer
به‌ترتیب به‌علت mutation درون‌جای featureهای decoder و اختلاف یک پیکسل از
۴۹۶٬۴۳۵٬۲۰۰ پیکسل تحت بازسازی BF16 متوقف شدند؛ هیچ metric، MLflow run، calibration،
outer یا checkpoint از آن‌ها تولید نشد. clone مستقل featureها و حفظ native foreground
logits همراه پایین‌آوردن صرف background، identity دقیق را برقرار کرد.

اجرای موفق train-only با MLflow run=`b0b6fe694a934181b27bb6137d3432f4`، یک epoch،
۳۰۳ step، `2,837,126` پارامتر trainable، زمان `112.49s` و peak VRAM=`1.927GiB`
کامل شد. hard-mask اولیه دقیقاً identity و foreground support نهایی دقیقاً invariant
بود. از ۲۳۲ پیکسل true-SAH که incumbent آن‌ها را IPH می‌دید، ۱۷۲ پیکسل (`74.14%`)
بازیابی شد. آسیب به IPHهای درست ۴٬۷۸۲ از ۹۰۱٬۰۰۰ (`0.531%`)، آسیب به سایر
subtypeهای درست ۴۷ از ۵۴۸٬۳۶۹ (`0.00857%`) و تغییر subtype روی true-background
incumbent-foreground برابر ۳٬۹۵۸ از ۴۲۵٬۸۰۵ (`0.930%`) بود.

بااین‌حال conditional accuracy از `0.992426` به `0.990390` افت کرد
(`-0.2036pp`) و macro recall فقط `+0.0894pp` بهتر شد؛ بنابراین ۶ gate از ۸ gate پاس
و تصمیم رسمی `reject_or_redesign_before_any_calibration` شد. checkpoint ذخیره یا
promote نشد. مهم‌تر اینکه baseline train دارای macro recall=`0.995270` بود و سقف
ریاضی بهبود آن فقط `0.47297pp` است؛ gate ازپیش‌ثبت‌شدهٔ `+1pp` روی این داده اصولاً
قابل‌دستیابی نبود. این gate پس از مشاهدهٔ نتیجه پایین آورده نشد؛ استنتاج درست این بود
که train برای اثبات generalization مناسب نیست. اختلاف ۲۳۲ خطای SAH→IPH در train با
۱۱٬۵۳۷ خطا در calibration قبلی نیز شکاف تعمیم را تأیید کرد.

### ۱۳.۷۸ ممیزی پیش از exp69: پنج checkpoint فولدی بیمارمحور موجود است، ولی final test نیست

پنج checkpoint Unet++/EfficientNet-B2 با outerهای ۰ تا ۴ و `evaluate_outer=false`
ممیزی شدند: exp30 `(outer0, cal1)`، exp32 `(outer1, cal0)`، exp26 `(outer2, cal1)`،
exp34 `(outer3, cal1)` و exp36 `(outer4, cal1)`. split رسمی پروژه برای همهٔ آن‌ها
تقاطع بیمار train/calibration/outer را صفر گزارش کرد. تعداد بیمار train به‌ترتیب
۱۹۳، ۱۹۳، ۱۹۵، ۱۹۵ و ۱۹۴ و تعداد بیمار outer به‌ترتیب ۶۶، ۶۱، ۶۴، ۶۴ و ۶۵ بود.

با جست‌وجوی provenance روشن شد که خود checkpointهای ۰/۱/۳/۴ outerشان را در run
اصلی ندیده‌اند، ولی foldهای outer پروژه قبلاً با مدل‌های دیگر در OOF بررسی شده‌اند.
بنابراین exp69 صریحاً `development_oof_not_final_test_or_leaderboard` تعریف شد، نه
test هرگز‌دیده‌نشده. recipe و ۱۴ gate قبل از inference این suite در
`exp69_conditional_subtype_correction_oof_v1/PREREGISTERED_PLAN.md` قفل شد.

علت علّی هدف تازه از exp68 آمد: CE فقط روی پیکسل‌هایی اعمال شد که incumbent در
foreground قرار داده ولی subtype را غلط گفته بود؛ تمام پیکسل‌های دیگر foreground با
soft-KL به distribution incumbent متصل شدند. برخلاف hard distillation، KL در
identity دقیقاً صفر است. BatchNorm running statistics کپی نیز فریز شد تا drift
ثبت‌نشده ایجاد نکند. square-root class weights، `lr=5e-5`، KL weight=`1`، یک epoch
و پنج outer مستقل ازپیش قفل شدند. ۹۰ تست مرتبط پیش از اجرا پاس شد.

### ۱۳.۷۹ نتیجهٔ exp69: accuracy ظاهراً بهتر شد، اما rare-subtype collapse قطعی است

suite پنج‌فولد روی commit=`2339f04` در `293.13s` با حداکثر peak VRAM=`1.887GiB`
کامل و در MLflow run=`9d3af6f9bda542389046bf355c232fbf` ثبت شد. هر پنج
identity و foreground-support gate دقیقاً پاس و patient overlap هر split صفر بود.
فایل aggregate محلی/سرور SHA256 یکسان
`05de2cc4eb19b0e36518a57e8c5ce69c4bf7a8e3c22def1ee1a1c75762ad7767`
دارد. checkpointهای refiner فقط diagnostic و `not_promoted=true` روی سرور/MLflow
ماندند و به `checkpoint/ich` منتقل نشدند.

نتیجهٔ تجمیعی ظاهراً conditional accuracy را از `0.840349` به `0.842913`
(`+0.2565pp`) رساند، اما macro recall از `0.639079` به `0.532696`
(`-10.638pp`) فرو ریخت. فقط ۱۰۲ از ۲۱٬۲۴۱ خطای SAH→IPH (`0.480%`) بازیابی شد؛
آسیب IPH درست پایین و برابر `0.0505%` بود، ولی ۸۹٬۶۴۰ از ۳۶۹٬۸۸۱ پیکسل صحیح
IVH/SDH/EDH (`24.235%`) و ۸۶٬۳۵۵ از ۵۸۵٬۶۳۴ false-positive پس‌زمینه
(`14.746%`) subtype عوض کردند. accuracy فقط در دو fold و macro recall در هیچ fold
غیرمنفی نبود؛ بدترین افت accuracy=`-1.977pp` و macro recall=`-14.740pp` بود.
تصمیم رسمی `reject_or_redesign_before_full_metric_oof` است؛ Dice/حجم کامل،
calibration، leaderboard و promotion مجاز نیست.

confusion تجمیعی علت افزایش گمراه‌کنندهٔ accuracy را نشان داد: recall IPH از
`92.30%` به `98.69%` و SDH اندکی از `75.64%` به `77.24%` رسید، اما IVH از
`74.28%` به `44.87%`، EDH از `44.79%` به `29.66%` و SAH از `32.53%` به
`15.88%` سقوط کرد. correction loss روی خطاها و KL روی بقیه جداگانه mean گرفته
می‌شدند؛ درنتیجه ۰.۵ تا ۷.۵٪ correction pixels هر fold، برخلاف جمعیت واقعی، جرمی
هم‌اندازهٔ میلیون‌ها stability pixels داشتند. چون IPH بزرگ‌ترین کلاس هدف در خطاها
بود، decoder کامل یک shortcut عمومی به سمت IPH آموخت. کاهش loss و تغییر پارامتر
حدود `2.2–2.46%` در همهٔ foldها نشان می‌دهد failure ناشی از optimizer خاموش نیست؛
بلکه normalization objective و آزادی ۲.۸ میلیون پارامتر علت مستقیم‌اند.

تکرار همان objective با تغییر پس‌نگر یک weight مجاز نیست. ادامهٔ قابل‌دفاع باید یک
آزمایش تازه و ازپیش‌ثبت‌شده باشد که جرم correction و preservation را روی کل جمعیت
foreground نرمال کند یا ظرفیت را به residual کم‌پارامتر/محدود محدود سازد. gateهای
cross-fold exp69 نباید پایین آورده شوند و هر طرح بعدی باید پیش از outer ابتدا
selectivity و trust-region خود را فقط روی train ثابت کند.

### ۱۳.۸۰ نتیجهٔ exp70: trust-region موفق شد، اما selectivity هنوز برای advance کافی نیست

exp70 علت مستقیم شکست exp69 را با دو تغییر ازپیش‌ثبت‌شده آزمود: کپی ۲.۷میلیونی
decoder با یک residual adapter صفرمقدار و فقط `3,285` پارامتر جایگزین شد و correction
CE/preservation KL به‌جای meanهای جداگانه با یک مخرج مشترکِ کل foreground جمعیتی
نرمال شدند. ۹۶ تست سرور پاس شد و اجرای train-only روی commit=`4359e83`، MLflow
run=`680c3bd78f4c4e2ca16d202e2802a3e7`، ۳۰۳ step و یک epoch در `85.70s` با
peak VRAM=`1.185GiB` کامل شد. calibration، outer، test، checkpoint تشخیصی و
prediction ردیفی استفاده یا ذخیره نشد.

قفل معماری دقیق بود: initial hard-mask identity و final foreground support هر دو
صفر mismatch داشتند و classification logits کاملاً از incumbent فریز آمدند. از
۲۳۲ خطای SAH→IPH تعداد ۶۶ پیکسل (`28.448%`) بازیابی شد. آسیب به ۹۰۱٬۰۰۰ IPH درست
فقط `0.1578%`، آسیب به ۵۴۸٬۳۶۹ پیکسل صحیح IVH/SDH/EDH فقط `0.0554%` و تغییر
subtype روی false-positiveهای پس‌زمینه `0.4929%` بود. در مقایسه با exp69، آسیب
other از `24.235%` به `0.0554%` و drift پس‌زمینه از `14.746%` به `0.4929%` رسید؛
پس causal redesign واقعاً collapse را حذف کرد.

بااین‌حال conditional accuracy از `0.992426` به `0.991659` افت کرد
(`-0.07671pp`) و macro recall از `0.995270` به `0.995233` رسید
(`-0.00372pp`). هشت gate از ده gate پاس شدند، اما دو gate non-decreasing accuracy
و macro-recall رد شدند و پس از مشاهدهٔ نتیجه پایین آورده نشدند. تصمیم رسمی
`reject_before_any_calibration_or_outer` است.

این miss کوچک صرفاً مجوز کاهش پس‌نگر lr یا correction weight نیست: کوچک‌کردن update
احتمالاً benefit و harm را با هم جمع می‌کند، درحالی‌که ماتریس confusion نشان می‌دهد
نسبت تصمیم‌های درست‌شده به تصمیم‌های صحیحِ خراب‌شده هنوز زیر یک است. exp70 ثابت کرد
trust-region کم‌پارامتر شدنی است، اما انتخاب‌پذیری feature/target کافی نیست. قدم بعد
باید مکانیزم تصمیم یا نمایش target را تغییر دهد و پیش از هر calibration/outer دوباره
فقط روی train اثبات شود.

### ۱۳.۸۱ نتیجهٔ exp71: gate خطا enrichment داشت، اما precision برای اصلاح ایمن کافی نبود

exp71 مکانیزم تصمیم exp70 را عوض کرد: residual پنج‌کاناله فقط پس از عبور یک gate
نظارتی از آستانهٔ ثابت `0.5` در evaluation اعمال شد. gate روی پیکسل‌هایی هدف یک
داشت که incumbent داخل foreground زیرنوع را غلط گفته بود؛ بقیهٔ foreground هدف صفر
بود. recipe شامل positive-weight=`200`، gate coefficient=`0.25`، همان population
loss exp70، `3,302` پارامتر و یک epoch پیش از اجرا قفل شد. ۱۰۱ تست پاس و run با
MLflow=`a878b2122aa14e2a95f3f8c919c3a2d9` در `133.41s` و peak VRAM=`1.201GiB`
کامل شد؛ calibration/outer/test/checkpoint/prediction ردیفی استفاده نشد.

prevalence خطای incumbent در foreground فقط `0.5832%` بود. gate روی `10.9336%`
کل foreground فعال شد، recall خطا=`40.780%` ولی precision=`2.1753%` داشت؛ یعنی
حدود `3.73×` enrichment نسبت به prevalence، اما بسیار پایین‌تر از gate ازپیش‌ثبت‌شدهٔ
precision=`10%`. از ۲۳۲ خطای SAH→IPH هیچ‌کدام بازیابی نشد. آسیب correct-IPH صفر،
آسیب correct-other=`0.2551%` و drift subtype روی true-background=`0.5392%` بود.
conditional accuracy به‌اندازهٔ `-0.01533pp` و macro recall به‌اندازهٔ
`-0.11039pp` افت کرد. شش gate کیفیت/انتخاب‌پذیری شامل SAH recovery شکست خوردند؛
تصمیم `reject_before_any_calibration_or_outer` است.

gate کاملاً تصادفی نبود، اما برای یافتن خطاهای کم‌شمار نزدیک یک پیکسل از هر نه پیکسل
foreground را فعال کرد و residual را بیشتر به confusionهای IVH/IPH/SDH هدایت کرد،
نه SAH→IPH. sweep پس‌نگر threshold مجاز نیست و مشکل separability را نیز ایجاد
نمی‌کند. مجموعه exp67 تا exp71 نشان می‌دهد adapterهای کوچک روی featureهای decoder
فریز می‌توانند support و FPR را ایمن نگه دارند، اما selectivity و اصلاح حجم subtype
کافی ندارند. شاخه frozen-feature adapter/router بسته می‌شود؛ آزمایش مدل بعدی باید
representation یا supervision مدل پایه را تغییر دهد.

### ۱۳.۸۲ نتیجهٔ exp72: جداسازی foreground ایمن است، اما CE+OVR فعلی IPH را بیش‌تقویت کرد

exp72 نخستین تغییر supervision در سطح مدل پایه پس از بستن شاخهٔ adapter بود. خروجی
شش‌کاناله و inference قبلی حفظ شد، ولی logit دودویی foreground به‌صورت دقیق از
`logsumexp(foreground)-background` ساخته شد؛ sigmoid آن دقیقاً برابر مجموع احتمال‌های
softmax پنج زیرنوع است. Dice/Focal فقط support خونریزی را یاد می‌گیرند و CE پنج‌کلاسه
به‌همراه OVR focal فقط پیکسل‌های true-foreground شناخته‌شده را می‌بینند. ۸۲ تست
سرور identity، backward، حذف گرادیان subtype روی background و قرارداد خروجی را پاس
کردند.

پروب ازپیش‌ثبت‌شده روی commit=`fca1452`، فقط ۲۴ batch train، بدون optimizer و بدون
خواندن calibration/outer/test اجرا و در MLflow run=`b94d6c8ae5b34fdebe2e26ffdcb78cdb`
ثبت شد. هر پنج کلاس بیش از ۸٬۰۰۰ پیکسل داشتند. cosine گرادیان decoder/head برابر
`0.56375` ماند و فشار absolute کانال‌های foreground روی true-background به
`0.61444×` incumbent کاهش یافت؛ پس hierarchy نه‌تنها FPR را ذاتاً رها نکرد، بلکه
جهت background را ایمن‌تر کرد.

اما توزیع subtype نامتعادل بود: IPH=`2.47457×` تقویت شد، درحالی‌که IVH=`0.77235×`،
SDH=`0.38803×` و EDH=`0.71413×` شدند. SAH فقط `1.09739×` بود و به gate=`1.25×`
نرسید؛ EDH نیز gate=`1.10×` را رد کرد. بنابراین weighting دقیق
`0.40 foreground Dice + 0.20 foreground focal + 0.30 conditional CE + 0.10 OVR`
با تصمیم `reject_exact_loss_weighting_before_calibration_or_outer` بسته شد.

این نتیجه hierarchy را رد نمی‌کند: دو گیت structural مهم آن پاس شدند. شکست نشان
می‌دهد OVR مستقل و CE شرطی در وضعیت فعلی عمدتاً روی خطاهای IPH-محور incumbent جرم
می‌گذارند و صرف حذف background، rare-subtype balance را تضمین نمی‌کند. ادامه باید
ابتدا سهم گرادیان CE و OVR را جدا کند و سپس خود objective زیرنوع را تغییر دهد؛ پایین
آوردن پس‌نگر gate یا صرفاً افزایش ضریب subtype مجاز و علمی نیست.

### ۱۳.۸۳ نتیجهٔ exp73: Balanced Softmax، EDH را نجات داد ولی علت IPH dominance عمیق‌تر است

exp73 به‌جای افزایش کور ضریب subtype، OVR مستقل را حذف و CE شرطی را با Balanced
Softmax جایگزین کرد. priorها فقط از پیکسل‌های mask‌شدهٔ train آمدند:
IVH=`207492`، IPH=`1038893`، SDH=`310139`، EDH=`72234` و SAH=`70484`.
recipe پیش از اجرا روی `0.40 Dice foreground + 0.20 focal foreground + 0.40
Balanced Softmax + 0 OVR` قفل شد. ۸۵ تست پاس و همان ۲۴ batch train-only، بدون
optimizer/held-out، در MLflow run=`71c281f9852940d083109911c1a7804f` ثبت شد.

EDH از نسبت exp72=`0.714×` به `1.384×` رسید و gate خود را پاس کرد؛ این شاهد مستقیم
است که prior correction برای یک tail class اثر مورد انتظار داشت. اما SAH=`0.939×`،
IVH=`0.648×` و SDH=`0.429×` شدند و IPH همچنان `2.541×` بود. گیت‌های structural
دوباره پاس شدند: background gradient=`0.614×` و decoder/head cosine=`0.543`.
بنابراین تصمیم رسمی `reject_exact_loss_weighting_before_calibration_or_outer` است.

توضیح علّی تازه این است که حتی با subtype loss متوازن، foreground binary loss از
`logsumexp` عبور می‌کند و مشتق آن میان پنج کانال متناسب با probability فعلی پخش
می‌شود. پس فشار support، margin زیرنوع‌ها را ناخواسته به‌نفع subtype غالب فعلی—اغلب
IPH—تغییر می‌دهد. علاوه‌براین، absolute target-channel gradient که در exp72/73
اندازه‌گیری شد common shift و subtype-margin را مخلوط می‌کند. آزمایش بعد باید forward
دقیق را نگه دارد، ولی backward foreground را به زیرفضای common-mode پنج subtype
محدود کند و گیت را روی گرادیان margin واقعی target-vs-other تعریف کند.

### ۱۳.۸۴ نتیجهٔ exp74: common-mode، coupling را حذف کرد ولی subtype Dice لازم است

exp74 forward دقیق probability را نگه داشت اما backward foreground را common-mode
کرد: مشتق هر پنج foreground logit دقیقاً `0.2` و مشتق background برابر `-1` است؛
پس هیچ foreground objective نمی‌تواند margin زیرنوع‌ها را عوض کند. معیار اصلی نیز
برای نخستین بار signed subtype-margin attraction شد. ۸۶ تست پاس و ۲۴ batch
train-only روی commit=`8a570ca` در MLflow run=`7e45b70c0d174c18b13f6a31b7cc7eb2`
ثبت شد؛ optimizer و held-out صفر بودند.

این جداسازی اثر علّی مورد انتظار داشت: absolute target-gradient IPH از `2.541×`
exp73 به `0.988×` رسید. background ratio=`0.614×` و decoder/head cosine=`0.435`
ایمن ماندند. اما روی معیار درست margin، فقط EDH با `1.299×` پاس شد؛ SAH=`1.022×`،
IVH=`0.666×`، IPH=`0.701×` و SDH=`0.114×` بودند. تصمیم رسمی
`reject_exact_loss_weighting_before_calibration_or_outer` است.

بنابراین common-mode به‌عنوان مکانیزم decoupling موفق حفظ می‌شود، ولی Balanced
Softmax CE تنها برای subtype کافی نیست. افت شدید SDH و عدم رشد SAH نشان می‌دهد حذف
Dice چندکلاسه، فشار overlap/morphology کلاس‌های diffuse را نیز حذف کرده است. گام
بعد باید conditional subtype Dice را فقط داخل true-foreground اضافه و Balanced
Softmax را با focal difficulty تعدیل کند؛ background همچنان نباید وارد رقابت
زیرنوع شود.

### ۱۳.۸۵ نتیجهٔ exp75: افزودن conditional Dice کافی نبود؛ loss-only branch بسته شد

exp75 common-mode را حفظ کرد و morphology زیرنوع را با `0.35 conditional Dice`
برگرداند؛ Balanced Softmax نیز gamma=`2` گرفت تا خطاهای سخت وزن بیشتری بگیرند.
باقی recipe شامل foreground Dice=`0.30`، foreground focal=`0.15`، Balanced Focal
CE=`0.20` و OVR=`0` بود. ۸۶ تست پاس و ۲۴ batch train-only روی commit=`f4e0039`
در MLflow run=`8d71c4c4a9d14f6fb733579af6bd0d1c` ثبت شد.

background pressure به `0.462×` کاهش یافت و decoder/head cosine=`0.510` مثبت بود؛
IPH margin نیز `0.959×` کنترل شد. بااین‌حال IVH=`0.538×`، SDH=`0.059×`،
EDH=`0.470×` و SAH=`0.911×` شدند. درنتیجه gateهای morphology/rare subtype شکست و
تصمیم `reject_exact_loss_weighting_before_calibration_or_outer` ثبت شد.

exp72 تا exp75 اکنون شواهد کافی برای بستن شاخهٔ loss-only روی head شش‌کلاسهٔ موجود
می‌دهند. hierarchy از نظر FPR ایمن است، ولی با همان representation/head، بازتوزیع
objective به‌تنهایی rare/diffuse margins را بهتر نمی‌کند. گام بعد باید معماری خروجی
را فاکتور کند: `P(foreground) × P(subtype|foreground)` با identity دقیق احتمال‌های
شش‌کلاسه در initialization، و decoder قابل‌آموزش برای تغییر representation.
