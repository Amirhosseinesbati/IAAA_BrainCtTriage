# گزارش جامع توقف و تحویل توسعه مدل شکستگی — ۳۱ اوت ۲۰۲۶

## حکم اجرایی

توسعه فعال مدل شکستگی بنا به درخواست کاربر **متوقف (paused)** شده است؛ Goal نه
`complete` و نه `blocked` شده و همچنان active باقی مانده است. هیچ مسیر
triage-aware در این مرحله ادامه داده نمی‌شود. سرور Vast.ai شماره `49251973`
روشن مانده، حذف یا متوقف نشده و در لحظه تحویل هیچ job فعال GPU ندارد.

بهترین مدل قابل استفاده فعلی بسته پنج‌fold `YOLOv8s + SA-MIL fixed-0.45` است.
نسخه snapshot رتبه‌بندی بهتری دارد، ولی تصمیم دودویی شکستگی را عمداً عوض
نمی‌کند؛ بنابراین مدل اصلی را جایگزین نمی‌کند و فقط artifact پژوهشی معتبر است.

## ۱. وضعیت زیرساخت و قابلیت ادامه

- Vast instance: `49251973`، GPU خانواده RTX 3060 با 12GB VRAM.
- دیسک: 50GB؛ هنگام تحویل 26GB مصرف و حدود 25GB آزاد.
- repository سرور: `/workspace/IAAA_BrainCtTriage`.
- branch محلی و سرور: `codex/competition-winning-pipeline`.
- HEAD محلی و سرور: `dc61773`.
- محیط Python سرور: `/workspace/IAAA_BrainCtTriage/.venv`.
- فایل محرمانه MLflow فقط روی سرور:
  `/workspace/.secrets/iaaa_fracture_mlflow.env`.
- wrapper اجرای امن MLflow:
  `scripts/run_with_fracture_mlflow_env.sh`.
- وضعیت GPU هنگام تحویل: صفر process محاسباتی.
- هیچ داده per-study، prediction خصوصی یا آرایه calibration به MLflow عمومی
  ارسال نشده است؛ فقط متریک‌ها و artifactهای تجمیعی امن ثبت شده‌اند.

## ۲. پروتکل اعتبارسنجی که باید حفظ شود

مجموعه شامل 338 مطالعه، 320 بیمار و فقط 28 مطالعه fracture-positive است. به
دلیل کمی نمونه مثبت، نتیجه یک fold به‌تنهایی بسیار ناپایدار است. تصمیم‌های نهایی
این مرحله بر پایه پنج outer fold بیمارمحور، پیش‌بینی OOF، انتخاب threshold بدون
دیدن fold نگه‌داشته‌شده و paired stratified bootstrap گرفته شدند.

اصول اجباری ادامه کار:

1. split در سطح بیمار حفظ شود؛ مطالعه‌های یک بیمار نباید بین train/validation
   پخش شوند.
2. مدل/epoch/pooling/threshold با داده همان fold ارزیابی انتخاب نشود.
3. box mAP فقط معیار detector است؛ معیار پذیرش، study-level AUC و F1 است.
4. مدل نهایی باید علاوه بر OOF، parity بسته، DICOM preprocessing، runtime و
   failure policy را بگذراند.
5. امتیاز حدود 0.914 لیدربرد، Macro-F1 نهایی تریاژ است و با AUC شکستگی قابل
   مقایسه مستقیم نیست.

## ۳. مسیر آزمایش‌ها و نتایج اصلی

| آزمایش | شاهد معتبر | نتیجه | تصمیم |
|---|---:|---:|---|
| YOLOv8s قدیمی، fold 0، max pooling | AUC `0.890625`، F1@0.5 `0.384615` | ranking مفید، calibration ضعیف | baseline قدیمی |
| detectorهای جدید COCO/positive-repeat | بهبود روی foldهای مستقل | recipe معتبر شد | پایه ensemble |
| SA-MIL تنها، پنج‌fold | macro AUC `0.845291`، worst `0.773438` | از detector مرجع ضعیف‌تر | به‌تنهایی رد |
| detector adjacent-pair مرجع | macro AUC `0.873702`، worst `0.791569` | baseline OOF جدید | نگه‌داری |
| fixed 0.45 detector + SA-MIL | macro AUC `0.907808`، worst `0.853175` | بهبود معنادار عملی | **incumbent** |
| threshold cross-fit بسته اصلی | F1 `0.548387`، P/R=`0.50/0.607143` | بهتر از reference F1=`0.40` | پذیرفته |
| hard-negative mining روی fold 4 | AUC حدود `0.8454` در برابر incumbent `0.8735` | افت روی held-out fold | رد |
| snapshot epoch10+15، fusion مستقیم | AUC `0.916479` ولی F1 `0.394366` | ranking بهتر، تصمیم بدتر | رد به‌عنوان classifier |
| snapshot decision-preserving | AUC cross-fit `0.917025`، F1 `0.548387` | تصمیم‌ها دقیقاً برابر incumbent | فقط ranking research |
| YOLOv8m fold 1 replication | epochهای 5/10/15 غربال شدند؛ بهترین AUC=`0.849206` در epoch 5 | complement فقط `+0.007937` با CI شامل صفر | رد؛ بدون گسترش پنج‌fold |

### معنی آماری بسته اصلی

- افزایش macro AUC نسبت به مرجع: `+0.034106`.
- paired bootstrap 20,000 تکرار برای اختلاف AUC:
  `[-0.005030, 0.033979, 0.074528]`؛ احتمال بهتر نبودن `0.0436`.
- افزایش cross-fit F1 نسبت به مرجع: `+0.148387`.
- bootstrap اختلاف F1:
  `[0.023125, 0.148712, 0.271121]`؛ احتمال بهتر نبودن `0.01075`.
- threshold deployment از OOF: `0.8363212059916904` که با نگاشت piecewise
  به cutoff رسمی احتمال `0.5` وصل می‌شود.

این شواهد برای انتخاب بهترین مدل آفلاین فعلی کافی است، اما جای public/private
leaderboard را نمی‌گیرد.

## ۴. مدل‌های منتقل‌شده به سیستم محلی

### ۴.۱ کاندید اصلی

مسیر:
`checkpoint/ich/fracture-yolov8s5fold-sa-mil-fixed045-v2-20260831`

- 5 detector از نوع YOLOv8s و 15 head کوچک SA-MIL؛ در مجموع 20 artifact.
- مجموع اندازه artifactها: `114,952,642` بایت.
- تمام 20 فایل در زمان تحویل با manifest دوباره hash شدند: `0` خطا.
- SHA-256 خود manifest:
  `d11644075e0276e6b6053efc10effc3366b48ff2272904e6cc769e3e881625d8`.
- OOF macro AUC=`0.907808`؛ worst-fold=`0.853175`.
- cross-fit F1=`0.548387`.
- parity بسته optimizer-stripped پاس شده است.
- benchmark DICOM end-to-end حدود `5.5 s/study` و peak VRAM حدود `1.9GB`
  گزارش شده؛ زیر محدودیت اجراست.

نسخه سرور:
`/workspace/IAAA_BrainCtTriage/experiments/fracture_mil_package/fixed045_v2_stripped`

### ۴.۲ بسته snapshot برای پژوهش ranking

مسیر:
`checkpoint/ich/fracture-yolov8s5fold-sa-mil-snapshot1015-fixed040-v1-20260831`

- 10 detector (epoch 10 و 15 برای هر fold) و 15 head SA-MIL؛ 25 artifact.
- مجموع اندازه artifactها: `227,559,592` بایت.
- تمام 25 فایل در زمان تحویل با manifest دوباره hash شدند: `0` خطا.
- SHA-256 manifest:
  `bc5c4efa1f924342063042170bb2ab5cc99d24719dd302a15048a8f68c717ed2`.
- وضعیت رسمی artifact:
  `validated_ranking_research_no_direct_triage_gain`.
- macro AUC decision-preserving=`0.917025`، اما F1 همان `0.548387` است.
- parity عددی DICOM روی نمونه‌های کنترل دقیقاً پاس شد؛ mean runtime حدود
  `5.51 s/study` و peak VRAM حدود `1.91GB`.
- MLflow aggregate run: `26204a8dfee44841915e7336ba05f1e7`.

نسخه سرور:
`/workspace/IAAA_BrainCtTriage/experiments/fracture_mil_package/snapshot_fixed040_v1_stripped`

این بسته نباید به‌عنوان A/B مستقل لیدربرد معرفی شود، چون binary decision آن با
incumbent یکسان است.

## ۵. مقایسه دقیق با `checkpoint/fracture`

پوشه قدیمی فقط دو فایل دارد و هیچ README، manifest، config مستقل یا گزارش کنار
وزن‌ها ندارد:

| فایل | SHA-256 | متادیتای بازیابی‌شده |
|---|---|---|
| `best.pt` | `97DC77BE75AF4B4EF689ADBD876839ED3C4E96FAA4350B8FA3C9B3F7EF1CDEF2` | YOLOv8s، fold 0، 512px، 150 epoch |
| `last.pt` | `06E072542A2D880AB4C014FA86F6AA3959B6825A5DDFCDED171933BDE1EB2187` | همان run؛ آخرین وزن نه بهترین وزن |

متادیتای داخلی نشان داد `best.pt` دارای precision=`0.59065`، recall=`0.53226`،
mAP50=`0.54160` و mAP50-95=`0.22197` است. `last.pt` ضعیف‌تر است:
precision=`0.59986`، recall=`0.48387`، mAP50=`0.45754` و
mAP50-95=`0.17494`. بنابراین `last.pt` کاندید انتقال یا ensemble نیست.

### مقایسه apples-to-apples روی fold 0

هر دو نتیجه زیر روی همان 70 مطالعه fold 0 با 6 مثبت محاسبه شده‌اند:

| معیار | best.pt قدیمی | بسته اصلی جدید | بهبود مطلق |
|---|---:|---:|---:|
| Study-level AUC | `0.890625` | `0.919271` | `+0.028646` |
| Study-level F1 | `0.384615` در cutoff 0.5 | `0.500000` با threshold انتخاب‌شده از چهار fold دیگر | `+0.115385` |
| TP / TN / FP / FN | `5 / 49 / 15 / 1` | `3 / 61 / 3 / 3` | FP از 15 به 3 کاهش یافت |

تفاوت F1 یک trade-off روشن دارد: مدل جدید false-positive را شدیداً کم کرده، اما
در fold کوچک 3 مثبت را از دست داده است. با فقط 6 مثبت، fold 0 به‌تنهایی برای
نتیجه قطعی کافی نیست؛ مزیت اصلی مدل جدید این است که برخلاف مدل قدیمی، شاهد کامل
پنج‌fold دارد:

paired stratified patient-cluster bootstrap با 50,000 تکرار این احتیاط را کمی
دقیق‌تر کرد. fold 0 شامل 70 مطالعه، 66 بیمار و فقط 6 بیمار مثبت است:

- اختلاف AUC نقطه‌ای `+0.028646`، CI 95٪ برابر
  `[-0.044444, 0.026882, 0.110215]` و احتمال بهتر نبودن candidate=`0.24212`.
- اختلاف F1 نقطه‌ای `+0.115385`، CI 95٪ برابر
  `[-0.252964, 0.111111, 0.408602]` و احتمال بهتر نبودن candidate=`0.26774`.
- MLflow aggregate-only run: `3dc6dfa7c9514102908ea77c4f55fec4`؛ هیچ
  prediction per-study آپلود نشد.

در نتیجه point estimate مدل جدید بهتر است، اما هیچ‌یک از دو اختلاف روی fold 0
به‌تنهایی از نظر آماری قطعی نیست. ادعای پیشرفت بر مجموعه شواهد پنج‌fold، بهبود
worst-fold و calibration cross-fit تکیه دارد، نه CI این fold کوچک.

- جدید: 338 مطالعه OOF، macro AUC=`0.907808`، worst-fold=`0.853175`،
  cross-fit F1=`0.548387`.
- قدیمی: فقط fold 0 معتبر و مستقل؛ تعمیم پنج‌fold اثبات نشده است.

بنابراین نتیجه منصفانه این است که **پیشرفت واقعی و قابل‌اندازه‌گیری داریم**، به‌ویژه
در کنترل false-positive و اعتبارسنجی بین‌fold؛ ولی هنوز نمی‌توان از این اعداد نتیجه
گرفت که Macro-F1 نهایی لیدربرد از 0.914 عبور می‌کند.

## ۶. آزمایش YOLOv8m که متوقف شد

- run name:
  `fracture-v2-posr4-f1-coco-y8m-lr2p5e4-replication`.
- MLflow run: `7de4238bd0b447edbfbd30722da1ae4f`.
- برنامه اولیه: 60 epoch، batch=8، image=512، LR=`2.5e-4`، patience=15.
- آموزش هنگام epoch 20 با `Ctrl+C` متوقف شد؛ GPU به‌طور کامل آزاد شد.
- MLflow پس از interrupt خودکار وضعیت `FAILED` گرفت؛ این وضعیت درست است و به
  `FINISHED` جعل نشد.
- آخرین checkpoint کامل: `epoch15.pt`؛ hash:
  `57807acbff35e6f40e728165e1e623f183e9bf264d5373f7288ff510e364c164`.
- `best.pt` فعلی همان hash epoch 15 را داشت.
- پس از توقف، تمام checkpointهای کامل به‌صورت یکسان روی 67 مطالعه fold 1 و 1346
  اسلایس غربال شدند. بهترین AUC هر checkpoint به‌ترتیب بود:
  epoch 5=`0.849206` (noisy-or)، epoch 10=`0.654762` (noisy-or) و
  epoch 15=`0.591270` (max). بنابراین وزن `best.pt` مبتنی بر box-fitness، بهترین
  مدل study-level نبود و آموزش طولانی‌تر در این run افت شدیدی ایجاد کرده است.
- آزمون complement ازپیش‌تعریف‌شده با وزن ثابت `0.3` روی epoch 5 فقط AUC را از
  `0.873016` به `0.880952` رساند: اختلاف `+0.007937`، bootstrap 50,000 تکرار
  `[-0.055556, 0.007937, 0.071429]` و احتمال بهتر نبودن `0.4079`. این شواهد
  برای هزینه پنج‌fold کافی نیست؛ خانواده YOLOv8m فعلی رد شد.
- MLflow run ارزیابی aggregate-only:
  `59d849caf84d45679c71e8c502e6d1b5`. هیچ prediction per-study در آن ثبت نشد.
- checkpoint به سیستم محلی منتقل نشد، چون فقط یک fold است و برتری قابل اتکا
  نسبت به incumbent اثبات نکرده است.

مسیر checkpoint روی سرور:
`/workspace/IAAA_BrainCtTriage/runs/detect/experiments/fracture_v2_coco_f1_y8m_replication/fracture-v2-posr4-f1-coco-y8m-lr2p5e4-replication/weights/epoch15.pt`

## ۷. کارهای ردشده و دلیل رد

- SA-MIL مستقل: complementary بود، اما به‌تنهایی از detector ضعیف‌تر شد.
- hard-negative mining فعلی: روی held-out fold افت کرد؛ صرفاً به‌خاطر ایده خوب
  ادامه داده نشد.
- snapshot direct calibration: AUC بالاتر ولی F1 پایین‌تر؛ ranking با decision
  quality اشتباه گرفته نشد.
- YOLOv8m: diagnostic اولیه نشانه complement داشت، اما CI گسترده و فقط یک fold
  بود؛ از گسترش پرهزینه جلوگیری شد.
- 2.5D اولیه: شاهد بهبود مستقل کافی نداشت؛ معماری فقط به اتکای مقاله پذیرفته نشد.
- بهینه‌سازی triage-aware threshold/gate: طبق درخواست کاربر به مرحله بعد موکول
  شد. patch محلی نیمه‌تمام آن کاملاً برگشت داده شد و فایل مربوطه نسبت به HEAD
  هیچ diff محلی ندارد.

## ۸. جمع‌بندی پژوهش مرتبط

پژوهش SciSpace سه جهت را تقویت کرد، اما هیچ‌کدام بدون آزمون مستقل پذیرفته نشد:

1. detection و segmentation هر دو برای شکستگی جمجمه کاربرد دارند و segmentation
   می‌تواند sensitivity بهتری بدهد؛ Frontiers in Neurology 2021،
   DOI `10.3389/fneur.2021.687931`.
2. 2.5D برای پیوستگی بین‌اسلایسی معقول است، ولی شواهد منتشرشده کوچک و
   dataset-dependent است؛ Applied Sciences 2025، DOI `10.3390/app16010147`.
3. در داده پزشکی نامتوازن، calibration و ارزیابی leakage-free به‌اندازه AUC مهم
   است؛ PLOS ONE 2022، DOI `10.1371/journal.pone.0262838`.

اثر عملی این پژوهش روی تصمیم‌ها: تمرکز از تعویض کور معماری به pooling سطح study،
calibration OOF، ensemble مکمل و کنترل false-positive منتقل شد.

## ۹. insight تریاژ که فعلاً فقط ثبت و defer شد

تحلیل oracle نشان داد fracture فقط در 6 مورد از 28 مطالعه مثبت تصمیم تریاژ را
تغییر می‌دهد؛ این موارد fracture همراه با حجم کل خونریزی حداقل 15mL بودند. این
یافته برای مرحله ادغام نهایی با ICH و MLS مهم است، اما در این مرحله به مدل
شکستگی یا threshold آن تزریق نمی‌شود. نتیجه triage-aware فعلی نباید معیار انتخاب
مدل fracture باشد و هیچ آزمایش جدیدی بر پایه آن شروع نشده است.

## ۱۰. نقطه شروع پیشنهادی در ادامه بعدی

هنگام resume، ابتدا هیچ آزمایش جدیدی اجرا نشود تا این سه بررسی انجام شوند:

1. صحت سرور، branch و hash دو package محلی/سرور دوباره بررسی شود.
2. incumbent ثابت بماند و هر candidate با paired OOF در برابر آن سنجیده شود.
3. فقط candidateای به پنج‌fold گسترش یابد که در یک replication از پیش‌تعریف‌شده
   بهبود F1 یا دست‌کم CI امیدوارکننده نشان دهد.

اولویت‌های fracture-only پیشنهادی:

- pooling قابل‌آموزش با هدف study-level و hard-negative curriculum کنترل‌شده؛
- segmentation/localization auxiliary head فقط پس از آزمون محدود و پیش‌ثبت‌شده؛
- snapshot/seed diversity با gate مستقیم F1، نه AUC صرف؛
- سپس packaging و leaderboard واقعی، و تازه بعد از آماده‌شدن ICH و MLS ادغام
  triage-aware.

فرمان‌های مرجع برای بررسی ادامه:

```bash
cd /workspace/IAAA_BrainCtTriage
git status --short
git rev-parse --short HEAD
nvidia-smi
df -h /workspace
```

آزمایش YOLOv8m عمداً auto-resume نشده است و با شواهد تکمیلی بالا نباید از
epoch 15 ادامه یابد. اگر در آینده دوباره بررسی شود، باید recipe یا objective
انتخاب checkpoint عوض شود و یک replication از پیش‌ثبت‌شده جدید ساخته شود؛ ادامه
همین trajectory توجیه ندارد.

## ۱۱. فایل‌های مرجع

- `reports/fracture_experiments/mil/deployable_fixed045_v2/summary.json`
- `reports/fracture_experiments/mil/threshold_crossfit_fixed045_v2/summary.json`
- `reports/fracture_experiments/mil/oof_v2/summary.json`
- `reports/fracture_experiments/mil/SNAPSHOT_FUSION_DECISION_20260831.md`
- `reports/fracture_experiments/mil/y8m_f1_replication_final_screen/epoch5_metrics.json`
- `reports/fracture_experiments/mil/y8m_f1_replication_final_screen/epoch10_metrics.json`
- `reports/fracture_experiments/mil/y8m_f1_replication_final_screen/epoch15_metrics.json`
- `reports/fracture_experiments/mil/y8m_f1_replication_final_screen/fixed030_rank_replication_summary.json`
- `reports/fracture_experiments/mil/legacy_vs_incumbent_paired_patient_bootstrap_fold0.json`
- `reports/checkpoint_evaluation/checkpoint_evaluation_report_fa.md`
- `reports/checkpoint_evaluation/fold_0_metrics.json`

## نتیجه نهایی

مدل جدید نسبت به وزن قدیمی `checkpoint/fracture/best.pt` فقط «پیچیده‌تر» نشده؛
روی fold مشترک AUC و F1 بهتر شده، false-positive به‌شدت کاهش یافته و مهم‌تر از
آن، اکنون شواهد کامل پنج‌fold، calibration cross-fit، بسته قابل اجرا، parity و
manifest قابل audit داریم. بااین‌حال ادعای برد لیدربرد زودهنگام است. وضعیت درست
پروژه در این لحظه: **یک incumbent شکستگی معتبر و قابل‌استفاده، یک artifact پژوهشی
ranking، چند مسیر ردشده مستند، سرور روشن، GPU آزاد و Goal فعال ولی paused**.
