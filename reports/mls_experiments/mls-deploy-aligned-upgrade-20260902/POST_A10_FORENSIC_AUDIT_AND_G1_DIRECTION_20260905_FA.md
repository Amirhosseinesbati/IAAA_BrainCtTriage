# گزارش forensic پس از A10 و جهت علمی بعدی MLS

> وضعیت: تصمیم طراحی؛ **هیچ‌یک از مدل‌های A9/A10 release یا submission-eligible نیستند.**
>
> تاریخ: 2026-09-05 (Asia/Tehran)

## نتیجهٔ اجرایی

پاسخ کوتاه به «چرا هنوز مدل بهتری نداریم؟» این نیست که داده یا ظرفیت مدل تمام شده است. شواهد نشان می‌دهد که خانوادهٔ محدودِ A9/A10 در یک جهت خاص به بازده نزولی رسیده: دقت پیوستهٔ MLS را بهتر می‌کند، اما تصمیم حساس در آستانهٔ `3 mm` را بدتر می‌کند. ادامهٔ همان refiner یا تغییر وزن loss، پژوهش تازه نیست و نباید انجام شود.

در عین حال، هنوز هیچ شاهد full-OOF پنج‌فولد و deploy-aligned نداریم که نشان دهد یک MLS جدید، با ثابت نگه‌داشتن ICH و fracture، واقعاً `Macro-F1` نهایی triage را بهتر کرده است. بنابراین عبارت «مدل بهتر» تا پیش از آن فقط دربارهٔ proxyهای MLS است، نه رتبهٔ مسابقه.

جهت موجه بعدی `MLS-G1` است: مشاهدهٔ anatomy محلی با context سه‌برشی/2.5D و landmarkهای موجود؛ نه pseudo-label کردن سطح midline یا یک sweep دیگر روی pooling، threshold، یا loss refiner.

## آنچه با اطمینان می‌دانیم

### 1. معیار نهایی و محل اثر MLS

- معیار رسمی مسابقه `Macro-F1` سه کلاس final triage است. `Urgent-F1` معیار رسمی جداگانه نیست، اما gate داخلی ضروری ما است تا بهبود ظاهری Macro-F1 با آسیب به کلاس Urgent پذیرفته نشود.
- از هفت intermediate، MLS در rule قطعی triage وارد می‌شود. عبور از `3 mm` می‌تواند Normal را Urgent کند و با ICH/volume مناسب به Critical برسد؛ عبور از `5 mm` در حضور ICH یا fracture می‌تواند Urgent را Critical کند.
- در truth 338 مطالعه‌ای، 37 مطالعه مستقیماً در band `MLS 3–5 mm` هستند، 28 مورد در `2.5–3.5 mm` و 27 مورد در `4–6 mm` قرار دارند. پس آستانه‌ها کم‌اهمیت یا صرفاً نظری نیستند.
- توزیع کلاس نهایی `165 Normal / 60 Urgent / 113 Critical` است. اندازهٔ کلاس Urgent و تعداد محدود borderline caseها علت اصلی نوسان زیاد F1 روی یک fold 70تایی‌اند؛ به همین دلیل screen تک‌fold فقط برای reject زودهنگام است، نه اثبات promotion.

شواهد: `Competition-Guide/iaaa-competition-2026-brain-ct-triage-challenge.pdf`، `src/evaluation/triage.py`، `reports/eda/eda_triage_simulation_results.json` و `scripts/evaluate_mls_deploy_aligned_seed_medians.py`.

### 2. نتیجهٔ واقعی A9/A10

هر دو آزمایش baseline واجدشرایط را freeze کردند و فقط outer reference refiner کوچک را آموزش دادند. A10 روی screen CUDA ثابت fold0/seed42 نسبت به baseline این تغییرها را داشت:

| معیار MLS | baseline | A10 | نتیجه |
|---|---:|---:|---|
| MAE (mm) | 1.470565 | **1.441921** | بهتر |
| F1@5 mm | 0.736842 | **0.810811** | بهتر |
| Boundary-F1 | 0.778257 | **0.798848** | بهتر |
| objective | 1.914051 | **1.844225** | بهتر |
| F1@3 mm | **0.819672** | 0.786885 | بدتر؛ gate fail |

بنابراین این‌ها failure بدون signal نیستند: localization و مرز 5mm قابل بهبود بوده‌اند. ولی خروجی برای مرز 3mm دقیقاً آن چیزی نیست که triage نیاز دارد. A10 نسبت به A9 فقط `0.006212 mm` MAE بهتر شد و هیچ F1 مرزی را تغییر نداد؛ یعنی تغییر loss فعلی دیگر اطلاعات یا رفتار تصمیم‌گیری جدیدی تولید نمی‌کند.

**تصمیم قطعی:** A9/A10 نه package، نه submission و نه replication ندارند. tuning پس از دیدن fold0، overfit به همان fold خواهد بود.

شواهد: `A9_PAIRED_RESULT_20260904.md`، `A9_THRESHOLD_DIAGNOSTIC_20260905.json` و `A10_EXPLORATORY_EVALUATION_REPORT_20260905_FA.md`.

### 3. ظرفیت واقعی داده برای یک مسیر متفاوت

داده برای یک معماری متفاوت وجود دارد، اما نه برای هر ادعایی:

| دارایی واقعی | نتیجهٔ طراحی |
|---|---|
| 1,781 slice با هر سه landmark، در 177 study مثبت | supervision مستقیم landmark/geometry داریم |
| 3,484 ردیف MLS (1,781 positive، 1,703 negative)، 338 study | selector و study-level coverage قابل ساخت است |
| SOP UID، `slice_index`، spacing و ترتیب z برای همهٔ rows | context و trajectory محدود در امتداد z قابل مدل‌سازی است |
| 175 از 177 positive study با interval annotation پیوسته | multi-slice context یک فرض دل‌بخواهی نیست |
| cache همسایه‌برشی 9-channel برای هر 338 study | الگوی implementation و coverage داده وجود دارد |
| maskهای segmentation فقط ICH هستند، نه brain/falx/hemisphere | supervised midline surface یا hemisphere segmentation فعلاً **ناممکن** است |

اختلاف `5,176` JSON annotation slice با `7,683` DICOM slice نشانهٔ انتقال خراب نیست؛ dataset ذاتاً partial-label است. هر 3,484 row MLS و هر 1,781 مثبت با SOP UID به manifest درست map شده‌اند.

نتیجه: مسیر 2.5D/landmark-aware بسیار feasible است؛ pseudo-surface با اتصال سه point، hemisphere segmentation بدون GT، و pooling-only sweep مسیرهای ناموجه‌اند.

### 4. مشکل provenance که باید جداگانه حل شود

سه مفهوم متفاوت در repository با نام submission/champion وجود دارد و نباید مخلوط شوند:

1. `submission/` checked-in، source قدیمی heatmap-only و بدون weight واقعی است.
2. `checkpoint/mls/mls-conservative-five-20260902/` یک candidate داخلی پذیرفته‌شده با package SHA `660770…d490b` است؛ status آن `internally_accepted_awaiting_leaderboard` است، نه Champion رسمی.
3. evaluator جدید یک prediction table remote و hash-pinned با نام `frozen_champion` مصرف می‌کند؛ composition آن در worktree قابل بازسازی نیست و به package بالا cryptographically bound نشده است.

conservative-five بهترین candidate داخلی MLS در این snapshot است (OOF 204-study: MAE `1.461522`، Boundary-F1 `0.855888`، objective `1.749745`) اما این **اثبات score رسمی، Champion، یا عبور از 0.914 نیست**. یک score حدود 0.90 که شفاهی گزارش شده بدون portal receipt، submission ID، ZIP SHA و runtime log را نمی‌توان به artifact مشخصی نسبت داد.

این یک نقص علمی/عملی مهم است، نه حاشیه: بدون comparator قابل‌بازسازی، ادعای بهبود نهایی قابل دفاع نیست. پیش از submission بعدی باید ZIP واقعی، SHA، portal submission ID و نتیجهٔ platform به هم bind شوند.

## چرا تاکنون به هدف نرسیده‌ایم؟

1. **هدف و proxy یکسان نبوده‌اند.** MAE slice/study و حتی F1@5 می‌توانند بهتر شوند، در حالی که Macro-F1 triage افت یا بدون تغییر بماند. A10 مثال مستقیم است.
2. **فرضیهٔ A9/A10 اطلاعات جدید وارد نکرد.** همان representation تک‌برشی/selector را با refiner و loss متفاوت فشار داد؛ در مقابل errorهای باقی‌مانده احتمالاً selection، visibility، context و aggregation مطالعه‌ای‌اند.
3. **بخش بزرگی از تاریخچه روی metricهای میانجی است.** اکنون evaluator deploy-aligned وجود دارد، ولی نتیجهٔ نهایی 338-study/5-fold برای candidate جدید هنوز تولید نشده است.
4. **sample کوچک در مرزهای حساس، امکان خودفریبی دارد.** به همین دلیل gates A9/A10 عمداً سخت‌گیرانه‌اند و از cherry-pick یک MAE بهتر جلوگیری کرده‌اند.
5. **project drift وجود دارد.** README/leaderboard قدیمی از QWK/heatmap package حرف می‌زنند، در حالی که pipeline جدید Macro-F1 و deploy-aligned است. این باید در packaging و documentation canonical شود؛ در غیر این صورت انتخاب مدل اشتباه یا upload artifact اشتباه محتمل است.

پس مسیر قبلی کاملاً غیرعلمی نبوده: وجود baseline freeze، hash، CUDA-only screen و reject gate دقیقاً رفتار صحیح علمی است. اما ادامهٔ همان خانواده پس از نتیجهٔ A10، علمی نبود؛ از این نقطه بسته شده است.

## فرضیهٔ بعدی: MLS-G1

### تغییر اصلی، مرحله‌به‌مرحله

```text
DICOM مرتب‌شده در z
  → context سه‌برشی [z-1, z, z+1] با windowهای MLS فعلی، در grid 512
  → landmark heatmap + visibility/selector صریح
  → فاصلهٔ هندسی با DICOM spacing و quality features
  → aggregation محافظه‌کارانهٔ study-level
  → MLS-mm و diagnostic ordinal outputs (>=3 mm، >=5 mm)
```

این مسیر از نظر پژوهشی با coarse-to-fine localization، heatmap/keypoint و multi-slice fusion هم‌راستاست، اما قرار نیست یک مقاله را بدون توجه به قرارداد داده تقلید کند. منابع اولیهٔ مفید:

- [Nguyen et al., ICCVW 2021](https://openaccess.thecvf.com/content/ICCV2021W/MIA-COV19D/html/Nguyen_Brain_Midline_Shift_Detection_and_Quantification_by_a_Cascaded_Deep_ICCVW_2021_paper.html): coarse-to-fine و fusion چندslice.
- [Yan et al., Diagnostics 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC8947005/): اندازه‌گیری MLS بر پایهٔ keypointهای anatomical.
- [Gong et al., MICCAI 2023](https://link.springer.com/chapter/10.1007/978-3-031-34048-2_6): استفادهٔ محتاطانه از دادهٔ کم‌برچسب/بدون‌برچسب برای deformation؛ فقط contingency ما، نه experiment اول.

### کنترل‌های لازم

cache آمادهٔ ICH مستقیم استفاده نمی‌شود، چون resolution/window آن با grid 512 MLS فرق دارد. ابتدا cache اختصاصی MLS در 512 و با windowهای baseline ساخته می‌شود؛ سپس:

1. **control هم‌ارز:** pipeline/cache جدید ولی فقط slice مرکزی؛ تا اثر infrastructure با اثر context اشتباه نشود.
2. **G1-A:** context 9-channel با headهای baseline و بدون تغییر همزمان pooling/threshold.
3. تنها در صورت سیگنال معتبر: visibility/selector reliability و سپس aggregation کوچکِ uncertainty-aware، یکی در هر زمان.
4. trajectory consistency فقط روی intervalهای observed و به‌صورت regularization ملایم؛ هیچ pseudo-label برای sliceهای بدون annotation ساخته نمی‌شود.

همهٔ augmentationهای spatial باید برای 9 channel یکسان باشند و inference باید دقیقاً همان contract cache را بازتولید کند. فایل‌های فعلی `dataset.py` و `predict.py` این قرارداد را هنوز پیاده نمی‌کنند؛ تغییر config به 9 channel بدون آن mismatch ایجاد می‌کند.

## پروتکل پذیرش و توقف

- fold0 به‌دلیل A9/A10 دیگر evidence مستقل نیست. screen اول روی fold واقعاً استفاده‌نشده پس از pre-registration انجام می‌شود.
- هر candidate فقط پس از static provenance/data/cache gate وارد CUDA می‌شود؛ CPU model forward ممنوع است.
- ICH و fracture در comparator frozen و hash-pinned می‌مانند؛ فقط MLS_mm عوض می‌شود. اثر causal MLS از این طریق قابل اندازه‌گیری است.
- screen اولیه باید direction مثبت Macro-F1 و Urgent-F1 در frozen و oracle context، عدم افت F1@3/F1@5 و absence of catastrophic-error regression نشان دهد. شکست هر gate یعنی توقف آن hypothesis، بدون rescue tuning.
- promotion واقعی: 338 study / 320 patient / پنج fold patient-grouped، سه seed ثابت، median مشخص، 10,000 patient-bootstrap با probability-of-improvement حداقل 0.95، hashes و full coverage. فقط پس از آن package واقعی با CLI رسمی و GPU 24GB smoke می‌شود.

## برآورد صادقانهٔ امید

- ادامهٔ خانوادهٔ A9/A10/refiner: احتمال ارزشمند بودن بسیار پایین (برآورد مهندسی کمتر از 10٪)، زیرا A9 و A10 خروجی threshold یکسان و نامطلوب دادند.
- یک G1 که واقعاً observation model را به 2.5D تبدیل کند: شانس معقول اما نامطمئن برای عبور از gate سخت MLS/triage وجود دارد (برآورد مشروط و غیرآماری حدود 15–30٪). این عدد prediction leaderboard نیست؛ فقط قضاوت اولویت‌دهی برای هزینهٔ یک مطالعهٔ کنترل‌شده است.
- احتمال عبور از رتبهٔ اول یا `0.914` را فعلاً نمی‌توان علمی عددگذاری کرد: score رسمیِ artifact کنونی به SHA مشخص bind نشده و ICH/fracture نیز در نتیجهٔ نهایی نقش دارند.

## تصمیم عملی ثبت‌شده

1. A9/A10 بسته و آرشیو می‌مانند.
2. قبل از CUDA جدید، contract داده، cache 512 و pre-registration `G1` ساخته می‌شود.
3. hypothesis فقط در صورت feasibility و gateهای روشن، روی GPU 3090 اجرا خواهد شد.
4. هر نتیجهٔ جدید با triage final و provenance package گزارش می‌شود، نه فقط با MAE.
5. score رسمی بعدی فقط زمانی «نتیجهٔ مدل X» نام می‌گیرد که portal receipt + submission ID + ZIP SHA + runtime log ثبت شده باشند.

