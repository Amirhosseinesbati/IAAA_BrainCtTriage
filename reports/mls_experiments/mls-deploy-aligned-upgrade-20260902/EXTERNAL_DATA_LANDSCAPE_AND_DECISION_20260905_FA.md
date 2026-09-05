# بررسی عمیق دادهٔ تکمیلی برای Brain CT Triage — ۲۰۲۶-۰۹-۰۵

## تصمیم اجرایی

در وضعیت فعلی **هیچ تصویر، annotation، metadata، pseudo-label، یا وزن pretrained
خارجی نباید دانلود یا در آموزش/سابمیشن وارد شود**. Rule 1 راهنمای رسمی فقط دادهٔ
مسابقه و external resourceهایی را مجاز می‌داند که برگزارکننده صریحاً اجازه داده باشد.
مجوز public/non-commercial خودِ یک دیتاست، جای آن اجازه را نمی‌گیرد.

این نتیجه به معنی بی‌فایده بودن پژوهش نیست. جست‌وجوی عمیق نشان می‌دهد اگر برگزارکننده
بعداً اجازهٔ کتبی بدهد، دو مسیر بالقوهٔ باارزش وجود دارند؛ اما هیچ‌کدام اکنون آماده یا
هم‌ارز supervision مسابقه نیستند:

1. **CENTER-TBI / FITBIR**: تنها خانواده‌های داده‌ای که به‌طور مستند MLS بر حسب mm
   را در کنار CT دارند یا CDE آن را تعریف می‌کنند. دسترسی محدود، label semantics متفاوت
   و نبود keypointهای متناظر با مسابقه، آن را به یک گزینهٔ تحقیقاتی مشروط تبدیل می‌کند.
2. **CQ500**: نزدیک‌ترین مجموعهٔ public NCCT از نظر pathology، اما فقط 491 بیمار و
   label MLS در سطح study (نه MLS_mm یا keypoint). برای regression/مرز 3 mm مناسب نیست؛
   نهایتاً می‌تواند یک auxiliary severe-MLS screen باشد.

تا تأیید کتبی برگزارکننده، مسیر با بیشترین نسبت سود/ریسک، استفادهٔ کامل‌تر و leak-free از
دادهٔ رسمی خودمان است؛ نه واردکردن دادهٔ نامتجانس بیرونی.

## معیار تطابق با هدف واقعی

هدف MLS این مسابقه یک label عمومیِ «shift دارد/ندارد» نیست. مقدار `MLS_mm` در سطح
foramen of Monro از سه keypoint اختصاصی falx به‌دست می‌آید و مرزهای triage در 1، 3 و
5 mm حساس‌اند. EDA رسمی پروژه نیز نشان می‌دهد:

- 338 series / 320 patient و 7,683 DICOM رسمی وجود دارد؛
- 1,781 slice هر سه keypoint را دارد و geometry آنها `MLS_mm` را با خطای تقریباً صفر
  بازتولید می‌کند؛
- 28 series در فاصلهٔ 0.5 mm از 3 mm و 18 series در فاصلهٔ 0.5 mm از 5 mm هستند.

پس datasetی که فقط `MLS > 5 mm` یا presence را دارد، نمی‌تواند label میلی‌متری را
جایگزین کند، threshold را calibrate کند، یا pseudo-keypoint معتبر بسازد.

## منابع بررسی‌شده

| رتبهٔ بالقوه در صورت مجوز | منبع | داده و label قابل اتکا | تطابق با challenge | تصمیم امروز |
|---|---|---|---|---|
| 1 | **CENTER-TBI** | بیش از 4,500 بیمار TBI؛ dictionary آن `CRFCTMidlineShiftMeasure` را mm ثبت‌شده توسط investigator، و `MidlineShift` را central review برای `>5 mm` معرفی می‌کند. | نزدیک‌ترین candidate میلی‌متری است؛ اما سطح اندازه‌گیری، زمان CT، reader و reference line ممکن است با foramen-of-Monro/keypoint مسابقه فرق داشته باشد و keypoint ندارد. | **Hold / no-go**: Study Plan، DUA، non-commercial/no-redistribution و اجازهٔ کتبی برگزارکننده لازم است. |
| 2 | **FITBIR / TRACK-TBI** | CDE رسمی «MLS supratentorial measurement» را mm و در foramen of Monro یا محل بیشینه تعریف می‌کند؛ repository برای پژوهشگر qualified access می‌دهد. | از نظر تعریف بالینی نزدیک است، ولی باید قبل از هر استفاده ثابت شود که یک cohort مشخص raw CT paired با این CDE، completeness کافی و اجازهٔ export دارد. | **Hold / no-go**: access-controlled و Rule 1. تنها candidateی است که ارزش inquiry بعدی دارد. |
| 3 | **CQ500** | 491 NCCT؛ labels study-level شامل ICH subtype، fracture، mass effect و MLS؛ گزارش‌های ادبی حدود 65 MLS-positive را بیان می‌کنند. | modality/pathology نزدیک، ولی MLS عددی/keypoint/slice target ندارد؛ قدرت بسیار کم برای مرز 3 mm و regression. | **Hold / no-go**: فقط پس از مجوز، auxiliary binary severe-MLS ablation؛ هرگز pseudo-mm یا calibration target نیست. |
| 4 | **RSNA ICH 2019** | بیش از 25k examination / 874,035 image با label slice-level پنج subtype ICH. | برای MLS مستقیم تقریباً صفر؛ برای representation شاخهٔ ICH ممکن است مفید باشد، اما mask حجم یا MLS ندارد. | **Hold / no-go**: licence research/non-commercial و Rule 1 هر دو باید پاس شوند. |
| 5 | **BHX / Seg-CQ500 / INSTANCE / PhysioNet CT-ICH / PHE-SICH** | bounding box یا mask خونریزی/edema در مقیاس‌های محدود. | فقط proxy برای ICH؛ MLS-mm/landmark ندارند و می‌توانند shortcut pathology بسازند. | **رد برای MLS**؛ در صورت مجوز هم فقط برای ICH با ablation مستقل. |
| — | **TBI-IT / HICH-IT و mirrorهای Kaggle/HF** | ادعای brain-midline/ventricle/hematoma دارند، اما availability/license/data lineage کامل قابل‌راستی‌آزمایی نیست؛ TBI-IT صریحاً می‌گوید بخشی fictionalized و فقط sample نمایش می‌دهد. | dataset قابل‌دفاع برای training نیست. | **رد**. |

## چرا CENTER-TBI و FITBIR «جالب» هستند، ولی هنوز انتخاب نیستند

ACR برای use case MLS، خروجی عددی 0–10 mm در سطح foramen of Monro را مشخص می‌کند و
در عین حال می‌گوید dataset public مرتبط شناخته‌شده‌ای ندارد. CENTER-TBI از نظر حجم و
تنوع scanner جذاب است، اما مستنداتش label میلی‌متری را investigator-scored معرفی می‌کند؛
بنابراین پیش از هر training باید با یک audit ثابت شود که measurement definition، CT timepoint
و image-to-label pairing مناسب‌اند. FITBIR تعریف CDE نزدیک‌تری دارد، ولی وجود CDE به معنی
وجود raw DICOM جفت‌شده و قابل export برای هر study نیست.

نتیجه: این‌ها **lead برای درخواست دسترسی و audit** هستند، نه data source آماده.

## چرا CQ500 پاسخ «دادهٔ بیشتر برای MLS» نیست

CQ500 احتمالاً تنها dataset عمومی معروفی است که هم NCCT و هم label MLS/mass-effect را
دارد. با این حال label آن study-level binary است؛ تعداد positive کم است؛ و objective مسابقه
باید فاصلهٔ هندسی در mm را دقیقاً در ناحیهٔ حساس 3/5 mm بازسازی کند. تبدیل presence به mm،
ساختن keypoint از آن، یا تنظیم thresholdهای رسمی با آن از نظر علمی نامعتبر است.

اگر و فقط اگر مجوز برگزارکننده داده شد، design قابل دفاع این است:

1. backbone/MLS regressor فقط با supervision رسمی مسابقه آموزش ببیند؛
2. CQ500 فقط یک head جدا برای binary severe MLS داشته باشد؛
3. انتخاب checkpoint و calibration فقط با held-out official studies، fold-isolated و
   preregistered انجام شود؛
4. پذیرش تنها با بهبود هم‌زمان MAE، F1@3 mm، F1@5 mm و triage، بدون ادعای transfer صرف.

## مسیر پربازده و کاملاً مجاز در همین امروز

پژوهش داده نشان می‌دهد کمبود ما «تصویر CT» به‌تنهایی نیست؛ کمبود label دقیق MLS است. بنابراین
بهبودهای کم‌ریسک‌تر و محتمل‌تر این‌ها هستند:

1. تمام 1,781 keypoint slice رسمی را با geometry/spacing واقعی و negative/non-target slices
   در selector supervision استفاده کنیم؛
2. augmentation هندسی را با تبدیل دقیق هر سه keypoint و invariant بودن MLS کنترل کنیم؛
3. SSL فقط روی partition training رسمی اجرا شود؛ هر fold validation حتی بدون label وارد pretrain
   آن fold نشود؛
4. sampling مرزآگاه برای 3 و 5 mm، hard-negativeهای no-MLS و error analysis scanner/kernel
   داشته باشیم؛
5. هر idea فقط پس از CUDA evaluation deploy-aligned و triage gate سه-seed بررسی شود.

این اقدامات همان data diversity موردنیاز را از cohort رسمی استخراج می‌کنند، بدون ریسک
disqualification یا label mismatch.

## دروازهٔ قطعی پیش از بازکردن external-data track

1. **پاسخ کتبی برگزارکننده**: آیا external image/label، external pretrained weights، SSL فقط
   روی DICOM رسمی، و test-time adaptation/pseudo-labeling hidden test مجازند؟
2. **مجوز منبع**: DUA/licence اجازهٔ intended competition use و انتقال weight را بدهد.
3. **audit بدون training**: lineage، checksum، patient/study mapping، raw-DICOM/label pairing،
   measurement definition، prevalence و duplicate/overlap بررسی و receipt شود.
4. **ablation preregistered**: external و official-only فقط روی split رسمی برابر مقایسه شوند؛
   external data هرگز validation/test یا threshold tuning را آلوده نکند.
5. **promotion gate**: بهبود واقعی triage/Macro-F1 و Urgent-F1 در ارزیابی deploy-aligned؛ نه
   صرفاً metric داخلی external set.

بدون عبور از هر پنج دروازه، external data وارد workspace یا submission نمی‌شود.

## منابع کلیدی

- [راهنمای رسمی مسابقه، Rule 1، صفحهٔ 13](../../../Competition-Guide/iaaa-competition-2026-brain-ct-triage-challenge.pdf)
- [ACR MLS use case و تعریف numeric output / نبود dataset public](https://www.acr.org/Data-Science-and-Informatics/AI-in-Your-Practice/AI-Use-Cases/Use-Cases/Midline-Shift)
- [CENTER-TBI data access](https://www.center-tbi.eu/data) و [dictionary MLS](https://www.center-tbi.eu/data/dictionary)
- [FITBIR Imaging Read CDE](https://fitbir.nih.gov/dictionary/publicData/dataStructureAction!view.action?dataStructureName=ImagingRead_FITBIR&publicArea=true&style.key=fitbir-style)
- [CQ500 access/documentation pointer](https://github.com/muschellij2/cq500_code) و [تحلیل CQ500 MLS](https://www.mdpi.com/2076-3417/16/2/890)
- [RSNA ICH data card](https://atlas.rsna.org/cards/bd41bbe2-f161-4c19-841c-85d8b645e306) و [terms](https://www.rsna.org/-/media/files/rsna/education/ai-resources-and-training/ai-image-challenge/rsna_ich_ai_challenge_2019_terms_of_use_and_attribution_final.pdf?hash=A262E3D25E755CD1DCD1C158887C31CB&rev=eee3030315f349ff9271989b67198a69)
- [CENTER-TBI MLS methodology/heterogeneity evidence](https://pmc.ncbi.nlm.nih.gov/articles/PMC6551991/)

## ابزار پژوهش

Consensus برای discovery مقاله استفاده شد و رکوردهای یافته‌شده پیش از استناد fetch شدند.
SciSpace با وجود ارسال `searchQuestion` مطابق schema، در این session با خطای server-side
`Unknown tool` پاسخ داد؛ بنابراین نتیجه بر وب‌سایت‌های رسمی dataset/registry و مقالات primary
قابل‌دسترسی بنا شده است، نه بر نتیجهٔ unverified آن افزونه.
