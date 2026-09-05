# امکان‌سنجی دادهٔ خارجی برای Brain CT Triage — ۲۰۲۶-۰۹-۰۵

## حکم اجرایی

در این مقطع هیچ دادهٔ خارجی نباید دانلود، cache، یا وارد آموزش شود. راهنمای رسمی
مسابقه در Rule 1 اجازهٔ استفاده را به «external resources explicitly allowed by the
organizers» محدود کرده است. پس حتی دادهٔ public و non-commercial بدون تأیید کتبی
برگزارکننده، ریسک مستقیم صلاحیت submission دارد.

این گزارش یک غربال علمی/حقوقی است، نه مجوز استفاده از هیچ داده‌ای.

## مسئله‌ای که واقعاً باید تقویت شود

هدف MLS مسابقه یک مقدار پیوسته در میلی‌متر، در سطح foramen of Monro، است. supervision
آموزشی نیز از سه keypoint اختصاصی falx می‌آید. از آن مقدار پیوسته، مرزهای triage حساس
۱، ۳ و ۵ میلی‌متر مشتق می‌شوند. بنابراین داده‌ای که فقط "MLS present/absent" دارد، نمی‌تواند
به‌تنهایی label صحیح regression یا calibration در این مرزها را بسازد.

## بررسی گزینه‌ها

| گزینه | مقیاس/برچسب | هم‌خوانی با MLS | ارزش ممکن | مانع اصلی | تصمیم |
|---|---:|---|---|---|---|
| **CQ500** | 491 CT؛ ICH subtype، fracture، mass effect و MLS در سطح study | MLS **دودوییِ شدید** (منبع ثانویه آن را `>5 mm` گزارش می‌کند)؛ نه mm پیوسته و نه keypoint | auxiliary head برای MLS شدید / pretraining یا OOD sanity-check | تنها حدود 65 مورد MLS مثبت گزارش شده؛ label target با مسابقه یکسان نیست؛ مجوز مسابقه باید صریح شود | بهترین کاندید کوچک، اما **فعلاً hold** |
| **RSNA ICH 2019** | بیش از 25k exam و پنج subtype ICH، در سطح slice | هیچ MLS/mm/keypoint و mask حجمی رسمی ندارد | فقط representation pretraining برای ICH، نه رفع خطای MLS | non-commercial، controlled access، حجم بسیار زیاد و mismatch با label حجم/MLS | برای MLS **رد**؛ فقط در برنامهٔ مستقل ICH پس از اجازه |
| **PhysioNet ICH** | 82 بیمار، segmentation ICH در سطح pixel | MLS label ندارد | کمک محدود برای segmentation ICH | کوچک است و به MLS نمی‌خورد | برای MLS **رد** |
| **Qure25k** | مجموعهٔ بسیار بزرگ‌تر Qure با findings CT | در ادبیات عمومی به‌عنوان در دسترس عمومی معرفی نشده | در تئوری ارزشمند | دادهٔ قابل دریافت عمومیِ تأییدشده نیست | **رد** |
| مخزن `neuro-ml/midline-shift-detection` | کد و نمونه annotation | ورودی MRI NIfTI، نه CT | هیچ | modality mismatch | **رد** |

## چرا CQ500 تنها گزینهٔ قابل بررسی است

CQ500 از نظر modality (non-contrast head CT) و findings (پنج subtype ICH، fracture، mass
effect و MLS) به challenge نزدیک است. اما label MLS آن binary و در منابع ثانویه به‌صورت
MLS شدید (`>5 mm`) گزارش شده است، در حالی که مسابقه mm در یک سطح آناتومیک تعریف‌شده می‌خواهد.
بنابراین راه صحیحِ احتمالی فقط این است:

1. مدل اصلی regression/keypoint فقط با دادهٔ رسمی و labels رسمی آموزش می‌بیند.
2. اگر برگزارکننده صریحاً اجازه داد، CQ500 فقط به شکل auxiliary binary head برای MLS شدید یا pretraining
   کنترل‌شده وارد شود؛ نه به‌عنوان pseudo-mm label و نه برای threshold calibration.
3. split بر اساس patient/study، checkpoint provenance، و ablation زوجی لازم است.
4. gate پذیرش: بهبود هم‌زمان در held-out official studies، خصوصاً F1@3mm/F1@5mm و triage,
   بدون افت mean absolute error. نتیجهٔ CQ500 به‌تنهایی هیچ تصمیم release نمی‌سازد.

## ریسک‌ها

- **Label mismatch:** تبدیل binary MLS شدید به مقدار mm یا دستکاری label هدف، از نظر علمی نامعتبر
  و برای مرزهای 3/5mm مخرب است.
- **Domain shift:** منبع، ضخامت slice، reconstruction kernel و mix pathology متفاوت‌اند.
- **Leakage/overclaim:** یک dataset خارجی نباید validation اصلی یا تنظیم threshold مسابقه باشد.
- **Licensing:** مجوز خود dataset با مجوز مسابقه متفاوت است؛ هر دو لازم‌اند.
- **هزینه-فایده:** CQ500 کوچک است و تعداد MLS positive گزارش‌شده محدود؛ بنابراین انتظار
  جهش بزرگ از آن واقع‌بینانه نیست.

## اقدام پیشنهادی، به ترتیب

1. از برگزارکننده یک اجازهٔ مکتوب بگیریم: «آیا استفاده از CQ500/RSNA برای pretraining یا
auxiliary supervision مجاز است، و آیا وزن‌های حاصل در submission قابل استفاده‌اند؟»
2. فقط اگر پاسخ مثبت بود، شرایط دسترسی/attribution منبع را جداگانه بخوانیم؛ سپس یک manifest
قابل بازتولید با checksum و عدم‌اختلاط با official validation بسازیم.
3. ابتدا CQ500 را صرفاً audit کنیم: تعداد usable studies، prevalence واقعی MLS، series
selection و DICOM integrity. این audit یک GPU training نیست.
4. تنها پس از آن، یک ablation کوچک و از پیش ثبت‌شده مقابل بهترین baseline رسمی اجرا شود.

## منابع بررسی‌شده

- راهنمای رسمی challenge، صفحات 1–3 و Rule 1 صفحه 13.
- CQ500: Chilamkurthy et al./Qure dataset page که در مطالعات بعدی به‌عنوان 491 CT با labels
  study-level گزارش شده است.
- RSNA ICH data card و terms of use رسمی.
- بررسی مستقیم README مخزن neuro-ml که صریحاً ورودی NIfTI MRI را اعلام می‌کند.

## نتیجه

دادهٔ خارجیِ عمومیِ آماده‌ای که واقعاً target MLS-mm/keypoint این مسابقه را پوشش دهد پیدا
نشد. CQ500 ارزش یک آزمایش محدود و قانون‌مند دارد، اما تنها پس از مجوز صریح برگزارکننده و
بدون تبدیل label دودویی آن به میلی‌متر. فعلاً بهترین مسیر MLS همچنان تقویت مدل با annotation
های رسمی، ارزیابی deploy-aligned و آزمایش R1 فعال است.
