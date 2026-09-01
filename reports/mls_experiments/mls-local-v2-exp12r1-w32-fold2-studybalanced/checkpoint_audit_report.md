# گزارش ممیزی end-to-end آزمایش Exp12

## نتیجهٔ اجرایی

آموزش `mls-local-v2-exp12r1-w32-fold2-studybalanced` هر ۲۳ ایپاک را سالم، فقط روی CUDA و بدون OOM، NaN یا CPU fallback کامل کرد. سپس snapshotهای ازپیش‌ثبت‌شدهٔ `13/15/17/19/21/23` روی تمام ۶۷ مطالعهٔ fold2 ارزیابی شدند. هر `402/402` inference موفق بود و failure صفر ثبت شد؛ مجموع زمان inference مدل `628.93 s` بود.

آزمایش اصلی ازپیش‌ثبت‌شده شکست خورد: Exp12/epoch15 روی profile production ثابت به `MAE=1.9153 mm` و `Boundary-F1=0.8059` رسید، در حالی که Exp10/epoch15 با همان profile `MAE=1.7144 mm` و `Boundary-F1=0.8947` داشت. تغییر sampler، MAE را `0.2009 mm` یا `11.72%` بدتر و Boundary-F1 را `0.0888` کم کرد. بنابراین `study_class_balanced` در شکل فعلی جای baseline را نمی‌گیرد و به fold0/1 گسترش داده نمی‌شود.

## پروتکل و ایمنی

- تنها متغیر اصلی در برابر Exp10، `sampling_mode=study_class_balanced` بود؛ معماری HRNet-W32، hybrid-soft target، fold، seed، optimizer، augmentation، batch size، loss، schedule و snapshotها ثابت ماندند.
- تمام forward/backward/inference مدل روی `cuda:0` انجام شد. CPU فقط برای DataLoader، I/O، تجمیع سبک CSV و نوشتن گزارش استفاده شد.
- برای هر checkpoint، کامل‌بودن `67/67` مطالعه اجباری بود و اسکریپت با هر failure متوقف می‌شد.
- predictionهای per-study/per-slice فقط محلی ماندند و به MLflow ارسال نشدند.
- common grid برای هر checkpoint دقیقاً ۶۰۴۸ profile داشت؛ فضای جست‌وجوی تازه‌ای مخصوص نتیجهٔ Exp12 ساخته نشد.

## نتیجه روی profile production قفل‌شده

profile ثابت:

`severity_window(size=3, gate=0.5, min_active=3, q=0.75, probability_weighted=true, guard=0)`

| checkpoint | Exp10 MAE | Exp10 Boundary-F1 | Exp12 MAE | Exp12 Boundary-F1 | نتیجه |
|---:|---:|---:|---:|---:|---|
| 13 | 2.1574 | 0.8693 | 1.9483 | 0.8454 | MAE بهتر، boundary ضعیف‌تر |
| 15 | **1.7144** | **0.8947** | 1.9153 | 0.8059 | شکست معیار اصلی |
| 17 | **1.6815** | **0.9145** | 1.9635 | 0.8468 | هر دو معیار بدتر |
| 19 | 2.3201 | 0.8449 | **1.8729** | 0.8296 | MAE بهتر، boundary اندکی ضعیف‌تر |
| 21 | 2.0197 | 0.8536 | **1.9873** | **0.8756** | بهبود کوچک هر دو معیار، ولی ضعیف‌تر از baseline production |
| 23 | 2.0913 | 0.8422 | **2.0816** | **0.8486** | تقریباً خنثی |

بهترین Exp12 روی profile production، epoch19 با `MAE=1.8729` است؛ این مقدار همچنان `0.1586 mm` بدتر از baseline اصلی Exp10/epoch15 است و Boundary-F1 آن نیز `0.0651` پایین‌تر است.

## اختلاف proxy با full-study E2E

proxy داخلی برای epochهای 13، 17 و 21 بسیار امیدوارکننده بود؛ اما این برتری به profile production منتقل نشد:

| epoch | proxy study MAE | proxy Boundary-F1 | E2E production MAE | E2E production Boundary-F1 |
|---:|---:|---:|---:|---:|
| 13 | 1.3549 | 0.9573 | 1.9483 | 0.8454 |
| 15 | 1.9442 | 0.8325 | 1.9153 | 0.8059 |
| 17 | 1.3170 | 0.9467 | 1.9635 | 0.8468 |
| 19 | 1.6649 | 0.8992 | 1.8729 | 0.8296 |
| 21 | 1.5312 | 0.9230 | 1.9873 | 0.8756 |
| 23 | 1.5106 | 0.9324 | 2.0816 | 0.8486 |

این mismatch تأیید می‌کند که validation برش‌محور/نمونه‌برداری‌شده برای انتخاب checkpoint نهایی کافی نیست؛ ranking snapshot باید با full-study E2E انجام شود و سپس روی foldهای دیگر انتقال یابد.

## سیگنال مثبت تشخیصی که نباید بیش‌تفسیر شود

- بهترین profile تشخیصی کل grid مربوط به epoch13 بود: `topk(size=5, gate=0.6, min_active=3, q=0.65, weighted=true, guard=0)` با `MAE=1.5142`، `Boundary-F1=0.9504` و objective=`1.6134`.
- profile دیگری که پیش‌تر فقط با fold0/1 منجمد شده بود، روی Exp12/epoch13 به `MAE=1.6906` و `Boundary-F1=0.9253` رسید؛ همان profile روی Exp10/epoch13 `MAE=2.1446` و `Boundary-F1=0.8887` داشت.
- این دو مشاهده نشان می‌دهند representation کاملاً خراب نشده و sampler جدید در epoch13 اطلاعات مفیدی تولید کرده است. بااین‌حال انتخاب epoch13/profile پس از دیدن fold2 یک برآورد unbiased نیست و نمی‌تواند شکست معیار اصلی epoch15/profile production را لغو کند.

## تشخیص علت

sampler جدید جداسازی study-level بعضی featureها را بهتر کرد. برای نمونه در epoch13، AUC جداسازی `selector_max` از `0.8378` در Exp10 به `0.8636` و AUC `heatmap_top3_mean` از `0.6239` به `0.7273` رسید. بنابراین علت شکست، collapse معماری یا خرابی localization نیست.

مشکل غالب، تغییر calibration و exposure رگرسیون MLS است:

- epoch15 روی profile production bias=`-1.5169 mm` دارد؛ مدل شدت MLS را به‌طور سیستماتیک کم‌برآورد می‌کند.
- برابرکردن کامل مطالعه‌ها سهم studyهای بلند را شدیداً کم کرده است. این کار ranking وجود MLS را در بعضی epochها بهتر می‌کند، اما تعداد مشاهده‌های مفید برای کالیبراسیون مقدار MLS و رفتار full-study pooling را کاهش می‌دهد.
- profile بهینهٔ in-fold از `severity_window` به `topk` و gate بالاتر جابه‌جا شد؛ یعنی توزیع probability/MLS خروجی نسبت به baseline تغییر کرده و profile production قبلی دیگر مناسب نیست. تنظیم profile روی همان fold می‌تواند این جابه‌جایی را پنهان کند و overfit بسازد.
- نوسان شدید میان epoch13/15/17 نشان می‌دهد sampler کامل study-balanced مسئلهٔ variance زمانی را حل نکرده است.

## تصمیم و اقدام بعدی

1. baseline production فعلی، hybrid/epoch15 با sampler قدیمی و profile ثابت باقی می‌ماند.
2. `study_class_balanced` کامل رد می‌شود و fold0/1 برای آن آموزش داده نمی‌شوند؛ این کار از حدود چند ساعت GPU بدون پشتوانه جلوگیری می‌کند.
3. سیگنال مثبت epoch13 فقط به‌عنوان فرضیه حفظ می‌شود: یک sampler میانی باید بخشی از اصلاح bias مطالعه را نگه دارد، ولی exposure برش‌های studyهای بلند را صفر یا بیش‌ازحد کم نکند.
4. آزمایش بعدی، در صورت اجرا، باید تک‌متغیره و با یک ضریب اختلاط ازپیش‌ثبت‌شده بین وزن‌های slice-balanced و study-balanced باشد؛ checkpoint/profile اصلی آن پیش از آموزش قفل می‌شود و همان قانون E2E اعمال خواهد شد.

## فایل‌های ممیزی

- `end_to_end_checkpoint_audit/epoch{013,015,017,019,021,023}/metrics.json`
- `end_to_end_checkpoint_audit/epoch{013,015,017,019,021,023}/study_slice_predictions.csv` (فقط محلی)
- `checkpoint_pooling_expanded/checkpoint_pooling_grid.csv`
- `checkpoint_pooling_expanded/checkpoint_pooling_summary.json`

