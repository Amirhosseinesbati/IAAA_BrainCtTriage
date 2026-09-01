# handoff نهایی و pause کنترل‌شدهٔ مسیر MLS

تاریخ: 2026-08-28  
وضعیت: **pause عملیاتی؛ goal ناتمام حفظ شود**  
قانون: تا ازسرگیری صریح کاربر، هیچ آموزش یا آزمایش جدیدی آغاز نشود.

## 1. حکم اجرایی

بهترین راهکار قابل‌دفاع فعلی برای MLS، خانوادهٔ سه‌fold زیر است:

- backbone: `HRNet-W32`
- target/selector: `hybrid-soft / peak-aware-soft`
- schedule انتخاب checkpoint: **epoch15 ثابت برای هر سه fold**
- sampler: `slice_class_balanced` قدیمی، نه study-balanced کامل و نه hybrid میانی
- deployment پیشنهادی: ensemble سه مدل fold0/1/2 در سطح prediction، مشروط به ممیزی نهایی زمان inference و بستهٔ leaderboard

Exp12 با sampler کاملاً study-balanced و Exp13 با sampler میانی هر دو در معیار اصلی production شکست خوردند. بنابراین شاخهٔ sampler بسته است و تکرار آن بدون فرضیهٔ تازه توجیه ندارد.

## 2. مدل‌هایی که همین حالا واقعاً روی سیستم موجودند

### انتخاب production-safe فعلی

| مدل | fold | checkpoint | E2E MAE ثابت epoch15 | وضعیت |
|---|---:|---:|---:|---|
| Exp08 hybrid-soft | 0 | epoch15 | 1.6646 mm | موجود و عضو baseline سه‌fold |
| Exp09 hybrid-soft-transfer | 1 | epoch15 | 1.2587 mm | موجود و قوی‌ترین نتیجه منفرد بین epoch15ها |
| Exp10 hybrid-soft-transfer | 2 | epoch15 | 1.7144 mm | موجود و بدترین fold baseline |

مسیر فایل‌ها:

1. `models/checkpoints/mls_multitask/mls-local-v2-exp08-w32-fold0-hybridsoft-snapshots/mls_multitask_epoch_015.pth`
2. `models/checkpoints/mls_multitask/mls-local-v2-exp09-w32-fold1-hybridsoft-transfer/mls_multitask_epoch_015.pth`
3. `models/checkpoints/mls_multitask/mls-local-v2-exp10-w32-fold2-hybridsoft-transfer/mls_multitask_epoch_015.pth`

هر فایل در زمان ممیزی حدود `124,890,565` بایت بود. روی profile diagnostic مشترک، میانگین MAE سه fold=`1.5459 mm`، بدترین fold=`1.7144 mm` و Boundary-F1 میانگین=`0.8469` بود. برآورد سخت‌گیرانهٔ balanced برابر `Mean MAE=1.6273 mm`، `Worst=1.7999 mm` و Boundary-F1=`0.8208` است.

اگر فقط یک فایل منفرد لازم باشد، Exp09/fold1/epoch15 بهترین عدد میان checkpointهای ثابت epoch15 را دارد؛ اما برای تست ناشناخته، تکیه بر سه‌fold منطقی‌تر از انتخاب fold1 است.

### بهترین عدد خام، نه بهترین انتخاب production

`Exp08/fold0/epoch21` با `MAE=1.2334 mm`، Boundary-F1=`0.8348` و objective=`1.5637` بهترین عدد خام یک checkpoint منفرد بود:

`models/checkpoints/mls_multitask/mls-local-v2-exp08-w32-fold0-hybridsoft-snapshots/mls_multitask_epoch_021.pth`

این checkpoint روی همان fold انتخاب شده و نسبت به انتخاب ثابت سه‌fold خوش‌بینانه است؛ نباید به‌عنوان مدل نهایی تست ناشناخته جایگزین epoch15 شود.

### مدل Exp13 که فقط diagnostic است

فایل‌های سالم Exp13:

- `.../mls-local-v2-exp13-w32-fold2-hybridsampler/mls_multitask_epoch_013.pth`
- `.../mls-local-v2-exp13-w32-fold2-hybridsampler/mls_multitask_epoch_015.pth`
- `.../mls-local-v2-exp13-w32-fold2-hybridsampler/mls_multitask_epoch_017.pth`

بهترین diagnostic همان fold برای epoch13 با top-k7 به `MAE=1.5377 mm` و Boundary-F1=`0.9227` رسید، اما profile بعد از دیدن fold2 انتخاب شده و production-safe نیست. روی profile production ثابت، epoch15 برابر `1.7550 / 0.8626` بود و baseline Exp10 (`1.7144 / 0.8947`) را شکست نداد.

## 3. خلاصهٔ آزمایش‌ها و تصمیم‌ها

| آزمایش | سؤال | نتیجه | تصمیم |
|---|---|---|---|
| Exp08–10 | آیا HRNet-W32 hybrid-soft در سه fold تعمیم دارد؟ | مثبت؛ epoch15 پایدارترین انتخاب مشترک شد | baseline فعلی |
| Exp11 | snapshot median/SWA/calibration | median 13/15/17 MAE متوسط را بهتر کرد، اما سه inference لازم داشت؛ SWA boundary/worst-fold را ضعیف کرد | default نشد |
| Exp12 | study-balanced کامل | primary epoch15=`1.9153 / 0.8059` در برابر Exp10=`1.7144 / 0.8947` | رد؛ fold0/1 اجرا نشد |
| Exp13 | exposure میانی با توان 0.5 | primary epoch15=`1.7550 / 0.8626`؛ MAE `2.37%` و objective `5.44%` بدتر | رد؛ sampler branch بسته شد |

نتیجهٔ علمی مهم: localization/selector proxy بهتر لزوماً به full-study calibration بهتر منتقل نمی‌شود. checkpoint و pooling باید با protocol قفل‌شده و cross-fold سنجیده شوند؛ انتخاب آزاد epoch/profile روی همان fold مجاز نیست.

## 4. وضعیت دقیق Exp13

- training/validation تا epoch19 با CUDA و بدون OOM/NaN/CPU fallback انجام شد.
- پس از validation epoch19، DNS مربوط به DagsHub در `mlflow.log_metrics` قطع شد. چون logging قبل از snapshot save بود، process با exit code 1 بسته و epoch19 ذخیره نشد.
- snapshotهای 13/15/17 موجودند؛ 19/21/23 موجود نیستند.
- restart انجام نشد، زیرا checkpointها optimizer/scheduler/RNG state ندارند و ادامه از epoch17 trajectory دقیق قبلی نبود.
- E2E سه checkpoint: `201/201` موفق، صفر failure، runtime مجموع=`316.07 s`.
- MLflow Run ID: `0a2cf48a6fce417ba2f89c50a7ad185f`.
- status MLflow عمداً `FAILED` باقی مانده تا توقف واقعی پنهان نشود.
- گزارش audit، aggregate JSON، سه metrics.json، history و grid summary با موفقیت روی همان run sync شدند.
- raw `study_slice_predictions.csv`ها فقط محلی هستند و آپلود نشدند.

گزارش جزئی Exp13:

`reports/mls_experiments/mls-local-v2-exp13-w32-fold2-hybridsampler/checkpoint_audit_report.md`

## 5. پیش از هر آموزش آینده چه چیزی باید اصلاح شود

این چهار مورد gate اجباری‌اند:

1. snapshot، history و report قبل از هر تماس شبکه‌ای ذخیره شوند.
2. MLflow logging retry/deferred و غیرکشنده شود؛ قطع شبکه نباید training را متوقف کند.
3. resume checkpoint شامل model، optimizer، scheduler، scaler، epoch و RNG state باشد.
4. یک تست قطع مصنوعی MLflow نوشته شود و ثابت کند training محلی و checkpoint save ادامه پیدا می‌کند.

تا عبور از این gate، آموزش تازه روی GPU توصیه نمی‌شود.

## 6. ترتیب پیشنهادی ادامه پس از resume

1. ابتدا دوام trainer/MLflow و resume واقعی را اصلاح و با تست سبک اثبات کن.
2. سپس evaluator مجازی، ordering برش، spacing/orientation، grouping مطالعه و schema خروجی را با راهنمای مسابقه تطبیق بده.
3. بستهٔ inference سه مدل epoch15 را در محیط سازگار با packageهای leaderboard smoke-test کن و runtime/VRAM را ثبت کن.
4. یک dry-run submission بدون استفاده از leaderboard برای tuning انجام بده.
5. فقط اگر نتیجهٔ رسمی یا held-out نشان داد MLS ناکافی است، مسیر پژوهشی را دوباره باز کن.

اولویت پژوهشی در صورت بازشدن دوباره:

- نخست calibration/aggregation مشترک cross-fold با predictionهای موجود؛ بدون train جدید.
- سپس ensemble کم‌هزینه یا weight averaging فقط با معیار transferable.
- در صورت نیاز واقعی، selector دو-head یا context سبک 2.5D؛ هرکدام با یک متغیر اصلی، برنامه ازپیش‌ثبت‌شده و fold تأییدی.
- sampler تازه در اولویت نیست؛ دو نقطهٔ full و half هر دو production را بهبود ندادند.

## 7. مواردی که هنوز ادعای قطعی درباره‌شان نداریم

- metricهای MLS با score ترکیبی نهایی مسابقه یکسان نیستند؛ مدل خوب MLS به‌تنهایی score حدود 0.914 را تضمین نمی‌کند.
- هیچ submission رسمی انجام نشده، پس رتبه یا برابری evaluator مجازی با leaderboard رسمی اثبات نشده است.
- انتخاب سه‌fold ensemble باید با محدودیت زمان/VRAM سرور leaderboard سنجیده شود.
- نتیجهٔ Exp13 به‌علت نبود snapshotهای 19/21/23 کامل نیست، هرچند شکست primary epoch15 برای رد sampler کافی است.

## 8. نقطهٔ شروع دفعه بعد

به‌ترتیب این فایل‌ها خوانده شوند:

1. همین handoff: `reports/mls_experiments/MLS_FINAL_PAUSE_HANDOFF_2026-08-28_FA.md`
2. گزارش عمیق کل مسیر: `reports/mls_experiments/MLS_ONE_DAY_DEEP_PROGRESS_REPORT_2026-08-28_FA.md`
3. audit Exp13: `reports/mls_experiments/mls-local-v2-exp13-w32-fold2-hybridsampler/checkpoint_audit_report.md`
4. aggregate Exp13: `reports/mls_experiments/mls-local-v2-exp13-w32-fold2-hybridsampler/e2e_aggregate_metrics.json`
5. audit baseline fold2: `reports/mls_experiments/mls-local-v2-exp10-w32-fold2-hybridsoft-transfer/checkpoint_audit_report.md`

پس از خواندن این پنج مورد، تحلیل EDA/MLS یا sampler نباید از صفر تکرار شود. ادامه باید از gate دوام trainer و آماده‌سازی submission آغاز شود.

## 9. وضعیت pause

- هیچ process آموزش یا inference فعال باقی نمانده است.
- هیچ آزمایش تازه‌ای در صف نیست.
- goal نباید complete یا blocked شود؛ فقط ناتمام نگه داشته شود.
- ازسرگیری تنها با درخواست صریح کاربر انجام شود.
