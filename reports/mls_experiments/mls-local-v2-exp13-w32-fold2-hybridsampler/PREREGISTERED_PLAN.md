# برنامهٔ ازپیش‌ثبت‌شدهٔ Exp13 — sampler میانی

## فرضیه

Exp12 نشان داد برابرکردن کامل exposure مطالعه‌ها جداسازی selector/heatmap را در بعضی epochها بهتر می‌کند، اما calibration رگرسیون MLS و profile production را خراب می‌کند. فرضیهٔ Exp13 این است که exposure متناسب با `sqrt(rows_per_study_class)` می‌تواند بخشی از کاهش bias اندازهٔ مطالعه را نگه دارد، بدون اینکه مشاهده‌های studyهای بلند را به‌اندازهٔ sampler کامل حذف کند.

## تنها متغیر

- Exp10: study mass تقریباً متناسب با تعداد ردیف‌ها؛ max/min=`7.8724×`.
- Exp12: study mass برابر در هر class؛ max/min کل=`2.8951×`.
- Exp13: study mass متناسب با ریشهٔ تعداد ردیف‌ها؛ max/min=`4.7193×`.

class mass در هر سه سیاست دقیقاً `1.0/1.0` باقی می‌ماند. معماری، target، fold، seed، optimizer، augmentation، batch size، loss، schedule و snapshotها با Exp10/12 یکسان‌اند.

## محاسبه و ایمنی

- forward/backward/inference مدل فقط روی `cuda:0`؛ CPU fallback ممنوع.
- CPU فقط برای DataLoader، I/O، metric aggregation و گزارش سبک.
- loss/metric باید finite بماند؛ هر OOM، NaN یا device mismatch اجرای مدل را متوقف می‌کند.
- raw data و predictionهای per-study/per-slice به MLflow ارسال نمی‌شوند.
- پس از ذخیره checkpoint نهایی، allocationهای training پیش از آپلود MLflow از CUDA آزاد می‌شوند.

## checkpointها و معیارها

- snapshotهای ثابت: `13/15/17/19/21/23`.
- معیار اصلی: Exp13/epoch15 روی profile production ثابت
  `severity_window(size=3, gate=0.5, min_active=3, q=0.75, weighted=true, guard=0)`.
- baseline اصلی: Exp10/epoch15 با `MAE=1.7143587902` و `Boundary-F1=0.8946969697`.
- معیار ثانویهٔ ازپیش‌ثبت‌شده: Exp13/epoch13 روی profile قدیمی frozen از fold0/1. Exp12 در آن profile `MAE=1.690618` و `Boundary-F1=0.925253` داشت.
- بهترین profile درون-fold فقط diagnostic است و حق تغییر production را ندارد.

## قانون ادامه/توقف

1. activation selector/study در epochهای 8–9 باید رخ دهد؛ در غیر این صورت run متوقف می‌شود.
2. اگر آموزش سالم باشد، فقط تا epoch23 ادامه می‌یابد؛ افزایش blind epoch ممنوع است.
3. گسترش به fold0/1 فقط در صورتی مجاز است که معیار اصلی epoch15 بهبود material بدهد و Boundary-F1 افت نکند. معیار ثانویه به‌تنهایی مجوز گسترش نیست.
4. اگر Exp13 بین Exp10 و Exp12 یا بدتر باشد، شاخهٔ sampler بسته می‌شود و آزمایش sampler دیگری اجرا نمی‌شود.

