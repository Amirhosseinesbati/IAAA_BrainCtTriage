# گزارش نهایی Exp16 — نوسازی مدل MLS برای fold0

## نتیجه‌ی اجرایی

آموزش 23 epoch روی RTX 3060 سرور Vast با حالت deterministic سخت و سیاست
`cuda_only_no_cpu_fallback` کامل شد. Run رسمی MLflow با شناسه‌ی
`a2478b8410d74de2b2806ef08d79051d` در وضعیت `FINISHED` قرار دارد. سپس ده
checkpoint روی تمام 70 مطالعه‌ی fold0، در مجموع 700 ارزیابی study-checkpoint،
با inference مستقل GPU-only بررسی شدند. هر ده ارزیابی exit code صفر و failure
صفر داشتند.

مدل منتخب `mls_multitask_best_selector_auc.pth` مربوط به epoch 16 است. این
انتخاب نه از روی val loss و نه از روی metric آنلاین training، بلکه از روی
full-series audit مستقل و profile تولید از پیش قفل‌شده انجام شده است.

## قرارداد ارزیابی معتبر

profile تولید قبل از مشاهده‌ی نتایج Exp16 قفل شده بود:

- خانواده‌ی pooling: `severity_window`
- شعاع پنجره: 3 slice
- selector gate: 0.5
- حداقل slice فعال: 3
- quantile: 0.75
- probability weighting: فعال
- heatmap guard: صفر

جست‌وجوی 6048 profile برای هر checkpoint نیز اجرا شد، اما فقط diagnostic است و
برای promotion استفاده نشد. این تفکیک مانع overfit شدن post-processing به
fold0 می‌شود.

## مقایسه‌ی نهایی با مرجع تاریخی fold0

| مدل | MAE (mm) | RMSE (mm) | Bias (mm) | Boundary F1 | Objective |
|---|---:|---:|---:|---:|---:|
| مرجع تاریخی Exp08 epoch15 | 1.664553 | — | — | 0.822263 | 2.020027 |
| Exp16 best-selector / epoch16 | **1.604478** | 3.351475 | +0.037588 | **0.827333** | **1.949813** |

بهبودها:

- MAE به اندازه‌ی 0.060076 میلی‌متر، معادل حدود 3.61٪، کمتر شد.
- Boundary F1 حدود 0.00507 افزایش یافت.
- Objective حدود 0.070215 کمتر شد.
- bias تقریباً خنثی است؛ بنابراین بهبود با یک جابه‌جایی سیستماتیک بزرگ حاصل
  نشده است.

هر سه شرط promotion پاس شدند: MAE حداکثر 1.664553، boundary F1 حداقل 0.82 و
objective حداکثر 2.020027.

## نتیجه‌ی همه‌ی checkpointها تحت profile قفل‌شده

| Candidate | MAE | Boundary F1 | Objective | تصمیم |
|---|---:|---:|---:|---|
| best_objective (epoch11) | 2.220655 | 0.775253 | 2.670150 | رد |
| best_study (epoch11) | 2.220655 | 0.775253 | 2.670150 | رد |
| best_study_boundary (epoch11) | 2.220655 | 0.775253 | 2.670150 | رد |
| best_selector_auc (epoch16) | **1.604478** | **0.827333** | **1.949813** | قبول |
| epoch013 | 1.883834 | 0.792822 | 2.298190 | رد |
| epoch015 | 1.934129 | 0.750469 | 2.433191 | رد |
| epoch017 | 1.655361 | 0.729885 | 2.195591 | رد |
| epoch019 | 1.948001 | 0.748599 | 2.450802 | رد |
| epoch021 | 1.959859 | 0.748252 | 2.463355 | رد |
| epoch023 | 1.722596 | 0.759475 | 2.203647 | رد |

## چرا metric آنلاین epoch11 انتخاب نهایی نشد؟

در training، epoch11 بهترین metric آنلاین را نشان داد: study MAE حدود 1.134،
boundary F1 حدود 0.879 و objective حدود 1.423. بااین‌حال این ارزیابی آنلاین با
مسیر full-series production یکسان نیست؛ ترکیب validation loader، پوشش sliceها
و aggregation آن برای انتخاب سریع checkpoint طراحی شده است. audit مستقل تمام
سری هر 70 مطالعه را دوباره از DICOMهای معتبر عبور می‌دهد و سپس profile production
ثابت را اعمال می‌کند. در این معیار معتبر، epoch11 به MAE 2.221 افت کرد، درحالی‌که
epoch16 هر سه gate را پاس کرد. بنابراین اتکا به metric آنلاین به‌تنهایی یک انتخاب
اشتباه می‌ساخت و checkpoint epoch11 به‌درستی رد شد.

## diagnostic پژوهشی مهم

بهترین ترکیب همان-fold از grid، checkpoint epoch11 با `topk=3`، gate=0.6،
quantile=0.65 و بدون probability weighting بود: MAE=1.3713، boundary F1=0.8565
و objective=1.6582. این نتیجه جذاب است اما چون روی خود fold0 انتخاب شده، قابل
استفاده در production نیست. تنها مسیر معتبر برای ارتقای آن، فریزکردن کامل این
profile و آزمایش transfer روی foldهای دیگر بدون retune است.

## یک نقص عملیاتی کشف‌شده

پس از اتمام train، لاگ، آزادشدن GPU، حذف run lock و وضعیت رسمی MLflow همگی پایان
موفق را تأیید کردند، اما فایل status محلی launcher روی `running` باقی ماند. این
stale status به مدل یا MLflow آسیب نزد، ولی launcher باید طوری اصلاح شود که
status نهایی را در برابر race/قطع worker مقاوم‌تر بنویسد و یک reconciliation
read-only با MLflow داشته باشد. تا آن زمان، status محلی به‌تنهایی منبع حقیقت
پایان run نیست.

## مصنوعات ماندگار

- checkpoint محلی: `checkpoint/mls/mls-vast-exp16-w32-fold0-strict-ensemble-refresh/mls_multitask_best_selector_auc.pth`
- SHA256: `bddcda5013cb88905a421095e71a28189181fde657aa3576be88f276d88ad15b`
- گزارش training و 23 epoch metrics
- audit status و prediction/metrics هر ده checkpoint
- grid کامل 60,480 ترکیب pooling
- promotion gate و aggregate metrics ماشین‌خوان

## ادامه‌ی مسیر

1. fold0 مدل ensemble را در یک submission tree ایزوله با checkpoint منتخب جایگزین
   و exact packaged-runtime را روی GPU و تمام 70 مطالعه‌ی fold0 اعتبارسنجی کنیم.
2. گزارش و معیارهای aggregate را به همان MLflow run آپلود کنیم؛ raw medical
   predictions نباید به MLflow ارسال شوند.
3. در صورت parity کامل package، همین recipe بدون تغییر training factor روی fold1
   به‌عنوان Exp17 اجرا شود تا دومین عضو تاریخی ensemble نیز نوسازی شود.
4. profile diagnostic قوی epoch11 فقط به‌صورت frozen cross-fold transfer بررسی
   شود؛ تا قبل از موفقیت روی foldهای دیگر، production profile تغییر نکند.
5. سرور Vast تا هماهنگی کاربر و validation واقعی leaderboard روشن بماند و destroy
   نشود.
