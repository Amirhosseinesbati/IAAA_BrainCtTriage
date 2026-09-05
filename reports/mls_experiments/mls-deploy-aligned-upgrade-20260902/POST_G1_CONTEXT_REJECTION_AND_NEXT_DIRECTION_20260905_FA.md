# تصمیم پس از G1 — رد context سه‌برشی و مسیر بعدی

> وضعیت: تصمیم ثبت‌شده؛ هیچ checkpoint جدیدی از G1 release، package یا submission-eligible نیست.
>
> تاریخ: 2026-09-05 (Asia/Tehran)

## نتیجهٔ قطعی G1

G1 یک آزمون علّی، هم‌شرایط و از پیش ثبت‌شده بود؛ نه یک مقایسهٔ post-hoc.
هر دو بازو روی fold=3، seed=42، epoch=15، با cache، split، loss، selector،
pooling و hardware یکسان اجرا شدند. تنها تفاوت موردنظر، ورودی بود:

| بازو | کانال ورودی | تفسیر |
| --- | ---: | --- |
| C0 | 3 | تصویر مرکزی با سه window تاریخی |
| A | 9 | سه slice مجاور، هر کدام با همان سه window |

در حالی که A خطای slice را کاهش داد، در معیارهای study که برای triage معنادارند
افت کرد:

| معیار در epoch ثابت 15 | C0 مرکزی | A سه‌برشی | A − C0 |
| --- | ---: | ---: | ---: |
| Slice MLS MAE (mm؛ کمتر بهتر) | 2.541216 | **2.334665** | -0.206551 |
| Study MLS MAE (mm؛ کمتر بهتر) | **2.915887** | 3.175073 | +0.259186 |
| Study F1@3 mm | **0.857143** | 0.807692 | -0.049451 |
| Study F1@5 mm | **0.857143** | 0.736842 | -0.120301 |
| Study Boundary-F1 | **0.857143** | 0.772267 | -0.084876 |
| selection objective (کمتر بهتر) | **3.236042** | 3.670488 | +0.434446 |
| selector AUC | **0.931119** | 0.920102 | -0.011018 |

بنابراین **G1-A رد می‌شود**. seed یا fold یا epoch دیگری برای نجات دادن این
فرضیه اجرا نمی‌شود. این یک ردِ input-context سادهٔ ±1 slice در همین contract است؛
ردّ همهٔ روش‌های 3D، همهٔ fusionهای چندslice یا همهٔ مدل‌های geometry-aware نیست.

## شواهد و provenance

- cache manifest: `c50ece4167b25661a7e36305bfab6f177253981c8418c0fd85acbf23bde4e672`
- receipt معتبر cache: `c0812f85d74ae6759a6d7c5ca473826de30923b7552d2878ac0b381d0ba8beb1`
- preregistration matrix: `d9f49e109bdcdd75a03d7e8d81520738a3b31ec68966d72c0103df6d360917a5`
- C0 checkpoint epoch 15:
  `cc7aafc92f0e5c8b122230ef00329bf2cd7df3ae9ed5dca3aa0b0ef6278c6b05`
  (MLflow run `1b065d49d930432e977addb430a1d57c`)
- A checkpoint epoch 15:
  `1e1ace96b948e57ebd2b2774e9dae5540e29cd9d5a39425e7cc13c05dba54c52`
  (MLflow run `2c890e3fa6cc47aea661f92a6a6918f6`)
- هر دو checkpoint با `weights_only=True` برای source-map بررسی شدند و با
  18 hash از matrix از پیش ثبت‌شده دقیقاً تطابق داشتند.

فایل‌های بزرگ prediction خصوصی هستند و فقط روی سرور/MLflow باقی می‌مانند؛ این
سند فقط metric و receipt عمومی را ثبت می‌کند.

## برداشت علمی درست

1. کاهش MAE در سطح slice تضمین نمی‌کند peak selector و pooling در سطح study
   بهتر شوند. G1 نمونهٔ مستقیم این اختلاف هدف است.
2. در نتیجه اضافه‌کردن context به trunk فعلی احتمالاً noise یا anatomy مجاور
   نامربوط را وارد selector می‌کند، یا calibration رتبه‌بندی sliceها را تغییر
   می‌دهد. این علت هنوز یک فرضیه است، نه یافتهٔ اثبات‌شده.
3. تکرار A1 (ordinal auxiliary فقط در training)، A2/A6 (loss geometry)،
   A3 (study bag)، A4/A5 (ranking)، A7 (translation consistency)، A8–A10
   (reference refiner) یا G1 با نام جدید، اطلاعات علمی تازه تولید نمی‌کند و
   ممنوع است.

## وضعیت هدف نهایی

هیچ مدل MLS جدیدی هنوز به‌طور leak-free و deploy-aligned، `Macro-F1` نهایی
triage و مخصوصاً `Urgent-F1` را نسبت به frozen Champion بهتر نکرده است. C0 یک
control معتبر است، نه release candidate. پس ZIP ساخته نمی‌شود و checkpointی از
G1 به پوشهٔ best-model محلی منتقل نمی‌گردد.

## جهت بعدیِ مجاز

پژوهش پس از G1 باید یک **mechanism جدید در inference** ارائه دهد که:

- از geometry keypoint فعلی برای MLS-mm پیوسته جدا نشود؛
- صرفاً auxiliary loss یا تغییر threshold/pooling نباشد؛
- با A8–A10 (refinement heatmap مبتنی بر feature) و G1 (stack ورودی خام)
  هم‌ارز نباشد؛
- contract کامل inference، checkpoint و CUDA-only evaluator داشته باشد؛
- پیش از training با یک control هم‌شرایط و screen روی fold استفاده‌نشده
  preregister شود.

دو ردهٔ قابل‌بررسی، **نه مجوز اجرای خودکار**، عبارت‌اند از: (الف) یک cascade
واقعی coarse-to-fine با مشاهدهٔ image-level مستقل و خروجی landmark در inference؛
و (ب) مدل‌سازی صریح axis/midline یا symmetry که فقط در صورت وجود supervision یا
اعتبارسنجی مستقل کافی اجرا شود. ادبیات MLS نیز میان keypoint/landmark،
coarse-to-fine fusion و midline-surface تمایز می‌گذارد؛ این به‌تنهایی دلیل
کافی برای پیاده‌سازی نیست. Nguyen et al. یک pipeline cascaded و fusion چندslice
گزارش کرده‌اند، و Yan et al. اندازه‌گیری keypoint-based را گزارش کرده‌اند.

منابع اولیه: [Nguyen et al., ICCVW 2021](https://openaccess.thecvf.com/content/ICCV2021W/MIA-COV19D/html/Nguyen_Brain_Midline_Shift_Detection_and_Quantification_by_a_Cascaded_Deep_ICCVW_2021_paper.html)،
[Yan et al., Diagnostics 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC8947005/).

تا زمانی که data feasibility و یک pre-registration قابل‌اجرای چنین mechanismی
تکمیل نشده، اجرای GPU جدید انجام نمی‌شود؛ این توقف، جلوگیری از sweep کورکورانه
است نه پایان goal.
