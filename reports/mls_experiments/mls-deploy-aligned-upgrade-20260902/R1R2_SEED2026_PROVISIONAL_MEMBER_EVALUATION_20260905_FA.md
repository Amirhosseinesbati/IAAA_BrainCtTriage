# R1R2 / fold1 / seed2026 — ارزیابی provisional تک‌عضو

تاریخ: ۲۰۲۶-۰۹-۰۵  
وضعیت: `artifact-first`؛ candidate به‌صورت provisional محلی منتقل شد، اما promotion یا submission مجاز نیست.

## قرارداد و حدود اعتبار

- آموزش از checkout مهروموم‌شدهٔ R1R2 در commit `648ddf6` انجام شد.
- هر دو checkpoint در epoch ثابت ۱۵، fold ۱ و seed `2026` هستند.
- تنها اختلاف recipe: `horizontal_flip_prob`: control=`0.0` و candidate=`0.5`.
- evaluator عمومی CUDA در commit `0fd8375`، به‌طور صریح با `--expected-members 1` اجرا شد؛ بنابراین protocol خروجی `heldout_fold_fixed_epoch15_single_member_provisional` است، نه gate مهروموم‌شدهٔ ensemble سه-seed.
- هر arm روی تمام ۶۷ study held-out، با raw DICOM و همان fold manifest/truth table قرارداد ارزیابی شد. predictionهای خام به MLflow آپلود نشدند.

## checksumها

| مورد | SHA-256 |
| --- | --- |
| control epoch15 | `b516617d3bc307326e3186e8153ec83070314e48e7b734c95747a9626f9c228b` |
| candidate epoch15 | `9b6822cb1fa022c1cd8df7e91100f20ad986737131ace7ec2fd609aeef641c87` |
| control private prediction receipt | `33716debd2a8fd44055dccf19a912475ad738aa70095c86f4ddb0ca2e4b1bc9c` |
| candidate private prediction receipt | `a700861509d9a33011b016c6e3f50b25413c77fe2f1134c3bc1e2478ae5c51f9` |

## نتیجهٔ جفت‌شده

| معیار raw-DICOM | control | candidate | delta candidate-control |
| --- | ---: | ---: | ---: |
| MAE (mm) | 1.938867 | 1.682207 | -0.256660 |
| Boundary-F1 | 0.496101 | 0.727941 | +0.231840 |
| F1@1mm | 0.640000 | 0.766667 | +0.126667 |
| F1@3mm | 0.473684 | 0.750000 | +0.276316 |
| F1@5mm | 0.518519 | 0.705882 | +0.187364 |
| selection objective | 2.946664 | 2.226325 | -0.720340 |

candidate در تمام معیارهای ازپیش‌تعیین‌شدهٔ screen از control هم‌seed بهتر شد. بنابراین مدل زیر بلافاصله برای آزمایش ensemble/threshold به‌عنوان artifact قابل استفاده است:

`checkpoint/mls/mls-r1r2-reflection-candidate-fold1-seed2026-epoch15/`

## تشریح خطای جفت‌شده (فقط برای تصمیم‌سازی)

یک aggregation سبک روی receipt خصوصیِ همان ۶۷ study انجام شد؛ این تحلیل inference تازه‌ای نیست و prediction خام نیز منتقل یا upload نشده است. اعداد زیر MAE گروهی (mm) هستند و برای نتیجه‌گیری نهایی آماری کافی نیستند:

| گروه حقیقت | n | candidate | control | delta candidate-control |
| --- | ---: | ---: | ---: | ---: |
| triage class 0، MLS <3mm | 33 | 0.4864 | 0.0000 | +0.4864 |
| triage class 1، MLS ≥3mm | 9 | 2.0135 | 3.4073 | -1.3937 |
| triage class 2، MLS ≥3mm | 19 | 3.5966 | 4.7106 | -1.1141 |

برداشت عملی: reflection عمدتاً در shiftهای clinically salient کمک کرده، اما برای class-0 کم‌شیفت calibration را بدتر کرده است. پس این checkpoint حتی اگر در gate سه-seed مدل مستقل برتر نشود، candidate معقولی برای ensemble یا threshold-aware routing است. به‌علت کوچک بودن گروه‌ها و ناسازگاری سطح مطلق raw evaluator با proxy، این بخش **هیچ promotion یا ادعای Macro-F1/Urgent-F1** را توجیه نمی‌کند؛ فقط فرضیهٔ audit بعدی را دقیق‌تر می‌کند.

## ثبت MLflow

metrics aggregate-only و receipt candidate به run آموزش زیر افزوده شدند:

`951fa72b590d47c4bb4462fa258a69bf`

predictionهای خصوصی هرگز upload نشدند.

## محدودیت مهم و مرحلهٔ بعد

سطح مطلق این evaluator با proxyهای داخل آموزش هم‌خوان نیست؛ در نتیجه فقط **delta جفت‌شده** در این مرحله قابل استفاده است. علت باید قبل از promotion نهایی با audit runtime/deploy بررسی شود. این مدل هنوز فقط یک seed و یک fold است و Macro-F1/Urgent-F1 triage نهایی را اثبات نمی‌کند.

مرحلهٔ بعدیِ ازپیش‌ثبت‌شده: تکمیل control/candidate برای seed `3407` با همان recipe و سپس gate سه-seed و triage مهروموم‌شده. تا آن زمان ساخت ZIP submission ممنوع است.
