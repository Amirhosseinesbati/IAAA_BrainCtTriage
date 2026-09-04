# گزارش ارزیابی اکتشافی A10 — حفظ مرز ۳ میلی‌متر

## تصمیم

**A10 برای promotion، ساخت ZIP یا submission مجاز نیست.**

این تصمیم به‌دلیل شکست gate اجباری `F1@3mm` گرفته شد، نه به‌دلیل loss آموزش، سرعت اجرا، یا یک مقایسهٔ غیرهم‌ران‌تای تاریخی. A10 فقط یک screen اکتشافی روی fold0/seed42 بود؛ چون همان fold در ساخت فرضیه دخیل بود، حتی عبور از همهٔ gateها نیز به‌تنهایی برای انتشار کافی نبود.

## سؤال A10 و مداخلهٔ انجام‌شده

A9 در همان runtime، MAE و مرز ۵ میلی‌متر را بهبود داد اما در مرز حساس ۳ میلی‌متر افت کرد. A10 همان baseline واجدشرایط را کاملاً freeze کرد و فقط refiner بیرونی 47,617 پارامتری را برای 10 epoch / 1,690 optimizer step آموزش داد. تفاوت A10 با A9 فقط loss حفظ مرز ۳ میلی‌متر بود:

- `weight = 0.10`
- `truth margin = 0.25 mm`
- `prediction safety margin = 0.25 mm`
- decoder، pooling، threshold، split، precision و checkpoint epoch همگی ثابت ماندند.

پس این آزمایش پاسخ مشخصی می‌دهد: آیا این regularizer محدود می‌تواند gainهای A9 را نگه دارد و افت `F1@3mm` را بدون دست‌کاری inference جبران کند؟

## نتیجهٔ CUDA هم‌ران‌تای 70 مطالعه

| معیار | baseline واجدشرایط | A9 | A10 | A10 − baseline | A10 − A9 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MAE (mm) | 1.470565 | 1.448133 | **1.441921** | **-0.028644** | -0.006212 |
| RMSE (mm) | 2.440431 | 2.435763 | **2.426710** | -0.013721 | -0.009053 |
| F1 @ 1 mm | 0.820513 | 0.825000 | 0.825000 | +0.004487 | 0.000000 |
| F1 @ 3 mm | **0.819672** | 0.786885 | 0.786885 | **-0.032787** | 0.000000 |
| F1 @ 5 mm | 0.736842 | 0.810811 | **0.810811** | +0.073969 | 0.000000 |
| Boundary-F1 | 0.778257 | 0.798848 | **0.798848** | +0.020591 | 0.000000 |
| objective | 1.914051 | 1.850437 | **1.844225** | **-0.069826** | -0.006212 |

گیت‌ها به‌ترتیب MAE، F1@5، Boundary-F1 و objective را پاس کردند؛ تنها `F1@3mm >= 0.819672` fail شد. بنابراین A10 در خروجی پیوسته کمی بهتر از A9 است، اما classification مرزی A9 را در هیچ‌یک از `F1@1/3/5` یا Boundary-F1 تغییر نداد. این یعنی retention loss با تنظیم ثابت‌شدهٔ حاضر برای انتقال مطالعات حساسِ ۳ میلی‌متری کافی نبوده است.

## تفسیر صحیح

1. **ظرفیت صفر نیست.** A9 و A10 هر دو در runtime یکسان، MAE/objective/Boundary-F1/F1@5 را بهتر از baseline کردند. بنابراین گفتنِ «هیچ signalی وجود ندارد» با داده‌ها سازگار نیست.
2. **اما مسیر A9/A10 برای هدف اصلی کافی نیست.** بهبود پیوستهٔ slice-level یا MAE به‌خودی‌خود تضمین نمی‌کند decision سطح-study در آستانهٔ ۳ میلی‌متر بهتر شود. در A10، gain اضافه فقط `0.006212 mm` MAE بود و مرزهای طبقه‌بندی را جابه‌جا نکرد.
3. **baseline فعلاً انتخاب امن‌تر برای مرز ۳ میلی‌متر است.** A9/A10 در Boundary-F1 و مرز ۵ میلی‌متر بهترند، ولی برای سناریویی که ۳ میلی‌متر gate غیرقابل‌مذاکره است، baseline واجدشرایط برتر است.
4. **نباید A10 را روی fold0 tune کرد.** تغییر weight، margin، epoch، pooling یا threshold با دیدن این نتیجه، overfit کردن به همان evidence است. A10 باید در این hypothesis بسته تلقی شود.

## صحت اجرا و provenance

- checkpoint A10: `ced0b592b3b4483df21d6c67883de2f409d08f30e1bbabc4a6d4c6a4cfc899b2`
- parent training run: `e5d36fc3d8334a0c8bbe462647a794be`
- evaluation projection run: `a87307505ff54ec798b33814565915ff`
- aggregate audit SHA-256: `a760a1a48248c0d18307fc4a3b8cdc95a7361d63b128e042a16420d299114e0a`
- runtime: 47.65 ثانیه روی RTX 3090؛ peak VRAM حدود 0.452 GiB
- precision: FP32، بدون AMP و TF32
- shared evaluator SHA قبل و بعد از CUDA یکسان بود.
- MLflow projection readback با وضعیت `evaluation_projection_readback_verified` تأیید شد.
- predictionهای study-level خصوصی روی سرور باقی ماندند؛ نه به MLflow و نه به این worktree منتقل شدند.

دو اجرای ناموفق پیش از اجرای واقعی صرفاً bugهای wrapper بودند: (1) ردشدن بیش‌ازحد سخت‌گیرانهٔ symlink استاندارد و hash-pinned `Data/` و (2) فرض اشتباه object بودن `training_history.json`. هر دو پیش از ساخت مدل/CUDA متوقف شدند. پس از اصلاح و static preflight نهایی، execution سوم همان checkpoint ثابت را با موفقیت ارزیابی کرد.

## مسیر علمی بعدی، نه یک sweep دیگر

پژوهش هدفمند نشان می‌دهد که MLS معمولاً به‌صورت مسئلهٔ geometry با محور تقارن ایده‌آل و landmark/ventricle، یا با سطح midline سه‌بعدی و hemisphere segmentation مدل می‌شود؛ نه صرفاً یک regression مستقل برای هر slice. این جهت با شکست فعلی هم‌راستا است: مسئلهٔ باقی‌مانده selection/aggregation مطالعه‌ای در ناحیهٔ مرزی است. منابع اولیهٔ مفید:

- [Liao et al., 2018](https://doi.org/10.1155/2018/4303161) مرورِ روش‌های symmetry- و landmark-based برای MLS است.
- [Yan et al., 2022](https://doi.org/10.3390/diagnostics12030693) از keypoint detection برای اندازه‌گیری MLS روی NCCT استفاده کرد.
- [Wu et al., 2022](https://doi.org/10.1109/TMI.2022.3160184) midline را به‌صورت سطح 3D با hemisphere segmentation صورت‌بندی کرد.

فرضیهٔ بعدی باید **از A10 مستقل و از پیش ثبت‌شده** باشد: یک مسیر geometry-aware با context چندبرشی/3D-lite، landmark یا surface/midline auxiliary، و aggregator سطح-study که عدم‌قطعیت را صریحاً مدل کند. ابتدا feasibility داده و label contract بررسی می‌شود؛ سپس evaluation فقط روی fold استفاده‌نشده یا multi-fold leak-free انجام خواهد شد. تا آن زمان هیچ checkpoint A9/A10 به package یا submission اضافه نمی‌شود.

## فایل‌های شواهد عمومی

- `A10_EXPLORATORY_EVALUATION_RESULT_20260905.json`
- `A10_EVALUATION_BINDING_MANIFEST_20260905.json`
- `A10_EVALUATION_BINDING_STATUS_20260905.json`
- `A10_EVALUATION_STATIC_CONTRACT_PREFLIGHT_REMOTE_FINAL_20260905.json`
- `A9_PAIRED_RESULT_20260904.md`
- `A9_THRESHOLD_DIAGNOSTIC_20260905.json`
