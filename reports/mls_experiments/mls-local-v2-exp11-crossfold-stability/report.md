# Experiment 11 — تحلیل پایداری cross-fold، snapshot ensemble، SWA و calibration

**نوع آزمایش:** post-training stability analysis  
**دادهٔ ورودی:** predictionهای CUDA-only ذخیره‌شدهٔ Exp08/09/10  
**محاسبات مدل:** فقط سه ممیزی SWA روی GPU؛ blending، pooling محدود و آمار با I/O/aggregation سبک  
**حریم داده:** هیچ prediction در سطح مطالعه/برش یا دادهٔ پزشکی به MLflow ارسال نمی‌شود.

## سؤال آزمایش

آیا نوسان checkpointهای 13/15/17 را می‌توان با هزینه‌ای کمتر از آموزش معماری بزرگ‌تر کاهش داد؟ سه مسیر مقایسه شد:

1. ensemble در فضای prediction با mean/median؛
2. weight averaging همان checkpointها روی CUDA برای حفظ یک inference؛
3. offset calibration یک‌پارامتری در سطح study با انتخاب nested LOO.

baseline ثابت، hybrid epoch15 با پروفایل `severity_window(size=3, gate=0.5, min_active=3, q=0.75, weighted=true)` است.

## نتیجهٔ baseline

| معیار | مقدار |
|---|---:|
| Mean MAE | `1.5459 mm` |
| Worst-fold MAE | `1.7144 mm` |
| Mean Boundary-F1 | `0.8469` |
| Mean objective | `1.8521` |

## Snapshot blending در فضای prediction

| کاندیدا | Fold0 MAE | Fold1 MAE | Fold2 MAE | Mean | Worst | Boundary-F1 | Objective |
|---|---:|---:|---:|---:|---:|---:|---:|
| single epoch15 | `1.6646` | `1.2587` | `1.7144` | `1.5459` | `1.7144` | `0.8469` | `1.8521` |
| mean 13/15/17 | `1.5419` | `1.3679` | `1.5681` | `1.4926` | **`1.5681`** | `0.8269` | `1.8388` |
| median 13/15/17 | `1.5340` | `1.3144` | `1.6200` | **`1.4894`** | `1.6200` | **`0.8396`** | **`1.8102`** |

median نسبت به single، mean MAE را حدود `3.65%` و worst-fold را حدود `5.50%` بهتر می‌کند؛ Boundary-F1 فقط `0.0073` افت دارد و objective نیز بهتر است. این گزینه سه inference لازم دارد، پس کاندیدای accuracy است نه کاندیدای latency.

غربال محدود nested میان هشت blend و چهار profile به `Mean held-out MAE=1.5521` و `Worst=1.7999` رسید. چون خود shortlist پس از تحلیل قبلی foldها تعریف شده، این عدد برآورد تازه و کاملاً unbiased محسوب نمی‌شود؛ استفادهٔ آن فقط برای غربال بوده است.

## CUDA weight averaging

- برای هر fold، tensorهای epochهای 13/15/17 با وزن برابر روی CUDA میانگین شدند.
- هر checkpoint شامل 1938 tensor بود؛ همهٔ arithmetic روی GPU و اوج VRAM ساخت حدود `382 MB`.
- full-study audit: fold0 `70/70`، fold1 `67/67`، fold2 `67/67`، صفر failure و مجموع زمان `296.0 s`.

نتیجه روی همان profile ثابت:

| معیار | SWA | single epoch15 |
|---|---:|---:|
| Mean MAE | `1.4989` | `1.5459` |
| Worst-fold MAE | `1.7559` | **`1.7144`** |
| Mean Boundary-F1 | `0.8134` | **`0.8469`** |
| Objective | `1.8720` | **`1.8521`** |

SWA میانگین MAE را بهتر کرد، اما fold2، boundary و objective را بدتر کرد. بنابراین function-space ensemble به weight-space interpolation قابل تبدیل مستقیم نیست و SWA مساوی رد می‌شود. اجرای وزن 25/50/25 نیز با توجه به افت objective و سیگنال prediction-space ضعیف‌تر، مصرف GPU موجهی ندارد.

## Calibration یک‌پارامتری

offset فقط روی studyهایی اعمال شد که پیش از calibration فعال بودند؛ sentinel منفی `0.1 mm` دست‌نخورده ماند. offset هر held-out fold فقط با دو fold دیگر انتخاب شد.

برای single epoch15، هر سه انتخاب LOO به offset=`-0.1 mm` رسیدند:

- Mean held-out MAE=`1.5522`؛ اندکی بدتر از بدون calibration.
- Worst=`1.7182`.
- Boundary-F1=`0.8575`؛ حدود `0.0106` بهتر.
- Objective=`1.8372`؛ کمی بهتر از `1.8521`.

پس offset راه‌حل MAE نیست؛ فقط یک گزینهٔ boundary-oriented است و تا روشن‌بودن دقیق metric نهایی MLS نباید default شود. برای median ensemble نیز calibration سود material تازه‌ای ایجاد نکرد.

## یافتهٔ sampler و علت آزمایش بعدی

بازبینی کد نشان داد `balanced_sampling=True` فعلی فقط تعداد target/nontarget **برش‌ها** را برابر می‌کند؛ exposure مطالعه را برابر نمی‌کند. به‌علت تفاوت تعداد برش هر study، نسبت بیشترین به کمترین sampling mass در train برابر بود با:

- fold0: `7.19×`
- fold1: `7.90×`
- fold2: `7.87×`

سیاست جدید class→study→slice، ضمن حفظ mass برابر دو کلاس، هر study را درون هر کلاس برابر می‌کند و همین نسبت‌ها را به `2.94× / 2.90× / 2.90×` کاهش می‌دهد. این تغییر مستقیماً با گلوگاه cross-fold variance مرتبط است و هیچ جزء معماری، target یا loss را تغییر نمی‌دهد.

## تصمیم

1. single epoch15 کاندیدای سریع و پیش‌فرض باقی می‌ماند.
2. median 13/15/17 کاندیدای accuracy است و فقط در صورت عبور از latency/package gate وارد submission می‌شود.
3. SWA مساوی رد شد؛ checkpointهای آن فقط برای audit محلی حفظ می‌شوند و به MLflow آپلود نمی‌شوند.
4. offset `-0.1 mm` فقط گزینهٔ boundary-oriented است، نه calibration پیش‌فرض.
5. آزمایش آموزشی بعدی `study_class_balanced` است؛ ابتدا روی fold2 سخت با تمام تنظیمات Exp10 ثابت اجرا می‌شود. ادامه به fold0/1 مشروط به E2E بهتر یا دست‌کم کاهش variance بدون افت boundary است.
