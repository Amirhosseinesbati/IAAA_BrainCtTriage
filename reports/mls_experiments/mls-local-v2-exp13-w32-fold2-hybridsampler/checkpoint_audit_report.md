# ممیزی نهایی Exp13 — hybrid study/class-balanced sampler

## حکم کوتاه

Exp13 جای baseline را نمی‌گیرد. معیار اصلی ازپیش‌ثبت‌شده در epoch15 روی profile production ثابت به `MAE=1.7550 mm` و `Boundary-F1=0.8626` رسید؛ Exp10/epoch15 با همان profile `MAE=1.7144 mm` و `Boundary-F1=0.8947` داشت. بنابراین Exp13 در MAE حدود `0.0406 mm` یا `2.37%` بدتر، در Boundary-F1 حدود `0.0321` پایین‌تر و در objective حدود `5.44%` ضعیف‌تر است.

نتیجه: sampler میانی `hybrid_study_class_balanced` رد می‌شود، fold0/1 برای آن آموزش داده نمی‌شوند و baseline قابل‌دفاع فعلی همان خانوادهٔ Exp08/09/10 با epoch15 ثابت می‌ماند.

## سلامت اجرا و deviation

- مدل، validation و full-study inference فقط روی CUDA اجرا شدند؛ CPU fallback وجود نداشت.
- آموزش و validation تا پایان epoch19 کامل شد. metric داخلی epoch19 نیز در `epoch_metrics.jsonl` موجود است: `study MAE=1.5774` و `Boundary-F1=0.9345`.
- پس از validation epoch19، DNS مربوط به DagsHub هنگام `mlflow.log_metrics` قطع شد. چون logging شبکه‌ای در کد پیش از snapshot save قرار داشت، process با exit code 1 بسته شد و checkpoint epoch19 ذخیره نشد.
- snapshotهای معتبر 13/15/17 هر سه موجودند. checkpointهای دوره‌ای optimizer/scheduler state ندارند؛ در نتیجه restart از epoch17 ادامهٔ دقیق trajectory قبلی نبود و مطابق تصمیم توقف کنترل‌شده انجام نشد.
- E2E سه snapshot روی `201/201` study-checkpoint با صفر failure و مجموع runtime حدود `316.07 s` کامل شد. snapshotهای 19/21/23 در دسترس نیستند و این محدودیت در نتیجه حفظ شده است.

## نتیجه روی profile production قفل‌شده

Profile: `severity_window(size=3, gate=0.5, min_active=3, q=0.75, probability_weighted=true, guard=0)`.

| checkpoint | MAE (mm) | Boundary-F1 | bias (mm) | objective |
|---|---:|---:|---:|---:|
| Exp13 epoch13 | 1.7624 | 0.9074 | -0.4499 | 1.9476 |
| **Exp13 epoch15 — معیار اصلی** | **1.7550** | **0.8626** | **-0.5729** | **2.0297** |
| Exp13 epoch17 | 2.1050 | 0.8672 | -0.0936 | 2.3705 |
| Exp10 epoch15 — baseline | 1.7144 | 0.8947 | — | 1.9250 |

epoch13 در همین profile Boundary-F1 خوبی دارد، اما epoch ازپیش‌قفل‌شدهٔ معیار اصلی 15 بوده است. تغییر آزاد epoch پس از دیدن fold2 مجاز نیست و نمی‌تواند شکست primary را معکوس کند.

## معیار ثانویه frozen از fold0/1

profile قدیمی `relative_component(ratio=0.3, gate=0.7, min_active=1, q=0.75, weighted=true, guard=0)` روی Exp13/epoch13 به `MAE=1.9965 mm` و `Boundary-F1=0.9055` رسید. این نسبت به Exp10/epoch13 (`2.1446 / 0.8887`) حدود `6.90%` MAE بهتر است، اما از Exp12/epoch13 (`1.6906 / 0.9253`) حدود `18.10%` بدتر است. بنابراین representation نسبت به baseline قدیمی سیگنال مثبتی دارد، ولی primary شکست خورده و این معیار به‌تنهایی مجوز production یا آموزش دو fold دیگر نیست.

## diagnostic درون-fold

بهترین ردیف grid برای Exp13/epoch13 از خانوادهٔ top-k با `size=7, gate=0.4, min_active=3, q=0.9, unweighted, guard=0.5` بود: `MAE=1.5377 mm`، `Boundary-F1=0.9227` و objective=`1.6923`. این profile روی همان fold انتخاب شده و صرفاً diagnostic است؛ برای تست ناشناخته یا submission قفل نمی‌شود.

## تفسیر

Exp13 فرضیهٔ میانی را به‌درستی آزمود: exposure مطالعه‌ها از نسبت `7.87×` در Exp10 به `4.72×` کاهش یافت، درحالی‌که Exp12 تا `2.90×` پایین می‌رفت. activation مطالعه در epoch8 مشابه Exp10 و سریع‌تر از Exp12 رخ داد و proxyهای epoch9/12–15 قوی بودند. بااین‌حال این بهبود proxy در checkpoint ثابت epoch15 به full-study production منتقل نشد. نتیجه نشان می‌دهد دست‌کاری sampler به‌تنهایی calibration/aggregation قابل‌انتقال را حل نمی‌کند.

## مشکل زیرساختی ثبت‌شده برای آینده

قبل از هر آموزش بعدی باید ترتیب و دوام trainer اصلاح شود:

1. snapshot/report محلی قبل از هر فراخوانی شبکه‌ای ذخیره شود؛
2. MLflow logging با retry/deferred queue غیرکشنده شود؛
3. resume checkpoint شامل model، optimizer، scheduler، scaler، epoch و RNG state باشد؛
4. قطع MLflow نباید یک آموزش سالم CUDA را متوقف کند.

تا انجام این موارد و ازسرگیری صریح goal، هیچ آزمایش جدیدی آغاز نمی‌شود.
