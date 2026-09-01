# Exp10 fold2 — گزارش ممیزی کامل checkpoint و تصمیم cross-fold

**Run:** `a4c44492fcc141058e5aae71266c6c33`  
**معماری:** HRNet-W32 multitask با hybrid-soft selector target (`peak_base=0.75`)  
**Fold:** 2، تعداد 67 مطالعه  
**سیاست محاسباتی:** تمام forwardهای مدل فقط روی CUDA؛ هیچ inference یا fallback مدل روی CPU انجام نشد. CPU صرفاً برای I/O و تجمیع کوتاه جدول‌ها استفاده شد.  
**وضعیت:** آموزش 23/23، MLflow و ممیزی E2E کامل.

## 1. صحت اجرا و قابلیت بازیابی

- شش snapshot ازپیش‌ثبت‌شدهٔ epochهای 13، 15، 17، 19، 21 و 23 ارزیابی شدند.
- هر snapshot روی هر 67 مطالعه اجرا شد: در مجموع `402/402` ارزیابی موفق و `0` failure.
- زمان‌های inference به‌ترتیب حدود `104.8`، `100.5`، `103.7`، `99.7`، `102.9` و `103.3` ثانیه؛ مجموع حدود `614.7` ثانیه.
- آموزش و ارزیابی بدون OOM، NaN یا CPU fallback بود. اوج حافظهٔ گزارش‌شدهٔ PyTorch حدود `4.55 GB` و اوج درایور حدود `5.5 GB` روی GTX 1660 Ti Max-Q 6GB بود.
- فایل‌های prediction در سطح مطالعه/برش فقط محلی نگه داشته شدند و به MLflow ارسال نشدند.

## 2. نتیجهٔ proxy داخلی آموزش

ناحیهٔ قوی داخلی در epochهای 14 تا 17 شکل گرفت:

| Epoch | Study MAE | Boundary-F1 | Objective | برداشت |
|---:|---:|---:|---:|---|
| 13 | `1.947` | `0.860` | `2.283` | اولین snapshot، قابل استفاده ولی ضعیف‌تر از basin میانی |
| 14 | `1.518` | `0.957` | **`1.647`** | بهترین objective/boundary داخلی |
| 15 | **`1.476`** | `0.922` | `1.676` | بهترین MAE داخلی |
| 17 | `1.546` | `0.912` | `1.765` | هنوز در basin قوی |
| 19 | `1.769` | `0.908` | `2.001` | شروع افت late |
| 21 | `1.809` | `0.866` | `2.133` | افت late تأیید شد |
| 23 | `1.716` | `0.875` | `2.017` | recovery ناکامل؛ ادامهٔ آموزش موجه نبود |

نتیجهٔ proxy: ادامه بعد از epoch17 کیفیت پایدار تازه‌ای نساخت. بااین‌حال proxy برای انتخاب نهایی کافی نبود و همهٔ snapshotهای preregistered وارد E2E شدند.

## 3. ممیزی full-study و grid مشترک fold2

برای هر snapshot دقیقاً همان 6048 پروفایل pooling مورد استفاده در fold0 و fold1 بررسی شد:

| Snapshot | بهترین MAE متوازن | Boundary-F1 | Objective |
|---:|---:|---:|---:|
| 13 | `1.7912` | `0.9296` | `1.9320` |
| 15 | `1.6669` | `0.9094` | `1.8482` |
| 17 | **`1.5998`** | **`0.9412`** | **`1.7174`** |
| 19 | `1.8796` | `0.9006` | `2.0784` |
| 21 | `1.6380` | `0.9048` | `1.8285` |
| 23 | `1.6676` | `0.9094` | `1.8489` |

برندهٔ in-fold در fold2، epoch17 با پروفایل `smooth_component`، نسبت `0.7`، gate=`0.4`، حداقل سه برش فعال، quantile=`0.9` و weighting فعال است. این نتیجه فقط diagnostic است؛ چون epoch و profile با نگاه به خود fold2 انتخاب شده‌اند، برای production lock نمی‌شود.

## 4. mismatch بین proxy و E2E

- proxy داخلی epoch14/15 را برتر می‌دید، ولی grid مخصوص fold2 epoch17 را انتخاب کرد.
- checkpointهای 19 تا 23 با وجود localization قابل‌قبول، از نظر aggregation پایدار نبودند.
- بنابراین representation مکانی collapse نکرده است؛ نوسان اصلی در calibration selector، gate و نحوهٔ تبدیل predictionهای برش به عدد مطالعه رخ می‌دهد.
- نتیجهٔ روش‌شناختی: best checkpoint داخلی، best checkpoint in-fold E2E و best checkpoint transferable سه مفهوم متفاوت‌اند.

## 5. تحلیل strict cross-fold روی foldهای 0/1/2

سه grid کامل foldها با کلید یکسان join شدند؛ هیچ inference تازه‌ای اجرا نشد. تعداد ترکیب‌های هم‌تراز `36,288 = 6 snapshot × 6,048 profile` بود.

### 5.1 انتخاب آزاد هم‌زمان epoch و profile

| معیار انتخاب nested | Mean held-out MAE | Worst held-out MAE | Mean Boundary-F1 |
|---|---:|---:|---:|
| MAE-first | `1.9927` | `2.6559` | `0.7971` |
| Boundary-first | `1.7903` | `1.9302` | `0.8040` |
| Balanced | `1.8111` | `1.9302` | `0.8135` |

افت MAE-first ناشی از خود مدل نیست: دو fold دیگر epoch19 را انتخاب کردند، اما همان انتخاب روی fold2 به `2.6559 mm` سقوط کرد. آزادی انتخاب snapshot با فقط دو fold برای selection، variance بیش‌ازحد می‌سازد.

### 5.2 ثابت‌کردن epoch قبل از انتخاب pooling

| Epoch ثابت | Nested balanced Mean MAE | Worst MAE | Mean Boundary-F1 |
|---:|---:|---:|---:|
| 13 | `2.0539` | `2.6230` | `0.7497` |
| **15** | **`1.6273`** | **`1.7999`** | **`0.8208`** |
| 17 | `1.9708` | `2.2689` | `0.7359` |
| 19 | `1.9338` | `2.6559` | `0.7645` |
| 21 | `1.7704` | `2.1038` | `0.8208` |
| 23 | `1.8428` | `2.3205` | `0.7574` |

برای epoch15، سیاست boundary-first حتی به `Mean MAE=1.6045 mm`، `Worst=1.7999 mm` و `Mean Boundary-F1=0.8344` رسید. بنابراین قفل‌کردن checkpoint به epoch15 هم MAE و هم پایداری را به‌وضوح بهتر می‌کند.

### 5.3 نتیجهٔ diagnostic با استفاده از هر سه fold

بهترین پروفایل مشترک diagnostic نیز epoch15 را انتخاب کرد:

- `severity_window`, size=`3`, gate=`0.5`, min-active=`3`, quantile=`0.75`, probability-weighted=`true`, guard=`0`.
- fold0 MAE=`1.6646`، fold1=`1.2587`، fold2=`1.7144`.
- Mean MAE=`1.5459`، Worst MAE=`1.7144`، Mean Boundary-F1=`0.8469`، objective=`1.8521`.

این عدد برآورد نهایی unbiased نیست، ولی برای انتخاب hyperparameter روی همهٔ validation foldها و سپس آموزش/ensemble روی کل داده مناسب است. برآورد محافظه‌کارانه همان nested epoch15 است.

## 6. مقایسه با نسل قبلی

| روش | Diagnostic Mean MAE | Diagnostic Worst | Mean Boundary-F1 | Strict/Nested Mean MAE |
|---|---:|---:|---:|---:|
| Peak-aware single | `1.5806` | `1.8821` | `0.8143` | `1.7426` |
| Binary + peak-aware ensemble | `1.5097` | `1.8083` | `0.8186` | `1.6350` |
| Hybrid epoch15 single | `1.5459` | `1.7144` | `0.8469` | `1.6273` balanced / `1.6045` boundary-first |

برداشت:

- hybrid single در diagnostic نسبت به peak-aware single حدود `2.2%` mean-MAE و `8.9%` worst-fold بهبود دارد و Boundary-F1 را `0.0326` افزایش می‌دهد.
- پروفایل epoch15 که فقط با fold0/1 انتخاب شد، روی fold2 به `1.7999 mm` و Boundary-F1=`0.8754` رسید؛ در برابر peak-aware frozen fold0/1→fold2 با `1.9744 mm` و Boundary-F1≈`0.8658`، حدود `8.8%` MAE بهتر است.
- hybrid epoch15 single از نظر strict MAE با ensemble دو-مدلی قبلی هم‌سطح یا بهتر است، Boundary-F1 بالاتری دارد و فقط یک inference لازم دارد.

## 7. تشخیص علت و تصمیم

1. معماری فعلی شکسته نیست؛ بهبود روی fold2 مستقل منتقل شده است.
2. مشکل اصلی باقی‌مانده، ناپایداری checkpoint/calibration است، نه کمبود ظرفیت HRNet-W32.
3. ادامهٔ کورکورانهٔ epochهای بیشتر رد می‌شود؛ epochهای 18 تا 23 سود transferable نساختند.
4. checkpoint پایهٔ دور بعد `epoch15` است. انتخاب epoch دوباره روی test/leaderboard انجام نمی‌شود.
5. پروفایل production فعلی برای single-model همان `severity_window(size=3, gate=0.5, min_active=3, q=0.75, weighted=true)` است؛ نسخهٔ guard=`0.5` یک گزینهٔ محافظه‌کار boundary برای بررسی نهایی است.
6. آزمایش بعدی باید variance را کاهش دهد، نه صرفاً مدل را بزرگ‌تر کند. گزینهٔ اول، weight averaging محدود در basin 13/15/17 یا calibration مشترک با یک متغیر اصلی است؛ فقط در صورت انتقال strict اجرا/حفظ می‌شود.
7. snapshot ensemble چند-inference‌ای فقط وقتی مجاز است که سود strict آن از single epoch15 معنادار باشد و محدودیت زمان leaderboard را پاس کند.

## 8. artifactهای مرجع

- `end_to_end_checkpoint_audit/epoch013` تا `epoch023`: prediction و metric محلی هر snapshot.
- `checkpoint_pooling_expanded/checkpoint_pooling_summary.json`: نتیجهٔ کامل fold2.
- `../hybridsoft_crossfold_snapshot_pooling/crossfold_snapshot_pooling_summary.json`: انتخاب strict و تحلیل ثابت هر epoch.
- raw prediction CSVها و gridهای بزرگ برای audit محلی حفظ می‌شوند و به MLflow ارسال نمی‌شوند.
