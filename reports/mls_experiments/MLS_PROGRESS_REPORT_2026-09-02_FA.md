# گزارش جامع پیشرفت مدل MLS تا پایان ممیزی Exp18

تاریخ snapshot: 2026-09-02 (Asia/Tehran)

## خلاصه مدیریتی

تا این لحظه فرایند MLS از یک چرخه آموزش صرف به یک pipeline قابل ممیزی تبدیل
شده است: آموزش و inference مدل فقط روی CUDA اجرا می‌شوند، آزمایش‌ها طرح
ازپیش‌ثبت‌شده و گزارش ماندگار دارند، runها در MLflow ثبت می‌شوند، checkpointها
به‌جای اتکا به validation داخلی روی تمام مطالعه‌های held-out ارزیابی می‌شوند و
انتخاب production با یک pooling profile قفل‌شده انجام می‌شود.

دو ارتقای معتبر برای fold0 و fold2 به دست آمده‌اند. آزمایش Exp17 روی fold1 با
وجود validation امیدوارکننده، در ممیزی واقعی رد شد؛ همین ردکردن از انتخاب یک
مدل ظاهراً خوب ولی غیرقابل‌انتقال جلوگیری کرد. Exp18 با selector دو-head نیز
23/23 epoch را سالم تمام کرد و بهترین نتیجه online این مرحله را ثبت کرد، اما
در ممیزی کامل 804 ارزیابی CUDA شکست خورد و production promotion نشد.

## وضعیت فعلی در یک نگاه

| بخش | وضعیت مستند |
|---|---|
| Vast instance | `49527185`؛ روشن نگه داشته شده و stop/destroy نشده است |
| GPU | RTX 3060 12GB |
| policy محاسبه مدل | CUDA-only؛ بدون fallback forward/backward/validation/inference به CPU |
| قرارداد MLS data | 3484 ردیف، 338 مطالعه، 1781 target، 1703 non-target، تمام 3484 مسیر resolve شده و spacing/truth کامل |
| Exp18 training | completed، exit code صفر، 23/23 epoch |
| Exp18 full-study audit | completed، 804/804، صفر failure؛ promotion رد شد |
| Exp18 MLflow | run ID: `18474f1d10234ca5900caefe3f62c2eb` |
| Exp19 fold0 replication | training؛ status running، session حاضر، epoch1 کامل |
| Exp19 MLflow | run ID: `5383a78d31bf4a79a5bf6aff3c086e8c` |
| Exp18 server commit | `441eba6f699ccbc07bc958116571bbaf179001b9` |
| نتیجه leaderboard | هنوز هیچ نتیجه رسمی برای این challenger نداریم |

اعتبارسنجی بالا نبودن/کم‌بودن فایل را از دید قرارداد MLS، labelها، studyها و
مسیرهای مصرف‌شده رد می‌کند. این ادعا معادل checksum سراسری byte-for-byte همه
فایل‌های خام نیست؛ بنابراین از آن نتیجه‌ای بیشتر از پوشش کامل داده مورد استفاده
مدل نمی‌گیریم.

## کارهایی که انجام شده است

### 1. ساخت فرایند آزمایش قابل بازیابی

- برای هر run، manifest، plan ازپیش‌ثبت‌شده، گزارش epochها، وضعیت durable،
  checkpointهای named و snapshotهای دوره‌ای ثبت شده‌اند.
- اجرای Vast داخل tmux و launcher پایدار انجام می‌شود؛ پایان Exp18 با سه شاهد
  مستقل تأیید شد: `state=completed`، `exit_code=0` و حذف نشست tmux پس از پایان.
- گزارش‌ها و metricهای آموزش Exp18 از سرور به workspace محلی همگام شده‌اند؛
  تصمیم و failure analysis ممیزی نهایی نیز محلی ثبت شده و CSVهای خام پزشکی
  عمداً release/MLflow نشده‌اند.
- ثبت MLflow فعال بوده و پایان run Exp18 پس از آزادسازی CUDA و upload artifact
  با موفقیت گزارش شده است.

### 2. جلوگیری از ارزیابی گمراه‌کننده

validation داخلی برای انتخاب نهایی کافی نیست. در چند fold دیده شد checkpoint
برنده در validation لزوماً برنده full-study نیست. به همین دلیل:

- تمام candidateها روی همه studyهای held-out با inference واقعی CUDA ممیزی
  می‌شوند؛
- هزاران ترکیب pooling فقط برای تحلیل اجرا می‌شود؛
- promotion نهایی با profile قفل‌شده و thresholdهای ازپیش‌تعیین‌شده انجام
  می‌شود؛
- same-fold grid حق انتخاب production ندارد؛
- checkpoint آخر به‌طور خودکار انتخاب نمی‌شود.

### 3. نتایج معتبر foldهای فعلی

| fold / run | candidate | MAE (mm) | Boundary-F1 | objective | تصمیم |
|---|---|---:|---:|---:|---|
| fold2 / Exp15r | epoch17 | 1.548354 | 0.892593 | 1.763169 | promotion gate پاس شد |
| fold0 / Exp16 | best-selector-AUC، epoch16 | 1.604478 | 0.827333 | 1.949813 | هر سه gate پاس شد |
| fold1 / Exp09 incumbent | epoch15 با profile قفل‌شده | 1.258665 | 0.823729 | 1.611207 | هنوز عضو معتبر fold1 است |
| fold1 / Exp17 challenger | بهترین candidate قفل‌شده | 1.399901 | 0.793443 | 1.813016 | رد شد؛ هر سه gate را باخت |
| fold1 / Exp18 dual-selector | epoch21، بهترین candidate قفل‌شده | 1.392992 | 0.771236 | 1.850521 | رد شد؛ هر سه gate را باخت |

Exp15r روی 10 checkpoint و 670 study-checkpoint inference بدون failure ممیزی
شد. Exp16 نیز 10 checkpoint و 700 inference را بدون failure گذراند. Exp17 روی
11 checkpoint و 67 مطالعه، یعنی 737/737 inference CUDA، بدون failure اجرا شد؛
پس شکست Exp17 ناشی از خرابی pipeline نبود.

برای Exp17، grid کامل 6048 profile برای هر 11 checkpoint، در مجموع 66528 ردیف،
تحلیل شد. حتی profile انتخاب‌شده فقط از fold0/fold2 نیز Exp17 را نجات نداد.
blendهای 10، 25، 50 و 75 درصد Exp17 با Exp09 همگی نتیجه profile تولید را بدتر
کردند؛ حتی 90% Exp09 + 10% Exp17، MAE را از 1.258665 به 1.264027 و
Boundary-F1 را از 0.823729 به 0.805556 رساند. بنابراین Exp17 نه جایگزین مناسب
بود و نه مکمل مفید.

Exp18 نیز روی 12 checkpoint و تمام 67 مطالعه ممیزی شد: 804/804 ارزیابی واقعی
CUDA با صفر failure. برای هر checkpoint تعداد 6048 profile و در مجموع 72576
ردیف pooling تحلیل شد. بهترین checkpoint تحت profile تولید قفل‌شده epoch21 بود،
اما MAE=1.392992، Boundary-F1=0.771236 و objective=1.850521 ثبت کرد؛ بنابراین
هیچ‌کدام از سه شرط promotion را پاس نکرد. این ممیزی کامل نشان می‌دهد شکست
Exp18 را نمی‌توان به ناقص‌بودن داده، crash یا انتخاب تصادفی یک checkpoint نسبت
داد.

### 4. طراحی و پیاده‌سازی Exp18

علت مفهومی Exp17 این‌گونه فرض شد: selector تک‌خروجی مجبور بود هم‌زمان دو کار
متفاوت را انجام دهد—تشخیص وجود slice هدف برای gate/count و تخمین شدت نسبی برای
ranking/weighting. Exp18 این تعارض را به دو head جدا شکست:

1. presence head با target باینری، برای gate و حداقل تعداد slice فعال؛
2. peak/severity head با target نسبی، برای anchor، ranking و weighting.

میانگین دو BCE طوری نرمال شد که scale کلی loss selector نسبت به recipe قبلی
تقریباً ثابت بماند. backbone، fold، seed، sampler، optimizer، schedule، augmentation
و سایر عوامل عمداً ثابت نگه داشته شدند تا اثر تغییر معماری قابل تفسیر باشد.

تغییرات در config model، معماری، training، prediction، evaluator، pooling search،
blend و snapshot screening پیاده شد. دو تست جدید نیز اضافه شد:

- smoke واقعی CUDA که finite forward/loss/backward و gradient هر دو ردیف selector
  را تأیید کرد؛ peak VRAM آن حدود 0.936GB بود؛
- backward-parity مسیر pooling قدیمی که روی پیش‌بینی‌های ذخیره‌شده Exp09 دقیقاً
  MAE=1.2586648668 و Boundary-F1=0.8237288136 با اختلاف مطلق صفر بازتولید کرد.

### 5. نتیجه آموزش Exp18

Exp18 در حدود 1 ساعت و 27 دقیقه، با peak VRAM برابر 4.6476GB و بدون NaN، OOM
یا CPU model fallback پایان یافت. بهترین online validation در epoch12 بود:

| metric | Exp18 epoch12 |
|---|---:|
| study MAE | **0.899815 mm** |
| study Boundary-F1 | **0.847662** |
| online objective | **1.254940** |
| slice MAE | 1.974124 mm |
| keypoint MAE | 8.498 px |
| presence AUC | 0.899101 |
| peak/severity AUC | 0.796073 |
| presence selector F1 | 0.808759 |

برای مقایسه کنترل‌شده، Exp17 در همان fold و epoch12 به online MAE=0.98500،
Boundary-F1=0.84034 و objective=1.34408 رسیده بود. بنابراین Exp18 در proxy
داخلی حدود 0.0852mm MAE کمتر، 0.0073 Boundary-F1 بیشتر و 0.0891 objective کمتر
ثبت کرده است. این شواهد به نفع ایده dual-selector هستند، اما هنوز شاهد نهایی
انتقال نیستند.

epoch23 به MAE=1.024643، Boundary-F1=0.783636 و objective=1.511084 رسید و از
epoch12 ضعیف‌تر بود. این افت اشکال نیست، چون checkpointهای named و snapshotها
حفظ شده‌اند؛ همچنین یک شاهد دیگر است که انتخاب last epoch تصمیم نادرستی است.

### 6. نتیجه قطعی full-study برای Exp18

نتیجه online بالا به ارزیابی کامل منتقل نشد. بهترین checkpoint online یعنی
epoch12 با همان profile تولید، در تمام 67 مطالعه به MAE=1.939928،
Boundary-F1=0.755952 و objective=2.428023 رسید. در مقابل، epoch21 که online
برنده نبود، بهترین نتیجه قفل‌شده را با MAE=1.392992، Boundary-F1=0.771236 و
objective=1.850521 داد. این جابه‌جایی ranking اثبات می‌کند metric حین آموزش
برای انتخاب release کافی نیست.

بهترین نتیجه داخل grid همان fold برای checkpoint `best_objective` با
`severity_window(radius=2)`, gate=0.7، حداقل سه slice، quantile=0.65 و وزن‌دهی
احتمالی بود: MAE=1.127074، Boundary-F1=0.809942 و objective=1.507190. این نتیجه
نشان می‌دهد regression signal مفیدی در Exp18 وجود دارد، ولی دو محدودیت دارد:

- Boundary-F1 هنوز از کف قفل‌شده 0.82 پایین‌تر است؛
- profile روی همان fold انتخاب شده و شاهد بدون‌سوگیری برای production نیست.

در مقایسه با Exp17، Exp18/epoch21 حدود 0.0069mm MAE بهتر شد، اما
Boundary-F1 حدود 0.0222 و objective حدود 0.0375 بدتر شدند. بنابراین ایده
dual-selector از نظر proxy و بخشی از regression امیدبخش است، ولی در شکل فعلی
مسئلهٔ calibration/aggregation و مرزهای 3/5mm را حل نکرده است.

### 7. screen مؤلفه‌ای Exp09/Exp18

پس از بازتولید Exp09 روی RTX3060، 41 candidate قفل‌شده و یک profile guarded
تشخیصی برای هر candidate، در مجموع 82 ردیف، بررسی شدند. چهار mode و پنج alpha
پیش از مشاهده نتیجه ثابت شده بودند. هفت candidate هر سه gate را پاس کردند و
همه alpha=0.10 داشتند.

بهترین نتیجه عددی Exp18/epoch21 با 10٪ سهم challenger بود:

| ترکیب | MAE | Boundary-F1 | objective |
|---|---:|---:|---:|
| Exp09 بازتولیدشده | 1.259036 | 0.823729 | 1.611578 |
| 90% Exp09 + 10% Exp18/epoch21 | **1.248085** | **0.831034** | **1.586016** |

در هر چهار mode نتیجه دقیقاً یکسان بود؛ بنابراین selector، peak و heatmap
Exp18 در این سود دخیل نبودند و blend فقط `mls_mm` کافی است. نسبت به baseline
همان runtime، MAE حدود 0.01095mm، Boundary-F1 حدود 0.00731 و objective حدود
0.02556 بهتر شد. tie-break حداقل‌پیچیدگی candidate `regression_only` را انتخاب
کرد.

این یک سیگنال مکمل واقعی ولی same-fold است. تنها alpha کوچک 0.10 موفق بود و
غربال شامل 40 ترکیب غیرbaseline بود؛ بنابراین هنوز حق release یا leaderboard
ندارد. aggregateهای grid/summary/report به MLflow run Exp18 ارسال شدند و تمام
CSVهای study-level exclude شدند.

## چه چیزی اکنون بهترین مدل محسوب می‌شود؟

دو پاسخ متفاوت وجود دارد:

- **بهترین مدل اثبات‌شده fold1 برای production:** همچنان Exp09/epoch15 با
  MAE=1.258665، Boundary-F1=0.823729 و objective=1.611207 است.
- **وضعیت Exp18:** ممیزی نهایی آن شکست خورده است. epoch12 فقط برنده proxy حین
  آموزش بود و epoch21 بهترین candidate full-study شد، اما هیچ‌کدام مجوز release
  نگرفتند. به همین دلیل checkpoint Exp18 به پوشه مدل‌های release محلی منتقل نشد.
- **بهترین challenger ترکیبی تشخیصی:** Exp09 با 10٪ regression خروجی
  Exp18/epoch21 هر سه gate عددی fold1 را پاس کرد، ولی تا replication مستقل
  fold0 مدل production یا release محسوب نمی‌شود.

برای کل ensemble نیز هنوز ادعای «بهترین نهایی» مجاز نیست. مدل‌های fold0 و fold2
ارتقا یافته‌اند، اما تست تشخیصی قبلی نشان داد جایگزینی یک عضو می‌تواند Boundary-F1
را بهتر و MAE ensemble را بدتر کند. release واقعی نیازمند بازسازی کنترل‌شده سه
fold و سپس leaderboard است.

## مسائل زیرساختی حل‌شده و باقی‌مانده

- data contract روی سرور مجدداً پاس شد و تمام داده مورد نیاز MLS قابل resolve
  بود.
- MLflow preflight و run remote Exp18 موفق بودند. aggregateهای audit نهایی نیز
  با pending queue برابر صفر upload شدند؛ raw prediction و داده پزشکی عمداً
  upload نشد.
- GPU training سالم و مصرف حافظه بسیار پایین‌تر از 12GB بود؛ بنابراین فضای
  کافی برای audit وجود دارد.
- فایل‌های دقیق MLS روی local و server حاضرند. commit مستقل سرور ساخته شده است.
- push سرور به شاخه اختصاصی GitHub به‌علت نبود credential روی سرور انجام نشد.
  به‌دلیل dirty بودن worktree محلی و وجود تغییرات هم‌زمان ICH/submission، نباید
  commit یا merge کور انجام شود؛ فقط فایل‌های تحت مالکیت این آزمایش باید stage
  شوند.
- سرور Vast روشن مانده و بدون هماهنگی کاربر stop/destroy نخواهد شد.

## کار دقیق بعدی

1. Exp19 همان dual-selector را با تنها تغییر held-out fold از 1 به 0 آموزش دهد؛
   seed، معماری، losses، sampler و schedule ثابت بمانند. این run اکنون با
   commit `cb7ccf2` روی RTX3060 در حال اجرا است.
2. primary test از قبل قفل است: Exp16/best-selector به‌عنوان baseline، فقط
   Exp19/epoch21، فقط regression-only و alpha=0.10، بدون retune.
3. موفقیت نیازمند MAE و Boundary-F1 غیرضعیف‌تر و حداقل 0.01 بهبود objective روی
   تمام 70 مطالعه fold0 است. انتخاب checkpoint دیگر حق نجات primary test را
   ندارد.
4. اگر انتقال پاس شد، همان recipe روی fold سوم تأیید و سپس runtime/package
   دو-model بررسی شود. اگر رد شد، سیگنال fold1 overfit محسوب می‌شود و مسیر بعدی
   به 2.5D context و supervision صریح 3/5mm می‌رود.
5. فقط checkpoint/ensembleای که cross-fold، package CUDA و در نهایت leaderboard
   را پاس کند با SHA256 و README به پوشه release محلی منتقل شود.

## مسیرهای مرجع

- Exp18 plan: `reports/mls_experiments/mls-vast-exp18-w32-fold1-dual-selector-transfer/PREREGISTERED_PLAN.md`
- Exp18 training report: `reports/mls_experiments/mls-vast-exp18-w32-fold1-dual-selector-transfer/report.md`
- Exp18 epoch history: `reports/mls_experiments/mls-vast-exp18-w32-fold1-dual-selector-transfer/epoch_metrics.jsonl`
- Exp18 terminal status: `reports/mls_experiments/mls-vast-exp18-w32-fold1-dual-selector-transfer/training_status.json`
- Exp18 locked failure analysis: `reports/mls_experiments/mls-vast-exp18-w32-fold1-dual-selector-transfer/LOCKED_AUDIT_AND_FAILURE_ANALYSIS.md`
- Exp18 component screen: `reports/mls_experiments/mls-vast-exp18-w32-fold1-dual-selector-transfer/crossrun_component_blend_screen_exp09_exp18/`
- Exp19 preregistered plan: `reports/mls_experiments/mls-vast-exp19-w32-fold0-dual-selector-replication/PREREGISTERED_PLAN.md`
- Exp19 live status: `reports/mls_experiments/mls-vast-exp19-w32-fold0-dual-selector-replication/LAUNCH_STATUS.md`
- Exp17 failure analysis: `reports/mls_experiments/mls-vast-exp17-w32-fold1-strict-ensemble-refresh/LOCKED_TRANSFER_AND_BLEND_ANALYSIS.md`
- Exp15r gate: `reports/mls_experiments/mls-vast-exp15r-w32-fold2-strict-repro-control/promotion_gate.json`
- Exp16 gate: `reports/mls_experiments/mls-vast-exp16-w32-fold0-strict-ensemble-refresh/promotion_gate.json`
