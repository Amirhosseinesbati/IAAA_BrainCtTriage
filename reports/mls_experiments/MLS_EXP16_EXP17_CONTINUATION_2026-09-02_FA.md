# گزارش ادامه MLS روی Vast — Exp16 package gate و شروع Exp17

زمان گزارش: 2026-09-02 (Asia/Tehran)

## نتیجه قطعی Exp16 روی fold0

مدل منتخب Exp16، checkpoint مربوط به `best_selector_auc` در epoch 16 است:

- run: `mls-vast-exp16-w32-fold0-strict-ensemble-refresh`
- MLflow run ID: `a2478b8410d74de2b2806ef08d79051d`
- SHA256: `bddcda5013cb88905a421095e71a28189181fde657aa3576be88f276d88ad15b`
- profile قفل‌شده: `severity_window`، radius=3، selector gate=0.5،
  min-active=3، weighted q0.75 و guard=0
- OOF fold0: MAE=`1.604478`، RMSE=`3.351475`، bias=`+0.037588`،
  Boundary-F1=`0.827333` و objective=`1.949813`

این مدل هر سه گیت ازپیش‌ثبت‌شده را در مقایسه با Exp08/epoch15 پاس کرد؛
بنابراین به‌عنوان single-fold candidate معتبر است.

## بسته دو-fold و ممیزی integration

بسته زیر fold0 جدید Exp16 و fold2 جدید Exp15r را بدون تغییر source tree کنار
fold1 تاریخی قرار می‌دهد:

- archive: `submission/iaaa_brain_ct_triage_mls_exp16_fold0_exp15r_fold2_20260902.zip`
- bytes: `562649792`
- SHA256: `3423ed8f46288c05fb1b587c784db0546151de67ed51eeff1b1a3f3f93783a43`
- extracted server root:
  `/workspace/iaaa_artifacts/package_exp16_fold0_exp15r_fold2`

ممیزی CUDA-only روی تمام 70 مطالعه fold0 کامل شد. parity خروجی slice-level
بسته با ممیزی مستقل پاس شد: صفر index mismatch، صفر اختلاف selector و heatmap،
و حداکثر اختلاف MLS برابر `1.1920929e-7 mm` بود.

### نتیجه مهم ensemble

| حالت | MAE | Boundary-F1 | objective |
|---|---:|---:|---:|
| fold0 baseline | 1.665417 | 0.822263 | 2.020892 |
| fold0 Exp16 | **1.604478** | **0.827333** | **1.949813** |
| ensemble baseline (diagnostic) | **0.960407** | 0.871154 | **1.218099** |
| ensemble با Exp16 (diagnostic) | 1.028320 | **0.878418** | 1.271483 |

جایگزینی fold0 در ensemble، Boundary-F1 را `+0.007264` بهتر ولی MAE را
`+0.067913 mm` و objective را `+0.053384` بدتر کرد. این مقایسه ensemble
ارزیابی مستقل نیست، چون مدل‌های fold1/fold2 مطالعه‌های fold0 را در training
دیده‌اند. بااین‌حال هشدار عملی مهمی است: بسته دو-fold فعلاً challenger است و
نباید صرفاً بر پایه بهبود single-fold release قطعی شود.

گزارش‌های aggregate در مسیر زیر نگهداری می‌شوند:

`reports/mls_experiments/mls-vast-exp16-w32-fold0-strict-ensemble-refresh/submission_integration/`

## smoke کامل ZIP استخراج‌شده

study 1164 مستقیماً با `model.py` داخل ZIP استخراج‌شده اجرا شد. ICH، پنج fold
شکستگی و سه مدل MLS همگی با CUDA بارگذاری و inference شدند:

- CUDA: NVIDIA GeForce RTX 3060
- load runtime: `10.680 s`
- inference runtime: `10.449 s`
- peak VRAM: `1.582 GiB`
- MLS package output: `17.351613998413086 mm`
- independent integration reference: `17.351613998413086 mm`
- absolute delta: `0.0`
- schema و تمام خروجی‌ها finite بودند.

## تصمیم و preregistration برای Exp17

Exp17 همان recipe strict موفق Exp15r/Exp16 را به fold1 منتقل می‌کند؛ تنها عامل
آزمایشی تغییر held-out fold از 0 به 1 است. baseline منصفانه Exp09/epoch15 روی
همان profile تولید قفل‌شده از grid ازقبل محاسبه‌شده استخراج شد:

- MAE=`1.258665`
- RMSE=`1.976092`
- bias=`-0.182218`
- Boundary-F1=`0.823729`
- objective=`1.611207`

هر checkpoint جدید فقط وقتی eligible است که هم‌زمان MAE<=1.258665،
Boundary-F1>=0.82 و objective<=1.611207 باشد. online validation و grid مخصوص
fold1 به‌تنهایی اجازه promotion نمی‌دهند.

## وضعیت اجرای Exp17

- run: `mls-vast-exp17-w32-fold1-strict-ensemble-refresh`
- MLflow run ID: `15d921e271f14d808f3a7286168e6b4c`
- Vast instance: `49527185`
- tmux: `mls_exp17_fold1`
- server commit: `eb00f89bbfb5d57d0d1ebe768b7646fd863e481c`
- durable run directory:
  `/workspace/iaaa_artifacts/logs/mls-vast-exp17-w32-fold1-strict-ensemble-refresh`
- started UTC: `2026-09-02T04:16:12.182159+00:00`
- policy: strict CUDA-only، 23 epoch، no warm-start، no CPU model fallback
- early observation: epoch1 active، حدود 2.7 batch/s، VRAM حدود 5.37GB، loss
  finite و MLflow system monitor فعال است.

پس از پایان epoch1، report خود trainer همین MLflow run ID را ثبت کرد، پنج
checkpoint انتخابی به‌علاوه resume checkpoint ساخته شدند و هیچ payload تازه‌ای
در queue آفلاین MLflow دیده نشد؛ بنابراین ثبت remote فعال است.

### پایش اولیه Exp17

تا پایان epoch3، run بدون NaN/OOM، با همان PID و VRAM پایدار ادامه داشت. بخش
مکان‌یابی به‌سرعت یاد گرفت: keypoint MAE از `314.00px` در epoch1 به `19.16px`
در epoch2 و `15.21px` در epoch3 رسید؛ slice MLS MAE نیز از `109.57mm` به
`9.99mm` و سپس `4.14mm` کاهش یافت. در مقابل selector هنوز عقب است: AUC در
epochهای 1/2/3 به‌ترتیب `0.442/0.232/0.313` بود و در epoch3 میانگین probability
مثبت اندکی کمتر از منفی ماند. strictهای موفق قبلی معمولاً بین epoch3 و epoch5
جهش selector داشتند، پس این وضعیت یک هشدار پایش است نه مجوز توقف. گیت بعدی
برای قضاوت trajectory پس از epoch5 تعیین شد؛ recipe در حال اجرا تغییر نکرد.

### گیت epoch5 Exp17

هشدار اولیه در epoch4/5 به recovery واقعی تبدیل شد، نه صرفاً کاهش loss. در
epoch4، selector AUC به `0.68242` برگشت و در epoch5 به `0.82035` رسید؛ در
epoch5 میانگین score مثبت `0.45604` و منفی `0.44958` بود، پس جهت رتبه‌بندی نیز
صحیح شد. این AUC تقریباً هم‌سطح Exp09 fold1 در epoch5 (`0.8288`)، بهتر از
Exp16 fold0 (`0.7864`) و تنها کمی پایین‌تر از Exp15r fold2 (`0.8247`) است.
هم‌زمان keypoint MAE به `12.13px` و slice MLS MAE به `2.406mm` رسید. خروجی
study-level هنوز fallback بود (`3.58396mm` و Boundary-F1 صفر)، چون selector-F1
در gate ثابت 0.5 هنوز فعال نشده است؛ تاریخچه runهای موفق نشان می‌دهد این بخش
معمولاً حوالی epoch8 فعال می‌شود. بنابراین گیت epoch5 پاس شد و تصمیم ثبت‌شده
ادامه بدون تغییر recipe تا دست‌کم epoch8 است.

در epoch7، selector AUC/F1 به `0.87560/0.53165`، keypoint MAE به `9.34px` و
slice MAE به `2.010mm` رسید. اولین activation واقعی study aggregation در
epoch8 رخ داد: selector AUC/F1=`0.87551/0.74711`، slice MAE=`2.091mm`، اما
study MAE=`3.15577mm`، Boundary-F1=`0.18071` و objective=`4.85660` بود. پس
run از fallback خارج شده ولی خروجی study-level هنوز immature است. Exp09 fold1
در epoch8 به study MAE=`1.70147` و Boundary-F1=`0.66349` رسیده بود، بنابراین
Exp17 در همین نقطه عقب‌تر است؛ بااین‌حال Exp16 نیز در epoch8 وضع مشابهی داشت
(`3.54553mm`/`0.15152`) و در epoch10 به `1.24230mm`/`0.84167` جهش کرد. نتیجه:
نه promotion و نه توقف در epoch8؛ ادامه تا گیت بعدی epoch10/13 بدون تغییر
recipe، سپس قضاوت بر اساس study-level trajectory.

گیت epoch10 جهش aggregation را تأیید کرد. epoch9 به study MAE=`2.00397`،
Boundary-F1=`0.53571` و objective=`3.00900` رسید؛ در epoch10 این اعداد به
`1.28215mm`، `0.78524` و `1.76959` بهبود یافتند. selector AUC/F1 در epoch10
`0.88415/0.73367`، keypoint MAE=`8.24px` و slice MAE=`1.943mm` بود. با وجود
بهبود بزرگ، gate promotion تاریخی Exp09 هنوز پاس نشده است: MAE باید حداکثر
`1.258665`، Boundary-F1 حداقل `0.82` و objective حداکثر `1.611207` باشد.
فاصله فعلی به‌ترتیب `+0.02349mm`، `-0.03476` و `+0.15838` است. تصمیم:
ادامه تا snapshot ازپیش‌ثبت‌شده epoch13؛ هیچ تغییر میان-run یا promotion زودرس.

epoch12 قوی‌ترین candidate آنلاین تا این نقطه شد: study MAE=`0.98500mm`،
Boundary-F1=`0.84034` و objective=`1.34408`، با selector AUC=`0.92049`.
این اعداد هر سه threshold تاریخی را پاس می‌کنند و نسبت به baseline Exp09
به‌ترتیب `0.27366mm` MAE کمتر، `0.01661` Boundary-F1 بیشتر و `0.26713`
objective کمتر دارند. بااین‌حال این فقط غربال online است، نه promotion قطعی:
trainer در این مرحله aggregation داخلی `relative_component` را گزارش می‌کند؛
promotion فقط پس از audit همه 67 مطالعه با profile قفل‌شده `severity_window`
مجاز است. checkpointهای `best_selector_auc`، `best_study_boundary`، `best_study`
و `best` همگی با timestamp epoch12 روی سرور حفظ شدند. epoch13 افت کرد
(`1.92041mm`/`0.55194`/`2.86923`) و فقط checkpoint slice-level `best_mae` را
به‌روزرسانی کرد؛ بنابراین انتخاب last epoch مردود است و run تا 23 epoch ادامه
می‌یابد تا candidateهای بعدی نیز بدون hindsight ناقص جمع شوند.

epoch14 دوباره غربال online را پاس کرد: MAE=`1.04634mm`،
Boundary-F1=`0.82353` و objective=`1.44254`. epoch15 افت کرد و به
`1.27217mm`، `0.77083` و `1.77419` رسید. این نوسان تأیید می‌کند که epoch آخر
یا یک metric منفرد مبنای قابل‌اعتماد انتخاب نیست. از بین named checkpointهای
فعلی، epoch12 همچنان بهترین candidate آنلاین و محفوظ است؛ snapshot epoch15 نیز
مطابق preregistration ذخیره شد و آموزش برای جمع‌آوری epochهای 17/19/21/23 ادامه
دارد.

در epoch16 خروجی online به MAE=`1.19674mm`، Boundary-F1=`0.79474` و
objective=`1.66149` رسید؛ MAE gate را پاس کرد، اما Boundary-F1 و objective به
ترتیب حدود `0.02526` پایین‌تر و `0.05028` بالاتر از حد لازم بودند. epoch17
ضعیف‌تر شد (`1.79220mm`/`0.61987`/`2.60485`). بنابراین snapshot epoch17 محفوظ
است ولی candidate اصلی تغییر نکرد و epoch12 همچنان بهترین named state است.
آموزش بدون NaN/OOM/CPU fallback و با VRAM پایدار تا snapshot epoch19 ادامه
یافت.

epoch18 به MAE=`1.33481mm`، Boundary-F1=`0.75758` و objective=`1.86669`
رسید؛ epoch19 نیز `1.45937mm`، `0.71848` و `2.07047` بود. هیچ‌کدام غربال
promotion را پاس نکردند و named candidate epoch12 را جابه‌جا نکردند. در این
نقطه LR به `8.69e-6` رسیده و snapshot epoch19 ذخیره شده است؛ ادامه تا epoch21
و 23 صرفاً برای تکمیل plan و audit بدون bias انجام می‌شود، نه به‌خاطر فرض
خودکار بهترشدن epochهای دیرتر.

epoch20 با `1.41779mm`/`0.70058`/`2.06921` و epoch21 با
`1.29008mm`/`0.78524`/`1.76858` نیز candidate epoch12 را جابه‌جا نکردند.
LR در پایان epoch21 به `2.22e-6` رسیده و snapshot epoch21 محفوظ است. دو epoch
پایانی برای تکمیل preregistration اجرا شدند؛ پس از آن باید terminal success،
23/23 metric، MLflow remote، checkpoint integrity و audit CUDA همگی جداگانه
تأیید شوند.

قرارداد dataset بلافاصله پیش از launch با validator رسمی پاس شد: 3484 ردیف،
338 مطالعه، 1781 مثبت، 1703 منفی، 3484 مسیر resolve، spacing و study truth کامل.

## وضعیت Git و بازیابی

سه فایل Exp17 ابتدا در local commit مشترک `791900e` قرار گرفتند، چون یک فرایند
هم‌زمان ICH همان index را commit کرد. push آن commit مشترک به‌دلیل محافظ egress
انجام نشد. برای جلوگیری از انتقال ناخواسته تغییرات دیگران، فقط سه فایل MLS با
SCP منتقل و در commit مستقل سرور `eb00f89` ثبت شدند. این commit هنوز به GitHub
push نشده است. هنگام sync بعدی باید فقط تغییرات MLS آن با تاریخچه جدید شاخه
ادغام شوند و تغییرات هم‌زمان دیگران حفظ شوند.

## کار بعد از پایان training

1. وضعیت MLflow باید `FINISHED` و همه 23 epoch باید finite باشند.
2. named checkpoints و epochs 13/15/17/19/21/23 روی تمام 67 مطالعه fold1،
   CUDA-only ممیزی شوند.
3. profile تولید قفل‌شده معیار اصلی promotion باشد؛ grid fold1 فقط diagnostic.
4. checkpoint واجد هر سه گیت، همراه گزارش و هش به سیستم محلی منتقل شود.
5. سپس ensemble سه‌عضوی strict به‌طور OOF/cross-fold بازسازی شود؛ نتیجه diagnostic
   fold0 فعلی به‌تنهایی مبنای release نیست.
6. سرور Vast بدون هماهنگی کاربر stop یا destroy نشود.

## نتیجه نهایی Exp17 و تصمیم Exp18

Exp17 با exit code صفر، MLflow run ID
`15d921e271f14d808f3a7286168e6b4c` و 23/23 ایپاک پایان یافت. ممیزی strict
تمام 11 checkpoint روی 67 مطالعه، یعنی `737/737` inference، بدون failure و
بدون fallback مدل به CPU کامل شد. هش خروجی‌های audit، pooling grid و promotion
gate پس از SCP میان Vast و سیستم محلی دقیقاً یکسان بود.

بااین‌حال بهترین candidate Exp17 روی profile تولید قفل‌شده فقط به
MAE=`1.399901`، Boundary-F1=`0.793443` و objective=`1.813016` رسید و هر سه
گیت Exp09 را رد کرد. grid همان-fold بهترین تشخیصی `1.212315/0.818397` را نشان
داد، اما حق promotion ندارد. در آزمون سخت‌گیرانه‌تر، profile/snapshot انتخاب‌شده
فقط از fold0 و fold2، روی Exp09 fold1 به MAE=`1.484237` و روی Exp17 به
`1.817143` رسید؛ پس افت صرفاً ناشی از profile قفل‌شده نامناسب نبود.

blendهای 10/25/50/75 درصد Exp17 با Exp09 نیز profile تولید را بهتر نکردند. حتی
محافظه‌کارانه‌ترین حالت 90% Exp09 + 10% Exp17، MAE را از `1.258665` به
`1.264027` و Boundary-F1 را از `0.823729` به `0.805556` بدتر کرد. بنابراین
Exp17 نه جایگزین و نه مکمل مناسب Exp09 است.

Exp18 شاخه sampler یا seed sweep را باز نمی‌کند. تغییر مفهومی آن شکستن selector
یک‌خروجی به دو head است: presence باینری برای gate/count و relative-severity
برای ranking/weighting. مجموع دو BCE نرمال می‌شود تا scale loss selector ثابت
بماند و همه اجزای دیگر recipe فریز هستند. plan در
`reports/mls_experiments/mls-vast-exp18-w32-fold1-dual-selector-transfer/PREREGISTERED_PLAN.md`
ثبت شده است. معیار promotion همچنان هر سه شرط Exp09 تحت full-study audit
CUDA-only است؛ online validation به‌تنهایی مجوز انتخاب نمی‌دهد.
