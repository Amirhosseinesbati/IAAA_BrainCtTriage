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
- Vast instance: `49527185`
- tmux: `mls_exp17_fold1`
- server commit: `eb00f89bbfb5d57d0d1ebe768b7646fd863e481c`
- durable run directory:
  `/workspace/iaaa_artifacts/logs/mls-vast-exp17-w32-fold1-strict-ensemble-refresh`
- started UTC: `2026-09-02T04:16:12.182159+00:00`
- policy: strict CUDA-only، 23 epoch، no warm-start، no CPU model fallback
- early observation: epoch1 active، حدود 2.7 batch/s، VRAM حدود 5.37GB، loss
  finite و MLflow system monitor فعال است.

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
