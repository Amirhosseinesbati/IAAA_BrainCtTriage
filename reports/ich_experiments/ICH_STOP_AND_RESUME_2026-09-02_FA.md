# نقطهٔ توقف و راهنمای ادامهٔ پژوهش ICH

تاریخ ثبت: ۲۰۲۶-۰۹-۰۲  
وضعیت رسمی: **متوقف به درخواست کاربر؛ قابل ادامه؛ بدون اجرای فعال**

این سند snapshot واحدِ مسیر پژوهش خونریزی داخل‌جمجمه‌ای در چالش IAAA Brain CT
Triage 2026 است. هدف آن این است که پس از destroy شدن اینستنس Vast نیز بتوان کار
را بدون تکرار ممیزی داده، بازسازی تصمیم‌های قدیمی یا حدس‌زدن provenance ادامه داد.

## ۱. وضعیت دقیق هنگام توقف

- اینستنس Vast: `49378919`، RTX 3090، مسیر پروژه `/workspace/project`.
- بررسی نهایی `pgrep` هیچ پردازش `train_ich`، `diagnose_ich` یا `crossfit_ich`
  نشان نداد؛ GPU پیش‌تر نیز بیکار و با مصرف حدود 17 MiB ثبت شد.
- هیچ اجرای تازه‌ای پس از فرمان توقف آغاز نشد. **Exp89r2 فقط پیش‌ثبت شده و هرگز
  اجرا نشده است.**
- آخرین تست سرور: `227 passed in 14.05s` برای `tests/test_ich*.py`.
- هیچ submission رسمی به leaderboard ارسال نشده است؛ امتیاز حدود `0.914` صرفاً
  مرجع نفر اول گزارش‌شده توسط کاربر است.
- Goal حذف، complete یا blocked نشده است. هدف بلندمدت محفوظ مانده، ولی تا فرمان
  صریح کاربر هیچ اجرای محاسباتی یا جست‌وجوی مدل جدید نباید آغاز شود.
- destroy اینستنس در اختیار کاربر است؛ از این گفتگو هیچ stop/destroyای انجام نشد.

## ۲. آنچه پیش از destroy روی سیستم محلی نجات داده شد

1. وزن و متادیتای کامل Exp61 در:
   `checkpoint/ich/smp/2p5d/segmentation-models-pytorch-efficientnetv2-s-exp61-calibration-candidate-20260902/`
   وزن 98,936,107 بایت و SHA256 آن
   `5f304018340e88e1d858d0842e432d8a163c96ed26117796c7068b6d399c18d4` است.
2. checkpoint، history و summaryهای موجود Exp89 و Exp89r1 در پوشه‌های خودشان
   زیر `reports/ich_experiments/2p5d_segmentation/` کپی شدند.
3. مدل پذیرفته‌شدهٔ پنج‌fold 2.5D پیش‌تر در
   `checkpoint/ich/smp/2p5d/unetplusplus-efficientnet-b2-hardpixel-fprselect-oof-candidate-20260901/`
   موجود بود.
4. مدل ترکیبی تاریخی gate+SegResNet در
   `checkpoint/ich/composite/2p5d-gated-segresnet-exp01-20260831/` موجود بود.
5. checkpoint تشخیصی Exp88 و تمام aggregateهای Exp55 تا Exp89 در گزارش‌های محلی
   حفظ شده‌اند. prediction ردیفی پزشکی عمداً برای شاخه‌های تشخیصی جدید ذخیره نشده
   بود؛ aggregateها و هش‌ها برای بازتولید کافی‌اند.

## ۳. پایهٔ داده و یافته‌های تثبیت‌شده

| موضوع | نتیجهٔ معتبر |
|---|---|
| کل داده | 338 study از 320 بیمار |
| supervision جزئی | 198 study دارای JSON جزئی؛ 7,653 slice برچسب‌دار |
| clean negative | 140 study با `triage_class=0` و صفر بودن هر پنج زیرنوع |
| slice واقعاً بدون JSON | 30 slice پس از تطبیق دقیق SOP |
| foreground | 7,171,440 voxel؛ صفر foreground در ناحیهٔ unknown |
| raw local/Vast | 12,860 فایل، 2,908,071,754 بایت؛ تطبیق تأیید شده |
| SHA256 raw manifest | `b7ba4b0d332d27738d21fae774dd776e8241182af6f8386ea4d4d7218b482271` |
| SHA256 ICH-v2 manifest قدیمی | `e1d6294de29166ca011bf67e8b3c847af9106e5e59f51681dc6408f97ee87c6f` |
| SHA256 schema4 manifest | `0455e6d24590a652b324c58730d81750d675b8c8d2442e67c12f11c16531ec37` |

قواعدی که نباید شکسته شوند:

- slice بدون JSON در study مثبت، background نیست؛ unknown است و از loss حذف می‌شود.
- `AnyICH` برای تعریف clean-negative کافی نیست، چون IVH-only را جا می‌اندازد.
- split باید patient-level باشد؛ در آزمایش‌های OOF ثبت‌شده overlap بیمار صفر بود.
- threshold یا checkpoint انتخاب‌شده روی یک fold نباید همان fold را به‌عنوان outer
  بی‌طرف جا بزند.
- حجم باید در فضای فیزیکی از determinant affine محاسبه شود؛ inverse resize سادهٔ
  labelmap از نظر هندسی رد شده است.
- خروجی ICH مستقل از MLS و شکستگی توسعه یافته است. برای امتیاز نهایی لازم است هر سه
  جزء در یک ارزیابی end-to-end جداگانه ادغام شوند.

## ۴. مسیر مدل‌ها و بهترین دستاوردهای معتبر

### ۴.۱ baseline سه‌بعدی تاریخی

MONAI SegResNet سه‌بعدی با `4,700,982` پارامتر، SHA256
`5859c3dd6ebe4101a8169b0f5b940c23b48eba298110dbf9a731d01271839788`.
روی fold0 پس از فیلتر component برابر 0.1mL: Macro-F1=`0.7177`، FPR=`0.4595`،
presence F1=`0.7500` و MAE=`11.538mL`. این baseline نشان داد معماری سه‌بعدی
به‌تنهایی جواب برتر فعلی نیست و ضعف اصلی، normal false positive و SAH/SDH بود.

### ۴.۲ مدل ترکیبی تاریخی 2.5D + SegResNet

گیت 2.5D روی fold0 AUC=`0.9320` و presence F1=`0.8571` داشت. ترکیب تاریخی با
SegResNet به oracle-context Macro-F1 حدود `0.84975` رسید. این عدد final score نیست
و به‌علت oracle context/یک fold نباید با leaderboard مقایسه شود؛ فقط مرجع تاریخی است.

### ۴.۳ مدل پذیرفته‌شدهٔ OOF پنج‌fold

Unet++/EfficientNet-B2 با hard-empty pixel loss و checkpoint selection ریسک‌آگاه
در پنج outer مستقل، روی تمام 338 study ارزیابی شد:

| معیار | pixel baseline | hard-pixel candidate | تغییر |
|---|---:|---:|---:|
| selection | 0.63213 | **0.64402** | +0.01188 |
| mean Dice | 0.42233 | **0.43506** | +0.01273 |
| Any-ICH AUC | 0.92400 | **0.93453** | +0.01053 |
| normal FPR | 0.42778 | **0.17222** | -0.25556 |
| presence F1 | 0.79177 | **0.86826** | +0.07649 |
| volume MAE | 8.772 | **8.719** | -0.053 |

کاهش FPR و رشد F1 با bootstrap قوی بود، ولی CI رشد Dice و selection صفر را قطع
نکرد. بنابراین این مدل بهترین candidate مستقل و deployable فعلی است، نه اثبات رتبهٔ
اول leaderboard.

### ۴.۴ Exp61؛ incumbent پژوهشی EfficientNetV2-S

Exp61 یک calibration-only screen روی fold1 با outer2 دست‌نخورده است:

- checkpoint epoch 9؛ score=`0.586668`؛ selection=`0.666162`؛ Dice=`0.459106`؛
- Any AUC=`0.923387`؛ macro subtype AUC=`0.910920`؛
- FPR=`0.194444`؛ F1=`0.882353`؛ MAE=`10.7627mL`؛ bias=`-6.2364mL`؛
- Dice زیرنوع‌ها: IVH=`0.648020`، IPH=`0.674845`، SDH=`0.381665`،
  EDH=`0.537976`، SAH=`0.053024`.

این مدل برخی معیارهای calibration را بهتر کرد، اما شکاف حجم بسته نشد و outer/OOF
ندارد؛ بنابراین فقط incumbent تحقیقاتی است. انتقال محلی و هش وزن تأیید شده است.

## ۵. خلاصهٔ همهٔ شاخه‌های آزمایشی و تصمیم‌ها

جزئیات عددی exp00 تا exp75 در `ICH_V2_RESEARCH_LOG.md` ثبت شده است. نتیجهٔ تصمیمی
تمام مسیر در جدول زیر یکجا آمده است:

| بازه | سؤال/روش | نتیجه و تصمیم |
|---|---|---|
| exp00–03 | SegResNet، partial supervision و distillation | پیاده‌سازی و هندسه اصلاح شد؛ raw 3D به سقف لازم نرسید؛ ranking آن فقط برای gate تاریخی حفظ شد. |
| 2.5D exp01 و OOF اولیه | سه slice × سه window | از 3D بهتر و پایهٔ اصلی پژوهش شد. |
| exp15–22 | hard-empty pixel و FPR-aware selection | کاهش بزرگ FPR و رشد F1 در OOF پنج‌fold؛ مدل پذیرفته‌شدهٔ فعلی. |
| exp23–47 | study balancing، IVH center، TTA، adapters فضایی/پنج‌برشی | چند سیگنال موضعی یافت شد، اما تکرار مستقل یا گیت فضایی/حجمی را پاس نکرد؛ promotion نشد. |
| exp50–52 | pooling ترتیبی، logistic meta-head و head-only rebalance | سود subtype کوچک یا overfit؛ شاخهٔ post-hoc pooling بسته شد. |
| exp53 | frozen encoder + temporal BiGRU | calibration: Any `+0.009857`، macro `+0.016695`، proxy `+0.005461`؛ outer2: Any `-0.003136` با وجود macro `+0.029078`؛ به‌علت safety رد شد و ایدهٔ Any-invariant شکل گرفت. |
| exp54–58 | temporal volume و threshold cross-fit | exp55 FPR را بدتر کرد؛ exp56 FPR را ثابت نگه داشت ولی bias/CI پاس نشد؛ exp58 جهت‌دار اما بین fold ناپایدار بود. |
| exp59–67 | EfficientNetV2-S، volume losses و اصلاح SAH | Exp61 بهترین incumbent تک-fold شد؛ volume/SAH gains یا SDH/FPR را خراب کردند یا کوچک‌تر از gate بودند؛ هیچ‌کدام OOF-promote نشدند. |
| exp68–71 | conditional relabel، trust region و error gate | ظرفیت اصلاح train وجود داشت، ولی rare-subtype collapse یا precision ناکافی روی held-out؛ frozen-feature router بسته شد. |
| exp72–75 | hierarchical/common-mode loss | background coupling بهتر شد، اما IPH dominance و ضعف rare/diffuse subtype ماند؛ loss-only branch بسته شد. |
| exp76 | factorized foreground/subtype architecture | identity و جداسازی گرادیان پاس؛ 90 تست؛ فقط smoke مجاز شد. |
| exp77 | نخستین smoke factorized | BF16 near-tie باعث شکست identity metric شد؛ gate پایین آورده نشد. |
| exp78 | BF16-exact composition | exact logits/probabilities/argmax پاس؛ 91 تست؛ smoke مجاز شد. |
| exp79–80 | factorized screen | calibration سریعاً به سمت IVH/IPH رفت و SDH/SAH/حجم آسیب دید؛ promotion رد شد. |
| exp81–82 | causal attribution | train-only شکست را بازتولید نکرد؛ calibration attribution نشان داد shared decoder/foreground pressure علت اصلی است. |
| exp83–84 | فقط residual heads، 870 پارامتر | short-horizon ایمن، اما سه epoch: Dice `-0.008219`، score `-0.009086`، MAE `+0.598mL`؛ SAH `+0.009428` به بهای SDH؛ رد شد. |
| exp85 | تفکیک foreground/subtype scope | foreground branch منبع اصلی آسیب؛ subtype-only اندکی مفید ولی SDH را کم کرد و زیر gate ماند؛ factorized recipe بسته شد. |
| exp86 | independent SAH expert | AP/AUC broad=`0.003873/0.51112` در برابر incumbent=`0.031781/0.61461`؛ بدون checkpoint و رد شد. |
| exp87 | scope attribution داخلی SAH expert | head-only بهترین scope بود ولی AP=`0.111027` از incumbent=`0.122135` کمتر؛ کل شاخه بسته شد. |
| exp88 | Any-invariant temporal subtype residual، calibration | Any دقیقاً ثابت؛ macro `+0.014186` و proxy `+0.002128`؛ چهار subtype بهتر و IPH `-0.009434`؛ best epoch 13. diagnostic checkpoint حفظ شد، ولی برای پذیرش نیازمند OOF بود. |
| exp89 | OOF نخست | evaluator روی fold بدون positive subtype مقدار null را درست مدیریت نمی‌کرد؛ اجرا متوقف و کد اصلاح شد. |
| exp89r1 | OOF دوم | fold0 outer macro `-0.003805` و SAH `-0.046875`؛ fold1 identity؛ fold2 calibration `+0.015359` ولی stale cache metadata مانع outer شد. اجرای ناقص، غیرقابل‌promotion و متوقف. |
| exp89r2 | fresh-cache OOF | فقط طرح پیش‌ثبت‌شده؛ **هیچ‌گاه اجرا نشد**. |

## ۶. آخرین فرضیهٔ باز و وضعیت Exp89

Exp88 نشان داد residual ترتیبی کم‌پارامتر می‌تواند logits زیرنوع‌ها را تغییر دهد و
logit `Any` را دقیقاً invariant نگه دارد. این نخستین سیگنال معماری بعد از بسته‌شدن
شاخه‌های spatial/factorized بود که calibration gate را پاس کرد.

اما Exp89r1 هنوز شواهد تعمیم مثبت نمی‌دهد:

- outer fold0: macro subtype `-0.003805`، SAH `-0.046875`، SDH `+0.040073`؛
- outer fold1: همهٔ deltaها صفر و best epoch صفر؛
- fold2: calibration macro `+0.015359`، ولی outer به‌دلیل cache metadata قدیمی
  اجرا نشد؛
- overlap بیمار train/calibration/outer در foldهای کامل‌شده صفر بود؛
- checkpointهای کوچک و summaryها اکنون محلی‌اند.

بنابراین Exp88/89 هنوز «مدل بهتر» محسوب نمی‌شود. اگر کار ادامه یابد، تنها ادامهٔ
علمی مجاز، rerun کامل با cache تازه و همان gateهای ازپیش‌ثبت‌شده است؛ نه تفسیر
خوش‌بینانهٔ foldهای ناقص و نه پایین‌آوردن gate.

## ۷. provenance کد و وضعیت Git

- branch محلی: `codex/competition-winning-pipeline`.
- آخرین commitهای تثبیت‌شدهٔ ICH:
  - `0e63d32` — freeze factorized representation after Exp82؛
  - `f30f25c` — residual head trade-off diagnostics؛
  - `d70e59f` — Any-invariant temporal subtype head.
- server HEAD در پایان چرخه قدیمی‌تر (`add56b3`) بود و فایل‌های جدید با SCP sync
  شده بودند؛ اتکا به checkout باقی‌مانده روی سرور پس از destroy ممنوع است.
- workspace محلی dirty و مشترک با کارهای MLS/fracture است. در ادامه فقط فایل‌های
  ICH خود این مسیر را stage/commit کنید و تغییرات نامرتبط را دست نزنید.
- هیچ push تازه‌ای در نقطهٔ توقف انجام نشد. پیش از سرور جدید، local branch و remote
  باید مقایسه و فقط با مجوز کاربر push/pull شوند.

## ۸. دستور کار دقیق برای ادامه در آینده

1. Goal موجود را با تأیید کاربر از همین نقطه ادامه دهید؛ Goal تازه و موازی نسازید.
2. یک Vast جدید با RTX 3090 یا بهتر، CPU/اینترنت مناسب و فضای کافی اجاره و عملکرد
   واقعی آن benchmark شود.
3. repo را از Git restore کنید، سپس داده را با DVC منتقل و raw local/server را با
   تعداد، حجم و manifest hash دوباره تطبیق دهید.
4. checkpointهای پذیرفته‌شدهٔ پنج‌fold و در صورت نیاز Exp61 را از پوشهٔ محلی بالا
   به سرور منتقل کنید. فایل `.env` فقط با اجازهٔ صریح، SSH و permission `600` منتقل
   شود؛ token در log/commit نیاید.
5. تست‌های `tests/test_ich*.py` باید پیش از اجرا دوباره سبز باشند (baseline این
   snapshot: 227 تست).
6. Exp89r2 را دقیقاً از `PREREGISTERED_PLAN.md` با cache تازه، برای نمونه
   `/workspace/cache/ich_temporal_exp89r2_fresh` اجرا کنید. cache قدیمی Exp53/89 را
   reuse نکنید مگر validator کامل checkpoint/manifest/context را پاس کند.
7. OOF را برای همهٔ foldها کامل کنید؛ Any باید دقیقاً invariant بماند و promotion
   فقط براساس aggregate OOF و subtype safety ازپیش‌ثبت‌شده باشد.
8. اگر OOF رد شد، recipe temporal residual فعلی بسته شود و منابع صرف sweep همان
   hyperparameterها نشود. اگر پاس شد، checkpointها به ساختار محلی منتقل و سپس در
   pipeline نهایی سه‌تسکی و submission واقعی ارزیابی شوند.
9. تنها امتیاز leaderboard می‌تواند ادعای نزدیک‌شدن/عبور از `0.914` را معتبر کند.

## ۹. منابع مرجع داخل پروژه

- گزارش عمیق EDA: `reports/eda/deep/`
- ممیزی معماری فعلی: `reports/current_pipeline_forensic_audit.md`
- راهبرد مستقل و ادبیات: `reports/literature_and_independent_strategy.md`
- راهبرد کل مسابقه: `reports/master_competition_strategy.md`
- دفترچهٔ عددی کامل ICH: `reports/ich_experiments/ICH_V2_RESEARCH_LOG.md`
- طرح زیرساخت agentic/Vast: `reports/ich_agentic_vast_research_platform.md`
- طرح ادامهٔ Exp89r2:
  `reports/ich_experiments/2p5d_segmentation/exp89r2_subtype_temporal_residual_oof_v1/PREREGISTERED_PLAN.md`

## ۱۰. جمع‌بندی صادقانه

بزرگ‌ترین دستاورد قطعی، کاهش FPR از `0.42778` به `0.17222` و افزایش presence F1
از `0.79177` به `0.86826` در OOF کامل است. Exp61 و Exp88 فرضیه‌های امیدوارکننده
برای backbone و context ساخته‌اند، اما هنوز مدل OOF-validated بهتر از candidate
پنج‌fold نداریم. مسیرهای loss-only، generic adapter، factorized residual و independent
SAH expert با شواهد کافی بسته شده‌اند. نزدیک‌بودن به سقف واقعی یا رتبهٔ اول بدون
submission رسمی قابل اثبات نیست؛ ادامه فقط وقتی می‌ارزد که OOF تازه، ارزیابی
end-to-end سه‌تسکی و leaderboard feedback محور تصمیم باشند.

