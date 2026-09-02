# تحویل آمادهٔ submission مدل MLS محافظه‌کارانه

تاریخ snapshot: 2026-09-02 (Asia/Tehran)

## تصمیم فعلی

مدل `mls-conservative-five-20260902` بهترین candidate داخلی فعلی MLS و آمادهٔ
یک submission محدود رسمی است. تا دریافت نتیجهٔ واقعی leaderboard، آموزش یا
screen تازه نباید آغاز شود و هیچ ادعای عبور از امتیاز `0.914` مجاز نیست.

## بستهٔ ثابت برای ارسال

- مسیر روی Vast:
  `/workspace/iaaa_artifacts/packages/iaaa_brain_ct_triage_mls_conservative_five_20260902.zip`
- اندازه: `812453997` بایت
- تعداد فایل: `42`
- compression: `ZIP_STORED`
- SHA-256:
  `660770225b53e5389ba0e8dde70cc7e1a65f732ca854887aba6ba8deff1d490b`
- نسخهٔ extractشده روی Vast:
  `/workspace/iaaa_artifacts/package_mls_conservative_five_20260902`
- نسخهٔ پنج-checkpoint محلی:
  `checkpoint/mls/mls-conservative-five-20260902/`

هیچ rebuild یا دستکاری بسته پس از دریافت نتیجه مجاز نیست؛ اگر rebuild لازم شد،
SHA جدید باید به‌عنوان candidate متفاوت ثبت شود.

## ترکیب MLS داخل بسته

1. fold0: 90٪ Exp16 + 10٪ regression از Exp19/epoch21؛
2. fold1: 90٪ Exp09/epoch15 + 10٪ regression از Exp18/epoch21؛
3. fold2: Exp15r/epoch17 بدون blend؛
4. خروجی نهایی MLS: median سه عضو fold.

Exp18 و Exp19 فقط component هستند و standalone release نیستند. Exp20 به علت
شکست gate حذف شده است.

## شواهد آمادگی

- ممیزی کامل 204 مطالعهٔ OOF روی RTX3060 و CUDA-only کامل شد؛
- MAE: `1.461521959 mm`؛
- Boundary-F1: `0.855888430`؛
- objective: `1.749745100`؛
- هر هفت gate مربوط به index، slice MLS، selector، peak-selector، heatmap،
  member aggregation و metric parity پاس شدند؛
- حداکثر residual عضو: `0mm`؛
- audit MLS برای 204 مطالعه `552.441s` زمان و `0.927GiB` peak VRAM مصرف کرد؛
- smoke کامل `model.py` روی یک study شامل پنج مدل MLS و پنج fold شکستگی با
  load=`12.496s`، inference=`11.813s` و peak VRAM=`1.820GiB` کامل شد؛
- schema هشت‌ستونی خروجی finite و معتبر بود؛
- هیچ model forward روی CPU و هیچ network dependency در runtime وجود ندارد؛
- هر پنج checkpoint محلی از نظر اندازه و SHA-256 با manifest بسته match هستند.

## تطابق با راهنمای رسمی

راهنما submission را به‌صورت مدل تعریف می‌کند، نه CSV مستقیم. بسته باید
`model.py` و پوشهٔ `models/` با تمام وزن‌ها را داشته باشد، در Python 3.12 و
CUDA 12.8 بدون نصب package هنگام ارزیابی اجرا شود، روی یک GPU با 24GB VRAM
کار کند و کل inference را در سقف تقریبی 30 دقیقه—ترجیحاً حداکثر 15 دقیقه—تمام
کند. Ensembling مجاز است.

ساختار بسته و مصرف VRAM این شرایط را پاس می‌کنند. smoke کامل تک-study و audit
204-study MLS نیز حاشیهٔ زمانی مناسبی نشان می‌دهند؛ بااین‌حال فقط اجرای رسمی
hidden set شاهد قطعی محدودیت زمان کل سه task است.

## مانع واقعی ارسال

نه PDF رسمی و نه فایل‌های repository هیچ URL، API، CLI، credential یا دستور
upload برای leaderboard ارائه نمی‌کنند. بنابراین ارسال خارجی بدون یکی از موارد
زیر قابل انجام نیست:

1. لینک صفحهٔ submission و دسترسی session واردشده؛ یا
2. CLI/API رسمی و credential مربوط؛ یا
3. انجام upload دستی توسط کاربر.

اگر portal فقط file picker محلی داشته باشد، ZIP کامل باید از Vast به ویندوز
منتقل شود. انتقال مستقیم قبلی به علت VPN کند بود؛ وزن‌های MLS محلی شده‌اند، اما
ZIP کامل عمداً دوباره منتقل نشده تا پهنای باند و فضای تکراری هدر نرود.

## پروتکل بعد از نتیجه

- امتیاز، زمان اجرا، خطاهای runtime و هر feedback رسمی بدون تغییر candidate ثبت شود؛
- اگر submission fail شد، ابتدا علت packaging/runtime رفع شود و همان مدل با SHA
  جدید یا ثابت—بسته به نوع اصلاح—ممیزی شود؛
- اگر score رقابتی بود، این candidate به champion ارتقا یابد؛
- اگر score ضعیف بود، فقط بر اساس خطای end-to-end سه task تصمیم به ادامهٔ MLS
  یا تمرکز روی ICH/fracture گرفته شود؛
- instance Vast تا هماهنگی کاربر stop یا destroy نشود.
