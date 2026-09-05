# گزارش آمادگی کمپین G1 برای MLS — ۵ سپتامبر ۲۰۲۶

## حکم فعلی

تا زمان نگارش این سند، **هیچ مدل MLS جدیدی هنوز به‌صورت deploy-aligned و
leak-free ثابت نکرده که از comparator داخلیِ hash-pinned بهتر است**. بنابراین
نباید A9/A10 یا هر run محلی را «مدل برتر مسابقه» نامید.

با این حال، مسیر MLS تمام نشده است. نتیجهٔ منفی A9/A10 نشان می‌دهد که تغییر
loss/refinement روی همان ورودی تک‌اسلایسی، مخصوصاً در مرز ۳ mm، بهبود پایداری
نداده است؛ نه اینکه داده یا مسئله فاقد signal باشد. G1 یک فرضیهٔ معماریِ جدید
و قابل آزمون است: کانتکست سه‌اسلایسی در همان مختصات مرکزی، با windowهای
brain/subdural/bone و بدون تغییر هم‌زمان در loss، pooling، split، seed یا
augmentation.

## نتیجهٔ audit داده و انتقال

اختلاف `7,683` و `7,508` نشانهٔ خرابی انتقال داده نیست:

| واقعیت | مقدار |
|---|---:|
| DICOM خام | 7,683 |
| ردیف metadata خام | 7,508 |
| rowهای `slice_targets.csv` | 7,683 |
| rowهای MLS multitask | 3,484 |
| rowهای landmark مثبت | 1,781 |
| study | 338 |
| clean-negative study | 161 |

در ۲۲ study منفی، metadata برای همهٔ sliceها موجود نیست (۱۷۵ slice
`metadata_missing`)؛ اما `slice_targets.csv` از کل DICOMهای مرتب‌شده ساخته شده
است. بنابراین dataset G1 باید کل حجم DICOM را نگه دارد، نه فقط sliceهای دارای
metadata. همهٔ 3,484 مرکز آموزشی به SOP خام یکتا map می‌شوند.

نتیجهٔ عملی: cache G1 در سطح study و با شکل `[D, 3, 512, 512]` است؛ هر DICOM
خام دقیقاً یک بار window می‌شود. ۹ کانال هنگام خواندن نمونه ساخته می‌شوند:

```text
[z-1 brain, z-1 subdural, z-1 bone,
  z   brain, z   subdural, z   bone,
  z+1 brain, z+1 subdural, z+1 bone]
```

در ابتدا/انتهای حجم edge replication به‌کار می‌رود. cache `float32` است تا با
windowing runtime CUDA یکی باشد؛ PNG تاریخیِ `uint8` برای این آزمایش control
معتبر نیست. برآورد حجم cache کامل حدود 22.5 GiB است.

## چرا تاکنون مدل بهتر اثبات نشده است؟

1. A9/A10 در MAE/شاخص localization نشانه‌ای از سود داشتند، ولی F1 در مرز 3 mm
   افت کرد. چون MLS بین 3 تا 5 mm به‌تنهایی Urgent را تعیین می‌کند، این شکست
   برای هدف مسابقه مهم‌تر از بهبود کوچک MAE است.
2. comparator قبلی از نظر provenance برای ادعای نهایی کامل نبود: منبع package
   submission قدیمی با artifact frozen دقیقاً بازسازی‌پذیر نبود. بنابراین از آن
   فقط به‌عنوان comparator داخلیِ hash-pinned استفاده می‌شود، نه «Champion
   رسمیِ اثبات‌شده».
3. cache ICH فعلی 320px و windowهای متفاوت دارد؛ استفادهٔ مجدد از آن برای MLS
   train/deploy mismatch ایجاد می‌کرد.
4. پیش از G1، کنترل ۳کاناله و challenger ۹کاناله می‌توانستند به‌علت مصرف RNG
   هنگام ساخت conv ۹کاناله headهای متفاوتی داشته باشند. این مسئله اصلاح شده:
   RNG پس از ساخت conv جدید بازگردانده می‌شود و وزن‌های pretrained فقط در
   کانال‌های مرکزی `3:6` قرار می‌گیرند؛ همسایه‌ها از صفر شروع می‌شوند.

## طراحی causal G1

دو arm از ابتدا و با recipe یکسان train می‌شوند:

| arm | ورودی | نقش |
|---|---:|---|
| G1-C0 | 3-channel central float32 cache | control معتبر برای pipeline جدید |
| G1-A | 9-channel `z-1/z/z+1` float32 cache | آزمون اثر context |

تنها اختلاف مجاز در `training_config` پس از حذف `fold` و `seed`:

```text
input_channels: 3  ↔  input_channels: 9
```

recipe قفل‌شده: HRNet-W32، 23 epoch، audit ثابت در epoch 15، seedهای
`42/2026/3407`، strict determinism، selector single، همان pooling و loss
baseline deploy-aligned. C0 در برابر incumbent قدیمی فقط «context» را ایزوله
نمی‌کند (چون float32 cache جای PNG uint8 آمده است)؛ ادعای causal دقیق فقط
G1-C0 در برابر G1-A است.

## guardrailهای اجرا

پیش از CUDA، cache باید این‌ها را pass کند:

- عضویت دقیق همان 338 study در labels، metadata، fold و `slice_targets`.
- depth هر study برابر `dicom_series.NumDicomFiles` و ترتیب کامل همهٔ SOPها
  برابر `slice_targets`.
- قرارداد spacing نیز به DICOM runtime بسته است: metadata واقعی فقط
  `PixelSpacing0/1` دارد و دقیقاً با convention reader به‌ترتیب
  `spacing_y/spacing_x` نگاشت می‌شود؛ builder و validator هر دو این برابری را
  برای تک‌تک studyها fail-closed بررسی می‌کنند.
- shape/dtype/hash هر volume، hash کامل bytes هر DICOM خام و hash manifest.
- cache نیمه‌کاره، cache بدون manifest، یا cache با source/window/fold/builder
  متفاوت قابل reuse نیست.
- receipt مستقلِ validation باید hash خودش را در training config داشته باشد؛
  بنابراین اجرای مستقیم trainer با cache صرفاً manifestدار مجاز نیست. hash
  `config/folds.csv` زنده نیز باید با fold hash ثبت‌شده در cache یکی باشد.
- runtime qualification روی raw DICOM نشان دهد ورودی 3 و 9 کانال cache با
  ورودی deployment دقیقاً `array_equal` هستند؛ forward کوچک فقط با CUDA انجام
  می‌شود.

هر run G1 در MLflow tagهای صریح دارد: campaign، arm، fold، input_channels،
cache-manifest/validation-receipt hash و hashهای source input/model/predict/
train دارد. evaluator فقط foldهای 3/4 و دقیقاً seedهای `42/2026/3407` را
می‌پذیرد، median را پیش از hash در artifact خصوصی persist می‌کند، و gate
سه‌طرفهٔ aggregate → private predictions → evaluator summary را checksum می‌کند.
receipt fold-3 نیز فقط وقتی fold-4 را باز می‌کند که campaign/fold/study/گیت و
hash همان preregistration فعلی را داشته باشد. private predictionها به MLflow
upload نمی‌شوند.

## تصمیم‌گیری مرحله‌ای

| مرحله | data | تصمیم |
|---|---|---|
| screen | fold 3، 66 study / 64 patient | فقط شش run (دو arm × سه seed) |
| confirmation | fold 4، 68 study / 65 patient | فقط اگر screen کامل pass شود |
| promotion study | هر پنج fold، 338 study | فقط پس از تأیید cross-fold |

gate fold 3 نیاز دارد frozen-context Macro-F1 و Urgent-F1 هر دو strictly بهتر
شوند، accuracy افت نکند، Normal/Critical بیش از 0.01 افت نکنند، F1های 3/5mm و
catastrophic error بدتر نشوند، و جهت Macro/Urgent در oracle منفی نباشد. شکست
هر gate یعنی G1-A متوقف می‌شود؛ sweep نجات‌دهنده یا tuning پس از دیدن همان fold
مجاز نیست.

## وضعیت پیاده‌سازی در این commit در حال آماده‌سازی

- `input_contract.py`: قرارداد واحد central/2.5D و edge replication.
- `build_mls_2p5d_cache.py`: cache staging، checksum و SOP/raw integrity.
- `validate_mls_2p5d_cache.py`: receipt fail-closed پیش از train.
- dataset/trainer: loader cache، augment مشترک برای ۹ channel، memmap امن و
  tagهای MLflow.
- qualification، evaluator سه-seed G1، staged triage gate و materializer/
  validator مستقل برای matrix 12-config.
- یک ممیزی خط‌به‌خط قبل از GPU، خطاهای بالقوهٔ واقعی (ستون spacing اشتباه،
  median persistنشده، receipt bypass، fold فایل زنده، و binding ناکافی gate)
  را پیدا و در source اصلاح کرده است؛ این‌ها به‌معنای نتیجهٔ بهتر نیستند، اما
  از اجرای expensive با evidence نامعتبر جلوگیری می‌کنند.

هنوز cache ساخته نشده و هیچ training جدیدی شروع نشده است. ترتیب درست بعد از
freeze/commit/push این است: remote preflight → cache build → cache validation →
materialize/validate matrix → شش train fold-3 → CUDA qualification → audit
three-seed → triage gate.

### رسید انتقال و preflight سرور 3090

در clone ایزولهٔ G1، commit `142752c` از Git bundle کامل و SHA-256
`beeb171a…a4bc7c7` دریافت شد؛ workspaceهای dirty قبلی دست‌نخورده مانده‌اند.
`raw` read-only symlink شده و cache آینده فقط در `Data/processed` همین clone
ساخته خواهد شد. انتقال با سه لایه تأیید شد: DVC cloud در هر دو سمت sync بود؛
کل raw دقیقاً `12,860` file و `2,908,071,754` byte و training دقیقاً `7,683`
file و `2,900,103,674` byte در هر دو سمت داشت؛ و SHA-256 هشت نمونهٔ پراکنده
از annotation/DICOM برابر بود. `training_df.pkl`، labelهای MLS و
`slice_targets.csv` نیز checksum یکسان دارند.

preflight header-only روی سرور pass شد: 338 study، 7,683 DICOM و 3,484 row
بدون pixel decode یا model compute. PyTorch `2.10.0+cu128`، CUDA `12.8` و
RTX 3090 آماده‌اند. venv قدیمی صرفاً پس از برابر بودن `pyproject.toml` و
`uv.lock` reuse می‌شود؛ دوباره‌سازی venv فقط disk/زمان را مصرف می‌کرد و هیچ
تغییر dependency لازم نبود. build واقعی cache به‌صورت service مدیریت‌شده اجرا
می‌شود و status/log مستقل ثبت می‌کند.

### دفتر اجرای واقعی — به‌روزرسانی 2026-09-05

- cache واقعی `mls_2p5d_v1` کامل شد: 338 study، 3,484 row (1,781 positive و
  1,703 negative)، حجم `24,168,671,488` byte؛ manifest SHA-256 برابر
  `c50ece4167b25661a7e36305bfab6f177253981c8418c0fd85acbf23bde4e672` است.
- validator کامل، نه صرفاً header-only، روی raw fingerprints و همهٔ metadataها
  pass شد و receipt در خود ریشهٔ cache نوشته شد. SHA-256 receipt برابر
  `aca8e9890f094e8b3b727852c0ae404deced8270a444c8c9f104dd5133dda6c2` است.
  یک تلاش پیشین پیش از ساخت receipt در مسیر موردانتظار متوقف شد؛ هیچ model
  construction یا نتیجه‌ای تولید نکرد و evidence آزمایشی محسوب نمی‌شود.
- تلاش بعدی در محیط clone فاقد credential، پیش از guard نهایی، MLflow محلی
  ساخته بود و برای حفظ الزام tracking راه‌دور متوقف شد. این run نیز invalid
  است و هیچ checkpoint/metric آن در campaign پذیرفته نمی‌شود. از آن پس wrapper
  فقط چهار متغیر اختصاصی DagsHub را از فایل با permission `0600` می‌خواند و
  `IAAA_REQUIRE_REMOTE_MLFLOW=1` هر fallback به SQLite/file را fail-closed
  می‌کند. probe read-only MLflow راه‌دور pass شده است؛ هیچ مقدار secret در
  log یا repository ذخیره نشده است.
- اجرای معتبر اول `G1-C0 / fold3 / seed42` با Supervisor آغاز شده و هنگام
  آخرین مشاهده در epoch 3 از 23، روی RTX 3090 با حدود 5.5 GiB VRAM و استفادهٔ
  GPU حدود 89--96٪ فعال بود. تا پایان آن هیچ metric یا ادعای بهبود ثبت
  نمی‌شود.
- پیکربندی اجرای زوج `G1-A / context9 / fold3 / seed42` در commit
  `bb94e32ed112740f1cb6b6fb7f0ca37dad442707` آماده است. SHA-256 bundle انتقالی
  `027601762ce94d587f0edd5674c577dd8c50b8d675feb7f8266b0ee5bc02ea1b` در دو
  سمت برابر بود. چون clone فعال دارای artifactهای اجراست، merge آن عمداً
  انجام نشد؛ commit تنها به ref محلی `bundle/g1-pair` fetch شده و manifest
  context9 آن تأیید شده است. این کار از overwrite خروجی C0 جلوگیری می‌کند.

## برآورد امید

احتمال عددی قابل‌اعتماد برای عبور از leaderboard نداریم؛ leaderboard private
است و comparator رسمی کامل بازسازی نشده. بر مبنای signal داده، ادبیات MLS
(localization چنداسلایسی/landmark-based) و شکست A9/A10، مسیر G1 نسبت به
loss-tuning تکراری فرصت واقعی‌تری دارد. برآورد عملیِ مشروط برای گذر از gate
سخت G1 در نخستین screen حدود **15 تا 30 درصد** است؛ این پیش‌بینی نیست و تنها
برای تصمیم منابع است. احتمال رشد در همان خانوادهٔ A9/A10 کمتر از 10 درصد
ارزیابی می‌شود.
