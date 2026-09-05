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
- اجرای معتبر اول `G1-C0 / fold3 / seed42` با Supervisor کامل شد: همهٔ 23
  epoch ثبت شدند و report وضعیت `completed` دارد. run راه‌دور MLflow با شناسهٔ
  `dcc9034832354bd4909708b8f7362bf7` مستقل بازخوانی شد: status=`FINISHED`،
  52 metric، policy=`cuda_only_no_cpu_fallback`، campaign/arm/fold و هر دو
  hash cache/receipt دقیقاً با contract برابرند. checkpointهای epoch 15 و
  selector/MAE/study/boundary به‌صورت محلی روی سرور باقی مانده‌اند. این فقط
  control زوج G1 است، نه مدل release یا ادعای بهبود.
- پس از completion C0، split validation کد تاریخی و fold پین‌شدهٔ cache برای
  fold3 با محاسبهٔ مستقیم برابر شد: هر دو 66 study و symmetric difference صفر.
  بنابراین سخت‌گیری provenance جدید، نمونه‌های train/validation مقایسهٔ C0/A
  را تغییر نمی‌دهد.
- commit `a2aa37e5b63cd95448e3490624ca1fe945d231a0` سخت‌گیری provenance را
  اضافه می‌کند: manifest از همان bytes hash/parse می‌شود؛ cache loader فقط
  labels manifest و رکورد file/bytes/shape همان study را می‌پذیرد؛ split cache
  از fold پین‌شدهٔ labels می‌آید؛ validator schema کامل target/keypoint را
  الزامی می‌کند. bundle آن در دو سمت SHA-256
  `66894694ce47409c9f9168ce415b3cfd8a3eb97531eff3d8f748c91548cb853a` دارد.
  unit contract سبک روی سرور 7/7 pass شد؛ هیچ DICOM یا forward مدل در آن اجرا
  نشد.
- پیکربندی اجرای زوج `G1-A / context9 / fold3 / seed42` در commit
  `bb94e32ed112740f1cb6b6fb7f0ca37dad442707` آماده است. SHA-256 bundle انتقالی
  `027601762ce94d587f0edd5674c577dd8c50b8d675feb7f8266b0ee5bc02ea1b` در دو
  سمت برابر بود. چون clone فعال دارای artifactهای اجراست، merge آن عمداً
  انجام نشد؛ commit تنها به ref محلی `bundle/g1-pair` fetch شده و manifest
  context9 آن تأیید شده است. worktree زوجِ متوقف سپس clean و detached به commit
  hardened `a2aa37e` منتقل شد و `G1-A` با Supervisor آغاز شد. این کار از
  overwrite خروجی C0 جلوگیری می‌کند.

### ممیزی تغییر source پس از قفل matrix — 2026-09-05

- receipt اصلی matrix از source control تاریخی در `2026-09-05T02:39:57Z`
  دوباره validate شد و **pass** کرد: 12 config، cache hash
  `c50ece…e672`، preregistration SHA-256
  `290522db625e8d9f4139a29da5d2f86bcf72b5f76539acc8d76ade93f3353c8f` و تنها
  causal difference=`input_channels`. پس خود matrix پیش از outcomeهای بعدی
  وجود داشته و contract آن سالم است.
- همان validator در worktree hardened `a2aa37e` عمداً fail-closed شد:
  `source hash changed after materialization: cache_validator`. diff دقیق نشان
  می‌دهد تغییر آن فایل تنها اجباری‌شدن ستون‌های `is_target` و شش keypoint در
  **validator cache** است؛ trainer یا معماری مدل را عوض نمی‌کند. بااین‌حال
  source hashهای `dataset/context_cache` نیز در hardening تغییر کرده‌اند و
  receipt قدیمی دیگر نمی‌تواند صادقانه ثابت کند C0 و A فقط در کانال ورودی
  تفاوت داشته‌اند.
- بنابراین `G1-A/fold3/seed42` که اکنون با source hardened اجرا می‌شود فقط
  یک **pilot اکتشافی** است: برای فهم ارزش 2.5D حفظ و پس از پایان audit می‌شود،
  اما به‌تنهایی نه gate می‌گیرد و نه مبنای promotion/submission است. برای
  screen تأییدی، پیش از هر اجرای تازه matrix جدید باید با source hardened
  materialize و validate شود و هر دو arm (از جمله C0) تحت همان receipt اجرا
  شوند. این تصمیم عمداً از نتیجه‌گیری نادرست، نه از مصرف GPU، جلوگیری می‌کند.

### matrix رسمیِ hardened برای screen بعدی

matrix جدید پیش از هر اجرای تأییدی تازه در مسیر جداگانهٔ
`/workspace/iaaa_artifacts/mls_g1_2p5d_hardened_a2_20260905/matrix` ساخته و
بلافاصله validate شد. preregistration SHA-256 آن
`15da3f43cb0699cf844e4be72a187ea78192de14eeb56484c8f983b7995dae3c` است؛
receipt اعتبارسنجی در `2026-09-05T02:42:25Z` pass شد: دقیقاً 12 config (6
fold-3 و 6 fold-4 شرطی)، cache immutable پیشین، و تنها تفاوت علّی
`input_channels`. این matrix جای receipt تاریخی برای **اجرای تأییدی جدید** را
می‌گیرد، اما pilot در حال اجرا را بازنویسی یا مشروع جلوه نمی‌دهد.

### hardening دوم: provenance داخل checkpoint

review evaluator نشان داد tagهای MLflow به‌تنهایی برای bind کردن source training
به checkpoint کافی نیستند؛ checkpoint قابل‌انتقال است و gate نباید به UI/حافظه
متکی باشد. source اکنون در هر checkpoint 2.5D نقشهٔ کامل SHA-256 همهٔ source
فایل‌های قفل‌شده را persist می‌کند؛ evaluator نبود/فرمت نادرست آن را reject و
gate همان نقشه را byte-for-byte با preregistration مقایسه می‌کند. runtime
evaluator نیز برای fileهای مشترک با matrix تطبیق داده می‌شود. نتیجه: حتی یک
checkpoint که با code پس از matrix ساخته شود، دیگر نمی‌تواند مخفیانه وارد gate
شود.

این hardening source matrix `a2` را هم منقضی می‌کند؛ پیش از نخستین اجرای
**تأییدی** باید source تازه commit/bundle شود و matrix `a3` جدید materialize و
validate گردد. این اثر عمدیِ fail-closed است. pilot فعلی همچنان فقط exploratory
است و نه source آن و نه artifact آن تغییر داده نمی‌شود.

commit `dac3312555083f2d220a32ac5b8a1060ee20ad0c` همین hardening را دارد. bundle
`g1-dac3312.bundle` با SHA-256
`069a04c5d20cc6feff6279b8765a4d3b843ebc67161ae6c0bc717d72b7730e09` به سرور
منتقل و برابر شد. worktree رسمیِ جدا (`g1_formal_dac3312`) با symlink همان raw
و cache ساخته شد؛ هیچ raw/cache جدیدی کپی نشد. `py_compile` سه ماژول تغییرکرده
در venv واقعی سرور pass شد.

matrix رسمی جدید در
`/workspace/iaaa_artifacts/mls_g1_2p5d_formal_dac3312_20260905/matrix` پیش از
هر CUDA outcome مربوط به آن source materialize و validate شد: SHA-256
preregistration=`9f23f8b8df750cccf6fd72c55531b501478ef2225af4f433d1c0dbb24ec125d6`،
12 config (6 fold-3 و 6 fold-4 شرطی)، تفاوت فقط `input_channels` و validation
در `2026-09-05T02:50:28Z` pass. فقط همین matrix مجاز است مبنای runهای رسمی
بعدی باشد.

برای جلوگیری از خطای دستی، شش config Supervisor متناظر با دقیقاً همین شش
manifest fold-3 در `/etc/supervisor/conf.d` سرور ساخته و `reread` شدند. هر شش
مورد در status=`available/manual` هستند: `g1_c0_3ch` و `g1_a_9ch` برای seedهای
42، 2026 و 3407. `autostart=false` و `autorestart=false` است؛ پس این اقدام هیچ
training تازه یا GPU allocation آغاز نکرده و فقط صف قابل‌ممیزی را آماده کرده
است. logها نیز در artifact directory رسمی جدا ثبت می‌شوند.

پس از ساخت، binding هر شش مورد به‌صورت read-only بررسی شد: برای هر arm/seed
مسیر worktree رسمی، نام دقیق YAML همان matrix، `--allow-training`، wrapper
MLflow با `IAAA_REQUIRE_REMOTE_MLFLOW=1`، و هر دو گزینهٔ `autostart=false` و
`autorestart=false` همگی pass شدند. بنابراین هیچ config دستی یا manifest
اشتباهی در صف رسمی باقی نمانده است.

MLflow pilot نیز از مسیر wrapper خود job به‌صورت read-only بازخوانی شد:
`62bf813c528b446cbf60f758bb6eb453` در status=`RUNNING` با 52 metric و tagهای
`g1_a_9ch`، fold=3، channel=9، cache immutable و policy CUDA-only است. یک probe
دستی نخست wrapper را bypass کرده و SQLite محلی `mlflow.db` با حجم 872,448 byte
ساخته بود؛ این file پیش از ثبت هر evidence حذف شد. probe صحیح فقط از wrapper
استفاده کرد و remote run را تأیید نمود؛ هیچ secret/prediction/raw artifact چاپ یا
منتقل نشد.

### پایان pilot A و تصمیم اجرای رسمی

pilot `G1-A/fold3/seed42` همهٔ 23 epoch را کامل کرد؛ report=`completed` و
Supervisor پس از upload artifact در `2026-09-05 03:10 UTC` خارج شد. checkpoint
قفل‌شدهٔ epoch 15 SHA-256
`183a7e576f3595cb5f6ef30c42c116c7f9862b644fb8c1b9bd379befe14390d6` دارد.
در epoch قفل‌شده، A نسبت به C0 تاریخی signal منفی دارد: slice-MAE
`2.3347` در برابر `2.5412` بهتر است، اما study-MAE `3.1751` در برابر `2.9159`
و boundary-F1 `0.7723` در برابر `0.8571` بدترند. بهترین validation داخلی pilot
در epoch 16 (`study-MAE=2.5752`, `boundary-F1=0.9168`) ظاهر شد، اما چون epoch
16 پس از fixed epoch=15 است، برای انتخاب یا ادعای رسمی قابل‌استفاده نیست. این
دقیقاً دلیل freeze شدن epoch پیش از مشاهدهٔ outcome است.

pilot به‌علت source provenance پیشین نه release است و نه gate را تغییر می‌دهد؛
فقط ریسک مسیر 9-channel را نشان می‌دهد. پس از خروج کامل آن، نخستین اجرای رسمی
`G1-C0/fold3/seed42` از queue hardened با Supervisor شروع شد (PID 76813). A
رسمی تا completion و audit این control شروع نمی‌شود؛ این ترتیب هم GPU را تک‌job
نگه می‌دارد و هم یک baseline دقیق با checkpoint-provenance جدید می‌سازد.

checkpoint رسمی C0 در epoch=15 ساخته شد: SHA-256
`4f4936ba62891c3ba495069384d30d70946164a445ca8b884cf81d1d37e3280e`.
مقایسهٔ مستقیم 18 source hash embedded آن با 18 source hash matrix رسمی
`9f23…125d6` برابر کامل و mismatch صفر بود. metricهای epoch 15 نیز دقیقاً C0
تاریخی را بازتولید کردند (`slice-MAE=2.5412`, `study-MAE=2.9159`,
`boundary-F1=0.8571`)، که نشان می‌دهد hardening provenance مسیر عددی مدل را
تغییر نداده است. training همچنان تا epoch 23 در حال اجراست؛ این receipt
checkpoint فقط pre-audit است و جای completion/MLflow audit کامل را نمی‌گیرد.

`G1-C0/fold3/seed42` در `2026-09-05T03:57:16Z` همهٔ 23 epoch را completed کرد؛
پس از پایان upload، Supervisor در `03:59 UTC` به `EXITED` و MLflow run
`7a3c9199d4a046348523edeba2953d43` به `FINISHED` رسید (52 metric،
`g1_c0_3ch`، CUDA-only). سپس و فقط سپس، `G1-A/fold3/seed42` رسمی از queue
همان matrix به Supervisor add/start شد (PID 79099). این A با checkpoint
provenance source جدید است؛ دیگر همان pilot اکتشافیِ قدیمی نیست.

## برآورد امید

احتمال عددی قابل‌اعتماد برای عبور از leaderboard نداریم؛ leaderboard private
است و comparator رسمی کامل بازسازی نشده. بر مبنای signal داده، ادبیات MLS
(localization چنداسلایسی/landmark-based) و شکست A9/A10، مسیر G1 نسبت به
loss-tuning تکراری فرصت واقعی‌تری دارد. برآورد عملیِ مشروط برای گذر از gate
سخت G1 در نخستین screen حدود **15 تا 30 درصد** است؛ این پیش‌بینی نیست و تنها
برای تصمیم منابع است. احتمال رشد در همان خانوادهٔ A9/A10 کمتر از 10 درصد
ارزیابی می‌شود.

### ممیزی زنجیرهٔ cache پس از launch رسمی

یک review مستقل در ابتدا چهار ریسک محتمل را دربارهٔ schema spacing، یکپارچگی
volumeهای cache، split fold و labels مطرح کرد. بررسی مستقیم snapshot دقیق
رسمی `dac3312` نشان داد این‌ها در source frozen حاضر **رفع‌شده‌اند** و نباید
به‌عنوان defect اجرای G1 ثبت شوند: builder از `PixelSpacing0/1` واقعی استفاده
می‌کند (با نگاشت DICOM row/column درست)، dataset فقط CSV پین‌شده در manifest را
می‌پذیرد، record هر study شامل filename/hash/bytes/shape است و loader آن را
بررسی می‌کند، و در cache mode split از ستون `fold` داخل labels معتبر cache
می‌آید نه manifest قابل‌تغییر پروژه. بنابراین این ممیزی هیچ دلیل جدیدی برای
ابطال C0 رسمی یا A رسمیِ در حال اجرا ایجاد نکرد. با این حال، هر outcome هنوز
تا audit نهایی checkpoint، Supervisor و MLflow، صرفاً pre-gate است.

### نتیجهٔ review علمی مستقل و pivot مشروط

شواهد فعلی نه از فقدان ظرفیت MLS، بلکه از mismatch میان regression-MAE و تصمیم
triage آستانه‌ای حکایت دارند: در A10، MAE بهتر شد اما F1 در مرز 3mm افت کرد؛
چنین جابه‌جایی کوچکی برای نمونه‌های نزدیک 3 و 5mm می‌تواند کلاس Urgent/Critical
را عوض کند. در دادهٔ توسعه به‌ترتیب 28 و 27 study نزدیک این دو مرز وجود دارد؛
پس MAE به‌تنهایی surrogate مناسبی برای Macro-F1 یا Urgent-F1 نیست.

G1 همچنان screen درستِ نخست است، چون تنها context سه اسلایس مجاور را با سه
window اضافه می‌کند و همهٔ عوامل دیگر ثابت‌اند. ادبیات نیز فرضیهٔ spatial
context را پشتیبانی می‌کند (Yan et al., *Diagnostics* 2022؛ Nguyen et al.,
ICCVW 2021). اما اگر A در epoch=15 رسمی signal منفیِ pilot را تکرار کند، این
فقط hypothesis محدود «این context ±1 با این backbone» را رد می‌کند، نه کل
مسیر 2.5D/3D را. در آن حالت، قبل از هر training تازه، campaign جدید و مستقل
با preregistration جدید ساخته می‌شود: trunk مشترک، regression MLS همراه
`P(MLS>=3)` و `P(MLS>=5)` با قید monotonic (CORAL/CORN)، calibration تنها در
inner-fold، و گزارش اجباری signed-error، F1@3، F1@5، Macro-F1 و Urgent-F1 روی
موارد مرزی. این pivot مستقیم‌تر با هدف leaderboard هم‌راستاست و از انتخاب
epoch یا loss پس از دیدن validation جلوگیری می‌کند.
