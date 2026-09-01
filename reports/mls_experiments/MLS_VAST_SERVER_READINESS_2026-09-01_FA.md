# گزارش آمادگی سرور Vast برای ادامه بهبود MLS — ۲۰۲۶-۰۹-۰۱

## وضعیت اجرایی

- وضعیت فعلی: **آمادگی ناقص / در انتظار بازگشت اتصال پایدار و health-check نهایی**
- Goal آموزش MLS: **فعال نشده است**
- هیچ آزمایش آموزشی روی سرور شروع نشده است.
- سرور طبق دستور کاربر روشن نگه داشته شده و هیچ stop/destroy/recycle انجام نشده است.
- شناسه instance: `49527185`
- label: `iaaa-mls-vast-20260901`
- هزینه ثبت‌شده هنگام اجاره: حدود `0.06963 USD/hour` با دیسک ۴۰ گیگابایت

## انتخاب و ارزیابی عملی میزبان

پیشنهاد منتخب یک RTX 3060 در Ontario بود که نسبت به گزینه ارزان‌تر ویتنام، با اختلاف هزینه ناچیز، reliability بالاتر، CPU مؤثر بیشتر و هزینه انتقال داده بسیار کمتر داشت.

نتایج مشاهده و آزمون عملی:

- GPU: NVIDIA GeForce RTX 3060 با ۱۲٬۲۸۸ MiB VRAM و توان ۱۷۰ وات
- driver میزبان: `580.82.09`
- CPU: دو سوکت Xeon E5-2680 v4، مجموع ۵۶ thread؛ مقدار اعلامی بازار ۱۴ core مؤثر
- RAM: حافظه قابل مشاهده بسیار بیشتر از حداقل ۳۲GB موردنیاز؛ `mem_limit` گزارش Vast حدود ۱۱۰.۸GB بود
- دیسک مستقیم: حدود ۲.۰ GB/s نوشتن و ۳.۰ GB/s خواندن برای فایل ۵۱۲MiB
- CPU SHA-256 چندپردازه: حدود ۲.۹ GB/s در بلوک بزرگ
- شبکه تک‌جریان خارجی: حدود ۸۲ Mbps دانلود و ۳۷ Mbps آپلود
- عدد نزدیک ۹۰۰ Mbps آگهی به‌صورت end-to-end تأیید نشد.
- SSH مستقیم هنگام انتقال موازی چند بار reset شد؛ مسیر proxy خود Vast همراه SFTP resume پایدارتر بود و باید برای artifactهای مهم ترجیح داده شود.

## مسئله CUDA و رفع آن

`nvidia-smi` از ابتدا GPU را می‌دید، اما PyTorch ایمیج با خطای CUDA 803 بالا نمی‌آمد. علت، جلو افتادن کتابخانه forward-compatibility ایمیج بود:

- libcuda میزبان: `libcuda.so.580.82.09`
- libcuda ناسازگار compat داخل ایمیج: `libcuda.so.560.35.05`

با قرار دادن `/usr/lib/x86_64-linux-gnu` در ابتدای `LD_LIBRARY_PATH`، هم PyTorch خود ایمیج و هم محیط قفل‌شده پروژه سالم شدند:

- محیط پروژه: `torch 2.10.0+cu128`
- CUDA wheel: 12.8
- `torch.cuda.is_available() == True`
- بنچمارک سبک FP16 ماتریسی: حدود ۲۶.۵ TFLOPS و peak VRAM حدود ۱.۵GiB

این اصلاح در commit `c5fe9a2` ثبت و push شد:

- `scripts/bootstrap_vast_workspace.sh`
- `scripts/run_vast_mls_experiment.sh`

## Git، محیط و DVC

- clone اولیه از شاخه `codex/competition-winning-pipeline` روی commit `c5fe9a2` انجام شد.
- clone قبل از دریافت داده clean بود.
- محیط پروژه با `uv sync --frozen --no-install-package vastai` ساخته شد؛ Vast CLI عمداً داخل محیط training نصب نشده است.
- فایل secrets فقط شامل چهار تنظیم ضروری DagsHub و دو تنظیم غیرمحرمانه runtime است.
- مجوز secrets: `600 root:root`
- هیچ مقدار secret داخل Git، گزارش، command output عمدی یا MLflow queue نوشته نشده است.
- `dvc pull -r origin Data/raw.dvc` موفق شد.
- فضای دیسک پس از نصب محیط و DVC pull: حدود ۲۶GB آزاد از ۴۰GB.

## تطبیق مستقل Data/raw

کنترل DVC و سپس SHA-256 مستقل تک‌تک فایل‌ها انجام شد:

- تعداد فایل محلی: ۱۲٬۸۶۰
- تعداد فایل سرور: ۱۲٬۸۶۰
- مجموع بایت دو سمت: ۲٬۹۰۸٬۰۷۱٬۷۵۴
- SHA-256 درخت canonical دو سمت:
  `308cf43a46d999c34a1f9bfeea0511a65a21b125680a0a38e59bf2b820425b93`
- فایل گمشده در هر سمت: صفر
- فایل دارای اختلاف اندازه یا hash: صفر

نتیجه: `Data/raw` محلی و سرور به‌صورت فایل‌به‌فایل و byte-for-byte یکسان‌اند.

## artifactهای منتقل‌شده خارج Git/DVC

artifactها در `/workspace/iaaa_artifacts` نگهداری می‌شوند تا repository سرور dirty نشود.

### baselineهای معتبر MLS

سه snapshot ثابت epoch 15 که handoff قبلی آن‌ها را baseline دفاع‌پذیر معرفی کرده بود:

- fold0 / Exp08: `9427c2353d81e3fda412f7e113077e84f9d86d47a06ecd856042aa919d9a7716`
- fold1 / Exp09: `98923f724b2d61c4a8671ef0405ab7c205913e0829b72c0e32ae317ef23cfccb`
- fold2 / Exp10: `7cd2b5b09bacd7f8803b97c35f8cc1e451ee2003a704742239c88ee6d1d1770f`

SHA-256 هر سه در دو سمت یکسان است.

### مدل‌های ICH — رفع ابهام مهم

**همه checkpointهای ICH منتقل نشده‌اند.** فقط این دو گروه منتقل شدند:

1. دو وزن واقعاً مصرف‌شده توسط submission فعلی:
   - `presence_gate.pth`
   - `segresnet.pth`
2. پنج fold پوشه `hardpixel-fprselect-oof-candidate-20260901` به همراه metadata آن.

گروه دوم صرفاً «کاندید جدید منسجم دارای OOF report» است و **بهترین مدل ICH اثبات‌شده محسوب نمی‌شود**. هیچ audit جامع برای رتبه‌بندی تمام مدل‌های ICH در این مرحله انجام نشده است. این کاندیدها برای آموزش MLS به‌طور خودکار load نمی‌شوند و تا زمان ارزیابی مستقل باید با برچسب زیر در نظر گرفته شوند:

`UNVERIFIED_FOR_MLS_BASE_USE`

تمام ۱۲ فایل ICH منتقل‌شده و هر سه فایل MLS با SHA-256 محلی تطبیق داده شدند و اختلاف صفر بود.

## رخداد اتصال پس از انتقال

پس از کامل‌شدن DVC و انتقال/hash همه artifactها، هر دو مسیر SSH مستقیم و proxy timeout شدند و Vast CLI نیز موقتاً برای logs/status پاسخ نداد. وضعیت رسمی Vast در همان زمان outage سراسری اعلام نمی‌کرد. بنابراین readiness نهایی تا بازگشت اتصال باز می‌ماند.

هیچ داده یا checkpointی در معرض ناقص‌بودن نیست؛ آخرین عملیات موفق قبل از رخداد، hash کامل همه artifactها بود.

## گیت‌های باقی‌مانده پیش از فعال‌کردن Goal

1. بازگشت SSH پایدار و تأیید `actual_status=running`.
2. `git fetch` و `git pull --ff-only` و سپس تأیید clean بودن clone.
3. اجرای `dvc status -c`.
4. health-check فقط‌خواندنی MLflow/DagsHub بدون ساخت run آزمایشی.
5. CUDA smoke نهایی در محیط پروژه و load-state-dict کنترل‌شده یکی از baselineهای MLS روی GPU، بدون train روی CPU.
6. ثبت manifest اولین آزمایش جدید و سپس—و فقط سپس—فعال‌کردن Goal و بازکردن gate `--allow-training`.

## تصمیم عملی بعدی

تا وقتی گیت‌های بالا پاس نشده‌اند، هیچ آموزش جدیدی شروع نمی‌شود. پس از readiness، تمرکز فقط MLS است؛ artifactهای ICH صرفاً مرجع خارجی‌اند و انتخاب یا توسعه ICH خارج از scope این مرحله است.
