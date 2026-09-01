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

## به‌روزرسانی نهایی readiness پس از بازگشت موقت اتصال

پس از روشن‌شدن VPN، Vast یک‌بار وضعیت authoritative زیر را برگرداند:

- instance: `49527185`
- `actual_status=running`
- intended/next state: `running`
- GPU utilization: صفر در زمان مشاهده
- disk usage: حدود ۱۵GB

سپس SSH proxy دوباره برقرار شد و همه گیت‌های قبلی واقعاً اجرا شدند:

1. clone با `git pull --ff-only` به‌روزرسانی و پس از pull clean بود.
2. محیط پروژه `torch 2.10.0+cu128`، CUDA 12.8 و RTX 3060 با 11.63GiB را دید.
3. `dvc status -c Data/raw.dvc` اعلام کرد cache و remote `origin` sync هستند.
4. ۲۶GB فضای آزاد باقی بود.
5. health-check فقط‌خواندنی MLflow موفق بود و حداقل یک experiment دیده شد.
6. baseline fold2 با `strict=True` روی CUDA load شد؛ یک forward واقعی CUDA خروجی‌های
   finite با شکل heatmap `(1, 3, 128, 128)` و selector `(1,)` تولید کرد. مصرف
   تخصیص‌یافته حدود 0.916GiB بود.
7. manifest کنترل‌شده Exp14 روی سرور validate شد و `training_config` معتبرشده آن
   دقیقاً با Exp10 یکسان بود.

manifest در commit `81c5d3c` ثبت شد:

`config/experiments/mls-vast-exp14-w32-fold2-hybridsoft-repro.yaml`

Goal قبلی سپس توسط کاربر به‌صورت دستی resume شد. **با وجود این، Exp14 هنوز شروع
نشده است.** تلاش اولیه‌ی launch پیش از ساخت tmux session توسط محافظ `pgrep`
متوقف شد، زیرا pattern می‌توانست متن فرمان جاری را نیز match کند. پس از آن SSH
proxy دوباره timeout شد؛ هیچ session، پردازش آموزش یا MLflow run جدیدی از این
تلاش ساخته نشد.

## سخت‌سازی پس از readiness

برای اینکه رخداد بالا تکرار نشود، این تغییرات local-first پیاده‌سازی، تست، commit
و push شدند:

- `976f124`: sigma annealing اختیاری و resume-deterministic؛ رفتار Exp14 ثابت ماند.
- `6feb95d`: preregistration تک‌عاملی Exp15، مشروط به عبور Exp14.
- `69cb2d9`: گیت خودکار بازتولید Exp14 با خروجی JSON ماندگار.
- `a40851d`: لانچر tmux با lock اتمیک و `status.json`؛ بدون `pgrep` مبهم.

لانچر جدید اجرای یکتا را با `run.lock` اثبات می‌کند و commit، زمان شروع/پایان،
exit code، مسیر لاگ، `compute_policy=cuda_only` و `auto_destroy=false` را ثبت
می‌کند. وجود پوشه‌ی خالی لاگ از تلاش قبلی مانع launch نیست؛ فقط lock یا status
واقعی مانع اجرای دوباره خواهد بود. ۱۶ تست سبک مرتبط پاس شده‌اند و هیچ forward،
backward یا inference مدل روی CPU محلی اجرا نشده است.

## مانع فعلی و فرمان بعدی

CLI رسمی `vastai` دیگر در PATH نشست Windows/WSL قابل مشاهده نیست؛ بنابراین طبق
قاعده افزونه، `vastai logs 49527185` پس از SSH timeout قابل اجرا نبود. از نصب
دوباره CLI، فراخوانی مستقیم API یا binary جایگزین عمداً خودداری شد.

پس از اجازه صریح کاربر برای retry مستقیم SSH بدون Vast logs، ترتیب بعدی باید این
باشد:

1. pull شاخه تا commit `a40851d` یا جدیدتر و تأیید clean بودن clone؛
2. بررسی نبود `status.json`/`run.lock`/tmux session برای Exp14؛
3. launch دقیقاً یک‌بار با `scripts/launch_vast_mls_tmux.py`؛
4. اثبات شروع از `status.json`، لاگ، `nvidia-smi` و MLflow run؛
5. پایش تا completion و اجرای `scripts/evaluate_mls_repro_gate.py`؛
6. اجرای Exp15 فقط در صورت پاس‌شدن گیت preregisterشده.

سرور نباید بدون هماهنگی کاربر stop یا destroy شود. تمرکز این مرحله فقط MLS است؛
هیچ artifact مربوط به ICH ورودی Exp14 یا Exp15 نیست.
