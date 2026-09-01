# هَندآف مهاجرت MLS به Vast.ai

تاریخ: 2026-09-01  
وضعیت: **فاز آماده‌سازی؛ goal آموزشی هنوز فعال نشده است**  
قاعدهٔ lifecycle: هیچ stop/destroy/recycle برای سرور این مسیر بدون هماهنگی کاربر انجام نمی‌شود.

## 1. نقطهٔ شروع علمی که نباید دوباره از صفر تحلیل شود

- baseline قابل‌دفاع: سه مدل `HRNet-W32` با target نوع `hybrid-soft`، sampler قدیمی
  `slice_class_balanced` و checkpoint ثابت epoch15 در foldهای 0/1/2.
- E2E diagnostic به‌ترتیب fold: `1.6646 / 1.2587 / 1.7144 mm`؛ mean=`1.5459`،
  worst=`1.7144` و mean Boundary-F1=`0.8469`.
- برآورد سخت‌گیرانهٔ nested: mean MAE=`1.6273 mm`، worst=`1.7999` و
  Boundary-F1=`0.8208`؛ سیاست boundary-first برای epoch15 به mean=`1.6045` و
  Boundary-F1=`0.8344` رسید.
- Exp12 با study balancing کامل و Exp13 با exposure میانی هر دو معیار production
  قفل‌شده را شکست ندادند؛ شاخهٔ sampler بدون فرضیهٔ تازه بسته است.
- بهترین عدد خام منفرد Exp08/fold0/epoch21 با MAE=`1.2334 mm` diagnostic است و
  به‌علت انتخاب درون همان fold، production-safe محسوب نمی‌شود.
- گلوگاه اثبات‌شده بیشتر variance checkpoint، calibration selector و pooling مطالعه
  است، نه خرابی بنیادی backbone.

منابع مرجع:

1. `MLS_FINAL_PAUSE_HANDOFF_2026-08-28_FA.md`
2. `MLS_ONE_DAY_DEEP_PROGRESS_REPORT_2026-08-28_FA.md`
3. `mls-local-v2-exp10-w32-fold2-hybridsoft-transfer/checkpoint_audit_report.md`
4. `mls-local-v2-exp13-w32-fold2-hybridsampler/checkpoint_audit_report.md`

## 2. gateهای دوام که پیش از train تازه پیاده شدند

علت: Exp13 پس از validation ایپاک 19 در `mlflow.log_metrics` با DNS failure متوقف
شد و چون تماس شبکه پیش از snapshot بود، checkpoint ایپاک 19 از دست رفت.

اصلاحات این دور:

- تمام best checkpointها، audit snapshotها، report و full recovery checkpoint پیش از
  ثبت metric شبکه‌ای ذخیره می‌شوند.
- نوشتن checkpoint با temporary file و `os.replace` اتمیک است.
- `mls_multitask_resume_latest.pth` شامل model، optimizer، scheduler، AMP scaler،
  epoch، history، best-state و RNGهای Python/NumPy/Torch/CUDA است.
- `resume_checkpoint` به config اضافه شد و mismatchهای backbone/fold/channels/sampler
  پیش از restore رد می‌شوند.
- تماس‌های MLflow پس از retry کوتاه nonfatal می‌شوند، circuit breaker دارند و event
  قابل replay را بدون credential در `reports/mlflow_pending_events.jsonl` ثبت می‌کنند.
- `scripts/replay_mlflow_queue.py` برای sync مجدد metric/param/tag/artifact اضافه شد.
- تست قطع مصنوعی شبکه و تست manifest داده پاس شدند؛ هیچ train یا model inference روی
  CPU محلی اجرا نشد.

## 3. وضعیت داده و ظرفیت دیسک

`Data/raw.dvc`:

- DVC dir hash: `e448815d41bdee148f38acdf57e192a9.dir`
- file count: `12,860`
- bytes: `2,908,071,754` (`2.708 GiB`)
- `dvc status -c`: cache محلی و remote `origin` در sync هستند.

manifest مستقل محلی که DVC را دور می‌زند و هر فایل را SHA-256 می‌کند:

- file count: `12,860`
- bytes: `2,908,071,754`
- tree SHA-256: `308cf43a46d999c34a1f9bfeea0511a65a21b125680a0a38e59bf2b820425b93`
- فایل موقت محلی: `.tmp/vast_migration/local_raw_sha256.jsonl`؛ به Git/MLflow
  فرستاده نمی‌شود چون نام فایل‌های پزشکی ممکن است linkage داشته باشد.

فضای فعلی مورد نیاز:

- raw: `2.71 GiB`
- processed MLS: حدود `8.80 GiB`
- محیط Python مشابه محلی: حدود `5.70 GiB`
- baseline سه‌fold epoch15: حدود `0.35 GiB`
- ICHهای لازم برای provenance/full-pipeline: حدود `0.27 GiB`

با انتقال انتخابی artifactها، 40GB قابل استفاده است؛ free-space باید بعد از image،
`uv sync`، DVC pull و ساخت processed data دوباره اندازه‌گیری شود و پیش از هر run حداقل
headroom امن برای full recovery checkpoint و snapshotها وجود داشته باشد.

## 4. artifactهای خارج DVC که باید منتقل و hash-verify شوند

### MLS baseline

| fold | فایل | bytes | SHA-256 |
|---:|---|---:|---|
| 0 | Exp08 epoch15 | 124890565 | `9427c2353d81e3fda412f7e113077e84f9d86d47a06ecd856042aa919d9a7716` |
| 1 | Exp09 epoch15 | 124890565 | `98923f724b2d61c4a8671ef0405ab7c205913e0829b72c0e32ae317ef23cfccb` |
| 2 | Exp10 epoch15 | 124890565 | `7cd2b5b09bacd7f8803b97c35f8cc1e451ee2003a704742239c88ee6d1d1770f` |

### ICH

- بستهٔ submission فعلی: `presence_gate.pth` و `segresnet.pth` با hashهای ثبت‌شده در
  `submission/MODEL_MANIFEST.json`.
- کاندید جدیدتر hard-pixel پنج‌fold: fold0..4 با hashهای ثبت‌شده در
  `checkpoint/ich/smp/2p5d/unetplusplus-efficientnet-b2-hardpixel-fprselect-oof-candidate-20260901/README.md`.
- برای جلوگیری از ابهام بین «مدل داخل submission فعلی» و «جدیدترین کاندید ICH»، هر دو
  مجموعه منتقل می‌شوند؛ هیچ‌کدام مبنای pretrain MLS فرض نمی‌شوند مگر آزمایش مستقلی آن
  را توجیه کند.

## 5. سیاست Git در شاخهٔ مشترک

- شاخه: `codex/competition-winning-pipeline`.
- قبل از commit، `fetch` انجام و local HEAD با origin هم‌تراز شد.
- worktree تغییرات زیاد و هم‌زمان از ICH، fracture، submission و UI دارد؛ commit کلی
  ممنوع است.
- فقط فایل‌ها/هَنک‌های مشخص MLS durability، ابزار integrity، bootstrap و همین گزارش
  stage می‌شوند. checkpoint، prediction خام، `.env` و تغییرات سایر افراد وارد commit
  نمی‌شوند.
- اگر روی سرور تغییر اضطراری ایجاد شود، ابتدا diff دقیق ثبت می‌شود، سپس commit/push
  محدود و در اولین فرصت local pull/reconcile انجام خواهد شد.

## 6. سیاست secrets و bootstrap سرور

- secretها در command line، Git، MLflow artifact یا گزارش چاپ نمی‌شوند.
- `.env` با `scp` به مسیر root-only روی سرور منتقل و permission آن `600` می‌شود.
- DVC credential فقط در `.dvc/config.local` با permission محدود نوشته می‌شود.
- Git clone برای read از URL عمومی انجام می‌شود؛ token GitHub به سرور منتقل نمی‌شود.
- `scripts/bootstrap_vast_workspace.sh` فقط workspace/env/DVC را آماده می‌کند؛ هیچ
  آزمایشی شروع و هیچ سروری stop/destroy نمی‌کند.

## 7. معیار انتخاب و ممیزی عملی RTX 3060

فیلتر اولیه:

- یک `RTX 3060` با VRAM واقعی حداقل 12GB، verified/rentable و direct SSH؛
- reliability ترجیحاً بالاتر از `0.97`؛
- CPU/RAM کافی برای DataLoader و preprocessing؛
- اینترنت اعلامی قوی و قیمت on-demand پایین؛ spot برای run پایه انتخاب نمی‌شود؛
- disk صریح `40GB` و `--cancel-unavail`.

پس از اجاره، اعداد آگهی کافی نیستند. نگه‌داشتن سرور منوط به ثبت عملی این موارد است:

1. `nvidia-smi`، نام GPU، VRAM، driver/CUDA و خطاهای ECC/Xid؛
2. CUDA tensor smoke و benchmark کوتاه GPU؛
3. تعداد/مدل CPU، RAM و benchmark کوتاه چندریسمانی؛
4. throughput و latency واقعی دانلود/آپلود؛
5. sequential/random disk I/O و فضای آزاد؛
6. پایداری SSH و نبود throttling آشکار.

اگر سرور ضعیف یا مغایر آگهی باشد، بدون هماهنگی کاربر destroy/stop نمی‌شود و نتیجه با
پیشنهاد جایگزین گزارش می‌شود.

## 8. ترتیب ادامه و شرط فعال‌سازی goal

1. commit/push محدود فایل‌های آماده‌سازی.
2. بررسی account/balance/SSH key و offerهای Vast.
3. اجاره، benchmark عملی و تأیید نگه‌داری.
4. clone/pull، bootstrap، DVC pull و ساخت manifest SHA-256 سرور.
5. مقایسهٔ فایل‌به‌فایل manifest محلی/سرور.
6. SCP و SHA-256 artifactهای MLS/ICH خارج DVC.
7. smoke پایداری checkpoint/MLflow و readiness report.
8. **فقط در این نقطه** goal آموزشی فعال و اولین آزمایش preregister می‌شود.

اولویت علمی پس از فعال‌سازی: ابتدا calibration/aggregation و context سبک/selector
با پروتکل cross-fold قفل‌شده بررسی می‌شود؛ samplerهای Exp12/13 بدون فرضیهٔ تازه تکرار
نمی‌شوند. SciSpace/مقالات فقط برای تصمیم‌های معماری واقعی استفاده و ادعاها از metricهای
production-safe جدا نگه داشته می‌شوند.
