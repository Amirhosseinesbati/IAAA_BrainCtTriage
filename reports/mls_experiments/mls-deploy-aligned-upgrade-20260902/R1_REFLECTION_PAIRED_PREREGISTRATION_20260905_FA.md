# R1 — آزمون paired بازتاب افقی MLS

> وضعیت به‌روزشده: control کامل و با raw-DICOM و runtime پکیج تأیید شده است؛
> candidate از پیش‌ثبت‌شده در حال آموزش است. این R1 همچنان یک screen تک-fold/تک-seed
> است و هیچ checkpoint یا ZIP برای production تأیید نشده است.

## پرسش

آیا بازتاب افقیِ training-only، با حفظ دقیق landmarkها و MLS مطلق، می‌تواند
generalization مدل central-slice را بهبود دهد بدون آن‌که contract inference،
pooling یا threshold تغییر کند؟ این فرضیه از G1 متمایز است: context جدید،
head جدید، loss جدید، TTA یا تغییر aggregation ندارد.

## طراحی قفل‌شده

- fold: `1` (استفاده‌نشده در screenهای A9/A10 و G1 fold3)
- seed: `42`
- checkpoint audit: epoch ثابت `15`
- input: cache معتبر 2.5D، اما فقط central three-window (`input_channels=3`)
- control: `horizontal_flip_prob=0.0`
- candidate: `horizontal_flip_prob=0.5`
- اختلاف مجاز بین configها: فقط `horizontal_flip_prob`
- CUDA-only و MLflow اجباری؛ CPU model fallback ممنوع

matrix قبل از CUDA ساخته و بررسی شده است:

- مسیر remote: `/workspace/iaaa_artifacts/mls_reflection_r1_20260905/matrix/r1_preregistration.json`
- SHA-256: `fafd150929703f42764290711b7247f30966092d458be61b31570b05d29ce4c0`
- cache manifest: `c50ece4167b25661a7e36305bfab6f177253981c8418c0fd85acbf23bde4e672`
- receipt cache: `c0812f85d74ae6759a6d7c5ca473826de30923b7552d2878ac0b381d0ba8beb1`

## ترتیب اجرا و تصمیم

1. control باید completion، MLflow و checkpoint دقیق epoch 15 را با source/cache
   provenance معتبر نشان دهد.
2. فقط در این صورت candidate شروع می‌شود؛ restart، epoch selection یا config
   rescue مجاز نیست.
3. هر دو checkpoint با evaluator `evaluate_mls_r1_single_fold_cuda.py` روی raw
   DICOM و CUDA ارزیابی می‌شوند. predictionهای per-study خصوصی باقی می‌مانند.
4. candidate باید حداقل در study MAE، F1@3، F1@5 و Boundary-F1 نسبت به control
   non-inferior باشد تا وارد ارزیابی triage frozen شود.
5. عبور از این screen به‌تنهایی promotion نیست. فقط بهبود leak-free و
   deploy-aligned `Macro-F1` و `Urgent-F1` در gate نهایی، با عدم regression
   catastrophic error، اجازهٔ انتقال checkpoint یا ساخت ZIP می‌دهد.

هر شکست در هر مرحله، R1 را بدون seed/fold/threshold sweep جدید رد می‌کند.

## به‌روزرسانی اجرایی — control و gateهای فنی

### C0 — تکمیل control

اجرای control در `2026-09-05T14:41:22Z` کامل شد. checkpoint قفل‌شدهٔ epoch 15:

- مسیر: `/workspace/IAAA_BrainCtTriage_mls_reflection/models/checkpoints/mls_multitask/mls-reflection-control-fold1-seed42/mls_multitask_epoch_015.pth`
- SHA-256: `8a8240e0d952627dd6c2a22b6366aa8dd289b151c82421be5ea6050549e01102`
- MLflow run: `fbc9a8f38a74457eb45e9d48f60c31e8`
- سیاست compute: `cuda_only_no_cpu_model_fallback`

log داخلی training در epoch 15 عددهای `study_MAE=0.986` و `study_BF1=0.818`
را نشان می‌دهد. این‌ها برای انتخاب checkpoint ثابت ثبت شده‌اند، اما **مبنای تصمیم
کیفی یا ادعای leaderboard نیستند**؛ evaluator مستقل raw-DICOM در ادامه مبنا است.

### E1 — ارزیابی مستقل raw-DICOM

اولین invocation evaluator پیش از inference به‌دلیل نبودن مسیر truth table در clone
مجزای evaluation متوقف شد؛ هیچ checkpoint یا داده‌ای تغییر نکرد. سپس مسیر truth
موجود، SHA آن، و یک الزام fail-closed جدید به evaluator افزوده شد. اجرای نهایی در
`2026-09-05T15:17:04Z` با receipt زیر کامل شد:

- receipt: `/workspace/iaaa_artifacts/mls_reflection_r1_20260905/strict_evaluation/control_fold1_seed42/aggregate_summary.json`
- پوشش: `67/67` study از fold 1، CUDA-only
- SHA truth table: `70a3551d9460c73e665cdd3ca6037407f1854152b211e7dfee09394bae149a94`
- SHA cache receipt: `c0812f85d74ae6759a6d7c5ca473826de30923b7552d2878ac0b381d0ba8beb1`
- runtime: `39.02 s`؛ peak VRAM: `0.560 GiB`

| معیار مستقل | control epoch 15 |
|---|---:|
| MAE | 1.531936 mm |
| F1@1 mm | 0.794521 |
| F1@3 mm | 0.754717 |
| F1@5 mm | 0.702703 |
| Boundary-F1 | 0.728710 |
| selection objective | 2.074517 |

این شکاف با metric داخلی training نباید به‌عنوان فساد داده تعبیر شود: metric داخلی
روی مسیر validation/cache همان trainer به‌دست آمده، در حالی‌که E1 یک اجرای مستقل
روی DICOM خام و reader/aggregation deployment است. از این نقطه به بعد فقط E1 و
ارزیابی‌های deploy-aligned بعدی معیار تصمیم هستند. predictionهای per-study خصوصی
مانده‌اند و تنها SHA آن‌ها در receipt ثبت شده است.

### P1 — parity پکیج submission

در `2026-09-05T15:22:36Z`، `verify_mls_r1_submission_parity_cuda.py` همان
checkpoint را بین source runtime و runtime self-contained پکیج در همهٔ 67 study
مقایسه کرد:

- receipt: `/workspace/iaaa_artifacts/mls_reflection_r1_20260905/package_parity/control_fold1_seed42.json`
- `status=passed`، `batch_size=8`، `atol=1e-6`، `errors=[]`
- همهٔ deltaهای `selector_probability`، `peak_probability`، `heatmap_peak`،
  `mls_mm` و `study_mls_mm` برابر `0.0` بودند.
- input volume HU و `spacing_x` نیز برای هر study برابر بودند.

پس افت E1 از پکیج submission ناشی نمی‌شود. P1 فقط هم‌ارزی اجرایی را اثبات می‌کند؛
نه برتری بالینی/leaderboard و نه مجوز ساخت ZIP.

### وضعیت candidate و تصمیم بعدی

پس از C0/E1/P1، candidate قفل‌شدهٔ `horizontal_flip_prob=0.5` با همان fold، seed،
epoch، cache، backbone و MLflow در GPU شروع شد. هیچ sweep یا rescue انجام نمی‌شود.
پس از اتمام آن، checkpoint epoch 15 با همین E1 و P1 ارزیابی می‌شود و فقط اگر در
MAE، F1@3، F1@5 و Boundary-F1 نسبت به control raw-DICOM non-inferior باشد، وارد
gate کامل triage خواهد شد. تا آن زمان:

- `promotion_eligible=false`
- `submission_zip_allowed=false`
- Macro-F1 و Urgent-F1 نهایی هنوز محاسبه نشده‌اند.
