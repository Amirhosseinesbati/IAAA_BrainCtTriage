# R1 — آزمون paired بازتاب افقی MLS

> وضعیت به‌روزشده: control و candidate کامل، با raw-DICOM و runtime پکیج تأیید
> شده‌اند؛ R1R برای دو seed مستقل باقی‌مانده قفل و نخستین replica شروع شده است.
> این evidence هنوز development-only است و هیچ checkpoint یا ZIP برای production
> تأیید نشده است.

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

تحلیل مسیرها علت دقیق‌تر شکاف را نشان می‌دهد. validation داخل trainer تنها
`3484` row انتخاب‌شدهٔ cache را می‌بیند، نه همهٔ sliceهای سری کامل؛ سپس آن‌ها را
global rank کرده و top-k/quantile می‌گیرد. این proxy از `relative_component` محوری
R1، شرط `min_active_slices=3` روی سری کامل، و sliceهای بدون label استفاده نمی‌کند.
همچنین decode آن coarse grid argmax است، در حالی که E1 spatial-softmax و DARK
sub-pixel را اجرا می‌کند. پس تعریف MAE/Boundary-F1 عوض نشده، اما support، decoder و
pooling عوض شده‌اند. cache receipt از fingerprint فایل‌های raw، SOP order، spacing و
geometry محافظت می‌کند و P1 هم برای HU/spacing و تمام خروجی‌ها delta صفر داده است؛
بنابراین فعلاً نشانه‌ای از corruption انتقال داده نداریم.

با وجود این، پیش از gate نهایی یک audit مدل‌-free باقی می‌ماند: fingerprint/SOP/spacing
هر 67 study فعلی با cache قفل‌شده دوباره تطبیق و مقدار truth E1 با `study_mls_mm`
cache دقیقاً assert می‌شود. این audit برای توضیح provenance است، نه بهانه‌ای برای
تغییر model یا انتخاب مجدد epoch.

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

### C1 — تکمیل candidate بازتاب افقی

candidate قفل‌شده با `horizontal_flip_prob=0.5` بدون تغییر recipe، epoch، cache یا
pooling تکمیل شد. checkpoint ثابت epoch 15 آن:

- مسیر: `/workspace/IAAA_BrainCtTriage_mls_reflection/models/checkpoints/mls_multitask/mls-reflection-reflect-fold1-seed42/mls_multitask_epoch_015.pth`
- SHA-256: `5b15d62f101b570d85c489006f8f5a6288e8c80b9a4964334e5cdb0315d8815d`
- MLflow run: `a2a76ccb710046e1b56748537db51cd2`
- training proxy در epoch 15: `study_MAE=0.890` و `study_BF1=0.850`؛ مانند C0 این
  اعداد معیار promotion نیستند.

E1 candidate در `2026-09-05T16:12:35Z` روی raw-DICOM و همان 67 study انجام شد:

- receipt: `/workspace/iaaa_artifacts/mls_reflection_r1_20260905/strict_evaluation/candidate_fold1_seed42/aggregate_summary.json`
- CUDA-only، 67/67 study، `40.37 s` و peak VRAM برابر `0.560 GiB`

| معیار مستقل | control C0 | candidate C1 | C1 − C0 |
|---|---:|---:|---:|
| MAE | 1.531936 mm | 1.224877 mm | −0.307059 mm |
| F1@3 mm | 0.754717 | 0.842105 | +0.087388 |
| F1@5 mm | 0.702703 | 0.777778 | +0.075075 |
| Boundary-F1 | 0.728710 | 0.809942 | +0.081232 |
| selection objective | 2.074517 | 1.604994 | −0.469523 |

P1 candidate در `2026-09-05T16:16:30Z` نیز passed شد:

- receipt: `/workspace/iaaa_artifacts/mls_reflection_r1_20260905/package_parity/candidate_fold1_seed42.json`
- `batch_size=8`، `atol=1e-6`، `errors=[]` و همهٔ deltaهای runtime برابر صفر.

پس بهبود C1 فقط اثر یک decoder یا پکیج متفاوت نیست؛ در E1 و runtime هم‌تراز دیده
شده است. با این حال هنوز **فقط یک fold و یک seed** است و هیچ ادعای اثر قطعی یا
leaderboard از آن ساخته نمی‌شود.

### R1 paired screen gate

در `2026-09-05T16:20:08Z` receipt زیر هر چهار non-inferiority شرط raw-MLS را
passed کرد:

`/workspace/iaaa_artifacts/mls_reflection_r1_20260905/screen_gate/control_vs_candidate.json`

این receipt به parent preregistration و هر دو E1 receipt hash-bound است و وضعیتش
`next_gate_authorized=true` است، اما صراحتاً `promotion_eligible=false` و
`submission_zip_allowed=false` باقی می‌ماند. معنای آن فقط این است که ارزش دارد
اثر C1 را با seeds مستقل بررسی کنیم؛ نه اینکه C1 مدل نهایی است.

### R1R — continuation سه-seed پس از screen موفق

میان‌بُر معتبر از R1 تک-seed به Macro-F1/Urgent-F1 وجود ندارد. ارزیاب canonical
triage به سه عضو دقیق `42, 2026, 3407` نیاز دارد؛ کپی‌کردن prediction seed42 یا
قرار دادن آن در ensemble دو بار، اعتبار آزمون را از بین می‌برد. بنابراین یک قرارداد
جدید، جدا از preregistration تاریخی R1 و بدون تغییر آن، پیش از هر نتیجهٔ replica
قفل شد:

- contract: `/workspace/iaaa_artifacts/mls_reflection_r1r_20260905/matrix/r1r_fold1_replication_contract.json`
- SHA-256: `1a5ac4ab2643150d75d27ee6ef013da68f657d17cea03b6923733c0d72095a1d`
- static receipt: `/workspace/iaaa_artifacts/mls_reflection_r1r_20260905/matrix/static_validation_receipt.json` (`passed`)
- تنها چهار train جدید مجاز: C0/C1 × seedهای `2026` و `3407`؛ seed42 هر arm از
  checkpoint و evidence موجود inherited می‌شود.
- درون هر arm تنها `seed` می‌تواند متفاوت باشد؛ میان control/candidate تنها
  `horizontal_flip_prob` متفاوت است. batch size همچنان `5`، AMP/epochs/loss/
  pooling همان R1 هستند. RTX 3090 سرعت اجرای همین recipe را افزایش می‌دهد، اما
  به‌خاطر comparability batch را تغییر نداده‌ایم.

نسخهٔ generic three-seed evaluator نیز قبل از R1R سخت‌تر شد: ستون
`median_MLS_mm` اکنون **پیش از** hash در CSV خصوصی persist می‌شود. قبلاً summary
می‌توانست به CSV فاقد median اشاره کند و canonical reducer آن را به‌درستی رد کند.
تست regression این handoff روی 3090 passed شده است؛ هیچ artifact قدیمی دستی
ویرایش نخواهد شد و هر audit تازه در output جدید ساخته می‌شود.

ترتیب اجرایی R1R ثابت است: `control/2026 → candidate/2026 → control/3407 →
candidate/3407`. فقط نخستین job پس از contract و static receipt زیر Supervisor
شروع شده؛ سه job دیگر عمداً خاموش‌اند تا GPU مشترک یا restart پنهان رخ ندهد.
بعد از شش checkpoint epoch 15، برای هر arm raw-DICOM three-seed audit انجام می‌شود
و تنها سپس canonical triage روی fold1 با frozen ICH/fracture اجرا می‌گردد. آن خروجی
صرفاً `development_oof_subset` است و حتی در صورت عبور، مجوز ZIP ندارد؛ confirmation
چند-fold مرحلهٔ بعدی خواهد بود.

برای مشاهدهٔ MLflow در R1R باید tagهای manifest یعنی `campaign_id`,
`experiment_key`, `phase`, `arm`, `fold`, `seed` فیلتر شوند. یک بدهی قدیمی در
trainer، tagهای convenience مربوط به G1 را برای cache 2.5D نیز می‌نویسد؛ آن‌ها
معیار شناسایی R1R نیستند و در این continuation برای حفظ source continuity با
seed42 تغییر داده نشده‌اند.

بنابراین در وضعیت فعلی:

- قوی‌ترین evidence MLS: C1/seed42 در raw-DICOM، با بهبود چشمگیر نزدیک مرزهای
  3 و 5 میلی‌متر؛
- هنوز Macro-F1 و Urgent-F1 نهایی محاسبه نشده‌اند؛ آن‌ها فقط پس از triage سه-seed
  و با frozen branches قابل‌تفسیر خواهند بود؛
- `promotion_eligible=false` و `submission_zip_allowed=false`.
