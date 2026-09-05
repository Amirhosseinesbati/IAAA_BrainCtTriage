# R1 — آزمون paired بازتاب افقی MLS

> وضعیت هنگام نگارش: control روی RTX 3090 در حال اجرا است؛ candidate هنوز شروع نشده است.

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
