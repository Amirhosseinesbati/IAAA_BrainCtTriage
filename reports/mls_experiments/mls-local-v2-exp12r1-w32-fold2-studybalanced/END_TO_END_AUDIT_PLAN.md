# Exp12 fold2 — برنامهٔ ازپیش‌ثبت‌شدهٔ ممیزی end-to-end

## سؤال آزمایش

آیا تغییر تنها sampler از `slice_class_balanced` به `study_class_balanced`، بدون تغییر معماری، loss، seed، augmentation یا schedule، کیفیت و پایداری full-study MLS را نسبت به Exp10 بهتر می‌کند؟

## محاسبه و ایمنی

- آموزش، forward مدل و inference فقط روی `cuda:0` مجاز است؛ CPU fallback ممنوع است.
- ممیزی هم‌زمان با آموزش اجرا نمی‌شود تا VRAM شش‌گیگابایتی oversubscribe نشود.
- raw data و predictionهای per-study/per-slice به MLflow آپلود نمی‌شوند؛ فقط metricها، config، گزارش تجمیعی و checkpointهای مجاز ثبت می‌شوند.
- هر study پس از inference اتمیک در CSV محلی ثبت می‌شود تا اجرای قطع‌شده قابل ادامه باشد.

## checkpointهای ثابت

فقط snapshotهای ازپیش‌تعریف‌شدهٔ `13, 15, 17, 19, 21, 23` وارد ممیزی می‌شوند. افزودن epoch بر اساس نتیجهٔ validation یا نتیجهٔ همان fold مجاز نیست.

## مجموعه و پروتکل

- fold ارزیابی: `2`، همان 67 مطالعهٔ Exp10.
- برای هر checkpoint: `67/67` مطالعه باید موفق باشد؛ هر failure نتیجه را نامعتبر می‌کند.
- خروجی مدل با `scripts/evaluate_mls_multitask_checkpoint.py` و batch size سازگار با VRAM تولید می‌شود.
- همان grid مشترک `6048` پروفایل aggregation آزمایش‌های Exp08/09/10 استفاده می‌شود؛ فضای جست‌وجوی تازه مخصوص Exp12 ساخته نمی‌شود.

## مقایسه‌های اصلی

1. مقایسهٔ preregistered و اصلی: Exp12/epoch15 در برابر Exp10/epoch15 روی profile production ثابت:
   `severity_window(size=3, gate=0.5, min_active=3, q=0.75, probability_weighted=true, guard=0)`.
2. مقایسهٔ robustness: MAE، Boundary-F1 و objective هر شش epoch روی همان common grid.
3. مقایسهٔ proxy↔E2E: آیا برندهٔ validation داخلی به full-study منتقل می‌شود یا mismatch قبلی تکرار می‌شود؟
4. پس از نتایج fold2، گسترش sampler به fold0/1 فقط اگر بهبود material باشد و Boundary-F1/بدترین‌حالت قربانی نشوند.

## قانون تصمیم

- پیروزی صرفاً in-fold یا صرفاً slice-level برای ادامه کافی نیست.
- اگر epoch15 روی profile ثابت MAE را بهبود دهد و Boundary-F1 افت معنادار نکند، sampler کاندیدای گسترش controlled به foldهای دیگر است.
- اگر فقط MAE بهتر ولی Boundary-F1/بدترین‌حالت ضعیف‌تر شود، نتیجه به‌عنوان trade-off ثبت می‌شود و sampler جای baseline را نمی‌گیرد.
- اگر activation مطالعه رخ ندهد یا کیفیت E2E از Exp10 بدتر باشد، علت با exposure توزیع class/study و selector calibration تحلیل می‌شود؛ آموزش blind یا tuning روی held-out fold انجام نمی‌شود.

