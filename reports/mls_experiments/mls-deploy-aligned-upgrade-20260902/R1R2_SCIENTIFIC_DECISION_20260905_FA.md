# تصمیم علمی R1R2 برای بهبود MLS — 2026-09-05

## تصمیم اجرایی

مسیر فعلی `heatmap/keypoint + spacing DICOM + selector/aggregation + loss حساس به
مرز` حفظ می‌شود. نخست replication کوتاه و pairedِ horizontal reflection در
`R1R2` کامل می‌شود و هر checkpoint بهتر، بدون انتظار برای promotion نهایی، به‌صورت
artifact موقت به `checkpoint/mls` منتقل خواهد شد. این تصمیم هم علمی است و هم با
سیاست artifact-first مسابقه سازگار است.

## شواهدی که این تصمیم را پشتیبانی می‌کنند

1. [Yan و همکاران، 2022، Diagnostics](https://pmc.ncbi.nlm.nih.gov/articles/PMC8947005/)
   از keypoint detection و augmentationهای sagittal flip/affine برای MLS استفاده
   کرده‌اند. استفادهٔ آن‌ها از وزن pretrained برای ما مجاز فرض نمی‌شود، اما خود
   augmentation روی دادهٔ رسمی کاملاً سازگار است. بنابراین اثر مثبت C1 با
   `horizontal_flip_prob=0.5` مبنای علمی دارد، هرچند هنوز تک-fold/تک-seed است.
2. [Nguyen و همکاران، ICCVW 2021](https://openaccess.thecvf.com/content/ICCV2021W/MIA-COV19D/html/Nguyen_Brain_Midline_Shift_Detection_and_Quantification_by_a_Cascaded_Deep_ICCVW_2021_paper.html)
   pipeline cascadingِ localisation، heatmap سه landmark و fusion چند slice را
   گزارش می‌کنند؛ این با معماری فعلی هم‌راستا است و دلیل مثبتی برای جایگزینی فوری
   آن با معماری نامرتبط نمی‌دهد.
3. [Wei و همکاران، 2020](https://pubmed.ncbi.nlm.nih.gov/32471017/) نشان می‌دهند
   در deformation شدید، خط مرجع می‌تواند از symmetry/landmark خام پایدارتر باشد.
   این فقط یک فرضیهٔ بعدی است: اگر error audit نشان دهد endpointهای falx منشأ خطاهای
   نزدیک 3/5mm هستند، از همان annotation رسمی یک distance-map/line auxiliary
   می‌سازیم.

## ترتیب تصمیم‌ها

1. **اکنون:** R1R2 control/candidate برای seedهای 2026 و 3407 با epoch ثابت 15.
   هر candidate که raw-DICOM held-out، parity و gateهای سبک را پاس کند، فوراً با
   model card محلی ذخیره می‌شود.
2. **پس از replication:** three-seed audit برای تشخیص این‌که اثر reflection پایدار
   است یا فقط شانس fold/seed بوده است. triage/Macro-F1 فقط برای promotion و تحلیل
   است، نه شرط نگهداری artifact.
3. **فقط در صورت سازگاری جهت اثر:** ablation کوچک GPU-only برای Flip-TTA، با
   برگشت دقیق مختصات keypoint به فضای اصلی و پذیرش فقط در صورت non-inferiority
   MAE و Boundary-F1. این آزمایش با threshold/calibration مخلوط نمی‌شود.
4. **فقط در صورت خطای هندسی اثبات‌شده:** auxiliary ideal-line از دو endpoint falx.

## No-Go فعلی

- diffusion یا 3D سنگین: به داده/وزن/هزینه‌ای نامتناسب با 320 مطالعهٔ رسمی نیاز
  دارد و ارزش مورد انتظارش از replication کنترل‌شده کمتر است.
- دیتاست یا وزن خارجی: تا مجوز کتبی برگزارکننده دریافت نشود وارد آموزش یا
  submission نمی‌شود.
- تغییر هم‌زمان head/loss/aggregation/TTA/threshold: تشخیص علت بهبود را ناممکن
  می‌کند و با هدف artifact-first سازگار نیست.

## وضعیت آماده‌سازی Flip-TTA — 2026-09-05

پیاده‌سازی opt-in در commit `af0cb6d` موجود است، ولی **در R1R2 فعال نیست** و
remote checkout sealedِ R1R2 نیز عمداً به آن pull نمی‌شود. این implementation در
هر batch دو forward دارد؛ heatmapهای هر view جداگانه spatial-softmax می‌شوند، view
reflected روی محور x به فضای اصلی برگردانده می‌شود و آنگاه probabilityها میانگین
می‌شوند. probabilityهای selector/peak نیز جداگانه میانگین می‌شوند. بنابراین نه
مختصات خام و نه logits نامتقارن میانگین نمی‌شوند. test contract نوشته شده اما به
علت سیاست CUDA-only، اجرای آن و هر inference مربوطه فقط در validation session بعدی
روی GPU انجام خواهد شد.

## یادداشت ابزار پژوهش

SciSpace دوباره با درخواست طبیعی فراخوانی شد، اما backend آن `Unknown tool` بازگرداند.
Consensus جست‌وجو و fetch paperهای منتخب را با موفقیت انجام داد؛ نتیجهٔ بالا فقط
بر شواهد paperهای قابل‌بازبینی و لینک‌شده بنا شده است.

## یادداشت پایداری artifact روی Vast — 2026-09-05

بررسی live server با `vast-capabilities` نشان داد `workspace_is_volume=false` است.
یعنی `/workspace` با recycle یا destroy شدن instance پایدار نیست، هرچند stop/start
عادی آن را حفظ می‌کند. بنابراین policy عملی artifact-first سخت‌تر می‌شود: هر
checkpoint candidate که ارزیابی raw-DICOM مستقل، package parity و gate سبکِ همان
seed را بگذراند، همان روز به
`D:\Projects\My projects\IAAA_Compet\IAAA_BrainCtTriage\checkpoint\mls`
منتقل و SHA-256 محلی/remote تطبیق داده می‌شود. تکمیل سه-seed فقط برای تصمیم
promotion و نه برای نگه‌داشتن یا تحویل candidate به عامل ensemble است.
