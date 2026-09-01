# گزارش عمیق یک‌روزهٔ توسعه، آموزش و ارزیابی مدل MLS

**پروژه:** IAAA Brain CT Triage  
**تاریخ ثبت:** ۱۴۰۵/۰۶/۰۶ — 2026-08-28  
**وضعیت سند:** زنده و به‌روز؛ آموزش، MLflow و ممیزی GPU-only آزمایش‌های 08، 09 و 10 روی هر سه fold کامل شده است. تحلیل Exp11 بدون sweep آموزشی تازه انجام و گزینه‌های ناموفق حذف شده‌اند. آموزش 23 ایپاکی و ممیزی `402/402` inference آزمایش Exp12 نیز کامل شد؛ sampler کاملاً study-balanced معیار اصلی E2E را شکست نداد و رد شد. checkpoint ثابت epoch15 همچنان baseline production است.  
**محدوده:** تسک Midline Shift (MLS)، از بازبینی پیاده‌سازی تا آموزش، ارزیابی end-to-end، تحلیل cross-fold، ensemble و طراحی آزمایش بعدی.

---

## 1. خلاصهٔ مدیریتی

در این چرخه فقط یک مدل تازه آموزش داده نشد؛ زنجیرهٔ ارزیابی MLS نیز بازبینی و اصلاح شد تا معلوم شود کدام بهبود واقعاً به رفتار مدل روی یک مطالعهٔ کامل CT منتقل می‌شود.

### پاسخ مستقیم: چرا تعداد آزمایش‌ها زیاد به نظر می‌رسد؟

تعداد نام‌های Exp زیاد است، اما تعداد آموزش‌های مستقل و تغییرات معماری زیاد نیست. بخش عمدهٔ کار «تکرار یک مدل روی سه fold برای اثبات تعمیم»، «ارزیابی checkpointهای ذخیره‌شده» و «تحلیل predictionهای موجود» بوده است. اگر فقط fold0 یا بهترین validation داخلی را قبول می‌کردیم، به نتیجه‌ای خوش‌بینانه و ناپایدار می‌رسیدیم؛ fold2 و strict leave-one-fold-out دقیقاً نشان دادند انتخاب آزاد checkpoint می‌تواند MAE را تا `2.6559 mm` خراب کند.

| مرحله | واقعاً چه کاری بود؟ | چرا لازم بود؟ | نتیجه/تصمیم |
|---|---|---|---|
| Exp08 | یک آموزش hybrid روی fold0 | آزمون اولیهٔ selector جدید و snapshot schedule | موفق؛ مجوز انتقال گرفت |
| Exp09 | همان config روی fold1 | سنجش تکرارپذیری، نه معماری تازه | موفق؛ epoch15 در E2E قوی شد |
| Exp10 | همان config روی fold2 | آزمون robustness روی سخت‌ترین fold | موفق؛ معماری سالم، variance checkpoint آشکار شد |
| Exp11 | عمدتاً تحلیل predictionهای ذخیره‌شده؛ آموزش تازه نبود | انتخاب strict، snapshot blend، calibration و SWA | checkpoint averaging رد شد؛ offset فقط boundary-oriented؛ epoch15 قفل شد |
| Exp12 | یک آموزش تک‌متغیره روی fold2؛ فقط sampler عوض شده | رفع bias اندازهٔ مطالعه که exposure را تا `7.2–7.9×` نامتوازن می‌کرد | آموزش و E2E کامل؛ epoch15 production بدتر شد و sampler کامل رد شد |

بنابراین چند آزمایش هم‌زمان اجرا نشده است؛ Exp12 تنها **یک run روی یک GPU** بود و آموزش و ممیزی آن اکنون کامل است. این run از روی کنجکاوی یا sweep بی‌هدف نبود: ممیزی داده نشان داد sampler قبلی به studyهای پربرش چند برابر وزن می‌دهد. نتیجهٔ منفی نیز اطلاعات روشنی داد: حذف کامل این bias، calibration مقدار MLS و profile production را خراب می‌کند؛ بنابراین شاخه بدون آموزش دو fold دیگر بسته شد.

### چرا Exp12 پیش از epoch23 قطع نشد؟

- در MLS، metric مطالعه تا قبل از عبور selector از gate معمولاً ثابت و بی‌اطلاع است. در Exp10 این گذار در epoch8 و بلوغ آن در epoch9 رخ داد؛ قضاوت در epochهای 1–7 عملاً فقط warm-up را اندازه می‌گیرد.
- Exp12 در epoch1 localization ضعیف داشت، اما در epoch2 بیش از `96%` recovery کرد و در epoch4 چهار شاخص validation مکانی/ranking هم‌سطح یا بهتر از Exp10 شد. پس شواهدی از خرابی مدل وجود ندارد.
- اولین snapshot ازپیش‌ثبت‌شده epoch13 است. اگر مدل در epoch8–9 فعال و سالم باشد، توقف پیش از 13 آزمایش را بدون checkpoint قابل ممیزی رها می‌کند و GPU مصرف‌شده را هدر می‌دهد.
- checkpointهای Exp10 محفوظ‌اند و Exp12 تا وقتی E2E بهتر نشود جای آن‌ها را نمی‌گیرد؛ ادامهٔ run ریسک از دست‌دادن baseline ندارد.

### قانون توقف و ادامهٔ دقیق

1. اگر تا پایان epoch9 study pooling فعال نشود، یا NaN/OOM/fallback رخ دهد، run متوقف و علت تحلیل می‌شود.
2. اگر activation سالم باشد، آموزش فقط تا schedule ازپیش‌تعریف‌شدهٔ epoch23 ادامه می‌یابد؛ ادامهٔ blind بعد از 23 ممنوع است.
3. پس از پایان، فقط snapshotهای `13/15/17/19/21/23` روی GPU و 67 مطالعهٔ fold2 ممیزی می‌شوند؛ snapshot دلخواه اضافه نمی‌شود.
4. معیار اصلی، Exp12/epoch15 در برابر Exp10/epoch15 روی profile production قفل‌شده است. بهبود validation داخلی یا in-fold grid به‌تنهایی مجوز تغییر baseline نیست.
5. فقط اگر MAE full-study بهتر و Boundary-F1 بدون افت معنادار باشد، sampler روی fold0/1 گسترش می‌یابد؛ در غیر این صورت Exp12 رد و شاخه بسته می‌شود.

### وضعیت قطعی، وضعیت باز و برنامهٔ بعد

- **قطعی:** HRNet-W32/hybrid خراب نیست؛ epoch15 امن‌ترین checkpoint مشترک فعلی است؛ انتخاب آزاد checkpoint و training طولانی‌تر از 23 ناپایدار/بی‌فایده است؛ SWA وزنی رد شده است.
- **پاسخ Exp12:** حذف کامل bias اندازهٔ study چند proxy داخلی را بهتر کرد، ولی روی معیار اصلی epoch15/profile production، MAE را `11.72%` بدتر و Boundary-F1 را `0.0888` کم کرد؛ sampler کامل رد شد.
- **باز:** آیا یک sampler میانی می‌تواند سود representation در epoch13 را حفظ کند، بدون اینکه exposure لازم برای calibration رگرسیون MLS را از studyهای بلند بگیرد؟ این سؤال فقط با یک ضریب اختلاط ازپیش‌قفل‌شده و E2E کنترل‌شده قابل بررسی است.

مهم‌ترین یافته‌ها تا این لحظه:

1. **هدف peak-aware یک بهبود واقعی است.** در مقایسهٔ سه‌fold و با انتخاب صحیح checkpoint، میانگین MAE از `1.6343 mm` در نسخهٔ binary به `1.5806 mm` رسیده و بدترین fold نیز از `1.9442 mm` به `1.8821 mm` بهتر شده است. Boundary-F1 متوسط نیز از `0.8014` به `0.8143` افزایش یافته است.
2. **انتخاب checkpoint فقط با validation داخلی قابل اتکا نیست.** در fold 2، checkpoint برندهٔ objective داخلی در epoch 19 روی ارزیابی کامل مطالعه `2.3436 mm` MAE داشت، اما checkpoint نهایی epoch 27 با objective داخلی ضعیف‌تر به `1.9744 mm` رسید. علت اصلی، اختلاف توزیع validation برش‌محور با inference واقعی روی همهٔ برش‌های مطالعه است.
3. **ensemble مدل binary و peak-aware در نسل قبلی قوی‌ترین نتیجهٔ cross-fold بود.** ارزیابی strict leave-one-fold-out آن به میانگین `1.6350 mm` رسید؛ در برابر حدود `1.7426 mm` برای peak-aware تنها و `1.7601 mm` برای binary تنها. این نتیجه بعداً با single hybrid/epoch15 هم‌سطح یا بهتر شد و دیگر production candidate اصلی نیست.
4. **یک خطای روش‌شناختی در نسخهٔ اول جست‌وجوی ensemble کشف و قبل از تصمیم‌گیری اصلاح شد.** candidate profileها ابتدا با نگاه به هر سه fold ساخته می‌شدند و نتیجه خوش‌بینانه بود. اکنون هم ساخت candidate pool و هم انتخاب blend برای هر fold فقط با دو fold دیگر انجام می‌شود.
5. **Experiment 08 با موفقیت کامل و همهٔ snapshotها E2E ممیزی شد.** بهترین objective داخلی epoch 10 بود، اما بهترین checkpoint تشخیصی full-study epoch 21 شد؛ epoch 23 با وجود proxy داخلی قوی در E2E افت کرد. این اختلاف بار دیگر ثابت کرد checkpoint selection داخلی surrogate کافی نیست.
6. **ensemble محافظه‌کارانهٔ Exp05 و hybrid مسیر موقتِ صدور مجوز fold0 بود، نه تصمیم نهایی.** ترکیب `75% Exp05 + 25% Exp08/epoch21` در grid یکسان به نتیجهٔ متوازن `MAE=1.2369 mm`، `Boundary-F1=0.8474` و `Combined-F1=0.6392` رسید. با پروفایل frozen قبلی نیز `MAE=1.2504 mm` ثبت شد؛ حدود 5.15٪ بهتر از Exp05 با همان پروفایل و با افت Boundary-F1 فقط `0.0098`. پس از تکمیل سه fold، single hybrid/epoch15 به‌دلیل سادگی و تعمیم بهتر جای این مسیر موقت را گرفت.
7. **انتقال hybrid به fold1 با موفقیت تا انتهای برنامهٔ ازپیش‌تعریف‌شده اجرا شد.** Exp09 هر 23 ایپاک را بدون OOM، NaN یا fallback مدل به CPU کامل کرد. بهترین objective متوازن در ایپاک 17 (`1.375` با `MAE=0.950 mm` و `Boundary-F1=0.812`)، بهترین MAE داخلی در ایپاک 21 (`0.941 mm`) و بهترین Boundary-F1 در ایپاک 13 (`0.825`) به دست آمد. ایپاک 23 با `MAE=1.101 mm` و `Boundary-F1=0.772` رکورد تازه‌ای نساخت. snapshotهای مستقل 13، 15، 17، 19، 21 و 23، هرکدام با حجم `124,890,565` بایت، و checkpoint نهایی روی دیسک تأیید شدند. run با وضعیت `FINISHED` و artifactهای مورد انتظار در MLflow بسته شد. این نتایج هنوز proxy داخلی‌اند و تا E2E کامل نباید با نتیجهٔ fold0 یا leaderboard هم‌ارز فرض شوند.
8. **ممیزی کامل fold1 بهبود واقعی hybrid را تأیید کرد.** هر `402/402` ارزیابی study-checkpoint روی CUDA و بدون failure تمام شد. روی grid یکسان، epoch15 با `MAE=1.2722`، `Boundary-F1=0.8635` و objective=`1.5453` از baseline Exp06 با `1.3301`، `0.8035` و `1.7231` بهتر بود. مهم‌تر، روی پروفایل کاملاً frozen از fold0، MAE از `1.6206` به `1.4812` و Boundary-F1 از `0.7340` به `0.7902` رسید؛ یعنی 8.60٪ بهبود MAE بدون tuning مخصوص fold1. blend محافظه‌کارانهٔ `75% Exp06 + 25% epoch21` همین پروفایل frozen را به `MAE=1.4096` رساند. شرط ادامه به fold2 پاس شده است.
9. **Exp10/fold2 نیز کامل و سالم اجرا شد.** هر 23 ایپاک بدون OOM، NaN یا CPU fallback تمام شد، MLflow run با وضعیت `FINISHED` بسته شد و ممیزی شش snapshot روی `402/402` study-checkpoint موفق بود. در grid مخصوص fold2، epoch17 به `MAE=1.5998` و `Boundary-F1=0.9412` رسید.
10. **انتخاب آزاد snapshot ناپایدار، اما معماری hybrid سالم است.** در strict LOO، انتخاب هم‌زمان epoch و profile می‌توانست به‌علت انتخاب epoch19 روی fold2 تا `2.6559 mm` سقوط کند. وقتی epoch پیشاپیش روی 15 ثابت شد، strict balanced به `Mean MAE=1.6273`، `Worst=1.7999` و `Boundary-F1=0.8208` رسید؛ سیاست boundary-first نیز `1.6045 mm` و `0.8344` ثبت کرد.
11. **کاندیدای single فعلی از نسل قبلی عبور کرده است.** نتیجهٔ diagnostic epoch15 hybrid برابر `Mean MAE=1.5459`، `Worst=1.7144` و `Boundary-F1=0.8469` است. strict fixed-epoch آن با ensemble دو-مدلی binary+peak-aware (`1.6350 mm`) هم‌سطح یا بهتر است، با Boundary-F1 بالاتر و فقط یک inference.

نتیجهٔ عملی فعلی: معماری موجود شکست‌خورده یا بی‌ارزش نیست؛ برعکس، پایهٔ HRNet-W32 سالم و رقابتی است. گلوگاه اصلی فعلی بیشتر در **تعریف target selector، تجمیع برش‌ها در سطح مطالعه و انتخاب checkpoint** است تا صرفاً بزرگ‌ترکردن backbone.

---

## 2. محدودیت‌های اجرایی و قواعدی که رعایت شده‌اند

- تمام forward، backward و inference مدل فقط روی GPU/CUDA انجام شده و می‌شود.
- CPU فقط برای کارهای سبک و اجتناب‌ناپذیر مثل خواندن فایل، DataLoader، تجمیع CSV، محاسبهٔ آمار ساده و نوشتن گزارش استفاده شده است؛ هیچ آموزش یا inference مدل روی CPU انجام نشده است.
- سخت‌افزار محلی: NVIDIA GTX 1660 Ti Max-Q با 6GB VRAM.
- پروفایل فعلی HRNet-W32 با batch size برابر 5، بدون AMP، حدود `1.78–1.83 it/s` سرعت و تقریباً `4.55 GB` مصرف اوج VRAM دارد.
- دادهٔ خام پزشکی، DICOM یا PNG به MLflow ارسال نشده است. فقط config، metric، گزارش، سورس مرتبط و checkpointهای مجاز ارسال می‌شوند.
- snapshotهای پرتعداد Experiment 08 عمداً محلی هستند و خودکار به MLflow فرستاده نمی‌شوند تا پهنای‌باند و فضای artifact بیهوده مصرف نشود. checkpoint منتخب پس از ارزیابی مشخص خواهد شد.

---

## 3. چه کارهایی در این چرخه انجام شد؟

### 3.1 بازبینی نتایج و baselineهای موجود

ابتدا checkpointهای جدید هر تسک و به‌خصوص MLS از نو بررسی شدند. برای MLS روشن شد که metricهای slice-level و validation داخلی به‌تنهایی پاسخ سؤال مسابقه را نمی‌دهند؛ چون خروجی نهایی باید از کل سری CT و با یک pooling study-level ساخته شود.

baselineهای اصلی MLS:

| آزمایش | Fold | Target selector | نکتهٔ اصلی |
|---|---:|---|---|
| Exp01 | 0 | binary، HRNet-W18 | baseline سبک؛ E2E MAE حدود `1.909 mm` |
| Exp02 | 0 | binary، HRNet-W32 | slice MAE=`1.839`، AUC=`0.913` |
| Exp03 | 1 | binary، HRNet-W32 | slice MAE=`1.877`، AUC=`0.905` |
| Exp04 | 2 | binary، HRNet-W32 | fold سخت‌تر؛ best-AUC E2E حدود `2.104 mm` |

این مرحله نشان داد W32 نسبت به W18 ظرفیت مفیدتری دارد، ولی فاصلهٔ اصلی دیگر از backbone نمی‌آید و باید study-level behavior اصلاح شود.

### 3.2 پیاده‌سازی selector هدف peak-aware

در مسئلهٔ MLS، برچسب binary ساده فقط می‌گوید یک برش مثبت است یا نه؛ اما همهٔ برش‌های مثبت برای تخمین بیشینهٔ جابه‌جایی اهمیت یکسان ندارند. بنابراین target نرم و peak-aware اضافه شد تا selector به برش‌های نزدیک به بیشینهٔ MLS وزن بیشتری بدهد.

تغییرات اصلی:

- اضافه‌شدن حالت‌های selector target و پارامترهای peak-aware به config؛
- ساخت label نرم بر اساس شدت نسبی MLS هر برش نسبت به peak همان مطالعه؛
- اضافه‌شدن selector peak AUC، study proxy MAE، F1 و Boundary-F1 به پایش آموزش؛
- اصلاح معنای checkpointهای `best_study` و `best_study_boundary`؛
- اضافه‌شدن poolingهای مقاوم مانند relative component، severity window، anchor window، smooth/joint component؛
- افزودن weighted aggregation و heatmap guard؛
- گسترش grid جست‌وجوی cross-fold.

### 3.3 آموزش سه fold مدل peak-aware

| آزمایش | Fold | checkpoint منتخب نهایی | مهم‌ترین نتیجه |
|---|---:|---|---|
| Exp05 | 0 | best objective، epoch 13 | E2E robust MAE=`1.3360 mm` در پروفایل مشترک سه‌fold |
| Exp06 | 1 | best objective، epoch 14 | E2E robust MAE=`1.5237 mm` در پروفایل مشترک سه‌fold |
| Exp07 | 2 | **final، epoch 27** | E2E robust MAE=`1.8821 mm`؛ بهتر از best-objective داخلی |

در Exp07 بنا به تصمیم آگاهانه آموزش بعد از یک افت کوتاه فوراً قطع نشد. epoch 19 در validation داخلی جهش خوبی داشت، اما ادامه تا epoch 27 نشان داد checkpoint نهایی در inference کامل مطالعه بهتر است. این مشاهده مستقیماً باعث طراحی snapshotهای منظم در Exp08 شد.

### 3.4 ارزیابی کامل end-to-end و کشف mismatch

برای هر checkpoint مهم، تمام برش‌های validation fold روی GPU inference شدند؛ سپس فقط predictionهای ذخیره‌شده با CPU سبک تجمیع و تحلیل شدند.

نمونهٔ روشن mismatch در Exp07/fold2:

| Checkpoint | وضعیت validation داخلی | E2E MAE با پروفایل frozen |
|---|---|---:|
| epoch 19 / best objective | بهترین objective داخلی (`1.650`) | `2.3436 mm` |
| epoch 21 / boundary | Boundary-F1 داخلی قوی | `2.2340 mm` |
| epoch 27 / final | objective داخلی ضعیف‌تر (`1.820`) | **`1.9744 mm`** |

برداشت فنی: validation فعلی عمدتاً روی نمونه‌های پردازش‌شده/annotated و توزیع برش متفاوت انجام می‌شود، در حالی که inference مسابقه باید کل مطالعه را ببیند. بنابراین ranking checkpointها می‌تواند جابه‌جا شود. از این پس checkpoint selection نهایی باید دست‌کم برای چند snapshot منتخب با full-study CUDA audit انجام شود.

### 3.5 تحلیل سه‌fold pooling

#### مدل binary

بهترین پروفایل diagnostic روی هر سه fold:

- family=`severity_window`
- size=`3`
- selector gate=`0.9`
- min active slices=`3`
- quantile=`0.75`
- probability weighted=`true`

نتایج:

| معیار | مقدار |
|---|---:|
| Mean MAE | `1.6343 mm` |
| Worst-fold MAE | `1.9442 mm` |
| Mean Boundary-F1 | `0.8014` |

نتیجهٔ strict leave-one-fold-out:

- fold0: `1.5039 mm`
- fold1: `1.6726 mm`
- fold2: `2.1037 mm`
- میانگین: `1.7601 mm`

#### مدل peak-aware با checkpoint درست fold2

بهترین پروفایل diagnostic مقاوم:

- family=`relative_component`
- component ratio=`0.3`
- selector gate=`0.6`
- min active slices=`3`
- quantile=`0.75`
- probability weighted=`true`

نتایج:

| معیار | مقدار |
|---|---:|
| Fold0 MAE | `1.3360 mm` |
| Fold1 MAE | `1.5237 mm` |
| Fold2 MAE | `1.8821 mm` |
| Mean MAE | **`1.5806 mm`** |
| Worst-fold MAE | **`1.8821 mm`** |
| Mean Boundary-F1 | **`0.8143`** |

نتیجهٔ strict leave-one-fold-out:

- fold0: `1.6655 mm`
- fold1: `1.5881 mm`
- fold2: `1.9744 mm`
- میانگین: `1.7426 mm`

پس peak-aware در diagnostic و strict LOO نسبت به binary برتری دارد، هرچند اندازهٔ بهبود strict LOO کوچک‌تر است و نباید آن را بیش‌ازحد تفسیر کرد.

### 3.6 بررسی مکمل‌بودن خطاها و ensemble

هم‌بستگی خطای مطلق binary و peak-aware:

- fold0: `0.650`
- fold1: `0.603`
- fold2: `0.897`

در fold0 و fold1 اختلاف الگوی خطا برای ensemble مفید است؛ fold2 سخت‌تر و خطاها هم‌بسته‌ترند.

بهترین نتیجهٔ diagnostic مشترک با وزن peak برابر `0.75`:

| Fold | MAE |
|---|---:|
| 0 | `1.2584 mm` |
| 1 | `1.4625 mm` |
| 2 | `1.8083 mm` |
| Mean | **`1.5097 mm`** |
| Worst | **`1.8083 mm`** |
| Mean Boundary-F1 | **`0.8186`** |

این عدد diagnostic است و انتخاب روی هر سه fold انجام شده؛ بنابراین معیار قابل‌اتکاتر strict LOO است:

| Fold نگه‌داشته‌شده | MAE | Boundary-F1 |
|---|---:|---:|
| fold0 | `1.4990 mm` | `0.7935` |
| fold1 | `1.4887 mm` | `0.7361` |
| fold2 | `1.9174 mm` | `0.8500` |
| Mean | **`1.6350 mm`** | — |

ensemble در strict LOO حدود 6.2٪ نسبت به peak-aware تنها و حدود 7.1٪ نسبت به binary تنها MAE را کاهش می‌دهد. هزینهٔ آن تقریباً دو inference است؛ پس ورود آن به submission نهایی مشروط به محدودیت زمان اجرا و سود واقعی leaderboard خواهد بود.

### 3.7 اصلاح نشت در جست‌وجوی ensemble

نسخهٔ اولیهٔ ابزار ensemble، profile candidateها را با اطلاعات همهٔ foldها می‌ساخت؛ هرچند blend برای fold نگه‌داشته‌شده جدا می‌شد، candidate pool هنوز به آن fold نگاه کرده بود. این یک leakage ظریف و خوش‌بینانه بود.

اصلاح انجام‌شده:

- برای هر held-out fold، candidate profileهای binary و peak-aware فقط از دو fold دیگر انتخاب می‌شوند؛
- وزن blend نیز فقط روی همان دو fold انتخاب می‌شود؛
- سپس specification ثابت روی held-out fold اعمال می‌شود.

اعداد strict LOO این سند از نسخهٔ اصلاح‌شده هستند.

---

## 4. تغییرات مهم کد و artifactها

فایل‌های کلیدی تغییرکرده یا افزوده‌شده:

- `src/strategies/config_models.py`
  - پارامترهای peak-aware، pooling و snapshot schedule.
- `src/strategies/mls_heatmap/train_multitask.py`
  - soft target، metricهای study-level، checkpoint variants و snapshotهای دوره‌ای.
- dataset مربوط به MLS
  - تعریف MLS رسمی مطالعه به‌صورت بیشینهٔ metadata برش‌ها و انتقال اطلاعات لازم به trainer.
- predictor/pooling مربوط به MLS
  - تجمیع مقاوم، component filtering، weighted quantile و guard.
- `scripts/search_mls_crossfold_pooling.py`
  - جست‌وجوی وسیع اما سبک روی CSV predictionها.
- `scripts/search_mls_checkpoint_pooling.py`
  - مقایسهٔ checkpointها با پروفایل‌های frozen و diagnostic.
- `scripts/blend_mls_slice_predictions.py`
  - ترکیب reproducible خروجی کامل برش‌ها با mean/median و اعتبارسنجی سخت‌گیرانهٔ study، ground truth و slice index؛ بدون model inference.
- `scripts/search_mls_crossfold_ensemble.py`
  - ensemble binary/peak-aware با strict LOO بدون leakage.
- `config/experiments/mls-local-v2-exp08-w32-fold0-hybridsoft-snapshots.yaml`
  - manifest آزمایش فعال.

اعتبارسنجی فنی اخیر:

- فایل‌های تغییرکرده با `py_compile` بررسی شده‌اند.
- manifest واقعی Exp08 از مسیر validation خود strategy عبور کرده است.
- اجرای full test suite در این لحظه انجام نشده، چون بخشی از تست‌های integration ممکن است model execution روی CPU ایجاد کنند و با قاعدهٔ صریح این پروژه ناسازگار است. تست‌های مدل‌محور باید بعداً با guard اجباری CUDA اجرا شوند.

---

## 5. وضعیت MLflow و قابلیت بازتولید

| آزمایش | MLflow Run ID |
|---|---|
| Exp01 W18 | `8fee771402924977bcfdc6e028c6625e` |
| Exp02 W32 fold0 binary | `695ef10c0a1f4cd19f12135eb3e2974` |
| Exp03 W32 fold1 binary | `df4717b978054f4b9874b0121fef579e` |
| Exp04 W32 fold2 binary | `c78a77479f3e4492be0ed9cb54e707e2` |
| Exp05 W32 fold0 peak-aware | `532d62f07a84421681c8f199ccba462d` |
| Exp06 W32 fold1 peak-aware | `4646e57c62e240ae8e415027ef8006e7` |
| Exp07 W32 fold2 peak-aware | `3b07a5d204b6452696ad89c3a03ec1d9` |
| Exp08 W32 fold0 hybrid-soft | `85a9cba212fa45a19ebc6f972106a802` |

لینک run تکمیل‌شدهٔ Exp08: `https://dagshub.com/amiresbati62/BrainCtTriage.mlflow/#/experiments/16/runs/85a9cba212fa45a19ebc6f972106a802`

مسیرهای نتایج تجمیعی مهم:

- `reports/mls_experiments/binary_exp02_exp03_exp04_crossfold_pooling/`
- `reports/mls_experiments/peakaware_exp05_exp06_exp07final_crossfold_pooling/`
- `reports/mls_experiments/binary_peakaware_final_crossfold_ensemble/`
- `reports/mls_experiments/mls-local-v2-exp05-w32-fold0-peakaware/`
- `reports/mls_experiments/mls-local-v2-exp06-w32-fold1-peakaware-transfer/`
- `reports/mls_experiments/mls-local-v2-exp07-w32-fold2-peakaware-crossfold/`
- `reports/mls_experiments/mls-local-v2-exp08-w32-fold0-hybridsoft-snapshots/end_to_end_checkpoint_audit/`
- `reports/mls_experiments/mls-local-v2-exp08-w32-fold0-hybridsoft-snapshots/checkpoint_pooling_expanded/`
- `reports/mls_experiments/mls-local-v2-exp08-w32-fold0-hybridsoft-snapshots/snapshot_blend_pooling_expanded/`

نکتهٔ مهم: prediction CSVها شامل خروجی مدل برای ارزیابی و metadata لازم هستند؛ دادهٔ خام تصویر به MLflow فرستاده نشده است.

---

## 6. Experiment 08: آموزش، ممیزی snapshot و نتیجهٔ ensemble

هدف این آزمایش پاسخ به یک سؤال مشخص است: آیا target کاملاً peak-aware بخشی از robustness مدل binary را از دست می‌دهد، و آیا یک target ترکیبی با پایهٔ binary برابر `0.75` می‌تواند calibration و localization را هم‌زمان بهتر کند؟

مشخصات:

- Backbone: HRNet-W32
- Fold: 0
- Epochs: 30
- Early-stop patience: 30، یعنی آموزش عملاً تا epoch 30 ادامه می‌یابد.
- Selector target: `peak_aware_soft`
- Peak base: `0.75`
- Peak power: `1.0`
- Snapshot: از epoch 13، هر دو epoch (`13, 15, ..., 29`)
- اجرای مدل: فقط CUDA

روند validation داخلی تا پایان epoch 12:

| Epoch | Slice MAE | Study proxy MAE | Boundary-F1 | Keypoint px | Selector AUC | Selector F1 | Objective |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 2.450 | 3.344 | 0.269 | 9.54 | 0.868 | 0.729 | 4.871 |
| 9 | 2.272 | 1.249 | 0.810 | 9.50 | 0.884 | 0.788 | 1.687 |
| 10 | 2.153 | **1.100** | 0.811 | 9.40 | 0.908 | 0.837 | **1.524** |
| 11 | 2.245 | 2.056 | 0.666 | 9.40 | 0.893 | 0.579 | 2.778 |
| 12 | **2.044** | 1.386 | 0.774 | 9.30 | **0.912** | **0.852** | 1.882 |
| 13 | 2.144 | 1.181 | **0.820** | **8.80** | 0.910 | 0.789 | 1.585 |
| 14 | 2.439 | 1.270 | 0.781 | 9.20 | **0.912** | 0.824 | 1.752 |
| 15 | 2.120 | 1.664 | 0.789 | 9.20 | 0.909 | 0.664 | 2.131 |
| 16 | 2.186 | 1.403 | 0.804 | 9.10 | 0.906 | 0.844 | 1.842 |
| 17 | 2.280 | 2.484 | 0.474 | 9.20 | 0.887 | 0.425 | 3.592 |
| 18 | 2.070 | 1.475 | 0.754 | 9.00 | 0.895 | 0.737 | 2.019 |
| 19 | 2.171 | **1.143** | 0.817 | 9.00 | 0.891 | 0.728 | 1.563 |
| 20 | 2.163 | 1.207 | 0.787 | 9.10 | 0.902 | 0.791 | 1.682 |
| 21 | **1.993** | 1.274 | **0.821** | **8.70** | 0.894 | 0.795 | 1.685 |
| 22 | 2.031 | 1.256 | 0.747 | 8.90 | 0.898 | 0.730 | 1.812 |
| 23 | **1.859** | **1.118** | 0.807 | **8.50** | 0.883 | 0.765 | 1.562 |
| 24 | 2.107 | 1.365 | 0.790 | 8.80 | 0.872 | 0.640 | 1.849 |
| 25 | 2.034 | 1.260 | 0.794 | **8.40** | 0.889 | 0.736 | 1.726 |
| 26 | 2.101 | 1.299 | 0.796 | 8.60 | 0.877 | 0.692 | 1.768 |
| 27 | 2.131 | 1.250 | 0.783 | 8.60 | 0.882 | 0.735 | 1.744 |
| 28 | 2.083 | 1.278 | 0.778 | **8.40** | 0.882 | 0.732 | 1.781 |
| 29 | 2.079 | 1.262 | 0.803 | 8.50 | 0.883 | 0.738 | 1.714 |
| 30 | 2.114 | 1.261 | 0.785 | **8.40** | 0.879 | 0.725 | 1.751 |

تفسیر موقت:

- شبکه بعد از فعال‌شدن gate در epoch 8 وارد ناحیهٔ یادگیری مفید study-level شده است.
- epoch 10 بهترین objective فعلی است، ولی با تجربهٔ fold2 حق نداریم آن را بهترین checkpoint واقعی فرض کنیم.
- AUC بهتر epoch 12 همراه با study MAE ضعیف‌تر از epoch 10 نشان می‌دهد ranking و calibration/pooling یکسان نیستند.
- epoch 13 کامل شد و نشان داد افت‌وخیز objective لزوماً روند یادگیری landmark را متوقف نکرده است: keypoint error به `8.8 px` و Boundary-F1 به `0.820` رسید.
- اولین snapshot زمان‌بندی‌شده با موفقیت در مسیر `models/checkpoints/mls_multitask/mls-local-v2-exp08-w32-fold0-hybridsoft-snapshots/mls_multitask_epoch_013.pth` ذخیره و وجود آن روی دیسک تأیید شد.
- epoch 14 نسبت به epoch 13 در MAE و Boundary-F1 افت کرد، با این حال AUC در `0.912` ماند؛ این یک شاهد دیگر برای جداکردن ranking از calibration/pooling است.
- epoch 15 نیز objective داخلی را بهتر نکرد. با این حال snapshotهای قبلی محفوظ‌اند و تجربهٔ Exp07 نشان داده است توقف بر پایهٔ دو epoch validation داخلی می‌تواند checkpoint بهتر E2E را از دست بدهد؛ بنابراین اجرای ازپیش‌تعریف‌شدهٔ 30 epoch ادامه دارد.
- epoch 16 نسبت به epoch 15 بازیابی شد و Boundary-F1 از `0.789` به `0.804` رسید، اما هنوز رکورد epoch 10/13 را نشکست.
- epoch 17 در پروکسی داخلی افت شدید داشت. این snapshot در اولویت پایین E2E قرار می‌گیرد، ولی برای اندازه‌گیری mismatch نگه داشته شده است.
- epoch 18 بخش زیادی از افت epoch 17 را جبران کرد، ولی رکورد تازه‌ای نساخت؛ بنابراین افت epoch 17 یک فروپاشی پایدار نبوده است.
- epoch 19 به `study MAE=1.143` و objective=`1.563` بازگشت و تقریباً با بهترین epochهای 10 و 13 هم‌سطح شد. این نتیجه با شواهد مستقیم نشان می‌دهد ادامه‌دادن بعد از دو epoch ضعیف تصمیم درستی بوده است.
- وجود snapshotهای `013`، `015` و `017` روی دیسک و یکسان‌بودن اندازهٔ مورد انتظار آن‌ها تأیید شد.
- snapshot epoch 19 یکی از نامزدهای اصلی full-study E2E است.
- epoch 20 نیز study MAE پایین `1.207` را حفظ کرد؛ بنابراین بازیابی epoch 19 فقط یک نوسان تک‌نقطه‌ای نبوده است.
- epoch 21 بهترین slice MAE، keypoint error و Boundary-F1 فعلی را ثبت کرد. این snapshot در کنار epoch 19 نامزد اصلی E2E است: 19 برای study proxy و 21 برای localization/boundary.
- epoch 22 localization خوب را حفظ کرد، اما Boundary-F1 آن به `0.747` افت کرد؛ بنابراین فعلاً از snapshot 21 ضعیف‌تر است.
- epoch 23 بهترین slice MAE و keypoint error را ثبت کرد و با `study MAE=1.118` و objective=`1.562` قوی‌ترین کاندیدای متوازن داخلی فعلی شد؛ این snapshot حتماً در E2E ارزیابی می‌شود.
- epoch 24 نسبت به 23 افت کرد و selector-F1 آن به `0.640` رسید؛ snapshot 23 همچنان کاندیدای برتر است.
- epoch 25 رکورد keypoint error را به `8.4 px` رساند، ولی از نظر study-level از epoch 23 ضعیف‌تر بود؛ در shortlist ثانویه باقی می‌ماند.
- epoch 26 نتیجهٔ پایدار اما بدون رکورد تازه داشت. snapshot 27 مستقل از metric داخلی در shortlist دیرهنگام می‌ماند، چون در Exp07 همین ناحیه بهترین E2E را ساخته بود.
- epoch 27 از نظر پروکسی داخلی از epoch 23 ضعیف‌تر بود، اما به‌عنوان late-stage control برای E2E حفظ شد.
- epoch 28 رکورد تازه‌ای نساخت؛ E2E مستقیم آن لازم نیست چون snapshotهای 27 و 29 دو طرف آن را پوشش می‌دهند.
- epoch 29 یک checkpoint دیرهنگام متوازن ساخت، ولی در پروکسی داخلی همچنان از epoch 23 ضعیف‌تر است.
- epoch 30 نیز رکورد تازه‌ای نساخت و checkpoint نهایی از نظر پروکسی داخلی از epoch 23 ضعیف‌تر بود.
- افت موقت throughput در ابتدای epoch 25 بررسی شد: دمای GPU `70°C`، کلاک `1665 MHz` و حافظهٔ کل `5526/6144 MB` بود و سرعت خودکار به `~1.79 it/s` برگشت؛ OOM یا thermal shutdown رخ نداد.
- هر 30 epoch بدون OOM، fallback به CPU یا epoch ناقص تمام شد. همهٔ snapshotهای `013, 015, ..., 029` و checkpoint نهایی روی دیسک تأیید شدند.
- گزارش experiment و history کامل نوشته شد؛ MLflow با exit code صفر و status=`completed` پایان یافت.
- Run ID نهایی: `85a9cba212fa45a19ebc6f972106a802`.

### 6.1 ممیزی full-study همهٔ checkpointها

برای جلوگیری از cherry-picking، 12 حالت وزن ارزیابی شد: best-objective، best-selector-AUC، snapshotهای 13 تا 29 و checkpoint نهایی epoch 30. هر ارزیابی تمام 70 مطالعهٔ fold0 را روی GPU پردازش کرد. نتیجهٔ اجرایی: `840/840` study-checkpoint موفق، صفر failure، بدون OOM و بدون CPU fallback مدل.

| Checkpoint | MAE پروفایل frozen robust | Boundary-F1 robust | MAE پروفایل frozen balanced | Boundary-F1 balanced |
|---|---:|---:|---:|---:|
| best objective / epoch 10 | `1.6095` | `0.8330` | `1.5558` | `0.8069` |
| best selector AUC / epoch 14 | `1.4144` | `0.8069` | `1.4345` | `0.7718` |
| epoch 13 | **`1.3819`** | **`0.8351`** | `1.4059` | **`0.8343`** |
| epoch 15 | `1.5717` | `0.8026` | `1.8709` | `0.7574` |
| epoch 17 | `2.4117` | `0.5584` | `2.6529` | `0.4606` |
| epoch 19 | `1.4956` | `0.7936` | **`1.3916`** | `0.7872` |
| epoch 21 | `1.4006` | `0.8020` | `1.3955` | `0.7939` |
| epoch 23 | `1.7539` | `0.7539` | `1.5277` | `0.7293` |
| epoch 25 | `1.6705` | `0.7788` | `1.5363` | `0.7368` |
| epoch 27 | `1.6903` | `0.7679` | `1.6001` | `0.7568` |
| epoch 29 | `1.6607` | `0.7834` | `1.5213` | `0.7568` |
| final / epoch 30 | `1.6400` | `0.7926` | `1.5275` | `0.7781` |

برداشت‌ها:

- epoch 13 بهترین checkpoint تک‌مدلی با پروفایل frozen robust است و تقریباً به Exp05 نزدیک می‌شود، اما آن را با حاشیهٔ لازم شکست نمی‌دهد.
- epoch 17 هم در metric داخلی و هم E2E فروپاشید؛ بنابراین افت همزمان objective، selector F1 و Boundary-F1 یک علامت توقف معتبر است.
- epoch 23 نمونهٔ صریح mismatch است: بهترین کاندیدای late-training در proxy داخلی، ولی ضعیف‌تر در inference کامل مطالعه.
- ادامه تا epoch 30 برای کشف trajectory مفید بود، اما برای foldهای بعدی snapshotهای بعد از 23 ارزش اصلی ندارند؛ افت late-training پایدار است.

### 6.2 جست‌وجوی یکسان 6048 پروفایل روی 14 candidate

همهٔ 12 checkpoint hybrid، Exp05 peak-aware و Exp02 binary با grid کاملاً یکسان سنجیده شدند. این جست‌وجو هیچ model inference نداشت و فقط prediction CSVهای حاصل از CUDA را تجمیع کرد.

| Candidate | بهترین MAE diagnostic | Boundary-F1 همان پروفایل | بهترین objective متوازن |
|---|---:|---:|---:|
| Exp08 epoch 21 | **`1.2334`** | **`0.8348`** | **`1.5637`** |
| Exp08 epoch 19 | `1.2418` | `0.8139` | `1.6139` |
| Exp05 baseline | `1.2887` | `0.8115` | `1.6619` |
| Exp08 epoch 13 | `1.2955` | `0.8410` | `1.5788` |
| Exp02 binary | `1.4337` | `0.8069` | `1.8199` |

عدد epoch21 نسبت به بهترین MAE تشخیصی Exp05 حدود 4.29٪ بهتر است و Boundary-F1 نیز `+0.0234` افزایش دارد، اما پروفایل optimum تیز بود: فقط 4 پروفایل در فاصلهٔ `0.025 mm` از optimum قرار داشتند، در برابر 11 پروفایل برای Exp05. همچنین همان پروفایل روی epochهای مجاور افت زیادی داشت. بنابراین تک‌مدل epoch21 به‌تنهایی معیار robustness را پاس نکرد.

### 6.3 snapshot/cross-model ensemble بدون inference مجدد

14 ترکیب هدفمند از checkpointهای 13/19/21/23 و Exp05 ساخته شد. ترکیب در سطح خروجی هر برش انجام شد؛ study ID، ground truth و index همهٔ برش‌ها قبل از blend تطبیق داده شدند.

نتایج کلیدی:

| Ensemble | MAE متوازن | Boundary-F1 | Combined-F1 | Objective |
|---|---:|---:|---:|---:|
| `75% Exp05 + 25% epoch21` | `1.2369` | **`0.8474`** | `0.6392` | **`1.5422`** |
| `50% Exp05 + 50% epoch13` | `1.2632` | **`0.8572`** | `0.6240` | `1.5488` |
| `25% Exp05 + 75% epoch21` | **`1.2245`** | `0.8282` | `0.6145` | `1.5680` |
| `50% Exp05 + 50% epoch19` | `1.2303` | `0.8293` | `0.6210` | `1.5716` |
| median epochs 13/19/21 | `1.2760` | `0.8388` | **`0.6568`** | `1.5984` |
| Exp05 baseline | `1.3183` | `0.8282` | `0.6215` | `1.6619` |

مهم‌ترین آزمون محافظه‌کارانه، اعمال پروفایل frozen قبلی Exp05 بود:

- Exp05 تنها: `MAE=1.3183`, `Boundary-F1=0.8282`؛
- `75% Exp05 + 25% epoch21`: `MAE=1.2504`, `Boundary-F1=0.8184`؛
- بهبود MAE: حدود **5.15٪**؛ افت Boundary-F1: **0.0098**، درست در محدودهٔ ازپیش‌تعیین‌شدهٔ قابل قبول.

پایداری grid نیز بهتر شد: blend محافظه‌کارانه 16 پروفایل در فاصلهٔ `0.025 mm` و 31 پروفایل در فاصلهٔ `0.05 mm` از بهترین MAE دارد؛ Exp05 به‌ترتیب 11 و 20 پروفایل داشت. بنابراین سود فقط یک minimum منفرد نیست. بااین‌حال profile انتخاب‌شده روی fold0 هنوز in-fold است و تعمیم آن باید با fold1/2 و strict LOO بررسی شود.

---

## 7. برداشت عمیق از مسئله و معماری فعلی

### 7.1 مسئله فقط localization نیست

مدل هم باید landmark/heatmap مناسب تولید کند، هم برش‌های مرتبط با peak را رتبه‌بندی کند، و هم این خروجی‌ها در pooling سطح مطالعه به یک MLS پایدار تبدیل شوند. ممکن است slice MAE یا AUC بهتر شود ولی study MAE بدتر شود؛ داده‌های فعلی دقیقاً این پدیده را نشان می‌دهند.

### 7.2 ranking و calibration دو مسئلهٔ جدا هستند

AUC بالا یعنی ترتیب برش‌ها اغلب درست است، اما تضمین نمی‌کند probabilityها در gate ثابت به‌درستی calibration شده باشند. poolingهای relative component و quantile نسبت به تغییر calibration مقاوم‌تر از threshold مطلق‌اند و به همین دلیل در cross-fold خوب ظاهر شده‌اند.

### 7.3 fold2 دشوارتر و معیار اصلی robustness است

همهٔ خانواده‌های مدل روی fold2 خطای بیشتری دارند. بنابراین انتخاب profile صرفاً بر اساس mean می‌تواند آسیب‌پذیری پنهان بسازد. در گزارش‌ها هم mean و هم worst-fold نگه داشته شده‌اند.

### 7.4 معماری پایه مشکل بنیادی ندارد

HRNet-W32 ظرفیت کافی برای localization و selector مناسب نشان داده است. شواهد فعلی تغییر کامل backbone را توجیه نمی‌کند. بیشترین بازده فعلی از اصلاح target، pooling، checkpoint selection و ensemble حاصل شده است. معماری جدید فقط وقتی باید وارد شود که آزمایش کنترل‌شده نشان دهد این گلوگاه‌ها اشباع شده‌اند.

### 7.5 عدد 0.914 leaderboard مستقیماً با MAE محلی قابل مقایسه نیست

امتیاز leaderboard یک score مسابقه‌ای/ترکیبی است، در حالی که این سند عمدتاً MAE میلی‌متری، RMSE، AUC و Boundary-F1 تسک MLS را گزارش می‌کند. رسیدن MLS به MAE پایین شرط مهمی است، اما از این اعداد به‌تنهایی نمی‌توان score نهایی `0.914` را نتیجه گرفت. قبل از ادعای رتبه باید pipeline submission و metric مجازی دقیقاً با راهنمای مسابقه تطبیق و سپس حداقل یک submission رسمی انجام شود.

---

## 8. برنامهٔ مرحلهٔ بعد

### فاز A — تکمیل امن Experiment 08: انجام‌شده

1. هر 30 epoch فقط روی CUDA تکمیل شد.
2. تمام metricها و snapshotهای 13 تا 29 ثبت و وجود فایل‌ها تأیید شد.
3. MLflow با status=`completed` و Run ID=`85a9cba212fa45a19ebc6f972106a802` پایان یافت.
4. peak VRAM حدود `4.55 GB` بود؛ OOM، fallback یا epoch ناقص نداشتیم.

### فاز B — full-study CUDA audit و snapshot ensemble: انجام‌شده

1. هر 12 checkpoint لازم، نه فقط shortlist دلخواه، روی 70 مطالعه ارزیابی شد.
2. 14 مدل/checkpoint با `6048` پروفایل مشترک مقایسه شدند.
3. 14 snapshot/cross-model blend هدفمند ساخته و با همان grid ارزیابی شدند.
4. epoch21 بهترین ظرفیت diagnostic را داشت؛ blend محافظه‌کارانهٔ Exp05/epoch21 معیار ازپیش‌تعیین‌شدهٔ ادامه را پاس کرد.

### فاز C — گسترش کنترل‌شدهٔ hybrid-soft به foldهای دیگر: مرحلهٔ فعال

معیارهای قبولی fold0 پاس شدند:

- بهبود MAE با پروفایل frozen و پروتکل یکسان: حدود `5.15%`، بیشتر از حداقل 3٪؛
- افت Boundary-F1 frozen: `0.0098`، کمتر از سقف `0.01`؛
- پایداری profile بهتر از baseline: 16 در برابر 11 پروفایل در فاصلهٔ `0.025 mm` از optimum.

برنامهٔ اجرایی:

1. ساخت manifestهای fold1 و fold2 با همان معماری، target و seed policy؛ tuning جداگانه روی هر fold ممنوع است.
2. کوتاه‌کردن schedule به ناحیهٔ مفید و ذخیرهٔ snapshotهای `13, 15, 17, 19, 21, 23`؛ شواهد fold0 ادامهٔ بعد از 23 را توجیه نمی‌کند.
3. آموزش fold1 روی GPU محلی و ثبت کامل در MLflow؛ سپس E2E محدود به checkpointهای ازپیش‌تعریف‌شده.
4. اگر fold1 نشانهٔ complementarity/بهبود transfer را حفظ نکرد، قبل از fold2 توقف و تحلیل می‌شود. اگر حفظ کرد، fold2 نیز آموزش داده می‌شود.
5. وزن‌های blend بدون استفاده از held-out fold انتخاب و strict leave-one-fold-out دوباره محاسبه می‌شود.

وضعیت اجرا در زمان آخرین به‌روزرسانی این سند:

- manifest: `config/experiments/mls-local-v2-exp09-w32-fold1-hybridsoft-transfer.yaml`
- Run ID در MLflow: `5e1449cbe73b4152b66589df7f20898c`
- وضعیت: هر 23 ایپاک کامل شده‌اند؛ snapshotهای ایپاک 13، 15، 17، 19، 21 و 23 و checkpoint نهایی محفوظ‌اند. run روی MLflow با وضعیت `FINISHED` بسته و چهار checkpoint انتخابی (`best`، `best_mae`، `best_selector_auc` و `best_study`) به‌همراه گزارش، history و summary روی سرور تأیید شده‌اند.
- sanity epoch1: `slice MAE=38.434`، `study MAE=3.584`، `keypoint=148.1px`، `selector AUC=0.387` و همهٔ مقادیر finite؛ هم‌الگوی warm-up آزمایش 08 و هنوز فاقد ارزش رتبه‌بندی.
- sanity epoch2: `slice MAE=10.765`، `study MAE=3.584`، `keypoint=37.4px` و `selector AUC=0.607`. افت 72٪ در slice MAE و 75٪ در keypoint error نسبت به epoch1 نشان می‌دهد optimization درست شروع شده است؛ صفرماندن study Boundary-F1 در این مرحله ناشی از نرسیدن selector به gate عملی است.
- sanity epoch3: `slice MAE=7.704`، `study MAE=3.584`، `keypoint=44.4px` و `selector AUC=0.798`. ranking selector و خطای برش بهبود واضح دارند؛ نوسان موقت keypoint در warm-up بدون بدترشدن سایر معیارها دلیل توقف نیست.
- sanity epoch4: `slice MAE=2.803`، `study MAE=3.584`، `keypoint=22.4px` و `selector AUC=0.777`. localization جهش مثبت داشت؛ study metric هنوز پیش از فعال‌شدن مؤثر selector صفر است و با trajectory fold0 هم‌خوانی دارد.
- sanity epoch5: `slice MAE=2.104`، `study MAE=3.584`، `keypoint=10.2px` و `selector AUC=0.829`. هر سه شاخهٔ localization/ranking/MLS وارد ناحیهٔ مفید شده‌اند؛ عدم تغییر study metric هنوز ناشی از gate پیش از activation است.
- sanity epoch6: `slice MAE=2.141`، `study MAE=3.584`، `keypoint=9.6px`، `selector AUC=0.837` و برای نخستین‌بار `selector F1=0.101`. احتمال‌های selector در حال عبور از warm-up هستند؛ نوسان جزئی slice MAE دلیل توقف نیست.
- sanity epoch7: `slice MAE=1.744`، `study MAE=3.584`، `study Boundary-F1=0.000`، `keypoint=8.7px`، `selector AUC=0.845` و `selector F1=0.742`. جهش selector-F1 از `0.101` به `0.742` همراه با بهبود localization نشان می‌دهد head انتخاب‌گر از warm-up عبور کرده، اما pooling مطالعه هنوز فعال نشده است. در fold0 نیز metrics مطالعه یک epoch دیرتر فعال شدند؛ بنابراین توقف در این نقطه زودهنگام و برخلاف پروتکل ازپیش‌تعریف‌شده است و epochهای 8 و 9 برای قضاوت ضروری‌اند.
- sanity epoch8: `slice MAE=1.893`، `study MAE=1.701`، `study Boundary-F1=0.663`، `keypoint=9.2px`، `selector AUC=0.887`، `selector F1=0.792` و objective=`2.431`. جهش همزمان study MAE از `3.584` به `1.701` و Boundary-F1 از صفر به `0.663` تأیید می‌کند gate/pooling مطالعه بالاخره وارد ناحیهٔ عملی شده است. این بهبود با فعال‌شدن تدریجی epoch7 سازگار است و نوسان تصادفیِ بدون پیش‌زمینه نیست؛ epoch9 برای سنجش پایداری و ادامهٔ calibration لازم است.
- sanity epoch9: `slice MAE=1.726`، `study MAE=2.125`، `study Boundary-F1=0.468`، `keypoint=8.5px`، `selector AUC=0.873`، `selector F1=0.622` و objective=`3.253`. localization برش/landmark بهتر شد، اما calibration و pooling نسبت به epoch8 پس‌رفت کردند؛ بنابراین افت به کمبود ظرفیت feature extractor نسبت داده نمی‌شود. این نخستین epoch ضعیف بعد از activation است، نه روند افت پایدار. اجرای epoch10 و رسیدن به نخستین snapshot ازپیش‌ثبت‌شدهٔ epoch13 حفظ می‌شود؛ checkpoint برتر قبلی نیز مستقل ذخیره شده و ادامه، نتیجهٔ epoch8 را از بین نمی‌برد.
- sanity epoch10: `slice MAE=1.880`، `study MAE=1.158`، `study Boundary-F1=0.820`، `keypoint=7.7px`، `selector AUC=0.905`، `selector F1=0.800` و objective=`1.566`. مدل نه‌فقط افت epoch9 را کامل بازیابی کرد، بلکه نسبت به بهترین MAE مطالعه در epoch8 حدود `31.9%` بهتر شد و هم‌زمان رکوردهای localization، ranking و boundary را بهبود داد. این recovery فرضیهٔ نوسان calibration/pooling را تأیید و فرضیهٔ شکست معماری fold1 را رد می‌کند. ادامه تا snapshotهای ازپیش‌ثبت‌شدهٔ 13 تا 23 حفظ می‌شود و انتخاب نهایی تنها پس از E2E full-study انجام خواهد شد.
- sanity epoch11: `slice MAE=1.909`، `study MAE=1.576`، `study Boundary-F1=0.704`، `keypoint=9.4px`، `selector AUC=0.898`، `selector F1=0.758` و objective=`2.220`. معیارها نسبت به peak ایپاک 10 افت کردند، اما از collapse ایپاک‌های قبل از activation فاصله دارند و recovery کلی حفظ شده است. ایپاک 10 فعلاً peak داخلی قوی و ایپاک 11 شاهد نوسان calibration است؛ این اختلاف دلیل دیگری برای نگه‌داشتن چند checkpoint و انجام E2E به‌جای انتخاب کور بر اساس آخرین ایپاک است.
- sanity epoch12: `slice MAE=1.745`، `study MAE=1.553`، `study Boundary-F1=0.689`، `keypoint=7.9px`، `selector AUC=0.892`، `selector F1=0.732` و objective=`2.229`. MAE مطالعه اندکی از epoch11 بهتر و localization به‌وضوح بازیابی شد، اما peak چندمعیارهٔ epoch10 شکسته نشد. epochهای 11 و 12 یک plateau نوسانی اما سالم پس از peak هستند؛ نخستین snapshot مستقل epoch13 برای E2E طبق برنامه حفظ می‌شود.
- sanity epoch13: `slice MAE=1.806`، `study MAE=1.050`، `study Boundary-F1=0.825`، `keypoint=8.1px`، `selector AUC=0.898`، `selector F1=0.803` و objective=`1.452`. این ایپاک رکورد داخلی تازه ساخت: MAE مطالعه حدود `9.3%` از peak قبلی epoch10 بهتر شد و Boundary-F1 نیز اندکی افزایش یافت. snapshot مستقل `mls_multitask_epoch_013.pth` با حجم `124,890,565` بایت و timestamp جدید مستقیماً روی دیسک تأیید شد. بااین‌حال به‌علت mismatch اثبات‌شدهٔ proxy/E2E، این نتیجه هنوز کاندیدا است نه برندهٔ نهایی.
- sanity epoch14: `slice MAE=1.743`، `study MAE=0.984`، `study Boundary-F1=0.800`، `keypoint=7.8px`، `selector AUC=0.901`، `selector F1=0.807` و objective=`1.433`. MAE مطالعه حدود `6.3%` دیگر نسبت به epoch13 بهتر شد و ثابت کرد ناحیهٔ peak پایدارتر از یک نوسان تک‌ایپاکی است؛ در مقابل Boundary-F1 به‌اندازهٔ `0.025` افت کرد. در نتیجه epoch14 بهترین کاندیدای MAE داخلی و epoch13 کاندیدای متوازن‌تر boundary/MAE هستند؛ E2E و احتمالاً blend snapshotها باید این trade-off را حل کنند.
- sanity epoch15: `slice MAE=1.827`، `study MAE=1.016`، `study Boundary-F1=0.811`، `keypoint=8.4px`، `selector AUC=0.890`، `selector F1=0.783` و objective=`1.450`. این snapshot از نظر MAE میان epoch13 و 14 و از نظر Boundary-F1 نیز میان آن‌ها قرار گرفت؛ بنابراین یک کاندیدای متوازن مستقل برای E2E است. فایل `mls_multitask_epoch_015.pth` با حجم `124,890,565` بایت و timestamp جدید مستقیماً روی دیسک تأیید شد.
- sanity epoch16: `slice MAE=1.774`، `study MAE=1.090`، `study Boundary-F1=0.741`، `keypoint=7.4px`، `selector AUC=0.905`، `selector F1=0.757` و objective=`1.656`. شاخهٔ localization و ranking رکوردهای قوی ساختند، اما calibration/pooling مطالعه نسبت به epochهای 13 تا 15 افت کرد. این جدایی به معنی overfitting سراسری feature extractor نیست؛ epoch17 برای تشخیص recovery یا آغاز افت پایدار حفظ می‌شود و snapshotهای قبلی مستقل و امن‌اند.
- sanity epoch17: `slice MAE=1.813`، `study MAE=0.950`، `study Boundary-F1=0.812`، `keypoint=7.4px`، `selector AUC=0.903`، `selector F1=0.767` و objective=`1.375`. مدل افت study-level ایپاک 16 را کامل جبران کرد و بهترین MAE و objective داخلی run را ساخت؛ بنابراین فرضیهٔ آغاز overfitting پایدار رد شد و نوسان calibration/pooling تأیید شد. فایل `mls_multitask_epoch_017.pth` با حجم `124,890,565` بایت و timestamp جدید مستقیماً روی دیسک تأیید شد.
- sanity epoch18: `slice MAE=1.787`، `study MAE=1.045`، `study Boundary-F1=0.808`، `keypoint=7.4px`، `selector AUC=0.889`، `selector F1=0.723` و objective=`1.484`. نسبت به رکورد epoch17 افت ملایم داشت، اما هم MAE و هم Boundary-F1 در ناحیهٔ قوی باقی ماندند؛ بنابراین هنوز روند overfitting پایدار وجود ندارد. snapshot ایپاک 19 برای سنجش ماندگاری این ناحیه طبق برنامه حفظ می‌شود.
- sanity epoch19: `slice MAE=1.676`، `study MAE=1.153`، `study Boundary-F1=0.727`، `keypoint=7.3px`، `selector AUC=0.893`، `selector F1=0.740` و objective=`1.753`. بهترین localization برش و landmark فعلی هم‌زمان با افت study-level رخ داد؛ این شاهد مستقیم دیگری است که calibration/pooling و نه کیفیت feature خام، گلوگاه نوسان است. snapshot مستقل `mls_multitask_epoch_019.pth` با حجم `124,890,565` بایت و timestamp جدید تأیید شد و برای E2E نگه داشته می‌شود، زیرا proxy داخلی به‌تنهایی برای حذف آن کافی نیست.
- sanity epoch20: `slice MAE=1.717`، `study MAE=1.244`، `study Boundary-F1=0.759`، `keypoint=7.3px`، `selector AUC=0.891`، `selector F1=0.745` و objective=`1.780`. این دومین ایپاک پیاپی پس از peak17 است که study-level ضعیف‌تر شده، درحالی‌که localization خوب مانده است؛ احتمال ورود calibration/pooling به روند افت افزایش یافته، اما snapshot ازپیش‌ثبت‌شدهٔ epoch21 برای آزمون recovery و جلوگیری از توقف زودهنگام حفظ می‌شود.
- sanity epoch21: `slice MAE=1.736`، `study MAE=0.941`، `study Boundary-F1=0.800`، `keypoint=7.2px`، `selector AUC=0.895`، `selector F1=0.786` و objective=`1.394`. مدل پس از افت epochهای 19 و 20 recovery کامل کرد و بهترین MAE مطالعه و keypoint error run را ثبت کرد؛ در مقابل objective متوازن epoch17 با `1.375` هنوز اندکی بهتر است. snapshot مستقل `mls_multitask_epoch_021.pth` با حجم `124,890,565` بایت و timestamp جدید تأیید شد و هر دو epoch17 و 21 باید در E2E shortlist اصلی باشند.
- sanity epoch22: `slice MAE=1.730`، `study MAE=1.217`، `study Boundary-F1=0.751`، `keypoint=7.2px`، `selector AUC=0.892`، `selector F1=0.750` و objective=`1.768`. localization تقریباً در سطح قوی epoch21 ماند، اما MAE و boundary مطالعه هم‌زمان افت کردند؛ بنابراین این تغییر بار دیگر به calibration/pooling نسبت داده می‌شود و نه collapse ویژگی‌های مکانی. طبق برنامه فقط ایپاک نهایی 23 ادامه یافت.
- sanity epoch23: `slice MAE=1.744`، `study MAE=1.101`، `study Boundary-F1=0.772`، `keypoint=7.3px`، `selector AUC=0.895`، `selector F1=0.739` و objective=`1.610`. نسبت به epoch22 بازیابی ناقص رخ داد، اما هیچ‌یک از رکوردهای epochهای 13، 17 یا 21 شکسته نشد. snapshot مستقل `mls_multitask_epoch_023.pth` با حجم `124,890,565` بایت و checkpoint `mls_multitask_final.pth` با حجم `124,882,917` بایت مستقیماً تأیید شدند. پایان schedule در 23 موجه بود؛ شواهد fold0 و fold1 هیچ توجیهی برای ادامهٔ بی‌هدف آموزش پس از این ناحیه نمی‌دهند.
- نتیجهٔ پایان آموزش: سرعت حدود `1.78–1.81 it/s` و اوج VRAM حدود `4.55 GB` پایدار ماند؛ 23/23 ایپاک بدون NaN، OOM، epoch ناقص یا fallback محاسبات مدل به CPU کامل شد. انتخاب نهایی عمداً به E2E شش snapshot واگذار شده است، زیرا سه معیار داخلی سه epoch متفاوت را برنده می‌دانند.
- وضعیت MLflow: run با شناسهٔ `5e1449cbe73b4152b66589df7f20898c` در وضعیت `FINISHED` بسته شد. چهار checkpoint انتخابی و `reports/epoch_metrics.jsonl`، `reports/report.md` و `reports/run_summary.json` روی سرور رؤیت شدند؛ snapshotهای دوره‌ای و داده/پیش‌بینی خام فقط محلی باقی ماندند.
- سلامت اولیه: حدود `1.78–1.81 it/s`، GPU utilization حدود 98٪، دما `65°C` و بدون NaN/OOM/fallback.
- لینک run: `https://dagshub.com/amiresbati62/BrainCtTriage.mlflow/#/experiments/16/runs/5e1449cbe73b4152b66589df7f20898c`

#### نتیجهٔ ممیزی full-study fold1

- شش checkpoint ازپیش‌تعریف‌شده روی هر 67 مطالعه ارزیابی شدند: `402/402` موفق، صفر failure، بدون OOM و بدون CPU fallback مدل؛ زمان inference ثبت‌شده در مجموع حدود 549 ثانیه بود.
- proxy داخلی برندهٔ واحد نداشت: epoch17 بهترین objective، epoch21 بهترین MAE و epoch13 بهترین Boundary-F1 را داشتند؛ اما common-grid E2E، epoch15 را به‌عنوان بهترین single متوازن انتخاب کرد. این mismatch نشان می‌دهد ادامهٔ انتخاب checkpoint فقط با validation داخلی غیرقابل دفاع است.
- روی 6048 پروفایل مشترک، single epoch15 به `MAE=1.2722`، `Boundary-F1=0.8635` و objective=`1.5453` رسید. epoch19 کمترین MAE تشخیصی (`1.1877`) را داشت، ولی Boundary-F1 آن `0.7983` بود و به‌تنهایی کاندیدای متوازن مناسبی نیست.
- baseline منصفانهٔ Exp06 روی همان grid: `MAE=1.3301`، `Boundary-F1=0.8035` و objective=`1.7231`. بنابراین مزیت hybrid فقط ناشی از فضای جست‌وجوی متفاوت نیست.
- روی پروفایل frozen fold0→fold1، Exp06 برابر `MAE=1.6206` و `Boundary-F1=0.7340` بود؛ Exp09/epoch15 به `1.4812` و `0.7902` رسید. کاهش MAE `8.60%` و افزایش هم‌زمان boundary، معیار اصلی صدور مجوز fold2 است.
- median سه snapshot 15/17/19 درون‌fold به `MAE=1.1195`، `Boundary-F1=0.8175` و objective=`1.4846` رسید. نتیجه امیدوارکننده است، اما تا تأیید fold2/LOO به‌علت هزینهٔ سه‌برابری inference و انتخاب profile در همان fold، production candidate قطعی نیست.
- روی همان پروفایل frozen، `75% Exp06 + 25% Exp09/epoch21` بهترین گزینه شد: `MAE=1.4096`، `Boundary-F1=0.7859` و objective=`1.8379`. single epoch15 ساده‌تر و کمی boundary-safe‌تر است؛ blend MAE و objective بهتری دارد.
- نتیجهٔ تصمیم: آموزش بیشتر fold1 بعد از epoch23 متوقف می‌ماند، زیرا late training سود E2E پایداری نساخت. آزمایش بعدی fold2 با همان معماری، target، seed و schedule اجرا می‌شود؛ tuning مخصوص fold ممنوع است.

#### Experiment 10: آزمون robustness روی fold2

- manifest: `config/experiments/mls-local-v2-exp10-w32-fold2-hybridsoft-transfer.yaml`
- Run ID در MLflow: `a4c44492fcc141058e5aae71266c6c33`
- تنها متغیر تغییرکرده نسبت به Exp09، fold از 1 به 2 است؛ معماری، hybrid-soft target با `peak_base=0.75`، seed، optimizer، augmentation، batch size، schedule 23 ایپاک و snapshotهای 13/15/17/19/21/23 ثابت مانده‌اند.
- وضعیت شروع: manifest معتبر، دادهٔ پردازش‌شده موجود، CUDA guard فعال، GPU utilization حدود 98٪، مصرف گزارش‌شدهٔ درایور حدود `5537 MiB`، دمای آغاز پایدار حدود `60°C` و سرعت پس از warm-up حدود `1.83–1.85 it/s`.
- sanity epoch1: `slice MAE=28.498`، `study MAE=4.882`، `study Boundary-F1=0.000`، `keypoint=46.7px`، `selector AUC=0.376`، `selector F1=0.000` و objective=`7.194`. همهٔ مقادیر finite هستند؛ localization اولیه از warm-up fold1 بهتر است، اما baseline study-level fold2 سخت‌تر است. صفر بودن gate در ایپاک اول با Exp08/09 هم‌الگو است و هیچ دلیل توقفی ایجاد نمی‌کند.
- sanity epoch2: `slice MAE=6.505`، `study MAE=4.882`، `study Boundary-F1=0.000`، `keypoint=40.0px`، `selector AUC=0.277`، `selector F1=0.000` و objective=`7.243`. slice MAE حدود 77٪ نسبت به epoch1 بهتر شد، اما selector هنوز activate نشده و افت موقت AUC نشان می‌دهد ranking head در warm-up است. جدایی واضح میان بهبود شاخهٔ MLS و ثابت‌ماندن gate دلیل ادامه تا activation است، نه توقف.
- sanity epoch3: `slice MAE=3.275`، `study MAE=4.882`، `study Boundary-F1=0.000`، `keypoint=14.1px`، `selector AUC=0.510`، `selector F1=0.000` و objective=`7.126`. نسبت به epoch2، slice MAE تقریباً نصف و keypoint error حدود 65٪ کمتر شد و AUC نیز بازیابی شد. شاخه‌های localization/ranking سالم‌اند؛ فقط selector هنوز از threshold عملی عبور نکرده است.
- sanity epoch4: `slice MAE=2.898`، `study MAE=4.882`، `study Boundary-F1=0.000`، `keypoint=13.9px`، `selector AUC=0.773`، `selector F1=0.000` و objective=`6.995`. جهش AUC نشان می‌دهد ranking selector به‌سرعت یاد گرفته است، ولی probability calibration هنوز gate ثابت را فعال نکرده؛ این جدایی ranking/calibration با یافته‌های fold0/1 هم‌خوان است.
- sanity epoch5: `slice MAE=2.407`، `study MAE=4.882`، `study Boundary-F1=0.000`، `keypoint=11.1px`، `selector AUC=0.812`، `selector F1=0.000` و objective=`6.976`. ranking و localization هر دو وارد ناحیهٔ مفید شده‌اند، اما calibration fold2 دیرتر فعال می‌شود. مقایسه با Exp09 که selector-F1 را در epoch6 و study metric را در epoch8 فعال کرد، ادامهٔ epochهای 6 تا 8 را ضروری می‌کند.
- sanity epoch6: `slice MAE=2.619`، `study MAE=4.882`، `study Boundary-F1=0.000`، `keypoint=10.7px`، `selector AUC=0.836`، `selector F1=0.000` و objective=`6.964`. افزایش خفیف slice MAE در برابر بهترین مقدار epoch5 با بهبود هم‌زمان keypoint، AUC و objective همراه است؛ بنابراین نشانهٔ واگرایی نیست. مسئلهٔ فعلی به‌طور مشخص calibration/gating است: مدل موارد مفید را بهتر rank می‌کند، ولی احتمال‌ها هنوز از آستانهٔ عملی عبور نکرده‌اند. تصمیم از پیش تعیین‌شده حفظ شد: مشاهده تا حداقل epoch8 و قضاوت بر اساس activation و E2E snapshotها، نه توقف بر مبنای proxy زودهنگام.
- sanity epoch7: `slice MAE=2.364`، `study MAE=4.882`، `study Boundary-F1=0.000`، `keypoint=11.4px`، `selector AUC=0.840`، `selector F1=0.752` و objective=`6.962`. selector مطابق انتظار از حالت صفر خارج شد و F1 آن بلافاصله وارد بازهٔ مفید شد؛ در نتیجه representation/ranking سالم است. ثابت‌ماندن study metric نشان می‌دهد فعال‌شدن probability به‌تنهایی هنوز شرط pooling/حداقل شواهد study را پاس نکرده است. epoch8 آزمون ازپیش‌تعیین‌شده برای گذار کامل gate→study است و توقف در این نقطه اشتباه خواهد بود.
- sanity epoch8: `slice MAE=2.607`، `study MAE=3.266`، `study Boundary-F1=0.573`، `keypoint=11.0px`، `selector AUC=0.864`، `selector F1=0.710` و objective=`4.187`. گذار کامل gate→study رخ داد: study MAE نسبت به baseline ثابت epochهای 1–7 حدود `33.1%` کم شد و Boundary-F1 از صفر خارج شد. افزایش جزئی slice MAE نسبت به epoch7 در برابر این جهش بزرگ E2E-proxy بی‌اهمیت است و دوباره نشان می‌دهد slice proxy به‌تنهایی معیار توقف یا انتخاب checkpoint نیست. fold2 سخت‌تر از fold1 است، اما مسیر یادگیری سالم و ادامه تا snapshotهای preregistered کاملاً موجه است.
- sanity epoch9: `slice MAE=2.356`، `study MAE=2.110`، `study Boundary-F1=0.872`، `keypoint=9.38px`، `selector AUC=0.880`، `selector F1=0.760` و objective=`2.426`. پس از activation اولیه، pooling در یک ایپاک دیگر بالغ شد: study MAE نسبت به epoch8 حدود `35.4%` بهتر و Boundary-F1 حدود `0.299` بیشتر شد. هر چهار محور localization، ranking، calibration و study aggregation هم‌زمان سیگنال مثبت دارند؛ در نتیجه ادامه تا پنجرهٔ snapshot 13–23 هم از نظر کیفیت و هم از نظر مصرف منابع موجه است.
- sanity epoch10: `slice MAE=2.484`، `study MAE=2.782`، `study Boundary-F1=0.699`، `keypoint=9.9px`، `selector AUC=0.883`، `selector F1=0.634` و objective=`3.442`. نسبت به جهش epoch9، calibration/pooling نوسان منفی داشته، اما AUC اندکی بهتر و شاخص‌های localization همچنان سالم‌اند؛ پس representation collapse رخ نداده است. با توجه به mismatch اثبات‌شدهٔ proxy/E2E و اینکه هنوز پیش از اولین snapshot هستیم، این افت یک‌ایپاکی نه مجوز توقف است و نه مجوز تغییر تنظیمات؛ پایداری در snapshotهای 13 به بعد معیار تصمیم خواهد بود.
- sanity epoch11: `slice MAE=2.641`، `study MAE=1.646`، `study Boundary-F1=0.935`، `keypoint=9.0px`، `selector AUC=0.908`، `selector F1=0.823` و objective=`1.823`. بهترین proxy fold2 تا این نقطه ثبت شد و افت epoch10 جبران گردید. تضاد مهم: با وجود slice MAE ضعیف‌تر از epoch9، study MAE حدود `22.0%` بهتر و Boundary-F1 حدود `0.063` بالاتر است؛ این شواهد مستقیم دیگری است که انتخاب checkpoint با slice MAE یا حتی یک metric منفرد خطاست. ادامه تا snapshotها ضروری است تا این basin با E2E واقعی سنجیده شود.
- sanity epoch12: `slice MAE=2.404`، `study MAE=1.707`، `study Boundary-F1=0.869`، `keypoint=9.9px`، `selector AUC=0.908`، `selector F1=0.803` و objective=`2.014`. کیفیت بالا تقریباً حفظ شد: study MAE فقط حدود `3.7%` از epoch11 بدتر است و AUC بدون افت مانده، هرچند Boundary-F1 نوسان دارد. حضور دو epoch متوالی در basin قوی، ادامه تا اولین snapshot epoch13 را تأیید می‌کند؛ انتخاب نهایی همچنان پس از E2E کامل خواهد بود.
- snapshot epoch13: `slice MAE=2.230`، `study MAE=1.947`، `study Boundary-F1=0.860`، `keypoint=9.0px`، `selector AUC=0.889`، `selector F1=0.679` و objective=`2.283`. نسبت به epoch11–12 ضعیف‌تر اما هنوز در basin مفید است. وجود فایل `models/checkpoints/mls_multitask/mls-local-v2-exp10-w32-fold2-hybridsoft-transfer/mls_multitask_epoch_013.pth` با اندازهٔ حدود `124.9 MB` مستقلاً روی دیسک تأیید شد؛ بنابراین اولین candidate ممیزی E2E و recovery point پایدار آماده است.
- sanity epoch14: `slice MAE=2.378`، `study MAE=1.518`، `study Boundary-F1=0.957`، `keypoint=9.3px`، `selector AUC=0.914`، `selector F1=0.824` و objective=`1.647`. بهترین proxy فعلی روی هر سه محور study MAE، boundary و objective است. یک batch با loss لحظه‌ای `8.17` بلافاصله به بازهٔ معمول برگشت و بهترین validation کل run را تولید کرد؛ بنابراین spike ناشی از سختی/ترکیب batch بوده و شواهدی برای instability یا توقف نیست. epoch15 به‌عنوان snapshot دوم ادامه یافت.
- snapshot epoch15: `slice MAE=2.337`، `study MAE=1.476`، `study Boundary-F1=0.922`، `keypoint=8.9px`، `selector AUC=0.911`، `selector F1=0.814` و objective=`1.676`. کمترین study MAE داخلی تا این نقطه ثبت شد، ولی epoch14 هنوز boundary و objective اندکی بهتر دارد؛ این trade-off انتخاب proxy-only را نامعتبر می‌کند و نگه‌داشتن هر دو candidate برای grid E2E را توجیه می‌کند. وجود `mls_multitask_epoch_015.pth` روی دیسک مستقلاً تأیید شد.
- sanity epoch16: `slice MAE=2.239`، `study MAE=1.542`، `study Boundary-F1=0.958`، `keypoint=8.8px`، `selector AUC=0.912`، `selector F1=0.773` و objective=`1.670`. سومین epoch متوالی 14–16 با objective حدود `1.65–1.68` ثبت شد؛ بنابراین بهبود late-middle training یک peak تصادفی نیست و basin پایدار است. epoch17 به‌عنوان snapshot سوم ارزش ممیزی بالایی دارد.
- snapshot epoch17: `slice MAE=2.287`، `study MAE=1.546`، `study Boundary-F1=0.912`، `keypoint=9.0px`، `selector AUC=0.912`، `selector F1=0.805` و objective=`1.765`. یک batch با loss `9.93` فوراً recover شد و validation همچنان قوی ماند؛ پس اثر مخرب ماندگار نداشت. proxy آن اندکی از 14–16 ضعیف‌تر است، اما به‌علت mismatch قبلی حذف نمی‌شود. فایل `mls_multitask_epoch_017.pth` با اندازهٔ `124,890,565` بایت روی دیسک تأیید شد.
- sanity epoch18: `slice MAE=2.393`، `study MAE=1.791`، `study Boundary-F1=0.875`، `keypoint=8.4px`، `selector AUC=0.905`، `selector F1=0.728` و objective=`2.087`. pooling/calibration نسبت به basin 14–17 پس‌رفت کرده، در حالی که keypoint error به بهترین مقدار رسیده است؛ این جدایی مجدد ثابت می‌کند localization بهتر الزاماً study aggregation بهتر نمی‌سازد. چون epoch19 snapshot preregistered است و هیچ instability وجود ندارد، run ادامه یافت؛ الگوی late degradation در تصمیم schedule دور بعد لحاظ خواهد شد.
- snapshot epoch19: `slice MAE=2.573`، `study MAE=1.769`، `study Boundary-F1=0.908`، `keypoint=8.7px`، `selector AUC=0.906`، `selector F1=0.807` و objective=`2.001`. نسبت به epoch18 کمی بازیابی شد، ولی از basin 14–17 عقب است؛ sweet spot داخلی فعلاً حوالی 14–17 دیده می‌شود. با این حال snapshotهای 21/23 طبق preregistration حفظ می‌شوند تا نتیجه بدون cherry-picking و با E2E واقعی تثبیت شود. فایل epoch19 با اندازهٔ `124,890,565` بایت تأیید شد.
- sanity epoch20: `slice MAE=2.449`، `study MAE=1.730`، `study Boundary-F1=0.883`، `keypoint=8.7px`، `selector AUC=0.901`، `selector F1=0.783` و objective=`2.014`. نتیجه تقریباً سطح late epoch18–19 را حفظ کرد و از basin 14–17 بهتر نشد؛ بنابراین شواهد sweet spot میانی قوی‌تر شده، ولی snapshotهای 21/23 برای آزمون E2E و حذف احتمال proxy mismatch همچنان لازم‌اند.
- snapshot epoch21: `slice MAE=2.509`، `study MAE=1.809`، `study Boundary-F1=0.866`، `keypoint=8.6px`، `selector AUC=0.888`، `selector F1=0.750` و objective=`2.133`. افت late نسبت به basin 14–17 تأیید شد و sweet spot داخلی اکنون شواهد محکمی دارد. با این حال epoch23 برای سنجش proxy/E2E mismatch و complementarity حفظ شد. فایل epoch21 با اندازهٔ `124,890,565` بایت روی دیسک تأیید شد.
- sanity epoch22: `slice MAE=2.496`، `study MAE=1.734`، `study Boundary-F1=0.878`، `keypoint=8.7px`، `selector AUC=0.896`، `selector F1=0.778` و objective=`2.031`. late plateau دوباره تکرار شد و از sweet spot داخلی بهتر نشد. epoch23 آخرین نقطهٔ schedule باقی ماند؛ آموزش بیشتر از 23 بدون شواهد E2E مجاز نیست تا GPU بی‌هدف مصرف نشود.
- snapshot epoch23: `slice MAE=2.421`، `study MAE=1.716`، `study Boundary-F1=0.875`، `keypoint=8.5px`، `selector AUC=0.896`، `selector F1=0.741` و objective=`2.017`. snapshot نهایی late plateau را تغییر نداد؛ sweet spot داخلی همچنان 14–17 است. فایل epoch23 و `mls_multitask_final.pth` هر دو روی دیسک ثبت شدند و آموزش بیشتر متوقف شد، زیرا schedule ازپیش‌ثبت‌شده کامل و ادامهٔ blind فاقد توجیه E2E بود.
- پایان آموزش Exp10: هر 23 epoch با CUDA-only و بدون OOM، NaN یا CPU fallback کامل شد. Run `a4c44492fcc141058e5aae71266c6c33` در MLflow با وضعیت `FINISHED` و end-time معتبر بسته شد. چهار checkpoint انتخابی (`best`، `best_mae`، `best_selector_auc`، `best_study`) و سه artifact گزارش (`epoch_metrics.jsonl`، `report.md`، `run_summary.json`) روی remote مستقلاً تأیید شدند. آپلود به‌علت retry/backoff artifact store کند بود، اما همهٔ موارد مجاز در نهایت کامل شدند؛ raw data و prediction CSV ارسال نشدند.
- raw medical data و predictionهای per-study به MLflow ارسال نمی‌شوند. انتخاب checkpoint پس از ممیزی شش snapshot و strict cross-fold analysis انجام خواهد شد.
- لینک run: `https://dagshub.com/amiresbati62/BrainCtTriage.mlflow/#/experiments/16/runs/a4c44492fcc141058e5aae71266c6c33`

#### نتیجهٔ ممیزی full-study fold2 و تحلیل strict سه‌fold

- شش snapshot 13/15/17/19/21/23 روی هر 67 مطالعه ارزیابی شدند: `402/402` موفق، صفر failure، مجموع زمان حدود `614.7 s` و تمام forwardها CUDA-only.
- common grid مخصوص fold2، epoch17 را با `MAE=1.5998`، `Boundary-F1=0.9412` و objective=`1.7174` برنده کرد. epoch15 به `1.6669` و epoch21 به `1.6380 mm` رسیدند؛ epochهای 19 تا 23 کیفیت transferable تازه‌ای نساختند.
- انتخاب joint epoch+profile در nested LOO ناپایدار بود: balanced mean=`1.8111` و worst=`1.9302`؛ MAE-first به‌علت انتخاب epoch19 برای fold2 تا mean=`1.9927` و worst=`2.6559` افت کرد.
- با ثابت‌کردن epoch، تفاوت علت آشکار شد. epoch15 در nested balanced به `Mean=1.6273`، `Worst=1.7999` و `Boundary-F1=0.8208` رسید. boundary-first برای همین epoch `Mean=1.6045` و `Boundary-F1=0.8344` ثبت کرد. سایر epochهای ثابت در balanced: epoch13=`2.0539`، epoch17=`1.9708`، epoch19=`1.9338`، epoch21=`1.7704` و epoch23=`1.8428 mm`.
- بهترین diagnostic مشترک هر سه fold نیز epoch15 با `severity_window(size=3, gate=0.5, min_active=3, q=0.75, weighted=true)` شد: fold0=`1.6646`، fold1=`1.2587`، fold2=`1.7144`، mean=`1.5459`، worst=`1.7144` و Boundary-F1=`0.8469`.
- در انتقال کاملاً frozen fold0/1→fold2، epoch15 با profile محافظه‌کار به `MAE=1.7999` و Boundary-F1=`0.8754` رسید؛ peak-aware قبلی در همان منطق انتقال `1.9744 mm` و حدود `0.8658` داشت. کاهش MAE حدود `8.8%` است.
- تفسیر نهایی: معماری hybrid شکست‌خورده نیست؛ گلوگاه، variance انتخاب checkpoint و calibration/aggregation است. checkpoint پایه روی epoch15 قفل می‌شود و آموزش blind بعد از 23 یا انتخاب epoch با leaderboard ممنوع می‌ماند.
- گزارش کامل ممیزی: `reports/mls_experiments/mls-local-v2-exp10-w32-fold2-hybridsoft-transfer/checkpoint_audit_report.md`.

#### Experiment 11: پایداری snapshot، SWA و calibration

- غربال prediction-space نشان داد median epochهای 13/15/17 روی profile ثابت به `Mean MAE=1.4894`، `Worst=1.6200`، `Boundary-F1=0.8396` و objective=`1.8102` می‌رسد؛ در برابر single epoch15 با `1.5459`، `1.7144`، `0.8469` و `1.8521`. سود MAE واقعی است، اما هزینه سه inference است.
- سه checkpoint weight-averaged برای foldها با arithmetic کاملاً CUDA ساخته و روی `204/204` مطالعه بدون failure ارزیابی شدند. SWA به `Mean MAE=1.4989` رسید، ولی fold2=`1.7559`، Boundary-F1=`0.8134` و objective=`1.8720` شد؛ پس به‌رغم MAE متوسط بهتر، پایداری/مرزها از single ضعیف‌تر و این شاخه رد شد.
- nested offset calibration برای single هر سه بار `-0.1 mm` را انتخاب کرد. Boundary-F1 به `0.8575` و objective به `1.8372` بهتر شد، اما MAE به `1.5522` بدتر شد؛ بنابراین فقط گزینهٔ boundary-oriented است.
- ممیزی sampler نشان داد class balancing برش‌محور فعلی به studyهایی با برش بیشتر تا `7.2–7.9×` sampling mass می‌دهد. سیاست class→study→slice این دامنه را به حدود `2.9×` کاهش می‌دهد و متغیر مستقل آزمایش آموزشی بعدی است.
- گزارش کامل: `reports/mls_experiments/mls-local-v2-exp11-crossfold-stability/report.md`.

#### Experiment 12: sampler متوازن در سطح مطالعه — در حال اجرا

- فرضیهٔ آزمایش از ممیزی Exp11 می‌آید: sampler قدیمی ابتدا class برش را متوازن می‌کرد، اما درون هر class به مطالعه‌هایی که تعداد برش بیشتری داشتند sampling mass بیشتری می‌داد. در foldها نسبت بیشینه به کمینهٔ exposure مطالعه حدود `7.2–7.9×` بود. sampler جدید وزن هر ردیف را به‌صورت معکوسِ «تعداد مطالعه‌های همان class × تعداد ردیف‌های همان مطالعه در همان class» می‌سازد؛ در نتیجه classها همچنان متوازن‌اند، هر مطالعه درون class سهم یکسان دارد و دامنهٔ exposure کل به حدود `2.9×` کاهش می‌یابد.
- این تغییر opt-in است و default قدیمی حفظ شده تا بازتولید آزمایش‌های قبلی نشکند. دو آزمون واحد، برابری mass بین classها و برابری mass مطالعه‌ها درون هر class را تأیید کردند.
- manifest: `config/experiments/mls-local-v2-exp12-w32-fold2-studybalanced.yaml`. آزمایش کنترل‌شده است: fold2، HRNet-W32، hybrid-soft target، seed، optimizer، augmentation، batch size، lossها، schedule 23 ایپاک و snapshotهای 13/15/17/19/21/23 دقیقاً با Exp10 یکسان‌اند؛ تنها متغیر اصلی `sampling_mode=study_class_balanced` است.
- تلاش اول از مسیر orchestration پیش از ساخت trainer متوقف شد، چون schema پایگاه‌دادهٔ ZenML نسخهٔ `0.96.2` و client محلی `0.95.1` بود. برای پرهیز از تغییر global environment، آموزش با launcher مستقیمِ موجود و همان config ادامه یافت.
- تلاش مستقیم اول با Run ID=`01cb742a48b8454a809c4207df6355fb` پیش از DataLoader و هر batch متوقف شد، زیرا فیلد جدید ابتدا سهواً در config مدل دیگری قرار گرفته بود. فیلد به `MLSHeatmapConfig` منتقل، parse manifest و تست sampler دوباره تأیید و retry با نام مستقل آغاز شد؛ بنابراین هیچ checkpoint یا metric نامعتبر از run ناموفق وارد مقایسه نمی‌شود.
- اجرای سالم: `mls-local-v2-exp12r1-w32-fold2-studybalanced`، MLflow Run ID=`4af89dc814d8439590e69f886c30909b`. policy محاسبات مدل CUDA-only است؛ epoch اول با سرعت پایدار حدود `1.78 it/s`، peak VRAM=`4.76 GB` و بدون OOM، NaN یا CPU fallback کامل شد.
- sanity epoch1: `train loss=4.6888`، `slice MAE=128.989`، `study MAE=4.882`، `study Boundary-F1=0.000`، `keypoint=254.9px`، `selector AUC=0.483` و objective=`7.140`. در مقایسه با Exp10/epoch1، study MAE یکسان و objective اندکی بهتر است، اما شاخهٔ localization شروع بسیار ضعیف‌تری دارد. این نقطه هنوز پیش از activation selector/pooling است و برای رد فرضیه کافی نیست؛ مطابق پروتکل، حداقل trajectory ایپاک‌های بعدی و به‌ویژه گذار epochهای 8–9 بررسی می‌شود.
- sanity epoch2: `train loss=4.2860`، `slice MAE=4.211`، `study MAE=4.882`، `study Boundary-F1=0.000`، `keypoint=14.9px`، `selector AUC=0.238` و objective=`7.263`. نسبت به epoch1، slice MAE حدود `96.7%` و keypoint error حدود `94.2%` کاهش یافتند؛ بنابراین شروع نامطلوب localization یک warm-up گذرا بود، نه شکست معماری یا sampler. در مقایسهٔ کنترل‌شده با Exp10/epoch2 نیز localization بهتر است (`4.211` در برابر `6.505 mm` و `14.9` در برابر `40.0px`)، هرچند AUC انتخاب‌گر فعلاً اندکی پایین‌تر است (`0.238` در برابر `0.277`). study pooling هنوز فعال نشده و ادامه تا ناحیهٔ activation ضروری است.
- sanity epoch3: `train loss=3.4260`، `slice MAE=3.341`، `study MAE=4.882`، `study Boundary-F1=0.000`، `keypoint=12.78px`، `selector AUC=0.452` و `peak AUC=0.417`. در برابر Exp10/epoch3، slice MAE تقریباً برابر (`3.341` در برابر `3.275`)، keypoint بهتر (`12.78` در برابر `14.11px`)، AUC خام پایین‌تر (`0.452` در برابر `0.510`) و peak-AUC اندکی بهتر (`0.417` در برابر `0.405`) است. در نتیجه تا این نقطه تغییر sampler اثر مخرب ساختاری ندارد، اما مزیت study-level نیز هنوز ظاهر نشده است.
- sanity epoch4: `train loss=2.8703`، `slice MAE=2.884`، `study MAE=4.882`، `study Boundary-F1=0.000`، `keypoint=12.94px`، `selector AUC=0.784`، `peak AUC=0.683` و objective=`6.990`. نسبت به Exp10/epoch4، هر چهار شاخص validation مکانی/ranking اندکی بهترند: slice MAE=`2.898`، keypoint=`13.86px`، AUC=`0.773` و peak-AUC=`0.625` در baseline. جهش AUC از epoch3 نشان می‌دهد عقب‌ماندگی موقت selector جبران شده و sampler جدید تا این نقطه به ranking آسیب نزده است؛ study gate هنوز طبق الگوی warm-up غیرفعال است.
- sanity epoch5: `train loss=2.5707`، `slice MAE=2.571`، `study MAE=4.882`، `study Boundary-F1=0.000`، `keypoint=10.73px`، `selector AUC=0.794`، `peak AUC=0.659` و objective=`6.985`. Exp10/epoch5 به‌ترتیب `2.407`، `11.08px`، `0.812` و `0.677` ثبت کرده بود؛ sampler جدید keypoint را اندکی بهتر کرده ولی slice MAE و ranking اندکی عقب‌اند. اختلاف کوچک و study gate هنوز غیرفعال است، بنابراین این نقطه نه پیروزی و نه معیار توقف است.
- sanity epoch6: `train loss=2.4482`، `slice MAE=2.425`، `study MAE=4.882`، `study Boundary-F1=0.000`، `keypoint=10.48px`، `selector AUC=0.810`، `peak AUC=0.704` و objective=`6.976`. Exp10/epoch6 به‌ترتیب `2.619`، `10.66px`، `0.836` و `0.722` داشت؛ پس geometry/MLS برش بهتر ولی ranking کمی ضعیف‌تر است. یک hard batch با loss حدود `29.6` بلافاصله recovery کرد و validation finite و سالم ماند، بنابراین instability ماندگار نیست. selector-F1 همچنان صفر است و epoch7 آزمون آغاز activation خواهد بود.
- sanity epoch7: `train loss=2.3685`، `slice MAE=2.292`، `study MAE=4.882`، `study Boundary-F1=0.000`، `keypoint=9.38px`، `selector AUC=0.827`، `selector F1=0.206`، `peak AUC=0.680` و objective=`6.968`. نسبت به Exp10/epoch7، localization بهتر است (`2.292` در برابر `2.364 mm` و `9.38` در برابر `11.45px`) و AUC فقط کمی پایین‌تر است (`0.827` در برابر `0.840`)، اما F1 انتخاب‌گر بسیار دیرتر calibrate شده (`0.206` در برابر `0.752`). تفسیر: ranking سالم است، ولی probabilityها هنوز به‌اندازهٔ کافی از threshold عملی عبور نکرده‌اند؛ epoch8–9 gate قطعی ادامه/توقف است.
- sanity epoch8: `slice MAE=2.118`، `study MAE=4.882`، `study Boundary-F1=0.000`، `keypoint=9.5px`، `selector AUC=0.855`، `selector F1=0.677` و objective=`6.954`. localization/ranking سالم و F1 انتخاب‌گر فعال شده، اما study pooling هنوز از حالت ثابت خارج نشده است. Exp10 در همین epoch به `study MAE=3.266` و `Boundary-F1=0.573` رسید؛ بنابراین Exp12 فعلاً از نظر probability calibration/حداقل شواهد مطالعه‌ای یک epoch عقب است. مطابق قانون ازپیش‌نوشته‌شده، epoch9 آخرین gate پیش از تصمیم توقف یا ادامه تا نخستین snapshot است.
- sanity epoch9: `slice MAE=2.668`، `study MAE=2.299`، `study Boundary-F1=0.764`، `keypoint=9.7px`، `selector AUC=0.877`، `selector F1=0.778` و objective=`2.833`. gate مطالعه فعال شد و MAE از مقدار ثابت `4.882` حدود `52.9%` کاهش یافت، پس run از gate فنی ادامه عبور کرد. بااین‌حال Exp10/epoch9 هنوز proxy قوی‌تری داشت (`study MAE=2.110`، `Boundary-F1=0.872`، objective=`2.426`)؛ AUC تقریباً برابر و F1 انتخاب‌گر Exp12 اندکی بهتر است. نتیجهٔ صادقانه: sampler جدید خراب نیست، اما تا این نقطه نیز بهبود اثبات‌شده ندارد؛ ادامه فقط برای رسیدن به basin/snapshot ازپیش‌ثبت‌شده است و baseline جایگزین نشده است.
- sanity epoch10: `slice MAE=2.345`، `study MAE=1.689`، `study Boundary-F1=0.898`، `keypoint=9.6px`، `selector AUC=0.893`، `selector F1=0.794` و objective=`1.946`. پس از activation دیرتر، recovery قوی رخ داد؛ Exp10/epoch10 در همین پروتکل `study MAE=2.782`، `Boundary-F1=0.699` و objective=`3.442` داشت. این برتری proxy بزرگ است و ادامه تا snapshot13 را توجیه می‌کند، اما به‌علت mismatch اثبات‌شدهٔ proxy/E2E هنوز مجوز جایگزینی baseline یا گسترش sampler به foldهای دیگر نیست.
- sanity epoch11: `slice MAE=2.164`، `study MAE=2.326`، `study Boundary-F1=0.780`، `keypoint=9.0px`، `selector AUC=0.894`، `selector F1=0.729` و objective=`2.819`. study-level نسبت به peak ایپاک10 افت کرد، در حالی که localization و ranking سالم و حتی بهتر ماندند؛ بنابراین افت به calibration/pooling نسبت داده می‌شود، نه collapse ویژگی‌ها. Exp10/epoch11 با `MAE=1.646` و `Boundary-F1=0.935` قوی‌تر بود. epoch10 فعلاً یک peak امیدوارکننده اما منفرد است و epoch12–13 باید recovery/پایداری را روشن کنند.
- sanity epoch12: `slice MAE=2.287`، `study MAE=1.555`، `study Boundary-F1=0.927`، `keypoint=9.0px`، `selector AUC=0.917`، `selector F1=0.827` و objective=`1.743`. افت epoch11 کامل recovery شد و نسبت به Exp10/epoch12 (`MAE=1.707`، `Boundary-F1=0.869`، objective=`2.014`) هر سه معیار study-level بهتر شدند. بنابراین epoch10 دیگر peak منفرد نیست و یک basin بالقوه در 10/12 دیده می‌شود؛ snapshot13 نخستین آزمون ذخیره‌شدهٔ این فرضیه است.
- snapshot epoch13: `slice MAE=2.334`، `study MAE=1.355`، `study Boundary-F1=0.957`، `keypoint=9.47px`، `selector AUC=0.919`، `selector F1=0.825` و objective=`1.481`. نسبت به Exp10/epoch13 (`MAE=1.947`، `Boundary-F1=0.860`، objective=`2.283`) بهبود proxy بزرگ و هم‌زمان در MAE و boundary ثبت شد. فایل `models/checkpoints/mls_multitask/mls-local-v2-exp12r1-w32-fold2-studybalanced/mls_multitask_epoch_013.pth` با اندازهٔ `124,890,629` بایت و timestamp جدید مستقلاً روی دیسک تأیید شد. این نخستین شاهد قوی به نفع sampler جدید است، اما تا full-study E2E هنوز production claim نیست.
- sanity epoch14: `slice MAE=2.303`، `study MAE=2.359`، `study Boundary-F1=0.790`، `keypoint=9.91px`، `selector AUC=0.894`، `selector F1=0.681` و objective=`2.832`. افت شدید study-level پس از peak13 درحالی رخ داد که localization برش سالم ماند؛ بنابراین sampler هنوز basin پایدارتر را اثبات نکرده و variance calibration/pooling بالاست. epoch13 فعلاً peak قوی و epoch15 مهم‌ترین snapshot متناظر با production baseline است.
- snapshot epoch15: `slice MAE=2.321`، `study MAE=1.944`، `study Boundary-F1=0.832`، `keypoint=9.57px`، `selector AUC=0.880`، `selector F1=0.683` و objective=`2.339`. این proxy از Exp10/epoch15 (`MAE=1.476`، `Boundary-F1=0.922`، objective=`1.676`) ضعیف‌تر است؛ در نتیجه sampler با وجود peak بسیار قوی epoch13، پایداری checkpoint متناظر production را هنوز بهتر نکرده است. فایل `mls_multitask_epoch_015.pth` با اندازهٔ `124,890,629` بایت و timestamp جدید تأیید شد. به‌علت mismatch اثبات‌شدهٔ proxy/E2E، snapshot حذف نمی‌شود و همراه سایر نقاط ازپیش‌ثبت‌شده ممیزی خواهد شد.
- sanity epoch16: `slice MAE=2.383`، `study MAE=1.718`، `study Boundary-F1=0.886`، `keypoint=8.9px`، `selector AUC=0.920`، `selector F1=0.788` و objective=`1.987`. recovery نسبی نسبت به epoch15 رخ داد، اما Exp10/epoch16 با `MAE=1.542`، `Boundary-F1=0.958` و objective=`1.670` همچنان بهتر بود. localization/ranking قوی است و snapshot17 باید مشخص کند study calibration نیز recovery کامل می‌کند یا نه.
- snapshot epoch17: `slice MAE=2.362`، `study MAE=1.317`، `study Boundary-F1=0.947`، `keypoint=9.1px`، `selector AUC=0.912`، `selector F1=0.824` و objective=`1.467`. recovery کامل رخ داد و Exp10/epoch17 (`MAE=1.546`، `Boundary-F1=0.912`، objective=`1.765`) را در هر سه معیار study-level شکست داد. فایل `mls_multitask_epoch_017.pth` با اندازهٔ `124,890,629` بایت و timestamp جدید تأیید شد. اکنون epochهای 13 و 17 ظرفیت proxy بهتر را نشان می‌دهند، ولی افت epoch15 ثابت می‌کند variance زمانی حل نشده است؛ E2E چند snapshot ضروری باقی می‌ماند.
- sanity epoch18: `slice MAE=2.313`، `study MAE=1.736`، `study Boundary-F1=0.886`، `keypoint=8.8px`، `selector AUC=0.903`، `selector F1=0.770` و objective=`2.012`. از peak17 فاصله گرفت، اما در مقایسه با Exp10/epoch18 (`MAE=1.791`، `Boundary-F1=0.875`، objective=`2.087`) اندکی بهتر ماند. بنابراین recovery کاملاً تک‌ایپاکی نبود، ولی amplitude نوسان study-level هنوز قابل‌توجه است.
- snapshot epoch19: `slice MAE=2.324`، `study MAE=1.665`، `study Boundary-F1=0.899`، `keypoint=9.11px`، `selector AUC=0.887`، `selector F1=0.744` و objective=`1.923`. در مقایسه با Exp10/epoch19 (`MAE=1.769`، `Boundary-F1=0.908`، objective=`2.001`) MAE و objective بهتر و Boundary-F1 فقط حدود `0.009` پایین‌تر است. فایل `mls_multitask_epoch_019.pth` با اندازهٔ `124,890,629` بایت و timestamp جدید تأیید شد. پس late-stage نیز بخشی از سود sampler را حفظ کرده، هرچند trade-off مرزی باقی است.
- sanity epoch20: `slice MAE=2.317`، `study MAE=1.493`، `study Boundary-F1=0.936`، `keypoint=8.9px`، `selector AUC=0.901`، `selector F1=0.808` و objective=`1.671`. Exp10/epoch20 به `MAE=1.730`، `Boundary-F1=0.883` و objective=`2.014` رسیده بود؛ بنابراین Exp12 در دو epoch پیاپی 19–20 سود late-stage را حفظ و در epoch20 هم MAE و هم boundary را بهتر کرده است. snapshot21 اکنون آزمون مهم پایداری این basin است.
- snapshot epoch21: `slice MAE=2.355`، `study MAE=1.531`، `study Boundary-F1=0.923`، `keypoint=8.9px`، `selector AUC=0.891`، `selector F1=0.786` و objective=`1.740`. Exp10/epoch21 به `MAE=1.809`، `Boundary-F1=0.866` و objective=`2.133` رسیده بود؛ بنابراین snapshot late نیز بهبود هم‌زمان MAE و boundary را حفظ کرد. فایل `mls_multitask_epoch_021.pth` با اندازهٔ `124,890,629` بایت و timestamp جدید تأیید شد. سه snapshot 13، 17 و 21 proxy واضحاً بهتر ساخته‌اند؛ این تناوب چهارایپاکی جالب است، اما تا E2E نباید به قانون checkpoint جدید تبدیل شود.
- sanity epoch22: `slice MAE=2.384`، `study MAE=1.525`، `study Boundary-F1=0.932`، `keypoint=8.9px`، `selector AUC=0.896`، `selector F1=0.801` و objective=`1.712`. این نتیجه تقریباً هم‌سطح epoch21 و از نظر boundary اندکی بهتر بود؛ در همان زمان نشان می‌داد مزیت proxy late-stage به یک snapshot منفرد محدود نیست، اما نوسان epoch15 مانع ادعای پایداری کامل باقی ماند.
- snapshot epoch23: `train loss=1.6186`، `slice MAE=2.415`، `study MAE=1.511`، `study Boundary-F1=0.932`، `keypoint=8.85px`، `selector AUC=0.894`، `selector F1=0.796` و objective=`1.699`. فایل‌های `mls_multitask_epoch_023.pth` و `mls_multitask_final.pth` روی دیسک تأیید شدند؛ snapshot شمارهٔ ۲۳ همان اندازهٔ ثابت `124,890,629` بایت را دارد. ایپاک‌های ۲۱، ۲۲ و ۲۳ در proxy انتهای نسبتاً پایداری ساختند، اما E2E بعدی نشان داد این پایداری به profile production منتقل نشده است.
- آموزش Exp12 هر `23/23` ایپاک را بدون OOM، NaN یا CPU fallback کامل کرد. peak VRAM ثبت‌شدهٔ trainer حدود `4.55 GB` بود؛ مصرف دیده‌شده در سطح driver حدود `5.46 GB` و utilization حین آموزش `98%` بود. پس از آموزش، به‌جای اجرای run تازه، شش snapshot ازپیش‌ثبت‌شدهٔ 13/15/17/19/21/23 به‌صورت CUDA-only روی تمام ۶۷ مطالعهٔ fold2 ارزیابی شدند.
- ممیزی E2E هر شش snapshot با `402/402` inference موفق، failure صفر و مجموع runtime=`628.93 s` کامل شد. روی profile production قفل‌شده، Exp12/epoch15 به `MAE=1.9153`، `Boundary-F1=0.8059` و bias=`-1.5169 mm` رسید؛ Exp10/epoch15 با همان profile `MAE=1.7144` و `Boundary-F1=0.8947` داشت. پس MAE به‌اندازهٔ `0.2009 mm` یا `11.72%` بدتر و Boundary-F1 به‌اندازهٔ `0.0888` پایین‌تر شد؛ معیار اصلی ازپیش‌ثبت‌شده شکست خورد.
- بهترین Exp12 روی همان profile production epoch19 با `MAE=1.8729` و Boundary-F1=`0.8296` بود؛ این نیز از baseline Exp10/epoch15 هم در MAE (`+0.1586 mm`) و هم boundary (`-0.0651`) ضعیف‌تر است. بنابراین صرف تغییر checkpoint شکست production را حل نمی‌کند.
- یک سیگنال مثبت تشخیصی باقی ماند: grid درون-fold برای Exp12/epoch13 به `MAE=1.5142` و Boundary-F1=`0.9504` رسید. همچنین profile قدیمی و کاملاً frozen از fold0/1 روی epoch13 نتیجهٔ `MAE=1.6906` و Boundary-F1=`0.9253` داد؛ همان profile برای Exp10/epoch13 `2.1446` و `0.8887` بود. این نشان می‌دهد representation خراب نشده، اما انتخاب epoch/profile پس از دیدن fold2 unbiased نیست و مجوز production یا دو آموزش تازه نمی‌دهد.
- تحلیل featureها علت را از collapse localization جدا کرد: در epoch13، AUC جداسازی `selector_max` از `0.8378` در Exp10 به `0.8636` و AUC `heatmap_top3_mean` از `0.6239` به `0.7273` بهتر شد. شکست اصلی به calibration مقدار MLS/exposure مربوط است؛ full study balancing در epoch15 کم‌برآوردی شدید `-1.5169 mm` ساخت و profile بهینه را از severity-window به top-k منتقل کرد.
- تصمیم نهایی Exp12: `study_class_balanced` کامل جای baseline را نمی‌گیرد و روی fold0/1 گسترش داده نمی‌شود. گزارش تفصیلی: `reports/mls_experiments/mls-local-v2-exp12r1-w32-fold2-studybalanced/checkpoint_audit_report.md`؛ metricهای تجمیعی: `e2e_aggregate_metrics.json`.
- قواعد توقف دقیقاً رعایت شدند: finite بودن loss/metric، سلامت CUDA، recovery localization در epochهای 2–3 و activation مطالعه در 8–9 اجازهٔ ادامه تا schedule ثابت را دادند؛ تصمیم نهایی نیز فقط با full-study CUDA audit snapshotهای ازپیش‌ثبت‌شده و همان profile قفل‌شدهٔ Exp10 گرفته شد.
- گزارش زندهٔ خودکار: `reports/mls_experiments/mls-local-v2-exp12r1-w32-fold2-studybalanced/report.md`.
- لینک run: `https://dagshub.com/amiresbati62/BrainCtTriage.mlflow/#/experiments/16/runs/4af89dc814d8439590e69f886c30909b`

#### Experiment 13: sampler میانی با توان 0.5 — آمادهٔ اجرا

- علت اجرای این آزمایش، sweep تصادفی نیست: Exp12 در epoch13 feature separation بهتری ساخت، اما full study balancing در epoch15 calibration را خراب کرد. Exp13 فقط نقطهٔ میانی هندسی exposure را می‌سنجد تا این trade-off را causal جدا کند.
- وزن هر ردیف درون class متناسب با `1/sqrt(rows_per_study_class)` است؛ در نتیجه mass هر study متناسب با ریشهٔ تعداد ردیف‌هایش می‌شود. target/nontarget class mass همچنان دقیقاً برابر می‌ماند.
- ممیزی واقعی fold2: نسبت max/min study mass از `7.8724×` در Exp10 به `4.7193×` در Exp13 می‌رسد؛ Exp12 مقدار `2.8951×` داشت. نسبت q90/q10 نیز به‌ترتیب `4.3322×`، `3.5544×` و `2.8951×` است؛ پس candidate واقعاً میان دو سیاست قرار دارد.
- سه آزمون واحد sampler پاس شدند: حفظ وزن legacy، برابری کامل study در mode کامل و نسبت ریشه‌ای exposure در mode میانی. manifest نیز با config معتبر parse شد.
- معیار اصلی پیش از آموزش قفل شد: epoch15 روی profile production ثابت در برابر Exp10/epoch15 (`MAE=1.7144`، Boundary-F1=`0.8947`). معیار ثانویه epoch13 روی profile frozen fold0/1 است، اما به‌تنهایی مجوز گسترش به foldهای دیگر نمی‌دهد.
- plan: `reports/mls_experiments/mls-local-v2-exp13-w32-fold2-hybridsampler/PREREGISTERED_PLAN.md`؛ manifest: `config/experiments/mls-local-v2-exp13-w32-fold2-hybridsampler.yaml`.
- اجرای CUDA-only با MLflow Run ID=`0a2cf48a6fce417ba2f89c50a7ad185f` آغاز شد. GPU در شروع حدود `5.46/6 GB` حافظه، `98%` utilization و بدون OOM/fallback داشت.
- sanity epoch1: `slice MAE=78.556`، `study MAE=4.882`، Boundary-F1=`0`، keypoint=`148.8px`، selector AUC=`0.449` و objective=`7.157`. شروع localization از Exp12/epoch1 (`128.989 mm` و `254.9px`) بهتر است، ولی study gate طبق انتظار warm-up هنوز غیرفعال است؛ تصمیم به recovery epochهای 2–3 موکول می‌شود.
- sanity epoch2: `train loss=4.5779`، `slice MAE=9.930`، `study MAE=4.882`، Boundary-F1=`0`، keypoint=`117.75px`، selector AUC=`0.540` و objective=`7.111`. localization نسبت به epoch1 سریعاً recover کرد، اما selector هنوز همهٔ studyها را زیر gate عملیاتی نگه می‌داشت. هیچ OOM، NaN یا CPU fallback رخ نداد.
- sanity epoch3: `train loss=4.2002`، `slice MAE=6.265`، `study MAE=4.882`، Boundary-F1=`0`، keypoint=`33.99px` و selector AUC=`0.344`. این نقطه از Exp10/epoch3 (`3.275 mm`) و Exp12/epoch3 (`3.341 mm`) از نظر slice MAE عقب بود، ولی هر سه آزمایش در study-level روی همان مقدار ثابت قفل بودند؛ بنابراین برای توقف زودهنگام کافی نبود.
- sanity epoch4: `slice MAE=3.560`، keypoint=`14.1px` و selector AUC=`0.621`. فاصلهٔ localization با Exp10 کاهش یافت و افت AUC ایپاک قبل recover شد، ولی شرط study aggregation هنوز فعال نشده بود.
- sanity epoch5: `slice MAE=2.607`، keypoint=`10.6px` و selector AUC=`0.820`. Exp10 در همین epoch `2.407 mm` و AUC=`0.812` داشت؛ Exp13 از نظر localization اندکی عقب و از نظر ranking اندکی جلو بود و عملاً به trajectory baseline رسید.
- sanity epoch6: `slice MAE=2.468`، keypoint=`10.5px` و selector AUC=`0.829`. این نتیجه دقیقاً میان Exp10 (`2.619 / 0.836`) و Exp12 (`2.425 / 0.810`) قرار گرفت؛ در نتیجه رفتار تجربی sampler نیز با exposure میانی ازپیش‌ممیزی‌شده سازگار است.
- sanity epoch7: `slice MAE=2.647`، keypoint=`10.2px`، selector AUC=`0.844` و selector F1=`0.725`. selector از حالت all-negative خارج شد، اما gate production شامل threshold=`0.6` و حداقل سه slice فعال هنوز study prediction را باز نکرد؛ study MAE همان `4.882` باقی ماند.
- sanity epoch8: زنجیرهٔ end-to-end فعال شد: `study MAE=3.308`، Boundary-F1=`0.538`، `slice MAE=2.564`، keypoint=`9.72px`، selector AUC=`0.865` و F1=`0.778`. Exp10 در همان epoch `3.266 mm / 0.573` داشت، درحالی‌که Exp12 هنوز روی `4.882 / 0` قفل بود. بنابراین hybrid تأخیر activation سیاست fully study-balanced را رفع کرد و تقریباً به baseline رسید.
- sanity epoch9: نخستین سیگنال قوی study-level ثبت شد: `study MAE=1.549`، Boundary-F1=`0.936`، selector AUC=`0.888` و F1=`0.796`. Exp10/epoch9 به `2.110 / 0.872` و Exp12 به `2.299 / 0.764` رسیده بودند؛ پس Exp13 در proxy داخلی به‌وضوح جلو افتاد، با وجود slice MAE ضعیف‌تر (`2.741`). این جدایی با فرضیهٔ بهبود exposure در سطح study سازگار است، اما به‌دلیل in-fold بودن validation، مجوز production نیست و verdict تا E2E snapshotهای ازپیش‌ثبت‌شدهٔ epoch13/15 نگه داشته می‌شود.
- زمان‌بندی بازبینی‌شده پس از epoch9: snapshot اول epoch13 حدود ۳۰ دقیقه و checkpoint معیار اصلی epoch15 حدود ۴۲ دقیقه بعد قابل استخراج است. پایان 23 epoch و ممیزی E2E شش snapshot در حالت عادی حدود ۱.۵ تا ۲ ساعت دیگر زمان می‌برد. اگر معیار اصلی شکست بخورد شاخه بسته می‌شود؛ فقط در صورت پیروزی واقعی، دو fold تأییدی آموزش داده خواهند شد.
- sanity epoch10: `study MAE=1.714`، Boundary-F1=`0.854`، slice MAE=`2.639` و selector AUC=`0.883`. نسبت به peak9 پس‌رفت رخ داد، اما Exp10/epoch10 فقط `2.782 / 0.699` و Exp12 `1.689 / 0.898` داشتند؛ پس Exp13 همچنان نسبت به baseline قدیمی بهتر و میان دو sampler باقی ماند.
- sanity epoch11: `study MAE=1.711`، Boundary-F1=`0.867`، slice MAE=`2.415` و selector AUC=`0.907`. تکرار سطح epoch10 نشان داد پس از peak9 collapse رخ نداده و plateau حدود `1.71 mm` شکل گرفته است.
- sanity epoch12: model دوباره به ناحیهٔ قوی برگشت: `study MAE=1.549`، Boundary-F1=`0.924`، slice MAE=`2.325`، selector AUC=`0.909` و F1=`0.805`. بنابراین سود proxy epoch9 تک‌نقطه‌ای نبود و نوسان میان حدود `1.55–1.71 mm` تثبیت شد.
- snapshot epoch13: `study MAE=1.523`، Boundary-F1=`0.924`، slice MAE=`2.534`، keypoint=`9.23px`، selector AUC=`0.904` و F1=`0.807`. این نتیجه از Exp10/epoch13 (`1.947 / 0.860`) بهتر و از Exp12/epoch13 (`1.355 / 0.957`) ضعیف‌تر است. فایل `mls_multitask_epoch_013.pth` با اندازهٔ `124,890,629` بایت روی دیسک تأیید شد.
- sanity epoch14: `study MAE=1.503`، Boundary-F1=`0.925`، slice MAE=`2.239`، keypoint=`9.0px`، selector AUC=`0.906` و objective=`1.700`. بهبود epochهای 12–14 یک ناحیهٔ میانی نسبتاً پایدار ساخته است.
- snapshot معیار اصلی epoch15: `study MAE=1.526`، Boundary-F1=`0.924`، slice MAE=`2.460`، keypoint=`9.26px`، selector AUC=`0.903` و objective=`1.726`. مقایسهٔ proxy هم‌epoch به‌تنهایی پیروزی نشان نمی‌دهد: Exp10/epoch15=`1.476 / 0.922`؛ یعنی Exp13 در MAE داخلی حدود `0.050 mm` بدتر و در boundary حدود `0.002` بهتر است. فایل `mls_multitask_epoch_015.pth` با اندازهٔ `124,890,629` بایت تأیید شد. verdict production تا full-study E2E روی profile ثابت (`Exp10: MAE=1.7144, BF1=0.8947`) معلق می‌ماند.
- sanity epoch16: `study MAE=1.835`، Boundary-F1=`0.889`، slice MAE=`2.504`، keypoint=`8.86px` و selector AUC=`0.912`. نسبت به basin 12–15 افت رخ داد، اما collapse عددی یا مشکل CUDA وجود نداشت. در بخشی از epoch، GPU به‌علت `SW Power Cap/SW Thermal Slowdown` نرم‌افزاری به P5 و memory clock=`810 MHz` محدود شد؛ دما حدود `61°C` و HW thermal slowdown غیرفعال بود. محدودیت بعداً خودکار رفع و GPU به P0/6000MHz برگشت؛ فقط زمان اجرا متاثر شد.
- snapshot epoch17: `study MAE=1.728`، Boundary-F1=`0.927`، slice MAE=`2.484`، keypoint=`8.72px`، selector AUC=`0.919` و objective=`1.948`. نسبت به epoch15، MAE ضعیف‌تر ولی boundary بهتر است و بنابراین برای ممیزی E2E مکمل باقی ماند. فایل `mls_multitask_epoch_017.pth` با اندازهٔ `124,890,629` بایت تأیید شد.
- sanity epoch18: `study MAE=1.986`، Boundary-F1=`0.900`، slice MAE=`2.517` و selector AUC=`0.907`. late degradation واضح‌تر شد، ولی محاسبات finite و CUDA-only ماندند.
- epoch19 train و validation کامل شد و در `epoch_metrics.jsonl` ثبت گردید: `study MAE=1.577`، Boundary-F1=`0.935`، slice MAE=`2.404`، keypoint=`8.91px` و selector AUC=`0.921`. سپس فراخوانی `mlflow.log_metrics` به‌علت DNS failure برای `dagshub.com` exception داد. در پیاده‌سازی فعلی MLflow logging پیش از best/snapshot save قرار دارد؛ بنابراین process با exit code 1 بسته شد و snapshot epoch19 ذخیره نشد.
- deviation از برنامهٔ شش snapshot: آخرین snapshot قابل بازیابی epoch17 است و checkpointهای دوره‌ای optimizer/scheduler state ندارند. restart از epoch17 trajectory دقیق قبلی را بازسازی نمی‌کند و آزمایش تازه محسوب می‌شود. مطابق تصمیم توقف کنترل‌شدهٔ کاربر، restart آموزشی انجام نمی‌شود؛ Exp13 با وضعیت `terminated_early_external_mlflow_dns_after_epoch19_validation` بسته و E2E فقط روی snapshotهای سالم 13/15/17 اجرا می‌شود. نبود 19/21/23 باید در تفسیر نهایی به‌عنوان محدودیت صریح لحاظ شود.
- backlog فنی الزامی برای شروع آینده: checkpoint/snapshot و report باید پیش از logging شبکه‌ای ذخیره شوند؛ `mlflow.log_metrics` باید با retry/deferred queue غیرکشنده شود؛ و resume checkpoint باید model+optimizer+scheduler+scaler+epoch+RNG state را ذخیره کند. این اصلاح در این دور به اجرای آموزشی تازه منجر نمی‌شود.
- ممیزی E2E سه snapshot سالم 13/15/17 روی `201/201` مطالعه-checkpoint، صفر failure و مجموع runtime=`316.07 s` کامل شد. تمام model inferenceها CUDA-only بودند؛ CSVهای per-study محلی ماندند.
- روی profile production قفل‌شده، نتایج epochهای 13/15/17 به‌ترتیب `MAE=1.7624/1.7550/2.1050 mm` و Boundary-F1=`0.9074/0.8626/0.8672` بود. معیار اصلی epoch15 در برابر Exp10/epoch15 (`1.7144 / 0.8947`) MAE را `0.0406 mm` یا `2.37%` بدتر، Boundary-F1 را `0.0321` پایین‌تر و objective را `5.44%` ضعیف‌تر کرد؛ بنابراین Exp13 رد شد.
- معیار ثانویه frozen fold0/1 برای Exp13/epoch13 به `MAE=1.9965` و Boundary-F1=`0.9055` رسید؛ نسبت به Exp10/epoch13 (`2.1446 / 0.8887`) سیگنال مثبت دارد، ولی از Exp12/epoch13 (`1.6906 / 0.9253`) ضعیف‌تر و برای override شکست primary ناکافی است.
- بهترین diagnostic درون-fold Exp13/epoch13 با top-k7 به `MAE=1.5377` و Boundary-F1=`0.9227` رسید، اما چون profile پس از دیدن همان fold انتخاب شده production-safe نیست.
- گزارش audit، summary grid، metricهای سه E2E، تاریخچه epochها و aggregate JSON با موفقیت به MLflow run `0a2cf48a6fce417ba2f89c50a7ad185f` ارسال شدند؛ سه CSV خام prediction صریحاً exclude شدند. status run عمداً `FAILED` باقی ماند تا توقف واقعی epoch19 پنهان نشود.
- تصمیم نهایی Exp13: sampler میانی نیز جای baseline را نمی‌گیرد. مدل قابل‌دفاع فعلی همچنان Exp08/09/10، HRNet-W32 hybrid-soft، epoch15 ثابت است. طبق درخواست کاربر، هیچ آزمایش تازه‌ای آغاز نمی‌شود و کار پس از handoff نهایی pause می‌شود.

### فاز D — تصمیم single-model در برابر ensemble

- پس از تکمیل fold2، epoch15 hybrid کاندیدای اصلی single است؛ strict balanced آن `1.6273 mm` و boundary-first آن `1.6045 mm` است.
- ensemble قدیمی binary+peak-aware با strict `1.6350 mm` دیگر برتری قانع‌کننده‌ای نسبت به single hybrid ندارد و هزینهٔ دو inference را توجیه نمی‌کند.
- median/mean snapshot فقط کاندیدای پژوهشی است. قبل از هر inference چندبرابری باید با predictionهای ذخیره‌شده و انتخاب strict ثابت شود که سود آن material و transferable است.
- weight averaging درون basin 13/15/17 گزینهٔ جذاب‌تری است، چون در صورت موفقیت یک inference باقی می‌ماند؛ tensor averaging باید CUDA-only و E2E نیز GPU-only باشد.
- inference time، peak VRAM و محدودیت packageهای leaderboard قبل از lock نهایی اندازه‌گیری می‌شوند.

### فاز E — آماده‌سازی submission قابل اعتماد

1. تطبیق inference با packageهای نصب‌شدهٔ سرور leaderboard.
2. smoke test GPU-only روی محیط سازگار.
3. ممیزی ordering برش‌ها، spacing، orientation، study grouping و schema خروجی.
4. اعتبارسنجی leaderboard مجازی با تست‌های sanity و دادهٔ held-out.
5. ساخت اولین submission رسمی تنها بعد از عبور از این gateها.

### ایده‌های بعدی؛ فقط در صورت نیاز شواهدی

اولویت آزمایش‌های بعدی پس از شواهد سه‌fold، به‌ترتیب بازده مورد انتظار:

1. ارزیابی ارزان snapshot blend روی predictionهای ذخیره‌شده و سپس weight averaging محدود 13/15/17 فقط اگر سیگنال strict مثبت باشد؛
2. pooling مقاوم به calibration یا temperature calibration مشترک، با profile و checkpoint ازپیش‌قفل‌شده؛
3. selector دو-head: یک head برای وجود MLS و یک head برای نزدیکی به peak؛
4. sampler میانی با ضریب اختلاط ثابت بین slice-balanced و study-balanced؛ نسخهٔ fully study-balanced در Exp12 رد شده است؛
5. context سبک 2.5D برای کاهش ابهام تک‌برش، فقط اگر سود مورد انتظار هزینهٔ VRAM و آموزش سه‌fold را توجیه کند.

هر آزمایش باید یک متغیر اصلی را تغییر دهد تا علت بهبود قابل تشخیص بماند.

---

## 9. ریسک‌ها و عدم قطعیت‌های باز

- validation داخلی هنوز surrogate کامل metric مطالعه نیست.
- تعداد foldها محدود است و diagnostic grid می‌تواند overfit شود؛ strict LOO برای همین نگه داشته شده است.
- انتخاب checkpoint با E2E روی همان validation fold نیز می‌تواند خوش‌بینانه شود؛ snapshot selection باید محدود و از پیش تعریف‌شده باشد.
- ensemble زمان inference را تقریباً دو برابر می‌کند.
- نتیجهٔ MLS به‌تنهایی موفقیت score ترکیبی مسابقه را تضمین نمی‌کند.
- سلامت leaderboard مجازی و برابری دقیق آن با evaluator رسمی هنوز باید جداگانه اثبات شود.
- تا زمانی که submission رسمی نداریم، هیچ ادعای قطعی دربارهٔ جایگاه leaderboard صحیح نیست.

---

## 10. جمع‌بندی تصمیم فعلی

### تصمیم توقف کنترل‌شده در پایان Exp13

- به درخواست کاربر، پس از تکمیل آموزش جاری Exp13 و ممیزی E2E لازم برای معتبرکردن نتیجهٔ همین run، هیچ آموزش، fold تأییدی، sampler تازه یا آزمایش جدیدی آغاز نمی‌شود.
- کار ناتمام/goal نباید `complete` یا `blocked` علامت بخورد؛ این نقطه یک pause عملیاتی است تا در آینده، فقط اگر دقت فعلی ناکافی بود یا شواهد تازه‌ای به دست آمد، ادامه از همین گزارش و checkpointها انجام شود.
- خروجی نهایی این دور باید شامل جدول بهترین مدل‌های واقعاً موجود، تفکیک نتایج diagnostic از production-safe، وضعیت MLflow/checkpointها، آزمایش‌های ردشده و backlog اولویت‌بندی‌شده باشد.
- raw medical data و predictionهای per-study/per-slice همچنان فقط محلی می‌مانند و به MLflow ارسال نمی‌شوند.

مسیر فعلی بر پایهٔ شواهد این است:

1. HRNet-W32 را به‌عنوان backbone پایه نگه می‌داریم.
2. peak-aware را نسبت به binary یک بهبود معتبر می‌دانیم.
3. checkpoint را با full-study E2E انتخاب می‌کنیم، نه صرفاً validation objective.
4. Exp08/09/10 هر سه کامل و روی full-study ممیزی شدند؛ انتقال fold2 نشان داد hybrid بهبود واقعی و مستقل دارد.
5. کاندیدای اصلی فعلی single hybrid در epoch15 است؛ diagnostic mean=`1.5459` و برآورد strict balanced=`1.6273 mm`.
6. انتخاب آزاد checkpoint کنار گذاشته می‌شود. epoch15 قبل از submission قفل است و leaderboard برای checkpoint tuning استفاده نمی‌شود.
7. ensemble قدیمی دو-مدلی فعلاً کنار می‌رود، مگر آنکه آزمایش strict جدید سود material بالاتر از single epoch15 نشان دهد.
8. Exp12 ثابت کرد full study balancing بیش‌ازحد تهاجمی است؛ این sampler به foldهای دیگر گسترش نمی‌یابد. اگر آزمایش آموزشی دیگری اجرا شود، تنها متغیر آن ضریب اختلاط ثابت بین وزن‌های slice و study خواهد بود و profile/epoch اصلی پیشاپیش قفل می‌شود.
9. strict LOO معیار تصمیم نهایی می‌ماند؛ هیچ نتیجهٔ in-fold به‌تنهایی production lock نمی‌شود.
10. Exp13 همان ضریب میانی را آزمود و در معیار اصلی epoch15 شکست خورد؛ بنابراین شاخهٔ sampler بسته است. ادامهٔ آینده باید ابتدا دوام checkpoint/MLflow و سپس submission/evaluator را حل کند، نه sampler تازه.
11. این دور در وضعیت pause عملیاتی بسته می‌شود؛ goal ناتمام حفظ می‌شود و آغاز هر آموزش جدید نیازمند ازسرگیری صریح است.

این گزارش باید مرجع شروع دور بعد باشد؛ بنابراین تکرار تحلیل‌های قبلی لازم نیست مگر آنکه داده، split، metric رسمی یا پیاده‌سازی inference تغییر کند.
