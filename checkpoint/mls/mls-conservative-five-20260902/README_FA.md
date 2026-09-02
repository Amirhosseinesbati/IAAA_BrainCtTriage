# مدل ترکیبی محافظه‌کارانه MLS (2026-09-02)

این پوشه نمایندهٔ بهترین candidate فعلی MLS برای بستهٔ مسابقه است. این مدل یک
checkpoint منفرد نیست؛ خروجی سه عضو fold با median ترکیب می‌شود:

- fold0: 90٪ خروجی regression مدل Exp16 + 10٪ خروجی regression مدل Exp19/epoch21
- fold1: 90٪ خروجی regression مدل Exp09/epoch15 + 10٪ خروجی regression مدل Exp18/epoch21
- fold2: خروجی Exp15r/epoch17 بدون blend

هر پنج checkpoint با نام‌های استاندارد بسته در همین پوشه حاضرند. چهار وزن قبلی
به‌صورت hardlink از پوشه‌های آزمایش خودشان قرار گرفته‌اند تا فضای تکراری مصرف
نشود؛ `fold1_regression.pth` همان Exp18/epoch21 است که مستقیماً از Vast منتقل
شد. مسیر منبع، نقش و SHA-256 دقیق هر جزء در `PACKAGE_COMPONENTS.json` و نتیجهٔ
اعتبارسنجی محلی در `LOCAL_INTEGRITY_STATUS.json` ثبت شده است.

ممیزی CUDA روی تمام 204 مطالعهٔ OOF بدون failure انجام شد و هر هفت gate parity
پاس شدند. نتیجهٔ candidate برابر MAE=1.461521959 mm، Boundary-F1=0.855888430 و
objective=1.749745100 بود؛ baseline متناظر MAE=1.472591075، Boundary-F1=0.850206612
و objective=1.772177852 داشت.

این candidate از نظر داخلی پذیرفته شده، اما هنوز با submission رسمی leaderboard
اثبات نشده است. Exp18 و Exp19 component-only هستند و نباید به‌تنهایی به‌عنوان
مدل release استفاده شوند. Exp20 به علت شکست gate وارد این بسته نشده است.

ZIP کامل قابل بازسازی روی سرور با SHA-256 زیر ساخته و ممیزی شده است:

`660770225b53e5389ba0e8dde70cc7e1a65f732ca854887aba6ba8deff1d490b`
