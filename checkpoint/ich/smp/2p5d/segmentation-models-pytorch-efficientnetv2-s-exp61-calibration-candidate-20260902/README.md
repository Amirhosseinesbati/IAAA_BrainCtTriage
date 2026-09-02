# ICH Exp61 calibration candidate

این پوشه در نقطهٔ توقف ۲۰۲۶-۰۹-۰۲ از اینستنس Vast `49378919` به سیستم محلی
منتقل شد تا با destroy شدن سرور، بهترین کاندید تک‌foldِ شاخهٔ EfficientNetV2-S
از بین نرود.

- معماری: `segmentation_models_pytorch`، Unet++ با encoder
  `tu-efficientnetv2_rw_s`، ورودی 2.5D و شش خروجی segmentation.
- اجرا: `exp61_effnetv2s_hardempty001_fprvolselect_schema4_calonly_f2`.
- split: outer fold 2 کنار گذاشته شد و فقط calibration fold 1 برای انتخاب
  checkpoint استفاده شد؛ outer evaluation انجام نشده است.
- epoch منتخب: 9.
- checkpoint score: `0.586668`؛ selection: `0.666162`؛ mean Dice: `0.459106`.
- Any-ICH AUC: `0.923387`؛ macro subtype AUC: `0.910920`.
- normal FPR در 0.1 mL: `0.194444`؛ presence F1: `0.882353`.
- total-volume MAE/bias: `10.7627 / -6.2364 mL`.
- SHA256 وزن: `5f304018340e88e1d858d0842e432d8a163c96ed26117796c7068b6d399c18d4`.
- SHA256 manifest: `0455e6d24590a652b324c58730d81750d675b8c8d2442e67c12f11c16531ec37`.

این مدل «accepted OOF» یا leaderboard-validated نیست. ارزش آن، incumbent پژوهشی
برای ادامهٔ آزمایش‌های calibration و temporal subtype است. فایل‌های JSON و CSV
همراه وزن، provenance و تاریخچهٔ کامل انتخاب را نگه می‌دارند.

