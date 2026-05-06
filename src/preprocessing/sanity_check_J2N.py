import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

# اسم بیمار تستی خود را وارد کنید
patient_id = "902"  # یا 447

# لود کردن تصویر فاز 1 و ماسک فاز 2
img = nib.load(f"data/interim/images/{patient_id}.nii.gz").get_fdata()
mask = nib.load(f"data/interim/masks/{patient_id}_mask.nii.gz").get_fdata()

# پیدا کردن اسلایسی که بیشترین پیکسل خونریزی را دارد تا بتوانیم ببینیم
z_slice = np.argmax(np.sum(mask > 0, axis=(0, 1)))

plt.figure(figsize=(10, 5))

# رسم تصویر اصلی
plt.subplot(1, 2, 1)
# استفاده از ویندوی بافت نرم به صورت دستی برای بهتر دیدن
img_windowed = np.clip(img[:, :, z_slice], -20, 100) 
plt.imshow(img_windowed, cmap='gray')
plt.title(f"Original CT - Slice {z_slice}")

# رسم تصویر به همراه ماسک نیمه‌شفاف
plt.subplot(1, 2, 2)
plt.imshow(img_windowed, cmap='gray')
# ماسک را با رنگ قرمز (alpha=0.5) روی آن میندازیم
mask_slice = mask[:, :, z_slice]
plt.imshow(np.ma.masked_where(mask_slice == 0, mask_slice), cmap='autumn', alpha=0.5)
plt.title("CT + Hemorrhage Mask")

plt.show()