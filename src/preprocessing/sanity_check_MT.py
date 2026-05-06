import numpy as np
import matplotlib.pyplot as plt

# اسم بیمار تستی خود را وارد کنید
patient_id = "271912"  # یا 447

# لود کردن فایل‌های npy فوق‌سریع
img = np.load(f"data/processed/images/{patient_id}_img.npy")
mask = np.load(f"data/processed/masks/{patient_id}_mask.npy")

# پیدا کردن اسلایسی که خونریزی دارد
z_slice = np.argmax(np.sum(mask > 0, axis=(0, 1)))

# رسم کانال‌های مختلف
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# کانال 1: Brain (لخته‌های خون داخل بافت اینجا خوب دیده میشن)
axes[0].imshow(img[0, :, :, z_slice], cmap='gray')
axes[0].set_title("Channel 1: Brain Window")

# کانال 2: Subdural (خونریزی‌های نزدیک جمجمه)
axes[1].imshow(img[1, :, :, z_slice], cmap='gray')
axes[1].set_title("Channel 2: Subdural Window")

# کانال 3: Bone (فقط استخوان شفاف است)
axes[2].imshow(img[2, :, :, z_slice], cmap='gray')
axes[2].set_title("Channel 3: Bone Window")

# ماسک سگمنتیشن روی کانال اول
axes[3].imshow(img[0, :, :, z_slice], cmap='gray')
axes[3].imshow(np.ma.masked_where(mask[:, :, z_slice] == 0, mask[:, :, z_slice]), cmap='jet', alpha=0.5)
axes[3].set_title("Brain + Segmentation Mask")

plt.show()