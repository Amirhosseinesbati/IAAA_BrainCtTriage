import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO
import nibabel as nib

# وارد کردن معماری‌های MLS که در زمان آموزش تعریف کردیم
from src.training.mls_models import SliceSelectorModel, KeypointModel
from src.preprocessing.core.dicom_reader import BrainDicomReader
from src.config import PROJECT_ROOT

# ترفند: برای استفاده از nnU-Net در حالت Inference، باید متغیرهای محیطی را ست کنیم
BASE_DIR = PROJECT_ROOT
NNUNET_DIR = BASE_DIR / "Data" / "processed" / "nnUNet"
os.environ["nnUNet_results"] = str(NNUNET_DIR / "nnUNet_results")
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

class ICHPredictor:
    def __init__(self, device='cuda'):
        print("Initializing ICH Predictor (nnU-Net)...")
        # این مقادیر باید با مقادیر زمان آموزش شما یکی باشند
        self.predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=torch.device(device),
            verbose=False,
            
        )

        model_path = os.environ.get("ICH_MODEL_PATH")
        if not model_path:
            raise ValueError("ICH_MODEL_PATH must point to an nnU-Net trained-model folder")
        folds = tuple(
            int(value.strip())
            for value in os.getenv("ICH_FOLDS", "0").split(",")
            if value.strip()
        )
        if not folds:
            raise ValueError("ICH_FOLDS must contain at least one fold index")
        self.predictor.initialize_from_trained_model_folder(
            model_training_output_dir=model_path,
            use_folds=folds,
            checkpoint_name="checkpoint_best.pth",
        )

    def predict(self, reader):
        """
        ورودی: یک آبجکت از کلاس BrainDicomReader که load_and_sort شده باشد
        خروجی: یک دیکشنری شامل حجم هر نوع خونریزی به میلی‌لیتر (ml)
        """
        # nnU-Net به فرمت خاصی برای ورودی نیاز دارد: [[input_file], [output_file]]
        # ما یک فایل موقت برای این کار می‌سازیم
        temp_dir = BASE_DIR / "temp"
        temp_dir.mkdir(exist_ok=True)
        patient_id = reader.metadata['patient_id']
        temp_input_path = str(temp_dir / f"{patient_id}_0000.nii.gz")
        
        # ذخیره فایل NIfTI موقت (nnU-Net با آرایه مستقیم کار نمی‌کند)
        reader.save_as_nifti(temp_input_path)
        
        # اجرای پیش‌بینی
        output_file = str(temp_dir / f"{patient_id}.nii.gz")
        # ... داخل متد predict از کلاس ICHPredictor
        self.predictor.predict_from_files(
            [[temp_input_path]],          # <-- ورودی به عنوان اولین آرگومان، بدون نام
            [output_file],   # <-- نام پارامتر خروجی اصلاح شد
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1
        )
        
        # خواندن ماسک پیش‌بینی شده
        predicted_mask_path = str(temp_dir / f"{reader.metadata['patient_id']}.nii.gz")
        mask_obj = nib.load(predicted_mask_path)
        mask_data = mask_obj.get_fdata()
        
        # محاسبه حجم
        voxel_volume_ml = np.prod(mask_obj.header.get_zooms()) / 1000.0  # حجم یک واکسل به ml
        volumes = {
            "IVH": np.sum(mask_data == 1) * voxel_volume_ml,
            "IPH": np.sum(mask_data == 2) * voxel_volume_ml,
            "SDH": np.sum(mask_data == 3) * voxel_volume_ml,
            "EDH": np.sum(mask_data == 4) * voxel_volume_ml,
            "SAH": np.sum(mask_data == 5) * voxel_volume_ml
        }
        
        # پاک کردن فایل‌های موقت
        os.remove(temp_input_path)
        os.remove(predicted_mask_path)
        
        return volumes

class FracturePredictor:
    def __init__(self, model_path, device='cuda'):
        print("Initializing Fracture Predictor (YOLO)...")
        self.model = YOLO(model_path)
        self.device = device

    def predict(self, reader):
        """
        ورودی: BrainDicomReader
        خروجی: True اگر شکستگی پیدا شد، False در غیر این صورت
        """
        image_hu = reader.get_3d_volume_hu()
        
        for z in range(image_hu.shape[2]):
            slice_hu = image_hu[:, :, z]
            # فقط Bone Window را به مدل می‌دهیم
            bone_img = reader.apply_windowing(slice_hu, 1000, 400)
            bone_img_8bit = (bone_img * 255).astype('uint8')
            img_rgb = cv2.cvtColor(bone_img_8bit, cv2.COLOR_GRAY2RGB)
            
            # اجرای YOLO
            results = self.model.predict(img_rgb, device=self.device, verbose=False)
            
            # اگر در هر اسلایسی حتی یک باکس با اطمینان بالای 0.5 پیدا شد، یعنی شکستگی وجود دارد
            for res in results:
                if len(res.boxes) > 0 and res.boxes.conf[0] > 0.5:
                    return True
        return False

class MLSPredictor:
    def __init__(self, slice_model_path, kp_model_path, device='cuda'):
        print("Initializing Midline Shift Predictor (MLS)...")
        self.device = device
        
        # تابعی کوچک برای حذف پیشوند 'model.' از کلیدهای دیکشنری
        def clean_state_dict(state_dict):
            cleaned_dict = {}
            for key, value in state_dict.items():
                if key.startswith('model.'):
                    # اگر با .model شروع شد، آن 6 کاراکتر اول را حذف کن
                    cleaned_dict[key[6:]] = value 
                else:
                    cleaned_dict[key] = value
            return cleaned_dict

        # لود کردن مدل A: انتخابگر اسلایس
        self.slice_model = SliceSelectorModel()
        slice_state_dict = torch.load(slice_model_path, map_location=device)['state_dict']
        self.slice_model.load_state_dict(clean_state_dict(slice_state_dict))
        self.slice_model = self.slice_model.to(device).eval()
        
        # لود کردن مدل B: تشخیص کی‌پوینت
        self.kp_model = KeypointModel()
        kp_state_dict = torch.load(kp_model_path, map_location=device)['state_dict']
        self.kp_model.load_state_dict(clean_state_dict(kp_state_dict))
        self.kp_model = self.kp_model.to(device).eval()

    def _create_3channel_window(self, hu_image):
        ch1 = BrainDicomReader.apply_windowing(hu_image, 80, 40)
        ch2 = BrainDicomReader.apply_windowing(hu_image, 200, 80)
        ch3 = BrainDicomReader.apply_windowing(hu_image, 1000, 400)
        return np.stack([ch1, ch2, ch3], axis=0) # (3, H, W)

    def _calculate_mls(self, coords_pixels, spacing_x):
        x1, y1, x2, y2, x3, y3 = coords_pixels
        num = abs((x2 - x1)*(y1 - y3) - (x1 - x3)*(y2 - y1))
        den = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return (num / den if den > 0 else 0.0) * spacing_x

    @torch.no_grad()
    def predict(self, reader):
        """
        ورودی: BrainDicomReader
        خروجی: مقدار انحراف خط میانی به میلی‌متر (mm)
        """
        image_hu = reader.get_3d_volume_hu()
        
        # آماده‌سازی بچ کامل از اسلایس‌ها برای مدل A
        slices_256 = []
        for z in range(image_hu.shape[2]):
            slice_3ch = self._create_3channel_window(image_hu[:, :, z]) # (3, 512, 512)
            slice_tensor = torch.tensor(slice_3ch, dtype=torch.float32)
            # تغییر سایز به 256x256
            resized_tensor = F.interpolate(slice_tensor.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False)
            slices_256.append(resized_tensor)
            
        batch_tensor = torch.cat(slices_256, dim=0).to(self.device)
        
        # مرحله A: پیدا کردن بهترین اسلایس
        slice_logits = self.slice_model(batch_tensor)
        best_z = torch.argmax(slice_logits).item()
        
        # مرحله B: پیدا کردن کی‌پوینت‌ها در اسلایس برنده
        slice_512_hu = image_hu[:, :, best_z]
        slice_512_3ch = self._create_3channel_window(slice_512_hu)
        kp_input = torch.tensor(slice_512_3ch, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        coords_normalized = self.kp_model(kp_input).squeeze()
        coords_pixels = (coords_normalized * 512.0).cpu().numpy()
        
        # مرحله C: محاسبه MLS به میلی‌متر
        spacing_x = reader.metadata['spacing_x']
        mls_mm = self._calculate_mls(coords_pixels, spacing_x)
        
        return mls_mm
