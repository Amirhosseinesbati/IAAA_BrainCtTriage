import os
import cv2
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.preprocessing.core.dicom_reader import BrainDicomReader
from src.preprocessing.core.json_parser import AnnotationParser

class MlsDatasetBuilder:
    def __init__(self, raw_dicom_dir, raw_json_dir, output_dir):
        self.raw_dicom_dir = raw_dicom_dir
        self.raw_json_dir = raw_json_dir
        self.output_dir = output_dir
        
        # پوشه تصاویر
        self.out_img_dir = os.path.join(self.output_dir, "images")
        os.makedirs(self.out_img_dir, exist_ok=True)
        
        # نام نقاط کلیدی که برای محاسبه MLS الزامی هستند
        self.target_kps = [
            'AnteriorFalxAttachment', 
            'PosteriorFalxAttachment', 
            'OutermostPointOfTheFalx'
        ]

    def _create_3channel_window(self, hu_image):
        """ساخت تصویر ۳ کاناله از مقادیر HU"""
        # کانال 1: Brain (W:80, L:40)
        ch1 = BrainDicomReader.apply_windowing(hu_image, 80, 40)
        # کانال 2: Subdural (W:200, L:80)
        ch2 = BrainDicomReader.apply_windowing(hu_image, 200, 80)
        # کانال 3: Bone (W:1000, L:400)
        ch3 = BrainDicomReader.apply_windowing(hu_image, 1000, 400)
        
        # روی هم قرار دادن کانال‌ها و تبدیل به 0-255 (فرمت RGB)
        img_rgb = np.stack([ch1, ch2, ch3], axis=-1) * 255.0
        return img_rgb.astype(np.uint8)

    def build(self):
        print("Building MLS Dataset (Slice Selector & Keypoints)...")
        patient_ids = [d for d in os.listdir(self.raw_dicom_dir) 
                    if os.path.isdir(os.path.join(self.raw_dicom_dir, d))]
        
        csv_data = []
        
        for pid in tqdm(patient_ids, desc="Processing MLS Slices"):
            json_dir = os.path.join(self.raw_json_dir, pid)
            dicom_dir = os.path.join(self.raw_dicom_dir, pid)
            
            if not os.path.exists(json_dir):
                continue
                
            parser = AnnotationParser(json_dir)
            
            # اسکن سریع برای پیدا کردن اسلایس‌های هدف (Positive)
            positive_slices = []
            json_files = os.listdir(json_dir)
            for jf in json_files:
                dcm_name = jf.replace('.json', '.dcm')
                data = parser.parse_slice(dcm_name)
                
                # بررسی اینکه آیا هر 3 نقطه وجود دارند؟
                kps = data["keypoints"]
                if all(kp in kps for kp in self.target_kps):
                    positive_slices.append({
                        "dcm_name": dcm_name,
                        "points": kps
                    })
            
            # اگر این بیمار هیچ اسلایس هدفی برای MLS نداشت، از او رد می‌شویم
            if not positive_slices:
                continue
                
            # لود کردن دایکام‌ها برای استخراج تصاویر
            try:
                reader = BrainDicomReader(dicom_dir).load_and_sort()
            except ValueError:
                continue

            pos_dcm_names = [p["dcm_name"] for p in positive_slices]
            all_dcm_names = [os.path.basename(s.filename) for s in reader.slices]
            neg_dcm_names = [n for n in all_dcm_names if n not in pos_dcm_names]
            
            # انتخاب 2 اسلایس منفی تصادفی برای آموزش مدل Slice Selector (جلوگیری از عدم تعادل کلاس‌ها)
            selected_negatives = random.sample(neg_dcm_names, min(2, len(neg_dcm_names)))
            
            slices_to_process = pos_dcm_names + selected_negatives
            
            for dcm_slice in reader.slices:
                dcm_name = os.path.basename(dcm_slice.filename)
                
                if dcm_name not in slices_to_process:
                    continue
                    
                # 1. تبدیل به HU و اعمال 3 ویندوی تخصصی
                slope = reader.metadata["rescale_slope"]
                intercept = reader.metadata["rescale_intercept"]
                hu_img = (dcm_slice.pixel_array * slope) + intercept
                
                img_3ch = self._create_3channel_window(hu_img)
                
                # OpenCV تصاویر را BGR ذخیره می‌کند، پس باید RGB را به BGR تبدیل کنیم تا رنگ‌ها جابجا نشوند
                img_bgr = cv2.cvtColor(img_3ch, cv2.COLOR_RGB2BGR)
                
                img_filename = f"{pid}_{dcm_name.replace('.dcm', '.png')}"
                out_path = os.path.join(self.out_img_dir, img_filename)
                cv2.imwrite(out_path, img_bgr)
                
                # 2. ذخیره اطلاعات در CSV
                is_target = 1 if dcm_name in pos_dcm_names else 0
                row_data = {
                    "patient_id": pid,
                    "image_name": img_filename,
                    "is_target": is_target,
                    "x1": 0, "y1": 0, "x2": 0, "y2": 0, "x3": 0, "y3": 0
                }
                
                if is_target:
                    # پیدا کردن مختصات از لیست positive_slices
                    kp_data = next(item for item in positive_slices if item["dcm_name"] == dcm_name)["points"]
                    row_data["x1"], row_data["y1"] = kp_data['AnteriorFalxAttachment']
                    row_data["x2"], row_data["y2"] = kp_data['PosteriorFalxAttachment']
                    row_data["x3"], row_data["y3"] = kp_data['OutermostPointOfTheFalx']
                    
                csv_data.append(row_data)

        # 3. ذخیره فایل CSV نهایی
        df = pd.DataFrame(csv_data)
        csv_path = os.path.join(self.output_dir, "mls_labels.csv")
        df.to_csv(csv_path, index=False)
        print(f"MLS Dataset successfully built! Data saved to {self.output_dir}")