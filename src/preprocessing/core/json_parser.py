import os
import glob
import json
import numpy as np

class AnnotationParser:
    def __init__(self, json_dir):
        """
        json_dir: مسیر پوشه جیسون‌های یک بیمار (مثلا Data/raw/annotations/272179)
        """
        self.json_dir = json_dir
        # از آنجا که فایل‌های JSON همنام با دایکام نیستند (گاهی)، بر اساس Z_index مرتبشان میکنیم
        # ولی فرض مسابقه این است که ترتیب فایل‌ها با ترتیب دایکام‌ها (بعد از سورت) همخوانی دارد.
        # پس ما لیست نام فایل‌ها را از بیرون (هنگام اجرا) می‌گیریم.

    @staticmethod
    def decode_rle(counts, shape=(512, 512)):
        """تبدیل RLE به ماسک"""
        if not counts:
            return np.zeros(shape, dtype=np.uint8)
        values = counts[0::2]
        lengths = counts[1::2]
        mask_flat = np.repeat(values, lengths).astype(np.uint8)
        mask_2d = mask_flat.reshape(shape, order='C')
        return mask_2d

    def parse_slice(self, dcm_filename):
        """خواندن اطلاعات یک اسلایس خاص بر اساس نام فایل دایکام آن"""
        json_filename = dcm_filename.replace('.dcm', '.json')
        json_path = os.path.join(self.json_dir, json_filename)
        
        result = {
            "has_label": False,
            "mask_2d": np.zeros((512, 512), dtype=np.uint8),
            "keypoints": {},
            "bboxes": []
        }
        
        if os.path.exists(json_path):
            result["has_label"] = True
            with open(json_path, 'r') as jf:
                data = json.load(jf)
                
                # استخراج ماسک خونریزی
                if "segmentation_rle" in data and "counts" in data["segmentation_rle"]:
                    result["mask_2d"] = self.decode_rle(
                        data["segmentation_rle"]["counts"], 
                        shape=data["segmentation_rle"]["shape"]
                    )
                
                # استخراج نقاط خط میانی
                if "keypoints" in data and data["keypoints"]:
                    for kp_name, coords in data["keypoints"].items():
                        if coords: # اگر لیست خالی نبود
                            result["keypoints"][kp_name] = coords
                
                # استخراج باکس‌های شکستگی
                if "boxes_xywh" in data and data["boxes_xywh"]:
                    result["bboxes"] = data["boxes_xywh"]
                    
        return result