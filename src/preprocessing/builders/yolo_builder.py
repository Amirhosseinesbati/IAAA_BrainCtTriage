import os
import random
import cv2
from tqdm import tqdm

from src.preprocessing.core.dicom_reader import BrainDicomReader
from src.preprocessing.core.json_parser import AnnotationParser

class YoloDatasetBuilder:
    def __init__(self, raw_dicom_dir, raw_json_dir, output_dir, split_ratio=0.8):
        self.raw_dicom_dir = raw_dicom_dir
        self.raw_json_dir = raw_json_dir
        self.output_dir = output_dir
        self.split_ratio = split_ratio
        
        self.dirs = ["images/train", "images/val", "labels/train", "labels/val"]
        for d in self.dirs:
            os.makedirs(os.path.join(self.output_dir, d), exist_ok=True)

    def _fast_scan_labels(self):
        """اسکن سریع جیسون‌ها برای پیدا کردن بیماران دارای شکستگی و سالم"""
        fractured_dict = {}
        healthy_patients = []
        
        patient_ids = [d for d in os.listdir(self.raw_dicom_dir) 
                    if os.path.isdir(os.path.join(self.raw_dicom_dir, d))]
                    
        print("Scanning annotations for split...")
        for pid in tqdm(patient_ids, desc="Scanning JSONs"):
            json_dir = os.path.join(self.raw_json_dir, pid)
            if not os.path.exists(json_dir):
                healthy_patients.append(pid)
                continue
                
            parser = AnnotationParser(json_dir)
            json_files = os.listdir(json_dir)
            
            patient_has_fracture = False
            for jf in json_files:
                # ترفند: اسم فایل دایکام را از روی جیسون می‌سازیم
                dcm_name = jf.replace('.json', '.dcm')
                data = parser.parse_slice(dcm_name)
                
                if data["bboxes"]:
                    if pid not in fractured_dict:
                        fractured_dict[pid] = {}
                    # ذخیره بر اساس نام فایل دایکام
                    fractured_dict[pid][dcm_name] = data["bboxes"]
                    patient_has_fracture = True
                    
            if not patient_has_fracture:
                healthy_patients.append(pid)
                
        return fractured_dict, healthy_patients

    def build(self):
        # 1. اسکن اولیه و گروه‌بندی
        fractured_dict, healthy_patients = self._fast_scan_labels()
        fractured_patients = list(fractured_dict.keys())
        
        # 2. تقسیم‌بندی (Train/Val Split)
        random.seed(42)
        random.shuffle(fractured_patients)
        random.shuffle(healthy_patients)
        
        f_split = int(len(fractured_patients) * self.split_ratio)
        train_patients = set(fractured_patients[:f_split] + healthy_patients[:int(len(healthy_patients)*self.split_ratio)])
        
        print(f"Fractured: {len(fractured_patients)} | Healthy: {len(healthy_patients)}")
        all_patients = fractured_patients + healthy_patients
        
        # 3. تولید تصاویر
        for pid in tqdm(all_patients, desc="Extracting YOLO Slices"):
            subset = "train" if pid in train_patients else "val"
            img_out = os.path.join(self.output_dir, "images", subset)
            lbl_out = os.path.join(self.output_dir, "labels", subset)
            
            dicom_dir = os.path.join(self.raw_dicom_dir, pid)
            try:
                reader = BrainDicomReader(dicom_dir).load_and_sort()
            except ValueError:
                continue
            
            is_fractured = pid in fractured_patients
            slices_to_process = []
            
            # منطق انتخاب نمونه‌های منفی و مثبت
            if is_fractured:
                pos_dcm_names = list(fractured_dict[pid].keys())
                all_dcm_names = [os.path.basename(s.filename) for s in reader.slices]
                neg_dcm_names = [n for n in all_dcm_names if n not in pos_dcm_names]
                
                selected_negatives = random.sample(neg_dcm_names, min(1, len(neg_dcm_names)))
                slices_to_process = pos_dcm_names + selected_negatives
            else:
                if random.random() <= 0.2: # 20 درصد شانس
                    all_dcm_names = [os.path.basename(s.filename) for s in reader.slices]
                    slices_to_process = random.sample(all_dcm_names, min(1, len(all_dcm_names)))

            # پردازش اسلایس‌های انتخاب شده
            for dcm_slice in reader.slices:
                dcm_name = os.path.basename(dcm_slice.filename)
                if dcm_name not in slices_to_process:
                    continue
                    
                # استخراج یک اسلایس و تبدیل به HU
                slope = reader.metadata["rescale_slope"]
                intercept = reader.metadata["rescale_intercept"]
                hu_img = (dcm_slice.pixel_array * slope) + intercept
                
                # اعمال Bone Window
                bone_img = reader.apply_windowing(hu_img, window_width=1000, window_level=400)
                bone_img_8bit = (bone_img * 255).astype('uint8')
                slice_rgb = cv2.cvtColor(bone_img_8bit, cv2.COLOR_GRAY2RGB)
                
                filename = f"{pid}_{dcm_name.replace('.dcm', '')}"
                cv2.imwrite(os.path.join(img_out, f"{filename}.jpg"), slice_rgb)
                
                # ذخیره لیبل‌ها
                txt_path = os.path.join(lbl_out, f"{filename}.txt")
                with open(txt_path, 'w') as f:
                    if is_fractured and dcm_name in pos_dcm_names:
                        for box in fractured_dict[pid][dcm_name]:
                            x_min, y_min, w, h = box
                            x_c, y_c = (x_min + w/2)/512, (y_min + h/2)/512
                            w_n, h_n = w/512, h/512
                            f.write(f"0 {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")

        # 4. ساخت فایل yaml
        self._create_yaml()

    def _create_yaml(self):
        yaml_content = f"path: {os.path.abspath(self.output_dir)}\ntrain: images/train\nval: images/val\nnames:\n  0: fracture"
        with open(os.path.join(self.output_dir, "dataset.yaml"), "w") as f:
            f.write(yaml_content)