from src.preprocessing.builders.nnunet_builder import NNUnetDatasetBuilder
from src.preprocessing.builders.yolo_builder import YoloDatasetBuilder
from src.preprocessing.builders.mls_builder import MlsDatasetBuilder

RAW_DICOM = "Data/raw/training"
RAW_JSON = "Data/raw/annotations"

def main():
    print("=== Central Data Pipeline Started ===")
    
    # 1. آماده‌سازی دیتاسِت nnU-Net (برای 5 نوع خونریزی)
    NNUNET_RAW = "Data/processed/nnUNet/nnUNet_raw"
    nnunet_builder = NNUnetDatasetBuilder(RAW_DICOM, RAW_JSON, NNUNET_RAW)
    nnunet_builder.build()
    
    # 2. آماده‌سازی دیتاسِت YOLO (برای شکستگی جمجمه)
    # YOLO_OUT = "Data/processed/yolo_fracture"
    # yolo_builder = YoloDatasetBuilder(RAW_DICOM, RAW_JSON, YOLO_OUT)
    # yolo_builder.build()
    
    # # 3. آماده‌سازی دیتاسِت MLS (برای تشخیص خط میانی)
    # MLS_OUT = "Data/processed/mls_dataset"
    # mls_builder = MlsDatasetBuilder(RAW_DICOM, RAW_JSON, MLS_OUT)
    # mls_builder.build()
    
    print("\n=== All Datasets Prepared Successfully! You are ready to Train! ===")

if __name__ == "__main__":
    main()