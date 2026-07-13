import argparse
from src.pipelines.pipelines import nnunet_pipeline, yolo_pipeline, mls_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run specific Medical AI pipelines.")
    parser.add_argument("--run", type=str, required=True, 
                        choices=["nnunet", "yolo", "mls", "all"], 
                        help="Which pipeline to execute?")
    
    args = parser.parse_args()

    if args.run == "nnunet":
        print("🚀 Launching ONLY nnU-Net Pipeline...")
        nnunet_pipeline()
    elif args.run == "yolo":
        print("🚀 Launching ONLY YOLO Pipeline...")
        yolo_pipeline()
    elif args.run == "mls":
        print("🚀 Launching ONLY MLS Pipeline...")
        mls_pipeline()
    elif args.run == "all":
        print("🚀 Launching ALL Pipelines sequentially...")
        nnunet_pipeline()
        yolo_pipeline()
        mls_pipeline()