"""
test_mls_heatmap_local.py — Quick test of the MLS heatmap pipeline on CPU.

This script:
1. Loads a few samples from the dataset
2. Creates the HRNet model and runs a forward pass
3. Decodes keypoints with DARK
4. Computes MLS

Usage:
    cd D:\Projects\My projects\IAAA_Compet\IAAA_BrainCtTriage
    .venv\Scripts\python.exe scripts\test_mls_heatmap_local.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np

from src.strategies.mls_heatmap.model import HRNetHeatmapModel
from src.strategies.mls_heatmap.dataset import MLSHeatmapDataset
from src.strategies.mls_heatmap.utils import (
    decode_heatmap_dark_batch,
    compute_mls_batch,
)
from src.config import MLS_DIR


def main():
    print("=" * 60)
    print("🧪 MLS Heatmap Pipeline - Local Test")
    print("=" * 60)

    # 1. Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Device: {device}")

    # 2. Load dataset (small subset)
    csv_path = str(MLS_DIR / "mls_labels.csv")
    img_dir = str(MLS_DIR / "images")

    dataset = MLSHeatmapDataset(
        csv_path=csv_path,
        img_dir=img_dir,
        img_size=512,
        heatmap_size=128,
        heatmap_sigma=2.0,
        augment=False,
    )
    print(f"📊 Dataset size: {len(dataset)} samples")

    # Test with first 4 samples
    sample_count = min(4, len(dataset))
    batch_images = []
    batch_heatmaps = []
    batch_masks = []

    for i in range(sample_count):
        img, hm, mask = dataset[i]
        batch_images.append(img.unsqueeze(0))
        batch_heatmaps.append(hm.unsqueeze(0))
        batch_masks.append(mask.unsqueeze(0))

    images = torch.cat(batch_images, dim=0).to(device)
    heatmap_targets = torch.cat(batch_heatmaps, dim=0)
    masks = torch.cat(batch_masks, dim=0)

    print(f"📦 Batch shape: {images.shape}")

    # 3. Create model (pretrained=False for CPU speed in test)
    model = HRNetHeatmapModel(
        backbone_name="hrnet_w18",  # Use W18 for faster CPU test
        in_channels=3,
        num_keypoints=3,
        pretrained=True,
    ).to(device).eval()

    # 4. Forward pass
    with torch.no_grad():
        heatmap_pred = model(images)
    print(f"🔥 Heatmap output shape: {heatmap_pred.shape}")

    # 5. DARK decode
    coords_pred, scores = decode_heatmap_dark_batch(
        heatmap_pred.cpu(), heatmap_size=128, img_size=512
    )
    coords_true, true_scores = decode_heatmap_dark_batch(
        heatmap_targets, heatmap_size=128, img_size=512
    )

    print(f"\n📍 Keypoint Predictions (first sample):")
    kp_names = [
        "AnteriorFalxAttachment",
        "PosteriorFalxAttachment",
        "OutermostPointOfTheFalx",
    ]
    for k in range(3):
        print(f"  {kp_names[k]}:")
        print(f"    True:  ({coords_true[0, k, 0]:.1f}, {coords_true[0, k, 1]:.1f})")
        print(f"    Pred:  ({coords_pred[0, k, 0]:.1f}, {coords_pred[0, k, 1]:.1f})")
        err = np.sqrt(
            (coords_pred[0, k, 0] - coords_true[0, k, 0]) ** 2
            + (coords_pred[0, k, 1] - coords_true[0, k, 1]) ** 2
        )
        print(f"    Error: {err:.2f} px")

    # 6. Compute MLS
    spacing_x = 0.5  # approximate (DICOM real value used in production)
    mls_true = compute_mls_batch(coords_true, spacing_x)
    mls_pred = compute_mls_batch(coords_pred, spacing_x)

    print(f"\n📏 MLS Results (spacing_x={spacing_x} mm/px):")
    for i in range(sample_count):
        print(f"  Sample {i+1}: True MLS={mls_true[i]:.2f}mm, Pred MLS={mls_pred[i]:.2f}mm")

    print("\n" + "=" * 60)
    print("✅ Pipeline test complete!")
    print(f"   Predictions are random — model needs training first!")
    print("=" * 60)

    print("\n📋 Next steps for real training:")
    print("  Option A: Install CUDA PyTorch for GPU training:")
    print("    uv pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121")
    print("    --index-url https://download.pytorch.org/whl/cu121")
    print()
    print("  Option B: Train on CPU (slow, ~30min per epoch):")
    print("    .venv\\Scripts\\python.exe -c \"")
    print("from src.strategies.mls_heatmap.train import train_mls_heatmap")
    print("from src.strategies.config_models import MLSHeatmapConfig")
    print("config = MLSHeatmapConfig(backbone='hrnet_w18', epochs=2, batch_size=2)")
    print("train_mls_heatmap(config)")
    print('    "')
    print()
    print("  Option C: Deploy to cloud GPU via Streamlit UI:")
    print("    uv run streamlit run src/deploy/deployApp.py")


if __name__ == "__main__":
    main()
