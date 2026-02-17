"""
Prepare CASIA2 dataset for training by organizing into train/val splits.
Authentic images go into class 0, tampered images into class 1.
"""

import os
import shutil
from pathlib import Path
import random
from tqdm import tqdm

def prepare_casia_splits(casia_dir, val_ratio=0.1, seed=42):
    random.seed(seed)
    
    # Setup paths
    casia_dir = Path(casia_dir)
    authentic_dir = casia_dir / "authentic"
    tampered_dir = casia_dir / "tampered"
    
    train_dir = casia_dir / "train"
    val_dir = casia_dir / "val"
    
    # Create train/val directory structure
    for split_dir in [train_dir, val_dir]:
        (split_dir / "authentic").mkdir(parents=True, exist_ok=True)
        (split_dir / "tampered").mkdir(parents=True, exist_ok=True)
    
    # Process authentic images
    authentic_images = list(authentic_dir.glob("*.jp*g")) + list(authentic_dir.glob("*.tif*"))
    random.shuffle(authentic_images)
    val_size = int(len(authentic_images) * val_ratio)
    
    print(f"Processing {len(authentic_images)} authentic images...")
    for idx, img_path in enumerate(tqdm(authentic_images)):
        target_dir = val_dir if idx < val_size else train_dir
        shutil.copy2(img_path, target_dir / "authentic" / img_path.name)
    
    # Process tampered images
    tampered_images = list(tampered_dir.glob("*.jp*g")) + list(tampered_dir.glob("*.tif*"))
    random.shuffle(tampered_images)
    val_size = int(len(tampered_images) * val_ratio)
    
    print(f"Processing {len(tampered_images)} tampered images...")
    for idx, img_path in enumerate(tqdm(tampered_images)):
        target_dir = val_dir if idx < val_size else train_dir
        shutil.copy2(img_path, target_dir / "tampered" / img_path.name)
    
    # Print statistics
    print("\nDataset statistics:")
    print(f"Train authentic: {len(list((train_dir / 'authentic').glob('*.*')))}")
    print(f"Train tampered: {len(list((train_dir / 'tampered').glob('*.*')))}")
    print(f"Val authentic: {len(list((val_dir / 'authentic').glob('*.*')))}")
    print(f"Val tampered: {len(list((val_dir / 'tampered').glob('*.*')))}")

if __name__ == "__main__":
    prepare_casia_splits("data/CASIA2")