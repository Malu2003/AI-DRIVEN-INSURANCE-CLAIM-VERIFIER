"""
Prepare LC25000 dataset for forgery detection fine-tuning.
Organizes authentic and forge images into 2-class structure.
"""
import os
import shutil
from pathlib import Path
from tqdm import tqdm
import random

def prepare_forgery_dataset(
    lc25000_dir="data/LC25000",
    output_dir="data/LC25000_forgery",
    val_ratio=0.15,
    balance_ratio=1.0,  # Ratio of authentic:tampered (1.0 = 1:1, 2.0 = 2:1, etc.)
    seed=42
):
    """
    Organize LC25000 authentic and forge images into 2-class dataset.
    
    Input structure:
        LC25000/
            train/
                colon_aca/, colon_n/  (authentic colon)
                lung_aca/, lung_n/, lung_scc/  (authentic lung)
            forge_train/
                forge_colon/  (tampered colon)
                forge_lung/   (tampered lung)
    
    Parameters:
        balance_ratio: Ratio of authentic to tampered images
            1.0 = 1:1 (balanced, best for learning)
            2.0 = 2:1 (slight imbalance, realistic)
            3.0 = 3:1 (moderate imbalance, realistic)
    
    Output structure:
        LC25000_forgery/
            train/
                authentic/  (subset of real LC25000 images)
                tampered/   (all forge images)
            val/
                authentic/
                tampered/
    """
    random.seed(seed)
    
    lc25000_path = Path(lc25000_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    for split in ['train', 'val']:
        for cls in ['authentic', 'tampered']:
            (output_path / split / cls).mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("LC25000 FORGERY DATASET PREPARATION")
    print("=" * 80)
    
    # =====================================================================
    # STEP 1: Collect authentic images (colon + lung)
    # =====================================================================
    print("\n[1/4] Collecting authentic images...")
    authentic_images = []
    
    # Colon authentic (colon_aca + colon_n)
    colon_classes = ['colon_aca', 'colon_n']
    for cls in colon_classes:
        cls_path = lc25000_path / "train" / cls
        if cls_path.exists():
            imgs = list(cls_path.glob("*.jpeg")) + list(cls_path.glob("*.jpg")) + list(cls_path.glob("*.png"))
            authentic_images.extend(imgs)
            print(f"   Found {len(imgs)} images in {cls}")
    
    # Lung authentic (lung_aca + lung_n + lung_scc)
    lung_classes = ['lung_aca', 'lung_n', 'lung_scc']
    for cls in lung_classes:
        cls_path = lc25000_path / "train" / cls
        if cls_path.exists():
            imgs = list(cls_path.glob("*.jpeg")) + list(cls_path.glob("*.jpg")) + list(cls_path.glob("*.png"))
            authentic_images.extend(imgs)
            print(f"   Found {len(imgs)} images in {cls}")
    
    print(f"\n   Total authentic images: {len(authentic_images)}")
    
    # =====================================================================
    # STEP 2: Collect forge/tampered images
    # =====================================================================
    print("\n[2/4] Collecting forge images...")
    tampered_images = []
    
    # Forge colon
    forge_colon_path = lc25000_path / "forge_train" / "forge_colon"
    if forge_colon_path.exists():
        imgs = list(forge_colon_path.glob("*.jpeg")) + list(forge_colon_path.glob("*.jpg")) + list(forge_colon_path.glob("*.png"))
        tampered_images.extend(imgs)
        print(f"   Found {len(imgs)} forge colon images")
    
    # Forge lung
    forge_lung_path = lc25000_path / "forge_train" / "forge_lung"
    if forge_lung_path.exists():
        imgs = list(forge_lung_path.glob("*.jpeg")) + list(forge_lung_path.glob("*.jpg")) + list(forge_lung_path.glob("*.png"))
        tampered_images.extend(imgs)
        print(f"   Found {len(imgs)} forge lung images")
    
    print(f"\n   Total tampered images: {len(tampered_images)}")
    
    # =====================================================================
    # BALANCE AUTHENTIC TO MATCH TAMPERED RATIO
    # =====================================================================
    print(f"\n   Balancing dataset (ratio={balance_ratio})...")
    target_authentic_count = int(len(tampered_images) * balance_ratio)
    
    if target_authentic_count < len(authentic_images):
        print(f"   Subsampling authentic: {len(authentic_images)} → {target_authentic_count}")
        authentic_images = random.sample(authentic_images, target_authentic_count)
    else:
        print(f"   Using all {len(authentic_images)} authentic images (less than target {target_authentic_count})")
    
    print(f"   Balanced authentic count: {len(authentic_images)}")
    
    # =====================================================================
    # STEP 3: Split into train/val
    # =====================================================================
    print(f"\n[3/4] Splitting into train/val (val_ratio={val_ratio})...")
    
    # Shuffle both lists
    random.shuffle(authentic_images)
    random.shuffle(tampered_images)
    
    # Calculate split sizes
    auth_val_size = int(len(authentic_images) * val_ratio)
    tamp_val_size = int(len(tampered_images) * val_ratio)
    
    # Split authentic
    auth_val = authentic_images[:auth_val_size]
    auth_train = authentic_images[auth_val_size:]
    
    # Split tampered
    tamp_val = tampered_images[:tamp_val_size]
    tamp_train = tampered_images[tamp_val_size:]
    
    print(f"   Authentic: {len(auth_train)} train, {len(auth_val)} val")
    print(f"   Tampered:  {len(tamp_train)} train, {len(tamp_val)} val")
    
    # =====================================================================
    # STEP 4: Copy files to organized structure
    # =====================================================================
    print("\n[4/4] Copying files...")
    
    # Copy train authentic
    print(f"   Copying {len(auth_train)} authentic train images...")
    for img_path in tqdm(auth_train, desc="Train Authentic"):
        shutil.copy2(img_path, output_path / "train" / "authentic" / img_path.name)
    
    # Copy train tampered
    print(f"   Copying {len(tamp_train)} tampered train images...")
    for img_path in tqdm(tamp_train, desc="Train Tampered"):
        shutil.copy2(img_path, output_path / "train" / "tampered" / img_path.name)
    
    # Copy val authentic
    print(f"   Copying {len(auth_val)} authentic val images...")
    for img_path in tqdm(auth_val, desc="Val Authentic"):
        shutil.copy2(img_path, output_path / "val" / "authentic" / img_path.name)
    
    # Copy val tampered
    print(f"   Copying {len(tamp_val)} tampered val images...")
    for img_path in tqdm(tamp_val, desc="Val Tampered"):
        shutil.copy2(img_path, output_path / "val" / "tampered" / img_path.name)
    
    # =====================================================================
    # SUMMARY
    # =====================================================================
    print("\n" + "=" * 80)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 80)
    print(f"\nOutput directory: {output_path}")
    print(f"\nDataset statistics:")
    print(f"  Train authentic: {len(list((output_path / 'train' / 'authentic').glob('*.*')))}")
    print(f"  Train tampered:  {len(list((output_path / 'train' / 'tampered').glob('*.*')))}")
    print(f"  Val authentic:   {len(list((output_path / 'val' / 'authentic').glob('*.*')))}")
    print(f"  Val tampered:    {len(list((output_path / 'val' / 'tampered').glob('*.*')))}")
    print(f"\nTotal images:     {len(authentic_images) + len(tampered_images)}")
    print(f"\nReady for fine-tuning! Run:")
    print(f"  python train_lc25000_forgery.py")
    print("=" * 80)


if __name__ == "__main__":
    prepare_forgery_dataset(balance_ratio=1.0)  # 1:1 ratio (3000 authentic + 3000 tampered)
