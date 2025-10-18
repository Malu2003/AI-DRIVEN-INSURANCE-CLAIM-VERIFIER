"""
Organize LC25000 dataset into proper train/test splits.
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm

def organize_lc25000(base_dir):
    base_dir = Path(base_dir)
    classes = ['colon_aca', 'colon_n', 'lung_aca', 'lung_n', 'lung_scc']
    
    # Create new directory structure
    new_train_dir = base_dir / "train_organized"
    new_test_dir = base_dir / "test_organized"
    
    for class_name in classes:
        (new_train_dir / class_name).mkdir(parents=True, exist_ok=True)
        (new_test_dir / class_name).mkdir(parents=True, exist_ok=True)
    
    # Move files from old to new structure
    print("Organizing training data...")
    for class_name in classes:
        src_dir = base_dir / "train" / class_name
        if not src_dir.exists():
            print(f"Warning: {src_dir} not found")
            continue
            
        files = list(src_dir.glob("*.jpeg")) + list(src_dir.glob("*.jpg"))
        for file in tqdm(files, desc=f"Processing {class_name}"):
            shutil.copy2(file, new_train_dir / class_name / file.name)
    
    print("\nOrganizing test data...")
    for class_name in classes:
        src_dir = base_dir / "test" / class_name
        if not src_dir.exists():
            print(f"Warning: {src_dir} not found")
            continue
            
        files = list(src_dir.glob("*.jpeg")) + list(src_dir.glob("*.jpg"))
        for file in tqdm(files, desc=f"Processing {class_name}"):
            shutil.copy2(file, new_test_dir / class_name / file.name)
    
    # Rename directories
    if (base_dir / "train").exists():
        (base_dir / "train").rename(base_dir / "train_old")
    if (base_dir / "test").exists():
        (base_dir / "test").rename(base_dir / "test_old")
        
    new_train_dir.rename(base_dir / "train")
    new_test_dir.rename(base_dir / "test")
    
    print("\nDataset organization completed!")
    
    # Print statistics
    print("\nDataset statistics:")
    for split in ["train", "test"]:
        total = 0
        print(f"\n{split.capitalize()} split:")
        for class_name in classes:
            count = len(list((base_dir / split / class_name).glob("*.jp*g")))
            total += count
            print(f"{class_name}: {count} images")
        print(f"Total: {total} images")

if __name__ == "__main__":
    organize_lc25000("data/LC25000")