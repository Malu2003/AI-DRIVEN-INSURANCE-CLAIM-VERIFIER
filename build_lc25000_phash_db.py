"""
Build pHash database from authentic LC25000 images for proper medical image comparison.

This solves the domain mismatch problem properly instead of adjusting thresholds.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from utils import phash as phash_utils
import glob
import csv

print("="*90)
print("BUILDING LC25000 PHASH DATABASE")
print("="*90)
print("\nThis creates a proper reference database for medical images.")
print("pHash scores will now compare medical images to medical images (not photos).\n")

# Collect all authentic LC25000 images
print("Scanning for authentic LC25000 images...")
train_images = glob.glob('data/LC25000/train/**/*.jpeg', recursive=True)
val_authentic = glob.glob('data/LC25000_forgery/val/authentic/*.jpeg', recursive=True)

all_authentic = train_images + val_authentic
print(f"Found {len(all_authentic)} authentic medical images\n")

# Compute pHash for each
print("Computing perceptual hashes...")
phash_data = []

for i, img_path in enumerate(all_authentic, 1):
    if i % 500 == 0:
        print(f"  Processed {i}/{len(all_authentic)}...")
    
    try:
        ph = phash_utils.compute_phash(img_path)
        filename = Path(img_path).name
        phash_data.append((filename, ph))
    except Exception as e:
        print(f"  Error processing {img_path}: {e}")

print(f"\nSuccessfully computed {len(phash_data)} hashes")

# Save to CSV
output_path = 'data/phash_lc25000_authentic.csv'
print(f"\nSaving to {output_path}...")

with open(output_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['filename', 'phash'])
    writer.writerows(phash_data)

print(f"✓ Saved {len(phash_data)} entries to {output_path}")

print("\n" + "="*90)
print("NEXT STEPS")
print("="*90)
print("""
1. Update pipeline to use new database:
   - Edit: pipeline/claim_verification_pipeline.py
   - Change: phash_db from 'phash_casia_authentic.csv' to 'phash_lc25000_authentic.csv'

2. Revert pHash thresholds to original (stricter) values:
   - phash_high_risk: back to >0.8 (from 0.95)
   - phash_suspicious: back to >0.6 (from 0.85)
   
3. Test again - should work properly now!

This is the CORRECT solution (same-domain comparison).
""")
