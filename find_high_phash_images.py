"""Find LC25000 images with high pHash scores that might trigger false positives."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from utils import phash as phash_utils
import glob

# Load pHash database
db = phash_utils.load_phash_db('data/phash_casia_authentic.csv')
print(f"Loaded pHash database with {len(db)} entries\n")

# Find LC25000 images
lc25000_images = glob.glob('data/LC25000/train/**/*.jpeg', recursive=True)[:50]

print("="*90)
print("SCANNING FOR HIGH-PHASH LC25000 IMAGES")
print("="*90)
print("\nLooking for images with pHash >0.8 (old threshold) or >0.95 (new threshold)...\n")

high_phash_images = []

for img_path in lc25000_images:
    img_name = img_path.split('\\')[-1]
    
    try:
        ph = phash_utils.compute_phash(img_path)
        phash_score, best_fn, best_h = phash_utils.compute_phash_score(ph, db)
        
        if phash_score > 0.8:
            high_phash_images.append((img_name, phash_score, img_path))
            status = ">>> PROBLEM" if phash_score > 0.95 else ">>> old threshold"
            print(f"{img_name:<35} pHash: {phash_score:.4f}  {status}")
            
    except Exception as e:
        pass

print("\n" + "="*90)
print("RESULTS")
print("="*90)
print(f"Total scanned:                {len(lc25000_images)}")
print(f"With pHash >0.8 (old fail):   {len([x for x in high_phash_images if x[1] > 0.8])}")
print(f"With pHash >0.95 (new fail):  {len([x for x in high_phash_images if x[1] > 0.95])}")

if high_phash_images:
    print(f"\n*** RECOMMENDATION:")
    print(f"Found {len(high_phash_images)} images with high pHash dissimilarity.")
    print(f"This is EXPECTED - medical images differ from photo databases.")
    print(f"New threshold (0.95) reduces false positives from {len([x for x in high_phash_images if x[1] > 0.8])} to {len([x for x in high_phash_images if x[1] > 0.95])}")
    
    if len([x for x in high_phash_images if x[1] > 0.95]) > 0:
        print(f"\nFor best results, consider:")
        print(f"  1. Build LC25000-specific pHash database")
        print(f"  2. Or disable pHash checking for medical images")
        print(f"  3. Or increase threshold to 0.98")
else:
    print("\n*** GOOD NEWS: No high-pHash images found in sample!")
    print("Current thresholds should work well.")
