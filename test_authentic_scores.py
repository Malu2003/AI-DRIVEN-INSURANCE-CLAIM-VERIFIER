"""Check what scores real authentic images get."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from inference import image_forgery_score as infer
from utils import ela as ela_utils, phash as phash_utils
import glob

# Find authentic images from LC25000
authentic = glob.glob('data/LC25000/train/**/*.jpeg', recursive=True)[:5]

if not authentic:
    print("No authentic images found")
    sys.exit(1)

print("Testing REAL AUTHENTIC LC25000 images:")
print("="*80)
print(f"{'Image':<35} {'CNN':<8} {'ELA':<8} {'pHash':<8} {'Fused':<8}")
print("="*80)

for img in authentic:
    name = img.split('\\')[-1][:33]
    try:
        cnn = infer.compute_cnn_score(img, model_ckpt='checkpoints/lc25000_forgery/best.pth.tar', tampered_index=1)
        
        diff = ela_utils.compute_ela(img, quality=90, scale=10)
        ela = ela_utils.compute_ela_score(diff)
        
        ph = phash_utils.compute_phash(img)
        db = phash_utils.load_phash_db('data/phash_casia_authentic.csv')
        phash, _, _ = phash_utils.compute_phash_score(ph, db)
        
        fused = infer.fuse_scores(cnn, ela, phash, weights=(0.5, 0.3, 0.2))
        
        print(f"{name:<35} {cnn:>7.4f} {ela:>7.4f} {phash:>7.4f} {fused:>7.4f}")
    except Exception as e:
        print(f"{name:<35} ERROR: {str(e)[:40]}")

print("\n" + "="*80)
print("For reference, thresholds are:")
print("  CNN suspicious:     > 0.25")
print("  ELA suspicious:     > 0.15")
print("  pHash high-risk:    > 0.80")
print("  Fused >= 0.5:       TAMPERED")
print("  CNN AND ELA both:   TAMPERED")
print("\n❌ Problem: Authentic images are hitting the thresholds!")
