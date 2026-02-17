"""Test authentic LC25000 images that previously showed as tampered due to high pHash."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from pipeline.image_module import run_image_forgery
import glob

# Find authentic LC25000 images
authentic_images = glob.glob('data/LC25000/train/**/*.jpeg', recursive=True)[:10]

if not authentic_images:
    print("❌ No LC25000 images found")
    sys.exit(1)

print("="*80)
print("TESTING AUTHENTIC LC25000 IMAGES WITH NEW PHASH THRESHOLDS")
print("="*80)
print("\nOld thresholds: pHash high-risk >0.8, suspicious >0.6")
print("New thresholds: pHash high-risk >0.95, suspicious >0.85")
print("\nReason: Medical images naturally differ from photo databases (CASIA)\n")
print("="*80)

results = []
for img_path in authentic_images[:5]:  # Test first 5
    img_name = img_path.split('\\')[-1][:30]
    
    try:
        result = run_image_forgery(
            img_path,
            model_ckpt='checkpoints/lc25000_forgery/best.pth.tar'
        )
        
        verdict = result.get('forgery_verdict')
        cnn = result.get('cnn_score')
        ela = result.get('ela_score')
        phash = result.get('phash_score')
        fused = result.get('fused_score')
        
        # Check if it would have been wrong with old thresholds
        old_phash_high_risk = phash is not None and phash > 0.8
        old_verdict = "tampered" if old_phash_high_risk else verdict
        
        status = "[OK]" if verdict in ['authentic', 'suspicious'] else "[FAIL]"
        changed = ">>> FIXED!" if (old_verdict == "tampered" and verdict != "tampered") else ""
        
        print(f"\n{img_name:<32}")
        print(f"  CNN: {cnn:>6.4f}  ELA: {ela:>6.4f}  pHash: {phash:>6.4f}  Fused: {fused:>6.4f}")
        print(f"  Old verdict: {old_verdict:12}  New verdict: {verdict:12} {status} {changed}")
        
        results.append({
            'name': img_name,
            'verdict': verdict,
            'phash': phash,
            'correct': verdict in ['authentic', 'suspicious']
        })
        
    except Exception as e:
        print(f"\n{img_name:<32} ERROR: {str(e)[:40]}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

correct_count = sum(1 for r in results if r['correct'])
high_phash_count = sum(1 for r in results if r['phash'] and r['phash'] > 0.8)
very_high_phash_count = sum(1 for r in results if r['phash'] and r['phash'] > 0.95)

print(f"Total tested:                    {len(results)}")
print(f"Correctly identified:            {correct_count}/{len(results)}")
print(f"Images with pHash >0.8:          {high_phash_count} (would fail with old threshold)")
print(f"Images with pHash >0.95:         {very_high_phash_count} (fail with new threshold)")

if correct_count == len(results):
    print("\n*** SUCCESS! All authentic images correctly identified!")
else:
    print(f"\n*** WARNING: {len(results) - correct_count} images still incorrectly flagged")
    print("Consider building a medical image pHash database for better accuracy")
