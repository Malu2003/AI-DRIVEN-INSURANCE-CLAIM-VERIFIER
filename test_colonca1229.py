"""Test the EXACT image from user's screenshot: colonca1229.jpeg"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from pipeline.image_module import run_image_forgery
from utils import phash as phash_utils

# The exact image from user's test
test_image = "data/LC25000_forgery/val/authentic/colonca1229.jpeg"

print("="*90)
print("TESTING YOUR EXACT IMAGE: colonca1229.jpeg")
print("="*90)
print(f"Image path: {test_image}\n")

# Test with NEW thresholds
result = run_image_forgery(test_image, model_ckpt='checkpoints/lc25000_forgery/best.pth.tar')

verdict = result.get('forgery_verdict')
cnn = result.get('cnn_score')
ela = result.get('ela_score')
phash = result.get('phash_score')
fused = result.get('fused_score')
confidence = result.get('confidence')

print("RESULTS WITH NEW THRESHOLDS:")
print("-"*90)
print(f"  Forgery Verdict:  {verdict.upper()}")
print(f"  Confidence:       {confidence}")
print(f"  CNN Score:        {cnn:.4f}")
print(f"  ELA Score:        {ela:.4f}")
print(f"  pHash Score:      {phash:.4f}")
print(f"  Fused Score:      {fused:.4f}")

print("\n" + "="*90)
print("COMPARISON WITH YOUR SCREENSHOT")
print("="*90)

print("\nYour screenshot showed:")
print("  Forgery Verdict:  TAMPERED (RED)")
print("  CNN Score:        0.460")
print("  ELA Score:        0.250")
print("  pHash Score:      0.920")
print("  Fused Score:      0.489")

print("\nNow with fixed thresholds:")
print(f"  Forgery Verdict:  {verdict.upper()}")
print(f"  CNN Score:        {cnn:.4f}")
print(f"  ELA Score:        {ela:.4f}")
print(f"  pHash Score:      {phash:.4f}")
print(f"  Fused Score:      {fused:.4f}")

# Analyze the decision
print("\n" + "="*90)
print("DECISION LOGIC ANALYSIS")
print("="*90)

print("\nOLD THRESHOLDS (caused false positive):")
print(f"  pHash high-risk threshold: >0.8")
print(f"  Your pHash score: 0.920 > 0.8 = TRUE >>> TRIGGERED TAMPERED")

print("\nNEW THRESHOLDS (should fix):")
print(f"  pHash high-risk threshold: >0.95")
print(f"  Your pHash score: {phash:.3f} > 0.95 = {phash > 0.95}")

if phash is not None and phash > 0.95:
    print(f"  >>> STILL HIGH! Consider increasing to 0.98 or disabling pHash")
elif phash is not None and phash > 0.8:
    print(f"  >>> FIXED! No longer triggers high-risk threshold")
    
print(f"\nFused score check: {fused:.3f} >= 0.5 = {fused >= 0.5}")
print(f"CNN suspicious: {cnn:.3f} > 0.35 = {cnn > 0.35}")
print(f"ELA suspicious: {ela:.3f} > 0.35 = {ela > 0.35}")
print(f"Both CNN AND ELA: {cnn > 0.35 and ela > 0.35}")

print("\n" + "="*90)
print("VERDICT")
print("="*90)

if verdict in ['authentic', 'suspicious']:
    print(f"✅ SUCCESS! Image correctly identified as {verdict.upper()}")
    print("The false positive has been FIXED!")
else:
    print(f"❌ Still showing as TAMPERED")
    print(f"This means pHash is >{0.95} or fused >= 0.5 or (CNN and ELA both suspicious)")
    print(f"Recommendation: Increase pHash threshold to 0.98 or disable pHash for medical images")

print("\n" + "="*90)
print("NEXT STEPS")
print("="*90)
print("""
1. Restart your backend: cd backend && python app.py
2. Upload the same colonca1229.jpeg image
3. It should now show as AUTHENTIC or SUSPICIOUS (not TAMPERED)
4. If still TAMPERED, run this script and share the output
""")
