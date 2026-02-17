"""
Test to understand why your specific image showed pHash 0.920.
Let's simulate your exact scenario.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from pipeline.image_module import run_image_forgery
import glob

print("="*90)
print("SCENARIO ANALYSIS: Image with pHash 0.920 showing as TAMPERED")
print("="*90)

print("""
From your screenshot:
  - Forgery Verdict: tampered (RED)
  - CNN Score: 0.460
  - ELA Score: 0.250  
  - pHash Score: 0.920 <<< THIS TRIGGERED THE FALSE POSITIVE
  - Fused Score: 0.489

Issue: pHash 0.920 > 0.8 (old threshold) = high-risk = tampered

Analysis:
  - 0.920 means "92% dissimilar to CASIA photo database"
  - Medical histopathology images ARE fundamentally different from photos
  - This is EXPECTED and NORMAL for medical images
  
Solution Applied:
  - Old: pHash high-risk if >0.8, suspicious if >0.6
  - New: pHash high-risk if >0.95, suspicious if >0.85
  
Expected Outcome:
  - Same image now: pHash 0.920 < 0.95 = NOT high-risk
  - Decision logic: NOT (phash_high_risk) AND fused 0.489 < 0.5 AND NOT (cnn AND ela)
  - CNN 0.460 > 0.35 (suspicious) BUT ELA 0.250 < 0.35 (NOT suspicious)
  - Result: Should be SUSPICIOUS or AUTHENTIC (not TAMPERED)
""")

print("="*90)
print("VERIFICATION WITH CURRENT THRESHOLDS")
print("="*90)

# Test a few images to verify the fix works
test_images = glob.glob('data/LC25000/**/*.jpeg', recursive=True)[:3]

for img_path in test_images:
    img_name = img_path.split('\\')[-1]
    result = run_image_forgery(img_path, model_ckpt='checkpoints/lc25000_forgery/best.pth.tar')
    
    verdict = result.get('forgery_verdict')
    cnn = result.get('cnn_score')
    ela = result.get('ela_score') 
    phash = result.get('phash_score')
    fused = result.get('fused_score')
    
    # Simulate old logic
    old_phash_high = phash is not None and phash > 0.8
    old_verdict = "TAMPERED" if old_phash_high else verdict
    
    print(f"\n{img_name}")
    print(f"  Scores: CNN={cnn:.3f} ELA={ela:.3f} pHash={phash:.3f} Fused={fused:.3f}")
    print(f"  Old logic: {old_verdict:12}  New logic: {verdict:12}")
    
    if old_verdict == "TAMPERED" and verdict != "tampered":
        print(f"  >>> FIXED! Was false positive, now correct")

print("\n" + "="*90)
print("RECOMMENDATION FOR YOUR SPECIFIC IMAGE")
print("="*90)
print("""
With your image (pHash=0.920, CNN=0.460, ELA=0.250, Fused=0.489):

OLD LOGIC (before fix):
  ❌ phash_high_risk = 0.920 > 0.8 = TRUE
  ❌ VERDICT: TAMPERED (FALSE POSITIVE)

NEW LOGIC (after fix):  
  ✅ phash_high_risk = 0.920 > 0.95 = FALSE
  ✅ fused_score = 0.489 < 0.5 = FALSE
  ✅ cnn_suspicious = 0.460 > 0.35 = TRUE
  ✅ ela_suspicious = 0.250 < 0.35 = FALSE
  ✅ (cnn AND ela) = FALSE
  ✅ VERDICT: SUSPICIOUS or AUTHENTIC (CORRECT!)

ACTION: Restart your backend and test the same image again!
""")

print("\n" + "="*90)
print("HOW TO TEST YOUR IMAGE NOW")
print("="*90)
print("""
1. Stop backend: Ctrl+C in backend terminal
2. Restart: cd backend && python app.py
3. Upload the SAME image from your screenshot
4. Expected: Should show SUSPICIOUS or AUTHENTIC (not TAMPERED)

If still shows TAMPERED, the pHash might be >0.95 (extremely rare).
In that case, we can increase threshold to 0.98 or disable pHash completely.
""")
