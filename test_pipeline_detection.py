"""Test the actual pipeline to see if images are detected correctly now."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.claim_verification_pipeline import ClaimVerificationPipeline
import os

# Find test images
import glob
test_images = glob.glob("batch_test_results/**/*.jpg", recursive=True)

if not test_images:
    print("❌ No test images found in batch_test_results/")
    sys.exit(1)

print(f"Found {len(test_images)} test images")

# Take 2: one that should be authentic (blur)  and one that should be tampered (compression)
authentic_candidate = [img for img in test_images if 'blur' in img][0] if any('blur' in img for img in test_images) else test_images[0]
tampered_candidate = [img for img in test_images if 'compression' in img][0] if any('compression' in img for img in test_images) else test_images[1]

print(f"\n{'='*70}")
print("🔍 TESTING PIPELINE IMAGE DETECTION")
print(f"{'='*70}")
print(f"Authentic candidate: {authentic_candidate}")
print(f"Tampered candidate:  {tampered_candidate}")

# Initialize pipeline (now with LC25000 model by default)
pipeline = ClaimVerificationPipeline()

print(f"\nModel checkpoint:  {pipeline.model_ckpt}")
print(f"Checkpoint exists: {os.path.exists(pipeline.model_ckpt)}")

# Test image function directly  
from pipeline.image_module import run_image_forgery

print(f"\n{'='*70}")
print("Testing AUTHENTIC candidate (blur image)")
print(f"{'='*70}")

result1 = run_image_forgery(authentic_candidate, model_ckpt=pipeline.model_ckpt)
print(f"\nResult:")
print(f"  Verdict:      {result1.get('forgery_verdict')}")
print(f"  Fused Score:  {result1.get('fused_score')}")
print(f"  CNN Score:    {result1.get('cnn_score')}")
print(f"  ELA Score:    {result1.get('ela_score')}")
print(f"  pHash Score:  {result1.get('phash_score')}")
print(f"  Confidence:   {result1.get('confidence')}")

print(f"\n{'='*70}")
print("Testing TAMPERED candidate (compression image)")
print(f"{'='*70}")

result2 = run_image_forgery(tampered_candidate, model_ckpt=pipeline.model_ckpt)
print(f"\nResult:")
print(f"  Verdict:      {result2.get('forgery_verdict')}")
print(f"  Fused Score:  {result2.get('fused_score')}")
print(f"  CNN Score:    {result2.get('cnn_score')}")
print(f"  ELA Score:    {result2.get('ela_score')}")
print(f"  pHash Score:  {result2.get('phash_score')}")
print(f"  Confidence:   {result2.get('confidence')}")

print(f"\n{'='*70}")
print("✅ ANALYSIS")
print(f"{'='*70}")

# Check if verdicts make sense
auth_correct = result1.get('forgery_verdict') in ['authentic', 'suspicious']  # blur might be suspicious
tamp_correct = result2.get('forgery_verdict') in ['tampered', 'suspicious']  # compression should be tampered

print(f"Blur (should be authentic): {result1.get('forgery_verdict')} {'✅' if auth_correct else '❌'}")
print(f"Compression (should be tampered): {result2.get('forgery_verdict')} {'✅' if tamp_correct else '❌'}")

if auth_correct and tamp_correct:
    print(f"\n✅ PIPELINE WORKING CORRECTLY!")
else:
    print(f"\n❌ PIPELINE NOT WORKING - SCORES ARE WRONG")
