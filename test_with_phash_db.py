#!/usr/bin/env python3
"""Re-test with pHash database populated"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.image_module import run_image_forgery

# Test with the same tampered image
tampered_image = str(PROJECT_ROOT / "data/CASIA2/tampered/Tp_D_CND_S_N_txt00028_txt00006_10848.jpg")

print("RE-TESTING WITH PHASH DATABASE POPULATED:")
print("="*80)

result = run_image_forgery(tampered_image, output_dir=str(PROJECT_ROOT / "test_images"))
phash_score = result['phash_score']
phash_display = f"{phash_score:.4f}" if phash_score is not None else "N/A"

print("\nRESULT:")
print(f"CNN Score:        {result['cnn_score']:.4f}")
print(f"ELA Score:        {result['ela_score']:.4f}")
print(f"pHash Score:      {phash_display}")
print(f"Fused Score:      {result['fused_score']:.4f}")
print(f"\nForgery Verdict:  {result['forgery_verdict'].upper()}")
print(f"Confidence:       {result['confidence']}")

# Decision logic analysis
print("\n" + "="*80)
print("DECISION LOGIC ANALYSIS:")
print("="*80)

cnn_score = result['cnn_score']
ela_score = result['ela_score']
fused_score = result['fused_score']

cnn_suspicious = cnn_score > 0.25
ela_suspicious = ela_score > 0.15
phash_high_risk = phash_score is not None and phash_score < 0.3
phash_suspicious = phash_score is not None and phash_score < 0.5
suspicious_count = sum([cnn_suspicious, ela_suspicious, phash_suspicious])

print(f"\nThreshold Checks:")
print(f"  CNN > 0.25?              {str(cnn_suspicious):5} (score: {cnn_score:.4f})")
print(f"  ELA > 0.15?              {str(ela_suspicious):5} (score: {ela_score:.4f})")
print(f"  pHash < 0.3?             {str(phash_high_risk):5} (score: {phash_display})")
print(f"  pHash < 0.5?             {str(phash_suspicious):5} (score: {phash_display})")
print(f"  Suspicious Count:        {suspicious_count}")

print(f"\nTAMPERED Condition: (cnn_suspicious AND ela_suspicious) = {cnn_suspicious and ela_suspicious}")
