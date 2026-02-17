#!/usr/bin/env python3
"""
Test script to analyze a tampered image through the forgery detection pipeline.
Shows detailed scores, decision logic, and identifies why misclassification occurs.
"""

import sys
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.image_module import run_image_forgery

def analyze_tampered_image(image_path):
    """Analyze image and show detailed decision logic."""
    
    print("=" * 80)
    print("TAMPERED IMAGE ANALYSIS")
    print("=" * 80)
    print(f"\nImage Path: {image_path}")
    print(f"Image Exists: {Path(image_path).exists()}")
    
    # Run through image forgery module
    print("\n" + "-" * 80)
    print("Running image forgery detection...")
    print("-" * 80)
    
    result = run_image_forgery(image_path, output_dir=str(PROJECT_ROOT / "test_images"))
    cnn_score = result['cnn_score']
    ela_score = result['ela_score']
    phash_score = result['phash_score']
    fused_score = result['fused_score']
    phash_display = f"{phash_score:.4f}" if phash_score is not None else "N/A"
    
    print("\n✓ Analysis Complete!\n")
    
    # Display all scores
    print("=" * 80)
    print("SCORES BREAKDOWN")
    print("=" * 80)
    print(f"CNN Score:        {cnn_score:.4f}")
    print(f"ELA Score:        {ela_score:.4f}")
    print(f"pHash Score:      {phash_display}")
    print(f"Fused Score:      {fused_score:.4f}")
    
    # Decision logic analysis
    print("\n" + "=" * 80)
    print("DECISION LOGIC ANALYSIS")
    print("=" * 80)
    
    # Check individual thresholds
    cnn_suspicious = cnn_score > 0.25
    ela_suspicious = ela_score > 0.15
    phash_high_risk = phash_score is not None and phash_score < 0.3
    phash_suspicious = phash_score is not None and phash_score < 0.5
    
    suspicious_count = sum([cnn_suspicious, ela_suspicious, phash_suspicious])
    
    print(f"\nThreshold Checks:")
    print(f"  CNN > 0.25?       {cnn_suspicious:5} (score: {cnn_score:.4f})")
    print(f"  ELA > 0.15?       {ela_suspicious:5} (score: {ela_score:.4f})")
    print(f"  pHash < 0.3?      {phash_high_risk:5} (score: {phash_display})")
    print(f"  pHash < 0.5?      {phash_suspicious:5} (score: {phash_display})")
    print(f"  Suspicious Count: {suspicious_count}")
    
    print(f"\nCondition Evaluations:")
    print(f"  phash_high_risk:                    {phash_high_risk}")
    print(f"  fused_score >= 0.45:                {fused_score >= 0.45} (fused: {fused_score:.4f})")
    print(f"  (cnn_suspicious AND ela_suspicious): {cnn_suspicious and ela_suspicious}")
    print(f"  → TAMPERED verdict triggered?       {phash_high_risk or fused_score >= 0.45 or (cnn_suspicious and ela_suspicious)}")
    
    print(f"\n  suspicious_count >= 2:              {suspicious_count >= 2}")
    print(f"  fused_score >= 0.25:                {fused_score >= 0.25} (fused: {fused_score:.4f})")
    print(f"  phash_suspicious:                   {phash_suspicious}")
    print(f"  → SUSPICIOUS verdict triggered?    {suspicious_count >= 2 or fused_score >= 0.25 or phash_suspicious}")
    
    # Final verdict
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    print(f"Forgery Verdict:  {result['forgery_verdict'].upper()}")
    print(f"Confidence:       {result['confidence']}")
    print(f"Explanation:      {result['explanation']}")
    
    # Print full result
    print("\n" + "=" * 80)
    print("FULL RESULT JSON")
    print("=" * 80)
    print(json.dumps(result, indent=2))
    
    return result

if __name__ == "__main__":
    # Test with the uploaded tampered image
    image_path = str(PROJECT_ROOT / "test_images" / "tampered_image.jpg")
    
    try:
        result = analyze_tampered_image(image_path)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
