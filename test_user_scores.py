#!/usr/bin/env python3
"""Test with user's exact scores to verify decision logic"""

# User's scores for a TAMPERED image
cnn_score = 0.182
ela_score = 0.201
phash_score = 0.921
fused_score = 0.167

print("TESTING WITH USER'S SCORES (KNOWN TAMPERED IMAGE):")
print("="*80)
print(f"CNN Score:        {cnn_score:.4f}")
print(f"ELA Score:        {ela_score:.4f}")
print(f"pHash Score:      {phash_score:.4f}")
print(f"Fused Score:      {fused_score:.4f}")

# Apply decision logic
cnn_suspicious = cnn_score > 0.25
ela_suspicious = ela_score > 0.15
phash_high_risk = phash_score > 0.8  # FIXED: HIGH phash = dissimilar = suspicious
phash_suspicious = phash_score > 0.6  # FIXED
suspicious_count = sum([cnn_suspicious, ela_suspicious, phash_suspicious])

print("\n" + "="*80)
print("DECISION LOGIC:")
print("="*80)
print(f"\nThreshold Checks:")
print(f"  CNN > 0.25?              {str(cnn_suspicious):5} (score: {cnn_score:.4f})")
print(f"  ELA > 0.15?              {str(ela_suspicious):5} (score: {ela_score:.4f})")
print(f"  pHash > 0.8?             {str(phash_high_risk):5} (score: {phash_score:.4f})")
print(f"  pHash > 0.6?             {str(phash_suspicious):5} (score: {phash_score:.4f})")
print(f"  Suspicious Count:        {suspicious_count}")

print(f"\nTAMPERED Condition Checks:")
print(f"  1. phash_high_risk?                   {phash_high_risk}")
print(f"  2. fused_score >= 0.5?                {fused_score >= 0.5} (fused: {fused_score:.4f})")
print(f"  3. (cnn AND ela suspicious)?          {cnn_suspicious and ela_suspicious}")
print(f"  => ANY TRUE?                          {phash_high_risk or fused_score >= 0.5 or (cnn_suspicious and ela_suspicious)}")

if phash_high_risk or fused_score >= 0.5 or (cnn_suspicious and ela_suspicious):
    verdict = "TAMPERED"
elif suspicious_count >= 2 or fused_score >= 0.3 or phash_suspicious:
    verdict = "SUSPICIOUS"
else:
    verdict = "AUTHENTIC"

print(f"\n" + "="*80)
print(f"FINAL VERDICT: {verdict}")
print("="*80)

if verdict == "TAMPERED" or verdict == "SUSPICIOUS":
    print("\n✓ CORRECT! Tampered image detected.")
else:
    print("\n✗ WRONG! Should be TAMPERED or SUSPICIOUS.")
