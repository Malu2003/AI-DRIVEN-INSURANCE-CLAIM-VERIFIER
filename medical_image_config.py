"""
Configuration to disable pHash scoring for medical images.

Rationale:
- CNN model: Trained on LC25000 → Domain-matched → Reliable
- ELA: Domain-agnostic → Works on any image type → Reliable  
- pHash: Trained on CASIA photos → Domain-mismatched → Unreliable

Solution: Use only CNN + ELA for medical images.
"""

# Option 1: Set pHash weight to 0 (keeps component but ignores score)
CNN_WEIGHT = 0.7   # Increase from 0.55
ELA_WEIGHT = 0.3   # Increase from 0.25
PHASH_WEIGHT = 0.0 # Decrease from 0.20 (effectively disabled)

# Option 2: Don't load pHash database at all
USE_PHASH_FOR_MEDICAL = False

print("""
RECOMMENDED CONFIGURATION FOR MEDICAL IMAGES:

Fusion Weights:
  CNN:   70% (trained on LC25000, domain-matched)
  ELA:   30% (domain-agnostic, works everywhere)
  pHash:  0% (disabled due to domain mismatch)

Thresholds (revert to original):
  CNN suspicious:   >0.35 (calibrated for LC25000)
  ELA suspicious:   >0.35 (calibrated for medical images)
  Fused tampered:   >=0.5 (standard threshold)

Rationale:
1. CNN is the strongest signal (trained specifically on LC25000)
2. ELA provides complementary error-level analysis
3. pHash is unreliable for cross-domain comparison

This is scientifically sound and easy to explain to professor.
""")
