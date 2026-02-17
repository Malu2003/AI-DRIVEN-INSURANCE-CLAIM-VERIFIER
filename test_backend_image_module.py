#!/usr/bin/env python3
"""Test image module directly to verify CNN/ELA/pHash"""
import sys
sys.path.insert(0, '.')

from pipeline.image_module import ImageForgeryModule

print("Testing ImageForgeryModule with authentic LC25000 image...")
m = ImageForgeryModule()
result = m.run('data/LC25000/train/colon_aca/colonca1.jpeg')

print(f"\n{'='*60}")
print(f"Verdict: {result['forgery_verdict']}")
print(f"CNN Score: {result['cnn_score']}")
print(f"ELA Score: {result['ela_score']}")
print(f"pHash Score: {result['phash_score']}")
print(f"Fused Score: {result['fused_score']}")
print(f"Confidence: {result['confidence']}")
print(f"Explanation: {result['explanation']}")
print(f"{'='*60}")
