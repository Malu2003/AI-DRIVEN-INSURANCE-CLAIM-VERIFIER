#!/usr/bin/env python3
"""Test backend startup"""
import sys
import os

os.chdir(r"D:\tech_squad\AI-Driven-Image-Forgery-Detection")
sys.path.insert(0, ".")

print("Testing backend imports...")
try:
    from flask import Flask
    print("✓ Flask imported")
except Exception as e:
    print(f"✗ Flask import failed: {e}")

try:
    from pipeline.claim_verification_pipeline import ClaimVerificationPipeline
    print("✓ ClaimVerificationPipeline imported")
except Exception as e:
    print(f"✗ ClaimVerificationPipeline import failed: {e}")
    import traceback
    traceback.print_exc()

try:
    print("Initializing ClaimVerificationPipeline...")
    pipeline = ClaimVerificationPipeline()
    print("✓ Pipeline initialized successfully")
except Exception as e:
    print(f"✗ Pipeline initialization failed: {e}")
    import traceback
    traceback.print_exc()
