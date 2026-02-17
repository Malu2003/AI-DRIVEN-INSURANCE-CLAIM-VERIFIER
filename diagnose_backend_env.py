"""Diagnose backend environment - check if PyTorch is actually available."""
import sys
print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("Python path:")
for p in sys.path:
    print(f"  {p}")

print("\nTrying to import torch...")
try:
    import torch
    print(f"✓ torch {torch.__version__} loaded successfully")
except Exception as e:
    print(f"✗ Failed to import torch: {e}")

print("\nTrying to import torchvision...")
try:
    import torchvision
    print(f"✓ torchvision {torchvision.__version__} loaded successfully")
except Exception as e:
    print(f"✗ Failed to import torchvision: {e}")

print("\nTrying Flask imports...")
try:
    from flask import Flask
    print("✓ Flask imported successfully")
except Exception as e:
    print(f"✗ Failed to import Flask: {e}")

print("\nTrying inference imports...")
try:
    from inference import image_forgery_score as infer
    print("✓ inference.image_forgery_score imported successfully")
    
    # Check if torch is available inside the module
    print(f"  torch available in module: {infer.torch is not None}")
except Exception as e:
    print(f"✗ Failed to import inference: {e}")
    import traceback
    traceback.print_exc()

print("\nTrying to import pipeline...")
try:
    from pipeline.claim_verification_pipeline import ClaimVerificationPipeline
    print("✓ ClaimVerificationPipeline imported successfully")
except Exception as e:
    print(f"✗ Failed to import pipeline: {e}")
    import traceback
    traceback.print_exc()
