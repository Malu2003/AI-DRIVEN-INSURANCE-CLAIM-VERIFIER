"""Debug CNN inference to see exact error."""
import sys
from pathlib import Path
import os

# Setup paths
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Test CNN inference directly
from inference import image_forgery_score as infer

# Test with a real image from batch_test_results
test_image = "batch_test_results/blur/colonca1.jpg"

if not os.path.exists(test_image):
    print(f"❌ Test image not found: {test_image}")
    print("\nLooking for test images...")
    import glob
    test_images = glob.glob("batch_test_results/**/*.jpg", recursive=True)
    if test_images:
        test_image = test_images[0]
        print(f"✅ Using: {test_image}")
    else:
        print("No test images found. Using a real LC25000 image...")
        test_images = glob.glob("data/LC25000/train/**/*.jpeg", recursive=True)
        if test_images:
            test_image = test_images[0]
            print(f"✅ Using: {test_image}")
        else:
            print("❌ No images found at all!")
            sys.exit(1)

model_ckpt = "checkpoints/lc25000_forgery/best.pth.tar"

print(f"\n{'='*70}")
print("🔍 DEBUGGING CNN INFERENCE")
print(f"{'='*70}")
print(f"Test Image:    {test_image}")
print(f"Model Ckpt:    {model_ckpt}")
print(f"Ckpt Exists:   {os.path.exists(model_ckpt)}")
print(f"Image Exists:  {os.path.exists(test_image)}")
print(f"Ckpt Size:     {os.path.getsize(model_ckpt) / 1024 / 1024:.1f} MB")

print(f"\n{'='*70}")
print("🚀 ATTEMPTING CNN INFERENCE...")
print(f"{'='*70}\n")

try:
    score = infer.compute_cnn_score(
        test_image,
        model_ckpt=model_ckpt,
        tampered_index=1
    )
    print(f"\n✅ SUCCESS!")
    print(f"CNN Score (tampering probability): {score:.4f}")
    print(f"Interpretation: {'TAMPERED' if score > 0.5 else 'AUTHENTIC'}")
except Exception as e:
    print(f"\n❌ ERROR DURING INFERENCE:")
    print(f"Type: {type(e).__name__}")
    print(f"Message: {e}")
    print(f"\nFull Traceback:")
    import traceback
    traceback.print_exc()
