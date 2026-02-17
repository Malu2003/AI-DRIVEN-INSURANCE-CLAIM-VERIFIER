"""
Quick test script to check if an image is tampered or genuine
using the actual DenseNet model (not demo mode).
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.image_module import run_image_forgery

def test_image(image_path: str):
    """Test a single image for forgery detection."""
    print("=" * 80)
    print(f"Testing Image: {image_path}")
    print("=" * 80)
    
    result = run_image_forgery(
        image_path=image_path,
        output_dir="test_output"
    )
    
    print("\n📊 RESULTS:")
    print("-" * 80)
    print(f"Success: {result.get('success')}")
    print(f"Forgery Verdict: {result.get('forgery_verdict').upper()}")
    print(f"Confidence: {result.get('confidence')}")
    print()
    print(f"CNN Score: {result.get('cnn_score'):.4f}")
    print(f"ELA Score: {result.get('ela_score'):.4f}")
    print(f"pHash Score: {result.get('phash_score')}")
    print(f"Fused Score: {result.get('fused_score'):.4f}")
    print()
    print(f"Explanation: {result.get('explanation')}")
    print("=" * 80)
    
    # Check if demo mode
    if result.get('cnn_score', 0) == 0.0 and result.get('forgery_verdict') == 'error':
        print("\n❌ ERROR: Image processing failed")
    elif 0.20 <= result.get('cnn_score', 0) <= 0.49:
        print("\n⚠️  WARNING: Likely running in DEMO MODE (scores look random)")
        print("   Real DenseNet model not active.")
    else:
        print("\n✅ Using REAL CNN MODEL")
    
    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_image_detection.py <image_path>")
        print()
        print("Example:")
        print('  python test_image_detection.py "test_images/sample.jpg"')
        sys.exit(1)
    
    image_path = sys.argv[1]
    test_image(image_path)
