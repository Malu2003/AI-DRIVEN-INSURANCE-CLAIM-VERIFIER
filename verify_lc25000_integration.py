"""
Verification Script: LC25000 Fine-tuned Model Integration Check
================================================================

This script verifies that the LC25000 fine-tuned model is properly integrated
into the pipeline for testing.

Checks:
1. Fine-tuned checkpoint exists
2. Model architecture compatibility
3. Pipeline configuration
4. Inference capability with sample image
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torchvision import models
import torch.nn as nn

def check_checkpoint_exists():
    """Check if LC25000 fine-tuned checkpoint exists"""
    print("\n" + "="*70)
    print("1. CHECKING FINE-TUNED CHECKPOINT")
    print("="*70)
    
    ckpt_dir = PROJECT_ROOT / "checkpoints" / "lc25000_forgery"
    best_ckpt = ckpt_dir / "best.pth.tar"
    
    if not ckpt_dir.exists():
        print("❌ FAIL: Checkpoint directory not found:", ckpt_dir)
        return False
    
    if not best_ckpt.exists():
        print("❌ FAIL: Best checkpoint not found:", best_ckpt)
        return False
    
    print("✅ PASS: Checkpoint directory exists")
    print(f"   Location: {ckpt_dir}")
    print(f"   Best checkpoint: {best_ckpt.name}")
    
    # List all checkpoints
    checkpoints = list(ckpt_dir.glob("*.pth.tar"))
    print(f"\n   Available checkpoints ({len(checkpoints)}):")
    for ckpt in sorted(checkpoints):
        size_mb = ckpt.stat().st_size / (1024 * 1024)
        print(f"      - {ckpt.name} ({size_mb:.1f} MB)")
    
    return True


def check_model_architecture():
    """Check model architecture and number of classes"""
    print("\n" + "="*70)
    print("2. CHECKING MODEL ARCHITECTURE")
    print("="*70)
    
    best_ckpt = PROJECT_ROOT / "checkpoints" / "lc25000_forgery" / "best.pth.tar"
    
    try:
        ckpt = torch.load(best_ckpt, map_location='cpu')
        print("✅ PASS: Checkpoint loaded successfully")
        
        # Check metadata
        if 'epoch' in ckpt:
            print(f"   Training epoch: {ckpt['epoch']}")
        if 'best_f1' in ckpt:
            print(f"   Best F1 score: {ckpt['best_f1']:.4f}")
        if 'classes' in ckpt:
            classes = ckpt['classes']
            print(f"   Number of classes: {len(classes)}")
            print(f"   Classes: {classes}")
        
        # Check state dict
        state_dict = ckpt.get('state_dict', ckpt)
        
        # Check classifier layer
        classifier_keys = [k for k in state_dict.keys() if 'classifier' in k]
        print(f"\n   Classifier layers: {len(classifier_keys)}")
        for key in classifier_keys:
            shape = state_dict[key].shape
            print(f"      {key}: {shape}")
        
        # Detect number of output classes from final layer
        final_layer = 'classifier.1.weight' if 'classifier.1.weight' in state_dict else 'classifier.weight'
        if final_layer in state_dict:
            num_classes = state_dict[final_layer].shape[0]
            print(f"\n   ✅ Model output classes: {num_classes}")
            
            if num_classes == 2:
                print("      ✅ CORRECT: Model has 2 classes (binary: authentic/tampered)")
                print("      This is perfect for forgery detection!")
                return True  # Success case
            elif num_classes == 5:
                print("      ⚠️  WARNING: Model has 5 classes (LC25000 tissue types)")
                print("      This is NOT suitable for forgery detection!")
                print("      Expected: 2 classes for authentic/tampered classification")
                return False
            else:
                print(f"      ⚠️  UNEXPECTED: Model has {num_classes} classes")
                return False
        else:
            print("   ❌ FAIL: Could not find classifier layer in state dict")
            return None
            
    except Exception as e:
        print(f"❌ FAIL: Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return None


def check_pipeline_configuration():
    """Check if pipeline is configured to use LC25000 model"""
    print("\n" + "="*70)
    print("3. CHECKING PIPELINE CONFIGURATION")
    print("="*70)
    
    # Check default model path in image_module.py
    image_module_path = PROJECT_ROOT / "pipeline" / "image_module.py"
    
    if not image_module_path.exists():
        print("❌ FAIL: image_module.py not found")
        return False
    
    with open(image_module_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for default checkpoint path
    if 'checkpoints/casia/best.pth.tar' in content:
        print("⚠️  WARNING: Pipeline still uses CASIA checkpoint as default")
        print("   Default path: checkpoints/casia/best.pth.tar")
        print("   Fine-tuned path: checkpoints/lc25000_forgery/best.pth.tar")
        print("\n   🔧 RECOMMENDATION: Update pipeline to use LC25000 checkpoint")
        return False
    elif 'checkpoints/lc25000_forgery/best.pth.tar' in content:
        print("✅ PASS: Pipeline configured to use LC25000 fine-tuned model")
        return True
    else:
        print("⚠️  WARNING: Could not determine default checkpoint path")
        return None


def check_inference_capability():
    """Test inference with sample image"""
    print("\n" + "="*70)
    print("4. CHECKING INFERENCE CAPABILITY")
    print("="*70)
    
    # Check if sample images exist
    sample_dirs = [
        PROJECT_ROOT / "data" / "LC25000" / "train" / "colon_aca",
        PROJECT_ROOT / "data" / "LC25000" / "train" / "lung_aca",
        PROJECT_ROOT / "data" / "LC25000_forgery" / "train" / "authentic",
        PROJECT_ROOT / "data" / "LC25000_forgery" / "train" / "tampered",
    ]
    
    sample_image = None
    for sample_dir in sample_dirs:
        if sample_dir.exists():
            images = list(sample_dir.glob("*.jpeg")) + list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png"))
            if images:
                sample_image = images[0]
                print(f"✅ PASS: Found sample image")
                print(f"   Image: {sample_image.relative_to(PROJECT_ROOT)}")
                break
    
    if not sample_image:
        print("⚠️  WARNING: No sample images found for testing")
        print("   Inference test skipped")
        return None
    
    # Try to run inference
    try:
        from inference.image_forgery_score import compute_cnn_score
        best_ckpt = str(PROJECT_ROOT / "checkpoints" / "lc25000_forgery" / "best.pth.tar")
        
        print("\n   Running inference test...")
        score = compute_cnn_score(str(sample_image), model_ckpt=best_ckpt, tampered_index=1, device='cpu')
        
        if score is not None:
            print(f"   ✅ PASS: Inference successful")
            print(f"   Forgery score: {score:.4f}")
            return True
        else:
            print("   ❌ FAIL: Inference returned None")
            return False
            
    except Exception as e:
        print(f"   ❌ FAIL: Inference error: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_summary(results):
    """Print final verification summary"""
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    all_pass = all(r is True for r in results.values() if r is not None)
    
    for check, result in results.items():
        status = "✅ PASS" if result is True else "⚠️  WARNING" if result is None else "❌ FAIL"
        print(f"{status}: {check}")
    
    print("\n" + "="*70)
    
    if all_pass:
        print("✅ ALL CHECKS PASSED - READY FOR TESTING!")
        print("="*70)
        return True
    else:
        print("⚠️  ISSUES FOUND - REVIEW RECOMMENDATIONS ABOVE")
        print("="*70)
        
        # Provide specific guidance
        if not results.get('pipeline_configuration'):
            print("\n🔧 CRITICAL: Pipeline needs to be updated to use LC25000 model")
            print("\nTo fix:")
            print("1. Update pipeline/image_module.py default checkpoint path")
            print("2. Change: 'checkpoints/casia/best.pth.tar'")
            print("3. To:     'checkpoints/lc25000_forgery/best.pth.tar'")
        
        if results.get('model_architecture') == False:
            print("\n🔧 CRITICAL: Model has wrong architecture!")
            print("\nYour fine-tuned model may have wrong number of classes")
            print("Forgery detection needs 2 classes (authentic/tampered)")
            print("\nYou need to:")
            print("1. Check train_lc25000_forgery.py for correct 2-class training")
            print("2. Re-run fine-tuning with forgery dataset (authentic vs tampered)")
            print("3. NOT tissue classification (5 classes)")
        
        return False


def main():
    """Run all verification checks"""
    print("\n" + "="*70)
    print("LC25000 FINE-TUNED MODEL INTEGRATION VERIFICATION")
    print("="*70)
    print("This script checks if your fine-tuned model is properly integrated")
    print("into the pipeline for end-to-end testing.")
    print("="*70)
    
    results = {}
    
    # Run checks
    results['checkpoint_exists'] = check_checkpoint_exists()
    results['model_architecture'] = check_model_architecture()
    results['pipeline_configuration'] = check_pipeline_configuration()
    results['inference_capability'] = check_inference_capability()
    
    # Print summary
    ready_for_testing = print_summary(results)
    
    return 0 if ready_for_testing else 1


if __name__ == "__main__":
    sys.exit(main())
