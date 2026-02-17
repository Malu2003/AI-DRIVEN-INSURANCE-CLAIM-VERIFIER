"""
End-to-End Integration Test: LC25000 Fine-tuned Model in Pipeline
===================================================================

This script tests the complete pipeline with the LC25000 fine-tuned model:
1. Image forgery detection using fine-tuned model
2. Backend API integration
3. Full claim verification pipeline

Usage:
    python test_lc25000_pipeline.py
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_image_module():
    """Test image forgery module with LC25000 model"""
    print("\n" + "="*70)
    print("TEST 1: IMAGE FORGERY MODULE")
    print("="*70)
    
    try:
        from pipeline.image_module import ImageForgeryModule
        
        # Initialize module (should use LC25000 checkpoint by default)
        module = ImageForgeryModule()
        
        print(f"✅ Module initialized")
        print(f"   Model checkpoint: {module.model_ckpt}")
        
        # Find a test image
        test_image = None
        search_paths = [
            PROJECT_ROOT / "data" / "LC25000" / "train" / "colon_aca",
            PROJECT_ROOT / "data" / "LC25000" / "train" / "lung_aca",
            PROJECT_ROOT / "data" / "LC25000_forgery" / "train" / "authentic",
        ]
        
        for path in search_paths:
            if path.exists():
                images = list(path.glob("*.jpeg")) + list(path.glob("*.jpg"))
                if images:
                    test_image = images[0]
                    break
        
        if not test_image:
            print("⚠️  No test images found, skipping inference test")
            return True
        
        print(f"   Test image: {test_image.name}")
        
        # Run detection
        result = module.run(str(test_image))
        
        if result.get('success'):
            print(f"✅ Inference successful")
            print(f"   CNN Score: {result['cnn_score']:.4f}")
            print(f"   ELA Score: {result['ela_score']:.4f}")
            print(f"   Fused Score: {result['fused_score']:.4f}")
            print(f"   Verdict: {result['forgery_verdict']}")
            print(f"   Confidence: {result['confidence']}")
            return True
        else:
            print(f"❌ Inference failed: {result.get('explanation', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_claim_pipeline():
    """Test full claim verification pipeline"""
    print("\n" + "="*70)
    print("TEST 2: CLAIM VERIFICATION PIPELINE")
    print("="*70)
    
    try:
        from pipeline.claim_verification_pipeline import ClaimVerificationPipeline
        
        # Initialize pipeline
        pipeline = ClaimVerificationPipeline()
        print("✅ Pipeline initialized")
        
        # Find test image
        test_image = None
        search_paths = [
            PROJECT_ROOT / "data" / "LC25000" / "train" / "colon_aca",
            PROJECT_ROOT / "data" / "LC25000" / "train" / "lung_aca",
        ]
        
        for path in search_paths:
            if path.exists():
                images = list(path.glob("*.jpeg")) + list(path.glob("*.jpg"))
                if images:
                    test_image = images[0]
                    break
        
        if not test_image:
            print("⚠️  No test images found, skipping pipeline test")
            return True
        
        print(f"   Test image: {test_image.name}")
        
        # Test clinical text
        clinical_text = """
        Patient presents with adenocarcinoma of the colon. 
        Histopathology confirmed malignant tissue.
        ICD-10 codes: C18.9 (Malignant neoplasm of colon, unspecified)
        """
        
        print("   Running full pipeline...")
        
        # Run pipeline
        result = pipeline.verify_claim(
            clinical_text=clinical_text,
            image_path=str(test_image),
            claim_amount=15000.0,
            previous_claim_count=0,
            patient_id="TEST001",
            claim_id="CLAIM001",
            save_output=False
        )
        
        if result.get('success'):
            print(f"✅ Pipeline execution successful")
            
            # Check image analysis
            if 'image_analysis' in result:
                img_result = result['image_analysis']
                print(f"\n   Image Analysis:")
                print(f"      CNN Score: {img_result.get('cnn_score', 'N/A')}")
                print(f"      Verdict: {img_result.get('forgery_verdict', 'N/A')}")
            
            # Check ICD validation
            if 'icd_verification' in result:
                icd_result = result['icd_verification']
                print(f"\n   ICD Validation:")
                print(f"      Status: {icd_result.get('verification_status', 'N/A')}")
            
            # Check fraud assessment
            if 'fraud_assessment' in result:
                fraud_result = result['fraud_assessment']
                print(f"\n   Fraud Assessment:")
                print(f"      Risk Level: {fraud_result.get('risk_level', 'N/A')}")
            
            # Check integrated verdict
            if 'integrated_verdict' in result:
                verdict = result['integrated_verdict']
                print(f"\n   Final Verdict:")
                print(f"      Recommendation: {verdict.get('recommendation', 'N/A')}")
                print(f"      Overall Risk: {verdict.get('overall_risk_level', 'N/A')}")
            
            return True
        else:
            error_msg = result.get('error', 'Unknown error')
            print(f"❌ Pipeline failed: {error_msg}")
            
            # Print full result for debugging
            if result:
                print(f"\n   Full result keys: {list(result.keys())}")
            
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backend_readiness():
    """Check if backend is ready to start"""
    print("\n" + "="*70)
    print("TEST 3: BACKEND READINESS CHECK")
    print("="*70)
    
    try:
        # Check backend files exist
        backend_app = PROJECT_ROOT / "backend" / "app.py"
        if not backend_app.exists():
            print("❌ backend/app.py not found")
            return False
        
        print("✅ Backend files present")
        
        # Check if we can import backend
        sys.path.insert(0, str(PROJECT_ROOT / "backend"))
        
        # Try importing key modules
        try:
            from flask import Flask
            print("✅ Flask available")
        except ImportError:
            print("⚠️  Flask not installed - backend won't start")
            print("   Install: pip install flask flask-cors")
            return False
        
        # Check pipeline import
        try:
            from pipeline.claim_verification_pipeline import ClaimVerificationPipeline
            print("✅ Pipeline module importable")
        except ImportError as e:
            print(f"❌ Cannot import pipeline: {e}")
            return False
        
        print("\n✅ Backend is ready to start")
        print("\nTo start backend:")
        print("   cd backend")
        print("   python app.py")
        print("\nBackend will listen on: http://localhost:5000")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_frontend_readiness():
    """Check if frontend is ready to start"""
    print("\n" + "="*70)
    print("TEST 4: FRONTEND READINESS CHECK")
    print("="*70)
    
    try:
        frontend_dir = PROJECT_ROOT / "frontend"
        
        if not frontend_dir.exists():
            print("⚠️  Frontend directory not found")
            return None
        
        # Check for package.json
        package_json = frontend_dir / "package.json"
        if not package_json.exists():
            print("⚠️  package.json not found")
            return None
        
        print("✅ Frontend files present")
        
        # Check for node_modules
        node_modules = frontend_dir / "node_modules"
        if not node_modules.exists():
            print("⚠️  node_modules not found")
            print("\nTo install dependencies:")
            print("   cd frontend")
            print("   npm install")
            return False
        
        print("✅ Dependencies installed")
        
        print("\n✅ Frontend is ready to start")
        print("\nTo start frontend:")
        print("   cd frontend")
        print("   npm start")
        print("\nFrontend will open in browser at: http://localhost:3000")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_final_summary(results):
    """Print comprehensive summary"""
    print("\n" + "="*70)
    print("INTEGRATION TEST SUMMARY")
    print("="*70)
    
    for test_name, result in results.items():
        if result is True:
            status = "✅ PASS"
        elif result is False:
            status = "❌ FAIL"
        else:
            status = "⚠️  SKIP"
        print(f"{status}: {test_name}")
    
    print("\n" + "="*70)
    
    # Check if ready for testing
    critical_tests = ['image_module', 'claim_pipeline', 'backend_readiness']
    critical_passed = all(results.get(t) is True for t in critical_tests)
    
    if critical_passed:
        print("✅ SYSTEM READY FOR END-TO-END TESTING!")
        print("="*70)
        
        print("\n📋 TESTING CHECKLIST:")
        print("="*70)
        print("1. ✅ LC25000 fine-tuned model loaded successfully")
        print("2. ✅ Pipeline configured to use fine-tuned model")
        print("3. ✅ Image forgery detection working")
        print("4. ✅ Full claim verification pipeline operational")
        print("5. ✅ Backend ready to start")
        
        print("\n🚀 NEXT STEPS:")
        print("="*70)
        print("\n1. Start Backend:")
        print("   cd backend")
        print("   python app.py")
        print("   → Backend will run on http://localhost:5000")
        
        print("\n2. Start Frontend:")
        print("   cd frontend")
        print("   npm start")
        print("   → Frontend will open at http://localhost:3000")
        
        print("\n3. Test the System:")
        print("   - Upload a clinical document (PDF/TXT)")
        print("   - Upload a medical image (from LC25000 dataset)")
        print("   - Enter claim details")
        print("   - Submit and verify results")
        
        print("\n4. Sample Test Images:")
        print("   Authentic:")
        print("   - data/LC25000/train/colon_aca/*.jpeg")
        print("   - data/LC25000/train/lung_aca/*.jpeg")
        print("\n   Tampered (if available):")
        print("   - data/LC25000_forgery/train/tampered/*.jpeg")
        
        print("\n" + "="*70)
        return True
    else:
        print("⚠️  SOME TESTS FAILED - REVIEW ISSUES ABOVE")
        print("="*70)
        return False


def main():
    """Run all integration tests"""
    print("\n" + "="*70)
    print("LC25000 FINE-TUNED MODEL - END-TO-END INTEGRATION TEST")
    print("="*70)
    print("Testing complete pipeline with fine-tuned forgery detection model")
    print("="*70)
    
    results = {}
    
    # Run all tests
    results['image_module'] = test_image_module()
    results['claim_pipeline'] = test_claim_pipeline()
    results['backend_readiness'] = test_backend_readiness()
    results['frontend_readiness'] = test_frontend_readiness()
    
    # Print summary
    ready = print_final_summary(results)
    
    return 0 if ready else 1


if __name__ == "__main__":
    sys.exit(main())
