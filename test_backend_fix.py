"""Test the fixed backend with authentic and tampered test images."""
import requests
import json
from pathlib import Path

# Backend endpoint
BACKEND_URL = "http://localhost:5000/api/verify-claim"

# Test images from our batch test results
authentic_image = "batch_test_results/blur/colonca1000_blur.jpg"  # Should be authentic
tampered_image = "batch_test_results/compression/colonca1000_compression.jpg"  # Should be tampered

# Dummy clinical document
clinical_doc = """
PATIENT: John Doe
DOB: 01/15/1965
ADMIT DATE: 2025-01-10
DISCHARGE DATE: 2025-01-17

DIAGNOSIS:
- Colon adenocarcinoma stage II
- Hypertension
- Type 2 Diabetes Mellitus

PROCEDURE: Colonoscopy with biopsy
FINDINGS: Adenomatous polyp identified in sigmoid colon, biopsied for pathology.

ASSESSMENT AND PLAN:
1. Pathology results pending
2. Follow-up CT abdomen/pelvis for staging
3. Oncology consultation recommended
4. Continue current medications
"""

def test_image(image_path, image_label):
    """Test a single image through the API."""
    print(f"\n{'='*70}")
    print(f"🔍 Testing: {image_label} ({image_path})")
    print(f"{'='*70}")
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return False
    
    try:
        # Prepare files
        files = {
            'clinical_document': ('clinical.txt', clinical_doc, 'text/plain'),
            'image': (Path(image_path).name, open(image_path, 'rb'), 'image/jpeg'),
        }
        data = {
            'claim_amount': '5000',
            'claim_id': f'CLM_TEST_{image_label}',
            'patient_id': 'PAT_TEST_001',
            'previous_claim_count': '0',
        }
        
        # Send request
        response = requests.post(BACKEND_URL, files=files, data=data, timeout=30)
        
        # Close file
        files['image'][1].close()
        
        if response.status_code != 200:
            print(f"❌ API Error: {response.status_code}")
            print(response.text)
            return False
        
        result = response.json()
        
        # Extract image analysis
        image_analysis = result.get('image_analysis', {})
        verdict = image_analysis.get('forgery_verdict')
        fused_score = image_analysis.get('fused_score')
        cnn_score = image_analysis.get('cnn_score')
        ela_score = image_analysis.get('ela_score')
        confidence = image_analysis.get('confidence')
        
        print(f"\n📊 Results:")
        print(f"  Verdict:      {verdict}")
        print(f"  Fused Score:  {fused_score}")
        print(f"  CNN Score:    {cnn_score}")
        print(f"  ELA Score:    {ela_score}")
        print(f"  Confidence:   {confidence}")
        
        # Save full response
        print(f"\n💾 Full Response:")
        print(json.dumps(result, indent=2)[:500] + "...")
        
        return verdict
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

print("\n" + "="*70)
print("🚀 TESTING FIXED BACKEND")
print("="*70)

# Test authentic image
verdict1 = test_image(authentic_image, "BLUR (should be AUTHENTIC)")

# Test tampered image
verdict2 = test_image(tampered_image, "COMPRESSION (should be TAMPERED)")

# Summary
print(f"\n{'='*70}")
print("✅ SUMMARY")
print(f"{'='*70}")
print(f"Blur/Authentic:       {verdict1} {'✅' if verdict1 in ['authentic', 'suspicious'] else '❌'}")
print(f"Compression/Tampered: {verdict2} {'✅' if verdict2 == 'tampered' else '❌'}")

if verdict1 in ['authentic', 'suspicious'] and verdict2 == 'tampered':
    print(f"\n🎉 SUCCESS! The backend is working correctly now!")
else:
    print(f"\n❌ STILL BROKEN - Verdicts don't match expected values")
