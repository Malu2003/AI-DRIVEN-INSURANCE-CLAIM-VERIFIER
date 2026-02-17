#!/usr/bin/env python3
"""Test backend endpoint with a simple image"""
import requests
import os
from pathlib import Path

# Test with authentic LC25000 image
test_image = "data/LC25000/train/colon_aca/colonca1.jpeg"

# Create minimal test PDF (just text content)
test_pdf = "backend/temp_uploads/test_doc.pdf"
os.makedirs("backend/temp_uploads", exist_ok=True)

# Check if test image exists
if not Path(test_image).exists():
    print(f"ERROR: Test image not found: {test_image}")
    exit(1)

# Prepare request
url = "http://localhost:5000/api/verify-claim"
data = {
    'claim_amount': 2500.0,
    'claim_id': 'TEST-BACKEND-001',
    'patient_id': 'PAT-001',
    'previous_claim_count': 0
}

# Open both files
with open(test_image, 'rb') as img_file:
    # For document, we can send the image as PDF if PDF doesn't exist
    # Or use a sample PDF if available
    doc_file_path = "docs/sample_clinical_document.pdf"
    if not Path(doc_file_path).exists():
        # Try to find any PDF
        doc_files = list(Path("docs").glob("*.pdf")) if Path("docs").exists() else []
        if doc_files:
            doc_file_path = str(doc_files[0])
        else:
            print("ERROR: No PDF document found for testing")
            exit(1)
    
    with open(doc_file_path, 'rb') as doc_file:
        files = {
            'clinical_document': doc_file,
            'image': img_file
        }
        
        print(f"Testing /api/verify-claim endpoint...")
        print(f"  Image: {test_image}")
        print(f"  Document: {doc_file_path}")
        print(f"  URL: {url}")
        print()
        
        try:
            response = requests.post(url, files=files, data=data, timeout=60)
            print(f"Response Status: {response.status_code}")
            
            if response.status_code == 200:
                import json
                result = response.json()
                print("\n✓ SUCCESS: Backend processed claim")
                print(f"\nImage Analysis:")
                if 'image_analysis' in result:
                    img = result['image_analysis']
                    print(f"  Verdict: {img.get('forgery_verdict', 'N/A')}")
                    print(f"  CNN Score: {img.get('cnn_score', 'N/A')}")
                    print(f"  ELA Score: {img.get('ela_score', 'N/A')}")
                    print(f"  pHash Score: {img.get('phash_score', 'N/A')}")
                    print(f"  Fused Score: {img.get('fused_score', 'N/A')}")
                    print(f"  Confidence: {img.get('confidence', 'N/A')}")
                    
                print(f"\nOverall Recommendation: {result.get('recommendation', 'N/A')}")
            else:
                print(f"\n✗ ERROR: Backend returned {response.status_code}")
                print(f"Response: {response.text}")
                
        except Exception as e:
            print(f"✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
