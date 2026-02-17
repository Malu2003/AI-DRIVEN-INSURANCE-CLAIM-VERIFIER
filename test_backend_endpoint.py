#!/usr/bin/env python3
"""Test backend endpoint directly by making HTTP request"""
import requests
import json
from pathlib import Path

# Test data
test_image_path = "data/LC25000/train/colon_aca/colonca1.jpeg"
test_pdf_path = "docs/sample_clinical_document.pdf"  # Use any existing PDF

# Prepare the request
url = "http://localhost:5000/api/verify-claim"
files = {
    'image': open(test_image_path, 'rb'),
}
data = {
    'claim_amount': 5000.0,
    'claim_id': 'TEST-001',
    'patient_id': 'PAT-001'
}

# If PDF doesn't exist, skip document submission
if Path(test_pdf_path).exists():
    files['clinical_document'] = open(test_pdf_path, 'rb')

print("Testing backend verify-claim endpoint...")
print(f"URL: {url}")
print(f"Data: {data}")

try:
    response = requests.post(url, files=files, data=data)
    print(f"\nResponse Status: {response.status_code}")
    print(f"Response Headers: {dict(response.headers)}")
    print(f"Response Body:")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"ERROR: {e}")
finally:
    # Close files
    for file_obj in files.values():
        file_obj.close()
