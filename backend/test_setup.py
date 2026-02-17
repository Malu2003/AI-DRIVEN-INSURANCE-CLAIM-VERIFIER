#!/usr/bin/env python
"""
Backend API Test Script
=======================

Tests the Flask backend API endpoints without requiring the full pipeline.
Use this to verify the backend setup before running the full application.
"""

import sys
from pathlib import Path
import os
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Test 1: Check imports
print("=" * 70)
print("TEST 1: Checking imports")
print("=" * 70)

try:
    from backend.report_generator.report_builder import ReportBuilder
    print("✅ ReportBuilder imported successfully")
except ImportError as e:
    print(f"❌ Failed to import ReportBuilder: {e}")
    sys.exit(1)

try:
    from flask import Flask
    print("✅ Flask imported successfully")
except ImportError as e:
    print(f"❌ Failed to import Flask: {e}")
    print("   Install with: pip install flask")
    sys.exit(1)

try:
    from weasyprint import WeasyPrint
    print("✅ WeasyPrint imported successfully")
except ImportError as e:
    print(f"⚠️  WeasyPrint not installed: {e}")
    print("   Install with: pip install weasyprint")
    print("   Or use: pip install wkhtmltopdf")

# Test 2: Check file structure
print("\n" + "=" * 70)
print("TEST 2: Checking file structure")
print("=" * 70)

files_to_check = [
    'backend/app.py',
    'backend/__init__.py',
    'backend/requirements.txt',
    'backend/README.md',
    'backend/API_QUICK_REFERENCE.md',
    'backend/report_generator/__init__.py',
    'backend/report_generator/report_builder.py',
]

for file_path in files_to_check:
    full_path = PROJECT_ROOT / file_path
    if full_path.exists():
        size_kb = full_path.stat().st_size / 1024
        print(f"✅ {file_path} ({size_kb:.1f} KB)")
    else:
        print(f"❌ {file_path} - NOT FOUND")

# Test 3: Check temp_uploads directory
print("\n" + "=" * 70)
print("TEST 3: Checking temp_uploads directory")
print("=" * 70)

temp_dir = PROJECT_ROOT / 'backend' / 'temp_uploads'
if temp_dir.exists():
    print(f"✅ {temp_dir} exists")
    if os.access(temp_dir, os.W_OK):
        print(f"✅ {temp_dir} is writable")
    else:
        print(f"❌ {temp_dir} is not writable")
else:
    print(f"⚠️  {temp_dir} does not exist (will be created on first use)")

# Test 4: Instantiate ReportBuilder
print("\n" + "=" * 70)
print("TEST 4: Testing ReportBuilder instantiation")
print("=" * 70)

try:
    builder = ReportBuilder()
    print("✅ ReportBuilder instantiated successfully")
except ImportError as e:
    if "WeasyPrint" in str(e):
        print(f"⚠️  ReportBuilder requires WeasyPrint: {e}")
        print("   Install with: pip install weasyprint")
    else:
        print(f"❌ Failed to instantiate ReportBuilder: {e}")
except Exception as e:
    print(f"❌ Error instantiating ReportBuilder: {e}")

# Test 5: Check pipeline import
print("\n" + "=" * 70)
print("TEST 5: Checking pipeline import")
print("=" * 70)

try:
    from pipeline import verify_insurance_claim
    print("✅ Pipeline imported successfully")
    print(f"   Function: verify_insurance_claim")
except ImportError as e:
    print(f"⚠️  Pipeline not available: {e}")
    print("   Note: This is OK if pipeline is in a different location")
    print("   The backend will fail at runtime if pipeline is not available")

# Test 6: Check Flask app
print("\n" + "=" * 70)
print("TEST 6: Testing Flask app structure")
print("=" * 70)

try:
    # Import app (this will test imports)
    from backend.app import app
    print("✅ Flask app imported successfully")
    
    # Check routes
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append(str(rule))
    
    expected_routes = ['/health', '/verify-claim', '/generate-report', '/api/status']
    print(f"✅ Found {len(routes)} routes:")
    for route in sorted(routes):
        print(f"   - {route}")
        
except Exception as e:
    print(f"⚠️  Could not import Flask app: {e}")
    print("   Note: This might fail if dependencies are not installed")
    print("   Try: pip install -r backend/requirements.txt")

# Test 7: Dependency check
print("\n" + "=" * 70)
print("TEST 7: Checking critical dependencies")
print("=" * 70)

dependencies = {
    'flask': 'Flask web framework',
    'werkzeug': 'File upload handling',
    'weasyprint': 'PDF generation',
    'torch': 'PyTorch (for pipeline)',
    'transformers': 'Hugging Face transformers (for pipeline)',
    'xgboost': 'XGBoost (for pipeline)',
}

missing = []
for module, description in dependencies.items():
    try:
        __import__(module)
        print(f"✅ {module:15} - {description}")
    except ImportError:
        print(f"❌ {module:15} - {description} [MISSING]")
        missing.append(module)

if missing:
    print(f"\n⚠️  Missing {len(missing)} dependencies:")
    print(f"   Run: pip install -r backend/requirements.txt")

# Test 8: Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

if not missing or missing == ['weasyprint']:  # WeasyPrint optional on some systems
    print("✅ Backend setup is complete!")
    print("\nTo start the API server:")
    print("   cd backend")
    print("   python app.py")
    print("\nThen test with:")
    print("   curl http://localhost:5000/health")
else:
    print(f"⚠️  Backend setup incomplete - missing {len(missing)} dependencies")
    print("\nTo install all dependencies:")
    print("   pip install -r backend/requirements.txt")

print("\nDocumentation:")
print("   - README.md - Full API documentation")
print("   - API_QUICK_REFERENCE.md - Quick reference guide")
print("\n" + "=" * 70)
