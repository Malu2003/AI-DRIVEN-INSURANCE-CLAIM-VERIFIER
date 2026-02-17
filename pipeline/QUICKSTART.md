# 🚀 Pipeline Quick Start Guide

**Status**: ✅ Ready to use  
**Last Updated**: January 17, 2026

---

## 📋 Files in Pipeline Directory

```
pipeline/
├── __init__.py                      # Package exports
├── claim_verification_pipeline.py   # Main orchestrator (350+ lines)
├── icd_module.py                    # ICD validation (200+ lines)
├── image_module.py                  # Image forgery detection (250+ lines)
├── fraud_module.py                  # Fraud risk classification (270+ lines)
├── pipeline_demo.py                 # Demo script (300+ lines)
├── tests.py                         # Test suite (320+ lines)
├── README.md                        # Full API documentation
├── INTEGRATION_SUMMARY.md           # Technical specifications
├── ARCHITECTURE.md                  # System design diagrams
└── outputs/                         # Generated reports (auto-created)
```

---

## ⚡ 5-Minute Quick Start

### 1. Run Demo (Requires CASIA Data)
```bash
cd pipeline
python pipeline_demo.py --mode mock
```

**Output**: Complete claim verification report printed to console

### 2. Run Tests
```bash
python tests.py
```

**Output**: Test results showing all modules working correctly

### 3. Use in Your Code
```python
from pipeline import verify_insurance_claim

result = verify_insurance_claim(
    clinical_text="Patient with pneumonia (J18.9)...",
    image_path="path/to/image.jpg",
    claim_amount=1850.50,
    previous_claim_count=2,
    claim_id="CLM_2025_001"
)

print(f"Verdict: {result['integrated_verdict']['overall_recommendation']}")
```

---

## 📚 Documentation Quick Links

| Document | Purpose | Read Time |
|----------|---------|-----------|
| [README.md](README.md) | Complete API reference | 15 min |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design & diagrams | 10 min |
| [INTEGRATION_SUMMARY.md](INTEGRATION_SUMMARY.md) | Technical specs | 20 min |

---

## 🎯 What Each Module Does

### ICD Validation Module
```python
from pipeline.icd_module import run_icd_verification

result = run_icd_verification("Patient with pneumonia (J18.9)...")
# Returns: {predicted_icds, match_score, status, explanation, ...}
```

### Image Forgery Detection Module
```python
from pipeline.image_module import run_image_forgery

result = run_image_forgery("path/to/chest_xray.jpg")
# Returns: {cnn_score, ela_score, phash_score, fused_score, verdict, ...}
```

### Fraud Risk Classification Module
```python
from pipeline.fraud_module import run_fraud_risk

result = run_fraud_risk({
    "icd_match_score": 0.92,
    "cnn_forgery_score": 0.12,
    "ela_score": 0.06,
    "phash_score": 0.0,
    "final_image_forgery_score": 0.09,
    "claim_amount_log": 7.52,
    "previous_claim_count": 2
})
# Returns: {fraud_risk_percentage, risk_level, recommendation, ...}
```

### Complete Pipeline (Recommended)
```python
from pipeline import verify_insurance_claim

result = verify_insurance_claim(
    clinical_text="...",
    image_path="...",
    claim_amount=1850.50,
    previous_claim_count=2,
    claim_id="CLM_001"
)
# Returns: {metadata, icd_verification, image_analysis, fraud_assessment, integrated_verdict, ...}
```

---

## 📊 Expected Output Format

All modules return JSON-serializable dictionaries:

```python
{
    "success": True,                    # bool
    "explanation": "...",               # str
    "status": "valid|error",            # str
    # ... module-specific fields
}
```

Complete pipeline returns:
```python
{
    "metadata": {...},
    "icd_verification": {...},
    "image_analysis": {...},
    "fraud_assessment": {...},
    "integrated_verdict": {...},
    "explanation": "..."
}
```

---

## 🔧 Troubleshooting

### ImportError: No module named 'pipeline'
```bash
# Make sure you're in the project root
cd d:\tech_squad\AI-Driven-Image-Forgery-Detection
python -c "from pipeline import verify_insurance_claim; print('OK')"
```

### FileNotFoundError: Image not found
```python
from pathlib import Path
assert Path("path/to/image.jpg").exists()  # Check file exists
```

### ModuleNotFoundError: No module named 'icd_validation'
```bash
# Install dependencies
pip install -r requirements.txt
```

### CUDA/GPU Errors
```python
# Pipeline defaults to CPU, which is slower but works on any system
# No configuration needed - it will automatically fall back to CPU
```

---

## 📈 Testing & Validation

### Run All Tests
```bash
python pipeline/tests.py
```

Expected output:
```
✅ Module imports
✅ ICD module: valid input
✅ ICD module: output types
✅ Image module: valid input
✅ Image module: output validation
✅ Fraud module: valid input
✅ Fraud module: output validation
✅ Integrated pipeline: all modules executed
✅ Integrated pipeline: verdict structure
✅ Integrated pipeline: JSON serializable

TEST SUMMARY: 10/10 passed
```

### Run Demo
```bash
# Mock test with sample data
python pipeline/pipeline_demo.py --mode mock

# Interactive mode (enter your own data)
python pipeline/pipeline_demo.py --mode interactive
```

---

## 🔌 Integration Examples

### With FastAPI
```python
from fastapi import FastAPI, File, UploadFile, Form
from pipeline import verify_insurance_claim

app = FastAPI()

@app.post("/api/verify-claim")
async def api_verify(clinical_text: str = Form(...), image: UploadFile = File(...), ...):
    return verify_insurance_claim(clinical_text, ...)
```

### With Flask
```python
from flask import Flask, request, jsonify
from pipeline import verify_insurance_claim

app = Flask(__name__)

@app.route('/verify', methods=['POST'])
def verify():
    result = verify_insurance_claim(
        clinical_text=request.form['text'],
        image_path=request.files['image'].filename,
        ...
    )
    return jsonify(result)
```

### With Django
```python
from django.http import JsonResponse
from pipeline import verify_insurance_claim

def verify_claim(request):
    result = verify_insurance_claim(...)
    return JsonResponse(result)
```

### Batch Processing
```python
import json
from pipeline import verify_insurance_claim

claims = json.load(open('claims.json'))
results = []

for claim in claims:
    result = verify_insurance_claim(
        clinical_text=claim['text'],
        image_path=claim['image'],
        claim_amount=claim['amount'],
        claim_id=claim['id']
    )
    results.append(result)

json.dump(results, open('results.json', 'w'), indent=2)
```

---

## 💾 Output Files

The pipeline automatically saves results to:
```
pipeline/outputs/claim_{claim_id}_report.json
```

Example:
```
pipeline/outputs/claim_CLM_2025_001_report.json
pipeline/outputs/image_xray_ela_heatmap.png  # If ELA visualization enabled
```

---

## 🎯 Typical Workflow

```
1. User submits claim (text + image + metadata)
                ↓
2. Pipeline validates ICD codes
                ↓
3. Pipeline detects image forgery
                ↓
4. Pipeline assesses fraud risk
                ↓
5. Pipeline generates integrated verdict
                ↓
6. Result returned: approve / review / reject
                ↓
7. Report saved to pipeline/outputs/
```

---

## ⏱️ Performance Expectations

| Operation | Time | Notes |
|-----------|------|-------|
| ICD validation | ~1-2 sec | Text processing |
| Image forgery | ~2-5 sec | CNN inference |
| Fraud risk | ~100 ms | Fast classification |
| **Total** | **~3-7 sec** | Per claim |

---

## 📞 Common Questions

**Q: Can I use this without medical images?**  
A: No, medical image forgery detection is core. But image validation failure won't block the entire pipeline.

**Q: Can I use this without clinical text?**  
A: You can call `run_image_forgery()` independently, but the full integrated pipeline needs both.

**Q: Can I train custom models?**  
A: This package uses pre-trained models. To retrain, modify the respective module files.

**Q: Is GPU required?**  
A: No, pipeline automatically falls back to CPU if GPU unavailable (slower but works).

**Q: Can I deploy this to the cloud?**  
A: Yes, fully compatible with Docker, AWS Lambda, GCP, Azure.

---

## 🚀 Next Steps

1. **Try the demo**: `python pipeline/pipeline_demo.py --mode mock`
2. **Read the docs**: Open `pipeline/README.md`
3. **Review architecture**: Open `pipeline/ARCHITECTURE.md`
4. **Integrate into your app**: Import and use in your FastAPI/Flask/Django server
5. **Deploy**: Docker → Cloud (AWS/GCP/Azure)

---

## 📖 Documentation Map

```
START HERE
    ↓
pipeline/README.md ................... API Documentation
    ↓
pipeline/ARCHITECTURE.md ............. System Design
    ↓
pipeline/INTEGRATION_SUMMARY.md ....... Technical Specs
    ↓
pipeline_demo.py ..................... Working Examples
    ↓
tests.py ............................ Validation
```

---

## ✅ Checklist Before Production

- [ ] Read README.md
- [ ] Run tests.py (all pass)
- [ ] Run pipeline_demo.py --mode mock (works)
- [ ] Test with your own data
- [ ] Review ARCHITECTURE.md
- [ ] Configure error logging
- [ ] Set up output directory
- [ ] Test integration with your app
- [ ] Deploy to staging
- [ ] Validate with real claims
- [ ] Deploy to production

---

## 🎓 For Researchers

**To cite this pipeline**:
```bibtex
@software{claim_verification_pipeline_2025,
  title={End-to-End Claim Verification Pipeline},
  author={Team},
  year={2025},
  url={https://github.com/tech_squad/...}
}
```

**For publication**:
- Include `pipeline/` directory in supplementary materials
- Reference ARCHITECTURE.md for system description
- Reference README.md for API documentation
- Include test results from tests.py

---

**Version**: 1.0.0  
**Status**: ✅ Production Ready  
**Last Updated**: January 17, 2026

Ready to verify claims! 🚀
