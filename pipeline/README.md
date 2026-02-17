# 🏥 Claim Verification Pipeline - Complete Integration Guide

## Overview

The Claim Verification Pipeline is a modular, end-to-end system that automatically verifies insurance claims by analyzing:

1. **ICD Code Validation** (NLP Module) - Clinical text analysis using ClinicalBERT
2. **Medical Image Forgery Detection** (Vision Module) - CNN + ELA + pHash fusion
3. **Fraud Risk Scoring** (ML Module) - XGBoost-based risk assessment

**Status**: ✅ Production-ready for academic evaluation and backend deployment

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CLAIM SUBMISSION                             │
│         (Clinical Text + Image + Metadata)                      │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
         ┌──────▼──────┐ ┌────▼────┐  ┌─────▼──────┐
         │ ICD Module  │ │ Image   │  │  Metadata  │
         │(ClinicalBERT)
         │(ClinicalBERT)
         │             │ │ Module  │  │ Features   │
         └──────┬──────┘ └────┬────┘  └─────┬──────┘
                │ icd_match   │ forgery     │ claim_amount
                │             │ scores      │ prev_claims
                └─────────────┼────────────┘
                              │ Feature Vector
                      ┌───────▼────────┐
                      │ Fraud Risk     │
                      │ Classifier     │
                      │ (XGBoost)      │
                      └───────┬────────┘
                              │
                    ┌─────────▼──────────┐
                    │ Integrated Verdict │
                    │ & Recommendation   │
                    └────────────────────┘
```

---

## Pipeline Structure

```
pipeline/
├── __init__.py                          # Package exports
├── claim_verification_pipeline.py       # Main orchestrator
├── icd_module.py                        # ICD validation wrapper
├── image_module.py                      # Image forgery detection wrapper
├── fraud_module.py                      # Fraud risk classification wrapper
├── pipeline_demo.py                     # Demo and testing script
├── README.md                            # This file
└── outputs/                             # Generated reports (created at runtime)
```

---

## Module Interfaces

### 1. ICD Code Validation Module

**Function**: `run_icd_verification(clinical_text: str, top_k: int = 5) -> Dict`

**Input**:
- `clinical_text`: Discharge summary or clinical notes (string)
- `top_k`: Number of top ICD codes to return (default: 5)

**Output**:
```json
{
  "success": true,
  "predicted_icds": [["J18.9", 0.92], ["J20.9", 0.88]],
  "match_score": 0.92,
  "status": "valid",
  "num_icds_detected": 2,
  "explanation": "High confidence ICD code match (score: 0.92)..."
}
```

**Status Values**:
- `valid`: High confidence (score ≥ 0.8)
- `uncertain`: Moderate confidence (0.6 ≤ score < 0.8)
- `flagged`: Low confidence (score < 0.6)

---

### 2. Medical Image Forgery Detection Module

**Function**: `run_image_forgery(image_path: str, ...) -> Dict`

**Input**:
- `image_path`: Path to medical image file
- `model_ckpt`: Path to CNN checkpoint (optional)
- `phash_db`: Path to pHash database (optional)
- `output_dir`: Directory for ELA heatmap output (optional)
- `ela_quality`: JPEG quality for ELA (default: 90)
- `ela_scale`: Scale factor for ELA visualization (default: 10)

**Output**:
```json
{
  "success": true,
  "cnn_score": 0.1234,
  "ela_score": 0.0567,
  "phash_score": 0.0,
  "fused_score": 0.0892,
  "forgery_verdict": "authentic",
  "confidence": "high",
  "ela_heatmap_path": "pipeline/outputs/image_name_ela_heatmap.png",
  "explanation": "LOW RISK: Image appears authentic..."
}
```

**Forgery Verdicts**:
- `authentic`: Score < 0.5 (low risk)
- `suspicious`: 0.5 ≤ score < 0.7 (medium risk)
- `tampered`: Score ≥ 0.7 (high risk)

**Score Breakdown**:
- CNN: 0.5 weight (deep learning model)
- ELA: 0.3 weight (forensic analysis)
- pHash: 0.2 weight (duplicate detection)

---

### 3. Fraud Risk Classification Module

**Function**: `run_fraud_risk(features: Dict, model_path: str = None) -> Dict`

**Input** (Feature Dictionary):
```python
{
    "icd_match_score": 0.92,              # From ICD module
    "cnn_forgery_score": 0.12,            # From Image module
    "ela_score": 0.06,                    # From Image module
    "phash_score": 0.0,                   # From Image module
    "final_image_forgery_score": 0.09,    # From Image module
    "claim_amount_log": 7.52,             # log(1850.50)
    "previous_claim_count": 2             # Metadata
}
```

**Output**:
```json
{
  "success": true,
  "fraud_risk_percentage": 12.5,
  "risk_level": "low",
  "recommendation": "approve",
  "risk_factors": ["No major risk factors identified"],
  "feature_scores": {...},
  "feature_importances": {...},
  "explanation": "LOW FRAUD RISK: Fraud risk score is 12.5%..."
}
```

**Risk Levels**:
- `low`: Score < 25% → Recommend: `approve`
- `medium`: 25% ≤ score < 50% → Recommend: `review`
- `high`: 50% ≤ score < 75% → Recommend: `review`
- `critical`: Score ≥ 75% → Recommend: `reject`

---

## Main Orchestrator: ClaimVerificationPipeline

**Function**: `verify_insurance_claim(clinical_text, image_path, claim_amount, ...) -> Dict`

**Integrated Output** (Complete Report):
```json
{
  "metadata": {
    "timestamp": "2025-01-17T14:30:45.123456",
    "claim_id": "CLM_2025_001",
    "patient_id": "PAT_001",
    "claim_amount": 1850.50,
    "previous_claim_count": 2
  },
  "icd_verification": {...},
  "image_analysis": {...},
  "fraud_assessment": {...},
  "integrated_verdict": {
    "overall_recommendation": "approve",
    "confidence": 0.95,
    "risk_summary": "All validation checks passed. Claim appears legitimate."
  },
  "explanation": "Comprehensive multi-line explanation..."
}
```

---

## Usage Examples

### Basic Usage

```python
from pipeline import verify_insurance_claim

# Verify a claim
result = verify_insurance_claim(
    clinical_text="Patient diagnosed with pneumonia (J18.9)...",
    image_path="path/to/chest_xray.jpg",
    claim_amount=1850.50,
    previous_claim_count=2,
    claim_id="CLM_2025_001"
)

# Print recommendation
print(f"Verdict: {result['integrated_verdict']['overall_recommendation']}")
print(f"Risk: {result['fraud_assessment']['risk_level']}")
```

### Advanced: Module-by-Module

```python
from pipeline.icd_module import run_icd_verification
from pipeline.image_module import run_image_forgery
from pipeline.fraud_module import run_fraud_risk

# Step 1: Validate ICD codes
icd_result = run_icd_verification("Patient has pneumonia...")

# Step 2: Check image authenticity
image_result = run_image_forgery("path/to/image.jpg")

# Step 3: Assess fraud risk
features = {
    "icd_match_score": icd_result["match_score"],
    "cnn_forgery_score": image_result["cnn_score"],
    "ela_score": image_result["ela_score"],
    "phash_score": image_result.get("phash_score", 0),
    "final_image_forgery_score": image_result["fused_score"],
    "claim_amount_log": np.log1p(1850.50),
    "previous_claim_count": 2
}

fraud_result = run_fraud_risk(features)
print(f"Fraud Risk: {fraud_result['fraud_risk_percentage']:.1f}%")
```

### Running the Demo

```bash
# Mock test with sample data
python pipeline/pipeline_demo.py --mode mock

# Interactive test
python pipeline/pipeline_demo.py --mode interactive
```

---

## Integration with FastAPI (Future)

The pipeline is designed for easy FastAPI integration:

```python
from fastapi import FastAPI, File, UploadFile, Form
from pipeline import verify_insurance_claim

app = FastAPI()

@app.post("/verify-claim")
async def verify_claim(
    clinical_text: str = Form(...),
    image: UploadFile = File(...),
    claim_amount: float = Form(...),
    previous_claim_count: int = Form(0),
    claim_id: str = Form(...)
):
    # Save uploaded image
    image_path = f"uploads/{image.filename}"
    with open(image_path, "wb") as f:
        f.write(await image.read())
    
    # Run pipeline
    result = verify_insurance_claim(
        clinical_text=clinical_text,
        image_path=image_path,
        claim_amount=claim_amount,
        previous_claim_count=previous_claim_count,
        claim_id=claim_id
    )
    
    return result
```

---

## Integration with Report Generation (PDF)

```python
from pipeline import verify_insurance_claim
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

result = verify_insurance_claim(...)

# Generate PDF report
c = canvas.Canvas(f"reports/claim_{result['metadata']['claim_id']}.pdf", pagesize=letter)
c.drawString(100, 750, f"Claim ID: {result['metadata']['claim_id']}")
c.drawString(100, 730, f"Verdict: {result['integrated_verdict']['overall_recommendation']}")
c.drawString(100, 710, result['explanation'])
c.save()
```

---

## Key Features

✅ **Modular Design**: Each module operates independently and can be used standalone

✅ **JSON Output**: All results are JSON-serializable for API/frontend integration

✅ **Error Handling**: Graceful error responses with informative messages

✅ **Explainability**: Human-readable explanations for all decisions

✅ **Scalability**: Designed for batch processing and parallel execution

✅ **Extensibility**: Easy to add new modules or modify existing ones

✅ **Backward Compatibility**: All underlying models remain unchanged

---

## Output Files

Generated reports are saved to `pipeline/outputs/`:

- `claim_{claim_id}_report.json` - Complete verification report

Optional visualizations (if output_dir provided):

- `{image_name}_ela_heatmap.png` - ELA forensic heatmap

---

## Performance Characteristics

| Module | Processing Time | Input Size | Output |
|--------|-----------------|------------|--------|
| ICD Validation | ~1-2 seconds | ~500-5000 words | ICD codes + scores |
| Image Forgery | ~2-5 seconds | Medical image | Forgery verdict + heatmap |
| Fraud Risk | ~100ms | 7 features | Risk percentage |
| **Total Pipeline** | **~3-7 seconds** | Text + Image | Complete report |

---

## Future Enhancements

1. **Real Fraud Labels**: Retrain fraud classifier with actual fraud/non-fraud claims
2. **Multi-Image Support**: Process multiple medical images per claim
3. **Batch Processing**: Submit multiple claims simultaneously
4. **Monitoring Dashboard**: Track system performance and false positive rates
5. **Explainability**: SHAP values, Grad-CAM for detailed feature attribution
6. **A/B Testing**: Compare different model configurations
7. **Active Learning**: Auto-flag uncertain cases for human annotation

---

## Support & Debugging

**Common Issues**:

1. **Import Error**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Image File Not Found**: Verify image path exists
   ```python
   from pathlib import Path
   assert Path(image_path).exists()
   ```

3. **Model Not Found**: Check checkpoint paths
   ```python
   from pathlib import Path
   assert Path("checkpoints/casia/best.pth.tar").exists()
   ```

**Enable Debugging**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## Citation

If you use this pipeline in research, please cite:

```bibtex
@software{claim_verification_2025,
  title={AI-Driven Claim Verification Pipeline},
  author={Team},
  year={2025},
  url={https://github.com/tech_squad/AI-Driven-Image-Forgery-Detection}
}
```

---

**Last Updated**: January 17, 2025  
**Version**: 1.0.0  
**Status**: ✅ Production Ready
