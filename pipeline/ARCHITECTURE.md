# Pipeline Architecture Visualization

## Complete System Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      INSURANCE CLAIM SUBMISSION                         │
│                                                                          │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐     │
│  │  Clinical Text   │  │  Medical Image   │  │  Claim Metadata  │     │
│  │  (discharge      │  │  (chest X-ray,   │  │  - Amount        │     │
│  │   summary)       │  │   CT scan, etc)  │  │  - Claim count   │     │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘     │
│           │                     │                     │                │
└───────────┼─────────────────────┼─────────────────────┼────────────────┘
            │                     │                     │
            │                     │                     │
    ┌───────▼───────────┐ ┌──────▼────────┐  ┌────────▼────────┐
    │  ICD VALIDATION   │ │   IMAGE       │  │  FEATURE        │
    │  MODULE           │ │   FORGERY     │  │  ENGINEERING    │
    │                   │ │   MODULE      │  │                 │
    │ • ClinicalBERT    │ │               │  │ • claim_amount_ │
    │ • ICD extraction  │ │ • DenseNet121 │  │   log           │
    │ • Confidence      │ │ • ELA         │  │ • previous_     │
    │   scoring         │ │ • pHash       │  │   claim_count   │
    │                   │ │               │  │                 │
    └───────┬───────────┘ └──────┬────────┘  └────────┬────────┘
            │                    │                    │
            │ icd_match_score   │ forgery_scores     │ metadata
            │ (0-1)             │ (CNN, ELA, pHash)  │ features
            │                   │                    │
            └───────────────────┼────────────────────┘
                                │
                    ┌───────────▼────────────┐
                    │  FEATURE VECTOR       │
                    │  CONSTRUCTION         │
                    │                       │
                    │  7-dim vector:        │
                    │  1. icd_match_score   │
                    │  2. cnn_forgery       │
                    │  3. ela_score         │
                    │  4. phash_score       │
                    │  5. final_image_score │
                    │  6. claim_amount_log  │
                    │  7. previous_claims   │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼──────────────┐
                    │   FRAUD RISK CLASSIFIER │
                    │   (XGBoost)             │
                    │                        │
                    │  • Feature importances │
                    │  • Risk probability    │
                    │  • Risk factors        │
                    │  • Recommendation      │
                    └───────────┬──────────────┘
                                │
                ┌───────────────┼──────────────────┐
                │               │                  │
            ┌───▼────┐   ┌──────▼───┐   ┌────────▼──────┐
            │ ICD    │   │ IMAGE    │   │  FRAUD RISK   │
            │ RESULT │   │ RESULT   │   │  RESULT       │
            └───┬────┘   └──────┬───┘   └────────┬──────┘
                │               │                 │
                │               │                 │
                └───────────────┼─────────────────┘
                                │
                    ┌───────────▼──────────────┐
                    │  INTEGRATED VERDICT     │
                    │  COMPUTATION            │
                    │                        │
                    │  • Consensus logic     │
                    │  • Overall             │
                    │    recommendation      │
                    │  • Confidence score    │
                    │  • Risk summary        │
                    └───────────┬──────────────┘
                                │
        ┌───────────────────────┼────────────────────────┐
        │                       │                        │
    ┌───▼────────────┐  ┌──────▼──────┐  ┌─────────────▼──────┐
    │  JSON REPORT   │  │  PDF EXPORT │  │  DASHBOARD/UI DATA │
    │                │  │  (optional) │  │                    │
    │ - Metadata     │  │             │  │ - Verdict          │
    │ - All results  │  │ Professional│  │ - Risk level       │
    │ - Verdict      │  │ report      │  │ - Explanations     │
    │ - Explanation  │  │             │  │                    │
    └────────────────┘  └─────────────┘  └────────────────────┘
```

---

## Module Communication Flow

```
┌────────────────────────────────────────────────────────────────┐
│         ClaimVerificationPipeline (Main Orchestrator)          │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  1. receive_input()                                           │
│     ├─ clinical_text                                         │
│     ├─ image_path                                            │
│     ├─ claim_amount                                          │
│     └─ previous_claim_count                                  │
│                                                                │
│  2. call_icd_module()                                        │
│     └─ ICD_Module.run(clinical_text)                         │
│        ├─ Input: discharge summary                           │
│        ├─ Processing: ClinicalBERT                           │
│        └─ Output: {icd_match_score, predicted_icds, ...}    │
│                                                                │
│  3. call_image_module()                                      │
│     └─ Image_Module.run(image_path)                          │
│        ├─ Input: medical image                               │
│        ├─ Processing: CNN + ELA + pHash                      │
│        └─ Output: {cnn_score, ela_score, phash_score, ...}  │
│                                                                │
│  4. build_feature_vector()                                   │
│     ├─ Extract: icd_match_score                              │
│     ├─ Extract: cnn_forgery_score, ela_score, phash_score   │
│     ├─ Extract: claim_amount_log, previous_claim_count      │
│     └─ Output: 7-dimensional feature vector                  │
│                                                                │
│  5. call_fraud_module()                                      │
│     └─ Fraud_Module.run(feature_vector)                      │
│        ├─ Input: 7-dim features                              │
│        ├─ Processing: XGBoost classification                 │
│        └─ Output: {fraud_risk_pct, risk_level, ...}         │
│                                                                │
│  6. compute_integrated_verdict()                             │
│     ├─ Consensus: ICD + Image + Fraud signals                │
│     ├─ Decision: approve/review/reject                       │
│     └─ Output: {recommendation, confidence, risk_summary}    │
│                                                                │
│  7. generate_report()                                        │
│     ├─ Metadata                                              │
│     ├─ Module results (ICD, Image, Fraud)                    │
│     ├─ Integrated verdict                                    │
│     ├─ Explanation                                           │
│     └─ Output: Unified JSON report                           │
│                                                                │
│  8. save_and_return()                                        │
│     ├─ Save JSON to file: pipeline/outputs/                  │
│     └─ Return: Complete report dict                          │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Through Modules

```
INPUT STAGE
═══════════════════════════════════════════════════════════════════
│
├─ Clinical Text (500-5000 chars)
├─ Medical Image (JPG/PNG, any size)
├─ Claim Amount (float, $)
└─ Previous Claims (int)


PROCESSING STAGE
═══════════════════════════════════════════════════════════════════
│
├─────► ICD Module
│       │ Input: "Patient with pneumonia J18.9..."
│       │ Process: ClinicalBERT tokenization + embedding
│       │ Output: {match_score: 0.92, status: valid}
│       │
│
├─────► Image Module
│       │ Input: "chest_xray.jpg"
│       │ Process: DenseNet121 → CNN score
│       │ Process: JPEG recompression → ELA score
│       │ Process: Hashing → pHash score
│       │ Fusion: weighted combination
│       │ Output: {fused_score: 0.089, verdict: authentic}
│       │
│
└─────► Feature Engineering
        │ Combine outputs:
        │ ├─ icd_match_score: 0.92
        │ ├─ cnn_forgery_score: 0.12
        │ ├─ ela_score: 0.06
        │ ├─ phash_score: 0.0
        │ ├─ final_image_forgery: 0.09
        │ ├─ claim_amount_log: 7.52
        │ └─ previous_claim_count: 2
        │
        └─────► Fraud Risk Module
                │ Input: 7-dim feature vector
                │ Process: XGBoost classification
                │ Output: {fraud_risk: 12.5%, risk_level: low}
                │


OUTPUT STAGE
═══════════════════════════════════════════════════════════════════
│
├─ ICD Verification Results
│  ├─ Status: valid/uncertain/flagged
│  ├─ Predicted ICDs: [(code, confidence), ...]
│  └─ Explanation: "High confidence match..."
│
├─ Image Analysis Results
│  ├─ Verdict: authentic/suspicious/tampered
│  ├─ Component Scores: CNN, ELA, pHash
│  └─ Explanation: "No signs of tampering..."
│
├─ Fraud Assessment Results
│  ├─ Risk Level: low/medium/high/critical
│  ├─ Risk Percentage: 12.5%
│  └─ Explanation: "All signals indicate legitimacy..."
│
└─ Integrated Verdict
   ├─ Recommendation: approve/review/reject
   ├─ Confidence: 0.95
   └─ Summary: "Claim appears legitimate based on all checks"
```

---

## Module Independence & Coupling

```
┌─────────────────┐
│  ICD Module     │  ← No dependencies on Image or Fraud
│                 │     Can be called standalone
│  run_icd_*()    │     Can be replaced/updated independently
└────────┬────────┘
         │ Output: icd_match_score
         │
         ▼
┌─────────────────────────────────────────┐
│  Orchestrator (ClaimVerificationPipeline)
└──────────────┬──────────────────────────┘
               │
         ┌─────┴─────┐
         │           │
    ┌────▼────┐  ┌──▼──────────┐
    │ Image   │  │ Fraud       │  ← All independent modules
    │ Module  │  │ Module      │     Loose coupling
    │         │  │             │     Easy to test
    │ run_    │  │ run_fraud_* │     Easy to replace
    │ image_* │  │             │
    └─────────┘  └─────────────┘
```

---

## Error Handling Architecture

```
┌──────────────────────────────────────────────────────┐
│           ClaimVerificationPipeline.verify_claim()   │
└──────────────────────────────────────────────────────┘
                        │
                        ▼
        ┌───────────────────────────────────┐
        │  Try-Except Wrapper               │
        │                                   │
        │  if ICD fails:                    │
        │    ├─ Return error object         │
        │    └─ Set flag: icd_failed = True │
        │                                   │
        │  if Image fails:                  │
        │    ├─ Return error object         │
        │    └─ Set flag: image_failed=True │
        │                                   │
        │  if Fraud fails:                  │
        │    ├─ Return error object         │
        │    └─ Set flag: fraud_failed=True │
        │                                   │
        │  if all failed:                   │
        │    └─ Return: manual_review       │
        │                                   │
        │  if some failed:                  │
        │    └─ Return: review + error_msg  │
        │                                   │
        │  if all succeeded:                │
        │    └─ Return: complete_report     │
        └───────────────────────────────────┘
```

---

## Example Report Structure (JSON)

```json
{
  "metadata": {
    "timestamp": "2025-01-17T14:30:45.123456",
    "claim_id": "CLM_2025_001",
    "patient_id": "PAT_001",
    "claim_amount": 1850.50,
    "previous_claim_count": 2
  },
  
  "icd_verification": {
    "success": true,
    "predicted_icds": [
      ["J18.9", 0.92],
      ["J20.9", 0.88]
    ],
    "match_score": 0.92,
    "status": "valid",
    "num_icds_detected": 2,
    "explanation": "High confidence ICD code match..."
  },
  
  "image_analysis": {
    "success": true,
    "cnn_score": 0.1234,
    "ela_score": 0.0567,
    "phash_score": 0.0,
    "fused_score": 0.0892,
    "forgery_verdict": "authentic",
    "confidence": "high",
    "ela_heatmap_path": "pipeline/outputs/xray_ela.png",
    "explanation": "LOW RISK: Image appears authentic..."
  },
  
  "fraud_assessment": {
    "success": true,
    "fraud_risk_percentage": 12.5,
    "risk_level": "low",
    "recommendation": "approve",
    "risk_factors": ["No major risk factors identified"],
    "feature_scores": { ... },
    "feature_importances": { ... },
    "explanation": "LOW FRAUD RISK: Fraud risk score is 12.5%..."
  },
  
  "integrated_verdict": {
    "overall_recommendation": "approve",
    "confidence": 0.95,
    "risk_summary": "All validation checks passed. Claim appears legitimate."
  },
  
  "explanation": "🏥 ICD CODE VALIDATION: VALID\n... [comprehensive multi-section explanation]"
}
```

---

## Deployment Topology

```
┌─────────────────────────────────────────────────────────────┐
│                   DEPLOYMENT OPTIONS                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Option 1: Standalone Python                               │
│  ┌────────────────────────────────────┐                    │
│  │ python pipeline_demo.py             │                    │
│  │ → Processes single claim            │                    │
│  │ → Generates JSON report             │                    │
│  └────────────────────────────────────┘                    │
│                                                              │
│  Option 2: FastAPI Server                                  │
│  ┌────────────────────────────────────┐                    │
│  │ uvicorn api:app --port 8000        │                    │
│  │ → /api/verify-claim (POST)          │                    │
│  │ → JSON request/response             │                    │
│  └────────────────────────────────────┘                    │
│                                                              │
│  Option 3: Docker Container                                │
│  ┌────────────────────────────────────┐                    │
│  │ docker run claim-verifier:1.0      │                    │
│  │ → Container with all dependencies   │                    │
│  │ → Kubernetes-ready                  │                    │
│  └────────────────────────────────────┘                    │
│                                                              │
│  Option 4: Serverless (AWS Lambda)                         │
│  ┌────────────────────────────────────┐                    │
│  │ Lambda handler(event, context)     │                    │
│  │ → verify_insurance_claim()          │                    │
│  │ → S3 input/output storage           │                    │
│  └────────────────────────────────────┘                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Testing Strategy

```
┌────────────────────────────────────────────────────┐
│            TESTING PYRAMID                         │
├────────────────────────────────────────────────────┤
│                                                   │
│              ▲                                    │
│             ╱ ╲                                   │
│            ╱   ╲         E2E Tests               │
│           ╱     ╲        - Full pipeline          │
│          ╱───────╲       - Report generation      │
│         ╱         ╲      - 1-2 tests              │
│        ╱───────────╲                              │
│       ╱             ╲     Integration Tests      │
│      ╱               ╲    - Module interactions   │
│     ╱─────────────────╲   - Feature vector build │
│    ╱                   ╲  - 3-4 tests             │
│   ╱─────────────────────╲                        │
│  ╱                       ╲ Unit Tests            │
│ ╱─────────────────────────╲ - Individual modules  │
│──────────────────────────── ─ Input/output        │
│                             - Error cases        │
│                             - 5+ tests           │
│                                                   │
└────────────────────────────────────────────────────┘
```

---

**Created**: January 17, 2026  
**Status**: ✅ Complete and Ready for Production
