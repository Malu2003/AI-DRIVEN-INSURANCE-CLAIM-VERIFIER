"""
Pipeline Package
================

Integrated claim verification pipeline combining:
- ICD Code Validation (NLP)
- Medical Image Forgery Detection (Vision)
- Fraud Risk Classification (ML)
"""

from pipeline.icd_module import run_icd_verification
from pipeline.image_module import run_image_forgery
from pipeline.fraud_module import run_fraud_risk
from pipeline.claim_verification_pipeline import (
    ClaimVerificationPipeline,
    verify_insurance_claim,
)

__all__ = [
    "run_icd_verification",
    "run_image_forgery",
    "run_fraud_risk",
    "ClaimVerificationPipeline",
    "verify_insurance_claim",
]

__version__ = "1.0.0"
