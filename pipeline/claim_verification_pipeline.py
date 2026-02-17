"""
End-to-End Claim Verification Pipeline Orchestrator
=====================================================

Integrates ICD validation, image forgery detection, and fraud risk scoring
into a single unified claim verification pipeline.

Input:
  - Clinical text (discharge summary, medical notes)
  - Medical image path
  - Claim metadata (amount, previous claim count)

Output:
  - Unified JSON response with results from all three modules
  - Human-readable explanations and verdicts
  - Recommendations for approval/review/rejection
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import json
from datetime import datetime
import numpy as np

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.icd_module import run_icd_verification
from pipeline.image_module import run_image_forgery
from pipeline.fraud_module import run_fraud_risk
from pipeline.patient_identity_validator import run_patient_identity_validation


class ClaimVerificationPipeline:
    """
    Unified claim verification pipeline orchestrator.
    
    Combines results from three independent modules:
    1. ICD Code Validation (NLP)
    2. Medical Image Forgery Detection (Vision)
    3. Fraud Risk Classification (ML)
    """

    def __init__(
        self,
        model_ckpt: Optional[str] = None,
        phash_db: Optional[str] = None,
        fraud_model: Optional[str] = None,
    ):
        """
        Initialize the claim verification pipeline.

        Args:
            model_ckpt (str): Path to CNN checkpoint (optional)
            phash_db (str): Path to pHash database (optional)
            fraud_model (str): Path to fraud risk model (optional)
        """
        # Use LC25000 fine-tuned model by default
        self.model_ckpt = (
            model_ckpt or str(PROJECT_ROOT / "checkpoints" / "lc25000_forgery" / "best.pth.tar")
        )
        # Use LC25000-specific pHash database (medical images compared to medical images)
        self.phash_db = (
            phash_db or str(PROJECT_ROOT / "data" / "phash_lc25000_authentic.csv")
        )
        self.fraud_model = fraud_model
        self.output_dir = str(PROJECT_ROOT / "pipeline" / "outputs")

    def verify_claim(
        self,
        clinical_text: str,
        image_path: str,
        claim_amount: float,
        previous_claim_count: int = 0,
        patient_id: Optional[str] = None,
        claim_id: Optional[str] = None,
        save_output: bool = True,
    ) -> Dict[str, Any]:
        """
        Run complete claim verification pipeline.

        Args:
            clinical_text (str): Patient discharge summary or clinical notes
            image_path (str): Path to medical image (CT, X-ray, etc.)
            claim_amount (float): Insurance claim amount in dollars
            previous_claim_count (int): Number of prior claims by patient
            patient_id (str): Patient identifier (optional)
            claim_id (str): Claim identifier (optional)
            save_output (bool): Whether to save output JSON (default: True)

        Returns:
            dict: Unified claim verification report containing:
                - metadata: Claim and timestamp info
                - icd_verification: ICD code validation results
                - image_analysis: Image forgery detection results
                - fraud_assessment: Fraud risk classification results
                - integrated_verdict: Final recommendation and risk level
                - explanation: Comprehensive summary for user
        """
        try:
            # Timestamp
            timestamp = datetime.now().isoformat()

            # Step 1: ICD Code Validation
            icd_result = self._run_icd_module(clinical_text)

            # Step 2: Image Forgery Detection
            image_result = self._run_image_module(image_path)

            # Step 3: PATIENT IDENTITY CONSISTENCY VALIDATION (NEW)
            # Ensures document and image belong to same patient
            patient_identity_result = self._run_patient_identity_validator(
                icd_result=icd_result,
                image_result=image_result,
                clinical_text=clinical_text,
            )

            # Step 4: Build Feature Vector for Fraud Risk
            # Now includes patient_match_score
            feature_vector = self._build_feature_vector(
                icd_result=icd_result,
                image_result=image_result,
                patient_identity_result=patient_identity_result,
                claim_amount=claim_amount,
                previous_claim_count=previous_claim_count,
            )

            # Step 5: Fraud Risk Assessment
            fraud_result = self._run_fraud_module(feature_vector)

            # Step 6: Generate Integrated Verdict
            integrated_verdict = self._compute_integrated_verdict(
                icd_result=icd_result,
                image_result=image_result,
                patient_identity_result=patient_identity_result,
                fraud_result=fraud_result,
            )

            # Compile final report
            report = {
                "metadata": {
                    "timestamp": timestamp,
                    "patient_id": patient_id,
                    "claim_id": claim_id,
                    "claim_amount": claim_amount,
                    "previous_claim_count": previous_claim_count,
                },
                "icd_verification": icd_result,
                "image_analysis": image_result,
                "patient_identity_validation": patient_identity_result,
                "fraud_assessment": fraud_result,
                "integrated_verdict": integrated_verdict,
                "explanation": self._generate_final_explanation(
                    icd_result=icd_result,
                    image_result=image_result,
                    patient_identity_result=patient_identity_result,
                    fraud_result=fraud_result,
                    integrated_verdict=integrated_verdict,
                ),
            }

            # Save output if requested
            if save_output:
                self._save_report(report, claim_id or "unknown")

            return report

        except Exception as e:
            return self._error_report(str(e))

    def _run_icd_module(self, clinical_text: str) -> Dict[str, Any]:
        """Run ICD validation module."""
        try:
            result = run_icd_verification(clinical_text, top_k=5)
            return result
        except Exception as e:
            return {
                "success": False,
                "status": "error",
                "explanation": f"ICD module error: {str(e)}",
            }

    def _run_image_module(self, image_path: str) -> Dict[str, Any]:
        """Run image forgery detection module."""
        try:
            import os

            os.makedirs(self.output_dir, exist_ok=True)
            result = run_image_forgery(
                image_path,
                model_ckpt=self.model_ckpt,
                phash_db=self.phash_db,
                output_dir=self.output_dir,
            )
            return result
        except Exception as e:
            return {
                "success": False,
                "forgery_verdict": "error",
                "explanation": f"Image module error: {str(e)}",
            }

    def _run_patient_identity_validator(
        self,
        icd_result: Dict[str, Any],
        image_result: Dict[str, Any],
        clinical_text: str,
    ) -> Dict[str, Any]:
        """
        Run patient identity consistency validation.
        
        Ensures that the submitted document and image belong to the same patient,
        leveraging TCGA-COAD training data as a reference.
        """
        try:
            result = run_patient_identity_validation(
                icd_result=icd_result,
                image_result=image_result,
                clinical_text=clinical_text,
            )
            return result
        except Exception as e:
            return {
                "success": False,
                "patient_match_score": 0.5,
                "match_status": "uncertain",
                "explanation": f"Patient identity validation error: {str(e)}",
            }

    def _run_fraud_module(self, feature_vector: Dict[str, float]) -> Dict[str, Any]:
        """Run fraud risk classification module."""
        try:
            result = run_fraud_risk(feature_vector, model_path=self.fraud_model)
            return result
        except Exception as e:
            return {
                "success": False,
                "risk_level": "error",
                "explanation": f"Fraud module error: {str(e)}",
            }

    def _build_feature_vector(
        self,
        icd_result: Dict[str, Any],
        image_result: Dict[str, Any],
        patient_identity_result: Dict[str, Any],
        claim_amount: float,
        previous_claim_count: int,
    ) -> Dict[str, float]:
        """
        Build feature vector from module outputs for fraud risk classifier.

        Combines signals from ICD validation, image analysis, and patient identity
        consistency with contextual features.
        """
        import math

        # Handle None values by converting to 0.0
        phash_score = image_result.get("phash_score")
        phash_score = 0.0 if phash_score is None else float(phash_score)
        
        # Patient match score: 1.0 = perfect match, 0.0 = clear mismatch
        patient_match_score = float(patient_identity_result.get("patient_match_score", 0.5))
        
        return {
            "icd_match_score": float(icd_result.get("match_score", 0.0)),
            "cnn_forgery_score": float(image_result.get("cnn_score", 0.0)),
            "ela_score": float(image_result.get("ela_score", 0.0)),
            "phash_score": phash_score,
            "final_image_forgery_score": float(image_result.get("fused_score", 0.0)),
            "patient_match_score": patient_match_score,  # NEW: Patient identity consistency
            "claim_amount_log": float(math.log1p(claim_amount)),
            "previous_claim_count": float(previous_claim_count),
        }

    def _compute_integrated_verdict(
        self,
        icd_result: Dict[str, Any],
        image_result: Dict[str, Any],
        patient_identity_result: Dict[str, Any],
        fraud_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compute integrated verdict across all four modules.

        Returns unified recommendation based on consensus of all signals,
        including patient identity consistency.
        """
        verdict = {
            "overall_recommendation": "manual_review",
            "confidence": 0.0,
            "risk_summary": "",
        }

        if (
            not icd_result.get("success", False)
            or not image_result.get("success", False)
            or not fraud_result.get("success", False)
        ):
            verdict["overall_recommendation"] = "manual_review"
            verdict["risk_summary"] = "One or more modules encountered errors; manual review required."
            return verdict

        # Scoring logic with patient identity check
        icd_valid = icd_result.get("status") == "valid"
        image_authentic = image_result.get("forgery_verdict") == "authentic"
        patient_matched = patient_identity_result.get("match_status") == "matched"
        fraud_low_risk = fraud_result.get("risk_level") == "low"

        # Count positive signals (now includes patient identity)
        positive_signals = sum(
            [icd_valid, image_authentic, patient_matched, fraud_low_risk]
        )

        # KEY INSIGHT: Even if document & image are both authentic,
        # a mismatch in patient identity should downgrade approval
        if patient_identity_result.get("match_status") == "unmatched":
            # Force review even if other signals are good
            verdict["overall_recommendation"] = "review"
            verdict["confidence"] = 0.85
            verdict["risk_summary"] = (
                "⚠️  DOCUMENT-IMAGE MISMATCH DETECTED: "
                "The submitted document and medical image appear to belong to different patients. "
                "Even though both may be authentic, they are not linked. Manual review required."
            )
            return verdict

        # Standard multi-signal scoring
        if positive_signals == 4:
            verdict["overall_recommendation"] = "approve"
            verdict["confidence"] = 0.98
            verdict["risk_summary"] = "All validation checks passed. Claim appears legitimate and all documents link to same patient."

        elif positive_signals == 3:
            verdict["overall_recommendation"] = "approve"
            verdict["confidence"] = 0.90
            verdict["risk_summary"] = "Most validation checks passed. Patient identity and primary claims verified."

        elif positive_signals == 2:
            verdict["overall_recommendation"] = "review"
            verdict["confidence"] = 0.65
            verdict["risk_summary"] = "Some validation checks flagged. Manual review recommended."

        elif positive_signals == 1:
            verdict["overall_recommendation"] = "review"
            verdict["confidence"] = 0.50
            verdict["risk_summary"] = "Multiple validation checks flagged. Manual review required."

        else:
            verdict["overall_recommendation"] = "reject"
            verdict["confidence"] = 0.92
            verdict["risk_summary"] = "Critical issues detected across validation modules."

        return verdict

    def _generate_final_explanation(
        self,
        icd_result: Dict[str, Any],
        image_result: Dict[str, Any],
        patient_identity_result: Dict[str, Any],
        fraud_result: Dict[str, Any],
        integrated_verdict: Dict[str, Any],
    ) -> str:
        """Generate comprehensive human-readable explanation."""
        sections = []

        # ICD section
        icd_status = icd_result.get("status", "unknown")
        sections.append(
            f"🏥 ICD CODE VALIDATION: {icd_status.upper()}\n"
            f"   {icd_result.get('explanation', 'No details available')}"
        )

        # Image section
        image_verdict = image_result.get("forgery_verdict", "unknown")
        sections.append(
            f"\n🖼️  IMAGE ANALYSIS: {image_verdict.upper()}\n"
            f"   {image_result.get('explanation', 'No details available')}"
        )

        # Patient Identity section (NEW)
        patient_match_status = patient_identity_result.get("match_status", "unknown")
        patient_match_score = patient_identity_result.get("patient_match_score", 0.0)
        sections.append(
            f"\n🔗 PATIENT IDENTITY CHECK: {patient_match_status.upper()} (Score: {patient_match_score:.2f})\n"
            f"   {patient_identity_result.get('explanation', 'No details available')}"
        )

        # Fraud risk section
        fraud_level = fraud_result.get("risk_level", "unknown")
        sections.append(
            f"\n⚠️  FRAUD RISK ASSESSMENT: {fraud_level.upper()}\n"
            f"   {fraud_result.get('explanation', 'No details available')}"
        )

        # Integrated verdict
        sections.append(
            f"\n✅ FINAL VERDICT: {integrated_verdict['overall_recommendation'].upper()}\n"
            f"   {integrated_verdict['risk_summary']}"
        )

        return "\n".join(sections)

    def _save_report(self, report: Dict[str, Any], claim_id: str) -> None:
        """Save verification report to JSON file."""
        import os

        os.makedirs(self.output_dir, exist_ok=True)
        output_path = Path(self.output_dir) / f"claim_{claim_id}_report.json"
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

    def _error_report(self, error_message: str) -> Dict[str, Any]:
        """Generate error report."""
        return {
            "success": False,
            "error": error_message,
            "timestamp": datetime.now().isoformat(),
            "integrated_verdict": {
                "overall_recommendation": "manual_review",
                "confidence": 0.0,
                "risk_summary": f"Pipeline error: {error_message}",
            },
        }


def verify_insurance_claim(
    clinical_text: str,
    image_path: str,
    claim_amount: float,
    previous_claim_count: int = 0,
    patient_id: Optional[str] = None,
    claim_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Public interface: Verify an insurance claim using the complete pipeline.

    Args:
        clinical_text (str): Patient discharge summary or medical notes
        image_path (str): Path to medical image file
        claim_amount (float): Claim amount in dollars
        previous_claim_count (int): Number of previous claims (default: 0)
        patient_id (str): Patient ID (optional)
        claim_id (str): Claim ID (optional)

    Returns:
        dict: Complete claim verification report
    """
    pipeline = ClaimVerificationPipeline()
    return pipeline.verify_claim(
        clinical_text=clinical_text,
        image_path=image_path,
        claim_amount=claim_amount,
        previous_claim_count=previous_claim_count,
        patient_id=patient_id,
        claim_id=claim_id,
        save_output=True,
    )
