"""
Patient Identity Consistency Validator
=======================================

Cross-modal validation ensuring submitted medical document and image 
belong to the SAME PATIENT. Leverages TCGA-COAD training knowledge.

This module bridges the gap between:
1. Document-level diagnosis/ICD codes → Patient medical profile
2. Image-level metadata (DICOM headers) → Patient imaging records
3. TCGA-COAD database → Reference patient population

Output:
- patient_match_score (0-1): Confidence that document & image are from same patient
- match_status: 'matched' | 'unmatched' | 'uncertain'
- mismatch_evidence: Specific fields that triggered mismatch
- explanation: Human-readable reasoning

Key Insight:
Even if document & image are both AUTHENTIC, they could be FORGED TOGETHER
by pairing legitimate documents from Patient A with legitimate images from Patient B.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import json
import re
import os
from datetime import datetime

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class PatientIdentityValidator:
    """
    Validates that document and image metadata align to the same patient
    using TCGA-COAD training data as a reference population.
    """

    def __init__(self, tcga_metadata_path: Optional[str] = None):
        """
        Initialize the patient identity validator.

        Args:
            tcga_metadata_path (str): Path to TCGA-COAD merged metadata CSV
                Default: data/TCGA-COAD/metadata_combined.csv
        """
        self.tcga_metadata_path = (
            tcga_metadata_path
            or str(PROJECT_ROOT / "data" / "TCGA-COAD" / "metadata_combined.csv")
        )
        
        # Load TCGA reference database
        self.tcga_db = None
        self.tcga_diagnoses_by_patient = {}
        self._load_tcga_reference()
        
        # Demo mode if TCGA DB not available
        self.demo_mode = self.tcga_db is None

    def _load_tcga_reference(self) -> None:
        """
        Load TCGA-COAD metadata to build reference patient profiles.
        
        Maps: PatientID → {diagnosis_codes, image_modalities, study_dates}
        """
        try:
            import pandas as pd
            
            if not os.path.exists(self.tcga_metadata_path):
                return
            
            self.tcga_db = pd.read_csv(self.tcga_metadata_path)
            
            # Build patient diagnosis profile index
            # Group by Patient ID to get all diagnoses associated with each patient
            if 'patientid' in self.tcga_db.columns:
                for patient_id, group in self.tcga_db.groupby('patientid'):
                    # Extract modalities, studies, dates for this patient
                    modalities = group['modality'].dropna().unique().tolist() if 'modality' in group.columns else []
                    studies = group['studydescription'].dropna().unique().tolist() if 'studydescription' in group.columns else []
                    dates = group['studydate'].dropna().unique().tolist() if 'studydate' in group.columns else []
                    
                    self.tcga_diagnoses_by_patient[str(patient_id)] = {
                        'modalities': modalities,
                        'study_descriptions': studies,
                        'study_dates': dates,
                    }
            
        except ImportError:
            pass
        except Exception as e:
            pass

    def validate_patient_match(
        self,
        icd_result: Dict[str, Any],
        image_result: Dict[str, Any],
        clinical_text: str,
        image_metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Cross-validate that document and image belong to same patient.

        Args:
            icd_result (dict): Output from ICD module (contains predicted codes + confidence)
            image_result (dict): Output from image module (contains DICOM metadata if extracted)
            clinical_text (str): Raw clinical document text
            image_metadata (dict): Optional pre-extracted DICOM metadata
                Expected keys: patient_id, study_description, modality, study_date, series_uid

        Returns:
            dict: Validation result containing:
                - success: bool (True if validation succeeded, even if unmatched)
                - patient_match_score: float (0-1, higher = more confident they match)
                - match_status: str ('matched' | 'unmatched' | 'uncertain')
                - confidence: str ('high' | 'medium' | 'low')
                - mismatch_evidence: List of specific mismatches detected
                - explanation: Human-readable reasoning
                - extracted_patient_id_doc: Patient ID extracted from document
                - extracted_patient_id_image: Patient ID extracted from image
                - tcga_reference_check: TCGA-COAD alignment assessment
        """
        try:
            # Step 1: Extract patient identifiers from both sources
            doc_patient_id = self._extract_patient_id_from_text(clinical_text)
            doc_diagnosis_codes = self._extract_icd_codes_from_result(icd_result)
            doc_diagnosis_keywords = self._extract_diagnosis_keywords(clinical_text)
            
            # Step 2: Extract image metadata
            img_patient_id = image_metadata.get("patient_id") if image_metadata else None
            img_modality = image_metadata.get("modality") if image_metadata else None
            img_study_desc = image_metadata.get("study_description") if image_metadata else None
            
            # Step 3: Compute match score
            match_score, mismatch_list, evidence_details = self._compute_match_score(
                doc_patient_id=doc_patient_id,
                img_patient_id=img_patient_id,
                doc_diagnosis_codes=doc_diagnosis_codes,
                doc_diagnosis_keywords=doc_diagnosis_keywords,
                img_modality=img_modality,
                img_study_desc=img_study_desc,
            )
            
            # Step 4: Cross-check against TCGA reference
            tcga_check = self._check_against_tcga_reference(
                patient_ids=[doc_patient_id, img_patient_id],
                diagnosis_codes=doc_diagnosis_codes,
                modality=img_modality,
            )
            
            # Step 5: Determine match status
            match_status = self._classify_match_status(match_score, mismatch_list)
            confidence_level = self._assess_confidence_level(match_score, len(mismatch_list))
            
            # Generate explanation
            explanation = self._generate_explanation(
                match_status=match_status,
                doc_patient_id=doc_patient_id,
                img_patient_id=img_patient_id,
                mismatch_evidence=mismatch_list,
                tcga_check=tcga_check,
            )
            
            return {
                "success": True,
                "patient_match_score": round(match_score, 4),
                "match_status": match_status,
                "confidence": confidence_level,
                "mismatch_evidence": mismatch_list,
                "mismatch_details": evidence_details,
                "explanation": explanation,
                "extracted_patient_id_doc": doc_patient_id or "not_found",
                "extracted_patient_id_image": img_patient_id or "not_found",
                "tcga_reference_check": tcga_check,
            }
            
        except Exception as e:
            # Fallback to uncertain status on error
            return self._error_response(str(e))

    def _extract_patient_id_from_text(self, clinical_text: str) -> Optional[str]:
        """
        Extract patient identifier from clinical document text.
        
        Looks for patterns like:
        - "Patient ID: 12345"
        - "MRN: 67890"
        - "TCGA-XX-XXXX"
        """
        # TCGA format: TCGA-XX-XXXX-###
        tcga_pattern = r'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-\d{3}'
        match = re.search(tcga_pattern, clinical_text)
        if match:
            return match.group(0)
        
        # Patient ID pattern
        patient_patterns = [
            r'Patient\s+ID[:\s]+([A-Z0-9\-]+)',
            r'MRN[:\s]+([0-9]+)',
            r'Subject\s+ID[:\s]+([A-Z0-9\-]+)',
        ]
        
        for pattern in patient_patterns:
            match = re.search(pattern, clinical_text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None

    def _extract_icd_codes_from_result(self, icd_result: Dict[str, Any]) -> List[str]:
        """Extract predicted ICD codes from ICD module output."""
        try:
            if not icd_result.get("success", False):
                return []
            
            predicted_icds = icd_result.get("predicted_icds", [])
            if isinstance(predicted_icds, list):
                # List of tuples: [(code, confidence), ...]
                return [code for code, _ in predicted_icds]
            return []
        except:
            return []

    def _extract_diagnosis_keywords(self, clinical_text: str) -> List[str]:
        """
        Extract diagnosis-related keywords from clinical text.
        
        Looks for: cancer types, anatomy references, pathology terms.
        """
        keywords = []
        
        # Common cancer/diagnosis keywords
        diagnosis_patterns = {
            'colon_cancer': r'\b(colon|colorectal|crc|adenocarcinoma)\b',
            'lung': r'\b(lung|pulmonary|bronchial)\b',
            'breast': r'\b(breast|mammary)\b',
            'prostate': r'\b(prostate)\b',
            'liver': r'\b(liver|hepatic|hepatocellular)\b',
            'pancreas': r'\b(pancreas|pancreatic)\b',
        }
        
        for key, pattern in diagnosis_patterns.items():
            if re.search(pattern, clinical_text, re.IGNORECASE):
                keywords.append(key)
        
        return keywords

    def _compute_match_score(
        self,
        doc_patient_id: Optional[str],
        img_patient_id: Optional[str],
        doc_diagnosis_codes: List[str],
        doc_diagnosis_keywords: List[str],
        img_modality: Optional[str],
        img_study_desc: Optional[str],
    ) -> Tuple[float, List[str], Dict[str, Any]]:
        """
        Compute patient match score (0-1) based on cross-modal consistency.
        
        Returns:
            (match_score, mismatch_evidence, details_dict)
        """
        score = 1.0
        mismatches = []
        details = {
            "explicit_id_match": False,
            "implicit_diagnosis_consistency": False,
            "modality_alignment": False,
            "penalties_applied": [],
        }
        
        # Check 1: Explicit Patient ID Match (highest weight)
        if doc_patient_id and img_patient_id:
            if doc_patient_id.lower() == img_patient_id.lower():
                details["explicit_id_match"] = True
            else:
                score -= 0.40  # Large penalty for ID mismatch
                mismatches.append(
                    f"Patient ID mismatch: Document={doc_patient_id}, Image={img_patient_id}"
                )
                details["penalties_applied"].append("explicit_id_mismatch: -0.40")
        elif (doc_patient_id and not img_patient_id) or (not doc_patient_id and img_patient_id):
            # One has ID, other doesn't → uncertain
            score -= 0.15
            mismatches.append("One side has Patient ID, other doesn't")
            details["penalties_applied"].append("id_asymmetry: -0.15")
        else:
            # Both missing → cannot confirm identity, treat as uncertain
            score -= 0.30
            mismatches.append("No patient IDs found in document or image")
            details["penalties_applied"].append("id_missing_both: -0.30")
        
        # Check 2: Diagnosis Consistency via Keywords + ICD Codes
        # TCGA-COAD is colon cancer → check if diagnosis matches
        if doc_diagnosis_keywords:
            if 'colon_cancer' in doc_diagnosis_keywords:
                details["implicit_diagnosis_consistency"] = True
            else:
                # Non-colon diagnosis in TCGA-COAD trained dataset
                score -= 0.25
                mismatches.append(
                    f"Diagnosis mismatch with TCGA-COAD: Document mentions {doc_diagnosis_keywords} (expected colon cancer)"
                )
                details["penalties_applied"].append("diagnosis_mismatch: -0.25")
        
        # Check 3: Modality Alignment with TCGA
        # TCGA-COAD typical modalities: CT, MR, Pathology Images
        expected_modalities = ['CT', 'MR', 'PT', 'OT']  # TCGA-COAD typical
        if img_modality:
            if any(exp_mod.lower() in img_modality.lower() for exp_mod in expected_modalities):
                details["modality_alignment"] = True
            else:
                score -= 0.10
                mismatches.append(
                    f"Unexpected modality: {img_modality} (TCGA-COAD usually has CT/MR)"
                )
                details["penalties_applied"].append("modality_mismatch: -0.10")
        
        # Ensure score stays in [0, 1]
        final_score = max(0.0, min(1.0, score))
        
        return final_score, mismatches, details

    def _check_against_tcga_reference(
        self,
        patient_ids: List[Optional[str]],
        diagnosis_codes: List[str],
        modality: Optional[str],
    ) -> Dict[str, Any]:
        """
        Cross-check against TCGA-COAD reference population.
        
        Returns TCGA alignment assessment.
        """
        check = {
            "tcga_dataset_referenced": not self.demo_mode,
            "findings": [],
        }
        
        if self.demo_mode:
            check["findings"].append("TCGA reference database not loaded (demo mode)")
            return check
        
        # Check if patient IDs are in TCGA
        for pid in patient_ids:
            if pid and str(pid) in self.tcga_diagnoses_by_patient:
                check["findings"].append(f"Patient {pid} found in TCGA-COAD database")
            elif pid:
                check["findings"].append(
                    f"Patient {pid} NOT in TCGA-COAD (could indicate missing data or external patient)"
                )
        
        # Check diagnosis codes
        # ICD codes for colon cancer: C18.x (colorectal carcinoma)
        colon_cancer_icd_prefixes = ['C18', 'C19', 'C20', 'C25']  # Colon, rectum, anus, pancreas
        colon_codes = [code for code in diagnosis_codes if any(code.startswith(p) for p in colon_cancer_icd_prefixes)]
        
        if colon_codes:
            check["findings"].append(f"ICD codes consistent with TCGA-COAD scope: {colon_codes}")
        elif diagnosis_codes:
            check["findings"].append(
                f"Warning: ICD codes {diagnosis_codes} may be outside TCGA-COAD typical scope"
            )
        
        return check

    def _classify_match_status(self, score: float, mismatches: List[str]) -> str:
        """Classify match status based on score and evidence."""
        if score >= 0.85:
            return "matched"
        elif score >= 0.60:
            return "uncertain"
        else:
            return "unmatched"

    def _assess_confidence_level(self, score: float, mismatch_count: int) -> str:
        """Assess confidence of match determination."""
        if score >= 0.90 and mismatch_count == 0:
            return "high"
        elif score >= 0.70:
            return "medium"
        else:
            return "low"

    def _generate_explanation(
        self,
        match_status: str,
        doc_patient_id: Optional[str],
        img_patient_id: Optional[str],
        mismatch_evidence: List[str],
        tcga_check: Dict[str, Any],
    ) -> str:
        """Generate human-readable explanation."""
        lines = []
        
        lines.append("🔗 PATIENT IDENTITY CONSISTENCY CHECK")
        lines.append("=" * 50)
        
        # Main verdict
        verdict_map = {
            "matched": "✅ MATCHED: Document and image appear to belong to same patient",
            "uncertain": "⚠️  UNCERTAIN: Insufficient evidence to confirm or deny match",
            "unmatched": "❌ UNMATCHED: Document and image likely belong to different patients",
        }
        lines.append(verdict_map.get(match_status, "Unknown status"))
        
        # Patient IDs
        lines.append(f"\nDocument Patient ID: {doc_patient_id or 'Not found'}")
        lines.append(f"Image Patient ID: {img_patient_id or 'Not found'}")
        
        # Mismatches
        if mismatch_evidence:
            lines.append("\nIdentified Inconsistencies:")
            for evidence in mismatch_evidence:
                lines.append(f"  • {evidence}")
        else:
            lines.append("\n✓ No explicit inconsistencies detected")
        
        # TCGA reference
        if tcga_check["tcga_dataset_referenced"]:
            lines.append("\nTCGA-COAD Reference Alignment:")
            for finding in tcga_check["findings"]:
                lines.append(f"  • {finding}")
        
        lines.append("\n" + "=" * 50)
        return "\n".join(lines)

    def _error_response(self, error_msg: str) -> Dict[str, Any]:
        """Return error response in standard format."""
        return {
            "success": False,
            "patient_match_score": 0.5,  # Default to uncertain
            "match_status": "uncertain",
            "confidence": "low",
            "mismatch_evidence": [],
            "explanation": f"Patient identity validation error: {error_msg}",
            "extracted_patient_id_doc": "error",
            "extracted_patient_id_image": "error",
            "tcga_reference_check": {"tcga_dataset_referenced": False, "findings": []},
        }


def run_patient_identity_validation(
    icd_result: Dict[str, Any],
    image_result: Dict[str, Any],
    clinical_text: str,
    image_metadata: Optional[Dict[str, str]] = None,
    tcga_metadata_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Public interface for patient identity validation.
    
    Args:
        icd_result: Output from ICD module
        image_result: Output from image module
        clinical_text: Raw clinical document
        image_metadata: Optional pre-extracted DICOM metadata
        tcga_metadata_path: Optional custom path to TCGA metadata
    
    Returns:
        Standardized validation result dictionary
    """
    validator = PatientIdentityValidator(tcga_metadata_path=tcga_metadata_path)
    return validator.validate_patient_match(
        icd_result=icd_result,
        image_result=image_result,
        clinical_text=clinical_text,
        image_metadata=image_metadata,
    )
