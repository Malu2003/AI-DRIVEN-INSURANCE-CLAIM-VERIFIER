"""
Pipeline Demo and Testing Script
=================================

Demonstrates the complete end-to-end claim verification pipeline
with mock and real test cases.

Usage:
    python pipeline_demo.py [--mode mock|real] [--claim-id CLAIM123]
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import argparse

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.claim_verification_pipeline import verify_insurance_claim


def create_mock_test_case() -> tuple:
    """
    Create a realistic mock test case for demonstration.

    Returns:
        tuple: (clinical_text, image_path, claim_amount, previous_claim_count)
    """
    clinical_text = """
    DISCHARGE SUMMARY
    Patient: Jane Doe, DOB: 01/15/1975
    Admission Date: 2025-01-10
    Discharge Date: 2025-01-17
    
    DIAGNOSIS:
    1. Acute pneumonia - ICD-10: J18.9
    2. Severe acute bronchitis - ICD-10: J20.9
    3. Hypoxemia secondary to pneumonia
    
    CLINICAL PRESENTATION:
    Patient presented with persistent cough, fever (101.2°F), and dyspnea.
    Chest X-ray showed consolidation in right lower lobe. CT scan confirmed
    bilateral pneumonic infiltrates with no pleural effusion.
    
    TREATMENT:
    - IV antibiotics (azithromycin 500mg daily x 3 days)
    - Oxygen support (2L nasal cannula)
    - Antipyretics and supportive care
    - Daily monitoring with pulse oximetry
    
    MEDICATIONS AT DISCHARGE:
    - Amoxicillin-clavulanate 875/125mg BID x 7 days
    - Guaifenesin with dextromethorphan as needed
    - Acetaminophen 500mg TID as needed
    
    CLINICAL OUTCOME:
    Patient showed good response to treatment. Fever resolved by day 3.
    Oxygen saturation improved to 96-98% on room air at discharge.
    Patient educated on follow-up care and symptom monitoring.
    
    Follow-up: Chest X-ray in 4 weeks to confirm resolution
    """

    # Use a sample image from the CASIA dataset if available
    sample_image = PROJECT_ROOT / "data" / "CASIA2" / "val" / "authentic" / "Au_000.jpg"
    if not sample_image.exists():
        # Try alternative paths
        alt_images = list((PROJECT_ROOT / "data" / "CASIA2" / "val").glob("*/*/*.jpg"))
        if alt_images:
            sample_image = str(alt_images[0])
        else:
            sample_image = None

    claim_amount = 1850.50  # Realistic medical imaging + treatment cost
    previous_claim_count = 2

    return clinical_text, str(sample_image) if sample_image else None, claim_amount, previous_claim_count


def run_mock_test():
    """Run pipeline with mock test data."""
    print("\n" + "=" * 80)
    print("MOCK TEST: Claim Verification Pipeline")
    print("=" * 80)

    clinical_text, image_path, claim_amount, prev_claims = create_mock_test_case()

    if image_path is None or not Path(image_path).exists():
        print("\n⚠️  Warning: Sample image not found. Using placeholder.")
        print("   To run full tests, ensure CASIA2 data is available at: data/CASIA2/")
        image_path = "data/CASIA2/val/authentic/Au_000.jpg"

    print(f"\n📋 CLAIM DETAILS:")
    print(f"   Clinical Text: {len(clinical_text)} characters")
    print(f"   Image Path: {image_path}")
    print(f"   Claim Amount: ${claim_amount:.2f}")
    print(f"   Previous Claims: {prev_claims}")

    print(f"\n🔄 Running claim verification pipeline...")
    print(f"   Step 1: ICD Code Validation")
    print(f"   Step 2: Image Forgery Detection")
    print(f"   Step 3: Fraud Risk Assessment")

    result = verify_insurance_claim(
        clinical_text=clinical_text,
        image_path=image_path,
        claim_amount=claim_amount,
        previous_claim_count=prev_claims,
        patient_id="PAT_001",
        claim_id="CLM_2025_001",
    )

    # Print results
    print_results(result)

    return result


def print_results(result: dict) -> None:
    """Pretty-print pipeline results."""
    print("\n" + "=" * 80)
    print("VERIFICATION RESULTS")
    print("=" * 80)

    # Check for errors
    if not result.get("success", True) and "error" in result:
        print(f"\n❌ Pipeline Error: {result['error']}")
        return

    # Metadata
    metadata = result.get("metadata", {})
    print(f"\n📌 CLAIM INFORMATION:")
    print(f"   Claim ID: {metadata.get('claim_id', 'N/A')}")
    print(f"   Patient ID: {metadata.get('patient_id', 'N/A')}")
    print(f"   Amount: ${metadata.get('claim_amount', 0):.2f}")
    print(f"   Timestamp: {metadata.get('timestamp', 'N/A')}")

    # ICD Verification
    icd = result.get("icd_verification", {})
    if icd.get("success"):
        print(f"\n🏥 ICD CODE VALIDATION:")
        print(f"   Status: {icd.get('status', 'N/A').upper()}")
        print(f"   Match Score: {icd.get('match_score', 0):.3f}")
        print(f"   Codes Found: {icd.get('num_icds_detected', 0)}")
        if icd.get("predicted_icds"):
            print(f"   Top ICDs:")
            for code, conf in icd.get("predicted_icds", [])[:3]:
                print(f"      - {code}: {conf:.3f}")
        print(f"   ➜ {icd.get('explanation', 'No details')}")
    else:
        print(f"\n🏥 ICD CODE VALIDATION: ERROR")
        print(f"   ➜ {icd.get('explanation', 'Unknown error')}")

    # Image Analysis
    image = result.get("image_analysis", {})
    if image.get("success"):
        print(f"\n🖼️  IMAGE FORGERY ANALYSIS:")
        print(f"   Verdict: {image.get('forgery_verdict', 'N/A').upper()}")
        print(f"   Confidence: {image.get('confidence', 'N/A').upper()}")
        print(f"   CNN Score: {image.get('cnn_score', 0):.3f}")
        print(f"   ELA Score: {image.get('ela_score', 0):.3f}")
        if image.get("phash_score") is not None:
            print(f"   pHash Score: {image.get('phash_score'):.3f}")
        print(f"   Fused Score: {image.get('fused_score', 0):.3f}")
        if image.get("ela_heatmap_path"):
            print(f"   ELA Heatmap: {image.get('ela_heatmap_path')}")
        print(f"   ➜ {image.get('explanation', 'No details')}")
    else:
        print(f"\n🖼️  IMAGE FORGERY ANALYSIS: ERROR")
        print(f"   ➜ {image.get('explanation', 'Unknown error')}")

    # Fraud Risk Assessment
    fraud = result.get("fraud_assessment", {})
    if fraud.get("success"):
        print(f"\n⚠️  FRAUD RISK ASSESSMENT:")
        print(f"   Risk Level: {fraud.get('risk_level', 'N/A').upper()}")
        print(f"   Fraud Risk: {fraud.get('fraud_risk_percentage', 0):.1f}%")
        print(f"   Recommendation: {fraud.get('recommendation', 'N/A').upper()}")
        if fraud.get("risk_factors"):
            print(f"   Risk Factors:")
            for factor in fraud.get("risk_factors", [])[:3]:
                print(f"      • {factor}")
        print(f"   ➜ {fraud.get('explanation', 'No details')}")
    else:
        print(f"\n⚠️  FRAUD RISK ASSESSMENT: ERROR")
        print(f"   ➜ {fraud.get('explanation', 'Unknown error')}")

    # Integrated Verdict
    verdict = result.get("integrated_verdict", {})
    print(f"\n" + "=" * 80)
    print(f"✅ FINAL DECISION")
    print(f"=" * 80)
    print(f"   Recommendation: {verdict.get('overall_recommendation', 'N/A').upper()}")
    print(f"   Confidence: {verdict.get('confidence', 0):.0%}")
    print(f"   Summary: {verdict.get('risk_summary', 'No summary available')}")

    # Full explanation
    print(f"\n📝 DETAILED EXPLANATION:")
    explanation = result.get("explanation", "No explanation available")
    for line in explanation.split("\n"):
        print(f"   {line}")

    # Save location
    print(f"\n💾 Report saved to: pipeline/outputs/")


def run_interactive_test():
    """Run pipeline with user-provided inputs."""
    print("\n" + "=" * 80)
    print("INTERACTIVE TEST: Claim Verification Pipeline")
    print("=" * 80)

    print("\n📋 Enter claim details:")

    # Get clinical text
    print("\n1. Clinical Text (discharge summary):")
    print("   Enter text (press Enter twice to finish):")
    lines = []
    while True:
        line = input()
        if line:
            lines.append(line)
        else:
            if lines:
                break
    clinical_text = "\n".join(lines)

    # Get image path
    image_path = input("\n2. Medical Image Path: ").strip()
    if not Path(image_path).exists():
        print(f"   ⚠️  Warning: File not found at {image_path}")

    # Get claim amount
    try:
        claim_amount = float(input("\n3. Claim Amount ($): "))
    except ValueError:
        claim_amount = 1000.0

    # Get previous claim count
    try:
        prev_claims = int(input("4. Previous Claim Count: "))
    except ValueError:
        prev_claims = 0

    print(f"\n🔄 Running claim verification pipeline...")

    result = verify_insurance_claim(
        clinical_text=clinical_text,
        image_path=image_path,
        claim_amount=claim_amount,
        previous_claim_count=prev_claims,
        claim_id=f"CLM_{datetime.now().strftime('%Y%m%d%H%M%S')}",
    )

    print_results(result)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Claim Verification Pipeline Demo & Testing"
    )
    parser.add_argument(
        "--mode",
        choices=["mock", "interactive"],
        default="mock",
        help="Test mode: mock (default) or interactive",
    )
    parser.add_argument(
        "--claim-id", default=None, help="Custom claim ID for tracking"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress output")

    args = parser.parse_args()

    if args.mode == "interactive":
        run_interactive_test()
    else:
        run_mock_test()

    if not args.quiet:
        print(f"\n" + "=" * 80)
        print("✅ Pipeline execution completed")
        print("=" * 80)


if __name__ == "__main__":
    main()
