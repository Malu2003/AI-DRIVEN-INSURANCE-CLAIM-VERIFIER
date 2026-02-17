"""
Integration Test Suite for Claim Verification Pipeline
========================================================

Comprehensive tests for all pipeline modules and integrations.

Run with: pytest pipeline/tests.py -v
Or: python pipeline/tests.py
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import pipeline modules
try:
    from pipeline.icd_module import run_icd_verification
    from pipeline.image_module import run_image_forgery
    from pipeline.fraud_module import run_fraud_risk
    from pipeline.claim_verification_pipeline import verify_insurance_claim
    IMPORTS_OK = True
except ImportError as e:
    IMPORTS_OK = False
    IMPORT_ERROR = str(e)


class TestResults:
    """Track test results."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def add_pass(self, test_name: str):
        self.passed += 1
        print(f"✅ {test_name}")

    def add_fail(self, test_name: str, reason: str):
        self.failed += 1
        self.errors.append(f"{test_name}: {reason}")
        print(f"❌ {test_name}: {reason}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*70}")
        print(f"TEST SUMMARY: {self.passed}/{total} passed")
        if self.errors:
            print("\nFailures:")
            for error in self.errors:
                print(f"  • {error}")
        print(f"{'='*70}")
        return self.failed == 0


def test_imports():
    """Test that all modules can be imported."""
    results = TestResults()

    if IMPORTS_OK:
        results.add_pass("Module imports")
    else:
        results.add_fail("Module imports", IMPORT_ERROR)

    return results


def test_icd_module():
    """Test ICD validation module."""
    results = TestResults()

    if not IMPORTS_OK:
        results.add_fail("ICD module", "Imports failed")
        return results

    try:
        # Test with valid input
        clinical_text = "Patient with pneumonia (J18.9) and bronchitis (J20.9)"
        result = run_icd_verification(clinical_text)

        assert isinstance(result, dict), "Output must be dict"
        assert "success" in result, "Missing 'success' key"
        assert "predicted_icds" in result, "Missing 'predicted_icds' key"
        assert "match_score" in result, "Missing 'match_score' key"
        assert "status" in result, "Missing 'status' key"

        results.add_pass("ICD module: valid input")

        # Verify output types
        assert isinstance(result["predicted_icds"], list), "predicted_icds must be list"
        assert isinstance(result["match_score"], (int, float)), "match_score must be numeric"
        assert result["status"] in ["valid", "uncertain", "flagged", "error"], "Invalid status"

        results.add_pass("ICD module: output types")

    except AssertionError as e:
        results.add_fail("ICD module", str(e))
    except Exception as e:
        results.add_fail("ICD module", f"Unexpected error: {str(e)}")

    return results


def test_image_module():
    """Test image forgery detection module."""
    results = TestResults()

    if not IMPORTS_OK:
        results.add_fail("Image module", "Imports failed")
        return results

    try:
        # Find a sample image
        sample_images = list((PROJECT_ROOT / "data" / "CASIA2" / "val").glob("*/*"))
        if not sample_images:
            results.add_fail("Image module", "No sample images found in data/CASIA2/val")
            return results

        image_path = str(sample_images[0])

        # Test image forgery detection
        result = run_image_forgery(image_path)

        assert isinstance(result, dict), "Output must be dict"
        assert "success" in result, "Missing 'success' key"
        assert "cnn_score" in result, "Missing 'cnn_score' key"
        assert "ela_score" in result, "Missing 'ela_score' key"
        assert "fused_score" in result, "Missing 'fused_score' key"
        assert "forgery_verdict" in result, "Missing 'forgery_verdict' key"

        results.add_pass("Image module: valid input")

        # Verify output types and ranges
        assert 0 <= result["cnn_score"] <= 1, "CNN score out of range"
        assert 0 <= result["ela_score"] <= 1, "ELA score out of range"
        assert 0 <= result["fused_score"] <= 1, "Fused score out of range"
        assert result["forgery_verdict"] in [
            "authentic",
            "suspicious",
            "tampered",
            "error",
        ], "Invalid verdict"

        results.add_pass("Image module: output validation")

    except AssertionError as e:
        results.add_fail("Image module", str(e))
    except Exception as e:
        results.add_fail("Image module", f"Unexpected error: {str(e)}")

    return results


def test_fraud_module():
    """Test fraud risk classification module."""
    results = TestResults()

    if not IMPORTS_OK:
        results.add_fail("Fraud module", "Imports failed")
        return results

    try:
        # Test with valid features
        features = {
            "icd_match_score": 0.9,
            "cnn_forgery_score": 0.1,
            "ela_score": 0.05,
            "phash_score": 0.0,
            "final_image_forgery_score": 0.08,
            "claim_amount_log": 7.5,
            "previous_claim_count": 2,
        }

        result = run_fraud_risk(features)

        assert isinstance(result, dict), "Output must be dict"
        assert "success" in result, "Missing 'success' key"
        assert "fraud_risk_percentage" in result, "Missing 'fraud_risk_percentage' key"
        assert "risk_level" in result, "Missing 'risk_level' key"
        assert "recommendation" in result, "Missing 'recommendation' key"

        results.add_pass("Fraud module: valid input")

        # Verify output types and ranges
        assert 0 <= result["fraud_risk_percentage"] <= 100, "Risk percentage out of range"
        assert result["risk_level"] in [
            "low",
            "medium",
            "high",
            "critical",
            "error",
        ], "Invalid risk level"
        assert result["recommendation"] in [
            "approve",
            "review",
            "reject",
            "manual_review",
        ], "Invalid recommendation"

        results.add_pass("Fraud module: output validation")

    except AssertionError as e:
        results.add_fail("Fraud module", str(e))
    except Exception as e:
        results.add_fail("Fraud module", f"Unexpected error: {str(e)}")

    return results


def test_integrated_pipeline():
    """Test the complete integrated pipeline."""
    results = TestResults()

    if not IMPORTS_OK:
        results.add_fail("Integrated pipeline", "Imports failed")
        return results

    try:
        # Prepare test data
        clinical_text = "Patient diagnosed with acute pneumonia (J18.9). Chest X-ray shows consolidation."

        # Find sample image
        sample_images = list((PROJECT_ROOT / "data" / "CASIA2" / "val").glob("*/*"))
        if not sample_images:
            results.add_fail("Integrated pipeline", "No sample images found")
            return results

        image_path = str(sample_images[0])
        claim_amount = 1850.50
        previous_claim_count = 2

        # Run pipeline
        result = verify_insurance_claim(
            clinical_text=clinical_text,
            image_path=image_path,
            claim_amount=claim_amount,
            previous_claim_count=previous_claim_count,
            claim_id="TEST_001",
        )

        assert isinstance(result, dict), "Output must be dict"
        assert "metadata" in result, "Missing 'metadata' key"
        assert "icd_verification" in result, "Missing 'icd_verification' key"
        assert "image_analysis" in result, "Missing 'image_analysis' key"
        assert "fraud_assessment" in result, "Missing 'fraud_assessment' key"
        assert "integrated_verdict" in result, "Missing 'integrated_verdict' key"

        results.add_pass("Integrated pipeline: all modules executed")

        # Verify integrated verdict structure
        verdict = result["integrated_verdict"]
        assert "overall_recommendation" in verdict, "Missing recommendation"
        assert "confidence" in verdict, "Missing confidence"
        assert verdict["overall_recommendation"] in [
            "approve",
            "review",
            "reject",
            "manual_review",
        ], "Invalid recommendation"

        results.add_pass("Integrated pipeline: verdict structure")

        # Verify JSON serializable
        json_str = json.dumps(result)
        assert len(json_str) > 0, "JSON serialization failed"

        results.add_pass("Integrated pipeline: JSON serializable")

    except AssertionError as e:
        results.add_fail("Integrated pipeline", str(e))
    except Exception as e:
        results.add_fail("Integrated pipeline", f"Unexpected error: {str(e)}")

    return results


def main():
    """Run all tests."""
    print("=" * 70)
    print("CLAIM VERIFICATION PIPELINE - INTEGRATION TESTS")
    print("=" * 70)

    all_results = TestResults()

    # Run test suites
    test_suites = [
        ("Imports", test_imports),
        ("ICD Module", test_icd_module),
        ("Image Module", test_image_module),
        ("Fraud Module", test_fraud_module),
        ("Integrated Pipeline", test_integrated_pipeline),
    ]

    for suite_name, test_func in test_suites:
        print(f"\n🧪 {suite_name}:")
        print("-" * 70)
        suite_results = test_func()
        all_results.passed += suite_results.passed
        all_results.failed += suite_results.failed
        all_results.errors.extend(suite_results.errors)

    # Print summary
    all_results.summary()

    # Return exit code
    return 0 if all_results.failed == 0 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
