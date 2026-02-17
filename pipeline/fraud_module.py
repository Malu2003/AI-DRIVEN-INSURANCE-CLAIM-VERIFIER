"""
Fraud Risk Classification Module Interface
==========================================

Wrapper around the XGBoost fraud risk classifier.
Exposes a single public function: run_fraud_risk(features)

Returns standardized JSON output with:
- Fraud risk percentage (0-100)
- Risk level (low/medium/high/critical)
- Feature importances
- Risk factors identified
- Human-readable explanation
- Recommendation (approve/review/reject)
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import joblib
import numpy as np

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class FraudRiskModule:
    """Wrapper for fraud risk classification using XGBoost."""

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the fraud risk classification module.

        Args:
            model_path (str): Path to trained XGBoost model (default: fraud_risk_module/models/fraud_model.pkl)
        """
        self.model_path = (
            model_path
            or str(PROJECT_ROOT / "fraud_risk_module" / "models" / "fraud_model.pkl")
        )

        # Try to load model, use demo mode if not available
        self.demo_mode = False
        try:
            self.model = joblib.load(self.model_path)
        except Exception:
            self.demo_mode = True
            self.model = None

        # Define feature order (must match training)
        # NOTE: patient_match_score is new; if model was trained without it,
        # we'll handle it gracefully in demo mode
        self.feature_names = [
            "icd_match_score",
            "cnn_forgery_score",
            "ela_score",
            "phash_score",
            "final_image_forgery_score",
            "patient_match_score",  # NEW: Patient identity consistency (0-1)
            "claim_amount_log",
            "previous_claim_count",
        ]
        
        # Legacy feature names for backward compatibility with old models
        self.legacy_feature_names = [
            "icd_match_score",
            "cnn_forgery_score",
            "ela_score",
            "phash_score",
            "final_image_forgery_score",
            "claim_amount_log",
            "previous_claim_count",
        ]

    def run(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Classify fraud risk based on integrated features from all modules.

        Args:
            features (dict): Feature dictionary containing:
                - icd_match_score: ICD validation confidence (0-1)
                - cnn_forgery_score: CNN forgery probability (0-1)
                - ela_score: Error Level Analysis score (0-1)
                - phash_score: Perceptual hash match (0-1)
                - final_image_forgery_score: Fused image forgery score (0-1)
                - patient_match_score: Patient identity consistency (0-1, NEW)
                - claim_amount_log: Log-normalized claim amount
                - previous_claim_count: Historical claim frequency

        Returns:
            dict: Standardized output containing:
                - fraud_risk_percentage: 0-100 probability of fraud
                - risk_level: 'low' | 'medium' | 'high' | 'critical'
                - recommendation: 'approve' | 'review' | 'reject'
                - risk_factors: List of identified risk indicators
                - feature_scores: Dict of individual feature contributions
                - explanation: Human-readable summary
        """
        try:
            # Handle backward compatibility: model might be trained without patient_match_score
            feature_names_to_use = self.feature_names
            if not all(f in features for f in self.feature_names):
                # Fall back to legacy features if new feature is missing
                if all(f in features for f in self.legacy_feature_names):
                    feature_names_to_use = self.legacy_feature_names
                    # If patient_match_score missing, provide default (uncertain)
                    if "patient_match_score" not in features:
                        features = dict(features)  # Copy to avoid mutation
                        features["patient_match_score"] = 0.5
                else:
                    missing_features = set(feature_names_to_use) - set(features.keys())
                    return self._error_response(
                        f"Missing required features: {', '.join(missing_features)}"
                    )

            # Build feature vector in correct order (legacy or new)
            X = np.array(
                [[features.get(fname, 0.0) for fname in feature_names_to_use]]
            )

            # Get fraud probability
            if self.demo_mode:
                proba = self._demo_predict(features)
            elif hasattr(self.model, "predict_proba"):
                # If model expects fewer features (legacy), use only those
                if len(X[0]) > len(self.legacy_feature_names) and hasattr(self.model, "n_features_in_"):
                    # Model was trained with legacy features, trim new ones
                    if self.model.n_features_in_ == len(self.legacy_feature_names):
                        X = X[:, :len(self.legacy_feature_names)]
                proba = self.model.predict_proba(X)[0]
            else:
                proba = np.array([0.5, 0.5])
            
            fraud_risk_prob = float(proba[1])  # Probability of fraud class
            fraud_risk_percentage = round(fraud_risk_prob * 100, 2)

            # Determine risk level
            if fraud_risk_percentage >= 75:
                risk_level = "critical"
                recommendation = "reject"
            elif fraud_risk_percentage >= 50:
                risk_level = "high"
                recommendation = "review"
            elif fraud_risk_percentage >= 25:
                risk_level = "medium"
                recommendation = "review"
            else:
                risk_level = "low"
                recommendation = "approve"

            # Identify risk factors
            risk_factors = self._identify_risk_factors(features, fraud_risk_prob)

            # Get feature importances if available
            feature_importances = {}
            if hasattr(self.model, "feature_importances_"):
                importances = self.model.feature_importances_
                feature_importances = {
                    name: round(float(imp), 4)
                    for name, imp in zip(self.feature_names, importances)
                }

            # Generate explanation
            explanation = self._generate_explanation(
                fraud_risk_percentage, risk_level, risk_factors
            )

            return {
                "success": True,
                "fraud_risk_percentage": fraud_risk_percentage,
                "risk_level": risk_level,
                "recommendation": recommendation,
                "risk_factors": risk_factors,
                "feature_scores": {
                    name: round(float(features.get(name, 0.0)), 4)
                    for name in self.feature_names
                },
                "feature_importances": feature_importances,
                "explanation": explanation,
            }

        except Exception as e:
            return self._error_response(f"Fraud risk assessment error: {str(e)}")

    def _identify_risk_factors(
        self, features: Dict[str, float], fraud_risk_prob: float
    ) -> List[str]:
        """Identify key risk factors contributing to fraud assessment."""
        risk_factors = []

        # ICD mismatch indicator
        icd_score = features.get("icd_match_score", 1.0)
        if icd_score < 0.6:
            risk_factors.append(
                f"ICD code mismatch: Low confidence diagnosis validation (score: {icd_score:.2f})"
            )

        # Image forgery indicators
        image_forgery = features.get("final_image_forgery_score", 0.0)
        if image_forgery > 0.6:
            risk_factors.append(
                f"Image manipulation detected: High forgery likelihood (score: {image_forgery:.2f})"
            )

        cnn_score = features.get("cnn_forgery_score", 0.0)
        if cnn_score > 0.7:
            risk_factors.append(
                f"CNN detects image tampering: CNN forgery score {cnn_score:.2f}"
            )

        ela_score = features.get("ela_score", 0.0)
        if ela_score > 0.6:
            risk_factors.append(
                f"Error Level Analysis alert: Significant compression inconsistencies (ELA: {ela_score:.2f})"
            )

        # Claim amount indicator
        claim_amount_log = features.get("claim_amount_log", 0.0)
        if claim_amount_log > 7.0:  # Roughly $1000+
            risk_factors.append(
                f"High claim amount: Large financial exposure (log amount: {claim_amount_log:.2f})"
            )

        # Claim history indicator
        prev_claims = features.get("previous_claim_count", 0)
        if prev_claims > 5:
            risk_factors.append(
                f"Frequent claimant: Multiple previous claims on record ({int(prev_claims)})"
            )

        return risk_factors if risk_factors else ["No major risk factors identified"]

    def _generate_explanation(
        self, risk_pct: float, risk_level: str, risk_factors: List[str]
    ) -> str:
        """Generate human-readable explanation of fraud assessment."""
        level_text = {
            "critical": "CRITICAL FRAUD RISK",
            "high": "HIGH FRAUD RISK",
            "medium": "MODERATE FRAUD RISK",
            "low": "LOW FRAUD RISK",
        }

        base_explanation = (
            f"{level_text.get(risk_level, 'UNKNOWN RISK')}: "
            f"Fraud risk score is {risk_pct:.1f}%. "
        )

        if risk_factors and risk_factors[0] != "No major risk factors identified":
            base_explanation += f"Key indicators: {'; '.join(risk_factors[:2])}."
        else:
            base_explanation += "Claim appears legitimate based on all validation checks."

        return base_explanation

    def _error_response(self, message: str) -> Dict[str, Any]:
        """Generate standardized error response."""
        return {
            "success": False,
            "fraud_risk_percentage": 0.0,
            "risk_level": "error",
            "recommendation": "manual_review",
            "risk_factors": [message],
            "feature_scores": {},
            "feature_importances": {},
            "explanation": message,
        }

    def _demo_predict(self, features: Dict[str, float]) -> np.ndarray:
        """Generate demo fraud predictions based on feature heuristics."""
        import math
        
        # Extract key features
        icd_score = features.get('icd_match_score', 0.5)
        image_score = features.get('final_image_forgery_score', 0.3)
        claim_log = features.get('claim_amount_log', 5.0)
        prev_claims = features.get('previous_claim_count', 0)
        
        # Heuristic scoring
        fraud_prob = 0.0
        
        # Low ICD match increases risk
        if icd_score < 0.6:
            fraud_prob += 0.25
        
        # High image forgery score increases risk
        if image_score > 0.5:
            fraud_prob += 0.30
        
        # High claim amount increases risk
        claim_amount = math.exp(claim_log) - 1
        if claim_amount > 10000:
            fraud_prob += 0.15
        elif claim_amount > 5000:
            fraud_prob += 0.10
        
        # Many previous claims increases risk
        if prev_claims > 3:
            fraud_prob += 0.20
        elif prev_claims > 1:
            fraud_prob += 0.10
        
        # Clamp to [0, 1]
        fraud_prob = min(1.0, max(0.0, fraud_prob))
        
        return np.array([1 - fraud_prob, fraud_prob])


def run_fraud_risk(
    features: Dict[str, float], model_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Public interface: Assess fraud risk using integrated module outputs.

    Args:
        features (dict): Feature dictionary from integrated modules
        model_path (str): Path to trained XGBoost model (optional)

    Returns:
        dict: Standardized fraud risk assessment results
    """
    module = FraudRiskModule(model_path=model_path)
    return module.run(features)
