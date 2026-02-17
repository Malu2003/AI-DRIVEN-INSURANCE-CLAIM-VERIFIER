"""Load trained fraud model and produce risk prediction with explanation.

Example output JSON:
{
  "fraud_risk_percentage": 82.3,
  "risk_level": "HIGH",
  "explanation": {
    "icd_mismatch": true,
    "image_forgery_detected": true,
    "key_drivers": ["Low ICD match score", "High image forgery confidence"]
  }
}
"""
import joblib
import json
from pathlib import Path
import numpy as np

MODEL_PATH = Path(__file__).parent / 'models' / 'fraud_model.pkl'


def load_model(path: str = None):
    p = Path(path) if path else MODEL_PATH
    if not p.exists():
        raise FileNotFoundError(f'Model not found: {p} — run train_xgboost.py first')
    return joblib.load(p)


def risk_level_from_pct(pct: float) -> str:
    if pct <= 30:
        return 'LOW'
    if pct <= 70:
        return 'MEDIUM'
    return 'HIGH'


def explain_features(feature_names, importances, top_k=3):
    # Use feature importances to pick top drivers
    idx = np.argsort(-importances)[:top_k]
    return [feature_names[i] for i in idx]


def predict_risk(model, features: dict) -> dict:
    # Build feature vector in order expected by training
    feat_order = ['icd_match_score', 'cnn_forgery_score', 'ela_score', 'phash_score', 'final_image_forgery_score', 'claim_amount_log', 'previous_claim_count']
    x = np.array([features.get(k, 0.0) for k in feat_order], dtype=float).reshape(1, -1)

    prob = float(model.predict_proba(x)[0, 1])
    pct = round(prob * 100.0, 2)
    level = risk_level_from_pct(pct)

    # Explanation pieces
    icd_mismatch = features.get('icd_match_score', 1.0) < 0.4
    image_forgery_detected = features.get('final_image_forgery_score', 0.0) > 0.7

    feature_names = feat_order
    importances = getattr(model, 'feature_importances_', None)
    key_drivers = []
    if importances is not None:
        drivers = explain_features(feature_names, importances, top_k=3)
        # Map driver names to human friendly messages
        for d in drivers:
            if 'icd' in d:
                key_drivers.append('Low ICD match score')
            elif 'final_image' in d or 'cnn' in d or 'ela' in d or 'phash' in d:
                key_drivers.append('High image forgery confidence')
            elif 'claim' in d:
                key_drivers.append('Large claim amount')
            elif 'previous' in d:
                key_drivers.append('High previous claim count')
    else:
        # Fallback: rule-based drivers
        if icd_mismatch:
            key_drivers.append('Low ICD match score')
        if image_forgery_detected:
            key_drivers.append('High image forgery confidence')

    explanation = {
        'icd_mismatch': bool(icd_mismatch),
        'image_forgery_detected': bool(image_forgery_detected),
        'key_drivers': key_drivers
    }

    return {
        'fraud_risk_percentage': pct,
        'risk_level': level,
        'explanation': explanation
    }


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=str(MODEL_PATH))
    p.add_argument('--input-json', help='JSON file with features (see README)')
    p.add_argument('--interactive', action='store_true')
    args = p.parse_args()

    model = load_model(args.model)

    if args.interactive:
        print('Enter JSON with keys: icd_match_score, cnn_forgery_score, ela_score, phash_score, final_image_forgery_score, claim_amount_log, previous_claim_count')
        s = input('> ')
        data = json.loads(s)
        out = predict_risk(model, data)
        print(json.dumps(out, indent=2))
    elif args.input_json:
        with open(args.input_json) as f:
            data = json.load(f)
        out = predict_risk(model, data)
        print(json.dumps(out, indent=2))
    else:
        # Demo: small sample
        demo = {
            'icd_match_score': 0.2,
            'cnn_forgery_score': 0.85,
            'ela_score': 0.6,
            'phash_score': 0.7,
            'final_image_forgery_score': 0.78,
            'claim_amount_log': float(np.log1p(1200)),
            'previous_claim_count': 2
        }
        out = predict_risk(model, demo)
        print(json.dumps(out, indent=2))