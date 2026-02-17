"""Train an XGBoost classifier on synthetic fraud data.

Saves model to fraud_risk_module/models/fraud_model.pkl and prints basic metrics.
"""
import joblib
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb

from feature_builder import load_dataset, build_X_y

MODEL_DIR = Path(__file__).parent / 'models'
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / 'fraud_model.pkl'


def train_and_save(test_size=0.2, seed=42, num_round=100):
    df = load_dataset()
    X, y = build_X_y(df)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

    clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    y_prob = clf.predict_proba(X_val)[:, 1]

    print('Classification report:')
    print(classification_report(y_val, y_pred, digits=4))

    # Compute metrics (handle potential single-class case for AUC)
    metrics = {}
    try:
        auc = roc_auc_score(y_val, y_prob)
        metrics['roc_auc'] = float(auc)
        print('Validation AUC:', auc)
    except Exception:
        metrics['roc_auc'] = None
        print('AUC could not be computed (single-class in split)')

    # Add precision/recall/f1 and counts
    from sklearn.metrics import precision_score, recall_score, f1_score
    metrics['precision'] = float(precision_score(y_val, y_pred, zero_division=0))
    metrics['recall'] = float(recall_score(y_val, y_pred, zero_division=0))
    metrics['f1'] = float(f1_score(y_val, y_pred, zero_division=0))
    metrics['num_train'] = int(len(X_train))
    metrics['num_val'] = int(len(X_val))

    # Save model and metrics
    joblib.dump(clf, MODEL_PATH)
    with open(MODEL_DIR / 'metrics.json', 'w') as f:
        import json
        json.dump(metrics, f, indent=2)

    print('Saved model to', MODEL_PATH)
    print('Saved metrics to', MODEL_DIR / 'metrics.json')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--test-size', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    train_and_save(test_size=args.test_size, seed=args.seed)