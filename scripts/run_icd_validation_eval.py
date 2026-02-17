"""Run ICD validation evaluation headlessly and save results to CSV.

This script mirrors notebooks/icd_validation_evaluation.ipynb but is safe to run in the current Python interpreter.
"""
import logging
from pathlib import Path
import csv

logging.basicConfig(level=logging.INFO)

MODEL_DIR = Path('checkpoints/tfidf_demo')
SAMPLE_CSV = Path('data/mimic_training_sample.csv')
OUT_CSV = Path('results/icd_validation_eval_results.csv')
OUT_CSV.parent.mkdir(exist_ok=True, parents=True)

# imports
try:
    import pandas as pd
except Exception as e:
    pd = None
    logging.warning('pandas not available; falling back to csv reader: %s', e)

from icd_validation.tfidf import TFIDFClassifier
from icd_validation.infer import predict_and_score
from icd_validation.scorer import compute_confidence


def load_samples():
    if SAMPLE_CSV.exists():
        if pd is not None:
            df = pd.read_csv(SAMPLE_CSV)
            if 'text' not in df.columns:
                raise ValueError('Sample CSV must contain a `text` column')
            df = df.reset_index(drop=True).reset_index().rename(columns={'index': 'doc_id'})
            return df.to_dict(orient='records')
        else:
            with open(SAMPLE_CSV, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if not rows or 'text' not in rows[0]:
                logging.warning('Sample CSV missing or lacks `text`; using built-in sample')
                return [
                    {'doc_id': 0, 'text': 'Diagnosis: Acute myocardial infarction. ICD-10: I21.9'},
                    {'doc_id': 1, 'text': 'Diagnosis: Diabetes mellitus without mention of complication. ICD10: E11.9'},
                    {'doc_id': 2, 'text': 'Patient with abdominal pain; declared code: K52.9 - noninfective gastroenteritis and colitis, unspecified'},
                ]
            return [{'doc_id': i, 'text': r['text']} for i, r in enumerate(rows)]
    else:
        logging.warning('Sample CSV not found at %s; using built-in sample', SAMPLE_CSV)
        return [
            {'doc_id': 0, 'text': 'Diagnosis: Acute myocardial infarction. ICD-10: I21.9'},
            {'doc_id': 1, 'text': 'Diagnosis: Diabetes mellitus without mention of complication. ICD10: E11.9'},
            {'doc_id': 2, 'text': 'Patient with abdominal pain; declared code: K52.9 - noninfective gastroenteritis and colitis, unspecified'},
        ]


def main(limit=None):
    clf = TFIDFClassifier()
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f'Model directory not found: {MODEL_DIR}')
    clf.load(str(MODEL_DIR))
    logging.info('Loaded TFIDF model; labels=%d', len(clf.labels))

    samples = load_samples()
    if limit:
        samples = samples[:limit]

    rows = []
    for s in samples:
        doc_id = int(s.get('doc_id', -1))
        text = str(s.get('text', ''))
        report = predict_and_score(text, clf, compute_score=compute_confidence)
        explains = report.get('explain', [])
        if not explains:
            top = report.get('predicted', [])
            if top:
                pred, prob = top[0]
            else:
                pred, prob = (None, 0.0)
            rows.append({'doc_id': doc_id, 'declared_icd': None, 'predicted_icd': pred, 'match_type': 'no_declared', 'score': None, 'pred_prob': prob, 'text': text})
        else:
            for rec in explains:
                rows.append({'doc_id': doc_id, 'declared_icd': rec.get('declared'), 'predicted_icd': rec.get('predicted_top'), 'match_type': rec.get('match_type'), 'score': rec.get('score'), 'pred_prob': rec.get('predicted_prob'), 'text': text})

    # Save CSV
    if pd is not None:
        df = pd.DataFrame(rows)
        df.to_csv(OUT_CSV, index=False)
    else:
        # write CSV via csv module
        if rows:
            keys = rows[0].keys()
            with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, keys)
                writer.writeheader()
                writer.writerows(rows)
    logging.info('Saved results to %s (rows=%d)', OUT_CSV, len(rows))

    # Print simple summary
    from collections import Counter
    counts = Counter([r['match_type'] for r in rows])
    logging.info('Match type counts: %s', dict(counts))

    low = [r for r in rows if r.get('score') is not None and r.get('score') <= 0.3]
    logging.info('Low-score cases (<=0.3): %d', len(low))
    if low:
        logging.info('Sample low-score case: %s', low[0])

    return rows


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--limit', type=int, default=20, help='Limit number of documents to evaluate')
    args = p.parse_args()
    main(limit=args.limit)
