"""Synthetic dataset generator for Fraud Risk Module

Produces a CSV with columns:
- icd_match_score (0-1)
- cnn_forgery_score (0-1)
- ela_score (0-1)
- phash_score (0-1)
- final_image_forgery_score (0-1)
- claim_amount (float)
- previous_claim_count (int)
- fraud (0 or 1)  # synthetic label derived by rule

Note: Labels are synthetic and intended for academic demonstration only.
"""
import csv
import random
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / 'data'
OUT.mkdir(exist_ok=True)

DEFAULT_N = 2000


def synth_score(low=0.0, high=1.0, skew=1.0):
    """Sample a score in [low, high] with optional skew (>1 concentrates toward low)."""
    r = random.random() ** skew
    return float(low + (high - low) * r)


def generate(n=DEFAULT_N, seed=42, out_csv=str(OUT / 'synthetic.csv')):
    random.seed(seed)
    np.random.seed(seed)

    rows = []
    for i in range(n):
        # Draw basic image forgery and icd match scores with some correlations
        icd_match_score = synth_score(skew=0.8)
        cnn_forgery_score = synth_score(skew=1.2)
        ela_score = synth_score(skew=1.0)
        phash_score = synth_score(skew=1.0)

        # fuse to final_image_forgery_score similar to project's fuse function (weighted)
        final_image_forgery_score = 0.55 * cnn_forgery_score + 0.25 * ela_score + 0.20 * phash_score

        # Simulate claim metadata
        claim_amount = round(max(100.0, np.random.exponential(scale=500.0)), 2)
        previous_claim_count = int(np.random.poisson(lam=1.2))

        # Synthetic labeling rule (academic demonstration only)
        fraud = 1 if (icd_match_score < 0.4 and final_image_forgery_score > 0.7) else 0

        rows.append({
            'icd_match_score': round(icd_match_score, 4),
            'cnn_forgery_score': round(cnn_forgery_score, 4),
            'ela_score': round(ela_score, 4),
            'phash_score': round(phash_score, 4),
            'final_image_forgery_score': round(final_image_forgery_score, 4),
            'claim_amount': claim_amount,
            'previous_claim_count': previous_claim_count,
            'fraud': fraud,
        })

    # Save CSV
    keys = list(rows[0].keys())
    with open(out_csv, 'w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    print(f'Wrote {len(rows)} synthetic samples to {out_csv}')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=DEFAULT_N)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', default=str(OUT / 'synthetic.csv'))
    args = p.parse_args()
    generate(n=args.n, seed=args.seed, out_csv=args.out)