"""Build features for fraud risk model from raw inputs or synthetic CSV."""
import pandas as pd
from pathlib import Path

DATA = Path(__file__).parent / 'data' / 'synthetic.csv'


def load_dataset(path: str = None):
    p = Path(path) if path else DATA
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")
    df = pd.read_csv(p)
    return df


def build_features(df):
    # Select and normalize features where appropriate
    feats = df[['icd_match_score', 'cnn_forgery_score', 'ela_score', 'phash_score', 'final_image_forgery_score', 'claim_amount', 'previous_claim_count']].copy()

    # Normalize claim_amount by log scaling for stability
    import numpy as _np
    feats['claim_amount_log'] = feats['claim_amount'].apply(lambda x: float(_np.log1p(x)))
    feats = feats.drop(columns=['claim_amount'])

    # Optionally: scale other features to [0,1] if not already
    # For this demo, these are already in [0,1]

    return feats


def build_X_y(df):
    X = build_features(df)
    y = df['fraud']
    return X, y


if __name__ == '__main__':
    df = load_dataset()
    X, y = build_X_y(df)
    print('Dataset shape:', df.shape)
    print('Feature sample:')
    print(X.head())