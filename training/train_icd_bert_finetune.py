"""Level-1 fine-tuning script for ICD prediction on MIMIC-like CSVs.

Level-1 fine-tuning on MIMIC-IV notes with ClinicalBERT frozen; only the
classification head is trained. This follows the project requirements:
- Encoder: emilyalsentzer/Bio_ClinicalBERT
- Freeze encoder
- Pooling: mean pooling over last hidden states
- Loss: BCEWithLogitsLoss
- Optimizer: AdamW (head only)
- Learning rate: 1e-3
- Epochs: 5-10 (configurable)

This script saves the trained head (via TFIDFClassifier.save) and writes a
small metrics JSON and a simple training log to the output dir.
"""
import argparse
import os
import json
import random
from pathlib import Path
from typing import List

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from icd_validation.dataset import load_text_icd_csv, build_label_list
from icd_validation.tfidf import TFIDFClassifier
from icd_validation import utils


def prepare_samples(path: str, sep: str = ';', top_k: int = 500, val_fraction: float = 0.1, random_seed: int = 42):
    samples = load_text_icd_csv(path, icd_col='icd_codes', sep=sep)
    # Build label list from training corpus
    label_list = build_label_list(samples, top_k=top_k)

    # Filter out samples with no labels in label_list
    filtered = [(t, [c for c in codes if c in label_list]) for t, codes in samples]
    filtered = [(t, codes) for t, codes in filtered if codes]

    texts = [t for t, _ in filtered]
    labels = [l for _, l in filtered]

    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=val_fraction, random_state=random_seed)
    return (train_texts, train_labels), (val_texts, val_labels), label_list


def mask_texts(texts: List[str]) -> List[str]:
    return [utils.mask_icd_mentions(t) for t in texts]


def train(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    (train_texts, train_labels), (val_texts, val_labels), label_list = prepare_samples(args.data, sep=args.sep, top_k=args.top_k, val_fraction=args.val_frac, random_seed=args.seed)
    print(f"Samples: train={len(train_texts)}, val={len(val_texts)}, labels={len(label_list)}")

    # Mask ICD mentions as required
    train_texts = mask_texts(train_texts)
    val_texts = mask_texts(val_texts)

    # Initialize ClinicalBERT-based classifier (encoder frozen by default)
    clf = TFIDFClassifier(model_name=args.model_name, embedding_pool='mean')
    # Ensure label binarizer is consistent with label_list
    # Convert list labels to list-of-lists for fit
    clf.mlb = None

    # Fit (this computes embeddings using frozen encoder and trains head only)
    print('Training head (ClinicalBERT frozen) ...')
    clf.fit(train_texts, train_labels, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

    # Evaluate on validation set
    print('Evaluating on validation set...')
    probs = clf.predict_proba(val_texts)
    # convert to binary predictions at 0.5
    classes = clf.labels
    Ytrue = np.array([[1 if c in labs else 0 for c in classes] for labs in val_labels], dtype=int)
    Ypred = np.array([[1 if probs[i].get(c, 0.0) >= 0.5 else 0 for c in classes] for i in range(len(probs))], dtype=int)

    micro = f1_score(Ytrue, Ypred, average='micro', zero_division=0)
    macro = f1_score(Ytrue, Ypred, average='macro', zero_division=0)

    metrics = {
        'val_micro_f1': float(micro),
        'val_macro_f1': float(macro),
        'num_train': len(train_texts),
        'num_val': len(val_texts),
        'labels': len(classes)
    }

    print('Validation Micro F1:', metrics['val_micro_f1'])
    print('Validation Macro F1:', metrics['val_macro_f1'])

    # Save model and metadata
    clf.save(str(outdir))
    with open(outdir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Also write a short line to the comparison file for later reporting
    compf = Path('reports/output/icd_feature_comparison.txt')
    compf.parent.mkdir(parents=True, exist_ok=True)
    with open(compf, 'a') as f:
        f.write(f"\nClinicalBERT_finetuned_on_{Path(args.data).stem}\t{metrics['val_micro_f1']:.4f}\t{metrics['val_macro_f1']:.4f}\n")

    print('Saved model to', outdir)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/mimic_training_sample.csv', help='Path to MIMIC-like CSV with text and icd_codes columns')
    p.add_argument('--sep', default=',', help='Separator used in icd_codes field (default is comma for sample)')
    p.add_argument('--outdir', default='checkpoints/icd_bert_finetuned', help='Output directory to save model and metrics')
    p.add_argument('--model-name', default='emilyalsentzer/Bio_ClinicalBERT')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--top-k', type=int, default=500, help='Top-K ICD labels to use')
    p.add_argument('--val-frac', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    train(args)
