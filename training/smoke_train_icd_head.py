"""Smoke train for ICD head-only fine-tune using processed JSONL files.

Reads `processed/mimic_notes/train.jsonl` and `processed/mimic_notes/val.jsonl` and runs
1-3 epoch head-only training and reports losses and sample predictions.
"""
import argparse
import json
from pathlib import Path
import random
from sklearn.metrics import f1_score
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from icd_validation.tfidf import TFIDFClassifier
from icd_validation import utils


def read_jsonl(path):
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            out.append(json.loads(line))
    return out


def to_samples(jsonl):
    texts = [o['text'] for o in jsonl]
    labels = [o['labels'] for o in jsonl]
    return texts, labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train', default='processed/mimic_notes/train.jsonl')
    p.add_argument('--val', default='processed/mimic_notes/val.jsonl')
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--model-name', default='emilyalsentzer/Bio_ClinicalBERT')
    p.add_argument('--save-out', default=None, help='Directory to save trained head (optional)')
    args = p.parse_args()

    train_json = read_jsonl(args.train)
    val_json = read_jsonl(args.val)
    train_texts, train_labels = to_samples(train_json)
    val_texts, val_labels = to_samples(val_json)

    print(f"Train samples: {len(train_texts)} | Val samples: {len(val_texts)}")

    # Instantiate classifier
    clf = TFIDFClassifier(model_name=args.model_name)

    # Fit head
    print('Starting head-only training...')
    clf.fit(train_texts, train_labels, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

    # Evaluate
    print('Predicting on validation set...')
    probs = clf.predict_proba(val_texts)
    classes = clf.labels

    Ytrue = np.array([[1 if c in labs else 0 for c in classes] for labs in val_labels], dtype=int)
    Ypred = np.array([[1 if probs[i].get(c, 0.0) >= 0.5 else 0 for c in classes] for i in range(len(probs))], dtype=int)

    micro = f1_score(Ytrue, Ypred, average='micro', zero_division=0)
    macro = f1_score(Ytrue, Ypred, average='macro', zero_division=0)

    print('Validation Micro F1:', micro)
    print('Validation Macro F1:', macro)

    # Optionally save the trained head
    if args.save_out:
        print(f"Saving trained head to {args.save_out}")
        clf.save(args.save_out)

    # Print a few sample predictions
    print('\nSample predictions:')
    for i in range(min(5, len(val_texts))):
        text = val_texts[i][:300].replace('\n','\\n')
        pred = sorted(probs[i].items(), key=lambda x: -x[1])[:5]
        print(f"NOTE {i+1}: text_snippet={text}")
        print('Top preds:', pred)
        print('True labels:', val_labels[i])
        print('---')

if __name__ == '__main__':
    main()
