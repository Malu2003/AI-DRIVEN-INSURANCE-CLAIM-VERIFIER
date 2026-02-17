"""Train a TF-IDF + Logistic Regression ICD predictor using a simple CSV."""
import argparse
import csv
import logging
from pathlib import Path

from icd_validation.tfidf import TFIDFClassifier


def load_csv(path, text_col='text', icd_col='icd_codes'):
    texts = []
    labels = []
    with open(path, encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        for r in reader:
            text = (r.get(text_col) or '').strip()
            codes = (r.get(icd_col) or '').strip()
            codes_list = [c.strip().upper() for c in codes.split(';') if c.strip()]
            if text:
                texts.append(text)
                labels.append(codes_list)
    return texts, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--out', default='checkpoints/tfidf_demo')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    texts, labels = load_csv(args.csv)
    if args.limit:
        texts = texts[:args.limit]
        labels = labels[:args.limit]

    logging.info('Loaded %d samples', len(texts))
    model = TFIDFClassifier()
    model.fit(texts, labels)
    model.save(args.out)
    logging.info('Saved TFIDF model to %s', args.out)


if __name__ == '__main__':
    main()
