"""Small utility to compare TF-IDF baseline vs ClinicalBERT embedding classifier.

This script is lightweight: if no CSV is provided, runs on a small synthetic dataset.
It prints micro and macro F1 scores for both models.
"""
from typing import List
import argparse
import csv
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from icd_validation.tfidf import TFIDFClassifier
except Exception as e:
    print('Error importing TFIDFClassifier:', e)
    raise


def load_csv(path: str):
    import pandas as pd
    df = pd.read_csv(path, dtype=str)
    if 'text' not in df.columns or 'labels' not in df.columns:
        raise ValueError('CSV must contain `text` and `labels` columns; labels are semicolon-separated codes')
    texts = df['text'].astype(str).tolist()
    labels = [ [s.strip() for s in lab.split(';') if s.strip()] for lab in df['labels'].astype(str).tolist() ]
    return texts, labels


def synth_dataset():
    texts = ["Patient with colon adenocarcinoma of sigmoid colon",
             "Diabetes mellitus type 2 without complications",
             "Asthma exacerbation with bronchitis",
             "Hypertension and chronic kidney disease",
             "Acute myocardial infarction"]
    labels = [['C18.7'], ['E11.9'], ['J45'], ['I10','N18.9'], ['I21.9']]
    return texts, labels


def train_tfidf_baseline(train_texts: List[str], train_labels: List[List[str]]):
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(train_labels)
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X = vec.fit_transform(train_texts)
    clf = OneVsRestClassifier(LogisticRegression(max_iter=1000))
    clf.fit(X, Y)
    return vec, clf, mlb


def eval_tfidf(vec, clf, mlb, texts, labels):
    X = vec.transform(texts)
    Ytrue = mlb.transform(labels)
    probs = clf.predict_proba(X)
    Ypred = (probs >= 0.5).astype(int)
    micro = f1_score(Ytrue, Ypred, average='micro', zero_division=0)
    macro = f1_score(Ytrue, Ypred, average='macro', zero_division=0)
    return micro, macro


def train_bert_model(train_texts, train_labels, epochs=2):
    model = TFIDFClassifier()
    model.fit(train_texts, train_labels, epochs=epochs, batch_size=4, lr=1e-3)
    return model


def eval_bert(model, texts, labels):
    # get binary predictions using threshold 0.5
    probs_list = model.predict_proba(texts)
    classes = model.labels
    mlb = MultiLabelBinarizer(classes=classes)
    Ytrue = mlb.fit_transform(labels)
    Ypred = []
    for p in probs_list:
        row = [1 if p.get(c, 0.0) >= 0.5 else 0 for c in classes]
        Ypred.append(row)
    Ypred = np.array(Ypred)
    micro = f1_score(Ytrue, Ypred, average='micro', zero_division=0)
    macro = f1_score(Ytrue, Ypred, average='macro', zero_division=0)
    return micro, macro


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--train-csv', default=None)
    p.add_argument('--test-csv', default=None)
    p.add_argument('--bert-epochs', type=int, default=2, help='Number of epochs to train the ClinicalBERT head')
    args = p.parse_args()

    if args.train_csv:
        train_texts, train_labels = load_csv(args.train_csv)
    else:
        train_texts, train_labels = synth_dataset()

    if args.test_csv:
        test_texts, test_labels = load_csv(args.test_csv)
    else:
        test_texts, test_labels = train_texts, train_labels

    print('Training TF-IDF baseline...')
    vec, clf, mlb = train_tfidf_baseline(train_texts, train_labels)
    print('Evaluating TF-IDF baseline...')
    tf_micro, tf_macro = eval_tfidf(vec, clf, mlb, test_texts, test_labels)

    print('\nTraining ClinicalBERT-based classifier...')
    bert_model = train_bert_model(train_texts, train_labels, epochs=args.bert_epochs)
    print('Evaluating ClinicalBERT model...')
    b_micro, b_macro = eval_bert(bert_model, test_texts, test_labels)

    print('\nComparison (Micro F1, Macro F1):')
    print('Model\tMicro F1\tMacro F1')
    print(f'TF-IDF\t{tf_micro:.4f}\t{tf_macro:.4f}')
    print(f'ClinicalBERT\t{b_micro:.4f}\t{b_macro:.4f}')
