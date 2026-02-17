"""Batch evaluation for CASIA forgery detection.

Outputs:
 - CSV with per-image scores
 - ROC and PR plots for CNN and fused score
 - AUC values printed and saved
"""
import os
import argparse
import csv
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

import torch
from torchvision import datasets

# ensure imports from project
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from inference import image_forgery_score as infer
from utils import phash as phash_utils
from utils import ela as ela_utils


def load_image_list(data_dir):
    # prefer 'val' subfolder
    val_dir = os.path.join(data_dir, 'val') if os.path.exists(os.path.join(data_dir, 'val')) else data_dir
    ds = datasets.ImageFolder(val_dir)
    samples = [(p, l) for p, l in ds.samples]
    classes = ds.classes
    return samples, classes


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    samples, classes = load_image_list(args.data_dir)
    print('Classes:', classes)

    # load phash DB
    phash_db = None
    if args.phash_db and os.path.exists(args.phash_db):
        phash_db = phash_utils.load_phash_db(args.phash_db)

    rows = []
    y_true = []
    cnn_scores = []
    fused_scores = []

    for path, label in tqdm(samples, desc='CASIA eval'):
        cnn = infer.compute_cnn_score(path, model_ckpt=args.model, tampered_index=1)
        # ELA
        diff = ela_utils.compute_ela(path, quality=args.ela_quality, scale=args.ela_scale)
        ela_score = ela_utils.compute_ela_score(diff)
        # pHash
        ph = phash_utils.compute_phash(path)
        if phash_db:
            ph_score, best_fn, best_h = phash_utils.compute_phash_score(ph, phash_db)
        else:
            ph_score, best_fn, best_h = None, None, None

        fused = infer.fuse_scores(cnn, ela_score, ph_score, weights=(0.5,0.3,0.2))

        rows.append({'path': path, 'label': int(label), 'cnn': cnn, 'ela': ela_score, 'phash': ph_score, 'fused': fused})
        y_true.append(int(label))
        cnn_scores.append(cnn)
        fused_scores.append(fused)

    # save CSV
    csv_path = os.path.join(args.out_dir, 'casia_results.csv')
    with open(csv_path, 'w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=['path','label','cnn','ela','phash','fused'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    y_true = np.array(y_true)
    cnn_scores = np.array(cnn_scores)
    fused_scores = np.array(fused_scores)

    # ROC and AUC
    fpr_c, tpr_c, _ = roc_curve(y_true, cnn_scores)
    auc_c = auc(fpr_c, tpr_c)

    fpr_f, tpr_f, _ = roc_curve(y_true, fused_scores)
    auc_f = auc(fpr_f, tpr_f)

    plt.figure()
    plt.plot(fpr_c, tpr_c, label=f'CNN AUC={auc_c:.4f}')
    plt.plot(fpr_f, tpr_f, label=f'Fused AUC={auc_f:.4f}')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.title('CASIA ROC')
    plt.savefig(os.path.join(args.out_dir, 'casia_roc.png'))
    plt.close()

    # PR curves
    prec_c, rec_c, _ = precision_recall_curve(y_true, cnn_scores)
    ap_c = average_precision_score(y_true, cnn_scores)
    prec_f, rec_f, _ = precision_recall_curve(y_true, fused_scores)
    ap_f = average_precision_score(y_true, fused_scores)

    plt.figure()
    plt.plot(rec_c, prec_c, label=f'CNN AP={ap_c:.4f}')
    plt.plot(rec_f, prec_f, label=f'Fused AP={ap_f:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.title('CASIA Precision-Recall')
    plt.savefig(os.path.join(args.out_dir, 'casia_pr.png'))
    plt.close()

    # Save metrics
    metrics = {'cnn_auc': float(auc_c), 'fused_auc': float(auc_f), 'cnn_ap': float(ap_c), 'fused_ap': float(ap_f)}
    with open(os.path.join(args.out_dir, 'casia_metrics.json'), 'w', encoding='utf8') as f:
        json.dump(metrics, f, indent=2)

    print('CASIA metrics:', metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/CASIA2')
    parser.add_argument('--model', default='checkpoints/casia/best.pth.tar')
    parser.add_argument('--phash_db', default='data/phash_casia_authentic.csv')
    parser.add_argument('--out_dir', default='eval/casia')
    parser.add_argument('--ela_quality', type=int, default=90)
    parser.add_argument('--ela_scale', type=int, default=10)
    args = parser.parse_args()
    main(args)
