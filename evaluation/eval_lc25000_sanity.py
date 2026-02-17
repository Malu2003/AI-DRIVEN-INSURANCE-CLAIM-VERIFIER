"""Sanity evaluation on LC25000 to check false-positive rates on medical images."""
import os
import argparse
import csv
import json
from tqdm import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt

# project imports
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from inference import image_forgery_score as infer
from utils import phash as phash_utils
from utils import ela as ela_utils


def build_phash_db_if_missing(data_dir, out_csv):
    if os.path.exists(out_csv):
        return out_csv
    # build from train directory
    train_dir = os.path.join(data_dir, 'train') if os.path.exists(os.path.join(data_dir, 'train')) else data_dir
    print('Building phash DB from', train_dir)
    phash_utils.process_directory_to_csv(train_dir, out_csv)
    return out_csv


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    # collect image list
    candidates = []
    for root, _, files in os.walk(args.data_dir):
        for fn in files:
            if fn.lower().endswith(('.jpg', '.jpeg', '.png')):
                candidates.append(os.path.join(root, fn))
    print('Total images found:', len(candidates))
    random.seed(args.seed)
    sample = random.sample(candidates, min(args.sample_size, len(candidates)))

    # phash DB
    if args.phash_db:
        phash_db = args.phash_db
        if args.build_phash_db:
            phash_db = build_phash_db_if_missing(args.data_dir, phash_db)
        phash_db_entries = phash_utils.load_phash_db(phash_db) if os.path.exists(phash_db) else None
    else:
        phash_db_entries = None

    cnn_scores = []
    fused_scores = []
    rows = []
    for p in tqdm(sample, desc='LC25000 sample'):
        cnn = infer.compute_cnn_score(p, model_ckpt=args.model, tampered_index=1)
        diff = ela_utils.compute_ela(p, quality=args.ela_quality, scale=args.ela_scale)
        ela_score = ela_utils.compute_ela_score(diff)
        ph = phash_utils.compute_phash(p)
        if phash_db_entries:
            ph_score, best_fn, best_h = phash_utils.compute_phash_score(ph, phash_db_entries)
        else:
            ph_score, best_fn, best_h = None, None, None

        fused = infer.fuse_scores(cnn, ela_score, ph_score, weights=(0.5,0.3,0.2))
        rows.append({'path': p, 'cnn': cnn, 'ela': ela_score, 'phash': ph_score, 'fused': fused})
        cnn_scores.append(cnn)
        fused_scores.append(fused)

    # save CSV
    csv_path = os.path.join(args.out_dir, 'lc25000_sample_results.csv')
    with open(csv_path, 'w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=['path','cnn','ela','phash','fused'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # distributions
    cnn_scores = np.array(cnn_scores)
    fused_scores = np.array(fused_scores)

    plt.figure()
    plt.hist(cnn_scores, bins=50)
    plt.title('LC25000 CNN score distribution')
    plt.savefig(os.path.join(args.out_dir, 'lc25000_cnn_hist.png'))
    plt.close()

    plt.figure()
    plt.hist(fused_scores, bins=50)
    plt.title('LC25000 fused score distribution')
    plt.savefig(os.path.join(args.out_dir, 'lc25000_fused_hist.png'))
    plt.close()

    # false positives > threshold
    thr = 0.5
    fp_pct = float((fused_scores >= thr).sum()) / len(fused_scores) * 100.0
    stats = {'cnn_mean': float(cnn_scores.mean()), 'cnn_std': float(cnn_scores.std()), 'fused_mean': float(fused_scores.mean()), 'fused_std': float(fused_scores.std()), 'fp_pct': fp_pct, 'n': len(fused_scores)}
    with open(os.path.join(args.out_dir, 'lc25000_stats.json'), 'w', encoding='utf8') as f:
        json.dump(stats, f, indent=2)

    print('LC25000 stats:', stats)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/LC25000')
    parser.add_argument('--model', default='checkpoints/casia/best.pth.tar')
    parser.add_argument('--phash_db', default='data/phash_lc25000.csv')
    parser.add_argument('--build_phash_db', action='store_true')
    parser.add_argument('--out_dir', default='eval/lc25000')
    parser.add_argument('--sample_size', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ela_quality', type=int, default=90)
    parser.add_argument('--ela_scale', type=int, default=10)
    args = parser.parse_args()
    main(args)
