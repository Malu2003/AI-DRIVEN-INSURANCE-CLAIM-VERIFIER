#!/usr/bin/env python3
"""Prepare MIMIC notes + diagnoses for ICD fine-tuning.

Outputs (under --out dir):
 - all.jsonl (intermediate) containing masked notes with labels (top-k)
 - train.jsonl, val.jsonl
 - label_map.json
 - preprocess.log

This is intended for the demo MIMIC files shipped in the repo; it streams files
and is memory-friendly.

Usage example:
  python scripts/prepare_mimic_for_icd.py \
    --notes data/mimic-iv-notes-2.2/note/discharge.csv.gz \
    --diagnoses data/mimic-iv-clinical-database-demo-2.2/hosp/diagnoses_icd.csv.gz \
    --admissions data/mimic-iv-clinical-database-demo-2.2/hosp/admissions.csv.gz \
    --out processed/mimic_notes --top_k 200 --min_count 25 --note_types DS --val_frac 0.1 --seed 42
"""
import argparse
import gzip
import json
import os
import random
from collections import Counter, defaultdict

try:
    import pandas as pd
except Exception as e:
    raise SystemExit("pandas is required to run this script. Install it and retry.")

import sys, pathlib
# ensure repo root is on sys.path so we can import local packages when running the script
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from icd_validation.utils import mask_icd_mentions


def iter_csv_gz(path, chunksize=100000):
    # yields DataFrame chunks
    return pd.read_csv(path, compression='gzip', dtype=str, chunksize=chunksize)


def build_hadm_to_codes(diagnoses_path, min_count=1):
    print(f"Building hadm->ICD map from {diagnoses_path} (min_count={min_count})")
    code_counts = Counter()
    hadm2codes = defaultdict(set)
    for chunk in iter_csv_gz(diagnoses_path, chunksize=50000):
        # expected columns: 'hadm_id', 'icd_code' or 'icd9_code' depending on file
        if 'icd_code' in chunk.columns:
            code_col = 'icd_code'
        elif 'icd9_code' in chunk.columns:
            code_col = 'icd9_code'
        else:
            raise SystemExit(f"No icd_code-like column in {diagnoses_path}")
        for _, row in chunk.iterrows():
            hadm = str(row.get('hadm_id') or '').strip()
            code = row.get(code_col) or ''
            code = str(code).upper().replace(' ', '')
            if not hadm or not code:
                continue
            hadm2codes[hadm].add(code)
            code_counts[code] += 1
    # filter by min_count
    codes_kept = [c for c,ct in code_counts.items() if ct >= min_count]
    print(f"Found {len(code_counts)} distinct ICD codes, {len(codes_kept)} meet min_count")
    return hadm2codes, code_counts


def build_hadm_to_subject(admissions_path):
    print(f"Building hadm->subject map from {admissions_path}")
    hadm2subj = {}
    for chunk in iter_csv_gz(admissions_path, chunksize=50000):
        for _, row in chunk.iterrows():
            hadm = str(row.get('hadm_id') or '').strip()
            subj = str(row.get('subject_id') or '').strip()
            if hadm:
                hadm2subj[hadm] = subj
    print(f"Mapped {len(hadm2subj)} admissions to subjects")
    return hadm2subj


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--notes', required=True)
    p.add_argument('--diagnoses', required=True)
    p.add_argument('--admissions', required=True)
    p.add_argument('--out', default='processed/mimic_notes')
    p.add_argument('--top_k', type=int, default=200)
    p.add_argument('--min_count', type=int, default=25)
    p.add_argument('--note_types', nargs='+', default=['DS'])
    p.add_argument('--val_frac', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, 'preprocess.log')
    log = open(log_path, 'w', encoding='utf-8')
    import builtins as _builtins
    def _log_print(*args2, **kwargs):
        log.write(' '.join(str(a) for a in args2) + '\n')
        _builtins.print(*args2, **kwargs)
    print = _log_print

    hadm2codes, code_counts = build_hadm_to_codes(args.diagnoses, min_count=1)
    hadm2subj = build_hadm_to_subject(args.admissions)

    # choose top-k by global frequency but respecting min_count
    code_items = sorted(code_counts.items(), key=lambda x: -x[1])
    top_codes = [c for c, ct in code_items if ct >= args.min_count][:args.top_k]
    top_set = set(top_codes)
    print(f"Selected top {len(top_codes)} ICD codes (min_count={args.min_count}, top_k={args.top_k})")

    # build hadm->toplabels mapping
    hadm2top = {}
    for hadm, codes in hadm2codes.items():
        sel = sorted(c for c in codes if c in top_set)
        if sel:
            hadm2top[hadm] = sel
    print(f"Admissions with at least one top-code: {len(hadm2top)}")

    all_jsonl = os.path.join(args.out, 'all.jsonl')
    written = 0
    missing_hadm = 0
    note_types = set(args.note_types)

    # stream notes and emit JSONL rows when hadm maps to labels
    for chunk in pd.read_csv(args.notes, compression='gzip', dtype=str, chunksize=50000):
        if 'note_type' in chunk.columns:
            chunk = chunk[chunk['note_type'].isin(note_types)]
        else:
            # if no note_type column, keep all
            pass
        chunk['hadm_id'] = chunk.get('hadm_id', '').fillna('').astype(str)
        chunk['subject_id'] = chunk.get('subject_id', '').fillna('').astype(str)
        for _, row in chunk.iterrows():
            hadm = str(row.get('hadm_id') or '').strip()
            subj = str(row.get('subject_id') or '').strip()
            note_id = row.get('note_id') or ''
            text = row.get('text') or ''
            if not hadm:
                missing_hadm += 1
                continue
            labels = hadm2top.get(hadm, [])
            if not labels:
                continue
            text_masked = mask_icd_mentions(text)
            obj = {
                'note_id': note_id,
                'subject_id': subj,
                'hadm_id': hadm,
                'note_type': row.get('note_type'),
                'text': text_masked,
                'labels': labels
            }
            with open(all_jsonl, 'a', encoding='utf-8') as outf:
                outf.write(json.dumps(obj, ensure_ascii=False) + '\n')
            written += 1
    print(f"Wrote {written} labeled notes to {all_jsonl} (skipped {missing_hadm} notes with no hadm_id)")

    # build label map
    from collections import Counter
    label_counter = Counter()
    with open(all_jsonl, 'r', encoding='utf-8') as inf:
        for line in inf:
            obj = json.loads(line)
            for c in obj['labels']:
                label_counter[c] += 1
    # re-compute top_k sorted by count
    top_by_count = [c for c, _ in label_counter.most_common(args.top_k)]
    label_to_idx = {c: i for i, c in enumerate(top_by_count)}
    with open(os.path.join(args.out, 'label_map.json'), 'w', encoding='utf-8') as f:
        json.dump({'label_to_idx': label_to_idx, 'idx_to_label': {i:c for c,i in label_to_idx.items()}}, f, indent=2)
    print(f"Saved label_map with {len(label_to_idx)} labels")

    # read all subjects and split by subject to avoid leakage
    subjects = set()
    with open(all_jsonl, 'r', encoding='utf-8') as inf:
        for line in inf:
            obj = json.loads(line)
            subjects.add(obj['subject_id'])
    subjects = sorted(s for s in subjects if s)
    random.seed(args.seed)
    random.shuffle(subjects)
    n_val = max(1, int(len(subjects) * args.val_frac))
    val_subj = set(subjects[:n_val])
    print(f"Total subjects: {len(subjects)} | val subjects: {len(val_subj)}")

    train_out = os.path.join(args.out, 'train.jsonl')
    val_out = os.path.join(args.out, 'val.jsonl')
    counts = {'train':0, 'val':0}
    samples_masked = []
    with open(all_jsonl, 'r', encoding='utf-8') as inf, \
         open(train_out, 'w', encoding='utf-8') as ftrain, \
         open(val_out, 'w', encoding='utf-8') as fval:
        for line in inf:
            obj = json.loads(line)
            subj = obj['subject_id']
            # convert labels to indices and filter out those not in label_map
            label_indices = [label_to_idx[c] for c in obj['labels'] if c in label_to_idx]
            if not label_indices:
                continue
            obj['label_indices'] = label_indices
            # sample for logging
            if len(samples_masked) < 5:
                samples_masked.append({'note_id': obj['note_id'], 'text_snippet': obj['text'][:300], 'labels': obj['labels']})
            if subj in val_subj:
                fval.write(json.dumps(obj, ensure_ascii=False) + '\n')
                counts['val'] += 1
            else:
                ftrain.write(json.dumps(obj, ensure_ascii=False) + '\n')
                counts['train'] += 1
    print(f"Saved train={counts['train']} val={counts['val']}")
    print('\nSample masked texts:')
    for s in samples_masked:
        print('NOTE_ID:', s['note_id'])
        print('TEXT_SNIPPET:', s['text_snippet'].replace('\n','\\n'))
        print('LABELS:', s['labels'])
        print('---')

    print('\nPreprocessing complete. Logs written to', log_path)
    log.close()

if __name__ == '__main__':
    main()
