"""Prepare a simple MIMIC-derived training CSV mapping diagnosis descriptions to ICD codes.

This script uses the demo MIMIC files under `data/mimic-iv-clinical-database-demo-2.2/hosp`.
It produces a CSV with columns: `text`, `icd_codes` (semicolon separated).

It is a lightweight way to create a training set for ClinicalBERT when full notes are not available.
"""
import csv
import gzip
import argparse
from pathlib import Path


def load_icd_descriptions(d_icd_path):
    d = {}
    open_func = gzip.open if str(d_icd_path).endswith('.gz') else open
    with open_func(d_icd_path, 'rt', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        for r in reader:
            code = (r.get('icd_code') or r.get('ICD_CODE') or r.get('code') or '').strip().upper()
            desc = (r.get('long_title') or r.get('short_title') or r.get('label') or '').strip()
            if code:
                d[code] = desc
    return d


def build_examples(diagnoses_icd_path, icd_descs, out_path, limit=None):
    open_func = gzip.open if str(diagnoses_icd_path).endswith('.gz') else open
    seen = set()
    with open_func(diagnoses_icd_path, 'rt', encoding='utf-8', errors='ignore') as rf, open(out_path, 'w', newline='', encoding='utf-8') as wf:
        reader = csv.DictReader(rf)
        writer = csv.writer(wf)
        writer.writerow(['text', 'icd_codes'])
        for r in reader:
            code = (r.get('icd_code') or '').strip().upper()
            hadm_id = r.get('hadm_id')
            if not code:
                continue
            desc = icd_descs.get(code, '')
            text = desc or f'Diagnosis code {code}'
            # simple augmentation: template
            text = f"Diagnosis: {text}." if text else f"Diagnosis code {code}."
            key = (text, code)
            if key in seen:
                continue
            seen.add(key)
            writer.writerow([text, code])
            if limit and len(seen) >= limit:
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mimic_dir', default='data/mimic-iv-clinical-database-demo-2.2/hosp')
    parser.add_argument('--out_csv', default='data/mimic_training.csv')
    parser.add_argument('--limit', type=int, default=2000)
    args = parser.parse_args()

    mimic_dir = Path(args.mimic_dir)
    d_icd_path = mimic_dir / 'd_icd_diagnoses.csv.gz'
    diagnoses_icd_path = mimic_dir / 'diagnoses_icd.csv.gz'

    icd_descs = load_icd_descriptions(d_icd_path)
    build_examples(diagnoses_icd_path, icd_descs, args.out_csv, limit=args.limit)
    print('wrote', args.out_csv)


if __name__ == '__main__':
    main()
