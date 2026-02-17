import gzip, csv, os, sys
from collections import Counter
HERE = r"d:\tech_squad\AI-Driven-Image-Forgery-Detection"
sys.path.insert(0, HERE)
from icd_validation.utils import mask_icd_mentions, extract_declared_icds

files = [
    (r"d:\tech_squad\AI-Driven-Image-Forgery-Detection\data\mimic-iv-notes-2.2\note\discharge.csv.gz","discharge"),
    (r"d:\tech_squad\AI-Driven-Image-Forgery-Detection\data\mimic-iv-notes-2.2\note\radiology.csv.gz","radiology"),
]
for path, label in files:
    if not os.path.exists(path):
        print("Missing:", path)
        continue
    print("==", label, path)
    counts = Counter()
    total = 0
    samples = []
    with gzip.open(path, 'rt', encoding='utf-8', errors='replace') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            total += 1
            nt = row.get('note_type') or row.get('note_type', '')
            counts[nt] += 1
            if len(samples) < 3:
                text = row.get('text','')
                samples.append(text)
    print('Total rows:', total)
    print('Note type counts:', dict(counts))
    for i, s in enumerate(samples):
        print('\n-- Sample', i+1)
        print('Original (first 300 chars):', s[:300].replace('\n','\\n'))
        print('Declared ICDs:', extract_declared_icds(s))
        print('Masked (first 300 chars):', mask_icd_mentions(s)[:300].replace('\n','\\n'))
