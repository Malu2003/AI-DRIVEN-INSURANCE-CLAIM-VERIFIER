import os, gzip, csv, sys
from collections import defaultdict

ROOT = 'data/mimic-iv-clinical-database-demo-2.2'
KEYWORDS = ['text','note','note_text','description','short_description','long_description','chart','remark','comments','value','label','note','diagnosis','diagnoses','reason']

candidates = defaultdict(list)

for dirpath, dirs, files in os.walk(ROOT):
    for fn in files:
        if fn.endswith('.gz') and fn.endswith('.csv.gz'):
            p = os.path.join(dirpath, fn)
            try:
                with gzip.open(p, 'rt', encoding='utf-8', errors='ignore') as f:
                    r = csv.DictReader(f)
                    header = r.fieldnames or []
                    header_lower = [h.lower() if h else '' for h in header]
                    found = [h for h in header if any(k in (h or '').lower() for k in KEYWORDS)]
                    if found:
                        # sample up to 5 non-empty values per found column
                        samples = {h: [] for h in found}
                        cnt = 0
                        for row in r:
                            cnt += 1
                            for h in found:
                                v = row.get(h,'')
                                if v and len(samples[h]) < 5:
                                    samples[h].append(v.strip()[:400])
                            if all(len(s)>=5 for s in samples.values()) or cnt>200:
                                break
                        candidates[p] = {'header': header, 'found': found, 'samples': samples}
            except Exception as e:
                print('ERR reading', p, e, file=sys.stderr)

# Print summary
print('Found candidate files with potential text columns:')
for p, info in candidates.items():
    print('\nFILE:', p)
    print('Columns found:', info['found'])
    for h, s in info['samples'].items():
        print('Sample values for', h, '->')
        for i, v in enumerate(s):
            print(f'  [{i+1}]', v[:300])

if not candidates:
    print('No obvious text-like columns found in the demo files.')
else:
    print('\nScan complete: consider using a notes CSV with `text` and `icd_codes` columns for fine-tuning.')
