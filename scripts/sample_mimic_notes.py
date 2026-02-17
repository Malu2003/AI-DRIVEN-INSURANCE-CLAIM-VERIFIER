import gzip, csv, os
files = [
r"d:\tech_squad\AI-Driven-Image-Forgery-Detection\data\mimic-iv-notes-2.2\note\discharge.csv.gz",
r"d:\tech_squad\AI-Driven-Image-Forgery-Detection\data\mimic-iv-notes-2.2\note\discharge_detail.csv.gz",
r"d:\tech_squad\AI-Driven-Image-Forgery-Detection\data\mimic-iv-notes-2.2\note\radiology.csv.gz",
r"d:\tech_squad\AI-Driven-Image-Forgery-Detection\data\mimic-iv-notes-2.2\note\radiology_detail.csv.gz",
]
for f in files:
    if not os.path.exists(f):
        print("Missing:", f)
        continue
    print("==", f)
    with gzip.open(f, 'rt', encoding='utf-8', errors='replace') as fh:
        reader = csv.reader(fh)
        try:
            hdr = next(reader)
        except StopIteration:
            print("empty")
            continue
        print("HEADER:", hdr)
        for i,row in zip(range(2), reader):
            print(row)
