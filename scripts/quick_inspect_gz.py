import gzip, csv, sys
files=[
 'data/mimic-iv-clinical-database-demo-2.2/hosp/diagnoses_icd.csv.gz',
 'data/mimic-iv-clinical-database-demo-2.2/hosp/admissions.csv.gz',
 'data/mimic-iv-clinical-database-demo-2.2/hosp/procedures_icd.csv.gz'
]
for p in files:
    try:
        with gzip.open(p,'rt',encoding='utf-8') as f:
            r=csv.reader(f)
            header=next(r)
            print(p, '->', header[:10])
    except Exception as e:
        print('ERR',p,e)
