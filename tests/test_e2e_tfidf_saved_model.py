import csv
import tempfile
from pathlib import Path

from icd_validation.tfidf import TFIDFClassifier
from icd_validation.infer import predict_and_score
from icd_validation.scorer import compute_confidence


def test_e2e_train_save_load_and_score(tmp_path):
    # create small csv
    p = tmp_path / 'small.csv'
    rows = [
        ('Diagnosis: Colon cancer.', 'C18.9'),
        ('Diagnosis: Diabetes mellitus type 2.', 'E11.9'),
        ('Diagnosis: Asthma.', 'J45')
    ]
    with open(p, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'icd_codes'])
        for t, c in rows:
            writer.writerow([t, c])

    # load csv and train
    texts = [r[0] for r in rows]
    labels = [[r[1]] for r in rows]
    model = TFIDFClassifier()
    model.fit(texts, labels)

    # save and reload
    outdir = tmp_path / 'model'
    model.save(str(outdir))
    model2 = TFIDFClassifier()
    model2.load(str(outdir))

    # sample document with declared ICD code
    doc = 'Patient note: C18.9. Patient presented with abdominal pain.'

    # use model2 with predict_and_score pipeline
    result = predict_and_score(doc, model2, compute_score=compute_confidence)

    assert 'declared' in result
    assert 'explain' in result
    assert isinstance(result['explain'], list)
    # there should be one declared code
    assert len(result['declared']) == 1

    rec = result['explain'][0]
    # check required explain fields exist
    for k in ['declared', 'predicted_top', 'predicted_prob', 'match_type', 'score', 'details']:
        assert k in rec
    assert rec['declared'] == 'C18.9'
