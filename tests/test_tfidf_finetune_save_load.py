import os
from icd_validation.tfidf import TFIDFClassifier


def test_save_load(tmp_path):
    texts = ["Patient has colon cancer","Diabetes mellitus type 2"]
    labels = [['C18.9'], ['E11.9']]
    model = TFIDFClassifier()
    model.fit(texts, labels, epochs=1, batch_size=2)
    outdir = tmp_path / 'model'
    model.save(str(outdir))

    m2 = TFIDFClassifier()
    m2.load(str(outdir))
    assert set(model.labels) == set(m2.labels)
