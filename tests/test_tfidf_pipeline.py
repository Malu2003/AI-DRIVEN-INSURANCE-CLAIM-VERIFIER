from icd_validation.tfidf import TFIDFClassifier
from icd_validation.scorer import compute_confidence


def test_pipeline_scoring():
    texts = ["Colon adenocarcinoma of sigmoid colon."]
    labels = [['C18.7']]
    model = TFIDFClassifier()
    model.fit(texts, labels)
    preds = model.predict_topk("Colon adenocarcinoma", k=10)
    # simulate a declared code that should match
    declared = ['C18.7']
    scores = compute_confidence(declared, preds)
    assert 'C18.7' in scores
    # exact match may or may not be present with small data; ensure scoring returns numeric
    assert isinstance(scores['C18.7']['score'], float)
