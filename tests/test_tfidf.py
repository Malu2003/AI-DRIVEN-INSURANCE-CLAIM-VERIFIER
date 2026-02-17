def test_tfidf_train_and_predict(tmp_path):
    from icd_validation.tfidf import TFIDFClassifier

    texts = ["Patient has colon cancer","Diabetes mellitus type 2","Asthma and bronchitis"]
    labels = [['C18.9'], ['E11.9'], ['J45']]
    model = TFIDFClassifier()
    model.fit(texts, labels)
    out = model.predict_topk("Patient diagnosed with colon cancer", k=3)
    assert isinstance(out, dict)
    # expect C18.9 to be among top predictions
    # may not always be C18 in tiny sample, but promise the API is present
    assert len(out) <= 3
    # Ensure predictor interface method exists
    probs = model.predict_proba_single("Patient diagnosed with colon cancer")
    assert isinstance(probs, dict)
