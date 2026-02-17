import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from icd_validation.tfidf import TFIDFClassifier
except Exception as e:
    print('IMPORT ERROR:', repr(e))
    raise

texts = ["Patient has colon cancer","Diabetes mellitus type 2","Asthma and bronchitis"]
labels = [['C18.9'], ['E11.9'], ['J45']]
model = TFIDFClassifier()
model.fit(texts, labels, epochs=1, batch_size=2)
print('fit done')
out = model.predict_topk('Patient diagnosed with colon cancer', k=3)
print('topk:', out)
probs = model.predict_proba_single('Patient diagnosed with colon cancer')
print('proba keys count:', len(probs))
