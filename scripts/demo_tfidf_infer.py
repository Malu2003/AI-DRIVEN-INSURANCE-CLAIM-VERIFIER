"""Demo: load TF-IDF model, predict and score a sample document."""
import argparse
from icd_validation.tfidf import TFIDFClassifier
from icd_validation.infer import predict_and_score
from icd_validation.scorer import compute_confidence


def load_model(path):
    m = TFIDFClassifier()
    m.load(path)
    return m


def predictor(model):
    def _pred(text):
        return model.predict_topk(text, k=50)
    return _pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='checkpoints/tfidf_demo')
    parser.add_argument('--text', required=False)
    args = parser.parse_args()

    model = load_model(args.model)
    pred = predictor(model)

    text = args.text or "Diagnosis: Colon cancer."
    out = predict_and_score(text, pred, compute_score=compute_confidence)
    import json
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
