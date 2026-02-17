"""Demo script: extract declared ICDs from a document and produce explainable JSON report.

Usage examples:
  python scripts/demo_document_infer.py --file sample.txt --model checkpoints/tfidf_demo
  python scripts/demo_document_infer.py --text "Patient had C18.9" --model checkpoints/tfidf_demo
"""
import argparse
import json
from pathlib import Path

from icd_validation.tfidf import TFIDFClassifier
from icd_validation.infer import extract_text_from_pdf, predict_and_score
from icd_validation.scorer import compute_confidence


def load_model(path: str):
    model = TFIDFClassifier()
    model.load(path)
    return model


def read_text_from_file(path: Path) -> str:
    if path.suffix.lower() == '.pdf':
        # may raise RuntimeError if pymupdf not installed
        return extract_text_from_pdf(str(path))
    else:
        return path.read_text(encoding='utf-8', errors='ignore')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--file', help='Path to text or PDF file (optional)')
    p.add_argument('--text', help='Text string to run inference on (optional)')
    p.add_argument('--model', default='checkpoints/tfidf_demo', help='Path to TF-IDF or model dir')
    p.add_argument('--out', help='Path to write JSON output (default stdout)')
    args = p.parse_args()

    if not args.file and not args.text:
        raise SystemExit('Provide --file or --text')

    model = load_model(args.model)

    if args.file:
        txt = read_text_from_file(Path(args.file))
    else:
        txt = args.text

    # run the pipeline — predictor is the model object which exposes predict_proba
    report = predict_and_score(txt, model, compute_score=compute_confidence)

    out_json = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(out_json, encoding='utf-8')
        print('Wrote', args.out)
    else:
        print(out_json)

    # print a short human-readable summary
    try:
        from icd_validation.infer import summarize_report
        lines = summarize_report(report)
        print('\nSummary:')
        for l in lines:
            print(' -', l)
    except Exception:
        pass


if __name__ == '__main__':
    main()
