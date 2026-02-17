ClinicalBERT-based ICD feature extractor

This module replaces the original TF-IDF vectorizer with ClinicalBERT embeddings while
preserving the original `TFIDFClassifier` API (fit, predict_proba, save, load, predict_topk).

Highlights
- Uses `emilyalsentzer/Bio_ClinicalBERT` tokenizer + encoder to compute text embeddings.
- Trains a small PyTorch linear head on top of frozen BERT embeddings using BCEWithLogitsLoss.
- Backwards compatible: code that imports `TFIDFClassifier` should continue to work.

Saving & Loading
- `save(path)` stores `head.pth` and a `tfidf_model.joblib` metadata file in `path`.
- `load(path)` restores metadata and head weights and re-initializes the tokenizer and BERT encoder.

Quick commands
- Compare TF-IDF baseline vs ClinicalBERT on a synthetic dataset:
  python -m icd_validation.compare_tfidf_vs_bert --bert-epochs 1

Notes
- The training is done on the linear head by default (BERT frozen). You can edit `icd_validation/tfidf.py`
  to enable fine-tuning of the encoder if desired.
- Tests or CI may download ClinicalBERT and take time; consider using a smaller model for fast tests if
  necessary.
