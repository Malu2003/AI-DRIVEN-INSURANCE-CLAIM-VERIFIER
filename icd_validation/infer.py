"""Inference pipeline for ICD prediction and scoring."""
from typing import Callable, Dict, Any, List
from .utils import extract_declared_icds

try:
    import fitz  # pymupdf
    from PIL import Image
    import pytesseract
except Exception:
    fitz = None


def extract_text_from_pdf(path_or_file) -> str:
    """Extract text from PDF; requires PyMuPDF. If not available returns empty string."""
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) required for PDF extraction; install pymupdf")
    doc = fitz.open(path_or_file)
    chunks = []
    for page in doc:
        txt = page.get_text()
        if txt and txt.strip():
            chunks.append(txt)
        else:
            pix = page.get_pixmap(dpi=150)
            img = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
            chunks.append(pytesseract.image_to_string(img))
    return '\n'.join(chunks)


def predict_and_score(text: str, predictor: Callable[[str], Dict[str, float]]|object, *,
                      compute_score: Callable[[List[str], Dict[str, float]], Dict[str, Any]] = None) -> Dict:
    """Given document text and a predictor (callable or object with `predict_proba`), extract declared ICDs, predict, and compute score.

    Returns an explainable structure with per-declared entries:
      - declared: the declared ICD
      - predicted_top: top predicted ICD (or None)
      - predicted_prob: its probability
      - match_type: reason (exact_match, same_category, related, mismatch)
      - score: numeric score in [0,1]
      - details: additional info from the scorer
    """
    declared = extract_declared_icds(text)
    # support either a callable that returns dict or an object with predict_proba(text)
    preds = {}
    if hasattr(predictor, 'predict_proba_single'):
        preds = predictor.predict_proba_single(text)
    elif hasattr(predictor, 'predict_proba'):
        # try calling with single text; fallback to list input
        try:
            res = predictor.predict_proba(text)
        except TypeError:
            res = predictor.predict_proba([text])
        if isinstance(res, list):
            preds = res[0]
        else:
            preds = res
    else:
        preds = predictor(text)
    if compute_score is not None:
        scores = compute_score(declared, preds)
    else:
        scores = {d: {'score': 0.0, 'reason': 'no_scorer'} for d in declared}

    # assemble explainable records
    sorted_preds = sorted(preds.items(), key=lambda x: -x[1])[:50]
    def top_pred():
        return sorted_preds[0] if sorted_preds else (None, 0.0)

    explain = []
    for d in declared:
        rec = {}
        rec['declared'] = d
        topk = top_pred()
        rec['predicted_top'] = topk[0]
        rec['predicted_prob'] = float(topk[1]) if topk[0] is not None else 0.0
        sc = scores.get(d, {'score': 0.0, 'reason': 'missing'})
        rec['match_type'] = sc.get('reason')
        rec['score'] = float(sc.get('score', 0.0))
        rec['details'] = {k: v for k, v in sc.items() if k not in ('score', 'reason')}
        explain.append(rec)

    out = {
        'declared': declared,
        'predicted': sorted_preds[:10],
        'explain': explain,
        'scores': scores,
        'features': {
            'max_pred_prob': max(preds.values()) if preds else 0.0,
            'num_declared': len(declared),
        }
    }
    return out


def summarize_report(report: Dict, accept_threshold: float = 0.6, review_threshold: float = 0.3) -> List[str]:
    """Produce a short human-readable summary from a predict_and_score report.

    Decision logic:
      - score >= accept_threshold -> 'ACCEPT'
      - score >= review_threshold -> 'REVIEW'
      - else -> 'REJECT'
    Returns list of summary lines (strings).
    """
    lines = []
    explains = report.get('explain', [])
    if not explains:
        lines.append('No declared ICDs found.')
        return lines
    for rec in explains:
        declared = rec.get('declared')
        pred = rec.get('predicted_top') or 'None'
        prob = rec.get('predicted_prob', 0.0)
        match = rec.get('match_type', 'unknown')
        score = float(rec.get('score', 0.0))
        if score >= accept_threshold:
            decision = 'ACCEPT'
        elif score >= review_threshold:
            decision = 'REVIEW'
        else:
            decision = 'REJECT'
        lines.append(f"Declared: {declared} | Predicted: {pred} (p={prob:.3f}) | Match: {match} | Score: {score:.2f} | Decision: {decision}")
    return lines
