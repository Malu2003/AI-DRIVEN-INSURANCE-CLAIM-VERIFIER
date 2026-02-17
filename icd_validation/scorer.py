"""Compute match confidence between declared and predicted ICDs."""
from typing import Dict, Callable, Optional
import numpy as np


def same_category(icd1: str, icd2: str) -> bool:
    if not icd1 or not icd2:
        return False
    return icd1[:3] == icd2[:3]


def compute_confidence(
    declared_list, predicted_probs: Dict[str, float], *,
    icd_desc_embs: Optional[Dict[str, any]] = None, embed_fn: Optional[Callable] = None
) -> Dict[str, Dict]:
    """Compute per-declared ICD match scores and reasons.

    Rules:
      - exact match (pred prob >= 0.5) -> 1.0
      - same 3-char category -> 0.6
      - related via embeddings (if available) -> 0.3
      - otherwise 0.0
    """
    results = {}
    top_preds = sorted(predicted_probs.items(), key=lambda x: -x[1])
    for declared in declared_list:
        declared = declared.upper()
        if declared in predicted_probs and predicted_probs[declared] >= 0.5:
            results[declared] = {'score': 1.0, 'reason': 'exact_match', 'pred_prob': predicted_probs[declared]}
            continue
        # category match
        found = False
        for icd, p in top_preds[:10]:
            if same_category(icd, declared):
                results[declared] = {'score': 0.6, 'reason': 'same_category', 'pred_top': icd, 'pred_prob': p}
                found = True
                break
        if found:
            continue
        # related via embeddings (optional)
        if icd_desc_embs and embed_fn:
            try:
                decl_emb = embed_fn(declared)
                best = None
                best_sim = -1.0
                for icd, emb in icd_desc_embs.items():
                    sim = float((decl_emb @ emb) / ((np.linalg.norm(decl_emb) * np.linalg.norm(emb)) + 1e-8))
                    if sim > best_sim:
                        best_sim = sim
                        best = icd
                if best_sim > 0.65:
                    results[declared] = {'score': 0.3, 'reason': 'related', 'best_match': best, 'sim': float(best_sim)}
                    continue
            except Exception:
                pass
        results[declared] = {'score': 0.0, 'reason': 'mismatch'}
    return results
