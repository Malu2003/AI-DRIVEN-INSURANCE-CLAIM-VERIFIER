import numpy as np
from icd_validation.scorer import compute_confidence


def test_exact_same_mismatch():
    declared = ['C18.9']
    predicted = {'C18.9': 0.9, 'E11.9': 0.2}
    res = compute_confidence(declared, predicted)
    assert res['C18.9']['score'] == 1.0

    declared2 = ['C20.1']
    predicted2 = {'C20.5': 0.3}
    res2 = compute_confidence(declared2, predicted2)
    assert res2['C20.1']['score'] == 0.6

    declared3 = ['Z99.9']
    predicted3 = {'A00': 0.1}
    res3 = compute_confidence(declared3, predicted3)
    assert res3['Z99.9']['score'] == 0.0


def test_related_via_embeddings():
    # make an icd_desc_embs with a close vector
    icd_embs = {
        'A123': np.array([1.0, 0.0]),
        'B456': np.array([0.0, 1.0])
    }

    def embed_fn(x):
        # declared 'A999' is close to 'A123'
        return np.array([0.9, 0.1])

    declared = ['A999']
    predicted = {'C00': 0.05}
    res = compute_confidence(declared, predicted, icd_desc_embs=icd_embs, embed_fn=embed_fn)
    # related should be selected
    assert res['A999']['score'] == 0.3
