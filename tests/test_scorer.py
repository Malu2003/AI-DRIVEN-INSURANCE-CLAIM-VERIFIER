from icd_validation.scorer import compute_confidence


def test_compute_confidence_simple():
    declared = ['C18.9']
    predicted = {'C18.9': 0.8, 'E11.65': 0.4}
    res = compute_confidence(declared, predicted)
    assert res['C18.9']['score'] == 1.0

    declared2 = ['C20.1']
    # predicted has same category C20.x
    predicted2 = {'C20.5': 0.3}
    res2 = compute_confidence(declared2, predicted2)
    assert res2['C20.1']['score'] == 0.6
