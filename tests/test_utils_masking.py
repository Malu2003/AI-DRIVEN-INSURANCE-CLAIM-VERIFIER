from icd_validation.utils import mask_icd_mentions


def test_mask_icd_mentions():
    s = 'Diagnosis: Acute myocardial infarction. ICD-10: I21.9 and code I21.9 again.'
    out = mask_icd_mentions(s)
    assert 'I21.9' not in out
    assert '[ICD_CODE]' in out
