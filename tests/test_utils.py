import tempfile
from icd_validation.utils import extract_declared_icds, normalize_icd


def test_normalize_icd():
    assert normalize_icd(' c18.9 ') == 'C18.9'
    assert normalize_icd('e1165') == 'E1165'


def test_extract_declared_icds():
    text = 'Patient was coded with C18.9 and E11.65; also mentions A00'
    codes = extract_declared_icds(text)
    assert 'C18.9' in codes
    assert 'E11.65' in codes
    assert 'A00' in codes
