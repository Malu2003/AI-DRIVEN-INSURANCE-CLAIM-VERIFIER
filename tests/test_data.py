import os
import pytest
from icd_validation.data import load_diagnosis_csv


def test_load_diagnosis_csv(tmp_path):
    try:
        import pandas as pd
    except Exception as e:
        import pytest
        pytest.skip(f'pandas import failed in this environment: {e}')
    df = pd.DataFrame({'Code': ['A00', 'C18.9'], 'ShortDescription': ['Cholera', 'Colon cancer']})
    p = tmp_path / 'diagnosis_sample.csv'
    df.to_csv(p, index=False)
    mapping, df_out = load_diagnosis_csv(str(p))
    assert 'A00' in mapping
    assert mapping['C18.9'] == 'Colon cancer'
