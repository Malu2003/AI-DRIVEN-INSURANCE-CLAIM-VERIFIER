"""Data utilities for ICD validation."""
from typing import Tuple, Dict


def load_diagnosis_csv(path: str):
    """Load diagnosis CSV and return mapping ICD->description and DataFrame.

    Imports pandas lazily to avoid hard dependency during light-weight unit tests.
    """
    import pandas as pd
    from pandas import DataFrame

    df = pd.read_csv(path, dtype=str)
    # expected columns: Code, ShortDescription, LongDescription
    codes = df['Code'].str.upper().str.strip()
    desc = df.get('ShortDescription') if 'ShortDescription' in df.columns else df.get('LongDescription')
    icd_to_desc = {c: (d if pd.notna(d) else '') for c, d in zip(codes, desc.fillna(''))}
    return icd_to_desc, df
    """Load diagnosis CSV and return mapping ICD->description and DataFrame.

    Args:
        path: path to diagnosis.csv

    Returns:
        (icd_to_desc, df)
    """
    df = pd.read_csv(path, dtype=str)
    # expected columns: Code, ShortDescription, LongDescription
    codes = df['Code'].str.upper().str.strip()
    desc = df.get('ShortDescription') if 'ShortDescription' in df.columns else df.get('LongDescription')
    icd_to_desc = {c: (d if pd.notna(d) else '') for c, d in zip(codes, desc.fillna(''))}
    return icd_to_desc, df
