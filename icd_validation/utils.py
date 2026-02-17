"""Utility functions: ICD extraction and normalization."""
import re
from typing import List

# ICD-10 regex: letter (except U for certain reserved) + 2 digits, optional dot and 1-4 alphanum
ICD10_RE = re.compile(r"\b([A-TV-Z][0-9]{2}(?:\.[0-9A-Za-z]{1,4})?)\b", re.IGNORECASE)


def normalize_icd(icd: str) -> str:
    if not icd or not isinstance(icd, str):
        return ''
    s = icd.strip().upper()
    s = s.replace(' ', '')
    return s


def extract_declared_icds(text: str) -> List[str]:
    """Extract declared ICDs from free form text using regex.

    Returns normalized, unique codes.
    """
    if not text:
        return []
    found = {normalize_icd(m.group(1)) for m in ICD10_RE.finditer(text)}
    return sorted(found)


def mask_icd_mentions(text: str, mask_token: str = '[ICD_CODE]') -> str:
    """Mask explicit ICD mentions found in the text.

    Replaces matches from `ICD10_RE` with `mask_token`. This prevents the
    model from trivially copying explicitly written ICD codes from the note
    into predictions. Use this during training and inference preprocessing.
    """
    if not text:
        return text
    return ICD10_RE.sub(mask_token, text)
