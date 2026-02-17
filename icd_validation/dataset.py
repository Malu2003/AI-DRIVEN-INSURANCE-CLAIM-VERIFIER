"""Dataset utilities for building multi-label ICD datasets from MIMIC-like CSVs."""
from typing import List, Tuple, Dict, Iterable
from collections import Counter


def load_text_icd_csv(path: str, text_col: str = 'text', icd_col: str = 'icd_codes', sep: str = ';') -> List[Tuple[str, List[str]]]:
    """Load a CSV where each row has text and a separator-delimited icd code list.

    Returns list of (text, [icd1, icd2,...]).
    """
    import csv
    rows = []
    with open(path, encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        for r in reader:
            text = (r.get(text_col) or '').strip()
            codes = (r.get(icd_col) or '').strip()
            codes_list = [c.strip().upper() for c in codes.split(sep) if c.strip()] if codes else []
            if text:
                rows.append((text, codes_list))
    return rows


def build_label_list(samples: Iterable[Tuple[str, List[str]]], top_k: int = 500) -> List[str]:
    """Return the most frequent ICD codes up to top_k as the label list."""
    ctr = Counter()
    for _, codes in samples:
        ctr.update(codes)
    return [c for c, _ in ctr.most_common(top_k)]


def make_multihot(labels: List[str], label_list: List[str]) -> List[int]:
    idx = {c: i for i, c in enumerate(label_list)}
    out = [0] * len(label_list)
    for c in labels:
        if c in idx:
            out[idx[c]] = 1
    return out


class ICDTextDataset:
    """A minimal dataset that tokenizes on the fly using a HuggingFace tokenizer."""
    def __init__(self, samples: List[Tuple[str, List[str]]], tokenizer, label_list: List[str], max_len: int = 512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.label_list = label_list
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        text, codes = self.samples[i]
        enc = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        labels = make_multihot(codes, self.label_list)
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item['labels'] = labels
        return item
