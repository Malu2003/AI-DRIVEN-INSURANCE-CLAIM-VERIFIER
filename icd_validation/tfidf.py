"""ClinicalBERT-based ICD predictor (replaces TF-IDF feature extractor).

This file preserves the original `TFIDFClassifier` class name and public API
for backward compatibility with the rest of the codebase, but replaces the
TF-IDF vectorizer + sklearn classifier with a Hugging Face ClinicalBERT
feature extractor + a lightweight PyTorch linear head trained with
BCEWithLogitsLoss (multi-label).

Notes:
- This is a feature extractor upgrade only: dataset splits, label encoding,
  evaluation metrics, and scoring logic are unchanged.
- The `fit` method now trains a small PyTorch classifier head on top of
  ClinicalBERT embeddings. By default the BERT encoder is kept frozen (no
  fine-tuning) for speed and stability; this can be changed if desired.
"""
from typing import List, Dict, Optional
import os
import joblib
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MultiLabelBinarizer

from transformers import AutoTokenizer, AutoModel


class _TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: Optional[List[List[int]]], tokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        t = self.texts[idx]
        enc = self.tokenizer(t, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        item = {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
        }
        if self.labels is not None:
            item['label'] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item


class TFIDFClassifier:
    """Compatibility wrapper: uses ClinicalBERT embeddings + linear head.

    Public API kept the same as the old TF-IDF implementation:
    - fit(texts, labels)
    - predict_proba(texts)
    - predict_proba_single(text)
    - save(path)
    - load(path)
    - predict_topk(text)

    Comments mark the parts that replaced TF-IDF.
    """

    def __init__(self, model_name: str = 'emilyalsentzer/Bio_ClinicalBERT', device: Optional[torch.device] = None, embedding_pool: str = 'mean'):
        # Replaces TF-IDF vectorizer with HF tokenizer + encoder
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.embedding_pool = embedding_pool  # 'cls' or 'mean'

        # Default device
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.bert.to(self.device)

        # By default we freeze BERT parameters (only train the linear head)
        for p in self.bert.parameters():
            p.requires_grad = False

        self.head: Optional[nn.Module] = None
        self.labels: List[str] = []
        self.mlb: Optional[MultiLabelBinarizer] = None

    def _embed_texts(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        self.bert.eval()
        embeddings = []
        ds = _TextDataset(texts, labels=None, tokenizer=self.tokenizer)
        dl = DataLoader(ds, batch_size=batch_size)
        with torch.no_grad():
            for batch in dl:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                out = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
                if self.embedding_pool == 'cls' and hasattr(out, 'pooler_output') and out.pooler_output is not None:
                    emb = out.pooler_output
                else:
                    # mean pooling over sequence length using attention mask
                    last = out.last_hidden_state  # (B, L, H)
                    mask = attention_mask.unsqueeze(-1).type_as(last)
                    summed = (last * mask).sum(1)
                    counts = mask.sum(1).clamp(min=1e-9)
                    emb = summed / counts
                embeddings.append(emb.cpu())
        return torch.cat(embeddings, dim=0)

    def fit(self, texts: List[str], labels: List[List[str]], epochs: int = 3, batch_size: int = 32, lr: float = 1e-3):
        """Train only the classification head on top of ClinicalBERT embeddings.

        Level-1 fine-tuning on MIMIC-IV (ClinicalBERT frozen):
        - BERT encoder is frozen (no fine-tuning)
        - Embeddings are computed via mean pooling over last hidden states
        - Head is trained using BCEWithLogitsLoss and AdamW (lr default 1e-3)

        This preserves the existing training semantics while enabling task-specific
        learning from clinical notes.
        """
        # Build multilabel binarizer (unchanged)
        self.mlb = MultiLabelBinarizer()
        Y = self.mlb.fit_transform(labels)
        self.labels = list(self.mlb.classes_)

        num_classes = len(self.labels)
        # Compute embeddings (frozen BERT by default) and train a linear head on top
        X_emb = self._embed_texts(texts, batch_size=batch_size)

        # Create head if missing or mismatched
        emb_dim = X_emb.size(1)
        if self.head is None or (hasattr(self.head, 'out_features') and self.head.out_features != num_classes):
            self.head = nn.Linear(emb_dim, num_classes)
        self.head.to(self.device)

        dataset = torch.utils.data.TensorDataset(X_emb, torch.tensor(Y, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.head.parameters(), lr=lr)

        self.head.train()
        for epoch in range(epochs):
            ep_loss = 0.0
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                logits = self.head(xb)
                loss = criterion(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ep_loss += loss.item() * xb.size(0)
            ep_loss /= len(dataset)
            # minimal logging — preserve modularity
            print(f"[TFIDF->BERT] Epoch {epoch+1}/{epochs} loss={ep_loss:.4f}")

    def predict_proba(self, texts: List[str]) -> List[Dict[str, float]]:
        X_emb = self._embed_texts(texts, batch_size=32)
        self.head.eval()
        with torch.no_grad():
            logits = self.head(X_emb.to(self.device))
            probs = torch.sigmoid(logits).cpu().numpy()
        out = []
        for row in probs:
            d = {self.labels[i]: float(row[i]) for i in range(len(self.labels))}
            out.append(d)
        return out

    def predict_proba_single(self, text: str) -> Dict[str, float]:
        return self.predict_proba([text])[0]

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        # Save head weights and metadata; do not attempt to save full BERT weights here.
        head_path = os.path.join(path, 'head.pth')
        torch.save(self.head.state_dict() if self.head is not None else {}, head_path)
        joblib.dump({'labels': self.labels, 'model_name': self.model_name, 'embedding_pool': self.embedding_pool, 'head_file': 'head.pth'}, os.path.join(path, 'tfidf_model.joblib'))

    def load(self, path: str):
        d = joblib.load(os.path.join(path, 'tfidf_model.joblib'))
        self.labels = d['labels']
        self.model_name = d.get('model_name', self.model_name)
        self.embedding_pool = d.get('embedding_pool', self.embedding_pool)
        head_file = d.get('head_file', 'head.pth')

        # Re-init tokenizer and BERT and head with correct shapes
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.bert = AutoModel.from_pretrained(self.model_name)
        self.bert.to(self.device)
        for p in self.bert.parameters():
            p.requires_grad = False

        # create head and load state
        if len(self.labels) == 0:
            raise ValueError('No labels found in saved model')
        # create a dummy embedding to get embedding dim
        self.bert.eval()
        with torch.no_grad():
            # single space token to get dimensions
            dummy = self.tokenizer(' ', return_tensors='pt', truncation=True, padding=True)
            dummy = {k: v.to(self.device) for k, v in dummy.items()}
            out = self.bert(**dummy, return_dict=True)
            if self.embedding_pool == 'cls' and hasattr(out, 'pooler_output') and out.pooler_output is not None:
                emb_dim = out.pooler_output.size(1)
            else:
                last = out.last_hidden_state
                emb_dim = last.size(2)

        self.head = nn.Linear(emb_dim, len(self.labels))
        self.head.load_state_dict(torch.load(os.path.join(path, head_file), map_location=self.device))
        self.head.to(self.device)

    def predict_topk(self, text: str, k: int = 10) -> Dict[str, float]:
        out = self.predict_proba([text])[0]
        return dict(sorted(out.items(), key=lambda x: -x[1])[:k])

