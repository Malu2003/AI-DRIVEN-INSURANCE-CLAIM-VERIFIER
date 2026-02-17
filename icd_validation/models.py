"""Light wrapper for ClinicalBERT multi-label classifier."""
from typing import List, Dict, Optional

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
except Exception:
    torch = None


class ClinicalBERTClassifier:
    def __init__(self, model_name: str = "emilyalsentzer/Bio_ClinicalBERT", label_list: Optional[List[str]] = None):
        if torch is None:
            raise RuntimeError("torch and transformers are required to instantiate ClinicalBERTClassifier")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.base = AutoModel.from_pretrained(model_name)
        self.label_list = label_list or []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classifier = torch.nn.Linear(self.base.config.hidden_size, len(self.label_list))
        self.to(self.device)

    def to(self, device):
        self.device = device
        self.base.to(device)
        self.classifier.to(device)

    def save(self, path: str):
        import os
        os.makedirs(path, exist_ok=True)
        torch.save({'base': self.base.state_dict(), 'classifier': self.classifier.state_dict(), 'labels': self.label_list}, os.path.join(path, 'model.pth'))

    def load(self, path: str):
        d = torch.load(path, map_location=self.device)
        self.base.load_state_dict(d['base'])
        self.classifier.load_state_dict(d['classifier'])
        self.label_list = d.get('labels', self.label_list)

    def forward(self, input_ids, attention_mask):
        out = self.base(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled = out.pooler_output if hasattr(out, 'pooler_output') and out.pooler_output is not None else out.last_hidden_state[:, 0]
        logits = self.classifier(pooled)
        return logits

    def predict(self, text: str, top_k: int = 10) -> Dict[str, float]:
        enc = self.tokenizer(text, truncation=True, padding=True, return_tensors='pt', max_length=512)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self.forward(enc['input_ids'], enc['attention_mask'])
            probs = torch.sigmoid(logits)[0].cpu().numpy()
        pairs = [(self.label_list[i], float(probs[i])) for i in range(len(self.label_list))]
        pairs = sorted(pairs, key=lambda x: -x[1])[:top_k]
        return {k: v for k, v in pairs}

    def predict_proba(self, text: str) -> Dict[str, float]:
        """Return full probability dict for the given text to match predictor interface."""
        enc = self.tokenizer(text, truncation=True, padding=True, return_tensors='pt', max_length=512)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self.forward(enc['input_ids'], enc['attention_mask'])
            probs = torch.sigmoid(logits)[0].cpu().numpy()
        return {self.label_list[i]: float(probs[i]) for i in range(len(self.label_list))}
