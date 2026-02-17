"""Training script skeleton for ClinicalBERT ICD prediction."""
import argparse
import logging


import os
import argparse
import logging

try:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AdamW
except Exception:
    torch = None

from icd_validation.dataset import load_text_icd_csv, build_label_list, ICDTextDataset
from icd_validation.models import ClinicalBERTClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_csv', required=False, help='CSV with text and icd_codes columns')
    parser.add_argument('--text_col', default='text')
    parser.add_argument('--icd_col', default='icd_codes')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--top_k', type=int, default=200)
    parser.add_argument('--smoke', action='store_true', help='Run a tiny smoke-train without external data')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.smoke:
        # tiny synthetic dataset
        samples = [ (f"Patient has colon cancer stage {i}", ['C18.9']) for i in range(32) ]
    else:
        if not args.data_csv:
            raise SystemExit('Provide --data_csv or use --smoke')
        samples = load_text_icd_csv(args.data_csv, text_col=args.text_col, icd_col=args.icd_col)

    label_list = build_label_list(samples, top_k=args.top_k)
    if not label_list:
        raise SystemExit('No labels found; cannot train')

    # setup tokenizer and dataset
    model = ClinicalBERTClassifier(label_list=label_list)
    dataset = ICDTextDataset(samples, model.tokenizer, label_list, max_len=128)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda batch: collate_batch(batch, model.tokenizer))

    if torch is None:
        raise RuntimeError('torch and transformers are required to train')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optim = AdamW(list(model.base.parameters()) + list(model.classifier.parameters()), lr=2e-5)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    model.base.train(); model.classifier.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].float().to(device)
            optim.zero_grad()
            logits = model.forward(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            optim.step()
            total_loss += float(loss.item())
        logging.info('Epoch %d loss %.4f', epoch+1, total_loss/len(dataloader))

    os.makedirs(args.output_dir, exist_ok=True)
    model.save(args.output_dir)
    logging.info('Saved model to %s', args.output_dir)


def collate_batch(batch, tokenizer):
    import torch
    # batch is list of items returned by ICDTextDataset: dicts with input_ids, attention_mask, labels(list)
    input_ids = torch.stack([b['input_ids'] for b in batch])
    attention_mask = torch.stack([b['attention_mask'] for b in batch])
    labels = torch.tensor([b['labels'] for b in batch], dtype=torch.float32)
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


if __name__ == '__main__':
    main()
