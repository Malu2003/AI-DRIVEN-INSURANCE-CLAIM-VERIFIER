"""Extended head-only training for ClinicalBERT ICD classification.

Trains only the linear head on top of ClinicalBERT embeddings (encoder frozen).
Saves metrics, head checkpoint, and a loss curve plot.
"""
import argparse
import json
from pathlib import Path
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import matplotlib.pyplot as plt



import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from icd_validation.tfidf import TFIDFClassifier


def read_jsonl(path):
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            out.append(json.loads(line))
    return out


def build_label_list_from_map(path):
    with open(path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    # label_map.json: {'label_to_idx': {...}, 'idx_to_label': {...}}
    if 'label_to_idx' in d:
        # sort by idx
        idx_to_label = {int(k):v for k,v in d.get('idx_to_label', {}).items()}
        if idx_to_label:
            return [idx_to_label[i] for i in sorted(idx_to_label.keys())]
        # fallback
        label_to_idx = d['label_to_idx']
        return sorted(label_to_idx.keys(), key=lambda x: label_to_idx[x])
    # legacy simple list
    if isinstance(d, list):
        return d
    raise SystemExit('Unsupported label_map.json format')


def to_multihot(labels, label_list):
    idx = {c:i for i,c in enumerate(label_list)}
    out = np.zeros(len(label_list), dtype=np.float32)
    for c in labels:
        if c in idx:
            out[idx[c]] = 1.0
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train', default='processed/mimic_notes/train.jsonl')
    p.add_argument('--val', default='processed/mimic_notes/val.jsonl')
    p.add_argument('--label-map', default='processed/mimic_notes/label_map.json')
    # Phase 1 (head-only) args
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--out', default='checkpoints/icd_head_extended')
    p.add_argument('--model-name', default='emilyalsentzer/Bio_ClinicalBERT')
    p.add_argument('--device', default=None, choices=['cpu','cuda'], help='Device to use (default: auto-detect)')

    # Phase 2 (optional finetune) args
    p.add_argument('--finetune', action='store_true', help='Enable partial fine-tuning of last N layers')
    p.add_argument('--unfreeze-layers', type=int, default=2, help='Number of last encoder layers to unfreeze during finetune')
    p.add_argument('--finetune-epochs', type=int, default=4, help='Epochs to run during finetune phase')
    p.add_argument('--finetune-batch-size', type=int, default=8, help='Batch size during finetune (tokenized)')
    p.add_argument('--finetune-lr', type=float, default=2e-5, help='LR for finetune phase')
    p.add_argument('--use-amp', action='store_true', help='Use mixed precision (AMP) during finetune')
    p.add_argument('--finetune-out', default=None, help='Output dir for finetune phase (defaults to <out>_finetune)')

    args = p.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # load data
    train_json = read_jsonl(args.train)
    val_json = read_jsonl(args.val)
    print(f"Loaded train={len(train_json)} val={len(val_json)}")

    label_list = build_label_list_from_map(args.label_map)
    print(f"Label list length: {len(label_list)}")

    train_texts = [o['text'] for o in train_json]
    train_labels = [o['labels'] for o in train_json]
    val_texts = [o['text'] for o in val_json]
    val_labels = [o['labels'] for o in val_json]

    # instantiate a TFIDFClassifier to access tokenizer and bert (encoder will remain frozen)
    clf = TFIDFClassifier(model_name=args.model_name)
    # allow user override of device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Device:', device)

    # compute embeddings (these are CPU tensors in _embed_texts implementation)
    print('Computing train embeddings...')
    X_train = clf._embed_texts(train_texts, batch_size=32)
    print('Computing val embeddings...')
    X_val = clf._embed_texts(val_texts, batch_size=32)

    # convert labels to multihot using provided label_list
    Y_train = np.stack([to_multihot(l, label_list) for l in train_labels])
    Y_val = np.stack([to_multihot(l, label_list) for l in val_labels])

    X_train = X_train.float()
    X_val = X_val.float()
    Y_train = torch.tensor(Y_train, dtype=torch.float32)
    Y_val = torch.tensor(Y_val, dtype=torch.float32)

    # setup head
    emb_dim = X_train.size(1)
    num_classes = len(label_list)
    head = nn.Linear(emb_dim, num_classes)
    head = head.to(device)

    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    train_ds = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(args.epochs):
        head.train()
        ep_loss = 0.0
        total = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = head(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * xb.size(0)
            total += xb.size(0)
        ep_loss /= total
        history['train_loss'].append(ep_loss)

        # validation loss
        head.eval()
        with torch.no_grad():
            xb = X_val.to(device)
            yb = Y_val.to(device)
            logits = head(xb)
            v_loss = criterion(logits, yb).item()
        history['val_loss'].append(v_loss)

        print(f"Epoch {epoch+1}/{args.epochs} | train_loss={ep_loss:.4f} | val_loss={v_loss:.4f}")

    # save head weights and metadata (mimic TFIDFClassifier.save semantics)
    head_path = outdir / 'head.pth'
    torch.save(head.state_dict(), str(head_path))
    meta = {'labels': label_list, 'model_name': args.model_name, 'embedding_pool': clf.embedding_pool, 'head_file': 'head.pth'}
    import joblib
    joblib.dump(meta, str(outdir / 'tfidf_model.joblib'))

    # metrics
    metrics = {
        'train_loss_per_epoch': history['train_loss'],
        'val_loss_per_epoch': history['val_loss'],
        'num_train': len(train_json),
        'num_val': len(val_json),
        'labels': num_classes
    }
    with open(outdir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print('Saved metrics to', outdir / 'metrics.json')

    # plot loss curve
    try:
        plt.figure()
        plt.plot(range(1, args.epochs+1), history['train_loss'], label='train')
        plt.plot(range(1, args.epochs+1), history['val_loss'], label='val')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / 'loss_curve.png')
        print('Saved loss plot to', outdir / 'loss_curve.png')
    except Exception as e:
        print('Could not save plot:', e)

    # sample predictions on validation (head-only model)
    with torch.no_grad():
        head.eval()
        logits = head(X_val.to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
    # print top-5 preds for first few val samples
    for i in range(min(5, len(val_texts))):
        row_probs = probs[i]
        top_idx = np.argsort(-row_probs)[:5]
        top_preds = [(label_list[j], float(row_probs[j])) for j in top_idx]
        print(f"Val sample {i+1}: top_preds={top_preds} | true_labels={val_labels[i]}")

    print('Extended head-only training complete. Check', outdir)

    # Phase 2: optional partial fine-tuning
    if args.finetune:
        finetune_out = args.finetune_out or (str(outdir) + '_finetune')
        print('Starting finetune phase ->', finetune_out)
        finetune_out = Path(finetune_out)
        finetune_out.mkdir(parents=True, exist_ok=True)

        # prepare tokenized datasets (only labels that appear in label_list)
        def build_samples(jsonl):
            samples = []
            for o in jsonl:
                codes = [c for c in o['labels'] if c in label_list]
                if not codes:
                    continue
                samples.append((o['text'], codes))
            return samples

        train_samples = build_samples(train_json)
        val_samples = build_samples(val_json)
        print(f"Finetune samples: train={len(train_samples)} val={len(val_samples)}")

        # import here to ensure repo path is set
        from icd_validation.dataset import ICDTextDataset
        train_dataset = ICDTextDataset(train_samples, tokenizer=clf.tokenizer, label_list=label_list)
        val_dataset = ICDTextDataset(val_samples, tokenizer=clf.tokenizer, label_list=label_list)

        def collate_fn(batch):
            input_ids = torch.stack([item['input_ids'] for item in batch])
            attention_mask = torch.stack([item['attention_mask'] for item in batch])
            labels = torch.tensor([item['labels'] for item in batch], dtype=torch.float32)
            return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

        train_loader = DataLoader(train_dataset, batch_size=args.finetune_batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=args.finetune_batch_size, shuffle=False, collate_fn=collate_fn)

        # attach head to clf and move models to device
        clf.head = head
        clf.head.to(device)
        clf.bert.to(device)

        # freeze all bert params, then unfreeze last N layers
        for p in clf.bert.parameters():
            p.requires_grad = False
        n_unfreeze = max(1, int(args.unfreeze_layers))
        try:
            layers = clf.bert.encoder.layer
            to_unfreeze = layers[-n_unfreeze:]
        except Exception:
            # fallback (HuggingFace naming may vary)
            print('Could not unfreeze requested layers; skipping')
            to_unfreeze = []
        for layer in to_unfreeze:
            for p in layer.parameters():
                p.requires_grad = True

        # ensure head params trainable
        for p in clf.head.parameters():
            p.requires_grad = True

        bert_params = [p for p in clf.bert.parameters() if p.requires_grad]
        head_params = [p for p in clf.head.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW([{'params': head_params, 'lr': args.lr}, {'params': bert_params, 'lr': args.finetune_lr}])
        criterion = nn.BCEWithLogitsLoss()

        use_amp = args.use_amp and device.type == 'cuda'
        scaler = GradScaler() if use_amp else None

        finetune_train_losses = []
        finetune_val_losses = []
        for epoch in range(args.finetune_epochs):
            clf.bert.train()
            clf.head.train()
            ep_loss = 0.0
            total = 0
            for batch in train_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                optimizer.zero_grad()
                with autocast(enabled=use_amp):
                    out = clf.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
                    if clf.embedding_pool == 'cls' and hasattr(out, 'pooler_output') and out.pooler_output is not None:
                        emb = out.pooler_output
                    else:
                        last = out.last_hidden_state
                        mask = attention_mask.unsqueeze(-1).type_as(last)
                        summed = (last * mask).sum(1)
                        counts = mask.sum(1).clamp(min=1e-9)
                        emb = summed / counts
                    logits = clf.head(emb)
                    loss = criterion(logits, labels)
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                ep_loss += loss.item() * input_ids.size(0)
                total += input_ids.size(0)
            ep_loss /= max(1, total)
            finetune_train_losses.append(ep_loss)

            # validation
            clf.bert.eval()
            clf.head.eval()
            v_loss = 0.0
            v_total = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    with autocast(enabled=use_amp):
                        out = clf.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
                        if clf.embedding_pool == 'cls' and hasattr(out, 'pooler_output') and out.pooler_output is not None:
                            emb = out.pooler_output
                        else:
                            last = out.last_hidden_state
                            mask = attention_mask.unsqueeze(-1).type_as(last)
                            summed = (last * mask).sum(1)
                            counts = mask.sum(1).clamp(min=1e-9)
                            emb = summed / counts
                        logits = clf.head(emb)
                        loss = criterion(logits, labels)
                    v_loss += loss.item() * input_ids.size(0)
                    v_total += input_ids.size(0)
            v_loss = v_loss / max(1, v_total)
            finetune_val_losses.append(v_loss)

            print(f"Finetune Epoch {epoch+1}/{args.finetune_epochs} | train_loss={ep_loss:.4f} | val_loss={v_loss:.4f}")

            # save checkpoint for this epoch (model + optimizer + scaler)
            ckpt = {
                'epoch': epoch+1,
                'bert_state_dict': clf.bert.state_dict(),
                'head_state_dict': clf.head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'label_list': label_list
            }
            if scaler is not None:
                ckpt['scaler_state_dict'] = scaler.state_dict()
            torch.save(ckpt, finetune_out / f'finetune_epoch_{epoch+1}.pth')

        # append finetune history to main history for combined plotting
        history['train_loss'].extend(finetune_train_losses)
        history['val_loss'].extend(finetune_val_losses)

        # save final finetune checkpoint
        final_ckpt = {
            'epoch': args.finetune_epochs,
            'bert_state_dict': clf.bert.state_dict(),
            'head_state_dict': clf.head.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'label_list': label_list
        }
        if scaler is not None:
            final_ckpt['scaler_state_dict'] = scaler.state_dict()
        torch.save(final_ckpt, finetune_out / 'finetune_final.pth')
        print('Saved finetune checkpoint to', finetune_out / 'finetune_final.pth')

        # update and save metrics
        metrics['train_loss_per_epoch'] = history['train_loss']
        metrics['val_loss_per_epoch'] = history['val_loss']
        with open(finetune_out / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        # save combined loss plot
        try:
            plt.figure()
            plt.plot(range(1, len(history['train_loss'])+1), history['train_loss'], label='train')
            plt.plot(range(1, len(history['val_loss'])+1), history['val_loss'], label='val')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend()
            plt.tight_layout()
            plt.savefig(finetune_out / 'loss_curve.png')
            print('Saved finetune loss plot to', finetune_out / 'loss_curve.png')
        except Exception as e:
            print('Could not save finetune plot:', e)

        # sample predictions on validation using finetuned model
        clf.bert.eval(); clf.head.eval()
        sample_count = min(5, len(val_samples))
        for i in range(sample_count):
            sample = val_samples[i]
            enc = clf.tokenizer(sample[0], truncation=True, padding='max_length', max_length=512, return_tensors='pt')
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad(), autocast(enabled=use_amp):
                out = clf.bert(**enc, return_dict=True)
                if clf.embedding_pool == 'cls' and hasattr(out, 'pooler_output') and out.pooler_output is not None:
                    emb = out.pooler_output
                else:
                    last = out.last_hidden_state
                    mask = enc['attention_mask'].unsqueeze(-1).type_as(last)
                    summed = (last * mask).sum(1)
                    counts = mask.sum(1).clamp(min=1e-9)
                    emb = summed / counts
                logits = clf.head(emb)
                probs = torch.sigmoid(logits).cpu().numpy()[0]
            top_idx = np.argsort(-probs)[:5]
            top_preds = [(label_list[j], float(probs[j])) for j in top_idx]
            print(f"Finetune Val sample {i+1}: top_preds={top_preds} | true_labels={sample[1]}")

        print('Finetune phase complete. Check', finetune_out)

if __name__ == '__main__':
    main()
