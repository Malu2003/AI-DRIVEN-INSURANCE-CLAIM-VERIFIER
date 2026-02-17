"""
Fine-tune DenseNet121 on LC25000 dataset.
Loads a checkpoint (optional) and continues training while saving checkpoints.
"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm

import numpy as np
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(state, checkpoint_dir, filename="checkpoint.pth.tar"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    torch.save(state, path)


def build_densenet121(num_classes=2, pretrained=True, freeze_until_layer=None):
    model = models.densenet121(pretrained=pretrained)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    if freeze_until_layer:
        for name, param in model.features.named_parameters():
            if freeze_until_layer not in name:
                param.requires_grad = False
    return model


def train_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    all_targets = []
    all_probs = []

    pbar = tqdm(loader, desc="Train", leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

        probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
        all_probs.append(probs)
        all_targets.append(targets.detach().cpu().numpy())

        pbar.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(loader.dataset)
    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets)
    try:
        # Handle binary and multiclass AUC correctly
        if all_probs.shape[1] == 2:
            auc = roc_auc_score(all_targets, all_probs[:, 1])
        else:
            auc = roc_auc_score(all_targets, all_probs, multi_class='ovr', average='macro')
    except Exception:
        auc = float('nan')
    return epoch_loss, auc


def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_probs = []
    with torch.no_grad():
        pbar = tqdm(loader, desc="Val", leave=False)
        for images, targets in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)

            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(targets.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets)
    try:
        if all_probs.shape[1] == 2:
            auc = roc_auc_score(all_targets, all_probs[:, 1])
        else:
            auc = roc_auc_score(all_targets, all_probs, multi_class='ovr', average='macro')
    except Exception:
        auc = float('nan')
    return epoch_loss, auc


def main(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.08, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "test") if os.path.exists(os.path.join(args.data_dir, "test")) else os.path.join(args.data_dir, "val")

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"train directory not found: {train_dir}")

    train_ds = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_ds = datasets.ImageFolder(val_dir, transform=val_transforms) if os.path.exists(val_dir) else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True) if val_ds else None

    print("Classes:", train_ds.classes)
    print("Train samples:", len(train_ds))
    if val_ds:
        print("Val samples:", len(val_ds))

    # Determine number of classes from dataset and build model accordingly
    num_classes = len(train_ds.classes)
    model = build_densenet121(num_classes=num_classes, pretrained=True, freeze_until_layer=args.unfreeze_block)
    model = model.to(device)

    start_epoch = 1
    best_auc = 0.0
    # optionally load checkpoint (partial load to handle classifier size mismatch)
    if args.pretrained_ckpt and os.path.exists(args.pretrained_ckpt):
        ckpt = torch.load(args.pretrained_ckpt, map_location=device)
        pretrained_state = ckpt.get('state_dict', ckpt)
        model_state = model.state_dict()

        # keep only matching parameter shapes (skip classifier if shapes differ)
        matched = {k: v for k, v in pretrained_state.items() if k in model_state and model_state[k].shape == v.shape}
        model_state.update(matched)
        model.load_state_dict(model_state)
        skipped = len(pretrained_state) - len(matched)
        print(f"Loaded pretrained weights from {args.pretrained_ckpt} (skipped {skipped} incompatible keys)")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['state_dict'])
        start_epoch = ckpt.get('epoch', 1) + 1
        best_auc = ckpt.get('best_auc', 0.0)
        print(f"Resumed from {args.resume}, starting epoch {start_epoch}, best_auc={best_auc:.4f}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'='*70}")
        print(f"📊 EPOCH {epoch}/{args.epochs} | Time: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")

        train_loss, train_auc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        print(f"✓ Train loss: {train_loss:.4f} | Train AUC: {train_auc:.4f}")

        val_loss, val_auc = (float('nan'), float('nan'))
        if val_loader:
            val_loss, val_auc = validate_epoch(model, val_loader, criterion, device)
            print(f"✓ Val loss:   {val_loss:.4f} | Val AUC:   {val_auc:.4f}")
        else:
            print("⚠ No validation loader found; skipping validation.")

        scheduler.step()

        is_best = False
        if val_loader and not np.isnan(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            is_best = True

        ckpt_name = f"epoch_{epoch:03d}.pth.tar"
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_auc': best_auc,
        }, args.output_dir, filename=ckpt_name)

        if is_best:
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_auc': best_auc,
            }, args.output_dir, filename="best.pth.tar")
            print(f"🏆 New best model saved with Val AUC: {best_auc:.4f}")

        progress_pct = (epoch / args.epochs) * 100
        print(f"📈 Progress: {progress_pct:.1f}% ({epoch}/{args.epochs})")

    print(f"\n{'='*70}")
    print("🎉 FINE-TUNING COMPLETED")
    print(f"Best val AUC: {best_auc:.4f}")
    print(f"Best model saved: {os.path.join(args.output_dir, 'best.pth.tar')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune DenseNet121 on LC25000')
    parser.add_argument('--data_dir', type=str, default='data/LC25000', help='Path to LC25000 dataset (train/ and test/ or val/)')
    parser.add_argument('--output_dir', type=str, default='checkpoints/lc25000', help='Where to save checkpoints')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--pretrained_ckpt', type=str, default='checkpoints/casia/best.pth.tar', help='Path to pretrained checkpoint to init weights')
    parser.add_argument('--unfreeze_block', type=str, default='denseblock4')
    args = parser.parse_args()
    main(args)
