"""
Train DenseNet121 on CASIA2.0 (pretrain stage).
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

# -------------------------
# Utils
# -------------------------
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

# -------------------------
# Model builder
# -------------------------
def build_densenet121(num_classes=2, pretrained=True, freeze_until_layer=None):
    """
    Builds DenseNet121 and replaces classifier head.
    If freeze_until_layer is a substring (e.g., 'denseblock4'), layers not containing that substring will be frozen.
    """
    model = models.densenet121(pretrained=pretrained)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    if freeze_until_layer:
        for name, param in model.features.named_parameters():
            if freeze_until_layer not in name:
                param.requires_grad = False
    return model

# -------------------------
# Training & Validation
# -------------------------
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

        probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
        all_probs.append(probs)
        all_targets.append(targets.detach().cpu().numpy())

        pbar.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(loader.dataset)
    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        auc = float('nan')  # happened if only one class present in batch/val
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

            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_probs.append(probs)
            all_targets.append(targets.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        auc = float('nan')
    return epoch_loss, auc

# -------------------------
# Main
# -------------------------
def main(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Transforms
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

    # Datasets - expecting ImageFolder structure
    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"train directory not found: {train_dir}")
    if not os.path.exists(val_dir):
        print("WARNING: val dir not found. Using a portion of train as val split if you want, create 'val' folder under CASIA2 structure.")

    train_ds = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_ds = datasets.ImageFolder(val_dir, transform=val_transforms) if os.path.exists(val_dir) else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True) if val_ds else None

    print("Classes:", train_ds.classes)
    print("Train samples:", len(train_ds))
    if val_ds:
        print("Val samples:", len(val_ds))

    # Model
    model = build_densenet121(num_classes=2, pretrained=True, freeze_until_layer=args.unfreeze_block)
    model = model.to(device)

    # Optionally resume
    start_epoch = 1
    best_auc = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['state_dict'])
        start_epoch = ckpt.get('epoch', 1) + 1
        best_auc = ckpt.get('best_auc', 0.0)
        print(f"Resumed from {args.resume}, starting epoch {start_epoch}, best_auc={best_auc:.4f}")

    # Loss / Optimizer / Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = GradScaler()

    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_auc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        print(f"Train loss: {train_loss:.4f} | Train AUC: {train_auc:.4f}")

        val_loss, val_auc = (float('nan'), float('nan'))
        if val_loader:
            val_loss, val_auc = validate_epoch(model, val_loader, criterion, device)
            print(f"Val loss:   {val_loss:.4f} | Val AUC:   {val_auc:.4f}")
        else:
            print("No validation loader found; skipping validation.")

        # Scheduler step
        scheduler.step()

        # Checkpointing
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
            print(f"New best model saved with Val AUC: {best_auc:.4f}")

    print("Training finished. Best val AUC:", best_auc)

# -------------------------
# Argparse
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DenseNet121 CASIA Pretrain")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to CASIA dataset (expects 'train' and 'val' subfolders)")
    parser.add_argument("--output_dir", type=str, default="checkpoints/casia", help="Where to save checkpoints")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--unfreeze_block", type=str, default="denseblock4", help="Substring of layer names to keep unfrozen; others will be frozen")
    args = parser.parse_args()

    main(args)