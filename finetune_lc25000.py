"""
Fine-tune DenseNet121 (pretrained on CASIA2) on LC25000 medical images.
"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support

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

def load_casia_pretrained(model, checkpoint_path):
    """Load weights from CASIA2 pretraining"""
    print(f"Loading CASIA2 pretrained weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Remove classifier weights from checkpoint since we have a new classifier
    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    
    for k, v in state_dict.items():
        if not k.startswith('classifier'):
            new_state_dict[k] = v
            
    # Load only the feature extractor weights
    model.load_state_dict(new_state_dict, strict=False)
    print("Successfully loaded CASIA2 feature extractor weights")
    return model

# -------------------------
# Model builder
# -------------------------
def build_densenet121(num_classes=5, pretrained=True):
    """
    Builds DenseNet121 and adapts classifier head for LC25000 classes.
    Note: num_classes=5 for LC25000 medical image types
    """
    model = models.densenet121(pretrained=False)  # We'll load CASIA weights instead
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),  # Add dropout for fine-tuning
        nn.Linear(in_features, num_classes)
    )
    return model

# -------------------------
# Training & Validation
# -------------------------
def train_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

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
        
        preds = torch.argmax(outputs, dim=1)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

        pbar.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')
    
    return epoch_loss, accuracy, precision, recall, f1

def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Val", leave=False)
        for images, targets in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')
    
    return epoch_loss, accuracy, precision, recall, f1

# -------------------------
# Main
# -------------------------
def main(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Transforms - using similar augmentations as CASIA training
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

    # Dataset
    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "test")  # LC25000 uses test instead of val

    train_ds = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_ds = datasets.ImageFolder(val_dir, transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                            num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size*2, shuffle=False, 
                          num_workers=args.num_workers, pin_memory=True)

    print("Classes:", train_ds.classes)
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")

    # Model
    model = build_densenet121(num_classes=len(train_ds.classes), pretrained=False)
    model = load_casia_pretrained(model, args.casia_ckpt)
    model = model.to(device)

    # Loss / Optimizer / Scheduler
    criterion = nn.CrossEntropyLoss()
    
    # Use different learning rates for pretrained layers and new classifier
    classifier_params = model.classifier.parameters()
    pretrained_params = [p for n, p in model.named_parameters() if not n.startswith('classifier')]
    
    optimizer = optim.AdamW([
        {'params': pretrained_params, 'lr': args.lr * 0.1},  # Lower LR for pretrained weights
        {'params': classifier_params, 'lr': args.lr}  # Higher LR for new classifier
    ], weight_decay=args.weight_decay)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()

    # Training loop
    best_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler)
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        
        val_loss, val_acc, val_prec, val_rec, val_f1 = validate_epoch(
            model, val_loader, criterion, device)
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

        # Scheduler step
        scheduler.step()

        # Checkpointing
        is_best = val_f1 > best_f1
        if is_best:
            best_f1 = val_f1

        ckpt_name = f"epoch_{epoch:03d}.pth.tar"
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_f1': best_f1,
            'classes': train_ds.classes
        }, args.output_dir, filename=ckpt_name)

        if is_best:
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_f1': best_f1,
                'classes': train_ds.classes
            }, args.output_dir, filename="best.pth.tar")
            print(f"New best model saved with Val F1: {best_f1:.4f}")

    print("Training finished. Best val F1:", best_f1)

# -------------------------
# Argparse
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune DenseNet121 on LC25000")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to LC25000 dataset")
    parser.add_argument("--casia_ckpt", type=str, required=True, help="Path to CASIA2 pretrained checkpoint")
    parser.add_argument("--output_dir", type=str, default="checkpoints/lc25000", help="Where to save checkpoints")
    parser.add_argument("--epochs", type=int, default=20)  # Fewer epochs for fine-tuning
    parser.add_argument("--batch_size", type=int, default=32)  # Can use larger batch size
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)  # Slightly stronger regularization
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(args)