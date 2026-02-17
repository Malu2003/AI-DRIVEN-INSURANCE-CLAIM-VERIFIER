"""
Fine-tune DenseNet121 on LC25000 forgery detection (authentic vs tampered).
Loads CASIA pre-trained checkpoint for transfer learning.
"""
import os
import argparse
from pathlib import Path
from tqdm import tqdm

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

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
    print(f"   Checkpoint saved: {path}")


def build_densenet121(num_classes=2, pretrained=True):
    """Build DenseNet121 for binary forgery detection"""
    model = models.densenet121(pretrained=pretrained)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model


def load_casia_pretrained(model, casia_ckpt_path):
    """Load CASIA pre-trained weights (transfer learning)"""
    if not os.path.exists(casia_ckpt_path):
        print(f"   CASIA checkpoint not found: {casia_ckpt_path}")
        print(f"   Training from ImageNet pre-trained weights")
        return model
    
    print(f"   Loading CASIA pre-trained weights: {casia_ckpt_path}")
    checkpoint = torch.load(casia_ckpt_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    # Load only feature extractor weights (classifier will be random for 2-class)
    model_state = model.state_dict()
    matched = {}
    
    for k, v in state_dict.items():
        if k in model_state and model_state[k].shape == v.shape:
            if not k.startswith('classifier'):  # Skip classifier layer
                matched[k] = v
    
    model_state.update(matched)
    model.load_state_dict(model_state)
    print(f"   Loaded {len(matched)} layers from CASIA checkpoint")
    return model


def train_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []
    all_probs = []
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * inputs.size(0)
        
        # Predictions
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        all_targets.extend(targets.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs[:, 1].detach().cpu().numpy())  # Prob of class 1 (tampered)
        
        pbar.set_postfix(loss=loss.item())
    
    epoch_loss = running_loss / len(loader.dataset)
    accuracy = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='binary', zero_division=0
    )
    auc = roc_auc_score(all_targets, all_probs) if len(np.unique(all_targets)) > 1 else 0.0
    
    return {
        'loss': epoch_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }


def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation", leave=False)
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            
            pbar.set_postfix(loss=loss.item())
    
    epoch_loss = running_loss / len(loader.dataset)
    accuracy = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='binary', zero_division=0
    )
    auc = roc_auc_score(all_targets, all_probs) if len(np.unique(all_targets)) > 1 else 0.0
    
    return {
        'loss': epoch_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune DenseNet121 on LC25000 forgery detection")
    parser.add_argument('--data', type=str, default='data/LC25000_forgery',
                        help='Path to prepared LC25000 forgery dataset')
    parser.add_argument('--casia_ckpt', type=str, default='checkpoints/casia/best.pth.tar',
                        help='Path to CASIA pre-trained checkpoint for transfer learning')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/lc25000_forgery',
                        help='Directory to save checkpoints')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    print("=" * 80)
    print("LC25000 FORGERY DETECTION FINE-TUNING")
    print("=" * 80)
    print(f"Data directory:    {args.data}")
    print(f"CASIA checkpoint:  {args.casia_ckpt}")
    print(f"Output directory:  {args.checkpoint_dir}")
    print(f"Device:            {args.device}")
    print(f"Epochs:            {args.epochs}")
    print(f"Batch size:        {args.batch_size}")
    print(f"Learning rate:     {args.lr}")
    print("=" * 80)
    
    # =====================================================================
    # DATA LOADING
    # =====================================================================
    print("\n[1/4] Loading dataset...")
    
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_ds = datasets.ImageFolder(os.path.join(args.data, 'train'), transform=train_transform)
    val_ds = datasets.ImageFolder(os.path.join(args.data, 'val'), transform=val_transform)
    
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    print(f"   Train samples: {len(train_ds)}")
    print(f"   Val samples:   {len(val_ds)}")
    print(f"   Classes:       {train_ds.classes}")
    
    # =====================================================================
    # MODEL SETUP
    # =====================================================================
    print("\n[2/4] Building model...")
    
    model = build_densenet121(num_classes=2, pretrained=True)
    model = load_casia_pretrained(model, args.casia_ckpt)
    model = model.to(args.device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()
    
    print(f"   Model loaded on {args.device}")
    
    # =====================================================================
    # TRAINING LOOP
    # =====================================================================
    print(f"\n[3/4] Training for {args.epochs} epochs...")
    
    best_f1 = 0.0
    best_epoch = 0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 80)
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, args.device, scaler)
        print(f"Train - Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.4f} | "
              f"F1: {train_metrics['f1']:.4f} | AUC: {train_metrics['auc']:.4f}")
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, args.device)
        print(f"Val   - Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | "
              f"F1: {val_metrics['f1']:.4f} | AUC: {val_metrics['auc']:.4f}")
        
        # Learning rate step
        scheduler.step()
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            save_checkpoint(
                {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                },
                args.checkpoint_dir,
                filename=f"epoch_{epoch:03d}.pth.tar"
            )
        
        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_epoch = epoch
            save_checkpoint(
                {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                },
                args.checkpoint_dir,
                filename="best.pth.tar"
            )
            print(f"   ✓ New best F1: {best_f1:.4f}")
    
    # =====================================================================
    # FINAL SUMMARY
    # =====================================================================
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best epoch:     {best_epoch}")
    print(f"Best val F1:    {best_f1:.4f}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print(f"\nTo use this model, update image_module.py:")
    print(f"  model_ckpt = 'checkpoints/lc25000_forgery/best.pth.tar'")
    print("=" * 80)


if __name__ == "__main__":
    main()
