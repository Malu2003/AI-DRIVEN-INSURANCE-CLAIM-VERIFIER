"""Evaluate LC25000 checkpoints and pick the true best by validation AUC.

Usage:
    python evaluation/eval_lc25000_checkpoints.py --data_dir data/LC25000 --ckpt_dir checkpoints/lc25000
"""
import os
import glob
import argparse
import torch
import numpy as np
from torchvision import transforms, datasets, models
import torch.nn as nn
from sklearn.metrics import roc_auc_score


def build_densenet121(num_classes=2, pretrained=False):
    m = models.densenet121(pretrained=pretrained)
    in_features = m.classifier.in_features
    m.classifier = nn.Linear(in_features, num_classes)
    return m


def compute_auc_for_ckpt(ckpt_path, data_dir):
    device = torch.device('cpu')

    # prepare data
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    val_dir = os.path.join(data_dir, 'test') if os.path.exists(os.path.join(data_dir, 'test')) else os.path.join(data_dir, 'val')
    val_ds = datasets.ImageFolder(val_dir, transform=val_transforms)
    from torch.utils.data import DataLoader
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    num_classes = len(val_ds.classes)
    model = build_densenet121(num_classes=num_classes, pretrained=False)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get('state_dict', ckpt)
    # load matching weights only
    model_state = model.state_dict()
    matched = {k: v for k, v in state.items() if k in model_state and model_state[k].shape == v.shape}
    model_state.update(matched)
    model.load_state_dict(model_state)
    model.eval()

    all_targets = []
    all_probs = []
    with torch.no_grad():
        for images, targets in val_loader:
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(targets.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets)
    try:
        if all_probs.shape[1] == 2:
            auc = roc_auc_score(all_targets, all_probs[:, 1])
        else:
            auc = roc_auc_score(all_targets, all_probs, multi_class='ovr', average='macro')
    except Exception:
        auc = float('nan')
    return auc


def main(args):
    ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, 'epoch_*.pth.tar')))
    best_auc = -1.0
    best_ckpt = None
    print(f"Evaluating {len(ckpts)} checkpoints in {args.ckpt_dir}")
    for c in ckpts:
        auc = compute_auc_for_ckpt(c, args.data_dir)
        print(os.path.basename(c), 'val_auc=', auc)
        if not np.isnan(auc) and auc > best_auc:
            best_auc = auc
            best_ckpt = c

    if best_ckpt:
        print(f"Best checkpoint: {best_ckpt} with AUC={best_auc}")
        # copy to best.pth.tar
        import shutil
        shutil.copyfile(best_ckpt, os.path.join(args.ckpt_dir, 'best.pth.tar'))
        print('Copied to best.pth.tar')
    else:
        print('No valid best checkpoint found (all AUC NaN)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--ckpt_dir', required=True)
    args = parser.parse_args()
    main(args)
