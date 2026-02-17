"""
Check CASIA training progress and find the latest checkpoint to resume from.
"""

import os
import torch
import re
from pathlib import Path

def get_latest_checkpoint(checkpoint_dir="checkpoints/casia"):
    """Find the latest checkpoint in the checkpoint directory."""
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return None, None
    
    # Find all epoch checkpoints
    checkpoints = []
    for file in os.listdir(checkpoint_dir):
        if file.startswith("epoch_") and file.endswith(".pth.tar"):
            match = re.search(r"epoch_(\d+)", file)
            if match:
                epoch = int(match.group(1))
                checkpoints.append((epoch, file))
    
    if not checkpoints:
        print(f"⚠️  No epoch checkpoints found in {checkpoint_dir}")
        return None, None
    
    # Sort by epoch number and get the latest
    checkpoints.sort(key=lambda x: x[0])
    latest_epoch, latest_file = checkpoints[-1]
    
    return latest_epoch, os.path.join(checkpoint_dir, latest_file)

def check_checkpoint_integrity(checkpoint_path):
    """Verify that the checkpoint is not corrupted and contains required keys."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        required_keys = ['state_dict', 'epoch', 'optimizer']
        missing_keys = [k for k in required_keys if k not in checkpoint]
        
        if missing_keys:
            print(f"⚠️  Checkpoint missing keys: {missing_keys}")
            return False, checkpoint
        
        print(f"✓ Checkpoint is valid")
        print(f"  - Epoch: {checkpoint.get('epoch', 'Unknown')}")
        print(f"  - Best AUC: {checkpoint.get('best_auc', 'Unknown'):.4f}" if isinstance(checkpoint.get('best_auc'), float) else f"  - Best AUC: {checkpoint.get('best_auc', 'Unknown')}")
        print(f"  - Model params: {len(checkpoint['state_dict'])}")
        return True, checkpoint
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return False, None

def main():
    print("=" * 70)
    print("🔍 CASIA Training Progress Check")
    print("=" * 70)
    print()
    
    checkpoint_dir = "checkpoints/casia"
    latest_epoch, checkpoint_path = get_latest_checkpoint(checkpoint_dir)
    
    if checkpoint_path is None:
        print("No checkpoints found to resume from.")
        print("Training will start from epoch 1.")
        return
    
    print(f"📁 Latest checkpoint found:")
    print(f"   Epoch: {latest_epoch}")
    print(f"   Path: {checkpoint_path}")
    print()
    
    # Check if corrupt file exists
    corrupt_file = f"{checkpoint_path}.corrupt"
    if os.path.exists(corrupt_file):
        print(f"⚠️  Corrupt file detected: {corrupt_file}")
        print(f"   This checkpoint may have failed to save. Will attempt to use the last valid checkpoint.")
        print()
    
    print("Verifying checkpoint integrity...")
    is_valid, checkpoint_data = check_checkpoint_integrity(checkpoint_path)
    
    if not is_valid:
        print(f"❌ Checkpoint is corrupted or invalid!")
        # Try to find previous checkpoint
        if latest_epoch > 1:
            print(f"Attempting to use checkpoint from epoch {latest_epoch - 1}...")
            prev_checkpoint = os.path.join(checkpoint_dir, f"epoch_{latest_epoch-1:03d}.pth.tar")
            if os.path.exists(prev_checkpoint):
                is_valid, checkpoint_data = check_checkpoint_integrity(prev_checkpoint)
                if is_valid:
                    checkpoint_path = prev_checkpoint
                    latest_epoch = latest_epoch - 1
        
        if not is_valid:
            print("No valid checkpoint found. You may need to start fresh.")
            return
    
    print()
    print("=" * 70)
    print(f"✅ Ready to resume training from epoch {latest_epoch + 1}/100")
    print(f"   Remaining epochs: {100 - latest_epoch}")
    print("=" * 70)
    print()
    print("To resume training, run:")
    print(f"python train_casia.py --data_dir data/CASIA2 --output_dir {checkpoint_dir} --epochs 100 --batch_size 16 --resume \"{checkpoint_path}\"")
    print()

if __name__ == "__main__":
    main()
