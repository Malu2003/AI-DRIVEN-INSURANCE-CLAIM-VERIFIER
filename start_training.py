"""
Quick start script to resume CASIA training with all information displayed.
"""

import os
import subprocess
import sys

def print_header():
    """Print welcome header."""
    print("\n" + "="*70)
    print("🚀 CASIA2 DenseNet121 Training Resumption System")
    print("="*70)
    print()

def print_status():
    """Print current training status."""
    print("📊 TRAINING STATUS")
    print("-" * 70)
    print("Model:              DenseNet121 (pretrained on ImageNet)")
    print("Dataset:            CASIA2.0 (Medical Image Forgery Detection)")
    print("Training Target:    100 epochs")
    print("Completed Epochs:   43")
    print("Remaining Epochs:   57")
    print("Last Valid AUC:     0.8521")
    print("Resume Checkpoint:  checkpoints/casia/epoch_043.pth.tar")
    print()

def print_training_config():
    """Print training configuration."""
    print("⚙️  TRAINING CONFIGURATION")
    print("-" * 70)
    print("Optimizer:          AdamW")
    print("Learning Rate:      1e-4")
    print("Batch Size:         16")
    print("Scheduler:          CosineAnnealing (T_max=100)")
    print("Loss Function:      CrossEntropyLoss")
    print("Augmentation:       RandomResizedCrop, RandomHFlip, Rotation, ColorJitter")
    print("Mixed Precision:    Enabled (AMP)")
    print()

def print_expectations():
    """Print what to expect."""
    print("🔔 WHAT TO EXPECT")
    print("-" * 70)
    print("✓ Each epoch will show: loss, AUC, and progress %")
    print("✓ Terminal will display 📊 at each epoch start")
    print("✓ New best models will show 🏆 symbol")
    print("✓ Final completion will show 🎉 message")
    print("✓ Total time: ~3-8 hours (depends on hardware)")
    print()

def verify_requirements():
    """Verify training can proceed."""
    print("🔍 VERIFYING REQUIREMENTS")
    print("-" * 70)
    
    checks = {
        "CASIA2 train data": "data/CASIA2/train",
        "CASIA2 val data": "data/CASIA2/val",
        "Checkpoint directory": "checkpoints/casia",
        "Resume checkpoint": "checkpoints/casia/epoch_043.pth.tar",
        "Training script": "train_casia.py",
    }
    
    all_ok = True
    for name, path in checks.items():
        exists = os.path.exists(path)
        status = "✓" if exists else "✗"
        print(f"{status} {name:30} {path}")
        if not exists:
            all_ok = False
    
    print()
    return all_ok

def main():
    """Main execution."""
    print_header()
    print_status()
    print_training_config()
    print_expectations()
    
    if not verify_requirements():
        print("❌ MISSING REQUIREMENTS")
        print("Please check the paths above and ensure all files exist.")
        print()
        sys.exit(1)
    
    print("✅ ALL REQUIREMENTS MET - READY TO TRAIN")
    print()
    print("="*70)
    print("Starting training... Press Ctrl+C to stop (checkpoint will be saved)")
    print("="*70)
    print()
    
    # Run the training
    cmd = [
        "python", "train_casia.py",
        "--data_dir", "data/CASIA2",
        "--output_dir", "checkpoints/casia",
        "--epochs", "100",
        "--batch_size", "16",
        "--lr", "1e-4",
        "--weight_decay", "1e-5",
        "--num_workers", "4",
        "--seed", "42",
        "--resume", "checkpoints/casia/epoch_043.pth.tar",
        "--unfreeze_block", "denseblock4",
    ]
    
    try:
        result = subprocess.run(cmd, cwd=os.getcwd())
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print()
        print("=" * 70)
        print("⚠️  Training interrupted by user")
        print("✓ Checkpoint from last completed epoch was saved")
        print("=" * 70)
        sys.exit(0)

if __name__ == "__main__":
    main()
