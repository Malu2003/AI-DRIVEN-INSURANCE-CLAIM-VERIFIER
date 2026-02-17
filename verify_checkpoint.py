import torch

ckpt = torch.load('checkpoints/casia/epoch_043.pth.tar', map_location='cpu')
print('✓ Checkpoint loaded successfully')
print(f'  Epoch: {ckpt["epoch"]}')
print(f'  Best AUC: {ckpt["best_auc"]:.4f}')
print(f'  Keys: {list(ckpt.keys())}')
print(f'  Model params: {len(ckpt["state_dict"])} tensors')
