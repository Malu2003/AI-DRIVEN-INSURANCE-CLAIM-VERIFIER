import os
import torch
from glob import glob

ckpt_dir = os.path.join('checkpoints','casia')
ckpt_paths = sorted(glob(os.path.join(ckpt_dir,'epoch_*.pth.tar')))
ckpt_paths = ckpt_paths[::-1]  # newest first

if not ckpt_paths:
    print('No epoch checkpoints found')
    exit(1)

for p in ckpt_paths:
    try:
        size = os.path.getsize(p)
        print(f'Testing {p} (size={size} bytes) ...', end=' ')
        ck = torch.load(p, map_location='cpu')
        print('OK')
        print('-> Using', p)
        break
    except Exception as e:
        print('FAILED')
        print('  Error:', e)
else:
    print('No valid checkpoints found')
