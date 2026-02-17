import torch, os
p='checkpoints/casia/best.pth.tar'
print('Files present (last 10):')
files=sorted(os.listdir('checkpoints/casia'))
for f in files[-10:]:
    print(' ',f)
if os.path.exists(p):
    ck=torch.load(p,map_location='cpu')
    print('\nBest checkpoint:')
    print(' epoch ->', ck.get('epoch'))
    print(' best_auc ->', ck.get('best_auc'))
else:
    print('\nNo best.pth.tar found')
