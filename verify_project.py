import os

required_paths = [
    'train_casia.py',
    'train_lc25000.py',
    'run_training.ps1',
    'run_finetune.ps1',
    'checkpoints/casia',
    'data/CASIA2',
    'data/LC25000',
]

print('Project verification report:')
ok = True
for p in required_paths:
    exists = os.path.exists(p)
    print(f" - {'OK' if exists else 'MISSING'}: {p}")
    if not exists:
        ok = False

# list checkpoint folders if present
if os.path.exists('checkpoints/casia'):
    files = sorted(os.listdir('checkpoints/casia'))
    print('\ncheckpoints/casia (last 10):')
    for f in files[-10:]:
        print('   ', f)

if os.path.exists('checkpoints/lc25000'):
    files = sorted(os.listdir('checkpoints/lc25000'))
    print('\ncheckpoints/lc25000 (last 10):')
    for f in files[-10:]:
        print('   ', f)

print('\nProject structure looks {}.'.format('GOOD' if ok else 'INCOMPLETE'))
