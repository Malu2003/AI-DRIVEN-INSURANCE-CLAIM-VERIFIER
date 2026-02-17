import os, time
from datetime import datetime

ckpt_dir = 'checkpoints/casia'
files = sorted(os.listdir(ckpt_dir)) if os.path.exists(ckpt_dir) else []

# find epoch files
epochs = []
for f in files:
    if f.startswith('epoch_') and f.endswith('.pth.tar'):
        try:
            n = int(f.split('_')[1].split('.')[0])
            epochs.append((n,f))
        except:
            pass

epochs.sort()
latest_epoch = epochs[-1][0] if epochs else None
latest_epoch_file = epochs[-1][1] if epochs else None

best_file = 'best.pth.tar' if 'best.pth.tar' in files else None

# file modification times
def mtime(path):
    try:
        return os.path.getmtime(path)
    except:
        return None

now = time.time()
recent_threshold = 10*60  # 10 minutes

latest_mtime = mtime(os.path.join(ckpt_dir, latest_epoch_file)) if latest_epoch_file else None
latest_age = now - latest_mtime if latest_mtime else None

best_info = None
if best_file:
    try:
        import torch
        ck = torch.load(os.path.join(ckpt_dir,best_file), map_location='cpu')
        best_info = {'epoch': ck.get('epoch'), 'best_auc': ck.get('best_auc')}
    except Exception as e:
        best_info = {'error': str(e)}

status = {
    'checkpoint_count': len([f for f in files if f.endswith('.pth.tar')]),
    'latest_epoch': latest_epoch,
    'latest_epoch_file': latest_epoch_file,
    'latest_epoch_age_sec': int(latest_age) if latest_age else None,
    'best_checkpoint_exists': bool(best_file),
    'best_info': best_info,
    'corrupt_files': [f for f in files if f.endswith('.corrupt')],
}

# determine completion
status['training_completed'] = (latest_epoch == 100)
status['recent_activity'] = (latest_age is not None and latest_age < recent_threshold)

print('Training status report:')
print('  checkpoint_count:', status['checkpoint_count'])
print('  latest_epoch:', status['latest_epoch'])
print('  latest_epoch_file:', status['latest_epoch_file'])
if status['latest_epoch_age_sec'] is not None:
    print('  latest_epoch_age_sec:', status['latest_epoch_age_sec'])
    print('  latest_epoch_mod_time:', datetime.fromtimestamp(latest_mtime).isoformat())
print('  training_completed:', status['training_completed'])
print('  recent_activity (<10m):', status['recent_activity'])
print('  best_checkpoint_exists:', status['best_checkpoint_exists'])
print('  best_info:', status['best_info'])
if status['corrupt_files']:
    print('  corrupt_files:', status['corrupt_files'])

# ETA estimate if not completed
if not status['training_completed']:
    # attempt to read a simple progress file if exists
    print('\nNote: If training is running, you can check live terminal output.\n')

