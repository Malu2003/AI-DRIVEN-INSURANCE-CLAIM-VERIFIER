import os
import time
import re

ckpt_dir = os.path.join('checkpoints','casia')
log_file = 'train_casia_resume.log'

print('Starting checkpoint monitor for', ckpt_dir)
seen = set(os.listdir(ckpt_dir))
print('Currently present:', sorted(seen))

def get_epoch_num(name):
    m = re.match(r'epoch_(\d+)\.pth\.tar', name)
    if m:
        return int(m.group(1))
    return None

last_epoch = max([get_epoch_num(f) or 0 for f in seen])
print('Last known epoch:', last_epoch)

while True:
    try:
        files = set(os.listdir(ckpt_dir))
        new = sorted(files - seen)
        if new:
            print('New files:', new)
            for f in new:
                e = get_epoch_num(f)
                if e:
                    print(f'Found checkpoint epoch {e}: {f}')
                    last_epoch = max(last_epoch, e)
                    if last_epoch >= 100:
                        print('Training reached epoch', last_epoch)
                        # Optionally print last lines of log
                        if os.path.exists(log_file):
                            with open(log_file,'rb') as fh:
                                fh.seek(0,2)
                                size = fh.tell()
                                seek = max(0,size-5000)
                                fh.seek(seek)
                                print('\n--- Last log tail ---')
                                print(fh.read().decode(errors='replace'))
                        print('Monitor exiting')
                        raise SystemExit(0)
        seen = files
        time.sleep(30)
    except KeyboardInterrupt:
        print('Monitor interrupted')
        break
    except Exception as e:
        print('Monitor error:', e)
        time.sleep(30)
