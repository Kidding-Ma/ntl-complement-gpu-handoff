"""Schedule independent fold/final jobs, one model per visible GPU."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ntl.common import sha256
from ntl.checkpoint import check_checkpoint

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True)
    p.add_argument('--output-root', required=True)
    p.add_argument('--devices', default='0', help='Indices within the current CUDA_VISIBLE_DEVICES namespace')
    p.add_argument('--jobs', default='fold0,fold1,fold2,fold3,fold4,final')
    p.add_argument('--skip-complete', action='store_true', help='Verify matching complete checkpoints before skipping')
    args = p.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    devices = [d.strip() for d in args.devices.split(',')]
    if not devices or len(set(devices)) != len(devices) or not all(d.isdigit() for d in devices):
        p.error('Provide unique nonnegative GPU indices')
    outer = os.environ.get('CUDA_VISIBLE_DEVICES')
    mapping = outer.split(',') if outer is not None else None
    if mapping is not None:
        if any(int(d) >= len(mapping) for d in devices):
            p.error('Requested device is outside CUDA_VISIBLE_DEVICES')
        devices = [mapping[int(d)].strip() for d in devices]
    jobs = [j.strip() for j in args.jobs.split(',')]
    if len(set(jobs)) != len(jobs) or any(j not in plan['jobs'] for j in jobs):
        p.error('Unknown or duplicate jobs')
    root = Path(args.output_root).resolve(); root.mkdir(parents=True, exist_ok=True)
    pending = []
    for job in jobs:
        ck = root / job
        if ck.exists():
            if not args.skip_complete:
                raise FileExistsError(f'{ck}: use a fresh output root, or explicitly verify/skip complete jobs')
            meta = check_checkpoint(ck)
            if meta['plan_sha256'] != sha256(args.plan) or meta['job'] != job:
                raise ValueError('Existing checkpoint belongs to a different plan/job')
            print(f'Verified existing checkpoint: {job}', flush=True)
        else:
            pending.append(job)
    running = {}
    try:
        while pending or running:
            for device in devices:
                if device in running or not pending:
                    continue
                job = pending.pop(0)
                log = (root / f'{job}.log').open('w', encoding='utf-8')
                env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = device
                command = [sys.executable, str(Path(__file__).with_name('train.py')),
                           '--plan', str(Path(args.plan).resolve()), '--job', job, '--output', str(root / job)]
                proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                running[device] = (proc, log, job)
                print(f'Started {job} on visible GPU {device}: {root / (job + ".log")}', flush=True)
            for device, (proc, log, job) in list(running.items()):
                if proc.poll() is None:
                    continue
                log.close(); del running[device]
                if proc.returncode:
                    tail = (root / f'{job}.log').read_text(errors='replace')[-5000:]
                    raise RuntimeError(f'{job} failed ({proc.returncode}):\n{tail}')
                check_checkpoint(root / job)
                print(f'Complete: {job}', flush=True)
            if running:
                time.sleep(2)
    finally:
        for proc, log, job in running.values():
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill(); proc.wait()
            log.close()

if __name__ == '__main__':
    main()
