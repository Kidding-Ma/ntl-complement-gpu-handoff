"""Run exact V25 recipe in fresh local Linux staging, with strict output acceptance."""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def logged(cmd, cwd, path, env):
    with path.open('w', encoding='utf-8') as log:
        process = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1)
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            rc = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
            raise
    if rc:
        raise RuntimeError(f'Process failed ({rc}); inspect {path.name}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root', type=Path, required=True)
    p.add_argument('--work-dir', type=Path, required=True)
    a = p.parse_args()
    if sys.platform != 'linux':
        p.error('Use Linux/WSL2 CUDA. No native Windows GPU execution has been validated.')
    from preflight import inspect_inputs
    from verify_workspace import verify
    from validate_submission import validate, check_components
    verify()
    inp, work = a.input_root.resolve(), a.work_dir.resolve()
    if work.exists() and any(work.iterdir()):
        p.error('--work-dir must be new/empty to prevent stale output acceptance')
    if work == inp or work.is_relative_to(inp) or inp.is_relative_to(work):
        p.error('Input and output roots must not contain each other')
    work.mkdir(parents=True, exist_ok=True)
    status = {'status': 'running', 'started_unix': time.time(), 'command': sys.argv,
              'input_root': str(inp), 'work_dir': str(work), 'source_version': 25}
    stop = threading.Event()

    def monitor():
        with (work / 'gpu-monitor.csv').open('w', encoding='utf-8') as f:
            f.write('timestamp,index,name,memory.used [MiB],utilization.gpu [%]\n')
            while not stop.is_set():
                try:
                    r = subprocess.run(['nvidia-smi', '--query-gpu=timestamp,index,name,memory.used,utilization.gpu', '--format=csv,noheader'], capture_output=True, text=True, timeout=10)
                    f.write(r.stdout)
                    f.flush()
                except (OSError, subprocess.TimeoutExpired) as e:
                    f.write('# monitor error: ' + type(e).__name__ + '\n')
                stop.wait(10)
    thread = None
    try:
        report = inspect_inputs(inp, gpu=True)
        (work / 'preflight.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        if not report['ok']:
            raise RuntimeError('Preflight failed: ' + '; '.join(report['errors']))
        env = {**os.environ, 'NTL_DROP_Q35': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'PYTHONUNBUFFERED': '1'}
        stage = work / 'stage'
        stage.mkdir()
        def transport(s):
            return s.replace('/kaggle/input', inp.as_posix()).replace('/kaggle/working', work.as_posix())
        for name in ('gauss.py', 'infer.py', 'comp_stack.py'):
            (stage / name).write_text(transport((ROOT / 'src' / name).read_text(encoding='utf-8')), encoding='utf-8')
        setup = transport((ROOT / 'references/upstream/cell_01.py').read_text(encoding='utf-8'))
        setup = setup.replace('capture_output=True, text=True).returncode)', 'capture_output=True, text=True, check=True).returncode)')
        # Preserve model configuration; convert shell launching into a separately logged command.
        final = (ROOT / 'references/upstream/cell_06.py').read_text(encoding='utf-8')
        model_config = transport(final.split('!python infer.py', 1)[0])
        append = '\nfor _x in whl:\n    subprocess.run([sys.executable,"-m","pip","show","bitsandbytes"],check=True)\n'
        append += '\n' + model_config + '\njson.dump({"DATA":DATA,"MODELS":MODELS,"EXTRA":EXTRA,"POOL":POOL,"ATLAS":ATLAS,"B8":B8},open("runtime-config.json","w"),indent=2)\n'
        (stage / 'setup.py').write_text(setup + append.replace('ATLAS_BUDGET = int(8.3 * 3600 - COMP_SEC - 600)', ''), encoding='utf-8')
        for name, cmd in [('pip-freeze.txt', [sys.executable, '-m', 'pip', 'freeze']), ('nvidia-smi-before.txt', ['nvidia-smi'])]:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            (work / name).write_text(r.stdout + r.stderr, encoding='utf-8')
        (work / 'environment.json').write_text(json.dumps({'python': sys.version, 'platform': sys.platform,
                     'source_container': json.loads((ROOT / 'references/upstream/metadata.json').read_text())['dockerImage'],
                     'actual_container': os.environ.get('CONTAINER_IMAGE', 'unknown'), 'NTL_DROP_Q35': '1', 'offline': True}, indent=2), encoding='utf-8')
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        logged([sys.executable, str(stage / 'setup.py')], work, work / 'setup.log', env)
        cfg = json.loads((work / 'runtime-config.json').read_text(encoding='utf-8'))
        t0 = time.time()
        logged([sys.executable, str(stage / 'comp_stack.py')], work, work / 'complement.log', env)
        comp_seconds = time.time() - t0
        budget = int(8.3 * 3600 - comp_seconds - 600)
        if budget <= 0:
            raise RuntimeError('Complement exhausted the original total time budget')
        cmd = [sys.executable, str(stage / 'infer.py'), '--data', cfg['DATA'], '--models', json.dumps(cfg['MODELS']),
               '--perpassage', '1', '--extra', cfg['EXTRA'], '--extra_w', '0.5', '--tfidf_w', '0', '--distort_w', '0',
               '--svc_w', '1.8', '--pool', cfg['POOL'], '--adapt_w', '1.0', '--adapt_thr', '0.15', '--sinkhorn', '1.0',
               '--post', '1', '--atlas_dir', cfg['ATLAS'], '--atlas_base', cfg['B8'], '--atlas_w', '0.16', '--atlas_margin', '0.38',
               '--atlas_bs', '4', '--atlas_chunk', '3', '--atlas_budget', str(budget), '--comp_path', str(work / 'comp_logit.npy'),
               '--comp_w', '0.15', '--out', str(work / 'submission.parquet'), '--cache', str(work / 'cache')]
        status['inference_command'], status['complement_seconds'], status['atlas_budget'] = cmd, comp_seconds, budget
        logged(cmd, work, work / 'inference.log', env)
        test = Path(cfg['DATA']) / 'test.csv'
        status['components'] = check_components((work / 'complement.log').read_text(encoding='utf-8'), (work / 'inference.log').read_text(encoding='utf-8'), work, test)
        valid = validate(test, work / 'submission.parquet')
        (work / 'submission-check.json').write_text(json.dumps(valid, indent=2), encoding='utf-8')
        import pandas as pd
        pd.read_parquet(work / 'submission.parquet').to_csv(work / 'submission.csv', index=False)
        status['status'] = 'visible_run_passed_hidden_score_unverified'
    except BaseException as e:
        status.update(status='failed', error=type(e).__name__ + ': ' + str(e), traceback=traceback.format_exc())
        raise
    finally:
        stop.set()
        if thread:
            thread.join(timeout=15)
        try:
            r = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=15)
            (work / 'nvidia-smi-after.txt').write_text(r.stdout + r.stderr, encoding='utf-8')
        except (OSError, subprocess.TimeoutExpired):
            pass
        status['finished_unix'] = time.time()
        status['seconds'] = status['finished_unix'] - status['started_unix']
        (work / 'run-status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
