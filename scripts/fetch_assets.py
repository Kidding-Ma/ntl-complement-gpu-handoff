"""List/download declared inputs using the installed official Kaggle CLI.

No model source or training notebook is executed. No credentials are embedded.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def command(e, target):
    ref, kind, version = e['ref'], e['kind'], e.get('version')
    if kind in ('dataset', 'kernel') and version:
        ref += '/' + str(version)
    if kind == 'dataset':
        return ['kaggle', 'datasets', 'download', ref, '-p', str(target), '--unzip']
    if kind == 'kernel':
        return ['kaggle', 'kernels', 'output', ref, '-p', str(target)]
    if kind == 'model':
        return ['kaggle', 'models', 'instances', 'versions', 'download', ref, '-p', str(target), '--untar']
    return ['kaggle', 'competitions', 'download', '-c', ref, '-p', str(target)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=ROOT / 'config/inputs.json')
    p.add_argument('--input-root', type=Path, default=ROOT / 'input')
    p.add_argument('--list', action='store_true')
    p.add_argument('--download', action='store_true')
    p.add_argument('--allow-unpinned', action='store_true')
    p.add_argument('--only', help='Exact ref from the manifest')
    a = p.parse_args()
    entries = json.loads(a.manifest.read_text(encoding='utf-8'))['inputs']
    if a.only:
        entries = [e for e in entries if e['ref'] == a.only]
        if not entries:
            p.error('--only is not a declared ref')
    base = a.input_root.resolve()
    for e in entries:
        print(e['kind'], e['ref'], 'version=' + str(e.get('version')), '->', e['target'])
    if not a.download:
        return
    unresolved = [e['ref'] for e in entries if e['kind'] in ('dataset', 'kernel') and not e.get('version')]
    if unresolved and not a.allow_unpinned:
        p.error('Resolve attachment versions or explicitly use --allow-unpinned: ' + ', '.join(unresolved))
    if not shutil.which('kaggle'):
        p.error('Install official Kaggle CLI and authenticate on this machine first')
    base.mkdir(parents=True, exist_ok=True)
    report_path = base / 'download-report.json'
    report = json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else []
    failed = False
    for e in entries:
        dest = (base / e['target']).resolve()
        if not dest.is_relative_to(base):
            raise ValueError('Unsafe target')
        dest.mkdir(parents=True, exist_ok=True)
        row = {**e, 'started_unix': time.time(), 'command': command(e, dest), 'target_absolute': str(dest)}
        rc = subprocess.call(row['command'])
        row['exit_code'] = rc
        if rc == 0 and e['kind'] == 'competition':
            for z in dest.glob('*.zip'):
                with zipfile.ZipFile(z) as archive:
                    for member in archive.infolist():
                        if not (dest / member.filename).resolve().is_relative_to(dest):
                            raise ValueError('Unsafe archive path')
                    archive.extractall(dest)
        files = [f for f in dest.rglob('*') if f.is_file()]
        row['files'] = []
        for f in files:
            h = hashlib.sha256()
            with f.open('rb') as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    h.update(chunk)
            row['files'].append({'path': f.relative_to(base).as_posix(), 'bytes': f.stat().st_size, 'sha256': h.hexdigest()})
        row['completed_unix'] = time.time()
        row['ok'] = rc == 0 and bool(files)
        report.append(row)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        if not row['ok']:
            failed = True
            print('FAILED/EMPTY:', e['ref'], file=sys.stderr)
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
