"""CPU-only immutable source and notebook structure check."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify():
    lock = json.loads((ROOT / 'config/source-lock.json').read_text(encoding='utf-8'))
    failures = []
    for name, expected in lock['files'].items():
        p = ROOT / name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != expected:
            failures.append(name)
    for p in list((ROOT / 'scripts').glob('*.py')) + list((ROOT / 'src').glob('*.py')):
        ast.parse(p.read_text(encoding='utf-8'), filename=str(p))
    for p in (ROOT / 'notebooks').glob('*.ipynb'):
        nb = json.loads(p.read_text(encoding='utf-8'))
        if nb.get('nbformat') != 4 or not nb.get('cells'):
            failures.append(str(p))
        for c in nb['cells']:
            if c['cell_type'] == 'code':
                ast.parse(''.join(c['source']), filename=str(p))
    if failures:
        raise RuntimeError('Immutable source mismatch: ' + ', '.join(failures))
    print(f'PASS: {len(lock["files"])} source hashes; Python syntax and generated notebook structure. No GPU executed.')


if __name__ == '__main__':
    verify()
