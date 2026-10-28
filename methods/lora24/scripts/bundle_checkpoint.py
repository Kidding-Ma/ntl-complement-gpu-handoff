"""Create a weights-only zip for private transfer / Kaggle Dataset upload."""
import argparse
import sys
import zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ntl.checkpoint import check_checkpoint
from ntl.common import sha256, write_json

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    ck = Path(args.checkpoint)
    meta = check_checkpoint(ck)
    if meta['job'] != 'final':
        raise ValueError('Package the full-data final checkpoint')
    out = Path(args.output)
    if out.exists() or out.resolve().is_relative_to(ck.resolve()):
        raise ValueError('Choose a fresh zip path outside the checkpoint directory')
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(ck.rglob('*')):
            if path.is_file():
                archive.write(path, 'own_lora24/final/' + path.relative_to(ck).as_posix())
    write_json(str(out) + '.sha256.json', {'sha256': sha256(out),
        'layout': 'own_lora24/final', 'identity': meta['identity']})
    print(f'Weights transfer archive (do not commit to GitHub): {out}')

if __name__ == '__main__':
    main()
