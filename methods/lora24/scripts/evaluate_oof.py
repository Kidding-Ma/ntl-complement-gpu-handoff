"""Evaluate prompt-disjoint plain-head predictions; never fit on validation labels."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ntl.checkpoint import check_checkpoint
from ntl.common import LABELS, balanced_accuracy, read_train, sha256, softmax, write_json

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True)
    p.add_argument('--checkpoints', required=True, help='Parent directory containing fold0..fold4')
    p.add_argument('--output', default='artifacts/oof_report.json')
    args = p.parse_args()
    import numpy as np
    plan = json.loads(Path(args.plan).read_text())
    if sha256(plan['train_csv']) != plan['train_sha256']:
        raise ValueError('Changed train data')
    frame = read_train(plan['train_csv'])
    pred = np.empty(len(frame), int); coverage = np.zeros(len(frame), int); audits = []
    plan_hash = sha256(args.plan)
    for name, indices in plan['jobs'].items():
        if name == 'final':
            continue
        ck = Path(args.checkpoints) / name
        meta = check_checkpoint(ck)
        if meta['job'] != name or meta['plan_sha256'] != plan_hash:
            raise ValueError('Checkpoint was fitted with a different plan/job')
        rows = np.asarray(indices['valid'], int)
        with np.load(ck / 'validation_features.npz', allow_pickle=False) as bank:
            if not np.array_equal(bank['rows'], rows) or bank['ids'].tolist() != frame.id.iloc[rows].tolist():
                raise ValueError('Validation row identity/order mismatch')
            features = bank['features']
        with np.load(ck / 'readouts.npz', allow_pickle=False) as ro:
            logits = features @ ro['plain_coef'].T + ro['plain_intercept']
        if not np.isfinite(logits).all():
            raise ValueError('Nonfinite held-out scores')
        pred[rows] = logits.argmax(1); coverage[rows] += 1
        y = np.searchsorted(LABELS, frame.model.iloc[rows].to_numpy())
        audits.append({'job': name, 'rows': len(rows), 'plain_balanced_accuracy': balanced_accuracy(y, pred[rows])})
    if not np.all(coverage == 1):
        raise ValueError('Each training row must be held out exactly once')
    y = np.searchsorted(LABELS, frame.model.to_numpy())
    report = {'status': 'COMPLETE', 'train_sha256': plan['train_sha256'], 'folds': audits,
              'plain_oof_balanced_accuracy': balanced_accuracy(y, pred),
              'scope': 'Own plain head only; no centered-head validation or ensemble tuning performed',
              'leaderboard_score': None}
    write_json(args.output, report)
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
