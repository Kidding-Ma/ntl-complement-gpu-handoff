"""Reload a saved adapter and reproduce its first three training features."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ntl.common import read_train, sha256, softmax, write_json
from ntl.checkpoint import check_checkpoint

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--base-model', required=True)
    p.add_argument('--train-csv', required=True)
    p.add_argument('--allow-smoke', action='store_true')
    p.add_argument('--output', required=True, help='Outside checkpoint directory (manifest must stay immutable)')
    args = p.parse_args()
    ck = Path(args.checkpoint)
    if Path(args.output).resolve().is_relative_to(ck.resolve()):
        raise ValueError('Write parity report outside the checkpoint directory')
    meta = check_checkpoint(ck, args.allow_smoke)
    if sha256(args.train_csv) != meta['train_sha256']:
        raise ValueError('Training rows differ from export')
    if sha256(Path(args.base_model) / 'config.json') != meta['base_config_sha256']:
        raise ValueError('Base configuration differs from export')
    import numpy as np
    from ntl.gpu import reload_checkpoint, tokenize, extract
    frame = read_train(args.train_csv)
    model, norms, tok, _ = reload_checkpoint(args.base_model, ck)
    with np.load(ck / 'anchor_features.npz', allow_pickle=False) as saved:
        ids = saved['ids']; expected = saved['features']
    positions = frame.set_index('id').index.get_indexer(ids)
    if (positions < 0).any():
        raise ValueError('Anchor IDs not found')
    tokens = tokenize(tok, frame.text.iloc[positions].tolist(), meta['config'])
    actual = extract(model, norms, tok, tokens, np.arange(len(tokens)), meta['config'])
    rel = float(np.abs(actual - expected).mean() / (np.abs(expected).mean() + 1e-9))
    with np.load(ck / 'readouts.npz', allow_pickle=False) as ro:
        prob = softmax(actual @ ro['plain_coef'].T + ro['plain_intercept'])
    passed = bool(np.isfinite(actual).all() and rel < .02 and np.isfinite(prob).all()
                  and np.allclose(prob.sum(1), 1) and np.all((prob >= 0) & (prob <= 1)))
    result = {'passed': passed, 'mean_relative_feature_error': rel,
              'max_absolute_feature_error': float(np.abs(actual - expected).max()),
              'threshold': .02, 'train_sha256': meta['train_sha256'], 'job': meta['job'],
              'smoke_only': meta['smoke_only'], 'checkpoint_marker_sha256': sha256(
                  ck / ('SMOKE_ONLY.json' if meta['smoke_only'] else 'COMPLETE.json')),
              'probability_shape': list(prob.shape)}
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
