"""Standalone plain-head control. This is NOT the original multi-model ensemble."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ntl.checkpoint import check_checkpoint
from ntl.common import LABELS, sha256, softmax, write_json

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--base-model', required=True)
    p.add_argument('--test-csv', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--batch-size', type=int, default=3)
    args = p.parse_args()
    import numpy as np
    import pandas as pd
    meta = check_checkpoint(args.checkpoint)
    if meta['job'] != 'final':
        raise ValueError('Use the full-data final checkpoint for submission')
    if sha256(Path(args.base_model) / 'config.json') != meta['base_config_sha256']:
        raise ValueError('Wrong base configuration')
    from ntl.gpu import reload_checkpoint, tokenize, extract
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    test = pd.read_csv(args.test_csv, dtype=str, keep_default_na=False)
    if not {'id', 'text'} <= set(test) or not len(test) or not test.id.is_unique:
        raise ValueError('Invalid test ID/text schema')
    # Empty hidden-test passages remain defined; whitespace produces a real text token.
    texts = test.text.map(lambda s: s if s.strip() else ' ').tolist()
    model, norms, tok, _ = reload_checkpoint(args.base_model, args.checkpoint)
    tokens = tokenize(tok, texts, meta['config'])
    z = extract(model, norms, tok, tokens, np.arange(len(tokens)), meta['config'], args.batch_size)
    with np.load(Path(args.checkpoint) / 'readouts.npz', allow_pickle=False) as ro:
        prob = softmax(z @ ro['plain_coef'].T + ro['plain_intercept'])
    if not np.isfinite(prob).all() or not np.allclose(prob.sum(1), 1):
        raise ValueError('Invalid probabilities')
    result = pd.DataFrame({'id': test.id, 'model': np.asarray(LABELS)[prob.argmax(1)]})
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out, index=False)
    write_json(out.with_suffix('.audit.json'), {'rows': len(result), 'sha256': sha256(out),
        'checkpoint_marker_sha256': sha256(Path(args.checkpoint) / 'COMPLETE.json'),
        'prediction_counts': result.model.value_counts().to_dict(),
        'identity': 'own plain LoRA control; no original ensemble score claim'})
    print(f'Wrote {len(result)} rows: {out}')

if __name__ == '__main__':
    main()
