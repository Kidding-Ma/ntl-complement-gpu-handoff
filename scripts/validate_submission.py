"""Validate against actual test IDs and the exact 18 model names."""
import argparse
import hashlib
import json
from pathlib import Path

LABELS = sorted(['claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
                 'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash', 'glm-5.3',
                 'glm-5.3-flash', 'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra', 'kimi-k3', 'qwen3.8-max'])


def validate(test_path, submission_path):
    import pandas as pd
    test, sub = pd.read_csv(test_path, dtype={'id': str}), pd.read_parquet(submission_path)
    if list(sub.columns) != ['id', 'model']:
        raise ValueError('Submission columns must be exactly [id, model]')
    if len(test) == 0 or len(sub) != len(test):
        raise ValueError('Submission row count differs from nonempty test')
    if sub.isna().any().any() or test['id'].isna().any():
        raise ValueError('Null values')
    if not test['id'].is_unique or not sub['id'].is_unique:
        raise ValueError('Duplicate IDs')
    if not all(isinstance(x, str) for x in sub['id']):
        raise ValueError('Submission IDs must preserve CSV string identities')
    if set(sub['id']) != set(test['id']):
        raise ValueError('Submission ID set differs from test')
    if not sub['model'].isin(LABELS).all():
        raise ValueError('Invalid label')
    return {'ok': True, 'rows': len(sub), 'sha256': hashlib.sha256(Path(submission_path).read_bytes()).hexdigest(),
            'label_counts': sub['model'].value_counts().to_dict(), 'evidence': 'format only; no score established'}


def check_components(complement_log, inference_log, work_dir, test_path):
    import numpy as np
    import pandas as pd
    if 'COMP STACK DONE' not in complement_log or 'smoke False' not in complement_log:
        raise ValueError('Missing real complement completion evidence')
    for word in ['COMP SKIPPED', 'ATLAS SKIPPED', 'Traceback (most recent call last)']:
        if word in complement_log + inference_log:
            raise ValueError('Component failure: ' + word)
    skips = [line for line in complement_log.splitlines() if 'SKIP ' in line]
    if any('SKIP lora24_public:' not in line for line in skips):
        raise ValueError('Unexpected skipped member: ' + repr(skips))
    if 'comp stack loaded (' not in inference_log:
        raise ValueError('Complement not consumed')
    import re
    if not re.search(r'atlas .* mixed at 0\.16(?:\s|$)', inference_log):
        raise ValueError('ATLAS was not mixed at original V25 weight 0.16')
    expected = {'q14v5': ['q14v5_f' + str(i) for i in range(5)] + ['q14v5'],
                'q8v5': ['q8v5_f' + str(i) for i in range(5)] + ['q8v5'],
                'll8v5': ['ll8v5_f' + str(i) for i in range(5)] + ['ll8v5'],
                'll8_ts': ['ts_llama8_f' + str(i) for i in (0, 2, 4)] + ['ts_llama8'],
                'q8s7': ['ts8_s7_f' + str(i) for i in (0, 2, 4)] + ['ts8_s7'],
                'n14b': ['lora_qwen3_14b'], 'q25_14': ['lora_qwen25_14b']}
    for name, adapters in expected.items():
        for adapter in adapters:
            if f'l24 {name}: adapter {adapter} done ' not in inference_log:
                raise ValueError('Missing adapter completion: ' + name + '/' + adapter)
    if not re.search(r'\bdeb done\b', inference_log):
        raise ValueError('DeBERTa not completed')
    work = Path(work_dir)
    meta = json.loads((work / 'comp_meta.json').read_text(encoding='utf-8'))
    test = pd.read_csv(test_path, dtype={'id': str})
    if meta['ids'] != test['id'].tolist() or sorted(meta['labels']) != LABELS or len(set(meta['labels'])) != 18:
        raise ValueError('Complement ID/label alignment mismatch')
    logits = np.load(work / 'comp_logit.npy', allow_pickle=False)
    if logits.shape != (len(test), 18) or not np.isfinite(logits).all():
        raise ValueError('Bad complement logits shape or nonfinite values')
    return {'ok': True, 'allowed_historical_skips': skips, 'required_adapter_count': sum(map(len, expected.values())),
            'atlas_weight': 0.16, 'comp_shape': list(logits.shape), 'evidence': 'execution markers + output structure; no hidden score'}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--test', type=Path, required=True)
    p.add_argument('--submission', type=Path, required=True)
    p.add_argument('--report', type=Path)
    a = p.parse_args()
    result = validate(a.test, a.submission)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if a.report:
        a.report.write_text(text, encoding='utf-8')
