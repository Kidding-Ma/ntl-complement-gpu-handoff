import glob, json, os
from pathlib import Path

def _complete_model_weights(directory):
    directory = Path(directory)
    for filename in ('model.safetensors', 'pytorch_model.bin'):
        p = directory / filename
        if p.is_file() and p.stat().st_size > 0:
            return True
    for filename in ('model.safetensors.index.json', 'pytorch_model.bin.index.json'):
        p = directory / filename
        if p.is_file():
            try:
                shards = set(json.loads(p.read_text())['weight_map'].values())
                if shards and all((directory / s).is_file() and (directory / s).stat().st_size > 0 for s in shards):
                    return True
            except (OSError, ValueError, KeyError, TypeError):
                pass
    return False

def _weighted_configs():
    for filename in sorted(glob.glob('/kaggle/input/**/config.json', recursive=True)):
        directory = Path(filename).parent
        if directory.name in ('base_config', 'adapter', 'tokenizer') or not _complete_model_weights(directory):
            continue
        try:
            config = json.loads(Path(filename).read_text())
        except (OSError, ValueError):
            continue
        yield str(directory), config

def _resolve_weighted_model(spec=None, dir_name=None):
    matches = []
    for directory, config in _weighted_configs():
        if dir_name and Path(directory).name != dir_name:
            continue
        if spec and not all(config.get(k) == v for k, v in spec.items()):
            continue
        matches.append(directory)
    if not matches:
        raise FileNotFoundError(f'No complete model weights found for {spec or dir_name}. Attach the original base model, not its base_config snapshot.')
    # Official model mounts precede datasets when both contain a matching model.
    matches.sort(key=lambda p: ('/models/' not in p, p))
    print(f'[base resolved] {spec or dir_name}: {matches[0]}', flush=True)
    return matches[0]

import numpy as np
print('REPAIR v1: checking complete base weights and checkpoint files before inference', flush=True)
_required_markers = ['**/own_lora24/final/COMPLETE.json', '**/name-that-llm-train-lora24-s7/lora_out/job_final/COMPLETE.json', '**/ntl-qwen3-8b-lora-claude-head/**/final/COMPLETE.json', '**/ntl-train-lora-qwen25-instyle/lora_out/job_final/COMPLETE.json', '**/tokattn_final.pt', '**/tokattn_debl23_final.pt']
for marker in _required_markers:
    hits = sorted(glob.glob('/kaggle/input/' + marker, recursive=True))
    if len(hits) != 1:
        raise RuntimeError(f'Expected exactly one checkpoint for {marker}, found {hits}')
for model_spec in [dict(model_type='qwen3', hidden_size=4096, num_hidden_layers=36), dict(model_type='qwen3', hidden_size=5120, num_hidden_layers=40), dict(model_type='qwen3', hidden_size=2560), dict(model_type='qwen2', hidden_size=3584)]:
    _resolve_weighted_model(model_spec)
_own = Path(glob.glob('/kaggle/input/**/own_lora24/final/COMPLETE.json', recursive=True)[0]).parent
for relative in ['adapter/adapter_config.json', 'adapter/adapter_model.safetensors', 'heads.safetensors', 'readouts.npz', 'anchor_features.npz', 'tokenizer/tokenizer_config.json', 'training_meta.json']:
    if not (_own / relative).is_file():
        raise FileNotFoundError(str(_own / relative))
_ro = np.load(_own / 'readouts.npz', allow_pickle=False)
_meta = json.loads((_own / 'training_meta.json').read_text())
assert list(_ro['classes']) == _meta['labels'], 'Own checkpoint label order mismatch'
assert list(_ro['pool_layers']) == _meta['config']['pool_layers'], 'Own checkpoint pool layer mismatch'
assert _ro['plain_coef'].shape == (18, 32768)
assert _ro['center_coef'].shape == (18, 32768)
print('REPAIR PREFLIGHT PASSED: base weights exist; own checkpoint contract valid', flush=True)
