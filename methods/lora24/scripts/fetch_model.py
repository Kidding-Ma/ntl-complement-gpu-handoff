"""Download the public official Qwen3-8B Base, then verify complete local shards."""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ntl.common import sha256, write_json

MODEL_ID = 'Qwen/Qwen3-8B-Base'

def validate_model(root):
    root = Path(root)
    cfg = json.loads((root / 'config.json').read_text(encoding='utf-8'))
    expected = {'model_type': 'qwen3', 'hidden_size': 4096, 'num_hidden_layers': 36,
                'intermediate_size': 12288, 'vocab_size': 151936}
    if any(cfg.get(k) != v for k, v in expected.items()):
        raise ValueError(f'Not the required official Qwen3-8B Base configuration: {root}')
    index = root / 'model.safetensors.index.json'
    if index.is_file():
        shards = sorted(set(json.loads(index.read_text())['weight_map'].values()))
    else:
        shards = ['model.safetensors']
    for name in shards:
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or path.stat().st_size < 1024 * 1024:
            raise ValueError(f'Missing/incomplete model shard: {name}')
    for name in ['tokenizer.json', 'tokenizer_config.json']:
        if not (root / name).is_file():
            raise ValueError(f'Missing tokenizer file: {name}')
    return {'model_id': MODEL_ID, 'base_config_sha256': sha256(root / 'config.json'),
            'shards': [{'file': name, 'bytes': (root / name).stat().st_size} for name in shards]}

def fetch(root, source='auto'):
    root = Path(root).resolve()
    try:
        report = validate_model(root)
        print(f'Using complete existing base: {root}', flush=True)
        return root
    except (OSError, ValueError, KeyError):
        pass
    root.mkdir(parents=True, exist_ok=True)
    errors = []
    sources = ['modelscope', 'huggingface'] if source == 'auto' else [source]
    for selected in sources:
        print(f'Downloading {MODEL_ID} from {selected} into {root} (~16 GB)', flush=True)
        started = time.monotonic()
        try:
            if selected == 'modelscope':
                from modelscope.hub.snapshot_download import snapshot_download
                snapshot_download(MODEL_ID, revision='master', local_dir=str(root))
                revision = 'master'
            else:
                from huggingface_hub import HfApi, snapshot_download
                revision = HfApi().model_info(MODEL_ID, token=False).sha
                snapshot_download(MODEL_ID, revision=revision, local_dir=str(root), token=False,
                    allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', 'LICENSE*', 'README*'])
            report = validate_model(root)
            report.update(source=selected, revision=revision, seconds=time.monotonic() - started)
            write_json(root / 'download_report.json', report)
            return root
        except Exception as error:
            errors.append({'source': selected, 'error': f'{type(error).__name__}: {error}'})
            print(f'{selected} failed; trying the next configured public source.', flush=True)
    write_json(root / 'download_errors.json', errors)
    raise RuntimeError('Model download failed. See models/.../download_errors.json; '
                       'install requirements-fetch.txt or pass an existing --base-model directory.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='models/Qwen3-8B-Base')
    parser.add_argument('--source', choices=['auto', 'modelscope', 'huggingface'], default='auto')
    args = parser.parse_args()
    fetch(args.output, args.source)
