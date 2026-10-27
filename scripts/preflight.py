"""Input structure and optional GPU checks, without loading model weights."""
import argparse
import importlib.util
import json
import os
from pathlib import Path


def inspect_inputs(input_root, gpu=False):
    root = Path(input_root).resolve()
    errors, warnings, found = [], [], {}
    for key in ('NTL_SMOKE', 'NTL_FAKE_TEST', 'NTL_TINY'):
        if os.environ.get(key):
            errors.append(f'Remove simulation environment variable {key}')
    if not root.is_dir():
        errors.append('Input directory does not exist: ' + str(root))
    # One tree traversal; no recursive glob for each adapter across large weight trees.
    files = [p for p in root.rglob('*') if p.is_file()] if root.exists() else []
    by_name = {}
    for f in files:
        by_name.setdefault(f.name, []).append(f)

    def require(key, pred, unique=True):
        hits = [f for f in files if pred(f)]
        found[key] = [str(f) for f in hits]
        if not hits:
            errors.append('Missing ' + key)
        elif unique and len(hits) != 1:
            errors.append('Ambiguous ' + key + ': ' + ', '.join(map(str, hits)))
        return hits

    data_dirs = [root / 'competitions/name-that-llm', root / 'name-that-llm']
    data = next((d for d in data_dirs if (d / 'train.csv').exists()), None)
    if not data:
        errors.append('Competition train.csv missing at expected location')
    else:
        found['data_dir'] = str(data)
        for name in ('train.csv', 'test.csv'):
            if not (data / name).is_file():
                errors.append('Missing ' + str(data / name))
        if (data / 'test.csv').exists():
            import csv
            for name, expected in [('train.csv', {'id', 'text', 'model', 'prompt_id'}), ('test.csv', {'id', 'text'})]:
                with (data / name).open(encoding='utf-8-sig', newline='') as stream:
                    reader = csv.DictReader(stream)
                    if not expected.issubset(set(reader.fieldnames or [])):
                        errors.append(name + ' has invalid columns')
    configs = by_name.get('config.json', [])
    for key, parts in {'B8': ('/qwen-3/', '/8b-base/'), 'B14': ('/qwen-3/', '/14b-base/'),
                       'B25': ('/qwen2.5/', '/7b/'), 'I8': ('/qwen-3/', '/8b/'), 'I14': ('/qwen-3/', '/14b/'),
                       'B4': ('/qwen-3/', '/4b-base/'), 'BL': ('/llama-3.1/', '/8b/'), 'B25_14': ('/qwen2.5/', '/14b/')}.items():
        require(key, lambda f, parts=parts: f.name == 'config.json' and all(x in f.as_posix().lower() for x in parts))
    for name in ('nlm_matched_clean.parquet', 'nlm_pool_v5.parquet', 'atlas_bundle.json', 'q3_8bx.npz', 'tokattn_debl23_final.pt'):
        require(name, lambda f, name=name: f.name == name)
    for atlas in by_name.get('atlas_bundle.json', []):
        for child in ('adapters.safetensors', 'ntl_atlas/model.py', 'ntl_rlcd/data.py'):
            if not (atlas.parent / child).is_file():
                errors.append('ATLAS dependency missing: ' + str(atlas.parent / child))
    for llama in found.get('BL', []):
        try:
            c = json.loads(Path(llama).read_text(encoding='utf-8'))
            if c.get('model_type') != 'llama' or c.get('hidden_size') != 4096 or c.get('num_hidden_layers') != 32 or c.get('vocab_size') != 128256:
                errors.append('Llama config does not match base Llama 3.1 8B')
        except ValueError:
            errors.append('Invalid Llama config')
    if not by_name.get('llama_asset_verification.json'):
        warnings.append('Llama cache verification report absent; independently verify official pinned file hashes')
    for name in ('ntl_openrouter_glm_kimi.csv',):
        hits = by_name.get(name, [])
        found[name] = [str(f) for f in hits]
        if not hits:
            errors.append('Declared GLM/Kimi input missing; omission changes V25 extra data')
    wheels = by_name.keys()
    if not any(n.startswith('bitsandbytes-') and n.endswith('.whl') for n in wheels):
        warnings.append('No offline bitsandbytes wheel; ensure working installed version and Kaggle offline wheel input')
    for name in ['ts_q3b8_f' + str(i) for i in range(5)] + ['ts_q3b8'] + ['ts_q3b14_f' + str(i) for i in (0, 2, 4)] + ['ts_q3b14', 'ts_q25_7b'] + ['ts_llama8_f' + str(i) for i in (0, 2, 4)] + ['ts_llama8'] + ['ts8_s7_f' + str(i) for i in (0, 2, 4)] + ['ts8_s7'] + [prefix + suffix for prefix in ('q8v5', 'q14v5', 'll8v5') for suffix in ['_f' + str(i) for i in range(5)] + ['']]:
        hits = require('adapter:' + name, lambda f, name=name: f.name == 'heads.pt' and f.parent.name == name)
        for f in hits:
            ad = f.parent / 'adapter' if (f.parent / 'adapter/adapter_config.json').exists() else f.parent
            for child in ('adapter_config.json', 'adapter_model.safetensors'):
                if not (ad / child).exists():
                    errors.append('Missing ' + str(ad / child))
    for name in ('lora8b_v1', 'lora14b_v1', 'lora_qwen3_14b', 'lora_qwen25_14b', 'deberta_v3_large'):
        hits = require('meta:' + name, lambda f, name=name: f.name == 'meta.json' and f.parent.name == name)
        for f in hits:
            try:
                m = json.loads(f.read_text(encoding='utf-8'))
                if name != 'deberta_v3_large' and not isinstance(m.get('cfg'), dict):
                    errors.append('cfg missing in ' + str(f))
                if name != 'deberta_v3_large':
                    if not (f.parent / 'heads.pt').exists() and not (f.parent / 'heads.safetensors').exists():
                        errors.append('Heads missing beside ' + str(f))
                    for child in ('adapter/adapter_config.json', 'adapter/adapter_model.safetensors'):
                        if not (f.parent / child).exists():
                            errors.append('Missing ' + str(f.parent / child))
                else:
                    for child in ('encoder/config.json', 'head.safetensors'):
                        if not (f.parent / child).exists():
                            errors.append('Missing ' + str(f.parent / child))
            except (ValueError, OSError) as e:
                errors.append(str(f) + ': ' + str(e))
    for name in ('deberta-v3-large', 'electra-large-discriminator', 'desklib-detector'):
        require('encoder:' + name, lambda f, name=name: f.name == 'config.json' and f.parent.name == name)
    glm = []
    ultra = []
    for f in configs:
        try:
            c = json.loads(f.read_text(encoding='utf-8'))
        except (ValueError, OSError):
            continue
        if c.get('model_type') == 'glm4' and c.get('hidden_size') == 4096:
            glm.append(str(f))
        if c.get('model_type') == 'ultrabert' and f.parent.name.startswith('fold') and f.parent.parent.name == 'ultrabert-380m-ntl':
            ultra.append(str(f))
        # Only base/encoder model configs, not adapter configs. Verify shard existence, not weight contents.
        if c.get('model_type'):
            indexes = list(f.parent.glob('*.index.json'))
            if indexes:
                for index in indexes:
                    try:
                        index_data = json.loads(index.read_text(encoding='utf-8'))
                        for shard in set(index_data.get('weight_map', {}).values()):
                            if not (f.parent / shard).is_file():
                                errors.append('Missing shard: ' + str(f.parent / shard))
                    except ValueError:
                        errors.append('Invalid index: ' + str(index))
            elif not any((f.parent / w).exists() for w in ('model.safetensors', 'pytorch_model.bin')):
                errors.append('Model weights missing beside ' + str(f))
    found['glm4_4096'], found['ultrabert_folds'] = glm, ultra
    if not glm:
        errors.append('GLM4 hidden_size=4096 checkpoint missing')
    if len(ultra) != 5 or len({str(Path(x).parent.parent) for x in ultra}) != 1:
        errors.append('Exactly one UltraBERT root with 5 fold configs is required')
    for name, pattern in [('lora24_seed7', '**/name-that-llm-train-lora24-s7/lora_out/job_final/COMPLETE.json'),
                           ('lora24_claude', '**/ntl-qwen3-8b-lora-claude-head/**/final/COMPLETE.json'),
                           ('lora20_q25_instyle', '**/ntl-train-lora-qwen25-instyle/lora_out/job_final/COMPLETE.json')]:
        hits = list(root.glob(pattern))
        found[name] = [str(f) for f in hits]
        if len(hits) != 1:
            errors.append('Missing/ambiguous complement LoRA marker: ' + name)
        for f in hits:
            for child in ('adapter/adapter_config.json', 'adapter/adapter_model.safetensors', 'readouts.npz',
                          'tokenizer/tokenizer_config.json', 'heads.safetensors' if name == 'lora24_claude' else 'norms.safetensors',
                          'anchor_features.npz' if name == 'lora24_claude' else 'valid_features.npz'):
                if not (f.parent / child).exists():
                    errors.append('Missing ' + str(f.parent / child))
    warnings.append('lora24_public historical skip must be disclosed; all other missing members fail acceptance')
    warnings.append('Structural check does not prove weight identity, label authenticity or GPU compatibility')
    if gpu:
        for module in ('torch', 'transformers', 'peft', 'accelerate', 'bitsandbytes', 'numpy', 'pandas', 'pyarrow', 'scipy', 'sklearn', 'lightgbm', 'safetensors', 'sentencepiece'):
            if importlib.util.find_spec(module) is None:
                errors.append('Missing Python package ' + module)
        try:
            import torch
            found['gpu'] = {'torch': torch.__version__, 'cuda': torch.version.cuda, 'count': torch.cuda.device_count(),
                            'devices': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
            if not torch.cuda.is_available():
                errors.append('CUDA unavailable')
        except Exception as e:
            errors.append('GPU inspection failed: ' + str(e))
    return {'ok': not errors, 'input_root': str(root), 'errors': errors, 'warnings': warnings, 'found': found}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root', type=Path, required=True)
    p.add_argument('--gpu', action='store_true')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    report = inspect_inputs(a.input_root, a.gpu)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if a.output:
        a.output.write_text(text, encoding='utf-8')
    raise SystemExit(0 if report['ok'] else 1)
