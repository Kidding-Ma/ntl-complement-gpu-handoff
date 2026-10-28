"""Patch a separately downloaded original notebook; do not execute its source."""
import argparse
import ast
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ntl.common import sha256, write_json

OLD_MARKER = '**/lora24/final/COMPLETE.json'

def patch_source(source, marker):
    tree = ast.parse(source)
    found = 0
    class Update(ast.NodeTransformer):
        def visit_Assign(self, node):
            nonlocal found
            if any(isinstance(t, ast.Name) and t.id == '_GAUSSIAN_SOURCE' for t in node.targets):
                if not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, str):
                    raise ValueError('_GAUSSIAN_SOURCE is no longer a literal; review the new source')
                embedded = node.value.value
                if embedded.count(OLD_MARKER) != 1 or 'lora24_public' not in embedded:
                    raise ValueError('Expected original checkpoint configuration not found exactly once')
                embedded = embedded.replace(OLD_MARKER, marker).replace('lora24_public', 'lora24_own')
                ast.parse(embedded)
                node.value = ast.copy_location(ast.Constant(embedded), node.value)
                found += 1
            return node
    result = Update().visit(tree)
    ast.fix_missing_locations(result)
    return ast.unparse(result) + '\n', found

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--checkpoint-glob', default='**/own_lora24/final/COMPLETE.json')
    args = p.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    if Path(args.input).resolve() == Path(args.output).resolve():
        raise ValueError('Keep the original notebook unchanged')
    if args.checkpoint_glob.startswith('/') or '..' in Path(args.checkpoint_glob).parts:
        raise ValueError('Use a relative glob under /kaggle/input')
    nb = json.loads(Path(args.input).read_text(encoding='utf-8'))
    matches = 0
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        source = cell['source']; source = ''.join(source) if isinstance(source, list) else source
        if '_GAUSSIAN_SOURCE' in source:
            source, count = patch_source(source, args.checkpoint_glob)
            matches += count
        cell['source'] = source.splitlines(keepends=True)
        cell['outputs'] = []; cell['execution_count'] = None
    if matches != 1:
        raise ValueError(f'Expected exactly one original stack, got {matches}')
    markers = [args.checkpoint_glob,
        '**/name-that-llm-train-lora24-s7/lora_out/job_final/COMPLETE.json',
        '**/ntl-qwen3-8b-lora-claude-head/**/final/COMPLETE.json',
        '**/ntl-train-lora-qwen25-instyle/lora_out/job_final/COMPLETE.json',
        '**/tokattn_final.pt', '**/tokattn_debl23_final.pt']
    early = ('# Fail early on absent inputs; this does not validate model identity or GPU compatibility.\n'
             'import glob\n'
             f'_required_markers = {markers!r}\n'
             '_missing = [p for p in _required_markers if not glob.glob("/kaggle/input/" + p, recursive=True)]\n'
             'if _missing:\n    raise FileNotFoundError("Missing inputs: " + repr(_missing))\n'
             'print("Input marker precheck passed; verify all other models and run GPU parity separately.")\n')
    nb['cells'].insert(0, {'cell_type': 'markdown', 'metadata': {}, 'source': [
        '# Own retrained LoRA replacement\n',
        'One original member is replaced by an independently trained checkpoint. This is a new unscored experiment.\n',
        'Source: https://www.kaggle.com/code/kiza123123/ntl-0604-dual-lora-gaussian-hungarian?scriptVersionId=360450415\n']})
    nb['cells'].insert(1, {'cell_type': 'code', 'metadata': {}, 'source': early.splitlines(keepends=True),
                            'outputs': [], 'execution_count': None})
    nb['metadata'].pop('widgets', None)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, nb)
    write_json(str(args.output) + '.patch.json', {
        'input_sha256': sha256(args.input), 'output_sha256': sha256(args.output),
        'replaced_member': 'lora24_public -> lora24_own', 'new_marker': args.checkpoint_glob,
        'other_changes': 'Outputs cleared; early file checks; ast.unparse formatting; other model logic preserved',
        'validation': 'Python syntax only; full GPU ensemble not run'})
    print(f'Wrote new unscored notebook: {args.output}')

if __name__ == '__main__':
    main()
