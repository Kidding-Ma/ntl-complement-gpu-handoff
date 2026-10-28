"""Verify delivery hashes, syntax, baseline notebook identity, data and manifest."""
import ast, hashlib, json, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def main():
    manifest=json.loads((ROOT/'manifests/workspace_sha256.json').read_text(encoding='utf-8'))
    errors=[]
    for name,expected in manifest.items():
        p=ROOT/name
        if not p.is_file(): errors.append('MISSING '+name)
        elif hashlib.sha256(p.read_bytes()).hexdigest()!=expected: errors.append('CHANGED '+name)
    parsed=0
    for p in [*ROOT.glob('scripts/*.py'),*ROOT.glob('upstream/**/*.py')]:
        try: ast.parse(p.read_text(encoding='utf-8'),filename=str(p)); parsed+=1
        except SyntaxError as e: errors.append(str(e))
    nb=json.loads((ROOT/'upstream/ntl_part_two_complement_v2.ipynb').read_text(encoding='utf-8'))
    ready=json.loads((ROOT/'ntl_part_two_complement_ready.ipynb').read_text(encoding='utf-8'))
    assert [c['source'] for c in nb['cells']]==[c['source'] for c in ready['cells']]
    assert ready['nbformat_minor']==5
    for i,c in enumerate(ready['cells']):
        if c['cell_type']=='code': ast.parse(''.join(c['source']),filename=f'cell_{i}')
    assets=json.loads((ROOT/'manifests/assets.json').read_text())['assets']
    assert len(assets)==32 and len({(a['kind'],a['ref']) for a in assets})==32
    import csv
    with (ROOT/'data/competition/train.csv').open(encoding='utf-8',newline='') as f: train=list(csv.DictReader(f))
    with (ROOT/'data/competition/test.csv').open(encoding='utf-8',newline='') as f: test=list(csv.DictReader(f))
    assert len(train)==2699 and len(test)==3 and len({x['model'] for x in train})==18
    assert all(k in train[0] for k in ['id','prompt_id','text','model'])
    result=dict(ok=not errors,files_hashed=len(manifest),python_files_parsed=parsed,notebook_cells=len(nb['cells']),source_identical=True,assets=len(assets),train_rows=len(train),test_preview_rows=len(test),gpu_tested_by_this_check=False,errors=errors)
    print(json.dumps(result,indent=2)); return 0 if not errors else 1
if __name__=='__main__': sys.exit(main())
