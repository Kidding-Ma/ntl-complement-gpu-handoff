"""Normalize notebook schema, leaving every upstream inference cell unchanged."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def main():
    nb=json.loads((ROOT/'upstream/ntl_part_two_complement_v2.ipynb').read_text(encoding='utf-8'))
    nb['nbformat']=4; nb['nbformat_minor']=5
    for i,c in enumerate(nb['cells']):
        c['id']=f'ntl-cell-{i:03d}'; c.setdefault('metadata',{})
        if c['cell_type']=='code': c['outputs']=[]; c['execution_count']=None
    out=ROOT/'ntl_part_two_complement_ready.ipynb'
    out.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    original=json.loads((ROOT/'upstream/ntl_part_two_complement_v2.ipynb').read_text(encoding='utf-8'))
    assert [c['source'] for c in original['cells']]==[c['source'] for c in nb['cells']]
    print(out)
if __name__=='__main__': main()
