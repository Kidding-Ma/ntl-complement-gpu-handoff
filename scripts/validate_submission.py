import argparse, json
from pathlib import Path

def validate(work,inputs):
    import numpy as np
    import pandas as pd
    data=next(p for p in [inputs/'competitions/name-that-llm',inputs/'name-that-llm'] if (p/'test.csv').is_file())
    test=pd.read_csv(data/'test.csv',dtype={'id':str}); train=pd.read_csv(data/'train.csv'); sub=pd.read_parquet(work/'submission.parquet')
    assert list(sub.columns)==['id','model']
    assert len(sub)==len(test) and sub.id.is_unique and sub.notna().all().all()
    assert sub.id.astype(str).tolist()==test.id.tolist()
    assert set(sub.model).issubset(set(train.model))
    proof=json.loads((work/'enhanced_verification.json').read_text())
    assert proof['rows']==len(test) and proof['all_required_components_verified'] and proof['full_actual_model_inference']
    assert proof['comp_weight']==.15 and proof['atlas_weight']==.1
    expected={'stylegram','stylometry_lgb','debv3l','electl','desk','lora24_public','lora24_seed7','lora24_claude','lora20_q25_instyle','tokattn_q35','tokattn_deb','ultrabert'}
    assert expected.issubset(set(proof['actual_complement_members']))
    for name in ['comp_members.npz','part_two_members.npz','enhanced_probabilities.npz']:
        with np.load(work/name,allow_pickle=False) as z:
            assert z['ids'].astype(str).tolist()==test.id.tolist()
            assert len(set(z['labels']))==18
            for key in z.files:
                if np.issubdtype(z[key].dtype,np.number): assert np.isfinite(z[key]).all(),(name,key)
    result=dict(rows=len(sub),format_only=len(sub)==3,accuracy_validated=False,all_required_components=True)
    (work/'handoff_validation.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--working',type=Path,default=Path('/kaggle/working')); p.add_argument('--input',type=Path,default=Path('/kaggle/input')); a=p.parse_args(); validate(a.working,a.input)
