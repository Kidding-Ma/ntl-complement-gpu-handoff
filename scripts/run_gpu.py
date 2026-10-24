"""Execute each original notebook cell in order on Linux, retaining source math."""
import argparse, json, os, platform, sys, time, traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--working',default='/kaggle/working'); p.add_argument('--sync-load',action='store_true',help='Opt-in unverified Qwen3.5 sequential loader diagnostic'); a=p.parse_args()
    if platform.system()!='Linux': p.error('Use a Linux GPU server or Kaggle. Upstream paths are /kaggle/input and /kaggle/working.')
    if a.working!='/kaggle/working': p.error('Unmodified upstream writes /kaggle/working; mount your output volume there.')
    for key in ['NTL_SMOKE','NTL_FAKE_TEST','NTL_TINY']:
        if os.environ.get(key): p.error(key+' must be unset for real inference')
    if a.sync_load: os.environ['HF_DEACTIVATE_ASYNC_LOAD']='1'
    sys.path.insert(0,str(ROOT/'scripts'))
    from preflight import check
    report=check('/kaggle/input',True)
    work=Path(a.working); work.mkdir(parents=True,exist_ok=True)
    (work/'handoff_preflight.json').write_text(json.dumps(report,indent=2))
    if not report['ok']: sys.exit('Input/GPU preflight failed. Read handoff_preflight.json before spending GPU time.')
    os.chdir(work); sys.path.insert(0,str(work))
    nb=json.loads((ROOT/'upstream/ntl_part_two_complement_v2.ipynb').read_text(encoding='utf-8'))
    ns={'__name__':'__main__'}; results=[]; start=time.monotonic()
    try:
        for i,c in enumerate(nb['cells']):
            if c['cell_type']!='code': continue
            source=''.join(c['source']); print('CELL_START',i,round(time.monotonic()-start,2),flush=True)
            began=time.monotonic(); exec(compile(source,f'upstream_cell_{i}','exec'),ns)
            results.append(dict(cell=i,seconds=time.monotonic()-began)); print('CELL_DONE',i,flush=True)
    except BaseException:
        (work/'handoff_failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        (work/'handoff_cell_timings.json').write_text(json.dumps(results,indent=2))
    from validate_submission import validate
    validate(work,Path('/kaggle/input'))

if __name__=='__main__': main()
