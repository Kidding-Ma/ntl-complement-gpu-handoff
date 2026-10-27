"""Build self-contained Kaggle derivative; preserve source recipe, fail on partial output."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build():
    cells = []
    def cell(kind, source):
        obj = {'cell_type': kind, 'id': f'handoff-{len(cells):02d}', 'metadata': {}, 'source': source.splitlines(keepends=True)}
        if kind == 'code':
            obj.update(execution_count=None, outputs=[])
        cells.append(obj)
    cell('markdown', '''# Name That LLM Solution V25 · Strict GPU handoff

Source: https://www.kaggle.com/code/mohammedafreedpasha/name-that-llm-solution

This derivative preserves V25's **actual** model settings (ATLAS .16 / margin .38 / Llama v5 1.5 / DeBERTa 1.3).
The upstream Markdown's .651 claim concerns another run and is **not verified for this notebook**.
Attach all 33 inputs from kernel-metadata.template.json, choose NVIDIA GPU, Internet OFF.
Transport/execution changes: check process failures, save unfiltered logs, reject missing components/stale outputs.
Main recipe and v5 adapters credited upstream to Dhruv Tibarewal; original comments remain in embedded modules.
Built with Llama. Each model/data input retains its own license; see repository NOTICE.md.
No training or GPU run was performed by the handoff author. Hidden-test evaluation remains required.
''')
    pre = (ROOT / 'scripts/preflight.py').read_text(encoding='utf-8')
    validation = (ROOT / 'scripts/validate_submission.py').read_text(encoding='utf-8')
    cell('code', '''import os, sys, json, glob, subprocess, time
from pathlib import Path
import torch
assert torch.cuda.is_available(), 'GPU required'
assert not any(os.environ.get(k) for k in ('NTL_SMOKE','NTL_FAKE_TEST','NTL_TINY')), 'Simulation disabled'
WORK = Path('/kaggle/working')
os.chdir(WORK)
for name in ('submission.parquet','comp_logit.npy','comp_meta.json','cache'):
    assert not (WORK/name).exists(), f'Use a fresh Kaggle session, stale output: {name}'
os.environ['NTL_DROP_Q35'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['PYTHONUNBUFFERED'] = '1'
for wheel in glob.glob('/kaggle/input/**/bitsandbytes-*.whl', recursive=True):
    subprocess.run([sys.executable,'-m','pip','install','--break-system-packages','--no-index','--no-deps','-q',wheel],check=True)
Path('pip-freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
Path('nvidia-smi.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
''')
    cell('code', 'pre_ns = {"__name__": "handoff_preflight"}\nexec(compile(' + repr(pre) + ',"embedded_preflight.py","exec"), pre_ns)\npreflight = pre_ns["inspect_inputs"]("/kaggle/input", gpu=True)\nPath("preflight.json").write_text(json.dumps(preflight,indent=2))\nassert preflight["ok"], preflight["errors"]\n')
    cell('code', (ROOT / 'references/upstream/cell_01.py').read_text(encoding='utf-8'))
    module_write = ''
    for name in ('gauss.py', 'infer.py', 'comp_stack.py'):
        module_write += f'Path({name!r}).write_bytes({(ROOT / "src" / name).read_bytes()!r})\n'
    cell('code', module_write)
    cell('code', '''def run_logged(cmd, path):
    with open(path,'w',encoding='utf-8') as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8',errors='replace',bufsize=1)
        try:
            for line in proc.stdout:
                print(line,end='',flush=True); log.write(line); log.flush()
            code = proc.wait()
        except BaseException:
            proc.terminate()
            raise
    assert code == 0, f'{path}: process exited {code}'
_c0 = time.time()
run_logged([sys.executable,'comp_stack.py'],'complement.log')
COMP_SEC = time.time() - _c0
assert 'COMP STACK DONE' in Path('complement.log').read_text(), 'Complement incomplete'
''')
    config = (ROOT / 'references/upstream/cell_06.py').read_text(encoding='utf-8').split('!python infer.py', 1)[0]
    cell('code', config + '''
assert ATLAS_BUDGET > 0, 'Original time budget exhausted'
cmd = [sys.executable,'infer.py','--data',DATA,'--models',json.dumps(MODELS),
 '--perpassage','1','--extra',EXTRA,'--extra_w','0.5','--tfidf_w','0','--distort_w','0','--svc_w','1.8',
 '--pool',POOL,'--adapt_w','1.0','--adapt_thr','0.15','--sinkhorn','1.0','--post','1',
 '--atlas_dir',ATLAS,'--atlas_base',B8,'--atlas_w','0.16','--atlas_margin','0.38','--atlas_bs','4',
 '--atlas_chunk','3','--atlas_budget',str(ATLAS_BUDGET),'--comp_path','/kaggle/working/comp_logit.npy',
 '--comp_w','0.15','--out','submission.parquet','--cache','/kaggle/working/cache']
Path('runtime-config.json').write_text(json.dumps({'MODELS':MODELS,'command':cmd,'COMP_SEC':COMP_SEC},indent=2))
run_logged(cmd,'inference.log')
''')
    cell('code', 'val_ns = {"__name__": "handoff_validation"}\nexec(compile(' + repr(validation) + ',"embedded_validation.py","exec"),val_ns)\n' + '''components = val_ns['check_components'](Path('complement.log').read_text(),Path('inference.log').read_text(),WORK,Path(DATA)/'test.csv')
submission_check = val_ns['validate'](Path(DATA)/'test.csv',WORK/'submission.parquet')
Path('submission-check.json').write_text(json.dumps({'components':components,'submission':submission_check,'hidden_score':'unverified'},indent=2))
pd.read_parquet('submission.parquet').to_csv('submission.csv',index=False)
print(submission_check)
print('Visible execution accepted. Official hidden test and score remain unverified.')
''')
    nb = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}, 'language_info': {'name': 'python'}}, 'cells': cells}
    target = ROOT / 'notebooks/solution_strict.ipynb'
    target.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(target)


if __name__ == '__main__':
    build()
