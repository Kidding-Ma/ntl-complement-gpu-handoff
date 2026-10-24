"""Static input and GPU inventory. A pass is necessary, not a forward-pass validation."""
import argparse, hashlib, json, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def check(root, gpu=False):
    root=Path(root); checks=[]
    def add(name, matches, minimum=1):
        paths=[str(x) for x in matches]
        checks.append(dict(name=name,ok=len(paths)>=minimum,paths=paths,minimum=minimum))
    for name in ['train.csv','test.csv','sample_submission.parquet']:
        add('official '+name,[p/name for p in [root/'competitions/name-that-llm',root/'name-that-llm'] if (p/name).is_file()])
    for pattern in ['pool_v4.parquet','**/lora_qwen3_14b/meta.json','**/lora_qwen25_14b/meta.json',
                    '**/deberta_v3_large/meta.json','legacy_recovery_manifest.json','atlas_bundle.json',
                    'tokattn_final.pt','tokattn_debl23_final.pt',
                    '**/name-that-llm-train-lora24-s7/lora_out/job_final/COMPLETE.json',
                    '**/ntl-qwen3-8b-lora-claude-head/**/final/COMPLETE.json',
                    '**/ntl-train-lora-qwen25-instyle/lora_out/job_final/COMPLETE.json']:
        add(pattern,list(root.glob('**/'+pattern.removeprefix('**/'))))
    configs=[]
    for p in root.rglob('config.json'):
        try: configs.append((p,json.loads(p.read_text())))
        except (ValueError,UnicodeError): pass
    specs=[('qwen3',5120,40),('qwen2',5120,48),('qwen3',4096,36),('qwen2',3584,None),('qwen3',2560,None),('glm4',4096,None),('qwen3_5',None,None)]
    for typ,h,layers in specs:
        add('backbone '+repr((typ,h,layers)),[p for p,c in configs if c.get('model_type')==typ and (h is None or c.get('hidden_size')==h) and (layers is None or c.get('num_hidden_layers')==layers)])
    for folder in ['deberta-v3-large','electra-large-discriminator','desklib-detector']:
        add(folder,[p for p,c in configs if p.parent.name==folder])
    add('UltraBERT folds',[p for p,c in configs if c.get('model_type')=='ultrabert' and p.parent.name.startswith('fold')],5)
    for mf in root.rglob('legacy_recovery_manifest.json'):
        try:
            m=json.loads(mf.read_text()); ok=bool(m['independently_verified_published_migration_manifest']) and bool(m['files'])
            for rel,rec in m['files'].items():
                rp=Path(rel); fp=mf.parent/rec['flat_file']
                ok=ok and not rp.is_absolute() and '..' not in rp.parts and fp.is_file()
                if fp.is_file():
                    with fp.open('rb') as stream: ok=ok and hashlib.file_digest(stream,'sha256').hexdigest()==rec['sha256']
            checks.append(dict(name='legacy archive integrity',ok=ok,paths=[str(mf)]))
        except (KeyError,ValueError,OSError) as e: checks.append(dict(name='legacy archive integrity',ok=False,error=str(e)))
    bundles=list(root.rglob('atlas_bundle.json'))
    checks.append(dict(name='exactly one ATLAS bundle',ok=len(bundles)==1))
    for b in bundles:
        for path in ['adapters.safetensors','ntl_atlas/model.py','ntl_rlcd/data.py']:
            add('ATLAS '+path,[b.parent/path] if (b.parent/path).is_file() else [])
    gpu_info={}
    if gpu:
        try:
            import torch
            gpu_info=dict(torch=torch.__version__,cuda=torch.version.cuda,devices=[dict(name=torch.cuda.get_device_name(i),memory=torch.cuda.get_device_properties(i).total_memory) for i in range(torch.cuda.device_count())])
            checks.append(dict(name='CUDA available',ok=torch.cuda.is_available()))
        except ImportError: checks.append(dict(name='torch installed',ok=False))
    return dict(input=str(root),ok=all(c['ok'] for c in checks),checks=checks,gpu=gpu_info,scope='Static markers/configs only; run all actual model forwards next. Check checkpoint tensors, adapters and tokenizer assets as well.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--input',default='/kaggle/input'); p.add_argument('--gpu',action='store_true'); p.add_argument('--output',default=str(ROOT/'reports/preflight.json')); a=p.parse_args()
    report=check(a.input,a.gpu); out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2))
    for c in report['checks']: print('OK' if c['ok'] else 'MISSING', c['name'])
    print('REPORT',out); sys.exit(0 if report['ok'] else 2)
