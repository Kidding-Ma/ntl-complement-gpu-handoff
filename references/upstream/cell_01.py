# Name That LLM - full solution: fold-averaged two-stage LoRA24 (Qwen3-8B-Base, Qwen3-14B-Base, Qwen2.5-7B),
# Nishant two-stage checkpoints, 4-view n-gram SVM on train+pool, shared-prompt Gaussian, adaptive centred twins,
# 10% public ATLAS (source-conditioned likelihood, arXiv 2610.19583) before Sinkhorn balancing.
import glob, os, subprocess, json, sys
whl = glob.glob('/kaggle/input/**/bitsandbytes-*.whl', recursive=True)
if whl: print(subprocess.run([sys.executable, '-m', 'pip', 'install', '--break-system-packages', '--no-index', '--no-deps', '-q'] + whl, capture_output=True, text=True).returncode)
cfgs = sorted(glob.glob('/kaggle/input/**/config.json', recursive=True))
def mdir(*parts):
    c = [os.path.dirname(x) for x in cfgs if all(p in x.lower() for p in parts)]
    assert c, parts; return c[0]
def adir(name):
    h = glob.glob(f'/kaggle/input/**/{name}/heads.pt', recursive=True); assert h, name; return os.path.dirname(h[0])
DATA = next(p for p in ['/kaggle/input/competitions/name-that-llm', '/kaggle/input/name-that-llm'] if os.path.exists(p + '/train.csv'))
HEADS = os.path.dirname(next(iter(glob.glob('/kaggle/input/**/heads/q3_8bx.npz', recursive=True))))
EXTRA = next(iter(glob.glob('/kaggle/input/**/nlm_matched_clean.parquet', recursive=True)))
POOL = next(iter(glob.glob('/kaggle/input/**/nlm_pool_v5.parquet', recursive=True)))
B8, B14, B25 = mdir('/qwen-3/', '/8b-base/'), mdir('/qwen-3/', '/14b-base/'), mdir('/qwen2.5/', '/7b/')
I8, I14 = mdir('/qwen-3/', '/8b/'), mdir('/qwen-3/', '/14b/')
A8 = [adir(f'ts_q3b8_f{k}') for k in range(5)] + [adir('ts_q3b8')]
A14 = [adir(f'ts_q3b14_f{k}') for k in (0, 2, 4)] + [adir('ts_q3b14')]
A25 = [adir('ts_q25_7b')]
print(B8, B14, B25, I8, I14, A8, A14, A25, HEADS, EXTRA, POOL, sep='\n')
def nmeta(name):
    h = glob.glob(f'/kaggle/input/**/{name}/meta.json', recursive=True); assert h, name
    d = os.path.dirname(h[0]); c = json.load(open(h[0]))['cfg']; return d, c
N8, N8C = nmeta('lora8b_v1'); N14, N14C = nmeta('lora14b_v1')
print('nishant', N8, N8C['keep'], N8C['max_len'], N14, N14C['keep'], N14C['max_len'])
ATLAS = os.path.dirname(next(iter(glob.glob('/kaggle/input/**/atlas_bundle.json', recursive=True))))
print('atlas', ATLAS)
BL = mdir('/llama-3.1/', '/8b/')
AL = [adir(f'ts_llama8_f{k}') for k in (0, 2, 4)] + [adir('ts_llama8')]
print('llama', BL, AL)
A8S7 = [adir(f'ts8_s7_f{k}') for k in (0, 2, 4)] + [adir('ts8_s7')]
print('seed7', A8S7)
B25_14 = mdir('/qwen2.5/', '/14b/')
N14B, N14BC = nmeta('lora_qwen3_14b'); Q25_14, Q25_14C = nmeta('lora_qwen25_14b')
DEB = os.path.dirname(glob.glob('/kaggle/input/**/deberta_v3_large/meta.json', recursive=True)[0])
import shutil; shutil.copytree(DEB, '/kaggle/working/deberta_v3_large', dirs_exist_ok=True); DEB = '/kaggle/working/deberta_v3_large'
print('part two', N14B, N14BC['keep'], N14BC['layers'], N14BC.get('eos_id'), Q25_14, Q25_14C['keep'], Q25_14C['layers'], DEB)
AQ8V5 = [adir(f'q8v5_f{k}') for k in range(5)] + [adir('q8v5')]
print('q8v5', AQ8V5)
AQ14V5 = [adir(f'q14v5_f{k}') for k in range(5)] + [adir('q14v5')]
ALL8V5 = [adir(f'll8v5_f{k}') for k in range(5)] + [adir('ll8v5')]
print('v5', AQ14V5, ALL8V5)


# Combine EXTRA with GLM/Kimi dataset if present
GLM_KIMI = glob.glob('/kaggle/input/**/ntl_openrouter_glm_kimi.csv', recursive=True)
if GLM_KIMI:
    import pandas as pd
    df_extra = pd.read_parquet(EXTRA)
    df_gk = pd.read_csv(GLM_KIMI[0])[['text', 'model']]
    df_combined = pd.concat([df_extra, df_gk], ignore_index=True).drop_duplicates(subset=['text'])
    COMBINED_EXTRA = '/kaggle/working/extra_combined.parquet'
    df_combined.to_parquet(COMBINED_EXTRA, index=False)
    print(f'Combined extra created: {len(df_extra)} + {len(df_gk)} -> {len(df_combined)} rows')
    EXTRA = COMBINED_EXTRA
