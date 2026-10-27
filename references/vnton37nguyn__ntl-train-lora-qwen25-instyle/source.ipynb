# NTL: Qwen2.5-7B LoRA (same recipe/seed/folds as ntl-train-lora-qwen25-7b) + in-style generated passages
# IN-STYLE extra passages (prompt-bank protocol; public proxy/API sets + any attached ntl_generated*.csv, prompts with
# >= 12 labels), grouped by their prompt for same-prompt triplets. Jobs 0,1,2 (compare with the plain run) + final.
# NTL: LoRA24 recipe (lucasboesen/name-that-llm-train-lora24-s7) on Qwen2.5-7B (28 layers -> keep 20, hidden 3584),
# pooled layers 2/4/6/8/10/12/16/20, seed 11, prompt folds seed 0. Outputs: lora_out/job_{0..4,final}.
# Training kernel (no submission): our own run of the public "LoRA24" recipe (lightningv08/ntl-lora24-fivefold-training,
# whose 5-fold OOF plain head scores 0.4816 and adds +0.0046 held out to our v13 set), on OUR prompt folds
# (scripts/common.py folds(seed 0)) with a different seed, for a second LoRA member with aligned OOF.
# Recipe: Qwen3-8B-Base truncated to 24 layers, LoRA r16/a32/dropout .05 on q,k,v,o,gate,up,down; mean pool with the
# first token masked over layers 2/4/6/8/12/16/20/24, a LayerNorm(4096) per layer, concatenated -> plain and centered
# Linear heads; microbatches of 3 same-prompt answers; loss CE(plain(z)) + CE(centered(z - microbatch mean));
# 6-step accumulation; AdamW lr 1e-4 (adapters) / 1e-3 (heads+norms), wd .01; 10% warmup + cosine; clip 1.0; 3 epochs;
# fp16 frozen base, fp32 trainable params, fp16 autocast + GradScaler(256); gradient checkpointing. max_len 447 + EOS.
# Jobs: outer folds 0-4 (OOF z + plain/center readouts for held-out rows) and 'final' (all rows; adapter saved for
# inference). Two workers, one per T4; each job writes its outputs as it finishes, so a timeout keeps finished folds.
SEED, FOLD_SEED = 11, 0   # same folds and seed as ntl-train-lora-qwen25-7b -> fold accs directly comparable
JOBS = {0: ['0', 'final'], 1: ['1', '2']}
LAYERS = [2, 4, 6, 8, 10, 12, 16, 20]
KEEP, MAX_LEN, EPOCHS, ACCUM = 20, 447, 3, 6
LR_ADAPTER, LR_HEAD, WD, WARMUP = 1e-4, 1e-3, 0.01, 0.10
R, ALPHA, LDROP = 16, 32, 0.05

import glob
import json
import os
import pathlib
import subprocess
import sys
import time

import numpy as np
import pandas as pd

WORKER = r'''
import json, math, os, random, sys, time
import importlib.util as _iu
_fs = _iu.find_spec   # Kaggle ships torchao 0.10, which peft rejects; hide it (unused)
_iu.find_spec = lambda name, *a, **k: None if name.split('.')[0] == 'torchao' else _fs(name, *a, **k)
import numpy as np, pandas as pd, torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model
import peft.import_utils as _pi
_pi.is_torchao_available = lambda *a, **k: False
from safetensors.torch import save_file

cfg = json.loads(sys.argv[1]); jobs = sys.argv[2].split(',')
train = pd.read_csv(cfg['train_csv'])
LABELS = sorted(train.model.unique()); y = train.model.map({l: i for i, l in enumerate(LABELS)}).values
pid = train.prompt_id.values
NTR = len(train)
import re

CODE = re.compile(r'```|`|[{}<>]|=>|\b(def|class|import|return|function|const|var)\b|\.(py|js|ts|php|json|md)\b|/[a-z_]+/')
BULLET = re.compile(r'^\s*(?:[-*•]|\d+[.)]|#+)\s*')


def clean_text(t):
    lines = [l.strip() for l in str(t).split('\n') if l.strip()]
    if not lines:
        return None
    bullets = sum(bool(BULLET.match(l)) for l in lines)
    if bullets > 0.3 * len(lines) or len(lines) > 4:
        return None                                # lists / structured answers: not paragraph prose
    s = ' '.join(BULLET.sub('', l) for l in lines)
    s = re.sub(r'\*\*(.+?)\*\*', r'\1', s)
    s = re.sub(r'\s+', ' ', s).strip()
    if CODE.search(s):
        return None
    n = len(s.split())
    if n < 40 or n > 250 or s[-1] not in '.!?"”)':
        return None
    return s


def load_clean(long, cap=250, seed=0):
    long = long.copy()
    long['text'] = long['text'].map(clean_text)
    long = long.dropna(subset=['text']).drop_duplicates(subset=['text'])
    rng = np.random.RandomState(seed)
    parts = [g if len(g) <= cap else g.iloc[rng.permutation(len(g))[:cap]] for _, g in long.groupby('model')]
    return pd.concat(parts).reset_index(drop=True)

import glob, os
EXTRA_MIN_LABELS = 12   # keep only prompts answered by >= this many labels (avoids topic -> label shortcuts)
EXTRA_W = 0.5           # per class, share of the class weight moved to its extra rows (Seyit's scheme)


def load_extra_passages():
    frames = []
    for pat, lab, keep in [('**/ntl_extra_passages_claude.csv', 'model', None), ('**/gen_passages.csv', 'model', None),
                           ('**/ntl_generations.csv', 'label', 'finish_reason'), ('**/ntl_proxy_generations.csv', 'label', None),
                           ('**/ntl_generated*.csv', 'model', 'finish')]:
        for f in sorted(glob.glob('/kaggle/input/' + pat, recursive=True)):
            d = pd.read_csv(f)
            if keep and keep in d:
                d = d[d[keep].eq('stop')]
            d = d.rename(columns={lab: 'model'})[['prompt_id', 'model', 'text']].assign(src=os.path.basename(f))
            frames.append(d)
    if not frames:
        return pd.DataFrame(columns=['prompt_id', 'model', 'text'])
    x = pd.concat(frames, ignore_index=True)
    x = x[x.model.isin(LABELS)].copy()
    x['text'] = x.text.astype(str).map(lambda t: re.sub(r'\s+', ' ', re.sub(r'^\*\*(.*)\*\*$', r'\1', t.strip())).strip())
    x = x[x.text.str.split().str.len().between(30, 300)].drop_duplicates(subset=['model', 'text'])
    x = x.drop_duplicates(subset=['prompt_id', 'model'], keep='first')
    x = x[x.groupby('prompt_id').model.transform('nunique') >= EXTRA_MIN_LABELS].reset_index(drop=True)
    print('extra passages:', len(x), 'rows,', x.prompt_id.nunique(), 'prompts |', x.model.value_counts().to_dict(), flush=True)
    return x


EXT = load_extra_passages().rename(columns={'prompt_id': 'id'})
y_all = np.concatenate([y, EXT.model.map({l: i for i, l in enumerate(LABELS)}).values])
pid_all = np.concatenate([pid, ('ext_' + EXT['id'].astype(str)).values])
texts_all = train.text.tolist() + EXT.text.tolist()
prompts = np.array(sorted(train.prompt_id.unique()))
rng0 = np.random.RandomState(cfg['fold_seed']); rng0.shuffle(prompts)   # == scripts/common.py folds()
fold = train.prompt_id.map({p: i % 5 for i, p in enumerate(prompts)}).values
tok = AutoTokenizer.from_pretrained(cfg['model_dir'])
tok.padding_side = 'right'
if tok.pad_token is None: tok.pad_token = tok.eos_token
tokens = [ids + [tok.eos_token_id] for ids in
          tok(texts_all, add_special_tokens=False, truncation=True, max_length=cfg['max_len'])['input_ids']]
L = cfg['layers']; dev = torch.device('cuda:0'); H = cfg['hidden']

def triplets(rows, rng):
    out = []; loose = []
    for p in np.unique(pid_all[rows]):
        m = rng.permutation(rows[pid_all[rows] == p])
        if len(m) < 3:   # external prompts answered by < 3 models: pooled into random mixed triplets below
            loose.extend(int(v) for v in m)
            continue
        for s in range(0, len(m), 3):
            b = list(m[s:s + 3])
            while len(b) < 3:   # fill a short last chunk with other answers to the same prompt
                b.append(int(rng.choice([x for x in m if x not in b])))
            out.append(b)
    loose = list(rng.permutation(loose))
    out += [loose[s:s + 3] for s in range(0, len(loose) - 2, 3)]
    out = [out[i] for i in rng.permutation(len(out))]
    while len(out) % cfg['accum']: out.append(out[rng.randint(len(out))])
    return out

for job in jobs:
    t0 = time.time()
    outdir = os.path.join('lora_out', f'job_{job}'); os.makedirs(outdir, exist_ok=True)
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed']); torch.cuda.manual_seed_all(cfg['seed'])
    rng = np.random.RandomState(cfg['seed'] + (99 if job == 'final' else int(job)))
    fit = np.arange(len(train)) if job == 'final' else np.where(fold != int(job))[0]
    fit = np.concatenate([fit, NTR + np.arange(len(EXT))])   # + all cleaned external rows
    valid = np.arange(3) if job == 'final' else np.where(fold == int(job))[0]
    mc = AutoConfig.from_pretrained(cfg['model_dir'])
    mc.num_hidden_layers, mc.use_cache = cfg['keep'], False
    if getattr(mc, 'layer_types', None) is not None: mc.layer_types = mc.layer_types[:cfg['keep']]
    base = AutoModel.from_pretrained(cfg['model_dir'], config=mc, dtype=torch.float16).to(dev)
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    base.enable_input_require_grads()
    model = get_peft_model(base, LoraConfig(r=cfg['r'], lora_alpha=cfg['alpha'], lora_dropout=cfg['ldrop'], bias='none',
                           target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']))
    for p in model.parameters():
        if p.requires_grad: p.data = p.data.float()
    norms = nn.ModuleList([nn.LayerNorm(H) for _ in L]).to(dev).float()
    plain = nn.Linear(H * len(L), len(LABELS)).to(dev).float()
    centered = nn.Linear(H * len(L), len(LABELS)).to(dev).float()
    drop = nn.Dropout(0.1)
    ad = [p for p in model.parameters() if p.requires_grad]
    hd = list(norms.parameters()) + list(plain.parameters()) + list(centered.parameters())
    opt = torch.optim.AdamW([{'params': ad, 'lr': cfg['lr_adapter']}, {'params': hd, 'lr': cfg['lr_head']}], weight_decay=cfg['wd'])
    plan = [triplets(fit, rng) for _ in range(cfg['epochs'])]
    total = sum(len(e) for e in plan) // cfg['accum']; warm = max(1, int(cfg['warmup'] * total))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                            0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler(init_scale=256.0)

    def rep(idx):
        batch = tok.pad({'input_ids': [tokens[i] for i in idx]}, return_tensors='pt').to(dev)
        with torch.autocast('cuda', dtype=torch.float16):
            hs = model(**batch, output_hidden_states=True, use_cache=False).hidden_states
        m = batch['attention_mask'].unsqueeze(-1).float(); m[:, 0] = 0
        with torch.autocast('cuda', enabled=False):
            return torch.cat([nm((hs[l].float() * m).sum(1) / m.sum(1)) for l, nm in zip(L, norms)], -1)

    model.train(); norms.train(); step = 0; skipped = 0; losses = []
    for ep, batches in enumerate(plan):
        for i, b in enumerate(batches):
            z = rep(b)
            yy = torch.as_tensor(y_all[b], device=dev)
            loss = nn.functional.cross_entropy(plain(drop(z)), yy) + \
                nn.functional.cross_entropy(centered(drop(z - z.mean(0, keepdim=True))), yy)
            scaler.scale(loss / cfg['accum']).backward(); losses.append(float(loss.detach()))
            if (i + 1) % cfg['accum'] == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(ad + hd, 1.0)
                before = scaler.get_scale(); scaler.step(opt); scaler.update()
                if scaler.get_scale() < before: skipped += 1
                opt.zero_grad(set_to_none=True); sch.step(); step += 1
                if step % 30 == 0:
                    print(f'job {job} ep {ep} step {step}/{total} loss {np.mean(losses[-180:]):.4f} skipped {skipped} '
                          f'({time.time() - t0:.0f}s)', flush=True)
    model.eval(); norms.eval()
    Z = np.zeros((len(valid), H * len(L)), np.float32)
    order = np.argsort([len(tokens[i]) for i in valid], kind='stable')
    with torch.inference_mode():
        for s in range(0, len(order), 6):
            pos = order[s:s + 6]
            Z[pos] = rep(valid[pos]).cpu().numpy()
    assert np.isfinite(Z).all()
    W = {n: t.weight.detach().cpu().numpy() for n, t in (('plain', plain), ('center', centered))}
    B = {n: t.bias.detach().cpu().numpy() for n, t in (('plain', plain), ('center', centered))}
    np.savez_compressed(os.path.join(outdir, 'readouts.npz'), classes=np.array(LABELS), pool_layers=np.array(L),
                        plain_coef=W['plain'], plain_intercept=B['plain'], center_coef=W['center'], center_intercept=B['center'])
    np.savez_compressed(os.path.join(outdir, 'valid_features.npz'), ids=train.id.values[valid], features=Z.astype(np.float16))
    lg = Z @ W['plain'].T + B['plain']
    acc = float((lg.argmax(1) == y[valid]).mean())
    if job == 'final':
        model.save_pretrained(os.path.join(outdir, 'adapter'), safe_serialization=True)
        save_file({**{f'norms.{k}': v.detach().cpu().contiguous() for k, v in norms.state_dict().items()}},
                  os.path.join(outdir, 'norms.safetensors'))
        tok.save_pretrained(os.path.join(outdir, 'tokenizer'))
    json.dump({'job': job, 'valid_acc': acc, 'steps': step, 'skipped': skipped, 'secs': time.time() - t0},
              open(os.path.join(outdir, 'COMPLETE.json'), 'w'))
    print(f'job {job} done: valid acc {acc:.4f} steps {step} skipped {skipped} ({time.time() - t0:.0f}s)', flush=True)
    del model, base, opt, norms, plain, centered; torch.cuda.empty_cache()
'''

T0 = time.time()
DATA_DIR = next(p for p in [pathlib.Path('/kaggle/input/competitions/name-that-llm'),
                            pathlib.Path('/kaggle/input/name-that-llm'), pathlib.Path('data')]
                if (p / 'train.csv').exists())
model_dir = None
for c in sorted(glob.glob('/kaggle/input/**/config.json', recursive=True)):
    try:
        j = json.load(open(c))
    except Exception:
        continue
    if j.get('model_type') == 'qwen2' and j.get('hidden_size') == 3584 and j.get('num_hidden_layers') == 28:
        model_dir = os.path.dirname(c)
        break
print('base', model_dir, flush=True)
assert model_dir
cfg = dict(train_csv=str(DATA_DIR / 'train.csv'), model_dir=model_dir, hidden=3584, ext_cap=250, fold_seed=FOLD_SEED, seed=SEED, layers=LAYERS,
           keep=KEEP, max_len=MAX_LEN, epochs=EPOCHS, accum=ACCUM, lr_adapter=LR_ADAPTER, lr_head=LR_HEAD, wd=WD,
           warmup=WARMUP, r=R, alpha=ALPHA, ldrop=LDROP)
open('worker.py', 'w').write(WORKER)
procs = []
for g, jl in JOBS.items():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    procs.append((g, subprocess.Popen([sys.executable, 'worker.py', json.dumps(cfg), ','.join(jl)], env=env,
                                      stdout=open(f'worker{g}.log', 'w'), stderr=subprocess.STDOUT)))
for g, p in procs:
    p.wait()
    print(f'---- worker {g} (rc {p.returncode})')
    print(open(f'worker{g}.log').read()[-4000:])
done = sorted(glob.glob('lora_out/job_*/COMPLETE.json'))
print('completed jobs', [json.load(open(d)) for d in done], f'total {time.time() - T0:.0f}s')
pd.DataFrame({'id': [], 'model': []}).to_parquet('submission.parquet')   # placeholder (training notebook)
