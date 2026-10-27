%%writefile infer.py
"""End-to-end attribution pipeline (self-contained; runs on Kaggle T4x2 or locally in --sim mode).
1. Mean-pooled hidden states from frozen LMs for train+test (computed here so precision/quantisation match).
2. Per model: LR on raw + LR on prompt-centred features (train centred by prompt_id, test by estimated cluster).
3. TF-IDF (char_wb + word/punct) prompt-centred LR.
4. Test prompts recovered by size-capped average-linkage clustering on content-word TF-IDF.
5. Sum log-probs, then Hungarian assignment inside each cluster (each model answers each prompt at most once).
"""
import os, sys, time, gc, argparse, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, scipy.sparse as sp, torch
from scipy.optimize import linear_sum_assignment

LABELS = ['claude-fable-5','claude-opus-4-6','claude-opus-4-8','claude-opus-5','claude-sonnet-5',
 'gemini-3.1-pro-preview','gemini-3.6-flash','gemini-3.7-flash','gemini-3.8-flash','glm-5.3','glm-5.3-flash',
 'gpt-5.5','gpt-5.6-luna','gpt-5.6-sol','gpt-5.6-terra','gpt-6-astra','kimi-k3','qwen3.8-max']
L2I = {l: i for i, l in enumerate(LABELS)}

p = argparse.ArgumentParser()
p.add_argument('--data', default='/kaggle/input/competitions/name-that-llm')
p.add_argument('--models', default='')   # json list of {"path":..,"layers":[fractions],"quant":"fp16|4bit|8bit","bs":16}
p.add_argument('--comp_path', default=''); p.add_argument('--comp_w', type=float, default=0.0)
p.add_argument('--fake_test', type=int, default=0, help='timing probe: score N pool passages as a stand-in test')
p.add_argument('--atlas_probe', type=int, default=0, help='timing probe: stop ATLAS after N passages')
p.add_argument('--sim', type=str, default='', help='comma list of folds held out of train.csv as fake test')
p.add_argument('--out', default='submission.parquet')
p.add_argument('--cache', default='cache')
p.add_argument('--dtype', default='fp16')
p.add_argument('--C', type=float, default=0.005)
p.add_argument('--tfidf_w', type=float, default=1.0)
p.add_argument('--distort_w', type=float, default=0.0, help='weight of topic-masked (text distortion) TF-IDF member')
p.add_argument('--adapt_thr', type=float, default=0.15, help='cluster cut (avg cosine sim on content-word TF-IDF)')
p.add_argument('--adapt_w', type=float, default=0.0, help='weight of centred twin evidence on rows in clusters of size>=2 (0=off)')
p.add_argument('--post', type=int, default=0, help='1: spaced em-dash guard + Hungarian on dense test clusters (12-18)')
p.add_argument('--atlas_dir', default=''); p.add_argument('--atlas_base', default=''); p.add_argument('--atlas_w', default='0', help='mix weight per bundle (comma list or one value)')
p.add_argument('--atlas_margin', type=float, default=0.0, help='>0: score ATLAS only on rows whose top-2 margin is below this')
p.add_argument('--atlas_bs', type=int, default=4); p.add_argument('--atlas_chunk', type=int, default=3); p.add_argument('--atlas_budget', type=float, default=8.3 * 3600)
p.add_argument('--sk_div', type=int, default=1, help='1: divide blend by #members before Sinkhorn (ours); 0: raw blend (public notebook)')
p.add_argument('--sinkhorn', type=float, default=0.0, help='alpha of Sinkhorn class-marginal balancing (0=off)')
p.add_argument('--sim_k', default='', help="sim only: keep 'lo-hi' random passages per held-out prompt")
p.add_argument('--extra', default='', help='parquet/csv (text, model) of matched external passages, used by TF-IDF/distort members')
p.add_argument('--extra_w', type=float, default=0.5)
p.add_argument('--svc_w', type=float, default=0.0, help='4-view n-gram LinearSVC on train + --pool')
p.add_argument('--pool', default='')
p.add_argument('--save_lp', default='', help='also save normalised blend log-probs (n x 18, LABELS order) to this .npy')
p.add_argument('--perpassage', type=int, default=1, help='1: test passages treated independently (no clustering/centring/assignment)')
a, _ = p.parse_known_args()
os.makedirs(a.cache, exist_ok=True)
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False   # fp32 sums inside fp16 matmuls (free on T4)
T0 = time.time()
def log(*x): print(f'[{time.time()-T0:7.0f}s]', *x, flush=True)

# ---------------- data
train = pd.read_csv(os.path.join(a.data, 'train.csv'))
if a.sim:
    rng = np.random.RandomState(0); pids = np.array(sorted(train.prompt_id.unique())); rng.shuffle(pids)
    pf = {q: i % 5 for i, q in enumerate(pids)}; fo = train.prompt_id.map(pf).values
    hold = np.isin(fo, [int(x) for x in a.sim.split(',')])
    if a.sim_k:
        lo_, hi_ = map(int, a.sim_k.split('-')); keep = np.zeros(len(train), bool); idx_h = np.where(hold)[0]
        for g in np.unique(train.prompt_id.values[idx_h]):
            ii = idx_h[train.prompt_id.values[idx_h] == g]; keep[rng.choice(ii, min(len(ii), rng.randint(lo_, hi_ + 1)), replace=False)] = True
        hold = keep | (hold & False)
    test = train[hold].drop(columns=['model']).reset_index(drop=True); test_truth = train[hold].reset_index(drop=True)
    hold_any = np.isin(fo, [int(x) for x in a.sim.split(',')])
    train = train[~hold_any].reset_index(drop=True)
elif a.fake_test:
    test = pd.read_parquet(a.pool).sample(a.fake_test, random_state=0)[['text']].reset_index(drop=True)
    test['id'] = [f'fake{i}' for i in range(len(test))]
else:
    test = pd.read_csv(os.path.join(a.data, 'test.csv'))
train['text'] = train.text.fillna(''); test['text'] = test.text.fillna('')
y = train.model.map(L2I).values; Gtr = train.prompt_id.values
log('train', len(train), 'test', len(test))

# ---------------- utils
def center_groups(X, groups):
    X = X.copy()
    for g in np.unique(groups):
        i = groups == g; X[i] -= X[i].mean(0)
    return X

def torch_lr(Xtr, ytr, Xte, C, iters=200):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    Xtr = torch.as_tensor(Xtr, dtype=torch.float32, device=dev); Xte = torch.as_tensor(Xte, dtype=torch.float32, device=dev)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    yt = torch.as_tensor(ytr, dtype=torch.long, device=dev)
    W = torch.zeros(Xtr.shape[1], 18, device=dev, requires_grad=True); b = torch.zeros(18, device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=iters, history_size=20, line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Xtr @ W + b, yt, reduction='sum') + 0.5 / C * (W ** 2).sum()
        loss.backward(); return loss
    opt.step(closure)
    with torch.no_grad(): return torch.log_softmax(Xte @ W + b, -1).cpu().numpy()

def capped_agglo(S, cap=18, thr=0.02):
    """Average linkage (Lance-Williams), never merging past `cap` members; stop when best avg sim < thr."""
    n = len(S); A = S.astype(np.float32).copy(); np.fill_diagonal(A, -np.inf)
    size = np.ones(n, np.int32); alive = np.ones(n, bool); lab = np.arange(n)
    while True:
        ok = (size[:, None] + size[None, :] <= cap) & alive[:, None] & alive[None, :]
        M = np.where(ok, A, -np.inf); k = int(np.argmax(M)); i, j = divmod(k, n)
        if not np.isfinite(M[i, j]) or M[i, j] < thr: break
        new = (size[i] * A[i] + size[j] * A[j]) / (size[i] + size[j])
        A[i] = new; A[:, i] = new; A[i, i] = -np.inf; A[j] = -np.inf; A[:, j] = -np.inf
        size[i] += size[j]; alive[j] = False; lab[lab == j] = i
    return lab

def hungarian(lp, groups):
    pred = lp.argmax(1).copy()
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]; r, c = linear_sum_assignment(-lp[idx]); pred[idx[r]] = c
    return pred

# ---------------- 1. recover test prompts
from sklearn.feature_extraction.text import TfidfVectorizer
if a.perpassage and a.adapt_w <= 0:
    Gte = np.arange(len(test))
else:
  try:
    v = TfidfVectorizer(sublinear_tf=True, stop_words='english', min_df=2, max_df=0.05)
    Xc = v.fit_transform(test.text.values)
    Gte = capped_agglo((Xc @ Xc.T).toarray(), 18, a.adapt_thr if a.perpassage else 0.02)
  except ValueError as e:  # tiny interactive test file: nothing survives df pruning
    log('clustering fallback:', e); Gte = np.arange(len(test))
log('test clusters', len(np.unique(Gte)), 'sizes', np.bincount(np.bincount(Gte)[np.bincount(Gte) > 0]).tolist())
if a.sim:
    from sklearn.metrics import adjusted_rand_score
    log('cluster ARI vs truth', round(adjusted_rand_score(test_truth.prompt_id, Gte), 4))

LP = {}
EXT_T, EXT_Y = np.array([], dtype=object), np.array([], dtype=int)
if a.extra:
    _E = pd.read_parquet(a.extra) if a.extra.endswith('.parquet') else pd.read_csv(a.extra)
    _E = _E[_E.model.isin(LABELS)]; EXT_T = _E.text.astype(str).values; EXT_Y = _E.model.map(L2I).values
    log('extra rows', len(EXT_T))
FIT_T = np.concatenate([train.text.values, EXT_T]); FIT_Y = np.concatenate([y, EXT_Y])
FIT_W = np.concatenate([np.ones(len(train)), np.full(len(EXT_T), a.extra_w)])
# ---------------- 2. TF-IDF centred LR
from sklearn.linear_model import LogisticRegression
mats_tr, mats_te = [], []
for kw in [dict(analyzer='char_wb', ngram_range=(2, 5), min_df=3, max_features=60000),
           dict(token_pattern=r"\w+|[^\w\s]", ngram_range=(1, 2), min_df=3, max_features=30000)]:
    vec = TfidfVectorizer(sublinear_tf=True, lowercase=False, **kw).fit(FIT_T if a.perpassage else train.text.values)
    mats_tr.append(vec.transform(FIT_T if a.perpassage else train.text.values)); mats_te.append(vec.transform(test.text.values))
if a.perpassage:
    LP['tfidf'] = LogisticRegression(C=20, max_iter=3000).fit(sp.hstack(mats_tr).tocsr(), FIT_Y, sample_weight=FIT_W).predict_log_proba(sp.hstack(mats_te).tocsr()) * a.tfidf_w
else:
    Xtr = center_groups(sp.hstack(mats_tr).toarray().astype(np.float32), Gtr)
    Xte = center_groups(sp.hstack(mats_te).toarray().astype(np.float32), Gte)
    LP['tfidf'] = LogisticRegression(C=5, max_iter=3000).fit(Xtr, y).predict_log_proba(Xte) * a.tfidf_w
    del Xtr, Xte
gc.collect(); log('tfidf done')
if a.svc_w > 0:  # public 4-view n-gram LinearSVC (char / word+punct / shape / function words) on train + pool
    import re as _re
    from scipy.sparse import hstack as _hs
    from sklearn.svm import LinearSVC
    _FW = set("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little".split())
    _shape = lambda t: _re.sub(r"[^\W\d_]+", lambda m_: "X" if m_[0].isupper() else "x", t)
    _func = lambda t: " ".join(w.lower() if w.lower() in _FW or not w.isalnum() else "CONTENT" for w in _re.findall(r"\w+|[^\w\s]", t))
    _P = pd.read_parquet(a.pool); _P = _P[_P.model.isin(LABELS)]
    _fit = list(train.text.values) + list(_P.text.astype(str).values); _fy = np.r_[y, _P.model.map(L2I).values]
    _A, _B = [], []
    for view, lo, hi, mf, wv in [("char", 3, 5, 250000, 1.0), ("word", 1, 3, 180000, 0.7), ("shape", 2, 6, 60000, 0.5), ("function", 1, 3, 50000, 0.5)]:
        prep = _shape if view == "shape" else _func if view == "function" else (lambda s_: s_)
        an = "char" if view == "shape" else "word" if view == "function" else view
        o = dict(analyzer=an, ngram_range=(lo, hi), lowercase=False, sublinear_tf=True, min_df=2, max_features=mf, dtype=np.float32)
        if an == "word": o["token_pattern"] = r"(?u)\b\w+\b|[^\w\s]"
        _v = TfidfVectorizer(**o); _A.append(_v.fit_transform([prep(t) for t in _fit]) * wv); _B.append(_v.transform([prep(t) for t in test.text.values]) * wv)
    _d = LinearSVC(C=0.5, random_state=42, max_iter=5000, class_weight='balanced').fit(_hs(_A, format='csr'), _fy).decision_function(_hs(_B, format='csr'))
    LP['svc4'] = (_d - _d.max(1, keepdims=True) - np.log(np.exp(_d - _d.max(1, keepdims=True)).sum(1, keepdims=True))) * a.svc_w
    del _A, _B; gc.collect(); log('svc4 done')
if a.distort_w > 0:  # Stamatatos-style text distortion: keep top-100 frequent words + punctuation, mask the rest
    import re, collections
    wre = re.compile(r"[A-Za-z’']+")
    cnt = collections.Counter(w.lower() for t in FIT_T for w in wre.findall(t))
    keep = {w for w, _ in cnt.most_common(100)}
    dist = lambda arr: [wre.sub(lambda m_: m_.group(0) if m_.group(0).lower() in keep else '*', t) for t in arr]
    Dtr, Dte = dist(FIT_T), dist(test.text.values)
    vs = [TfidfVectorizer(sublinear_tf=True, lowercase=False, analyzer='char', ngram_range=(2, 6), min_df=2),
          TfidfVectorizer(sublinear_tf=True, lowercase=True, token_pattern=r"[^\s]+", ngram_range=(1, 3), min_df=2)]
    Xa = sp.hstack([v_.fit_transform(Dtr) for v_ in vs]).tocsr(); Xb = sp.hstack([v_.transform(Dte) for v_ in vs]).tocsr()
    LP['distort'] = LogisticRegression(C=10, max_iter=3000).fit(Xa, FIT_Y, sample_weight=FIT_W).predict_log_proba(Xb) * a.distort_w
    log('distort done')

# ---------------- 3. LM embeddings
from transformers import AutoTokenizer, AutoModel
@torch.no_grad()
def embed(spec, texts):
    tok = AutoTokenizer.from_pretrained(spec['path'])
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    kw = dict(device_map='auto')
    q = spec.get('quant', 'fp16')
    if q == '4bit':
        from transformers import BitsAndBytesConfig
        kw['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.float16)
    elif q == '8bit':
        from transformers import BitsAndBytesConfig
        kw['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)
    kw['dtype'] = torch.bfloat16 if q == 'bf16' else torch.float16
    m = AutoModel.from_pretrained(spec['path'], **kw).eval()
    enc = tok(list(texts), truncation=True, max_length=512)['input_ids']
    order = np.argsort([len(x) for x in enc]); out = None; bs = spec.get('bs', 16); nan_rows = 0
    first = next(m.parameters()).device
    for i in range(0, len(order), bs):
        b = order[i:i + bs]; L = max(len(enc[j]) for j in b)
        ids = torch.full((len(b), L), tok.pad_token_id); att = torch.zeros((len(b), L), dtype=torch.long)
        for r, j in enumerate(b): ids[r, :len(enc[j])] = torch.tensor(enc[j]); att[r, :len(enc[j])] = 1
        hs = m(input_ids=ids.to(first), attention_mask=att.to(first), output_hidden_states=True).hidden_states
        nL = len(hs) - 1; sel = spec['layers_abs'] if 'layers_abs' in spec else [max(1, round(f * nL)) for f in spec['layers']]
        pooled = []
        for l in sel:
            h = hs[l].float(); w = att.to(h.device).unsqueeze(-1).float()
            pooled.append(((h * w).sum(1) / w.sum(1)).cpu())
        pooled = torch.cat(pooled, 1).numpy()
        nan_rows += int((~np.isfinite(pooled)).any(1).sum())
        if out is None: out = np.zeros((len(texts), pooled.shape[1]), np.float32); log('  layers', sel, 'of', nL)
        out[b] = pooled
    del m; gc.collect(); torch.cuda.empty_cache()
    if nan_rows: log('  WARNING non-finite rows:', nan_rows)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

TPL = 'Identify which AI language model wrote the following passage, based only on its writing style.\n\nPassage:\n{}\n\nThe author model is:'
@torch.no_grad()
def lora_predict(spec, texts):
    """LoRA classifier WITHOUT peft: merge W += (B @ A) * alpha/r into the base weights, load the trained score head."""
    from transformers import AutoModelForSequenceClassification
    from safetensors.torch import load_file
    tok = AutoTokenizer.from_pretrained(spec['path'])
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    m = AutoModelForSequenceClassification.from_pretrained(spec['path'], num_labels=18, dtype=torch.float16, device_map='auto')
    m.config.pad_token_id = tok.pad_token_id
    cfg = json.load(open(os.path.join(spec['adapter'], 'adapter_config.json')))
    scale = cfg['lora_alpha'] / cfg['r']
    sd = load_file(os.path.join(spec['adapter'], 'adapter_model.safetensors'))
    mods = dict(m.named_modules()); n_merged = 0; head = None
    for k, v in sd.items():
        key = k.replace('base_model.model.', '', 1)
        if '.lora_A.' in key:
            mod_name = key.split('.lora_A.')[0]; B = sd[k.replace('.lora_A.', '.lora_B.')]
            lin = mods[mod_name]; W = lin.weight
            W += ((B.to(W.device, torch.float32) @ v.to(W.device, torch.float32)) * scale).to(W.dtype); n_merged += 1
        elif key.startswith('score.') and 'weight' in key and 'original_module' not in key:
            head = v
    assert head is not None and n_merged > 0, (n_merged, [k for k in sd if 'score' in k])
    m.score.weight.copy_(head.to(m.score.weight.device, m.score.weight.dtype))
    log(f'  merged {n_merged} LoRA pairs (scale {scale}), head {tuple(head.shape)}')
    del mods, sd, lin, W, B, head; gc.collect()  # drop every reference to the model's modules
    m.eval()
    enc = tok(list(texts), truncation=True, max_length=512)['input_ids']
    order = np.argsort([len(x) for x in enc]); out = np.zeros((len(texts), 18), np.float32); bs = spec.get('bs', 16)
    first = next(m.parameters()).device
    for i in range(0, len(order), bs):
        b = order[i:i + bs]; L = max(len(enc[j]) for j in b)
        ids = torch.full((len(b), L), tok.pad_token_id); att = torch.zeros((len(b), L), dtype=torch.long)
        for r, j in enumerate(b): ids[r, :len(enc[j])] = torch.tensor(enc[j]); att[r, :len(enc[j])] = 1
        lo = m(input_ids=ids.to(first), attention_mask=att.to(first)).logits.float()
        out[b] = torch.log_softmax(lo, -1).cpu().numpy()
    del m; gc.collect(); torch.cuda.empty_cache()
    return out

@torch.no_grad()
def l24_predict(spec, texts, groups=None, train_texts=None):
    """LoRA24-recipe member(s): truncated base loaded ONCE; for each adapter dir in spec['adapters'] (or spec['adapter'])
    merge W += B A * s, compute masked mean-pooled multi-layer features, read plain (+ centred) heads, then unmerge.
    Returns log-probs averaged over adapters (plain, centred-or-None)."""
    from transformers import AutoConfig
    from safetensors.torch import load_file
    adapters = spec.get('adapters') or [spec['adapter']]
    tok = AutoTokenizer.from_pretrained(spec['path'])
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    cfg = AutoConfig.from_pretrained(spec['path']); tc = cfg.text_config if hasattr(cfg, 'text_config') else cfg
    tc.num_hidden_layers = spec['keep']; tc.use_cache = False
    if getattr(tc, 'layer_types', None) is not None: tc.layer_types = tc.layer_types[:spec['keep']]
    m = AutoModel.from_pretrained(spec['path'], config=cfg, dtype=torch.float16, device_map='auto').eval()
    if spec.get('no_final_norm'): m.norm = torch.nn.Identity()
    mods = dict(m.named_modules()); first = next(m.parameters()).device; L = spec['layers']
    _eos = spec.get('eos_id')   # checkpoints trained with an end token appended (Nishant's) need the same id at inference
    _tk = lambda tx: [x + ([_eos] if _eos is not None else []) for x in tok(list(tx), truncation=True, max_length=spec.get('max_len', 384) - (1 if _eos is not None else 0))['input_ids']]
    enc = _tk(texts)
    order = np.argsort([len(x) for x in enc]); bs = spec.get('bs', 16)
    def lsm(z): return z - z.max(1, keepdims=True) - np.log(np.exp(z - z.max(1, keepdims=True)).sum(1, keepdims=True))
    P_acc, C_acc = 0.0, 0.0; FEATS = None; OVF = [0]
    if train_texts is not None:
        enc_tr = _tk(train_texts)
    def featurize(enc_, norms, hid):
        order_ = np.argsort([len(x) for x in enc_]); Z_ = np.zeros((len(enc_), hid * len(L)), np.float32)
        for i in range(0, len(order_), bs):
            b = order_[i:i + bs]; Lm = max(len(enc_[j]) for j in b)
            ids = torch.full((len(b), Lm), tok.pad_token_id); att = torch.zeros((len(b), Lm), dtype=torch.long)
            for r, j in enumerate(b): ids[r, :len(enc_[j])] = torch.tensor(enc_[j]); att[r, :len(enc_[j])] = 1
            hs = m(input_ids=ids.to(first), attention_mask=att.to(first), output_hidden_states=True, use_cache=False).hidden_states
            if not all(torch.isfinite(hs[l]).all() for l in L):
                m.to(torch.bfloat16); hs = m(input_ids=ids.to(first), attention_mask=att.to(first), output_hidden_states=True, use_cache=False).hidden_states; m.to(torch.float16); OVF[0] += 1
            mask = att.unsqueeze(-1).float(); mask[:, 0] = 0
            Z_[b] = torch.cat([nm((hs[l].float().cpu() * mask).sum(1) / mask.sum(1)) for l, nm in zip(L, norms)], -1).numpy()
        return Z_
    for ai, ad in enumerate(adapters):
        adp = os.path.join(ad, 'adapter') if os.path.exists(os.path.join(ad, 'adapter', 'adapter_config.json')) else ad
        acfg = json.load(open(os.path.join(adp, 'adapter_config.json'))); scale = acfg['lora_alpha'] / acfg['r']
        sd = load_file(os.path.join(adp, 'adapter_model.safetensors')); deltas = []
        for k, v in sd.items():
            if '.lora_A.' not in k: continue
            name = k.replace('base_model.model.', '', 1).split('.lora_A.')[0]
            if name not in mods and ('model.' + name) in mods: name = 'model.' + name
            W = mods[name].weight; B = sd[k.replace('.lora_A.', '.lora_B.')]
            W += ((B.to(W.device, torch.float32) @ v.to(W.device, torch.float32)) * scale).to(W.dtype)
            deltas.append((W, B, v))   # keep only the low-rank factors; the dense delta is recomputed to unmerge
        if os.path.exists(os.path.join(ad, 'heads.pt')): H = torch.load(os.path.join(ad, 'heads.pt'), map_location='cpu')
        else: H = dict(load_file(os.path.join(ad, 'heads.safetensors')))
        if os.path.exists(os.path.join(ad, 'meta.json')):   # remap head rows from the checkpoint's label order to ours
            _lab = json.load(open(os.path.join(ad, 'meta.json')))['labels']; _perm = [_lab.index(l) for l in LABELS]
            for _p in ('plain', 'centered'): H[f'{_p}.weight'] = H[f'{_p}.weight'][_perm]; H[f'{_p}.bias'] = H[f'{_p}.bias'][_perm]
        hid = H['norms.0.weight'].shape[0]
        norms = [torch.nn.LayerNorm(hid) for _ in L]
        for i, nm in enumerate(norms): nm.load_state_dict({'weight': H[f'norms.{i}.weight'], 'bias': H[f'norms.{i}.bias']})
        Z = np.zeros((len(texts), hid * len(L)), np.float32)
        for i in range(0, len(order), bs):
            b = order[i:i + bs]; Lm = max(len(enc[j]) for j in b)
            ids = torch.full((len(b), Lm), tok.pad_token_id); att = torch.zeros((len(b), Lm), dtype=torch.long)
            for r, j in enumerate(b): ids[r, :len(enc[j])] = torch.tensor(enc[j]); att[r, :len(enc[j])] = 1
            hs = m(input_ids=ids.to(first), attention_mask=att.to(first), output_hidden_states=True, use_cache=False).hidden_states
            if not all(torch.isfinite(hs[l]).all() for l in L):   # fp16 overflow (un-normed deep states): redo this batch in bf16
                m.to(torch.bfloat16); hs = m(input_ids=ids.to(first), attention_mask=att.to(first), output_hidden_states=True, use_cache=False).hidden_states; m.to(torch.float16); OVF[0] += 1
            mask = att.unsqueeze(-1).float(); mask[:, 0] = 0
            Z[b] = torch.cat([nm((hs[l].float().cpu() * mask).sum(1) / mask.sum(1)) for l, nm in zip(L, norms)], -1).numpy()
        if train_texts is not None and ai == len(adapters) - 1:   # full-data adapter: features for the Gaussian head
            FEATS = (Z.copy(), featurize(enc_tr, norms, hid))
        for W, B, A_ in deltas:   # unmerge before the next adapter
            W -= ((B.to(W.device, torch.float32) @ A_.to(W.device, torch.float32)) * scale).to(W.dtype)
        torch.cuda.empty_cache()
        del deltas, sd; gc.collect()
        hd = lambda p, X: lsm(X @ H[f'{p}.weight'].float().numpy().T + H[f'{p}.bias'].float().numpy())
        P_acc = P_acc + hd('plain', Z) / len(adapters)
        if groups is not None:
            Zc = Z.copy()
            for g in np.unique(groups):
                ii = groups == g; Zc[ii] -= Zc[ii].mean(0)
            C_acc = C_acc + hd('centered', Zc) / len(adapters)
        log(f'  l24 {spec["name"]}: adapter {os.path.basename(ad)} done (bf16 redo batches {OVF[0]})')
    del m, mods; gc.collect(); torch.cuda.empty_cache()
    if train_texts is not None: return P_acc, (C_acc if groups is not None else None), FEATS
    return P_acc, (C_acc if groups is not None else None)

def gpu_mem():
    return ' '.join(f'gpu{i}:{torch.cuda.memory_allocated(i)/1e9:.1f}GB' for i in range(torch.cuda.device_count()))

specs = json.loads(a.models) if a.models else []
all_text = np.concatenate([train.text.values, test.text.values])
for spec in specs:
    if spec.get('type') == 'l24':
        log('l24', spec['name'])
        if len(test) < 50:
            _p40, _ = l24_predict(spec, train.text.values[:40]); _acc = float((_p40.argmax(1) == y[:40]).mean())
            log(f'  {spec["name"]} in-sample check acc {_acc:.2f}'); assert _acc > 0.5, f'{spec["name"]} l24 does not reproduce its fit'
        if spec.get('gauss', 0) > 0:
            plain, cent, (Zte_g, Ztr_g) = l24_predict(spec, test.text.values, Gte if a.adapt_w > 0 else None, train_texts=train.text.values)
            import gauss as _ga
            prm = _ga.fit(Ztr_g, y, Gtr, dict(red='wpca', d=128, se_shrink='lw', sp_shrink='lw', lam_floor=1e-4, n_exact=3, inj=True, iters=5, repel=True, sp_scale=1.0))
            prm['T1'] = prm['Tg'] = 3.0
            _sz = np.bincount(Gte)[Gte]; _rows = (_sz >= 2) & (_sz <= 18)
            _q = _ga.scores(Ztr_g[[0, 1]], np.zeros(2, dtype=int), prm); assert np.allclose(np.exp(_q).sum(1), 1)
            _G = np.zeros((len(test), 18))
            if _rows.any():
                _G[_rows] = _ga.scores(Zte_g[_rows], np.unique(Gte[_rows], return_inverse=True)[1], prm)
            LP[spec['name'] + '_gauss'] = _G * spec['gauss']; log(f"  gaussian head {spec['name']}: {int(_rows.sum())} grouped test rows, weight {spec['gauss']}")
        else:
            plain, cent = l24_predict(spec, test.text.values, Gte if a.adapt_w > 0 else None)
        LP[spec['name']] = plain * spec.get('w', 1.0)
        if cent is not None:
            rows = np.bincount(Gte)[Gte] >= 2
            LP[spec['name'] + '_twin'] = np.where(rows[:, None], cent, 0.0) * a.adapt_w * spec.get('w', 1.0) * spec.get('twin', 1.0)
        gc.collect(); torch.cuda.empty_cache(); log(spec['name'], 'done', gpu_mem()); continue
    if spec.get('type') == 'enc':   # fully fine-tuned encoder (DeBERTa-style): fp32, mean of the last layer, linear head
        log('enc', spec['name']); from safetensors.torch import load_file as _lf
        _ck = spec['ckpt']; _meta = json.load(open(os.path.join(_ck, 'meta.json')))
        _perm = [_meta['labels'].index(l) for l in LABELS]
        _hd = _lf(os.path.join(_ck, 'head.safetensors')); _W = _hd['weight'].float().numpy()[_perm]; _b = _hd['bias'].float().numpy()[_perm]
        _etok = AutoTokenizer.from_pretrained(os.path.join(_ck, 'encoder'))
        _enc = AutoModel.from_pretrained(os.path.join(_ck, 'encoder'), dtype=torch.float32).to('cuda:0').eval()
        def _enc_feats(texts):
            _ids = _etok(list(texts), truncation=True, max_length=_meta['max_len'])['input_ids']
            _ord = np.argsort([len(x) for x in _ids], kind='stable'); Z_ = np.zeros((len(_ids), _enc.config.hidden_size), np.float32)
            with torch.inference_mode():
                for _s in range(0, len(_ord), 32):
                    _idx = _ord[_s:_s + 32]; _bt = _etok.pad({'input_ids': [_ids[i] for i in _idx]}, return_tensors='pt').to('cuda:0')
                    _m = _bt['attention_mask'].unsqueeze(-1).float()
                    Z_[_idx] = ((_enc(**_bt).last_hidden_state * _m).sum(1) / _m.sum(1).clamp(min=1)).cpu().numpy()
            return Z_
        if len(test) < 50:
            _acc = float(((_enc_feats(train.text.values[:40]) @ _W.T + _b).argmax(1) == y[:40]).mean())
            log(f'  {spec["name"]} in-sample check acc {_acc:.2f}'); assert _acc > 0.5, spec['name']
        _zz = _enc_feats(test.text.values) @ _W.T + _b
        LP[spec['name']] = (_zz - _zz.max(1, keepdims=True) - np.log(np.exp(_zz - _zz.max(1, keepdims=True)).sum(1, keepdims=True))) * spec.get('w', 1.0)
        del _enc; gc.collect(); torch.cuda.empty_cache(); log(spec['name'], 'done', gpu_mem()); continue
    if spec.get('type') == 'head':
        log('head', spec['name']); H = np.load(spec['head'])
        Ete = embed(dict(spec, layers=sorted(float(v) for v in H['layers'])), test.text.values)  # extraction stored layers in sorted order
        if len(test) < 50:  # interactive run: in-sample sanity check on 40 train rows (head was fit on them)
            _E40 = embed(dict(spec, layers=sorted(float(v) for v in H['layers'])), train.text.values[:40])
            _acc = float(((((_E40 - H['mu']) / H['sd']) @ H['W'] + H['b']).argmax(1) == y[:40]).mean())
            log(f'  {spec["name"]} in-sample check acc {_acc:.2f}'); assert _acc > 0.5, f'{spec["name"]} head does not reproduce its fit'
        Z = (Ete - H['mu']) / H['sd']; zz = Z @ H['W'] + H['b']
        LP[spec['name']] = (zz - zz.max(1, keepdims=True) - np.log(np.exp(zz - zz.max(1, keepdims=True)).sum(1, keepdims=True))) * spec.get('w', 1.0)
        gc.collect(); torch.cuda.empty_cache(); log(spec['name'], 'done', gpu_mem()); continue
    if spec.get('type') == 'lora':
        log('lora', spec['name']); LP[spec['name']] = lora_predict(spec, test.text.values) * spec.get('w', 1.0)
        gc.collect(); torch.cuda.empty_cache(); log(spec['name'], 'done', gpu_mem()); continue
    name = spec['name']; cp = os.path.join(a.cache, f"{name}_{'sim' + a.sim + a.sim_k if a.sim else 'real'}.npy")
    if os.path.exists(cp): E = np.load(cp)
    else:
        log('embedding', name); E = embed(spec, all_text); np.save(cp, E)
    Etr, Ete = E[:len(train)], E[len(train):]
    r = torch_lr(Etr, y, Ete, spec.get('C', a.C))
    if a.perpassage:
        LP[name] = r * spec.get('w', 1.0)
        if a.adapt_w > 0:
            sizes = np.bincount(Gte)[Gte]; rows = sizes >= 2
            c = torch_lr(center_groups(Etr, Gtr), y, center_groups(Ete, Gte), spec.get('C', a.C))
            LP[name + '_twin'] = np.where(rows[:, None], c, 0.0) * a.adapt_w * spec.get('w', 1.0)
    else:
        c = torch_lr(center_groups(Etr, Gtr), y, center_groups(Ete, Gte), a.C)
        LP[name] = (r + c) * spec.get('w', 1.0)
    gc.collect(); torch.cuda.empty_cache(); log(name, 'done', gpu_mem())

# ---------------- 4. combine + assign
# typography guards: (mark, models that ever use it in train). Zero violations on the 38k matched pool for every rule.
GUARDS = [(' \u2014 ', {'claude-opus-4-6', 'claude-opus-5', 'glm-5.3', 'glm-5.3-flash'}),
          ('\u2019', {l for l in LABELS if not l.startswith(('claude', 'glm'))}),
          ('"', {'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5', 'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash', 'glm-5.3', 'glm-5.3-flash', 'gpt-5.5', 'kimi-k3'}),
          ('\u201c', {'gemini-3.8-flash', 'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra', 'kimi-k3', 'qwen3.8-max'}),
          ('\u2013', {'claude-opus-5', 'gemini-3.1-pro-preview', 'glm-5.3-flash', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'kimi-k3'})]
tot = sum(LP.values())
def atlas_scores(bundle, rows, budget):
    """Source-conditioned likelihood probabilities (public ATLAS bundle format) for test rows `rows`, in our label order.
    Rows are scored shortest-first; raises if the token-proportional projection would overrun `budget` (s since start)."""
    sys.path.insert(0, bundle)
    for _m in [m_ for m_ in list(sys.modules) if m_.startswith(('ntl_atlas', 'ntl_rlcd'))]: del sys.modules[_m]
    from ntl_atlas.model import load_model as _atlas_load
    from ntl_rlcd.data import LABELS as _alab
    from safetensors.torch import load_file as _lf
    sys.path.remove(bundle)
    _cfg = json.load(open(os.path.join(bundle, 'atlas_bundle.json')))['training_config']
    _am, _atok = _atlas_load(a.atlas_base, _alab, _cfg, device='auto')
    try:
        _am.load_trainable(_lf(os.path.join(bundle, 'adapters.safetensors'))); _am.eval()
        _enc = _atok(test.text.values[rows].tolist(), truncation=True, max_length=_cfg['max_length'])['input_ids']
        _ord = np.argsort([len(t) for t in _enc], kind='stable'); out = np.zeros((len(rows), 18), np.float32); _t1 = time.time()
        _ctok = np.cumsum(np.array([len(_enc[int(i)]) for i in _ord], np.float64))
        for _b in range(0, len(_ord), a.atlas_bs):
            _idx = _ord[_b:_b + a.atlas_bs]
            _pk = _atok.pad({'input_ids': [_enc[int(i)] for i in _idx]}, padding=True, return_tensors='pt').to(_am.device)
            with torch.inference_mode(): _p, _ = _am.score(_pk['input_ids'], _pk['attention_mask'], source_chunk=a.atlas_chunk)
            out[_idx] = _p.float().cpu().numpy(); _done = _b + len(_idx)
            if _b % (25 * a.atlas_bs) == 0:
                _proj = (time.time() - T0) + (time.time() - _t1) / _ctok[_done - 1] * (_ctok[-1] - _ctok[_done - 1]) * 1.1   # token-proportional
                log(f'atlas {os.path.basename(bundle)} {_done}/{len(_ord)} projected finish {_proj:.0f}s')
                if _done > 50 and _proj > budget: raise RuntimeError('atlas would overrun the time budget')
                if a.atlas_probe and _done >= a.atlas_probe: raise RuntimeError(f'atlas probe stop; projected {_proj:.0f}s')
        assert np.isfinite(out).all() and np.allclose(out.sum(1), 1, atol=1e-4)
        return out[:, [_alab.index(l) for l in LABELS]]
    finally:
        del _am; gc.collect(); torch.cuda.empty_cache()

COMP_P = None
if a.comp_w > 0 and a.comp_path and os.path.exists(a.comp_path):   # complementary public stack, run in its own process
    try:
        _cm = json.load(open(a.comp_path.replace('comp_logit.npy', 'comp_meta.json')))
        assert _cm['ids'] == test['id'].astype(str).tolist(), 'comp stack row order differs'
        _cl = np.load(a.comp_path)[:, [_cm['labels'].index(l) for l in LABELS]]
        COMP_P = np.exp(_cl - _cl.max(1, keepdims=True)); COMP_P /= COMP_P.sum(1, keepdims=True)
        assert np.isfinite(COMP_P).all(); log(f"comp stack loaded ({_cm['seconds']:.0f}s)")
    except Exception as _e:
        log('COMP SKIPPED:', repr(_e)); COMP_P = None
elif a.comp_w > 0: log('COMP SKIPPED: no stack output')
_bundles = [d for d in a.atlas_dir.split(',') if d]; _aw = [float(x) for x in str(a.atlas_w).split(',')] if _bundles else []
if _bundles and len(_aw) == 1: _aw = _aw * len(_bundles)
if (a.sinkhorn > 0 and len(test) >= 18) or _bundles or COMP_P is not None:
    wsum = sum(abs(v).max() > 0 for v in LP.values())
    lpn = tot / max(1, wsum) if a.sk_div else tot; P = np.exp(lpn - lpn.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
    if COMP_P is not None: P = (1 - a.comp_w) * P + a.comp_w * COMP_P; P /= P.sum(1, keepdims=True)
    if _bundles and sum(_aw) > 0:
        _s = np.sort(P, 1); _margin = _s[:, -1] - _s[:, -2]
        rows = np.where(_margin < a.atlas_margin)[0] if a.atlas_margin > 0 and len(test) >= 50 else np.arange(len(test))
        log(f'atlas on {len(rows)}/{len(test)} rows (top-2 margin < {a.atlas_margin})')
        mixed = P.copy(); _wleft = 1.0
        for _bd, _w in zip(_bundles, _aw):
            try:
                A_ = atlas_scores(_bd, rows, a.atlas_budget)
                mixed[rows] = mixed[rows] - _w * P[rows] + _w * A_; log(f'atlas {_bd} mixed at {_w}', gpu_mem())
            except Exception as _e:
                log('ATLAS SKIPPED:', _bd, repr(_e))
        P = mixed / mixed.sum(1, keepdims=True)
    Q = P.copy()
    if not (a.sinkhorn > 0 and len(test) >= 18): tot = np.log(P + 1e-12)
    if a.sinkhorn > 0 and len(test) >= 18:
        for _ in range(100):
            Q *= (len(Q) / 18) / Q.sum(0, keepdims=True); Q /= Q.sum(1, keepdims=True)
        tot = (1 - a.sinkhorn) * np.log(P + 1e-12) + a.sinkhorn * np.log(Q + 1e-12)
if a.post and a.perpassage:
    _nfix = 0
    for _mark, _allowed in GUARDS:
        _forb = np.array([l not in _allowed for l in LABELS])
        for _i, _t in enumerate(test.text.values):
            if _mark in _t and LABELS[int(tot[_i].argmax())] not in _allowed:
                tot[_i, _forb] = -1e9; _nfix += 1
    _sz = np.bincount(Gte)[Gte]; _nh = 0
    for _g in np.unique(Gte[(_sz >= 12) & (_sz <= 18)]):
        _ii = np.where(Gte == _g)[0]; _r, _c = linear_sum_assignment(-tot[_ii]); _nh += int((tot[_ii].argmax(1)[_r] != _c).sum())
        tot[_ii[_r]] = -1e9; tot[_ii[_r], _c] = 0.0
    log(f'post: em-dash guard fixed {_nfix}, dense-cluster Hungarian changed {_nh}')
pred = tot.argmax(1) if a.perpassage else hungarian(tot, Gte)
if a.sim:
    from sklearn.metrics import balanced_accuracy_score as bas
    yt = test_truth.model.map(L2I).values
    for k, v_ in LP.items(): log(f'  {k:12s} arg {bas(yt, v_.argmax(1)):.4f} hung {bas(yt, hungarian(v_, Gte)):.4f}')
    log(f'ENSEMBLE arg {bas(yt, tot.argmax(1)):.4f} | hung(est clusters) {bas(yt, pred):.4f} | hung(true prompts) {bas(yt, hungarian(tot, test_truth.prompt_id.values)):.4f}')
    np.save(os.path.join(a.cache, f'sim{a.sim}_lp.npy'), tot)
if a.save_lp:
    wtot = sum(abs(float(s_.get('w', 1.0))) for s_ in specs) + (a.tfidf_w if a.tfidf_w > 0 else 0) + (a.distort_w if a.distort_w > 0 else 0) + (a.svc_w if a.svc_w > 0 else 0)
    from scipy.special import log_softmax as _lsm
    np.save(a.save_lp, _lsm(tot / max(wtot, 1e-6), axis=1).astype(np.float32)); log('saved', a.save_lp, 'norm', wtot)
sub = pd.DataFrame({'id': test.id.values, 'model': [LABELS[i] for i in pred]})
assert sub.model.isin(LABELS).all() and sub.id.is_unique and len(sub) == len(test)
sub.to_parquet(a.out, index=False); log('wrote', a.out, sub.model.value_counts().to_dict())
