%%writefile comp_stack.py
"""Complementary frozen-feature stack (public Apache-2.0 notebooks by prvsiyan / lucasboesen / vnton37nguyn / kirilltym,
as embedded in the public 0.6xx ATLAS notebooks). Run as a separate process; writes its raw log-probability blend."""
# ============================================================ 0. configuration
import gc, glob, json, os, pathlib, re, time
import numpy as np, pandas as pd
try:
    import torch
except ImportError:   # local smoke without torch
    torch = None
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse import csr_matrix, diags, hstack, vstack
from scipy.spatial.distance import squareform
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

SMOKE = os.environ.get('NTL_SMOKE') == '1'   # local code-path check: pseudo features instead of checkpoints, CPU
T0 = time.time()
LABELS = sorted(['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
                 'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
                 'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
                 'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3'])
K = len(LABELS)
LAB2I = {l: i for i, l in enumerate(LABELS)}
FAMILY_IDX = {f: np.array([i for i, l in enumerate(LABELS) if l.startswith(f)]) for f in ('claude', 'gpt', 'gemini', 'glm')}

# ---- test-pool grouping
GROUP_THRESH, GROUP_LAYER, GROUP_MIN, GROUP_MAX = 0.60, 24, 2, 18   # mixed content + GLM-L24 distance, average linkage
STYLE_GROUP_THRESH = 0.90                                           # stylegram's own median-normalised content distance

# ---- blend weights: log-probability multipliers of the 0.563 notebook (its L-BFGS fit on prompt-grouped OOF)
W = {'stylegram': 1.516, 'stylometry_lgb': 0.063, 'ultrabert': 0.095,
     'q3_8b': 0.0, 'q25_7b': 0.0, 'q3_4b': 0.0, 'glm9b': 0.0, 'debv3l': 0.05, 'electl': 0.032, 'desk': 0.132,
     'lora24_public': 0.177, 'lora24_seed7': 0.113, 'lora24_claude': 0.177, 'lora20_q25_instyle': 0.17,
     'tokattn_q35': 0.153, 'tokattn_deb': 0.185}
TWIN_W = {'q3_8b': 1.5 * 0.219, 'q25_7b': 1.5 * 0.141, 'q3_4b': 1.5 * 0.083, 'glm9b': 1.5 * 0.110, 'debv3l': 0.25}
LORA_TWIN_W = {'lora24_public': 0.15, 'lora24_claude': 0.15}
GATE_LAM = {'claude_style': 1.5, 'claude_lora': 1.0, 'family': 0.25}
BALANCE_ALPHA = 0.5
EXTRA_MIN_LABELS, EXTRA_SHARE = 12, 0.5

# ---- the addition: shared-prompt Gaussian group heads on four measured views, weight = the centred twin of the same backbone,
#      reduction / temperatures from our production fit (research/gaussian/production_weights_v8g.json, 135 training prompts)
GAUSS_HEADS = {'gauss_q3_8b_L22': dict(src='q3_8b', part='L22_mean', d=384, T1=2.351, Tg=2.306, w=TWIN_W['q3_8b']),
               'gauss_glm9b_L8': dict(src='glm9b', part='L8_mean', d=384, T1=2.722, Tg=2.596, w=TWIN_W['glm9b']),
               'gauss_q3_4b_L4': dict(src='q3_4b', part='L4_mean', d=384, T1=2.725, Tg=2.578, w=TWIN_W['q3_4b']),
               'gauss_debv3l_L24': dict(src='debv3l', part='L24_mean', d=384, T1=3.112, Tg=3.069, w=TWIN_W['debv3l'])}
GAUSS_SETTING = dict(red='wpca', se_shrink='lw', sp_shrink='lw', lam_floor=1e-4, n_exact=3, inj=True, iters=5, repel=True, sp_scale=1.0)

# ---- member specifications
LM_HEADS = [  # key, model_type, hidden size, parts of the plain head [(layer, pool)], extra parts extracted in the same pass, C
    ('glm9b', 'glm4', 4096, [(8, 'mean'), (3, 'mean')], [(GROUP_LAYER, 'mean')], 0.0025),   # first: its L24 builds the groups
    ('q3_8b', 'qwen3', 4096, [(7, 'mean'), (22, 'mean')], [], 0.0025),
    ('q25_7b', 'qwen2', 3584, [(6, 'mean'), (20, 'mean')], [], 0.0025),
    ('q3_4b', 'qwen3', 2560, [(4, 'style'), (7, 'style')], [(4, 'mean')], 0.0025)]
ENC_HEADS = [  # key, input directory name, parts, C
    ('debv3l', 'deberta-v3-large', [(24, 'mean'), (6, 'mean')], 0.005),
    ('electl', 'electra-large-discriminator', [(21, 'mean'), (24, 'style')], 0.005),
    ('desk', 'desklib-detector', [(23, 'mean'), (23, 'style')], 0.005)]
LORA_RUNS = [  # (trunk model_type, hidden, kept layers), pooled layers, [(key, checkpoint marker, norms file, anchor features file)]
    (('qwen3', 4096, 24), [2, 4, 6, 8, 12, 16, 20, 24], [
        ('lora24_public', '**/lora24/final/COMPLETE.json', 'heads.safetensors', 'anchor_features.npz'),
        ('lora24_seed7', '**/name-that-llm-train-lora24-s7/lora_out/job_final/COMPLETE.json', 'norms.safetensors', 'valid_features.npz'),
        ('lora24_claude', '**/ntl-qwen3-8b-lora-claude-head/**/final/COMPLETE.json', 'heads.safetensors', 'anchor_features.npz')]),
    (('qwen2', 3584, 20), [2, 4, 6, 8, 10, 12, 16, 20], [
        ('lora20_q25_instyle', '**/ntl-train-lora-qwen25-instyle/lora_out/job_final/COMPLETE.json', 'norms.safetensors', 'valid_features.npz')])]
LORA_MAX_LEN, LORA_BS = 447, 8
TOKATTN_HEADS = [('tokattn_q35', 'tokattn_final.pt', 'qwen3_5'), ('tokattn_deb', 'tokattn_debl23_final.pt', 'deberta')]
if os.environ.get('NTL_DROP_Q35') == '1':   # Qwen3.5-9B token-attention member: 1.7-2.7 h fixed cost on Kaggle, weight 0.153 of ~3.3
    TOKATTN_HEADS = [t for t in TOKATTN_HEADS if t[0] != 'tokattn_q35']

# ---- data
if SMOKE:
    _all = pd.read_csv(os.environ['NTL_DATA'] + '/train.csv')
    _p = _all['prompt_id'].drop_duplicates().sample(frac=1, random_state=0).tolist()
    train = _all[_all['prompt_id'].isin(_p[:30])].reset_index(drop=True)
    test = _all[_all['prompt_id'].isin(_p[30:42])].sample(frac=1, random_state=1).reset_index(drop=True)
    test = pd.concat([test[test['prompt_id'] == p].head(k) for p, k in zip(_p[30:42], [1, 1, 2, 2, 2, 3, 3, 4, 6, 9, 18, 18])]).reset_index(drop=True)
    if os.environ.get('NTL_TINY') == '1':
        test = test.head(3).reset_index(drop=True)
    SMOKE_Y = test['model'].map(LAB2I).to_numpy(dtype=int)
else:
    DATA_DIR = next(p for p in [pathlib.Path('/kaggle/input/competitions/name-that-llm'), pathlib.Path('/kaggle/input/name-that-llm')]
                    if (p / 'train.csv').exists())
    train, test = pd.read_csv(DATA_DIR / 'train.csv'), pd.read_csv(DATA_DIR / 'test.csv')
    if os.environ.get('NTL_FAKE_TEST'):   # timing probe: pool passages as a stand-in test (same rows/seed as infer.py --fake_test)
        test = pd.read_parquet(os.environ['NTL_POOL']).sample(int(os.environ['NTL_FAKE_TEST']), random_state=0)[['text']].reset_index(drop=True)
        test['id'] = [f'fake{i}' for i in range(len(test))]
test['text'] = test['text'].fillna('').astype(str)
TR, TE = train['text'].tolist(), test['text'].tolist()
y = train['model'].map(LAB2I).to_numpy(dtype=int)
PROMPT = train['prompt_id'].to_numpy(dtype=object)
TINY = len(TE) < 50   # the interactive 3-row test: exercise the group paths on training rows so the hidden rerun cannot hit untested code
DEV = 'cuda' if torch is not None and torch.cuda.is_available() else 'cpu'
print(f'train {train.shape}  test {test.shape}  gpus {torch.cuda.device_count() if torch is not None else 0}  smoke {SMOKE}', flush=True)

LOGIT = np.zeros((len(TE), K))           # the running log-probability blend


def add_member(key, prob, rows=None, weight=None):
    """Add weight * log(prob) to the blend (all rows, or the given row subset)."""
    w = W[key] if weight is None else weight
    lp = np.log(np.clip(prob, 1e-9, 1.0))
    if rows is None:
        LOGIT[:] += w * lp
    else:
        LOGIT[rows] += w * lp
    print(f'  + {key:20s} weight {w:.3f} on {len(TE) if rows is None else int(np.sum(rows))} rows  ({time.time() - T0:.0f}s)', flush=True)


def softmax_rows(S):
    S = S - S.max(1, keepdims=True)
    P = np.exp(S)
    return P / P.sum(1, keepdims=True)


def full_prob(P_part, classes):
    """Expand a (n, len(classes)) probability block to (n, K) on the global label order."""
    P = np.zeros((len(P_part), K))
    P[:, classes] = P_part
    return P


def group_centre_dense(X, groups):
    X = np.asarray(X, dtype=np.float32).copy()
    for g in np.unique(groups):
        i = groups == g
        X[i] -= X[i].mean(0)
    return X


def self_test_rows():
    """Training rows of the first three prompts, grouped by their true prompt (used only when the test file is tiny)."""
    idx = np.where(np.isin(PROMPT, pd.unique(PROMPT)[:3]))[0]
    return idx, np.unique(PROMPT[idx], return_inverse=True)[1]

# ============================================================ 1. grouping the test pool
def average_linkage_groups(D, thresh):
    """Cluster ids (1-based) from a square distance matrix, average linkage, cut at thresh."""
    if len(D) < 2:
        return np.ones(len(D), dtype=int)
    D = np.asarray(D, dtype=np.float64).copy()
    np.fill_diagonal(D, 0.0)
    return fcluster(linkage(squareform(D, checks=False), method='average'), thresh, criterion='distance')


def content_cosine_distance(texts, max_features=None):
    X = TfidfVectorizer(stop_words='english', sublinear_tf=True, max_features=max_features, dtype=np.float32).fit_transform(texts)
    return np.clip(1.0 - (X @ X.T).toarray(), 0.0, 1.0)


def mixed_groups(texts, E, thresh=GROUP_THRESH):
    """Average of the content-word TF-IDF cosine distance and the cosine distance of mean-centred, L2-normalised embeddings E."""
    if len(texts) < 2:
        return np.ones(len(texts), dtype=int)
    Z = np.asarray(E, dtype=np.float64)
    Z = Z - Z.mean(0)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
    D = 0.5 * (content_cosine_distance(texts) + np.clip(1.0 - Z @ Z.T, 0.0, 2.0))
    return average_linkage_groups(D, thresh)


def normalised_content_groups(texts, thresh=STYLE_GROUP_THRESH):
    """The stylegram's grouping: content distance divided by its median off-diagonal value, cut at thresh."""
    if len(texts) < 2:
        return np.ones(len(texts), dtype=int)
    D = content_cosine_distance(texts, max_features=100_000)
    med = max(float(np.median(D[np.triu_indices(len(texts), 1)])), 1e-6)
    return average_linkage_groups(D / med, thresh)


def eligible(groups, lo=GROUP_MIN, hi=GROUP_MAX):
    size = np.bincount(groups)[groups]
    return (size >= lo) & (size <= hi)


def describe_groups(name, groups):
    size = np.bincount(groups)[groups]
    print(f'{name}: {len(groups)} rows | singletons {int((size == 1).sum())}  2-3: {int(((size >= 2) & (size <= 3)).sum())}'
          f'  4-18: {int(((size >= 4) & (size <= 18)).sum())}  >18: {int((size > 18).sum())}', flush=True)


G_TE, GROUP_ROWS = None, None   # set after the GLM pass (section 4)

# ============================================================ 2. style n-grams + extra passages + gates
FUNCTION_WORDS = frozenset("a an the and or but if while although because as so than that which who whom whose this these those it its "
                           "they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with "
                           "without for of by over under between through into about is are was were be been being have has had do does did "
                           "can could will would shall should may might must not no nor all any some each every both either neither other "
                           "another such only also just very too more most less least here there when where why how what whether then now "
                           "still yet even much many few little".split())
TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def shape_view(text):
    return re.sub(r"[^\W\d_]+", lambda m: 'X' if m[0].isupper() else 'x', text)


def function_view(text):
    return ' '.join(t.lower() if (t.lower() in FUNCTION_WORDS or not t.isalnum()) else 'CONTENT' for t in TOKEN_RE.findall(text))


class StyleGrams:
    """Four stacked TF-IDF views: char 3-5, word 1-3, shape (char 2-6 of the x/X text), function-word (word 1-3 of the masked text)."""
    VIEWS = [('char', str, (3, 5), 250_000, 1.0), ('word', str, (1, 3), 180_000, 0.7),
             ('char', shape_view, (2, 6), 60_000, 0.5), ('word', function_view, (1, 3), 50_000, 0.5)]

    def fit(self, texts):
        self.vecs = []
        for analyzer, view, ngr, maxf, w in self.VIEWS:
            opts = dict(analyzer=analyzer, ngram_range=ngr, lowercase=False, sublinear_tf=True, min_df=2, max_features=maxf, dtype=np.float64)
            if analyzer == 'word':
                opts['token_pattern'] = r"(?u)\b\w+\b|[^\w\s]"
            self.vecs.append((TfidfVectorizer(**opts).fit([view(t) for t in texts]), view, w))
        return self

    def transform(self, texts):
        return hstack([v.transform([view(t) for t in texts]) * w for v, view, w in self.vecs], format='csr')


def sparse_group_centre(X, groups, lo=GROUP_MIN, hi=GROUP_MAX):
    """Subtract each group's mean row (sparse); groups outside [lo, hi] are left as they are."""
    X = csr_matrix(X)
    _, inv, cnt = np.unique(np.asarray(groups), return_inverse=True, return_counts=True)
    keep = np.flatnonzero((cnt >= lo) & (cnt <= hi))
    if not len(keep):
        return X.copy()
    col = np.full(len(cnt), -1); col[keep] = np.arange(len(keep))
    rows = np.flatnonzero(col[inv] >= 0)
    M = csr_matrix((np.ones(len(rows)), (rows, col[inv][rows])), shape=(X.shape[0], len(keep)))
    means = diags(1.0 / cnt[keep]) @ (M.T @ X)
    return (X - M @ means).tocsr()


def svc_probs(svc, X):
    return full_prob(softmax_rows(svc.decision_function(X)), svc.classes_)


# ---- public in-style passages (prompt-bank protocol), one per (prompt, label); prompts answered by >= EXTRA_MIN_LABELS labels
def load_extra_passages():
    sources = [('**/ntl_extra_passages_claude.csv', 'model', None), ('**/gen_passages.csv', 'model', None),
               ('**/ntl_generations.csv', 'label', 'finish_reason'), ('**/ntl_proxy_generations.csv', 'label', None),
               ('**/ntl_generated*.csv', 'model', 'finish'), ('**/passages_clean.csv', 'model', None)]
    parts = []
    for pattern, label_col, finish_col in sources:
        for f in sorted(glob.glob('/kaggle/input/' + pattern, recursive=True)):
            d = pd.read_csv(f)
            if finish_col and finish_col in d:
                d = d[d[finish_col].eq('stop')]
            d = d.rename(columns={label_col: 'model'})[['prompt_id', 'model', 'text']].copy()
            d['prompt_id'] = ('pv_' if f.endswith('passages_clean.csv') else '') + d['prompt_id'].astype(str)   # prvsiyan's prompts are a different bank
            parts.append(d)
    if not parts:
        return pd.DataFrame(columns=['prompt_id', 'model', 'text'])
    x = pd.concat(parts, ignore_index=True)
    x = x[x['model'].isin(LABELS)].copy()
    x['text'] = x['text'].astype(str).map(lambda t: re.sub(r'\s+', ' ', re.sub(r'^\*\*(.*)\*\*$', r'\1', t.strip())).strip())
    x = x[x['text'].str.split().str.len().between(30, 300)].drop_duplicates(subset=['model', 'text'])
    x = x.drop_duplicates(subset=['prompt_id', 'model'])
    x = x[x.groupby('prompt_id')['model'].transform('nunique') >= EXTRA_MIN_LABELS].reset_index(drop=True)
    return x


EXTRA = load_extra_passages()
EX_T, EX_Y, EX_G = EXTRA['text'].tolist(), EXTRA['model'].map(LAB2I).to_numpy(dtype=int), EXTRA['prompt_id'].to_numpy(dtype=object)
print(f'extra passages: {len(EXTRA)} rows, {EXTRA["prompt_id"].nunique()} prompts', flush=True)


def extra_sample_weights():
    """Per class: original rows keep (1 - EXTRA_SHARE) of the class weight, the class's extra rows share EXTRA_SHARE of it."""
    n_tr, n_ex = np.bincount(y, minlength=K).astype(float), np.bincount(EX_Y, minlength=K).astype(float)
    w_tr = np.where(n_ex[y] > 0, 1.0 - EXTRA_SHARE, 1.0)
    w_ex = EXTRA_SHARE * n_tr[EX_Y] / np.maximum(n_ex[EX_Y], 1.0)
    return np.concatenate([w_tr, w_ex])


grams = StyleGrams().fit(TR + EX_T)                       # vocabulary and idf from train + extra
X_tr, X_ex, X_te = grams.transform(TR), grams.transform(EX_T) if len(EX_T) else None, grams.transform(TE)


def fit_svc(X, labels, C=0.5, **kw):
    return LinearSVC(C=C, random_state=42, max_iter=4000).fit(X, labels, **kw)


if X_ex is not None:
    svc_plain = fit_svc(vstack([X_tr, X_ex]).tocsr(), np.concatenate([y, EX_Y]), sample_weight=extra_sample_weights())
else:
    svc_plain = fit_svc(X_tr, y)
P_style = svc_probs(svc_plain, X_te)

# centred twin: prompt-centred training rows (extra rows centred within their own prompts) -> inferred test groups
X_tr_c = sparse_group_centre(X_tr, PROMPT)
if X_ex is not None:
    svc_centred = fit_svc(vstack([X_tr_c, sparse_group_centre(X_ex, EX_G, GROUP_MIN, 10 ** 6)]).tocsr(), np.concatenate([y, EX_Y]),
                          sample_weight=extra_sample_weights())
else:
    svc_centred = fit_svc(X_tr_c, y)
G_style = normalised_content_groups(TE)
rows_style = eligible(G_style)
describe_groups('stylegram groups', G_style)
if rows_style.any():
    P_style[rows_style] = svc_probs(svc_centred, sparse_group_centre(X_te[rows_style], G_style[rows_style]))
if TINY:   # exercise the centred path on training rows grouped by their true prompt (output discarded)
    st_idx, st_g = self_test_rows()
    acc = np.mean(svc_centred.classes_[svc_centred.decision_function(sparse_group_centre(X_tr[st_idx], st_g)).argmax(1)] == y[st_idx])
    print(f'  centred stylegram self-test on {len(st_idx)} training rows (seen in training): acc {acc:.2f}', flush=True)
add_member('stylegram', P_style)
del X_tr_c, svc_plain, svc_centred, P_style


# ---- gates: style SVCs that only re-rank inside one family after the blend
def family_gate(texts, labels, classes, C, sample_weight=None, balanced=False):
    g = StyleGrams().fit(texts)
    svc = LinearSVC(C=C, class_weight='balanced' if balanced else None, random_state=42, max_iter=4000).fit(
        g.transform(texts), labels, sample_weight=sample_weight)
    assert list(svc.classes_) == list(classes)
    d = svc.decision_function(g.transform(TE))
    return np.column_stack([-d, d]) if len(classes) == 2 else d


GATES = {}
api_claude = sorted(glob.glob('/kaggle/input/**/ntl_extra_passages_claude.csv', recursive=True))
if api_claude:
    api = pd.read_csv(api_claude[0])
    m = np.isin(y, FAMILY_IDX['claude'])
    GATES['claude'] = family_gate([t for t, k in zip(TR, m) if k] + api['text'].astype(str).tolist(),
                                  np.concatenate([y[m], api['model'].map(LAB2I).to_numpy(dtype=int)]), FAMILY_IDX['claude'], 0.5)
    print(f'claude gate: {int(m.sum())} official + {len(api)} API rows', flush=True)
    del api
else:
    print('Claude API passages not attached: Claude gate off', flush=True)
for fam, C in [('gpt', 1.0), ('gemini', 1.0), ('glm', 0.5)]:
    m, me = np.isin(y, FAMILY_IDX[fam]), np.isin(EX_Y, FAMILY_IDX[fam])
    GATES[fam] = family_gate([t for t, k in zip(TR, m) if k] + [t for t, k in zip(EX_T, me) if k], np.concatenate([y[m], EX_Y[me]]),
                             FAMILY_IDX[fam], C, sample_weight=np.concatenate([np.ones(int(m.sum())), np.full(int(me.sum()), 0.25)]), balanced=True)
    print(f'{fam} gate: {int(m.sum())} official + {int(me.sum())} extra rows', flush=True)
del grams, X_tr, X_ex, X_te

# ============================================================ 3. stylometric features + LightGBM
PUNCT = list(',.;:!?()[]"\'-') + ['—', '–', '‘', '’', '“', '”', '…', '*', '/', '%']
STOPS = sorted(ENGLISH_STOP_WORDS)
SENT_RE, WORD_RE = re.compile(r'(?<=[.!?])\s+'), re.compile(r"[A-Za-z’']+")


def stylometry(t):
    words = WORD_RE.findall(t)
    low = [w.lower() for w in words]
    nw, nc = max(len(words), 1), max(len(t), 1)
    sents = [s for s in SENT_RE.split(t.strip()) if s]
    sl = np.array([len(WORD_RE.findall(s)) for s in sents] or [0], dtype=float)
    wl = np.array([len(w) for w in words] or [0], dtype=float)
    f = [np.log1p(nw), len(sents), sl.mean(), sl.std(), sl.max(), sl.min(), wl.mean(), wl.std(), (wl >= 9).mean(), len(set(low)) / nw,
         sum(c.isupper() for c in t) / nc, sum(c.isdigit() for c in t) / nc, sum(ord(c) > 127 for c in t) / nc,
         t.count(', ') / nw, t.count(' and ') / nw, t.count(' but ') / nw,
         sum(1 for s in sents if s[:1].isupper() and s.split()[0].lower() in ('this', 'these', 'it', 'they')) / max(len(sents), 1),
         sum(w.endswith('ly') for w in words) / nw, sum(w.endswith('ing') for w in words) / nw, sum(w.endswith('tion') for w in words) / nw]
    f += [t.count(q) / nw for q in PUNCT]
    counts = pd.Series(low).value_counts() if low else pd.Series(dtype=float)
    f += [counts.get(s, 0) / nw for s in STOPS]
    return f


F_tr, F_te = (np.array([stylometry(t) for t in txt], dtype=np.float32) for txt in (TR, TE))
try:
    import lightgbm as lgb
    booster = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=10, subsample=0.8, subsample_freq=1,
                                 colsample_bytree=0.5, reg_lambda=1.0, random_state=0, verbose=-1, n_jobs=4).fit(F_tr, y)
    add_member('stylometry_lgb', full_prob(booster.predict_proba(F_te), booster.classes_))
    del booster
except ImportError:
    assert SMOKE, 'lightgbm missing'
    print('smoke: lightgbm unavailable, stylometry member skipped', flush=True)
del F_tr, F_te

# ============================================================ 4b. shared-prompt Gaussian group head
# Verbatim core of our research tool tools/group_gaussian.py (sha256 8a42427b77a37f56...), numpy only.
# Model per dense view, in a reduced space z (standardise -> within-prompt-residual PCA, fitted on training rows):
#   z_i = mu_{y_i} + p_g + e_i,  p_g ~ N(0, S_p) shared by every passage of prompt group g,  e_i ~ N(0, S_e).
# A joint diagonalisation makes the dimensions independent; for a detected group of n passages the exact joint
# log-likelihood of a label tuple is sum_i a_i(c_i) + sum_{i<j} b_ij(c_i, c_j). Groups of 2-3 are decoded exactly over
# all label tuples with the at-most-once constraint; larger groups by a cavity EM with the finite-population-corrected
# prompt-effect estimate lam / (1 + (n-1) lam); singletons integrate the prompt effect out (never used here).
import itertools
K = 18
DEFAULT = {"red": "wpca", "d": 256, "se_shrink": "lw", "sp_shrink": "lw", "lam_floor": 1e-4, "n_exact": 3, "inj": True,
           "iters": 5, "repel": True, "sp_scale": 1.0}

# =================================================================== primitives ====
def _lse(A, axis=None, keepdims=False):
    m = np.max(A, axis=axis, keepdims=True); m = np.where(np.isfinite(m), m, 0.0)
    out = np.log(np.sum(np.exp(A - m), axis=axis, keepdims=True)) + m
    if keepdims: return out
    return np.squeeze(out, axis=axis) if axis is not None else float(out.item())

def log_softmax(L): return L - _lse(L, axis=1, keepdims=True)

def _lw_shrinkage(X):
    """Ledoit-Wolf shrinkage coefficient for rows X that are already centred (sklearn formula, assume_centered)."""
    from sklearn.covariance import ledoit_wolf_shrinkage
    return float(ledoit_wolf_shrinkage(X, assume_centered=True))

def _shrunk(S, a):
    d = S.shape[0]; return (1 - a) * S + a * (np.trace(S) / d) * np.eye(d)

def _two_way(Z, y, pinv, G, iters=4):
    """additive fit Z = mu[y] + P[prompt] + E by alternating means (balanced design -> converges in 1-2 rounds)."""
    P = np.zeros((G, Z.shape[1])); cnt_p = np.bincount(pinv, minlength=G).astype(float)
    cls = np.unique(y); mu = np.zeros((K, Z.shape[1]))
    for _ in range(iters):
        R = Z - P[pinv]
        for c in cls: mu[c] = R[y == c].mean(0)
        R = Z - mu[y]; P = np.zeros_like(P); np.add.at(P, pinv, R); P /= cnt_p[:, None]
    return mu, P, cnt_p

def fit(X, y, prompts, cfg=None):
    """fit the shared-prompt Gaussian on training rows. X (N, D) raw features, y (N,) labels in 0..17, prompts (N,) group ids.
    Returns a params dict (numpy arrays + floats); temperatures default to 1 (set by the caller's cross-fit)."""
    c = dict(DEFAULT); c.update(cfg or {})
    X = np.asarray(X, dtype=np.float64); y = np.asarray(y); N, D = X.shape
    pu, pinv = np.unique(np.asarray(prompts), return_inverse=True); G = len(pu)
    mx = X.mean(0); sx = X.std(0); sx = np.where(sx > 1e-8, sx, 1.0); Xs = (X - mx) / sx
    if c["red"] == "none":
        V = np.eye(D)
    else:
        if c["red"] == "wpca":
            Pm = np.zeros((G, D)); np.add.at(Pm, pinv, Xs); Pm /= np.bincount(pinv)[:, None]; R = Xs - Pm[pinv]
        elif c["red"] == "pca": R = Xs
        else: raise ValueError(c["red"])
        d = int(min(c["d"], N - G - 1, D))
        # economy SVD through the (N x N) Gram matrix (N << D for the real views)
        if N < D:
            w, U = np.linalg.eigh(R @ R.T); o = np.argsort(w)[::-1][:d]; w, U = np.clip(w[o], 1e-12, None), U[:, o]
            V = (R.T @ U) / np.sqrt(w)
        else:
            _, _, Vt = np.linalg.svd(R, full_matrices=False); V = Vt[:d].T
    Z = Xs @ V; d = Z.shape[1]
    mu, P, ng = _two_way(Z, y, pinv, G)
    E = Z - mu[y] - P[pinv]; dof = max(N - G - len(np.unique(y)) + 1, 1)
    Se_raw = E.T @ E / dof
    ae = _lw_shrinkage(E) if c["se_shrink"] == "lw" else float(c["se_shrink"]); Se = _shrunk(Se_raw, ae)
    Pc = P - P.mean(0); Sp_raw = Pc.T @ Pc / max(G - 1, 1) - np.mean(1.0 / ng) * Se_raw
    ap = _lw_shrinkage(Pc) if c["sp_shrink"] == "lw" else float(c["sp_shrink"])
    tr = max(np.trace(Sp_raw), 1e-8 * np.trace(Se)); Sp = (1 - ap) * Sp_raw + ap * (tr / d) * np.eye(d); Sp *= c["sp_scale"]
    # joint diagonalisation: A Se A' = I, A Sp A' = diag(lam)
    L = np.linalg.cholesky(Se); Li = np.linalg.inv(L); M = Li @ Sp @ Li.T; M = 0.5 * (M + M.T)
    lam, Q = np.linalg.eigh(M); lam = np.clip(lam, c["lam_floor"], None); A = Q.T @ Li
    W = V @ A.T
    return {"mx": mx, "sx": sx, "W": W, "m": mu @ A.T, "lam": lam, "T1": 1.0, "Tg": 1.0,
            "info": {"d": int(d), "a_e": round(ae, 4), "a_p": round(ap, 4), "lam_pct": np.round(np.percentile(lam, [5, 50, 95, 100]), 3).tolist()}, "cfg": c}

def transform(X, prm): return ((np.asarray(X, dtype=np.float64) - prm["mx"]) / prm["sx"]) @ prm["W"]

_MASK = {}
def _inj_mask(n, C):
    if (n, C) not in _MASK:
        grids = np.meshgrid(*[np.arange(C)] * n, indexing="ij"); bad = np.zeros([C] * n, bool)
        for i, j in itertools.combinations(range(n), 2): bad |= grids[i] == grids[j]
        _MASK[(n, C)] = bad
    return _MASK[(n, C)]

def singleton_ll(U, m, lam):
    """(n, C) log N(u; m_c, I + diag lam) up to a constant (p integrated out)."""
    R = U[:, None, :] - m[None]; return -0.5 * (R ** 2 / (1 + lam)).sum(-1)

def joint_terms(U, m, lam):
    """a (n, C) and B (n, n, C, C) of the exact joint log-likelihood of one group (constant in c dropped)."""
    n = len(U); gam = lam / (1 + n * lam); R = U[:, None, :] - m[None]
    a = -0.5 * (R ** 2 * (1 - gam)).sum(-1); G = R * np.sqrt(gam); B = np.einsum("icd,jed->ijce", G, G)
    return a, B

def joint_table(a, B):
    """full tuple table Q[c_1, ..., c_n] = sum_i a_i + sum_{i<j} B_ij."""
    n, C = a.shape; Q = np.zeros([C] * n)
    for i in range(n):
        sh = [1] * n; sh[i] = C; Q = Q + a[i].reshape(sh)
    for i, j in itertools.combinations(range(n), 2):
        sh = [1] * n; sh[i] = C; sh[j] = C; Q = Q + B[i, j].reshape(sh)
    return Q

def exact_marginals(Q, T=1.0, inj=True):
    """per-passage log marginal posteriors (n, C) from a tuple table Q (unnormalised log joint), tempered by 1/T."""
    n = Q.ndim; C = Q.shape[0]; Qt = Q / T
    if inj and n >= 2: Qt = np.where(_inj_mask(n, C), -np.inf, Qt)
    lz = _lse(Qt); out = np.empty((n, C))
    for i in range(n):
        ax = tuple(k for k in range(n) if k != i); out[i] = (_lse(Qt, axis=ax) if ax else Qt) - lz
    return out

def em_marginals(U, m, lam, T=1.0, iters=5, repel=False, q0=None):
    """cavity EM for one group (any n >= 2): returns (n, C) log q."""
    n = len(U); v = lam / (1 + (n - 1) * lam)
    L = singleton_ll(U, m, lam) / T if q0 is None else np.log(np.clip(q0, 1e-300, None))
    q = np.exp(log_softmax(L))
    for _ in range(iters):
        s = U - q @ m; S = s.sum(0); Pm = (S[None] - s) * v                  # cavity prompt-effect means (n, d)
        R = U[:, None, :] - Pm[:, None, :] - m[None]; L = -0.5 * (R ** 2 / (1 + v)).sum(-1) / T
        if repel:   # mean-field at-most-once: discourage classes the other members already hold
            lo = np.log(np.clip(1 - q, 1e-6, None)); L = L + (lo.sum(0)[None] - lo)
        q = np.exp(log_softmax(L))
    return log_softmax(L)

LAM_DIFF = 1e6
def lam_g(prm, cfg=None):
    """lam used for groups: the fitted one, or ~infinity in the paired-difference ablation (cfg 'diff'): then the joint likelihood
    only sees within-group differences (x_i - x_j around mu_a - mu_b with covariance 2 S_e; triples: centred observations vs
    centred prototypes) and the prompt term cancels exactly."""
    c = dict(prm.get("cfg", DEFAULT)); c.update(cfg or {})
    return np.full_like(prm["lam"], LAM_DIFF) if c.get("diff") else prm["lam"]

def group_logpost(U, prm, T1=None, Tg=None, cfg=None):
    c = dict(prm.get("cfg", DEFAULT)); c.update(cfg or {}); n = len(U); lg = lam_g(prm, cfg)
    T1 = prm["T1"] if T1 is None else T1; Tg = prm["Tg"] if Tg is None else Tg
    if n == 1: return log_softmax(singleton_ll(U, prm["m"], prm["lam"]) / T1)
    if n <= c["n_exact"]:
        a, B = joint_terms(U, prm["m"], lg); return exact_marginals(joint_table(a, B), Tg, c["inj"])
    return em_marginals(U, prm["m"], lg, Tg, c["iters"], c["repel"])

def scores(X_pool, lab, prm, T1=None, Tg=None, cfg=None, U=None):
    """(n, 18) log-probabilities for every pool row; lab = detected group labels of the pool rows (singletons allowed)."""
    U = transform(X_pool, prm) if U is None else U; lab = np.asarray(lab); n = len(U); out = np.empty((n, prm["m"].shape[0]))
    sizes = np.bincount(lab); s1 = sizes[lab] == 1
    T1 = prm["T1"] if T1 is None else T1
    if s1.any(): out[s1] = log_softmax(singleton_ll(U[s1], prm["m"], prm["lam"]) / T1)
    for g in np.where(sizes >= 2)[0]:
        ii = np.where(lab == g)[0]; out[ii] = group_logpost(U[ii], prm, T1, Tg, cfg)
    return out

# ============================================================ 4. frozen models: plain heads, centred twins, Gaussian group heads
if not SMOKE:
    from transformers import AutoConfig, AutoModel, AutoTokenizer
CACHED_PARTS = {}   # key -> {part name: (train features, test features)}


def find_checkpoint(model_type, hidden=None, dir_name=None):
    """A checkpoint directory under /kaggle/input by model_type (+ hidden size) or by directory name."""
    pattern = f'/kaggle/input/**/{dir_name}/config.json' if dir_name else '/kaggle/input/**/config.json'
    for cfg in sorted(glob.glob(pattern, recursive=True)):
        try:
            c = json.load(open(cfg))
        except Exception:
            continue
        if dir_name or (c.get('model_type') == model_type and (hidden is None or c.get('hidden_size') == hidden)):
            return os.path.dirname(cfg)
    return None


def load_truncated_decoder(path, keep, dtype):
    cfg = AutoConfig.from_pretrained(path)
    cfg.num_hidden_layers = keep
    if getattr(cfg, 'layer_types', None):
        cfg.layer_types = cfg.layer_types[:keep]
    kw = dict(config=cfg, dtype=dtype)
    if DEV == 'cuda':
        kw['device_map'] = 'auto'
    return AutoModel.from_pretrained(path, **kw).eval()


def style_token_mask(tok):
    """Vocabulary mask of 'style' tokens: the decoded piece is an English stop word or pure punctuation."""
    pieces = [tok.decode([i]) for i in range(len(tok))]
    return torch.tensor([(p.strip().lower() in ENGLISH_STOP_WORDS) or bool(re.fullmatch(r'\s*[^\w\s]+\s*', p)) for p in pieces])


@torch.no_grad()
def pooled_states(model, tok, texts, parts, decoder, bs, max_len=512):
    """{'L{layer}_{pool}': (n, hidden)} masked mean pools of hidden_states[layer] (all tokens, or style tokens only)."""
    H = model.config.hidden_size
    out = {f'L{L}_{pool}': np.zeros((len(texts), H), dtype=np.float32) for L, pool in parts}
    smask = style_token_mask(tok) if any(pool == 'style' for _, pool in parts) else None
    dev = next(model.parameters()).device
    order = np.argsort([len(t) for t in texts])
    retried = 0
    for b0 in range(0, len(texts), bs):
        idx = order[b0:b0 + bs]
        if decoder:
            batch = [(tok.bos_token or '') + texts[i] for i in idx]
            enc = tok(batch, return_tensors='pt', padding=True, truncation=True, max_length=max_len, add_special_tokens=False).to(dev)
        else:
            enc = tok([texts[i] for i in idx], return_tensors='pt', padding=True, truncation=True, max_length=max_len).to(dev)
        hs = model(**enc, output_hidden_states=True).hidden_states
        if decoder and not all(torch.isfinite(hs[L]).all() for L, _ in parts):   # fp16 overflow: redo the batch in bf16
            retried += 1
            model.to(torch.bfloat16)
            hs = model(**enc, output_hidden_states=True).hidden_states
        m_all = enc['attention_mask'].float()
        for L, pool in parts:
            m = m_all * smask.to(m_all.device)[enc['input_ids']].float() if pool == 'style' else m_all
            m = m.unsqueeze(-1).to(hs[L].device)
            out[f'L{L}_{pool}'][idx] = ((hs[L].float() * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy()
        del hs, enc
    if retried:
        print(f'  {retried} batches redone in bf16', flush=True)
    return {k: np.nan_to_num(v) for k, v in out.items()}


def pseudo_parts(parts, dim=48):
    """SMOKE only: deterministic stand-ins for the model passes (TF-IDF SVD of the texts), same downstream code."""
    from sklearn.decomposition import TruncatedSVD
    tf = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), sublinear_tf=True, min_df=2).fit(TR + TE)
    out = {}
    for j, (L, pool) in enumerate(parts):
        svd = TruncatedSVD(dim, random_state=L + 7 * j).fit(tf.transform(TR))
        out[f'L{L}_{pool}'] = (svd.transform(tf.transform(TR)).astype(np.float32), svd.transform(tf.transform(TE)).astype(np.float32))
    return out


def extract_parts(key, parts, loader):
    """Train and test pools for every requested part; loader() returns (model, tok, decoder flag, batch size)."""
    if SMOKE:
        return pseudo_parts(parts)
    model, tok, decoder, bs = loader()
    tr, te = pooled_states(model, tok, TR, parts, decoder, bs), pooled_states(model, tok, TE, parts, decoder, bs)
    del model, tok
    gc.collect()
    if DEV == 'cuda':
        torch.cuda.empty_cache()
    return {k: (tr[k], te[k]) for k in tr}


def decoder_loader(model_type, hidden, keep):
    def load():
        path = find_checkpoint(model_type, hidden)
        assert path, f'no {model_type} checkpoint with hidden size {hidden} under /kaggle/input'
        tok = AutoTokenizer.from_pretrained(path)
        tok.padding_side = 'right'
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return load_truncated_decoder(path, keep, torch.float16 if DEV == 'cuda' else torch.float32), tok, True, 8
    return load


def encoder_loader(dir_name):
    def load():
        path = find_checkpoint(None, dir_name=dir_name)
        assert path, f'encoder directory {dir_name} not found under /kaggle/input'
        return AutoModel.from_pretrained(path, dtype=torch.float32).to(DEV).eval(), AutoTokenizer.from_pretrained(path), False, 16
    return load


def concat_parts(cache, parts):
    return (np.concatenate([cache[f'L{L}_{p}'][0] for L, p in parts], 1), np.concatenate([cache[f'L{L}_{p}'][1] for L, p in parts], 1))


def plain_and_twin(key, Etr, Ete, C):
    """Plain head (test-side standardisation) for the blend; centred twin evidence on grouped rows."""
    if W[key] > 0:
        sc = StandardScaler().fit(Etr)
        clf = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(Etr), y)
        Xte = (Ete - Ete.mean(0)) / (Ete.std(0) + 1e-6) if len(Ete) >= 2 else sc.transform(Ete)
        add_member(key, full_prob(clf.predict_proba(Xte), clf.classes_))
    else:
        print(f'  {key}: plain head has weight 0 in the 0.563 blend, not fitted', flush=True)
    if key in TWIN_W:
        csc = StandardScaler().fit(group_centre_dense(Etr, PROMPT))
        twin = LogisticRegression(C=C, max_iter=5000).fit(csc.transform(group_centre_dense(Etr, PROMPT)), y)
        if GROUP_ROWS.any():
            pc = twin.predict_proba(csc.transform(group_centre_dense(Ete[GROUP_ROWS], G_TE[GROUP_ROWS])))
            add_member(key + '_twin', full_prob(pc, twin.classes_), rows=GROUP_ROWS, weight=TWIN_W[key])
        if TINY:
            st_idx, st_g = self_test_rows()
            ps = twin.predict_proba(csc.transform(group_centre_dense(Etr[st_idx], st_g)))
            print(f'  {key} twin self-test on {len(st_idx)} training rows (seen in training): acc {np.mean(twin.classes_[ps.argmax(1)] == y[st_idx]):.2f}', flush=True)


def gaussian_group_head(name, spec, Xtr, Xte):
    """Our addition: fit the shared-prompt Gaussian on the training prompts, score the grouped test rows jointly within their group."""
    prm = fit(Xtr, y, PROMPT, dict(GAUSS_SETTING, d=spec['d']))
    prm['T1'], prm['Tg'] = spec['T1'], spec['Tg']
    print(f'  {name}: d {prm["info"]["d"]}, shrinkage e/p {prm["info"]["a_e"]}/{prm["info"]["a_p"]}, lam pct {prm["info"]["lam_pct"]}', flush=True)
    if GROUP_ROWS.any():
        LP = scores(Xte[GROUP_ROWS], np.unique(G_TE[GROUP_ROWS], return_inverse=True)[1], prm)
        add_member(name, np.exp(LP), rows=GROUP_ROWS, weight=spec['w'])
    if TINY:
        st_idx, st_g = self_test_rows()
        LP = scores(Xtr[st_idx], st_g, prm)
        print(f'  {name} self-test on {len(st_idx)} training rows (seen in training): acc {np.mean(LP.argmax(1) == y[st_idx]):.2f}', flush=True)


for key, model_type, hidden, parts, extra_parts, C in LM_HEADS:
    print(f'[{key}] {model_type} hidden {hidden} parts {parts + extra_parts}', flush=True)
    cache = extract_parts(key, parts + extra_parts, decoder_loader(model_type, hidden, max(L for L, _ in parts + extra_parts) + 1))
    if G_TE is None:   # the GLM pass comes first: its layer-24 pool defines the test groups
        G_TE = mixed_groups(TE, cache[f'L{GROUP_LAYER}_mean'][1])
        GROUP_ROWS = eligible(G_TE)
        describe_groups('test groups (content + GLM L24)', G_TE)
        if SMOKE:
            same_g, same_p = G_TE[:, None] == G_TE[None, :], test['prompt_id'].to_numpy(dtype=object)[:, None] == test['prompt_id'].to_numpy(dtype=object)[None, :]
            off = ~np.eye(len(TE), dtype=bool)
            print(f'  smoke grouping: pair precision {(same_g & same_p & off).sum() / max((same_g & off).sum(), 1):.2f} '
                  f'recall {(same_g & same_p & off).sum() / max((same_p & off).sum(), 1):.2f}', flush=True)
    CACHED_PARTS[key] = cache
    plain_and_twin(key, *concat_parts(cache, parts), C)
    for name, spec in GAUSS_HEADS.items():
        if spec['src'] == key:
            gaussian_group_head(name, spec, *cache[spec['part']])
    del cache
    CACHED_PARTS.pop(key)
    gc.collect()

for key, dir_name, parts, C in ENC_HEADS:
    print(f'[{key}] {dir_name} parts {parts}', flush=True)
    cache = extract_parts(key, parts, encoder_loader(dir_name))
    plain_and_twin(key, *concat_parts(cache, parts), C)
    for name, spec in GAUSS_HEADS.items():
        if spec['src'] == key:
            gaussian_group_head(name, spec, *cache[spec['part']])
    del cache
    gc.collect()

# ============================================================ 5. LoRA classifiers (inference only)
@torch.no_grad()
def score_lora_checkpoints(trunk_spec, pool_layers, checkpoints):
    """One truncated trunk, adapters swapped in turn. Plain readout -> blend member; centred readout -> twin on grouped rows;
    within-Claude readout -> Claude gate."""
    import importlib.util as iu
    _find = iu.find_spec   # Kaggle's torchao build is rejected by peft; hide it (unused here)
    iu.find_spec = lambda name, *a, **k: None if name.split('.')[0] == 'torchao' else _find(name, *a, **k)
    from peft import PeftModel
    import peft.import_utils as pi
    pi.is_torchao_available = lambda *a, **k: False
    from safetensors.torch import load_file
    model_type, H, keep = trunk_spec
    base = find_checkpoint(model_type, H)
    assert base, f'no {model_type} base checkpoint under /kaggle/input'
    cfg = AutoConfig.from_pretrained(base)
    cfg.num_hidden_layers, cfg.use_cache = keep, False
    if getattr(cfg, 'layer_types', None) is not None:
        cfg.layer_types = cfg.layer_types[:keep]
    trunk = AutoModel.from_pretrained(base, config=cfg, dtype=torch.float16, device_map='auto').eval()
    for key, marker, norms_file, anchor_file in checkpoints:
        hits = sorted(glob.glob('/kaggle/input/' + marker, recursive=True))
        if not hits:   # upstream training kernel no longer public: drop this member, keep the rest of the stack
            print(f'  SKIP {key}: checkpoint not attached ({marker})', flush=True); continue
        ck = pathlib.Path(os.path.dirname(hits[0]))
        print(f'[{key}] {ck}', flush=True)
        tok = AutoTokenizer.from_pretrained(ck / 'tokenizer')
        tok.padding_side = 'right'
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        lm = PeftModel.from_pretrained(trunk, ck / 'adapter').eval()
        state = load_file(str(ck / norms_file))
        norms = torch.nn.ModuleList([torch.nn.LayerNorm(H) for _ in pool_layers])
        norms.load_state_dict({k[len('norms.'):]: v for k, v in state.items() if k.startswith('norms.')})
        ro = np.load(ck / 'readouts.npz')
        assert list(ro['classes']) == LABELS and list(ro['pool_layers']) == pool_layers
        dev = lm.get_input_embeddings().weight.device
        norms = norms.to(dev).float()

        def token_ids(texts):
            return [ids + [tok.eos_token_id] for ids in tok(list(texts), add_special_tokens=False, truncation=True, max_length=LORA_MAX_LEN)['input_ids']]   # these checkpoints' anchors match the tokenizer's own eos

        def features(texts, bs):
            ids = token_ids(texts)
            Z = np.zeros((len(ids), H * len(pool_layers)), dtype=np.float32)
            order = np.argsort([len(t) for t in ids], kind='stable')
            for b0 in range(0, len(order), bs):
                idx = order[b0:b0 + bs]
                batch = tok.pad({'input_ids': [ids[i] for i in idx]}, return_tensors='pt').to(dev)
                with torch.autocast('cuda', dtype=torch.float16):
                    hs = lm(**batch, output_hidden_states=True, use_cache=False).hidden_states
                m = batch['attention_mask'].unsqueeze(-1).float()
                m[:, 0] = 0   # the first token is masked out of the mean, as in training
                Z[idx] = torch.cat([nm((hs[L].float() * m.to(hs[L].device)).sum(1).to(dev) / m.sum(1).to(dev))
                                    for L, nm in zip(pool_layers, norms)], -1).cpu().numpy()
                del hs, batch
            return Z

        if (ck / anchor_file).exists():   # the checkpoint's own features for training rows 0..2: our forward pass must reproduce them
            saved = np.load(ck / anchor_file)['features'].astype(np.float32)
            ours = features(TR[:3], 3)
            rel = float(np.abs(ours - saved).mean() / (np.abs(saved).mean() + 1e-9))
            print(f'  parity with the checkpoint anchors: max abs {float(np.abs(ours - saved).max()):.4f}, mean rel {rel:.5f}', flush=True)
            assert rel < 0.02, f'{key}: our features do not match the checkpoint'
        Z = features(TE, LORA_BS)
        add_member(key, softmax_rows(Z @ ro['plain_coef'].T + ro['plain_intercept']))
        if 'claude_coef' in ro.files:
            lc = Z @ ro['claude_coef'].T + ro['claude_intercept']
            GATES['claude_lora'] = lc - np.log(np.exp(lc - lc.max(1, keepdims=True)).sum(1, keepdims=True)) - lc.max(1, keepdims=True)
        if key in LORA_TWIN_W and GROUP_ROWS.any():
            Zc = group_centre_dense(Z[GROUP_ROWS], G_TE[GROUP_ROWS])
            add_member(key + '_twin', softmax_rows(Zc @ ro['center_coef'].T + ro['center_intercept']), rows=GROUP_ROWS, weight=LORA_TWIN_W[key])
        trunk = lm.unload()
        del lm, Z
        gc.collect()
        torch.cuda.empty_cache()
    del trunk
    gc.collect()
    torch.cuda.empty_cache()


if not SMOKE:
    for trunk_spec, pool_layers, checkpoints in LORA_RUNS:
        score_lora_checkpoints(trunk_spec, pool_layers, checkpoints)

# ============================================================ 6. token-attention heads + UltraBERT
class TokenAttention(torch.nn.Module if torch is not None else object):   # the module of the public training kernels (lucasboesen/name-that-llm-train-tokattn, -tokattn2)
    def __init__(self, k, h=4096):
        super().__init__()
        self.drop, self.proj, self.score, self.out = torch.nn.Dropout(0.2), torch.nn.Linear(h, 64), torch.nn.Linear(64, 1), torch.nn.Linear(128, k)

    def forward(self, x, m):
        h = torch.tanh(self.proj(self.drop(x)))
        a = torch.softmax(self.score(h).squeeze(-1).masked_fill(~m, -1e4), -1).unsqueeze(-1)
        mf = m.unsqueeze(-1).float()
        return self.out(self.drop(torch.cat([(h * mf).sum(1) / mf.sum(1).clamp(min=1), (a * h).sum(1)], -1)))


@torch.no_grad()
def token_attention_probs(texts, head_file, backbone, bs=8):
    hits = sorted(glob.glob('/kaggle/input/**/' + head_file, recursive=True))
    assert hits, f'training-kernel output holding {head_file} not attached'
    d = torch.load(hits[0], map_location='cpu')
    assert list(d['classes']) == LABELS
    decoder = d.get('decoder', True)
    path = find_checkpoint('qwen3_5') if backbone == 'qwen3_5' else find_checkpoint(None, dir_name='deberta-v3-large')
    assert path, f'backbone {backbone} not attached'
    cfg = AutoConfig.from_pretrained(path)
    if decoder:
        tc = getattr(cfg, 'text_config', None) or cfg
        tc.num_hidden_layers = d['keep']
        if getattr(tc, 'layer_types', None):
            tc.layer_types = tc.layer_types[:d['keep']]
        lm = AutoModel.from_pretrained(path, config=cfg, dtype=torch.float16, device_map='auto').eval()
    else:
        lm = AutoModel.from_pretrained(path, dtype=torch.float32).to(DEV).eval()
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = 'right'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    H = d.get('hidden', 4096)
    net = TokenAttention(K, H)
    net.load_state_dict(d['state_dict'])
    dev = lm.get_input_embeddings().weight.device
    net, mu, sd = net.to(dev).eval(), d['mu'].float().to(dev), d['sd'].float().to(dev)
    ids = tok(list(texts), add_special_tokens=not decoder, truncation=True, max_length=d['max_len'])['input_ids']
    P = np.zeros((len(texts), K))
    order = np.argsort([len(t) for t in ids])
    for b0 in range(0, len(order), bs):
        idx = order[b0:b0 + bs]
        batch = tok.pad({'input_ids': [ids[i] for i in idx]}, return_tensors='pt').to(dev)
        out = lm(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], output_hidden_states=True)
        hs = (out.hidden_states if getattr(out, 'hidden_states', None) is not None else out.language_model_outputs.hidden_states)[d['layer']]
        n = batch['attention_mask'].sum(1)
        T = max(int(n.max()) - 1, 1)
        x = torch.zeros(len(idx), T, H, device=dev)
        m = torch.zeros(len(idx), T, dtype=torch.bool, device=dev)
        for r in range(len(idx)):   # first token dropped, states rounded through fp16, as in training
            k = int(n[r]) - 1
            if k > 0:
                x[r, :k] = ((torch.nan_to_num(hs[r, 1:k + 1].to(dev).float()).half().float() - mu) / sd).clamp(-8, 8)
                m[r, :k] = True
        P[idx] = torch.softmax(net(x, m), -1).cpu().numpy()
        del hs, out, batch
    del lm
    gc.collect()
    torch.cuda.empty_cache()
    return P


@torch.no_grad()
def ultrabert_probs(texts):
    from transformers import AutoModelForSequenceClassification
    folds = None
    for root in glob.glob('/kaggle/input/**/ultrabert-380m-ntl', recursive=True):
        fs = [f for f in sorted(glob.glob(os.path.join(root, 'fold*'))) if os.path.exists(os.path.join(f, 'config.json'))
              and json.load(open(os.path.join(f, 'config.json'))).get('model_type') == 'ultrabert']
        if len(fs) == 5:
            folds = fs
            break
    assert folds, 'ultrabert-380m-ntl (5 folds) not attached'
    tok = AutoTokenizer.from_pretrained(folds[0], trust_remote_code=True)
    ids = tok(texts, truncation=True, max_length=512)['input_ids']
    order = np.argsort([len(e) for e in ids])
    P = np.zeros((len(texts), K))
    for fp in folds:
        em = AutoModelForSequenceClassification.from_pretrained(fp, trust_remote_code=True, dtype=torch.float16, attn_implementation='sdpa').to(DEV).eval()
        cols = [LAB2I[em.config.id2label[i]] for i in range(len(em.config.id2label))]
        for b0 in range(0, len(order), 32):
            idx = order[b0:b0 + 32]
            b = tok.pad({'input_ids': [ids[j] for j in idx]}, return_tensors='pt').to(DEV)
            P[np.ix_(idx, cols)] += em(**b).logits.float().softmax(-1).cpu().numpy() / len(folds)
        del em
        gc.collect()
        torch.cuda.empty_cache()
    return P


if not SMOKE:
    for key, head_file, backbone in TOKATTN_HEADS:
        if TINY:   # in-sample check on training rows the head was fitted on: must be high, or the checkpoint is not being reproduced
            acc = float(np.mean(token_attention_probs(TR[:40], head_file, backbone).argmax(1) == y[:40]))
            print(f'  {key} in-sample check on 40 training rows: acc {acc:.2f}', flush=True)
            assert acc > 0.5, f'{key} does not reproduce its training fit'
        add_member(key, token_attention_probs(TE, head_file, backbone))
    add_member('ultrabert', ultrabert_probs(TE))



np.save('/kaggle/working/comp_logit.npy', LOGIT)
json.dump({'labels': LABELS, 'ids': test['id'].astype(str).tolist(), 'seconds': time.time() - T0}, open('/kaggle/working/comp_meta.json', 'w'))
print(f'COMP STACK DONE {time.time() - T0:.0f}s', flush=True)
