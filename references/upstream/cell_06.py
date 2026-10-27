W_N14, W_N8 = 1.0, 0.5
L = [2, 4, 6, 8, 12, 16, 20, 24]
MODELS = [
  {"name": "q14v5", "type": "l24", "path": B14, "adapters": AQ14V5, "keep": 27, "layers": L, "w": 3.0, "bs": 8, "max_len": 384, "twin": 0.35, "gauss": 0.70},
  {"name": "q8v5", "type": "l24", "path": B8, "adapters": AQ8V5, "keep": 24, "layers": L, "w": 2.0, "bs": 16, "max_len": 384, "twin": 0.35, "gauss": 0.44},
  {"name": "ll8v5", "type": "l24", "path": BL, "adapters": ALL8V5, "keep": 24, "layers": L, "w": 1.5, "bs": 16, "max_len": 384, "twin": 0.35},
  {"name": "ll8_ts", "type": "l24", "path": BL, "adapters": AL, "keep": 24, "layers": L, "w": 1.0, "bs": 16, "max_len": 384, "twin": 0.35},
  {"name": "q8s7", "type": "l24", "path": B8, "adapters": A8S7, "keep": 24, "layers": L, "w": 1.0, "bs": 16, "max_len": 384, "twin": 0.35},
  {"name": "n14b", "type": "l24", "path": B14, "adapters": [N14B], "keep": N14BC['keep'], "layers": N14BC['layers'], "w": 2.5, "bs": 8, "max_len": N14BC['max_len'], "twin": 0.35, "no_final_norm": True, "eos_id": N14BC['eos_id']},
  {"name": "q25_14", "type": "l24", "path": B25_14, "adapters": [Q25_14], "keep": Q25_14C['keep'], "layers": Q25_14C['layers'], "w": 1.0, "bs": 8, "max_len": Q25_14C['max_len'], "twin": 0.35, "no_final_norm": True, "eos_id": Q25_14C['eos_id']},
  {"name": "deb", "type": "enc", "ckpt": DEB, "w": 1.3},
]
import pandas as pd
ATLAS_BUDGET = int(8.3 * 3600 - COMP_SEC - 600)
!python infer.py --data {DATA} --models '{json.dumps(MODELS)}' --perpassage 1 --extra {EXTRA} --extra_w 0.5 --tfidf_w 0 --distort_w 0 --svc_w 1.8 --pool {POOL} --adapt_w 1.0 --adapt_thr 0.15 --sinkhorn 1.0 --post 1 --atlas_dir {ATLAS} --atlas_base {B8} --atlas_w 0.16 --atlas_margin 0.38 --atlas_bs 4 --atlas_chunk 3 --atlas_budget {ATLAS_BUDGET} --comp_path /kaggle/working/comp_logit.npy --comp_w 0.15 --out submission.parquet --cache /kaggle/working/cache
print(pd.read_parquet('submission.parquet').head())

sub_df = pd.read_parquet('submission.parquet')
sub_df.to_csv('submission.csv', index=False)
print('Verified submission shape:', sub_df.shape)
print(sub_df.head())
