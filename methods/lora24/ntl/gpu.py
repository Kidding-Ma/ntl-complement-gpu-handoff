"""GPU-only model construction. Import this module only on the training machine."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. CPU tests do not train this model.")

def load_base(base_path, config):
    require_cuda()
    cfg = AutoConfig.from_pretrained(base_path, local_files_only=True)
    if (cfg.model_type, cfg.hidden_size, cfg.num_hidden_layers) != ("qwen3", 4096, 36):
        raise ValueError("Use the official 36-layer Qwen3-8B Base, not an instruct model")
    cfg.num_hidden_layers = config["keep_layers"]
    cfg.use_cache = False
    if getattr(cfg, "layer_types", None) is not None:
        cfg.layer_types = cfg.layer_types[:cfg.num_hidden_layers]
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[config["dtype"]]
    model = AutoModel.from_pretrained(
        base_path, config=cfg, dtype=dtype, local_files_only=True,
        low_cpu_mem_usage=True, device_map={"": 0}, attn_implementation="sdpa")
    return model

def tokenizer_at(path):
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    tok.padding_side = "right"
    if tok.eos_token_id is None:
        raise ValueError("Tokenizer has no EOS token")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

def tokenize(tok, texts, config):
    ids = tok(list(texts), add_special_tokens=False, truncation=True,
              max_length=config["max_text_tokens"])["input_ids"]
    ids = [part + [tok.eos_token_id] for part in ids]
    # A hidden passage may be empty: EOS alone is valid. The pool denominator
    # is clamped in representation, matching the original reader's fallback.
    return ids

def representation(model, norms, tok, token_rows, layers, dtype):
    device = model.get_input_embeddings().weight.device
    batch = tok.pad({"input_ids": token_rows}, return_tensors="pt").to(device)
    with torch.autocast("cuda", dtype=dtype):
        hs = model(**batch, output_hidden_states=True, use_cache=False).hidden_states
    mask = batch["attention_mask"].unsqueeze(-1).float()
    mask[:, 0] = 0
    # Match the original inference contract: drop the first token, mean including EOS,
    # apply an independent FP32 LayerNorm to each pool, then concatenate.
    with torch.autocast("cuda", enabled=False):
        z = torch.cat([norm((hs[layer].float() * mask).sum(1) / mask.sum(1).clamp_min(1))
                       for layer, norm in zip(layers, norms)], dim=-1)
    if not torch.isfinite(z).all():
        raise FloatingPointError("Nonfinite feature values")
    return z

def extract(model, norms, tok, tokens, rows, config, batch_size=3):
    rows = np.asarray(rows, dtype=int)
    out = np.empty((len(rows), 4096 * len(config["pool_layers"])), dtype=np.float32)
    order = np.argsort([len(tokens[i]) for i in rows], kind="stable")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[config["dtype"]]
    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            positions = order[start:start + batch_size]
            z = representation(model, norms, tok, [tokens[rows[i]] for i in positions],
                               config["pool_layers"], dtype)
            out[positions] = z.cpu().numpy()
    return out

def reload_checkpoint(base_path, checkpoint):
    from peft import PeftModel
    from safetensors.torch import load_file
    checkpoint = Path(checkpoint)
    meta = json.loads((checkpoint / "training_meta.json").read_text())
    base = load_base(base_path, meta["config"])
    model = PeftModel.from_pretrained(base, checkpoint / "adapter").eval()
    norms = nn.ModuleList([nn.LayerNorm(4096) for _ in meta["config"]["pool_layers"]]).cuda().float()
    state = load_file(str(checkpoint / "heads.safetensors"))
    norms.load_state_dict({k[len("norms."):]: v for k, v in state.items() if k.startswith("norms.")})
    norms.eval()
    return model, norms, tokenizer_at(checkpoint / "tokenizer"), meta
