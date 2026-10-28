"""Train one prompt-disjoint fold or final model on one visible CUDA GPU."""
import argparse
import gc
import importlib.metadata
import json
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ntl.common import LABELS, POOL_LAYERS, epoch_batches, read_train, sha256, write_json

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True)
    p.add_argument("--job", required=True, help="fold0..fold4 or final")
    p.add_argument("--output", required=True, help="Fresh checkpoint directory; never overwrites")
    p.add_argument("--max-updates", type=int, help="Smoke only: no COMPLETE.json is written")
    args = p.parse_args()
    if args.max_updates is not None and args.max_updates < 1:
        p.error("--max-updates must be positive")
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(f"Choose a new output directory, or archive the old run yourself: {out}")
    plan = json.loads(Path(args.plan).read_text())
    config = plan["config"]
    if config["epochs"] < 1 or config["accumulation"] < 1:
        raise ValueError("epochs and accumulation must be positive")
    if config["pool_layers"] != POOL_LAYERS or config["keep_layers"] != 24 or config["microbatch"] != 3:
        raise ValueError("This implementation fixes the original 24-layer / 8-pool / triple contract")
    if sha256(plan["train_csv"]) != plan["train_sha256"]:
        raise ValueError("Training data changed after preparing the plan")
    if sha256(Path(plan["base_model"]) / "config.json") != plan["base_config_sha256"]:
        raise ValueError("Base configuration changed after preparing the plan")
    frame = read_train(plan["train_csv"])
    import numpy as np
    import torch
    from torch import nn
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import save_file
    from ntl.gpu import extract, load_base, representation, require_cuda, tokenize, tokenizer_at
    require_cuda()
    seed = config["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    fit = np.asarray(plan["jobs"][args.job]["fit"], dtype=int)
    valid = np.asarray(plan["jobs"][args.job]["valid"], dtype=int)
    groups = frame.prompt_id.to_numpy()
    if set(groups[fit]) & set(groups[valid]) or set(fit) & set(valid):
        raise ValueError("Prompt or row leakage in plan")
    if sorted(frame.iloc[fit].model.unique()) != LABELS:
        raise ValueError("Fitting side must contain every class")
    if plan["labels"] != LABELS:
        raise ValueError("Label order changed")
    target = np.searchsorted(LABELS, frame.model.to_numpy())
    rng = np.random.default_rng(seed)
    epochs = [epoch_batches(fit, groups, rng, config["accumulation"])
              for _ in range(config["epochs"])]
    total = sum(len(batches) // config["accumulation"] for batches, _ in epochs)
    limit = min(total, args.max_updates) if args.max_updates else total
    out.mkdir(parents=True)
    report = {"status": "STARTING", "job": args.job, "smoke_only": args.max_updates is not None,
              "train_sha256": plan["train_sha256"], "fit_rows": len(fit), "valid_rows": len(valid),
              "planned_updates": total, "update_limit": limit,
              "epoch_sampling": [audit for _, audit in epochs], "updates": [],
              "gpu": torch.cuda.get_device_name(0), "python": sys.version,
              "packages": {k: importlib.metadata.version(k) for k in
                           ["torch", "transformers", "peft", "accelerate", "safetensors", "numpy", "pandas"]}}
    write_json(out / "training_report.json", report)
    started = time.monotonic()
    try:
        tok = tokenizer_at(plan["base_model"])
        tokens = tokenize(tok, frame.text.tolist(), config)
        base = load_base(plan["base_model"], config)
        base.requires_grad_(False)
        model = get_peft_model(base, LoraConfig(
            r=config["lora_rank"], lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"], bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        norms = nn.ModuleList([nn.LayerNorm(4096) for _ in POOL_LAYERS]).cuda().float()
        plain = nn.Linear(4096 * len(POOL_LAYERS), len(LABELS)).cuda().float()
        centered = nn.Linear(4096 * len(POOL_LAYERS), len(LABELS)).cuda().float()
        dropout = nn.Dropout(config["head_dropout"])
        adapter_params = [v for v in model.parameters() if v.requires_grad]
        head_params = list(norms.parameters()) + list(plain.parameters()) + list(centered.parameters())
        parameters = adapter_params + head_params
        if not adapter_params or not all(v.dtype == torch.float32 for v in parameters):
            raise ValueError("Expected FP32 trainable adapters and heads")
        report["trainable_parameters"] = sum(v.numel() for v in parameters)
        optimizer = torch.optim.AdamW([
            {"params": adapter_params, "lr": config["adapter_lr"]},
            {"params": head_params, "lr": config["head_lr"]}], weight_decay=config["weight_decay"])
        warmup = max(1, int(total * config["warmup_fraction"]))
        def schedule(step):
            if step < warmup:
                return (step + 1) / warmup
            return .5 * (1 + math.cos(math.pi * min(1., (step - warmup) / max(1, total - warmup))))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        fp16 = config["dtype"] == "float16"
        dtype = torch.float16 if fp16 else torch.bfloat16
        scaler = torch.amp.GradScaler("cuda", enabled=fp16, init_scale=256.)
        model.train(); norms.train(); plain.train(); centered.train()
        torch.cuda.reset_peak_memory_stats()
        report["status"] = "TRAINING"
        for epoch, (batches, _) in enumerate(epochs, start=1):
            for start in range(0, len(batches), config["accumulation"]):
                if len(report["updates"]) >= limit:
                    break
                chunk = batches[start:start + config["accumulation"]]
                cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
                update_started = time.monotonic()
                for attempt in range(config["max_overflow_retries"] + 1):
                    torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state(cuda_rng)
                    optimizer.zero_grad(set_to_none=True)
                    losses = []
                    for indices in chunk:
                        z = representation(model, norms, tok, [tokens[i] for i in indices], POOL_LAYERS, dtype)
                        y = torch.as_tensor(target[indices], device="cuda", dtype=torch.long)
                        with torch.autocast("cuda", enabled=False):
                            lp = nn.functional.cross_entropy(plain(dropout(z)), y)
                            lc = nn.functional.cross_entropy(centered(dropout(z - z.mean(0, keepdim=True))), y)
                            loss = lp + lc
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Nonfinite forward loss; no automatic masking")
                        losses.append(float(loss.detach()))
                        scaler.scale(loss / len(chunk)).backward()
                    scaler.unscale_(optimizer)
                    norm = torch.nn.utils.clip_grad_norm_(parameters, config["gradient_clip"])
                    finite = bool(torch.isfinite(norm))
                    if not finite and not fp16:
                        raise FloatingPointError("Nonfinite BF16 gradients")
                    if finite:
                        if not any(v.grad is not None and bool(torch.count_nonzero(v.grad))
                                   for v in adapter_params):
                            raise RuntimeError("All adapter gradients are zero; training path is broken")
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        break
                    # Never step on a nonfinite total norm, even if individual gradients
                    # were finite and only the norm reduction overflowed.
                    scaler.update(new_scale=scaler.get_scale() / 2)
                    print(json.dumps({"event": "AMP_OVERFLOW", "attempt": attempt + 1,
                                      "scale": scaler.get_scale()}), flush=True)
                    if attempt == config["max_overflow_retries"] or scaler.get_scale() < 1:
                        raise FloatingPointError("Repeated AMP overflow; inspect environment/dtype")
                torch.cuda.synchronize()
                record = {"number": len(report["updates"]) + 1, "epoch": epoch,
                          "loss": float(np.mean(losses)), "gradient_norm": float(norm),
                          "overflow_retries": attempt, "scale": scaler.get_scale(),
                          "seconds": time.monotonic() - update_started}
                report["updates"].append(record)
                write_json(out / "training_report.json", report)
                print(json.dumps(record), flush=True)
            if len(report["updates"]) >= limit:
                break
        model.eval(); norms.eval(); plain.eval(); centered.eval()
        model.gradient_checkpointing_disable()
        optimizer.zero_grad(set_to_none=True)
        valid_z = extract(model, norms, tok, tokens, valid, config)
        anchors = np.arange(min(3, len(frame)))
        anchor_z = extract(model, norms, tok, tokens, anchors, config)
        model.save_pretrained(out / "adapter", safe_serialization=True, save_embedding_layers=False)
        tok.save_pretrained(out / "tokenizer")
        base.config.save_pretrained(out / "base_config")
        state = {}
        for prefix, module in [("norms", norms), ("plain", plain), ("centered", centered)]:
            state.update({f"{prefix}.{k}": v.detach().cpu().contiguous() for k, v in module.state_dict().items()})
        save_file(state, str(out / "heads.safetensors"))
        np.savez_compressed(out / "readouts.npz", classes=np.asarray(LABELS), pool_layers=np.asarray(POOL_LAYERS),
            plain_coef=plain.weight.detach().cpu().numpy(), plain_intercept=plain.bias.detach().cpu().numpy(),
            center_coef=centered.weight.detach().cpu().numpy(), center_intercept=centered.bias.detach().cpu().numpy(),
            train_sha256=plan["train_sha256"])
        np.savez_compressed(out / "validation_features.npz", ids=frame.id.iloc[valid].to_numpy(dtype=str),
                            rows=valid, features=valid_z, classes=np.asarray(LABELS))
        np.savez_compressed(out / "anchor_features.npz", ids=frame.id.iloc[anchors].to_numpy(dtype=str),
                            features=anchor_z, classes=np.asarray(LABELS))
        source_root = Path(__file__).resolve().parents[1]
        source_hashes = {str(p.relative_to(source_root)): sha256(p)
                         for folder in ["ntl", "scripts"] for p in (source_root / folder).glob("*.py")}
        write_json(out / "training_meta.json", {
            "schema": 1, "job": args.job, "experiment": config["experiment"], "config": config,
            "labels": LABELS, "train_sha256": plan["train_sha256"],
            "base_config_sha256": plan["base_config_sha256"], "plan_sha256": sha256(args.plan),
            "source_sha256": source_hashes, "smoke_only": args.max_updates is not None,
            "identity": "independent retraining; not verified identical to lightningv08 checkpoint"})
        report.update(status="SMOKE_COMPLETE" if args.max_updates else "COMPLETE",
                      seconds=time.monotonic() - started,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      successful_updates=len(report["updates"]))
        write_json(out / "training_report.json", report)
        files = {str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob("*")) if p.is_file()}
        marker = "SMOKE_ONLY.json" if args.max_updates else "COMPLETE.json"
        write_json(out / marker, {"schema": 1, "job": args.job, "train_sha256": plan["train_sha256"], "files": files})
        print(f"Exported {marker}: {out}", flush=True)
    except Exception as error:
        report.update(status="ERROR", error=f"{type(error).__name__}: {error}")
        write_json(out / "training_report.json", report)
        raise

if __name__ == "__main__":
    main()
