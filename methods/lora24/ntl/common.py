"""CPU-safe contracts shared by preparation, training and inference."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

LABELS = sorted([
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5", "claude-fable-5",
    "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3",
])
POOL_LAYERS = [2, 4, 6, 8, 12, 16, 20, 24]

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)

def read_train(path):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = ["id", "prompt_id", "text", "model"]
    if not set(required) <= set(frame):
        raise ValueError(f"train.csv must contain {required}")
    if not len(frame) or not frame.id.is_unique or (frame[required] == "").any().any():
        raise ValueError("Training IDs must be unique; required fields must be nonempty")
    if sorted(frame.model.unique()) != LABELS:
        raise ValueError("Training labels differ from the frozen 18-label contract")
    return frame

def make_plan(frame, seed=42, folds=5):
    groups = np.asarray(sorted(frame.prompt_id.unique()), dtype=str)
    if len(groups) < folds:
        raise ValueError("Not enough prompts for the requested number of folds")
    groups = np.random.default_rng(seed).permutation(groups)
    jobs = {}
    for index, held_out in enumerate(np.array_split(groups, folds)):
        mask = frame.prompt_id.isin(held_out).to_numpy()
        fit, valid = np.flatnonzero(~mask), np.flatnonzero(mask)
        if sorted(frame.iloc[fit].model.unique()) != LABELS:
            raise ValueError(f"Fold {index} fitting side misses classes")
        jobs[f"fold{index}"] = {"fit": fit.tolist(), "valid": valid.tolist()}
    jobs["final"] = {"fit": list(range(len(frame))), "valid": []}
    return {"seed": seed, "folds": folds, "labels": LABELS, "jobs": jobs,
            "split_method": "shuffle unique prompt IDs then array_split; all answers stay together"}

def epoch_batches(fit, groups, rng, accumulation=6):
    """Each microbatch is three distinct answers to one fitting-side prompt."""
    fit, groups = np.asarray(fit, dtype=int), np.asarray(groups)
    batches = []
    for group in np.unique(groups[fit]):
        members = rng.permutation(fit[groups[fit] == group])
        if len(members) < 3:
            raise ValueError(f"Prompt {group!r} has fewer than three fitting answers")
        for start in range(0, len(members), 3):
            batch = members[start:start + 3]
            if len(batch) < 3:
                batch = np.concatenate([batch, rng.choice(
                    np.setdiff1d(members, batch), 3 - len(batch), replace=False)])
            batches.append(batch)
    batches = [batches[i] for i in rng.permutation(len(batches))]
    original = len(batches)
    while len(batches) % accumulation:
        batches.append(batches[int(rng.integers(original))].copy())
    return batches, {"original_microbatches": original, "padded_microbatches": len(batches) - original,
                     "example_exposures": len(batches) * 3, "unique_fit_rows": len(fit)}

def softmax(values):
    values = np.asarray(values, dtype=np.float64)
    exp = np.exp(values - values.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)

def balanced_accuracy(y, pred):
    return float(np.mean([np.mean(pred[y == k] == k) for k in np.unique(y)]))
