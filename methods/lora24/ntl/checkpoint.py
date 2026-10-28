"""Validate exported checkpoints without importing torch or executing pickle."""
import json
import struct
from pathlib import Path
import numpy as np
from ntl.common import LABELS, POOL_LAYERS, sha256

def check_checkpoint(directory, allow_smoke=False):
    root = Path(directory).resolve()
    marker = root / "COMPLETE.json"
    if not marker.exists() and allow_smoke:
        marker = root / "SMOKE_ONLY.json"
    if not marker.is_file():
        raise ValueError("Missing completion marker; interrupted/smoke outputs are not production models")
    manifest = json.loads(marker.read_text())
    files = manifest["files"]
    required = ["adapter/adapter_config.json", "adapter/adapter_model.safetensors",
                "tokenizer/tokenizer_config.json", "heads.safetensors", "readouts.npz",
                "anchor_features.npz", "validation_features.npz", "base_config/config.json",
                "training_meta.json", "training_report.json"]
    for name in required:
        if name not in files:
            raise ValueError(f"Required file absent from manifest: {name}")
    for name, expected in files.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Invalid or missing manifest path: {name}")
        if sha256(path) != expected:
            raise ValueError(f"Hash mismatch: {name}")
    actual = {str(p.relative_to(root)).replace('\\', '/') for p in root.rglob('*') if p.is_file()}
    expected_files = {str(Path(k)).replace('\\', '/') for k in files} | {marker.name}
    if actual != expected_files:
        raise ValueError("Unmanifested files or conflicting completion markers found")
    meta = json.loads((root / "training_meta.json").read_text())
    report = json.loads((root / "training_report.json").read_text())
    if meta["labels"] != LABELS or meta["config"]["pool_layers"] != POOL_LAYERS:
        raise ValueError("Label order or pooled layers differ from the original reader contract")
    if meta["config"]["keep_layers"] != 24:
        raise ValueError("Expected 24-layer trunk")
    if meta["train_sha256"] != manifest["train_sha256"] or meta["job"] != manifest["job"]:
        raise ValueError("Manifest and metadata disagree")
    if marker.name == "COMPLETE.json" and (meta["smoke_only"] or report["status"] != "COMPLETE"):
        raise ValueError("Smoke/error output disguised as a complete model")
    if marker.name == "COMPLETE.json" and report["successful_updates"] != report["planned_updates"]:
        raise ValueError("Full training did not finish every planned update")
    adapter = json.loads((root / "adapter/adapter_config.json").read_text())
    if adapter.get("peft_type") != "LORA" or adapter.get("r") != meta["config"]["lora_rank"]:
        raise ValueError("Adapter rank/type differs from metadata")
    with np.load(root / "readouts.npz", allow_pickle=False) as ro:
        if ro["classes"].tolist() != LABELS or ro["pool_layers"].tolist() != POOL_LAYERS:
            raise ValueError("Readout order mismatch")
        for name, shape in {"plain_coef": (18, 32768), "center_coef": (18, 32768),
                            "plain_intercept": (18,), "center_intercept": (18,)}.items():
            if ro[name].shape != shape or not np.isfinite(ro[name]).all():
                raise ValueError(f"Invalid readout: {name}")
    with (root / "heads.safetensors").open('rb') as stream:
        size = struct.unpack('<Q', stream.read(8))[0]
        if size > 16 * 1024 * 1024:
            raise ValueError("Invalid safetensors header")
        header = json.loads(stream.read(size))
    for i in range(8):
        for field in ("weight", "bias"):
            if header.get(f"norms.{i}.{field}", {}).get("shape") != [4096]:
                raise ValueError("Missing or invalid per-pool LayerNorm")
    return meta
