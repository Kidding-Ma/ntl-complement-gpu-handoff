"""Create a prompt-disjoint training plan from labeled training data."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ntl.common import make_plan, read_train, sha256, write_json

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-csv", required=True)
    p.add_argument("--base-model", required=True, help="Local complete Qwen3-8B Base directory")
    p.add_argument("--config", default="configs/train.json")
    p.add_argument("--output", default="artifacts/plan.json")
    args = p.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(f"Choose a fresh plan path: {args.output}")
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    base = Path(args.base_model).resolve()
    model_config = json.loads((base / "config.json").read_text(encoding="utf-8"))
    expected = {"model_type": "qwen3", "hidden_size": 4096, "num_hidden_layers": 36}
    if any(model_config.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Expected unmodified Qwen3-8B Base: {expected}")
    index = base / "model.safetensors.index.json"
    if index.exists():
        shards = set(json.loads(index.read_text())["weight_map"].values())
        if not all((base / name).is_file() for name in shards):
            raise ValueError("Some indexed model shards are missing")
    elif not (base / "model.safetensors").exists():
        raise ValueError("Missing model.safetensors or its shard index")
    if not (base / "tokenizer_config.json").is_file():
        raise ValueError("Missing tokenizer_config.json")
    frame = read_train(args.train_csv)
    plan = make_plan(frame, config["seed"], config["folds"])
    plan.update(train_csv=str(Path(args.train_csv).resolve()), train_sha256=sha256(args.train_csv),
                train_rows=len(frame), train_prompts=int(frame.prompt_id.nunique()),
                base_model=str(base), base_config_sha256=sha256(base / "config.json"), config=config,
                identity="independent retraining; no original-weight or score parity claim")
    write_json(args.output, plan)
    print(f"Prepared {len(frame)} rows / {frame.prompt_id.nunique()} prompts -> {args.output}")
    print("Jobs:", ", ".join(plan["jobs"]))

if __name__ == "__main__":
    main()
