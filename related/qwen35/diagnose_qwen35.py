"""Run in a fresh Kaggle GPU session with the same inputs as version 361482022.

Diagnostic only: load the same truncated Qwen3.5 backbone and forward one real
training example. This does not create a submission or change ensemble weights.
Python stack traces repeat every 120 seconds to locate an otherwise silent wait.
"""
import faulthandler
import gc
import glob
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

START = time.monotonic()


def report(stage):
    print(f"\n[{time.monotonic() - START:.1f}s] {stage}", flush=True)
    try:
        import psutil
        proc = psutil.Process()
        print(f"RAM available={psutil.virtual_memory().available / 2**30:.2f} GiB; "
              f"process RSS={proc.memory_info().rss / 2**30:.2f} GiB", flush=True)
    except ImportError:
        pass
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        print("GPU index, utilization %, memory used MiB, total MiB:\n" +
              (result.stdout or result.stderr), flush=True)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        print(f"GPU status unavailable: {type(exc).__name__}", flush=True)


def main():
    if '--sync' in sys.argv:
        os.environ['HF_DEACTIVATE_ASYNC_LOAD'] = '1'
    print('HF_DEACTIVATE_ASYNC_LOAD=', os.environ.get('HF_DEACTIVATE_ASYNC_LOAD', 'unset'), flush=True)
    faulthandler.enable()
    faulthandler.dump_traceback_later(120, repeat=True)
    try:
        report("START: imports")
        import pandas as pd
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        for package in ("torch", "transformers", "accelerate", "safetensors"):
            try:
                print(package, importlib.metadata.version(package), flush=True)
            except importlib.metadata.PackageNotFoundError:
                print(package, "not installed", flush=True)
        if not torch.cuda.is_available():
            raise RuntimeError("No GPU is available. Select the same GPU accelerator first.")

        base = Path("/kaggle/input/datasets/bbucxi/qwen35-9b-safetensors")
        if not (base / "config.json").is_file():
            matches = []
            for filename in glob.glob("/kaggle/input/**/config.json", recursive=True):
                config = json.loads(Path(filename).read_text())
                if config.get("model_type") == "qwen3_5":
                    matches.append(Path(filename).parent)
            if len(matches) != 1:
                raise RuntimeError(f"Select the original Qwen3.5 directory explicitly: {matches}")
            base = matches[0]

        heads = sorted(glob.glob("/kaggle/input/**/tokattn_final.pt", recursive=True))
        if len(heads) != 1:
            raise RuntimeError(f"Expected one Qwen3.5 attention checkpoint: {heads}")
        report("START: load attention checkpoint metadata")
        head = torch.load(heads[0], map_location="cpu", weights_only=True)
        config = AutoConfig.from_pretrained(str(base), local_files_only=True)
        text_config = getattr(config, "text_config", None) or config
        text_config.num_hidden_layers = int(head["keep"])
        if getattr(text_config, "layer_types", None):
            text_config.layer_types = text_config.layer_types[:int(head["keep"])]
        print("base:", base, "keep:", head["keep"], "layer:", head["layer"], flush=True)

        gc.collect()
        torch.cuda.empty_cache()
        report("START: AutoModel.from_pretrained")
        model = AutoModel.from_pretrained(
            str(base), config=config, dtype=torch.float16,
            device_map="auto", local_files_only=True).eval()
        report("DONE: AutoModel.from_pretrained")
        print("device map:", getattr(model, "hf_device_map", None), flush=True)

        report("START: tokenizer and one training example")
        tokenizer = AutoTokenizer.from_pretrained(str(base), local_files_only=True)
        tokenizer.padding_side = "right"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        training_paths = [Path("/kaggle/input/competitions/name-that-llm/train.csv"),
                          Path("/kaggle/input/name-that-llm/train.csv")]
        training = next((p for p in training_paths if p.is_file()), None)
        if training is None:
            raise FileNotFoundError("Attach the Name That LLM competition data.")
        text = str(pd.read_csv(training, nrows=1).iloc[0]["text"])
        batch = tokenizer([text], add_special_tokens=not head.get("decoder", True),
                          truncation=True, max_length=int(head["max_len"]),
                          return_tensors="pt")
        device = model.get_input_embeddings().weight.device
        batch = batch.to(device)
        print("input shape:", tuple(batch["input_ids"].shape), "device:", device, flush=True)
        report("START: single-example forward")
        with torch.inference_mode():
            output = model(input_ids=batch["input_ids"],
                           attention_mask=batch["attention_mask"], output_hidden_states=True)
        for index in range(torch.cuda.device_count()):
            torch.cuda.synchronize(index)
        report("DONE: single-example forward")
        states = getattr(output, "hidden_states", None)
        if states is None:
            states = output.language_model_outputs.hidden_states
        state = states[int(head["layer"])]
        print("selected hidden-state shape:", tuple(state.shape),
              "finite:", bool(torch.isfinite(state).all()), flush=True)
        report("DIAGNOSTIC COMPLETE: backbone load and one forward succeeded")
    finally:
        faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
