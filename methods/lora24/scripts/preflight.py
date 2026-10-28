"""Check hardware, imports and input contracts before spending GPU time."""
import argparse
import importlib.metadata
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ntl.common import sha256, write_json

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", default="artifacts/preflight.json")
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    report = {"python": sys.version, "passed": False, "packages": {}}
    try:
        for name in ["torch", "transformers", "peft", "accelerate", "safetensors", "numpy", "pandas"]:
            report["packages"][name] = importlib.metadata.version(name)
        import torch
        import peft
        import transformers
        assert torch.cuda.is_available(), "CUDA unavailable; this step needs a GPU machine"
        x = torch.randn(32, 32, device='cuda', requires_grad=True)
        (x @ x).square().mean().backward()
        torch.cuda.synchronize()
        assert bool(torch.isfinite(x.grad).all()), 'CUDA kernel/backward failed'
        del x
        torch.cuda.empty_cache()
        report['cuda_kernel_and_backward'] = 'PASS'
        report['torch_cuda'] = torch.version.cuda
        report["gpus"] = [{"name": torch.cuda.get_device_name(i),
                           "total_bytes": torch.cuda.get_device_properties(i).total_memory,
                           "free_bytes": torch.cuda.mem_get_info(i)[0]}
                          for i in range(torch.cuda.device_count())]
        assert sha256(plan["train_csv"]) == plan["train_sha256"], "Training CSV changed"
        assert sha256(Path(plan["base_model"]) / "config.json") == plan["base_config_sha256"]
        report["plan_sha256"] = sha256(args.plan)
        report["passed"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    write_json(args.output, report)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
