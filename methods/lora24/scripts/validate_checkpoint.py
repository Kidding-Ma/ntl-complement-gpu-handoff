import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ntl.checkpoint import check_checkpoint

if __name__ == '__main__':
    p = argparse.ArgumentParser(description="CPU structural/hash check; GPU parity is a separate required step")
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--allow-smoke', action='store_true')
    args = p.parse_args()
    meta = check_checkpoint(args.checkpoint, args.allow_smoke)
    print(json.dumps({"structure_and_hashes": "PASS", "job": meta['job'],
                      "smoke_only": meta['smoke_only'], "gpu_parity": "NOT_CHECKED"}, indent=2))
