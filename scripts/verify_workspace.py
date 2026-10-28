"""Check retained file hashes, runnable Python/notebook syntax and local links."""
import ast
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    errors = []
    hashes = json.loads((ROOT / "manifests/workspace_sha256.json").read_text(encoding="utf-8"))
    for name, expected in hashes.items():
        path = ROOT / name
        if not path.is_file():
            errors.append("MISSING " + name)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append("CHANGED " + name)

    python_files = 0
    for folder in (ROOT / "scripts", ROOT / "methods"):
        for path in folder.rglob("*.py"):
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                python_files += 1
            except SyntaxError as exc:
                errors.append(str(exc))

    cells = 0
    for name in ("verified/score_0636/ntl_complement_fixed.ipynb",
                 "methods/lora24/received/own_lora24_ensemble_fixed.ipynb"):
        notebook = json.loads((ROOT / name).read_text(encoding="utf-8"))
        for i, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            source = "\n".join(line for line in source.splitlines()
                                if not line.lstrip().startswith(("%", "!")))
            try:
                ast.parse(source, filename=f"{name}:cell{i}")
                cells += 1
            except SyntaxError as exc:
                errors.append(str(exc))

    links = 0
    for path in ROOT.rglob("*.md"):
        if any(part in {".git", ".venv", "outputs"} for part in path.relative_to(ROOT).parts):
            continue
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if target.startswith(("https://", "http://", "#", "mailto:")):
                continue
            links += 1
            if not (path.parent / target.split("#")[0]).exists():
                errors.append(f"BROKEN LINK {path.relative_to(ROOT)}: {target}")

    weight = ROOT / "checkpoints/own_lora24_checkpoint.zip"
    expected = "3d84cdadc2c2741c21c60cdead22a57076aca88563631da9719c4bba998de28b"
    if weight.is_file() and weight.stat().st_size > 1024:
        weight_status = "hash_verified"
        if hashlib.sha256(weight.read_bytes()).hexdigest() != expected:
            errors.append("CHANGED LoRA24 ZIP")
    else:
        weight_status = "optional_lfs_weight_not_downloaded"

    print(json.dumps(dict(ok=not errors, files_hashed=len(hashes),
                          python_files_parsed=python_files, notebook_cells_parsed=cells,
                          local_links_checked=links, lora24=weight_status, errors=errors), indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
