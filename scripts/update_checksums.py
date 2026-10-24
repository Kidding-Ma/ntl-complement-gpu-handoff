"""Deliberately refresh delivery hashes after reviewed edits; run verification next."""
import hashlib, json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    files={}
    for p in sorted(ROOT.rglob('*')):
        rel=p.relative_to(ROOT)
        if not p.is_file() or any(x in rel.parts for x in ['.git','__pycache__','.venv','assets','outputs','reports']): continue
        if rel.as_posix()=='manifests/workspace_sha256.json': continue
        files[rel.as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
    (ROOT/'manifests/workspace_sha256.json').write_text(json.dumps(files,indent=2)+'\n',encoding='utf-8')
    print('SHA256 inventory:',len(files),'files')
