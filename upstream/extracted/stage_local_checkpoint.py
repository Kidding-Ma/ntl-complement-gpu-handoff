"""Sequentially stage a small immutable checkpoint, preserving every file byte.

Intended for the slow mounted Ultrabert fold checkpoints. No weight changes;
source/destination hashes are recorded. Uses scratch outside saved outputs.
"""
from pathlib import Path
import hashlib,json,shutil,time

def stage_named_files(source,names):
    """Copy selected flattened public files to scratch under local aliases."""
    source=Path(source).resolve();assert source.is_relative_to(Path('/kaggle/input').resolve())
    for origin,alias in names.items():
        assert Path(origin).name==origin and Path(alias).name==alias
        assert (source/origin).is_file()
    key=hashlib.sha256((str(source)+json.dumps(names,sort_keys=True)).encode()).hexdigest()[:16]
    target=Path('/kaggle/temp/ntl-selected-checkpoints')/key;target.mkdir(parents=True,exist_ok=True)
    ready=target/'stage-audit.json'
    if not ready.exists():
        total=sum((source/name).stat().st_size for name in names)
        assert shutil.disk_usage(target).free>total+2**30
        records={};began=time.monotonic()
        for name,alias in names.items():
            digest=hashlib.sha256();count=0
            with (source/name).open('rb') as src,(target/alias).open('wb') as dst:
                while chunk:=src.read(16*1024*1024):dst.write(chunk);digest.update(chunk);count+=len(chunk)
            assert count==(source/name).stat().st_size
            with (target/alias).open('rb') as check:assert hashlib.file_digest(check,'sha256').hexdigest()==digest.hexdigest()
            records[name]=dict(local_alias=alias,bytes=count,sha256=digest.hexdigest())
        ready.write_text(json.dumps(dict(source=str(source),files=records,seconds=time.monotonic()-began),indent=2))
        print('VERIFIED_SELECTED_CHECKPOINT_STAGE',total,round(time.monotonic()-began,2),flush=True)
    audit_dir=Path('/kaggle/working/model_stage_audits');audit_dir.mkdir(exist_ok=True)
    (audit_dir/(key+'.json')).write_bytes(ready.read_bytes())
    return target

def stage_checkpoint(source,max_bytes=3*2**30):
    source=Path(source).resolve()
    assert source.is_relative_to(Path('/kaggle/input').resolve())
    files=[p for p in source.rglob('*') if p.is_file()]
    total=sum(p.stat().st_size for p in files)
    if total>max_bytes:
        print('CHECKPOINT_STAGE_SIZE_GUARD',str(source),total,flush=True);return source
    target=Path('/kaggle/temp/ntl-small-checkpoints')/hashlib.sha256(str(source).encode()).hexdigest()[:16]
    ready=target/'stage-audit.json'
    if ready.exists():
        audit_dir=Path('/kaggle/working/model_stage_audits');audit_dir.mkdir(exist_ok=True)
        (audit_dir/(target.name+'.json')).write_bytes(ready.read_bytes())
        return target
    target.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(target).free<total+2**30:
        print('CHECKPOINT_STAGE_DISK_GUARD',str(source),flush=True);return source
    began=time.monotonic();records={}
    try:
        for origin in files:
            relative=origin.relative_to(source);destination=target/relative;destination.parent.mkdir(parents=True,exist_ok=True);digest=hashlib.sha256();count=0
            with origin.open('rb') as src,destination.open('wb') as dst:
                while chunk:=src.read(16*1024*1024):dst.write(chunk);digest.update(chunk);count+=len(chunk)
            assert count==origin.stat().st_size
            with destination.open('rb') as dst:assert hashlib.file_digest(dst,'sha256').hexdigest()==digest.hexdigest()
            records[str(relative)]={'bytes':count,'sha256':digest.hexdigest()}
        ready.write_text(json.dumps(dict(source=str(source),files=records,seconds=time.monotonic()-began),indent=2))
        audit_dir=Path('/kaggle/working/model_stage_audits');audit_dir.mkdir(exist_ok=True)
        (audit_dir/(target.name+'.json')).write_bytes(ready.read_bytes())
        print('VERIFIED_LOCAL_CHECKPOINT_STAGE',str(source),total,round(time.monotonic()-began,2),flush=True);return target
    except Exception as exc:
        print('CHECKPOINT_STAGE_FALLBACK',str(source),type(exc).__name__,flush=True);return source
