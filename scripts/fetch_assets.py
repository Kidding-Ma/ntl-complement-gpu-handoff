"""Download original inputs to the layout expected by unmodified upstream code."""
import argparse, json, subprocess, sys, time, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def unpack(root):
    for archive in root.glob('*.zip'):
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                target = (root / info.filename).resolve()
                if not target.is_relative_to(root.resolve()):
                    raise ValueError('Unsafe archive path')
            z.extractall(root)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=Path('/kaggle/input'))
    p.add_argument('--only', action='append', help='Exact ref; repeat for several. Default: all inputs.')
    p.add_argument('--list', action='store_true')
    a = p.parse_args()
    assets = json.loads((ROOT/'manifests/assets.json').read_text())['assets']
    if a.only:
        unknown=set(a.only)-{x['ref'] for x in assets}
        if unknown: p.error('Unknown refs: '+repr(unknown))
        assets=[x for x in assets if x['ref'] in a.only]
    if a.list:
        for x in assets: print(x['kind'], x['ref'], '->', a.input/x['relative_dir'])
        return
    results=[]
    for x in assets:
        dest=a.input/x['relative_dir']; dest.mkdir(parents=True,exist_ok=True)
        kind,ref=x['kind'],x['ref']
        if kind=='dataset': cmd=['kaggle','datasets','download','-d',ref,'-p',str(dest),'--unzip']
        elif kind=='kernel': cmd=['kaggle','kernels','output',ref,'-p',str(dest)]
        elif kind=='model': cmd=['kaggle','models','instances','versions','download',ref,'-p',str(dest),'--untar']
        else: cmd=['kaggle','competitions','download','-c',ref,'-p',str(dest)]
        print('DOWNLOAD',ref,flush=True)
        started=time.monotonic()
        proc=subprocess.run(cmd)
        result=dict(kind=kind,ref=ref,path=str(dest),returncode=proc.returncode,seconds=round(time.monotonic()-started,1))
        if proc.returncode==0:
            unpack(dest)
            files=[p for p in dest.rglob('*') if p.is_file()]
            result.update(file_count=len(files),bytes=sum(p.stat().st_size for p in files))
            (dest/'handoff_download_receipt.json').write_text(json.dumps(result,indent=2))
        results.append(result)
    (ROOT/'reports').mkdir(exist_ok=True)
    (ROOT/'reports/download_results.json').write_text(json.dumps(results,indent=2))
    if any(r['returncode'] for r in results):
        sys.exit('Some original inputs unavailable. Read reports/download_results.json. Do not replace or skip members silently.')

if __name__=='__main__': main()
