"""Archive public upstream notebook source; never execute downloaded code."""
import concurrent.futures, hashlib, json, urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def fetch(ref):
    url='https://www.kaggle.com/api/v1/kernels/pull/'+ref
    result={'ref':ref,'url':url}
    try:
        req=urllib.request.Request(url,headers={'User-Agent':'NTL-reproducibility-handoff'})
        with urllib.request.urlopen(req,timeout=45) as r: raw=r.read(); result['http_status']=r.status
        obj=json.loads(raw); src=obj['blob'].get('source') or obj['blob'].get('sourceNullable')
        nb=json.loads(src)
        dest=ROOT/'references'/ref.replace('/','__'); dest.mkdir(parents=True,exist_ok=True)
        (dest/'source.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
        (dest/'metadata.json').write_text(json.dumps(obj['metadata'],ensure_ascii=False,indent=2),encoding='utf-8')
        result.update(source_sha256=hashlib.sha256(src.encode()).hexdigest(),cells=len(nb['cells']),source_saved=True)
    except Exception as e:
        result.update(source_saved=False,error=str(e))
    return result

if __name__=='__main__':
    assets=json.loads((ROOT/'manifests/assets.json').read_text())['assets']
    refs=[x['ref'] for x in assets if x['kind']=='kernel']+['nishantdahal/two-stage-lora-part-two','lightningv08/ntl-lora24-fivefold-training']
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: results=list(pool.map(fetch,refs))
    (ROOT/'evidence/reference_fetch.json').write_text(json.dumps(results,indent=2))
    for r in results: print(r['ref'],r['source_saved'],r.get('http_status'),r.get('error',''))
