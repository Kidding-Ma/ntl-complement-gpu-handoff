from pathlib import Path
from safetensors.torch import load_file as original_load_file
from stage_local_checkpoint import stage_named_files
def load_file(filename,*args,**kwargs):
    path=Path(filename)
    if str(path).startswith('/kaggle/input/'):
        path=stage_named_files(path.parent,{path.name:path.name})/path.name
    return original_load_file(str(path),*args,**kwargs)
