"""Move pooled vectors instead of complete token activations to the CPU.

LayerNorm remains on its original CPU path. Each new tensor geometry/device is
checked against the original arithmetic before the reduced-copy path is used.
The original implementation is retained as a numerical fallback.
"""
import json
from pathlib import Path
import torch

_checks = {}
_enabled = True

def pool_hidden(h, mask):
    global _enabled
    def original():
        return (h.float().cpu() * mask).sum(1) / mask.sum(1)
    if not _enabled or h.device.type == 'cpu':
        return original()
    gpu_mask = mask.to(h.device)
    result = ((h.float() * gpu_mask).sum(1) / gpu_mask.sum(1)).cpu()
    key = str((str(h.device), str(h.dtype), h.shape[-1]))
    if key not in _checks:
        reference = original()
        accepted = torch.allclose(result, reference, rtol=1e-4, atol=1e-5)
        _checks[key] = dict(accepted=bool(accepted), shape=list(h.shape),
            max_absolute_error=float((result-reference).abs().max()),
            relative_error=float(torch.linalg.vector_norm(result-reference) /
                torch.linalg.vector_norm(reference).clamp_min(1e-12)))
        print('GPU_POOL_PARITY', key, _checks[key], flush=True)
        Path('/kaggle/working/gpu_pool_parity.json').write_text(
            json.dumps(_checks, indent=2))
        if not accepted:
            _enabled = False
            return reference
    return result
