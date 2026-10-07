"""Streams the released controlfoley.pth (11 GB fp32) over HTTP range requests into one bf16 safetensors (matrices and
convolution kernels bf16, everything else fp32), so the full network fits a 16 GB host.

Usage: python convert_network_bf16.py <output.safetensors> [url-or-path]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from safetensors.torch import save_file
from remote_pth import RemotePth

out = sys.argv[1]
src = sys.argv[2] if len(sys.argv) > 2 else "https://huggingface.co/YJX-Xiaomi/ControlFoley/resolve/main/weights/controlfoley.pth"
r = RemotePth(src)
tensors = {}
for i, key in enumerate(r.keys()):
    t = torch.from_numpy(r.read(key).copy())
    tensors[key] = t.to(torch.bfloat16) if t.dim() in (2, 3) else t.float()
    if i % 100 == 0:
        print(i, key, flush=True)
save_file(tensors, out)
print("done", len(tensors))
