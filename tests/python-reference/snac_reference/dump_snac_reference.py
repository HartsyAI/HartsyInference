"""SNAC 24 kHz parity oracle.

Downloads hubertsiuzdak/snac_24khz, dumps the state dict as a safetensors the C# loader can read,
encodes a fixed pseudo-random waveform to codes, then decodes with torch.randn stubbed to zeros (so the
stochastic NoiseBlock becomes deterministic, matching the C# decoder run with NoiseScale=0). Saves the
codes and the decoded waveform to a reference safetensors.
"""
import os, numpy as np, torch
from snac import SNAC
from safetensors.torch import save_file

OUT = os.path.dirname(os.path.abspath(__file__))
W_PATH = os.path.join(OUT, "snac_24khz.safetensors")
REF_PATH = os.path.join(OUT, "snac_ref_io.safetensors")

torch.manual_seed(0)
model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()

# 1) weights -> safetensors (contiguous f32)
sd = {k: v.detach().contiguous().float().cpu() for k, v in model.state_dict().items()}
save_file(sd, W_PATH)
print("wrote weights:", W_PATH, "tensors:", len(sd))
print("sample keys:", [k for k in list(sd)[:6]])
print("has decoder noise linear:", any("noise" in k or ".block.2.linear" in k for k in sd))
print("decoder.model.0 keys:", [k for k in sd if k.startswith("decoder.model.0.")])
print("decoder.model.1 keys:", [k for k in sd if k.startswith("decoder.model.1.")])

# 2) fixed input -> codes
x = torch.randn(1, 1, 24000)  # 1 second
with torch.no_grad():
    codes = model.encode(x)
for i, c in enumerate(codes):
    print(f"codes[{i}] shape {tuple(c.shape)} dtype {c.dtype} min {int(c.min())} max {int(c.max())}")

# 3) decode with deterministic (zero) noise
orig_randn = torch.randn
def zero_randn(*a, **k):
    if len(a) == 1 and isinstance(a[0], (tuple, list, torch.Size)):
        return torch.zeros(*a, **k)
    return torch.zeros(*a, **k)
torch.randn = zero_randn
try:
    with torch.no_grad():
        audio = model.decode(codes)
finally:
    torch.randn = orig_randn
print("audio shape", tuple(audio.shape), "rms", float(audio.pow(2).mean().sqrt()))

ref = {f"codes_{i}": c.to(torch.int32).cpu().contiguous() for i, c in enumerate(codes)}
ref["audio_out"] = audio.float().cpu().contiguous()
ref["n_codebooks"] = torch.tensor([len(codes)], dtype=torch.int32)
save_file(ref, REF_PATH)
print("wrote ref:", REF_PATH)
