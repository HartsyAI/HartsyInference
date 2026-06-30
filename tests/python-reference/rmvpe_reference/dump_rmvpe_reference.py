"""RMVPE E2E (mel -> 360-bin pitch posterior) parity oracle. Feeds a fixed random mel directly to the network
(isolating it from the mel front-end) and dumps the final hidden + per-stage tensors for C# parity debugging.
Reuses the standalone model classes in rmvpe_model.py (extracted from RVC-WebUI's rmvpe.py)."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safetensors.torch import save_file
from rmvpe_model import E2E

OUT = os.path.dirname(os.path.abspath(__file__))
CKPT = os.environ.get("RMVPE_PT", os.path.join(OUT, "rmvpe.pt"))
sd = torch.load(CKPT, map_location="cpu", weights_only=False)
if isinstance(sd, dict) and "model" in sd and not any(k.startswith("unet") for k in sd):
    sd = sd["model"]

model = E2E(4, 1, (2, 2))
missing, unexpected = model.load_state_dict(sd, strict=False)
print("missing", len(missing), "unexpected", len(unexpected))
if missing: print("  missing sample:", missing[:4])
if unexpected: print("  unexpected sample:", unexpected[:4])
model.eval()

torch.manual_seed(0)
# mel is [B, n_mels=128, T]; E2E.forward transposes to [B,1,T,128] internally. T multiple of 32 keeps shapes clean.
T = 64
mel = torch.randn(1, 128, T)
d = {}
with torch.no_grad():
    m = mel.transpose(-1, -2).unsqueeze(1)        # [1,1,T,128]
    u = model.unet(m)
    d["unet_out"] = u.float().contiguous()
    c = model.cnn(u)
    d["cnn_out"] = c.float().contiguous()
    flat = c.transpose(1, 2).flatten(-2)          # [1,T,384]
    d["flat"] = flat.float().contiguous()
    gru = model.fc[0](flat)                        # BiGRU -> [1,T,512]
    d["gru_out"] = gru.float().contiguous()
    hidden = model.fc(flat)                        # full fc incl Linear+sigmoid -> [1,T,360]
    d["hidden"] = hidden.float().contiguous()
d["mel"] = mel.float().contiguous()
for k, v in d.items():
    print(f"{k:10s} {tuple(v.shape)}  mean={float(v.mean()):+.4f} std={float(v.std()):.4f}")

# also dump the weights as safetensors so the C# test loads identical tensors without a .pt pickle loader.
w = {k: v.float().contiguous() for k, v in sd.items()}
save_file(w, os.path.join(OUT, "rmvpe_weights.safetensors"))
save_file(d, os.path.join(OUT, "rmvpe_ref_io.safetensors"))
print("wrote rmvpe weights + ref")
