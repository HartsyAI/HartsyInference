"""Reference dump for MiniMaxH3AudioVaeParityTests, using the module the checkpoint itself ships.

    python minimax_h3_audio_vae_ref.py <checkpoint>/audio_vae <dump-dir>

Writes latent.bin + wave_ref.bin (little-endian f32) for the C# side to replay.
"""
import sys, json, torch
from pathlib import Path
D = Path(sys.argv[1])
# the bundle uses package-relative imports; expose it as a package
import importlib.util, types
pkg = types.ModuleType("h3avae"); pkg.__path__ = [str(D)]; sys.modules["h3avae"] = pkg
def load(name):
    spec = importlib.util.spec_from_file_location(f"h3avae.{name}", D / f"{name}.py")
    m = importlib.util.module_from_spec(spec); sys.modules[f"h3avae.{name}"] = m
    spec.loader.exec_module(m); return m
for n in ["dac_utils","dac_activations","dac_alias_free_filter","dac_alias_free_resample",
          "dac_alias_free_act","dac_attn_proj","dac_bigvgan","dac_audio_vae"]:
    load(n)
mm = load("minimax_h3_audio_vae")
model = mm.MiniMaxH3AudioVAE.from_pretrained(str(D))

OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
cfg = json.load(open(D / "config.json"))
mean = torch.tensor(cfg["latents_mean"], dtype=torch.float32).view(1, -1, 1)
std = torch.tensor(cfg["latents_std"], dtype=torch.float32).view(1, -1, 1)

# An ENCODED signal, not white noise: random latents are far out of distribution and decode ~30 dB louder
# than anything the DiT emits, so tolerances measured on them say nothing about the real operating point.
import math
SR = 32000
n = int(SR * 0.4)
tt = torch.arange(n, dtype=torch.float32) / SR
sig = (0.3 * torch.sin(2 * math.pi * (2000 + 1500 * torch.sin(2 * math.pi * 6 * tt)) * tt)
       * (0.5 + 0.5 * torch.sign(torch.sin(2 * math.pi * 3 * tt)))).expand(2, n).contiguous().unsqueeze(0)
with torch.no_grad():
    b, s_, L = sig.shape
    pad = math.ceil(L / model.model.hop_length) * model.model.hop_length - L
    xp = torch.nn.functional.pad(sig, (0, pad)).reshape(b * s_, 1, -1)
    e = model.model.encoder(xp)
    e = model.model.pre_block(e.transpose(1, 2)).transpose(1, 2)
    zz = (model.model.mean_proj(e) - mean) / std
    z = zz.reshape(b, s_, zz.shape[1], zz.shape[2]).permute(0, 2, 1, 3).contiguous()
    T = z.shape[3]
    zd = z.permute(0, 2, 1, 3).reshape(b * s_, z.shape[1], T) * std + mean
    wav = model.decode(zd).reshape(b, s_, -1)
print("latent", tuple(z.shape), "wave", tuple(wav.shape), "rms", float(wav.pow(2).mean().sqrt()))
z.numpy().astype("<f4").tofile(OUT / "latent.bin")
wav.numpy().astype("<f4").tofile(OUT / "wave_ref.bin")
json.dump({"latent_shape": list(z.shape), "wave_shape": list(wav.shape)}, open(OUT/"meta.json","w"))
