"""Reference-side layer dump for the LTX-2.5 NA diffusion decoder.

Runs ComfyUI's NADiffusionDecoder on a file-supplied latent + noise and writes raw f32 dumps at the same
points the C# side dumps, so the two can be diffed. Deliberately calls forward_pre_diffusion /
forward_diff_step directly rather than forward(), whose internal randn cannot be fed from a file.
"""
import os, sys, json, struct
import numpy as np
import torch

COMFY = "/home/hartsy/Desktop/Swarm/SwarmUI.not too old/dlbackend/ComfyUI"
sys.path.insert(0, COMFY)
from comfy.ldm.lightricks.vae.na_diffusion_decoder import NADiffusionDecoder

OUT = sys.argv[1]
CKPT = "/home/hartsy/Desktop/HartsyInference/Models/VAE/LTX-2/ltx-2.5-video-vae-bf16.safetensors"
os.makedirs(OUT, exist_ok=True)

# ---- geometry (small: the reference's fp32 na3d runs eager) ----
T_LAT, H_LAT, W_LAT = 3, 8, 4
torch.manual_seed(0)
rng = np.random.RandomState(1234)
latent = rng.randn(1, 128, T_LAT, H_LAT, W_LAT).astype(np.float32)

def save(name, arr):
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    a.tofile(os.path.join(OUT, name + ".f32"))
    meta[name] = list(a.shape)
    print(f"  {name:28s} {str(list(a.shape)):28s} mean {a.mean():+.5f} std {a.std():.5f}")

meta = {}

# ---- load ----
def load_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            s, e = v["data_offsets"]
            f.seek(base + s)
            raw = f.read(e - s)
            assert v["dtype"] == "BF16", v["dtype"]
            t = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(v["shape"])
            out[k] = t.float()          # C# casts VAE weights to F32 — match it
        return out

full = load_safetensors(CKPT)
state = {k[len("decoder."):]: v for k, v in full.items() if k.startswith("decoder.")}

dec = NADiffusionDecoder()
missing, unexpected = dec.load_state_dict(state, strict=False)
# The trap the advisor named: a silent partial load leaves nn.Linear at random init and every downstream
# comparison becomes a phantom. Assert the load is exact before trusting one number out of this script.
assert not missing, f"MISSING KEYS ({len(missing)}): {missing[:10]}"
assert list(unexpected) == ["type_emb"], f"UNEXPECTED: {unexpected}"
print(f"load OK: 0 missing, unexpected == ['type_emb']")
dec = dec.float().eval().cuda()

z = torch.from_numpy(latent).cuda()

with torch.no_grad():
    print("dumping:")
    save("latent", latent)

    # ---- stages 1-4 -> context, with per-stage taps ----
    n = dec.trailing_pad_latent_frames
    x = torch.cat([z, z[:, :, -1:].expand(-1, -1, n, -1, -1)], dim=2)
    x = x.permute(0, 2, 3, 4, 1)
    x = dec.conv_in(x)
    save("conv_in", x.cpu().numpy())
    for si, blocks in enumerate(dec.det_stages):
        for b in blocks:
            x = b(x)
        save(f"stage{si}_blocks", x.cpu().numpy())
        x = dec.upsamples[si](x, drop_leading_frame=True)
        save(f"stage{si}_up", x.cpu().numpy())
    x = x[:, :-(n * dec.temporal_upscale)]
    context = x
    save("context", context.cpu().numpy())

    b_, t5, h5, w5, _ = context.shape
    noise = rng.randn(1, dec.out_channels, t5, h5 * dec.patch_size, w5 * dec.patch_size).astype(np.float32)
    save("noise", noise)
    x_t = torch.from_numpy(noise).cuda()

    # ---- stage 5, tapped per diff block ----
    from comfy.ldm.lightricks.vae.na_diffusion_decoder import patchify, unpatchify
    xp = patchify(x_t, patch_size_hw=dec.patch_size, patch_size_t=1)
    h = dec.conv_in_x_t(xp.permute(0, 2, 3, 4, 1))
    save("conv_in_x_t", h.cpu().numpy())
    t_now = torch.tensor([1.0], device=z.device)
    t_emb = dec.t_embedder(dec.timestep_scale_multiplier * t_now, dtype=h.dtype)
    save("t_emb", t_emb.cpu().numpy())
    modulation = dec.shared_adaln(t_emb)
    save("modulation", torch.cat([m.reshape(1, -1) for m in modulation], dim=0).cpu().numpy())
    for i, blk in enumerate(dec.diff_blocks):
        h = blk(h, context, modulation)
        save(f"diff{i}", h.cpu().numpy())
    h = dec.norm_out(h)
    save("norm_out", h.cpu().numpy())
    h = dec.conv_out(h)
    save("conv_out", h.cpu().numpy())
    out = unpatchify(h.permute(0, 4, 1, 2, 3), patch_size_hw=dec.patch_size, patch_size_t=1)
    save("pixels", out.cpu().numpy())

with open(os.path.join(OUT, "shapes.json"), "w") as f:
    json.dump(meta, f, indent=1)
print("done ->", OUT)
