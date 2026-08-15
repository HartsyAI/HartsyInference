"""Reference-side layer dump for the LTX-2 CONVOLUTIONAL video VAE decoder.

Runs ComfyUI's `causal_video_autoencoder.Decoder` — the reference implementation of this exact VAE — on a
file-supplied latent and writes raw f32 at the same points the C# side taps, so the two can be diffed stage by
stage with the pipeline out of the picture.

Deliberately drives conv_in / up_blocks / conv_out DIRECTLY rather than calling `decoder.forward()`, whose
chunked `run_up` writes into an output buffer and exposes no per-stage tensors. The direct drive reproduces
forward_orig's contract exactly: `mark_conv3d_ended` on every conv-owning module (so the non-causal trailing
replicate frame is produced) and `causal=False` on every call.

  ref_dump.py <outdir> [--misconfig]

`--misconfig` builds the reference the way our C# decoder is currently CONFIGURED rather than the way the
checkpoint's own metadata says: spatial_padding_mode="reflect" and residual=True on every DepthToSpaceUpsample.
If our decoder matches THAT bit-for-bit, the whole delta is those two flags and there is no third bug.
"""
import os, sys, json, struct
import numpy as np
import torch

COMFY = "/home/hartsy/Desktop/Swarm/SwarmUI.not too old/dlbackend/ComfyUI"
sys.path.insert(0, COMFY)
import comfy.ldm.lightricks.vae.causal_video_autoencoder as cva

CKPT = "/home/hartsy/Desktop/HartsyInference/Models/VAE/LTX-2/ltx-2.5-video-vae-conv-bf16.safetensors"

OUT = sys.argv[1]
_flags = sys.argv[2:]
MISCONFIG = "--misconfig" in _flags
REFLECT = MISCONFIG or "--reflect" in _flags       # spatial_padding_mode: zeros -> reflect
RESIDUAL = MISCONFIG or "--residual" in _flags     # DepthToSpaceUpsample.residual: False -> True
os.makedirs(OUT, exist_ok=True)

# ---- geometry ----
# Non-square, non-power-of-two H: a square latent hides an H/W transposition and both axes must exceed the
# reflect-pad width. Small enough that the whole decode is CPU-trivial on both sides.
T_LAT, H_LAT, W_LAT = 3, 6, 4

rng = np.random.RandomState(1234)
latent = rng.randn(1, 128, T_LAT, H_LAT, W_LAT).astype(np.float32)

meta = {}


def save(name, arr):
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    a.tofile(os.path.join(OUT, name + ".f32"))
    meta[name] = list(a.shape)
    print(f"  {name:16s} {str(list(a.shape)):26s} mean {a.mean():+.5f} std {a.std():.5f}")


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
        return out, hdr["__metadata__"]


state, file_meta = load_safetensors(CKPT)

# The checkpoint carries its OWN config; ComfyUI's sd.py prefers it over get_default_config(version), and the
# built-in version-2 default does NOT describe this file (5/5/5/5 res blocks + timestep_conditioning). Using the
# default would load partially and give a confident wrong reference — the strict assert below is what catches it.
cfg = json.loads(file_meta["config"])["vae"]
print("checkpoint config:")
print("  decoder_blocks       ", cfg["decoder_blocks"])
print("  spatial_padding_mode ", cfg.get("spatial_padding_mode", "<unset -> decoder default 'reflect'>"))
print("  timestep_conditioning", cfg.get("timestep_conditioning"))

if REFLECT:
    cfg = dict(cfg)
    cfg["spatial_padding_mode"] = "reflect"
    print("  [misconfig] spatial_padding_mode -> reflect")

vae = cva.VideoVAE(config=cfg)
missing, unexpected = vae.load_state_dict(state, strict=False)
# The trap the README names: a silent partial load leaves a conv at random init and every downstream comparison
# becomes a phantom. Assert the load is exact before trusting one number out of this script.
assert not missing, f"MISSING KEYS ({len(missing)}): {sorted(missing)[:12]}"
assert not unexpected, f"UNEXPECTED KEYS ({len(unexpected)}): {sorted(unexpected)[:12]}"
print(f"load OK: 0 missing, 0 unexpected ({len(state)} tensors)")

dec = vae.decoder.float().eval()
assert dec.causal is False, dec.causal
assert dec.timestep_conditioning is False, dec.timestep_conditioning
assert dec.patch_size == 4, dec.patch_size

# Structure the C# side must independently infer from weight shapes alone.
struct_desc = []
for i, blk in enumerate(dec.up_blocks):
    if isinstance(blk, cva.DepthToSpaceUpsample):
        if RESIDUAL:
            blk.residual = True
        struct_desc.append(dict(idx=i, kind="upsample", stride=list(blk.stride),
                                reduction=blk.out_channels_reduction_factor, residual=bool(blk.residual)))
    else:
        struct_desc.append(dict(idx=i, kind="res", num_layers=len(blk.res_blocks),
                                channels=blk.res_blocks[0].in_channels))
print("decoder structure:")
for d in struct_desc:
    print("  ", d)

# Name the dumps after the C# stage taps so the diff table lines up 1:1.
#   up_blocks.0            -> mid_resnets      (the C# decoder's mid block)
#   up_blocks.{2k+1}       -> up{k}_up         (upsampler)
#   up_blocks.{2k+2}       -> up{k}_res        (that stage's res blocks)
def tap_name(i):
    if i == 0:
        return "mid_resnets"
    return f"up{(i - 1) // 2}_up" if i % 2 == 1 else f"up{i // 2 - 1}_res"


with torch.no_grad():
    print("dumping:")
    save("latent", latent)
    z = torch.from_numpy(latent)

    x = vae.per_channel_statistics.un_normalize(z)
    save("denorm", x.numpy())

    cva.mark_conv3d_ended(dec.conv_in)
    x = dec.conv_in(x, causal=dec.causal)
    save("conv_in", x.numpy())

    for i, blk in enumerate(dec.up_blocks):
        cva.mark_conv3d_ended(blk)
        x = blk(x, causal=dec.causal)
        save(tap_name(i), x.numpy())

    x = dec.conv_norm_out(x)
    save("norm_out", x.numpy())
    x = dec.conv_act(x)

    cva.mark_conv3d_ended(dec.conv_out)
    x = dec.conv_out(x, causal=dec.causal)
    save("conv_out", x.numpy())

    x = cva.unpatchify(x, patch_size_hw=dec.patch_size, patch_size_t=1)
    save("pixels", x.numpy())

meta["_structure"] = struct_desc
meta["_misconfig"] = dict(reflect=REFLECT, residual=RESIDUAL)
with open(os.path.join(OUT, "shapes.json"), "w") as f:
    json.dump(meta, f, indent=1)
print("done ->", OUT, f"(reflect={REFLECT} residual={RESIDUAL})")
