"""Reference-side layer dump for the LTX-2.5 LATENT SPATIAL UPSAMPLER (x2).

Runs ComfyUI's `comfy.ldm.lightricks.latent_upsampler.LatentUpsampler` — the reference implementation of this
exact module — on a file-supplied latent and writes raw f32 at the same points the C# side taps, so the two can
be diffed stage by stage with the pipeline out of the picture.

    ref_dump.py <outdir>

Run under the ComfyUI venv (`dlbackend/ComfyUI/venv/bin/python`); the module chain pulls in `comfy_aimdo`, which
only that venv has.

The module is built from the checkpoint's OWN `__metadata__.config`, never from the class defaults — the
reference defaults `mid_channels` to 512 and this file is 1024, so a default-built reference would load
partially and give a confident wrong answer. The strict-load assert below is what catches that.
"""
import os, sys, json, struct
import numpy as np
import torch

COMFY = "/home/hartsy/Desktop/Swarm/SwarmUI.not too old/dlbackend/ComfyUI"
sys.path.insert(0, COMFY)
import comfy.ops
from comfy.ldm.lightricks.latent_upsampler import LatentUpsampler

CKPT = ("/home/hartsy/Desktop/HartsyInference/Models/latent_upscale_models/"
        "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors")

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)

# ---- geometry ----
# All three axes distinct so a D/H/W transposition cannot hide, and small enough that the C# side's naive
# gather Conv3d finishes in seconds: mid_channels is 1024 and the post-upsample stage runs at 4x the voxels.
T_LAT, H_LAT, W_LAT = 2, 4, 3

rng = np.random.RandomState(4321)
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
            out[k] = t.float()          # C# casts these weights to F32 — match it
        return out, hdr["__metadata__"]


state, file_meta = load_safetensors(CKPT)
cfg = json.loads(file_meta["config"])
print("checkpoint config:", cfg)
assert cfg["_class_name"] == "LatentUpsampler", cfg["_class_name"]

model = LatentUpsampler.from_config(cfg, operations=comfy.ops.disable_weight_init)
missing, unexpected = model.load_state_dict(state, strict=False)
assert not missing, f"MISSING KEYS ({len(missing)}): {sorted(missing)[:12]}"
assert not unexpected, f"UNEXPECTED KEYS ({len(unexpected)}): {sorted(unexpected)[:12]}"
print(f"load OK: 0 missing, 0 unexpected ({len(state)} tensors)")
model = model.float().eval()
assert model.dims == 3 and model.spatial_upsample and not model.temporal_upsample
assert not model.rational_resampler

latent = rng.randn(1, cfg["in_channels"], T_LAT, H_LAT, W_LAT).astype(np.float32)

with torch.no_grad():
    print("dumping:")
    save("latent", latent)
    x = torch.from_numpy(latent)

    x = model.initial_conv(x)
    save("initial_conv", x.numpy())
    x = model.initial_activation(model.initial_norm(x))
    save("initial_act", x.numpy())

    for i, block in enumerate(model.res_blocks):
        x = block(x)
        save(f"res{i}", x.numpy())

    b, c, f = x.shape[0], x.shape[1], x.shape[2]
    x = x.permute(0, 2, 1, 3, 4).reshape(b * f, c, x.shape[3], x.shape[4])
    x = model.upsampler[0](x)
    save("up_conv", x.permute(0, 1, 2, 3).reshape(b, f, x.shape[1], x.shape[2], x.shape[3])
         .permute(0, 2, 1, 3, 4).numpy())
    x = model.upsampler[1](x)
    x = x.reshape(b, f, x.shape[1], x.shape[2], x.shape[3]).permute(0, 2, 1, 3, 4)
    save("up_shuffle", x.numpy())

    for i, block in enumerate(model.post_upsample_res_blocks):
        x = block(x)
        save(f"post{i}", x.numpy())

    x = model.final_conv(x)
    save("output", x.numpy())

    # The direct drive above must reproduce forward() exactly; a stage tap that quietly diverges from the real
    # module ordering would make every comparison below it meaningless.
    ref = model(torch.from_numpy(latent))
    assert torch.equal(ref, x), (ref - x).abs().max().item()

meta["_config"] = cfg
with open(os.path.join(OUT, "shapes.json"), "w") as f:
    json.dump(meta, f, indent=1)
print("done ->", OUT)
