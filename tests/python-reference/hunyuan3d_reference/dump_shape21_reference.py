"""Dumps Hunyuan3D-2.1 shape-model reference tensors (fp32, CPU) for Hunyuan3D21ShapeParityTests.

Run with `python -I` against a checkout of Tencent-Hunyuan/Hunyuan3D-2.1 (pinned 82920d64) and the tencent/Hunyuan3D-2.1
checkpoint (revision 0b946776, hunyuan3d-dit-v2-1) repacked to safetensors with the engine's safe pickle repacker (the upstream
file is a pickle and must never be loaded with torch.load here). Needs torch, transformers, torchvision, safetensors, einops,
pyyaml, pillow, opencv-python, numpy; `timm` and `diffusers` are replaced by the small stand-ins below.

  python -I dump_shape21_reference.py --hy3dshape-src <checkout>/hy3dshape/hy3dshape --ckpt model.fp16.safetensors \
      --config config.yaml --image demo.png --out <HARTSY_PARITY_DIR>/hunyuan3d_21_shape_fp32.npz

It runs the DiT at the given sigmas with the image tokens and once with the zero tokens, records the outputs of a few blocks, and
decodes a random latent set on a 65^3 grid with the bundled VAE.
"""
import argparse, os, sys, time, types
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file

ap = argparse.ArgumentParser()
ap.add_argument("--hy3dshape-src", required=True, help="path of the hy3dshape package directory inside the checkout")
ap.add_argument("--ckpt", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--image", required=True, help="RGBA foreground image")
ap.add_argument("--out", required=True, help=".npz path")
ap.add_argument("--sigmas", default="0.35,0.7")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--threads", type=int, default=0)
args = ap.parse_args()
if args.threads:
    torch.set_num_threads(args.threads)


from transformers import Dinov2Model, Dinov2Config  # before the stand-ins: transformers probes for a real timm


# Stand-ins for the two libraries the reference imports only for FeedForward and unused layers.
class _GELUProj(nn.Module):
    def __init__(self, dim_in, dim_out, bias=True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)

    def forward(self, x):
        return F.gelu(self.proj(x))


class FeedForward(nn.Module):  # diffusers.models.attention.FeedForward, activation_fn="gelu"
    def __init__(self, dim, dropout=0.0, activation_fn="gelu", final_dropout=False, inner_dim=None, bias=True):
        super().__init__()
        inner_dim = inner_dim or dim * 4
        self.net = nn.ModuleList([_GELUProj(dim, inner_dim, bias=bias), nn.Dropout(dropout), nn.Linear(inner_dim, dim, bias=bias)])

    def forward(self, x):
        for module in self.net:
            x = module(x)
        return x


def stub(name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules[name] = module
    return module


stub("timm"); stub("timm.models"); stub("timm.models.vision_transformer", PatchEmbed=None, Attention=None, Mlp=None)
stub("diffusers"); stub("diffusers.models"); stub("diffusers.models.attention", FeedForward=FeedForward)

# Register the packages without running their __init__ files, which pull in training and rendering dependencies.
for name, sub in (("hy3dshape", ""), ("hy3dshape.models", "models"), ("hy3dshape.models.denoisers", "models/denoisers"),
                  ("hy3dshape.models.autoencoders", "models/autoencoders")):
    pkg = types.ModuleType(name)
    pkg.__path__ = [os.path.join(args.hy3dshape_src, sub)]
    sys.modules[name] = pkg


class _Timer:
    def __init__(self, name=None):
        pass

    def __call__(self, fn):
        return fn

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


import logging
stub("hy3dshape.utils", logger=logging.getLogger("hy3dshape"), synchronize_timer=_Timer, smart_load_model=None)

from hy3dshape.models.denoisers.hunyuandit import HunYuanDiTPlain
from hy3dshape.models.autoencoders.model import ShapeVAE
from hy3dshape.preprocessors import ImageProcessorV2
from PIL import Image
import torchvision.transforms as T

with open(args.config) as config_file:
    cfg = yaml.safe_load(config_file)
sd = load_file(args.ckpt, device="cpu")


def sub_state(prefix):
    return {k[len(prefix):]: v.float() for k, v in sd.items() if k.startswith(prefix)}


dit = HunYuanDiTPlain(**cfg["model"]["params"])
dit.load_state_dict(sub_state("model."), strict=True)
dit.eval()
vae = ShapeVAE(**cfg["vae"]["params"])
vae.load_state_dict(sub_state("vae."), strict=True)
vae.eval()
dcfg = cfg["conditioner"]["params"]["main_image_encoder"]["kwargs"]["config"]
dino = Dinov2Model(Dinov2Config.from_dict(dcfg))
dino.load_state_dict({k[len("main_image_encoder.model."):]: v for k, v in sub_state("conditioner.").items()}, strict=True)
dino.eval()

out = {}
pre = ImageProcessorV2(size=512, border_ratio=0.15)(Image.open(args.image).convert("RGBA"))["image"]
out["image_pre"] = pre.numpy()
tf = T.Compose([T.Resize(518, T.InterpolationMode.BILINEAR, antialias=True), T.CenterCrop(518),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
t0 = time.time()
with torch.no_grad():
    cond = dino(tf((pre + 1) / 2)).last_hidden_state
out["cond"] = cond.numpy()
print("dino", tuple(cond.shape), round(time.time() - t0), "s", flush=True)

latents = torch.randn(1, cfg["model"]["params"]["input_size"], 64, generator=torch.Generator().manual_seed(args.seed))
out["noise"] = latents.numpy()
captured = {}
for index in (0, 10, 15, 20):
    dit.blocks[index].register_forward_hook(lambda m, i, o, index=index: captured.__setitem__(f"block{index}", o.detach().numpy()))
dit.x_embedder.register_forward_hook(lambda m, i, o: captured.__setitem__("x_embedded", o.detach().numpy()))
dit.t_embedder.register_forward_hook(lambda m, i, o: captured.__setitem__("t_token", o.detach().numpy()))

sigmas = [float(s) for s in args.sigmas.split(",")]
out["sigmas"] = np.array(sigmas, dtype=np.float32)
for n, sigma in enumerate(sigmas):
    with torch.no_grad():
        v = dit(latents, torch.tensor([sigma]), {"main": cond})
    out[f"velocity_cond_{n}"] = v.numpy()
    if n == 0:
        for key, value in captured.items():
            out[f"{key}"] = value
    print("sigma", sigma, round(time.time() - t0), "s", flush=True)
with torch.no_grad():
    out["velocity_uncond_0"] = dit(latents, torch.tensor([sigmas[0]]), {"main": torch.zeros_like(cond)}).numpy()
print("uncond", round(time.time() - t0), "s", flush=True)

# VAE: decode a random latent set on a 65^3 grid (x slowest), as the engine's Decode does.
vae_latents = torch.randn(1, cfg["vae"]["params"]["num_latents"], 64, generator=torch.Generator().manual_seed(args.seed + 1))
out["vae_latents"] = vae_latents.numpy()
with torch.no_grad():
    processed = vae(vae_latents / vae.scale_factor)
    axis = np.linspace(-1.01, 1.01, 65, dtype=np.float32)
    xs, ys, zs = np.meshgrid(axis, axis, axis, indexing="ij")
    queries = torch.from_numpy(np.stack([xs, ys, zs], -1).reshape(-1, 3))
    logits = torch.cat([vae.geo_decoder(queries=queries[s:s + 8000][None], latents=processed)[0, :, 0]
                        for s in range(0, queries.shape[0], 8000)]).reshape(65, 65, 65)
out["vae_grid"] = logits.numpy()
np.savez(args.out, **out)
print("wrote", args.out)
