"""Dumps Hunyuan3D-2 shape-model reference tensors (fp32, CPU) for Hunyuan3DShapeParityTests.

Run with `python -I` against a checkout of Tencent-Hunyuan/Hunyuan3D-2 (pinned f8db6309) and the
tencent/Hunyuan3D-2 checkpoint (revision 9cd649ba, hunyuan3d-dit-v2-0/model.fp16.safetensors + config.yaml).
Needs torch, transformers, torchvision, safetensors, einops, pyyaml, pillow, opencv-python, numpy.
The sampler mirrors Hunyuan3DDiTFlowMatchingPipeline with FlowMatchEulerDiscreteScheduler (shift 1) without diffusers.

  python -I dump_shape_reference.py --hy3dgen-src <checkout>/hy3dgen --ckpt model.fp16.safetensors \
      --config config.yaml --image demo.png --out <HARTSY_PARITY_DIR>/hunyuan3d2_shape_fp32.npz

The checkpoint's config.yaml selects the variant: `guidance_embed` runs one forward per step with the guidance embedded (Fast, Turbo),
the consistency scheduler picks sigmas from its pcm table (Turbo), and `num_latents` sets the latent count (mini). With
`--flashvdm-vae-dir` (the hunyuan3d-vae-v2-0-turbo files) the field is also decoded with FlashVDM, written as `flashvdm_grid`.
"""
import argparse, os, sys, time, types
import numpy as np
import torch
import yaml
from safetensors.torch import load_file

ap = argparse.ArgumentParser()
ap.add_argument("--hy3dgen-src", required=True, help="path of the hy3dgen package directory inside the checkout")
ap.add_argument("--ckpt", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--image", default="", help="RGBA foreground image (single-view checkpoints)")
ap.add_argument("--views", default="", help="multiview checkpoints: front=a.png,left=b.png,back=c.png,right=d.png (any subset)")
ap.add_argument("--out", required=True, help=".npz path")
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--res", type=int, default=128, help="octree resolution; the grid holds res+1 points per axis")
ap.add_argument("--cfg", type=float, default=5.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--threads", type=int, default=0)
ap.add_argument("--flashvdm-vae-dir", default="", help="directory with the turbo VAE config.yaml and model.fp16.safetensors")
args = ap.parse_args()
if args.threads:
    torch.set_num_threads(args.threads)

# Register the packages without running their __init__ files, which import diffusers and rendering extensions.
for name, sub in (("hy3dgen", ""), ("hy3dgen.shapegen", "shapegen"), ("hy3dgen.shapegen.models", "shapegen/models"),
                  ("hy3dgen.shapegen.models.autoencoders", "shapegen/models/autoencoders"),
                  ("hy3dgen.shapegen.models.denoisers", "shapegen/models/denoisers")):
    pkg = types.ModuleType(name)
    pkg.__path__ = [os.path.join(args.hy3dgen_src, sub)]
    sys.modules[name] = pkg

from hy3dgen.shapegen.models.denoisers.hunyuan3ddit import Hunyuan3DDiT
from hy3dgen.shapegen.models.autoencoders.model import ShapeVAE
from hy3dgen.shapegen.preprocessors import ImageProcessorV2, MVImageProcessorV2
from hy3dgen.shapegen.models.conditioner import DinoImageEncoderMV
from transformers import Dinov2Model, Dinov2Config
from PIL import Image
import torchvision.transforms as T

cfg = yaml.safe_load(open(args.config))
sd = load_file(args.ckpt, device="cpu")


def sub(prefix):
    return {k[len(prefix):]: v.float() for k, v in sd.items() if k.startswith(prefix)}


dit = Hunyuan3DDiT(**cfg["model"]["params"])
dit.load_state_dict(sub("model."), strict=True)
dit.eval()
vae = ShapeVAE(**cfg["vae"]["params"])
r = vae.load_state_dict(sub("vae."), strict=False)  # the bundled VAE is decode-only
assert not r.unexpected_keys and all(k.startswith(("encoder.", "pre_kl.")) for k in r.missing_keys)
vae.eval()
dcfg = cfg["conditioner"]["params"]["main_image_encoder"]["kwargs"]["config"]
dino = Dinov2Model(Dinov2Config.from_dict(dcfg))
dino.load_state_dict({k[len("main_image_encoder.model."):]: v for k, v in sub("conditioner.").items()}, strict=True)
dino.eval()

out = {}
multiview = cfg["conditioner"]["params"]["main_image_encoder"]["type"] == "DinoImageEncoderMV"
if multiview:
    # upstream's own multiview encoder: per-view DINOv2 tokens plus the sin/cos view embedding, concatenated in view order
    views = dict(item.split("=", 1) for item in args.views.split(","))
    mv = MVImageProcessorV2(size=512, border_ratio=0.15)({k: Image.open(v).convert("RGBA") for k, v in views.items()})
    enc = DinoImageEncoderMV(config=dcfg, image_size=518)
    enc.load_state_dict({k[len("main_image_encoder."):]: v for k, v in sub("conditioner.").items()}, strict=True)
    enc.eval()
    out["image_pre"] = mv["image"].numpy()
    out["view_idxs"] = np.array(mv["view_idxs"], dtype=np.int32)
    with torch.no_grad():
        cond = enc(mv["image"], view_idxs=[mv["view_idxs"]])
    uncond = torch.zeros_like(cond)
else:
    pre = ImageProcessorV2(size=512, border_ratio=0.15)(Image.open(args.image).convert("RGBA"))["image"]
    out["image_pre"] = pre.numpy()
    tf = T.Compose([T.Resize(518, T.InterpolationMode.BILINEAR, antialias=True), T.CenterCrop(518),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    with torch.no_grad():
        cond = dino(tf((pre + 1) / 2)).last_hidden_state
    uncond = torch.zeros_like(cond)
out["cond"] = cond.numpy()

params = cfg["model"]["params"]
guidance_embed = bool(params.get("guidance_embed", False))
if cfg["scheduler"]["target"].endswith("ConsistencyFlowMatchEulerDiscreteScheduler"):
    # mirrors ConsistencyFlowMatchEulerDiscreteScheduler.__init__ and set_timesteps
    pcm = cfg["scheduler"]["params"]["pcm_timesteps"]
    table = np.linspace(0, 1, 1000)[np.asarray([0] + (np.round(np.arange(1, pcm) * (1000 // pcm)).astype(np.int64) - 1).tolist())]
    table = table.astype(np.float32)
    sig = table[np.floor(np.linspace(0, pcm, num=args.steps, endpoint=False)).astype(np.int64)]
else:
    sig = np.linspace(0, 1, args.steps).astype(np.float32)
sigmas = np.concatenate([sig, [1.0]]).astype(np.float32)
latents = torch.randn(1, *vae.latent_shape, generator=torch.Generator().manual_seed(args.seed))
out["noise"] = latents.numpy()
saved = sorted({args.steps // 4, args.steps // 2, args.steps - 2})
out["steps"] = np.array([args.steps], dtype=np.int32)
out["octree_resolution"] = np.array([args.res], dtype=np.int32)
out["saved_steps"] = np.array(saved, dtype=np.int32)
t0 = time.time()
for i in range(args.steps):
    if guidance_embed:
        with torch.no_grad():  # one forward, the guidance scale embedded; no CFG batch
            c = dit(latents, torch.tensor([sig[i]]), {"main": cond}, guidance=torch.tensor([args.cfg]))
        v = c
    else:
        tin = torch.tensor([sig[i]]).expand(2)  # the pipeline feeds sigma = timestep / 1000
        with torch.no_grad():
            c, u = dit(torch.cat([latents, latents]), tin, {"main": torch.cat([cond, uncond])}).chunk(2)
        v = u + args.cfg * (c - u)
    if i in saved:
        out[f"lat_in_step{i}"] = latents.numpy()
        out[f"v_cond_step{i}"] = c.numpy()
    latents = latents + (sigmas[i + 1] - sigmas[i]) * v
    print("step", i, round(time.time() - t0), "s", flush=True)
out["latents_final"] = latents.numpy()

with torch.no_grad():
    processed = vae(1.0 / vae.scale_factor * latents)
    ax = np.linspace(-1.01, 1.01, args.res + 1, dtype=np.float32)
    xs, ys, zs = np.meshgrid(ax, ax, ax, indexing="ij")
    q = torch.from_numpy(np.stack([xs, ys, zs], -1).reshape(-1, 3))
    logits = torch.cat([vae.geo_decoder(queries=q[s:s + 8000][None], latents=processed)[0, :, 0]
                        for s in range(0, q.shape[0], 8000)]).reshape(args.res + 1, args.res + 1, args.res + 1)
out["vae_processed"] = processed.numpy()
out["grid_logits"] = logits.numpy()
if args.flashvdm_vae_dir:
    from hy3dgen.shapegen.models.autoencoders.model import ShapeVAE as TurboVAE
    vcfg = yaml.safe_load(open(os.path.join(args.flashvdm_vae_dir, "config.yaml")))
    tvae = TurboVAE(**vcfg["params"])
    tvae.load_state_dict(load_file(os.path.join(args.flashvdm_vae_dir, "model.fp16.safetensors")), strict=False)
    tvae.float().eval()
    tvae.enable_flashvdm_decoder(enabled=True, adaptive_kv_selection=True, topk_mode="mean", mc_algo="mc")
    with torch.no_grad():
        tprocessed = tvae(1.0 / tvae.scale_factor * latents)
        fgrid = tvae.volume_decoder(tprocessed, tvae.geo_decoder, bounds=1.01, num_chunks=10000,
                                    octree_resolution=args.res, mc_level=0.0, enable_pbar=False)
    out["flashvdm_processed"] = tprocessed.numpy()
    out["flashvdm_grid"] = fgrid[0].numpy()
np.savez(args.out, **out)
print("wrote", args.out)
