"""Idol conditioning on the CPU in float32: preprocessing, DINOv3 ViT-H+/16 tokens and FLUX.2 VAE tokens.

Usage: python -I dump_idol_conditioning.py <image> <out dir> [--mode synthetic|real]

The image must carry a real alpha channel (BiRefNet matting is compared on its own, in dump_birefnet_matting.py). An RGB image
is given one here by keying out its plain background (corner colour, tolerance 24), and that RGBA is written as input.png, so
the engine test reads exactly the pixels this script saw.

Writes conditioning.safetensors:
  input_rgba       uint8 [H, W, 4]        the image the script preprocessed (its alpha is the matte)
  prepared         uint8 [1024, 1024, 3]  idol.preprocess.preprocess_image output (composited on black)
  pixels           [1, 3, 1024, 1024]     prepared / 255
  dino_prenorm     [1, 4101, 1280]        hidden state after the last block, before DINOv3's own final norm
  dino_last        [1, 4101, 1280]        HF last_hidden_state (after DINOv3's final norm)
  feature1         [1, 4101, 1280]        F.layer_norm(dino_last) without affine: Idol's feature1
  vae_mean         [1, 32, 128, 128]      FLUX.2 VAE posterior mode (Budget3D's IdolEncoder)
  feature2         [1, 4101, 128]         5 zero tokens + patchified, BatchNorm-normalised mean
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # common.py; python -I drops the script folder
import common

ap = argparse.ArgumentParser()
ap.add_argument("image")
ap.add_argument("out")
ap.add_argument("--mode", default="synthetic", choices=["synthetic", "real"])
a = ap.parse_args()

from idol.conditioning import IMAGENET_MEAN, IMAGENET_STD
from idol.preprocess import CROP_MARGIN, has_real_alpha, preprocess_image

image = Image.open(a.image)
if not has_real_alpha(image):
    rgb = np.asarray(image.convert("RGB")).astype(np.int32)
    corner = np.median(np.stack([rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]]), axis=0)
    alpha = (np.abs(rgb - corner).max(-1) > 24).astype(np.uint8) * 255
    image = Image.fromarray(np.dstack([rgb.astype(np.uint8), alpha]), "RGBA")
os.makedirs(a.out, exist_ok=True)
image.save(os.path.join(a.out, "input.png"))

prepared = preprocess_image(image, background_remover=None, image_size=1024, crop_margin=CROP_MARGIN[a.mode])
pixels = torch.from_numpy(np.array(prepared)).permute(2, 0, 1).float().div(255.0)[None]

# DINOv3 ViT-H+/16 from the Comfy repack (HF layout, BF16 values), run in float32
from transformers import DINOv3ViTConfig, DINOv3ViTModel
cfg = DINOv3ViTConfig(patch_size=16, hidden_size=1280, intermediate_size=5120, num_hidden_layers=32, num_attention_heads=20,
                      num_register_tokens=4, use_gated_mlp=True, hidden_act="silu")
dino = DINOv3ViTModel(cfg)
state = common.f32(common.load_file(os.path.join(common.IDOL_DIR, "clip_vision", "dino_v3_vit_h_bf16.safetensors")))
missing, unexpected = dino.load_state_dict(state, strict=False)
print("dinov3 missing", missing, "unexpected", unexpected)
dino = dino.float().eval()
mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
with torch.no_grad(), common.Timer("dinov3"):
    out = dino(pixel_values=(pixels - mean) / std, output_hidden_states=True)
dino_last = out.last_hidden_state.float()
dino_prenorm = out.hidden_states[-1].float()
feature1 = F.layer_norm(dino_last, dino_last.shape[-1:])

# FLUX.2 VAE from the Comfy repack (ldm key layout), run in float32
from diffusers import AutoencoderKLFlux2
from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint
# FLUX.2-klein-4B vae/config.json (the config Budget3D loads); the Comfy file is the same VAE in ldm key layout
vae_cfg = dict(act_fn="silu", batch_norm_eps=1e-4, batch_norm_momentum=0.1, block_out_channels=[128, 256, 512, 512],
               down_block_types=["DownEncoderBlock2D"] * 4, force_upcast=True, in_channels=3, latent_channels=32,
               layers_per_block=2, mid_block_add_attention=True, norm_num_groups=32, out_channels=3, patch_size=[2, 2],
               sample_size=1024, up_block_types=["UpDecoderBlock2D"] * 4, use_post_quant_conv=True, use_quant_conv=True)
vae = AutoencoderKLFlux2(**vae_cfg)
ldm = common.f32({k: v for k, v in common.load_file(os.path.join(common.IDOL_DIR, "vae", "flux2-vae.safetensors")).items()
                  if v.is_floating_point()})
state = convert_ldm_vae_checkpoint(ldm, vae_cfg)
state.update({k: v for k, v in ldm.items() if k.startswith("bn.")})
# ComfyUI keeps the quant convs inside the encoder and decoder
for src, dst in (("encoder.quant_conv", "quant_conv"), ("decoder.post_quant_conv", "post_quant_conv")):
    for p in ("weight", "bias"):
        state[f"{dst}.{p}"] = ldm[f"{src}.{p}"]
missing, unexpected = vae.load_state_dict(state, strict=False)
print("vae missing", missing, "unexpected", unexpected)
assert not [k for k in missing if not k.endswith("num_batches_tracked")] and not unexpected
vae = vae.float().eval()
with torch.no_grad(), common.Timer("vae"):
    vae_mean = vae.encode(pixels * 2 - 1).latent_dist.mode()
b, c, h, w = vae_mean.shape
lat = vae_mean.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)
bn_mean = vae.bn.running_mean.view(1, -1, 1, 1)
bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps)
tokens = ((lat - bn_mean) / bn_std).float().flatten(2).transpose(1, 2).contiguous()
feature2 = torch.cat([torch.zeros(b, 5, tokens.shape[2]), tokens], dim=1)

common.save(a.out, "conditioning", {
    "input_rgba": torch.from_numpy(np.array(image.convert("RGBA"))),
    "prepared": torch.from_numpy(np.array(prepared)), "pixels": pixels, "dino_prenorm": dino_prenorm, "dino_last": dino_last,
    "feature1": feature1, "vae_mean": vae_mean, "feature2": feature2,
}, meta={"mode": a.mode, "batch_norm_eps": vae.config.batch_norm_eps, "image": os.path.basename(a.image)})
