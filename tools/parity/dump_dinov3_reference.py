"""Offline DINOv3 ViT-L/16 reference dump for HartsyInference parity. Engine code never imports or runs this.

The forward follows the official code at facebookresearch/dinov3 @ 6876159a11b4df116f30f667f8c9888617df0751
(dinov3/models/vision_transformer.py, dinov3/layers/block.py, attention.py, rope_position_encoding.py), in float32 on
CPU. Weights are read from an HF-layout safetensors file and mapped with HairDeck's hub_state_dict
(hyz317/HairDeck @ cb20f7073419a33c9ab88f22f686fcb9060a99f0, hairdeck/models/imgmodel.py), including its zero qkv
bias mask. The image is a fixed synthetic pattern, so no image decoder is involved.

Usage (run isolated, the weights file is untrusted data):
    python -I dump_dinov3_reference.py <model.safetensors> <out_dir>
"""
import hashlib
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

RES = 448
PATCH = 16
HEADS = 16
DEPTH = 24
DIM = 1024
STORAGE = 4
EPS = 1e-5  # dinov3_vitl16 uses norm_layer "layernormbf16" (eps 1e-5)
DUMP_BLOCKS = (0, 11, 23)


def hub_state_dict(sd, depth, heads):
    """HairDeck imgmodel.hub_state_dict, verbatim."""
    out = {'cls_token': sd['embeddings.cls_token'], 'storage_tokens': sd['embeddings.register_tokens'],
           'mask_token': sd['embeddings.mask_token'].reshape(1, -1),
           'patch_embed.proj.weight': sd['embeddings.patch_embeddings.weight'],
           'patch_embed.proj.bias': sd['embeddings.patch_embeddings.bias'],
           'norm.weight': sd['norm.weight'], 'norm.bias': sd['norm.bias']}
    names = {'norm1': 'norm1', 'norm2': 'norm2', 'attention.o_proj': 'attn.proj', 'mlp.up_proj': 'mlp.fc1',
             'mlp.down_proj': 'mlp.fc2'}
    for i in range(depth):
        s, t = f'layer.{i}.', f'blocks.{i}.'
        for a, b in names.items():
            for p in ('weight', 'bias'):
                out[f'{t}{b}.{p}'] = sd[f'{s}{a}.{p}']
        out[f'{t}ls1.gamma'] = sd[f'{s}layer_scale1.lambda1']
        out[f'{t}ls2.gamma'] = sd[f'{s}layer_scale2.lambda1']
        q, k, v = (f'{s}attention.{x}_proj' for x in 'qkv')
        out[f'{t}attn.qkv.weight'] = torch.cat([sd[f'{q}.weight'], sd[f'{k}.weight'], sd[f'{v}.weight']])
        out[f'{t}attn.qkv.bias'] = torch.cat([sd[f'{q}.bias'], torch.zeros_like(sd[f'{q}.bias']), sd[f'{v}.bias']])
        out[f'{t}attn.qkv.bias_mask'] = torch.zeros_like(out[f'{t}attn.qkv.bias'])
    d = sd['norm.weight'].numel() // heads
    out['rope_embed.periods'] = (100.0 ** (2 * torch.arange(d // 4, dtype=torch.float32) / (d // 2))).to(torch.bfloat16).float()
    return out


def rope_sincos(H, W, periods):
    """RopePositionEmbedding.forward for normalize_coords='separate', eval mode (no shift, jitter or rescale)."""
    coords_h = torch.arange(0.5, H, dtype=torch.float32) / H
    coords_w = torch.arange(0.5, W, dtype=torch.float32) / W
    coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'), dim=-1).flatten(0, 1)
    coords = 2.0 * coords - 1.0
    angles = 2 * math.pi * coords[:, :, None] / periods[None, None, :]
    angles = angles.flatten(1, 2).tile(2)
    return torch.sin(angles), torch.cos(angles)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def attention(x, sd, i, sin, cos, qkv_bias):
    B, N, C = x.shape
    qkv = F.linear(x, sd[f'blocks.{i}.attn.qkv.weight'], qkv_bias)
    q, k, v = qkv.reshape(B, N, 3, HEADS, C // HEADS).unbind(2)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    prefix = N - sin.shape[0]
    q_rot = q[:, :, prefix:] * cos + rotate_half(q[:, :, prefix:]) * sin
    k_rot = k[:, :, prefix:] * cos + rotate_half(k[:, :, prefix:]) * sin
    q = torch.cat([q[:, :, :prefix], q_rot], dim=2)
    k = torch.cat([k[:, :, :prefix], k_rot], dim=2)
    out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, C)
    return F.linear(out, sd[f'blocks.{i}.attn.proj.weight'], sd[f'blocks.{i}.attn.proj.bias'])


def block(x, sd, i, sin, cos, qkv_bias):
    x = x + sd[f'blocks.{i}.ls1.gamma'] * attention(
        F.layer_norm(x, (DIM,), sd[f'blocks.{i}.norm1.weight'], sd[f'blocks.{i}.norm1.bias'], EPS), sd, i, sin, cos, qkv_bias)
    h = F.layer_norm(x, (DIM,), sd[f'blocks.{i}.norm2.weight'], sd[f'blocks.{i}.norm2.bias'], EPS)
    h = F.gelu(F.linear(h, sd[f'blocks.{i}.mlp.fc1.weight'], sd[f'blocks.{i}.mlp.fc1.bias']))
    h = F.linear(h, sd[f'blocks.{i}.mlp.fc2.weight'], sd[f'blocks.{i}.mlp.fc2.bias'])
    return x + sd[f'blocks.{i}.ls2.gamma'] * h


def forward(pixels, sd, qkv_bias_fn, dump):
    """Returns the normed token sequence and dumps prenorm stages through `dump(name, tensor)`."""
    x = F.conv2d(pixels, sd['patch_embed.proj.weight'], sd['patch_embed.proj.bias'], stride=PATCH)
    grid = x.shape[-1]
    x = x.flatten(2).transpose(1, 2)
    x = torch.cat([sd['cls_token'].expand(1, -1, -1), sd['storage_tokens'].expand(1, -1, -1), x], dim=1)
    dump('embed', x)
    sin, cos = rope_sincos(grid, grid, sd['rope_embed.periods'])
    for i in range(DEPTH):
        x = block(x, sd, i, sin, cos, qkv_bias_fn(i))
        if i in DUMP_BLOCKS:
            dump(f'block{i:02d}', x)
    return F.layer_norm(x, (DIM,), sd['norm.weight'], sd['norm.bias'], EPS)


def write_f32(path, tensor):
    arr = tensor.detach().float().contiguous().numpy()
    arr.astype('<f4').tofile(path)
    return list(arr.shape)


def synthetic_image():
    y, x = np.mgrid[0:RES, 0:RES]
    img = np.zeros((RES, RES, 3), np.uint8)
    img[..., 0] = (x * 7 + y * 3) % 256
    img[..., 1] = (x * 2 + y * 11) % 256
    img[..., 2] = (((x // 16) + (y // 16)) % 2) * 255
    return img


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    weights_path, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    torch.manual_seed(0)

    raw = load_file(weights_path)
    sd = hub_state_dict(raw, DEPTH, HEADS)
    qv_bias_absmax = {
        'q': max(float(raw[f'layer.{i}.attention.q_proj.bias'].abs().max()) for i in range(DEPTH)),
        'v': max(float(raw[f'layer.{i}.attention.v_proj.bias'].abs().max()) for i in range(DEPTH)),
    }

    img = synthetic_image()
    img.tofile(os.path.join(out_dir, 'input_rgb_u8.bin'))
    pixels = torch.from_numpy(img).permute(2, 0, 1).float().div(255).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    pixels = (pixels - mean) / std
    shapes = {'pixels': write_f32(os.path.join(out_dir, 'pixels_f32.bin'), pixels)}

    dumped = {}

    def dump(name, tensor):
        dumped[name] = tensor.detach().clone()

    with torch.no_grad():
        zero_bias = lambda i: sd[f'blocks.{i}.attn.qkv.bias'] * sd[f'blocks.{i}.attn.qkv.bias_mask']
        final = forward(pixels, sd, zero_bias, dump)
        for name, tensor in dumped.items():
            shapes[name] = write_f32(os.path.join(out_dir, f'{name}_f32.bin'), tensor[0])
        shapes['final'] = write_f32(os.path.join(out_dir, 'final_f32.bin'), final[0])

        # Official mask semantics (k-bias masked, q and v biases kept), to show whether HairDeck's zero mask matters.
        official_bias = lambda i: torch.cat([raw[f'layer.{i}.attention.q_proj.bias'],
                                             torch.zeros_like(raw[f'layer.{i}.attention.q_proj.bias']),
                                             raw[f'layer.{i}.attention.v_proj.bias']])
        final_official = forward(pixels, sd, official_bias, lambda name, t: None)
        diff = (final_official - final).abs()

    meta = {
        'weights': os.path.basename(weights_path),
        'weights_sha256': sha256_of(weights_path),
        'res': RES, 'patch': PATCH, 'grid': RES // PATCH, 'tokens': 1 + STORAGE + (RES // PATCH) ** 2,
        'hidden': DIM, 'depth': DEPTH, 'heads': HEADS, 'layernorm_eps': EPS,
        'dump_blocks': list(DUMP_BLOCKS), 'shapes': shapes,
        'periods': [float(v) for v in sd['rope_embed.periods']],
        'qv_bias_absmax_in_file': qv_bias_absmax,
        'official_mask_vs_zero_mask_max_abs': float(diff.max()),
        'official_mask_vs_zero_mask_rms': float(diff.pow(2).mean().sqrt()),
        'torch': torch.__version__,
    }
    with open(os.path.join(out_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print(json.dumps({k: meta[k] for k in ('weights_sha256', 'tokens', 'qv_bias_absmax_in_file',
                                           'official_mask_vs_zero_mask_max_abs', 'torch')}, indent=2))


if __name__ == '__main__':
    main()
