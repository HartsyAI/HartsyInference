"""
TripoSR (stabilityai/TripoSR) parity oracle.

Runs the REAL upstream `tsr` library on the downloaded `model.ckpt` and dumps:
  1. triposr.safetensors        full F32 state dict (so the C# SafeTensorsLoader can read it; the
                                published checkpoint is a torch pickle .ckpt)
  2. triposr_ref_io.safetensors per-stage reference IO for the C# parity test:
       pixels            [1,3,512,512]   raw conditioning image in [0,1] (PRE imagenet-norm)
       pos_embed_interp  [1,1025,768]    bicubic-interpolated ViT position embeddings (14x14 -> 32x32)
       image_tokens      [1,1025,768]    DINO ViT last_hidden_state (encoder_hidden_states)
       triplane_tokens   [1,1024,3072]   Triplane1DTokenizer output
       backbone_out      [1,1024,3072]   Transformer1D output
       scene_codes       [1,3,40,64,64]  post_processor (upsampled triplane)
       probe_points      [N,3]           random query points in [-radius,radius]
       probe_density_act [N,1]           exp(density_raw + density_bias) at the probe points
       probe_color       [N,3]           sigmoid(features) at the probe points
       grid_density_act  [R*R*R,1]       density_act over the extract_mesh grid (R=GRID_RES), ij order

Usage:
  python3 tests/python-reference/triposr_reference/dump_triposr_reference.py
Env:
  TRIPOSR_CKPT_DIR   dir with model.ckpt + config.yaml (default /tmp/triposr)
"""
import os
import sys

import numpy as np
import torch
from safetensors.torch import save_file

CKPT_DIR = os.environ.get("TRIPOSR_CKPT_DIR", "/tmp/triposr")
TSR_SRC = os.environ.get("TRIPOSR_SRC", "/tmp/TripoSR")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
N_PROBE = 4096
GRID_RES = 64

sys.path.insert(0, TSR_SRC)
# torchmcubes needs CUDA-compiled native code we don't have; parity compares the density grid directly,
# so stub it out to allow importing tsr.system.
import types  # noqa: E402
_stub = types.ModuleType("torchmcubes")
_stub.marching_cubes = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stubbed"))
sys.modules["torchmcubes"] = _stub
# rembg (background removal) pulls in onnxruntime; unused for parity -> stub.
sys.modules["rembg"] = types.ModuleType("rembg")
sys.modules["rembg"].remove = lambda *a, **k: a[0]
sys.modules["rembg"].new_session = lambda *a, **k: None
from tsr.system import TSR  # noqa: E402
from tsr.utils import scale_tensor  # noqa: E402


@torch.no_grad()
def main():
    torch.manual_seed(0)
    model = TSR.from_pretrained(CKPT_DIR, config_name="config.yaml", weight_name="model.ckpt")
    model.eval()
    radius = model.renderer.cfg.radius
    print(f"loaded TSR, renderer.radius={radius}, density_act={model.renderer.cfg.density_activation}, "
          f"density_bias={model.renderer.cfg.density_bias}")

    # --- full state dict -> safetensors (F32, contiguous) ---
    sd = {k: v.float().contiguous() for k, v in model.state_dict().items()}
    save_file(sd, os.path.join(OUT_DIR, "triposr.safetensors"))
    print(f"wrote triposr.safetensors ({len(sd)} tensors)")

    refs = {}

    # --- conditioning image: fixed [1,3,512,512] in [0,1] (pre imagenet-norm) ---
    pixels = torch.rand(1, 3, 512, 512)
    refs["pixels"] = pixels.clone()

    tok = model.image_tokenizer
    mean = tok.image_mean.view(1, 3, 1, 1)
    std = tok.image_std.view(1, 3, 1, 1)
    norm_pixels = (pixels - mean) / std

    emb = tok.model.embeddings
    # interpolated position embeddings (the bicubic step the C# must reproduce)
    patch = emb.patch_embeddings(norm_pixels, interpolate_pos_encoding=True)  # [1,1024,768]
    cls = emb.cls_token.expand(1, -1, -1)
    embeddings = torch.cat((cls, patch), dim=1)  # [1,1025,768]
    pos = emb.interpolate_pos_encoding(embeddings, 512, 512)  # [1,1025,768]
    refs["pos_embed_interp"] = pos.clone()

    out = tok.model(norm_pixels, interpolate_pos_encoding=True)
    image_tokens = out.last_hidden_state  # [1,1025,768]
    refs["image_tokens"] = image_tokens.clone()

    triplane_tokens = model.tokenizer(1)  # [1,1024,3072]
    refs["triplane_tokens"] = triplane_tokens.clone()

    backbone_out = model.backbone(triplane_tokens, encoder_hidden_states=image_tokens)
    refs["backbone_out"] = backbone_out.clone()

    scene_codes = model.post_processor(model.tokenizer.detokenize(backbone_out))  # [1,3,40,64,64]
    refs["scene_codes"] = scene_codes.clone()
    scene_code = scene_codes[0]  # [3,40,64,64]

    # --- probe points: random in [-radius, radius] ---
    probe = (torch.rand(N_PROBE, 3) * 2 - 1) * radius
    net = model.renderer.query_triplane(model.decoder, probe, scene_code)
    refs["probe_points"] = probe.clone()
    refs["probe_density_act"] = net["density_act"].reshape(N_PROBE, -1).clone()
    refs["probe_color"] = net["color"].reshape(N_PROBE, 3).clone()

    # --- density grid over the extract_mesh grid (ij order, [0,1]->[-r,r]) ---
    lin = torch.linspace(0, 1, GRID_RES)
    gx, gy, gz = torch.meshgrid(lin, lin, lin, indexing="ij")
    grid = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=-1)  # [R^3,3] in [0,1]
    grid = scale_tensor(grid, (0, 1), (-radius, radius))
    gnet = model.renderer.query_triplane(model.decoder, grid, scene_code)
    refs["grid_density_act"] = gnet["density_act"].reshape(-1, 1).clone()

    refs = {k: v.float().detach().cpu().contiguous() for k, v in refs.items()}
    save_file(refs, os.path.join(OUT_DIR, "triposr_ref_io.safetensors"))
    for k, v in refs.items():
        print(f"  {k:18s} {tuple(v.shape)}")
    print(f"wrote triposr_ref_io.safetensors -> {OUT_DIR}")


if __name__ == "__main__":
    main()
