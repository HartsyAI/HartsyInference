# Regenerates sana_wm_s2_cam_branch.safetensors: one tiny SANA-WM camera-branch case (T=4, H_lat=2, W_lat=3,
# one head of dim 112) evaluated with the pinned upstream torch reference functions on CPU.
# Run: <venv>/bin/python generate_sana_wm_s2_cam_branch_fixture.py <upstream Sana src dir> <Fixtures dir>
# The upstream modules import fla and triton at module level; fla is stubbed (only ShortConvolution is named)
# and the bidirectional short conv is transcribed from BidirectionalGDN._bidirectional_causal_conv_1d.
import json
import math
import os
import sys
import types

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["GDN_DISABLE_COMPILE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

UP = sys.argv[1]
OUT_DIR = sys.argv[2]


def _package(name, path):
    module = types.ModuleType(name)
    module.__path__ = [path]
    sys.modules[name] = module


_package("diffusion", os.path.join(UP, "diffusion"))
_package("diffusion.model", os.path.join(UP, "diffusion/model"))
_package("diffusion.model.nets", os.path.join(UP, "diffusion/model/nets"))
_package("diffusion.model.ops", os.path.join(UP, "diffusion/model/ops"))
_fla = types.ModuleType("fla")
_fla.__path__ = []
_fla_modules = types.ModuleType("fla.modules")
_fla_modules.ShortConvolution = type("ShortConvolution", (), {})
_fla.modules = _fla_modules
sys.modules["fla"] = _fla
sys.modules["fla.modules"] = _fla_modules
_chunkwise = types.ModuleType("diffusion.model.ops.fused_gdn_chunkwise")
_chunkwise.cam_scan_chunkwise = None
sys.modules["diffusion.model.ops.fused_gdn_chunkwise"] = _chunkwise
sys.path.insert(0, UP)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from diffusion.model.nets.sana_camctrl_blocks import _prepare_ray_apply_fns  # noqa: E402
from diffusion.model.ops.fused_cam_gdn import (  # noqa: E402
    _invert_SE3,
    _prepare_ucpe_rope_tables,
    _process_camera_conditions_raymats_only,
    _torch_cam_prep_reference,
    _torch_cam_scan_reference,
)

T, H_LAT, W_LAT = 4, 2, 3
S = H_LAT * W_LAT
N = T * S
HEADS = 1
HEAD_DIM = 112
C = HEADS * HEAD_DIM
THETA = 10000.0
EPS = 1e-5
PATCH = (1, 1, 1)
HW = (T, H_LAT, W_LAT)
B = 1


def bidirectional_short_conv(x, weight):
    """BidirectionalGDN._bidirectional_causal_conv_1d over (B, N, C) tokens, with ShortConvolution as a causal conv."""
    batch, n, channels = x.shape
    xt = x.reshape(batch, T, S, channels).permute(0, 2, 1, 3).contiguous().reshape(batch * S, T, channels)
    kernel = weight.shape[-1]

    def causal(seq_bct):
        return F.conv1d(F.pad(seq_bct, (kernel - 1, 0)), weight, groups=channels)

    xc = xt.transpose(1, 2)
    y_fwd = causal(xc).transpose(1, 2)
    y_bwd = causal(xc.flip(2)).transpose(1, 2).flip(1)
    center = xt * weight[:, 0, -1].view(1, 1, channels)
    y = y_fwd + y_bwd - center
    return y.reshape(batch, S, T, channels).permute(0, 2, 1, 3).reshape(batch, n, channels)


def main_rope_tables():
    """Complex 3D Wan RoPE, columns [t(20) | h(18) | w(18)] for head dim 112, built in float64 and cast."""
    h_dim = 2 * (HEAD_DIM // 6)
    t_dim = HEAD_DIM - 2 * h_dim

    def axis(dim, pos):
        inv = 1.0 / (THETA ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
        return pos.double()[:, None] * inv[None, :]

    frames = torch.arange(T).repeat_interleave(S)
    rows = torch.arange(H_LAT).repeat_interleave(W_LAT).repeat(T)
    cols = torch.arange(W_LAT).repeat(T * H_LAT)
    angles = torch.cat([axis(t_dim, frames), axis(h_dim, rows), axis(h_dim, cols)], dim=1)
    return torch.cos(angles).float(), torch.sin(angles).float()


def rodrigues(axis, angle):
    k = axis
    K = torch.tensor([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], dtype=torch.float64)
    return torch.eye(3, dtype=torch.float64) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)


def camera_conditions(gen):
    rows = []
    for t in range(T):
        axis = torch.randn(3, generator=gen, dtype=torch.float64)
        axis = axis / axis.norm()
        angle = 0.35 * float(torch.randn(1, generator=gen, dtype=torch.float64))
        c2w = torch.eye(4, dtype=torch.float64)
        c2w[:3, :3] = rodrigues(axis, angle)
        c2w[:3, 3] = 0.8 * torch.randn(3, generator=gen, dtype=torch.float64)
        intrinsics = [3.0 + 0.1 * t, 3.0 - 0.05 * t, 1.0 + 0.02 * t, 0.5 - 0.01 * t]
        rows.append(torch.cat([c2w.reshape(-1), torch.tensor(intrinsics, dtype=torch.float64)]))
    return torch.stack(rows).float().unsqueeze(0)


def linear_params(gen, out_dim, scale):
    weight = torch.randn(out_dim, C, generator=gen) * (scale / math.sqrt(C))
    bias = 0.1 * torch.randn(out_dim, generator=gen)
    return weight.float(), bias.float()


def upstream_cam_branch(x, cc, full_cos, full_sin, beta, decay, p):
    q_raw = F.linear(x, p["q_w"], p["q_b"])
    k_raw = F.linear(x, p["k_w"], p["k_b"])
    v_raw = F.linear(x, p["v_w"], p["v_b"])
    k_raw = bidirectional_short_conv(k_raw, p["conv_w"])

    P = _process_camera_conditions_raymats_only(cc, B, HW, PATCH).reshape(B, N, 4, 4)
    P_T = P.transpose(-1, -2).contiguous()
    P_inv = _invert_SE3(P).contiguous()

    rotary_full = torch.complex(full_cos, full_sin).reshape(1, 1, N, HEAD_DIM // 2)
    orig_t = HEAD_DIM // 2 - 2 * (HEAD_DIM // 6)
    orig_h = HEAD_DIM // 6
    new_half = HEAD_DIM // 2
    new_t = new_half // 2 - 2 * (new_half // 6)
    new_h = new_half // 6
    new_w = new_half // 6
    rotary_cam = torch.cat(
        [
            rotary_full[..., :new_t],
            rotary_full[..., orig_t: orig_t + new_h],
            rotary_full[..., orig_t + orig_h: orig_t + orig_h + new_w],
        ],
        dim=-1,
    )
    rope_cos, rope_sin = _prepare_ucpe_rope_tables(rotary_cam, N, HEAD_DIM // 2, x.device)

    k_scale = (HEAD_DIM ** -0.5) * (S ** -0.5)
    q_t, k_t, v_t, infl = _torch_cam_prep_reference(
        q_raw.reshape(B, N, HEADS, HEAD_DIM),
        k_raw.reshape(B, N, HEADS, HEAD_DIM),
        v_raw.reshape(B, N, HEADS, HEAD_DIM),
        q_norm_weight=p["q_norm"],
        k_norm_weight=p["k_norm"],
        proj_q=P_T,
        proj_kv=P_inv,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        k_scale=k_scale,
        norm_eps=EPS,
    )
    frame_infl = infl.view(B, HEADS, T, S).mean(dim=-1)
    beta_cam = beta / frame_infl.unsqueeze(-1).clamp_min(1.0)

    out = _torch_cam_scan_reference(q_t, k_t, v_t, beta_cam, decay, reverse=False)
    out = out + _torch_cam_scan_reference(q_t, k_t, v_t, beta_cam, decay, reverse=True)

    _, _, apply_fn_o = _prepare_ray_apply_fns(head_dim=HEAD_DIM, P=P, P_T=P_T, P_inv=P_inv, rotary_emb=rotary_cam)
    out = apply_fn_o(out.transpose(-1, -2)).transpose(-1, -2).contiguous()
    out = out.reshape(B, C, N).permute(0, 2, 1)
    cam_raw = F.linear(out, p["o_w"], p["o_b"])
    return cam_raw, P, infl, frame_infl


def main():
    gen = torch.Generator().manual_seed(20261109)
    x = torch.randn(B, N, C, generator=gen).float()
    cc = camera_conditions(gen)
    full_cos, full_sin = main_rope_tables()
    beta = torch.sigmoid(torch.randn(B, HEADS, T, S, generator=gen)).float()
    decay = (0.7 + 0.29 * torch.rand(B, HEADS, T, generator=gen)).float()
    q_w, q_b = linear_params(gen, C, 1.0)
    k_w, k_b = linear_params(gen, C, 1.0)
    v_w, v_b = linear_params(gen, C, 1.0)
    o_w, o_b = linear_params(gen, C, 1.0)
    conv_w = (0.5 * torch.randn(C, 1, 4, generator=gen)).float()
    q_norm = (1.0 + 0.1 * torch.randn(C, generator=gen)).float()
    k_norm = (1.0 + 0.1 * torch.randn(C, generator=gen)).float()
    p = {"q_w": q_w, "q_b": q_b, "k_w": k_w, "k_b": k_b, "v_w": v_w, "v_b": v_b, "o_w": o_w, "o_b": o_b,
         "conv_w": conv_w, "q_norm": q_norm, "k_norm": k_norm}

    with torch.no_grad():
        cam_raw, P, infl, frame_infl = upstream_cam_branch(x, cc, full_cos, full_sin, beta, decay, p)
        P_flat = P.reshape(B, N, 16).float().contiguous()

    print("cam_raw absmax", float(cam_raw.abs().max()), "std", float(cam_raw.std()))
    print("frame mean inflation", frame_infl.reshape(-1).tolist())
    print("P translation absmax", float(P_flat[..., 3].abs().max()), float(P_flat[..., 7].abs().max()))

    tensors = {
        "x": x.contiguous(),
        "camera_conditions": cc.contiguous(),
        "rope_cos": full_cos.contiguous(),
        "rope_sin": full_sin.contiguous(),
        "beta": beta.contiguous(),
        "decay": decay.contiguous(),
        "q_proj_cam.weight": q_w.contiguous(),
        "q_proj_cam.bias": q_b.contiguous(),
        "k_proj_cam.weight": k_w.contiguous(),
        "k_proj_cam.bias": k_b.contiguous(),
        "v_proj_cam.weight": v_w.contiguous(),
        "v_proj_cam.bias": v_b.contiguous(),
        "conv_k_cam.weight": conv_w.contiguous(),
        "q_norm_cam.weight": q_norm.contiguous(),
        "k_norm_cam.weight": k_norm.contiguous(),
        "out_proj_cam.weight": o_w.contiguous(),
        "out_proj_cam.bias": o_b.contiguous(),
        "ray_p": P_flat,
        "expected_cam_raw": cam_raw.reshape(B, N, C).contiguous(),
    }
    config = {
        "source": "Sana src diffusion/model/ops/fused_cam_gdn.py (_torch_cam_prep_reference, _torch_cam_scan_reference), "
                  "diffusion/model/nets/sana_camctrl_blocks.py (_process_camera_conditions_raymats_only, "
                  "_prepare_ray_apply_fns), sana_gdn_blocks.py (_bidirectional_causal_conv_1d)",
        "frames": T, "latent_height": H_LAT, "latent_width": W_LAT, "heads": HEADS, "head_dim": HEAD_DIM,
        "rope_theta": THETA, "norm_eps": EPS, "patch_size": list(PATCH), "dtype": "float32",
    }
    save_file(tensors, os.path.join(OUT_DIR, "sana_wm_s2_cam_branch.safetensors"),
              metadata={"config": json.dumps(config)})


main()
