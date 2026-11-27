#!/usr/bin/env python3
"""Generates the sana_wm_s1_* fixtures for SanaWmGdnBidi (CPU only, float64 reference stored as float32).

The reference is the upstream bidirectional GDN main branch transcribed twice: upstream column form (the
torch_recurrent_sana_gdn recurrence) and the row form the C# class implements. The upstream torch functions are also
run directly as a cross-check. The Triton kernel is not run: it cannot execute on CPU.

Run with the pinned upstream venv:
  CUDA_VISIBLE_DEVICES= SANA_WM_UPSTREAM_SRC=<upstream Sana src> <venv>/bin/python generate_sana_wm_s1_fixtures.py
"""
import math
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

UPSTREAM_SRC = os.environ.get("SANA_WM_UPSTREAM_SRC", "")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PREFIX = "sana_wm_s1_"

# Tiny config: S = H_lat * W_lat = 6 tokens per frame, N = T * S = 24 tokens.
FRAMES, LAT_H, LAT_W = 4, 2, 3
HEADS, HEAD_DIM = 2, 8
BATCH = 2
C = HEADS * HEAD_DIM
S = LAT_H * LAT_W
N = FRAMES * S
K_SCALE = (HEAD_DIM ** -0.5) * (S ** -0.5)
EPS_RMS = 1e-5
EPS_DEN = 1e-8
DT = torch.float64


def rms_norm(v, weight, eps=EPS_RMS):
    return v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps) * weight


def causal_conv_depthwise(x_btc, w_c4):
    """y[t] = sum_j w[j] x[t-3+j] with zero history. x_btc: [M, T, C]; w_c4: [C, 4]."""
    t = x_btc.shape[1]
    y = F.conv1d(x_btc.transpose(1, 2), w_c4.unsqueeze(1), padding=3, groups=x_btc.shape[2])
    return y[..., :t].transpose(1, 2)


def conv_literal(x_btc, w_c4):
    """Plan form: conv(x) + conv(flip x).flip - w[3] * x."""
    fwd = causal_conv_depthwise(x_btc, w_c4)
    bwd = causal_conv_depthwise(x_btc.flip(1), w_c4).flip(1)
    return fwd + bwd - w_c4[:, 3] * x_btc


def conv_seven_tap(x_btc, w_c4):
    """Equivalent symmetric kernel [w0 w1 w2 w3 w2 w1 w0] at offsets -3..3 with zero padding."""
    t = x_btc.shape[1]
    pad = F.pad(x_btc, (0, 0, 3, 3))
    taps = [w_c4[:, 0], w_c4[:, 1], w_c4[:, 2], w_c4[:, 3], w_c4[:, 2], w_c4[:, 1], w_c4[:, 0]]
    out = torch.zeros_like(x_btc)
    for i, w in enumerate(taps):
        out = out + pad[:, i:i + t] * w
    return out


def flip_and_shift(x, dim, shift_val):
    x_flip = torch.flip(x, dims=[dim])
    x_shifted = x_flip.narrow(dim, 0, x.shape[dim] - 1)
    pad_shape = list(x.shape)
    pad_shape[dim] = 1
    padding = torch.full(pad_shape, shift_val, dtype=x.dtype)
    return torch.cat([padding, x_shifted], dim=dim)


def scan_column_form(q, k, v, q_rot, k_rot, beta, decay):
    """Upstream torch_recurrent_sana_gdn in column form. Time-structured inputs: [B, H, T, D, S] (q, k, v, rotated),
    beta [B, H, T, S], decay [B, H, T]. Returns num [B, H, T, D, S] and den [B, H, T, 1, S] (processing order)."""
    b, h, frames, d, s = q.shape
    state_kv = torch.zeros(b, h, d, d, dtype=DT)
    state_z = torch.zeros(b, h, d, 1, dtype=DT)
    num_list, den_list = [], []
    for t in range(frames):
        q_t, k_t, v_t = q[:, :, t], k[:, :, t], v[:, :, t]
        qr_t, kr_t = q_rot[:, :, t], k_rot[:, :, t]
        beta_t = beta[:, :, t].unsqueeze(-2)
        dec_t = decay[:, :, t].view(b, h, 1, 1)
        state_kv = state_kv * dec_t
        state_z = state_z * dec_t
        v_pred = state_kv @ kr_t
        delta_v = (v_t - v_pred) * beta_t
        state_kv = state_kv + delta_v @ kr_t.transpose(-1, -2)
        z_pred = state_z.transpose(-1, -2) @ k_t
        delta_z = (1.0 - z_pred) * beta_t
        state_z = state_z + k_t @ delta_z.transpose(-1, -2)
        num_list.append(state_kv @ qr_t)
        den_list.append(state_z.transpose(-1, -2) @ q_t)
    return torch.stack(num_list, dim=2), torch.stack(den_list, dim=2)


def scan_row_form(kr_rows, k_rows, v_rows, qr_rows, q_rows, beta_rows, decay_rows):
    """Plan row form for one (batch, head): rows are tokens. kr/k/v/qr/q: [T, S, D]; beta: [T, S]; decay: [T].
    A = state_kv^T (key x value), z = state_z. Returns num [T, S, D] and den [T, S]."""
    frames, s, d = kr_rows.shape
    a_state = torch.zeros(d, d, dtype=DT)
    z_state = torch.zeros(d, dtype=DT)
    nums, dens = [], []
    for t in range(frames):
        a_bar = decay_rows[t] * a_state
        z_bar = decay_rows[t] * z_state
        beta_t = beta_rows[t]
        pred = kr_rows[t] @ a_bar
        om = v_rows[t] - pred
        kb = (kr_rows[t] * beta_t.unsqueeze(-1)).transpose(0, 1)
        a_state = a_bar + kb @ om
        zp = k_rows[t] @ z_bar
        om_z = 1.0 - zp
        kbz = (k_rows[t] * beta_t.unsqueeze(-1)).transpose(0, 1)
        z_state = z_bar + kbz @ om_z
        nums.append(qr_rows[t] @ a_state)
        dens.append(q_rows[t] @ z_state)
    return torch.stack(nums), torch.stack(dens)


def gdn_plan_form(x, w, cfg_rope):
    """Main branch from the plan equations. Returns [B, N, C] before the output gate and proj."""
    cos, sin = cfg_rope
    b = x.shape[0]
    qkv = x @ w["qkv"].t()
    q_raw, k_raw, v_raw = qkv[..., :C], qkv[..., C:2 * C], qkv[..., 2 * C:]

    k_tc = k_raw.reshape(b, FRAMES, S, C).permute(0, 2, 1, 3).reshape(b * S, FRAMES, C)
    k_conv = conv_literal(k_tc, w["conv"]).reshape(b, S, FRAMES, C).permute(0, 2, 1, 3).reshape(b, N, C)

    beta = torch.sigmoid(x @ w["beta_w"].t() + w["beta_b"])  # [B, N, H]
    x_frame = x.reshape(b, FRAMES, S, C).mean(dim=2)
    a_out = x_frame @ w["gate_w"].t() + w["gate_b"]  # [B, T, H]
    a_val = torch.exp(w["a_log"])
    decay = torch.exp(-a_val * F.softplus(a_out + w["dt_bias"]))  # [B, T, H]

    q_n = F.relu(rms_norm(q_raw, w["q_norm"]))
    k_n = F.relu(rms_norm(k_conv, w["k_norm"])) * K_SCALE

    def heads_tshd(v):  # [B, N, C] -> [B, H, T, S, D]
        return v.reshape(b, FRAMES, S, HEADS, HEAD_DIM).permute(0, 3, 1, 2, 4)

    def rope_complex(v_hts):  # v [B, H, T, S, D] -> rotated; complex pairs (2i, 2i+1)
        v_flat = v_hts.reshape(b, HEADS, N, HEAD_DIM)
        pairs = torch.view_as_complex(v_flat.reshape(b, HEADS, N, HEAD_DIM // 2, 2).contiguous())
        freqs = torch.complex(cos, sin)  # [N, D/2]
        out = torch.view_as_real(pairs * freqs).reshape(b, HEADS, N, HEAD_DIM)
        return out.reshape(b, HEADS, FRAMES, S, HEAD_DIM)

    q_h = heads_tshd(q_n)
    k_h = heads_tshd(k_n)
    v_h = heads_tshd(v_raw)
    q_rot = rope_complex(q_h)
    k_rot = rope_complex(k_h)

    beta_h = beta.reshape(b, FRAMES, S, HEADS).permute(0, 3, 1, 2)  # [B, H, T, S]
    decay_h = decay.permute(0, 2, 1)  # [B, H, T]

    def to_cols(v):  # [B, H, T, S, D] -> [B, H, T, D, S]
        return v.transpose(-1, -2)

    q_c, k_c, v_c = to_cols(q_h), to_cols(k_h), to_cols(v_h)
    qr_c, kr_c = to_cols(q_rot), to_cols(k_rot)

    num_f, den_f = scan_column_form(q_c, k_c, v_c, qr_c, kr_c, beta_h, decay_h)

    q_b = torch.flip(qr_c, dims=[2])
    qq_b = torch.flip(q_c, dims=[2])
    k_b = flip_and_shift(k_c, 2, 0.0)
    v_b = flip_and_shift(v_c, 2, 0.0)
    kr_b = flip_and_shift(kr_c, 2, 0.0)
    beta_b = flip_and_shift(beta_h, 2, 0.0)
    decay_b = flip_and_shift(decay_h, 2, 1.0)
    num_bf, den_bf = scan_column_form(qq_b, k_b, v_b, q_b, kr_b, beta_b, decay_b)
    num_b = torch.flip(num_bf, dims=[2])
    den_b = torch.flip(den_bf, dims=[2])

    # num_*: [B, H, T, D, S] -> [B, H, D, N] token order (T-major, then S)
    def to_bhdn(v):
        return v.permute(0, 1, 3, 2, 4).reshape(b, HEADS, v.shape[3], N)

    out = (to_bhdn(num_f) + to_bhdn(num_b)) / (to_bhdn(den_f) + to_bhdn(den_b) + EPS_DEN)
    return out.permute(0, 3, 1, 2).reshape(b, N, C)


def gdn_plan_row_form(x, w, cfg_rope):
    """Same main branch with the row-form scan the C# class implements (per batch and head)."""
    cos, sin = cfg_rope
    b = x.shape[0]
    qkv = x @ w["qkv"].t()
    q_raw, k_raw, v_raw = qkv[..., :C], qkv[..., C:2 * C], qkv[..., 2 * C:]
    k_tc = k_raw.reshape(b, FRAMES, S, C).permute(0, 2, 1, 3).reshape(b * S, FRAMES, C)
    k_conv = conv_literal(k_tc, w["conv"]).reshape(b, S, FRAMES, C).permute(0, 2, 1, 3).reshape(b, N, C)
    beta = torch.sigmoid(x @ w["beta_w"].t() + w["beta_b"])
    x_frame = x.reshape(b, FRAMES, S, C).mean(dim=2)
    a_out = x_frame @ w["gate_w"].t() + w["gate_b"]
    decay = torch.exp(-torch.exp(w["a_log"]) * F.softplus(a_out + w["dt_bias"]))
    q_n = F.relu(rms_norm(q_raw, w["q_norm"]))
    k_n = F.relu(rms_norm(k_conv, w["k_norm"])) * K_SCALE

    def tokens_hdk(v):  # [B, N, C] -> [B, H, N, D]
        return v.reshape(b, N, HEADS, HEAD_DIM).permute(0, 2, 1, 3)

    def rope_rows(v_bhnd):
        pairs = torch.view_as_complex(v_bhnd.reshape(b, HEADS, N, HEAD_DIM // 2, 2).contiguous())
        out = torch.view_as_real(pairs * torch.complex(cos, sin)).reshape(b, HEADS, N, HEAD_DIM)
        return out

    q_h, k_h, v_h = tokens_hdk(q_n), tokens_hdk(k_n), tokens_hdk(v_raw)
    qr_h, kr_h = rope_rows(q_h), rope_rows(k_h)
    beta_h = beta.reshape(b, FRAMES, S, HEADS).permute(0, 3, 1, 2)
    decay_h = decay.permute(0, 2, 1)

    def tfirst(v):  # [B, H, N, D] -> [B, H, T, S, D]
        return v.reshape(b, HEADS, FRAMES, S, HEAD_DIM)

    q_t, k_t, v_t = tfirst(q_h), tfirst(k_h), tfirst(v_h)
    qr_t, kr_t = tfirst(qr_h), tfirst(kr_h)
    num_f = torch.zeros(b, HEADS, FRAMES, S, HEAD_DIM, dtype=DT)
    den_f = torch.zeros(b, HEADS, FRAMES, S, dtype=DT)
    num_b = torch.zeros_like(num_f)
    den_b = torch.zeros_like(den_f)
    for bi in range(b):
        for hi in range(HEADS):
            n_f, d_f = scan_row_form(kr_t[bi, hi], k_t[bi, hi], v_t[bi, hi], qr_t[bi, hi], q_t[bi, hi],
                                     beta_h[bi, hi], decay_h[bi, hi])
            num_f[bi, hi], den_f[bi, hi] = n_f, d_f
            # Reverse: frame f receives frames f+1..T-1 (exclusive). Shifted inputs and flipped queries.
            kr_s = flip_and_shift(kr_t[bi, hi], 0, 0.0)
            k_s = flip_and_shift(k_t[bi, hi], 0, 0.0)
            v_s = flip_and_shift(v_t[bi, hi], 0, 0.0)
            beta_s = flip_and_shift(beta_h[bi, hi], 0, 0.0)
            decay_s = flip_and_shift(decay_h[bi, hi], 0, 1.0)
            n_b, d_b = scan_row_form(kr_s, k_s, v_s, torch.flip(qr_t[bi, hi], [0]), torch.flip(q_t[bi, hi], [0]),
                                     beta_s, decay_s)
            num_b[bi, hi] = torch.flip(n_b, [0])
            den_b[bi, hi] = torch.flip(d_b, [0])
    num = num_f + num_b
    den = den_f + den_b + EPS_DEN
    out = num / den.unsqueeze(-1)  # [B, H, T, S, D]
    return out.permute(0, 2, 3, 1, 4).reshape(b, N, C)


def upstream_torch_path(x, w, cfg_rope):
    """Runs the upstream torch functions on CPU: BidirectionalGDN conv combination, the module-level gate and RoPE
    helpers, sana_v2v torch_recurrent_sana_gdn and its flip/combine. Upstream RMSNorm computes in float32."""
    sys.path.insert(0, UPSTREAM_SRC)
    from diffusion.model.nets import sana_gdn_blocks as gdn_mod  # noqa: E402
    from diffusion.model.nets import sana_v2v_attn_blocks as v2v  # noqa: E402
    from diffusion.model.norms import RMSNorm  # noqa: E402

    cos, sin = cfg_rope
    b = x.shape[0]
    freqs = torch.complex(cos, sin)  # [N, D/2]
    qkv = F.linear(x, w["qkv"]).reshape(b, N, 3, HEADS, HEAD_DIM)
    q_raw = qkv[:, :, 0].reshape(b, N, C)
    k_raw = qkv[:, :, 1].reshape(b, N, C)
    v_raw = qkv[:, :, 2].reshape(b, N, C)

    class StubCausalConv:
        def __init__(self, weight):
            self.weight = weight

        def __call__(self, xin):
            y = causal_conv_depthwise(xin, self.weight[:, 0, :])
            return y, None

    conv = StubCausalConv(w["conv"].unsqueeze(1))
    k_tc = k_raw.reshape(b, FRAMES, S, C).permute(0, 2, 1, 3).reshape(b * S, FRAMES, C)
    k_conv_tc = gdn_mod.BidirectionalGDN._bidirectional_causal_conv_1d(k_tc, conv)
    k_conv = k_conv_tc.reshape(b, S, FRAMES, C).permute(0, 2, 1, 3).reshape(b, N, C)

    beta_bhts, decay = gdn_mod._compute_frame_gates(
        x, FRAMES, S, HEADS, w["beta_w"], w["beta_b"], w["gate_w"], w["gate_b"], w["dt_bias"], w["a_log"])
    beta = beta_bhts.permute(0, 2, 3, 1)  # torch_recurrent_sana_gdn takes [B, T, S, H]

    q_norm = RMSNorm(C, scale_factor=1.0, eps=EPS_RMS).to(DT)
    k_norm = RMSNorm(C, scale_factor=1.0, eps=EPS_RMS).to(DT)
    q_norm.weight.data = w["q_norm"].clone()
    k_norm.weight.data = w["k_norm"].clone()

    q_n = F.relu(q_norm(q_raw.to(DT)))
    k_n = F.relu(k_norm(k_conv.to(DT))) * K_SCALE
    q_cols = q_n.transpose(-1, -2).reshape(b, HEADS, HEAD_DIM, N)
    k_cols = k_n.transpose(-1, -2).reshape(b, HEADS, HEAD_DIM, N)
    v_cols = v_raw.transpose(-1, -2).reshape(b, HEADS, HEAD_DIM, N)

    q_rot = gdn_mod._apply_rotary_emb(q_cols, freqs)
    k_rot = gdn_mod._apply_rotary_emb(k_cols, freqs)

    num_f, den_f = v2v.torch_recurrent_sana_gdn(
        q_cols, k_cols, v_cols, q_rot, k_rot, beta, decay, recall_gate=1, eps=EPS_DEN, return_components=True)

    def to_time(t):
        return t.view(b, HEADS, HEAD_DIM, FRAMES, S).permute(0, 1, 3, 2, 4)

    def from_time(t):
        return t.permute(0, 1, 3, 2, 4).reshape(b, HEADS, HEAD_DIM, N)

    q_t, k_t, v_t = to_time(q_cols), to_time(k_cols), to_time(v_cols)
    qr_t, kr_t = to_time(q_rot), to_time(k_rot)
    q_bwd = torch.flip(q_t, dims=[2])
    qr_bwd = torch.flip(qr_t, dims=[2])
    k_bwd = v2v.flip_and_shift(k_t, dim=2, shift_val=0.0)
    v_bwd = v2v.flip_and_shift(v_t, dim=2, shift_val=0.0)
    kr_bwd = v2v.flip_and_shift(kr_t, dim=2, shift_val=0.0)
    beta_bwd = v2v.flip_and_shift(beta, dim=1, shift_val=0.0)
    decay_bwd = v2v.flip_and_shift(decay, dim=2, shift_val=1.0)
    num_bf, den_bf = v2v.torch_recurrent_sana_gdn(
        from_time(q_bwd), from_time(k_bwd), from_time(v_bwd), from_time(qr_bwd), from_time(kr_bwd),
        beta_bwd, decay_bwd, recall_gate=1, eps=EPS_DEN, return_components=True)

    def flip_back(t):
        actual = t.shape[2]
        return torch.flip(t.view(b, HEADS, actual, FRAMES, S), dims=[3]).reshape(b, HEADS, actual, N)

    out = (num_f + flip_back(num_bf)) / (den_f + flip_back(den_bf) + EPS_DEN)
    return out.permute(0, 3, 1, 2).reshape(b, N, C)


def make_weights(gen):
    def randn(*shape, std):
        return torch.randn(*shape, generator=gen, dtype=DT) * std

    return {
        "qkv": randn(3 * C, C, std=1.0 / math.sqrt(C)),
        "conv": randn(C, 4, std=0.5),
        "beta_w": randn(HEADS, C, std=1.0 / math.sqrt(C)),
        "beta_b": randn(HEADS, std=0.5),
        "gate_w": randn(HEADS, C, std=1.0 / math.sqrt(C)),
        "gate_b": randn(HEADS, std=0.5),
        "a_log": torch.log(torch.rand(HEADS, generator=gen, dtype=DT) * 3.5 + 0.5),
        "dt_bias": randn(HEADS, std=1.0),
        "q_norm": 1.0 + 0.1 * randn(C, std=1.0),
        "k_norm": 1.0 + 0.1 * randn(C, std=1.0),
    }


def make_rope(gen):
    """Interleaved-pair RoPE table [N, D]: cos/sin duplicated across each pair, one angle per pair."""
    pos = torch.arange(N, dtype=DT)
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, HEAD_DIM, 2, dtype=DT) / HEAD_DIM))
    jitter = 1.0 + 0.05 * torch.randn(N, HEAD_DIM // 2, generator=gen, dtype=DT)
    angles = pos.unsqueeze(1) * inv_freq.unsqueeze(0) * jitter  # [N, D/2]
    cos_pairs, sin_pairs = torch.cos(angles), torch.sin(angles)
    cos = cos_pairs.repeat_interleave(2, dim=1)
    sin = sin_pairs.repeat_interleave(2, dim=1)
    return cos, sin, cos_pairs, sin_pairs


def configure(frames, lat_h, lat_w):
    """The reference functions read these module globals; set them before each case."""
    global FRAMES, LAT_H, LAT_W, S, N, K_SCALE
    FRAMES, LAT_H, LAT_W = frames, lat_h, lat_w
    S = lat_h * lat_w
    N = FRAMES * S
    K_SCALE = (HEAD_DIM ** -0.5) * (S ** -0.5)


def build_case(name, frames, lat_h, lat_w, seed, cross_check):
    configure(frames, lat_h, lat_w)
    gen = torch.Generator().manual_seed(seed)
    w = make_weights(gen)
    x = torch.randn(BATCH, N, C, generator=gen, dtype=DT)
    cos, sin, cos_pairs, sin_pairs = make_rope(gen)
    rope_pairs = (cos_pairs, sin_pairs)

    if cross_check:
        xk = torch.randn(3, FRAMES, C, generator=gen, dtype=DT)
        conv_err = (conv_literal(xk, w["conv"]) - conv_seven_tap(xk, w["conv"])).abs().max().item()
        print(f"[{name}] conv literal vs 7-tap max abs diff: {conv_err:.3e}")

    plan = gdn_plan_form(x, w, rope_pairs)
    row = gdn_plan_row_form(x, w, rope_pairs)
    print(f"[{name}] column form vs row form max abs diff: {(plan - row).abs().max().item():.3e}")
    if cross_check and UPSTREAM_SRC:
        try:
            up = upstream_torch_path(x, w, rope_pairs)
            print(f"[{name}] column form vs upstream torch path max abs diff: {(plan - up).abs().max().item():.3e}")
        except Exception as exc:  # noqa: BLE001 - report and keep the Python reference fixture
            print(f"[{name}] upstream torch path not run: {type(exc).__name__}: {exc}")

    tensors = {
        "x": x.float().contiguous(),
        "qkv_weight": w["qkv"].float().contiguous(),
        "conv_k_weight": w["conv"].unsqueeze(1).float().contiguous(),
        "beta_proj_weight": w["beta_w"].float().contiguous(),
        "beta_proj_bias": w["beta_b"].float().contiguous(),
        "gate_proj_weight": w["gate_w"].float().contiguous(),
        "gate_proj_bias": w["gate_b"].float().contiguous(),
        "a_log": w["a_log"].float().contiguous(),
        "dt_bias": w["dt_bias"].float().contiguous(),
        "q_norm_weight": w["q_norm"].float().contiguous(),
        "k_norm_weight": w["k_norm"].float().contiguous(),
        "rope_cos": cos.float().contiguous(),
        "rope_sin": sin.float().contiguous(),
        "expected": plan.float().contiguous(),
    }
    metadata = {
        "frames": str(FRAMES),
        "latent_height": str(LAT_H),
        "latent_width": str(LAT_W),
        "heads": str(HEADS),
        "head_dim": str(HEAD_DIM),
        "batch": str(BATCH),
        "reference": "float64 CPU; upstream column form transcription, stored float32",
    }
    path = os.path.join(OUT_DIR, PREFIX + name + ".safetensors")
    save_file(tensors, path, metadata=metadata)
    print(f"[{name}] wrote {path} ({os.path.getsize(path)} bytes), output mean abs {plan.abs().mean().item():.4f}")


def main():
    build_case("gdn_bidi_tiny", FRAMES, LAT_H, LAT_W, 20261109, cross_check=True)
    build_case("gdn_bidi_single_frame", 1, LAT_H, LAT_W, 20261110, cross_check=False)


if __name__ == "__main__":
    main()
