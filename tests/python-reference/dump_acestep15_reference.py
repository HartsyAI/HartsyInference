"""Standalone f32 torch reimplementation of ACE-Step v1.5 turbo DiT decoder + condition encoder.

Logic copied verbatim from acestep/models/turbo/modeling_acestep_v15_turbo.py (cloned upstream) +
the Qwen3 primitives it imports. Loads the Comfy-Org turbo safetensors (cast bf16->f32), runs fixed
inputs through the condition encoder and the DiT velocity head, and dumps IO for the C# parity test.

Used as the parity oracle because the sandbox transformers (4.27) lacks models.qwen3 and
vector_quantize_pytorch, so we cannot import the real module. Every formula here mirrors the source.
"""
import json, struct, math, sys
import numpy as np
import torch
import torch.nn.functional as F

torch.manual_seed(0)
DIT = "/home/kalebbroo/.cache/hartsyinference/models/Comfy-Org--ace_step_1.5_ComfyUI_files/split_files/diffusion_models/acestep_v1.5_turbo.safetensors"

# ---- config (configuration_acestep_v15.py defaults) ----
HID = 2048; NLAY = 24; NHEAD = 16; NKV = 8; HD = 128; INTER = 6144
EPS = 1e-6; THETA = 1_000_000.0; SLIDE = 128; INCH = 192; PATCH = 2; LATCH = 64
TEXTDIM = 1024; TIMBREDIM = 64; NLYRIC = 8; NTIMBRE = 4; FREQ = 256

def load_st(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = f.tell()
        out = {}
        DT = {'F32': torch.float32, 'F16': torch.float16, 'BF16': torch.bfloat16}
        for k, v in hdr.items():
            if k == '__metadata__': continue
            a, b = v['data_offsets']
            f.seek(base + a)
            raw = f.read(b - a)
            t = torch.frombuffer(bytearray(raw), dtype=DT[v['dtype']]).reshape(v['shape'])
            out[k] = t.float()  # cast every tensor to f32 (this is the reference, not the 3060)
    return out

W = load_st(DIT)
def g(k): return W[k]

# ---------------- Qwen3 primitives (verbatim) ----------------
def rms_norm(x, weight, eps=EPS):
    v = x.float()
    v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
    return (weight * v).type_as(x)

def rotary_tables(seq_len, head_dim=HD, theta=THETA):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    pos = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(pos, inv)              # [S, head_dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)    # [S, head_dim]  (Qwen3 default rope: not interleaved)
    return emb.cos(), emb.sin()

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(q, k, cos, sin):
    # q,k: [B, H, S, D]; cos,sin: [S, D]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q2 = (q * cos) + (rotate_half(q) * sin)
    k2 = (k * cos) + (rotate_half(k) * sin)
    return q2, k2

def repeat_kv(x, n):
    b, h, s, d = x.shape
    if n == 1: return x
    return x[:, :, None, :, :].expand(b, h, n, s, d).reshape(b, h * n, s, d)

def eager_attn(q, k, v, mask, scaling):
    # q:[B,Hq,S,D] k,v:[B,Hkv,L,D]
    groups = q.shape[1] // k.shape[1]
    k = repeat_kv(k, groups); v = repeat_kv(v, groups)
    w = torch.matmul(q, k.transpose(2, 3)) * scaling
    if mask is not None:
        w = w + mask[:, :, :, : k.shape[-2]]
    w = torch.softmax(w, dim=-1)
    o = torch.matmul(w, v)              # [B,Hq,S,D]
    return o.transpose(1, 2).contiguous()  # [B,S,Hq,D]

def create_4d_mask(seq_len, sliding=None, is_sliding=False):
    idx = torch.arange(seq_len)
    diff = idx.unsqueeze(1) - idx.unsqueeze(0)
    valid = torch.ones((seq_len, seq_len), dtype=torch.bool)
    if is_sliding and sliding is not None:
        valid = valid & (torch.abs(diff) <= sliding)
    valid = valid.unsqueeze(0).unsqueeze(0)
    m = torch.full(valid.shape, torch.finfo(torch.float32).min, dtype=torch.float32)
    m.masked_fill_(valid, 0.0)
    return m

def layer_type(i):  # config builds: "sliding_attention" if (i+1)%2 else "full_attention"
    return "sliding" if (i + 1) % 2 == 1 else "full"

def attention(prefix, x, kv_src, cos, sin, mask, use_rope):
    qb = (x @ g(prefix + '.q_proj.weight').t())
    kb = (kv_src @ g(prefix + '.k_proj.weight').t())
    vb = (kv_src @ g(prefix + '.v_proj.weight').t())
    B, S, _ = x.shape; L = kv_src.shape[1]
    q = qb.view(B, S, NHEAD, HD); k = kb.view(B, L, NKV, HD); v = vb.view(B, L, NKV, HD)
    q = rms_norm(q, g(prefix + '.q_norm.weight'))
    k = rms_norm(k, g(prefix + '.k_norm.weight'))
    q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)  # [B,H,S,D]
    if use_rope:
        q, k = apply_rope(q, k, cos, sin)
    o = eager_attn(q, k, v, mask, HD ** -0.5)
    o = o.reshape(B, S, -1)
    return o @ g(prefix + '.o_proj.weight').t()

def mlp(prefix, x):
    gate = F.silu(x @ g(prefix + '.gate_proj.weight').t())
    up = x @ g(prefix + '.up_proj.weight').t()
    return (gate * up) @ g(prefix + '.down_proj.weight').t()

# ---------------- encoder layer (bidirectional, pre-norm) ----------------
def encoder_layer(prefix, x, cos, sin, mask):
    res = x
    h = rms_norm(x, g(prefix + '.input_layernorm.weight'))
    h = attention(prefix + '.self_attn', h, h, cos, sin, mask, use_rope=True)
    x = res + h
    res = x
    h = rms_norm(x, g(prefix + '.post_attention_layernorm.weight'))
    h = mlp(prefix + '.mlp', h)
    return res + h

def run_encoder(prefix, x, nlayers, norm_key):
    S = x.shape[1]
    cos, sin = rotary_tables(S)
    full = create_4d_mask(S, None, False)
    slide = create_4d_mask(S, SLIDE, True)
    for i in range(nlayers):
        m = slide if layer_type(i) == "sliding" else full
        x = encoder_layer(f'{prefix}.layers.{i}', x, cos, sin, m)
    return rms_norm(x, g(norm_key))

# ---------------- condition encoder ----------------
def condition_encode(text_hidden, lyric_hidden, timbre_latent=None):
    # text_projector (no bias)
    text = text_hidden @ g('encoder.text_projector.weight').t()
    # lyric encoder: Linear(1024->2048) + 8 blocks + norm
    le = lyric_hidden @ g('encoder.lyric_encoder.embed_tokens.weight').t() + g('encoder.lyric_encoder.embed_tokens.bias')
    lyric = run_encoder('encoder.lyric_encoder', le, NLYRIC, 'encoder.lyric_encoder.norm.weight')
    parts = [lyric]
    if timbre_latent is not None:
        # NOTE: reference timbre_encoder.forward has the special_token cat COMMENTED OUT.
        te = timbre_latent @ g('encoder.timbre_encoder.embed_tokens.weight').t() + g('encoder.timbre_encoder.embed_tokens.bias')
        te = run_encoder('encoder.timbre_encoder', te, NTIMBRE, 'encoder.timbre_encoder.norm.weight')
        parts.append(te[:, :1, :])  # cls = first position
    parts.append(text)
    # pack order: pack(lyric, timbre) then pack(_, text)  -> with no padding == concat [lyric, timbre, text]
    return torch.cat(parts, dim=1)

# ---------------- DiT ----------------
def timestep_embedding(t, dim=FREQ, max_period=10000, scale=1000.0):
    t = t * scale
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

def time_embed(t, prefix):
    tf = timestep_embedding(t)
    temb = tf @ g(prefix + '.linear_1.weight').t() + g(prefix + '.linear_1.bias')
    temb = F.silu(temb)
    temb = temb @ g(prefix + '.linear_2.weight').t() + g(prefix + '.linear_2.bias')
    proj = F.silu(temb) @ g(prefix + '.time_proj.weight').t() + g(prefix + '.time_proj.bias')
    proj = proj.view(temb.shape[0], 6, -1)
    return temb, proj

def dit_layer(i, x, temb_proj, cos, sin, mask, enc):
    p = f'decoder.layers.{i}'
    table = g(p + '.scale_shift_table')  # [1,6,2048]
    sh = (table + temb_proj).chunk(6, dim=1)
    shift_msa, scale_msa, gate_msa, c_shift, c_scale, c_gate = [s.squeeze(1) for s in sh]  # squeeze the 6-dim
    # self-attn AdaLN
    n = rms_norm(x, g(p + '.self_attn_norm.weight')) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
    a = attention(p + '.self_attn', n, n, cos, sin, mask, use_rope=True)
    x = x + a * gate_msa.unsqueeze(1)
    # cross-attn (no rope, no mask for batch-1 no-pad)
    n = rms_norm(x, g(p + '.cross_attn_norm.weight'))
    a = attention(p + '.cross_attn', n, enc, None, None, None, use_rope=False)
    x = x + a
    # mlp AdaLN
    n = rms_norm(x, g(p + '.mlp_norm.weight')) * (1 + c_scale.unsqueeze(1)) + c_shift.unsqueeze(1)
    f = mlp(p + '.mlp', n)
    x = x + f * c_gate.unsqueeze(1)
    return x

def dit_velocity(noisy, context, conditions, t_val, r_val):
    # noisy [B,T,64], context [B,T,128], conditions [B,L,2048]
    B, T, _ = noisy.shape
    t = torch.tensor([t_val], dtype=torch.float32)
    r = torch.tensor([r_val], dtype=torch.float32)
    temb_t, proj_t = time_embed(t, 'decoder.time_embed')
    temb_r, proj_r = time_embed(t - r, 'decoder.time_embed_r')
    temb = temb_t + temb_r
    proj = proj_t + proj_r  # [B,6,2048]
    # cat context + noisy, patchify Conv1d
    h = torch.cat([context, noisy], dim=-1)  # [B,T,192]
    hc = h.transpose(1, 2)  # [B,192,T]
    w = g('decoder.proj_in.1.weight'); b = g('decoder.proj_in.1.bias')
    hc = F.conv1d(hc, w, b, stride=PATCH)
    x = hc.transpose(1, 2)  # [B,S,2048]
    S = x.shape[1]
    enc = conditions @ g('decoder.condition_embedder.weight').t() + g('decoder.condition_embedder.bias')
    cos, sin = rotary_tables(S)
    full = create_4d_mask(S, None, False)
    slide = create_4d_mask(S, SLIDE, True)
    for i in range(NLAY):
        m = slide if layer_type(i) == "sliding" else full
        x = dit_layer(i, x, proj, cos, sin, m, enc)
    # final
    table = g('decoder.scale_shift_table')  # [1,2,2048]
    sh = (table + temb.unsqueeze(1)).chunk(2, dim=1)
    shift, scale = sh[0].squeeze(1), sh[1].squeeze(1)
    x = rms_norm(x, g('decoder.norm_out.weight')) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    xc = x.transpose(1, 2)
    wo = g('decoder.proj_out.1.weight'); bo = g('decoder.proj_out.1.bias')
    xc = F.conv_transpose1d(xc, wo, bo, stride=PATCH)
    return xc.transpose(1, 2)  # [B,T,64]

# ---------------- fixed inputs + dump ----------------
if __name__ == '__main__':
    rng = np.random.RandomState(1234)
    T_text = 7; T_lyric = 11; T = 24  # latent frames (even)
    text_hidden = torch.tensor(rng.randn(1, T_text, TEXTDIM).astype(np.float32))
    lyric_hidden = torch.tensor(rng.randn(1, T_lyric, TEXTDIM).astype(np.float32))
    cond = condition_encode(text_hidden, lyric_hidden, None)
    print('cond shape', cond.shape, 'maxabs', cond.abs().max().item())

    noisy = torch.tensor(rng.randn(1, T, LATCH).astype(np.float32))
    src = torch.tensor(rng.randn(1, T, LATCH).astype(np.float32))
    chunk = torch.ones(1, T, LATCH)
    context = torch.cat([src, chunk], dim=-1)
    t_val = 0.75; r_val = 0.75
    v = dit_velocity(noisy, context, cond, t_val, r_val)
    print('velocity shape', v.shape, 'maxabs', v.abs().max().item(), 'mean', v.mean().item())

    np.savez('/tmp/ace15ref/ref.npz',
             text_hidden=text_hidden.numpy(), lyric_hidden=lyric_hidden.numpy(),
             cond=cond.numpy(),
             noisy=noisy.numpy(), src=src.numpy(), chunk=chunk.numpy(),
             t_val=np.float32(t_val), r_val=np.float32(r_val),
             velocity=v.numpy())
    print('wrote /tmp/ace15ref/ref.npz')

def full_loop_dump():
    """8-step turbo shift-3 Euler with fixed noise; dump per-step + final latent for the C# scheduler check."""
    rng = np.random.RandomState(99)
    T_text = 5; T_lyric = 9; T = 20
    text_hidden = torch.tensor(rng.randn(1, T_text, TEXTDIM).astype(np.float32))
    lyric_hidden = torch.tensor(rng.randn(1, T_lyric, TEXTDIM).astype(np.float32))
    cond = condition_encode(text_hidden, lyric_hidden, None)
    src = torch.tensor(rng.randn(1, T, LATCH).astype(np.float32))
    chunk = torch.ones(1, T, LATCH)
    context = torch.cat([src, chunk], dim=-1)
    noise = torch.tensor(rng.randn(1, T, LATCH).astype(np.float32))
    sched = [1.0, 0.9545454545454546, 0.9, 0.8333333333333334, 0.75, 0.6428571428571429, 0.5, 0.3]
    xt = noise.clone()
    for i in range(len(sched)):
        tc = sched[i]
        vt = dit_velocity(xt, context, cond, tc, tc)
        if i == len(sched) - 1:
            xt = xt - vt * tc
        else:
            dt = tc - sched[i + 1]
            xt = xt - vt * dt
    print('full-loop final latent maxabs', xt.abs().max().item())
    np.savez('/tmp/ace15ref/loop.npz',
             text_hidden=text_hidden.numpy(), lyric_hidden=lyric_hidden.numpy(),
             src=src.numpy(), chunk=chunk.numpy(), noise=noise.numpy(),
             final=xt.numpy())
    import struct
    def w(name, arr):
        arr = np.ascontiguousarray(arr.astype(np.float32))
        with open(f'/tmp/ace15ref/loop_{name}.bin', 'wb') as f:
            f.write(struct.pack('<i', arr.ndim))
            for s in arr.shape: f.write(struct.pack('<i', s))
            f.write(arr.tobytes())
    for k in ['src', 'chunk', 'noise', 'final', 'text_hidden', 'lyric_hidden']:
        w(k, np.load('/tmp/ace15ref/loop.npz')[k])
    print('wrote loop_*.bin')

full_loop_dump()
