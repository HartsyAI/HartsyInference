"""
ACE-Step Music-DCAE decoder — Python step-by-step parity reference.

Independent NumPy float64 re-implementation of `MusicDcaeDecoder.Decode` (latent [1,8,H,W] -> mel [1,2,H*8,W*8]),
mirroring src/HartsyInference.Diffusion/Models/Music/{MusicDcaeDecoder,ResBlock2d,EfficientVitBlock}.cs. The C# diff
test (AceStepDcaeDiffTests) GENERATES the tiny synthetic checkpoint + fixed latent and runs the C# decode with
ACE_STEP_DEBUG_DIR; this script loads the SAME weights+input, runs this reference, and dumps the same tap points.
diff_ace_step_dcae.py compares. Validates: conv_in + repeat-interleave channel shortcut, ResBlock2d (3x3 conv -> SiLU
-> 3x3 -> channel-RMSNorm -> residual), the deepest-stage EfficientViT block (ReLU-linear attention + GLUMBConv), the
nearest-upsample + pixel-shuffle shortcut, and the final RMSNorm -> ReLU -> conv_out.

Run AFTER the C# test (which writes Output/ace_step_dcae_parity/):
  python3 tests/python-reference/dump_ace_step_dcae.py
"""
import json
import os
import numpy as np
from safetensors.numpy import load_file

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.join(REPO, "Output", "ace_step_dcae_parity")
INP = os.path.join(ROOT, "inputs")
REF = os.path.join(ROOT, "ref")
os.makedirs(REF, exist_ok=True)

with open(os.path.join(INP, "meta.json")) as f:
    M = json.load(f)
LC = M["latent_channels"]; OUT_CH = M["out_channels"]
BLOCK = M["block_out"]; LPB = M["layers_per_block"]; GLUMB = M["glumb_hidden"]
HEAD_DIM = M["head_dim"]; ATTN_STAGES = M["attention_stages"]
HLAT = M["h_lat"]; WLAT = M["w_lat"]

W = {k: v.astype(np.float64) for k, v in load_file(os.path.join(ROOT, "decoder.safetensors")).items()}
latent = np.fromfile(os.path.join(INP, "latent.bin"), np.float32).astype(np.float64).reshape(1, LC, HLAT, WLAT)

index = []
def tap(name, arr):
    arr = np.ascontiguousarray(arr)
    arr.astype(np.float32).ravel().tofile(os.path.join(REF, f"{name.replace('.', '_')}.bin"))
    index.append(dict(name=name, shape=list(arr.shape), file=f"{name.replace('.', '_')}.bin"))

def silu(x):
    return x / (1.0 + np.exp(-x))

def conv2d(x, w, b, stride=1, pad=1, groups=1):
    """x [1,Cin,H,W], w [Cout, Cin/groups, kh, kw] -> [1,Cout,Ho,Wo]. Standard cross-correlation, zero-pad."""
    _, cin, h, ww = x.shape
    cout, cing, kh, kw = w.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    ho = (h + 2 * pad - kh) // stride + 1
    wo = (ww + 2 * pad - kw) // stride + 1
    out = np.zeros((1, cout, ho, wo))
    gout = cout // groups
    for o in range(cout):
        g = o // gout
        acc = np.zeros((ho, wo))
        for ci in range(cing):
            src = g * cing + ci
            for ky in range(kh):
                for kx in range(kw):
                    acc += xp[0, src, ky:ky + ho * stride:stride, kx:kx + wo * stride:stride] * w[o, ci, ky, kx]
        out[0, o] = acc + (b[o] if b is not None else 0.0)
    return out

def rms_norm_channels(x, weight, bias, eps=1e-5):
    """Channelwise RMSNorm over the channel dim of NCHW (matches MusicDcaeDecoder.RmsNormChannels)."""
    c = x.shape[1]
    ms = (x * x).mean(axis=1, keepdims=True)         # [1,1,H,W]
    inv = 1.0 / np.sqrt(ms + eps)
    out = x * inv
    out = out * weight[None, :, None, None]
    if bias is not None:
        out = out + bias[None, :, None, None]
    return out

def relu(x):
    return np.maximum(x, 0.0)

# ── conv_in with repeat-interleave channel shortcut ──
def conv_in():
    deepest = BLOCK[-1]
    h = conv2d(latent, W["decoder.conv_in.weight"], W.get("decoder.conv_in.bias"))
    repeats = deepest // LC
    # AddRepeatInterleaveChannels: target[ci*repeats + r] += input[ci]
    short = np.repeat(latent, repeats, axis=1)        # channel ci -> [ci*repeats .. ci*repeats+repeats)
    return h + short

# ── ResBlock2d ──
def resblock(x, p):
    h1 = conv2d(x, W[f"{p}.conv1.weight"], W.get(f"{p}.conv1.bias"))
    h1 = silu(h1)
    h2 = conv2d(h1, W[f"{p}.conv2.weight"], None)
    h2 = rms_norm_channels(h2, W[f"{p}.norm.weight"], W.get(f"{p}.norm.bias"))
    return h2 + x

# ── EfficientViT block (no multiscale branch in the synthetic build) ──
def to_tokens(x):
    # [1,C,H,W] -> [S,C], S row-major over (h,w)
    _, c, h, w = x.shape
    return x[0].reshape(c, h * w).T.copy()

def from_tokens(tok, h, w):
    c = tok.shape[1]
    return tok.T.reshape(1, c, h, w).copy()

def linear_attention(q, k, v, head_dim):
    # tokens [S,C]; per head ReLU-linear attention with ones-row normalizer
    s, c = q.shape
    heads = c // head_dim
    out = np.zeros((s, c))
    for hh in range(heads):
        base = hh * head_dim
        qh = relu(q[:, base:base + head_dim])         # [S,d]
        kh = relu(k[:, base:base + head_dim])         # [S,d]
        vh = v[:, base:base + head_dim]               # [S,d]
        vk = vh.T @ kh                                 # [dv,dk]
        norm_row = kh.sum(axis=0)                       # [dk]
        num = qh @ vk.T                                 # [S,dv]
        den = qh @ norm_row + 1e-15                     # [S]
        out[:, base:base + head_dim] = num / den[:, None]
    return out

def evit(x, p, head_dim):
    _, c, h, w = x.shape
    tok = to_tokens(x)
    q = tok @ W[f"{p}.attn.to_q.weight"].T
    k = tok @ W[f"{p}.attn.to_k.weight"].T
    v = tok @ W[f"{p}.attn.to_v.weight"].T
    attn = linear_attention(q, k, v, head_dim)         # [S,C]
    # proj_out (key may be to_out or proj_out)
    pw = W.get(f"{p}.attn.to_out.weight", W.get(f"{p}.attn.proj_out.weight"))
    pb = W.get(f"{p}.attn.to_out.bias", W.get(f"{p}.attn.proj_out.bias"))
    proj = attn @ pw.T
    if pb is not None:
        proj = proj + pb
    res = from_tokens(proj, h, w)
    res = rms_norm_channels(res, W[f"{p}.attn.norm_out.weight"], W.get(f"{p}.attn.norm_out.bias"))
    x = res + x                                         # attention residual
    ff = glumb_conv2d(x, p)
    return ff + x                                       # ff residual

def glumb_conv2d(x, p):
    inv = conv2d(x, W[f"{p}.conv_out.conv_inverted.weight"], W.get(f"{p}.conv_out.conv_inverted.bias"), pad=0)
    inv = silu(inv)
    cdw = W[f"{p}.conv_out.conv_depth.weight"]
    depth = conv2d(inv, cdw, W.get(f"{p}.conv_out.conv_depth.bias"), pad=cdw.shape[-1] // 2, groups=inv.shape[1])
    hidden = depth.shape[1] // 2
    val = depth[:, :hidden]
    gate = depth[:, hidden:]
    gated = val * silu(gate)
    out = conv2d(gated, W[f"{p}.conv_out.conv_point.weight"], None, pad=0)
    out = rms_norm_channels(out, W[f"{p}.conv_out.norm.weight"], W.get(f"{p}.conv_out.norm.bias"))
    return out

# ── nearest-upsample + 3x3 conv + repeat-interleave/pixel-shuffle shortcut ──
def upsample(x, p_conv_w, p_conv_b, out_ch):
    _, c, h, w = x.shape
    interp = np.repeat(np.repeat(x, 2, axis=2), 2, axis=3)   # nearest 2x
    conv = conv2d(interp, p_conv_w, p_conv_b)                 # [1,out,2h,2w]
    # shortcut: repeat channels to out*4, pixel-shuffle(2), add (matches UpsampleInterpolateConv)
    repeats = out_ch * 4 // c
    for oc in range(out_ch):
        for py in range(2):
            for px in range(2):
                stacked = (oc * 2 + py) * 2 + px
                src = stacked // repeats
                conv[0, oc, py::2, px::2] += x[0, src]
    return conv

def main():
    h = conv_in(); tap("dcae.conv_in", h)
    n = len(BLOCK)
    for i in range(n - 1, -1, -1):
        attention = i >= n - ATTN_STAGES
        # blocks: child index = (1 if has upsample else 0) ..
        child = 1 if i < n - 1 else 0
        for j in range(LPB[i]):
            p = f"decoder.up_blocks.{i}.{child + j}"
            if attention:
                h = evit(h, p, HEAD_DIM)
            else:
                h = resblock(h, p)
        tap(f"dcae.stage.{i}", h)
        if i > 0:
            # upsample of stage i-1 fires when entering stage i-1 from stage i
            up_p = f"decoder.up_blocks.{i - 1}.0.conv"
            h = upsample(h, W[f"{up_p}.weight"], W.get(f"{up_p}.bias"), BLOCK[i - 1])
            tap(f"dcae.upsample.{i - 1}", h)
    h = rms_norm_channels(h, W["decoder.norm_out.weight"], W.get("decoder.norm_out.bias")); tap("dcae.norm_out", h)
    h = relu(h)
    mel = conv2d(h, W["decoder.conv_out.weight"], W.get("decoder.conv_out.bias")); tap("dcae.conv_out", mel)

    with open(os.path.join(REF, "index.json"), "w") as f:
        json.dump(index, f, indent=2)
    print(f"Wrote {len(index)} ACE-Step DCAE reference taps to {REF}")
    print(f"  mel shape={mel.shape} mean={mel.mean():.6f} abs_mean={np.abs(mel).mean():.6f} std={mel.std():.6f}")

if __name__ == "__main__":
    main()
