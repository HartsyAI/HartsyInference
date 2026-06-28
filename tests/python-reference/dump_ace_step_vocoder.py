"""
ACE-Step ADaMoS HiFi-GAN vocoder — Python step-by-step parity reference.

Independent NumPy float64 re-implementation of `AdaMosHiFiGanV1.Decode` (mel [1,melBins,T] -> waveform
[1,1,T*rateProduct]), mirroring src/HartsyInference.Diffusion/Models/Music/AdaMosHiFiGanV1.cs. The C# diff test
(AceStepVocoderDiffTests) GENERATES the tiny synthetic checkpoint + fixed mel and runs the C# decode with
ACE_STEP_DEBUG_DIR; this script loads the SAME weights+input, runs this reference, and dumps the same tap points.
diff_ace_step_vocoder.py compares.

Validates: ConvNeXt backbone (stem k7 conv -> channels-first LayerNorm -> per-stage [channels-first LayerNorm ->
k1 channel transition -> depthwise-k7/pointwise-GELU block + gamma layer-scale]), and the HiFi-GAN head (conv_pre k5
-> 2 x [SiLU -> ConvTranspose1d upsample (symmetric pad (k-rate)/2) -> multi-receptive-field ResBlock1 average] ->
SiLU -> conv_post k5 -> tanh). GELU is the tanh approximation (matches the CPU backend). SiLU = x*sigmoid(x).

Run AFTER the C# test (which writes Output/ace_step_vocoder_parity/):
  python3 tests/python-reference/dump_ace_step_vocoder.py
"""
import json
import os
import numpy as np
from safetensors.numpy import load_file

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.join(REPO, "Output", "ace_step_vocoder_parity")
INP = os.path.join(ROOT, "inputs")
REF = os.path.join(ROOT, "ref")
os.makedirs(REF, exist_ok=True)

with open(os.path.join(INP, "meta.json")) as f:
    M = json.load(f)
MEL_BINS = M["mel_bins"]
DIMS = M["dims"]
DEPTHS = M["depths"]
UP_RATES = M["upsample_rates"]
UP_KERNELS = M["upsample_kernels"]
RES_KERNELS = M["resblock_kernels"]
T = M["t"]

W = {k: v.astype(np.float64) for k, v in load_file(os.path.join(ROOT, "vocoder.safetensors")).items()}
mel = np.fromfile(os.path.join(INP, "mel.bin"), np.float32).astype(np.float64).reshape(1, MEL_BINS, T)

index = []
def tap(name, arr):
    arr = np.ascontiguousarray(arr)
    arr.astype(np.float32).ravel().tofile(os.path.join(REF, f"{name.replace('.', '_')}.bin"))
    index.append(dict(name=name, shape=list(arr.shape), file=f"{name.replace('.', '_')}.bin"))

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def silu(x):
    return x * sigmoid(x)

# GELU tanh approximation (matches ActivationKernels.Gelu).
SQRT_2_OVER_PI = np.sqrt(2.0 / np.pi)
GELU_COEFF = 0.044715
def gelu(x):
    inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x * x * x)
    return x * 0.5 * (1.0 + np.tanh(inner))

def conv1d(x, w, b, stride=1, pad=0, dilation=1, groups=1):
    """x [1,Cin,Tin], w [Cout, Cin/groups, K] -> [1,Cout,Tout]. Cross-correlation, zero-pad."""
    _, cin, tin = x.shape
    cout, cing, k = w.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad)))
    eff = dilation * (k - 1) + 1
    tout = (tin + 2 * pad - eff) // stride + 1
    out = np.zeros((1, cout, tout))
    gout = cout // groups
    for o in range(cout):
        g = o // gout
        acc = np.zeros(tout)
        for ci in range(cing):
            src = g * cing + ci
            for kk in range(k):
                acc += xp[0, src, kk * dilation: kk * dilation + tout * stride: stride] * w[o, ci, kk]
        out[0, o] = acc + (b[o] if b is not None else 0.0)
    return out

def conv_transpose1d(x, w, b, stride, pad):
    """x [1,Cin,Tin], w [Cin, Cout, K] (PyTorch ConvTranspose1d layout) -> [1,Cout,Tout]. Symmetric pad."""
    _, cin, tin = x.shape
    cinw, cout, k = w.shape
    assert cinw == cin
    tout_raw = (tin - 1) * stride + k
    tout = tout_raw - 2 * pad
    out = np.zeros((1, cout, tout))
    if b is not None:
        out += b[None, :, None]
    for ic in range(cin):
        for oc in range(cout):
            for i in range(tin):
                start = i * stride - pad
                for kk in range(k):
                    j = start + kk
                    if 0 <= j < tout:
                        out[0, oc, j] += x[0, ic, i] * w[ic, oc, kk]
    return out

def layernorm_channels(x, weight, bias, eps=1e-6):
    """LayerNorm over the channel axis of [1,C,T] (ConvNeXt channels_first)."""
    mean = x.mean(axis=1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=1, keepdims=True)
    out = (x - mean) / np.sqrt(var + eps)
    out = out * weight[None, :, None]
    if bias is not None:
        out = out + bias[None, :, None]
    return out

def convnext_block(x, p):
    _, c, t = x.shape
    dw = W[f"{p}.dwconv.weight"]  # [c,1,7]
    k = dw.shape[-1]
    h = conv1d(x, dw, W.get(f"{p}.dwconv.bias"), pad=k // 2, groups=c)
    h = layernorm_channels(h, W[f"{p}.norm.weight"], W.get(f"{p}.norm.bias"))
    # Pointwise MLP per time step (token rows [T,C]).
    rows = h[0].T  # [T,C]
    pw1 = W[f"{p}.pwconv1.weight"]  # [hidden,c]
    mid = rows @ pw1.T + W[f"{p}.pwconv1.bias"][None, :]
    act = gelu(mid)
    pw2 = W[f"{p}.pwconv2.weight"]  # [c,hidden]
    outrows = act @ pw2.T + W[f"{p}.pwconv2.bias"][None, :]
    res = outrows.T[None, :, :]  # [1,c,t]
    gamma = W.get(f"{p}.gamma")
    if gamma is not None:
        res = res * gamma[None, :, None]
    return x + res

def hifi_resblock(x, p, k):
    dilations = [1, 3, 5]
    cur = x.copy()
    for j, d in enumerate(dilations):
        w1 = W[f"{p}.convs1.{j}.weight"]  # [c,c,k]
        a = silu(cur)
        h1 = conv1d(a, w1, W.get(f"{p}.convs1.{j}.bias"), pad=d * (k - 1) // 2, dilation=d)
        h1 = silu(h1)
        w2 = W[f"{p}.convs2.{j}.weight"]
        h2 = conv1d(h1, w2, W.get(f"{p}.convs2.{j}.bias"), pad=k // 2)
        cur = h2 + cur
    return cur

def main():
    h = mel.copy()
    n = len(DIMS)
    for i in range(n):
        if i == 0:
            cw = W["backbone.channel_layers.0.0.weight"]  # [d0,mel,7]
            k = cw.shape[-1]
            h = conv1d(h, cw, W.get("backbone.channel_layers.0.0.bias"), pad=k // 2)
            h = layernorm_channels(h, W["backbone.channel_layers.0.1.weight"],
                                   W.get("backbone.channel_layers.0.1.bias"))
        else:
            h = layernorm_channels(h, W[f"backbone.channel_layers.{i}.0.weight"],
                                   W.get(f"backbone.channel_layers.{i}.0.bias"))
            tw = W[f"backbone.channel_layers.{i}.1.weight"]  # [di,di-1,1]
            h = conv1d(h, tw, W.get(f"backbone.channel_layers.{i}.1.bias"), pad=0)
        for j in range(DEPTHS[i]):
            h = convnext_block(h, f"backbone.stages.{i}.{j}")
        tap(f"voc.stage.{i}", h)
    h = layernorm_channels(h, W["backbone.norm.weight"], W.get("backbone.norm.bias"))
    tap("voc.backbone_norm", h)

    # HiFi-GAN head.
    cpw = W["head.conv_pre.weight"]  # [Cpre,d_last,5]
    pk = cpw.shape[-1]
    cur = conv1d(h, cpw, W.get("head.conv_pre.bias"), pad=pk // 2)
    tap("voc.conv_pre", cur)

    num_kernels = len(RES_KERNELS)
    for i in range(len(UP_RATES)):
        cur = silu(cur)
        uw = W[f"head.ups.{i}.weight"]  # [Cin,Cout,K]
        rate = UP_RATES[i]; kernel = UP_KERNELS[i]
        pad = (kernel - rate) // 2
        up = conv_transpose1d(cur, uw, W.get(f"head.ups.{i}.bias"), rate, pad)
        tap(f"voc.upsample.{i}", up)
        acc = np.zeros_like(up)
        for j in range(num_kernels):
            r = hifi_resblock(up, f"head.resblocks.{i * num_kernels + j}", RES_KERNELS[j])
            acc += r
        cur = acc / num_kernels
        tap(f"voc.mrf.{i}", cur)

    cur = silu(cur)
    cpostw = W["head.conv_post.weight"]  # [1,Cpost,5]
    postk = cpostw.shape[-1]
    wav = conv1d(cur, cpostw, W.get("head.conv_post.bias"), pad=postk // 2)
    wav = np.tanh(wav)
    tap("voc.waveform", wav)

    with open(os.path.join(REF, "index.json"), "w") as f:
        json.dump(index, f, indent=2)
    print(f"Wrote {len(index)} ACE-Step vocoder reference taps to {REF}")
    print(f"  wav shape={wav.shape} mean={wav.mean():.6f} abs_mean={np.abs(wav).mean():.6f} std={wav.std():.6f}")

if __name__ == "__main__":
    main()
