"""Standalone reference for the YuE x-codec (SoundStream) ENCODE path.

Reimplements, directly from the real ckpt_00360000.pth (`codec_model` state dict — the .pth is read
in place, no safetensors export needed), the full `SoundStream.encode` of
`m-a-p/xcodec_mini_infer/models/soundstream_hubert_new.py`:

    wav [B, 1, S] (16 kHz mono, raw float, NO normalization)
      ├─ semantic branch   x[:,0,:] -> F.pad(x, (160,160)) -> HuBERT-base(output_hidden_states=True)
      │                    -> stack all 13 hidden states -> mean(dim=1)          -> [B, T, 768]
      │                    -> .transpose(1,2) -> encoder_semantic (RepCodec)     -> [B, 768, T]
      ├─ acoustic branch   encoder (descript dac2.Encoder, d_model=64,
      │                    strides=[8,5,4,2], d_latent=256)                      -> [B, 256, T]
      │                    (if the two branches disagree on T, the acoustic branch is RE-RUN on
      │                     F.pad(x[:,0,:], (160,160)) — see PAD-FALLBACK below)
      ├─ cat([e_acoustic, e_semantic], dim=1)                                    -> [B, 1024, T]
      ├─ fc_prior (Linear 1024->1024 over channels)                              -> [B, 1024, T]
      └─ quantizer: EMA-VQ ResidualVectorQuantization, per stage
             ind = argmin_k ||residual - embed[k]||^2  (== argmax_k 2·x·e_k - ||e_k||^2)
             residual -= embed[ind]                                              -> codes [n_q, B, T]

YuE's ICL path (`inference/infer.py::encode_audio`) calls this with `target_bw=0.5`, and
`n_q = floor(bandwidth / (log2(bins)·frame_rate/1000)) = floor(0.5 / 0.5) = 1`, i.e. codebook 0 only.
This script dumps all 12 stages so a C# port can gate the residual loop deeper than YuE needs.

PAD-FALLBACK: the HuBERT stack (7 valid convs, strides 5,2,2,2,2,2,2 on a 160-padded waveform) and the
DAC stack (symmetric padding, strides 8,5,4,2) do NOT agree on the frame count for most input lengths —
they only agree when S is a multiple of 320 (and even then not universally). Upstream handles this by
re-running the acoustic encoder on the SAME 160-padded waveform. ~71% of lengths in 6000..10000 hit
this branch, so it is the common case, not an edge case. The `pad_*` tensors below exercise it.

We re-derive each submodule from the named tensors so we do NOT need the upstream packages
(transformers / RepCodec / descriptaudiocodec / einops). Weight-norm is fused by hand; note the two
DIFFERENT conventions in this checkpoint:
  * everything DAC-side (encoder.*, decoder_2.*) is weight_norm(dim=0) -> weight_g [C_out, 1, 1]
  * HuBERT's pos_conv is  weight_norm(dim=2) -> weight_g [1, 1, K]  (per kernel position)

Outputs a safetensors of named IO tensors a C# parity test reads. Layouts are called out per key
because upstream flips between channels-first and channels-last repeatedly:

  wav_in              [B, 1, S]      f32   the exact input waveform
  hubert_conv_out     [B, 512, T]    f32   HuBERT feature_extractor output (channels-first, post-GELU)
  hubert_hs_first     [B, T, 768]    f32   hidden_states[0] = encoder input (pos_conv + encoder LayerNorm)
  hubert_hs_last      [B, T, 768]    f32   hidden_states[12] = last_hidden_state
  hubert_feats        [B, T, 768]    f32   get_regress_target output = mean over all 13 hidden states
  semantic_enc_out    [B, 768, T]    f32   encoder_semantic(hubert_feats.transpose(1,2))
  acoustic_enc_out    [B, 256, T]    f32   encoder(wav)
  fc_prior_out        [B, 1024, T]   f32   fc_prior(cat([acoustic, semantic]))
  codes_i32           [12, B, T]     i32   RVQ code indices, all 12 stages
  rvq_quantized_out   [B, 1024, T]   f32   sum of the 12 stage lookups (== quantizer output `e_hat`)
  decoded_out         [B, 1, S']     f32   decoder_2(fc_post2(rvq_decode(codes, n_q=12)))
  decoded_cb0_out     [B, 1, S']     f32   same with n_q=1 — the YuE ICL round trip
  pad_wav_in          [B, 1, S2]     f32   an input length that TRIGGERS the pad-fallback
  pad_acoustic_enc_out[B, 256, T2]   f32   acoustic branch AFTER the fallback re-run
  pad_semantic_enc_out[B, 768, T2]   f32
  pad_fc_prior_out    [B, 1024, T2]  f32
  pad_codes_i32       [12, B, T2]    i32
"""
import math
import os
import sys
import types
import wave

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

CKPT = "/home/hartsy/Desktop/HartsyInference/Models/audio/music/YuE/xcodec/ckpt_00360000.pth"
CLIP = "/home/hartsy/Desktop/HartsyInference/tests/python-reference/silerovad_reference/jfk.wav"
OUT = ("/home/hartsy/Desktop/HartsyInference/tests/python-reference/"
       "yue_xcodec_encode_reference/io.safetensors")

DUMP_SAMPLES = 32_000      # 2.0 s @ 16 kHz -> 100 frames, both branches agree
PAD_SAMPLES = 8_080        # acoustic 25 vs semantic 26 frames -> forces the pad-fallback
N_Q = 12
EPS_LN = 1e-5
HEADS = 12
HEAD_DIM = 64

torch.manual_seed(0)

# ----- load codec_model.* from the raw .pth (needs an omegaconf stub: the ckpt pickles a DictConfig) -----
for _n in ["omegaconf", "omegaconf.dictconfig", "omegaconf.listconfig", "omegaconf.base", "omegaconf.nodes"]:
    sys.modules[_n] = types.ModuleType(_n)


class _Stub:
    def __init__(self, *a, **k): pass
    def __setstate__(self, s): self.__dict__.update(s if isinstance(s, dict) else {})


for _c in ["DictConfig", "ListConfig", "ContainerMetadata", "Metadata", "AnyNode", "ValueNode", "Node", "Container"]:
    for _m in ["omegaconf", "omegaconf.dictconfig", "omegaconf.listconfig", "omegaconf.base", "omegaconf.nodes"]:
        setattr(sys.modules[_m], _c, type(_c, (_Stub,), {}))

W = {k: v.float() for k, v in torch.load(CKPT, map_location="cpu", weights_only=False)["codec_model"].items()}


# ----- primitives -----
def wn(prefix):
    """Fuse weight_norm(dim=0): w = g * v / ||v||, norm over all dims except dim 0."""
    g, v = W[f"{prefix}.weight_g"], W[f"{prefix}.weight_v"]
    return g * v / v.norm(dim=tuple(range(1, v.dim())), keepdim=True)


def wn_dim2(prefix):
    """Fuse weight_norm(dim=2) — HuBERT's positional conv: one magnitude per kernel position.
    g is [1, 1, K], v is [C_out, C_in/groups, K]; the norm is over dims (0, 1)."""
    g, v = W[f"{prefix}.weight_g"], W[f"{prefix}.weight_v"]
    return g * v / v.norm(dim=(0, 1), keepdim=True)


def snake(x, alpha):
    """Snake1d: x + (1/alpha)·sin(alpha·x)^2; alpha is [1, C, 1]."""
    return x + (1.0 / (alpha + 1e-9)) * torch.sin(alpha * x) ** 2


def conv1d(x, prefix, pad, stride=1, dilation=1):
    return F.conv1d(x, wn(prefix), W[f"{prefix}.bias"], stride=stride, padding=pad, dilation=dilation)


def linear_over_channels(x, prefix):
    """Apply a Linear across the CHANNEL axis of a [B, C, T] tensor (upstream's
    `fc(x.transpose(1,2)).transpose(1,2)`)."""
    return F.linear(x.transpose(1, 2), W[f"{prefix}.weight"], W[f"{prefix}.bias"]).transpose(1, 2)


# ----- HuBERT-base semantic model (config: semantic_ckpts/hf_1_325000/config.json) -----
# conv_kernel [10,3,3,3,3,2,2], conv_stride [5,2,2,2,2,2,2], conv_bias false, feat_extract_norm "group"
# (GroupNorm on layer 0 only, num_groups == num_channels == 512), feat_extract_activation "gelu" (exact erf),
# hidden 768, 12 layers, 12 heads, ffn 3072, layer_norm_eps 1e-5, do_stable_layer_norm false (post-LN),
# num_conv_pos_embeddings 128, num_conv_pos_embedding_groups 16.
HUBERT_KERNELS = [10, 3, 3, 3, 3, 2, 2]
HUBERT_STRIDES = [5, 2, 2, 2, 2, 2, 2]


def hubert_feature_extractor(wav_bt):
    """wav [B, S] -> [B, 512, T]. All convs are VALID (no padding) and bias-free."""
    x = wav_bt.unsqueeze(1)
    for i, (k, s) in enumerate(zip(HUBERT_KERNELS, HUBERT_STRIDES)):
        x = F.conv1d(x, W[f"semantic_model.feature_extractor.conv_layers.{i}.conv.weight"], None, stride=s)
        if i == 0:      # feat_extract_norm="group": GroupNorm(512, 512) after layer 0 ONLY, before the act
            x = F.group_norm(x, 512,
                             W["semantic_model.feature_extractor.conv_layers.0.layer_norm.weight"],
                             W["semantic_model.feature_extractor.conv_layers.0.layer_norm.bias"], EPS_LN)
        x = F.gelu(x)   # exact erf GELU, not the tanh approximation
    return x


def hubert_layer(h, i):
    """One post-LayerNorm transformer layer (transformers' Wav2Vec2EncoderLayer)."""
    p = f"semantic_model.encoder.layers.{i}"
    b, t, _ = h.shape
    q = F.linear(h, W[f"{p}.attention.q_proj.weight"], W[f"{p}.attention.q_proj.bias"])
    k = F.linear(h, W[f"{p}.attention.k_proj.weight"], W[f"{p}.attention.k_proj.bias"])
    v = F.linear(h, W[f"{p}.attention.v_proj.weight"], W[f"{p}.attention.v_proj.bias"])
    shape = (b, t, HEADS, HEAD_DIM)
    q = q.view(shape).transpose(1, 2)
    k = k.view(shape).transpose(1, 2)
    v = v.view(shape).transpose(1, 2)
    a = F.scaled_dot_product_attention(q, k, v)                       # bidirectional, no mask
    a = a.transpose(1, 2).reshape(b, t, HEADS * HEAD_DIM)
    a = F.linear(a, W[f"{p}.attention.out_proj.weight"], W[f"{p}.attention.out_proj.bias"])
    h = F.layer_norm(h + a, (768,), W[f"{p}.layer_norm.weight"], W[f"{p}.layer_norm.bias"], EPS_LN)
    f = F.gelu(F.linear(h, W[f"{p}.feed_forward.intermediate_dense.weight"],
                        W[f"{p}.feed_forward.intermediate_dense.bias"]))
    f = F.linear(f, W[f"{p}.feed_forward.output_dense.weight"], W[f"{p}.feed_forward.output_dense.bias"])
    return F.layer_norm(h + f, (768,), W[f"{p}.final_layer_norm.weight"], W[f"{p}.final_layer_norm.bias"], EPS_LN)


def hubert_hidden_states(wav_bt):
    """wav [B, S] -> (conv_out [B,512,T], list of 13 hidden states, each [B, T, 768]).

    hidden_states[0] is collected AFTER pos_conv + encoder.layer_norm (transformers collects it at the
    top of the layer loop, and for do_stable_layer_norm=False the encoder LayerNorm runs before the
    loop). LayerDrop and SpecAugment are training-only — all 12 layers always run in eval."""
    conv = hubert_feature_extractor(wav_bt)                            # [B, 512, T]
    x = conv.transpose(1, 2)                                           # [B, T, 512]
    x = F.layer_norm(x, (512,), W["semantic_model.feature_projection.layer_norm.weight"],
                     W["semantic_model.feature_projection.layer_norm.bias"], EPS_LN)
    h = F.linear(x, W["semantic_model.feature_projection.projection.weight"],
                 W["semantic_model.feature_projection.projection.bias"])   # [B, T, 768]

    # Positional conv embedding: grouped Conv1d(768, 768, k=128, pad=64, groups=16), weight_norm(dim=2),
    # then drop the LAST frame (num_pad_remove = 1 because the kernel is even), then GELU, then add.
    pos = F.conv1d(h.transpose(1, 2), wn_dim2("semantic_model.encoder.pos_conv_embed.conv"),
                   W["semantic_model.encoder.pos_conv_embed.conv.bias"], padding=64, groups=16)
    pos = F.gelu(pos[..., :-1]).transpose(1, 2)
    h = F.layer_norm(h + pos, (768,), W["semantic_model.encoder.layer_norm.weight"],
                     W["semantic_model.encoder.layer_norm.bias"], EPS_LN)

    states = [h]
    for i in range(12):
        h = hubert_layer(h, i)
        states.append(h)
    return conv, states


def get_regress_target(x):
    """soundstream_hubert_new.SoundStream.get_regress_target: [B,1,S] -> [B, T, 768].

    NOTE the raw waveform goes straight into the model — the checkpoint's preprocessor_config.json says
    do_normalize=true, but the inference code never builds a Wav2Vec2FeatureExtractor, so NO zero-mean /
    unit-variance normalization is applied. Only the 160-sample reflect-free zero pad on each side."""
    padded = F.pad(x[:, 0, :], (160, 160))
    conv, states = hubert_hidden_states(padded)
    return conv, states, torch.stack(states, dim=1).mean(1)


# ----- encoder_semantic: RepCodec Encoder(input_channels=768, encode_channels=768) -----
# defaults: channel_ratios=(1,1), strides=(1,1), kernel_size=3, block_dilations=(1,1), unit_kernel_size=3.
# RepCodec's Conv1d wrapper derives padding = (k-1)//2 * dilation. The stem and both res-unit convs are
# bias-FREE (ResidualUnit's bias defaults to False and EncoderBlock never overrides it); only the
# block-exit conv carries a bias. Activation is ELU(alpha=1.0), applied BEFORE each conv.
def sem_res_unit(x, p):
    y = F.conv1d(F.elu(x), W[f"{p}.conv1.conv.weight"], None, padding=1)     # k=3, dil=1, no bias
    y = F.conv1d(F.elu(y), W[f"{p}.conv2.weight"], None)                     # Conv1d1x1, no bias
    return x + y


def encoder_semantic(x):
    """[B, 768, T] -> [B, 768, T] (strides are all 1, so T is preserved)."""
    x = F.conv1d(x, W["encoder_semantic.conv.conv.weight"], None, padding=1)  # stem, bias=False
    for b in range(2):
        p = f"encoder_semantic.conv_blocks.{b}"
        x = sem_res_unit(x, f"{p}.res_units.0")
        x = sem_res_unit(x, f"{p}.res_units.1")
        # stride==1 so the block-exit conv keeps kernel_size=3 (RepCodec's "special case")
        x = F.conv1d(x, W[f"{p}.conv.conv.weight"], W[f"{p}.conv.conv.bias"], stride=1, padding=1)
    return x


# ----- acoustic encoder: dac2.Encoder(d_model=64, strides=[8,5,4,2], d_latent=256) -----
def residual_unit(x, prefix, dilation):
    # block.0 Snake, block.1 WNConv1d k7 dilated, block.2 Snake, block.3 WNConv1d k1
    y = snake(x, W[f"{prefix}.block.0.alpha"])
    y = conv1d(y, f"{prefix}.block.1", ((7 - 1) * dilation) // 2, dilation=dilation)
    y = snake(y, W[f"{prefix}.block.2.alpha"])
    y = conv1d(y, f"{prefix}.block.3", 0)
    p = (x.shape[-1] - y.shape[-1]) // 2
    if p > 0:
        x = x[..., p:-p]
    return x + y


def encoder_block(x, prefix, stride):
    # block.0/1/2 ResidualUnit(d=1,3,9) at dim/2, block.3 Snake, block.4 WNConv1d(dim/2 -> dim)
    x = residual_unit(x, f"{prefix}.block.0", 1)
    x = residual_unit(x, f"{prefix}.block.1", 3)
    x = residual_unit(x, f"{prefix}.block.2", 9)
    x = snake(x, W[f"{prefix}.block.3.alpha"])
    return conv1d(x, f"{prefix}.block.4", math.ceil(stride / 2), stride=stride)   # k = 2*stride


def acoustic_encoder(x):
    """[B, 1, S] -> [B, 256, S/320]."""
    x = conv1d(x, "encoder.block.0", 3)                       # WNConv1d(1->64, k7, pad3)
    for i, stride in enumerate([8, 5, 4, 2]):
        x = encoder_block(x, f"encoder.block.{i + 1}", stride)
    x = snake(x, W["encoder.block.5.alpha"])
    return conv1d(x, "encoder.block.6", 1)                    # WNConv1d(1024->256, k3, pad1)


# ----- decode side (mirrors dump_yue_xcodec_reference.py, for the round trip) -----
def decoder_block(x, prefix, stride, out_pad):
    x = snake(x, W[f"{prefix}.block.0.alpha"])
    x = F.conv_transpose1d(x, wn(f"{prefix}.block.1"), W[f"{prefix}.block.1.bias"],
                           stride=stride, padding=math.ceil(stride / 2), output_padding=out_pad)
    x = residual_unit(x, f"{prefix}.block.2", 1)
    x = residual_unit(x, f"{prefix}.block.3", 3)
    return residual_unit(x, f"{prefix}.block.4", 9)


def decoder_2(x):
    x = conv1d(x, "decoder_2.model.0", 3)
    for i, stride in enumerate([8, 5, 4, 2]):
        x = decoder_block(x, f"decoder_2.model.{i + 1}", stride, 1 if i == 1 else 0)
    x = snake(x, W["decoder_2.model.5.alpha"])
    return conv1d(x, "decoder_2.model.6", 3)                  # no tanh — upstream commented it out


def rvq_decode(codes):
    out = None
    for i in range(codes.shape[0]):
        q = F.embedding(codes[i], W[f"quantizer.vq.layers.{i}._codebook.embed"]).transpose(1, 2)
        out = q if out is None else out + q
    return out


def codec_decode(codes):
    return decoder_2(linear_over_channels(rvq_decode(codes), "fc_post2"))


# ----- RVQ encode -----
def rvq_encode(e, n_q=N_Q):
    """[B, D, T] -> (codes [n_q, B, T] int64, quantized [B, D, T], per-stage residual RMS).

    Mirrors EuclideanCodebook.quantize verbatim: dist = -(|x|^2 - 2·x·Eᵀ + |E|^2), argmax over the
    codebook axis. The |x|^2 term is constant per position so argmax(dist) == argmin ||x - e||;
    torch's `.max(dim=-1).indices` takes the FIRST maximum on ties (EMA codebooks can hold duplicate
    dead-code rows, so exact ties are possible — a C# port must break ties the same way)."""
    residual = e
    codes, quantized, rms = [], None, [float(e.pow(2).mean().sqrt())]
    for i in range(n_q):
        embed = W[f"quantizer.vq.layers.{i}._codebook.embed"]          # [1024, 1024]
        x = residual.transpose(1, 2)                                    # [B, T, D]
        flat = x.reshape(-1, x.shape[-1])
        et = embed.t()
        dist = -(flat.pow(2).sum(1, keepdim=True) - 2 * flat @ et + et.pow(2).sum(0, keepdim=True))
        ind = dist.max(dim=-1).indices.view(x.shape[:-1])               # [B, T]
        q = F.embedding(ind, embed).transpose(1, 2)                     # [B, D, T]
        residual = residual - q
        quantized = q if quantized is None else quantized + q
        codes.append(ind)
        rms.append(float(residual.pow(2).mean().sqrt()))
    return torch.stack(codes), quantized, rms


# ----- full encode -----
def encode(x, n_q=N_Q, verbose=False):
    """[B, 1, S] -> dict of every stage boundary."""
    conv, states, feats = get_regress_target(x)
    semantic = encoder_semantic(feats.transpose(1, 2))                  # [B, 768, T]
    acoustic = acoustic_encoder(x)                                      # [B, 256, T]
    fallback = acoustic.shape[2] != semantic.shape[2]
    if fallback:
        # Upstream: self.encoder(torch.transpose(F.pad(x[:,0,:], (160,160)).unsqueeze(0), 0, 1))
        acoustic = acoustic_encoder(torch.transpose(F.pad(x[:, 0, :], (160, 160)).unsqueeze(0), 0, 1))
    if acoustic.shape[2] != semantic.shape[2]:
        raise RuntimeError(f"branches still disagree after the pad-fallback: "
                           f"{acoustic.shape[2]} vs {semantic.shape[2]}")
    e = torch.cat([acoustic, semantic], dim=1)                          # [B, 1024, T]
    e = linear_over_channels(e, "fc_prior")
    codes, quantized, rms = rvq_encode(e, n_q)
    if verbose:
        print(f"  pad-fallback: {fallback}   frames: {int(semantic.shape[2])}")
        print("  residual rms per stage:", " ".join(f"{r:.4f}" for r in rms))
    return {"conv": conv, "states": states, "feats": feats, "semantic": semantic,
            "acoustic": acoustic, "fc_prior": e, "codes": codes, "quantized": quantized,
            "rms": rms, "fallback": fallback}


# ----- metrics -----
def corr_rel_l2(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    n = min(a.numel(), b.numel())
    a, b = a[:n], b[:n]
    corr = float(((a - a.mean()) * (b - b.mean())).sum() / ((a - a.mean()).norm() * (b - b.mean()).norm() + 1e-12))
    return corr, float((a - b).norm() / (b.norm() + 1e-12))


def log_mel(w, sr=16_000, n_fft=1024, hop=256, n_mel=80):
    """Slaney-free HTK log-mel — hand-rolled so this file stays dependency-light."""
    power = torch.stft(w.flatten(), n_fft, hop, window=torch.hann_window(n_fft),
                       return_complex=True).abs() ** 2
    freqs = torch.linspace(0, sr / 2, n_fft // 2 + 1)
    hz2mel = lambda h: 2595.0 * torch.log10(1.0 + h / 700.0)
    mel2hz = lambda mm: 700.0 * (10.0 ** (mm / 2595.0) - 1.0)
    pts = mel2hz(torch.linspace(float(hz2mel(torch.tensor(0.0))),
                                float(hz2mel(torch.tensor(sr / 2.0))), n_mel + 2))
    fb = torch.zeros(n_mel, n_fft // 2 + 1)
    for i in range(n_mel):
        lo, ctr, hi = pts[i], pts[i + 1], pts[i + 2]
        fb[i] = torch.clamp(torch.minimum((freqs - lo) / (ctr - lo + 1e-9),
                                          (hi - freqs) / (hi - ctr + 1e-9)), min=0)
    return (fb @ power).add(1e-8).log()


def log_mel_corr(a, b):
    n = min(a.numel(), b.numel())
    return corr_rel_l2(log_mel(a.flatten()[:n]), log_mel(b.flatten()[:n]))[0]


def load_wav_mono16k(path):
    with wave.open(path, "rb") as f:
        assert f.getframerate() == 16_000 and f.getsampwidth() == 2, "expected 16-bit 16 kHz PCM"
        ch, n = f.getnchannels(), f.getnframes()
        pcm = np.frombuffer(f.readframes(n), dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        pcm = pcm.reshape(-1, ch).mean(axis=1)
    return torch.from_numpy(pcm.copy()).view(1, 1, -1)


# ----- run -----
full = load_wav_mono16k(CLIP)
print(f"clip {CLIP}  {full.shape[-1]} samples ({full.shape[-1] / 16000:.2f} s)")

print(f"[dump] S={DUMP_SAMPLES}")
x = full[..., :DUMP_SAMPLES].clone()
r = encode(x, verbose=True)

print(f"[pad-fallback] S={PAD_SAMPLES}")
xp = full[..., :PAD_SAMPLES].clone()
rp = encode(xp, verbose=True)
assert rp["fallback"], "PAD_SAMPLES did not trigger the acoustic pad-fallback"

decoded = codec_decode(r["codes"])
decoded_cb0 = codec_decode(r["codes"][:1])

os.makedirs(os.path.dirname(OUT), exist_ok=True)
save_file({
    "wav_in": x.contiguous(),
    "hubert_conv_out": r["conv"].contiguous(),
    "hubert_hs_first": r["states"][0].contiguous(),
    "hubert_hs_last": r["states"][12].contiguous(),
    "hubert_feats": r["feats"].contiguous(),
    "semantic_enc_out": r["semantic"].contiguous(),
    "acoustic_enc_out": r["acoustic"].contiguous(),
    "fc_prior_out": r["fc_prior"].contiguous(),
    "codes_i32": r["codes"].to(torch.int32).contiguous(),
    "rvq_quantized_out": r["quantized"].contiguous(),
    "decoded_out": decoded.contiguous(),
    "decoded_cb0_out": decoded_cb0.contiguous(),
    "pad_wav_in": xp.contiguous(),
    "pad_acoustic_enc_out": rp["acoustic"].contiguous(),
    "pad_semantic_enc_out": rp["semantic"].contiguous(),
    "pad_fc_prior_out": rp["fc_prior"].contiguous(),
    "pad_codes_i32": rp["codes"].to(torch.int32).contiguous(),
}, OUT)
print("wrote", OUT)

for k, v in [("hubert_feats", r["feats"]), ("semantic_enc_out", r["semantic"]),
             ("acoustic_enc_out", r["acoustic"]), ("fc_prior_out", r["fc_prior"]),
             ("rvq_quantized_out", r["quantized"])]:
    print(f"  {k:<18} {tuple(v.shape)}  mean {float(v.mean()):+.5f}  std {float(v.std()):.5f}")
print("  codes_i32         ", tuple(r["codes"].shape),
      " cb0[:12]", r["codes"][0, 0, :12].tolist())

# ----- round trip on the full clip -----
#
# READ THIS BEFORE CALLING A LOW NUMBER A BUG. The x-codec waveform decoder is adversarially trained
# and does NOT preserve waveform phase: sample-aligned correlation between input and reconstruction is
# ~0.2 in magnitude even for a correct encode (best cross-correlation over ±2000 lags is only ~0.27,
# and per-200 ms-frame correlation hovers around 0). The meaningful round-trip metric is spectral.
# Random codes are printed as the control: log-mel corr ~0.35 random vs ~0.96 for a real encode.
print(f"\n[roundtrip] full clip, S={full.shape[-1]}")
rf = encode(full, verbose=True)
ref = full[0, 0]
rand_codes = torch.randint(0, 1024, rf["codes"].shape, generator=torch.Generator().manual_seed(1))
for tag, codes in [("n_q=12", rf["codes"]), ("n_q=1 (YuE ICL)", rf["codes"][:1]),
                   ("n_q=12 RANDOM ctrl", rand_codes)]:
    rec = codec_decode(codes)[0, 0]
    n = min(ref.numel(), rec.numel())
    c, rl2 = corr_rel_l2(rec[:n], ref[:n])
    print(f"  {tag:<20} wave corr {c:+.4f}  relL2 {rl2:.4f}  log-mel corr {log_mel_corr(rec[:n], ref[:n]):+.4f}"
          f"  rms in/out {float(ref.pow(2).mean().sqrt()):.4f}/{float(rec.pow(2).mean().sqrt()):.4f}")

# Idempotence — the real proof the encode is self-consistent with the (already C#-verified) decode:
# re-encoding the reconstruction must recover most of the same latent and a large fraction of cb0.
r2 = encode(codec_decode(rf["codes"]).detach())
tt = min(rf["codes"].shape[2], r2["codes"].shape[2])
cb0 = float((rf["codes"][0, 0, :tt] == r2["codes"][0, 0, :tt]).float().mean())
lc, lr = corr_rel_l2(r2["fc_prior"][..., :tt], rf["fc_prior"][..., :tt])
print(f"  idempotence: fc_prior corr {lc:+.4f} relL2 {lr:.4f}   cb0 exact-match {cb0 * 100:.1f}% "
      f"(chance = {100 / 1024:.2f}%)")
