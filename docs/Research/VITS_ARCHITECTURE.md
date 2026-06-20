# VITS (SynthesizerTrn) — Architecture

> Build-ready spec for the shared VITS TTS core (Piper first consumer; reused by MeloTTS, GPT-SoVITS's
> SoVITS half, OpenVoice). Sources: `jaywalnut310/vits` (`models.py`/`modules.py`/`attentions.py`/
> `commons.py`), `rhasspy/piper` (`piper_train` config + `*.onnx.json`), `myshell-ai/MeloTTS`. Fetched
> 2026-06-19.

## Inference graph (`SynthesizerTrn.infer`, TTS path — `enc_q` posterior is training-only)
```
x, m_p, logs_p = enc_p(phonemes)              # TextEncoder, [1,192,Tx]
logw = sdp(x, reverse) or dp(x)               # log-durations
w_ceil = ceil(exp(logw) * length_scale)       # ≥1
attn  = generate_path(w_ceil)                 # monotonic; == repeat each text frame w_ceil[i] times
m_p, logs_p = expand(m_p, logs_p, w_ceil)     # [1,192,Ty]
z_p = m_p + randn*exp(logs_p)*noise_scale
z   = flow(z_p, reverse=True)                 # ResidualCouplingBlock
o   = dec(z)                                  # HiFi-GAN → waveform
```

## TextEncoder (`enc_p`)
emb(n_vocab,192)·√192 → 6 layers [rel-pos MHA → +res → LN → ConvFFN(k3) → +res → LN] → Conv1d(192→384) →
split (m_p, logs_p). Relative-position MHA: hidden 192, 2 heads, k_ch 96, `emb_rel_k/v` shape
`(1, 2·window+1=9, 96)`, window ±4. **Implemented directly**: for query i, key j, bias = `q_i·rel_k[clip(j-i,±4)+4]`
and output += `Σ p·rel_v[...]` (equivalent to the pad-reshape rel↔abs trick, far simpler). LayerNorm is over
the channel dim. FFN: Conv1d(192→768,k3)→ReLU→Conv1d(768→192,k3).

## Duration
- **Deterministic** (`dp`, built): conv1(192→256,k3)→chLN→ReLU→conv2→chLN→ReLU→proj(256→1) → logw.
- **Stochastic** (`sdp`, Piper default, staged): DDSConv + ConvFlow rational-quadratic spline (10 bins) +
  Log/Flip, `noise_scale_w` (0.8). The spline is the hard piece — deferred.

## Flow (`flow`, ResidualCouplingBlock, built)
4× [ResidualCouplingLayer(mean_only) + Flip]. Coupling: split → pre(Conv1d 96→192) → WN → post(192→96)=m →
reverse `x1=(x1-m)`. Flip reverses channels. WN = 4 dilated gated layers (in_layer Conv1d(192→384,k5,dil^i)
→ tanh·sigmoid → res_skip), weight-normed.

## HiFi-GAN decoder (`dec`, built)
conv_pre(192→512,k7) → per stage [leaky0.1 → ConvTranspose1d(stride=up_rate) → MRF (avg of resblocks)] →
leaky → conv_post(→1,k7) → tanh. ResBlock type1 = 2 convs (dilated+dil1)/residual; type2 = 1 conv. ∏up_rates
== hop. Weight-normed convs.

## Piper config (`*.onnx.json` omits arch — inject by quality)
sample_rate 22050, noise_scale 0.667, length_scale 1, noise_w 0.8, num_symbols 256, single-speaker (gin 0).
**Medium**: inter/hidden 192, filter 768, 2 heads, 6 layers, use_sdp true, resblock "2", kernels [3,5,7],
dilations [[1,2],[2,6],[3,12]], upsample [8,8,4]/[16,16,8]. **High**: resblock "1", [3,7,11], [[1,3,5]×3],
upsample [8,8,2,2]/[16,16,4,4]. Phoneme layout: `[BOS=1, p0, blank=0, p1, 0, …, pN, EOS=2]` (blank between
every phoneme — REQUIRED).

## Weight keys (PyTorch SynthesizerTrn)
`enc_p.{emb, encoder.attn_layers.{i}.{conv_q,k,v,o,emb_rel_k,emb_rel_v}, encoder.norm_layers_{1,2}.{i}.{gamma,beta},
encoder.ffn_layers.{i}.{conv_1,conv_2}, proj}`; `dp.{conv_1,norm_1,conv_2,norm_2,proj}` or `sdp.*`;
`flow.flows.{2i}.{pre,enc(in_layers/res_skip_layers/cond_layer),post}`; `dec.{conv_pre, ups.{i},
resblocks.{i}.{convs1,convs2}, conv_post}`; `emb_g` (multispeaker). Convs are `weight_g`/`weight_v` (fuse).
ONNX initializer names differ + weight-norm already fused.

## Variant deltas (for the next builds)
- **MeloTTS**: TextEncoder + `tone_emb`/`language_emb` + `bert_proj`(1024→h)/`ja_bert_proj`(768→h) summed;
  optional TransformerCouplingBlock flow; both SDP+DP blended by `sdp_ratio`; gin 256.
- **GPT-SoVITS SoVITS**: replaces text/prior with **semantic tokens** + reference encoder; reuses flow +
  posterior + HiFi-GAN; no espeak text encoder.
- **OpenVoice ToneColorConverter**: flow + posterior + HiFi-GAN run as a flow over mel conditioned on
  source→target speaker embeddings; **no text encoder / duration**.

## C# reuse / build status
Built (`Models/Vits/`): `VitsConfig`, `VitsTextEncoder` (direct rel-pos), `VitsDurationPredictor` (det),
`VitsWaveNet`, `VitsFlow`, `VitsHiFiGan`, `VitsLengthRegulator`, `VitsSynthesizer`, `VitsWeights` (weight-norm
fuse), + `PiperPipeline`. Reuses `IBackend` Conv1d/ConvTranspose1d (dilation+groups), `WeightNormFusion`,
`DeterministicRng`. **Staged:** the stochastic duration predictor spline, multispeaker `g` conditioning,
ONNX direct-load. The whole graph is synthetic-forward verified (finite audio).
