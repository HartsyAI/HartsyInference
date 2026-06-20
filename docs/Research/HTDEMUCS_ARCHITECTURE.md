# HTDemucs (Hybrid Transformer Demucs v4) — Architecture

> Build-ready spec for music source separation (4 stereo stems). Sources: `facebookresearch/demucs`
> (`htdemucs.py`, `hdemucs.py`, `transformer.py`, `demucs.py`, `states.py`), paper arXiv:2211.08553.
> Fetched 2026-06-20.

## Structure
Dual-branch U-Net joined by a cross-domain transformer. **Spec branch**: 2D convs over the complex-as-channels
STFT. **Time branch**: 1D convs over the waveform. Both encode in parallel; at the bottleneck a
`CrossTransformerEncoder` mixes the two token streams; both decode with U-Net skips; the spec output → complex
spectrogram → iSTFT, **summed** with the time-branch waveform. Out `(B, 4, 2, L)`, stems
[drums, bass, other, vocals] (read from ckpt `kwargs`). Stereo 44.1 kHz.

Dims: channels 48, depth 4, growth 2 (→48,96,192,384), nfft 4096, hop 1024, bottom_channels 512, t_layers 5,
t_heads 8, FFN 2048, freq_emb 0.2. norm_starts 4 (no GroupNorm in main convs at depth 4).

## STFT / complex-as-channels
`spectro`: reflect-pad, `torch.stft(4096, hop 1024, hann, center=False, normalized=True)`, drop Nyquist →
`z [B,C,2048,T]` complex. `_magnitude` (cac): view_as_real → permute → `[B, 2·C, 2048, T]` (= 4 ch stereo).
`_ispec`: re-pad Nyquist + temporal, inverse stft. cac → no explicit mask multiply (net predicts complex spec).

## Spec/time conv layers (HEncLayer/HDecLayer)
Enc: main conv (2D kernel (8,1)/stride (4,1) freq-only; 1D kernel 8/stride 4) → GELU → 1×1 rewrite + **plain
GLU** (`a·σ(b)`, NOT GeGLU) → optional DConv residual (off at depth 4). Layer 0 adds `freq_emb`. Dec: GLU
rewrite → transposed conv → GELU (skip on last); returns `(z, pre)`; `x += skip` before. Freq collapses to 1 at
the bottleneck; the time branch's `inject` is added into the freq encoder there (the merge).

## CrossTransformerEncoder
2 parallel ModuleLists (`layers` spec, `layers_t` time). Even idx = self-attn, odd idx = cross-attn (each
stream's query attends to the other stream). Pre-norm, gated **LayerScale** (`gamma_1`/`gamma_2`), 2D sin
pos-emb (spec, tokens `(t1 fr)`) + 1D sin (time), learnable `weight_pos_embed`, `norm_in`/`norm_in_t`. dim 512,
8 heads, FFN 2048 GELU. `bottom_channels` 1×1 Conv1d up/down-samplers around it.

## Forward
wav → time-normalize; STFT → cac → freq-normalize; encode loop (time enc → inject into freq enc, save skips);
bottom_channels up → CrossTransformer → down; decode loop (freq dec returns `pre` → feeds time dec, skips);
freq → denorm → complex → iSTFT; time → denorm; **out = time + spec**.

## Weights / `.th`
`{klass, args, kwargs, state}` pickle (fp16 → cast fp32); released ships as a "bag" YAML of signatures. Keys:
`encoder/decoder.{i}.*` (freq), `tencoder/tdecoder.{i}.*` (time), `crosstransformer.{layers,layers_t}.{i}.*` +
`norm_in*` + `channel_{up,down}sampler[_t]`, `freq_emb.embedding.weight`. Conv layer: `.conv`, `.rewrite`,
`.dconv.layers.{j}`; cross/self: `{self,cross}_attn`, `linear1/2`, `norm1/2/3`, `gamma_1/2.scale`.

## C# build status (`Models/Demucs/`)
- [x] [`HtDemucsConfig`](../../src/HartsyInference.Audio/Models/Demucs/HtDemucsConfig.cs) — 4 stereo stems, dims. **Tested.**
- [x] [`DemucsCrossTransformer`](../../src/HartsyInference.Audio/Models/Demucs/DemucsCrossTransformer.cs) — the defining novel piece: self/cross alternating layers, 2D/1D sin pos-emb, LayerScale, reusing SDPA + `DiaHeads`. **Synthetic-forward verified** (mixes both streams, finite).
- [x] [`DemucsConvBlock`](../../src/HartsyInference.Audio/Models/Demucs/DemucsConvBlock.cs) — HEnc/HDec 1D+2D GLU conv block (Conv2D/ConvTranspose2d/Conv1d + GELU + plain GLU). **Synthetic-forward verified** (enc + dec, finite).
- [ ] **Staged (full-graph assembly):** the STFT→cac→dual-branch→**freq-collapse + time-inject merge**→
  transformer→decode→mask→iSTFT→sum `HtDemucs` orchestration (the freq-collapse/merge is the bit-exact-risky
  part — needs a reference run to lock down), the DConv residual branch, `freq_emb`, the `.th` bag loader (reuse
  the GameCraft `.pt` pickle loader), and Conv2D dilation (not needed for main convs). The novel components are
  verified; the assembly is the documented follow-up.
