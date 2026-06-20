# RVC v2 (Retrieval-based Voice Conversion) — Architecture

> Build-ready spec for `SynthesizerTrnMs768NSFsid`. Sources: `RVC-Project/Retrieval-based-Voice-Conversion-WebUI`
> (`infer/lib/infer_pack/models.py`, `modules.py`, `attentions.py`, `infer/modules/vc/pipeline.py`),
> configs/v2. Fetched 2026-06-20. Shares the **HuBERT** content encoder with GPT-SoVITS.

## Synthesizer config (40k / 48k v2)
inter/hidden 192, filter 768, 2 heads, 6 enc layers, k3, gin **256**, resblock "1" ([3,7,11]/[[1,3,5]×3]),
upsample **[10,10,2,2]** kernels [16,16,4,4] (40k) / [12,10,2,2] [24,20,4,4] (48k), up_init 512, spk_embed 109,
ContentVec dim **768** (v2). `∏upsample == hop == 400/480`.

Inference: `m_p,logs_p,x_mask = enc_p(phone, pitch); z_p = m_p + exp(logs_p)*randn*0.66666; z = flow(z_p, g,
reverse); o = dec(z, nsff0, g)`. `pitch` (coarse bins) → enc_p; `nsff0` (continuous Hz) → dec source; `sid` →
`emb_g` → g (flow + dec).

## enc_p (`TextEncoder768`)
`emb_phone` Linear(768→192) + `emb_pitch` Embedding(256→192) (added when f0), `×√192`, LeakyReLU(0.1), → VITS
rel-pos Encoder (6L/2h/768/k3) → proj Conv1d(192→384) → split m_p/logs_p.

## dec (`GeneratorNSF`) — the key net-new piece
Plain HiFi-GAN (conv_pre, ConvTranspose ups, ResBlock1 MRF, conv_post tanh, cond g) **plus NSF source
injection**: `m_source = SourceModuleHnNSF(sr, harmonic_num=0)` builds a single phase-accumulated sine source
(+noise, voiced/unvoiced, `tanh(Linear(1,1))`) at audio rate from F0; per stage `i`, `noise_convs[i]` (Conv1d
in=1, out=stage_ch, k=`stride_f0·2`, stride=`stride_f0=∏rates[i+1:]`, pad=`stride_f0/2`; last k=1) downsamples
the source to the stage length and is **added to the upsampled features before the MRF**.

## F0 path
`f0 *= 2^(f0_up_key/12)`; same shifted f0 → `nsff0` (Hz, to source) AND coarse bins: `f0_mel=1127·ln(1+f0/700)`,
scaled over [f0_min 50, f0_max 1100] → `round((mel−mel_min)·254/(mel_max−mel_min)+1)` clamped [1,255] (0 reserved).
RMVPE (deferred) outputs per-frame Hz f0 at the hop rate — a clean caller-supplied drop-in.

## Weight keys
`enc_p.{emb_phone, emb_pitch, encoder.*, proj}`; `flow.flows.{0,2,4,6}.*` (4 flows, k5/n_layers3, cond_layer g);
`dec.{conv_pre, ups.{i}, noise_convs.{i}, resblocks.{i}.{convs1,convs2}, conv_post, cond, m_source.l_linear}`;
`emb_g [109,256]`. `enc_q.*` present but inference-unused.

## C# build status (`Models/Rvc/`)
- [x] **NSF source injection added to `VitsHiFiGan`** (`noise_convs` + optional `harSource` param, backward-
  compatible — plain HiFi-GAN consumers unchanged). The source is `NsfVocoderDsp.GenerateHarmonicSource`
  (reused; `harmonics=1`, `voicedThreshold=0`).
- [x] [`RvcTextEncoder.cs`](../../src/HartsyInference.Audio/Models/Rvc/RvcTextEncoder.cs) — emb_phone/emb_pitch front-end → **reused `VitsTextEncoder` layers** (MeloTTS pattern).
- [x] [`RvcPitch.cs`](../../src/HartsyInference.Audio/Models/Rvc/RvcPitch.cs) — mel coarse quantization + pitch shift. **Exact + tested.**
- [x] [`RvcPipeline.cs`](../../src/HartsyInference.Audio/Pipelines/RvcPipeline.cs) — content + F0 → enc_p → **reused `VitsFlow`** → NSF `VitsHiFiGan`. **Synthetic-forward verified** (finite audio). Content from the built `Hubert`.
- [ ] **Staged:** RMVPE pitch extraction (caller supplies per-frame Hz f0), the FAISS index retrieval blend,
  and ContentVec-vs-HuBERT layer-tap choice for v1 (256-d).
