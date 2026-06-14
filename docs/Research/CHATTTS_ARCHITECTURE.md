# ChatTTS — Architecture Research Notes

> Status: Complete | Last Updated: 2026-05-17 | Needed Before: HartsyInference.Audio (ChatTTS pipeline)

## Summary

ChatTTS (2noise team, 2024) is a conversational text-to-speech model designed for dialogue and chat use cases, with native paralinguistic control (laughs, sighs, breaks, orality level) and Chinese + English support. The architecture is a four-stage neural pipeline: **text BPE tokens -> GPT (semantic LM) -> audio token sequence (4-codebook RVQ) -> DVAE Decoder / Decoder head -> 100-bin mel spectrogram -> Vocos vocoder -> 24 kHz waveform**. Both transformers are LLaMA-style causal decoders sharing the same `LlamaConfig` shape (hidden=768, layers=20, heads=12, intermediate=3072, max position=4096) but with two separate roles:

- **GPT** (the "semantic LM") generates the audio token sequence one frame at a time. Each frame is `num_vq=4` parallel codebook indices over a vocabulary of `num_audio_tokens=626`. It is conditioned on the BERT-tokenized text plus a `[spk_emb]` token whose embedding is replaced at runtime by a 768-dim projection of the 192-dim speaker latent.
- **DVAE Decoder head** (called `decoder` in the codebase, separate file from the DVAE proper) converts the 4-codebook latent stream into a 100-channel mel spectrogram. The "DVAE" file additionally contains a Grouped Finite-Scalar-Quantizer (GFSQ) with `levels=[5,5,5,5]`, `G=2`, `R=2`, used during training and as a fallback path.

Voice identity is controlled by a 768-dim speaker embedding sampled from a stored mean+std Gaussian (`spk_stat.pt`, ~4 KB). The base release does not include zero-shot voice cloning (the DVAE encoder is reserved in the official roadmap), but the community has shipped clones via `Embed.safetensors` and the GFSQ encoder path. Streaming is supported by emitting partial DVAE-decoded mel chunks to Vocos as the GPT generates them; first-audio latency is ~500 ms on a single 4090 with `stream_speed=12000` (samples per batch) and `pass_first_n_batches=2`.

This file covers the model architecture and pipeline. The Vocos vocoder details (ConvNeXt backbone, iSTFT head) live in [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md) under the "Vocos" section. Mel spectrogram preprocessing is in [MEL_SPECTROGRAM.md](MEL_SPECTROGRAM.md). Generic BPE tokenization is in [TOKENIZERS.md](TOKENIZERS.md). The audio codec / RVQ design is in [AUDIO_CODECS.md](AUDIO_CODECS.md).

Sources: [2noise/ChatTTS](https://github.com/2noise/ChatTTS), [2Noise/ChatTTS HF](https://huggingface.co/2Noise/ChatTTS), [config/gpt.yaml](https://huggingface.co/2Noise/ChatTTS/raw/main/config/gpt.yaml), [config/decoder.yaml](https://huggingface.co/2Noise/ChatTTS/raw/main/config/decoder.yaml), [config/dvae.yaml](https://huggingface.co/2Noise/ChatTTS/raw/main/config/dvae.yaml), [config/vocos.yaml](https://huggingface.co/2Noise/ChatTTS/raw/main/config/vocos.yaml), [Vocos paper (arXiv:2306.00814)](https://arxiv.org/abs/2306.00814).

## Detailed Findings

### Overall Architecture

ChatTTS is a four-stage cascade:

```
[BPE text tokens + paralinguistic tags + [spk_emb]]
       |
       v
   GPT (LlamaModel, causal LM, 4 parallel VQ heads)        <-- "GPT" / semantic LM
       |
       v
   Audio token sequence: shape (T_audio, num_vq=4) over vocab 626
       |
       v
   Decoder (ConvNeXt-stack, mel head)                      <-- "Decoder" / DVAE decoder head
       |   (optional: route through full DVAE encoder+GFSQ)
       v
   Mel spectrogram: shape (100, T_audio)
       |
       v
   Vocos (ConvNeXt backbone + iSTFT head)                  <-- vocoder
       |
       v
   Waveform: 24,000 Hz mono float32
```

Two design subtleties matter for implementation:

1. **"GPT" and "Decoder" are not what the task brief suggests.** Both are not separate Llama transformers. The GPT is the only autoregressive Llama. The "Decoder" is a stack of dilated 1D ConvNeXt blocks (`DVAEDecoder` class) that maps the GPT's 4-VQ token embeddings (sum-pooled across codebooks then projected from `dim=384` -> `hidden=512`) to a 100-channel mel. The brief's description of "two transformers" is partly true if you count the older OpenVoice-style "RefineText" pass (the GPT can also be run with `[Sasr]`/`[Pasr]` prompts to refine raw text into ChatTTS-style text with inserted `[uv_break]`/`[laugh]` tags) — but that is the same GPT weights re-prompted, not a second transformer.
2. **There are TWO mel-decoder code paths.** `decoder.yaml` defines a small (`dim=384`, `hidden=512`, `n_layer=12`, `bn_dim=128`, no VQ) ConvNeXt decoder that consumes GPT hidden states directly (path: `Decoder.safetensors`, 104 MB). `dvae.yaml` defines a full DVAE with the same decoder shape plus an `Encoder` and a `GFSQ` quantizer (`dim=1024, levels=[5,5,5,5], G=2, R=2`) for going mel -> tokens -> mel (path: `DVAE.safetensors`, 60 MB). At inference, the default is `use_decoder=True` which selects the `Decoder` path; `DVAE` is used for the GFSQ quantization/round-trip and is essential if you want voice cloning. The token vocab `num_audio_tokens=626` is `5*5*5*5 + 1` per group, times 2 groups; this matches the GFSQ output space.

### GPT (Semantic Language Model)

The GPT is a `transformers.LlamaModel` with a custom embedding and a custom multi-head output. Source: `ChatTTS/model/gpt.py`, config: `config/gpt.yaml`.

**Llama configuration (`config/gpt.yaml`):**
```yaml
num_audio_tokens: 626
num_text_tokens: 21178

gpt_config:
  hidden_size: 768
  intermediate_size: 3072
  num_attention_heads: 12
  num_hidden_layers: 20
  use_cache: False
  max_position_embeddings: 4096
  spk_emb_dim: 192
  spk_KL: False
  num_audio_tokens: 626
  num_text_tokens: null
  num_vq: 4
```

So: 20 LLaMA decoder layers, 12 heads (head_dim=64), FFN inner=3072, RoPE positional embeddings, RMSNorm, SiLU/SwiGLU MLP, KV cache supported (toggled by `enable_cache`). Context window 4096 tokens.

**Custom IO:**
- `self.emb_text`: nn.Embedding(`num_text_tokens=21178`, 768)
- `self.emb_code[i]` for i in 0..3: nn.Embedding(`num_audio_tokens=626`, 768) — one per VQ codebook
- `self.head_text`: Linear(768, 21178) — text token logits
- `self.head_code[i]` for i in 0..3: Linear(768, 626) — per-codebook audio logits
- `self.emb_spk_proj`: Linear(`spk_emb_dim=192`, 768) — projects speaker latent into embedding space; the embedding at position(s) of `[spk_emb]` tokens is replaced (in-place add) with the projected speaker vector

**Two-mode operation:** A single set of weights, two runtime modes selected by which embedding table is active and which head reads off logits.

- *RefineText mode:* input is text tokens, output is text tokens. Used to insert paralinguistic tags into bare input text. Reads/writes via `emb_text`/`head_text`. Defaults: temperature=0.7, top_K=20, top_P=0.7, repetition_penalty=1.0, max_new_token=384, prompt `[Sasr] ... [Pasr]` -> generated -> `[Easr]`.
- *InferCode mode:* input is text + `[spk_emb]` + `[Stts]`, output is 4-VQ audio tokens per step. The "audio frame" at each step is read off by summing the per-VQ embeddings: `emb = sum_i emb_code[i](audio_tok[i])`. Logits at each step are the concatenation of 4 head outputs, sampled independently per VQ. Defaults: temperature=0.3, top_K=20, top_P=0.7, repetition_penalty=1.05, max_new_token=2048, prompt suffix `[speed_5]`.

### Decoder (DVAE Decoder Head) and DVAE

Source: `ChatTTS/model/dvae.py`, configs: `config/decoder.yaml` and `config/dvae.yaml`.

#### `Decoder.safetensors` path (default at inference)

```yaml
# config/decoder.yaml
dim: 384

decoder_config:
  idim: 384
  odim: 384
  hidden: 512
  n_layer: 12
  bn_dim: 128

vq_config: null
```

The `decoder` consumes the GPT's audio-token-summed embeddings. Each generated frame has 4 codebook indices; embedding lookup + sum gives a 768-dim vector, but for the Decoder path it is first projected/transformed to `dim=384` then through `DVAEDecoder` to produce a 100-channel mel frame. Internally `DVAEDecoder` is a stack of 12 `ConvNeXtBlock`-style dilated 1D conv blocks (kernel=7, dilation grows by powers of 2, bottleneck `bn_dim=128`), GroupNorm, GELU, residual. The output dim 384 is then projected to `n_mels=100`.

Note: `idim==odim==384` in the YAML; the projection 384 -> 100 happens in a final conv head outside the `DVAEDecoder` class. Total parameter count ~52M (single Decoder file is 104 MB FP32).

#### `DVAE.safetensors` path (encoder + GFSQ + decoder)

```yaml
# config/dvae.yaml
dim: 512
decoder_config:
  idim: 512
  odim: 512
  n_layer: 12
  bn_dim: 128

vq_config:
  dim: 1024
  levels: [5,5,5,5]
  G: 2
  R: 2
```

Full discrete VAE. Encoder mirrors decoder (ConvNeXt stack), produces a 1024-dim latent, quantized by **GFSQ** (`GroupedResidualFSQ` — see [AUDIO_CODECS.md](AUDIO_CODECS.md) RVQ section). GFSQ params:

- `levels=[5,5,5,5]` -> per-quantizer scalar levels (FSQ with 4 dims at 5 levels each = 5^4 = 625 codewords per quantizer, plus an "empty/skip" giving 626; matches `num_audio_tokens`).
- `G=2` groups (so the 1024-dim latent is split into 2 groups of 512).
- `R=2` residual quantizers per group (total 2 groups x 2 residuals = 4 codebooks per frame -> matches `num_vq=4`).

The "4 codebooks per audio frame" exposed to the GPT therefore correspond to (group, residual) pairs in GFSQ ordering `(g0,r0), (g0,r1), (g1,r0), (g1,r1)`.

For inference *without* voice cloning we only need the **Decoder path**: GPT tokens -> `emb_code` sum-pool -> projection -> `DVAEDecoder` -> mel. The GFSQ is not on the inference critical path (the GPT already generates token indices directly).

For voice cloning we need the **DVAE encoder + GFSQ**: take a reference waveform, mel-encode it, run encoder, GFSQ-quantize to get prompt audio tokens, feed those as a prefix to GPT (analogous to "speech continuation").

### Vocos Vocoder

Source: `ChatTTS/model/vocos.py` (wraps the `gemelo-ai/vocos` reference), config: `config/vocos.yaml`.

```yaml
feature_extractor:
  class_path: vocos.feature_extractors.MelSpectrogramFeatures
  init_args:
    sample_rate: 24000
    n_fft: 1024
    hop_length: 256
    n_mels: 100
    padding: center

backbone:
  class_path: vocos.models.VocosBackbone
  init_args:
    input_channels: 100
    dim: 512
    intermediate_dim: 1536
    num_layers: 8

head:
  class_path: vocos.heads.ISTFTHead
  init_args:
    dim: 512
    n_fft: 1024
    hop_length: 256
    padding: center
```

A standard Vocos: 8-layer ConvNeXt backbone at width 512, FFN 1536, predicting STFT magnitude+phase at n_fft=1024 / hop=256, then iSTFT. Hop ratio 256 samples @ 24 kHz = 10.67 ms per mel frame. **Full architecture in [HIFIGAN_VOCODER.md#vocos-architecture-alternative](HIFIGAN_VOCODER.md)** — same shape (input_channels=100, dim=512, num_layers=8, intermediate_dim=1536) as `vocos-mel-24khz` listed there.

File: `Vocos.safetensors`, 54.3 MB FP32, ~13.5M params.

### Text Tokenizer

Source: `ChatTTS/model/tokenizer.py`, asset: `asset/tokenizer/` directory (saved via `BertTokenizerFast.save_pretrained`).

- **Backend:** HuggingFace `BertTokenizerFast` (WordPiece, not pure BPE despite project lore; the underlying vocab is a Chinese+English BERT vocab extended with paralinguistic special tokens).
- **Vocab size:** 21,178 (`num_text_tokens` in gpt.yaml). This includes ~21,128 BERT base tokens plus the additional special tokens below.
- **Standard tokens:** `[CLS]`, `[SEP]`, `[PAD]`, `[MASK]`, `[UNK]`.

**Full additional-special-token list** (from `asset/tokenizer/special_tokens_map.json`):

| Category | Tokens |
|---|---|
| ASR markers | `[Sasr]`, `[Pasr]`, `[Easr]` |
| TTS markers | `[Stts]`, `[Ptts]`, `[Etts]` |
| Break markers (control) | `[Sbreak]`, `[Pbreak]`, `[Ebreak]` |
| Inline break tags | `[uv_break]` (unvoiced/short pause), `[v_break]` (voiced pause), `[lbreak]` (long break), `[llbreak]` (very long break) |
| Paralinguistic | `[laugh]`, `[undefine]` |
| Speaker control | `[spk_emb]`, `[empty_spk]` |
| Stream classification | `[music]`, `[pure]` |
| Numbered break (intensity 0..7) | `[break_0]`, `[break_1]`, ..., `[break_7]` |
| Numbered laugh (intensity 0..2) | `[laugh_0]`, `[laugh_1]`, `[laugh_2]` |
| Orality level (0=formal..9=informal) | `[oral_0]`, `[oral_1]`, ..., `[oral_9]` |
| Speech speed (0=slowest..9=fastest, default 5) | `[speed_0]`, `[speed_1]`, ..., `[speed_9]` |

Total additional special tokens: 61. Speed default at inference is `[speed_5]`.

### Paralinguistic Control — Inline Tag Vocabulary

The user-facing tag set, as documented in repo READMEs and verified against `special_tokens_map.json`:

| Tag | Effect |
|---|---|
| `[laugh]` | Inserts a laugh; intensity sampled |
| `[laugh_0]` `[laugh_1]` `[laugh_2]` | Laugh with explicit intensity 0..2 (low / medium / high) |
| `[oral_0]` .. `[oral_9]` | Orality level — 0 = very formal, 9 = very informal/colloquial. Controls global speaking style for the utterance |
| `[uv_break]` | Short unvoiced pause (breath / phrase break) |
| `[v_break]` | Voiced/filled pause (~"uh" insert) |
| `[lbreak]` | Long pause (sentence-level) |
| `[llbreak]` | Very long pause |
| `[break_0]` .. `[break_7]` | Quantized pause length 0..7 (sub-tag of `[*break]` family) |
| `[speed_0]` .. `[speed_9]` | Speech rate 0=slow..9=fast; default `[speed_5]` is auto-appended to InferCode prompts |
| `[spk_emb]` | Single placeholder token whose embedding is overwritten with the speaker projection at the GPT input |
| `[empty_spk]` | Used when no speaker latent is supplied (zero/neutral voice) |
| `[Stts]` `[Ptts]` `[Etts]` | Begin / middle / end of TTS segment (wraps text input to GPT in InferCode mode) |
| `[Sasr]` `[Pasr]` `[Easr]` | Begin / middle / end of refine-text segment (wraps text in RefineText mode) |
| `[Sbreak]` `[Pbreak]` `[Ebreak]` | Begin / middle / end of break-only segment |
| `[music]` `[pure]` | Stream-classification tokens (music vs pure speech) |
| `[undefine]` | Fallback for unknown paralinguistic events |

Typical user input: `Hello [uv_break] world [laugh] this is so cool [lbreak] really.` plus an implicit `[oral_2][speed_5]` prefix and `[spk_emb]` token added by the wrapper.

### Speaker Control / Voice Latent

Source: `ChatTTS/model/speaker.py`, asset: `asset/spk_stat.pt` (~4.26 KB).

**Speaker latent dimensionality:** 768 dims (in the embedding/input space). However the *learned distribution* is parameterised at `spk_emb_dim=192` for compactness, then projected up to 768 by `emb_spk_proj: Linear(192, 768)` inside the GPT. The user-facing "speaker vector" stored / sampled / cloned is the **768-dim** post-projection vector (this is what gets compressed and shared as a base16384 string).

**Sampling distribution:** Gaussian with per-dim `mean` and `std` learned during training and stored in `spk_stat.pt`. The file is a 2 x 768 float16 tensor (mean row, std row) packaged as base16384(LZMA2(float16)).

```python
def sample_random():
    rand = torch.randn(768)
    spk = rand * std + mean       # both shape (768,) float16
    return _encode(spk)            # -> base16384 string
```

**Encoding format for share/transport:** float16 tensor -> LZMA2 (preset 9 | PRESET_EXTREME) -> pybase16384 string. This is what the official samples mean by "voice seed strings".

**Seed-fixed sampling:** Set `torch.manual_seed(seed)` then call `sample_random()` -> deterministic voice for any given seed.

**Voice cloning (community / roadmap):** The base release does NOT ship a voice encoder. Cloning is done via the DVAE encoder + GFSQ path: mel-encode a reference clip, tokenize to 4-VQ codes, feed as prefix audio tokens to GPT (speech continuation). The repo roadmap lists "Open-source DVAE encoder and zero-shot inferring code" as planned; the `DVAE_full.pt` (60.4 MB, contains encoder+decoder+GFSQ) is the file used for this.

### Inference Pipeline Pseudocode

```
INPUT:
  text:         "Hello [uv_break] world [laugh] cool stuff."
  spk_emb:      torch.Tensor (768,) float16   # sampled or fixed by seed
  use_decoder:  True                          # else use DVAE path
  stream:       False                         # see Streaming section
PIPELINE:

  # ---- 1. Text preparation ----
  text = normalizer(text)                      # number/zh normalization
  text = "[Stts][spk_emb][speed_5]" + text + "[Etts]"
  token_ids = tokenizer.encode(text)
  # shape: (1, T_text)  int64

  # ---- 2. Speaker injection at GPT input ----
  spk_proj = emb_spk_proj(spk_emb)             # (768,) -> (768,)
  input_embeds = emb_text(token_ids)           # (1, T_text, 768)
  input_embeds[:, idx([spk_emb]), :] = spk_proj

  # ---- 3. GPT autoregressive generation (InferCode mode) ----
  # State: KV cache of 20 layers, 12 heads, head_dim=64
  audio_tokens = []   # list of (4,) per step
  while not done:
      logits_text, logits_audio = gpt.forward_step(input_embeds, kv_cache)
      # logits_audio: list of 4 tensors, each (1, vocab=626)
      next_tok = [sample(logits_audio[i], temp=0.3, top_k=20, top_p=0.7,
                         rep_pen=1.05) for i in range(4)]
      audio_tokens.append(next_tok)
      if next_tok == eos_token or len(audio_tokens) >= 2048:
          done = True
      # Re-embed by summing 4 codebook embeddings:
      next_emb = sum(emb_code[i](next_tok[i]) for i in range(4))  # (1, 1, 768)
      input_embeds = next_emb
  # shape after loop: audio_tokens (T_audio, 4)  int64 in [0, 625]

  # ---- 4. Decoder (mel synthesis) ----
  # Re-embed and sum:
  audio_emb = sum(emb_code[i](audio_tokens[:, i]) for i in range(4))
  # shape: (T_audio, 768)
  audio_emb = audio_emb.transpose(0, 1).unsqueeze(0)   # (1, 768, T_audio)
  # Project 768 -> 384, then DVAEDecoder (12 ConvNeXt blocks, hidden=512, bn=128):
  mel = decoder(audio_emb)
  # shape: (1, 100, T_audio)

  # ---- 5. Vocos (mel -> waveform) ----
  # ConvNeXt backbone (8 layers, dim=512) -> ISTFT head (n_fft=1024, hop=256)
  audio = vocos.decode(mel)
  # shape: (1, num_samples)  where num_samples = T_audio * 256
  # sample rate: 24000 Hz, mono, float32 in [-1, 1]

OUTPUT:
  audio  # write to WAV at 24000 Hz
```

**Frame-rate arithmetic:** Audio mel frames at 24000 / 256 = 93.75 Hz; the GPT generates 1 audio frame per autoregressive step; reference quote of "~7 semantic tokens per second" refers to **post-DVAE-downsample** semantic units, while the raw codec rate is 4 codebooks x 93.75 frames/s ≈ 375 token-codes/s. The 2048-step `max_new_token` cap therefore allows up to ~21.8 s of audio per generate call.

### Sampling Parameters

| Param | RefineText (text->text) | InferCode (text->audio) |
|---|---|---|
| temperature | 0.7 | 0.3 |
| top_K | 20 | 20 |
| top_P | 0.7 | 0.7 |
| repetition_penalty | 1.0 | 1.05 |
| max_new_token | 384 | 2048 |
| min_new_token | 0 | 0 |
| prompt suffix | (none) | `[speed_5]` |

Repetition penalty in InferCode is applied **per VQ codebook independently** — each of the 4 heads tracks its own recent-token set. EOS for audio is a fixed `eos_token` id within the 626-vocab.

### Streaming

ChatTTS supports streaming via chunked decode:

- **Mechanism:** the GPT generate loop yields audio tokens in chunks (default `stream_speed=12000` samples-worth per emit, i.e., 12000 / 256 ≈ 47 mel frames ≈ 47 GPT steps per chunk). Each chunk is run through Decoder + Vocos and yielded as a partial waveform.
- **Warmup:** `pass_first_n_batches=2` — the first 2 chunks are accumulated before the first audio is yielded, so streaming actually begins after ~94 frames (~1.0 s of audio worth of GPT generation), but those are computed in parallel with the Decoder warmup, giving an end-user **first-audio latency of ~500 ms** on a 4090 (per the upstream README).
- **Why warmup is needed:** Vocos uses centered STFT with reflective padding; the boundary frames of a chunk have edge artifacts if not overlapped with neighbours. The first 2 batches give context for centered convolutions in both Decoder (kernel=7, dilation up to 2^11) and Vocos.
- **Implementation note:** the Decoder has receptive field on the order of 256 mel frames due to ConvNeXt dilations. Chunk boundaries should overlap by at least half the receptive field and the leading edge of each output chunk discarded — see "Implementation Notes" #6.

### HuggingFace Files

From `https://huggingface.co/2Noise/ChatTTS/tree/main` and `asset/` directory:

| Path | Size | Format | Purpose |
|---|---|---|---|
| `asset/GPT.pt` | 901 MB | PyTorch pickle | LlamaModel weights + emb_code/head_code/emb_spk_proj. **NOT shipped as safetensors in main repo.** |
| `asset/gpt/*` | (dir) | safetensors shards | Same GPT weights converted to safetensors (sharded). Use this for pure-C# loading. |
| `asset/Embed.safetensors` | 146 MB | safetensors | Text + 4 audio embedding tables (`emb_text` 21178x768, `emb_code[0..3]` 626x768 each) + `emb_spk_proj` (192x768). Split out so users can swap voice embeddings independently. |
| `asset/Decoder.pt` | 104 MB | PyTorch pickle | DVAEDecoder weights for `Decoder.safetensors` path (`dim=384, hidden=512, n_layer=12`). |
| `asset/Decoder.safetensors` | 104 MB | safetensors | Same as `Decoder.pt`. |
| `asset/DVAE.pt` | 27.7 MB | PyTorch pickle | DVAE decoder-only weights (no encoder), `dim=512` variant. |
| `asset/DVAE.safetensors` | 60.4 MB | safetensors | Same DVAE decoder, safetensors copy (note: size differs from `.pt` because safetensors stores additional metadata and possibly the GFSQ codebooks). |
| `asset/DVAE_full.pt` | 60.4 MB | PyTorch pickle | Full DVAE: encoder + GFSQ + decoder, needed for voice cloning. |
| `asset/Vocos.pt` | 54.4 MB | PyTorch pickle | Vocos backbone + iSTFT head + feature extractor. |
| `asset/Vocos.safetensors` | 54.3 MB | safetensors | Same as Vocos.pt. |
| `asset/spk_stat.pt` | 4.26 KB | PyTorch pickle | (2, 768) float16: row 0 = mean, row 1 = std of speaker latent Gaussian. |
| `asset/tokenizer.pt` | 337 KB | PyTorch pickle | Pickled BertTokenizerFast (legacy single-file copy). |
| `asset/tokenizer/` | (dir) | HF tokenizer save | `vocab.txt`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`. Use this directory. |
| `config/gpt.yaml` | 346 B | YAML | GPT hyperparameters (see above). |
| `config/decoder.yaml` | 117 B | YAML | Decoder hyperparameters (see above). |
| `config/dvae.yaml` | 143 B | YAML | DVAE hyperparameters (see above). |
| `config/vocos.yaml` | 460 B | YAML | Vocos hyperparameters (see above). |
| `config/path.yaml` | 309 B | YAML | Default asset paths. |

Total repo size: ~2.37 GB. Minimum runtime set (no cloning): `GPT.safetensors` (or `gpt/`) + `Embed.safetensors` + `Decoder.safetensors` + `Vocos.safetensors` + `spk_stat.pt` + `tokenizer/` = ~1.2 GB FP32 / ~600 MB FP16.

### Memory and Performance

| Setting | Value |
|---|---|
| GPU VRAM (FP32, no cloning, batch 1) | ~5.5 GB |
| GPU VRAM (FP16, no cloning, batch 1) | ~3 GB |
| GPU VRAM minimum (per README, 30 s clip) | 4 GB |
| Real-Time Factor (RTF) — 4090 FP16, flash attn | ~0.3 |
| RTF — 4090D (Chinese SKU) | ~0.65 |
| GPT tokens / sec — 4090 | ~94 (= 1 mel frame / step / 10.67 ms) |
| First-audio latency (streaming, 4090) | ~500 ms |
| `stream_speed` default | 12000 samples / batch |
| `pass_first_n_batches` | 2 |
| KV-cache memory @ 4096 ctx, FP16, batch 1 | ~96 MB (20 layers * 12 heads * 64 head_dim * 4096 ctx * 2 (k+v) * 2 bytes) |

Total parameters: GPT ~220M (LlamaModel) + ~30M IO heads = ~250M. Decoder ~52M. DVAE ~30M. Vocos ~13.5M. Aggregate ~345M.

## Key Numbers / Constants

| Constant | Value | Notes |
|---|---|---|
| Sample rate | 24,000 Hz | Vocos output, fixed |
| Mel channels (n_mels) | 100 | Vocos input, Decoder output |
| Mel hop length | 256 | Frame rate 93.75 Hz |
| Mel n_fft | 1024 | Vocos STFT |
| Mel frame rate | 93.75 frames/s | 24000 / 256 |
| `num_text_tokens` | 21,178 | BERT vocab + 61 specials |
| `num_audio_tokens` | 626 | FSQ levels 5^4 + 1 per group |
| `num_vq` | 4 | (G=2) x (R=2) codebooks per audio frame |
| GPT hidden_size | 768 | LLaMA model dim |
| GPT layers | 20 | Decoder-only causal LM |
| GPT attention heads | 12 | head_dim = 64 |
| GPT intermediate_size | 3072 | SwiGLU FFN inner |
| GPT max_position_embeddings | 4096 | RoPE context window |
| GPT max_new_token (InferCode) | 2048 | ~21.8 s of audio |
| GPT max_new_token (RefineText) | 384 | text -> tagged text |
| spk_emb_dim | 192 | Pre-projection latent dim |
| Speaker vector (user-facing) | 768 dim float16 | Post-projection, base16384(LZMA2) encoded |
| Decoder dim | 384 | Input/output dim of DVAEDecoder |
| Decoder hidden | 512 | Internal ConvNeXt width |
| Decoder n_layer | 12 | Dilated ConvNeXt blocks |
| Decoder bn_dim | 128 | Bottleneck dim |
| DVAE dim | 512 | DVAE decoder I/O |
| DVAE GFSQ levels | [5,5,5,5] | FSQ scalar levels per quantizer |
| DVAE GFSQ G | 2 | Codebook groups |
| DVAE GFSQ R | 2 | Residual quantizers per group |
| Vocos dim | 512 | ConvNeXt width |
| Vocos intermediate_dim | 1536 | FFN inner |
| Vocos num_layers | 8 | ConvNeXt blocks |
| Stream chunk samples | 12,000 | `stream_speed` |
| Stream warmup batches | 2 | `pass_first_n_batches` |
| GPT InferCode temperature | 0.3 | |
| GPT InferCode top_K | 20 | |
| GPT InferCode top_P | 0.7 | |
| GPT InferCode repetition_penalty | 1.05 | Per VQ codebook |
| GPT RefineText temperature | 0.7 | |
| GPT RefineText top_K | 20 | |
| GPT RefineText top_P | 0.7 | |
| GPT RefineText repetition_penalty | 1.0 | |

## Data Layouts / Formats

### Text Input Tokenization

```
User input:   "Hello [uv_break] world [laugh] cool."
After wrap:   "[Stts][spk_emb][speed_5]Hello [uv_break] world [laugh] cool.[Etts]"
After BERT:   [CLS] [Stts] [spk_emb] [speed_5] hello [uv_break] world [laugh] cool . [Etts] [SEP]
Token IDs:    (1, T_text) int64,  T_text typically 20-200
```

### Speaker Latent

```
Stored shape:    (768,) float16
Sampling:        randn(768) * std + mean   where std, mean from spk_stat.pt (768,) each
Transport:       base16384(LZMA2_preset9_extreme(float16_bytes))
                 -> ASCII string ~1.5 KB per voice
GPT projection:  Linear(192, 768)   # NOTE: per gpt.yaml spk_emb_dim=192, but
                                    # the stored vector is 768-dim and shape-matches
                                    # the embedding space directly; some forks project
                                    # 768->192->768. Verify with reference run.
```

### GPT Audio Token Stream

```
Per step output: (4,) int64  in [0, 625]
Full sequence:   (T_audio, 4) int64,  T_audio in [1, 2048]
Re-embedded:     sum_i emb_code[i](tok[:, i])  -> (T_audio, 768) float
```

### Mel Spectrogram (Decoder output / Vocos input)

```
Shape:           (1, 100, T_audio) float32
Frame rate:      93.75 Hz
Range:           log-mel scale (Vocos's MelSpectrogramFeatures normalization)
```

### Audio Output

```
Shape:           (num_samples,) float32, range ~[-1, 1]
Sample rate:     24,000 Hz
num_samples:     T_audio * 256
Format:          mono PCM, write to WAV via soundfile / NAudio / similar
```

### `spk_stat.pt`

```
Format: PyTorch pickle of (2, 768) float16 tensor
        OR: base16384-encoded compressed bytes in some forks
Layout: row 0 = mean (768,), row 1 = std (768,)
Size:   ~3 KB on disk; 4.26 KB with pickle overhead
```

## Algorithm Steps

### Full Inference (Non-Streaming)

```
1. TEXT INPUT
   text = "Hello [uv_break] world [laugh] cool."
   text = normalizer(text)                # Chinese number/punctuation normalization
   text = "[Stts][spk_emb][speed_5]" + text + "[Etts]"

2. TOKENIZATION (BertTokenizerFast)
   token_ids = tokenizer.encode(text)     # (1, T_text)
   attention_mask, text_mask              # text_mask marks where text tokens are
                                          # (vs paralinguistic specials)

3. SPEAKER LATENT
   if seed: torch.manual_seed(seed)
   spk_emb = sample_random(spk_stat.mean, spk_stat.std)   # (768,) float16
   spk_proj = emb_spk_proj(spk_emb.float())               # (768,) -> (768,)

4. INPUT EMBEDDINGS
   inp = emb_text(token_ids)                              # (1, T_text, 768)
   inp[:, idx_of([spk_emb]), :] = spk_proj                # in-place patch

5. GPT FORWARD (InferCode mode)
   kv_cache = init_kv_cache(20 layers, 12 heads, head_dim=64)
   audio_tokens = []
   # First step: full context
   h = gpt(inp, kv_cache=kv_cache)                        # (1, T_text, 768)
   logits = [head_code[i](h[:, -1, :]) for i in range(4)]
   for step in range(max_new_token=2048):
       tok = [sample(l, temp=0.3, top_k=20, top_p=0.7, rep_pen=1.05)
              for l in logits]                            # 4 ints
       if any(tok == eos): break
       audio_tokens.append(tok)
       # Re-embed
       e = sum(emb_code[i](tok[i]) for i in range(4))     # (1, 1, 768)
       h = gpt(e, kv_cache=kv_cache)                      # incremental
       logits = [head_code[i](h[:, -1, :]) for i in range(4)]
   audio_tokens = stack(audio_tokens)                     # (T_audio, 4)

6. DECODER (mel synthesis)
   emb = sum(emb_code[i](audio_tokens[:, i]) for i in range(4))   # (T_audio, 768)
   emb = emb.T.unsqueeze(0)                                       # (1, 768, T_audio)
   # In the Decoder path, an internal projection 768 -> 384 happens here
   mel = decoder(emb)                                             # (1, 100, T_audio)

7. VOCOS (waveform synthesis)
   audio = vocos.head(vocos.backbone(mel))                # (1, T_audio * 256)
   audio = audio.squeeze(0)                               # (num_samples,)

8. OUTPUT
   write_wav("out.wav", audio.cpu().numpy(), 24000)
```

### Streaming Inference

Same as above through step 4. Then:

```
5. STREAMING GPT + DECODER + VOCOS
   chunk_frames = stream_speed // 256       # 12000 // 256 = 46
   buffer = []                              # accumulated audio_tokens
   batch_idx = 0
   while True:
       # Generate chunk_frames new audio tokens via GPT incremental decoding
       new_tokens = generate_n_tokens(gpt, chunk_frames, kv_cache, ...)
       buffer.extend(new_tokens)
       batch_idx += 1
       if eos_seen: break
       if batch_idx <= pass_first_n_batches=2:
           continue                          # warmup, accumulate
       # Decode the LATEST window (with overlap-and-discard at boundary)
       window = buffer[-(chunk_frames + overlap):]
       mel = decoder(embed_sum(window))      # (1, 100, len(window))
       audio = vocos(mel)                    # (1, len(window)*256)
       # Discard `overlap*256` samples at the leading edge
       yield audio[overlap*256:]
   # Final flush: decode any remaining tail
```

First-yield happens after batch 3 (~3 * 12000 = 36000 samples worth of generated tokens = ~1.5 s wallclock of GPT generation at 0.3 RTF -> ~450-500 ms perceived latency).

### Speaker Sampling

```
1. spk_stat.pt -> (2, 768) float16 -> (mean, std) each (768,)
2. (optional) torch.manual_seed(seed)
3. raw = torch.randn(768)
4. spk = raw * std + mean                    # (768,) float16
5. spk_str = base16384(lzma2(spk.bytes))      # ~1.5 KB string
6. At inference: spk = lzma2_decode(base16384_decode(spk_str)).view(float16, 768)
7. inject into GPT: emb_spk_proj(spk.float())
```

## Open Questions

- [ ] `spk_emb_dim=192` in gpt.yaml vs the 768-dim shared/sampled vector: is there an inner 768 -> 192 -> 768 bottleneck, or does `emb_spk_proj` actually take 768 dims directly? Verify against the reference forward by inspecting the loaded weight shape of `emb_spk_proj`.
- [ ] Exact ordering of the 4 codebook indices per frame (group-major `(g0r0, g0r1, g1r0, g1r1)` vs residual-major `(g0r0, g1r0, g0r1, g1r1)`). Affects which head_code corresponds to which GFSQ index. Validate by checking the DVAE-encoded mel of a known clip and matching `emb_code` weight statistics.
- [ ] EOS token id within the 626-vocab — not explicitly documented; needs source-read of `core.py`.
- [ ] Does the Decoder path apply any input-side projection 768 -> 384, or is the 384-dim coming from a separate `Embed.safetensors` table (`emb_code` 626x384 distinct from GPT's 626x768)? The `Embed.safetensors` size (146 MB) suggests it stores embeddings at GPT-width 768, but the Decoder consumes 384 — clarify with a load-and-print.
- [ ] Repetition penalty per-VQ: confirm whether the "recent token set" for the penalty is shared across all 4 VQ heads or strictly per-codebook. The `gen_logits` source comment says per-codebook; verify.
- [ ] Streaming overlap-and-discard exact window sizes for both Decoder and Vocos at chunk boundaries. The reference implementation may use a fixed `pad_left`/`pad_right` rather than a true overlap-add.
- [ ] Whether the `decoder.yaml` `vq_config: null` means the Decoder-path takes embeddings directly (no quantization), and the GFSQ in `dvae.yaml` is only for the round-trip encode path. Reading `core.py` suggests yes; confirm.

## Implementation Notes for HartsyInference

1. **LlamaModel is standard.** We will have already built a causal LLaMA implementation for dotLLM (RoPE + RMSNorm + SwiGLU FFN, KV cache). The GPT here is a stock LLaMA with hidden=768, layers=20, heads=12 — reuse the dotLLM Llama block directly. The only ChatTTS-specific wrapper is:
   - Replace `emb_tokens` with a switchable `emb_text` (21178x768) + 4x `emb_code[i]` (626x768 each). At each step pick the right embedding table based on which mode we're in.
   - Multi-head output: 4 `head_code[i]` (768 -> 626) for InferCode, 1 `head_text` (768 -> 21178) for RefineText. Sample each VQ head independently and concatenate.
   - Speaker injection: a single in-place embedding overwrite at `[spk_emb]` token positions. Use a span-find on token_ids before the first forward.

2. **KV cache is mandatory.** 4096-token context, 20 layers, 12 heads, head_dim=64, two tensors (K and V) per layer. Pre-allocate at session start (`KvCache.Allocate(batch=1, max_seq=4096, layers=20, heads=12, head_dim=64, dtype=fp16)` ~ 96 MB). The yaml says `use_cache: False` but that is a training-time default; we always enable cache at inference (`enable_cache=True` in core.py).

3. **Two model files, not three.** The "GPT" and "Decoder" share NO weights and load from separate safetensors:
   - `GPT.safetensors` (or sharded `gpt/`) -> Llama + emb_text + emb_code[0..3] + head_text + head_code[0..3] + emb_spk_proj.
   - `Decoder.safetensors` -> DVAEDecoder + input projection + output mel head.
   - `Vocos.safetensors` -> ConvNeXt + iSTFT head.
   - `Embed.safetensors` is *redundant* with weights inside GPT.safetensors — it's a convenience split for cloning experiments. Load only if a user supplies a custom Embed override.

4. **DVAEDecoder is a dilated ConvNeXt stack, NOT a transformer.** Implement as 12 residual blocks:
   ```
   block(x) = x + pointwise_conv(GELU(GroupNorm(depthwise_conv_d(x))))
   ```
   with depthwise kernel=7 and dilation per-block doubling from 1 up to 2^11. Bottleneck projects to `bn_dim=128` mid-block. Final `Conv1d(384, 100, 1)` projects to mel. Build this as `HartsyInference.Audio/Modules/DvaeDecoder.cs`.

5. **GFSQ for cloning path (optional v2).** Group-Residual Finite Scalar Quantization. The encoder maps mel -> 1024-dim latent -> reshape to `(G=2, R=2, 4)` where 4 is the FSQ-dim and the levels are `[5,5,5,5]`. FSQ quantization is `round(tanh(z) * (L-1)/2) -> int in [-(L-1)/2, (L-1)/2]`. Decode by reversing. Implement as `HartsyInference.Audio/Modules/GroupedResidualFSQ.cs` — see [AUDIO_CODECS.md](AUDIO_CODECS.md) for the math.

6. **Vocos reuses existing code.** We have a Vocos implementation planned for F5-TTS and other models — see [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md) Vocos section. Same exact config (input_channels=100, dim=512, num_layers=8, intermediate_dim=1536, iSTFTHead with n_fft=1024, hop=256). Single shared Vocos module across ChatTTS / F5-TTS / Kokoro (note: Kokoro uses iSTFTNet not Vocos — different module).

7. **Tokenizer:** BertTokenizerFast is HF's WordPiece, not BPE. We need a pure-C# WordPiece tokenizer that supports Chinese character-splitting and 61 additional special tokens. Plan: a generic WordPiece in `HartsyInference.Core/Tokenization/WordPieceTokenizer.cs`. The `tokenizer.json` from the HF tokenizer directory contains the full vocab + merge rules + special token list — parse once at load time.

8. **Special token handling:** `[spk_emb]` is a single token ID that needs to be located in the encoded sequence (one or more positions); the embedding at that position is replaced. Cache the ID at tokenizer-load time. Same for `[empty_spk]`. All `[break_X]`, `[laugh_X]`, `[oral_X]`, `[speed_X]` are normal tokens that go through the embedding table — no special handling beyond standard tokenization.

9. **Speaker latent transport:** Implement base16384 + LZMA2 codec so users can paste/share voice strings. .NET has `System.IO.Compression` but no LZMA — use `LZMA-SDK` or `SharpCompress` (already an indirect dep via the safetensors loader? verify). base16384 is unusual — port the ~50 lines from pybase16384 to C#. Keep it pure-C# (no native deps).

10. **Sampling:** Each of the 4 audio heads samples *independently* per step with temperature=0.3, top_k=20, top_p=0.7, repetition_penalty=1.05. Use the shared `HartsyInference.Core.Sampling.LogitsSampler` — no special ChatTTS-only sampler needed. Repetition penalty: track the last N (N=64? confirm) generated tokens per codebook; multiply logits of those tokens by `1/1.05` before top-k.

11. **Streaming requires chunked Decoder + Vocos.** Plan: a `ChatTtsStreamingSession` that owns the GPT KV-cache, a ring buffer of recent audio tokens (size = chunk + overlap), and a callback fired on each chunk. The first 2 chunks are computed but suppressed; from chunk 3 onward each new chunk is decoded with an overlap window, the leading samples of the Decoder output are dropped, and the rest is yielded. Same overlap/discard pattern for Vocos (centered STFT has reflective padding artifacts).

12. **Speaker stat loading:** `spk_stat.pt` is a tiny PyTorch pickle. Either ship a pre-converted `spk_stat.bin` (raw 2x768 float16, 3 KB) at model packaging time, or implement a minimal pickle reader for the float16 tensor case. Recommend pre-conversion.

13. **Determinism:** RNG for speaker sampling and for sampling tokens uses PyTorch's Mersenne Twister. We need bit-exact reproducibility against the reference for validation. Plan: use a HartsyInference-specific seedable PRNG and only compare final waveforms within an audio-quality tolerance (PESQ > 4.0, mel-cepstral distance < 1.0) — bit-exact PyTorch RNG reproduction is not worth the effort.

14. **Validation tolerances:**
    - GPT logits: cosine similarity > 0.9999 vs reference at FP32, > 0.999 at FP16.
    - Decoder mel output: mean abs error < 0.01 dB per mel bin vs reference.
    - Vocos waveform: PCM MAE < 1e-3 at FP32, < 5e-3 at FP16. PESQ vs reference > 4.2.
    - End-to-end: mel-cepstral distance < 0.5 between HartsyInference output and PyTorch output for the same text + seed.

15. **Memory budget at FP16:**
    - GPT weights: ~440 MB
    - Decoder weights: ~52 MB
    - Vocos weights: ~13.5 MB
    - Speaker stat: ~3 KB
    - KV cache (4096 ctx): ~96 MB
    - Activations (one chunk): ~50 MB
    - Total resident: ~650 MB; with 1 GB safety margin we're at <2 GB VRAM for a single session.

## Reference Implementations

- [2noise/ChatTTS](https://github.com/2noise/ChatTTS) — Official Python/PyTorch reference.
- [2Noise/ChatTTS HF](https://huggingface.co/2Noise/ChatTTS) — Official model weights and configs.
- [lenML/ChatTTS-Forge](https://huggingface.co/spaces/lenML/ChatTTS-Forge) — Community fork with extra tooling, voice cloning experiments, and webUI.
- [gemelo-ai/vocos](https://github.com/gemelo-ai/vocos) — Reference Vocos implementation (we mirror its math).
- [Vocos paper (arXiv:2306.00814)](https://arxiv.org/abs/2306.00814) — Vocos: Closing the gap between time-domain and Fourier-based neural vocoders.
- [Finite Scalar Quantization (arXiv:2309.15505)](https://arxiv.org/abs/2309.15505) — FSQ paper (the basis of GFSQ).
- [lucidrains/vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch) — Reference `GroupedResidualFSQ` implementation that ChatTTS uses.
- [HuggingFace LlamaModel docs](https://huggingface.co/docs/transformers/model_doc/llama) — Reference for the GPT backbone.
