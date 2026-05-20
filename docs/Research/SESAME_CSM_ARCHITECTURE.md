# Sesame CSM — Architecture Research Notes

> Status: Complete | Last Updated: 2026-05-17 | Needed Before: SharpInference.Audio (Sesame CSM pipeline)

## Summary

**CSM (Conversational Speech Model)** is SesameAI's open-weight, real-time conversational TTS released March 13 2025 as `sesame/csm-1b` (Apache 2.0). It is the architecture behind the viral "Maya / Miles" demo at [sesame.com](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice) and is purpose-built for low-latency, full-duplex spoken dialogue. The design is a **dual-transformer**: a Llama-3.2-style **backbone** (1B params, 16 layers, 2048 hidden, GQA 32 heads / 8 KV heads) consumes interleaved text + audio frames and predicts the **semantic codebook (codebook 0)** of the next 80 ms Mimi frame; a much smaller **audio decoder** (~100 M params, 4 layers, 1024 hidden, GQA 8 / 2) auto-regressively predicts the remaining 7 acoustic codebooks of that same frame conditioned on the backbone's hidden state and the semantic token. The 8 codebooks are then fed to the **Mimi codec** ([Kyutai, see AUDIO_CODECS.md](AUDIO_CODECS.md) Mimi section), which decodes them to 24 kHz PCM in a single causal pass — yielding one frame = 1920 samples = 80 ms of audio. End-to-end first-audio latency is ~150–250 ms on a consumer GPU. Speaker identity, style, and prosody come from a **conversation history prefix**: a list of `Segment(speaker_id, text, audio)` objects that are tokenized (text via the Llama-3.2 BPE, audio via Mimi.encode) and concatenated as context before the prompt — there are no learned speaker embeddings.

This file covers the dual-transformer LM, the conversation-format prompt assembly, and the streaming generation loop. The Mimi codec internals (causal SEANet + bottleneck transformer + split-RVQ decoder) are in [AUDIO_CODECS.md](AUDIO_CODECS.md) Mimi section. The Llama-3.2 backbone follows the patterns in [DOTLLM_ARCHITECTURE.md](DOTLLM_ARCHITECTURE.md) (RoPE, GQA, RMSNorm, SwiGLU). General TTS context-prompt patterns appear in [TEXT_ENCODERS.md](TEXT_ENCODERS.md).

Sources: [SesameAILabs/csm](https://github.com/SesameAILabs/csm) (`models.py`, `generator.py`, `run_csm.py`), [HuggingFace sesame/csm-1b](https://huggingface.co/sesame/csm-1b), [Sesame research blog — Crossing the uncanny valley of voice](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice), [HF Transformers `CsmForConditionalGeneration`](https://huggingface.co/docs/transformers/model_doc/csm) (added in v4.52.1, May 20 2025), [torchtune llama3_2 builder](https://github.com/pytorch/torchtune), [kyutai-labs/moshi `loaders.get_mimi`](https://github.com/kyutai-labs/moshi).

## Detailed Findings

### Model Family Overview

Only one variant is publicly released to date:

| Model        | Params  | HF repo           | Release    | License    | Sample rate | Frame rate | Codebooks | Languages    |
|--------------|--------:|-------------------|------------|------------|------------:|-----------:|----------:|--------------|
| **CSM-1B**   | ~1.1 B  | `sesame/csm-1b`   | 2025-03-13 | Apache 2.0 | 24 kHz      | 12.5 Hz    | 8 (of 32) | English (primary; weak multilingual) |

Sesame's own production model (the "Maya / Miles" demo) is rumored to be a larger "medium" variant (~3 B backbone) — **not released**. The repo's `FLAVORS` dict in `models.py` is parameterized to support arbitrary backbone/decoder combinations, but `sesame/csm-1b` is the only checkpoint published.

Total **on-disk size** (`sesame/csm-1b`):

- `ckpt.pt` — 6.22 GB (FP32 torch pickle, legacy format)
- `model.safetensors` — 6.21 GB (FP32 safetensors)
- `transformers-00001-of-00002.safetensors` + `transformers-00002-of-00002.safetensors` — 4.94 GB + 2.19 GB = 7.13 GB (HF Transformers-native packaging; includes Mimi weights bundled)

Loading the model in **bfloat16** (the recommended inference dtype) cuts VRAM to ~2.2 GB for the LM (backbone + decoder) plus ~400 MB for Mimi.

### Architecture: Dual Transformer

The model in `models.py` is exactly two `torchtune.modules.transformer.TransformerDecoder` instances (the same building block as Llama-3.2) glued together by three thin linear layers and two embedding tables.

#### Backbone (`llama-1B` flavor)

From `models.py`:

```python
def llama3_2_1B() -> torchtune.modules.transformer.TransformerDecoder:
    return llama3_2.llama3_2(
        vocab_size=128_256,
        num_layers=16,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=2048,
        max_seq_len=2048,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
    )
```

This is **functionally identical to `meta-llama/Llama-3.2-1B`**:

- `embed_dim = 2048`, `num_layers = 16`, `num_heads = 32`, `num_kv_heads = 8` → **GQA 4:1**, `head_dim = 64`.
- `intermediate_dim = 8192` SwiGLU FFN.
- RMSNorm (`norm_eps = 1e-5`), pre-norm.
- RoPE with `rope_base = 500_000` and **Llama-3.2 frequency rescaling** (`scale_factor = 32` — extends context from 8 k → 128 k in the base Llama; here capped at 2048 by `max_seq_len`).
- `vocab_size = 128_256` (Llama-3 tokenizer + reserved special tokens).
- `max_seq_len = 2048` (hard cap — the entire conversation context + generation must fit).

The backbone's `tok_embeddings` and `output` layers are **replaced with `nn.Identity()`** by `_prepare_transformer`. Inputs come pre-embedded, outputs are read raw — the embedding tables and the codebook-0 head live on the outer `Model` class instead.

#### Decoder (`llama-100M` flavor)

```python
def llama3_2_100M() -> torchtune.modules.transformer.TransformerDecoder:
    return llama3_2.llama3_2(
        vocab_size=128_256,
        num_layers=4,
        num_heads=8,
        num_kv_heads=2,
        embed_dim=1024,
        max_seq_len=2048,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
    )
```

- `embed_dim = 1024`, `num_layers = 4`, `num_heads = 8`, `num_kv_heads = 2` → **GQA 4:1**, `head_dim = 128`.
- Same activation (SwiGLU), same normalization (RMSNorm), same RoPE (`rope_base = 500_000`, `scale_factor = 32`), same `vocab_size = 128_256` placeholder (also identity-stubbed and unused).
- `intermediate_dim = 8192` — note this is **8× hidden** (not the standard ~2.6× SwiGLU ratio); the FFN is unusually wide for a 4-layer model. This is intentional: the decoder only ever runs for at most **`audio_num_codebooks = 8`** steps per audio frame (very short sequences), so depth was traded for FFN capacity.

The decoder is **only ever invoked on sequences of length ≤ 8** — it runs once per audio frame, stepping codebook 1 → 7 (the loop in `generate_frame`). KV-cache is sized to `decoder_max_seq_len = config.audio_num_codebooks`, i.e. 8 entries per layer per head. Per-frame compute is essentially free compared to the backbone.

#### Outer `Model` glue

```python
@dataclass
class ModelArgs:
    backbone_flavor: str      # "llama-1B"
    decoder_flavor: str       # "llama-100M"
    text_vocab_size: int      # 128_256 (Llama-3.2 vocab)
    audio_vocab_size: int     # 2048   (Mimi codebook size)
    audio_num_codebooks: int  # 32 declared, only 8 used

class Model(nn.Module):
    def __init__(self, config):
        self.backbone, backbone_dim = _prepare_transformer(FLAVORS[config.backbone_flavor]())
        self.decoder, decoder_dim   = _prepare_transformer(FLAVORS[config.decoder_flavor]())

        self.text_embeddings  = nn.Embedding(config.text_vocab_size, backbone_dim)              # 128256 x 2048
        self.audio_embeddings = nn.Embedding(config.audio_vocab_size * config.audio_num_codebooks,
                                             backbone_dim)                                       # (2048*32) x 2048
        self.projection       = nn.Linear(backbone_dim, decoder_dim, bias=False)                 # 2048 -> 1024
        self.codebook0_head   = nn.Linear(backbone_dim, config.audio_vocab_size, bias=False)     # 2048 -> 2048
        self.audio_head       = nn.Parameter(torch.empty(config.audio_num_codebooks - 1,
                                                          decoder_dim, config.audio_vocab_size)) # (31, 1024, 2048)
```

Key facts:

- **`audio_num_codebooks` is 32 in config, but the runtime only uses the first 8**. The 32 slots match Mimi's full RVQ depth (`mimi.set_num_codebooks(32)` is called at load — see Tokenizers section); the extra heads and embeddings are *trained* but never sampled at inference. They are part of the checkpoint so the model can in principle be run at higher bitrate later.
- **Audio embedding is a single flat table of `vocab_size * num_codebooks = 65 536` rows.** Codebook `k` token `t` looks up row `k * 2048 + t`. This is how `_embed_audio(codebook, tokens)` works:

  ```python
  def _embed_audio(self, codebook, tokens):
      return self.audio_embeddings(tokens + codebook * self.config.audio_vocab_size)
  ```

  No per-codebook embedding matrices, no embedding-summation hack from MusicGen — every codebook lives in its own band of the same 65k-row table.

- **`codebook0_head`** is a normal linear head on the backbone's last hidden state (2048 → 2048) producing logits over Mimi semantic tokens.

- **`audio_head`** is a 3-D parameter of shape `(31, decoder_dim=1024, audio_vocab_size=2048)`. At step `i` (1 ≤ i ≤ 7 at inference) the decoder hidden state is projected via `audio_head[i-1]` to per-codebook logits. **There is no shared output head** — each acoustic codebook has its own 1024×2048 projection. This is *exactly* the "tied input embeddings vs untied output heads" choice you would expect from a residual VQ predictor.

#### Per-frame "wide" input vector

The input to the backbone is a sequence of frames of shape `(seq_len, audio_num_codebooks + 1) = (seq_len, 33)`. Each row is one 80 ms time step packed as:

```
[ ac_0, ac_1, ac_2, ..., ac_31, text_token ]
```

where `ac_k` is the codebook-`k` token (or 0 + masked-off) and `text_token` is the Llama-3 text token (or 0 + masked-off). At each row exactly one of "all 32 audio slots" or "the text slot" is active — the **mask tensor `tokens_mask`** of shape `(seq_len, 33)` selects which.

The embedding step `_embed_tokens`:

```python
def _embed_tokens(self, tokens):
    text_embeds = self.text_embeddings(tokens[:, :, -1]).unsqueeze(-2)   # (B, S, 1, D)
    audio_tokens = tokens[:, :, :-1] + (audio_vocab_size * torch.arange(audio_num_codebooks, device=...))
    audio_embeds = self.audio_embeddings(audio_tokens.view(-1)).reshape(B, S, audio_num_codebooks, D)
    return torch.cat([audio_embeds, text_embeds], dim=-2)                # (B, S, 33, D)
```

The result is `(B, S, 33, D)`, then the mask is applied and the **33 vectors are summed** to give the per-frame backbone input `h: (B, S, D)`:

```python
masked_embeds = embeds * tokens_mask.unsqueeze(-1)
h = masked_embeds.sum(dim=2)
```

This is the same "sum of K codebook embeddings" trick used in MusicGen, generalized to also include a text channel. Text-only rows contribute only the text embedding; audio-only rows contribute the sum of all 32 codebook embeddings.

### Mimi Audio Codec

CSM uses **Mimi (Kyutai)** as its waveform tokenizer. Full architecture is in [AUDIO_CODECS.md](AUDIO_CODECS.md) "Mimi (Kyutai, the Moshi codec)" section. Key facts relevant to CSM:

- **24 kHz** mono, **12.5 Hz** frame rate → **1920 samples per frame = 80 ms per frame**.
- **32 codebooks total at training**, **8 at inference for CSM** (Moshi uses the same split-8: 1 semantic + 7 acoustic).
- Codebook size **2048**, codebook dim **256**.
- **Split RVQ**: codebook 0 is a *standalone* VQ distilled from WavLM (semantic / phonetic content); codebooks 1–7 are a residual VQ on the acoustic detail.
- **Fully causal** SEANet + bottleneck-transformer encoder/decoder — supports streaming, ~80 ms algorithmic latency.
- Bitrate at 8 codebooks: 8 × 11 × 12.5 = **1.1 kbps**.
- Checkpoint: `kyutai/mimi/model.safetensors` (385 MB), loaded via `moshi.models.loaders.get_mimi(...)`. `generator.py` calls `mimi.set_num_codebooks(32)` at load (preserves the full quantizer in case the user wants more codebooks; CSM still only samples 8).

The split-VQ is *the* reason the dual-transformer makes sense: codebook 0 carries semantic / phonetic information (slow-varying, needs deep model and conversation context → backbone), while codebooks 1–7 carry residual acoustic detail (fast-varying, but conditionally near-independent once you know cb0 → small decoder is enough).

### Streaming Dual-Decoder Loop

Per Mimi frame (80 ms of audio), the model performs the following inside `Model.generate_frame`:

```python
@torch.inference_mode()
def generate_frame(self, tokens, tokens_mask, input_pos, temperature, topk):
    # 1) Embed and sum -> (B, S, D) frame embeddings
    embeds = self._embed_tokens(tokens)
    h = (embeds * tokens_mask.unsqueeze(-1)).sum(dim=2)

    # 2) Backbone forward (uses incremental KV cache for S=1 after the prompt)
    h = self.backbone(h, input_pos=input_pos, mask=curr_backbone_mask)
    last_h = h[:, -1, :]                                # (B, D=2048)

    # 3) Sample codebook 0 (semantic) from the backbone head
    c0_logits = self.codebook0_head(last_h)             # (B, 2048)
    c0_sample = sample_topk(c0_logits, topk, temperature)
    c0_embed  = self._embed_audio(0, c0_sample)         # (B, 1, D)

    # 4) Decoder loop: codebooks 1..7
    curr_h = torch.cat([last_h.unsqueeze(1), c0_embed], dim=1)   # (B, 2, D)
    curr_sample = c0_sample.clone()
    curr_pos = arange(0, curr_h.size(1))                # [0, 1]
    self.decoder.reset_caches()
    for i in range(1, self.config.audio_num_codebooks):  # = 8 at inference
        decoder_h = self.decoder(self.projection(curr_h), input_pos=curr_pos, mask=...)
        ci_logits = decoder_h[:, -1, :] @ self.audio_head[i - 1]   # (B, 2048)
        ci_sample = sample_topk(ci_logits, topk, temperature)
        ci_embed  = self._embed_audio(i, ci_sample)
        curr_h = ci_embed                               # decoder uses KV-cache, only feed new token
        curr_sample = torch.cat([curr_sample, ci_sample], dim=1)
        curr_pos = curr_pos[:, -1:] + 1
    return curr_sample                                  # (B, 8) one full Mimi frame
```

Important details:

- **Backbone runs once per audio frame** (consuming 1 new time step in incremental mode), producing one hidden state `last_h`.
- **Decoder runs 7 times per audio frame** (codebooks 1..7), each step consuming a single token in its own KV cache. The first decoder step gets a 2-token input `[backbone_h, c0_embed]`; subsequent steps get only the new `ci_embed` (KV-cache holds the rest). The decoder cache is **reset every frame** — there is no cross-frame state in the decoder, only intra-frame.
- **The decoder's input is `projection(curr_h)`**, projecting backbone-dim 2048 → decoder-dim 1024.
- **Stop condition**: when all 8 codebooks of the sampled frame are 0 — `if torch.all(sample == 0): break` in `generator.py`. CSM treats the all-zero frame as EOS.
- After collecting all sampled frames, `mimi.decode(stack(samples).permute(1, 2, 0))` reconstructs PCM. Permute changes `(T_frames, B, 8)` → `(B, 8, T_frames)` which is Mimi's expected layout.

**Per-frame latency budget on a consumer GPU (RTX 4090, bf16):**

| Step                          | Approx ms |
|-------------------------------|----------:|
| Backbone forward (1 token)    | ~6–10     |
| 7× decoder forward (1 tok ea) | ~2–4      |
| Mimi decode (1 frame)         | ~5–10     |
| **Per-frame total**           | **~15–25 ms** |

Since one frame = 80 ms of audio, **real-time factor (RTF) ≈ 0.2–0.3** — i.e. the model produces audio ~3–5× faster than wall-clock playback. This is the basis for low-latency streaming: you can start emitting audio as soon as the first frame (or a small handful) is ready.

### Conversational Context Format

Conversation context is the only mechanism for **speaker identity** and **style transfer** — there are no learned speaker embeddings, no style vectors. Each conversational turn is a `Segment`:

```python
@dataclass
class Segment:
    speaker: int          # integer speaker ID, used as a literal token "[0]", "[1]", ...
    text: str             # transcript of what this speaker said
    audio: torch.Tensor   # the actual PCM waveform of that turn at 24 kHz
```

`generator.py` builds the full prompt by *interleaving* text and audio tokens for every segment in order, then appending the target-text-only segment for which we want to generate audio:

```
[ text(speaker_0) ][ audio(speaker_0) ][ text(speaker_1) ][ audio(speaker_1) ] ... [ text(target_speaker) ] → GENERATE
```

#### Text tokenization (`_tokenize_text_segment`)

```python
text_tokens = self._text_tokenizer.encode(f"[{speaker}]{text}")
text_frame      = torch.zeros(len(text_tokens), 33).long()
text_frame_mask = torch.zeros(len(text_tokens), 33).bool()
text_frame[:, -1] = torch.tensor(text_tokens)
text_frame_mask[:, -1] = True
```

- **Speaker is literally a string prefix**: `"[0]Hello there"` is tokenized as-is by the Llama-3.2 BPE. No reserved token IDs are used for speakers — they fall through the BPE as ordinary characters. Practical limit: stick to `[0]`–`[9]` for the cleanest tokenization.
- BOS/EOS are added by a custom `TemplateProcessing` postprocessor: `"<|begin_of_text|>:0 $A:0 <|end_of_text|>:0"`. The pair-form is `<|begin_of_text|>:0 $A:0 <|end_of_text|>:0 <|begin_of_text|>:1 $B:1 <|end_of_text|>:1` (used when the tokenizer encodes two strings — not used in normal generation).
- Text occupies **only column 32 (the last column)** of the 33-wide frame. All 32 audio columns are masked off.

#### Audio tokenization (`_tokenize_audio`)

```python
audio_tokens = self._audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))[0]  # (32, T_frames)
eos_frame    = torch.zeros(audio_tokens.size(0), 1)
audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)                       # append EOS frame
audio_frame      = torch.zeros(audio_tokens.size(1), 33).long()
audio_frame_mask = torch.zeros(audio_tokens.size(1), 33).bool()
audio_frame[:, :-1] = audio_tokens.transpose(0, 1)
audio_frame_mask[:, :-1] = True
```

- Mimi encodes the waveform to 32 codebooks (`set_num_codebooks(32)` at load — **even though generation only samples 8**, the encoder produces all 32 so context audio carries more acoustic detail).
- A single trailing **all-zero frame** is appended as an end-of-segment marker.
- Audio occupies **columns 0..31** (the 32 audio codebook slots). Column 32 (text) is masked off.

#### Segment concatenation

```python
def _tokenize_segment(self, segment):
    text_tokens, text_masks = self._tokenize_text_segment(segment.text, segment.speaker)
    audio_tokens, audio_masks = self._tokenize_audio(segment.audio)
    return torch.cat([text_tokens, audio_tokens], dim=0), torch.cat([text_masks, audio_masks], dim=0)
```

Text rows come **first**, then audio rows. There is no interleaving within a segment.

#### Full prompt

```
prompt = concat( [ tokenize(seg) for seg in context ] + [ tokenize_text_segment(target_text, target_speaker) ] )
```

The model then generates frames one at a time, each new frame written to all 32 audio columns (columns 0..7 carry meaningful samples, 8..31 are 0 — only 8 are sampled, the rest are written as zeros for the embedding sum).

### Speaker Control

There are exactly two levers:

1. **Speaker ID prefix** (`[0]`, `[1]`, …) — a soft signal; the model learned to associate IDs with whatever speaker was labelled that way during training. Without reference audio, the same ID can produce different voices on different runs (the model is "varied voice" not "fixed voice", per the model card).

2. **Reference audio context** — the *real* speaker-cloning mechanism. The `run_csm.py` example loads two ~30-second reference clips (`prompts/conversational_a.wav`, `prompts/conversational_b.wav`), each with transcript, then prefixes every generation with both reference Segments. The model picks up timbre, prosody, accent, and speaking style from the audio prefix because the LM has full access to the Mimi semantic + acoustic tokens of that audio.

Practical guidance from the released example:

- Use **10–30 s of reference audio per speaker** for stable cloning.
- Include **accurate transcript** for the reference audio — the text alignment helps the LM learn the voice→text mapping for that speaker.
- Append **previously generated turns** to the context so subsequent generations are consistent with what the speaker already "said" in the conversation. The example loops:
  ```python
  context = prompt_segments + generated_segments
  audio = generator.generate(text=..., speaker=..., context=context, max_audio_length_ms=10_000)
  generated_segments.append(Segment(text=..., speaker=..., audio=audio))
  ```

There are no built-in voices in `sesame/csm-1b`. The repo ships only the two `prompts/conversational_*.wav` files as demo references.

### Text Tokenizer

CSM uses the **stock Llama-3.2 BPE** directly: `AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")` with a custom post-processor that wraps `<|begin_of_text|>` … `<|end_of_text|>` around each text segment. No CSM-specific tokens are added. Vocab size 128 256. Speaker IDs are not reserved tokens — they ride through as the literal characters `[`, `0`, `]`, etc.

This means the **shared text-embedding table** with Llama-3.2 1B (`128_256 × 2048` = 525 M parameters in the embedding alone — about half the model's parameter count) is loaded directly from the Llama checkpoint format and is *not* fine-tuned by CSM (or at least is initialized from it).

### Sampling

`sample_topk` in `models.py`:

```python
def sample_topk(logits, topk, temperature):
    logits = logits / temperature
    indices_to_remove = logits < torch.topk(logits, topk)[0][..., -1, None]
    scores = logits.masked_fill(indices_to_remove, -inf)
    scores = log_softmax(scores, dim=-1)
    probs  = softmax(scores, dim=-1)
    return multinomial(probs, num_samples=1)
```

Defaults from `Generator.generate`:

| Parameter           | Default     | Notes                                                            |
|---------------------|-------------|------------------------------------------------------------------|
| `temperature`       | `0.9`       | Standard TTS sampling temperature.                               |
| `topk`              | `50`        | Top-k truncation per-codebook.                                   |
| `max_audio_length_ms` | `90_000`  | Hard cap; converted to `max_audio_length_ms / 80` frames (=1125). |

**Sampling is applied independently to every codebook** of every frame (8 per frame). There is no nucleus / typical / min-p sampling, no repetition penalty, no CFG. Temperature and top-k are the only knobs exposed.

Notable: the same `(temperature, topk)` are reused for codebooks 1..7 (no per-codebook schedule). In practice the residual codebooks are very low-entropy after fixing codebook 0, so the effective sampling distribution is much sharper than the configured temperature suggests.

### Inference Pipeline Pseudocode (with shapes)

```
Inputs:
  context: List[Segment]           # speaker, text, audio
  text:    str                     # next utterance to speak
  speaker: int                     # ID for the next utterance
  max_audio_length_ms: float = 90_000
  temperature: float = 0.9
  topk: int = 50

Setup:
  text_tok = LlamaTokenizer("meta-llama/Llama-3.2-1B") + BOS/EOS template post-processor
  mimi     = MimiCodec(num_codebooks=32, sample_rate=24_000, frame_rate=12.5)
  model.setup_caches(batch_size=1)           # allocates backbone KV cache for max_seq_len=2048

Step 1: assemble prompt
  prompt_tokens = []   # list of (rows, 33) tensors
  prompt_masks  = []
  for seg in context:
    t_tokens = text_tok.encode("[" + str(seg.speaker) + "]" + seg.text)          # 1-D
    t_row    = zeros(len(t_tokens), 33);  t_row[:, 32]   = t_tokens
    t_mask   = zeros(len(t_tokens), 33);  t_mask[:, 32]  = True

    a_codes  = mimi.encode(seg.audio)                                            # (32, T_frames)
    a_codes  = concat([a_codes, zeros(32, 1)], dim=1)                            # append EOS frame
    a_row    = zeros(a_codes.shape[1], 33); a_row[:, 0:32]  = a_codes.T
    a_mask   = zeros(a_codes.shape[1], 33); a_mask[:, 0:32] = True

    prompt_tokens += [t_row, a_row]
    prompt_masks  += [t_mask, a_mask]

  # Target-text segment (no audio yet)
  t_tokens = text_tok.encode("[" + str(speaker) + "]" + text)
  t_row    = zeros(len(t_tokens), 33);  t_row[:, 32]   = t_tokens
  t_mask   = zeros(len(t_tokens), 33);  t_mask[:, 32]  = True
  prompt_tokens += [t_row];  prompt_masks += [t_mask]

  prompt_tokens = concat(prompt_tokens, dim=0).to(device)                        # (S_prompt, 33)
  prompt_masks  = concat(prompt_masks,  dim=0).to(device)                        # (S_prompt, 33)
  assert S_prompt < 2048 - (max_audio_length_ms / 80)                            # ~1125 frame budget

Step 2: prefill the backbone KV cache
  curr_tokens = prompt_tokens.unsqueeze(0)                                       # (1, S_prompt, 33)
  curr_masks  = prompt_masks .unsqueeze(0)                                       # (1, S_prompt, 33)
  curr_pos    = arange(0, S_prompt).unsqueeze(0)                                 # (1, S_prompt)

Step 3: autoregressive frame loop (incremental)
  max_frames = int(max_audio_length_ms / 80)
  samples = []                                                                   # list of (1, 8)
  for step in range(max_frames):
    frame = model.generate_frame(curr_tokens, curr_masks, curr_pos,
                                 temperature, topk)                              # (1, 8)
    if all(frame == 0): break                                                    # EOS
    samples.append(frame)
    # build next input row: the just-sampled audio frame (with 0s for cb 8..31 and text)
    next_tokens = concat([frame, zeros(1, 32 - 8 + 1)], dim=1).unsqueeze(1)      # (1, 1, 33)
    next_mask   = concat([ones(1, 8).bool(),
                          zeros(1, 32 - 8 + 1).bool()], dim=1).unsqueeze(1)      # (1, 1, 33)
    curr_tokens = next_tokens
    curr_masks  = next_mask
    curr_pos    = curr_pos[:, -1:] + 1                                           # advance position by 1

  # NB: the reference code as-shipped writes all 8 codes into columns 0..7 of the wide frame
  # and zeros in columns 8..31, but the mask only marks 1..len(sample) — see code for exact slicing.

Step 4: Mimi decode
  codes = stack(samples).permute(1, 2, 0)                                        # (1, 8, T_gen_frames)
  audio = mimi.decode(codes)                                                     # (1, 1, T_gen_frames * 1920)
  return audio                                                                   # 24 kHz PCM
```

**Streaming variant** (not in the reference repo but trivial given the structure): yield each frame immediately after `mimi.decode(frame)` — Mimi is causal, so single-frame decode is well-defined and produces exactly 1920 samples per call (with the codec keeping its own causal conv ring buffers across calls; see [AUDIO_CODECS.md](AUDIO_CODECS.md) Mimi streaming notes).

### HuggingFace Files (`sesame/csm-1b`)

| File                                              | Size       | Purpose                                                                            |
|---------------------------------------------------|-----------:|------------------------------------------------------------------------------------|
| `ckpt.pt`                                         | 6.22 GB    | Legacy torch pickle of the full model state dict (FP32). Used by the original `generator.py`. |
| `model.safetensors`                               | 6.21 GB    | Same weights in safetensors format. **Use this — no pickle exec risk.**            |
| `transformers-00001-of-00002.safetensors`         | 4.94 GB    | HF Transformers-native sharded checkpoint (added v4.52.1 May 2025); includes Mimi bundled. |
| `transformers-00002-of-00002.safetensors`         | 2.19 GB    | Shard 2 of 2.                                                                      |
| `transformers.safetensors.index.json`             | 59.7 kB    | Sharding index mapping tensor names → shard files.                                 |
| `config.json`                                     | 3.28 kB    | Model config: architectures, audio_num_codebooks=32, audio_vocab_size=2048, backbone_flavor="llama-1B", decoder_flavor="llama-100M", text_vocab_size=128256, torch_dtype. |
| `generation_config.json`                          | 264 B      | Default `temperature`, `top_k`, `max_new_tokens` for HF generate.                  |
| `tokenizer.json`                                  | 17.2 MB    | Llama-3.2 BPE tokenizer (merges + vocab + pre-tokenizer + post-processor template). |
| `tokenizer_config.json`                           | 50.6 kB    | Tokenizer metadata (special tokens, chat template hooks).                          |
| `special_tokens_map.json`                         | 449 B      | BOS / EOS / PAD mappings.                                                          |
| `chat_template.jinja`                             | 2 kB       | Jinja template for `[speaker]text` formatting (used by `processor.apply_chat_template`). |
| `preprocessor_config.json`                        | 271 B      | Mimi feature-extractor metadata (sample rate, target length).                      |
| `README.md`                                       | 12.1 kB    | Model card.                                                                        |
| `.gitattributes`                                  | 1.95 kB    | Git LFS pointers.                                                                  |
| `prompts/conversational_a.wav`                    | ~3 MB      | Demo reference audio for speaker 0.                                                |
| `prompts/conversational_b.wav`                    | ~3 MB      | Demo reference audio for speaker 1.                                                |

**Loading note**: the reference `generator.py` calls `Model.from_pretrained("sesame/csm-1b")` (PyTorchModelHubMixin), which loads `model.safetensors` (or `ckpt.pt` fallback). It does **not** include Mimi — Mimi is downloaded *separately* from `kyutai/mimi` via `loaders.get_mimi(hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME))`. In contrast, the HF Transformers `transformers-*.safetensors` shards **bundle Mimi** under a `codec_model.*` prefix.

### Memory and Performance

**Parameter breakdown (CSM-1B, FP16/BF16):**

| Component                  | Params     | bf16 size |
|----------------------------|-----------:|----------:|
| `text_embeddings` (128 256 × 2048) | 263 M | 525 MB |
| `audio_embeddings` (65 536 × 2048) | 134 M | 268 MB |
| Backbone Llama-1B (16 × Llama block, no embed/output) | ~750 M | 1.5 GB |
| Decoder Llama-100M (4 × Llama block, no embed/output) | ~100 M | 200 MB |
| `projection` (2048 × 1024) | 2.1 M | 4.2 MB |
| `codebook0_head` (2048 × 2048) | 4.2 M | 8.4 MB |
| `audio_head` (31 × 1024 × 2048) | 65 M | 130 MB |
| **Total LM**               | **~1.32 B** | **~2.6 GB** |
| Mimi codec (separate)      | ~98 M | ~200 MB |
| **Total VRAM (weights only)** | | **~2.8 GB** |

The released FP32 `model.safetensors` is ~6.2 GB simply because it stores everything at fp32. Inference should run in bf16 throughout (Llama-3.2 was trained in bf16; the original Sesame `load_csm_1b` calls `model.to(dtype=torch.bfloat16)`).

**KV-cache memory (bf16, max_seq_len=2048, batch=1):**

- Backbone: 16 layers × 2 (K,V) × 8 KV heads × 64 head_dim × 2048 seq × 2 bytes = **67 MB**.
- Decoder: 4 layers × 2 × 2 KV heads × 128 head_dim × 8 seq × 2 bytes = **32 kB** (negligible; reset every frame anyway).

**Latency (RTX 4090, bf16, batch=1, ~30 s of conversation history):**

| Stage                                       | Time      |
|---------------------------------------------|----------:|
| Prefill backbone over ~500-token prompt     | ~80–150 ms |
| First generated frame (backbone + 7 decoder + Mimi decode) | ~20–30 ms |
| Steady-state per-frame                      | ~15–25 ms |
| **First-audio latency** (prompt-end → first PCM byte) | **~100–200 ms** |
| **End-to-end "user finished speaking" → first audio chunk plays** | **~150–250 ms** |
| RTF                                         | **~0.2–0.3** (3–5× faster than playback) |

On lower-end consumer GPUs (RTX 3060 12 GB, bf16): first-audio ~250–400 ms, RTF ~0.4–0.6 — still real-time but margin is smaller. On Apple Silicon (MPS): MPS does not support some bf16 ops cleanly; community forks report ~500 ms first-audio at fp16 on M2 Max.

Sesame's blog claims their internal (larger, unreleased) model achieves ~150 ms p50 user-stop-to-audio latency in their voice demo, suggesting CSM-1B at ~150–250 ms is in the same ballpark.

### C# Implementation Notes (SharpInference)

#### Component reuse map

| CSM Component                          | SharpInference reuse                                                                  |
|----------------------------------------|---------------------------------------------------------------------------------------|
| Backbone (Llama-3.2 1B with GQA, RoPE, RMSNorm, SwiGLU) | **Direct dotLLM reuse.** This is bit-identical Llama-3.2-1B except `tok_embeddings`/`output` are stubbed. Use the dotLLM `LlamaBlock`, KV cache, attention, RoPE with `scale_factor=32`. |
| Decoder (4-layer Llama-100M)           | **Same dotLLM blocks, smaller config.** Note `intermediate_dim=8192` (8× hidden, not 2.6×) — make sure the FFN dim is read from config, not derived. KV cache size = 8 entries per layer per head. |
| Mimi codec                             | **New code path.** See [AUDIO_CODECS.md](AUDIO_CODECS.md) Mimi section. Critical: 12.5 Hz frame rate (1920-sample hop), causal convs with ring-buffer streaming, bottleneck transformer with RoPE+GELU, split-RVQ with 1 semantic + 7 acoustic codebooks. |
| Llama-3.2 BPE tokenizer                | **Reuse dotLLM tokenizer** (`meta-llama/Llama-3.2-1B`). Add a TemplateProcessing post-processor equivalent that wraps `<|begin_of_text|>` … `<|end_of_text|>`. |
| Top-k sampling                         | Already in dotLLM. No nucleus, no repetition penalty needed. |
| Watermarker (SilentCipher)             | **Skip for v1.** `watermarking.py` references `CSM_1B_GH_WATERMARK` and SilentCipher. It is optional (not load-bearing for audio quality) and the Apache 2.0 license does not require it. Document the omission. |

#### Critical correctness items

1. **Wide-frame embedding sum.** The per-frame input is the *sum* of up to 33 embedding lookups, gated by a mask. Implement this as a fused kernel if possible: `sum_k(mask[k] * E[token[k] + k*vocab])`. For text rows, k=32 (text), for audio rows, k=0..31. Validate against PyTorch reference within 1e-5 RMS.

2. **Audio embedding flat layout.** `audio_embeddings` is a single `(65 536, 2048)` table — *not* 32 separate tables. Token lookup index is `token + codebook * 2048`. A bug here is silent (will produce garbage tokens but compile fine).

3. **Decoder KV cache reset per frame.** `self.decoder.reset_caches()` is called at the top of every `generate_frame`. The decoder has **no cross-frame state**. Easy to mess up if porting naively.

4. **Decoder input on step i=1.** First decoder call gets a 2-token input `[projection(backbone_last_h), projection(c0_embed)]` (positions 0 and 1). Subsequent calls feed only the new `projection(ci_embed)` at position i. The KV cache accumulates within the frame.

5. **`audio_head` is a 3-D parameter, not a Linear layer.** Shape `(31, 1024, 2048)`. The forward is `decoder_h @ audio_head[i-1]`. No bias. Store as a single `(31, 1024, 2048)` tensor and index it.

6. **`codebook0_head` has no bias.** Same for `projection`. Easy to introduce a spurious bias when wrapping in a generic Linear class.

7. **EOS condition.** All 8 codebook tokens of a frame being 0 means stop. Do **not** treat this as a valid audio frame — discard and end generation.

8. **Speaker tokenization.** `f"[{speaker}]{text}"` — verify your Llama BPE tokenizes `[0]` the same way as the Python reference (`[`, `0`, `]` typically becomes 3-4 BPE pieces, not a single token). A diff here changes speaker behavior subtly.

9. **Mimi `set_num_codebooks(32)` at load.** Context audio is encoded at 32 codebooks even though generation samples 8. This is intentional — the context carries more information about the reference voice than the generation can express. If you encode context at 8 codebooks (to save memory), expect somewhat degraded voice cloning quality. Match the reference unless you have a reason not to.

#### Streaming pipeline shape (target SharpInference API)

```csharp
public interface ISesameCsmGenerator
{
    // Non-streaming: returns full PCM after generation completes.
    Task<AudioBuffer> GenerateAsync(
        string text,
        int speaker,
        IReadOnlyList<Segment> context,
        SesameCsmSamplingParams samplingParams,
        CancellationToken ct);

    // Streaming: yields 80 ms PCM chunks as they are produced. THIS is the showcase API.
    IAsyncEnumerable<AudioChunk> GenerateStreamingAsync(
        string text,
        int speaker,
        IReadOnlyList<Segment> context,
        SesameCsmSamplingParams samplingParams,
        CancellationToken ct);
}
```

Implementation notes for the streaming path:

- **Frame loop must be allocation-free.** Use pooled `NativeMemory.AlignedAlloc` buffers for the 33-wide frame, the codebook samples, and the Mimi output PCM. Per-frame allocations show up as audible jitter.
- **Decoder KV cache should be a fixed pre-allocated arena** (size = 4 layers × 2 × (2 heads × 128 dim × 8 seq) × sizeof(bf16) = 32 kB total). Reset by zeroing the position counter, not by reallocating.
- **Backbone KV cache is the big one** (~67 MB at 2048 seq). Pre-allocate once at model load; reuse across generate calls (with reset between turns).
- **Mimi decode is per-frame.** Implement causal streaming Mimi decode with per-layer ring buffers for the causal Conv1d transposes (see [AUDIO_CODECS.md](AUDIO_CODECS.md) "Causal Conv1d with KV-cache" notes). Each `mimi.decode_frame(codes_8)` call must return exactly 1920 samples (80 ms at 24 kHz) and update internal state.
- **Yield from the IAsyncEnumerable on a background CUDA stream**, with the consumer's `await foreach` running on the playback thread. The natural producer/consumer split is: GPU thread produces (`generate_frame` + `mimi.decode_frame`), CPU thread enqueues to the audio output device.
- **Prefill optimization.** For long context (~30 s = 375 frames + ~200 text tokens ≈ 600 tokens), prefill takes 80–150 ms. To hit <200 ms first-audio, **start playback after the first generated frame (80 ms)** — do not wait for any buffering. The first frame is ready 100–200 ms after prefill-end, and at RTF 0.2 the buffer immediately fills.
- **Cancellation must be fine-grained.** A user-interrupted dialogue cancels mid-generation; cancel between frames (~20 ms granularity), drop any unstreamed PCM, and reset both KV caches.

#### Suggested implementation order

1. **Mimi decoder + encoder** in `SharpInference.Audio.Codecs.Mimi` — both directions are needed (encoder for context audio, decoder for generation output). Validate against the HF `MimiModel` port (Python) within 1e-3 RMS on a fixed test waveform. See [AUDIO_CODECS.md](AUDIO_CODECS.md) Mimi section for the full op list.
2. **CSM `Model` class** in `SharpInference.Audio.Tts.Sesame` — wraps two dotLLM-Llama instances + the embedding tables + the three projection/head parameters. Implement the wide-frame embedding sum and the `generate_frame` loop. Validate `generate_frame` output (the 8 sampled codebooks) is plausible (cb0 distribution matches Mimi-semantic distribution for known audio).
3. **Generator + tokenization** in `SharpInference.Audio.Tts.Sesame.Generator` — port `_tokenize_text_segment`, `_tokenize_audio`, `_tokenize_segment`, the conversation-context loop. Reuse the dotLLM Llama-3.2 tokenizer.
4. **Non-streaming `GenerateAsync`** — batched frame loop, single Mimi decode at the end. End-to-end validate output PCM against the Python reference: same prompt, same seed → audibly identical waveform (perfect bit-match is unlikely due to sampling).
5. **Streaming `GenerateStreamingAsync`** — refactor the frame loop to emit per-frame. Per-frame Mimi decode with causal ring buffers. Build the `IAsyncEnumerable<AudioChunk>` plumbing and the producer/consumer split. **This is the showcase real-time path for SharpInference.**
6. **Performance pass** — fuse the wide-frame embedding sum kernel, ensure backbone KV cache writes are coalesced, profile single-frame latency target ≤ 25 ms on RTX 4090 bf16.

#### What is *not* needed

- No CFG, no nucleus / typical / min-p sampling, no repetition penalty.
- No diffusion, no vocoder beyond Mimi.
- No text encoder, no T5, no PLBERT — text goes directly into the Llama-3.2 BPE.
- No phonemizer / G2P — text is consumed as raw characters by the BPE.
- No speaker embedding table.
- No SilentCipher watermarker for v1 (it's optional and the model still works fine without it; document the omission).

## Key Numbers / Constants

| Name                              | Value          | Source                                      |
|-----------------------------------|---------------:|---------------------------------------------|
| Sample rate                       | 24 000 Hz      | Mimi codec                                  |
| Frame rate                        | 12.5 Hz        | Mimi codec                                  |
| Samples per frame                 | 1920           | 24000 / 12.5                                |
| Frame duration                    | 80 ms          | 1 / 12.5                                    |
| Backbone hidden dim               | 2048           | `embed_dim` in `llama3_2_1B`                |
| Backbone layers                   | 16             | `num_layers` in `llama3_2_1B`               |
| Backbone heads / KV heads         | 32 / 8         | `num_heads`, `num_kv_heads`                 |
| Backbone head dim                 | 64             | 2048 / 32                                   |
| Backbone FFN dim                  | 8192           | `intermediate_dim`                          |
| Backbone RoPE base / scale_factor | 500 000 / 32   | Llama-3.2 long-context scaling              |
| Decoder hidden dim                | 1024           | `embed_dim` in `llama3_2_100M`              |
| Decoder layers                    | 4              | `num_layers`                                |
| Decoder heads / KV heads          | 8 / 2          |                                             |
| Decoder head dim                  | 128            | 1024 / 8                                    |
| Decoder FFN dim                   | 8192           | `intermediate_dim` (8× hidden — unusually wide for a 4-layer model) |
| Max sequence length (backbone)    | 2048           | `max_seq_len`; hard cap for context+gen     |
| Decoder max seq len               | 8              | `audio_num_codebooks`; reset every frame    |
| `audio_num_codebooks` (config)    | 32             | All 32 trained; only first 8 sampled        |
| `audio_num_codebooks` (inference) | 8              | 1 semantic + 7 acoustic                     |
| `audio_vocab_size`                | 2048           | Mimi codebook size                          |
| `text_vocab_size`                 | 128 256        | Llama-3.2 vocab                             |
| Default temperature               | 0.9            | `Generator.generate` default                |
| Default top-k                     | 50             | `Generator.generate` default                |
| Default max_audio_length_ms       | 90 000         | = 1125 frames                               |
| Total LM params (weights only)    | ~1.32 B        | embeddings (397 M) + backbone (~750 M) + decoder (~100 M) + heads (~70 M) |
| LM VRAM (bf16, weights only)      | ~2.6 GB        |                                             |
| Mimi VRAM                         | ~200 MB        |                                             |
| Backbone KV cache (bf16, 2048 seq)| 67 MB          |                                             |
| Per-frame latency (RTX 4090 bf16) | ~15–25 ms      | RTF ≈ 0.2–0.3                              |
| First-audio p50 (RTX 4090)        | ~150–250 ms    | Prefill + 1 frame                           |

## Data Layouts / Formats

### Wide frame tensor

`tokens` shape `(B, S, 33)`, dtype int64:

```
column index: 0   1   2   ...   31      32
content:      cb0 cb1 cb2 ...   cb31    text_token
```

`tokens_mask` shape `(B, S, 33)`, dtype bool. Exactly one of the two groups is set per row at construction:

- **Text row**: `mask[s, 32] = True`, `mask[s, 0:32] = False`.
- **Audio row**: `mask[s, 0:32] = True`, `mask[s, 32] = False`.

After embedding, all 33 vectors are masked then summed → `(B, S, D=2048)`.

### Backbone input position

`input_pos` shape `(B, S)`, dtype int64. Continuous 0..S-1 for prefill; advanced by 1 per generated frame in incremental mode (just `curr_pos[:, -1:] + 1`).

### Codebook sample tensor

Output of `generate_frame`: shape `(B, 8)` int64. Concatenated to form `(B, 8, T_gen)` after permute, then passed to `mimi.decode`.

### Mimi codes tensor (encoder output / decoder input)

Shape `(B, n_codebooks, T_frames)`:

- Encoder: `mimi.encode(wav).shape == (B, 32, T_frames)` (at `set_num_codebooks(32)`).
- Decoder: accepts `(B, n_q, T_frames)` for any `n_q ≤ 32`; CSM generation passes `(B, 8, T_gen_frames)`.

### Audio output tensor

`mimi.decode(codes)` returns `(B, 1, T_frames * 1920)` float32 in `[-1, 1]`. The reference `Generator.generate` then runs SilentCipher watermarking and a sample-rate-identity resample (the resample appears to be a no-op safety net for cases where the watermarker changes sample rate). For SharpInference v1, skip both — return the raw PCM.

### Llama tokenizer

Standard Llama-3.2 BPE (`tokenizer.json` 17.2 MB) plus a post-processor that wraps each input with BOS/EOS. Reuse dotLLM's Llama-3.2 tokenizer; just set the post-processor template.

## Algorithm Steps

### Per-frame generation (the inner loop)

```
inputs:
  curr_tokens : (B, s, 33)  int64    # s=1 in steady state, s=S_prompt on first call
  curr_mask   : (B, s, 33)  bool
  curr_pos    : (B, s)      int64
  temperature : float
  topk        : int

1.  embeds        = embed_tokens(curr_tokens)                       # (B, s, 33, 2048)
2.  h_in          = sum(embeds * curr_mask.unsqueeze(-1), dim=2)    # (B, s, 2048)
3.  h             = backbone(h_in, input_pos=curr_pos, mask=cm)     # (B, s, 2048)
4.  last_h        = h[:, -1, :]                                      # (B, 2048)
5.  c0_logits     = codebook0_head(last_h)                           # (B, 2048)
6.  c0_sample     = sample_topk(c0_logits, topk, temperature)        # (B, 1)
7.  c0_embed      = audio_embeddings[c0_sample + 0*2048]             # (B, 1, 2048)
8.  curr_h        = concat([last_h.unsqueeze(1), c0_embed], dim=1)   # (B, 2, 2048)
9.  curr_sample   = c0_sample                                        # (B, 1)
10. decoder.reset_caches()
11. dec_pos       = [0, 1]                                           # (B, 2)
12. for i in 1..7:
13.     dec_in        = projection(curr_h)                           # (B, n_step, 1024)
14.     dec_out       = decoder(dec_in, input_pos=dec_pos, mask=dm)  # (B, n_step, 1024)
15.     ci_logits     = dec_out[:, -1, :] @ audio_head[i-1]          # (B, 2048)
16.     ci_sample     = sample_topk(ci_logits, topk, temperature)    # (B, 1)
17.     ci_embed      = audio_embeddings[ci_sample + i*2048]         # (B, 1, 2048)
18.     curr_h        = ci_embed                                     # (B, 1, 2048) — next step feeds 1 token
19.     curr_sample   = concat([curr_sample, ci_sample], dim=1)      # (B, i+1)
20.     dec_pos       = dec_pos[:, -1:] + 1
21. return curr_sample                                               # (B, 8)
```

### Outer generation loop

```
max_frames = max_audio_length_ms / 80
prefill curr_tokens = full prompt, curr_pos = arange(0, S_prompt)
samples = []
for step in 0..max_frames-1:
    frame = generate_frame(curr_tokens, curr_mask, curr_pos, T, k)
    if all(frame == 0): break                  # EOS
    samples.append(frame)
    curr_tokens = build_next_frame_row(frame)  # (B, 1, 33) with frame in cols 0..7, zeros elsewhere
    curr_mask   = build_next_frame_mask()      # (B, 1, 33) with True in cols 0..7
    curr_pos    = curr_pos[:, -1:] + 1
codes = stack(samples).permute(1, 2, 0)        # (B, 8, T_gen)
pcm   = mimi.decode(codes)                     # (B, 1, T_gen * 1920)
```

### Streaming variant (target SharpInference)

```
prefill as above
mimi.reset_streaming_state()
async for step in 0..max_frames-1:
    frame = generate_frame(...)
    if all(frame == 0): break
    pcm_chunk = mimi.decode_one_frame(frame)   # (1920,) float32 — uses causal ring buffers
    yield AudioChunk(pcm_chunk, sample_rate=24000)
    advance curr_tokens, curr_mask, curr_pos
```

### Sampling (top-k + temperature)

```
logits ← logits / temperature
threshold ← topk(logits, k=topk)[-1]           # k-th largest value
logits[logits < threshold] ← -inf
probs ← softmax(logits)
token ← multinomial(probs, num_samples=1)
```

Applied independently per codebook of every sampled frame. **No shared random state across codebooks** in the Python reference — each `sample_topk` call has its own `multinomial` draw. Match this in C# or expect subtle distribution differences.

## Reference Implementations

| Implementation                                    | Repo / path                                                                 | Notes                                                                              |
|---------------------------------------------------|-----------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| Original Sesame reference                         | [SesameAILabs/csm](https://github.com/SesameAILabs/csm)                     | `models.py`, `generator.py`, `run_csm.py`, `watermarking.py`. Uses torchtune Llama. |
| HuggingFace Transformers port                     | `transformers.models.csm.CsmForConditionalGeneration` (v4.52.1+, 2025-05-20) | Native HF API. Bundles Mimi. Supports static cache + CUDA graphs. Best reference for HF-format checkpoint shape. |
| Mimi codec (Kyutai)                               | [kyutai-labs/moshi](https://github.com/kyutai-labs/moshi) `moshi/models/loaders.py`, `moshi/modules/seanet.py`, `moshi/quantization/{vq,core_vq}.py` | The codec implementation. Use as reference for the Mimi C# port. |
| Llama-3.2 builder used by Sesame                  | [pytorch/torchtune](https://github.com/pytorch/torchtune) `torchtune/models/llama3_2/` | The exact `llama3_2.llama3_2(...)` factory called by `models.py`. Verifies hyperparameters match dotLLM. |

## Differences Between Implementations

- **torchtune Llama vs HF Llama vs dotLLM Llama.** All three implement identical math (RoPE + GQA + RMSNorm + SwiGLU) but with different code paths and tensor naming. The CSM checkpoint is **torchtune-named** (`backbone.layers.0.attn.q_proj.weight`, etc.). The HF-Transformers `transformers-*.safetensors` rename to the HF convention (`model.layers.0.self_attn.q_proj.weight`). When loading from safetensors in C#, support both naming schemes or document which one we target.

- **Mimi `num_codebooks` at load.** The reference uses 32 (encodes context fully); HF Transformers CSM defaults to 8 (only what's needed for generation). The 8-codebook path is faster and saves memory, but **context audio loses ~75% of its acoustic information**, weakening voice cloning. We should default to 32 to match the reference behavior.

- **Watermarking.** Reference applies SilentCipher watermarking on every generation. HF version does not. We follow HF — no watermark in v1.

- **Resampling step at end of generate().** Reference does `torchaudio.functional.resample(audio, orig_freq=wm_sample_rate, new_freq=self.sample_rate)` after watermarking. Without the watermarker, `wm_sample_rate == self.sample_rate` so this is a no-op. Skip it.

## Open Questions

1. **Does CSM-1B's audio_head[i-1] for i=8..31 contain meaningful weights?** They are loaded but never used at inference. If we want to support 32-codebook output (~4× higher bitrate, audible quality bump), we need to validate they are not garbage. Worth a numerical test (sample at 32 codebooks, compare audio quality with 8-codebook generation).

2. **Is there a Sesame "medium" or "large" model planned for release?** The blog mentions tiny/medium/large variants during research; only 1B is released. Track future HF releases; the architecture should generalize trivially (just larger `embed_dim` / `num_layers` in `FLAVORS`).

3. **CUDA-graph viability for `generate_frame`.** HF claims static-cache + CUDA-graph support. The inner decoder loop (7 iterations) has fixed shape per iteration, which is graph-friendly. We should target capturing the full per-frame path as a single graph for the steady-state loop — could halve per-frame latency to ~10 ms.

4. **Exact Mimi setup_streaming semantics.** Confirm whether per-frame `mimi.decode` calls (one frame at a time, with internal causal-conv ring buffers) produce **byte-identical PCM** to a one-shot `mimi.decode` on the same stacked codes. The Mimi paper claims yes (full causality); validate against the Kyutai reference before committing to the streaming code path.

5. **bf16 vs fp16 numerical behavior.** Sesame uses bf16. Some C# CUDA paths may default to fp16. For RoPE and softmax especially, fp16 vs bf16 differences can change top-k sampling. Match bf16 throughout to be safe.

6. **End-of-segment marker correctness.** The reference appends a single all-zero frame after every context audio segment in `_tokenize_audio`. We should verify this matches the training-time format; an off-by-one here would silently degrade context conditioning.

## Implementation Notes for SharpInference

### What SharpInference already has (or will, from dotLLM)

- Llama-3.2 transformer block (RoPE + GQA + RMSNorm + SwiGLU) — both backbone (16 layers) and decoder (4 layers) use this directly.
- Llama-3 BPE tokenizer.
- KV-cache infrastructure (incremental decode).
- Top-k sampling.
- Safetensors loader.
- bf16 PTX kernels for matmul, softmax, RMSNorm, RoPE.

### What is new and must be built

1. **Mimi codec** (`SharpInference.Audio.Codecs.Mimi`):
   - Causal SEANet encoder + decoder with streaming ring buffers (see [AUDIO_CODECS.md](AUDIO_CODECS.md)).
   - Bottleneck Transformer (RoPE + GELU, 8 layers × 8 heads × 512 dim, causal with 250-frame finite context).
   - Split-RVQ: standalone semantic VQ + 7-step residual VQ on top.
   - Both `encode(wav) → codes[B, 32, T]` and `decode(codes) → wav[B, 1, T*1920]`.
   - **Streaming `decode_one_frame(codes[B, 8]) → pcm[B, 1920]`** — this is the critical path for low-latency.

2. **CSM `Model`** (`SharpInference.Audio.Tts.Sesame.SesameCsmModel`):
   - Two `LlamaTransformer` instances (1B backbone, 100M decoder).
   - Three parameters / heads: `projection (2048→1024)`, `codebook0_head (2048→2048)`, `audio_head (31, 1024, 2048)`.
   - Two embedding tables: `text_embeddings (128256, 2048)`, `audio_embeddings (65536, 2048)`.
   - `_embed_tokens(wide_frame) → (B, S, 33, D)`: 33 lookups per row.
   - `generate_frame(...)` implementing the dual-loop above. Decoder KV cache reset per frame.

3. **CSM `Generator`** (`SharpInference.Audio.Tts.Sesame.SesameCsmGenerator`):
   - Conversation-context tokenizer (text + Mimi encode).
   - Outer frame loop with EOS detection.
   - Both `GenerateAsync` (non-streaming) and `GenerateStreamingAsync` (showcase path).

### Suggested implementation order

1. **Mimi codec** (encoder + decoder + streaming decoder). Validate against `transformers.MimiModel` to ≤1e-3 RMS.
2. **CSM Model class** with greedy (T=0) sampling. Validate `generate_frame` on a fixed prompt + seed against the Python reference at the codebook level (8 ints per frame should match).
3. **Non-streaming Generator**. Validate end-to-end PCM is perceptually equivalent to the Python reference (no bit-match expected due to multinomial sampling).
4. **Streaming Generator** with `IAsyncEnumerable<AudioChunk>`. Validate per-frame decode produces same PCM as one-shot decode on the same codes.
5. **Performance pass**: fuse wide-frame embed-sum kernel; pre-allocate all per-frame buffers; aim for ≤25 ms per-frame on RTX 4090 bf16.
6. **(Optional) CUDA-graph capture** of the steady-state inner loop for sub-15 ms per-frame.

### Performance targets

| Metric                                            | Target           | Stretch          |
|---------------------------------------------------|------------------|------------------|
| First-audio latency (RTX 4090 bf16, ~30 s ctx)    | ≤ 250 ms         | ≤ 150 ms         |
| Steady-state per-frame                            | ≤ 25 ms          | ≤ 15 ms          |
| Steady-state RTF (lower is better)                | ≤ 0.3            | ≤ 0.2            |
| VRAM (model + KV cache, batch=1)                  | ≤ 3.5 GB         | ≤ 3.0 GB         |
| Numerical agreement with Python ref (codebook 0)  | ≥ 95% top-1 match at T=0 | 100% bit-match at T=0 with same RNG seed |

This is the showcase real-time TTS pipeline for SharpInference — getting the streaming path right is more important than matching every last ms of single-frame throughput.
