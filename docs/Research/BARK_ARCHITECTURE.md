# Bark (Suno) — Architecture Research Notes

> Status: Complete | Last Updated: 2026-05-17 | Needed Before: SharpInference.Audio (Bark pipeline)

## Summary

Bark is a text-prompted generative audio model released by Suno in April 2023. Unlike conventional TTS systems, Bark produces not just speech but also music, ambient noise, sound effects, laughter, sighs, gasps, and other non-verbal vocalizations from a single text prompt. The pipeline is a three-stage cascade of GPT-2-style Transformers followed by an EnCodec 24 kHz decoder:

1. **Semantic Transformer** — causal LM. Takes BERT-tokenized text (offset into a shared 129,600-entry vocab) plus an optional speaker-prompt prefix and predicts a stream of semantic tokens at 49.9 Hz from a custom 10,000-entry HuBERT-derived semantic codebook.
2. **Coarse Transformer** — causal LM. Takes semantic tokens (offset) plus optional history and predicts the first two EnCodec acoustic codebooks at 75 Hz, interleaved as a single flat sequence (book 0, book 1, book 0, book 1, ...).
3. **Fine Transformer** — non-causal masked LM. Takes the 2 coarse codebooks and iteratively fills in EnCodec codebooks 2..7 in six refinement passes (codebook 0 is given by the coarse output and acts as input embedding base; the architecture supports `n_codes_total=8` with `n_codes_given=1`, leaving 7 trainable output heads).
4. **EnCodec 24 kHz, 8-codebook decoder** — converts the 8 acoustic codebook streams back to 24 kHz mono PCM.

Two variants ship: **Bark** (~1.0B params per stage HF-counted, ~900M trainable per stage) using hidden=1024 / 24 layers / 16 heads, and **Bark-Small** (~80M per stage trainable, ~300M per HF param count) using hidden=768 / 12 layers / 12 heads (a vanilla GPT-2-small footprint). Text is tokenized with `bert-base-multilingual-cased` (~120k WordPiece vocab) and shifted by `TEXT_ENCODING_OFFSET=10048` to live in the upper region of the 129,600-entry semantic input vocab. Voice control is via pre-extracted "speaker prompts" — triples of `(semantic, coarse, fine)` token streams from ~10s reference clips, prepended as in-context history at each of the three stages. ~100+ official voice prompts ship for 13 languages (English, German, Spanish, French, Hindi, Italian, Japanese, Korean, Polish, Portuguese, Russian, Turkish, Chinese).

The EnCodec component is covered in detail in [AUDIO_CODECS.md](AUDIO_CODECS.md) under "EnCodec (Meta / Défossez 2022)". The streaming considerations are in [STREAMING_AUDIO_INFERENCE.md](STREAMING_AUDIO_INFERENCE.md) under TTS section 3.1. The Kokoro-style TTS pipeline is in [KOKORO_ARCHITECTURE.md](KOKORO_ARCHITECTURE.md). For phonemizer-free TTS like Bark there is no G2P step.

Sources: [suno-ai/bark](https://github.com/suno-ai/bark), [suno/bark](https://huggingface.co/suno/bark), [suno/bark-small](https://huggingface.co/suno/bark-small), [HF transformers Bark](https://github.com/huggingface/transformers/tree/main/src/transformers/models/bark).

## Detailed Findings

### Variants and File Inventory

Two official variants, otherwise architecturally identical (same vocab sizes, same context length, same EnCodec, same generation defaults):

| Variant | Hidden | Layers | Heads | Per-stage params (trainable) | HF repo | Total disk |
|---|---|---|---|---|---|---|
| **Bark** (full) | 1024 | 24 | 16 | ~300M | `suno/bark` | 22.2 GB |
| **Bark-Small** | 768 | 12 | 12 | ~80M | `suno/bark-small` | 1.69 GB |

Sun's original release split the full-version weights across three separate `.pt` files per stage; HF's port consolidates them into a single `pytorch_model.bin`. Files in the `suno/bark` repo on HuggingFace:

| File | Size | Purpose |
|---|---|---|
| `pytorch_model.bin` | 4.49 GB | Consolidated HF weights (semantic+coarse+fine+EnCodec) |
| `text.pt` | 2.32 GB | Original-format Bark semantic GPT (v1) |
| `text_2.pt` | 5.35 GB | Original-format Bark semantic GPT (v2) |
| `coarse.pt` | 1.25 GB | Original-format coarse GPT (v1) |
| `coarse_2.pt` | 3.93 GB | Original-format coarse GPT (v2) |
| `fine.pt` | 1.11 GB | Original-format fine GPT (v1) |
| `fine_2.pt` | 3.74 GB | Original-format fine GPT (v2) |
| `tokenizer.json` | 2.92 MB | bert-base-multilingual-cased fast tokenizer |
| `vocab.txt` | 996 kB | BERT WordPiece vocab |
| `config.json` | 8.81 kB | Per-stage HF config |
| `generation_config.json` | 4.91 kB | Per-stage generation defaults |
| `speaker_embeddings_path.json` | 61.1 kB | Voice-prompt registry (HF: ~550 voices including v1+v2+announcer across 14 languages) |

For Bark-Small only `pytorch_model.bin` (1.68 GB) + the same tokenizer/config files. **The "v2" suffix refers to retrained-and-improved checkpoints — same architecture, better quality. We should prefer `_2.pt` (or the HF consolidated bin which is v2).**

The original Suno repo's bare-pickle `.pt` files contain a `{"model_args": {...}, "model": state_dict}` dict (vanilla nanoGPT-style training output, single-file pickle). HF re-packages them into `BarkSemanticModel`, `BarkCoarseModel`, `BarkFineModel`, `BarkModel` (the wrapper) plus an `EncodecModel` instance.

### Three-Stage Pipeline (Architecture per Stage)

All three GPTs are GPT-2-style pre-norm Transformer blocks (`LayerNorm → Attention → residual → LayerNorm → MLP → residual`). MLP is `4*hidden`-wide GELU sandwich. Positional embeddings are **learned absolute** of size `block_size=1024`. **`bias=False` on Linears** in the HF port (the original Suno code defaults `bias=True` per nanoGPT, but the actually-released weights have `bias=False` everywhere except LayerNorms — config.json reports `bias=false`). Dropout 0.0 at inference. No rotary embeddings, no GQA — pure 2019-vintage GPT-2.

| Stage | Hidden (full / small) | Layers (full/small) | Heads (full/small) | Block size | Input vocab | Output vocab | Attention mask |
|---|---|---|---|---|---|---|---|
| Semantic (`BarkSemanticModel`) | 1024 / 768 | 24 / 12 | 16 / 12 | 1024 | 129,600 | 10,048 | causal |
| Coarse   (`BarkCoarseModel`)   | 1024 / 768 | 24 / 12 | 16 / 12 | 1024 | 12,096 | 12,096 | causal |
| Fine     (`BarkFineModel`)     | 1024 / 768 | 24 / 12 | 16 / 12 | 1024 | 1,056 | 1,056 | **non-causal** (full bidirectional) |

`d_head = hidden / heads = 64` in both variants. The arithmetic is comfortable for FlashAttention-style kernels.

Sanity-check on parameter counts per stage at full size, ignoring embeddings: `24 × (4 × 1024² + 8 × 1024²) = 24 × 12.6M ≈ 302M`. Add the token embedding (`129600 × 1024 = 132M` for semantic input) plus the output projection (10048 × 1024 = 10M for semantic output) plus positional embeddings (1024×1024 = 1M) → semantic total ≈ 445M params (HF reports ~447M for `BarkSemanticModel`). Coarse/fine have much smaller vocabs, so totals are closer to ~300M each.

#### Semantic Transformer

Causal GPT. Forward pass during generation:

1. Tokens are integer IDs in `[0, 129599]`. The semantic vocab `[0..9999]` is reserved for semantic tokens (10000 is `SEMANTIC_PAD_TOKEN` / EOS). Text tokens from BERT are shifted by `TEXT_ENCODING_OFFSET=10048` so a BERT vocab id of `5612` becomes input id `15660`. The text region effectively occupies `[10048, 129594]`. Special tokens: `TEXT_PAD_TOKEN=129595`, `SEMANTIC_PAD_TOKEN=10000`, `SEMANTIC_INFER_TOKEN=129599`, plus an unused-but-reserved `INFER_ARTIFACT_TOKEN`.
2. Input layout for inference: `[ text_tokens (right-padded with TEXT_PAD_TOKEN to length 256), semantic_history (left-padded or repeat-padded with SEMANTIC_PAD_TOKEN to length 256), SEMANTIC_INFER_TOKEN ]`. Total prefix length 513 tokens.
3. Embedding lookup → add learned position embedding → 24 (or 12) blocks → final LayerNorm → output projection of size 10,048 → sample with temperature/top-k → append → repeat.
4. Stop on `SEMANTIC_PAD_TOKEN` (which doubles as EOS, id=10000) OR after `max_gen_duration_s * SEMANTIC_RATE_HZ` tokens. There is also a `min_eos_p` early-stop threshold (0.2) on the EOS probability.

#### Coarse Transformer

Causal GPT predicting two interleaved EnCodec codebooks per audio timestep.

- Audio frame rate is **75 Hz** (EnCodec hop=320 at 24kHz → 75). Semantic rate is **49.9 Hz**. Ratio `75 / 49.9 × 2 = 3.005` ≈ 3 → for every 1 semantic token, the coarse model emits 3 coarse tokens (i.e. ~1.5 timesteps × 2 codebooks).
- Vocab layout (input and output share the same 12,096 IDs):
  - `[0, 1023]` — codebook-0 acoustic tokens.
  - `[1024, 2047]` — codebook-1 acoustic tokens (offset by `CODEBOOK_SIZE × 1`).
  - `12048` — `COARSE_SEMANTIC_PAD_TOKEN`.
  - `12050` — `COARSE_INFER_TOKEN`.
  - Semantic tokens from the prior stage are also remapped into this vocab; they live above the codebook regions and below the special tokens.
- Input layout per generation call: `[ flattened_semantic_window (≤ sliding_window_len=60 tokens), COARSE_INFER_TOKEN, flattened_coarse_history (≤ max_coarse_history=630 tokens) ]`. The coarse history is `(2, T) → transpose → reshape(-1)` giving `[c0_t0, c1_t0, c0_t1, c1_t1, ...]`.
- During sampling, an `AlternatingCodebooksLogitsProcessor` forces even-position outputs to land in `[0, 1023]` (codebook 0 range) and odd-position outputs to land in `[1024, 2047]` (codebook 1 range), by masking the other logits to `-inf` before softmax.
- After the chunk is generated, the codebook-1 tokens are subtracted by 1024 to recover EnCodec indices and the `(2, T_new)` tensor is appended to history. Then the window slides forward 60 semantic tokens.

#### Fine Transformer

**Non-causal masked LM.** This is the only one of the three that uses bidirectional attention (`is_causal=False`).

- Architecturally it has `n_codes_total=8` separate `Embedding(input_vocab_size=1056, hidden)` modules — one per codebook — and `8 - n_codes_given = 7` separate `Linear(hidden, output_vocab_size=1056, bias=False)` heads (codebook 0 is given by the coarse stage; codebooks 1..7 are predicted).
- Input is shape `(B, T, 8)` of int tokens. To prepare an embedding at step `i ∈ [n_codes_given .. 7]` the model **sums the embeddings of codebooks 0..i**: `h = Σ_{k=0..i} embed_k(tokens[:,:,k])`. Codebooks `i+1..7` are not embedded at this step.
- Forward pass runs 24 (or 12) non-causal blocks → final LayerNorm → `head_{i - n_codes_given}` → logits shape `(B, T, 1056)`. Sample (greedy by default — `do_sample=false` for fine), assign to `tokens[:,:,i]`, then increment `i`.
- Vocab size is 1056 = 1024 codebook entries + 32 reserved pad/special tokens (only `~1024` actually used).
- The fine model is "iteratively refined" by running 7 forward passes (for i=1..7); each pass uses the full 1024-token chunk in parallel. Total flops ≈ 7 × one transformer forward — but with no autoregressive blow-up, since all positions are predicted at once per pass.
- The fine model can only operate on a **fixed 1024-token chunk** at a time. For longer outputs, coarse history is chunked into overlapping 1024-token windows (stride 512), processed independently, then stitched with a 512-token discard at the join (HF: `max_fine_history_length=512`).

### EnCodec 24 kHz Decoder

A standard EnCodec 24 kHz model is bundled (Facebook `facebook/encodec_24khz` weights, optionally co-distributed inside `pytorch_model.bin`). Full architectural details in [AUDIO_CODECS.md](AUDIO_CODECS.md) "EnCodec" section. Key facts that matter to Bark:

- Sample rate **24,000 Hz**, mono.
- Hop length **320** → frame rate **75 Hz**.
- 8 RVQ codebooks of size 1024, codebook dim 128 — but Bark only uses the **first 8** (which is all there are at the highest bandwidth). The 24 kbps target bandwidth is implicit.
- Decoder is `Conv1d(128, 1024, kernel=7) → 2-layer LSTM(1024) → 4 × DecoderBlock(stride∈{2,4,5,8}) → Conv1d(32, 1, kernel=7)`. Upsampling ratios are **[8, 5, 4, 2]** (so 320× upsample total).
- Input: `(B, 8, T)` int codebook indices. Output: `(B, 1, T × 320)` float waveform, range roughly [-1, 1].

### Text Encoding (BERT tokenizer)

Bark uses `bert-base-multilingual-cased` **as the tokenizer only** — no BERT encoder is run. Only the WordPiece tokenizer's `encode()` step is used to map a text string to integer ids. Then those ids are shifted by `+10048` and embedded directly by the Semantic Transformer's input embedding (which sees them as an extension of its own vocab).

- Tokenizer file: `tokenizer.json` (2.92 MB, HF Fast format) or `vocab.txt` + `tokenizer_config.json` (slow format).
- Vocab size: **119,547** (the "cased" mBERT vocab — the config padding gets us to 129,595).
- Output: pad/truncate to length **256** with `TEXT_PAD_TOKEN=129595`. No `[CLS]`, no `[SEP]` — Bark drops them. Right-padded.
- The text is fed through the embedding lookup of the Semantic Transformer; there is no separate text encoder.

### Voice Prompts (Speaker Prompts / History)

Bark has no learned voice embedding. Voice control is purely "in-context" via pre-recorded prompt triples:

- A speaker prompt is **three numpy arrays**: `<voice>_semantic_prompt.npy` (int64, shape `(T_sem,)`, typically ~250-350 tokens covering ~5-7 s), `<voice>_coarse_prompt.npy` (int64, shape `(2, T_coarse)` ~ `(2, 384)`), and `<voice>_fine_prompt.npy` (int64, shape `(8, T_fine)` ~ `(8, 384)`).
- These are extracted offline from ~10 s of reference audio by Suno's internal tooling (HuBERT-based clusterer for semantic; EnCodec encoder for coarse/fine). The extraction pipeline is **not released**.
- ~120 official voices ship in `bark/assets/prompts/` in the original repo (10 per language × ~12 languages, plus a few extras). HF's `ylacombe/bark-large` bundles ~550 prompts including v1 + v2 + an `announcer` neutral voice.
- Naming: `<lang>_speaker_<N>` for N in 0..9, e.g. `en_speaker_3`, `ja_speaker_5`, `zh_speaker_0`. Plus `announcer` for a neutral non-language-coded voice.
- A v2 set lives under `speaker_embeddings/v2/` — same speakers, re-extracted from the v2 model. **Use v2 with the v2 weights.**

At generation time:
- Semantic stage: the semantic prompt is right-padded with `SEMANTIC_PAD_TOKEN` to length 256 and placed in the second 256-slot of the input layout (between text and the inference token).
- Coarse stage: the coarse prompt is interleaved-flattened (transpose then reshape) and prepended to the history slot.
- Fine stage: the fine prompt is concatenated as `(8, prompt_T)` to the front of the coarse output before fine refinement.

History after one generation can also be passed to the next call (`return_output_full=True` returns a fresh history triple), enabling multi-sentence consistency.

### Special Markers in Text

Bark interprets several literal text markers (case-sensitive) — they are not tokenized specially, just learned from training data labels:

- `[laughs]`, `[laughter]`, `[sighs]`, `[gasps]`, `[clears throat]` — non-speech vocalizations
- `[music]` — switch to background music generation
- `[MAN]`, `[WOMAN]` — bias toward male / female voice (only loosely respected; voice prompts are stronger)
- `♪` (U+266A) — lyrics / singing
- `—` (em-dash) and `...` — pauses / hesitations
- ALL-CAPS WORDS — word-level emphasis

Code-switching across languages is implicit — just write a sentence with the target language and use a voice prompt from that language. There are no explicit `<en>` / `<ja>` tags.

### Sampling Defaults (from `generation_config.json`)

| Stage | temperature | top_k | top_p | do_sample | num_beams | max new tokens | other |
|---|---|---|---|---|---|---|---|
| Semantic | 0.7 | 50 | 1.0 | true | 1 | 768 | min_eos_p (early-stop) 0.2; max duration ~13.6 s |
| Coarse | 0.7 | 50 | 1.0 | true | 1 | (window-driven) | window 60 sem tokens, history 630 |
| Fine | 0.5 | 50 | 1.0 | **false** (greedy) | 1 | 7 (codebook count) | max_fine_history_length 512 |

`repetition_penalty=1.0` (off) everywhere. `use_cache=true` for the two causal stages.

### Memory and Performance

| Metric | Full Bark | Bark-Small |
|---|---|---|
| Total params (HF count, 3 GPTs + EnCodec) | ~1.5B | ~430M |
| Per-stage params | ~300-450M | ~80-150M |
| VRAM (FP32) | ~12 GB | ~4 GB |
| VRAM (FP16) | ~6 GB | ~2 GB |
| Disk (HF v2 single bin) | 4.49 GB | 1.68 GB |
| Latency: 1s audio on A100, FP16 | ~0.7-1.0 s (real-time) | ~0.3 s |
| Latency: 1s audio on RTX 4090, FP16 | ~1.0-1.5 s | ~0.4 s |
| First-audio latency (full Bark) | ~1.5-3.0 s | ~0.5-1.0 s |

With `SUNO_OFFLOAD_CPU=True` the three GPTs can be paged through 2 GB VRAM (each stage loaded only when running) at heavy latency cost.

Per-stage compute breakdown for a 10s utterance on A100 FP16, full Bark:
- Semantic: ~500 tokens × autoregressive → ~0.4 s
- Coarse: ~1500 tokens × autoregressive → ~1.2 s
- Fine: 7 × 1 forward of 1024 tokens → ~0.4 s
- EnCodec decode: ~0.05 s

### Multilingual Support

Officially supports 13 languages: **English, German, Spanish, French, Hindi, Italian, Japanese, Korean, Polish, Portuguese (BR), Russian, Turkish, Chinese (Simplified)**. Quality is heavily skewed toward English; non-Latin scripts (zh, ja, ko, hi) are usable but with frequent prosody glitches and occasional code-switching back to English. Use a same-language voice prompt for best results.

Code-switching within a single utterance works because the BERT-tokenizer + speaker-prompt-only design imposes no language tag.

## Key Numbers / Constants

| Constant | Value | Notes |
|---|---|---|
| `SAMPLE_RATE` | 24,000 Hz | EnCodec / output audio |
| `SEMANTIC_RATE_HZ` | 49.9 Hz | Custom HuBERT-derived semantic codebook |
| `COARSE_RATE_HZ` | 75 Hz | EnCodec frame rate (24000/320) |
| `SEMANTIC_VOCAB_SIZE` | 10,000 | Semantic codebook size |
| `SEMANTIC_PAD_TOKEN` | 10,000 | Doubles as EOS |
| `CODEBOOK_SIZE` | 1,024 | EnCodec per-codebook size |
| `N_COARSE_CODEBOOKS` | 2 | Coarse stage outputs |
| `N_FINE_CODEBOOKS` | 8 | Total EnCodec codebooks |
| `TEXT_ENCODING_OFFSET` | 10,048 | BERT token id shift |
| `TEXT_PAD_TOKEN` | 129,595 | Text-side pad |
| `SEMANTIC_INFER_TOKEN` | 129,599 | "Begin semantic generation" marker |
| `COARSE_SEMANTIC_PAD_TOKEN` | 12,048 | Pad in coarse input vocab |
| `COARSE_INFER_TOKEN` | 12,050 | "Begin coarse generation" marker |
| Semantic input length (text+history+infer) | 513 | 256 + 256 + 1 |
| `max_coarse_history` | 630 | Coarse history truncation |
| `sliding_window_len` (coarse) | 60 | Semantic tokens consumed per window |
| `max_fine_history_length` | 512 | Fine model context overlap |
| `block_size` (all three GPTs) | 1024 | Max sequence length |
| `n_codes_total` (fine) | 8 | EnCodec codebooks |
| `n_codes_given` (fine) | 1 | Codebook 0 is provided input |
| `min_eos_p` (semantic) | 0.2 | Early-stop threshold on EOS probability |
| Default temperatures | sem=0.7, coarse=0.7, fine=0.5 | Fine is greedy (`do_sample=false`) |
| Default top_k | 50 (all stages) | |
| Hidden dim (full / small) | 1024 / 768 | All three GPTs |
| Layers (full / small) | 24 / 12 | All three GPTs |
| Heads (full / small) | 16 / 12 | `d_head=64` both |
| EnCodec upsample ratios | [8, 5, 4, 2] | Product = 320 |

## Data Layouts / Formats

### Semantic Transformer Input (513 tokens)
```
Position:   [0 ......... 255]      [256 ......... 511]     [512]
Content:    text tokens             semantic history        SEMANTIC_INFER_TOKEN
            (BERT IDs + 10048,      (prompt tokens,         (129599)
             right-pad with         right-pad with
             TEXT_PAD_TOKEN=129595) SEMANTIC_PAD_TOKEN=10000)
Dtype:      int64
Shape:      (B=1, 513)
```

### Semantic Transformer Output
```
A 1-D sequence of int IDs in [0, 9999], length up to 768.
EOS = 10000 (SEMANTIC_PAD_TOKEN).
Frame rate: 49.9 Hz → 768 tokens ≈ 15.4 seconds.
```

### Coarse Transformer Input Window (per sliding step)
```
Position:   [0 .... 59]                  [60]                [61 ... 691]
Content:    semantic window               COARSE_INFER_TOKEN  flat coarse history
            (60 semantic tokens, no       (12050)             ([c0,c1,c0,c1,...]
             offset — they live in the                        with codebook-1
             upper region of vocab=12096                      tokens offset by +1024,
             alongside coarse tokens)                         length ≤ 630)
Dtype:      int64
Shape:      (B=1, 1 + 60 + ≤630) = (1, ≤691)
```

### Coarse Transformer Output (per window)
```
~180 flattened tokens per window (60 semantic × ~3 coarse-per-semantic).
Even positions: codebook 0 in [0, 1023].
Odd positions:  codebook 1 in [1024, 2047] (subtract 1024 after sampling).
After full run: reshape to (2, T_coarse) of EnCodec indices in [0, 1023].
```

### Fine Transformer Input/Output (one 1024-window pass)
```
Input:  int64 tensor (B=1, 1024, 8) where columns 0..i-1 are filled (i = pred_idx),
        columns i..7 contain placeholder zeros.
        First fine_pred_idx columns may include the speaker fine_prompt prepended.
Forward: embeddings of columns 0..i summed → 24-block bidirectional transformer →
         head[i - 1] → logits (B, 1024, 1056).
Sample (greedy by default), write column i, increment i. Repeat for i=1..7.

For sequences longer than 1024: slide forward 512 tokens (overlap 512),
re-run all 7 passes, keep the last 512 of each subsequent window.
```

### Speaker Prompt File Set
```
Three .npy files per voice (numpy save format):
  <voice>_semantic_prompt.npy  shape (~256,)        int64
  <voice>_coarse_prompt.npy    shape (2, ~384)      int64
  <voice>_fine_prompt.npy      shape (8, ~384)      int64
Total size per voice: ~10-20 kB
```

### EnCodec Decoder Input/Output
```
Input:  int64 tensor (B=1, 8, T) — 8 codebook indices in [0, 1023]
        T = coarse_T = (semantic_T × 75 / 49.9) ≈ semantic_T × 1.503
Output: float32 tensor (B=1, 1, T*320) — mono 24kHz PCM, range ~[-1, 1]
```

### HF Consolidated Weight File (`pytorch_model.bin`)
```
PyTorch pickle, single state_dict with hierarchical keys:
  "semantic.*"       (BarkSemanticModel weights)
  "coarse_acoustics.*" (BarkCoarseModel weights)
  "fine_acoustics.*"   (BarkFineModel weights)
  "codec_model.*"      (EncodecModel weights)
Total: 4.49 GB (full) / 1.68 GB (small), FP32.
```

### Original Suno Per-Stage `.pt` Files
```
Each file is a pickled dict:
{
  "model_args": { "n_layer": int, "n_head": int, "n_embd": int,
                  "block_size": int, "input_vocab_size": int,
                  "output_vocab_size": int, "bias": bool, ... },
  "model": OrderedDict of "<prefix>module.<key>": Tensor
}
The "module." prefix is from nn.DataParallel training and must be stripped on load.
Key naming follows nanoGPT: transformer.wte, transformer.wpe,
  transformer.h.<n>.{ln_1,attn,ln_2,mlp}.*, transformer.ln_f, lm_head.
```

## Algorithm Steps

### Full Inference Pipeline (one utterance, no streaming)

```
1. TEXT PREPROCESSING
   a. Normalize (lowercase NO — BERT is cased)
   b. Run BertTokenizer.encode(text, add_special_tokens=False)
      → list of int IDs (e.g., 12-30 tokens for a sentence)
   c. Drop [CLS]/[SEP] (we asked tokenizer for no specials)
   d. Add TEXT_ENCODING_OFFSET=10048 to every id
   e. Right-pad with TEXT_PAD_TOKEN=129595 to length 256
   f. Truncate if longer (warn)
   Output: int64 (256,)

2. LOAD SPEAKER PROMPT (optional)
   a. Load <voice>_semantic_prompt.npy, _coarse_prompt.npy, _fine_prompt.npy
   b. If none, use SEMANTIC_PAD_TOKEN-filled (256,), zero (2, 0), zero (8, 0)

3. ASSEMBLE SEMANTIC INPUT
   a. Pad semantic_prompt with SEMANTIC_PAD_TOKEN to length 256
   b. Concat: [text_padded (256), semantic_prompt_padded (256), [SEMANTIC_INFER_TOKEN]]
   c. Shape (513,)

4. SEMANTIC GENERATION (causal AR)
   for step in 0..768:
     forward Semantic Transformer with KV cache
     logits = output[..., -1, :10001]  (only first 10001 IDs are valid outputs)
     apply temperature=0.7, top_k=50, softmax, sample
     if sampled == SEMANTIC_PAD_TOKEN (10000): break
     if P(SEMANTIC_PAD_TOKEN) > min_eos_p (0.2): break
     append to sequence
   Output: int64 (~500,) semantic tokens, IDs in [0, 9999]

5. COARSE GENERATION (causal AR, windowed)
   semantic_to_coarse_ratio = 75 / 49.9 * 2 ≈ 3.005
   coarse_so_far = empty (2, 0); offset codebook-1 indices later
   semantic_offset = 0
   while semantic_offset < len(semantic_tokens):
     a. Take next 60 semantic tokens as window
     b. Build input: [semantic_window, COARSE_INFER_TOKEN, flat(coarse_history[-630:])]
        Coarse history is transposed (T, 2) then reshape(-1).
        Codebook-1 tokens get +CODEBOOK_SIZE (=1024).
     c. Run Coarse Transformer in AR loop for ceil(60 * 3.005) ≈ 181 steps,
        enforcing AlternatingCodebooksLogitsProcessor:
          even-position outputs in [0,1023], odd-position in [1024,2047]
        Sampling: temperature=0.7, top_k=50
     d. Subtract 1024 from odd-position outputs
     e. Reshape to (2, 90), append to coarse_so_far
     f. semantic_offset += 60
   Output: int64 (2, T_coarse), IDs in [0, 1023]

6. FINE GENERATION (non-causal masked LM, chunked)
   Build (8, T_coarse) starting with coarse_so_far in rows 0..1, zeros in rows 2..7.
   Optionally prepend speaker fine_prompt → fine_input.
   For each 1024-token window (overlap 512):
     for i in 1..7:
       a. embeddings = sum_k=0..i of embed_k(fine_input[:, :, k])
       b. add positional embedding
       c. 24/12 non-causal blocks → final LN
       d. head_{i-1} → logits (1, 1024, 1056)
       e. sample (greedy, temperature=0.5 only when do_sample), assign to column i
   Stitch windows: keep last 512 of each window after the first.
   Strip the prepended fine_prompt.
   Output: int64 (8, T_coarse), IDs in [0, 1023]

7. ENCODEC DECODE
   Input (1, 8, T_coarse) → see AUDIO_CODECS.md "EnCodec" section
   Output: float32 (T_coarse * 320,) waveform at 24 kHz

8. POST-PROCESSING
   a. Optional: trim leading/trailing silence
   b. Optional: normalize peak amplitude
   c. Write WAV at 24,000 Hz mono
```

### Codebook-Alternation Detail (Coarse Stage)

`AlternatingCodebooksLogitsProcessor` keeps a step counter. On even steps, sets logits `[1024..]` to `-inf` (forces a codebook-0 output). On odd steps, sets logits `[0..1023]` and `[2048..]` to `-inf` (forces a codebook-1 output). The model has learned to mostly do this anyway, but the processor guarantees it.

### Fine-Stage Window Stitching

For coarse outputs longer than 1024 frames:
1. First window: `coarse[:, :, 0:1024]`, fully predict columns 1..7, keep all 1024 frames.
2. Each subsequent window: `coarse[:, :, k*512 : k*512+1024]` where the leading 512 frames are already-finalized fine output, the trailing 512 are fresh coarse. Predict columns 1..7 over all 1024 frames, but only **keep the last 512**.
3. Concatenate kept slices. Total predicted frames = original coarse length, possibly with a partial last window.

## Open Questions

- [ ] Exact training corpus, hours, and HuBERT variant used to define the 10000-entry semantic codebook (Suno has not published).
- [ ] How `[MAN]` / `[WOMAN]` biasing actually trained — they appear to be just labeled prefixes in training data, no architectural change.
- [ ] How speaker prompts were originally extracted: confirmed HuBERT for semantic, EnCodec for coarse/fine, but the exact HuBERT checkpoint/cluster count is undocumented.
- [ ] Whether the v1 → v2 weight upgrade changed any architecture details (we believe no — just retraining — but config.json doesn't track this).
- [ ] What `INFER_ARTIFACT_TOKEN` is used for (referenced in HF source but appears unused at inference).
- [ ] What the 32 extra entries in the fine-stage `output_vocab_size=1056 = 1024 + 32` are (likely all-pad / unused).

## Implementation Notes for SharpInference

1. **Three independent GPT-2-style models**: all three stages reuse the same Transformer block. Implement one `GptBlock` (LayerNorm + CausalSelfAttention/NonCausalSelfAttention + LayerNorm + MLP) and one `GptModel` shell parametrized by `(hidden, layers, heads, vocab, block_size, attention_kind)`. Then instantiate three times. Estimated ~600 lines of C# total for the model definitions.

2. **`bias=False` everywhere except LayerNorms** in the released weights (despite the original nanoGPT defaulting to `bias=True`). Mirror the HF behavior: Linear modules omit bias, LayerNorms keep bias. Saves ~3% memory and is required to load the released checkpoints.

3. **BERT WordPiece tokenizer** (mBERT, ~120k vocab). Cross-reference [TOKENIZERS.md](TOKENIZERS.md) for the WordPiece implementation. We need:
   - UTF-8 NFC normalization, no lowercasing (cased model).
   - Whitespace splitting + Chinese character segmentation (a single CJK character is its own word).
   - WordPiece greedy longest-match-first with `##` continuation prefix.
   - The vocab can be loaded from `vocab.txt` (one token per line, line number = id) — much simpler than parsing the full `tokenizer.json`.
   - We do NOT need `[CLS]`/`[SEP]`/`[MASK]` insertion — Bark drops them.

4. **EnCodec decoder reuse**: implement once per [AUDIO_CODECS.md](AUDIO_CODECS.md). The exact same decoder is shared with MusicGen, AudioGen, and many other models — high reuse value. Bark only needs the **decoder path**; we can skip the encoder unless we want a "voice cloning from raw audio" feature (which would require also re-implementing the HuBERT semantic clusterer — much harder).

5. **Speaker-prompt format**: simplest representation is three flat `int32` arrays per voice. Convert from `.npy` to our packed format once at packaging time. Voice file size is <20 kB, so we can ship all ~120 official voices for a few MB.

6. **`AlternatingCodebooksLogitsProcessor`**: implement as a single boolean toggle in the coarse sampler. Before the softmax: if `step_count % 2 == 0` set logits indices 1024..2047 to `float.MinValue`; else set 0..1023 and 2048.. to MinValue.

7. **Fine-stage non-causal attention**: most-easy if our `GptBlock` accepts an `IsCausal` flag and we generate a different attention mask (all-ones vs lower-triangular). Both kernels can be unified.

8. **Fine-stage embedding sum**: 8 separate `Embedding(1056, hidden)` modules per `BarkFineModel`. At step `i`, compute `Σ_{k=0..i} W_emb_k[tokens[:,:,k]]`. This is `i+1` parallel gathers + a sum reduction along a 9th axis. A naive loop of i+1 gathers each producing `(B, T, H)` and adding to an accumulator is fine.

9. **Sampling**: temperature + top-k is straightforward — sort logits, take top 50, softmax, multinomial. Same code as for any GPT-2 generation. Fine stage is greedy: just argmax.

10. **KV cache**: only needed for the two causal stages (semantic, coarse). Fine is one-shot per pass, no cache needed. The KV cache size for full Bark semantic = `24 layers × 2 (k,v) × 1024 max_seq × 1024 hidden × 2 (fp16) = 100 MB` worst case. Tractable.

11. **Memory plan** (full Bark, FP16, single GPU):
    - Semantic weights: ~900 MB (mostly the 132M embedding table)
    - Coarse weights: ~600 MB
    - Fine weights: ~700 MB (8 embedding tables × 1056 × 1024 × 2 + heads + transformer)
    - EnCodec weights: ~40 MB
    - KV caches: ~200 MB peak
    - Activations: ~500 MB peak
    - **Total ≈ 3 GB FP16, 6 GB FP32**. Comfortably fits a single 8 GB card.
    Plan: ship FP16 weights by default; allow opt-in BF16 / FP32.

12. **Streaming (cross-reference [STREAMING_AUDIO_INFERENCE.md](STREAMING_AUDIO_INFERENCE.md) §3.1)**: real streaming is awkward because the fine stage requires a full coarse window. Two paths:
    - **No streaming**: simplest, ~1.5-3 s first-audio latency on a fast GPU.
    - **Coarse-only streaming**: emit raw EnCodec-decoded coarse-only audio as the coarse stage progresses (audio quality is degraded — missing 6 fine codebooks ≈ 1.5 kbps quality), then re-emit the fully-fine version when it finishes. Probably not worth the code complexity for v1.
    - Defer streaming to v2.

13. **Determinism**: with the default sampling, runs are non-deterministic unless we set the RNG seed. The HF code respects PyTorch's RNG. For SharpInference we need a deterministic seedable sampler — `Random(seed)` for the multinomial draws, parameterized via a constructor argument.

14. **Loading the original Suno `.pt` files vs HF consolidated `pytorch_model.bin`**: both are PyTorch pickle. Convert offline to safetensors as a packaging step. Cross-reference [SAFETENSORS_FORMAT.md](SAFETENSORS_FORMAT.md). Suno per-stage files have a `model_args` dict + a `model` state_dict with `"module."` prefix; HF consolidated has hierarchical keys (`semantic.*`, `coarse_acoustics.*`, `fine_acoustics.*`, `codec_model.*`) with no `module.` prefix. Both convert cleanly.

15. **`SUNO_USE_SMALL_MODELS` / `SUNO_OFFLOAD_CPU`**: not needed in our design — we'll expose a `BarkVariant` enum (`Full | Small`) and an explicit `OffloadPolicy` API that pages each GPT into VRAM only when running.

16. **Validation harness**: end-to-end golden tests should compare our output to Python's `bark.generate_audio()` output bit-for-bit at fixed seed. EnCodec decode is deterministic; the three GPT forward passes are deterministic; only the sampling step introduces variance. With seed control we can match Python exactly.

## Reference Implementations

- [suno-ai/bark](https://github.com/suno-ai/bark) — Official reference. Nanagpt-derived, ~2000 LoC total.
  - `bark/model.py` — Semantic + Coarse GPT (causal).
  - `bark/model_fine.py` — Fine GPT (non-causal masked LM).
  - `bark/generation.py` — All inference orchestration, the constants table, sampling defaults.
  - `bark/api.py` — High-level `generate_audio()` wrapper.
- [huggingface/transformers `models/bark/`](https://github.com/huggingface/transformers/tree/main/src/transformers/models/bark) — HF port with KV cache, batched generation, FlashAttention support.
  - `modeling_bark.py` — `BarkSemanticModel`, `BarkCoarseModel`, `BarkFineModel`, `BarkModel` wrapper.
  - `configuration_bark.py` — Config dataclasses (good source for default hyperparams).
  - `generation_configuration_bark.py` — Per-stage generation defaults.
  - `processing_bark.py` — Text + voice-prompt preprocessing.
- [suno/bark](https://huggingface.co/suno/bark) — Full-size weights.
- [suno/bark-small](https://huggingface.co/suno/bark-small) — Small variant.
- [ylacombe/bark-large](https://huggingface.co/ylacombe/bark-large) — Speaker-embedding registry (~550 voices).
- [serp-ai/bark-with-voice-clone](https://github.com/serp-ai/bark-with-voice-clone) — Community fork that demonstrates speaker-prompt extraction (HuBERT + EnCodec). Useful only if we want to expose voice cloning.

## Differences Between Implementations

| Aspect | Original Suno | HF Transformers |
|---|---|---|
| Weight format | 3 × `.pt` per variant | 1 × `pytorch_model.bin` |
| Class structure | Plain `GPT` + `FineGPT` | `BarkSemanticModel`, `BarkCoarseModel`, `BarkFineModel`, `BarkModel` |
| Sampling | Manual top-k loop in `generation.py` | `transformers.generation.utils.GenerationMixin` |
| Coarse alternation | Hard-coded `if step%2` masking | `AlternatingCodebooksLogitsProcessor` (subclass of `LogitsProcessor`) |
| KV cache | Manual list-of-tuples cache | HF's `Cache` abstraction |
| Speaker prompts | Bundled `.npz` per voice in `assets/prompts/` | External `ylacombe/bark-large` repo with per-prompt `.npy` files |
| Tokenizer | Direct call to `BertTokenizer` | `BarkProcessor` wraps it |
| EnCodec | Imported from `encodec` pip package | Internal `EncodecModel` from `transformers` |
| Bias on Linears | nanoGPT default `True` (but weights have none) | Honors config (`bias=false`) |
| Attention impl | Manual + optional `F.scaled_dot_product_attention` | SDPA / FlashAttention2 / eager all supported |

For SharpInference we model after the HF version (cleaner separation, config-driven), but use the original Suno per-stage `.pt` only if it's smaller for download.
