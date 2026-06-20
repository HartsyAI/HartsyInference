# Kyutai Delayed-Streams Models (STT + TTS) — Architecture

> Build-ready spec for `kyutai/stt-1b-en_fr`, `kyutai/stt-2.6b-en`, `kyutai/tts-1.6b-en_fr`.
> Sources: Moshi paper [arXiv:2410.00037](https://arxiv.org/abs/2410.00037), DSM paper
> [arXiv:2509.08753](https://arxiv.org/abs/2509.08753), HF `kyutai_speech_to_text` transformers module,
> kyutai-labs/moshi + delayed-streams-modeling, and the model `config.json`s (fetched 2026-06-19).

## Shared foundation

The backbone ("Helium") is a **Llama-family decoder** = our `Qwen2Model` with `AttentionBias=false`,
SwiGLU (SiLU gate), RMSNorm, RoPE, no GQA at these sizes (`num_kv_heads == num_heads`),
`tie_word_embeddings=false`. The non-standard pieces are: (a) a single shared embedding table for text +
audio codes, (b) the delay framing, (c) the TTS depth transformer.

**Mimi codec** (already built): 24 kHz mono, codebook_size 2048, codebook_dim 256, 1 semantic + acoustic
RVQ. **DSM configs set `num_codebooks`/`num_quantizers = 32`** (vs original Moshi's 8). Per-codebook audio
vocab = **2049** (2048 codes + 1 pad/BOS). ⚠️ **Validate against a reference run**: 32 vs 8 codebooks, and
Mimi's true frame rate (Moshi = 12.5 Hz / 1920 samples; our `MimiConfig.Mimi24kHz` currently computes 25 Hz —
a Mimi reconcile item shared with CSM).

## STT

### Helium backbone (instantiate `Qwen2Config`, `AttentionBias=false`)

| field | stt-1b-en_fr | stt-2.6b-en |
|---|---|---|
| hidden_size | 2048 | 2048 |
| num_hidden_layers | 16 | 48 |
| num_attention_heads | 16 | 32 |
| num_key_value_heads | 16 (MHA) | 32 (MHA) |
| head_dim | 128 | 64 |
| ffn_dim (intermediate) | 11264 | 11264 |
| text vocab_size | 8001 | 4001 |
| rope_theta | 100000 | 100000 |
| rms_norm_eps | 1e-8 | 1e-8 |
| max_position_embeddings | 375 | 750 |
| sliding_window | 375 | 375 |

`dep_q=0` → **STT has no depth transformer** (it consumes audio, emits text).

### Input embedding (net-new vs a standard LM)

One shared `embed_tokens` of shape `[text_vocab + num_codebooks*2049 + 1, hidden]`. Per 80 ms frame the
transformer input = **sum** of: the previous text-token embedding + the 32 audio-code embeddings, each
audio code `c` of codebook `k` looked up at row `text_vocab + k*2049 + c`. `lm_head` is tied to this table;
sampling is restricted to the first `text_vocab` rows. HF keys: `model.embed_tokens.embed_tokens.weight`
(double-nested), gated MLP is `mlp.fc1` (→ ffn_dim, split gate|value) + `mlp.fc2` (ffn_dim/2 → hidden),
NOT `gate/up/down_proj`.

### Delay = silence padding (no per-codebook roll for STT)

Left-pad `audio_silence_prefix_seconds`, right-pad `audio_delay_seconds`, both at 24 kHz before Mimi-encode.
1b: prefix 0.0 / delay 0.5 s (6.25 frames). 2.6b: prefix 1.0 / delay 2.5 s (31.25 frames). Per-codebook
`delays` array is all-zero for STT.

### Output text stream

One text token per frame: **PAD (id 3)** = no word this frame; **WORD** = word boundary; otherwise
word-piece tokens. Word timestamps = frame_index/12.5 − delay. No EOS token (terminate on input end).

### Generation loop

Mimi-encode the silence-padded audio once → `[32, 1, T]` codes. For frame t: input = embed[prevText] +
Σ embed[text_vocab + k*2049 + code[k,t]]; one Helium step (KV cache, sliding window); project to text
logits (first text_vocab rows); sample. Audio is teacher-forced (no audio generated).

## TTS (`tts-1.6b-en_fr`) — deferred to a later pass

Temporal backbone: dim 2048, 16 layers, 16 heads, SwiGLU (hidden_scale 4.125), rope_theta **10000**,
context 500 (40 s). **Depth transformer ("depformer")**: dim 1024, 4 layers, 16 heads, FFN 3072,
`weights_per_step` with an 11-set schedule (codebooks 0–7 unique, 8–15→set 8, 16–23→9, 24–31→10),
low-rank per-codebook embeddings (128), `dep_q=32`. Per-codebook `delays=[0,0,2,2,…,2]`; text-vs-audio
stream delay 1.28 s (16 steps) with a 2-step-ahead second text stream. Text fed via a PAD/EPAD/WORD state
machine. Voice = cross-attention on a 512-dim speaker embedding (`kyutai/tts-voices`, ≤5 speakers); CFG +
control are summed LUT conditioners. Generation: text → temporal step → depformer iterates 32 codebooks →
un-delay → Mimi decode.

## C# build implications

1. Reuse `Qwen2Model` headless for the Helium body; manage the shared embedding + tied head outside it
   (the CSM headless pattern). Drive via `ForwardEmbeds`; project via `WhisperOps.ProjectLinear` over the
   first `text_vocab` rows.
2. STT custom input embed = gather-and-sum from the shared table with offsets `text_vocab + k*2049`.
3. STT delay = pad silence around the PCM; emit PAD/WORD per frame.
4. TTS additionally needs the depformer + per-codebook delay roll + speaker cross-attention (later pass).
5. Mimi needs a 32-codebook config variant for DSM.
