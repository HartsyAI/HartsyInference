# Chatterbox (ResembleAI) TTS — perf verification, 2026-07-18

**Verdict: no perf pass needed. Chatterbox is already faster than real-time (RTF 0.69–0.90) and word-perfect.**

The handoff flagged Chatterbox as "~219 s/clip, slowest remaining correct model." That figure is **stale** —
it comes from `audio_tts_stt_2026-07-12.md`, before the 2026-07-17 CosyVoice2 S3Gen GPU-residency refactor.
Chatterbox's S3Gen (CosyVoiceFlow + HiFTNetVocoder) is the *same shared code*; it inherited the speedup with
zero Chatterbox-specific work.

## Setup
- GPU: RTX 3060 (engine `CudaBackend(1)`), `LD_LIBRARY_PATH=$HOME/.local/lib/cuda13`.
- Real weights: `ResembleAI--chatterbox` (`t3_cfg.safetensors` + `s3gen.safetensors`), default voice from
  `conds.pt` (precomputed T3 speaker emb [256] + flow embedding [192] + 150 cond-prompt speech tokens).
- Path exercised: T3 AR LM → CosyVoice2 S3Gen flow (10 Euler steps) → HiFTNet vocoder → 24 kHz.
  Precomputed-conditionals mode (no reference mel → S3 tokenizer / CAM++ encoder not on the hot path).
- Harness: `<scratchpad>/cbbench/` (mirrors `ChatterboxEndToEndTests`, adds per-stage timing via the
  pipeline `progress` callback; warmup gen then timed gen).

## Numbers (warm)
| Clip | Text tok | Speech tok | Audio | Wall | RTF | T3 | Flow | Vocoder |
|------|---------:|-----------:|------:|-----:|----:|---:|-----:|--------:|
| short | 56 | 127 | 5.08 s | 3.51 s | **0.69** | 1.49 s | 1.66 s | 0.35 s |
| long  | ~70 | 345 | 13.80 s | 12.36 s | **0.90** | 4.02 s | 7.39 s | 0.95 s |

Cold (first) gen: 6.20 s short / 16.61 s long (weight-cache fault-in).

## Scaling
- **T3** (net-new Llama_520M: hidden 1024 / 30L / 16 heads) is **linear** in speech tokens — ~11.7 ms/step,
  KV-cache decode healthy. 127→1.49 s, 345→4.02 s (2.72× tokens → 2.70× time).
- **Flow** is **super-linear** (127→1.66 s, 345→7.39 s = 4.45×) — CFM decoder attention is O(T²) over the mel
  sequence; it becomes the dominant stage on long clips. This is the shared CosyVoice code, already GPU-resident.
- Vocoder is negligible (~7%).

## Correctness
Whisper (medium.en) transcribes both clips verbatim:
- short: "Hello there, this is a test of the Chatterbox Speech Synthesizer, running end-to-end."
- long: pangram sentences verbatim (only whisper-side mishearings of nonsense pangram words).

## If more speed is ever wanted (not currently justified — already RTF < 1)
- Flow is the long-clip lever (O(T²) CFM). Any win there is shared with CosyVoice 2 / VibeVoice's flow users.
- T3 per-step is ~11.7 ms with the usual per-step host glue (SliceLast memcpy + logits host-read for sampling +
  AddEmbPlusPos host loop). CUDA-graph replay is a NO-GO on this AR loop (TF32-sensitive; same reason VibeVoice
  reverted it).
