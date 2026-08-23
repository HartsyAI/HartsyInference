# MiniMax Music 3 — architecture notes

> **Stub.** This model is built and parity-verified (AR corr 0.9999989, flow corr 0.999996 — see
> `MODEL_STATUS_AUDIO.md` and `PARITY_VERIFICATION.md`), so the C# is the source of truth for *how it works*.
> This file only carries what the code cannot tell you: upstream provenance, constants to diff a suspect port
> against, and bring-up traps.

Provenance for `minimaxmusic3`. Everything derivable from the code lives in the code; this records what the
checkpoint does not tell you and what cost time to establish.

**Reference**: diffusers PR [#14456](https://github.com/huggingface/diffusers/pull/14456), pinned commit
`dafe3733fcfdbf3c48915fe77be3aef65b5d6a2d`. Reference tensors: `tests/python-reference/dump_minimax_music3_reference.py`.

## What the repo ships, and what to ignore

`MiniMaxAI/MiniMax-Music3` carries the model **twice**. Load only the diffusers-format subfolders —
`language_model/` (17.2 GB BF16), `transformer/` (9.6 GB **F32**), `rvq_depth_decoder/` (1.3 GB BF16),
`condition_encoder/`, `vocoder/`, `tokenizer/`. The SGLang-native copy (`qwen_7B/` 18 GB,
`flowmatching_vae.pth` 9.8 GB, `dav.pth`) is another 28 GB the engine never reads, which is why
`AudioCheckpoints.LoadSubfolderAsync` exists at all: the generic loader looks for weights at the repo root, and a
whole-repo pull here doubles the download.

## Stage shapes

```
caption + lyrics
  → clean/normalize (exact string port; whitespace changes change the audio)
  → "<|im_start|><|caption_start|>…<|caption_end|><|lyrics_start|>[start]\n…<|lyrics_end|><|im_end|><|audio_start|>"
  → ids [S];  uncond = ids with [1 .. S-3] replaced by <|audio_cfg|>  (keep index 0 and the LAST TWO)
AR, two batch-1 branches with their own KV caches:
  per frame → semantic code, 7 residual codes, and cat(LM hidden, 7 depth hiddens) = [1, 32768] layer-major
  → frame_hiddens [frames, 32768]        ← frame 0 is fed back but NOT emitted
Flow, per 200-frame window (hop 100):
  condition_encoder → [1, L, 2048], L = int(frames · 3.4453125)
  30 Euler steps, t = i/30, dt = 1/30, uncond conditioning is ZEROS
  → vocoder [1,128,L] → fold to [2,64,L] → [1, 2, L·512] at 44.1 kHz
```

## Constants (checkpoint contract, not knobs)

AR CFG 1.5 · AR top-k 50 · flow CFG 1.7 · 30 steps · chunk 200 frames / hop 100 · overlap 172 latents ·
crop-left 86 / crop-right 258 · 25 Hz frames · latent hop 512 · 3.4453125 latents per frame · max prompt 5000
tokens · max 9000 frames.

Token ids read from the checkpoint's own `tokenizer.json`: `<|audio_cfg|>` 151654, `<|audio_start|>` 151669,
`<|audio_end|>` 151670, `<|caption_start|>` 151671, `<|caption_end|>` 151672, `<|lyrics_start|>` 151673,
`<|lyrics_end|>` 151674, audio-code offset **151675**, semantic vocab 16384.

## Things that cost time

- **Output is 44.1 kHz stereo on this path** (`vocoder/config.json`). The model card's "32 kHz" describes the
  SGLang stack. Do not reconcile them.
- **The scheduler config is a red herring.** `invert_sigmas: true` with `num_train_timesteps: 1` over
  `linspace(1, 1/N, N)` reduces to `sigma_i = i/N`, a uniform-step Euler walk from t=0 (noise) to t=1 (data).
  That is why `MiniMaxMusic3FlowPipeline` integrates inline instead of using the shared scheduler.
- **The feed-forward gate order is reversed.** `ff_in.chunk(2)` then `first · silu(second)` — the opposite of
  the usual convention, and the classic garbled-but-finite failure if you get it backwards.
- **The AR top-k candidate set comes from the CONDITIONAL logits** but the distribution sampled is the guided
  one. Guidance over two masked `-inf` logits is NaN, so the mask goes on after the blend.
- **`lm_head` is sliced at load** to the 16384 semantic rows plus the end-of-audio row. Mathematically identical
  to the reference's vocabulary mask, and it drops the head from 1.6 GB to 268 MB.
- **The checkpoint is mixed-dtype and several tensors are read host-side.** The depth decoder's
  `audio_embeddings`/`pos_embedding`/norms and the transformer's `time_proj` are read as raw floats, so they must
  be widened at load; reading a BF16 tensor as F32 silently halves the span and lands out of range. Both were live
  bugs found by the first real generation, not by any shape check.
- **Snake alpha**: the reference divides by `alpha + 1e-9`, the shared kernel divides by `alpha`, and the
  checkpoint's smallest `|alpha|` is 4.7e-6 — the epsilon is folded into the weights at load rather than into the
  kernel.
- **The vocoder is a DAC decoder.** Same Snake, same `k=7 dilated + k=1` residual units at dilations 1/3/9, same
  `kernel = 2·stride, padding = ceil(stride/2)`. `MiniMaxMusic3Vocoder` is a key remap over `DacDecoder`, not a
  reimplementation. Stereo is a channel fold of the 128 latent channels, not a batch axis.
