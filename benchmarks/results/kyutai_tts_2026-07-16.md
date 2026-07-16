# Kyutai TTS (DSM, tts-1.6b-en_fr) — HartsyInference vs moshi (2026-07-16)

Text → 24 kHz speech, pure-C# `MoshiTtsGenerator` (backbone + 32-codebook depformer + Mimi decode) vs the official
`moshi` Python package. Script = `"hello there, this is a test of the kyutai text to speech model"` (13 words),
voice = `expresso/ex03-ex01_happy_001_channel1_334s`, cfg-distilled (cfg_coef 2.0 as a condition), temp 0.6.
Both produce intelligible, script-correct speech (whisper medium.en); our output = *"So hello there, this is a test
of the Cuta[=Kyutai] text-to-speech model"*, peak 0.654 (no clipping). Frame counts differ run-to-run because both
sample at temp 0.6 — compare **ms/frame**, not wall-clock. GPU = RTX 3060 (engine `CudaBackend`,
`CUDA_VISIBLE_DEVICES=1`; moshi bf16 + torch CUDA graphs). CPU = same box, F32.

| Device | HartsyInference | moshi (Python) | ms/frame (ours / ref) | Notes |
|---|---|---|---|---|
| **RTX 3060** | **9.7 s** / 62 fr (156 ms/fr, RTF 0.51×) | 2.32 s / 65 fr (35.7 ms/fr, RTF 2.25×) | 156 / 35.7 (**4.4×**) | ours bf16-weights-as-F32, no CUDA graph; moshi bf16 + CUDA graph |
| **CPU (F32)** | 73.1 s / 54 fr (1354 ms/fr) | 25.0 s / 68 fr (367 ms/fr, RTF 0.22×) | 1354 / 367 (3.7×) | — |

Real-time is 80 ms/frame (12.5 Hz Mimi). We are ~2× slower than real time on the 3060; moshi is ~2.25× faster
than real time (it ships CUDA-graph capture of the per-frame step).

## What moved the number (3060, 63.0 s → 9.7 s, 6.5×)

The depformer predicts 32 codebooks per frame, each a 4-layer mini-transformer using a **weights-per-step**
schedule (11 weight sets). The per-set QKV / out-proj / gate projections were being **sliced out of the packed
weight fresh on every `Block` call** (`SliceRows(...)` → a new host tensor per codebook × layer × frame). The
device weight cache is keyed by tensor identity, so every one of those slices was a cache **miss** and re-uploaded
the weight to the GPU on every op.

Fix: pre-slice all per-set projections **once at load time** (`_selfIn[l,s]`, `_selfOut[l,s]`,
`_gateInGate/_gateInUp[l,s]`), so they are stable, device-resident tensors. Also GPU-ported the SwiGLU gate
(`Silu` + `Mul` on-device instead of a host sigmoid loop that drained the GPU per codebook).

Profile (40-frame gen, `HARTSY_PROFILE`):

| op | before | after |
|---|---|---|
| `Linear` | 21617 calls, 7696 ms (0.356 ms/call) | 55063 calls, 2500 ms (**0.045 ms/call**) |
| `H2D_MISS_BIG` (weight re-uploads) | 6469 calls, 6339 ms | **466 calls, 807 ms** |
| `SDPA` | 3712 calls, 1238 ms | 9480 calls, 2749 ms |

After the fix the cost is per-op **launch overhead** on the 32-codebook cascade (`SDPA` 2.7 s + `Linear` 2.5 s):
128 tiny SDPA + ~576 tiny Linear per frame, each dominated by fixed launch cost, not FLOPs. Closing the remaining
~4× to moshi needs a **per-frame CUDA-graph capture** of the backbone-step + depformer-cascade (moshi's lever) —
tracked as the follow-up perf pass; not attempted here to avoid destabilising the just-verified correctness.

## Correctness (the harder win)

Earlier the pipeline ran but produced non-speech + clipping. Root cause: `MoshiConditioner.ComputeCross` built the
cross-attention voice source from only the `T` real voice rows. moshi pads the voice to `max_speakers=5` slots (the
4 empty slots become the learned `speaker_wavs.learnt_padding` vector), then adds a continuous sin pos-emb over all
`5·T` rows — and cross-attention attends over **all** of them. Omitting the 500 padding rows shifted every
cross-attention output ~1% (concentrated in outlier activation dims), compounding over the autoregressive loop into
garbage. Fix: emit the full `[1, 5·T, 2048]` source. Also: the text token must be **sampled** (temp_text 0.6 /
top-k 25), not argmax — its new_word/pad choice paces the words.

Repro: `KyutaiTtsEndToEndTests.EndToEnd_ProducesAudio` with `KYUTAI_TTS_WEIGHTS` / `KYUTAI_MIMI` / `KYUTAI_SPM` /
`KYUTAI_VOICE` set (optional `GSV_CUDA=1` + `GSV_PTX`). moshi baseline: `moshi_bench.py` (`pip install moshi`).
