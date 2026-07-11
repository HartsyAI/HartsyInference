# HeartMuLa-oss-3B music — autoregressive decode benchmark (2026-07-11)

Model-level record for the HeartMuLa (Sesame-CSM dual-transformer + HeartCodec) decode-speed arc. HeartMuLa's
LM decodes codec frames **autoregressively at 12.5 Hz**, so the metric is **milliseconds per frame** (lower is
better), not wall-clock-per-image. End-to-end through the SwarmUI API (AudioLab `heartlib_music` provider).

**Hardware:** RTX 3060 12 GB (SM 8.6, CUDA 13.2). Host RAM 31 GB shared with an active VS Code session (~20 GB) —
relevant to the quant OOM note below.
**Workload:** model `Audio Models/HeartLib/3b-base` (`HeartMuLa-oss-3B` + `HeartCodec-oss-20260123`), prompt
"upbeat pop female vocal", lyrics "[Verse] hello world testing", 4 s clip (50 frames), seed 11, cond+uncond CFG
(scale 1.5). **Metric:** warm, steady-state **frames 10→50** (excludes the 2-frame graph-capture warmup and
model load). Every run visually/aurally confirmed to produce coherent audio — a fast broken decode is not a result.

## Per-frame decode time (bf16)

| Config | ms/frame | frames/s | ≈ realtime (AR only) | Change that got there |
|---|---:|---:|---:|---|
| bf16 eager (baseline) | 91.5 | 10.9 | ~0.87× | — |
| + CUDA-graph decode, backbone only | ~85.5 | ~11.7 | ~0.94× | capture the single-frame 3B backbone step, replay per frame (`HARTSY_CSM_GRAPH`) |
| + CUDA-graph decode, backbone **+ depth** (default) | ~85–90 | ~11.2–11.7 | ~0.9× | + capture the depth-decoder steps (persistent session KV cache); cond+uncond = up to 4 graphs |

Graph-decode audio is **bit-identical** to eager (same output WAV md5, both CFG modes) — verified in
`CsmIncrementalDecodeTests.StepFrame_GraphDecode_MatchesEager` and on the real 3B model. Run-to-run system
variance is ~3–4 ms/frame, so backbone-only (6.3%) and backbone+depth (4.7%) are both **~5% and within noise**.

## Why only ~5% — the per-frame cost model

HeartMuLa is **memory-bandwidth-bound, not launch-bound**. Per-frame weight streaming (cond+uncond, RTX 3060
~360 GB/s effective) vs the measured ~85–91 ms/frame:

| Component | GB streamed / frame | ≈ ms |
|---|---:|---:|
| backbone 3B (×1) | 6.0 | 33 |
| depth decoder 295M (×7) | 4.1 | 23 |
| heads + embeds (small audio-codebook vocab ~2048, **not** the 128k text vocab) | rest | ~30 |
| **total** | | **~86–91** |

Launch overhead — all a CUDA graph can remove — is only ~8 ms/frame (~28 backbone layers + ~3 depth layers ×
~10 kernels × cond/uncond × ~8 µs). So graph decode's ceiling here is ~9%, and it delivered ~5%. Graph decode is
the right tool for **launch-bound** models (the FX/vocoder decoders gain 2×+); it is a small lever for this one.

## Weight quantization — the real bandwidth lever (1.41× faster)

Cutting the weight bytes cuts the frame time. Implemented as **disk-cached** quantization
(`HARTSY_HEARTMULA_QUANT=q8_0|q4_k`, `CsmWeightCache`): the projection/head matrices are quantized **once** to a
GGUF cache (streaming convert — peak is one tensor's F32 buffer, **no OOM**; must run **post-`Remap`** since the
remap splits the combined audio embed/head tensors), then the ~4.5 GB Q8 cache is mmapped directly (never the
6.6 GB bf16).

| Config | ms/frame | frames/s | vs bf16 |
|---|---:|---:|---:|
| bf16 | 91.5 | 10.9 | 1.0× |
| **Q8_0 (disk cache)** | **64.8** | **15.4** | **1.41× faster** (~1/2 VRAM, past real-time) |

**The catch was residency, not the kernel.** A first build measured Q8 ~3–8× *slower* + degrading frame-to-frame.
Root cause: the CSM decode never `PreloadWeights`'d its weights (unlike Dia/MusicGen/F5/YuE and the LLM path), so
the GGUF-mmap quant weights were re-uploaded/cache-thrashed every step. A direct microbench proves the fused Q8
GEMV (`LaunchMulMatVecQ8_0F32`) is actually **faster than cuBLAS bf16** when the weight is resident:

| GEMV shape (M=1, weight resident) | bf16 µs | Q8 µs | Q8/bf16 |
|---|---:|---:|---:|
| attn q/o 3072×3072 | 64.4 | 46.5 | 0.72× |
| mlp up 8192×3072 | 155.9 | 114.4 | 0.73× |
| mlp down 3072×8192 | 169.8 | 109.6 | 0.65× |

The fix: `HeartMulaPipeline` pins the LM **device** matmul weights (`CsmModel.EnumerateDeviceWeights` — bodies +
heads + proj, *not* the host-gathered embeds, which would waste ~2.4 GB and OOM the card) via `PreloadWeights`,
**only when the weights are quantized** (`CsmModel.WeightsQuantized`) — the bf16 path caches its F16 cast on first
use and stays fast without pinning; pinning its 5.4 GB bodies non-evictably would OOM.

## Reproduce

```bash
# baseline (bf16) + graph decode default-on
HARTSY_CSM_GRAPH=1 ./src/bin/live_release/SwarmUI --launch_mode none   # graph on (default)
HARTSY_CSM_GRAPH=0 ./src/bin/live_release/SwarmUI --launch_mode none   # eager baseline
# Q8 quant (1.41× faster; first gen converts to the disk cache, then mmaps it):
HARTSY_HEARTMULA_QUANT=q8_0 ./src/bin/live_release/SwarmUI --launch_mode none
# then POST /API/GenerateText2Image {model:"Audio Models/HeartLib/3b-base", textaudioduration:4, seed:11, ...}
# read frame-rate from the [AudioLab][HeartMuLa] Frame N/M log timestamps (steady-state frames 10→50).
# NOTE: run one gen at a time — the 3B model is ~7-10 GB host RAM; concurrent gens / a heavy desktop OOM the box.
```
