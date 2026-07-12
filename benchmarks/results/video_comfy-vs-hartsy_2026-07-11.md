# Video models — through-Swarm scoreboard on engine 44.84 (2026-07-11)

Warm end-to-end wall-clock through the **SwarmUI API**, same methodology as
[`video_comfy-vs-hartsy_2026-07-08.md`](video_comfy-vs-hartsy_2026-07-08.md) and the 07-03 baseline:
25 frames, 512×320, h264-mp4, `videoresolution=Image`, **seed randomized per gen** (defeats the
identical-params result cache), warm = the agreeing reps of 2 (contaminated reps discarded — see below).
Engine `alpha.44.84-local`, backend #2 (`hartsyinference`) on the **RTX 4090**. Harness:
`benchmarks/swarm_video_bench/bench_t2v.py`.

**Comfy column is carried forward from the 07-03 head-to-head** — ComfyUI's performance did not change
between our engine versions (only the Hartsy backend did), so the existing Comfy warm numbers remain the
valid baseline. Only models that never had a Comfy workflow (Hunyuan, LTX-0.9.7 13B) still need a fresh
Comfy run; LTX-2.3 has no comparable Comfy workflow on this box.

## Results — warm generation (model resident)

| Model | Steps | Comfy warm | Hartsy 07-03 | Hartsy 07-08 | **Hartsy 44.84** | Gap to Comfy | Verdict |
|---|---|---|---|---|---|---|---|
| **Wan 2.1 T2V 14B fp8** | 15 | **30.62 s** | 180.4 s | 37 s | **30.58 s** | **1.00× — TIED** | ✅ **at parity** |
| LTX-0.9 2B | 20 | 2.84 s | ~12–13 s | 10.3 s | **4.59 s** | 1.62× | much closer |
| Wan 2.1 T2V 1.3B | 20 | 6.28 s | 67.6 s | 17 s | **11.22 s** | 1.79× | much closer |
| Wan 2.2 TI2V-5B | 20 | 4.52 s | 37.9 s | 22 s | **15.5 s** | 3.4× | improved |
| LTX-2.3 22B (video+audio) | 20 | *n/a* | ~451 s | 95.5 s gen | **42.3 s** | — | internal: 451→42 s (10.7×) |

Per-rep detail (warm, seconds): Wan-1.3B 11.21/11.22 · Wan-14B 30.28/30.58 · LTX-0.9 4.44/4.59 ·
LTX-2.3 42.25/42.29 · TI2V-5B 15.82 / **15.51** (two separate clean reps). Cold (first gen incl. model
load + C# convert): Wan-1.3B 418 s (heavy convert), Wan-14B 60 s, TI2V-5B 52–56 s, LTX-0.9 28 s,
LTX-2.3 95 s. Peak VRAM (warm): Wan-1.3B 10.0 GB, Wan-14B 21.0 GB, TI2V-5B 17.6 GB, LTX-0.9 7.2 GB,
LTX-2.3 19.1 GB.

## The headline

**Wan 14B fp8 has reached parity with ComfyUI — 30.58 s vs 30.62 s, dead even.** First video model to
catch Comfy (was 5.9× behind on 07-03, 1.2× on 07-08). Per the campaign handoff, Wan is now at its **fp8
compute floor** — rounds 9/10 closed CUDA-graph and batched-CFG as dead-ends with evidence — so parity is
where Wan 14B stays absent a fundamentally faster fp8 GEMM. **No video model decisively beats Comfy yet**,
but every model improved sharply from 07-08 (gaps 2.7–4.9× → 1.6–3.4×). The residual gap lives on the
small/fast models (LTX-0.9, Wan 1.3B), where Comfy's 3–6 s is kernel-launch-overhead-bound.

## Contention note (the one contaminated data point)

Benching on a **shared 4090** (a second agent was running Animate round 12): the first TI2V-5B pass read
warm[0]=65.5 s / warm[1]=15.8 s — a 4× swing from the other agent generating concurrently during rep 0.
A clean re-bench in a confirmed-idle window read 15.51 s, matching the earlier clean 15.82 s. The
`bench_t2v.py` median-of-2 picks the *slower* rep after sort, so it will report a contaminated number if
either rep is poisoned — **always discard reps that disagree by >~5% and re-bench on a quiet Swarm**
(`nvidia-smi` idle-gate + memory-stable). All other models' two reps agreed within ~1–2%.

## Coherence

No `ffmpeg` on this box to extract frames, but all output mp4s are healthy sizes (117–493 KB for 25f
clips; a failed/black gen is near-empty), and these exact checkpoints/pipelines were visually validated
coherent by the 07-08→07-11 rounds 1–11 work with **no engine change since** — coherence is inherited.
Clips viewable in the Swarm gallery (local/raw/2026-07-11 IDs 1915–1925).

## Still-open scoreboard rows (need a fresh Comfy workflow or wiring)

- **HunyuanVideo 13B fp8** — engine-side ~1.29 s/step; needs a Comfy Hunyuan T2V workflow for a head-to-head.
- **LTX-0.9.7 13B** — needs a Comfy LTX-13B workflow.
- **Kandinsky-5 Lite 2B** — cannot gen through Swarm until in-engine text encoders are wired (Step 1a).
- **Wan I2V / VACE / S2V** — have Comfy refs; not benched this round (focused on the T2V core question).
