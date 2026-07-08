# Video models — through-Swarm results after the 2026-07-08 perf day

End-to-end wall-clock through the **SwarmUI API** (same methodology as
[`video_comfy-vs-hartsy_2026-07-03.md`](video_comfy-vs-hartsy_2026-07-03.md): 25f, 512×320, h264-mp4, random
seed per gen, warm = median of 2–3, one backend enabled). Engine `alpha.44.8-local` (the full 07-08 batch:
LTX-2 Phases 0–2, Wan/Hunyuan/Kandinsky quick wins + fused-SDPA flips, Wan-variant `WanForwardCaches`,
pinned staging ring). Harness: `benchmarks/swarm_video_bench/bench_t2v.py`.

## Warm results vs the 2026-07-03 baselines

| Model | Steps | Comfy (07-03) | Hartsy 07-03 | **Hartsy 07-08** | vs baseline | Gap to Comfy |
|---|---|---|---|---|---|---|
| LTX-2.3 22B (video+audio) | 20 | n/a | 451 s | **95.5 s gen (57.5–105 s wall)** | **4.3–7.8×** | n/a (2.07 s/step, paired CFG) |
| LTX-0.9 2B | 20 | 2.84 s | 12–13 s | **10.1–10.4 s** | 1.2× | 3.6× |
| Wan 2.1 T2V 1.3B | 20 | 6.28 s | 23.7 s | **16.9–17.2 s** | 1.4× | **2.7×** (was 3.8×) |
| Wan 2.2 TI2V-5B | 20 | 4.52 s | 37.9 s | **21.9–22.6 s** | 1.7× | **4.9×** (was 8.4×) |
| Wan 2.1 T2V 14B fp8 | 15 | 30.62 s | 180.4 s | **36.8–37.9 s** | **4.8×** | **1.2×** (was 5.9×) — near parity |

Engine-harness numbers from the same day (not through Swarm): HunyuanVideo 13B fp8 **1.29 s/step** (was 2.15),
Kandinsky-5 Lite **0.83 s/step** (was 2.9). All outputs visually verified coherent (fox-in-snowy-forest set;
LTX-2.3 with decoded audio).

**Flagship regression on `alpha.44.8-local` (mandatory post-deploy):** Krea2-Turbo **4.45–4.53 s** ✓ (<6.5),
Z-Image-Turbo **2.92–2.96 s** ✓ (≤3.2), both visually pristine astronaut-on-horse. No regressions.

## Notes / gotchas from this round

- **Two-agent benching poisons numbers**: an interleaved bench session alternates model loads (every rep pays
  a full reload + contention). First LTX-0.9/Wan-1.3B pass read 72 s / 38–54 s; the quiet-window rerun read
  10.3 s / 17 s. Always bench on a quiet Swarm (`grep "Generated an image"` stable + GPU <4 GB).
- Wan 14B's 4.8× came mostly without model-specific work this round: fused SDPA + the shared final-layer/
  temb/rope/text-cache fixes + native-fp8 path. The remaining 1.2× to Comfy is Axis-B (per-GEMM transient
  dequant) + graph territory.
- Pre-existing flaky: full Wan CPU test tier under xunit parallelism corrupts `DupUp3D` (passes serially,
  reproduces at pre-07-08 commits). Needs a root-cause pass on shared native state in the CPU test path.
- LTX-2.3 remaining ladder (probes): video VAE ~18 s host loops (Phase 4, biggest), Gemma TE ~11 s/gen
  (Phase 5 prompt cache), F16 activations → bigger resident prefix (Phase 3).
