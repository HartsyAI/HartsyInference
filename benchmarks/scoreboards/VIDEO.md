# Video models — HartsyInference vs ComfyUI scoreboard

Canonical, single-source-of-truth scoreboard for video (T2V) diffusion models. Consolidates the
`video_comfy-vs-hartsy_*` campaign write-ups and the per-model bring-up benchmarks that formerly lived
as separate dated files in [`benchmarks/results/`](../results/) (now retired — this table is the
successor) into one table. Where multiple source runs covered the same model, the **freshest scoreboard
run wins** (07-11 over 07-08 over 07-03), unless a later per-model or per-feature result gave a more
precise number for that specific model — see Notes below for the one case where that applies (Wan2.2
TI2V-5B step-cache).

**Hardware:** RTX 4090 24 GB only — no video benchmarks have been run on the RTX 3060.
**Methodology:** end-to-end wall-clock through the **SwarmUI API** — the identical generation request
routed to the ComfyUI backend, then to the HartsyInference backend, on the same GPU, same request, warm
(model resident). This is the user-perceived latency gap, not an isolated kernel/pipeline timing. See
[`README.md`](README.md) for the engine's default performance profile and
how to reproduce these numbers. Standard workload (unless noted): 25 frames, 512×320, h264-mp4,
`videoresolution=Image`, seed randomized per gen to defeat SwarmUI's identical-params result cache.

## Results — warm generation (model resident)

| Model | GPU | HartsyInference | ComfyUI | Ratio | Date | Source |
|---|---|---:|---:|---:|---|---|
| Wan 2.1 T2V 14B (fp8, 15 steps) | RTX 4090 | 30.58 s | 30.62 s | 1.00× — tied (parity) | 2026-07-11 | video_comfy-vs-hartsy_2026-07-11.md |
| Wan 2.1 T2V 1.3B (fp16, 20 steps) | RTX 4090 | 11.22 s | **6.28 s** | 1.79× slower | 2026-07-11 | video_comfy-vs-hartsy_2026-07-11.md |
| LTX-0.9 2B (fp16, 20 steps) | RTX 4090 | 4.59 s | **2.84 s** | 1.62× slower | 2026-07-11 | video_comfy-vs-hartsy_2026-07-11.md |
| Wan 2.2 TI2V-5B (fp16, 20 steps) | RTX 4090 | 15.5 s | **4.52 s** | 3.4× slower | 2026-07-11 | video_comfy-vs-hartsy_2026-07-11.md |
| LTX-2.3 22B (video+audio, 20 steps) | RTX 4090 | 42.3 s | n/a — no comparable Comfy workflow | n/a | 2026-07-11 | video_comfy-vs-hartsy_2026-07-11.md |
| HunyuanVideo 13B T2V (fp8, 20 steps) | RTX 4090 | 1m26s e2e (~2.15 s/step) | n/a — no Comfy Hunyuan T2V workflow benched yet | n/a | 2026-07-02 | hunyuanvideo_e2e_2026-07-02.md |
| Kandinsky-5.0 T2V Lite (2B, 30 steps) | RTX 4090 | 102.0 s e2e (~2.9 s/step) | n/a — not yet wired through SwarmUI (in-engine text encoders pending) | n/a | 2026-07-02 | kandinsky5_t2v_e2e_2026-07-02.md |

Row count: 7. Bold marks the faster (lower-wall-clock) side of each head-to-head comparison; rows with no
ComfyUI baseline are left unbolded.

## SeedVR2-3B restoration — bring-up baseline vs Python reference (2026-08-01)

Not a T2V row: restoration (`hartsy restore`), measured at the E2E-parity operating point — 9-frame
Big Buck Bunny 360p clip, 640×360-area output, 4090, N=5, 95% CI (Student-t df=4). Correctness is
settled separately (C# output ≡ Python at SSIM 0.99950 with injected reference noises — see
`PARITY_VERIFICATION.md`); this row is the SPEED baseline for the future perf pass.

| Impl | Shape | Wall (9 frames) | s/frame | Peak VRAM |
|---|---|---|---|---|
| Python reference | **warm in-process**, bf16, causal slicing, dit-offload | 1.45 s ± 0.09 | 0.161 | 17.6 GiB |
| HartsyInference (bring-up) | **cold CLI e2e** (process + 13.6 GB fp32 mmap load + ffmpeg decode/mux), fp32, host-math DiT | 44.00 s ± 0.27 | 4.89 | ~16 GiB |

**Read the caveats before quoting a ratio:** the runs differ in warmth (in-process warm vs full CLI
cold start), dtype (bf16 vs fp32), and DiT execution (torch device kernels vs the deliberate host-math
bring-up shape — window gather/scatter, RoPE, qk-norm, AdaSingle all CPU-side). From the E2E gate run,
pipeline-only C# time at this shape is ~52.7 s *including first CUDA touch*; the perf-pass levers
(device window pack/unpack, GPU RoPE à la `HunyuanImageRope.ApplyGpu`, F16 activations, graph capture)
are enumerated in `MODEL_STATUS_VIDEO.md` §SeedVR2 follow-ups. Matrix-scale numbers (25f, 960×540-area):
~14 s/frame, 17.1 GB peak, 7/7 clips green.

## Notes

- **Wan 2.1 T2V 14B is the only video model at parity with ComfyUI** (30.58 s vs 30.62 s) — first video
  model to catch Comfy, up from 5.9× behind on 2026-07-03. Per the campaign write-up it has reached its
  fp8 compute floor (CUDA-graph and batched-CFG closed out as dead ends with evidence), so parity is
  where it is expected to stay absent a fundamentally faster fp8 GEMM.
- **ComfyUI column is carried forward from the 2026-07-03 head-to-head** for every model that has one —
  ComfyUI's own performance did not change across engine versions, only the Hartsy side did (per the
  07-11 file), so reusing the 07-03 Comfy numbers against the 07-11 Hartsy numbers is valid.
- **Wan2.2 TI2V-5B step-cache is opt-in and NOT the shipped-default number in the table above.**
  `2026-07-22_accel_stepcache_wan_4090.md` measured
  1.18–1.55× speedups (44.1–57.7 s vs a 68 s warm baseline) via `HARTSY_STEP_CACHE`, but on a *different*
  workload (832×480, 33 frames, 50 steps — not the standard 512×320/25f/20-step scoreboard workload, so
  the 68 s baseline there isn't directly comparable to the 15.5 s row above). More importantly, the
  benchmark's own verdict is negative for the pinned gate: no threshold holds SSIM ≥ 0.95 (best case 0.88
  at 1.18×), because Wan's 50-step UniPC trajectory is chaotically sensitive to any reuse — outputs stay
  coherent and prompt-faithful but diverge from the un-cached seed. The engine ships this **default OFF**
  as a "fast non-reproducible sampling" opt-in, not a transparent accelerator; `PERFORMANCE.md` (retired) §1's
  default-on feature table and §6 experimental-switch table both omit `HARTSY_STEP_CACHE` entirely,
  confirming it is not part of the standard profile.
- **HunyuanVideo 13B and Kandinsky-5.0 T2V Lite have no ComfyUI baseline yet** — per the 07-11 scoreboard
  these are still open rows pending a Comfy Hunyuan T2V workflow and in-engine text-encoder wiring
  (Kandinsky-5) respectively. The numbers shown are engine-side e2e wall-clock only, from their
  2026-07-02 bring-up benchmarks (not re-measured on a later engine build in these sources).
  HunyuanVideo's per-step number came down from ~75 s/step (bf16, block-swapped) to ~2.15 s/step via
  fp8-resident weights + GPU RoPE + `HARTSY_FP8_NATIVE`.
- **LTX-2.3 22B has no comparable Comfy workflow on this box**, so its row is internal-progress-only:
  451 s (2026-07-03) → 95.5 s (07-08) → 42.3 s (07-11), a 10.7× cumulative improvement, block-swap-bound
  (streams ~19 GB/forward on a 24 GB card).
