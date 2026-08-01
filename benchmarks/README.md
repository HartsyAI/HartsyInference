# HartsyInference Benchmarks

This directory holds the benchmarking infrastructure for [Phase B GPU performance optimization](../docs/Checklists/PHASE_B_GPU_PERFORMANCE.md). Read [`docs/Research/CUDA_PERFORMANCE_PLAN.md`](../docs/Research/CUDA_PERFORMANCE_PLAN.md) and [`docs/Research/PROFILING_METHODOLOGY.md`](../docs/Research/PROFILING_METHODOLOGY.md) before adding new benchmarks — the methodology is non-trivial.

## SeedVR2 restoration bring-up (2026-08-01)

First benchmark of the new `Modality.Restore` (SeedVR2-3B, one-step video restoration). Headline lives
in [`scoreboards/VIDEO.md`](scoreboards/VIDEO.md) §SeedVR2: Python reference (warm, bf16, sliced)
**0.161 s/frame** vs HartsyInference bring-up (cold CLI, fp32, host-math DiT) **4.89 s/frame** at the
E2E-parity point (9f BBB, 640×360-area, 4090, N=5, 95% CI both sides). No speedup is claimed — this is
the documented pre-optimization baseline; correctness is the shipped result (SSIM 0.99950 vs reference,
`PARITY_VERIFICATION.md`). Bench scripts: `tests/python-reference/seedvr2_reference/bench_seedvr2_python.py`
(needs dit-offload staging — co-resident bf16 vae+dit OOMs 24 GB) + 5× cold `hartsy restore` invocations.

## LLM decode throughput vs llama.cpp (2026-07-04)

Separate from the diffusion/Phase-B harness below. Docs: [`LLM_THROUGHPUT_BENCHMARK.md`](../docs/Checklists/LLM_THROUGHPUT_BENCHMARK.md) (baseline + method) and [`LLM_DECODE_PERF_GRIND.md`](../docs/Checklists/LLM_DECODE_PERF_GRIND.md) (optimization log). Assets:

- **Baseline**: CUDA-compiled `llama-bench` (`~/models/llamacpp/build/bin/`), swept over the 7 SOTA GGUFs in `Models/llm/` with `-ngl 99 -p 512 -n 128 -r 5`. Result: `results/llamacpp_baseline_3060.json`.
- **Tier-1 (engine, direct)**: the existing `samples/HartsyInference.TextGen.Cli` (`gguf cuda <ntok> "<prompt>"`) — loads a GGUF on CUDA, times decode, reports tok/s + D2H sync count. Raw: `results/tier1_engine_3060.txt`.
- **Tier-2 (engine, through Swarm)**: `swarm_llm_bench/swarm_llm_bench.py` drives the live SwarmUI `LLMAssistantSendMessageWS` WebSocket (hartsy-local provider), measuring client-side. Raw: `results/swarm_llm_3060.json`.
- **Outcome**: decode gap **20-54× → 1.94-2.88×** off llama.cpp (Llama-3.2-1B under 2×) via fused quantized GEMV + quantized lm_head + split-K flash-decode attention + vectorized loads.
- **CUDA graph foundation** (last lever): `hartsyinference-textgen graphtest` verifies `CudaGraph` capture/replay works on-GPU.

### Current LLM decode results (RTX 3060, warm, 128-token greedy, tok/s)

| Model | Quant | llama.cpp tg | engine tg | gap |
|---|---|---:|---:|---:|
| Llama-3.2-1B | Q8_0 | 215.9 | ~111.5 | 1.94× |
| Mistral-7B-v0.3 | Q4_K_M | 66.5 | ~30.7 | 2.12× |
| Qwen3-0.6B | Q4_K_M | 354.5 | ~157 | 2.26× |
| Gemma-3-1B | Q4_K_M | 229.8 | ~79.7 | 2.88× |

Prefill (pp512) is not the bottleneck; the remaining decode gap is launch-overhead on small models (next lever: CUDA graphs). Full per-phase log in `LLM_DECODE_PERF_GRIND.md`.

## Audio decode — HeartMuLa music (2026-07-11)

Autoregressive music LM (Sesame-CSM dual-transformer). Metric is **ms/frame** (12.5 Hz codec frames), measured e2e through the SwarmUI API on an **RTX 3060**. HeartMuLa is **memory-bandwidth-bound** (per-token weight streaming dominates), so CUDA-graph decode (`HARTSY_CSM_GRAPH`, default-on, bit-identical) buys ~5% and **Q8 weight quant is the real lever: 1.41× faster** (past real-time), once the quant weights are pinned GPU-resident (`PreloadWeights` — without it they re-upload every step and it's ~8× slower). Full write-up + cost model + reproduce steps: [`results/heartmula_music_e2e_2026-07-11.md`](results/heartmula_music_e2e_2026-07-11.md).

### Current HeartMuLa results (RTX 3060, 3b-base, warm, steady-state frames 10→50)

| Config | ms/frame | frames/s | vs bf16 |
|---|---:|---:|---:|
| bf16 eager | 91.5 | 10.9 | 1.0× |
| bf16 + CUDA-graph decode (default) | ~85–90 | ~11.2–11.7 | ~1.05× (bit-identical) |
| **Q8_0 disk-quant** (`HARTSY_HEARTMULA_QUANT=q8_0`) | **64.8** | **15.4** | **1.41×** (~1/2 VRAM) |

## Full audio fleet sweep — TTS/STT/Music/VC/Fx vs Swarm, 37 models (2026-07-25, canonical path)

Every local (non-cloud) AudioLab provider driven through `POST /API/GenerateText2Image` (TTS/Music/VC/Fx —
the same auto-saving path any image/video gen uses) and `POST /API/ProcessSTT` (STT), on the **RTX 4090**.
Supersedes the 2026-07-24 pass below, which used the raw `ProcessTTS`/`ProcessAudio` args dict — that path
was found to **bypass the real per-model param logic** (`BuildEngineArgs` in `DynamicAudioBackend.cs`),
root-causing ACE-Step's "mostly silence, no vocals" bug: a style description sent as `prompt` landed in the
engine's *lyrics* slot with `genre` empty. Docs:
[`AUDIO_THROUGHPUT_BENCHMARK.md`](../docs/Checklists/AUDIO_THROUGHPUT_BENCHMARK.md) (full per-model tables +
methodology + 10 numbered bugs). Harness:
[`swarm_audio_bench/swarm_audio_bench_v2.py`](swarm_audio_bench/swarm_audio_bench_v2.py); raw results:
[`swarm_audio_bench/swarm_audio_results_final.json`](swarm_audio_bench/swarm_audio_results_final.json).

**31/37 generated successfully** (up from 27/37 on 07-24 — 6 of that day's bugs are now fixed, confirmed by
this pass: Chatterbox, Zonos, Distil-Whisper, HeartMuLa, YuE, Demucs). Every output is content-quality-gated,
not just HTTP-200-gated: RMS/peak checked for near-silence or noise-clipping, and TTS/music outputs are
round-tripped through `whisper_stt` to confirm real transcribable content, not just non-zero bytes — the
07-24 pass only checked the latter, which is how the ACE-Step/AudioGen quality bugs shipped unnoticed. Best
RTF: Moonshine-streaming STT at 0.073× (13.7× faster than real-time). New bugs found this pass: **7 models
(HeartMuLa, NeuTTS, GPT-SoVITS, OpenVoice, RVC, Demucs, Resemble-Enhance) have real weights and functional
providers but are unselectable via `GenerateText2Image`** — root-caused to an engine-level vs. model-level
`installed`-flag mismatch, not yet fixed (needs its own pass); **Dia-1.6B hangs regardless of path** (3
independent timeouts); **AudioGen hard-hangs the whole Swarm process at 45s duration** (fine at 10s/20s,
required `SIGKILL` at 45s, reproduced twice). Full write-ups in the doc.

## Low-VRAM streaming + leak-on-OOM fix (2026-07-27) — supersedes several 2026-07-26 conclusions

Follow-on pass acting on the 2026-07-26 findings below. **Read this before trusting that pass's 3060 column.**
Full writeup: [`results/2026-07-27_lowvram_leak_fix.md`](results/2026-07-27_lowvram_leak_fix.md).

- **Leak-on-OOM fixed and verified**, including the cross-process claim: an OOM'd process that stays alive now
  holds **152 MiB instead of ~11.5 GB**, and a separate process on the same card succeeds where it previously failed.
- **Low-VRAM weight streaming is live** (`HARTSY_LOWVRAM`, three-state, `auto` by default). The machinery already
  existed (`BlockStreamingController`) but only 1 of ~25 image pipelines used it. **Three models that could not run on
  the 12 GB 3060 at all now do** (all 1024², quality-gate clean): **HunyuanImage-2.1** (19.7 s; 11.5 GB needed vs
  9.4 GB free), **Ideogram 4** — whose *pair* of 9.3 GB DiTs needs 19.7 GB against 9.2 GB available (20 steps in
  205 s, peak 5.1/11.6 GiB) — and **Qwen-Image**, a 20B MMDiT needing 14.3 GB against 9.7 GB available (20 steps in
  231 s, peak 9.96/11.6 GiB).
- **A bug had made streaming inert for every GGUF model**: `Q4_K.SizeInBytes` is 0, so block-size sums came out
  **zero bytes** and the "fits resident?" test was always trivially true. Flux had therefore never streamed a GGUF.
- **Lens solid-black: root-caused and fixed.** SageAttention's INT8 path materializes V as F16; Lens does not
  RMS-norm V, and `max|V|` crossed F16's 65504 at block 45. Not a Lens port bug — verified against ComfyUI's own
  reference implementation on the same checkpoint.
- **Anima: 792 → 3 D2H syncs/step**, 29× faster per step. Its documented "1024² hangs on the 4090" was never a hang.
- **The 2026-07-26 3060 column is partly unsound** — it ran 13 models in one process *with the leak active*, so only
  the first failure is necessarily genuine. Anima is proven contamination (failed in-sweep, passed standalone at
  7.5 GB the same day). Re-measured in fresh processes: **OmniGen2 passes** at 10.0 GB, and **Ideogram 4 never OOM'd
  at all** — it hit a hardcoded `>= 20 GB` refusal, i.e. a policy, not a memory failure. That constant is now a
  planner decision.

## Image T2I e2e vs ComfyUI, 4090 + 3060 (2026-07-26)

Full 4-lane sweep (Hartsy/Comfy × RTX 4090/RTX 3060) through the **SwarmUI API**, 16 models (13 from the
original sweep + SDXL, Chroma1-HD, Flux-Schnell added in a same-day follow-up that downloaded, benchmarked,
and — for Chroma1-HD — deleted each checkpoint to demonstrate the disk-constrained workflow end to end),
quality-gated + manually visually inspected. Supersedes the 2026-07-05 image pass below for anything it
overlaps with. Full table + reproduce commands: [`results/2026-07-26_image_comfy-vs-hartsy.md`](results/2026-07-26_image_comfy-vs-hartsy.md).
Handoff for the next agent (prioritized bug/perf list): [`../docs/Checklists/IMAGE_COMFY_BENCH_HANDOFF.md`](../docs/Checklists/IMAGE_COMFY_BENCH_HANDOFF.md).

**Headline**: on the 4090, Hartsy wins warm-generation on 7 of 13 models directly comparable to Comfy (up
to 1.6× faster, on Flux-Schnell); Comfy wins the rest, mostly within 1.05-1.45× except one real
Hartsy-side gap (Anima, ~60× slower — a missing perf pass, not a Comfy strength). Lens is excluded from
that count since it's currently broken on the Hartsy API path (see below) — Comfy is correct from the
identical checkpoint. On the 3060, Comfy wins essentially by default because it has VRAM-constrained
weight-offloading and Hartsy does not (10/13 models ran on Comfy vs 1/13 on Hartsy at identical params —
SDXL confirms the same pattern: OOM on Hartsy, 9.6GB and functional on Comfy).

> **Superseded 2026-07-27 (see the section above).** Hartsy now *has* VRAM-constrained streaming, and the 3060
> column here overstates the gap: that lane ran all 13 models in one process while the leak-on-OOM bug was live, so
> only the first failure is necessarily a genuine capacity limit. Do not cite the `OOM²` cells as capacity data.

**Bugs found this pass, ranked**: (1) **Lens produces solid-black output via the SwarmUI API** — 16/16
failures, both variants, 2 param sets, zero server-side errors; ComfyUI is correct from the identical
checkpoint (0/16 failures) — contradicts `MODEL_STATUS_IMAGE.md`'s existing ✅, see that doc's Lens entry
for the amended note. (2) **The engine leaks GPU memory on OOM** — one oversized request early in a
process poisons every subsequent request, even ones that would otherwise fit (isolated and proven: a
13-model 3060 sweep failed 13/13; a fresh process with only the one model that actually fits succeeded
cleanly). (3) No VRAM-constrained fallback path (see headline above). Full list of 10 findings in the
handoff doc.

> **All three of those are fixed as of 2026-07-27** — see the section above. Bug 1's root cause turned out to be
> SageAttention casting V to F16 (not anything Lens-specific); Bug 2 is fixed and verified cross-process; Bug 3's
> machinery already existed and just was not wired up.

## Diffusion / video e2e vs ComfyUI (2026-07-03)

End-to-end wall-clock through the **SwarmUI API** (the identical request routed to the ComfyUI backend, then the HartsyInference backend, on the same RTX 4090). This is the user-perceived-latency comparison; it complements the in-engine microbench harness. Full write-up + host-vs-compute diagnosis: [`results/video_comfy-vs-hartsy_2026-07-03.md`](results/video_comfy-vs-hartsy_2026-07-03.md).

| Model | Quant | Hartsy warm | Comfy warm | gap |
|---|---|---:|---:|---:|
| Wan 2.1 T2V 1.3B | fp16 | ~23.7 s | 6.28 s | ~3.8× |
| LTX-0.9 2B | fp16 | ~15 s | 2.84 s | ~5.3× |
| Wan 2.2 TI2V-5B | fp16 | ~37.9 s | 4.52 s | ~8.4× |
| Wan 2.1 T2V 14B | fp8 | ~180 s | 30.6 s | ~5.9× |

All outputs are coherent: this is a speed gap, not a correctness gap. Image architectures (Flux/SD3/Ideogram) were device-ported and run much closer; the video DiT blocks are the current frontier (no full flash-attention kernel yet, some F32-only elementwise ops, launch-overhead at small token counts).

## 3D mesh (image → mesh) e2e vs Python reference (2026-07-15)

Warm end-to-end seconds on an RTX 4090 vs the upstream Python reference on the same GPU. All perf changes are bit-exact or coherence-gated (`CudaOpBisectTests`). Full campaign write-up: [`results/threed_genperf_2026-07-15.md`](results/threed_genperf_2026-07-15.md).

| Model | Hartsy | Python ref | gap |
|---|---:|---:|---:|
| TripoSR (256³ density grid) | **2.1 s** | 0.58 s | 3.6× (was 26.2 s; our GPU density decode beats the reference) |
| Hunyuan3D-2 Shape (30 steps, grid 128, fp16) | **9.2 s** | 5.76 s | **1.6×** (was 71.3 s — 7.75× in-campaign) |

The Hunyuan3D wall was a pathological `Concat` (a per-slice `cuMemcpyDtoDAsync` loop → ~280k memcpy nodes/forward) — a fused kernel cut dit-loop 27.7 → 7.5 s bit-identical. Then DINOv2-giant host loops → device, fused DiT adaLN + QKV-split-norm kernels, and a device FourierEmbed. Unlike the video DiTs, the 3D forward is now near its compute floor (batched CFG was measured and ruled out).

## Image conditioning features + Lens/Lance genperf (2026-07-16/17)

Correctness/feature campaign for the image special-abilities set (ControlNet incl. union-type + FLUX-DiT + segmentation, IP-Adapter incl. FaceID / FaceID-Plus/PlusV2, FLUX Kontext/Fill/Canny/Depth/Redux, OmniGen2/Boogu/Qwen edits) — all live-verified through the SwarmUI API on an RTX 4090. Includes the shared-logic regression matrix (every changed primitive re-verified against its consumers, no model regressed) and the Lens/Lance generation-perf pass. Write-ups:

- [`results/image_conditioning_2026-07-16.md`](results/image_conditioning_2026-07-16.md) — Wave 1/2 feature verifications + the 14 engine bugs the campaign surfaced.
- [`results/image_deferred_wave_2026-07-17.md`](results/image_deferred_wave_2026-07-17.md) — deferred-wave (seg / union / FaceID-PlusV2 / Flux-depth kernel fix / Lens+Lance perf), flagship regression gate, and the shared-logic regression-coverage matrix.
- [`results/2026-07-16_lens_lance_genperf.md`](results/2026-07-16_lens_lance_genperf.md) — Lens 14.4→0.55 s/step (26×) and Lance 12.8→0.33 s/step (39×), parity-gated.

## Quick start

```bash
# 1. Set up Python venv (one time; see requirements.txt for pinned versions)
python3 -m venv benchmarks/python-baseline/.venv
source benchmarks/python-baseline/.venv/bin/activate
pip install -r benchmarks/python-baseline/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
deactivate

# 2. Run the full harness (~20-40 min depending on GPU + e2e flag)
bash benchmarks/run_benchmarks.sh --py-venv benchmarks/python-baseline/.venv

# 3. Smoke test (faster, 1 trial of MatMul only) for harness debugging
bash benchmarks/run_benchmarks.sh --smoke --py-venv benchmarks/python-baseline/.venv
```

Result lands in `benchmarks/results/run_<utc>_<gpu>/`. See [`results/README.md`](results/README.md) for what's in there.

## Directory layout

```
benchmarks/
├── README.md                                 ← this file
├── run_benchmarks.sh                         ← end-to-end harness
├── profile.sh                                ← Nsight Systems wrapper
├── analyze.py                                ← Welch's t-test joiner; emits comparison.{csv,md}
├── results/                                  ← committed result directories (raw data for the paper)
│   ├── README.md
│   └── run_*/
├── HartsyInference.Benchmarks/                ← legacy CPU benchmarks (existing; not Phase B)
├── HartsyInference.GpuBenchmarks/             ← C# GPU microbenchmarks via BenchmarkDotNet
│   ├── HartsyInference.GpuBenchmarks.csproj
│   ├── BenchmarkConfig.cs                    ← shared BDN config (1 warmup, 5 trials)
│   ├── BenchmarkFixture.cs                   ← CudaBackend + tensor allocation helpers
│   ├── Program.cs
│   ├── MatMulGpuBenchmarks.cs
│   ├── Conv2DGpuBenchmarks.cs
│   ├── NormGpuBenchmarks.cs
│   ├── SdpaGpuBenchmarks.cs
│   ├── ElementwiseGpuBenchmarks.cs
│   └── MemoryAllocFreeBenchmarks.cs
└── python-baseline/                          ← pinned PyTorch + diffusers parity scripts
    ├── README.md (TBD)
    ├── requirements.txt                      ← PyTorch 2.5.1 + cu124 etc., pinned
    ├── _common.py                            ← timing, fingerprints, CSV writer
    ├── run_all.sh
    ├── bench_pytorch_matmul.py
    ├── bench_pytorch_conv2d.py
    ├── bench_pytorch_norms.py
    ├── bench_pytorch_sdpa.py
    ├── bench_pytorch_elementwise.py
    └── bench_pytorch_e2e.py
```

## Statistical rigor

Every measurement reported in the paper is grounded in a `benchmarks/results/run_*/` directory.
Methodology:

- **Warmup**: 1 invocation discarded.
- **Trials**: N=5 per benchmark.
- **Confidence interval**: 95 % via Student-t (df=4).
- **Significance gate**: a "speedup" is reported only when (a) the new mean is outside the old 95 % CI AND (b) Welch's t-test rejects μ_new = μ_old at α = 0.01.
- **Fingerprinting**: every run captures hardware, software, and PTX/checkpoint digests — see [`results/README.md`](results/README.md).

See [`docs/Research/PROFILING_METHODOLOGY.md`](../docs/Research/PROFILING_METHODOLOGY.md) for the full procedure.

## What lives in each script

| Script | What it does |
|---|---|
| `run_benchmarks.sh` | Top-level harness: fingerprints → dotnet build → C# microbench → Python baselines → analyze.py → atomic move into `results/` |
| `profile.sh` | Wraps Nsight Systems (`nsys profile`) around a single end-to-end test; emits `.qdrep` + `nsys stats` summary |
| `analyze.py` | Joins C# and PyTorch microbench CSVs by (op, shape, dtype); runs Welch's t-test; emits comparison.{md,csv} |
| `python-baseline/run_all.sh` | Runs every PyTorch baseline script in sequence, appending to a single CSV |

## Common operations

```bash
# Smoke test the harness (1 trial each, MatMul only, no e2e)
bash benchmarks/run_benchmarks.sh --smoke

# Skip Python (e.g. to iterate on C# benches without re-running PyTorch)
bash benchmarks/run_benchmarks.sh --skip-python

# Profile a single SDXL run with Nsight Systems
bash benchmarks/profile.sh --test "FullyQualifiedName~Sdxl_GenerateImage_Gpu"

# Run only one C# benchmark class
dotnet run --no-build -c Release --project benchmarks/HartsyInference.GpuBenchmarks -- \
    --filter '*Sdpa*' --warmupCount 1 --iterationCount 5

# Run only one Python baseline (for debugging schema issues)
python3 benchmarks/python-baseline/bench_pytorch_matmul.py --output /tmp/test.csv --trials 1
```

## What does NOT belong here

- **Functional tests** (numerical correctness): those live under `tests/`. Benchmarks measure speed; tests measure math.
- **Per-checkpoint reference dumps**: those live under `tests/python-reference/`.
- **CPU benchmarks**: keep using `benchmarks/HartsyInference.Benchmarks/` (existing, BDN-based).

## When to add a new benchmark

Add a new `*GpuBenchmarks.cs` class when:
1. A new `IBackend` op is introduced
2. A model uses a shape combination not covered by the current grid
3. An optimization phase (B4.x) targets a workload pattern that needs its own microbench

When you add one, also add the matching Python script to `python-baseline/` so the comparison stays apples-to-apples.

## License + reproducibility

All result CSVs in `benchmarks/results/` are checked into the repository. They form the public reproducibility trail for the paper. The harness scripts (this directory) are part of the HartsyInference MIT-licensed source distribution — re-run them on any compatible CUDA box to reproduce or refute our numbers.
