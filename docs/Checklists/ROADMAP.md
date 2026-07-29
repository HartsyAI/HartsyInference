# HartsyInference — Engineering Roadmap

> The single forward-looking checklist for the whole engine: cross-cutting features and campaigns that
> aren't tied to one modality. **Per-model open work lives in the matching `MODEL_STATUS_*` doc**; this
> file is for infrastructure that spans modalities — multi-GPU, GPU-vendor support, kernel/quant perf,
> serving/CLI/API, and release. Only *open* work belongs here; when an item ships, move it to the
> relevant status doc's "done" evidence and delete the line. Distilled from the old `MULTI_GPU_*`,
> `PHASE_B_GPU_PERFORMANCE`, `*_PERF_GRIND`, `*_BENCHMARK`, `QUANT_GEMM_*`, `W8A8_HANDOFF`,
> `PRODUCTION_RELEASE_CRITERIA`, and `RELEASE_NUGET` docs (recoverable from git history).

**Legend:** `[ ]` open · `[~]` in progress · `[x]` done (delete on move) · 🔒 blocked (reason noted).

---

## 1. Multi-GPU / model sharding (headline feature)

Not "one gen per GPU" — a single large model **spread across GPUs**: big LLM/MoE servers, and stringing
together several small consumer cards (e.g. 8 GB each) into one usable pool. Unblocks the build-deferred
giants (Kimi-K2, DeepSeek-V3, Mixtral, Qwen3-MoE, Qwen2.5-VL-7B).

- [ ] **M0 — device topology + placement:** enumerate GPUs, PCIe/NVLink topology, per-device VRAM
  budget; a placement planner that assigns layers/experts to devices.
- [ ] **M1 — layer-split (pipeline) parallel:** shard transformer blocks across devices, micro-batch the
  pipeline; the entry path for "many small cards."
- [ ] **M2 — tensor parallel:** split individual GEMMs (QKV, MLP, lm_head) across devices; all-reduce on
  the seam.
- [ ] **M3 — expert parallel (MoE):** route experts to devices; ties to on-device MoE routing (§4).
- [ ] **M4 — DP-attention / sequence parallel:** for long context and video (Ulysses-style seq-parallel,
  see §2 D3).
- [ ] **M5 — disaggregated serving:** separate prefill and decode pools.
- [ ] **Collectives:** NCCL (NVIDIA) via Driver-API P/Invoke; **RCCL** for the AMD/ROCm path (§3).
- [ ] **Tiers to validate:** consumer multi-card (e.g. 2–8× 8 GB, Pascal+) and enterprise (H200-class).
- [ ] **Diffusion D1–D3:** cross-device latent/sequence parallel for video (Wan/LTX) once M4 lands.

## 2. GPU kernel performance

The dedicated benchmark harness is greenfield — build it first so every optimization is measured, not
assumed (see the profiling pitfalls in `TROUBLESHOOTING.md`).

- [ ] **Benchmark harness:** C# runner + PyTorch/llama.cpp baselines + NVTX ranges + run scripts;
  multi-device baselines (3060 / L40S / A100 / H100).
- [ ] **FlashAttention-2 PTX** (varlen/packed): `IBackend.PackedAttention` for video; general FA2 kernel.
- [ ] **cuDNN-free Winograd conv** for the conv-heavy VAEs/vision.
- [ ] **Kernel fusion** (QKV projection, gate-up, norm+activation, coopmat bias) where launch-bound.
- [ ] **Activation memory pool** + **CUDA-graph denoise** capture across the DiT/UNet loop.
- [ ] **F16/BF16 tensor-core sweep**; **FA3 / WGMMA** on Hopper.
- [ ] **Video shared-infra:** `DenoiseKvCache` (~2–3×), `DistilledFlowMatchEuler` (DMD/CM/Lightning few-step),
  `IDiscreteVideoTokenizer` (Cosmos), sparse video attention (Wan/LTX — measure-then-design).
- [ ] **Vision GPU-native ops:** `MaxPool2D`, `Conv2dDepthwise` (currently CPU); JPEG/WebP decoders.
- [ ] **3D residuals:** GPU marching cubes; DINO per-op fusion.
- [ ] **Kernel-source hygiene (maintainability):** ~28 shipped CUDA kernels (`softmax`, `groupnorm`,
  `layernorm`, `geglu`, `cast_*`, `elementwise_*`, `transpose`, `broadcast_add`, `spatial`) are hand-written
  PTX with no `.cu` source. Backfill a `.cu` for each opportunistically (verify PTX parity, then delete the
  hand-PTX); keep `hgemm_mma_sm80` hand-authored (intentional MMA). See `src/HartsyInference.Cuda/Kernels/README.md`.

## 3. AMD / ROCm + cross-vendor (Vulkan) support

The Vulkan backend exists (SPIR-V/GLSL) but is unvalidated on AMD/Intel hardware — that validation is the
whole point of the backend. See the Vulkan pitfalls in `TROUBLESHOOTING.md`.

- [ ] **AMD/Intel cross-vendor bring-up** on real hardware (🔒 needs a dual-vendor box); the "Anticipated
  Categories" AMD/Intel row is unvalidated.
- [ ] **Vulkan kernel/perf tuning:** currently ~6.5× CUDA, target ≤1.6×. Per-dispatch overhead is ~94% of
  Linear time → QKV fusion, pre-cast FP8 weights, coopmat bias fusion, vendor tile-size auto-tuner.
- [ ] **GGUF dequant shaders:** `dequant_q4_k`, `dequant_q8_0` (quant support on Vulkan).
- [ ] **FlashAttention `sdpa_flash`** SPIR-V (low priority).
- [ ] Small-subgroup `requiredSubgroupSize` pinning; im2col shader 64-bit widening (🔒 deferred — no
  SPIR-V compiler on the dev box); descriptor-pool `FlipPool` timeline wait; per-dispatch barrier scoping.
- [ ] Wire the INT8 quantizer into Vulkan model loading.
- [ ] **RCCL** collectives for multi-GPU on AMD (§1).

## 4. Quantization

Core "more quant support" axis. Note the hard-won lesson: local per-layer relL2 does **not** predict e2e
SSIM (errors add in quadrature) — always gate on end-to-end SSIM/FVD, not local diffs.

- [ ] **Native FP8 GEMM (Ada+):** e2e A/B + SSIM gate; **default-on-for-Ada decision pending user**.
- [ ] **W8A8 stage 4:** SmoothQuant fails the 0.95 SSIM gate (weight-side floor binding). Open options to
  choose: grouped-K quant on both operands · mixed-precision layer skipping · accept ~0.92 SSIM ·
  timestep-aware (NDTC) calibration. Then block-level + Swarm e2e A/B.
- [ ] **Diffusion low-VRAM GGUF** quant wiring (opt-in; 🔒 needs diffusion-GGUF weights).
- [ ] **KV-cache quantization** + paged KV (§5).
- [ ] **MMQ-class quantized-GEMM prefill kernels** — the one unsolved LLM prefill gap (GLM-4-9B TTFT).
- [ ] **In-kernel fused dequant-GEMV** (faster + zero transient); quantized embed/lm_head GPU gather.
- [ ] **Exotic quant codecs:** IQ1/IQ2/IQ3 readers.

## 5. LLM decode / serving throughput

Decode already beats llama.cpp on most models; these are the remaining levers (from the decode/throughput
grinds).

- [ ] **On-device MoE routing** — unblocks graph decode for olmoe/qwen2moe/granitemoe and fixes the D2H
  per-token readback bug (see `TROUBLESHOOTING.md` §Serving). Ties to expert-parallel (§1 M3).
- [ ] **On-device sampler** (temp/top-k/top-p) so graph decode isn't greedy-only.
- [ ] **FP16 activation pipeline** (decode Phase 3b); **QKV/Gate-Up projection fusion** (partial).
- [ ] **Big-model GEMV memory-access-pattern redesign** (the standing "big lever," partly de-risked).
- [ ] **Speculative decoding.**
- [ ] **PagedKvCache + continuous batching** (`MoeLayer` router/expert dispatch prerequisite).
- [ ] **VRAM leak on model swap** (high-sev, open); SSM decode benchmark harness.

## 6. Diffusion / accel-grind open levers

- [ ] Step-cache replicate to Chroma / HiDream (🔒 no weights); CFG-interval late-band replicate to
  HiDream / Wan.
- [ ] F16-ingest / F16-out Sage attention kernel (designed, build next).
- [ ] Wan2.2-Lightning / LTX-distilled loadable accelerators.

## 7. Robotics models (new modality — greenfield)

- [ ] Scope target robotics/action models (VLA-style) and the modality's request/result DTOs.
- [ ] Backend ops + recipe pattern; a `MODEL_STATUS_ROBOTICS.md` once the first model is scaffolded.

## 8. New SwarmUI extensions

- [ ] **3D extension** (image→mesh) — surface the 3D modality as its own Swarm extension.
- [ ] **World-model extension** — interactive/world models as a Swarm extension.
- [ ] Keep each thin over `HartsyInference.Engine` (no re-implemented load/generate orchestration).

## 9. CLI / API

- [ ] **HTTP public API reference doc** + API stability freeze (release gate).
- [ ] Wire the **T5/seq2seq generation loop** in `TextService` (currently unwired).
- [ ] Finish CLI catalog coverage + real-hardware verification runs (LLM ~26 entries, video recipes:
  `kandinsky5-video`, `cosmos-predict1-5b/13b` need `IVideoRecipe` wrappers; HunyuanVideo/LTX-2 verified
  but not surfaced). Assets/sha256 sourcing per entry.
- [ ] `--image` VLM flag, `--thinking`/`--no-thinking` real-weight verification.
- [ ] Per-provider audio unload endpoint.

## 10. Release / production hardening

From `PRODUCTION_RELEASE_CRITERIA` (~70% done) — the 1.0.0 gate.

- [ ] Full ~26-architecture verification sweep (real weights).
- [ ] Graceful shutdown / request draining; per-request timeout + max-size limits.
- [ ] `/metrics` endpoint; multi-hour SOAK test.
- [ ] Decode-round & native-crash fault isolation verified under load (see `TROUBLESHOOTING.md` §Serving).

## 11. NuGet publication

From `RELEASE_NUGET` — mostly unstarted; persist through the 1.0 release.

- [ ] Branding/metadata pass across all packages; license/readme per package.
- [ ] Quality gates (build/test/pack) in CI; symbol packages.
- [ ] Publish alpha → beta → stable progression; verify package-boundary dependency graph.
