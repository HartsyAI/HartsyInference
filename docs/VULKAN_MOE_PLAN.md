# Vulkan LLM and MoE: baseline, cause of the gap, port plan

Scope: GGUF text models on the Vulkan backend, dense and mixture-of-experts. The CUDA side it is measured against is
[MOE_ARCHITECTURE.md](MOE_ARCHITECTURE.md), "Device-resident routed stage (CUDA)". Model status per backend lives in
[VULKAN_STATUS.md](Checklists/VULKAN_STATUS.md); this file holds the measurements and the work that follows from them.

## Baseline

RTX 3060 12 GB, both backends on the same card, greedy, 128 new tokens, the same prompt, `hartsy text`. CUDA decode is
launch- and host-bound on this card and its runs vary with machine load, so the whole range is given; Vulkan was stable.
Text from the two backends was identical over the 128 tokens for both models.

| Model | CUDA tok/s (5 runs) | Vulkan tok/s (5 runs) | CUDA median / Vulkan median |
|---|---|---|---|
| Llama-3.2-1B Q8_0 | 96.5 to 147.3, median 109.7 | 15.4 to 15.8, median 15.5 | 7.1x (best CUDA run 9.5x) |
| granite-3.0-1b-a400m Q4_K_M (MoE, 32 experts, top-8) | 44.9 to 152.0, median 138.6 | 5.8 to 7.9, median 7.5 | 18x |

So the earlier "9.5x" is the best CUDA run against a typical Vulkan run; the typical dense gap is about 7x. Qwen3-4B Q4_K_M
does not fit this card on Vulkan (see the first cause below), so the dense baseline is the 1B model.

OLMoE-1B-7B Q4_K_M (4.2 GB: Q4_K gate/up and attention, Q6_K down) does not run on Vulkan on this card: it fails with
`ErrorOutOfDeviceMemory` after 11.8 GB of F32 weights are resident. It runs on CUDA.

## Why Vulkan is slow, measured

`diagnostics.vkProfileGpu=true` (timestamp queries per dispatch) and `diagnostics.vkProfile=true` (host time per op),
passed as `--set` on the CLI.

1. **Every GGUF is widened to F32 on the host before upload.** `VulkanBackend` reports `SupportsQuantized = false`, so
   `TextService` loads with `dequantizeToF32` and no weight stays compressed. A 1.3 GB Q8_0 file becomes 4.9 GB of F32 and
   7.2 GB of VRAM in use; Qwen3-4B Q4_K_M would need about 16 GB and OLMoE 7B about 27 GB. Every decode step then streams
   four to seven times the bytes a compressed read would.
2. **There is no GEMV.** On Llama-3.2-1B, 97.6% of GPU time is `matmul_tiled_f32`, 0.90 ms per call, about 59 ms of the
   roughly 65 ms a token takes. That reads 4.9 GB in 59 ms, about 83 GB/s, against 360 GB/s of device bandwidth: the
   generic tiled GEMM is run at M = 1, where most of each tile is idle. Attention is 1.1% and every other op under 0.5%.
3. **Host overhead is not the dense bottleneck.** About 392 dispatches a token, and the one host wait per token (the
   sampler's read of the logits) is the wait for the GPU itself. Host time per op totals about 1 s of the 8.4 s. This stops
   being true once the GEMV is fast: at a few ms of GPU time a token, recording 400 dispatches a token and the per-token
   drain become the next cost, which is what the step graph below is for.
4. **MoE adds host round trips and per-expert launches.** Vulkan implements `MoeRoute`, `MoeBuildDispatch`, `MoeCombine`
   and `TopKLastDim`, but nothing in the layer calls them: with no `SupportsMoeExpertIndexed`, `MoeFeedForward.Forward`
   takes the host-routed path. On granite that is 85,308 `Linear` calls (about 670 a token, one per active expert
   projection) and 22,397 host waits in `ScatterAddWeightedRows` (about 175 a token) on top of the router read per layer.
   GPU time was 8.0 s of a 17.3 s wall, so more than half the MoE run is host stall.

## What exists on Vulkan and what does not

Present: `moe_route`, `moe_build_dispatch`, `moe_combine`, `topk_lastdim`, `softplus` (parity-tested by
`VulkanMoePrimitiveTests`); `dequant_*` for Q4_0, Q5_0, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K and IQ4_XS, which write F16 and
are used by the lazy weight-cast path; the decode-step leaf ops (`kv_cache_append_dev`, `embed_gather_decode`,
`rope_decode_step`, `argmax_lastdim`, `repetition_penalty`, `history_append`, `FlashAttentionDev`); and `VulkanStepGraph`
(push descriptors, one persistent command buffer, fence-waited replay), which the diffusion denoisers already use. The LLM
decode graph is off: `GraphDecodeSupported` defaults to false, its doc comment asks for an end-to-end decode-loop parity test
before it is flipped, and no LLM path records a Vulkan step graph.

Absent (inheriting the throwing or false defaults in `IBackend.Moe.cs` and `IBackend.Sampling.cs`):

| IBackend member | Needed for |
|---|---|
| `SupportsMoeExpertIndexed`, `MoeExpertsResident` | the gate that selects the device-routed layer |
| `MoeExpertGateUp`, `MoeExpertDown` | decode and speculative verify, at most 16 tokens |
| `MoeCombineSlots` | decode combine with the shared expert |
| `SupportsMoeExpertsGrouped`, `MoeExpertsGrouped`, `MoeCombinePairs` | prefill |
| `DeviceSamplingSupported`, `AllocDeviceRng`, `FreeDeviceRng`, `SampleTopKInto` | stochastic sampling inside a captured step; greedy needs only `ArgMaxInto`, which exists |
| a grouped `PreloadWeightGroups` | experts of a projection in one contiguous allocation; Vulkan uploads each member alone |

## Steps 1 and 2: measured result

Same card, prompt and protocol as the baseline (greedy, 128 new tokens, 5 runs; CUDA with `CUDA_VISIBLE_DEVICES=1`).
"Before" is `origin/main` at 3c79b346 built fresh; peak VRAM is `nvidia-smi` memory used on the card during the run.

| Model | CUDA tok/s | Vulkan before tok/s | Vulkan after tok/s | VRAM CUDA / before / after (MiB) |
|---|---|---|---|---|
| Llama-3.2-1B Q8_0 | 142.4 (136.0 to 144.8) | 15.3 (15.1 to 15.4) | 118.7 (117.1 to 120.3) | 4123 / 7251 / 2211 |
| Qwen2.5-0.5B Q4_K_M | 120.3 (116.3 to 123.8) | 27.0 (27.0 to 27.0) | 98.8 (95.0 to 104.1) | 1561 / 2875 / 684 |
| granite-3.0-1b-a400m Q4_K_M (MoE) | 135.2 (125.0 to 143.7) | 8.0 (8.0 to 8.1) | 23.2 (22.6 to 23.3) | 3615 / 5375 / 911 |
| Qwen3-4B Q4_K_M | 79.8 (69.6 to 80.6) | does not fit | 49.0 (48.8 to 49.7) | 7981 / - / 4014 |
| OLMoE-1B-7B Q4_K_M (MoE, 3 runs) | 186.1 (178.0 to 188.5) | out of device memory | 27.1 (26.3 to 27.3) | 7819 / - / 4296 |

Output: Llama, Qwen2.5 and granite produce the same 128 tokens on Vulkan before and after, and the same as the CPU
backend. CUDA now differs from all three at about word 85 (Llama), 10 (granite) and 27 (Qwen2.5), on `main` as on this
branch, so that divergence is on the CUDA side: its decode GEMVs quantize the activation to int8, and the baseline above,
taken before #369 and #372, had CUDA and Vulkan identical. Qwen3-4B matches CUDA over all 128 tokens, OLMoE for about
100 words, and gpt2-medium Q4_K_M (fused QKV split, now kept packed) over all 128.

GEMV bandwidth, from `vkProfileGpu` timestamps: a shader's total time over the decode steps against the bytes of the
weights of that type a step reads (tied heads counted, untied embeddings and routed experts not).

| Shader | Model | Avg per call | Achieved |
|---|---|---|---|
| `gemv_quant_q8_0` | Llama-3.2-1B | 0.081 ms | about 250 GB/s |
| `gemv_quant_q4_k` | Qwen3-4B | 0.073 ms | about 190 GB/s |
| `gemv_quant_q6_k` | Qwen3-4B | 0.092 ms | about 210 GB/s |
| `gemv_quant_q5_0` | Qwen2.5-0.5B | 0.021 ms | about 115 GB/s (small rows, launch bound) |

The card's 360 GB/s is the ceiling. What the profile showed while tuning the shader: giving each lane a whole quant block
strides the activation loads by a block per lane and ran at about 50 GB/s; one or two 32-bit words of quants per lane, so
a subgroup reads one contiguous run of weights and activations, fixed that; and the activation row count has to be a
specialization constant, because indexed by a runtime bound the per-row accumulators spill to local memory (Q4_K went
from 0.100 to 0.073 ms a call). Workgroups of 64 or 256 lanes and 4-word units were both slower than 128 lanes and 2 words.

Prefill keeps the GEMM: a quantized weight at M above 8 is dequantized to F16, and the text path makes that copy per call
(`IBackend.CacheQuantizedWeightCasts` off) so it never sits resident beside the packed weight; other callers still cache it. That costs about 25 ms per prompt on Llama-3.2-1B and about 75 ms on Qwen3-4B.

What remains from steps 1 and 2:
- **Integer dot-product variant.** Quantize the activation to int8 per 32 and use `dotPacked4x8`, as the CUDA kernels use
  `dp4a`. Needs the LunarG SDK glslang. Llama's Q8_0 GEMV already reaches about 70% of bandwidth, so the gain is bounded.
- **An F32 GEMV.** Done in steps 3 and 4 below: `gemv_quant_f32`, 0.004 ms a call on the OLMoE router against 0.35 ms.
- **M from 9 to 16.** A speculative verify longer than 8 rows takes the dequantize-and-GEMM path.
- MoE decode was host-routed until steps 3 and 4 (below); prefill above 16 tokens still is, until step 5.

## Steps 3 and 4 and the F32 GEMV: measured result

RTX 3060, greedy, 128 new tokens, `hartsy text` with `-b vulkan:1`, 5 interleaved runs, median (range). "Before" is `origin/main` built fresh; peak VRAM is `nvidia-smi` on the card.

| Model | Vulkan before tok/s | Vulkan after tok/s | VRAM before / after (MiB) |
|---|---|---|---|
| granite-3.0-1b-a400m Q4_K_M (MoE) | 23.4 (20.5 to 24.1) | 127.6 (115.7 to 136.3) | 911 / 905 |
| OLMoE-1B-7B Q4_K_M (MoE) | 26.7 (25.1 to 27.1) | 132.5 (128.4 to 133.8) | 4296 / 4218 |
| Llama-3.2-1B Q8_0 | 118.2 (115.7 to 119.5) | 117.3 (110.8 to 118.4) | 2211 / 2211 |
| Qwen2.5-0.5B Q4_K_M | 94.7 (90.7 to 101.6) | 95.3 (87.3 to 103.8) | 682 / 684 |
| Qwen3-4B Q4_K_M | 49.2 (48.6 to 50.1) | 49.2 (49.0 to 49.6) | 4014 / 4014 |

Output: all five models give the same 128 tokens before and after. granite and OLMoE equal the CPU backend and the host-routed path of this branch (`numerics.moeIndexed=false`). Host waits in granite decode: 129 for 128 tokens, all the sampler's logits read, none inside a MoE layer (before: about 175 a token); the remaining waits are prefill, which is still host-routed.

Bandwidth from `vkProfileGpu` (bytes the dispatch reads over its time):

| Shader | Model | Avg per call | Achieved |
|---|---|---|---|
| `moe_expert_gate_up_q4_k` | OLMoE | 0.079 ms | about 240 GB/s |
| `moe_expert_down_q6_k` | OLMoE | 0.065 ms | about 212 GB/s |
| `moe_expert_down_q4_k` | OLMoE | 0.050 ms | about 189 GB/s |
| `moe_expert_gate_up_q4_k` | granite | 0.041 ms | about 115 GB/s (K = 1024, launch bound) |
| `gemv_quant_f32` | OLMoE router | 0.004 ms | n/a (was 0.35 ms) |

Design: one subgroup computes one output row of one (token, slot) pair, so the grid is (row groups, pairs) and the accumulators are constants; the inner loop is `gemv_quant_unit.glsl`, shared with `gemv_quant`. Workgroup sizes of 64, 128 and 256 lanes measured the same.

## Plan, in order

Each step lands as its own PR, with a parity test against the CPU reference for every new shader and a before/after
`tests/regression-ab.sh` on the flagship CUDA models only if shared code moved. Effort figures are estimates, not
measurements.

**0. Loader fix (this PR).** The per-expert and fused-QKV views of a dequantized tensor read freed memory, so any MoE GGUF
and gpt2 crashed on the first upload on a backend that widens to F32. They now root their source. Granite runs; OLMoE reaches
the device-memory limit above.

**1. Fused dequant GEMV for the dense path. Done: `gemv_quant` for Q8_0, Q4_0, Q5_0, Q4_K, Q5_K and Q6_K, plain float
math, M up to 8; results above.** The original plan: `gemv_q8_0`, `gemv_q4_k`, `gemv_q6_k`
(then Q4_0, Q5_0, Q5_K): one workgroup per few output rows, activation vector in shared memory as F32, each lane walks whole
quant blocks, dequantizes in registers with FMA, subgroup reduction to the row result, bias and residual as in
`matmul_tiled`. Q4_K and Q6_K blocks are 256 elements with packed scales; the `dequant_q4_k` and `dequant_q6_k` shaders
already hold the bit layouts. A second variant quantizes the activation to int8 per 32 and uses `dotPacked4x8` as the CUDA
kernels do with `dp4a`; it needs `GL_EXT_integer_dot_product`, which the distro `glslangValidator` 15.1.0 here cannot compile
(TROUBLESHOOTING, "Shader toolchain"), so build that variant with the LunarG SDK glslang or leave it for later. M = 2 to 16
(speculative verify) is the same shader with a column loop.
Expected effect: weight bytes per token fall to 1.3 GB for Llama Q8_0, so the arithmetic ceiling at 360 GB/s is about 270
tok/s against 15.5 today; what the shader reaches is unmeasured.

**2. Keep GGUF weights compressed on Vulkan. Done: `SupportsQuantized` is true and `QuantizedTypes` names the six
formats the GEMV reads; the loader keeps those packed and widens every other format to F32 as before, and LTX-2's resident
NVFP4 gate asks for `F4E2M1` specifically, so Vulkan still unpacks it.** The original plan: Set `SupportsQuantized = true` and keep the
loader's existing `GpuSupportedQuant` list. Types without a GEMV and prefill (M above the GEMV limit) take the existing
dequant-to-F16 then cooperative-matrix GEMM; with `CacheWeightCasts` off a 7B model cannot hold an F16 copy, so the cast is
per use and prefill pays one dequant per layer. Check the VRAM and prefill cost on Llama before turning it on by default; this
step is also what makes OLMoE and Qwen3-4B load at all on a 12 GB card.

**3. Grouped expert upload. Done: a projection's experts are one allocation, each expert a buffer bound at `i * stride` (256-byte aligned) plus a span buffer; `MoeExpertsResident` refuses a group with a freed or re-placed member.** The original plan: Override `PreloadWeightGroups` so a projection's experts share one buffer in
expert order, record base and stride, and answer `MoeExpertsResident` the way `CudaBackend` does, refusing a group whose
members were re-placed. Per-layer groups are 75 to 110 MB on OLMoE, below the 4 GiB storage-buffer range of the NVIDIA devices
(llvmpipe reports 128 MB, so a software-rasterizer test needs small shapes).

**4. Device-routed decode. Done: `moe_expert_gate_up_*`, `moe_expert_down_*` (Q8_0, Q4_K, Q6_K, Q4_0, Q5_0, Q5_K) and `moe_combine_slots`; results below.** The original plan: `SupportsMoeExpertIndexed(Q4_K | Q6_K | Q8_0)`, then:
`moe_expert_gate_up` (one workgroup per `(pair, row block)`; reads `topkIdx[pair]`, addresses `base + e * stride`, runs the
step-1 dot for gate and up against the token's activation, writes `act(gate) * up` in the same dispatch, SiLU or tanh-GELU by
specialization constant), `moe_expert_down` (same shape, the down projection per pair), and `moe_combine_slots` (the weighted
slot sum with the sigmoid-gated shared row; closely related to `moe_combine`, which already exists). The layer is then
`router GEMV, MoeRoute, GateUp, Down, CombineSlots` with no host read. Expected effect on granite: the 22k host waits go away
and the 85k per-expert launches become about 5 per layer. OLMoE reads roughly 0.8 GB a token (a top-8 of 64 slice of 4.2 GB
plus attention and the output head), so at 360 GB/s its ceiling is about 450 tok/s; that is arithmetic, not a measurement.

**5. Device-routed prefill (about a week).** `MoeBuildDispatch` exists. `MoeExpertsGrouped` needs per-expert GEMMs on
contiguous rows: dequantize the expert stacks to F16 in one dispatch per projection (the dequant shaders already treat a
stack as a flat run of blocks), then one cooperative-matrix GEMM per active expert, with the offsets read once per layer as
CUDA does. `MoeCombinePairs` is the pair-map variant of `moe_combine` plus the shared row. Prefill is where the F16 cache
question of step 2 bites: a layer's three stacks are 260 MB as Q4_K/Q6_K and about 1 GB as F16 for OLMoE.

**6. Command-buffer reuse for LLM decode (about a week, after step 4).** `VulkanStepGraph` is the Vulkan counterpart of a
CUDA graph, so the equivalent of `CaptureDecodeGraph` is: warm up one step so every weight and scratch buffer is resident
(capture refuses a cache miss), record the step against push descriptors with the device position and token id buffers the
leaf ops already take, then replay with one fence-waited submit per token. Requirements: `MoeGraphReady` true (step 4, since a
host-routed layer cannot be captured), `GraphDecodeSupported` set only after an end-to-end decode-loop parity test against
the eager path, and the capture's higher peak VRAM (every intermediate stays alive for the graph's life) checked on a 12 GB
card. This removes recording cost and the per-op submit batching decisions; whether it matters is decided by the step-1 and
step-4 timings, so measure the host share of a token before building it.

**7. Device sampler (about three days, only if sampled decode is wanted in the graph).** `AllocDeviceRng` (seed and draw
counter buffer), `SampleTopKInto` as a top-k dispatch (`topk_lastdim` exists; the vocabulary-wide case needs the two-stage
slice-then-merge form CUDA uses) plus a temperature, nucleus, min-p and multinomial draw dispatch with a splitmix64 generator.
Greedy graph decode does not need it.

## Order and reasons

Steps 1 and 2 first: they are the dense gap (about 7x) and the reason OLMoE and Qwen3-4B do not load, and every MoE kernel
reuses the step-1 inner loop. Step 3 is small and gates step 4. Steps 4 and 5 remove the MoE-specific 18x. Step 6 comes last
because it only pays once the GPU time per token is a few milliseconds. Step 7 is optional.

## How to reproduce

```
dotnet publish src/HartsyInference.Cli -c Release -f net10.0 -o <dir>
hartsy text "<prompt>" --model-path <gguf> -b vulkan:1 --temperature 0 --max-tokens 128 \
    --set diagnostics.vkProfileGpu=true --set diagnostics.vkProfileFile=<file>
```

`-b vulkan:N` indexes the loader's raw device list, which on this machine puts the RTX 4090 at 0, the RTX 3060 at 1 and
llvmpipe at 2; `-b cuda:0` with `CUDA_VISIBLE_DEVICES=1` is the same 3060. The profile is written to `<file>.gpu.txt`.
Compile shaders with `src/HartsyInference.Vulkan/Shaders/build.sh` (`glslangValidator` and `spirv-val` on PATH) and commit
the `.spv` files it writes to `Spirv/`.
