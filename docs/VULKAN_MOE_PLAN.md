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
| quantized `Linear` (`SupportsQuantized = true`) | everything above, and the dense gap |

## Plan, in order

Each step lands as its own PR, with a parity test against the CPU reference for every new shader and a before/after
`tests/regression-ab.sh` on the flagship CUDA models only if shared code moved. Effort figures are estimates, not
measurements.

**0. Loader fix (this PR).** The per-expert and fused-QKV views of a dequantized tensor read freed memory, so any MoE GGUF
and gpt2 crashed on the first upload on a backend that widens to F32. They now root their source. Granite runs; OLMoE reaches
the device-memory limit above.

**1. Fused dequant GEMV for the dense path (the largest single win; about a week).** `gemv_q8_0`, `gemv_q4_k`, `gemv_q6_k`
(then Q4_0, Q5_0, Q5_K): one workgroup per few output rows, activation vector in shared memory as F32, each lane walks whole
quant blocks, dequantizes in registers with FMA, subgroup reduction to the row result, bias and residual as in
`matmul_tiled`. Q4_K and Q6_K blocks are 256 elements with packed scales; the `dequant_q4_k` and `dequant_q6_k` shaders
already hold the bit layouts. A second variant quantizes the activation to int8 per 32 and uses `dotPacked4x8` as the CUDA
kernels do with `dp4a`; it needs `GL_EXT_integer_dot_product`, which the distro `glslangValidator` 15.1.0 here cannot compile
(TROUBLESHOOTING, "Shader toolchain"), so build that variant with the LunarG SDK glslang or leave it for later. M = 2 to 16
(speculative verify) is the same shader with a column loop.
Expected effect: weight bytes per token fall to 1.3 GB for Llama Q8_0, so the arithmetic ceiling at 360 GB/s is about 270
tok/s against 15.5 today; what the shader reaches is unmeasured.

**2. Keep GGUF weights compressed on Vulkan (about three days after step 1).** Set `SupportsQuantized = true` and keep the
loader's existing `GpuSupportedQuant` list. Types without a GEMV and prefill (M above the GEMV limit) take the existing
dequant-to-F16 then cooperative-matrix GEMM; with `CacheWeightCasts` off a 7B model cannot hold an F16 copy, so the cast is
per use and prefill pays one dequant per layer. Check the VRAM and prefill cost on Llama before turning it on by default; this
step is also what makes OLMoE and Qwen3-4B load at all on a 12 GB card.

**3. Grouped expert upload (about two days).** Override `PreloadWeightGroups` so a projection's experts share one buffer in
expert order, record base and stride, and answer `MoeExpertsResident` the way `CudaBackend` does, refusing a group whose
members were re-placed. Per-layer groups are 75 to 110 MB on OLMoE, below the 4 GiB storage-buffer range of the NVIDIA devices
(llvmpipe reports 128 MB, so a software-rasterizer test needs small shapes).

**4. Device-routed decode (about a week).** `SupportsMoeExpertIndexed(Q4_K | Q6_K | Q8_0)`, then:
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
