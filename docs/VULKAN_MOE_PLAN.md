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
- **M from 9 to 16.** Done in step 6 below: the shader already took the row count as a specialization constant, so the cap moved from 8 to 16.
- MoE decode was host-routed until steps 3 and 4, and prefill above 16 tokens until step 5 (both below).

## Steps 3 and 4 and the F32 GEMV: measured result

RTX 3060, greedy, 128 new tokens, `hartsy text` with `-b vulkan:1`, 5 interleaved runs, median (range). "Before" is `origin/main` built fresh; peak VRAM is `nvidia-smi` on the card.

| Model | Vulkan before tok/s | Vulkan after tok/s | VRAM before / after (MiB) |
|---|---|---|---|
| granite-3.0-1b-a400m Q4_K_M (MoE) | 23.4 (20.5 to 24.1) | 127.6 (115.7 to 136.3) | 911 / 905 |
| OLMoE-1B-7B Q4_K_M (MoE) | 26.7 (25.1 to 27.1) | 132.5 (128.4 to 133.8) | 4296 / 4218 |
| Llama-3.2-1B Q8_0 | 118.2 (115.7 to 119.5) | 117.3 (110.8 to 118.4) | 2211 / 2211 |
| Qwen2.5-0.5B Q4_K_M | 94.7 (90.7 to 101.6) | 95.3 (87.3 to 103.8) | 682 / 684 |
| Qwen3-4B Q4_K_M | 49.2 (48.6 to 50.1) | 49.2 (49.0 to 49.6) | 4014 / 4014 |

Output: all five models give the same 128 tokens before and after. granite and OLMoE equal the CPU backend and the host-routed path of this branch (`numerics.moeIndexed=false`). Host waits in granite decode: 129 for 128 tokens, all the sampler's logits read, none inside a MoE layer (before: about 175 a token); the remaining waits were prefill, which step 5 moved to the device.

Bandwidth from `vkProfileGpu` (bytes the dispatch reads over its time):

| Shader | Model | Avg per call | Achieved |
|---|---|---|---|
| `moe_expert_gate_up_q4_k` | OLMoE | 0.079 ms | about 240 GB/s |
| `moe_expert_down_q6_k` | OLMoE | 0.065 ms | about 212 GB/s |
| `moe_expert_down_q4_k` | OLMoE | 0.050 ms | about 189 GB/s |
| `moe_expert_gate_up_q4_k` | granite | 0.041 ms | about 115 GB/s (K = 1024, launch bound) |
| `gemv_quant_f32` | OLMoE router | 0.004 ms | n/a (was 0.35 ms) |

Design: one subgroup computes one output row of one (token, slot) pair, so the grid is (row groups, pairs) and the accumulators are constants; the inner loop is `gemv_quant_unit.glsl`, shared with `gemv_quant`. Workgroup sizes of 64, 128 and 256 lanes measured the same.

## Step 5: device-routed prefill, measured result

RTX 3060, `hartsy text` with `-b vulkan:1`, greedy, a 750-token prompt (granite) and a 754-token prompt (OLMoE), 5 interleaved runs, median (range). "Before" is `origin/main` at 64dca4e3 built fresh. The prefill time is the wall time of a one-token generation on the long prompt minus that of a one-token generation on a two-token prompt, so model load and start-up cancel; host waits are the blocking waits `diagnostics.vkProfileGpu` records over the same one-token run (prefill plus the first sampled token); peak VRAM is `nvidia-smi` on the card during a 128-token run.

| Model | Prefill before (s) | Prefill after (s) | Prefill tok/s before / after | Host waits before / after | Peak VRAM before / after (MiB) |
|---|---|---|---|---|---|
| granite-3.0-1b-a400m Q4_K_M (MoE) | 2.81 (2.67 to 2.86) | 1.30 (0.96 to 1.49) | 267 / 577 | 1692 / 58 | 1017 / 1079 |
| OLMoE-1B-7B Q4_K_M (MoE) | 3.47 (2.91 to 4.53) | 1.96 (1.39 to 3.49) | 217 / 385 | 2068 / 57 | 4465 / 4595 |

The range is wide because another agent shares the card between runs; the first clean set (before the dequant workspaces below) gave 2.85 to 1.45 s and 3.50 to 1.99 s. Decode is unchanged (granite 94.2 to 94.9 tok/s, OLMoE 66.8 to 67.0 on these long-prompt runs). The waits left are the layer's one read of the E + 1 offsets (24 and 16), the sampler's read, and allocator reclaims, none of them per expert; before, they were the router read and each expert's rows. Dense Qwen2.5-1.5B Q8_0 on the same 769-token prompt: identical text, 38.6 to 38.8 tok/s.

Output over 128 greedy tokens after a 750-token prompt: granite is identical on the device-routed path, the host-routed path (`numerics.moeIndexed=false`), `origin/main` and the CPU backend. OLMoE is identical across those three Vulkan runs and differs from the CPU backend at word 57 ("more accessible to the general public" against "accessible to a wider audience"); the host-routed path and `origin/main` produce the same text as the device path, so the split is the CPU's F32 against the GPU's F16 GEMMs at a near-tie, not the new path.

Design: `MoeExpertsGrouped` gathers the activation rows into expert-major order as F16 (`moe_gather_rows_f16`), then for each expert that received rows dequantizes its gate, up and down to F16 into three layer-long workspaces of one expert each (a cached cast is used if one exists), runs the cooperative-matrix GEMMs with F32 accumulation and F32 outputs, applies the activation in `moe_act_mul_f16`, and writes the down GEMM straight into the expert-major output at the expert's row offset. Dequantizing only the active experts keeps the transient at three experts' F16 weights (12 MB for OLMoE) instead of the 1 GB of a layer's F16 stacks; an earlier version that allocated a fresh buffer per projection per expert paid 267 blocking waits on OLMoE for allocator reclaims, which the workspaces removed. The op is available where the device has cooperative matrices (`SupportsMoeExpertsGrouped`); the register-tiled GEMM cannot write an F32 output at a row offset, so a device without them keeps the host-routed layer. `MoeCombinePairs` is `moe_combine_slots` through the pair map. GPU time in a 750-token granite prefill is now 86% `sdpa_flash_f32` (22 ms a call) and 11% the grouped experts, so prefill is attention-bound from here.

## Step 6 and M 9 to 16: measured result

**Does it pay: the host share of a token.** `diagnostics.vkProfile=true` on Llama-3.2-1B Q8_0 (after step 1, 8.6 ms a token): the host spends about 660 ms in ops over the 128 tokens and the prefill, roughly 5 ms a token, against 8.6 ms for the whole token, so decode is host-bound (recording about 390 dispatches a token and the per-op batching) and the GPU waits on the host. Replay removes that recording.

RTX 3060 (`-b vulkan:1`), greedy, 128 new tokens, `hartsy text`, 5 interleaved runs, median (range). "Before" is `origin/main` at 64dca4e3 (before the device-routed prefill PR; decode is unaffected by it) built fresh; "eager, this branch" is the branch with `numerics.vkLlmGraph=false`; "graph" is the graph decode path (`--graph-decode`, or the default for a request of 512 or more new tokens). The tok/s is the decode rate the CLI reports, which leaves out the one-time cost below. Peak VRAM is `nvidia-smi` memory used on the card minus the idle baseline.

| Model | Before tok/s | Eager, this branch tok/s | Graph tok/s | VRAM before / graph (MiB) |
|---|---|---|---|---|
| Llama-3.2-1B Q8_0 | 119.9 (118.9 to 120.7) | 119.5 (106.8 to 120.5) | 158.7 (155.1 to 159.3) | 2163 / 3165 |
| Qwen2.5-0.5B Q4_K_M | 104.2 (93.1 to 113.6) | 98.3 (94.7 to 116.8) | 214.8 (210.8 to 216.9) | 636 / 1155 |
| granite-3.0-1b-a400m Q4_K_M (MoE) | 110.7 (107.5 to 111.8) | 115.2 (112.2 to 120.5) | 261.5 (258.4 to 275.1) | 855 / 1241 |
| OLMoE-1B-7B Q4_K_M (MoE) | 134.2 (124.8 to 139.6) | 135.6 (134.5 to 141.8) | 168.7 (168.1 to 171.2) | 4138 / 4531 |
| Qwen3-4B Q4_K_M | 49.8 (49.3 to 50.1) | 49.6 (49.4 to 50.1) | 58.4 (58.3 to 58.6) | 3966 / 5262 |

The gain is the host share: it is largest where the GPU time per token is smallest (Qwen2.5-0.5B 2.1x, granite 2.4x) and smallest on Qwen3-4B (1.17x), which is GPU-bound at about 17 ms a token even with the graph. No model is slower, and the eager path of the branch matches `origin/main` within the run-to-run range (the card is shared, so the low runs of the eager columns are another process).

Output: the 128 tokens are identical to `origin/main` on all five models in all 15 runs of each (granite stops on its end token before 128; the others run to the length limit). A repetition penalty of 1.15 on Llama gives the same text before and after, and a sampled run (temperature 0.8) takes the eager loop, since Vulkan has no device sampler (step 7).

**One-time cost, and why the default needs 512 tokens.** The first capture in a process uploads the embedding table (F32 for every dtype except Q8_0, Q5_0, Q4_K and Q6_K, which are gathered quantized) and warms a step. The per-model numbers in the two paragraphs below supersede the first-capture column of this table for Llama, Qwen2.5, granite and Qwen3-4B. Extra wall time of a fresh `hartsy text` run with the graph (8 new tokens) over eager, and the tokens the graph needs to repay it at its per-token saving:

| Model | One-time wall cost | Saving per token | Break-even |
|---|---|---|---|
| Llama-3.2-1B Q8_0 | 0.95 s | 2.1 ms | about 460 tokens |
| Qwen2.5-0.5B Q4_K_M | 0.36 s | 5.5 ms | about 65 |
| granite-3.0-1b-a400m | 0.18 s | 4.9 ms | about 37 |
| OLMoE-1B-7B | 0.37 s | 1.5 ms | about 255 |
| Qwen3-4B Q4_K_M | 1.39 s | 3.0 ms | about 460 |

So a one-shot 128-token CLI run is slower in wall time with the graph on the models that still gather from an F32 table (Llama, Qwen2.5, granite and Qwen3-4B no longer do, since their tables are gathered quantized; the reported tok/s does not show the cost), which is why the implicit default applies only to requests of at least `TextGenerationPipeline.ImplicitGraphMinTokens` (512) new tokens; a long-lived host that wants it always sets `numerics.graphDecode`, and then pays the cost once per model. Replaying a graph in a process that already holds the table costs only the warm step and the recording. Gathering from the quantized table removed the Llama cost (below) and, with the Q5_0 and Q6_K kernels, the F32 copy for Qwen2.5, granite and Qwen3-4B; building the F32 table on the device for the remaining dtypes is not done.

**The graph gathers a Q8_0 or Q4_K embedding table from its quantized form.** The graph used to keep an F32 copy of the embedding table (`vocab * hidden * 4`), which was the whole VRAM growth of the graph and most of its first-capture cost. For a Q8_0 or Q4_K table the on-device gather (`embed_gather_quant_q8_0` / `embed_gather_quant_q4_k`) dequantizes one row straight from the quantized table with the codec's arithmetic and applies the embedding scale, so no F32 copy is made. The Q5_0 and Q6_K kernels came later (below); every other dtype keeps the F32 path, and so does CUDA. Measured on the RTX 3060 (`-b vulkan:1`, `origin/main` built fresh, this branch built fresh), graph decode, 8-token runs, fresh process, 3 interleaved reps:

| Model (embed dtype) | Graph VRAM before / after (MiB, per-process peak) | Eager VRAM, branch (MiB) | First-capture cost before / after (s, graph8 minus eager8, median) |
|---|---|---|---|
| Llama-3.2-1B (Q8_0, tied) | 3144 / 2144 | 2014 to 2144 | 1.00 / 0.15 |
| Qwen2.5-0.5B (Q5_0, untied) | 1136 / 1134 | 598 to 614 | 0.46 / 0.52 (F32 path, unchanged) |
| granite-3.0-1B-a400m (Q6_K, tied) | 1224 / 1224 | 829 | 0.25 / 0.10 (F32 path, unchanged) |

The graph's own overhead on Llama is within the eager run-to-run range (the 128-token peaks are 2144 MiB for both eager and graph). Qwen and granite are not Q8_0 or Q4_K tables, so their numbers only show that nothing moved; the F32 copy still costs about 0.5 GB on Qwen (its untied F32 table) and 0.2 GB on granite. The Q4_K kernel's only real-model check is OLMoE-1B-7B (untied Q4_K table), in the five-model text check below. The 128-token text is byte-identical between eager and graph, and between `origin/main` and this branch, on all three models. Break-even and `ImplicitGraphMinTokens` are not re-measured here; with the Llama first-capture cost gone, the 512-token default is conservative and should be re-measured before it changes.

**The graph gathers Q5_0 and Q6_K embedding tables too.** `embed_gather_quant_q5_0` and `embed_gather_quant_q6_k` are the same one-element-per-thread gather as the Q8_0 and Q4_K kernels. Each element follows its codec's operation order (`d * (q - 16)` for Q5_0; `(d * sc) * q` for Q6_K, whose FP16 scale is the last two bytes of the 210-byte super-block) and then takes the embedding scale. `EmbedGatherQuantDecodeStep_MatchesCpuDequantBitExact` covers both formats (rows 0, 1, 42 and vocab-1; scales 1 and 12; compared as bit patterns), and all eight cases pass on the RTX 3060. The affected models are Qwen2.5-0.5B Q4_K_M (`token_embd` Q5_0, untied), granite-3.0-1b-a400m Q4_K_M (Q6_K, tied) and Qwen3-4B Q4_K_M (Q6_K, tied).

Measured on the RTX 3060 (`-b vulkan:1`), `origin/main` at 9f8994a3 built fresh against this branch, interleaved, 128-token greedy runs, 3 reps. VRAM is the per-process peak from `nvidia-smi --query-compute-apps` filtered by pid, sampled every 200 ms.

| Model (token_embd) | Eager, main / branch (MiB) | Graph, main (MiB) | Graph, branch (MiB) | Branch graph minus eager (MiB) |
|---|---|---|---|---|
| Qwen2.5-0.5B Q4_K_M (Q5_0, untied) | 614 / 616 | 1136 | 706 | +90 |
| granite-3.0-1b-a400m Q4_K_M (Q6_K, tied) | 837 / 837 | 1224 | 839 | +2 |
| Qwen3-4B Q4_K_M (Q6_K, tied) | 3946 / 3946 | 5432 | 3949 | +3 |
| Llama-3.2-1B Q8_0 (tied, control) | 2144 / 2144 | 2144 | 2144 | 0 |
| OLMoE-1B-7B Q4_K_M (Q4_K, untied, control) | 4150 / 4150 | 4208 | 4208 | +58 |

granite and Qwen3-4B now sit within 3 MiB of eager. Qwen2.5 is 90 MiB above it, which misses a 50 MiB target; the next paragraph says why. OLMoE already carried the same kind of cost on `origin/main`.

**An untied quantized table costs its own size on the graph path.** Eager gathers `token_embd` on the host and never uploads it for an untied model, so a device gather has to upload the quantized table. Qwen2.5-0.5B's Q5_0 table is 151936 × 28 blocks × 22 bytes, about 89 MiB, which is the +90 above; OLMoE's Q4_K table (50304 × 8 super-blocks × 144 bytes, about 55 MiB) is the +58 that `origin/main` has had since #393. Removing that upload means dequantizing the one needed row on the host into a host-visible buffer that the captured step reads: one submit per token and no table on the device. That is not done here. It is the follow-up for this step: implement the host-staged row for untied quantized tables, then measure Qwen2.5-0.5B and OLMoE against the 50 MiB target and the per-token host cost.

First-capture cost, the wall clock of a fresh 8-token process, graph minus eager, paired within each arm and rep (seconds). The 3-pair median is the measure this step was asked to report; the 7-pair column is the same measure over more pairs.

| Model | Main, 3 pairs | Branch, 3 pairs | Main, 7 pairs | Branch, 7 pairs |
|---|---|---|---|---|
| Qwen2.5-0.5B Q4_K_M (Q5_0, untied) | 0.83 | 0.03 | 0.72 | 0.28 |
| granite-3.0-1b-a400m Q4_K_M (Q6_K, tied) | 0.11 | 0.01 | 0.24 | -0.04 |
| Qwen3-4B Q4_K_M (Q6_K, tied) | 1.57 | -0.12 | 1.45 | -0.04 |
| Llama-3.2-1B Q8_0 (control, path unchanged) | 0.05 | -0.03 | 0.05 | -0.03 |
| OLMoE-1B-7B Q4_K_M (control, path unchanged) | 0.24 | 0.00 | not run | not run |

Single pairs on the unchanged Llama path range from -0.24 to +0.18 s, so this method does not resolve differences below about 0.25 s. Qwen3-4B's first capture loses the 1.56 GB F32 upload, about 1.5 s. Qwen2.5's drop is clearer on the 3-pair median than on the 7-pair one. granite's is within the noise.

The 128-token greedy text is identical between eager and graph, and between `origin/main` and this branch, on all five models (Llama-3.2-1B Q8_0, Qwen2.5-0.5B, granite, OLMoE-1B-7B, Qwen3-4B), in all 12 runs of each (2 builds, 2 modes, 3 reps); every run stopped on the length limit. OLMoE's Q4_K table is the real-model case for the Q4_K kernel.

Decode rate, the CLI's `decode_tok_s` for the graph: Qwen2.5 217.3 on `origin/main` and 216.5 on this branch (3 reps); Qwen3-4B 58.2 and 58.0 (3 reps); granite 269.4 and 262.9 (7 reps; the ranges, 267.5 to 272.2 and 261.4 to 274.4, overlap, so the 2% is not resolved). Eager runs are unchanged within their run-to-run spread.

**Gates and what the graph refuses.** `IBackend.LlmGraphDecodeSupported` (push descriptors and the `numerics.vkLlmGraph` kill-switch) is separate from `GraphDecodeSupported`, which stays false: the SSM, CSM and MusicGen paths gate on that one and were never validated against a Vulkan capture. `LlmGraphDecodeDefault` makes graph decode the default for requests of 512 or more new tokens that do not ask either way; an explicit `GraphDecode`/`numerics.graphDecode` still throws on a capture failure, a default one logs a warning and decodes eagerly. The dynamic-batch scheduler keeps its explicit opt-in. Shapes the device-position kernels do not cover fail at capture instead of attending over zero keys: an F16 KV cache (`vram.kvF16`), a sliding window or soft-cap (Gemma-2/3), and head dims above `FlashMaxHeadDim`; each of those then takes the eager loop. Capture needs every weight resident (a miss throws), which the decode preload provides.

**Bugs the parity work found** (each invisible to the per-op tests): the o-projection receives the head-major attention output `[1, heads, 1, headDim]`, and Linear took K from its last dim (16) instead of the weight's, so the whole layer was wrong; `CopyTo` ran on the normal stream during capture, so QK-norm models copied once at capture and never on replay; and the interface's full-width QK-norm composition (OLMoE) roped one giant head. `VulkanLlmGraphDecodeTests` now runs a synthetic Llama, Qwen2, Qwen3 and OLMoE shape through a captured replay and compares the hidden state to the eager forward (exactly equal) and the greedy token loop; the MoE shapes are checked end to end on granite and OLMoE above, not in a unit test.

**M from 9 to 16.** `gemv_quant` takes the row count as a specialization constant, so no shader changed: `QuantGemvMaxRows` went from 8 to 16 (the same cap as the MoE indexed path, so a verify round is on the fused path in the dense and MoE layers). `VulkanQuantGemvTests` gained cases for 9 to 16 rows on every format and the F32 matrix. Prompt-lookup speculative decode (draft 8, so a 9-row verify), graph off, same 128-token prompt, median of 3 to 4 interleaved runs:

| Model | Spec decode before tok/s | After tok/s | Text equal to plain greedy |
|---|---|---|---|
| Llama-3.2-1B Q8_0 | 122.1 | 149.3 | yes |
| Qwen2.5-0.5B Q4_K_M | 201.6 | 244.6 | yes |
| granite-3.0-1b-a400m Q4_K_M | 259.0 | 308.5 | yes |
| OLMoE-1B-7B Q4_K_M | 118.1 | 139.5 | yes |
| Qwen3-4B Q4_K_M | 40.5 | 45.4 | yes |

Speculation only pays on repetitive text: this prompt repeats a sentence, and for Qwen3-4B and OLMoE the speculative run (45.4 and 139.5) is still not faster than plain greedy with the graph (58.4 and 168.7) or even without it (47.2 and 135.8), so speculative decode stays opt-in (`numerics.specDecode`) and is not combined with the graph (the pipeline picks the graph when both apply).

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

**5. Device-routed prefill. Done: `SupportsMoeExpertsGrouped`, `MoeExpertsGrouped` and `MoeCombinePairs`, with `moe_gather_rows_f16`, `moe_act_mul_f16` and `moe_combine_pairs`; results below.** The original plan: `MoeBuildDispatch` exists. `MoeExpertsGrouped` needs per-expert GEMMs on
contiguous rows: dequantize the expert stacks to F16 in one dispatch per projection (the dequant shaders already treat a
stack as a flat run of blocks), then one cooperative-matrix GEMM per active expert, with the offsets read once per layer as
CUDA does. `MoeCombinePairs` is the pair-map variant of `moe_combine` plus the shared row. Prefill is where the F16 cache
question of step 2 bites: a layer's three stacks are 260 MB as Q4_K/Q6_K and about 1 GB as F16 for OLMoE.

**6. Command-buffer reuse for LLM decode. Done: `CaptureGraph`/`LaunchGraph`/`DisposeGraph` on Vulkan record a decode step onto its own command buffer and replay it with one fence-waited submit per token; on by default for greedy decode, with an eager fallback; results below.** "Default" means greedy only: a sampled request (temperature, top-k or top-p) decodes eagerly, because there is no device sampler yet. The original plan:
`VulkanStepGraph` is the Vulkan counterpart of a
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
