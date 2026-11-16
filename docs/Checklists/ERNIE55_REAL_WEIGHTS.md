# ERNIE-5.5-Flash-Base real-weight run

Checkpoint: `baidu/ERNIE-5.5-Flash-Base` at revision `028544b2ffceea189b2d8768ed2b9d09b06f2343`. FP8 e4m3 with 128x128 F32
`weight_scale_inv` grids, 128 shards, 288,434,400,632 bytes. Base model: no chat template in the repo, so the harness test below feeds token ids and the Text path
sends a plain completion prompt.

## What is verified without the weights

- Config passes `Ernie5Config.UnimplementedFeatures` (the loader gate).
- The real `model.safetensors.index.json` maps every one of its 134,231 keys (`Real_IndexMapsEveryKey`).
- One expert and one MLP scale tensor are F32 powers of two (2^-12 and up), so the stored scales are UE8M0-valued and
  multiply the E4M3 weights (the convention `weight_scale_inv` names).
- Multimax matches `Ernie5_5ParallelLMHead.multimax` in vLLM-ERNIE term for term.
- The indexer's weight scale `head_dim^-0.5 * n_head^-0.5` matches the reference.
- Real dimensions, FP8 storage, BF16 routers/LM head/compressor/indexer projections, gated and VHA attention, and split-feature
  routing: CUDA and Vulkan match the CPU (`Cuda_/Vulkan_RealDimensionsFp8_LogitsMatchCpu`).

## Real-weight runs (CUDA)

An earlier run compared prefill numerics: every decoder layer's residual stream matched a CPU torch reference at cosine >= 0.999 for a
5-token prefill (`tools/ernie5-reference`). It is not repeated here.

The Text-service run below used the CUDA tests directly, not `scripts/ernie55_pod_run.sh`, so the runbook's size and SHA check and its
`Real_PrefillAndDecode_ProduceFiniteLogits` timing line were not run. RunPod EU-RO-1, image `kalebbroo/hartsy-gpu-runner:cuda13.0-ssh`,
RTX 4090 (24 GB, sm_89), driver 595.91.07, CUDA 13.2, 251 GB host RAM, checkpoint on the network volume, repo at `1779786fb`. Host RSS
was not measured.

- Tokenizer: `Real_CompletionPrompt_MatchesTheHfTokenizer` matches the Hugging Face `tokenizers` ids on all 12 cases from
  `tests/python-reference/dump_ernie55_tokenizer_cases.py`.
- GPU parity: all 8 `Ernie5ModelGpuParityTests` CUDA tests pass (F32 and FP8), including `Cuda_TextService_Fp8_LoadsOnTheDeviceAndMatchesCpu`.
- Generation: `Real_TextService_GeneratesFromAPlainPrompt` (greedy, 16 tokens, stop reason Length) continues "Large language models are"
  (5 prompt ids) with " a type of artificial intelligence that can generate human-like text. They are trained on".

| Stage | Time |
| --- | --- |
| Load (dense preload 7644 MB of a 20025 MB budget, 48.5 s of it) | 140.5 s |
| Prefill, 5 tokens | 174.5 s |
| Decode, 15 steps | 208.8 s (13.9 s per token, 5.5 to 17.8 s) |
| Total with load (measured; includes about 1 s outside the stages above) | 525 s |

Peak VRAM was about 22.5 GB; the device weight total grew from 9216 MB to over 20 GB during decode as routed experts were uploaded. The
preload budget applies only at load, so nothing bounds that growth: a longer generation or a smaller card can run out of device memory
mid-decode. Prefill takes about 12 decode steps' time for 5 tokens, likely because it is the first read of every layer's experts from the
network volume; not profiled. Decode time is spent in the layer forwards (build time 0 ms); routed-expert upload is the likely cost, but no
profile has confirmed it. Not tuned yet.

## What is not verified

- The indexer ReLU. The DeepGEMM `fp8_mqa_logits` kernel is not in the reference sources. The code takes the standard
  `sum_h w_h * ReLU(q_h . k)` form.
- Decode and MTP numerics on real weights. Only the 5-token prefill is compared against a reference (above).
- Production precision. CUDA GEMMs default to TF32 and Vulkan GEMMs round inputs to F16 (`EnableF16Gemm`), about 1e-3 relative
  to F32. `Ernie5RealCheckpointTests` pins the same two settings as the `reference` knob profile
  (`KnobProfiles.Reference`: `HighPrecisionGemm` on, `VkF16Gemm` off), so the verification run needs no extra configuration. Production
  defaults (TF32, F16 GEMM) are not measured by this harness.

## Run

Needs the full checkpoint in one directory (`config.json`, `model.safetensors.index.json` and the 128 shards), a CUDA GPU with at least
24 GB free, and the PTX built. The model does not fit in VRAM, so each forward streams the layers from the volume and uploads
their packed FP8 weights per call.

```bash
ERNIE_REAL_DIR=/path/to/ERNIE-5.5-Flash-Base \
ERNIE_REAL_DEVICE=cuda \
ERNIE_REAL_TOKENS=1,2,3,4,5 \
dotnet test tests/HartsyInference.Engine.Tests --filter "FullyQualifiedName~Ernie5RealCheckpointTests" --logger "console;verbosity=normal"
```

Expected: `Real_IndexMapsEveryKey` passes, and `Real_PrefillAndDecode_ProduceFiniteLogits` passes with finite, non-constant logits and
prints load, prefill and decode times. Two forwards read about 270 GiB each from the volume, so budget on the order of tens of minutes
of I/O on a network volume (an estimate, not a measurement).

For a rented pod (volume size, the pinned download, the GPU and PTX checks, and the PR record), follow
[ERNIE55_POD_RUNBOOK.md](ERNIE55_POD_RUNBOOK.md); `scripts/ernie55_pod_run.sh` runs its steps in order.

## Open

- Decode speed: profile a decode step and tune routed-expert residency.
- MTP speculative decoding in the Text path, and decode and MTP parity on real weights.
- A host or device memory guard before the load, and a bound on routed-expert device memory during decode.
- License confirmation (EU/UK/South Korea exclusion, US$50M commercial threshold) for any use beyond this verification.
- Chat template: this base checkpoint ships none, so the Text path sends a plain completion prompt with the `tokenizer.json` post-processor's
  `<|ERNIE_MASK|>` first, as `/v1/completions` does.
