# ERNIE-5.5-Flash-Base real-weight run

Checkpoint: `baidu/ERNIE-5.5-Flash-Base` at revision `028544b2ffceea189b2d8768ed2b9d09b06f2343`. FP8 e4m3 with 128x128 F32
`weight_scale_inv` grids, 128 shards, 288,434,400,632 bytes. Base model: no chat template in the repo, so the run feeds token ids.

## What is verified without the weights

- Config passes `Ernie5Config.UnimplementedFeatures` (the loader gate).
- The real `model.safetensors.index.json` maps every one of its 134,231 keys (`Real_IndexMapsEveryKey`).
- One expert and one MLP scale tensor are F32 powers of two (2^-12 and up), so the stored scales are UE8M0-valued and
  multiply the E4M3 weights (the convention `weight_scale_inv` names).
- Multimax matches `Ernie5_5ParallelLMHead.multimax` in vLLM-ERNIE term for term.
- The indexer's weight scale `head_dim^-0.5 * n_head^-0.5` matches the reference.
- Real dimensions, FP8 storage, BF16 routers/LM head/compressor/indexer projections, gated and VHA attention, and split-feature
  routing: CUDA and Vulkan match the CPU (`Cuda_/Vulkan_RealDimensionsFp8_LogitsMatchCpu`).

## What is not verified

- The indexer ReLU. The DeepGEMM `fp8_mqa_logits` kernel is not in the reference sources. The code takes the standard
  `sum_h w_h * ReLU(q_h . k)` form.
- Real-weight numerics. Nothing here compares against a reference run of the real model.
- Production precision. CUDA GEMMs default to TF32 and Vulkan GEMMs round inputs to F16 (`EnableF16Gemm`), about 1e-3 relative
  to F32. For the verification run, pin the `reference` knob profile (`KnobProfiles.Reference`, which sets `HighPrecisionGemm` on and
  `VkF16Gemm` off). The real-dimension parity tests set those two directly. How a profile is activated from the environment is not
  checked here.

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

## Open before a paid run

- License confirmation (EU/UK/South Korea exclusion, US$50M commercial threshold) and the decision to download about 288 GB to a volume.
- Chat template: the Text path requires `chat_template.jinja`, which this base checkpoint does not ship. The verification run does not use it.
