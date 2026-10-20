# Red-test ledger — origin/main e845dcd9 (2026-10-20)

RTX 3060 only (`CUDA_VISIBLE_DEVICES=<3060 uuid>`, `HARTSY_TEST_VULKAN_DEVICE=1`), Release, suites run one at a time.
Later phases gate on "no new failure by name" against this list.

| Suite | Filter | Passed | Failed |
|---|---|---|---|
| Core.Tests | CPU lane | 591 | 0 |
| Cpu.Tests | CPU lane | 295 | 1 |
| Cuda.Tests | Category=GpuIntegration | 687 | 0 |
| Vulkan.Tests | Category=GpuIntegration | 560 | 0 |
| Gpu.Tests | Category=GpuIntegration | 69 | 0 |
| LLM.Tests | attention classes (FlashAttention, FixedKvCache, Mla, GraphDecode, GenericTransformer, KvCaches, PrefixCacheReuse, Looped; TensorParallel excluded — needs two GPUs) | 79 | 0 |

Failing on main, unrelated to attention:
- `HartsyInference.Cpu.Tests.LowRankAdjunctCoverageTests.EveryBackendEntryTakingAWeightIsClassified` — IBackend gained
  `RmsNormModulate` without a LoRA-coverage classification.

Caveat: tests that `return` early when a resource is missing count as passed (xUnit has no skip from inside a fact);
the GPU and PTX were present, so the GPU suites ran their bodies. TRX files with per-test results are under `trx/`.

## Quality floors

Declared before any candidate kernel or policy exists, so a candidate's own degradation can never become its floor.
Every later `bounded` gate is judged against these.

| Level | Bar |
|---|---|
| Kernel | relL2 ≤ 6e-3 and cos ≥ 0.99998 against F32 (Comfy Kitchen's published acceptance, adopted as a fixed target) |
| Model | per flagship: the SSIM between today's default attention and every attention approximation off, minus 0.01 (table below), plus the montage check |
| LLM | teacher-forced mean \|Δ log p\| ≤ 0.02 nats and max per token ≤ 0.2 against the same strict arm, plus the greedy stream on the short rows |
| Audio | Whisper transcript equality, or PCM correlation ≥ 0.999 against the pre-change output |

The strict arm is the same build with `numerics.sdpaNoF16=true`, `numerics.sageAttn=false`, `numerics.fa2Prefill=false`
and `numerics.flashF16=false`: no F16 narrowing of an F32 call, no INT8, no F16 prefill. CUDA's F32 attention GEMMs
run TF32 whatever the settings say, so this arm is the strictest the engine can run today, not exact F32.
