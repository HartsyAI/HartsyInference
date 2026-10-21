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
- `HartsyInference.Diffusion.Tests.HiDreamDitShardingTests.ForwardSharded_MatchesUnsharded_BitParity` (all five
  `splitBlock` cases, 249/256 values differ). Found running the DiT sharding classes on the 3060; same count on main and
  on the stack.

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

Measured on the 4090, default arm vs strict arm on the same build: origin/main e845dcd9 (`4090/floors2.log`) and, for the
pinned H3 row and the LLM rows, 27e36bf9 (`4090/floors3.log`). Compare a row only with a later run of the same row. A floor is
the min SSIM over the seed pairs minus 0.01, rounded down to three decimals; a row at 1.000000 ran no approximation at its
shapes, so its floor is `identical`. Step ms is the harness's median over the warm seeds.

| Case | Default vs strict, min SSIM | Floor | Step ms, default → strict |
|---|---|---|---|
| sd15 | 0.998215 | 0.988 | 69 → 82 |
| krea2 | 0.966578 | 0.957 | 523 → 520 |
| zimage | 0.936570 | 0.927 | 291 → 290 |
| qwenimage-q4k | 0.988030 | 0.978 | 1621 → 2348 |
| sdxl | 0.998141 | 0.988 | 259 → 364 |
| flux1-dev | 1.000000 | identical | - |
| wan5b | 0.994821 | 0.985 | 1124 → 1863 |
| ideogram4 | 0.894457 | 0.884 | 821 → 946 |
| chroma-hd | 1.000000 | identical | - |
| lumina2 | 0.999098 | 0.989 | 688 → 1059 |
| ltx25-dev | 0.964689 | 0.955 | 742 → 764 |
| h3-fl2va-fp8, SDPA route pinned | 1.000000 | identical | 1656 → 1363, under the pin (not production speed) |

- ideogram4's 0.894 is the lowest; both arms were inspected and both images are clean.
- H3 unpinned read 0.39 because free VRAM moved the materialized/tiled choice between arms. Pinned
  (`vram.sdpaForceTiled=true`, `numerics.sdpaTile=256`, `vram.h3ChunkRows=1000000`), the arms are identical. The pin forces
  the tiled F32 path, so that row's step ms says nothing about production speed.

| LLM case | Mean \|Δ log p\| | Max \|Δ log p\| | Top-token flips | Within the bar |
|---|---|---|---|---|
| llama32-1b | 0.0146 | 0.114 | 2/256 | yes |
| qwen3-4b | 0.0145 | 0.238 | 1/256 | no: the max already exceeds 0.2 today, so 0.238 is this row's max floor |
| gemma2-2b | 0 | 0 | 0/256 | identical |

The log's `FAIL` on llama32-1b and qwen3-4b is the harness's equality check, not the floor. The flips are today's
default against strict on the forced tokens; a later bounded change may not add flips beyond these counts.
