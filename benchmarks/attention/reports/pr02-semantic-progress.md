# PR 02 semantic correctness progress

PR 02 is a stacked draft on PR 01, started under the user's subsequent authorization. PR 01's exit evidence
remains incomplete. This report does not freeze Bfix, approve later work, or claim primary-workload speed gains.

The subsequent [4090 evidence review](pr01-4090-review.md) records a Krea CUDA identity failure and incomplete
operation non-inferiority evidence. They block merge despite the reported 45 passing semantic cases. The active
P2 production pin remains immutable while the corrected harness and first-divergence diagnostics are audited.

## Changes and reproduced defects

- CPU FlashAttention now preserves a NaN normalization result and treats a negative-infinite sink as absent.
  Two empty-row sink controls failed before the fix; all five scalar controls pass afterward.
- CPU dense (`cpu.simd`) attention returned NaN on a fully excluded row: its softmax subtracted a −∞ maximum from
  −∞ scores. It now returns zero probabilities for an all-−∞ row while a NaN score still propagates. A direct
  control with a fully excluded row, a partly visible row and an excluded-plus-NaN row failed before the change.
- Both CUDA monolithic Flash variants now apply softcap before ALiBi. Four causal/noncausal and F32/F16 KV
  controls failed before the change, each with approximately 2.01 absolute output error.
- CUDA materialized F32/BF16 attention returned NaN on fully excluded rows; F16 returned the mean value, 13,
  because it clamped the F32 exclusion bias. Those three controls failed before the change.
- Materialized and query-tiled attention now use an attention-owned softmax kernel. It returns zero probabilities
  for all-negative-infinite rows and propagates non-finite normalization. Masked materialized attention adds F32
  bias inside this kernel, removing the clamp/cast buffers and separate add launches. General-purpose softmax is
  unchanged. No additional quadratic workspace is introduced.
- Native cuDNN F16 and BF16 pass the tested exclusion, large finite bias, NaN and positive-infinity cases on
  cuDNN 9.21.0. No cuDNN arithmetic patch or admission rejection was needed for those executed cases. This is
  cohort-specific evidence, not a guarantee about untested cuDNN versions.
- Vulkan scalar F32 passes the exclusion and finite/non-finite bias controls without source or SPIR-V changes.
  These assertions require the executed GPU implementation; a host fallback cannot satisfy them.
- Review controls reproduced NaN-to-zero loss in split decode with both KV formats and in both standalone/fused
  combine kernels. The combine now preserves NaN, including in the fused Q8 scale/sum metadata consumed by the
  next projection. Direct finite/empty controls and a NaN in only one value channel verify the sidecar boundary.
- Both native half-prefill epilogues now explicitly preserve a NaN denominator. The tested NaN query already
  propagated through their accumulators before this guard; it was not a reproduced zero-output defect there.
- Finite mask sentinels are additive biases, not exclusions. With nonzero scores plus a constant `-1e30` or
  `-float.MaxValue` F32 bias, addition rounds these scores equal and the row returns the uniform mean.
  Removing the old F16 clamp can change those outputs; do not treat them as an identity-preserving refactor.
  Only actual negative infinity represents the exclusion tested here. New F32/F16/BF16 controls pin finite
  sentinel behavior and separate nonzero-score/ordinary-bias mixing at the existing dtype tolerances.

## Krea CUDA identity failure: diagnosis

The 4090 report's P2 Krea CUDA image failed the identity gate while Vulkan stayed identical. Source tracing, then
a direct kernel control, located the mechanism before any candidate-output tolerance was considered:

- Krea 2's Qwen3-VL-4B text encoder runs strict-F32 SDPA (no `allowF16`) with a finite `-1e30` causal mask, so on
  CUDA it takes the materialized or query-tiled rows. Its DiT attention is unmasked grouped token-major cuDNN F16,
  which PR 02 does not change. The Vulkan scalar path has no PR 02 source change.
- PR 02 replaced the legacy `softmax_f32`/`softmax_f16` PTX on those rows with `attention_softmax`. Both compute
  `ex2.approx((x - max) * log2e)` with the same 256-thread block and reduction order, but the new kernel normalized
  with `rcp.rn` (`1.0f / sum`) where the legacy kernel used `rcp.approx`. Every finite F32 row therefore changed in
  its last bits, masked or not; F16 storage rounding hid the difference there.
- `AttentionSoftmaxReproducesLegacySoftmaxBitsOnFiniteRows` compares the two kernels bit for bit on 37×1031 rows
  (F32 masked with a `-1e30` sentinel, F32 unmasked, F16 unmasked). With the previous PTX both F32 cases fail at
  the first normalized row; with `rcp.approx` restored all three pass on the RTX 3060. Fully excluded rows still
  give zero and NaN still propagates.

This explains a CUDA-only, all-pixel identity break. It does not by itself clear the Krea gate: a real Krea 2
Turbo generation must still reproduce `main` exactly on the frozen request, and the 4090 cohort must rerun it.
The identity gate is unchanged; no tolerance was derived.

## Local validation

The new GPU semantic class covers 45 cases: softcap/ALiBi ordering, empty local windows with non-finite sinks,
mixed excluded/visible rows, large F32 biases, NaN/+infinity, and multiple batches/heads with a 516-key reduction
tail. Existing dtype tolerances are preserved; these analytic controls do not establish model quality.

Full affected GPU categories run sequentially on Windows 11, RTX 3060 (SM86), driver 610.88:

| Project | Reported passes | Failures | Resource skips |
|---|---:|---:|---:|
| Core | 19 | 0 | 0 |
| LLM | 101 | 0 | 19 |
| Diffusion | 18 | 0 | 2 |
| Engine | 12 | 0 | 10 |

Four reported passes are guarded two/eight-GPU NCCL/Kimi checks and remain unvalidated. Real-weight resource
skips remain unvalidated too. The standalone Cuda.Tests folder has no project; its loose source is not a
separately executed suite.

The final full CPU lane reported 3,219 passes, two failures and one skip across 13 projects. Both failures were
reproduced on pristine B0: API operation inventory and the Windows MiniMax-H3 flat VAE path layout. A reported
T5 standalone parity pass contains a missing-reference guard and is unvalidated. CUDA builds for net8.0 and
net10.0 and the benchmark build pass without warnings. These results do not cover the outstanding generation
checks below.

## Descriptive operation timing

This table measures the earlier production candidate `e4ff1d1a0` with the strict-F32 harness later committed
as `930daa625`. The sibling-kernel review fixes require a fresh registered hardware campaign; these numbers
are retained history, not measurements of the latest artifacts.

The initial default-arm F32 probe failed its unchanged strict-F32 tolerance on B0: that arm used TF32
(`rel_l2 = 3.03e-4`, versus `2e-5`). Its failed output remains in `smoke/`. The new explicit
`mat-f32-strict` benchmark arm disables TF32 and competing routes before backend construction. Both arms then
use that same harness, built against pristine B0 production code/artifacts or the candidate respectively.
Neither the F32 fixture nor its tolerance was relaxed.

Two alternating pairs per cell, five trials per process, ten warmups, shared serialized tensors, 32 serial
operations per resident trial, and one operation plus completion wait per public trial:

| Backend/storage | Boundary | B0 mean ms | Candidate mean ms | Candidate/B0 |
|---|---|---:|---:|---:|
| CUDA F32 strict | Public | 0.38039 | 0.39427 | 1.036 |
| CUDA F32 strict | Resident | 0.33492 | 0.36002 | 1.075 |
| CUDA F16 | Public | 0.23997 | 0.21254 | 0.886 |
| CUDA F16 | Resident | 0.19556 | 0.20761 | 1.062 |
| Vulkan F32 | Public | 3.75525 | 3.88857 | 1.036 |
| Vulkan F32 | Resident | 5.87534 | 7.73552 | 1.317 |
| CUDA monolithic Flash F32 | Public | 0.46900 | 0.41273 | 0.880 |
| CUDA monolithic Flash F32 | Resident | 0.44961 | 0.48090 | 1.070 |

All these controlled-arm runs pass quality and retain their expected S.14/S.15/A.2/F.7 routes. Flash uses a
separate frozen all-finite fixture with split-K disabled; ALiBi/softcap defects are verified by the analytic
controls, whose old outputs are deliberately incorrect. The timing runs use the documented
3060 lock path and a verified Swarm quiet interval; shared service state was not stopped or changed. The
interactive Windows desktop still has other GPU clients and variable clocks. Resident intervals include host
gaps between submitted operations; they are not active-kernel duration sums.

These mixed results are a **performance concern**, not passing statistical acceptance. The unchanged Vulkan
route is also slower in this smoke. Do not claim improvement or non-inferiority, discard slower cells, pool
boundaries, or keep repeating samples until a favorable result appears. A fixed, separately registered campaign
on the hardware handoff must resolve the concern. Full model timing and quality remain separate gates.

## Kernel provenance

The existing NVRTC helper now also builds on Windows, using the same architecture/include-only options.
An unmodified SM80 Flash source rebuild exactly matched shipped PTX after line-ending normalization, before
arithmetic edits. Changed Flash and new attention softmax ship both SM80 and SM75 PTX at ISA 9.0, generated
with CUDA 13.0.88 NVRTC. Both modules assemble without spills for SM75 and SM86. Only SM86 executed here;
SM75 hardware remains unverified.

The review rebuild uses the cached Linux CUDA 13.0.88 NVCC image
`nvidia/cuda@sha256:7d2f6a8c2071d911524f95061a0db363e24d27aa51ec831fcccf9e76eb72bc92` for the two
NVCC-owned modules. Before edits it reproduced the shipped fused PTX exactly after newline normalization.
NVRTC-owned modules and the fused SM75 floor keep their prescribed frontend. The helper now prints the loaded
NVRTC major/minor and target on every compile; the emitted header records the exact compiler build.

The public `LaunchSoftmaxF16` adapter and its module remain for external `CudaKernels` consumers. Removing a
public method because in-tree attention no longer calls it would break compatibility. Further per-head launch
fusion/register retention is proposed optimization work, not a correctness fix: mask fusion already replaces
the old per-head adds and shared softmax with per-head softmax calls. The timing concern above remains open;
do not attribute it to one mechanism without a controlled profile or mark that suggestion performance-validated.

Raw logs, assembly diagnostics, fixtures, binaries and smoke output are outside version control under
`artifacts/attention-pr02` in the PR worktree. Retain that directory through review and the hardware handoff.

## Remaining gates

Complete affected real-generation A/B correctness and timing, including the frozen Qwen/Krea requests on the
4090, before closing PR 02 or freezing Bfix. The local Qwen primary B0 run cannot pass its placement admission
on the 3060, and Krea weights are not present locally. Preserve PR 01's independent outstanding work and the
existing numerical, hardware and performance gates. Do not substitute synthetic checks or these operation
smokes for generation quality, registered paired campaigns, or Kitchen parity.
