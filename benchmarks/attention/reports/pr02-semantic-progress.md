# PR 02 semantic correctness progress

PR 02 is a stacked draft on PR 01, started under the user's subsequent authorization. PR 01's exit evidence
remains incomplete. This report does not freeze Bfix, approve later work, or claim primary-workload speed gains.

## Changes and reproduced defects

- CPU FlashAttention now preserves a NaN normalization result and treats a negative-infinite sink as absent.
  Two empty-row sink controls failed before the fix; all five scalar controls pass afterward.
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

## Local validation

The new GPU semantic class covers 26 cases: softcap/ALiBi ordering, empty local windows with non-finite sinks,
mixed excluded/visible rows, large F32 biases, NaN/+infinity, and multiple batches/heads with a 516-key reduction
tail. Existing dtype tolerances are preserved; these analytic controls do not establish model quality.

Full affected GPU categories run sequentially on Windows 11, RTX 3060 (SM86), driver 610.88:

| Project | Reported passes | Failures | Resource skips |
|---|---:|---:|---:|
| Core | 19 | 0 | 0 |
| LLM | 82 | 0 | 19 |
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

Raw logs, assembly diagnostics, fixtures, binaries and smoke output are outside version control under
`artifacts/attention-pr02` in the PR worktree. Retain that directory through review and the hardware handoff.

## Remaining gates

Complete affected real-generation A/B correctness and timing, including the frozen Qwen/Krea requests on the
4090, before closing PR 02 or freezing Bfix. The local Qwen primary B0 run cannot pass its placement admission
on the 3060, and Krea weights are not present locally. Preserve PR 01's independent outstanding work and the
existing numerical, hardware and performance gates. Do not substitute synthetic checks or these operation
smokes for generation quality, registered paired campaigns, or Kitchen parity.
