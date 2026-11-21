# PR 12: the Qwen primary against the correctness baseline

Plan item 12 asks for a repeatable, complete primary request-latency improvement against Bfix, with the four secondary
lanes preserved. This records the RTX 4090 campaign that measured it. It makes no claim for the RTX 3060, for Vulkan
beyond the lane below, or for Blackwell.

## What was compared

| Arm | Commit | What it is |
|---|---|---|
| R | `57b143f004050d484dc1b667ae8d5f91fc82ac22` | The PR 02 correctness pin (P2 in the [GPU handoff](../gpu-agent-handoff.md)) |
| C | `efbd2f6d7` | `main` at `e99279b84` plus the paged pool, the native paged decode and the cost accounting (#989, #990, #991, #992), all since merged |

**Bfix was never frozen by an earlier PR.** R is the PR 02 pin, chosen for this campaign because it is the correctness
fixes and nothing after them. The earlier reports say PR 02 must not be labeled Bfix until its gates close, and the
[4090 review](pr01-4090-review.md) records a Krea CUDA identity failure on that same pin that was never shown
resolved. So the result below is against R as named here; whether R is Bfix is a decision this report does not make.

Workload: the frozen protocol's Qwen primary ([protocol.json](../protocol.json)): Qwen3-4B Q4_K_M, 8,192 prompt
tokens, 128 generated, greedy, seed 42, continuous batching on, a 4 GiB KV pool, two warmup requests, the warm complete
Engine request timed.

Tool: one GenCapture source, identical in every arm's tree, built once per arm against that arm's production code. The
timed region is the same code in both. R's engine reports no prefill or decode split (both read 0 in every R result),
so only the total is compared.

## Path proof

| | R | C |
|---|---|---|
| Prompt, generated, finish | 8,192, 128, Length | 8,192, 128, Length |
| Text SHA-256 prefix | 26c68e74a9df5f02 | 26c68e74a9df5f02 |
| Prefill route | `cudnn.causal` x36 | `fa2.causal` x36 |
| Decode route | `flash.decode` x4,608 | `flash.decode.paged` x4,608 |
| `PrefixGather` | 9,288 records, 314,088,357,888 bytes | 72 records, 2,415,919,104 bytes |

On R every decode step of every layer gathers the whole F32 prefix again, about 33.8 MB each time. That, not the
attention kernel, is most of what follows.

## Primary result

Twenty pairs, all valid: three inferential sessions of five and the predeclared confirmation session of five,
alternating order. The text is identical on all 40 runs.

| | Value |
|---|---|
| C/R total, geometric mean | **0.0844** |
| Two-sided 95% interval | 0.0837 to 0.0853 |
| Confirmation session | 0.0832 |
| R total per request | 22,730 to 24,151 ms |
| C total per request | 1,938 to 2,010 ms |
| Peak VRAM, C minus R | -3,689 MiB on average (-2,753 to -4,903) |

The protocol's improvement rule is met: the interval is wholly below 1.0 and so is the confirmation.

Two things qualify the size of that number:

- **R runs out of device memory once in every process.** Each of the 20 R slots logs one out-of-memory warning, one
  retry and one release of all device memory, then completes with the right text. The log has no timestamps, so it is
  not known whether that falls in a warmup or in the timed request. No C slot logs one. The pairs were kept: this is
  how R behaves under the frozen workload on a 24 GB card, not a fault of the measurement.
- **Most of the gap was already closed on `main` before this campaign.** The same workload on `main` at `e99279b84`
  takes about 2,184 ms. #990's head against that `main` is 0.895 (0.8895 to 0.9000) over twenty pairs, and C, which is
  #990's head plus the cost accounting, measured 0.991 of #990's head over three sets. So the step from R's 23 s to
  about 2.2 s is work that was on `main` at that commit, chiefly the paged cache no longer gathering the whole prefix
  at every step; the step from 2.2 s to 1.96 s is #989 and #990.

## What was predicted

Written before any result, in the campaign brief:

| | Prediction | Outcome |
|---|---|---|
| Text | R's text equals C's | Met |
| Primary ratio | 0.80 to 0.92 | **Missed by an order of magnitude**: 0.084 |
| R's routes | Prefill not `fa2.causal`; decode `flash.decode` x4,608; about 9,288 gathers | Met; the gathered bytes were not predicted |
| Peak VRAM | C minus R within +500 MiB | Met: C is lower in every set |

## Secondary lanes

### CUDA dense (the pool off)

Twenty pairs, all valid, text identical on all 40 runs, no out-of-memory line in either arm.

| | Value |
|---|---|
| C/R total, geometric mean | **1.0065** |
| Two-sided 95% interval | 1.0021 to 1.0113 |
| One-sided 95% upper bound | 1.0104 |
| Confirmation session | 1.0104 |
| R total per request | 1,768 to 1,824 ms |
| C total per request | 1,766 to 1,841 ms |
| Peak VRAM | R 17,072 to 17,534 MiB; C 17,072 to 17,131 MiB |

The lane is preserved by the protocol's rule: the upper bound and the confirmation are under 1.03. It is not improved,
and it is not unchanged either: the interval sits wholly above 1.0, so **C is slower on the dense lane by about 0.3 to
1.1%**, some 12 ms a request. I predicted 0.90 to 1.00 and was wrong about the sign.

The cause is not established. Nothing since R was aimed at this lane's kernels. What the lane gained since R is the
planner's per-call validation and admission, about 4,600 attention calls a request; 12 ms over that many calls is
under 3 microseconds a call, which is the size such a path could cost. A smaller comparison of C with #990's head on
this lane (three sets, +0.70%, all three slower) suggested the cost accounting, but the code between those two heads
adds only a guarded evidence call and one delegate field on the per-call path, which cannot cost that much, so I take
that three-set difference as noise. No measurement isolates the cause.

### CUDA batched, CUDA graph, Vulkan paged

Pending at the time of writing.
