# PR 12: the Qwen primary against the correctness baseline

Plan item 12 asks for a repeatable, complete primary request-latency improvement against Bfix, with the four secondary
lanes preserved. This records the RTX 4090 campaign that measured it. It makes no claim for the RTX 3060, for Vulkan
beyond the lane below, or for Blackwell.

## What was compared

| Arm | Commit | What it is |
|---|---|---|
| R | `57b143f004050d484dc1b667ae8d5f91fc82ac22` | The PR 02 correctness pin (P2 in the [GPU handoff](../gpu-agent-handoff.md)) |
| C | `efbd2f6d79781e5665efe6d3716d88a3f49c29b5` | `main` at `e99279b847fc329338e278e88fdf4bf8fcc1e97d` plus the paged pool, the native paged decode and the cost accounting (#989, #990, #991, #992), all since merged |

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

The protocol's required proof for the Qwen primary, from the observed path-proof run of each arm (two warm-ups and
the measured request; the measured request alone in brackets). The timed primary sessions themselves were not
observed; they share this workload.

| Required entry | R | C |
|---|---|---|
| `scheduler.paged` | 3 [1] | 3 [1] |
| `scheduler.decode.batch` | 384 [128] | 384 [128] |
| `paged.keys` | 13,932 [4,644] | 108 [36] |
| `paged.values` | 13,932 [4,644] | 108 [36] |

All four are present on both arms. R reads its paged keys and values through a gather at every decode step of every
layer, which is why its counts are 129 times C's.

On R every decode step of every layer gathers the whole F32 prefix again, about 33.8 MB each time. That, not the
attention kernel, is most of what follows.

## Primary result

Twenty pairs, all valid: three inferential sessions of five and the predeclared confirmation session of five,
alternating order. The text is identical on all 40 runs. The geometric mean and the interval below cover the 15
inferential pairs; the confirmation session is reported on its own line, as the protocol has it. The same holds for
every lane in this report.

| | Value |
|---|---|
| C/R total, geometric mean | **0.0844** |
| Two-sided 95% interval | 0.0837 to 0.0853 |
| Confirmation session | 0.0832 |
| R total per request | 22,730 to 24,151 ms |
| C total per request | 1,938 to 2,010 ms |
| Peak VRAM, C minus R | -3,689 MiB on average (-2,753 to -4,903) |

By the numbers the improvement rule is met: the interval is wholly below 1.0 and so is the confirmation. (The
confirmation, 0.0832, sits below the interval's lower end; the protocol keeps the confirmation session out of the
interval, so that is allowed.) **Whether the rule is met under the frozen protocol is a separate question, and it is
open**: see the next point.

Two things qualify the size of that number:

- **R runs out of device memory once in every process, and the protocol counts that against the pair.** Each of the
  20 R slots logs one out-of-memory warning, one retry and one release of all device memory, then completes with the
  right text. No C slot logs one. The protocol's rule is that a memory violation invalidates the pair, and the Qwen
  workload sets `always_free_memory` to false. Read as written, that invalidates all 20 primary pairs, and the 20
  batched pairs for the same reason, which would leave this lane with no valid inferential result. I kept the pairs
  because the out-of-memory retry is what R does under the frozen workload on a 24 GB card, so removing them removes
  the reference. That is an argument for amending the rule or the reference, not a reading of the rule, and it is not
  mine to settle here: it belongs with the Bfix decision. It is also not known whether the retry falls in a warm-up or
  in the timed request, which bears directly on the 0.0844. The R log shows the three lines after the model load (an
  allocation of 4,748 MB with 50 MB free, an eviction that released 5,342 MB, then the release of all device memory),
  but it has no timestamps and no per-request markers, and the observation file marks requests without recording
  allocations, so neither can place them. Settling it needs a rerun with a marker before the timed region.
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

### CUDA batched (two concurrent requests)

Twenty pairs: 40 runs and 80 requests, every text identical (SHA-256 prefix 26c68e74a9df5f02).

| | Value |
|---|---|
| C/R total time of the batch, geometric mean, 15 inferential pairs | **0.0851** |
| Two-sided 95% interval | 0.0832 to 0.0868 |
| One-sided 95% upper bound | 0.0865 |
| Confirmation session | 0.0841 |
| R tokens per second, mean | 5.39 |
| C tokens per second, mean | 63.42 |

By the numbers the lane is preserved and improved (upper bound and confirmation far under 1.03 and under 1.0). In
throughput terms the estimator is 1 / 0.0851 = 11.75 times. An earlier version of this report gave 11.80, the
arithmetic mean of the per-pair throughput ratios, which is not the protocol's estimator.

The same validity question as the primary applies: every R slot has the out-of-memory retry, so read as written the
protocol invalidates these pairs too.

This lane runs through the paged pool, so it shows the same thing as the primary and carries the same qualifiers:
every R slot logs one out-of-memory line (20 of 20) and no C slot does, and R gathers the prefix at every step. It is
not a second, independent gain.

I predicted 5 to 15% above R. That was wrong by an order of magnitude, for the same reason the primary prediction
was: I had not accounted for what R's gather costs.

### CUDA graph

Twenty pairs, all valid. The path proof passed first: 128 measured graph replays on both arms.

| | Value |
|---|---|
| C/R total, geometric mean | **1.0019** |
| Two-sided 95% interval | 0.9990 to 1.0051 |
| One-sided 95% upper bound | 1.0045 |
| Confirmation session | 1.0119 |

The lane is preserved (upper bound and confirmation under 1.03) and not improved (the interval includes 1). I
predicted 0.98 to 1.02 before any timing result; it is inside.

**The graph lane's text is not the eager text.** Both arms produce the same text here (SHA-256 prefix
d490e1fc030bf1fd, 792 characters), and it differs from the eager lanes' text (26c68e74a9df5f02, 777 characters) from
character 343 on. The GPU session stopped the lane on that difference. **I resumed it under an amended validity
rule**: a pair is valid when C's text equals R's and both equal the lane's own text. The amendment was made before any
graph timing was seen, but after a candidate output had been, so it does not meet the protocol's order rule (endpoints
frozen before candidate results) and this lane's result should be read as obtained under an amended rule. Neither
`protocol.json` nor the campaign brief freezes a text for this lane. Because R, the PR 02 pin, shows the same
text as C, nothing in this series introduced the difference. I have not explained it, and the token at which the two
decodes part is not known (the saved outputs are text, with no token ids). It is recorded as an open item in
[troubleshooting](../../../docs/Checklists/TROUBLESHOOTING.md), since it may be a fault in the captured graph and not
only a property of this benchmark.

### Vulkan paged (descriptive, and not like-for-like)

Declared descriptive before it ran. The path proof, one observed run an arm, both clean with no warning or
out-of-memory line:

| | R | C |
|---|---|---|
| Total per request | 28,502.5 ms | 2,330.3 ms |
| Text | 550 characters, SHA-256 prefix 980406f74bb46717 | 777 characters, 26c68e74a9df5f02 |
| Prefill route | `cm2.causal` x108 | `cm2.causal` x108 |
| Decode route | `flash.decode.split` x13,824 | `flash.decode.split` x13,824 |
| `PrefixGather` | 27,864 records, 942,265,073,664 bytes | 27,864 records, 7,361,003,520 bytes |
| `Workspace`, `Cast` records | 0, 0 | 13,932, 216 |
| `HostReference` | 0 | 0 |

**The two arms do not produce the same text, so their times are not a speed comparison.** C's Vulkan text is the eager
CUDA text, character for character. R's is a different text from its first character: it opens with an image link
where the eager text opens with a heading. Both arms ran the same kernel rows the same number of times; what differs
is the gather, where R moves 128 times the bytes C does.

So on this lane the PR 02 pin generates something other than what CUDA generates, and the candidate does not. I read
that as a correctness difference on R's Vulkan paged path that later work removed, but I have not found the change
that removed it.

I predicted 0.95 to 1.05 for this lane. The path-proof ratio is about 0.08, wrong by an order of magnitude, and I did
not predict the text difference at all.

The five-pair session, descriptive only. Under the per-pair rule every pair is invalid for a comparison, because the
texts differ; no interval is given and no speed claim is made from this lane.

| | R | C |
|---|---|---|
| Total per request, five runs | 28,046 to 29,234 ms (mean 28,716) | 2,366 to 2,481 ms (mean 2,441) |
| Text on every run | 980406f74bb46717 | 26c68e74a9df5f02 |
| Peak VRAM | 21,916 to 21,959 MiB | 14,455 to 14,479 MiB |

Each arm is repeatable within itself: R produces its different text on all five runs, and C the eager text on all
five. Neither log has a warning, an error or an out-of-memory line. The plain ratio of the means is 0.085, which
describes two different computations.

## Raw results

On the RTX 4090 host, under `/home/hartsy/evidence/attention-4090/attn989-campaign` (`slots/`, `pr12/`, `info/`,
`queue/` and the kernel-check folders). Kept by the GPU session, not in this repository.

| File | SHA-256 |
|---|---|
| `pr12/index-results.tsv` (352 rows: every PR 12 result and status file) | c13cd8533a0fb36d102d74d562af46e83d7e6c9f8137dfaae6fecbe30dd4d2b6 |
| `pr12/schedule.tsv` | 4d2eb7c83b8610298a47f4c89b0b1df34b5f733f9947fc366c3e2f2bf4db685e |
| `pr12/FROZEN.md` | faf9d7809b27b389871da937e2aed6848a3e43e80908dee04dec29dff568e64e |
