# Attention diagnostic report v3 (round 3)

Scope of this round, as directed: make H2 available for review; run the separate 1,024-token batching configuration; resolve the R10 fixture and arm mismatch without changing its numerical policy; correct the timing estimator and its interpretation; run one separately registered P1-versus-B0 campaign after the definitions were fixed; record production optimization separately. Decisions taken by the user for this round: proceed with proposal 2 and a corrected version of proposal 5; defer proposals 1, 3, 4 and 6 (6 is recorded for a separate PR). The output-divergence policy stands: divergence alone is not a blocker, correctness checks stay, and Krea byte identity is no longer pursued unless evidence suggests an accuracy defect. Earlier reports (v1, v2) remain as evidence; where this report corrects them, it says so (§7).

## 0. Summary

- **H2 is on the remote.** `codex/attention-h2` = `1894b4b6b794018a5cd7a612f6fcdf12e31b1462` (new branch; the PR branches `codex/attention-evidence-pr01`, `codex/attention-semantic-correctness-pr02` and `codex/attention-contracts-pr03` were not touched).
- **H3 is committed locally, not pushed:** `diag/attention-h3` = `28dfa3b2b7749bfaab5815a2e693568149812626` (H2 plus the arm and a freeze option).
- **R10 fixture and arm.** The registered fixture (`legacy_asserted_bounded=false`, sha256 `5439493c…`) is unchanged. The new arm `mat-f16-unasserted` (the `mat-f16` knobs with no legacy assertion) executes S.15 on net10.0 on both roots, with the same output digits as the original R10 record (`mean_abs` 7.749317387847845e-06). The original `mat-f16` arm is refused on this fixture by the contract.
- **Complete-call estimator.** On the committed-build A/A samples, the mean of 200 calls is about 11 % above the median (stalls account for the gap). The mean is also noisier, and on identical binaries it does not meet the non-inferiority bound (upper95 1.0365 against 1.03). The median estimator does (1.0138). Both are retrospective.
- **Simulation.** Under the registered layout, a null-true campaign passes the non-inferiority rule about 24 % of the time (mean estimator; 29 % for the median estimator). Regression is detected about 14 % of the time at a true ratio of 1.05 and about 68 % at 1.10 (mean estimator). The 0.04 at 1.03 is the boundary size and is not a power deficit. (§4)
- **1,024-token batching (batch-proof-1k-4GiB):** proven for this secondary configuration: two completions, Count 2 in 127 measured decode steps, identical texts, no dropped telemetry, quiet window verified (§5).
- **Registered P1-versus-B0 campaign (v3.1: mean of 200 calls, identical-binary control first):** inconclusive. Point ratio 1.0469 with 95 % one-sided bounds 1.0126 to 1.0836; confirmation 0.961. Neither rule is met. The identical-binary control was valid and met the non-inferiority rule (upper 1.003). The v3.0 control attempt was invalid (setup error) and is preserved (§6, §7).
- **Deferred:** proposal 1 (FP8 diagnostic), proposal 3 (staggered 8k admission, predicted to fail), proposal 4 (larger timing campaign). Proposal 6 (last-row prefill logits) is recorded for a separate production PR (`reports/PROPOSAL-6-LAST-ROW-LOGITS.md`).

## 1. Source state, provenance and what is committed

### 1.1 Commits and branches

| Item | SHA | Branch | State |
|---|---|---|---|
| H2 harness (base `fe12180`, PR 01 head) | `1894b4b6b794018a5cd7a612f6fcdf12e31b1462` | `codex/attention-h2` (remote, new this round); `diag/attention-h2` (local) | pushed this round |
| H3 harness (H2 plus `mat-f16-unasserted` and `--target-tokens`) | `28dfa3b2b7749bfaab5815a2e693568149812626` | `diag/attention-h3` (local) | committed this round, not pushed |
| Production B0 / P1 / P2 | `a6c465bc8` / `3ee783e76` / `57b143f00` | `attn-b0`, `attn-p1`, `attn-p2` | unchanged |
| PR branches | `fe12180` (PR 01), `5013ced` (PR 02), `60ec888` (PR 03) | `codex/attention-evidence-pr01`, `codex/attention-semantic-correctness-pr02`, `codex/attention-contracts-pr03` | not touched |

### 1.2 Clarification of the commit wording

- The H2 commit `1894b4b6b` was made earlier in this work (2026-11-15 18:35:33 -0500, author kalebbroo), on local `diag/attention-h2`. It was not made in the last pass. This round pushed it to the new remote branch `codex/attention-h2`.
- The round-2 report says "committed locally, not pushed" for H2, and the round-2 reply said "nothing was committed or pushed in this pass". Both describe the time of writing, not this round: H2 was committed before round 2 and is now pushed; no commit was made in the last pass.
- H2 has no co-author trailer. Amending it would change the SHA that the H2 binaries and manifests cite, so it was left as it is. H3 carries the trailer.

### 1.3 Committed builds and the unidentified pre-commit probe binary

- **Committed-build evidence** (H2 or H3 SHA): the Krea TF32 runs (committed request builds and committed tap builds), the Qwen default (`b0-net10.0-request`), the 8k batch proof (`p1-net10.0-request`), the v2 A/A samples (`b0-net10.0-op`), the R10 bounded trace (`b0-net10.0-op` and `p1-net10.0-op`), the 1k batch proof (H3 `p1-net10.0-request`), the S.15 checks (H3 op builds), and the v3 control and campaign (H3 op builds).
- **Pre-commit probe binary (unidentified).** `h2/bin/probe-b0-net10-op` (GpuBenchmarks.dll `686a63fc4dbf116f`) was built at 18:27:52, before the commit at 18:35:33. Its Core.dll and Cuda.dll are identical to the committed B0 build, but its GpuBenchmarks.dll is not (`0646ca9c7eb22866` in the committed build). It was used for: the 400-trial decomposition (round-2 §3.2–3.4), the core-placement sweep (§3.4), and the default acceptance smoke (§2.4). **Conclusions that depend on this binary are provisional:** the host-side attribution of the stalls, the core-placement result, and the smoke's output identity. The committed v2 A/A samples corroborate the stall share (3.1 % of calls above twice the median, against 3.4 % in the probe), but they do not replicate the host-side or core-placement findings.
- **Not executed:** the net8.0 builds compile with 0 errors and 0 warnings, but the .NET 8 runtime is not installed on this machine (only 10.0.15), so net8.0 routes and outputs are unverified. Every result in this report is net10.0.

### 1.4 Build checks

H3 net10.0 and net8.0 builds (op and request tools, B0 and P1 roots): all exit 0, 0 errors and 0 warnings. The B0 and P1 operation DLLs are byte-identical (`bb709fdd0625b49a`, the production libraries differ). The request DLL `79cbb55b9d33b6d7` was reproduced by a second build from the same commit.

## 2. R10 fixture and arm: resolution without changing the numerical policy

### 2.1 Facts

- The registered R10 fixture is `run-20261115-a/fixtures/f16-mat/fixture.json` (sha256 `5439493c9d92847a3be8ae32dceb830e91c08e405f1bab3faaaecd3dea4b20a3`, `legacy_asserted_bounded=false`, Dense, F16). The copy `diag…/fixtures/f16-mat-replica` is byte-identical. The H-compliant copy `f16-mat-legacy` (sha `2bcc4681…`, `legacy_asserted_bounded=true`) differs only in that flag; it is the fixture the v2 A/A used.
- The original R10 campaign record (`run-20261115-a/ops/routes/p1-f16-mat-mat-f16.json`, tool sha `4423d4a0…`) reports branch S.15, quality passed, and `mean_abs` 7.749317387847845e-06 on this fixture.
- Under the current H contract, the arm `mat-f16` (AllowF16=true, so a legacy assertion) is refused on this fixture before any timing: "Arm mat-f16 and fixture disagree on the legacy numerical assertion." (net10.0, B0, H3; rc 134; the refusal is the intended outcome).

### 2.2 The arm

`mat-f16-unasserted` (`benchmarks/HartsyInference.GpuBenchmarks/Attention/AttentionArms.cs`): the same knobs as `mat-f16` (sageAttn, sdpaCudnn and flashF16 off), with AllowF16=false, so it declares no legacy assertion and matches the fixture's flag. The arm adds no knob beyond `mat-f16`.

### 2.3 Route and output checks (final H3 binaries)

| Run (`h3/gpu-s15-final`) | Branch | Quality | `mean_abs` | Telemetry |
|---|---|---|---|---|
| net10.0 B0, `mat-f16-unasserted` | S.15 | passed | 7.749317387847845e-06 | disabled |
| net10.0 P1, `mat-f16-unasserted` | S.15 | passed | 7.749317387847845e-06 | disabled |
| net10.0 B0, `mat-f16` (contract) | — | refused before timing | — | — |
| net8.0 B0 and P1 | not run | — | — | runtime not installed |

The `mean_abs` digits match the original R10 record. On this fixture the numerical output of the arm is the same as the original R10 campaign's.

### 2.4 What an earlier draft got wrong, and how it was tested

- An earlier draft of the arm added `numerics.sdpaF16=true`, on a reading of the route predicates that suggested S.15 needed a legacy assertion or that grant. The literal arm (no grant; `h3/gpu-literal`, observe mode) executed **S.15** with the same `mean_abs`, so the grant was unnecessary. A second expectation of S.14 for that arm was also wrong: the run was refused with "Executed branch does not match the requested evidence cell" (`h3/gpu-s15/b0-literal.log`). The grant was removed before the commit, and the final binaries were rebuilt and re-verified (§2.3).
- The conclusion: on this path, the legacy flag does not select the S.15 kernel. The R10 numerical policy (F16 Q/K/V/output through the materialized score matrix) is unchanged, the fixture is unchanged, and the only change is the arm's declaration.

## 3. Complete-call versus median: retrospective assessment on the committed-build A/A samples

Data: the v2 A/A campaign (identical binaries: committed B0 operation build and a byte-identical copy; fixture `f16-mat-legacy`; S.15; 40 processes; 200 retained calls each). This is a retrospective analysis for choosing an estimator. It is not an acceptance result. Script and output: `h3/analysis/aa_v2_retro.py`, `h3/analysis/aa-v2-complete-call.json`.

Per-process distribution across the 40 processes (min / median / max):

| Quantity | Min | Median | Max |
|---|---|---|---|
| Mean of 200 calls (ms) | 0.1360 | 0.1546 | 0.1902 |
| Median of 200 calls (ms) | 0.1292 | 0.1384 | 0.1597 |
| p90 (ms) | 0.1427 | 0.1664 | 0.2440 |
| p99 (ms) | 0.2327 | 0.3652 | 0.8702 |
| Max (ms) | 0.8669 | 0.9826 | 1.5509 |
| Share of calls above 2x median | 1.0 % | 2.5 % | 8.0 % (mean 3.09 %) |
| Share of total time above the median | 7.5 % | 12.3 % | 19.1 % |
| Mean / median | 1.049 | 1.113 | 1.192 |

Frozen analysis on the same samples (identical binaries; frozen module; seed 274019; 20,000 resamples):

| Estimator | Pair-ratio SD (log) | Ratio | Upper95 | Lower95 | Confirmation | Non-inferiority rule | Regression rule |
|---|---|---|---|---|---|---|---|
| Median of 200 (v2, as registered) | 0.087 | 0.9883 | 1.0138 | 0.9633 | 0.9977 | met | not met |
| Mean of 200 (complete call) | 0.112 | 0.9965 | 1.0365 | 0.9594 | 0.9844 | not met | not met |

The median result reproduces the stored v2 analysis exactly, and the pure-Python mirror of the frozen function agrees with the frozen module for both estimators. **Reading:** the mean is the quantity the user asked to measure, because stalls are part of the complete-call cost. It is also noisier, so the same layout reaches the non-inferiority bound less often. This is a property of the estimator, not a computational fault. The v3 registration (§6) therefore uses the mean and states this trade-off in advance.

## 4. Simulation of the designs: non-inferiority and regression

### 4.1 Method

Campaign layout as in the registration: 3 inferential sessions of 5 pairs and 1 confirmation session of 5 pairs (40 processes). Processes are drawn with replacement from the 40 measured A/A processes. For a true ratio r, every candidate call is multiplied by r. The decision uses the frozen algorithm (stratified-by-session pair bootstrap; one-sided 95 % bounds; thresholds 1.03; confirmation at 1.03). Campaigns: 1,000 per cell; 2,000 resamples per campaign. A check with 20,000 resamples on 200 campaigns agreed within Monte Carlo error. Script and output: `h3/analysis/sim_designs.py` and `h3/analysis/sim-designs.json`.

### 4.2 Results (probabilities; standard errors about 0.01)

| Design (estimator) | P(non-inferiority passed) at true 1.00 | P(regression passed) at 1.00 (false positive) | P(regression) at 1.03 (boundary) | P(regression) at 1.05 | P(regression) at 1.10 |
|---|---|---|---|---|---|
| Single call (v1) | 0.132 | 0.016 | 0.043 | 0.102 | 0.254 |
| Median of 200 (v2) | 0.292 | 0.003 | 0.046 | 0.187 | 0.819 |
| **Mean of 200 (v3, registered)** | **0.237** | **0.002** | **0.038** | **0.141** | **0.678** |

Sensitivity, mean of 200 with the process-level spread doubled (sd 0.068 added to each process): non-inferiority at 1.00 = 0.170; regression at 1.05 = 0.109; regression at 1.10 = 0.470. Validation with 20,000 resamples (200 campaigns, mean of 200): non-inferiority at 1.00 = 0.235 (±0.030); regression at 1.05 = 0.170 (±0.027).

### 4.3 Reading, and the "5 % power" statement

- **Null-true (1.00):** with this layout, non-inferiority is passed in about one campaign in four for the mean estimator, and about three in ten for the median. A null-true campaign will often end inconclusive. That is the design's operating characteristic.
- **Boundary (1.03):** the regression rule fires about 4 % of the time, which is the one-sided size at the margin. It is not a power deficit, and the earlier "true +3 % regression about 5 %" line was misleading for that reason. Power is meaningful only above the margin.
- **Above the margin:** regression is detected about 14 % of the time at 1.05 and about 68 % at 1.10 for the mean estimator. Detecting 5 % effects reliably needs a larger layout, which is deferred.
- **The v2 A/A campaign** met its registered non-inferiority criterion: upper95 1.0138 against 1.03, and confirmation 0.9977 against 1.03. One campaign does not establish the method's general reliability: the simulation says a campaign of this layout passes that criterion on identical binaries about 29 % of the time (median estimator). The round-2 figure of about 60 % came from an earlier parametric simulation (`h2/sim/protocol_sim.py`) with a different noise model. It is withdrawn; the empirical-population figure is used here.

### 4.4 Assumptions and limitations

1. Processes are exchangeable and drawn with replacement from one machine state (40 processes, one day).
2. A true ratio multiplies every candidate call, so the stall share is unchanged. A regression that changes the stall share is not modeled.
3. Calls within a process are treated as independent draws from the pooled sample; stall clustering in time is not modeled.
4. No drift over time and no change in GPU contention are modeled.
5. The bootstrap uses 2,000 resamples per campaign; 20,000 agreed on a check of 200 campaigns.
6. The frozen decision rules and session structure are reproduced exactly (checked against the frozen module on the real samples).
7. Results are for net10.0 only.

## 5. batch-proof-1k-4GiB: separate secondary configuration

Status: **proven for this secondary configuration only.** It does not turn the failed two-request 8k case (§5 of report v2) into a pass, and it does not alter the primary.

### 5.1 Frozen before the run
Manifest `h3/batch-1k/manifest-frozen.json` (sha256 `277a25c3f3f0e396c51615441d76a4a48d3312c6e0f10ee309e6212544e6d11e`), written before the job started. It fixes: the frozen 1,024-token workload (`h3/workload-1k/freeze/workload.json`, rendered prompt tokens 1,024; the same model and embedded tokenizer as the primary; 128 generated, greedy); the H3 P1 request build (`79cbb55b9d33b6d7`); the primary's 4 GiB pool (4,294,967,296 bytes); the two-request batched lane with `--require-batched-decode`; the required checks; and the job script (`h3/scripts/job_batch_1k.sh`). The primary workload and its stored result are recorded in the manifest by hash and were not touched.

### 5.2 Run and procedure
Run under `gpu_batch.sh` (flock on `/tmp/claude-1000/gpu4090.lock`; Swarm quiet-window wait, watch and verify). Recorded in `h3/batch-1k/status.txt`: lock 20:17:37; quiet wait passed 20:18:07; job rc 0; watcher rc 0; **quiet verify passed**; done 20:19:16. Preflight: 22,890 MiB free on the 4090 against 14,000 MiB required (`preflight.txt`). The informational single-request control ran after (rc 0).

### 5.3 Required checks

| Check | Result |
|---|---|
| Both requests complete | yes: batched rc 0; two completions, each with `stop: Length` |
| Prompt tokens per request | 1,024 and 1,024 (declared 1,024) |
| Completion tokens per request | 128 and 128 (declared 128) |
| Texts identical between the two batched requests | yes (sha256 of text, first 16 hex `b764ab04a7e78917` for both) |
| Simultaneous decode, `scheduler.decode.batch` Count = 2 | yes: 127 measured decode steps with Count 2 (`measured_count_histogram` {1: 2, 2: 127}); 386 decode-batch events in total, of which 129 are in the measured window; `max_active_sequences_measured` 2 |
| Telemetry dropped | 0 |
| Preflight and quiet window | passed (above) |

### 5.4 Informational results

- **Output identity with the single-request control** (paged lane, same pool, same workload): completion text identical (`b764ab04a7e78917`), 128 tokens.
- **Executed routes** (observation file, 85,496 events): prefill attention `F.2` (cuDNN-backed fused path, 216 events) and decode attention `F.4` (27,648 events); `cuda.causal.mask`, `cudnn.inputs` and `cudnn.casts` appear in prefill. Route codes are as named in `CudaBackend.cs` (F.n = FlashAttention).
- **Per-request latency (one run each, not acceptance timing):** batched pair 6,258.8 ms and 6,238.4 ms, both 128 tokens, concurrent; single-request control 3,348.8 ms. Aggregate throughput of the pair 40.9 tokens/s, against 38.2 tokens/s for the single request (128 / 3.349 s). No performance claim is made from one run.
- **VRAM (nvidia-smi sampler, about 0.2 s):** batched run: process peak 15,894 MiB; device used peak 17,095 MiB; device free minimum 6,988 MiB. Control: process peak 14,646 MiB; device free minimum 8,236 MiB. The 8k run's process peak was 22,198 MiB (report v2 §5.5).
- **Pool arithmetic reported by the tool:** `pages_per_request_expected` 73 and `pages_two_requests_expected` 146 agree with the arithmetic (1,024 + 128 + 1 tokens over 16 per page). The tool also reports `kv_pool_pages_expected` 0 and a negative spare count (-146), and `paged_kv_heads_observed` and `paged_head_dim_observed` 0. These are reporting gaps in the tool, not evidence of a failure. Pool capacity is 910 pages (4 GiB over 4,718,592 bytes per page), so 764 pages are spare after two requests.
- **Prefill logits buffer** (the all-positions F32 projection, §5.2 of report v2): predicted 593.5 MiB at 1,024 prompt tokens, against 4,748 MiB at 8,192. It did not block the run.

### 5.5 Failures and retries
None. The logs contain no error lines for either run, and no configuration was changed.

### 5.6 Limits
Both requests used the same prompt, so the run shows simultaneous decode and identical outputs, not batching across different prompts. One run only, with no statistics. The result is for this 1,024-token secondary configuration and does not establish batching at 8,192 tokens.


## 6. Registered P1-versus-B0 campaign under timing v3.1 (complete-call mean)

### 6.1 Registrations and attempts
- v3.0 registered (`v3/registration-v3.json`, sha256 `1ea080d2…`; `reports/TIMING-V3-REGISTRATION.md`). Its control attempt (`v3/aa-v3-mean200/`) is **invalid and preserved**: the reference arm was a folder holding only the GpuBenchmarks DLL, so 20 reference processes failed at process start (rc 131; "libhostpolicy.so not found") with no measurements. The 20 candidate processes ran (rc 0). Under the registered rule the control was invalid and the campaign was not run.
- v3.1 amendment (`reports/TIMING-V3.1-AMENDMENT.md`), written and frozen before the new control: complete reference copy (`v3/ref-bin-b0-net10.0-op-full/`, tree identical to the build, 473 files); control `AA-v3.1-mean200` (job sha256 `7d110967…`). Two analysis-script bugs in my scripts were fixed before the campaign: the secondary median analysis passed one session where three are required, and the loader treated a missing campaign file as an error. Neither changes the statistic or the decision rules. Campaign job sha256 `a5a47f9c…` (equal to the v3.0 registration).
- No measurement from the failed v3.0 reference arm exists, so nothing measured was discarded. The v3.1 control and campaign each ran once. No pairs were added and no reruns were made after a result.

### 6.2 Control (identical binaries, complete reference copy, v3.1)
Valid: 40 of 40 processes; no validity problems.

| Analysis | Ratio | Upper95 | Lower95 | Confirmation | Non-inferiority | Regression |
|---|---|---|---|---|---|---|
| Gating: mean of 200 (registered) | 0.9636 | 1.003 | 0.9265 | 0.9873 | met | not met |
| Secondary: median of 200 | 0.9811 | 1.0143 | 0.9496 | 0.981 | met | not met |

Per-process summary (reported, not gating; two B0 copies): median of per-process means 0.1496 and 0.1501 ms; median of per-process medians 0.1338 and 0.1346 ms; median p99 0.352 and 0.375 ms; calls above twice the median 2.8 % and 3.2 %; time above the median 11.8 % and 12.8 %.

The identical-binary control therefore met its registered non-inferiority criterion on this run (upper95 1.003 against 1.03). Under the simulation (§4) such a pass happens about 24 % of the time, so one pass is not evidence that the method is reliable in general. The control's point ratio, 0.9636, is itself 3.6 % from 1, a reminder of how much a single campaign of this layout can move.

### 6.3 Campaign P1 versus B0 (registered, once)
Valid: 40 of 40 processes; no validity problems.

| Analysis | Ratio | Upper95 | Lower95 | Confirmation | Non-inferiority | Regression |
|---|---|---|---|---|---|---|
| **Gating: mean of 200 (registered)** | **1.0469** | **1.0836** | **1.0126** | **0.961** | not met | not met |
| Secondary: median of 200 | 1.055 | 1.0934 | 1.0193 | 0.9759 | not met | not met |

**Outcome (registered rule): inconclusive.** The regression rule needs the lower bound above 1.03 and the confirmation ratio above 1.03. The lower bound is 1.0126 and the confirmation ratio is 0.961, so the rule is not met. The non-inferiority rule is not met either. Nothing was resampled.

Per-process summary (reported, not gating):

| Quantity | P1 (candidate) | B0 (reference) |
|---|---|---|
| Median of per-process means | 0.1551 ms (range 0.1376–0.1855) | 0.1486 ms (range 0.1444–0.1737) |
| Median of per-process medians | 0.1380 ms | 0.1331 ms |
| Median p99 | 0.364 ms | 0.396 ms |
| Calls above twice the median | 2.9 % | 3.3 % |
| Time above the median | 12.0 % | 12.8 % |

### 6.4 What this result supports
- Under the registered estimator the P1 point ratio is 1.0469 (95 % one-sided bounds 1.0126 to 1.0836). The data are compatible with a small slowdown of about 1 to 8 %. They do not meet the registered regression rule, and they do not establish equivalence.
- The difference is in the typical call (the body of the distribution: median 0.138 against 0.133 ms), not in the stall share. P1 has slightly fewer calls above twice the median and a smaller p99.
- The original R10 displayed ratio of 1.26 (a single-call campaign on the older pin) is not reproduced under the registered mean design: it lies above the current upper bound (1.0836). The original result is preserved as recorded and is not overwritten. Single-call noise is large (§4), so the earlier magnitude is not evidence of the present difference.
- **R10 status:** still open. The registered rule does not establish a regression, and it does not clear P1. A decisive answer at the 3–5 % level needs a larger layout (proposal 4), which is deferred.

### 6.5 Not claimed
No merge decision, no PR state change, no tolerance, and no claim about production arithmetic. The result is for one 20-pair campaign on one machine state.

## 7. Corrections to earlier wording and numbers

1. **"Non-inferiority passed" and "control PASS".** The v2 A/A campaign met its registered non-inferiority criterion (plainly stated now). Earlier phrasing that avoided this, and the "PASS" label, were confusing; the "PASS" label meant only that the regression rule was not met.
2. **"5 % power for a true 3 % slowdown."** Wrong framing. About 5 % at the margin is the test's size, not a power shortfall (§4.3).
3. **Round-2 v2 non-inferiority figure of about 60 %.** Withdrawn; the empirical-population simulation gives about 29 % (§4.3).
4. **"Nothing was committed in this pass."** Clarified in §1.2.
5. **Probe conclusions.** Provisional, because the probe binary is unidentified (§1.3).
6. **"Krea byte identity" as a goal.** Not pursued further. The softmax change's causal role is established for this workload; the amplification mechanism remains partly unresolved, as stated in round 2.
7. **An earlier R10 arm draft.** It included an unnecessary F16 grant. It was tested and removed (§2.4).
8. **The 8k two-request case** remains a failed case. The 1,024-token case is a separate configuration and does not convert it (§5).
9. **v3.0 control attempt (setup error).** The reference arm was an incomplete folder (one DLL, no runtime configuration or dependencies). 20 reference processes failed at start, with no measurements. The attempt is preserved (`v3/aa-v3-mean200/`, `v3/ref-bin-b0-net10.0-op/`) and was superseded by v3.1 (§6.1).
10. **Analysis-script bugs (mine), fixed before the campaign.** The secondary median analysis passed one session where three are required, and the loader failed on a missing campaign file. Neither touched the registered statistic or the decision rules.
11. **Probe binary, reproduction attempt.** One incremental build of the H2 commit against `attn-b0` gives the committed B0 hash (`0646ca9c7eb22866`), not the probe's `686a63fc4dbf116f`. The probe remains unidentified, so its conclusions stay provisional (§1.3).

## 8. Deferred items and recorded proposals

- **Proposal 2** (1,024-token batching): run (§5).
- **Proposal 5** (corrected R10 measurement): run as the registered v3.1 campaign (§6). Result: inconclusive; R10 stays open.
- **Proposal 1** (native FP8 GEMM off, Krea diagnostic): deferred, because byte identity is no longer pursued and FP8 amplification experiments wait until the merge blockers are resolved.
- **Proposal 3** (staggered 8k admission): not run. It is predicted to fail while the 8k prefill needs the all-positions logits buffer.
- **Proposal 4** (larger timing layout for a 3 % claim): deferred. The layout in §4 is the registered one. The inconclusive §6 result is the case where a larger layout would be the next step, if approved.
- **Proposal 6** (project only the last prompt row during prefill): recorded for a separate production PR with last-row equivalence and serving/scoring compatibility tests (`reports/PROPOSAL-6-LAST-ROW-LOGITS.md`). It is not implemented here, and the frozen primary is not altered.

## 9. Artifacts and hashes

- H2 diff: `reports/H2-harness-1894b4b6b.diff` (sha256 `9f2eaab86855d86bcc532bb05838e33ef88b6314273432bebc076852e397f727`). H3 diff: `git -C attn-h3 diff 1894b4b6b…28dfa3b2…`.
- Registration v3.0: `v3/registration-v3.json` (sha256 `1ea080d2…`); `reports/TIMING-V3-REGISTRATION.md`. Amendment v3.1: `reports/TIMING-V3.1-AMENDMENT.md`; control job `v3/job_aa_v3_1_mean200.sh` (sha256 `7d110967…`); campaign job `v3/job_p1_vs_b0_v3_mean200.sh` (sha256 `a5a47f9c…`). Analysis: `v3/analyze_v3.py` (sha256 `a39a66a06f7ee846…`), outputs `v3/analysis-v3.json`, `v3/analyze-v3-final.txt`.
- v3.0 failed control attempt (preserved): `v3/aa-v3-mean200/`. v3.1 control: `v3/aa-v3.1-mean200/`. Campaign: `v3/p1-vs-b0-v3-mean200/`. Gate and run logs: `v3/run-status-v3.1.txt`.
- H3 diff: `reports/H3-harness-28dfa3b2.diff` (sha256 `f0a6f25d564570dab146eef381b76bc0bb20eb4041232ab5b6a1dbd510a4d302`).
- 1k manifest (frozen before the run): `h3/batch-1k/manifest-frozen.json` (sha256 `277a25c3f3f0e396c51615441d76a4a48d3312c6e0f10ee309e6212544e6d11e`).
- Simulation: `h3/analysis/sim-designs.json`; retrospective analysis: `h3/analysis/aa-v2-complete-call.json`.
- R10 variability note (round 2, with the §8 correction): `reports/R10-VARIABILITY-v2.md`.
- Evidence folder: `/home/hartsy/evidence/attention-4090/diag-20261115-c/`; index `INDEX-SHA256.txt`.
