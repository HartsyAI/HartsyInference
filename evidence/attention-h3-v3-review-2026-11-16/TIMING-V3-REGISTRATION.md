# Timing v3 registration: complete-call mean, R10 fixture, unasserted arm


> **Status update.** The v3.0 control attempt was invalid (incomplete reference copy; preserved). It was superseded by `TIMING-V3.1-AMENDMENT.md` (same definitions, complete reference copy). The v3.1 control was valid and the campaign ran once. Result: inconclusive (`REPORT-diag-v3.md` §6).
Status: registered before any v3 run. Machine-readable copy: `v3/registration-v3.json` (sha256 `1ea080d248914d606b62f3ca15313494102bed8c7df117f0e27bcd6d41f6bd4d`).
Job scripts: control `v3/job_aa_v3_mean200.sh` (sha256 `e70cd371f836c5e3ccf6f2b4df4b8e1ed0ababc5a8c17ba55623f655ec58ecbb`); campaign `v3/job_p1_vs_b0_v3_mean200.sh` (sha256 `a5a47f9c5edf6672f6acc4e8e878dfd0e580bc65113f15379618b56a543779f8`).
Frozen statistics module: `benchmarks/attention/statistics_protocol.py` (sha256 `ff5cdeb49df32e0d4941e7685586571d4209f731421a0a6537278e8fe89c84fd`), unchanged.
Harness: H3 commit `28dfa3b2b7749bfaab5815a2e693568149812626` on `diag/attention-h3` (base H2 `1894b4b6b794018a5cd7a612f6fcdf12e31b1462`).

## 1. Why a new registration

v2 gated on the median of 200 public calls. A median describes typical latency and does not count the stalls that are part of complete-operation cost. On the committed-build A/A samples (40 processes x 200 retained calls, identical binaries):

- The per-process mean is 11 % above the median (median of mean/median ratios 1.11; range 1.05 to 1.19).
- Calls above twice the process median are 3.1 % of calls (range 1.0 to 8.0 % per process) and carry 12 % of total time (median per process).
- The mean is noisier. The standard deviation of log pair ratios is 0.112 for the mean and 0.087 for the median, on the same data.
- Retrospectively, under the frozen rules, the mean estimator on identical binaries gives an upper 95 % bound of 1.0365, so the non-inferiority criterion is not met. The median estimator gives 1.0138, which meets it. This retrospective result is not an acceptance result.

## 2. Definitions

| Item | Registered value |
|---|---|
| Question | P1 versus B0 under the complete-call mean, with an identical-binary control run first |
| Fixture | `run-20261115-a/fixtures/f16-mat/fixture.json`, sha256 `5439493c9d92847a3be8ae32dceb830e91c08e405f1bab3faaaecd3dea4b20a3`, `legacy_asserted_bounded=false` (the original R10 fixture, unchanged) |
| Arm | `mat-f16-unasserted`: the `mat-f16` knobs (sageAttn, sdpaCudnn, flashF16 off) with `AllowF16=false`, so no legacy assertion is declared |
| Route | S.15, verified on net10.0 for B0 and P1 (final H3 binaries). The net8.0 builds compile but were not executed (the .NET 8 runtime is not installed here), so net8.0 is not verified and not used |
| Original arm | `mat-f16` (AllowF16=true) is refused on this fixture by the contract before timing ("Arm mat-f16 and fixture disagree on the legacy numerical assertion."). Not used |
| Boundary, cohort | public; linux-4090; RTX 4090 (CUDA_VISIBLE_DEVICES=1, PCI order) |
| Trials | 200 retained public calls per process (harness warmups) |
| Gating statistic | per-process arithmetic mean of the 200 calls |
| Reported, not gating | median, p90, p99, max, share of calls above 2x median, share of total time above the median (per process and per arm) |
| Pair ratio | candidate per-process mean / reference per-process mean |
| Layout | 3 inferential sessions x 5 pairs, 1 confirmation session x 5 pairs; 40 processes per campaign (20 candidate, 20 reference); order rule as v2 |
| Analysis | frozen `analyze`: seed 274019, 20,000 resamples, one-sided 95 % bounds |
| Noninferiority | upper95 <= 1.03 and confirmation ratio <= 1.03 |
| Regression | lower95 > 1.03 and confirmation ratio > 1.03 |
| Validity per process | exit 0; branch S.15; quality passed; telemetry disabled; exactly 200 finite positive latencies |
| Control | AA-v3-mean200: B0 H3 build against a byte-identical copy, run first. An invalid control invalidates the campaign. Its noninferiority or regression outcome is reported, not gating |
| Campaign | P1-vs-B0-v3-mean200: P1 H3 candidate, B0 H3 reference |
| Resampling | none. No additional pairs. No reruns after a result. Failed or invalid processes are preserved and reported, not re-sampled |
| Preserved | the original R10 campaign record (`run-20261115-a/ops/routes/p1-f16-mat-mat-f16.json`: branch S.15, fixture sha above, displayed ratio 1.26) is unchanged and is not a gate |

## 3. Expectations registered in advance

From the simulation in `h3/analysis/sim-designs.json`: 1,000 campaigns per cell, 2,000 resamples per campaign (validated against 20,000 resamples on 200 campaigns, which agreed within Monte Carlo error). Processes are drawn from the 40 measured processes; a true ratio multiplies every candidate call.

| True ratio | P(noninferiority passed), mean | P(regression passed), mean |
|---|---|---|
| 1.00 | 0.24 | 0.002 |
| 1.03 | 0.04 | 0.04 (about the one-sided size at the boundary, not a power deficit) |
| 1.05 | 0.01 | 0.14 |
| 1.10 | 0.00 | 0.68 |

These are predictions for the design, not results. A null-true campaign will often be inconclusive for noninferiority. Doubling the process-level spread lowers the pass rate at 1.00 to 0.17.

## 4. Limits

- One campaign, one fixture, one machine state. The identical-binary control is the only null-true evidence for this estimator.
- No larger layout is registered. More pairs for noninferiority or 5 % regression power are a separate proposal (deferred).
- net8.0 is not verified (runtime not installed here).
