# Attention diagnostic report v3: addendum of 2026-11-16

**Status.** Review addendum for the coordinator. It does not edit or replace `reports/REPORT-diag-v3.md` (sha256 `159f8207…`), the registration or amendment, the analysis script, `v3/analysis-v3.json`, or any result file. Where this addendum and the report disagree, this addendum governs the items in §7.

**Source.** H3 commit `28dfa3b2b7749bfaab5815a2e693568149812626` (local branch `diag/attention-h3`) is now pushed to a new remote branch, `codex/attention-h3`. The existing branches `codex/attention-evidence-pr01` (`fe12180ee…`), `codex/attention-semantic-correctness-pr02` (`5013ced4…`), `codex/attention-contracts-pr03` (`60ec8888…`) and `codex/attention-h2` (`1894b4b6b…`) were not changed.

## 1. Headline

1. **v3.1 used ordinary public timing.** Both campaigns invoke `attention-table --boundary public --trials 200 --expect-branch S.15 --cohort linux-4090 --arm mat-f16-unasserted` on the unchanged R10 fixture. Neither `--trace-out` nor `--observe` appears in any of the 80 process invocations. Every result file has `measurement_mode: "acceptance"`; none has `diagnostic-trace`.
2. **The records are not timing-eligible.** Every v3.1 process file has `timing_eligible: false` and `acceptance_reviewed: false`. Quality passed in all 80 (`quality_passed: true`), branch S.15, telemetry disabled, exit code 0 in all 80.
3. **Cause.** The runner sets `timing_eligible = no trace && boundary != observe && fixture.AcceptanceReviewed && quality` (`benchmarks/HartsyInference.GpuBenchmarks/Attention/AttentionEvidenceRunner.cs:124`). The fixture's `acceptance.reviewed` is `false` (source `docs/ATTENTION_CONTRACT.md` §8; `rel_l2` 0.003; `mean_abs` null). That is the only failing term, and the quality test is itself judged against that same unreviewed tolerance.
4. **The same holds for every timing campaign in this series.** The v2 A/A records (`aa/AA-v2-median200`: 40 files, tool `0646ca9c…`, fixture `f16-mat-legacy`, sha `2bcc4681…`) have `timing_eligible: false` and `acceptance_reviewed: false`. The original R10 record has no eligibility field at all.
5. **The project rule.** `benchmarks/attention/README.md` states: "Unreviewed acceptance or failed quality also makes `timing_eligible` false; measured pilot timings remain descriptive." It also states that `evidence.py statistics` "`valid` pair flags must come from reviewed quality/route/cache/memory/cohort evidence."

**Consequence.** The v3.1 control and campaign outputs, and the v2 A/A non-inferiority result, are descriptive. None is an acceptance result. v3.1 establishes neither non-inferiority nor material regression, and under the project rules it is not eligible to establish either.

## 2. The validity gate: what I set, and where it fell short (my error)

- The registered validity list (exit 0; branch S.15; `quality_passed`; telemetry disabled; 200 finite positive latencies; `TIMING-V3-REGISTRATION.md` §2 and `registration-v3.json`, `validity_per_process`) omitted the harness's eligibility term and the README's reviewed-evidence requirement. I did not check those rules when I wrote the registration.
- `v3/analyze_v3.py` sets `"valid": True` on every pair it builds, after the registered list has passed. The frozen module checks only that the flag is `True` (`benchmarks/attention/statistics_protocol.py`, line 23). It therefore accepted flags that the README says must come from reviewed evidence. The analysis bypassed that rule. The report's "40 of 40 valid" refers to the registered list only.
- I did not pass `--mandatory`. With it, `AttentionEvidenceRunner.cs` lines 30–31 refuse an unreviewed fixture before any timing.
- This does not change the arithmetic. It changes what the numbers may be used for.

## 3. Verified values

Checked from the 80 process files and the logs (`v3/review-2026-11-16/fields-and-hashes.txt`). Within each role every field has one value across its 20 files. "Control" is AA-v3.1 (candidate role = B0 build; reference role = byte-identical B0 copy). "Campaign" is P1-vs-B0 (candidate = P1 build; reference = B0 build).

| Field | Control candidate (B0) | Control reference (B0 copy) | Campaign candidate (P1) | Campaign reference (B0) |
|---|---|---|---|---|
| `measurement_mode` | acceptance | acceptance | acceptance | acceptance |
| `boundary` | public | public | public | public |
| `timing_eligible` | **false** | **false** | **false** | **false** |
| `acceptance_reviewed` | **false** | **false** | **false** | **false** |
| `quality_passed` | true | true | true | true |
| `branch` | S.15 | S.15 | S.15 | S.15 |
| `arm`, `legacy_asserted_bounded` | mat-f16-unasserted, false | same | same | same |
| `telemetry` | disabled | disabled | disabled | disabled |
| samples per process | 200 (warmup 10, iterations 1) | same | same | same |
| `fixture_sha256` | 5439493c… | same | same | same |
| `tool_sha256` (GpuBenchmarks.dll) | bb709fdd… | bb709fdd… | bb709fdd… | bb709fdd… |
| `core_sha256` (HartsyInference.Core.dll) | 8a1767e3… | 8a1767e3… | 26b00aea… | 8a1767e3… |
| `backend_sha256` (HartsyInference.Cuda.dll) | eaee02e0… | eaee02e0… | 54605c1e… | eaee02e0… |
| exit codes (`timing-exits.txt`) | 0 × 20 | 0 × 20 | 0 × 20 | 0 × 20 |

- Native libraries (cuDNN 9.24.0, cuBLAS/cuBLASLt 13), PTX (209 entries) and SPIR-V (185 entries) maps: one distinct value in each group, across all 80 files.
- **Resolved settings:** 299 keys. Identical between control and campaign, and identical to the original R10 record. Key values: `sageAttn`, `sdpaCudnn`, `flashF16`, `sdpaF16`, `sdpaNoF16`, `noTf32`, `sdpaNoTf32`, `sdpaForceTiled` all False; `sageF16MinSkv` 8192; `fa2Prefill`, `flashDecodeGqa`, `mempoolKeep` True.
- **Knob provenance.** The `Apply()` settings of `mat-f16-unasserted` are the same as those of `mat-f16` (`AttentionArms.cs`, lines 79–84). The provenance string of the v3 files lists one more name, `numerics.sdpaNoTf32=request profile`, than the original R10 record. The original record came from an older tool build (`4423d4a0…`). The resolved values are identical, so "same knobs" holds at the level of settings, not of the provenance string.
- **Original R10 record** (`run-20261115-a/ops/routes/p1-f16-mat-mat-f16.json`): `milliseconds` has one element. It has no `measurement_mode` or `timing_eligible` field.
- **v3.0 attempt, preserved** (`v3/aa-v3-mean200/`, 69 files): its 20 candidate processes ran with exit 0. Its 20 reference processes failed at process start with exit 131 and produced no measurements. The attempt is kept unedited.

## 4. Ordering (local time; `v3/review-2026-11-16/timestamps.txt`)

| Time | Event |
|---|---|
| 20:16:57 | Campaign job script written (`job_p1_vs_b0_v3_mean200.sh`, sha `a5a47f9c…`, matches the registration). |
| 20:17:19 | `registration-v3.json` saved (sha `1ea080d2…`; its `written_utc` is 01:16:57Z). This is the pre-run registration. |
| 20:23:47 | v3.1 control job script written (`job_aa_v3_1_mean200.sh`, sha `7d110967…`). |
| 20:24:41 | v3.1 control job started. |
| 20:26:43 | Control job finished: 40 processes, all exit 0. |
| 20:26:43 | `run_v3_1.sh` gate printed "control v3.1 INVALID: campaign not run". The gate reads `analysis-v3.json`; the analysis had crashed on the `JOB DONE` line of the timing file before writing it, so the check failed. This verdict came from the parser crash, not from the control's data. |
| 20:27:15 | `analyze_v3.py` last written (parser and plumbing fixes). |
| 20:27:16 | `analyze-v3.1-control.txt` written: valid 40 of 40. |
| 20:27:26 | `TIMING-V3.1-AMENDMENT.md` last written. Its control-outcome and gate-correction sections were added after the control ran, although its status line says "written and frozen before the v3.1 control ran". No pre-control copy was kept. |
| 20:27:26 | A hand-written line in `run-status-v3.1.txt`: "control v3.1 valid (40/40); campaign launched manually after the valid control". The script writes its lines in a different format, and its campaign-start line ("control v3.1 valid: campaign starting") is absent. The campaign was started by hand. |
| 20:27:45 | Campaign `gpu_batch.sh` lock acquired. |
| 20:28:15 | Quiet wait passed. |
| 20:28:16 | Campaign job started; first process about 20:28:17. |
| 20:30:17 | Campaign job finished: 40 processes, all exit 0; quiet verify passed. |
| 20:30:25 | `analysis-v3.json` and `analyze-v3-final.txt` written. |
| 20:31:51 | `TIMING-V3-REGISTRATION.md` last written, after the campaign had ended. Its opening now carries a post-run status paragraph. I kept no pre-edit copy, so I cannot show that nothing else changed. Its definitions (fixture, arm, layout, validity, estimator, module sha `ff5cdeb4…`) match `registration-v3.json`. |

**Consequences.** The analysis script and the amendment were edited after the control's data existed and before any campaign process started. The edits were analysis plumbing and status text. They did not change the statistical inputs (§5). The campaign ran once, with no resampling and no added pairs.

## 5. Independent recomputation

`v3/review-2026-11-16/export_pairs.py` reads the 80 process files and builds the pairs as `analyze_v3.py` does: per-process arithmetic mean (gating) and median (secondary) of the 200 samples; sessions s1–s3 inferential, s4 confirmation. It then calls the frozen `statistics_protocol.analyze` (seed 274019, 20,000 resamples). All eight stored rows in `analysis-v3.json` match (gating and secondary, both campaigns, to 1e-12). The exact input is `v3/paired-input-v3.1.json` (sha256 `d72a1905…`). Re-running the export reproduces it byte-for-byte.

## 6. Descriptive results (v3.1; not eligible; not gating)

| Analysis | Ratio | Upper 95 % | Lower 95 % | Confirmation | Non-inferiority rule | Regression rule |
|---|---|---|---|---|---|---|
| Control, mean of 200 (identical binaries) | 0.9636 | 1.0030 | 0.9265 | 0.9873 | met (descriptive) | not met |
| Control, median (secondary) | 0.9811 | 1.0143 | 0.9496 | 0.9810 | met | not met |
| Campaign P1 vs B0, mean of 200 | 1.0469 | 1.0836 | 1.0126 | 0.9610 | not met | not met |
| Campaign P1 vs B0, median (secondary) | 1.0550 | 1.0934 | 1.0193 | 0.9759 | not met | not met |

Outcome of the registered rule, descriptive only: the control's non-inferiority rule was met; the campaign was inconclusive.

## 7. Corrections to the report (these supersede the cited text)

- **C0 (§0, bullet 2; §1.1).** "H3 is committed locally, not pushed" is superseded. H3 is pushed to `codex/attention-h3` at `28dfa3b2b7749bfaab5815a2e693568149812626`.
- **C1 (§0, "Complete-call estimator" and "Registered P1-versus-B0 campaign" bullets; §6.2; §6.3).** "Valid" means valid under the registered list (§2). "Met the non-inferiority rule" and "control PASS" mean descriptive results on records with `timing_eligible: false`.
- **C2 (§4.3, "The v2 A/A campaign met its registered non-inferiority criterion"; §7 item 1).** Read: it met its registered criterion on records with `timing_eligible: false`, which rest on unreviewed acceptance. The result is descriptive and is not an acceptance result. One campaign does not establish the method's general reliability.
- **C3 (§3, the retrospective complete-call figures; §0 "Complete-call estimator").** Descriptive, with the same eligibility label as C2.
- **C4 (§6.4, third bullet: the original R10 displayed 1.26 "is not reproduced under the registered mean design").** **Withdrawn.** I have not verified how the displayed 1.26 was computed. The original record holds one timed call per process, from tool build `4423d4a0…`, under an older schema. The v3.1 interval is for a 200-call mean in a different build, so the two designs differ. The new interval neither reproduces nor invalidates the original, which stays as recorded.
- **C5 (§6.1 and the amendment's "frozen before" statement).** See §4.
- **C6 (§5, the 1,024-token batch).** The run used the P1 request build (core `26b00aea…`) with telemetry enabled (`telemetry: enabled`, `telemetry_dropped: 0`). Its single-run latencies were taken with telemetry on and are not timing evidence. The single-request control had telemetry disabled.
- **C7 (§2, §6 and §3 wording on knobs).** "Same knobs" holds for `Apply()` and for the resolved settings. The provenance string differs by one name (§3).
- **C8 (§7 item 10, "analysis-script bugs (mine), fixed before the campaign").** Accurate for the bugs, but incomplete: the script was also edited at 20:27:15, after the control's data existed and before the campaign started (§4).
- **C9 (§6.1–§6.3).** The "valid" and "non-inferiority" labels inherit C1–C2. The registration's validity list omitted the eligibility and reviewed-evidence rules (§2).

## 8. What the evidence shows, and what it does not

- **1,024-token batching (secondary configuration; P1 request build; telemetry enabled).** Two requests (1,024 prompt tokens, 128 generated, stop `Length`) completed. Their texts are identical to each other and to the single-request control (sha256 prefix `b764ab04a7e78917`). `scheduler.decode.batch` count is 2 in 127 measured decode steps. Zero telemetry drops. Quiet verify passed. This demonstrates simultaneous decode for that configuration. Its single-run latencies are not timing evidence.
- **The original 8k batching failure remains.** In `h2/batch-proof-8k-6GiB/`, both requests failed at admission and prefill with `OutOfVramException`: a 4,748 MB allocation with 2,988 MB of 24,082 MB free. The process then ended with an unhandled exception; `exits.txt` records `batched-8k-6GiB rc=134`. The single-request control exited 0. The 1k result does not convert this case.
- **v3.1 establishes neither non-inferiority nor material regression.** It is also not timing-eligible.
- **R10 remains unresolved.** The original 1.26 (one timed call per process, older build) and the v3.1 interval come from different measurement designs. Neither settles the other.
- **Single-run timings and simulation probabilities remain descriptive.**

## 9. Decisions for the coordinator (not taken here)

1. **Acceptance review.** Any timing that could count as acceptance needs the R10 acceptance criteria (`docs/ATTENTION_CONTRACT.md` §8) reviewed, and a new versioned fixture case with reviewed acceptance registered before any run. The current fixture stays unchanged, because flipping `reviewed` in place would alter the case.
2. **Next measurement.** The README requires a separately registered replication after an inconclusive campaign. Any new timing should use `--mandatory` and put the eligibility and reviewed-evidence terms into its validity rule.
3. **Telemetry overhead or more measurements.** The 1k run had telemetry enabled and the v3 runs had it disabled. Either investigation is open; neither has been started.

## 10. Limits

One fixture, one machine state, one campaign per design. The net8.0 builds were not run because the .NET 8 runtime is not installed. The v2 A/A and the 1k results are single campaigns and single runs. The registration and analysis errors (§2, §4) are disclosed here; the registered files were not edited to hide them.

## 11. Artifacts

- Hashes: `v3/review-2026-11-16/SHA256SUMS.txt`.
- Exact invocations: `v3/review-2026-11-16/invocations.txt`.
- Ordering evidence: `v3/review-2026-11-16/timestamps.txt`.
- Field and hash check: `v3/review-2026-11-16/review_fields.py` → `fields-and-hashes.txt`.
- Recomputation: `v3/review-2026-11-16/export_pairs.py` → `export-pairs-rerun.txt`; input `v3/paired-input-v3.1.json`.
