# Timing v3.1: amendment of v3.0 for a setup failure, registered before the new control run

Status: written and frozen before the v3.1 control ran. The v3.0 control attempt is preserved, unedited, in `v3/aa-v3-mean200/`.

## What happened in the v3.0 control attempt
- The reference arm was a folder holding only `HartsyInference.GpuBenchmarks.dll`. The runtime configuration and the dependency assemblies were missing, so the host failed at process start (`libhostpolicy.so` not found; rc 131).
- 20 reference processes failed at start and produced **no measurements**. The 20 candidate processes ran (rc 0). Their timings are preserved and not used, because a control needs both arms.
- Under the v3.0 rules the control was invalid, so the campaign was not run. The gate behaved as registered. The analysis script also crashed on the job-done line of the timing file, which left the gate without a result; that is a parser bug in my script and is corrected below.

## Changes in v3.1 (setup only)
1. The reference arm is the complete B0 H3 operation build folder: `v3/ref-bin-b0-net10.0-op-full/`. Every file in the tree was checked by sha256 against `h3/bin/b0-net10.0-op/` (tree identical; GpuBenchmarks.dll `bb709fdd0625b49a…`).
2. The control id is `AA-v3.1-mean200`, output `v3/aa-v3.1-mean200/`, job `v3/job_aa_v3_1_mean200.sh` (sha256 `7d110967c5dc7b036de797b6d524032de50c2b27b4eb2cb07b51b6cdd7d074f2`).
3. The analysis script ignores non-timing lines in the timing file, and takes the control directory and id as arguments (sha256 `a4a6159cbeb323d5478f9773edf5414323ef354f43e80d8bd5a7d46195365338`). It changes no criterion.

## Unchanged from v3.0
Estimator (per-process mean of 200 retained calls); layout (3 x 5 inferential, 1 x 5 confirmation, 40 processes; order rule); fixture (`f16-mat`, sha256 `5439493c…`, legacy false); arm (`mat-f16-unasserted`); boundary, expected branch S.15, cohort; validity rules; frozen module (seed 274019, 20,000 resamples); decision rules (1.03 both claims); control first; campaign job `v3/job_p1_vs_b0_v3_mean200.sh` (sha256 `a5a47f9c5edf6672f6acc4e8e878dfd0e580bc65113f15379618b56a543779f8`); no resampling; no additional pairs.

## Why this is not result-driven resampling
No measured reference outcome was discarded: the failed attempt has no reference measurements. The correction changes the reference copy, not any outcome. The v3.0 attempt is kept in full and reported.

## Analysis-script corrections found during the v3.1 control gate (made before the campaign ran)
- The secondary median-based analysis passed one session where the frozen rule requires three inferential sessions and one confirmation session. Fixed (the primary, mean-based analysis was not affected).
- The loader treated a missing campaign timing file as an error. It now reports the campaign's processes as not yet available. Fixed.
- These are analysis-code fixes. The statistic, the frozen module, the seed, the resamples and the decision rules are unchanged.
- Analysis script sha256 after these fixes: `a39a66a06f7ee8460661dce2c7cc13ad9a1e38eed3945b0550599dda02be5780`.

## Control outcome (v3.1, identical binaries, complete reference copy)
Valid: 40 of 40 processes; no validity problems. Frozen analysis on mean-of-200 pairs: ratio 0.9636; upper95 1.0030; lower95 0.9265; confirmation 0.9873. Noninferiority rule met; regression rule not met. Reported, not gating.

## Campaign launch
The campaign job `v3/job_p1_vs_b0_v3_mean200.sh` (sha256 `a5a47f9c5edf6672f6acc4e8e878dfd0e580bc65113f15379618b56a543779f8`, equal to the registration) was launched once, after the valid control, under the same gpu_batch procedure. It is not re-run after any result.
