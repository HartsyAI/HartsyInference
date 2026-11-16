# Timing analyzer v4: fail-closed rules

`timing_analysis_v4.py` replaces the analyzers used for the v3 campaigns. It does not replace or edit them. The v3 script
(`analyze_v3.py`) and every historical output stay as they are.

## Modes

- `--mode acceptance` produces a claim only when every process and both campaigns pass every rule below. A refusal writes a
  report with `claim_status: refused` and the reasons, and exits with status 2.
- `--mode descriptive` produces the same numbers for records that fail the eligibility, review, memory and status rules. Every
  result is labelled `descriptive only`, and no acceptance claim is made. Integrity rules still apply.

## Integrity rules (both modes)

1. The exit ledger (`timing-exits.txt`) is complete and unambiguous, and it records status 0 for the process. See *Ledger*.
2. The record parses as a JSON object. A malformed file, a `null` or a non-object refuses.
3. Mode, boundary, branch, route, arm, legacy assertion, query and probability permissions, telemetry state, cohort and device
   equal the registration. A `diagnostic-trace` record always refuses.
4. Quality passed.
5. Fixture and tool hashes equal the registration.
6. Core and backend hashes equal the registration for that campaign and role (candidate or reference).
7. Resolved-settings, knob-provenance, native-library, PTX and SPIR-V identities equal the registration for that role.
   Each identity is a SHA-256 of canonical JSON, or of the knob string, computed from the record.
8. The samples are a list of exactly `trials` finite, positive numbers.
9. The layout is complete: 40 processes per campaign, two campaigns.
10. The A/A control binds identical candidate and reference identities. The registration is refused at load time if they differ
    in any identity field.

## Acceptance only

11. **Eligibility:** `timing_eligible` is true and `acceptance_reviewed` is true.
12. **Memory evidence:** the process log has no out-of-memory, allocator-retry or `[CudaMemory]` warning.
13. **Campaign status:** `status.txt` has exactly one `job rc=<n> watch rc=<n>` line, with both values 0; exactly one
    `quiet verify passed` and no `quiet verify FAILED`; exactly one `BATCH DONE`; and no line of unrecognised form.

The split is deliberate. Ledger findings apply in both modes, because they say which exit status belongs to which process. Status
findings apply to acceptance only. The batch script appends the watcher's and the verifier's own output to `status.txt`, so a
line of unrecognised form there (for example a traceback) is a reason not to accept the batch. It does not make the timed records
any less of a measurement.

## Ledger

`timing-exits.txt` must be a complete and unambiguous record of one job. Every non-blank line is matched in full against one of these
forms. A line that matches none is a finding. It is never skipped.

- `job start <timestamp>`, at most once.
- `<job> <process> <status> [timing fields]`. `<job>` is the campaign's `job_id`, `<process>` is one of the 40 process names, and
  `<status>` is an integer. The timing fields are not parsed.
- `<job> JOB DONE <timestamp>`, exactly once, for this job.

Each process name may appear at most once. A second entry refuses, whatever its status, so a later success cannot erase a failed
attempt. The ledger also refuses:

- a line for another job;
- an unexpected process name;
- a non-integer status;
- a line of unrecognised form;
- a missing `JOB DONE`;
- a missing ledger file.

These ledger findings refuse in both modes. A missing process entry refuses as `no exit status recorded`.

## Status

`status.txt` is read in acceptance mode only. These lines are known:

| Line | Meaning |
|---|---|
| `lock acquired <ts>`, `quiet wait passed <ts>` | steps passed |
| `lock failed <ts>`, `quiet wait failed <ts>`, `watcher not ready <ts>` | a step failed; the job did not run and acceptance refuses |
| `job rc=<n> watch rc=<n> <ts>` | the job's exit and the watcher's exit; exactly one line |
| `quiet verify passed`, `quiet verify FAILED` | the quiet-window verification |
| `BATCH DONE <ts>` | the batch finished; exactly one line |

Any other line refuses acceptance. Two job lines refuse acceptance as ambiguous. A watcher that exits nonzero refuses acceptance
even when the verification passed.

## Registration

A registration is a JSON file written before the campaign runs. The analyzer validates it and never writes one. It must carry
`schema: timing-registration-v4`, `trials`, `statistic` (`mean` or `median`), `seed` 274019, `resamples` 20000, thresholds of 1.03
for both claims, an `expected` block, and per-campaign `job_id`, `candidate` and `reference` identities for `control` and
`campaign`. The identity fields are `core_sha256`, `backend_sha256`, `resolved_settings_sha256`, `knobs_sha256`,
`native_libraries_sha256`, `ptx_sha256` and `spirv_sha256`.

For the `control` campaign, the `candidate` and `reference` identities must be identical in every field. A control that runs
two different binaries is not an A/A control, and the registration is refused.

## Output

Write-once. The `--out` path must not exist, and it must not lie inside an input folder. The file is written as UTF-8 with LF line
endings. The report records the analyzer hash, the registration hash, each campaign's exit-ledger and status hashes, and each
process's input hash and reasons. Results appear only when the claim status is `acceptance` or `descriptive only`.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Report written: `acceptance`, or `descriptive only` |
| 2 | Report written with `refused` |
| 1 | No report written: bad registration, existing or misplaced output path, or unreadable input |

## Verification

`test_timing_analysis_v4.py` covers ineligible records, missing fields, each identity, diagnostic traces, `null` and invalid
records, failed quality, malformed samples, memory events, layout gaps and write-once output. It also covers the ledger (a failed
attempt followed by a success, a success followed by a failed attempt, identical duplicates, unexpected names, non-integer
statuses, other-job lines, unrecognised lines, a missing `JOB DONE`, a missing file, and a process line with no trailing fields),
the status rules (a failed watcher with a passed verify, a failed job, a failed quiet verify with its traceback, two job lines, an
unrecognised line, a malformed job line, a failed batch step), and the A/A control (every identity field). It checks the exact
agreement of the descriptive mirror with the frozen function. It also runs the real v3.1 records when they are present. There,
acceptance is refused for exactly the two eligibility reasons, with no ledger or status finding, and the descriptive results match
the stored `analysis-v3.json` values.
