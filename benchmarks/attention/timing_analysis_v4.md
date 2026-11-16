# Timing analyzer v4: fail-closed rules

`timing_analysis_v4.py` replaces the analyzers used for the v3 campaigns. It does not replace or edit them. The v3 script
(`analyze_v3.py`) and every historical output stay as they are.

## Modes

- `--mode acceptance` produces a claim only when every process passes every rule below. A refusal writes a report with
  `claim_status: refused` and the reasons, and exits with status 2.
- `--mode descriptive` produces the same numbers for records that fail the eligibility, review and memory rules. Every
  result is labelled `descriptive only`, and no acceptance claim is made. Integrity rules still apply.

## Rules (both modes unless marked)

1. The exit ledger (`timing-exits.txt`) shows status 0 for the process. A missing or nonzero status refuses.
2. The record parses as JSON. A malformed file refuses.
3. Mode, boundary, branch, route, arm, legacy assertion, query and probability permissions, telemetry state, cohort and
   device equal the registration. A `diagnostic-trace` record always refuses.
4. Quality passed.
5. Fixture and tool hashes equal the registration.
6. Core and backend hashes equal the registration for that campaign and role (candidate or reference).
7. Resolved-settings, knob-provenance, native-library, PTX and SPIR-V identities equal the registration for that role.
   Each identity is a SHA-256 of canonical JSON, or of the knob string, computed from the record.
8. The samples are a list of exactly `trials` finite, positive numbers.
9. **Acceptance only:** `timing_eligible` is true and `acceptance_reviewed` is true.
10. **Acceptance only, memory evidence:** the process log has no out-of-memory, allocator-retry or `[CudaMemory]` warning.
11. **Acceptance only, campaign evidence:** `status.txt` has `quiet verify passed` and a `job rc=0` line.
12. The whole layout is present: 40 processes per campaign, two campaigns.

Validity is derived from these rules. A pair is marked valid only when both of its processes pass. A successful exit
never makes a pair valid, and the frozen function `statistics_protocol.analyze` still refuses any pair without `valid: True`.

## Registration

A registration is a JSON file written before the campaign runs. The analyzer validates it and never writes one. It must
carry `schema: timing-registration-v4`, `trials`, `statistic` (`mean` or `median`), `seed` 274019, `resamples` 20000,
thresholds of 1.03 for both claims, an `expected` block, and per-campaign `job_id`, `candidate` and `reference` identities
for `control` and `campaign`. The identity fields are `core_sha256`, `backend_sha256`, `resolved_settings_sha256`,
`knobs_sha256`, `native_libraries_sha256`, `ptx_sha256` and `spirv_sha256`.

## Output

Write-once. The `--out` path must not exist, and it must not lie inside an input folder. The report records the analyzer
hash, the registration hash, each campaign's exit-ledger and status hashes, and each process's input hash and reasons.
Results appear only when the claim status is `acceptance` or `descriptive only`.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Report written: `acceptance`, or `descriptive only` |
| 2 | Report written with `refused` |
| 1 | No report written: bad registration, existing or misplaced output path, or unreadable input |

## Verification

`test_timing_analysis_v4.py` covers ineligible records, missing fields, each identity, diagnostic traces, failed quality,
malformed samples, memory events, campaign evidence, layout gaps, write-once output, and the exact agreement of the
descriptive mirror with the frozen function. It also runs the real v3.1 records when they are present. There, acceptance
is refused for exactly the two eligibility reasons, and the descriptive results match the stored `analysis-v3.json` values.
