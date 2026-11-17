# RTX 4090 evidence review: PR 01 and PR 02

Both PRs remain draft. PR 03 depends on them and cannot merge ahead of them. The user-provided remote report is
evidence to audit, not a completed exit review. Its raw artifacts are external at
`/home/hartsy/evidence/attention-4090/run-20261115-a`, indexed by `INDEX-SHA256.txt`.
The received progress/report transcript has SHA-256
`a8ca7ca6ebcc8bf703de08c5173100b802a858f7ce75cdefdca0e02619948ec9`.

Production pins remain B0 `a6c465bc8aa9121aadcb9d088ddc72e815ae4288`,
P1 `3ee783e769b102234cfc7be1ed65f94ac9d984d7` and P2 `57b143f004050d484dc1b667ae8d5f91fc82ac22`.
The report records 45 passing semantic cases, natural 8192/128-token Qwen completion, paged-path evidence on CUDA
and Vulkan, P1 graph replay, and primary output identity for P1 on both backends. It reports P2 Qwen request
non-inferiority against B0; that does not establish an optimization gain or Bfix. Full raw results have not been
independently audited locally.

## Blocking findings

- P2 Krea CUDA fails the existing identity gate; Vulkan remains identical. Keep the CUDA result blocked and find
  the first divergent encoder/latent/denoiser/VAE component on identical inputs. A plausible image does not clear
  the gate. The remote gate document was written after a difference was known, although before its magnitude was
  calculated; record that ordering and do not derive a replacement tolerance from candidate errors.
- The two-request lane admits paged requests but records only single-sequence decode. It does not prove batched
  execution. Krea still needs executed-path/GQA/layout/numerical-policy evidence, and pristine B0 graph replay
  needs independent diagnostic evidence.
- Flash-labeled operation cells report a Dense-route branch. Source inspection confirms that the old shared
  runner applied arm knobs without validating its entry, dtypes or numerical assertion against the fixture, and
  could learn its expected branch from the candidate. Audit every recorded call before interpreting those
  campaigns. Preserve mislabeled cells as unvalidated evidence; do not silently relabel or discard them.
- Most operation campaigns do not establish non-inferiority. Failed confirmation sessions remain failed; wide
  intervals do not alone prove material regression. Register any replication separately. Report geometric means
  for the same 15 inferential pairs as the ratio, and list the five confirmation pairs separately.
- Qwen P1/B0 and request-level observation-overhead campaigns remain registered but unexecuted. Operation overhead
  does not establish request-level overhead. Request-specific peak VRAM and paged TTFT/decode breakdowns remain
  unmeasured; retain the reported OOM/eviction events and aggregate memory samples.
- Kitchen has 34 Python-fallback pilots with unreviewed tolerances, two genuine admission rejections, and 125
  unexecuted cells requiring unavailable native coverage. Correct the report table that groups missing-native
  cells under rejected. No matched-quality Kitchen performance gate is established.
- Missing SANA-WM references and MOSS-VL assets remain unvalidated. The report reproduces a Kimi numerical failure
  on B0; retain it as pre-existing evidence, not a passing test.

## Harness correction and next evidence

The corrected shared runner validates immutable fixture route, Q/O and K/V dtypes, the legacy assertion and any
explicit catalog grant before backend creation. It does not rewrite inputs or production admission. Timing
requires a registered branch; `observe` is discovery only. Results now include actual call metadata, and unknown
specialized fixture semantics remain rejected. The strict F32 catalog arm is available in the PR 01 harness so
all production arms can use one tool source without borrowing PR 02 changes.

Use a separately pinned corrected harness against the immutable production roots, with independent tool and
production hashes. Audit old campaigns before proposing narrowly scoped reruns; never mix harness versions in
one campaign. Retain GPU worktrees and raw artifacts for the investigation. These results do not establish Bfix
or clear the merge gates. The harness correction changes no production source or numerical tolerance.

Local correction checks: 203 Core CPU cases pass on each of net8.0 and net10.0, including 16 arm-contract cases;
18 Python tests pass. The benchmark builds on both targets without warnings. Default F32/F16/BF16 fixture
manifests and Q/K/V bytes match the preceding harness exactly. Actual CLI probes reject a Dense fixture under a
Flash label, a dtype mismatch and timing without a registered branch, producing no result files and reaching no
backend. No GPU timing or real-generation result is claimed for this tool-only correction.
