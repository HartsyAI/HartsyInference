# Continue PR 01 on the 4090

The local 3060 cannot admit the frozen Qwen candidate on B0: its placement check requires more free memory than
the card reports. Krea primary generation and native K0 also remain unverified. The user will move execution to
the 4090 VPS; no SSH connection details or rental authorization are assumed here.

## Preserve identities and shared state

1. Fetch `codex/attention-evidence-pr01` into its own `.claude/worktrees/` checkout. Preserve unrelated branches,
   worktrees, services and resident models. Retain the local raw evidence folder named in the
   [status report](reports/pr01-local-3060.json), or transfer it unchanged with hash verification.
2. Create a separate pristine worktree at `baseline_commit` from [the protocol](protocol.json). Use the
   [production-root build procedure](README.md#build-identities) for pristine operation and request tools.
   Record the exact tool hashes separately from production assemblies and PTX/SPIR-V.
3. Inspect GPU tenants and existing lock ownership. Use the existing 4090 lock and the quiet-window watcher from
   `run_attention_table.sh`; attach continuous quiet evidence to request campaigns too. Quiet Swarm counters do
   not prove sufficient free VRAM. A placement failure is a failed workload; never unload shared models or stop
   Swarm automatically to obtain a timing sample. Model loading, quiet waiting and hash checks remain untimed.

## Establish the native Kitchen cohort

Use a new isolated environment and wheelhouse. The local Python-only cohort used Python 3.12, PyTorch
2.9.1+cu130 and the exact K0 commit, with package/wheel hashes in its lock. Its native extension failed to build
under CUDA 13.0/MSVC 19.51; do not carry an unsupported-compiler override into a claimed supported toolchain.
Use a supported Linux CUDA/compiler pair, record versions/settings, and build the clean source with
`COMFY_CUDA_ARCHS=89` for this 4090 cohort. Keep the native wheel and full build logs. Do not patch K0 source.

Lock the environment only after installed Python source matches the pinned checkout and capability queries
complete. Record native availability separately from each public API's actual execution. A new platform forms a
new cohort; never pool local Windows fallback measurements with native Linux CUDA results. The local missing
CUPTI kernel trace is explicit: Python fallback calls and CUDA output device were recorded, but native kernel
trace validation remains required on the VPS.

## Complete workload and operation proof

1. Reuse or fetch the Qwen checkpoint through the existing benchmark asset cache. Verify the SHA-256 in the
   [candidate manifest summary](reports/pr01-local-3060.json). Recreate the prompt with the **pristine** freezer
   and verify prompt, embedded-tokenizer and rendered-token hashes before generation. Confirm an ordinary
   request naturally completes exactly at the frozen token limit. If it does not, preserve the failure and review
   the candidate before adopting a new prompt. Never force tokens or time an early EOS as a success.
2. Prove single paged serving, batched paged serving, solo dense eager, solo dense graph and long prefill. The
   request tool's `--lane batched` issues two concurrent requests; observations must show an actual multi-sequence
   decode batch and paged storage for both sequences. `--lane graph` must show fixed storage and actual backend
   graph replay. Capture-time implementation observations are distinct from replay events. Observe model load
   and warmups as well as the measured interval, using its recorded observation offset.
3. Pin Krea Turbo DiT, encoder, tokenizer, VAE, auxiliary assets, prompt and quality gate. The request tool checks
   the actual Turbo recipe, dimensions and seed, then writes PNG/pixel hashes outside timing. Use the existing
   generation capture/compare infrastructure for baseline/disabled/enabled outputs and affected quality checks.
4. Complete request VRAM and TTFT diagnostics using existing benchmark samplers, with separate diagnostic and
   timing runs. Do not replace the primary public request boundary with a stream or teacher-forcing workload.
5. Review and freeze concrete Kitchen cells from pinned API/native constraints, upstream tests and runtime
   queries. The current generated source probes are a starting inventory. Preserve negative cells, including
   Sol D64 and Sage dimensions beyond its public maximum. Complete current Hartsy capability/adapter gaps
   explicitly; do not silently use a different algorithm or a host fallback to fill a result table.

## Finish the PR 01 campaign

Check telemetry output/route identity on pristine, disabled and enabled builds, including pressure-sensitive
routes and nested fallbacks. An observation buffer overflow, wrong graph/cache path, changed route or unmatched
output invalidates attribution. Do not compensate by changing production memory budgets.

Validate empty and longer controlled timestamp regions, queue valid bits, flush coverage, output lifetimes and
preparation/reuse state. The current Vulkan reader explicitly rejects widths other than 64 bits. Native timing
must not be inferred from wall time. Keep diagnostic profiling separate from authoritative timing runs.

Freeze the reviewed numerical definitions, case definitions, output policies, cohorts, endpoints and campaign
schedule before candidate work. Follow the [exact statistical protocol](protocol.json), keep every required cell
and failed confirmation, and never average away a failing operation. PR 01 reports B0/K0 descriptively; Bfix does
not exist until an independently authorized PR 02 is reviewed and completed.

Run all affected CPU/GPU suites sequentially and retain resource guards as unvalidated. Complete all affected
real-generation correctness checks, baseline results, explicit exceptions and matrix amendment review. Keep this
PR 01 draft until all three groups pass the exit review. See the [implementation queue](proposed-queue.md) for
the user's subsequent PR 02 authorization; Blackwell provisioning and PR 03+ remain unauthorized.

For the stacked `codex/attention-semantic-correctness-pr02` branch, also carry the
[PR 02 progress report](reports/pr02-semantic-progress.md) and its external `artifacts/attention-pr02` directory.
Rebuild B0 and the candidate with the same harness and declared numerical policy. Preserve the failed default
F32 probe and every slower smoke cell. Register a fixed campaign before resolving the recorded timing concern;
the local smoke is not a passing non-inferiority result. Complete affected real-generation A/B correctness
alongside the primary Qwen/Krea path proof. The PR 02 candidate must not be labeled Bfix until review and all
required correctness evidence are complete.
