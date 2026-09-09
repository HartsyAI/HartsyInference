# Community benchmarks

Run a frozen workload on your own GPU, keep every trial and output, then submit the evidence by PR.
The [explorer](https://hartsyai.github.io/HartsyInference/) shows reviewed, merged, intact campaigns.
Its [dataset](https://hartsyai.github.io/HartsyInference/summary.json) and
[README image](https://hartsyai.github.io/HartsyInference/overview.svg) are generated from the same records.
Until Pages is enabled and results are accepted, the repository's [snapshot](generated/overview.svg) is empty.

## Run and contribute

Use a clean checkout of an exact engine commit and .NET 10 SDK, or a self-contained runner from the
[benchmark runner Releases](https://github.com/HartsyAI/HartsyInference/releases). Maintainers publish those
with [Release benchmark runner](../.github/workflows/benchmark-release.yml); CI artifacts also support pre-release testing. Keep its Ptx, Spirv and suites folders.
The GPU driver must work inside the pod. `doctor` probes the selected backend; it never falls back to CPU.

```bash
git rev-parse HEAD
dotnet publish benchmarks/HartsyInference.BenchmarkRunner -c Release -r linux-x64 --self-contained true -p:BenchmarkRevision=<commit-sha> -o artifacts/linux-x64
./artifacts/linux-x64/hartsy-bench doctor --device cuda:0
./artifacts/linux-x64/hartsy-bench fetch --suite standard-v1 --cache benchmark-cache
./artifacts/linux-x64/hartsy-bench run --suite standard-v1 --device cuda:0 --cache benchmark-cache --output benchmark-runs/run-1
./artifacts/linux-x64/hartsy-bench validate --input benchmark-runs/run-1
./artifacts/linux-x64/hartsy-bench export --input benchmark-runs/run-1 --bundle benchmark-runs/run-1.zip
```

Replace `<commit-sha>` with the full hash from `git rev-parse HEAD`; an unversioned development build is
ineligible for public comparisons. On Windows publish with `-r win-x64` and run `hartsy-bench.exe`.
Use `vulkan:0` for Vulkan, or `cpu` for local diagnostics. Software Vulkan devices are excluded from GPU comparisons. Run `list` to see suites. A second GPU uses an
explicit ordinal and a separate campaign directory. Do not run campaigns concurrently on one device.

The standard execution budget is 30 minutes, excluding the explicit download stage. `--minutes` changes
the execution budget, not the workload. `resume` accepts the same options as `run`, uses another budget
window, retains failed attempts and completed sessions, and refuses changed binaries, settings or devices.
First-request timing excludes checkpoint hash verification and worker startup; it includes model loading.
Timeout, crash, unsupported, OOM and budget skips remain visible. A nonzero exit does not erase evidence.

Inspect the saved text/PNG outputs and JSON before submission. Raw worker logs stay local and are not
exported; structured failure types are exported. Model weights and credentials are never bundled.
GPU UUIDs are hashed; a persistent local benchmark identity groups the host's runs.

With `git`, authenticated `gh`, and an existing public fork:

```bash
./artifacts/linux-x64/hartsy-bench submit --input benchmark-runs/run-1 --bundle benchmark-runs/run-1.zip --repository . --fork YOUR_LOGIN/HartsyInference
```

This explicit command uploads an immutable Release to your fork, creates an isolated worktree using your
Git identity, and opens a PR containing only two JSON files under `benchmarks/submissions/<campaign-sha>/`.
It does not add attribution trailers. If interrupted, it retains the worktree and staging release for recovery.
For manual submission, consult the top-level `--help`: upload the ZIP yourself,
then use `submission --input ... --bundle ... --output benchmarks/submissions --staging <release-url>`.

## Protocol and interpretation

| Suite | Workloads | Sessions | Public comparison |
|---|---|---:|---|
| quick-v1 | Qwen3 4B Q4_K_M, short context | 1 | No; diagnostic only |
| standard-v1 | Qwen3 short and long context; SD 1.5 FP16 at 512² | 3 per case | Complete cases only |
| extended-v1 | Standard cases plus SD 1.5 FP16 at 768² | 3 per case | Separate suite; 120-minute default budget |

The [JSON schemas](schemas/) document the strict source-generated contracts; the validator also checks cross-file invariants.
The frozen [suite manifests](suites/) pin repository revisions, full model SHA-256 values, sizes, inputs,
seeds and shapes. The long prompt repeats the manifest's exact prefix before each input; it does not guess
a token length. Actual prompt token counts are recorded. Each fresh worker performs two warmups, retaining
the first separately, then five fixed inputs. Text uses greedy decoding, thinking/graphs/speculation off,
and a 128-token maximum with natural EOS retained. Images use Euler, normal scheduling, CFG 7.5, batch one,
and 20 steps. No precision, model, dimensions or length may silently change to fit VRAM.

Measurements retain monotonic clock frequency/ticks and every native token timestamp. Validation recomputes
request latency, time to first token, and decode rate `(tokens - 1) / (last - first)` from these traces;
stream chunks are not tokens. Prefill completion includes first-token sampling. Image latency includes the
engine's complete returned RGB output. File encoding, hashing and quality checks occur outside warm timing.
Host memory is the worker's cumulative peak working set, not incremental model memory. GPU memory is currently
explicitly unavailable; it is not reported as zero or inferred from total VRAM.

The registry's default knobs are pinned in a request scope; resolved defaults, selected environment controls,
managed/runtime/kernel hashes and loaded CUDA/Vulkan library hashes are retained. CUDA driver values are the
Driver API version, Vulkan values are vendor-specific. Power/clocks and background utilization are currently
operator-controlled and unverified; disclose unusual power limits or sharing in the PR. These runs do not
support energy-efficiency or cost claims.

Automated checks establish structural consistency and detect obviously broken output. They do not establish
semantic/numerical parity or prove that a contributor's timings are honest. Maintainers inspect every output,
configuration and provenance before acceptance. Independent reproduction is a stronger, separate claim;
no automatic “independently verified” badge or external-engine speedup is currently emitted.

The explorer groups identical suite, case, engine revision, backend, GPU capacity/name, driver, OS/runtime,
settings, native libraries and binary hashes. Each session contributes its warm-input median; each physical
GPU contributes the median of its sessions; the chart reports the median across those GPUs. Repeated uploads
use the earliest accepted campaign for that cohort, never the fastest. The 95% interval is a deterministic
2,000-resample percentile bootstrap across GPUs. A single GPU has no population uncertainty estimate.
First-request latency, TTFT, decode rate and all attempts remain in the downloadable evidence.

## Review and publication

1. The data-only PR check uses trusted base-branch code. It checks the two-file/2 MiB metadata limit, downloads
   at most 256 MiB from GitHub Releases, rejects unsafe ZIPs and verifies every hash, output and protocol field.
2. Inspect the bundle's outputs and provenance. Run **Review benchmark evidence** on `main` with the PR number
   and the explicit output-review confirmation. It validates again, requires an engine commit reachable from
   `main`, mirrors the ZIP without replacement, and archives an immutable review receipt bound to the PR head.
3. Merge the reviewed PR. A new push requires a new review. Publication requires a matching receipt and the
   exact merged PR head; stale checks cannot make changed evidence appear in the dataset.
4. **Publish benchmark explorer** re-downloads and validates the mirrored evidence, compiles the static dataset
   and visual, and deploys Pages. It also checks archives weekly. Missing/corrupt bundles are excluded.
5. **Withdraw benchmark evidence** appends a receipt; it never edits the original submission or artifact.
   Run the publication workflow afterward to update Pages immediately. A replacement is a new campaign/PR.

Repository setup: enable Pages using GitHub Actions and require `benchmark-evidence-reviewed` for result PRs
(using the repository's review/ruleset policy). The archive Release is created by the first trusted review.
These settings require repository administration and are not changed by building this branch. Generated
Pages artifacts are the live source; `generated/overview.svg` is only the initial checked-in empty snapshot.

## Extend without breaking comparisons

Add a new suite version instead of editing a published workload. Pin every additional checkpoint and auxiliary
asset. Add a modality adapter through the Engine service and define output checks, exact request controls,
timing boundaries and failure states. The current worker dispatch supports text and image only. Add schema
versions for incompatible evidence changes and retain validators for accepted historical suites.

Optional reference-engine lanes, sampled GPU peak memory, load/prefill breakdown charts, independently
reproduced badges and multi-device workloads need their own pinned execution and quality contracts before
publication. The current system intentionally makes none of those claims. The existing reference scripts
remain available for separate investigations; their historical scores are not imported as community evidence.

## Historical and diagnostic tools

[Historical scoreboards](scoreboards/README.md) retain their original hardware/settings/limitations.
`run_benchmarks.sh`, `analyze.py`, `profile.sh`, BenchmarkDotNet projects, and `python-baseline/` remain available.
They are not the community submission protocol. Read [profiling methodology](../docs/Research/PROFILING_METHODOLOGY.md)
for kernel investigations and the [parity ledger](../docs/Checklists/PARITY_VERIFICATION.md) for numerical correctness.
