# Attention evidence program

PR 01 is observation and measurement work. PR 02 and the proposed implementation queue require the PR 01 exit
review. Production arithmetic, admission, memory budgets, workspaces, ordering, synchronization and fallback
selection must remain unchanged. Kernel sources and shipped PTX/SPIR-V stay in their existing locations.

The frozen endpoints, baseline/Kitchen revisions and statistical constants live in [protocol.json](protocol.json).
The independent semantic authority remains [the attention contract](../../docs/ATTENTION_CONTRACT.md).
These tools complement [OpParity and GenCapture](../../docs/VERIFICATION.md) and the existing
[benchmark runner](../README.md); the community suites are different workloads and cannot replace the primary endpoints.

## Build identities

Create a detached worktree at `baseline_commit` from the protocol. Build this PR's harness against that tree:

```bash
dotnet build benchmarks/HartsyInference.GpuBenchmarks -c Release -f net10.0 \
  -p:AttentionProductionRoot=/absolute/path/to/pristine-worktree -o /external/evidence/b0-operation
dotnet build tools/GenCapture -c Release -f net10.0 \
  -p:AttentionProductionRoot=/absolute/path/to/pristine-worktree -o /external/evidence/b0-request
dotnet build benchmarks/HartsyInference.GpuBenchmarks -c Release -f net10.0 -o /external/evidence/diagnostic-operation
dotnet build tools/GenCapture -c Release -f net10.0 -o /external/evidence/diagnostic-request
```

This substitutes **all production project references and kernel artifacts**, retaining only the measurement tool
from PR 01. The pristine assemblies have no `AttentionEvidence` type. Keep production assembly, native library,
kernel, tool, fixture and settings identities separately. Never call instrumented assemblies pristine B0.
Always select one target framework when using a common output directory.

Telemetry is default off. `AttentionEvidence.Begin(capacity)` allocates a bounded host-only buffer before observed
work; overflow or an unfinished writer invalidates the capture. Synchronous contexts do not cross `await` or graph
replay. Graph evidence needs both capture-time implementation attribution and actual replay observations. Host
positions, logical preparation bytes and candidate lowering declarations are distinct from device positions,
measured bandwidth and proof that preparation executed. Diagnostic tensor captures remain separate.

## Shared fixtures and operation cells

Use an external artifact directory. The example creates a pilot, **unreviewed** fixture:

```bash
python benchmarks/attention/evidence.py fixture --id pilot --heads 2 --kv-heads 2 \
  --sq 32 --skv 48 --dim 64 --dtype f16 --precision f16 --out /external/evidence/pilot
python benchmarks/attention/evidence.py verify-fixture /external/evidence/pilot/fixture.json
dotnet /external/evidence/diagnostic-operation/HartsyInference.GpuBenchmarks.dll attention-table \
  --fixture /external/evidence/pilot/fixture.json --boundary public --trials 1 --cohort COHORT \
  --out /external/evidence/result.json
```

Set `HARTSYINFERENCE_BENCH_BACKEND=vulkan` for Vulkan and its existing device selector when needed. Both
`attention-table --fixture` and `attention-evidence` use the shared runner. `--boundary resident` uses device
timestamps; `--observe /external/evidence/observations.json` enables the lightweight collector **during warmups**.
After recording the initial route, pass `--expect-branch ID` for later arms. Use fresh output paths; evidence is
immutable. Oracle comparisons, tensor loading and all hashing occur outside timed regions.

OpParity consumes the identical bytes:

```bash
dotnet run --project tools/OpParity -c Release -f net10.0 -- shared-attention \
  --fixture /external/evidence/pilot/fixture.json --backend cuda:0 --out /external/evidence/parity.json
```

The C# pilot reader currently certifies only its sampled dense oracle's supported cases. Unsupported mask,
metadata, differing value dimensions and nonstandard valid-length cases fail explicitly; they need oracle/adapter
extensions before the complete corpus can run. The Python fixtures also support boolean/int32 metadata,
broadcast masks, strided storage, speculative prefix state and Sol-Attn metadata. A common random seed is never
treated as equivalent input storage. New tolerances remain `reviewed: false`; `--mandatory true` rejects them.

```bash
python benchmarks/attention/evidence.py matrix --kitchen /external/kitchen-K0 --cohort COHORT \
  --out /external/evidence/matrix.json
python benchmarks/attention/evidence.py cell-fixture --matrix /external/evidence/matrix.json \
  --cell tree-merge --out /external/evidence/tree-merge
```

The generated cells are source-derived probes, **not a completed capability enumeration**. Review API/native
constraints, upstream tests, queried runtime capabilities, exact transforms, formats/scales and output policies
per cohort before freezing the matrix. Source availability and a valid public GPU fallback are separate facts.
Preserve K0-valid, K0-rejected, K0-invalid and Not executed cells. Missing hardware or a build failure is Not
executed. Required K0-invalid exceptions need exit review; never silently omit them or relabel a patched K1 as K0.

## Isolated Kitchen environment

Use a dedicated virtual environment with user site disabled, never the shared ComfyUI environment. Clone the
exact protocol commit, download pinned dependency wheels into an external wheelhouse, install from that
wheelhouse, and build a wheel from the clean pinned source. Keep full configure/build logs and compiler settings.
For the current source, `COMFY_CUDA_ARCHS` can restrict a cohort build; record it. Install that wheel into the
isolated environment, retain its hash, then lock:

```bash
/external/venv/bin/python -s benchmarks/attention/evidence.py lock-environment \
  --kitchen /external/kitchen-K0 --wheelhouse /external/wheelhouse --out /external/evidence/environment.json
/external/venv/bin/python -s benchmarks/python-baseline/bench_attention.py \
  --fixture /external/evidence/pilot/fixture.json --environment /external/evidence/environment.json \
  --api sage --boundary resident --trials 1 --out /external/evidence/kitchen.json
```

An unavailable extension records `native_ready: false`; a clean Python-only K0 can still probe public APIs with
valid GPU fallbacks. An incompatible import, source mismatch or unavailable CUDA writes a non-ready lock and
exits unsuccessfully. Native availability never establishes execution through a fallback. The lock records
executable, dependencies, wheels, installed Python source, extension and Torch-native hashes plus runtime GPU
queries. Before a mandatory campaign, review compiler flags and native identities, validate complete-operation
versus resident-device timing, prepared-state reuse accounting, and output-format/precision equivalence. An
unreviewed pilot result cannot establish a matched-quality Kitchen gate. Partial-block Sol rows explicitly marked
dead have unspecified output and are excluded from its numerical comparison.

## Requests and equivalence

`GenCapture attention-freeze-qwen --model FILE --out DIR` uses the production GGUF tokenizer/template to create
an ordinary request candidate, its rendered token hash and a separately hashed embedded-tokenizer description.
Run it with the pristine tool, then prove natural token-limit completion before accepting the prompt. No raw token
IDs, forced tokens, retained prefix, score collection or teacher forcing enter this primary request.

```bash
dotnet /external/evidence/b0-request/hartsy-gencapture.dll attention-request \
  --workload /external/evidence/qwen/workload.json --backend cuda:0 --lane paged \
  --out /external/evidence/qwen-b0.json
```

The request tool loads the model, performs the protocol warmups, then times the public Engine result including
request-owned scheduler cleanup. It rejects early EOS or changed lengths. `dense` and `graph` lanes are separate;
verify replay rather than assuming admission. The separate `batched` lane submits two concurrent ordinary requests
and records group throughput plus each request's latency; its trace must prove actual batching. Request-level
memory/TTFT sampling still needs campaign integration before exit. Krea manifests must pin DiT, encoder, VAE and auxiliary assets;
the request specifies the protocol image settings and rejects a different recipe variant or seed. Its PNG is
saved beside the result after timing, with pixel and artifact hashes. Freeze quality gates before comparing candidates.

`telemetry_check.py` compares pristine/disabled/enabled operation results, complete output hashes and existing
top-level route fields. It rejects divergence; its small-run overhead ratios are descriptive. It cannot establish
all nested routes, real-generation identity or memory-pressure equivalence. Those remain separate required runs.

## Timing, campaign review and shared hardware

Use the exact [protocol](protocol.json): separate device/public/preparation cells, warmups, issue policy,
inferential sessions and independent confirmation. CUDA events use the backend stream. Vulkan uses the reported
timestamp period, validates queue-family valid bits and brackets automatic command-buffer flushes. The current
baseline reader accepts 64 valid bits; other widths are explicitly unmeasured until a reviewed masked reader
exists. Empty-region checks are recorded; a longer controlled-work validation remains required. Timestamp intervals
include host gaps. Never substitute wall timing for an invalid device interval.

`evidence.py statistics --pairs FILE --out FILE` runs the fixed bootstrap. Its `valid` pair flags must come from
reviewed quality/route/cache/memory/cohort evidence. The calculator does not collect or certify those facts.
An inconclusive campaign requires a separately registered replication; keep failed confirmation sessions.

Kitchen timing runs omit `--diagnostics`. A separate diagnostic run records executed Python functions, ATen
operations and GPU kernel names when CUPTI is available; it is marked ineligible for timing acceptance. Returned
outputs are released before replacement; the final output is retained for untimed validation. Prepared state is
retained only in declared reuse calls, and preparation measurements do not retain an unrelated prepared copy.

On the 4090, use the existing GPU lock through `run_attention_table.sh`, its quiet-window wait/watcher and foreign
client checks. Quiet evidence must bracket every timed arm without gaps. Activity invalidates rows before table
insertion. Missing status counters or authentication fail closed. Never stop Swarm, evict its models or restart
shared GPU state. Perform diagnostic and timing runs separately and run only one GPU suite at a time.

The [local status](reports/pr01-local-3060.json) records incomplete evidence. PR 01 stays draft until workload
proof, numerical definitions, per-cohort runtime matrix, baseline campaigns, explicit exceptions and all three
commit groups pass exit review. Later work remains in the [proposed queue](proposed-queue.md).
