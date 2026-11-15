# RTX 4090 agent handoff: attention PR 01 and PR 02

## Objective and authority

Run the missing hardware evidence so the coordinating agent can decide whether these drafts are ready:

- PR 01: https://github.com/HartsyAI/HartsyInference/pull/905
- PR 02, stacked on PR 01: https://github.com/HartsyAI/HartsyInference/pull/910
- Repository: https://github.com/HartsyAI/HartsyInference

The user requested this test run and will paste your report back to the coordinating agent. Do not merge PRs,
change production behavior, weaken gates, start PR 03, provision paid hardware, or stop shared services.
Report failures and missing resources explicitly. Test infrastructure fixes may be proposed, but keep them
separate from the measured builds and obtain a new source pin before accepting results from changed code.

This is a verification handoff, not an instruction to make the results pass. The original requirement remains:
CUDA is the primary focus; Vulkan must actually execute and work. A CUDA-only report cannot close the gate.

## Source identities

| Build | Immutable production commit | Purpose |
|---|---|---|
| B0 | `a6c465bc8aa9121aadcb9d088ddc72e815ae4288` | Pristine initial baseline |
| P1 | `3ee783e769b102234cfc7be1ed65f94ac9d984d7` | PR 01 observation-only production |
| P2 | `57b143f004050d484dc1b667ae8d5f91fc82ac22` | PR 02 correctness candidate with review fixes |
| K0 | `7d6a22ae8281cd1053a8449530fa0e764a2e51cb` | Unmodified Comfy Kitchen |

The corrected operation harness is separately pinned at `fe12180ee389ab8aeb5720ddf44b041ca9998217` (H).
It rejects specialized fixture APIs/options/operands that the C# dense adapter and oracle cannot implement, and
validates the arm's declared route, dtypes and numerical assertion against the immutable fixture before GPU work.
Production pins above do not change. Any earlier C# passing dense result for a merge, Sol-Attn, Sage or decode API
cell is invalid attribution; retain the raw result, mark Hartsy coverage unvalidated, and do not call it a K0 failure.
Create an additional H worktree and use its benchmark/GenCapture sources for all three arms, recording H and each
tool binary hash independently from the production identity. Do not mix old/new harness results within a campaign;
register fresh comparisons if necessary. Unsupported cells need a complete adapter/oracle before parity claims.

The first remote campaign is recorded in the [4090 evidence review](reports/pr01-4090-review.md). Audit its
arm/fixture/branch mapping and diagnose the first Krea CUDA divergence before launching further full campaigns.
Keep the worktrees and artifacts. The CUDA identity failure remains blocking; no new tolerance is accepted.
Follow up with narrowly scoped, separately registered reruns. Qwen request-level P1/B0 and observation-overhead
comparisons remain unexecuted; operation-level overhead does not establish request overhead.

Read repository AGENTS.md and its routed instructions. Use separate worktrees under `.claude/worktrees/`.
Fetch both PR branches, verify that these pins exist and record current remote heads. If production code changed
after these pins, report the difference and request an updated test pin; never silently test a moving branch.

Keep raw evidence/checkpoints/wheels/fixtures outside tracked source, e.g. `/mnt/evidence/attention-4090/<run-id>`.
The Windows local artifacts are not assumed to be present. Generate shared fixtures from committed tools and
recreate the Qwen prompt with the pristine-production freezer below. Preserve every failed run.

## 1. Preflight and isolation

Record GPU name/UUID/PCI ID/SM/VRAM, driver, OS/kernel, .NET SDK/runtime, Python executable and hash, CUDA,
cuBLAS/cuDNN, compiler versions, Vulkan loader/driver/device, features and queue timestamp period/valid bits.
Save `nvidia-smi -q`, `dotnet --info`, `vulkaninfo --summary`, and full dependency/build logs.
Verify Vulkan selects the physical RTX 4090 rather than lavapipe/software rendering.

Acquire the existing GPU lock, normally `/tmp/claude-1000/gpu4090.lock` via `flock`. Inspect active compute
tenants first. On a shared Swarm node, use `quiet_window.py wait/watch/verify` and retain samples bracketing
every measured interval. See [the shared-machine procedure](run_attention_table.sh). If Swarm is not installed,
document that fact and establish exclusive compute use by process/tenant inspection and continuous sampling.
An unreachable Swarm API on a node that runs Swarm is not proof of quiet.

Never stop Swarm, unload somebody else's model, kill other GPU clients, reset the GPU, or change clocks/power
to force admission. Wait or report resource blocking. Quiet Swarm counters alone do not prove enough VRAM.
Sample GPU memory, utilization, clocks, temperature and throttling throughout each campaign.
Run GPU suites, benchmarks and generations sequentially.

## 2. Build one harness against three production roots

Create pristine worktrees for B0/P1/P2 and set absolute shell variables `B0`, `P1`, `P2`, and `RAW`.
Run these commands from H. The harness is identical; only production roots and their kernel artifacts vary.

```bash
for arm in b0 p1 p2; do
  case "$arm" in b0) prod="$B0";; p1) prod="$P1";; p2) prod="$P2";; esac
  dotnet build benchmarks/HartsyInference.GpuBenchmarks -c Release -f net10.0 \
    -p:AttentionProductionRoot="$prod" -o "$RAW/bin/$arm-op"
  dotnet build tools/GenCapture -c Release -f net10.0 \
    -p:AttentionProductionRoot="$prod" -o "$RAW/bin/$arm-request"
done
```

Do not copy candidate DLLs or PTX into B0. Verify B0 contains no `AttentionEvidence` type, while P1/P2 do.
Hash managed libraries, all PTX/SPIR-V, native libraries, tool DLLs, copied `attention-protocol.json`, settings,
workload manifests and input files separately. Record dirty state and build command/exit code for every build.
Build the changed libraries for net8.0 and net10.0 too. Do not rebuild unrelated kernels with another frontend.

CUDA 13.0.88 NVRTC owns the split/monolithic/general-half Flash modules; NVCC owns causal-half and fused-decode
baselines. The fused SM75 floor uses NVRTC. Current shipped artifacts are ISA 9.0. If a rebuild is needed, follow
the domain build lists and record the emitted header, compiler and artifact hashes. The cached Linux compiler
used locally was `nvidia/cuda@sha256:7d2f6a8c2071d911524f95061a0db363e24d27aa51ec831fcccf9e76eb72bc92`.
Do not substitute NVRTC for an NVCC-owned baseline. SM75 remains hardware-unverified even if assembly succeeds.

## 3. Full regression suites

Run the CPU lane across every top-level test project, retaining each exit code and test names:

```bash
cpu_filter='Category!=SyntheticSmoke&Category!=Integration&Category!=GpuIntegration&Category!=GpuBenchmark&Category!=Slow&Network!=Real'
for project in tests/*/*.csproj; do
  name=$(basename "$project" .csproj)
  if dotnet test "$project" -c Release --filter "$cpu_filter" >"$RAW/$name-cpu.log" 2>&1; then
    printf '%s 0\n' "$name" >>"$RAW/cpu-exits.txt"
  else
    printf '%s %s\n' "$name" "$?" >>"$RAW/cpu-exits.txt"
  fi
done
for name in Core LLM Diffusion Engine; do
  if dotnet test "tests/HartsyInference.$name.Tests" -c Release \
    --filter 'Category=GpuIntegration' >"$RAW/$name-gpu.log" 2>&1; then
    printf '%s 0\n' "$name" >>"$RAW/gpu-exits.txt"
  else
    printf '%s %s\n' "$name" "$?" >>"$RAW/gpu-exits.txt"
  fi
done
```

Ensure all 45 `AttentionSemanticParityTests` cases actually execute. They cover CUDA materialized F32/F16/BF16,
cuDNN F16/BF16, Vulkan scalar, softcap-before-ALiBi, non-finite sinks, excluded rows, finite sentinels, nonzero
scores, batch/head/tail indexing, forced split F.5, native S.2/F.3 and direct/fused combine with Q8 sidecars.
Keep route assertions enabled. A host fallback or another implementation is not equivalent coverage.

Local Windows results: CPU 3,219 reported passes / two known B0 failures / one skip; GPU Core 19, LLM 101,
Diffusion 18, Engine 12 reported passes, with 31 resource skips across the GPU projects. Four reported passes
are actually guarded two/eight-GPU checks. Do not repeat these counts as your own measurements.
Known CPU failures: `MeshOperateRouteTests.OperationsListsEveryOperationWithItsParameters` and
`MiniMaxH3FlatLayoutResolutionTests.ImageResolverFindsTheVaeAndEncoderInTheModelsRootLayout`.
Reproduce unexpected failures on B0 before attributing them; the Windows path-case failure may disappear on Linux.
The standalone `tests/HartsyInference.Cuda.Tests` directory has no project; do not claim a separate suite ran.

For real-weight tests, inspect resource guards and configure their documented asset/reference variables.
Use `HARTSY_REQUIRE_REAL_WEIGHTS=1` where supported. A skip, zero executed cases or a guarded reported pass is
unvalidated. List every missing fixture/checkpoint and affected test, not merely the process exit status.

## 4. Qwen: first prove the frozen primary path on B0

Fetch `Qwen/Qwen3-4B-GGUF` revision `bc640142c66e1fdd12af0bd68f40445458f3869b`, file
`Qwen3-4B-Q4_K_M.gguf`, using the existing asset cache or a download pinned to that revision.
Required SHA-256:
`7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5`.

```bash
dotnet "$RAW/bin/b0-request/hartsy-gencapture.dll" attention-freeze-qwen \
  --model /absolute/path/Qwen3-4B-Q4_K_M.gguf --out "$RAW/qwen-candidate"
dotnet "$RAW/bin/b0-request/hartsy-gencapture.dll" attention-request \
  --workload "$RAW/qwen-candidate/workload.json" --backend cuda:0 --lane paged \
  --out "$RAW/qwen-b0-path.json"
dotnet "$RAW/bin/p1-request/hartsy-gencapture.dll" attention-request \
  --workload "$RAW/qwen-candidate/workload.json" --backend cuda:0 --lane paged \
  --observe "$RAW/qwen-p1-path-observations.json" --out "$RAW/qwen-p1-path.json"
```

Required recreated identities:

| Item | SHA-256 |
|---|---|
| UTF-8 prompt | `dcd8cc318e27658facf20ecbd3ef7b0a8efb2ef23254047674d7f1b545064255` |
| Embedded tokenizer metadata | `fbbe7052cd80244d79f87fac7fba241b53c6e13b2e309a7f8959cbd30e005eef` |
| Rendered token bytes | `77b1324c1ab3c9f22c35843ebac1bf067f652ed54b6c3ce068aab08507055470` |

There must be 8,192 rendered prompt tokens and exactly 128 naturally generated tokens, ending at Length.
An early EOS, placement failure or token/hash mismatch fails workload proof. Preserve it and report for review;
do not silently select another prompt, force output IDs, shorten context, or reduce the four-GiB request KV budget.
This prompt is still a candidate until B0 proves natural completion and the intended route.

The primary is warm complete Engine request latency, including tokenization, scheduler, prefill, decode,
result construction and request-owned cleanup. Model loading and setup precede timing. The tool reads exactly
two warmup requests from the copied protocol. Each measurement uses fresh request-owned KV and no retained prefix.
Freeze greedy sampling/seed 42, thinking/speculation/graph admission off, continuous batching on, one request,
no scores/forced IDs/image input/unloading. Confirm actual settings, not only requested settings.

P1 observations must prove scheduler admission, paged storage and actual executed attention. Relevant sites
include `scheduler.paged`, `scheduler.decode.batch`, `paged.keys`, `paged.values`; preserve nested fallback,
valid lengths, preparation and component/layer facts. Check overflow/unfinished writers and use the recorded
`measured_observation_start` to distinguish load/warmup evidence from the measured request.
Do not try `--observe` with pristine B0.

Repeat path proof for `--lane dense`, `--lane graph`, `--lane batched`, and on `--backend vulkan:0`.
Dense must use fixed/dense storage. Graph must actually replay a backend graph; admission alone is insufficient.
Batched must admit both requests to paged storage and execute a real multi-sequence batch. An unsupported lane
is a failure to validate that lane, not a pass. Do not rewrite the endpoint to hide it.

Record full text/hash, counts, stop reason, prefill/TTFT, decode ms/token and tokens/s, throughput/per-request
latency for batches, total request latency and peak VRAM. The current tool reports prefill/decode durations;
do not relabel prefill duration as directly observed TTFT. Add an untimed diagnostic using existing progress/timing
hooks for TTFT and memory, or report them unmeasured. Teacher forcing/scoring are separate diagnostics.

## 5. Krea: pin assets and quality before candidate inspection

Primary: normal Engine Krea2 Turbo request, 1024x1024, eight steps, CFG 1, seed 42, batch one, complete text
encoding + denoising + VAE output. Confirm resolved recipe `krea2/turbo`; another Krea variant is invalid.
The local run never acquired these assets, so no checkpoint/prompt/quality freeze may be invented retrospectively.

Locate the existing GPU-node Krea2 Turbo assets and existing generation/parity infrastructure. Before P2 runs,
record immutable model revisions and SHA-256 for DiT, encoder, tokenizer, VAE and every auxiliary file, exact
prompt UTF-8 bytes/hash, sampler/scheduler/shift/dtype settings, resolved recipe and quality-reference identity.
Reuse an existing reviewed quality gate and cite its source. If no applicable gate exists, run baseline-only
diagnostics and return a proposed freeze/gate for review before inspecting candidate error. Do not derive a
tolerance from candidate output. PR 01 observation equivalence requires exact deterministic output identity.

Use an external GenCapture manifest with `schema:1`, `family:"krea"`, `model_id`, absolute `model_path`,
`model_sha256`, `prompt_path`, `prompt_sha256`, and `aux` entries mapping the recipe's actual auxiliary keys
to `{"path":"...","sha256":"..."}`. Inspect `Krea2Recipe.cs`/`Krea2Variants.cs` for those keys; do not guess.
The Qwen freezer cannot generate a Krea manifest.

Run the same `attention-request --workload ... --backend cuda:0` interface for B0/P1/P2; repeat on Vulkan.
It saves the PNG after timing. Preserve RGB/PNG hashes, resolved settings, seed, complete latency, per-step
diagnostic latency, text-encoder/denoiser/VAE time, peak VRAM, raw/pixel reference metrics and executed GQA,
layout, numerical policy and kernels. A plausible image is not a correctness pass. Keep diagnostic tensor
captures separate from timed runs. PR 02's finite-sentinel behavior change needs affected-generation coverage.

## 6. Telemetry equivalence and operation performance

Compare B0, P1 disabled and P1 lightweight-enabled with identical fixtures, policies, outputs and routes on both
backends. P2 is not an observation-equivalence arm: its arithmetic deliberately changes.
Include memory-sensitive route boundaries and nested fallback; never alter production budgets to force identity.
Quantify enabled/disabled overhead. Save binaries, settings and all raw outputs separately.

For operation fixtures, use the committed generator and verify bytes before either engine runs:

```bash
python benchmarks/attention/evidence.py fixture --id strict-f32 --heads 4 --kv-heads 4 \
  --sq 256 --skv 516 --dim 64 --dtype f32 --precision f32 --out "$RAW/fixtures/strict-f32"
python benchmarks/attention/evidence.py verify-fixture "$RAW/fixtures/strict-f32/fixture.json"
dotnet "$RAW/bin/p2-op/HartsyInference.GpuBenchmarks.dll" attention-table \
  --fixture "$RAW/fixtures/strict-f32/fixture.json" --arm mat-f32-strict --boundary public \
  --trials 5 --expect-branch S.14 --cohort linux-4090 --out "$RAW/strict-f32-p2-public.json"
```

Repeat with B0/P1 tools, `--boundary resident`, and `HARTSYINFERENCE_BENCH_BACKEND=vulkan` where applicable
(Vulkan's expected branch is A.2 for this scalar case). F16 fixture/arm `mat-f16` expects S.15 on CUDA.
Freeze F16 fixtures with `--legacy-asserted-bounded` to match that arm. Flash arms require a `--route Flash`
fixture with matching Q/O and K/V types; use `--dtype f32 --kv-dtype f16` for `fa-f16kv` and split variants.
Discover routes with `--boundary observe`, then review and freeze `--expect-branch` before timing. An S-series
Dense branch under a Flash label is not split evidence. Do not modify fixtures in place to repair old campaigns.
Use `--observe FILE` for separate diagnostics. Use `telemetry_check.py` for supported operation equivalence,
and independently check generation identity; that helper does not certify all nested/request paths.

The local timing concern remains open: candidate/B0 smoke ratios ranged 0.880-1.317, with slower unchanged
Vulkan too. Preserve those prior results; do not sample repeatedly until a favorable result appears.
Before the new campaign, register exact cells/arms, output lifetime, fresh/prepared state, sessions and endpoints.
Include materialized F32/F16, query-tiled F32, monolithic F32/F16-KV, split/combine, native half and causal-half
routes, and the graph fused-combine consumer where affected. Prove executed routes; a benchmark missing a
required route or timing adapter must report that gap rather than label a substitute implementation as measured.

Freeze ten operation warmups. Resident trials: one stream/queue, one timestamp region around 32 serial calls,
divide by 32; no intervening synchronization. Public trials: one public operation including preparation and
one completion wait. Load/hash/validate outside timing and drain prior work outside each trial.
Validate empty and longer device-clock regions. CUDA events must be on the executing stream; Vulkan must use
the reported timestamp period and queue valid bits and include automatic flushes. Invalid device timing stays
unmeasured; wall time cannot replace it. Record host gaps rather than claiming summed active-kernel duration.

For each registered statistical comparison: three sessions x five pairs (15 inferential pairs), alternating
arm order, plus a predeclared fourth confirmation session x five pairs. Use geometric mean candidate/reference
ratios and the fixed 20,000-resample, seed-274019 stratified paired bootstrap. Improvement: two-sided 95% CI
entirely below 1.0. Non-inferiority: one-sided 95% upper bound <=1.03. Material regression: one-sided lower
bound >1.03. Confirmation's geometric mean must pass the same threshold and have no quality/route/memory violation.
Do not pool backends/boundaries/cohorts, average away failing cells, replace a failed confirmation, or extend samples.

`evidence.py statistics --pairs FILE --out FILE` accepts
`{"claim":"noninferiority","sessions":[[five pair objects],[five],[five]],"confirmation":[five]}`;
each pair has `candidate_ms`, `reference_ms`, `valid` and should link to its raw evidence. Mark valid only after
quality, route, cache, memory, numerical policy and cohort checks actually pass.

## 7. Kitchen foundation

Use an isolated venv/wheelhouse; never modify shared ComfyUI. Pin clean K0 source and exact Python/PyTorch,
CUDA extensions/compiler/build flags and wheel hashes. Linux 4090 is a new cohort; Windows Torch 2.9.1+cu130
fallback results are not its baseline. Build native K0 with a supported toolchain and `COMFY_CUDA_ARCHS=89`;
capture native extension and imported Python-source hashes. Never patch K0 and keep calling it K0.

```bash
python -s benchmarks/attention/evidence.py lock-environment --kitchen "$KITCHEN" \
  --wheelhouse "$WHEELHOUSE" --out "$RAW/k0-environment.json"
python -s benchmarks/attention/evidence.py matrix --kitchen "$KITCHEN" --cohort linux-4090 \
  --out "$RAW/k0-matrix.json"
python -s benchmarks/attention/evidence.py cell-fixture --matrix "$RAW/k0-matrix.json" \
  --cell CONCRETE_CELL_ID --out "$RAW/fixtures/CONCRETE_CELL_ID"
python -s benchmarks/attention/kitchen_operation.py --fixture "$RAW/fixtures/CONCRETE_CELL_ID/fixture.json" \
  --environment "$RAW/k0-environment.json" --api API_FROM_CELL --boundary public \
  --trials 5 --out "$RAW/k0-CELL-public.json"
```

Use that venv's Python for every Kitchen command. Enumerate source/API constraints, upstream tests and runtime
capability queries; Hartsy's current abilities do not define Kitchen support. Cover Sage formats/layouts/strides,
MHA/GQA/MQA and D64/128/256 constraints, unequal lengths and masks, actual rotations/representative-key
quantization/prequantization, BF16 decode/lengths/LSE, speculative tree/step merge, and general Sol routed plus
pooled omitted-block/shared-denominator semantics. Specialized VSA is not general Sol.

For every concrete cell record K0-valid/rejected/invalid/Not executed, actual native/fallback path, independent
oracle and tolerance, CUDA/Vulkan Hartsy outcome, quality-policy equivalence and timing eligibility. Source
presence or native availability is not execution proof. Use separate `--diagnostics` runs for Python/ATen/GPU
kernel traces; timing runs omit that flag. Preserve oracle mismatches, OOMs, exceptions and negative admission.
Required K0-invalid cells need explicit review; build/resource failures remain Not executed.

Important adapter gaps: C# rejects per-sequence lengths differing from scalar length; it must not certify
`[0,53,73]` as three full-length calls. The sampled C# timing oracle does not cover masks or all sparse/merge
features; unsupported adapters are gaps, not parity passes. Finite prequantized reuse-count timing remains
ineligible until its state accounting is implemented and reviewed. Do not use `--mandatory` with unreviewed
acceptance or silently mark it reviewed. Initial valid intersection comparisons are required; implementing
missing Kitchen features belongs to the later program.

## 8. Return this report for copy/paste

Return a concise self-contained Markdown report, plus an artifact directory/index with SHA-256 for every raw
file. Do not require the coordinating agent to infer pass/fail from a zip. Use this structure:

1. **Verdict:** READY FOR REVIEW / NOT READY / BLOCKED. State PR 01 and PR 02 readiness separately. No merge.
2. **Pins/environment:** B0/P1/P2/K0, harness hashes, all managed/kernel/native hashes, GPU/driver/toolchain,
   Vulkan device/clock facts, exact build commands/exits and clean/dirty state.
3. **Regression table:** each project/backend, passed/failed/skipped/guarded-unvalidated, failing test names,
   B0 reproduction and raw log hashes. Confirm 45 semantic cases executed and list routes.
4. **Workload freeze:** checkpoint/tokenizer/prompt hashes, counts/stop reason, Krea full asset/recipe/settings
   and quality gate. List baseline freeze failures and gate proposals needing review.
5. **Executed-path table:** family/lane/backend, scheduler, dense/paged, actual batching/replay, implementations,
   nested fallback and preparation, observation overflow and measured interval boundaries.
6. **Correctness:** B0/P1 disabled/P1 enabled identities; P2 vs B0 metrics against the predeclared applicable
   gate; image/text/raw reference artifacts; non-finite/finite sentinel findings. Separate numerical accuracy
   from quantized algorithm matching. Give measured numbers and gates, not visual judgments.
7. **Performance:** one row per workload/cell/backend/boundary with absolute reference/candidate latency,
   ratio, 95% bounds, confirmation ratio, claim pass/fail/inconclusive, TTFT/decode/batch/peak VRAM, route,
   clocks/quiet evidence and memory violations. Keep resident/public/preparation/reuse separate.
8. **Kitchen matrix summary:** all four outcome counts, native/fallback proof, oracle errors, rejected cases,
   current Hartsy gaps, valid matched-quality comparisons and explicit exception proposals.
9. **Blockers:** precise missing assets/adapters/hardware/evidence, what remains necessary for merge and why.
   Include every mandatory unexecuted case. Blackwell and SM75 final hardware gates remain future/unverified;
   do not claim this 4090 run completed the whole refactor or created Bfix.

The coordinating agent will review the returned evidence, resolve remaining PR comments, merge only when ready,
and then start the next PR under the user's instruction. Do not change the frozen endpoints or waive missing
real-generation/performance evidence to make that sequence advance.
