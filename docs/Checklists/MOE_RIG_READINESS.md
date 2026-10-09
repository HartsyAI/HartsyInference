# MoE runtime: GPU rig readiness

What must hold before GPU time is spent, what is already verified on CPU, and the order to run things on the rig. Status
lives here until a rig run replaces it with evidence.

## Verified on CPU (this machine, no GPU)

CPU-lane results per test project, with the GPU and real-weight tests excluded. The first baseline ran before the GPU labels in
#307 and #309; the table shows the results after them.

| Project | Result |
|---|---|
| Sixteen others (API, Audio, Core, Cpu, LLM, ModelAssets, Vision, Voice, and eight more) | all passing |
| Diffusion | 1869 passing after the GPU methods were labelled (#307) |
| Video | 114 passing after the GPU methods were labelled (#307) |
| Cuda | 157 CPU-only tests passing, none failing; GPU methods labelled (#307) |
| Vulkan | 12 passing, exit 0 after the device tests were labelled (#309); the run hung before |

Subsystems with CPU tests in place:

- Sparse topology contracts and fingerprint (#299).
- Scalar reference path with parity against `MoeFeedForward` on softmax and grouped sigmoid routing (#300).
- Expert identity with banks; cache keyed by (bank, layer) (#303).
- Residency queries and no-upload acquisition; allocation-free per call (#304).
- Expert pack write, read, verify and GGUF packing, with refusals and corruption checks (#305, #316).
- Placement planner: residency decides placement, every routed pair is counted once, planning changes no state (#306).
- Heterogeneous executor, CPU side (#313).
- CUDA F32 expert kernel `expert_f32` (sm_75 PTX) and the lease-validated device runner (#335).
- CPU quantized expert kernels, Q8_0 and Q4_K, with scalar parity (#341); CPU worker pool (#345).
- Expert cache stress and failure injection, including upload and await faults (#343).
- Opt-in CPU expert runtime, with the reload fix (#340).

## This machine (checked)

- **GPU and CUDA.** RTX 2060 SUPER, Turing (sm_75), 8 GB. The CUDA driver API works: `CudaContext.IsAvailable()` returns
  true and `RigPreflightTests` passes.
- **nvidia-smi.** Still prints "Driver/library version mismatch" (NVML library 595.99). That is NVML only, not CUDA. It must be
  resolved, by a reboot or a matching reinstall, before the rig run so the stop rule below stays meaningful. That is a
  system change, so it was not made from here.
- **Expert kernels on this card.** `CudaExpertKernelTests` passes 5 of 5 (`expert_f32` is built for sm_75).
  `CudaExpertDeviceRunnerTests` fails 3 of 3 with PTX JIT error 218. The cause is that `CudaBackend` construction loads
  about 80 modules eagerly, some built for sm_80 or sm_120a, and the card cannot load those. Making the load lazy is a
  larger change and is deferred. Rent sm_80 or newer: the baseline PTX is sm_80.
- **PTX.** Every shipped PTX file is ISA 9.0 or older. 28 files are ISA 7.0, which the driver loads fine. No file is newer
  than the 9.0 ceiling in `src/HartsyInference.Cuda/Kernels/build_common.sh`.
- **Host memory.** 39 GB total. Check `free -g` before any large load (see the no-heavy-GPU-runs note).
- **VRAM.** 8 GB limits what this card can run. Large-MoE and real-checkpoint validation needs the rented GPU.

## Rig run order

1. Preflight. `RigPreflightTests` fails unless CUDA is usable, so a GPU suite cannot pass by skipping every test. It logs the
   device name and compute capability for the run record:
   `dotnet test tests/HartsyInference.Cuda.Tests --filter "FullyQualifiedName~HartsyInference.Cuda.Tests.RigPreflightTests."`
   `nvidia-smi` must also show no mismatch message.
2. Real-weight assets, set before any GPU suite. With `HARTSY_REQUIRE_REAL_WEIGHTS=1` a missing asset fails the test
   instead of logging `SKIPPED`, so a run with missing assets is a failed run, not a green one:
   ```
   export HARTSY_REQUIRE_REAL_WEIGHTS=1
   export HARTSY_DSV41_FLASH_DIR=<dir holding model-00003-of-00048.safetensors>
   export HARTSY_DSV41_SHARD3_FIXTURES=<dir holding manifest.tsv>   # default: ~/dsv41-ref/shard3_fixtures
   ```
   Confirm both paths exist before starting. `CudaExpertM1FixtureTests` is the only suite that reads them.
3. Expert-cache and expert suites first, one class per invocation, one at a time. xUnit does not run classes in a
   caller-chosen order, so the category filter cannot put these first. The trailing dot makes each filter match one class
   exactly:
   ```
   classes="CudaExpertCacheTests CudaExpertM1FixtureTests CudaMoePrimitiveTests"
   classes="$classes CudaMoeTests CudaQuantWorkspaceTests CudaStreamingWeightCacheTests"
   classes="$classes CudaExpertKernelTests CudaExpertDeviceRunnerTests"
   set -euo pipefail
   for c in $classes; do
     dotnet test tests/HartsyInference.Cuda.Tests --filter "FullyQualifiedName~HartsyInference.Cuda.Tests.$c."
   done
   ```
   `set -euo pipefail` stops the run at the first failing class, so a later pass cannot hide an earlier failure. Every class
   in the list matches at least one test (checked with `--list-tests`). `CudaExpertKernelTests` (#335) checks `expert_f32`
   against the reference for every program variant. `CudaExpertDeviceRunnerTests` (#335) runs the lease-validated device
   runner through `CudaBackend`; it needs sm_80 or newer.
   `CudaMoeTests` and `CudaStreamingWeightCacheTests` carry no `Category` trait, so the category filter does not select
   them; this explicit loop is the only way they run on the rig.
4. The rest of the GPU category, excluding the classes already run in step 3. Run only from a checkout that includes #307,
   which applies the GPU labels this filter depends on. Build the filter in a variable so it has no embedded whitespace.
   Record failures by test name:
   ```
   filter="Category=GpuIntegration"
   for c in $classes; do filter="$filter&FullyQualifiedName!~HartsyInference.Cuda.Tests.$c."; done
   dotnet test tests/HartsyInference.Cuda.Tests --filter "$filter"
   ```
5. **Deferred: refactor A/B for the cache changes.** Do not run `tests/regression-ab.sh` as evidence for #304 yet. No
   production path constructs `CudaExpertCache` (only the class and its tests do), and no core regression case drives it,
   so both arms would bypass the changed cache and could report identical output vacuously. The requirement that
   generations stay identical stays open. Evidence for the cache refactor on the rig is step 3. Reopen this step once an
   executor or a dedicated harness acquires and evicts experts through the cache; then run
   `tests/regression-ab.sh --fresh --expect identical --backend cuda` on that case.
6. Vulkan MoE primitives, after the CUDA lane has finished, with no other GPU run in progress:
   `dotnet test tests/HartsyInference.Vulkan.Tests --filter "FullyQualifiedName~HartsyInference.Vulkan.Tests.VulkanMoePrimitiveTests."`
   (the trailing dot matches this class only).
   This suite needs a local Vulkan device and no fixtures.

## Not yet possible, and what is missing

- **Packing a real checkpoint.** `hartsy moe pack|verify` exists (#316) and is tested on synthetic and GGUF fixtures. No
  real MoE checkpoint has been packed or verified yet.
- **Production use of the executor.** The planner (#306), the heterogeneous executor (#313), the opt-in CPU runtime (#340)
  and the CUDA F32 runner (#335) exist. No production model routes experts through them by default. The GPU grouped kernels
  (M6) are not built, and the packed CPU kernels (#341) are not wired into the executor.
- **Measurements the plan asks for.** Instrumented now: per-op timing (`OpProfile`) and expert-cache counters (hits, misses,
  uploads, bytes). Not yet instrumented: time to first token, tokens per forward pass, host-to-device and device-to-host
  bandwidth, peer bandwidth, CPU and GPU utilization, KV bytes.

## Known gaps before a run is evidence

- The pack fingerprint is provisional: it hashes GGUF geometry, not the runtime topology.
- The opt-in CPU path (`MoeFeedForward.UseHostExpertRuntime`) copies expert weights to F32 and allocates per call.
- Q2_0 has an encoder and a pack codec, but no kernel and no quality validation.
- Telemetry and residency policies (#342) and the speculation selector (#339) are not wired into the expert cache or the
  decode loop.

## Stop rules

- Stop if a GPU failure does not reproduce on a second run. Nondeterminism is a finding, not a pass.
- Stop if the driver mismatch returns. Do not measure through it.
- Stop if any real-weight suite reports `SKIPPED` while `HARTSY_REQUIRE_REAL_WEIGHTS=1` is set, or if its assets are missing.
