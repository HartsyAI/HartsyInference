# PR 03 organization and support-contract evidence

PR 03 is a stacked draft on PR 02 at `69883e4be8813b1ee35313a4ebdcfbbde1b4ecd6`. Its production extraction
commit is `5b0a1c99c23cd79adf0e6fb45d5032a0fc4b0037`; subsequent changes are tests/documentation.
The GPU agent's independent PR 01/02 pins remain unchanged. This does not establish Bfix or close those exit gates.

## Organization and compatibility

The pure `git mv` commit `560c25011` has no content changes. The extraction separates capability declarations,
the row contract, plan description, diagnostic logging and planner execution. Related capability flags stay with
the capability declaration under the user's instruction to organize by ownership rather than universally
splitting every public type into its own file. Public type names, namespaces, assembly, enum values, method
bodies, row preference, predicates and lowering/fallback behavior remain unchanged.

Normalized compiled snapshots compare member signatures, constants, resolved IL operands, local types,
exception regions and method flags across the complete Core assembly:

| Target | Compared declaration/body entries | Result |
|---|---:|---|
| net8.0 | 9,536 | Identical |
| net10.0 | 9,611 | Identical |

Physical DLL hashes may differ because metadata ordering/build identity changes; this is not a claim of
byte-identical managed assemblies. Both targets build without warnings. net8.0 remains temporary SwarmUI
compatibility until its move to net10.0; this PR adds no framework-specific abstraction or target changes.

All 396 staged PTX/SPIR-V runtime artifacts match the parent byte for byte. Ten generated PTX files had checkout
LF/CRLF differences, harmonized only for the runtime comparison and then restored to normal checkout form.
There is no tracked kernel/PTX/SPIR-V change, managed backend relocation, or backend implementation change.

## Regression evidence

Six planner tests cover diagnostic planning without launch, refused-format exclusion before admission,
recoverable priority, exception propagation without retry, unsupported calls preserving output, and allocation
growth. The allocation check compares 1,000 versus 10,000 calls with a fixed one-KiB instrumentation allowance;
it does not permit allocation proportional to calls. The Core CPU suite passes all 178 cases after that review change.

The full CPU lane across thirteen projects reports 3,225 passes, two known B0 failures and one skip. The failures
remain `MeshOperateRouteTests.OperationsListsEveryOperationWithItsParameters` and
`MiniMaxH3FlatLayoutResolutionTests.ImageResolverFindsTheVaeAndEncoderInTheModelsRootLayout`.
A reported T5 parity pass has a missing-reference guard and remains unvalidated.

GPU categories ran sequentially on RTX 3060 / SM86 / driver 610.88:

| Project | Reported passes | Failures | Resource skips |
|---|---:|---:|---:|
| Core | 19 | 0 | 0 |
| LLM | 101 | 0 | 19 |
| Diffusion | 18 | 0 | 2 |
| Engine | 12 | 0 | 10 |

Four reported passes are guarded two/eight-GPU NCCL/Kimi cases and remain unvalidated. Resource skips do not
establish real-generation coverage. CPU/GPU tests target net10.0; the net8.0 evidence here is library build and
compiled identity, not a separate net8.0 generation campaign.

## Route/output and timing smoke

The same operation harness was built against the parent and this production extraction. Fixtures are shared
serialized F32/F16 tensors, B1/H4/Sq256/Skv516/D64, with strict F32 and materialized F16 arms. Complete output
hashes, executed S.14/S.15/A.2 branches, numerical settings, PTX/SPIR-V hashes and separate diagnostic observation
arrays are identical. This is scoped identity evidence, not every possible route or memory-pressure case.

Two alternating pairs per cell, five trials per process, ten warmups. Resident trials contain 32 serial calls;
public trials contain one call and completion wait. The 3060 lock and bracketing Swarm quiet samples were checked.

| Backend/storage | Boundary | Parent mean ms | Candidate mean ms | Candidate/parent |
|---|---|---:|---:|---:|
| CUDA F32 strict | Public | 0.77178 | 0.88039 | 1.141 |
| CUDA F32 strict | Resident | 0.74344 | 0.76771 | 1.033 |
| CUDA F16 | Public | 0.56691 | 0.55143 | 0.973 |
| CUDA F16 | Resident | 0.46147 | 0.44917 | 0.973 |
| Vulkan F32 | Public | 3.30654 | 3.31415 | 1.002 |
| Vulkan F32 | Resident | 2.93093 | 2.89885 | 0.989 |

The Windows desktop timings are mixed, including a slower F32 public cell. These are descriptive smoke results,
not statistical improvement/non-inferiority acceptance. No favorable resampling or speed claim follows from
compiled identity. Preserve slower results and the program's registered performance gates. Real-generation,
hardware and parent exit-review evidence remain separate requirements.

## Review boundaries

The [support contract](../../../docs/ATTENTION_SUPPORT.md) proposes mandatory common cases, optional admission,
precision-specific expectations and unsupported-before-write obligations. It explicitly records current gaps.
A read-only CPU dense exclusion probe returned NaN instead of zero; that existing M03 gap is recorded on PR 02
for semantic follow-up, without modifying its active hardware-test pin. The mechanical extraction does not make
that result conformant, enforce shared validation, remove host fallback or implement a calibrated planner.

Raw logs, normalized snapshots, artifact hash manifests and smoke JSON remain outside tracked source under
`artifacts/attention-pr03` in this PR's worktree, indexed by `evidence-index.json`. Keep the draft until the
support matrix, parent evidence and remaining readiness concerns have been reviewed.
