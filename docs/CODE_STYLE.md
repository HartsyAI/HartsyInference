# Code style — mandatory

## Declarations and layout

- Explicit types; never var or dynamic. File-scoped namespaces; one public type per file.
- Explicit access modifiers. Prefer public APIs; internal for package implementation details, private for state/helpers.
- Classes sealed unless designed for inheritance. Mark immutable fields readonly.
- Use primary constructors for assignment-only initialization; ordinary constructors for validation,
  overload logic, nontrivial setup, or varying base calls. Validate constructor arguments.
- Value records are readonly record struct; configuration/options are record classes with init properties
  and required mandatory fields. Never mutable record struct.
- Positional records: parameters on one line, at most two packed lines within the length limits.
- Naming: PascalCase types/members/constants, I-prefixed interfaces, camelCase locals/parameters,
  _camelCase private fields, T-prefixed type parameters, Async suffix on async methods.
- Member order: constants/statics, fields, constructors, properties, public methods, helpers,
  IDisposable implementation, nested types.
- Allman braces; expression bodies for simple members. Soft line limit 120, hard limit 150.
  Wrap long method signatures at parameters; pack boolean conditions instead of one per line.
- One blank line between methods/logical sections; no repeated blank lines or padding inside braces.
  System usings first, no blank lines between groups, remove unused usings.
- Regions only for useful groups in large files, never a single member.

## Comments and errors

- Comments explain why, not what. XML documentation on public and internal APIs, concise single-line
  summary tags. Add param/returns tags only for non-obvious contracts.
- Validate shape, dtype, sizes, and arguments at boundaries; fail fast with meaningful exceptions.
  Never continue with corrupted state.
- Catch ordinary failures at pipeline/API boundaries, log with Logs.Error, and return/throw meaningful errors.
  No empty catches or low-level blanket catch(Exception). Expected cancellation may use debug logging.
- Unrecoverable compute-worker errors use Environment.FailFast; process recovery needs external supervision.

## Memory and performance

- Tensor storage: NativeMemory.AlignedAlloc or mmap. No managed tensor arrays on inference hot paths.
- No managed allocations, boxing, LINQ, interpolated strings, reflection, or per-step Task creation in hot loops.
- Use Span<T>; stackalloc for small scratch buffers (under approximately 1 KiB), unmanaged storage for larger ones.
  ValueTask may avoid async allocation where appropriate.
- Creator owns/disposes Tensor; borrowed views must not outlive backing memory. Make ownership explicit
  when a helper may return either a borrowed tensor or a converted copy.
- Unmanaged holders implement IDisposable, atomic idempotent release (Interlocked.Exchange), and a finalizer
  safety net where the holder directly owns unmanaged resources. Prefer sealed holders.
- Inline small hot helpers with AggressiveInlining; SkipLocalsInit only when reads are initialized safely.
- No GC.Collect or Thread.Sleep. No async void except event handlers.
- Reuse IBackend ops and shared utilities; check ownership/dtype behavior before merging similar helpers.
  [Architecture rules](Agents/AGENTS.md) list the shared utility entry points.

## Interop

- LibraryImport source generation, not DllImport. Native status returns are int; check every fallible
  call with ThrowOnError. Do not apply this return-type rule to APIs whose ABI returns another type.
- CUDA library identifier is cuda, resolved by CudaLibraryResolver; Vulkan uses VulkanLibraryResolver.
  Register through NativeLibrary.SetDllImportResolver.
- SuppressGCTransition only on short nonblocking calls; never a blocking native operation.
- Load PTX/SPIR-V from disk, not embedded resources. CUDA baseline sm_80; target-specific kernels may require more.
- Kernel handles are nint fields, not Dictionary<string,nint>.
- Marshal launch args with stackalloc void*[] pointing to local variables with stable addresses, never field refs.
- CUDA kernels and optional vendor libraries are backend implementation details, not model dependencies.

## Async and package contracts

- Return Task/ValueTask for asynchronous methods and IAsyncEnumerable<T> for service streams.
  CancellationToken is the last parameter and is passed through inner calls.
- Synchronous diffusion pipelines use Action<GenerationProgress>? callbacks; transport belongs in API.
- JSON uses source-generated contexts. No new NuGet dependency without discussion.
- Preserve package boundaries and published public signatures; see [BUILD_FEATURE.md](Agents/BUILD_FEATURE.md).

## Testing

A test earns its place when failure would be silent: kernel math, quant/codec round-trips, lifetime,
concurrency, padding/tiling, format/key mapping, and cross-device/backend equivalence. Do not add redundant
model-runs-at-all tests. Real-weight verification evidence still belongs in the parity/status ledger.

| Lane | Trait | Requirement |
|---|---|---|
| Unit | none | Fast, deterministic, runnable without GPU/checkpoints/network |
| SyntheticSmoke | Category=SyntheticSmoke | Unverified-model synthetic forward; opt in explicitly |
| Integration | Category=Integration | Real weights/reference fixtures; guard absent resources |
| GpuIntegration | Category=GpuIntegration | Actual GPU and compiled kernels; PTX existence alone is insufficient |
| Slow | Category=Slow | Long workloads, opt in |
| Network | Network=Real | External network, opt in |

Traits classify tests; they do not automatically filter dotnet test. Run the CPU lane explicitly:

    dotnet test --filter "Category!=SyntheticSmoke&Category!=Integration&Category!=GpuIntegration&Category!=Slow&Network!=Real"

There is no dedicated CPU/GPU CI workflow; publish-nuget.yml is the current workflow. TestTierLintTests
checks unguarded GPU/backend and gitignored fixture access when executed; it is not a build-time guarantee.
Use a recognized resource guard or a justified tier-lint: guarded annotation.

Parity tests live in tests/<Project>/Parity/ and end in *ParityTests, selected with
FullyQualifiedName~Parity. Resource-dependent tests remain resource-dependent after numerical verification;
only self-contained CPU tests can graduate to Unit. Never remove a resource trait just because parity passed.
Missing checkpoints may skip normal integration runs; mandatory campaigns use HARTSY_REQUIRE_REAL_WEIGHTS=1
where supported so missing weights fail rather than masquerade as validation.

Most tests/python-reference binary dumps are gitignored. Commit small deterministic Unit fixtures under
an explicit ignore exception; otherwise guard them and classify as Integration. Tests target net10.0;
libraries target net8.0 and net10.0. Run checks appropriate to the change, preserving existing tolerances.
