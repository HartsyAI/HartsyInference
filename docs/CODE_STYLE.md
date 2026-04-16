# SharpInference — Code Style & Guidelines

> **All agents must follow these rules.** This document is the single source of truth for coding conventions in SharpInference. When in doubt, follow what's written here.

---

## Type Declarations

### Never use `var` — always use explicit types

```csharp
// WRONG
var buffer = new NativeBuffer(1024);
var shape = new TensorShape(1, 3, 512, 512);
var count = shape.ElementCount;

// RIGHT
NativeBuffer buffer = new NativeBuffer(1024);
TensorShape shape = new TensorShape(1, 3, 512, 512);
long count = shape.ElementCount;
```

No exceptions. Even when the type is obvious from the right-hand side, spell it out. This makes code scannable without hovering or IDE support.

---

## Access Modifiers

### Prefer `public` — be explicit about visibility

- Default to `public` for types, methods, and properties unless there's a clear reason to restrict
- Use `internal` for implementation details that shouldn't leak across package boundaries
- Use `private` for truly internal state (backing fields, helper methods)
- Always write the access modifier explicitly — never rely on C# defaults

```csharp
// WRONG
class TensorPool { }           // implicitly internal
void ProcessData() { }         // implicitly private

// RIGHT
public class TensorPool { }
private void ProcessData() { }
```

---

## Comments

### No redundant comments — code should be self-explanatory

Do not add comments that repeat what the code already says. Comments should only explain **why**, never **what**.

```csharp
// WRONG — redundant
// Create a new tensor
Tensor tensor = new Tensor(shape, DType.F32);

// Dispose the buffer
buffer.Dispose();

// Loop through all elements
for (int i = 0; i < count; i++) { }

// RIGHT — explains why
// Accumulate in FP32 to prevent precision loss in FP16 GroupNorm
float sum = 0f;

// SD1.5 uses 0.18215, SDXL uses 0.13025
float vaeScale = config.VaeScaleFactor;
```

### XML doc comments

- Required on all `public` and `internal` APIs (classes, methods, properties, interfaces)
- Not required on `private` members unless the logic is non-obvious
- Keep them concise — one sentence is usually enough
- Don't restate the method name or type name in the summary
- Never state the obvious — the summary should tell the reader something they **can't** already see from the signature
- **Single-line format** — `<summary>` tags go on the same line as the text, never multi-line

```csharp
// WRONG — multi-line tags
/// <summary>
/// Maps named tensors from the model file to internal weight parameters.
/// </summary>
public void LoadWeights() { }

// WRONG — restates the name
/// <summary>Loads the weights.</summary>
public void LoadWeights() { }

// WRONG — redundant (obvious from the signature)
/// <summary>Gets the element count.</summary>
public long ElementCount { get; }

// RIGHT — single-line tags, adds real information
/// <summary>Maps named tensors from the model file to internal weight parameters.</summary>
public void LoadWeights() { }

/// <summary>Total number of elements across all dimensions.</summary>
public long ElementCount { get; }

/// <summary>Frees the underlying unmanaged memory via atomic pointer exchange.</summary>
public void Dispose() { }

/// <summary>Launches the PTX kernel with args marshaled on the stack via stackalloc.</summary>
public void LaunchKernel(nint function, KernelArgs args) { }
```

For `<param>` and `<returns>` tags, use the same single-line format and only include them when they add non-obvious information:

```csharp
/// <summary>Dequantizes a Q4_0 block into 32 FP32 values.</summary>
/// <param name="block">Pointer to the 18-byte Q4_0 block (2-byte scale + 16 nibble pairs).</param>
/// <param name="output">Must have space for 32 floats.</param>
public void DequantQ4_0(nint block, Span<float> output) { }
```

---

## Error Handling

### Always catch exceptions and log errors

Never let exceptions propagate silently. Every `catch` block must log the error using `Logs.Error()`. Never swallow exceptions with an empty catch.

```csharp
// WRONG — swallowed exception
try { LoadModel(path); }
catch { }

// WRONG — no logging
try { LoadModel(path); }
catch (Exception ex) { throw; }

// RIGHT
try
{
    LoadModel(path);
}
catch (Exception ex)
{
    Logs.Error($"Failed to load model from {path}", ex);
    throw;
}
```

### Fail fast at boundaries, handle gracefully at top level

- **Internal code:** Validate inputs at method entry, throw immediately if wrong
- **Pipeline/API level:** Catch, log, and return meaningful errors to the caller
- **Never silently continue** with bad state — corrupted tensors are worse than crashes

```csharp
// Validate at method entry
public void MatMul(Tensor output, Tensor a, Tensor b)
{
    if (a.Shape.Rank != 2 || b.Shape.Rank != 2)
        throw new SharpInferenceException($"MatMul requires 2D tensors, got {a.Shape} and {b.Shape}");

    if (a.Shape[1] != b.Shape[0])
        throw new SharpInferenceException($"MatMul inner dimensions must match: {a.Shape[1]} != {b.Shape[0]}");
}
```

---

## Naming Conventions

| Element | Convention | Example |
|---|---|---|
| Namespace | PascalCase matching folder | `SharpInference.Core.Tensors` |
| Public class/struct/record | PascalCase | `TensorShape`, `CpuBackend` |
| Interface | `I` prefix + PascalCase | `IBackend`, `IScheduler` |
| Public method | PascalCase | `LoadWeights()`, `Forward()` |
| Public property | PascalCase | `ElementCount`, `Device` |
| Private field | `_camelCase` with underscore | `_disposed`, `_ownedBuffer` |
| Local variable | camelCase | `byteSize`, `blockCount` |
| Constant | PascalCase | `MaxRank`, `DefaultAlignment` |
| Enum member | PascalCase | `DType.F32`, `DeviceType.Cuda` |
| Type parameter | `T` prefix | `AsSpan<T>()` |
| Async method | `Async` suffix | `GenerateAsync()`, `TranscribeAsync()` |

---

## Code Structure

### File-scoped namespaces — always

```csharp
// WRONG
namespace SharpInference.Core.Tensors
{
    public class Tensor { }
}

// RIGHT
namespace SharpInference.Core.Tensors;

public class Tensor { }
```

### One type per file

Each public class, struct, record, enum, or interface gets its own file. Small private helper types can live in the same file as the type that uses them.

### `sealed` by default

Classes should be `sealed` unless they are explicitly designed for inheritance. This helps the JIT optimize method dispatch.

```csharp
public sealed class NativeBuffer : IDisposable { }
```

### `readonly` aggressively

- Mark fields `readonly` whenever possible
- Use `readonly struct` for value types that don't mutate
- Use `readonly ref struct` for view types like `TensorView`

### File ordering

Within a class, organize members in this order:
1. Constants and static fields
2. Instance fields
3. Constructors
4. Public properties
5. Public methods
6. Internal/private methods
7. IDisposable implementation
8. Nested types

---

## Performance Rules

### Zero allocations on hot paths

The inference hot path (denoising loop, attention, matmul) must not allocate managed memory. This means:

- No `new` for reference types in inner loops
- No boxing (avoid casting value types to `object` or interfaces)
- No LINQ in compute kernels
- No `string` formatting or interpolation in inner loops
- No `Task` creation inside per-step code — use `ValueTask` if needed

### Use `Span<T>` over arrays

```csharp
// WRONG — allocates an array
float[] data = new float[count];

// RIGHT — stack or unmanaged
Span<float> data = stackalloc float[count];  // small, known size
Span<float> data = buffer.AsSpan<float>();    // from NativeBuffer
```

### Inline small hot-path methods

```csharp
[MethodImpl(MethodImplOptions.AggressiveInlining)]
public static int ElementSize(this DType dtype) => ...
```

### Performance attributes (following dotLLM)

| Attribute | When to Use |
|-----------|-------------|
| `[MethodImpl(AggressiveInlining)]` | Small hot-path methods (< ~20 IL bytes), tensor accessors, SIMD helpers |
| `[SkipLocalsInit]` | Methods with large `stackalloc` or performance-critical paths where zero-init is unnecessary |
| `[SuppressGCTransition]` | Short CUDA P/Invoke calls (< 1µs) to avoid GC cooperation overhead |

### Prefer `stackalloc` for small temporary buffers

Use `stackalloc` for buffers under ~1KB that are used within a single method scope.

### Use `readonly record struct` for zero-alloc value types

Small value types that are passed around frequently should be `readonly record struct` to ensure stack allocation and value semantics:

```csharp
public readonly record struct TensorRef(nint DataPointer, TensorShape Shape, DType DType, DeviceKind Device);
public readonly record struct GenerationProgress(int Step, int TotalSteps, double ElapsedMs);
```

### Use `Environment.FailFast` for unrecoverable compute errors

In compute worker threads, unrecoverable errors should crash the process immediately rather than producing silent wrong results:

```csharp
catch (Exception ex)
{
    Environment.FailFast($"Compute worker crashed: {ex.Message}", ex);
}
```

---

## Patterns

### IDisposable for unmanaged resources

Any class that holds unmanaged memory (NativeBuffer, MmapHandle, CUDA allocations) must:
1. Implement `IDisposable`
2. Use `Interlocked.Exchange` on the pointer for thread-safe disposal (following dotLLM's pattern)
3. Include a finalizer as safety net for forgotten `Dispose()` calls
4. Be `sealed` (avoids the need for the full Dispose pattern with virtual methods)

```csharp
public sealed class NativeBuffer : IDisposable
{
    private nint _pointer;

    public void DoWork()
    {
        if (_pointer == 0)
            throw new ObjectDisposedException(nameof(NativeBuffer));
        // ...
    }

    public void Dispose()
    {
        nint ptr = Interlocked.Exchange(ref _pointer, 0);
        if (ptr != 0)
            NativeMemory.AlignedFree((void*)ptr);
        GC.SuppressFinalize(this);
    }

    ~NativeBuffer()
    {
        nint ptr = Interlocked.Exchange(ref _pointer, 0);
        if (ptr != 0)
            NativeMemory.AlignedFree((void*)ptr);
    }
}
```

### Constructor validation

Validate all constructor parameters. Throw `ArgumentException` or `ArgumentNullException` for bad inputs.

```csharp
public Tensor(TensorShape shape, DType dtype)
{
    if (shape.ElementCount <= 0)
        throw new ArgumentException("Tensor shape must have positive element count.", nameof(shape));
}
```

### Async patterns

- Return `Task` or `ValueTask` for async methods
- Return `IAsyncEnumerable<T>` for streaming results
- Always accept `CancellationToken` as the last parameter
- Always pass cancellation tokens through to inner calls

---

## Formatting

### Braces

- Allman style (opening brace on new line) for types and methods
- Single-line expression bodies are fine for simple properties and methods

```csharp
// Multi-line: Allman braces
public void Process()
{
    // ...
}

// Single-line: expression body is fine
public long ElementCount => Shape.ElementCount;
```

### Line length

- Soft limit: 120 characters
- Hard limit: 150 characters
- Break long method signatures after each parameter

### Blank lines

- One blank line between methods
- One blank line between logical sections within a method
- No multiple consecutive blank lines
- No blank line after opening brace or before closing brace

### `using` directives

- Sort `System` namespaces first
- No blank line between using groups
- Remove unused usings

---

## What NOT to Do

- **Don't use `#region`** — ever
- **Don't use `dynamic`** — ever
- **Don't use `async void`** — except for event handlers (which we don't have)
- **Don't use `Thread.Sleep`** — use `Task.Delay` if you must wait
- **Don't use `GC.Collect()`** — if you think you need this, the real problem is elsewhere
- **Don't catch `Exception` at a low level** — only at pipeline/API boundaries
- **Don't use reflection on hot paths** — it's slow and allocates
- **Don't add NuGet packages without discussion** — every dependency is a liability
