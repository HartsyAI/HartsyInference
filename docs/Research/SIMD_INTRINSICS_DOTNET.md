# SIMD Intrinsics in .NET -- Research Notes


---

## Summary

.NET 10 provides a mature, layered SIMD programming model through `System.Runtime.Intrinsics`. The three tiers are: (1) hardware-specific intrinsics (`Avx2`, `Avx512F`, `AdvSimd`), (2) cross-platform fixed-width vector types (`Vector128<T>`, `Vector256<T>`, `Vector512<T>`), and (3) high-level tensor operations (`TensorPrimitives`). The JIT compiler treats `IsSupported` checks as compile-time constants, eliminating dead branches so that a single method can contain paths for AVX-512, AVX2, NEON, and scalar fallback with zero runtime overhead from the dispatch logic itself.

For SharpInference CPU kernels, the recommended strategy is:
- Use `TensorPrimitives` for standard element-wise and reduction operations (Add, Multiply, Dot, SoftMax, Sigmoid, Exp, etc.) -- it already dispatches to the best available SIMD width internally.
- Write hand-rolled intrinsics only for fused multi-operation kernels (e.g., fused dequantize+GEMV, custom attention, GroupNorm) where TensorPrimitives cannot express the combined operation.
- Always provide an AVX2 fallback for every AVX-512 path. Use `Vector512.IsHardwareAccelerated` (not `Avx512F.IsSupported`) to gate 512-bit paths, because .NET deliberately reports `IsHardwareAccelerated = false` on older CPUs (Skylake-X, Cascade Lake) where AVX-512 causes severe downclocking.

---

## Detailed Findings

### 1. Namespace and Type Hierarchy

All hardware intrinsics live in `System.Runtime.Intrinsics` and sub-namespaces:

| Namespace | Key Classes |
|---|---|
| `System.Runtime.Intrinsics` | `Vector128<T>`, `Vector256<T>`, `Vector512<T>` |
| `System.Runtime.Intrinsics.X86` | `Sse`, `Sse2`, `Avx`, `Avx2`, `Fma`, `Avx512F`, `Avx512BW`, `Avx512CD`, `Avx512DQ`, `Avx512Vbmi`, `Avx10v1` |
| `System.Runtime.Intrinsics.Arm` | `AdvSimd`, `AdvSimd.Arm64`, `Dp`, `Rdm`, `Sha1`, `Sha256`, `Aes` |

**Class inheritance** (relevant chain):
```
Sse -> Sse2 -> Sse3 -> Ssse3 -> Sse41 -> Sse42 -> Avx -> Avx2 -> Avx512F
                                                                   -> Avx10v1
```

`Avx512F` has nested classes:
- `Avx512F.VL` -- AVX-512 instructions on 128/256-bit vectors
- `Avx512F.X64` -- 64-bit-only operations

Similarly, `Avx512BW`, `Avx512DQ`, `Avx512CD`, `Avx512Vbmi` each expose their own `IsSupported` property.

**Avx10v1** (added in .NET 9) unifies AVX-512 subsets under the Intel AVX10 converged ISA. `Avx10v1` inherits from `Avx2` and provides AVX-512-like operations on 128/256-bit vectors without requiring full 512-bit support. `Avx10v1.V512` gates the 512-bit operations.

Sources:
- [Avx512F Class -- Microsoft Learn](https://learn.microsoft.com/en-us/dotnet/api/system.runtime.intrinsics.x86.avx512f?view=net-10.0)
- [Avx10v1 Class -- Microsoft Learn](https://learn.microsoft.com/en-us/dotnet/api/system.runtime.intrinsics.x86.avx10v1?view=net-9.0)
- [System.Runtime.Intrinsics.X86 Namespace -- Microsoft Learn](https://learn.microsoft.com/en-us/dotnet/api/system.runtime.intrinsics.x86?view=net-9.0)

### 2. SIMD Dispatch Pattern

The canonical dispatch pattern in .NET uses `IsSupported` checks that the JIT evaluates at compile time, eliminating unreachable branches as dead code:

```csharp
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics.Arm;

public static void VectorAdd(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result)
{
    int i = 0;
    int len = a.Length;

    if (Vector512.IsHardwareAccelerated)
    {
        // AVX-512 path: process 16 floats per iteration
        for (; i <= len - Vector512<float>.Count; i += Vector512<float>.Count)
        {
            var va = Vector512.LoadUnsafe(ref MemoryMarshal.GetReference(a), (nuint)i);
            var vb = Vector512.LoadUnsafe(ref MemoryMarshal.GetReference(b), (nuint)i);
            Vector512.StoreUnsafe(va + vb, ref MemoryMarshal.GetReference(result), (nuint)i);
        }
    }
    else if (Vector256.IsHardwareAccelerated)
    {
        // AVX2 path: process 8 floats per iteration
        for (; i <= len - Vector256<float>.Count; i += Vector256<float>.Count)
        {
            var va = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(a), (nuint)i);
            var vb = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(b), (nuint)i);
            Vector256.StoreUnsafe(va + vb, ref MemoryMarshal.GetReference(result), (nuint)i);
        }
    }
    else if (Vector128.IsHardwareAccelerated)
    {
        // SSE2 or NEON path: process 4 floats per iteration
        for (; i <= len - Vector128<float>.Count; i += Vector128<float>.Count)
        {
            var va = Vector128.LoadUnsafe(ref MemoryMarshal.GetReference(a), (nuint)i);
            var vb = Vector128.LoadUnsafe(ref MemoryMarshal.GetReference(b), (nuint)i);
            Vector128.StoreUnsafe(va + vb, ref MemoryMarshal.GetReference(result), (nuint)i);
        }
    }

    // Scalar tail
    for (; i < len; i++)
        result[i] = a[i] + b[i];
}
```

**Key JIT behaviors:**
- `IsSupported` and `IsHardwareAccelerated` are JIT intrinsics -- they resolve to `true` or `false` at JIT time.
- Branches guarded by a false `IsSupported` are eliminated entirely; no code is emitted.
- This means you can write multi-ISA code in a single method without any runtime dispatch overhead.
- The JIT will also opportunistically use SIMD instructions for existing code where it determines benefit.

**Hardware-specific intrinsics dispatch** (when you need specific instructions, not just cross-platform vectors):

```csharp
public static unsafe float HorizontalSum(ReadOnlySpan<float> data)
{
    float sum = 0;
    int i = 0;

    if (Avx512F.IsSupported)
    {
        var acc = Vector512<float>.Zero;
        for (; i <= data.Length - 16; i += 16)
        {
            fixed (float* p = &data[i])
                acc = Avx512F.Add(acc, Avx512F.LoadVector512(p));
        }
        // Reduce 512 -> scalar
        var lo256 = acc.GetLower();
        var hi256 = acc.GetUpper();
        var sum256 = Avx.Add(lo256, hi256);
        var hi128 = sum256.GetUpper();
        var sum128 = Sse.Add(sum256.GetLower(), hi128);
        sum128 = Sse.Add(sum128, Sse.MoveHighToLow(sum128, sum128));
        sum128 = Sse.AddScalar(sum128, Sse.Shuffle(sum128, sum128, 0x01));
        sum = sum128.ToScalar();
    }
    else if (Avx2.IsSupported)
    {
        var acc = Vector256<float>.Zero;
        for (; i <= data.Length - 8; i += 8)
        {
            fixed (float* p = &data[i])
                acc = Avx.Add(acc, Avx.LoadVector256(p));
        }
        var hi128 = acc.GetUpper();
        var sum128 = Sse.Add(acc.GetLower(), hi128);
        sum128 = Sse.Add(sum128, Sse.MoveHighToLow(sum128, sum128));
        sum128 = Sse.AddScalar(sum128, Sse.Shuffle(sum128, sum128, 0x01));
        sum = sum128.ToScalar();
    }
    else if (AdvSimd.IsSupported)
    {
        var acc = Vector128<float>.Zero;
        for (; i <= data.Length - 4; i += 4)
        {
            fixed (float* p = &data[i])
                acc = AdvSimd.Add(acc, AdvSimd.LoadVector128(p));
        }
        // NEON pairwise add for horizontal reduction
        var pair = AdvSimd.Arm64.AddPairwise(acc, acc);
        sum = AdvSimd.Arm64.AddPairwise(pair, pair).ToScalar();
    }

    // Scalar tail
    for (; i < data.Length; i++)
        sum += data[i];

    return sum;
}
```

Sources:
- [Hardware Intrinsics in .NET 8 -- .NET Blog](https://devblogs.microsoft.com/dotnet/dotnet-8-hardware-intrinsics/)
- [Unlocking SIMD in .NET: A Practical Guide](https://developersvoice.com/blog/scalability/unlocking-simd-dotnet-guide/)

### 3. IsHardwareAccelerated vs IsSupported

These two properties serve different purposes:

| Property | Meaning | When to use |
|---|---|---|
| `Avx512F.IsSupported` | CPU supports AVX-512F instructions | When you need a specific AVX-512 instruction |
| `Vector512.IsHardwareAccelerated` | The JIT will emit native 512-bit instructions AND it is safe to do so | When using cross-platform `Vector512<T>` APIs |

**Critical distinction for AVX-512:** On Skylake-X and Cascade Lake, `Avx512F.IsSupported` may return `true` but `Vector512.IsHardwareAccelerated` returns `false`. The runtime deliberately suppresses 512-bit acceleration on these CPUs because AVX-512 causes significant frequency downclocking (see Section 6). On Ice Lake and newer, both return `true`.

**Recommendation for SharpInference:** Gate all 512-bit paths on `Vector512.IsHardwareAccelerated` unless you have benchmarked the specific kernel on older hardware and determined the 512-bit path is still faster despite downclocking.

Sources:
- [Hardware Intrinsics in .NET 8 -- .NET Blog](https://devblogs.microsoft.com/dotnet/dotnet-8-hardware-intrinsics/)

### 4. TensorPrimitives API

`TensorPrimitives` (namespace `System.Numerics.Tensors`) provides SIMD-accelerated operations over `Span<T>` / `ReadOnlySpan<T>`. It was introduced in .NET 8 with `float`-only overloads, expanded in .NET 9 with generic `<T>` overloads for any type implementing the appropriate numeric interface, and further refined in .NET 10.

**AI/ML-relevant methods (all available as both `float` and generic `<T>`):**

| Method | Description | SharpInference Use |
|---|---|---|
| `Add` | Element-wise addition | Residual connections |
| `Subtract` | Element-wise subtraction | General |
| `Multiply` | Element-wise multiplication | Scaling, attention |
| `Divide` | Element-wise division | Normalization |
| `FusedMultiplyAdd` | `(x * y) + z` in one pass | Bias addition after matmul |
| `Dot` | Dot product of two spans | GEMV inner loop |
| `Sum` | Sum of all elements | Normalization denominators |
| `SoftMax` | Softmax over a span | Attention scores |
| `Sigmoid` | Element-wise sigmoid | Activation functions |
| `Exp` | Element-wise e^x | SoftMax internals, GELU |
| `Tanh` | Element-wise tanh | Activation functions |
| `Log` | Element-wise ln(x) | Loss computation |
| `CosineSimilarity` | Cosine similarity of two spans | Embedding comparison |
| `Distance` | Euclidean distance | Embedding comparison |
| `Max` / `Min` | Element-wise max/min | ReLU, clamping |
| `Abs` | Element-wise absolute value | General |
| `Sqrt` / `Cbrt` | Element-wise roots | Normalization (1/sqrt) |
| `Clamp` | Element-wise clamping | Activation clamping |
| `Negate` | Element-wise negation | General |
| `IndexOfMax` / `IndexOfMin` | Argmax/argmin | Token selection |
| `ConvertToHalf` / `ConvertToSingle` | Half <-> float conversion | FP16 model support |
| `Norm` | Vector norm | Layer normalization |
| `Product` | Product of all elements | General |

**When to use TensorPrimitives vs hand-written intrinsics:**

| Use TensorPrimitives when... | Use hand-written intrinsics when... |
|---|---|
| Operation maps to a single TensorPrimitives method | You need to fuse multiple operations in a single pass over data |
| You want automatic dispatch across AVX-512/AVX2/NEON | You need specific instructions (e.g., `vpternlog`, `vpermt2ps`) |
| Portability across architectures matters | You are implementing quantized dequantization with custom bit layouts |
| Code maintainability is a priority | You have benchmarked and found TensorPrimitives leaves performance on the table |

**Example: Using TensorPrimitives for SoftMax:**

```csharp
using System.Numerics.Tensors;

public static void SoftMax(ReadOnlySpan<float> input, Span<float> output)
{
    // TensorPrimitives handles the numerically-stable implementation
    // (subtracts max before exp) and SIMD dispatch internally
    TensorPrimitives.SoftMax(input, output);
}
```

**Example: When hand-written is needed -- fused scale + bias + GELU:**

```csharp
// TensorPrimitives cannot express this fused operation in one pass.
// Three separate calls (Multiply, Add, then a custom GELU) would
// read/write the data three times instead of once.
public static unsafe void FusedScaleBiasGelu(
    ReadOnlySpan<float> input, ReadOnlySpan<float> scale,
    ReadOnlySpan<float> bias, Span<float> output)
{
    int i = 0;
    if (Avx2.IsSupported)
    {
        var half = Vector256.Create(0.5f);
        var one = Vector256.Create(1.0f);
        var coeff = Vector256.Create(0.044715f);
        var sqrt2pi = Vector256.Create(0.7978845608f); // sqrt(2/pi)

        for (; i <= input.Length - 8; i += 8)
        {
            var x = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(input), (nuint)i);
            var s = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(scale), (nuint)i);
            var b = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(bias), (nuint)i);

            // Fused: x = x * scale + bias
            x = Fma.MultiplyAdd(x, s, b);

            // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            var x3 = x * x * x;
            var inner = sqrt2pi * Fma.MultiplyAdd(coeff, x3, x);
            // ... tanh approximation via intrinsics ...

            Vector256.StoreUnsafe(x, ref MemoryMarshal.GetReference(output), (nuint)i);
        }
    }
    // scalar tail...
}
```

Sources:
- [TensorPrimitives Class -- Microsoft Learn](https://learn.microsoft.com/en-us/dotnet/api/system.numerics.tensors.tensorprimitives?view=net-10.0-pp)
- [TensorPrimitives.SoftMax -- Microsoft Learn](https://learn.microsoft.com/en-us/dotnet/api/system.numerics.tensors.tensorprimitives.softmax?view=net-10.0-pp)
- [TensorPrimitives.Sigmoid -- Microsoft Learn](https://learn.microsoft.com/en-us/dotnet/api/system.numerics.tensors.tensorprimitives.sigmoid?view=net-10.0-pp)
- [What's new in .NET 9 libraries -- Microsoft Learn](https://learn.microsoft.com/en-us/dotnet/core/whats-new/dotnet-9/libraries)

### 5. Memory Alignment

Modern x86 CPUs handle unaligned loads/stores with minimal penalty for most operations. However, aligned memory can still help with:
- Avoiding cache-line splits on 64-byte boundaries (particularly relevant for AVX-512 which uses full 64-byte cache lines)
- Enabling aligned load instructions (`Avx.LoadAlignedVector256`) which are guaranteed not to fault on unaligned data

**NativeMemory.AlignedAlloc** (available since .NET 6):

```csharp
using System.Runtime.InteropServices;

// Allocate 4096 floats aligned to 64 bytes (AVX-512 cache line)
nuint byteCount = 4096 * sizeof(float);
nuint alignment = 64; // Must be power of 2
float* aligned = (float*)NativeMemory.AlignedAlloc(byteCount, alignment);

try
{
    var span = new Span<float>(aligned, 4096);
    // Use span with SIMD operations...
}
finally
{
    // MUST use AlignedFree, not Free
    NativeMemory.AlignedFree(aligned);
}
```

**Alignment constants for SharpInference:**

| Vector Width | Register Size | Recommended Alignment |
|---|---|---|
| `Vector128<float>` | 16 bytes (4 floats) | 16 bytes |
| `Vector256<float>` | 32 bytes (8 floats) | 32 bytes |
| `Vector512<float>` | 64 bytes (16 floats) | 64 bytes |

**Practical note:** The `LoadUnsafe` / `StoreUnsafe` methods on `Vector128/256/512` do NOT require alignment and are the recommended default. Use aligned loads (`LoadAlignedVector256`, etc.) only in hot inner loops where benchmarking shows measurable benefit.

Sources:
- [NativeMemory.AlignedAlloc -- Microsoft Learn](https://learn.microsoft.com/en-us/dotnet/api/system.runtime.interopservices.nativememory.alignedalloc?view=net-10.0)

### 6. AVX-512 Downclocking on Intel CPUs

AVX-512 instructions can cause CPU frequency reduction on certain Intel microarchitectures. This is a critical concern for inference workloads that mix SIMD and scalar code.

**Frequency license levels (Skylake-X / Skylake-SP):**

| License | Trigger | Typical Frequency (example Xeon) |
|---|---|---|
| L0 (Normal) | Non-AVX or 128-bit SSE | 3.2 GHz |
| L1 (AVX) | 256-bit instructions | 2.8 GHz (~87.5%) |
| L2 (AVX-512) | Any 512-bit instruction | 2.4 GHz (~75%) |

**Transition penalties:**
- Voltage-only transition: 8-20 us of 4x reduced dispatch rate
- Frequency transition: ~11 us complete execution halt
- Relaxation period: ~680 us from last wide instruction before license downgrade
- A single 512-bit instruction triggers a chip-wide frequency drop affecting ALL cores

**By microarchitecture:**

| Generation | AVX-512 Downclocking Severity |
|---|---|
| Skylake-X / Skylake-SP | Severe -- 3 license tiers, heavy/light distinction, ~25% frequency loss |
| Cascade Lake | Severe -- same as Skylake-X |
| Ice Lake (client) | Minimal -- only 100 MHz on single-core, zero multi-core downclocking |
| Ice Lake (server, Xeon) | Moderate -- ~175 MHz average drop |
| Sapphire Rapids | None -- peak frequency similar with/without AVX-512 |
| Alder Lake (client) | AVX-512 fused off by Intel; not available |
| AMD Zen 4 | None -- no frequency penalty |

**Implications for SharpInference:**
- .NET's `Vector512.IsHardwareAccelerated` already handles the worst cases by returning `false` on Skylake-X/Cascade Lake.
- On Ice Lake server, there is still a modest (~5%) frequency penalty. Since inference workloads are sustained SIMD, the wider registers outweigh the penalty.
- On Sapphire Rapids and AMD Zen 4, there is no downclocking concern at all.
- For mixed workloads (SIMD kernel followed by scalar post-processing), the 680 us relaxation period means the CPU returns to full speed quickly.

Sources:
- [Gathering Intel on Intel AVX-512 Transitions -- Travis Downs](https://travisdowns.github.io/blog/2020/01/17/avxfreq1.html)
- [Ice Lake AVX-512 Downclocking -- Travis Downs](https://travisdowns.github.io/blog/2020/08/19/icl-avx512-freq.html)
- [The dangers of AVX-512 throttling -- Daniel Lemire](https://lemire.me/blog/2018/08/15/the-dangers-of-avx-512-throttling-a-3-impact/)
- [On the dangers of Intel's frequency scaling -- Cloudflare](https://blog.cloudflare.com/on-the-dangers-of-intels-frequency-scaling/)

### 7. ARM NEON (AdvSimd) Performance Characteristics

ARM NEON provides 128-bit SIMD via `System.Runtime.Intrinsics.Arm.AdvSimd`. Key characteristics for inference workloads:

**Register file and throughput:**
- 32x 128-bit registers (vs 16x 256-bit on AVX2, 32x 512-bit on AVX-512)
- Many ARM cores have 2-4 NEON execution units, so a core with four 128-bit units has equivalent peak throughput to a core with two 256-bit AVX2 units
- No frequency penalty for using SIMD -- ARM does not have the AVX downclocking problem

**Key differences from x86 SIMD:**
- 128-bit only (no 256/512-bit widths in NEON; SVE/SVE2 offers wider but is not yet in .NET)
- Rich pairwise operations: `AddPairwise`, `MaxPairwise` for horizontal reductions without the awkward shuffle-and-add patterns needed on x86
- Native `Half` (FP16) support on ARMv8.2+: `AdvSimd.Arm64` includes FP16 arithmetic, potentially useful for FP16 inference
- Fused multiply-add: `AdvSimd.FusedMultiplyAdd` maps directly to `fmla` instruction

**AdvSimd dispatch in .NET:**

```csharp
if (AdvSimd.IsSupported)
{
    // Basic NEON -- available on all ARM64
}
if (AdvSimd.Arm64.IsSupported)
{
    // ARM64-specific extensions (pairwise add, FP64 ops, etc.)
}
```

**Performance comparison (approximate, depends on specific core):**
- Single ARM Neoverse V1 core (AWS Graviton 3): ~128 GFLOPS FP32
- Single Intel Ice Lake core (AVX-512): ~150 GFLOPS FP32
- ARM wins on perf/watt; x86 wins on peak single-core throughput

**SVE/SVE2 note:** .NET has an open tracking issue ([dotnet/runtime#93095](https://github.com/dotnet/runtime/issues/93095)) for SVE/SVE2 intrinsics. SVE offers variable-length vectors (128-2048 bits). Not yet available in .NET 10.

Sources:
- [Comparing SIMD on x86-64 and arm64 -- Code & Visuals](https://blog.yiningkarlli.com/2021/09/neon-vs-sse.html)
- [ARM Neon Intrinsics Reference](https://arm-software.github.io/acle/neon_intrinsics/advsimd.html)
- [Arm64: Add SVE/SVE2 support -- dotnet/runtime#93095](https://github.com/dotnet/runtime/issues/93095)

---

## Key Numbers / Constants

| Constant | Value | Notes |
|---|---|---|
| `Vector128<float>.Count` | 4 | SSE2, NEON |
| `Vector256<float>.Count` | 8 | AVX/AVX2 |
| `Vector512<float>.Count` | 16 | AVX-512 |
| Recommended alignment (AVX2) | 32 bytes | `NativeMemory.AlignedAlloc(n, 32)` |
| Recommended alignment (AVX-512) | 64 bytes | `NativeMemory.AlignedAlloc(n, 64)` |
| Cache line size (x86) | 64 bytes | One Vector512 = one cache line |
| Cache line size (ARM) | 64 bytes (typically) | Varies by core |
| AVX-512 license relaxation | ~680 us | Time to return to L0 after last 512-bit instruction |
| AVX-512 transition halt | ~11 us | CPU halts during frequency change (Skylake-X) |
| `sizeof(float)` | 4 bytes | |
| `sizeof(Half)` | 2 bytes | FP16 |

---

## Data Layouts / Formats

### Vector Register Layouts

```
Vector128<float> (16 bytes):
  [ f0 | f1 | f2 | f3 ]
   0    4    8    12     (byte offsets)

Vector256<float> (32 bytes):
  [ f0 | f1 | f2 | f3 | f4 | f5 | f6 | f7 ]
   0    4    8    12   16   20   24   28

Vector512<float> (64 bytes):
  [ f0 | f1 | f2 | ... | f14 | f15 ]
   0    4    8         56    60
```

### Memory Layout for Tensor Buffers

For SharpInference CPU kernels, tensor data should be stored in contiguous `float[]` or `NativeMemory`-allocated buffers. The layout is row-major (C-contiguous), matching the convention used by ONNX, PyTorch, and GGUF dequantized outputs.

For SIMD processing, the innermost dimension should ideally be a multiple of 16 (Vector512 width) to avoid scalar tail processing. Padding the innermost dimension to a multiple of 16 is acceptable if documented.

---

## Algorithm Steps

### SIMD Dispatch Decision Flow (for each kernel)

1. Check if the operation is available in `TensorPrimitives`. If yes, use it.
2. If the operation requires fusing multiple steps, write a hand-rolled kernel.
3. In the hand-rolled kernel, structure the dispatch as:
   ```
   if Vector512.IsHardwareAccelerated -> AVX-512 path
   else if Vector256.IsHardwareAccelerated -> AVX2 path (REQUIRED)
   else if Vector128.IsHardwareAccelerated -> SSE2/NEON path
   else -> scalar fallback
   ```
4. Each SIMD path processes `VectorN<float>.Count` elements per iteration.
5. After the SIMD loop, process remaining elements with a scalar tail loop.
6. For reductions (sum, max, dot product), reduce the vector accumulator to scalar after the SIMD loop.

### Horizontal Reduction Pattern

```
512-bit accumulator
  -> split into two 256-bit halves, add
  -> split into two 128-bit halves, add
  -> shuffle+add within 128 bits to get scalar
```

On ARM NEON, use `AddPairwise` instead of shuffle-based reduction.

---

## Reference Implementations

### 1. Cross-Platform Vector Add (using Vector256/512 APIs)
See Section 2 above for the complete implementation.

### 2. Hardware-Specific Horizontal Sum (using Avx2/Avx512F/AdvSimd)
See Section 2 above for the complete implementation.

### 3. Dot Product Using TensorPrimitives

```csharp
using System.Numerics.Tensors;

public static float DotProduct(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
{
    return TensorPrimitives.Dot(a, b);
}
```

### 4. Aligned Memory Allocation Helper

```csharp
using System.Runtime.InteropServices;

public sealed unsafe class AlignedBuffer<T> : IDisposable where T : unmanaged
{
    private T* _ptr;
    public int Length { get; }
    public Span<T> Span => new(_ptr, Length);

    public AlignedBuffer(int count, nuint alignment = 64)
    {
        Length = count;
        _ptr = (T*)NativeMemory.AlignedAlloc((nuint)(count * sizeof(T)), alignment);
        // Zero-initialize
        NativeMemory.Clear(_ptr, (nuint)(count * sizeof(T)));
    }

    public void Dispose()
    {
        if (_ptr != null)
        {
            NativeMemory.AlignedFree(_ptr);
            _ptr = null;
        }
    }
}
```

### 5. Complete Fallback-Chained Kernel Template

```csharp
/// <summary>
/// Template for a SharpInference CPU kernel with full SIMD dispatch.
/// </summary>
public static class KernelTemplate
{
    public static void Execute(ReadOnlySpan<float> src, Span<float> dst)
    {
        if (Vector512.IsHardwareAccelerated)
            Execute512(src, dst);
        else if (Vector256.IsHardwareAccelerated)
            Execute256(src, dst);
        else if (Vector128.IsHardwareAccelerated)
            Execute128(src, dst);
        else
            ExecuteScalar(src, dst);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Execute512(ReadOnlySpan<float> src, Span<float> dst)
    {
        int i = 0;
        ref float srcRef = ref MemoryMarshal.GetReference(src);
        ref float dstRef = ref MemoryMarshal.GetReference(dst);

        for (; i <= src.Length - Vector512<float>.Count; i += Vector512<float>.Count)
        {
            var v = Vector512.LoadUnsafe(ref srcRef, (nuint)i);
            // ... kernel operation ...
            Vector512.StoreUnsafe(v, ref dstRef, (nuint)i);
        }
        // Process remainder with 256-bit
        for (; i <= src.Length - Vector256<float>.Count; i += Vector256<float>.Count)
        {
            var v = Vector256.LoadUnsafe(ref srcRef, (nuint)i);
            Vector256.StoreUnsafe(v, ref dstRef, (nuint)i);
        }
        // Scalar tail
        for (; i < src.Length; i++)
            dst[i] = src[i];
    }

    // Execute256, Execute128, ExecuteScalar follow the same pattern...
}
```

---

## Differences Between Implementations

### TensorPrimitives vs Hand-Written SIMD

| Aspect | TensorPrimitives | Hand-Written Intrinsics |
|---|---|---|
| **Dispatch** | Automatic internal dispatch to best ISA | Manual `if/else` chain |
| **Fusion** | Single-operation only; each call traverses data once | Can fuse multiple operations in one pass |
| **Portability** | Works on all .NET platforms, downlevel via NuGet | Must write separate paths per ISA |
| **Performance ceiling** | Near-optimal for single operations | Can exceed TensorPrimitives for fused kernels |
| **Maintenance** | Zero -- Microsoft maintains the code | High -- must maintain per-ISA implementations |
| **Numeric stability** | Built-in (e.g., SoftMax subtracts max internally) | Must implement stability techniques manually |

### AVX2 vs AVX-512 for Inference

| Aspect | AVX2 (256-bit) | AVX-512 (512-bit) |
|---|---|---|
| Elements per vector (float) | 8 | 16 |
| Theoretical throughput gain | baseline | 2x |
| Actual gain (typical kernels) | baseline | 1.3-1.8x (due to lower clocks on some CPUs) |
| Availability | All x86-64 CPUs since ~2013 | Intel Ice Lake+, AMD Zen 4+ |
| Masking support | Requires blending | Native k-mask registers |
| Unique instructions | -- | `vpternlog`, `vpermt2ps`, `vfixupimm`, etc. |

### x86 SIMD vs ARM NEON

| Aspect | x86 (AVX2/AVX-512) | ARM (NEON) |
|---|---|---|
| Max vector width | 512 bits | 128 bits (NEON); 128-2048 bits (SVE, not in .NET yet) |
| Register count | 16 (AVX2) / 32 (AVX-512) | 32 |
| Horizontal reductions | Awkward shuffle+add | Native `AddPairwise` |
| FP16 arithmetic | AVX-512 FP16 (Sapphire Rapids+) | AdvSimd.Arm64 on ARMv8.2+ |
| Frequency penalty | Yes (pre-Sapphire Rapids) | None |
| Perf/watt | Lower | Higher |

---

## Open Questions

- [ ] Exact performance of TensorPrimitives.SoftMax vs hand-rolled SoftMax with fused max-subtraction -- needs benchmarking on target hardware.
- [ ] Whether `Avx10v1` should be a dispatch target separate from `Avx512F` in SharpInference, or if `Vector256/512.IsHardwareAccelerated` covers all cases.
- [ ] Impact of .NET 10 tiered compilation on SIMD kernel performance -- do Tier 0 (quick-JIT) compilations of SIMD code cause performance cliffs that matter for inference warmup?

---

## Implementation Notes

### For SharpInference.Cpu Kernel Authors

1. **Always benchmark.** SIMD performance is highly dependent on the specific CPU, data size, and access pattern. Never assume wider is faster.

2. **Prefer `LoadUnsafe`/`StoreUnsafe` over pointer-based loads.** The `ref`-based APIs are GC-safe and the JIT generates identical machine code. Use `fixed` and raw pointers only when interfacing with native memory.

3. **Use `MethodImplOptions.AggressiveInlining`** on all ISA-specific helper methods. The JIT needs to see the `IsSupported` check and the intrinsic call in the same compilation unit to eliminate dead branches.

4. **Tail handling options:**
   - Scalar loop (simplest, always correct)
   - Overlapping last vector (process `data[len - VectorSize .. len]`, may re-process some elements; safe for idempotent operations like ReLU)
   - Masked operations (AVX-512 only, via `Avx512F` mask registers)

5. **Thread safety:** SIMD operations are inherently thread-safe (no shared state in registers). Parallelize over independent rows/channels using `Parallel.For` or the SharpInference thread pool.

6. **NuGet package:** `TensorPrimitives` is in `System.Numerics.Tensors` -- it ships in-box with .NET 8+ and is available as a NuGet package for older frameworks. For SharpInference targeting .NET 10, no additional package reference is needed.

7. **Testing:** Every kernel must be tested with:
   - Length = 0 (empty span)
   - Length = 1 (scalar only)
   - Length = VectorSize - 1 (just under one full vector)
   - Length = VectorSize (exactly one vector)
   - Length = VectorSize * N + remainder (exercises both SIMD loop and tail)
   - Verify results against `TensorPrimitives` or a known-good scalar implementation with tolerance of 1e-6 for float32.
