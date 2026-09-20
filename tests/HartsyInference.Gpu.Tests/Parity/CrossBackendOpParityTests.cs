using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Gpu.Tests.Parity;

/// <summary>The same op, on every GPU backend, against the CPU implementation.
///
/// <para>One test per op instead of one per op per backend. Each suite has been checking its own backend against a
/// reference written inline in that suite, so the two backends are compared to two different references and never
/// to each other — and a test can only be written for hardware the author had. A theory over
/// <see cref="BackendGate.GpuKinds"/> runs on whatever is present and says plainly which rows it skipped.</para>
///
/// <para>This exists mainly for what comes next. Moving a backend onto the shared residency cache is a change that
/// can alter results without failing anything, and the Vulkan migration's bugs were caught only because its suite
/// happens to build and tear down a backend per test. CUDA should have that on purpose.</para></summary>
[Trait("Category", "GpuIntegration")]
public sealed class CrossBackendOpParityTests(ITestOutputHelper output)
{
    private readonly ITestOutputHelper _out = output;

    private static Tensor Random(TensorShape shape, int seed, float offset = 0f)
    {
        Tensor tensor = new(shape, DType.F32);
        Random rng = new(seed);
        Span<float> values = tensor.AsSpan<float>();
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = (float)(rng.NextDouble() * 2 - 1) + offset;
        }
        return tensor;
    }

    [Theory]
    [MemberData(nameof(BackendGate.GpuKinds), MemberType = typeof(BackendGate))]
    public void Silu_Matches_The_Cpu(string kind)
    {
        if (!BackendGate.TryOpen(kind, _out.WriteLine, out IBackend? gpu))
        {
            return;
        }
        using IBackend backend = gpu!;
        using CpuBackend cpu = new();

        using Tensor input = Random(new TensorShape(4, 1024), seed: 7);
        using Tensor actual = new(input.Shape, DType.F32);
        using Tensor expected = new(input.Shape, DType.F32);

        backend.Silu(actual, input);
        cpu.Silu(expected, input);

        TensorAssert.Close(actual, expected, because: $"on {kind}");
    }

    /// <summary>Per-row argmax, including the exact ties a reduction resolves by accident.</summary>
    /// <remarks>Two rows carry a deliberate tie for the maximum. Which thread holds which candidate in a tree
    /// reduction is an artifact of the stride order, so without an explicit tie-break the winner depends on the
    /// workgroup size — and a tie is not exotic for a head that saturates. The row width is deliberately larger
    /// than one workgroup so each thread makes several candidates before the reduction starts.</remarks>
    [Theory]
    [MemberData(nameof(BackendGate.GpuKinds), MemberType = typeof(BackendGate))]
    public void ArgMaxLastDim_Matches_The_Cpu(string kind)
    {
        if (!BackendGate.TryOpen(kind, _out.WriteLine, out IBackend? gpu))
        {
            return;
        }
        using IBackend backend = gpu!;
        using CpuBackend cpu = new();

        const int rows = 6, cols = 777;
        using Tensor input = Random(new TensorShape(rows, cols), seed: 71);
        Span<float> values = input.AsSpan<float>();
        // Row 2: the maximum appears twice, first at 40. Row 4: three times, first at 5.
        values[2 * cols + 40] = 9f;
        values[2 * cols + 600] = 9f;
        values[4 * cols + 5] = 12f;
        values[4 * cols + 300] = 12f;
        values[4 * cols + 776] = 12f;

        using Tensor actual = new(new TensorShape(rows), DType.I32);
        using Tensor expected = new(actual.Shape, DType.I32);

        backend.ArgMaxLastDim(actual, input);
        ((IBackend)cpu).ArgMaxLastDim(expected, input);

        Assert.Equal(40, actual.AsReadOnlySpan<int>()[2]);
        Assert.Equal(5, actual.AsReadOnlySpan<int>()[4]);
        TensorAssert.Identical(actual, expected, because: $"on {kind}");
    }

    /// <summary>The GPT-J pairing, against the GPT-NeoX one it now shares a kernel with.</summary>
    /// <remarks>Both conventions are exercised, and two partial rotaries: the two differ only in which elements
    /// pair up and where their frequencies live, so a kernel that confused them still produces plausible numbers
    /// of the right magnitude in the right places. Only a reference comparison separates them.
    ///
    /// <para>An ODD rotary dim is in there because it is where a pair-based convention has to decide what to do
    /// with the pair straddling the boundary, and the reference and the kernel answered that differently until it
    /// was asked. Five is not a shape any model ships, which is exactly why nothing caught it.</para></remarks>
    [Theory]
    [InlineData("cuda", 0)]
    [InlineData("cuda", 4)]
    [InlineData("cuda", 5)]
    [InlineData("vulkan", 0)]
    [InlineData("vulkan", 4)]
    [InlineData("vulkan", 5)]
    public void ApplyRopeInterleaved_Matches_The_Cpu(string kind, int rotaryDim)
    {
        if (!BackendGate.TryOpen(kind, _out.WriteLine, out IBackend? gpu))
        {
            return;
        }
        using IBackend backend = gpu!;
        using CpuBackend cpu = new();

        const int batch = 2, seqLen = 5, heads = 3, headDim = 8;
        using Tensor actual = Random(new TensorShape(batch, seqLen, heads, headDim), seed: 31);
        using Tensor expected = new(actual.Shape, DType.F32);
        actual.AsReadOnlySpan<float>().CopyTo(expected.AsSpan<float>());
        using Tensor cos = Random(new TensorShape(batch, seqLen, headDim), seed: 32);
        using Tensor sin = Random(new TensorShape(batch, seqLen, headDim), seed: 33);

        backend.ApplyRopeInterleaved(actual, cos, sin, rotaryDim);
        ((IBackend)cpu).ApplyRopeInterleaved(expected, cos, sin, rotaryDim);

        TensorAssert.Close(actual, expected, because: $"on {kind}, rotaryDim {rotaryDim}");
    }

    /// <summary>The split-half pairing, re-checked because it now comes off the same kernel.</summary>
    [Theory]
    [MemberData(nameof(BackendGate.GpuKinds), MemberType = typeof(BackendGate))]
    public void ApplyRopeSingle_Matches_The_Cpu(string kind)
    {
        if (!BackendGate.TryOpen(kind, _out.WriteLine, out IBackend? gpu))
        {
            return;
        }
        using IBackend backend = gpu!;
        using CpuBackend cpu = new();

        const int batch = 2, seqLen = 5, heads = 3, headDim = 8;
        using Tensor actual = Random(new TensorShape(batch, seqLen, heads, headDim), seed: 41);
        using Tensor expected = new(actual.Shape, DType.F32);
        actual.AsReadOnlySpan<float>().CopyTo(expected.AsSpan<float>());
        using Tensor cos = Random(new TensorShape(batch, seqLen, headDim), seed: 42);
        using Tensor sin = Random(new TensorShape(batch, seqLen, headDim), seed: 43);

        backend.ApplyRopeSingle(actual, cos, sin);
        ((IBackend)cpu).ApplyRopeSingle(expected, cos, sin);

        TensorAssert.Close(actual, expected, because: $"on {kind}");
    }

    /// <summary>F32 to bfloat16, against the rounding rule the GPU kernels actually implement.</summary>
    /// <remarks>Deliberately NOT compared against <c>CpuBackend</c>. <c>Tensor.CastTo(BF16)</c> truncates — its own
    /// doc says so — while both GPU kernels round to nearest, ties to even, which is what hardware and every other
    /// framework do. That divergence is pre-existing and shipped: CUDA has overridden this op for a long time and
    /// differs from the host cast by one unit in the last place on roughly half of all inputs. Picking a side here
    /// would change one backend's numerics to make a test pass, so this pins what the GPUs do and the difference
    /// is reported rather than papered over.
    ///
    /// <para>Exact, and the count is not a multiple of the workgroup size. NaN, both infinities and both zeros
    /// are planted in the input: infinities and zeros round like anything else, but a NaN does not — rounding can
    /// carry its mantissa into the exponent — so the kernel has a branch for it that random values never reach.
    /// The NaN element is checked for NaN-ness rather than bytes: the two backends disagree on its sign, and IEEE
    /// fixes neither the sign nor the payload of a produced NaN.</para></remarks>
    [Theory]
    [MemberData(nameof(BackendGate.GpuKinds), MemberType = typeof(BackendGate))]
    public void CastToBf16_RoundsToNearestEven(string kind)
    {
        if (!BackendGate.TryOpen(kind, _out.WriteLine, out IBackend? gpu))
        {
            return;
        }
        using IBackend backend = gpu!;

        using Tensor input = Random(new TensorShape(1025), seed: 61);
        using Tensor actual = new(input.Shape, DType.BF16);
        using Tensor expected = new(input.Shape, DType.BF16);

        Span<float> writable = input.AsSpan<float>();
        // The kernel has a branch for non-finite input — rounding a NaN can carry its mantissa into the exponent,
        // so it emits a canonical quiet NaN instead. Random values never reach it, which would leave that branch
        // free to rot.
        writable[0] = float.NaN;
        writable[1] = float.PositiveInfinity;
        writable[2] = float.NegativeInfinity;
        writable[3] = 0f;
        writable[4] = -0f;

        ReadOnlySpan<float> source = input.AsReadOnlySpan<float>();
        Span<ushort> reference = expected.AsSpan<ushort>();
        for (int i = 0; i < source.Length; i++)
        {
            uint bits = BitConverter.SingleToUInt32Bits(source[i]);
            // Add half an output ULP, biased by the low bit of the result, then drop the low 16 mantissa bits.
            reference[i] = (ushort)((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
        }

        backend.CastToBf16(actual, input);

        // The NaN is checked for NaN-ness, not for bytes. Vulkan emits the canonical positive quiet NaN and CUDA
        // keeps the input's sign, so they differ at that element — and IEEE 754 fixes neither the sign nor the
        // payload of a produced NaN, so neither is wrong. Aligning them would mean editing a shipped kernel's
        // output to satisfy a test. Every other element, infinities and both zeros included, is compared exactly.
        ushort produced = actual.AsReadOnlySpan<ushort>()[0];
        Assert.True((produced & 0x7F80) == 0x7F80 && (produced & 0x007F) != 0,
            $"a NaN input produced 0x{produced:X4} on {kind}, which is not a NaN");
        reference[0] = produced;

        TensorAssert.Identical(actual, expected, because: $"on {kind}");
    }

    /// <summary>F16 input to the same op, which both backends reach by chaining through F32.</summary>
    /// <remarks>The values are chosen to be exactly representable in F16 so the comparison is about the BF16 step
    /// and not about what the F16 leg rounded away. Exact on both backends.</remarks>
    [Theory]
    [MemberData(nameof(BackendGate.GpuKinds), MemberType = typeof(BackendGate))]
    public void CastToBf16_AcceptsF16Input(string kind)
    {
        if (!BackendGate.TryOpen(kind, _out.WriteLine, out IBackend? gpu))
        {
            return;
        }
        using IBackend backend = gpu!;

        float[] values = [0f, 1f, -1f, 0.5f, -0.25f, 2f, -8f, 1.5f, 0.125f, -3.25f, 16f, -0.0625f, 4f];
        using Tensor wide = new(new TensorShape(values.Length), DType.F32);
        values.CopyTo(wide.AsSpan<float>());
        using Tensor half = new(wide.Shape, DType.F16);
        backend.CastToF16(half, wide);

        using Tensor actual = new(wide.Shape, DType.BF16);
        using Tensor expected = new(wide.Shape, DType.BF16);
        backend.CastToBf16(actual, half);
        Span<ushort> reference = expected.AsSpan<ushort>();
        for (int i = 0; i < values.Length; i++)
        {
            uint bits = BitConverter.SingleToUInt32Bits(values[i]);
            reference[i] = (ushort)((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
        }

        TensorAssert.Identical(actual, expected, because: $"on {kind}");
    }

    /// <summary>Per-head RoPE tables, where the whole difference from the shared-table form is one index.</summary>
    /// <remarks>Non-identity tables and more than one head on purpose: with a single head, or with cos/sin equal
    /// across heads, the per-head and shared layouts address the same bytes and a wrong index passes. The rotation
    /// is checked in place, as its callers use it.</remarks>
    [Theory]
    [MemberData(nameof(BackendGate.GpuKinds), MemberType = typeof(BackendGate))]
    public void WanRopeInterleavedPerHead_Matches_The_Cpu(string kind)
    {
        if (!BackendGate.TryOpen(kind, _out.WriteLine, out IBackend? gpu))
        {
            return;
        }
        using IBackend backend = gpu!;
        using CpuBackend cpu = new();

        const int seqLen = 6, heads = 3, headDim = 8;
        using Tensor actual = Random(new TensorShape(seqLen, heads, headDim), seed: 11);
        using Tensor expected = new(actual.Shape, DType.F32);
        actual.AsReadOnlySpan<float>().CopyTo(expected.AsSpan<float>());
        using Tensor cos = Random(new TensorShape(heads, seqLen, headDim), seed: 12);
        using Tensor sin = Random(new TensorShape(heads, seqLen, headDim), seed: 13);

        backend.WanRopeInterleavedPerHead(actual, cos, sin, seqLen, heads, headDim);
        // Through the interface: the reference for both forms is IBackend's own default, which
        // CpuBackend does not redeclare.
        ((IBackend)cpu).WanRopeInterleavedPerHead(expected, cos, sin, seqLen, heads, headDim);

        TensorAssert.Close(actual, expected, because: $"on {kind}");
    }

    /// <summary>The shared-table form, which now comes off the same shader — so it has to be re-checked.</summary>
    [Theory]
    [MemberData(nameof(BackendGate.GpuKinds), MemberType = typeof(BackendGate))]
    public void WanRopeInterleaved_Matches_The_Cpu(string kind)
    {
        if (!BackendGate.TryOpen(kind, _out.WriteLine, out IBackend? gpu))
        {
            return;
        }
        using IBackend backend = gpu!;
        using CpuBackend cpu = new();

        const int seqLen = 6, heads = 3, headDim = 8;
        using Tensor actual = Random(new TensorShape(seqLen, heads, headDim), seed: 21);
        using Tensor expected = new(actual.Shape, DType.F32);
        actual.AsReadOnlySpan<float>().CopyTo(expected.AsSpan<float>());
        using Tensor cos = Random(new TensorShape(seqLen, headDim), seed: 22);
        using Tensor sin = Random(new TensorShape(seqLen, headDim), seed: 23);

        backend.WanRopeInterleaved(actual, cos, sin, seqLen, heads, headDim);
        ((IBackend)cpu).WanRopeInterleaved(expected, cos, sin, seqLen, heads, headDim);

        TensorAssert.Close(actual, expected, because: $"on {kind}");
    }

    /// <summary>The last step of an image generation: F32 CHW in [-1,1] to u8 HWC.</summary>
    /// <remarks>Odd dimensions on purpose. The output is three bytes per pixel, so only every fourth pixel starts
    /// on a word boundary, and a kernel that composes whole words has a tail to get wrong — 5x7 leaves the final
    /// word holding one real byte and three past the image. The rounding is round-half-up rather than
    /// round-to-even, and a pixel off by one is a byte off in the PNG, so the comparison is exact.</remarks>
    [Theory]
    [MemberData(nameof(BackendGate.GpuKinds), MemberType = typeof(BackendGate))]
    public void ChwF32ToHwcU8_Matches_The_Cpu(string kind)
    {
        if (!BackendGate.TryOpen(kind, _out.WriteLine, out IBackend? gpu))
        {
            return;
        }
        using IBackend backend = gpu!;
        using CpuBackend cpu = new();

        const int height = 5, width = 7;
        // Offset so a good share of the values land outside [-1,1]: the clamp is part of the contract, and an
        // unclamped kernel wraps rather than saturating.
        using Tensor input = Random(new TensorShape(1, 3, height, width), seed: 51, offset: 0.6f);
        using Tensor actual = new(new TensorShape(height, width, 3), DType.U8);
        using Tensor expected = new(actual.Shape, DType.U8);

        backend.ChwF32ToHwcU8(actual, input);
        ((IBackend)cpu).ChwF32ToHwcU8(expected, input);

        TensorAssert.Identical(actual, expected, because: $"on {kind}");
    }

    [Theory]
    [MemberData(nameof(BackendGate.GpuKinds), MemberType = typeof(BackendGate))]
    public void RmsNorm_Matches_The_Cpu(string kind)
    {
        if (!BackendGate.TryOpen(kind, _out.WriteLine, out IBackend? gpu))
        {
            return;
        }
        using IBackend backend = gpu!;
        using CpuBackend cpu = new();

        using Tensor input = Random(new TensorShape(1, 8, 512), seed: 11, offset: 0.25f);
        using Tensor weight = Random(new TensorShape(512), seed: 12, offset: 1.0f);
        using Tensor actual = new(input.Shape, DType.F32);
        using Tensor expected = new(input.Shape, DType.F32);

        backend.RmsNorm(actual, input, weight, 1e-6f);
        cpu.RmsNorm(expected, input, weight, 1e-6f);

        // A reduction runs in a different order on a GPU, so this is a numerical agreement rather than an identity.
        TensorAssert.Close(actual, expected, rtol: 1e-4f, because: $"on {kind}");
    }

    /// <summary>The op whose batch handling kept SDXL off one backend entirely. Worth comparing across backends
    /// rather than each to its own reference, since the failure was a shape one backend simply refused.</summary>
    [Theory]
    [MemberData(nameof(BackendGate.GpuKinds), MemberType = typeof(BackendGate))]
    public void Batched_Conv2D_Matches_The_Cpu(string kind)
    {
        if (!BackendGate.TryOpen(kind, _out.WriteLine, out IBackend? gpu))
        {
            return;
        }
        using IBackend backend = gpu!;
        using CpuBackend cpu = new();

        const int B = 2, Cin = 8, Cout = 8, H = 16, W = 16;
        using Tensor input = Random(new TensorShape(B, Cin, H, W), seed: 21, offset: 0.4f);
        using Tensor weight = Random(new TensorShape(Cout, Cin, 3, 3), seed: 22);
        using Tensor bias = Random(new TensorShape(Cout), seed: 23);
        using Tensor actual = new(new TensorShape(B, Cout, H, W), DType.F32);
        using Tensor expected = new(new TensorShape(B, Cout, H, W), DType.F32);

        // Compare like with like. A backend that promotes F32 to a reduced-precision tensor-core format is not
        // wrong, but it is not answering the same question as an F32 CPU reference: caught immediately here, where
        // CUDA missed an F32 tolerance by ~3x on this op while Vulkan met it, because CUDA had TF32's ten mantissa
        // bits and Vulkan's F32 path has no tensor-core equivalent to promote to.
        backend.HighPrecisionGemm = true;
        backend.Conv2D(actual, input, weight, bias, 1, 1, 1, 1);
        cpu.Conv2D(expected, input, weight, bias, 1, 1, 1, 1);

        TensorAssert.Close(actual, expected, rtol: 1e-4f, because: $"on {kind}");
    }

    /// <summary>Prints what ran and what did not, so a green suite cannot be mistaken for full coverage.</summary>
    [Fact]
    public void Report_Which_Backends_Are_Present()
    {
        foreach (string kind in BackendGate.Kinds)
        {
            string? reason = BackendGate.UnavailableReason(kind);
            _out.WriteLine(reason is null ? $"{kind}: available" : $"{kind}: unavailable — {reason}");
        }
    }

    /// <summary>The split-half rotation on a head-major tensor, where the frequency row is not the vector index.</summary>
    /// <remarks>Batch and head counts are both greater than one on purpose. The two layouts hold the same element
    /// count with heads and seq swapped, so at heads == 1 or batch == 1 they coincide element for element and a
    /// kernel using the wrong one passes. Only [B &gt; 1, heads &gt; 1] separates them.
    ///
    /// <para>A partial rotary is included because it decides where the untouched tail of each head begins, and
    /// that tail is what a wrong row mapping would leave correct while corrupting everything before it.</para></remarks>
    [Theory]
    [InlineData("cuda", 0)]
    [InlineData("cuda", 4)]
    [InlineData("vulkan", 0)]
    [InlineData("vulkan", 4)]
    public void ApplyRopeSingleHeadMajor_Matches_The_Cpu(string kind, int rotaryDim)
    {
        if (!BackendGate.TryOpen(kind, _out.WriteLine, out IBackend? gpu))
        {
            return;
        }
        using IBackend backend = gpu!;
        using CpuBackend cpu = new();

        const int batch = 2, heads = 3, seqLen = 5, headDim = 8;
        using Tensor actual = Random(new TensorShape(batch, heads, seqLen, headDim), seed: 51);
        using Tensor expected = new(actual.Shape, DType.F32);
        actual.AsReadOnlySpan<float>().CopyTo(expected.AsSpan<float>());
        using Tensor cos = Random(new TensorShape(batch, seqLen, headDim), seed: 52);
        using Tensor sin = Random(new TensorShape(batch, seqLen, headDim), seed: 53);

        backend.ApplyRopeSingleHeadMajor(actual, cos, sin, rotaryDim);
        ((IBackend)cpu).ApplyRopeSingleHeadMajor(expected, cos, sin, rotaryDim);

        TensorAssert.Close(actual, expected, because: $"on {kind}, rotaryDim {rotaryDim}");
    }

    /// <summary>The head-major QKV split, over every subset of outputs a caller can ask for.</summary>
    /// <remarks>Four cases, because slot resolution has two branches and the interesting one is not the full call:
    /// a 3-wide source keeps the canonical q=0, k=1, v=2 segments even when only some outputs are wanted, while a
    /// narrowed source carries only what it names, numbered in q,k,v order. Reading k from segment 0 of a
    /// <c>[k|v]</c> buffer and reading it from segment 1 of a <c>[q|k|v]</c> one are both correct, for different
    /// sources, and a kernel that picks the wrong rule returns a well-formed tensor of the wrong values.
    ///
    /// <para>headDim is 96 — larger than the workgroup, so every thread accumulates several elements before the
    /// reduction, and not a power of two, so the cross-subgroup fold is not a no-op.</para></remarks>
    [Theory]
    [InlineData("cuda", 3, true, true, true)]
    [InlineData("cuda", 3, false, true, true)]
    [InlineData("cuda", 2, false, true, true)]
    [InlineData("cuda", 1, true, false, false)]
    [InlineData("vulkan", 3, true, true, true)]
    [InlineData("vulkan", 3, false, true, true)]
    [InlineData("vulkan", 2, false, true, true)]
    [InlineData("vulkan", 1, true, false, false)]
    public void QkvSplitNormHeadMajor_Matches_The_Cpu(string kind, int packStride, bool wantQ, bool wantK, bool wantV)
    {
        if (!BackendGate.TryOpen(kind, _out.WriteLine, out IBackend? gpu))
        {
            return;
        }
        using IBackend backend = gpu!;
        using CpuBackend cpu = new();

        const int batch = 2, heads = 2, seq = 3, headDim = 96;
        const int tokens = batch * seq, w = heads * headDim;
        const float eps = 1e-6f;
        TensorShape headed = new(batch, heads, seq, headDim);

        using Tensor qkv = Random(new TensorShape(tokens, packStride * w), seed: 61);
        using Tensor qWeight = Random(new TensorShape(headDim), seed: 62, offset: 1f);
        using Tensor kWeight = Random(new TensorShape(headDim), seed: 63, offset: 1f);

        using Tensor? qA = wantQ ? new Tensor(headed, DType.F32) : null;
        using Tensor? kA = wantK ? new Tensor(headed, DType.F32) : null;
        using Tensor? vA = wantV ? new Tensor(headed, DType.F32) : null;
        using Tensor? qE = wantQ ? new Tensor(headed, DType.F32) : null;
        using Tensor? kE = wantK ? new Tensor(headed, DType.F32) : null;
        using Tensor? vE = wantV ? new Tensor(headed, DType.F32) : null;

        backend.QkvSplitNormHeadMajor(qA, kA, vA, qkv, qWeight, kWeight, eps);
        ((IBackend)cpu).QkvSplitNormHeadMajor(qE, kE, vE, qkv, qWeight, kWeight, eps);

        string because = $"on {kind}, packStride {packStride}, q={wantQ} k={wantK} v={wantV}";
        if (wantQ) TensorAssert.Close(qA!, qE!, because: $"q {because}");
        if (wantK) TensorAssert.Close(kA!, kE!, because: $"k {because}");
        // v is copied, not normalized, so it is the one output that must match bit for bit.
        if (wantV) TensorAssert.Identical(vA!, vE!, because: $"v {because}");
    }

    /// <summary>Two chunks into one head-major buffer, which is the op — a single call is indistinguishable from
    /// a concat.</summary>
    /// <remarks>The second call has to find the buffer the first one left behind: allocating a fresh destination
    /// per call would pass a single-chunk test and silently lose chunk 1. The chunks are different lengths and
    /// neither they nor the head dim is a multiple of a workgroup, so a per-head destination stride that is off by
    /// one lands inside the assertion rather than past the end.
    ///
    /// <para>The chunks COVER the destination, and that is a deliberate limit on what this asserts. Both GPU
    /// backends allocate the destination without uploading its host contents — the destination is an attention
    /// key/value buffer and uploading it to write a chunk would move hundreds of megabytes — so rows outside every
    /// chunk hold whatever the allocation came with, while the interface reference leaves them untouched. Every
    /// shipped caller fills the whole buffer, so the divergence is unreachable; asserting on it here would pin the
    /// host's behaviour on hardware that deliberately does not implement it.</para></remarks>
    [Theory]
    [MemberData(nameof(BackendGate.GpuKinds), MemberType = typeof(BackendGate))]
    public void ScatterSeqHeadMajor_Matches_The_Cpu(string kind)
    {
        if (!BackendGate.TryOpen(kind, _out.WriteLine, out IBackend? gpu))
        {
            return;
        }
        using IBackend backend = gpu!;
        using CpuBackend cpu = new();

        const int first = 4, second = 3;
        const int heads = 3, seq = first + second, hd = 5;
        using Tensor actual = Random(new TensorShape(1, heads, seq, hd), seed: 71);
        using Tensor expected = new(actual.Shape, DType.F32);
        actual.AsReadOnlySpan<float>().CopyTo(expected.AsSpan<float>());

        using Tensor chunkA = Random(new TensorShape(1, heads, first, hd), seed: 72);
        using Tensor chunkB = Random(new TensorShape(1, heads, second, hd), seed: 73);

        backend.ScatterSeqHeadMajor(actual, chunkA, 0);
        backend.ScatterSeqHeadMajor(actual, chunkB, first);
        ((IBackend)cpu).ScatterSeqHeadMajor(expected, chunkA, 0);
        ((IBackend)cpu).ScatterSeqHeadMajor(expected, chunkB, first);

        TensorAssert.Identical(actual, expected, because: $"on {kind}");
    }
}
