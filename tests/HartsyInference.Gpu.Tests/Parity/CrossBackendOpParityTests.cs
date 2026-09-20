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
}
