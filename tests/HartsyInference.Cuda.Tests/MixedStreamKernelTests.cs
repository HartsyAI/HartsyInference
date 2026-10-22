using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>The kernels that join an F32 residual stream to F16 branches. LayerNorm and LayerNormModulate from an F32 input into
/// F16 match the F32 kernels rounded to F16 within one F16 step (the two reduce the row in a different order); GatedResidualLastDim
/// adding an F16 value to an F32 residual matches the F32 kernel over the widened value bit for bit.</summary>
[Collection("CudaSerial")]
public sealed unsafe class MixedStreamKernelTests(ITestOutputHelper output)
{
    private const int Rows = 37, Dim = 3072;

    private readonly ITestOutputHelper _output = output;

    [Trait("Category", "GpuIntegration")]
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ANormIntoF16MatchesTheF32NormRoundedToF16(bool modulate)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        using CudaBackend cuda = new(0, PtxDir());
        IBackend b = cuda;
        int seed = 7;
        using Tensor x = Stream(ref seed);
        using Tensor first = Uniform(ref seed, modulate ? 0.5f : 0.1f, modulate ? 0f : 1f, 1, Dim);
        using Tensor second = Uniform(ref seed, modulate ? 0.5f : 0.05f, 0f, 1, Dim);
        TensorShape shape = new(Rows, Dim);
        using Tensor wide = new(shape, DType.F32);
        using Tensor narrow = new(shape, DType.F16);
        if (modulate)
        {
            b.LayerNormModulate(wide, x, first, second, 1e-6f);
            b.LayerNormModulate(narrow, x, first, second, 1e-6f);
        }
        else
        {
            b.LayerNorm(wide, x, first, second, 1e-6f);
            b.LayerNorm(narrow, x, first, second, 1e-6f);
        }
        cuda.Sync();

        float* w = (float*)wide.DataPointer;
        Half* n = (Half*)narrow.DataPointer;
        int worst = 0, off = 0;
        for (long i = 0; i < shape.ElementCount; i++)
        {
            int steps = Math.Abs(Ordered((Half)w[i]) - Ordered(n[i]));
            if (steps > 0) off++;
            worst = Math.Max(worst, steps);
        }
        _output.WriteLine($"{(modulate ? "LayerNormModulate" : "LayerNorm")} F32 into F16: {off} of {shape.ElementCount} values off by up to {worst} F16 step(s).");
        Assert.True(worst <= 1, $"The F16 norm is {worst} F16 steps from the F32 norm rounded to F16.");
    }

    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void AnF16ValueAddedToAnF32ResidualMatchesTheF32KernelBitForBit()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        using CudaBackend cuda = new(0, PtxDir());
        IBackend b = cuda;
        int seed = 11;
        using Tensor residual = Stream(ref seed);
        using Tensor value16 = ToF16(Uniform(ref seed, 40f, 0f, Rows, Dim));
        using Tensor value32 = value16.CastTo(DType.F32);
        using Tensor gate = Uniform(ref seed, 0.6f, 0f, 1, Dim);
        TensorShape shape = new(Rows, Dim);
        using Tensor expected = new(shape, DType.F32);
        using Tensor mixed = new(shape, DType.F32);
        b.GatedResidualLastDim(expected, residual, value32, gate);
        b.GatedResidualLastDim(mixed, residual, value16, gate);
        cuda.Sync();

        ReadOnlySpan<int> e = new(expected.DataPointer, Rows * Dim), m = new(mixed.DataPointer, Rows * Dim);
        Assert.True(e.SequenceEqual(m), "The F16-value gated residual differs from the F32 kernel over the widened value.");
    }

    /// <summary>F16 bits as an integer that orders like the values, with both zeros at 0, so a difference counts F16 steps.</summary>
    private static int Ordered(Half h)
    {
        int bits = (ushort)BitConverter.HalfToInt16Bits(h);
        return (bits & 0x8000) != 0 ? -(bits & 0x7FFF) : bits;
    }

    private static Tensor ToF16(Tensor f32)
    {
        Tensor h = f32.CastTo(DType.F16);
        f32.Dispose();
        return h;
    }

    /// <summary>A residual stream row set: unit-scale values with a few outlier channels in the hundreds.</summary>
    private static Tensor Stream(ref int seed)
    {
        Tensor t = Uniform(ref seed, 6f, 0f, Rows, Dim);
        float* p = (float*)t.DataPointer;
        for (int c = 0; c < 8; c++)
            for (int r = 0; r < Rows; r++)
                p[(long)r * Dim + (c * 389 + 5) % Dim] = ((c & 1) == 0 ? 1f : -1f) * (500f + 3f * r);
        return t;
    }

    private static Tensor Uniform(ref int seed, float amplitude, float center, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++)
        {
            seed = seed * 1103515245 + 12345;
            p[i] = center + ((seed >> 16) & 0x7fff) / 32768f * 2f * amplitude - amplitude;
        }
        return t;
    }

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }
}
