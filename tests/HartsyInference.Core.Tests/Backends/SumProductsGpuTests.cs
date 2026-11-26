using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Backends.Reduction;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Core.Tests.Backends;

/// <summary><see cref="IBackend.SumProducts"/> on CUDA against the host reference, within the contract's bound, at the sizes its callers use.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class SumProductsGpuTests(ITestOutputHelper output)
{
    private const ProductTerms All = ProductTerms.Dot | ProductTerms.SquaredA | ProductTerms.SquaredB;

    /// <summary>LongCat's CFG-zero projection: one group over the guided frames of a video latent, and of a tiny one.</summary>
    [CudaTheory]
    [InlineData(16, 21, 60 * 104, 0)]
    [InlineData(16, 21, 60 * 104, 1)]
    [InlineData(4, 5, 24, 2)]
    [InlineData(1, 2, 3, 1)]
    public void CfgZeroProjection_MatchesTheReference(int channels, int frames, int hw, int fromFrame)
    {
        Random rng = new(1);
        using CudaBackend cuda = Open();
        using Tensor pos = Filled(rng, channels, frames, hw), neg = Filled(rng, channels, frames, hw);
        using Tensor posDevice = OnDevice(cuda, pos), negDevice = OnDevice(cuda, neg);
        StridedView view = StridedView.Of(pos.Shape).Slice(1, fromFrame, frames - fromFrame);
        (double dot, double squared) = ProductSums.DotAndSquaredB(new CpuBackend(), pos, neg, view);
        cuda.Sync();

        cuda.ResetD2hSyncCount();
        Stopwatch watch = Stopwatch.StartNew();
        (double actualDot, double actualSquared) = ProductSums.DotAndSquaredB(cuda, posDevice, negDevice, view);
        watch.Stop();

        output.WriteLine($"D2H_SYNCS cfg zero: {cuda.GetD2hSyncCount()}; first call {watch.Elapsed.TotalMilliseconds:F3} ms; {view.ElementCount} elements");
        Assert.Equal(1, cuda.GetD2hSyncCount());
        double magnitude = 0;
        ReadOnlySpan<float> p = pos.AsReadOnlySpan<float>(), n = neg.AsReadOnlySpan<float>();
        for (int i = 0; i < p.Length; i++) magnitude += Math.Abs((double)p[i] * n[i]);
        AssertWithin(dot, actualDot, view.ElementCount, magnitude, "dot");
        AssertWithin(squared, actualSquared, view.ElementCount, squared, "squared");
    }

    /// <summary>Rows, columns, a broadcast operand, one tensor twice, and groups on both sides of the size that takes partial sums.</summary>
    [CudaTheory]
    [InlineData("heads")]
    [InlineData("columns")]
    [InlineData("broadcast")]
    [InlineData("same")]
    [InlineData("whole block")]
    [InlineData("past a block")]
    [InlineData("most partials")]
    public void EveryGrouping_MatchesTheReference(string name)
    {
        (long[] aShape, long[] bShape, SumProductsSpec spec, bool same) = Case(name);
        Random rng = new(2);
        using CudaBackend cuda = Open();
        using Tensor a = Filled(rng, aShape), b = same ? a : Filled(rng, bShape);
        using Tensor aDevice = OnDevice(cuda, a);
        using Tensor bDevice = same ? aDevice : OnDevice(cuda, b);
        long groups = 1, perGroup = 1;
        for (int axis = 0; axis < spec.A.Rank; axis++)
        {
            if (axis < spec.KeptAxes) groups *= spec.A[axis];
            else perGroup *= spec.A[axis];
        }
        using Tensor expected = new(new TensorShape(groups, spec.TermCount), DType.F64), actual = new(expected.Shape, DType.F64);
        using Tensor narrow = new(expected.Shape, DType.F32), magnitude = new(new TensorShape(groups, 2), DType.F64);
        IBackend cpu = new CpuBackend();
        cpu.SumProducts(expected, a, b, spec);
        cpu.SumProducts(magnitude, a, b, new SumProductsSpec(spec.A, spec.B, spec.KeptAxes, ProductTerms.SquaredA | ProductTerms.SquaredB));
        cuda.Sync();

        cuda.ResetD2hSyncCount();
        cuda.SumProducts(actual, aDevice, bDevice, spec);
        cuda.SumProducts(narrow, aDevice, bDevice, spec);
        Assert.Equal(0, cuda.GetD2hSyncCount());

        ReadOnlySpan<double> e = expected.AsReadOnlySpan<double>(), r = actual.AsReadOnlySpan<double>(), m = magnitude.AsReadOnlySpan<double>();
        ReadOnlySpan<float> f = narrow.AsReadOnlySpan<float>();
        for (int i = 0; i < e.Length; i++)
        {
            // By Cauchy-Schwarz no term of the group sums past this in magnitude.
            int group = i / spec.TermCount;
            double bound = Math.Max(Math.Sqrt(m[group * 2] * m[group * 2 + 1]), Math.Max(m[group * 2], m[group * 2 + 1]));
            AssertWithin(e[i], r[i], perGroup, bound, $"{name}[{i}]");
            Assert.True(MathF.Abs(f[i] - (float)r[i]) <= MathF.BitIncrement(MathF.Abs(f[i])) - MathF.Abs(f[i]), $"{name}[{i}]: F32 {f[i]} vs F64 {r[i]}.");
        }
    }

    private static (long[] A, long[] B, SumProductsSpec Spec, bool Same) Case(string name)
    {
        StridedView columns = StridedView.Of([37, 50]).Permute([1, 0]);
        return name switch
        {
            // FastFrames' per-head rows: 12 heads of 128 over a frame's tokens.
            "heads" => ([1560, 12 * 128], [1560, 12 * 128], SumProductsSpec.PerRow(1560L * 12 * 128, 128, All), false),
            "columns" => ([37, 50], [37, 50], new SumProductsSpec(columns, columns, 1, All), false),
            "broadcast" => ([9, 70], [70], new SumProductsSpec(StridedView.Of([9, 70]), StridedView.VectorAlong([9, 70], 1), 1, ProductTerms.Dot), false),
            "same" => ([5, 300], [5, 300], SumProductsSpec.PerRow(1500, 300, ProductTerms.Dot), true),
            "whole block" => ([3, 16384], [3, 16384], SumProductsSpec.PerRow(3 * 16384, 16384, All), false),
            "past a block" => ([3, 16385], [3, 16385], SumProductsSpec.PerRow(3 * 16385, 16385, All), false),
            _ => ([2, 16384 * 300], [2, 16384 * 300], SumProductsSpec.PerRow(2L * 16384 * 300, 16384 * 300, ProductTerms.Dot | ProductTerms.SquaredB), false),
        };
    }

    [CudaFact]
    public void AnF64Output_SurvivesTheSweepWhenPinned_AndReadsBack()
    {
        // The F64 output is an activation like any other: pinned, it survives the sweep; read, it gives both doubles.
        using CudaBackend cuda = Open();
        using Tensor a = Filled(new Random(3), 64);
        using Tensor device = OnDevice(cuda, a), sums = new(new TensorShape(1), DType.F64);
        cuda.SumProducts(sums, device, device, SumProductsSpec.Over(StridedView.Of(a.Shape), ProductTerms.Dot));
        cuda.PinActivation(sums);
        cuda.FreeActivations();

        double expected = 0;
        foreach (float v in a.AsReadOnlySpan<float>()) expected += (double)v * v;
        AssertWithin(expected, sums.AsReadOnlySpan<double>()[0], 64, expected, "pinned");
        cuda.UnpinActivation(sums);
    }

    /// <summary>The contract's bound between two implementations.</summary>
    private static void AssertWithin(double expected, double actual, long count, double magnitude, string what)
    {
        double bound = count * Math.ScaleB(1, -52) * magnitude;
        Assert.True(Math.Abs(expected - actual) <= bound, $"{what}: {actual:R} vs {expected:R}, {Math.Abs(expected - actual):E3} apart, bound {bound:E3}.");
    }

    private static Tensor OnDevice(CudaBackend cuda, Tensor host)
    {
        Tensor device = new(host.Shape, DType.F32);
        cuda.Scale(device, host, 1f);
        return device;
    }

    private static Tensor Filled(Random rng, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        foreach (ref float v in t.AsSpan<float>()) v = (float)(rng.NextDouble() * 4 - 2);
        return t;
    }

    private static CudaBackend Open()
    {
        string beside = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (Directory.Exists(beside)) return new CudaBackend(0, beside);
        for (DirectoryInfo? dir = new(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "src", "HartsyInference.Cuda", "Ptx");
            if (Directory.Exists(candidate)) return new CudaBackend(0, candidate);
        }
        throw new DirectoryNotFoundException("No CUDA PTX directory beside the test assembly or in the repository.");
    }

    private sealed class CudaFactAttribute : FactAttribute
    {
        public CudaFactAttribute()
        {
            if (!CudaContext.IsAvailable()) Skip = "SKIPPED: CUDA unavailable.";
        }
    }

    private sealed class CudaTheoryAttribute : TheoryAttribute
    {
        public CudaTheoryAttribute()
        {
            if (!CudaContext.IsAvailable()) Skip = "SKIPPED: CUDA unavailable.";
        }
    }
}
