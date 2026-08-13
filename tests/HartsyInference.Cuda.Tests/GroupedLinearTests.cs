using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Pins grouped resident-int8 <c>LinearMulti</c> against the per-op <c>Linear</c> it replaces. The bar is
/// BYTE-IDENTICAL, not close: grouping only shares the activation's rotate+quant pass between weights, so the same
/// input pointer through the same kernel must produce the same int8 codes and therefore the same output bits. Any
/// drift means the shared buffer is being read wrong (a stale row chunk, a mis-sized accumulator, a row-scale
/// pointer crossed between targets), which as video would be invisible.</summary>
[Collection("CudaSerial")]
[Trait("Category", "GpuIntegration")]
public sealed unsafe class GroupedLinearTests
{
    private readonly ITestOutputHelper _output;
    public GroupedLinearTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    private static Tensor Int8Weight(int n, int k, int group, Random rng)
    {
        Tensor w = new Tensor(new TensorShape(n, k), DType.I8);
        sbyte* p = (sbyte*)w.DataPointer;
        for (long i = 0; i < (long)n * k; i++) p[i] = (sbyte)rng.Next(-127, 128);
        Tensor rowScale = new Tensor(new TensorShape(n), DType.F32);
        float* s = (float*)rowScale.DataPointer;
        for (int i = 0; i < n; i++) s[i] = (float)(0.002 + rng.NextDouble() * 0.004);
        w.QuantInfo = new QuantWeightInfo { Format = "int8_tensorwise", RowScale = rowScale, ConvRotGroupSize = group };
        return w;
    }

    private static Tensor RandomF16(Random rng, params long[] shape)
    {
        Tensor f32 = new Tensor(new TensorShape(shape), DType.F32);
        float* p = (float*)f32.DataPointer;
        for (long i = 0; i < f32.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2 - 1);
        Tensor f16 = f32.CastTo(DType.F16);
        f32.Dispose();
        return f16;
    }

    private static void AssertIdentical(ITestOutputHelper output, string label, Tensor grouped, Tensor perOp)
    {
        ReadOnlySpan<ushort> a = new ReadOnlySpan<ushort>((void*)grouped.DataPointer, (int)grouped.ElementCount);
        ReadOnlySpan<ushort> b = new ReadOnlySpan<ushort>((void*)perOp.DataPointer, (int)perOp.ElementCount);
        long diffs = 0;
        string first = "";
        for (int i = 0; i < a.Length; i++)
        {
            if (a[i] == b[i]) continue;
            if (diffs == 0)
            {
                int cols = (int)grouped.Shape[grouped.Shape.Rank - 1];
                first = $"  first at [{i / cols},{i % cols}]: 0x{a[i]:X4} vs 0x{b[i]:X4}"
                    + $" ({(float)BitConverter.UInt16BitsToHalf(a[i])} vs {(float)BitConverter.UInt16BitsToHalf(b[i])})";
            }
            diffs++;
        }
        output.WriteLine($"{label}: {diffs} of {a.Length} F16 words differ{first}");
        Assert.Equal(0, diffs);
    }

    /// <summary>The LTX-2.5 self-attention shape: gate (n=32) + q/k/v (n=4096) all projecting one [rows, 4096]
    /// activation, which is exactly the four-way redundancy the grouping exists to remove. <paramref name="rows"/>
    /// spans both sides of the 32-row cuBLASLt padding granularity.</summary>
    [Theory]
    [InlineData(97, 4096, 256)]
    [InlineData(37, 2048, 256)]
    [InlineData(8, 1024, 64)]
    public void GroupedMatchesPerOpBitExactly(int rows, int k, int group)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: no CUDA device"); return; }
        using CudaBackend backend = new CudaBackend(0, PtxDir());

        Random rng = new Random(20260813);
        using Tensor input = RandomF16(rng, rows, k);
        int inner = k;
        using Tensor gateW = Int8Weight(32, k, group, rng);
        using Tensor qW = Int8Weight(inner, k, group, rng);
        using Tensor kW = Int8Weight(inner, k, group, rng);
        using Tensor vW = Int8Weight(inner, k, group, rng);
        // Only q carries a bias, so a bias pointer crossed between targets shows up rather than cancelling.
        using Tensor qB = RandomF16(rng, inner);

        Tensor[] groupedOut =
        [
            new Tensor(new TensorShape(rows, 32), DType.F16),
            new Tensor(new TensorShape(rows, inner), DType.F16),
            new Tensor(new TensorShape(rows, inner), DType.F16),
            new Tensor(new TensorShape(rows, inner), DType.F16),
        ];
        Tensor[] perOpOut =
        [
            new Tensor(new TensorShape(rows, 32), DType.F16),
            new Tensor(new TensorShape(rows, inner), DType.F16),
            new Tensor(new TensorShape(rows, inner), DType.F16),
            new Tensor(new TensorShape(rows, inner), DType.F16),
        ];
        try
        {
            backend.LinearMulti(input, [
                new(groupedOut[0], gateW, null), new(groupedOut[1], qW, qB),
                new(groupedOut[2], kW, null), new(groupedOut[3], vW, null)]);
            backend.Linear(perOpOut[0], input, gateW, null);
            backend.Linear(perOpOut[1], input, qW, qB);
            backend.Linear(perOpOut[2], input, kW, null);
            backend.Linear(perOpOut[3], input, vW, null);
            backend.Sync();

            string[] names = ["gate", "q", "k", "v"];
            for (int i = 0; i < names.Length; i++) AssertIdentical(_output, names[i], groupedOut[i], perOpOut[i]);
        }
        finally
        {
            foreach (Tensor t in groupedOut) t.Dispose();
            foreach (Tensor t in perOpOut) t.Dispose();
        }
    }

    /// <summary>A group the resident chain cannot serve whole: an F16 weight and a mismatched ConvRot group sit
    /// beside eligible int8 weights. Both must fall out to an ordinary Linear and still land correct — a group is
    /// partitioned, never refused, and never silently quantized against the wrong rotation.</summary>
    [Fact]
    public void MixedGroupPartitionsInsteadOfRefusing()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: no CUDA device"); return; }
        using CudaBackend backend = new CudaBackend(0, PtxDir());

        const int Rows = 64, K = 2048, N = 512;
        Random rng = new Random(4242);
        using Tensor input = RandomF16(rng, Rows, K);
        using Tensor int8A = Int8Weight(N, K, 256, rng);
        using Tensor int8B = Int8Weight(N, K, 256, rng);
        using Tensor otherGroup = Int8Weight(N, K, 64, rng);   // valid, but not this group's rotation
        using Tensor f16W = RandomF16(rng, N, K);

        Tensor[] grouped = [.. Enumerable.Range(0, 4).Select(_ => new Tensor(new TensorShape(Rows, N), DType.F16))];
        Tensor[] perOp = [.. Enumerable.Range(0, 4).Select(_ => new Tensor(new TensorShape(Rows, N), DType.F16))];
        try
        {
            Tensor[] weights = [int8A, otherGroup, int8B, f16W];
            backend.LinearMulti(input, [
                new(grouped[0], weights[0], null), new(grouped[1], weights[1], null),
                new(grouped[2], weights[2], null), new(grouped[3], weights[3], null)]);
            for (int i = 0; i < 4; i++) backend.Linear(perOp[i], input, weights[i], null);
            backend.Sync();

            // The two same-group int8 weights are the ones that actually shared a quant pass, so they are the
            // bit-identical pair; the stragglers ran through the very same Linear on both sides.
            string[] names = ["int8A(grouped)", "otherGroup(straggler)", "int8B(grouped)", "f16(straggler)"];
            for (int i = 0; i < 4; i++) AssertIdentical(_output, names[i], grouped[i], perOp[i]);
        }
        finally
        {
            foreach (Tensor t in grouped) t.Dispose();
            foreach (Tensor t in perOp) t.Dispose();
        }
    }

    /// <summary>The fused mma GEMM+dequant path, exercised THROUGH <c>Linear</c> rather than through its launcher.
    /// It ships opt-in-off (it loses end-to-end), so without this nothing covers the route a caller actually takes
    /// — <c>Int8MmaGemmTests</c> calls the kernel directly and would not notice the eligibility gate, the row
    /// chunking, or the scale/bias plumbing around it silently breaking.</summary>
    /// <remarks>Shape chosen to clear every bound in <c>UseFusedMmaGemm</c>: rows ≥ 1024, n a multiple of 256,
    /// k ≤ 2n, F16 out. The kernel is bit-exact against the cuBLASLt pair, so the two routes must agree EXACTLY.</remarks>
    [Fact]
    public void FusedMmaPathMatchesCublasLtPairThroughLinear()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: no CUDA device"); return; }
        using CudaBackend backend = new CudaBackend(0, PtxDir());

        const int Rows = 1088, K = 1024, N = 512;
        Random rng = new Random(8675309);
        using Tensor input = RandomF16(rng, Rows, K);
        using Tensor w = Int8Weight(N, K, 256, rng);
        using Tensor b = RandomF16(rng, N);
        using Tensor fused = new Tensor(new TensorShape(Rows, N), DType.F16);
        using Tensor pair = new Tensor(new TensorShape(Rows, N), DType.F16);

        bool saved = CudaBackend.FusedMmaGemm;
        try
        {
            CudaBackend.FusedMmaGemm = true;
            backend.Linear(fused, input, w, b);
            CudaBackend.FusedMmaGemm = false;
            backend.Linear(pair, input, w, b);
            backend.Sync();
        }
        finally { CudaBackend.FusedMmaGemm = saved; }

        AssertIdentical(_output, "fused-vs-pair through Linear", fused, pair);
    }

    /// <summary>The kill switch has to be a real seam: with it off, LinearMulti must degrade to the per-op loop and
    /// still produce the same bytes, so a bisect can turn the feature off and trust the result.</summary>
    [Fact]
    public void KillSwitchFallsBackToPerOp()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: no CUDA device"); return; }
        using CudaBackend backend = new CudaBackend(0, PtxDir());

        const int Rows = 48, K = 1024, N = 256;
        Random rng = new Random(99);
        using Tensor input = RandomF16(rng, Rows, K);
        using Tensor wa = Int8Weight(N, K, 256, rng);
        using Tensor wb = Int8Weight(N, K, 256, rng);
        using Tensor on0 = new Tensor(new TensorShape(Rows, N), DType.F16);
        using Tensor on1 = new Tensor(new TensorShape(Rows, N), DType.F16);
        using Tensor off0 = new Tensor(new TensorShape(Rows, N), DType.F16);
        using Tensor off1 = new Tensor(new TensorShape(Rows, N), DType.F16);

        bool saved = CudaBackend.GroupedLinear;
        try
        {
            CudaBackend.GroupedLinear = true;
            backend.LinearMulti(input, [new(on0, wa, null), new(on1, wb, null)]);
            CudaBackend.GroupedLinear = false;
            backend.LinearMulti(input, [new(off0, wa, null), new(off1, wb, null)]);
            backend.Sync();
        }
        finally { CudaBackend.GroupedLinear = saved; }

        AssertIdentical(_output, "op0", on0, off0);
        AssertIdentical(_output, "op1", on1, off1);
    }
}
