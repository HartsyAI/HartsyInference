using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Equivalence of <see cref="IBackend.ApplyRopeSingleHeadMajor"/> with the composed
/// <see cref="IBackend.ApplyRopeSingle"/> + <see cref="IBackend.Permute0213"/> it replaces. Both run the same
/// rotation in the same order — only the vec→cos/sin-row map differs — so the assertion is BIT-EXACT (any
/// difference is an addressing bug, not rounding). Covers the CPU host impl (Unit tier) and the CUDA F32
/// kernel; the GPU cases skip cleanly when CUDA is unavailable.</summary>
[Collection("CudaSerial")]
public sealed unsafe class RopeHeadMajorTests
{
    private readonly ITestOutputHelper _output;
    public RopeHeadMajorTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    private static Tensor Random(TensorShape shape, int seed)
    {
        Tensor t = new Tensor(shape, DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new Random(seed);
        long n = shape.ElementCount;
        for (long i = 0; i < n; i++) p[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return t;
    }

    private void AssertBitExact(Tensor expected, Tensor actual, string name)
    {
        long n = expected.ElementCount;
        Assert.Equal(n, actual.ElementCount);
        long mismatches = 0, firstBad = -1;
        float* a = (float*)expected.DataPointer, b = (float*)actual.DataPointer;
        for (long i = 0; i < n; i++)
            if (BitConverter.SingleToInt32Bits(a[i]) != BitConverter.SingleToInt32Bits(b[i]))
            { mismatches++; if (firstBad < 0) firstBad = i; }
        _output.WriteLine($"{name}: {n - mismatches}/{n} bit-exact");
        Assert.True(mismatches == 0, $"{name}: {mismatches} of {n} elements differ, first at index {firstBad}");
    }

    /// <summary>batch, heads, seq, headDim, rotaryDim. batch &gt; 1 exercises the head-major batch divisor (with
    /// B=1 a wrong divisor is invisible); heads/seq/headDim are pairwise distinct so a transposed index cannot
    /// hide; rotaryDim &lt; headDim proves the tail passes through; rotaryDim 0 / == headDim cover full rotary
    /// (a launcher that forgot to clamp rotaryDim would size the grid at 0 and silently rotate nothing).</summary>
    public static TheoryData<int, int, int, int, int> Shapes() => new()
    {
        { 2, 3, 5, 8, 6 },
        { 1, 3, 5, 8, 6 },
        { 2, 4, 7, 8, 0 },
        { 2, 4, 7, 8, 8 },
        { 2, 6, 13, 128, 96 },
    };

    [Theory]
    [MemberData(nameof(Shapes))]
    public void HostImpl_MatchesRopeThenPermute(int b, int heads, int seq, int headDim, int rotaryDim)
    {
        IBackend cpu = new CpuBackend();
        try
        {
            RunCase(cpu, b, heads, seq, headDim, rotaryDim, $"cpu[{b},{heads},{seq},{headDim}] rot={rotaryDim}", null);
        }
        finally
        {
            cpu.Dispose();
        }
    }

    [Theory]
    [MemberData(nameof(Shapes))]
    [Trait("Category", "GpuIntegration")]
    public void CudaF32_MatchesRopeThenPermute(int b, int heads, int seq, int headDim, int rotaryDim)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        RunCase(cuda, b, heads, seq, headDim, rotaryDim, $"cuda[{b},{heads},{seq},{headDim}] rot={rotaryDim}", cuda);
    }

    /// <summary>Both paths share one input: the head-major copy is taken BEFORE the token-major rope runs, since
    /// rope is in-place and would otherwise leave the second path rotating already-rotated values.</summary>
    private void RunCase(IBackend backend, int b, int heads, int seq, int headDim, int rotaryDim, string tag, CudaBackend? cuda)
    {
        using Tensor xTok = Random(new TensorShape(b, seq, heads, headDim), seed: 7 + headDim + rotaryDim);
        using Tensor cos = Random(new TensorShape(b, seq, headDim), seed: 13);
        using Tensor sin = Random(new TensorShape(b, seq, headDim), seed: 29);
        using Tensor xHeadMajor = new Tensor(new TensorShape(b, heads, seq, headDim), DType.F32);
        using Tensor reference = new Tensor(new TensorShape(b, heads, seq, headDim), DType.F32);

        backend.Permute0213(xHeadMajor, xTok, seq, heads, headDim);
        backend.ApplyRopeSingle(xTok, cos, sin, rotaryDim);
        backend.Permute0213(reference, xTok, seq, heads, headDim);
        backend.ApplyRopeSingleHeadMajor(xHeadMajor, cos, sin, rotaryDim);

        if (cuda is not null)
        {
            cuda.Sync();
            _ = *(byte*)reference.DataPointer;      // forces the lazy D2H while the context is alive
            _ = *(byte*)xHeadMajor.DataPointer;
        }
        AssertBitExact(reference, xHeadMajor, tag);
    }

    [Theory]
    [MemberData(nameof(Shapes))]
    public void HostImpl_ChainedQkvSplitAndRope_MatchesTokenMajorChain(int b, int heads, int seq, int headDim, int rotaryDim)
    {
        IBackend cpu = new CpuBackend();
        try
        {
            RunChainedCase(cpu, b, heads, seq, headDim, rotaryDim, $"cpu[{b},{heads},{seq},{headDim}] rot={rotaryDim}", null);
        }
        finally
        {
            cpu.Dispose();
        }
    }

    [Theory]
    [MemberData(nameof(Shapes))]
    [Trait("Category", "GpuIntegration")]
    public void CudaF32_ChainedQkvSplitAndRope_MatchesTokenMajorChain(int b, int heads, int seq, int headDim, int rotaryDim)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        RunChainedCase(cuda, b, heads, seq, headDim, rotaryDim, $"cuda[{b},{heads},{seq},{headDim}] rot={rotaryDim}", cuda);
    }

    /// <summary>The whole attention prologue the head-major pair replaces, so the in-place rope runs against a
    /// tensor another head-major op cached (the activation-callback reset that isolated cases never exercise).</summary>
    private void RunChainedCase(IBackend backend, int b, int heads, int seq, int headDim, int rotaryDim, string tag, CudaBackend? cuda)
    {
        int w = heads * headDim;
        using Tensor qkv = Random(new TensorShape(b * seq, 3 * w), seed: 101 + headDim);
        using Tensor qW = Random(new TensorShape(headDim), seed: 103);
        using Tensor kW = Random(new TensorShape(headDim), seed: 107);
        using Tensor cos = Random(new TensorShape(b, seq, headDim), seed: 109);
        using Tensor sin = Random(new TensorShape(b, seq, headDim), seed: 113);

        TensorShape tokenMajor = new TensorShape(b, seq, heads, headDim);
        TensorShape headMajor = new TensorShape(b, heads, seq, headDim);
        using Tensor qTok = new Tensor(tokenMajor, DType.F32);
        using Tensor kTok = new Tensor(tokenMajor, DType.F32);
        using Tensor vTok = new Tensor(tokenMajor, DType.F32);
        using Tensor qRef = new Tensor(headMajor, DType.F32);
        using Tensor kRef = new Tensor(headMajor, DType.F32);
        using Tensor vRef = new Tensor(headMajor, DType.F32);
        using Tensor qHm = new Tensor(headMajor, DType.F32);
        using Tensor kHm = new Tensor(headMajor, DType.F32);
        using Tensor vHm = new Tensor(headMajor, DType.F32);

        backend.QkvSplitNorm(qTok, kTok, vTok, qkv, qW, kW, 1e-6f);
        backend.ApplyRopeSingle(qTok, cos, sin, rotaryDim);
        backend.ApplyRopeSingle(kTok, cos, sin, rotaryDim);
        backend.Permute0213(qRef, qTok, seq, heads, headDim);
        backend.Permute0213(kRef, kTok, seq, heads, headDim);
        backend.Permute0213(vRef, vTok, seq, heads, headDim);

        backend.QkvSplitNormHeadMajor(qHm, kHm, vHm, qkv, qW, kW, 1e-6f);
        backend.ApplyRopeSingleHeadMajor(qHm, cos, sin, rotaryDim);
        backend.ApplyRopeSingleHeadMajor(kHm, cos, sin, rotaryDim);

        if (cuda is not null)
        {
            cuda.Sync();
            foreach (Tensor t in new[] { qRef, kRef, vRef, qHm, kHm, vHm })
                _ = *(byte*)t.DataPointer;          // forces the lazy D2H while the context is alive
        }
        AssertBitExact(qRef, qHm, $"chained {tag} q");
        AssertBitExact(kRef, kHm, $"chained {tag} k");
        AssertBitExact(vRef, vHm, $"chained {tag} v");
    }
}
