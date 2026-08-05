using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Equivalence of <see cref="IBackend.QkvSplitNormHeadMajor"/> with the composed
/// <see cref="IBackend.QkvSplitNorm"/> + <see cref="IBackend.Permute0213"/> it replaces. Both do the same
/// arithmetic in the same order — only the store address differs — so the assertion is BIT-EXACT (any
/// difference is an addressing bug, not rounding). Covers the CPU host impl (Unit tier) and the CUDA F32 +
/// F16 kernels; the GPU cases skip cleanly when CUDA is unavailable.</summary>
[Collection("CudaSerial")]
public sealed unsafe class QkvSplitNormHeadMajorTests
{
    private readonly ITestOutputHelper _output;
    public QkvSplitNormHeadMajorTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    private static Tensor Random(TensorShape shape, int seed, float lo = -1f, float hi = 1f)
    {
        Tensor t = new Tensor(shape, DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new Random(seed);
        long n = shape.ElementCount;
        for (long i = 0; i < n; i++) p[i] = (float)(rng.NextDouble() * (hi - lo) + lo);
        return t;
    }

    private static Tensor RandomF16(TensorShape shape, int seed, float lo = -1f, float hi = 1f)
    {
        Tensor t = new Tensor(shape, DType.F16);
        Half* p = (Half*)t.DataPointer;
        Random rng = new Random(seed);
        long n = shape.ElementCount;
        for (long i = 0; i < n; i++) p[i] = (Half)(float)(rng.NextDouble() * (hi - lo) + lo);
        return t;
    }

    private void AssertBitExact(Tensor expected, Tensor actual, string name)
    {
        long n = expected.ElementCount;
        Assert.Equal(n, actual.ElementCount);
        long mismatches = 0, firstBad = -1;
        if (expected.DType == DType.F32)
        {
            float* a = (float*)expected.DataPointer, b = (float*)actual.DataPointer;
            for (long i = 0; i < n; i++)
                if (BitConverter.SingleToInt32Bits(a[i]) != BitConverter.SingleToInt32Bits(b[i]))
                { mismatches++; if (firstBad < 0) firstBad = i; }
        }
        else
        {
            Half* a = (Half*)expected.DataPointer, b = (Half*)actual.DataPointer;
            for (long i = 0; i < n; i++)
                if (BitConverter.HalfToInt16Bits(a[i]) != BitConverter.HalfToInt16Bits(b[i]))
                { mismatches++; if (firstBad < 0) firstBad = i; }
        }
        _output.WriteLine($"{name}: {n - mismatches}/{n} bit-exact");
        Assert.True(mismatches == 0, $"{name}: {mismatches} of {n} elements differ, first at index {firstBad}");
    }

    /// <summary>Shapes deliberately have batch &gt; 1 and heads != seq so a transposed store cannot hide.</summary>
    public static TheoryData<int, int, int, int> Shapes() => new()
    {
        { 2, 3, 5, 64 },
        { 1, 4, 7, 128 },
        { 3, 8, 16, 64 },
    };

    [Theory]
    [MemberData(nameof(Shapes))]
    public void HostImpl_MatchesSplitNormThenPermute(int b, int heads, int seq, int headDim)
    {
        int w = heads * headDim;
        using Tensor qkv = Random(new TensorShape(b * seq, 3 * w), seed: 11 + headDim);
        using Tensor qW = Random(new TensorShape(headDim), seed: 22, lo: 0.5f, hi: 1.5f);
        using Tensor kW = Random(new TensorShape(headDim), seed: 33, lo: 0.5f, hi: 1.5f);

        using Tensor qTok = new Tensor(new TensorShape(b, seq, heads, headDim), DType.F32);
        using Tensor kTok = new Tensor(new TensorShape(b, seq, heads, headDim), DType.F32);
        using Tensor vTok = new Tensor(new TensorShape(b, seq, heads, headDim), DType.F32);
        using Tensor qRef = new Tensor(new TensorShape(b, heads, seq, headDim), DType.F32);
        using Tensor kRef = new Tensor(new TensorShape(b, heads, seq, headDim), DType.F32);
        using Tensor vRef = new Tensor(new TensorShape(b, heads, seq, headDim), DType.F32);
        using Tensor qHm = new Tensor(new TensorShape(b, heads, seq, headDim), DType.F32);
        using Tensor kHm = new Tensor(new TensorShape(b, heads, seq, headDim), DType.F32);
        using Tensor vHm = new Tensor(new TensorShape(b, heads, seq, headDim), DType.F32);

        IBackend cpu = new CpuBackend();
        cpu.QkvSplitNorm(qTok, kTok, vTok, qkv, qW, kW, 1e-6f);
        cpu.Permute0213(qRef, qTok, seq, heads, headDim);
        cpu.Permute0213(kRef, kTok, seq, heads, headDim);
        cpu.Permute0213(vRef, vTok, seq, heads, headDim);
        cpu.QkvSplitNormHeadMajor(qHm, kHm, vHm, qkv, qW, kW, 1e-6f);
        cpu.Dispose();

        AssertBitExact(qRef, qHm, $"cpu q[{b},{heads},{seq},{headDim}]");
        AssertBitExact(kRef, kHm, $"cpu k[{b},{heads},{seq},{headDim}]");
        AssertBitExact(vRef, vHm, $"cpu v[{b},{heads},{seq},{headDim}]");
    }

    [Theory]
    [MemberData(nameof(Shapes))]
    [Trait("Category", "GpuIntegration")]
    public void CudaF32_MatchesSplitNormThenPermute(int b, int heads, int seq, int headDim)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        RunCudaCase(b, heads, seq, headDim, DType.F32);
    }

    [Theory]
    [MemberData(nameof(Shapes))]
    [Trait("Category", "GpuIntegration")]
    public void CudaF16_MatchesSplitNormThenPermute(int b, int heads, int seq, int headDim)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        RunCudaCase(b, heads, seq, headDim, DType.F16);
    }

    private void RunCudaCase(int b, int heads, int seq, int headDim, DType dtype)
    {
        int w = heads * headDim;
        TensorShape tokenMajor = new TensorShape(b, seq, heads, headDim);
        TensorShape headMajor = new TensorShape(b, heads, seq, headDim);
        using Tensor qkv = dtype == DType.F16
            ? RandomF16(new TensorShape(b * seq, 3 * w), seed: 44 + headDim)
            : Random(new TensorShape(b * seq, 3 * w), seed: 44 + headDim);
        using Tensor qW = Random(new TensorShape(headDim), seed: 55, lo: 0.5f, hi: 1.5f);
        using Tensor kW = Random(new TensorShape(headDim), seed: 66, lo: 0.5f, hi: 1.5f);

        using Tensor qTok = new Tensor(tokenMajor, dtype);
        using Tensor kTok = new Tensor(tokenMajor, dtype);
        using Tensor vTok = new Tensor(tokenMajor, dtype);
        using Tensor qRef = new Tensor(headMajor, dtype);
        using Tensor kRef = new Tensor(headMajor, dtype);
        using Tensor vRef = new Tensor(headMajor, dtype);
        using Tensor qHm = new Tensor(headMajor, dtype);
        using Tensor kHm = new Tensor(headMajor, dtype);
        using Tensor vHm = new Tensor(headMajor, dtype);

        using (CudaBackend cuda = new CudaBackend(0, PtxDir()))
        {
            IBackend gpu = cuda;
            gpu.QkvSplitNorm(qTok, kTok, vTok, qkv, qW, kW, 1e-6f);
            gpu.Permute0213(qRef, qTok, seq, heads, headDim);
            gpu.Permute0213(kRef, kTok, seq, heads, headDim);
            gpu.Permute0213(vRef, vTok, seq, heads, headDim);
            gpu.QkvSplitNormHeadMajor(qHm, kHm, vHm, qkv, qW, kW, 1e-6f);
            cuda.Sync();
            foreach (Tensor t in new[] { qRef, kRef, vRef, qHm, kHm, vHm })
                _ = *(byte*)t.DataPointer;   // forces the lazy D2H while the context is alive
        }

        string tag = $"cuda-{dtype} [{b},{heads},{seq},{headDim}]";
        AssertBitExact(qRef, qHm, $"{tag} q");
        AssertBitExact(kRef, kHm, $"{tag} k");
        AssertBitExact(vRef, vHm, $"{tag} v");
    }
}
