using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Checks QKV split/norm layouts and numerics against an independent scalar RMSNorm oracle.</summary>
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

    private static float ReadValue(Tensor tensor, long index)
    {
        return tensor.DType == DType.F32
            ? ((float*)tensor.DataPointer)[index]
            : (float)((Half*)tensor.DataPointer)[index];
    }

    private static void WriteValue(Tensor tensor, long index, float value)
    {
        if (tensor.DType == DType.F32)
            ((float*)tensor.DataPointer)[index] = value;
        else
            ((Half*)tensor.DataPointer)[index] = (Half)value;
    }

    private static void FillOracle(
        Tensor q,
        Tensor k,
        Tensor v,
        Tensor qkv,
        Tensor qWeight,
        Tensor kWeight,
        int batch,
        int heads,
        int seq,
        int headDim,
        bool headMajor,
        float eps)
    {
        int width = heads * headDim;
        int tokens = batch * seq;
        float* qWeightPointer = (float*)qWeight.DataPointer;
        float* kWeightPointer = (float*)kWeight.DataPointer;
        for (int token = 0; token < tokens; token++)
        {
            int batchIndex = token / seq;
            int sequenceIndex = token % seq;
            long inputTokenOffset = (long)token * 3 * width;
            for (int head = 0; head < heads; head++)
            {
                long qInputOffset = inputTokenOffset + (long)head * headDim;
                long kInputOffset = inputTokenOffset + width + (long)head * headDim;
                long vInputOffset = inputTokenOffset + 2L * width + (long)head * headDim;
                double qSquares = 0.0;
                double kSquares = 0.0;
                for (int d = 0; d < headDim; d++)
                {
                    double qValue = ReadValue(qkv, qInputOffset + d);
                    double kValue = ReadValue(qkv, kInputOffset + d);
                    qSquares += qValue * qValue;
                    kSquares += kValue * kValue;
                }

                float qInverseRms = (float)(1.0 / Math.Sqrt(qSquares / headDim + eps));
                float kInverseRms = (float)(1.0 / Math.Sqrt(kSquares / headDim + eps));
                long outputOffset = headMajor
                    ? ((long)batchIndex * heads * seq + (long)head * seq + sequenceIndex) * headDim
                    : (long)token * width + (long)head * headDim;
                for (int d = 0; d < headDim; d++)
                {
                    WriteValue(q, outputOffset + d, ReadValue(qkv, qInputOffset + d) * qInverseRms * qWeightPointer[d]);
                    WriteValue(k, outputOffset + d, ReadValue(qkv, kInputOffset + d) * kInverseRms * kWeightPointer[d]);
                    WriteValue(v, outputOffset + d, ReadValue(qkv, vInputOffset + d));
                }
            }
        }
    }

    private void AssertClose(Tensor expected, Tensor actual, float absoluteTolerance, float relativeTolerance, string name)
    {
        Assert.Equal(expected.DType, actual.DType);
        Assert.Equal(expected.ElementCount, actual.ElementCount);
        long mismatches = 0;
        long firstBad = -1;
        float largestError = 0f;
        float firstExpected = 0f;
        float firstActual = 0f;
        for (long i = 0; i < expected.ElementCount; i++)
        {
            float expectedValue = ReadValue(expected, i);
            float actualValue = ReadValue(actual, i);
            float error = MathF.Abs(expectedValue - actualValue);
            float tolerance = absoluteTolerance + relativeTolerance * MathF.Abs(expectedValue);
            if (!float.IsFinite(actualValue) || error > tolerance)
            {
                mismatches++;
                if (firstBad < 0)
                {
                    firstBad = i;
                    firstExpected = expectedValue;
                    firstActual = actualValue;
                }
            }
            largestError = MathF.Max(largestError, error);
        }

        _output.WriteLine($"{name}: max absolute error {largestError:G9}");
        Assert.True(mismatches == 0,
            $"{name}: {mismatches} values exceed tolerance, first at index {firstBad}: expected {firstExpected:G9}, actual {firstActual:G9}");
    }

    /// <summary>Shapes deliberately have batch &gt; 1 and heads != seq so a transposed store cannot hide.</summary>
    public static TheoryData<int, int, int, int> Shapes() => new()
    {
        { 2, 3, 5, 17 },
        { 2, 3, 5, 64 },
        { 2, 5, 7, 96 },
        { 1, 4, 7, 128 },
        { 1, 2, 3, 257 },
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

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void Cuda_RejectsInvalidDtypesAndShapesBeforeLaunch()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        using Tensor qkv = Random(new TensorShape(2, 3 * 3 * 64), seed: 71);
        using Tensor qWeight = Random(new TensorShape(64), seed: 72);
        using Tensor kWeight = Random(new TensorShape(64), seed: 73);
        using Tensor q = new Tensor(new TensorShape(2, 3, 64), DType.F32);
        using Tensor k = new Tensor(q.Shape, DType.F32);
        using Tensor v = new Tensor(q.Shape, DType.F32);
        using Tensor qF16 = new Tensor(q.Shape, DType.F16);
        using Tensor malformedQkv = Random(new TensorShape(2, 3 * 3 * 64 - 1), seed: 74);
        using Tensor indivisibleWidthQkv = Random(new TensorShape(2, 3 * 193), seed: 75);
        using Tensor wrongOutput = new Tensor(new TensorShape(2, 2, 64), DType.F32);
        using Tensor wrongWeightDtype = RandomF16(new TensorShape(64), seed: 76);

        Assert.Throws<NotSupportedException>(() => cuda.QkvSplitNorm(qF16, k, v, qkv, qWeight, kWeight, 1e-6f));
        Assert.Throws<HartsyInferenceException>(() => cuda.QkvSplitNorm(q, k, v, malformedQkv, qWeight, kWeight, 1e-6f));
        Assert.Throws<HartsyInferenceException>(() => cuda.QkvSplitNorm(q, k, v, indivisibleWidthQkv, qWeight, kWeight, 1e-6f));
        Assert.Throws<HartsyInferenceException>(() => cuda.QkvSplitNorm(wrongOutput, k, v, qkv, qWeight, kWeight, 1e-6f));
        Assert.Throws<NotSupportedException>(() => cuda.QkvSplitNorm(q, k, v, qkv, wrongWeightDtype, kWeight, 1e-6f));
        Assert.Throws<HartsyInferenceException>(() => cuda.QkvSplitNorm(q, k, v, qkv, qWeight, kWeight, 0f));
    }

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void Cuda_RejectsInvalidHeadMajorLayoutBeforeLaunch()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        using Tensor qkv = Random(new TensorShape(10, 3 * 3 * 64), seed: 81);
        using Tensor qWeight = Random(new TensorShape(64), seed: 82);
        using Tensor kWeight = Random(new TensorShape(64), seed: 83);
        using Tensor q = new Tensor(new TensorShape(2, 5, 3, 64), DType.F32);
        using Tensor k = new Tensor(q.Shape, DType.F32);
        using Tensor v = new Tensor(q.Shape, DType.F32);

        Assert.Throws<HartsyInferenceException>(() =>
            cuda.QkvSplitNormHeadMajor(q, k, v, qkv, qWeight, kWeight, 1e-6f));
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
        using Tensor qTokenOracle = new Tensor(tokenMajor, dtype);
        using Tensor kTokenOracle = new Tensor(tokenMajor, dtype);
        using Tensor vTokenOracle = new Tensor(tokenMajor, dtype);
        using Tensor qHeadOracle = new Tensor(headMajor, dtype);
        using Tensor kHeadOracle = new Tensor(headMajor, dtype);
        using Tensor vHeadOracle = new Tensor(headMajor, dtype);
        FillOracle(qTokenOracle, kTokenOracle, vTokenOracle, qkv, qW, kW, b, heads, seq, headDim,
            headMajor: false, eps: 1e-6f);
        FillOracle(qHeadOracle, kHeadOracle, vHeadOracle, qkv, qW, kW, b, heads, seq, headDim,
            headMajor: true, eps: 1e-6f);

        using (CudaBackend cuda = new CudaBackend(0, PtxDir()))
        {
            IBackend gpu = cuda;
            gpu.QkvSplitNorm(qTok, kTok, vTok, qkv, qW, kW, 1e-6f);
            gpu.Permute0213(qRef, qTok, seq, heads, headDim);
            gpu.Permute0213(kRef, kTok, seq, heads, headDim);
            gpu.Permute0213(vRef, vTok, seq, heads, headDim);
            gpu.QkvSplitNormHeadMajor(qHm, kHm, vHm, qkv, qW, kW, 1e-6f);
            cuda.Sync();
            foreach (Tensor t in new[] { qTok, kTok, vTok, qRef, kRef, vRef, qHm, kHm, vHm })
                _ = *(byte*)t.DataPointer;   // forces the lazy D2H while the context is alive
        }

        string tag = $"cuda-{dtype} [{b},{heads},{seq},{headDim}]";
        AssertBitExact(qRef, qHm, $"{tag} q");
        AssertBitExact(kRef, kHm, $"{tag} k");
        AssertBitExact(vRef, vHm, $"{tag} v");
        float tolerance = dtype == DType.F32 ? 1e-5f : 1e-3f;
        AssertClose(qTokenOracle, qTok, tolerance, tolerance, $"{tag} token-major q oracle");
        AssertClose(kTokenOracle, kTok, tolerance, tolerance, $"{tag} token-major k oracle");
        AssertClose(vTokenOracle, vTok, 0f, 0f, $"{tag} token-major v oracle");
        AssertClose(qHeadOracle, qHm, tolerance, tolerance, $"{tag} head-major q oracle");
        AssertClose(kHeadOracle, kHm, tolerance, tolerance, $"{tag} head-major k oracle");
        AssertClose(vHeadOracle, vHm, 0f, 0f, $"{tag} head-major v oracle");
    }
}
