using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Projecting two rows in one call must equal projecting them one at a time. The dual-stream decode step
/// batches its QKV projection into a single 2-row GEMM while the eager path does two 1-row ones, so any
/// precision difference between those shapes lands directly on <c>GraphDecodeDualEmbedsTests</c>. cuBLAS selects
/// per shape and per architecture, which is exactly the kind of difference that shows on one card and not another.</summary>
[Collection("CudaSerial")]
public sealed unsafe class BatchedGemmPrecisionProbe
{
    private readonly ITestOutputHelper _output;
    public BatchedGemmPrecisionProbe(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    private static uint _rng = 0x1234567u;
    private static float Rand()
    {
        _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5;
        return (_rng & 0xFFFF) / 65535f - 0.5f;
    }

    private static Tensor Rnd(params long[] dims)
    {
        Tensor t = new(new TensorShape(dims), DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) p[i] = Rand();
        return t;
    }

    [Theory]
    [InlineData(32, 32, false)]      // GraphDecodeDualEmbedsTests' geometry
    [InlineData(4096, 4096, false)]  // MiniMax Music 3's real projection width
    [InlineData(32, 32, true)]
    [InlineData(4096, 4096, true)]
    public void TwoRowsInOneCall_EqualsTwoSingleRowCalls(int inDim, int outDim, bool highPrecision)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        using CudaBackend backend = new(0, PtxDir());
        backend.HighPrecisionGemm = highPrecision;
        IBackend b = backend;

        using Tensor weight = Rnd(outDim, inDim);
        using Tensor rows = Rnd(1, 2, inDim);
        using Tensor row0 = Rnd(1, 1, inDim);
        using Tensor row1 = Rnd(1, 1, inDim);
        float* src = (float*)rows.DataPointer;
        Buffer.MemoryCopy(src, (void*)row0.DataPointer, (long)inDim * 4, (long)inDim * 4);
        Buffer.MemoryCopy(src + inDim, (void*)row1.DataPointer, (long)inDim * 4, (long)inDim * 4);

        using Tensor batched = new(new TensorShape(1, 2, outDim), DType.F32);
        using Tensor single0 = new(new TensorShape(1, 1, outDim), DType.F32);
        using Tensor single1 = new(new TensorShape(1, 1, outDim), DType.F32);
        b.Linear(batched, rows, weight, null);
        b.Linear(single0, row0, weight, null);
        b.Linear(single1, row1, weight, null);
        backend.Sync();

        float* pb = (float*)batched.DataPointer;
        float* p0 = (float*)single0.DataPointer;
        float* p1 = (float*)single1.DataPointer;
        float max0 = 0f, max1 = 0f, peak = 0f;
        for (int i = 0; i < outDim; i++)
        {
            max0 = MathF.Max(max0, MathF.Abs(pb[i] - p0[i]));
            max1 = MathF.Max(max1, MathF.Abs(pb[outDim + i] - p1[i]));
            peak = MathF.Max(peak, MathF.Abs(p0[i]));
        }
        float rel = MathF.Max(max0, max1) / MathF.Max(peak, 1e-9f);
        _output.WriteLine($"in={inDim} out={outDim} highPrecision={highPrecision}: row0 {max0:E3}, row1 {max1:E3}, "
            + $"peak {peak:E3} (relative {rel:E3})");

        // Under full precision the two shapes must agree to F32 rounding. Under the default they need not, and on
        // Ada they do not: TF32's 10-bit mantissa costs ~1e-4 relative even at 32 wide, while a 3060 declines
        // tensor cores at that size and looks exact. That difference is a cuBLAS heuristic, so nothing may assume
        // batched and single-row projections agree bit-for-bit at default precision.
        float tol = highPrecision ? 1e-5f : 2e-3f;
        Assert.True(rel <= tol,
            $"a 2-row projection differs from two 1-row projections by {rel:E3} relative "
            + $"(highPrecision={highPrecision}) — beyond what the selected compute type explains.");
    }
}
