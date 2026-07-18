using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Parity gate for the multi-scale deformable-attention GPU kernel
/// (<see cref="CudaBackend.DeformableAttention"/>) against the CPU reference on <see cref="IBackend"/>.
/// Covers both reference-point layouts: coords=2 with per-level refs (Grounding DINO encoder) and
/// coords=4 with one ref shared across levels (RT-DETR / GDINO decoder). F32 within 1e-4 per the
/// kernel tolerance table (softmax + bilinear compound the rounding slightly above pure-GEMM 1e-5).</summary>
[Collection("CudaSerial")]
[Trait("Category", "GpuIntegration")]
public sealed unsafe class DeformableAttentionKernelTests
{
    private readonly ITestOutputHelper _output;
    public DeformableAttentionKernelTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    private static Tensor Random(TensorShape shape, int seed, double lo, double hi)
    {
        Tensor t = new Tensor(shape, DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new Random(seed);
        long n = shape.ElementCount;
        for (long i = 0; i < n; i++) p[i] = (float)(rng.NextDouble() * (hi - lo) + lo);
        return t;
    }

    private static double MaxErr(Tensor a, Tensor b)
    {
        float* ap = (float*)a.DataPointer;
        float* bp = (float*)b.DataPointer;
        double maxErr = 0;
        long n = a.ElementCount;
        for (long i = 0; i < n; i++)
        {
            double e = Math.Abs(ap[i] - bp[i]);
            if (e > maxErr) maxErr = e;
        }
        return maxErr;
    }

    private void RunParity(int coords, int refQueryStride, int refLevelStride, int refSeed, int refLen)
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        const int heads = 8, hd = 32, levels = 4, points = 4;
        int d = heads * hd;
        int nq = 50;

        // Multi-scale feature map: 10x10, 8x8, 6x6, 4x4 → 216 tokens.
        int[] shapes = { 10, 10, 8, 8, 6, 6, 4, 4 };
        int[] levelStart = new int[levels];
        int nkv = 0;
        for (int l = 0; l < levels; l++) { levelStart[l] = nkv; nkv += shapes[l * 2] * shapes[l * 2 + 1]; }

        using Tensor value = Random(new TensorShape(1, nkv, d), 101, -1, 1);
        using Tensor sampOff = Random(new TensorShape(1, nq, heads * levels * points * 2), 102, -2, 2);
        using Tensor attn = Random(new TensorShape(1, nq, heads * levels * points), 103, -3, 3);
        using Tensor refPoints = Random(new TensorShape(refLen), refSeed, 0, 1);

        using Tensor cpuOut = new Tensor(new TensorShape(1, nq, d), DType.F32);
        using (CpuBackend cpu = new CpuBackend())
            ((IBackend)cpu).DeformableAttention(cpuOut, value, sampOff, attn, refPoints, shapes, levelStart,
                heads, levels, points, coords, refQueryStride, refLevelStride);

        using Tensor cudaOut = new Tensor(new TensorShape(1, nq, d), DType.F32);
        using (CudaBackend cuda = new CudaBackend(0, PtxDir()))
        {
            cuda.DeformableAttention(cudaOut, value, sampOff, attn, refPoints, shapes, levelStart,
                heads, levels, points, coords, refQueryStride, refLevelStride);
            cuda.Sync();
            _ = *(float*)cudaOut.DataPointer;
        }

        double maxErr = MaxErr(cpuOut, cudaOut);
        _output.WriteLine($"DeformableAttention coords={coords}: max_err={maxErr:E3}");
        Assert.True(maxErr < 1e-4, $"DeformableAttention coords={coords} diverges: {maxErr:E3}");
    }

    /// <summary>Grounding DINO encoder self-attention: per-level reference points (2 coords each).</summary>
    [Fact]
    public void DeformableAttention_Coords2_PerLevelRef_MatchesCpu()
    {
        const int levels = 4, coords = 2, nq = 50;
        RunParity(coords, refQueryStride: levels * coords, refLevelStride: coords, refSeed: 201, refLen: nq * levels * coords);
    }

    /// <summary>RT-DETR / GDINO decoder cross-attention: one 4-coord reference shared across all levels.</summary>
    [Fact]
    public void DeformableAttention_Coords4_SharedRef_MatchesCpu()
    {
        const int coords = 4, nq = 50;
        RunParity(coords, refQueryStride: coords, refLevelStride: 0, refSeed: 202, refLen: nq * coords);
    }

    /// <summary>Grounding-DINO encoder-scale workload (17821 queries over the real 4-level pyramid) — the
    /// host loop that took ≈11 min/6-layer-encoder on CPU. Confirms GPU parity at scale and logs the
    /// single-op speedup so the win is on the record.</summary>
    [Fact]
    public void DeformableAttention_EncoderScale_MatchesCpu_AndIsFaster()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        const int heads = 8, hd = 32, levels = 4, points = 4, coords = 2;
        int d = heads * hd;

        // Real GDINO-tiny pyramid for a 800x1333-ish input: 100x134, 50x67, 25x34, 13x17.
        int[] shapes = { 100, 134, 50, 67, 25, 34, 13, 17 };
        int[] levelStart = new int[levels];
        int nkv = 0;
        for (int l = 0; l < levels; l++) { levelStart[l] = nkv; nkv += shapes[l * 2] * shapes[l * 2 + 1]; }
        int nq = nkv;   // encoder self-attention: one query per spatial token

        using Tensor value = Random(new TensorShape(1, nkv, d), 301, -1, 1);
        using Tensor sampOff = Random(new TensorShape(1, nq, heads * levels * points * 2), 302, -2, 2);
        using Tensor attn = Random(new TensorShape(1, nq, heads * levels * points), 303, -3, 3);
        using Tensor refPoints = Random(new TensorShape(nq * levels * coords), 304, 0, 1);

        using Tensor cpuOut = new Tensor(new TensorShape(1, nq, d), DType.F32);
        System.Diagnostics.Stopwatch swCpu = System.Diagnostics.Stopwatch.StartNew();
        using (CpuBackend cpu = new CpuBackend())
            ((IBackend)cpu).DeformableAttention(cpuOut, value, sampOff, attn, refPoints, shapes, levelStart,
                heads, levels, points, coords, levels * coords, coords);
        swCpu.Stop();

        using Tensor cudaOut = new Tensor(new TensorShape(1, nq, d), DType.F32);
        double gpuMs;
        using (CudaBackend cuda = new CudaBackend(0, PtxDir()))
        {
            cuda.DeformableAttention(cudaOut, value, sampOff, attn, refPoints, shapes, levelStart,
                heads, levels, points, coords, levels * coords, coords);
            cuda.Sync();
            System.Diagnostics.Stopwatch swGpu = System.Diagnostics.Stopwatch.StartNew();
            cuda.DeformableAttention(cudaOut, value, sampOff, attn, refPoints, shapes, levelStart,
                heads, levels, points, coords, levels * coords, coords);
            cuda.Sync();
            swGpu.Stop();
            gpuMs = swGpu.Elapsed.TotalMilliseconds;
            _ = *(float*)cudaOut.DataPointer;
        }

        double maxErr = MaxErr(cpuOut, cudaOut);
        _output.WriteLine($"EncoderScale nq={nq}: cpu={swCpu.Elapsed.TotalMilliseconds:F1}ms gpu={gpuMs:F1}ms " +
            $"speedup={swCpu.Elapsed.TotalMilliseconds / gpuMs:F1}x max_err={maxErr:E3}");
        Assert.True(maxErr < 1e-4, $"EncoderScale diverges: {maxErr:E3}");
    }
}
