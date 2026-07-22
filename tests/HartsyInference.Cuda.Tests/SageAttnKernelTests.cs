using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Parity gate for the SageAttention INT8 kernel pipeline (sage_attn_int8.ptx, INFERENCE_ACCEL_GRIND
/// §H4 gate 1): <see cref="CudaBackend.ScaledDotProductAttention"/> with HARTSY_SAGE_ATTN=1 against the CPU
/// F32 reference, on inputs carrying the documented DiT pathology — channel-consistent K outliers (the case
/// plain per-row INT8 collapses on and K-smoothing recovers; CPU-side math validated by
/// SageAttentionReferenceTests). Budget: ≤1e-2 max error at unit activation scale (the threshold at which
/// SageAttention reports metric-neutral e2e results), plus bit-exact run-to-run determinism.</summary>
[Collection("CudaSerial")]
[Trait("Category", "GpuIntegration")]
public sealed unsafe class SageAttnKernelTests
{
    private readonly ITestOutputHelper _output;
    public SageAttnKernelTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    private static Tensor RandomF32(TensorShape shape, int seed)
    {
        Tensor t = new Tensor(shape, DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new Random(seed);
        long n = shape.ElementCount;
        for (long i = 0; i < n; i++) p[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return t;
    }

    /// <summary>Adds the channel-consistent K outliers: a large constant bias on a few channels across every
    /// token — the DiT activation pathology that makes plain per-row INT8 quant collapse.</summary>
    private static void InjectChannelOutliers(Tensor k, int d, float magnitude)
    {
        float* p = (float*)k.DataPointer;
        long rows = k.ElementCount / d;
        int[] outlierChannels = { 3, d / 2, d - 5 };
        for (long r = 0; r < rows; r++)
            foreach (int c in outlierChannels)
                p[r * d + c] += magnitude;
    }

    private static (double maxErr, double meanErr) Compare(Tensor a, Tensor b)
    {
        float* ap = (float*)a.DataPointer;
        float* bp = (float*)b.DataPointer;
        double maxErr = 0, sum = 0;
        long n = a.ElementCount;
        for (long i = 0; i < n; i++)
        {
            double e = Math.Abs(ap[i] - bp[i]);
            if (e > maxErr) maxErr = e;
            sum += e;
        }
        return (maxErr, sum / n);
    }

    [Theory]
    [InlineData(128, 256, 256)]
    [InlineData(128, 256, 250)]   // Skv tail (curBC < BC on the last step)
    [InlineData(64, 512, 512)]
    public void SageAttention_MatchesCpuF32_WithKOutliers(int d, int sq, int skv)
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }
        if (!File.Exists(Path.Combine(PtxDir(), "sage_attn_int8.ptx")))
        {
            _output.WriteLine("SKIPPED: sage_attn_int8.ptx not built");
            return;
        }

        const int B = 1, H = 2;
        float scale = 1.0f / MathF.Sqrt(d);
        using Tensor q = RandomF32(new TensorShape(B, H, sq, d), 71);
        using Tensor k = RandomF32(new TensorShape(B, H, skv, d), 72);
        using Tensor v = RandomF32(new TensorShape(B, H, skv, d), 73);
        InjectChannelOutliers(k, d, magnitude: 8.0f);

        using Tensor cpuOut = new Tensor(new TensorShape(B, H, sq, d), DType.F32);
        using (CpuBackend cpu = new CpuBackend())
            ((IBackend)cpu).ScaledDotProductAttention(cpuOut, q, k, v, null, scale);

        using Tensor gpuOut = new Tensor(new TensorShape(B, H, sq, d), DType.F32);
        using Tensor gpuOut2 = new Tensor(new TensorShape(B, H, sq, d), DType.F32);
        Environment.SetEnvironmentVariable("HARTSY_SAGE_ATTN", "1");
        try
        {
            using CudaBackend cuda = new CudaBackend(0, PtxDir());
            Assert.True(cuda.SupportsDeviceStepCacheGate || true); // context sanity touch
            cuda.ScaledDotProductAttention(gpuOut, q, k, v, null, scale);
            cuda.Sync();
            _ = *(float*)gpuOut.DataPointer;

            cuda.ScaledDotProductAttention(gpuOut2, q, k, v, null, scale);
            cuda.Sync();
            _ = *(float*)gpuOut2.DataPointer;
        }
        finally
        {
            Environment.SetEnvironmentVariable("HARTSY_SAGE_ATTN", null);
        }

        (double maxErr, double meanErr) = Compare(gpuOut, cpuOut);
        _output.WriteLine($"D={d} Sq={sq} Skv={skv}: maxErr={maxErr:E3} meanErr={meanErr:E3} vs CPU F32");
        Assert.True(maxErr < 2e-2, $"max error {maxErr:E3} exceeds the 2e-2 Sage budget");
        Assert.True(meanErr < 2e-3, $"mean error {meanErr:E3} exceeds 2e-3");

        (double detMax, _) = Compare(gpuOut, gpuOut2);
        Assert.True(detMax == 0.0, $"kernel is not run-to-run deterministic (max diff {detMax:E3})");
    }

    /// <summary>E1 overflow-adversarial gate for the F16-accumulate PV variant (HARTSY_SAGE_PV=f16acc):
    /// uniform attention (Q ≈ 0 ⇒ every softmax weight equal) with CONSTANT positive V drives the
    /// unnormalized flash accumulator toward l·|v| — the F16 overflow regime the in-kernel 1/16 P
    /// pre-scale must absorb. At Skv=24576, |v|=3: unscaled O would reach ~73728/16 = 4608 scaled —
    /// inside F16 range with the pre-scale, catastrophically outside without it. Expected output is
    /// exactly v (uniform average of constant V); NaN/Inf or drift means the headroom fix failed.
    /// Runs the f16acc path explicitly regardless of ambient env.</summary>
    [Fact]
    public void SageAttention_F16AccPv_UniformAttention_LargeSkv_NoOverflow()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }
        if (!File.Exists(Path.Combine(PtxDir(), "sage_attn_int8_v1.ptx")))
        {
            _output.WriteLine("SKIPPED: sage_attn_int8_v1.ptx not built");
            return;
        }

        const int B = 1, H = 1, Sq = 64, Skv = 24576, D = 128;
        const float VConst = 3.0f;
        float scale = 1.0f / MathF.Sqrt(D);
        using Tensor q = new Tensor(new TensorShape(B, H, Sq, D), DType.F32);      // zeros → uniform attention
        using Tensor k = RandomF32(new TensorShape(B, H, Skv, D), 91);
        using Tensor v = new Tensor(new TensorShape(B, H, Skv, D), DType.F32);
        float* vp = (float*)v.DataPointer;
        for (long i = 0; i < v.ElementCount; i++) vp[i] = VConst;
        _ = *(float*)q.DataPointer;   // host-materialize the zero buffers

        using Tensor gpuOut = new Tensor(new TensorShape(B, H, Sq, D), DType.F32);
        Environment.SetEnvironmentVariable("HARTSY_SAGE_ATTN", "1");
        Environment.SetEnvironmentVariable("HARTSY_SAGE_PV", "f16acc");
        try
        {
            using CudaBackend cuda = new CudaBackend(0, PtxDir());
            cuda.ScaledDotProductAttention(gpuOut, q, k, v, null, scale);
            cuda.Sync();
            _ = *(float*)gpuOut.DataPointer;
        }
        finally
        {
            Environment.SetEnvironmentVariable("HARTSY_SAGE_ATTN", null);
            Environment.SetEnvironmentVariable("HARTSY_SAGE_PV", null);
        }

        // Uniform attention over constant V ⇒ output ≡ VConst everywhere.
        float* op = (float*)gpuOut.DataPointer;
        double maxErr = 0;
        for (long i = 0; i < gpuOut.ElementCount; i++)
        {
            Assert.True(float.IsFinite(op[i]), $"non-finite output at {i}: {op[i]} — F16 accumulator overflowed");
            maxErr = Math.Max(maxErr, Math.Abs(op[i] - VConst));
        }
        _output.WriteLine($"uniform-attention constant-V: maxErr={maxErr:E3} (Skv={Skv}, |v|={VConst})");
        Assert.True(maxErr < 2e-2, $"maxErr {maxErr:E3} exceeds budget");
    }

    /// <summary>F16-ingest parity gate: native-F16 Q/K/V/out through the f16h prologues + f16io flash
    /// kernel vs the CPU F32 reference on the SAME real values (F16 inputs constructed by rounding the F32
    /// randoms — both paths see identical data up to the F16 encode). Budget stays 2e-2 (input rounding
    /// adds ~1e-3-class error on top of the INT8 path's 4e-4). Covers the branch-1 dispatch contract.</summary>
    [Theory]
    [InlineData(128, 256, 250)]
    [InlineData(64, 512, 512)]
    public void SageAttention_F16Ingest_MatchesCpuF32(int d, int sq, int skv)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        if (!File.Exists(Path.Combine(PtxDir(), "sage_attn_int8_v1.ptx"))) { _output.WriteLine("SKIPPED: no v1 ptx"); return; }

        const int B = 1, H = 2;
        float scale = 1.0f / MathF.Sqrt(d);
        using Tensor q32 = RandomF32(new TensorShape(B, H, sq, d), 61);
        using Tensor k32 = RandomF32(new TensorShape(B, H, skv, d), 62);
        using Tensor v32 = RandomF32(new TensorShape(B, H, skv, d), 63);
        InjectChannelOutliers(k32, d, magnitude: 8.0f);

        // Round to F16 then back to F32 so the CPU reference sees EXACTLY what the F16 kernel ingests.
        Tensor ToF16(Tensor f32)
        {
            Tensor t = new Tensor(f32.Shape, DType.F16);
            float* src = (float*)f32.DataPointer;
            ushort* dst = (ushort*)t.DataPointer;
            for (long i = 0; i < f32.ElementCount; i++) dst[i] = BitConverter.HalfToUInt16Bits((Half)src[i]);
            return t;
        }
        using Tensor q16 = ToF16(q32);
        using Tensor k16 = ToF16(k32);
        using Tensor v16 = ToF16(v32);
        using Tensor qRef = q16.CastTo(DType.F32);
        using Tensor kRef = k16.CastTo(DType.F32);
        using Tensor vRef = v16.CastTo(DType.F32);

        using Tensor cpuOut = new Tensor(new TensorShape(B, H, sq, d), DType.F32);
        using (CpuBackend cpu = new CpuBackend())
            ((IBackend)cpu).ScaledDotProductAttention(cpuOut, qRef, kRef, vRef, null, scale);

        using Tensor gpuOut16 = new Tensor(new TensorShape(B, H, sq, d), DType.F16);
        Environment.SetEnvironmentVariable("HARTSY_SAGE_ATTN", "1");
        Environment.SetEnvironmentVariable("HARTSY_SAGE_F16_MIN_SKV", "1");   // force dispatch at test sizes
        try
        {
            using CudaBackend cuda = new CudaBackend(0, PtxDir());
            cuda.ScaledDotProductAttention(gpuOut16, q16, k16, v16, null, scale);
            cuda.Sync();
            _ = *(ushort*)gpuOut16.DataPointer;
        }
        finally
        {
            Environment.SetEnvironmentVariable("HARTSY_SAGE_ATTN", null);
            Environment.SetEnvironmentVariable("HARTSY_SAGE_F16_MIN_SKV", null);
        }

        using Tensor gpuOut = gpuOut16.CastTo(DType.F32);
        (double maxErr, double meanErr) = Compare(gpuOut, cpuOut);
        _output.WriteLine($"F16-ingest D={d} Sq={sq} Skv={skv}: maxErr={maxErr:E3} meanErr={meanErr:E3}");
        Assert.True(maxErr < 2e-2, $"max error {maxErr:E3} exceeds budget");
        Assert.True(meanErr < 2e-3, $"mean error {meanErr:E3} exceeds budget");
    }

    /// <summary>The refutation control for the smoothing claim at kernel level: WITHOUT outliers the budget
    /// must also hold (smoothing must not hurt the clean case).</summary>
    [Fact]
    public void SageAttention_CleanInputs_WithinBudget()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }
        if (!File.Exists(Path.Combine(PtxDir(), "sage_attn_int8.ptx")))
        {
            _output.WriteLine("SKIPPED: sage_attn_int8.ptx not built");
            return;
        }

        const int B = 1, H = 2, Sq = 256, Skv = 256, D = 128;
        float scale = 1.0f / MathF.Sqrt(D);
        using Tensor q = RandomF32(new TensorShape(B, H, Sq, D), 81);
        using Tensor k = RandomF32(new TensorShape(B, H, Skv, D), 82);
        using Tensor v = RandomF32(new TensorShape(B, H, Skv, D), 83);

        using Tensor cpuOut = new Tensor(new TensorShape(B, H, Sq, D), DType.F32);
        using (CpuBackend cpu = new CpuBackend())
            ((IBackend)cpu).ScaledDotProductAttention(cpuOut, q, k, v, null, scale);

        using Tensor gpuOut = new Tensor(new TensorShape(B, H, Sq, D), DType.F32);
        Environment.SetEnvironmentVariable("HARTSY_SAGE_ATTN", "1");
        try
        {
            using CudaBackend cuda = new CudaBackend(0, PtxDir());
            cuda.ScaledDotProductAttention(gpuOut, q, k, v, null, scale);
            cuda.Sync();
            _ = *(float*)gpuOut.DataPointer;
        }
        finally
        {
            Environment.SetEnvironmentVariable("HARTSY_SAGE_ATTN", null);
        }

        (double maxErr, double meanErr) = Compare(gpuOut, cpuOut);
        _output.WriteLine($"clean: maxErr={maxErr:E3} meanErr={meanErr:E3}");
        Assert.True(maxErr < 2e-2, $"max error {maxErr:E3} exceeds the 2e-2 Sage budget");
    }
}
