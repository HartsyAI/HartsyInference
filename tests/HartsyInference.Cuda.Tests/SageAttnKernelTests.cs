using HartsyInference.Core.Configuration;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Parity gate for the SageAttention INT8 pipeline (sage_attn_int8.ptx): native-F16
/// <see cref="CudaBackend.ScaledDotProductAttention"/> against the CPU F32 reference on the same F16-rounded values,
/// on inputs carrying the documented DiT pathology — channel-consistent K outliers (the case plain per-row INT8 collapses
/// on and K-smoothing recovers; CPU-side math validated by SageAttentionReferenceTests). Budget: ≤2e-2 max error at unit
/// activation scale, plus bit-exact run-to-run determinism.</summary>
[Collection("CudaSerial")]
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

    private bool Skip()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return true; }
        if (!File.Exists(Path.Combine(PtxDir(), "sage_attn_int8.ptx"))) { _output.WriteLine("SKIPPED: sage_attn_int8.ptx not built"); return true; }
        return false;
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

    private static Tensor ToF16(Tensor f32)
    {
        Tensor t = new Tensor(f32.Shape, DType.F16);
        float* src = (float*)f32.DataPointer;
        ushort* dst = (ushort*)t.DataPointer;
        for (long i = 0; i < f32.ElementCount; i++) dst[i] = BitConverter.HalfToUInt16Bits((Half)src[i]);
        return t;
    }

    private static (double maxErr, double meanErr) Compare(Tensor a, Tensor b)
    {
        float* ap = (float*)a.DataPointer;
        float* bp = (float*)b.DataPointer;
        double maxErr = 0, sum = 0;
        long n = a.ElementCount;
        for (long i = 0; i < n; i++)
        {
            if (!float.IsFinite(ap[i]) || !float.IsFinite(bp[i]))
                return (double.PositiveInfinity, double.PositiveInfinity);
            double e = Math.Abs(ap[i] - bp[i]);
            if (e > maxErr) maxErr = e;
            sum += e;
        }
        return (maxErr, sum / n);
    }

    /// <summary>Runs native-F16 Q/K/V through Sage twice on one backend (the gate lowered so test shapes engage it) and
    /// returns both outputs widened to F32.</summary>
    private static (Tensor First, Tensor Second) RunSage(Tensor q16, Tensor k16, Tensor v16, float scale)
    {
        using Tensor out1 = new Tensor(q16.Shape, DType.F16);
        using Tensor out2 = new Tensor(q16.Shape, DType.F16);
        KnobStore.Set(EngineKnobs.SageAttn, true);
        KnobStore.Set(EngineKnobs.SageF16MinSkv, 1);
        try
        {
            using CudaBackend cuda = new CudaBackend(0, PtxDir());
            cuda.ScaledDotProductAttention(out1, q16, k16, v16, null, scale);
            cuda.ScaledDotProductAttention(out2, q16, k16, v16, null, scale);
            cuda.Sync();
            Assert.Equal(2L, cuda.SageAttentionExecutionCount);   // a fallback path would leave the checks below vacuous
            return (out1.CastTo(DType.F32), out2.CastTo(DType.F32));
        }
        finally
        {
            KnobStore.Clear(EngineKnobs.SageAttn);
            KnobStore.Clear(EngineKnobs.SageF16MinSkv);
        }
    }

    /// <summary>Head dims 64 and 128 against the F32 reference on the same F16 values, with key tails inside a 32-key step
    /// (1025, 250) and a query tail inside a 64-row tile (250), and bitwise equal across two runs.</summary>
    [Trait("Category", "GpuIntegration")]
    [Theory]
    [InlineData(128, 256, 1024)]
    [InlineData(128, 256, 1025)]
    [InlineData(128, 256, 250)]
    [InlineData(64, 512, 512)]
    [InlineData(64, 250, 1025)]
    public void SageAttention_MatchesCpuF32_WithKOutliers(int d, int sq, int skv)
    {
        if (Skip()) return;
        const int B = 1, H = 2;
        float scale = 1.0f / MathF.Sqrt(d);
        using Tensor q32 = RandomF32(new TensorShape(B, H, sq, d), 71);
        using Tensor k32 = RandomF32(new TensorShape(B, H, skv, d), 72);
        using Tensor v32 = RandomF32(new TensorShape(B, H, skv, d), 73);
        InjectChannelOutliers(k32, d, magnitude: 8.0f);
        using Tensor q16 = ToF16(q32), k16 = ToF16(k32), v16 = ToF16(v32);
        using Tensor qRef = q16.CastTo(DType.F32), kRef = k16.CastTo(DType.F32), vRef = v16.CastTo(DType.F32);

        using Tensor cpuOut = new Tensor(new TensorShape(B, H, sq, d), DType.F32);
        using (CpuBackend cpu = new CpuBackend())
            ((IBackend)cpu).ScaledDotProductAttention(cpuOut, qRef, kRef, vRef, null, scale);

        (Tensor gpuOut, Tensor gpuOut2) = RunSage(q16, k16, v16, scale);
        using (gpuOut)
        using (gpuOut2)
        {
            (double maxErr, double meanErr) = Compare(gpuOut, cpuOut);
            _output.WriteLine($"D={d} Sq={sq} Skv={skv}: maxErr={maxErr:E3} meanErr={meanErr:E3} vs CPU F32");
            Assert.True(maxErr < 2e-2, $"max error {maxErr:E3} exceeds the 2e-2 Sage budget");
            Assert.True(meanErr < 2e-3, $"mean error {meanErr:E3} exceeds 2e-3");
            (double detMax, _) = Compare(gpuOut, gpuOut2);
            Assert.True(detMax == 0.0, $"kernel is not run-to-run deterministic (max diff {detMax:E3})");
        }
    }

    /// <summary>Overflow gate for the F16-accumulate PV: uniform attention (Q = 0 ⇒ every softmax weight equal) over
    /// constant positive V drives each key step's F16 tile sum toward its bound, which the 1/16 P pre-scale and the per-step
    /// drain into F32 must absorb at any key count. The output must be exactly v; NaN/Inf or drift means the headroom failed.</summary>
    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void SageAttention_F16AccPv_UniformAttention_LargeSkv_NoOverflow()
    {
        if (Skip()) return;
        const int B = 1, H = 1, Sq = 64, Skv = 24576, D = 128;
        const float VConst = 3.0f;
        float scale = 1.0f / MathF.Sqrt(D);
        using Tensor q32 = new Tensor(new TensorShape(B, H, Sq, D), DType.F32);      // zeros → uniform attention
        using Tensor k32 = RandomF32(new TensorShape(B, H, Skv, D), 91);
        using Tensor v32 = new Tensor(new TensorShape(B, H, Skv, D), DType.F32);
        float* vp = (float*)v32.DataPointer;
        for (long i = 0; i < v32.ElementCount; i++) vp[i] = VConst;
        _ = *(float*)q32.DataPointer;   // host-materialize the zero buffer
        using Tensor q16 = ToF16(q32), k16 = ToF16(k32), v16 = ToF16(v32);

        (Tensor gpuOut, Tensor gpuOut2) = RunSage(q16, k16, v16, scale);
        using (gpuOut)
        using (gpuOut2)
        {
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
    }

    /// <summary>The refutation control for the smoothing claim at kernel level: WITHOUT outliers the budget
    /// must also hold (smoothing must not hurt the clean case).</summary>
    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void SageAttention_CleanInputs_WithinBudget()
    {
        if (Skip()) return;
        const int B = 1, H = 2, Sq = 256, Skv = 1024, D = 128;
        float scale = 1.0f / MathF.Sqrt(D);
        using Tensor q32 = RandomF32(new TensorShape(B, H, Sq, D), 81);
        using Tensor k32 = RandomF32(new TensorShape(B, H, Skv, D), 82);
        using Tensor v32 = RandomF32(new TensorShape(B, H, Skv, D), 83);
        using Tensor q16 = ToF16(q32), k16 = ToF16(k32), v16 = ToF16(v32);
        using Tensor qRef = q16.CastTo(DType.F32), kRef = k16.CastTo(DType.F32), vRef = v16.CastTo(DType.F32);

        using Tensor cpuOut = new Tensor(new TensorShape(B, H, Sq, D), DType.F32);
        using (CpuBackend cpu = new CpuBackend())
            ((IBackend)cpu).ScaledDotProductAttention(cpuOut, qRef, kRef, vRef, null, scale);

        (Tensor gpuOut, Tensor gpuOut2) = RunSage(q16, k16, v16, scale);
        using (gpuOut)
        using (gpuOut2)
        {
            (double maxErr, double meanErr) = Compare(gpuOut, cpuOut);
            _output.WriteLine($"clean: maxErr={maxErr:E3} meanErr={meanErr:E3}");
            Assert.True(maxErr < 2e-2, $"max error {maxErr:E3} exceeds the 2e-2 Sage budget");
        }
    }
}
