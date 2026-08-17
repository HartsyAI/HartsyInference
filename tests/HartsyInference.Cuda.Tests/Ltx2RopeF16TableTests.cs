using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Pins the F16-cos/sin variant of the fused LTX-2 QK path against the F32-table original. The tables are
/// half of that kernel's traffic at LTX-2.5's video shape and it is bandwidth-bound, so halving them is worth real
/// step time — but cos/sin live in [-1,1] where F16 carries ~3 decimal digits, so the gate is a relative-error
/// bound, not bit-identity. Shapes are non-square (seq != heads != headDim) so a wrong lane stride cannot survive.</summary>
[Collection("CudaSerial")]
public sealed unsafe class Ltx2RopeF16TableTests
{
    private const float Eps = 1e-6f;
    private readonly ITestOutputHelper _output;
    public Ltx2RopeF16TableTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    private static Tensor Uniform(TensorShape shape, int seed, DType dtype)
    {
        Tensor t = new Tensor(shape, DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new Random(seed);
        for (long i = 0; i < shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2 - 1);
        if (dtype == DType.F32) return t;
        Tensor cast = t.CastTo(dtype);
        t.Dispose();
        return cast;
    }

    /// <summary>cos/sin of real phases rather than uniform noise — an F16 table's error depends on where the value
    /// sits in [-1,1], so feeding it a genuine rotation is the only representative input.</summary>
    private static (Tensor Cos, Tensor Sin) RopeTables(int seq, int halfW, int seed)
    {
        Tensor cos = new Tensor(new TensorShape(seq, halfW), DType.F32);
        Tensor sin = new Tensor(new TensorShape(seq, halfW), DType.F32);
        float* c = (float*)cos.DataPointer;
        float* s = (float*)sin.DataPointer;
        Random rng = new Random(seed);
        for (long i = 0; i < (long)seq * halfW; i++)
        {
            double phase = (rng.NextDouble() * 2 - 1) * Math.PI;
            c[i] = (float)Math.Cos(phase);
            s[i] = (float)Math.Sin(phase);
        }
        return (cos, sin);
    }

    /// <summary>RMS weights sit near 1 in every real checkpoint, which is what makes the rotated row unit-scale and
    /// an elementwise relative error meaningful; a uniform[-1,1] weight would push most outputs into the band where
    /// F16's own output quantization, not the table, sets the error.</summary>
    private static Tensor NormWeight(int inner, int seed)
    {
        Tensor t = new Tensor(new TensorShape(inner), DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new Random(seed);
        for (int i = 0; i < inner; i++) p[i] = (float)(1.0 + (rng.NextDouble() * 2 - 1) * 0.1);
        return t;
    }

    /// <summary>max abs, norm-relative L2, and elementwise max rel restricted to <c>|ref| >= 0.5</c>. Below that the
    /// F16 OUTPUT's own ULP (~1e-3 at unit scale) dominates any table difference, so an unrestricted ratio measures
    /// the store dtype rather than the tables. relL2 is the metric attention actually feels — scores are dot
    /// products, so a norm-relative perturbation of Q/K is what moves them.</summary>
    private static (double MaxAbs, double RelL2, double MaxRel) Compare(Tensor expected, Tensor actual)
    {
        using Tensor e32 = expected.CastTo(DType.F32);
        using Tensor a32 = actual.CastTo(DType.F32);
        float* e = (float*)e32.DataPointer;
        float* a = (float*)a32.DataPointer;
        double maxAbs = 0, maxRel = 0, sumSqDiff = 0, sumSqRef = 0;
        for (long i = 0; i < e32.ElementCount; i++)
        {
            double d = Math.Abs(e[i] - a[i]);
            maxAbs = Math.Max(maxAbs, d);
            sumSqDiff += d * d;
            sumSqRef += (double)e[i] * e[i];
            if (Math.Abs(e[i]) >= 0.5) maxRel = Math.Max(maxRel, d / Math.Abs(e[i]));
        }
        return (maxAbs, Math.Sqrt(sumSqDiff / sumSqRef), maxRel);
    }

    [Theory]
    [InlineData(37, 8, 64)]
    [InlineData(512, 32, 128)]
    public void TokenMajor_F16Tables_MatchF32Tables(int seq, int heads, int headDim)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        int inner = heads * headDim;
        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        using Tensor x = Uniform(new TensorShape(seq, inner), seed: 3, DType.F16);
        using Tensor normW = NormWeight(inner, seed: 4);
        (Tensor cos32, Tensor sin32) = RopeTables(seq, inner / 2, seed: 5);
        using Tensor cos = cos32, sin = sin32;
        using Tensor cos16 = cos32.CastTo(DType.F16);
        using Tensor sin16 = sin32.CastTo(DType.F16);

        using Tensor expected = new Tensor(new TensorShape(seq, inner), DType.F16);
        cuda.Ltx2QkNormRopeTokenMajor(expected, x, normW, cos, sin, seq, heads, headDim, Eps);
        using Tensor actual = new Tensor(new TensorShape(seq, inner), DType.F16);
        cuda.Ltx2QkNormRopeTokenMajor(actual, x, normW, cos16, sin16, seq, heads, headDim, Eps);
        cuda.Sync();

        (double maxAbs, double relL2, double maxRel) = Compare(expected, actual);
        _output.WriteLine($"seq={seq} heads={heads} headDim={headDim}  max abs {maxAbs:E3}  relL2 {relL2:E3}  max rel {maxRel:E3}");
        Assert.True(maxRel < 5e-3, $"F16 rope tables: max rel {maxRel} >= 5e-3 (max abs {maxAbs}, relL2 {relL2})");
        Assert.True(relL2 < 5e-3, $"F16 rope tables: relL2 {relL2} >= 5e-3");
    }

    /// <summary>Head-major twin of the token-major gate — the other layout shares the kernel body, so a table-load
    /// mistake that only the scattered store exposes still fails here.</summary>
    [Fact]
    public void HeadMajor_F16Tables_MatchF32Tables()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        const int seq = 37, heads = 8, headDim = 64, inner = heads * headDim;
        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        using Tensor x = Uniform(new TensorShape(seq, inner), seed: 13, DType.F16);
        using Tensor normW = NormWeight(inner, seed: 14);
        (Tensor cos32, Tensor sin32) = RopeTables(seq, inner / 2, seed: 15);
        using Tensor cos = cos32, sin = sin32;
        using Tensor cos16 = cos32.CastTo(DType.F16);
        using Tensor sin16 = sin32.CastTo(DType.F16);

        using Tensor expected = new Tensor(new TensorShape(1, heads, seq, headDim), DType.F16);
        cuda.Ltx2QkNormRopeHeadMajor(expected, x, normW, cos, sin, seq, heads, headDim, Eps);
        using Tensor actual = new Tensor(new TensorShape(1, heads, seq, headDim), DType.F16);
        cuda.Ltx2QkNormRopeHeadMajor(actual, x, normW, cos16, sin16, seq, heads, headDim, Eps);
        cuda.Sync();

        (double maxAbs, double relL2, double maxRel) = Compare(expected, actual);
        _output.WriteLine($"head-major  max abs {maxAbs:E3}  relL2 {relL2:E3}  max rel {maxRel:E3}");
        Assert.True(maxRel < 5e-3, $"F16 rope tables (head-major): max rel {maxRel} >= 5e-3");
        Assert.True(relL2 < 5e-3, $"F16 rope tables (head-major): relL2 {relL2} >= 5e-3");
    }

    /// <summary>Times the kernel itself at LTX-2.5's video shape — raw device buffers, null stream, no per-call
    /// upload/alloc — because the delta being measured is ~50 µs and the backend op's bookkeeping would drown it.
    /// Diagnostic, not a gate: it asserts only that both variants ran.</summary>
    [Fact]
    public void TokenMajor_TableDtype_Throughput()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        const int seq = 4992, heads = 32, headDim = 128, inner = heads * headDim;
        using CudaBackend cuda = new CudaBackend(0, PtxDir());   // establishes the CUDA context the module loads into
        using CudaKernels kernels = new CudaKernels(PtxDir());
        using Tensor x = Uniform(new TensorShape(seq, inner), seed: 3, DType.F16);
        using Tensor normW = Uniform(new TensorShape(inner), seed: 4, DType.F32);
        (Tensor cos32, Tensor sin32) = RopeTables(seq, inner / 2, seed: 5);
        using Tensor cos = cos32, sin = sin32;
        using Tensor cos16 = cos32.CastTo(DType.F16);
        using Tensor sin16 = sin32.CastTo(DType.F16);

        ulong pX = Upload(x), pW = Upload(normW), pC32 = Upload(cos), pS32 = Upload(sin);
        ulong pC16 = Upload(cos16), pS16 = Upload(sin16);
        CudaDriverApi.cuMemAlloc(out ulong pOut, (nuint)((long)seq * inner * 2)).ThrowOnError();
        try
        {
            double f32Ms = TimeRope(kernels, pOut, pX, pW, pC32, pS32, seq, heads, headDim, ropeF16: false);
            double f16Ms = TimeRope(kernels, pOut, pX, pW, pC16, pS16, seq, heads, headDim, ropeF16: true);
            double actBytes = 2.0 * seq * inner * 2;                       // F16 in + F16 out
            _output.WriteLine($"seq={seq} heads={heads} headDim={headDim}");
            _output.WriteLine($"  F32 tables: {f32Ms:F4} ms  {(actBytes + 2.0 * seq * inner / 2 * 4) / (f32Ms * 1e-3) / 1e9:F0} GB/s");
            _output.WriteLine($"  F16 tables: {f16Ms:F4} ms  {(actBytes + 2.0 * seq * inner / 2 * 2) / (f16Ms * 1e-3) / 1e9:F0} GB/s");
            Assert.True(f32Ms > 0 && f16Ms > 0);
        }
        finally
        {
            CudaDriverApi.cuMemFree(pOut);
            foreach (ulong p in new[] { pX, pW, pC32, pS32, pC16, pS16 }) CudaDriverApi.cuMemFree(p);
        }
    }

    private static ulong Upload(Tensor t)
    {
        nuint bytes = (nuint)(t.Shape.ElementCount * (t.DType == DType.F32 ? 4 : 2));
        CudaDriverApi.cuMemAlloc(out ulong p, bytes).ThrowOnError();
        CudaDriverApi.cuMemcpyHtoD(p, (nint)t.DataPointer, bytes).ThrowOnError();
        return p;
    }

    private static double TimeRope(CudaKernels kernels, ulong pOut, ulong pX, ulong pW, ulong pCos, ulong pSin,
        int seq, int heads, int headDim, bool ropeF16)
    {
        for (int i = 0; i < 10; i++)
            kernels.LaunchLtx2QkNormRopeTokenMajor(pOut, pX, pW, pCos, pSin, seq, heads, headDim, Eps, 0, f16: true, ropeF16);
        CudaDriverApi.cuStreamSynchronize(0).ThrowOnError();
        const int Reps = 50;
        System.Diagnostics.Stopwatch sw = System.Diagnostics.Stopwatch.StartNew();
        for (int i = 0; i < Reps; i++)
            kernels.LaunchLtx2QkNormRopeTokenMajor(pOut, pX, pW, pCos, pSin, seq, heads, headDim, Eps, 0, f16: true, ropeF16);
        CudaDriverApi.cuStreamSynchronize(0).ThrowOnError();
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds / Reps;
    }
}
