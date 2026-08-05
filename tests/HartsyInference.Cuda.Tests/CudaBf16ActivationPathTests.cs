using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Covers the half-precision activation paths a BF16 or F16 DiT residual stream drives: the native
/// <c>dit_rmsnorm_bf16</c> / <c>dit_glu_act_bf16</c> / <c>dit_glu_act_f16</c> kernels (compared against their
/// already-validated F32 twins), and the dtype guards on the
/// ops that used to re-dispatch through <c>((IBackend)this)</c>. That cast resolved straight back to the CudaBackend
/// method that issued it, so a non-F32 argument recursed until the stack overflowed and killed the test host —
/// uncatchable in-process, which is why these tests assert on values/exceptions rather than <c>Assert.Throws</c> on
/// the overflow: if the recursion returns, the run aborts instead of failing one test. Skips when CUDA is absent.</summary>
[Collection("CudaSerial")]
public sealed unsafe class CudaBf16ActivationPathTests
{
    private readonly ITestOutputHelper _output;
    public CudaBf16ActivationPathTests(ITestOutputHelper output) => _output = output;

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

    /// <summary>Host read of one element as float, for F32/F16/BF16 result tensors.</summary>
    private static float ReadAs(Tensor t, long i)
    {
        if (t.DType == DType.F32) return ((float*)t.DataPointer)[i];
        if (t.DType == DType.F16) return (float)((Half*)t.DataPointer)[i];
        return BitConverter.Int32BitsToSingle(((ushort*)t.DataPointer)[i] << 16);
    }

    private void AssertClose(Tensor expected, Tensor actual, float tol, string name)
    {
        long n = expected.ElementCount;
        double maxErr = 0;
        long maxIdx = 0;
        for (long i = 0; i < n; i++)
        {
            double e = Math.Abs(ReadAs(expected, i) - ReadAs(actual, i));
            if (e > maxErr) { maxErr = e; maxIdx = i; }
        }
        _output.WriteLine($"{name}: max_err={maxErr:E3} @ {maxIdx} over {n} elems (tol {tol:E0})");
        Assert.True(maxErr <= tol, $"{name}: max_err {maxErr:E3} exceeds tol {tol:E0} at index {maxIdx}");
    }

    /// <summary>BF16 I/O RMSNorm vs the F32 kernel: both accumulate in F32 over the same row, so the gap is BF16
    /// I/O rounding only (~2^-8 relative). Rows here are the DiT shape (one block per token row).</summary>
    [Fact]
    public void RmsNormBf16_MatchesF32Twin()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        const int Seq = 37, Hidden = 128;
        TensorShape shape = new TensorShape(Seq, 1, Hidden);
        using Tensor x = Random(shape, seed: 11);
        using Tensor xBf = x.CastTo(DType.BF16);
        using Tensor weight = Random(new TensorShape(Hidden), seed: 12, 0.5f, 1.5f);

        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        IBackend b = cuda;
        using Tensor o32 = new Tensor(shape, DType.F32);
        using Tensor oBf = new Tensor(shape, DType.BF16);
        b.RmsNorm(o32, x, weight, 1e-6f);
        b.RmsNorm(oBf, xBf, weight, 1e-6f);
        cuda.Sync();

        AssertClose(o32, oBf, 3e-2f, "RmsNorm BF16");
    }

    /// <summary>BF16 gated-FFN epilogue vs the F32 kernel. Multi-row on purpose: the gate/up split is on the LAST
    /// dim, so a flat-midpoint split would pass at one row and garble the rest.</summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void GluActivateBf16_MatchesF32Twin(bool gelu)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        const int Rows = 5, Ff = 64;
        using Tensor gateUp = Random(new TensorShape(Rows, 2 * Ff), seed: 21);
        using Tensor gateUpBf = gateUp.CastTo(DType.BF16);

        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        IBackend b = cuda;
        using Tensor o32 = new Tensor(new TensorShape(Rows, Ff), DType.F32);
        using Tensor oBf = new Tensor(new TensorShape(Rows, Ff), DType.BF16);
        b.GluActivate(o32, gateUp, Ff, gelu);
        b.GluActivate(oBf, gateUpBf, Ff, gelu);
        cuda.Sync();

        AssertClose(o32, oBf, 3e-2f, $"GluActivate BF16(gelu={gelu})");
    }

    /// <summary>F16 gated-FFN epilogue vs the F32 kernel — the dtype the MiniMax-H3 MLP runs at. Multi-row for the
    /// last-dim split, and a second magnitude band because F16 (unlike BF16) can overflow on the stored product:
    /// act(g)·u leaves range once both factors reach ~256, long before BF16 would.</summary>
    [Theory]
    [InlineData(false, 1f)]
    [InlineData(true, 1f)]
    [InlineData(false, 20f)]
    [InlineData(true, 20f)]
    public void GluActivateF16_MatchesF32Twin(bool gelu, float mag)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        const int Rows = 5, Ff = 64;
        using Tensor gateUp = Random(new TensorShape(Rows, 2 * Ff), seed: 21, -mag, mag);
        using Tensor gateUpF16 = gateUp.CastTo(DType.F16);

        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        IBackend b = cuda;
        using Tensor o32 = new Tensor(new TensorShape(Rows, Ff), DType.F32);
        using Tensor oF16 = new Tensor(new TensorShape(Rows, Ff), DType.F16);
        b.GluActivate(o32, gateUp, Ff, gelu);
        b.GluActivate(oF16, gateUpF16, Ff, gelu);
        cuda.Sync();

        // KERNEL.md's 1e-3 FP16 figure is relative; peak |act(g)·u| here is ~mag², so scale the absolute bound by it.
        AssertClose(o32, oF16, 1e-3f * mag * mag, $"GluActivate F16(gelu={gelu}, mag={mag})");
    }

    /// <summary>The fused norm ops fall back to their unfused composition for dtypes their kernels don't cover.
    /// Regression for the recursive <c>((IBackend)this)</c> fallback: every call here took that branch.</summary>
    [Fact]
    public void FusedNormOps_Bf16_ComposeWithoutReentering()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        const int Rows = 8, Hidden = 64;
        TensorShape shape = new TensorShape(1, Rows, Hidden);
        using Tensor a = Random(shape, seed: 31);
        using Tensor bIn = Random(shape, seed: 32);
        using Tensor w1 = Random(new TensorShape(Hidden), seed: 33, 0.5f, 1.5f);
        using Tensor w2 = Random(new TensorShape(Hidden), seed: 34, 0.5f, 1.5f);
        using Tensor aBf = a.CastTo(DType.BF16);
        using Tensor bBf = bIn.CastTo(DType.BF16);

        using Tensor cpuResid = new Tensor(shape, DType.F32);
        using Tensor cpuNorm = new Tensor(shape, DType.F32);
        using Tensor cpuRmsAdd = new Tensor(shape, DType.F32);
        using Tensor cpuSandwichResid = new Tensor(shape, DType.F32);
        using Tensor cpuSandwichNorm = new Tensor(shape, DType.F32);
        using (IBackend cpu = new CpuBackend())
        {
            cpu.AddRmsNorm(cpuResid, cpuNorm, a, bIn, w2, 1e-6f);
            cpu.RmsNormAdd(cpuRmsAdd, a, bIn, w2, 1e-6f);
            cpu.NormAddRmsNormEmitQ8(cpuSandwichResid, cpuSandwichNorm, a, bIn, w1, w2, 1e-6f);
        }

        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        IBackend g = cuda;
        using Tensor resid = new Tensor(shape, DType.BF16);
        using Tensor norm = new Tensor(shape, DType.BF16);
        using Tensor rmsAdd = new Tensor(shape, DType.BF16);
        using Tensor sandwichResid = new Tensor(shape, DType.BF16);
        using Tensor sandwichNorm = new Tensor(shape, DType.BF16);
        g.AddRmsNorm(resid, norm, aBf, bBf, w2, 1e-6f);
        g.RmsNormAdd(rmsAdd, aBf, bBf, w2, 1e-6f);
        g.NormAddRmsNormEmitQ8(sandwichResid, sandwichNorm, aBf, bBf, w1, w2, 1e-6f);
        cuda.Sync();

        AssertClose(cpuResid, resid, 3e-2f, "AddRmsNorm BF16 resid");
        AssertClose(cpuNorm, norm, 3e-2f, "AddRmsNorm BF16 norm");
        AssertClose(cpuRmsAdd, rmsAdd, 3e-2f, "RmsNormAdd BF16");
        AssertClose(cpuSandwichResid, sandwichResid, 3e-2f, "NormAddRmsNorm BF16 resid");
        AssertClose(cpuSandwichNorm, sandwichNorm, 3e-2f, "NormAddRmsNorm BF16 norm");
    }

    /// <summary>Ops with no kernel for the requested dtype must say so. Before the fix each of these re-dispatched
    /// through the interface into itself and overflowed the stack instead of reporting anything.</summary>
    [Fact]
    public void UnsupportedDtypes_ThrowNotSupported_InsteadOfRecursing()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        const int Rows = 4, Hidden = 32;
        using Tensor src = Random(new TensorShape(Rows, Hidden), seed: 41);
        using Tensor srcBf = src.CastTo(DType.BF16);
        using Tensor srcF16 = src.CastTo(DType.F16);
        using Tensor dstBf = new Tensor(new TensorShape(Rows, Hidden), DType.BF16);
        using Tensor dstF16 = new Tensor(new TensorShape(Rows / 2, Hidden), DType.F16);
        using Tensor indices = new Tensor(new TensorShape(Rows), DType.I32);

        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        IBackend b = cuda;
        // F32/F16/BF16 all have kernels now; only a MIXED pair is unsupported.
        Assert.Throws<NotSupportedException>(() => b.GluActivate(dstF16, srcBf, Hidden / 2, gelu: false));
        Assert.Throws<NotSupportedException>(() => b.GluActivate(dstBf, srcF16, Hidden / 2, gelu: false));
        Assert.Throws<NotSupportedException>(() => b.GatherRows(dstBf, srcBf, [0, 1, 2, 3]));
        Assert.Throws<NotSupportedException>(() => b.ScatterAddWeightedRows(dstBf, srcBf, [0, 1, 2, 3], [1f, 1f, 1f, 1f]));
        Assert.Throws<NotSupportedException>(() => b.ArgMaxLastDim(indices, srcBf));
        Assert.Throws<NotSupportedException>(() => b.KvCacheAppend(dstBf, srcBf, 0));
        Assert.Throws<NotSupportedException>(() => b.SliceTimeRange(dstBf, srcBf, 0, 1));
        Assert.Throws<NotSupportedException>(() => b.RopeApplyDecodeStep(srcBf, src, src, Hidden, interleaved: false, devicePos: 0));

        // The fused graph-decode epilogues: devicePos = 0 means graph decode was never set up, so there is no
        // position to scatter at — the same guard the non-F32 KV cache dtypes take.
        Assert.Throws<NotSupportedException>(() => b.QkvRopeScatterDecodeStep(
            src, src, src, src, src, src, 1, 1, Hidden, Hidden, interleaved: false, devicePos: 0));
        Assert.Throws<NotSupportedException>(() => b.QkRopeScatterVDecodeStep(
            src, src, src, src, src, src, src, 1, 1, Hidden, Hidden, interleaved: false, devicePos: 0));
        Assert.Throws<NotSupportedException>(() => b.RopeScatterKvDecodeStep(
            src, src, src, src, src, src, src, src, 1, 1, Hidden, Hidden, interleaved: false, devicePos: 0));
        Assert.Throws<NotSupportedException>(() => b.QkvNormRopeScatterDecodeStep(
            src, src, src, src, src, src, 1e-6f, src, src, 1, 1, Hidden, Hidden, interleaved: false, devicePos: 0));
        Assert.Throws<NotSupportedException>(() => b.QkNormRopeScatterVDecodeStep(
            src, src, src, src, src, src, src, 1e-6f, src, src, 1, 1, Hidden, Hidden, interleaved: false, devicePos: 0));
    }
}
