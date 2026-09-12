using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Covers <see cref="CudaBackend.ApplyRopeSingle"/>'s F16 activation path against the F32 one.
/// <para>The contract is deliberately asymmetric — an F16 <c>x</c> with an F32 cos/sin table — because
/// <c>dit_rope_f16</c> declares the table as <c>const float*</c>. Handing it an F16 table over-reads by exactly
/// 2x and faults, which is how an earlier attempt at this crashed; the fix was not a new kernel but keeping the
/// table F32. These tests pin both halves of that contract so it cannot be relaxed by accident.</para></summary>
[Collection("CudaSerial")]
public sealed unsafe class RopeSingleF16Tests
{
    private readonly ITestOutputHelper _out;
    public RopeSingleF16Tests(ITestOutputHelper output) => _out = output;

    private static uint _rng = 0x1234567u;
    private static float Rand() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return ((_rng & 0xFFFF) / 65535f - 0.5f); }

    private static CudaBackend Make()
    {
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return new CudaBackend(0, ptxDir);
    }

    /// <summary>Builds the same split-half table GenericTransformer.BuildRope emits: both halves duplicated.</summary>
    private static (Tensor Cos, Tensor Sin) Table(int seq, int headDim)
    {
        Tensor cos = new(new TensorShape(1, seq, headDim), DType.F32);
        Tensor sin = new(new TensorShape(1, seq, headDim), DType.F32);
        float* pc = (float*)cos.DataPointer, ps = (float*)sin.DataPointer;
        int half = headDim / 2;
        for (int s = 0; s < seq; s++)
        {
            for (int i = 0; i < half; i++)
            {
                double angle = s / Math.Pow(10000.0, 2.0 * i / headDim);
                float c = (float)Math.Cos(angle), sn = (float)Math.Sin(angle);
                long b = (long)s * headDim;
                pc[b + i] = c; pc[b + i + half] = c;
                ps[b + i] = sn; ps[b + i + half] = sn;
            }
        }
        return (cos, sin);
    }

    [Theory]
    [InlineData(0)]     // full rotary
    [InlineData(32)]    // partial rotary — the rest of each head passes through untouched
    public void ApplyRopeSingle_F16_MatchesF32(int rotaryDim)
    {
        if (!CudaContext.IsAvailable()) { _out.WriteLine("SKIPPED: CUDA unavailable"); return; }
        const int seq = 24, heads = 4, headDim = 64;

        using CudaBackend backend = Make();
        IBackend b = backend;
        using Tensor x32 = new(new TensorShape(1, seq, heads, headDim), DType.F32);
        float* px = (float*)x32.DataPointer;
        for (long i = 0; i < x32.ElementCount; i++) px[i] = Rand();

        using Tensor x16 = new(x32.Shape, DType.F16);
        b.CastToF16(x16, x32);
        backend.Sync();

        (Tensor cos, Tensor sin) = Table(seq, headDim);
        using (cos)
        using (sin)
        {
            b.ApplyRopeSingle(x32, cos, sin, rotaryDim);
            b.ApplyRopeSingle(x16, cos, sin, rotaryDim);
            backend.Sync();
        }

        using Tensor back = new(x32.Shape, DType.F32);
        b.CastToF32(back, x16);
        backend.Sync();

        float* a = (float*)x32.DataPointer;
        float* g = (float*)back.DataPointer;
        float maxDiff = 0f;
        for (long i = 0; i < x32.ElementCount; i++) maxDiff = MathF.Max(maxDiff, MathF.Abs(a[i] - g[i]));
        _out.WriteLine($"rotaryDim={rotaryDim} maxDiff={maxDiff:E3}");
        // The rotation itself runs in fp32 inside the kernel; the error is the F16 round-trip of the operands.
        Assert.True(maxDiff < 2e-3f, $"F16 rope diverged from F32: maxDiff={maxDiff:E3}");
    }

    [Fact]
    public void ApplyRopeSingle_RejectsF16Table()
    {
        if (!CudaContext.IsAvailable()) { _out.WriteLine("SKIPPED: CUDA unavailable"); return; }
        const int seq = 8, heads = 2, headDim = 64;
        using CudaBackend backend = Make();
        IBackend b = backend;
        using Tensor x16 = new(new TensorShape(1, seq, heads, headDim), DType.F16);
        (Tensor cos32, Tensor sin32) = Table(seq, headDim);
        using (cos32)
        using (sin32)
        {
            using Tensor cos16 = new(cos32.Shape, DType.F16);
            using Tensor sin16 = new(sin32.Shape, DType.F16);
            b.CastToF16(cos16, cos32);
            b.CastToF16(sin16, sin32);
            backend.Sync();
            // The kernel would read this 4 bytes at a time and run off the end of the table.
            Assert.Throws<NotSupportedException>(() => b.ApplyRopeSingle(x16, cos16, sin16));
        }
    }

    [Fact]
    public void ApplyRopeSingle_RejectsNonRank4()
    {
        if (!CudaContext.IsAvailable()) { _out.WriteLine("SKIPPED: CUDA unavailable"); return; }
        using CudaBackend backend = Make();
        IBackend b = backend;
        using Tensor x = new(new TensorShape(8, 2, 64), DType.F32);
        (Tensor cos, Tensor sin) = Table(8, 64);
        using (cos)
        using (sin)
        {
            // Heads/headDim come from Shape[2]/Shape[3]; a rank-3 tensor would read garbage dims.
            Assert.Throws<NotSupportedException>(() => b.ApplyRopeSingle(x, cos, sin));
        }
    }
}
