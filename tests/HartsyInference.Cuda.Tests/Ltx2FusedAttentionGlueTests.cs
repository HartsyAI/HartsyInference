using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Pins the two fused LTX-2 attention-glue ops against the op sequences they replace. Both are compared
/// to the EXPLICIT unfused chain (RmsNorm → Ltx2SplitRope → Permute0213, and Sigmoid → Scale → expand-GEMM →
/// Mul), not to a second copy of themselves, so a shared mistake cannot pass. Shapes are deliberately non-square
/// (seq != heads != headDim) so a transposed axis or a wrong head stride cannot survive.</summary>
[Collection("CudaSerial")]
public sealed unsafe class Ltx2FusedAttentionGlueTests
{
    private const float Eps = 1e-6f;
    private readonly ITestOutputHelper _output;
    public Ltx2FusedAttentionGlueTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    private static Tensor Random(TensorShape shape, int seed, DType? dt = null)
    {
        Tensor t = new Tensor(shape, DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new Random(seed);
        for (long i = 0; i < shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2 - 1);
        DType dtype = dt ?? DType.F32;
        if (dtype == DType.F32) return t;
        Tensor cast = t.CastTo(dtype);
        t.Dispose();
        return cast;
    }

    private static void AssertClose(Tensor expected, Tensor actual, float tol, string what)
    {
        Tensor e32 = expected.DType == DType.F32 ? expected : expected.CastTo(DType.F32);
        Tensor a32 = actual.DType == DType.F32 ? actual : actual.CastTo(DType.F32);
        float* e = (float*)e32.DataPointer;
        float* a = (float*)a32.DataPointer;
        double maxErr = 0;
        for (long i = 0; i < e32.ElementCount; i++) maxErr = Math.Max(maxErr, Math.Abs(e[i] - a[i]));
        if (!ReferenceEquals(e32, expected)) e32.Dispose();
        if (!ReferenceEquals(a32, actual)) a32.Dispose();
        Assert.True(maxErr <= tol, $"{what}: max abs err {maxErr} > {tol}");
    }

    [Theory]
    [InlineData(false, 2e-5f)]
    [InlineData(true, 4e-3f)]
    public void QkNormRopeHeadMajor_MatchesUnfusedSequence(bool f16, float tol)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        DType act = f16 ? DType.F16 : DType.F32;
        const int seq = 5, heads = 3, headDim = 8;
        int inner = heads * headDim;
        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        using Tensor x = Random(new TensorShape(seq, inner), seed: 3, act);
        using Tensor normW = Random(new TensorShape(inner), seed: 4);
        using Tensor cos = Random(new TensorShape(seq, inner / 2), seed: 5);
        using Tensor sin = Random(new TensorShape(seq, inner / 2), seed: 6);

        // Reference: the three ops the fused kernel replaces, run individually on the same backend.
        using Tensor normed = new Tensor(x.Shape, act);
        cuda.RmsNorm(normed, x, normW, Eps);
        cuda.Ltx2SplitRope(normed, cos, sin, seq, heads, headDim);
        using Tensor expected = new Tensor(new TensorShape(1, heads, seq, headDim), act);
        cuda.Permute0213(expected, normed, seq, heads, headDim);

        using Tensor actual = new Tensor(new TensorShape(1, heads, seq, headDim), act);
        cuda.Ltx2QkNormRopeHeadMajor(actual, x, normW, cos, sin, seq, heads, headDim, Eps);
        cuda.Sync();
        AssertClose(expected, actual, tol, $"fused QK path ({act})");
    }

    /// <summary>Text cross-attention passes no RoPE, so the null-cos path has to reduce to norm + head-major.</summary>
    [Fact]
    public void QkNormRopeHeadMajor_WithoutRope_IsNormThenHeadMajor()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        const int seq = 7, heads = 2, headDim = 4;
        int inner = heads * headDim;
        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        using Tensor x = Random(new TensorShape(seq, inner), seed: 13);
        using Tensor normW = Random(new TensorShape(inner), seed: 14);

        using Tensor normed = new Tensor(x.Shape, DType.F32);
        cuda.RmsNorm(normed, x, normW, Eps);
        using Tensor expected = new Tensor(new TensorShape(1, heads, seq, headDim), DType.F32);
        cuda.Permute0213(expected, normed, seq, heads, headDim);

        using Tensor actual = new Tensor(new TensorShape(1, heads, seq, headDim), DType.F32);
        cuda.Ltx2QkNormRopeHeadMajor(actual, x, normW, null, null, seq, heads, headDim, Eps);
        cuda.Sync();
        AssertClose(expected, actual, 2e-5f, "fused QK path, no rope");
    }

    [Theory]
    [InlineData(false, 2e-6f)]
    [InlineData(true, 3e-3f)]
    public void HeadGate_MatchesSigmoidScaleExpandMultiply(bool f16, float tol)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        DType act = f16 ? DType.F16 : DType.F32;
        const int seq = 6, heads = 3, headDim = 4;
        int inner = heads * headDim;
        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        using Tensor x = Random(new TensorShape(seq, inner), seed: 21, act);
        using Tensor logits = Random(new TensorShape(seq, heads), seed: 22, act);

        // Reference: the host form of what the old chain computed — 2*sigmoid(logit) broadcast within each head.
        using Tensor xRef = x.CastTo(DType.F32);
        using Tensor logitsRef = logits.CastTo(DType.F32);
        float* xr = (float*)xRef.DataPointer;
        float* lr = (float*)logitsRef.DataPointer;
        for (int s = 0; s < seq; s++)
            for (int h = 0; h < heads; h++)
            {
                float g = 2f / (1f + MathF.Exp(-lr[s * heads + h]));
                for (int d = 0; d < headDim; d++) xr[(long)s * inner + h * headDim + d] *= g;
            }

        cuda.Ltx2HeadGate(x, logits, seq, heads, headDim);
        cuda.Sync();
        AssertClose(xRef, x, tol, $"fused head gate ({act})");
    }

    /// <summary>The gate mutates in place, so the managed default and the CUDA kernel must agree on a shape the
    /// CPU backend can also serve — this is what catches an in-place op whose result never left the device cache.</summary>
    [Fact]
    public void HeadGate_CudaMatchesManagedDefault()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        const int seq = 4, heads = 2, headDim = 6;
        int inner = heads * headDim;
        using Tensor xCpu = Random(new TensorShape(seq, inner), seed: 31);
        using Tensor xGpu = Random(new TensorShape(seq, inner), seed: 31);
        using Tensor logits = Random(new TensorShape(seq, heads), seed: 32);

        ((IBackend)new CpuBackend()).Ltx2HeadGate(xCpu, logits, seq, heads, headDim);
        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        cuda.Ltx2HeadGate(xGpu, logits, seq, heads, headDim);
        cuda.Sync();
        AssertClose(xCpu, xGpu, 2e-6f, "head gate CUDA vs managed default");
    }
}
