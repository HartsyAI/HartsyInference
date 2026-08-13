using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using Xunit;

namespace HartsyInference.Cuda.Tests;

/// <summary>Pins the token-major LTX-2 attention route against the head-major one it replaces. The rope kernel is
/// compared to the EXISTING head-major kernel read through the layout mapping (not a re-derivation of the same
/// math), and the full attention is compared to the head-major branch of the very same <c>Forward</c>. Shapes are
/// non-square (seq != heads != headDim) so a swapped axis or a wrong head stride cannot survive.</summary>
[Collection("CudaSerial")]
public sealed unsafe class Ltx2TokenMajorAttentionTests
{
    private const float Eps = 1e-6f;

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    private static Tensor Random(TensorShape shape, int seed, DType? dt = null, float amp = 1f)
    {
        Tensor t = new Tensor(shape, DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new Random(seed);
        for (long i = 0; i < shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2 - 1) * amp;
        DType dtype = dt ?? DType.F32;
        if (dtype == DType.F32) return t;
        Tensor cast = t.CastTo(dtype);
        t.Dispose();
        return cast;
    }

    private static float[] ToF32(Tensor t)
    {
        Tensor f = t.DType == DType.F32 ? t : t.CastTo(DType.F32);
        float[] o = new float[f.ElementCount];
        new Span<float>((void*)f.DataPointer, (int)f.ElementCount).CopyTo(o);
        if (!ReferenceEquals(f, t)) f.Dispose();
        return o;
    }

    /// <summary>The token-major kernel must be the head-major kernel's output re-indexed, element for element:
    /// <c>headMajor[(h*seq + s)*headDim + d] == tokenMajor[s*heads*headDim + h*headDim + d]</c>.</summary>
    [Theory]
    [InlineData(37, 8, 64, false)]
    [InlineData(37, 8, 64, true)]
    [InlineData(512, 32, 128, false)]
    [InlineData(512, 32, 128, true)]
    public void QkNormRopeTokenMajor_IsHeadMajorReindexed(int seq, int heads, int headDim, bool f16)
    {
        DType act = f16 ? DType.F16 : DType.F32;
        int inner = heads * headDim;
        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        using Tensor x = Random(new TensorShape(seq, inner), seed: 101, act);
        using Tensor normW = Random(new TensorShape(inner), seed: 102);
        using Tensor cos = Random(new TensorShape(seq, inner / 2), seed: 103);
        using Tensor sin = Random(new TensorShape(seq, inner / 2), seed: 104);

        using Tensor hm = new Tensor(new TensorShape(1, heads, seq, headDim), act);
        cuda.Ltx2QkNormRopeHeadMajor(hm, x, normW, cos, sin, seq, heads, headDim, Eps);
        using Tensor tm = new Tensor(new TensorShape(seq, inner), act);
        cuda.Ltx2QkNormRopeTokenMajor(tm, x, normW, cos, sin, seq, heads, headDim, Eps);
        cuda.Sync();
        AssertReindexed(ToF32(hm), ToF32(tm), seq, heads, headDim, f16 ? 4e-3f : 2e-6f);
    }

    /// <summary>Text cross-attention passes no RoPE, so the null-cos path has to agree too.</summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void QkNormRopeTokenMajor_WithoutRope_IsHeadMajorReindexed(bool f16)
    {
        DType act = f16 ? DType.F16 : DType.F32;
        const int seq = 37, heads = 8, headDim = 64;
        int inner = heads * headDim;
        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        using Tensor x = Random(new TensorShape(seq, inner), seed: 111, act);
        using Tensor normW = Random(new TensorShape(inner), seed: 112);

        using Tensor hm = new Tensor(new TensorShape(1, heads, seq, headDim), act);
        cuda.Ltx2QkNormRopeHeadMajor(hm, x, normW, null, null, seq, heads, headDim, Eps);
        using Tensor tm = new Tensor(new TensorShape(seq, inner), act);
        cuda.Ltx2QkNormRopeTokenMajor(tm, x, normW, null, null, seq, heads, headDim, Eps);
        cuda.Sync();
        AssertReindexed(ToF32(hm), ToF32(tm), seq, heads, headDim, f16 ? 4e-3f : 2e-6f);
    }

    /// <summary>Token-major SDPA must match the head-major dispatch it wraps, on a cross-attention shape (Sq != Sk)
    /// so a Q/K sequence-stride mixup cannot pass.</summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ScaledDotProductAttentionTokenMajor_MatchesHeadMajorDispatch(bool f16)
    {
        DType act = f16 ? DType.F16 : DType.F32;
        const int sq = 37, sk = 53, heads = 8, headDim = 64;
        int inner = heads * headDim;
        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        using Tensor q = Random(new TensorShape(sq, inner), seed: 201, act, amp: 0.5f);
        using Tensor k = Random(new TensorShape(sk, inner), seed: 202, act, amp: 0.5f);
        using Tensor v = Random(new TensorShape(sk, inner), seed: 203, act, amp: 0.5f);
        float scale = 1f / MathF.Sqrt(headDim);

        using Tensor qMh = new Tensor(new TensorShape(1, heads, sq, headDim), act);
        using Tensor kMh = new Tensor(new TensorShape(1, heads, sk, headDim), act);
        using Tensor vMh = new Tensor(new TensorShape(1, heads, sk, headDim), act);
        using Tensor oMh = new Tensor(new TensorShape(1, heads, sq, headDim), act);
        cuda.Permute0213(qMh, q, sq, heads, headDim);
        cuda.Permute0213(kMh, k, sk, heads, headDim);
        cuda.Permute0213(vMh, v, sk, heads, headDim);
        cuda.ScaledDotProductAttention(oMh, qMh, kMh, vMh, null, scale, allowF16: true);
        using Tensor expected = new Tensor(new TensorShape(sq, inner), act);
        cuda.Permute0213(expected, oMh, heads, sq, headDim);

        using Tensor actual = new Tensor(new TensorShape(sq, inner), act);
        cuda.ScaledDotProductAttentionTokenMajor(actual, q, k, v, null, heads, headDim, scale, allowF16: true);
        cuda.Sync();
        AssertRelClose(ToF32(expected), ToF32(actual), f16 ? 2e-2f : 1e-4f, "token-major SDPA");
    }

    /// <summary>The whole attention, both routes, through the SAME <c>Forward</c> — the head-major branch is the
    /// explicit reference, reached by flipping the route's kill switch rather than by re-deriving it here.</summary>
    [Theory]
    [InlineData(false, 1e-4f)]
    [InlineData(true, 2e-2f)]
    public void Forward_TokenMajorMatchesHeadMajor(bool f16, float tol)
    {
        DType act = f16 ? DType.F16 : DType.F32;
        const int sq = 37, sk = 53, heads = 8, headDim = 64, qIn = 96, kvIn = 80, outDim = 72;
        int inner = heads * headDim;
        using CudaBackend cuda = new CudaBackend(0, PtxDir());

        Dictionary<string, Tensor> w = new()
        {
            ["a.to_q.weight"] = Random(new TensorShape(inner, qIn), seed: 301, amp: 0.1f),
            ["a.to_k.weight"] = Random(new TensorShape(inner, kvIn), seed: 302, amp: 0.1f),
            ["a.to_v.weight"] = Random(new TensorShape(inner, kvIn), seed: 303, amp: 0.1f),
            ["a.to_out.0.weight"] = Random(new TensorShape(outDim, inner), seed: 304, amp: 0.1f),
            ["a.q_norm.weight"] = Random(new TensorShape(inner), seed: 305),
            ["a.k_norm.weight"] = Random(new TensorShape(inner), seed: 306),
            ["a.to_gate_logits.weight"] = Random(new TensorShape(heads, qIn), seed: 307, amp: 0.1f),
        };
        LtxVideo2Attention attn = new LtxVideo2Attention(qIn, kvIn, heads, headDim, outDim, Eps);
        attn.LoadWeights(w, "a");

        // Only Flavor is read off the rope; the fused kernels take the cos/sin tables directly.
        LtxVideo2Rope rope = LtxVideo2Rope.ForAudio(inner, theta: 10000.0, baseFrames: 128, audioScaleFactor: 1,
            causalOffset: 0, samplingRate: 16000, hopLength: 256, LtxVideo2Rope.RopeType.Split, heads, headDim);
        using Tensor qIn0 = Random(new TensorShape(sq, qIn), seed: 311, act, amp: 0.5f);
        using Tensor kvIn0 = Random(new TensorShape(sk, kvIn), seed: 312, act, amp: 0.5f);
        using Tensor qCos = Random(new TensorShape(sq, inner / 2), seed: 313);
        using Tensor qSin = Random(new TensorShape(sq, inner / 2), seed: 314);
        using Tensor kCos = Random(new TensorShape(sk, inner / 2), seed: 315);
        using Tensor kSin = Random(new TensorShape(sk, inner / 2), seed: 316);

        bool saved = LtxVideo2Attention.TokenMajorAttention;
        float[] headMajor, tokenMajor;
        try
        {
            LtxVideo2Attention.TokenMajorAttention = false;
            using Tensor a = attn.Forward(cuda, qIn0, kvIn0, rope, qCos, qSin, rope, kCos, kSin, null);
            cuda.Sync();
            headMajor = ToF32(a);
            LtxVideo2Attention.TokenMajorAttention = true;
            using Tensor b = attn.Forward(cuda, qIn0, kvIn0, rope, qCos, qSin, rope, kCos, kSin, null);
            cuda.Sync();
            tokenMajor = ToF32(b);
        }
        finally
        {
            LtxVideo2Attention.TokenMajorAttention = saved;
            foreach (Tensor t in w.Values) t.Dispose();
        }
        AssertRelClose(headMajor, tokenMajor, tol, $"LtxVideo2Attention.Forward ({act})");
    }

    private static void AssertReindexed(float[] headMajor, float[] tokenMajor, int seq, int heads, int headDim, float tol)
    {
        int inner = heads * headDim;
        double maxErr = 0;
        for (int h = 0; h < heads; h++)
            for (int s = 0; s < seq; s++)
                for (int d = 0; d < headDim; d++)
                {
                    float e = headMajor[((long)h * seq + s) * headDim + d];
                    float a = tokenMajor[(long)s * inner + h * headDim + d];
                    maxErr = Math.Max(maxErr, Math.Abs(e - a));
                }
        Assert.True(maxErr <= tol, $"token-major vs head-major re-indexed: max abs err {maxErr} > {tol}");
    }

    private static void AssertRelClose(float[] expected, float[] actual, float tol, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        double scale = 1e-6;
        for (int i = 0; i < expected.Length; i++) scale = Math.Max(scale, Math.Abs(expected[i]));
        double maxRel = 0;
        for (int i = 0; i < expected.Length; i++) maxRel = Math.Max(maxRel, Math.Abs(expected[i] - actual[i]) / scale);
        Assert.True(maxRel <= tol, $"{what}: max rel err {maxRel} > {tol}");
    }
}
