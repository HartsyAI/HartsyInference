using Xunit;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Pure-logic correctness tests for the generic reusable Lance backbone pieces — the Qwen2.5-VL <see cref="Multimodal3DRope"/> and the shared <see cref="DiTUtils.Sincos3DPositionEmbedding"/> — with no GPU/checkpoint. These guard the rotation invariants (identity at position 0, norm preservation, position sensitivity) that any model reusing them depends on.</summary>
public unsafe class LanceBackboneTests
{
    private static readonly (int, int, int) _section = (16, 24, 24); // sums to 64 = head_dim/2 for head_dim 128

    [Fact]
    public void Rope_AtPositionZero_IsIdentity()
    {
        Multimodal3DRope rope = new(headDim: 128, theta: 1_000_000.0, section: _section);
        int seq = 3, heads = 2, headDim = 128;

        Tensor pos = Zeros(seq, 3);   // all positions (t,h,w) = 0
        (Tensor cos, Tensor sin) = rope.BuildCosSin(pos);

        // cos must be all 1, sin all 0 at position 0.
        float* c = (float*)cos.DataPointer;
        float* s = (float*)sin.DataPointer;
        for (int i = 0; i < seq * headDim; i++)
        {
            Assert.Equal(1.0f, c[i], 5);
            Assert.Equal(0.0f, s[i], 5);
        }

        Tensor q = RandomTensor(seq, heads, headDim, seed: 1);
        Tensor original = Clone(q);
        rope.ApplyRotary(q, cos, sin);
        AssertTensorsClose(original, q, 1e-5f); // identity rotation
    }

    [Fact]
    public void Rope_PreservesPerHeadNorm()
    {
        Multimodal3DRope rope = new(headDim: 128, theta: 1_000_000.0, section: _section);
        int seq = 4, heads = 3, headDim = 128;

        Tensor pos = new Tensor(new TensorShape(seq, 3), DType.F32);
        float* p = (float*)pos.DataPointer;
        for (int i = 0; i < seq; i++) { p[i * 3 + 0] = i; p[i * 3 + 1] = 2 * i; p[i * 3 + 2] = 3 * i + 1; }

        (Tensor cos, Tensor sin) = rope.BuildCosSin(pos);
        Tensor q = RandomTensor(seq, heads, headDim, seed: 7);
        float[] before = PerHeadNorms(q, seq, heads, headDim);
        rope.ApplyRotary(q, cos, sin);
        float[] after = PerHeadNorms(q, seq, heads, headDim);
        for (int i = 0; i < before.Length; i++)
            Assert.Equal(before[i], after[i], 3); // rotation is orthogonal → norm preserved
    }

    [Fact]
    public void Rope_IsPositionSensitive()
    {
        Multimodal3DRope rope = new(headDim: 128, theta: 1_000_000.0, section: _section);
        int heads = 1, headDim = 128;

        Tensor posA = Zeros(1, 3);
        Tensor posB = new Tensor(new TensorShape(1, 3), DType.F32);
        float* pb = (float*)posB.DataPointer; pb[0] = 5; pb[1] = 5; pb[2] = 5;

        Tensor qA = RandomTensor(1, heads, headDim, seed: 3);
        Tensor qB = Clone(qA);
        (Tensor cosA, Tensor sinA) = rope.BuildCosSin(posA);
        (Tensor cosB, Tensor sinB) = rope.BuildCosSin(posB);
        rope.ApplyRotary(qA, cosA, sinA);
        rope.ApplyRotary(qB, cosB, sinB);

        float maxDiff = 0f;
        float* a = (float*)qA.DataPointer;
        float* b = (float*)qB.DataPointer;
        for (int i = 0; i < headDim; i++) maxDiff = MathF.Max(maxDiff, MathF.Abs(a[i] - b[i]));
        Assert.True(maxDiff > 0.1f, $"rotations at different positions should differ, max diff={maxDiff}");
    }

    [Fact]
    public void Rope_RejectsBadSection()
    {
        // (10,24,24) sums to 58 != head_dim/2 = 64.
        Assert.Throws<ArgumentException>(() => new Multimodal3DRope(128, 1e6, (10, 24, 24)));
    }

    [Fact]
    public void Sincos3D_ShapeAndZeroPosition()
    {
        int t = 2, h = 3, w = 4, dim = 48;
        Tensor pe = DiTUtils.Sincos3DPositionEmbedding(t, h, w, dim);
        Assert.Equal(t * h * w, (int)pe.Shape[0]);
        Assert.Equal(dim, (int)pe.Shape[1]);

        // Row 0 = position (0,0,0): every axis block is [sin(0)=0 ... cos(0)=1 ...].
        float* p = (float*)pe.DataPointer;
        // dim split: d=48/3=16 (even) → dimT=16, dimH=16, dimW=16; each half (8) sin then cos.
        // First 8 = sin(0)=0, next 8 = cos(0)=1 (axis t), repeats per axis.
        for (int axis = 0; axis < 3; axis++)
        {
            int baseOff = axis * 16;
            for (int k = 0; k < 8; k++) Assert.Equal(0.0f, p[baseOff + k], 5);
            for (int k = 0; k < 8; k++) Assert.Equal(1.0f, p[baseOff + 8 + k], 5);
        }
    }

    [Fact]
    public void Sincos3D_IsDeterministic()
    {
        Tensor a = DiTUtils.Sincos3DPositionEmbedding(2, 2, 2, 96);
        Tensor b = DiTUtils.Sincos3DPositionEmbedding(2, 2, 2, 96);
        AssertTensorsClose(a, b, 0f);
    }

    [Fact]
    public void LanceConfig_DerivedValues()
    {
        LanceConfig c = LanceConfig.Image;
        Assert.Equal(128, c.HeadDim);          // 2048 / 16
        Assert.Equal(48, c.PatchFeatureDim);   // 48 × 1 × 1 × 1 — real checkpoint's vae2llm is Linear(48→2048)
        Assert.Equal(8, c.NumHeads / c.NumKvHeads); // GQA factor
        Assert.True(c.QkNorm);                 // real checkpoint ships q_norm/k_norm [128]
        Assert.Equal(64, c.MaxLatentSize);     // latent_pos_embed table is 64·64 rows per frame
    }

    // ── helpers ──
    private static Tensor Zeros(int a, int b)
    {
        Tensor t = new Tensor(new TensorShape(a, b), DType.F32);
        new Span<float>((float*)t.DataPointer, a * b).Clear();
        return t;
    }

    private static Tensor RandomTensor(int seq, int heads, int headDim, int seed)
    {
        Tensor t = new Tensor(new TensorShape(seq, heads, headDim), DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new(seed);
        for (int i = 0; i < seq * heads * headDim; i++) p[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    private static Tensor Clone(Tensor src)
    {
        Tensor t = new Tensor(src.Shape, DType.F32);
        long n = src.Shape.ElementCount;
        Buffer.MemoryCopy((float*)src.DataPointer, (float*)t.DataPointer, n * 4, n * 4);
        return t;
    }

    private static float[] PerHeadNorms(Tensor q, int seq, int heads, int headDim)
    {
        float* p = (float*)q.DataPointer;
        float[] norms = new float[seq * heads];
        for (int v = 0; v < seq * heads; v++)
        {
            double sum = 0;
            for (int d = 0; d < headDim; d++) { float x = p[v * headDim + d]; sum += x * x; }
            norms[v] = (float)Math.Sqrt(sum);
        }
        return norms;
    }

    private static void AssertTensorsClose(Tensor a, Tensor b, float tol)
    {
        long n = a.Shape.ElementCount;
        float* pa = (float*)a.DataPointer;
        float* pb = (float*)b.DataPointer;
        for (long i = 0; i < n; i++)
            Assert.True(MathF.Abs(pa[i] - pb[i]) <= tol, $"mismatch at {i}: {pa[i]} vs {pb[i]}");
    }
}
