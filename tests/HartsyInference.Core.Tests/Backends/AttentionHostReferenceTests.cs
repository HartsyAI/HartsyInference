using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests.Backends;

/// <summary>The host contract reference (the default <see cref="IBackend.Attention"/>) against hand-built cases and against
/// <see cref="AttentionReference.FlashAttention"/>, which shipping code already trusts.</summary>
public sealed unsafe class AttentionHostReferenceTests
{
    private static Tensor Rand(TensorShape shape, int seed)
    {
        Tensor t = new(shape, DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new(seed);
        for (long i = 0; i < shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    private static float MaxDiff(Tensor a, Tensor b)
    {
        float* x = (float*)a.DataPointer, y = (float*)b.DataPointer;
        float m = 0;
        for (long i = 0; i < a.ElementCount; i++) m = MathF.Max(m, MathF.Abs(x[i] - y[i]));
        return m;
    }

    [Theory]
    [InlineData(false, 0, 0f, false, false)]
    [InlineData(true, 0, 0f, false, false)]
    [InlineData(true, 5, 0f, false, false)]
    [InlineData(true, 0, 20f, false, false)]
    [InlineData(true, 0, 0f, true, false)]
    [InlineData(true, 0, 0f, false, true)]
    public void MatchesTheFlashReference_GroupedCachedCausalWindowSoftcapSinkAlibi(bool causal, int window, float softcap, bool sink, bool alibi)
    {
        const int hq = 4, hkv = 2, tq = 3, lk = 16, kvLen = 12, d = 8, qOffset = 9;
        using Tensor q = Rand(new TensorShape(1, hq, tq, d), 1), k = Rand(new TensorShape(1, hkv, lk, d), 2);
        using Tensor v = Rand(new TensorShape(1, hkv, lk, d), 3);
        using Tensor expected = new(new TensorShape(1, hq, tq, d), DType.F32), actual = new(new TensorShape(1, hq, tq, d), DType.F32);
        using Tensor? sinks = sink ? Rand(new TensorShape(hq), 4) : null;
        using Tensor? slopes = alibi ? Rand(new TensorShape(hq), 5) : null;
        float scale = 1f / MathF.Sqrt(d);
        AttentionReference.FlashAttention(expected, q, k, v, kvLen, hq / hkv, causal, qOffset, scale, softcap, sinks, window, slopes);
        AttentionHostReference.Run(AttentionCall.FromFlashAttention(actual, q, k, v, kvLen, causal, qOffset, scale, softcap, sinks, window, slopes));
        Assert.True(MaxDiff(expected, actual) < 1e-5f, $"max diff {MaxDiff(expected, actual)}");
    }

    [Fact]
    public void AFullyExcludedRow_WritesZeros()
    {
        using Tensor q = Rand(new TensorShape(1, 1, 2, 4), 1), k = Rand(new TensorShape(1, 1, 3, 4), 2), v = Rand(new TensorShape(1, 1, 3, 4), 3);
        using Tensor o = new(new TensorShape(1, 1, 2, 4), DType.F32);
        using Tensor bias = new(new TensorShape(1, 1, 2, 3), DType.F32);
        float* bp = (float*)bias.DataPointer;
        for (int j = 0; j < 3; j++) bp[j] = float.NegativeInfinity;   // query row 0 sees nothing
        AttentionHostReference.Run(AttentionCall.FromScaledDotProductAttention(o, q, k, v, bias, 0.5f, allowF16: false));
        float* op = (float*)o.DataPointer;
        for (int c = 0; c < 4; c++) Assert.Equal(0f, op[c]);
        Assert.NotEqual(0f, op[4]);
    }

    [Fact]
    public void AFiniteSentinel_KeepsItsProbability()
    {
        using Tensor q = Rand(new TensorShape(1, 1, 1, 4), 1), k = Rand(new TensorShape(1, 1, 2, 4), 2);
        using Tensor v = new(new TensorShape(1, 1, 2, 4), DType.F32);
        float* vp = (float*)v.DataPointer;
        for (int c = 0; c < 4; c++) { vp[c] = 0f; vp[4 + c] = 1f; }
        using Tensor o = new(new TensorShape(1, 1, 1, 4), DType.F32);
        using Tensor bias = new(new TensorShape(1, 1, 1, 2), DType.F32);
        ((float*)bias.DataPointer)[1] = MathF.Log(1e-4f);   // a finite regional bias, not an exclusion
        AttentionHostReference.Run(AttentionCall.FromScaledDotProductAttention(o, q, k, v, bias, 0f, allowF16: false));
        float expected = 1e-4f / (1f + 1e-4f);
        Assert.Equal(expected, ((float*)o.DataPointer)[0], 6);
    }

    [Fact]
    public void TokenMajorAndHeadMajorViews_Agree()
    {
        const int s = 5, h = 3, d = 4;
        using Tensor qh = Rand(new TensorShape(1, h, s, d), 1), kh = Rand(new TensorShape(1, h, s, d), 2), vh = Rand(new TensorShape(1, h, s, d), 3);
        using Tensor qt = Transpose(qh, s, h, d), kt = Transpose(kh, s, h, d), vt = Transpose(vh, s, h, d);
        using Tensor oh = new(new TensorShape(1, h, s, d), DType.F32), ot = new(new TensorShape(1, s, h, d), DType.F32);
        AttentionHostReference.Run(AttentionCall.FromScaledDotProductAttention(oh, qh, kh, vh, null, 0.5f, false));
        AttentionHostReference.Run(AttentionCall.FromTokenMajor(ot, qt, kt, vt, null, h, h, d, 0.5f, false));
        using Tensor back = Transpose(ot, h, s, d);   // [S,H,D] -> [H,S,D]
        Assert.True(MaxDiff(oh, back) < 1e-6f);
    }

    [Fact]
    public void NaN_Propagates_EvenWhenEveryScoreOfARowIsNaN()
    {
        using Tensor q = Rand(new TensorShape(1, 1, 3, 4), 1), k = Rand(new TensorShape(1, 1, 5, 4), 2), v = Rand(new TensorShape(1, 1, 5, 4), 3);
        ((float*)q.DataPointer)[4] = float.NaN;   // query row 1: every score is NaN
        using Tensor o = new(new TensorShape(1, 1, 3, 4), DType.F32);
        float[] lse = new float[3];
        AttentionHostReference.Run(AttentionCall.FromScaledDotProductAttention(o, q, k, v, null, 0.5f, false), lse);
        float* op = (float*)o.DataPointer;
        for (int c = 0; c < 4; c++)
        {
            Assert.True(float.IsFinite(op[c]) && float.IsFinite(op[8 + c]));
            Assert.True(float.IsNaN(op[4 + c]));
        }
        Assert.True(float.IsNaN(lse[1]) && float.IsFinite(lse[0]));

        // A NaN sink on rows that see no key: NaN, not the zero an excluded row writes.
        using Tensor sink = new(new TensorShape(1), DType.F32);
        ((float*)sink.DataPointer)[0] = float.NaN;
        AttentionHostReference.Run(AttentionCall.FromFlashAttention(o, q, k, v, kvLen: 0, causal: false, qOffset: 0, scale: 0.5f,
            softcap: 0f, sink: sink, slidingWindow: 0, alibiSlopes: null), lse);
        for (int i = 0; i < 12; i++) Assert.True(float.IsNaN(op[i]));
        Assert.All(lse, x => Assert.True(float.IsNaN(x)));

        // A +∞ sink makes the row NaN whether or not it sees a key (contract §3).
        ((float*)sink.DataPointer)[0] = float.PositiveInfinity;
        foreach (int kvLen in (int[])[5, 0])
        {
            new Span<float>(op, 12).Clear();
            Array.Clear(lse);
            AttentionHostReference.Run(AttentionCall.FromFlashAttention(o, q, k, v, kvLen, causal: false, qOffset: 0,
                scale: 0.5f, softcap: 0f, sink: sink, slidingWindow: 0, alibiSlopes: null), lse);
            for (int i = 0; i < 12; i++) Assert.True(float.IsNaN(op[i]), $"kvLen {kvLen}, element {i}");
            Assert.All(lse, x => Assert.True(float.IsNaN(x)));
        }
    }

    [Fact]
    public void AnEmptyCache_AttendsNothing()
    {
        using Tensor q = Rand(new TensorShape(1, 2, 1, 4), 1), k = Rand(new TensorShape(1, 2, 8, 4), 2), v = Rand(new TensorShape(1, 2, 8, 4), 3);
        using Tensor o = Rand(new TensorShape(1, 2, 1, 4), 4);
        AttentionHostReference.Run(AttentionCall.FromFlashAttention(o, q, k, v, kvLen: 0, causal: false, qOffset: 0, scale: 0.5f,
            softcap: 0f, sink: null, slidingWindow: 0, alibiSlopes: null));
        float* op = (float*)o.DataPointer;
        for (int i = 0; i < 8; i++) Assert.Equal(0f, op[i]);
    }

    [Fact]
    public void HeadsThatDoNotDivide_AreRefused()
    {
        using Tensor q = Rand(new TensorShape(1, 6, 2, 4), 1), k = Rand(new TensorShape(1, 4, 3, 4), 2), v = Rand(new TensorShape(1, 4, 3, 4), 3);
        using Tensor o = new(new TensorShape(1, 6, 2, 4), DType.F32);
        Assert.Throws<ArgumentException>(() =>
            AttentionHostReference.Run(AttentionCall.FromScaledDotProductAttention(o, q, k, v, null, 0.5f, false)));
    }

    [Fact]
    public void ABiasBroadcastOverKeys_MatchesItsExpandedForm()
    {
        const int sq = 3, skv = 5;
        using Tensor q = Rand(new TensorShape(1, 2, sq, 4), 1), k = Rand(new TensorShape(1, 2, skv, 4), 2), v = Rand(new TensorShape(1, 2, skv, 4), 3);
        using Tensor perRow = Rand(new TensorShape(1, 1, sq, 1), 4);
        using Tensor full = new(new TensorShape(1, 1, sq, skv), DType.F32);
        float* src = (float*)perRow.DataPointer, dst = (float*)full.DataPointer;
        for (int r = 0; r < sq; r++)
            for (int j = 0; j < skv; j++) dst[r * skv + j] = src[r];
        using Tensor a = new(new TensorShape(1, 2, sq, 4), DType.F32), b = new(new TensorShape(1, 2, sq, 4), DType.F32);
        AttentionHostReference.Run(AttentionCall.FromScaledDotProductAttention(a, q, k, v, perRow, 0.5f, false));
        AttentionHostReference.Run(AttentionCall.FromScaledDotProductAttention(b, q, k, v, full, 0.5f, false));
        Assert.True(MaxDiff(a, b) < 1e-6f);
    }

    [Fact]
    public void ACausalDenseCall_IsNotCarriedByTheLegacyEntry()
    {
        using Tensor q = Rand(new TensorShape(1, 1, 2, 4), 1), k = Rand(new TensorShape(1, 1, 2, 4), 2), v = Rand(new TensorShape(1, 1, 2, 4), 3);
        using Tensor o = new(new TensorShape(1, 1, 2, 4), DType.F32);
        AttentionCall call = AttentionCall.FromScaledDotProductAttention(o, q, k, v, null, 0.5f, false);
        Assert.Null(call.LegacyGap());
        Assert.NotNull((call with { Mask = call.Mask with { Causal = true } }).LegacyGap());
    }

    /// <summary>Swaps the two middle axes of a <c>[1, A, B, D]</c> tensor.</summary>
    private static Tensor Transpose(Tensor x, int a, int b, int d)
    {
        int srcA = (int)x.Shape[1], srcB = (int)x.Shape[2];
        Tensor t = new(new TensorShape(1, srcB, srcA, d), DType.F32);
        float* src = (float*)x.DataPointer, dst = (float*)t.DataPointer;
        for (int i = 0; i < srcA; i++)
            for (int j = 0; j < srcB; j++)
                for (int c = 0; c < d; c++)
                    dst[((long)j * srcA + i) * d + c] = src[((long)i * srcB + j) * d + c];
        return t;
    }
}
