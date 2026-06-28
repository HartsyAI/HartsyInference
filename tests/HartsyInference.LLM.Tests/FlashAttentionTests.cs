using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>Parity for <see cref="IBackend.FlashAttention"/> (the GQA-aware online-softmax attention used by
/// GenericTransformer) against the reference path: replicate K/V to the query head count + a causal additive
/// mask + <see cref="IBackend.ScaledDotProductAttention"/>. Covers prefill (Tq&gt;1) and decode (Tq=1).</summary>
public sealed unsafe class FlashAttentionTests
{
    private static uint _rng = 0xA17Cu;
    private static float Rand() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return ((_rng & 0xFFFF) / 65535f - 0.5f); }
    private static Tensor Rnd(int a, int b, int c, int e) { Tensor t = new(new TensorShape(a, b, c, e), DType.F32); float* p = (float*)t.DataPointer; for (long i = 0; i < t.ElementCount; i++) p[i] = Rand(); return t; }

    [Theory]
    [InlineData(true)]   // prefill: Tq = Lk, qOffset = 0, full causal triangle
    [InlineData(false)]  // decode: Tq = 1, qOffset = Lk-1, attends whole prefix
    public void Flash_MatchesRepeatKvSdpa(bool prefill)
    {
        const int hq = 4, hkv = 2, d = 8, lk = 6, group = hq / hkv;
        int tq = prefill ? lk : 1;
        int qOffset = prefill ? 0 : lk - 1;
        float scale = 1f / MathF.Sqrt(d);

        using CpuBackend cpu = new();
        IBackend b = cpu;
        using Tensor q = Rnd(1, hq, tq, d);
        using Tensor k = Rnd(1, hkv, lk, d);
        using Tensor v = Rnd(1, hkv, lk, d);

        // Reference: replicate K/V to Hq, build a causal mask, run SDPA.
        using Tensor kRep = new(new TensorShape(1, hq, lk, d), DType.F32);
        using Tensor vRep = new(new TensorShape(1, hq, lk, d), DType.F32);
        b.RepeatKvHeads(kRep, k, hkv, group);
        b.RepeatKvHeads(vRep, v, hkv, group);
        using Tensor mask = new(new TensorShape(1, 1, tq, lk), DType.F32);
        float* mp = (float*)mask.DataPointer;
        for (int r = 0; r < tq; r++)
            for (int c = 0; c < lk; c++)
                mp[r * lk + c] = c <= qOffset + r ? 0f : -1e30f;
        using Tensor refOut = new(new TensorShape(1, hq, tq, d), DType.F32);
        b.ScaledDotProductAttention(refOut, q, kRep, vRep, mask, scale);

        using Tensor flashOut = new(new TensorShape(1, hq, tq, d), DType.F32);
        b.FlashAttention(flashOut, q, k, v, lk, group, causal: true, qOffset, scale);

        float* a = (float*)refOut.DataPointer;
        float* f = (float*)flashOut.DataPointer;
        float maxDiff = 0f;
        for (long i = 0; i < refOut.ElementCount; i++) maxDiff = MathF.Max(maxDiff, MathF.Abs(a[i] - f[i]));
        Assert.True(maxDiff <= 1e-4f, $"FlashAttention diverges from SDPA by {maxDiff:E3} (prefill={prefill}).");
    }

    /// <summary>GPT-OSS attention sink: each head carries a learned logit that joins the softmax denominator
    /// but contributes no value. Validates <see cref="IBackend.FlashAttention"/>'s sink path against a direct
    /// per-row softmax([scores, sink]) reference, and checks the two limits: a hugely-negative sink reproduces
    /// the no-sink output, and a dominant positive sink bleeds the output toward zero.</summary>
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void Flash_Sink_MatchesAugmentedSoftmax(bool prefill)
    {
        const int hq = 4, hkv = 2, d = 8, lk = 6, group = hq / hkv;
        int tq = prefill ? lk : 1;
        int qOffset = prefill ? 0 : lk - 1;
        float scale = 1f / MathF.Sqrt(d);

        using CpuBackend cpu = new();
        IBackend b = cpu;
        using Tensor q = Rnd(1, hq, tq, d);
        using Tensor k = Rnd(1, hkv, lk, d);
        using Tensor v = Rnd(1, hkv, lk, d);
        using Tensor sink = new(new TensorShape(hq), DType.F32);
        float* sp = (float*)sink.DataPointer;
        for (int h = 0; h < hq; h++) sp[h] = Rand() * 4f;   // per-head sink logits

        // Reference: explicit softmax over [scores_0..scores_kMax, sink_h] with the sink dropped from the value sum.
        float* qp = (float*)q.DataPointer, kp = (float*)k.DataPointer, vp = (float*)v.DataPointer;
        using Tensor refOut = new(new TensorShape(1, hq, tq, d), DType.F32);
        float* rp = (float*)refOut.DataPointer;
        float* acc = stackalloc float[d];
        for (int h = 0; h < hq; h++)
            for (int r = 0; r < tq; r++)
            {
                int hkvIdx = h / group, kMax = qOffset + r;
                float m = sp[h];
                for (int c = 0; c <= kMax; c++)
                {
                    float s = 0f; for (int x = 0; x < d; x++) s += qp[((h * tq) + r) * d + x] * kp[((hkvIdx * lk) + c) * d + x];
                    m = MathF.Max(m, s * scale);
                }
                float z = MathF.Exp(sp[h] - m);
                for (int x = 0; x < d; x++) acc[x] = 0f;
                for (int c = 0; c <= kMax; c++)
                {
                    float s = 0f; for (int x = 0; x < d; x++) s += qp[((h * tq) + r) * d + x] * kp[((hkvIdx * lk) + c) * d + x];
                    float p = MathF.Exp(s * scale - m); z += p;
                    for (int x = 0; x < d; x++) acc[x] += p * vp[((hkvIdx * lk) + c) * d + x];
                }
                for (int x = 0; x < d; x++) rp[((h * tq) + r) * d + x] = acc[x] / z;
            }

        using Tensor flashOut = new(new TensorShape(1, hq, tq, d), DType.F32);
        b.FlashAttention(flashOut, q, k, v, lk, group, causal: true, qOffset, scale, softcap: 0f, sink);
        float* fp = (float*)flashOut.DataPointer;
        float maxDiff = 0f;
        for (long i = 0; i < refOut.ElementCount; i++) maxDiff = MathF.Max(maxDiff, MathF.Abs(rp[i] - fp[i]));
        Assert.True(maxDiff <= 1e-4f, $"FlashAttention sink diverges from augmented softmax by {maxDiff:E3} (prefill={prefill}).");

        // Limit 1: a hugely-negative sink contributes ~0 to the denominator → identical to the no-sink path.
        using Tensor noSink = new(new TensorShape(1, hq, tq, d), DType.F32);
        b.FlashAttention(noSink, q, k, v, lk, group, causal: true, qOffset, scale);
        using Tensor negSinkT = new(new TensorShape(hq), DType.F32);
        float* np = (float*)negSinkT.DataPointer; for (int h = 0; h < hq; h++) np[h] = -1e30f;
        using Tensor negSinkOut = new(new TensorShape(1, hq, tq, d), DType.F32);
        b.FlashAttention(negSinkOut, q, k, v, lk, group, causal: true, qOffset, scale, softcap: 0f, negSinkT);
        float* op = (float*)noSink.DataPointer, gp = (float*)negSinkOut.DataPointer;
        float negDiff = 0f; for (long i = 0; i < noSink.ElementCount; i++) negDiff = MathF.Max(negDiff, MathF.Abs(op[i] - gp[i]));
        Assert.True(negDiff <= 1e-5f, $"Hugely-negative sink should match no-sink, diff {negDiff:E3}.");

        // Limit 2: a dominant positive sink captures nearly all the mass → outputs collapse toward zero.
        for (int h = 0; h < hq; h++) np[h] = 60f;
        using Tensor bigSinkOut = new(new TensorShape(1, hq, tq, d), DType.F32);
        b.FlashAttention(bigSinkOut, q, k, v, lk, group, causal: true, qOffset, scale, softcap: 0f, negSinkT);
        float* bp = (float*)bigSinkOut.DataPointer;
        float maxAbs = 0f; for (long i = 0; i < bigSinkOut.ElementCount; i++) maxAbs = MathF.Max(maxAbs, MathF.Abs(bp[i]));
        Assert.True(maxAbs <= 1e-3f, $"Dominant sink should drive outputs toward zero, got max |out| {maxAbs:E3}.");
    }
}
