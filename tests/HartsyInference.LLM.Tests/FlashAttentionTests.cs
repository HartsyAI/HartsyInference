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
}
