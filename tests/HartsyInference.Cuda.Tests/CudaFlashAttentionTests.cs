using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Validates the CUDA FlashAttention kernel against the reference (replicated K/V + causal mask +
/// SDPA), for both prefill and decode shapes and a GQA head ratio.</summary>
[Collection("CudaSerial")]
public sealed unsafe class CudaFlashAttentionTests
{
    private readonly ITestOutputHelper _output;
    public CudaFlashAttentionTests(ITestOutputHelper output) => _output = output;

    private static uint _rng = 0x9E3Du;
    private static float Rand() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return ((_rng & 0xFFFF) / 65535f - 0.5f); }
    private static Tensor Rnd(int a, int b, int c, int d) { Tensor t = new(new TensorShape(a, b, c, d), DType.F32); float* p = (float*)t.DataPointer; for (long i = 0; i < t.ElementCount; i++) p[i] = Rand(); return t; }

    [Theory]
    [InlineData(true)]   // prefill
    [InlineData(false)]  // decode
    public void Flash_MatchesSdpa(bool prefill)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");

        const int hq = 8, hkv = 2, d = 64, lk = 7, group = hq / hkv;
        int tq = prefill ? lk : 1;
        int qOffset = prefill ? 0 : lk - 1;
        float scale = 1f / MathF.Sqrt(d);

        using CudaBackend backend = new(0, ptxDir);
        IBackend b = backend;
        using Tensor q = Rnd(1, hq, tq, d);
        using Tensor k = Rnd(1, hkv, lk, d);
        using Tensor v = Rnd(1, hkv, lk, d);

        using Tensor kRep = new(new TensorShape(1, hq, lk, d), DType.F32);
        using Tensor vRep = new(new TensorShape(1, hq, lk, d), DType.F32);
        b.RepeatKvHeads(kRep, k, hkv, group);
        b.RepeatKvHeads(vRep, v, hkv, group);
        using Tensor mask = new(new TensorShape(1, 1, tq, lk), DType.F32);
        float* mp = (float*)mask.DataPointer;
        for (int r = 0; r < tq; r++) for (int c = 0; c < lk; c++) mp[r * lk + c] = c <= qOffset + r ? 0f : -1e30f;
        using Tensor refOut = new(new TensorShape(1, hq, tq, d), DType.F32);
        b.ScaledDotProductAttention(refOut, q, kRep, vRep, mask, scale);
        backend.Sync();

        using Tensor flashOut = new(new TensorShape(1, hq, tq, d), DType.F32);
        b.FlashAttention(flashOut, q, k, v, lk, group, causal: true, qOffset, scale);
        backend.Sync();

        float* a = (float*)refOut.DataPointer;
        float* f = (float*)flashOut.DataPointer;
        float maxDiff = 0f;
        for (long i = 0; i < refOut.ElementCount; i++) maxDiff = MathF.Max(maxDiff, MathF.Abs(a[i] - f[i]));
        _output.WriteLine($"prefill={prefill}: max |SDPA - Flash| = {maxDiff:E3}");
        Assert.True(maxDiff <= 2e-3f, $"CUDA FlashAttention diverges from SDPA by {maxDiff:E3} (prefill={prefill}).");
    }

    /// <summary>Vision-tower regime: non-power-of-two head_dim (SigLIP/Gemma-3 = 72) and bidirectional
    /// (non-causal) attention over many patch tokens. Exercises the block-padding path that the decode tests
    /// (d=64, causal) never hit. Compared against SDPA with an all-visible mask.</summary>
    [Theory]
    [InlineData(72, 16, 64)]   // Gemma-3 SigLIP: head_dim 72, 16 heads
    [InlineData(80, 4, 33)]    // another non-pow2 head_dim, odd key count
    public void Flash_NonPow2HeadDim_NonCausal_MatchesSdpa(int d, int heads, int lk)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");

        float scale = 1f / MathF.Sqrt(d);
        using CudaBackend backend = new(0, ptxDir);
        IBackend b = backend;
        using Tensor q = Rnd(1, heads, lk, d);
        using Tensor k = Rnd(1, heads, lk, d);
        using Tensor v = Rnd(1, heads, lk, d);

        using Tensor mask = new(new TensorShape(1, 1, lk, lk), DType.F32);   // all visible (bidirectional)
        float* mp = (float*)mask.DataPointer;
        for (long i = 0; i < mask.ElementCount; i++) mp[i] = 0f;
        using Tensor refOut = new(new TensorShape(1, heads, lk, d), DType.F32);
        b.ScaledDotProductAttention(refOut, q, k, v, mask, scale);
        backend.Sync();

        using Tensor flashOut = new(new TensorShape(1, heads, lk, d), DType.F32);
        b.FlashAttention(flashOut, q, k, v, lk, 1, causal: false, qOffset: 0, scale);
        backend.Sync();

        float* a = (float*)refOut.DataPointer;
        float* f = (float*)flashOut.DataPointer;
        float maxDiff = 0f;
        for (long i = 0; i < refOut.ElementCount; i++) maxDiff = MathF.Max(maxDiff, MathF.Abs(a[i] - f[i]));
        _output.WriteLine($"d={d} heads={heads} lk={lk}: max |SDPA - Flash| = {maxDiff:E3}");
        Assert.True(maxDiff <= 2e-3f, $"CUDA non-causal FlashAttention (d={d}) diverges from SDPA by {maxDiff:E3}.");
    }

    /// <summary>FixedKvCache case: the K/V buffer's seq stride (maxSeq) exceeds the valid key count, and the
    /// head config matches Llama-3.2 (Hq=32, Hkv=8, group=4, headDim=64). Flash must read only the first kvLen
    /// keys at the buffer stride. Compared against an inline online-softmax reference.</summary>
    [Fact]
    public void Flash_StridedBuffer_LlamaGqa_MatchesReference()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");

        const int hq = 32, hkv = 8, d = 64, group = hq / hkv, tq = 12, kvLen = 12, stride = 40;
        float scale = 1f / MathF.Sqrt(d);

        using CudaBackend backend = new(0, ptxDir);
        using Tensor q = Rnd(1, hq, tq, d);
        using Tensor kBuf = Rnd(1, hkv, stride, d);   // full buffer; only first kvLen rows are valid
        using Tensor vBuf = Rnd(1, hkv, stride, d);
        using Tensor outT = new(new TensorShape(1, hq, tq, d), DType.F32);
        ((IBackend)backend).FlashAttention(outT, q, kBuf, vBuf, kvLen, group, causal: true, qOffset: 0, scale);
        backend.Sync();

        float* qp = (float*)q.DataPointer, kp = (float*)kBuf.DataPointer, vp = (float*)vBuf.DataPointer, op = (float*)outT.DataPointer;
        float max = 0f;
        for (int h = 0; h < hq; h++)
        {
            int kvh = h / group;
            for (int r = 0; r < tq; r++)
            {
                int kMax = Math.Min(kvLen - 1, r);          // causal, qOffset 0
                float m = float.NegativeInfinity, denom = 0f;
                float[] acc = new float[d];
                for (int k = 0; k <= kMax; k++)
                {
                    float dot = 0f;
                    long qBase = ((long)h * tq + r) * d;
                    long kBase = ((long)kvh * stride + k) * d;   // buffer stride
                    for (int e = 0; e < d; e++) dot += qp[qBase + e] * kp[kBase + e];
                    dot *= scale;
                    float nm = MathF.Max(m, dot);
                    float corr = m <= float.NegativeInfinity ? 0f : MathF.Exp(m - nm);
                    float w = MathF.Exp(dot - nm);
                    denom = denom * corr + w;
                    long vBase = ((long)kvh * stride + k) * d;
                    for (int e = 0; e < d; e++) acc[e] = acc[e] * corr + w * vp[vBase + e];
                    m = nm;
                }
                long oBase = ((long)h * tq + r) * d;
                for (int e = 0; e < d; e++) max = MathF.Max(max, MathF.Abs(acc[e] / denom - op[oBase + e]));
            }
        }
        _output.WriteLine($"strided Llama-GQA: max |ref - flash| = {max:E3}");
        Assert.True(max <= 2e-3f, $"CUDA strided FlashAttention diverges from reference by {max:E3}");
    }
}
