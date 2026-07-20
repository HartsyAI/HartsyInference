using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.ZipVoice;

/// <summary>One <c>Zipformer2EncoderLayer</c>. Exact eval-time sub-block order (verified directly against
/// <c>zipvoice/models/modules/zipformer.py:Zipformer2EncoderLayer.forward</c>, all training-only dropout/skip
/// branches omitted as they're unconditionally no-ops when not training):
/// <code>
/// srcOrig = src
/// attnWeights = self_attn_weights(src, posEmb)              // [H,T,T], computed ONCE, reused below
/// src += timeEmb                                            // (fm_decoder only; text_encoder passes null)
/// src += feed_forward1(src)
/// src += nonlin_attention(src, attnWeights[0:1])
/// src += self_attn1(src, attnWeights)
/// src += timeEmb
/// src += conv_module1(src)
/// src += feed_forward2(src)
/// src = bypass_mid(srcOrig, src)                            // highway vs the ORIGINAL layer input
/// src += self_attn2(src, attnWeights)                       // reuses the SAME attnWeights
/// src += timeEmb
/// src += conv_module2(src)
/// src += feed_forward3(src)
/// src = norm(src)                                           // BiasNorm, applied ONCE here
/// src = bypass(srcOrig, src)                                // final highway, also vs the ORIGINAL input
/// </code></summary>
internal sealed unsafe class ZipformerEncoderLayer
{
    private readonly int _dim;
    private readonly ZipformerAttentionWeights _attnWeights;
    private readonly ZipformerSelfAttentionValue _selfAttn1, _selfAttn2;
    private readonly ZipformerNonlinAttention _nonlinAttention;
    private readonly ZipformerFeedForward _ff1, _ff2, _ff3;
    private readonly ZipformerConvModule _conv1, _conv2;
    private readonly ZipformerBiasNorm _norm;
    private readonly ZipformerBypass _bypassMid, _bypass;

    public ZipformerEncoderLayer(int dim, int numHeads, int queryHeadDim, int posHeadDim, int posDim,
        int valueHeadDim, int feedforwardDim, int cnnKernel)
    {
        _dim = dim;
        _attnWeights = new ZipformerAttentionWeights(dim, numHeads, queryHeadDim, posHeadDim, posDim);
        _selfAttn1 = new ZipformerSelfAttentionValue(dim, numHeads, valueHeadDim);
        _selfAttn2 = new ZipformerSelfAttentionValue(dim, numHeads, valueHeadDim);
        _nonlinAttention = new ZipformerNonlinAttention(dim);
        _ff1 = new ZipformerFeedForward(dim, feedforwardDim * 3 / 4);
        _ff2 = new ZipformerFeedForward(dim, feedforwardDim);
        _ff3 = new ZipformerFeedForward(dim, feedforwardDim * 5 / 4);
        _conv1 = new ZipformerConvModule(dim, cnnKernel);
        _conv2 = new ZipformerConvModule(dim, cnnKernel);
        _norm = new ZipformerBiasNorm(dim);
        _bypassMid = new ZipformerBypass(dim);
        _bypass = new ZipformerBypass(dim);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _attnWeights.LoadWeights(w, $"{prefix}.self_attn_weights");
        _selfAttn1.LoadWeights(w, $"{prefix}.self_attn1");
        _selfAttn2.LoadWeights(w, $"{prefix}.self_attn2");
        _nonlinAttention.LoadWeights(w, $"{prefix}.nonlin_attention");
        _ff1.LoadWeights(w, $"{prefix}.feed_forward1");
        _ff2.LoadWeights(w, $"{prefix}.feed_forward2");
        _ff3.LoadWeights(w, $"{prefix}.feed_forward3");
        _conv1.LoadWeights(w, $"{prefix}.conv_module1");
        _conv2.LoadWeights(w, $"{prefix}.conv_module2");
        _norm.LoadWeights(w, $"{prefix}.norm");
        _bypassMid.LoadWeights(w, $"{prefix}.bypass_mid.bypass_scale");
        _bypass.LoadWeights(w, $"{prefix}.bypass.bypass_scale");
    }

    /// <summary><paramref name="src"/> is <c>[1, T, dim]</c> (NOT disposed by this method — caller owns it).
    /// <paramref name="stageTimeEmb"/> is <c>[dim]</c> (already <c>SwooshR→Linear</c> stage-projected) or null
    /// for the text encoder. Returns a new <c>[1, T, dim]</c> tensor.</summary>
    public Tensor Forward(IBackend backend, Tensor src, Tensor posEmb, Tensor? stageTimeEmb, int t)
    {
        Tensor attnWeights = _attnWeights.Forward(backend, src, posEmb, t);
        Tensor headZero = SliceHead0(attnWeights, t);

        Tensor cur = stageTimeEmb is not null ? AddBroadcast(src, stageTimeEmb, t) : Copy(src, t);

        AddInPlace(cur, _ff1.Forward(backend, cur, t), t);
        AddInPlace(cur, _nonlinAttention.Forward(backend, cur, headZero, t), t);
        headZero.Dispose();
        AddInPlace(cur, _selfAttn1.Forward(backend, cur, attnWeights, t), t);

        if (stageTimeEmb is not null) AddBroadcastInPlace(cur, stageTimeEmb, t);
        AddInPlace(cur, _conv1.Forward(backend, cur, t), t);
        AddInPlace(cur, _ff2.Forward(backend, cur, t), t);

        Tensor afterMid = _bypassMid.Forward(src, cur, t);
        cur.Dispose();
        cur = afterMid;

        AddInPlace(cur, _selfAttn2.Forward(backend, cur, attnWeights, t), t);
        attnWeights.Dispose();

        if (stageTimeEmb is not null) AddBroadcastInPlace(cur, stageTimeEmb, t);
        AddInPlace(cur, _conv2.Forward(backend, cur, t), t);
        AddInPlace(cur, _ff3.Forward(backend, cur, t), t);

        Tensor normed = _norm.Forward(cur, t);
        cur.Dispose();

        Tensor output = _bypass.Forward(src, normed, t);
        normed.Dispose();
        return output;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _attnWeights.EnumerateWeights()) yield return t;
        foreach (Tensor t in _selfAttn1.EnumerateWeights()) yield return t;
        foreach (Tensor t in _selfAttn2.EnumerateWeights()) yield return t;
        foreach (Tensor t in _nonlinAttention.EnumerateWeights()) yield return t;
        foreach (Tensor t in _ff1.EnumerateWeights()) yield return t;
        foreach (Tensor t in _ff2.EnumerateWeights()) yield return t;
        foreach (Tensor t in _ff3.EnumerateWeights()) yield return t;
        foreach (Tensor t in _conv1.EnumerateWeights()) yield return t;
        foreach (Tensor t in _conv2.EnumerateWeights()) yield return t;
        foreach (Tensor t in _norm.EnumerateWeights()) yield return t;
        foreach (Tensor t in _bypassMid.EnumerateWeights()) yield return t;
        foreach (Tensor t in _bypass.EnumerateWeights()) yield return t;
    }

    private Tensor SliceHead0(Tensor attnWeights, int t)
    {
        Tensor head0 = new(new TensorShape(1, t, t), DType.F32);
        Buffer.MemoryCopy((float*)attnWeights.DataPointer, (float*)head0.DataPointer,
            (long)t * t * sizeof(float), (long)t * t * sizeof(float));
        return head0;
    }

    private Tensor Copy(Tensor x, int t)
    {
        Tensor output = new(new TensorShape(1, t, _dim), DType.F32);
        Buffer.MemoryCopy((float*)x.DataPointer, (float*)output.DataPointer,
            (long)t * _dim * sizeof(float), (long)t * _dim * sizeof(float));
        return output;
    }

    private Tensor AddBroadcast(Tensor x, Tensor bias, int t)
    {
        Tensor output = new(new TensorShape(1, t, _dim), DType.F32);
        float* xp = (float*)x.DataPointer;
        float* bp = (float*)bias.DataPointer;
        float* op = (float*)output.DataPointer;
        for (int i = 0; i < t; i++)
        {
            long off = (long)i * _dim;
            for (int d = 0; d < _dim; d++) op[off + d] = xp[off + d] + bp[d];
        }
        return output;
    }

    private void AddBroadcastInPlace(Tensor x, Tensor bias, int t)
    {
        float* xp = (float*)x.DataPointer;
        float* bp = (float*)bias.DataPointer;
        for (int i = 0; i < t; i++)
        {
            long off = (long)i * _dim;
            for (int d = 0; d < _dim; d++) xp[off + d] += bp[d];
        }
    }

    private static void AddInPlace(Tensor dst, Tensor src, int t)
    {
        float* dp = (float*)dst.DataPointer;
        float* sp = (float*)src.DataPointer;
        long n = dst.ElementCount;
        for (long i = 0; i < n; i++) dp[i] += sp[i];
        src.Dispose();
    }
}
