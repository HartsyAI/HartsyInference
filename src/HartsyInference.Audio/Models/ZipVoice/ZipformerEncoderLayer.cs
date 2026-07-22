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

    /// <summary>Cached <c>[1, dim]</c> ones row for the GPU-resident "a + 1·b" / "a + 1·bias(broadcast)"
    /// idioms (<see cref="IBackend.GatedResidualLastDim"/> / <see cref="IBackend.AffineBroadcastLastDim"/>) —
    /// same pattern used throughout the DiT blocks (e.g. <c>LtxVideoBlock.OnesRow</c>).</summary>
    private Tensor? _onesDim;

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

        Tensor cur = new(new TensorShape(1, t, _dim), DType.F32);
        if (stageTimeEmb is not null) backend.AffineBroadcastLastDim(cur, src, OnesDim(), stageTimeEmb);
        else backend.CopyTo(cur, src);

        AddInPlace(backend, cur, _ff1.Forward(backend, cur, t));
        AddInPlace(backend, cur, _nonlinAttention.Forward(backend, cur, headZero, t));
        headZero.Dispose();
        AddInPlace(backend, cur, _selfAttn1.Forward(backend, cur, attnWeights, t));

        if (stageTimeEmb is not null) AddBroadcastInPlace(backend, cur, stageTimeEmb);
        AddInPlace(backend, cur, _conv1.Forward(backend, cur, t));
        AddInPlace(backend, cur, _ff2.Forward(backend, cur, t));

        Tensor afterMid = _bypassMid.Forward(backend, src, cur, t);
        cur.Dispose();
        cur = afterMid;

        AddInPlace(backend, cur, _selfAttn2.Forward(backend, cur, attnWeights, t));
        attnWeights.Dispose();

        if (stageTimeEmb is not null) AddBroadcastInPlace(backend, cur, stageTimeEmb);
        AddInPlace(backend, cur, _conv2.Forward(backend, cur, t));
        AddInPlace(backend, cur, _ff3.Forward(backend, cur, t));

        Tensor normed = _norm.Forward(cur, t);
        cur.Dispose();

        Tensor output = _bypass.Forward(backend, src, normed, t);
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

    /// <summary>GPU-resident in-place residual add: <c>dst += src</c>, via <see cref="IBackend.GatedResidualLastDim"/>
    /// with a constant ones gate (<c>dst = dst + 1·src</c>) — replaces a host <c>float*</c> loop that forced a
    /// device→host sync on every sub-block call (was the dominant per-layer round-trip count).</summary>
    private void AddInPlace(IBackend backend, Tensor dst, Tensor src)
    {
        backend.GatedResidualLastDim(dst, dst, src, OnesDim());
        src.Dispose();
    }

    /// <summary>GPU-resident in-place broadcast add: <c>x[i,:] += bias[:]</c> for every row, via
    /// <see cref="IBackend.AffineBroadcastLastDim"/> with scale=1 (pure broadcast add).</summary>
    private void AddBroadcastInPlace(IBackend backend, Tensor x, Tensor bias)
        => backend.AffineBroadcastLastDim(x, x, OnesDim(), bias);

    private Tensor OnesDim()
    {
        if (_onesDim is null)
        {
            _onesDim = new Tensor(new TensorShape(1, _dim), DType.F32);
            float* p = (float*)_onesDim.DataPointer;
            for (int i = 0; i < _dim; i++) p[i] = 1f;
        }
        return _onesDim;
    }
}
