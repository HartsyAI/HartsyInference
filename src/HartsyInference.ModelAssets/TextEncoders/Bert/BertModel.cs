using System;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.TextEncoders.Bert;

/// <summary>Standard HuggingFace BERT encoder (post-norm: <c>LayerNorm(x + sublayer(x))</c>), reusable for any
/// BERT-conditioned model (GPT-SoVITS / MeloTTS text prosody, Grounding DINO open-vocab text tower). Embeddings are
/// <c>word + learned-position + token-type</c> then LayerNorm; each layer is multi-head self-attention then a GELU
/// FFN, both post-normed.
///
/// <para>Two entry points: <see cref="Forward(IBackend, ReadOnlySpan{int}, int)"/> is the unpadded single-segment
/// path (token-type 0, positions 0..T-1, no mask) used by the audio prosody towers; the overload taking explicit
/// position ids and an additive attention mask is used by Grounding DINO, whose caption is split into per-phrase
/// sub-sentences via a block-diagonal self-attention mask and reset position ids.</para>
///
/// <para>Self-contained — routes all math through <see cref="IBackend"/> so it lives in the shared ModelHandler
/// package (both Audio and Vision reference it) without dragging the audio stack across a package boundary. Weights
/// load from the HF state-dict names (<c>{prefix}.embeddings.*</c> / <c>{prefix}.encoder.layer.N.*</c>, prefix
/// default <c>bert</c>).</para></summary>
public sealed unsafe class BertModel : IDisposable
{
    private readonly BertConfig _cfg;
    private readonly BertLayer[] _layers;
    private Tensor? _wordEmb, _posEmb, _typeEmb, _embLnW, _embLnB;
    private int _disposed;

    public BertConfig Config => _cfg;

    public BertModel(BertConfig cfg)
    {
        _cfg = cfg;
        _layers = new BertLayer[cfg.NumLayers];
        for (int i = 0; i < cfg.NumLayers; i++) _layers[i] = new BertLayer(cfg);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "bert")
    {
        string p = prefix.Length == 0 ? "" : prefix + ".";
        _wordEmb = BertOps.EnsureF32(w[$"{p}embeddings.word_embeddings.weight"]);
        _posEmb = BertOps.EnsureF32(w[$"{p}embeddings.position_embeddings.weight"]);
        _typeEmb = BertOps.EnsureF32(w[$"{p}embeddings.token_type_embeddings.weight"]);
        _embLnW = BertOps.EnsureF32(w[$"{p}embeddings.LayerNorm.weight"]);
        _embLnB = BertOps.EnsureF32(w[$"{p}embeddings.LayerNorm.bias"]);
        for (int i = 0; i < _cfg.NumLayers; i++) _layers[i].LoadWeights(w, $"{p}encoder.layer.{i}");
    }

    /// <summary>Encodes token ids → hidden states <c>[1, T, hidden]</c> after <paramref name="numLayers"/>
    /// transformer layers (defaults to all). GPT-SoVITS passes <c>NumLayers - 2</c> for the <c>hidden_states[-3]</c>
    /// tap. Token-type is 0 for every position; positions are 0..T-1; unmasked.</summary>
    public Tensor Forward(IBackend backend, ReadOnlySpan<int> tokenIds, int numLayers = -1)
        => Forward(backend, tokenIds, default, default, null, numLayers);

    /// <summary>Full BERT forward with explicit position ids and an optional additive self-attention mask. When
    /// <paramref name="positionIds"/> is empty, positions default to <c>0..T-1</c>; when <paramref name="tokenTypeIds"/>
    /// is empty, all segments are 0. <paramref name="additiveAttnMask"/> (if non-null) must be <c>[1, numHeads, T, T]</c>
    /// with 0 on attended pairs and a large negative on masked pairs; it is added to the pre-softmax scores.</summary>
    public Tensor Forward(IBackend backend, ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positionIds,
        ReadOnlySpan<int> tokenTypeIds, Tensor? additiveAttnMask, int numLayers = -1)
    {
        if (_wordEmb is null) throw new InvalidOperationException("BertModel weights not loaded.");
        int t = tokenIds.Length, h = _cfg.Hidden;
        int run = numLayers < 0 ? _cfg.NumLayers : Math.Min(numLayers, _cfg.NumLayers);

        Tensor emb = new(new TensorShape(1, t, h), DType.F32);
        float* ep = (float*)emb.DataPointer;
        float* wp = (float*)_wordEmb.DataPointer, pp = (float*)_posEmb!.DataPointer, tp = (float*)_typeEmb!.DataPointer;
        for (int i = 0; i < t; i++)
        {
            int pos = positionIds.IsEmpty ? i : positionIds[i];
            int typ = tokenTypeIds.IsEmpty ? 0 : tokenTypeIds[i];
            long dst = (long)i * h, wsrc = (long)tokenIds[i] * h, psrc = (long)pos * h, tsrc = (long)typ * h;
            for (int c = 0; c < h; c++) ep[dst + c] = wp[wsrc + c] + pp[psrc + c] + tp[tsrc + c];
        }
        Tensor x = new(emb.Shape, DType.F32);
        backend.LayerNorm(x, emb, _embLnW!, _embLnB!, _cfg.LayerNormEps);
        emb.Dispose();

        for (int i = 0; i < run; i++)
        {
            Tensor next = _layers[i].Forward(backend, x, t, additiveAttnMask);
            x.Dispose();
            x = next;
        }
        return x;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] top = [_wordEmb, _posEmb, _typeEmb, _embLnW, _embLnB];
        foreach (Tensor? tt in top) if (tt is not null) yield return tt;
        foreach (BertLayer l in _layers) foreach (Tensor tt in l.EnumerateWeights()) yield return tt;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }
}

/// <summary>One BERT encoder layer: post-norm multi-head self-attention then a post-norm GELU FFN.</summary>
internal sealed unsafe class BertLayer
{
    private readonly BertConfig _cfg;
    private Tensor? _qW, _qB, _kW, _kB, _vW, _vB, _aoW, _aoB, _aoLnW, _aoLnB;
    private Tensor? _interW, _interB, _outW, _outB, _outLnW, _outLnB;

    public BertLayer(BertConfig cfg) => _cfg = cfg;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _qW = BertOps.EnsureF32(w[$"{prefix}.attention.self.query.weight"]); _qB = BertOps.EnsureF32(w[$"{prefix}.attention.self.query.bias"]);
        _kW = BertOps.EnsureF32(w[$"{prefix}.attention.self.key.weight"]); _kB = BertOps.EnsureF32(w[$"{prefix}.attention.self.key.bias"]);
        _vW = BertOps.EnsureF32(w[$"{prefix}.attention.self.value.weight"]); _vB = BertOps.EnsureF32(w[$"{prefix}.attention.self.value.bias"]);
        _aoW = BertOps.EnsureF32(w[$"{prefix}.attention.output.dense.weight"]); _aoB = BertOps.EnsureF32(w[$"{prefix}.attention.output.dense.bias"]);
        _aoLnW = BertOps.EnsureF32(w[$"{prefix}.attention.output.LayerNorm.weight"]); _aoLnB = BertOps.EnsureF32(w[$"{prefix}.attention.output.LayerNorm.bias"]);
        _interW = BertOps.EnsureF32(w[$"{prefix}.intermediate.dense.weight"]); _interB = BertOps.EnsureF32(w[$"{prefix}.intermediate.dense.bias"]);
        _outW = BertOps.EnsureF32(w[$"{prefix}.output.dense.weight"]); _outB = BertOps.EnsureF32(w[$"{prefix}.output.dense.bias"]);
        _outLnW = BertOps.EnsureF32(w[$"{prefix}.output.LayerNorm.weight"]); _outLnB = BertOps.EnsureF32(w[$"{prefix}.output.LayerNorm.bias"]);
    }

    public Tensor Forward(IBackend backend, Tensor x, int t, Tensor? additiveAttnMask)
    {
        int h = _cfg.Hidden, nh = _cfg.NumHeads, hd = _cfg.HeadDim;
        Tensor q = BertOps.ProjectLinear(backend, x, _qW!, _qB, 1, t, h);
        Tensor k = BertOps.ProjectLinear(backend, x, _kW!, _kB, 1, t, h);
        Tensor v = BertOps.ProjectLinear(backend, x, _vW!, _vB, 1, t, h);

        Tensor qH = new(new TensorShape(1, nh, t, hd), DType.F32);
        Tensor kH = new(new TensorShape(1, nh, t, hd), DType.F32);
        Tensor vH = new(new TensorShape(1, nh, t, hd), DType.F32);
        BertOps.ReshapeToMultiHead4D(qH, q, 1, t, nh, hd);
        BertOps.ReshapeToMultiHead4D(kH, k, 1, t, nh, hd);
        BertOps.ReshapeToMultiHead4D(vH, v, 1, t, nh, hd);
        q.Dispose(); k.Dispose(); v.Dispose();

        Tensor attn = new(new TensorShape(1, nh, t, hd), DType.F32);
        backend.ScaledDotProductAttention(attn, qH, kH, vH, additiveAttnMask, 1f / MathF.Sqrt(hd));
        qH.Dispose(); kH.Dispose(); vH.Dispose();

        Tensor ctx = new(new TensorShape(1, t, h), DType.F32);
        BertOps.ReshapeFromMultiHead4D(ctx, attn, 1, t, nh, hd);
        attn.Dispose();
        Tensor ao = BertOps.ProjectLinear(backend, ctx, _aoW!, _aoB, 1, t, h);
        ctx.Dispose();
        AddInPlace(ao, x);                                  // residual (x + attn_out)
        Tensor x1 = new(ao.Shape, DType.F32);
        backend.LayerNorm(x1, ao, _aoLnW!, _aoLnB!, _cfg.LayerNormEps);
        ao.Dispose();

        int ff = _cfg.Intermediate;
        Tensor inter = BertOps.ProjectLinear(backend, x1, _interW!, _interB, 1, t, ff);
        BertOps.ErfGelu(inter);
        Tensor outd = BertOps.ProjectLinear(backend, inter, _outW!, _outB, 1, t, h);
        inter.Dispose();
        AddInPlace(outd, x1);                               // residual (x1 + ffn_out)
        x1.Dispose();
        Tensor x2 = new(outd.Shape, DType.F32);
        backend.LayerNorm(x2, outd, _outLnW!, _outLnB!, _cfg.LayerNormEps);
        outd.Dispose();
        return x2;
    }

    private static void AddInPlace(Tensor dst, Tensor src)
    {
        float* d = (float*)dst.DataPointer, s = (float*)src.DataPointer;
        for (long i = 0; i < dst.ElementCount; i++) d[i] += s[i];
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_qW, _qB, _kW, _kB, _vW, _vB, _aoW, _aoB, _aoLnW, _aoLnB,
            _interW, _interB, _outW, _outB, _outLnW, _outLnB];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }
}

/// <summary>Small self-contained tensor helpers for the BERT tower (mirrors the shape of the Whisper/CLIP encoder
/// helpers): F32 casting, linear projection through <see cref="IBackend.Linear"/>, multi-head reshape, and an exact
/// erf-GELU. Kept here so the shared package has no dependency on the audio package's WhisperOps.</summary>
internal static unsafe class BertOps
{
    public static Tensor EnsureF32(Tensor t) => t.DType != DType.F32 ? t.CastTo(DType.F32) : t;

    public static Tensor ProjectLinear(IBackend backend, Tensor input, Tensor weight, Tensor? bias, int batch, int seqLen, int outDim)
    {
        Tensor output = new(new TensorShape(batch, seqLen, outDim), DType.F32);
        backend.Linear(output, input, weight, bias);
        return output;
    }

    public static void ReshapeToMultiHead4D(Tensor output, Tensor input, int batch, int seqLen, int numHeads, int headDim)
    {
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        for (int b = 0; b < batch; b++)
            for (int s = 0; s < seqLen; s++)
                for (int hh = 0; hh < numHeads; hh++)
                {
                    int inOffset = (b * seqLen + s) * (numHeads * headDim) + hh * headDim;
                    int outOffset = ((b * numHeads + hh) * seqLen + s) * headDim;
                    for (int d = 0; d < headDim; d++) outPtr[outOffset + d] = inPtr[inOffset + d];
                }
    }

    public static void ReshapeFromMultiHead4D(Tensor output, Tensor input, int batch, int seqLen, int numHeads, int headDim)
    {
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        for (int b = 0; b < batch; b++)
            for (int s = 0; s < seqLen; s++)
                for (int hh = 0; hh < numHeads; hh++)
                {
                    int inOffset = ((b * numHeads + hh) * seqLen + s) * headDim;
                    int outOffset = (b * seqLen + s) * (numHeads * headDim) + hh * headDim;
                    for (int d = 0; d < headDim; d++) outPtr[outOffset + d] = inPtr[inOffset + d];
                }
    }

    /// <summary>Exact erf-GELU in place (BERT/DETR use the erf form, not the tanh approximation).</summary>
    public static void ErfGelu(Tensor x)
    {
        float* p = (float*)x.DataPointer;
        long n = x.ElementCount;
        const float invSqrt2 = 0.70710678118654752f;
        for (long i = 0; i < n; i++)
        {
            float v = p[i];
            p[i] = v * 0.5f * (1f + Erf(v * invSqrt2));
        }
    }

    /// <summary>erf via Abramowitz &amp; Stegun 7.1.26 (max abs error ~1.5e-7).</summary>
    public static float Erf(float x)
    {
        float sign = x < 0 ? -1f : 1f;
        float ax = MathF.Abs(x);
        float tt = 1f / (1f + 0.3275911f * ax);
        float y = 1f - (((((1.061405429f * tt - 1.453152027f) * tt) + 1.421413741f) * tt - 0.284496736f) * tt
            + 0.254829592f) * tt * MathF.Exp(-ax * ax);
        return sign * y;
    }
}
