using HartsyInference.Audio.Layers;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Bert;

/// <summary>Standard HuggingFace BERT encoder (post-norm: <c>LayerNorm(x + sublayer(x))</c>), reusable for any
/// BERT-conditioned model. Built for <c>chinese-roberta-wwm-ext-large</c> (GPT-SoVITS text features). Embeddings
/// are <c>word + learned-position + token-type</c> then LayerNorm; each layer is multi-head self-attention then
/// a GELU FFN, both post-normed. <see cref="Forward"/> returns the hidden states after a chosen number of
/// layers (GPT-SoVITS taps <c>hidden_states[-3]</c> = the output of layer <c>NumLayers - 2</c>).
///
/// <para>Single-segment (token-type 0), unpadded input — no attention mask needed. Weights load from the HF
/// state-dict names (<c>{prefix}.embeddings.*</c> / <c>{prefix}.encoder.layer.N.*</c>, prefix default
/// <c>bert</c>). Matmuls / attention / LayerNorm via <see cref="IBackend"/>; exact GELU via
/// <see cref="Activations"/>.</para></summary>
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
        _wordEmb = WhisperOps.EnsureF32(w[$"{p}embeddings.word_embeddings.weight"]);
        _posEmb = WhisperOps.EnsureF32(w[$"{p}embeddings.position_embeddings.weight"]);
        _typeEmb = WhisperOps.EnsureF32(w[$"{p}embeddings.token_type_embeddings.weight"]);
        _embLnW = WhisperOps.EnsureF32(w[$"{p}embeddings.LayerNorm.weight"]);
        _embLnB = WhisperOps.EnsureF32(w[$"{p}embeddings.LayerNorm.bias"]);
        for (int i = 0; i < _cfg.NumLayers; i++) _layers[i].LoadWeights(w, $"{p}encoder.layer.{i}");
    }

    /// <summary>Encodes token ids → hidden states <c>[1, T, hidden]</c> after <paramref name="numLayers"/>
    /// transformer layers (defaults to all). GPT-SoVITS passes <c>NumLayers - 2</c> for the <c>hidden_states[-3]</c>
    /// tap. Token-type is 0 for every position; positions are 0..T-1.</summary>
    public Tensor Forward(IBackend backend, ReadOnlySpan<int> tokenIds, int numLayers = -1)
    {
        if (_wordEmb is null) throw new InvalidOperationException("BertModel weights not loaded.");
        int t = tokenIds.Length, h = _cfg.Hidden;
        int run = numLayers < 0 ? _cfg.NumLayers : Math.Min(numLayers, _cfg.NumLayers);

        // Embeddings: word + position + token-type(0), then LayerNorm.
        Tensor emb = new(new TensorShape(1, t, h), DType.F32);
        float* ep = (float*)emb.DataPointer;
        float* wp = (float*)_wordEmb.DataPointer, pp = (float*)_posEmb!.DataPointer, tp = (float*)_typeEmb!.DataPointer;
        for (int i = 0; i < t; i++)
        {
            long dst = (long)i * h, wsrc = (long)tokenIds[i] * h, psrc = (long)i * h;
            for (int c = 0; c < h; c++) ep[dst + c] = wp[wsrc + c] + pp[psrc + c] + tp[c];
        }
        Tensor x = new(emb.Shape, DType.F32);
        backend.LayerNorm(x, emb, _embLnW!, _embLnB!, _cfg.LayerNormEps);
        emb.Dispose();

        for (int i = 0; i < run; i++)
        {
            Tensor next = _layers[i].Forward(backend, x, t);
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
        _qW = WhisperOps.EnsureF32(w[$"{prefix}.attention.self.query.weight"]); _qB = WhisperOps.EnsureF32(w[$"{prefix}.attention.self.query.bias"]);
        _kW = WhisperOps.EnsureF32(w[$"{prefix}.attention.self.key.weight"]); _kB = WhisperOps.EnsureF32(w[$"{prefix}.attention.self.key.bias"]);
        _vW = WhisperOps.EnsureF32(w[$"{prefix}.attention.self.value.weight"]); _vB = WhisperOps.EnsureF32(w[$"{prefix}.attention.self.value.bias"]);
        _aoW = WhisperOps.EnsureF32(w[$"{prefix}.attention.output.dense.weight"]); _aoB = WhisperOps.EnsureF32(w[$"{prefix}.attention.output.dense.bias"]);
        _aoLnW = WhisperOps.EnsureF32(w[$"{prefix}.attention.output.LayerNorm.weight"]); _aoLnB = WhisperOps.EnsureF32(w[$"{prefix}.attention.output.LayerNorm.bias"]);
        _interW = WhisperOps.EnsureF32(w[$"{prefix}.intermediate.dense.weight"]); _interB = WhisperOps.EnsureF32(w[$"{prefix}.intermediate.dense.bias"]);
        _outW = WhisperOps.EnsureF32(w[$"{prefix}.output.dense.weight"]); _outB = WhisperOps.EnsureF32(w[$"{prefix}.output.dense.bias"]);
        _outLnW = WhisperOps.EnsureF32(w[$"{prefix}.output.LayerNorm.weight"]); _outLnB = WhisperOps.EnsureF32(w[$"{prefix}.output.LayerNorm.bias"]);
    }

    public Tensor Forward(IBackend backend, Tensor x, int t)
    {
        int h = _cfg.Hidden, nh = _cfg.NumHeads, hd = _cfg.HeadDim;
        Tensor q = WhisperOps.ProjectLinear(backend, x, _qW!, _qB, 1, t, h, h);
        Tensor k = WhisperOps.ProjectLinear(backend, x, _kW!, _kB, 1, t, h, h);
        Tensor v = WhisperOps.ProjectLinear(backend, x, _vW!, _vB, 1, t, h, h);

        Tensor qH = new(new TensorShape(1, nh, t, hd), DType.F32);
        Tensor kH = new(new TensorShape(1, nh, t, hd), DType.F32);
        Tensor vH = new(new TensorShape(1, nh, t, hd), DType.F32);
        WhisperOps.ReshapeToMultiHead4D(qH, q, 1, t, nh, hd);
        WhisperOps.ReshapeToMultiHead4D(kH, k, 1, t, nh, hd);
        WhisperOps.ReshapeToMultiHead4D(vH, v, 1, t, nh, hd);
        q.Dispose(); k.Dispose(); v.Dispose();

        Tensor attn = new(new TensorShape(1, nh, t, hd), DType.F32);
        backend.ScaledDotProductAttention(attn, qH, kH, vH, null, 1f / MathF.Sqrt(hd));
        qH.Dispose(); kH.Dispose(); vH.Dispose();

        Tensor ctx = new(new TensorShape(1, t, h), DType.F32);
        WhisperOps.ReshapeFromMultiHead4D(ctx, attn, 1, t, nh, hd);
        attn.Dispose();
        Tensor ao = WhisperOps.ProjectLinear(backend, ctx, _aoW!, _aoB, 1, t, h, h);
        ctx.Dispose();
        AddInPlace(ao, x);                                  // residual (x + attn_out)
        Tensor x1 = new(ao.Shape, DType.F32);
        backend.LayerNorm(x1, ao, _aoLnW!, _aoLnB!, _cfg.LayerNormEps);
        ao.Dispose();

        int ff = _cfg.Intermediate;
        Tensor inter = WhisperOps.ProjectLinear(backend, x1, _interW!, _interB, 1, t, h, ff);
        Activations.ErfGelu(inter);
        Tensor outd = WhisperOps.ProjectLinear(backend, inter, _outW!, _outB, 1, t, ff, h);
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
