using SharpInference.Audio.Models.Whisper;
using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Audio.Models.Kokoro;

/// <summary>Kokoro's phoneme-level BERT (PLBERT). A 12-layer ALBERT encoder — but the
/// "12 layers" share a single weight set, so we materialize one
/// <see cref="AlbertLayer"/> and loop it 12 times. End-to-end:
///
/// <code>
///   token_ids [1, T] long       (we use int32 here — IDs are < 178)
///       │
///       ▼
///   word_emb [178, 128] + position_emb [512, 128] + token_type_emb [2, 128]
///       │  sum, then LayerNorm(128)
///       ▼
///   [1, T, 128]                 ALBERT factorized embedding
///       │
///       ▼
///   embedding_hidden_mapping_in: Linear(128 → 768) [+bias]
///       │
///       ▼
///   loop ×12 (one shared AlbertLayer):
///     self-attention (Q/K/V/O), residual → LayerNorm(attention.LayerNorm)
///     ffn 768→2048 → GELU → 768, residual → LayerNorm(full_layer_layer_norm)
///       │
///       ▼
///   [1, T, 768]
///       │
///       ▼  bert_encoder: Linear(768 → 512)   ← PLBERT-specific, sibling key (no "bert." prefix)
///       ▼
///   d_bert [1, T, 512]          consumed by the prosody predictor's text-encoder
/// </code>
///
/// <para>token-type embeddings are loaded but Kokoro is monolingual at inference — all
/// inputs get token_type=0, so only row 0 is ever read. Position embeddings cap at
/// 512: a sequence longer than 512 phonemes would overrun the position table.</para></summary>
public sealed unsafe class KokoroPlBert
{
    private readonly KokoroConfig _cfg;
    private readonly KokoroConfig.PlBertConfig _plBert;

    // Embedding-layer weights.
    private Tensor? _wordEmb;
    private Tensor? _posEmb;
    private Tensor? _tokTypeEmb;
    private Tensor? _embLnW;
    private Tensor? _embLnB;
    private Tensor? _embToHiddenW;     // [768, 128]
    private Tensor? _embToHiddenB;

    // Single shared ALBERT layer (looped 12 times).
    private readonly AlbertLayer _layer;

    // Final projection (PLBERT-only): [B, T, 768] → [B, T, 512].
    private Tensor? _bertEncW;     // [512, 768]
    private Tensor? _bertEncB;

    /// <summary>Output dim of <see cref="Forward"/> — the predictor expects 512.</summary>
    public int OutputDim => 512;

    public KokoroPlBert(KokoroConfig cfg)
    {
        _cfg = cfg;
        _plBert = cfg.PlBert;
        _layer = new AlbertLayer(_plBert);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _wordEmb = WhisperOps.EnsureF32(w["bert.embeddings.word_embeddings.weight"]);
        _posEmb = WhisperOps.EnsureF32(w["bert.embeddings.position_embeddings.weight"]);
        _tokTypeEmb = WhisperOps.EnsureF32(w["bert.embeddings.token_type_embeddings.weight"]);
        _embLnW = WhisperOps.EnsureF32(w["bert.embeddings.LayerNorm.weight"]);
        _embLnB = WhisperOps.EnsureF32(w["bert.embeddings.LayerNorm.bias"]);

        _embToHiddenW = WhisperOps.EnsureF32(w["bert.encoder.embedding_hidden_mapping_in.weight"]);
        _embToHiddenB = WhisperOps.EnsureF32(w["bert.encoder.embedding_hidden_mapping_in.bias"]);

        _layer.LoadWeights(w, "bert.encoder.albert_layer_groups.0.albert_layers.0");

        _bertEncW = WhisperOps.EnsureF32(w["bert_encoder.weight"]);
        _bertEncB = WhisperOps.EnsureF32(w["bert_encoder.bias"]);
    }

    /// <summary>Runs the full PLBERT stack. <paramref name="tokenIds"/> is a 1-D int span
    /// containing the BOS/EOS-padded token IDs. Returns <c>[1, T, 512]</c> in F32,
    /// channels-last — the form the prosody predictor consumes.</summary>
    public Tensor Forward(IBackend backend, ReadOnlySpan<int> tokenIds)
    {
        int t = tokenIds.Length;
        if (t > _plBert.MaxPositionEmbeddings)
            throw new ArgumentException($"PLBERT input length {t} exceeds max position {_plBert.MaxPositionEmbeddings}.");

        // 1. Embedding lookup → [1, T, 128].
        int embDim = _plBert.EmbeddingSize;
        Tensor emb = new(new TensorShape(1, t, embDim), DType.F32);
        EmbedLookup(emb, tokenIds, embDim);

        // 2. LayerNorm over the 128-dim axis.
        Tensor embLN = new(emb.Shape, DType.F32);
        backend.LayerNorm(embLN, emb, _embLnW!, _embLnB!, _plBert.LayerNormEps);
        emb.Dispose();

        // 3. Project to hidden: [1, T, 128] → [1, T, 768].
        int hidden = _plBert.HiddenSize;
        Tensor hiddenT = WhisperOps.ProjectLinear(backend, embLN, _embToHiddenW!, _embToHiddenB, 1, t, embDim, hidden);
        embLN.Dispose();

        // 4. Loop the single shared layer 12 times.
        for (int i = 0; i < _plBert.NumHiddenLayers; i++)
        {
            Tensor next = _layer.Forward(backend, hiddenT, t);
            hiddenT.Dispose();
            hiddenT = next;
        }

        // 5. Final 768 → 512 projection (the "bert_encoder" key, outside the bert. tree).
        Tensor output = WhisperOps.ProjectLinear(backend, hiddenT, _bertEncW!, _bertEncB, 1, t, hidden, 512);
        hiddenT.Dispose();
        return output;
    }

    /// <summary>Sums word + position + token_type embeddings into the given <paramref name="output"/>
    /// (already allocated [1, T, 128]). Token type is 0 for every position — Kokoro is
    /// monolingual at inference, so we only ever read row 0 of token_type_embeddings.</summary>
    private void EmbedLookup(Tensor output, ReadOnlySpan<int> tokenIds, int dim)
    {
        float* op = (float*)output.DataPointer;
        float* wp = (float*)_wordEmb!.DataPointer;
        float* pp = (float*)_posEmb!.DataPointer;
        float* tp = (float*)_tokTypeEmb!.DataPointer;
        int t = tokenIds.Length;
        for (int i = 0; i < t; i++)
        {
            int id = tokenIds[i];
            float* w = wp + (long)id * dim;
            float* p = pp + (long)i * dim;
            // token_type 0 row only.
            for (int d = 0; d < dim; d++)
                op[i * dim + d] = w[d] + p[d] + tp[d];
        }
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_wordEmb, _posEmb, _tokTypeEmb, _embLnW, _embLnB, _embToHiddenW, _embToHiddenB, _bertEncW, _bertEncB];
        foreach (Tensor? x in all) if (x is not null) yield return x;
        foreach (Tensor x in _layer.EnumerateWeights()) yield return x;
    }
}

/// <summary>One shared ALBERT transformer layer. Standard post-LN ordering:
/// <c>x = LN(x + AttnProj(Attention(x)))</c> then <c>x = LN(x + FfnDown(GELU(FfnUp(x))))</c>.
/// Loaded once from the safetensors group and re-used <c>NumHiddenLayers</c> times.</summary>
internal sealed unsafe class AlbertLayer
{
    private readonly KokoroConfig.PlBertConfig _cfg;

    private Tensor? _qW, _qB;
    private Tensor? _kW, _kB;
    private Tensor? _vW, _vB;
    private Tensor? _oW, _oB;     // attention.dense (output proj)
    private Tensor? _attnLnW, _attnLnB;
    private Tensor? _ffnUpW, _ffnUpB;     // 768 → 2048
    private Tensor? _ffnDownW, _ffnDownB; // 2048 → 768
    private Tensor? _fullLnW, _fullLnB;

    public AlbertLayer(KokoroConfig.PlBertConfig cfg) { _cfg = cfg; }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _qW = WhisperOps.EnsureF32(w[$"{prefix}.attention.query.weight"]);
        _qB = WhisperOps.EnsureF32(w[$"{prefix}.attention.query.bias"]);
        _kW = WhisperOps.EnsureF32(w[$"{prefix}.attention.key.weight"]);
        _kB = WhisperOps.EnsureF32(w[$"{prefix}.attention.key.bias"]);
        _vW = WhisperOps.EnsureF32(w[$"{prefix}.attention.value.weight"]);
        _vB = WhisperOps.EnsureF32(w[$"{prefix}.attention.value.bias"]);
        _oW = WhisperOps.EnsureF32(w[$"{prefix}.attention.dense.weight"]);
        _oB = WhisperOps.EnsureF32(w[$"{prefix}.attention.dense.bias"]);
        _attnLnW = WhisperOps.EnsureF32(w[$"{prefix}.attention.LayerNorm.weight"]);
        _attnLnB = WhisperOps.EnsureF32(w[$"{prefix}.attention.LayerNorm.bias"]);
        _ffnUpW = WhisperOps.EnsureF32(w[$"{prefix}.ffn.weight"]);
        _ffnUpB = WhisperOps.EnsureF32(w[$"{prefix}.ffn.bias"]);
        _ffnDownW = WhisperOps.EnsureF32(w[$"{prefix}.ffn_output.weight"]);
        _ffnDownB = WhisperOps.EnsureF32(w[$"{prefix}.ffn_output.bias"]);
        _fullLnW = WhisperOps.EnsureF32(w[$"{prefix}.full_layer_layer_norm.weight"]);
        _fullLnB = WhisperOps.EnsureF32(w[$"{prefix}.full_layer_layer_norm.bias"]);
    }

    /// <summary>Forward pass. <paramref name="x"/> is <c>[1, T, hidden]</c> in F32.
    /// Returns a fresh <c>[1, T, hidden]</c> tensor.</summary>
    public Tensor Forward(IBackend backend, Tensor x, int t)
    {
        int hidden = _cfg.HiddenSize;
        int heads = _cfg.NumAttentionHeads;
        int headDim = hidden / heads;
        int ffnInner = _cfg.IntermediateSize;
        float scale = 1f / MathF.Sqrt(headDim);

        // ── Self-attention ───────────────────────────────────────────────
        Tensor q = WhisperOps.ProjectLinear(backend, x, _qW!, _qB, 1, t, hidden, hidden);
        Tensor k = WhisperOps.ProjectLinear(backend, x, _kW!, _kB, 1, t, hidden, hidden);
        Tensor v = WhisperOps.ProjectLinear(backend, x, _vW!, _vB, 1, t, hidden, hidden);

        TensorShape mhShape = new(1, heads, t, headDim);
        Tensor qMh = new(mhShape, DType.F32);
        Tensor kMh = new(mhShape, DType.F32);
        Tensor vMh = new(mhShape, DType.F32);
        WhisperOps.ReshapeToMultiHead4D(qMh, q, 1, t, heads, headDim); q.Dispose();
        WhisperOps.ReshapeToMultiHead4D(kMh, k, 1, t, heads, headDim); k.Dispose();
        WhisperOps.ReshapeToMultiHead4D(vMh, v, 1, t, heads, headDim); v.Dispose();

        Tensor attnMh = new(mhShape, DType.F32);
        backend.ScaledDotProductAttention(attnMh, qMh, kMh, vMh, mask: null, scale);
        qMh.Dispose(); kMh.Dispose(); vMh.Dispose();

        Tensor attnFlat = new(new TensorShape(1, t, hidden), DType.F32);
        WhisperOps.ReshapeFromMultiHead4D(attnFlat, attnMh, 1, t, heads, headDim);
        attnMh.Dispose();

        Tensor attnProj = WhisperOps.ProjectLinear(backend, attnFlat, _oW!, _oB, 1, t, hidden, hidden);
        attnFlat.Dispose();

        // Residual + post-attention LayerNorm.
        Tensor sum1 = new(x.Shape, DType.F32);
        backend.Add(sum1, x, attnProj);
        attnProj.Dispose();

        Tensor h1 = new(sum1.Shape, DType.F32);
        backend.LayerNorm(h1, sum1, _attnLnW!, _attnLnB!, _cfg.LayerNormEps);
        sum1.Dispose();

        // ── FFN ──────────────────────────────────────────────────────────
        Tensor ffnUp = WhisperOps.ProjectLinear(backend, h1, _ffnUpW!, _ffnUpB, 1, t, hidden, ffnInner);
        backend.Gelu(ffnUp, ffnUp);     // in-place
        Tensor ffnDown = WhisperOps.ProjectLinear(backend, ffnUp, _ffnDownW!, _ffnDownB, 1, t, ffnInner, hidden);
        ffnUp.Dispose();

        // Residual + full-layer LayerNorm.
        Tensor sum2 = new(h1.Shape, DType.F32);
        backend.Add(sum2, h1, ffnDown);
        h1.Dispose();
        ffnDown.Dispose();

        Tensor output = new(sum2.Shape, DType.F32);
        backend.LayerNorm(output, sum2, _fullLnW!, _fullLnB!, _cfg.LayerNormEps);
        sum2.Dispose();
        return output;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all =
        [
            _qW, _qB, _kW, _kB, _vW, _vB, _oW, _oB, _attnLnW, _attnLnB,
            _ffnUpW, _ffnUpB, _ffnDownW, _ffnDownB, _fullLnW, _fullLnB,
        ];
        foreach (Tensor? x in all) if (x is not null) yield return x;
    }
}
