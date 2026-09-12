using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.Audio.Models.Music;

/// <summary>YuE2's autoregressive stack: the Qwen3-geometry LM that plans an ABC score and then emits semantic codec
/// tokens. It is a plain causal decoder with an output head — the interesting part is what happens to its KV cache
/// afterwards, which <see cref="Yue2AcousticTransformer"/> consumes as an attention prefix.</summary>
public sealed class Yue2ArLm : IDisposable
{
    private readonly GenericTransformer _transformer;
    private readonly Yue2Config _config;
    private int _disposed;

    public Yue2ArLm(Yue2Config config)
    {
        ArgumentNullException.ThrowIfNull(config);
        _config = config;
        _transformer = new GenericTransformer(BuildConfig(config.Ar, Yue2Protocol.VocabSize));
    }

    /// <summary>Layers in the stack; the acoustic transformer must match it one for one.</summary>
    public int NumLayers => _config.Ar.NumHiddenLayers;

    public int KvHeads => _config.Ar.NumKeyValueHeads;

    public int HeadDim => _config.Ar.HeadDim;

    /// <summary>Builds the shared decoder settings both YuE2 stacks run under. Split-half (NeoX) RoPE and per-head
    /// q/k RMSNorm are Qwen3 conventions the checkpoint follows; the embeddings are untied (the checkpoint ships a
    /// separate <c>lm_head</c>).</summary>
    internal static TransformerConfig BuildConfig(Models.LanguageModels.Qwen3.Qwen3Config body, int vocabSize) => new()
    {
        HiddenSize = body.HiddenSize,
        NumLayers = body.NumHiddenLayers,
        NumHeads = body.NumAttentionHeads,
        NumKvHeads = body.NumKeyValueHeads,
        HeadDim = body.HeadDim,
        IntermediateSize = body.IntermediateSize,
        VocabSize = vocabSize,
        MaxPositionEmbeddings = body.MaxPositionEmbeddings,
        RopeTheta = body.RopeTheta,
        RmsNormEps = body.RmsNormEps,
        AttentionBias = false,
        QkNorm = true,
        TieWordEmbeddings = false,
        Rope = RopeStyle.SplitHalf,
    };

    /// <summary>Loads the converted <c>text_encoders.</c> stack.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        ThrowIfDisposed();
        _transformer.LoadWeights(weights, "model", "lm_head.weight");
    }

    /// <summary>A decode cache sized for <paramref name="maxSeqLen"/> tokens. One per CFG branch: the two branches
    /// have different prefixes and therefore different lengths, so they cannot share.</summary>
    public IKvCache CreateCache(int maxSeqLen)
        => KvCaches.ForDecode(NumLayers, KvHeads, HeadDim, maxSeqLen);

    /// <summary>Runs <paramref name="tokenIds"/> through the stack and writes the final position's logits over the
    /// full vocabulary into <paramref name="logits"/>. <paramref name="posStart"/> is the cache length before this
    /// call.</summary>
    /// <remarks>The caller supplies the buffer because a song is up to 9,000 decode steps and the vocabulary is
    /// 184,704 wide — returning a fresh array per token would churn gigabytes through the GC.</remarks>
    public void Forward(IBackend backend, ReadOnlySpan<int> tokenIds, int posStart, IKvCache cache, Span<float> logits)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(cache);
        ThrowIfDisposed();
        if (tokenIds.IsEmpty) throw new ArgumentException("YuE2 needs at least one token to run.", nameof(tokenIds));
        if (logits.Length < Yue2Protocol.VocabSize)
            throw new ArgumentException($"The logit buffer must hold {Yue2Protocol.VocabSize} entries.", nameof(logits));

        using Tensor hidden = _transformer.Forward(backend, tokenIds, posStart, cache);

        // Project only the last position: a whole-prefill projection against a 184,704-wide head would cost more
        // than the prefill it followed.
        using Tensor last = new(new TensorShape(1, 1, _config.Ar.HiddenSize), DType.F32);
        backend.GatherRows(last, hidden, [tokenIds.Length - 1]);
        using Tensor projected = _transformer.ProjectLogits(backend, last, 1);
        projected.AsReadOnlySpan<float>()[..Yue2Protocol.VocabSize].CopyTo(logits);
    }

    /// <summary>A logit buffer sized for this model's vocabulary, for <see cref="Forward"/> to write into.</summary>
    public static float[] AllocateLogits() => new float[Yue2Protocol.VocabSize];

    /// <summary>The per-layer post-RoPE K/V the acoustic transformer attends over. Returned as cache-owned views:
    /// the caller must consume them before the cache is reset or disposed.</summary>
    /// <remarks>Each tensor is the cache's <b>whole capacity</b> buffer, <c>[1, kv_heads, capacity, head_dim]</c>,
    /// not a view trimmed to the populated length — so a consumer must slice to <see cref="IKvCache.CurrentLength"/>
    /// and stride by the capacity, never by the token count.</remarks>
    public (Tensor Key, Tensor Value)[] ExportPrefix(IKvCache cache)
    {
        ArgumentNullException.ThrowIfNull(cache);
        ThrowIfDisposed();
        (Tensor, Tensor)[] prefix = new (Tensor, Tensor)[NumLayers];
        for (int i = 0; i < NumLayers; i++) prefix[i] = (cache.KeyPrefix(i), cache.ValuePrefix(i));
        return prefix;
    }

    public IEnumerable<Tensor> EnumerateWeights() => _transformer.EnumerateWeights();

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed != 0, this);

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _transformer.Dispose();
    }
}
