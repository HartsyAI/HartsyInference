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

    /// <summary>Runs <paramref name="tokenIds"/> through the stack and returns the final position's logits over the
    /// full vocabulary. <paramref name="posStart"/> is the cache length before this call.</summary>
    public float[] Forward(IBackend backend, ReadOnlySpan<int> tokenIds, int posStart, IKvCache cache)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(cache);
        ThrowIfDisposed();
        if (tokenIds.IsEmpty) throw new ArgumentException("YuE2 needs at least one token to run.", nameof(tokenIds));

        using Tensor hidden = _transformer.Forward(backend, tokenIds, posStart, cache);
        return LastRowLogits(backend, hidden, tokenIds.Length);
    }

    /// <summary>Projects only the last position — the vocabulary is 184,704 wide, so projecting a whole prefill
    /// would cost more than the prefill itself.</summary>
    private float[] LastRowLogits(IBackend backend, Tensor hidden, int t)
    {
        using Tensor last = new(new TensorShape(1, 1, _config.Ar.HiddenSize), DType.F32);
        backend.GatherRows(last, hidden, [t - 1]);
        using Tensor logits = _transformer.ProjectLogits(backend, last, 1);
        return [.. logits.AsReadOnlySpan<float>()[..Yue2Protocol.VocabSize]];
    }

    /// <summary>The per-layer post-RoPE K/V the acoustic transformer attends over. Returned as cache-owned views:
    /// the caller must consume them before the cache is reset or disposed.</summary>
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
