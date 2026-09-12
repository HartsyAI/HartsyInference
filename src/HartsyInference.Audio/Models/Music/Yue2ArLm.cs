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
    private Tensor? _semanticHead;
    private Tensor? _lastScratch;
    private Tensor? _semanticLogits;
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

        // The semantic pass can only ever draw from one contiguous 32,769-id run, so it gets its own head: an owned
        // BF16 row-slice (134 MB) that the per-token GEMM reads instead of the full 756 MB one.
        (int start, int count) = Yue2Protocol.Window(Yue2Phase.Semantic);
        if (weights.TryGetValue("lm_head.weight", out Tensor? head) && head.Shape.Rank == 2 && head.Shape[0] >= start + count)
            _semanticHead = RowSlice(head, start, count);
    }

    /// <summary>An owned copy of <paramref name="numRows"/> rows of a row-major 2-D weight, keeping its dtype.</summary>
    private static unsafe Tensor RowSlice(Tensor src, int startRow, int numRows)
    {
        long cols = src.ElementCount / src.Shape[0];
        long rowBytes = cols * src.DType.SizeInBytes;
        Tensor dst = new(new TensorShape(numRows, cols), src.DType);
        Buffer.MemoryCopy((byte*)src.DataPointer + startRow * rowBytes, (void*)dst.DataPointer,
            numRows * rowBytes, numRows * rowBytes);
        return dst;
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
    public void Forward(IBackend backend, ReadOnlySpan<int> tokenIds, int posStart, IKvCache cache, Span<float> logits,
        Yue2Phase phase)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(cache);
        ThrowIfDisposed();
        if (tokenIds.IsEmpty) throw new ArgumentException("YuE2 needs at least one token to run.", nameof(tokenIds));
        int width = Yue2Protocol.Window(phase).Count;
        if (logits.Length < width)
            throw new ArgumentException($"The logit buffer must hold {width} entries for the {phase} phase.", nameof(logits));

        using Tensor hidden = _transformer.Forward(backend, tokenIds, posStart, cache);

        // Project only the last position: a whole-prefill projection against a 184,704-wide head would cost more
        // than the prefill it followed.
        _lastScratch ??= new Tensor(new TensorShape(1, 1, _config.Ar.HiddenSize), DType.F32);
        backend.GatherRows(_lastScratch, hidden, [tokenIds.Length - 1]);
        Project(backend, _lastScratch, logits, phase);
    }

    /// <summary>Projects one post-final-norm hidden onto the phase's head window.</summary>
    private void Project(IBackend backend, Tensor last, Span<float> logits, Yue2Phase phase)
    {
        (int _, int count) = Yue2Protocol.Window(phase);
        if (phase == Yue2Phase.Semantic && _semanticHead is not null)
        {
            _semanticLogits ??= new Tensor(new TensorShape(1, 1, count), DType.F32);
            backend.Linear(_semanticLogits, last, _semanticHead, null);
            _semanticLogits.AsReadOnlySpan<float>()[..count].CopyTo(logits);
            return;
        }
        // The ABC window starts at id 0, so the full projection's leading `count` entries are exactly the window.
        using Tensor projected = _transformer.ProjectLogits(backend, last, 1);
        projected.AsReadOnlySpan<float>()[..count].CopyTo(logits);
    }

    /// <summary>A logit buffer sized for <paramref name="phase"/>'s head window, for <see cref="Forward"/> and
    /// <see cref="DecodeStep"/> to write into.</summary>
    public static float[] AllocateLogits(Yue2Phase phase) => new float[Yue2Protocol.Window(phase).Count];

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
        _semanticHead?.Dispose();
        _lastScratch?.Dispose();
        _semanticLogits?.Dispose();
        _transformer.Dispose();
    }
}
