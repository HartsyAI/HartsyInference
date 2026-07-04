using HartsyInference.Audio.Streaming;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.Audio.Models.LanguageModels.Qwen2;

/// <summary>Qwen2 / Qwen2.5 autoregressive language model. This is now a thin compatibility wrapper over the
/// shared <see cref="GenericTransformer"/> in <c>HartsyInference.LLM</c> (the single Qwen-family decoder
/// implementation); the per-layer math lives there. The public surface is unchanged so the audio models that
/// depend on it (VibeVoice, Sesame CSM, Kyutai, SparkTTS, YuE, Fish-Speech, PocketTTS, Chatterbox, CosyVoice)
/// need no changes.
///
/// <para>Two-phase use: <see cref="Forward"/> with all prompt tokens (prefill, T &gt; 1, causal mask), then
/// <see cref="Forward"/> with one token per decode step. Headless / inputs-embeds / partial-layer-range paths
/// are supported via <see cref="LoadWeightsHeadless"/> and <see cref="ForwardEmbeds"/>.</para></summary>
public sealed class Qwen2Model : IDisposable
{
    private readonly Qwen2Config _cfg;
    private readonly GenericTransformer _transformer;
    private int _disposed;

    public Qwen2Config Config => _cfg;
    public int NumLayers => _cfg.NumHiddenLayers;
    public int HiddenSize => _cfg.HiddenSize;
    public int NumKeyValueHeads => _cfg.NumKeyValueHeads;
    public int HeadDim => _cfg.HeadDim;

    public Qwen2Model(Qwen2Config cfg)
    {
        _cfg = cfg;
        _transformer = new GenericTransformer(ToTransformerConfig(cfg));
    }

    /// <summary>Maps the Qwen2 config onto the shared transformer config: QKV bias, split-half RoPE, no per-head
    /// Q/K norm, head_dim coupled to hidden/heads.</summary>
    internal static TransformerConfig ToTransformerConfig(Qwen2Config cfg) => new()
    {
        HiddenSize = cfg.HiddenSize,
        NumLayers = cfg.NumHiddenLayers,
        NumHeads = cfg.NumAttentionHeads,
        NumKvHeads = cfg.NumKeyValueHeads,
        HeadDim = cfg.HeadDim,
        IntermediateSize = cfg.IntermediateSize,
        VocabSize = cfg.VocabSize,
        MaxPositionEmbeddings = cfg.MaxPositionEmbeddings,
        RopeTheta = cfg.RopeTheta,
        RopeScaling = cfg.RopeScaling ?? HartsyInference.Core.Rope.RopeScaling.None,
        RmsNormEps = cfg.RmsNormEps,
        AttentionBias = cfg.AttentionBias,
        QkNorm = false,
        TieWordEmbeddings = cfg.TieWordEmbeddings,
        // Per-config opt-in OR the global env (TransformerConfig.LowVramQuant defaults to the env; OR-ing preserves it).
        LowVramQuant = cfg.LowVramQuant || Environment.GetEnvironmentVariable("HARTSY_LOWVRAM_QUANT") == "1",
        Rope = cfg.Rope,
    };

    /// <summary>Loads all LM weights (embed_tokens + layers + final norm + optional lm_head).
    /// <paramref name="prefix"/> is everything up to <c>embed_tokens</c> (e.g. <c>"model"</c> standalone,
    /// <c>"model.language_model"</c> for VibeVoice).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix, string? lmHeadKey = null)
    {
        ThrowIfDisposed();
        _transformer.LoadWeights(w, prefix, lmHeadKey);
    }

    /// <summary>Loads a headless body (layers + final norm only; no embed/lm_head). Drive via
    /// <see cref="ForwardEmbeds"/>; do not call <see cref="ProjectLogits"/>.</summary>
    public void LoadWeightsHeadless(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        ThrowIfDisposed();
        _transformer.LoadWeightsHeadless(w, prefix);
    }

    /// <summary>Token-IDs-in path: embedding lookup then the decoder stack. Returns the final
    /// <c>[1, T, hidden]</c> hidden state (post final RMSNorm).</summary>
    public Tensor Forward(IBackend backend, ReadOnlySpan<int> tokenIds, int batch, int posStart, IKvCache cache)
    {
        ThrowIfDisposed();
        RequireBatchOne(batch);
        return _transformer.Forward(backend, tokenIds, posStart, cache);
    }

    /// <summary>Embedding-in path with optional layer range and final-norm toggle (VibeVoice streaming splits
    /// the stack). <paramref name="batch"/> must be 1. The cache is any <see cref="IKvCache"/> — pass a
    /// <see cref="FixedKvCache"/> for O(1)-append incremental decode, or a <see cref="StreamingKvCache"/> for the
    /// host-copy prefix path.</summary>
    public Tensor ForwardEmbeds(IBackend backend, Tensor inputsEmbeds, int batch, int t, int posStart,
        IKvCache cache, int startLayer = 0, int? endLayer = null, bool applyFinalNorm = true)
    {
        ThrowIfDisposed();
        RequireBatchOne(batch);
        return _transformer.ForwardEmbeds(backend, inputsEmbeds, t, posStart, cache, applyFinalNorm, startLayer, endLayer);
    }

    /// <summary>Allocates the efficient incremental decode cache for this model (<see cref="KvCaches.ForDecode"/>) —
    /// a <see cref="FixedKvCache"/> (O(1) appends). Prefer this over hand-rolling a cache. Dispose when done.</summary>
    public IKvCache CreateDecodeCache(int maxSeqLen) => KvCaches.ForDecode(NumLayers, NumKeyValueHeads, HeadDim, maxSeqLen);

    /// <summary>Creates a single-sequence <see cref="FixedKvCache"/> directly (the concrete type
    /// <see cref="ForwardBatchDecode"/> needs, one per batched sequence). Dispose when done.</summary>
    public FixedKvCache CreateFixedCache(int maxSeqLen) => new(NumLayers, batch: 1, NumKeyValueHeads, HeadDim, Math.Max(1, maxSeqLen));

    /// <summary>Batched decode step: <paramref name="embeds"/> is <c>[1, B, hidden]</c> (one token per sequence),
    /// <paramref name="positions"/>[b] is sequence b's absolute position, <paramref name="caches"/>[b] its own cache.
    /// Returns post-norm hidden <c>[1, B, hidden]</c> (rows = B, ready for <see cref="ProjectLogits"/>). See
    /// <see cref="GenericTransformer.ForwardBatchDecode"/>.</summary>
    public Tensor ForwardBatchDecode(IBackend backend, Tensor embeds, ReadOnlySpan<int> positions, FixedKvCache[] caches)
    {
        ThrowIfDisposed();
        return _transformer.ForwardBatchDecode(backend, embeds, positions, caches);
    }

    /// <summary>Projects the final hidden state <c>[1, T, hidden]</c> to vocab logits <c>[1, T, vocab]</c>.</summary>
    public Tensor ProjectLogits(IBackend backend, Tensor hidden, int batch, int t)
    {
        ThrowIfDisposed();
        RequireBatchOne(batch);
        return _transformer.ProjectLogits(backend, hidden, t);
    }

    /// <summary>Performs the embedding lookup for a batch of token IDs into <paramref name="output"/>
    /// <c>[1, T, hidden]</c>.</summary>
    public void EmbedLookup(Tensor output, ReadOnlySpan<int> tokenIds, int batch, int t)
    {
        ThrowIfDisposed();
        RequireBatchOne(batch);
        _transformer.EmbedLookup(output, tokenIds);
    }

    /// <summary>All weight tensors for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights() => _transformer.EnumerateWeights();

    private static void RequireBatchOne(int batch)
    {
        if (batch != 1) throw new NotSupportedException($"Qwen2Model supports batch=1, got {batch}.");
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(Qwen2Model));
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _transformer.Dispose();
    }
}
