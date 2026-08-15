using HartsyInference.Audio.Streaming;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.Audio.Models.LanguageModels.Qwen3;

/// <summary>Headless Qwen3 decoder backbone (per-head q_norm/k_norm, decoupled head_dim, GQA, SwiGLU,
/// bias-free attention, 1D interleaved RoPE). Now a thin compatibility wrapper over the shared
/// <see cref="GenericTransformer"/> in <c>HartsyInference.LLM</c>; the per-layer math lives there. Public
/// surface is unchanged so the Qwen3-TTS talker and MTP code-predictor (which own their embeddings + output
/// heads and drive this via <see cref="ForwardEmbeds"/>) need no changes.
///
/// <para>RoPE is the interleaved (GPT-J) convention, matching the prior implementation. (The talker's 3D
/// mRoPE remains a future refinement, exactly as before.)</para></summary>
public sealed class Qwen3Model : IDisposable
{
    private readonly Qwen3Config _cfg;
    private readonly GenericTransformer _transformer;
    private int _disposed;

    public Qwen3Config Config => _cfg;
    public int NumLayers => _cfg.NumHiddenLayers;
    public int KvHeads => _cfg.NumKeyValueHeads;
    public int HeadDim => _cfg.HeadDim;

    public Qwen3Model(Qwen3Config cfg)
    {
        _cfg = cfg;
        _transformer = new GenericTransformer(new TransformerConfig
        {
            HiddenSize = cfg.HiddenSize,
            NumLayers = cfg.NumHiddenLayers,
            NumHeads = cfg.NumAttentionHeads,
            NumKvHeads = cfg.NumKeyValueHeads,
            HeadDim = cfg.HeadDim,
            IntermediateSize = cfg.IntermediateSize,
            VocabSize = 1, // headless: embed/lm_head live on the outer model; vocab is unused here.
            MaxPositionEmbeddings = cfg.MaxPositionEmbeddings,
            RopeTheta = cfg.RopeTheta,
            RmsNormEps = cfg.RmsNormEps,
            AttentionBias = false,
            QkNorm = true,
            TieWordEmbeddings = true,
            // NeoX split-half RoPE (rotate_half), matching real Qwen3 / Qwen3-TTS. The "interleaved" in the
            // model's rope_scaling refers to mRoPE *section* interleaving, NOT the dim-pairing convention.
            Rope = RopeStyle.SplitHalf,
        });
    }

    /// <summary>Loads the headless transformer body (layers + final RMSNorm). Drive via
    /// <see cref="ForwardEmbeds"/>.</summary>
    public void LoadWeightsHeadless(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        ThrowIfDisposed();
        _transformer.LoadWeightsHeadless(w, prefix);
    }

    /// <summary>Runs the decoder stack over <paramref name="embeds"/> <c>[1, t, hidden]</c> and returns the
    /// final-normed <c>[1, t, hidden]</c> hidden state. <paramref name="posStart"/> = cache length before this step.</summary>
    public Tensor ForwardEmbeds(IBackend backend, Tensor embeds, int t, int posStart, IKvCache cache)
    {
        ThrowIfDisposed();
        return _transformer.ForwardEmbeds(backend, embeds, t, posStart, cache);
    }

    /// <summary>Batched decode step: <paramref name="embeds"/> is <c>[1, B, hidden]</c> (one token per sequence),
    /// <paramref name="positions"/>[b] is sequence b's absolute position and <paramref name="caches"/>[b] its own
    /// cache. Returns the final-normed <c>[1, B, hidden]</c>. The projections run as one GEMM over all B rows, so
    /// the weights stream from HBM once instead of once per sequence — the whole point at B tokens of decode.
    /// See <see cref="GenericTransformer.ForwardBatchDecode"/>.</summary>
    public Tensor ForwardBatchDecode(IBackend backend, Tensor embeds, ReadOnlySpan<int> positions, IKvCache[] caches)
    {
        ThrowIfDisposed();
        return _transformer.ForwardBatchDecode(backend, embeds, positions, caches);
    }

    /// <summary>True when the body can run the two-stream (B=2, shared-position) CFG graph decode step.
    /// See <see cref="GenericTransformer.SupportsDualGraphDecode"/>.</summary>
    public bool SupportsDualGraphDecode(IBackend backend) => _transformer.SupportsDualGraphDecode(backend);

    /// <summary>Builds/returns the device-resident graph-decode RoPE table sized to <paramref name="minCapacity"/>.</summary>
    public (Tensor cos, Tensor sin) EnsureRopeTableForGraphDecode(IBackend backend, int minCapacity)
        => _transformer.EnsureRopeTableForGraphDecode(backend, minCapacity);

    /// <summary>Graph-capturable two-stream (cond+uncond, position-aligned) body step from a fixed <c>[1,2,H]</c>
    /// input buffer to a fixed <c>[1,2,H]</c> output buffer, one KV cache per stream. Does NOT advance either
    /// cache. See <see cref="GenericTransformer.ForwardGraphDecodeStepDualEmbeds"/>.</summary>
    public void ForwardGraphDecodeStepDualEmbeds(IBackend backend, Tensor inEmbed, IKvCache cacheA, IKvCache cacheB,
        Tensor cosTable, Tensor sinTable, ulong devicePos, Tensor outHidden)
        => _transformer.ForwardGraphDecodeStepDualEmbeds(backend, inEmbed, cacheA, cacheB, cosTable, sinTable, devicePos, outHidden);

    /// <summary>Allocates the efficient incremental decode cache for this model (<see cref="KvCaches.ForDecode"/>).
    /// Prefer this over hand-rolling a cache. Dispose when done.</summary>
    public IKvCache CreateDecodeCache(int maxSeqLen) => KvCaches.ForDecode(NumLayers, KvHeads, HeadDim, maxSeqLen);

    /// <summary>All weight tensors for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights() => _transformer.EnumerateWeights();

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(Qwen3Model));
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _transformer.Dispose();
    }
}
