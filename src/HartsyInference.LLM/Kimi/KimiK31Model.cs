using HartsyInference.Core.Tensors.Quant;
using HartsyInference.LLM.Transformer.Quant;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;
using HartsyInference.LLM.Transformer.AttnRes;
using HartsyInference.LLM.Transformer.Dsa;
using HartsyInference.LLM.Transformer.Kda;

namespace HartsyInference.LLM.Kimi;

/// <summary>Weights of one Kimi K3.1 decoder layer. Exactly one of <see cref="Kda"/> and <see cref="Mla"/> is set, and exactly one of
/// <see cref="DenseGate"/> (with <see cref="DenseUp"/> and <see cref="DenseDown"/>) and <see cref="Moe"/> is set.</summary>
public sealed record KimiK31LayerWeights
{
    /// <summary>Pre-attention RMSNorm weight, <c>[Hidden]</c>.</summary>
    public required Tensor InputNorm { get; init; }

    /// <summary>Pre-FFN RMSNorm weight, <c>[Hidden]</c>.</summary>
    public required Tensor PostNorm { get; init; }

    /// <summary>Attention-residual pseudo-query before the attention sublayer, <c>[Hidden, 1]</c>, and its key norm, <c>[Hidden]</c>.</summary>
    public required Tensor AttnResAttnQuery { get; init; }

    /// <summary>Key norm of <see cref="AttnResAttnQuery"/>.</summary>
    public required Tensor AttnResAttnNorm { get; init; }

    /// <summary>Attention-residual pseudo-query before the FFN sublayer, <c>[Hidden, 1]</c>.</summary>
    public required Tensor AttnResMlpQuery { get; init; }

    /// <summary>Key norm of <see cref="AttnResMlpQuery"/>.</summary>
    public required Tensor AttnResMlpNorm { get; init; }

    /// <summary>KDA weights, on KDA layers.</summary>
    public KdaWeights? Kda { get; init; }

    /// <summary>NoPE MLA weights with the DSA indexer, on MLA layers.</summary>
    public DsaAttentionWeights? Mla { get; init; }

    /// <summary>Dense FFN gate projection, on dense layers.</summary>
    public Tensor? DenseGate { get; init; }

    /// <summary>Dense FFN up projection, on dense layers.</summary>
    public Tensor? DenseUp { get; init; }

    /// <summary>Dense FFN down projection, on dense layers.</summary>
    public Tensor? DenseDown { get; init; }

    /// <summary>The latent MoE block's weights keyed as <see cref="MoeFeedForward.LoadWeights"/> reads them, under <see cref="MoePrefix"/>.</summary>
    public IReadOnlyDictionary<string, Tensor>? Moe { get; init; }

    /// <summary>MXFP4-packed routed experts, one per expert, when the checkpoint stores them packed; null for F32 experts.</summary>
    public IReadOnlyList<PackedMxfp4Expert>? PackedExperts { get; init; }

    /// <summary>Key prefix of <see cref="Moe"/>.</summary>
    public string MoePrefix { get; init; } = "layer";
}

/// <summary>The non-layer weights of a Kimi K3.1 decoder and the per-layer weights.</summary>
public sealed record KimiK31Weights
{
    /// <summary>Token embedding, <c>[Vocab, Hidden]</c>.</summary>
    public required Tensor Embedding { get; init; }

    /// <summary>Final RMSNorm weight, <c>[Hidden]</c>.</summary>
    public required Tensor FinalNorm { get; init; }

    /// <summary>LM head, <c>[Vocab, Hidden]</c>.</summary>
    public required Tensor LmHead { get; init; }

    /// <summary>Output attention-residual pseudo-query, <c>[Hidden, 1]</c>, and its key norm.</summary>
    public required Tensor OutputResQuery { get; init; }

    /// <summary>Key norm of <see cref="OutputResQuery"/>.</summary>
    public required Tensor OutputResNorm { get; init; }

    /// <summary>Per-layer weights, one per decoder layer.</summary>
    public required IReadOnlyList<KimiK31LayerWeights> Layers { get; init; }

    /// <summary>The tensors the loader created for this decoder; the owner disposes them after the model.</summary>
    public IReadOnlyList<Tensor> Owned { get; init; } = [];
}

/// <summary>The per-sequence state of a Kimi K3.1 decoder: each KDA layer's conv and recurrent state and each MLA layer's latent and
/// indexer caches, per tensor-parallel rank (each rank keeps the caches of its own heads), plus the attention-residual block summaries,
/// which every rank holds identically.</summary>
public sealed class KimiK31State : IDisposable
{
    private readonly KdaState?[][] _kda;      // [rank][layer]
    private readonly DsaAttentionState?[][] _mla;
    private bool _disposed;

    /// <summary>Creates an empty single-rank state for <paramref name="config"/> holding up to <paramref name="capacity"/> tokens.</summary>
    public KimiK31State(KimiK31Config config, int capacity) : this(config, capacity, ranks: 1)
    {
    }

    /// <summary>Creates an empty state split over <paramref name="ranks"/> tensor-parallel ranks, holding up to <paramref name="capacity"/> tokens.</summary>
    /// <exception cref="ArgumentException">The config does not split across <paramref name="ranks"/>.</exception>
    public KimiK31State(KimiK31Config config, int capacity, int ranks)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentOutOfRangeException.ThrowIfLessThan(capacity, 1);
        KimiK31Sharding.Require(config, ranks);
        Capacity = capacity;
        Ranks = ranks;
        _kda = new KdaState?[ranks][];
        _mla = new DsaAttentionState?[ranks][];
        KdaSettings kda = KimiK31Sharding.KdaSettingsFor(config, ranks);
        DsaAttentionSettings mla = KimiK31Sharding.MlaSettingsFor(config, ranks);
        try
        {
            for (int r = 0; r < ranks; r++)
            {
                _kda[r] = new KdaState?[config.Layers];
                _mla[r] = new DsaAttentionState?[config.Layers];
                for (int i = 0; i < config.Layers; i++)
                {
                    if (config.IsMla(i)) _mla[r][i] = new DsaAttentionState(capacity, mla);
                    else _kda[r][i] = new KdaState(kda);
                }
            }
        }
        catch
        {
            Dispose();
            throw;
        }
        Blocks = [];
        Partial = new float[config.Hidden];
    }

    /// <summary>Tokens the state holds.</summary>
    public int Capacity { get; }

    /// <summary>Tensor-parallel ranks the state is split over.</summary>
    public int Ranks { get; }

    /// <summary>Tokens processed so far.</summary>
    public int Position { get; internal set; }

    /// <summary>Completed attention-residual block summaries, each <c>[Hidden]</c>. The first is the embedding.</summary>
    public List<float[]> Blocks { get; }

    /// <summary>The running partial sum of the current block, <c>[Hidden]</c>.</summary>
    public float[] Partial { get; }

    internal KdaState Kda(int rank, int layer) => _kda[rank][layer] ?? throw new InvalidOperationException($"Layer {layer} is not a KDA layer.");

    internal DsaAttentionState Mla(int rank, int layer) => _mla[rank][layer] ?? throw new InvalidOperationException($"Layer {layer} is not an MLA layer.");

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        for (int r = 0; r < _kda.Length; r++)
        {
            foreach (KdaState? k in _kda[r]) k?.Dispose();
            foreach (DsaAttentionState? m in _mla[r]) m?.Dispose();
        }
    }
}

/// <summary>A Kimi K3.1 text decoder on the shared backend ops, sharded over tensor-parallel ranks. Each rank runs its heads of every
/// attention layer, its share of the dense and shared FFN, its contiguous range of routed experts and its vocabulary rows. The residual
/// stream is replicated: after each sublayer the ranks' partial outputs are summed by one AllReduce. Routed experts run expert-parallel
/// (<see cref="ExpertParallelMoe"/>). A single backend is the one-rank case, with no collectives. Each token runs through every layer
/// (attention then FFN, each mixed in by a block attention residual), so chunked prefill and decode share one path and carry the same state.</summary>
public sealed class KimiK31Model : IDisposable
{
    private readonly IReadOnlyList<IBackend> _backends;
    private readonly ICollectiveComm? _comm;
    private readonly KimiK31Config _config;
    private readonly KimiK31Weights _weights;
    private readonly IReadOnlyList<KimiK31Weights> _rankWeights;
    private readonly int _ranks;
    private readonly KdaSettings _kdaSettings;
    private readonly RankShard[] _shards;
    private readonly ExpertParallelMoe?[] _experts;
    private readonly List<Tensor> _owned = [];
    private bool _disposed;

    /// <summary>Builds the single-backend decoder. <paramref name="maxPositions"/> sizes the identity rotary tables the NoPE attention reads.</summary>
    public KimiK31Model(IBackend backend, KimiK31Config config, KimiK31Weights weights, int maxPositions)
        : this([backend ?? throw new ArgumentNullException(nameof(backend))], null, config,
            new KimiK31Sharding.RankWeightSet([weights ?? throw new ArgumentNullException(nameof(weights))], []), maxPositions)
    {
    }

    /// <summary>Builds the decoder sharded over the placement's ranks, cutting every rank's share out of <paramref name="weights"/> in memory.
    /// A rank count the dimensions do not divide is refused by name.</summary>
    /// <param name="placement">One backend per rank and the communicator over them.</param>
    /// <param name="maxPositions">Sizes the identity rotary tables the NoPE attention reads.</param>
    public KimiK31Model(TpPlacement placement, KimiK31Config config, KimiK31Weights weights, int maxPositions)
        : this(RankBackendsOf(placement), placement.Comm, config, KimiK31Sharding.Slice(config, weights, placement.Degree), maxPositions)
    {
    }

    /// <summary>Builds the decoder over the placement's ranks from each rank's own weights, as <see cref="KimiK31WeightLoader"/> loads them.
    /// The model does not dispose <paramref name="rankWeights"/>; their owner does, after the model.</summary>
    /// <param name="rankWeights">One entry per rank, in rank order, each holding only that rank's share.</param>
    public KimiK31Model(TpPlacement placement, KimiK31Config config, IReadOnlyList<KimiK31Weights> rankWeights, int maxPositions)
        : this(RankBackendsOf(placement), placement.Comm, config, new KimiK31Sharding.RankWeightSet(rankWeights ?? throw new ArgumentNullException(nameof(rankWeights)), []), maxPositions)
    {
    }

    private static IReadOnlyList<IBackend> RankBackendsOf(TpPlacement placement) =>
        (placement ?? throw new ArgumentNullException(nameof(placement))).RankBackends;

    /// <summary>Tensor-parallel ranks the decoder runs on; a sequence state must be split the same way.</summary>
    public int Ranks => _ranks;

    private KimiK31Model(IReadOnlyList<IBackend> backends, ICollectiveComm? comm, KimiK31Config config, KimiK31Sharding.RankWeightSet set, int maxPositions)
    {
        _config = config ?? throw new ArgumentNullException(nameof(config));
        ArgumentNullException.ThrowIfNull(backends);
        ArgumentNullException.ThrowIfNull(set);
        _ranks = backends.Count;
        KimiK31Sharding.Require(config, _ranks);
        if (set.Ranks.Count != _ranks) throw new ArgumentException($"Expected weights for {_ranks} ranks, got {set.Ranks.Count}.", nameof(set));
        foreach (KimiK31Weights rankWeights in set.Ranks)
        {
            ArgumentNullException.ThrowIfNull(rankWeights, nameof(set));
            if (rankWeights.Layers.Count != config.Layers)
                throw new ArgumentException($"Expected {config.Layers} layers, got {rankWeights.Layers.Count}.", nameof(set));
        }
        if (_ranks > 1 && (comm is null || comm.RankCount != _ranks))
            throw new ArgumentException($"A Kimi K3.1 decoder over {_ranks} ranks needs a communicator spanning every rank.", nameof(comm));
        _rankWeights = set.Ranks;
        _weights = set.Ranks[0];
        _owned.AddRange(set.Owned);
        _backends = backends;
        _comm = comm;
        _kdaSettings = KimiK31Sharding.KdaSettingsFor(config, _ranks);
        _shards = new RankShard[_ranks];
        _experts = new ExpertParallelMoe?[config.Layers];
        try
        {
            for (int r = 0; r < _ranks; r++)
            {
                _shards[r] = new RankShard(config.Layers);
                BuildShard(_shards[r], r, maxPositions);
            }
            for (int i = 0; i < config.Layers; i++)
            {
                if (config.IsDense(i)) continue;
                MoeFeedForward[] blocks = new MoeFeedForward[_ranks];
                for (int r = 0; r < _ranks; r++) blocks[r] = _shards[r].Moe[i]!;
                _experts[i] = new ExpertParallelMoe(_backends, blocks, _comm, config.Experts, config.ExpertsPerToken);
            }
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    /// <summary>The latent SiTU MoE block's shape.</summary>
    public static MoeConfig MoeSettings(KimiK31Config c) => new()
    {
        NumExperts = c.Experts,
        NumExpertsPerTok = c.ExpertsPerToken,
        MoeIntermediateSize = c.MoeIntermediate,
        NormTopKProb = true,
        Scoring = MoeScoring.Sigmoid,
        SharedExpertIntermediateSize = c.SharedExperts * c.MoeIntermediate,
        Activation = ActivationKind.Situ,
        ActivationSituBeta = c.SituBeta,
        ActivationSituLinearBeta = c.SituLinearBeta,
        FirstDenseLayers = c.DenseLayers,
        RoutedScalingFactor = 1f,
        SplitFeatureRouting = false,
        LearnedRoutedScale = false,
        LatentSize = c.LatentSize,
        LatentNorm = MoeLatentNorm.AfterCombine,
        LatentNormEps = c.RmsNormEps,
    };

    /// <summary>Runs one token through the decoder, advancing <paramref name="state"/>, and returns its logits <c>[Vocab]</c>.</summary>
    public float[] ForwardToken(int token, KimiK31State state) => ProjectHidden(ForwardTokenHidden(token, state));

    /// <summary>Runs one token through the decoder, advancing <paramref name="state"/>, and returns its final normed hidden row <c>[Hidden]</c>.</summary>
    public float[] ForwardTokenHidden(int token, KimiK31State state)
    {
        ArgumentNullException.ThrowIfNull(state);
        if (state.Position >= state.Capacity) throw new InvalidOperationException("The sequence state is full.");
        if (state.Ranks != _ranks) throw new ArgumentException($"The state is split over {state.Ranks} ranks; this decoder runs on {_ranks}.", nameof(state));
        float[] x = Embedding(token);
        // The depth-residual stack is per token: each token's block summaries start from its own embedding. Only the KDA and
        // MLA caches carry across calls.
        state.Blocks.Clear();
        state.Blocks.Add(x);
        Array.Clear(state.Partial);
        for (int i = 0; i < _config.Layers; i++)
        {
            if (i > 0 && i % _config.AttnResBlock == 0)
            {
                state.Blocks.Add(state.Partial.ToArray());
                Array.Clear(state.Partial);
            }
            KimiK31LayerWeights lw = _weights.Layers[i];
            float[] mixedAttn = Mix(state, lw.AttnResAttnQuery, lw.AttnResAttnNorm, AttentionMixIncludesPartial(i, _config.AttnResBlock));
            float[] normAttn = Normalize(mixedAttn, lw.InputNorm);
            float[] attn = Attention(i, normAttn, state);
            AddInto(state.Partial, attn);

            float[] mixedMlp = Mix(state, lw.AttnResMlpQuery, lw.AttnResMlpNorm, includePartial: true);
            float[] normMlp = Normalize(mixedMlp, lw.PostNorm);
            float[] ffn = Ffn(i, normMlp);
            AddInto(state.Partial, ffn);
        }
        float[] head = Mix(state, _weights.OutputResQuery, _weights.OutputResNorm, includePartial: true);
        float[] final = Normalize(head, _weights.FinalNorm);
        state.Position++;
        return final;
    }

    /// <summary>Projects a final hidden row <c>[Hidden]</c> through the LM head to logits <c>[Vocab]</c>.</summary>
    public float[] ProjectHidden(float[] hidden)
    {
        ArgumentNullException.ThrowIfNull(hidden);
        if (hidden.Length != _config.Hidden) throw new ArgumentException($"Expected {_config.Hidden} hidden values.", nameof(hidden));
        return Logits(hidden);
    }

    /// <summary>Runs <paramref name="tokens"/> through the decoder in order and returns the last token's logits.</summary>
    public float[] Forward(ReadOnlySpan<int> tokens, KimiK31State state)
    {
        if (tokens.Length == 0) throw new ArgumentException("At least one token is required.", nameof(tokens));
        float[] logits = Array.Empty<float>();
        foreach (int t in tokens) logits = ForwardToken(t, state);
        return logits;
    }

    // Rank r's share of every layer, from the rank's own weights: its vocabulary rows, its attention heads, its FFN columns and its routed
    // experts. The weights are already cut to these shares, so nothing is sliced here.
    private void BuildShard(RankShard s, int r, int maxPositions)
    {
        KimiK31Config c = _config;
        KimiK31Weights rw = _rankWeights[r];
        IBackend backend = _backends[r];
        long perRankVocab = c.Vocab / _ranks;
        if (rw.Embedding.Shape[0] != perRankVocab || rw.LmHead.Shape[0] != perRankVocab)
            throw new ArgumentException($"Rank {r} holds {rw.Embedding.Shape[0]} vocabulary rows; the decoder needs {perRankVocab}.", "rankWeights");
        s.Embedding = rw.Embedding;
        s.Head = rw.LmHead;
        for (int i = 0; i < c.Layers; i++)
        {
            KimiK31LayerWeights lw = rw.Layers[i];
            if (c.IsMla(i))
            {
                DsaAttentionWeights mla = lw.Mla ?? throw new ArgumentException($"Layer {i} is MLA but has no MLA weights.", "rankWeights");
                s.Mla[i] = new DsaAttention(backend, KimiK31Sharding.MlaSettingsFor(c, _ranks), mla,
                    new IdentityRopeRows(c.QkRopeDim / 2, maxPositions), new IdentityRopeRows(1, maxPositions));
            }
            else
            {
                s.Kda[i] = lw.Kda ?? throw new ArgumentException($"Layer {i} is KDA but has no KDA weights.", "rankWeights");
            }
            if (c.IsDense(i))
            {
                if (lw.DenseGate is null || lw.DenseUp is null || lw.DenseDown is null)
                    throw new ArgumentException($"Layer {i} is dense but has no dense FFN weights.", "rankWeights");
                s.Ffn[i] = new FfnWeights(lw.DenseGate, lw.DenseUp, lw.DenseDown);
                continue;
            }
            if (lw.Moe is null) throw new ArgumentException($"Layer {i} is MoE but has no MoE weights.", "rankWeights");
            if (c.SharedExperts > 0)
            {
                string shared = lw.MoePrefix + ".mlp.shared_expert.";
                s.Shared[i] = new FfnWeights(lw.Moe[shared + "gate_proj.weight"], lw.Moe[shared + "up_proj.weight"], lw.Moe[shared + "down_proj.weight"]);
            }
            s.Moe[i] = BuildMoeBlock(lw, r);
        }
    }

    // Rank r's routed block: the router, latent projections and norms, the experts of its range (by global id) or their packed form.
    private MoeFeedForward BuildMoeBlock(KimiK31LayerWeights lw, int r)
    {
        KimiK31Config c = _config;
        int perRank = KimiK31Sharding.ExpertsPerRank(c, _ranks), first = r * perRank;
        MoeFeedForward moe = new(KimiK31Sharding.BlockSettings(c), c.Hidden, lowVram: false);
        moe.ShardExperts(first, perRank);
        moe.LoadWeights(lw.Moe!, lw.MoePrefix);
        if (lw.PackedExperts is not null) moe.LoadPackedExperts(lw.PackedExperts);
        return moe;
    }

    private float[] Embedding(int token)
    {
        int d = _config.Hidden, v = _config.Vocab, perRank = v / _ranks;
        if ((uint)token >= (uint)v) throw new ArgumentOutOfRangeException(nameof(token));
        // The owning rank holds the row; every other rank contributes zeros, and the sum is the row.
        int owner = token / perRank;
        Tensor[] parts = new Tensor[_ranks];
        try
        {
            for (int r = 0; r < _ranks; r++)
            {
                parts[r] = new Tensor(new TensorShape(1, d), DType.F32);
                parts[r].AsSpan<float>().Clear();
            }
            _shards[owner].Embedding.AsReadOnlySpan<float>().Slice((token - owner * perRank) * d, d).CopyTo(parts[owner].AsSpan<float>());
            return Reduce(parts);
        }
        finally
        {
            DisposeAll(parts);
        }
    }

    /// <summary>Whether a layer's attention mix reads the running partial sum. The partial exists only inside a block: the first layer of
    /// each block has none yet, so its mix reads the completed block summaries alone (NeMo <c>_forward_attn_residual</c>).</summary>
    internal static bool AttentionMixIncludesPartial(int layer, int attnResBlock) => layer % attnResBlock != 0;

    // The block attention residual over the completed block summaries, plus the running partial sum when the caller includes it.
    internal float[] Mix(KimiK31State state, Tensor query, Tensor normWeight, bool includePartial)
    {
        int d = _config.Hidden;
        int n = state.Blocks.Count + (includePartial ? 1 : 0);
        float[] stacked = new float[n * d];
        for (int b = 0; b < state.Blocks.Count; b++) state.Blocks[b].CopyTo(stacked.AsSpan(b * d, d));
        if (includePartial) state.Partial.CopyTo(stacked.AsSpan((n - 1) * d, d));
        using Tensor sources = Make(stacked, n, d);
        using Tensor output = new(new TensorShape(1, d), DType.F32);
        AttentionResidual.Mix(_backends[0], sources, normWeight, query, output);
        return output.AsReadOnlySpan<float>().ToArray();
    }

    private float[] Normalize(float[] x, Tensor weight)
    {
        int d = _config.Hidden;
        using Tensor input = Make(x, 1, d);
        using Tensor output = new(new TensorShape(1, d), DType.F32);
        _backends[0].RmsNorm(output, input, weight, _config.RmsNormEps);
        return output.AsReadOnlySpan<float>().ToArray();
    }

    // Every rank runs its heads on the same normalized input; the output projections are row-parallel, so the partials sum to the layer.
    private float[] Attention(int layer, float[] x, KimiK31State state)
    {
        int d = _config.Hidden;
        Tensor[] parts = new Tensor[_ranks];
        try
        {
            for (int r = 0; r < _ranks; r++)
            {
                if (_config.IsMla(layer))
                {
                    float[] y = new float[d];
                    _shards[r].Mla[layer]!.Forward(x, 1, state.Position, state.Mla(r, layer), y);
                    parts[r] = Make(y, 1, d);
                    continue;
                }
                using Tensor input = Make(x, 1, d);
                parts[r] = new Tensor(new TensorShape(1, d), DType.F32);
                KdaLayer.Forward(_backends[r], _kdaSettings, _shards[r].Kda[layer]!, input, parts[r], state.Kda(r, layer), _config.RmsNormEps);
            }
            return Reduce(parts);
        }
        finally
        {
            DisposeAll(parts);
        }
    }

    private float[] Ffn(int layer, float[] x) => _config.IsDense(layer) ? DenseFfn(layer, x) : MoeFfn(layer, x);

    private float[] DenseFfn(int layer, float[] x)
    {
        int d = _config.Hidden;
        Tensor[] parts = new Tensor[_ranks];
        try
        {
            for (int r = 0; r < _ranks; r++)
            {
                FfnWeights f = _shards[r].Ffn[layer]!.Value;
                using Tensor input = Make(x, 1, 1, d);
                parts[r] = new Tensor(new TensorShape(1, 1, d), DType.F32);
                DenseGluFfn.Apply(_backends[r], input, f.Gate, f.Up, f.Down, parts[r], _config.DenseIntermediate / _ranks,
                    ActivationKind.Situ, _config.SituBeta, _config.SituLinearBeta);
            }
            return Reduce(parts);
        }
        finally
        {
            DisposeAll(parts);
        }
    }

    // Routed experts run expert-parallel into the per-rank accumulators; each rank then adds its shared-expert partial, and one AllReduce
    // sums the lot.
    internal float[] MoeFfn(int layer, float[] x)
    {
        int d = _config.Hidden;
        Tensor[] inputs = new Tensor[_ranks];
        Tensor[] accum = new Tensor[_ranks];
        try
        {
            for (int r = 0; r < _ranks; r++)
            {
                inputs[r] = Make(x, 1, 1, d);
                accum[r] = new Tensor(new TensorShape(1, 1, d), DType.F32);
            }
            _experts[layer]!.Forward(inputs, accum, 1);
            if (_config.SharedExperts > 0)
            {
                for (int r = 0; r < _ranks; r++)
                {
                    FfnWeights s = _shards[r].Shared[layer]!.Value;
                    using Tensor shared = new(new TensorShape(1, 1, d), DType.F32);
                    DenseGluFfn.Apply(_backends[r], inputs[r], s.Gate, s.Up, s.Down, shared, _config.SharedExperts * _config.MoeIntermediate / _ranks,
                        ActivationKind.Situ, _config.SituBeta, _config.SituLinearBeta);
                    Span<float> sum = accum[r].AsSpan<float>();
                    ReadOnlySpan<float> part = shared.AsReadOnlySpan<float>();
                    for (int j = 0; j < d; j++) sum[j] += part[j];
                }
            }
            return Reduce(accum);
        }
        finally
        {
            DisposeAll(inputs);
            DisposeAll(accum);
        }
    }

    private float[] Logits(float[] final)
    {
        int d = _config.Hidden, v = _config.Vocab, perRank = v / _ranks;
        Tensor[] shards = new Tensor[_ranks];
        Tensor[] full = new Tensor[_ranks];
        try
        {
            for (int r = 0; r < _ranks; r++)
            {
                using Tensor input = Make(final, 1, d);
                shards[r] = new Tensor(new TensorShape(1, perRank), DType.F32);
                _backends[r].Linear(shards[r], input, _shards[r].Head, null);
            }
            if (_ranks == 1) return shards[0].AsReadOnlySpan<float>().ToArray();
            for (int r = 0; r < _ranks; r++) full[r] = new Tensor(new TensorShape(1, v), DType.F32);
            _comm!.AllGather(shards, full);
            return full[0].AsReadOnlySpan<float>().ToArray();
        }
        finally
        {
            DisposeAll(shards);
            DisposeAll(full);
        }
    }

    // Sums the ranks' partials in place with one AllReduce and returns rank 0's copy. The caller disposes the parts.
    private float[] Reduce(IReadOnlyList<Tensor> parts)
    {
        if (_ranks > 1) _comm!.AllReduceSum(parts);
        return parts[0].AsReadOnlySpan<float>().ToArray();
    }

    private static void AddInto(float[] acc, float[] x)
    {
        for (int i = 0; i < acc.Length; i++) acc[i] += x[i];
    }

    private static void DisposeAll(Tensor?[] tensors)
    {
        foreach (Tensor? t in tensors) t?.Dispose();
    }

    private static Tensor Make(float[] values, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        values.CopyTo(t.AsSpan<float>());
        return t;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (RankShard? s in _shards) s?.DisposeAttention();
        foreach (Tensor t in _owned) t.Dispose();
        _owned.Clear();
    }

    private readonly record struct FfnWeights(Tensor Gate, Tensor Up, Tensor Down);

    // One rank's share of the decoder. The attention, FFN and MoE entries are null on the layers that do not use them.
    private sealed class RankShard(int layers)
    {
        public Tensor Embedding { get; set; } = null!;

        public Tensor Head { get; set; } = null!;

        public KdaWeights?[] Kda { get; } = new KdaWeights?[layers];

        public DsaAttention?[] Mla { get; } = new DsaAttention?[layers];

        public FfnWeights?[] Ffn { get; } = new FfnWeights?[layers];

        public FfnWeights?[] Shared { get; } = new FfnWeights?[layers];

        public MoeFeedForward?[] Moe { get; } = new MoeFeedForward?[layers];

        public void DisposeAttention()
        {
            foreach (DsaAttention? m in Mla) m?.Dispose();
        }
    }
}
