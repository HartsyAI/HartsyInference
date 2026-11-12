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

/// <summary>The per-sequence state of a Kimi K3.1 decoder: each KDA layer's conv and recurrent state, each MLA layer's latent and
/// indexer caches, and the attention-residual block summaries.</summary>
public sealed class KimiK31State : IDisposable
{
    private readonly KdaState?[] _kda;
    private readonly DsaAttentionState?[] _mla;
    private bool _disposed;

    /// <summary>Creates an empty state for <paramref name="config"/> holding up to <paramref name="capacity"/> tokens.</summary>
    public KimiK31State(KimiK31Config config, int capacity)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentOutOfRangeException.ThrowIfLessThan(capacity, 1);
        Capacity = capacity;
        _kda = new KdaState?[config.Layers];
        _mla = new DsaAttentionState?[config.Layers];
        for (int i = 0; i < config.Layers; i++)
        {
            if (config.IsMla(i)) _mla[i] = new DsaAttentionState(capacity, config.Mla);
            else _kda[i] = new KdaState(config.Kda);
        }
        Blocks = [];
        Partial = new float[config.Hidden];
    }

    /// <summary>Tokens the state holds.</summary>
    public int Capacity { get; }

    /// <summary>Tokens processed so far.</summary>
    public int Position { get; internal set; }

    /// <summary>Completed attention-residual block summaries, each <c>[Hidden]</c>. The first is the embedding.</summary>
    public List<float[]> Blocks { get; }

    /// <summary>The running partial sum of the current block, <c>[Hidden]</c>.</summary>
    public float[] Partial { get; }

    internal KdaState Kda(int layer) => _kda[layer] ?? throw new InvalidOperationException($"Layer {layer} is not a KDA layer.");

    internal DsaAttentionState Mla(int layer) => _mla[layer] ?? throw new InvalidOperationException($"Layer {layer} is not an MLA layer.");

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (KdaState? k in _kda) k?.Dispose();
        foreach (DsaAttentionState? m in _mla) m?.Dispose();
    }
}

/// <summary>A Kimi K3.1 text decoder on the shared backend ops. Each token runs through every layer: the attention sublayer (KDA or NoPE
/// MLA with the DSA indexer) and the FFN sublayer (dense SiTU or latent MoE), each mixed in by a block attention residual over the
/// block summaries and the running partial sum. The model processes one token per call, so chunked prefill and decode share one path
/// and carry the same state.</summary>
public sealed class KimiK31Model : IDisposable
{
    private readonly IBackend _backend;
    private readonly KimiK31Config _config;
    private readonly KimiK31Weights _weights;
    private readonly DsaAttention?[] _mla;
    private readonly MoeFeedForward?[] _moe;
    private bool _disposed;

    /// <summary>Builds the model. <paramref name="maxPositions"/> sizes the identity rotary tables the NoPE attention reads.</summary>
    public KimiK31Model(IBackend backend, KimiK31Config config, KimiK31Weights weights, int maxPositions)
    {
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        _config = config ?? throw new ArgumentNullException(nameof(config));
        _weights = weights ?? throw new ArgumentNullException(nameof(weights));
        if (weights.Layers.Count != config.Layers) throw new ArgumentException($"Expected {config.Layers} layers, got {weights.Layers.Count}.", nameof(weights));
        _mla = new DsaAttention?[config.Layers];
        _moe = new MoeFeedForward?[config.Layers];
        try
        {
            for (int i = 0; i < config.Layers; i++)
            {
                KimiK31LayerWeights lw = weights.Layers[i];
                if (config.IsMla(i))
                {
                    if (lw.Mla is null) throw new ArgumentException($"Layer {i} is MLA but has no MLA weights.", nameof(weights));
                    _mla[i] = new DsaAttention(backend, config.Mla, lw.Mla,
                        new IdentityRopeRows(config.QkRopeDim / 2, maxPositions), new IdentityRopeRows(1, maxPositions));
                }
                else if (lw.Kda is null) throw new ArgumentException($"Layer {i} is KDA but has no KDA weights.", nameof(weights));
                if (!config.IsDense(i))
                {
                    if (lw.Moe is null) throw new ArgumentException($"Layer {i} is MoE but has no MoE weights.", nameof(weights));
                    _moe[i] = new MoeFeedForward(MoeSettings(config), config.Hidden, lowVram: false);
                    _moe[i]!.LoadWeights(lw.Moe, lw.MoePrefix);
                    if (lw.PackedExperts is not null) _moe[i]!.LoadPackedExperts(lw.PackedExperts);
                }
                else if (lw.DenseGate is null || lw.DenseUp is null || lw.DenseDown is null)
                {
                    throw new ArgumentException($"Layer {i} is dense but has no dense FFN weights.", nameof(weights));
                }
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
        int d = _config.Hidden;
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
            float[] mixedAttn = Mix(state, lw.AttnResAttnQuery, lw.AttnResAttnNorm);
            float[] normAttn = Normalize(mixedAttn, lw.InputNorm);
            float[] attn = Attention(i, normAttn, state);
            AddInto(state.Partial, attn);

            float[] mixedMlp = Mix(state, lw.AttnResMlpQuery, lw.AttnResMlpNorm);
            float[] normMlp = Normalize(mixedMlp, lw.PostNorm);
            float[] ffn = Ffn(i, normMlp);
            AddInto(state.Partial, ffn);
        }
        float[] head = Mix(state, _weights.OutputResQuery, _weights.OutputResNorm);
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

    private float[] Embedding(int token)
    {
        int d = _config.Hidden;
        if ((uint)token >= (uint)_config.Vocab) throw new ArgumentOutOfRangeException(nameof(token));
        float[] row = new float[d];
        _weights.Embedding.AsReadOnlySpan<float>().Slice(token * d, d).CopyTo(row);
        return row;
    }

    // The block attention residual over the completed block summaries and the partial sum, for one token.
    private float[] Mix(KimiK31State state, Tensor query, Tensor normWeight)
    {
        int d = _config.Hidden;
        int n = state.Blocks.Count + 1;
        float[] stacked = new float[n * d];
        for (int b = 0; b < state.Blocks.Count; b++) state.Blocks[b].CopyTo(stacked.AsSpan(b * d, d));
        state.Partial.CopyTo(stacked.AsSpan((n - 1) * d, d));
        using Tensor sources = Make(stacked, n, d);
        using Tensor output = new(new TensorShape(1, d), DType.F32);
        AttentionResidual.Mix(_backend, sources, normWeight, query, output);
        return output.AsReadOnlySpan<float>().ToArray();
    }

    private float[] Normalize(float[] x, Tensor weight)
    {
        int d = _config.Hidden;
        using Tensor input = Make(x, 1, d);
        using Tensor output = new(new TensorShape(1, d), DType.F32);
        _backend.RmsNorm(output, input, weight, _config.RmsNormEps);
        return output.AsReadOnlySpan<float>().ToArray();
    }

    private float[] Attention(int layer, float[] x, KimiK31State state)
    {
        int d = _config.Hidden;
        KimiK31LayerWeights lw = _weights.Layers[layer];
        float[] y = new float[d];
        if (_config.IsMla(layer))
        {
            _mla[layer]!.Forward(x, 1, state.Position, state.Mla(layer), y);
            return y;
        }
        using Tensor input = Make(x, 1, d);
        using Tensor output = new(new TensorShape(1, d), DType.F32);
        KdaLayer.Forward(_backend, _config.Kda, lw.Kda!, input, output, state.Kda(layer), _config.RmsNormEps);
        return output.AsReadOnlySpan<float>().ToArray();
    }

    private float[] Ffn(int layer, float[] x)
    {
        int d = _config.Hidden;
        KimiK31LayerWeights lw = _weights.Layers[layer];
        using Tensor input = Make(x, 1, 1, d);
        float[] y = new float[d];
        if (_config.IsDense(layer))
        {
            using Tensor output = new(new TensorShape(1, 1, d), DType.F32);
            DenseGluFfn.Apply(_backend, input, lw.DenseGate!, lw.DenseUp!, lw.DenseDown!, output, _config.DenseIntermediate,
                ActivationKind.Situ, _config.SituBeta, _config.SituLinearBeta);
            output.AsReadOnlySpan<float>().CopyTo(y);
            return y;
        }
        using Tensor routed = _moe[layer]!.Forward(_backend, input, 1);
        routed.AsReadOnlySpan<float>().CopyTo(y);
        return y;
    }

    private float[] Logits(float[] final)
    {
        int d = _config.Hidden, v = _config.Vocab;
        using Tensor input = Make(final, 1, d);
        using Tensor logits = new(new TensorShape(1, v), DType.F32);
        _backend.Linear(logits, input, _weights.LmHead, null);
        return logits.AsReadOnlySpan<float>().ToArray();
    }

    private static void AddInto(float[] acc, float[] x)
    {
        for (int i = 0; i < acc.Length; i++) acc[i] += x[i];
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
        foreach (DsaAttention? m in _mla) m?.Dispose();
    }
}
