using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>The DSpark draft head of V4.1 (upstream <c>mtp.0-2</c> with <c>DSparkBlock</c>): three stages that draft a block of tokens from the target's hidden states,
/// then a Markov head that adds each drafted token's bias to the next position, and a confidence head.</summary>
/// <remarks>Greedy drafting only: each draft token is the argmax, which is what upstream does at temperature 0. The embedding and head are the target model's.
/// <see cref="Draft"/> writes the current target latent into its window slot, so a caller that rolls the target back must also restore the window slots that later positions
/// overwrote; the proposer loop owns that. <see cref="Seed"/> expects a fresh state.</remarks>
public sealed class DeepSeekV41DSpark
{
    private readonly IBackend _backend;
    private readonly int _dim, _hc, _vocab, _rank, _blockSize, _noiseTokenId;
    private readonly float _normEps;
    private readonly DeepSeekV41Weight _mainProj;
    private readonly int _mainIn;
    private readonly float[] _mainNorm;
    private readonly DeepSeekV41Weight _embed;
    private readonly DeepSeekV41Weight _head;
    private readonly DeepSeekV41DSparkStage[] _stages;
    private readonly float[] _finalNorm;
    private readonly DeepSeekV41Weight _markovEmbed;
    private readonly DeepSeekV41Weight _markovHead;
    private readonly float[] _confidence;

    /// <summary>Diagnostic tap: called with a kind (<c>attn</c>, <c>ffn</c>, <c>x</c>), the stage index and the values, during <see cref="Draft"/>. Null in normal use.</summary>
    public Action<string, int, float[]>? Probe { get; set; }

    /// <summary>Tokens drafted per call, including the position the call's input token occupies.</summary>
    public int BlockSize => _blockSize;

    /// <summary>The token that fills the draft positions after the input token.</summary>
    public int NoiseTokenId => _noiseTokenId;

    /// <summary>Number of draft stages.</summary>
    public int StageCount => _stages.Length;

    private DeepSeekV41DSpark(IBackend backend, int dim, int hc, int vocab, int rank, int blockSize, int noiseTokenId, float normEps,
        DeepSeekV41Weight mainProj, int mainIn, float[] mainNorm, DeepSeekV41Weight embed, DeepSeekV41Weight head, DeepSeekV41DSparkStage[] stages, float[] finalNorm,
        DeepSeekV41Weight markovEmbed, DeepSeekV41Weight markovHead, float[] confidence)
    {
        _backend = backend;
        _dim = dim;
        _hc = hc;
        _vocab = vocab;
        _rank = rank;
        _blockSize = blockSize;
        _noiseTokenId = noiseTokenId;
        _normEps = normEps;
        _mainProj = mainProj;
        _mainIn = mainIn;
        _mainNorm = mainNorm;
        _embed = embed;
        _head = head;
        _stages = stages;
        _finalNorm = finalNorm;
        _markovEmbed = markovEmbed;
        _markovHead = markovHead;
        _confidence = confidence;
    }

    /// <summary>Loads the draft stages and their heads from the <c>mtp.*</c> keys of the checkpoint, reading the target's embedding and head from it too.</summary>
    public static DeepSeekV41DSpark Load(IBackend backend, DeepSeekV41Checkpoint checkpoint, DeepSeekV41LoadOptions options)
    {
        ArgumentNullException.ThrowIfNull(checkpoint);
        ArgumentNullException.ThrowIfNull(options);
        DeepSeekV41HostModelLoader.Reader read = new(checkpoint, options.Residency);
        DeepSeekV41Config config = checkpoint.Config;
        return Load(backend, checkpoint, options, read, read.Weight("embed.weight", config.VocabSize, config.HiddenSize),
            read.Weight("head.weight", config.VocabSize, config.HiddenSize));
    }

    /// <summary>Loads the draft stages and their heads, sharing the target model's <paramref name="embed"/> and <paramref name="head"/> instead of reading them again.</summary>
    public static DeepSeekV41DSpark Load(IBackend backend, DeepSeekV41Checkpoint checkpoint, DeepSeekV41LoadOptions options, DeepSeekV41Weight embed, DeepSeekV41Weight head)
    {
        ArgumentNullException.ThrowIfNull(checkpoint);
        ArgumentNullException.ThrowIfNull(embed);
        ArgumentNullException.ThrowIfNull(head);
        ArgumentNullException.ThrowIfNull(options);
        return Load(backend, checkpoint, options, new DeepSeekV41HostModelLoader.Reader(checkpoint, options.Residency), embed, head);
    }

    private static DeepSeekV41DSpark Load(IBackend backend, DeepSeekV41Checkpoint checkpoint, DeepSeekV41LoadOptions options, DeepSeekV41HostModelLoader.Reader read,
        DeepSeekV41Weight embed, DeepSeekV41Weight head)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(checkpoint);
        ArgumentNullException.ThrowIfNull(options);
        options.Validate();
        DeepSeekV41Config cfg = checkpoint.Config;
        if (cfg.DsparkBlockSize < 1) throw new HartsyInferenceException("This checkpoint has no DSpark block size.");
        if (cfg.NumNextnPredictLayers < 1) throw new HartsyInferenceException("This checkpoint has no draft layers.");
        int dim = cfg.HiddenSize, hc = cfg.HcMult, vocab = cfg.VocabSize, rank = cfg.DsparkMarkovRank, heads = cfg.NumAttentionHeads, hd = cfg.HeadDim;
        int qLora = cfg.QLoraRank, targets = cfg.DsparkTargetLayerIds.Count, mix = (2 + hc) * hc;
        float normEps = (float)cfg.RmsNormEps;
        DeepSeekV41RopeTable rope = DeepSeekV41RopeTable.Build(cfg.QkRopeHeadDim, options.MaxTokens, cfg.RopeTheta, null);

        int experts = cfg.DsparkNRoutedExperts, topk = cfg.DsparkNumExpertsPerTok, inter = cfg.MoeIntermediateSize;
        DeepSeekV41ExpertCache cache = new(key =>
        {
            (int layer, int expert) = DeepSeekV41HostModelLoader.LayerExperts.Unpack(key, experts);
            return DeepSeekV41ExpertLoader.Load(checkpoint.ExpertBank(layer), expert, dim, inter, options.Residency);
        }, options.ExpertCacheCapacity);
        MoeRouteArgs route = new(experts, topk, ParseScoring(cfg.ScoringFunc), Renormalize: cfg.NormTopkProb && topk > 1, RenormEpsilon: 1e-20f,
            Scale: (float)cfg.RoutedScalingFactor);

        DeepSeekV41DSparkStage[] stages = new DeepSeekV41DSparkStage[cfg.NumNextnPredictLayers];
        for (int s = 0; s < stages.Length; s++)
        {
            string l = $"mtp.{s}.", a = l + "attn.", f = l + "ffn.";
            int layer = cfg.NumHiddenLayers + s;
            DeepSeekV41AttentionSettings settings = new DeepSeekV41AttentionSettings(dim, heads, hd, cfg.QkRopeHeadDim, qLora, cfg.OGroups, cfg.OLoraRank, cfg.SlidingWindow, 0, false, false,
                false, false, cfg.IndexNHeads, cfg.IndexHeadDim, cfg.IndexTopk, cfg.CandidateTopkBlocks, cfg.CandidateBlockSize, normEps)
                with { QuantizeLatents = options.QuantizeLatents };
            DeepSeekV41AttentionWeights weights = new(read.Weight(a + "wq_a.weight", qLora, dim), read.Vector(a + "q_norm.weight", qLora),
                read.Weight(a + "wq_b.weight", heads * hd, qLora), read.Weight(a + "wkv.weight", hd, dim), read.Vector(a + "kv_norm.weight", hd),
                read.Weight(a + "wo_a.weight", cfg.OGroups * cfg.OLoraRank, heads * hd / cfg.OGroups), read.Weight(a + "wo_b.weight", dim, cfg.OGroups * cfg.OLoraRank),
                read.Vector(a + "attn_sink", heads), null, null);
            DeepSeekV41DSparkAttention attention = new(backend, settings, weights, rope);

            DeepSeekV41SwigluWeights shared = new(dim, inter, read.Weight(f + "shared_experts.w1.weight", inter, dim),
                read.Weight(f + "shared_experts.w2.weight", dim, inter), read.Weight(f + "shared_experts.w3.weight", inter, dim));
            DeepSeekV41MoeLayer ffn = new(backend, read.Matrix(f + "gate.weight", experts, dim), read.Vector(f + "gate.bias", experts), null, route,
                new DeepSeekV41HostModelLoader.LayerExperts(cache, layer, experts), shared, (float)cfg.SwigluLimit);

            DeepSeekV41HyperConnection hcAttn = new(backend, hc, dim, cfg.HcSinkhornIters, (float)cfg.HcEps, normEps, read.Matrix(l + "hc_attn_fn", mix, hc * dim),
                read.Vector(l + "hc_attn_scale", 3), read.Vector(l + "hc_attn_base", mix));
            DeepSeekV41HyperConnection hcFfn = new(backend, hc, dim, cfg.HcSinkhornIters, (float)cfg.HcEps, normEps, read.Matrix(l + "hc_ffn_fn", mix, hc * dim),
                read.Vector(l + "hc_ffn_scale", 3), read.Vector(l + "hc_ffn_base", mix));
            stages[s] = new DeepSeekV41DSparkStage(dim, hc, normEps, hcAttn, hcFfn, read.Vector(l + "attn_norm.weight", dim), read.Vector(l + "ffn_norm.weight", dim),
                attention, ffn);
        }
        int last = stages.Length - 1;
        return new DeepSeekV41DSpark(backend, dim, hc, vocab, rank, cfg.DsparkBlockSize, cfg.DsparkNoiseTokenId, normEps,
            read.Weight("mtp.0.main_proj.weight", dim, targets * dim), targets * dim, read.Vector("mtp.0.main_norm.weight", dim), embed, head, stages,
            read.Vector($"mtp.{last}.norm.weight", dim), read.Weight($"mtp.{last}.markov_head.embed.weight", vocab, rank),
            read.Weight($"mtp.{last}.markov_head.head.weight", vocab, rank), read.Vector($"mtp.{last}.confidence_head.proj.weight", dim + rank));
    }

    /// <summary>Per-stage window state for one sequence.</summary>
    public DeepSeekV41DSparkState CreateState(int maxTokens)
    {
        DeepSeekV41AttentionState[] stages = new DeepSeekV41AttentionState[_stages.Length];
        for (int s = 0; s < stages.Length; s++) stages[s] = new DeepSeekV41AttentionState(_stages[s].AttentionSettings, maxTokens);
        return new DeepSeekV41DSparkState(stages);
    }

    /// <summary>Seeds every stage's window from the target's hidden states of the prompt; <paramref name="mainHidden"/> is <c>[tokens, 3 x Dim]</c>.</summary>
    public void Seed(ReadOnlySpan<float> mainHidden, int tokens, DeepSeekV41DSparkState state)
    {
        ArgumentNullException.ThrowIfNull(state);
        float[] mainX = MainLatents(mainHidden, tokens);
        for (int s = 0; s < _stages.Length; s++) _stages[s].Seed(mainX, tokens, state.Stages[s]);
    }

    /// <summary>Drafts one block after <paramref name="token"/>, which sits at <paramref name="startPos"/>; <paramref name="mainHidden"/> is the target's hidden states for that
    /// position <c>[3 x Dim]</c>. The first returned id is <paramref name="token"/>; the next <see cref="BlockSize"/> are the drafted tokens.</summary>
    public DeepSeekV41DSparkDraft Draft(int token, ReadOnlySpan<float> mainHidden, int startPos, DeepSeekV41DSparkState state)
    {
        ArgumentNullException.ThrowIfNull(state);
        if (token < 0 || token >= _vocab) throw new ArgumentOutOfRangeException(nameof(token), token, "The token is outside the vocabulary.");
        int block = _blockSize;
        float[] mainX = MainLatents(mainHidden, 1);

        int[] inputIds = new int[block];
        inputIds[0] = token;
        for (int b = 1; b < block; b++) inputIds[b] = _noiseTokenId;
        float[] x = new float[block * _hc * _dim], row = new float[_dim];
        for (int b = 0; b < block; b++)
        {
            _embed.CopyRow(inputIds[b], row);
            for (int c = 0; c < _hc; c++) row.CopyTo(x.AsSpan((b * _hc + c) * _dim, _dim));
        }
        float[] pre = new float[block * _hc];
        for (int b = 0; b < block; b++) pre[b * _hc] = 1f;
        for (int s = 0; s < _stages.Length; s++)
        {
            float[] next = new float[block * _hc];
            int stage = s;
            _stages[s].Draft(x, block, pre, next, startPos, mainX, state.Stages[s],
                Probe is null ? null : (kind, values) => Probe(kind, stage, values));
            pre = next;
        }

        float[] collapsed = new float[block * _dim];
        _stages[^1].Collapse(x, pre, block, collapsed);
        float[] normed = (float[])collapsed.Clone();
        DeepSeekV41HostMath.RmsNormRows(normed, _finalNorm, _dim, _normEps);
        float[] logits = _head.Linear(normed, block, _dim, _vocab);

        // greedy Markov chain: position i's logits gain the bias of the token at input i, and the argmax feeds position i+1
        int[] ids = new int[block + 1];
        ids[0] = token;
        float[][] markov = new float[block][];
        for (int i = 0; i < block; i++)
        {
            float[] e = new float[_rank];
            _markovEmbed.CopyRow(ids[i], e);
            markov[i] = e;
            float[] bias = _markovHead.Linear(e, 1, _rank, _vocab);
            for (int v = 0; v < _vocab; v++) logits[i * _vocab + v] += bias[v];
            ids[i + 1] = ArgMax(logits.AsSpan(i * _vocab, _vocab));
        }

        float[] confidence = new float[block];
        for (int b = 0; b < block; b++)
        {
            double sum = 0;
            for (int d = 0; d < _dim; d++) sum += (double)_confidence[d] * collapsed[b * _dim + d];
            for (int r = 0; r < _rank; r++) sum += (double)_confidence[_dim + r] * markov[b][r];
            confidence[b] = (float)sum;
        }
        return new DeepSeekV41DSparkDraft(ids, logits, confidence);
    }

    private float[] MainLatents(ReadOnlySpan<float> mainHidden, int tokens)
    {
        if (tokens < 1) throw new ArgumentOutOfRangeException(nameof(tokens));
        if (mainHidden.Length != (long)tokens * _mainIn) throw new ArgumentException("mainHidden must hold tokens x 3 x Dim values.", nameof(mainHidden));
        float[] mainX = _mainProj.Linear(mainHidden, tokens, _mainIn, _dim);
        DeepSeekV41HostMath.RmsNormRows(mainX, _mainNorm, _dim, _normEps);
        return mainX;
    }

    private static int ArgMax(ReadOnlySpan<float> values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++) if (values[i] > values[best]) best = i;
        return best;
    }

    private static MoeRouteScoring ParseScoring(string name) => name switch
    {
        "sqrtsoftplus" => MoeRouteScoring.SqrtSoftplus,
        "sigmoid" => MoeRouteScoring.Sigmoid,
        "softmax" => MoeRouteScoring.Softmax,
        _ => throw new HartsyInferenceException($"Unsupported scoring_func '{name}'."),
    };
}

/// <summary>The window state of every DSpark stage for one sequence.</summary>
public sealed class DeepSeekV41DSparkState(DeepSeekV41AttentionState[] stages)
{
    /// <summary>One window per stage.</summary>
    public DeepSeekV41AttentionState[] Stages { get; } = stages ?? throw new ArgumentNullException(nameof(stages));
}

/// <summary>One drafted block.</summary>
/// <param name="Ids">The input token followed by the drafted tokens, <c>block + 1</c> ids.</param>
/// <param name="Logits">The target-side logits of each draft position, <c>[block, vocab]</c> row-major, after the Markov bias.</param>
/// <param name="Confidence">The confidence of each draft position, <c>[block]</c>.</param>
public sealed record DeepSeekV41DSparkDraft(int[] Ids, float[] Logits, float[] Confidence);
