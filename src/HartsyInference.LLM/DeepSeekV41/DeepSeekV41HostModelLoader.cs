using HartsyInference.Core.Backends;
using HartsyInference.Core.Engram;
using HartsyInference.Core.Exceptions;
using HartsyInference.LLM.DeepSeekV41.Engram;
using HartsyInference.ModelAssets.BlockScale;

namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>Builds a <see cref="DeepSeekV41HostModel"/> from an opened DeepSeek-V4.1 checkpoint: dense weights dequantized to F32, routed experts read lazily, Engram rows read from disk.</summary>
/// <remarks>This is the reference path, so dense weights are held as F32 (about 30 GiB of host memory for the official checkpoint). Draft, vision and DSpark tensors are not read.</remarks>
public static class DeepSeekV41HostModelLoader
{
    /// <summary>Opens <paramref name="directory"/> and loads it; the returned object owns the checkpoint.</summary>
    public static DeepSeekV41LoadedModel Load(IBackend backend, string directory, DeepSeekV41LoadOptions options)
    {
        DeepSeekV41Checkpoint checkpoint = DeepSeekV41Checkpoint.Open(directory);
        try
        {
            return Load(backend, checkpoint, options, ownsCheckpoint: true);
        }
        catch
        {
            checkpoint.Dispose();
            throw;
        }
    }

    /// <summary>Loads an already-open checkpoint; it must outlive the returned model.</summary>
    /// <param name="ownsCheckpoint">Whether disposing the result also disposes <paramref name="checkpoint"/>.</param>
    public static DeepSeekV41LoadedModel Load(IBackend backend, DeepSeekV41Checkpoint checkpoint, DeepSeekV41LoadOptions options, bool ownsCheckpoint = false)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(checkpoint);
        ArgumentNullException.ThrowIfNull(options);
        options.Validate();
        DeepSeekV41Config cfg = checkpoint.Config;
        if (cfg.NSharedExperts != 1) throw new HartsyInferenceException($"Only one shared expert is supported; the config has {cfg.NSharedExperts}.");
        if (cfg.LayerPlans.Count < cfg.NumHiddenLayers) throw new HartsyInferenceException("The config's layer plan does not cover every backbone layer.");

        List<EngramTableStore> stores = [];
        try
        {
            Reader read = new(checkpoint);
            int dim = cfg.HiddenSize, hc = cfg.HcMult, rd = cfg.QkRopeHeadDim;
            float normEps = (float)cfg.RmsNormEps;

            // two rope variants exist: compressed layers use the compress theta with YaRN, the rest the base theta plain
            DeepSeekV41RopeTable plainRope = DeepSeekV41RopeTable.Build(rd, options.MaxTokens, cfg.RopeTheta, null);
            DeepSeekV41RopeTable compressRope = DeepSeekV41RopeTable.Build(rd, options.MaxTokens, cfg.CompressRopeTheta, cfg.RopeScaling);

            // one cache across all layers; a key packs (layer, expert)
            int experts = cfg.NRoutedExperts;
            DeepSeekV41ExpertCache expertCache = new(key => DeepSeekV41ExpertLoader.Load(checkpoint.ExpertBank(key / experts), key % experts, dim, cfg.MoeIntermediateSize),
                options.ExpertCacheCapacity);

            DeepSeekV41Block[] blocks = new DeepSeekV41Block[cfg.NumHiddenLayers];
            for (int layer = 0; layer < blocks.Length; layer++)
                blocks[layer] = BuildBlock(backend, checkpoint, read, cfg, layer, plainRope, compressRope, expertCache, options, stores);

            DeepSeekV41HostModel model = new(dim, hc, cfg.VocabSize, normEps, read.Matrix("embed.weight", cfg.VocabSize, dim), blocks,
                read.Vector("norm.weight", dim), read.Matrix("head.weight", cfg.VocabSize, dim));
            return new DeepSeekV41LoadedModel(model, checkpoint, ownsCheckpoint, stores);
        }
        catch
        {
            foreach (EngramTableStore store in stores) store.Dispose();
            throw;
        }
    }

    private static DeepSeekV41Block BuildBlock(IBackend backend, DeepSeekV41Checkpoint checkpoint, Reader read, DeepSeekV41Config cfg, int layer,
        DeepSeekV41RopeTable plainRope, DeepSeekV41RopeTable compressRope, DeepSeekV41ExpertCache expertCache, DeepSeekV41LoadOptions options,
        List<EngramTableStore> stores)
    {
        DeepSeekV41LayerPlan plan = cfg.LayerPlans[layer];
        int dim = cfg.HiddenSize, hc = cfg.HcMult, heads = cfg.NumAttentionHeads, hd = cfg.HeadDim, qLora = cfg.QLoraRank;
        int ratio = plan.CompressRatio;
        bool kvSource = plan.IsKvSource && ratio > 0, indexSource = plan.IsIndexSource && ratio > 0;
        int candidateLayer = cfg.CandidateSourceLayerId;
        float normEps = (float)cfg.RmsNormEps;
        string l = $"layers.{layer}.", a = l + "attn.", f = l + "ffn.";

        DeepSeekV41AttentionSettings settings = new(dim, heads, hd, cfg.QkRopeHeadDim, qLora, cfg.OGroups, cfg.OLoraRank, cfg.SlidingWindow, ratio, kvSource,
            indexSource, layer == candidateLayer, candidateLayer >= 0 && candidateLayer < layer, cfg.IndexNHeads, cfg.IndexHeadDim, cfg.IndexTopk,
            cfg.CandidateTopkBlocks, cfg.CandidateBlockSize, normEps);

        DeepSeekV41CompressorWeights? compressor = kvSource
            ? new(read.Matrix(a + "compressor.wkv.weight", hd, dim), ratio > 1 ? read.Matrix(a + "compressor.wgate.weight", hd, dim) : null,
                read.Vector(a + "compressor.norm.weight", hd))
            : null;
        DeepSeekV41IndexerWeights? indexer = indexSource
            ? new(read.Matrix(a + "indexer.wq_b.weight", cfg.IndexNHeads * cfg.IndexHeadDim, qLora), read.Matrix(a + "indexer.weights_proj.weight", cfg.IndexNHeads, dim),
                kvSource ? read.Matrix(a + "indexer.wk.weight", cfg.IndexHeadDim, hd) : null, kvSource ? read.Vector(a + "indexer.k_norm.weight", cfg.IndexHeadDim) : null)
            : null;
        DeepSeekV41AttentionWeights weights = new(read.Matrix(a + "wq_a.weight", qLora, dim), read.Vector(a + "q_norm.weight", qLora),
            read.Matrix(a + "wq_b.weight", heads * hd, qLora), read.Matrix(a + "wkv.weight", hd, dim), read.Vector(a + "kv_norm.weight", hd),
            read.Matrix(a + "wo_a.weight", cfg.OGroups * cfg.OLoraRank, heads * hd / cfg.OGroups), read.Matrix(a + "wo_b.weight", dim, cfg.OGroups * cfg.OLoraRank),
            read.Vector(a + "attn_sink", heads), compressor, indexer);
        DeepSeekV41Attention attention = new(backend, settings, weights, ratio > 0 ? compressRope : plainRope);

        int experts = cfg.NRoutedExperts, inter = cfg.MoeIntermediateSize;
        DeepSeekV41SwigluWeights shared = new(dim, inter, read.Matrix(f + "shared_experts.w1.weight", inter, dim), read.Matrix(f + "shared_experts.w2.weight", dim, inter),
            read.Matrix(f + "shared_experts.w3.weight", inter, dim));
        MoeRouteArgs route = new(experts, cfg.NumExpertsPerTok, ParseScoring(cfg.ScoringFunc), Renormalize: cfg.NormTopkProb && cfg.NumExpertsPerTok > 1,
            RenormEpsilon: 1e-20f, Scale: (float)cfg.RoutedScalingFactor);
        DeepSeekV41MoeLayer ffn = new(backend, read.Matrix(f + "gate.weight", experts, dim), read.Vector(f + "gate.bias", experts), null, route,
            new LayerExperts(expertCache, layer, experts), shared, (float)cfg.SwigluLimit);

        int mix = (2 + hc) * hc;
        DeepSeekV41HyperConnection hcAttn = new(backend, hc, dim, cfg.HcSinkhornIters, (float)cfg.HcEps, normEps, read.Matrix(l + "hc_attn_fn", mix, hc * dim),
            read.Vector(l + "hc_attn_scale", 3), read.Vector(l + "hc_attn_base", mix));
        DeepSeekV41HyperConnection hcFfn = new(backend, hc, dim, cfg.HcSinkhornIters, (float)cfg.HcEps, normEps, read.Matrix(l + "hc_ffn_fn", mix, hc * dim),
            read.Vector(l + "hc_ffn_scale", 3), read.Vector(l + "hc_ffn_base", mix));

        DeepSeekV41EngramModule? engram = null;
        if (plan.EngramSlot is { } slot)
        {
            int columns = (cfg.EngramMaxNgramSize - 1) * cfg.EngramNHeads, headDim = cfg.EngramHeadDim;
            if (columns != EngramConstants.ColumnsPerLayer)
                throw new HartsyInferenceException($"Engram needs {EngramConstants.ColumnsPerLayer} hash columns per layer; the config gives {columns}.");
            EngramTableStore store = EngramTableStores.Open(checkpoint.EngramTable(layer), EngramBacking.Storage, options.EngramBudgetBytes);
            stores.Add(store);
            engram = new DeepSeekV41EngramModule(dim, hc, columns, headDim, normEps, read.Matrix(l + "engram.wkv.weight", dim * (hc + 1), columns * headDim),
                read.Matrix(l + "engram.q_weight", hc, dim), read.Matrix(l + "engram.k_weight", hc, dim), store.Gather);
            return new DeepSeekV41Block(dim, hc, normEps, hcAttn, hcFfn, read.Vector(l + "attn_norm.weight", dim), read.Vector(l + "ffn_norm.weight", dim), attention, ffn,
                engram, slot);
        }
        return new DeepSeekV41Block(dim, hc, normEps, hcAttn, hcFfn, read.Vector(l + "attn_norm.weight", dim), read.Vector(l + "ffn_norm.weight", dim), attention, ffn);
    }

    private static MoeRouteScoring ParseScoring(string name) => name switch
    {
        "sqrtsoftplus" => MoeRouteScoring.SqrtSoftplus,
        "sigmoid" => MoeRouteScoring.Sigmoid,
        "softmax" => MoeRouteScoring.Softmax,
        _ => throw new HartsyInferenceException($"Unsupported scoring_func '{name}'."),
    };

    // Reads canonical keys as F32, checking the logical shape named by the config.
    private sealed class Reader(DeepSeekV41Checkpoint checkpoint)
    {
        public float[] Matrix(string key, long rows, long cols) =>
            DeepSeekV41ExpertLoader.ReadMatrix(key, checkpoint.GetWeight(key), checkpoint.GetQuant(key), rows, cols);

        public float[] Vector(string key, long length)
        {
            float[] values = WeightDequantizer.ToF32(checkpoint.GetWeight(key), checkpoint.GetQuant(key));
            if (values.Length != length) throw new HartsyInferenceException($"{key} holds {values.Length} values, expected {length}.");
            return values;
        }
    }

    // One layer's view of the shared expert cache.
    private sealed class LayerExperts(DeepSeekV41ExpertCache cache, int layer, int experts) : IDeepSeekV41ExpertSource
    {
        public DeepSeekV41SwigluWeights GetExpert(int expert) => cache.GetExpert(layer * experts + expert);
    }
}
