namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>Header-and-config estimates of the host memory the V4.1 reference model needs besides its mapped checkpoint bytes, so a load can be refused or planned before any weight is read.</summary>
/// <remarks>These are sizing estimates, not measurements: the activation figure counts the live F32 buffers of one block over a whole prefill pass with a safety factor, and the real peak has not been measured on the official checkpoint.</remarks>
public static class DeepSeekV41WorkingMemory
{
    // buffers of one block alive at once beyond the named terms: attention copies, MoE scratch, allocator slack
    private const double ActivationSafetyFactor = 1.5;

    /// <summary>Host bytes of sequence state (sliding window, compressed latents, partial pooling group, index keys) for one sequence of <paramref name="maxTokens"/> positions across the backbone layers.</summary>
    public static long SequenceStateBytes(DeepSeekV41Config cfg, int maxTokens)
    {
        ArgumentNullException.ThrowIfNull(cfg);
        long bytes = 0;
        for (int layer = 0; layer < cfg.NumHiddenLayers; layer++)
            bytes += DeepSeekV41AttentionState.EstimateBytes(DeepSeekV41AttentionSettings.ForLayer(cfg, layer), maxTokens);
        return bytes;
    }

    /// <summary>F32 working bytes per token of a prefill pass: the hyper-connection streams, the attention projections and outputs, the sparse-attention index row, the grouped output projection and the mixes.</summary>
    public static long ActivationBytesPerToken(DeepSeekV41Config cfg)
    {
        ArgumentNullException.ThrowIfNull(cfg);
        long dim = cfg.HiddenSize, hc = cfg.HcMult, mix = (2 + hc) * hc;
        long stream = 3 * hc * dim;                                              // stream, residual and expanded stream
        long block = 4 * dim + cfg.QLoraRank + mix + 2 * hc * dim;               // normed input, branch output, q latent, mixes, collapsed copies
        long attention = 4L * cfg.NumAttentionHeads * cfg.HeadDim                // q, attention output, rotated copy, tensor copy
            + 3L * cfg.HeadDim + (long)cfg.OGroups * cfg.OLoraRank;              // kv, compressor rows, grouped projection
        long indices = (long)cfg.SlidingWindow + cfg.IndexTopk + cfg.CandidateTopkBlocks; // int32 index row, counted as the same width
        return (long)(sizeof(float) * (stream + block + attention + indices) * ActivationSafetyFactor);
    }

    /// <summary>Working bytes of a prefill pass over <paramref name="tokens"/> positions, plus the one logits row.</summary>
    public static long ActivationBytes(DeepSeekV41Config cfg, int tokens)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(tokens, 1);
        return ActivationBytesPerToken(cfg) * tokens + (long)cfg.VocabSize * sizeof(float);
    }

    /// <summary>Bytes of the small float tensors the loader always widens: gates, hyper-connection projections, norms, sinks and Engram gating vectors.</summary>
    public static long SmallTensorBytes(DeepSeekV41Config cfg)
    {
        ArgumentNullException.ThrowIfNull(cfg);
        long dim = cfg.HiddenSize, hc = cfg.HcMult, mix = (2 + hc) * hc;
        long perLayer = (long)cfg.NRoutedExperts * dim + 2L * cfg.NRoutedExperts                 // gate weight, bias
            + 2 * (mix * hc * dim + 3 + mix)                                                      // two hyper-connection blocks
            + 2 * dim + cfg.QLoraRank + cfg.HeadDim + cfg.NumAttentionHeads;                      // norms, sink
        long engram = (long)cfg.EngramLayerIds.Count * 2 * hc * dim;
        return sizeof(float) * (perLayer * cfg.NumHiddenLayers + engram + dim);
    }

    /// <summary>Bytes the routed-expert cache holds: a view per expert when weights stay stored (negligible), three F32 matrices per expert when widened.</summary>
    public static long ExpertCacheBytes(DeepSeekV41Config cfg, DeepSeekV41LoadOptions options)
    {
        ArgumentNullException.ThrowIfNull(cfg);
        ArgumentNullException.ThrowIfNull(options);
        return options.Residency == DeepSeekV41Residency.Stored ? 0 : options.ExpertCacheCapacity * 3L * cfg.MoeIntermediateSize * cfg.HiddenSize * sizeof(float);
    }

    /// <summary>Anonymous host memory a load with <paramref name="options"/> needs on top of the file-backed checkpoint bytes, whose pages the kernel can drop and re-read.</summary>
    /// <param name="cfg">The parsed config.</param>
    /// <param name="denseStoredBytes">Stored bytes of the dense, embedding and head classes; counted only when <see cref="DeepSeekV41Residency.WidenedF32"/> widens them.</param>
    /// <param name="options">Load options.</param>
    public static long AnonymousBytes(DeepSeekV41Config cfg, long denseStoredBytes, DeepSeekV41LoadOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        // widening takes 4 bytes per stored FP8 byte and 2 per BF16 byte; 4x is the safe upper bound used before and still right for the dominant FP8
        long widened = options.Residency == DeepSeekV41Residency.WidenedF32 ? 4 * denseStoredBytes : 0;
        return widened + SmallTensorBytes(cfg) + ExpertCacheBytes(cfg, options) + EngramCacheBytes(cfg, options) + RowWindowBytes
            + SequenceStateBytes(cfg, options.MaxTokens) + ActivationBytes(cfg, options.MaxTokens);
    }

    /// <summary>Row-window scratch of a few concurrent dequantized products, charged once per load.</summary>
    public static long RowWindowBytes => 3L * (1 << 22) * sizeof(float);

    /// <summary>The Engram row cache a load with <paramref name="options"/> holds: one budget per Engram layer.</summary>
    public static long EngramCacheBytes(DeepSeekV41Config cfg, DeepSeekV41LoadOptions options) => options.EngramBudgetBytes * cfg.EngramLayerIds.Count;
}
