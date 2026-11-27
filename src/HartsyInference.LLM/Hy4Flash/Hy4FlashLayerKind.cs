namespace HartsyInference.LLM.Hy4Flash;

/// <summary>Which attention a Hy4 Flash decoder layer runs.</summary>
public enum Hy4FlashLayerKind
{
    /// <summary>32-head gated MLA over the keys of the groups a compressed indexer recalls (<c>deepseek_sparse_attention</c> in the shipped config).</summary>
    Indexed,

    /// <summary>64-head gated MLA over the last <see cref="Hy4FlashConfig.SlidingWindow"/> keys, no indexer.</summary>
    Sliding,
}
