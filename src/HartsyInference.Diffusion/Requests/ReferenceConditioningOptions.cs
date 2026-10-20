namespace HartsyInference.Diffusion.Requests;

/// <summary>How a pipeline conditions on reference images.</summary>
public sealed record ReferenceConditioningOptions
{
    /// <summary>The options a reference-trained LoRA expects unless it says otherwise: cached keys and values.</summary>
    public static ReferenceConditioningOptions Default { get; } = new();

    /// <summary>The attention layout, which must match how the adapter was trained.</summary>
    public ReferenceAttentionMode Mode { get; init; } = ReferenceAttentionMode.CachedKv;

    /// <summary>Share of the device's free memory the reference caches may take before a warning is logged.</summary>
    public float CacheBudgetFraction { get; init; } = 0.5f;
}
