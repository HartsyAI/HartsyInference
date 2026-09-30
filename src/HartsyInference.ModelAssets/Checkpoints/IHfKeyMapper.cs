namespace HartsyInference.ModelAssets.Checkpoints;

/// <summary>Translates one producer's tensor names to the canonical (official DeepSeek) names the model consumes.</summary>
public interface IHfKeyMapper
{
    /// <summary>Components this producer left out of the checkpoint (for example <c>mtp</c>); their absence is not an error.</summary>
    IReadOnlySet<string> StrippedComponents { get; }

    /// <summary>The canonical name for <paramref name="sourceKey"/>, or null when it has no canonical equivalent (a stripped or unknown tensor).</summary>
    string? MapToCanonical(string sourceKey);
}
