namespace HartsyInference.ModelAssets.Checkpoints;

/// <summary>The MLX 4-bit conversion, which keeps the official tensor names and adds <c>.scales</c> / <c>.biases</c> companions beside each packed <c>.weight</c>.</summary>
/// <remarks>Companions map to themselves; the quant binder pairs them with their weight. Nothing is stripped: the conversion's <c>mtp.2</c> draft is incomplete rather than absent, which the catalog reports.</remarks>
public sealed class MlxV41KeyMapper : IHfKeyMapper
{
    /// <inheritdoc/>
    public IReadOnlySet<string> StrippedComponents { get; } = new HashSet<string>(StringComparer.Ordinal);

    /// <inheritdoc/>
    public string? MapToCanonical(string sourceKey)
    {
        ArgumentNullException.ThrowIfNull(sourceKey);
        return sourceKey;
    }
}
