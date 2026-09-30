namespace HartsyInference.ModelAssets.Checkpoints;

/// <summary>DeepSeek's own release, whose names are the canonical ones, so every key maps to itself.</summary>
/// <remarks>The NVIDIA and Quark derivatives keep these names too (EXL3 renames only the head, see <see cref="Exl3V41KeyMapper"/>); they differ only in companion tensors, which the quant binder owns.</remarks>
public sealed class OfficialV41KeyMapper : IHfKeyMapper
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
