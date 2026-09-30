namespace HartsyInference.ModelAssets.Checkpoints;

/// <summary>The EXL3 derivative, which keeps the official names except that the output head is <c>lm_head.*</c> rather than <c>head.*</c>.</summary>
/// <remarks>Read from the real index of sfxnz/DeepSeek-V4.1-Flash-EXL3 @ 982b7045: <c>lm_head.weight</c> and <c>lm_head.weight_scale</c>.</remarks>
public sealed class Exl3V41KeyMapper : IHfKeyMapper
{
    private const string SourceHead = "lm_head.";
    private const string CanonicalHead = "head.";

    /// <inheritdoc/>
    public IReadOnlySet<string> StrippedComponents { get; } = new HashSet<string>(StringComparer.Ordinal);

    /// <inheritdoc/>
    public string? MapToCanonical(string sourceKey)
    {
        ArgumentNullException.ThrowIfNull(sourceKey);
        return sourceKey.StartsWith(SourceHead, StringComparison.Ordinal)
            ? string.Concat(CanonicalHead, sourceKey.AsSpan(SourceHead.Length))
            : sourceKey;
    }
}
