namespace HartsyInference.Engine.Dispatch;

/// <summary>A model token split into its family id and optional variant: <c>id</c> or <c>id:variant</c>
/// (<c>qwen-image:edit</c>, <c>whisper:large-v3</c>). The one parser every modality shares.</summary>
/// <param name="Id">The part before the first colon, trimmed and lowercased.</param>
/// <param name="Variant">The part after the first colon, trimmed; null when the token has no colon.</param>
public readonly record struct ModelSelector(string Id, string? Variant)
{
    /// <summary>Splits <paramref name="token"/>; a null or blank token yields an empty id.</summary>
    public static ModelSelector Parse(string? token)
    {
        string trimmed = (token ?? string.Empty).Trim();
        int separator = trimmed.IndexOf(':', StringComparison.Ordinal);
        return separator < 0
            ? new ModelSelector(trimmed.ToLowerInvariant(), null)
            : new ModelSelector(trimmed[..separator].Trim().ToLowerInvariant(), trimmed[(separator + 1)..].Trim());
    }
}
