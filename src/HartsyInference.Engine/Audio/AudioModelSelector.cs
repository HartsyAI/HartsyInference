using HartsyInference.Engine.Dispatch;

namespace HartsyInference.Engine.Audio;

/// <summary>Splits a <see cref="ModelSpec"/> into the catalog descriptor id and the per-model variant hint the descriptors resolve — the Engine-native form of the extension's (provider id, <c>__model_id</c>) pair. The request token is <c>id</c> or <c>id:variant</c> (e.g. <c>whisper:large-v3</c>, <c>acestep:turbo</c>); with no variant the whole token is passed through so a descriptor that accepts a bare repo id still works.</summary>
internal readonly record struct AudioModelSelector(string Id, string Variant, string? LocalPath)
{
    /// <summary>Parses the selector out of a resolved model spec.</summary>
    internal static AudioModelSelector Parse(ModelSpec spec)
    {
        ArgumentNullException.ThrowIfNull(spec);
        string token = (spec.Catalog?.Id ?? spec.Requested ?? string.Empty).Trim();
        ModelSelector selector = ModelSelector.Parse(token);
        return new AudioModelSelector(selector.Id, selector.Variant ?? token, spec.LocalPath);
    }
}
