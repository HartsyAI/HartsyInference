namespace HartsyInference.ModelAssets.Gguf.KeyMappers;

/// <summary>Flux.2 (Klein 4B / Klein 9B / Dev) GGUF mapper. Flux.2 ships with BFL-style naming carrying <c>double_stream_modulation_*</c> top-level shared modulation linears + <c>double_blocks.*</c> + <c>single_blocks.*</c>. <see cref="HartsyInference.ModelAssets.CheckpointConverters.Flux2CheckpointConverter.Convert"/> handles the BFL→canonical remap; we just pass through.</summary>
public sealed class Flux2KeyMapper : IGgufKeyMapper
{
    public string Architecture => "flux2";

    public bool MatchesByKeys(IEnumerable<string> tensorNames)
    {
        bool hasSharedModulation = false;
        bool hasDoubleStreamMlp = false;
        foreach (string name in tensorNames)
        {
            if (name.Contains("double_stream_modulation_img.", StringComparison.Ordinal)) hasSharedModulation = true;
            if (name.Contains("double_stream_modulation_txt.", StringComparison.Ordinal)) hasSharedModulation = true;
            if (name.Contains("single_stream_modulation.", StringComparison.Ordinal)) hasSharedModulation = true;
            if (name.Contains("double_blocks.", StringComparison.Ordinal) && name.Contains(".mlp.linear_in.", StringComparison.Ordinal))
                hasDoubleStreamMlp = true;
            if (hasSharedModulation && hasDoubleStreamMlp) return true;
        }
        return false;
    }

    public string? MapKey(string ggufKey) => ggufKey;
}
